from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BitsAndBytesConfig, CLIPVisionModel

# from utils.utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN

from .univi.model.language_model.llama import ChatUniViLlamaForCausalLM, ChatUniViLlamaModel 

from sam2.build_sam import _build_sam2_
from model.univi.constants import IMAGE_TOKEN_INDEX

import time
import numpy as np


def dice_loss(
    inputs   : torch.Tensor,
    targets  : torch.Tensor,
    num_masks: float,
    scale    : float =1000,
    eps      : float =1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary classification label for each element in inputs (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary classification label for each element in inputs (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss


class VrshqMetaModel:
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(VrshqMetaModel, self).__init__(config)

        self.config = config
        if not hasattr(self.config, "train_mask_decoder"):
            self.config.train_mask_decoder = kwargs["train_mask_decoder"]
            self.config.out_dim = kwargs["out_dim"]
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
        else:
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
            self.initialize_lisa_modules(self.config)

    def initialize_lisa_modules(self, config):
        # SAM
        self.visual_model = _build_sam2_(ckpt_path=self.vision_pretrained, video_inference=False)
        
        for param in self.visual_model.parameters():  # frozen the image encoder weights
            param.requires_grad = False
        if config.train_mask_decoder:            # like the visa-1 version, we only fine tune the mask decoder
            self.visual_model.sam_mask_decoder.train()       
            for param in self.visual_model.sam_mask_decoder.parameters():
                param.requires_grad = True

        # Projection layer
        in_dim = config.hidden_size
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])      # 用啦映射seg token
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True


class VrshqModel(VrshqMetaModel, ChatUniViLlamaModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(VrshqModel, self).__init__(config, **kwargs)

        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False


class VrshqForCausalLM(ChatUniViLlamaForCausalLM):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        if not hasattr(config, "train_mask_decoder"):
            config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
            config.mm_vision_tower = kwargs.get("vision_tower", "openai/clip-vit-large-patch14")
            self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
            self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
            self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
            self.target_loss_weight = kwargs.pop("target_loss_weight", None)
            # config._attn_implementation = "flash_attention_2"
        else:
            config.mm_vision_tower = config.vision_tower
            
        self.seg_token_idx = kwargs.pop("seg_token_idx")
        self.track_token_idx = kwargs.pop("track_token_idx")
        self.num_seg_token = kwargs.pop("seg_token_num")
        self.alpha = kwargs.pop("alpha")

        super().__init__(config)

        self.model = VrshqModel(config, **kwargs)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            image_embeddings = self.model.visual_model.image_encoder(pixel_values)
        return image_embeddings

    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)
    
    def sample_cond_frame(self, target, sample_list, num_frame):
        index = sample_list.index(target)
        # print(index, sample_list)
        if index == 0:
            frame_ids_sam = sample_list[0: index + num_frame]
        elif index == len(sample_list) -1:
            frame_ids_sam = sample_list[index - num_frame + 1: ]
        else:
            frame_ids_sam = sample_list[index - num_frame // 2 : index + num_frame // 2 + 1]
           
        return frame_ids_sam, frame_ids_sam.index(index)

    def model_forward(
        self,
        images: torch.FloatTensor,    # list: num_conv, t, 3, h, w
        images_clip: torch.FloatTensor,      # list: num_conv, t, 3, h, w
        input_ids: torch.LongTensor,     # num_sentence in a batch, num tokens in a sentence
        labels: torch.LongTensor,      # num_sentence in a batch, num tokens in a sentence, 问题部分全部屏蔽
        attention_masks: torch.LongTensor,
        offset: torch.LongTensor,        # 由于每个batch的conversation数量不一致，因此需要设置offset来区分不同的conversation分别属于哪个batch
        masks_list: List[torch.FloatTensor],
        label_list: List[torch.Tensor],
        resize_list: List[tuple],
        conversation_list: List[str], 
        num_frame_list: List[int],
        num_conv_list: List[int],
        inference: bool = False,
        **kwargs,
    ):

        batch_size = len(images)
        # image_embeddings = self.get_visual_embs(torch.cat(images, dim=0))    # vision encoder: bs, dim, h, w
        assert batch_size == len(offset) - 1
        for batch_idx in range(batch_size):         # 对于每一个batch size，确定同一batch中的每个视频对应几个conversation
            assert num_conv_list[batch_idx] == offset[batch_idx + 1] - offset[batch_idx]

        if inference:
            length = input_ids.shape[0]
            assert len(images_clip) == 1, f'Inference only supports one video, but got {len(images_clip)} videos.'
            images_clip = [        # num_conv = 1 * t, dim, h, w all the frames
                images_clip[0].unsqueeze(0).expand(length, -1, -1, -1, -1).contiguous().flatten(0, 1)
            ]

            output_i = super().forward(
                images=images_clip,
                attention_mask=attention_masks,
                input_ids=input_ids,
                output_hidden_states=True,
            )
            torch.cuda.empty_cache()

            output_hidden_states = output_i.hidden_states
            output = None

            num_image_ori_token = (input_ids[0] == IMAGE_TOKEN_INDEX).sum()
            assert all(
                [
                    (input_ids[i] == IMAGE_TOKEN_INDEX).sum() == num_image_ori_token for i in range(length)
                ]
            )
            token_add = 111 * num_image_ori_token

            seg_token_mask = input_ids[:, 1:] == self.seg_token_idx        # 将每个conversation token中的 <seg> token转换为True
            seg_token_mask = torch.cat([seg_token_mask,  torch.zeros((seg_token_mask.shape[0], 1)).bool().cuda(), ], dim=1, )          # num_sentences, ntokens + 1
            seg_token_mask = torch.cat([torch.zeros((seg_token_mask.shape[0], token_add)).bool().cuda(), seg_token_mask], dim=1, )
            all_conv_seg_token_num = seg_token_mask.sum(dim=1).tolist()
            
            target_token_mask = input_ids[:, 1:] == self.target_token_idx        # 将每个conversation token中的 <target> token转换为True
            target_token_mask = torch.cat([target_token_mask, torch.zeros((target_token_mask.shape[0], 1)).bool().cuda()], dim=1, )    # num_sentences, ntokens
            target_token_mask = torch.cat([torch.zeros((target_token_mask.shape[0], token_add)).bool().cuda(), target_token_mask], dim=1)

        else:
            images_clip_list = []
            for batch_idx in range(batch_size):
                bs_conv_num = num_conv_list[batch_idx]     # 该视频中的conversation数量
                images_clip_i = images_clip[batch_idx].unsqueeze(0).expand(bs_conv_num, -1, -1, -1, -1).contiguous()      # num_conv, t, 3, h, w
                images_clip_list.append(images_clip_i)
            images_clip_list = [i.flatten(0, 1) for i in images_clip_list]     # list: num_conv * t, 3, h, w

            output = super().forward(
                images=images_clip_list,
                attention_mask=attention_masks,       # num_conv, tokens
                input_ids=input_ids,
                labels=labels,
                output_hidden_states=True,
            )
            output_hidden_states = output.hidden_states        # L层, num_sentences, num_tokens, in_dim

            seg_token_mask = output.labels[..., 1:] == self.seg_token_idx        # 将每个conversation token中的 <seg> token转换为True
            seg_token_mask = torch.cat([seg_token_mask,  torch.zeros((seg_token_mask.shape[0], 1), device=output.labels.device).bool(), ], dim=1)    # num_sentences, ntokens + 1
            all_conv_seg_token_num = seg_token_mask.sum(dim=1).tolist()
            
            track_token_mask = output.labels[..., 1:] == self.track_token_idx         # 每个conversation中的track token转换为True
            track_token_mask = torch.cat([track_token_mask,  torch.zeros((track_token_mask.shape[0], 1), device=output.labels.device).bool(), ], dim=1)    # num_sentences, 1
            
        assert len(self.model.text_hidden_fcs) == 1
        
        pred_embeddings = self.model.text_hidden_fcs[0](output_hidden_states[-1][seg_token_mask])       # num_sentences in the batch, out_dim
        seg_token_counts = seg_token_mask.int().sum(-1)       # num_sentences in the batch   有几个seg_token

        pred_track_embeddings = self.model.text_hidden_fcs[0](output_hidden_states[-1][track_token_mask])        # num_conv in the batch, out_dim
        track_token_counts = track_token_mask.int().sum(-1)       # num_conv in the batch   有几个track_token
        track_token_offset = track_token_counts.cumsum(-1)       # 这里需要注意：因为有的sentence中不含有track_token，因此需要通过一定方式将所有的track_token提取出来
        track_token_offset = torch.cat(
            [torch.zeros(1).long().cuda(), track_token_offset], dim=0
        )
        track_token_offset = track_token_offset[offset]

        seg_token_offset = seg_token_counts.cumsum(-1)       # 这里需要注意：因为有的sentence中不含有seg_token，因此需要通过一定方式将所有的seg_token提取出来
        seg_token_offset = torch.cat(
            [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
        )

        seg_token_offset = seg_token_offset[offset]

        pred_embeddings_ = []
        pred_track_embeddings_ = []
        for i in range(len(track_token_offset) - 1):
            start_i, end_i = track_token_offset[i], track_token_offset[i + 1]      # 得到属于各个视频的每个conversation对应的seg_token索引
            num_frame = len(images[i])
            start_frame_i = 0 * num_frame
            end_frame_i = (end_i - start_i) * num_frame    
            batch_pred_embeddings = pred_embeddings[start_frame_i: end_frame_i]
            batch_pred_embeddings = batch_pred_embeddings.reshape(len(batch_pred_embeddings) // num_frame, num_frame, batch_pred_embeddings.shape[-1])
            pred_embeddings_.append(batch_pred_embeddings)
            pred_embeddings = pred_embeddings[end_frame_i: ]
            
            pred_track_embeddings_.append(pred_track_embeddings[start_i: end_i])
            
        pred_track_embeddings = pred_track_embeddings_     # list: num_conversation, dim
        pred_embeddings = pred_embeddings_        # list: num_conversations, t, dim

        assert len(pred_embeddings) == batch_size
        assert len(pred_track_embeddings) == batch_size
        
        # mask decoder
        pred_masks = []
        gt_masks = []
        alpha = self.alpha
        for i in range(batch_size):     # 训练阶段的batch size数量较大
            
            if len(images[i]) == 1:       # 图像数据，因为视频的帧数一定超过1
                
                # 进行seg token以及track token的融合
                global_embedding = []
                track_embeddings = pred_track_embeddings[i].unsqueeze(1)       # num_conv, 1, dim
                seg_embeddings = pred_embeddings[i]      # num_conv, t, dim
                if track_embeddings.shape[0] != 0:
                    for num_conv, seg_token in enumerate(track_embeddings):
                        track_embed = track_embeddings[num_conv]       # 1, dim
                        seg_embed = seg_embeddings[num_conv]       # t, dim
                        similarity = seg_embed @ track_embed.transpose(0, 1)       # t, 1
                        similarity = torch.sigmoid(similarity)     # t, 1
                        
                        # 对于图像和视频，都将各帧seg token按相似度和global seg token进行加权融合
                        track_embed = track_embed # + alpha * similarity.transpose(0, 1) @ seg_embed      # 1, dim
                        global_embedding.append(track_embed)
                    global_embedding = torch.cat(global_embedding, dim=0)      # num_conv, dim
                else:
                    global_embedding = track_embeddings.squeeze(1)
                
                # vision backbone
                backbone_out = self.model.visual_model.forward_image(images[i])      # 1, 3, h, w
                _, vision_feats, _, _ = self.model.visual_model._prepare_backbone_features(backbone_out)      # list: hw, 1, c
                
                if self.model.visual_model.directly_add_no_mem_embed:    # add no memory embeddings to the first stage
                        vision_feats[-1] = vision_feats[-1] + self.model.visual_model.no_mem_embed

                feats = [
                    feat.permute(1, 2, 0).view(1, -1, *feat_size)     # 1, c, h, w
                    for feat, feat_size in zip(vision_feats[::-1], self.model.visual_model._bb_feat_sizes[::-1])
                ][::-1]
                self._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}      # feats[-1]: 最高级特征且分辨率最低，feats[:-1]: 两层低级特征，分辨率较高
                
                sparse_embeddings, dense_embeddings = self.model.visual_model.sam_prompt_encoder(
                                                            points=None,
                                                            boxes=None,
                                                            masks=None,
                                                            batch_size=global_embedding.shape[0],    # n
                                                        )
                sparse_embeddings = torch.cat((sparse_embeddings, global_embedding.unsqueeze(1)), dim=1).to(global_embedding.dtype)     # n, 1, dim
                
                if sparse_embeddings.shape[0] == 0:
                    pass
                
                high_res_features = [feat_level[0].unsqueeze(0).to(global_embedding.dtype) for feat_level in self._features["high_res_feats"]]
                
                multimask_output = False      # 这里是否仅输出一个mask
                low_res_masks, _, _, _ = self.model.visual_model.sam_mask_decoder(
                    image_embeddings=self._features["image_embed"][0].unsqueeze(0).to(global_embedding.dtype),
                    image_pe=self.model.visual_model.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                    repeat_image=True,
                    high_res_features=high_res_features,
                )
                
                video_masks = self.model.visual_model._transforms.postprocess_masks(      
                    masks=low_res_masks, orig_hw=(1024, 1024)
                )
                video_masks = video_masks[..., : resize_list[i][0], : resize_list[i][1]]
                video_masks = F.interpolate(
                        video_masks, label_list[i].shape, mode="bilinear", align_corners=False
                    )      # 1, 1, h, w
                
                pred_masks.append(video_masks[:, 0])    # num_conv, h, w
                gt_masks.append(masks_list[i])       # num_conv, t, h, w
                
            else:     # 视频数据，帧数和送入MLLM中的一致
                # 根据真实的关键帧数据以及pred_target_logits的分数来选择关键帧送入SAM中
                
                video_mask_list_ = []
                gt_masks_list = []
                mask = masks_list[i]     # num_conv, t, h, w
                
                # 进行全局和局部的seg token融合
                track_embeddings = pred_track_embeddings[i].unsqueeze(1)       # num_conv, 1, dim
                seg_embeddings = pred_embeddings[i]      # num_conv, t, dim
                sam_images_ = images[i]      # t, 3, h, w
                
                for num_conv, seg_token in enumerate(track_embeddings):
                    track_embed = track_embeddings[num_conv]       # 1, dim
                    seg_embed = seg_embeddings[num_conv]       # t, dim
                    # similarity = seg_embed @ track_embed.transpose(0, 1)       # t, 1

                    similarity = F.cosine_similarity(seg_embed, track_embed, dim=1).unsqueeze(1)
                    similarity = torch.softmax(similarity, dim=0)     # t, 1
                    # 对于图像和视频，都将各帧seg token按相似度和global seg token进行加权融合
                    track_embed = track_embed + alpha * similarity.transpose(0, 1) @ seg_embed      # 1, dim
                    
                    seg_index = torch.argmax(similarity[:, 0], dim=0)       # t 筛选出相似度最高的帧级seg token
                    frame_ids_sam, cond_frame_idx = self.sample_cond_frame(seg_index, sample_list=list(range(len(sam_images_))), num_frame=3)
                    sam_images = torch.stack([sam_images_[id] for id in frame_ids_sam], dim=0)      # 3, 3, h, w
                    gt_masks_list.append(torch.stack([mask[num_conv][id] for id in frame_ids_sam]))     # 3, h, w
                
                    # initialize the sam2 for training
                    inference_state = self.model.visual_model.train_init_state(sam_images)        # 3, 3, h, w
                    self.model.visual_model.reset_training_state(inference_state)
                
                    # 训练阶段仅提供一个条件帧，这里对于不同目标输入的视频不同
                    _, out_obj_ids, out_mask_logits, _, _ = self.model.visual_model.train_add_new_points(
                            inference_state=inference_state,
                            frame_idx=cond_frame_idx,
                            obj_id=(num_conv + 1),
                            pred_embeddings=track_embed,        # 1, dim
                    )
                    
                    if cond_frame_idx == 0:
                        video_mask_list = []
                        for out_frame_idx, out_obj_ids, out_mask_logits, _, _ in self.model.visual_model.train_propagate_in_video(inference_state, 
                                                                                                                                start_frame_idx=0, 
                                                                                                                                reverse=False):
                            masks = out_mask_logits[..., : resize_list[i][0], : resize_list[i][1]]
                            masks = F.interpolate(
                                masks, label_list[i].shape, mode="bilinear", align_corners=False
                            )      # 1, 1, h, w
                            video_mask_list.append(masks[0, 0])     # h, w
                            
                        video_masks = torch.stack(video_mask_list, dim=0)     # t, h, w
                    
                    elif cond_frame_idx == len(sam_images) - 1:
                        video_mask_list = []
                        for out_frame_idx, out_obj_ids, out_mask_logits, _, _ in self.model.visual_model.train_propagate_in_video(inference_state, 
                                                                                                                                start_frame_idx=cond_frame_idx, 
                                                                                                                                reverse=True):      # bs=1, 1, h, w
                            masks = out_mask_logits[..., : resize_list[i][0], : resize_list[i][1]]
                            masks = F.interpolate(
                                masks, label_list[i].shape, mode="bilinear", align_corners=False
                            )      # 1, 1, h, w
                            video_mask_list.append(masks[0, 0])     # h, w
                            
                        video_masks = torch.stack(video_mask_list, dim=0)     # t, h, w
                        video_masks = torch.flip(video_masks, dims=[0])   # t, h, w
                        # video_masks = video_masks.transpose(0, 1)     # 3, t, h, w
                    
                    else:
                        video_mask_list1 = []
                        video_mask_list2 = []
                        for reverse in [False, True]:
                            for out_frame_idx, out_obj_ids, out_mask_logits, _, _ in self.model.visual_model.train_propagate_in_video(inference_state, 
                                                                                                                                    start_frame_idx=cond_frame_idx, 
                                                                                                                                    reverse=reverse):
                                masks = out_mask_logits[..., : resize_list[i][0], : resize_list[i][1]]
                                masks = F.interpolate(
                                    masks, label_list[i].shape, mode="bilinear", align_corners=False
                                )      # 1, 1, h, w
                                
                                if reverse == False:
                                    video_mask_list1.append(masks[0, 0])
                                else:
                                    video_mask_list2.append(masks[0, 0])
                                
                            # 再预测一遍条件帧信息
                            self.model.visual_model.reset_training_state(inference_state)
                            _, out_obj_ids, out_mask_logits, _, _ = self.model.visual_model.train_add_new_points(
                                    inference_state=inference_state,
                                    frame_idx=cond_frame_idx,
                                    obj_id=(num_conv + 1),
                                    pred_embeddings=track_embed,        # 1, dim
                            )
                                        
                        video_pred_masks1 = torch.stack(video_mask_list1, dim=0)     # t, h, w
                        video_pred_masks2 = torch.stack(video_mask_list2, dim=0)     # t, h, w
                        video_pred_masks2 = torch.flip(video_pred_masks2, dims=[0])     # t, h, w
                        video_masks = torch.cat((video_pred_masks2[: -1], video_pred_masks1), dim=0)     # T, h, w
                        
                    video_mask_list_.append(video_masks)
                
                video_masks_ = torch.stack(video_mask_list_, dim=0)      # num_conv, 3, h, w
                pred_masks.append(video_masks_.flatten(0, 1))     # num_conv * 3, h, w
                
                gt_masks.append(torch.stack(gt_masks_list, dim=0))    # num_conv, 3, h, w
                
        model_output = output
        
        gt_masks = [mm.flatten(0, 1) for mm in gt_masks]
            
        if inference:
            return {
                "pred_masks": video_masks,
                "gt_masks": gt_masks,
            }

        output = model_output.logits

        ce_loss = model_output.loss
        ce_loss = ce_loss * self.ce_loss_weight
        mask_bce_loss = 0
        mask_dice_loss = 0
        num_masks = 0
        for batch_idx in range(batch_size):       # 损失计算
            gt_mask = gt_masks[batch_idx]
            pred_mask = pred_masks[batch_idx]

            assert (
                gt_mask.shape[0] == pred_mask.shape[0]
            ), "gt_mask.shape: {}, pred_mask.shape: {}".format(
                gt_mask.shape, pred_mask.shape
            )
            mask_bce_loss += (
                sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            mask_dice_loss += (
                dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            
            num_masks += gt_mask.shape[0]

        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss

        loss = ce_loss + mask_loss
        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss": mask_loss,
        }

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError("This method is not implemented.")
