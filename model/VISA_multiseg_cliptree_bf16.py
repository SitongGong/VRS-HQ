from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BitsAndBytesConfig, CLIPVisionModel

from .univi.model.language_model.llama import ChatUniViLlamaForCausalLM, ChatUniViLlamaModel
from transformers import CLIPModel

from sam2.build_sam import _build_sam2_
from model.univi.constants import IMAGE_TOKEN_INDEX
from model.llava.mm_utils import tokenizer_image_token

import time
import numpy as np
import math


DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_IMAGE_TOKEN = "<image>"


def convert2imagesplit(sent: str, video_len: int) -> str:
    assert DEFAULT_VIDEO_TOKEN in sent, "only support video token"
    assert sent.count(DEFAULT_VIDEO_TOKEN) == 1, "only support one video token"
    replace_sent = ", ".join(f'({i}){DEFAULT_IMAGE_TOKEN}' for i in range(video_len))
    return sent.replace(DEFAULT_VIDEO_TOKEN, replace_sent)

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
            config._attn_implementation = "flash_attention_2"
        else:
            config.mm_vision_tower = config.vision_tower
            
        self.seg_token_idx = kwargs.pop("seg_token_idx")
        self.track_token_idx = kwargs.pop("track_token_idx")
        self.num_seg_token = kwargs.pop("seg_token_num")
        self.alpha = kwargs.pop("alpha")

        self.max_image_token = 12

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
    
    def initialize_clip_modules(self, clip_vision_tower='/openai/clip-vit-large-patch14-336'):
        # 这里使用clip来判断各帧和expression的特征相似度从而直接筛选出关键帧
        self.clip_model = CLIPModel.from_pretrained(clip_vision_tower)
        self.clip_model = self.clip_model.to(self.device, self.dtype)

    def sample_frames(self, index_frame, images_clip):
        if self.max_image_token != 1:  # 最多可以输入多少帧
            to_devide = (self.max_image_token - 1)
            step_size = math.ceil(len(images_clip) / to_devide)
            idx_start = index_frame % step_size
            idx_select = list(range(idx_start, len(images_clip), step_size))  # 以step size为间隔，等距离抽帧
        else:
            idx_select = [index_frame, ]
        assert index_frame in idx_select

        select_images = [images_clip[idx] for idx in idx_select]
        # select_images = select_images + [images_clip[index_frame]]

        return torch.stack(select_images, dim=0), idx_select

    # 在推理阶段尝试几种不同的帧采样方式
    def sample_frames_segment(self, prob_per_frame, images_clip):
        # 先取出概率最大的帧的索引
        index_frame = torch.argmax(prob_per_frame[:, 0], dim=0)

        prob_per_frame = prob_per_frame[:, 0].tolist()  # t
        if self.max_image_token >= len(images_clip):
            select_images = [images_clip[i] for i in range(len(images_clip))] # + [images_clip[index_frame]]
            frame_ids = list(range(len(images_clip)))
        else:
            if self.max_image_token != 1:
                split_point = np.linspace(0, len(images_clip), num=self.max_image_token + 1, dtype=int)
                frame_ids = [prob_per_frame.index(sorted(prob_per_frame[split_point[i]: split_point[i + 1]])[-1]) for i
                             in range(self.max_image_token)]
                select_images = [images_clip[i] for i in frame_ids] # + [images_clip[index_frame]]
            else:
                select_images = [images_clip[index_frame]] # + [images_clip[index_frame]]
                frame_ids = [index_frame, ]

        return torch.stack(select_images, dim=0), frame_ids

    # 分别取出clip相似度分数最高的几帧，并从其它帧中进行均匀采样
    def sample_frames_local_global(self, prob_per_frame, images_clip):
        # 先取出概率最大的帧的索引
        index_frame = torch.argmax(prob_per_frame[:, 0], dim=0)

        prob_per_frame = prob_per_frame[:, 0].tolist()  # t
        if self.max_image_token >= len(images_clip):
            select_images = [images_clip[i] for i in range(len(images_clip))] # + [images_clip[index_frame]]
            frame_ids = list(range(len(images_clip)))
        else:
            if self.max_image_token != 1:
                num_max_tokens = self.max_image_token // 2
                sorted_prob = sorted(prob_per_frame, reverse=True)  # 降序排列
                select_index1 = [prob_per_frame.index(sorted_prob[i]) for i in range(num_max_tokens)]
                select_index2 = [item for item in list(range(len(images_clip))) if item not in select_index1]
                # 对于剩余视频帧进行均匀采样
                split_point = np.linspace(0, len(select_index2), num=num_max_tokens + 1, dtype=int)
                select_index2 = [select_index2[np.random.randint(split_point[i], split_point[i + 1])] for i in
                                 range(num_max_tokens)]
                select_index = select_index1 + select_index2
                select_index = sorted(select_index)
                assert len(select_index) == self.max_image_token
                select_images = [images_clip[i] for i in select_index] # + [images_clip[index_frame]]
                frame_ids = select_index
            else:
                select_images = [images_clip[index_frame]] # + [images_clip[index_frame]]
                frame_ids = [index_frame, ]

        return torch.stack(select_images, dim=0), frame_ids


    def uniform_sampling(self, images_clip):
        num_length = len(images_clip)
        if num_length > self.max_image_token:
            split_point = np.linspace(0, num_length, num=self.max_image_token + 1, dtype=int)      # 从0开始均匀采样num_frames_per_sample + 1帧
            frame_ids = [np.random.randint(split_point[i], split_point[i + 1]) for i in range(self.max_image_token)]      # 每两个数字之间随机采样一帧，从而得到均匀分布的采样帧
        else:
            frame_ids = list(range(len(images_clip)))
        
        select_images = [images_clip[idx] for idx in frame_ids]
        
        return torch.stack(select_images, dim=0), frame_ids


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
        cond_frame_list: List[int],
        clip_input_list: List[int],
        tokenizer=None, 
        # frame_ids_list: List[int], 
        inference: bool = False,
        **kwargs,
    ):

        batch_size = len(images)
        # image_embeddings = self.get_visual_embs(torch.cat(images, dim=0))    # vision encoder: bs, dim, h, w
        assert batch_size == len(offset) - 1
        for batch_idx in range(batch_size):         # 对于每一个batch size，确定同一batch中的每个视频对应几个conversation
            assert num_conv_list[batch_idx] == offset[batch_idx + 1] - offset[batch_idx]

        if inference:
            inputs = clip_input_list[0]
            start_time = time.time()
            inputs = {key: value.to(input_ids.device) if key != "pixel_values" else value.to(input_ids.device, self.dtype) for key, value in inputs.items()}
            outputs = self.clip_model(**inputs)
            prob_per_frame = outputs.logits_per_image.softmax(dim=0)
            index_frame = torch.argmax(prob_per_frame[:, 0], dim=0)
            cond_frame_list[0][0] = index_frame
            
            # 尝试使用不同的采样方式（借助clip选择出与文本相关性高的帧）
            images_clip_, frame_ids = self.sample_frames(index_frame, images_clip[0])
            
            # images_clip_, frame_ids = self.sample_frames_segment(prob_per_frame, images_clip[0])
            images_clip = [images_clip_]
            
            frame_ids_list = [frame_ids]
            
            num_frame = len(images_clip[0])
            
            for i in range(len(conversation_list)):
                if DEFAULT_VIDEO_TOKEN in conversation_list[i]:   # 若存在 <video> token 在 conversation 中
                    if conversation_list[i].count(DEFAULT_VIDEO_TOKEN) == 1:      # 将<video>token替换为多个<image>token
                        conversation_list[i] = convert2imagesplit(conversation_list[i], num_frame)
                    else:
                        raise ValueError("num video token > 1: ", conversation_list[i].count(DEFAULT_VIDEO_TOKEN))

            # 将各帧均赋予seg token
            seg_replace = ", ".join(f'({i}) [SEG]' for i in range(num_frame))
            for i in range(len(conversation_list)):
                conversation_list[i] = conversation_list[i].replace("[SEG]", seg_replace)

            input_ids = [
                tokenizer_image_token(prompt, tokenizer, return_tensors="pt").to(input_ids)
                for prompt in conversation_list
            ]
            input_ids = torch.nn.utils.rnn.pad_sequence(          # num_conversation, num_tokens
                input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
            )
            attention_masks = input_ids.ne(tokenizer.pad_token_id).to(input_ids)
            
            # 处理语言模型的输入
            length = input_ids.shape[0]
            assert len(images_clip) == 1, f'Inference only supports one video, but got {len(images_clip)} videos.'
            images_clip = [        # num_conv = 1 * t, dim, h, w all the frames
                images_clip[0].unsqueeze(0).expand(length, -1, -1, -1, -1).contiguous().flatten(0, 1)
            ]
            
            end_time = time.time()
            print('clip time: ' + str(end_time - start_time))

            start_time = time.time()
            output_i = super().forward(
                images=images_clip,
                attention_mask=attention_masks,
                input_ids=input_ids,
                output_hidden_states=True,
            )
            torch.cuda.empty_cache()
            
            end_time = time.time()
            
            print('MLLM time: ' + str(end_time - start_time))

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
            
            track_token_mask = input_ids[:, 1:] == self.track_token_idx
            track_token_mask = torch.cat([track_token_mask, torch.zeros((track_token_mask.shape[0], 1)).bool().cuda(), ], dim=1, )
            track_token_mask = torch.cat([torch.zeros((track_token_mask.shape[0], token_add)).bool().cuda(), track_token_mask], dim=1, )

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
        
        start_time = time.time()
        
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
            num_frame = len(images_clip[i])
            start_frame_i = 0 * num_frame
            end_frame_i = (end_i - start_i) * num_frame    
            batch_pred_embeddings = pred_embeddings[start_frame_i: end_frame_i]
            batch_pred_embeddings = batch_pred_embeddings.reshape(len(batch_pred_embeddings) // num_frame, num_frame, batch_pred_embeddings.shape[-1])
            pred_embeddings_.append(batch_pred_embeddings)
            pred_embeddings = pred_embeddings[end_frame_i: ]
            
            pred_track_embeddings_.append(pred_track_embeddings[start_i: end_i])
            
        pred_track_embeddings = pred_track_embeddings_     # list: num_conversation, 1, dim
        pred_embeddings = pred_embeddings_        # list: num_conversations, t, dim

        assert len(pred_embeddings) == batch_size
        assert len(pred_track_embeddings) == batch_size
        
        # mask decoder
        pred_masks = []
        alpha = self.alpha       # 原来为0.25
        for i in range(batch_size):     # 训练阶段的batch size数量较大    
            # 视频数据，帧数和送入MLLM中的一致
            # 根据真实的关键帧数据以及pred_target_logits的分数来选择关键帧送入SAM中
            video_mask_list_ = []
            frame_ids = frame_ids_list[i]
            # 进行全局和局部的seg token融合
            track_embeddings = pred_track_embeddings[i].unsqueeze(1)       # num_conv, 1, dim
            seg_embeddings = pred_embeddings[i]      # num_conv, t, dim
            sam_images = images[i]      # t, 3, h, w
            
            for num_conv, seg_token in enumerate(track_embeddings):
                track_embed = track_embeddings[num_conv]       # 1, dim
                seg_embed = seg_embeddings[num_conv]       # t, dim
                
                similarity_ = F.cosine_similarity(seg_embed, track_embed, dim=1).unsqueeze(1)
                similarity = torch.softmax(similarity_, dim=0)     # t, 1
                
                # 对于图像和视频，都将各帧seg token按相似度和global seg token进行加权融合
                track_embed = track_embed + alpha * similarity.transpose(0, 1) @ seg_embed      # 1, dim

                # 通过clip和seg token的相似度来共同选择关键帧
                prob_ = outputs.logits_per_image[frame_ids]
                prob_ = prob_.softmax(dim=0)
                # similarity = prob_ + similarity     # t, 1
                
                end_time = time.time()
                print(end_time - start_time)
                
                # 进一步优化推理阶段关键帧的选择过程，使用object score进行筛选
                
                start_time = time.time()
                
                obj_score_per_frame = []
                for frame_idx in frame_ids:
                    # initialize the sam2 for training
                    inference_state = self.model.visual_model.init_state(sam_images)        # 3, 3, h, w
                    self.model.visual_model.reset_state(inference_state)
                    
                    _, out_obj_ids, out_mask_logits, ious, object_score_logits = self.model.visual_model.add_new_points(       # torch.float h, w
                                                                                    inference_state=inference_state,
                                                                                    frame_idx=frame_idx,
                                                                                    obj_id=1,
                                                                                    pred_embeddings=track_embed,
                                                                                    )
                    
                    obj_score_per_frame.append({'frame_ids': frame_idx,
                                                'ious': ious, 
                                                'object_score_logits': object_score_logits[0].item(), 
                                                'clip_scores': prob_[:, 0][frame_ids.index(frame_idx)].item(),      
                                                'similarity': similarity[frame_ids.index(frame_idx), 0].item(),
                                                'out_mask_logits': out_mask_logits,
                                                })
                    
                # 选择出clip score和遮挡分数最大的一组参考帧和关键帧
                clip_scores = [obj_score['clip_scores'] for obj_score in obj_score_per_frame]      # 归一化后
                object_scores = [obj_score['object_score_logits'] for obj_score in obj_score_per_frame]       # 未归一化
                object_tensor = torch.softmax(torch.tensor(object_scores).cuda(), dim=0)
                object_scores = object_tensor.tolist()
                
                seg_token_scores = [obj_score['similarity'] for obj_score in obj_score_per_frame]       # 归一化后
                # results = [a + b + c for a, b, c in zip(clip_scores, object_scores, seg_token_scores)]
                # results = [b for b in object_scores]
                results = [b + c for b, c in zip(object_scores, seg_token_scores)]
                # results = [a + b for a, b in zip(clip_scores, object_scores)]
                idx = results.index(max(results))
                cond_frame_idx, out_mask_logits = obj_score_per_frame[idx]['frame_ids'], obj_score_per_frame[idx]['out_mask_logits']
                            
                end_time = time.time()
                
                print("TKS time: " + str(end_time - start_time))
                 
                # seg_index = torch.argmax(similarity[:, 0], dim=0)       # t 筛选出相似度最高的帧级seg token
                # cond_frame_idx = frame_ids[seg_index]
                
                # frame_ids_sam, cond_frame_idx = self.sample_cond_frame(seg_index, sample_list=list(range(len(sam_images_))), num_frame=3)
                # sam_images = torch.stack([sam_images_[id] for id in frame_ids_sam], dim=0)      # 3, 3, h, w
                # gt_masks_list.append(torch.stack([mask[num_conv][id] for id in frame_ids_sam]))     # 3, h, w

                start_time = time.time()
            
                # initialize the sam2 for training
                inference_state = self.model.visual_model.init_state(sam_images)        # 3, 3, h, w
                self.model.visual_model.reset_state(inference_state)
            
                # 训练阶段仅提供一个条件帧，这里对于不同目标输入的视频不同
                _, out_obj_ids, out_mask_logits, _, _ = self.model.visual_model.add_new_points(
                        inference_state=inference_state,
                        frame_idx=cond_frame_idx,
                        obj_id=(num_conv + 1),
                        pred_embeddings=track_embed,        # 1, dim
                )
                
                # 对先前的所有状态进行重置，而后将条件帧的mask作为条件输入到模型中
                self.model.visual_model.reset_state(inference_state)
                mask_input = out_mask_logits > 0
                frame_idx, out_obj_ids, out_mask_logits = self.model.visual_model.add_new_mask_bf16(inference_state=inference_state,
                                                                                               frame_idx=cond_frame_idx,
                                                                                               obj_id=1,
                                                                                               mask=mask_input[0][0])
                
                if cond_frame_idx == 0:
                    video_mask_list = []
                    for out_frame_idx, out_obj_ids, out_mask_logits in self.model.visual_model.propagate_in_video(inference_state, 
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
                    for out_frame_idx, out_obj_ids, out_mask_logits in self.model.visual_model.propagate_in_video(inference_state, 
                                                                                                                  start_frame_idx=cond_frame_idx, 
                                                                                                                  reverse=True):      # bs=1, 1, h, w
                        masks = out_mask_logits[..., : resize_list[i][0], : resize_list[i][1]]
                        masks = F.interpolate(
                            masks, label_list[i].shape, mode="bilinear", align_corners=False
                        )      # 1, 1, h, w
                        video_mask_list.append(masks[0, 0])     # h, w
                        
                    video_masks = torch.stack(video_mask_list, dim=0)     # t, h, w
                    video_masks = torch.flip(video_masks, dims=[0])   # t, h, w
                
                else:
                    video_mask_list1 = []
                    video_mask_list2 = []
                    for reverse in [False, True]:
                        for out_frame_idx, out_obj_ids, out_mask_logits in self.model.visual_model.propagate_in_video(inference_state, 
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
                        self.model.visual_model.reset_state(inference_state)
                        frame_idx, _, _ = self.model.visual_model.add_new_mask_bf16(inference_state, cond_frame_idx, 1, mask_input[0][0])     # 清空所有条件帧，然后重新进行mask预测
                                    
                    video_pred_masks1 = torch.stack(video_mask_list1, dim=0)     # t, h, w
                    video_pred_masks2 = torch.stack(video_mask_list2, dim=0)     # t, h, w
                    video_pred_masks2 = torch.flip(video_pred_masks2, dims=[0])     # t, h, w
                    video_masks = torch.cat((video_pred_masks2[: -1], video_pred_masks1), dim=0)     # T, h, w
                    
                video_mask_list_.append(video_masks)
            
                end_time = time.time()
                
                print("Propagate time: " + str(end_time - start_time))
            
            video_masks_ = torch.stack(video_mask_list_, dim=0)      # num_conv, 3, h, w
            pred_masks.append(video_masks_.flatten(0, 1))     # num_conv * 3, h, w
                
        model_output = output
        
        gt_masks = [mm.flatten(0, 1) for mm in masks_list]
            
        if inference:
            return {
                "pred_masks": video_masks_.flatten(0, 1),
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