# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch.nn.functional as F

def build_sam2(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
):

    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        ]
    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides_extra)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model

# @hydra.main(config_name="config_file.yaml", config_path="18515601223/SAM2Long/sam2/configs/sam2")
def build_sam2_video_predictor(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
):
    hydra_overrides = [
        "++model._target_=sam2.sam2long_video_predictor.SAM2VideoPredictor",
    ]
    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            # the sigmoid mask logits on interacted frames with clicks in the memory encoder so that the encoded masks are exactly as what users see from clicking
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            # fill small holes in the low-res masks up to `fill_hole_area` (before resizing them to the original video resolution)
            "++model.fill_hole_area=8",
        ]
    hydra_overrides.extend(hydra_overrides_extra)

    # Read config and init model
    # with initialize(config_path="18515601223/SAM2Long/sam2/configs/sam2"):
    #     cfg = compose(config_name="sam2_hiera_b+.yaml", overrides=hydra_overrides)
    cfg = compose(config_name=config_file, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    # _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


# 重写一下sam的构建过程，可以不用直接安装
# from sam2.sam2_video_predictor_ import SAM2VideoPredictor      # SAM1 inference
from sam2.sam2_video_inference_ import SAM2VideoPredictor
# from sam2.sam2_video_adapter import SAM2VideoPredictor
# from sam2.sam2_video_predictor import SAM2VideoPredictor

from sam2.modeling.backbones.image_encoder import ImageEncoder, FpnNeck
# from sam2.modeling.backbones.hieradet_adapter import Hiera
from sam2.modeling.backbones.hieradet import Hiera
from sam2.modeling.position_encoding import PositionEmbeddingSine

from sam2.modeling.memory_attention import MemoryAttention, MemoryAttentionLayer
from sam2.modeling.sam.transformer import RoPEAttention

from sam2.modeling.memory_encoder import MemoryEncoder, MaskDownSampler, Fuser, CXBlock
# from sam2.sam2_video_propagator import SAM2VideoTrainer
# from sam2.sam2_inference import SAM2VideoPredictor

def _build_sam2_(ckpt_path=None,
                 device="cuda",
                 mode="eval",
                 video_inference=False,
                 if_substitute=False):
    
    if video_inference:
        sam2 = SAM2VideoPredictor
    else:
        sam2 = SAM2VideoPredictor
    
    model = sam2(image_encoder=ImageEncoder(trunk=Hiera(embed_dim=144,
                                                        num_heads=2,
                                                        window_spec=[8, 4, 16, 8],
                                                        window_pos_embed_bkg_spatial_size=[7, 7],
                                                        global_att_blocks=[23, 33, 43],
                                                        stages=[2, 6, 36, 4],
                                                           ),
                                               neck=FpnNeck(d_model=256,
                                                            backbone_channel_list=[1152, 576, 288, 144],
                                                            fpn_top_down_levels=[2, 3],  # output level 0 and 1 directly use the backbone features
                                                            fpn_interp_model='nearest',
                                                            position_encoding=PositionEmbeddingSine(num_pos_feats=256,
                                                                                                    normalize=True,
                                                                                                    scale=None,
                                                                                                    temperature=10000),),
                                               scalp=1),
                    memory_attention=MemoryAttention(d_model=256,
                                                     pos_enc_at_input=True,
                                                     num_layers=4,
                                                     layer=MemoryAttentionLayer(activation='relu',
                                                                                dim_feedforward=2048,
                                                                                dropout=0.1,
                                                                                pos_enc_at_attn=False,
                                                                                d_model=256,
                                                                                pos_enc_at_cross_attn_keys=True,
                                                                                pos_enc_at_cross_attn_queries=False,
                                                                                self_attention=RoPEAttention(rope_theta=10000.0,
                                                                                                             feat_sizes=[32, 32],
                                                                                                             embedding_dim=256,
                                                                                                             num_heads=1,
                                                                                                             downsample_rate=1,
                                                                                                             dropout=0.1),
                                                                                cross_attention=RoPEAttention(rope_theta=10000.0,
                                                                                                             feat_sizes=[32, 32],
                                                                                                             embedding_dim=256,
                                                                                                             num_heads=1,
                                                                                                             downsample_rate=1,
                                                                                                             dropout=0.1,
                                                                                                             kv_in_dim=64,
                                                                                                             rope_k_repeat=True, 
                                                                                                             ),),),
                    memory_encoder=MemoryEncoder(out_dim=64,
                                                 position_encoding=PositionEmbeddingSine(num_pos_feats=64,
                                                                                         normalize=True,
                                                                                         scale=None,
                                                                                         temperature=10000),
                                                 mask_downsampler=MaskDownSampler(kernel_size=3, stride=2, padding=1),
                                                 fuser=Fuser(CXBlock(dim=256,
                                                                     kernel_size=7,
                                                                     padding=3,
                                                                     layer_scale_init_value=1e-6,
                                                                     use_dwconv=True),
                                                             num_layers=2),),
                    num_maskmem=7,
                    image_size=1024,
                    # apply scaled sigmoid on mask logits for memory encoder, and directly feed input mask as output mask
                    sigmoid_scale_for_mem_enc=20.0,
                    sigmoid_bias_for_mem_enc=-10.0,
                    use_mask_input_as_output_without_sam=True,
                    # Memory
                    directly_add_no_mem_embed=True,
                    # use high-resolution feature map in the SAM mask decoder
                    use_high_res_features_in_sam=True,
                    # output 3 masks on the first click on initial conditioning frames
                    multimask_output_in_sam=False, # True if video_inference else False,
                    # SAM heads
                    iou_prediction_use_sigmoid=True,
                    # cross-attend to object pointers from other frames (based on SAM output tokens) in the encoder
                    use_obj_ptrs_in_encoder=True,  #  if video_inference else False,
                    add_tpos_enc_to_obj_ptrs=False,
                    only_obj_ptrs_in_the_past_for_eval=True,
                    # object occlusion prediction
                    pred_obj_scores=True, # if video_inference else False,
                    pred_obj_scores_mlp=True, #,  if video_inference else False,
                    fixed_no_obj_ptr=True, # if video_inference else False,
                    # multimask tracking settings
                    multimask_output_for_tracking=False, # True if video_inference else False,
                    use_multimask_token_for_obj_ptr=False, # True if video_inference else False,
                    multimask_min_pt_num=0,
                    multimask_max_pt_num=1,
                    use_mlp_for_obj_ptr_proj=True,
                    # Compilation flag
                    compile_image_encoder=False,
                    # post processing
                    # the sigmoid mask logits on interacted frames with clicks in the memory encoder so that the encoded masks are exactly as what users see from clicking
                    binarize_mask_from_pts_for_mem_enc=True,
                    # fill small holes in the low-res masks up to `fill_hole_area` (before resizing them to the original video resolution)
                    fill_hole_area=8, 
                    sam_mask_decoder_extra_args={ 
                    # dynamically fall back to multi-mask if the single mask is not stable
                    "dynamic_multimask_via_stability": False, # True,
                    "dynamic_multimask_stability_delta": 0.05,
                    "dynamic_multimask_stability_thresh": 0.98})
    # if not video_inference:
    _load_checkpoint(model, ckpt_path, if_substitute)
    # model = model.to(device)
    if mode == "eval":
        model.eval()
    return model
    

def _load_checkpoint(model, ckpt_path, if_substitute=False):
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["model"]
        
        # 将sam_mask_decoder的权重进行替换
        # if if_substitute:
        #     new_state_dict = torch.load("/18515601223/segment-anything-2/save_weights/pytorch_model-00003-of-00003.bin", map_location="cpu") 
        #     for key, value in sd.items():
        #         if key.split('.')[0] == "sam_mask_decoder" or key.split('.')[0] == "memory_encoder" or key.split('.')[0] == 'image_encoder' or key.split('.')[0] == "memory_attention"\
        #         or key.split('.')[0] == "sam_prompt_encoder" or key.split('.')[0] == "mask_downsample" or key.split('.')[0] == "maskmem_tpos_enc"\
        #         or key.split('.')[0] == 'obj_ptr_proj' or key.split('.')[0] == 'no_obj_ptr' or key.split('.')[0] == 'no_mem_embed' or key.split('.')[0] == 'no_mem_pos_enc':
        #             print(key)
        #             sd[key] = new_state_dict['model.visual_model.' + key]
        
        missing_keys, unexpected_keys = model.load_state_dict(sd, strict=False)
        
        print(missing_keys, unexpected_keys)
        # if missing_keys:
        #     logging.error(missing_keys)
        #     raise RuntimeError()
        # if unexpected_keys:
        #     logging.error(unexpected_keys)
        #     raise RuntimeError()
        logging.info("Loaded checkpoint sucessfully")


if __name__ == "__main__":
    # use bfloat16 for the entire notebook
    # torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    
    # sam2_trainer = _build_sam2_(ckpt_path='/18515601223/segment-anything-2/checkpoints/sam2_hiera_large.pt')
    # image = torch.rand((1, 3, 1024, 1024)).to(device='cuda', dtype=torch.float32)
    # video = torch.rand((5, 3, 1024, 1024)).to(device='cuda', dtype=torch.float32)
    # pred_embeddings = [torch.rand((0, 1, 256)).to(device='cuda', dtype=torch.float32), torch.rand((0, 2, 256)).to(device='cuda', dtype=torch.float32)]
    # image_list = [image, video]
    # cond_frame_list = [[0], [0, 2]]
    # sam2_trainer.init_state(video_path='/18515601223/segment-anything-2/notebooks/videos/bedroom')
    # _ = sam2_trainer.video_forward(image_list, pred_embeddings, cond_frame_list)
    
    sam2_inference = _build_sam2_(ckpt_path=None, video_inference=True)
    sam2_inference = sam2_inference.to('cuda')
    
    video = torch.rand((20, 3, 1024, 1024)).to(device='cuda', dtype=torch.float32)
    cond_frame_list = [10]
    pred_embeddings = torch.rand((3, 1, 256)).to(device='cuda', dtype=torch.float32)
    inference_state = sam2_inference.init_state(video)
    sam2_inference.reset_state(inference_state)
    for num, seg_token in enumerate(pred_embeddings):
        _, out_obj_ids, out_mask_logits = sam2_inference.add_new_points(
            inference_state=inference_state,
            frame_idx=cond_frame_list[0],
            obj_id=num+1,
            pred_embeddings=seg_token.unsqueeze(0),      # 1, 1, dim
        )
    
    # sam2_inference.reset_state(inference_state)
    out_mask_logits = out_mask_logits.to(dtype=torch.bool)
    frame_idx, out_obj_ids, out_mask_logits = sam2_inference.add_new_mask(inference_state=inference_state,
                                frame_idx=cond_frame_list[0],
                                obj_id=1,
                                mask=out_mask_logits[0][0])
    print(out_obj_ids, out_mask_logits)

    video_mask_list1 = []
    video_mask_list2 = []
    for reverse in [False, True]:
        for out_frame_idx, out_obj_ids, out_mask_logits in sam2_inference.propagate_in_video(inference_state,  
                                                                    start_frame_idx=frame_idx, 
                                                                    reverse=reverse):
                
            if reverse == False:
                video_mask_list1.append(out_mask_logits[0][0])
            else:
                video_mask_list2.append(out_mask_logits[0][0])
                
        sam2_inference.reset_state(inference_state)
        out_mask_logits = out_mask_logits.to(dtype=torch.bool)
        frame_idx, out_obj_ids, out_mask_logits = sam2_inference.add_new_mask(inference_state=inference_state,
                                frame_idx=cond_frame_list[0],
                                obj_id=1,
                                mask=out_mask_logits[0][0])
                
    video_pred_masks1 = torch.stack(video_mask_list1, dim=0)     # t, h, w
    video_pred_masks2 = torch.stack(video_mask_list2, dim=0)     # t, h, w
    video_pred_masks2 = torch.flip(video_pred_masks2, dims=[0])     # t, h, w
    video_masks = torch.cat((video_pred_masks2[:-1], video_pred_masks1), dim=0)     # T, h, w