###########################################################################
# Created by: BUAA
# Email: clyanhh@gmail.com
# Copyright (c) 2024
###########################################################################
import itertools
import json
import os
import os.path as osp
import pickle
import sys
import cv2
import time
import random
import logging
import math
import torch
import torch.nn.functional as F
from pprint import pprint
from tqdm import tqdm
from termcolor import colored

import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle
import pycocotools.mask as maskUtils
from model.llava import conversation as conversation_lib
# from model.videollama2 import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide
from transformers import CLIPImageProcessor
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from model.llava.mm_utils import tokenizer_image_token

from .d2_datasets.refytvos_utils import load_refytvos_json
from .d2_datasets.mevis_utils import load_mevis_json
from .d2_datasets.refytvos_val_videos import REFYTVOS_VAL_VIDEOS
from .utils import (
    UNIFIED_SHORT_QUESTION_LIST, UNIFIED_LONG_QUESTION_LIST, ANSWER_LIST,
    DEFAULT_VIDEO_TOKEN, DEFAULT_IMAGE_TOKEN, convert2imagesplit
)
from .utils import (
    DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX, 
    convert2imagesplit, UNIFIED_SHORT_QUESTION_LIST, UNIFIED_LONG_QUESTION_LIST
)

from .dataset_config import RVOS_DATA_INFO as _DATA_INFO
from .dataset_config import RVOS_ROOT
import time

logger = logging.getLogger(__name__)

def get_zero_image(processor):
    i = Image.new('RGB', (224, 224), (0, 0, 0))
    return processor.preprocess(i, return_tensors='pt')['pixel_values'][0]


def sample_frames(index_frame, images_clip, max_image_token=12):
    if max_image_token != 1:    # 最多可以输入多少帧
        to_devide = (max_image_token - 1)
        step_size = math.ceil(len(images_clip) / to_devide)
        idx_start = index_frame % step_size
        idx_select = list(range(idx_start, len(images_clip), step_size))      # 以step size为间隔，等距离抽帧
    else:
        idx_select = [index_frame, ]
    assert index_frame in idx_select
    
    select_images = [images_clip[idx] for idx in idx_select]
    select_images = select_images + [images_clip[index_frame]]
    
    return torch.stack(select_images, dim=0)


def collate_fn_(
    batch, tokenizer=None, conv_type="llava_v1", use_mm_start_end=True, clip_model=None, local_rank=-1
):
    image_path_list = []
    images_list = []
    images_clip_list = []
    conversation_list = []
    masks_list = []
    label_list = []
    resize_list = []
    questions_list = []
    sampled_classes_list = []
    offset_list = [0]
    cnt = 0
    inferences = []
    num_frame_list = []
    num_conv_list = []
    conditional_frame_ids = []
    clip_input_list = []
    for (
        image_path,
        images,
        images_clip,
        conversations,
        masks,
        label,
        resize,
        questions,
        sampled_classes,
        condition_frame_ids, 
        inputs,  
        inference,
    ) in batch:          # 对于一个batch中的图像的集中处理
        image_path_list.append(image_path)

        # inputs = {key: value.to('cuda') for key, value in inputs.items()}
        # outputs = clip_model(**inputs)
        # prob_per_frame = outputs.logits_per_image.softmax(dim=0)
        # index_frame = torch.argmax(prob_per_frame[:, 0], dim=0)
        # condition_frame_ids[0] = index_frame
        
        # images_clip = sample_frames(index_frame, images_clip)

        if images.ndim > 3:
            pass           # t, 3, 1024, 1024

        if images.ndim == 3:
            images = images.unsqueeze(0)
        assert images.ndim == 4
        images_list.append(images)    # t, 3, 1024, 1024

        if images_clip.ndim == 3:
            images_clip = images_clip.unsqueeze(0)
        assert images_clip.ndim == 4
        images_clip_list.append(images_clip)
        num_frame = images_clip.shape[0]        # T, 3, 224, 224
        num_frame_list.append(num_frame)

        conversation_list.extend(conversations)
        label_list.append(label)          # 720, 1280
        num_conv_list.append(len(conversations))

        if masks.ndim == 3:  # [num_classes, H, W]
            if masks.shape[0] == 0:  # [0, H, W] -> [num_classes, 0, H, W]
                masks = torch.stack([masks, ] * len(conversations), dim=0).float()
            else: # [num_classes, H, W] -> [num_classes, 1, H, W]
                masks = masks.unsqueeze(1).float()     # num_classes, 1, H, W
        assert masks.ndim == 4        # num_classes, num_frames, H, W
        masks_list.append(masks.float())

        resize_list.append(resize)          # 14, 1024
        questions_list.append(questions)
        sampled_classes_list.append(sampled_classes)

        cnt += len(conversations)
        offset_list.append(cnt)
        inferences.append(inference)
        conditional_frame_ids.append(condition_frame_ids)
        clip_input_list.append(inputs)

    if use_mm_start_end:      # <image> token -> <im_start> <image> <im_end> token
        # replace <image> token 
        for i in range(len(conversation_list)):
            replace_token = DEFAULT_IMAGE_TOKEN
            replace_token = (
                DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            )
            conversation_list[i] = conversation_list[i].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )

    # 将所有的conversation标记化
    input_ids = [
        tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        for prompt in conversation_list
    ]
    input_ids = torch.nn.utils.rnn.pad_sequence(          # num_conversation, num_tokens
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    conv = conversation_lib.default_conversation.copy()
    targets = input_ids.clone()

    if conv_type == "llava_v1":
        sep = conv.sep + conv.roles[1] + ": "    # 找到分割user和assistant的标记
    else:
        sep = "[/INST] "
    for conversation, target in zip(conversation_list, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)    # 对话轮数
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            # if len(parts) != 2:
            #     break
            assert len(parts) == 2, (len(parts), rou)
            parts[0] += sep

            if DEFAULT_IMAGE_TOKEN in conversation:
                round_len = len(tokenizer_image_token(rou, tokenizer))      
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2   
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX    

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            assert cur_len == total_len

    return {
        "image_paths": image_path_list,
        "images": images_list, #BS : T(or 1 for chatunivi) * 3 * H * W
        "images_clip": images_clip_list, #BS : T * 3 * H * W
        "input_ids": input_ids,      # num_conversation, ntokens
        "labels": targets,
        "attention_masks": attention_masks,
        "masks_list": masks_list, # [Conv*Frame*H*W, ...]
        "label_list": label_list, # [H*W, ...]    gt label
        "resize_list": resize_list,       # [14, 1024]
        "offset": torch.LongTensor(offset_list), #[0, num_conv0, num_conv1, ...]  
        "questions_list": questions_list,        # bs
        "sampled_classes_list": sampled_classes_list,
        "inference": inferences[0],
        "conversation_list": conversation_list,      
        "num_frame_list": num_frame_list,      # bs
        "num_conv_list": num_conv_list,      # bs
        "clip_input_list": clip_input_list, 
        "tokenizer": tokenizer, 
        "cond_frame_list": conditional_frame_ids,    # None, [0], [0, 3, 7, ...]
    }


class RVOSEvalDataset(torch.utils.data.Dataset):
    # davis17_train, refytvos_train, mevis_train
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255
    def __init__(
        self, 
        tokenizer,
        vision_tower,
        output_dir: str,  # osp.join(output_dir, "Annotations", video_name, exp_id, f"{frame_id}.png")
        precision               : str  = "fp32",
        image_size              : int  = 224,
        rvos_dataset_name       : str  = "refytvos_train",
        max_image_token         : int  = 12,
        clip_vision_tower       : str  = "openai/clip-vit-large-patch14-336", 
        device                  : str  = "cuda"
    ):
        assert rvos_dataset_name in _DATA_INFO.keys(), f"dataset {rvos_dataset_name} not found!"
        self.root = RVOS_ROOT
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.device = device
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
        
        self.clip_processsor = CLIPProcessor.from_pretrained(clip_vision_tower)

        assert max_image_token < 20, "max_image_token must < 20"
        self.max_image_token = max_image_token

        self.long_question_list = UNIFIED_LONG_QUESTION_LIST
        self.short_question_list = UNIFIED_SHORT_QUESTION_LIST

        self.answer_list = ANSWER_LIST

        self.output_dir = output_dir
        self.rvos_dataset_name = rvos_dataset_name
        assert self.rvos_dataset_name in _DATA_INFO.keys(), f"dataset {self.rvos_dataset_name} not found!"
        print(f"loading dataset {self.rvos_dataset_name} into memory...")
        image_root, json_file = _DATA_INFO[self.rvos_dataset_name]
        self.image_root = osp.join(self.root, image_root)
        self.json_file = osp.join(self.root, json_file)
        self.d2_dataset_dicts, self.lisa_dataset_dicts = self.load_target_frame()

    def __len__(self):
        return len(self.lisa_dataset_dicts)
    
    def load_target_frame(self, ):
        metas, _, _, _ = load_mevis_json(self.image_root, self.json_file, self.rvos_dataset_name)
    
        d2_dataset_dicts = []
        lisa_dataset_dicts = []
        
        tmp_valid_vid = 0
        for idx_vd, vid_dict in tqdm(enumerate(metas), desc=f'Loading {self.rvos_dataset_name} ...'):
            record = {}
            if (self.rvos_dataset_name == "refytvos_valid") and (vid_dict['video'] not in REFYTVOS_VAL_VIDEOS):
                continue
            record["file_names"] = [
                os.path.join(self.image_root, 'JPEGImages', vid_dict['video'], vid_dict["frames"][i]+ '.jpg') 
                for i in range(vid_dict["length"])
            ]
            record["length"] = vid_dict["length"]
            video_name, exp, anno_ids, obj_ids, category, exp_id = \
                vid_dict['video'], vid_dict['exp'], vid_dict['anno_id'], vid_dict['obj_id'], vid_dict['category'],  vid_dict['exp_id']

            exp = " ".join(exp.lower().split())
            if "eval_idx" in vid_dict:
                record["eval_idx"] = vid_dict["eval_idx"]

            video_objs = []
            record["annotations"] = video_objs
            record["sentence"]    = exp
            record["exp_id"]      = exp_id
            record["video_name"]  = video_name
            d2_dataset_dicts.append(record)       # used to restore the content of each frame
                  
            lisa_dataset_dicts.append(
                dict(
                    idx_d2 = tmp_valid_vid, 
                    frame_path = record['file_names'], # [target_frame_ids], 
                    output_path = [osp.join(self.output_dir, "Annotations", video_name, exp_id, vid_dict["frames"][i]+ '.png') for i in range(vid_dict["length"])], 
                    # inputs = inputs, 
                    # osp.join(self.output_dir, "Annotations", video_name, exp_id, target_image + '.png')
                )
            )
            
            tmp_valid_vid += 1
            
        return d2_dataset_dicts, lisa_dataset_dicts


    def __getitem__(self, idx):
        data_lisa   = self.lisa_dataset_dicts[idx]
        data_d2     = self.d2_dataset_dicts[data_lisa['idx_d2']]
        frame_path_list  = data_lisa['frame_path']
        image_shape = cv2.imread(frame_path_list[0]).shape[:2]
        zero_mask   = np.zeros(image_shape, dtype=np.uint8)
        data = {
            "video_name"           : data_d2['video_name'],       # video_name
            "video_frame_path_list": data_d2['file_names'],       # all the frame
            "seg_frame_path"       : frame_path_list,      # target_frame
            "exp_mask_pairs"       : [(data_d2['sentence'], zero_mask)],
        }

        clip_frame_list = [Image.open(frame_path) for frame_path in frame_path_list]
        exp, _ = data["exp_mask_pairs"][0]
        inputs = self.clip_processsor(text=[exp], images=clip_frame_list, return_tensors="pt", padding=True)
        
        data['video_frame_clip'] = [data['video_frame_path_list'][i] for i in range(len(data['video_frame_path_list']))]

        # frames that will be sent to the clip vision encoder
        frame_list = [cv2.imread(x) for x in data['video_frame_clip']]
        frame_list = [cv2.cvtColor(x, cv2.COLOR_BGR2RGB) for x in frame_list]
        frame_clip_list = [self.clip_image_processor(x, return_tensors="pt")["pixel_values"][0] for x in frame_list]

        video_len = len(frame_clip_list)
        
        # target frame that will be sent to the sam encoder, we will sent all the frames to the sam encoder
        condition_index = 0 # data['video_frame_path_list'].index(data['seg_frame_path'])    # the target frame index
        seg_frame_sam = [cv2.imread(x) for x in data['video_frame_path_list']]
        seg_frame_list = [cv2.cvtColor(x, cv2.COLOR_BGR2RGB) for x in seg_frame_sam]
        seg_frame_list = [self.transform.apply_image(x) for x in seg_frame_list]
        resize = seg_frame_list[0].shape[:2]
        seg_frame_sam_ = [self.preprocess(torch.from_numpy(x).permute(2, 0, 1).contiguous()) for x in seg_frame_list]

        questions = []
        answers = []
        conversations = []
        masks = []
        conv = conversation_lib.default_conversation.copy()
        for exp, mask in data['exp_mask_pairs']:
            text = exp.strip()
            assert len(text.split('||')) == 1
            # question_template = random.choice(self.long_question_list)
            if text[-1] == '?':
                question = self.long_question_list[0].format(sent=text)
            else:
                question = self.short_question_list[0].format(sent=text)
            
            questions.append(question)
            answers.append(self.answer_list[0])

            conv.messages = []
            conv.append_message(conv.roles[0], questions[-1])
            conv.append_message(conv.roles[1], answers[-1])
            conversations.append(conv.get_prompt())

            masks.append(mask)

        masks = torch.from_numpy(np.stack(masks, axis=0))
        label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label

        return (
            data_lisa['output_path'], 
            torch.stack(seg_frame_sam_, dim=0),           # images sent to sam2
            torch.stack(frame_clip_list, dim=0),          # images sent to clip encoder
            conversations,
            masks,
            label,
            resize,
            questions,
            [exp for exp, _ in data['exp_mask_pairs']],
            [condition_index], 
            inputs, 
            True,
        )

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

