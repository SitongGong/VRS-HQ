import cv2
import os
import numpy as np
import json
import pycocotools.mask as maskUtils


output_path = '/18515601223/segment-anything-2/add_masks/8_'
os.makedirs(output_path, exist_ok=True)

video_path = "/18515601223/VISA-main/dataset/rvos_root/ReVOS/JPEGImages/UVO/all/2E4T5NCywcE"
mask_path = "/18515601223/segment-anything-2/failure_cases/9"

video_list = sorted(os.listdir(video_path))
mask_list = sorted(os.listdir(mask_path))

num = 0
for video_name, mask_name in zip(video_list, mask_list):
    visa_mask_path = os.path.join(video_path, video_name)
    hqvrs_mask_path = os.path.join(mask_path, mask_name)
    
    image = cv2.imread(visa_mask_path)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mask = cv2.imread(hqvrs_mask_path, cv2.IMREAD_GRAYSCALE)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    mask = mask > 0
    img_mask = image.copy()
    img_mask[mask] = (image * 0.6
                    + mask[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.4
                    )[mask]
    img_mask[mask == False] = (image * 0.5)[mask == False]

    img_mask = cv2.cvtColor(img_mask, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(output_path, mask_name), img_mask)
    