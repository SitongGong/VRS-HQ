from enum import Enum

import numpy as np
import torch
import torch.distributed as dist

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
VIDEO_TOKEN_INDEX = -201
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_AUDIO_TOKEN = "<audio>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_SINGLE_IMAGE_TOKEN = "<single_image>"

SHORT_QUESTION_LIST = [
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you segment the {class_name} in this image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Please segment the {class_name} in this image.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "What is {class_name} in this image? Please respond with segmentation mask.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "What is {class_name} in this image? Please output segmentation mask.",
]

LONG_QUESTION_LIST = [
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please respond with segmentation mask.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please output segmentation mask.",
]

EXPLANATORY_QUESTION_LIST = [
    "Reference Video: " + DEFAULT_VIDEO_TOKEN + " \n"
    "{sent} Please output segmentation mask and explain why.",
    "Reference Video: " + DEFAULT_VIDEO_TOKEN + " \n"
    "{sent} Please output segmentation mask and explain the reason.",
    "Reference Video: " + DEFAULT_VIDEO_TOKEN + " \n"
    "{sent} Please output segmentation mask and give some explanation.",
]

UNIFIED_SHORT_QUESTION_LIST = [
    "Reference Video: " + DEFAULT_VIDEO_TOKEN + " \n"
    "What is {sent} in the Reference Video? Please respond with segmentation masks for each frame and for the entire video separately.",
    "Reference Video: " + DEFAULT_VIDEO_TOKEN + " \n"
    "What is {sent} in the Reference Video? Please output segmentation masks for each frame and for the entire video separately.",
    "Reference Video: " + DEFAULT_VIDEO_TOKEN + " \n"
    "Please find {sent} in the Reference Video and segment it in each frame and in the entire video respectively.",
]

UNIFIED_LONG_QUESTION_LIST = [
    "Reference Video: " + DEFAULT_VIDEO_TOKEN + " \n"
    "{sent} Please find it in the Reference Video and respond with segmentation masks for each frame and for the entire video separately.",
    "Reference Video: " + DEFAULT_VIDEO_TOKEN + " \n"
    "{sent} Please find it in the Reference Video and output segmentation masks in frame-level and video-level separately.",
]




ANSWER_LIST = [
    "Sure, the segmentation result of each frame is [SEG], while for the entire video it is [TAK].",
    "Sure, in the frame-level it is [SEG], and in the video-level it is [TAK].",
    "The results for each frame is [SEG], while the segmentation result for the entire video is [TAK].",
    "Sure, the segmentation result for individual frame is [SEG], and for the complete video, it's [TAK].",
]


RVOS_MULTISEG_QUESTION_LIST = [
    DEFAULT_VIDEO_TOKEN + "\n" + "Can you segment the {sent} in this video?",
    DEFAULT_VIDEO_TOKEN + "\n" + "Please segment the {sent} in this video.",
    DEFAULT_VIDEO_TOKEN
    + "\n"
    + "What is {sent} in this video? Please respond with segmentation mask.",
    DEFAULT_VIDEO_TOKEN
    + "\n"
    + "What is {sent} in this video? Please output segmentation mask.",
]


RVOS_MULTISEG_ANSWER_LIST = [
    "It is {seg}.",
    "Sure, {seg}.",
    "Sure, it is {seg}.",
    "Sure, the segmentation result is {seg}.",
]

RVOS_COT_ANSWER = "Sure, I will output the segmentation result step by step: " + " \n" + \
    "Step 1: First, I will analyze the reference video frames to understand the overall scene." + " \n" + \
    "Step 2: Then, I will carefully examine the provided description, identifying the main object and its key attributes, such as color, position, and action." + " \n" + \
    "Step 3: Based on this analysis, I will locate the object that best matches the description in the target frame." + "\n" + \
    "Finally, I will generate and output the segmentation result for the target object in the target frame." + "\n"


def convert2imagesplit(sent: str, video_len: int) -> str:
    assert DEFAULT_VIDEO_TOKEN in sent, "only support video token"
    assert sent.count(DEFAULT_VIDEO_TOKEN) == 1, "only support one video token"
    replace_sent = ", ".join(f'({i}){DEFAULT_IMAGE_TOKEN}' for i in range(video_len))
    return sent.replace(DEFAULT_VIDEO_TOKEN, replace_sent)

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(self.sum, np.ndarray):
            total = torch.tensor(
                self.sum.tolist()
                + [
                    self.count,
                ],
                dtype=torch.float32,
                device=device,
            )
        else:
            total = torch.tensor(
                [self.sum, self.count], dtype=torch.float32, device=device
            )

        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        if total.shape[0] > 2:
            self.sum, self.count = total[:-1].cpu().numpy(), total[-1].cpu().item()
        else:
            self.sum, self.count = total.tolist()
        self.avg = self.sum / (self.count + 1e-5)

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K - 1)
    area_output = torch.histc(output, bins=K, min=0, max=K - 1)
    area_target = torch.histc(target, bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def dict_to_cuda(input_dict):
    for k, v in input_dict.items():
        if isinstance(input_dict[k], torch.Tensor):
            input_dict[k] = v.cuda(non_blocking=True)
        elif (
            isinstance(input_dict[k], list)
            and len(input_dict[k]) > 0
            and isinstance(input_dict[k][0], torch.Tensor)
        ):
            input_dict[k] = [ele.cuda(non_blocking=True) for ele in v]
    return input_dict
