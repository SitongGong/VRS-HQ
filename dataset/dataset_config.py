import os.path as osp

# LISA    /defaultShare/dataset/rvos_root
LISA_ROOT = " /mnt/dataset/lisa_root"

# ChatUniVi
ChatUniVi_ROOT = " /mnt/dataset/chatunivi"
MIMIC_imageonly = {
    "chat_path": osp.join(ChatUniVi_ROOT, "Fine-tuning/MIMIC_imageonly/MIMIC-IT-imageonly.json"),
    "CDG"      : osp.join(ChatUniVi_ROOT, "Fine-tuning/MIMIC_imageonly/CGD/images"),
    "LA"       : osp.join(ChatUniVi_ROOT, "Fine-tuning/MIMIC_imageonly/LA/images"),
    "SD"       : osp.join(ChatUniVi_ROOT, "Fine-tuning/MIMIC_imageonly/SD/images"),
}
VIDEO = {
    "chat_path": osp.join(ChatUniVi_ROOT, "Fine-tuning/VIDEO/video_chat.json"),
    "VIDEO"    : osp.join(ChatUniVi_ROOT, "Fine-tuning/VIDEO/Activity_Videos"),
}
SQA = {
    "chat_path": osp.join(ChatUniVi_ROOT, "ScienceQA_tuning/llava_train_QCM-LEA.json"),
    "ScienceQA": osp.join(ChatUniVi_ROOT, "ScienceQA_tuning/train"),
}

# RVOS
RVOS_ROOT = "/mnt/dataset/rvos_root"
RVOS_DATA_INFO = {
    "mevis_train"   : ("MeViS/train",           "MeViS/train/meta_expressions.json"),
    "mevis_val"     : ("MeViS/valid_u",         "MeViS/valid_u/meta_expressions.json"),
    "mevis_test"    : ("MeViS/valid",           "MeViS/valid/meta_expressions.json"),
    "refytvos_train": ('Ref_Youtube_VOS/train', 'Ref_Youtube_VOS/meta_expressions/train/meta_expressions.json'),
    "refytvos_valid": ('Ref_Youtube_VOS/valid', 'Ref_Youtube_VOS/meta_expressions/valid/meta_expressions.json'),
    "davis17_train" : ('davis17/train',         'davis17/meta_expressions/train/meta_expressions.json'),
    "davis17_valid" : ('davis17/valid',         'davis17/meta_expressions/valid/meta_expressions.json'),
    "revos_train"   : ('ReVOS',                 'ReVOS/meta_expressions_train_.json'),
    "revos_valid"   : ('ReVOS',                 'ReVOS/meta_expressions_valid__llamavid.json'),    # ReVOS/meta_expressions_valid__llamavid.json
    "lvvis_train"   : ("lvvis/train",           "lvvis/train/meta_expressions.json"),
}
