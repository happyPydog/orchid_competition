import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()


# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Path to dataset, could be overwritten by command line argument
_C.DATA.DATA_PATH = "orchid_dataset"
# Image size for training on low resolution
_C.DATA.TRAINING_IMAGE_SIZE = 192
# Image size for fine-tune on high resolution
_C.DATA.FINE_TUNE_IMAGE_SIZE = 384
# Interpolation to resize image (random, bilinear, bicubic)
_C.DATA.INTERPOLATION = "bicubic"
# Number of batch size for training
_C.TRAINING_BATCH_SIZE = 32
# Number of batch size for fine-tune
_C.FINE_TUNE_BATCH_SIZE = 2


# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Pretrained weight from checkpoint, could be imagenet22k pretrained weight
# could be overwritten by command line argument
_C.MODEL.PRETRAINED = ""
# Number of classes, overwritten in data preparation
_C.MODEL.NUM_CLASSES = 219
# Dropout rate
_C.MODEL.DROP_RATE = 0.1
# Attention dropout rate
_C.MODEL.ATTN_DROP_RATE = 0.1
# Drop path rate
_C.MODEL.DROP_PATH_RATE = 0.1
# Label Smoothing
_C.MODEL.LABEL_SMOOTHING = 0.1

# -----------------------------------------------------------------------------
# Swin Transformer parameters
# -----------------------------------------------------------------------------
_C.MODEL.SWIN = CN()
# Model name for training on low resolution
_C.MODEL.TRAINING_NAME = "swinv2_base_window12_192_22k"
# Model name for fine tune on high resolution
_C.MODEL.FINE_TUNE_NAME = "swinv2_base_window12to24_192to384_22kft1k"
# model papermeters setting
_C.MODEL.SWINV2.PATCH_SIZE = 4
_C.MODEL.SWINV2.IN_CHANS = 3
_C.MODEL.SWINV2.EMBED_DIM = 128
_C.MODEL.SWINV2.DEPTHS = [2, 2, 18, 2]
_C.MODEL.SWINV2.NUM_HEADS = [4, 8, 16, 32]
_C.MODEL.SWINV2.TRAINING_WINDOW_SIZE = 12
_C.MODEL.SWINV2.FINE_TUNE_WINDOW_SIZE = 24
_C.MODEL.SWINV2.MLP_RATIO = 4.0
_C.MODEL.SWINV2.QKV_BIAS = True
_C.MODEL.SWINV2.APE = False
_C.MODEL.SWINV2.PATCH_NORM = True
_C.MODEL.SWINV2.PRETRAINED_WINDOW_SIZES = [12, 12, 12, 6]

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.EPOCHS = 300
_C.TRAIN.WARMUP_EPOCHS = 5
_C.TRAIN.WEIGHT_DECAY = 0.02
_C.TRAIN.BASE_LR = 3e-3
_C.TRAIN.WARMUP_LR = 1e-5
_C.TRAIN.MIN_LR = 1e-5
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 5.0

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = "cosine"

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = "adamw"
