# --------------------------------------------------------
# Unified Contrastive Learning (UniCL)
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Jianwei Yang (jianwyan@microsoft.com)
# Based on Swin Transformer written by Zhe Liu
# --------------------------------------------------------

import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()
_C.VERBOSE = False

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 128
# Path to dataset, could be overwritten by command line argument
_C.DATA.DATA_PATH = ''
# Dataset name
_C.DATA.DATASET = 'imagenet'
# Input image size
_C.DATA.IMG_SIZE = 224
# Interpolation to resize image (random, bilinear, bicubic)
_C.DATA.INTERPOLATION = 'bicubic'
# Use zipped dataset instead of folder dataset
# could be overwritten by command line argument
_C.DATA.ZIP_MODE = False
# Cache Data in Memory, could be overwritten by command line argument
_C.DATA.CACHE_MODE = 'part'
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 8

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model name
_C.MODEL.NAME = ''
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ''
# Number of classes, overwritten in data preparation
_C.MODEL.NUM_CLASSES = 0
# Label Smoothing
_C.MODEL.LABEL_SMOOTHING = 0.1
# Whether load pretrained model
_C.MODEL.PRETRAINED = ''
# Projection dimension
_C.MODEL.DIM_PROJECTION = 512
# Mode specific
_C.MODEL.SPEC = CN(new_allowed=True)
# -----------------------------------------------------------------------------
# Build Image Encoder
# -----------------------------------------------------------------------------
_C.MODEL.IMAGE_ENCODER = CN()
# Image encoder type
_C.MODEL.IMAGE_ENCODER.TYPE = 'swin'
# Input image size
_C.MODEL.IMAGE_ENCODER.IMG_SIZE = 224
# Dropout rate
_C.MODEL.IMAGE_ENCODER.DROP_RATE = 0.0
# Drop path rate
_C.MODEL.IMAGE_ENCODER.DROP_PATH_RATE = 0.1

# Swin Transformer parameters
_C.MODEL.IMAGE_ENCODER.SWIN = CN()
_C.MODEL.IMAGE_ENCODER.SWIN.PATCH_SIZE = 4
_C.MODEL.IMAGE_ENCODER.SWIN.IN_CHANS = 3
_C.MODEL.IMAGE_ENCODER.SWIN.EMBED_DIM = 96
_C.MODEL.IMAGE_ENCODER.SWIN.DEPTHS = [2, 2, 6, 2]
_C.MODEL.IMAGE_ENCODER.SWIN.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.IMAGE_ENCODER.SWIN.WINDOW_SIZE = 7
_C.MODEL.IMAGE_ENCODER.SWIN.MLP_RATIO = 4.
_C.MODEL.IMAGE_ENCODER.SWIN.QKV_BIAS = True
_C.MODEL.IMAGE_ENCODER.SWIN.QK_SCALE = None
_C.MODEL.IMAGE_ENCODER.SWIN.APE = False
_C.MODEL.IMAGE_ENCODER.SWIN.PATCH_NORM = True

# FocalNet parameters
_C.MODEL.IMAGE_ENCODER.FOCAL = CN()
_C.MODEL.IMAGE_ENCODER.FOCAL.PATCH_SIZE = 4
_C.MODEL.IMAGE_ENCODER.FOCAL.IN_CHANS = 3
_C.MODEL.IMAGE_ENCODER.FOCAL.EMBED_DIM = 96
_C.MODEL.IMAGE_ENCODER.FOCAL.DEPTHS = [2, 2, 6, 2]
_C.MODEL.IMAGE_ENCODER.FOCAL.MLP_RATIO = 4.
_C.MODEL.IMAGE_ENCODER.FOCAL.PATCH_NORM = True
_C.MODEL.IMAGE_ENCODER.FOCAL.FOCAL_LEVELS = [2, 2, 2, 2]
_C.MODEL.IMAGE_ENCODER.FOCAL.FOCAL_WINDOWS = [3, 3, 3, 3]
_C.MODEL.IMAGE_ENCODER.FOCAL.FOCAL_FACTORS = [2, 2, 2, 2]
_C.MODEL.IMAGE_ENCODER.FOCAL.USE_CONV_EMBED = False
_C.MODEL.IMAGE_ENCODER.FOCAL.USE_LAYERSCALE = False
_C.MODEL.IMAGE_ENCODER.FOCAL.USE_POSTLN = False

# -----------------------------------------------------------------------------
# Build Text Encoder
# -----------------------------------------------------------------------------
_C.MODEL.TEXT_ENCODER = CN()

_C.MODEL.TEXT_ENCODER.NAME = 'transformer'
_C.MODEL.TEXT_ENCODER.LOAD_PRETRAINED = False
_C.MODEL.TEXT_ENCODER.PRETRAINED = ''
_C.MODEL.TEXT_ENCODER.TOKENIZER = 'clip'
_C.MODEL.TEXT_ENCODER.CONTEXT_LENGTH = 77
_C.MODEL.TEXT_ENCODER.WIDTH = 1024
_C.MODEL.TEXT_ENCODER.HEADS = 16
_C.MODEL.TEXT_ENCODER.LAYERS = 12
_C.MODEL.TEXT_ENCODER.AUTOGRESSIVE = True

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 32
_C.TRAIN.WARMUP_EPOCHS = 5
_C.TRAIN.WEIGHT_DECAY = 0.1
_C.TRAIN.BASE_LR = 5e-4
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 5.0
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = True
# Gradient accumulation steps
# could be overwritten by command line argument
_C.TRAIN.ACCUMULATION_STEPS = 0
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
_C.TRAIN.USE_CHECKPOINT = False

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
# Color jitter factor
_C.AUG.COLOR_JITTER = 0.4
# Use AutoAugment policy. "v0" or "original"
_C.AUG.AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1'
# Random erase prob
_C.AUG.REPROB = 0.25
# Random erase mode
_C.AUG.REMODE = 'pixel'
# Random erase count
_C.AUG.RECOUNT = 1
# Mixup alpha, mixup enabled if > 0
_C.AUG.MIXUP = 0.8
# Cutmix alpha, cutmix enabled if > 0
_C.AUG.CUTMIX = 1.0
# Cutmix min/max ratio, overrides alpha and enables cutmix if set
_C.AUG.CUTMIX_MINMAX = None
# Probability of performing mixup or cutmix when either/both is enabled
_C.AUG.MIXUP_PROB = 1.0
# Probability of switching to cutmix when both mixup and cutmix enabled
_C.AUG.MIXUP_SWITCH_PROB = 0.5
# How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
_C.AUG.MIXUP_MODE = 'batch'

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Whether to use center crop when testing
_C.TEST.CROP = True

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# Mixed precision opt level, if O0, no amp is used ('O0', 'O1', 'O2')
# overwritten by command line argument
_C.AMP_OPT_LEVEL = ''
# Path to output folder, overwritten by command line argument
_C.OUTPUT = ''
# Tag of experiment, overwritten by command line argument
_C.TAG = 'default'
# Frequency to save checkpoint
_C.SAVE_FREQ = 1
# Frequency to logging info
_C.PRINT_FREQ = 100
# Fixed random seed
_C.SEED = 0
# Perform evaluation only, overwritten by command line argument
_C.EVAL_MODE = False
# Test throughput only, overwritten by command line argument
_C.THROUGHPUT_MODE = False
# Debug only so that skip dataloader initialization, overwritten by command line argument
_C.DEBUG_MODE = False
# local rank for DistributedDataParallel, given by command line argument
_C.LOCAL_RANK = 0


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args)

    config.defrost()
    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config
