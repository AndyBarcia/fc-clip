# -*- coding: utf-8 -*-
"""
This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 

Reference: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/config.py
"""
from detectron2.config import CfgNode as CN


def add_maskformer2_config(cfg):
    """
    Add config for MASK_FORMER.
    """
    # NOTE: configs from original maskformer
    # data config
    # select the dataset mapper
    cfg.INPUT.DATASET_MAPPER_NAME = "mask_former_semantic"
    # Color augmentation
    cfg.INPUT.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = -1

    # solver config
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    # mask_former model config
    cfg.MODEL.MASK_FORMER = CN()

    # loss
    cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION = True
    cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.MASK_FORMER.CLASS_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.CLASS_FOCAL_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.DICE_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.MASK_WEIGHT = 20.0
    cfg.MODEL.MASK_FORMER.BBOX_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.GIOU_WEIGHT = 1.0

    # transformer config
    cfg.MODEL.MASK_FORMER.NHEADS = 8
    cfg.MODEL.MASK_FORMER.DROPOUT = 0.1
    cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD = 2048
    cfg.MODEL.MASK_FORMER.ENC_LAYERS = 0
    cfg.MODEL.MASK_FORMER.DEC_LAYERS = 6
    cfg.MODEL.MASK_FORMER.PRE_NORM = False

    cfg.MODEL.MASK_FORMER.HIDDEN_DIM = 256
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 100

    cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE = "res5"
    cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ = False

    # mask_former inference config
    cfg.MODEL.MASK_FORMER.TEST = CN()
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False

    # Sometimes `backbone.size_divisibility` is set to 0 for some backbone (e.g. ResNet)
    # you can use this config to override
    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY = 32

    # pixel decoder config
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    # adding transformer in pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 0
    # pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "BasePixelDecoder"

    # swin transformer backbone
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SWIN.USE_CHECKPOINT = False

    # NOTE: maskformer2 extra configs
    # transformer module
    cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME = "MultiScaleMaskedTransformerDecoder"

    # LSJ aug
    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.INPUT.MIN_SCALE = 0.1
    cfg.INPUT.MAX_SCALE = 2.0

    # MSDeformAttn encoder configs
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_OUT_FEATURES = 3
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_POINTS = 4
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_HEADS = 8

    # point loss configs
    # Number of points sampled during training for a mask point head.
    cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS = 112 * 112
    # Oversampling parameter for PointRend point sampling during training. Parameter `k` in the
    # original paper.
    cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO = 3.0
    # Importance sampling parameter for PointRend point sampling during training. Parametr `beta` in
    # the original paper.
    cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO = 0.75

    # Compilation options
    cfg.MODEL.COMPILE = CN()
    cfg.MODEL.COMPILE.ENABLED = False
    cfg.MODEL.COMPILE.TRACE = False
    cfg.MODEL.COMPILE.MODE = "default"

    # Training precission options
    cfg.SOLVER.AMP.PRECISION = "float16"

def add_fcclip_config(cfg):
    # FC-CLIP model config.
    cfg.MODEL.FC_CLIP = CN()
    cfg.MODEL.FC_CLIP.CLIP_MODEL_NAME = "convnext_large_d_320"
    cfg.MODEL.FC_CLIP.CLIP_PRETRAINED_WEIGHTS = "laion2b_s29b_b131k_ft_soup"
    cfg.MODEL.FC_CLIP.EMBED_DIM = 768
    cfg.MODEL.FC_CLIP.GEOMETRIC_ENSEMBLE_ALPHA = 0.4
    cfg.MODEL.FC_CLIP.GEOMETRIC_ENSEMBLE_BETA = 0.8
    cfg.MODEL.FC_CLIP.ENSEMBLE_ON_VALID_MASK = False
    cfg.MODEL.FC_CLIP.USE_NEL_COST = False
    cfg.MODEL.FC_CLIP.FOCAL_ALPHA = 0.8
    cfg.MODEL.FC_CLIP.FOCAL_GAMMA = 2.0
    # DINOv3 backbone config. Note that access to the DINOv3 weights
    # is forbidden without the appropiate policy key in the url. 
    cfg.MODEL.FC_CLIP.DINOV3_TOKENIZER_PATH = "https://dl.fbaipublicfiles.com/dinov3/thirdparty/bpe_simple_vocab_16e6.txt.gz"
    cfg.MODEL.FC_CLIP.DINOV3_BACKBONE_PATH = "https://dinov3.llamameta.net/dinov3_vitl16/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
    cfg.MODEL.FC_CLIP.DINOV3_ADAPTER_PATH = "https://dinov3.llamameta.net/dinov3_vitl16/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth"


def add_zegfc_config(cfg):
    # ZEG-FC model config.
    cfg.MODEL.ZEG_FC = CN()
    cfg.MODEL.ZEG_FC.USE_RELATIONSHIP_DESCRIPTOR = False
    cfg.MODEL.ZEG_FC.TEXT_ATTN = False
    cfg.MODEL.ZEG_FC.TEXT_ATTN_CLS = False
    cfg.MODEL.ZEG_FC.MEM_ATTN_MASK = False
    cfg.MODEL.ZEG_FC.MASK_EMBED_TYPE = "mlp"  # Options: "mlp", "linear"
    cfg.MODEL.ZEG_FC.CLASS_EMBED_TYPE = "mlp"  # Options: "mlp", "linear"
    cfg.MODEL.ZEG_FC.ATTN_CONV_KERNEL_SIZE = None
    cfg.MODEL.ZEG_FC.BOX_REGRESSION_TYPE = None  # Options: "mlp", "bitmask", "mask2box", "stn"
    cfg.MODEL.ZEG_FC.CROSS_ATTN_TYPE = "standard"  # Options: "standard", "pos_mlp_brpb", "pos_mlp_rpb", "gaussian"
    cfg.MODEL.ZEG_FC.SELF_ATTN_TYPE = "standard"  # Options: "standard", "pos_mlp_brpb", "pos_mlp_rpb"
    cfg.MODEL.ZEG_FC.MASK_POS_MLP_TYPE = "none"  # Options: "none", "brpb", "rpb"