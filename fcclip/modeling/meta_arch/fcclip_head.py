"""
This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 

Reference: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/meta_arch/mask_former_head.py
"""

import torch
import logging
from einops import rearrange, einsum
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple, Union

import fvcore.nn.weight_init as weight_init
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from ..transformer_decoder.fcclip_transformer_decoder import build_transformer_decoder
from ..pixel_decoder.msdeformattn import build_pixel_decoder


@SEM_SEG_HEADS_REGISTRY.register()
class FCCLIPHead(nn.Module):

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        clip_embedding_dim: int,
        use_rd: bool,
        num_classes: int,
        pixel_decoder: nn.Module,
        loss_weight: float = 1.0,
        ignore_value: int = -1,
        # extra parameters
        transformer_predictor: nn.Module,
        transformer_in_feature: str,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            num_classes: number of classes to predict
            pixel_decoder: the pixel decoder module
            loss_weight: loss weight
            ignore_value: category id to be ignored during training.
            transformer_predictor: the transformer decoder that makes prediction
            transformer_in_feature: input feature name to the transformer_predictor
        """
        super().__init__()
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]
        feature_strides = [v.stride for k, v in input_shape]
        feature_channels = [v.channels for k, v in input_shape]

        self.ignore_value = ignore_value
        self.common_stride = 4
        self.loss_weight = loss_weight

        self.pixel_decoder = pixel_decoder
        self.predictor = transformer_predictor
        self.transformer_in_feature = transformer_in_feature

        self.num_classes = num_classes

        self.use_rd = use_rd
        self.clip_embedding_dim = clip_embedding_dim
        if use_rd:
            self.class_embed = nn.Linear(self.clip_embedding_dim*2, self.clip_embedding_dim)

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        # figure out in_channels to transformer predictor
        if cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "multi_scale_pixel_decoder":
            transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        else:
            raise NotImplementedError

        return {
            "input_shape": {
                k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
            },
            "clip_embedding_dim": cfg.MODEL.FC_CLIP.EMBED_DIM,
            "use_rd": cfg.MODEL.ZEG_FC.USE_RELATIONSHIP_DESCRIPTOR,
            "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "pixel_decoder": build_pixel_decoder(cfg, input_shape),
            "loss_weight": cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT,
            "transformer_in_feature": cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE,
            "transformer_predictor": build_transformer_decoder(
                cfg,
                transformer_predictor_in_channels,
                mask_classification=True,
            ),
        }

    def get_relationship_descriptor(
        self,
        text: torch.Tensor, # (T,C) or (B,T,C) 
        img: torch.Tensor, # (B,C)
    ) -> torch.Tensor:
        if len(text.shape) == 2:
            rd = einsum(text, img, "t c, b c -> b t c") # (B,T,C)
            rd = torch.concat((rd,text.expand(img.shape[0],-1,-1)), dim=-1) # (B,T,2*C)
        else:
            rd = einsum(text, img, "b t c, b c -> b t c") # (B,T,C)
            rd = torch.concat((rd,text), dim=-1) # (B,T,2*C)
        rd = self.class_embed(rd).float()
        return rd # (B,T,C')

    def forward(self, features, mask=None):
        return self.layers(features, mask)

    def layers(self, features, mask=None):
        # Deformable-attention encoder.
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(features)

        # Semantically enrich text embeddings with relationship descriptor
        if self.use_rd:
            text_classifier = self.get_relationship_descriptor(
                features["text_classifier"],
                features["clip_embedding"]
            )
        else:
            text_classifier = features["text_classifier"]

        # FC-CLIP decoder.
        if self.transformer_in_feature == "multi_scale_pixel_decoder":
            predictions = self.predictor(
                multi_scale_features, 
                mask_features, 
                mask,
                text_classifier=text_classifier, 
                num_templates=features["num_templates"]
            )
        else:
            raise NotImplementedError
        return predictions
