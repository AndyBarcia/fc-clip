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
        thing_stuff_adapter: str = "linear",
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
        self.thing_stuff_adapter = thing_stuff_adapter
        self.clip_embedding_dim = clip_embedding_dim
        if use_rd:
            if thing_stuff_adapter == "linear":
                self.thing_class_embed = nn.Linear(self.clip_embedding_dim*2, self.clip_embedding_dim)
                self.stuff_class_embed = nn.Linear(self.clip_embedding_dim*2, self.clip_embedding_dim)
            else:
                self.class_embed = nn.Linear(self.clip_embedding_dim*2, self.clip_embedding_dim)
        elif thing_stuff_adapter == "linear":
            self.thing_class_embed = nn.Linear(self.clip_embedding_dim, self.clip_embedding_dim)
            self.stuff_class_embed = nn.Linear(self.clip_embedding_dim, self.clip_embedding_dim)
    
        if thing_stuff_adapter == "bias":
            self.thing_bias = nn.Parameter(torch.zeros(self.clip_embedding_dim))
            self.stuff_bias = nn.Parameter(torch.zeros(self.clip_embedding_dim))

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
            "thing_stuff_adapter": cfg.MODEL.ZEG_FC.THING_STUFF_ADAPTER_TYPE,
        }

    def get_relationship_descriptor(
        self,
        text: torch.Tensor,   # (T,C) or (B,T,C) 
        img: torch.Tensor,    # (B,C)
        thing_mask: torch.Tensor,  # (T',) or (B,T')
        num_templates: list # [T',]
    ) -> torch.Tensor:
        B = img.shape[0]

        # Expand thing mask for templates.
        num_templates = torch.tensor(num_templates, dtype=torch.long, device=thing_mask.device)  # (T',)
        thing_mask = torch.repeat_interleave(thing_mask, num_templates, dim=-1) # (T) or (B,T)
        stuff_mask = ~thing_mask  # (T) or (B,T)

        # Normalize text to (B,T,C) upfront to handle both Bias and RD logic uniformly
        if text.dim() == 2:  # (T,C)
            text_b = text.unsqueeze(0).expand(B, -1, -1)  # (B,T,C)
        else:
            B_t, T, C = text.shape
            assert B_t == B, "Batch size of text and img must match"
            text_b = text

        # Add bias if needed
        # Modified to handle all-thing or all-stuff cases (DDP fix)
        if self.thing_stuff_adapter == "bias":
            text_bias = text_b.clone()
            
            # Ensure masks are (B,T) for consistent indexing
            if thing_mask.dim() == 1:
                tm = thing_mask.unsqueeze(0).expand(B, -1)
                sm = stuff_mask.unsqueeze(0).expand(B, -1)
            else:
                tm, sm = thing_mask, stuff_mask

            if tm.any():
                text_bias[tm] = text_b[tm] + self.thing_bias
            else:
                # Add 0 * param to graph if unused
                text_bias = text_bias + 0.0 * self.thing_bias.sum()

            if sm.any():
                text_bias[sm] = text_b[sm] + self.stuff_bias
            else:
                # Add 0 * param to graph if unused
                text_bias = text_bias + 0.0 * self.stuff_bias.sum()
        else:
            text_bias = text_b

        # Build base descriptor rd
        if self.use_rd:
            if text.dim() == 2:
                # (T,C) × (B,C) → (B,T,C)
                rd = einsum(text, img, "t c, b c -> b t c")
                rd = torch.cat((rd, text_bias), dim=-1)  # (B,T,2C)
            else:
                # (B,T,C) × (B,C) → (B,T,C)
                rd = einsum(text, img, "b t c, b c -> b t c")
                rd = torch.cat((rd, text_bias), dim=-1)  # (B,T,2C)
        else:
            rd = text_bias  # (B,T,C)

        # Apply linear adapters
        # Modified to handle all-thing or all-stuff cases (DDP fix)
        if self.thing_stuff_adapter == "linear":
            T = rd.shape[1]
            
            # Normalize thing_mask to (B,T) bool
            if thing_mask.dim() == 1:
                tm = thing_mask.bool().unsqueeze(0).expand(B, -1)  # (B,T)
                sm = stuff_mask.bool().unsqueeze(0).expand(B, -1)  # (B,T)
            else:
                tm = thing_mask.bool()
                sm = stuff_mask.bool()

            # Per-type linear heads
            rd_out = torch.empty(B, T, self.clip_embedding_dim, device=rd.device, dtype=rd.dtype)
            
            if tm.any():
                rd_out[tm] = self.thing_class_embed(rd[tm]).to(rd.dtype)
            else:
                # Fake computation for DDP
                rd_out = rd_out + 0.0 * self.thing_class_embed.weight.sum()
                if self.thing_class_embed.bias is not None:
                    rd_out = rd_out + 0.0 * self.thing_class_embed.bias.sum()

            if sm.any():
                rd_out[sm] = self.stuff_class_embed(rd[sm]).to(rd.dtype)
            else:
                # Fake computation for DDP
                rd_out = rd_out + 0.0 * self.stuff_class_embed.weight.sum()
                if self.stuff_class_embed.bias is not None:
                    rd_out = rd_out + 0.0 * self.stuff_class_embed.bias.sum()

        elif self.use_rd:
            # Shared linear head
            rd_out = self.class_embed(rd)
        else:
            # No linear adapters
            rd_out = rd

        return rd_out.float()  # (B,T,C)

    def forward(self, features, mask_to_clip_logits_fn=None, logits_to_clip_mask_fn=None, mask=None):
        # Deformable-attention encoder.
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(features)

        # Semantically enrich text embeddings with relationship descriptor
        text_classifier = self.get_relationship_descriptor(
            features["text_classifier"],
            features["clip_embedding"],
            features['thing_mask'],
            features['num_templates']
        )

        # FC-CLIP decoder.
        if self.transformer_in_feature == "multi_scale_pixel_decoder":
            predictions = self.predictor(
                multi_scale_features, 
                mask_features, 
                mask,
                text_classifier=text_classifier, 
                thing_mask=features['thing_mask'],
                num_templates=features["num_templates"],
                mask_to_clip_logits_fn=mask_to_clip_logits_fn,
                logits_to_clip_mask_fn=logits_to_clip_mask_fn,
                clip_dense_features=features.get("clip_dense_embedding"),
            )
        else:
            raise NotImplementedError
        return predictions
