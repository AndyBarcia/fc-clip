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
                self.thing_class_embed = nn.Linear(self.clip_embedding_dim*2, self.clip_embedding_dim*2)
                self.stuff_class_embed = nn.Linear(self.clip_embedding_dim*2, self.clip_embedding_dim*2)
            else:
                self.class_embed = nn.Linear(self.clip_embedding_dim*2, self.clip_embedding_dim*2)
        elif thing_stuff_adapter == "linear":
            self.thing_class_embed = nn.Linear(self.clip_embedding_dim, self.clip_embedding_dim*2)
            self.stuff_class_embed = nn.Linear(self.clip_embedding_dim, self.clip_embedding_dim*2)
    
        if thing_stuff_adapter == "bias":
            self.thing_bias = nn.Parameter(torch.zeros(self.clip_embedding_dim*2))
            self.stuff_bias = nn.Parameter(torch.zeros(self.clip_embedding_dim*2))

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
        text: torch.Tensor,          # (T,C) or (B,T,C)
        img: torch.Tensor,           # (B,C)
        thing_mask: torch.Tensor,    # (T',) or (B,T')
        num_templates: list          # [T',]
    ) -> torch.Tensor:               # (B,T,2,C)
        B = img.shape[0]

        # Expand thing mask for templates.
        num_templates_t = torch.tensor(
            num_templates, dtype=torch.long, device=thing_mask.device
        )  # (T',)
        thing_mask = torch.repeat_interleave(thing_mask, num_templates_t, dim=-1).bool()
        # Normalize masks to (B,T)
        if thing_mask.dim() == 1:
            tm = thing_mask.unsqueeze(0).expand(B, -1)  # (B,T)
        else:
            assert thing_mask.shape[0] == B, "Batch size of thing_mask and img must match"
            tm = thing_mask.bool()
        sm = ~tm

        # Normalize text to (B,T,C)
        if text.dim() == 2:  # (T,C)
            text_b = text.unsqueeze(0).expand(B, -1, -1)  # (B,T,C)
        else:
            assert text.shape[0] == B, "Batch size of text and img must match"
            text_b = text

        _, T, C = text_b.shape
        assert C == self.clip_embedding_dim, "Unexpected embedding dim"

        # Build base feature tensor BEFORE adapters:
        #   - if use_rd: (B,T,2C) = [rd_img | text]
        #   - else:      (B,T,C)  = text
        if self.use_rd:
            # (B,T,C) × (B,C) -> (B,T,C)
            rd_img = einsum(text_b, img, "b t c, b c -> b t c")
            feat = torch.cat((rd_img, text_b), dim=-1)  # (B,T,2C)
        else:
            feat = text_b  # (B,T,C)

        # Bias adapter
        if self.thing_stuff_adapter == "bias":
            # Ensure feat is (B,T,2C) so we can add 2C bias
            if feat.shape[-1] == self.clip_embedding_dim:
                feat = torch.cat((feat, feat), dim=-1)  # (B,T,2C)

            feat_bias = feat.clone()

            # Modified to handle all-thing or all-stuff cases (DDP fix)
            if tm.any():
                feat_bias[tm] = feat[tm] + self.thing_bias
            else:
                feat_bias = feat_bias + 0.0 * self.thing_bias.sum()

            if sm.any():
                feat_bias[sm] = feat[sm] + self.stuff_bias
            else:
                feat_bias = feat_bias + 0.0 * self.stuff_bias.sum()

            feat = feat_bias

        # Linear adapters
        if self.thing_stuff_adapter == "linear":
            out_dim = self.clip_embedding_dim * 2  # heads output 2C
            out = torch.empty(B, T, out_dim, device=feat.device, dtype=feat.dtype)

            # Modified to handle all-thing or all-stuff cases (DDP fix)
            if tm.any():
                out[tm] = self.thing_class_embed(feat[tm]).to(feat.dtype)
            else:
                out = out + 0.0 * self.thing_class_embed.weight.sum()
                if self.thing_class_embed.bias is not None:
                    out = out + 0.0 * self.thing_class_embed.bias.sum()

            if sm.any():
                out[sm] = self.stuff_class_embed(feat[sm]).to(feat.dtype)
            else:
                out = out + 0.0 * self.stuff_class_embed.weight.sum()
                if self.stuff_class_embed.bias is not None:
                    out = out + 0.0 * self.stuff_class_embed.bias.sum()

        elif self.use_rd:
            # Shared linear head (expects/provides 2C in your modified setup)
            out = self.class_embed(feat)  # (B,T,2C)
        else:
            # No linear adapters
            out = feat  # (B,T,C) or (B,T,2C if bias duplicated)

        if out.shape[-1] == self.clip_embedding_dim * 2:
            out = out.reshape(B, T, 2, self.clip_embedding_dim)
        else:
            raise ValueError(f"Unexpected last dim {out.shape[-1]} (expected 2C).")

        return out.float()  # (B,T,2,C)

    def forward(self, features, mask=None):
        return self.layers(features, mask)

    def layers(self, features, mask=None):
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
                num_templates=features["num_templates"]
            )
        else:
            raise NotImplementedError
        return predictions
