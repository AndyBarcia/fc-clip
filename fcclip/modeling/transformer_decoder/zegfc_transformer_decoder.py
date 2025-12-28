"""
This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 

Reference: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/transformer_decoder/mask2former_transformer_decoder.py
"""

import math
import logging
import fvcore.nn.weight_init as weight_init
from typing import Optional, Tuple, Union
import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d
from detectron2.utils.registry import Registry

from .position_encoding import PositionEmbeddingSine
from .fcclip_transformer_decoder import (
    TRANSFORMER_DECODER_REGISTRY,
    get_classification_logits,
    MaskPooling,
    SelfAttentionLayer,
    CrossAttentionLayer,
    SlotCrossAttention,
    FFNLayer,
    MLP,
    _get_activation_fn
)
from .box_regression import (
    BboxMaskInitialization, 
    BboxMaskSTN, 
    BBoxMLPRegression
)
from .pos_mlp_bias.functions import (
    PosMLPAttention,
    PosMLPSelfAttention,
    PosMLP
)


def _find_multiple(n: int, k: int) -> int:
    return int(math.ceil(n / k) * k)


class CastedLinear(nn.Linear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x.to(self.weight.dtype), self.weight, self.bias)
        return out.to(x.dtype) if out.dtype != x.dtype else out


class Conv2dSwiGLU(nn.Module):
    """
    NHWC variant of ConvSwiGLU:
      x: (B, H, W, C)
      out: (B, H, W, C)

    Uses:
      gate_up_proj: Linear(C -> 2*inter)
      SwiGLU: silu(gate) * up
      depthwise Conv2d over spatial dims (H, W) in expanded space (inter channels)
      down_proj: Linear(inter -> C)
    """
    def __init__(
        self,
        hidden_size: int,
        expansion: float,
        conv_kernel: Union[int, Tuple[int, int]] = 2,
        intermediate_size: Optional[int] = None,
        conv_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()

        inter = (
            intermediate_size
            if intermediate_size is not None
            else _find_multiple(round(expansion * hidden_size * 2 / 3), 256)
        )
        self.inter = inter

        # Produces (gate, up) packed along the last dim
        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)

        # Depthwise Conv2d expects NCHW, so we will permute in forward
        if isinstance(conv_kernel, int):
            kH, kW = conv_kernel, conv_kernel
        else:
            kH, kW = conv_kernel

        pH, pW = kH // 2, kW // 2

        self.dwconv = nn.Conv2d(
            in_channels=inter,
            out_channels=inter,
            kernel_size=(kH, kW),
            padding=(pH, pW),
            groups=inter,   # depthwise
            bias=True,
        ).to(dtype=conv_dtype)

        self.act = nn.SiLU()
        self.down_proj = CastedLinear(inter, hidden_size, bias=False)

    def forward(self, x: torch.Tensor, timer: Optional[object] = None, prefix: str = ""):
        """
        x: (B, H, W, C)
        """
        B, H, W, C = x.shape

        # (B, H, W, 2*inter) -> two tensors (B, H, W, inter)
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)

        # SwiGLU
        x_ffn = F.silu(gate) * up  # (B, H, W, inter)

        # Conv2d expects (B, inter, H, W)
        x_nchw = x_ffn.permute(0, 3, 1, 2).to(self.dwconv.weight.dtype)  # (B, inter, H, W)
        x_conv = self.dwconv(x_nchw)  # (B, inter, H_out, W_out)

        # If kernel is even, padding=(k//2) yields H_out = H+1 (same for W).
        # Slice back to original spatial size.
        x_conv = x_conv[:, :, :H, :W]

        x_conv = self.act(x_conv)

        # Back to NHWC: (B, H, W, inter)
        x_conv = x_conv.permute(0, 2, 3, 1).contiguous()

        # Project back to hidden size: (B, H, W, C)
        x_out = self.down_proj(x_conv)
        return x_out


class Conv2dSwiGLUFFNLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        normalize_before: bool = False,
        query_h: int = 1,
        query_w: int = 1,
        conv_kernel: Union[int, Tuple[int, int]] = 2,
        conv_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.query_h = query_h
        self.query_w = query_w
        self.normalize_before = normalize_before

        self.ffn = Conv2dSwiGLU(
            hidden_size=d_model,
            expansion=dim_feedforward / d_model,
            conv_kernel=conv_kernel,
            intermediate_size=dim_feedforward,
            conv_dtype=conv_dtype,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def _reshape_to_grid(self, tgt: Tensor) -> Tensor:
        num_queries, bs, dim = tgt.shape
        expected = self.query_h * self.query_w
        if num_queries != expected:
            raise ValueError(
                f"Expected {expected} queries to reshape into ({self.query_h}, {self.query_w}) grid, "
                f"but got {num_queries}."
            )
        return tgt.transpose(0, 1).reshape(bs, self.query_h, self.query_w, dim)

    def _reshape_to_sequence(self, tgt: Tensor) -> Tensor:
        bs, h, w, dim = tgt.shape
        return tgt.reshape(bs, h * w, dim).transpose(0, 1)

    def forward_post(self, tgt: Tensor) -> Tensor:
        tgt_grid = self._reshape_to_grid(tgt)
        tgt_ffn = self.ffn(tgt_grid)
        tgt_ffn = self._reshape_to_sequence(tgt_ffn).to(tgt.dtype)
        tgt = tgt + self.dropout(tgt_ffn)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt: Tensor) -> Tensor:
        tgt2 = self.norm(tgt)
        tgt_grid = self._reshape_to_grid(tgt2)
        tgt_ffn = self.ffn(tgt_grid)
        tgt_ffn = self._reshape_to_sequence(tgt_ffn).to(tgt.dtype)
        tgt = tgt + self.dropout(tgt_ffn)
        return tgt

    def forward(self, tgt: Tensor) -> Tensor:
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


class TextCrossAttentionLayer(nn.Module):

    def __init__(
        self, 
        d_model, 
        d_clip: int,
        nhead, 
        dropout=0.0,
        activation="relu", 
        normalize_before=False
    ):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, kdim=d_clip, vdim=d_clip, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def _compute_attn_logits(self, q, k):
        # Project queries and keys
        if self.multihead_attn._qkv_same_embed_dim:
            # Combined projection weights
            w_q, w_k, _ = self.multihead_attn.in_proj_weight.chunk(3, dim=0)
            b_q, b_k, _ = self.multihead_attn.in_proj_bias.chunk(3)
        else:
            # Separate projection weights
            w_q = self.multihead_attn.q_proj_weight
            w_k = self.multihead_attn.k_proj_weight
            b_q = self.multihead_attn.in_proj_bias[:self.multihead_attn.embed_dim]
            b_k = self.multihead_attn.in_proj_bias[self.multihead_attn.embed_dim:2*self.multihead_attn.embed_dim]

        q = F.linear(q, w_q, b_q)
        k = F.linear(k, w_k, b_k)
        
        # Prepare dimensions
        tgt_len, bsz, embed_dim = q.size()
        src_len = k.size(0)
        head_dim = embed_dim // self.multihead_attn.num_heads
        
        # Reshape queries and keys for efficient computation
        q = q.contiguous().view(tgt_len, bsz, self.multihead_attn.num_heads, head_dim)
        k = k.contiguous().view(src_len, bsz, self.multihead_attn.num_heads, head_dim)
        
        # Compute scaled dot-product attention logits
        # Using einsum for efficient computation of averaged attention
        attn_logits = torch.einsum('ibhd,jbhd->bij', q, k) / math.sqrt(head_dim)
        
        # Result is already averaged over heads due to einsum operation
        return attn_logits  # Shape: [bsz, tgt_len, src_len]

    def forward_post(
        self, 
        tgt, 
        text_classification,
        query_pos: Optional[Tensor] = None,
        return_attn_logits: bool = False,
    ):
        q = self.with_pos_embed(tgt, query_pos)
        k = v = text_classification
        
        if return_attn_logits:
            # Compute attention logits (already averaged over heads)
            attn_logits = self._compute_attn_logits(q, k)
            
            # Use multihead_attn for output computation
            tgt2 = self.multihead_attn(
                query=q,
                key=k,
                value=v,
                attn_mask=None,
                key_padding_mask=None
            )[0]
        else:
            tgt2 = self.multihead_attn(
                query=q,
                key=k,
                value=v,
                attn_mask=None,
                key_padding_mask=None
            )[0]
        
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        
        return (tgt, attn_logits) if return_attn_logits else (tgt, None)

    def forward_pre(
        self, 
        tgt, 
        text_classification,
        query_pos: Optional[Tensor] = None,
        return_attn_logits: bool = False
    ):
        tgt2 = self.norm(tgt)
        q = self.with_pos_embed(tgt2, query_pos)
        k = v = text_classification
        
        if return_attn_logits:
            # Compute attention logits (already averaged over heads)
            attn_logits = self._compute_attn_logits(q, k)
            
            # Use multihead_attn for output computation
            tgt2 = self.multihead_attn(
                query=q,
                key=k,
                value=v,
                attn_mask=None,
                key_padding_mask=None
            )[0]
        else:
            tgt2 = self.multihead_attn(
                query=q,
                key=k,
                value=v,
                attn_mask=None,
                key_padding_mask=None
            )[0]
        
        tgt = tgt + self.dropout(tgt2)
        return (tgt, attn_logits) if return_attn_logits else (tgt, None)

    def forward(
        self, 
        tgt, 
        text_classification, # (B,T,C) or (T,C)
        query_pos: Optional[Tensor] = None,
        return_attn_logits: bool = False
    ):
        if len(text_classification.shape) == 2:
            text_classification = text_classification[:,None].expand(-1,tgt.shape[1],-1) # (T,B,C)
        else:
            text_classification = text_classification.transpose(0,1) # (T,B,C)

        if self.normalize_before:
            return self.forward_pre(
                tgt, text_classification, query_pos, 
                return_attn_logits=return_attn_logits
            )
        else:
            return self.forward_post(
                tgt, text_classification, query_pos,
                return_attn_logits=return_attn_logits
            )


@TRANSFORMER_DECODER_REGISTRY.register()
class MultiScaleExtendedMaskedTransformerDecoder(nn.Module):

    @configurable
    def __init__(
        self,
        in_channels,
        mask_classification=True,
        *,
        hidden_dim: int,
        query_h: int,
        query_w: int,
        nheads: int,
        dim_feedforward: int,
        ffn_type: str = "mlp",
        ffn_conv_kernel_size: int = 2,
        ffn_conv_dtype: torch.dtype = torch.bfloat16,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        mask_embed_type: str = "mlp",
        class_embed_type: str = "mlp",
        box_reg_type: str = "none",
        cross_attn_type: str = "standard",
        self_attn_type: str = "standard",
        mask_pos_mlp_type: str = "none",
        enforce_input_project: bool,
        attn_conv_kernel_size: Optional[int] = 3,
        text_attn: bool,
        text_atnn_cls: bool,
        clip_embedding_dim: int,
        separate_thing_stuff_mask_embed: bool = False,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            hidden_dim: Transformer feature dimension
            query_h: number of query rows
            query_w: number of query columns
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        
        # Box regression module
        self.box_reg_type = box_reg_type
        self.query_h = query_h
        self.query_w = query_w
        assert self.query_h > 0 and self.query_w > 0, "query_h and query_w must be positive"
        self.num_queries = self.query_h * self.query_w
        if self.box_reg_type == "mlp":
            self._bbox_embed = BBoxMLPRegression(hidden_dim)
            self.query_bbox = nn.Embedding(self.num_queries, 4) if self._bbox_embed != None else None
        elif self.box_reg_type in ['bitmask', 'mask2box']:
            self.mask2box_threshold = 0.0
            self._bbox_embed = BboxMaskInitialization(
                fast_bbox = self.box_reg_type=="mask2box",
                threshold=self.mask2box_threshold
            )
            self.query_bbox = None
        elif self.box_reg_type == "stn":
            self._bbox_embed = BboxMaskSTN(pooling="mean", learn_format="cxcywh")
            self.query_bbox = None
        else:
            self._bbox_embed = None
            self.query_bbox = None
        
        # Attention type
        self.self_attn_type = self_attn_type
        self.cross_attn_type = cross_attn_type
        self.ffn_type = ffn_type
        self.ffn_conv_kernel_size = ffn_conv_kernel_size
        self.ffn_conv_dtype = ffn_conv_dtype

        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_text_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                ) if self.self_attn_type == "standard" else
                PosMLPSelfAttention(
                    dim=hidden_dim,
                    hidden_dim=16,
                    n_heads=nheads,
                    batched_rpb=(self.self_attn_type == "pos_mlp_brpb"),
                    dropout=0.0,
                    normalize_before=pre_norm
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
                if self.cross_attn_type == "standard" else
                PosMLPAttention(
                    dim=hidden_dim,
                    hidden_dim=16,
                    n_heads=nheads,
                    batched_rpb=(self.cross_attn_type == "pos_mlp_brpb"),
                    dropout=0.0,
                    normalize_before=pre_norm
                )
                if "rpb" in self.cross_attn_type else
                SlotCrossAttention(
                    dim=hidden_dim,
                    n_heads=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
                if "slot"in self.cross_attn_type else None
            )

            if text_attn:
                self.transformer_text_cross_attention_layers.append(
                    TextCrossAttentionLayer(
                        d_model=hidden_dim,
                        d_clip=clip_embedding_dim,
                        nhead=nheads,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )
                )

            if self.ffn_type == "mlp":
                ffn_layer = FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            elif self.ffn_type == "conv2d_swiglu":
                ffn_layer = Conv2dSwiGLUFFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                    query_h=self.query_h,
                    query_w=self.query_w,
                    conv_kernel=self.ffn_conv_kernel_size,
                    conv_dtype=self.ffn_conv_dtype,
                )
            else:
                raise ValueError(f"Unknown ffn_type: {self.ffn_type}")
            self.transformer_ffn_layers.append(ffn_layer)

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        # 2D query grid
        self.query_pos_embed = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
        self.query_feat_proj = Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        weight_init.c2_xavier_fill(self.query_feat_proj)

        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        self.mask_embed_type = mask_embed_type
        self.separate_thing_stuff_mask_embed = separate_thing_stuff_mask_embed
        if self.mask_embed_type == "mlp":
            self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim*2 if separate_thing_stuff_mask_embed else mask_dim, 3)
        elif self.mask_embed_type == "linear":
            self.mask_embed = nn.Linear(hidden_dim, mask_dim*2 if separate_thing_stuff_mask_embed else mask_dim)
            weight_init.c2_xavier_fill(self.mask_embed)
        else:
            raise ValueError(f"Unknown mask_embed_type: {self.mask_embed_type}")

        if self.separate_thing_stuff_mask_embed:
            self.thing_stuff_temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # FC-CLIP
        self.mask_pooling = MaskPooling()
        self._mask_pooling_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim))
        
        self.class_embed_type = class_embed_type
        if self.class_embed_type == "mlp":
            self.class_embed = MLP(hidden_dim, hidden_dim, clip_embedding_dim, 3)
        elif self.class_embed_type == "linear":
            self.class_embed = nn.Linear(hidden_dim, clip_embedding_dim)
            weight_init.c2_xavier_fill(self.class_embed)
        else:
            raise ValueError(f"Unknown class_embed_type: {self.class_embed_type}")
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # ZEG-FC
        self.text_attn = text_attn
        self.text_atnn_cls = text_atnn_cls
        assert not (self.text_atnn_cls and not self.text_attn), "text_atnn_cls requires text_attn to be True"
        if self.text_atnn_cls:
            self.intermediate_text_cross_attention_layer = TextCrossAttentionLayer(
                d_model=hidden_dim,
                d_clip=clip_embedding_dim,
                nhead=nheads,
                dropout=0.0,
                normalize_before=pre_norm,
            )
        
        self.attn_conv_kernel_size = attn_conv_kernel_size
        if self.attn_conv_kernel_size:
            self.attn_conv_layer = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=1, out_channels=1,
                    kernel_size=self.attn_conv_kernel_size, groups=1, padding="same", bias=False
                )
            )
        
        self.mask_pos_mlp_type = mask_pos_mlp_type
        if self.mask_pos_mlp_type != "none":
            self.mask_pos_mlp = PosMLP(hidden_dim, 16, batched=("brpb" in self.mask_pos_mlp_type))
        else:
            self.mask_pos_mlp = None

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = {}
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification
        
        ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        ret["query_h"] = cfg.MODEL.ZEG_FC.QUERY_H
        ret["query_w"] = cfg.MODEL.ZEG_FC.QUERY_W
        # Transformer parameters:
        ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD
        ret["ffn_type"] = cfg.MODEL.ZEG_FC.FFN_TYPE
        ret["ffn_conv_kernel_size"] = cfg.MODEL.ZEG_FC.FFN_CONV_KERNEL_SIZE
        ffn_conv_dtype = cfg.MODEL.ZEG_FC.FFN_CONV_DTYPE
        if isinstance(ffn_conv_dtype, str):
            if not hasattr(torch, ffn_conv_dtype):
                raise ValueError(f"Unsupported dtype string for FFN convolution: {ffn_conv_dtype}")
            ffn_conv_dtype = getattr(torch, ffn_conv_dtype)
        ret["ffn_conv_dtype"] = ffn_conv_dtype

        # NOTE: because we add learnable query features which requires supervision,
        # we add minus 1 to decoder layers to be consistent with our loss
        # implementation: that is, number of auxiliary losses is always
        # equal to number of decoder layers. With learnable query features, the number of
        # auxiliary losses equals number of decoders plus 1.
        assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
        ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
        ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
        ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ
        
        ret["text_attn"] = cfg.MODEL.ZEG_FC.TEXT_ATTN
        ret["text_atnn_cls"] = cfg.MODEL.ZEG_FC.TEXT_ATTN_CLS
        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        ret["clip_embedding_dim"] = cfg.MODEL.FC_CLIP.EMBED_DIM
        ret["mask_embed_type"] = cfg.MODEL.ZEG_FC.MASK_EMBED_TYPE
        ret["class_embed_type"] = cfg.MODEL.ZEG_FC.CLASS_EMBED_TYPE
        ret["attn_conv_kernel_size"] = cfg.MODEL.ZEG_FC.ATTN_CONV_KERNEL_SIZE
        ret["box_reg_type"] = cfg.MODEL.ZEG_FC.BOX_REGRESSION_TYPE
        ret["cross_attn_type"] = cfg.MODEL.ZEG_FC.CROSS_ATTN_TYPE
        ret["self_attn_type"] = cfg.MODEL.ZEG_FC.SELF_ATTN_TYPE
        ret["mask_pos_mlp_type"] = cfg.MODEL.ZEG_FC.MASK_POS_MLP_TYPE
        ret["separate_thing_stuff_mask_embed"] = cfg.MODEL.ZEG_FC.SEPARATE_THING_STUFF_MASK_EMBED
        return ret

    def forward(self, x, mask_features, mask = None, text_classifier=None, thing_mask=None, num_templates=None):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # initialize 2D queries from the lowest-resolution feature map
        # choose the feature level with the smallest spatial size
        areas = [h * w for h, w in size_list]
        low_res_level = areas.index(min(areas))
        low_res_feat = self.input_proj[low_res_level](x[low_res_level])
        query_feat = F.adaptive_avg_pool2d(
            self.query_feat_proj(low_res_feat),
            output_size=(self.query_h, self.query_w),
        )
        query_pos_map = self.query_pos_embed(query_feat, None)
        # maintain 2D spatial structure (Hq, Wq, B, C)
        query_embed = query_pos_map.permute(2, 3, 0, 1)
        output = query_feat.permute(2, 3, 0, 1)
        query_bbox_unsigmoid = self.query_bbox.weight.unsqueeze(1).repeat(1, bs, 1) if self.query_bbox else None

        predictions_class = []
        predictions_mask = []
        predictions_bbox = []

        # Optional starting cross-attention for text. 
        if self.text_atnn_cls:
            output_flat = output.flatten(0, 1)
            query_embed_flat = query_embed.flatten(0, 1)
            output_flat, text_attn_logits = self.intermediate_text_cross_attention_layer(
                output_flat, text_classifier,
                query_pos=query_embed_flat,
                return_attn_logits=self.text_atnn_cls
            )
            output = output_flat.view(self.query_h, self.query_w, bs, -1)
        else:
            text_attn_logits = None

        # prediction heads on learnable query features
        outputs_class, outputs_mask, output_box, query_bbox_unsigmoid, attn_mask = self.forward_prediction_heads(
            output.flatten(0, 1), 
            mask_features,
            text_attn_logits,
            query_bbox_unsigmoid=query_bbox_unsigmoid,
            attn_mask_size=size_list[0],
            attn_mask_target_size=size_list[0],
            text_classifier=text_classifier, 
            thing_mask=thing_mask,
            num_templates=num_templates
        )
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)
        predictions_bbox.append(output_box)
        output = output.flatten(0, 1).view(self.query_h, self.query_w, bs, -1)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            
            # Un-mask queries that are not paying attention to any pixels (Row check)
            query_sees_nothing = attn_mask.all(dim=2) # Shape: [B*Heads, Q]
            attn_mask[query_sees_nothing.unsqueeze(2).expand_as(attn_mask)] = False

            # When using SlotAttention, Un-mask pixels that are not being attended to 
            # by any query (Column check)
            if "slot" in self.cross_attn_type:
                pixel_seen_by_no_one = attn_mask.all(dim=1) # Shape: [B*Heads, HW]            
                attn_mask[pixel_seen_by_no_one.unsqueeze(1).expand_as(attn_mask)] = False

            # attention: cross-attention first
            output_flat = output.flatten(0, 1)
            query_embed_flat = query_embed.flatten(0, 1)
            output_flat, _ = self.transformer_cross_attention_layers[i](
                output_flat.transpose(0,1), # (B,Q,C)
                src[level_index].transpose(0,1).view(bs, size_list[level_index][0], size_list[level_index][1], -1), # (B,H,W,C)
                attn_mask=attn_mask.view(bs, self.num_heads, self.num_queries, size_list[level_index][0], size_list[level_index][1]), # (B, num_heads, Q, H,W)
                memory_pos_emb=pos[level_index].transpose(0,1).view(bs, size_list[level_index][0], size_list[level_index][1], -1), # (B,H,W,C), 
                query_pos_emb=query_embed_flat.transpose(0,1), # (B,Q,C)
                pos=query_bbox_unsigmoid.sigmoid().transpose(0,1) if query_bbox_unsigmoid is not None else None, # (B,Q,[x,y,w,j])
            )
            output_flat = output_flat.transpose(0,1) # (Q,B,C)

            # then, text cross-attention
            if self.text_attn:
                output_flat, text_attn_logits = self.transformer_text_cross_attention_layers[i](
                    output_flat, text_classifier,
                    query_pos=query_embed_flat,
                    return_attn_logits=self.text_atnn_cls
                )
            else:
                text_attn_logits = None
            # then, self-attention
            output_flat, _ = self.transformer_self_attention_layers[i](
                output_flat.transpose(0,1), # (B,Q,C) 
                pos_emb=query_embed_flat.transpose(0,1), # # (B,Q,C) 
                pos=query_bbox_unsigmoid.sigmoid().transpose(0,1) if query_bbox_unsigmoid is not None else None, # (B,Q,[x,y,w,j])
            )
            output_flat = output_flat.transpose(0,1) # (Q,B,C)
            
            # FFN
            output_flat = self.transformer_ffn_layers[i](
                output_flat
            )

            outputs_class, outputs_mask, output_box, query_bbox_unsigmoid, attn_mask = self.forward_prediction_heads(
                output_flat, 
                mask_features, 
                text_attn_logits,
                query_bbox_unsigmoid=query_bbox_unsigmoid,
                attn_mask_size=size_list[i % self.num_feature_levels],
                attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels],
                text_classifier=text_classifier, 
                thing_mask=thing_mask,
                num_templates=num_templates
            )
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
            predictions_bbox.append(output_box)
            output = output_flat.view(self.query_h, self.query_w, bs, -1)

        assert len(predictions_class) == self.num_layers + 1

        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'pred_boxes': predictions_bbox[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask, predictions_bbox
            )
        }
        return out

    @torch.compiler.disable(recursive=False)
    def forward_prediction_heads(
        self, 
        output, 
        mask_features, 
        text_attn_logits, 
        query_bbox_unsigmoid,
        attn_mask_size,
        attn_mask_target_size, 
        text_classifier, 
        thing_mask, # (T,)
        num_templates
    ):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        mask_embed = self.mask_embed(decoder_output) # (B,Q,2*C)

        if self.separate_thing_stuff_mask_embed:
            thing_mask_embed, stuff_mask_embed = mask_embed.chunk(2, dim=-1) # (B,Q,C), (B,Q,C)

            class_embed = self.class_embed(decoder_output) # (B,Q,C)
            class_logits = get_classification_logits(class_embed, text_classifier, self.logit_scale, num_templates, text_attn_logits)[:,:,:-1].detach() # (B,Q,T)
            temp = self.thing_stuff_temperature.exp()
            outputs_class = F.softmax(class_logits / temp, dim=-1) # (B,Q,T)
            output_thing_mask = torch.einsum("bqt,t->bq", outputs_class, thing_mask.to(outputs_class.dtype).to(outputs_class.device))  # (B,Q)

            thing_mask = torch.einsum("bqc,bchw->bqhw", thing_mask_embed, mask_features)
            stuff_mask = torch.einsum("bqc,bchw->bqhw", stuff_mask_embed, mask_features)
            outputs_mask = torch.einsum("bqhw,bq->bqhw", thing_mask, output_thing_mask) + torch.einsum("bqhw,bq->bqhw", stuff_mask, 1-output_thing_mask)  # (B,Q,H,W)
        else:
            outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)  # (B,Q,H,W)

        # Apply convolution MLP to the mask features.
        if self.attn_conv_kernel_size:
            outputs_mask = outputs_mask.flatten(0,1).unsqueeze(1)  # [B*Q, 1, H, W]
            outputs_mask = outputs_mask + self.attn_conv_layer(outputs_mask)  # [B*Q, 1, H, W]
            bs = decoder_output.shape[0]
            outputs_mask = outputs_mask.squeeze(1).unflatten(0, (bs, -1))  # [B, Q, H, W]

        # Do box regression if provided with look forward twice
        if self._bbox_embed != None:
            outputs_bbox, query_bbox_unsigmoid_detached = self._bbox_embed(
                x=decoder_output, 
                reference_points=query_bbox_unsigmoid.transpose(0,1) if query_bbox_unsigmoid is not None else None, 
                masks=outputs_mask, 
                normalized_space=False
            )
            outputs_bbox = outputs_bbox.sigmoid()
            query_bbox_unsigmoid_detached = query_bbox_unsigmoid_detached.transpose(0,1)
        else:
            outputs_bbox, query_bbox_unsigmoid_detached = None, None

        # Apply pos mlp bias
        if self.mask_pos_mlp is not None:            
            outputs_mask = outputs_mask + self.mask_pos_mlp(
                outputs_bbox,
                outputs_mask.shape[-2:],
                decoder_output
            )

        # fcclip head
        maskpool_embeddings = self.mask_pooling(x=mask_features, mask=outputs_mask) # [B, Q, C]
        maskpool_embeddings = self._mask_pooling_proj(maskpool_embeddings)
        class_embed = self.class_embed(maskpool_embeddings + decoder_output)
        outputs_class = get_classification_logits(class_embed, text_classifier, self.logit_scale, num_templates, text_attn_logits)

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, outputs_bbox, query_bbox_unsigmoid_detached, attn_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks, outputs_bboxes):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b, "pred_boxes": c}
                for a, b, c in zip(outputs_class[:-1], outputs_seg_masks[:-1], outputs_bboxes[:-1])
            ]
        else:
            return [
                {"pred_masks": b, "pred_boxes": c}
                for b, c in zip(outputs_seg_masks[:-1], outputs_bboxes[:-1])
            ]
