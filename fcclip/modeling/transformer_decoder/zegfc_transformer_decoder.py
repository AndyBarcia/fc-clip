"""
This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 

Reference: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/transformer_decoder/mask2former_transformer_decoder.py
"""

import math
import logging
import fvcore.nn.weight_init as weight_init
from typing import Optional
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
    get_untemplated_classification_logits,
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

def build_attn_mask_maxpool(outputs_mask, attn_mask_target_size, num_heads, thresh=0.5):
    """
    outputs_mask: [B, Q, H, W] (mask logits at higher resolution)
    attn_mask_target_size: (h, w) target spatial size for cross-attn keys
    returns: Bool attn_mask [B*num_heads, Q, h*w], where True = blocked
    """
    B, Q, H, W = outputs_mask.shape
    h, w = attn_mask_target_size

    # 1) convert logits -> probs, then max-pool down to (h, w)
    probs = outputs_mask.sigmoid()                          # [B, Q, H, W]
    x = probs.view(B * Q, 1, H, W)                          # [B*Q, 1, H, W]
    pooled = F.adaptive_max_pool2d(x, output_size=(h, w))   # [B*Q, 1, h, w]
    pooled = pooled.view(B, Q, h, w)                        # [B, Q, h, w]

    # 2) build per-head boolean mask (True = NOT allowed to attend)
    attn_mask = (pooled.flatten(2) < thresh)                # [B, Q, h*w]
    attn_mask = attn_mask.unsqueeze(1).repeat(1, num_heads, 1, 1)  # [B, heads, Q, h*w]
    attn_mask = attn_mask.flatten(0, 1).bool()              # [B*heads, Q, h*w]

    # 3) safety valve (Mask2Former-style): if a query blocks everything, allow all
    all_blocked = attn_mask.sum(-1) == attn_mask.shape[-1]
    attn_mask[all_blocked] = False

    return attn_mask.detach()

class TextCrossAttentionLayer(nn.Module):
    """
    Manual multi-head cross-attention:
      - q: (Q, B, d_model)
      - k/v: (T, B, d_clip)
    Additionally:
      - length-normalize q,k (per head) before dot-product
      - optionally ensemble qk logits with out_of_vocab_logits (from mask->CLIP)
        *inside this module* to produce the attention logits.
    """
    def __init__(
        self,
        d_model: int,
        d_clip: int,
        nhead: int,
        dropout: float = 0.0,
        activation: str = "relu",
        normalize_before: bool = False,
    ):
        super().__init__()
        assert d_model % nhead == 0, f"d_model={d_model} must be divisible by nhead={nhead}"
        self.d_model = d_model
        self.d_clip = d_clip
        self.num_heads = nhead
        self.head_dim = d_model // nhead

        # Projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_clip, d_model)
        self.v_proj = nn.Linear(d_clip, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.normalize_before = normalize_before
        self.activation = _get_activation_fn(activation)

        self.in_vocab_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.register_buffer(
            "ensemble_alpha",
            torch.linspace(0.0, 1.0, steps=nhead).view(1, nhead, 1, 1),
            persistent=False,
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for m in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def _mha_forward(
        self,
        q: Tensor,  # (Q,B,d_model)
        k: Tensor,  # (T,B,d_clip)
        v: Tensor,  # (T,B,d_clip)
        *,
        out_of_vocab_logits: Optional[Tensor] = None,  # (B,Q,C) or (B,Q,T)
        return_attn_logits: bool = False,
    ):
        Q_len, B, _ = q.shape
        T_len = k.shape[0]

        # Project
        q = self.q_proj(q)  # (Q,B,d_model)
        k = self.k_proj(k)  # (T,B,d_model)
        v = self.v_proj(v)  # (T,B,d_model)

        # (B, heads, L, head_dim)
        q = q.permute(1, 0, 2).reshape(B, Q_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.permute(1, 0, 2).reshape(B, T_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.permute(1, 0, 2).reshape(B, T_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Length-normalize queries and keys (per-head)
        q = F.normalize(q, p=2, dim=-1, eps=1e-6)
        k = F.normalize(k, p=2, dim=-1, eps=1e-6)

        # Cosine attention logits: (B, heads, Q, T)
        logit_scale = self.in_vocab_logit_scale.exp().clamp(max=100)
        in_vocab_attn_logits = torch.einsum("bhqd,bhkd->bhqk", q, k) * logit_scale

        # Ensemble inside attention logits (logit-space)
        if out_of_vocab_logits is not None:
            out_of_vocab_logits = out_of_vocab_logits.unsqueeze(1)  # (B,1,Q,T) broadcast over heads
            alpha = self.ensemble_alpha.to(
                device=in_vocab_attn_logits.device, 
                dtype=in_vocab_attn_logits.dtype
            )  # (1,H,1,1)
            attn_logits = in_vocab_attn_logits * (1.0 - alpha) + out_of_vocab_logits * alpha
        else:
            attn_logits = in_vocab_attn_logits

        # Attention weights and output
        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = torch.einsum("bhqk,bhkd->bhqd", attn_weights, v)  # (B,heads,Q,head_dim)

        # Merge heads: (Q,B,d_model)
        out = out.permute(0, 2, 1, 3).reshape(B, Q_len, self.d_model).permute(1, 0, 2)
        out = self.out_proj(out)

        # Return logits averaged over heads
        attn_logits_avg = attn_logits.mean(dim=1) if return_attn_logits else None  # (B,Q,T)

        return out, attn_logits_avg

    def forward_post(
        self,
        tgt: Tensor,
        text_classification: Tensor,
        query_pos: Optional[Tensor] = None,
        *,
        out_of_vocab_logits: Optional[Tensor] = None,
        return_attn_logits: bool = False,
    ):
        q = self.with_pos_embed(tgt, query_pos)
        k = v = text_classification

        tgt2, attn_logits = self._mha_forward(
            q, k, v,
            out_of_vocab_logits=out_of_vocab_logits,
            return_attn_logits=return_attn_logits,
        )

        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return (tgt, attn_logits) if return_attn_logits else (tgt, None)

    def forward_pre(
        self,
        tgt: Tensor,
        text_classification: Tensor,
        query_pos: Optional[Tensor] = None,
        *,
        out_of_vocab_logits: Optional[Tensor] = None,
        return_attn_logits: bool = False,
    ):
        tgt2 = self.norm(tgt)
        q = self.with_pos_embed(tgt2, query_pos)
        k = v = text_classification

        tgt2, attn_logits = self._mha_forward(
            q, k, v,
            out_of_vocab_logits=out_of_vocab_logits,
            return_attn_logits=return_attn_logits,
        )

        tgt = tgt + self.dropout(tgt2)
        return (tgt, attn_logits) if return_attn_logits else (tgt, None)

    def forward(
        self,
        tgt: Tensor,
        text_classification: Tensor,  # (B,T,C) or (T,C)
        query_pos: Optional[Tensor] = None,
        *,
        out_of_vocab_logits: Optional[Tensor] = None,
        return_attn_logits: bool = False,
    ):
        # Make text_classification (T,B,C)
        if len(text_classification.shape) == 2:
            text_classification = text_classification[:, None].expand(-1, tgt.shape[1], -1)
        else:
            text_classification = text_classification.transpose(0, 1)

        if self.normalize_before:
            return self.forward_pre(
                tgt, text_classification, query_pos,
                out_of_vocab_logits=out_of_vocab_logits,
                return_attn_logits=return_attn_logits,
            )
        else:
            return self.forward_post(
                tgt, text_classification, query_pos,
                out_of_vocab_logits=out_of_vocab_logits,
                return_attn_logits=return_attn_logits,
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
        query_init_type: str = "avg_pool",
        query_pos_init_type: str = "learned",
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
        self.query_h = query_h
        self.query_w = query_w
        assert self.query_h > 0 and self.query_w > 0, "query_h and query_w must be positive"
        self.num_queries = self.query_h * self.query_w
        self._bbox_embed = None
        self.query_bbox = None
        
        # Attention type
        self.self_attn_type = self_attn_type
        self.cross_attn_type = cross_attn_type

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

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.query_init_type = query_init_type
        if self.query_init_type not in {"avg_pool", "feature", "learned"}:
            raise ValueError(f"Unknown query_init_type: {self.query_init_type}")
        self.query_pos_init_type = query_pos_init_type
        if self.query_pos_init_type not in {"learned", "sine"}:
            raise ValueError(f"Unknown query_pos_init_type: {self.query_pos_init_type}")
        
        # Learned query features
        if self.query_init_type == "learned":
            self.query_feat = nn.Embedding(self.num_queries, hidden_dim)

        # 2D query grid
        self.query_pos_embed = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
        if self.query_pos_init_type == "learned":
            self.query_embed = nn.Embedding(self.num_queries, hidden_dim)
        else:
            self.query_embed = None
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

        self.obj_head = MLP(hidden_dim, hidden_dim, 1, 3)

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

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = {}
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification
        
        ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        ret["query_h"] = cfg.MODEL.ZEG_FC.QUERY_H
        ret["query_w"] = cfg.MODEL.ZEG_FC.QUERY_W
        ret["query_init_type"] = cfg.MODEL.ZEG_FC.QUERY_INIT_TYPE
        ret["query_pos_init_type"] = cfg.MODEL.ZEG_FC.QUERY_POS_INIT_TYPE
        # Transformer parameters:
        ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD

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

    def forward(
        self, 
        x, 
        mask_features, 
        mask = None, 
        text_classifier=None, 
        thing_mask=None, 
        num_templates=None,
        mask_to_clip_logits_fn=None,
        clip_dense_features=None,
    ):
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

        # initialize 2D queries
        if self.query_init_type == "learned":
            query_feat = self.query_feat.weight.view(self.query_h, self.query_w, -1)
            output = query_feat.permute(0, 1, 2).unsqueeze(2).repeat(1, 1, bs, 1)
            if self.query_pos_init_type == "sine":
                query_pos_map = self.query_pos_embed(query_feat.unsqueeze(2), None)
                query_embed = query_pos_map.permute(0, 1, 2).unsqueeze(2).repeat(1, 1, bs, 1)
        elif self.query_init_type == "avg_pool":
            # choose the feature level with the smallest spatial size
            areas = [h * w for h, w in size_list]
            low_res_level = areas.index(min(areas))
            low_res_feat = self.input_proj[low_res_level](x[low_res_level])
            query_feat = F.adaptive_avg_pool2d(
                self.query_feat_proj(low_res_feat),
                output_size=(self.query_h, self.query_w),
            )
            output = query_feat.permute(2, 3, 0, 1)
            if self.query_pos_init_type == "sine":
                query_pos_map = self.query_pos_embed(query_feat, None)
                query_embed = query_pos_map.permute(2, 3, 0, 1)
        else:
            patch_output = torch.cat(src, dim=0)
            patch_pos = torch.cat(pos, dim=0)
            outputs_class, _, _, _, _, _, _ = self.forward_prediction_heads(
                patch_output,
                mask_features,
                None,
                query_bbox_unsigmoid=None,
                attn_mask_size=size_list[0],
                attn_mask_target_size=size_list[0],
                text_classifier=text_classifier,
                thing_mask=thing_mask,
                num_templates=num_templates,
                mask_to_clip_logits_fn=mask_to_clip_logits_fn,
                clip_dense_features=clip_dense_features,
                prev_outputs_class=None
            )
            patch_scores = outputs_class[...,:-1].max(dim=-1).values
            topk_indices = patch_scores.topk(self.num_queries, dim=1).indices
            patch_output = patch_output.permute(1, 0, 2)
            topk_features = torch.gather(
                patch_output,
                1,
                topk_indices.unsqueeze(-1).expand(-1, -1, patch_output.shape[-1]),
            )
            if self.query_pos_init_type == "sine":
                patch_pos = patch_pos.permute(1, 0, 2)
                topk_pos = torch.gather(
                    patch_pos,
                    1,
                    topk_indices.unsqueeze(-1).expand(-1, -1, patch_pos.shape[-1]),
                )
            output = topk_features.permute(1, 0, 2).view(self.query_h, self.query_w, bs, -1)
            if self.query_pos_init_type == "sine":
                query_embed = topk_pos.permute(1, 0, 2).view(self.query_h, self.query_w, bs, -1)

        if self.query_pos_init_type == "learned":
            query_embed = self.query_embed.weight.view(self.query_h, self.query_w, 1, -1).repeat(1, 1, bs, 1)
        query_bbox_unsigmoid = self.query_bbox.weight.unsqueeze(1).repeat(1, bs, 1) if self.query_bbox else None

        predictions_class = []
        predictions_mask = []
        predictions_bbox = []

        # Optional starting cross-attention for text. 
        if self.text_atnn_cls:
            output_flat = output.flatten(0, 1)
            query_embed_flat = query_embed.flatten(0, 1)
            output_flat, text_attn_logits = self.intermediate_text_cross_attention_layer(
                output_flat, 
                text_classifier,
                query_pos=query_embed_flat,
                out_of_vocab_logits=None,
                return_attn_logits=self.text_atnn_cls
            )
            output = output_flat.view(self.query_h, self.query_w, bs, -1)
        else:
            text_attn_logits = None

        # prediction heads on learnable query features
        outputs_class, out_of_vocab_logits, out_of_vocab_mask_logits, outputs_mask, output_box, query_bbox_unsigmoid, attn_mask = self.forward_prediction_heads(
            output.flatten(0, 1), 
            mask_features,
            text_attn_logits,
            query_bbox_unsigmoid=query_bbox_unsigmoid,
            attn_mask_size=size_list[0],
            attn_mask_target_size=size_list[0],
            text_classifier=text_classifier, 
            thing_mask=thing_mask,
            num_templates=num_templates,
            mask_to_clip_logits_fn=mask_to_clip_logits_fn,
            clip_dense_features=clip_dense_features,
            prev_outputs_class=None
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
                out_of_vocab_attn_logits=out_of_vocab_mask_logits,
                pos=query_bbox_unsigmoid.sigmoid().transpose(0,1) if query_bbox_unsigmoid is not None else None, # (B,Q,[x,y,w,j])
            )
            output_flat = output_flat.transpose(0,1) # (Q,B,C)

            # then, text cross-attention
            if self.text_attn:
                output_flat, text_attn_logits = self.transformer_text_cross_attention_layers[i](
                    output_flat, 
                    text_classifier,
                    query_pos=query_embed_flat,
                    out_of_vocab_logits=out_of_vocab_logits,
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

            outputs_class, out_of_vocab_logits, out_of_vocab_mask_logits, outputs_mask, output_box, query_bbox_unsigmoid, attn_mask = self.forward_prediction_heads(
                output_flat, 
                mask_features,
                text_attn_logits,
                query_bbox_unsigmoid=query_bbox_unsigmoid,
                attn_mask_size=size_list[i % self.num_feature_levels],
                attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels],
                text_classifier=text_classifier, 
                thing_mask=thing_mask,
                num_templates=num_templates,
                mask_to_clip_logits_fn=mask_to_clip_logits_fn,
                clip_dense_features=clip_dense_features,
                prev_outputs_class=predictions_class[-1].detach()
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
        num_templates,
        mask_to_clip_logits_fn,
        clip_dense_features,
        prev_outputs_class=None
    ):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        mask_embed = self.mask_embed(decoder_output) # (B,Q,2*C)

        # Get classification logits from attention patterns.
        vocab_logits = get_untemplated_classification_logits(
            text_attn_logits, 
            num_templates, 
            append_void_class=False
        ) # (B,Q,T)

        # Get mask predictions from mask embeddings.
        if self.separate_thing_stuff_mask_embed:
            thing_mask_embed, stuff_mask_embed = mask_embed.chunk(2, dim=-1) # (B,Q,C), (B,Q,C)

            # Use the class predictions of the previous layer to predict thing vs. stuff.
            # For the very first layer, just use the in-vocab logits to predict thing vs. stuff
            top_class = vocab_logits.argmax(dim=-1)  # (B,Q)  long indices
            if thing_mask.dim() == 1:
                thing_mask = thing_mask[None, :].expand(top_class.shape[0], -1)  # (B,T)
            output_thing_mask = thing_mask.to(top_class.device).gather(1, top_class).to(mask_embed.dtype)  # (B,Q)

            thing_mask = torch.einsum("bqc,bchw->bqhw", thing_mask_embed, mask_features)
            stuff_mask = torch.einsum("bqc,bchw->bqhw", stuff_mask_embed, mask_features)
            outputs_mask = torch.einsum("bqhw,bq->bqhw", thing_mask, output_thing_mask) + torch.einsum("bqhw,bq->bqhw", stuff_mask, 1-output_thing_mask)  # (B,Q,H,W)
        else:
            outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)  # (B,Q,H,W)

        # Get mask-to-clip logits from predicted masks.
        with torch.no_grad():
            out_of_vocab_logits = mask_to_clip_logits_fn(outputs_mask) # (B,Q,T)

        # Append objectness logit to learn to predict no-object.
        obj_logits = self.obj_head(decoder_output).squeeze(-1) # (B,Q)
        outputs_class = torch.cat([vocab_logits, obj_logits.unsqueeze(-1)], dim=-1) # (B,Q,T+1)

        out_of_vocab_mask_logits = None
        if clip_dense_features is not None:
            if clip_dense_features.shape[-2:] != attn_mask_target_size:
                clip_dense_features = F.interpolate(
                    clip_dense_features,
                    size=attn_mask_target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            # Build a query-specific CLIP prototype from class logits and text embeddings.
            class_weights = F.softmax(vocab_logits, dim=-1)
            text_prototypes = text_classifier
            if text_prototypes.dim() == 2:
                text_prototypes = text_prototypes.unsqueeze(0).expand(class_weights.shape[0], -1, -1)
            text_prototypes = F.normalize(text_prototypes, dim=-1)
            clip_query_proto = torch.matmul(class_weights, text_prototypes)
            clip_query_proto = F.normalize(clip_query_proto, dim=-1)
            clip_dense_features = F.normalize(clip_dense_features, dim=1)
            out_of_vocab_mask_logits = torch.einsum("bqc,bchw->bqhw", clip_query_proto, clip_dense_features)

        # The final attention mask for the next layer is built from the predicted masks at the current layer
        attn_mask = build_attn_mask_maxpool(
            outputs_mask,
            attn_mask_target_size,
            self.num_heads,
            thresh=0.5,
        )

        outputs_bbox = None
        query_bbox_unsigmoid_detached = None

        return outputs_class, out_of_vocab_logits, out_of_vocab_mask_logits, outputs_mask, outputs_bbox, query_bbox_unsigmoid_detached, attn_mask

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
