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
    PosMLP,
    PosGaussianAttention,
    PosPairGaussianAttention
)


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
        num_queries: int,
        num_support_queries: int,
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
        mem_attn_mask: bool,
        clip_embedding_dim: int,
        num_transformer_out_features: int,
        use_nel_loss: bool,
        use_one2many_head: bool
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            num_support_queries: number of support queries
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
        if self.box_reg_type == "mlp":
            self._bbox_embed = BBoxMLPRegression(hidden_dim)
        elif self.box_reg_type in ['bitmask', 'mask2box']:
            self.mask2box_threshold = 0.0
            self._bbox_embed = BboxMaskInitialization(
                fast_bbox = self.box_reg_type=="mask2box", 
                threshold=self.mask2box_threshold
            )
        elif self.box_reg_type == "stn":
            self._bbox_embed = BboxMaskSTN(pooling="mean", learn_format="cxcywh")
        else:
            self._bbox_embed = None
        
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
                PosPairGaussianAttention(
                    dim=hidden_dim,
                    n_heads=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                    learned_scale=("learned_scale" in self.cross_attn_type),
                    normalize=("normalize" in self.cross_attn_type),
                    only_gaussian_logits=("only_gaussian_logits" in self.cross_attn_type),
                    forced_multiscale=("forced_multiscale" in self.cross_attn_type),
                )
                if "pair_gaussian" in self.cross_attn_type else
                PosGaussianAttention(
                    dim=hidden_dim,
                    n_heads=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                    learned_scale=("learned_scale" in self.cross_attn_type),
                    normalize=("normalize" in self.cross_attn_type),
                    only_gaussian_logits=("only_gaussian_logits" in self.cross_attn_type),
                )
                if "gaussian" in self.cross_attn_type else
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

        self.num_queries = num_queries
        self.num_support_queries = num_support_queries
        total_queries = self.num_queries + self.num_support_queries
        # learnable query features
        self.query_feat = nn.Embedding(total_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(total_queries, hidden_dim)
        # optional fixed boxes
        self.query_bbox = nn.Embedding(total_queries, 4) if self._bbox_embed != None else None

        # level embedding (we always use 3 scales)
        self.num_feature_levels = num_transformer_out_features
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        self.mask_embed_type = mask_embed_type
        if self.mask_embed_type == "mlp":
            self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
        elif self.mask_embed_type == "linear":
            self.mask_embed = nn.Linear(hidden_dim, mask_dim)
            weight_init.c2_xavier_fill(self.mask_embed)
        else:
            raise ValueError(f"Unknown mask_embed_type: {self.mask_embed_type}")

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
        
        self.use_one2many_head = use_one2many_head
        if self.use_one2many_head:
            self.round_pred_embed = MLP(hidden_dim, hidden_dim, 1, 3)
        else:
            self.round_pred_embed = None
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        if use_nel_loss:
            self.logit_bias = nn.Parameter(torch.ones([]) * -10.0) # Sig-LIP initialization
        else:
            self.logit_bias = None

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
        self.mem_attn_mask = mem_attn_mask
        if self.mem_attn_mask:
            self.intermediate_mem_cross_attention_layer = CrossAttentionLayer(
                d_model=hidden_dim,
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
        ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        ret["num_support_queries"] = cfg.MODEL.MASK_FORMER.NUM_SUPPORT_QUERIES
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
        ret["mem_attn_mask"] = cfg.MODEL.ZEG_FC.MEM_ATTN_MASK
        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        ret["clip_embedding_dim"] = cfg.MODEL.FC_CLIP.EMBED_DIM
        ret["mask_embed_type"] = cfg.MODEL.ZEG_FC.MASK_EMBED_TYPE
        ret["class_embed_type"] = cfg.MODEL.ZEG_FC.CLASS_EMBED_TYPE
        ret["attn_conv_kernel_size"] = cfg.MODEL.ZEG_FC.ATTN_CONV_KERNEL_SIZE
        ret["box_reg_type"] = cfg.MODEL.ZEG_FC.BOX_REGRESSION_TYPE
        ret["cross_attn_type"] = cfg.MODEL.ZEG_FC.CROSS_ATTN_TYPE
        ret["self_attn_type"] = cfg.MODEL.ZEG_FC.SELF_ATTN_TYPE
        ret["mask_pos_mlp_type"] = cfg.MODEL.ZEG_FC.MASK_POS_MLP_TYPE
        ret["num_transformer_out_features"] = cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_OUT_FEATURES
        ret["use_nel_loss"] = cfg.MODEL.FC_CLIP.USE_NEL_COST
        ret["use_one2many_head"] = cfg.MODEL.ZEG_FC.USE_ONE2MANY_HEAD
        return ret

    def forward(self, x, mask_features, mask = None, text_classifier=None, num_templates=None):
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

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
        query_bbox_unsigmoid = self.query_bbox.weight.unsqueeze(1).repeat(1, bs, 1) if self.query_bbox else None

        predictions_class = []
        predictions_mask = []
        predictions_bbox = []
        predictions_round = []

        # Optional starting cross-attention for text. 
        if self.text_atnn_cls:
            output, text_attn_logits = self.intermediate_text_cross_attention_layer(
                output, text_classifier,
                query_pos=query_embed,
                return_attn_logits=self.text_atnn_cls
            )
        else:
            text_attn_logits = None
        
        # Optional starting cross-attention for memory attention mask.
        if self.mem_attn_mask:
            output, mem_attn_logits = self.intermediate_mem_cross_attention_layer(
                output.transpose(0,1), # (B,Q,C)
                src[0].transpose(0,1).view(bs, size_list[0][0], size_list[0][1], -1), # (B,H,W,C)
                memory_pos_emb=pos[0].transpose(0,1).view(bs, size_list[0][0], size_list[0][1], -1), # (B,H,W,C), 
                query_pos_emb=query_embed.transpose(0,1), # (B,Q,C)
                pos=query_bbox_unsigmoid.sigmoid().transpose(0,1) if query_bbox_unsigmoid is not None else None, # (B,Q,[x,y,w,j])
                return_attn_logits=self.mem_attn_mask,
            )
            output = output.transpose(0,1) # (Q,B,C)
        else:
            mem_attn_logits = None

        # prediction heads on learnable query features
        main_output = output[:self.num_queries]
        main_query_bbox_unsigmoid = query_bbox_unsigmoid[:self.num_queries] if query_bbox_unsigmoid is not None else None
        main_mem_attn_logits = mem_attn_logits[:, :self.num_queries] if mem_attn_logits is not None else None
        main_text_attn_logits = text_attn_logits[:, :self.num_queries] if text_attn_logits is not None else None

        outputs_class, outputs_mask, output_box, outputs_round, updated_main_query_bbox_unsigmoid, main_attn_mask = self.forward_prediction_heads(
            main_output,
            mask_features,
            main_mem_attn_logits,
            main_text_attn_logits,
            query_bbox_unsigmoid=main_query_bbox_unsigmoid,
            attn_mask_size=size_list[0],
            attn_mask_target_size=size_list[0],
            text_classifier=text_classifier,
            num_templates=num_templates,
        )
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)
        predictions_bbox.append(output_box)
        predictions_round.append(outputs_round)
        
        if self.num_support_queries > 0:
            if query_bbox_unsigmoid is not None:
                support_query_bbox_unsigmoid = query_bbox_unsigmoid[self.num_queries:]
                query_bbox_unsigmoid = torch.cat([updated_main_query_bbox_unsigmoid, support_query_bbox_unsigmoid], dim=0)
            
            b_h, _, hw = main_attn_mask.shape
            support_attn_mask = torch.zeros(
                b_h, self.num_support_queries, hw, dtype=torch.bool, device=output.device
            )
            attn_mask = torch.cat([main_attn_mask, support_attn_mask], dim=1)
        else:
            query_bbox_unsigmoid = updated_main_query_bbox_unsigmoid
            attn_mask = main_attn_mask

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            
            total_queries = self.num_queries + self.num_support_queries
            # attention: cross-attention first
            output, mem_attn_logits = self.transformer_cross_attention_layers[i](
                output.transpose(0,1), # (B,Q,C)
                src[level_index].transpose(0,1).view(bs, size_list[level_index][0], size_list[level_index][1], -1), # (B,H,W,C)
                attn_mask=attn_mask.view(bs, self.num_heads, total_queries, size_list[level_index][0], size_list[level_index][1]), # (B, num_heads, Q, H,W)
                memory_pos_emb=pos[level_index].transpose(0,1).view(bs, size_list[level_index][0], size_list[level_index][1], -1), # (B,H,W,C), 
                query_pos_emb=query_embed.transpose(0,1), # (B,Q,C)
                pos=query_bbox_unsigmoid.sigmoid().transpose(0,1) if query_bbox_unsigmoid is not None else None, # (B,Q,[x,y,w,j])
                return_attn_logits=self.mem_attn_mask,
            )
            output = output.transpose(0,1) # (Q,B,C)

            # then, text cross-attention
            if self.text_attn:
                output, text_attn_logits = self.transformer_text_cross_attention_layers[i](
                    output, text_classifier,
                    query_pos=query_embed,
                    return_attn_logits=self.text_atnn_cls
                )
            else:
                text_attn_logits = None
            # then, self-attention
            output, _ = self.transformer_self_attention_layers[i](
                output.transpose(0,1), # (B,Q,C) 
                pos_emb=query_embed.transpose(0,1), # # (B,Q,C) 
                pos=query_bbox_unsigmoid.sigmoid().transpose(0,1) if query_bbox_unsigmoid is not None else None, # (B,Q,[x,y,w,j])
            )
            output = output.transpose(0,1) # (Q,B,C)
            
            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            main_output = output[:self.num_queries]
            main_query_bbox_unsigmoid = query_bbox_unsigmoid[:self.num_queries] if query_bbox_unsigmoid is not None else None
            main_mem_attn_logits = mem_attn_logits[:, :self.num_queries] if mem_attn_logits is not None else None
            main_text_attn_logits = text_attn_logits[:, :self.num_queries] if text_attn_logits is not None else None

            outputs_class, outputs_mask, output_box, outputs_round, updated_main_query_bbox_unsigmoid, main_attn_mask = self.forward_prediction_heads(
                main_output,
                mask_features,
                main_mem_attn_logits,
                main_text_attn_logits,
                query_bbox_unsigmoid=main_query_bbox_unsigmoid,
                attn_mask_size=size_list[i % self.num_feature_levels],
                attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels],
                text_classifier=text_classifier,
                num_templates=num_templates,
            )
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
            predictions_bbox.append(output_box)
            predictions_round.append(outputs_round)

            if self.num_support_queries > 0:
                if query_bbox_unsigmoid is not None:
                    support_query_bbox_unsigmoid = query_bbox_unsigmoid[self.num_queries:]
                    query_bbox_unsigmoid = torch.cat([updated_main_query_bbox_unsigmoid, support_query_bbox_unsigmoid], dim=0)
            
                b_h, _, hw = main_attn_mask.shape
                support_attn_mask = torch.zeros(
                    b_h, self.num_support_queries, hw, dtype=torch.bool, device=output.device
                )
                attn_mask = torch.cat([main_attn_mask, support_attn_mask], dim=1)
            else:
                query_bbox_unsigmoid = updated_main_query_bbox_unsigmoid
                attn_mask = main_attn_mask


        assert len(predictions_class) == self.num_layers + 1

        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'pred_boxes': predictions_bbox[-1],
            'pred_round': predictions_round[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, 
                predictions_mask, 
                predictions_bbox,
                predictions_round
            )
        }
        return out

    @torch.compiler.disable(recursive=False)
    def forward_prediction_heads(
        self, 
        output, 
        mask_features, 
        mem_attn_logits,
        text_attn_logits, 
        query_bbox_unsigmoid,
        attn_mask_size,
        attn_mask_target_size, 
        text_classifier, 
        num_templates
    ):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)

        mask_embed = self.mask_embed(decoder_output)

        if self.use_one2many_head:
            outputs_round = self.round_pred_embed(decoder_output)
        else:
            outputs_round = None

        if self.mem_attn_mask:
            # Get attention logits from memory attention to the propper size.
            mem_attn_logits = mem_attn_logits.unflatten(-1, attn_mask_size) # [B, Q, H, W]
            mem_attn_logits = F.interpolate(
                mem_attn_logits, 
                size=mask_features.shape[-2:], 
                mode="bilinear", 
                align_corners=False
            )
            outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features) + mem_attn_logits
        else:
            outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        # Apply convolution MLP to the mask features.
        if self.attn_conv_kernel_size:
            outputs_mask = outputs_mask.flatten(0,1).unsqueeze(1)  # [B*Q, 1, H, W]
            outputs_mask = outputs_mask + self.attn_conv_layer(outputs_mask)  # [B*Q, 1, H, W]
            bs = decoder_output.shape[0]
            outputs_mask = outputs_mask.squeeze(1).unflatten(0, (bs, -1))  # [B, Q, H, W]

        # Do box regression if provided with look forward twice
        if query_bbox_unsigmoid != None:
            outputs_bbox, query_bbox_unsigmoid_detached = self._bbox_embed(
                decoder_output, 
                query_bbox_unsigmoid.transpose(0,1), 
                outputs_mask, 
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
        outputs_class = get_classification_logits(
            class_embed, 
            text_classifier, 
            self.logit_scale, 
            self.logit_bias, 
            num_templates, 
            text_attn_logits
        )

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, outputs_bbox, outputs_round, query_bbox_unsigmoid_detached, attn_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks, outputs_bboxes, outputs_round):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b, "pred_boxes": c, "pred_round": d}
                for a, b, c, d in zip(outputs_class[:-1], outputs_seg_masks[:-1], outputs_bboxes[:-1], outputs_round[:-1])
            ]
        else:
            return [
                {"pred_masks": b, "pred_boxes": c, "pred_round": d}
                for b, c, d in zip(outputs_seg_masks[:-1], outputs_bboxes[:-1], outputs_round[:-1])
            ]