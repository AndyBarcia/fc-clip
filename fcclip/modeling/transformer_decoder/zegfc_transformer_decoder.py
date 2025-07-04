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
    FFNLayer,
    MLP,
    _get_activation_fn
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
        text_classification,
        query_pos: Optional[Tensor] = None,
        return_attn_logits: bool = False
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt, text_classification.transpose(0,1), query_pos, 
                return_attn_logits=return_attn_logits
            )
        else:
            return self.forward_post(
                tgt, text_classification.transpose(0,1), query_pos,
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
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,
        text_atnn_cls: bool,
        clip_embedding_dim: int
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
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
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

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
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

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

        # output FFNs
        # if self.mask_classification:
        #     self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

        # FC-CLIP
        self.mask_pooling = MaskPooling()
        self._mask_pooling_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim))
        self.class_embed = MLP(hidden_dim, hidden_dim, clip_embedding_dim, 3)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # ZEG-FC
        self.text_atnn_cls = text_atnn_cls
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
        ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
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

        ret["text_atnn_cls"] = cfg.MODEL.ZEG_FC.TEXT_ATTN_CLS
        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        ret["clip_embedding_dim"] = cfg.MODEL.FC_CLIP.EMBED_DIM
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

        predictions_class = []
        predictions_mask = []

        # Optional starting cross-attention for text. 
        if self.text_atnn_cls:
            output, text_attn_logits = self.intermediate_text_cross_attention_layer(
                output, text_classifier,
                query_pos=query_embed,
                return_attn_logits=self.text_atnn_cls
            )
        else:
            text_attn_logits = None

        # prediction heads on learnable query features
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
            output, 
            mask_features, 
            text_attn_logits,
            attn_mask_target_size=size_list[0],
            text_classifier=text_classifier, 
            num_templates=num_templates
        )
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )
            # then, text cross-attention
            output, text_attn_logits = self.transformer_text_cross_attention_layers[i](
                output, text_classifier,
                query_pos=query_embed,
                return_attn_logits=self.text_atnn_cls
            )
            # then, self-attention
            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )
            
            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
                output, 
                mask_features, 
                text_attn_logits,
                attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels],
                text_classifier=text_classifier, 
                num_templates=num_templates
            )
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        assert len(predictions_class) == self.num_layers + 1

        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask
            )
        }
        return out

    @torch.compiler.disable(recursive=False)
    def forward_prediction_heads(
        self, 
        output, 
        mask_features, 
        text_attn_logits, 
        attn_mask_target_size, 
        text_classifier, 
        num_templates
    ):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

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

        return outputs_class, outputs_mask, attn_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]
