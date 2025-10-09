"""
This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 

Reference: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/transformer_decoder/mask2former_transformer_decoder.py
"""

import math
import logging
import fvcore.nn.weight_init as weight_init
from typing import Optional, Tuple
import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d
from detectron2.utils.registry import Registry


from .position_encoding import PositionEmbeddingSine


TRANSFORMER_DECODER_REGISTRY = Registry("TRANSFORMER_MODULE")
TRANSFORMER_DECODER_REGISTRY.__doc__ = """
Registry for transformer module in MaskFormer.
"""


def build_transformer_decoder(cfg, in_channels, mask_classification=True):
    """
    Build a instance embedding branch from `cfg.MODEL.INS_EMBED_HEAD.NAME`.
    """
    name = cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME
    return TRANSFORMER_DECODER_REGISTRY.get(name)(cfg, in_channels, mask_classification)


def get_classification_logits(x, text_classifier, logit_scale, logit_bias=None, num_templates=None, text_attn_logits=None):
    # x in shape of [B, *, C]
    # text_classifier: either [num_classes, C] or [B, num_classes, C]
    # text_attn_logits: optional [B, Q, num_classes]
    # logit_scale: scalar
    # num_templates: list of template counts per non-void class
    # Returns: [B, *, num_classes_final] where num_classes_final = len(num_templates) + 1
    
    # Normalize input features
    x = F.normalize(x, dim=-1)
    logit_scale = torch.clamp(logit_scale.exp(), max=100)
    
    # Handle different text_classifier dimensions
    if text_classifier.dim() == 2:
        # Original case: [num_classes, C]
        text_classifier = F.normalize(text_classifier, dim=-1)
        if text_attn_logits is None:
            pred_logits = logit_scale * (x @ text_classifier.T)
        else:
            pred_logits = logit_scale * ((x @ text_classifier.T) + text_attn_logits)
    elif text_classifier.dim() == 3:
        # New case: [B, num_classes, C]
        text_classifier = F.normalize(text_classifier, dim=-1)
        # Batched matrix multiplication: [B, *, C] @ [B, C, num_classes] -> [B, *, num_classes]
        if text_attn_logits is None:
            pred_logits = logit_scale * torch.matmul(x, text_classifier.transpose(-1, -2))
        else:
            pred_logits = logit_scale * (torch.matmul(x, text_classifier.transpose(-1, -2)) + text_attn_logits)
    else:
        raise ValueError(f"text_classifier must be 2D or 3D, got {text_classifier.dim()}D")
    
    if logit_bias is not None:
        pred_logits = pred_logits + logit_bias

    # Max ensembling over templates
    final_pred_logits = []
    cur_idx = 0
    # Process each group of templates for non-void classes
    for num_t in num_templates:
        # Slice current template group and take max
        group_logits = pred_logits[..., cur_idx:cur_idx + num_t]
        final_pred_logits.append(group_logits.max(-1).values)
        cur_idx += num_t
    # Append void class (last element)
    final_pred_logits.append(pred_logits[..., -1])
    # Stack along new class dimension
    final_pred_logits = torch.stack(final_pred_logits, dim=-1)
    
    return final_pred_logits


# Ref: https://github.com/NVlabs/ODISE/blob/e97b06c424c575fec9fc5368dd4b3e050d91abc4/odise/modeling/meta_arch/odise.py#L923
class MaskPooling(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x, mask):
        """
        Args:
            x: [B, C, H, W]
            mask: [B, Q, H, W]
        """
        if not x.shape[-2:] == mask.shape[-2:]:
            # reshape mask to x
            mask = F.interpolate(mask, size=x.shape[-2:], mode='bilinear', align_corners=False)
        with torch.no_grad():
            mask = mask.detach()
            mask = (mask > 0).to(mask.dtype)
            denorm = mask.sum(dim=(-1, -2), keepdim=True) + 1e-8

        mask_pooled_x = torch.einsum(
            "bchw,bqhw->bqc",
            x,
            mask / denorm,
        )
        return mask_pooled_x

class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[torch.Tensor]):
        return tensor if pos is None else tensor + pos

    def _compute_attn_logits(self, q, k):
        # Project queries and keys
        if self.self_attn._qkv_same_embed_dim:
            w_q, w_k, _ = self.self_attn.in_proj_weight.chunk(3, dim=0)
            b_q, b_k, _ = self.self_attn.in_proj_bias.chunk(3)
        else:
            w_q = self.self_attn.q_proj_weight
            w_k = self.self_attn.k_proj_weight
            b_q = self.self_attn.in_proj_bias[:self.self_attn.embed_dim] if self.self_attn.in_proj_bias is not None else None
            b_k = self.self_attn.in_proj_bias[self.self_attn.embed_dim:2*self.self_attn.embed_dim] if self.self_attn.in_proj_bias is not None else None

        q = F.linear(q, w_q, b_q)
        k = F.linear(k, w_k, b_k)

        # Prepare dimensions
        tgt_len, bsz, embed_dim = q.size()
        head_dim = embed_dim // self.self_attn.num_heads
        
        # Reshape queries and keys
        q = q.contiguous().view(tgt_len, bsz, self.self_attn.num_heads, head_dim)
        k = k.contiguous().view(k.size(0), bsz, self.self_attn.num_heads, head_dim)
        
        # Compute scaled dot-product attention logits
        attn_logits = torch.einsum('ibhd,jbhd->bij', q, k) / math.sqrt(head_dim)
        return attn_logits  # Shape: [bsz, tgt_len, src_len]

    def forward_post(self, tgt,
                     tgt_mask: Optional[torch.Tensor] = None,
                     tgt_key_padding_mask: Optional[torch.Tensor] = None,
                     query_pos: Optional[torch.Tensor] = None,
                     return_attn_logits: bool = False):
        q = k = self.with_pos_embed(tgt, query_pos)
        
        attn_logits = None
        if return_attn_logits:
            attn_logits = self._compute_attn_logits(q, k)

        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt, attn_logits

    def forward_pre(self, tgt,
                    tgt_mask: Optional[torch.Tensor] = None,
                    tgt_key_padding_mask: Optional[torch.Tensor] = None,
                    query_pos: Optional[torch.Tensor] = None,
                    return_attn_logits: bool = False):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)

        attn_logits = None
        if return_attn_logits:
            attn_logits = self._compute_attn_logits(q, k)

        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        return tgt, attn_logits

    def forward(self, tgt,
                tgt_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                pos_emb: Optional[torch.Tensor] = None,
                return_attn_logits: bool = False,
                pos: Optional[torch.Tensor] = None):
        tgt = tgt.transpose(0, 1)
        if pos_emb is not None:
            pos_emb = pos_emb.transpose(0, 1)

        if self.normalize_before:
            output, attn_logits = self.forward_pre(tgt, tgt_mask,
                                                   tgt_key_padding_mask, pos_emb, return_attn_logits)
        else:
            output, attn_logits = self.forward_post(tgt, tgt_mask,
                                                    tgt_key_padding_mask, pos_emb, return_attn_logits)

        return output.transpose(0, 1), attn_logits

class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
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
    
    def forward_post(self, tgt, memory,
                     attn_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     return_attn_logits: bool = False):
        q = self.with_pos_embed(tgt, query_pos)
        k = self.with_pos_embed(memory, pos)
        v = memory
        
        if return_attn_logits:
            # Compute attention logits (already averaged over heads)
            attn_logits = self._compute_attn_logits(q, k)
            
            # Use multihead_attn for output computation
            tgt2 = self.multihead_attn(query=q,
                                       key=k,
                                       value=v, 
                                       attn_mask=attn_mask,
                                       key_padding_mask=memory_key_padding_mask)[0]
        else:
            tgt2 = self.multihead_attn(query=q,
                                       key=k,
                                       value=v, 
                                       attn_mask=attn_mask,
                                       key_padding_mask=memory_key_padding_mask)[0]
        
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        
        return (tgt, attn_logits) if return_attn_logits else (tgt, None)
    
    def forward_pre(self, tgt, memory,
                    attn_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None,
                    return_attn_logits: bool = False):
        tgt2 = self.norm(tgt)
        q = self.with_pos_embed(tgt2, query_pos)
        k = self.with_pos_embed(memory, pos)
        v = memory
        
        if return_attn_logits:
            # Compute attention logits (already averaged over heads)
            attn_logits = self._compute_attn_logits(q, k)
            
            # Use multihead_attn for output computation
            tgt2 = self.multihead_attn(query=q,
                                       key=k,
                                       value=v, 
                                       attn_mask=attn_mask,
                                       key_padding_mask=memory_key_padding_mask)[0]
        else:
            tgt2 = self.multihead_attn(query=q,
                                       key=k,
                                       value=v, 
                                       attn_mask=attn_mask,
                                       key_padding_mask=memory_key_padding_mask)[0]
        
        tgt = tgt + self.dropout(tgt2)
        return (tgt, attn_logits) if return_attn_logits else (tgt, None)
    
    def forward(
        self, 
        tgt, # (B,Q,C) 
        memory,  # (B,H,W,C)
        attn_mask: Optional[Tensor] = None, # (B, num_heads, Q, H, W)
        memory_key_padding_mask: Optional[Tensor] = None,
        memory_pos_emb: Optional[Tensor] = None, # (B,H,W,C), 
        query_pos_emb: Optional[Tensor] = None, # (B,Q,C)
        return_attn_logits: bool = False,
        pos: Optional[Tensor] = None, # (B,Q,[x,y,w,j])
    ):
        tgt = tgt.transpose(0,1) # (Q,B,C)
        memory = memory.flatten(1,2).transpose(0,1) # (H*W,B,C)
        attn_mask = attn_mask.flatten(0,1).flatten(2,3) # (B*num_heads, Q, H*W)
        memory_pos_emb = memory_pos_emb.flatten(1,2).transpose(0,1) # (H*W,B,C)
        query_pos_emb = query_pos_emb.transpose(0,1) # (Q,B,C)

        if self.normalize_before:
            output, logits = self.forward_pre(tgt, memory, attn_mask,
                                    memory_key_padding_mask, memory_pos_emb, query_pos_emb,
                                    return_attn_logits=return_attn_logits)
        else:
            output, logits = self.forward_post(tgt, memory, attn_mask,
                                 memory_key_padding_mask, memory_pos_emb, query_pos_emb,
                                 return_attn_logits=return_attn_logits)
    
        return output.transpose(0,1), logits # (Q,B,C)

class SlotCrossAttention(nn.Module):
    def __init__(
        self, 
        dim: int, 
        n_heads: int, 
        dropout: float = 0.1, 
        normalize_before: bool = False
    ):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"d_model ({dim}) debe ser divisible por nhead ({n_heads})")

        self.d_model = dim
        self.nhead = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.normalize_before = normalize_before

        # Capas de proyección lineal para Q, K, V
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        # Capa de salida
        self.to_out = nn.Linear(dim, dim)

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor: Tensor, pos: Optional[Tensor]) -> Tensor:
        return tensor if pos is None else tensor + pos

    def _slot_attention_forward(
        self, 
        q_input: Tensor, 
        k_input: Tensor, 
        v_input: Tensor,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        B, Q, C = q_input.shape
        _B, K, _C = k_input.shape

        q = self.to_q(q_input).view(B, Q, self.nhead, self.head_dim).transpose(1, 2) # (B, H, Q, D)
        k = self.to_k(k_input).view(B, K, self.nhead, self.head_dim).transpose(1, 2) # (B, H, K, D)
        v = self.to_v(v_input).view(B, K, self.nhead, self.head_dim).transpose(1, 2) # (B, H, K, D)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale # (B, H, Q, K)

        # Aplicar máscaras antes del softmax
        if attn_mask is not None:
            # attn_mask (B, H, Q, K) es binaria. True significa enmascarar.
            # Rellenamos con -inf donde la máscara es True.
            dots = dots.masked_fill(attn_mask, float('-inf')) # <-- CAMBIO CLAVE

        if key_padding_mask is not None:
            # key_padding_mask (B, K) es binaria. True significa enmascarar.
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, K)
            dots = dots.masked_fill(mask, float('-inf'))

        attn = dots.softmax(dim=2)
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)

        updates = torch.einsum('bhij,bhjd->bhid', attn, v)
        updates = updates.transpose(1, 2).contiguous().view(B, Q, C)
        output = self.to_out(updates)
        
        return output, attn

    def forward_post(self, tgt, memory, attn_mask, memory_key_padding_mask, memory_pos_emb, query_pos_emb, return_attn_logits):
        q_input = self.with_pos_embed(tgt, query_pos_emb)
        k_input = self.with_pos_embed(memory, memory_pos_emb)
        tgt2, attn = self._slot_attention_forward(
            q_input, k_input, memory, attn_mask, memory_key_padding_mask
        )
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return (tgt, attn) if return_attn_logits else (tgt, None)

    def forward_pre(self, tgt, memory, attn_mask, memory_key_padding_mask, memory_pos_emb, query_pos_emb, return_attn_logits):
        tgt2 = self.norm(tgt)
        q_input = self.with_pos_embed(tgt2, query_pos_emb)
        k_input = self.with_pos_embed(memory, memory_pos_emb)
        tgt3, attn = self._slot_attention_forward(
            q_input, k_input, memory, attn_mask, memory_key_padding_mask
        )
        tgt = tgt + self.dropout(tgt3)
        return (tgt, attn) if return_attn_logits else (tgt, None)

    def forward(
        self, 
        tgt: Tensor, # (B, Q, C) 
        memory: Tensor,  # (B, H, W, C)
        attn_mask: Optional[Tensor] = None, # (B, num_heads, Q, H, W)
        memory_key_padding_mask: Optional[Tensor] = None, # (B, H*W)
        memory_pos_emb: Optional[Tensor] = None, # (B, H, W, C), 
        query_pos_emb: Optional[Tensor] = None, # (B, Q, C)
        return_attn_logits: bool = False,
        pos: Optional[Tensor] = None,
    ):
        memory = memory.flatten(1, 2)
        if memory_pos_emb is not None:
            memory_pos_emb = memory_pos_emb.flatten(1, 2)
        
        if attn_mask is not None:
            attn_mask = attn_mask.flatten(3, 4).bool()

        if self.normalize_before:
            output, logits = self.forward_pre(
                tgt, memory, attn_mask, memory_key_padding_mask, memory_pos_emb, query_pos_emb, return_attn_logits
            )
        else:
            output, logits = self.forward_post(
                tgt, memory, attn_mask, memory_key_padding_mask, memory_pos_emb, query_pos_emb, return_attn_logits
            )
    
        return output, logits

class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@TRANSFORMER_DECODER_REGISTRY.register()
class MultiScaleMaskedTransformerDecoder(nn.Module):

    @configurable
    def __init__(
        self,
        in_channels,
        mask_classification=True,
        *,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,
        clip_embedding_dim: int
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
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

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = {}
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification
        
        ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
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

        # prediction heads on learnable query features
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
            output, 
            mask_features, 
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
            output, _ = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )

            output, _ = self.transformer_self_attention_layers[i](
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
                attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels],
                text_classifier=text_classifier, 
                num_templates=num_templates
            )
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        assert len(predictions_class) == self.num_layers + 1

        out = {
            'pred_logits': predictions_class[-1],
            'pred_panoptic_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask
            )
        }
        return out

    @torch.compiler.disable(recursive=False)
    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size, text_classifier, num_templates):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        # fcclip head
        maskpool_embeddings = self.mask_pooling(x=mask_features, mask=outputs_mask) # [B, Q, C]
        maskpool_embeddings = self._mask_pooling_proj(maskpool_embeddings)
        class_embed = self.class_embed(maskpool_embeddings + decoder_output)

        # TODO here convert text_classifier to RD descriptors.

        outputs_class = get_classification_logits(class_embed, text_classifier, self.logit_scale, num_templates)

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
                {"pred_logits": a, "pred_panoptic_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [{"pred_panoptic_masks": b} for b in outputs_seg_masks[:-1]]
