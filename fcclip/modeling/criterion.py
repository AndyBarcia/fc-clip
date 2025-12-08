"""
This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 

Reference: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/criterion.py

FC-CLIP criterion.
"""

import logging

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.utils.comm import get_world_size

from ..utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list
from .transformer_decoder.box_regression import generalized_box_iou, box_cxcywh_to_xyxy
from .mask_utils import compute_mask_block_counts


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        self.tv_eps = 1e-6

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"][:,:-1].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}
        return losses
    
    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks using exact high-resolution targets."""
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        # Classification logits -> per-query scores
        src_logits = outputs["pred_logits"].float()  # (B, Q, C)
        mask_scores = F.softmax(src_logits, dim=-1).max(-1).values  # (B, Q)

        pred_masks = outputs["pred_masks"]  # (B, Q_all, H, W)
        B, Q_all, H, W = pred_masks.shape
        Q_fg = Q_all - 1  # last query is background

        src_masks = pred_masks[src_idx]  # (N, H, W)
        masks = [t["masks"] for t in targets]

        # Full GT tensor, including *all* masks per image (for the background union)
        target_masks_full, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks_full = target_masks_full.to(src_masks)

        # Matched GTs only (for foreground Dice and foreground terms)
        target_masks = target_masks_full[tgt_idx]  # (N, H_t, W_t)

        if src_masks.shape[0] == 0:
            losses = {
                "loss_mask": outputs["pred_masks"].sum() * 0.0,
                "loss_dice": outputs["pred_masks"].sum() * 0.0,
                "loss_panoptic": outputs["pred_masks"].sum() * 0.0,
            }
            return losses

        # ------------------------------------------------------------------
        # Foreground block stats and targets (per instance)
        # ------------------------------------------------------------------
        logits_fg = src_masks.reshape(src_masks.shape[0], -1)  # (N, H*W)
        N = logits_fg.shape[0]

        # Downsample foreground GTs to prediction resolution
        pos_counts_fg, block_area, H_t, W_t = compute_mask_block_counts(
            target_masks, src_masks.shape[-2:]
        )  # pos_counts_fg: (N, H*W)
        pos_counts_fg = pos_counts_fg.to(device=logits_fg.device, dtype=logits_fg.dtype)

        # [0,1] fractional foreground target per block
        mean_targets_fg = pos_counts_fg / block_area  # (N, H*W)

        # ------------------------------------------------------------------
        # Background GT: complement of the union of all GT masks per image
        # ------------------------------------------------------------------
        # union of all FG masks in each image, at original resolution
        union_fg = (target_masks_full > 0).any(dim=1)  # (B, H_t, W_t)
        union_fg = union_fg.to(dtype=logits_fg.dtype)

        # Downsample union to prediction resolution
        pos_counts_union, block_area_bg, _, _ = compute_mask_block_counts(
            union_fg, src_masks.shape[-2:]
        )  # (B, H*W)
        pos_counts_union = pos_counts_union.to(
            device=logits_fg.device, dtype=logits_fg.dtype
        )

        # Background pixels = remaining pixels in each block
        bg_pos_counts = block_area - pos_counts_union  # (B, H*W)
        mean_targets_bg = bg_pos_counts / block_area   # (B, H*W)

        # Raw background logits (for BCE + Dice)
        bg_logits_flat = pred_masks[:, -1].view(B, H * W)  # (B, H*W)

        # ------------------------------------------------------------------
        # 1) BCE loss on each mask (foreground + background)
        # ------------------------------------------------------------------
        # Foreground BCE
        bce_fg = F.binary_cross_entropy_with_logits(
            logits_fg,                      # (N, H*W)
            mean_targets_fg,                # (N, H*W)
            reduction="none",
        )  # (N, H*W)

        # Background BCE (one mask per image)
        bce_bg = F.binary_cross_entropy_with_logits(
            bg_logits_flat,                 # (B, H*W)
            mean_targets_bg,                # (B, H*W)
            reduction="none",
        )  # (B, H*W)

        # IMPORTANT: each foreground mask and each background mask gets the same weight
        loss_mask = (bce_fg.sum() + bce_bg.sum()) / ((num_masks + B) * H * W)

        # ------------------------------------------------------------------
        # 2) Dice loss: foreground (per instance) + background (per image)
        #    (unchanged, using raw logits)
        # ------------------------------------------------------------------
        # Foreground Dice
        probs_fg = torch.sigmoid(logits_fg)  # (N, H*W)
        intersection_fg = (probs_fg * pos_counts_fg).sum(dim=1)
        pred_sum_fg = probs_fg.sum(dim=1) * block_area
        target_sum_fg = pos_counts_fg.sum(dim=1)
        dice_fg = 1 - (2 * intersection_fg) / (pred_sum_fg + target_sum_fg + 1e-6)

        # Background Dice (one term per image)
        probs_bg = torch.sigmoid(bg_logits_flat)  # (B, H*W)
        intersection_bg = (probs_bg * bg_pos_counts).sum(dim=1)
        pred_sum_bg = probs_bg.sum(dim=1) * block_area
        target_sum_bg = bg_pos_counts.sum(dim=1)
        dice_bg = 1 - (2 * intersection_bg) / (pred_sum_bg + target_sum_bg + 1e-6)

        # IMPORTANT: each foreground mask and each background mask gets the same weight
        loss_dice = (dice_fg.sum() + dice_bg.sum()) / (num_masks + B)

        # ------------------------------------------------------------------
        # 3) Panoptic softmax CE on score-weighted mask logits
        #    (old softmax objective, but on "mask * score" logits)
        # ------------------------------------------------------------------
        # Gate logits such that sigmoid(mask_pan) = sigmoid(mask) * sigmoid(score)
        def gate_logits(x, y):
            zeros = torch.zeros_like(x)
            # log(1 + exp(x) + exp(y)) in a stable way
            log_denom = torch.logsumexp(torch.stack([zeros, x, y], dim=0), dim=0)
            return x + y - log_denom
        score_logits = torch.logit(mask_scores).view(B, Q_all, 1, 1).expand(-1,-1,H,W)  # (B,Q_all,1,1)
        weighted_masks = gate_logits(pred_masks, score_logits)  # (B, Q_all, H, W)

        # Concatenate background as last class
        logits_flat = weighted_masks.view(B, Q_all, H * W)                     # (B, Q_all, HW)

        # Foreground expectation term (one row per matched instance)
        assigned_logits = logits_flat[src_idx]            # (N, H*W)
        expected_fg_logit = (assigned_logits * mean_targets_fg).sum()

        # Background expectation term (one row per image)
        bg_logits_flat_pan = weighted_masks[:, -1].view(B, H * W)  # (B, H*W)
        expected_bg_logit = (bg_logits_flat_pan * mean_targets_bg).sum()

        expected_logit = expected_fg_logit + expected_bg_logit

        # Partition function
        logZ = torch.logsumexp(weighted_masks, dim=1)  # (B, H, W)
        loss_panoptic = (logZ.sum() - expected_logit) / (num_masks * H * W)

        losses = {
            "loss_mask": loss_mask,
            "loss_dice": loss_dice,
            "loss_panoptic": loss_panoptic,
        }

        return losses
    
    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        if 'pred_boxes' not in outputs or outputs['pred_boxes'] is None:
            return {}
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)
        ))
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses

    def loss_tv(self, outputs, targets, indices, num_masks):
        """Total variation regularization for all predicted masks."""

        if "pred_masks" not in outputs:
            return {}

        pred_masks = outputs["pred_masks"]

        if pred_masks.numel() == 0:
            return {"loss_tv": pred_masks.sum() * 0.0}

        probs = torch.sigmoid(pred_masks)
        dx = probs[:, :, 1:, :] - probs[:, :, :-1, :]
        dy = probs[:, :, :, 1:] - probs[:, :, :, :-1]

        dx = F.pad(dx, (0, 0, 0, 1))
        dy = F.pad(dy, (0, 1, 0, 0))

        grad_mag = torch.sqrt(dx * dx + dy * dy + self.tv_eps)

        tv_per_mask = grad_mag.sum(dim=[2, 3]) / (probs.shape[-2] * probs.shape[-1])
        loss_tv = tv_per_mask.mean()

        return {"loss_tv": loss_tv}

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
            'boxes': self.loss_boxes,
            'tv': self.loss_tv
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
