"""
This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 

Reference: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/criterion.py

FC-CLIP criterion.
"""

import logging

import torch
from torch import autograd
import torch.nn.functional as F
from torch import nn

from detectron2.utils.comm import get_world_size

from ..utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list
from .transformer_decoder.box_regression import generalized_box_iou, box_cxcywh_to_xyxy
from .mask_utils import compute_mask_block_counts


class ShiftToMatchArea(autograd.Function):
    @staticmethod
    def forward(ctx, z, K, max_iters: int = 20, tol: float = 1e-7):
        """
        z:  (B,Q,N) logits, any dtype
        K:  (B,Q)   target counts
        Returns z': (B,Q,N) with sum sigmoid(z') == K (safeguarded Newton).
        """
        B, Q, N = z.shape
        # Work in float64 for the root solve
        zd = z.double()
        Kd = K.to(dtype=torch.float64, device=z.device).clamp_(0.0, float(N))

        # Brackets (guarantee feasibility): sigma(z + L) ~ 0, sigma(z + U) ~ 1
        z_max = zd.max(dim=2, keepdim=True).values
        z_min = zd.min(dim=2, keepdim=True).values
        L = -20.0 - z_max           # (B,Q,1)
        U =  20.0 - z_min           # (B,Q,1)

        def _logit(x, eps=1e-6):
            x = x.clamp(min=eps, max=1 - eps)
            return torch.log(x) - torch.log1p(-x)

        # Reasonable initializer via mean logit difference
        p0_mean = torch.sigmoid(zd).mean(dim=2)      # (B,Q)
        t_mean  = Kd / float(N)                      # (B,Q)
        lam = (_logit(t_mean) - _logit(p0_mean)).unsqueeze(-1)  # (B,Q,1)
        # Project init into bracket
        lam = lam.clamp_min(L).clamp_max(U)

        Kcol = Kd.unsqueeze(-1)

        # Helper
        def eval_fS(l):
            p = torch.sigmoid(zd + l)
            # TODO here sum or binary sum?
            # f = (p>0.5).sum(dim=2, keepdim=True) - Kcol
            f = p.sum(dim=2, keepdim=True) - Kcol
            s = (p * (1 - p)).sum(dim=2, keepdim=True)
            return f, s

        f, S = eval_fS(lam)

        for _ in range(max_iters):
            done = f.abs() <= tol
            if done.all():
                break

            # Newton proposal (guard S)
            S_safe = S + 1e-18
            newton_step = f / S_safe
            lam_newton = lam - newton_step

            # If Newton leaves bracket or doesn't improve |f|, use bisection
            out_of_bracket = (lam_newton < L) | (lam_newton > U)

            # Evaluate Newton where valid & not done
            lam_try = torch.where(out_of_bracket | done, lam, lam_newton)
            f_try, S_try = eval_fS(lam_try)

            no_improve = (f_try.abs() > f.abs()) & (~out_of_bracket) & (~done)

            # Bisection fallback where needed
            mid = 0.5 * (L + U)
            lam_next = torch.where(out_of_bracket | no_improve, mid, lam_try)

            # Update brackets using sign at lam_next
            f_next, S_next = eval_fS(lam_next)
            go_upper = f_next > 0  # sum(sigmoid) > K => need smaller lambda (move U down)
            U = torch.where(go_upper, lam_next, U)
            L = torch.where(~go_upper, lam_next, L)

            lam, f, S = lam_next, f_next, S_next

        # Final clamp to bracket and compute shifted logits in original dtype
        lam = lam.clamp_min(L).clamp_max(U).to(dtype=z.dtype)
        z_shifted = z + lam
        p_final = torch.sigmoid(z_shifted)
        ctx.save_for_backward(p_final)
        return z_shifted, lam

    @staticmethod
    def backward(ctx, grad_zprime, _):
        (p,) = ctx.saved_tensors
        s = p * (1 - p)
        S = s.sum(dim=2, keepdim=True) + 1e-12
        sum_g = grad_zprime.sum(dim=2, keepdim=True)
        grad_z = grad_zprime - (s / S) * sum_g
        return grad_z, None, None, None


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

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

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
        """Compute mask losses with area-conditioned (shifted) logits.
        Keeps compute_mask_block_counts optimization. Uses the provided ShiftToMatchArea.
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"][src_idx]              # (M, Hs, Ws)
        masks = [t["masks"] for t in targets]
        target_masks, _ = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)[tgt_idx]      # (M, Ht, Wt)

        if src_masks.shape[0] == 0:
            zero = outputs["pred_masks"].sum() * 0.0
            return {"loss_mask": zero, "loss_dice": zero}

        # Flatten logits and get aggregated target counts per logit
        logits = src_masks.reshape(src_masks.shape[0], -1)                      # (M, N)
        pos_counts, block_area, H_t, W_t = compute_mask_block_counts(           # pos_counts: (M, N)
            target_masks, src_masks.shape[-2:]
        )
        pos_counts = pos_counts.to(device=logits.device, dtype=logits.dtype)
        block_area = float(block_area)  # scalar (e.g., 16 for 4x4)

        # --- Area-conditioned shift (uniform replication): sum_j σ(z_j + λ) = K_low ---
        # Convert high-res positive pixels to low-res target count
        K_high = pos_counts.sum(dim=1)                             # (M,)
        K_low  = (K_high / block_area).clamp_(0.0, float(logits.shape[1]))

        # Use your original ShiftToMatchArea on shape (M,1,N) / (M,1)
        z_in = logits.unsqueeze(1)                                  # (M, 1, N)
        K_in = K_low.unsqueeze(1)                                   # (M, 1)
        z_shift, _ = ShiftToMatchArea.apply(z_in, K_in, 20, 1e-7)   # returns (M,1,N)
        z_shift = z_shift.squeeze(1)                                # (M, N)

        # ---------- BCE with shifted logits (aggregated, numerically stable) ----------
        abs_logits = z_shift.abs()
        max_logits = torch.clamp(z_shift, min=0)
        logexp = torch.log1p(torch.exp(-abs_logits))

        # Each logit represents block_area micro-pixels; pos_counts_j positives among them
        loss_block = block_area * max_logits - z_shift * pos_counts + block_area * logexp  # (M, N)
        loss_mask = (loss_block.sum(dim=1) / (H_t * W_t)).sum() / num_masks

        # ---------- Dice with shifted logits ----------
        probs = torch.sigmoid(z_shift)
        intersection = (probs * pos_counts).sum(dim=1)                 # (M,)
        pred_sum = block_area * probs.sum(dim=1)                       # (M,)
        target_sum = K_high                                            # (M,)
        loss_dice = 1 - (2 * intersection + 1) / (pred_sum + target_sum + 1)
        loss_dice = loss_dice.sum() / num_masks

        return {"loss_mask": loss_mask, "loss_dice": loss_dice}
    
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

    def loss_areas(self, outputs, targets, indices, num_masks):
        if "pred_areas" not in outputs or outputs["pred_areas"] is None:
            return {}

        idx = self._get_src_permutation_idx(indices)
        src_areas = outputs["pred_areas"][idx]

        if src_areas.numel() == 0:
            return {"loss_area": outputs["pred_areas"].sum() * 0.0}

        masks = [t["masks"] for t in targets]
        target_masks, _ = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_areas)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        target_masks = target_masks[tgt_idx]
        target_areas = target_masks.flatten(1).to(dtype=src_areas.dtype).mean(dim=1)

        loss_area = F.l1_loss(src_areas, target_areas, reduction="none")
        losses = {"loss_area": loss_area.sum() / num_masks}

        return losses

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
            'areas': self.loss_areas
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
