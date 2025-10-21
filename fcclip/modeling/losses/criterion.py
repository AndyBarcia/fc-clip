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

from .mask_loss.functions.sigmoid_ce import SigmoidCELossFunction
from .mask_loss.functions.dice import DiceLossFunction
from .mask_loss.functions.matching import MaskMatchingFunction


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(
        self,
        weight_dict,
        losses,
        use_nel_loss: bool = False,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            use_nel_loss: whether to use Non-mutually Exclusive Loss (NEL) instead of CE
        """
        super().__init__()
        self.weight_dict = weight_dict
        self.losses = losses

        # loss toggles / hyperparams
        self.use_nel_loss = use_nel_loss
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

    def loss_panoptic(self, outputs, targets):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_semantic_masks" in outputs
        src_logits = outputs["pred_panoptic_masks"] # (L,B,Q,h,w)
        assert "pred_objectness" in outputs
        obj_logits = outputs["pred_objectness"] # (L,B,Q,1)

        L,B,Q = src_logits.shape[:3]
        num_masks = sum([len(x) for x in targets["labels"]])

        losses = {}
        if num_masks > 0:
            target_sem_idx = targets["pan_seg"] # (B,H,W)
            # Get bce and dice losses for each layer, as well as the matches.
            matches, layer_mask_mean, layer_dice_mean, _ = MaskMatchingFunction.apply(
                src_logits, # (L,B,Q,h,w)
                target_sem_idx, # (B,H,W)
                1.0, # smooth
                self.weight_dict["loss_mask"], # sigmoid_scale
                self.weight_dict["loss_dice"], # dice_scale
                0, # background_index
                1e30, # inf_thresh
                num_masks # num_masks
            ) # (L, B, GT), (L,), (L,)

            # Determine which detections were matched with a ground truth.
            target_objectness = torch.zeros_like(obj_logits, device=obj_logits.device) # (L,B,Q,1)            
            l_indices, b_indices, gt_indices = torch.where(matches >= 0)
            q_indices = matches[l_indices, b_indices, gt_indices]
            target_objectness[l_indices, b_indices, q_indices] = 1.0
            
            # Compute binary cross-entropy loss per layer
            num_queries = B * Q
            for l_idx in range(L-1):
                loss_obj = F.binary_cross_entropy_with_logits(
                    obj_logits[l_idx], 
                    target_objectness[l_idx], 
                    reduction="sum"
                ) / num_queries
                losses[f"loss_objectness_{l_idx}"] = loss_obj * self.weight_dict["loss_objectness"]
            
            # Final layer objectness loss
            loss_obj = F.binary_cross_entropy_with_logits(
                obj_logits[-1], 
                target_objectness[-1], 
                reduction="sum"
            ) / num_queries
            losses["loss_objectness"] = loss_obj * self.weight_dict["loss_objectness"]

            # Export the losses to a dictionary.
            for l_idx in range(L-1):
                losses[f"loss_mask_{l_idx}"] = layer_mask_mean[l_idx]
                losses[f"loss_dice_{l_idx}"] = layer_dice_mean[l_idx]
            losses[f"loss_mask"] = layer_mask_mean[-1]
            losses[f"loss_dice"] = layer_dice_mean[-1]
        else:
            x = src_logits.sum() * 0.0
            for l_idx in range(L-1):
                losses[f"loss_mask_{l_idx}"] = x
                losses[f"loss_dice_{l_idx}"] = x
            losses[f"loss_mask"] = x
            losses[f"loss_dice"] = x
            losses["loss_objectness"] = x

        return losses

    def loss_semantic(self, outputs, targets):
        """
        Compute pixel-wise sigmoid cross-entropy and dice loss for semantic segmentation.
        """
        assert "pred_semantic_masks" in outputs
        src_logits = outputs["pred_semantic_masks"]
            
        L,B,C = src_logits.shape[:3]
        num_masks = float(B*C)

        losses = {}
        if num_masks > 0:
            target_sem_idx = targets["sem_seg"] # (B,H,W)
            loss_mask = SigmoidCELossFunction.apply(
                src_logits, # (L,B,C,h,w)
                target_sem_idx, # (B,H,W)
                num_masks,
                self.weight_dict["loss_semantic_mask"], # scale
            )
            loss_dice = DiceLossFunction.apply(
                src_logits, # (L,B,C,h,w)
                target_sem_idx,  # (B,H,W)
                1.0, # smooth
                num_masks,
                self.weight_dict["loss_semantic_dice"], # scale
            )

            # Export the losses to a dictionary.
            for l_idx in range(L-1):
                losses[f"loss_semantic_mask_{l_idx}"] = loss_mask[l_idx]
                losses[f"loss_semantic_dice_{l_idx}"] = loss_dice[l_idx]
            losses[f"loss_semantic_mask"] = loss_mask[-1]
            losses[f"loss_semantic_dice"] = loss_dice[-1]
        else:
            x = src_logits.sum() * 0.0
            for l_idx in range(L-1):
                losses[f"loss_semantic_mask_{l_idx}"] = x
                losses[f"loss_semantic_dice_{l_idx}"] = x
            losses[f"loss_semantic_mask"] = x
            losses[f"loss_semantic_dice"] = x

        return losses

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # Compute all the requested losses
        losses = {}
        if "semantic" in self.losses:
            losses.update(self.loss_semantic(outputs, targets))
        if "panoptic" in self.losses:
            losses.update(self.loss_panoptic(outputs, targets))

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)