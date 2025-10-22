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

from .mask_loss.functions.sigmoid.sigmoid_ce import SigmoidCELossFunction
from .mask_loss.functions.dice.dice_loss import DiceLossFunction
from .mask_loss.functions.mask.matching import MaskMatchingFunction


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

    def loss_panoptic(self, outputs, targets):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_logits" in outputs
        cls_logits = outputs["pred_logits"]  # (L,B,Q,C)
        assert "pred_masks" in outputs
        mask_logits = outputs["pred_masks"] # (L,B,Q,h,w)
        assert "pan_seg" in targets
        target_pan_idx = targets["pan_seg"] # (B,H,W) default 0
        assert "labels" in targets
        target_cls_idx = targets["labels"]  # (B,GT_max) default -1

        smooth = 1.0
        sigmoid_scale = self.weight_dict["loss_mask"]
        dice_scale = self.weight_dict["loss_dice"]
        cls_scale = self.weight_dict["loss_ce"]

        # This is the value of the panoptic map that corresponds to background.
        # It should be ignored alltogether.
        background_index = 0

        # The costs that will be considered as "no match".
        inf_thresh = 1e30

        # The number of total GTs for loss normalization.
        L,B,Q = mask_logits.shape[:3]
        num_masks = sum([len(x) for x in targets["labels"]])

        # Whether unmatched queries should try to predict background.
        force_unmatched_class_to_background = True
        force_unmatched_masks_to_empty = False

        losses = {}
        if num_masks > 0:
            
            (
                pred_to_gt, # (L,B,Q)
                layer_mask_mean, # (L,)
                layer_dice_mean, # (L,)
                layer_cls_mean, # (L,)
            ) = MaskMatchingFunction.apply(
                mask_logits, # (L,B,Q,h,w)
                target_pan_idx, # (B,H,W) default 0
                cls_logits, # (L,B,Q,C)
                target_cls_idx, # (B,GT_max) default -1
                smooth,
                sigmoid_scale,
                dice_scale,
                cls_scale,
                background_index,
                inf_thresh,
                num_masks,
                force_unmatched_class_to_background,
                force_unmatched_masks_to_empty
            )

            losses[f"loss_ce"] = layer_cls_mean[-1]
            losses[f"loss_mask"] = layer_mask_mean[-1]
            losses[f"loss_dice"] = layer_dice_mean[-1]

            # Export the losses to a dictionary.
            for l_idx in range(L-1):
                losses[f"loss_ce_{l_idx}"] = layer_cls_mean[l_idx]
                losses[f"loss_mask_{l_idx}"] = layer_mask_mean[l_idx]
                losses[f"loss_dice_{l_idx}"] = layer_dice_mean[l_idx]
        else:
            x = mask_logits.sum() * 0.0 + cls_logits.sum() * 0.0
            losses["loss_ce"] = x
            losses[f"loss_mask"] = x
            losses[f"loss_dice"] = x
            for l_idx in range(L-1):
                losses[f"loss_ce_{l_idx}"] = x
                losses[f"loss_mask_{l_idx}"] = x
                losses[f"loss_dice_{l_idx}"] = x

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