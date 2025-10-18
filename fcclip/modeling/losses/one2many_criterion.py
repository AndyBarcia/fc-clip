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
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from ...utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list
from ..transformer_decoder.box_regression import generalized_box_iou, box_cxcywh_to_xyxy

from .criterion import calculate_uncertainty


class One2ManySetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict,
        eos_coef,
        losses,
        num_points,
        oversample_ratio,
        importance_sample_ratio,
        use_nel_loss: bool = False,
        focal_alpha: float = 0.8,
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
            nel_alpha / nel_beta / nel_gamma: weights and focal gamma for NEL
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

        # NEL toggles / hyperparams
        self.use_nel_loss = use_nel_loss
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

    def loss_labels(self, outputs, targets, indices, num_masks, **kwargs):
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

        # Get per-query loss
        loss_ce_unreduced = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight, reduction='none')

        # Get round-based weights
        rounds = kwargs.get('rounds')
        assert rounds is not None
        weights = torch.ones_like(loss_ce_unreduced)
        # Apply weights only to matched queries
        for i in range(len(indices)):
            _, src_idx = indices[i]
            batch_rounds = rounds[i][src_idx]
            round_weights = 0.5 ** batch_rounds.float().to(weights.device)
            weights[i, src_idx] = round_weights

        loss_ce = (loss_ce_unreduced * weights).mean()
        losses = {"loss_ce": loss_ce}
        return losses
    
    def loss_labels_nel(self, outputs, targets, indices, num_masks, **kwargs):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:,:,:-1]

        # Compute focal loss per query
        prob = src_logits.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(src_logits, target_classes_onehot, reduction="none")
        p_t = prob * target_classes_onehot + (1 - prob) * (1 - target_classes_onehot)
        focal_term = (1.0 - p_t) ** self.focal_gamma
        
        alpha_t = self.focal_alpha * target_classes_onehot + (1 - self.focal_alpha) * (1 - target_classes_onehot)
        
        loss_per_class = alpha_t * focal_term * ce_loss
        loss_per_query = loss_per_class.mean(2) # Shape: [B, Q]

        # Get round-based weights
        rounds = kwargs.get('rounds')
        assert rounds is not None
        weights = torch.ones_like(loss_per_query)
        # Apply weights only to matched queries
        for i in range(len(indices)):
            _, src_idx = indices[i]
            batch_rounds = rounds[i][src_idx]
            round_weights = 0.5 ** batch_rounds.float().to(weights.device)
            weights[i, src_idx] = round_weights

        weighted_loss = (loss_per_query * weights).sum()
        loss_ce = weighted_loss / num_masks * src_logits.shape[1]
        
        losses = {'loss_label_focal': loss_ce}
        return losses
    
    def loss_masks(self, outputs, targets, indices, num_masks, **kwargs):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        if src_masks.shape[0] == 0:
            return {
                "loss_mask": outputs["pred_masks"].sum() * 0.0,
                "loss_dice": outputs["pred_masks"].sum() * 0.0,
            }

        with torch.no_grad():
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        # Get round-based weights for each matched pair
        rounds = kwargs.get('rounds')
        assert rounds is not None
        matched_rounds = torch.cat([rounds[i][I] for i, (I, J) in enumerate(indices)])
        round_weights = 0.5 ** matched_rounds.float().to(point_logits.device)

        # Sigmoid CE loss
        ce_loss_per_mask = F.binary_cross_entropy_with_logits(point_logits, point_labels, reduction="none").mean(1)
        weighted_ce_loss = (ce_loss_per_mask * round_weights).sum() / num_masks

        # Dice loss
        dice_inputs = point_logits.sigmoid().flatten(1)
        dice_targets = point_labels.flatten(1)
        numerator = 2 * (dice_inputs * dice_targets).sum(-1)
        denominator = dice_inputs.sum(-1) + dice_targets.sum(-1)
        dice_loss_per_mask = 1 - (numerator + 1) / (denominator + 1)
        weighted_dice_loss = (dice_loss_per_mask * round_weights).sum() / num_masks

        losses = {
            "loss_mask": weighted_ce_loss,
            "loss_dice": weighted_dice_loss,
        }

        return losses
    
    def loss_boxes(self, outputs, targets, indices, num_masks, **kwargs):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        if 'pred_boxes' not in outputs or outputs['pred_boxes'] is None:
            return {}
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # All boxes are matched, so all losses are scaled
        if src_boxes.shape[0] == 0:
            return {
                "loss_bbox": src_boxes.sum() * 0.0,
                "loss_giou": src_boxes.sum() * 0.0,
            }

        # Get round-based weights for each matched pair
        rounds = kwargs.get('rounds')
        assert rounds is not None
        matched_rounds = torch.cat([rounds[i][I] for i, (I, J) in enumerate(indices)])
        round_weights = 0.5 ** matched_rounds.float().to(src_boxes.device)
        
        # Bbox L1 loss
        loss_bbox_per_box = F.l1_loss(src_boxes, target_boxes, reduction='none').sum(dim=1)
        weighted_bbox_loss = (loss_bbox_per_box * round_weights).sum() / num_masks

        # GIoU loss
        loss_giou_per_box = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)
        ))
        weighted_giou_loss = (loss_giou_per_box * round_weights).sum() / num_masks

        losses = {
            'loss_bbox': weighted_bbox_loss,
            'loss_giou': weighted_giou_loss,
        }
        return losses

    def loss_rounds(self, outputs, targets, indices, num_masks, **kwargs):
        """
        Binary classification loss for the first-round prediction head, scaled by round.
        """
        assert "pred_round" in outputs
        src_logits = outputs["pred_round"]
        idx = self._get_src_permutation_idx(indices)
        
        # Get the round number for each matched prediction
        rounds = kwargs.get('rounds')
        assert rounds is not None
        matched_rounds = torch.cat([r[I] for r, (I, _) in zip(rounds, indices)])
        
        # Create the binary target: 1 if round is 0, else 0.
        target_rounds_o = (matched_rounds == 0).float()
        src_logits_matched = src_logits[idx]

        if src_logits_matched.shape[0] == 0:
            return {"loss_round": src_logits.sum() * 0.0}

        # Get round-based weights
        round_weights = 0.5 ** matched_rounds.float().to(src_logits.device)

        # Compute weighted binary cross-entropy loss
        loss_round_per_item = F.binary_cross_entropy_with_logits(
            src_logits_matched.squeeze(1), target_rounds_o.to(src_logits_matched.device), reduction="none"
        )
        weighted_loss = (loss_round_per_item * round_weights).sum()
        
        losses = {"loss_round": weighted_loss / num_masks}
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

    def get_loss(self, loss, outputs, targets, indices, num_masks, **kwargs):
        loss_map = {
            'labels': self.loss_labels_nel if self.use_nel_loss else self.loss_labels,
            'masks': self.loss_masks,
            'boxes': self.loss_boxes,
            'rounds': self.loss_rounds
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        # Pass kwargs (containing rounds) to the loss function
        return loss_map[loss](outputs, targets, indices, num_masks, **kwargs)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices, rounds = self.matcher(outputs_without_aux, targets)

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
            kwargs = {'rounds': rounds}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices_aux, rounds_aux = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {'rounds': rounds_aux}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_aux, num_masks, **kwargs)
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