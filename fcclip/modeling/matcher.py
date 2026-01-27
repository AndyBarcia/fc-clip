"""
This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates.

Reference: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/matcher.py

Modules to compute the matching cost and solve the corresponding LSAP.
"""

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.cuda.amp import autocast

from detectron2.projects.point_rend.point_features import point_sample
from .mask_utils import compute_mask_block_counts


def batch_sigmoid_ce_loss(
    logits: torch.Tensor,
    target_counts: torch.Tensor,
    block_area: int,
    original_hw: int,
) -> torch.Tensor:
    abs_logits = logits.abs()
    max_logits = torch.clamp(logits, min=0)
    logexp = torch.log1p(torch.exp(-abs_logits))
    sum_max = block_area * max_logits.sum(dim=1)
    sum_logexp = block_area * logexp.sum(dim=1)
    dot = torch.einsum("qc,mc->qm", logits, target_counts)
    loss = sum_max[:, None] - dot + sum_logexp[:, None]
    return loss / original_hw


def batch_dice_loss(
    logits: torch.Tensor, target_counts: torch.Tensor, block_area: int
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    intersection = torch.einsum("qc,mc->qm", probs, target_counts)
    pred_sum = block_area * probs.sum(dim=1)
    target_sum = target_counts.sum(dim=1)
    #loss = 1 - (2 * intersection + 1) / (pred_sum[:, None] + target_sum[None, :] + 1)
    loss = 1 - (2 * intersection) / (pred_sum[:, None] + target_sum[None, :])
    return loss


def sampling_batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


sampling_batch_dice_loss_jit = torch.jit.script(
    sampling_batch_dice_loss
)  # type: torch.jit.ScriptModule


def sampling_batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )

    return loss / hw


sampling_batch_sigmoid_ce_loss_jit = torch.jit.script(
    sampling_batch_sigmoid_ce_loss
)  # type: torch.jit.ScriptModule

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
        self, 
        cost_class: float = 1, 
        cost_mask: float = 1, 
        cost_dice: float = 1, 
        num_points: int = 0,
        sapmpling_loss: bool = False,
    ):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, "all costs cant be 0"

        self.num_points = num_points
        self.sapmpling_loss = sapmpling_loss

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        """More memory-friendly matching"""
        bs, num_queries = outputs["pred_obj"].shape

        indices = []

        # Iterate through batch size
        for b in range(bs):

            out_prob = outputs["pred_obj"][b].sigmoid()  # [num_queries]
            tgt_ids = targets[b]["labels"]

            if tgt_ids.numel() > 0:
                # Compute the classification cost as just the inverse of the probability of the query
                # being predicted as an object.
                cost_class = -out_prob[:, None].repeat(1,tgt_ids.shape[-1])  # [num_queries, 1]

                out_mask = outputs["pred_masks"][b]  # [num_queries, H_pred, W_pred]
                tgt_mask = targets[b]["masks"].to(out_mask)

                if self.sapmpling_loss:
                    out_mask = out_mask[:, None]
                    tgt_mask = tgt_mask[:, None]
                    # all masks share the same set of points for efficient matching!
                    point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)
                    # get gt labels
                    tgt_mask = point_sample(
                        tgt_mask,
                        point_coords.repeat(tgt_mask.shape[0], 1, 1),
                        align_corners=False,
                    ).squeeze(1)

                    out_mask = point_sample(
                        out_mask,
                        point_coords.repeat(out_mask.shape[0], 1, 1),
                        align_corners=False,
                    ).squeeze(1)

                    with autocast(enabled=False):
                        out_mask = out_mask.float()
                        tgt_mask = tgt_mask.float()
                        # Compute the focal loss between masks
                        cost_mask = sampling_batch_sigmoid_ce_loss_jit(out_mask, tgt_mask)

                        # Compute the dice loss betwen masks
                        cost_dice = sampling_batch_dice_loss_jit(out_mask, tgt_mask)
                else:
                    target_counts, block_area, H_t, W_t = compute_mask_block_counts(
                        tgt_mask, out_mask.shape[-2:]
                    )
                    out_mask = out_mask.flatten(1)
                    target_counts = target_counts.to(device=out_mask.device, dtype=out_mask.dtype)

                    with autocast(enabled=False):
                        out_mask = out_mask.float()
                        target_counts = target_counts.float()
                        # Compute the focal loss between masks
                        cost_mask = batch_sigmoid_ce_loss(
                            out_mask, target_counts, block_area, H_t * W_t
                        )
                        # Compute the dice loss between masks
                        cost_dice = batch_dice_loss(out_mask, target_counts, block_area)

                # Final cost matrix
                C = (
                    self.cost_mask * cost_mask
                    + self.cost_class * cost_class
                    + self.cost_dice * cost_dice
                )
            else:
                # This block runs if there are NO ground-truth objects.
                # We create an empty cost matrix. The shape [num_queries, 0] is
                # important. linear_sum_assignment will correctly handle this
                # by returning empty indices, which is the desired behavior.
                C = torch.empty(num_queries, 0, device=out_prob.device)
            C = C.reshape(num_queries, -1).cpu()
            # Make sure the matrix contains no NaN values, or scipy fails 
            C[C.isnan()] = 0.0

            indices.append(linear_sum_assignment(C))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        return self.memory_efficient_forward(outputs, targets)

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_mask: {}".format(self.cost_mask),
            "cost_dice: {}".format(self.cost_dice),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
