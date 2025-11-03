"""
This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 

Reference: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/matcher.py

Modules to compute the matching cost and solve the corresponding LSAP.
"""

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.cuda.amp import autocast

from detectron2.projects.point_rend.point_features import point_sample


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_mask: float = 1, cost_dice: float = 1, num_points: int = 0):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight associated with the signed distance field matching cost
            cost_dice: Additional relative weight associated with the signed distance field matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, "all costs cant be 0"

        self.num_points = num_points

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        """More memory-friendly matching"""
        bs, num_queries = outputs["pred_logits"].shape[:2]

        indices = []

        # Iterate through batch size
        for b in range(bs):

            out_prob = outputs["pred_logits"][b].softmax(-1)  # [num_queries, num_classes]
            tgt_ids = targets[b]["labels"]

            if tgt_ids.numel() > 0:                
                # Compute the classification cost.
                cost_class = -out_prob[:, tgt_ids]

                out_mask = outputs["pred_masks"][b]  # [num_queries, H_pred, W_pred]
                tgt_mask = targets[b]["masks"].to(out_mask)

                out_mask = out_mask[:, None]
                tgt_mask = tgt_mask[:, None]

                # all masks share the same set of points for efficient matching!
                point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)

                # Sample points from ground-truth masks
                tgt_mask = point_sample(
                    tgt_mask,
                    point_coords.repeat(tgt_mask.shape[0], 1, 1),
                    align_corners=False,
                ).squeeze(1)

                # Sample points from predicted masks
                out_mask = point_sample(
                    out_mask,
                    point_coords.repeat(out_mask.shape[0], 1, 1),
                    align_corners=False,
                ).squeeze(1)

                with autocast(enabled=False):
                    out_mask = out_mask.float()
                    tgt_mask = tgt_mask.float()
                    # Compute the L1 cost between signed distance fields
                    cost_mask = torch.cdist(out_mask, tgt_mask, p=1)
                    # Normalize by the number of sampled points to keep scale comparable
                    cost_mask = cost_mask / max(out_mask.shape[-1], 1)

                # Final cost matrix
                C = (
                    (self.cost_mask + self.cost_dice) * cost_mask
                    + self.cost_class * cost_class
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
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted signed distance fields

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target signed distance fields

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
