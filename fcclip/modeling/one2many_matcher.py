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

from .matcher import batch_dice_loss_jit
from .matcher import batch_sigmoid_ce_loss_jit

class HungarianOne2ManyMatcher(nn.Module):
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
        use_nel_cost: bool = False,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0
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
        self.use_nel_cost = use_nel_cost
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, "all costs cant be 0"

        self.num_points = num_points

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        """More memory-friendly matching"""
        bs, num_queries = outputs["pred_logits"].shape[:2]

        indices = []
        rounds = []

        # Iterate through batch size
        for b in range(bs):
            
            if self.use_nel_cost:
                out_prob = outputs["pred_logits"][b].sigmoid()  # [num_queries, num_classes]
            else:
                out_prob = outputs["pred_logits"][b].softmax(-1)  # [num_queries, num_classes]
            tgt_ids = targets[b]["labels"]

            if tgt_ids.numel() > 0:                
                # Compute the classification cost.
                if self.use_nel_cost:
                    # focal loss
                    neg_cost_class = (1 - self.focal_alpha) * (out_prob ** self.focal_gamma) * (-(1 - out_prob + 1e-8).log())
                    pos_cost_class = self.focal_alpha * ((1 - out_prob) ** self.focal_gamma) * (-(out_prob + 1e-8).log())
                    cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
                else:
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
                    # Compute the focal loss between masks
                    cost_mask = batch_sigmoid_ce_loss_jit(out_mask, tgt_mask)
                    # Compute the dice loss between masks
                    cost_dice = batch_dice_loss_jit(out_mask, tgt_mask)
                
                # Final cost matrix
                C = (
                    self.cost_mask * cost_mask
                    + self.cost_class * cost_class
                    + self.cost_dice * cost_dice
                )
            else:
                C = torch.empty(num_queries, 0, device=out_prob.device)

            C = C.reshape(num_queries, -1).cpu()
            # Make sure the matrix contains no NaN values, or scipy fails 
            C = torch.nan_to_num(C, nan=0.0, posinf=1e+5, neginf=-1e+5)
            
            num_gts = C.shape[1]
            
            if num_gts == 0:
                # No ground truths, so no matches
                indices.append((torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)))
                rounds.append(torch.full((num_queries,), -1, dtype=torch.long))
                continue

            # Repeat ground truths to cover all detections
            num_repeats = (num_queries + num_gts - 1) // num_gts
            tiled_C = C.repeat(1, num_repeats)

            # The cost matrix for assignment should be at least as wide as it is tall
            # to ensure every detection gets a match.
            cost_matrix_for_assignment = tiled_C[:, :num_queries]
            
            # Perform Hungarian matching
            row_ind, col_ind = linear_sum_assignment(cost_matrix_for_assignment)

            # Convert to tensors
            row_ind_tensor = torch.as_tensor(row_ind, dtype=torch.long)
            col_ind_tensor = torch.as_tensor(col_ind, dtype=torch.long)

            # Map the column indices from the tiled matrix back to the original ground truth indices
            original_gt_indices = col_ind_tensor % num_gts

            # Store the final indices for loss computation
            indices.append((row_ind_tensor, original_gt_indices))

            # --- Calculate the "round" based on the cost-ordering for each GT ---
            all_rounds = torch.full((num_queries,), -1, dtype=torch.long)
            
            # Get the costs of the final matches from the original cost matrix C
            match_costs = C[row_ind_tensor, original_gt_indices]

            # Find unique ground truths that have been matched
            unique_gts = torch.unique(original_gt_indices)

            for gt_idx in unique_gts:
                # Find all detections matched to this ground truth
                mask_current_gt = (original_gt_indices == gt_idx)
                
                # Get the detection indices and their corresponding costs
                detections_for_gt = row_ind_tensor[mask_current_gt]
                costs_for_gt = match_costs[mask_current_gt]
                
                # Sort these detections by their cost
                sorted_cost_indices = torch.argsort(costs_for_gt)
                
                # The "round" is the rank in this sorted list (0, 1, 2, ...)
                ranks = torch.arange(len(sorted_cost_indices))
                
                # Get the original detection indices in their sorted order
                sorted_detections_for_gt = detections_for_gt[sorted_cost_indices]
                
                # Assign the ranks to the corresponding detection indices in the final rounds tensor
                all_rounds[sorted_detections_for_gt] = ranks

            rounds.append(all_rounds)

        return indices, rounds

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
