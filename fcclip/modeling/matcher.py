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

from .mask_utils import compute_mask_block_counts
from .criterion import ShiftToMatchArea


@torch.no_grad()
def pairwise_area_conditioned_costs(
    logits_qn: torch.Tensor,          # [Q, N] flattened mask logits for all queries
    tgt_pos_counts_mn: torch.Tensor,  # [M, N] blockwise positive counts per target (0..block_area per cell)
    block_area: int,
    hw: int,                          # H_t * W_t (target resolution area)
    chunk_size: int = 64,
):
    """
    Returns:
      cost_bce:  [Q, M] area-conditioned BCE cost (with shifted logits)
      cost_dice: [Q, M] area-conditioned Dice cost (with shifted logits)
    """
    device = logits_qn.device
    dtype = torch.float32

    Q, N = logits_qn.shape
    M = tgt_pos_counts_mn.shape[0]
    if M == 0:
        return torch.empty(Q, 0, device=device), torch.empty(Q, 0, device=device)

    # numerics in float32
    z_qn = logits_qn.to(dtype=dtype)
    pos_mn = tgt_pos_counts_mn.to(device=device, dtype=dtype)

    # High-res positives per target and low-res target counts for the shifter
    K_high_m = pos_mn.sum(dim=1)                       # [M] (in pixels at target res)
    K_low_m = (K_high_m / block_area).clamp_(0.0, float(N))  # [M] (in "logit units")

    cost_bce  = torch.empty(Q, M, device=device, dtype=dtype)
    cost_dice = torch.empty(Q, M, device=device, dtype=dtype)

    for m0 in range(0, M, chunk_size):
        m1 = min(M, m0 + chunk_size)
        Mc = m1 - m0

        # Shape to (Q, Mc, N) so we can shift per (q,m) pair with a single call
        z = z_qn.unsqueeze(1).expand(Q, Mc, N)                        # [Q, Mc, N]
        K = K_low_m[m0:m1].unsqueeze(0).expand(Q, Mc)                 # [Q, Mc]

        # Area-conditioned shift (uniform replication): sum_j σ(z_{q,m,j} + λ_{q,m}) = K_low[m]
        z_shift, _ = ShiftToMatchArea.apply(z, K, 5, 1e-7)           # [Q, Mc, N]

        # ---- BCE with shifted logits (aggregate using scalar block_area) ----
        abs_logits = z_shift.abs()
        max_logits = torch.clamp(z_shift, min=0)
        logexp = torch.log1p(torch.exp(-abs_logits))

        pos = pos_mn[m0:m1].unsqueeze(0).expand(Q, Mc, N)             # [Q, Mc, N]
        loss_block = block_area * max_logits - z_shift * pos + block_area * logexp
        bce = loss_block.sum(dim=2) / float(hw)                       # [Q, Mc]

        # ---- Dice with shifted logits (aggregate using scalar block_area) ----
        probs = torch.sigmoid(z_shift)
        intersection = (probs * pos).sum(dim=2)                       # [Q, Mc]
        pred_sum = block_area * probs.sum(dim=2)                      # [Q, Mc]
        target_sum = K_high_m[m0:m1].unsqueeze(0).expand(Q, Mc)       # [Q, Mc]
        dice = 1.0 - (2.0 * intersection + 1.0) / (pred_sum + target_sum + 1.0)

        cost_bce[:, m0:m1]  = bce
        cost_dice[:, m0:m1] = dice

    return cost_bce, cost_dice


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_mask: float = 1, cost_dice: float = 1,
                 cost_area: float = 0, num_points: int = 0):
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
        self.cost_area = cost_area

        assert (
            cost_class != 0 or cost_mask != 0 or cost_dice != 0 or cost_area != 0
        ), "all costs cant be 0"

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
                target_counts, block_area, H_t, W_t = compute_mask_block_counts(
                    tgt_mask, out_mask.shape[-2:]
                )
                out_mask = out_mask.flatten(1)
                target_counts = target_counts.to(device=out_mask.device, dtype=out_mask.dtype)

                with autocast(enabled=False):
                    out_mask = out_mask.float()
                    target_counts = target_counts.float()
                    # Compute the focal loss between masks
                    cost_mask, cost_dice = pairwise_area_conditioned_costs(
                        out_mask, target_counts, block_area, H_t * W_t
                    )

                if self.cost_area != 0 and "pred_areas" in outputs:
                    pred_areas = outputs["pred_areas"][b].float()
                    target_areas = tgt_mask.float().flatten(1).mean(dim=1)
                    cost_area = (pred_areas[:, None] - target_areas[None, :]).abs()
                else:
                    cost_area = 0

                # Final cost matrix
                C = (
                    self.cost_mask * cost_mask
                    + self.cost_class * cost_class
                    + self.cost_dice * cost_dice
                    + self.cost_area * cost_area
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
            "cost_area: {}".format(self.cost_area),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
