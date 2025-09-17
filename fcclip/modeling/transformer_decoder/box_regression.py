import torch
import torch.nn as nn
from detectron2.structures import BitMasks
from torchvision.ops.boxes import box_area
import torch.nn.functional as F

from .fcclip_transformer_decoder import MLP


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)


    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / (union + 1e-6)
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / (area + 1e-6)


# modified from torchvision to also return the union
def box_iou_pairwise(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # [N,2]
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # [N,2]

    wh = (rb - lt).clamp(min=0)  # [N,2]
    inter = wh[:, 0] * wh[:, 1]  # [N]

    union = area1 + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou_pairwise(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    Input:
        - boxes1, boxes2: N,4
    Output:
        - giou: N, 4
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    assert boxes1.shape == boxes2.shape
    iou, union = box_iou_pairwise(boxes1, boxes2) # N, 4

    lt = torch.min(boxes1[:, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,2]
    area = wh[:, 0] * wh[:, 1]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]
    device = masks.device

    y = torch.arange(0, h, dtype=torch.float, device=device)
    x = torch.arange(0, w, dtype=torch.float, device=device)
    y, x = torch.meshgrid(y, x, indexing='ij')

    # Convert to boolean once for efficiency
    bool_masks = masks.bool()
    
    # Compute min/max coordinates
    x_mask = (bool_masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~bool_masks, float('inf')).flatten(1).min(-1)[0]

    y_mask = (bool_masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~bool_masks, float('inf')).flatten(1).min(-1)[0]

    # Stack into [x_min, y_min, x_max, y_max] format
    boxes = torch.stack([x_min, y_min, x_max, y_max], 1)
    
    # Identify empty masks (no True values)
    empty = ~bool_masks.flatten(1).any(1)
    
    # Convert to exclusive coordinates for non-empty masks
    non_empty = ~empty
    boxes[non_empty, 2] += 1  # x_max becomes exclusive
    boxes[non_empty, 3] += 1  # y_max becomes exclusive
    
    # For empty masks, use entire image box [0, 0, w, h]
    boxes[empty] = torch.tensor([0, 0, w, h], device=device, dtype=torch.float)
    
    return boxes



class BBoxMLPRegression(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # Create MLP with 3 layers: hidden_dim -> hidden_dim -> 4
        self._bbox_embed = MLP(hidden_dim, hidden_dim, 4, num_layers=3)
        
        # Initialize last layer weights and biases to zero
        nn.init.constant_(self._bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self._bbox_embed.layers[-1].bias.data, 0)

    def forward(self, x, reference_points, masks=None, normalized_space=True):
        """
        Args:
            x: Query embeddings 
                shape: (batch_size, num_queries, hidden_dim)
            reference_points: Current reference points (normalized 0-1)
                shape: (batch_size, num_queries, 4)
            masks: Masks associated with the queries. Not used here.
                shape: (batch_size, num_queries, height, width)
        
        Returns:
            new_reference_points: Updated reference points after refinement
                shape: (batch_size, num_queries, 4)
        """        
        # Step 1: Apply inverse sigmoid to reference points
        if normalized_space:
            reference_points_unsigmoid = inverse_sigmoid(reference_points)
        else:
            reference_points_unsigmoid = reference_points

        # Step 2: Compute bbox delta through MLP
        delta_unsig = self._bbox_embed(x)
        
        # Step 3: Add delta to reference_before_sigmoid
        outputs_unsig = delta_unsig + reference_points_unsigmoid
        
        # Step 4: Apply sigmoid to get normalized coordinates
        if normalized_space:
            outputs = torch.sigmoid(outputs_unsig)
        else:
            outputs = outputs_unsig
        
        # Step 5: detached reference version (used for feeding next stage without backprop into ref)
        outputs_unsigmoid_detach = delta_unsig + reference_points_unsigmoid.detach()

        return outputs, outputs_unsigmoid_detach


class BboxMaskInitialization(nn.Module):
    def __init__(self, fast_bbox=True, threshold=0.0):
        """
        Args:
            fast_bbox: whether to use the faster or slower (and more accurate)
                implementation.
            threshold: the threshold used to determine whether a point is inside
                or outside the mask. Should be 0 if the masks are logits
                or 0.5 if the masks are probabilities.
        """
        super().__init__()
        self.fast_bbox = fast_bbox
        self.threshold = threshold

    def forward(self, x=None, reference_points=None, masks=None, normalized_space=True):
        """
        Args:
            masks: Masks associated with the queries
                shape: (batch_size, num_queries, height, width)
            x: Query embeddings. Not used here
                shape: (batch_size, num_queries, hidden_dim)
            reference_points: Current reference points. Not used here.
                shape: (batch_size, num_queries, 4)
        
        Returns:
            refpoint_embed: The reference points of the queries
                shape: (batch_size, num_queries, 4)
        """
        # Step 1: Flatten the batches and queries
        B,Q,_,_ = masks.shape

        flaten_mask = masks.detach().flatten(0, 1) # (B*Q, H,W)

        # Step 2: Obtain the raw (x,y,x,y) boxes with the given method.
        if self.fast_bbox:
            outputs_sigmoid = masks_to_boxes(flaten_mask > self.threshold).to(masks.device)
        else:
            outputs_sigmoid = BitMasks(flaten_mask > self.threshold).get_bounding_boxes().tensor.to(masks.device)

        # Step 3: Reescale and prepare for output.
        h, w = masks.shape[-2:]
        sizes = torch.as_tensor([w, h, w, h], dtype=torch.float).to(masks.device)
        outputs_sigmoid = box_xyxy_to_cxcywh(outputs_sigmoid) / sizes
        outputs_sigmoid = outputs_sigmoid.reshape(B, Q, 4)

        # Step 4: If not in normalized space, inverse sigmoid.
        if normalized_space:
            outputs = outputs_sigmoid
            outputs_unsigmoid = inverse_sigmoid(outputs_sigmoid)
        else:
            outputs = inverse_sigmoid(outputs_sigmoid)
            outputs_unsigmoid = outputs

        # Step 5: Reshape
        #outputs = outputs.transpose(0, 1) # (Q, B, 4)
        #outputs_unsigmoid = outputs_unsigmoid.transpose(0, 1) # (Q, B, 4)

        return outputs, outputs_unsigmoid


class BboxMaskSTN(nn.Module):
    def __init__(self, pooling='mean', learn_format="cxcywh"):
        """
        Args:
            pooling: The pooling type for the localization network. The
                mean type is easier to learn.
            learn_format: Whether to learn directly the (cx,cy,w,h) format
                or to learn the (x,y,x,y) format. The default one is easier
                to learn.
        """
        super().__init__()
        # Spatial transformer localization-network
        self.learn_format = learn_format
        assert learn_format in ["cxcywh", "xyxy"]
        assert pooling in ["max", "mean"], f"Unsupported pooling type: {pooling}. Choose 'max' or 'mean'."
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 16, kernel_size=5),
            nn.ReLU(True),
            nn.AdaptiveMaxPool2d(1) if pooling == 'max' else nn.AdaptiveAvgPool2d(1),
            nn.Flatten(-3, -1)
        )
        # Regressor for the (cx,cy,h,w) box
        self.fc_loc = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(True),
            nn.Linear(64, 4)
        )

    def forward(self, x=None, reference_points=None, masks=None, normalized_space=True):
        """
        Args:
            masks: Masks associated with the queries
                shape: (batch_size, num_queries, height, width)
        Returns:
            refpoint_embed: The reference points of the queries in [cx, cy, w, h] format
                shape: (batch_size, num_queries, 4)
        """
        # Step 1: Flatten the batches and queries
        B, Q, _, _ = masks.shape
        flaten_mask = masks.detach().flatten(0, 1).unsqueeze(1)  # (B*Q, 1, H, W)

        # Step 2: Rescale masks to fixed size of 32x32
        flaten_mask = F.interpolate(
            flaten_mask, 
            size=(32, 32), 
            mode='bilinear', 
            align_corners=False
        )  # (B*Q, 1, 32, 32)

        # Step 3: Obtain the raw box embeddings
        box_embed = self.localization(flaten_mask)  # (B*Q, 16)
        outputs_unsigmoid = self.fc_loc(box_embed)  # (B*Q, [cx, cy, w, h])

        # Step 4: Prepare for output.            
        outputs_sigmoid = outputs_unsigmoid.sigmoid().reshape(B, Q, 4)
        if self.learn_format == "xyxy":
            outputs_sigmoid = box_xyxy_to_cxcywh(outputs_sigmoid)
        
        if normalized_space:
            outputs = outputs_sigmoid
            outputs_unsigmoid = inverse_sigmoid(outputs_sigmoid)
        else:
            outputs = inverse_sigmoid(outputs_sigmoid)
            outputs_unsigmoid = outputs

        # Step 5: Reshape
        outputs = outputs.transpose(0, 1) # (Q, B, 4)
        outputs_unsigmoid = outputs_unsigmoid.transpose(0, 1) # (Q, B, 4)

        return outputs, outputs_unsigmoid
