#!/usr/bin/env python
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import numpy as np
import json
import itertools
import logging
from typing import Optional

import PIL.Image as Image
from panopticapi.utils import IdGenerator, rgb2id

from detectron2.data import MetadataCatalog
from detectron2.utils import comm
from detectron2.utils.file_io import PathManager
from detectron2.evaluation.evaluator import DatasetEvaluator

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from scipy.optimize import linear_sum_assignment

from ..modeling.matcher import batch_ce_loss, batch_dice_loss
from ..modeling.mask_utils import compute_mask_block_counts

logger = logging.getLogger(__name__)

OFFSET = 256 * 256 * 256
VOID = 0


class ExportEvaluator(DatasetEvaluator):
    """
    Export raw detections + panoptic output with detailed matching info:
      - raw_detection_matching:
          * pairwise classification / mask / dice costs [Q x M]
          * Hungarian assignment query -> GT index.
      - panoptic_matching:
          * pairwise IoU [#GT x #Pred]
          * Hungarian assignment on IoU (GT/pred IDs + IoU).
      - panoptic IDs (map + segments_info) are remapped so that matched
        predicted segments get the GT segment IDs.
    """

    def __init__(
        self,
        dataset_name: str,
        output_dir: Optional[str] = None,
        cost_class: float = 1.0,
        cost_mask: float = 1.0,
        cost_dice: float = 1.0,
    ):
        """
        Args:
            dataset_name: name of the dataset
            output_dir: output directory to save results for evaluation.
            cost_*: matching cost weights; set to the same values as your
                    HungarianMatcher to get identical assignments.
        """
        self._metadata = MetadataCatalog.get(dataset_name)
        self._thing_contiguous_id_to_dataset_id = {
            v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
        }
        self._stuff_contiguous_id_to_dataset_id = {
            v: k for k, v in self._metadata.stuff_dataset_id_to_contiguous_id.items()
        }

        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

        self.label_divisor = self._metadata.label_divisor

        self._output_dir = output_dir
        if self._output_dir is not None:
            PathManager.mkdirs(self._output_dir)
            PathManager.mkdirs(os.path.join(self._output_dir, "images"))

        # Load panoptic GT annotations (required; if missing, we fail)
        gt_json_path = PathManager.get_local_path(self._metadata.panoptic_json)
        gt_root = PathManager.get_local_path(self._metadata.panoptic_root)
        with PathManager.open(gt_json_path, "r") as f:
            gt_json = json.load(f)
        self._gt_ann_by_image_id = {
            ann["image_id"]: ann for ann in gt_json["annotations"]
        }
        self._gt_panoptic_root = gt_root

        logger.info(
            f"Loaded panoptic GT annotations from {self._metadata.panoptic_json}"
        )

        self.reset()

    def reset(self):
        self._predictions = []


    @staticmethod
    def _get_cropped_mask_size(input_dict, out_mask: torch.Tensor):
        h, w = input_dict["instances"].image_size

        def _round_up(x, m=32):
            return (x + m - 1) // m * m

        # 2) Round up to multiple of 32
        h_32 = _round_up(int(h), 32)
        w_32 = _round_up(int(w), 32)

        # 3) Convert to mask-logit resolution (downsample factor 4)
        h_mask = h_32 // 4
        w_mask = w_32 // 4

        # 4) Clamp to actual mask logits size (ignore any extra batch padding)
        H_pred, W_pred = out_mask.shape[-2:]
        h_mask = min(h_mask, H_pred)
        w_mask = min(w_mask, W_pred)

        return h_mask, w_mask

    # ------------------------------------------------------------------
    # Raw detection matching (Hungarian + per-query costs)
    # ------------------------------------------------------------------
    def _compute_detection_matching(self, input_dict, cls_logits, mask_logits):
        """
        Compute pairwise classification/mask/dice costs between each query and
        each GT instance, using the same full-resolution block-count scheme
        as the HungarianMatcher, and run Hungarian on the combined cost.

        Returns:
            detection_dbg: dict
        """
        device = cls_logits.device
        instances = input_dict["instances"].to(device)

        gt_labels = instances.gt_classes  # [M]
        num_queries = cls_logits.shape[0]
        num_gt = int(gt_labels.numel())

        # No GT instances: costs are empty, all queries unmatched.
        if num_gt == 0:
            return {
                "gt_labels": [],
                "cost_class": [],
                "cost_mask": [],
                "cost_dice": [],
                "hungarian_query_to_gt": [-1] * num_queries,
                "num_queries": num_queries,
                "num_gt": 0,
            }

        # Classification cost: negative log prob of GT class
        out_prob = cls_logits.softmax(-1)          # [Q, C]
        cost_class = -out_prob[:, gt_labels]       # [Q, M]

        # Mask-related costs using full-res GT + compute_mask_block_counts
        out_mask = mask_logits                      # [Q, H_pred, W_pred]
        tgt_mask = instances.gt_masks.to(out_mask)  # [M, H_gt, W_gt]

        # Pad to a size divisible by 32 to make size compatible with
        # cropped mask logits.
        H_gt, W_gt = tgt_mask.shape[-2:]
        pad_h = (32 - H_gt % 32) % 32
        pad_w = (32 - W_gt % 32) % 32
        tgt_mask = F.pad(tgt_mask, (0, pad_w, 0, pad_h), mode='constant', value=0)

        target_counts, block_area, H_t, W_t = compute_mask_block_counts(
            tgt_mask, out_mask.shape[-2:]
        )

        out_mask_flat = out_mask.flatten(1)  # [Q, N_blocks]
        target_counts = target_counts.to(
            device=out_mask_flat.device, dtype=out_mask_flat.dtype
        )

        with autocast(enabled=False):
            out_mask_float = out_mask_flat.float()
            target_counts_float = target_counts.float()
            cost_mask = batch_ce_loss(
                out_mask_float, target_counts_float, block_area, H_t * W_t
            )   # [Q, M]
            cost_dice = batch_dice_loss(
                out_mask_float, target_counts_float, block_area
            )   # [Q, M]

        # Combined cost for Hungarian
        total_cost = (
            self.cost_class * cost_class
            + self.cost_mask * cost_mask
            + self.cost_dice * cost_dice
        )

        C = total_cost.detach().cpu().numpy()  # [Q, M]
        hungarian_query_to_gt = [-1] * num_queries
        if C.size > 0:
            row_ind, col_ind = linear_sum_assignment(C)
            for q_idx, g_idx in zip(row_ind.tolist(), col_ind.tolist()):
                hungarian_query_to_gt[q_idx] = int(g_idx)

        detection_dbg = {
            "gt_labels": gt_labels.tolist(),                             # [M]
            "cost_class": cost_class.detach().cpu().tolist(),            # [Q, M]
            "cost_mask": cost_mask.detach().cpu().tolist(),              # [Q, M]
            "cost_dice": cost_dice.detach().cpu().tolist(),              # [Q, M]
            "hungarian_query_to_gt": hungarian_query_to_gt,              # [Q], -1 = unmatched
            "num_queries": num_queries,
            "num_gt": num_gt,
        }
        return detection_dbg

    def _compute_panoptic_matching_and_remap(self, image_id, panoptic_img, segments_info):
        """
        Compute pairwise IoU between predicted and GT panoptic segments,
        run Hungarian on IoU, and then:

          * For each predicted segment, compute a new panoptic id as:
                panoptic_label = pred_class * label_divisor + instance_id
            where pred_class is the predicted category_id and instance_id is
            an integer instance index (0 for stuff, 1, 2, ... for thing
            instances of the same class).

          * In segments_info, add 'gt_id' with the id of the matched GT
            segment (or -1 if unmatched).

        Args:
            image_id: current image id
            panoptic_img: np.ndarray (H, W), int predicted IDs
            segments_info: list of predicted segments_info dicts

        Returns:
            panoptic_img_remapped: np.ndarray (H, W) with encoded ids
            segments_info_remapped: list of dicts (id, instance_id, gt_id, ...)
            panoptic_dbg: dict
        """
        # Remap predicted IDs to encoded panoptic labels
        # panoptic_label = pred_class * label_divisor + instance_id
        panoptic_img_remapped = np.full_like(panoptic_img, VOID)
        segments_info_remapped = []
        old_det_id_to_new_inst_id = {}

        # Get generator for the output colors.
        categories_dict = { i:{
            "id": i,
            "isthing": i in self._metadata.thing_dataset_id_to_contiguous_id.values(),
            "color": color
        } for i,color in enumerate(self._metadata.stuff_colors) }
        id_generator = IdGenerator(categories_dict)

        for s in segments_info:
            s = dict(s)
            old_id = int(s["id"])
            category_id = int(s["category_id"])

            panoptic_label = id_generator.get_id(category_id)
            panoptic_img_remapped[panoptic_img == old_id] = panoptic_label

            # Update segment info
            s["id"] = int(panoptic_label)
            segments_info_remapped.append(s)
            old_det_id_to_new_inst_id[old_id] = panoptic_label

        gt_ann = self._gt_ann_by_image_id[image_id]

        # Load GT panoptic RGB and convert to ids
        gt_png_path = os.path.join(self._gt_panoptic_root, gt_ann["file_name"])
        pan_gt = rgb2id(
            np.array(Image.open(gt_png_path), dtype=np.uint32)
        )  # (H, W) GT ids

        gt_segms = {s["id"]: dict(s) for s in gt_ann["segments_info"]}
        pred_segms = {s["id"]: dict(s) for s in segments_info}

        # Areas for predicted segments
        labels, counts = np.unique(panoptic_img, return_counts=True)
        for lbl, cnt in zip(labels, counts):
            if lbl == VOID:
                continue
            if lbl not in pred_segms:
                # Same behaviour as the panoptic PQ code: fail loudly.
                raise KeyError(
                    f"Segment {lbl} in PNG not in JSON for image {image_id}"
                )
            pred_segms[lbl]["area"] = int(cnt)

        # Combined map to get intersections
        combined = pan_gt.astype(np.uint64) * OFFSET + panoptic_img.astype(np.uint64)
        pairs, intsct = np.unique(combined, return_counts=True)
        gt_pred_map = {(int(p // OFFSET), int(p % OFFSET)): int(c)
                       for p, c in zip(pairs, intsct)}

        gt_ids = sorted(gt_segms.keys())
        pred_ids = sorted(pred_segms.keys())

        num_gt = len(gt_ids)
        num_pred = len(pred_ids)

        # IoU matrix [num_gt, num_pred]
        pairwise_iou = np.zeros((num_gt, num_pred), dtype=np.float32)

        for gi, gt_id in enumerate(gt_ids):
            g = gt_segms[gt_id]
            for pj, pred_id in enumerate(pred_ids):
                p = pred_segms[pred_id]
                inter = gt_pred_map.get((gt_id, pred_id), 0)
                if inter == 0:
                    continue
                union = (
                    p.get("area", 0)
                    + g.get("area", 0)
                    - inter
                    - gt_pred_map.get((VOID, pred_id), 0)
                )
                if union > 0:
                    pairwise_iou[gi, pj] = inter / float(union)

        # Hungarian on IoU (maximize IoU -> minimize 1 - IoU)
        panoptic_dbg_matches = []
        pred_to_gt = {}  # pred segment id -> matched gt id (if any)

        if num_gt > 0 and num_pred > 0:
            cost = 1.0 - pairwise_iou
            row_ind, col_ind = linear_sum_assignment(cost)
            for gi, pj in zip(row_ind.tolist(), col_ind.tolist()):
                iou_val = float(pairwise_iou[gi, pj])
                gt_id = int(gt_ids[gi])
                pred_id = old_det_id_to_new_inst_id[int(pred_ids[pj])]
                panoptic_dbg_matches.append(
                    {
                        "gt_index": gi,
                        "pred_index": pj,
                        "gt_id": gt_id,
                        "pred_id": pred_id,
                        "iou": iou_val,
                    }
                )
                if iou_val > 0.0:
                    pred_to_gt[pred_id] = gt_id

        panoptic_dbg = {
            "gt_ids": [int(i) for i in gt_ids],
            "pred_ids": [old_det_id_to_new_inst_id[int(i)] for i in pred_ids],
            "pairwise_iou": pairwise_iou.tolist(),    # [num_gt][num_pred]
            "hungarian_matches": panoptic_dbg_matches
        }

        return panoptic_img_remapped, segments_info_remapped, panoptic_dbg

    def process(self, inputs, outputs):
        from panopticapi.utils import id2rgb

        for input, output in zip(inputs, outputs):
            image_id = input["image_id"]
            file_name = input["file_name"]

            # Raw logits (keep as tensors for cost computation)
            cls_logits = output["raw_cls"]          # [Q, C]
            mask_logits = output["raw_seg"]         # [Q, H, W]

            # Crop mask logits to ignore batch padding
            H_crop, W_crop = self._get_cropped_mask_size(input, mask_logits)
            mask_logits = mask_logits[..., :H_crop, :W_crop]   # [Q, H_eff, W_eff]

            # Probabilities for export
            raw_seg = mask_logits.sigmoid().cpu().numpy()      # (Q, H, W)
            raw_cls = cls_logits.softmax(dim=-1).cpu().numpy() # (Q, C)

            panoptic_img, segments_info = output["panoptic_seg"]
            panoptic_img = panoptic_img.cpu().numpy()

            # 1) Raw detection matching: pairwise costs + Hungarian
            detection_matching = self._compute_detection_matching(
                input, cls_logits, mask_logits
            )

            # 2) Panoptic IoU + Hungarian + ID remap
            (
                panoptic_img_remapped,
                segments_info_remapped,
                panoptic_matching,
            ) = self._compute_panoptic_matching_and_remap(
                image_id, panoptic_img, segments_info
            )

            # Save images
            image_dir = os.path.join(self._output_dir, "images", str(image_id))
            PathManager.mkdirs(image_dir)

            # panoptic_output.png with remapped IDs
            pan_rgb = id2rgb(panoptic_img_remapped)
            pan_image = Image.fromarray(pan_rgb)
            pan_image.save(os.path.join(image_dir, "panoptic_output.png"), optimize=True)

            # detection_{q}.png: sigmoid(raw_seg[q]) as grayscale
            """
            for q in range(raw_seg.shape[0]):
                prob_map = raw_seg[q]  # (H, W)
                prob_uint8 = np.clip(prob_map * 255.0, 0, 255).astype(np.uint8)
                det_img = Image.fromarray(prob_uint8, mode="L")
                det_img.save(os.path.join(image_dir, f"detection_{q}.png"), optimize=True)
            """
                
            record = {
                "image_id": image_id,
                "image_name": os.path.basename(file_name),
                #"cls_probs": raw_cls.tolist(),               # [Q, C]
                "segments_info": segments_info_remapped,     # IDs remapped to GT IDs where matched
                #"raw_detection_matching": detection_matching,
                "panoptic_matching": panoptic_matching,
            }
            self._predictions.append(record)

    def evaluate(self):
        # Gather from all workers
        all_predictions = comm.gather(self._predictions, dst=0)

        if not comm.is_main_process():
            return {}

        all_predictions = list(itertools.chain.from_iterable(all_predictions))

        output_path = os.path.join(self._output_dir, "detections.json")
        with PathManager.open(output_path, "w") as f:
            json.dump({"predictions": all_predictions}, f)
        logger.info(f"Saved detections json to {output_path}")

        # No metrics to report (this is just an export), so return an empty dict
        return {}
