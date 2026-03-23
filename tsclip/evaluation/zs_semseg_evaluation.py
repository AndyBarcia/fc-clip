"""
This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates.

Reference: https://github.com/facebookresearch/detectron2/blob/main/detectron2/evaluation/sem_seg_evaluation.py
"""

import itertools
import json
import logging
import os
from collections import OrderedDict

import numpy as np
import torch
from tabulate import tabulate

from detectron2.data import MetadataCatalog
from detectron2.evaluation import SemSegEvaluator
from detectron2.evaluation.sem_seg_evaluation import load_image_into_numpy_array
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.file_io import PathManager


class ZSSemSegEvaluator(SemSegEvaluator):
    """Evaluate semantic segmentation metrics on seen and unseen subsets,
    and optionally save confusion-analysis artifacts.

    Important:
        Detectron2 confusion matrix is indexed as:
            conf_matrix[pred_class, gt_class]
    """

    def __init__(
        self,
        dataset_name,
        distributed=True,
        output_dir=None,
        *,
        sem_seg_loading_fn=load_image_into_numpy_array,
        num_classes=None,
        ignore_label=None,
        confusion_topk=5,
        confusion_log_classes=30,
        confusion_log_pairs=30,
        save_confusion_matrix=True,
        save_confusion_analysis=True,
    ):
        super().__init__(
            dataset_name,
            distributed=distributed,
            output_dir=output_dir,
            sem_seg_loading_fn=sem_seg_loading_fn,
            num_classes=num_classes,
            ignore_label=ignore_label,
        )
        self._logger = logging.getLogger(__name__)
        meta = MetadataCatalog.get(dataset_name)

        unseen_mapping = getattr(meta, "unseen_dataset_id_to_contiguous_id", None)
        self._unseen_contiguous_ids = (
            set(unseen_mapping.values()) if unseen_mapping else None
        )

        superclass_mapping = getattr(meta, "superclass_dataset_id_to_contiguous_id", None)
        self._superclass_contiguous_ids = (
            set(superclass_mapping.values()) if superclass_mapping else None
        )

        subclass_mapping = getattr(meta, "subclass_dataset_id_to_contiguous_id", None)
        self._subclass_contiguous_ids = (
            set(subclass_mapping.values()) if subclass_mapping else None
        )

        self._confusion_topk = confusion_topk
        self._confusion_log_classes = confusion_log_classes
        self._confusion_log_pairs = confusion_log_pairs
        self._save_confusion_matrix = save_confusion_matrix
        self._save_confusion_analysis = save_confusion_analysis

    def _build_subset_mask(self, contiguous_ids):
        if contiguous_ids is None:
            return None
        mask = np.zeros(self._num_classes, dtype=bool)
        valid_ids = [idx for idx in contiguous_ids if 0 <= idx < self._num_classes]
        if len(valid_ids) > 0:
            mask[valid_ids] = True
        return mask

    def _get_subset_specs(self):
        subset_specs = []

        unseen_mask = self._build_subset_mask(self._unseen_contiguous_ids)
        if unseen_mask is not None and np.any(unseen_mask):
            seen_mask = ~unseen_mask
            if np.any(seen_mask):
                subset_specs.append(("Seen", "se", seen_mask))
            subset_specs.append(("Unseen", "un", unseen_mask))

        superclass_mask = self._build_subset_mask(self._superclass_contiguous_ids)
        if superclass_mask is not None and np.any(superclass_mask):
            subset_specs.append(("Superclass", "superclass", superclass_mask))

        subclass_mask = self._build_subset_mask(self._subclass_contiguous_ids)
        if subclass_mask is not None and np.any(subclass_mask):
            subset_specs.append(("Subclass", "subclass", subclass_mask))

        return subset_specs

    def _add_subset_results(self, res, table_rows, acc, iou, tp, pos_gt, pos_pred):
        for display_name, suffix, subset_mask in self._get_subset_specs():
            metrics = self._compute_subset_metrics(
                acc, iou, tp, pos_gt, pos_pred, subset_mask
            )

            res[f"mIoU_{suffix}"] = 100 * metrics["mIoU"]
            res[f"fwIoU_{suffix}"] = 100 * metrics["fwIoU"]
            res[f"mACC_{suffix}"] = 100 * metrics["mACC"]
            res[f"pACC_{suffix}"] = 100 * metrics["pACC"]

            table_rows.append(
                [
                    display_name,
                    100 * metrics["mIoU"],
                    100 * metrics["fwIoU"],
                    100 * metrics["mACC"],
                    100 * metrics["pACC"],
                ]
            )

    @staticmethod
    def _compute_subset_metrics(acc, iou, tp, pos_gt, pos_pred, subset_mask):
        subset_pos_gt = pos_gt[subset_mask]
        subset_pos_pred = pos_pred[subset_mask]
        subset_tp = tp[subset_mask]
        subset_acc = acc[subset_mask]
        subset_iou = iou[subset_mask]

        acc_valid = subset_pos_gt > 0
        union = subset_pos_gt + subset_pos_pred - subset_tp
        iou_valid = np.logical_and(acc_valid, union > 0)

        macc = (
            np.sum(subset_acc[acc_valid]) / np.sum(acc_valid)
            if np.sum(acc_valid) > 0
            else 0.0
        )
        miou = (
            np.sum(subset_iou[iou_valid]) / np.sum(iou_valid)
            if np.sum(iou_valid) > 0
            else 0.0
        )

        total_pos_gt = np.sum(subset_pos_gt)
        if total_pos_gt > 0:
            class_weights = subset_pos_gt / total_pos_gt
            fiou = np.sum(subset_iou[iou_valid] * class_weights[iou_valid])
            pacc = np.sum(subset_tp) / total_pos_gt
        else:
            fiou = 0.0
            pacc = 0.0

        return {
            "mIoU": miou,
            "fwIoU": fiou,
            "mACC": macc,
            "pACC": pacc,
        }

    @staticmethod
    def _safe_div(num, den):
        return float(num) / float(den) if den > 0 else 0.0

    @staticmethod
    def _float_or_none(x):
        if x is None:
            return None
        if isinstance(x, (np.floating, float)) and np.isnan(x):
            return None
        return float(x)

    def _build_confusion_analysis(self, conf_matrix, acc, iou):
        """
        Args:
            conf_matrix: [C, C] array with indexing [pred, gt], ignore label removed.

        Returns:
            Dict with:
                - per_class: detailed confusion analysis
                - worst_rows: rows for logging
                - global_top_pairs: largest off-diagonal confusions
        """
        conf_matrix = conf_matrix.astype(np.int64, copy=False)

        tp = np.diag(conf_matrix).astype(np.int64)
        gt_pixels = conf_matrix.sum(axis=0).astype(np.int64)
        pred_pixels = conf_matrix.sum(axis=1).astype(np.int64)

        per_class = []
        worst_rows = []

        for cls_idx, cls_name in enumerate(self._class_names):
            gt_count = int(gt_pixels[cls_idx])
            pred_count = int(pred_pixels[cls_idx])
            tp_count = int(tp[cls_idx])

            # For this GT class, where did its pixels go?
            # Column cls_idx = fixed GT, rows = predicted classes.
            gt_to_pred = conf_matrix[:, cls_idx].copy()
            gt_to_pred[cls_idx] = 0
            top_pred_idx = np.argsort(-gt_to_pred)[: self._confusion_topk]

            top_confused_as = []
            for pred_idx in top_pred_idx:
                count = int(gt_to_pred[pred_idx])
                if count <= 0:
                    continue
                top_confused_as.append(
                    {
                        "pred_id": int(pred_idx),
                        "pred_class": self._class_names[pred_idx],
                        "count": count,
                        "gt_fraction": self._safe_div(count, gt_count),
                        "pred_fraction": self._safe_div(count, pred_pixels[pred_idx]),
                    }
                )

            # For this predicted class, where do its false positives come from?
            # Row cls_idx = fixed prediction, columns = GT classes.
            pred_from_gt = conf_matrix[cls_idx, :].copy()
            pred_from_gt[cls_idx] = 0
            top_gt_idx = np.argsort(-pred_from_gt)[: self._confusion_topk]

            top_false_positive_sources = []
            for gt_idx in top_gt_idx:
                count = int(pred_from_gt[gt_idx])
                if count <= 0:
                    continue
                top_false_positive_sources.append(
                    {
                        "gt_id": int(gt_idx),
                        "gt_class": self._class_names[gt_idx],
                        "count": count,
                        "pred_fraction": self._safe_div(count, pred_count),
                        "gt_fraction": self._safe_div(count, gt_pixels[gt_idx]),
                    }
                )

            iou_val = self._float_or_none(iou[cls_idx])
            acc_val = self._float_or_none(acc[cls_idx])

            per_class.append(
                {
                    "class_id": int(cls_idx),
                    "class_name": cls_name,
                    "gt_pixels": gt_count,
                    "pred_pixels": pred_count,
                    "true_positive_pixels": tp_count,
                    "acc": acc_val,
                    "iou": iou_val,
                    "top_confused_as": top_confused_as,
                    "top_false_positive_sources": top_false_positive_sources,
                }
            )

            if gt_count > 0:
                if len(top_confused_as) > 0:
                    top1 = top_confused_as[0]
                    top1_pred = top1["pred_class"]
                    top1_count = top1["count"]
                    top1_rate = 100.0 * top1["gt_fraction"]
                else:
                    top1_pred = "-"
                    top1_count = 0
                    top1_rate = 0.0

                worst_rows.append(
                    [
                        int(cls_idx),
                        cls_name,
                        gt_count,
                        float("nan") if acc_val is None else 100.0 * acc_val,
                        float("nan") if iou_val is None else 100.0 * iou_val,
                        top1_pred,
                        top1_count,
                        top1_rate,
                    ]
                )

        # Largest global off-diagonal entries.
        offdiag = conf_matrix.copy()
        np.fill_diagonal(offdiag, 0)

        flat_sorted = np.argsort(offdiag, axis=None)[::-1]
        global_top_pairs = []
        max_pairs = max(self._confusion_log_pairs, 100)

        flat_view = offdiag.reshape(-1)
        for flat_idx in flat_sorted:
            count = int(flat_view[flat_idx])
            if count <= 0:
                break
            pred_idx, gt_idx = np.unravel_index(flat_idx, offdiag.shape)
            global_top_pairs.append(
                {
                    "pred_id": int(pred_idx),
                    "pred_class": self._class_names[pred_idx],
                    "gt_id": int(gt_idx),
                    "gt_class": self._class_names[gt_idx],
                    "count": count,
                    "gt_fraction": self._safe_div(count, gt_pixels[gt_idx]),
                    "pred_fraction": self._safe_div(count, pred_pixels[pred_idx]),
                }
            )
            if len(global_top_pairs) >= max_pairs:
                break

        # Sort worst classes by IoU ascending, then by gt_pixels descending.
        worst_rows.sort(key=lambda x: (np.inf if np.isnan(x[4]) else x[4], -x[2]))

        return {
            "per_class": per_class,
            "worst_rows": worst_rows,
            "global_top_pairs": global_top_pairs,
        }

    def _save_and_log_confusion_analysis(self, acc, iou):
        """
        Save raw confusion artifacts and log useful summary tables.

        Saves:
            - sem_seg_conf_matrix.npy
            - sem_seg_conf_matrix_gt_norm.npy
            - sem_seg_confusion_analysis.json
        """
        conf_matrix = self._conf_matrix[:-1, :-1].copy()  # remove ignore row/col
        analysis = self._build_confusion_analysis(conf_matrix, acc, iou)

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)

            if self._save_confusion_matrix:
                raw_path = os.path.join(self._output_dir, "sem_seg_conf_matrix.npy")
                with PathManager.open(raw_path, "wb") as f:
                    np.save(f, conf_matrix)

                # Normalize by GT count (columns), so each column sums to 1 when gt_pixels > 0.
                gt_norm = conf_matrix.astype(np.float64)
                gt_den = gt_norm.sum(axis=0, keepdims=True)
                np.divide(gt_norm, gt_den, out=gt_norm, where=gt_den > 0)

                gt_norm_path = os.path.join(
                    self._output_dir, "sem_seg_conf_matrix_gt_norm.npy"
                )
                with PathManager.open(gt_norm_path, "wb") as f:
                    np.save(f, gt_norm)

            if self._save_confusion_analysis:
                json_path = os.path.join(
                    self._output_dir, "sem_seg_confusion_analysis.json"
                )
                payload = {
                    "note": "Confusion matrix indexing is [pred_class, gt_class].",
                    "topk": int(self._confusion_topk),
                    "per_class": analysis["per_class"],
                    "global_top_pairs": analysis["global_top_pairs"],
                }
                with PathManager.open(json_path, "w") as f:
                    f.write(json.dumps(payload, indent=2))

        # Log worst classes by IoU.
        if self._confusion_log_classes > 0 and len(analysis["worst_rows"]) > 0:
            rows = analysis["worst_rows"][: self._confusion_log_classes]
            headers = [
                "id",
                "class",
                "gt_px",
                "ACC",
                "IoU",
                "top1 confused as",
                "top1 px",
                "top1 %gt",
            ]
            table = tabulate(
                rows,
                headers=headers,
                tablefmt="pipe",
                floatfmt=".3f",
                stralign="left",
                numalign="right",
            )
            self._logger.info(
                "Worst classes by IoU "
                "(top1 confused as = most common predicted class for that GT class):\n"
                + table
            )

        # Log largest global confusions.
        if self._confusion_log_pairs > 0 and len(analysis["global_top_pairs"]) > 0:
            rows = []
            for item in analysis["global_top_pairs"][: self._confusion_log_pairs]:
                rows.append(
                    [
                        item["gt_id"],
                        item["gt_class"],
                        item["pred_id"],
                        item["pred_class"],
                        item["count"],
                        100.0 * item["gt_fraction"],
                    ]
                )
            headers = ["gt_id", "gt_class", "pred_id", "pred_class", "pixels", "% of gt"]
            table = tabulate(
                rows,
                headers=headers,
                tablefmt="pipe",
                floatfmt=".3f",
                stralign="left",
                numalign="right",
            )
            self._logger.info(
                "Largest global off-diagonal confusions "
                "(GT class -> predicted class):\n" + table
            )

    def evaluate(self):
        if self._distributed:
            synchronize()
            conf_matrix_list = all_gather(self._conf_matrix)
            b_conf_matrix_list = all_gather(self._b_conf_matrix)
            self._predictions = all_gather(self._predictions)
            self._predictions = list(itertools.chain(*self._predictions))
            if not is_main_process():
                return

            self._conf_matrix = np.zeros_like(self._conf_matrix)
            for conf_matrix in conf_matrix_list:
                self._conf_matrix += conf_matrix

            self._b_conf_matrix = np.zeros_like(self._b_conf_matrix)
            for b_conf_matrix in b_conf_matrix_list:
                self._b_conf_matrix += b_conf_matrix

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "sem_seg_predictions.json")
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(self._predictions))

        acc = np.full(self._num_classes, np.nan, dtype=float)
        iou = np.full(self._num_classes, np.nan, dtype=float)

        tp = self._conf_matrix.diagonal()[:-1].astype(float)
        pos_gt = np.sum(self._conf_matrix[:-1, :-1], axis=0).astype(float)
        class_weights = pos_gt / np.sum(pos_gt)
        pos_pred = np.sum(self._conf_matrix[:-1, :-1], axis=1).astype(float)

        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]

        union = pos_gt + pos_pred - tp
        iou_valid = np.logical_and(acc_valid, union > 0)
        iou[iou_valid] = tp[iou_valid] / union[iou_valid]

        macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)
        miou = np.sum(iou[iou_valid]) / np.sum(iou_valid)
        fiou = np.sum(iou[iou_valid] * class_weights[iou_valid])
        pacc = np.sum(tp) / np.sum(pos_gt)

        if self._compute_boundary_iou:
            b_iou = np.full(self._num_classes, np.nan, dtype=float)
            b_tp = self._b_conf_matrix.diagonal()[:-1].astype(float)
            b_pos_gt = np.sum(self._b_conf_matrix[:-1, :-1], axis=0).astype(float)
            b_pos_pred = np.sum(self._b_conf_matrix[:-1, :-1], axis=1).astype(float)
            b_union = b_pos_gt + b_pos_pred - b_tp
            b_iou_valid = b_union > 0
            b_iou[b_iou_valid] = b_tp[b_iou_valid] / b_union[b_iou_valid]

        res = {
            "mIoU": 100 * miou,
            "fwIoU": 100 * fiou,
        }
        for i, name in enumerate(self._class_names):
            res[f"IoU-{name}"] = 100 * iou[i]
            if self._compute_boundary_iou:
                res[f"BoundaryIoU-{name}"] = 100 * b_iou[i]
                res[f"min(IoU, B-Iou)-{name}"] = 100 * min(iou[i], b_iou[i])

        res["mACC"] = 100 * macc
        res["pACC"] = 100 * pacc
        for i, name in enumerate(self._class_names):
            res[f"ACC-{name}"] = 100 * acc[i]

        headers = ["", "mIoU", "fwIoU", "mACC", "pACC"]
        data = [
            ["All", 100 * miou, 100 * fiou, 100 * macc, 100 * pacc],
        ]

        self._add_subset_results(res, data, acc, iou, tp, pos_gt, pos_pred)

        table = tabulate(
            data,
            headers=headers,
            tablefmt="pipe",
            floatfmt=".3f",
            stralign="center",
            numalign="center",
        )
        self._logger.info("Semantic Segmentation Evaluation Results:\n" + table)

        self._save_and_log_confusion_analysis(acc, iou)

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "sem_seg_evaluation.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(res, f)

        results = OrderedDict({"sem_seg": res})
        self._logger.info(results)
        return results
