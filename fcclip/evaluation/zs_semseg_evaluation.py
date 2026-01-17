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

from detectron2.data import MetadataCatalog
from detectron2.evaluation import SemSegEvaluator
from detectron2.evaluation.sem_seg_evaluation import load_image_into_numpy_array
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.file_io import PathManager


class ZSSemSegEvaluator(SemSegEvaluator):
    """Evaluate semantic segmentation metrics on seen and unseen subsets."""

    def __init__(
        self,
        dataset_name,
        distributed=True,
        output_dir=None,
        *,
        sem_seg_loading_fn=load_image_into_numpy_array,
        num_classes=None,
        ignore_label=None,
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
        if unseen_mapping:
            self._unseen_contiguous_ids = set(unseen_mapping.values())
        else:
            self._unseen_contiguous_ids = None

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

    def evaluate(self):
        if self._unseen_contiguous_ids is None:
            return super().evaluate()

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

        unseen_mask = np.zeros(self._num_classes, dtype=bool)
        unseen_indices = [idx for idx in self._unseen_contiguous_ids if idx < self._num_classes]
        unseen_mask[unseen_indices] = True
        seen_mask = ~unseen_mask

        seen_metrics = self._compute_subset_metrics(acc, iou, tp, pos_gt, pos_pred, seen_mask)
        unseen_metrics = self._compute_subset_metrics(acc, iou, tp, pos_gt, pos_pred, unseen_mask)

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

        res["mIoU_se"] = 100 * seen_metrics["mIoU"]
        res["fwIoU_se"] = 100 * seen_metrics["fwIoU"]
        res["mACC_se"] = 100 * seen_metrics["mACC"]
        res["pACC_se"] = 100 * seen_metrics["pACC"]
        res["mIoU_un"] = 100 * unseen_metrics["mIoU"]
        res["fwIoU_un"] = 100 * unseen_metrics["fwIoU"]
        res["mACC_un"] = 100 * unseen_metrics["mACC"]
        res["pACC_un"] = 100 * unseen_metrics["pACC"]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "sem_seg_evaluation.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(res, f)
        results = OrderedDict({"sem_seg": res})
        self._logger.info(results)
        return results
