"""
Utility evaluator that stores model predictions, ground truth targets, and
pairwise matching costs for further offline analysis.
"""

import os
from typing import Any, Dict, List, Optional

import torch
from detectron2.evaluation.evaluator import DatasetEvaluator


class MaskPredictionExporter(DatasetEvaluator):
    """Collect predictions and ground-truth targets for offline inspection."""

    def __init__(self, output_dir: Optional[str] = None) -> None:
        self._output_dir = output_dir
        self._records: List[Dict[str, Any]] = []

    def reset(self) -> None:
        self._records = []

    def process(self, inputs: List[Dict[str, Any]], outputs: List[Dict[str, Any]]) -> None:
        for input_dict, output_dict in zip(inputs, outputs):
            analysis = output_dict.get("analysis")
            if analysis is None:
                continue

            record: Dict[str, Any] = {
                "image_id": analysis.get("image_id", input_dict.get("image_id")),
                "pred_logits": analysis.get("pred_logits"),
                "pred_masks": analysis.get("pred_masks"),
                "pairwise_costs": analysis.get("pairwise_costs"),
                "gt_classes": analysis.get("gt_classes"),
                "gt_masks": analysis.get("gt_masks"),
                "gt_boxes": analysis.get("gt_boxes"),
            }

            # Ensure tensors are on CPU for serialization.
            for key in ["pred_logits", "pred_masks", "gt_classes", "gt_masks", "gt_boxes"]:
                value = record.get(key)
                if isinstance(value, torch.Tensor):
                    record[key] = value.cpu()

            pairwise_costs = record.get("pairwise_costs")
            if isinstance(pairwise_costs, dict):
                record["pairwise_costs"] = {
                    cost_name: tensor.cpu() if isinstance(tensor, torch.Tensor) else tensor
                    for cost_name, tensor in pairwise_costs.items()
                }

            self._records.append(record)

    def evaluate(self) -> Dict[str, Any]:
        if self._output_dir and len(self._records) > 0:
            os.makedirs(self._output_dir, exist_ok=True)
            torch.save(self._records, os.path.join(self._output_dir, "analysis_outputs.pth"))
        return {}
