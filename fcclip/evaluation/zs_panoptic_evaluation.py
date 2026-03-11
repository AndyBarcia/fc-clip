#!/usr/bin/env python
from __future__ import absolute_import, division, print_function, unicode_literals

import contextlib
import io
import logging
from tabulate import tabulate

from .panoptic_evaluation import COCOPanopticEvaluator, pq_compute

logger = logging.getLogger(__name__)


class COCOZSPanopticEvaluator(COCOPanopticEvaluator):
    _EXPORT_METRICS = [
        ("pq", "PQ"),
        ("sq", "SQ"),
        ("rq", "RQ"),
        ("pr", "PR"),
        ("re", "RE"),
        ("pqca", "PQca"),
        ("sqca", "SQca"),
        ("rqca", "RQca"),
        ("prca", "PRca"),
        ("reca", "REca"),
    ]

    @staticmethod
    def _safe_div(num, den):
        return num / den if den != 0 else 0

    def _get_metadata_dataset_id_set(self, attr_name):
        """
        Panoptic pq_res['per_class'] is keyed by dataset/category ids,
        so here we use mapping.keys(), not mapping.values().
        """
        mapping = getattr(self._metadata, attr_name, None)
        if not mapping:
            return None
        return {int(k) for k in mapping.keys()}

    def _get_subset_specs(self, per_class):
        """
        Returns:
            list of (display_name, result_suffix, class_id_set)
        """
        present_class_ids = {int(k) for k in per_class.keys()}
        subset_specs = []

        unseen_ids = self._get_metadata_dataset_id_set(
            "unseen_dataset_id_to_contiguous_id"
        )
        if unseen_ids:
            unseen_ids = present_class_ids & unseen_ids
            seen_ids = present_class_ids - unseen_ids
            if seen_ids:
                subset_specs.append(("Seen", "se", seen_ids))
            if unseen_ids:
                subset_specs.append(("Unseen", "un", unseen_ids))

        for display_name, suffix, attr_name in [
            ("TypeShifted", "ts", "typeshift_dataset_id_to_contiguous_id"),
            ("Superclass", "superclass", "superclass_dataset_id_to_contiguous_id"),
            ("Subclass", "subclass", "subclass_dataset_id_to_contiguous_id"),
        ]:
            class_ids = self._get_metadata_dataset_id_set(attr_name)
            if class_ids:
                class_ids = present_class_ids & class_ids
                if class_ids:
                    subset_specs.append((display_name, suffix, class_ids))

        return subset_specs

    def _aggregate_subset_pq(self, per_class, class_ids):
        pq = sq = rq = 0.0
        pqca = sqca = rqca = 0.0

        tp = fp = fn = 0.0
        tpca = fpca = fnca = 0.0

        n = 0

        for class_id, res in per_class.items():
            class_id = int(class_id)
            if class_id not in class_ids:
                continue

            n += 1

            pq += res["pq"]
            sq += res["sq"]
            rq += res["rq"]
            tp += res["tp"]
            fp += res["fp"]
            fn += res["fn"]

            pqca += res["pqca"]
            sqca += res["sqca"]
            rqca += res["rqca"]
            tpca += res["tpca"]
            fpca += res["fpca"]
            fnca += res["fnca"]

        return {
            "pq": self._safe_div(pq, n),
            "sq": self._safe_div(sq, n),
            "rq": self._safe_div(rq, n),
            "pr": self._safe_div(tp, tp + fp),
            "re": self._safe_div(tp, tp + fn),
            "pqca": self._safe_div(pqca, n),
            "sqca": self._safe_div(sqca, n),
            "rqca": self._safe_div(rqca, n),
            "prca": self._safe_div(tpca, tpca + fpca),
            "reca": self._safe_div(tpca, tpca + fnca),
            "n": n,
        }
    
    def _add_subset_metrics_to_pq_res(self, pq_res):
        per_class = pq_res["per_class"]
        for display_name, _, class_ids in self._get_subset_specs(per_class):
            pq_res[display_name] = self._aggregate_subset_pq(per_class, class_ids)

    def _add_subset_metrics_to_res(self, res, pq_res):
        for display_name, suffix, _ in self._get_subset_specs(pq_res["per_class"]):
            if display_name not in pq_res:
                continue
            subset_res = pq_res[display_name]
            for src_key, dst_prefix in self._EXPORT_METRICS:
                res[f"{dst_prefix}_{suffix}"] = 100 * subset_res[src_key]

    def _get_print_categories(self, pq_res):
        categories = ["All", "Things", "Stuff"]
        for name in ["Seen", "Unseen", "TypeShifted", "Superclass", "Subclass"]:
            if name in pq_res:
                categories.append(name)
        return categories

    def pq_compute_to_res_dic(self, pq_res):
        self._add_subset_metrics_to_pq_res(pq_res)
        res = super().pq_compute_to_res_dic(pq_res)
        self._add_subset_metrics_to_res(res, pq_res)
        return res

    def _print_panoptic_results(self, pq_res):
        categories = self._get_print_categories(pq_res)

        headers = ["", "PQ", "SQ", "RQ", "PR", "RE", "#categories"]
        data = []
        for name in categories:
            row = [name] + [
                pq_res[name][k] * 100
                for k in ["pq", "sq", "rq", "pr", "re"]
            ] + [pq_res[name]["n"]]
            data.append(row)
        table = tabulate(
            data,
            headers=headers,
            tablefmt="pipe",
            floatfmt=".3f",
            stralign="center",
            numalign="center",
        )
        logger.info("Panoptic Evaluation Results:\n" + table)

        data = []
        for name in categories:
            row = [name] + [
                pq_res[name][k] * 100
                for k in ["pqca", "sqca", "rqca", "prca", "reca"]
            ] + [pq_res[name]["n"]]
            data.append(row)
        table = tabulate(
            data,
            headers=headers,
            tablefmt="pipe",
            floatfmt=".3f",
            stralign="center",
            numalign="center",
        )
        logger.info("Panoptic Evaluation Results (Class-Agnostic):\n" + table)


if __name__ == "__main__":
    from detectron2.utils.logger import setup_logger

    logger = setup_logger()
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-json")
    parser.add_argument("--gt-dir")
    parser.add_argument("--pred-json")
    parser.add_argument("--pred-dir")
    args = parser.parse_args()

    with contextlib.redirect_stdout(io.StringIO()):
        pq_res = pq_compute(
            args.gt_json, args.pred_json, gt_folder=args.gt_dir, pred_folder=args.pred_dir
        )

    categories = ["All", "Things", "Stuff"]
    for extra in ["Seen", "Unseen", "TypeShifted", "Superclass", "Subclass"]:
        if extra in pq_res:
            categories.append(extra)

    headers = ["", "PQ", "SQ", "RQ", "PR", "RE", "#categories"]
    data = []
    for name in categories:
        row = [name] + [
            pq_res[name][k] * 100
            for k in ["pq", "sq", "rq", "pr", "re"]
        ] + [pq_res[name]["n"]]
        data.append(row)
    table = tabulate(
        data, headers=headers, tablefmt="pipe", floatfmt=".3f", stralign="center", numalign="center"
    )
    logger.info("Panoptic Evaluation Results:\n" + table)

    data = []
    for name in categories:
        row = [name] + [
            pq_res[name][k] * 100
            for k in ["pqca", "sqca", "rqca", "prca", "reca"]
        ] + [pq_res[name]["n"]]
        data.append(row)
    table = tabulate(
        data, headers=headers, tablefmt="pipe", floatfmt=".3f", stralign="center", numalign="center"
    )
    logger.info("Panoptic Evaluation Results (Class-Agnostic):\n" + table)
