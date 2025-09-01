#!/usr/bin/env python
from __future__ import absolute_import, division, print_function, unicode_literals

import contextlib
import io
import logging
import os
from collections import OrderedDict
from tabulate import tabulate

# Assuming the previously modified panoptic_evaluation is in the same directory
# or accessible via the python path.
from .panoptic_evaluation import COCOPanopticEvaluator, pq_compute

logger = logging.getLogger(__name__)


class COCOZSPanopticEvaluator(COCOPanopticEvaluator):

    def _add_seen_unseen_to_pq_res(self, pq_res):    
        # Standard metrics
        seen_pq, seen_sq, seen_rq, seen_n = 0, 0, 0, 0
        seen_tp, seen_fp, seen_fn, seen_gt = 0, 0, 0, 0
        unseen_pq, unseen_sq, unseen_rq, unseen_n = 0, 0, 0, 0
        unseen_tp, unseen_fp, unseen_fn, unseen_gt = 0, 0, 0, 0

        # Class-agnostic
        seen_pqca, seen_sqca, seen_rqca = 0, 0, 0
        seen_tpca, seen_fpca, seen_fnca, seen_gtca = 0, 0, 0, 0
        unseen_pqca, unseen_sqca, unseen_rqca = 0, 0, 0
        unseen_tpca, unseen_fpca, unseen_fnca, unseen_gtca = 0, 0, 0, 0

        # Unique assignment
        seen_pqua, seen_squa, seen_rqua = 0, 0, 0
        seen_tpua, seen_fpua, seen_fnua = 0, 0, 0
        unseen_pqua, unseen_squa, unseen_rqua = 0, 0, 0
        unseen_tpua, unseen_fpua, unseen_fnua = 0, 0, 0

        # Unique assignment, class-agnostic
        seen_pquaca, seen_squaca, seen_rquaca = 0, 0, 0
        seen_tpuaca, seen_fpuaca, seen_fnuaca = 0, 0, 0
        unseen_pquaca, unseen_squaca, unseen_rquaca = 0, 0, 0
        unseen_tpuaca, unseen_fpuaca, unseen_fnuaca = 0, 0, 0

        unseen_dataset_ids = list(self._metadata.unseen_dataset_id_to_contiguous_id.keys())
        for class_id, res in pq_res['per_class'].items():
            if class_id in unseen_dataset_ids:
                unseen_n += 1
                unseen_pq += res['pq']
                unseen_sq += res['sq']
                unseen_rq += res['rq']
                unseen_tp += res['tp']
                unseen_fp += res['fp']
                unseen_fn += res['fn']
                unseen_gt += res['uq'] * res['tp']

                unseen_pqca += res['pqca']
                unseen_sqca += res['sqca']
                unseen_rqca += res['rqca']
                unseen_tpca += res['tpca']
                unseen_fpca += res['fpca']
                unseen_fnca += res['fnca']
                unseen_gtca += res['uqca'] * res['tpca']
                
                unseen_pqua += res['pqua']
                unseen_squa += res['squa']
                unseen_rqua += res['rqua']
                unseen_tpua += res['tpua']
                unseen_fpua += res['fpua']
                unseen_fnua += res['fnua']

                unseen_pquaca += res['pquaca']
                unseen_squaca += res['squaca']
                unseen_rquaca += res['rquaca']
                unseen_tpuaca += res['tpuaca']
                unseen_fpuaca += res['fpuaca']
                unseen_fnuaca += res['fnuaca']
            else:
                seen_n += 1
                seen_pq += res['pq']
                seen_sq += res['sq']
                seen_rq += res['rq']
                seen_tp += res['tp']
                seen_fp += res['fp']
                seen_fn += res['fn']
                seen_gt += res['uq'] * res['tp']

                seen_pqca += res['pqca']
                seen_sqca += res['sqca']
                seen_rqca += res['rqca']
                seen_tpca += res['tpca']
                seen_fpca += res['fpca']
                seen_fnca += res['fnca']
                seen_gtca += res['uqca'] * res['tpca']

                seen_pqua += res['pqua']
                seen_squa += res['squa']
                seen_rqua += res['rqua']
                seen_tpua += res['tpua']
                seen_fpua += res['fpua']
                seen_fnua += res['fnua']

                seen_pquaca += res['pquaca']
                seen_squaca += res['squaca']
                seen_rquaca += res['rquaca']
                seen_tpuaca += res['tpuaca']
                seen_fpuaca += res['fpuaca']
                seen_fnuaca += res['fnuaca']

        pq_res["Seen"] = {
            'pq': (seen_pq / seen_n) if seen_n != 0 else 0, 
            'sq': (seen_sq / seen_n) if seen_n != 0 else 0,
            'rq': (seen_rq / seen_n) if seen_n != 0 else 0, 
            'pr': (seen_tp / (seen_tp + seen_fp)) if (seen_tp + seen_fp) != 0 else 0, 
            're': (seen_tp / (seen_tp + seen_fn)) if (seen_tp + seen_fn) != 0 else 0,
            'uq': (seen_gt / seen_tp) if seen_tp != 0 else 0,

            'pqca': (seen_pqca / seen_n) if seen_n != 0 else 0,
            'sqca': (seen_sqca / seen_n) if seen_n != 0 else 0,
            'rqca': (seen_rqca / seen_n) if seen_n != 0 else 0,
            'prca': (seen_tpca / (seen_tpca + seen_fpca)) if (seen_tpca + seen_fpca) != 0 else 0,
            'reca': (seen_tpca / (seen_tpca + seen_fnca)) if (seen_tpca + seen_fnca) != 0 else 0,
            'uqca': (seen_gtca / seen_tpca) if seen_tpca != 0 else 0,

            'pqua': (seen_pqua / seen_n) if seen_n != 0 else 0,
            'squa': (seen_squa / seen_n) if seen_n != 0 else 0,
            'rqua': (seen_rqua / seen_n) if seen_n != 0 else 0,
            'prua': (seen_tpua / (seen_tpua + seen_fpua)) if (seen_tpua + seen_fpua) != 0 else 0,
            'reua': (seen_tpua / (seen_tpua + seen_fnua)) if (seen_tpua + seen_fnua) != 0 else 0,

            'pquaca': (seen_pquaca / seen_n) if seen_n != 0 else 0,
            'squaca': (seen_squaca / seen_n) if seen_n != 0 else 0,
            'rquaca': (seen_rquaca / seen_n) if seen_n != 0 else 0,
            'pruaca': (seen_tpuaca / (seen_tpuaca + seen_fpuaca)) if (seen_tpuaca + seen_fpuaca) != 0 else 0,
            'reuaca': (seen_tpuaca / (seen_tpuaca + seen_fnuaca)) if (seen_tpuaca + seen_fnuaca) != 0 else 0,
            
            'n': seen_n
        }
        pq_res["Unseen"] = {
            'pq': (unseen_pq / unseen_n) if unseen_n != 0 else 0, 
            'sq': (unseen_sq / unseen_n) if unseen_n != 0 else 0,
            'rq': (unseen_rq / unseen_n) if unseen_n != 0 else 0, 
            'pr': (unseen_tp / (unseen_tp + unseen_fp)) if (unseen_tp + unseen_fp) != 0 else 0, 
            're': (unseen_tp / (unseen_tp + unseen_fn)) if (unseen_tp + unseen_fn) != 0 else 0,
            'uq': (unseen_gt / unseen_tp) if unseen_tp != 0 else 0,

            'pqca': (unseen_pqca / unseen_n) if unseen_n != 0 else 0,
            'sqca': (unseen_sqca / unseen_n) if unseen_n != 0 else 0,
            'rqca': (unseen_rqca / unseen_n) if unseen_n != 0 else 0,
            'prca': (unseen_tpca / (unseen_tpca + unseen_fpca)) if (unseen_tpca + unseen_fpca) != 0 else 0,
            'reca': (unseen_tpca / (unseen_tpca + unseen_fnca)) if (unseen_tpca + unseen_fnca) != 0 else 0,
            'uqca': (unseen_gtca / unseen_tpca) if unseen_tpca != 0 else 0,
            
            'pqua': (unseen_pqua / unseen_n) if unseen_n != 0 else 0,
            'squa': (unseen_squa / unseen_n) if unseen_n != 0 else 0,
            'rqua': (unseen_rqua / unseen_n) if unseen_n != 0 else 0,
            'prua': (unseen_tpua / (unseen_tpua + unseen_fpua)) if (unseen_tpua + unseen_fpua) != 0 else 0,
            'reua': (unseen_tpua / (unseen_tpua + unseen_fnua)) if (unseen_tpua + unseen_fnua) != 0 else 0,

            'pquaca': (unseen_pquaca / unseen_n) if unseen_n != 0 else 0,
            'squaca': (unseen_squaca / unseen_n) if unseen_n != 0 else 0,
            'rquaca': (unseen_rquaca / unseen_n) if unseen_n != 0 else 0,
            'pruaca': (unseen_tpuaca / (unseen_tpuaca + unseen_fpuaca)) if (unseen_tpuaca + unseen_fpuaca) != 0 else 0,
            'reuaca': (unseen_tpuaca / (unseen_tpuaca + unseen_fnuaca)) if (unseen_tpuaca + unseen_fnuaca) != 0 else 0,

            'n': unseen_n
        }

    def pq_compute_to_res_dic(self, pq_res):
        self._add_seen_unseen_to_pq_res(pq_res)
        res = super().pq_compute_to_res_dic(pq_res)

        res["PQ_se"] = 100 * pq_res["Seen"]["pq"]
        res["SQ_se"] = 100 * pq_res["Seen"]["sq"]
        res["RQ_se"] = 100 * pq_res["Seen"]["rq"]
        res["PR_se"] = 100 * pq_res["Seen"]["pr"]
        res["RE_se"] = 100 * pq_res["Seen"]["re"]
        res["UQ_se"] = 100 * pq_res["Seen"]["uq"]
        res["PQca_se"] = 100 * pq_res["Seen"]["pqca"]
        res["SQca_se"] = 100 * pq_res["Seen"]["sqca"]
        res["RQca_se"] = 100 * pq_res["Seen"]["rqca"]
        res["PRca_se"] = 100 * pq_res["Seen"]["prca"]
        res["REca_se"] = 100 * pq_res["Seen"]["reca"]
        res["UQca_se"] = 100 * pq_res["Seen"]["uqca"]
        res["PQua_se"] = 100 * pq_res["Seen"]["pqua"]
        res["SQua_se"] = 100 * pq_res["Seen"]["squa"]
        res["RQua_se"] = 100 * pq_res["Seen"]["rqua"]
        res["PRua_se"] = 100 * pq_res["Seen"]["prua"]
        res["REua_se"] = 100 * pq_res["Seen"]["reua"]
        res["PQuaca_se"] = 100 * pq_res["Seen"]["pquaca"]
        res["SQuaca_se"] = 100 * pq_res["Seen"]["squaca"]
        res["RQuaca_se"] = 100 * pq_res["Seen"]["rquaca"]
        res["PRuaca_se"] = 100 * pq_res["Seen"]["pruaca"]
        res["REuaca_se"] = 100 * pq_res["Seen"]["reuaca"]

        res["PQ_un"] = 100 * pq_res["Unseen"]["pq"]
        res["SQ_un"] = 100 * pq_res["Unseen"]["sq"]
        res["RQ_un"] = 100 * pq_res["Unseen"]["rq"]
        res["PR_un"] = 100 * pq_res["Unseen"]["pr"]
        res["RE_un"] = 100 * pq_res["Unseen"]["re"]
        res["UQ_un"] = 100 * pq_res["Unseen"]["uq"]
        res["PQca_un"] = 100 * pq_res["Unseen"]["pqca"]
        res["SQca_un"] = 100 * pq_res["Unseen"]["sqca"]
        res["RQca_un"] = 100 * pq_res["Unseen"]["rqca"]
        res["PRca_un"] = 100 * pq_res["Unseen"]["prca"]
        res["REca_un"] = 100 * pq_res["Unseen"]["reca"]
        res["UQca_un"] = 100 * pq_res["Unseen"]["uqca"]
        res["PQua_un"] = 100 * pq_res["Unseen"]["pqua"]
        res["SQua_un"] = 100 * pq_res["Unseen"]["squa"]
        res["RQua_un"] = 100 * pq_res["Unseen"]["rqua"]
        res["PRua_un"] = 100 * pq_res["Unseen"]["prua"]
        res["REua_un"] = 100 * pq_res["Unseen"]["reua"]
        res["PQuaca_un"] = 100 * pq_res["Unseen"]["pquaca"]
        res["SQuaca_un"] = 100 * pq_res["Unseen"]["squaca"]
        res["RQuaca_un"] = 100 * pq_res["Unseen"]["rquaca"]
        res["PRuaca_un"] = 100 * pq_res["Unseen"]["pruaca"]
        res["REuaca_un"] = 100 * pq_res["Unseen"]["reuaca"]

        return res

    def _print_panoptic_results(self, pq_res):        
        categories = ["All", "Things", "Stuff", "Seen", "Unseen"]

        headers = ["", "PQ", "SQ", "RQ", "PR", "RE", "UQ", "#categories"]
        data = []
        for name in categories:
            row = [name] + [
                pq_res[name][k] * 100 
                for k in ["pq", "sq", "rq", "pr", "re", "uq"]
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
                for k in ["pqca", "sqca", "rqca", "prca", "reca", "uqca"]
            ] + [pq_res[name]["n"]]
            data.append(row)
        table = tabulate(
            data, headers=headers, tablefmt="pipe", floatfmt=".3f", stralign="center", numalign="center"
        )
        logger.info("Panoptic Evaluation Results (Class-Agnostic):\n" + table)

        headers = ["", "PQ", "SQ", "RQ", "PR", "RE", "#categories"]
        data = []
        for name in categories:
            row = [name] + [
                pq_res[name][k] * 100 for k in ["pqua", "squa", "rqua", "prua", "reua"]
            ] + [pq_res[name]["n"]]
            data.append(row)
        table = tabulate(
            data, headers=headers, tablefmt="pipe", floatfmt=".3f", stralign="center", numalign="center"
        )
        logger.info("Panoptic Evaluation Results (Unique Assignment):\n" + table)
        
        data = []
        for name in categories:
            row = [name] + [
                pq_res[name][k] * 100 for k in ["pquaca", "squaca", "rquaca", "pruaca", "reuaca"]
            ] + [pq_res[name]["n"]]
            data.append(row)
        table = tabulate(
            data, headers=headers, tablefmt="pipe", floatfmt=".3f", stralign="center", numalign="center"
        )
        logger.info("Panoptic Evaluation Results (Unique Assignment, Class-Agnostic):\n" + table)


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

    categories = ["All", "Things", "Stuff", "Seen", "Unseen"]
    headers = ["", "PQ", "SQ", "RQ", "PR", "RE", "UQ", "#categories"]
    data = []
    for name in categories:
        row = [name] + [
            pq_res[name][k] * 100 
            for k in ["pq", "sq", "rq", "pr", "re", "uq"]
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
            for k in ["pqca", "sqca", "rqca", "prca", "reca", "uqca"]
        ] + [pq_res[name]["n"]]
        data.append(row)
    table = tabulate(
        data, headers=headers, tablefmt="pipe", floatfmt=".3f", stralign="center", numalign="center"
    )
    logger.info("Panoptic Evaluation Results (Class-Agnostic):\n" + table)
    
    headers = ["", "PQ", "SQ", "RQ", "PR", "RE", "#categories"]
    data = []
    for name in categories:
        row = [name] + [
            pq_res[name][k] * 100 for k in ["pqua", "squa", "rqua", "prua", "reua"]
        ] + [pq_res[name]["n"]]
        data.append(row)
    table = tabulate(
        data, headers=headers, tablefmt="pipe", floatfmt=".3f", stralign="center", numalign="center"
    )
    logger.info("Panoptic Evaluation Results (Unique Assignment):\n" + table)
    
    data = []
    for name in categories:
        row = [name] + [
            pq_res[name][k] * 100 for k in ["pquaca", "squaca", "rquaca", "pruaca", "reuaca"]
        ] + [pq_res[name]["n"]]
        data.append(row)
    table = tabulate(
        data, headers=headers, tablefmt="pipe", floatfmt=".3f", stralign="center", numalign="center"
    )
    logger.info("Panoptic Evaluation Results (Unique Assignment, Class-Agnostic):\n" + table)