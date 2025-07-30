import contextlib
import io
import logging
from tabulate import tabulate

from .panoptic_evaluation import COCOPanopticEvaluator, pq_compute

logger = logging.getLogger(__name__)


class COCOZSPanopticEvaluator(COCOPanopticEvaluator):

    def _add_seen_unseen_to_pq_res(self, pq_res):    
        seen_pq, seen_sq, seen_rq, seen_n = 0, 0, 0, 0
        seen_pqca, seen_sqca, seen_rqca = 0, 0, 0

        seen_tp, seen_fp, seen_fn = 0, 0, 0
        seen_tpca, seen_fpca, seen_fnca = 0, 0, 0

        unseen_pq, unseen_sq, unseen_rq, unseen_n = 0, 0, 0, 0
        unseen_pqca, unseen_sqca, unseen_rqca = 0, 0, 0

        unseen_tp, unseen_fp, unseen_fn = 0, 0, 0
        unseen_tpca, unseen_fpca, unseen_fnca = 0, 0, 0

        unseen_dataset_ids = list(self._metadata.unseen_dataset_id_to_contiguous_id.keys())
        for class_id,res in pq_res['per_class'].items():
            if class_id in unseen_dataset_ids:
                unseen_pq += res['pq']
                unseen_sq += res['sq']
                unseen_rq += res['rq']
                unseen_n += 1

                unseen_pqca += res['pqca']
                unseen_sqca += res['sqca']
                unseen_rqca += res['rqca']

                unseen_tp += res['tp']
                unseen_fp += res['fp']
                unseen_fn += res['fn']

                unseen_tpca += res['tpca']
                unseen_fpca += res['fpca']
                unseen_fnca += res['fnca']
            else:
                seen_pq += res['pq']
                seen_sq += res['sq']
                seen_rq += res['rq']
                seen_n += 1

                seen_pqca += res['pqca']
                seen_sqca += res['sqca']
                seen_rqca += res['rqca']

                seen_tp += res['tp']
                seen_fp += res['fp']
                seen_fn += res['fn']

                seen_tpca += res['tpca']
                seen_fpca += res['fpca']
                seen_fnca += res['fnca']

        seen_pred_sum = seen_tp + seen_fp
        seen_actual_sum = seen_tp + seen_fn

        unseen_pred_sum = unseen_tp + unseen_fp
        unseen_actual_sum = unseen_tp + unseen_fn

        pq_res["Seen"] = {
            'pq': (seen_pq/seen_n) if seen_n != 0 else 0, 
            'sq': (seen_sq/seen_n) if seen_n != 0 else 0,
            'rq': (seen_rq/seen_n) if seen_n != 0 else 0, 
            'pqca': (seen_pqca/seen_n) if seen_n != 0 else 0,
            'sqca': (seen_sqca/seen_n) if seen_n != 0 else 0,
            'rqca': (seen_rqca/seen_n) if seen_n != 0 else 0,
            'pr': (seen_tp / (seen_tp + seen_fp)) if (seen_tp + seen_fp) != 0 else 0, 
            're': (seen_tp / (seen_tp + seen_fn)) if (seen_tp + seen_fn) != 0 else 0,
            'prca': (seen_tpca / (seen_tpca + seen_fpca)) if (seen_tpca + seen_fpca) != 0 else 0,
            'reca': (seen_tpca / (seen_tpca + seen_fnca)) if (seen_tpca + seen_fnca) != 0 else 0,
            'n': seen_n
        }
        pq_res["Unseen"] = {
            'pq': (unseen_pq/unseen_n) if unseen_n != 0 else 0, 
            'sq': (unseen_sq/unseen_n) if unseen_n != 0 else 0,
            'rq': (unseen_rq/unseen_n) if unseen_n != 0 else 0, 
            'pqca': (unseen_pqca/unseen_n) if unseen_n != 0 else 0,
            'sqca': (unseen_sqca/unseen_n) if unseen_n != 0 else 0,
            'rqca': (unseen_rqca/unseen_n) if unseen_n != 0 else 0,
            'pr': (unseen_tp / (unseen_tp + unseen_fp)) if (unseen_tp + unseen_fp) != 0 else 0, 
            're': (unseen_tp / (unseen_tp + unseen_fn)) if (unseen_tp + unseen_fn) != 0 else 0,
            'prca': (unseen_tpca / (unseen_tpca + unseen_fpca)) if (unseen_tpca + unseen_fpca) != 0 else 0,
            'reca': (unseen_tpca / (unseen_tpca + unseen_fnca)) if (unseen_tpca + unseen_fnca) != 0 else 0,
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
        res["PQca_se"] = 100 * pq_res["Seen"]["pqca"]
        res["SQca_se"] = 100 * pq_res["Seen"]["sqca"]
        res["RQca_se"] = 100 * pq_res["Seen"]["rqca"]
        res["PRca_se"] = 100 * pq_res["Seen"]["prca"]
        res["REca_se"] = 100 * pq_res["Seen"]["reca"]

        res["PQ_un"] = 100 * pq_res["Unseen"]["pq"]
        res["SQ_un"] = 100 * pq_res["Unseen"]["sq"]
        res["RQ_un"] = 100 * pq_res["Unseen"]["rq"]
        res["PR_un"] = 100 * pq_res["Unseen"]["pr"]
        res["RE_un"] = 100 * pq_res["Unseen"]["re"]
        res["PQca_un"] = 100 * pq_res["Unseen"]["pqca"]
        res["SQca_un"] = 100 * pq_res["Unseen"]["sqca"]
        res["RQca_un"] = 100 * pq_res["Unseen"]["rqca"]
        res["PRca_un"] = 100 * pq_res["Unseen"]["prca"]
        res["REca_un"] = 100 * pq_res["Unseen"]["reca"]

        return res

    def _print_panoptic_results(self, pq_res):
        headers = ["", "PQ", "SQ", "RQ", "PR", "RE", "#categories"]
        data = []
        for name in ["All", "Things", "Stuff", "Seen", "Unseen"]:
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
        for name in ["All", "Things", "Stuff", "Seen", "Unseen"]:
            row = [name] + [
                pq_res[name][k] * 100 
                for k in ["pqca", "sqca", "rqca", "prca", "reca"]
            ] + [pq_res[name]["n"]]
            data.append(row)
        table = tabulate(
            data, headers=headers, tablefmt="pipe", floatfmt=".3f", stralign="center", numalign="center"
        )
        logger.info("Panoptic Evaluation Results (Class-Agnostic):\n" + table)
        logger.info(pq_res)


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

        headers = ["", "PQ", "SQ", "RQ", "PR", "RE", "#categories"]
        data = []
        for name in ["All", "Things", "Stuff", "Seen", "Unseen"]:
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
        for name in ["All", "Things", "Stuff", "Seen", "Unseen"]:
            row = [name] + [
                pq_res[name][k] * 100 
                for k in ["pqca", "sqca", "rqca", "prca", "reca"]
            ] + [pq_res[name]["n"]]
            data.append(row)
        table = tabulate(
            data, headers=headers, tablefmt="pipe", floatfmt=".3f", stralign="center", numalign="center"
        )
        logger.info("Panoptic Evaluation Results (Class-Agnostic):\n" + table)
        logger.info(pq_res)