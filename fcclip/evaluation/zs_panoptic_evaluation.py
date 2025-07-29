import contextlib
import io
import logging
from tabulate import tabulate

from .panoptic_evaluation import COCOPanopticEvaluator, pq_compute

logger = logging.getLogger(__name__)


class COCOZSPanopticEvaluator(COCOPanopticEvaluator):

    def _add_seen_unseen_to_pq_res(self, pq_res):    
        seen_pq, seen_sq, seen_sqca, seen_rq, seen_n = 0, 0, 0, 0, 0
        seen_tp, seen_fp, seen_fn, seen_tpca = 0, 0, 0, 0
        unseen_pq, unseen_sq, unseen_sqca, unseen_rq, unseen_n = 0, 0, 0, 0, 0
        unseen_tp, unseen_fp, unseen_fn, unseen_tpca = 0, 0, 0, 0

        unseen_dataset_ids = list(self._metadata.unseen_dataset_id_to_contiguous_id.keys())
        for class_id,res in pq_res['per_class'].items():
            if class_id in unseen_dataset_ids:
                unseen_pq += res['pq']
                unseen_sq += res['sq']
                unseen_sqca += res['sqca']
                unseen_rq += res['rq']
                unseen_n += 1

                unseen_tp += res['tp']
                unseen_fp += res['fp']
                unseen_fn += res['fn']
                unseen_tpca += res['tpca']
            else:
                seen_pq += res['pq']
                seen_sq += res['sq']
                seen_sqca += res['sqca']
                seen_rq += res['rq']
                seen_n += 1

                seen_tp += res['tp']
                seen_fp += res['fp']
                seen_fn += res['fn']
                seen_tpca += res['tpca']
        
        seen_pred_sum = seen_tp + seen_fp
        seen_actual_sum = seen_tp + seen_fn

        unseen_pred_sum = unseen_tp + unseen_fp
        unseen_actual_sum = unseen_tp + unseen_fn

        pq_res["Seen"] = {
            'pq': (seen_pq/seen_n) if seen_n != 0 else 0, 
            'sq': (seen_sq/seen_n) if seen_n != 0 else 0,
            'sqca': (seen_sqca/seen_n) if seen_n != 0 else 0, 
            'rq': (seen_rq/seen_n) if seen_n != 0 else 0, 
            'pr': (seen_tp / seen_pred_sum) if seen_pred_sum != 0 else 0, 
            'prca': (seen_tpca / seen_pred_sum) if seen_pred_sum != 0 else 0,
            're': (seen_tp / seen_actual_sum) if seen_actual_sum != 0 else 0,
            'reca': (seen_tpca / seen_actual_sum) if seen_actual_sum != 0 else 0,
            'n': seen_n
        }
        pq_res["Unseen"] = {
            'pq': (unseen_pq/unseen_n) if unseen_n != 0 else 0, 
            'sq': (unseen_sq/unseen_n) if unseen_n != 0 else 0, 
            'sqca': (unseen_sqca/unseen_n) if unseen_n != 0 else 0, 
            'rq': (unseen_rq/unseen_n) if unseen_n != 0 else 0, 
            'pr': (unseen_tp / unseen_pred_sum) if unseen_pred_sum != 0 else 0, 
            'prca': (unseen_tpca / unseen_pred_sum) if unseen_pred_sum != 0 else 0,
            're': (unseen_tp / unseen_actual_sum) if unseen_actual_sum != 0 else 0,
            'reca': (unseen_tpca / unseen_actual_sum) if unseen_actual_sum != 0 else 0,
            'n': unseen_n
        }

    def pq_compute_to_res_dic(self, pq_res):
        self._add_seen_unseen_to_pq_res(pq_res)
        res = super().pq_compute_to_res_dic(pq_res)

        res["PQ_se"] = 100 * pq_res["Seen"]["pq"]
        res["SQ_se"] = 100 * pq_res["Seen"]["sq"]
        res["SQca_se"] = 100 * pq_res["Seen"]["sqca"]
        res["RQ_se"] = 100 * pq_res["Seen"]["rq"]
        res["PR_se"] = 100 * pq_res["Seen"]["pr"]
        res["PRca_se"] = 100 * pq_res["Seen"]["prca"]
        res["RE_se"] = 100 * pq_res["Seen"]["re"]
        res["REca_se"] = 100 * pq_res["Seen"]["reca"]

        res["PQ_un"] = 100 * pq_res["Unseen"]["pq"]
        res["SQ_un"] = 100 * pq_res["Unseen"]["sq"]
        res["SQca_un"] = 100 * pq_res["Unseen"]["sqca"]
        res["RQ_un"] = 100 * pq_res["Unseen"]["rq"]
        res["PR_un"] = 100 * pq_res["Unseen"]["pr"]
        res["PRca_un"] = 100 * pq_res["Unseen"]["prca"]
        res["RE_un"] = 100 * pq_res["Unseen"]["re"]
        res["REca_un"] = 100 * pq_res["Unseen"]["reca"]

        return res

    def _print_panoptic_results(self, pq_res):
        headers = ["", "PQ", "SQ", "SQca", "RQ", "PR", "PRca", "RE", "REca", "#categories"]
        data = []
        for name in ["All", "Things", "Stuff", "Seen", "Unseen"]:
            row = [name] + [
                pq_res[name][k] * 100 
                for k in ["pq", "sq", "sqca", "rq", "pr", "prca", "re", "reca"]
            ] + [pq_res[name]["n"]]
            data.append(row)
        table = tabulate(
            data, headers=headers, tablefmt="pipe", floatfmt=".3f", stralign="center", numalign="center"
        )
        logger.info(pq_res)
        logger.info("Panoptic Evaluation Results:\n" + table)


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
        headers = ["", "PQ", "SQ", "SQca", "RQ", "PR", "PRca", "RE", "REca", "#categories"]
        data = []
        for name in ["All", "Things", "Stuff", "Seen", "Unseen"]:
            row = [name] + [
                pq_res[name][k] * 100 
                for k in ["pq", "sq", "sqca", "rq", "pr", "prca", "re", "reca"]
            ] + [pq_res[name]["n"]]
            data.append(row)
        table = tabulate(
            data, headers=headers, tablefmt="pipe", floatfmt=".3f", stralign="center", numalign="center"
        )
        logger.info(pq_res)
        logger.info("Panoptic Evaluation Results:\n" + table)