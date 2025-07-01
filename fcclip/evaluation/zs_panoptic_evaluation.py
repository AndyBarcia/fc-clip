import contextlib
import io
import os
import json
import logging
import itertools
from collections import OrderedDict
from tabulate import tabulate
import tempfile

from detectron2.evaluation.panoptic_evaluation import COCOPanopticEvaluator

from detectron2.utils import comm
from detectron2.utils.file_io import PathManager

logger = logging.getLogger(__name__)


class COCOZSPanopticEvaluator(COCOPanopticEvaluator):
    
    def evaluate(self):
        comm.synchronize()

        self._predictions = comm.gather(self._predictions)
        self._predictions = list(itertools.chain(*self._predictions))
        if not comm.is_main_process():
            return

        # PanopticApi requires local files
        gt_json = PathManager.get_local_path(self._metadata.panoptic_json)
        gt_folder = PathManager.get_local_path(self._metadata.panoptic_root)

        with tempfile.TemporaryDirectory(prefix="panoptic_eval") as pred_dir:
            logger.info("Writing all panoptic predictions to {} ...".format(pred_dir))
            for p in self._predictions:
                with open(os.path.join(pred_dir, p["file_name"]), "wb") as f:
                    f.write(p.pop("png_string"))

            with open(gt_json, "r") as f:
                json_data = json.load(f)
            json_data["annotations"] = self._predictions

            output_dir = self._output_dir or pred_dir
            predictions_json = os.path.join(output_dir, "predictions.json")
            with PathManager.open(predictions_json, "w") as f:
                f.write(json.dumps(json_data))

            from panopticapi.evaluation import pq_compute

            with contextlib.redirect_stdout(io.StringIO()):
                pq_res = pq_compute(
                    gt_json,
                    PathManager.get_local_path(predictions_json),
                    gt_folder=gt_folder,
                    pred_folder=pred_dir,
                )
                self._add_seen_unseen_to_pq_res(pq_res)

        res = {}
        res["PQ"] = 100 * pq_res["All"]["pq"]
        res["SQ"] = 100 * pq_res["All"]["sq"]
        res["RQ"] = 100 * pq_res["All"]["rq"]
        res["PQ_th"] = 100 * pq_res["Things"]["pq"]
        res["SQ_th"] = 100 * pq_res["Things"]["sq"]
        res["RQ_th"] = 100 * pq_res["Things"]["rq"]
        res["PQ_st"] = 100 * pq_res["Stuff"]["pq"]
        res["SQ_st"] = 100 * pq_res["Stuff"]["sq"]
        res["RQ_st"] = 100 * pq_res["Stuff"]["rq"]
        res["PQ_se"] = 100 * pq_res["Seen"]["pq"]
        res["SQ_se"] = 100 * pq_res["Seen"]["sq"]
        res["RQ_se"] = 100 * pq_res["Seen"]["rq"]
        res["PQ_un"] = 100 * pq_res["Unseen"]["pq"]
        res["SQ_un"] = 100 * pq_res["Unseen"]["sq"]
        res["RQ_un"] = 100 * pq_res["Unseen"]["rq"]

        results = OrderedDict({"panoptic_seg": res})
        _print_panoptic_results(pq_res)

        return results

    def _add_seen_unseen_to_pq_res(self, pq_res):    
        seen_pq, seen_sq, seen_rq, seen_n = 0, 0, 0, 0
        unseen_pq, unseen_sq, unseen_rq, unseen_n = 0, 0, 0, 0

        unseen_dataset_ids = list(self._metadata.unseen_dataset_id_to_contiguous_id.keys())
        for class_id,res in pq_res['per_class'].items():
            if class_id in unseen_dataset_ids:
                unseen_pq += res['pq']
                unseen_sq += res['sq']
                unseen_rq += res['rq']
                unseen_n += 1
            else:
                seen_pq += res['pq']
                seen_sq += res['sq']
                seen_rq += res['rq']
                seen_n += 1
        
        pq_res["Seen"] = {
            'pq': (seen_pq/seen_n) if seen_n != 0 else 0, 
            'sq': (seen_sq/seen_n) if seen_n != 0 else 0, 
            'rq': (seen_rq/seen_n) if seen_n != 0 else 0, 
            'n': seen_n
        }
        pq_res["Unseen"] = {
            'pq': (unseen_pq/unseen_n) if unseen_n != 0 else 0, 
            'sq': (unseen_sq/unseen_n) if unseen_n != 0 else 0, 
            'rq': (unseen_rq/unseen_n) if unseen_n != 0 else 0, 
            'n': unseen_n
        }

def _print_panoptic_results(pq_res):
    headers = ["", "PQ", "SQ", "RQ", "#categories"]
    data = []
    for name in ["All", "Things", "Stuff", "Seen", "Unseen"]:
        row = [name] + [pq_res[name][k] * 100 for k in ["pq", "sq", "rq"]] + [pq_res[name]["n"]]
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

    from panopticapi.evaluation import pq_compute

    with contextlib.redirect_stdout(io.StringIO()):
        pq_res = pq_compute(
            args.gt_json, args.pred_json, gt_folder=args.gt_dir, pred_folder=args.pred_dir
        )
        _print_panoptic_results(pq_res)
