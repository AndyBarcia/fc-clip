#!/usr/bin/env python
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import numpy as np
import json
import time
from collections import defaultdict, OrderedDict
import argparse
import multiprocessing
import contextlib
import io
import itertools
import logging
import tempfile
from typing import Optional
from tabulate import tabulate

import PIL.Image as Image
from panopticapi.utils import get_traceback, rgb2id

from detectron2.data import MetadataCatalog
from detectron2.utils import comm
from detectron2.utils.file_io import PathManager
from detectron2.evaluation.evaluator import DatasetEvaluator

logger = logging.getLogger(__name__)


OFFSET = 256 * 256 * 256
VOID = 0


class PQStatCat:
    """
    Per-category statistics including standard and class-agnostic metrics.
    """
    def __init__(self):
        self.iou = 0.0      # Sum of IoUs for true positives (same class)
        self.tp = 0         # True positives (same class)
        self.fp = 0         # False positives (same class)
        self.fn = 0         # False negatives (same class)
        self.iou_ca = 0.0   # Sum of IoUs for all IoU > 0.5 matches (class-agnostic)
        self.tp_ca = 0      # Total count of IoU > 0.5 matches (class-agnostic)

    def __iadd__(self, other):
        for attr in ['iou', 'tp', 'fp', 'fn', 'iou_ca', 'tp_ca']:
            setattr(self, attr, getattr(self, attr) + getattr(other, attr))
        return self


class PQStat:
    def __init__(self):
        self.pq_per_cat = defaultdict(PQStatCat)

    def __getitem__(self, idx):
        return self.pq_per_cat[idx]

    def __iadd__(self, other):
        for label, cat_stat in other.pq_per_cat.items():
            self.pq_per_cat[label] += cat_stat
        return self

    def pq_average(self, categories, isthing=None):
        pq_sum = sq_sum = rq_sum = sqca_sum = 0.0
        tp_sum = fp_sum = fn_sum = tpca_sum = 0
        n = 0
        per_class = {}

        for label, info in categories.items():
            cat_isthing = (info.get('isthing', 0) == 1)
            if isthing is not None and cat_isthing != isthing:
                continue

            stat = self.pq_per_cat[label]

            tp_sum += stat.tp
            fp_sum += stat.fp
            fn_sum += stat.fn
            tpca_sum += stat.tp_ca

            denom = stat.tp + 0.5 * stat.fp + 0.5 * stat.fn
            if denom == 0:
                per_class[label] = {
                    'pq': 0.0, 'sq': 0.0, 'rq': 0.0, 'sqca': 0.0,
                    'tp': stat.tp, 'fp': stat.fp, 'fn': stat.fn, 'tpca': stat.tp_ca
                }
                continue

            n += 1
            pq_c = stat.iou / denom
            sq_c = stat.iou / stat.tp if stat.tp > 0 else 0.0
            rq_c = stat.tp / denom
            sqca_c = stat.iou_ca / stat.tp_ca if stat.tp_ca > 0 else 0.0

            per_class[label] = {
                'pq': pq_c, 'sq': sq_c, 'rq': rq_c, 'sqca': sqca_c,
                'tp': stat.tp, 'fp': stat.fp, 'fn': stat.fn, 'tpca': stat.tp_ca
            }

            pq_sum += pq_c
            sq_sum += sq_c
            rq_sum += rq_c
            sqca_sum += sqca_c

        if n > 0:
            pred_sum = tp_sum + fp_sum
            actual_sum = tp_sum + fn_sum
            overall = {
                'pq': pq_sum / n,
                'sq': sq_sum / n,
                'rq': rq_sum / n,
                'sqca': sqca_sum / n,
                'pr': (tp_sum / pred_sum) if pred_sum != 0 else 0, 
                'prca': (tpca_sum / pred_sum) if pred_sum != 0 else 0,
                're': (tp_sum / actual_sum) if actual_sum != 0 else 0,
                'reca': (tpca_sum / actual_sum) if actual_sum != 0 else 0,
                'n': n
            }
        else:
            overall = {
                'pq': 0.0, 'sq': 0.0, 'rq': 0.0, 'sqca': 0.0,
                'pr': 0.0, 'prca': 0.0, 're': 0.0, 'reca': 0.0,
                'n': 0
            }

        return overall, per_class


@get_traceback
def pq_compute_single_core(proc_id, annotation_set, gt_folder, pred_folder, categories):
    pq_stat = PQStat()

    for idx, (gt_ann, pred_ann) in enumerate(annotation_set):
        if idx % 100 == 0:
            print(f"Core: {proc_id}, {idx} of {len(annotation_set)} processed")

        pan_gt = rgb2id(np.array(Image.open(os.path.join(gt_folder, gt_ann['file_name'])), dtype=np.uint32))
        pan_pred = rgb2id(np.array(Image.open(os.path.join(pred_folder, pred_ann['file_name'])), dtype=np.uint32))

        gt_segms = {s['id']: s for s in gt_ann['segments_info']}
        pred_segms = {s['id']: s for s in pred_ann['segments_info']}

        labels, counts = np.unique(pan_pred, return_counts=True)
        for lbl, cnt in zip(labels, counts):
            if lbl == VOID:
                continue
            if lbl not in pred_segms:
                raise KeyError(f"Segment {lbl} in PNG not in JSON for image {gt_ann['image_id']}")
            pred_segms[lbl]['area'] = cnt

        combined = pan_gt.astype(np.uint64) * OFFSET + pan_pred.astype(np.uint64)
        pairs, intsct = np.unique(combined, return_counts=True)
        gt_pred_map = {(p // OFFSET, p % OFFSET): c for p, c in zip(pairs, intsct)}

        matched_gt, matched_pred = set(), set()

        # Match same-category segments
        for (gt_id, pred_id), inter in gt_pred_map.items():
            if gt_id not in gt_segms or pred_id not in pred_segms:
                continue

            g, p = gt_segms[gt_id], pred_segms[pred_id]
            if g['iscrowd'] == 1 or g['category_id'] != p['category_id']:
                continue

            union = p['area'] + g['area'] - inter - gt_pred_map.get((VOID, pred_id), 0)
            iou = inter / union if union > 0 else 0.0

            if iou > 0.5:
                cat = g['category_id']
                stat = pq_stat[cat]
                stat.tp += 1
                stat.iou += iou
                stat.tp_ca += 1
                stat.iou_ca += iou
                matched_gt.add(gt_id)
                matched_pred.add(pred_id)

        # Class-agnostic matching for remaining unmatched pairs
        for (gt_id, pred_id), inter in gt_pred_map.items():
            if gt_id in matched_gt or pred_id in matched_pred:
                continue
            if gt_id not in gt_segms or pred_id not in pred_segms:
                continue

            g, p = gt_segms[gt_id], pred_segms[pred_id]
            if g['iscrowd'] == 1:
                continue

            union = p['area'] + g['area'] - inter - gt_pred_map.get((VOID, pred_id), 0)
            iou = inter / union if union > 0 else 0.0

            if iou > 0.5:
                cat = g['category_id']
                stat = pq_stat[cat]
                stat.tp_ca += 1
                stat.iou_ca += iou
                matched_gt.add(gt_id)
                matched_pred.add(pred_id)

        # Count false negatives
        crowd_by_cat = {}
        for gt_id, g in gt_segms.items():
            if gt_id in matched_gt:
                continue
            if g['iscrowd'] == 1:
                crowd_by_cat[g['category_id']] = gt_id
                continue
            pq_stat[g['category_id']].fn += 1

        # Count false positives
        for pred_id, p in pred_segms.items():
            if pred_id in matched_pred:
                continue
            inter = gt_pred_map.get((VOID, pred_id), 0)
            if p['category_id'] in crowd_by_cat:
                inter += gt_pred_map.get((crowd_by_cat[p['category_id']], pred_id), 0)
            if inter / p['area'] > 0.5:
                continue
            pq_stat[p['category_id']].fp += 1

    print(f"Core: {proc_id}, all {len(annotation_set)} processed")
    return pq_stat


def pq_compute_multi_core(matched_list, gt_folder, pred_folder, categories):
    cpu = multiprocessing.cpu_count()
    splits = np.array_split(matched_list, cpu)
    pool = multiprocessing.Pool(cpu)

    procs = [
        pool.apply_async(pq_compute_single_core, (i, split, gt_folder, pred_folder, categories))
        for i, split in enumerate(splits)
    ]

    pool.close()
    pool.join()

    total = PQStat()
    for p in procs:
        total += p.get()

    return total


def pq_compute(gt_json, pred_json, gt_folder=None, pred_folder=None):
    start = time.time()

    with open(gt_json) as f:
        gt = json.load(f)

    with open(pred_json) as f:
        pred = json.load(f)

    gt_folder = gt_folder or gt_json.replace('.json', '')
    pred_folder = pred_folder or pred_json.replace('.json', '')

    categories = {c['id']: c for c in gt['categories']}
    pred_map = {a['image_id']: a for a in pred['annotations']}

    matched = []
    for a in gt['annotations']:
        if a['image_id'] not in pred_map:
            raise Exception(f"No prediction for image {a['image_id']}")
        matched.append((a, pred_map[a['image_id']]))

    stats = pq_compute_multi_core(matched, gt_folder, pred_folder, categories)

    metrics = [('All', None), ('Things', True), ('Stuff', False)]
    results = {}
    for name, isthing in metrics:
        stats_res, per_cls = stats.pq_average(categories, isthing)
        results[name] = stats_res
        if name == 'All':
            results['per_class'] = per_cls

    return results


class COCOPanopticEvaluator(DatasetEvaluator):
    """
    Evaluate Panoptic Quality metrics on COCO using PanopticAPI.
    It saves panoptic segmentation prediction in `output_dir`

    It contains a synchronize call and has to be called from all workers.
    """

    def __init__(self, dataset_name: str, output_dir: Optional[str] = None):
        """
        Args:
            dataset_name: name of the dataset
            output_dir: output directory to save results for evaluation.
        """
        self._metadata = MetadataCatalog.get(dataset_name)
        self._thing_contiguous_id_to_dataset_id = {
            v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
        }
        self._stuff_contiguous_id_to_dataset_id = {
            v: k for k, v in self._metadata.stuff_dataset_id_to_contiguous_id.items()
        }

        self._output_dir = output_dir
        if self._output_dir is not None:
            PathManager.mkdirs(self._output_dir)

    def reset(self):
        self._predictions = []

    def _convert_category_id(self, segment_info):
        isthing = segment_info.pop("isthing", None)
        if isthing is None:
            # the model produces panoptic category id directly. No more conversion needed
            return segment_info
        if isthing is True:
            segment_info["category_id"] = self._thing_contiguous_id_to_dataset_id[
                segment_info["category_id"]
            ]
        else:
            segment_info["category_id"] = self._stuff_contiguous_id_to_dataset_id[
                segment_info["category_id"]
            ]
        return segment_info

    def process(self, inputs, outputs):
        from panopticapi.utils import id2rgb

        for input, output in zip(inputs, outputs):
            panoptic_img, segments_info = output["panoptic_seg"]
            panoptic_img = panoptic_img.cpu().numpy()
            if segments_info is None:
                # If "segments_info" is None, we assume "panoptic_img" is a
                # H*W int32 image storing the panoptic_id in the format of
                # category_id * label_divisor + instance_id. We reserve -1 for
                # VOID label, and add 1 to panoptic_img since the official
                # evaluation script uses 0 for VOID label.
                label_divisor = self._metadata.label_divisor
                segments_info = []
                for panoptic_label in np.unique(panoptic_img):
                    if panoptic_label == -1:
                        # VOID region.
                        continue
                    pred_class = panoptic_label // label_divisor
                    isthing = (
                        pred_class in self._metadata.thing_dataset_id_to_contiguous_id.values()
                    )
                    segments_info.append(
                        {
                            "id": int(panoptic_label) + 1,
                            "category_id": int(pred_class),
                            "isthing": bool(isthing),
                        }
                    )
                # Official evaluation script uses 0 for VOID label.
                panoptic_img += 1

            file_name = os.path.basename(input["file_name"])
            file_name_png = os.path.splitext(file_name)[0] + ".png"
            with io.BytesIO() as out:
                Image.fromarray(id2rgb(panoptic_img)).save(out, format="PNG")
                segments_info = [self._convert_category_id(x) for x in segments_info]
                self._predictions.append(
                    {
                        "image_id": input["image_id"],
                        "file_name": file_name_png,
                        "png_string": out.getvalue(),
                        "segments_info": segments_info,
                    }
                )

    def pq_compute_to_res_dic(self, pq_res):
        # 'pr': 0.0, 'prca': 0.0, 're': 0.0, 'reca': 0.0,
        res = {}
        res["PQ"] = 100 * pq_res["All"]["pq"]
        res["SQ"] = 100 * pq_res["All"]["sq"]
        res["SQca"] = 100 * pq_res["All"]["sqca"]
        res["RQ"] = 100 * pq_res["All"]["rq"]
        res["PR"] = 100 * pq_res["All"]["pr"]
        res["PRca"] = 100 * pq_res["All"]["prca"]
        res["RE"] = 100 * pq_res["All"]["re"]
        res["REca"] = 100 * pq_res["All"]["reca"]

        res["PQ_th"] = 100 * pq_res["Things"]["pq"]
        res["SQ_th"] = 100 * pq_res["Things"]["sq"]
        res["SQca_th"] = 100 * pq_res["Things"]["sqca"]
        res["RQ_th"] = 100 * pq_res["Things"]["rq"]
        res["PR_th"] = 100 * pq_res["Things"]["pr"]
        res["PRca_th"] = 100 * pq_res["Things"]["prca"]
        res["RE_th"] = 100 * pq_res["Things"]["re"]
        res["REca_th"] = 100 * pq_res["Things"]["reca"]

        res["PQ_st"] = 100 * pq_res["Stuff"]["pq"]
        res["SQ_st"] = 100 * pq_res["Stuff"]["sq"]
        res["SQca_st"] = 100 * pq_res["Stuff"]["sqca"]
        res["RQ_st"] = 100 * pq_res["Stuff"]["rq"]
        res["PR_st"] = 100 * pq_res["Stuff"]["pr"]
        res["PRca_st"] = 100 * pq_res["Stuff"]["prca"]
        res["RE_st"] = 100 * pq_res["Stuff"]["re"]
        res["REca_st"] = 100 * pq_res["Stuff"]["reca"]
        return res

    def _print_panoptic_results(self, pq_res):
        headers = ["", "PQ", "SQ", "SQca", "RQ", "PR", "PRca", "RE", "REca", "#categories"]
        data = []
        for name in ["All", "Things", "Stuff"]:
            row = [name] + [
                pq_res[name][k] * 100 
                for k in ["pq", "sq", "sqca", "rq", "pr", "prca", "re", "reca"]
            ] + [pq_res[name]["n"]]
            data.append(row)
        table = tabulate(
            data, headers=headers, tablefmt="pipe", floatfmt=".3f", stralign="center", numalign="center"
        )
        logger.info("Panoptic Evaluation Results:\n" + table)

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

            with contextlib.redirect_stdout(io.StringIO()):
                pq_res = pq_compute(
                    gt_json,
                    PathManager.get_local_path(predictions_json),
                    gt_folder=gt_folder,
                    pred_folder=pred_dir,
                )

        res = self.pq_compute_to_res_dic(pq_res)
        results = OrderedDict({"panoptic_seg": res})
        self._print_panoptic_results(pq_res)

        return results


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
        for name in ["All", "Things", "Stuff"]:
            row = [name] + [
                pq_res[name][k] * 100 
                for k in ["pq", "sq", "sqca", "rq", "pr", "prca", "re", "reca"]
            ] + [pq_res[name]["n"]]
            data.append(row)
        table = tabulate(
            data, headers=headers, tablefmt="pipe", floatfmt=".3f", stralign="center", numalign="center"
        )
        logger.info("Panoptic Evaluation Results:\n" + table)