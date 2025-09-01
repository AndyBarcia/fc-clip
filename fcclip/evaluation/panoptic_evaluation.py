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
        self.gt = 0         # Number of unique ground truths for true positives (used to measure duplication rate).
        self.iou_ca = 0.0   # Sum of IoUs for all IoU > 0.5 matches (class-agnostic)
        self.tp_ca = 0      # Total count of IoU > 0.5 matches (class-agnostic)
        self.fp_ca = 0      # False positives (class-agnostic)
        self.fn_ca = 0      # False negatives (class-agnostic)
        self.gt_ca = 0      # Number of unique ground truths for true positives (class-agnostic).
        self.iou_ua = 0.0   # Sum of IoUs for true positives (unique assignment matching).
        self.tp_ua = 0      # True positives (unique assignment matching).
        self.fp_ua = 0      # False positives (unique assignment matching).
        self.fn_ua = 0      # False negatives (unique assignment matching).
        self.iou_uaca = 0.0 # Sum of IOus for true positives (unique assignment matching, class-agnostic).
        self.tp_uaca = 0    # True positives (unique assignment matching, class-agnostic).
        self.fp_uaca = 0    # False positives (unique assignment matching, class-agnostic).
        self.fn_uaca = 0    # False negatives (unique assignment matching, class-agnostic).

    def __iadd__(self, other):
        for attr in [
            'iou', 'tp', 'fp', 'fn', 'gt',
            'iou_ca', 'tp_ca', 'fp_ca', 'fn_ca', 'gt_ca',
            'iou_ua', 'tp_ua', 'fp_ua', 'fn_ua',
            'iou_uaca', 'tp_uaca', 'fp_uaca', 'fn_uaca'
        ]:
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
        pq_sum = sq_sum = rq_sum = 0.0
        tp_sum = fp_sum = fn_sum = gt_sum = 0

        pqca_sum = sqca_sum = rqca_sum = 0.0
        tpca_sum = fpca_sum = fnca_sum = gtca_sum = 0
        
        pqua_sum = squa_sum = rqua_sum = 0.0
        tpua_sum = fpua_sum = fnua_sum = 0
        
        pquaca_sum = squaca_sum = rquaca_sum = 0.0
        tpuaca_sum = fpuaca_sum = fnuaca_sum = 0

        n = 0
        n_ca = 0
        n_ua = 0
        n_uaca = 0
        per_class = {}

        for label, info in categories.items():
            cat_isthing = (info.get('isthing', 0) == 1)
            if isthing is not None and cat_isthing != isthing:
                continue

            stat = self.pq_per_cat[label]

            tp_sum += stat.tp
            fp_sum += stat.fp
            fn_sum += stat.fn
            gt_sum += stat.gt

            tpca_sum += stat.tp_ca
            fpca_sum += stat.fp_ca
            fnca_sum += stat.fn_ca
            gtca_sum += stat.gt_ca
            
            tpua_sum += stat.tp_ua
            fpua_sum += stat.fp_ua
            fnua_sum += stat.fn_ua

            tpuaca_sum += stat.tp_uaca
            fpuaca_sum += stat.fp_uaca
            fnuaca_sum += stat.fn_uaca

            uniqueness_rate = stat.gt / stat.tp if stat.tp > 0 else 0.0
            uniqueness_rate_ca = stat.gt_ca / stat.tp_ca if stat.tp_ca > 0 else 0.0

            denom = stat.tp + 0.5 * stat.fp + 0.5 * stat.fn
            if denom != 0:
                n += 1
                pq_c = stat.iou / denom
                sq_c = stat.iou / stat.tp if stat.tp > 0 else 0
                rq_c = stat.tp / denom

                pq_sum += pq_c
                sq_sum += sq_c
                rq_sum += rq_c
            else:
                pq_c = sq_c = rq_c = 0.0

            denom_ca = stat.tp_ca + 0.5 * stat.fp_ca + 0.5 * stat.fn_ca
            if denom_ca != 0:
                n_ca += 1
                pqca_c = stat.iou_ca / denom_ca
                sqca_c = stat.iou_ca / stat.tp_ca if stat.tp_ca > 0 else 0
                rqca_c = stat.tp_ca / denom_ca

                pqca_sum += pqca_c
                sqca_sum += sqca_c
                rqca_sum += rqca_c
            else:
                pqca_c = sqca_c = rqca_c = 0.0
                
            denom_ua = stat.tp_ua + 0.5 * stat.fp_ua + 0.5 * stat.fn_ua
            if denom_ua != 0:
                n_ua += 1
                pqua_c = stat.iou_ua / denom_ua
                squa_c = stat.iou_ua / stat.tp_ua if stat.tp_ua > 0 else 0
                rqua_c = stat.tp_ua / denom_ua

                pqua_sum += pqua_c
                squa_sum += squa_c
                rqua_sum += rqua_c
            else:
                pqua_c = squa_c = rqua_c = 0.0

            denom_uaca = stat.tp_uaca + 0.5 * stat.fp_uaca + 0.5 * stat.fn_uaca
            if denom_uaca != 0:
                n_uaca += 1
                pquaca_c = stat.iou_uaca / denom_uaca
                squaca_c = stat.iou_uaca / stat.tp_uaca if stat.tp_uaca > 0 else 0
                rquaca_c = stat.tp_uaca / denom_uaca

                pquaca_sum += pquaca_c
                squaca_sum += squaca_c
                rquaca_sum += rquaca_c
            else:
                pquaca_c = squaca_c = rquaca_c = 0.0

            per_class[label] = {
                'pq': pq_c, 'sq': sq_c, 'rq': rq_c,
                'pqca': pqca_c, 'sqca': sqca_c, 'rqca': rqca_c,
                'pqua': pqua_c, 'squa': squa_c, 'rqua': rqua_c,
                'pquaca': pquaca_c, 'squaca': squaca_c, 'rquaca': rquaca_c,
                'tp': stat.tp, 'fp': stat.fp, 'fn': stat.fn,
                'tpca': stat.tp_ca, 'fpca': stat.fp_ca, 'fnca': stat.fn_ca,
                'tpua': stat.tp_ua, 'fpua': stat.fp_ua, 'fnua': stat.fn_ua,
                'tpuaca': stat.tp_uaca, 'fpuaca': stat.fp_uaca, 'fnuaca': stat.fn_uaca,
                'uq': uniqueness_rate, 'uqca': uniqueness_rate_ca,
            }

        # Overall metrics
        pq = pq_sum / n if n > 0 else 0.0
        sq = sq_sum / n if n > 0 else 0.0
        rq = rq_sum / n if n > 0 else 0.0

        pr = (tp_sum / (tp_sum + fp_sum)) if (tp_sum + fp_sum) > 0 else 0.0
        re = (tp_sum / (tp_sum + fn_sum)) if (tp_sum + fn_sum) > 0 else 0.0

        uniqueness_rate = gt_sum / tp_sum if tp_sum > 0 else 0.0

        # Overall class-agnostic metrics
        pqca = pqca_sum / n_ca if n_ca > 0 else 0.0
        sqca = sqca_sum / n_ca if n_ca > 0 else 0.0
        rqca = rqca_sum / n_ca if n_ca > 0 else 0.0

        preca = (tpca_sum / (tpca_sum + fpca_sum)) if (tpca_sum + fpca_sum) > 0 else 0.0
        reca = (tpca_sum / (tpca_sum + fnca_sum)) if (tpca_sum + fnca_sum) > 0 else 0.0

        uniqueness_rate_ca = gtca_sum / tpca_sum if tpca_sum > 0 else 0.0
        
        # Overall unique assignment metrics
        pqua = pqua_sum / n_ua if n_ua > 0 else 0.0
        squa = squa_sum / n_ua if n_ua > 0 else 0.0
        rqua = rqua_sum / n_ua if n_ua > 0 else 0.0

        prua = (tpua_sum / (tpua_sum + fpua_sum)) if (tpua_sum + fpua_sum) > 0 else 0.0
        reua = (tpua_sum / (tpua_sum + fnua_sum)) if (tpua_sum + fnua_sum) > 0 else 0.0

        # Overall unique assignment class-agnostic metrics
        pquaca = pquaca_sum / n_uaca if n_uaca > 0 else 0.0
        squaca = squaca_sum / n_uaca if n_uaca > 0 else 0.0
        rquaca = rquaca_sum / n_uaca if n_uaca > 0 else 0.0

        pruaca = (tpuaca_sum / (tpuaca_sum + fpuaca_sum)) if (tpuaca_sum + fpuaca_sum) > 0 else 0.0
        reuaca = (tpuaca_sum / (tpuaca_sum + fnuaca_sum)) if (tpuaca_sum + fnuaca_sum) > 0 else 0.0

        overall = {
            'pq': pq, 'sq': sq, 'rq': rq,
            'pqca': pqca, 'sqca': sqca, 'rqca': rqca,
            'pqua': pqua, 'squa': squa, 'rqua': rqua,
            'pquaca': pquaca, 'squaca': squaca, 'rquaca': rquaca,
            'pr': pr, 'prca': preca, 're': re, 'reca': reca,
            'prua': prua, 'reua': reua, 'pruaca': pruaca, 'reuaca': reuaca,
            'uq': uniqueness_rate, 'uqca': uniqueness_rate_ca,
            'n': n,
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

        # Matched IDs when taking into account only same-category segments
        matched_gt, matched_pred = set(), set()
        # Matched IDs when taking into account class-agnostic segments
        matched_gt_ca, matched_pred_ca = set(), set()
        # Per-category matched IDs to count number of unique ground truths. 
        per_cat_matched_gt, per_cat_matched_gt_ca = defaultdict(set), defaultdict(set)

        # All potential matches before unique assignment
        potential_matches = defaultdict(list)
        potential_matches_ca = defaultdict(list)

        # Match segments
        for (gt_id, pred_id), inter in gt_pred_map.items():
            if gt_id not in gt_segms or pred_id not in pred_segms:
                continue
            
            # Ignore if crowd
            g, p = gt_segms[gt_id], pred_segms[pred_id]
            if g['iscrowd'] == 1:
                continue

            # Compute IoU of detection and ground truth
            union = p['area'] + g['area'] - inter - gt_pred_map.get((VOID, pred_id), 0)
            iou = inter / union if union > 0 else 0.0

            # If it is a match, count them.
            if iou > 0.5:
                cat = g['category_id']
                stat = pq_stat[cat]
                # Class-agnostic metrics.
                stat.tp_ca += 1
                stat.iou_ca += iou
                matched_gt_ca.add(gt_id)
                matched_pred_ca.add(pred_id)
                per_cat_matched_gt_ca[cat].add(gt_id)
                stat.gt_ca = len(per_cat_matched_gt_ca[cat])
                potential_matches_ca[cat].append((iou, gt_id, pred_id))
                # Class-aware metrics if category matches
                if g['category_id'] == p['category_id']:
                    stat.tp += 1
                    stat.iou += iou
                    matched_gt.add(gt_id)
                    matched_pred.add(pred_id)
                    per_cat_matched_gt[cat].add(gt_id)
                    stat.gt = len(per_cat_matched_gt[cat])
                    potential_matches[cat].append((iou, gt_id, pred_id))

        # Count false negatives
        crowd_by_cat = {}
        crowd_by_cat_ca = {}
        for gt_id, g in gt_segms.items():
            # If the ground truth wasn't matched, count it as a false negative.
            if gt_id not in matched_gt:
                if g['iscrowd'] == 1:
                    crowd_by_cat[g['category_id']] = gt_id
                else:
                    pq_stat[g['category_id']].fn += 1
            # If the ground truth wasn't matched in class-agnostic mode, count it 
            # as a class-agnostic false negative, which is worse than a normal
            # false negative.
            if gt_id not in matched_gt_ca:
                if g['iscrowd'] == 1:
                    crowd_by_cat_ca[g['category_id']] = gt_id
                else:
                    pq_stat[g['category_id']].fn_ca += 1

        # Count false positives
        for pred_id, p in pred_segms.items():
            # If the prediction wasn't matched, count it as a false positive.
            if pred_id not in matched_pred:
                inter = gt_pred_map.get((VOID, pred_id), 0)
                if p['category_id'] in crowd_by_cat:
                    inter += gt_pred_map.get((crowd_by_cat[p['category_id']], pred_id), 0)
                if inter / p['area'] <= 0.5:
                    pq_stat[p['category_id']].fp += 1
            # If the prediction wasn't matched in class-agnostic mode, count it
            # as a class-agnostic false positive, which is worse than a normal
            # false positive.
            if pred_id not in matched_pred_ca:
                inter = gt_pred_map.get((VOID, pred_id), 0)
                if p['category_id'] in crowd_by_cat_ca:
                    inter += gt_pred_map.get((crowd_by_cat_ca[p['category_id']], pred_id), 0)
                if inter / p['area'] <= 0.5:
                    pq_stat[p['category_id']].fp_ca += 1
    
    # Greedy one-to-one matching
    matched_gt, matched_pred = set(), set()
    matched_gt_ca, matched_pred_ca = set(), set()

    # Class-aware matching
    for cat_id, matches in potential_matches.items():
        # Sort matches by IoU in descending order
        matches.sort(key=lambda x: x[0], reverse=True)
        stat = pq_stat[cat_id]
        for iou, gt_id, pred_id in matches:
            if gt_id not in matched_gt and pred_id not in matched_pred:
                stat.tp_ua += 1
                stat.iou_ua += iou
                matched_gt.add(gt_id)
                matched_pred.add(pred_id)

    # Class-agnostic matching
    for cat_id, matches in potential_matches_ca.items():
        matches.sort(key=lambda x: x[0], reverse=True)
        stat = pq_stat[cat_id]
        for iou, gt_id, pred_id in matches:
            if gt_id not in matched_gt_ca and pred_id not in matched_pred_ca:
                stat.tp_uaca += 1
                stat.iou_uaca += iou
                matched_gt_ca.add(gt_id)
                matched_pred_ca.add(pred_id)

    # Count false negatives
    for gt_id, g in gt_segms.items():
        if g['iscrowd'] == 0:
            cat_id = g['category_id']
            if gt_id not in matched_gt:
                pq_stat[cat_id].fn_ua += 1
            if gt_id not in matched_gt_ca:
                pq_stat[cat_id].fn_uaca += 1

    # Count false positives
    for pred_id, p in pred_segms.items():
        cat_id = p['category_id']
        is_matched = pred_id in matched_pred
        is_matched_ca = pred_id in matched_pred_ca

        # Logic to handle predictions overlapping with crowd regions
        inter_with_crowd = 0
        for gt_id, g in gt_segms.items():
            if g['iscrowd'] == 1 and g['category_id'] == cat_id:
                if (gt_id, pred_id) in gt_pred_map:
                    inter_with_crowd += gt_pred_map[(gt_id, pred_id)]

        if not is_matched:
            if inter_with_crowd / p.get('area', 1) <= 0.5:
                pq_stat[cat_id].fp_ua += 1

        if not is_matched_ca:
            if inter_with_crowd / p.get('area', 1) <= 0.5:
                pq_stat[cat_id].fp_uaca += 1

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
        res = {}
        res["PQ"] = 100 * pq_res["All"]["pq"]
        res["SQ"] = 100 * pq_res["All"]["sq"]
        res["RQ"] = 100 * pq_res["All"]["rq"]
        res["PR"] = 100 * pq_res["All"]["pr"]
        res["RE"] = 100 * pq_res["All"]["re"]
        res["UQ"] = 100 * pq_res["All"]["uq"]
        res["PQca"] = 100 * pq_res["All"]["pqca"]
        res["SQca"] = 100 * pq_res["All"]["sqca"]
        res["RQca"] = 100 * pq_res["All"]["rqca"]
        res["PRca"] = 100 * pq_res["All"]["prca"]
        res["REca"] = 100 * pq_res["All"]["reca"]
        res["UQca"] = 100 * pq_res["All"]["uqca"]
        res["PQua"] = 100 * pq_res["All"]["pqua"]
        res["SQua"] = 100 * pq_res["All"]["squa"]
        res["RQua"] = 100 * pq_res["All"]["rqua"]
        res["PRua"] = 100 * pq_res["All"]["prua"]
        res["REua"] = 100 * pq_res["All"]["reua"]
        res["PQuaca"] = 100 * pq_res["All"]["pquaca"]
        res["SQuaca"] = 100 * pq_res["All"]["squaca"]
        res["RQuaca"] = 100 * pq_res["All"]["rquaca"]
        res["PRuaca"] = 100 * pq_res["All"]["pruaca"]
        res["REuaca"] = 100 * pq_res["All"]["reuaca"]

        res["PQ_th"] = 100 * pq_res["Things"]["pq"]
        res["SQ_th"] = 100 * pq_res["Things"]["sq"]
        res["RQ_th"] = 100 * pq_res["Things"]["rq"]
        res["PR_th"] = 100 * pq_res["Things"]["pr"]
        res["RE_th"] = 100 * pq_res["Things"]["re"]
        res["UQ_th"] = 100 * pq_res["Things"]["uq"]
        res["PQca_th"] = 100 * pq_res["Things"]["pqca"]
        res["SQca_th"] = 100 * pq_res["Things"]["sqca"]
        res["RQca_th"] = 100 * pq_res["Things"]["rqca"]
        res["PRca_th"] = 100 * pq_res["Things"]["prca"]
        res["REca_th"] = 100 * pq_res["Things"]["reca"]
        res["UQca_th"] = 100 * pq_res["Things"]["uqca"]
        res["PQua_th"] = 100 * pq_res["Things"]["pqua"]
        res["SQua_th"] = 100 * pq_res["Things"]["squa"]
        res["RQua_th"] = 100 * pq_res["Things"]["rqua"]
        res["PRua_th"] = 100 * pq_res["Things"]["prua"]
        res["REua_th"] = 100 * pq_res["Things"]["reua"]
        res["PQuaca_th"] = 100 * pq_res["Things"]["pquaca"]
        res["SQuaca_th"] = 100 * pq_res["Things"]["squaca"]
        res["RQuaca_th"] = 100 * pq_res["Things"]["rquaca"]
        res["PRuaca_th"] = 100 * pq_res["Things"]["pruaca"]
        res["REuaca_th"] = 100 * pq_res["Things"]["reuaca"]

        res["PQ_st"] = 100 * pq_res["Stuff"]["pq"]
        res["SQ_st"] = 100 * pq_res["Stuff"]["sq"]
        res["RQ_st"] = 100 * pq_res["Stuff"]["rq"]
        res["PR_st"] = 100 * pq_res["Stuff"]["pr"]
        res["RE_st"] = 100 * pq_res["Stuff"]["re"]
        res["UQ_st"] = 100 * pq_res["Stuff"]["uq"]
        res["PQca_st"] = 100 * pq_res["Stuff"]["pqca"]
        res["SQca_st"] = 100 * pq_res["Stuff"]["sqca"]
        res["RQca_st"] = 100 * pq_res["Stuff"]["rqca"]
        res["PRca_st"] = 100 * pq_res["Stuff"]["prca"]
        res["REca_st"] = 100 * pq_res["Stuff"]["reca"]
        res["UQca_st"] = 100 * pq_res["Stuff"]["uqca"]
        res["PQua_st"] = 100 * pq_res["Stuff"]["pqua"]
        res["SQua_st"] = 100 * pq_res["Stuff"]["squa"]
        res["RQua_st"] = 100 * pq_res["Stuff"]["rqua"]
        res["PRua_st"] = 100 * pq_res["Stuff"]["prua"]
        res["REua_st"] = 100 * pq_res["Stuff"]["reua"]
        res["PQuaca_st"] = 100 * pq_res["Stuff"]["pquaca"]
        res["SQuaca_st"] = 100 * pq_res["Stuff"]["squaca"]
        res["RQuaca_st"] = 100 * pq_res["Stuff"]["rquaca"]
        res["PRuaca_st"] = 100 * pq_res["Stuff"]["pruaca"]
        res["REuaca_st"] = 100 * pq_res["Stuff"]["reuaca"]
        return res

    def _print_panoptic_results(self, pq_res):
        headers = ["", "PQ", "SQ", "RQ", "PR", "RE", "UQ", "#categories"]
        data = []
        for name in ["All", "Things", "Stuff"]:
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
        for name in ["All", "Things", "Stuff"]:
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
        for name in ["All", "Things", "Stuff"]:
            row = [name] + [
                pq_res[name][k] * 100 for k in ["pqua", "squa", "rqua", "prua", "reua"]
            ] + [pq_res[name]["n"]]
            data.append(row)
        table = tabulate(
            data, headers=headers, tablefmt="pipe", floatfmt=".3f", stralign="center", numalign="center"
        )
        logger.info("Panoptic Evaluation Results (Unique Assignment):\n" + table)
        
        data = []
        for name in ["All", "Things", "Stuff"]:
            row = [name] + [
                pq_res[name][k] * 100 for k in ["pquaca", "squaca", "rquaca", "pruaca", "reuaca"]
            ] + [pq_res[name]["n"]]
            data.append(row)
        table = tabulate(
            data, headers=headers, tablefmt="pipe", floatfmt=".3f", stralign="center", numalign="center"
        )
        logger.info("Panoptic Evaluation Results (Unique Assignment, Class-Agnostic):\n" + table)

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
        
        headers = ["", "PQ", "SQ", "RQ", "PR", "RE", "UQ", "#categories"]
        data = []
        for name in ["All", "Things", "Stuff"]:
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
        for name in ["All", "Things", "Stuff"]:
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
        for name in ["All", "Things", "Stuff"]:
            row = [name] + [
                pq_res[name][k] * 100 for k in ["pqua", "squa", "rqua", "prua", "reua"]
            ] + [pq_res[name]["n"]]
            data.append(row)
        table = tabulate(
            data, headers=headers, tablefmt="pipe", floatfmt=".3f", stralign="center", numalign="center"
        )
        logger.info("Panoptic Evaluation Results (Unique Assignment):\n" + table)
        
        data = []
        for name in ["All", "Things", "Stuff"]:
            row = [name] + [
                pq_res[name][k] * 100 for k in ["pquaca", "squaca", "rquaca", "pruaca", "reuaca"]
            ] + [pq_res[name]["n"]]
            data.append(row)
        table = tabulate(
            data, headers=headers, tablefmt="pipe", floatfmt=".3f", stralign="center", numalign="center"
        )
        logger.info("Panoptic Evaluation Results (Unique Assignment, Class-Agnostic):\n" + table)