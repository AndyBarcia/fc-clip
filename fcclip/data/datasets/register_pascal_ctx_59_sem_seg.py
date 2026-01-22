"""
Copyright (2023) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License. 
"""

import os

import numpy as np

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

from . import openseg_classes

PASCAL_CTX_59_CATEGORIES=openseg_classes.get_pascal_ctx_59_categories_with_prompt_eng()

PASCAL_CTX_59_COLORS = [k["color"] for k in PASCAL_CTX_59_CATEGORIES]

MetadataCatalog.get("openvocab_pascal_ctx59_sem_seg_train").set(
    stuff_colors=PASCAL_CTX_59_COLORS[:],
)

MetadataCatalog.get("openvocab_pascal_ctx59_sem_seg_val").set(
    stuff_colors=PASCAL_CTX_59_COLORS[:],
)

UNSEEN_PASCAL_CTX_59_CLASSES = [
    1, # bag,bags
    3, # bedclothes
    18, # computer case
    29, # ground,soil,soil ground,dirt ground
    40, # street,streets
    44, # sidewalk
    45, # sign,signs
    58, # wood piece
]

def _get_ctx59_meta():
    meta = {}
    # Id 0 is reserved for ignore_label, we change ignore_label for 0
    # to 255 in our pre-processing, so all ids are shifted by 1.
    stuff_ids = [k["id"] for k in PASCAL_CTX_59_CATEGORIES]
    assert len(stuff_ids) == 459, len(stuff_ids)

    stuff_classes = [k["name"] for k in PASCAL_CTX_59_CATEGORIES]

    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}
    seen_dataset_id_to_contiguous_id = {}
    seen_dataset_id_to_seen_contiguous_id = {}
    unseen_dataset_id_to_contiguous_id = {}

    seen_dataset_id_to_thing_contigous_id = {}
    unseen_dataset_id_to_thing_contigous_id = {}
    last_thing_id = 0

    contiguous_id_to_seen_contiguous_id = []
    last_seen_id = 0

    max_dataset_id = max([ cat["id"] for cat in PASCAL_CTX_59_CATEGORIES ])
    dataset_id_to_seen_contigous_id = [ -1 for _ in range(max_dataset_id+1) ]

    for i, cat in enumerate(PASCAL_CTX_59_CATEGORIES):
        if cat["isthing"]:
            thing_dataset_id_to_contiguous_id[cat["id"]] = i
            if cat["id"] in UNSEEN_PASCAL_CTX_59_CLASSES:
                unseen_dataset_id_to_thing_contigous_id[cat["id"]] = last_thing_id
            else:
                seen_dataset_id_to_thing_contigous_id[cat["id"]] = last_thing_id
            last_thing_id += 1
        # else:
        #     stuff_dataset_id_to_contiguous_id[cat["id"]] = i
        
        # in order to use sem_seg evaluator
        stuff_dataset_id_to_contiguous_id[cat["id"]] = i

        if cat["id"] in UNSEEN_PASCAL_CTX_59_CLASSES:
            # If this category is unseen, map it to -1 category.
            # This allows then easy filtering on unseen categories.
            contiguous_id_to_seen_contiguous_id.append(-1)
            unseen_dataset_id_to_contiguous_id[cat["id"]] = i
        else:
            contiguous_id_to_seen_contiguous_id.append(last_seen_id)
            seen_dataset_id_to_seen_contiguous_id[cat["id"]] = last_seen_id
            dataset_id_to_seen_contigous_id[cat["id"]] = last_seen_id
            last_seen_id += 1
            seen_dataset_id_to_contiguous_id[cat["id"]] = i


    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id
    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["seen_dataset_id_to_contiguous_id"] = seen_dataset_id_to_contiguous_id
    meta["dataset_id_to_seen_contigous_id"] = dataset_id_to_seen_contigous_id
    meta["seen_dataset_id_to_seen_contiguous_id"] = seen_dataset_id_to_seen_contiguous_id
    meta["unseen_dataset_id_to_contiguous_id"] = unseen_dataset_id_to_contiguous_id
    meta["seen_dataset_id_to_thing_contigous_id"] = seen_dataset_id_to_thing_contigous_id
    meta["unseen_dataset_id_to_thing_contigous_id"] = unseen_dataset_id_to_thing_contigous_id
    meta["contiguous_id_to_seen_contiguous_id"] = contiguous_id_to_seen_contiguous_id

    meta["stuff_classes"] = stuff_classes
    return meta



def register_all_ctx59(root):
    root = os.path.join(root, "pascal_ctx_d2")
    meta = _get_ctx59_meta()
    for name, dirname in [("train", "training"), ("val", "validation")]:
        image_dir = os.path.join(root, "images", dirname)
        gt_dir = os.path.join(root, "annotations_ctx59", dirname)
        name = f"openvocab_pascal_ctx59_sem_seg_{name}"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
        )
        MetadataCatalog.get(name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            gt_ext="png",
            **meta
        )
        
_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_ctx59(_root)