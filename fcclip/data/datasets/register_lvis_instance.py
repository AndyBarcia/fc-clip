"""
This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates.

Reference: https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/datasets/lvis.py
"""

import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.lvis import load_lvis_json

from . import openseg_classes


_PREDEFINED_SPLITS = {
    "openvocab_lvis_v1_val": ("coco/", "lvis/lvis_v1_val.json"),
}


def _get_lvis_instances_meta():
    return {
        "thing_classes": openseg_classes.get_lvis_1203_categories_with_prompt_eng(),
    }


def register_lvis_instances(name, metadata, json_file, image_root):
    DatasetCatalog.register(name, lambda: load_lvis_json(json_file, image_root, dataset_name=None))
    MetadataCatalog.get(name).set(
        json_file=json_file,
        image_root=image_root,
        evaluator_type="lvis",
        **metadata,
    )


def register_all_lvis_instance(root):
    metadata = _get_lvis_instances_meta()
    for key, (image_root, json_file) in _PREDEFINED_SPLITS.items():
        register_lvis_instances(
            key,
            metadata,
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_lvis_instance(_root)
