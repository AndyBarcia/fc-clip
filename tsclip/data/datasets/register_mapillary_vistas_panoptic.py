"""
This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 

Reference: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/data/datasets/register_mapillary_vistas_panoptic.py
"""

import json
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager

from . import openseg_classes

MAPILLARY_VISTAS_SEM_SEG_CATEGORIES = openseg_classes.get_mapillary_vistas_categories_with_prompt_eng()

def load_mapillary_vistas_panoptic_json(json_file, image_dir, gt_dir, semseg_dir, meta):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/coco/train2017".
        gt_dir (str): path to the raw annotations. e.g., "~/coco/panoptic_train2017".
        json_file (str): path to the json file. e.g., "~/coco/annotations/panoptic_train2017.json".
    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """

    def _convert_category_id(segment_info, meta):
        if segment_info["category_id"] in meta["thing_dataset_id_to_contiguous_id"]:
            segment_info["category_id"] = meta["thing_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
            segment_info["isthing"] = True
        else:
            segment_info["category_id"] = meta["stuff_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
            segment_info["isthing"] = False
        return segment_info

    with PathManager.open(json_file) as f:
        json_info = json.load(f)

    ret = []
    for ann in json_info["annotations"]:
        image_id = ann["image_id"]
        # TODO: currently we assume image and label has the same filename but
        # different extension, and images have extension ".jpg" for COCO. Need
        # to make image extension a user-provided argument if we extend this
        # function to support other COCO-like datasets.
        image_file = os.path.join(image_dir, os.path.splitext(ann["file_name"])[0] + ".jpg")
        label_file = os.path.join(gt_dir, ann["file_name"])
        sem_label_file = os.path.join(semseg_dir, ann["file_name"])
        segments_info = [_convert_category_id(x, meta) for x in ann["segments_info"]]
        ret.append(
            {
                "file_name": image_file,
                "image_id": image_id,
                "pan_seg_file_name": label_file,
                "sem_seg_file_name": sem_label_file,
                "segments_info": segments_info,
            }
        )
    assert len(ret), f"No images found in {image_dir}!"
    assert PathManager.isfile(ret[0]["file_name"]), ret[0]["file_name"]
    assert PathManager.isfile(ret[0]["pan_seg_file_name"]), ret[0]["pan_seg_file_name"]
    assert PathManager.isfile(ret[0]["sem_seg_file_name"]), ret[0]["sem_seg_file_name"]
    return ret


def register_mapillary_vistas_panoptic(
    name, metadata, image_root, panoptic_root, semantic_root, panoptic_json, instances_json=None
):
    """
    Register a "standard" version of ADE20k panoptic segmentation dataset named `name`.
    The dictionaries in this registered dataset follows detectron2's standard format.
    Hence it's called "standard".
    Args:
        name (str): the name that identifies a dataset,
            e.g. "ade20k_panoptic_train"
        metadata (dict): extra metadata associated with this dataset.
        image_root (str): directory which contains all the images
        panoptic_root (str): directory which contains panoptic annotation images in COCO format
        panoptic_json (str): path to the json panoptic annotation file in COCO format
        sem_seg_root (none): not used, to be consistent with
            `register_coco_panoptic_separated`.
        instances_json (str): path to the json instance annotation file
    """
    panoptic_name = name
    DatasetCatalog.register(
        panoptic_name,
        lambda: load_mapillary_vistas_panoptic_json(
            panoptic_json, image_root, panoptic_root, semantic_root, metadata
        ),
    )
    MetadataCatalog.get(panoptic_name).set(
        panoptic_root=panoptic_root,
        image_root=image_root,
        panoptic_json=panoptic_json,
        json_file=instances_json,
        evaluator_type="zs_mapillary_vistas_panoptic_seg",
        ignore_label=65,  # different from other datasets, Mapillary Vistas sets ignore_label to 65
        label_divisor=1000,
        **metadata,
    )


_PREDEFINED_SPLITS_ADE20K_PANOPTIC = {
    "openvocab_mapillary_vistas_panoptic_train": (
        "mapillary_vistas/training/images",
        "mapillary_vistas/training/panoptic",
        "mapillary_vistas/training/panoptic/panoptic_2018.json",
        "mapillary_vistas/training/labels",
    ),
    "openvocab_mapillary_vistas_panoptic_val": (
        "mapillary_vistas/validation/images",
        "mapillary_vistas/validation/panoptic",
        "mapillary_vistas/validation/panoptic/panoptic_2018.json",
        "mapillary_vistas/validation/labels",
    ),
}


UNSEEN_MAPILLARY_VISTAS_CATEGORY_IDS = [
    2,  # Ground Animal
    3,  # Curb
    5,  # Guard Rail
    6,  # Barrier
    8,  # Bike Lane
    9,  # Crosswalk - Plain
    10, # Curb Cut
    11, # Parking
    12, # Pedestrian Area
    15, # Service Lane
    16, # Sidewalk
    19, # Tunnel
    21, # Bicyclist
    22, # Motorcyclist
    23, # Other Rider
    24, # Lane Marking - Crosswalk
    25, # Lane Marking - General
    30, # Terrain
    35, # Bike Rack
    36, # Billboard
    37, # Catch Basin
    38, # CCTV Camera
    40, # Junction Box
    41, # Mailbox
    42, # Manhole
    43, # Phone Booth
    44, # Pothole
    45, # Street Light
    46, # Pole
    47, # Traffic Sign Frame
    48, # Utility Pole
    50, # Traffic Sign (Back)
    51, # Traffic Sign (Front)
    52, # Trash Can
    57, # Caravan
    60, # Other Vehicle
    61, # Trailer
    63, # Wheeled Slow
    64, # Car Mount
    65, # Ego Vehicle
]

TYPE_SHIFT_MAPILLARY_VISTAS_CATEGORY_IDS = [
    33,  # Banner (COCO: banner)
]

def get_metadata():
    meta = {}
    # The following metadata maps contiguous id from [0, #thing categories +
    # #stuff categories) to their names and colors. We have to replica of the
    # same name and color under "thing_*" and "stuff_*" because the current
    # visualization function in D2 handles thing and class classes differently
    # due to some heuristic used in Panoptic FPN. We keep the same naming to
    # enable reusing existing visualization functions.
    thing_classes = [k["name"] for k in MAPILLARY_VISTAS_SEM_SEG_CATEGORIES]
    thing_colors = [k["color"] for k in MAPILLARY_VISTAS_SEM_SEG_CATEGORIES]
    stuff_classes = [k["name"] for k in MAPILLARY_VISTAS_SEM_SEG_CATEGORIES]
    stuff_colors = [k["color"] for k in MAPILLARY_VISTAS_SEM_SEG_CATEGORIES]

    seen_classes = [k["name"] for k in MAPILLARY_VISTAS_SEM_SEG_CATEGORIES if k["id"] not in MAPILLARY_VISTAS_SEM_SEG_CATEGORIES]
    seen_colors = [k["color"] for k in MAPILLARY_VISTAS_SEM_SEG_CATEGORIES if k["id"] not in MAPILLARY_VISTAS_SEM_SEG_CATEGORIES]
    unseen_classes = [k["name"] for k in MAPILLARY_VISTAS_SEM_SEG_CATEGORIES if k["id"] in MAPILLARY_VISTAS_SEM_SEG_CATEGORIES]
    unseen_colors = [k["color"] for k in MAPILLARY_VISTAS_SEM_SEG_CATEGORIES if k["id"] in MAPILLARY_VISTAS_SEM_SEG_CATEGORIES]

    thing_mask = [k["isthing"] == 1 for k in MAPILLARY_VISTAS_SEM_SEG_CATEGORIES]

    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors

    meta["seen_classes"] = seen_classes
    meta["seen_colors"] = seen_colors
    meta["unseen_classes"] = unseen_classes
    meta["unseen_colors"] = unseen_colors

    meta["thing_mask"] = thing_mask

    # Convert category id for training:
    #   category id: like semantic segmentation, it is the class id for each
    #   pixel. Since there are some classes not used in evaluation, the category
    #   id is not always contiguous and thus we have two set of category ids:
    #       - original category id: category id in the original dataset, mainly
    #           used for evaluation.
    #       - contiguous category id: [0, #classes), in order to train the linear
    #           softmax classifier.
    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}
    seen_dataset_id_to_contiguous_id = {}
    seen_dataset_id_to_seen_contiguous_id = {}
    unseen_dataset_id_to_contiguous_id = {}

    typeshift_dataset_id_to_contiguous_id = {}

    seen_dataset_id_to_thing_contigous_id = {}
    unseen_dataset_id_to_thing_contigous_id = {}
    last_thing_id = 0

    contiguous_id_to_seen_contiguous_id = []
    last_seen_id = 0

    max_dataset_id = max([ cat["id"] for cat in MAPILLARY_VISTAS_SEM_SEG_CATEGORIES ])
    dataset_id_to_seen_contigous_id = [ -1 for _ in range(max_dataset_id+1) ]

    for i, cat in enumerate(MAPILLARY_VISTAS_SEM_SEG_CATEGORIES):
        if cat["isthing"]:
            thing_dataset_id_to_contiguous_id[cat["id"]] = i
            if cat["id"] in UNSEEN_MAPILLARY_VISTAS_CATEGORY_IDS:
                unseen_dataset_id_to_thing_contigous_id[cat["id"]] = last_thing_id
            else:
                seen_dataset_id_to_thing_contigous_id[cat["id"]] = last_thing_id
            last_thing_id += 1
        # else:
        #     stuff_dataset_id_to_contiguous_id[cat["id"]] = i

        # in order to use sem_seg evaluator
        stuff_dataset_id_to_contiguous_id[cat["id"]] = i

        if cat["id"] in UNSEEN_MAPILLARY_VISTAS_CATEGORY_IDS:
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
        
        if cat["id"] in TYPE_SHIFT_MAPILLARY_VISTAS_CATEGORY_IDS:
            typeshift_dataset_id_to_contiguous_id[cat["id"]] = i

    meta["typeshift_dataset_id_to_contiguous_id"] = typeshift_dataset_id_to_contiguous_id
    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id
    meta["seen_dataset_id_to_contiguous_id"] = seen_dataset_id_to_contiguous_id
    meta["dataset_id_to_seen_contigous_id"] = dataset_id_to_seen_contigous_id
    meta["seen_dataset_id_to_seen_contiguous_id"] = seen_dataset_id_to_seen_contiguous_id
    meta["unseen_dataset_id_to_contiguous_id"] = unseen_dataset_id_to_contiguous_id
    meta["seen_dataset_id_to_thing_contigous_id"] = seen_dataset_id_to_thing_contigous_id
    meta["unseen_dataset_id_to_thing_contigous_id"] = unseen_dataset_id_to_thing_contigous_id
    meta["contiguous_id_to_seen_contiguous_id"] = contiguous_id_to_seen_contiguous_id

    return meta


def register_all_mapillary_vistas_panoptic(root):
    metadata = get_metadata()
    for (
        prefix,
        (image_root, panoptic_root, panoptic_json, semantic_root),
    ) in _PREDEFINED_SPLITS_ADE20K_PANOPTIC.items():
        # The "standard" version of COCO panoptic segmentation dataset,
        # e.g. used by Panoptic-DeepLab
        register_mapillary_vistas_panoptic(
            prefix,
            metadata,
            os.path.join(root, image_root),
            os.path.join(root, panoptic_root),
            os.path.join(root, semantic_root),
            os.path.join(root, panoptic_json),
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_mapillary_vistas_panoptic(_root)