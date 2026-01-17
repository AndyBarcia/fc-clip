"""
This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 

Reference: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/data/datasets/register_ade20k_panoptic.py
"""

import json
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager
from detectron2.data.datasets.coco import load_sem_seg


from . import openseg_classes

ADE20K_150_CATEGORIES = openseg_classes.get_ade20k_categories_with_prompt_eng()

ADE20k_COLORS = [k["color"] for k in ADE20K_150_CATEGORIES]

MetadataCatalog.get("openvocab_ade20k_sem_seg_train").set(
    stuff_colors=ADE20k_COLORS[:],
)

MetadataCatalog.get("openvocab_ade20k_sem_seg_val").set(
    stuff_colors=ADE20k_COLORS[:],
)


def load_ade20k_panoptic_json(json_file, image_dir, gt_dir, semseg_dir, meta):
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


def register_ade20k_panoptic(
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
        lambda: load_ade20k_panoptic_json(
            panoptic_json, image_root, panoptic_root, semantic_root, metadata
        ),
    )
    MetadataCatalog.get(panoptic_name).set(
        panoptic_root=panoptic_root,
        image_root=image_root,
        panoptic_json=panoptic_json,
        semantic_root=semantic_root,
        json_file=instances_json,
        evaluator_type="zs_ade20k_panoptic_seg",
        ignore_label=255,
        label_divisor=1000,
        **metadata,
    )


_PREDEFINED_SPLITS_ADE20K_PANOPTIC = {
    "openvocab_ade20k_panoptic_train": (
        "ADEChallengeData2016/images/training",
        "ADEChallengeData2016/ade20k_panoptic_train",
        "ADEChallengeData2016/ade20k_panoptic_train.json",
        "ADEChallengeData2016/annotations_detectron2/training",
        "ADEChallengeData2016/ade20k_instance_train.json",
    ),
    "openvocab_ade20k_panoptic_val": (
        "ADEChallengeData2016/images/validation",
        "ADEChallengeData2016/ade20k_panoptic_val",
        "ADEChallengeData2016/ade20k_panoptic_val.json",
        "ADEChallengeData2016/annotations_detectron2/validation",
        "ADEChallengeData2016/ade20k_instance_val.json",
    ),
}


UNSEEN_ADE20K_CATEGORY_IDS =  [
    # 93 Categories in ADE20K not present in COCO.
    13, # ground
    17, # plant
    22, # picture
    23, # sofa
    29, # field
    30, # armchair
    31, # seat
    33, # desk
    35, # press
    36, # lamp
    37, # tub
    38, # rail
    39, # cushion
    40, # stand
    41, # box
    42, # pillar
    43, # sign
    44, # dresser
    48, # skyscraper
    49, # fireplace
    51, # covered stand
    52, # path
    54, # runway
    55, # vitrine
    56, # snooker table
    58, # screen
    59, # staircase
    62, # bookcase
    64, # coffee table
    68, # hill
    70, # countertop
    71, # stove
    72, # palm tree
    73, # kitchen island
    74, # computer
    75, # swivel chair
    77, # bar
    78, # arcade machine
    79, # shanty
    84, # tower
    85, # chandelier
    86, # sunblind
    87, # street lamp
    88, # booth
    90, # plane
    91, # dirt track
    92, # clothes
    93, # pole
    94, # soil
    95, # handrail
    96, # moving stairway
    97, # hassock
    100, # card
    101, # stage
    102, # van
    103, # ship
    104, # fountain
    105, # transporter
    106, # canopy
    107, # washing machine
    108, # toy
    109, # pool
    110, # stool
    111, # cask
    112, # handbasket
    113, # falls
    115, # bag
    116, # motorbike
    117, # cradle
    119, # ball
    121, # stair
    122, # storage tank
    123, # trade name
    125, # pot
    126, # animal
    128, # lake
    129, # dishwasher
    130, # screen
    132, # sculpture
    133, # exhaust hood
    134, # sconce
    137, # tray
    138, # trash can
    139, # fan
    140, # pier
    141, # crt screen
    142, # plate
    143, # monitor
    144, # bulletin board
    145, # shower
    146, # radiator
    147, # drinking glass
    149, # flag
]


def get_metadata():
    meta = {}
    # The following metadata maps contiguous id from [0, #thing categories +
    # #stuff categories) to their names and colors. We have to replica of the
    # same name and color under "thing_*" and "stuff_*" because the current
    # visualization function in D2 handles thing and class classes differently
    # due to some heuristic used in Panoptic FPN. We keep the same naming to
    # enable reusing existing visualization functions.
    thing_classes = [k["name"] for k in ADE20K_150_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in ADE20K_150_CATEGORIES if k["isthing"] == 1]
    stuff_classes = [k["name"] for k in ADE20K_150_CATEGORIES]
    stuff_colors = [k["color"] for k in ADE20K_150_CATEGORIES]

    seen_classes = [k["name"] for k in ADE20K_150_CATEGORIES if k["id"] not in UNSEEN_ADE20K_CATEGORY_IDS]
    seen_colors = [k["color"] for k in ADE20K_150_CATEGORIES if k["id"] not in UNSEEN_ADE20K_CATEGORY_IDS]
    unseen_classes = [k["name"] for k in ADE20K_150_CATEGORIES if k["id"] in UNSEEN_ADE20K_CATEGORY_IDS]
    unseen_colors = [k["color"] for k in ADE20K_150_CATEGORIES if k["id"] in UNSEEN_ADE20K_CATEGORY_IDS]

    thing_mask = [k["isthing"] == 1 for k in ADE20K_150_CATEGORIES]

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

    seen_dataset_id_to_thing_contigous_id = {}
    unseen_dataset_id_to_thing_contigous_id = {}
    last_thing_id = 0

    contiguous_id_to_seen_contiguous_id = []
    last_seen_id = 0

    max_dataset_id = max([ cat["id"] for cat in ADE20K_150_CATEGORIES ])
    dataset_id_to_seen_contigous_id = [ -1 for _ in range(max_dataset_id+1) ]

    for i, cat in enumerate(ADE20K_150_CATEGORIES):
        if cat["isthing"]:
            thing_dataset_id_to_contiguous_id[cat["id"]] = i
            if cat["id"] in UNSEEN_ADE20K_CATEGORY_IDS:
                unseen_dataset_id_to_thing_contigous_id[cat["id"]] = last_thing_id
            else:
                seen_dataset_id_to_thing_contigous_id[cat["id"]] = last_thing_id
            last_thing_id += 1
        # else:
        #     stuff_dataset_id_to_contiguous_id[cat["id"]] = i
        
        # in order to use sem_seg evaluator
        stuff_dataset_id_to_contiguous_id[cat["id"]] = i

        if cat["id"] in UNSEEN_ADE20K_CATEGORY_IDS:
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


def register_all_ade20k_panoptic(root):
    metadata = get_metadata()
    for (
        prefix,
        (image_root, panoptic_root, panoptic_json, semantic_root, instance_json),
    ) in _PREDEFINED_SPLITS_ADE20K_PANOPTIC.items():
        # The "standard" version of COCO panoptic segmentation dataset,
        # e.g. used by Panoptic-DeepLab
        register_ade20k_panoptic(
            prefix,
            metadata,
            os.path.join(root, image_root),
            os.path.join(root, panoptic_root),
            os.path.join(root, semantic_root),
            os.path.join(root, panoptic_json),
            os.path.join(root, instance_json),
        )

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_ade20k_panoptic(_root)
