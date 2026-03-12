"""Register ADE20K-612 semantic segmentation dataset."""

import os

import numpy as np

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

from . import openseg_classes

ADE20K_612_CATEGORIES = openseg_classes.get_ade20k_612_categories_with_prompt_eng()

UNSEEN_ADE20K_612_CATEGORY_IDS = [8, 17, 24, 25, 26, 27, 28, 29, 30, 31, 33, 35, 38, 40, 41, 42, 44, 46, 47, 48, 49, 51, 52, 53, 55, 56, 57, 59, 61, 62, 64, 66, 67, 68, 71, 72, 73, 74, 75, 76, 78, 80, 82, 83, 84, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 98, 99, 101, 102, 104, 105, 107, 109, 111, 112, 114, 115, 116, 117, 118, 119, 120, 121, 122, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 139, 140, 141, 142, 143, 144, 145, 146, 148, 149, 151, 152, 153, 154, 155, 156, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 266, 267, 268, 269, 270, 271, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 311, 312, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611]

ADE20K_612_COLORS = [
    np.random.randint(256, size=3).tolist() for _ in ADE20K_612_CATEGORIES
]

MetadataCatalog.get("ade20k_612_sem_seg_train").set(stuff_colors=ADE20K_612_COLORS[:])
MetadataCatalog.get("ade20k_612_sem_seg_val").set(stuff_colors=ADE20K_612_COLORS[:])


def get_metadata():
    meta = {}

    stuff_classes = [k["name"] for k in ADE20K_612_CATEGORIES]
    assert len(stuff_classes) == 612, len(stuff_classes)

    stuff_dataset_id_to_contiguous_id = {}
    seen_dataset_id_to_contiguous_id = {}
    seen_dataset_id_to_seen_contiguous_id = {}
    unseen_dataset_id_to_contiguous_id = {}
    contiguous_id_to_seen_contiguous_id = []
    last_seen_id = 0

    max_dataset_id = max([cat["trainId"] for cat in ADE20K_612_CATEGORIES])
    dataset_id_to_seen_contigous_id = [-1 for _ in range(max_dataset_id + 1)]

    unseen_set = set(UNSEEN_ADE20K_612_CATEGORY_IDS)

    for i, cat in enumerate(ADE20K_612_CATEGORIES):
        dataset_id = cat["trainId"]
        stuff_dataset_id_to_contiguous_id[dataset_id] = i

        if dataset_id in unseen_set:
            contiguous_id_to_seen_contiguous_id.append(-1)
            unseen_dataset_id_to_contiguous_id[dataset_id] = i
        else:
            contiguous_id_to_seen_contiguous_id.append(last_seen_id)
            seen_dataset_id_to_seen_contiguous_id[dataset_id] = last_seen_id
            dataset_id_to_seen_contigous_id[dataset_id] = last_seen_id
            last_seen_id += 1
            seen_dataset_id_to_contiguous_id[dataset_id] = i

    meta["stuff_classes"] = stuff_classes
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id
    meta["seen_dataset_id_to_contiguous_id"] = seen_dataset_id_to_contiguous_id
    meta["dataset_id_to_seen_contigous_id"] = dataset_id_to_seen_contigous_id
    meta["seen_dataset_id_to_seen_contiguous_id"] = seen_dataset_id_to_seen_contiguous_id
    meta["unseen_dataset_id_to_contiguous_id"] = unseen_dataset_id_to_contiguous_id
    meta["contiguous_id_to_seen_contiguous_id"] = contiguous_id_to_seen_contiguous_id

    return meta


def register_all_ade20k_612(root):
    root = os.path.join(root, "ADE20K_2021_17_01")
    metadata = get_metadata()

    for split_name, dirname in [("train", "training"), ("val", "validation")]:
        image_dir = os.path.join(root, "images_detectron2", dirname)
        gt_dir = os.path.join(root, "annotations_detectron2_superclass_merged_equiv_612", dirname)
        dataset_name = f"ade20k_612_sem_seg_{split_name}"

        DatasetCatalog.register(
            dataset_name,
            lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="tif", image_ext="jpg"),
        )
        MetadataCatalog.get(dataset_name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="zs_sem_seg",
            ignore_label=65535,
            gt_ext="tif",
            **metadata,
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_ade20k_612(_root)
