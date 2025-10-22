"""
This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 

Reference: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/data/dataset_mappers/coco_panoptic_new_baseline_dataset_mapper.py
"""

import copy
import logging

import numpy as np
import torch

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.transforms import TransformGen
from detectron2.structures import BitMasks, Boxes, Instances

__all__ = ["COCOPanopticNewBaselineDatasetMapper"]


def build_transform_gen(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    assert is_train, "Only support training augmentation"
    image_size = cfg.INPUT.IMAGE_SIZE
    min_scale = cfg.INPUT.MIN_SCALE
    max_scale = cfg.INPUT.MAX_SCALE

    augmentation = []

    if cfg.INPUT.RANDOM_FLIP != "none":
        augmentation.append(
            T.RandomFlip(
                horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
                vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
            )
        )

    augmentation.extend([
        T.ResizeScale(
            min_scale=min_scale, max_scale=max_scale, target_height=image_size, target_width=image_size
        ),
        T.FixedSizeCrop(crop_size=(image_size, image_size)),
    ])

    return augmentation


# This is specifically designed for the COCO dataset.
class COCOPanopticNewBaselineDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer.

    This dataset mapper applies the same transformation as DETR for COCO panoptic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        tfm_gens,
        image_format,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            crop_gen: crop augmentation
            tfm_gens: data augmentation
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        self.tfm_gens = tfm_gens
        logging.getLogger(__name__).info(
            "[COCOPanopticNewBaselineDatasetMapper] Full TransformGens used in training: {}".format(
                str(self.tfm_gens)
            )
        )

        self.img_format = image_format
        self.is_train = is_train

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        tfm_gens = build_transform_gen(cfg, is_train)

        ret = {
            "is_train": is_train,
            "tfm_gens": tfm_gens,
            "image_format": cfg.INPUT.FORMAT,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "pan_seg_file_name" in dataset_dict:
            pan_seg_gt = utils.read_image(dataset_dict.pop("pan_seg_file_name"), "RGB")
            segments_info = dataset_dict["segments_info"]

            # apply the same transformation to panoptic segmentation
            pan_seg_gt = transforms.apply_segmentation(pan_seg_gt)

            from panopticapi.utils import rgb2id

            # Work in torch without allocating huge [max_id + 1] arrays.
            pan_seg_gt = rgb2id(pan_seg_gt)
            pan_seg_gt = torch.from_numpy(pan_seg_gt.astype(np.int64))  # (H,W), long for torch.unique

            # Unique IDs actually present in the (possibly cropped/augmented) image
            unique_ids, inv = torch.unique(pan_seg_gt, return_inverse=True)  # unique_ids: (K,), inv: (H*W,)
            K = unique_ids.numel()

            # We'll map from the K unique IDs -> sequential IDs and class IDs.
            # This will be filled with information from the segments_info array.
            # If some ID present in the panoptic map doesn't have its corresponding
            # segment, it will default to 0 for the panoptic map (background) and
            # to -1 for the label.
            seq_map = torch.zeros((K,), dtype=torch.int64)
            cls_map = torch.full((K,), -1, dtype=torch.int64)

            # Fast lookup: unique ID value -> position in unique_ids
            uid2pos = {int(uid.item()): i for i, uid in enumerate(unique_ids)}

            # Keep class labels in the same order as segments_info (skipping crowd), like before
            classes = []
            for idx, segment_info in enumerate(segments_info):
                if segment_info["iscrowd"]:
                    continue
                classes.append(int(segment_info["category_id"]))

            # Fill mapping tables using the SAME sequential convention (idx+1 over segments_info)
            for idx, segment_info in enumerate(segments_info):
                if segment_info["iscrowd"]:
                    continue
                sid = int(segment_info["id"])
                pos = uid2pos.get(sid)
                if pos is not None:  # segment might be fully cropped out
                    seq_map[pos] = idx + 1
                    cls_map[pos] = int(segment_info["category_id"])

            # Apply the mapping back to image shape using the inverse index
            pan_seg_gt_sequential = seq_map[inv].reshape(pan_seg_gt.shape)  # (H,W) int64 default 0
            sem_seg_gt = cls_map[inv].reshape(pan_seg_gt.shape)             # (H,W) int64 default -1
            
            gt_boxes = [] # (GT,4)
            for idx, segment_info in enumerate(segments_info):
                if segment_info["iscrowd"]:
                    continue

                mask = (pan_seg_gt_sequential == idx+1)
                if not mask.any():
                    gt_boxes.append([0, 0, 0, 0])
                    continue
                
                H,W = pan_seg_gt_sequential.shape
                ys, xs = np.where(mask)
                x_min = int(xs.min()) / W
                y_min = int(ys.min()) / H
                x_max = int(xs.max()) / W
                y_max = int(ys.max()) / H

                # xyxy unnormalized format
                gt_boxes.append([x_min, y_min, x_max, y_max])
            gt_boxes = torch.tensor(gt_boxes)

            dataset_dict["instances"] = {
                "pan_seg": pan_seg_gt_sequential, # (H,W) int64 default 0
                "sem_seg": sem_seg_gt, # (H,W) int64 default -1
                "labels": cls_map, # (GT_max) int64 default -1
                "boxes": gt_boxes # (GT,4)
            }

        return dataset_dict