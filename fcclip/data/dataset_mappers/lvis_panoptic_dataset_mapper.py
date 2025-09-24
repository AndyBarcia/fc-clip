"""
This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 

Reference: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/data/dataset_mappers/coco_panoptic_new_baseline_dataset_mapper.py
"""

import copy
import logging
import cv2

import numpy as np
import torch

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.transforms import TransformGen
from detectron2.structures import BitMasks, Boxes, Instances

__all__ = ["LVISPanopticDatasetMapper"]


def build_transform_gen(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    This now includes resizing and flipping.
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


def coco_segm_to_poly(segm):
    """
    Converts COCO segmentation format to a list of polygons.

    The COCO segmentation format can be a list of polygons, where each
    polygon is a flat list of [x1, y1, x2, y2, ...].

    Args:
        segm (list[list[float]]): The segmentation data in COCO format.

    Returns:
        list[np.ndarray]: A list of numpy arrays, where each array has a
                          shape of (n, 2) representing a polygon's vertices.
    """
    polygons = []
    for poly_list in segm:
        # COCO polygons are flat lists of [x1, y1, x2, y2, ...]
        # We reshape them into (n, 2) arrays.
        points = np.asarray(poly_list, dtype=np.int32).reshape(-1, 2)
        polygons.append(points)
    return polygons


# This is specifically designed for the COCO dataset.
class LVISPanopticDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and maps it into a format used by MaskFormer.

    This dataset mapper applies the same transformation as DETR for COCO panoptic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name".
    2. Converts polygon segmentations from the dataset into binary instance masks.
    3. Applies geometric transforms (e.g., resizing, flipping) to the image and masks.
    4. Finds and applies suitable cropping to the image and masks.
    5. Prepares the image and instance annotations (classes, masks, boxes) as Tensors.
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
            is_train (bool): whether for training or inference
            tfm_gens (list[TransformGen]): data augmentation strategy
            image_format (str): an image format supported by :func:`detection_utils.read_image`.
        """
        self.tfm_gens = tfm_gens
        logging.getLogger(__name__).info(
            "[LVISPanopticDatasetMapper] Full TransformGens used in training: {}".format(
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

        # Apply transformations to the image
        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("segments_info", None)
            return dataset_dict

        if "segments_info" in dataset_dict:
            segments_info = dataset_dict["segments_info"]
            
            # Create an Instances object to store annotations.
            instances = Instances(image_shape)
            classes = []
            masks = []
            
            for segment_info in segments_info:
                classes.append(segment_info["category_id"])
                
                segmentation = segment_info["segmentation"]
                
                # Generate a binary mask from the polygon segmentation.
                # 1. Create a blank mask with the original image dimensions.
                h, w = dataset_dict["height"], dataset_dict["width"]
                segment_mask = np.zeros((h, w), dtype=np.uint8)
                
                # 2. Convert COCO RLE/polygon format to a list of polygons.
                polygons = coco_segm_to_poly(segmentation)
                
                # 3. Fill the polygons on the blank mask.
                cv2.fillPoly(segment_mask, pts=polygons, color=1)

                # Apply the same geometric transformations to the mask
                segment_mask = transforms.apply_segmentation(segment_mask)
                masks.append(segment_mask)

            classes = np.array(classes)
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)

            if len(masks) == 0:
                # Handle images that have no annotations
                instances.gt_masks = torch.zeros((0, image_shape[0], image_shape[1]))
                instances.gt_boxes = Boxes(torch.zeros((0, 4)))
            else:
                # Convert the list of numpy masks to a BitMasks object
                masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
                )
                instances.gt_masks = masks.tensor
                instances.gt_boxes = masks.get_bounding_boxes()

            dataset_dict["instances"] = instances

        return dataset_dict