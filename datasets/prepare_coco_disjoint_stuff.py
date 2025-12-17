"""Split COCO panoptic stuff segments into disjoint components ahead of training."""
import argparse
import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def rgb2id(color):
    if isinstance(color, np.ndarray) and len(color.shape) == 3:
        if color.dtype == np.uint8:
            color = color.astype(np.int32)
        return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
    return int(color[0] + 256 * color[1] + 256 * 256 * color[2])


def id2rgb(id_map):
    if isinstance(id_map, np.ndarray):
        id_map_copy = id_map.copy()
        rgb_shape = tuple(list(id_map.shape) + [3])
        rgb_map = np.zeros(rgb_shape, dtype=np.uint8)
        for i in range(3):
            rgb_map[..., i] = id_map_copy % 256
            id_map_copy //= 256
        return rgb_map
    color = []
    for _ in range(3):
        color.append(id_map % 256)
        id_map //= 256
    return color


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split stuff segments in COCO panoptic annotations into connected components "
            "and save a new panoptic dataset."
        )
    )
    parser.add_argument(
        "--panoptic-json",
        required=True,
        help="Path to the original COCO panoptic json file (e.g. panoptic_train2017.json)",
    )
    parser.add_argument(
        "--panoptic-root",
        required=True,
        help="Directory containing the original COCO panoptic PNG annotations",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Path to save the new json with split stuff segments",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Directory to save the new panoptic PNG annotations",
    )
    return parser.parse_args()


def load_category_isthing_map(panoptic_json: Dict) -> Dict[int, bool]:
    category_isthing = {}
    for category in panoptic_json.get("categories", []):
        category_isthing[int(category["id"])] = bool(category.get("isthing", 0))
    if not category_isthing:
        raise ValueError("No categories with isthing flag found in the panoptic json.")
    return category_isthing


def compute_bbox(mask: np.ndarray) -> List[int]:
    y_indices, x_indices = np.where(mask)
    if len(y_indices) == 0:
        return [0, 0, 0, 0]
    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()
    return [int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1)]


def process_annotation(
    annotation: Dict,
    category_isthing: Dict[int, bool],
    panoptic_root: Path,
    output_root: Path,
) -> Dict:
    image_path = panoptic_root / annotation["file_name"]
    panoptic_np = np.asarray(Image.open(image_path), dtype=np.uint8)
    panoptic_ids = rgb2id(panoptic_np)

    new_panoptic = np.zeros_like(panoptic_ids, dtype=np.int32)
    new_segments_info: List[Dict] = []

    next_segment_id = 1  # 0 is reserved for void
    for segment in annotation["segments_info"]:
        segment_id = segment["id"]
        category_id = int(segment["category_id"])
        mask = panoptic_ids == segment_id
        if not mask.any():
            continue

        isthing = category_isthing.get(category_id, False)
        if isthing:
            component_masks = [mask]
        else:
            num_components, labels = cv2.connectedComponents(mask.astype(np.uint8))
            component_masks = [labels == i for i in range(1, num_components)]

        for component_mask in component_masks:
            if not component_mask.any():
                continue
            current_id = next_segment_id
            next_segment_id += 1
            new_panoptic[component_mask] = current_id
            area = int(component_mask.sum())
            bbox = compute_bbox(component_mask)
            new_segments_info.append(
                {
                    "id": current_id,
                    "category_id": category_id,
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": int(segment.get("iscrowd", 0)),
                }
            )

    output_path = output_root / annotation["file_name"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(id2rgb(new_panoptic)).save(output_path)

    new_annotation = dict(annotation)
    new_annotation["segments_info"] = new_segments_info
    return new_annotation


def main() -> None:
    args = parse_args()

    with open(args.panoptic_json, "r") as f:
        panoptic_data = json.load(f)

    category_isthing = load_category_isthing_map(panoptic_data)

    panoptic_root = Path(args.panoptic_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    new_annotations = []
    for annotation in tqdm(panoptic_data.get("annotations", []), desc="Splitting stuff masks"):
        new_annotations.append(
            process_annotation(annotation, category_isthing, panoptic_root, output_root)
        )

    output_json = dict(panoptic_data)
    output_json["annotations"] = new_annotations
    with open(args.output_json, "w") as f:
        json.dump(output_json, f)


if __name__ == "__main__":
    main()