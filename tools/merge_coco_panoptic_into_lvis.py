import json
import os
import numpy as np
import cv2
from PIL import Image
from typing import List, Union, Dict
import argparse
import copy
from tqdm import tqdm
from panopticapi.utils import rgb2id

COCO_ID_TO_LVIS_ID = {
    1: 793,  # person
    2: 94,  # bicycle
    3: 207,  # car
    4: 703,  # motorcycle
    5: 3,  # airplane
    6: 173,  # bus
    7: 1115,  # train
    8: 1123,  # truck
    9: 118,  # boat
    10: 1112,  # traffic light
    11: 445,  # fire hydrant
    13: 1019,  # stop sign
    14: 766,  # parking meter
    15: 90,  # bench
    16: 99,  # bird
    17: 225,  # cat
    18: 378,  # dog
    19: 569,  # horse
    20: 943,  # sheep
    21: 80,  # cow
    22: 422,  # elephant
    23: 76,  # bear
    24: 1202,  # zebra
    25: 496,  # giraffe
    27: 34,  # backpack
    28: 1133,  # umbrella
    31: 35,  # handbag
    32: 716,  # tie
    33: 36,  # suitcase
    34: 474,  # frisbee
    35: 964,  # skis
    36: 976,  # snowboard
    37: 41,  # sports ball
    38: 611,  # kite
    39: 58,  # baseball bat
    40: 60,  # baseball glove
    41: 962,  # skateboard
    42: 1037,  # surfboard
    43: 1079,  # tennis racket
    44: 133,  # bottle
    46: 1190,  # wine glass
    47: 344,  # cup
    48: 469,  # fork
    49: 615,  # knife
    50: 1000,  # spoon
    51: 139,  # bowl
    52: 45,  # banana
    53: 12,  # apple
    54: 912,  # sandwich
    55: 735,  # orange
    56: 154,  # broccoli
    57: 217,  # carrot
    59: 816,  # pizza
    60: 387,  # donut
    61: 183,  # cake
    62: 232,  # chair
    63: 982,  # couch
    64: 837,  # potted plant
    65: 77,  # bed
    67: 367,  # dining table
    70: 1097,  # toilet
    72: 1077,  # tv
    73: 631,  # laptop
    74: 705,  # mouse
    75: 881,  # remote
    76: 296,  # keyboard
    77: 230,  # cell phone
    78: 687,  # microwave
    79: 739,  # oven
    80: 1095,  # toaster
    81: 961,  # sink
    82: 421,  # refrigerator
    84: 127,  # book
    85: 271,  # clock
    86: 1139,  # vase
    87: 923,  # scissors
    88: 1071,  # teddy bear
    89: 534,  # hair drier
    90: 1102,  # toothbrush
}

COCO_ID_TO_LVIS_DATA = {
    58: {
        'name': 'hot_dog',
        'isthing': 1,
        'synonyms': ['hot_dog'],
    },
    92: {
        'name': 'banner',
        'isthing': 0,
        'synonyms': ['banner', 'banners'],
    },
    93: {
        'name': 'blanket',
        'isthing': 0,
        'synonyms': ['blanket', 'blankets'],
    },
    95: {
        'name': 'bridge',
        'isthing': 0,
        'synonyms': ['bridge'],
    },
    100: {
        'name': 'cardboard',
        'isthing': 0,
        'synonyms': ['cardboard'],
    },
    107: {
        'name': 'counter',
        'isthing': 0,
        'synonyms': ['counter'],
    },
    109: {
        'name': 'curtain',
        'isthing': 0,
        'synonyms': ['curtain', 'curtains'],
    },
    112: {
        'name': 'door',
        'isthing': 0,
        'synonyms': ['door', 'doors'],
    },
    118: {
        'name': 'wood_floor',
        'isthing': 0,
        'synonyms': ['wood_floor'],
    },
    119: {
        'name': 'flower',
        'isthing': 0,
        'synonyms': ['flower', 'flowers'],
    },
    122: {
        'name': 'fruit',
        'isthing': 0,
        'synonyms': ['fruit', 'fruits'],
    },
    125: {
        'name': 'gravel',
        'isthing': 0,
        'synonyms': ['gravel'],
    },
    128: {
        'name': 'house',
        'isthing': 0,
        'synonyms': ['house'],
    },
    130: {
        'name': 'lamp',
        'isthing': 0,
        'synonyms': ['lamp', 'bulb', 'lamps', 'bulbs'],
    },
    133: {
        'name': 'mirror',
        'isthing': 0,
        'synonyms': ['mirror'],
    },
    138: {
        'name': 'tennis_net',
        'isthing': 0,
        'synonyms': ['tennis_net'],
    },
    141: {
        'name': 'pillow',
        'isthing': 0,
        'synonyms': ['pillow', 'pillows'],
    },
    144: {
        'name': 'platform',
        'isthing': 0,
        'synonyms': ['platform'],
    },
    145: {
        'name': 'playingfield',
        'isthing': 0,
        'synonyms': ['playingfield', 'tennis_court', 'baseball_field', 'soccer_field', 'tennis_field'],
    },
    147: {
        'name': 'railroad',
        'isthing': 0,
        'synonyms': ['railroad'],
    },
    148: {
        'name': 'river',
        'isthing': 0,
        'synonyms': ['river'],
    },
    149: {
        'name': 'road',
        'isthing': 0,
        'synonyms': ['road'],
    },
    151: {
        'name': 'roof',
        'isthing': 0,
        'synonyms': ['roof'],
    },
    154: {
        'name': 'sand',
        'isthing': 0,
        'synonyms': ['sand'],
    },
    155: {
        'name': 'sea',
        'isthing': 0,
        'synonyms': ['sea', 'sea_wave', 'wave', 'waves'],
    },
    156: {
        'name': 'shelf',
        'isthing': 0,
        'synonyms': ['shelf'],
    },
    159: {
        'name': 'snow',
        'isthing': 0,
        'synonyms': ['snow'],
    },
    161: {
        'name': 'stairs',
        'isthing': 0,
        'synonyms': ['stairs'],
    },
    166: {
        'name': 'tent',
        'isthing': 0,
        'synonyms': ['tent'],
    },
    168: {
        'name': 'towel',
        'isthing': 0,
        'synonyms': ['towel'],
    },
    171: {
        'name': 'brick_wall',
        'isthing': 0,
        'synonyms': ['brick_wall'],
    },
    175: {
        'name': 'stone_wall',
        'isthing': 0,
        'synonyms': ['stone_wall'],
    },
    176: {
        'name': 'tile_wall',
        'isthing': 0,
        'synonyms': ['tile_wall'],
    },
    177: {
        'name': 'wood_wall',
        'isthing': 0,
        'synonyms': ['wood_wall'],
    },
    178: {
        'name': 'water',
        'isthing': 0,
        'synonyms': ['water'],
    },
    180: {
        'name': 'window_blind',
        'isthing': 0,
        'synonyms': ['window_blind'],
    },
    181: {
        'name': 'window',
        'isthing': 0,
        'synonyms': ['window'],
    },
    184: {
        'name': 'tree',
        'isthing': 0,
        'synonyms': ['tree', 'trees', 'palm_tree', 'bushes'],
    },
    185: {
        'name': 'fence',
        'isthing': 0,
        'synonyms': ['fence', 'fences'],
    },
    186: {
        'name': 'ceiling',
        'isthing': 0,
        'synonyms': ['ceiling'],
    },
    187: {
        'name': 'sky',
        'isthing': 0,
        'synonyms': ['sky', 'clouds'],
    },
    188: {
        'name': 'cabinet',
        'isthing': 0,
        'synonyms': ['cabinet', 'cabinets'],
    },
    189: {
        'name': 'table',
        'isthing': 0,
        'synonyms': ['table'],
    },
    190: {
        'name': 'floor',
        'isthing': 0,
        'synonyms': ['floor', 'flooring', 'tile_floor'],
    },
    191: {
        'name': 'pavement',
        'isthing': 0,
        'synonyms': ['pavement'],
    },
    192: {
        'name': 'mountain',
        'isthing': 0,
        'synonyms': ['mountain', 'mountains'],
    },
    193: {
        'name': 'grass',
        'isthing': 0,
        'synonyms': ['grass'],
    },
    194: {
        'name': 'dirt',
        'isthing': 0,
        'synonyms': ['dirt'],
    },
    195: {
        'name': 'paper',
        'isthing': 0,
        'synonyms': ['paper'],
    },
    196: {
        'name': 'food',
        'isthing': 0,
        'synonyms': ['food'],
    },
    197: {
        'name': 'building',
        'isthing': 0,
        'synonyms': ['building', 'buildings'],
    },
    198: {
        'name': 'rock',
        'isthing': 0,
        'synonyms': ['rock'],
    },
    199: {
        'name': 'wall',
        'isthing': 0,
        'synonyms': ['wall', 'walls'],
    },
    200: {
        'name': 'rug',
        'isthing': 0,
        'synonyms': ['rug'],
    },
}


def iou_from_masks(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """
    Compute IoU between two binary masks. Accepts bool or 0/1 uint8 arrays.
    Returns IoU as float. If union is zero, returns 0.0.
    """
    a = (mask_a > 0)
    b = (mask_b > 0)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def _to_xy_array(poly: Union[List[float], np.ndarray]) -> np.ndarray:
    """
    Convert a polygon (flat list or Nx2 array) to an (N,2) int32 numpy array.
    """
    arr = np.asarray(poly)
    if arr.ndim == 1:
        if arr.size % 2 != 0:
            raise ValueError("Flat polygon list must have even length.")
        arr = arr.reshape(-1, 2)
    elif arr.ndim == 2 and arr.shape[1] == 2:
        pass
    else:
        raise ValueError("Unsupported polygon shape: %r" % (arr.shape,))
    return arr.astype(np.int32)


def _signed_area(pts: np.ndarray) -> float:
    """
    Signed area of polygon pts (N,2). Positive or negative shows winding.
    """
    x = pts[:, 0].astype(np.float64)
    y = pts[:, 1].astype(np.float64)
    return 0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))


def polygons_to_mask(polygons: Union[List[float], np.ndarray],
                     height: int,
                     width: int,
                     *,
                     hole_winding: str = "negative") -> np.ndarray:
    """
    Convert a list of polygons into a binary mask (H, W) with dtype uint8 (0/1).
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    if len(polygons) == 0:
        return mask

    fill_polys = []
    hole_polys = []

    for poly in polygons:
        pts = _to_xy_array(poly)
        if pts.shape[0] < 3:
            continue
        if hole_winding == "none":
            fill_polys.append(pts)
        else:
            area = _signed_area(pts)
            is_hole = (area < 0) if hole_winding == "negative" else (area > 0)
            if is_hole:
                hole_polys.append(pts)
            else:
                fill_polys.append(pts)

    if fill_polys:
        cv2.fillPoly(mask, [p.reshape(-1, 2) for p in fill_polys], color=1)

    if hole_polys:
        cv2.fillPoly(mask, [p.reshape(-1, 2) for p in hole_polys], color=0)

    return mask


def mask_to_polygons(mask, *, epsilon=2.0, min_area=10, simplify=True):
    """
    Convert a binary mask (H,W) -> list of polygons in COCO flat format.
    """
    mask_u8 = (mask > 0).astype(np.uint8)
    contours, hierarchy = cv2.findContours(mask_u8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    polygons = []
    if contours is None:
        return polygons

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        if simplify and epsilon is not None and epsilon > 0:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon if epsilon > 1 else 0.01 * peri, True)
        else:
            approx = cnt

        pts = approx.reshape(-1, 2)
        if pts.shape[0] < 3:
            continue
        flat = pts.astype(np.int32).reshape(-1).tolist()
        polygons.append(flat)

    return polygons


def merge_coco_lvis(coco_json_path: str, coco_panoptic_dir: str, lvis_json_path: str, output_json_path: str):
    """
    Merges COCO panoptic annotations into LVIS annotations, including categories not originally in LVIS,
    and computes the instance and image counts for these new categories.
    """
    print("Loading COCO annotations...")
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    print("Loading LVIS annotations...")
    with open(lvis_json_path, 'r') as f:
        lvis_data = json.load(f)

    print("Cloning LVIS data for modification...")
    output_data = copy.deepcopy(lvis_data)

    print("Updating LVIS categories with 'isthing' attribute...")
    for category in output_data['categories']:
        category['isthing'] = 1

    full_coco_to_lvis_map = copy.deepcopy(COCO_ID_TO_LVIS_ID)

    print("Adding new COCO-derived categories to LVIS...")
    max_lvis_cat_id = max(cat['id'] for cat in output_data['categories']) if output_data['categories'] else 0
    
    newly_added_cat_ids = set()

    for coco_cat_id, cat_data in COCO_ID_TO_LVIS_DATA.items():
        if coco_cat_id in full_coco_to_lvis_map:
            continue
        
        max_lvis_cat_id += 1
        new_category = {
            'id': max_lvis_cat_id,
            'name': cat_data['name'],
            'synonyms': cat_data['synonyms'],
            'isthing': cat_data['isthing'],
            'synset': 'placeholder.n.01', 
            'def': '',
            'frequency': 'c',
            'image_count': 0,
            'instance_count': 0
        }
        output_data['categories'].append(new_category)
        full_coco_to_lvis_map[coco_cat_id] = new_category['id']
        newly_added_cat_ids.add(new_category['id'])

    print(f"Added {len(newly_added_cat_ids)} COCO categories to LVIS.")

    # Data structures for computing counts for new categories
    new_cat_instance_counts = {cat_id: 0 for cat_id in newly_added_cat_ids}
    new_cat_image_sets = {cat_id: set() for cat_id in newly_added_cat_ids}

    lvis_image_map: Dict[str, Dict] = {f"{img['id']:012d}": img for img in lvis_data['images']}
    lvis_img_ann_map: Dict[int, List] = {img['id']: [] for img in lvis_data['images']}
    for ann in lvis_data['annotations']:
        lvis_img_ann_map[ann['image_id']].append(ann)

    new_ann_id = max((ann['id'] for ann in output_data['annotations']), default=0) + 1
    
    images_skipped = 0
    coco_anns_overlapped = 0
    coco_anns_empty = 0
    added_annotations = 0

    print("Processing COCO annotations to merge and count...")
    for coco_ann in tqdm(coco_data['annotations']):
        coco_panoptic_file_name = coco_ann['file_name']
        coco_img_base_name = os.path.splitext(coco_panoptic_file_name)[0]
        
        if coco_img_base_name not in lvis_image_map:
            images_skipped += 1
            continue

        lvis_image = lvis_image_map[coco_img_base_name]
        image_id = lvis_image['id']
        height, width = lvis_image['height'], lvis_image['width']

        panoptic_path = os.path.join(coco_panoptic_dir, coco_panoptic_file_name)
        panoptic = np.asarray(Image.open(panoptic_path), dtype=np.uint32)
        panoptic_mask_ids = rgb2id(panoptic)

        for segment_info in coco_ann['segments_info']:
            coco_cat_id = segment_info['category_id']
            if coco_cat_id not in full_coco_to_lvis_map:
                continue

            lvis_cat_id = full_coco_to_lvis_map[coco_cat_id]
            segment_id = segment_info['id']
            coco_mask = (panoptic_mask_ids == segment_id).astype(np.uint8)

            ignore_coco_mask = False
            if image_id in lvis_img_ann_map:
                for lvis_ann in lvis_img_ann_map[image_id]:
                    if lvis_ann['category_id'] == lvis_cat_id:
                        lvis_mask = polygons_to_mask(lvis_ann['segmentation'], height, width)
                        if iou_from_masks(coco_mask, lvis_mask) > 0.5:
                            ignore_coco_mask = True
                            break
            if ignore_coco_mask:
                coco_anns_overlapped += 1
                continue

            polygons = mask_to_polygons(coco_mask)
            if not polygons:
                coco_anns_empty += 1
                continue

            x, y, w, h = cv2.boundingRect(coco_mask)
            area = int(np.sum(coco_mask))

            new_ann = {
                'bbox': [int(x), int(y), int(w), int(h)],
                'category_id': lvis_cat_id,
                'image_id': image_id,
                'id': new_ann_id,
                'segmentation': polygons,
                'area': area
            }
            output_data['annotations'].append(new_ann)
            new_ann_id += 1
            added_annotations += 1

            # Update counts if the category is one of the new ones
            if lvis_cat_id in newly_added_cat_ids:
                new_cat_instance_counts[lvis_cat_id] += 1
                new_cat_image_sets[lvis_cat_id].add(image_id)

    print(f"Skipped {images_skipped} COCO images not found in LVIS.")
    print(f"Added {added_annotations} new annotations to LVIS.")
    print(f"Ignored {coco_anns_overlapped} COCO annotations due to overlap.")
    print(f"Ignored {coco_anns_empty} COCO annotations that were empty.")

    # Finalize the computed counts in the categories list
    print("Finalizing instance and image counts for new categories...")
    cat_id_to_cat_map = {cat['id']: cat for cat in output_data['categories']}
    for cat_id in newly_added_cat_ids:
        if cat_id in cat_id_to_cat_map:
            category_object = cat_id_to_cat_map[cat_id]
            category_object['instance_count'] = new_cat_instance_counts[cat_id]
            category_object['image_count'] = len(new_cat_image_sets[cat_id])

    print("Writing the merged annotations to the output file...")
    with open(output_json_path, 'w') as f:
        json.dump(output_data, f)
    print("Merging process completed successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Merge COCO panoptic annotations into LVIS annotations.")
    parser.add_argument('coco_json', type=str, help="Path to the COCO panoptic annotations JSON file.")
    parser.add_argument('coco_panoptic_dir', type=str, help="Path to the directory containing COCO panoptic PNG masks.")
    parser.add_argument('lvis_json', type=str, help="Path to the LVIS annotations JSON file.")
    parser.add_argument('output_json', type=str, help="Path to the output merged annotations JSON file.")
    args = parser.parse_args()

    merge_coco_lvis(args.coco_json, args.coco_panoptic_dir, args.lvis_json, args.output_json)
