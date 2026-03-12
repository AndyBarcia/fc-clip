"""
Prepare ADE20K-Full superclass-merged semantic labels with compact class IDs.

This script expects `prepare_ade20k_full_sem_seg.py` has already produced
`annotations_detectron2/{training,validation}`.
"""

import os
from pathlib import Path

import numpy as np
import tqdm
from PIL import Image


IGNORE_LABEL = 65535

SUPERCLASSES_ADE20K_847_CATEGORY_IDS = {
    1: { # Building, buildings, edifice. Superclass of:
        26,  # house exterior
        59,  # skyscraper, skyscrapers
        418, # temple
        101, # hovel, hut, shack, shanty
        317, # shanty
        256, # shanties
        262, # cabin
        139, # tent, collapsible shelter
        339, # shop, store
        399, # shops
        106, # booth, cubicle, stall, kiosk
        205, # telephone booth
        361, # greenhouse
        266, # henhouse
        522, # pergola
        116, # gazebo
        269, # windmill
    },
    2: { # Sky, clouds. Contains/Superclass of:
        720, # sun
        738, # star
        395, # smoke
    },
    9: { # Earth, ground. Superclass of:
        54,  # sand
        213, # sand dune
        142, # dirt track
        338, # sand trap
    },
    16: { # Plant, flora, plant life, bushes. Superclass of:
        3,   # tree, trees
        63,  # palm tree, palm trees
        12,  # grass, grass field
        417, # weeds
        73,  # flower, flowers
        358, # alga, algae
        496, # leaf, leafage, foliage
        387, # leaves
        651, # root
        568, # branch
    },
    24: { # Water. Superclass of:
        22,  # sea, ocean
        35,  # river
        546, # stream
        134, # lake
        111, # pond
        473, # puddle
        136, # ice, water ice
        536, # ice floe, floe
        304, # iceberg
        218, # swimming pool, swimming bath
        86,  # fountain
        71,  # waterfall, falls
    },
    33: { # Rock, stone, rocks, stones. Superclass of:
        445, # rocky formation
        53,  # cliff, drop, drop-off
        383, # rubble
    },
    34: { # Seat, seats. Superclass of:
        18,  # chair, chairs
        29,  # armchair, armchairs
        21,  # sofa, couch, sofas, couches
        404, # sofa bed
        83,  # bench, benches
        133, # stool, stools
        126, # ottoman, pouf, hassock
        81,  # swivel chair
        319, # rocking chair, rocker
        719, # chaise lounge
        297, # deck chair, beach chair
        330, # hairdresser chair
        616, # dental chair
        551, # chair desk
    },
    82: { # Light, light source. Superclass of:
        32,  # lamp, lamps
        105, # chandelier, pendant
        110, # sconce, sconce lamp
        206, # spotlight, spot
        686, # spotlights
        756, # floor spotlight
        78,  # streetlight, street lamp
        478, # light bulb, bulb, bulbs
        763, # bulbs
        219, # fluorescent, fluorescent fixture
        469, # candle, taper
        824, # night light
    },
    102: { # Apparel, wearing apparel, dress, clothes. Superclass of:
        202, # jacket
        484, # coat
        350, # shirt
        314, # jersey, t-shirt
        517, # sweater, jumper
        656, # sweatshirt
        728, # sweatshirts
        283, # trouser, pant
        481, # skirt
        315, # shoe
        442, # gym shoe, sneaker
        516, # boot
        750, # flip flop
        249, # hat, chapeau
        386, # cap
        590, # helmet
        384, # scarf
        748, # scarfs
        801, # belt
        834, # tie
        684, # shawl
        658, # mitten
    },
    104: { # Animal, animate being. Superclass of:
        500, # cat
        259, # dog, domestic dog
        165, # horse
        320, # bird
        569, # duck
        508, # hen, biddy
        301, # fish
        838, # sea star
    },
    118: { # Land, soil. Superclass of:
        31,  # field
        90,  # hill
        15,  # mountain, mount
        435, # valley, vale
        494, # canyon
        368, # island
        624, # shore
    },
    152: { # Food, solid food. Superclass of:
        234, # fruit
        375, # bread
        705, # breads
        683, # bread rolls
        419, # patty, cake
        311, # candy, confect
        609, # chips
        787, # potatoes
        781, # onions
        533, # pumpkin
    },
    160: { # Container. Superclass of:
        61,  # box, boxes
        127, # ashcan, trash can, garbage can, wastebin
        294, # recycling bin
        199, # bucket, pail
        682, # buckets
        137, # basket, handbasket
        678, # baskets
        142, # barrel, cask, barrels
        385, # barrels
        135, # tank, storage tank
        356, # tanks
        114, # bag, bags, gift bag, paper bag
        252, # bag, handbag, pocketbook, purse
        191, # traveling bag, suitcase
        239, # backpack, knapsack
        128, # bottle, bottles
        240, # jar
        823, # spice jar
        355, # can, tin, tin can
        745, # tins
        426, # canister
    },
    173: { # Machine. Superclass of:
        464, # tractor
        493, # forklift
        397, # bulldozer
        456, # excavator
        290, # steam shovel
        560, # mechanical shovel
        145, # dishwasher
        149, # washer, automatic washer, washing machine
        510, # washing machines
        440, # tumble dryer
        652, # sewing machine
        236, # vending machine
        345, # slot machine, coin machine
        120, # arcade machine, arcade machines
        460, # scale, weighing machine
        846, # adding machine
    },
    232: { # Rubbish, trash, scrap. Superclass of:
        544, # junk
        246, # landfill
        657, # recycling materials
    },
    357: { # Fabric, cloth, material, textile. Superclass of (or material of):
        430, # canvas
        446, # plastic
        377, # wood
        178, # paper
        471, # cardboard
    },
    696: { # Cutlery (Also semantically linked to Utensils). Superclass of:
        437, # knife
        625, # fork
        638, # spoon
        815, # ladle
        798, # tongs
        827, # slotted spoon
        708, # skimmer
    },
    736: { # Tools. Superclass of:
        833, # hammer
        810, # spanners
        504, # shovel
        722, # rake
        775, # scissors
    },
    773: { # Utensils. Superclass of:
        696, # cutlery
        437, # knife
        625, # fork
        638, # spoon
        815, # ladle
        798, # tongs
        827, # slotted spoon
        708, # skimmer
    }
}



EQUIVALENT_ADE20K_847_CLASS_GROUPS = [
    (8, 377),
    (18, 341, 589),
    (44, 78, 107),
    (47, 173),
    (50, 314),
    (102, 257, 318),
    (124, 280),
    (131, 189, 701),
    (136, 357),
    (138, 679),
    (143, 386),
    (150, 511),
    (175, 256),
    (186, 689),
    (191, 307),
    (193, 813),
    (199, 209, 769),
    (200, 683),
    (205, 621),
    (215, 413),
    (230, 664),
    (236, 281),
    (242, 480),
    (293, 612),
    (317, 715, 752),
    (327, 345),
    (338, 635, 650),
    (340, 400),
    (349, 528),
    (356, 427, 746),
    (376, 706),
    (385, 749),
    (394, 605),
    (397, 643),
    (401, 478),
    (407, 678),
    (414, 823),
    (424, 699),
    (432, 644),
    (456, 598),
    (463, 826),
    (479, 764),
    (484, 719),
    (504, 704),
    (513, 794),
    (575, 747),
    (597, 669),
    (603, 646),
    (622, 698),
    (657, 729),
    (667, 814),
    (745, 840),
    (803, 833),
]

def _build_old_to_new_train_id_mapping():
    superclass_ids = set(SUPERCLASSES_ADE20K_847_CATEGORY_IDS.keys())
    subclass_ids = set()
    for children in SUPERCLASSES_ADE20K_847_CATEGORY_IDS.values():
        subclass_ids.update(children)

    kept_old_train_ids = [
        train_id
        for train_id in range(847)
        if train_id in superclass_ids or train_id not in subclass_ids
    ]
    kept_old_train_ids_set = set(kept_old_train_ids)

    old_to_superclass_old_id = {old_id: old_id for old_id in range(847)}
    for superclass_id, child_ids in SUPERCLASSES_ADE20K_847_CATEGORY_IDS.items():
        for child_id in child_ids:
            old_to_superclass_old_id[child_id] = superclass_id

    parent = {old_id: old_id for old_id in kept_old_train_ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if ra < rb:
            parent[rb] = ra
        else:
            parent[ra] = rb

    for group in EQUIVALENT_ADE20K_847_CLASS_GROUPS:
        mapped = sorted({class_id - 1 for class_id in group if (class_id - 1) in kept_old_train_ids_set})
        if len(mapped) < 2:
            continue
        base = mapped[0]
        for class_id in mapped[1:]:
            union(base, class_id)

    final_old_train_ids = sorted({find(old_id) for old_id in kept_old_train_ids})
    old_to_final_root = {
        old_id: find(old_to_superclass_old_id[old_id])
        for old_id in range(847)
    }
    final_root_to_new_train_id = {
        root_old_id: new_train_id for new_train_id, root_old_id in enumerate(final_old_train_ids)
    }

    return final_root_to_new_train_id, old_to_final_root


if __name__ == "__main__":
    dataset_dir = Path(os.getenv("DETECTRON2_DATASETS", "datasets"))
    ade_root = dataset_dir / "ade" / "ADE20K_2021_17_01"

    final_root_to_new_train_id, old_to_final_root = _build_old_to_new_train_id_mapping()

    for split in ["training", "validation"]:
        src_dir = ade_root / "annotations_detectron2" / split
        dst_dir = ade_root / "annotations_detectron2_superclass_merged_equiv_623" / split
        dst_dir.mkdir(parents=True, exist_ok=True)

        for src_path in tqdm.tqdm(sorted(src_dir.glob("*.tif")), desc=f"Processing {split}"):
            mask = np.array(Image.open(src_path), dtype=np.uint16)

            merged = np.full(mask.shape, IGNORE_LABEL, dtype=np.uint16)

            unique_ids = np.unique(mask)
            for old_train_id in unique_ids:
                if old_train_id == IGNORE_LABEL:
                    continue
                final_root = old_to_final_root[int(old_train_id)]
                new_train_id = final_root_to_new_train_id[final_root]
                merged[mask == old_train_id] = new_train_id

            Image.fromarray(merged).save(dst_dir / src_path.name)
