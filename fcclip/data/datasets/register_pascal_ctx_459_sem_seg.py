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

PASCAL_CTX_459_CATEGORIES=openseg_classes.get_pascal_ctx_459_categories_with_prompt_eng()

PASCAL_CTX_459_COLORS = [k["color"] for k in PASCAL_CTX_459_CATEGORIES]

MetadataCatalog.get("openvocab_pascal_ctx459_sem_seg_train").set(
    stuff_colors=PASCAL_CTX_459_COLORS[:],
)

MetadataCatalog.get("openvocab_pascal_ctx459_sem_seg_val").set(
    stuff_colors=PASCAL_CTX_459_COLORS[:],
)

UNSEEN_PASCAL_CTX_459_CLASSES = [
    0, # accordion
    2, # air conditioner
    3, # antenna
    4, # artillery
    5, # ashtray
    6, # atrium
    7, # baby carriage
    8, # bag,bags
    9, # ball
    10, # balloon
    11, # bamboo weaving
    12, # barrel
    14, # basket
    15, # basketball backboard
    16, # bathtub
    18, # bedclothes
    19, # beer
    20, # bell
    23, # binoculars
    25, # bird cage
    26, # bird feeder
    27, # bird nest
    28, # blackboard
    29, # board
    31, # bone
    34, # bottle opener
    36, # box
    37, # bracelet
    38, # brick
    40, # broom
    41, # brush
    42, # bucket
    46, # cabinet door
    47, # cage
    49, # calculator
    50, # calendar
    51, # camel
    52, # camera
    53, # camera lens
    54, # can
    55, # candle
    56, # candle holder
    57, # cap
    59, # card
    60, # cart
    61, # computer case
    62, # casette recorder
    63, # cash register
    65, # cd
    66, # cd player
    69, # cello
    70, # chain
    72, # chessboard
    73, # chicken
    74, # chopstick
    75, # clip
    76, # clippers
    78, # closet
    80, # clothes tree
    81, # coffee
    82, # coffee machine
    83, # comb
    84, # computer
    85, # concrete
    86, # cone
    87, # container
    88, # control booth
    89, # controller
    90, # cooker
    91, # copying machine
    92, # coral
    93, # cork
    94, # corkscrew
    96, # court
    98, # crabstick
    99, # crane
    100, # crate
    101, # cross
    102, # crutch
    105, # cushion
    106, # cutting board
    107, # dais
    108, # disc
    109, # disc case
    110, # dishwasher
    111, # dock
    113, # dolphin
    115, # drainer
    116, # dray
    117, # drink dispenser
    118, # drinking machine
    119, # drop
    120, # drug
    121, # drum
    122, # drum kit
    123, # duck
    124, # dumbbell
    125, # earphone
    126, # earrings
    127, # egg
    128, # electric fan
    129, # electric iron
    130, # electric pot
    131, # electric saw
    132, # electronic keyboard
    133, # engine
    134, # envelope
    135, # equipment
    136, # escalator
    137, # exhibition booth
    138, # extinguisher
    139, # eyeglass
    140, # fan
    141, # faucet
    142, # fax machine
    144, # ferris wheel
    145, # fire extinguisher
    147, # fire place
    148, # fish
    149, # fish tank
    150, # fishbowl
    151, # fishing net
    152, # fishing pole
    153, # flag
    154, # flagstaff
    155, # flame
    156, # flashlight
    159, # fly
    160, # foam
    162, # footbridge
    163, # forceps
    165, # forklift
    166, # fountain
    167, # fox
    168, # frame
    170, # frog
    172, # funnel
    173, # furnace
    174, # game controller
    175, # game machine
    176, # gas cylinder
    177, # gas hood
    178, # gas stove
    179, # gift box
    180, # glass
    181, # glass marble
    182, # globe
    183, # glove
    184, # goal
    185, # grandstand
    187, # gravestone
    188, # ground,soil,soil ground,dirt ground
    189, # guardrail
    190, # guitar
    191, # gun
    192, # hammer
    193, # hand cart
    194, # handle
    195, # handrail
    196, # hanger
    197, # hard disk drive
    198, # hat
    199, # hay
    200, # headphone
    201, # heater
    202, # helicopter
    203, # helmet
    204, # holder
    205, # hook
    207, # horse-drawn carriage
    208, # hot-air balloon
    209, # hydrovalve
    210, # ice
    211, # inflator pump
    212, # ipod
    213, # iron
    214, # ironing board
    215, # jar
    216, # kart
    217, # kettle
    218, # key
    220, # kitchen range
    223, # knife block
    224, # ladder
    225, # ladder truck
    226, # ladle
    228, # leaves
    229, # lid
    230, # life buoy
    232, # light bulb
    233, # lighter
    234, # line
    235, # lion
    236, # lobster
    237, # lock
    238, # machine
    239, # mailbox
    240, # mannequin
    241, # map
    242, # mask
    243, # mat
    244, # match book
    245, # mattress
    246, # menu
    247, # metal
    248, # meter box
    249, # microphone
    252, # missile
    253, # model
    254, # money
    255, # monkey
    256, # mop
    260, # mouse pad
    261, # musical instrument
    262, # napkin
    263, # net
    264, # newspaper
    265, # oar
    266, # ornament
    267, # outlet
    269, # oxygen bottle
    270, # pack
    271, # pan
    273, # paper box
    274, # paper cutter
    275, # parachute
    276, # parasol
    277, # parterre
    278, # patio
    279, # pelage
    280, # pen
    281, # pen container
    282, # pencil
    284, # photo
    285, # piano
    286, # picture
    287, # pig
    288, # pillar
    290, # pipe
    291, # pitcher
    292, # plant
    293, # plastic
    296, # player
    297, # playground
    298, # pliers
    299, # plume
    300, # poker
    301, # poker chip
    302, # pole
    303, # pool table
    304, # postcard
    305, # poster
    306, # pot
    308, # printer
    309, # projector
    310, # pumpkin
    311, # rabbit
    312, # racket
    313, # radiator
    314, # radio
    315, # rail
    316, # rake
    317, # ramp
    318, # range hood
    319, # receiver
    320, # recorder
    321, # recreational machines
    324, # robot
    326, # rocket
    327, # rocking horse
    328, # rope
    330, # ruler
    331, # runway
    332, # saddle
    334, # saw
    335, # scale
    336, # scanner
    338, # scoop
    339, # screen
    340, # screwdriver
    341, # sculpture
    342, # scythe
    343, # sewer
    344, # sewing machine
    345, # shed
    347, # shell
    349, # shoe
    350, # shopping cart
    351, # shovel
    352, # sidecar
    353, # sidewalk
    354, # sign,signs
    355, # signal light
    358, # ski
    360, # sled
    361, # slippers
    362, # smoke
    363, # snail
    364, # snake
    366, # snowmobiles
    368, # spanner
    369, # spatula
    370, # speaker
    371, # speed bump
    372, # spice container
    374, # sprayer
    375, # squirrel
    376, # stage
    377, # stair
    378, # stapler
    379, # stick
    380, # sticky note
    381, # stone
    382, # stool
    383, # stove
    384, # straw
    385, # stretcher
    386, # sun
    387, # sunglass
    388, # sunshade
    389, # surveillance camera
    390, # swan
    391, # sweeper
    392, # swim ring
    393, # swimming pool
    394, # swing
    395, # switch
    397, # tableware
    398, # tank
    399, # tap
    400, # tape
    401, # tarp
    402, # telephone
    403, # telephone booth
    405, # tire
    408, # tong
    409, # tool
    412, # toy
    413, # toy car
    416, # trampoline
    417, # trash bin
    418, # tray
    420, # tricycle
    421, # tripod
    422, # trophy
    424, # tube
    425, # turtle
    427, # tweezers
    428, # typewriter
    430, # unknown
    431, # vacuum cleaner
    432, # vending machine
    433, # video camera
    434, # video game console
    435, # video player
    436, # video tape
    437, # violin
    438, # wakeboard
    440, # wallet
    441, # wardrobe
    442, # washing machine
    443, # watch
    445, # water dispenser
    446, # water pipe
    447, # water skate board
    448, # watermelon
    449, # whale
    450, # wharf
    451, # wheel
    452, # wheelchair
    454, # window blinds
    456, # wire
    457, # wood piece
    458, # wool
]

def _get_ctx459_meta():
    meta = {}
    # Id 0 is reserved for ignore_label, we change ignore_label for 0
    # to 255 in our pre-processing, so all ids are shifted by 1.
    stuff_ids = [k["id"] for k in PASCAL_CTX_459_CATEGORIES]
    assert len(stuff_ids) == 459, len(stuff_ids)

    stuff_classes = [k["name"] for k in PASCAL_CTX_459_CATEGORIES]

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

    max_dataset_id = max([ cat["id"] for cat in PASCAL_CTX_459_CATEGORIES ])
    dataset_id_to_seen_contigous_id = [ -1 for _ in range(max_dataset_id+1) ]

    for i, cat in enumerate(PASCAL_CTX_459_CATEGORIES):
        if cat["isthing"]:
            thing_dataset_id_to_contiguous_id[cat["id"]] = i
            if cat["id"] in UNSEEN_PASCAL_CTX_459_CLASSES:
                unseen_dataset_id_to_thing_contigous_id[cat["id"]] = last_thing_id
            else:
                seen_dataset_id_to_thing_contigous_id[cat["id"]] = last_thing_id
            last_thing_id += 1
        # else:
        #     stuff_dataset_id_to_contiguous_id[cat["id"]] = i
        
        # in order to use sem_seg evaluator
        stuff_dataset_id_to_contiguous_id[cat["id"]] = i

        if cat["id"] in UNSEEN_PASCAL_CTX_459_CLASSES:
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


def register_all_ctx459(root):
    root = os.path.join(root, "pascal_ctx_d2")
    meta = _get_ctx459_meta()
    for name, dirname in [("train", "training"), ("val", "validation")]:
        image_dir = os.path.join(root, "images", dirname)
        gt_dir = os.path.join(root, "annotations_ctx459", dirname)
        name = f"openvocab_pascal_ctx459_sem_seg_{name}"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="tif", image_ext="jpg")
        )
        MetadataCatalog.get(name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="zs_sem_seg",
            ignore_label=65535,  # NOTE: gt is saved in 16-bit TIFF images
            gt_ext="tif",
            **meta
        )

        
_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_ctx459(_root)