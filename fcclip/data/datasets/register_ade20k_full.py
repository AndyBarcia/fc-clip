"""
This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 

Reference: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/data/datasets/register_ade20k_full.py
"""

import os

import numpy as np

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

from . import openseg_classes

ADE20K_847_CATEGORIES = openseg_classes.get_ade20k_847_categories_with_prompt_eng()

ADE20k_847_COLORS = [np.random.randint(256, size=3).tolist() for k in ADE20K_847_CATEGORIES]

MetadataCatalog.get("openvocab_ade20k_full_sem_seg_train").set(
    stuff_colors=ADE20k_847_COLORS[:],
)

MetadataCatalog.get("openvocab_ade20k_full_sem_seg_val").set(
    stuff_colors=ADE20k_847_COLORS[:],
)

UNSEEN_ADE20K_847_CATEGORY_IDS = [
    9, # earth,ground
    23, # painting,picture,paintings,pictures,wallart,framed canvas
    26, # house exterior
    29, # armchair,armchairs
    31, # field
    34, # seat,seats
    36, # desk,desks
    37, # bathtub,bathing tub,bath,tub
    38, # railing,rail
    39, # signboard,sign,signboards,signs
    40, # cushion,cushions
    41, # path
    42, # work surface
    44, # column,pillar
    46, # wardrobe,closet,press,wardrobes,closets
    49, # pedestal
    51, # blind
    52, # runway
    53, # cliff,drop,drop-off
    55, # fireplace,hearth,open fireplace
    57, # screen door,shower door
    59, # skyscraper,skyscrapers
    60, # grandstand,covered stand
    61, # box,boxes
    62, # pool table,billiard table,snooker table
    64, # double door
    65, # coffee table,cocktail table
    67, # countertop,counter top,worktop
    68, # chest of drawers,chest,bureau,dresser
    69, # kitchen island
    71, # waterfall,falls
    72, # stove,kitchen stove,kitchen range,kitchen range,cooking stove
    74, # bookcase
    75, # controls
    77, # stairway,staircase
    78, # streetlight,street lamp
    79, # computer,computing machine,computing device,data processor,electronic computer,information processing system
    81, # swivel chair
    82, # light,light source
    84, # case,display case,showcase,vitrine
    86, # fountain
    87, # embankment
    89, # van
    90, # hill
    91, # awning,sunshade,sunblind
    92, # poster,posting,placard,notice,bill,card
    95, # pole
    96, # tower,towers
    97, # court
    98, # ball,balls
    99, # aircraft carrier,carrier,flattop,attack aircraft carrier
    100, # buffet,sideboard
    101, # hovel,hut,hutch,shack,shanty
    103, # minibike,motorbike
    105, # chandelier,pendant,pendent
    106, # step,stair
    107, # booth,cubicle,stall,kiosk
    109, # doorframe,doorcase
    110, # sconce,sconce lamp,sconce light
    111, # pond
    112, # trade name,brand name
    113, # bannister,banister,balustrade,balusters,handrail
    114, # bag,bags,gift bag,paper bag
    116, # gazebo
    117, # escalator,moving staircase,moving stairway
    118, # land,soil
    119, # board,plank
    120, # arcade machine,arcade machines
    121, # eiderdown,duvet,continental quilt
    122, # bar
    123, # stall,stand,sales booth
    124, # playground
    125, # ship
    126, # ottoman,pouf,pouffe,puff,hassock
    127, # ashcan,trash can,garbage can,wastebin,ash bin,ash-bin,ashbin,dustbin,trash barrel,trash bin
    129, # cradle
    130, # pot,flowerpot
    131, # conveyer belt,conveyor belt,conveyer,conveyor,transporter
    133, # stool,stools
    134, # lake
    135, # tank,storage tank
    136, # ice,water ice
    137, # basket,handbasket
    138, # manhole
    140, # canopy
    142, # barrel,cask,barrels,casks
    143, # dirt track
    144, # beam
    145, # dishwasher,dish washer,dishwashing machine
    147, # crt screen
    148, # ruins
    149, # washer,automatic washer,washing machine
    151, # plaything,toy,toys
    153, # projection screen
    155, # stage
    156, # beacon,lighthouse,beacon light,pharos
    158, # sculpture,sculptures
    159, # aqueduct
    160, # container
    161, # scaffolding,staging
    162, # exhaust hood
    163, # curb,curbing,kerb
    164, # roller coaster
    166, # catwalk
    167, # glass,drinking glass
    169, # central reservation
    170, # carousel
    171, # radiator
    172, # closet
    173, # machine
    174, # pier,wharf,wharfage,dock
    175, # ceiling fan,floor fan
    176, # inflatable bounce game
    177, # pitch
    179, # arcade,colonnade
    180, # hot tub
    181, # helicopter
    182, # tray,trays
    183, # partition,divider
    184, # vineyard
    186, # bullring
    187, # flag,flags
    188, # pot
    189, # footbridge,overcrossing,pedestrian bridge
    190, # shower
    192, # bulletin board,notice board
    193, # confessional booth
    194, # trunk,tree trunk,bole
    195, # forest
    196, # elevator door
    198, # instrument panel
    199, # bucket,pail
    200, # tapestry,tapis
    202, # jacket
    203, # gate
    204, # monitor,monitoring device,monitors
    205, # telephone booth,phone booth,call box,telephone box,telephone kiosk
    206, # spotlight,spot
    207, # ring
    208, # control panel
    209, # blackboard,chalkboard
    210, # air conditioner,air conditioning
    211, # chest
    213, # sand dune
    214, # pipe,pipage,piping
    215, # vault
    216, # table football
    217, # cannon
    218, # swimming pool,swimming bath
    219, # fluorescent,fluorescent fixture
    220, # statue
    221, # loudspeaker,speaker,speaker unit,loudspeaker system,speaker system
    222, # exhibitor
    223, # ladder
    224, # carport
    225, # dam
    226, # pulpit
    227, # skylight,fanlight
    228, # water tower
    229, # grill,grille,grillwork
    230, # display board
    231, # pane,pane of glass,window glass
    232, # rubbish,trash,scrap
    233, # ice rink
    235, # patio
    236, # vending machine
    237, # telephone,phone,telephone set
    238, # net
    240, # jar
    241, # track
    242, # magazine
    243, # shutter
    246, # landfill
    247, # post
    248, # altarpiece,reredos
    249, # hat,chapeau,lid
    250, # arch,archway
    251, # table game
    253, # document,written document,papers
    254, # dome
    255, # pier
    256, # shanties
    257, # forecourt
    258, # crane
    260, # piano,pianoforte,forte-piano
    261, # drawing
    262, # cabin
    263, # ad,advertisement,advertizement,advertising,advertizing,advert
    264, # amphitheater,amphitheatre,coliseum
    265, # monument
    266, # henhouse
    267, # cockpit
    268, # heater,warmer
    269, # windmill,aerogenerator,wind generator
    270, # pool
    271, # elevator,lift
    272, # decoration,ornament,ornamentation
    273, # labyrinth
    274, # text,textual matter
    275, # printer
    276, # mezzanine,first balcony
    277, # mattress
    278, # straw
    279, # stalls
    280, # patio,terrace
    281, # billboard,hoarding
    282, # bus stop
    283, # trouser,pant
    284, # console table,console
    285, # rack
    286, # notebook
    287, # shrine
    288, # pantry
    289, # cart
    290, # steam shovel
    291, # porch
    292, # postbox,mailbox,letter box
    293, # figurine,statuette
    294, # recycling bin
    295, # folding screen
    296, # telescope
    297, # deck chair,beach chair
    298, # kennel
    299, # coffee maker
    300, # altar,communion table,lord's table
    301, # fish
    302, # easel
    303, # artificial golf green
    304, # iceberg
    305, # candlestick,candle holder
    306, # shower stall,shower bath
    307, # television stand
    308, # wall socket,wall plug,electric outlet,electrical outlet,outlet,electric receptacle
    309, # skeleton
    310, # grand piano,grand
    311, # candy,confect
    312, # grille door
    313, # pedestal,plinth,footstall
    314, # jersey,t-shirt,tee shirt
    315, # shoe
    316, # gravestone,headstone,tombstone
    317, # shanty
    318, # structure
    319, # rocking chair,rocker
    321, # place mat
    322, # tomb
    323, # big top
    324, # gas pump,gasoline pump,petrol pump,island dispenser
    325, # lockers
    326, # cage
    327, # finger
    328, # bleachers
    329, # ferris wheel
    330, # hairdresser chair
    331, # mat
    332, # stands
    333, # aquarium,fish tank,marine museum
    334, # streetcar,tram,tramcar,trolley,trolley car
    335, # napkin,table napkin,serviette
    336, # dummy
    337, # booklet,brochure,folder,leaflet,pamphlet
    338, # sand trap
    339, # shop,store
    340, # table cloth
    341, # service station
    342, # coffin
    343, # drawer
    344, # cages
    345, # slot machine,coin machine
    346, # balcony
    347, # volleyball court
    348, # table tennis
    349, # control table
    350, # shirt
    351, # merchandise,ware,product
    352, # railway
    353, # parterre
    354, # chimney
    355, # can,tin,tin can
    356, # tanks
    357, # fabric,cloth,material,textile
    358, # alga,algae
    359, # system
    360, # map
    361, # greenhouse
    362, # mug
    363, # barbecue
    364, # trailer
    365, # toilet tissue,toilet paper,bathroom tissue
    366, # organ
    367, # dishrag,dishcloth
    368, # island
    370, # trench
    371, # basket,basketball hoop,hoop
    372, # steering wheel,wheel
    373, # pitcher,ewer
    374, # goal
    375, # bread,breadstuff,staff of life
    377, # wood
    378, # file cabinet
    380, # motorboat
    381, # rope
    382, # guitar
    383, # rubble
    384, # scarf
    385, # barrels
    386, # cap
    387, # leaves
    388, # control tower
    389, # dashboard
    390, # bandstand
    391, # lectern
    392, # switch,electric switch,electrical switch
    393, # baseboard,mopboard,skirting board
    394, # shower room
    395, # smoke
    396, # faucet,spigot
    397, # bulldozer
    398, # saucepan
    399, # shops
    400, # meter
    401, # crevasse
    402, # gear
    403, # candelabrum,candelabra
    404, # sofa bed
    405, # tunnel
    406, # pallet
    407, # wire,conducting wire
    408, # kettle,boiler
    409, # bidet
    410, # baby buggy,baby carriage,carriage,perambulator,pram,stroller,go-cart,pushchair,pusher
    411, # music stand
    412, # pipe,tube
    415, # ice hockey rink
    416, # shelter
    417, # weeds
    418, # temple
    420, # ski slope
    421, # panel
    422, # wallet
    423, # wheel
    424, # towel rack,towel horse
    425, # roundabout
    426, # canister,cannister,tin
    427, # rod
    428, # soap dispenser
    429, # bell
    430, # canvas
    431, # box office,ticket office,ticket booth
    432, # teacup
    433, # trellis
    434, # workbench
    435, # valley,vale
    438, # podium
    439, # ramp
    440, # tumble dryer
    442, # gym shoe,sneaker,tennis shoe
    443, # lab bench
    444, # equipment
    445, # rocky formation
    446, # plastic
    447, # calendar
    448, # caravan
    449, # check-in-desk
    450, # ticket counter
    451, # brush
    452, # mill
    453, # covered bridge
    454, # bowling alley
    455, # hanger
    456, # excavator
    457, # trestle
    458, # revolving door
    459, # blast furnace
    460, # scale,weighing machine
    461, # projector
    462, # soap
    463, # locker
    464, # tractor
    465, # stretcher
    466, # frame
    467, # grating
    468, # alembic
    469, # candle,taper,wax light
    470, # barrier
    472, # cave
    473, # puddle
    474, # tarp
    475, # price tag
    476, # watchtower
    477, # meters
    479, # tracks
    480, # hair dryer
    481, # skirt
    482, # viaduct
    483, # paper towel
    484, # coat
    485, # sheet
    486, # fire extinguisher,extinguisher,asphyxiator
    487, # water wheel
    488, # pottery,clayware
    489, # magazine rack
    490, # teapot
    491, # microphone,mike
    492, # support
    493, # forklift
    494, # canyon
    495, # cash register,register
    496, # leaf,leafage,foliage
    498, # soap dish
    499, # windshield,windscreen
    501, # cue,cue stick,pool cue,pool stick
    502, # vent,venthole,vent-hole,blowhole
    503, # videos
    504, # shovel
    505, # eaves
    506, # antenna,aerial,transmitting aerial
    507, # shipyard
    508, # hen,biddy
    509, # traffic cone
    510, # washing machines
    511, # truck crane
    512, # cds
    513, # niche
    514, # scoreboard
    515, # briefcase
    516, # boot
    517, # sweater,jumper
    518, # hay
    519, # pack
    520, # bottle rack
    521, # glacier
    522, # pergola
    523, # building materials
    524, # television camera
    525, # first floor
    526, # rifle
    527, # tennis table
    528, # stadium
    529, # safety belt
    530, # cover
    531, # dish rack
    532, # synthesizer
    533, # pumpkin
    534, # gutter
    535, # fruit stand
    536, # ice floe,floe
    537, # handle,grip,handgrip,hold
    538, # wheelchair
    539, # mousepad,mouse mat
    540, # diploma
    541, # fairground ride
    542, # radio
    543, # hotplate
    544, # junk
    545, # wheelbarrow
    546, # stream
    547, # toll plaza
    548, # punching bag
    549, # trough
    550, # throne
    551, # chair desk
    552, # weighbridge
    553, # extractor fan
    554, # hanging clothes
    555, # dish,dish aerial,dish antenna,saucer
    556, # alarm clock,alarm
    557, # ski lift
    558, # chain
    559, # garage
    560, # mechanical shovel
    561, # wine rack
    562, # tramway
    563, # treadmill
    564, # menu
    565, # block
    566, # well
    567, # witness stand
    568, # branch
    569, # duck
    570, # casserole
    571, # frying pan
    572, # desk organizer
    573, # mast
    574, # spectacles,specs,eyeglasses,glasses
    575, # service elevator
    576, # dollhouse
    577, # hammock
    578, # clothes hanging
    579, # photocopier
    580, # notepad
    581, # golf cart
    582, # footpath
    583, # cross
    584, # baptismal font
    585, # boiler
    586, # skip
    587, # rotisserie
    588, # tables
    589, # water mill
    590, # helmet
    591, # cover curtain
    592, # brick
    593, # table runner
    594, # ashtray
    595, # street box
    596, # stick
    597, # hangers
    598, # cells
    599, # urinal
    600, # centerpiece
    601, # portable fridge
    602, # dvds
    603, # golf club
    604, # skirting board
    605, # water cooler
    606, # clipboard
    607, # camera,photographic camera
    608, # pigeonhole
    609, # chips
    610, # food processor
    611, # post box
    612, # lid
    613, # drum
    614, # blender
    615, # cave entrance
    616, # dental chair
    617, # obelisk
    618, # canoe
    620, # monitors
    621, # pool ball
    622, # cue rack
    623, # baggage carts
    624, # shore
    626, # paper filer
    627, # bicycle rack
    628, # coat rack
    629, # garland
    630, # sports bag
    631, # fish tank
    632, # towel dispenser
    633, # carriage
    634, # brochure
    635, # plaque
    636, # stringer
    637, # iron
    639, # flag pole
    640, # toilet brush
    641, # book stand
    642, # water faucet,water tap,tap,hydrant
    643, # ticket office
    644, # broom
    645, # dvd
    646, # ice bucket
    647, # carapace,shell,cuticle,shield
    648, # tureen
    649, # folders
    650, # chess
    651, # root
    652, # sewing machine
    653, # model
    654, # pen
    655, # violin
    656, # sweatshirt
    657, # recycling materials
    658, # mitten
    659, # chopping board,cutting board
    660, # mask
    661, # log
    663, # grill
    664, # hole
    665, # target
    666, # trash bag
    667, # chalk
    668, # sticks
    669, # balloon
    670, # score
    671, # hair spray
    672, # roll
    673, # runner
    674, # engine
    675, # inflatable glove
    676, # games
    677, # pallets
    678, # baskets
    679, # coop
    680, # dvd player
    681, # rocking horse
    682, # buckets
    683, # bread rolls
    684, # shawl
    685, # watering can
    686, # spotlights
    687, # post-it
    689, # security camera
    690, # runner cloth
    691, # lock
    692, # alarm,warning device,alarm system
    693, # side
    694, # roulette
    695, # bone
    696, # cutlery
    697, # pool balls
    698, # wheels
    699, # spice rack
    701, # towel ring
    702, # bread box
    703, # video
    704, # funfair
    705, # breads
    706, # tripod
    707, # ironing board
    708, # skimmer
    709, # hollow
    710, # scratching post
    711, # tricycle
    712, # file box
    713, # mountain pass
    714, # tombstones
    715, # cooker
    716, # card game,cards
    717, # golf bag
    718, # towel paper
    719, # chaise lounge
    720, # sun
    721, # toilet paper holder
    722, # rake
    723, # key
    724, # umbrella stand
    725, # dartboard
    726, # transformer
    727, # fireplace utensils
    728, # sweatshirts
    729, # cellular telephone,cellular phone,cellphone,cell,mobile phone
    730, # tallboy
    731, # stapler
    732, # sauna
    733, # test tube
    734, # palette
    735, # shopping carts
    736, # tools
    737, # push button,push,button
    738, # star
    739, # roof rack
    740, # barbed wire
    741, # spray
    742, # ear
    743, # sponge
    744, # racket
    745, # tins
    746, # eyeglasses
    747, # file
    748, # scarfs
    749, # sugar bowl
    750, # flip flop
    751, # headstones
    752, # laptop bag
    753, # leash
    754, # climbing frame
    755, # suit hanger
    756, # floor spotlight
    757, # plate rack
    758, # sewer
    759, # hard drive
    760, # sprinkler
    761, # tools box
    762, # necklace
    764, # steel industry
    765, # club
    766, # jack
    767, # door bars
    768, # control panel,instrument panel,control board,board,panel
    769, # hairbrush
    770, # napkin holder
    771, # office
    772, # smoke detector
    773, # utensils
    774, # apron
    776, # terminal
    777, # grinder
    778, # entry phone
    779, # newspaper stand
    780, # pepper shaker
    781, # onions
    782, # central processing unit,cpu,central processor,processor,mainframe
    783, # tape
    784, # bat
    785, # coaster
    786, # calculator
    787, # potatoes
    788, # luggage rack
    789, # salt
    790, # street number
    791, # viewpoint
    792, # sword
    793, # cd
    794, # rowing machine
    795, # plug
    797, # pepper
    798, # tongs
    799, # bonfire
    800, # dog dish
    801, # belt
    802, # dumbbells
    803, # videocassette recorder,vcr
    804, # hook
    805, # envelopes
    806, # shower faucet
    807, # watch
    808, # padlock
    809, # swimming pool ladder
    810, # spanners
    811, # gravy boat
    812, # notice board
    813, # trash bags
    814, # fire alarm
    815, # ladle
    816, # stethoscope
    817, # rocket
    818, # funnel
    819, # bowling pins
    820, # valve
    821, # thermometer
    823, # spice jar
    824, # night light
    825, # soaps
    826, # games table
    827, # slotted spoon
    828, # reel
    829, # scourer
    830, # sleeping robe
    831, # desk mat
    832, # dumbbell
    833, # hammer
    835, # typewriter
    836, # shaker
    837, # cheese dish
    838, # sea star
    839, # racquet
    840, # butane gas cylinder
    841, # paper weight
    842, # shaving brush
    843, # sunglasses
    844, # gear shift
    845, # towel rail
    846, # adding machine,totalizer,totaliser
]

def get_metadata():
    meta = {}

    # We only need class names
    stuff_classes = [k["name"] for k in ADE20K_847_CATEGORIES]
    assert len(stuff_classes) == 847, len(stuff_classes)

    stuff_dataset_id_to_contiguous_id = {}
    seen_dataset_id_to_contiguous_id = {}
    seen_dataset_id_to_seen_contiguous_id = {}
    unseen_dataset_id_to_contiguous_id = {}
    contiguous_id_to_seen_contiguous_id = []
    last_seen_id = 0

    max_dataset_id = max([ cat["id"] for cat in ADE20K_847_CATEGORIES ])
    dataset_id_to_seen_contigous_id = [ -1 for _ in range(max_dataset_id+1) ]

    for i, cat in enumerate(ADE20K_847_CATEGORIES):
        stuff_dataset_id_to_contiguous_id[cat["trainId"]] = i

        if cat["id"] in UNSEEN_ADE20K_847_CATEGORY_IDS:
            # If this category is unseen, map it to -1 category.
            # This allows then easy filtering on unseen categories.
            contiguous_id_to_seen_contiguous_id.append(-1)
            unseen_dataset_id_to_contiguous_id[cat["trainId"]] = i
        else:
            contiguous_id_to_seen_contiguous_id.append(last_seen_id)
            seen_dataset_id_to_seen_contiguous_id[cat["trainId"]] = last_seen_id
            dataset_id_to_seen_contigous_id[cat["trainId"]] = last_seen_id
            last_seen_id += 1
            seen_dataset_id_to_contiguous_id[cat["trainId"]] = i

    meta["stuff_classes"] = stuff_classes
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id
    meta["seen_dataset_id_to_contiguous_id"] = seen_dataset_id_to_contiguous_id
    meta["dataset_id_to_seen_contigous_id"] = dataset_id_to_seen_contigous_id
    meta["seen_dataset_id_to_seen_contiguous_id"] = seen_dataset_id_to_seen_contiguous_id
    meta["unseen_dataset_id_to_contiguous_id"] = unseen_dataset_id_to_contiguous_id
    meta["contiguous_id_to_seen_contiguous_id"] = contiguous_id_to_seen_contiguous_id

    return meta


def register_all_ade20k_847(root):
    root = os.path.join(root, "ADE20K_2021_17_01")
    metadata = get_metadata()
    for name, dirname in [("train", "training"), ("val", "validation")]:
        image_dir = os.path.join(root, "images_detectron2", dirname)
        gt_dir = os.path.join(root, "annotations_detectron2", dirname)
        name = f"openvocab_ade20k_full_sem_seg_{name}"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="tif", image_ext="jpg")
        )
        MetadataCatalog.get(name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="zs_sem_seg",
            ignore_label=65535,  # NOTE: gt is saved in 16-bit TIFF images
            gt_ext="tif",
            **metadata,
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_ade20k_847(_root)