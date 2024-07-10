from os.path import join as ospj
import math
import json
import codecs
import numpy as np
from PIL import Image
from munch import Munch as mch
import torch
import torchvision as tv
import torchvision.transforms.functional as F
from typing import List, Optional, Tuple, Union
from torch import Tensor
import numbers
from collections.abc import Sequence
import random
from PIL import ImageFilter, ImageOps


SUBSET_NAMES = {
    "imagenet": [
        "tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark", "electric ray",
        "stingray", "rooster", "hen", "ostrich", "brambling", "goldfinch", "house finch", "junco",
        "indigo bunting", "American robin", "bulbul", "jay", "magpie", "chickadee", "American dipper",
        "kite (bird of prey)", "bald eagle", "vulture", "great grey owl", "fire salamander",
        "smooth newt", "newt", "spotted salamander", "axolotl", "American bullfrog", "tree frog",
        "tailed frog", "loggerhead sea turtle", "leatherback sea turtle", "mud turtle", "terrapin",
        "box turtle", "banded gecko", "green iguana", "Carolina anole",
        "desert grassland whiptail lizard", "agama", "frilled-necked lizard", "alligator lizard",
        "Gila monster", "European green lizard", "chameleon", "Komodo dragon", "Nile crocodile",
        "American alligator", "triceratops", "worm snake", "ring-necked snake",
        "eastern hog-nosed snake", "smooth green snake", "kingsnake", "garter snake", "water snake",
        "vine snake", "night snake", "boa constrictor", "African rock python", "Indian cobra",
        "green mamba", "sea snake", "Saharan horned viper", "eastern diamondback rattlesnake",
        "sidewinder rattlesnake", "trilobite", "harvestman", "scorpion", "yellow garden spider",
        "barn spider", "European garden spider", "southern black widow", "tarantula", "wolf spider",
        "tick", "centipede", "black grouse", "ptarmigan", "ruffed grouse", "prairie grouse", "peafowl",
        "quail", "partridge", "african grey parrot", "macaw", "sulphur-crested cockatoo", "lorikeet",
        "coucal", "bee eater", "hornbill", "hummingbird", "jacamar", "toucan", "duck",
        "red-breasted merganser", "goose", "black swan", "tusker", "echidna", "platypus", "wallaby",
        "koala", "wombat", "jellyfish", "sea anemone", "brain coral", "flatworm", "nematode", "conch",
        "snail", "slug", "sea slug", "chiton", "chambered nautilus", "Dungeness crab", "rock crab",
        "fiddler crab", "red king crab", "American lobster", "spiny lobster", "crayfish", "hermit crab",
        "isopod", "white stork", "black stork", "spoonbill", "flamingo", "little blue heron",
        "great egret", "bittern bird", "crane bird", "limpkin", "common gallinule", "American coot",
        "bustard", "ruddy turnstone", "dunlin", "common redshank", "dowitcher", "oystercatcher",
        "pelican", "king penguin", "albatross", "grey whale", "killer whale", "dugong", "sea lion",
        "Chihuahua", "Japanese Chin", "Maltese", "Pekingese", "Shih Tzu", "King Charles Spaniel",
        "Papillon", "toy terrier", "Rhodesian Ridgeback", "Afghan Hound", "Basset Hound", "Beagle",
        "Bloodhound", "Bluetick Coonhound", "Black and Tan Coonhound", "Treeing Walker Coonhound",
        "English foxhound", "Redbone Coonhound", "borzoi", "Irish Wolfhound", "Italian Greyhound",
        "Whippet", "Ibizan Hound", "Norwegian Elkhound", "Otterhound", "Saluki", "Scottish Deerhound",
        "Weimaraner", "Staffordshire Bull Terrier", "American Staffordshire Terrier",
        "Bedlington Terrier", "Border Terrier", "Kerry Blue Terrier", "Irish Terrier",
        "Norfolk Terrier", "Norwich Terrier", "Yorkshire Terrier", "Wire Fox Terrier",
        "Lakeland Terrier", "Sealyham Terrier", "Airedale Terrier", "Cairn Terrier",
        "Australian Terrier", "Dandie Dinmont Terrier", "Boston Terrier", "Miniature Schnauzer",
        "Giant Schnauzer", "Standard Schnauzer", "Scottish Terrier", "Tibetan Terrier",
        "Australian Silky Terrier", "Soft-coated Wheaten Terrier", "West Highland White Terrier",
        "Lhasa Apso", "Flat-Coated Retriever", "Curly-coated Retriever", "Golden Retriever",
        "Labrador Retriever", "Chesapeake Bay Retriever", "German Shorthaired Pointer", "Vizsla",
        "English Setter", "Irish Setter", "Gordon Setter", "Brittany dog", "Clumber Spaniel",
        "English Springer Spaniel", "Welsh Springer Spaniel", "Cocker Spaniel", "Sussex Spaniel",
        "Irish Water Spaniel", "Kuvasz", "Schipperke", "Groenendael dog", "Malinois", "Briard",
        "Australian Kelpie", "Komondor", "Old English Sheepdog", "Shetland Sheepdog", "collie",
        "Border Collie", "Bouvier des Flandres dog", "Rottweiler", "German Shepherd Dog", "Dobermann",
        "Miniature Pinscher", "Greater Swiss Mountain Dog", "Bernese Mountain Dog",
        "Appenzeller Sennenhund", "Entlebucher Sennenhund", "Boxer", "Bullmastiff", "Tibetan Mastiff",
        "French Bulldog", "Great Dane", "St. Bernard", "husky", "Alaskan Malamute", "Siberian Husky",
        "Dalmatian", "Affenpinscher", "Basenji", "pug", "Leonberger", "Newfoundland dog",
        "Great Pyrenees dog", "Samoyed", "Pomeranian", "Chow Chow", "Keeshond", "brussels griffon",
        "Pembroke Welsh Corgi", "Cardigan Welsh Corgi", "Toy Poodle", "Miniature Poodle",
        "Standard Poodle", "Mexican hairless dog (xoloitzcuintli)", "grey wolf", "Alaskan tundra wolf",
        "red wolf or maned wolf", "coyote", "dingo", "dhole", "African wild dog", "hyena", "red fox",
        "kit fox", "Arctic fox", "grey fox", "tabby cat", "tiger cat", "Persian cat", "Siamese cat",
        "Egyptian Mau", "cougar", "lynx", "leopard", "snow leopard", "jaguar", "lion", "tiger",
        "cheetah", "brown bear", "American black bear", "polar bear", "sloth bear", "mongoose",
        "meerkat", "tiger beetle", "ladybug", "ground beetle", "longhorn beetle", "leaf beetle",
        "dung beetle", "rhinoceros beetle", "weevil", "fly", "bee", "ant", "grasshopper",
        "cricket insect", "stick insect", "cockroach", "praying mantis", "cicada", "leafhopper",
        "lacewing", "dragonfly", "damselfly", "red admiral butterfly", "ringlet butterfly",
        "monarch butterfly", "small white butterfly", "sulphur butterfly", "gossamer-winged butterfly",
        "starfish", "sea urchin", "sea cucumber", "cottontail rabbit", "hare", "Angora rabbit",
        "hamster", "porcupine", "fox squirrel", "marmot", "beaver", "guinea pig", "common sorrel horse",
        "zebra", "pig", "wild boar", "warthog", "hippopotamus", "ox", "water buffalo", "bison",
        "ram (adult male sheep)", "bighorn sheep", "Alpine ibex", "hartebeest", "impala (antelope)",
        "gazelle", "arabian camel", "llama", "weasel", "mink", "European polecat",
        "black-footed ferret", "otter", "skunk", "badger", "armadillo", "three-toed sloth", "orangutan",
        "gorilla", "chimpanzee", "gibbon", "siamang", "guenon", "patas monkey", "baboon", "macaque",
        "langur", "black-and-white colobus", "proboscis monkey", "marmoset", "white-headed capuchin",
        "howler monkey", "titi monkey", "Geoffroy's spider monkey", "common squirrel monkey",
        "ring-tailed lemur", "indri", "Asian elephant", "African bush elephant", "red panda",
        "giant panda", "snoek fish", "eel", "silver salmon", "rock beauty fish", "clownfish",
        "sturgeon", "gar fish", "lionfish", "pufferfish", "abacus", "abaya", "academic gown",
        "accordion", "acoustic guitar", "aircraft carrier", "airliner", "airship", "altar", "ambulance",
        "amphibious vehicle", "analog clock", "apiary", "apron", "trash can", "assault rifle",
        "backpack", "bakery", "balance beam", "balloon", "ballpoint pen", "Band-Aid", "banjo",
        "baluster / handrail", "barbell", "barber chair", "barbershop", "barn", "barometer", "barrel",
        "wheelbarrow", "baseball", "basketball", "bassinet", "bassoon", "swimming cap", "bath towel",
        "bathtub", "station wagon", "lighthouse", "beaker", "military hat (bearskin or shako)",
        "beer bottle", "beer glass", "bell tower", "baby bib", "tandem bicycle", "bikini",
        "ring binder", "binoculars", "birdhouse", "boathouse", "bobsleigh", "bolo tie", "poke bonnet",
        "bookcase", "bookstore", "bottle cap", "hunting bow", "bow tie", "brass memorial plaque", "bra",
        "breakwater", "breastplate", "broom", "bucket", "buckle", "bulletproof vest",
        "high-speed train", "butcher shop", "taxicab", "cauldron", "candle", "cannon", "canoe",
        "can opener", "cardigan", "car mirror", "carousel", "tool kit", "cardboard box / carton",
        "car wheel", "automated teller machine", "cassette", "cassette player", "castle", "catamaran",
        "CD player", "cello", "mobile phone", "chain", "chain-link fence", "chain mail", "chainsaw",
        "storage chest", "chiffonier", "bell or wind chime", "china cabinet", "Christmas stocking",
        "church", "movie theater", "cleaver", "cliff dwelling", "cloak", "clogs", "cocktail shaker",
        "coffee mug", "coffeemaker", "spiral or coil", "combination lock", "computer keyboard",
        "candy store", "container ship", "convertible", "corkscrew", "cornet", "cowboy boot",
        "cowboy hat", "cradle", "construction crane", "crash helmet", "crate", "infant bed",
        "Crock Pot", "croquet ball", "crutch", "cuirass", "dam", "desk", "desktop computer",
        "rotary dial telephone", "diaper", "digital clock", "digital watch", "dining table",
        "dishcloth", "dishwasher", "disc brake", "dock", "dog sled", "dome", "doormat", "drilling rig",
        "drum", "drumstick", "dumbbell", "Dutch oven", "electric fan", "electric guitar",
        "electric locomotive", "entertainment center", "envelope", "espresso machine", "face powder",
        "feather boa", "filing cabinet", "fireboat", "fire truck", "fire screen", "flagpole", "flute",
        "folding chair", "football helmet", "forklift", "fountain", "fountain pen", "four-poster bed",
        "freight car", "French horn", "frying pan", "fur coat", "garbage truck",
        "gas mask or respirator", "gas pump", "goblet", "go-kart", "golf ball", "golf cart", "gondola",
        "gong", "gown", "grand piano", "greenhouse", "radiator grille", "grocery store", "guillotine",
        "hair clip", "hair spray", "half-track", "hammer", "hamper", "hair dryer", "hand-held computer",
        "handkerchief", "hard disk drive", "harmonica", "harp", "combine harvester", "hatchet",
        "holster", "home theater", "honeycomb", "hook", "hoop skirt", "gymnastic horizontal bar",
        "horse-drawn vehicle", "hourglass", "iPod", "clothes iron", "carved pumpkin", "jeans", "jeep",
        "T-shirt", "jigsaw puzzle", "rickshaw", "joystick", "kimono", "knee pad", "knot", "lab coat",
        "ladle", "lampshade", "laptop computer", "lawn mower", "lens cap", "letter opener", "library",
        "lifeboat", "lighter", "limousine", "ocean liner", "lipstick", "slip-on shoe", "lotion",
        "music speaker", "loupe magnifying glass", "sawmill", "magnetic compass", "messenger bag",
        "mailbox", "tights", "one-piece bathing suit", "manhole cover", "maraca", "marimba", "mask",
        "matchstick", "maypole", "maze", "measuring cup", "medicine cabinet", "megalith", "microphone",
        "microwave oven", "military uniform", "milk can", "minibus", "miniskirt", "minivan", "missile",
        "mitten", "mixing bowl", "mobile home", "ford model t", "modem", "monastery", "monitor",
        "moped", "mortar and pestle", "graduation cap", "mosque", "mosquito net", "vespa",
        "mountain bike", "tent", "computer mouse", "mousetrap", "moving van", "muzzle", "metal nail",
        "neck brace", "necklace", "baby pacifier", "notebook computer", "obelisk", "oboe", "ocarina",
        "odometer", "oil filter", "pipe organ", "oscilloscope", "overskirt", "bullock cart",
        "oxygen mask", "product packet / packaging", "paddle", "paddle wheel", "padlock", "paintbrush",
        "pajamas", "palace", "pan flute", "paper towel", "parachute", "parallel bars", "park bench",
        "parking meter", "railroad car", "patio", "payphone", "pedestal", "pencil case",
        "pencil sharpener", "perfume", "Petri dish", "photocopier", "plectrum", "Pickelhaube",
        "picket fence", "pickup truck", "pier", "piggy bank", "pill bottle", "pillow", "ping-pong ball",
        "pinwheel", "pirate ship", "drink pitcher", "block plane", "planetarium", "plastic bag",
        "plate rack", "farm plow", "plunger", "Polaroid camera", "pole", "police van", "poncho",
        "pool table", "soda bottle", "plant pot", "potter's wheel", "power drill", "prayer rug",
        "printer", "prison", "projectile missile", "projector", "hockey puck", "punching bag", "purse", "quill",
        "quilt", "race car", "racket", "radiator", "radio", "radio telescope", "rain barrel",
        "recreational vehicle", "fishing casting reel", "reflex camera", "refrigerator",
        "remote control", "restaurant", "revolver", "rifle", "rocking chair", "rotisserie", "eraser",
        "rugby ball", "ruler measuring stick", "sneaker", "safe", "safety pin", "salt shaker", "sandal",
        "sarong", "saxophone", "scabbard", "weighing scale", "school bus", "schooner", "scoreboard",
        "CRT monitor", "screw", "screwdriver", "seat belt", "sewing machine", "shield", "shoe store",
        "shoji screen / room divider", "shopping basket", "shopping cart", "shovel", "shower cap",
        "shower curtain", "ski", "balaclava ski mask", "sleeping bag", "slide rule", "sliding door",
        "slot machine", "snorkel", "snowmobile", "snowplow", "soap dispenser", "soccer ball", "sock",
        "solar thermal collector", "sombrero", "soup bowl", "keyboard space bar", "space heater",
        "space shuttle", "spatula", "motorboat", "spider web", "spindle", "sports car", "spotlight",
        "stage", "steam locomotive", "through arch bridge", "steel drum", "stethoscope", "scarf",
        "stone wall", "stopwatch", "stove", "strainer", "tram", "stretcher", "couch", "stupa",
        "submarine", "suit", "sundial", "sunglasses", "dark sunglasses", "sunscreen", "suspension bridge",
        "mop", "sweatshirt", "swim trunks / shorts", "swing", "electrical switch", "syringe",
        "table lamp", "tank", "tape player", "teapot", "teddy bear", "television", "tennis ball",
        "thatched roof", "front curtain", "thimble", "threshing machine", "throne", "tile roof",
        "toaster", "tobacco shop", "toilet seat", "torch", "totem pole", "tow truck", "toy store",
        "tractor", "semi-trailer truck", "tray", "trench coat", "tricycle", "trimaran", "tripod",
        "triumphal arch", "trolleybus", "trombone", "hot tub", "turnstile", "typewriter keyboard",
        "umbrella", "unicycle", "upright piano", "vacuum cleaner", "vase", "vaulted or arched ceiling",
        "velvet fabric", "vending machine", "vestment", "viaduct", "violin", "volleyball",
        "waffle iron", "wall clock", "wallet", "wardrobe", "military aircraft", "sink",
        "washing machine", "water bottle", "water jug", "water tower", "whiskey jug", "whistle",
        "hair wig", "window screen", "window shade", "Windsor tie", "wine bottle", "airplane wing",
        "wok", "wooden spoon", "wool", "split-rail fence", "shipwreck", "sailboat", "yurt", "website",
        "comic book", "crossword", "traffic or street sign", "traffic light", "dust jacket", "menu",
        "plate", "guacamole", "consomme", "hot pot", "trifle", "ice cream", "popsicle", "baguette",
        "bagel", "pretzel", "cheeseburger", "hot dog", "mashed potatoes", "cabbage", "broccoli",
        "cauliflower", "zucchini", "spaghetti squash", "acorn squash", "butternut squash", "cucumber",
        "artichoke", "bell pepper", "cardoon", "mushroom", "Granny Smith apple", "strawberry", "orange",
        "lemon", "fig", "pineapple", "banana", "jackfruit", "cherimoya (custard apple)", "pomegranate",
        "hay", "carbonara", "chocolate syrup", "dough", "meatloaf", "pizza", "pot pie", "burrito",
        "red wine", "espresso", "tea cup", "eggnog", "mountain", "bubble", "cliff", "coral reef",
        "geyser", "lakeshore", "promontory", "sandbar", "beach", "valley", "volcano", "baseball player",
        "bridegroom", "scuba diver", "rapeseed", "daisy", "yellow lady's slipper", "corn", "acorn",
        "rose hip", "horse chestnut seed", "coral fungus", "agaric", "gyromitra", "stinkhorn mushroom",
        "earth star fungus", "hen of the woods mushroom", "bolete", "corn cob", "toilet paper"
    ],
    'imagenet_100': [
        "robin", "Gila monster", "hognose snake", "garter snake", "green mamba",
        "garden spider", "lorikeet", "goose", "rock crab", "fiddler crab",
        "American lobster", "little blue heron", "American coot", "Chihuahua", "Shih-Tzu",
        "papillon", "toy terrier", "Walker hound", "English foxhound", "borzoi",
        "Saluki", "American Staffordshire terrier", "Chesapeake Bay retriever", "vizsla", "kuvasz",
        "komondor", "Rottweiler", "Doberman", "boxer", "Great Dane",
        "standard poodle", "Mexican hairless", "coyote", "African hunting dog", "red fox",
        "tabby", "meerkat", "dung beetle", "walking stick", "leafhopper",
        "hare", "wild boar", "gibbon", "langur", "ambulance",
        "bannister", "bassinet", "boathouse", "bonnet", "bottlecap",
        "car wheel", "chime", "cinema", "cocktail shaker", "computer keyboard",
        "Dutch oven", "football helmet", "gasmask", "hard disc", "harmonica",
        "honeycomb", "iron", "jean", "lampshade", "laptop",
        "milk can", "mixing bowl", "modem", "moped", "mortarboard",
        "mousetrap", "obelisk", "park bench", "pedestal", "pickup",
        "pirate", "purse", "reel", "rocking chair", "rotisserie",
        "safety pin", "sarong", "ski mask", "slide rule", "stretcher",
        "theater curtain", "throne", "tile roof", "tripod", "tub",
        "vacuum", "window screen", "wing", "head cabbage", "cauliflower",
        "pineapple", "carbonara", "chocolate sauce", "gyromitra", "stinkhorn",
    ],
    'pets': [
        'Abyssinian', 'American Bulldog', 'American Pit Bull Terrier', 'Basset Hound', 'Beagle',
        'Bengal', 'Birman', 'Bombay', 'Boxer', 'British Shorthair',
        'Chihuahua', 'Egyptian Mau', 'English Cocker Spaniel', 'English Setter', 'German Shorthaired',
        'Great Pyrenees', 'Havanese', 'Japanese Chin', 'Keeshond', 'Leonberger',
        'Maine Coon', 'Miniature Pinscher', 'Newfoundland', 'Persian', 'Pomeranian',
        'Pug', 'Ragdoll', 'Russian Blue', 'Saint Bernard', 'Samoyed',
        'Scottish Terrier', 'Shiba Inu', 'Siamese', 'Sphynx', 'Staffordshire Bull Terrier',
        'Wheaten Terrier', 'Yorkshire Terrier'
    ],
    'stl10': [
        'airplane', 'bird', 'car', 'cat', 'deer',
        'dog', 'horse', 'monkey', 'ship', 'truck'
    ],
    'food101': [
        'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare',
        'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito',
        'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake',
        'ceviche', 'cheese_plate', 'cheesecake', 'chicken_curry', 'chicken_quesadilla',
        'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder',
        'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes',
        'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict',
        'escargots', 'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras',
        'french_fries', 'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice',
        'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich',
        'grilled_salmon', 'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup',
        'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna',
        'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons', 'miso_soup',
        'mussels', 'nachos', 'omelette', 'onion_rings', 'oysters',
        'pad_thai', 'paella', 'pancakes', 'panna_cotta', 'peking_duck',
        'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib',
        'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake', 'risotto',
        'samosa', 'sashimi', 'scallops', 'seaweed_salad', 'shrimp_and_grits',
        'spaghetti_bolognese', 'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake',
        'sushi', 'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare',
        'waffles'
    ],
    'fgvc_aircraft': [
        '707-320', '727-200', '737-200', '737-300', '737-400',
        '737-500', '737-600', '737-700', '737-800', '737-900',
        '747-100', '747-200', '747-300', '747-400', '757-200',
        '757-300', '767-200', '767-300', '767-400', '777-200',
        '777-300', 'A300B4', 'A310', 'A318', 'A319',
        'A320', 'A321', 'A330-200', 'A330-300', 'A340-200',
        'A340-300', 'A340-500', 'A340-600', 'A380', 'ATR-42',
        'ATR-72', 'An-12', 'BAE 146-200', 'BAE 146-300', 'BAE-125',
        'Beechcraft 1900', 'Boeing 717', 'C-130', 'C-47', 'CRJ-200',
        'CRJ-700', 'CRJ-900', 'Cessna 172', 'Cessna 208', 'Cessna 525',
        'Cessna 560', 'Challenger 600', 'DC-10', 'DC-3', 'DC-6',
        'DC-8', 'DC-9-30', 'DH-82', 'DHC-1', 'DHC-6',
        'DHC-8-100', 'DHC-8-300', 'DR-400', 'Dornier 328', 'E-170',
        'E-190', 'E-195', 'EMB-120', 'ERJ 135', 'ERJ 145',
        'Embraer Legacy 600', 'Eurofighter Typhoon', 'F-16A/B', 'F/A-18', 'Falcon 2000',
        'Falcon 900', 'Fokker 100', 'Fokker 50', 'Fokker 70', 'Global Express',
        'Gulfstream IV', 'Gulfstream V', 'Hawk T1', 'Il-76', 'L-1011',
        'MD-11', 'MD-80', 'MD-87', 'MD-90', 'Metroliner',
        'Model B200', 'PA-28', 'SR-20', 'Saab 2000', 'Saab 340',
        'Spitfire', 'Tornado', 'Tu-134', 'Tu-154', 'Yak-42'
    ],
    'eurosat': [
        'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
        'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'
    ],
    'cars': [
        'AM General Hummer SUV 2000', 'Acura RL Sedan 2012', 'Acura TL Sedan 2012', 'Acura TL Type-S 2008', 'Acura TSX Sedan 2012',
        'Acura Integra Type R 2001', 'Acura ZDX Hatchback 2012', 'Aston Martin V8 Vantage Convertible 2012', 'Aston Martin V8 Vantage Coupe 2012', 'Aston Martin Virage Convertible 2012',
        'Aston Martin Virage Coupe 2012', 'Audi RS 4 Convertible 2008', 'Audi A5 Coupe 2012', 'Audi TTS Coupe 2012', 'Audi R8 Coupe 2012',
        'Audi V8 Sedan 1994', 'Audi 100 Sedan 1994', 'Audi 100 Wagon 1994', 'Audi TT Hatchback 2011', 'Audi S6 Sedan 2011',
        'Audi S5 Convertible 2012', 'Audi S5 Coupe 2012', 'Audi S4 Sedan 2012', 'Audi S4 Sedan 2007', 'Audi TT RS Coupe 2012',
        'BMW ActiveHybrid 5 Sedan 2012', 'BMW 1 Series Convertible 2012', 'BMW 1 Series Coupe 2012', 'BMW 3 Series Sedan 2012', 'BMW 3 Series Wagon 2012',
        'BMW 6 Series Convertible 2007', 'BMW X5 SUV 2007', 'BMW X6 SUV 2012', 'BMW M3 Coupe 2012', 'BMW M5 Sedan 2010',
        'BMW M6 Convertible 2010', 'BMW X3 SUV 2012', 'BMW Z4 Convertible 2012', 'Bentley Continental Supersports Conv. Convertible 2012', 'Bentley Arnage Sedan 2009',
        'Bentley Mulsanne Sedan 2011', 'Bentley Continental GT Coupe 2012', 'Bentley Continental GT Coupe 2007', 'Bentley Continental Flying Spur Sedan 2007', 'Bugatti Veyron 16.4 Convertible 2009',
        'Bugatti Veyron 16.4 Coupe 2009', 'Buick Regal GS 2012', 'Buick Rainier SUV 2007', 'Buick Verano Sedan 2012', 'Buick Enclave SUV 2012',
        'Cadillac CTS-V Sedan 2012', 'Cadillac SRX SUV 2012', 'Cadillac Escalade EXT Crew Cab 2007', 'Chevrolet Silverado 1500 Hybrid Crew Cab 2012', 'Chevrolet Corvette Convertible 2012',
        'Chevrolet Corvette ZR1 2012', 'Chevrolet Corvette Ron Fellows Edition Z06 2007', 'Chevrolet Traverse SUV 2012', 'Chevrolet Camaro Convertible 2012', 'Chevrolet HHR SS 2010',
        'Chevrolet Impala Sedan 2007', 'Chevrolet Tahoe Hybrid SUV 2012', 'Chevrolet Sonic Sedan 2012', 'Chevrolet Express Cargo Van 2007', 'Chevrolet Avalanche Crew Cab 2012',
        'Chevrolet Cobalt SS 2010', 'Chevrolet Malibu Hybrid Sedan 2010', 'Chevrolet TrailBlazer SS 2009', 'Chevrolet Silverado 2500HD Regular Cab 2012', 'Chevrolet Silverado 1500 Classic Extended Cab 2007',
        'Chevrolet Express Van 2007', 'Chevrolet Monte Carlo Coupe 2007', 'Chevrolet Malibu Sedan 2007', 'Chevrolet Silverado 1500 Extended Cab 2012', 'Chevrolet Silverado 1500 Regular Cab 2012',
        'Chrysler Aspen SUV 2009', 'Chrysler Sebring Convertible 2010', 'Chrysler Town and Country Minivan 2012', 'Chrysler 300 SRT-8 2010', 'Chrysler Crossfire Convertible 2008',
        'Chrysler PT Cruiser Convertible 2008', 'Daewoo Nubira Wagon 2002', 'Dodge Caliber Wagon 2012', 'Dodge Caliber Wagon 2007', 'Dodge Caravan Minivan 1997',
        'Dodge Ram Pickup 3500 Crew Cab 2010', 'Dodge Ram Pickup 3500 Quad Cab 2009', 'Dodge Sprinter Cargo Van 2009', 'Dodge Journey SUV 2012', 'Dodge Dakota Crew Cab 2010',
        'Dodge Dakota Club Cab 2007', 'Dodge Magnum Wagon 2008', 'Dodge Challenger SRT8 2011', 'Dodge Durango SUV 2012', 'Dodge Durango SUV 2007',
        'Dodge Charger Sedan 2012', 'Dodge Charger SRT-8 2009', 'Eagle Talon Hatchback 1998', 'FIAT 500 Abarth 2012', 'FIAT 500 Convertible 2012',
        'Ferrari FF Coupe 2012', 'Ferrari California Convertible 2012', 'Ferrari 458 Italia Convertible 2012', 'Ferrari 458 Italia Coupe 2012', 'Fisker Karma Sedan 2012',
        'Ford F-450 Super Duty Crew Cab 2012', 'Ford Mustang Convertible 2007', 'Ford Freestar Minivan 2007', 'Ford Expedition EL SUV 2009', 'Ford Edge SUV 2012',
        'Ford Ranger SuperCab 2011', 'Ford GT Coupe 2006', 'Ford F-150 Regular Cab 2012', 'Ford F-150 Regular Cab 2007', 'Ford Focus Sedan 2007',
        'Ford E-Series Wagon Van 2012', 'Ford Fiesta Sedan 2012', 'GMC Terrain SUV 2012', 'GMC Savana Van 2012', 'GMC Yukon Hybrid SUV 2012',
        'GMC Acadia SUV 2012', 'GMC Canyon Extended Cab 2012', 'Geo Metro Convertible 1993', 'HUMMER H3T Crew Cab 2010', 'HUMMER H2 SUT Crew Cab 2009',
        'Honda Odyssey Minivan 2012', 'Honda Odyssey Minivan 2007', 'Honda Accord Coupe 2012', 'Honda Accord Sedan 2012', 'Hyundai Veloster Hatchback 2012',
        'Hyundai Santa Fe SUV 2012', 'Hyundai Tucson SUV 2012', 'Hyundai Veracruz SUV 2012', 'Hyundai Sonata Hybrid Sedan 2012', 'Hyundai Elantra Sedan 2007',
        'Hyundai Accent Sedan 2012', 'Hyundai Genesis Sedan 2012', 'Hyundai Sonata Sedan 2012', 'Hyundai Elantra Touring Hatchback 2012', 'Hyundai Azera Sedan 2012',
        'Infiniti G Coupe IPL 2012', 'Infiniti QX56 SUV 2011', 'Isuzu Ascender SUV 2008', 'Jaguar XK XKR 2012', 'Jeep Patriot SUV 2012',
        'Jeep Wrangler SUV 2012', 'Jeep Liberty SUV 2012', 'Jeep Grand Cherokee SUV 2012', 'Jeep Compass SUV 2012', 'Lamborghini Reventon Coupe 2008',
        'Lamborghini Aventador Coupe 2012', 'Lamborghini Gallardo LP 570-4 Superleggera 2012', 'Lamborghini Diablo Coupe 2001', 'Land Rover Range Rover SUV 2012', 'Land Rover LR2 SUV 2012',
        'Lincoln Town Car Sedan 2011', 'MINI Cooper Roadster Convertible 2012', 'Maybach Landaulet Convertible 2012', 'Mazda Tribute SUV 2011', 'McLaren MP4-12C Coupe 2012',
        'Mercedes-Benz 300-Class Convertible 1993', 'Mercedes-Benz C-Class Sedan 2012', 'Mercedes-Benz SL-Class Coupe 2009', 'Mercedes-Benz E-Class Sedan 2012', 'Mercedes-Benz S-Class Sedan 2012',
        'Mercedes-Benz Sprinter Van 2012', 'Mitsubishi Lancer Sedan 2012', 'Nissan Leaf Hatchback 2012', 'Nissan NV Passenger Van 2012', 'Nissan Juke Hatchback 2012',
        'Nissan 240SX Coupe 1998', 'Plymouth Neon Coupe 1999', 'Porsche Panamera Sedan 2012', 'Ram C/V Cargo Van Minivan 2012', 'Rolls-Royce Phantom Drophead Coupe Convertible 2012',
        'Rolls-Royce Ghost Sedan 2012', 'Rolls-Royce Phantom Sedan 2012', 'Scion xD Hatchback 2012', 'Spyker C8 Convertible 2009', 'Spyker C8 Coupe 2009',
        'Suzuki Aerio Sedan 2007', 'Suzuki Kizashi Sedan 2012', 'Suzuki SX4 Hatchback 2012', 'Suzuki SX4 Sedan 2012', 'Tesla Model S Sedan 2012',
        'Toyota Sequoia SUV 2012', 'Toyota Camry Sedan 2012', 'Toyota Corolla Sedan 2012', 'Toyota 4Runner SUV 2012', 'Volkswagen Golf Hatchback 2012',
        'Volkswagen Golf Hatchback 1991', 'Volkswagen Beetle Hatchback 2012', 'Volvo C30 Hatchback 2012', 'Volvo 240 Sedan 1993', 'Volvo XC90 SUV 2007',
        'smart fortwo Convertible 2012'],
    'dtd': [
        'banded', 'blotchy', 'braided', 'bubbly', 'bumpy',
        'chequered', 'cobwebbed', 'cracked', 'crosshatched', 'crystalline',
        'dotted', 'fibrous', 'flecked', 'freckled', 'frilly',
        'gauzy', 'grid', 'grooved', 'honeycombed', 'interlaced',
        'knitted', 'lacelike', 'lined', 'marbled', 'matted',
        'meshed', 'paisley', 'perforated', 'pitted', 'pleated',
        'polka-dotted', 'porous', 'potholed', 'scaly', 'smeared',
        'spiralled', 'sprinkled', 'stained', 'stratified', 'striped',
        'studded', 'swirly', 'veined', 'waffled', 'woven',
        'wrinkled', 'zigzagged'
    ],
    'flowers102': [
        'pink primrose', 'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea', 'english marigold',
        'tiger lily', 'moon orchid', 'bird of paradise', 'monkshood', 'globe thistle',
        'snapdragon', "colt's foot", 'king protea', 'spear thistle', 'yellow iris',
        'globe-flower', 'purple coneflower', 'peruvian lily', 'balloon flower', 'giant white arum lily',
        'fire lily', 'pincushion flower', 'fritillary', 'red ginger', 'grape hyacinth',
        'corn poppy', 'prince of wales feathers', 'stemless gentian', 'artichoke', 'sweet william',
        'carnation', 'garden phlox', 'love in the mist', 'mexican aster', 'alpine sea holly',
        'ruby-lipped cattleya', 'cape flower', 'great masterwort', 'siam tulip', 'lenten rose',
        'barbeton daisy', 'daffodil', 'sword lily', 'poinsettia', 'bolero deep blue',
        'wallflower', 'marigold', 'buttercup', 'oxeye daisy', 'common dandelion',
        'petunia', 'wild pansy', 'primula', 'sunflower', 'pelargonium',
        'bishop of llandaff', 'gaura', 'geranium', 'orange dahlia', 'pink-yellow dahlia',
        'cautleya spicata', 'japanese anemone', 'black-eyed susan', 'silverbush', 'californian poppy',
        'osteospermum', 'spring crocus', 'bearded iris', 'windflower', 'tree poppy',
        'gazania', 'azalea', 'water lily', 'rose', 'thorn apple',
        'morning glory', 'passion flower', 'lotus lotus', 'toad lily', 'anthurium',
        'frangipani', 'clematis', 'hibiscus', 'columbine', 'desert-rose',
        'tree mallow', 'magnolia', 'cyclamen', 'watercress', 'canna lily',
        'hippeastrum', 'bee balm', 'ball moss', 'foxglove', 'bougainvillea',
        'camellia', 'mallow', 'mexican petunia', 'bromelia', 'blanket flower',
        'trumpet creeper', 'blackberry lily'
    ],
    'sun397': [
        'abbey', 'airplane_cabin', 'airport_terminal', 'alley', 'amphitheater',
        'amusement_arcade', 'amusement_park', 'anechoic_chamber', 'outdoor apartment_building', 'indoor apse',
        'aquarium', 'aqueduct', 'arch', 'archive', 'outdoor arrival_gate',
        'art_gallery', 'art_school', 'art_studio', 'assembly_line', 'outdoor athletic_field',
        'public atrium', 'attic', 'auditorium', 'auto_factory', 'badlands',
        'indoor badminton_court', 'baggage_claim', 'shop bakery', 'exterior balcony', 'interior balcony',
        'ball_pit', 'ballroom', 'bamboo_forest', 'banquet_hall', 'bar',
        'barn', 'barndoor', 'baseball_field', 'basement', 'basilica',
        'outdoor basketball_court', 'bathroom', 'batters_box', 'bayou', 'indoor bazaar',
        'outdoor bazaar', 'beach', 'beauty_salon', 'bedroom', 'berth',
        'biology_laboratory', 'indoor bistro', 'boardwalk', 'boat_deck', 'boathouse',
        'bookstore', 'indoor booth', 'botanical_garden', 'indoor bow_window', 'outdoor bow_window',
        'bowling_alley', 'boxing_ring', 'indoor brewery', 'bridge', 'building_facade',
        'bullring', 'burial_chamber', 'bus_interior', 'butchers_shop', 'butte',
        'outdoor cabin', 'cafeteria', 'campsite', 'campus', 'natural canal',
        'urban canal', 'candy_store', 'canyon', 'backseat car_interior', 'frontseat car_interior',
        'carrousel', 'indoor casino', 'castle', 'catacomb', 'indoor cathedral', 'outdoor cathedral',
        'indoor cavern', 'cemetery', 'chalet', 'cheese_factory', 'chemistry_lab',
        'indoor chicken_coop', 'outdoor chicken_coop', 'childs_room', 'indoor church', 'outdoor church',
        'classroom', 'clean_room', 'cliff', 'indoor cloister', 'closet',
        'clothing_store', 'coast', 'cockpit', 'coffee_shop', 'computer_room',
        'conference_center', 'conference_room', 'construction_site', 'control_room', 'outdoor control_tower',
        'corn_field', 'corral', 'corridor', 'cottage_garden', 'courthouse',
        'courtroom', 'courtyard', 'exterior covered_bridge', 'creek', 'crevasse',
        'crosswalk', 'office cubicle', 'dam', 'delicatessen', 'dentists_office',
        'sand desert', 'vegetation desert', 'indoor diner', 'outdoor diner', 'home dinette',
        'vehicle dinette', 'dining_car', 'dining_room', 'discotheque', 'dock',
        'outdoor doorway', 'dorm_room', 'driveway', 'outdoor driving_range', 'drugstore',
        'electrical_substation', 'door elevator', 'interior elevator', 'elevator_shaft', 'engine_room',
        'indoor escalator', 'excavation', 'indoor factory', 'fairway', 'fastfood_restaurant',
        'cultivated field', 'wild field', 'fire_escape', 'fire_station', 'indoor firing_range',
        'fishpond', 'indoor florist_shop', 'food_court', 'broadleaf forest', 'needleleaf forest',
        'forest_path', 'forest_road', 'formal_garden', 'fountain', 'galley',
        'game_room', 'indoor garage', 'garbage_dump', 'gas_station', 'exterior gazebo',
        'indoor general_store', 'outdoor general_store', 'gift_shop', 'golf_course', 'indoor greenhouse',
        'outdoor greenhouse', 'indoor gymnasium', 'indoor hangar', 'outdoor hangar', 'harbor',
        'hayfield', 'heliport', 'herb_garden', 'highway', 'hill', 'home_office',
        'hospital', 'hospital_room', 'hot_spring', 'outdoor hot_tub', 'outdoor hotel',
        'hotel_room', 'house', 'outdoor hunting_lodge', 'ice_cream_parlor', 'ice_floe',
        'ice_shelf', 'indoor ice_skating_rink', 'outdoor ice_skating_rink', 'iceberg', 'igloo',
        'industrial_area', 'outdoor inn', 'islet', 'indoor jacuzzi', 'indoor jail', 'jail_cell',
        'jewelry_shop', 'kasbah', 'indoor kennel', 'outdoor kennel', 'kindergarden_classroom',
        'kitchen', 'kitchenette', 'outdoor labyrinth', 'natural lake', 'landfill', 'landing_deck',
        'laundromat', 'lecture_room', 'indoor library', 'outdoor library', 'outdoor lido_deck',
        'lift_bridge', 'lighthouse', 'limousine_interior', 'living_room', 'lobby', 'lock_chamber',
        'locker_room', 'mansion', 'manufactured_home', 'indoor market', 'outdoor market', 'marsh',
        'martial_arts_gym', 'mausoleum', 'medina', 'water moat', 'outdoor monastery', 'indoor mosque',
        'outdoor mosque', 'motel', 'mountain', 'mountain_snowy', 'indoor movie_theater', 'indoor museum',
        'music_store', 'music_studio', 'outdoor nuclear_power_plant', 'nursery', 'oast_house',
        'outdoor observatory', 'ocean', 'office', 'office_building', 'outdoor oil_refinery',
        'oilrig', 'operating_room', 'orchard', 'outdoor outhouse', 'pagoda', 'palace', 'pantry',
        'park', 'indoor parking_garage', 'outdoor parking_garage', 'parking_lot', 'parlor',
        'pasture', 'patio', 'pavilion', 'pharmacy', 'phone_booth', 'physics_laboratory', 'picnic_area',
        'indoor pilothouse', 'outdoor planetarium', 'playground', 'playroom', 'plaza', 'indoor podium',
        'outdoor podium', 'pond', 'establishment poolroom', 'home poolroom', 'outdoor power_plant',
        'promenade_deck', 'indoor pub', 'pulpit', 'putting_green', 'racecourse', 'raceway', 'raft',
        'railroad_track', 'rainforest', 'reception', 'recreation_room', 'residential_neighborhood',
        'restaurant', 'restaurant_kitchen', 'restaurant_patio', 'rice_paddy', 'riding_arena', 'river',
        'rock_arch', 'rope_bridge', 'ruin', 'runway', 'sandbar', 'sandbox', 'sauna', 'schoolhouse',
        'sea_cliff', 'server_room', 'shed', 'shoe_shop', 'shopfront', 'indoor shopping_mall', 'shower',
        'skatepark', 'ski_lodge', 'ski_resort', 'ski_slope', 'sky', 'skyscraper', 'slum', 'snowfield',
        'squash_court', 'stable', 'baseball stadium', 'football stadium', 'indoor stage', 'staircase',
        'street', 'subway_interior', 'platform subway_station', 'supermarket', 'sushi_bar', 'swamp',
        'indoor swimming_pool', 'outdoor swimming_pool', 'indoor synagogue', 'outdoor synagogue',
        'television_studio', 'east_asia temple', 'south_asia temple', 'indoor tennis_court',
        'outdoor tennis_court', 'outdoor tent', 'indoor_procenium theater', 'indoor_seats theater',
        'thriftshop', 'throne_room', 'ticket_booth', 'toll_plaza', 'topiary_garden', 'tower', 'toyshop',
        'outdoor track', 'train_railway', 'platform train_station', 'tree_farm', 'tree_house', 'trench',
        'coral_reef underwater', 'utility_room', 'valley', 'van_interior', 'vegetable_garden', 'veranda',
        'veterinarians_office', 'viaduct', 'videostore', 'village', 'vineyard', 'volcano',
        'indoor volleyball_court', 'outdoor volleyball_court', 'waiting_room', 'indoor warehouse',
        'water_tower', 'block waterfall', 'fan waterfall', 'plunge waterfall', 'watering_hole', 'wave',
        'wet_bar', 'wheat_field', 'wind_farm', 'windmill', 'barrel_storage wine_cellar',
        'bottle_storage wine_cellar', 'indoor wrestling_ring', 'yard', 'youth_hostel'
    ],
    'caltech101': [
        'face', 'leopard', 'motorbike', 'accordion', 'airplane',
        'anchor', 'ant', 'barrel', 'bass', 'beaver',
        'binocular', 'bonsai', 'brain', 'brontosaurus', 'buddha',
        'butterfly', 'camera', 'cannon', 'car_side', 'ceiling_fan',
        'cellphone', 'chair', 'chandelier', 'cougar_body', 'cougar_face',
        'crab', 'crayfish', 'crocodile', 'crocodile_head', 'cup',
        'dalmatian', 'dollar_bill', 'dolphin', 'dragonfly', 'electric_guitar',
        'elephant', 'emu', 'euphonium', 'ewer', 'ferry',
        'flamingo', 'flamingo_head', 'garfield', 'gerenuk', 'gramophone',
        'grand_piano', 'hawksbill', 'headphone', 'hedgehog', 'helicopter',
        'ibis', 'inline_skate', 'joshua_tree', 'kangaroo', 'ketch',
        'lamp', 'laptop', 'llama', 'lobster', 'lotus',
        'mandolin', 'mayfly', 'menorah', 'metronome', 'minaret',
        'nautilus', 'octopus', 'okapi', 'pagoda', 'panda',
        'pigeon', 'pizza', 'platypus', 'pyramid', 'revolver',
        'rhino', 'rooster', 'saxophone', 'schooner', 'scissors',
        'scorpion', 'sea_horse', 'snoopy', 'soccer_ball', 'stapler',
        'starfish', 'stegosaurus', 'stop_sign', 'strawberry', 'sunflower',
        'tick', 'trilobite', 'umbrella', 'watch', 'water_lilly',
        'wheelchair', 'wild_cat', 'windsor_chair', 'wrench', 'yin_yang'
     ]
}


TEMPLATES_SMALL = [
    "a {}photo of a {}",
    "a {}rendering of a {}",
    "a {}cropped photo of the {}",
    "the {}photo of a {}",
    "a {}photo of a clean {}",
    "a {}photo of a dirty {}",
    "a dark {}photo of the {}",
    "a {}photo of my {}",
    "a {}photo of the cool {}",
    "a close-up {}photo of a {}",
    "a bright {}photo of the {}",
    "a cropped {}photo of a {}",
    "a {}photo of the {}",
    "a good {}photo of the {}",
    "a {}photo of one {}",
    "a close-up {}photo of the {}",
    "a {}rendition of the {}",
    "a {}photo of the clean {}",
    "a {}rendition of a {}",
    "a {}photo of a nice {}",
    "a good {}photo of a {}",
    "a {}photo of the nice {}",
    "a {}photo of the small {}",
    "a {}photo of the weird {}",
    "a {}photo of the large {}",
    "a {}photo of a cool {}",
    "a {}photo of a small {}",
]


class UnNormalize(object):
    def __init__(self, 
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        unnormed_tensor = torch.zeros_like(tensor)
        for i, (t, m, s) in enumerate(zip(tensor, self.mean, self.std)):
            unnormed_tensor[i] = t.mul(s).add(m)
            # The normalize code -> t.sub_(m).div_(s)
        return unnormed_tensor

unnorm = UnNormalize()


def configure_metadata(metadata_root):
    metadata = mch()
    metadata.image_ids = ospj(metadata_root, 'image_ids.txt')
    metadata.image_ids_proxy = ospj(metadata_root, 'image_ids_proxy.txt')
    metadata.class_labels = ospj(metadata_root, 'class_labels.txt')
    return metadata


def get_image_ids(metadata, proxy=False):
    """
    image_ids.txt has the structure

    <path>
    path/to/image1.jpg
    path/to/image2.jpg
    path/to/image3.jpg
    ...
    """
    image_ids = []
    suffix = '_proxy' if proxy else ''
    with open(metadata['image_ids' + suffix]) as f:
        for line in f.readlines():
            image_ids.append(line.strip('\n'))
    return image_ids


def get_class_labels(metadata):
    """
    class_labels.txt has the structure

    <path>,<integer_class_label>
    path/to/image1.jpg,0
    path/to/image2.jpg,1
    path/to/image3.jpg,1
    ...
    """
    class_labels = {}
    with open(metadata.class_labels) as f:
        for line in f.readlines():
            image_id, class_label_string = line.strip('\n').split(',')
            class_labels[image_id] = int(class_label_string)
    return class_labels





class GaussianBlur(object):
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.0):
        self.p = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __repr__(self):
        return "{}(p={}, radius_min={}, radius_max={})".format(
            self.__class__.__name__, self.p, self.radius_min, self.radius_max
        )

    def __call__(self, img):
        if random.random() <= self.p:
            radius = random.uniform(self.radius_min, self.radius_max)
            return img.filter(ImageFilter.GaussianBlur(radius=radius))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __repr__(self):
        return "{}(p={})".format(self.__class__.__name__, self.p)

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img




