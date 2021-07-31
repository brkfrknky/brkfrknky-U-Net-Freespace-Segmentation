# -*- coding: utf-8 -*-
"""
Config

@author: berke
"""

import os

# Path to jsons
JSON_DIR = '../data/jsons'


MODEL_DIR='../data/models'

# Path to mask
MASK_DIR  = '../data/masks'
if not os.path.exists(MASK_DIR):
    os.mkdir(MASK_DIR)

# Path to output images
IMAGE_OUT_DIR = '../data/masked_images'
if not os.path.exists(IMAGE_OUT_DIR):
    os.mkdir(IMAGE_OUT_DIR)

# Path to original images
IMAGE_DIR = '../data/images'


# In order to visualize masked-image(s), change "False" with "True"
VISUALIZE = True

# Bacth size
BACTH_SIZE = 2

# Input dimension
HEIGHT = 224
WIDTH = 224

# Number of class, for this task it is 2: Non-drivable area and Driviable area
N_CLASS= 2

AUG_IMAGE='../data/augmentation'

AUG_MASK='../data/augmentation_mask'

TEST_MASK_DIR='../data/test_mask'

TEST_IMAGE_DIR='../data/test_img'

TEST_JSON='../data/test_json'