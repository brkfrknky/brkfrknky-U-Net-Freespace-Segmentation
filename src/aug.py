# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 00:58:26 2021

@author: berke
"""

import numpy as np
import cv2
import os
import tqdm 
from torchvision import transforms as T
from PIL import Image
from config import IMAGE_DIR , MASK_DIR




#The path to the masks folder is assigned to the variable
image_path=[] #empty list created
for name in os.listdir(IMAGE_DIR):
    image_path.append(os.path.join(IMAGE_DIR,name))
mask_path=[] #empty list created


for name in os.listdir(MASK_DIR):
    mask_path.append(os.path.join(MASK_DIR,name))

valid_size = 0.3
test_size  = 0.1
indices = np.random.permutation(len(image_path))
test_ind  = int(len(indices) * test_size)
valid_ind = int(test_ind + len(indices) * valid_size)
train_input_path_list = image_path[valid_ind:]#We got the elements of the image_path_list list from 1905 to the last element
train_label_path_list = mask_path[valid_ind:]#We got the elements of the mask_path_list list from 1905 to the last element

for image in tqdm.tqdm(train_input_path_list):
    img=Image.open(image)
    color_aug = T.ColorJitter(brightness=0.4, contrast=0.4, hue=0.06)

    img_aug = color_aug(img)
    new_path=image[:-4]+"-1"+".png"
    new_path=new_path.replace('images', 'augmentation')
    img_aug=np.array(img_aug)
    cv2.imwrite(new_path,img_aug)
    

    
for mask in tqdm.tqdm(train_label_path_list):
    msk=cv2.imread(mask)
    new_mask=msk
    newm_path=mask[:-4]+"-1"+".png"
    newm_path=newm_path.replace('masks', 'augmentation_mask')
    cv2.imwrite(newm_path,new_mask)
    
   
    
    
    

    
