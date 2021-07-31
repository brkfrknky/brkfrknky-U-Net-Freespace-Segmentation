# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 12:14:16 2021

@author: berke
"""

#from data_utils import EarlyStopping

import numpy as np
import tqdm
import torch
from preprocess import tensorize_image
import cv2
import os
import glob
from unet import UNet
import torch.nn.functional as F




######### PARAMETERS ##########
cuda =True
input_shape = (224, 224)#What size will the image resize
###############################


IMAGE_DIR = '../data/test_img'


image_path_list = glob.glob(os.path.join(IMAGE_DIR, '*'))
image_path_list.sort()





def predict(image_path_list):
     model = UNet(n_channels=3, n_classes=2, bilinear=True)
     model.load_state_dict(torch.load('../data/models/8000_model2.pth'))   
     if torch.cuda.is_available():
        model.cuda()
 
     for i in tqdm.tqdm(range(len(image_path_list))):
        batch_test = image_path_list[i:i+1]
        test_input = tensorize_image(batch_test, input_shape, cuda)
        outs = model(test_input)
        outs = F.upsample_bilinear(outs, size=(720, 1280))
        out=torch.argmax(outs,axis=1)
        out_cpu = out.cpu()
        outputs_list=out_cpu.detach().numpy()
        mask=np.squeeze(outputs_list,axis=0)
             
             
        img=cv2.imread(batch_test[0])
        mg=cv2.resize(img,(1280,720))
        mask_ind   = mask == 1
        cpy_img  = mg.copy()
        mg[mask==1 ,:] = (255, 0, 125)
        opac_image=(mg/2+cpy_img/2).astype(np.uint8)
        predict_name=batch_test[0]
        predict_path=predict_name.replace('test_img', 'predict_son')
        cv2.imwrite(predict_path,opac_image.astype(np.uint8))
 

if __name__ == "__main__":         
    

    predict(image_path_list)    
    