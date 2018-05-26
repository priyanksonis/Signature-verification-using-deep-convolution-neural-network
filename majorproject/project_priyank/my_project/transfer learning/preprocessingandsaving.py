#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 22:13:16 2018

@author: priyank
"""

import numpy as np
import os
import scipy.io
import cv2

# Functions to load and pre-process the images:
from scipy.misc import imread, imsave
from preprocess.normalize import normalize_image, resize_image, crop_center, preprocess_signature

#/media/priyank/240035CC0035A5A8/priyank_project_data
#/home/priyank/project_data/SignatureSyntheticGPDSOffLineOnline10000
signatures_path='/home/priyank/project_data/SignatureSyntheticGPDSOffLineOnline10000/'
save_path='/media/priyank/240035CC0035A5A8/SignatureSyntheticGPDS10000processed/'
user_folders = os.listdir(signatures_path)
a=0
b=0


##making folders of user number
list_of_user=[]
for individual in user_folders:
       list_of_user.append(individual)
       
for subfolder_name in list_of_user:
    os.makedirs(os.path.join(save_path, subfolder_name))       


#
for indi in user_folders:
        a=a+1
        PATH=signatures_path+indi
        signimage=os.listdir(PATH)
#        new_path=save_path+individual+'/'
#        os.makedirs(new_path)
                 
        for i in signimage:
               if 'mat' not in i:
                 PATH1=PATH+'/'+i
                 new_path=save_path+indi+'/'+i
                 img = imread(PATH1)
                 #preprocessing image
                 # Manually normalizing the image following the steps provided in the paper.
# These steps are also implemented in preprocess.normalize.preprocess_signature

                 normalized = 255 - normalize_image(img, size=(952, 1360))
                 #normalized = 255 - normalized    ##inverting
                 resized = resize_image(normalized, (170, 242))
                 cropped = crop_center(resized, (150,220))
                 cv2.imwrite(new_path,cropped)
                 #b=b+1
                 #if b==1:
                  #   break
                 

        #if a==2:
         #     break 
       
       
           