#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  1 18:54:12 2018

@author: emblab
"""



import numpy as np
np.random.seed(2017)

import cv2
import os
import time
#from keras.applications.resnet50 import ResNet50
#from resnet50 import ResNet50
from keras.preprocessing import image
from keras.layers import GlobalAveragePooling2D, Dense, Dropout,Activation,Flatten

from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Input
from keras.models import Model
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
import cv2
import numpy as np



def train_data():
    # Loading the training data
    PATH = os.getcwd()
    # Define data path
    #data_path ='/media/priyank/6442175942172F741/dataset_toy_processed1/development'
    #data_path ='/home/emblab/Desktop/majorproject/dataset1000_processed_toy/development'
    data_path ='/scratch/ee/mtech/eet162639/majorproject/dataset1000_processed/development'
    
    data_dir_list = os.listdir(data_path)
    data_dir_list=sorted(data_dir_list)
    
    
    img_data_list=[]
    
    a=[]
    c_count=0
    
    #k=
    #b=0
    classes=len(data_dir_list)
    for dataset in data_dir_list:
            #k=k+1
           	img_list=os.listdir(data_path+'/'+ dataset)
           	print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
           	for img in img_list:
                 try:     
                  if 'cf-' not in img:                
                      #c_count=c_count+1    
                      print(img)    
                      img_path = data_path + '/'+ dataset + '/'+ img
                      img1 = image.load_img(img_path,grayscale=True,target_size=(152,220))  
                      
                      #img1=cv2.imread(img_path,0)
                      # arr = np.array(img1)
                      x = image.img_to_array(img1)
                      
                      #x=np.transpose(x,(1,0,2))
                      #x =x.reshape(x.shape[1],x.shape[0],1)
                      #x=np.rollaxis(x,1,0)
                      #cv2.imwrite('new15.jpeg', arr)
                      #b=b+1
                      x = np.expand_dims(x, axis=0)
                      #x = preprocess_input(x)
                      print('Input image shape:', x.shape)
                      img_data_list.append(x)
                  #a.append(c_count)
                 except:
                    pass
            #if k==1:
             #       break
    
    
    #a= image.img_to_array(img)
    #cv2.imwrite('new1.jpg', a)
    
    
    
    img_data = np.array(img_data_list)
    #img_data = img_data.astype('float32')
    
    
    
    
    img_data=np.rollaxis(img_data,1,0)
    print (img_data.shape)
    img_data=img_data[0]
    print (img_data.shape)
    return img_data


if __name__=='__main__':
    X=train_data()
    #cv2.imwrite('new15.jpeg', X[0])
    
