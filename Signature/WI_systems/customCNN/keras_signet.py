#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 16:30:46 2018

@author: priyank
"""

import os
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from keras import backend as K
K.set_image_dim_ordering('th')
from keras.utils import np_utils
rom keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D ,ZeroPadding2D
from keras.optimizers import SGD,RMSprop,adam
from preprocess.normalize import preprocess_signature
#from scipy.misc import imread
from skimage.io import imread
canvas_size = (952, 1360)  # Maximum signature size
PATH='/media/priyank/240035CC0035A5A8/toy_dataset/development/'
#PATH='/home/ee/mtech/eet162639/toy_dataset/development/'

data_dir_list = os.listdir(PATH) #len(data_dir_list)
num_classes=len(data_dir_list)
img_data_list=[]
Y=[]
a=0
for dataset in data_dir_list:
 #a+=1
 img_list=os.listdir(PATH+'/'+ dataset)
 print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
 for img in img_list:
           if "mat" not in img:
             #print(img)  
             if 'cf' not in img:
               label=int(img.split('-')[1])   
               Y.append(label)
               original = imread(PATH+'/'+ dataset+'/'+ img,flatten=1) #original.shape
               processed = preprocess_signature(original, canvas_size)#processed.shape
               img_data_list.append(processed)#len(img_data_list)
             #img_data_list contains list of images of size (150x220)
 #if a==5:
  #break		    

#img_data_list[0].shape
X = np.array(img_data_list)#X.shape
X= X.astype('float32')
print (X.shape)
X= np.expand_dims(X, axis=1)#X.shape   len(Y) 
#converting Y to array
Y=np.array(Y) #Y.shape
#a is vector of unique numbers (sorted)
a=list(set(Y))#len(a)
a=np.array(a)#a.shape[0]
a=np.sort(a)

for i in range(Y.shape[0]):
    for j in range(a.shape[0]):
        if Y[i]==a[j]:
            Y[i]=j
# convert class labels to on-hot encoding
Y = np_utils.to_categorical(Y,num_classes)  #Y.shape Y[48]
#suffle dataset
x,y = shuffle(X,Y, random_state=2)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

# Defining the model
input_shape=X[0].shape #img_data.shape
print(input_shape)					
model = Sequential()
model.add(Conv2D(96,(11,11),subsample=(4, 4),input_shape=input_shape))#model.output_shape
model.add(MaxPooling2D(pool_size=(2, 2)))#model.output_shape
model.add(ZeroPadding2D(padding=(2, 2)))#model.output_shape
model.add(Conv2D(256, (5, 5)))#model.output_shape
model.add(MaxPooling2D(pool_size=(2, 2)))#model.output_shape
model.add(ZeroPadding2D(padding=(1, 1)))#model.output_shape
model.add(Conv2D(384, (3,3)))#model.output_shape
model.add(ZeroPadding2D(padding=(1, 1)))
model.add(Conv2D(384, (3,3)))#model.output_shape
model.add(ZeroPadding2D(padding=(1, 1)))#model.output_shape
model.add(Conv2D(256, (3,3)))#model.output_shape
model.add(MaxPooling2D(pool_size=(2, 2)))#model.output_shape
model.add(Flatten())#model.output_shape
model.add(Dense(2048))#model.output_shape
model.add(Dense(2048,name="dense_layer"))#model.output_shape    
model.add(Dense(num_classes))#model.output_shape
model.add(Activation('softmax'))#model.output_shape
model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=["accuracy"])
hist = model.fit(X_train, y_train, batch_size=16, nb_epoch=2, verbose=1,validation_data=(X_test, y_test))
model.save('my_model1.h5')    
    
