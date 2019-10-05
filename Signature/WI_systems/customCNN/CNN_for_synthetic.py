#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 22:32:42 2018

@author: emblab
"""

import numpy as np
np.random.seed(100)
import cv2
import os
from keras.applications.resnet50 import ResNet50
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Activation, Flatten, Input
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model,Sequential
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from keras.layers.convolutional import Conv2D, MaxPooling2D,ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras import callbacks
import time


PATH='/scratch/ee/mtech/eet162639/majorproject/dataset1000_processed_toy/development'
user_folders=os.listdir(PATH)

row=150
col=220
num_channnel=1

num_class=len(user_folders)


img_list=[]
a=[]
count=0
for i in user_folders:
    image_list=os.listdir(PATH+'/'+i)
    
    for j in image_list:
        
        if 'cf' not in j:
           count=count+1
           input_image=cv2.imread(PATH+'/'+i+'/'+j)
           input_image1=cv2.cvtColor(input_image,cv2.COLOR_RGB2GRAY)
           img_list.append(input_image1)
    a.append(count)   
           
img_data=np.array(img_list)     
img_data=img_data.astype('float32')
img_data=np.expand_dims(img_data,axis=4)

no_samples=img_data.shape[0]

labels=np.ones(no_samples,dtype='int64')


labels[0:a[0]]=0
for i in range(1,len(a)):
    labels[a[i-1]:a[i]]=i


Y=np_utils.to_categorical(labels,num_class)  #Y.shape

x,y=shuffle(img_data,Y, random_state=2)


X_train_val, X_test ,y_train_val, y_test=  train_test_split(x,y, test_size=0.2, random_state=2)

X_train, X_val,y_train, y_val=train_test_split(X_train_val,y_train_val,test_size=.2,random_state=2)

input_shape=img_data[0].shape



model=Sequential()

model.add(Conv2D(96,(11,11),strides=4,input_shape=input_shape ))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=3, strides=2))


model.add(ZeroPadding2D(padding=(2,2)))
model.add(Conv2D(256,(5,5),padding='same'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=3, strides=2))

model.add(ZeroPadding2D(padding=(1, 1)))
model.add(Conv2D(384,(3,3), padding='valid'))
model.add(BatchNormalization())

model.add(ZeroPadding2D(padding=(1,1)))
model.add(Conv2D(384,(3,3),padding='same'))
model.add(BatchNormalization())

model.add(ZeroPadding2D(padding=(1,1)))
model.add(Conv2D(256,(3,3), padding='same'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=3, strides=2))

model.add(Flatten())

model.add(Dense(2048))
model.add(BatchNormalization())

model.add(Dense(2048))
model.add(BatchNormalization())

model.add(Dense(num_class))
model.add(Activation('softmax'))



model.summary()



model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])




t=time.time()
csv_logger = callbacks.CSVLogger('model_training.csv')
hist=model.fit(X_train, y_train,batch_size=15,epochs=2,verbose=2, validation_data=(X_val,y_val),callbacks=[csv_logger])


print('Training time: %s' % (t - time.time()))





print(hist.history.keys())

(loss, accuracy) = model.evaluate(X_test, y_test, batch_size=10, verbose=2)

print(loss)
print(accuracy)



print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))




model.save('cnn_model.h5')

import matplotlib
matplotlib.use('Agg')


import matplotlib.pyplot as plt
# visualizing losses and accuracy
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(2)

fig=plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])
fig.savefig('/home/ee/mtech/eet162639/majorproject/final/acc.png')


fig1=plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])
fig1.savefig('/home/ee/mtech/eet162639/majorproject/final/loss.png')











