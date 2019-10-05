#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 21:37:55 2018

@author: priyank
"""


import numpy as np
np.random.seed(2017)

import cv2
import os
import time
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.layers import GlobalAveragePooling2D, Dense, Dropout,Activation,Flatten

from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Input
from keras.models import Model
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from keras.preprocessing.image import ImageDataGenerator



def generator(features,labels_dict,batch_size):
    batch_features=np.zeros(batch_size,224,224,3)
    batch_labels=np.zeros((batch_size,2))
    while True:
      for i in range(batch_size):
          index=np.random.choice(len(features),1)
          batch_features[i]=features[index[0]]
          batch_labels[i]=labels_dict[index[0]]
      yield batch_features,batch_labels   








# Loading the training data
PATH = os.getcwd()
# Define data path
#data_path ='/media/priyank/6442175942172F741/dataset_toy_processed1/development'
data_path ='/scratch/ee/mtech/eet162639/majorproject/datasetfinal_processed/development'
#data_path ='/media/priyank/6442175942172F741/dataset_toy_processed1/development'


data_dir_list = os.listdir(data_path)
data_dir_list=sorted(data_dir_list)




train_datagen = ImageDataGenerator() # randomly flipping half of the images horizontally
train_generator = train_datagen.flow_from_directory(
    data_path,batch_size = 200)








img_data_list=[]

a=[]
c_count=0

#k=
classes=len(data_dir_list)
for dataset in data_dir_list:
        #k=k+1
       	img_list=os.listdir(data_path+'/'+ dataset)
       	print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
       	for img in img_list:
           if 'cf-' not in img:
              c_count=c_count+1    
              print(img)    
              img_path = data_path + '/'+ dataset + '/'+ img
              img = image.load_img(img_path,target_size=(224, 224))  
              x = image.img_to_array(img)
              x = np.expand_dims(x, axis=0)
              x = preprocess_input(x)
              print('Input image shape:', x.shape)
              img_data_list.append(x)
        a.append(c_count)      
        #if k==1:
         #       break


#a= image.img_to_array(img)
#cv2.imwrite('new1.jpg', a)



img_data = np.array(img_data_list)
#img_data = img_data.astype('float32')
print (img_data.shape)
img_data=np.rollaxis(img_data,1,0)
print (img_data.shape)
img_data=img_data[0]
print (img_data.shape)


# Define the number of classes
#num_classes = 10

num_classes=classes
num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,),dtype='int64')


b=len(data_dir_list)

labels[0:a[0]]=0
for j in range(1,b):
     #print(j)   
    labels[a[j-1]:a[j]]=j    
       

# convert class labels to on-hot encoding
Y = np_utils.to_categorical(labels, num_classes)

#Shuffle the dataset
x,y = shuffle(img_data,Y, random_state=2)
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)


features={}
labels_dict={}

for i in range(X_train):
   features[i]=X_train[i]
   labels_dict[i]=y_train[i]
   
   




###########################################################################################################################
# Custom_resnet_model_1
#Training the classifier alone
image_input = Input(shape=(224, 224, 3))

model = ResNet50(input_tensor=image_input, include_top=True,weights='imagenet')
model.summary()
last_layer = model.get_layer('avg_pool').output
x= Flatten(name='flatten')(last_layer)
out = Dense(num_classes, activation='softmax', name='output_layer')(x)
custom_resnet_model = Model(inputs=image_input,outputs= out)
custom_resnet_model.summary()

for layer in custom_resnet_model.layers[:-1]:
	layer.trainable = False

custom_resnet_model.layers[-1].trainable

custom_resnet_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

t=time.time()
#hist = custom_resnet_model.fit(X_train, y_train, batch_size=32, epochs=1, verbose=1, validation_data=(X_test, y_test))
hist = custom_resnet_model.fit_generator(generator=generator(features,labels_dict,batch_size=1000),validation_data=(X_test, y_test),samples_per_epoch=10000,nb_epoch=10,use_multiprocessing=True,workers=6)

print('Training time: %s' % (t - time.time()))
(loss, accuracy) = custom_resnet_model.evaluate(X_test, y_test, batch_size=10, verbose=1)

print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))

custom_resnet_model.save('/home/ee/mtech/eet162639/majorproject/resnet50_0_1.h5')


###########################################################################################################################
###########################################################################################
import matplotlib
matplotlib.use('Agg')


import matplotlib.pyplot as plt
# visualizing losses and accuracy
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(1)

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
fig.savefig('/home/ee/mtech/eet162639/majorproject/resnet50_0_1_1.png')


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
fig1.savefig('/home/ee/mtech/eet162639/majorproject/resnet50_0_1_2.png')

