#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 01:56:38 2018

@author: emblab
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 13:41:45 2018

@author: priyank
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.models import load_model,Model
from scipy.misc import imread
#from preprocess.normalize import normalize_image, resize_image, crop_center, preprocess_signature
import numpy as np
import sys
import os
import scipy.io
from keras import backend as K
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from scipy.io import loadmat



#base_model=load_model('/home/priyank/Desktop/project_priyank/my_project/transfer learning/resnet50_0_1_0.h5')
#model = Model(inputs=base_model.input, outputs=base_model.get_layer('flatten').output)

#base_model=load_model('/home/ee/mtech/eet162639/majorproject/vgg16_0_60.h5')
#base_model.summary()
#
#model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)
#model.summary()

PATH='/scratch/ee/mtech/eet162639/majorproject/dataset1000_processed_toy/exploitation/'
save_PATH='/scratch/ee/mtech/eet162639/majorproject/dataset1000_processed_features_GAN_discriminator_toy/exploitation'

try:
    if not os.path.exists(save_PATH):
           os.makedirs(save_PATH)
except OSError:
    print ('Error: Creating directory of data')



users= os.listdir(PATH)



model4=load_model('discriminator_only_model.h5')
model4.summary()

layer_name = 'flatten_1'
intermediate_layer_model = Model(inputs=model4.input,outputs=model4.get_layer(layer_name).output)
#intermediate_output = intermediate_layer_model.predict(x)








for i in users:
     real_fv=[]
     forg_fv=[]
     images=os.listdir(PATH+i)
     for j in images:
            if 'cf' not in j:
               img_path = PATH+i+'/'+j
               #img = image.load_img(img_path, target_size=(224, 224))
               #x = image.img_to_array(img)
               #x = np.expand_dims(x, axis=0)
               #x = preprocess_input(x)
               #flatten_features = model.predict(x)
               #real_fv.append(flatten_features)#len(real_fv)  real_fv[0]
               
               
               
               
               
               img1 = image.load_img(img_path,grayscale=True,target_size=(152,220))  
               x = image.img_to_array(img1)
               x = np.expand_dims(x, axis=0)
               intermediate_output = intermediate_layer_model.predict(x)
               real_fv.append(intermediate_output)#len(real_fv)  real_fv[0]
               
               
               
            else:
#               img_path = PATH+i+'/'+j
#               img = image.load_img(img_path, target_size=(224, 224))
#               x = image.img_to_array(img)
#               x = np.expand_dims(x, axis=0)
#               x = preprocess_input(x)
#               flatten_features = model.predict(x)
               img_path = PATH+i+'/'+j 
               img1 = image.load_img(img_path,grayscale=True,target_size=(152,220))  
               x = image.img_to_array(img1)
               x = np.expand_dims(x, axis=0)
               intermediate_output = intermediate_layer_model.predict(x)
               forg_fv.append(intermediate_output)#len(real_fv)  real_fv[0]
     real_fv=np.concatenate(real_fv,axis=0)#real_fv.shape
     save_filename = os.path.join(save_PATH,'real_'+i+'.mat')
     scipy.io.savemat(save_filename,{'features':real_fv})   
      
     forg_fv=np.concatenate(forg_fv,axis=0)#real_fv.shape
     save_filename = os.path.join(save_PATH,'forg_'+i+'.mat')
     scipy.io.savemat(save_filename,{'features':forg_fv})   
      
 



#input_img=loadmat('/media/priyank/6442175942172F741/dataset_toy_processed1_features/forg_011.mat')['features']
