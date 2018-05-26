#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 20:40:03 2018

@author: priyank
"""

from keras.models import load_model,Model
from preprocess.normalize import preprocess_signature
from scipy.misc import imread
import numpy as np
from keras import backend as K

canvas_size = (952, 1360)  # Maximum signature size
original = imread('/home/priyank/Desktop/project_priyank/my_project/signatures/a1.png',flatten=1) #original.shape
processed = preprocess_signature(original, canvas_size)#processed.shape
#processed= np.expand_dims(processed, axis=0)              
processed=processed[np.newaxis, np.newaxis]
model=load_model('my_model.h5')
# with a Sequential model
get_dense_layer_output = K.function([model.layers[0].input],[model.layers[14].output])#model.layers[0].input.shape
layer_output = get_dense_layer_output([processed])[0]#layer_output.shape
#layer_output)


