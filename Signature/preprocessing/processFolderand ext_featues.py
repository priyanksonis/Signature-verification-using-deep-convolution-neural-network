""" This example extract features for all signatures in a folder,
    ##/media/priyank/240035CC0035A5A8/toy_dataset/exploitation/
    ##/media/priyank/240035CC0035A5A8/toy_dataset_extracted_features/exploitation/
"""

from keras.models import load_model,Model
from scipy.misc import imread
from preprocess.normalize import preprocess_signature
import numpy as np
import sys
import os
import scipy.io
from keras import backend as K

if len(sys.argv) not in [4,6]:
    print('Usage: python process_folder.py <signatures_path> <save_path> '
          '<model_path> [canvas_size]')
    exit(1)

signatures_path = sys.argv[1]
save_path = sys.argv[2]
model_path = sys.argv[3]
if len(sys.argv) == 4:
    canvas_size = (952, 1360)  # Maximum signature size
else:
    canvas_size = (int(sys.argv[4]), int(sys.argv[5]))


signatures_path='/media/priyank/240035CC0035A5A8/toy_dataset/exploitation/'
save_path='/media/priyank/240035CC0035A5A8/toy_dataset_extracted_features/exploitation/'
model_path='models/my_model.h5'

print('Processing images from folder "%s" and saving to folder "%s"' % (signatures_path, save_path))
print('Using model %s' % model_path)
print('Using canvas size: %s' % (canvas_size,))

# Load the model
##model_weight_path = 'models/signet.pkl'
##model = CNNModel(signet, model_weight_path)


model_weight_path = 'models/my_model.h5'
model=load_model(model_weight_path)

##files = os.listdir(signatures_path)
user_folderes = os.listdir(signatures_path)



# Note: it there is a large number of signatures to process, it is faster to
# process them in batches (i.e. use "get_feature_vector_multiple")
a=0
for i in user_folderes:
    real_fv=[]
    forg_fv=[]
    a=a+1
    PATH=signatures_path+i+'/'
    files=os.listdir(PATH)
    for f in files:
        # Load and pre-process the signature
       if 'mat' not in f:
           if 'cf' not in f:
            filename = os.path.join(PATH, f)
            original = imread(filename, flatten=1)
            processed = preprocess_signature(original, canvas_size)
            # Use the CNN to extract features
            ##feature_vector = model.get_feature_vector(processed)
            processed=processed[np.newaxis, np.newaxis]
            get_dense_layer_output = K.function([model.layers[0].input],[model.layers[14].output])#model.layers[0].input.shape
            feature_vector = get_dense_layer_output([processed])[0]#layer_output.shape
            real_fv.append(feature_vector)#len(real_fv)  real_fv[0]
            # Save in the matlab format
            ##save_filename = os.path.join(save_path, os.path.splitext(f)[0] + '.mat')
            ##scipy.io.savemat(save_filename, {'feature_vector':feature_vector})
           else:
            filename = os.path.join(PATH, f)
            original = imread(filename, flatten=1)
            processed = preprocess_signature(original, canvas_size)
            processed=processed[np.newaxis, np.newaxis]
            get_dense_layer_output = K.function([model.layers[0].input],[model.layers[14].output])#model.layers[0].input.shape
            feature_vector = get_dense_layer_output([processed])[0]#layer_output.shape
            forg_fv.append(feature_vector)  #len(forg_fv)
    break     
    real_fv=np.concatenate(real_fv,axis=0)#real_fv.shape
    save_filename = os.path.join(save_path,'real_'+i+ '.mat')
    scipy.io.savemat(save_filename,{'features':real_fv})   
    forg_fv=np.concatenate(forg_fv,axis=0)#real_fv.shape
    save_filename = os.path.join(save_path,'forg_'+i+ '.mat')
    scipy.io.savemat(save_filename,{'features':forg_fv})   
#save real_743.mat  24X2048
#save forg_743.mat  30x2048             
