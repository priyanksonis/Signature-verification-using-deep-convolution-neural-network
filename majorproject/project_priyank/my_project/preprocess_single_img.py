import numpy as np

# Functions to load and pre-process the images:
from scipy.misc import imread, imsave
from preprocess.normalize import normalize_image, resize_image, crop_center, preprocess_signature

# Functions to load the CNN model
##import signet
##from cnn_model import CNNModel

# Functions for plotting:
import matplotlib.pyplot as plt
#%matplotlib inline
plt.rcParams['image.cmap'] = 'Greys'
original = imread('data/some_signature.png')

# Manually normalizing the image following the steps provided in the paper.
# These steps are also implemented in preprocess.normalize.preprocess_signature

normalized = 255 - normalize_image(original, size=(952, 1360))
#normalized = 255 - normalized    ##inverting
resized = resize_image(normalized, (170, 242))
cropped = crop_center(resized, (150,220))

# Visualizing the intermediate steps

f, ax = plt.subplots(4,1, figsize=(6,15))
ax[0].imshow(original, cmap='Greys_r')
ax[1].imshow(normalized)
ax[2].imshow(resized)
ax[3].imshow(cropped)

ax[0].set_title('Original')
ax[1].set_title('Background removed/centered')
ax[2].set_title('Resized')
ax[3].set_title('Cropped center of the image')


