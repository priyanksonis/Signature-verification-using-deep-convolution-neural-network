#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 14:56:56 2018

@author: priyank
"""

import cv2
import numpy as np
from scipy import ndimage
from scipy.misc import imresize

# Functions to load the CNN model
##import signet
##from cnn_model import CNNModel

# Functions for plotting:
import matplotlib.pyplot as plt
#%matplotlib inline
plt.rcParams['image.cmap'] = 'Greys'
original = imread('c-001-01.jpg')
# normalize_image(img, size=(840, 1360)):
""" Centers an image in a pre-defined canvas size, and remove
    noise using OTSU's method.

    :param img: The image to be processed
    :param size: The desired canvas size
    :return: The normalized image
    """
size=(840, 1360)
img=original

max_r, max_c = size

    # 1) Crop the image before getting the center of mass

    # Apply a gaussian filter on the image to remove small components
    # Note: this is only used to define the limits to crop the image
blur_radius = 2
blurred_image = ndimage.gaussian_filter(img, blur_radius)

    # Binarize the image using OTSU's algorithm. This is used to find the center
    # of mass of the image, and find the threshold to remove background noise
threshold, binarized_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find the center of mass
r, c = np.where(binarized_image == 0)
r_center = int(r.mean() - r.min())
c_center = int(c.mean() - c.min())

    # Crop the image with a tight box
cropped = img[r.min(): r.max(), c.min(): c.max()]

    # 2) Center the image
img_r, img_c = cropped.shape

normalized_image = np.ones((max_r, max_c), dtype=np.uint8) * 255
r_start = max_r // 2 - r_center
c_start = max_c // 2 - c_center

    # Make sure the new image does not go off bounds
    # Case 1: image larger than required (height):  Crop.
    # Emit a warning since we don't want this for the signatures in the WD dataset (OK for feature wi)
if img_r > max_r:
        print ('Warning: cropping image. The signature should be smaller than the canvas size')
        r_start = 0
        difference = img_r - max_r
        crop_start = difference // 2
        cropped = cropped[crop_start:crop_start + max_r, :]
        img_r = max_r
else:
        extra_r = (r_start + img_r) - max_r
        # Case 2: centering exactly would require a larger image. relax the centering of the image
        if extra_r > 0:
            r_start -= extra_r
        if r_start < 0:
            r_start = 0

    # Case 2: image larger than required (width). Crop.
if img_c > max_c:
        print ('Warning: cropping image. The signature should be smaller than the canvas size')
        c_start = 0
        difference = img_c - max_c
        crop_start = difference // 2
        cropped = cropped[:, crop_start:crop_start + max_c]
        img_c = max_c
else:
        extra_c = (c_start + img_c) - max_c
        if extra_c > 0:
            c_start -= extra_c
        if c_start < 0:
            c_start = 0

    # Add the image to the blank canvas
normalized_image[r_start:r_start + img_r, c_start:c_start + img_c] = cropped

    # Remove noise - anything higher than the threshold. Note that the image is still grayscale
normalized_image[normalized_image > threshold] = 255

    #return normalized_image






f, ax = plt.subplots(5,1, figsize=(6,15))
ax[0].imshow(original, cmap='Greys_r')
ax[1].imshow(blurred_image)
ax[2].imshow(binarized_image)
ax[3].imshow(cropped)
ax[4].imshow(normalized_image)


