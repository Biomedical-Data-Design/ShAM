#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 17:01:06 2022

@author: mikewang
"""
# import the necessary packages
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io, color
import matplotlib.pyplot as plt
import argparse

# load the image and convert it to a floating point data type
image = img_as_float(io.imread("./superpixel_eg.png"))
# image = img_as_float(io.imread("a9159b11-ba49-4885-902e-00c8d5095b98.png"))


# apply SLIC and extract (approximately) the supplied number
# of segments
segments = slic(image, n_segments = 100, sigma = 10, compactness = 0.1)

# show the output of SLIC
fig = plt.figure("Superpixels -- %d segments%")
ax = fig.add_subplot(1, 1, 1)
ax.imshow(mark_boundaries(image, segments))
plt.axis("off")

# show the plots
plt.show()

# transform to superpixel
superpixel = color.label2rgb(segments, image, kind='avg')

#%%

#%%
