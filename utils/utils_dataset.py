# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 17:37:47 2022

@author: marti
"""

import numpy as np
import random
from scipy import ndimage
from skimage.morphology import erosion, disk

# ISPRS standard color palette
palette = {0 : (255, 255, 255), # Impervious surfaces (white)
           1 : (0, 0, 255),     # Buildings (blue)
           2 : (0, 255, 255),   # Low vegetation (cyan)
           3 : (0, 255, 0),     # Trees (green)
           4 : (255, 255, 0),   # Cars (yellow)
           5 : (255, 0, 0),     # Clutter (red)
           6 : (0, 0, 0)}       # Undefined (black)

invert_palette = {v: k for k, v in palette.items()}

def convert_to_color(arr_2d, palette=palette):
    """ Numeric labels to RGB-color encoding """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d

def convert_from_color(arr_3d, palette=invert_palette):
    """ RGB-color encoding to grayscale labels """
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d

def erode_gt(gt, kernel):
    gt1 = erosion(gt, kernel)
    gt2 = np.where(gt-gt1 != 0, 6, gt)
    return gt2

def conn_comp(gt, kernel): # taking the GT already converted in 1 channel
    """ removal of connected components in GTs """
    """ change the number of remaining components to lower the percentage of labels"""
    new_gt = 6 * np.ones(gt.shape) # completely black image of the size of the original GT
    threshold = 0.5 # stating the threshold for binarized 0,1 images
    for i in np.unique(gt):    # finding connected components for each class
        bin_gt = np.zeros(gt.shape)
        bin_gt[np.where(gt==i)] = 1 # binarizing the image discarding any class non-currently considered
        labeled, nr_objects = ndimage.label(bin_gt > threshold) 

        #print("Number of objects for class " + str(i) + " is {}".format(nr_objects))
        
        if i == 4 or i == 0: # in this case, cars and streets
            num = np.argsort(np.unique(labeled, return_counts=True)[1])[:-1][::2] # reorder the elements of the array of connected components 
                                                                                  # (from smaller to bigger, discarding the background)
                                                                                  # take one element every two
        else:
            num = np.argsort(np.unique(labeled, return_counts=True)[1])[:-1][::3]  
            labeled = erosion(labeled, kernel)                                    # further erosion                                                     
                                                                         
        for n in num:
            new_gt[np.where(labeled==n)] = i  # give back the right class number to the saved connected components

    return new_gt