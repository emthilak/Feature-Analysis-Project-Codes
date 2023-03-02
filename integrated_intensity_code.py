# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 15:29:49 2022

@author: emt38
"""

#Integrated Intensity Code

#Importing useful packages
import numpy as np
import cv2 as cv

#Loading image
img = cv.imread(r"C:\Users\emt38\Desktop\Eshan\Working Images\GFP1.tif",-1)

#Integrating the intensity over the whole numpy array
integ_intensity = np.sum(img)








