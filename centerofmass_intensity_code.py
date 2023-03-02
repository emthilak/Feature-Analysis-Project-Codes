# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 17:07:54 2022

@author: emt38
"""

#Scipy Documentation
#https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.measurements.center_of_mass.html


#Importing useful packages
import mahotas
import numpy as np
from pylab import imshow, show
import cv2 as cv
from scipy import ndimage

#Importing grayscale img
img = cv.imread(r"C:\Users\emt38\Desktop\Eshan\Working Images\GFP1.tif",-1)

#adding gaussian filter
image = mahotas.gaussian_filter(img, 4)

#setting threshold
threshimg = (image > image.mean())

#making labeled image
labeledimg, n = mahotas.label(threshimg)

#center of mass calculation
COM = ndimage.measurements.center_of_mass(img, labels = labeledimg)

#Return Intensity at COM
COMInt = img[round(COM[0]),round(COM[1])]














