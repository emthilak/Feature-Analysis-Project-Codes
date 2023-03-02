# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 15:47:13 2022

@author: emt38
"""
#Mahotas Documentation
#https://mahotas.readthedocs.io/en/latest/api.html

#Haralick Features
#Basic Background + gogul.dev method
#https://gogul.dev/software/texture-recognition
#GeeksforGeeks Method
#https://www.geeksforgeeks.org/mahotas-haralick-features/

#Original Paper
#http://haralick.org/journals/TexturalFeatures.pdf

#The following method is derived from the GeeksforGeeks method

#Importing useful packages
import mahotas
import numpy as np
from pylab import imshow, show
import cv2 as cv
import time

#Starting timer
start_time = time.time()

#Importing grayscale img
img = cv.imread(r"C:\Users\emt38\Desktop\Eshan\Working Images\GFP1.tif",-1)

#adding gaussian filter
image = mahotas.gaussian_filter(img, 4)

#setting threshold
threshimg = (image > image.mean())

#making labeled image
labeledimg, n = mahotas.label(threshimg)

#Showing img
print("labeled image")
imshow(labeledimg)
show()

#getting haralick features
h_feature = mahotas.features.haralick(labeledimg, compute_14th_feature=True)

#showing the feature
#print("Haralick Features")
#imshow(h_feature)
#show()


























