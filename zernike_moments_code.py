# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 15:55:52 2022

@author: emt38
"""

#Methods
#https://www.geeksforgeeks.org/mahotas-zernike-moments/
#https://cvexplained.wordpress.com/2020/07/21/10-5-zernike-moments/

#Mahotas Documentation
#https://mahotas.readthedocs.io/en/latest/api.html

#Reading
#DOI:  10.1080/09500340.2011.554896

#Paper on using Zernike Moments to identify highly invasive cancer cells
#by shape https://academic.oup.com/ib/article/8/11/1183/5163473?login=true#eqn4



# importing required libraries
import mahotas
from pylab import gray, imshow, show
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from scipy import ndimage

import matplotlib.pyplot as plt

# loading image
img = cv.imread(r"C:\Users\emt38\Desktop\Eshan\Working Images\GFP1.tif",-1)

# showing image
imshow(img)
show()

# radius
radius = 500

#Computing COM, from COMIntensity code
#adding gaussian filter
image = mahotas.gaussian_filter(img, 4)

#setting threshold
threshimg = (image > image.mean())

#making labeled image
labeledimg, n = mahotas.label(threshimg)

#center of mass calculation
COM = ndimage.measurements.center_of_mass(img, labels = labeledimg)

# computing zernike moments
zm = mahotas.features.zernike_moments(img, radius, degree = 4, cm = COM)
'''
zernike_array = []
for i in range(0, 51):
    zernike_array.append(mahotas.features.zernike_moments(img, radius, degree = i, cm = COM))
    
#zernike_moments = mahotas.features.zernike_moments(img, radius, degree = 9, cm = COM)
for j in range(0, len(zernike_array)):
    plt.plot(j, zernike_array[j].shape, '.')
    
plt.show()
'''












































