# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 16:56:39 2021

@author: emt38
"""

#COLOCALIZATION CODE
##############################################

#Packages
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
# Blob detection packages
from math import sqrt, dist
from skimage import segmentation, morphology, color, measure, restoration, img_as_ubyte, util
from skimage.feature import blob_log
from skimage.color import rgb2gray
import os
import pandas as pd
from scipy import ndimage as ndi
from skimage.measure import label, regionprops, regionprops_table
from skimage.color import label2rgb
#import fnmatch
#from cellpose import utils, io
#from timeit import default_timer as timer
import mahotas
import matplotlib.patches as mpatches
from scipy import ndimage
import matplotlib.lines as lines



img_GFP = cv.imread(r"C:\Users\emt38\Desktop\Eshan\Working Images\GFP1.tif", -1)
img_TRITC = cv.imread(r"C:\Users\emt38\Desktop\Eshan\Working Images\TRITC1.tif", -1)

def mask_generator(img):
    GFP_files_8bg_data= np.zeros((2048,2048))
    #rolling ball
    bg = restoration.rolling_ball(img, radius=200)
    img_bg = img - bg
    GFP_files_8bg_data[:,:]= 255*img_bg/65535 #conversion of 16 bit to 8 bit float by dividing with 257(255/65537) after backgroudn subtraction

    #LoG (Laplacian of Gaussian) for detecting puncta with updated radius
    blobs_8bit= [[0]]*len(GFP_files_8bg_data) #defining a null list
    data= GFP_files_8bg_data
    blobs_8bit = blob_log(data, min_sigma = 1, max_sigma=2, num_sigma=60, threshold=0.028) #change this.
    blobs_8bit[:,2]= blobs_8bit[:,2]* sqrt(2)
    
    #OpenCV cv.circle to draw puncta in a binary image
    puncta_mask_8bit = np.zeros((2048, 2048))
    img_circ = np.zeros((2048,2048))
    for blob in blobs_8bit:
        y, x, r = blob
        y = int(y)
        x = int(x)
        r = int(r)
        img_circ = cv.circle(img_circ, (x, y), r, color = (255, 0, 255), thickness = -1)
    distance = ndi.distance_transform_edt(img_circ)
    local_max = morphology.local_maxima(distance)
    max_coords = np.nonzero(local_max)
    markers = ndi.label(local_max)[0]
    puncta_mask_8bit[:,:] = (segmentation.watershed(img_circ, markers, mask = img_circ)).astype(int)
    return(puncta_mask_8bit)
    
GFP_mask = mask_generator(img_GFP)
TRITC_mask = mask_generator(img_TRITC)

intsctn_raw = cv.bitwise_and(GFP_mask, TRITC_mask)
# for i in range(len(intsctn)):
#     for j in range(len(intsctn[i])):
#         if intsctn[i][j] != 0:
#             intsctn[i][j] = 1

#Generating a mask for the intersection
intsctn_mask = mask_generator(intsctn_raw)


#Collecting centroids of intersection
cntds = []
for region in regionprops(intsctn_mask.astype(int)):
    cntds.append(region.centroid)

#Collecting mask ids where centroids of intersection also exist
coloc_puncta = []
for i in range(len(cntds)):
    if TRITC_mask[int(cntds[i][0]), int(cntds[i][1])] > 0:
        if TRITC_mask[int(cntds[i][0]), int(cntds[i][1])] not in coloc_puncta:
            coloc_puncta.append(TRITC_mask[int(cntds[i][0]), int(cntds[i][1])])

#Boxing Colocalized Puncta on TRITC Image
fig, ax = plt.subplots(figsize=(32, 32))
ax.imshow(img_TRITC)
bboxes = []
for region in regionprops(TRITC_mask.astype(int)):
    if TRITC_mask[int(region.centroid[0]), int(region.centroid[1])] in coloc_puncta:
        bboxes.append(region.bbox)
        # draw rectangles on masked puncta
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle((region.bbox[1], region.bbox[0]), region.bbox[3] - region.bbox[1], region.bbox[2] - region.bbox[0],
                                  fill=False, edgecolor='orange', linewidth=1)
        ax.add_patch(rect)

ax.set_axis_off()
plt.tight_layout()
plt.savefig("BBoxed_ColocPuncta.jpg",bbox_inches='tight',pad_inches=0, dpi=600)
plt.show()

#Mask with colocalized puncta removed
for i in range(len(TRITC_mask)):
    for j in range(len(TRITC_mask)):
        if TRITC_mask[i,j] in coloc_puncta:
            TRITC_mask[i,j] += -TRITC_mask[i,j]
    
