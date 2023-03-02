# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 14:40:57 2021

@author: Eshan
"""

# Packages
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from skimage.morphology import (erosion, dilation, closing, opening,
                                area_closing, area_opening)
import matplotlib.patches as mpatches
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import closing, square
from skimage.color import label2rgb
from skimage.filters import threshold_otsu, threshold_local



# BGSubbed Images
img_gfp_1_bgsub = cv.imread(
    r"C:\Users\Eshan\Desktop\ImportantStuff\PythonStuff\Results\SameIMGVaryingRadii_BGSub\GFP1_BGSubRadius_12dot5.jpg", -1)
img_gfp_2_bgsub = cv.imread(
    r"C:\Users\Eshan\Desktop\ImportantStuff\PythonStuff\Images\PostBGSub\GFP_2_BGSubRadius_12dot5.jpg", -1)
img_gfp_3_bgsub = cv.imread(
    r"C:\Users\Eshan\Desktop\ImportantStuff\PythonStuff\Images\PostBGSub\GFP_3_BGSubRadius_12dot5.jpg", -1)
img_tritc_1_bgsub = cv.imread(
    r"C:\Users\Eshan\Desktop\ImportantStuff\PythonStuff\Images\PostBGSub\TRITC_1_BGSubRadius_12dot5.jpg", -1)
img_tritc_2_bgsub = cv.imread(
    r"C:\Users\Eshan\Desktop\ImportantStuff\PythonStuff\Images\PostBGSub\TRITC_2_BGSubRadius_12dot5.jpg", -1)
img_tritc_3_bgsub = cv.imread(
    r"C:\Users\Eshan\Desktop\ImportantStuff\PythonStuff\Images\PostBGSub\TRITC_3_BGSubRadius_12dot5.jpg", -1)

# Other Misc Images
#img_dapi = cv.imread(r"C:\Users\Eshan\Desktop\ImportantStuff\PythonStuff\Images\NonGFP\DAPI1.tif", -1)
img_gfp_1 = cv.imread(
    r"C:\Users\Eshan\Desktop\ImportantStuff\PythonStuff\Results\SameIMGVaryingRadii_BGSub\GFP1_BGSubRadius_12dot5.jpg", -1)
img_gfp_BGSub_5 = cv.imread(
    r"C:\Users\Eshan\Desktop\ImportantStuff\PythonStuff\Images\PostBGSub\GFP_1_BGSubRadius_5.jpg", -1)

img = img_gfp_BGSub_5
# plt.imshow(img)

##########################################################################
# REGIONPROPS INDIVIDUAL IMAGE PUNCTA DETECTION

#Skimage label image regions thresholding + counting method
# apply threshold
#blocksize = 1
thresh = threshold_otsu(img)
#bw = closing(img > 100, square(3))

#binary = img > thresh
ret, th1 = cv.threshold(img,35,255,cv.THRESH_BINARY)

th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
            cv.THRESH_BINARY,201,2)
th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY,21,2)
th4 = cv.threshold(img, thresh, 255, cv.THRESH_TRIANGLE)

#plt.imshow(th1)
#plt.imshow(th1)
#cv.imwrite("MeanAdaptiveThreshold_BS_11_BGSub.jpg",th2)


# label image regions
label_image = label(th1, connectivity=2)


# to make the background transparent, pass the value of `bg_label`,
# and leave `bg_color` as `None` and `kind` as `overlay`
#image_label_overlay = label2rgb(label_image, image=img, bg_label=0)

fig, ax = plt.subplots(1,1, figsize=(36, 36), dpi=300)
ax.imshow(img)

intensities = []
radius_sizes = []
maxarea = 6000
poslist = []
sizelist = []

rprop = regionprops(label_image, img)

for region in rprop:
    # take regions with large enough areas
    if region.area <= maxarea:
        #Various data counters
        #Note that r describes a circle of equivalent 
        #area to the rectangle that encloses the puncta
        r = np.sqrt(region.area/(np.pi))
        radius_sizes.append(r)
        #intensities.append(mean_intensity)
        poslist.append(region.centroid)
        
        # draw circle/rectangle around segmented coins
        minr, minc, maxr, maxc = region.bbox
        #circ = mpatches.Circle((minc, minr), radius=r,
                               #fill=False, edgecolor='red', linewidth=0.5)
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)
    
    
ax.set_axis_off()
ax.set_title('Max Area: ' + str(maxarea)
             + ' Count: ' + str(len(radius_sizes))
             ) #+ str(usedr))
plt.tight_layout()
#plt.savefig("GFP_1_Threshold_35.jpg",bbox_inches='tight',pad_inches=0)

plt.show()

############################################################################
# GRAPHING PUNCTA COUNT W/ VARYING ALLOWED MAX RADIUS
'''
#Skimage label image regions thresholding + counting method
# apply threshold
thresh = 80
bw = closing(img > thresh, square(3))

# remove artifacts connected to image border
#Try removing the above line
#cleared = clear_border(bw)

# label image regions
label_image = label(bw)


# to make the background transparent, pass the value of `bg_label`,
# and leave `bg_color` as `None` and `kind` as `overlay`
image_label_overlay = label2rgb(label_image, image=img, bg_label=0)

intensities = []
maxarea = 0
maxrad = []
AMAX = []
count = []
maxcount = []
rprop = regionprops(label_image)

while True:
    radius_sizes = []
    for region in rprop:
        # take regions with large enough areas
        if region.area <= maxarea:
            #Various data counters
            #Note that r describes a circle of equivalent 
            #area to the rectangle that encloses the puncta
            r = np.sqrt(region.area/(np.pi))
            radius_sizes.append(r)
            
            
    #maxrad.append(round(np.sqrt(maxarea/(np.pi)),3))
    AMAX.append(maxarea)
    count.append(len(radius_sizes))
    maxarea += 10
    
    if len(radius_sizes) >= 0.9*len(rprop):
        break
    
    

for i in range(0, len(count)):
    maxcount.append(len(rprop))
    

  
#Graphs
plt.xlabel("Max Area")
plt.ylabel("Count")

plt.plot(AMAX, count, 'ro', label = 'Data Points')
plt.plot(AMAX, count, 'b-', label = 'Data Curve')

plt.plot(AMAX, maxcount, label = 'Upper Bound')

plt.title("TRITC 3 Image Puncta Count vs. Max Area")
plt.legend(loc='lower right')
#plt.savefig("TRITC_3_CountvsArea_Graph.jpg", dpi=300, bbox_inches='tight')
plt.show()
'''