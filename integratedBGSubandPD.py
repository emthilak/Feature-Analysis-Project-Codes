# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 14:25:40 2021

@author: Eshan
------------------------------------------------
Integrated BGSub and PD code

"""

#Packages
import numpy as np
import cv2 as cv
from skimage import restoration
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw
from tqdm import tqdm
import time

# Blob detection packages
from math import sqrt
from skimage import data
from skimage.feature import blob_log
from skimage.color import rgb2gray

#Regionprops Packages
from skimage.morphology import (erosion, dilation, closing, opening,
                                area_closing, area_opening)
import matplotlib.patches as mpatches
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import closing, square
from skimage.color import label2rgb
from skimage.filters import threshold_otsu, threshold_local
from skimage.draw import disk
 
#Importing and displaying the image, only works for flags = -1
#NOTE: For flags = 1 (color imaging), returns a black background,
#   but is the only value for which the color splitting code
#   does not return an error
img1 = cv.imread(r"C:\Users\emt38\Desktop\Eshan\Working Images\GFP1.tif",-1)
img2 = cv.imread(r"C:\Users\Eshan\Desktop\ImportantStuff\PythonStuff\Images\GFPs\GFP2.tif",-1)
img3 = cv.imread(r"C:\Users\Eshan\Desktop\ImportantStuff\PythonStuff\Images\GFPs\GFP3.tif",-1)
img4 = cv.imread(r"C:\Users\Eshan\Desktop\ImportantStuff\PythonStuff\Images\GFPs\GFP4.png",-1)
img_tritc1 = cv.imread(r"C:\Users\Eshan\Desktop\ImportantStuff\PythonStuff\Images\NonGFP\TRITC1.tif",-1)
img_tritc2 = cv.imread(r"C:\Users\Eshan\Desktop\ImportantStuff\PythonStuff\Images\NonGFP\TRITC2.tif",-1)
img_tritc3 = cv.imread(r"C:\Users\Eshan\Desktop\ImportantStuff\PythonStuff\Images\NonGFP\TRITC3.tif",-1)

img_1_full_RGB = cv.imread(r"C:\Users\Eshan\Desktop\ImportantStuff\PythonStuff\Images\NonGFP\img_1_Full_RGB.tif",-1)
img_1_full = cv.imread(r"C:\Users\Eshan\Desktop\ImportantStuff\PythonStuff\Images\NonGFP\img_1_Full.tif",-1)

img_gfp_BGSub_5 = cv.imread(
    r"C:\Users\emt38\Desktop\Eshan\Working Images\GFP_1_BGSubRadius_5.jpg", -1)

#Different Test Images for varying puncta conds
#First Isolated Raws
#1st
img_7302021_B02_1_T2_GFP = cv.imread(r"C:\Users\emt38\Desktop\Eshan\Working Images\Isolated Raws (Control Images)\Image 1\07302021_T2_B02-1_GFPNSB.tif", -1)
img_7302021_B02_1_T2_TRITC = cv.imread(r"C:\Users\emt38\Desktop\Eshan\Working Images\Isolated Raws (Control Images)\Image 1\07302021_T2_B02-1_TRITC_NSB.tif", -1)

#2nd
img_7302021_D02_4_T2_GFP = cv.imread(r"C:\Users\emt38\Desktop\Eshan\Working Images\Isolated Raws (Control Images)\Image 2\07302021_T2_D02-4_GFPNSB.tif", -1)
img_7302021_D02_4_T2_TRITC = cv.imread(r"C:\Users\emt38\Desktop\Eshan\Working Images\Isolated Raws (Control Images)\Image 2\07302021_T2_D02-4_TRITC_NSB.tif", -1)

#3rd
img_8022021_B02_1_T2_GFP = cv.imread(r"C:\Users\emt38\Desktop\Eshan\Working Images\Isolated Raws (Control Images)\Image 3\08022021_T2_B02-1_GFPNSB.tif", -1)
img_8022021_B02_1_T2_TRITC = cv.imread(r"C:\Users\emt38\Desktop\Eshan\Working Images\Isolated Raws (Control Images)\Image 3\08022021_T2_B02-1_TRITC_NSB.tif", -1)

#4th
img_8022021_D02_4_T2_GFP = cv.imread(r"C:\Users\emt38\Desktop\Eshan\Working Images\Isolated Raws (Control Images)\Image 4\08022021_T2_D02-4_GFPNSB.tif", -1)
img_8022021_D02_4_T2_TRITC = cv.imread(r"C:\Users\emt38\Desktop\Eshan\Working Images\Isolated Raws (Control Images)\Image 4\08022021_T2_D02-4_TRITC_NSB.tif", -1)

#Next, PD on Isolated Raws PostBGSub
#1st
Image1GFP = cv.imread(r"C:\Users\emt38\Desktop\Eshan\Working Images\PostBGSub (Control Images)\Image 1\img_7302021_B02_1_T2_GFP_BGSub_AKA_Image1GFP.jpg", -1)
Image1TRITC = cv.imread(r"C:\Users\emt38\Desktop\Eshan\Working Images\PostBGSub (Control Images)\Image 1\img_7302021_B02_1_T2_TRITC_BGSub_AKA_Image1TRITC.jpg", -1)

#2nd
Image2GFP = cv.imread(r"C:\Users\emt38\Desktop\Eshan\Working Images\PostBGSub (Control Images)\Image 2\img_7302021_D02_4_T2_GFP_BGSub_AKA_Image2GFP.jpg", -1)
Image2TRITC = cv.imread(r"C:\Users\emt38\Desktop\Eshan\Working Images\PostBGSub (Control Images)\Image 2\img_7302021_D02_4_T2_TRITC_BGSub_AKA_Image2TRITC.jpg", -1)

#3rd
Image3GFP = cv.imread(r"C:\Users\emt38\Desktop\Eshan\Working Images\PostBGSub (Control Images)\Image 3\img_8022021_B02_1_T2_GFP_BGSub_AKA_Image3GFP.jpg", -1)
Image3TRITC = cv.imread(r"C:\Users\emt38\Desktop\Eshan\Working Images\PostBGSub (Control Images)\Image 3\img_8022021_B02_1_T2_TRITC_BGSub_AKA_Image3TRITC.jpg", -1)

#4th
Image4GFP = cv.imread(r"C:\Users\emt38\Desktop\Eshan\Working Images\PostBGSub (Control Images)\Image 4\img_8022021_D02_4_T2_GFP_BGSub_AKA_Image4GFP.jpg", -1)
Image4TRITC = cv.imread(r"C:\Users\emt38\Desktop\Eshan\Working Images\PostBGSub (Control Images)\Image 4\img_8022021_D02_4_T2_TRITC_BGSub_AKA_Image4TRITC.jpg", -1)



img = Image1TRITC
##########################################################################
# BGSUB
'''
#Make Image Grayscale
#Conditional for images that are either mono or multi-channel
if len(img.shape) != 2:
    grayimg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    plt.imshow(grayimg)
else:
    grayimg = img

#DENOISING
kernel = np.ones((3,3),np.float32)/9
filt_2D = cv.filter2D(img, -1, kernel)

#EQUALIZATION
#NOTE: clipLimit increase makes cell contrast more noticeable
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl_img = clahe.apply(filt_2D)

#Skimage package rolling ball
bg = restoration.rolling_ball(cl_img, radius=5)
img_r = cl_img - bg
#cv.imwrite('img_8022021_D02_4_T2_TRITC_BGSub_AKA_Image4TRITC.jpg', img_r)
'''

##########################################################################
# BLOB DETECTION

ratio = np.amax(img) / 256 ;      
img8 = (img/ ratio).astype('uint8')
'''
#Reformatting image from 16-bit to 8-bit
info = np.iinfo(img.dtype) # Get the information of the incoming image type
stuff = img.astype(np.float64) / info.max # normalize the data to 0 - 1
stuff *= 255 # Now scale by 255
img_ready = stuff.astype(np.uint8)
'''
img_grey = rgb2gray(img8)

# LoG (Laplacian of Gaussian)
blobs_log = blob_log(img, min_sigma = 1, max_sigma=2, num_sigma=60, threshold=0.06)

# Compute (approximate) radii in the 3rd column.
blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)


fig, ax = plt.subplots(1, 1, figsize=(36, 36))


#img_circles = np.zeros((4096,4096), dtype=np.uint8)

#blackbg = np.zeros(img_grey.shape)
#cv.circle(blackbg, (500,500), 100, (0,0,255), -1)
#for i in tqdm(range (101), desc = "Loading...", ascii = False, ncols = 75):
 
for blob in blobs_log:
    y, x, r = blob
    #rr, cc = disk((y, x), r)
    #img_circles[rr,cc] = 255
    # Using cv2.circle() method
    # Draw a circle with blue line borders of thickness of 2 px
    #cv.circle(blackbg, (round(x), round(y)), round(r), (0,0,255), -1)
    c = plt.Circle((x, y), r, color='yellow', linewidth=2, fill=False)
    ax.add_patch(c)
ax.set_axis_off()

#ax.set_title("Laplacian of Gaussian")

ax.imshow(img)    

plt.tight_layout()

plt.savefig("RescalingTestImage2_NSB.jpg", bbox_inches='tight', pad_inches=0)

plt.show()

#newimg = img_circles[0:2048,0:2048]
#plt.imshow(newimg)
#cv.imwrite('OpenCVCirclesTest.jpg', image)

#cv.imwrite('GFP_1_Original.jpg', img)


##########################################################################
# REGIONPROPS INDIVIDUAL IMAGE PUNCTA DETECTION
'''
# label image regions
label_image = label(forlabel, connectivity=2)


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
'''
############################################################################

