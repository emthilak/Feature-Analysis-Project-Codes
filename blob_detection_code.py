# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 14:31:01 2021

@author: Eshan

    
"""

#Packages
import cv2 as cv
import matplotlib.pyplot as plt

# Blob detection packages
from math import sqrt
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray


# BGSubbed Images
img_gfp_1_bgsub = cv.imread(
    r"C:\Users\Eshan\Desktop\ImportantStuff\PythonStuff\Results\SameIMGVaryingRadii_BGSub\GFP1_BGSubRadius_12dot5.jpg", -1)
img_gfp_2_bgsub = cv.imread(
    r"C:\Users\Eshan\Desktop\ImportantStuff\PythonStuff\Images\PostBGSub\GFP_2_BGSubRadius_12dot5.jpg", -1)
img_gfp_3_bgsub = cv.imread(
    r"C:\Users\Eshan\Desktop\ImportantStuff\PythonStuff\Images\PostBGSub\GFP_3_BGSubRadius_12dot5.jpg", -1)
img_tritc_1_bgsub = cv.imread(
    r"C:\Users\emt38\Desktop\Eshan\Working Images\TRITC_1_BGSubRadius_5.jpg", -1)
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

img = img_tritc_1_bgsub
# plt.imshow(img)

##########################################################################
# BLOB DETECTION

# LoG (Laplacian of Gaussian)
for i in range(4, 7, 1):
    blobs_log = blob_log(img, min_sigma = 1, max_sigma=2, num_sigma=60, threshold=i/100)
    
    # DoG (Difference of Gaussian)
    #blobs_dog = blob_dog(img, max_sigma=20, threshold=.1)
    
    # DoH (Determinant of Hessian)
    #blobs_doh = blob_doh(img, max_sigma=20, threshold=.005)
    
    # Compute (approximate) radii in the 3rd column.
    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)
    #blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)
    
    
    fig, ax = plt.subplots(1, 1, figsize=(36, 36))
    
    for blob in blobs_log:
        y, x, r = blob
        c = plt.Circle((x, y), r, color='yellow', linewidth=2, fill=False)
        ax.add_patch(c)
    
    ax.set_axis_off()
    
    #ax.set_title("Laplacian of Gaussian")
    
    ax.imshow(img)
    
    plt.tight_layout()
    
    plt.savefig("GFP_1_R_5_LoG_MaxSig_" + str(2) + "_NumSig_" + str(60) + "_Thresh_0dot0" + str(i) + "_Overlap_" + str("NA") + "_MinSig_" + str(1) + ".jpg", bbox_inches='tight', pad_inches=0)
    
    plt.show()
    
    #cv.imwrite('GFP_1_Original.jpg', img)