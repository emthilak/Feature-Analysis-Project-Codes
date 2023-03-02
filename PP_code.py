# # -*- coding: utf-8 -*-
# """
# Created on Sun Sep  4 12:21:43 2022

# @author: Eshan
# """

#Packages
import cv2 as cv
from skimage import restoration
#import fnmatch
#from cellpose import utils, io
from timeit import default_timer as timer
#from skan.pre import threshold
#from skan import draw, skeleton_to_csgraph, Skeleton, summarize
from multiprocessing import Pool, Queue
import psutil


#Packages
import os
import fnmatch

#Packages
import matplotlib.pyplot as plt
import numpy as np
# Blob detection packages
from math import sqrt, dist
from skimage import segmentation, morphology, color, measure, img_as_ubyte, util
from skimage.feature import blob_log
from skimage.color import rgb2gray
import pandas as pd
from scipy import ndimage as ndi
from skimage.measure import label, regionprops, regionprops_table
from skimage.color import label2rgb
#import fnmatch
#from cellpose import utils, io
import mahotas
import matplotlib.patches as mpatches
from skan.pre import threshold
from skan import draw, skeleton_to_csgraph, Skeleton, summarize
import matplotlib.lines as lines
from collections import defaultdict
import imageio as io
cd=r'G:\.shortcut-targets-by-id\1dnE4QtTSMUOYzZ30lhKgQ04v0npPlpRR\singlecell-analysis\Nitin\celltrackers\codes/'
os.chdir(cd)
from channel_extract import channel_extract


cd=r'G:\My Drive\Keck Desktop (Eshan)\Eshan\Code_Files/'
os.chdir(cd)
#LARGE BATCH CODE
ino = 20 # 0<ino<48 (for current file), no. of total images run at a time (half GFP, half TRITC if even number)
inc = 20 # ino%inc == integer
img_st_index = 0
img_end_index = 26 #max 48 (for current file)


print('Loading Images')
GFPs, TRITCs = channel_extract(r'G:\My Drive\Keck Desktop (Eshan)\Eshan\Working Images\AllGFPs_2021_07_22/', '*GFPNSB*.tif', '*TRITC_NSB*.tif')
img_pairs = []
#Reading in desired images with cv.imread from same tps in each channel (red/green), grouping into a tuple,
#and putting tuple in list
for i in range(img_st_index, img_end_index):
    img_pairs.append((cv.imread(GFPs[i], -1), cv.imread(TRITCs[i], -1)))
print('Finished Loading Images')

# #Loading saved masks
print('Before Masks')
masks = []
for i in range(img_st_index, img_end_index):
    masks.append((np.loadtxt(r'G:\My Drive\Keck Desktop (Eshan)\Eshan\Working Images\LargeBatch_MaskFiles\GFP_mask_LB_' + str(i) + '.txt'), np.loadtxt(r'G:\My Drive\Keck Desktop (Eshan)\Eshan\Working Images\LargeBatch_MaskFiles\TRITC_mask_LB_' + str(i) + '.txt')))                                                                                                                                                                                                      
print('Masks Loaded')

#Temporary, to test max no. of imgs in p.starmap()
img_pairs = img_pairs[0:4] + img_pairs[8:12] + img_pairs[24:26]
masks = masks[0:4] + masks[8:12] + masks[24:26]

##############################################################################################################################################################################
#Mask Generation and time for Mask Generation
# print('Beginning Mask Generation...')
# bgsub_st = timer()
# bgs = []
# for pairno in range(len(img_pairs)):
#     GFP, TRITC = img_pairs[pairno]
    
#     TRITC_bg = restoration.rolling_ball(TRITC, radius=200)
#     GFP_files_8bg_data= np.zeros((2048,2048)) 
    
#     GFP_bg = restoration.rolling_ball(GFP, radius=200)

#     GFP_bg = GFP - GFP_bg #Creating the background subtracted version of the input image
#     GFP_files_8bg_data[:,:]= 255*GFP_bg/65535 #conversion of 16 bit to 8 bit float by multiplying with (255/65535) after background subtraction
    
#     #radius_parameters= [75, 150, 200, 250] #mention the radius parameters you want to test
#     #for i in range(len(GFP_files)):
    
    
#     #LoG (Laplacian of Gaussian) for detecting puncta with updated radius
#     blobs_8bit= [[0]]*len(GFP_files_8bg_data) #defining a null list
#     blobs_8bit = blob_log(GFP_files_8bg_data, min_sigma = 1, max_sigma=2, num_sigma=60, threshold=0.028) #LoG "blob" (circle) determination.  Outputs xy position and radius/sqrt(2).
#     blobs_8bit[:,2] = blobs_8bit[:,2]*sqrt(2) #Converting rooted distance to distance
    
#     #OpenCV cv.circle to draw puncta in a binary image
#     GFP_puncta_mask_8bit = np.zeros((2048, 2048)) #Creating a zeros array to generate a mask
#     img_circ = np.zeros((2048,2048)) #Creating a zeros array of the same shape as the input image on which circles are drawn
#     for blob in blobs_8bit:
#         y, x, r = blob #extracting xy position and radius for ith puncta
#         y = int(y) #Converting to an integer so as to be useful as discrete pixel measurements
#         x = int(x) #Converting to an integer so as to be useful as discrete pixel measurements
#         r = int(r) #Converting to an integer so as to be useful as discrete pixel measurements
#         img_circ = cv.circle(img_circ, (x, y), r, color = (255, 0, 255), thickness = -1) #Using OpenCV to draw the LoG circle on img_circ
    
#     #NEED TO UNDERSTAND - Watershed
#     distance = ndi.distance_transform_edt(img_circ)
#     local_max = morphology.local_maxima(distance)
#     #max_coords = np.nonzero(local_max)
#     markers = ndi.label(local_max)[0]
#     GFP_puncta_mask_8bit[:,:] = (segmentation.watershed(img_circ, markers, mask = img_circ)).astype(int) #Using watershed to create the puncta mask
    
#     np.savetxt('GFP_mask_LB_' + str(pairno) + '.txt', GFP_puncta_mask_8bit)
    
#     TRITC_files_8bg_data= np.zeros((2048,2048)) 
    
#     TRITC_bg = restoration.rolling_ball(TRITC, radius=200)

#     TRITC_bg = TRITC - TRITC_bg #Creating the background subtracted version of the input image
#     TRITC_files_8bg_data[:,:]= 255*TRITC_bg/65535 #conversion of 16 bit to 8 bit float by multiplying with (255/65535) after background subtraction
    
#     #radius_parameters= [75, 150, 200, 250] #mention the radius parameters you want to test
#     #for i in range(len(GFP_files)):
    
    
#     #LoG (Laplacian of Gaussian) for detecting puncta with updated radius
#     blobs_8bit= [[0]]*len(TRITC_files_8bg_data) #defining a null list
#     blobs_8bit = blob_log(TRITC_files_8bg_data, min_sigma = 1, max_sigma=2, num_sigma=60, threshold=0.028) #LoG "blob" (circle) determination.  Outputs xy position and radius/sqrt(2).
#     blobs_8bit[:,2] = blobs_8bit[:,2]*sqrt(2) #Converting rooted distance to distance
    
#     #OpenCV cv.circle to draw puncta in a binary image
#     TRITC_puncta_mask_8bit = np.zeros((2048, 2048)) #Creating a zeros array to generate a mask
#     img_circ = np.zeros((2048,2048)) #Creating a zeros array of the same shape as the input image on which circles are drawn
#     for blob in blobs_8bit:
#         y, x, r = blob #extracting xy position and radius for ith puncta
#         y = int(y) #Converting to an integer so as to be useful as discrete pixel measurements
#         x = int(x) #Converting to an integer so as to be useful as discrete pixel measurements
#         r = int(r) #Converting to an integer so as to be useful as discrete pixel measurements
#         img_circ = cv.circle(img_circ, (x, y), r, color = (255, 0, 255), thickness = -1) #Using OpenCV to draw the LoG circle on img_circ
    
#     #NEED TO UNDERSTAND - Watershed
#     distance = ndi.distance_transform_edt(img_circ)
#     local_max = morphology.local_maxima(distance)
#     #max_coords = np.nonzero(local_max)
#     markers = ndi.label(local_max)[0]
#     TRITC_puncta_mask_8bit[:,:] = (segmentation.watershed(img_circ, markers, mask = img_circ)).astype(int) #Using watershed to create the puncta mask
    
    
    
#     np.savetxt('TRITC_mask_LB_' + str(pairno) + '.txt', TRITC_puncta_mask_8bit)
    
    
#     bgs.append((GFP_bg, TRITC_bg))
# bgsub_en = timer()

# bgsub_time = bgsub_en - bgsub_st
# print('Finished Mask Generation. Total time: ' + str(bgsub_time))
    
##############################################################################################################################################################################            
   
def dataPacker(imgpair, maskpair):    
    
    GFP_img, TRITC_img = imgpair
    GFP_mask, TRITC_mask = maskpair
    
    return([GFP_img, GFP_mask], [TRITC_img, TRITC_mask])


#SMALL BATCH CODE
#ino = 8
#Loading images from small batch
# print('Before imgs')
# img_names = []
# for i in range(4):
#     img_names.append(channel_extract(r'G:\My Drive\Keck Desktop (Eshan)\Eshan\Working Images\SmallBatch_PP_Testing\Replicate ' + str(i + 1) + '/', '*GFPNSB*.tif', '*TRITC_NSB*.tif'))
# img_pairs = []
# for j in range(4):
#     img_pairs.append([cv.imread(img_names[j][0][0], -1), cv.imread(img_names[j][1][0], -1)])
# print('Imgs loaded')

# #Loading saved backgrounds
# print('Before Bgs')
# bgs = []
# for i in range(4):
#     bgs.append((np.loadtxt((r'G:\My Drive\Keck Desktop (Eshan)\Eshan\Working Images\GFP_bg_' + str(i) + '.txt')), np.loadtxt(r'G:\My Drive\Keck Desktop (Eshan)\Eshan\Working Images\TRITC_bg_' + str(i) + '.txt')))                                                                                                                                                                                                      
# print('Bgs Loaded')




print('Starting Multiprocessing')
if __name__ == '__main__':
    cd=r'G:\My Drive\Keck Desktop (Eshan)\Eshan\Code_Files/'
    os.chdir(cd)
    from untitled1 import allFeats
    datalist = []
    timelist = []
    for i in range(len(img_pairs[0:ino])):
        Gpair, Tpair = dataPacker(img_pairs[i], masks[i])
        
        #THis subsnippet used in Method 2
        # Gpair.append(i)
        # Gpair.append(True)
        # Tpair.append(i)
        # Tpair.append(False)
        
        datalist.append(Gpair)
        datalist.append(Tpair)
   
    print('Data Loaded! Beginning Pool...') 
#%%   
    with Pool() as p:
        
        for j in range(0, 3):
            st = timer()
        
            #Method 0: test dataset (works!)
            # a = p.starmap(allFeats, datalist[0:4])
            # b = p.starmap(allFeats, datalist[4:8])
            
            # Method 1: for loop (memory fail)
            # dfs = []
            # print('Starting...')
            # for i in range(0, int(ino/4)):
            #     a = p.starmap(allFeats, datalist[4*i : 4*i + 4])
            #     dfs.append(a)
            #     print('Completed ' + str(i))
                
            # Method 2: for loop, using np.savetxt in allFeats to save dfs and lower memory usage (memory fail)
            # print('Starting...')
            # for i in range(0, int(ino/4)):
            #     p.starmap(allFeats, datalist[4*i : 4*i + 4])
            #     print('Completed ' + str(i))
            
            
            # mid1 = timer()
            # print('Done! Total time: ' + str(mid1-st))
            
            # Method 3: for loop, using np.savetxt in main script to save dfs and lower memory usage ()
            print('Starting Run ' + str(j + 1) + '...')
            for i in range(0, int(ino/inc)):
                a = p.starmap(allFeats, datalist[inc*i : inc*i + inc])
                for ind in range(len(a)):
                    # np.savetxt('df_' + str(i) + '_' + str(ind) + '.txt', a[ind], fmt = '%s')
                    a[ind].to_csv('df_' + str(i) + '_' + str(ind) + '.csv')
                del(a)
                print('Completed set ' + str(i + 1) + ' of ' + str(int(ino/inc)))
            
            # Method 4: for loop, w/o np.savetxt in main script
            # dfs = []
            # print('Starting...')
            # for i in range(0, int(ino/inc)):
            #     a = p.starmap(allFeats, datalist[inc*i : inc*i + inc])
            #     for ind in range(len(a)):
            #         dfs.append(a[ind])
            #     del(a)
            #     print('Completed set ' + str(i + 1) + ' of ' + str(int(ino/inc)))
            
            #Method 5: Variation on 0, but using np.savetxt to see if faster than saving into variables (no complication w/ for loop)
            # # np.savetxt('firstfourimgs.txt', p.starmap(allFeats, datalist[0:4]))
            # # np.savetxt('secondfourimgs.txt', p.starmap(allFeats, datalist[4:8]))
            
            #Method 6: For loop, w/o np.savetxt but uncomplicated as compared to method 4
            # dfs = []
            # print('Starting...')
            # for i in range(0, 2):
            #     dfs.append(p.starmap(allFeats, datalist[4*i: 4*i + 4]))
                
            #     print('Completed set ' + str(i + 1) + ' of ' + str(2) + ', current time: ' + str(timer() - st))
            
        
            mid1 = timer()
            timelist.append(mid1 - st)
            print('(' + str(j + 1) + ') Done! Total time: ' + str(mid1-st))
        
        p.close()
        p.join()
        
        