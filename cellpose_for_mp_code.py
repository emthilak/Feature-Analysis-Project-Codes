# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 17:30:48 2022

@author: emt38
"""

import numpy as np
import time, os, sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import fnmatch
#%matplotlib inline
mpl.rcParams['figure.dpi'] = 300
from cellpose import utils, io
import nd2reader
from timeit import default_timer as timer
from cellpose import models

##additional packages needed
import pandas as pd
from skimage.measure import label, regionprops, regionprops_table
from skimage.color import label2rgb

def cellpose_for_mp_3(images):
    #################################################################################################################################
    print('Starting...')
    st = timer()
        
    #Read in nd2s
    GFP_files = []
    images = nd2reader.ND2Reader(r'G:\My Drive\Keck Desktop (Eshan)\Eshan\Working Images\ND2toArrayProject\minus20 (1).nd2')
    b = images.metadata
    fovs = len(b['fields_of_view'])
    tps = b['num_frames']
    channel_no = np.where(np.array(b['channels']) == 'GFP_RG')[0][0]

    model = models.Cellpose(gpu=True, model_type='cyto')
    channels = [0,0]
    i=-1
    masklist = []

    images.bundle_axes = 'xy'
    images.default_coords['c'] = channel_no
    for i in range(int(fovs/14)):
        images.default_coords['v'] = i
        images.iter_axes = 't'
        for j in range(tps):
            img = images[j]
            masks, flows, styles, diams = model.eval(img, diameter=120, channels=channels)
            masklist.append(masks)
            i= i+ 1
            label_img= label(masks)
            image_label_overlay = label2rgb(masks, image=img, bg_label=0)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(image_label_overlay)

            img_prop= regionprops(masks,img,'All')
            properties =['label','area','filled_area','convex_area','bbox','bbox_area','coords','equivalent_diameter', 'euler_number','extent','feret_diameter_max',
                          'major_axis_length', 'perimeter','minor_axis_length','centroid',
                          'eccentricity','mean_intensity','max_intensity','min_intensity','perimeter','solidity']
            df = pd.DataFrame(regionprops_table(masks,intensity_image=img, properties = properties))
        
            dfl = pd.DataFrame(regionprops_table(label_img, img, properties = properties))

    #################################################################################################################################


    en = timer()
    rt = en - st
    print('Done! Total time: ' + str(rt))
    return(masklist)

def cellpose_for_mp_4(img):

    model = models.Cellpose(gpu=True, model_type='cyto')
    channels = [0,0]
    i=-1

    masks, flows, styles, diams = model.eval(img, diameter=120, channels=channels)
    i= i+ 1
    label_img= label(masks)
    image_label_overlay = label2rgb(masks, image=img, bg_label=0)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image_label_overlay)

    img_prop= regionprops(masks,img,'All')
    properties =['label','area','filled_area','convex_area','bbox','bbox_area','coords','equivalent_diameter', 'euler_number','extent','feret_diameter_max',
                  'major_axis_length', 'perimeter','minor_axis_length','centroid',
                  'eccentricity','mean_intensity','max_intensity','min_intensity','perimeter','solidity']
    df = pd.DataFrame(regionprops_table(masks,intensity_image=img, properties = properties))

    dfl = pd.DataFrame(regionprops_table(label_img, img, properties = properties))
    
    return(masks)

def cellpose_for_mp_4_masksonly(img):

    model = models.Cellpose(gpu=True, model_type='cyto')
    masks, flows, styles, diams = model.eval(img, diameter=120, channels=[0,0])
    
    return(masks)

def cellpose_for_mp_7(img):
    
    model = models.Cellpose(gpu=True, model_type='cyto')
    masks, flows, styles, diams = model.eval(img, diameter=120, channels=[0,0])
    
    return(masks)