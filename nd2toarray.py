# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 16:44:11 2022

@author: emt38
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import scipy.optimize as so
import scipy.integrate as integrate

import nd2reader
a = []
images = nd2reader.ND2Reader(r'G:\My Drive\Keck Desktop (Eshan)\Eshan\Working Images\Raws (ND2)\07302021_postbaf_postwort_10mins.nd2')
b = images.metadata
fovs = len(b['fields_of_view'])
tps = b['num_frames']
channel_no = len(b['channels'])

#General
# images.bundle_axes = 'xy'
# for k in range(channel_no):
#     images.default_coords['c'] = k
#     for i in range(fovs):
#         images.default_coords['v'] = i
#         images.iter_axes = 't'
#         for j in range(tps):
#             a.append(images[j])




#GFP ONLY
images.bundle_axes = 'xy'
images.default_coords['c'] = 1
for i in range(fovs):
    images.default_coords['v'] = i
    images.iter_axes = 't'
    for j in range(tps):
        a.append(images[j])