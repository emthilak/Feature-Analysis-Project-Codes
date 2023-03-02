# -*- coding: utf-8 -*-
"""
Created on Sat Feb 14 11:17:26 2023

@author: emt38
"""

#Import packages
import nd2reader
from get_pns import assign_pointnames_all


def imgfromnd2(ch, well, tp, nd2filepath):
    #Gets image specified by channel, well name, and timepoint number using nd2 image filepath
    
    #Reading in nd2 file (nd2 object)
    ND2s = nd2reader.ND2Reader(nd2filepath)
    
    #Get wells in the order of nd2reader package
    if 'v' in ND2s.axes:
        allwells = assign_pointnames_all(nd2filepath)
    
    #Index image by channel, well, and timepoint
    if 'c' in ND2s.axes:
        for i in range(len(ND2s.metadata['channels'])):
            if ch == ND2s.metadata['channels'][i]:
                ND2s.default_coords['c'] = i #channel coords ranges from 1 (first channel) to n (nth channel)
                break
    if 'v' in ND2s.axes:
        ND2s.default_coords['v'] = allwells.index(well)
    else:
        print('POTENTIAL ERROR: one well detected')
    ND2s.iter_axes = 't'
    image = ND2s[tp] #tp ranges from 0 (first timepoint) to m-1 (mth timepoint)
    
    return(image)