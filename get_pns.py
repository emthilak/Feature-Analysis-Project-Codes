# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 09:02:04 2023

@author: emt38
"""

import numpy as np
import nd2 as n
import pandas as pd
import nd2reader
    

def assign_pointnames_all(imagepath):
    #Gets well positions for wells relevant to the experiment in the order that the nd2reader package reads in images with the help of the nd2 package (ie utilize in conjuction with the nd2reader package, NOT with the nd2 package)
    #Extract all x and y posns with related point names
    img = nd2reader.ND2Reader(imagepath)
    pninrawdata = img.parser._raw_metadata.image_metadata[b'SLxExperiment'][b'ppNextLevelEx'][b''][b'uLoopPars'][b'Points'][b'']
    allposns = []
    allpns = []
    for idx in range(len(pninrawdata)):
        allpns.append(pninrawdata[idx][b'dPosName'].decode())
        allposns.append([pninrawdata[idx][b'dPosX'], pninrawdata[idx][b'dPosY']])
        
    #Extract x and y posns of relevant wells only
    actposlist = []
    with (n.ND2File(imagepath)) as f:
        for k in range(len(f.experiment[1].parameters.points)):
            actposlist.append(f.experiment[1].parameters.points[k].stagePositionUm[0:2])
                    
        f.close()
    
    #Assign by matching xy posns
    pns = []
    for i in range(len(actposlist)):
        for j in range(len(allposns)):
            if actposlist[i] == allposns[j]:
                pns.append(allpns[j])
                
    return(pns)
    











