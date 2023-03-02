import numpy as np
import time, os, sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import fnmatch
#%matplotlib inline
mpl.rcParams['figure.dpi'] = 300
from cellpose import utils, io

##additional packages needed
import pandas as pd
from skimage.measure import label, regionprops, regionprops_table
from skimage.color import label2rgb
from get_pns import assign_pointnames_all
from nd2toimage import imgfromnd2

#Directories, organized

#U2OS
# TIFFloc = r"D:/Eshan/U2OSs/ND2_CellMaskGeneration_Testing/"
# nd2raw = r'G:\My Drive\Keck Desktop (Eshan)\Eshan\Working Images\Raws (ND2)\U2OSs\05022022_post4hours001_postbaf.nd2'
# nd2masks = r'D:/Eshan/U2OSs/MasksFromND2s/'
# specmask = r'D:/Eshan/U2OSs/MasksFromND2s/Mask_'

#Huh7
TIFFloc = r'D:/Eshan/Huh7s/ND2_CellMaskGeneration_Testing/'
nd2raw = r'G:/My Drive/Keck Desktop (Eshan)/Eshan/Working Images/Raws (ND2)/Huh7s/02262022_postbaf.nd2'
nd2masks = r'D:/Eshan/Huh7s/MasksFromND2s/'
specmask = r'D:/Eshan/Huh7s/MasksFromND2s/Mask_'

no_imgs = 10 #max is 24 for first A549 nd2 file

#importing tiff files
files = []
#TIFF Location
input_dir = os.path.dirname(TIFFloc)
for f in os.listdir(input_dir):
    filename = os.path.join(input_dir, f)
    files.append(filename)

#deleting DAPI and TRITC files
GFP_files=[]
for file in files:
    if fnmatch.fnmatch(file, '*_GFPNSB*.tif'):
        GFP_files.append(file)


#RUN cellpose
# cd=r'D:\Eshan\Exp1_Verification/'
# os.chdir(cd)

from cellpose import models, io
import cv2 as cv

# DEFINE CELLPOSE MODEL
# model_type='cyto' or model_type='nuclei'
#model = models.Cellpose(gpu=False, model_type='cyto')
model = models.CellposeModel(gpu=True, model_type='cyto')

# define CHANNELS to run segementation on
# grayscale=0, R=1, G=2, B=3
# channels = [cytoplasm, nucleus]
# if NUCLEUS channel does not exist, set the second channel to 0
# channels = [0,0]
# IF ALL YOUR IMAGES ARE THE SAME TYPE, you can give a list with 2 elements
# channels = [0,0] # IF YOU HAVE GRAYSCALE
# channels = [2,3] # IF YOU HAVE G=cytoplasm and B=nucleus
# channels = [2,1] # IF YOU HAVE G=cytoplasm and R=nucleus

# or if you have different types of channels in each image
channels = [2,0]

#if diameter is set to None, the size of the cells is estimated on a per image basis
# you can set the average cell `diameter` in pixels yourself (recommended) 
# diameter can be a list or a single number for all images

# you can run all in a list e.g.
# >>> imgs = [io.imread(filename) in files]
# >>> masks, flows, styles, diams = model.eval(imgs, diameter=None, channels=channels)
# >>> io.masks_flows_to_seg(imgs, masks, flows, diams, files, channels)
# >>> io.save_to_png(imgs, masks, flows, files)

#%%
#defining a null array for number of cells. 
i=0
tifflist = []
tiffrawlist = []
# or in a loop
for filename in GFP_files:
    img = io.imread(filename)
    tiffrawlist.append(img)
    # plt.imshow(img)
    # plt.show()
    a = model.eval(img, diameter=120, channels=channels)
    #plt.imshow(masks)
    #plt.show()
    #cv.imwrite('Mask_' + str(i + 1) + '.tif', masks)
    tifflist.append(a[0])
    i= i+ 1
    if i == no_imgs:
        break

#%%
allwells = assign_pointnames_all(nd2raw)
nd2rawlist = []
for j in range(no_imgs):
    nd2rawlist.append(imgfromnd2(0, allwells[j], 0, nd2raw))

cd=nd2masks
os.chdir(cd)
nd2list = []

for i in range(no_imgs):
    nd2list.append(np.loadtxt(specmask + str(i+1)))

#WHAT IS THIS PLOTTING???
# for i in range(10):    
#     figure, axis = plt.subplots(1, 2)
    
#     axis[0].plot(tifflist[i])
#     axis[0].set_title('TIFF ' + str(i + 1))
    
#     axis[1].plot(nd2list[i])
#     axis[1].set_title('ND2 ' + str(i + 1))
    
#     plt.show()

#%%
#Side by side comparison of TIFF masks vs ND2 masks
for i in range(no_imgs):
    fig = plt.figure(figsize=(7, 7))
    
    r = 2
    c = 2
    
    fig.add_subplot(r, c, 1)
    
    plt.imshow(tifflist[i])
    plt.axis('off')
    plt.title('TIFF ' + str(i + 1))
    
    fig.add_subplot(r, c, 2)
    
    plt.imshow(nd2list[i])
    plt.axis('off')
    plt.title('ND2 ' + str(i + 1))
    
    fig.add_subplot(r, c, 3)
    
    plt.imshow(tiffrawlist[i])
    plt.axis('off')
    plt.title('TIFF Raw ' + str(i + 1))
    
    fig.add_subplot(r, c, 4)
    
    plt.imshow(nd2rawlist[i])
    plt.axis('off')
    plt.title('ND2 Raw ' + str(i + 1))
    
    plt.show()

#%%
a = []
cellcountdifflist = []
#Numerical difference between nd2 and tiff masks (no apparent difference)
for idx in range(no_imgs):
    img = nd2list[idx] - tifflist[idx]
    print(str(idx + 1) + ' Max: ' + str(np.max(img)))
    print(str(idx + 1) + ' Min: ' + str(np.min(img)))
    print(str(idx + 1) + ' Median: ' + str(np.median(img)))
    print(str(idx + 1) + ' Mean: ' + str(np.average(img)))
    print(str(idx + 1) + ' StDev: ' + str(np.std(img)))
    print(str(idx + 1) + ' Cell Count Difference: ' + (str(np.max(nd2list[idx]) - np.max(tifflist[idx]))))
    cellcountdifflist.append((np.max(nd2list[idx]) - np.max(tifflist[idx])))
    plt.imshow(nd2list[idx] - tifflist[idx])
    plt.axis('off')
    plt.title('Difference Array ' + str(idx + 1))
    plt.show()