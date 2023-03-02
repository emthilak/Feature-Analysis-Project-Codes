# -*- coding: utf-8 -*-
"""
"""

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
from timeit import default_timer as timer
import mahotas
import matplotlib.patches as mpatches
#from skan.pre import threshold
#from skan import draw, skeleton_to_csgraph, Skeleton, summarize
import matplotlib.lines as lines
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
import umap

class Features:
    def __init__(self, inputdata):
        global bboxes, puncta_properties_dictionaries, distanceslist, radius, img
        
        start = timer()
        
        ###################################################################################################################
        #Mask Generation
        image_path, neighbors_r = inputdata
        
        img = cv.imread(image_path, -1)
        radius = neighbors_r

        GFP_files_8bg_data= np.zeros((2048,2048)) #Creating an empty array for background subtracted image
        #rolling ball
        #radius_parameters= [75, 150, 200, 250] #mention the radius parameters you want to test

        bg = restoration.rolling_ball(img, radius=200) #Determining the background of the input image
        img_bg = img - bg #Creating the background subtracted version of the input image
        GFP_files_8bg_data[:,:]= 255*img_bg/65535 #conversion of 16 bit to 8 bit float by multiplying with (255/65535) after background subtraction

        #radius_parameters= [75, 150, 200, 250] #mention the radius parameters you want to test
        #for i in range(len(GFP_files)):


        #LoG (Laplacian of Gaussian) for detecting puncta with updated radius
        blobs_8bit= [[0]]*len(GFP_files_8bg_data) #defining a null list
        blobs_8bit = blob_log(GFP_files_8bg_data, min_sigma = 1, max_sigma=2, num_sigma=60, threshold=0.028) #LoG "blob" (circle) determination.  Outputs xy position and radius/sqrt(2).
        blobs_8bit[:,2] = blobs_8bit[:,2]*sqrt(2) #Converting rooted distance to distance

        #OpenCV cv.circle to draw puncta in a binary image
        puncta_mask_8bit = np.zeros((2048, 2048)) #Creating a zeros array to generate a mask
        img_circ = np.zeros((2048,2048)) #Creating a zeros array of the same shape as the input image on which circles are drawn
        for blob in blobs_8bit:
            y, x, r = blob #extracting xy position and radius for ith puncta
            y = int(y) #Converting to an integer so as to be useful as discrete pixel measurements
            x = int(x) #Converting to an integer so as to be useful as discrete pixel measurements
            r = int(r) #Converting to an integer so as to be useful as discrete pixel measurements
            img_circ = cv.circle(img_circ, (x, y), r, color = (255, 0, 255), thickness = -1) #Using OpenCV to draw the LoG circle on img_circ

        #NEED TO UNDERSTAND - Watershed
        distance = ndi.distance_transform_edt(img_circ)
        local_max = morphology.local_maxima(distance)
        #max_coords = np.nonzero(local_max)
        markers = ndi.label(local_max)[0]
        puncta_mask_8bit[:,:] = (segmentation.watershed(img_circ, markers, mask = img_circ)).astype(int) #Using watershed to create the puncta mask
        self.mask = puncta_mask_8bit

        
        ###################################################################################################################
        #Fixing double puncta measurement issue of regionprops (regionprops sometimes boxes two sequential puncta at once)

        label_img = self.mask.astype(int) #Turning the mask into an integer array
        int_img = img.astype(int) #Turning the input image into an integer array
            
        df = regionprops_table(label_img,int_img,properties = ['label','bbox']) #Extracting bouding boxes from image using regionprops and mask image
        reg_props = pd.DataFrame(df) #for exporting
        rg_props = pd.DataFrame(df).to_numpy() #Converting regionprops dataframe to numpy array

        corrected_masks = [] #Creating an empty list to store the corrected cell masks
        corrected_int_imgs = [] #Creating an empty list to store the corrected intensity images

        bboxes = [] #Creating an empty list to store the bounding boxes produced by regionprops

        for puncta in range(int(np.max(puncta_mask_8bit[:,:]))):
            puncta_label = int(rg_props[puncta][0]) #Getting puncta label number from numpy array
            minr, minc, maxr, maxc = int(rg_props[puncta][1]), int(rg_props[puncta][2]), int(rg_props[puncta][3]), int(rg_props[puncta][4]) #Getting puncta bounding box from numpy array
            bboxes.append((minr, minc, maxr, maxc))
            cropped_mask = label_img[minr:maxr, minc:maxc] #Obtaining the puncta mask from label_img
            cropped_img = int_img[minr:maxr, minc:maxc] #Obtaining the intensity image mask from int_img
            if int(np.max(cropped_mask)) == puncta_label and int(np.min(cropped_mask)) == 0 and len(set(cropped_mask.ravel().tolist())) == 2:
                segmented_puncta = np.multiply(cropped_mask,cropped_img)/puncta_label #Condition for which bounding box is correct as is, in which case the puncta from the intensity image is extracted at the puncta mask
            else:
                cropped_mask[cropped_mask != puncta_label] = 0 #Condition for which bounding box contains puncta other than the label puncta, in which case the non-label puncta is made to be 0
                segmented_puncta = np.multiply(cropped_mask,cropped_img)/puncta_label #The puncta from the intensity image is extracted at the puncta mask
            corrected_masks.append(cropped_mask) #Storing the corrected mask into corrected_masks list
            corrected_int_imgs.append(segmented_puncta) #Storing the corrected intensity images into corrected_int_imgs list
                
        ###################################################################################################################
        #Feature Extraction

        self.totalpuncta = len(bboxes) #Determining the total number of puncta detected

        #All the additional features regionprops doesn't include being defined for extraction
        def mean(regionmask, intensityimage):
            return(np.average(intensityimage[regionmask])) #Computes the mean of the masked intensity image as a feature

        def standard_deviation(regionmask, intensityimage):
            return(np.std(intensityimage[regionmask])) #Computes the standard deviation of the masked intensity image as a feature

        def minimum(regionmask, intensityimage):
            return(np.min(intensityimage[regionmask])) #Computes the minimum of the masked intensity image as a feature

        def maximum(regionmask, intensityimage):
            return(np.max(intensityimage[regionmask])) #Computes the maximum of the masked intensity image as a feature

        def median(regionmask, intensityimage):
            return(np.median(intensityimage[regionmask])) #Computes the median of the masked intensity image as a feature

        def quartiles(regionmask, intensityimage):
            return(np.percentile(intensityimage[regionmask], q=(25, 50, 75))) #Computes the quartiles of the masked intensity image as a feature

        def integrated_intensity(regionmask, intensityimage):
            return(np.sum(intensityimage[regionmask])) #Computes the integrated intensity of the masked intensity image as a feature

        # def zernike_moments(regionmask, intensityimage):
        #     #implementing major axis as radius
        #     radius = radiuslist[i]

        #     #adding gaussian filter
        #     image = mahotas.gaussian_filter(intensityimage, 4)

        #     #setting threshold
        #     threshimg = (image > image.mean())

        #     #making labeled image
        #     labeledimg, n = mahotas.label(threshimg)

        #     #center of mass calculation
        #     COM = ndi.measurements.center_of_mass(intensityimage, labels = labeledimg)

        #     # computing zernike moments
        #     zernike_moments = mahotas.features.zernike_moments(intensityimage, radius, degree = 4, cm = COM)
            
        #     return(zernike_moments) #Computes the zernike moments of the masked intensity image as a feature
                        

        def haralick_mean(intensityimage):
            hfeats = mahotas.features.haralick(intensityimage.astype(int), compute_14th_feature=True) #Mahotas package determines haralick features of intensity image
            meanfeats = [] #An empty list is used to store the mean of haralick features computed in each direction
            for i in range(0, 14):
                templist = [] #A temporary list for averaging is created
                for j in range(0, 4):
                    templist.append(hfeats[j][i]) #Haralick features in each direction added to templist
                meanfeats.append(np.average(templist)) #Direction-averaged haralick features added to meanfeats
            return(meanfeats) #A list of direction-averaged haralick features is returned

        def haralick_range(intensityimage):
            hfeats = mahotas.features.haralick(intensityimage.astype(int), compute_14th_feature=True) #Mahotas package determines haralick features of intensity image
            rangelist = [] #An empty list is used to store the range of haralick features computed in each direction
            for i in range(0, 14):
                templist = [] #A temporary list for averaging is created
                for j in range(0, 4):
                    templist.append(hfeats[j][i]) #Haralick features in each direction added to templist
                rangelist.append(np.max(templist) - np.min(templist)) #Range of haralick features (maximum - minimum) for all directions is computed and added to rangelist
            return(rangelist) #A list of the range of haralick features across all directions is returned

        def haralick(i, imglist):
            #Computing Haralick Features of individual puncta
            f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14 = haralick_mean(imglist[bboxes[i][0]:bboxes[i][2], bboxes[i][1]:bboxes[i][3]]) #The mean haralick features of each puncta are computed
            har_tot = {"angularsecondmoment_mean":f1, "contrast_mean":f2, "correlation_mean":f3, "variance_mean":f4, "inversedifferentmoment_mean":f5, #The mean haralick features of each puncta are compiled into a dictionary
                       "sumaverage_mean":f6, "sumvariance_mean":f7, "sumentropy_mean":f8, "entropy_mean":f9, "differencevariance_mean":f10,
                       "differenceentropy_mean":f11, "correlation1_mean":f12, "correlation2_mean":f13, "maxcorrelationcoeff_mean":f14}
            f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14 = haralick_range(imglist[bboxes[i][0]:bboxes[i][2], bboxes[i][1]:bboxes[i][3]]) #The range of haralick features of each puncta are computed
            har_range = {"angularsecondmoment_range":f1, "contrast_range":f2, "correlation_range":f3, "variance_range":f4, "inversedifferentmoment_range":f5, #The range of haralick features of each puncta are compiled into a dictionary
                       "sumaverage_range":f6, "sumvariance_range":f7, "sumentropy_range":f8, "entropy_range":f9, "differencevariance_range":f10,
                       "differenceentropy_range":f11, "correlation1_range":f12, "correlation2_range":f13, "maxcorrelationcoeff_range":f14}
            har_tot.update(har_range) #The dictionaries storing both the mean and range of haralick features are combined
            return(har_tot) #A dictionary containing the mean and range of haralick features is returned

            
        #Puncta Properties Extraction from Bounding Boxes
        puncta_properties_dictionaries = [] #An empty list to store all the dictionaries containing the features being extracted
        for i in range(self.totalpuncta):
            props = regionprops_table(corrected_masks[i], #Collecting all the standard and user-created features for each puncta using regionprops
                                      corrected_int_imgs[i],
                                      properties = ['area', 'bbox_area', 'convex_area', 'filled_area', 'major_axis_length', 'minor_axis_length', 'bbox',
                                                    'centroid', 'local_centroid', 'weighted_centroid', 'weighted_local_centroid', 'coords', 'eccentricity',
                                                    'euler_number', 'extent', 'feret_diameter_max', 'image', 'convex_image', 'filled_image', 'intensity_image',
                                                    'inertia_tensor', 'inertia_tensor_eigvals', 'max_intensity', 'mean_intensity', 'min_intensity', 'label', 'moments',
                                                    'moments_central', 'moments_hu', 'moments_normalized', 'weighted_moments', 'weighted_moments_central', 'weighted_moments_hu', 'weighted_moments_normalized',
                                                    'orientation', 'perimeter', 'perimeter_crofton', 'slice', 'solidity'], 
                                      extra_properties = (mean, standard_deviation, minimum, maximum, quartiles, median, integrated_intensity))
            
            #props.update(haralick(i, img)) #Adding the haralick features to the collection of features from regionprops
            puncta_properties_dictionaries.append(props) #Adding all features computed for the ith puncta as a dictionary to puncta_properties_dictionaries


        # #Getting Additional Puncta Properties derived from already determined puncta properties
        # radiuslist = [] #Creating an empty list to store approximated radii
        # for i in range(len(puncta_properties_dictionaries)):
        #     radiuslist.append((puncta_properties_dictionaries[i]['major_axis_length'].tolist()).pop()/2) #radius is computed from the major axis length of the puncta divided by 2


        # #Puncta Additional Properties Extraction (Zernike Moments) from Bounding Boxes 
        # for i in range(self.totalpuncta):
        #     props = regionprops_table(corrected_masks[i], #Regionprops is used to compute the zernike moments of each puncta using the radius computed from major axis length (derived above)
        #                               corrected_int_imgs[i], 
        #                               extra_properties = [zernike_moments])
            
        #     puncta_properties_dictionaries[i].update(props) #Zernike moments feature is added to the list of other features in puncta_properties_dictionaries
        
        #Neighbors Code
        #distanceslist variable stores each individual distance for each puncta.
        #A single entry consists of the puncta label and its corresponding distances as a dictionary.  
        #Entries are ordered by increasing puncta id.
        distanceslist = [] #An empty list is created to store the distances to different neighbors for each puncta
        if neighbors_r > 0:
            #Computing neighbors for ith puncta
            for i in range(self.totalpuncta):
                x1 = bboxes[i][1] #obtaining x position of the top left corner of the ith puncta
                y1 = bboxes[i][0] #obtaining y position of the top left corner of the ith puncta
                    
                dx = np.round(puncta_properties_dictionaries[i].get('centroid-1')) #obtaining the x position of the centroid relative to (x1, y1)
                dy = np.round(puncta_properties_dictionaries[i].get('centroid-0')) #obtaining the y position of the centroid relative to (x1, y1)
                
                xy_centroid_i = [(x1 + dx), (y1 + dy)] #Obtaining the absolute position of the centroid
                
                label_centroid_i = puncta_mask_8bit[int(xy_centroid_i[1]), int(xy_centroid_i[0])] #Obtaining the puncta label of ith puncta
                xL = (xy_centroid_i[0] - radius) #Creating left boundary for neighbors bounding box
                xR = (xy_centroid_i[0] + radius) #Creating right boundary for neighbors bounding box
                yB = (xy_centroid_i[1] + radius) #Creating bottom boundary for neighbors bounding box
                yT = (xy_centroid_i[1] - radius) #Creating top boundary for neighbors bounding box
                
                
                #Dealing with cases where box lies outside image + within image case
                if xL < 0 : #Bounding box overflows on the left side of the image
                    xL = 0 #Bounding box is constrained to left side of image
                    
                if xR > len(img): #Bounding box overflows on right side of image
                    xR = len(img) #Bounding box is constrained to right side of image
                    
                if yT < 0: #Bounding box overflows on top side of image
                    yT = 0 #Bounding box is constrained to top side of image
                    
                if yB > len(img): #Bounding box overflows on bottom side of image
                    yB = len(img) #Bounding box is constrained to bottom of image
                    
                mask_chunk = puncta_mask_8bit[int(yT):int(yB), int(xL):int(xR)] #A portion of the mask is selected according to the neighbors bounding box
                
                #Determining the neighbors within the box on puncta i
                puncta_set = set(mask_chunk.ravel()) #Collecting the different puncta labels within the selected portion of the mask
                if len(puncta_set) > 2: #Condition for which there are more labels than just non-puncta (0's in the array) and puncta i contained (i in the array) (0 + i == 2)
                    set_of_neighbors = set(mask_chunk.ravel()) #Replicating the set of puncta labels obtained from the portion of mask selected
                    set_of_neighbors.remove(0) #Removing non-puncta (0's in the array)
                    set_of_neighbors.remove(label_centroid_i) #Removing puncta i (i in the array)
                    
                else: #Condition for which there are no neighbors (ie only 0's and i in the array)
                    set_of_neighbors = {} #Creating an empty dictionary representing no neighbors
                n_neighbors = len(set_of_neighbors) #Counting the number of neighbors present in the portion of mask
                
                #Determining the distances between puncta i and each of its neighbors
                dist_dict = {} #A dictionary to store puncta labels along with their corresponding distances (see distanceslist above)
                for j in range(n_neighbors): #Going through each neighbor in the portion of mask
                    puncta_idx = set_of_neighbors.pop() #Popping out the last puncta label from the set of neighbors
                    neighbory = bboxes[puncta_idx.astype(int)-1][0] + puncta_properties_dictionaries[puncta_idx.astype(int)-1].get('centroid-0') #Referencing bboxes and puncta_properties_dictionaries to extract the y position of the neighbor 
                    neighborx = bboxes[puncta_idx.astype(int)-1][1] + puncta_properties_dictionaries[puncta_idx.astype(int)-1].get('centroid-1') #Referencing bboxes and puncta_properties_dictionaries to extract the y position of the neighbor
                    pos_neighbor = [neighborx, neighbory] #Storing the neighbor position in (x, y) form
                    dist_dict[puncta_idx.astype(int)-1] = dist(pos_neighbor, xy_centroid_i) #computing the distance between the neighbor and puncta i, and adding the pair of puncta label and distance to the corresponding puncta id in the dictionary
                    
                distanceslist.append(dist_dict) #Storing the distances dictionary for puncta i into distanceslist
                avdist = np.average(list(dist_dict.values())) #Computing the average distance to neighbors for puncta i
                puncta_properties_dictionaries[i].update({'Average Distance':avdist, 'Neighbor Count':n_neighbors}) #Adding average neighbors distance for puncta i as a property to puncta_properties_dictionaries
                
                #Percent Touching
                total_perimeter = 0
                touchers_perimeter = defaultdict(int)
                for coord in (list(puncta_properties_dictionaries[i].get('coords')))[0].tolist():
                    xy_coord = (x1 + coord[1], y1 + coord[0])
                    if xy_coord[0] != 0:
                        if puncta_mask_8bit[xy_coord[1], xy_coord[0]-1] != label_centroid_i:
                            total_perimeter += 1
                            if puncta_mask_8bit[xy_coord[1], xy_coord[0]-1] != 0:
                                touchers_perimeter[puncta_mask_8bit[xy_coord[1], xy_coord[0]-1]] += 1

                    if xy_coord[0] != len(img)-1:        
                        if puncta_mask_8bit[xy_coord[1], xy_coord[0]+1] != label_centroid_i:
                            total_perimeter += 1
                            if puncta_mask_8bit[xy_coord[1], xy_coord[0]+1] != 0:
                                touchers_perimeter[puncta_mask_8bit[xy_coord[1], xy_coord[0]+1]] += 1
                        
                    if xy_coord[1] != 0:
                        if puncta_mask_8bit[xy_coord[1]-1, xy_coord[0]] != label_centroid_i:
                            total_perimeter += 1
                            if puncta_mask_8bit[xy_coord[1]-1, xy_coord[0]] != 0:
                                touchers_perimeter[puncta_mask_8bit[xy_coord[1]-1, xy_coord[0]]] += 1
                        
                    if xy_coord[1] != len(img)-1:
                        if puncta_mask_8bit[xy_coord[1]+1, xy_coord[0]] != label_centroid_i:
                            total_perimeter += 1
                            if puncta_mask_8bit[xy_coord[1]+1, xy_coord[0]] != 0:
                                touchers_perimeter[puncta_mask_8bit[xy_coord[1]+1, xy_coord[0]]] += 1
                
                if touchers_perimeter:
                    touchers_PT = dict(touchers_perimeter)
                    for key in touchers_PT:
                        touchers_PT[key] = touchers_PT[key]*100/total_perimeter
                        puncta_properties_dictionaries[i].update({'Percent Touching by Neighbor':touchers_PT})
                        
                    av_pt = np.average(list(touchers_PT.values()))
                    puncta_properties_dictionaries[i].update({'Average Percent Touching':av_pt})
                else:
                    touchers_PT = {}
                    av_pt = np.average(list(touchers_PT.values()))
                    puncta_properties_dictionaries[i].update({'Percent Touching by Neighbor':touchers_PT})
                    puncta_properties_dictionaries[i].update({'Average Percent Touching':av_pt})

            ##########################################################################################################################################
            
        self.puncta_properties_dictionaries = puncta_properties_dictionaries
        if distanceslist:
            self.distanceslist = distanceslist

        self.finaldf = pd.DataFrame(puncta_properties_dictionaries)
        self.bboxes = bboxes
        
        end = timer()
        
        self.runtime = end - start
        
    def box_one(self, pidx):
        xL = self.bboxes[pidx][1]
        yT = self.bboxes[pidx][0]
        xR = self.bboxes[pidx][3]
        yB = self.bboxes[pidx][2]
        
        if xL < 0 :
            xL = 0
        if xR > len(self.img):
            xR = len(self.img)
        if yT < 0:
            yT = 0
        if yB > len(self.img):
            yB = len(self.img)
        
        fig, ax = plt.subplots(figsize=(32, 32))
        ax.imshow(self.img)
        rect = mpatches.Rectangle((xL, yT), xR - xL, yB - yT,
                                      fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
        ax.set_axis_off()
        plt.tight_layout()
        #plt.savefig("PunctaDistanceTest_Rad_" + str(radius) + "_ID2142.jpg",bbox_inches='tight',pad_inches=0, dpi=200)
        plt.show()

    #Algorithm to validate neighbors feature.  Input: Puncta ID (puncta number in the dataframe).
    #Output: Draws the box on puncta i and draws lines to each
    #neighbor within the box, indicating distances and puncta's ID number from the dataframe.
    def puncta_neighbors_tester(self, pidx):
        x1 = self.bboxes[pidx][1]
        y1 = self.bboxes[pidx][0]
        dy = np.round(self.puncta_properties_dictionaries[pidx].get('centroid-0'))
        dx = np.round(self.puncta_properties_dictionaries[pidx].get('centroid-1'))
        neighbor_ids = list(self.distanceslist[pidx].keys())
        
        xy_centroid_i = [(x1 + dx), (y1 + dy)]
        xL = (xy_centroid_i[0] - radius)
        xR = (xy_centroid_i[0] + radius)
        yB = (xy_centroid_i[1] + radius)
        yT = (xy_centroid_i[1] - radius)
        
        if xL < 0 :
            xL = 0
        if xR > len(self.img):
            xR = len(self.img)
        if yT < 0:
            yT = 0
        if yB > len(self.img):
            yB = len(self.img)
        
        fig, ax = plt.subplots(figsize=(32, 32))
        ax.imshow(self.img)
        rect = mpatches.Rectangle((xL, yT), xR - xL, yB - yT,
                                      fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)
        for neighbor in neighbor_ids:
            x1_neighbor = self.bboxes[neighbor][1]
            y1_neighbor = self.bboxes[neighbor][0]
            dx_neighbor = np.round(self.puncta_properties_dictionaries[neighbor].get('centroid-1'))
            dy_neighbor = np.round(self.puncta_properties_dictionaries[neighbor].get('centroid-0'))
            xy_centroid_neighbor = [(x1_neighbor + dx_neighbor), (y1_neighbor + dy_neighbor)]
            
            xvals = [xy_centroid_i[0], xy_centroid_neighbor[0]]
            yvals = [xy_centroid_i[1], xy_centroid_neighbor[1]]
            plt.plot(xvals, yvals, 'yo', linestyle="dotted", alpha = 0.3, fillstyle = 'none', markeredgewidth = 0)
            plt.text(xvals[0]-0.015, yvals[0]+0.25, str(pidx), fontsize = 3)
            plt.text(xvals[1] - 0.050, yvals[1]-0.25, str(np.round(self.distanceslist[pidx][neighbor], 2)), fontsize = 3)
            
        
        ax.set_axis_off()
        plt.tight_layout()
        #plt.savefig("PunctaDistanceTest_Rad_" + str(radius) + "_ID2142.jpg",bbox_inches='tight',pad_inches=0, dpi=200)
        plt.show()
        
##########################################################################################################################################    
    #UMAP analysis

    # def umap(self):
    #     reducer = umap.UMAP()
        
    #     reduction_data = self.finaldf[
            
    #         [
    #          'angularsecondmoment_mean', 'contrast_mean', 'correlation_mean', 'variance_mean'
    #          # 'area', 'bbox_area', 'convex_area', 'filled_area', 'major_axis_length', 'minor_axis_length',
    #          # 'local_centroid-0', 'local_centroid-1', 'weighted_centroid-0', 'weighted_centroid-1', 'weighted_local_centroid-0', 'weighted_local_centroid-1', 'eccentricity',
    #          # 'euler_number', 'extent', 'feret_diameter_max', 'image', 'convex_image', 'filled_image', 'intensity_image',
    #          # 'inertia_tensor-0-0', 'inertia_tensor-1-0', 'inertia_tensor-1-0', 'inertia_tensor-1-1', 'inertia_tensor_eigvals-0', 'inertia_tensor_eigvals-1', 'max_intensity', 'mean_intensity', 'min_intensity', 'moments-0-0', 'moments-0-1', 'moments-0-2', 'moments-0-3', 'moments-1-0', 'moments-1-1', 'moments-1-2', 'moments-1-3', 'moments-2-0', 'moments-2-1', 'moments-2-2', 'moments-2-3', 'moments-3-0', 'moments-3-1', 'moments-3-2', 'moments-3-3', 'moments_central-0-0',
    #          # 'moments_central-0-1', 'moments_central-0-2', 'moments_central-0-3', 'moments_central-1-0', 'moments_central-1-1', 'moments_central-1-2', 'moments_central-1-3', 'moments_central-2-0', 'orientation', 'perimeter', 'perimeter_crofton', 'slice', 'solidity'
    #          ]
            
    #         ].values
        
    #     zscaled_data = StandardScaler().fit_transform(reduction_data)
        
    #     fitted_data = reducer.fit_transform(zscaled_data)
        
    #     plt.scatter(fitted_data[:,0], fitted_data[:,1])
    #     plt.show()

##########################################################################################################################################    

