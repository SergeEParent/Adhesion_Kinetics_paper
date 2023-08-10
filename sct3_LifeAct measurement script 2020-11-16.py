# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 11:20:14 2019

@author: Serge
"""


### Import what you need.
import numpy as np

### Import Pdfpages to save figures in to a multi-page PDF file:
from matplotlib.backends.backend_pdf import PdfPages

# NumPy is a core Python package and can create arrays of any dimension, images
# that are read in to Python are interpreted as arrays (so for example an RGB 
# image has the x and y axis)
import mahotas as mh

### From PIL import Image to read image files (PIL can handle TIFF tags)
from PIL import Image

# Mahotas is basically a collection of functions/algorithms useful for
# analyzing micrographs or other images found in biology.
from matplotlib import pyplot as plt
#from matplotlib import colors

# For creating 3D plots:
from mpl_toolkits.mplot3d import Axes3D

# clustering for a test:
from sklearn.cluster import KMeans
# from sklearn.cluster import AffinityPropagation
# from sklearn.cluster import spectral_clustering
# from sklearn.cluster import AgglomerativeClustering

### Import heapq to find the nth biggest number when cleaning up regions using 
### mahotas.
import heapq

### Import math so that you can calculate the angle between 2 vectors by using
### the cross product:
import math

### Import the os so you can access the file.
import os

### import glob so that you can look through a folder for files with a pre-
### defined file extension (eg. only .png files or only .jpg files)
import glob

### Import csv so that you can create comma style value files from your data.
### You can open csv files with excel.
import csv

### Import interpolate and optimization from scipy to interpolate intensity
### data and fit model functions/solve functions respectively:
#import scipy.interpolate as interpolate
#import scipy.optimize as optimization
import scipy.ndimage.morphology as scp_morph
#import scipy.spatial as scp_spatial
#from scipy.signal import argrelextrema
### Import the garbage collector - to keep memory trim:
import gc


import time

### Import orthogonal distance regression to make curvature estimates based on
### bins:
from scipy import  odr
#from scipy import ndimage as ndi
from scipy.signal import argrelextrema

### Import various computervision packages:
from skimage import feature
from skimage import segmentation
from skimage import measure
from skimage.metrics import structural_similarity
from skimage import morphology
from skimage import filters

### The 2 below are used for getting hough transforms. Used in the function
### hough_circle_fit:
from skimage import transform
from skimage.draw import circle_perimeter

### A lot of figures are going to be opened before they are saved in to a
### multi-page pdf, so increase the max figure open warning:
plt.rcParams.update({'figure.max_open_warning':300})




###############################################################################
# Define some functions that you will need:

def multipdf():
    """ This function saves a multi-page PDF of all the open figures. """
    pdf = PdfPages('%s Overlay & Intensity Plots.pdf' %csv_filename)
    figs = [plt.figure(f) for f in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pdf, format='pdf' , dpi=80)
    pdf.close()
###



def sub(abra,cadabra):
    """ This function will subtract b from a (ie. a - b) unless a-b
     is less than zero, in which case it will just give 0. This allows one to take
     a slice from a to b (ie. [a:b]) from one end of an image to the other as a
     sliding window. """
    if abra-cadabra < 0:
        return 0
    else:
        return abra-cadabra
### 



def pad_img_edges(image, value, width=1):
    """Will add a columns and rows of the specified width containing "value"
    to the edges of a 2D or 3D image. If width is set to be a negative number
    will instead remove columns and rows of specified width from each side.
    Eg. Original shape: (180, 156, 3)
    with width = 1 -> (182, 158, 3)
    with width = -1 -> (178, 154, 3)
    **Caution: The removing feature will remove rows and columns even if they
    are non-zero. Please be careful not to remove important image information."""
    width_sign = width / abs(width)
    width = abs(width)
    image_copy = np.copy(image)
    image_dim = image_copy.shape
    if len(image_dim) == 2:
        image_copy = image_copy.reshape(image_dim + (1,))
        image_dim = image_copy.shape
    elif len(image_dim) == 3:
        pass
    else:
        image_copy = float('nan')
    if width_sign < 0:
        image_copy = image_copy[width:-1*width, width:-1*width, :]
    else:
        cols = np.ones((image_dim[0], width, image_dim[2]), dtype = image_copy.dtype) * value
        rows = np.ones(( width, image_dim[1] + (width * 2), image_dim[2] ), dtype = image_copy.dtype) * value
        image_copy = np.column_stack((cols, image_copy, cols))
        image_copy = np.row_stack((rows, image_copy, rows))
    if image_dim[2] == 1:
        image_copy = np.squeeze(image_copy)
    return image_copy
###



def get_local_snr(arr, footprint):
    # snr = mean(array)**2 / standard_deviation(array)**2 ; only applicable to 
    # arrays with all positive values
    arr = (np.iinfo(arr.dtype).max * (arr / arr.max())).astype(np.iinfo(arr.dtype))
    if (arr < 0).any():
        return np.nan, np.zeros(arr.shape, dtype=float) * np.nan
    else:
        snr_arr = filters.rank.mean(arr.astype(arr.dtype.name), selem=footprint).astype(float)**2 / np.std(arr, dtype=float)**2
        snr_arr.min()
        return snr_arr.min(), snr_arr
###



def label_peaks_from_binary(binary_img, exclude_border=False, min_distance=1, footprint=None):
    dist = scp_morph.distance_transform_edt(binary_img)**2
    peaks = feature.peak_local_max(dist, indices=False, exclude_border=exclude_border, min_distance=min_distance, footprint=footprint)
    labeled_peaks = morphology.label(peaks)
    
    return labeled_peaks
###



def subdivide_img(img_shape, desired_subdivision_sidelength):
    n_subdivisions_xx, n_subdivisions_yy = np.divide(img_shape, desired_subdivision_sidelength)
    n_subdivisions = np.multiply(n_subdivisions_xx, n_subdivisions_yy)
    
    return n_subdivisions
###
        
    

def fast_region_compete(image, labeled_img, iterations=25, mask=None, compactness=0, min_size=None):
    if not(min_size):
        min_size = 0
    removed_label_list = []
    count = 0
    
    seg = morphology.remove_small_objects(labeled_img, min_size)
    removed_label_list.append(np.unique(labeled_img[seg == 0]))

    while count < iterations-1:
        seg_old = segmentation.watershed(image, seg, mask=mask, compactness=compactness)
        seg = morphology.remove_small_objects(seg_old, min_size)
        removed_label_list.append(np.unique(seg_old[seg == 0]))
        count += 1

    seg = segmentation.watershed(image, seg, mask=mask, compactness=compactness)

    return seg, removed_label_list
###
        

    
def kmeans_binary_sep(labeled_img, normed_metrics):
    """ normed_metrics MUST be a 2D array"""
    labeled_img_labels = np.unique(labeled_img)[np.nonzero(np.unique(labeled_img))]
#        labeled_img_labels = np.unique(labeled_img)

    #circle_or_polygon_score = [(regionprops((btest==label)*btest)[0].area / regionprops((btest==label)*btest)[0].perimeter**2) * (4 * np.pi) for label in btest_labels]
    #normed_metrics_labeled = np.column_stack((normed_metrics, labeled_img_labels))

    km = KMeans(n_clusters=2, random_state=42, n_init=100, max_iter=500, algorithm="full").fit(normed_metrics)
    km_group1 = labeled_img_labels * (km.labels_ > 0)
    km_labels_group1 = km_group1[np.nonzero(km_group1)]
    km_values_group1 = normed_metrics[np.where(km.labels_ > 0)]
    km_values_group1_score = km_values_group1.mean()
    #label_indices_group1 = np.asarray([np.where(normed_metrics_labeled[:,-1] == x)[0][0] for x in km_labels_group1])
    #plt.imshow(np.add.reduce([(btest==i) for i in km_labels]))
    labeled_img_group1 = np.add.reduce([(labeled_img==i) for i in km_labels_group1])
    if labeled_img_group1.any:
        pass
    else:
        labeled_img_group1 = np.zeros(labeled_img.shape)

    km_group2 = labeled_img_labels * (km.labels_ == 0)
    km_labels_group2 = km_group2[np.nonzero(km_group2)]
    km_values_group2 = normed_metrics[np.where(km.labels_ == 0)]
    km_values_group2_score = km_values_group2.mean()
    #label_indices_group2 = np.asarray([np.where(normed_metrics_labeled[:,-1] == x)[0][0] for x in km_labels_group2])
    #plt.imshow(np.add.reduce([(btest==i) for i in km_labels]))
    labeled_img_group2 = np.add.reduce([(labeled_img==i) for i in km_labels_group2])
    if labeled_img_group2.any:
        pass
    else:
        labeled_img_group2 = np.zeros(labeled_img.shape)


    if km_values_group2_score > km_values_group1_score:
        labeled_img_high_group = np.copy(labeled_img_group2)
        labeled_img_low_group = np.copy(labeled_img_group1)
    else:
        labeled_img_high_group = np.copy(labeled_img_group1)
        labeled_img_low_group = np.copy(labeled_img_group2)

    #fig, ax = plt.subplots(figsize=(8,8))
    #ax.scatter(km_values_group1[:,0], km_values_group1[:,1], c='b')
    #ax.scatter(km_values_group2[:,0], km_values_group2[:,1], c='r')
    #[ax.annotate('%s'%km_labels_group1[i], xy=(km_values_group1[i,0], km_values_group1[i,1]), xytext=(3,3), textcoords='offset points') for i in range(len(km_labels_group1))]
    #[ax.annotate('%s'%km_labels_group2[i], xy=(km_values_group2[i,0], km_values_group2[i,1]), xytext=(3,3), textcoords='offset points') for i in range(len(km_labels_group2))]
    #plt.show()


    return labeled_img_high_group, labeled_img_low_group
###



### All of this was a test is to try and find which pixels in red_ridges
### most closely fit a circle and to then give these pixels values that can
### then be used when trying to use kmeans_binary_sep to separate real signal
### from background signal:





#     from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
#     def gauss_mix_binary_sep(labeled_img, normed_metrics):
#         labeled_img_labels = np.unique(labeled_img)[np.nonzero(np.unique(labeled_img))]
#         #init_cluster_means = np.array([[0, 0, 0, 0, 0.1], [0.5, 0.5, 0.5, 0.5, 0.5]])
#         #init_cluster_weights = np.array([0.5, 0.5])
#         gm = BayesianGaussianMixture(n_components=2, random_state=42, tol=0.8, weight_concentration_prior_type='dirichlet_distribution', weight_concentration_prior=0.0001).fit(normed_metrics)
       
#         gm_group1 = labeled_img_labels * (gm.predict(normed_metrics) > 0)
#         gm_labels_group1 = gm_group1[np.nonzero(gm_group1)]
#         gm_values_group1 = normed_metrics[np.where(gm.predict(normed_metrics) > 0)]
#         gm_values_group1_score = gm_values_group1.mean()
#         labeled_img_group1 = np.add.reduce([(labeled_img == i) for i in gm_labels_group1])
        
#         gm_group2 = labeled_img_labels * (gm.predict(normed_metrics) == 0)
#         gm_labels_group2 = gm_group2[np.nonzero(gm_group2)]
#         gm_values_group2 = normed_metrics[np.where(gm.predict(normed_metrics) == 0)]
#         gm_values_group2_score = gm_values_group2.mean()
#         labeled_img_group2 = np.add.reduce([(labeled_img == i) for i in gm_labels_group2])
        
#         fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16,32))
#         ax1.imshow(mh.overlay(tl_channel, segmentation.find_boundaries(labeled_img_group1 * labeled_img)))
#         ax1.set_title("group1")
#         ax2.imshow(mh.overlay(tl_channel, segmentation.find_boundaries(labeled_img_group2 * labeled_img)))
#         ax2.set_title("group2")
#         plt.show()
        
#         fig = plt.figure(figsize=(8,8))
#         ax = fig.add_subplot()#projection='3d')
# #        ax.scatter(gm_values_group1[:,0], gm_values_group1[:,1], gm_values_group1[:,2], c='b')
# #        ax.scatter(gm_values_group2[:,0], gm_values_group2[:,1], gm_values_group2[:,2], c='r')
#         ax.scatter(gm_values_group1[:,0], gm_values_group1[:,1], c='b')
#         ax.scatter(gm_values_group2[:,0], gm_values_group2[:,1], c='r')
#         ax.set_xlabel('signal_median')
#         ax.set_ylabel('signal_std')
#         plt.show()
# #        ax.set_zlabel('eccentricity')
#         #[ax.annotate('%s'%gm_labels_group1[i], xy=(gm_values_group1[i,0], gm_values_group1[i,1]), xytext=(3,3), textcoords='offset points') for i in range(len(gm_labels_group1))]
#         #[ax.annotate('%s'%gm_labels_group2[i], xy=(gm_values_group2[i,0], gm_values_group2[i,1]), xytext=(3,3), textcoords='offset points') for i in range(len(gm_labels_group2))]
#         ax.scatter(gm_values_group2[np.where(gm_labels_group2 == 115), 0], gm_values_group2[np.where(gm_labels_group2 == 115), 1], gm_values_group2[np.where(gm_labels_group2 == 115), 2], c='k')
#         ax.scatter(gm_values_group2[np.where(gm_labels_group2 == 114), 0], gm_values_group2[np.where(gm_labels_group2 == 114), 1], gm_values_group2[np.where(gm_labels_group2 == 114), 2], c='k')
#         plt.show()

        
#         return labeled_img_labels
        
    
    
#    from sklearn.manifold import TSNE
#    def tsne_binary_sep(labeled_img, normed_metrics, details=False):
#        labeled_img_labels = np.unique(labeled_img)[np.nonzero(np.unique(labeled_img))]
#        ts = TSNE(n_components=1, random_state=42, method='exact', init='pca', ).fit_transform(normed_metrics[:,:1])
#        
#        
#        km = KMeans(n_clusters=2, random_state=42).fit(ts)
#        
#        group1 = km.labels_ == 0
#        ts_group1 = group1 * labeled_img_labels
#        ts_labels_group1 = ts_group1[np.nonzero(ts_group1)]
#        ts_values_group1 = normed_metrics[np.where(group1), :]
#        ts_values_group1_score = ts_values_group1.mean()
#        labeled_img_group1 = np.add.reduce([(labeled_img == i) for i in ts_labels_group1])
#
#        
#        group2 = km.labels_ > 0
#        ts_group2 = group2 * labeled_img_labels
#        ts_labels_group2 = ts_group2[np.nonzero(ts_group2)]
#        ts_values_group2 = normed_metrics[np.where(group2), :]
#        ts_values_group2_score = ts_values_group2.mean()
#        labeled_img_group2 = np.add.reduce([(labeled_img == i) for i in ts_labels_group2])
#    
#        fig, ax = plt.subplots()
#        ax.scatter(ts[group1, 0], ts[group1, 1], c='c')
#        ax.scatter(ts[group2, 0], ts[group2, 1], c='b')
#        plt.show()
#        
#        if details:
#            return labeled_img_group1, labeled_img_group2, group1, group2, ts, km
#        else:
#            return labeled_img_group1, labeled_img_group2
        
    
    
    
    # def agglom_clust_binary_sep(labeled_img, normed_metrics, connectivity=None, n_clusters=2, distance_threshold=None, compute_full_tree='auto'):
    #     """ 'connectivity' keyword has to be a connectivity matrix or 'None'."""
    #     labeled_img_labels = np.unique(labeled_img)[np.nonzero(np.unique(labeled_img))]
    #     ac = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward', connectivity=connectivity, distance_threshold=distance_threshold, compute_full_tree=compute_full_tree).fit(normed_metrics)
        
    #     ac_group1 = labeled_img_labels * (ac.labels_ > 0)
    #     ac_labels_group1 = ac_group1[np.nonzero(ac_group1)]
    #     ac_values_group1 = normed_metrics[np.where(ac.labels_ > 0)]
    #     ac_values_group1_score = ac_values_group1.mean()
    #     labeled_img_group1 = np.add.reduce([(labeled_img == i) for i in ac_labels_group1])
        
    #     ac_group2 = labeled_img_labels * (ac.labels_ == 0)
    #     ac_labels_group2 = ac_group2[np.nonzero(ac_group2)]
    #     ac_values_group2 = normed_metrics[np.where(ac.labels_ == 0)]
    #     ac_values_group2_score = ac_values_group2.mean()
    #     labeled_img_group2 = np.add.reduce([(labeled_img == i) for i in ac_labels_group2])
        
    #     fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16,32))
    #     ax1.imshow(labeled_img_group1)
    #     ax1.set_title("group1")
    #     ax2.imshow(labeled_img_group2)
    #     ax2.set_title("group2")
    #     plt.show()
        
    #     fig, ax = plt.subplots(figsize=(8,8))
    #     ax.scatter(ac_values_group1[:,0], ac_values_group1[:,1], c='b')
    #     ax.scatter(ac_values_group2[:,0], ac_values_group2[:,1], c='r')
    #     [ax.annotate('%s'%ac_labels_group1[i], xy=(ac_values_group1[i,0], ac_values_group1[i,1]), xytext=(3,3), textcoords='offset points') for i in range(len(ac_labels_group1))]
    #     [ax.annotate('%s'%ac_labels_group2[i], xy=(ac_values_group2[i,0], ac_values_group2[i,1]), xytext=(3,3), textcoords='offset points') for i in range(len(ac_labels_group2))]
    #     plt.show()
        
    #     return labeled_img_group1, labeled_img_group2




def hough_circle_fit(img_to_search, dist_btwn_pks, percentile_thresh, img_to_compare_against=None):
    """ Tries to fit hough circles to img_to_search and will return all 
    peaks with an occurence > the percentile_thresh (ranges between 0 and 1).
    Will return a copy of img_to_search where each pixel has a value equal 
    to it's squared distance from the nearest point on one of the circles 
    from good_circles. If img_to_compare is set, the distances from each 
    pixel in img_to_compare will be returned instead of img_to_search. An 
    example of it's use case would be if you provide a denoised img of the 
    edges of an image with round objects in it for the hough circles to try 
    and fit to and then compare those circles to the noisy img. Useful if 
    you are trying to use pixels that seem indistinguishable from noise to 
    complete shapes or if you need to compare noisy images or retain a 
    subset of noise that seems to complete a circle. img_to_search (and 
    img_to_compare if provided) must be boolean and of the same shape."""
    img_pad = pad_img_edges(img_to_search, 0, width=int(min(img_to_search.shape)/2)).astype(bool)
    approx_possible_num_cells_in_img = int(np.dot(*np.round(np.asarray(img_to_search.shape) / (dist_btwn_pks*2)) + 2,))
    hough_radii = range(int(dist_btwn_pks*2-10), int(dist_btwn_pks*2+11), 1)
    hough_circles = transform.hough_circle(img_pad, hough_radii)
    accums, cx, cy, radii = transform.hough_circle_peaks(hough_circles, 
                                                         hough_radii, 
                                                         min_xdistance=int(dist_btwn_pks),
                                                         min_ydistance=int(dist_btwn_pks),
                                                         num_peaks=approx_possible_num_cells_in_img,
                                                         #total_num_peaks=approx_possible_num_cells_in_img,
                                                         normalize=True)
    # test = np.zeros(img_pad.shape)
    # test[cy,cx] = 1
    
    max_project = mh.stretch(np.max(hough_circles, axis=0))
    hist, bins = np.histogram(max_project, bins=255)
    cumul_sum = np.cumsum(hist)
    thresh_indices = cumul_sum > cumul_sum.max() * percentile_thresh
    thresh = bins[np.where(thresh_indices)].min()
    threshed_houghs = max_project > thresh

    #gauss = filters.gaussian(max_project, sigma=2)
    vals = max_project * threshed_houghs
    pks = feature.peak_local_max(vals, footprint=morphology.disk(dist_btwn_pks), indices=True)
    indices = [np.argmin((cy - x[0])**2 + (cx - x[1])**2) for x in pks]
    pks_accums = [int(accums[i]) for i in indices]
    pks_radii = [int(radii[i]) for i in indices]
    pks_row, pks_col = [int(cy[i]) for i in indices], [int(cx[i]) for i in indices]
    
    
    ### for each circle defined by pks_row, pks_col, pks_radii the radius of the
    ### circle can be compared with the magnitude of the line from 
    ### (pks_row, pks_col) to (row, col) in red_ridges to give a distance
    if np.any(img_to_compare_against):
        img_to_compare_against_pad = pad_img_edges(img_to_compare_against, 0, width=int(min(img_to_search.shape)/2)).astype(bool)
        rid_rc = np.asarray(np.where(img_to_compare_against_pad))
    else:
        rid_rc = np.asarray(np.where(img_pad))

    pks_rc = np.array([pks_row, pks_col])
    img_dists_per_pk = np.array([np.sqrt(np.add.reduce((rid_rc.T - rc)**2, axis=1)) for rc in pks_rc.T])
    dists_per_pk = abs(img_dists_per_pk - np.array(pks_radii, ndmin=2).T).tolist()

    
    edges_with_dists_list = list()
    for i,v in enumerate(dists_per_pk):
        edges_with_dists = np.zeros(img_pad.shape)
        for j,w in enumerate(v):
            edges_with_dists[tuple(np.array(x) for x in rid_rc[:,j])] = dists_per_pk[i][j]
        edges_with_dists_list.append(edges_with_dists)
    circle_dists = np.min(edges_with_dists_list, axis=0)
    circle_dists = circle_dists[int(min(img_to_search.shape)/2) : -int(min(img_to_search.shape)/2),
                                int(min(img_to_search.shape)/2) : -int(min(img_to_search.shape)/2)]
    
    
    circles = np.zeros(img_pad.shape)
    for center_y, center_x, radius in zip(pks_row, pks_col, pks_radii):
        circy, circx = circle_perimeter(center_y, center_x, radius,
                                        shape=img_pad.shape)
        circles[circy, circx] = True
    good_circles = circles[int(min(img_to_search.shape)/2) : -int(min(img_to_search.shape)/2),
                           int(min(img_to_search.shape)/2) : -int(min(img_to_search.shape)/2)]
    
    ### circle_dists gives the red_ridges pixels with values reflecting their
    ### squared distance to the nearest circle

    return circle_dists, good_circles
###
   

    

def get_edge_thresh(list_of_segmentations, percent_cutoff):
    edge_map = np.add.reduce([segmentation.find_boundaries(x) for x in list_of_segmentations])
    # cumul_sum = np.cumsum(np.sort(np.ravel(edge_map)))
    # thresh_val_cumul_sum = cumul_sum.max() * percent_cutoff
    # thresh_cumul_sum = cumul_sum >= thresh_val_cumul_sum
    # thresh_val = np.sort(np.ravel(edge_map))[np.where(thresh_cumul_sum)].min()
    thresh_val = edge_map.max() * percent_cutoff
    
    seg_thresh_edges = edge_map >= thresh_val

    return seg_thresh_edges, edge_map
###




def fill_caveolae(binary_img, connectivity=1, min_radius=1, max_radius=1, area_threshold=64):
    radii = np.arange(min_radius, max_radius+1, step=1, dtype=int)
    if connectivity == 1:
        selem = [morphology.disk(r) for r in radii]
    elif connectivity == 2:
        selem = [morphology.square(2*r + 1) for r in radii]
    
    cav_fills = list()
    for s in selem:
        closed_img = morphology.closing(binary_img, selem=s)
        filled_img = morphology.remove_small_holes(closed_img, 
                                                   connectivity=connectivity, 
                                                   area_threshold=area_threshold)
        filled_binary = morphology.remove_small_holes(binary_img,
                                                      connectivity=connectivity,
                                                      area_threshold=area_threshold)
        closing_px = np.logical_xor(binary_img, closed_img)
        edge_closing_px = segmentation.find_boundaries(filled_binary) * closing_px
        fill_cav_no_closing_px = np.logical_xor(filled_img, 
                                                edge_closing_px)
        # fill_cav_no_closing_px = np.logical_xor(filled_img, 
        #                                         closing_px)
        cav_fill_only = np.logical_xor(filled_binary,
                                       fill_cav_no_closing_px)
        cav_fills.append(cav_fill_only)

    cav_fill_binary = np.logical_or.reduce([np.logical_or(cav, binary_img) 
                                            for cav in cav_fills], dtype=bool)
    return cav_fill_binary
###




def join_contig_regions(init_binary_img, seg, percent_cutoff=0.75, n_iters=10):
    """ The regions in 'seg' will be joined into 'initial_binary_img' until
    no more regions can be joined. They are joined if the percent of pixels
    from the outer borders of each region overlapping with 
    'initial_binary_img' exceeds 'percent_cutoff'. This can be repeated for
    n_iters. 'init_binary_img' and 'seg' must have same shape. 
    'init_binary_img' should be of dtype == bool, and 'seg' should be a 
    segmented image (eg. using watershed), where no regions of interest 
    have a 0 label, also it should have dtype == int. """

    ### Set up your initial images:
    growing_region = init_binary_img.copy().astype(bool)
    seg = seg.astype(int)
    seg *= ~growing_region

    ### These dictionaries will each contain lists associated with a key,
    ### where the key is the iteration that the function was on when it
    ### produced the list. The lists are lists of number of border pixels
    ### that overlap with the growing region, the total number of border
    ### pixels, and the percent of pixels that were overlapping with the
    ### growing region, respectively:
    bords_touching_list = dict()
    bords_all_list = dict()
    perc_bords_touching_list = dict()

    
    i = 0 # Set up initial iteration:
    while i < n_iters:
        ### Set up the keys (ie. iteration number) in the dictionaries:
        bords_touching_list[i] = []
        bords_all_list[i] = []
        perc_bords_touching_list[i] = []

        ### If there are any regions left to combine in 'seg'...
        if seg.any(): 
            ### cycle through each region in 'seg'...            
            for x in np.unique(seg):
                ### and figure out the percent of pixels touching the growing
                ### regions...
                bords_touching = np.count_nonzero(segmentation.find_boundaries(seg == x) * 
                                             ~(seg == x) * growing_region)
                bords_all = np.count_nonzero(segmentation.find_boundaries(seg == x) *
                                             ~(seg == x))
                percent_bords_touching = bords_touching / bords_all
                
                ### and then append this information to the dicts at their
                ### given key (iteration) number:
                bords_touching_list[i].append(bords_touching)
                bords_all_list[i].append(bords_all)
                perc_bords_touching_list[i].append(percent_bords_touching)
            ### Now update the 'growing_region' and 'seg':
            perc_bords_touching_arr = [(seg == x) * perc_bords_touching_list[i][idx] 
                                       for idx,x in enumerate(np.unique(seg))]
            perc_bords_touching_arr = np.add.reduce(perc_bords_touching_arr)
            growing_region_prior = growing_region.copy()
            growing_region += perc_bords_touching_arr > percent_cutoff
            seg *= ~growing_region
    
            ### Lastly, if there was no change between the current iterations
            ### growing region and that of the previous one, then stop early
            ### so you don't waste time going through the rest of the 
            ### iterations when nothing is going to change...
            if np.all(growing_region == growing_region_prior):
                break
            else:
                i += 1 # otherwise, move the iteration number forward 1.
        
        ### otherwise, if there are no regions to combine in 'seg', you're done.
        else: 
            break
    
    return growing_region
###
    



def find_cell_seeds(tl_channel, red_channel, grn_channel, blu_channel, footprint):


    red_gauss = mh.stretch(filters.gaussian(red_channel, sigma=1), 1, 255)
    log_red_gauss = np.log(red_gauss)
    log_red_gauss[np.where(~np.isfinite(log_red_gauss))] = 0
    log_red_gauss = mh.stretch(log_red_gauss)

    log_red_channel = np.log(mh.stretch(red_channel, 1, 255))
    log_red_channel[np.where(~np.isfinite(log_red_channel))] = 0
    log_red_channel = mh.stretch(log_red_channel)


    grn_gauss = mh.stretch(filters.gaussian(grn_channel, sigma=1), 1, 255)
    log_grn_gauss = np.log(grn_gauss)
    log_grn_gauss[np.where(~np.isfinite(log_grn_gauss))] = 0
    log_grn_gauss = mh.stretch(log_grn_gauss)

    log_grn_channel = np.log(mh.stretch(grn_channel, 1, 255))
    log_grn_channel[np.where(~np.isfinite(log_grn_channel))] = 0
    log_grn_channel = mh.stretch(log_grn_channel)
    
    # blu_gauss = mh.stretch(filters.gaussian(blu_channel, sigma=1))
    # log_blu_gauss = np.log(blu_gauss)
    # log_blu_gauss[np.where(~np.isfinite(log_blu_gauss))] = 0
    # log_blu_gauss = mh.stretch(log_blu_gauss)


    # log_blu_channel = np.log(blu_channel)
    # log_blu_channel[np.where(~np.isfinite(log_blu_channel))] = 0
    # log_blu_channel = mh.stretch(log_blu_channel)

    
    
    ### Old method used after Oct 04 2019 until March 28, 2020:
    # Take a local sliding window otsu threshold of blu_channel:
    # flood_fill_blu = mh.stretch(blu_channel) >= (filters.rank.otsu(mh.stretch(blu_channel), selem=morphology.disk(disk_radius*0.5)) * 0.75)
    flood_fill_blu = mh.stretch(blu_channel) > (filters.rank.otsu(mh.stretch(blu_channel), selem=morphology.disk(disk_radius)) * 0.75)
    # Canny edge
    if blu_channel.any():
        canny_blu = feature.canny(blu_channel, low_threshold=filters.threshold_otsu(blu_channel)*0.1)
    else:
        canny_blu = np.zeros(blu_channel.shape, dtype=bool)
    flood_fill_blu += canny_blu
    # Label regions:
    flood_fill_blu_lab = morphology.label(flood_fill_blu)
    # Return median fluorescence of each labeled region:
    flood_fill_blu_fluor_lab = [(flood_fill_blu_lab==x)*np.median(blu_channel[flood_fill_blu_lab==x]) for x in np.unique(flood_fill_blu_lab)]
    flood_fill_blu_fluor_lab = np.add.reduce(flood_fill_blu_fluor_lab)
    
    
    # Keep any regions that are brighter than the otsu threshold of the blu channel:
    # If there is too much noise then set the fluor_cutoff to be 0, this way 
    # everything will be counted as blu. This is counter-intuitive but is done 
    # because it is consistent with the behaviour of otsu thresholding of very
    # low signal images:
    # loc_snr_min_blu, local_snr_blu = get_local_snr(blu_channel, footprint)
    # if loc_snr_min_blu > snr_cutoff:
    #     fluor_cutoff = float(0)
    # else:
    #     fluor_cutoff = filters.threshold_otsu(blu_channel)*0.2
    if np.std(blu_channel) > std_cutoff:
        fluor_cutoff = filters.threshold_otsu(blu_channel)*0.2
    else:
        fluor_cutoff = float(-1)

    flood_fill_blu_thresh = flood_fill_blu_fluor_lab > fluor_cutoff
    
     
    
    # Keep any regions that are dimmer than the otsu threshold of the tl channel:
    # flood_fill_blu_tl_lab = [(flood_fill_blu_lab==x)*np.median(tl_channel[flood_fill_blu_lab==x]) for x in np.unique(flood_fill_blu_lab)]
    # flood_fill_blu_tl_lab = np.add.reduce(flood_fill_blu_tl_lab)
    # tl_cutoff = filters.threshold_otsu(tl_channel)#*0.2
    # flood_fill_blu_thresh_tl = flood_fill_blu_tl_lab < tl_cutoff

    
    # flood_fill_blu_thresh += flood_fill_blu_thresh_tl
    # flood_fill_blu_thresh = morphology.remove_small_objects(flood_fill_blu_thresh, min_size=32)
    
    
    # flood_fill_blu_thresh_fill = fill_caveolae(flood_fill_blu_thresh, min_radius=1, max_radius=1)
    # flood_fill_blu_thresh_fill = fill_caveolae(flood_fill_blu_thresh, min_radius=1, max_radius=9)
    # cav_areas = np.logical_xor(flood_fill_blu_thresh, flood_fill_blu_thresh_fill)
    
    
######### This part above was used up to March 28, 2020.
    
######### New method used March 29, 2020:
    if not(flood_fill_blu_thresh.all()):
        blu_scharr = filters.scharr(pad_img_edges(blu_channel, 0))[1:-1,1:-1]
        edge_map = mh.stretch(red_gauss/red_gauss.max() + blu_scharr/blu_scharr.max())
        # edge_map = mh.stretch(blu_scharr/blu_scharr.max())
        n_seg_list = [int(subdivide_img(tl_channel.shape, x)) for x in range(3,int(disk_radius))]
    
        ws_init_list = list()
        seg_test_list = list()
        for n_seg in n_seg_list:
            ws_init = segmentation.watershed(edge_map, markers=int(n_seg), compactness=0, mask=~flood_fill_blu_thresh)
            frc3, frc2 = fast_region_compete(edge_map, ws_init, iterations=10, min_size=64, mask=~flood_fill_blu_thresh)

            # ws_init = segmentation.watershed(edge_map, markers=int(n_seg), compactness=0)
            # frc3, frc2 = fast_region_compete(edge_map, ws_init, iterations=10, min_size=64)

            # frc3, frc4 = fast_region_compete(edge_map, ws_init, iterations=10, min_size=64, mask=~(blucell+not_bg_red_rid))
            # frc3, frc4 = fast_region_compete(red_channel, ws_init, iterations=10, min_size=64, mask=blucell)
            ws_init_list.append(ws_init)
            # frc1_list.append(frc1)
            seg_test_list.append(frc3)
            # seg_test_list.append(segmentation.join_segmentations(frc1, frc3))
            # seg_test_list.append(segmentation.join_segmentations(frc3, blucell))
        seg_thresh_edges_blu, seg_edge_map_blu = get_edge_thresh(seg_test_list, 0.5)    
        seg_thresh_blu = ~seg_thresh_edges_blu
        seg_thresh_blu_lab = morphology.label(seg_thresh_blu, connectivity=1)
        blu_seg = segmentation.watershed(edge_map, seg_thresh_blu_lab, mask=~flood_fill_blu_thresh)
        # blu_seg = segmentation.watershed(edge_map, seg_thresh_blu_lab)
    else:
        blu_seg = np.zeros(flood_fill_blu_thresh.shape, dtype=int)
        flood_fill_blu_thresh = np.zeros(flood_fill_blu_thresh.shape, dtype=bool) #added 2020-11-16

    
    
    blu_seg_median_lab = [(blu_seg==x) * np.median(blu_channel[blu_seg==x]) for x in np.unique(blu_seg)]
    blu_seg_median_lab = np.add.reduce(blu_seg_median_lab)
    fluor_cutoff = filters.threshold_otsu(blu_channel)*0.2#0.33#*0.2
    blu_seg_median_lab_thresh = blu_seg_median_lab > fluor_cutoff
    
    ### If the signal-to-noise ratio is too low then blu_seg_median_lab
    ### will be entirely all True (or very close to it I think), so
    ### need to set it all to 0:
    blu_px = np.count_nonzero(blu_seg_median_lab_thresh)
    total_px = np.multiply(*blu_seg_median_lab_thresh.shape,)
    # If the total number of pixels that are recognized as blue cell is
    # more than 80% of the picture, then assume that the singnal to noise
    # ratio is too low:
    if blu_px >= (0.80*total_px):
        blu_seg_median_lab_thresh = np.zeros(blu_seg_median_lab_thresh.shape, dtype=bool)
    
    
    norm = lambda x : (x-x.min()) / (x.max() - x.min())

    
        
    blu_seg_median_lab_tl = [(blu_seg==x) * np.median(tl_channel[blu_seg==x]) for x in np.unique(blu_seg)]
    blu_seg_median_lab_tl = np.add.reduce(blu_seg_median_lab_tl)
    blu1 = (blu_seg_median_lab_tl <= filters.threshold_otsu(tl_channel)) * blu_seg.astype(bool)
    
    # return the % of each region in blu1 that abuts flood_fill_blu_thresh:
    blu_seg_tl = blu1 * blu_seg
    blu_seg_touching_flood_fill = join_contig_regions(flood_fill_blu_thresh, blu_seg_tl, percent_cutoff=0.7)
    
    
    
    #####################################################################
    ### The stuff in this block can be deleted if analysis works well:
    ### Line below is old method used up to March 28th, 2020:
    # red_thresh = red_channel >= filters.threshold_otsu(red_channel)
    # blu_seg_median_lab_red = [(blu_seg==x) * np.mean(red_thresh[segmentation.find_boundaries(blu_seg==x)]) for x in np.unique(blu_seg)] 
    # blu_seg_median_lab_red = [(blu_seg==x) * np.max(red_channel[blu_seg==x]) for x in np.unique(blu_seg)] 
    blu_seg_median_lab_red = [(blu_seg==x) * np.median(red_channel[segmentation.find_boundaries(blu_seg==x)]) for x in np.unique(blu_seg)]
    blu_seg_median_lab_red = np.add.reduce(blu_seg_median_lab_red)
    blu2 = (blu_seg_median_lab_red <= filters.threshold_otsu(red_channel)) * blu_seg.astype(bool)
    # blu2 = blu_seg_median_lab_red > 0.55

    ### Line below is old method used up to March 28th, 2020:
    # grn_thresh = grn_channel >= filters.threshold_otsu(grn_channel)
    # blu_seg_median_lab_grn = [(blu_seg==x) * np.mean(grn_thresh[segmentation.find_boundaries(blu_seg==x)]) for x in np.unique(blu_seg)]
    # blu_seg_median_lab_grn = [(blu_seg==x) * np.max(grn_channel[blu_seg==x]) for x in np.unique(blu_seg)]
    blu_seg_median_lab_grn = [(blu_seg==x) * np.median(grn_channel[segmentation.find_boundaries(blu_seg==x)]) for x in np.unique(blu_seg)]
    blu_seg_median_lab_grn = np.add.reduce(blu_seg_median_lab_grn)
    blu3 = (blu_seg_median_lab_grn <= filters.threshold_otsu(grn_channel)) * blu_seg.astype(bool)
    # blu3 = blu_seg_median_lab_grn > 0.55
    ###############################################################


    blu_seg_median_lab_blu = [(blu_seg==x) * np.median(blu_channel[blu_seg==x]) for x in np.unique(blu_seg)]
    blu_seg_median_lab_blu = np.add.reduce(blu_seg_median_lab_blu)
    ### Line below is old method used up to March 31st, 2020:
    blu4 = (blu_seg_median_lab_blu > filters.threshold_otsu(blu_channel)*0.8) * blu_seg.astype(bool)
    ### This version is more generous... might be a problem with hazy signals...
    # blu4 = (blu_seg_median_lab_blu >= filters.threshold_otsu(blu_channel)*0.5) * blu_seg.astype(bool)
    
    #######################################################
    ### Can also delete this stuff:
    # Take the areas that you are sure are not from the red cell:
    not_rg = np.logical_and(blu2, blu3) # old method used before March 29, 2020:
    # and take where they overlap with the ambiguous trans_light regions:
    blu = np.logical_and(blu1, not_rg) # old method used before March 29, 2020:
    # and combine this with the regions that you are sure are blue:
    blu = np.logical_or(blu, blu4) # old method used before March 29, 2020:
    blu_bg = np.add.reduce([blu1, blu2, blu3, blu4], dtype=bool) # old method used before March 29, 2020:
    blu_bg = np.logical_xor(blu_bg, blu) # old method used before March 29, 2020:
    ######################################################

    blu = np.logical_or(blu_seg_touching_flood_fill, blu4)
    blu_bg = ~blu
    # blu_bg = np.logical_xor(blu_seg.astype(bool), blu)
    
    
    blucell_rough = np.logical_or(blu_seg_median_lab_thresh, blu)
    blucell_bad = np.logical_and(blu_seg_median_lab_thresh, blu_bg)
    # blucell_rough = np.logical_xor(blu_seg_median_lab_thresh, blucell_bad)
    blucell_rough = np.logical_xor(blucell_rough, blucell_bad)
   
    

    
    # Close any small openings:
    # blucell_rough = morphology.remove_small_holes(blucell_rough, area_threshold=64)
    # blu_close = np.logical_xor(morphology.closing(blucell_rough, selem=morphology.disk(3)), blucell_rough)
    # blucell_rough = morphology.closing(blucell_rough, selem=morphology.disk(3))
    blucell = morphology.remove_small_holes(blucell_rough, area_threshold=64)
    # blucell = np.logical_xor(blucell, blu_close)
    blucell = morphology.remove_small_objects(blucell, min_size=64)
    ### End of new method.
    # Now we have the blucell areas.
    
    # Take a threshold of the transmitted light channel:
    # tl_thresh = tl_channel <= filters.threshold_otsu(tl_channel)
    # tl_thresh = tl_gauss <= filters.rank.median(tl_gauss, selem=morphology.disk(24))
    # tl_entrop = filters.rank.entropy(tl_channel, selem=morphology.disk(12))
    # tl_canny = feature.canny(tl_channel)
    # tl_dist = mh.stretch(scp_morph.distance_transform_edt(tl_thresh)**2)
    # tl_dist = mh.stretch(mh.distance(tl_thresh))
    # tl_thresh = tl_dist >= filters.threshold_otsu(tl_dist)
    
    # Take a threshold of the lifeact-GFP channel:
    low = filters.threshold_otsu(grn_channel) * 0.5
    high = filters.threshold_otsu(grn_channel)
    grn_thresh = filters.apply_hysteresis_threshold(grn_channel, low=low, high=high)
    
    # The signal in the red channel is membrane-based and can be quite dim at
    # the contact which can be a challenge, but let's try to detect the redcell:
    # Average the information from all 4 channels to get a more robust estimate
    # of cell areas and boundaries:
    # mean_chan = mh.stretch(np.mean([mh.stretch(red_channel),mh.stretch(grn_channel),mh.stretch(blu_channel), mh.stretch(~tl_channel)], axis=0))
    mean_chan = mh.stretch(np.mean([mh.stretch(red_channel),mh.stretch(log_grn_gauss),mh.stretch(blu_channel), mh.stretch(~tl_channel)], axis=0))
    mean_thresh = mean_chan > filters.threshold_otsu(mean_chan)
    #mean_chan = mh.stretch(np.mean([mh.stretch(red_channel),mh.stretch(grn_channel),mh.stretch(blu_channel)], axis=0))

    

    # Use hysteresis threshold of the red channel:
    hyster_thresh_red = mh.stretch(red_channel) > filters.rank.otsu(mh.stretch(red_channel), selem=morphology.disk(5))
    hyster_thresh_grn = mh.stretch(grn_channel) > filters.rank.otsu(mh.stretch(grn_channel), selem=morphology.disk(5))
    # and find local maxima along axis 0 and axis 1 of red channel:
    peak_peak_dist = 3
    red_ridges = find_ridge_peaks(red_channel, axis0_peak_to_peak_dist=peak_peak_dist, axis1_peak_to_peak_dist=peak_peak_dist)
    grn_ridges = find_ridge_peaks(grn_channel, axis0_peak_to_peak_dist=peak_peak_dist, axis1_peak_to_peak_dist=peak_peak_dist)
    # add hyster_thresh and ridges together:
    mean_chan_hyster_ridge = np.logical_or(hyster_thresh_red, red_ridges)
    grn_chan_hyster_ridge = np.logical_or(hyster_thresh_grn, grn_ridges)
    # and add to mean_chan_hyster_ridge:
    mean_chan_overlaps = mean_chan_hyster_ridge + grn_chan_hyster_ridge + blucell
    
    # A test case used in assigning values to labeled regions so that clustering
    # algorithms can sort the regions into "cell" or "not cell":
    #multi_thresh_bool = np.add.reduce([tl_thresh, grn_thresh, blucell, red_ridges, grn_ridges, mean_chan_hyster_ridge, grn_chan_hyster_ridge], dtype=bool)
    #multi_thresh = np.add.reduce([tl_thresh, grn_thresh, blucell, red_ridges, grn_ridges, mean_chan_hyster_ridge, grn_chan_hyster_ridge], dtype=int)
    multi_thresh_bool = np.add.reduce([mean_thresh, grn_thresh, blucell, red_ridges, grn_ridges, mean_chan_hyster_ridge, grn_chan_hyster_ridge], dtype=bool)
    multi_thresh = np.add.reduce([mean_thresh, grn_thresh, blucell, red_ridges, grn_ridges, mean_chan_hyster_ridge, grn_chan_hyster_ridge], dtype=int)
    
    mtb_clean = morphology.remove_small_objects(multi_thresh_bool, min_size=3)
    multi_thresh_bool = morphology.remove_small_holes(mtb_clean, area_threshold=64)




###############################################################################

    
    
    red_ridges_less_noise = morphology.remove_small_objects(red_ridges, min_size=3, connectivity=2)  

    red_rid_dists, good_circles = hough_circle_fit(red_ridges_less_noise,
                                                   disk_radius/2,
                                                   0.998,
                                                   img_to_compare_against=red_ridges)

    ### We're going to use 3 types of features to determine whether or not 
    ### certain regions in the image are worth keeping: the mean intensity of 
    ### the ridges, the length of the ridges, and how closely those ridges fit
    ### to major circles in the image:

    red_ridges_lab = morphology.label(red_ridges)
    # red_rid_dists_normed = red_rid_dists / red_rid_dists.max()
    
    
    red_ridge_props = measure.regionprops(red_ridges_lab, red_channel)
    red_ridge_hough_props = measure.regionprops(red_ridges_lab, red_rid_dists)
    # red_ridge_hough_props = measure.regionprops(red_ridges_lab, red_rid_dists_normed)
    red_rid_means = [x.mean_intensity for x in red_ridge_props]
    #red_rid_medians = [np.median(x.intensity_image) for x in red_ridge_props]
    red_rid_houghs = [np.mean(x.intensity_image) for x in red_ridge_hough_props]
    red_rid_lengths = [x.area for x in red_ridge_props]
    
    
    metrics_means = np.array(red_rid_means, ndmin=2).T
    normed_metrics_means = np.apply_along_axis(norm, 0, metrics_means)
    # If none of the red_rid_means have valid norms then just use the un-normed
    # ones:
    if np.isnan(normed_metrics_means).all():
        normed_metrics_means = metrics_means.copy()
    else:
        pass
    
    red_rid_houghs_labels = np.array([x.label for x in red_ridge_hough_props], ndmin=2).T
    # metrics_houghs = np.array(red_rid_houghs, ndmin=2).T
    # normed_metrics_houghs = np.apply_along_axis(norm, 0, metrics_houghs)
    normed_metrics_houghs = np.array(red_rid_houghs, ndmin=2).T

    red_rid_lengths_labels = np.array([x.label for x in red_ridge_props], ndmin=2).T
    metrics_lengths = np.array(red_rid_lengths, ndmin=2).T
    #normed_metrics_lengths = np.apply_along_axis(norm, 0, metrics_lengths)
    
    
    # if the number of samples is more than 1 then separate samples into 
    # background and foreground:
    if len(normed_metrics_means) > 1:
        red_rid1, red_rid_bg1 = kmeans_binary_sep(red_ridges_lab, normed_metrics_means)
    # but if the number of samples is equal to 1 then the foreground should be obvious
    else:
        red_rid1, red_rid_bg1 = red_ridges_lab.astype(bool), np.zeros(red_ridges_lab.shape, dtype=bool)
    #red_rid2, red_rid_bg2 = kmeans_binary_sep(red_ridges_lab, normed_metrics_lengths)
    #red_rid_bg3, red_rid3 = kmeans_binary_sep(red_ridges_lab, normed_metrics_houghs)
        
    len_threshold = (2 * disk_radius * np.pi) / 10
    red_rid_lengths_threshed_labels = np.unique(red_rid_lengths_labels * (metrics_lengths > len_threshold))[np.unique(red_rid_lengths_labels * (metrics_lengths > len_threshold)) != 0]
    red_rid2 = np.logical_or.reduce([red_ridges_lab==x for x in red_rid_lengths_threshed_labels])

    
    tolerance = 1 # mean number of pixels away from circle that is acceptable
    red_rid_houghs_threshed_labels = np.unique(red_rid_houghs_labels * (normed_metrics_houghs < tolerance))[np.unique(red_rid_houghs_labels * (normed_metrics_houghs < tolerance)) != 0]
    red_rid3 = np.logical_or.reduce([red_ridges_lab==x for x in red_rid_houghs_threshed_labels])
    

    
    red_rid = red_rid1 + red_rid2 + red_rid3
    red_rid_lab = morphology.label(red_rid)
    # red_rid_clean = morphology.remove_small_objects(red_rid_lab, min_size=10, connectivity=2) 
    red_rid_clean = morphology.remove_small_objects(red_rid_lab, min_size=6, connectivity=2) 
    # If a region shows up in more than 50% of the filters then treat it as 
    # real signal:
    not_bg_red_rid = np.logical_and(red_rid_clean, red_rid >= (red_rid.max()*0.5) )
    ### Might need to use the non-cleaned up version for adhering cells if lots
    ### of single dim pixels show up...
    # not_bg_red_rid = np.logical_and(red_rid_clean, red_rid >= (red_rid.max()*0.5) )
    

###############################################################################    

    
    # Close up small breaches in the thresholded edges:
    #sch_bool_blu_close = morphology.closing(sch_bool_blu, selem=morphology.disk(3))
    #sch_bool_blu_close = morphology.closing(sch_bool_blu)
    #mean_chan_overlaps = morphology.closing(mean_chan_overlaps)
    
    # # Fill holes that are present in the thresholded image:
    # #sch_bool_blu_fill = scp_morph.binary_fill_holes(sch_bool_blu_close)
    # mean_chan_fill = scp_morph.binary_fill_holes(mean_chan_overlaps)
    # # Remove small bits and pieces of noise in the thresholded image:
    # #sch_bool_blu_clean = morphology.remove_small_objects(sch_bool_blu_fill, min_size=(np.pi * disk_radius**2) / 3)
    # mean_chan_bool_clean = morphology.remove_small_objects(mean_chan_fill, min_size=(np.pi * disk_radius**2) / 3, connectivity=2)
    # #mean_chan_bool_clean = np.logical_or(sch_bool_blu_clean, mean_chan_bool_clean) Commented out Oct 9, 2019 to test if cell_thresh is less generous.
    # mean_chan_bool_clean = morphology.remove_small_holes(mean_chan_bool_clean, area_threshold=np.pi * (disk_radius)**2, connectivity=1)
    # mean_chan_bool_clean = morphology.remove_small_objects(mean_chan_bool_clean, min_size=(np.pi * disk_radius**2) / 3, connectivity=1)



    # n_segs = subdivide_img(tl_channel.shape, 10)
    #slic = segmentation.slic(np.dstack((tl_channel/tl_channel.max() , red_channel/red_channel.max() , grn_channel/grn_channel.max() , blu_channel/blu_channel.max() )), n_segments=n_segs, compactness=0.1, sigma=0, multichannel=True, enforce_connectivity=True, max_iter=500)
    # slic = segmentation.slic(tl_channel/tl_channel.max() , n_segments=n_segs, compactness=0.1, sigma=0, multichannel=False, enforce_connectivity=True, max_iter=500)# + 1
    # slic_padded = pad_img_edges(slic, 0)
    # slic_dist = [scp_morph.distance_transform_edt(slic_padded == region) for region in np.unique(slic)]
    # slic_peaks = [feature.peak_local_max(dist, indices=False, exclude_border=False) for dist in slic_dist]
    # slic_seeds = np.add.reduce(slic_peaks) * slic_padded
    # slic_seeds = slic_seeds[1:-1, 1:-1]
    
    
    
    # def region_compete(image, seedpoints, iterations=25, mask=None, compactness=0, indices=False, exclude_border=False, min_size=None):
    #     time_start = time.time()
    #     if not(min_size):
    #         min_size = 0
    #     removed_label_list = []
    #     count = 0
        
    #     seg_old = segmentation.watershed(image, seedpoints, mask=mask, compactness=compactness)
    #     seg = morphology.remove_small_objects(seg_old, min_size)
    #     seg_labels = np.unique(seg[seg != 0])
    #     removed_label_list.append(np.unique(seg_old[seg == 0]))
    #     seg_padded = pad_img_edges(seg, 0)
        
    #     dist_list = [scp_morph.distance_transform_edt(seg_padded == region) for region in seg_labels]
    #     peak_list = [feature.peak_local_max(dist, indices=indices, exclude_border=exclude_border) for dist in dist_list]
    #     #seeds = np.add.reduce(peak_list) * seg_padded
    #     seeds = morphology.label(np.add.reduce(peak_list))
    #     seeds = seeds[1:-1, 1:-1]
    #     count += 1
        
    #     while count < iterations:
    #         seg_old = segmentation.watershed(image, seeds, mask=mask, compactness=compactness)
    #         seg = morphology.remove_small_objects(seg_old, min_size)
    #         seg_labels = np.unique(seg[seg != 0])
    #         removed_label_list.append(np.unique(seg_old[seg == 0]))
    #         seg_padded = pad_img_edges(seg, 0)
    #         dist_list = [scp_morph.distance_transform_edt(seg_padded == region) for region in np.unique(seg)]
    #         peak_list = [feature.peak_local_max(dist, indices=indices, exclude_border=exclude_border) for dist in dist_list]
    #         #seeds = np.add.reduce(peak_list) * seg_padded
    #         seeds = morphology.label(np.add.reduce(peak_list))
    #         seeds = seeds[1:-1, 1:-1]
    #         count += 1
    #     time_end = time.time()
    #     elapsed_time = time_end - time_start
    #     print("Elapsed time: %.3f"%elapsed_time)
    #     return seg_old, seeds, removed_label_list


    ### The chunk below will take watershed segmentations with a variety of 
    ### number of markers. This basically allows the watershed to be done with
    ### multiple scales of object size being considered.
    n_seg_list = [int(subdivide_img(tl_channel.shape, x)) for x in range(3,int(disk_radius))]
    ws_init_list = list()
    # frc1_list = list()
    frc3_list = list()
    seg_test_list = list()
    for n_seg in n_seg_list:
        ws_init = segmentation.watershed(red_channel, markers=int(n_seg), compactness=0)
        # frc1, frc2 = fast_region_compete(red_channel, ws_init, iterations=10, min_size=64)#, mask=blucell)
        frc3, frc4 = fast_region_compete(red_channel, ws_init, iterations=10, min_size=64, mask=~(blucell+not_bg_red_rid))
        # frc3, frc4 = fast_region_compete(red_channel, ws_init, iterations=10, min_size=64, mask=blucell)
        ws_init_list.append(ws_init)
        # frc1_list.append(frc1)
        frc3_list.append(frc3)
        # seg_test_list.append(segmentation.join_segmentations(frc1, frc3))
        seg_test_list.append(segmentation.join_segmentations(frc3, blucell))
    # seg_test_list = frc3_list
    

    ### All real boundaries should show up frequently, and some fake boundaries
    ### may show up frequently too (but hopefully less frequently) when doing a
    ### max project of all the many segmentations. Take the % most common ones 
    ### and repeat the watershed to "combine" super pixels into larger (still 
    ### accurate) super pixels. This is important because when giving each 
    ### super pixel properties if it is too small it is prone to error and 
    ### being mistakenly classified as bg instead of fg or vice-versa:
    # seg_thresh_edges, seg_edge_map = get_edge_thresh(seg_test_list, 0.65)
    seg_thresh_edges, seg_edge_map = get_edge_thresh(seg_test_list, 0.55)
    seg_thresh = ~seg_thresh_edges
    seg_thresh_labeled = morphology.label(seg_thresh, connectivity=1)

    
    ### Expected tricellular gap area between 3 cells that touch each other at
    ### a point but remain perfectly spherical:
    def get_expect_tricell_gap_area(radius):
        area_expected = radius**2 * (np.sqrt(3) - (np.pi / 2))
        return area_expected
    
    expect_gap_size = get_expect_tricell_gap_area(disk_radius/2)
    
    frc1, frc2 = fast_region_compete(red_channel, seg_thresh_labeled, iterations=10, min_size=expect_gap_size, mask=~(blucell+morphology.remove_small_holes(not_bg_red_rid)))
    # frc3, frc4 = fast_region_compete(red_channel, seg_thresh_labeled, iterations=10, min_size=64, mask=~blucell)
    frc3, frc4 = fast_region_compete(red_channel, seg_thresh_labeled, iterations=10, min_size=expect_gap_size, mask=blucell)
    seg_test_list = segmentation.join_segmentations(frc1, frc3)
    
        
    #######
    ### a test:
    seg_thresh_labeled = morphology.remove_small_objects(seg_thresh_labeled, min_size=expect_gap_size)
    seg_thresh_labeled_pad = pad_img_edges(seg_thresh_labeled, 0, 1)
    dist = [mh.distance(seg_thresh_labeled_pad == x) for x in np.unique(seg_thresh_labeled_pad)[np.nonzero(np.unique(seg_thresh_labeled_pad))]]
    pks = [feature.peak_local_max(x, indices=False, exclude_border=False, footprint=morphology.disk(disk_radius/2)) for x in dist]
    if pks and dist:
        pks = np.add.reduce(pks)[1:-1,1:-1]
        dist = np.add.reduce(dist)[1:-1,1:-1]
    else:
        pks = np.zeros(seg_thresh_labeled.shape, dtype=int)
        dist = np.zeros(seg_thresh_labeled.shape, dtype=int)
    pks_lab = seg_thresh_labeled * (np.logical_or(pks, blucell))
    blu_scharr = filters.scharr(pad_img_edges(blu_channel, 0))[1:-1,1:-1]
    # edge_map = mh.stretch(red_channel/red_channel.max() + blu_scharr/blu_scharr.max())
    edge_map = mh.stretch(red_gauss/red_gauss.max() + blu_scharr/blu_scharr.max())
    # seg = segmentation.watershed(edge_map, pks_lab, compactness=0.1)
    seg = segmentation.watershed(edge_map, seg_thresh_labeled, compactness=0.1)
    frc = fast_region_compete(edge_map, seg, iterations=10, min_size=expect_gap_size)
    seg_test_list = frc[0]
    #######

    ### Good, keep:
    seg_test_rough = np.logical_or(seg_test_list, morphology.remove_small_holes(not_bg_red_rid))
    seg_test_list[~seg_test_rough] = -1
    seg_test_list = segmentation.watershed(red_channel, seg_test_list)
    seg_test_list[seg_test_list == -1] = 0
    ###
    
    #######
    ### a test:
    # blu_mean_lab = [(seg_test_list==x) * np.mean(blucell[seg_test_list==x])
    #                 for x in np.unique(seg_test_list)]
    blu_mean_lab = [(seg_test_list==x) * np.median(blucell[seg_test_list==x])
                for x in np.unique(seg_test_list)]

    blu_mean_lab = np.add.reduce(blu_mean_lab)
    seg_test_blucell = blu_mean_lab > 0.5

    # blu_mean_lab = [(seg_test_list==x) * np.mean(bluchan[seg_test_list==x])
    #             for x in np.unique(seg_test_list)]
    # blu_mean_lab = np.add.reduce(blu_mean_lab)
    seg_test_bluchan = blu_mean_lab > filters.threshold_otsu(bluchan)*0.2
    seg_test_blu = np.logical_or(seg_test_blucell, seg_test_bluchan)
    seg_blu_dist = mh.distance(pad_img_edges(seg_test_blu, 0))[1:-1, 1:-1] # This was commented out before April 8 in favor of line below
    # seg_blu_dist = mh.distance(seg_test_blu, 0)
    seg_blu_pks = feature.peak_local_max(seg_blu_dist, indices=False, exclude_border=False, footprint=morphology.disk(disk_radius/2))
    seg_blu_pks = avg_too_close_peaks(seg_blu_pks, footprint=morphology.disk(disk_radius/2))
    seg_blu_pks = morphology.label(seg_blu_pks)
    edges_blu = ~mh.stretch(seg_blu_dist)
    # edges_blu = ~mh.stretch(seg_blu_dist)/(~mh.stretch(seg_blu_dist)).max() * blu_scharr
    # edges_blu = blu_scharr.copy()
    seg_test_blu = segmentation.watershed(edges_blu, seg_blu_pks, mask=seg_test_blu, compactness=0)
    #######
                                                 
                                                 
    # seg_test_list = frc1
    #plt.imshow(seg_test_list)
    # frc1, frc2 = fast_region_compete(red_channel, seg_test_list, iterations=10, min_size=64, mask=~blucell)
    # frc3, frc4 = fast_region_compete(red_channel, seg_test_list, iterations=10, min_size=64)
    
    
    # seg_test_list * blucell
    # seg_mean_lab = [(frc1==x) * np.mean(blucell.astype(float)[frc1==x]) for x in np.unique(frc1)]
    # seg_mean_lab = np.add.reduce(seg_mean_lab)
    # cutoff = 0.5
    # seg_mean_lab_thresh = seg_mean_lab > cutoff
    # seg_mean_lab_thresh *= blucell


    ### clean up the multi_thresh_bool areas by trimming off regions that were
    ### too generous using seg_test_list as a sort of "cookie cutter":
    mtb_lab_cut = np.logical_xor(multi_thresh_bool, 
                             segmentation.find_boundaries(seg_test_list)
                             ) * seg_test_list
    seg_test_list_cut = np.logical_xor(seg_test_list,
                                       segmentation.find_boundaries(seg_test_list)
                                       ) * seg_test_list
    ### Do morphological opening on mtb_lab_cut:
    mtb_lab_cut_clean = morphology.opening(mtb_lab_cut)
    ### Remove small regions that persist still in mtb_lab_cut_clean:
    mtb_lab_cut_cleaner = morphology.remove_small_objects(mtb_lab_cut_clean,
                                                          min_size=64)
    
    # frc1, frc2 = fast_region_compete(red_channel, mtb_lab_cut_cleaner, iterations=10, min_size=64, mask=blucell)
    # frc3, frc4 = fast_region_compete(red_channel, mtb_lab_cut_cleaner, iterations=10, min_size=64, mask=~blucell)
    # seg_test_list = segmentation.join_segmentations(frc1, frc3)

    
    multi_thresh_bool = mtb_lab_cut_cleaner.astype(bool)
    
    ### If the blue cell becomes over-fragmented the mtb_lab_cut_cleaner step
    ### can lead to loss of pieces of the blucell which results in poor
    ### segmentation of the blue cell. To fix, add back any regions of 
    ### mtb_lab_cut that are also in blucell:
    # multi_thresh_bool += (mtb_lab_cut * blucell).astype(bool)
    
    
    # ws_init = segmentation.watershed(red_channel, markers=int(n_segs), compactness=0)
    # frc1, frc2 = fast_region_compete(red_channel, ws_init, iterations=10, min_size=64, mask=~blucell)
    # frc3, frc4 = fast_region_compete(red_channel, ws_init, iterations=10, min_size=64, mask=blucell)
    # seg_test_list = segmentation.join_segmentations(frc1, frc3)



    def get_nrl(labeled_img, connectivity=1):
        """ Get Neighbouring Region Labels: \n Returns connectivity array that 
        associates each region label with a set of neighbouring region labels.
        """
        
        # If the array contains any 0 values as labels (or a perhaps because
        # a mask was applied when labeling the image) then convert those to -1
        # as 0 values will be introduced when finding boundaries and need to be
        # subsequently removed:
        if np.any(labeled_img == 0):
            labeled_img[labeled_img == 0] = -1
            
        # Create a dictionary of labels and their neighbouring regions:
        nrl = dict()
        for l in np.unique(labeled_img):
            # Assign each label of a region an array containing the labels of 
            # it's neighbours, but not its own label:
            nrl[l] = np.unique(segmentation.find_boundaries(labeled_img == l, connectivity=connectivity) * labeled_img * ~(labeled_img == l))
            
            # Remove 0 values from neighbour labels:
            nrl[l] = nrl[l][nrl[l] != 0]
            
        # Iterate through the dictionary again and return -1 values to their
        # their former value of 0:
        for l in nrl:
            nrl[l][nrl[l] == -1] = 0
        
        # To complete the return of -1 values to their former value of 0, 
        # replace the -1 key with 0, if there were any to start with:
        if -1 in nrl:
            nrl[0] = nrl.pop(-1)
        # also switch -1 back to 0 in the labeled_img:
        if np.any(labeled_img == -1):
            labeled_img[labeled_img == -1] = 0

        # Create a list of the region labels:
        labeled_img_labels = np.unique(labeled_img)
        nrl_indices = {lab:i for i,lab in enumerate(labeled_img_labels)}

        nrl_indices_dict = {nrl_indices[lab]:np.asarray([nrl_indices[connected_lab] for connected_lab in nrl[lab]]) for lab in nrl}
        
        connectivity_arr = np.zeros((len(nrl_indices_dict), len(nrl_indices_dict)), dtype=bool)

        for l in nrl_indices_dict:
            if nrl_indices_dict[l].any():
                connectivity_arr[l, nrl_indices_dict[l]] = True
                
        
        return connectivity_arr, labeled_img_labels


    # tl_region_props = measure.regionprops(label_image = seg_test_list[0], 
    #                                       intensity_image = tl_channel)
    # red_region_props = measure.regionprops(label_image = seg_test_list[0], 
    #                                        intensity_image = red_channel)
    # grn_region_props = measure.regionprops(label_image = seg_test_list[0], 
    #                                        intensity_image = grn_channel)
    # blu_region_props = measure.regionprops(label_image = seg_test_list[0],
    #                                        intensity_image = blu_channel)
    # multithresh_region_props = measure.regionprops(label_image = seg_test_list[0], 
    #                                                intensity_image = multi_thresh)
    # multithresh_borders_region_props = measure.regionprops(label_image = seg_test_list[0])
    # mtb_region_props = measure.regionprops(label_image = seg_test_list[0],
    #                                        intensity_image = multi_thresh_bool)
    
    
    
    # tl_orient = [x.orientation for x in tl_region_props]
    # tl_ite = [x.inertia_tensor_eigvals for x in tl_region_props]
    # tl_entropy = [measure.shannon_entropy(tl_channel[seg_test_list[0] == label]) for label in np.unique(seg_test_list[0])]
    # ## Note to self: region_props does not count the label '0' when determining
    # ## properties of labeled regions (note that the top blue cell has a label 
    # ## of '0' and I believe that the slic labeling method produces '0' kabels 
    # ## as well).
    # ## Also, implementing an algorithm that determins contour curvature relative
    # ## to the labeled regions centroid would probably be my best best for
    # ## separating cells from background at region where the borders of each
    # ## touch...
    # mt_mean = [np.mean(multi_thresh[seg_test_list[0] == label]) for label in np.unique(seg_test_list[0])]
    # mt_median = [np.median(multi_thresh[seg_test_list[0] == label]) for label in np.unique(seg_test_list[0])]
    # mt_std = [np.std(multi_thresh[seg_test_list[0] == label]) for label in np.unique(seg_test_list[0])]
    # mt_labels = [label for label in np.unique(seg_test_list[0])]
    # #plt.scatter(mt_mean, mt_std)
    # #[plt.annotate(mt_labels[i], (mt_mean[i], mt_std[i])) for i in range(len(mt_labels))]
    
    
    
    corners = [feature.corner_peaks(feature.corner_harris(seg_test_list == label, k=0.15), indices=False, exclude_border=False) for label in np.unique(seg_test_list)[np.nonzero(np.unique(seg_test_list))] ]
    corners_len = np.array([np.count_nonzero(x) for x in corners])

    #mtb_std = [np.std(multi_thresh_bool[seg_test_list == label]) for label in np.unique(seg_test_list)[np.nonzero(np.unique(seg_test_list))] ]
    mtb_mean = [np.mean(multi_thresh_bool[seg_test_list == label]) for label in np.unique(seg_test_list)[np.nonzero(np.unique(seg_test_list))] ]
    #mtb_area = [np.count_nonzero(multi_thresh_bool[seg_test_list == label]) for label in np.unique(seg_test_list)[np.nonzero(np.unique(seg_test_list))] ]
    #mtb_entropy = [measure.shannon_entropy(multi_thresh_bool[seg_test_list == label]) for label in np.unique(seg_test_list)[np.nonzero(np.unique(seg_test_list))] ]
    #mtb_ellipse_model = [measure.EllipseModel(mh.borders(seg_test_list[0] == label)) for label in np.unique(seg_test_list[0])[np.nonzero(np.unique(seg_test_list[0]))] ]
    #mtb_coords = [x.coords for x in mtb_region_props]
    mtb_labels = np.unique(seg_test_list)[np.nonzero(np.unique(seg_test_list))]
    
    ###########################################################################
    corners = [feature.corner_peaks(feature.corner_harris(seg_test_list == label, k=0.15), indices=False, exclude_border=False) for label in np.unique(seg_test_list)[np.nonzero(np.unique(seg_test_list))] ]
    corners_len = np.array([np.count_nonzero(x) for x in corners])
    mtb_mean = [np.mean(multi_thresh_bool[seg_test_list_cut == label]) for label in np.unique(seg_test_list_cut)[np.nonzero(np.unique(seg_test_list_cut))] ]
    mtb_labels = np.unique(seg_test_list_cut)[np.nonzero(np.unique(seg_test_list_cut))]
    ### seg_test_list and seg_test_list_cut should have the same number of labels
    ###########################################################################
    
    
    
    
    #mtb_labels = [x.label for x in mtb_region_props]
    #mtb_mean = [x.mean_intensity for x in mtb_region_props]
    # mtb_hu_moments = [x.weighted_moments_hu for x in mtb_region_props]
    # mtb_hu0 = [x[0] for x in mtb_hu_moments]
    # mtb_hu1 = [x[1] for x in mtb_hu_moments]
    # mtb_hu2 = [x[2] for x in mtb_hu_moments]
    # mtb_hu3 = [x[3] for x in mtb_hu_moments]
    # mtb_hu4 = [x[4] for x in mtb_hu_moments]
    # mtb_hu5 = [x[5] for x in mtb_hu_moments]
    # mtb_hu6 = [x[6] for x in mtb_hu_moments]
    
    # corr_fact = 1.0 # a correction factor, apparently you can use 0.95 if you want to account for pixelization, but for relative values it doesn't matter.
    # mtb_circularity = [(4 * np.pi * x.area) / ((x.perimeter*corr_fact)**2) for x in mtb_region_props]
    
    
    
    
    #plt.scatter(mtb_mean, corners_len)
    #[plt.annotate(mtb_labels[i], (mtb_mean[i], corners_len[i])) for i in range(len(mtb_labels))]


    
    #metrics = [np.row_stack([(np.median(tl_channel[np.where(seg_sample==label)]), np.std(tl_channel[np.where(seg_sample==label)]), regionprops((seg_sample==label)*seg_sample)[0].eccentricity) for label in np.unique(seg_sample)[np.nonzero(np.unique(seg_sample))]]) for seg_sample in seg_test_list]
    #metrics = [np.row_stack([( np.median(log_grn_gauss[np.where(seg_test_list[i]==label)]), np.std(log_grn_gauss[ np.where(seg_test_list[i]==label)])) for label in np.unique(seg_test_list[i])[np.nonzero(np.unique(seg_test_list[i]))]]) for i in range(len(seg_test_list))]
    #metrics = [np.row_stack([( np.median(grn_channel[np.where(seg_test_list[i]==label)]), np.std(grn_channel[np.where(seg_test_list[i]==label)])) for label in np.unique(seg_test_list[i])[np.nonzero(np.unique(seg_test_list[i]))]]) for i in range(len(seg_test_list))]
    #metrics = [np.row_stack([(np.mean(multi_thresh_bool[np.where(seg_test_list[i]==label)]), np.std(multi_thresh_bool[np.where(seg_test_list[i]==label)])) for label in np.unique(seg_test_list[i])[np.nonzero(np.unique(seg_test_list[i]))]]) for i in range(len(seg_test_list))]
    #metrics = [np.row_stack([(np.mean(multi_thresh_bool[np.where(seg_test_list[i]==label)]), np.median(multi_thresh_bool[np.where(seg_test_list[i]==label)])) for label in np.unique(seg_test_list[i])[np.nonzero(np.unique(seg_test_list[i]))]]) for i in range(len(seg_test_list))]
    #metrics = [np.row_stack([(np.mean(multi_thresh_bool[np.where(seg_test_list[i]==label)])) for label in np.unique(seg_test_list[i])[np.nonzero(np.unique(seg_test_list[i]))]]) for i in range(len(seg_test_list))]
    #metrics = np.column_stack([mtb_mean, corners_len])
    normed_metrics = np.array(mtb_mean, ndmin=2).T

    # normed_metrics = np.apply_along_axis(norm, 0, metrics)
    connectivity_arr, connectivity_labels = get_nrl(seg_test_list)
    
    
    
    




    # seg_test_list_boundaries = segmentation.find_boundaries(seg_test_list)
    # not_bg_hard_boundaries = ~np.logical_or.reduce(bg_hard) * seg_test_list_boundaries
    # dists, circles = hough_circle_fit(not_bg_hard_boundaries,
    #                                   disk_radius/2,
    #                                   0.998,
    #                                   seg_test_list_boundaries)

    # tri = [seg_test_list == label for label in mtb_labels[corners_len == 3]]
    # sqr = [seg_test_list == label for label in mtb_labels[corners_len == 4]]
    # amr = [seg_test_list == label for label in mtb_labels[corners_len > 4]]
    
    
    ### End of test.
   
    
    

    
    
    # def is_cell(labeled_img_group1, labeled_img_group2):
    #     tl_std_group1 = np.std(tl_channel[np.where(labeled_img_group1)])
    #     tl_std_group2 = np.std(tl_channel[np.where(labeled_img_group2)])
        
    #     fluor_grn_group1 = np.mean(grn_channel[np.where(labeled_img_group1)])
    #     fluor_grn_group2 = np.mean(grn_channel[np.where(labeled_img_group2)])
        
    #     fluor_red_group1 = np.mean(red_channel[np.where(labeled_img_group1)])
    #     fluor_red_group2 = np.mean(red_channel[np.where(labeled_img_group2)])
        
    #     group1 = np.array([tl_std_group1, fluor_grn_group1, fluor_red_group1])
    #     group2 = np.array([tl_std_group2, fluor_grn_group2, fluor_red_group2])
    
    #     # Assume that group1 is not the background. Then atleast 2 of the 3
    #     # parameters should be greater than those of group 2:
    #     assume_group1 = group1 > group2
        
    #     if np.count_nonzero(assume_group1) >= 2:
    #         not_bg = labeled_img_group1.copy()
    #     else:
    #         not_bg = labeled_img_group2.copy()
            
    #     return not_bg



    ### Test:
    if seg_test_list.any():
        #bg_hard = np.logical_or.reduce([seg_test_list == label for label in mtb_labels[normed_metrics[:,0] < 0.4]])
        fg_hard = np.logical_or.reduce([seg_test_list == label for label in mtb_labels[normed_metrics[:,0] >= 0.6]])
        #bg_soft = np.logical_or.reduce([seg_test_list == label for label in mtb_labels[normed_metrics[:,0] < 0.5]])
        fg_soft = np.logical_or.reduce([seg_test_list == label for label in mtb_labels[normed_metrics[:,0] >= 0.5]])
        #uncertain_regions = np.logical_and(~fg_hard, ~bg_hard)
        #kmeans_cells, kmeans_bg = fg_hard.copy(), ~fg_hard.copy()
        kmeans_cells, kmeans_bg = fg_soft.copy(), ~fg_soft.copy()
        if (kmeans_cells == False).all():
            ### The reason that this if statement is needed is because if
            ### 'label for label in list' used in fg_soft is an empty list then
            ### a single np.bool_ False value will be returned, and we want to
            ### ensure that kmeans cells has the same shape as our image.
            kmeans_cells, kmeans_bg = np.zeros(seg_test_list.shape, dtype=bool), np.zeros(seg_test_list.shape, dtype=bool)
            kmeans_bg[:] = True
        else:
            pass
    else:
        kmeans_cells, kmeans_bg = np.zeros(seg_test_list.shape, dtype=bool), np.zeros(seg_test_list.shape, dtype=bool)
        kmeans_bg[:] = True
    ### End of test.

    
    # ### The uncommented chunk below scatterplots the kmeans_cells and kmeans_bg
    # ### features in different colors:
    # classified_cell = np.unique(kmeans_cells * seg_test_list)[np.unique(kmeans_cells * seg_test_list) != 0]
    # classified_cell_idx = [np.where(x == mtb_labels) for x in classified_cell]
    # scl,xcl,ycl = list(zip(*[(mtb_labels[int(np.asarray(i))], mtb_mean[int(np.asarray(i))], corners_len[int(np.asarray(i))]) for i in classified_cell_idx]))
    # classified_bg = np.unique(kmeans_bg * seg_test_list)[np.unique(kmeans_bg * seg_test_list) != 0]
    # classified_bg_idx = [np.where(x == mtb_labels) for x in classified_bg]
    # sbg,xbg,ybg = list(zip(*[(mtb_labels[int(np.asarray(i))], mtb_mean[int(np.asarray(i))], corners_len[int(np.asarray(i))]) for i in classified_bg_idx]))
    # plt.scatter(xcl, ycl)
    # plt.scatter(xbg, ybg)
    # [plt.annotate(scl[i], (xcl[i], ycl[i])) for i in range(len(classified_cell_idx))]
    # [plt.annotate(sbg[i], (xbg[i], ybg[i])) for i in range(len(classified_bg_idx))]

    
    
    ### Split the kmeans_cells into separate regions using the red_ridges as a 
    ### sort of "cookie cutter" (this is to try and consolidate the many 
    ### labeled regions that make up kmeans_cells in to just 1 region for each
    ### cell):
    #kmc_and_redrid = np.logical_and(kmeans_cells, red_ridges)
    kmc_and_redrid = np.logical_and(kmeans_cells, not_bg_red_rid)
    kmc_cut = np.logical_xor(kmeans_cells, kmc_and_redrid)
    kmc_cut_open = morphology.opening(kmc_cut, selem=morphology.disk(3))
    # kmc_cut_open = kmc_cut_open * ~(segmentation.find_boundaries(blucell)*blucell)
    kmc_cut_open = kmc_cut_open * ~(segmentation.find_boundaries(seg_test_blu)*seg_test_blu.astype(bool))
    kmc_labeled = morphology.label(kmc_cut_open, connectivity=1)
    kmc_labeled = morphology.remove_small_objects(kmc_labeled, min_size=10)
    kmc_refined = segmentation.watershed(red_channel, kmc_labeled, mask=kmeans_cells)
    kmc_refined = morphology.remove_small_objects(kmc_refined, min_size=64)
    kmc_refined = segmentation.watershed(red_channel, kmc_refined, mask=kmeans_cells)
    #plt.imshow(kmc_labeled)
    #plt.imshow(kmc_refined)

    
    ### split up the  kmc_v_refined regions:
    kmc_v_refined_pad = pad_img_edges(kmc_refined, 0)
    kmc_v_refined_dist = [mh.distance(kmc_v_refined_pad == x) for x in np.unique(kmc_v_refined_pad)[np.nonzero(np.unique(kmc_v_refined_pad))]]
    kmc_v_refined_pks = [feature.peak_local_max(x, indices=False, exclude_border=False, footprint=morphology.disk(disk_radius/2)) for x in kmc_v_refined_dist]
    if kmc_v_refined_pks and kmc_v_refined_dist:
        kmc_v_refined_pks = np.add.reduce(kmc_v_refined_pks)
        kmc_v_refined_dist = np.add.reduce(kmc_v_refined_dist)
    else:
        kmc_v_refined_pks = np.zeros(kmc_v_refined_pad.shape, dtype=int)
        kmc_v_refined_dist = np.zeros(kmc_v_refined_pad.shape, dtype=float)
    # kmc_v_refined_pks = feature.peak_local_max(kmc_v_refined_dist, indices=False, exclude_border=False, footprint=morphology.disk(disk_radius/2))
    count = 0
    iterations = 2
    while count < iterations:
        kmc_v_refined_pks = avg_too_close_peaks(kmc_v_refined_pks, 
                                                footprint=morphology.disk(int(disk_radius*0.67)))
        ### footprint was originally morphology.disk(int(disk_radius/3), but
        ### was not aggressive enough.
        count += 1
    kmc_v_refined_pks = kmc_v_refined_pks[1:-1, 1:-1]
    kmc_v_refined_pks = morphology.label(kmc_v_refined_pks.astype(int))
    
    # basins = (~mh.stretch(kmc_v_refined_dist[1:-1, 1:-1])/2) + (mh.stretch(redchan)/2)
    if kmc_v_refined_dist.any():
        dist = (kmc_v_refined_dist[1:-1, 1:-1].max() - kmc_v_refined_dist[1:-1, 1:-1])/kmc_v_refined_dist[1:-1, 1:-1].max()
    else:
        dist = np.zeros(kmc_v_refined_dist[1:-1,1:-1].shape, dtype=float)
    
    if seg_test_blu.any():
        blu_edges = filters.gaussian(segmentation.find_boundaries(seg_test_blu), sigma=2)/filters.gaussian(segmentation.find_boundaries(seg_test_blu), sigma=2).max()
    else:
        blu_edges = np.zeros(seg_test_blu.shape, dtype=float)

    ridge = red_channel / red_channel.max()        
    basins = (dist/3 + ridge/3 + blu_edges/3)
    kmc_v_refined = segmentation.watershed(basins, kmc_v_refined_pks, mask=kmc_refined.astype(bool))
 
    
    
    #red_props = measure.regionprops(label_image=segmentation.find_boundaries(kmeans_cells_seg)*kmeans_cells_seg, intensity_image=red_channel)
    #grn_props = measure.regionprops(label_image=segmentation.find_boundaries(kmeans_cells_seg)*kmeans_cells_seg, intensity_image=grn_channel)
    #blu_props = measure.regionprops(label_image=kmeans_cells_seg, intensity_image=blu_channel)
    
    #logred_props = measure.regionprops(label_image=segmentation.find_boundaries(kmeans_cells_seg)*kmeans_cells_seg, intensity_image=log_red_channel)
    #loggrn_props = measure.regionprops(label_image=segmentation.find_boundaries(kmeans_cells_seg)*kmeans_cells_seg, intensity_image=log_grn_channel)
    #logblu_props = measure.regionprops(label_image=kmeans_cells_seg, intensity_image=log_blu_channel)    
    
    blucell_props = measure.regionprops(label_image=kmc_v_refined, intensity_image=blucell)

    ### You cant do binary separation on images that only have 1 non-zero 
    ### region, therefore include this as a checkpoint, and if it's not met 
    ### then make kmeans_blu all False:
    unique_regions_chkpt = np.count_nonzero(np.unique(kmc_v_refined)) >= 2
    # if blucell.any() and kmc_v_refined.any():
    if blucell.any():# and unique_regions_chkpt:
        kmeans_blu = [(kmc_v_refined==x)*np.mean(blu_channel[kmc_v_refined==x]) for x in np.unique(kmc_v_refined)]
        kmeans_blu = np.add.reduce(kmeans_blu)
        fluor_cutoff = filters.threshold_otsu(blu_channel)*0.2
        kmeans_blu = kmeans_blu > fluor_cutoff
        kmeans_other = np.logical_xor(kmeans_blu, kmc_v_refined)
        # kmeans_blu, kmeans_other = kmeans_binary_sep(kmc_v_refined, np.array([x.mean_intensity for x in blucell_props], ndmin=2).T)
    else:
        kmeans_blu, kmeans_other = np.zeros(kmc_v_refined.shape, dtype=bool), np.zeros(kmc_v_refined.shape, dtype=bool)
        kmeans_other[:] = True
        
    if kmeans_blu.any():
        kmeans_blu = kmc_v_refined * kmeans_blu
    else:
        kmeans_blu = np.zeros(kmc_v_refined.shape, dtype=kmc_v_refined.dtype)
    kmeans_other = kmc_v_refined * kmeans_other
 
    kmc_props = measure.regionprops(label_image=kmeans_other)
    
    ### Assign areas that are too small (eg. (np.pi*disk_radius**2)/5 ) to be bg
    ### regions when sorting the kmc_v_refined regions
    size_limit = (np.pi*disk_radius**2)/5
    kmeans_bg_lab = [(x.area < size_limit) * x.label for x in kmc_props]
    kmeans_bg_lab = [x for x in kmeans_bg_lab if x != 0]
    kmeans_bg = [(kmc_v_refined == x) * x for x in kmeans_bg_lab]
    if kmeans_bg:
        kmeans_bg = np.add.reduce(kmeans_bg, dtype=int)
    else:
        kmeans_bg = np.zeros(kmeans_other.shape, dtype=int)
    
    kmc_v_refined = np.logical_xor(kmeans_bg, kmc_v_refined, dtype=int) * kmc_v_refined


    if kmeans_other.any():
        kmeans_red = kmeans_other.astype(bool) * kmc_v_refined
    else:
        kmeans_red = np.zeros(kmc_v_refined.shape, dtype=kmc_v_refined.dtype)
    
    # kmr_mtb_mean = [np.mean(multi_thresh_bool[kmeans_red == label]) for label in np.unique(kmeans_red)[np.nonzero(np.unique(kmeans_red))] ]
    # kmr_mtb_labels = np.unique(kmeans_red)[np.nonzero(np.unique(kmeans_red))]

    # cutoff = 0.5
    # good_red = kmr_mtb_labels[kmr_mtb_mean > cutoff]
    # kmeans_red = np.logical_or.reduce([kmeans_red == x for x in good_red]).astype(int)
    # kmeans_red *= kmc_v_refined

###
    # km_mtb_mean = [np.mean(multi_thresh_bool[kmeans_red == label]) for label in np.unique(kmeans_red)[np.nonzero(np.unique(kmeans_red))] ]
    # km_mtb_labels = np.unique(kmeans_red)[np.nonzero(np.unique(kmeans_red))]

    # cutoff = 0.5
    # good_red = kmr_mtb_labels[kmr_mtb_mean_norm > cutoff]
    # kmeans_red = np.logical_or.reduce([kmeans_red == x for x in good_red]).astype(int)
    # kmeans_red *= kmc_v_refined
###


    # Find cell centres for seedpoints
    kmeans_red_props = measure.regionprops(label_image=kmeans_red)
    kmeans_blu_props = measure.regionprops(label_image=kmeans_blu)

    kmeans_red_centres = [np.round(x.centroid).astype(int) for x in kmeans_red_props]
    kmeans_red_spots = np.zeros(kmeans_red.shape, dtype=kmeans_red.dtype)
    kmeans_red_spots[tuple(np.asarray(kmeans_red_centres).T)] = 1
    kmeans_red_spots *= kmeans_red
    
    kmeans_blu_centres = [np.round(x.centroid).astype(int) for x in kmeans_blu_props]
    kmeans_blu_spots = np.zeros(kmeans_blu.shape, dtype=kmeans_blu.dtype)
    kmeans_blu_spots[tuple(np.asarray(kmeans_blu_centres).T)] = 1
    kmeans_blu_spots *= kmeans_blu
    
    #kmeans_bg_dist = mh.distance(kmeans_bg)
    #kmeans_bg_spots = feature.peak_local_max(kmeans_bg_dist, footprint=morphology.disk(disk_radius/2), exclude_border=False, indices=False).astype(int)
    #kmeans_bg_labeled = morphology.label(kmeans_bg)
    kmeans_bg_labeled = morphology.label(kmc_v_refined==0)
    kmeans_bg_dist = [mh.distance(kmeans_bg_labeled==label) for label in 
                      np.nonzero(np.unique(kmeans_bg_labeled))[0] ]
    kmeans_bg_spots = [feature.peak_local_max(x, footprint=footprint, 
                                              exclude_border=False, indices=False).astype(int) 
                       for x in kmeans_bg_dist]
    count = 0
    iterations = 2
    while count < iterations :
    # Iterate twice to capture peaks that are averaged closer together:
        kmeans_bg_spots = [avg_too_close_peaks(x, 
                                               footprint=morphology.disk(int(disk_radius/3)))
                           for x in kmeans_bg_spots]
        count += 1
    kmeans_bg_spots = np.logical_or.reduce(kmeans_bg_spots).astype(int)
    # Use binary image to exclude background seeds that might end up
    # overlapping cell areas:
    kmeans_bg_spots *= kmeans_bg_labeled.astype(bool)
    # bg_max has all the same label, make that label start after the red
    # and blu labels numbers:
    kmeans_bg_spots += kmc_v_refined.max()
    # Don't forget to make the unused areas 0:
    kmeans_bg_spots[np.where(kmeans_bg_spots == kmc_v_refined.max())] = 0
    # Now we have background seed points.

    
    # Create a composite of all of the seedpoints:
    kmeans_all_spots = kmeans_red_spots + kmeans_blu_spots + kmeans_bg_spots
    

    
    return kmeans_all_spots, kmeans_red_spots, kmeans_blu_spots, kmeans_bg_spots, kmeans_red, kmeans_blu, kmc_v_refined.astype(bool), kmeans_bg_labeled.astype(bool)
###



def avg_too_close_peaks(spots, footprint):
    """ Takes a boolean array (ie. 2D black-and-white image) consisting of
    point-like seed points (ie. spots) and returns the average distance between them if
    the distance between them is smaller than the footprint. Footprint is a
    structuring element, I used a disk shaped one built from 
    skimage.morphology.disk(disk_radius) """
    # If 2 pixels have the exact same value then peak_local_max will return both
    # of them, even if they are closer than the footprint normally allows.
    # To fix this, return the peak locations:
    peak_locs = list(zip(*np.where(spots)))
    # And check if a window of the same shape as footprint contains 2 of these
    # peaks:
    # Start by defining windows around peaks so that you can multiply footprint
    # with peaks:
    window_start_rc = [x + (np.array(footprint.shape) / 2 * -1).astype(int) for x in peak_locs]
    window_finish_rc = [x + (np.array(footprint.shape) / 2 * 1 + 1).astype(int) for x in peak_locs]
    window_lims_strt_rc_end_rc = np.column_stack((window_start_rc, window_finish_rc))
    
    bad_peaks = list()
    mean_peaks = list()
    pad_width = int(disk_radius + 1) # Pad the array so that the window is always valid.
    for x in window_lims_strt_rc_end_rc:
        peaks_local = np.pad(spots, pad_width=pad_width, mode='constant', constant_values=0)[x[0]+pad_width:x[2]+pad_width, x[1]+pad_width:x[3]+pad_width] * footprint
        num_peaks = np.count_nonzero(peaks_local)
        if num_peaks > 1:
            # multi_peaks = np.argwhere(peaks_local) + np.array([np.mean([x[2]-1,x[0]], dtype=int), np.mean([x[3]-1,x[1]], dtype=int)]) - np.array([pad_width - 1, pad_width - 1])
            # mean_peak = multi_peaks.mean(axis=0, dtype=int) + np.array([(x[2] - x[0])/2, (x[3] - x[1])/2], dtype=int)
            # bad_peaks.append(tuple([np.mean([x[2]-1,x[0]], dtype=int), np.mean([x[3]-1,x[1]], dtype=int)]))
            multi_peaks = np.asarray([pk_rc + x[0:2] for pk_rc in np.argwhere(peaks_local)])
            mean_peak = np.mean(multi_peaks, axis=0, dtype=int)
            [bad_peaks.append(bad_pk) for bad_pk in multi_peaks]
            mean_peaks.append(mean_peak)
    if bad_peaks:
        # spots[tuple(np.asarray(a) for a in zip(*bad_peaks))] = False # Remove peaks that are too close.
        spots[tuple(np.asarray(a) for a in zip(*bad_peaks))] = False # Remove peaks that are too close.
    if mean_peaks:
        # spots[tuple(np.asarray(a) for a in zip(*mean_peaks))] = True # And replace those bad ones with the mean peaks.
        spots[tuple(np.asarray(a) for a in zip(*mean_peaks))] = True # And replace those bad ones with the mean peaks.
    # And so finally double peaks that are too close because they have equivalent
    # values have been fixed.
    return spots.astype(bool)
###
    


def iter_seg(image, markers_to_grow, markers_not_to_grow, iterations=100, mask=None, compactness=None):
    """ A function for iterative watershed segmentation of membrane labeled 
    cells where there is some signal within the cell as well (eg. a 
    non-negligible amount of background signal or both weak cytoplasmic and 
    strong membrane signal). This function requires 'numpy', 
    'skimage.segmentation.watershed', and 'mahotas'.
    
    Arguments
    ---------
    image               # A 2D image for the watershed to work on. (I used 
                            membrane labeled cells.)
    markers_to_grow     # A 2D image of the labeled markers for the watershed 
                            segmentation to use. These markers with get larger 
                            and larger with successive iterations and should be
                            the markers of your object of interest.
    markers_not_to_grow # A 2D image of the labeled markers for the watershed 
                            segmentation to use. These markers will not change 
                            in size.
    iterations          # The number of iterations you want to use (default=100).
    mask                # A binary 2D image to pass on to the 'mask' argument
                            of the 'skimage.segmentation.watershed' function.
    compactness         # Compactness parameter to pass on to argument of the
                            same name from the 'skimage.segmentation.watershed' 
                            function.
    
    Returns
    -------
    final_seg
        The last segmentation in the iterations.
    
    growing_markers_list
        A list of all iterations of the segmentation.
        
    growing_markers_area_list
        A list of the areas of the markers_to_grow for all of the segmentation 
        iterations.
    
    growing_markers_label_list
        The list corresponding to growing_markers_area_list telling you which
        area was for which marker.
        
    first_constant_area_indices
        A list of the the indices where the area no longer changes with 
        subsequent iterations of watershed segmentation. If a number in this 
        list is equal to the number of iterations specified (default=100) then
        beware that further growth in marker areas may be observed with 
        increased iterations. Increasing the number of iterations so that no
        marker is still changing in area at the final segmentation is advised.
    
    Notes
    -----
    Will segment image using markers_to_grow and markers_not_to_grow as seed 
    points and return the segmentation with the segmented regions from 
    markers_not_to_grow removed. None of the markers can overlap. If markers DO
    overlap then this function won't work as intended. markers_to_grow and 
    markers_not_to_grow must be 2D arrays of the same shape.
    """
    
    growing_markers_list = list()
    seeds = np.add.reduce((markers_to_grow, markers_not_to_grow))
    growing_markers = segmentation.watershed(image, seeds, mask=mask)
    growing_markers = mh.labeled.remove_regions(growing_markers, np.unique(markers_not_to_grow))
    growing_markers_list.append(growing_markers)
    
    count = 0
    while count < iterations:        
        seeds = np.add.reduce((growing_markers, markers_not_to_grow))
        growing_markers = segmentation.watershed(image, seeds, mask=mask)
        growing_markers = mh.labeled.remove_regions(growing_markers, np.unique(markers_not_to_grow))
        growing_markers_list.append(growing_markers)
        count += 1
    
    growing_markers_area_list = list()
    growing_markers_label_list = list()
    for i in np.unique(markers_to_grow)[np.nonzero(np.unique(markers_to_grow))]:
        for growing_markers in growing_markers_list:
            growing_markers_area_list.append(np.count_nonzero(growing_markers == i))
            growing_markers_label_list.append(i)
    
    arr = np.column_stack((growing_markers_area_list, growing_markers_label_list))
    growing_markers_diff_list = list()
    for i in np.unique(markers_to_grow)[np.nonzero(np.unique(markers_to_grow))]:
        growing_markers_diff_list.append(np.diff(arr[arr[:,1] == i, 0]))
    
    # find where the last example of the area changing for each labeled region is:
    first_constant_area_indices = list()
    for growing_markers in growing_markers_diff_list:
        #print(growing_markers == 0)
        if not((growing_markers == 0).all()):
            first_constant_area_indices.append(len(growing_markers) - np.argmin(growing_markers[::-1]==0))
        else:
            first_constant_area_indices.append(0)
        
    first_constant_area_labels = np.unique(markers_to_grow)[np.nonzero(np.unique(markers_to_grow))].tolist()
    
    
    final_seg_list = list()
    for i,label in enumerate(first_constant_area_labels):
        index = first_constant_area_indices[i]
        final_seg_list.append((growing_markers_list[index] == label)*label)
        #print(i,label,index)
    final_seg = np.add.reduce(final_seg_list)

    # If there are no segmentations then final_seg_list will end up being an
    # empty list, and as a result final_seg will == 0.0. If that happens then
    # final_seg will be the wrong type and shape, so we will redefine it to be
    # an empty 2D array of the same shape as image:
    if type(final_seg) == np.float64:
        final_seg = np.zeros(image.shape, dtype=int)
    elif type(final_seg) == np.ndarray:
        pass
    else:
        print(type(final_seg))
        raise TypeError(type(final_seg))

    return final_seg, growing_markers_list, growing_markers_area_list, growing_markers_label_list, first_constant_area_indices
###
    


def find_ridge_peaks(image, axis0_peak_to_peak_dist=15, axis1_peak_to_peak_dist=15, return_component_peaks=False, mask=None):
    """ Returns the boolean combination of the relative maxima along axis 0 
    and axis 1 of an image. The end result approximates ridges."""
    grad = mh.stretch(image)
    if np.any(mask):
        grad[np.where(mask)] = 0
        
    ridges_ax0 = np.zeros(grad.shape)
    ridges_ax1 = np.zeros(grad.shape)
    ridges_ax0[argrelextrema(grad, np.greater, axis=0, order=axis0_peak_to_peak_dist)] = 1
    ridges_ax1[argrelextrema(grad, np.greater, axis=1, order=axis1_peak_to_peak_dist)] = 2
    ridges = np.logical_or(ridges_ax0, ridges_ax1)
    
    if return_component_peaks:
        return ridges, ridges_ax0, ridges_ax1
    else:
        return ridges
###
        
    
    
def fluor_assisted_tl_segmentation(transmitted_light_image2D, redchan, bluchan, cell_thresh_as_bool, red_areas_labeled, blu_areas_labeled, seedpoints, footprint):

    red_dist = mh.distance(red_areas_labeled.astype(bool))
    blu_dist = mh.distance(blu_areas_labeled.astype(bool))
    dist_inv = np.min( np.dstack((255 - mh.stretch(red_dist), 255 - mh.stretch(blu_dist))), axis=2 )
    
    seedpoints = np.logical_xor(red_areas_labeled.astype(bool), blu_areas_labeled.astype(bool))
    seedpoints = seedpoints * (red_areas_labeled + blu_areas_labeled)
    
    seg = segmentation.watershed(mh.stretch(redchan)/2 + dist_inv/2, seedpoints, mask=cell_thresh_as_bool)
    seg = segmentation.watershed(dist_inv, seedpoints, mask=cell_thresh_as_bool)

    seg_closed_list = [morphology.closing(seg == x, selem=morphology.disk(3)) * x for x in np.unique(seg)[np.nonzero(np.unique(seg) > 0)]]
    
    if seg_closed_list:
        seg_closed = np.add.reduce(seg_closed_list)#.astype(bool)
        seg_closed = scp_morph.binary_fill_holes(seg_closed)
        seg_refined = segmentation.watershed(dist_inv, seg, mask=seg_closed)
    else:
        seg_refined = np.zeros(red_areas_labeled.shape)

    return seg_refined
###



def iter_erode(image, iterations, selem=None, out=None, shift_x=False, shift_y=False):
    """ This function performs iterative erosions. """
    image = image.astype(bool)
    i = 0
    while i < iterations:
        image = morphology.erosion(image, selem=selem, out=out, shift_x=shift_x, shift_y=shift_y)
        i += 1
    return image
###



def round_down_to_even(integer):
    """ This function will round a number down to the nearest even value. """
    if type(integer) == int:# or type(integer) == long #this part was for Python2.7:
        if integer%2 == 0:
            rounded_value = integer
        elif integer%2 == 1:
            rounded_value = integer-1
    else:
        print("The number to be rounded needs to be and integer!")
        rounded_value = float('nan')
    return rounded_value
###



def calc_R(x, y, xc, yc):
    """ calculate the distance of each data point from the center (xc, yc) """
    return np.sqrt((x-xc)**2 + (y-yc)**2)
###



def curvature_estimate_sign(x_fit_arc_start_stop, y_fit_arc_start_stop, x_fit_arc, y_fit_arc, curvature_border_array):
    """Returns the sign of the curvature based on whether or not the rasterized
       version of a straight line draw from one end of the curve to the other is
       mostly (ie. more than 50%) contained in a filled version of set of 
       curvature points."""
        
    if np.isnan(x_fit_arc_start_stop).any() == False and np.isnan(y_fit_arc_start_stop).any() == False:
        curve_chord_len = int(round(np.sqrt((x_fit_arc_start_stop[1] - x_fit_arc_start_stop[0])**2 + (y_fit_arc_start_stop[1] - y_fit_arc_start_stop[0])**2))+1)
        x_chord = np.linspace(x_fit_arc_start_stop[0], x_fit_arc_start_stop[1], num=curve_chord_len, endpoint=True)
        y_chord = np.linspace(y_fit_arc_start_stop[0], y_fit_arc_start_stop[1], num=curve_chord_len, endpoint=True)
        x_chord, y_chord = np.round(x_chord), np.round(y_chord)
        x_chord, y_chord = x_chord.astype(np.int), y_chord.astype(np.int)
    #        y_x = zip(y,x)
        ### y corresponds to the column position of the pixel, x corresponds to the row
        ### Was working on the below block (Feb 22 2018 as of 10:40 PM)
        curvature_definitions = np.asarray(curvature_border_array)
        clean_curve_area = scp_morph.binary_fill_holes(curvature_definitions)
        clean_curve_dist = mh.distance(clean_curve_area)
        
        ### The following bit may need to be removed, it's only a test to try
        # and account for when the fitted arc goes beyond the image limits:
        clean_curve_dist = np.pad(clean_curve_dist, pad_width=100, mode='constant', constant_values=0)
#        mean_chord_intens = np.mean(clean_curve_dist[y_chord,x_chord])
        mean_chord_intens = np.mean(clean_curve_dist[x_chord,y_chord])

#        mean_curve_intens = np.mean(clean_curve_dist[ np.round(y_fit_arc[np.ma.notmasked_contiguous(y_fit_arc)]).astype(int), np.round(x_fit_arc[np.ma.notmasked_contiguous(x_fit_arc)]).astype(int) ])
        mean_curve_intens = np.mean(clean_curve_dist[ np.round(x_fit_arc[np.ma.notmasked_contiguous(x_fit_arc)]).astype(int), np.round(y_fit_arc[np.ma.notmasked_contiguous(y_fit_arc)]).astype(int) ])
        
    #        clean_curve_area = clean_curve_area.astype(np.uint8)
    #       clean_curve_area_locs = zip(*np.where(clean_curve_area))
    #        curve_chord_in_areas = [val in clean_curve_area_locs for val in y_x]
    
        ### Is the curvature positive? (True/False)
        positive_curvature = mean_chord_intens >= mean_curve_intens
        if positive_curvature == True:
            return_val = 1.0
        elif positive_curvature == False:
            return_val = -1.0
        else:
            print("Apparently the curvature is neither positive nor negative?")
    else:
        return_val = float('nan')
        
    return return_val
###



def f_3(beta, x):
    """ implicit definition of the circle """
    return (x[0]-beta[0])**2 + (x[1]-beta[1])**2 -beta[2]**2
###



def check_for_signal(image_of_cellpair_as_array):
    
    # If image_of_cellpair_as_array is RGB:
    if image_of_cellpair_as_array.ndim == 3:
        redchan = image_of_cellpair_as_array[:,:,0]
        grnchan = image_of_cellpair_as_array[:,:,1]
        bluchan = image_of_cellpair_as_array[:,:,2]
    
        if np.std(redchan) < 1:
            redchan = redchan*0
            print ("There is no signal in the red channel or the signal-to-noise"
                   " ratio in the red channel is too low."
                   )
            messages.append("There is no signal in the red channel or the signal-to-noise ratio in the red channel is too low.")
        if np.std(grnchan) < 1:
            grnchan = grnchan*0
            print ("There is no signal in the green channel or the signal-to-noise"
                   " ratio in the green channel is too low."
                   )
            messages.append("There is no signal in the green channel or the signal-to-noise ratio in the green channel is too low.")
        if np.std(bluchan) < 1:
            bluchan = bluchan*0
            print ("There is no signal in the blue channel or the signal-to-noise"
                   " ratio in the blue channel is too low."
                   )
            messages.append("There is no signal in the blue channel or the signal-to-noise ratio in the blue channel is too low.")
        image_of_cellpair_as_array_clean = np.array(np.dstack((redchan, grnchan, bluchan)))    
    # If image_of_cellpair_as_array is grayscale:
    elif image_of_cellpair_as_array.ndim == 2:
        if np.std(image_of_cellpair_as_array) < 0.5:
            image_of_cellpair_as_array = image_of_cellpair_as_array*0
            print ("There is no signal in this channel or the signal-to-noise"
                   " ratio in this channel is too low."
                   )
            messages.append("There is no signal in this channel or the signal-to-noise ratio in this channel is too low.")
        image_of_cellpair_as_array_clean = np.copy(image_of_cellpair_as_array)
    # If the image is neither grayscale nor RGB then just skip it.
    else:
        print("I don't understand this image structure, please use RBG or grayscale.")
        image_of_cellpair_as_array = image_of_cellpair_as_array*0
        image_of_cellpair_as_array_clean = np.copy(image_of_cellpair_as_array)

    return image_of_cellpair_as_array_clean
###
      


def nth_max(n, iterable):
    return heapq.nlargest(n, iterable)[-1]
###



def remove_bordering_and_small_regions(segmented_image_of_cellareas):
    segmented_image_of_cellareas = segmented_image_of_cellareas.astype(int)
    ### To remove cells touching the edges of the image:                
    no_border_cells_red = segmentation.clear_border(segmented_image_of_cellareas[:,:,0])
    # no_border_cells_red = mh.labeled.remove_bordering(segmented_image_of_cellareas[:,:,0])
    # no_border_cells_red = mh.label(no_border_cells_red)[0]
    no_border_cells_grn = segmentation.clear_border(segmented_image_of_cellareas[:,:,1])
    # no_border_cells_grn = mh.labeled.remove_bordering(segmented_image_of_cellareas[:,:,1])
    # no_border_cells_grn = mh.label(no_border_cells_grn)[0]
    no_border_cells_blu = segmentation.clear_border(segmented_image_of_cellareas[:,:,2])
    # no_border_cells_blu = mh.labeled.remove_bordering(segmented_image_of_cellareas[:,:,2])
    # no_border_cells_blu = mh.label(no_border_cells_blu)[0]
    no_border_cells = np.dstack((no_border_cells_red, no_border_cells_grn, no_border_cells_blu))
    
    ### Remove any areas that are too small
    no_border_cells = morphology.remove_small_objects(no_border_cells, min_size=size_filter)
    
    
    # sizes = mh.labeled.labeled_size(no_border_cells)
    # iterate = sizes
    # n = 3
    # thirdlargestregion = nth_max(n, iterate)
    # too_small = np.where(sizes < thirdlargestregion)
    # no_border_cells = mh.labeled.remove_regions(no_border_cells, too_small)

    # sizes = mh.labeled.labeled_size(no_border_cells[:,:,0])
    # iterate = sizes
    # n = 2
    # secondlargestregion = nth_max(n, iterate)
    # too_small = np.where(sizes < secondlargestregion)
    # no_border_cells[:,:,0] = mh.labeled.remove_regions(no_border_cells[:,:,0], too_small)


    # sizes = mh.labeled.labeled_size(no_border_cells[:,:,1])
    # iterate = sizes
    # n = 2
    # secondlargestregion = nth_max(n, iterate)
    # too_small = np.where(sizes < secondlargestregion)
    # no_border_cells[:,:,1] = mh.labeled.remove_regions(no_border_cells[:,:,1], too_small)


    # sizes = mh.labeled.labeled_size(no_border_cells[:,:,2])
    # iterate = sizes
    # n = 2
    # secondlargestregion = nth_max(n, iterate)
    # too_small = np.where(sizes < secondlargestregion)
    # no_border_cells[:,:,2] = mh.labeled.remove_regions(no_border_cells[:,:,2], too_small)
    
    return no_border_cells
###



def clean_up_borders(borders_of_interest):
    
    pad_width = 1
    pad_val = 0
    original_borders_of_interest = pad_img_edges(np.copy(borders_of_interest).astype(int), pad_val, pad_width).astype(int)
    
    # original_borders_of_interest = np.copy(borders_of_interest).astype(int)
    borders_of_interest = np.zeros(original_borders_of_interest.shape, dtype=int)
    
    
    cell_of_interest_red = scp_morph.binary_fill_holes(original_borders_of_interest[:,:,0])
    inner_area_red = np.logical_xor(cell_of_interest_red, original_borders_of_interest[:,:,0].astype(bool))
    inner_area_red = mh.label(inner_area_red)[0]
    sizes = mh.labeled.labeled_size(inner_area_red)
    iterate = sizes
    n = 2
    secondlargestregion = nth_max(n, iterate)
    too_small = np.where(sizes < secondlargestregion)
    inner_area_red = mh.labeled.remove_regions(inner_area_red, too_small)
### Create a structuring element with 4-connectivity to find the parts of
### original_borders_of_interest that touch the inner areas:
    temp_Bc = mh.get_structuring_elem(inner_area_red, 4)
#    temp_Bc = mh.get_structuring_elem(inner_area_red, 8)
    for val in zip(*np.where(inner_area_red)):
        if 1 in original_borders_of_interest[sub(val[0],1):val[0]+2, sub(val[1],1):val[1]+2, 0] * temp_Bc:
            borders_of_interest[sub(val[0],1):val[0]+2, sub(val[1],1):val[1]+2, 0] += original_borders_of_interest[sub(val[0],1):val[0]+2, sub(val[1],1):val[1]+2, 0] * temp_Bc
    borders_of_interest[:,:,0] = borders_of_interest[:,:,0].astype(bool) * 1
#    refined_borders_of_interest_red = border_of_interest_red * np.logical_not(inner_area_red)


    cell_of_interest_grn = scp_morph.binary_fill_holes(original_borders_of_interest[:,:,1])
    inner_area_grn = np.logical_xor(cell_of_interest_grn, original_borders_of_interest[:,:,1].astype(bool))
    inner_area_grn = mh.label(inner_area_grn)[0]
    sizes = mh.labeled.labeled_size(inner_area_grn)
    iterate = sizes
    n = 2
    secondlargestregion = nth_max(n, iterate)
    too_small = np.where(sizes < secondlargestregion)
    inner_area_grn = mh.labeled.remove_regions(inner_area_grn, too_small)
    temp_Bc = mh.get_structuring_elem(inner_area_grn, 4)
#    temp_Bc = mh.get_structuring_elem(inner_area_grn, 8)
    for val in zip(*np.where(inner_area_grn)):
        if 2 in original_borders_of_interest[sub(val[0],1):val[0]+2, sub(val[1],1):val[1]+2, 1] * temp_Bc:
            borders_of_interest[sub(val[0],1):val[0]+2, sub(val[1],1):val[1]+2, 1] += original_borders_of_interest[sub(val[0],1):val[0]+2, sub(val[1],1):val[1]+2, 1] * temp_Bc
    borders_of_interest[:,:,1] = borders_of_interest[:,:,1].astype(bool) * 2


    cell_of_interest_blu = scp_morph.binary_fill_holes(original_borders_of_interest[:,:,2])
    inner_area_blu = np.logical_xor(cell_of_interest_blu, original_borders_of_interest[:,:,2].astype(bool))
    inner_area_blu = mh.label(inner_area_blu)[0]
    sizes = mh.labeled.labeled_size(inner_area_blu)
    iterate = sizes
    n = 2
    secondlargestregion = nth_max(n, iterate)
    too_small = np.where(sizes < secondlargestregion)
    inner_area_blu = mh.labeled.remove_regions(inner_area_blu, too_small)
    temp_Bc = mh.get_structuring_elem(inner_area_blu, 4)
#    temp_Bc = mh.get_structuring_elem(inner_area_grn, 8)
    for val in zip(*np.where(inner_area_blu)):
        if 3 in original_borders_of_interest[sub(val[0],1):val[0]+2, sub(val[1],1):val[1]+2, 2] * temp_Bc:
            borders_of_interest[sub(val[0],1):val[0]+2, sub(val[1],1):val[1]+2, 2] += original_borders_of_interest[sub(val[0],1):val[0]+2, sub(val[1],1):val[1]+2, 2] * temp_Bc
    borders_of_interest[:,:,2] = borders_of_interest[:,:,2].astype(bool) * 3

           
### Start by trimming leftover branches (if they exist):
    trim_count = 0
    while trim_count < 400:
        trimming_done = True
        for (i,j,k), surfval in np.ndenumerate(borders_of_interest):
            if surfval != 0:
                if (len(np.where(borders_of_interest[sub(i,1):i+2, sub(j,1):j+2, k] == surfval)[0]) == 2):
                    borders_of_interest[i,j,k] = 0
                    trimming_done = False
                    print('Trimmed pixel @', i,j,k)
        trim_count += 1
        if trimming_done == True:
            trim_count = 401
            print('Finished trimming...')
    curve_excess_bord = np.zeros(borders_of_interest.shape, np.dtype(np.int32))
    
    
    num_cleanings = 0
    while num_cleanings < 40:
    ### Remove any excess corner pixels:
        extra_type1 = np.c_[[0,1,0], [1,1,0], [0,0,0]]
        extra_type2 = np.c_[[0,1,0], [0,1,1], [0,0,0]]
        extra_type3 = np.c_[[0,0,0], [1,1,0], [0,1,0]]
        extra_type4 = np.c_[[0,0,0], [0,1,1], [0,1,0]]
    
        curve_excess_bord = np.zeros(borders_of_interest.shape, np.dtype(np.int32))
        for (i,j,k), surfval in np.ndenumerate(borders_of_interest):
            if surfval != 0:
                if (len(np.where(extra_type1 * borders_of_interest[sub(i,1):i+2, sub(j,1):j+2, k] == surfval)[0]) > 2 or
                    len(np.where(extra_type2 * borders_of_interest[sub(i,1):i+2, sub(j,1):j+2, k] == surfval)[0]) > 2 or
                    len(np.where(extra_type3 * borders_of_interest[sub(i,1):i+2, sub(j,1):j+2, k] == surfval)[0]) > 2 or
                    len(np.where(extra_type4 * borders_of_interest[sub(i,1):i+2, sub(j,1):j+2, k] == surfval)[0]) > 2
                    ):
                        curve_excess_bord[i,j,k] += borders_of_interest[i,j,k]
                        borders_of_interest[i,j,k] = 0
                        # to speed things along we're going to reduce the
                        # amount of ndenumerates we need to do by making sure
                        # that only 1 pixel exists in the curve_excess_bord
                        # array with a break statement right after finding
                        # a curve_excess_bord pixel:
                        break
    
        ### Removing excess corners can make the cellborder discontinuous. To fix it:
        # pixels on the ends of the borders in an array:
        border_ends = np.zeros(borders_of_interest.shape)
        for (i,j,k), surfval in np.ndenumerate(borders_of_interest):
        # If you're looking at a cleanborder pixel and had excess corners - 
            if surfval != 0 and curve_excess_bord.any():
                if (
                    (len(np.where(borders_of_interest[sub(i,1):i+2, sub(j,1):j+2, k] == surfval)[0]) == 2)# and
                    ):
                    border_ends[i,j,k] = surfval


        # Check if any of the pixels in curve_excess_bord also contain a pixel 
        # in border_ends when combined with extra_type1, extra_type2, etc.
        # and if there are, remove the border ends pixel that corresponds to
        # the extra_type and replace the curve_excess_bord pixel:
        # for (i,j,k), surfval in np.ndenumerate(curve_excess_bord):
        # if curve_excess_bord.any():
        if border_ends.any():
            assert np.count_nonzero(curve_excess_bord)==1
            i,j,k = map(int, np.where(curve_excess_bord))
            surfval = curve_excess_bord[i,j,k]
            if np.count_nonzero(extra_type1 * borders_of_interest[sub(i,1):i+2, sub(j,1):j+2, k]) == 2:
                if (extra_type1 * border_ends[sub(i,1):i+2, sub(j,1):j+2, k]).any():
                    # print((i,j,k), surfval, 'extra1', np.expand_dims(np.asarray((i,j,k)), axis=1) + np.row_stack((np.asarray(np.where(extra_type1 * border_ends[sub(i,1):i+2, sub(j,1):j+2, k])) - 1, k)))
                    borders_of_interest[tuple(np.expand_dims(np.asarray((i,j,k)), axis=1) + np.row_stack((np.asarray(np.where(extra_type1 * border_ends[sub(i,1):i+2, sub(j,1):j+2, k])) - 1, k)))] = 0
                    borders_of_interest[i,j,k] = surfval
            if np.count_nonzero(extra_type2 * borders_of_interest[sub(i,1):i+2, sub(j,1):j+2, k]) == 2:
                if (extra_type2 * border_ends[sub(i,1):i+2, sub(j,1):j+2, k]).any():
                    # print((i,j,k), surfval, 'extra2', np.expand_dims(np.asarray((i,j,k)), axis=1) + np.row_stack((np.asarray(np.where(extra_type2 * border_ends[sub(i,1):i+2, sub(j,1):j+2, k])) - 1, k))) #This last bit after 'extra2' is the location of the culprit pixel to be removed, and the i,j,k pixel should be replaced
                    borders_of_interest[tuple(np.expand_dims(np.asarray((i,j,k)), axis=1) + np.row_stack((np.asarray(np.where(extra_type2 * border_ends[sub(i,1):i+2, sub(j,1):j+2, k])) - 1, k)))] = 0
                    borders_of_interest[i,j,k] = surfval
            if np.count_nonzero(extra_type3 * borders_of_interest[sub(i,1):i+2, sub(j,1):j+2, k]) == 2:
                if (extra_type3 * border_ends[sub(i,1):i+2, sub(j,1):j+2, k]).any():
                    # print((i,j,k), surfval, 'extra3', np.expand_dims(np.asarray((i,j,k)), axis=1) + np.row_stack((np.asarray(np.where(extra_type3 * border_ends[sub(i,1):i+2, sub(j,1):j+2, k])) - 1, k)))
                    borders_of_interest[tuple(np.expand_dims(np.asarray((i,j,k)), axis=1) + np.row_stack((np.asarray(np.where(extra_type3 * border_ends[sub(i,1):i+2, sub(j,1):j+2, k])) - 1, k)))] = 0
                    borders_of_interest[i,j,k] = surfval
            if np.count_nonzero(extra_type4 * borders_of_interest[sub(i,1):i+2, sub(j,1):j+2, k]) == 2:
                if (extra_type4 * border_ends[sub(i,1):i+2, sub(j,1):j+2, k]).any():
                    # print((i,j,k), surfval, 'extra4', np.expand_dims(np.asarray((i,j,k)), axis=1) + np.row_stack((np.asarray(np.where(extra_type4 * border_ends[sub(i,1):i+2, sub(j,1):j+2, k])) - 1, k)))
                    borders_of_interest[tuple(np.expand_dims(np.asarray((i,j,k)), axis=1) + np.row_stack((np.asarray(np.where(extra_type4 * border_ends[sub(i,1):i+2, sub(j,1):j+2, k])) - 1, k)))] = 0
                    borders_of_interest[i,j,k] = surfval

        if curve_excess_bord.any() == False and border_ends.any() == False:
            break
        else:
            num_cleanings += 1

    
    
    
    
    
    
        # Check if any of the pixels in curve_excess_bord also contain a pixel 
        # in border_ends when combined with extra_type1, extra_type2, etc.:
        #while count > 0:        
        #pixel_to_replace = np.zeros(borders_of_interest.shape)
    #     for (i,j,k), surfval in np.ndenumerate(curve_excess_bord):
    #         if surfval != 0:
    #             if np.count_nonzero(extra_type1 * borders_of_interest[sub(i,1):i+2, sub(j,1):j+2, k]) == 2:
    #                 if (extra_type1 * border_ends[sub(i,1):i+2, sub(j,1):j+2, k]).any():
    #                     print((i,j,k), surfval, 'extra1', np.expand_dims(np.asarray((i,j,k)), axis=1) + np.row_stack((np.asarray(np.where(extra_type1 * border_ends[sub(i,1):i+2, sub(j,1):j+2, k])) - 1, k)))
    #                     borders_of_interest[tuple(np.expand_dims(np.asarray((i,j,k)), axis=1) + np.row_stack((np.asarray(np.where(extra_type1 * border_ends[sub(i,1):i+2, sub(j,1):j+2, k])) - 1, k)))] = 0
    #                     borders_of_interest[i,j,k] = surfval
    #                     curve_excess_bord[tuple(np.expand_dims(np.asarray((i,j,k)), axis=1) + np.row_stack((np.asarray(np.where(extra_type1 * border_ends[sub(i,1):i+2, sub(j,1):j+2, k])) - 1, k)))] = surfval
    #                     curve_excess_bord[i,j,k] = 0
                        
    #             if np.count_nonzero(extra_type2 * borders_of_interest[sub(i,1):i+2, sub(j,1):j+2, k]) == 2:
    #                 if (extra_type2 * border_ends[sub(i,1):i+2, sub(j,1):j+2, k]).any():
    #                     print((i,j,k), surfval, 'extra2', np.expand_dims(np.asarray((i,j,k)), axis=1) + np.row_stack((np.asarray(np.where(extra_type2 * border_ends[sub(i,1):i+2, sub(j,1):j+2, k])) - 1, k))) #This last bit after 'extra2' is the location of the culprit pixel to be removed, and the i,j,k pixel should be replaced
    #                     borders_of_interest[tuple(np.expand_dims(np.asarray((i,j,k)), axis=1) + np.row_stack((np.asarray(np.where(extra_type2 * border_ends[sub(i,1):i+2, sub(j,1):j+2, k])) - 1, k)))] = 0
    #                     borders_of_interest[i,j,k] = surfval
    #                     curve_excess_bord[tuple(np.expand_dims(np.asarray((i,j,k)), axis=1) + np.row_stack((np.asarray(np.where(extra_type2 * border_ends[sub(i,1):i+2, sub(j,1):j+2, k])) - 1, k)))]
    #                     curve_excess_bord[i,j,k] = 0
                    
    #             if np.count_nonzero(extra_type3 * borders_of_interest[sub(i,1):i+2, sub(j,1):j+2, k]) == 2:
    #                 if (extra_type3 * border_ends[sub(i,1):i+2, sub(j,1):j+2, k]).any():
    #                     print((i,j,k), surfval, 'extra3', np.expand_dims(np.asarray((i,j,k)), axis=1) + np.row_stack((np.asarray(np.where(extra_type3 * border_ends[sub(i,1):i+2, sub(j,1):j+2, k])) - 1, k)))
    #                     borders_of_interest[tuple(np.expand_dims(np.asarray((i,j,k)), axis=1) + np.row_stack((np.asarray(np.where(extra_type3 * border_ends[sub(i,1):i+2, sub(j,1):j+2, k])) - 1, k)))] = 0
    #                     borders_of_interest[i,j,k] = surfval
    #                     curve_excess_bord[tuple(np.expand_dims(np.asarray((i,j,k)), axis=1) + np.row_stack((np.asarray(np.where(extra_type3 * border_ends[sub(i,1):i+2, sub(j,1):j+2, k])) - 1, k)))]
    #                     curve_excess_bord[i,j,k] = 0
                
    #             if np.count_nonzero(extra_type4 * borders_of_interest[sub(i,1):i+2, sub(j,1):j+2, k]) == 2:
    #                 if (extra_type4 * border_ends[sub(i,1):i+2, sub(j,1):j+2, k]).any():
    #                     print((i,j,k), surfval, 'extra4', np.expand_dims(np.asarray((i,j,k)), axis=1) + np.row_stack((np.asarray(np.where(extra_type4 * border_ends[sub(i,1):i+2, sub(j,1):j+2, k])) - 1, k)))
    #                     borders_of_interest[tuple(np.expand_dims(np.asarray((i,j,k)), axis=1) + np.row_stack((np.asarray(np.where(extra_type4 * border_ends[sub(i,1):i+2, sub(j,1):j+2, k])) - 1, k)))] = 0
    #                     borders_of_interest[i,j,k] = surfval
    #                     curve_excess_bord[tuple(np.expand_dims(np.asarray((i,j,k)), axis=1) + np.row_stack((np.asarray(np.where(extra_type4 * border_ends[sub(i,1):i+2, sub(j,1):j+2, k])) - 1, k)))]
    #                     curve_excess_bord[i,j,k] = 0
    #             #count -= 1
    #             #print(i,j,k)
                
    # # num_cleanings = 0
    # # while num_cleanings < 4:
    # # ### Remove any excess corner pixels:
    # #     extra_type1 = np.c_[[0,1,0], [1,1,0], [0,0,0]]
    # #     extra_type2 = np.c_[[0,1,0], [0,1,1], [0,0,0]]
    # #     extra_type3 = np.c_[[0,0,0], [1,1,0], [0,1,0]]
    # #     extra_type4 = np.c_[[0,0,0], [0,1,1], [0,1,0]]
    
    # #     for (i,j,k), surfval in np.ndenumerate(borders_of_interest):
    # #         if surfval != 0:
    # #             if (len(np.where(extra_type1 * borders_of_interest[sub(i,1):i+2, sub(j,1):j+2, k] == surfval)[0]) > 2 or
    # #                 len(np.where(extra_type2 * borders_of_interest[sub(i,1):i+2, sub(j,1):j+2, k] == surfval)[0]) > 2 or
    # #                 len(np.where(extra_type3 * borders_of_interest[sub(i,1):i+2, sub(j,1):j+2, k] == surfval)[0]) > 2 or
    # #                 len(np.where(extra_type4 * borders_of_interest[sub(i,1):i+2, sub(j,1):j+2, k] == surfval)[0]) > 2
    # #                 ):
    # #                     curve_excess_bord[i,j,k] += borders_of_interest[i,j,k]
    # #                     borders_of_interest[i,j,k] = 0
    
    # # ### Removing excess corners can make the cellborder discontinuous. To fix it:
    # #     # pixels on the ends of the borders in an array:
    # #     border_ends = np.zeros(borders_of_interest.shape)
    # #     for (i,j,k), surfval in np.ndenumerate(borders_of_interest):
    # #     # If you're looking at a cleanborder pixel and had excess corners - 
    # #         if surfval != 0 and curve_excess_bord.any():
    # #             # then if it's right next to the site of discontinuity:
    # #             if (
    # #                 (len(np.where(borders_of_interest[sub(i,1):i+2, sub(j,1):j+2, k] == surfval)[0]) == 2)# and
    # #                 ):
    # #                 border_ends[i,j,k] = surfval
    
    #     # Check if any of the pixels in curve_excess_bord also contain a pixel 
    #     # in border_ends when combined with extra_type1, extra_type2, etc.:
    #     #while count > 0:        
    #     #pixel_to_replace = np.zeros(borders_of_interest.shape)
    #     for (i,j,k), surfval in np.ndenumerate(curve_excess_bord):
    #         if surfval != 0:
    #             if np.count_nonzero(extra_type1 * borders_of_interest[sub(i,1):i+2, sub(j,1):j+2, k]) == 2:
    #                 if (extra_type1 * border_ends[sub(i,1):i+2, sub(j,1):j+2, k]).any():
    #                     print((i,j,k), surfval, 'extra1', np.expand_dims(np.asarray((i,j,k)), axis=1) + np.row_stack((np.asarray(np.where(extra_type1 * border_ends[sub(i,1):i+2, sub(j,1):j+2, k])) - 1, k)))
    #                     borders_of_interest[tuple(np.expand_dims(np.asarray((i,j,k)), axis=1) + np.row_stack((np.asarray(np.where(extra_type1 * border_ends[sub(i,1):i+2, sub(j,1):j+2, k])) - 1, k)))] = 0
    #                     borders_of_interest[i,j,k] = surfval
    #                     curve_excess_bord[tuple(np.expand_dims(np.asarray((i,j,k)), axis=1) + np.row_stack((np.asarray(np.where(extra_type1 * border_ends[sub(i,1):i+2, sub(j,1):j+2, k])) - 1, k)))] = surfval
    #                     curve_excess_bord[i,j,k] = 0
                        
    #             if np.count_nonzero(extra_type2 * borders_of_interest[sub(i,1):i+2, sub(j,1):j+2, k]) == 2:
    #                 if (extra_type2 * border_ends[sub(i,1):i+2, sub(j,1):j+2, k]).any():
    #                     print((i,j,k), surfval, 'extra2', np.expand_dims(np.asarray((i,j,k)), axis=1) + np.row_stack((np.asarray(np.where(extra_type2 * border_ends[sub(i,1):i+2, sub(j,1):j+2, k])) - 1, k))) #This last bit after 'extra2' is the location of the culprit pixel to be removed, and the i,j,k pixel should be replaced
    #                     borders_of_interest[tuple(np.expand_dims(np.asarray((i,j,k)), axis=1) + np.row_stack((np.asarray(np.where(extra_type2 * border_ends[sub(i,1):i+2, sub(j,1):j+2, k])) - 1, k)))] = 0
    #                     borders_of_interest[i,j,k] = surfval
    #                     curve_excess_bord[tuple(np.expand_dims(np.asarray((i,j,k)), axis=1) + np.row_stack((np.asarray(np.where(extra_type2 * border_ends[sub(i,1):i+2, sub(j,1):j+2, k])) - 1, k)))]
    #                     curve_excess_bord[i,j,k] = 0
                    
    #             if np.count_nonzero(extra_type3 * borders_of_interest[sub(i,1):i+2, sub(j,1):j+2, k]) == 2:
    #                 if (extra_type3 * border_ends[sub(i,1):i+2, sub(j,1):j+2, k]).any():
    #                     print((i,j,k), surfval, 'extra3', np.expand_dims(np.asarray((i,j,k)), axis=1) + np.row_stack((np.asarray(np.where(extra_type3 * border_ends[sub(i,1):i+2, sub(j,1):j+2, k])) - 1, k)))
    #                     borders_of_interest[tuple(np.expand_dims(np.asarray((i,j,k)), axis=1) + np.row_stack((np.asarray(np.where(extra_type3 * border_ends[sub(i,1):i+2, sub(j,1):j+2, k])) - 1, k)))] = 0
    #                     borders_of_interest[i,j,k] = surfval
    #                     curve_excess_bord[tuple(np.expand_dims(np.asarray((i,j,k)), axis=1) + np.row_stack((np.asarray(np.where(extra_type3 * border_ends[sub(i,1):i+2, sub(j,1):j+2, k])) - 1, k)))]
    #                     curve_excess_bord[i,j,k] = 0
                
    #             if np.count_nonzero(extra_type4 * borders_of_interest[sub(i,1):i+2, sub(j,1):j+2, k]) == 2:
    #                 if (extra_type4 * border_ends[sub(i,1):i+2, sub(j,1):j+2, k]).any():
    #                     print((i,j,k), surfval, 'extra4', np.expand_dims(np.asarray((i,j,k)), axis=1) + np.row_stack((np.asarray(np.where(extra_type4 * border_ends[sub(i,1):i+2, sub(j,1):j+2, k])) - 1, k)))
    #                     borders_of_interest[tuple(np.expand_dims(np.asarray((i,j,k)), axis=1) + np.row_stack((np.asarray(np.where(extra_type4 * border_ends[sub(i,1):i+2, sub(j,1):j+2, k])) - 1, k)))] = 0
    #                     borders_of_interest[i,j,k] = surfval
    #                     curve_excess_bord[tuple(np.expand_dims(np.asarray((i,j,k)), axis=1) + np.row_stack((np.asarray(np.where(extra_type4 * border_ends[sub(i,1):i+2, sub(j,1):j+2, k])) - 1, k)))]
    #                     curve_excess_bord[i,j,k] = 0
    #             #count -= 1
    #             #print(i,j,k)
                
    ### Removing excess corners can make the cellborder discontinuous. To fix it:
    #     count = len(list(zip(*np.where(curve_excess_bord))))
    # #    count = 0
    # #    while count < 8:
    #     while count > 0:        
    #         for (i,j,k), surfval in np.ndenumerate(borders_of_interest):
    #         # If you're looking at a cleanborder pixel and had excess corners - 
    #             if surfval != 0 and curve_excess_bord.any():
    #                 # then if it's right next to the site of discontinuity:
    #                 if (
    #                     (len(np.where(borders_of_interest[sub(i,1):i+2, sub(j,1):j+2, k] == surfval)[0]) == 2)# and
    #                     ):
    #                     #print(i,j,k)
    #                     try:
    #                         if ( ((i,j) in np.transpose(np.where(extra_type1 * borders_of_interest[sub(np.where(curve_excess_bord)[0][0],1):np.where(curve_excess_bord)[0][0]+2, sub(np.where(curve_excess_bord)[1][0],1):np.where(curve_excess_bord)[1][0]+2, k]))+np.transpose(np.where(curve_excess_bord))[:,:2][0]-1) and
    #                             (np.count_nonzero(extra_type1 * borders_of_interest[sub(np.where(curve_excess_bord)[0][0],1):np.where(curve_excess_bord)[0][0]+2, sub(np.where(curve_excess_bord)[1][0],1):np.where(curve_excess_bord)[1][0]+2, k]) == 2 )
    #                             ):
    #                             borders_of_interest[sub(i,1):i+2, sub(j,1):j+2, k] += curve_excess_bord[sub(i,1):i+2, sub(j,1):j+2, k]
    #                             borders_of_interest[i,j,k] = 0
    # #                                    excess_bord_arr[np.transpose(np.where(excess_bord_arr))[:,:2][0][0], np.transpose(np.where(excess_bord_arr))[:,:2][0][1], k] = 0
    #                             print(count, "curve border extra_type1", (i,j,k))
    #                             break
    #                         else:
    # #                            actual_excess += 1
    #                             print(count, (i,j,k))
    #                             pass
    #                     except ValueError:
    #                         print('fail')
    #                         pass
    #                     try:
    #                         if ( ((i,j) in np.transpose(np.where(extra_type2 * borders_of_interest[sub(np.where(curve_excess_bord)[0][0],1):np.where(curve_excess_bord)[0][0]+2, sub(np.where(curve_excess_bord)[1][0],1):np.where(curve_excess_bord)[1][0]+2, k]))+np.transpose(np.where(curve_excess_bord))[:,:2][0]-1) and
    #                             (np.count_nonzero(extra_type2 * borders_of_interest[sub(np.where(curve_excess_bord)[0][0],1):np.where(curve_excess_bord)[0][0]+2, sub(np.where(curve_excess_bord)[1][0],1):np.where(curve_excess_bord)[1][0]+2, k]) == 2 )
    #                             ):
    #                             borders_of_interest[sub(i,1):i+2, sub(j,1):j+2, k] += curve_excess_bord[sub(i,1):i+2, sub(j,1):j+2, k]
    #                             borders_of_interest[i,j,k] = 0
    # #                                    excess_bord_arr[np.transpose(np.where(excess_bord_arr))[:,:2][0][0], np.transpose(np.where(excess_bord_arr))[:,:2][0][1], k] = 0
    #                             print(count, "curve border extra_type2", (i,j,k))
    #                             break
    #                         else:
    # #                            actual_excess += 1
    #                             print(count, (i,j,k))
    #                             pass
    #                     except ValueError:
    #                         print('fail')
    #                         pass
    #                     try:
    #                         if ( ((i,j) in np.transpose(np.where(extra_type3 * borders_of_interest[sub(np.where(curve_excess_bord)[0][0],1):np.where(curve_excess_bord)[0][0]+2, sub(np.where(curve_excess_bord)[1][0],1):np.where(curve_excess_bord)[1][0]+2, k]))+np.transpose(np.where(curve_excess_bord))[:,:2][0]-1) and
    #                             (np.count_nonzero(extra_type3 * borders_of_interest[sub(np.where(curve_excess_bord)[0][0],1):np.where(curve_excess_bord)[0][0]+2, sub(np.where(curve_excess_bord)[1][0],1):np.where(curve_excess_bord)[1][0]+2, k]) == 2 )
    #                             ):
    #                             borders_of_interest[sub(i,1):i+2, sub(j,1):j+2, k] += curve_excess_bord[sub(i,1):i+2, sub(j,1):j+2, k]
    #                             borders_of_interest[i,j,k] = 0
    # #                                    excess_bord_arr[np.transpose(np.where(excess_bord_arr))[:,:2][0][0], np.transpose(np.where(excess_bord_arr))[:,:2][0][1], k] = 0
    #                             print(count, "curve border extra_type3", (i,j,k))
    #                             break
    #                         else:
    # #                            actual_excess += 1
    #                             print(count, (i,j,k))
    #                             pass
    #                     except ValueError:
    #                         print('fail')
    #                         pass
    #                     try:
    #                         if ( ((i,j) in np.transpose(np.where(extra_type4 * borders_of_interest[sub(np.where(curve_excess_bord)[0][0],1):np.where(curve_excess_bord)[0][0]+2, sub(np.where(curve_excess_bord)[1][0],1):np.where(curve_excess_bord)[1][0]+2, k]))+np.transpose(np.where(curve_excess_bord))[:,:2][0]-1) and
    #                             (np.count_nonzero(extra_type4 * borders_of_interest[sub(np.where(curve_excess_bord)[0][0],1):np.where(curve_excess_bord)[0][0]+2, sub(np.where(curve_excess_bord)[1][0],1):np.where(curve_excess_bord)[1][0]+2, k]) == 2 )
    #                             ):
    #                             borders_of_interest[sub(i,1):i+2, sub(j,1):j+2, k] += curve_excess_bord[sub(i,1):i+2, sub(j,1):j+2, k]
    #                             borders_of_interest[i,j,k] = 0
    # #                                    excess_bord_arr[np.transpose(np.where(excess_bord_arr))[:,:2][0][0], np.transpose(np.where(excess_bord_arr))[:,:2][0][1], k] = 0
    #                             print(count, "curve border extra_type4", (i,j,k))
    #                             break
    #                         else:
    # #                            actual_excess += 1
    #                             print(count, (i,j,k))
    #                             pass
    #                     except ValueError:
    #                         print('fail')
    #                         pass
    # #                                if (((i,j) in np.transpose(np.where(extra_type1 * cleanborders[sub(np.where(excess_bord_arr)[0][0],1):np.where(excess_bord_arr)[0][0]+2, sub(np.where(excess_bord_arr)[1][0],1):np.where(excess_bord_arr)[1][0]+2, k]))+np.transpose(np.where(excess_bord_arr))[:,:2]-1) or
    # #                                    ((i,j) in np.transpose(np.where(extra_type2 * cleanborders[sub(np.where(excess_bord_arr)[0][0],1):np.where(excess_bord_arr)[0][0]+2, sub(np.where(excess_bord_arr)[1][0],1):np.where(excess_bord_arr)[1][0]+2, k]))+np.transpose(np.where(excess_bord_arr))[:,:2]-1) or
    # #                                    ((i,j) in np.transpose(np.where(extra_type3 * cleanborders[sub(np.where(excess_bord_arr)[0][0],1):np.where(excess_bord_arr)[0][0]+2, sub(np.where(excess_bord_arr)[1][0],1):np.where(excess_bord_arr)[1][0]+2, k]))+np.transpose(np.where(excess_bord_arr))[:,:2]-1) or
    # #                                    ((i,j) in np.transpose(np.where(extra_type4 * cleanborders[sub(np.where(excess_bord_arr)[0][0],1):np.where(excess_bord_arr)[0][0]+2, sub(np.where(excess_bord_arr)[1][0],1):np.where(excess_bord_arr)[1][0]+2, k]))+np.transpose(np.where(excess_bord_arr))[:,:2]-1)
    # #                                    ):
    # #                                    cleanborders[sub(i,1):i+2, sub(j,1):j+2, k:k+1] += excess_bord_arr[sub(i,1):i+2, sub(j,1):j+2, k:k+1]
    # #                                    cleanborders[i,j,k] = 0
    
    #                         #print i,j,k
    # #                            if actual_excess == 8:
    # #                                print count, "actual_excess:%d"%actual_excess, (i,j,k), 
    # #                                excess_bord_arr[np.transpose(np.where(excess_bord_arr))[:,:2][0][0], np.transpose(np.where(excess_bord_arr))[:,:2][0][1], k] = 0
    # #        curve_excess_bord[np.transpose(np.where(curve_excess_bord))[:,:2][0][0], np.transpose(np.where(curve_excess_bord))[:,:2][0][1], k] = 0
    
    #         if curve_excess_bord.any():
    #             curve_excess_bord[tuple(np.transpose(np.where(curve_excess_bord))[0])] = 0
    # #        else:
    # #            break
    # #        count += 1
    #         count -= 1
    # #        if curve_excess_bord.any():            
    # #            curve_excess_bord[tuple(np.transpose(np.where(curve_excess_bord))[0])] = 0
    #     num_cleanings += 1
        
            
    ### Remove any pixels that are discontinuous with the cell border or that
    ### result in a cusp:
    # for (i,j,k), surfval in np.ndenumerate(borders_of_interest):
    #     if surfval != 0:
    #         if (len(np.where(borders_of_interest[sub(i,1):i+2, sub(j,1):j+2, sub(k,1):k+2] == surfval)[0]) < 3
    #             ):
    #                 borders_of_interest[i,j,k] = 0
 
    borders_of_interest = pad_img_edges(borders_of_interest, pad_val, -1*pad_width)                   
    return borders_of_interest
###



def count_pixels_around_borders_sequentially(cleanborders, cell1_id, cell2_id, channel1, channel2):
    """ This function takes an RGB image of a thinned border with no L-shaped
     corners and will tell you the perimeter and sequentially number the borders.
     You need to provide the 2 cells to be processed with a unique cell_id
     (I use 1 for red, 2 for green, and 3 for blue channel cells). You also need
     to say which 2 channels you want to analyze (0 for red, 1 for green, 2 for
     blue."""
    diag_struct_elem = np.array([[0,1,0], [1,0,1], [0,1,0]])
    diag_struct_elem = np.reshape( diag_struct_elem, (3,3))#,1))
    
    cross_struct_elem = np.array([[1,0,1], [0,0,0], [1,0,1]])
    cross_struct_elem = np.reshape( cross_struct_elem, (3,3))#,1))
    
    perimeter_cell1 = 0
    perimeter_cell2 = 0
    perimeter_arr = np.copy(cleanborders).astype(float)
    
    ### Remove 1 pixel from the borders so that the perimeter tracker has somewhere to start
    if cell1_id in perimeter_arr:
        first_indice = list(zip(*np.where(perimeter_arr == cell1_id)))[0]
        first_indice_arr = perimeter_arr[sub(first_indice[0],1):first_indice[0]+2, sub(first_indice[1],1):first_indice[1]+2, first_indice[2]]
        perimeter_cell1 += len(np.where(first_indice_arr * cross_struct_elem)[0]==1) * 1.0 + len(np.where(first_indice_arr * diag_struct_elem)[0]==1) * math.sqrt(2)
        perimeter_arr[first_indice] = 0
        for (i,j,k), value in np.ndenumerate(perimeter_arr):
            if value == cell1_id and (np.count_nonzero(perimeter_arr[sub(i,1):i+2, sub(j,1):j+2, k]) == 2):
                second_indice = (i,j,k)
                break
        perimeter_arr[second_indice[0], second_indice[1], second_indice[2]] *= -1
        ordered_border_indices_red = [(first_indice)]
    else:
        ordered_border_indices_red = [(float('nan'), float('nan'), float('nan')),]
        
    if cell2_id in perimeter_arr:
        first_indice = list(zip(*np.where(perimeter_arr == cell2_id)))[0]
        first_indice_arr = perimeter_arr[sub(first_indice[0],1):first_indice[0]+2, sub(first_indice[1],1):first_indice[1]+2, first_indice[2]]
        perimeter_cell2 += len(np.where(first_indice_arr * cross_struct_elem)[0]==2) * 1.0 + len(np.where(first_indice_arr * diag_struct_elem)[0]==2) * math.sqrt(2)
        perimeter_arr[first_indice] = 0
        for (i,j,k), value in np.ndenumerate(perimeter_arr):
            if value == cell2_id and (np.count_nonzero(perimeter_arr[sub(i,1):i+2, sub(j,1):j+2, k]) == 2):
                second_indice = (i,j,k)
                break
        perimeter_arr[second_indice[0], second_indice[1], second_indice[2]] *= -1
        ordered_border_indices_blu = [(first_indice)]
    else:
        ordered_border_indices_blu = [(float('nan'), float('nan'), float('nan')),]
        
    count = 0
    while (np.count_nonzero(perimeter_arr[:,:,channel1]) > 1 or np.count_nonzero(perimeter_arr[:,:,channel2]) > 1):
        for (i,j,k), value in np.ndenumerate(perimeter_arr):
            if value != 0 and (np.count_nonzero(perimeter_arr[sub(i,1):i+2, sub(j,1):j+2, k]) == 2):
                if value == -1 * cell1_id:
                    if np.count_nonzero(perimeter_arr[sub(i,1):i+2, sub(j,1):j+2, k] * cross_struct_elem) > 0:
                        perimeter_arr[i,j,k] = 0
                        perimeter_arr[sub(i,1):i+2, sub(j,1):j+2, k] *= -1.0 * cross_struct_elem.astype(float)
                        perimeter_cell1 += math.sqrt(2)
                        ordered_border_indices_red.append((i,j,k))
                    elif np.count_nonzero(perimeter_arr[sub(i,1):i+2, sub(j,1):j+2, k] * diag_struct_elem) > 0:
                        perimeter_arr[i,j,k] = 0
                        perimeter_arr[sub(i,1):i+2, sub(j,1):j+2, k] *= -1.0 * diag_struct_elem.astype(float)
                        perimeter_cell1 += 1.0
                        ordered_border_indices_red.append((i,j,k))
                elif value == -1 * cell2_id:
                    if np.count_nonzero(perimeter_arr[sub(i,1):i+2, sub(j,1):j+2, k] * cross_struct_elem) > 0:
                        perimeter_arr[i,j,k] = 0
                        perimeter_arr[sub(i,1):i+2, sub(j,1):j+2, k] *= -1.0 * cross_struct_elem.astype(float)
                        perimeter_cell2 += math.sqrt(2)
                        ordered_border_indices_blu.append((i,j,k))
                    elif np.count_nonzero(perimeter_arr[sub(i,1):i+2, sub(j,1):j+2, k] * diag_struct_elem) > 0:
                        perimeter_arr[i,j,k] = 0
                        perimeter_arr[sub(i,1):i+2, sub(j,1):j+2, k] *= -1.0 * diag_struct_elem.astype(float)
                        perimeter_cell2 += 1.0
                        ordered_border_indices_blu.append((i,j,k))
        count += 1
        print("filenumber:%s"%(filenumber+1), '/', total_files, ',', "count:%s"%count)
        if count > 400:
            print("I couldn't resolve the perimeter of the cell.")
            messages.append("I couldn't resolve the perimeter of the cell.")
            perimeter_cell1 = np.float('nan')
            perimeter_cell2 = np.float('nan')
            break

    if len(np.where(perimeter_arr == -1 * cell1_id)[0]) == 1:
        ordered_border_indices_red.append(list(zip(*np.where(perimeter_arr == -1 * cell1_id)))[0])
    if len(np.where(perimeter_arr == -1 * cell2_id)[0]) == 1:
        ordered_border_indices_blu.append(list(zip(*np.where(perimeter_arr == -1 * cell2_id)))[0])
        
        
    numbered_border_indices_red = [x for x in enumerate(ordered_border_indices_red)]
    numbered_border_indices_blu = [x for x in enumerate(ordered_border_indices_blu)]
    
    return ordered_border_indices_red, ordered_border_indices_blu, numbered_border_indices_red, numbered_border_indices_blu, perimeter_cell1, perimeter_cell2
###
    


def find_closest_pixels_indices_between_cell_borders(redborder, bluborder, ordered_border_indices_red, ordered_border_indices_blu, mean_closest_indice_loc_red_old, mean_closest_indice_loc_blu_old):
    redborder_xy = list(zip(*np.where(redborder)))
    bluborder_x, bluborder_y = np.where(bluborder)

    #if redborder_xy.any() and bluborder_x.any() and bluborder_y.any():
    dif_x = []
    dif_y = []
    for i in redborder_xy:
        dif_x.append([(j - i[0])**2 for j in bluborder_x])
        dif_y.append([(j - i[1])**2 for j in bluborder_y])
    dist_red_blu = np.asarray(dif_x) + np.asarray(dif_y)
    closest_redborder_indices_loc = np.where(dist_red_blu == np.min(dist_red_blu))[0].astype(int)
    closest_bluborder_indices_loc = np.where(dist_red_blu == np.min(dist_red_blu))[1].astype(int)
    closest_redborder_indices_rc = np.asarray(redborder_xy)[closest_redborder_indices_loc]
    closest_bluborder_indices_rc = np.column_stack((np.asarray(bluborder_x)[closest_bluborder_indices_loc], np.asarray(bluborder_y)[closest_bluborder_indices_loc]))
    
    ###

    ordered_border_indices_red_arr = np.asarray(ordered_border_indices_red)
    ordered_border_indices_blu_arr = np.asarray(ordered_border_indices_blu)
    
    ordered_border_red_closest_index_first = np.where(np.all(ordered_border_indices_red_arr[:,0:2] == closest_redborder_indices_rc[0], axis=1))[0]
    ordered_border_blu_closest_index_first = np.where(np.all(ordered_border_indices_blu_arr[:,0:2] == closest_bluborder_indices_rc[0], axis=1))[0]
    
    ordered_border_red_closest_index_last = np.where(np.all(ordered_border_indices_red_arr[:,0:2] == closest_redborder_indices_rc[-1], axis=1))[0]
    ordered_border_blu_closest_index_last = np.where(np.all(ordered_border_indices_blu_arr[:,0:2] == closest_bluborder_indices_rc[-1], axis=1))[0]
    
    ###
    
    arr_red = np.sort(np.array((0, ordered_border_red_closest_index_first, ordered_border_red_closest_index_last, len(ordered_border_indices_red))))
    if np.diff(arr_red)[1] < (np.diff(arr_red)[0] + np.diff(arr_red)[2]):
        mean_closest_indice_loc_red = np.mean(arr_red[1:3])
    elif np.diff(arr_red)[1] > (np.diff(arr_red)[0] + np.diff(arr_red)[2]):
        difference_red = (np.diff(arr_red)[0]+np.diff(arr_red)[2])
        mean_closest_indice_loc_red = arr_red[2] + difference_red / 2
        if mean_closest_indice_loc_red > len(ordered_border_indices_red) - 1:
            mean_closest_indice_loc_red -= ((len(ordered_border_indices_red) - 1) - 1)
                ### The extra -1 above is because 0 is a viable index when you loop,
                ### otherwise, the mean index will be overestimated by 1.
    if mean_closest_indice_loc_red % 1 == 0:
        # mean_closest_indice_loc_red = np.array([mean_closest_indice_loc_red])
        pass
    elif mean_closest_indice_loc_red != 0:
        # mean_closest_indice_loc_red = np.array([mean_closest_indice_loc_red - 0.5, mean_closest_indice_loc_red + 0.5])
        mean_closest_indice_loc_red = np.concatenate([mean_closest_indice_loc_red - 0.5, mean_closest_indice_loc_red + 0.5]).astype(int)
    mean_closest_indice_loc_red = mean_closest_indice_loc_red.astype(int)
        
    arr_blu = np.sort(np.array((0, ordered_border_blu_closest_index_first, ordered_border_blu_closest_index_last, len(ordered_border_indices_blu))))
    if np.diff(arr_blu)[1] < (np.diff(arr_blu)[0] + np.diff(arr_blu)[2]):
        mean_closest_indice_loc_blu = np.mean(arr_blu[1:3])
    elif np.diff(arr_blu)[1] > (np.diff(arr_blu)[0] + np.diff(arr_blu)[2]):
        difference_blu = (np.diff(arr_blu)[0]+np.diff(arr_blu)[2])
        mean_closest_indice_loc_blu = arr_blu[2] + difference_blu / 2
        if mean_closest_indice_loc_blu > len(ordered_border_indices_blu) - 1:
            mean_closest_indice_loc_blu -= ((len(ordered_border_indices_blu) - 1) - 1)
                ### The extra -1 above is because 0 is a viable index when you loop,
                ### otherwise, the mean indexx will be overestimated by 1.
    if mean_closest_indice_loc_blu % 1 == 0:
        # mean_closest_indice_loc_blu = np.array([mean_closest_indice_loc_blu])
        pass
    elif mean_closest_indice_loc_blu != 0:
        # mean_closest_indice_loc_blu = np.array([mean_closest_indice_loc_blu - 0.5, mean_closest_indice_loc_blu + 0.5])
        mean_closest_indice_loc_blu = np.concatenate([mean_closest_indice_loc_blu - 0.5, mean_closest_indice_loc_blu + 0.5])
    mean_closest_indice_loc_blu = mean_closest_indice_loc_blu.astype(int)
    
    #else:
    #    closest_redborder_indices_rc = np.asarray(redborder_xy)[np.array([0])]
    #    closest_bluborder_indices_rc = np.column_stack(( np.asarray(np.float('nan')), np.asarray(np.float('nan')) ))
    if len(mean_closest_indice_loc_red) >= 1:
        closest_redborder_idx_rc = ordered_border_indices_red_arr[mean_closest_indice_loc_red[:1], 0:2]
    else:
        closest_redborder_idx_rc = np.array([[np.nan, np.nan]])
    
    if len(mean_closest_indice_loc_blu) >= 1:
        closest_bluborder_idx_rc = ordered_border_indices_blu_arr[mean_closest_indice_loc_blu[:1], 0:2]
    else:
        closest_bluborder_idx_rc = np.array([[np.nan, np.nan]])
    
    # return closest_redborder_indices_rc, closest_bluborder_indices_rc, mean_closest_indice_loc_red, mean_closest_indice_loc_blu
    return closest_redborder_idx_rc, closest_bluborder_idx_rc, mean_closest_indice_loc_red, mean_closest_indice_loc_blu
###
    


def create_and_remove_bin_closest_to_other_cell(curveborder_array, ordered_border_indices, mean_closest_indice_loc, num_bins):
    closest_bin_size = np.count_nonzero( (curveborder_array) == True) // (num_bins)
    remainder = np.count_nonzero( (curveborder_array) == True) % (num_bins)
    remainder_as_even = round_down_to_even(remainder)
    
    size_list_of_other_bins = [closest_bin_size] * (num_bins-1)
    
    count = remainder_as_even
    i = 1
    while count > 0:
        size_list_of_other_bins[i-1] += 1
        size_list_of_other_bins[-i] += 1
        i += 1
        count -= 2
    size_list_of_other_bins[(num_bins - 2) // 2] += remainder - remainder_as_even
    ### The above will work, but I next need to re-assign the redborder 
    ### and grnborder sequences
    ### First need to find what constitutes the bin closest to the other cell and
    ### Remove it from redborders and grnborders:
    closest_bin_centre = mean_closest_indice_loc.astype(int)[-1]
    
    ### The following 2 if statements say what to do if the bin edges need to go
    ### outside of the available indices - ie. need to loop:
    if closest_bin_centre - closest_bin_size//2 >= 0:
        if closest_bin_centre + closest_bin_size - closest_bin_size//2 < np.count_nonzero( curveborder_array == True):
            closest_bin_indices = ordered_border_indices[int(closest_bin_centre - closest_bin_size//2):int(closest_bin_centre + closest_bin_size - closest_bin_size//2)]
            closest_bin_indices_locs = list(range(int(closest_bin_centre - closest_bin_size//2), int(closest_bin_centre + closest_bin_size - closest_bin_size//2)))
        elif closest_bin_centre + closest_bin_size - closest_bin_size//2 >= np.count_nonzero( curveborder_array == True):
            closest_bin_indices = ordered_border_indices[int(closest_bin_centre - closest_bin_size//2):(np.count_nonzero( curveborder_array == True))]
            closest_bin_indices += ordered_border_indices[0:int(closest_bin_centre + closest_bin_size - closest_bin_size//2 - np.count_nonzero( curveborder_array == True))]
            closest_bin_indices_locs = list(range(int(closest_bin_centre - closest_bin_size//2), (np.count_nonzero( curveborder_array == True))))
            closest_bin_indices_locs += list(range(0, int(closest_bin_centre + closest_bin_size - closest_bin_size//2 - np.count_nonzero( curveborder_array == True))))
    elif closest_bin_centre - closest_bin_size//2 < 0:
        if closest_bin_centre + closest_bin_size - closest_bin_size//2 < np.count_nonzero( curveborder_array == True):
            closest_bin_indices = ordered_border_indices[int(np.count_nonzero(curveborder_array==True) + closest_bin_centre - closest_bin_size//2):(np.count_nonzero(curveborder_array==True))]
            closest_bin_indices += ordered_border_indices[0:int(closest_bin_centre + closest_bin_size - closest_bin_size//2)]
            closest_bin_indices_locs = list(range(int(np.count_nonzero(curveborder_array==True) + closest_bin_centre - closest_bin_size//2), (np.count_nonzero(curveborder_array==True))))
            closest_bin_indices_locs += list(range(0, int(closest_bin_centre + closest_bin_size - closest_bin_size//2)))
    curveborder_array[np.asarray(closest_bin_indices)[:,0], np.asarray(closest_bin_indices)[:,1]] = 0
    
    return curveborder_array, size_list_of_other_bins, closest_bin_size, closest_bin_indices
###
    


def create_border_bin_sizes(border_array, ordered_border_indices, num_bins):
    starting_bin_size = np.count_nonzero( (border_array) == True) / (num_bins)
    remainder = np.count_nonzero( (border_array) == True) % (num_bins)
    remainder_as_even = round_down_to_even(remainder)
    
    bin_size_list = [starting_bin_size] * num_bins
    
    count = remainder_as_even
    i = 1
    while count > 0:
        bin_size_list[i-1] += 1
        bin_size_list[-i] += 1
        i += 1
        count -= 2
    bin_size_list[int((num_bins - 1) / 2)] += remainder - remainder_as_even
    
#    bin_start = []
#    bin_stop = []
#    for i in xrange(num_bins):
#        bin_start_red.append(np.sum(bin_size_list[:i], dtype=int))
#       bin_stop_red.append(np.sum(bin_size_list[:i+1], dtype=int))

#    bin_indices = []
#    for i in xrange(num_bins):
#        bin_indices_red.append(ordered_border_indices[bin_start[i]:bin_stop[i]])

    return bin_size_list
###
    


def make_open_border_indices_sequential(open_borders_arr, red_id, blu_id, channel1, channel2):
    # time.sleep(0.05)
    diag_struct_elem = np.array([[0,1,0], [1,0,1], [0,1,0]])
    diag_struct_elem = np.reshape( diag_struct_elem, (3,3))#,1))
    
    cross_struct_elem = np.array([[1,0,1], [0,0,0], [1,0,1]])
    cross_struct_elem = np.reshape( cross_struct_elem, (3,3))#,1))
    
    pad_val = 0
    pad_width = 1
    perimeter_arr = pad_img_edges(np.copy(open_borders_arr), pad_val, pad_width).astype(float)
    # perimeter_arr = np.copy(open_borders_arr).astype(float)
    perimeter_cell1 = 0
    perimeter_cell2 = 0
    if red_id in perimeter_arr:
        for (i,j,k), value in np.ndenumerate(perimeter_arr):
            if value == red_id and (np.count_nonzero(perimeter_arr[sub(i,1):i+2, sub(j,1):j+2, k]) == 2):
                second_indice = (i,j,k)
                break
        perimeter_arr[second_indice[0], second_indice[1], second_indice[2]] *= -1
        reordered_border_indices_red = []
    else:
        reordered_border_indices_red = [(np.nan, np.nan, np.nan)]
    if blu_id in perimeter_arr:
        for (i,j,k), value in np.ndenumerate(perimeter_arr):
            if value == blu_id and (np.count_nonzero(perimeter_arr[sub(i,1):i+2, sub(j,1):j+2, k]) == 2):
                second_indice = (i,j,k)
                break
        perimeter_arr[second_indice[0], second_indice[1], second_indice[2]] *= -1
        reordered_border_indices_blu = []
    else:
        reordered_border_indices_blu = [(np.nan, np.nan, np.nan)]
    
    count = 0
    while (np.count_nonzero(perimeter_arr[:,:,channel1]) > 1 or np.count_nonzero(perimeter_arr[:,:,channel2]) > 1):
        for (i,j,k), value in np.ndenumerate(perimeter_arr):
            if value != 0 and (np.count_nonzero(perimeter_arr[sub(i,1):i+2, sub(j,1):j+2, k]) == 2):
                if value == -1 * red_id:
                    if np.count_nonzero(perimeter_arr[sub(i,1):i+2, sub(j,1):j+2, k] * cross_struct_elem) > 0:
                        perimeter_arr[i,j,k] = 0
                        perimeter_arr[sub(i,1):i+2, sub(j,1):j+2, k] *= -1.0 * cross_struct_elem.astype(float)
                        perimeter_cell1 += math.sqrt(2)
                        reordered_border_indices_red.append((i,j,k))
                    elif np.count_nonzero(perimeter_arr[sub(i,1):i+2, sub(j,1):j+2, k] * diag_struct_elem) > 0:
                        perimeter_arr[i,j,k] = 0
                        perimeter_arr[sub(i,1):i+2, sub(j,1):j+2, k] *= -1.0 * diag_struct_elem.astype(float)
                        perimeter_cell1 += 1.0
                        reordered_border_indices_red.append((i,j,k))
                elif value == -1 * blu_id:
                    if np.count_nonzero(perimeter_arr[sub(i,1):i+2, sub(j,1):j+2, k] * cross_struct_elem) > 0:
                        perimeter_arr[i,j,k] = 0
                        perimeter_arr[sub(i,1):i+2, sub(j,1):j+2, k] *= -1.0 * cross_struct_elem.astype(float)
                        perimeter_cell2 += math.sqrt(2)
                        reordered_border_indices_blu.append((i,j,k))
                    elif np.count_nonzero(perimeter_arr[sub(i,1):i+2, sub(j,1):j+2, k] * diag_struct_elem) > 0:
                        perimeter_arr[i,j,k] = 0
                        perimeter_arr[sub(i,1):i+2, sub(j,1):j+2, k] *= -1.0 * diag_struct_elem.astype(float)
                        perimeter_cell2 += 1.0
                        reordered_border_indices_blu.append((i,j,k))
        count += 1
        print("filenumber:%s"%(filenumber+1), '/', total_files, ',', "count:%s"%count)
        if count > 400:
            print("I couldn't resolve the perimeter of the cell.")
            messages.append("I couldn't resolve the perimeter of the cell.")
            perimeter_cell1 = np.float('nan')
            perimeter_cell2 = np.float('nan')
            break
    
    if len(np.where(perimeter_arr == -1 * red_id)[0]) == 1:
        reordered_border_indices_red.append(list(zip(*np.where(perimeter_arr == -1 * red_id)))[0])
    if len(np.where(perimeter_arr == -1 * blu_id)[0]) == 1:
        reordered_border_indices_blu.append(list(zip(*np.where(perimeter_arr == -1 * blu_id)))[0])
    
    # We now need to undo the effects of the img edge padding that was done at 
    # the start of the function so that the reordered border indices will 
    # coincide with the original image (the effects are that all indices are
    # shifted one to the right and one down):
    if reordered_border_indices_red:
        reordered_border_indices_red_arr = np.asarray(list(zip(*reordered_border_indices_red)))[:,:]
        rows_fixed = np.asarray(list(zip(*reordered_border_indices_red)))[0,:] - pad_width
        cols_fixed = np.asarray(list(zip(*reordered_border_indices_red)))[1,:] - pad_width
        reordered_border_indices_red_arr[0,:] = rows_fixed
        reordered_border_indices_red_arr[1,:] = cols_fixed
        reordered_border_indices_red_fixed = list(zip(*reordered_border_indices_red_arr))
    
    if reordered_border_indices_blu:
        reordered_border_indices_blu_arr = np.asarray(list(zip(*reordered_border_indices_blu)))[:,:]
        rows_fixed = np.asarray(list(zip(*reordered_border_indices_blu)))[0,:] - pad_width
        cols_fixed = np.asarray(list(zip(*reordered_border_indices_blu)))[1,:] - pad_width
        reordered_border_indices_blu_arr[0,:] = rows_fixed
        reordered_border_indices_blu_arr[1,:] = cols_fixed
        reordered_border_indices_blu_fixed = list(zip(*reordered_border_indices_blu_arr))
    
    # return reordered_border_indices_red, reordered_border_indices_blu, perimeter_cell1, perimeter_cell2
    return reordered_border_indices_red_fixed, reordered_border_indices_blu_fixed, perimeter_cell1, perimeter_cell2
###
    


def sort_pixels_into_bins(reordered_border_indices, size_list_of_other_bins, closest_bin_indices, num_bins):
    bin_start = []
    bin_stop = []
    for i in range(num_bins):
        bin_start.append(np.sum(size_list_of_other_bins[:i], dtype=int))
        bin_stop.append(np.sum(size_list_of_other_bins[:i+1], dtype=int))
    bin_indices = []
    for i in range(num_bins):
        bin_indices.append(reordered_border_indices[bin_start[i]:bin_stop[i]])
    bin_indices.append(closest_bin_indices)
    #size_list_of_all_bins_red.append(len(bin_indices_red[7]))
    return bin_indices, bin_start, bin_stop
###
    


### The following line is global because it's used later on in the script:
bin_curvature_details_headers = ['bin_#','bin_color', '1 * curvature_sign / R_3', '1 / R_3', 'curvature_sign', 'xc_3', 'yc_3', 'R_3', 'Ri_3', 'residu_3', 'size_of_bin', 'x_fit_arc_start', 'y_fit_arc_start', 'x_fit_arc_stop', 'y_fit_arc_stop', 'arc_angle (in radians)', 'fitted curvature arclength']
def fit_curves_to_bins(bin_indices, one_channel_of_curveborders, odr_circle_fit_iterations):
    # time.sleep(0.05)
    bin_curvature_details = []
    bin_curvature_details.append(bin_curvature_details_headers)
#    binned_curvature_fig = plt.figure()
#    binned_curvature_ax = binned_curvature_fig.add_subplot(111)
    
    all_fits_all_bins = []
    color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'silver']
    for i in range(len(bin_indices)):#len(bin_indices)
        x,y = list(zip(*bin_indices[i]))[:2]
        
        # initial guess for parameters
        x_m = np.mean(x)
        y_m = np.mean(y)
        R_m = calc_R(x, y, x_m, y_m).mean()
        
        x_m_o = np.copy(x_m)
        y_m_o = np.copy(y_m)
#        R_m_o = np.copy(R_m)
#        N_m = 
        beta0 = [ x_m, y_m, R_m]
        
        # for implicit function :
        #       data.x contains both coordinates of the points (data.x = [x, y])
        #       data.y is the dimensionality of the response
        lsc_data  = odr.Data(np.row_stack([x, y]), y=1)
        
        lsc_model = odr.Model(f_3, implicit=True)
        lsc_odr   = odr.ODR(lsc_data, lsc_model, beta0)
        lsc_out   = lsc_odr.run()
        
#        lsc_model_lin = odr.Model(f_lin, implicit=True)
#        lsc_odr_lin = odr.ODR(lsc_data, lsc_model_lin, beta0)
#        lsc_out_lin = lsc_odr_lin.run()
        
        xc_3, yc_3, R_3 = lsc_out.beta
        Ri_3 = calc_R(x, y, xc_3, yc_3)
        residu_3 = sum((Ri_3 - R_3)**2)
        ###
        ###
        count_fit = 0
        all_fits_list = [(count_fit, xc_3, yc_3, R_3, Ri_3, residu_3)]
        print("filenumber:%s"%(filenumber+1), '/', total_files, ',', 'bin #%s'%(i+1), '/', len(bin_indices), "count %s"%count_fit,':', round(xc_3, 5), round(yc_3, 5), round(R_3, 5), round(residu_3, 5))
        while count_fit < odr_circle_fit_iterations:
            beta0 = [ xc_3, yc_3, R_3 ]
            lsc_data  = odr.Data(np.row_stack([x, y]), y=1)
            lsc_model = odr.Model(f_3, implicit=True)
            lsc_odr   = odr.ODR(lsc_data, lsc_model, beta0)
            lsc_out   = lsc_odr.run()
            
            xc_3, yc_3, R_3 = lsc_out.beta
            Ri_3 = calc_R(x, y, xc_3, yc_3)
            residu_3 = sum((Ri_3 - R_3)**2)
            count_fit += 1
            all_fits_list.append((count_fit, xc_3, yc_3, R_3, Ri_3, residu_3))
            print("filenumber:%s"%(filenumber+1), '/', total_files, ',', 'bin #%s'%(i+1), '/', len(bin_indices), "count %s"%count_fit,':', round(xc_3, 5), round(yc_3, 5), round(R_3, 5), round(residu_3, 5))
        all_fits = np.asarray(all_fits_list, dtype='object')
        count_fit, xc_3, yc_3, R_3, Ri_3, residu_3 = all_fits[ np.where(all_fits[:,5] == np.min(all_fits[:,5])), : ][0][0]
##        
        vec_x, vec_y = xc_3 - x_m_o , yc_3 - y_m_o
        xc_3, yc_3 = x_m_o + vec_x * -1.0 , y_m_o + vec_y * -1.0
        Ri_3 = calc_R(x, y, xc_3, yc_3)
        residu_3 = sum((Ri_3 - R_3)**2)
        count_fit = 0
        print("filenumber:%s"%(filenumber+1), '/', total_files, ',', 'bin #%s'%(i+1), '/', len(bin_indices), "count %s"%count_fit,':', round(xc_3, 5), round(yc_3, 5), round(R_3, 5), round(residu_3, 5))
        while count_fit < odr_circle_fit_iterations:
            beta0 = [ xc_3, yc_3, R_3 ]
            lsc_data  = odr.Data(np.row_stack([x, y]), y=1)
            lsc_model = odr.Model(f_3, implicit=True)
            lsc_odr   = odr.ODR(lsc_data, lsc_model, beta0)
            lsc_out   = lsc_odr.run()
            
            xc_3, yc_3, R_3 = lsc_out.beta
            Ri_3 = calc_R(x, y, xc_3, yc_3)
            residu_3 = sum((Ri_3 - R_3)**2)
            count_fit += 1
            all_fits_list.append((count_fit, xc_3, yc_3, R_3, Ri_3, residu_3))
            print("filenumber:%s"%(filenumber+1), '/', total_files, ',', 'bin #%s'%(i+1), '/', len(bin_indices), "count %s"%count_fit,':', round(xc_3, 5), round(yc_3, 5), round(R_3, 5), round(residu_3, 5))
        all_fits = np.asarray(all_fits_list, dtype='object')
        all_fits_all_bins.append(all_fits_list)
        count_fit, xc_3, yc_3, R_3, Ri_3, residu_3 = all_fits[ np.where(all_fits[:,5] == np.min(all_fits[:,5])), : ][0][0]


        theta = np.linspace(0, 2*np.pi, 72001)
        x_fit = R_3 * np.cos(theta) + xc_3
        y_fit = R_3 * np.sin(theta) + yc_3
        
        if len(bin_indices) == 1:
            curvature_sign = 1.0
            x_fit_arc_start_stop = np.array([float('nan'), float('nan')])
            y_fit_arc_start_stop = np.array([float('nan'), float('nan')])
            arc_angle = 2.0 * np.pi
            curvature_fit_arclength = 2.0 * np.pi * R_3
            
        elif len(bin_indices) > 1:
            x_fit_looped = np.hstack((x_fit, x_fit))
            y_fit_looped = np.hstack((y_fit, y_fit))
            
            distance_to_first_data_in_bin = ((x[0]-x_fit_looped)**2 + (y[0]-y_fit_looped)**2)
            distance_to_last_data_in_bin = ((x[-1]-x_fit_looped)**2 + (y[-1]-y_fit_looped)**2)
            fit_interval_start = np.where(distance_to_first_data_in_bin == np.min(distance_to_first_data_in_bin))
            fit_interval_stop = np.where(distance_to_last_data_in_bin == np.min(distance_to_last_data_in_bin))
            fit_interval = np.ones((x_fit_looped.shape), dtype='bool')
            if fit_interval_start[0][0] < fit_interval_stop[0][0]:
                fit_interval_1 = np.copy(fit_interval)
                fit_interval_2 = np.copy(fit_interval)
                fit_interval_1[fit_interval_start[0][0]:fit_interval_stop[0][0]] = False
                fit_interval_2[fit_interval_stop[0][0]:fit_interval_start[0][1]] = False
            elif fit_interval_start[0][0] > fit_interval_stop[0][0]:
                fit_interval_1 = np.copy(fit_interval)
                fit_interval_2 = np.copy(fit_interval)
                fit_interval_1[fit_interval_stop[0][0]:fit_interval_start[0][0]] = False
                fit_interval_2[fit_interval_start[0][0]:fit_interval_stop[0][1]] = False
        
            ### Find which fit_interval is spacially closer to the data points and
            ### use that one:        
            fit_arc_1 = []
            fit_arc_2 = []
            x_fit_arc_1 = np.ma.masked_array(x_fit_looped, fit_interval_1)
            y_fit_arc_1 = np.ma.masked_array(y_fit_looped, fit_interval_1)
            x_fit_arc_2 = np.ma.masked_array(x_fit_looped, fit_interval_2)
            y_fit_arc_2 = np.ma.masked_array(y_fit_looped, fit_interval_2)
            
            for j in range(len(x)):
                fit_arc_1.append( np.ma.MaskedArray.min((x_fit_arc_1 - x[j])**2 + (y_fit_arc_1 - y[j])**2) )
                fit_arc_2.append( np.ma.MaskedArray.min((x_fit_arc_2 - x[j])**2 + (y_fit_arc_2 - y[j])**2) )
            if np.median(fit_arc_1) < np.median(fit_arc_2):
                x_fit_arc = np.ma.MaskedArray.copy(x_fit_arc_1)
                y_fit_arc = np.ma.MaskedArray.copy(y_fit_arc_1)
            elif np.median(fit_arc_1) > np.median(fit_arc_2):
                x_fit_arc = np.ma.MaskedArray.copy(x_fit_arc_2)
                y_fit_arc = np.ma.MaskedArray.copy(y_fit_arc_2)
                    
            if len(np.ma.notmasked_contiguous(x_fit_arc)) == 1:
                x_fit_arc_edge_indices = np.ma.notmasked_edges(x_fit_arc)
            elif len(np.ma.notmasked_contiguous(x_fit_arc)) == 2:
                x_fit_arc_edge_indices = np.r_[np.ma.notmasked_contiguous(x_fit_arc)[0]][-1],np.r_[np.ma.notmasked_contiguous(x_fit_arc)[1]][0]
            if len(np.ma.notmasked_contiguous(y_fit_arc)) == 1:
                y_fit_arc_edge_indices = np.ma.notmasked_edges(y_fit_arc)
            elif len(np.ma.notmasked_contiguous(y_fit_arc)) == 2:
                y_fit_arc_edge_indices = np.r_[np.ma.notmasked_contiguous(y_fit_arc)[0]][-1],np.r_[np.ma.notmasked_contiguous(y_fit_arc)[1]][0]
                
            x_fit_arc_start_stop = x_fit_arc[np.array(x_fit_arc_edge_indices)]
            y_fit_arc_start_stop = y_fit_arc[np.array(y_fit_arc_edge_indices)]
            curvature_border = one_channel_of_curveborders
        
            curvature_sign = curvature_estimate_sign(x_fit_arc_start_stop, y_fit_arc_start_stop, x_fit_arc, y_fit_arc, curvature_border)
            arc_angle1 = np.arccos((x_fit_arc_start_stop[1] - xc_3) / R_3)
            arc_angle2 = np.arccos((x_fit_arc_start_stop[0] - xc_3) / R_3)
            arc_angle = abs(arc_angle1 - arc_angle2)
            curvature_fit_arclength = R_3 * arc_angle

        bin_curvature_details.append(['%s'%i, color[i], curvature_sign / R_3, 1.0 / R_3, curvature_sign, xc_3, yc_3, R_3, Ri_3, residu_3, [len(list(v)) for v in bin_indices][i], x_fit_arc_start_stop[0], y_fit_arc_start_stop[0], x_fit_arc_start_stop[1], y_fit_arc_start_stop[1], arc_angle, curvature_fit_arclength])
    
#        binned_curvature_ax.scatter(y,x, marker='.', lw=0, c=color[i])
#        binned_curvature_ax.plot(y_fit_arc, x_fit_arc, lw=0.5, c=color[i])
#        binned_curvature_ax.plot(y_fit_arc[np.array(y_fit_arc_edge_indices)], x_fit_arc[np.array(x_fit_arc_edge_indices)], lw=0.5, c=color[i])

#    plt.gca().set_aspect('equal', adjustable='box')
#    plt.gca().invert_yaxis()
    
    return bin_curvature_details, all_fits_all_bins#, binned_curvature_fig
###



def clean_up_contact_borders(rawcontactsurface, cleanborders, cell1_id, cell2_id):
    ### In the event that the contact surface is discontinuous (which happens
    ### occasionally believe it or not) we will fill in any holes by copying
    ### the cellborders values in to the window where the contactsurface exists:
    ###imin1 = min(np.where(rawcontactsurface==1)[0])-2 change to -5 as of 2018-06-21.

    # imin1 = min(np.where(rawcontactsurface == cell1_id)[0])-5
    # imax1 = max(np.where(rawcontactsurface == cell1_id)[0])+5

    # jmin1 = min(np.where(rawcontactsurface == cell1_id)[1])-5
    # jmax1 = max(np.where(rawcontactsurface == cell1_id)[1])+5

 

    # imin2 = min(np.where(rawcontactsurface == cell2_id)[0])-5
    # imax2 = max(np.where(rawcontactsurface == cell2_id)[0])+5

    # jmin2 = min(np.where(rawcontactsurface == cell2_id)[1])-5
    # jmax2 = max(np.where(rawcontactsurface == cell2_id)[1])+5


    # ### The part below will fill in the holes in each contact surface:
    # filledcontactsurface = np.zeros(rawcontactsurface.shape, np.dtype(np.int32))
    # filledcontactsurface[imin1:imax1, jmin1:jmax1, cell1_id-1] = cleanborders[imin1:imax1, jmin1:jmax1, cell1_id-1]
    # filledcontactsurface[imin2:imax2, jmin2:jmax2, cell2_id-1] = cleanborders[imin2:imax2, jmin2:jmax2, cell2_id-1]
    ### If the script works well for a while, delete the above
    
    ### A different approach: Label the noncontact borders of each cell
    ### and then if there is more than 1 label in each result, combine
    ### the smallest labels into the rawcontactsurface:
    # Find where borders don't touch:
    borders_no_touch = cleanborders - rawcontactsurface
    # Label cell borders:
    cell1_no_touch = morphology.label(borders_no_touch == cell1_id)
    cell2_no_touch = morphology.label(borders_no_touch == cell2_id)
    # If there is only 1 label in the no_touch borders, then the 
    # rawcontactsurface has no holes in it, so you can just copy 
    # the rawcontactsurface:
    filledcontactsurface = np.zeros(rawcontactsurface.shape, np.dtype(np.int32))
    filledcontactsurface[np.where(rawcontactsurface == cell1_id)] = cell1_id
    cell1_labels = np.unique(cell1_no_touch)[np.nonzero(np.unique(cell1_no_touch))]
    if np.count_nonzero(cell1_labels) <= 1:
        pass
    # Otherwise we will combine the shortest no_touch borders in
    # to the rawcontactsurface:
    elif np.count_nonzero(cell1_labels) > 1:
        longest_label_i = np.argmax([np.count_nonzero(cell1_no_touch == lab) for lab in cell1_labels])
        longest_label = cell1_labels[longest_label_i]
        filledcontactsurface[np.where(np.logical_and(cell1_no_touch != longest_label, 
                                                     cell1_no_touch != 0))] = cell1_id
    # Do this same check for cell2 as well:
    filledcontactsurface[np.where(rawcontactsurface == cell2_id)] = cell2_id
    cell2_labels = np.unique(cell2_no_touch)[np.nonzero(np.unique(cell2_no_touch))]
    if np.count_nonzero(cell2_labels) <= 1:
        pass
    elif np.count_nonzero(cell2_labels) > 1:
        longest_label_i = np.argmax([np.count_nonzero(cell2_no_touch == lab) for lab in cell2_labels])
        longest_label = cell2_labels[longest_label_i]
        filledcontactsurface[np.where(np.logical_and(cell2_no_touch != longest_label, 
                                                     cell2_no_touch != 0))] = cell2_id
        
    ### If the contact surface is particularly curved this can result in the
    ### contact surface being incorrectly extended for one cell, so chomp away at
    ### end of the contact surface if this is the case:
    false_contacts = np.zeros(filledcontactsurface.shape, np.dtype(np.int32))
    for (i,j,k), surfval in np.ndenumerate(filledcontactsurface):
        if surfval != 0:
            if (len(np.where(filledcontactsurface[sub(i,1):i+2, sub(j,1):j+2, :] == surfval)[0]) <= 2 and
            ((0 in filledcontactsurface[sub(i,1):i+2, sub(j,1):j+2, :]) and 
            (cell1_id in filledcontactsurface[sub(i,1):i+2, sub(j,1):j+2, :]) and 
            (cell2_id in filledcontactsurface[sub(i,1):i+2, sub(j,1):j+2, :])) == False
            ):
                false_contacts[i,j,k] = surfval
                filledcontactsurface[i,j,k] = 0
    
    while false_contacts.any():
        false_contacts = np.zeros(filledcontactsurface.shape)
        for (i,j,k), surfval in np.ndenumerate(filledcontactsurface):
            if surfval != 0:
                if (len(np.where(filledcontactsurface[sub(i,1):i+2, sub(j,1):j+2, :] == surfval)[0]) < 3 and
                ((0 in filledcontactsurface[sub(i,1):i+2, sub(j,1):j+2, :]) and 
                (cell1_id in filledcontactsurface[sub(i,1):i+2, sub(j,1):j+2, :]) and 
                (cell2_id in filledcontactsurface[sub(i,1):i+2, sub(j,1):j+2, :])) == False
                ):
                    false_contacts[i,j,k] = surfval
                    filledcontactsurface[i,j,k] = 0



    contactsurface = np.zeros(cleanareas.shape, np.dtype(np.int32))
    contactsurface = contactsurface + filledcontactsurface

    ### The part below should remove any "excess" pixels (ones where there are
    ### pixels that form a corner but could be simply connected by a diagonal)
    ### that remain on the contacting surfaces:
    extra_type1 = np.array([[0,1,0], [1,1,0], [0,0,0]])
    extra_type1 = np.reshape(extra_type1,(3,3,1))
    
    extra_type2 = np.array([[0,1,0], [0,1,1], [0,0,0]])
    extra_type2 = np.reshape(extra_type2, (3,3,1))
    
    extra_type3 = np.array([[0,0,0], [1,1,0], [0,1,0]])
    extra_type3 = np.reshape(extra_type3, (3,3,1))
    
    extra_type4 = np.array([[0,0,0], [0,1,1], [0,1,0]])
    extra_type4 = np.reshape(extra_type4, (3,3,1))
    
    excess_arr = np.zeros(cleanareas.shape, np.dtype(np.int32))
    
    count = 0
    while count < 8:
        for (i,j,k), surfval in np.ndenumerate(contactsurface):
            if surfval != 0:
                if (len(np.where(extra_type1 * contactsurface[sub(i,1):i+2, sub(j,1):j+2, :] == surfval)[0]) > 2 or
                    len(np.where(extra_type2 * contactsurface[sub(i,1):i+2, sub(j,1):j+2, :] == surfval)[0]) > 2 or
                    len(np.where(extra_type3 * contactsurface[sub(i,1):i+2, sub(j,1):j+2, :] == surfval)[0]) > 2 or
                    len(np.where(extra_type4 * contactsurface[sub(i,1):i+2, sub(j,1):j+2, :] == surfval)[0]) > 2
                    ):
                        excess_arr[i,j,k] += contactsurface[i,j,k]
                        contactsurface[i,j,k] = 0
    ### This can result in contactsurface ends that are not actually in contact 
    ### with the other cell, so we need to remove those and replace the falsely
    ### excess pixel: 
        for (i,j,k), surfval in np.ndenumerate(contactsurface):
            if surfval != 0:       
                if ((len(np.where(contactsurface[sub(i,1):i+2, sub(j,1):j+2, :] == surfval)[0]) == 2) and
                    (((0 in contactsurface[sub(i,1):i+2, sub(j,1):j+2, :]) and 
                    (cell1_id in contactsurface[sub(i,1):i+2, sub(j,1):j+2, :]) and 
                    (cell2_id in contactsurface[sub(i,1):i+2, sub( j,1):j+2, :])) == False) and
                    (contactsurface[i,j,k] in excess_arr[sub(i,1):i+2, sub(j,1):j+2, k])
                    ):
                        contactsurface[i,j,k] = 0
                        contactsurface[sub(i,1):i+2, sub(j,1):j+2, k:k+1] += excess_arr[sub(i,1):i+2, sub(j,1):j+2, k:k+1]
    ### This can also result in contactsurface ends that are not actually in contact
    ### with the noncontact borders, so we need to remove those and replace the falsely
    ### excess pixel:
        noncontactborders = cleanborders - contactsurface
        for (i,j,k), surfval in np.ndenumerate(contactsurface):
            if surfval != 0:
                if ((len(np.where(contactsurface[sub(i,1):i+2, sub(j,1):j+2, :] == surfval)[0]) == 2) and
                    (0 in contactsurface[sub(i,1):i+2, sub(j,1):j+2, :]) and 
                    (cell1_id in contactsurface[sub(i,1):i+2, sub(j,1):j+2, :]) and 
                    (cell2_id in contactsurface[sub(i,1):i+2, sub( j,1):j+2, :]) and
                    (contactsurface[i,j,k] in excess_arr[sub(i,1):i+2, sub(j,1):j+2, k]) and
                    ((noncontactborders-excess_arr)[sub(i,1):i+2, sub(j,1):j+2, k].any() == False)
                    ):
                        contactsurface[i,j,k] = 0
                        contactsurface[sub(i,1):i+2, sub(j,1):j+2, k:k+1] += excess_arr[sub(i,1):i+2, sub(j,1):j+2, k:k+1]
        for (i,j,k),v in np.ndenumerate(excess_arr):
            if v != 0:
                if excess_arr[i,j,k] == contactsurface[i,j,k]:
                    excess_arr[i,j,k] = 0

        count += 1
        
    for (i,j,k),v in np.ndenumerate(excess_arr):
        if v != 0:
            if excess_arr[i,j,k] == contactsurface[i,j,k]:
                excess_arr[i,j,k] = 0

    ### Now need to clean up those values for contact corners so that there aren't
    ### any adjacent values that are equal:
    ### First build a structuring element that will identify adjacent values:
        
    contactcorners = np.zeros(cleanareas.shape, np.dtype(np.int32))

    ### This should be the last part and should robustly find corners:
    for (i,j,k), value in np.ndenumerate(contactsurface):
        if value != 0:
            if len(np.where(contactsurface[sub(i,1):i+2, sub(j,1):j+2, :] == value)[0]) == 2:
                contactcorners[i,j,k] = contactsurface[i,j,k]

    return contactcorners, contactsurface, noncontactborders
###



def find_contact_angles(contactcorners, contactsurface, noncontactborders, cell1_id, cell2_id):    
    pathdestroyer = np.zeros(noncontactborders.shape, np.dtype(np.uint32))
    pathfinder = np.zeros(noncontactborders.shape, np.dtype(np.uint32))
    count = 0
    while count < (Nth_pixel - 1):
        ### The 'while' loop says iterate this indented chunk of code a certain
        ### number of times. the 'count = 0' and 'count < 9' says to do it 9 times,
        ### which corresponds to the 10th pixel at the end because the starting 
        ### pixel called 'contactcorners' is excluded. Pixels are sequentially
        ### removed from so that you can retrieve it's index
        ### and value:
        for (i,j,k), value in np.ndenumerate(noncontactborders):
            if value != 0:
                if ((np.count_nonzero(noncontactborders[sub(i,1):i+2, sub(j,1):j+2, :]) == 2)):
                    pathdestroyer[i,j,k] = noncontactborders[i,j,k]
                    
        noncontactborders -= pathdestroyer
        pathfinder += pathdestroyer
        pathdestroyer = np.zeros(noncontactborders.shape, np.dtype(np.uint32))
        count += 1
    
    
    ########    Create vectors from the cell border sections    ########
    
### Preprocess the cell borders that you will use to measure angles so that you
### can uniquely identify each angle:
    angle1 = np.zeros(noncontactborders.shape, np.dtype(np.uint32))
    angle2 = np.zeros(noncontactborders.shape, np.dtype(np.uint32))
    
    angles_cell_labeled = pathfinder + contactcorners
    angles_angle_labeled, n_angles = mh.label(angles_cell_labeled, Bc=Bc3D)
    
    for (i,j,k), value in np.ndenumerate(angles_angle_labeled):
        if value == 1:
            angle1[i,j,k] = angles_angle_labeled[i,j,k]
        elif value == 2:
            angle2[i,j,k] = angles_angle_labeled[i,j,k]
#            elif value == 3:
#                skip = 'y'
#                print "I got lost finding the angles..."
            
### Continue preprocessing so that you can uniquely identify separate vectors 
### for each angle:
    
    raw_vectors_1_2 = (angle1.astype(np.bool)) * angles_cell_labeled
    raw_vectors_3_4 = (angle2.astype(np.bool)) * angles_cell_labeled
    
    raw_vectors_1_2_startpoints = contactcorners*raw_vectors_1_2.astype(np.bool)
    raw_vectors_3_4_startpoints = contactcorners*raw_vectors_3_4.astype(np.bool)
    
    vectors_1_2_endpoints = []
    vectors_1_2_startpoints = []
    
    vectors_3_4_endpoints = []
    vectors_3_4_startpoints = []
    
### Define vector 1 and 2's endpoints:
    for (i,j,k), value in np.ndenumerate(raw_vectors_1_2):
        if value != 0:
            if ((np.count_nonzero(raw_vectors_1_2[sub(i,1):i+2, sub(j,1):j+2, :]) == 2)):
                vectors_1_2_endpoints.append([[i,j,k], value])
                
### Define vector 1 and 2's startpoints (ie. the corresponding contactcorners):
    for (i,j,k), value in np.ndenumerate(raw_vectors_1_2_startpoints):
        if value != 0:
            vectors_1_2_startpoints.append([[i,j,k], value])
    
### Do the same for vectors 3 and 4:            
    for (i,j,k), value in np.ndenumerate(raw_vectors_3_4):
        if value != 0:
            if ((np.count_nonzero(raw_vectors_3_4[sub(i,1):i+2, sub(j,1):j+2, :]) == 2)):
                vectors_3_4_endpoints.append([[i,j,k], value])
    
    for (i,j,k), value in np.ndenumerate(raw_vectors_3_4_startpoints):
        if value != 0:
            vectors_3_4_startpoints.append([[i,j,k], value])


    vector1 = np.array([float('nan'),float('nan')]) ### red cell angle 1
    vector2 = np.array([float('nan'),float('nan')]) ### green cell angle 1
    vector3 = np.array([float('nan'),float('nan')]) ### red cell angle 2
    vector4 = np.array([float('nan'),float('nan')]) ### green cell angle 2
            
### Now it's time to create the separate vectors 1 and 2:
    for (y,x,z), value_end in vectors_1_2_endpoints:
        for (b,a,c), value_start in vectors_1_2_startpoints:    
            if value_end == value_start and value_end == cell1_id:
                vector1 = np.asarray([x,y]) - np.asarray([a,b])
#                    vector1[1] *= -1
            elif value_end == value_start and value_end == cell2_id:
                vector2 = np.asarray([x,y]) - np.asarray([a,b])
#                    vector2[1] *= -1
    
### ... and also vectors 3 and 4:
    for (y,x,z), value_end in vectors_3_4_endpoints:
        for (b,a,c), value_start in vectors_3_4_startpoints:    
            if value_end == value_start and value_end == cell1_id:
                vector3 = np.asarray([x,y]) - np.asarray([a,b])
#                    vector3[1] *= -1
            elif value_end == value_start and value_end == cell2_id:
                vector4 = np.asarray([x,y]) - np.asarray([a,b])
#                    vector4[1] *= -1

### Determine the x,y coordinates of the vector start locations for later usage
### (see section below on plotting angles in a figure or search up ax.quiver):
    for (b,a,c), value_start in vectors_1_2_startpoints:
        if value_start == cell1_id:
            vector1_start = np.asarray([a,b])
    for (b,a,c), value_start in vectors_1_2_startpoints:
        if value_start == cell2_id:
            vector2_start = np.asarray([a,b])
    for (b,a,c), value_start in vectors_3_4_startpoints:
        if value_start == cell1_id:
            vector3_start = np.asarray([a,b])
    for (b,a,c), value_start in vectors_3_4_startpoints:
        if value_start == cell2_id:
            vector4_start = np.asarray([a,b])
    
### Determine the angle by using the dot-product of the two vectors:
    radians_angle1 = math.acos( np.round( ((np.dot(vector1,vector2)) / (((vector1[0]**2 + vector1[1]**2)**0.5)*((vector2[0]**2 + vector2[1]**2)**0.5)) ), decimals=10) )
    degree_angle1 = math.degrees(radians_angle1)
    ang1_annot = ((vector1 + vector2)/2.0)        
    ang1_annot_mag = np.sqrt(ang1_annot.dot(ang1_annot))
    if ang1_annot_mag == 0:
        ang1_annot_mag = 10**(-12)

    radians_angle2 = math.acos( np.round( ((np.dot(vector3,vector4)) / (((vector3[0]**2 + vector3[1]**2)**0.5)*((vector4[0]**2 + vector4[1]**2)**0.5)) ), decimals=10) )
    degree_angle2 = math.degrees(radians_angle2)
    ang2_annot = ((vector3 + vector4)/2.0)
    ang2_annot_mag = np.sqrt(ang2_annot.dot(ang2_annot))
    if ang2_annot_mag == 0:
        ang2_annot_mag = 10**(-12)

### When cells have spread really far on each other it is possible 
### the contact angle exceeds 180 degrees, therefore we need to 
### check if this is the case or not. If the angle is greater
### than 180 degrees then a line connecting the ends of each 
### vector that makes up either angle1 or angle2 should
### intersect a line connecting the start of angle1 to the
### start of angle2:
    # The line connecting the vector start points (with sub-pixel res):
    vec_start12 = np.asarray([coord[0] for coord in vectors_1_2_startpoints])
    vec_start12_subpx = np.mean(vec_start12, axis=0)
    vec_start34 = np.asarray([coord[0] for coord in vectors_3_4_startpoints])
    vec_start34_subpx = np.mean(vec_start34, axis=0)
    # The points defining a line approximating the contact:
    cont_pts = np.row_stack((vec_start12_subpx, vec_start34_subpx))

    # The line connecting the end points of the vectors making up angle1:
    vec_end12 = np.asarray([coord[0] for coord in vectors_1_2_endpoints])
    # The line connecting the end points of the vectors making up angle2:
    vec_end34 = np.asarray([coord[0] for coord in vectors_3_4_endpoints])
    
    # Turn the rc coordinates into arrays of x and y coordinates:
    arr_x1 = cont_pts[:,1]
    arr_y1 = cont_pts[:,0]

    arr_x2 = vec_end12[:,1]
    arr_y2 = vec_end12[:,0]
    
    arr_x3 = vec_end34[:,1]
    arr_y3 = vec_end34[:,0]

    ang1_cont_intersect = get_intersection(arr_x1, arr_y1, arr_x2, arr_y2)
    ang2_cont_intersect = get_intersection(arr_x1, arr_y1, arr_x3, arr_y3)

    # Convert the intersection points from x,y to r,c coordinates:
    ang1_cont_intersect = ang1_cont_intersect[::-1]
    ang2_cont_intersect = ang2_cont_intersect[::-1]
    
    # Make array of 4 points that lie on the line cont_pts so that we can test
    # which points are the most extreme of the point set:
    pt_set = np.row_stack((ang1_cont_intersect,
                           ang2_cont_intersect,
                           vec_start12_subpx[:2],
                           vec_start34_subpx[:2]))
   

    # If the contact angle has not yet passed beyond 180 degrees then the
    # ang1_cont_intersect and ang2_cont_intersect should be the extrema of the
    # point set (ang1_cont_intersect, ang2_cont_intersect, vec_start12_subpx, vec_start34_subpx)
    # that lies along the line cont_pts:
    max_r, max_c = np.max(pt_set, axis=0)
    min_r, min_c = np.min(pt_set, axis=0)
    # Since this is a line, we only need to check one axis for extrema 
    # properties. In the event that the angletips-contact intersection is 
    # between the vector start pixels, then the angle has passed 180 degrees
    # and we can find the correct angle by subtracting from 360 degrees:
    if ang1_cont_intersect[0] in [max_r, min_r]:
        pass
    else:
        degree_angle1 = 360 - degree_angle1
    
    if ang2_cont_intersect[0] in [max_r, min_r]:
        pass
    else:
        degree_angle2 = 360 - degree_angle2
    
    
    average_angle = (degree_angle1 + degree_angle2)/2.0

    return degree_angle1, degree_angle2, average_angle, vector1_start, vector1, vector2_start, vector2, vector3_start, vector3, vector4_start, vector4, pathfinder, angles_angle_labeled
###
    


def get_intersection(arr_x1, arr_y1, arr_x2, arr_y2):
    """arr_x and arr_y are each arrays of 2 points (since a line can be defined
    by 2 points). Therefore (arr_x1, arr_y1) make up line1 and (arr_x2, arr_y2)
    make up line2."""
    
    ### Solve for the intersection of line1 and line2. All of the try, except
    ### and if statements are to handle if the line is vertical and therefore
    ### has an infinite slope:
    with np.errstate(divide='raise', invalid='raise'):
        try:
            slope1 = np.subtract(*arr_y1) / np.subtract(*arr_x1)
            y_interc1 = arr_y1[0] - slope1 * arr_x1[0]
        except(FloatingPointError):
            slope1 = np.inf
            y_interc1 = np.inf

    with np.errstate(divide='raise', invalid='raise'):
        try:
            slope2 = np.subtract(*arr_y2) / np.subtract(*arr_x2)
            y_interc2 = arr_y2[0] - slope2 * arr_x2[0]
        except(FloatingPointError):
            slope2 = np.inf
            y_interc2 = np.inf

    if np.isfinite(slope1) and np.isfinite(slope2):
        intersect_x = (y_interc2 - y_interc1) / (slope1 - slope2)
        intersect_y = slope1 * intersect_x + y_interc1
    elif slope1 == slope2: #then these lines are parallel to one another
        intersect_x = np.nan
        intersect_y = np.nan
    else:
        if np.isinf(slope1):
            intersect_x = arr_x1[0]
            intersect_y = slope2 * intersect_x + y_interc2
        elif np.isinf(slope2):
            intersect_x = arr_x2[0]
            intersect_y = slope1 * intersect_x + y_interc1
    
    intersection_pt = np.array([intersect_x, intersect_y], dtype=float)
    
    return intersection_pt
    

#def perform_linescan(contactsurface, perpendicular2contact_vector, linescan_length, RGB_array_of_cells, cell1_id, cell2_id):

    ### Extend the vector that is perpendicular to the contact line by the user-
    ### specified length "linescan_length":        
#    linescan = linescan_length / np.sqrt(np.dot(perpendicular2contact_vector, perpendicular2contact_vector)) * perpendicular2contact_vector

    ### Convert the linescan indexing in to image format (row,col) (ie. (y,x)):
#    imgform_linescan = np.zeros((2))
#    imgform_linescan[0] = linescan[1]
#    imgform_linescan[1] = linescan[0]

    ### Create a list of contactsurface pixel x coords and y coords:
#    cell1_contact_indicesyx = [] #red = cell1 ; grn = cell2
#    cell2_contact_indicesyx = []
    
#    red_linescan_values = []
#    grn_linescan_values = []
#    blu_linescan_values = []

    
    
    ### Return all the indices of pixels of each cell border that is considered
    ### to be in contact with the other cell:
#    for i,v in np.ndenumerate(contactsurface):
#        if v == cell1_id:
#            cell1_contact_indicesyx.append(i)
#        elif v == cell2_id:
#            cell2_contact_indicesyx.append(i)
#    for i,v in enumerate(cell1_contact_indicesyx):
#        cell1_contact_indicesyx[i] = list(v)
#    for i,v in enumerate(cell2_contact_indicesyx):
#        cell2_contact_indicesyx[i] = list(v)

    # This next part does a linescan of the desired length (linescan_length) on
    # each pixel in the red contactsurface:
    ### For each pixel in the red cells contactsurface...
#    for i,v in enumerate(cell1_contact_indicesyx):        

        ### Define linescan start and end locations based on the desired length
        ### of the linescan...
#        linescan_start = v[:2] - (imgform_linescan/2.0)
#        linescan_end = v[:2] + (imgform_linescan/2.0)
        ### And create a list of equally spaced point between the start and end:
#        x,y = np.linspace(linescan_start[1], linescan_end[1], num=linescan_length, endpoint=False), np.linspace(linescan_start[0], linescan_end[0], num=linescan_length, endpoint=False)
        ### And then rasterize those points so that they are equivalent to indices
        ### in the image:
#        for j,k in enumerate(x):
#            x[j] = round(k)
#        for j,k in enumerate(y):
#            y[j] = round(k)
        ### Then use that rasterized line to do a linescan on the green and red
        ### channels for that pixel...
        #red_linescan = RGB_array_of_cells[y.astype(np.int),x.astype(np.int), 0]
        #grn_linescan = RGB_array_of_cells[y.astype(np.int),x.astype(np.int), 1]
        #blu_linescan = RGB_array_of_cells[y.astype(np.int),x.astype(np.int), 2]
        
        ### And also append this same information to a list containing all the
        ### other linescans for the other pixels in this particular cellpair
        ### at this particular time:
#        red_linescan_values.append(RGB_array_of_cells[y.astype(np.int),x.astype(np.int), 0])
#        grn_linescan_values.append(RGB_array_of_cells[y.astype(np.int),x.astype(np.int), 1])
#        blu_linescan_values.append(RGB_array_of_cells[y.astype(np.int),x.astype(np.int), 2])
        
        ### Then turn the linescans into a polynomial piecewise function by
        ### performing interpolation on the red_linescan and grn_linescan:
        #red_polyfunc = interpolate.interp1d(np.arange(linescan_length), red_linescan, fill_value="extrapolate")
        #grn_polyfunc = interpolate.interp1d(np.arange(linescan_length), grn_linescan, fill_value="extrapolate")
        #blu_polyfunc = interpolate.interp1d(np.arange(linescan_length), blu_linescan, fill_value="extrapolate")
        
        # The linescans can be plotted as follows:
        #plt.plot(red_linescan, c='r')
        #plt.plot(grn_linescan, c='g')


        
###############################################################################
    
    ### Create an array containing the many linescans from the red channel all 
    ### aligned:
#    red_gap_scan = np.reshape(red_linescan_values, (len(cell1_contact_indicesyx),linescan_length))
#    red_gap_scan = red_gap_scan.T                

    ### Create an array containing the many linescans from the green channel all 
    ### aligned:
#    grn_gap_scan = np.reshape(grn_linescan_values, (len(cell1_contact_indicesyx),linescan_length)) # Is that len(cell1_...) actually supposed to be len(cell2_...)?
#    grn_gap_scan = grn_gap_scan.T        
    
    ### Create an array containing the many linescans from the blue channel all 
    ### aligned:
#    blu_gap_scan = np.reshape(blu_linescan_values, (len(cell1_contact_indicesyx),linescan_length)) # Is that len(cell1_...) actually supposed to be len(cell2_...)?
#    blu_gap_scan = blu_gap_scan.T       
    
    ### Take the average intensity at each position along the many linescans 
    ### for each channel:
    #red_gap_scanmean = np.mean(red_gap_scan, 1)
    #grn_gap_scanmean = np.mean(grn_gap_scan, 1)
    #blu_gap_scanmean = np.mean(blu_gap_scan, 1)
    
    ### Make a list containing the position along the linescan of the fluorescence
    ### value (gap_scanx), and the fluorescence value at that point (gap_scany):
#    red_gap_scanx = []
#    red_gap_scany = []
#    for (i,j), l in np.ndenumerate(red_gap_scan):
#        red_gap_scanx.append(i)
#        red_gap_scany.append(l)
    
#    grn_gap_scanx = []
#    grn_gap_scany = []
#    for (i,j), l in np.ndenumerate(grn_gap_scan):
#        grn_gap_scanx.append(i)
#        grn_gap_scany.append(l)
        
#    blu_gap_scanx = []
#    blu_gap_scany = []
#    for (i,j), l in np.ndenumerate(blu_gap_scan):
#        blu_gap_scanx.append(i)
#        blu_gap_scany.append(l)

    
    ### Originally I tried to fit a Gaussian function to the data but seeing as
    ### the Gaussian appears to be a poor fit, find the intersection of the 2
    ### intensity curves assuming that they are described well enough by a 
    ### piecewise polynomial:
    #red_polyfunc = interpolate.interp1d(np.arange(linescan_length), red_gap_scanmean, fill_value="extrapolate")
    #grn_polyfunc = interpolate.interp1d(np.arange(linescan_length), grn_gap_scanmean, fill_value="extrapolate")
    #blu_polyfunc = interpolate.interp1d(np.arange(linescan_length), blu_gap_scanmean, fill_value="extrapolate")
    
#    return  red_gap_scanx, red_gap_scany, grn_gap_scanx, grn_gap_scany, blu_gap_scanx, blu_gap_scany
###
    


### Since I have so many lists I'm going to define a function to update them
### all for the sake of simplicity:
def listupdate():
    #master_red_linescan_mean.append(red_gap_scanmean)
    #master_grn_linescan_mean.append(grn_gap_scanmean)     
    #master_red_linescanx.append(red_gap_scanx)
    #master_red_linescany.append(red_gap_scany)
    #master_grn_linescanx.append(grn_gap_scanx)
    #master_grn_linescany.append(grn_gap_scany)
    time_list.append('')
    cell_pair_num_list.append((filenumber+1))
    angle_1_list.append((degree_angle1))
    angle_2_list.append((degree_angle2))
    average_angle_list.append((average_angle))
    contactsurface_len_red_list.append(contactsurface_len1)
    contactsurface_len_blu_list.append(contactsurface_len2)
    contactsurface_list.append(contactsurface_len)
    contact_corner1_list.append((vector1_start,vector3_start))
    contact_corner2_list.append((vector2_start,vector4_start))
    corner_corner_length1_list.append(((vector1_start - vector3_start)[0]**2 + (vector1_start - vector3_start)[1]**2)**0.5)
    corner_corner_length2_list.append(((vector2_start - vector4_start)[0]**2 + (vector2_start - vector4_start)[1]**2)**0.5)
    noncontactsurface1_list.append((noncontact_perimeter_cell1))
    noncontactsurface2_list.append((noncontact_perimeter_cell2))
    
    redchan_fluor_at_contact_list_inner.append(redchan_fluor_at_contact_inner.tolist())
    grnchan_fluor_at_contact_list_inner.append(grnchan_fluor_at_contact_inner.tolist())
    bluchan_fluor_at_contact_list_inner.append(bluchan_fluor_at_contact_inner.tolist())
    redchan_fluor_at_contact_list_outer.append(redchan_fluor_at_contact_outer.tolist())
    grnchan_fluor_at_contact_list_outer.append(grnchan_fluor_at_contact_outer.tolist())
    bluchan_fluor_at_contact_list_outer.append(bluchan_fluor_at_contact_outer.tolist())

    redchan_fluor_outside_contact_list_inner.append(redchan_fluor_outside_contact_inner.tolist())
    grnchan_fluor_outside_contact_list_inner.append(grnchan_fluor_outside_contact_inner.tolist())
    bluchan_fluor_outside_contact_list_inner.append(bluchan_fluor_outside_contact_inner.tolist())
    redchan_fluor_outside_contact_list_outer.append(redchan_fluor_outside_contact_outer.tolist())
    grnchan_fluor_outside_contact_list_outer.append(grnchan_fluor_outside_contact_outer.tolist())
    bluchan_fluor_outside_contact_list_outer.append(bluchan_fluor_outside_contact_outer.tolist())
    
    redchan_fluor_at_contact_list.append(redchan_fluor_at_contact.tolist())
    grnchan_fluor_at_contact_list.append(grnchan_fluor_at_contact.tolist())
    bluchan_fluor_at_contact_list.append(bluchan_fluor_at_contact.tolist())
    redchan_fluor_outside_contact_list.append(redchan_fluor_outside_contact.tolist())
    grnchan_fluor_outside_contact_list.append(grnchan_fluor_outside_contact.tolist())
    bluchan_fluor_outside_contact_list.append(bluchan_fluor_outside_contact.tolist())
    
    redchan_fluor_cytoplasmic_list.append(redchan_fluor_cytoplasmic.tolist())
    grnchan_fluor_cytoplasmic_list.append(grnchan_fluor_cytoplasmic.tolist())
    bluchan_fluor_cytoplasmic_list.append(bluchan_fluor_cytoplasmic.tolist())
    
    
    filename_list.append((os.path.basename(filename)))
    #red_gauss_list.append(gaussian_filter_value_red)
    #grn_gauss_list.append(gaussian_filter_value_grn)
    Nth_pixel_list.append(Nth_pixel)
    disk_radius_list.append(disk_radius)
    area_cell_1_list.append(area_cell_1)
    area_cell_2_list.append(area_cell_2)
    cell_pair_area_list.append((area_cell_1 + area_cell_2))
    cell1_perimeter_list.append(perimeter_cell1)
    cell2_perimeter_list.append(perimeter_cell2)
    #struct_elem_list.append(Bc3D.shape)
#    linescan_length_list.append(linescan_length)
    
    redcellborders_coords.append(list(zip(*np.where(cleanborders==red_id))))
    redcontactsurface_coords.append(list(zip(*np.where(contactsurface==red_id))))
    rednoncontact_coords.append(list(zip(*np.where((noncontactborders+pathfinder)==red_id))))
    
    blucellborders_coords.append(list(zip(*np.where(cleanborders==blu_id))))
    blucontactsurface_coords.append(list(zip(*np.where(contactsurface==blu_id))))
    blunoncontact_coords.append(list(zip(*np.where((noncontactborders+pathfinder)==blu_id))))
    
    redcontactvector1_start.append(list(vector1_start))
    blucontactvector1_start.append(list(vector2_start))
    redcontactvector2_start.append(list(vector3_start))
    blucontactvector2_start.append(list(vector4_start))

    redcontactvector1.append(list(vector1))
    blucontactvector1.append(list(vector2))
    redcontactvector2.append(list(vector3))
    blucontactvector2.append(list(vector4))
    try:
        red_maxima_coords.append( np.where(spots_red_of_interest) )
    except IndexError:
        red_maxima_coords.append( (np.array(float('nan')), np.array(float('nan'))) )
    try:
        blu_maxima_coords.append( np.where(spots_blu_of_interest) )
    except IndexError:
        blu_maxima_coords.append( (np.array(float('nan')), np.array(float('nan'))) )
    
    contact_vectors_list.append(list(contact_vector))
    perpendicular2contact_vectors_list.append(list(perpendicular2contact_vector))
    
    red_halfangle1_list.append(red_halfangle1)
    red_halfangle2_list.append(red_halfangle2)
    blu_halfangle1_list.append(blu_halfangle1)
    blu_halfangle2_list.append(blu_halfangle2)
    numbered_border_indices_red_list.append(numbered_border_indices_red)
    numbered_border_indices_blu_list.append(numbered_border_indices_blu)
    numbered_noncontactborder_indices_red_list.append(numbered_noncontactborder_indices_red)
    numbered_noncontactborder_indices_blu_list.append(numbered_noncontactborder_indices_blu)

    
    mask_list.append(skip=='y')
    
    curveborders1_protoarray.append(curveborders[:,:,channel1])
    curveborders2_protoarray.append(curveborders[:,:,channel2])
    noncontactborders1_protoarray.append(noncontactborders[:,:,channel1])
    noncontactborders2_protoarray.append(noncontactborders[:,:,channel2])
    contactsurface1_protoarray.append(contactsurface[:,:,channel1])
    contactsurface2_protoarray.append(contactsurface[:,:,channel2])
    pathfinder1_protoarray.append(pathfinder[:,:,channel1])
    pathfinder2_protoarray.append(pathfinder[:,:,channel2])
    
    angles_angle_labeled_red_protoarray.append(angles_angle_labeled[:,:,channel1])
    angles_angle_labeled_blu_protoarray.append(angles_angle_labeled[:,:,channel2])
        
    bin_curvature_details_red_list.append(bin_curvature_details_red)
    bin_curvature_details_blu_list.append(bin_curvature_details_blu)
    
    whole_cell_curvature_details_red_list.append(whole_curvature_details_red)
    whole_cell_curvature_details_blu_list.append(whole_curvature_details_blu)

    bin_indices_red_list.append(bin_indices_red)
    bin_indices_blu_list.append(bin_indices_blu)
    
    bin_start_red_list.append(bin_start_red_indices)
    bin_stop_red_list.append(bin_stop_red_indices)
    
    bin_start_blu_list.append(bin_start_blu_indices)
    bin_stop_blu_list.append(bin_stop_blu_indices)
    
    cells_touching_list.append(cells_touching)
    
    script_version_list.append(script_version)
    
    msg_list.append(messages)
    
    gc.collect()
    end_time = time.perf_counter()
    print("This image took %.3f seconds to process." %(end_time - start_time))
    processing_time_list.append(end_time - start_time)

    return
###
    


### Function that finds which cell maxima is closest to the previously found
### one. Works on 2D image only. Used to find which of multiple detected cells 
### in a channel are the closest to the previously found one (ie. it is
### selecting the cell that is closest to the last one found as the correct one.)
def nearest_cell_maxima(point_coords_per_timepoint_list, np_where_of_points_to_be_compared):
    """Returns empty np.where() array if no valid previous images are available
    to use."""
    # Return indices from red_ and grn_maxima_coords where all the cell_maxima 
    # row coords for that channel are NOT float('nan'):
#    valid_maxima_red = np.logical_not([np.isnan(x).all() for x in list(zip(*red_maxima_coords))[0]])
#    valid_maxima_grn = np.logical_not([np.isnan(x).all() for x in list(zip(*grn_maxima_coords))[0]])    
#    valid_maxima = np.logical_and(valid_maxima_red, valid_maxima_grn)

    valid_maxima = np.logical_not([np.isnan(x).all() for x in list(zip(*point_coords_per_timepoint_list))[0]])
    # In case there are somehow too many points in point coords, they need to 
    # be masked:
    valid_maxima *= [len(x[0]) <2 for x in point_coords_per_timepoint_list]
    loc_valid_maxima = np.where(valid_maxima)[0]
    
    # Use the second-most recent valid_maxima as your point of reference and
    # see which of the first-most recent valid maxima is the closest:
    # Define current and previous valid maxima coords:
    valid_max_coords = np.asarray(point_coords_per_timepoint_list)[loc_valid_maxima].astype(float)
    # Use a cumulative ditribution function as a weight:
    x = np.array(range(0, len(point_coords_per_timepoint_list)))
    # 0.5 in line below says that most of the weight should be given to the 
    # most recent half of the cells analyzed:
    u = len(point_coords_per_timepoint_list) * 0.5
    s = 2
    weight_function = 1. / ( 1. + np.exp(-1. * (x - u) / s) )
    weight_function = weight_function[loc_valid_maxima]
    if weight_function.any():
        general_cell_max_loc = np.average(valid_max_coords, axis=0, weights=weight_function)
    else:
        general_cell_max_loc = (np.array([float('nan')]), np.array([float('nan')]))
#    general_cell_max_loc = valid_max_coords.mean(axis=0, dtype=int)
#    last_valid_max_coords = point_coords_per_timepoint_list[loc_valid_maxima[-1]]
    current_max_coords = np_where_of_points_to_be_compared
    
    # Compare last_valid_max_coords against all current_max_coords:
#    distance_squared = (current_max_coords[0] - last_valid_max_coords[0])**2 + (current_max_coords[1] - last_valid_max_coords[1])**2
    distance_squared = (current_max_coords[0] - general_cell_max_loc[0])**2 + (current_max_coords[1] - general_cell_max_loc[1])**2
    probably_correct_cell_center = (current_max_coords[0][np.where(distance_squared == np.min(distance_squared))], \
            current_max_coords[1][np.where(distance_squared == np.min(distance_squared))])
    return probably_correct_cell_center
###
    


# Define an "X" structuring element:
diag_struct_elem = np.array([[0,1,0], [1,0,1], [0,1,0]])
diag_struct_elem = np.reshape( diag_struct_elem, (3,3,1))

# Define a "+" structuring element:
cross_struct_elem = np.array([[1,0,1], [0,0,0], [1,0,1]])
cross_struct_elem = np.reshape( cross_struct_elem, (3,3,1))



###############################################################################
# Start the script:

### Assign the path to your folder containing your cropped cell pair images.
while True:
    try:
        path = input('Please select the path to your cropped cell pair images.\n>')
        os.chdir(path)
#        ext = raw_input('What types of files will you be working with?\n(eg. ".jpg", ".tif")\n>')
        ext = ".tif"
        # Make sure all of the channels have TIF images in them:
        if (len(glob.glob(path+'/red/*'+ext)) == 0) or \
        (len(glob.glob(path+'/green/*'+ext)) == 0) or \
        (len(glob.glob(path+'/blue/*'+ext)) == 0) or \
        (len(glob.glob(path+'/trans_light/*'+ext)) == 0):
#            print 'This folder contains no files of the specified file type!'
            print('Not all channel folders contain .tif files!')
            continue
        # Also make sure that all of the channels have the SAME NUMBER of TIF
        # files in them:
        elif (len(np.unique( [len(glob.glob(path+'/red/*'+ext)), \
                              len(glob.glob(path+'/green/*'+ext)), \
                              len(glob.glob(path+'/blue/*'+ext)), \
                              len(glob.glob(path+'/trans_light/*'+ext))] )) != 1):
            print('Not all channel folders have the same number of .tif files!')
            continue
        else:
            for filename in glob.glob(path+'/*'+ext):
                Image.open(filename)
            break
    except(WindowsError, OSError) as err:
        print(err, "\nI couldn't find the directory or file, please try typing it in again and check your spelling.\n")
        continue
    except (RuntimeError, ValueError) as err:
        print(err, ".\nThis means that the filetype you selected can't be read by"
                   'the script. Please make sure you spelled the file extension '
                   'correctly and included the preceding dot.'
                   )
        continue


### Before beginning the image processing create a bunch of empty lists that 
### will later contain important information:

time_list = []
cell_pair_num_list = []
angle_1_list = []
angle_2_list = []
average_angle_list = []
contact_corner1_list = []
contact_corner2_list = []
corner_corner_length1_list = []
corner_corner_length2_list = []
contactsurface_list = []
noncontactsurface1_list = []
noncontactsurface2_list = []

redchan_fluor_at_contact_list_inner = []
grnchan_fluor_at_contact_list_inner = []
bluchan_fluor_at_contact_list_inner = []
redchan_fluor_at_contact_list_outer = []
grnchan_fluor_at_contact_list_outer = []
bluchan_fluor_at_contact_list_outer = []
redchan_fluor_outside_contact_list_inner = []
grnchan_fluor_outside_contact_list_inner = []
bluchan_fluor_outside_contact_list_inner = []
redchan_fluor_outside_contact_list_outer = []
grnchan_fluor_outside_contact_list_outer = []
bluchan_fluor_outside_contact_list_outer = []

redchan_fluor_at_contact_list = []
grnchan_fluor_at_contact_list = []
bluchan_fluor_at_contact_list = []
redchan_fluor_outside_contact_list = []
grnchan_fluor_outside_contact_list = []
bluchan_fluor_outside_contact_list = []

redchan_fluor_cytoplasmic_list = []
grnchan_fluor_cytoplasmic_list = []
bluchan_fluor_cytoplasmic_list = []

filename_list = []
#struct_elem_list = []
Nth_pixel_list = []
disk_radius_list = [] 
area_cell_1_list = []
area_cell_2_list = []
cell_pair_area_list = []
cell1_perimeter_list = []
cell2_perimeter_list = []
#master_red_linescan_mean = []
#master_grn_linescan_mean = []
#master_red_linescanx = []
#master_red_linescany = []     
#master_grn_linescanx = []
#master_grn_linescany = []
#linescan_length_list = []
#curve_length_list = []
processing_time_list = []
mask_list = []

redcellborders_coords = []
redcontactsurface_coords = []
rednoncontact_coords = []

blucellborders_coords = []
blucontactsurface_coords = []
blunoncontact_coords = []

redcontactvector1_start = []
blucontactvector1_start = []
redcontactvector2_start = []
blucontactvector2_start = []

redcontactvector1 = []
blucontactvector1 = []
redcontactvector2 = []
blucontactvector2 = []

red_maxima_coords = []
blu_maxima_coords = []

contact_vectors_list = []
perpendicular2contact_vectors_list = []

red_halfangle1_list = []
red_halfangle2_list = []
blu_halfangle1_list = []
blu_halfangle2_list = []

numbered_border_indices_red_list = []
numbered_border_indices_blu_list = []
numbered_noncontactborder_indices_red_list = []
numbered_noncontactborder_indices_blu_list = []

contactsurface_len_red_list = []
contactsurface_len_blu_list = []

curveborders1_protoarray = []
curveborders2_protoarray = []
noncontactborders1_protoarray = []
noncontactborders2_protoarray = []
contactsurface1_protoarray = []
contactsurface2_protoarray = []
pathfinder1_protoarray = []
pathfinder2_protoarray = []
angles_angle_labeled_red_protoarray = []
angles_angle_labeled_blu_protoarray = []


bin_curvature_details_red_list = []
bin_curvature_details_blu_list = []

whole_cell_curvature_details_red_list = []
whole_cell_curvature_details_blu_list = []

bin_indices_red_list = []
bin_indices_blu_list = []

bin_start_red_list = []
bin_stop_red_list = []

bin_start_blu_list = []
bin_stop_blu_list = []

cells_touching_list = []

script_version_list = []

msg_list = []




total_files = len(sorted(glob.glob((path)+'/red/*'+ext)))
filenames_red = sorted(glob.glob((path)+'/red/*'+ext))
filenames_grn = sorted(glob.glob((path)+'/green/*'+ext))
filenames_blu = sorted(glob.glob((path)+'/blue/*'+ext))
filenames_tl = sorted(glob.glob((path)+'/trans_light/*'+ext))
script_version = os.path.basename(__file__)


for filenumber in range(total_files):
    if filenumber == 0:
        start_time = time.perf_counter()
        
        messages = list()
        filename = filenames_tl[filenumber]
        # Open the the image from each channel. PIL will close them when no
        # longer needed:
        redchan_raw = np.array(Image.open(filenames_red[filenumber]))
        grnchan_raw = np.array(Image.open(filenames_grn[filenumber]))
        bluchan_raw = np.array(Image.open(filenames_blu[filenumber]))
        tlchan_raw = np.array(Image.open(filenames_tl[filenumber]))
        
        ### Check the signal to noise ratio for each channel:
        # If the signal in a channel has a standard deviation less than 1 in any 
        # channel, skip it because the signal to noise ratio won't be good enough
        # for proper edge detection and segmentation:
        redchan = check_for_signal(redchan_raw)
        grnchan = check_for_signal(grnchan_raw)
        bluchan = check_for_signal(bluchan_raw)
        #bluchan = np.copy(bluchan_raw)
        tlchan = check_for_signal(tlchan_raw)
        
        fluor_chans = np.dstack((redchan, grnchan, bluchan))
        channel1 = 0 #ie. red channel
        channel2 = 2 #ie. blue channel
        
        # Now get some user-defined inputs:                 
        ### Approximately how many pixels is the radius of your cell?:
        while True:
            try:
                # Use 6.67% of the hypotenuse of the sides of the image
                #disk_radius = np.hypot(*np.asarray(np.shape(redchan))) * 0.0667
                
                # Alternatively use 30 as the initial cell radius:
                disk_radius = 30
                
                #rounded to the nearest 0.5, as the radius of the disk:
                disk_radius = (np.round(disk_radius) * 2) / 2
                #disk_radius = input('How many pixels should the radius of your ' 
                #                         'structuring element be? [default/'
                #                         'recommended: %.1f (ie. 6.67%% of longest '
                #                         'side of image).]\n>' %(disk_radius)
                #                         ) or disk_radius
                                         
                disk_radius = input('About how many pixels is the radius of your ' 
                                         'cell? [default/recommended: %.1f]' 
                                         '\n>' %(disk_radius)
                                         ) or disk_radius
                
                disk_radius = float(disk_radius)
                break
                #else:
                #    print ("Sorry I didn't understand that. Please answer again "
                #           "with a decimal or integer number.")
                #    continue
                #if disk_radius == '':
                #    # If no input then just use the default disk_radius
                    
                #    break
                #else:
                #    disk_radius = float(disk_radius)
                #    break
            except (ValueError, NameError, SyntaxError, TypeError):
                print ("Sorry I didn't understand that. Please answer again "
                       "with a decimal or integer number.")
                continue

       
        ### How many pixels do you want to use to create vectors used to measure angles?
        while True:
            Nth_pixel = str(9)
            try:
                Nth_pixel = input("How many pixels would you like to use to measure"
                              " each arm of each angle? [default: %s]"
                              "\n>" %(Nth_pixel)
                              ) or Nth_pixel
                if type(eval(Nth_pixel)) == float or type(eval(Nth_pixel)) == int:
                    Nth_pixel = eval(Nth_pixel)
                    break
                else:
                    print("Sorry I didn't understand that. Please answer again.")
                    print(type(Nth_pixel))
                    continue
            except (ValueError, NameError, SyntaxError, TypeError) as e:
                print("Sorry I didn't understand that. Please answer again.")
                print(type(Nth_pixel))
                print(e)
                continue

    
###
        ### Define some default variables that are used in the script:
        std_cutoff = 100
        
        odr_circle_fit_iterations = 30
        num_bins = 8
        skip = 'n'


        gaussian_filter_value_red = 2.0
        gaussian_filter_value_grn = 2.0
        gaussian_filter_value_blu = 4.0
        gaussian_filter_value_tl = disk_radius/2.

###
        ### This is a threshold that determines how dim something in the image
        ### has to be to be considered background, it's set to 10% of the 
        ### images dtype max:
        #background_threshold = np.iinfo(tlchan.dtype).max * 0.1#0.05

###
        ### The size_filter determines how many px^2 the area of a region has
        ### to be in order to be recognized as a cell:
        size_filter = (np.pi * disk_radius**2) * 0.2 #500#20 #the 20 filter is used for the 2 circles simulation

        #upper_size_filter = 1500
        #upper_size_filter = 3000
        #upper_size_filter = 3500
        #upper_size_filter = 4000
        
        #particle_filter = 250#20 #the 20 filter is used for the 2 circle simulation

###
        ### Create a structuring element to try and find seed points for cells:
        maxima_Bc = morphology.disk(disk_radius) #15.5 = diameter of 30

        # Define a 3 dimensional structuring element for finding neighbours
        # between channels of RGB images:
        Bc3D = np.ones((3,3,5))


    ### Define the default values for the the data that will populate the table
    ### that will be output at the end:
        #red_gap_scanmean = np.asarray([float('nan')]*linescan_length)
        #grn_gap_scanmean = np.asarray([float('nan')]*linescan_length)
        #red_gap_scanx = [float('nan')]
        #red_gap_scany = [float('nan')]
        #grn_gap_scanx = [float('nan')]
        #grn_gap_scany = [float('nan')]
        degree_angle1 = float('nan')
        degree_angle2 = float('nan')
        average_angle = float('nan')
        contactsurface_len = float('nan')
        noncontactsurface_len1 = float('nan')
        noncontactsurface_len2 = float('nan')
        
        redchan_fluor_at_contact_inner = np.array([float('nan')])
        grnchan_fluor_at_contact_inner = np.array([float('nan')])
        bluchan_fluor_at_contact_inner = np.array([float('nan')])
        redchan_fluor_at_contact_outer = np.array([float('nan')])
        grnchan_fluor_at_contact_outer = np.array([float('nan')])
        bluchan_fluor_at_contact_outer = np.array([float('nan')])
        redchan_fluor_outside_contact_inner = np.array([float('nan')])
        grnchan_fluor_outside_contact_inner = np.array([float('nan')])
        bluchan_fluor_outside_contact_inner = np.array([float('nan')])
        redchan_fluor_outside_contact_outer = np.array([float('nan')])
        grnchan_fluor_outside_contact_outer = np.array([float('nan')])
        bluchan_fluor_outside_contact_outer = np.array([float('nan')])
        
        redchan_fluor_at_contact = np.array([float('nan')])
        grnchan_fluor_at_contact = np.array([float('nan')])
        bluchan_fluor_at_contact = np.array([float('nan')])
        redchan_fluor_outside_contact = np.array([float('nan')])
        grnchan_fluor_outside_contact = np.array([float('nan')])
        bluchan_fluor_outside_contact = np.array([float('nan')])
        
        redchan_fluor_cytoplasmic = np.array([float('nan')])
        grnchan_fluor_cytoplasmic = np.array([float('nan')])
        bluchan_fluor_cytoplasmic = np.array([float('nan')])
        
        area_cell_1 = float('nan')
        area_cell_2 = float('nan')
        perimeter_cell1 = float('nan')
        perimeter_cell2 = float('nan')
        noncontact_perimeter_cell1 = float('nan')
        noncontact_perimeter_cell2 = float('nan')
        cleanborders = np.zeros(fluor_chans.shape)
        contactsurface = np.zeros(fluor_chans.shape)
        noncontactborders = np.zeros(fluor_chans.shape)
        pathfinder = np.zeros(fluor_chans.shape)
        vector1_start = np.array([float('nan'), float('nan')])
        vector2_start = np.array([float('nan'), float('nan')])
        vector3_start = np.array([float('nan'), float('nan')])
        vector4_start = np.array([float('nan'), float('nan')])
        vector1 = np.array([float('nan'), float('nan')])
        vector2 = np.array([float('nan'), float('nan')])
        vector3 = np.array([float('nan'), float('nan')])
        vector4 = np.array([float('nan'), float('nan')])
        cleanspots = np.array([float('nan'), float('nan')])
        contact_vector = np.array([float('nan'), float('nan')])
        perpendicular2contact_vector = np.array([float('nan'), float('nan')])
        red_halfangle1_rad = float('nan')
        red_halfangle2_rad = float('nan')
        blu_halfangle1_rad = float('nan')
        blu_halfangle2_rad = float('nan')
        red_halfangle1 = float('nan')
        red_halfangle2 = float('nan')
        blu_halfangle1 = float('nan')
        blu_halfangle2 = float('nan')
        contactsurface_len1 = float('nan')
        contactsurface_len2 = float('nan')
        numbered_border_indices_red = [float('nan'), (float('nan'),float('nan'),float('nan'))]
        numbered_border_indices_blu = [float('nan'), (float('nan'),float('nan'),float('nan'))]
        numbered_noncontactborder_indices_red = [float('nan'), (float('nan'),float('nan'),float('nan'))]
        numbered_noncontactborder_indices_blu = [float('nan'), (float('nan'),float('nan'),float('nan'))]
        
        angles_angle_labeled = np.zeros(fluor_chans.shape)
        
        bin_curvature_details_red = [[float('nan')] * len(bin_curvature_details_headers)] * (num_bins+1)
        bin_curvature_details_blu = [[float('nan')] * len(bin_curvature_details_headers)] * (num_bins+1)
        
        whole_curvature_details_red = [[float('nan')] * len(bin_curvature_details_headers)] * 2
        whole_curvature_details_blu = [[float('nan')] * len(bin_curvature_details_headers)] * 2
        
        bin_indices_red = [[(float('nan'), float('nan'), float('nan'))]] * num_bins
        bin_indices_blu = [[(float('nan'), float('nan'), float('nan'))]] * num_bins
        
#        bin_start_red = [float('nan')] * num_bins
#        bin_stop_red = [float('nan')] * num_bins
        
#        bin_start_grn = [float('nan')] * num_bins
#        bin_stop_grn = [float('nan')] * num_bins

        bin_start_red_indices = np.empty((num_bins, 3))
        bin_start_red_indices[:] = np.nan
        
        bin_stop_red_indices = np.empty((num_bins, 3))
        bin_stop_red_indices[:] = np.nan

        bin_start_blu_indices = np.empty((num_bins, 3))
        bin_start_blu_indices[:] = np.nan

        bin_stop_blu_indices = np.empty((num_bins, 3))
        bin_stop_blu_indices[:] = np.nan

###
        ### These 2 variable assignments are unique to the first file in the
        ### folder:
        mean_closest_indice_loc_red_old = float('nan')
        mean_closest_indice_loc_blu_old = float('nan')

###
        ### Segment your cells:
        
        # Start by getting some seed points for a watershed:
        spots_all, spots_red, spots_blu, bg_max, redcell, blucell, cell_thresh, bg_thresh = find_cell_seeds(tlchan, redchan, grnchan, bluchan, maxima_Bc)



        ### The re-segmentation of the red channel to capture the membrane
        ### label should go here. But it doesn't work very well if the signal
        ### is really weak (eg. at cell-cell contact of a cellpair).
        # redseg = segmentation.watershed(redchan, spots_all, compactness=0.1)

                
        # Remove any areas that are too big for the red segmentation:
        red_cell_props = measure.regionprops(redcell)
        size_red = np.array([x.area for x in red_cell_props])
        labels_red = np.array([x.label for x in red_cell_props])
        too_big = size_red > (2 * np.pi * disk_radius**2)
        too_big_labels = labels_red[too_big]
        redcell_clean = mh.labeled.remove_regions(redcell, too_big_labels)
        
        
        
        ### Sometimes the cortical actin decouples from the plasma membrane 
        ### enough that the borders of the redcell don't line up with those of
        ### the actin in the grn channel. To try and fix this we're going to 
        ### find the red_ridges and grn_ridges, clean them up, and use them to 
        ### help us get a segmentation of the cell in the grn channel 
        ### (ie. LifeAct-GFP) such that the borders of that segmentation line 
        ### up nicely with the cortical actin:
        
        # Find the red ridges:
        red_ridges = find_ridge_peaks(redchan, axis0_peak_to_peak_dist=5, axis1_peak_to_peak_dist=5)
        #red_ridges = morphology.binary_closing(red_ridges, selem=morphology.disk(1))
        
        # Clean up the red ridges:
        okay_small_objects = morphology.remove_small_objects(np.logical_or(red_ridges, mh.borders(redcell_clean)))
        bad_small_objects = np.logical_xor(okay_small_objects, red_ridges)
        ridges_and_borders = np.logical_and(bad_small_objects, mh.borders(redcell_clean))
        red_ridges_clean = np.logical_xor(bad_small_objects, red_ridges)
        red_ridges_clean = np.logical_xor(red_ridges_clean, ridges_and_borders)
        
        # Find the green ridges:
        grn_ridges = find_ridge_peaks(grnchan, axis0_peak_to_peak_dist=5, axis1_peak_to_peak_dist=5)
        #grn_ridges = morphology.binary_closing(grn_ridges, selem=morphology.disk(1))
        
        # Clean up the green ridges
        okay_small_objects = morphology.remove_small_objects(np.logical_or(grn_ridges, mh.borders(redcell_clean)))
        bad_small_objects = np.logical_xor(okay_small_objects, grn_ridges)
        ridges_and_borders = np.logical_and(bad_small_objects, mh.borders(redcell_clean))
        grn_ridges_clean = np.logical_xor(bad_small_objects, grn_ridges)
        grn_ridges_clean = np.logical_xor(grn_ridges_clean, ridges_and_borders)
        
        # Use the grn_ridges_clean to assist segmentation of the actin:
        iterations = 100
        grnseg, growing_markers_grn, growing_area_grn, growing_labels_grn, first_indices_grn = iter_seg(filters.gaussian(segmentation.find_boundaries(redcell_clean) + grn_ridges_clean, sigma=2), spots_red, np.add.reduce((spots_blu, bg_max)), iterations=iterations, mask=~(red_ridges_clean + grn_ridges_clean), compactness=0.1)
        
        # If you add the grn_ridges_clean to the grnseg then this works pretty well for
        # restricting growth of the seeds during watershed segmentation.
        # msg = 'grnseg iterations: %s'%iterations + ','
        # msg += ' first_indices_grn: %s'%first_indices_grn + ','
        # messages.append(msg)

        
        ### Now that we have the green/LifeAct segmentation we need to clean
        ### that up as well:
        
        # Remove any areas that are too big from the green segmentation:
        grn_cell_props = measure.regionprops(grnseg)
        size_grn = np.array([x.area for x in grn_cell_props])
        labels_grn  = np.array([x.label for x in grn_cell_props])
        too_big = size_grn > (2 * np.pi * disk_radius**2)
        too_big_labels = labels_grn[too_big]
        grnseg = mh.labeled.remove_regions(grnseg, too_big_labels)
        
        # Add the grn_ridges_clean to the grnseg:
        grnseg = np.logical_or(grnseg, grn_ridges_clean)
        grnseg = morphology.remove_small_holes(grnseg)
        
        # Remove small pieces that make grnseg rough from adding grn_ridges_clean:
        grnseg = morphology.label(grnseg, connectivity=1)
        grnseg = morphology.remove_small_objects(grnseg).astype(bool)
        
        # When adding the grn_ridges_clean we end up with a morphological dilation of 
        # the grnseg borders, but we want the grn_ridges_clean to correspond to the 
        # outer edges. Therefore to rectify this take the internal region of the grnseg 
        # borders as being grnseg:
        grnseg = np.logical_xor(mh.borders(grnseg)*grnseg, grnseg)
        
        # Restore proper labeling:
        grnseg = segmentation.watershed(redchan, spots_red, mask=grnseg)
        



###

        cell_areas_rough = np.dstack((redcell_clean, np.zeros(redcell_clean.shape, dtype=int), blucell))


        ### Clean up the segmented image of any regions (ie. "cells") that are too
        ### small or that are touching the borders:
        cell_areas = remove_bordering_and_small_regions(cell_areas_rough).astype(bool) * cell_areas_rough.astype(int)
    
        # Fill in any holes in each region that might be present:
        cell_areas[:,:,0] = np.add.reduce([scp_morph.binary_fill_holes(cell_areas[:,:,0] == x) * x for x in np.unique(cell_areas[:,:,0])[1:]], dtype=int)
        cell_areas[:,:,1] = np.add.reduce([scp_morph.binary_fill_holes(cell_areas[:,:,1] == x) * x for x in np.unique(cell_areas[:,:,1])[1:]], dtype=int)
        cell_areas[:,:,2] = np.add.reduce([scp_morph.binary_fill_holes(cell_areas[:,:,2] == x) * x for x in np.unique(cell_areas[:,:,2])[1:]], dtype=int)

###

        ### Make sure that the cells meet the minimum size threshold:
        red_cell_props = measure.regionprops(cell_areas[:,:,0])
        grn_cell_props = measure.regionprops(cell_areas[:,:,1])
        blu_cell_props = measure.regionprops(cell_areas[:,:,2])
        
        size_red = np.array([x.area for x in red_cell_props])
        size_grn = np.array([x.area for x in grn_cell_props])
        size_blu = np.array([x.area for x in blu_cell_props])
        # size_red = mh.labeled.labeled_size(cell_areas[:,:,0])
        # size_grn = mh.labeled.labeled_size(cell_areas[:,:,1])
        # size_blu = mh.labeled.labeled_size(cell_areas[:,:,2])
        labels_red = np.array([x.label for x in red_cell_props])
        labels_grn = np.array([x.label for x in grn_cell_props])
        labels_blu = np.array([x.label for x in blu_cell_props])

        too_small_red = labels_red[size_red < size_filter]
        too_small_grn = labels_grn[size_grn < size_filter]
        too_small_blu = labels_blu[size_blu < size_filter]
        
        cell_areas[:,:,0] = mh.labeled.remove_regions(cell_areas[:,:,0], too_small_red)
        cell_areas[:,:,1] = mh.labeled.remove_regions(cell_areas[:,:,1], too_small_grn)
        cell_areas[:,:,2] = mh.labeled.remove_regions(cell_areas[:,:,2], too_small_blu)

        # Fill in any holes in each region that might be present (again):
        cell_areas[:,:,0] = np.add.reduce([scp_morph.binary_fill_holes(cell_areas[:,:,0] == x) * x for x in np.unique(cell_areas[:,:,0])[1:]])
        cell_areas[:,:,1] = np.add.reduce([scp_morph.binary_fill_holes(cell_areas[:,:,1] == x) * x for x in np.unique(cell_areas[:,:,1])[1:]])
        cell_areas[:,:,2] = np.add.reduce([scp_morph.binary_fill_holes(cell_areas[:,:,2] == x) * x for x in np.unique(cell_areas[:,:,2])[1:]])

        # Since this is the first image, if there is more than 1 cell in each 
        # channel of the image still, just take the largest one:
        if len(np.unique(cell_areas[:,:,0])) > 2:
            cell_areas[:,:,0] = cell_areas[:,:,0] == (labels_red[np.argmax(size_red)])
            
        if len(np.unique(cell_areas[:,:,1])) > 2:
            cell_areas[:,:,1] = cell_areas[:,:,1] == (labels_grn[np.argmax(size_grn)])
            
        if len(np.unique(cell_areas[:,:,2])) > 2:
            cell_areas[:,:,2] = cell_areas[:,:,2] == (labels_blu[np.argmax(size_blu)])
            
        # Refine spots_red to be only your red cell of interest:
        spots_red_of_interest = spots_red.astype(bool) * cell_areas[:,:,0]
        # Also do this for the blue spots:
        spots_blu_of_interest = spots_blu.astype(bool) * cell_areas[:,:,2]

###

        ### Because we do measurements of contact surface between cells and
        ### measurements of LifeAct intensity using the borders we will 
        ### eventually need to make the borders of the cell_areas and these
        ### borders will need to be made sequential in order to count the
        ### surface length or do directed line scans. In order to do this
        ### we need to thin the borders so that when considering a connectivity
        ### of 2 (or 8 depending on your nomenclature of choice) each pixel in
        ### the border is only connected to 2 other pixels. This needs to be 
        ### true the whole way around the cell areas for measuring fluorescence
        ### profiles, but when measuring contact surface or contact angle, the
        ### corners of the contact must be left untouched (even if they are
        ### connected to 3 other pixels (eg. an "L" shape)), this is because
        ### the if these corners are removed, then you can accidentally "open
        ### up" the contact and make it smaller than it really is, sometimes
        ### this even leads to the program missing cells that are definitely
        ### in contact.
     #--#
     #--#
     #--#
     #--#
        ### Having said all that, we need to make 2 copies of cleaned up 
        ### cell_areas and take their borders -- cleanborders will be used for
        ### measuring contact diameter and contact angle, and redseg_clean
        ### and grnseg_clean will be used to measure fluroescence profiles:
        
        
        
        # # And use the labels of these spots to pick out your cells of interest
        # # from the redseg and the bluseg:
        # if np.unique(spots_red_of_interest).any():
        #     redseg_clean_rough = redseg == np.unique(spots_red_of_interest)[np.nonzero(np.unique(spots_red_of_interest))]
        # else:
        #     redseg_clean_rough = np.zeros(redseg.shape)
        # if np.any(redseg_clean_rough):
        #     pass
        # else:
        #     redseg_clean_rough = np.zeros(redseg.shape)

        
        
        # Also, buff out sharp corners from redseg_clean as these will prevent 
        # clean_up_borders from working properly on both the inner and outer 
        # edges:
        # redseg_clean = morphology.binary_closing(redseg_clean_rough)
        redseg_clean = morphology.binary_closing(cell_areas[:,:,channel1])

        # Buffing out sharp corners leads to unnatural roundness, which is seen
        # as single pixels in borders, but the problems regions show up as multiple 
        # connected pixels. Therefore label the pixels added from binary_closing
        # and remove any regions with a size of 1:
        polish_pixels = np.logical_xor(cell_areas[:,:,channel1], redseg_clean)
        polish_pixels = morphology.label(polish_pixels)
        polish_px_bad_areas = [x.area == 1 for x in measure.regionprops(polish_pixels)]
        polish_px_bad_labels = np.asarray(polish_px_bad_areas) * np.asarray([x.label for x in measure.regionprops(polish_pixels)])
        polish_px_bad_indices = np.nonzero(polish_px_bad_labels)
        polish_px_bad_labels = polish_px_bad_labels[polish_px_bad_indices]
        polish_px_good = mh.labeled.remove_regions(polish_pixels, polish_px_bad_labels)
        polish_px_bad = np.logical_xor(polish_pixels, polish_px_good)
        
        redseg_clean = redseg_clean * ~polish_px_bad

        # Make sure that the redseg_clean is not a donut, as the program can't
        # analyze donuts:
        redseg_clean = scp_morph.binary_fill_holes(redseg_clean)
        
        # Also, make sure to remove any redseg_clean regions that are touching
        # the borders, as the program can't analyze those cells either:
        redseg_clean = segmentation.clear_border(redseg_clean)
        
        # And remove any redseg_clean regions that are less than the 
        # size_filter:
        redseg_clean = morphology.remove_small_objects(redseg_clean, 
                                                       min_size=size_filter)



###
        
        # Use the spots_red_of_interest to pick out your cell of interest from
        # the green channel:
        if np.unique(spots_red_of_interest).any():
            grnseg_clean_rough = grnseg == np.unique(spots_red_of_interest)[np.nonzero(np.unique(spots_red_of_interest))]
        else:
            grnseg_clean_rough = np.zeros(grnseg.shape)
        if np.any(grnseg_clean_rough):
            pass
        else:
            grnseg_clean_rough = np.zeros(grnseg.shape)


        # Also, buff out sharp corners from grnseg_clean as these will prevent 
        # clean_up_borders from working properly on both the inner and outer 
        # edges:
        grnseg_clean = morphology.binary_closing(grnseg_clean_rough)

        # Buffing out sharp corners leads to unnatural roundness, which is seen
        # as single pixels, but the problems regions show up as multiple 
        # connected pixels. Therefore label the pixels added from binary_closing
        # and remove any regions with a size of 1:
        polish_pixels = np.logical_xor(grnseg_clean_rough, grnseg_clean)
        polish_pixels = morphology.label(polish_pixels)
        polish_px_bad_areas = [x.area == 1 for x in measure.regionprops(polish_pixels)]
        polish_px_bad_labels = np.asarray(polish_px_bad_areas) * np.asarray([x.label for x in measure.regionprops(polish_pixels)])
        polish_px_bad_indices = np.nonzero(polish_px_bad_labels)
        polish_px_bad_labels = polish_px_bad_labels[polish_px_bad_indices]
        polish_px_good = mh.labeled.remove_regions(polish_pixels, polish_px_bad_labels)
        polish_px_bad = np.logical_xor(polish_pixels, polish_px_good)
        
        grnseg_clean = grnseg_clean * ~polish_px_bad

        # Make sure that the grnseg_clean is not a donut, as the program can't
        # analyze donuts:
        grnseg_clean = scp_morph.binary_fill_holes(grnseg_clean)
        
        # Also, make sure to remove any grnseg_clean regions that are touching
        # the borders, as the program can't analyze those cells either:
        grnseg_clean = segmentation.clear_border(grnseg_clean)
        
        # And remove any grnseg_clean regions that are less than the 
        # size_filter:
        grnseg_clean = morphology.remove_small_objects(grnseg_clean, 
                                                       min_size=size_filter)


###


        # Also, buff out sharp corners from redseg_clean as these will prevent 
        # clean_up_borders from working properly on both the inner and outer 
        # edges:
        # redseg_clean = morphology.binary_closing(redseg_clean_rough)
        # bluseg_clean = morphology.binary_closing(cell_areas[:,:,channel2])
        # We are going to try padding the image with a 1px-wide border made up
        # of zeros because binary_closing connects regions to the edge if the
        # cell is almost touching the borders in the non-padded image (i.e. if
        # the cell is only 1px away from the edge), we will then trim padding
        bluseg_clean = morphology.binary_closing(pad_img_edges(cell_areas[:,:,channel2], 0, 1))
        bluseg_clean = bluseg_clean[1:-1, 1:-1]


        # Buffing out sharp corners leads to unnatural roundness, which is seen
        # as single pixels in borders, but the problems regions show up as multiple 
        # connected pixels. Therefore label the pixels added from binary_closing
        # and remove any regions with a size of 1:
        polish_pixels = np.logical_xor(cell_areas[:,:,channel2], bluseg_clean)
        polish_pixels = morphology.label(polish_pixels)
        polish_px_bad_areas = [x.area == 1 for x in measure.regionprops(polish_pixels)]
        polish_px_bad_labels = np.asarray(polish_px_bad_areas) * np.asarray([x.label for x in measure.regionprops(polish_pixels)])
        polish_px_bad_indices = np.nonzero(polish_px_bad_labels)
        polish_px_bad_labels = polish_px_bad_labels[polish_px_bad_indices]
        polish_px_good = mh.labeled.remove_regions(polish_pixels, polish_px_bad_labels)
        polish_px_bad = np.logical_xor(polish_pixels, polish_px_good)
        
        bluseg_clean = bluseg_clean * ~polish_px_bad

        # Make sure that the bluseg_clean is not a donut, as the program can't
        # analyze donuts:
        bluseg_clean = scp_morph.binary_fill_holes(bluseg_clean)
        
        # Also, make sure to remove any redseg_clean regions that are touching
        # the borders, as the program can't analyze those cells either:
        bluseg_clean = segmentation.clear_border(bluseg_clean)
        
        # And remove any redseg_clean regions that are less than the 
        # size_filter:
        bluseg_clean = morphology.remove_small_objects(bluseg_clean, 
                                                       min_size=size_filter)


###############################################################################
                      ### *** Cleaned up to here *** ###
###############################################################################

###

        ### Use the refined cell segmentations to create the cell borders:
        new_area_red = cell_areas[:,:,0].astype(bool) * 1
        cell_borders_red = mh.borders(new_area_red.astype(bool)) * new_area_red
        cell_borders_red = mh.thin(cell_borders_red).astype(bool) * 1
        
        new_area_grn = cell_areas[:,:,1].astype(bool) * 2
        cell_borders_grn = mh.borders(new_area_grn.astype(bool)) * new_area_grn
        cell_borders_grn = mh.thin(cell_borders_grn).astype(bool) * 2
        
        new_area_blu = cell_areas[:,:,2].astype(bool) * 3
        cell_borders_blu = mh.borders(new_area_blu.astype(bool)) * new_area_blu
        cell_borders_blu = mh.thin(cell_borders_blu).astype(bool) * 3
        
        curveborders = np.dstack((cell_borders_red, cell_borders_grn, cell_borders_blu))
###
        #plt.imshow(mh.overlay(tlchan, red=cell_borders_red, blue=cell_borders_blu))



        ### Now that we have the borders of the cells, it's time to clean up 
        ### the borders so the algorithm used to count pixels and get 
        ### sequential info from the borders doesn't get lost:
        cleanborders = clean_up_borders(curveborders)
        
        ### Also use these cleaned up borders to define the cleaned cell areas:
        cleanareas = np.zeros(cleanborders.shape)
        cleanareas[:,:,0] = np.add.reduce([scp_morph.binary_fill_holes(cleanborders[:,:,0] == x) * x for x in np.unique(cleanborders[:,:,0])[1:]])
        cleanareas[:,:,1] = np.add.reduce([scp_morph.binary_fill_holes(cleanborders[:,:,1] == x) * x for x in np.unique(cleanborders[:,:,1])[1:]])
        cleanareas[:,:,2] = np.add.reduce([scp_morph.binary_fill_holes(cleanborders[:,:,2] == x) * x for x in np.unique(cleanborders[:,:,2])[1:]])

        ### Create an overlay of the original image and the borders for
        ### checking during coding:
        cleanoverlay = mh.overlay(tlchan, red=cleanborders[:,:,0], green=cleanborders[:,:,1], blue=cleanborders[:,:,2])
        
        cell_spots = spots_red_of_interest + spots_blu_of_interest
        cell_max_rcd = ([np.where(cell_spots.astype(bool))[0], np.where(cell_spots.astype(bool))[1]], [np.where(cell_spots.astype(bool))[0], np.where(cell_spots.astype(bool))[1]])
        bg_max_rcd = ([np.where(bg_max)[0], np.where(bg_max)[1]], [np.where(bg_max)[0], np.where(bg_max)[1]])



        ### Plot the cells and any parameters so far:
        fig = plt.figure(num=str(filenumber+1)+'_'+os.path.basename(filename), figsize=(16., 10.))
        plt.suptitle(str(filenumber+1)+' - '+os.path.basename(filename), verticalalignment='center', x=0.5, y=0.5, fontsize=14)
        ax1 = fig.add_subplot(241)
        ax1.yaxis.set_visible(False)
        ax1.xaxis.set_visible(False)
        ax1.imshow(tlchan, cmap='gray')
        
        ax2 = fig.add_subplot(242)
        ax2.yaxis.set_visible(False)
        ax2.xaxis.set_visible(False)
        #figcellborderred = ax2.imshow(mh.overlay(gray = mh.stretch(tlchan), red = cleanborders[:,:,0]))
        #figcellred = ax2.imshow(mh.overlay(gray = mh.stretch(redchan), red = cleanborders[:,:,0]))
        #fig_noncont_redcell = ax2.imshow(mh.overlay(mh.stretch(redchan), red=cleanborders[:,:,0]))
        fig_noncont_redcell = ax2.imshow(mh.overlay(mh.stretch(redchan), red=mh.borders(redseg_clean)))
        #fig_noncont_redcell = ax2.imshow(mh.stretch(redchan), cmap='gray')
        ax2.axes.autoscale(False)
        
        ax3 = fig.add_subplot(243)
        ax3.yaxis.set_visible(False)
        ax3.xaxis.set_visible(False)
        #figcellborderblu = ax3.imshow(mh.overlay(gray = mh.stretch(tlchan), blue = cleanborders[:,:,2]))
        #fig_noncont_grncell = ax3.imshow(mh.overlay(mh.stretch(grnchan), green=cleanborders[:,:,0]))
        fig_noncont_grncell = ax3.imshow(mh.overlay(mh.stretch(grnchan), green=mh.borders(grnseg_clean)))
        #fig_noncont_grncell = ax3.imshow(mh.stretch(grnchan), cmap='gray')
        ax3.axes.autoscale(False)
        
        ax4 = fig.add_subplot(246)
        ax4.yaxis.set_visible(False)
        ax4.xaxis.set_visible(False)
        #ax4.imshow(tlchan, cmap='gray')
        fig_cont_redcell = ax4.imshow(mh.stretch(redchan), cmap='gray')
        ax4.axes.autoscale(False)
        
        ax5 = fig.add_subplot(247)
        ax5.yaxis.set_visible(False)
        ax5.xaxis.set_visible(False)
        #ax5.imshow(tlchan, cmap='gray')
        fig_cont_grncell = ax5.imshow(mh.stretch(grnchan), cmap='gray')
        ax5.axes.autoscale(False)

        ax6 = fig.add_subplot(245)
        ax6.yaxis.set_visible(False)
        ax6.xaxis.set_visible(False)
        ax6.imshow(mh.overlay(gray = mh.stretch(tlchan), red = cleanborders[:,:,0], blue=cleanborders[:,:,2]))
        ax6.axes.autoscale(False)
        ax6.scatter(cell_max_rcd[0][1], cell_max_rcd[0][0], color='m', marker='o')
        ax6.scatter(cell_max_rcd[1][1], cell_max_rcd[1][0], color='m', marker='o')
        ax6.scatter(bg_max_rcd[0][1], bg_max_rcd[0][0], color='m', marker='x')
        ax6.scatter(bg_max_rcd[1][1], bg_max_rcd[1][0], color='m', marker='x')

        ax7 = fig.add_subplot(244)
        ax7.yaxis.set_visible(False)
        ax7.xaxis.set_visible(False)
        ax_noncont_blucell = ax7.imshow(mh.overlay(mh.stretch(bluchan), blue=mh.borders(bluseg_clean)))
        ax7.axes.autoscale(False)
        
        ax8 = fig.add_subplot(248)
        ax8.yaxis.set_visible(False)
        ax8.xaxis.set_visible(False)
        fig_cont_blucell = ax8.imshow(mh.stretch(bluchan), cmap='gray')
        ax8.axes.autoscale(False)


        plt.tight_layout()
        plt.draw()



        ### Tell me the filenumber and filename of what you're working on:
        print('\n',(filenumber+1), filename)

        ### Skip the file if there is only 1 cell in the picture (sometimes the
        ### microscope skips a channel):
#        if len(mh.labeled.labeled_size(cleanareas)) < 3:
        if len(np.unique(cleanborders)) < 3:
            msg = "There aren't 2 cells in this image."
            print(msg)
            messages.append(msg)
            # skip = 'y'
            
        if np.any(redseg_clean):
            pass
        else:
            msg = "Watershed in red channel failed."
            print(msg)
            messages.append(msg)
            skip = 'y'

        if np.any(grnseg_clean):
            pass
        else:
            msg = "Watershed in green channel failed."
            print(msg)
            messages.append(msg)
            skip = 'y'

        ### Create an initial guess at the borders of the cells that are in
        ### contact with one another:
        rawcontactsurface = np.zeros(cleanareas.shape, np.dtype(np.int32))
        bg_id, red_id, grn_id, blu_id = 0, 1, 2, 3 #These must correspond to the channel index + 1
        # as a result of this, you can only have 1 cell of interest per channel.
        for (i,j,k), value in np.ndenumerate(cleanborders):
            if (bg_id in cleanborders[sub(i,1):i+2, sub(j,1):j+2, :]) and (red_id in cleanborders[sub(i,1):i+2, sub(j,1):j+2, :]) and (blu_id in cleanborders[sub(i,1):i+2, sub(j,1):j+2, :]):
                rawcontactsurface[i,j,k] = cleanborders[i,j,k]

        ### If this initial guess has too few pixels in contact then consider
        ### the cells to not be in contact:
        if np.count_nonzero(rawcontactsurface) <= 6:
            print('These cells are not in contact.')
            messages.append('These cells are not in contact.')
            cells_touching = False
        elif np.count_nonzero(rawcontactsurface) > 6:
            cells_touching = True

        ### If there aren't 2 cells in the image just skip it.
        ### If either redseg_clean or grnseg_clean failed then skip the image:
        if skip == 'y':
            listupdate()
            continue



        ### If the cells are not in contact then set a bunch of values that will
        ### be in the table to be null and proceed to bin and estimate their 
        ### curvatures:
        if cells_touching == False and skip == 'n':

            ### Give the pixels of the cell borders sequential positional info
            ### starting from an arbitrary starting point:
            ordered_border_indices_red, ordered_border_indices_blu, numbered_border_indices_red, numbered_border_indices_blu, perimeter_cell1, perimeter_cell2 = count_pixels_around_borders_sequentially(cleanborders, red_id, blu_id, 0, 2)

            ### For 2 cells that are not in contact:
            ### Need to start by calculating smallest distance between the red and green borders
            if cleanborders[:,:,0].any() and cleanborders[:,:,2].any():
                closest_redborder_indices_rc, closest_bluborder_indices_rc, mean_closest_indice_loc_red, mean_closest_indice_loc_blu = find_closest_pixels_indices_between_cell_borders(cleanborders[:,:,0], cleanborders[:,:,2], ordered_border_indices_red, ordered_border_indices_blu, mean_closest_indice_loc_red_old, mean_closest_indice_loc_blu_old)
            else:
                closest_redborder_indices_rc, closest_bluborder_indices_rc, mean_closest_indice_loc_red, mean_closest_indice_loc_blu = np.array([ordered_border_indices_red[0][:2]]), np.array([ordered_border_indices_blu[0][:2]]), np.array([ordered_border_indices_red[0][:2]]), np.array([ordered_border_indices_blu[0][:2]])
            mean_closest_indice_loc_red_old = np.copy(mean_closest_indice_loc_red)
            mean_closest_indice_loc_blu_old = np.copy(mean_closest_indice_loc_blu)

            if cleanborders[:,:,0].any() and cleanborders[:,:,2].any():
                ### Find the bin closest to the other cell and remove it:
                num_bins = num_bins
                far_borders_red, size_list_of_other_bins_red, closest_bin_size_red, closest_bin_indices_red = create_and_remove_bin_closest_to_other_cell(cleanborders[:,:,0].astype(bool), ordered_border_indices_red, mean_closest_indice_loc_red, num_bins)
                
                ### Do the same for the green borders:
                far_borders_blu, size_list_of_other_bins_blu, closest_bin_size_blu, closest_bin_indices_blu = create_and_remove_bin_closest_to_other_cell(cleanborders[:,:,2].astype(bool), ordered_border_indices_blu, mean_closest_indice_loc_blu, num_bins)
    
                ### Combine the 2 borders into an RGB image and assign red to be 1 
                ### and green to be 2:
                borders_without_closest_bin = np.dstack((far_borders_red * red_id, np.zeros(tlchan.shape), far_borders_blu * blu_id))
                closest_bin_red_and_blu = np.zeros(np.shape(cleanareas))
                #closest_bin_red_and_blu[list(zip(*closest_bin_indices_red))] = red_id ###This method of indexing is going to become deprecated, so use below:
                #closest_bin_red_and_blu[list(zip(*closest_bin_indices_blu))] = blu_id
                closest_bin_red_and_blu[tuple(np.asarray(a) for a in zip(*closest_bin_indices_red))] = red_id
                closest_bin_red_and_blu[tuple(np.asarray(a) for a in zip(*closest_bin_indices_blu))] = blu_id
                reordered_closest_bin_indices_red, reordered_closest_bin_indices_blu, closest_bin_len_red, closest_bin_len_blu = make_open_border_indices_sequential(closest_bin_red_and_blu, red_id, blu_id, 0, 2)
    
                ### Re-define the sequence of pixels for the borders_without_closest_bin:
                reordered_border_indices_red, reordered_border_indices_blu, perimeter_of_bins_1to7_cell1, perimeter_of_bins_1to7_cell2 = make_open_border_indices_sequential(borders_without_closest_bin, red_id, blu_id, 0, 2)
                
                renumbered_border_indices_red = [p for p in enumerate(reordered_border_indices_red)]
                renumbered_border_indices_blu = [p for p in enumerate(reordered_border_indices_blu)]
                
                ### Next I need to use these renumbered border indices red and grn to define the
                ### other 7 bins and find their curvatures:
                num_bins = num_bins-1
                bin_indices_red, bin_start_red, bin_stop_red = sort_pixels_into_bins(reordered_border_indices_red, size_list_of_other_bins_red, closest_bin_indices_red, num_bins)
                bin_indices_blu, bin_start_blu, bin_stop_blu = sort_pixels_into_bins(reordered_border_indices_blu, size_list_of_other_bins_blu, closest_bin_indices_blu, num_bins)
                
                bin_start_red_indices = np.array(reordered_border_indices_red)[bin_start_red]            
                bin_start_red_indices = np.row_stack((bin_start_red_indices, reordered_closest_bin_indices_red[0]))
                bin_stop_red_indices = np.array(reordered_border_indices_red)[np.array(bin_stop_red)-1]
                bin_stop_red_indices = np.row_stack((bin_stop_red_indices, reordered_closest_bin_indices_red[-1]))
    
                bin_start_blu_indices = np.array(reordered_border_indices_blu)[bin_start_blu]            
                bin_start_blu_indices = np.row_stack((bin_start_blu_indices, reordered_closest_bin_indices_blu[0]))
                bin_stop_blu_indices = np.array(reordered_border_indices_blu)[np.array(bin_stop_blu)-1]
                bin_stop_blu_indices = np.row_stack((bin_stop_blu_indices, reordered_closest_bin_indices_blu[-1]))
                
    
                ### Finally, fit arcs to the binned borders and plot and return the
                ### the details to me:
                bin_curvature_details_red, all_fits_red_binned = fit_curves_to_bins(bin_indices_red, cleanborders[:,:,0], odr_circle_fit_iterations)
                bin_curvature_details_blu, all_fits_blu_binned = fit_curves_to_bins(bin_indices_blu, cleanborders[:,:,2], odr_circle_fit_iterations)
    
                ### While we're at it, also give me the curvature fit to the entire
                ### region of the cell borders not in contact with another cell:
                whole_curvature_details_red, all_fits_red_whole = fit_curves_to_bins([zip(*np.where(cleanborders[:,:,0]))], cleanborders[:,:,0], odr_circle_fit_iterations)
                whole_curvature_details_blu, all_fits_blu_whole = fit_curves_to_bins([zip(*np.where(cleanborders[:,:,2]))], cleanborders[:,:,2], odr_circle_fit_iterations)



            ### Determine the noncontactborders of the redseg:
            redseg_border_inner = np.dstack(( mh.borders(redseg_clean) * redseg_clean, np.zeros(redseg_clean.shape), np.zeros(redseg_clean.shape) ))
            redseg_border_clean_inner = clean_up_borders(redseg_border_inner)[:,:,0]
            
            redseg_border_outer = np.dstack(( mh.borders(redseg_clean) * ~redseg_clean, np.zeros(redseg_clean.shape), np.zeros(redseg_clean.shape) ))
            redseg_border_clean_outer = clean_up_borders(redseg_border_outer)[:,:,0]

            redseg_noncontactborders = np.logical_or(redseg_border_clean_inner, redseg_border_clean_outer)
            redseg_noncontactborders_filled = morphology.remove_small_holes(redseg_noncontactborders)

            ### Determine the noncontactborders of the grnseg:
            grnseg_border_inner = np.dstack(( mh.borders(grnseg_clean) * grnseg_clean, np.zeros(grnseg_clean.shape), np.zeros(grnseg_clean.shape) ))
            grnseg_border_clean_inner = clean_up_borders(grnseg_border_inner)[:,:,0]
            
            grnseg_border_outer = np.dstack(( mh.borders(grnseg_clean) * ~grnseg_clean, np.zeros(grnseg_clean.shape), np.zeros(grnseg_clean.shape) ))
            grnseg_border_clean_outer = clean_up_borders(grnseg_border_outer)[:,:,0]

            grnseg_noncontactborders = np.logical_or(grnseg_border_clean_inner, grnseg_border_clean_outer)
            grnseg_noncontactborders_filled = morphology.remove_small_holes(grnseg_noncontactborders)

            ### Determine the noncontactborders of the bluseg:
            bluseg_border_inner = np.dstack(( mh.borders(bluseg_clean) * bluseg_clean, np.zeros(bluseg_clean.shape), np.zeros(bluseg_clean.shape) ))
            bluseg_border_clean_inner = clean_up_borders(bluseg_border_inner)[:,:,0]# 0 because of the np.dstack line above
            
            bluseg_border_outer = np.dstack(( mh.borders(bluseg_clean) * ~bluseg_clean, np.zeros(bluseg_clean.shape), np.zeros(bluseg_clean.shape) ))
            bluseg_border_clean_outer = clean_up_borders(bluseg_border_outer)[:,:,0]

            bluseg_noncontactborders = np.logical_or(bluseg_border_clean_inner, bluseg_border_clean_outer)
            bluseg_noncontactborders_filled = morphology.remove_small_holes(bluseg_noncontactborders)


            # Since there is no redseg_contactsurface for this scenario find
            # the pixel closest to the other cell and remove it:
            # Since we already have the mean_closest_indice_loc_red just find 
            # the redseg_noncontactborders pixel closest to that:

            if redseg_clean.any():
                # redseg_noncontact_closest_px_distances_outer = (np.asarray(np.where(redseg_border_clean_outer)) - mean_closest_indice_loc_red[0].T)**2
                redseg_noncontact_closest_px_distances_outer = (np.asarray(np.where(redseg_border_clean_outer)) - closest_redborder_indices_rc.T)**2
                redseg_noncontact_closest_px_loc_outer = np.argmin(np.sum(redseg_noncontact_closest_px_distances_outer, axis=0))
                redseg_noncontact_closest_px_outer = list(zip(*np.where(redseg_border_clean_outer)))[redseg_noncontact_closest_px_loc_outer]
    
                # redseg_noncontact_closest_px_distances_inner = (np.asarray(np.where(redseg_border_clean_inner)) - mean_closest_indice_loc_red[0].T)**2
                redseg_noncontact_closest_px_distances_inner = (np.asarray(np.where(redseg_border_clean_inner)) - closest_redborder_indices_rc.T)**2
                redseg_noncontact_closest_px_loc_inner = np.argmin(np.sum(redseg_noncontact_closest_px_distances_inner, axis=0))
                redseg_noncontact_closest_px_inner = list(zip(*np.where(redseg_border_clean_inner)))[redseg_noncontact_closest_px_loc_inner]
    
    
                redseg_noncontactborders_outer_opened = np.copy(redseg_border_clean_outer)
                redseg_noncontactborders_outer_opened[redseg_noncontact_closest_px_outer] = False
                redseg_noncontactborders_outer_opened = np.dstack((redseg_noncontactborders_outer_opened, np.zeros(redseg_noncontactborders_outer_opened.shape), np.zeros(redseg_noncontactborders_outer_opened.shape)))
                redseg_noncontactborders_inner_opened = np.copy(redseg_border_clean_inner)
                redseg_noncontactborders_inner_opened[redseg_noncontact_closest_px_inner] = False
                redseg_noncontactborders_inner_opened = np.dstack((redseg_noncontactborders_inner_opened, np.zeros(redseg_noncontactborders_inner_opened.shape), np.zeros(redseg_noncontactborders_inner_opened.shape)))
                
    
                # Return the redseg_noncontactborders and redseg_contactsurface 
                # indices in sequential order from contactcorner to contactcorner:
                redseg_noncontact_borders_sequential_outer, blueseg_unused_outer, redchan_unused_perim_outer, bluchan_unused_perim_outer = make_open_border_indices_sequential(redseg_noncontactborders_outer_opened, red_id, blu_id, 0, 2)
                redseg_noncontact_borders_sequential_inner, blueseg_unused_inner, redchan_unused_perim_inner, bluchan_unused_perim_inner = make_open_border_indices_sequential(redseg_noncontactborders_inner_opened, red_id, blu_id, 0, 2)
    
                # Also include the closest pixel for inner and outer borders as the
                # first indice of the sequentially ordered borders:
                redseg_noncontact_borders_sequential_outer.insert(0, tuple(list(redseg_noncontact_closest_px_outer) + [0]))
                redseg_noncontact_borders_sequential_inner.insert(0, tuple(list(redseg_noncontact_closest_px_inner) + [0]))
                # Give the fluorescence for the noncontactborders of redchan and
                # grnchan in sequential order:
                redchan_fluor_outside_contact_outer = redchan[tuple(np.asarray(a) for a in zip(*redseg_noncontact_borders_sequential_outer))[:2]]
                redchan_fluor_outside_contact_inner = redchan[tuple(np.asarray(a) for a in zip(*redseg_noncontact_borders_sequential_inner))[:2]]
            
            
            # Since there is no grnseg_contactsurface for this scenario find
            # the pixel closest to the other cell and remove it:
            # Since we already have the mean_closest_indice_loc_red just find 
            # the grnseg_noncontactborders pixel closest to that:

            if grnseg_clean.any():
                # grnseg_noncontact_closest_px_distances_outer = (np.asarray(np.where(grnseg_border_clean_outer)) - mean_closest_indice_loc_red[0].T)**2
                grnseg_noncontact_closest_px_distances_outer = (np.asarray(np.where(grnseg_border_clean_outer)) - closest_redborder_indices_rc.T)**2
                grnseg_noncontact_closest_px_loc_outer = np.argmin(np.sum(grnseg_noncontact_closest_px_distances_outer, axis=0))
                grnseg_noncontact_closest_px_outer = list(zip(*np.where(grnseg_border_clean_outer)))[grnseg_noncontact_closest_px_loc_outer]
    
                # grnseg_noncontact_closest_px_distances_inner = (np.asarray(np.where(grnseg_border_clean_inner)) - mean_closest_indice_loc_red[0].T)**2
                grnseg_noncontact_closest_px_distances_inner = (np.asarray(np.where(grnseg_border_clean_inner)) - closest_redborder_indices_rc.T)**2
                grnseg_noncontact_closest_px_loc_inner = np.argmin(np.sum(grnseg_noncontact_closest_px_distances_inner, axis=0))
                grnseg_noncontact_closest_px_inner = list(zip(*np.where(grnseg_border_clean_inner)))[grnseg_noncontact_closest_px_loc_inner]
    
    
                grnseg_noncontactborders_outer_opened = np.copy(grnseg_border_clean_outer)
                grnseg_noncontactborders_outer_opened[grnseg_noncontact_closest_px_outer] = False
                grnseg_noncontactborders_outer_opened = np.dstack((grnseg_noncontactborders_outer_opened, np.zeros(grnseg_noncontactborders_outer_opened.shape), np.zeros(grnseg_noncontactborders_outer_opened.shape)))
                grnseg_noncontactborders_inner_opened = np.copy(grnseg_border_clean_inner)
                grnseg_noncontactborders_inner_opened[grnseg_noncontact_closest_px_inner] = False
                grnseg_noncontactborders_inner_opened = np.dstack((grnseg_noncontactborders_inner_opened, np.zeros(grnseg_noncontactborders_inner_opened.shape), np.zeros(grnseg_noncontactborders_inner_opened.shape)))
                
    
                # Return the redseg_noncontactborders and redseg_contactsurface 
                # indices in sequential order from contactcorner to contactcorner:
                grnseg_noncontact_borders_sequential_outer, blueseg_unused_outer, grnchan_unused_perim_outer, bluchan_unused_perim_outer = make_open_border_indices_sequential(grnseg_noncontactborders_outer_opened, red_id, blu_id, 0, 2) #grn_id
                grnseg_noncontact_borders_sequential_inner, blueseg_unused_inner, grnchan_unused_perim_inner, bluchan_unused_perim_inner = make_open_border_indices_sequential(grnseg_noncontactborders_inner_opened, red_id, blu_id, 0, 2) #grn_id
    
                # Also include the closest pixel for inner and outer borders as the
                # first indice of the sequentially ordered borders:
                grnseg_noncontact_borders_sequential_outer.insert(0, tuple(list(grnseg_noncontact_closest_px_outer) + [0]))
                grnseg_noncontact_borders_sequential_inner.insert(0, tuple(list(grnseg_noncontact_closest_px_inner) + [0]))
                # Give the fluorescence for the noncontactborders of redchan and
                # grnchan in sequential order:
                grnchan_fluor_outside_contact_outer = grnchan[tuple(np.asarray(a) for a in zip(*grnseg_noncontact_borders_sequential_outer))[:2]]
                grnchan_fluor_outside_contact_inner = grnchan[tuple(np.asarray(a) for a in zip(*grnseg_noncontact_borders_sequential_inner))[:2]]



            # Since there is no bluseg_contactsurface for this scenario find
            # the pixel closest to the other cell and remove it:
            # Since we already have the mean_closest_indice_loc_blu just find 
            # the bluseg_noncontactborders pixel closest to that:

            if bluseg_clean.any():
                bluseg_noncontact_closest_px_distances_outer = (np.asarray(np.where(bluseg_border_clean_outer)) - closest_bluborder_indices_rc.T)**2
                bluseg_noncontact_closest_px_loc_outer = np.argmin(np.sum(bluseg_noncontact_closest_px_distances_outer, axis=0))
                bluseg_noncontact_closest_px_outer = list(zip(*np.where(bluseg_border_clean_outer)))[bluseg_noncontact_closest_px_loc_outer]
    
                bluseg_noncontact_closest_px_distances_inner = (np.asarray(np.where(bluseg_border_clean_inner)) - closest_bluborder_indices_rc.T)**2
                bluseg_noncontact_closest_px_loc_inner = np.argmin(np.sum(bluseg_noncontact_closest_px_distances_inner, axis=0))
                bluseg_noncontact_closest_px_inner = list(zip(*np.where(bluseg_border_clean_inner)))[bluseg_noncontact_closest_px_loc_inner]
    
    
                bluseg_noncontactborders_outer_opened = np.copy(bluseg_border_clean_outer)
                bluseg_noncontactborders_outer_opened[bluseg_noncontact_closest_px_outer] = False
                bluseg_noncontactborders_outer_opened = np.dstack((bluseg_noncontactborders_outer_opened, np.zeros(bluseg_noncontactborders_outer_opened.shape), np.zeros(bluseg_noncontactborders_outer_opened.shape)))
                bluseg_noncontactborders_inner_opened = np.copy(bluseg_border_clean_inner)
                bluseg_noncontactborders_inner_opened[bluseg_noncontact_closest_px_inner] = False
                bluseg_noncontactborders_inner_opened = np.dstack((bluseg_noncontactborders_inner_opened, np.zeros(bluseg_noncontactborders_inner_opened.shape), np.zeros(bluseg_noncontactborders_inner_opened.shape)))
            

                # Return the redseg_noncontactborders and redseg_contactsurface 
                # indices in sequential order from contactcorner to contactcorner:
                bluseg_noncontact_borders_sequential_outer, redseg_noncontact_borders_sequential_outer, redchan_unused_perim_outer, bluchan_unused_perim_outer = make_open_border_indices_sequential(bluseg_noncontactborders_outer_opened, red_id, blu_id, 0, 2)
                bluseg_noncontact_borders_sequential_inner, redseg_noncontact_borders_sequential_inner, redchan_unused_perim_inner, bluchan_unused_perim_inner = make_open_border_indices_sequential(bluseg_noncontactborders_inner_opened, red_id, blu_id, 0, 2)
    
                # Also include the closest pixel for inner and outer borders as the
                # first indice of the sequentially ordered borders:
                bluseg_noncontact_borders_sequential_outer.insert(0, tuple(list(bluseg_noncontact_closest_px_outer) + [0]))
                bluseg_noncontact_borders_sequential_inner.insert(0, tuple(list(bluseg_noncontact_closest_px_inner) + [0]))
                # Give the fluorescence for the noncontactborders of bluchan in
                # sequential order:
                bluchan_fluor_outside_contact_outer = bluchan[tuple(np.asarray(a) for a in zip(*bluseg_noncontact_borders_sequential_outer))[:2]]
                bluchan_fluor_outside_contact_inner = bluchan[tuple(np.asarray(a) for a in zip(*bluseg_noncontact_borders_sequential_inner))[:2]]



            # As the sequentially ordered borders sometimes miss pixels that are
            # in between the inner and outer borders, also just give the
            # unordered fluorescences:
            redchan_fluor_outside_contact = redchan[np.where(redseg_noncontactborders_filled)]
            grnchan_fluor_outside_contact = grnchan[np.where(grnseg_noncontactborders_filled)]
            bluchan_fluor_outside_contact = bluchan[np.where(bluseg_noncontactborders_filled)]
            
            
            
            # Also get un-ordered cytoplasmic fluorescence values:
            redseg_cytopl_area = np.logical_xor(redseg_border_clean_inner, scp_morph.binary_fill_holes(redseg_border_clean_inner))
            grnseg_cytopl_area = np.logical_xor(grnseg_border_clean_inner, scp_morph.binary_fill_holes(grnseg_border_clean_inner))
            bluseg_cytopl_area = np.logical_xor(bluseg_border_clean_inner, scp_morph.binary_fill_holes(bluseg_border_clean_inner))

            redchan_fluor_cytoplasmic = redchan[redseg_cytopl_area]
            grnchan_fluor_cytoplasmic = grnchan[grnseg_cytopl_area]
            bluchan_fluor_cytoplasmic = bluchan[bluseg_cytopl_area]

            
            ### Define some more default values:
            area_cell_1 = len(np.where(cleanareas==red_id)[0])
            area_cell_2 = len(np.where(cleanareas==blu_id)[0])
            noncontact_perimeter_cell1 = perimeter_cell1
            noncontact_perimeter_cell2 = perimeter_cell2
            noncontactborders = np.copy(cleanborders)

            degree_angle1 = 0.0
            degree_angle2 = 0.0
            average_angle = 0.0
            red_halfangle1 = 0.0
            red_halfangle2 = 0.0
            blu_halfangle1 = 0.0
            blu_halfangle2 = 0.0
            contactsurface_len = 0.0
            

            ### Plot the cells and any parameters so far:
            fig.clf()
            fig = plt.figure(num=str(filenumber+1)+'_'+os.path.basename(filename), figsize=(16., 10.))
            plt.suptitle(str(filenumber+1)+' - '+os.path.basename(filename), verticalalignment='center', x=0.5, y=0.5, fontsize=14)
            ax1 = fig.add_subplot(241)
            ax1.yaxis.set_visible(False)
            ax1.xaxis.set_visible(False)
            ax1.imshow(tlchan, cmap='gray')
            
            ax2 = fig.add_subplot(242)
            ax2.yaxis.set_visible(False)
            ax2.xaxis.set_visible(False)
            #figcellborderred = ax2.imshow(mh.overlay(gray = mh.stretch(tlchan), red = cleanborders[:,:,0]))
            #figcellred = ax2.imshow(mh.overlay(gray = mh.stretch(redchan), red = cleanborders[:,:,0]))
            fig_noncont_redcell = ax2.imshow(mh.overlay(mh.stretch(redchan), red=redseg_noncontactborders))
            #fig_noncont_redcell = ax2.imshow(mh.stretch(redchan), cmap='gray')
            ax2.axes.autoscale(False)
            
            ax3 = fig.add_subplot(243)
            ax3.yaxis.set_visible(False)
            ax3.xaxis.set_visible(False)
            #figcellborderblu = ax3.imshow(mh.overlay(gray = mh.stretch(tlchan), blue = cleanborders[:,:,2]))
            fig_noncont_grncell = ax3.imshow(mh.overlay(mh.stretch(grnchan), green=grnseg_noncontactborders))
            #fig_noncont_grncell = ax3.imshow(mh.stretch(grnchan), cmap='gray')
            ax3.axes.autoscale(False)
            
            ax4 = fig.add_subplot(246)
            ax4.yaxis.set_visible(False)
            ax4.xaxis.set_visible(False)
            #ax4.imshow(tlchan, cmap='gray')
            fig_cont_redcell = ax4.imshow(mh.stretch(redchan), cmap='gray')
            ax4.axes.autoscale(False)
            
            ax5 = fig.add_subplot(247)
            ax5.yaxis.set_visible(False)
            ax5.xaxis.set_visible(False)
            #ax5.imshow(tlchan, cmap='gray')
            fig_cont_grncell = ax5.imshow(mh.stretch(grnchan), cmap='gray')
            ax5.axes.autoscale(False)
    
            ax6 = fig.add_subplot(245)
            ax6.yaxis.set_visible(False)
            ax6.xaxis.set_visible(False)
            ax6.imshow(mh.overlay(gray = mh.stretch(tlchan), red = cleanborders[:,:,0], blue=cleanborders[:,:,2]))
            ax6.axes.autoscale(False)
            ax6.scatter(cell_max_rcd[0][1], cell_max_rcd[0][0], color='m', marker='o')
            ax6.scatter(cell_max_rcd[1][1], cell_max_rcd[1][0], color='m', marker='o')
            ax6.scatter(bg_max_rcd[0][1], bg_max_rcd[0][0], color='m', marker='x')
            ax6.scatter(bg_max_rcd[1][1], bg_max_rcd[1][0], color='m', marker='x')
    
            ax7 = fig.add_subplot(244)
            ax7.yaxis.set_visible(False)
            ax7.xaxis.set_visible(False)
            # ax_noncont_blucell = ax7.imshow(mh.overlay(mh.stretch(bluchan), blue=cleanborders[:,:,2]))
            ax_noncont_blucell = ax7.imshow(mh.overlay(mh.stretch(bluchan), blue=bluseg_noncontactborders))
            ax7.axes.autoscale(False)
            
            ax8 = fig.add_subplot(248)
            ax8.yaxis.set_visible(False)
            ax8.xaxis.set_visible(False)
            fig_cont_blucell = ax8.imshow(mh.stretch(bluchan), cmap='gray')
            ax8.axes.autoscale(False)

    
            plt.tight_layout()
            plt.draw()


            ### Update the data table:
            listupdate()
            
            ### Move on to the next picture:
            continue


        ### If the cells ARE in contact, then proceed to calculate adhesiveness
        ### parameters:
        if cells_touching == True and skip == 'n':
            
            ### Clean up the contacts:
            contactcorners, contactsurface, noncontactborders = clean_up_contact_borders(rawcontactsurface, cleanborders, red_id, blu_id)
            
            ### Check to make sure that the contact corneres were properly
            ### detected:
            if len(np.where(contactcorners)[0]) == 4:
                print("Found the contact corners.")
            ### Or else just skip this image and fill any unfinished analyses
            ### with null data:
            else:
                print ("I failed to find the contact corners :c "
                       "\n Please review the processing of this file to see what "
                       "went wrong."
                       "\n The file is %s" % filename,
                       "\n The parameters used were: \n"
                       "\nRed gaussian filter value:", gaussian_filter_value_red,
                       "\nGreen gaussian filter value:", gaussian_filter_value_grn,
                       #"\nRed thresholding value:", redthreshval,
                       #"\nGreen thresholding value:", grnthreshval,
                       )
                messages.append("I failed to find the contact corners.")
                listupdate()
                continue

            ### Now that we have defined the corners of the contact, the contact
            ### surface, and the parts of the cell border that are not in
            ### contact, we can find the contact angles:
            temp = find_contact_angles(contactcorners, contactsurface, noncontactborders, cell1_id = red_id, cell2_id = blu_id)
            degree_angle1, degree_angle2, average_angle = temp[0], temp[1], temp[2]
            vector1_start, vector1 = temp[3], temp[4]
            vector2_start, vector2 = temp[5], temp[6]
            vector3_start, vector3 = temp[7], temp[8]
            vector4_start, vector4 = temp[9], temp[10]
            pathfinder = temp[11]
            angles_angle_labeled = temp[12]
            print('The angles are:', degree_angle1, 'and', degree_angle2)

            ### Let's also find the half-angles of these angles. First some
            ### preliminary steps: Find the vector that divides the contact angles:
            rawcontact_vector_start = (np.asarray(vector1_start[::-1]) + np.asarray(vector2_start[::-1])) / 2.0
            rawcontact_vector_end = (np.asarray(vector3_start[::-1]) + np.array(vector4_start[::-1])) / 2.0
            rawcontact_vector = rawcontact_vector_start - rawcontact_vector_end
                
            ### Convert the raw contact vector in to (x,y) format:
            contact_vector = np.zeros((2))
            contact_vector[0] = rawcontact_vector[1]
            contact_vector[1] = rawcontact_vector[0]
            perpendicular2contact_vector = rawcontact_vector[:2]
            perpendicular2contact_vector *=[-1.0,1.0]
            if np.dot(contact_vector,perpendicular2contact_vector) != 0:
                raise(ValueError)
                print("Linescan of gap is not perpendicular to contact surface!")
                messages.append("Linescan of gap is not perpendicular to contact surface!")
            
            n_contactsurface_px = int(len(np.where(contactsurface)[0])/2)
            cx,cy = np.linspace(rawcontact_vector_start[1], rawcontact_vector_end[1], num=n_contactsurface_px), np.linspace(rawcontact_vector_start[0], rawcontact_vector_end[0], num=n_contactsurface_px)
            imgform_contact_points = zip(cy,cx)
        
        
            ### Determine the half-angle for each of the red and green cells:
            red_halfangle1_rad = math.acos( np.round( ((np.dot(vector1,contact_vector)) / (((vector1[0]**2 + vector1[1]**2)**0.5)*((contact_vector[0]**2 + contact_vector[1]**2)**0.5)) ), decimals=10) )
            red_halfangle2_rad = math.acos( np.round( ((np.dot(vector3,contact_vector)) / (((vector3[0]**2 + vector3[1]**2)**0.5)*((contact_vector[0]**2 + contact_vector[1]**2)**0.5)) ), decimals=10) )
            blu_halfangle1_rad = math.acos( np.round( ((np.dot(vector2,contact_vector)) / (((vector2[0]**2 + vector2[1]**2)**0.5)*((contact_vector[0]**2 + contact_vector[1]**2)**0.5)) ), decimals=10) )
            blu_halfangle2_rad = math.acos( np.round( ((np.dot(vector4,contact_vector)) / (((vector4[0]**2 + vector4[1]**2)**0.5)*((contact_vector[0]**2 + contact_vector[1]**2)**0.5)) ), decimals=10) )
            ### and convert to degrees:
            red_halfangle1 = math.degrees(red_halfangle1_rad)
            red_halfangle2 = 180-math.degrees(red_halfangle2_rad)
            blu_halfangle1 = math.degrees(blu_halfangle1_rad)
            blu_halfangle2 = 180-math.degrees(blu_halfangle2_rad)


            ### Find the mean fluorescence intensity value that the red and grn
            ### linescans of the contact intersect at as an indirect measure of
            ### how close the 2 cells are:
            #red_gap_scanx, red_gap_scany, grn_gap_scanx, grn_gap_scany, blu_gap_scanx, blu_gap_scany = perform_linescan(contactsurface, perpendicular2contact_vector, linescan_length, fluor_chans, red_id, blu_id)

            ### Find the perimeter of the red and green cells:
            ordered_border_indices_red, ordered_border_indices_blu, numbered_border_indices_red, numbered_border_indices_blu, perimeter_cell1, perimeter_cell2 = count_pixels_around_borders_sequentially(cleanborders, red_id, blu_id, 0, 2)

            ### Give the noncontactborders sequential information:
            reordered_border_indices_red, reordered_border_indices_blu, noncontact_perimeter_cell1, noncontact_perimeter_cell2 = make_open_border_indices_sequential(noncontactborders+pathfinder, red_id, blu_id, 0, 2)


            renumbered_border_indices_red = [p for p in enumerate(reordered_border_indices_red)]
            renumbered_border_indices_blu = [p for p in enumerate(reordered_border_indices_blu)]

            ### Bin the noncontactborders into 7 bins:
            num_bins = num_bins-1
            bin_size_list_red = create_border_bin_sizes((noncontactborders[:,:,0]+pathfinder[:,:,0]).astype(bool), reordered_border_indices_red, num_bins)
            bin_size_list_blu = create_border_bin_sizes((noncontactborders[:,:,2]+pathfinder[:,:,2]).astype(bool), reordered_border_indices_blu, num_bins)

            ### Make the 8th bin the contactsurface:
#            contact_bin_indices_red = zip(*np.where(contactsurface==1))
#            contact_bin_indices_grn = zip(*np.where(contactsurface==2))
            reordered_contact_surface_red, reordered_contact_surface_blu, contactsurface_len1, contactsurface_len2 = make_open_border_indices_sequential(contactsurface, red_id, blu_id, 0, 2)
            contactsurface_len = (contactsurface_len1 + contactsurface_len2) / 2.0
            print("The length of the contact surface is %.3f pixels" %contactsurface_len)

            ### Return the bin-sorted indices for each pixel:
#            bin_indices_red, bin_start_red, bin_stop_red = sort_pixels_into_bins(reordered_border_indices_red, bin_size_list_red, contact_bin_indices_red, num_bins)
#            bin_indices_grn, bin_start_grn, bin_stop_grn = sort_pixels_into_bins(reordered_border_indices_grn, bin_size_list_grn, contact_bin_indices_grn, num_bins)
            bin_indices_red, bin_start_red, bin_stop_red = sort_pixels_into_bins(reordered_border_indices_red, bin_size_list_red, reordered_contact_surface_red, num_bins)
            bin_indices_blu, bin_start_blu, bin_stop_blu = sort_pixels_into_bins(reordered_border_indices_blu, bin_size_list_blu, reordered_contact_surface_blu, num_bins)

            bin_start_red_indices = np.array(reordered_border_indices_red)[bin_start_red]            
            bin_start_red_indices = np.row_stack((bin_start_red_indices, reordered_contact_surface_red[0]))
            bin_stop_red_indices = np.array(reordered_border_indices_red)[np.array(bin_stop_red)-1]
            bin_stop_red_indices = np.row_stack((bin_stop_red_indices, reordered_contact_surface_red[-1]))

            bin_start_blu_indices = np.array(reordered_border_indices_blu)[bin_start_blu]            
            bin_start_blu_indices = np.row_stack((bin_start_blu_indices, reordered_contact_surface_blu[0]))
            bin_stop_blu_indices = np.array(reordered_border_indices_blu)[np.array(bin_stop_blu)-1]
            bin_stop_blu_indices = np.row_stack((bin_stop_blu_indices, reordered_contact_surface_blu[-1]))



            ### Add the len of the 8th bin to the bin_size_list_red (and grn):
#            bin_size_list_red.append(len(bin_indices_red[7]))
#            bin_size_list_grn.append(len(bin_indices_grn[7]))

            bin_start_red.append(bin_indices_red[7][0])
            bin_stop_red.append(bin_indices_red[7][-1])
            bin_start_blu.append(bin_indices_blu[7][0])
            bin_stop_blu.append(bin_indices_blu[7][-1])

            ### Find the binned curvatures:
            bin_curvature_details_red, all_fits_red_binned = fit_curves_to_bins(bin_indices_red, cleanborders[:,:,0], odr_circle_fit_iterations)
            bin_curvature_details_blu, all_fits_blu_binned = fit_curves_to_bins(bin_indices_blu, cleanborders[:,:,2], odr_circle_fit_iterations)
                 
            ### While we're at it, also give me the curvature fit to the entire
            ### region of the cell borders not in contact with another cell:
            whole_curvature_details_red, all_fits_red_whole = fit_curves_to_bins([zip(*np.where((noncontactborders[:,:,0]+pathfinder[:,:,0]).astype(bool)))], (noncontactborders[:,:,0]+pathfinder[:,:,0]).astype(bool), odr_circle_fit_iterations)
            whole_curvature_details_blu, all_fits_blu_whole = fit_curves_to_bins([zip(*np.where((noncontactborders[:,:,2]+pathfinder[:,:,2]).astype(bool)))], (noncontactborders[:,:,2]+pathfinder[:,:,2]).astype(bool), odr_circle_fit_iterations)

            ### Define some more default values:
            area_cell_1 = len(np.where(cleanareas==red_id)[0])
            area_cell_2 = len(np.where(cleanareas==blu_id)[0])




            # Find redseg inner and outer border pixels that are closest to the
            # contact corners of red_areas:
            redseg_border_inner = np.dstack(( mh.borders(redseg_clean) * redseg_clean, np.zeros(redseg_clean.shape), np.zeros(redseg_clean.shape) ))
            redseg_border_clean_inner = clean_up_borders(redseg_border_inner)[:,:,0]
            
            redseg_border_outer = np.dstack(( mh.borders(redseg_clean) * ~redseg_clean, np.zeros(redseg_clean.shape), np.zeros(redseg_clean.shape) ))
            redseg_border_clean_outer = clean_up_borders(redseg_border_outer)[:,:,0]

            
            redseg_inner_corner_locs = []
            redseg_outer_corner_locs = []
            for corner in zip(*np.where(contactcorners[:, :, channel1])):
                redseg_inner_distances = np.ma.array((np.where(redseg_border_clean_inner)[0] - corner[0]) ** 2 + (np.where(redseg_border_clean_inner)[1] - corner[1]) ** 2)
                redseg_inner_closest_iloc = np.argmin(redseg_inner_distances)               
                redseg_inner_corner_indice = list(zip(*np.where(redseg_border_clean_inner)))[redseg_inner_closest_iloc]
                if redseg_inner_corner_indice in redseg_inner_corner_locs:
                    #print("This pixels has already been recorded!")
                    redseg_inner_distances[redseg_inner_closest_iloc] = np.ma.masked
                    redseg_inner_closest_iloc = np.argmin(redseg_inner_distances)               
                    redseg_inner_corner_indice = list(zip(*np.where(redseg_border_clean_inner)))[redseg_inner_closest_iloc]
                redseg_inner_corner_locs.append(redseg_inner_corner_indice)
                
                
                redseg_outer_distances = np.ma.array((np.where(redseg_border_clean_outer)[0] - corner[0]) ** 2 + (np.where(redseg_border_clean_outer)[1] - corner[1]) ** 2)
                redseg_outer_closest_iloc = np.argmin(redseg_outer_distances) 
                redseg_outer_corner_indice = list(zip(*np.where(redseg_border_clean_outer)))[redseg_outer_closest_iloc]
                if redseg_outer_corner_indice in redseg_outer_corner_locs:
                    #print("This pixels has already been recorded!")
                    redseg_outer_distances[redseg_outer_closest_iloc] = np.ma.masked
                    redseg_outer_closest_iloc = np.argmin(redseg_outer_distances) 
                    redseg_outer_corner_indice = list(zip(*np.where(redseg_border_clean_outer)))[redseg_outer_closest_iloc]
                redseg_outer_corner_locs.append(redseg_outer_corner_indice)

            redseg_inner_corners = np.zeros(redseg_clean.shape)
            redseg_inner_corners[tuple(np.asarray(a) for a in zip(*redseg_inner_corner_locs))] = True
            redseg_outer_corners = np.zeros(redseg_clean.shape)
            redseg_outer_corners[tuple(np.asarray(a) for a in zip(*redseg_outer_corner_locs))] = True
            
            redseg_inner_contactsurface = np.copy(redseg_border_clean_inner)
            redseg_inner_noncontactborders = np.copy(redseg_border_clean_inner)
            # Break the redseg_border_clean_inner at the contact corners:
            redseg_inner_contactsurface[np.where(redseg_inner_corners)] = 0
            redseg_inner_noncontactborders[np.where(redseg_inner_corners)] = 0
            
            # Label the resulting 2 pieces so that they can be classified as either
            # contactsurface or noncontactborders:
            redseg_inner_contactsurface = morphology.label(redseg_inner_contactsurface, connectivity=2)
            if len(np.unique(redseg_inner_contactsurface)) <= 3:
                pass
            else:
                redseg_inner_contactsurface = np.zeros(redseg_inner_contactsurface.shape)

            # When the contactsurface is small, sometimes the redseg_contactsurface
            # consists only of the redseg_contactcorners. In this case there will
            # be only 1 piece (the noncontactborders) when the contactcorners are
            # removed. Therefore add the redseg_contactcorners to each possible
            # surface:
            borders1 = (redseg_inner_contactsurface == 1) + redseg_inner_corners.astype(bool)
            borders2 = (redseg_inner_contactsurface == 2) + redseg_inner_corners.astype(bool)
            
            
            # Compare similarity between label 1 and label 2:
            #ssim_borders1 = structural_similarity((redseg_inner_contactsurface == 1).astype(bool), contactsurface[:,:,0].astype(bool))
            #ssim_borders2 = structural_similarity((redseg_inner_contactsurface == 2).astype(bool), contactsurface[:,:,0].astype(bool))
            ssim_borders1 = structural_similarity(borders1, contactsurface[:,:,0].astype(bool))
            ssim_borders2 = structural_similarity(borders2, contactsurface[:,:,0].astype(bool))
            # Classify the more similar label as being contactsurface,
            # put the noncontactborders in a new array and remove the 
            # noncontactborders from the contactsurface array:
            if ssim_borders2 > ssim_borders1:
                redseg_inner_noncontactborders[np.where(borders2)] = False
                redseg_inner_noncontactborders = redseg_inner_noncontactborders.astype(bool)
                redseg_inner_contactsurface[np.where(borders1)] = False
                redseg_inner_contactsurface = redseg_inner_contactsurface.astype(bool)
                redseg_inner_contactsurface[np.where(redseg_inner_corners)] = True
            elif ssim_borders1 > ssim_borders2:
                redseg_inner_noncontactborders[np.where(borders1)] = False
                redseg_inner_noncontactborders = redseg_inner_noncontactborders.astype(bool)
                redseg_inner_contactsurface[np.where(borders2)] = False
                redseg_inner_contactsurface = redseg_inner_contactsurface.astype(bool)
                redseg_inner_contactsurface[np.where(redseg_inner_corners)] = True
            else:
                redseg_inner_contactsurface = np.zeros(redseg_inner_contactsurface.shape)
                redseg_inner_noncontactborders = np.zeros(redseg_inner_contactsurface.shape)
            
            # Do the same for the outer borders:
            redseg_outer_contactsurface = np.copy(redseg_border_clean_outer)
            redseg_outer_noncontactborders = np.copy(redseg_border_clean_outer)
            
            redseg_outer_contactsurface[np.where(redseg_outer_corners)] = False
            redseg_outer_noncontactborders[np.where(redseg_outer_corners)] = False

            redseg_outer_contactsurface = morphology.label(redseg_outer_contactsurface, connectivity=2)
            if len(np.unique(redseg_outer_contactsurface)) <= 3:
                pass
            else:
                redseg_outer_contactsurface = np.zeros(redseg_outer_contactsurface.shape)
            
            # When the contactsurface is small, sometimes the redseg_contactsurface
            # consists only of the redseg_contactcorners. In this case there will
            # be only 1 piece (the noncontactborders) when the contactcorners are
            # removed. Therefore add the redseg_contactcorners to each possible
            # surface:
            borders1 = (redseg_outer_contactsurface == 1) + redseg_outer_corners.astype(bool)
            borders2 = (redseg_outer_contactsurface == 2) + redseg_outer_corners.astype(bool)
            
            
            # Compare similarity between label 1 and label 2:
            ssim_borders1 = structural_similarity(borders1, contactsurface[:,:,0].astype(bool))
            ssim_borders2 = structural_similarity(borders2, contactsurface[:,:,0].astype(bool))
            # Classify the more similar label as being contactsurface,
            # put the noncontactborders in a new array and remove the 
            # noncontactborders from the contactsurface array:
            if ssim_borders2 > ssim_borders1:
                redseg_outer_noncontactborders[np.where(borders2)] = False
                redseg_outer_noncontactborders = redseg_outer_noncontactborders.astype(bool)
                redseg_outer_contactsurface[np.where(borders1)] = False
                redseg_outer_contactsurface = redseg_outer_contactsurface.astype(bool)
                redseg_outer_contactsurface[np.where(redseg_outer_corners)] = True
            elif ssim_borders1 > ssim_borders2:
                redseg_outer_noncontactborders[np.where(borders1)] = False
                redseg_outer_noncontactborders = redseg_outer_noncontactborders.astype(bool)
                redseg_outer_contactsurface[np.where(borders2)] = False
                redseg_outer_contactsurface = redseg_outer_contactsurface.astype(bool)
                redseg_outer_contactsurface[np.where(redseg_outer_corners)] = True
            else:
                redseg_outer_contactsurface = np.zeros(redseg_outer_contactsurface.shape)
                redseg_outer_noncontactborders = np.zeros(redseg_outer_contactsurface.shape)

            # Combine the inner and outer contactsurface back together.
            # Also do the same for inner and outer noncontactborders.
            redseg_contactsurface = np.logical_or(redseg_inner_contactsurface, redseg_outer_contactsurface)
            redseg_noncontactborders = np.logical_or(redseg_inner_noncontactborders, redseg_outer_noncontactborders)

            # Restore any removed pixels from the clean_up_borders function to
            # either the contactsurface or noncontactborders that end up being
            # right in between the inner and outer borders:
            redseg_contactsurface_filled = morphology.remove_small_holes(redseg_contactsurface)
            redseg_noncontactborders_filled = morphology.remove_small_holes(redseg_noncontactborders)
            
            # Return pixels in sequential order for inner and outer contactsurface:
            redseg_inner_contactsurface = np.dstack((redseg_inner_contactsurface, np.zeros(redseg_inner_contactsurface.shape), np.zeros(redseg_inner_contactsurface.shape)))
            redseg_inner_contactsurface_seq, blu_unsused, red_unused_perim, blu_unused_perim = make_open_border_indices_sequential(redseg_inner_contactsurface, red_id, blu_id, 0, 2)
            redseg_outer_contactsurface = np.dstack((redseg_outer_contactsurface, np.zeros(redseg_outer_contactsurface.shape), np.zeros(redseg_outer_contactsurface.shape)))
            redseg_outer_contactsurface_seq, blu_unused, red_unused_perim, blue_unused_perim = make_open_border_indices_sequential(redseg_outer_contactsurface, red_id, blu_id, 0, 2)
            
            # Also do same for inner and outer noncontactborders:
            redseg_inner_noncontactborders = np.dstack((redseg_inner_noncontactborders, np.zeros(redseg_inner_noncontactborders.shape), np.zeros(redseg_inner_noncontactborders.shape)))
            redseg_inner_noncontactborders_seq, blu_unused, red_unused_perim, blu_unused_perim = make_open_border_indices_sequential(redseg_inner_noncontactborders, red_id, blu_id, 0, 2)
            redseg_outer_noncontactborders = np.dstack((redseg_outer_noncontactborders, np.zeros(redseg_outer_noncontactborders.shape), np.zeros(redseg_outer_noncontactborders.shape)))
            redseg_outer_noncontactborders_seq, blu_unused, red_unused_perim, blu_unused_perim = make_open_border_indices_sequential(redseg_outer_noncontactborders, red_id, blu_id, 0, 2)
            
            
            
            # Find grnseg inner and outer border pixels that are closest to the
            # contact corners of grn_areas:
            grnseg_border_inner = np.dstack(( mh.borders(grnseg_clean) * grnseg_clean, np.zeros(grnseg_clean.shape), np.zeros(grnseg_clean.shape) ))
            grnseg_border_clean_inner = clean_up_borders(grnseg_border_inner)[:,:,0]
            
            grnseg_border_outer = np.dstack(( mh.borders(grnseg_clean) * ~grnseg_clean, np.zeros(grnseg_clean.shape), np.zeros(grnseg_clean.shape) ))
            grnseg_border_clean_outer = clean_up_borders(grnseg_border_outer)[:,:,0]
            
            grnseg_inner_corner_locs = []
            grnseg_outer_corner_locs = []
            for corner in zip(*np.where(contactcorners[:, :, channel1])):
                grnseg_inner_distances  = np.ma.array((np.where(grnseg_border_clean_inner)[0] - corner[0]) ** 2 + (np.where(grnseg_border_clean_inner)[1] - corner[1]) ** 2)
                grnseg_inner_closest_iloc = np.argmin(grnseg_inner_distances) 
                grnseg_inner_corner_indice = list(zip(*np.where(grnseg_border_clean_inner)))[grnseg_inner_closest_iloc]
                if grnseg_inner_corner_indice in grnseg_inner_corner_locs:
                    #print("This pixels has already been recorded!")
                    grnseg_inner_distances[grnseg_inner_closest_iloc] = np.ma.masked
                    grnseg_inner_closest_iloc = np.argmin(grnseg_inner_distances) 
                    grnseg_inner_corner_indice = list(zip(*np.where(grnseg_border_clean_inner)))[grnseg_inner_closest_iloc]
                grnseg_inner_corner_locs.append(grnseg_inner_corner_indice)
                
                grnseg_outer_distances = np.ma.array((np.where(grnseg_border_clean_outer)[0] - corner[0]) ** 2 + (np.where(grnseg_border_clean_outer)[1] - corner[1]) ** 2)
                grnseg_outer_closest_iloc = np.argmin(grnseg_outer_distances) 
                grnseg_outer_corner_indice = list(zip(*np.where(grnseg_border_clean_outer)))[grnseg_outer_closest_iloc]
                if grnseg_outer_corner_indice in grnseg_outer_corner_locs:
                    #print("This pixels has already been recorded!")
                    grnseg_outer_distances[grnseg_outer_closest_iloc] = np.ma.masked
                    grnseg_outer_closest_iloc = np.argmin(grnseg_outer_distances) 
                    grnseg_outer_corner_indice = list(zip(*np.where(grnseg_border_clean_outer)))[grnseg_outer_closest_iloc]
                grnseg_outer_corner_locs.append(grnseg_outer_corner_indice)

            grnseg_inner_corners = np.zeros(grnseg_clean.shape)
            grnseg_inner_corners[tuple(np.asarray(a) for a in zip(*grnseg_inner_corner_locs))] = True
            grnseg_outer_corners = np.zeros(grnseg_clean.shape)
            grnseg_outer_corners[tuple(np.asarray(a) for a in zip(*grnseg_outer_corner_locs))] = True
            
            grnseg_inner_contactsurface = np.copy(grnseg_border_clean_inner)
            grnseg_inner_noncontactborders = np.copy(grnseg_border_clean_inner)
            # Break the redseg_border_clean_inner at the contact corners:
            grnseg_inner_contactsurface[np.where(grnseg_inner_corners)] = 0
            grnseg_inner_noncontactborders[np.where(grnseg_inner_corners)] = 0
            
            # Label the resulting 2 pieces so that they can be classified as either
            # contactsurface or noncontactborders:
            grnseg_inner_contactsurface = morphology.label(grnseg_inner_contactsurface, connectivity=2)
            if len(np.unique(grnseg_inner_contactsurface)) <= 3:
                pass
            else:
                grnseg_inner_contactsurface = np.zeros(grnseg_inner_contactsurface.shape)
            
            # When the contactsurface is small, sometimes the redseg_contactsurface
            # consists only of the redseg_contactcorners. In this case there will
            # be only 1 piece (the noncontactborders) when the contactcorners are
            # removed. Therefore add the redseg_contactcorners to each possible
            # surface:
            borders1 = (grnseg_inner_contactsurface == 1) + grnseg_inner_corners.astype(bool)
            borders2 = (grnseg_inner_contactsurface == 2) + grnseg_inner_corners.astype(bool)
            
            # Compare similarity between label 1 and label 2:
            #ssim_borders1 = structural_similarity((redseg_inner_contactsurface == 1).astype(bool), contactsurface[:,:,0].astype(bool))
            #ssim_borders2 = structural_similarity((redseg_inner_contactsurface == 2).astype(bool), contactsurface[:,:,0].astype(bool))
            ssim_borders1 = structural_similarity(borders1, contactsurface[:,:,0].astype(bool))
            ssim_borders2 = structural_similarity(borders2, contactsurface[:,:,0].astype(bool))
            # Classify the more similar label as being contactsurface,
            # put the noncontactborders in a new array and remove the 
            # noncontactborders from the contactsurface array:
            if ssim_borders2 > ssim_borders1:
                grnseg_inner_noncontactborders[np.where(borders2)] = False
                grnseg_inner_noncontactborders = grnseg_inner_noncontactborders.astype(bool)
                grnseg_inner_contactsurface[np.where(borders1)] = False
                grnseg_inner_contactsurface = grnseg_inner_contactsurface.astype(bool)
                grnseg_inner_contactsurface[np.where(grnseg_inner_corners)] = True
            elif ssim_borders1 > ssim_borders2:
                grnseg_inner_noncontactborders[np.where(borders1)] = False
                grnseg_inner_noncontactborders = grnseg_inner_noncontactborders.astype(bool)
                grnseg_inner_contactsurface[np.where(borders2)] = False
                grnseg_inner_contactsurface = grnseg_inner_contactsurface.astype(bool)
                grnseg_inner_contactsurface[np.where(grnseg_inner_corners)] = True
            else:
                grnseg_inner_contactsurface = np.zeros(grnseg_inner_contactsurface.shape)
                grnseg_inner_noncontactborders = np.zeros(grnseg_inner_contactsurface.shape)
            
            # Do the same for the outer borders:
            grnseg_outer_contactsurface = np.copy(grnseg_border_clean_outer)
            grnseg_outer_noncontactborders = np.copy(grnseg_border_clean_outer)
            
            grnseg_outer_contactsurface[np.where(grnseg_outer_corners)] = False
            grnseg_outer_noncontactborders[np.where(grnseg_outer_corners)] = False

            grnseg_outer_contactsurface = morphology.label(grnseg_outer_contactsurface, connectivity=2)
            if len(np.unique(grnseg_outer_contactsurface)) <= 3:
                pass
            else:
                grnseg_outer_contactsurface = np.zeros(grnseg_outer_contactsurface.shape)
            
            # When the contactsurface is small, sometimes the redseg_contactsurface
            # consists only of the redseg_contactcorners. In this case there will
            # be only 1 piece (the noncontactborders) when the contactcorners are
            # removed. Therefore add the redseg_contactcorners to each possible
            # surface:
            borders1 = (grnseg_outer_contactsurface == 1) + grnseg_outer_corners.astype(bool)
            borders2 = (grnseg_outer_contactsurface == 2) + grnseg_outer_corners.astype(bool)


            # Compare similarity between label 1 and label 2:
            ssim_borders1 = structural_similarity(borders1, contactsurface[:,:,0].astype(bool))
            ssim_borders2 = structural_similarity(borders2.astype(bool), contactsurface[:,:,0].astype(bool))
            # Classify the more similar label as being contactsurface,
            # put the noncontactborders in a new array and remove the 
            # noncontactborders from the contactsurface array:
            if ssim_borders2 > ssim_borders1:
                grnseg_outer_noncontactborders[np.where(borders2)] = False
                grnseg_outer_noncontactborders = grnseg_outer_noncontactborders.astype(bool)
                grnseg_outer_contactsurface[np.where(borders1)] = False
                grnseg_outer_contactsurface = grnseg_outer_contactsurface.astype(bool)
                grnseg_outer_contactsurface[np.where(grnseg_outer_corners)] = True
            elif ssim_borders1 > ssim_borders2:
                grnseg_outer_noncontactborders[np.where(borders1)] = False
                grnseg_outer_noncontactborders = grnseg_outer_noncontactborders.astype(bool)
                grnseg_outer_contactsurface[np.where(borders2)] = False
                grnseg_outer_contactsurface = grnseg_outer_contactsurface.astype(bool)
                grnseg_outer_contactsurface[np.where(grnseg_outer_corners)] = True
            else:
                grnseg_outer_contactsurface = np.zeros(grnseg_outer_contactsurface.shape)
                grnseg_outer_noncontactborders = np.zeros(grnseg_outer_contactsurface.shape)

            # Combine the inner and outer contactsurface back together.
            # Also do the same for inner and outer noncontactborders.
            grnseg_contactsurface = np.logical_or(grnseg_inner_contactsurface, grnseg_outer_contactsurface)
            grnseg_noncontactborders = np.logical_or(grnseg_inner_noncontactborders, grnseg_outer_noncontactborders)

            # Restore any removed pixels from the clean_up_borders function to
            # either the contactsurface or noncontactborders that end up being
            # right in between the inner and outer borders:
            grnseg_contactsurface_filled = morphology.remove_small_holes(grnseg_contactsurface)
            grnseg_noncontactborders_filled = morphology.remove_small_holes(grnseg_noncontactborders)
            
            
            # Return pixels in sequential order for inner and outer contactsurface:
            grnseg_inner_contactsurface = np.dstack((grnseg_inner_contactsurface, np.zeros(grnseg_inner_contactsurface.shape), np.zeros(grnseg_inner_contactsurface.shape)))
            grnseg_inner_contactsurface_seq, blu_unsused, grn_unused_perim, blu_unused_perim = make_open_border_indices_sequential(grnseg_inner_contactsurface, red_id, blu_id, 0, 2) #grn_id
            grnseg_outer_contactsurface = np.dstack((grnseg_outer_contactsurface, np.zeros(grnseg_outer_contactsurface.shape), np.zeros(grnseg_outer_contactsurface.shape)))
            grnseg_outer_contactsurface_seq, blu_unused, grn_unused_perim, blue_unused_perim = make_open_border_indices_sequential(grnseg_outer_contactsurface, red_id, blu_id, 0, 2) #grn_id
            
            # Also do same for inner and outer noncontactborders:
            grnseg_inner_noncontactborders = np.dstack((grnseg_inner_noncontactborders, np.zeros(grnseg_inner_noncontactborders.shape), np.zeros(grnseg_inner_noncontactborders.shape)))
            grnseg_inner_noncontactborders_seq, blu_unused, grn_unused_perim, blu_unused_perim = make_open_border_indices_sequential(grnseg_inner_noncontactborders, red_id, blu_id, 0, 2) #grn_id
            grnseg_outer_noncontactborders = np.dstack((grnseg_outer_noncontactborders, np.zeros(grnseg_outer_noncontactborders.shape), np.zeros(grnseg_outer_noncontactborders.shape)))
            grnseg_outer_noncontactborders_seq, blu_unused, grn_unused_perim, blu_unused_perim = make_open_border_indices_sequential(grnseg_outer_noncontactborders, red_id, blu_id, 0, 2) #grn_id



            # Find bluseg inner and outer border pixels that are closest to the
            # contact corners of blu_areas:
            bluseg_border_inner = np.dstack(( mh.borders(bluseg_clean) * bluseg_clean, np.zeros(bluseg_clean.shape), np.zeros(bluseg_clean.shape) ))
            bluseg_border_clean_inner = clean_up_borders(bluseg_border_inner)[:,:,0]
            
            bluseg_border_outer = np.dstack(( mh.borders(bluseg_clean) * ~bluseg_clean, np.zeros(bluseg_clean.shape), np.zeros(bluseg_clean.shape) ))
            bluseg_border_clean_outer = clean_up_borders(bluseg_border_outer)[:,:,0]
            
            bluseg_inner_corner_locs = []
            bluseg_outer_corner_locs = []
            for corner in zip(*np.where(contactcorners[:, :, channel2])):
                bluseg_inner_distances  = np.ma.array((np.where(bluseg_border_clean_inner)[0] - corner[0]) ** 2 + (np.where(bluseg_border_clean_inner)[1] - corner[1]) ** 2)
                bluseg_inner_closest_iloc = np.argmin(bluseg_inner_distances) 
                bluseg_inner_corner_indice = list(zip(*np.where(bluseg_border_clean_inner)))[bluseg_inner_closest_iloc]
                if bluseg_inner_corner_indice in bluseg_inner_corner_locs:
                    #print("This pixels has already been recorded!")
                    bluseg_inner_distances[bluseg_inner_closest_iloc] = np.ma.masked
                    bluseg_inner_closest_iloc = np.argmin(bluseg_inner_distances) 
                    bluseg_inner_corner_indice = list(zip(*np.where(bluseg_border_clean_inner)))[bluseg_inner_closest_iloc]
                bluseg_inner_corner_locs.append(bluseg_inner_corner_indice)
                
                bluseg_outer_distances = np.ma.array((np.where(bluseg_border_clean_outer)[0] - corner[0]) ** 2 + (np.where(bluseg_border_clean_outer)[1] - corner[1]) ** 2)
                bluseg_outer_closest_iloc = np.argmin(bluseg_outer_distances) 
                bluseg_outer_corner_indice = list(zip(*np.where(bluseg_border_clean_outer)))[bluseg_outer_closest_iloc]
                if bluseg_outer_corner_indice in bluseg_outer_corner_locs:
                    #print("This pixels has already been recorded!")
                    bluseg_outer_distances[bluseg_outer_closest_iloc] = np.ma.masked
                    bluseg_outer_closest_iloc = np.argmin(bluseg_outer_distances) 
                    bluseg_outer_corner_indice = list(zip(*np.where(bluseg_border_clean_outer)))[bluseg_outer_closest_iloc]
                bluseg_outer_corner_locs.append(bluseg_outer_corner_indice)

            bluseg_inner_corners = np.zeros(bluseg_clean.shape)
            bluseg_inner_corners[tuple(np.asarray(a) for a in zip(*bluseg_inner_corner_locs))] = True
            bluseg_outer_corners = np.zeros(bluseg_clean.shape)
            bluseg_outer_corners[tuple(np.asarray(a) for a in zip(*bluseg_outer_corner_locs))] = True
            
            bluseg_inner_contactsurface = np.copy(bluseg_border_clean_inner)
            bluseg_inner_noncontactborders = np.copy(bluseg_border_clean_inner)
            # Break the bluseg_border_clean_inner at the contact corners:
            bluseg_inner_contactsurface[np.where(bluseg_inner_corners)] = 0
            bluseg_inner_noncontactborders[np.where(bluseg_inner_corners)] = 0
            
            # Label the resulting 2 pieces so that they can be classified as either
            # contactsurface or noncontactborders:
            bluseg_inner_contactsurface = morphology.label(bluseg_inner_contactsurface, connectivity=2)
            if len(np.unique(bluseg_inner_contactsurface)) <= 3:
                pass
            else:
                bluseg_inner_contactsurface = np.zeros(bluseg_inner_contactsurface.shape)
            
            # When the contactsurface is small, sometimes the redseg_contactsurface
            # consists only of the redseg_contactcorners. In this case there will
            # be only 1 piece (the noncontactborders) when the contactcorners are
            # removed. Therefore add the redseg_contactcorners to each possible
            # surface:
            borders1 = (bluseg_inner_contactsurface == 1) + bluseg_inner_corners.astype(bool)
            borders2 = (bluseg_inner_contactsurface == 2) + bluseg_inner_corners.astype(bool)
            
            # Compare similarity between label 1 and label 2:
            #ssim_borders1 = structural_similarity((redseg_inner_contactsurface == 1).astype(bool), contactsurface[:,:,0].astype(bool))
            #ssim_borders2 = structural_similarity((redseg_inner_contactsurface == 2).astype(bool), contactsurface[:,:,0].astype(bool))
            ssim_borders1 = structural_similarity(borders1, contactsurface[:,:,channel2].astype(bool))
            ssim_borders2 = structural_similarity(borders2, contactsurface[:,:,channel2].astype(bool))
            # Classify the more similar label as being contactsurface,
            # put the noncontactborders in a new array and remove the 
            # noncontactborders from the contactsurface array:
            if ssim_borders2 > ssim_borders1:
                bluseg_inner_noncontactborders[np.where(borders2)] = False
                bluseg_inner_noncontactborders = bluseg_inner_noncontactborders.astype(bool)
                bluseg_inner_contactsurface[np.where(borders1)] = False
                bluseg_inner_contactsurface = bluseg_inner_contactsurface.astype(bool)
                bluseg_inner_contactsurface[np.where(bluseg_inner_corners)] = True
            elif ssim_borders1 > ssim_borders2:
                bluseg_inner_noncontactborders[np.where(borders1)] = False
                bluseg_inner_noncontactborders = bluseg_inner_noncontactborders.astype(bool)
                bluseg_inner_contactsurface[np.where(borders2)] = False
                bluseg_inner_contactsurface = bluseg_inner_contactsurface.astype(bool)
                bluseg_inner_contactsurface[np.where(bluseg_inner_corners)] = True
            else:
                bluseg_inner_contactsurface = np.zeros(bluseg_inner_contactsurface.shape)
                bluseg_inner_noncontactborders = np.zeros(bluseg_inner_contactsurface.shape)
            
            # Do the same for the outer borders:
            bluseg_outer_contactsurface = np.copy(bluseg_border_clean_outer)
            bluseg_outer_noncontactborders = np.copy(bluseg_border_clean_outer)
            
            bluseg_outer_contactsurface[np.where(bluseg_outer_corners)] = False
            bluseg_outer_noncontactborders[np.where(bluseg_outer_corners)] = False

            bluseg_outer_contactsurface = morphology.label(bluseg_outer_contactsurface, connectivity=2)
            if len(np.unique(bluseg_outer_contactsurface)) <= 3:
                pass
            else:
                bluseg_outer_contactsurface = np.zeros(bluseg_outer_contactsurface.shape)
            
            # When the contactsurface is small, sometimes the redseg_contactsurface
            # consists only of the redseg_contactcorners. In this case there will
            # be only 1 piece (the noncontactborders) when the contactcorners are
            # removed. Therefore add the redseg_contactcorners to each possible
            # surface:
            borders1 = (bluseg_outer_contactsurface == 1) + bluseg_outer_corners.astype(bool)
            borders2 = (bluseg_outer_contactsurface == 2) + bluseg_outer_corners.astype(bool)


            # Compare similarity between label 1 and label 2:
            ssim_borders1 = structural_similarity(borders1, contactsurface[:,:,channel2].astype(bool))
            ssim_borders2 = structural_similarity(borders2.astype(bool), contactsurface[:,:,channel2].astype(bool))
            # Classify the more similar label as being contactsurface,
            # put the noncontactborders in a new array and remove the 
            # noncontactborders from the contactsurface array:
            if ssim_borders2 > ssim_borders1:
                bluseg_outer_noncontactborders[np.where(borders2)] = False
                bluseg_outer_noncontactborders = bluseg_outer_noncontactborders.astype(bool)
                bluseg_outer_contactsurface[np.where(borders1)] = False
                bluseg_outer_contactsurface = bluseg_outer_contactsurface.astype(bool)
                bluseg_outer_contactsurface[np.where(bluseg_outer_corners)] = True
            elif ssim_borders1 > ssim_borders2:
                bluseg_outer_noncontactborders[np.where(borders1)] = False
                bluseg_outer_noncontactborders = bluseg_outer_noncontactborders.astype(bool)
                bluseg_outer_contactsurface[np.where(borders2)] = False
                bluseg_outer_contactsurface = bluseg_outer_contactsurface.astype(bool)
                bluseg_outer_contactsurface[np.where(bluseg_outer_corners)] = True
            else:
                bluseg_outer_contactsurface = np.zeros(bluseg_outer_contactsurface.shape)
                bluseg_outer_noncontactborders = np.zeros(bluseg_outer_contactsurface.shape)

            # Combine the inner and outer contactsurface back together.
            # Also do the same for inner and outer noncontactborders.
            bluseg_contactsurface = np.logical_or(bluseg_inner_contactsurface, bluseg_outer_contactsurface)
            bluseg_noncontactborders = np.logical_or(bluseg_inner_noncontactborders, bluseg_outer_noncontactborders)

            # Restore any removed pixels from the clean_up_borders function to
            # either the contactsurface or noncontactborders that end up being
            # right in between the inner and outer borders:
            bluseg_contactsurface_filled = morphology.remove_small_holes(bluseg_contactsurface)
            bluseg_noncontactborders_filled = morphology.remove_small_holes(bluseg_noncontactborders)
            
            
            # Return pixels in sequential order for inner and outer contactsurface:
            bluseg_inner_contactsurface = np.dstack((bluseg_inner_contactsurface, np.zeros(bluseg_inner_contactsurface.shape), np.zeros(bluseg_inner_contactsurface.shape)))
            bluseg_inner_contactsurface_seq, blu_unsused, grn_unused_perim, blu_unused_perim = make_open_border_indices_sequential(bluseg_inner_contactsurface, red_id, blu_id, 0, 2) #grn_id
            bluseg_outer_contactsurface = np.dstack((bluseg_outer_contactsurface, np.zeros(bluseg_outer_contactsurface.shape), np.zeros(bluseg_outer_contactsurface.shape)))
            bluseg_outer_contactsurface_seq, blu_unused, grn_unused_perim, blue_unused_perim = make_open_border_indices_sequential(bluseg_outer_contactsurface, red_id, blu_id, 0, 2) #grn_id
            
            # Also do same for inner and outer noncontactborders:
            bluseg_inner_noncontactborders = np.dstack((bluseg_inner_noncontactborders, np.zeros(bluseg_inner_noncontactborders.shape), np.zeros(bluseg_inner_noncontactborders.shape)))
            bluseg_inner_noncontactborders_seq, blu_unused, grn_unused_perim, blu_unused_perim = make_open_border_indices_sequential(bluseg_inner_noncontactborders, red_id, blu_id, 0, 2) #grn_id
            bluseg_outer_noncontactborders = np.dstack((bluseg_outer_noncontactborders, np.zeros(bluseg_outer_noncontactborders.shape), np.zeros(bluseg_outer_noncontactborders.shape)))
            bluseg_outer_noncontactborders_seq, blu_unused, grn_unused_perim, blu_unused_perim = make_open_border_indices_sequential(bluseg_outer_noncontactborders, red_id, blu_id, 0, 2) #grn_id
 
            
            
            
            # Return fluorescence in red and grn channel sequentially:
            redchan_fluor_at_contact_inner = redchan[tuple(np.asarray(a) for a in zip(*redseg_inner_contactsurface_seq))[:2]]
            redchan_fluor_at_contact_outer = redchan[tuple(np.asarray(a) for a in zip(*redseg_outer_contactsurface_seq))[:2]]
            grnchan_fluor_at_contact_inner = grnchan[tuple(np.asarray(a) for a in zip(*grnseg_inner_contactsurface_seq))[:2]]
            grnchan_fluor_at_contact_outer = grnchan[tuple(np.asarray(a) for a in zip(*grnseg_outer_contactsurface_seq))[:2]]
            bluchan_fluor_at_contact_inner = bluchan[tuple(np.asarray(a) for a in zip(*bluseg_inner_contactsurface_seq))[:2]]
            bluchan_fluor_at_contact_outer = bluchan[tuple(np.asarray(a) for a in zip(*bluseg_outer_contactsurface_seq))[:2]]

            redchan_fluor_outside_contact_inner = redchan[tuple(np.asarray(a) for a in zip(*redseg_inner_noncontactborders_seq))[:2]]
            redchan_fluor_outside_contact_outer = redchan[tuple(np.asarray(a) for a in zip(*redseg_outer_noncontactborders_seq))[:2]]
            grnchan_fluor_outside_contact_inner = grnchan[tuple(np.asarray(a) for a in zip(*grnseg_inner_noncontactborders_seq))[:2]]
            grnchan_fluor_outside_contact_outer = grnchan[tuple(np.asarray(a) for a in zip(*grnseg_outer_noncontactborders_seq))[:2]]
            bluchan_fluor_outside_contact_inner = bluchan[tuple(np.asarray(a) for a in zip(*bluseg_inner_noncontactborders_seq))[:2]]
            bluchan_fluor_outside_contact_outer = bluchan[tuple(np.asarray(a) for a in zip(*bluseg_outer_noncontactborders_seq))[:2]]

            
            
            # Sequentially ordered borders miss pixels between the inner and 
            # outer borders, so also take the unordered borders which have
            # those pixels:
            redchan_fluor_at_contact = redchan[np.where(redseg_contactsurface_filled)]
            grnchan_fluor_at_contact = grnchan[np.where(grnseg_contactsurface_filled)]
            bluchan_fluor_at_contact = bluchan[np.where(bluseg_contactsurface_filled)]

            redchan_fluor_outside_contact = redchan[np.where(redseg_noncontactborders_filled)]
            grnchan_fluor_outside_contact = grnchan[np.where(grnseg_noncontactborders_filled)]
            bluchan_fluor_outside_contact = bluchan[np.where(bluseg_noncontactborders_filled)]

            
            # Also get un-ordered cytoplasmic fluorescence values:
            redseg_cytopl_area = np.logical_xor(redseg_border_clean_inner, scp_morph.binary_fill_holes(redseg_border_clean_inner))
            grnseg_cytopl_area = np.logical_xor(grnseg_border_clean_inner, scp_morph.binary_fill_holes(grnseg_border_clean_inner))
            bluseg_cytopl_area = np.logical_xor(bluseg_border_clean_inner, scp_morph.binary_fill_holes(bluseg_border_clean_inner))

            redchan_fluor_cytoplasmic = redchan[redseg_cytopl_area]
            grnchan_fluor_cytoplasmic = grnchan[grnseg_cytopl_area]
            bluchan_fluor_cytoplasmic = bluchan[bluseg_cytopl_area]

            # Sample the intensity values at the contactsurface and at the
            # noncontactborders:
            #plt.hist(redchan[np.where(redseg_contactsurface_filled)], bins=redchan[np.where(redseg_contactsurface_filled)].max(), color='r', alpha=0.5)
            #plt.hist(redchan[np.where(redseg_noncontactborders_filled)], bins=redchan[np.where(redseg_noncontactborders_filled)].max(), color='r', alpha=0.5)
            #print( np.mean(redchan[np.where(redseg_contactsurface_filled)]) )
            #print( np.mean(redchan[np.where(redseg_noncontactborders_filled)]) )


            
            ### Now also sample the green channel:
            #plt.hist(grnchan[np.where(redseg_contactsurface_filled)], bins=grnchan[np.where(redseg_contactsurface_filled)].max(), color='g', alpha=0.5)
            #plt.hist(grnchan[np.where(redseg_noncontactborders_filled)], bins=grnchan[np.where(redseg_noncontactborders_filled)].max(), color='g', alpha=0.5)
            #print( np.mean(grnchan[np.where(redseg_contactsurface_filled)]) )
            #print( np.mean(grnchan[np.where(redseg_noncontactborders_filled)]) )

            # The fluorescence values of the contactsurface and noncontactsurface
            # for the red and green channels should be placed in their own
            # columns in the data table at the end (4 columns total):
            
            
            # Also you can just get rid of the user input prompt for the minimum
            # background threshold and objects of interest




            ### Re-draw the figure with the contact angles, contact surface, etc.

            ### Draw the angles over top of the clean overlay in a figure:
            ### need to collect and organize all the corresponding vector start x values,
            ### y values, and vector x values, y values for usage in ax.quiver:
            vector_start_x_vals = (vector1_start[0], vector2_start[0], vector3_start[0], vector4_start[0])
            vector_start_y_vals = (vector1_start[1], vector2_start[1], vector3_start[1], vector4_start[1])
            vector_x_vals = (vector1[0], vector2[0], vector3[0], vector4[0])
            vector_y_vals = (vector1[1], vector2[1], vector3[1], vector4[1])
            
            ang1_annot = ((vector1 + vector2)/2.0)        
            ang1_annot_mag = np.sqrt(ang1_annot.dot(ang1_annot))
            if ang1_annot_mag == 0:
                ang1_annot_mag = 10**(-12)
    
            ang2_annot = ((vector3 + vector4)/2.0)
            ang2_annot_mag = np.sqrt(ang2_annot.dot(ang2_annot))
            if ang2_annot_mag == 0:
                ang2_annot_mag = 10**(-12)
    
            ### Draw a line between the contact corners of the red cell:
            contact_surface_line_red_rcd = [(vector1_start[1], vector3_start[1]), (vector1_start[0], vector3_start[0])]
            ### Draw a line between the contact corners of the green cell:
            contact_surface_line_blu_rcd = [(vector2_start[1], vector4_start[1]), (vector2_start[0], vector4_start[0])]
            ### draw a line that is perpendicular to the contact surface line, in the middle
            ### of the contact surface:
            temp1_rcd = (vector1_start[::-1] + vector2_start[::-1])/2.0
            temp2_rcd = (vector3_start[::-1] + vector4_start[::-1])/2.0
            mean_contact_surface_line_rcd = (temp1_rcd + temp2_rcd)/2.0
            center_of_contact_surface_lines_rc = mean_contact_surface_line_rcd[:2]

            ### Create an overlay of the original image and the borders for
            ### checking during coding:
            cleanoverlay = mh.overlay(tlchan, red=cleanborders[:,:,0], green=cleanborders[:,:,1], blue=cleanborders[:,:,2])

            ### Draw what you've found over the images of the cells:
            fig.clf()
            fig = plt.figure(num=str(filenumber+1)+'_'+os.path.basename(filename), figsize=(16., 10.))
            plt.suptitle(str(filenumber+1)+' - '+os.path.basename(filename), verticalalignment='center', x=0.5, y=0.5, fontsize=14)
            ax1 = fig.add_subplot(241)
            ax1.yaxis.set_visible(False)
            ax1.xaxis.set_visible(False)
            ax1.imshow(tlchan, cmap='gray')
            
            ax2 = fig.add_subplot(242)
            ax2.yaxis.set_visible(False)
            ax2.xaxis.set_visible(False)
            #figcellborderred = ax2.imshow(mh.overlay(gray = mh.stretch(tlchan), red = cleanborders[:,:,0]))
            fig_noncont_redcell = fig_noncont_redcell = ax2.imshow(mh.overlay(gray = mh.stretch(redchan), red = redseg_noncontactborders))
            ax2.axes.autoscale(False)
            #ax2.scatter(cell_max_rcd[0][1], cell_max_rcd[0][0], color='m', marker='o')
            
            ax3 = fig.add_subplot(243)
            ax3.yaxis.set_visible(False)
            ax3.xaxis.set_visible(False)
            #figcellborderblu = ax3.imshow(mh.overlay(gray = mh.stretch(tlchan), blue = cleanborders[:,:,2]))
            fig_noncont_grncell = ax3.imshow(mh.overlay(gray = mh.stretch(grnchan), green = grnseg_noncontactborders))
            ax3.axes.autoscale(False)
            #ax3.scatter(cell_max_rcd[1][1], cell_max_rcd[1][0], color='m', marker='o')
            
            ax4 = fig.add_subplot(246)
            ax4.yaxis.set_visible(False)
            ax4.xaxis.set_visible(False)
            #ax4.imshow(tlchan, cmap='gray')
            fig_cont_redcell = ax4.imshow(mh.overlay(gray = mh.stretch(redchan), red = redseg_contactsurface))
            ax4.axes.autoscale(False)
            #drawn_arrows = ax4.quiver(vector_start_x_vals, vector_start_y_vals, vector_x_vals, vector_y_vals, units='xy',scale_units='xy', angles='xy', scale=0.33, color='c', width = 1.0)
            #angle1_annotations = ax4.annotate('%.1f' % degree_angle1, vector1_start, vector1_start + 30*(ang1_annot / ang1_annot_mag)*(1,1), color='c', fontsize='24', ha='center', va='center')
            #angle2_annotations = ax4.annotate('%.1f' % degree_angle2, vector3_start, vector3_start + 30*(ang2_annot / ang2_annot_mag)*(1,1), color='c', fontsize='24', ha='center', va='center')
            #redcell_contactvect = ax4.plot(contact_surface_line_red_rcd[1], contact_surface_line_red_rcd[0], color='c', lw=1)
            #grncell_contactvect = ax4.plot(contact_surface_line_blu_rcd[1], contact_surface_line_blu_rcd[0], color='c', lw=1)
            #contactvect = ax4.plot([mean_contact_surface_line_rcd[1]-2*contact_vector[0], mean_contact_surface_line_rcd[1]+2*contact_vector[0]], [mean_contact_surface_line_rcd[0]-2*contact_vector[1], mean_contact_surface_line_rcd[0]+2*contact_vector[1]], color='m')
            
            ax5 = fig.add_subplot(247)
            ax5.yaxis.set_visible(False)
            ax5.xaxis.set_visible(False)
            #ax5.imshow(mh.overlay(gray = mh.stretch(tlchan), red=contactsurface[:,:,0], blue=contactsurface[:,:,2]))
            fig_cont_grncell = ax5.imshow(mh.overlay(gray = mh.stretch(grnchan), green = grnseg_contactsurface))
            ax5.axes.autoscale(False)
    
            ax6 = fig.add_subplot(245)
            ax6.yaxis.set_visible(False)
            ax6.xaxis.set_visible(False)
            ax6.imshow(cleanoverlay)
            ax6.axes.autoscale(False)
            ax6.scatter(cell_max_rcd[1][1], cell_max_rcd[1][0], color='m', marker='o')
            ax6.scatter(cell_max_rcd[0][1], cell_max_rcd[0][0], color='m', marker='o')
            ax6.scatter(bg_max_rcd[0][1], bg_max_rcd[0][0], color='m', marker='x')
            ax6.scatter(bg_max_rcd[1][1], bg_max_rcd[1][0], color='m', marker='x')
            drawn_arrows = ax6.quiver(vector_start_x_vals, vector_start_y_vals, vector_x_vals, vector_y_vals, units='xy',scale_units='xy', angles='xy', scale=0.33, color='c', width = 1.0)
            angle1_annotations = ax6.annotate('%.1f' % degree_angle1, vector1_start, vector1_start + 30*(ang1_annot / ang1_annot_mag)*(1,1), color='c', fontsize='24', ha='center', va='center')
            angle2_annotations = ax6.annotate('%.1f' % degree_angle2, vector3_start, vector3_start + 30*(ang2_annot / ang2_annot_mag)*(1,1), color='c', fontsize='24', ha='center', va='center')
            redcell_contactvect = ax6.plot(contact_surface_line_red_rcd[1], contact_surface_line_red_rcd[0], color='c', lw=1)
            grncell_contactvect = ax6.plot(contact_surface_line_blu_rcd[1], contact_surface_line_blu_rcd[0], color='c', lw=1)
            contactvect = ax6.plot([mean_contact_surface_line_rcd[1]-2*contact_vector[0], mean_contact_surface_line_rcd[1]+2*contact_vector[0]], [mean_contact_surface_line_rcd[0]-2*contact_vector[1], mean_contact_surface_line_rcd[0]+2*contact_vector[1]], color='m')

            ax7 = fig.add_subplot(244)
            ax7.yaxis.set_visible(False)
            ax7.xaxis.set_visible(False)
            ax_noncont_blucell = ax7.imshow(mh.overlay(mh.stretch(bluchan), blue=bluseg_noncontactborders))
            ax7.axes.autoscale(False)
            
            ax8 = fig.add_subplot(248)
            ax8.yaxis.set_visible(False)
            ax8.xaxis.set_visible(False)
            fig_cont_blucell = ax8.imshow(mh.overlay(mh.stretch(bluchan), blue=bluseg_contactsurface))
            ax8.axes.autoscale(False)


            plt.tight_layout()
            plt.draw
            #plt.show('block' == False)
    
            listupdate()



    elif filenumber > 0: 
        start_time = time.perf_counter()

        messages = list()
        filename = filenames_tl[filenumber]
        # Open the the image from each channel. PIL will close them when no
        # longer needed:
        redchan_raw = np.array(Image.open(filenames_red[filenumber]))
        grnchan_raw = np.array(Image.open(filenames_grn[filenumber]))
        bluchan_raw = np.array(Image.open(filenames_blu[filenumber]))
        tlchan_raw = np.array(Image.open(filenames_tl[filenumber]))
        
        ### Check the signal to noise ratio for each channel:
        # If the signal in a channel has a standard deviation less than 1 in any 
        # channel, skip it because the signal to noise ratio won't be good enough
        # for proper edge detection and segmentation:
        redchan = check_for_signal(redchan_raw)
        grnchan = check_for_signal(grnchan_raw)
        bluchan = check_for_signal(bluchan_raw)
        #bluchan = np.copy(bluchan_raw)
        tlchan = check_for_signal(tlchan_raw)
        
        fluor_chans = np.dstack((redchan, grnchan, bluchan))
        channel1 = 0
        channel2 = 2


        
        odr_circle_fit_iterations = 30
        num_bins = 8
        skip = 'n'


        gaussian_filter_value_red = 2.0
        gaussian_filter_value_grn = 2.0
        gaussian_filter_value_blu = 4.0
        gaussian_filter_value_tl = disk_radius/2.

###
        ### This is a threshold that determines how dim something in the image
        ### has to be to be considered background, it's set to 10% of the 
        ### images dtype max:
        #background_threshold = np.iinfo(tlchan.dtype).max * 0.1#0.05

###
        ### The size_filter determines how many px^2 the area of a region has
        ### to be in order to be recognized as a cell:
        size_filter = (np.pi * disk_radius**2) * 0.2 #500#20 #the 20 filter is used for the 2 circles simulation

        #particle_filter = 250#20 #the 20 filter is used for the 2 circle simulation

###
        ### Create a structuring element to try and find seed points for cells:
        maxima_Bc = morphology.disk(disk_radius) #15.5 = diameter of 30

        # Define a 3 dimensional structuring element for finding neighbours
        # between channels of RGB images:
        Bc3D = np.ones((3,3,5))



    ### Define the default values for the the data that will populate the table
    ### that will be output at the end:
        #red_gap_scanmean = np.asarray([float('nan')]*linescan_length)
        #grn_gap_scanmean = np.asarray([float('nan')]*linescan_length)
        #red_gap_scanx = [float('nan')]
        #red_gap_scany = [float('nan')]
        #grn_gap_scanx = [float('nan')]
        #grn_gap_scany = [float('nan')]
        degree_angle1 = float('nan')
        degree_angle2 = float('nan')
        average_angle = float('nan')
        contactsurface_len = float('nan')
        noncontactsurface_len1 = float('nan')
        noncontactsurface_len2 = float('nan')
        
        redchan_fluor_at_contact_inner = np.array([float('nan')])
        grnchan_fluor_at_contact_inner = np.array([float('nan')])
        bluchan_fluor_at_contact_inner = np.array([float('nan')])
        redchan_fluor_at_contact_outer = np.array([float('nan')])
        grnchan_fluor_at_contact_outer = np.array([float('nan')])
        bluchan_fluor_at_contact_outer = np.array([float('nan')])
        redchan_fluor_outside_contact_inner = np.array([float('nan')])
        grnchan_fluor_outside_contact_inner = np.array([float('nan')])
        bluchan_fluor_outside_contact_inner = np.array([float('nan')])
        redchan_fluor_outside_contact_outer = np.array([float('nan')])
        grnchan_fluor_outside_contact_outer = np.array([float('nan')])
        bluchan_fluor_outside_contact_outer = np.array([float('nan')])
        
        redchan_fluor_at_contact = np.array([float('nan')])
        grnchan_fluor_at_contact = np.array([float('nan')])
        bluchan_fluor_at_contact = np.array([float('nan')])
        redchan_fluor_outside_contact = np.array([float('nan')])
        grnchan_fluor_outside_contact = np.array([float('nan')])
        bluchan_fluor_outside_contact = np.array([float('nan')])
        
        redchan_fluor_cytoplasmic = np.array([float('nan')])
        grnchan_fluor_cytoplasmic = np.array([float('nan')])
        bluchan_fluor_cytoplasmic = np.array([float('nan')])

        area_cell_1 = float('nan')
        area_cell_2 = float('nan')
        perimeter_cell1 = float('nan')
        perimeter_cell2 = float('nan')
        noncontact_perimeter_cell1 = float('nan')
        noncontact_perimeter_cell2 = float('nan')
        cleanborders = np.zeros(fluor_chans.shape)
        contactsurface = np.zeros(fluor_chans.shape)
        noncontactborders = np.zeros(fluor_chans.shape)
        pathfinder = np.zeros(fluor_chans.shape)
        vector1_start = np.array([float('nan'), float('nan')])
        vector2_start = np.array([float('nan'), float('nan')])
        vector3_start = np.array([float('nan'), float('nan')])
        vector4_start = np.array([float('nan'), float('nan')])
        vector1 = np.array([float('nan'), float('nan')])
        vector2 = np.array([float('nan'), float('nan')])
        vector3 = np.array([float('nan'), float('nan')])
        vector4 = np.array([float('nan'), float('nan')])
        cleanspots = np.array([float('nan'), float('nan')])
        contact_vector = np.array([float('nan'), float('nan')])
        perpendicular2contact_vector = np.array([float('nan'), float('nan')])
        red_halfangle1_rad = float('nan')
        red_halfangle2_rad = float('nan')
        blu_halfangle1_rad = float('nan')
        blu_halfangle2_rad = float('nan')
        red_halfangle1 = float('nan')
        red_halfangle2 = float('nan')
        blu_halfangle1 = float('nan')
        blu_halfangle2 = float('nan')
        contactsurface_len1 = float('nan')
        contactsurface_len2 = float('nan')
        numbered_border_indices_red = [float('nan'), (float('nan'),float('nan'),float('nan'))]
        numbered_border_indices_blu = [float('nan'), (float('nan'),float('nan'),float('nan'))]
        numbered_noncontactborder_indices_red = [float('nan'), (float('nan'),float('nan'),float('nan'))]
        numbered_noncontactborder_indices_blu = [float('nan'), (float('nan'),float('nan'),float('nan'))]
        
        angles_angle_labeled = np.zeros(fluor_chans.shape)
        
        bin_curvature_details_red = [[float('nan')] * len(bin_curvature_details_headers)] * (num_bins+1)
        bin_curvature_details_blu = [[float('nan')] * len(bin_curvature_details_headers)] * (num_bins+1)
        
        whole_curvature_details_red = [[float('nan')] * len(bin_curvature_details_headers)] * 2
        whole_curvature_details_blu = [[float('nan')] * len(bin_curvature_details_headers)] * 2
        
        bin_indices_red = [[(float('nan'), float('nan'), float('nan'))]] * num_bins
        bin_indices_blu = [[(float('nan'), float('nan'), float('nan'))]] * num_bins
        
#        bin_start_red = [float('nan')] * num_bins
#        bin_stop_red = [float('nan')] * num_bins
        
#        bin_start_grn = [float('nan')] * num_bins
#        bin_stop_grn = [float('nan')] * num_bins

        bin_start_red_indices = np.empty((num_bins, 3))
        bin_start_red_indices[:] = np.nan
        
        bin_stop_red_indices = np.empty((num_bins, 3))
        bin_stop_red_indices[:] = np.nan

        bin_start_blu_indices = np.empty((num_bins, 3))
        bin_start_blu_indices[:] = np.nan

        bin_stop_blu_indices = np.empty((num_bins, 3))
        bin_stop_blu_indices[:] = np.nan



###
        ### Segment your cells:
        
        # Start by getting some seed points for a watershed:
        spots_all, spots_red, spots_blu, bg_max, redcell, blucell, cell_thresh, bg_thresh = find_cell_seeds(tlchan, redchan, grnchan, bluchan, maxima_Bc)


        ### The re-segmentation of the red channel to capture the membrane
        ### label should go here. But it doesn't work very well if the signal
        ### is really week (eg. at cell-cell contact of a cellpair).
        # redseg = segmentation.watershed(redchan, spots_all, compactness=0.1)

                
        # Remove any areas that are too big for the red segmentation:
        red_cell_props = measure.regionprops(redcell)
        size_red = np.array([x.area for x in red_cell_props])
        labels_red = np.array([x.label for x in red_cell_props])
        too_big = size_red > (2 * np.pi * disk_radius**2)
        too_big_labels = labels_red[too_big]
        redcell_clean = mh.labeled.remove_regions(redcell, too_big_labels)
        
        
        
        ### Sometimes the cortical actin decouples from the plasma membrane 
        ### enough that the borders of the redcell don't line up with those of
        ### the actin in the grn channel. To try and fix this we're going to 
        ### find the red_ridges and grn_ridges, clean them up, and use them to 
        ### help us get a segmentation of the cell in the grn channel 
        ### (ie. LifeAct-GFP) such that the borders of that segmentation line 
        ### up nicely with the cortical actin:
        
        # Find the red ridges:
        red_ridges = find_ridge_peaks(redchan, axis0_peak_to_peak_dist=5, axis1_peak_to_peak_dist=5)
        #red_ridges = morphology.binary_closing(red_ridges, selem=morphology.disk(1))
        
        # Clean up the red ridges:
        okay_small_objects = morphology.remove_small_objects(np.logical_or(red_ridges, mh.borders(redcell_clean)))
        bad_small_objects = np.logical_xor(okay_small_objects, red_ridges)
        ridges_and_borders = np.logical_and(bad_small_objects, mh.borders(redcell_clean))
        red_ridges_clean = np.logical_xor(bad_small_objects, red_ridges)
        red_ridges_clean = np.logical_xor(red_ridges_clean, ridges_and_borders)
        
        # Find the green ridges:
        grn_ridges = find_ridge_peaks(grnchan, axis0_peak_to_peak_dist=5, axis1_peak_to_peak_dist=5)
        #grn_ridges = morphology.binary_closing(grn_ridges, selem=morphology.disk(1))
        
        # Clean up the green ridges
        okay_small_objects = morphology.remove_small_objects(np.logical_or(grn_ridges, mh.borders(redcell_clean)))
        bad_small_objects = np.logical_xor(okay_small_objects, grn_ridges)
        ridges_and_borders = np.logical_and(bad_small_objects, mh.borders(redcell_clean))
        grn_ridges_clean = np.logical_xor(bad_small_objects, grn_ridges)
        grn_ridges_clean = np.logical_xor(grn_ridges_clean, ridges_and_borders)
        
        # Use the grn_ridges_clean to assist segmentation of the actin:
        iterations = 100
        grnseg, growing_markers_grn, growing_area_grn, growing_labels_grn, first_indices_grn = iter_seg(filters.gaussian(segmentation.find_boundaries(redcell_clean) + grn_ridges_clean, sigma=2), spots_red, np.add.reduce((spots_blu, bg_max)), iterations=iterations, mask=~(red_ridges_clean + grn_ridges_clean), compactness=0.1)
        
        # If you add the grn_ridges_clean to the grnseg then this works pretty well for
        # restricting growth of the seeds during watershed segmentation.
        # msg = 'grnseg iterations: %s'%iterations + ','
        # msg += ' first_indices_grn: %s'%first_indices_grn + ','
        # messages.append(msg)
        
        ### Now that we have the green/LifeAct segmentation we need to clean
        ### that up as well:
        
        # Remove any areas that are too big from the green segmentation:
        grn_cell_props = measure.regionprops(grnseg)
        size_grn = np.array([x.area for x in grn_cell_props])
        labels_grn  = np.array([x.label for x in grn_cell_props])
        too_big = size_grn > (2 * np.pi * disk_radius**2)
        too_big_labels = labels_grn[too_big]
        grnseg = mh.labeled.remove_regions(grnseg, too_big_labels)
        
        # Add the grn_ridges_clean to the grnseg:
        grnseg = np.logical_or(grnseg, grn_ridges_clean)
        grnseg = morphology.remove_small_holes(grnseg)
        
        # Remove small pieces that make grnseg rough from adding grn_ridges_clean:
        grnseg = morphology.label(grnseg, connectivity=1)
        grnseg = morphology.remove_small_objects(grnseg).astype(bool)
        
        # When adding the grn_ridges_clean we end up with a morphological dilation of 
        # the grnseg borders, but we want the grn_ridges_clean to correspond to the 
        # outer edges. Therefore to rectify this take the internal region of the grnseg 
        # borders as being grnseg:
        grnseg = np.logical_xor(mh.borders(grnseg)*grnseg, grnseg)
        
        # Restore proper labeling:
        grnseg = segmentation.watershed(redchan, spots_red, mask=grnseg)

###

        cell_areas_rough = np.dstack((redcell_clean, np.zeros(redcell_clean.shape, dtype=int), blucell))


        ### Clean up the segmented image of any regions (ie. "cells") that are too
        ### small or that are touching the borders:
        cell_areas = remove_bordering_and_small_regions(cell_areas_rough).astype(bool) * cell_areas_rough.astype(int)
    
        # Fill in any holes in each region that might be present:
        cell_areas[:,:,0] = np.add.reduce([scp_morph.binary_fill_holes(cell_areas[:,:,0] == x) * x for x in np.unique(cell_areas[:,:,0])[1:]], dtype=int)
        cell_areas[:,:,1] = np.add.reduce([scp_morph.binary_fill_holes(cell_areas[:,:,1] == x) * x for x in np.unique(cell_areas[:,:,1])[1:]], dtype=int)
        cell_areas[:,:,2] = np.add.reduce([scp_morph.binary_fill_holes(cell_areas[:,:,2] == x) * x for x in np.unique(cell_areas[:,:,2])[1:]], dtype=int)

###



        # Make a copy of the cell maxima that uses their current labeling scheme:
        #new_cell_maxima = cell_areas.astype(bool) * cell_areas
        new_cell_maxima = np.copy(cell_areas)
        # Take the geometric mean of each label:
        new_cell_maxima_locs = []
        new_cell_maxima_locs.extend([np.append((np.mean(np.where(new_cell_maxima[:,:,channel1] == i), axis=1)).astype(int), channel1) for i in np.unique(new_cell_maxima[:,:,channel1])[1:]])
        new_cell_maxima_locs.extend([np.append((np.mean(np.where(new_cell_maxima[:,:,channel2] == i), axis=1)).astype(int), channel2) for i in np.unique(new_cell_maxima[:,:,channel2])[1:]])
        # and assign that pixel as the new spot in a new array:
        new_cell_maxima_means = np.zeros(new_cell_maxima.shape, dtype=int)
        new_cell_maxima_means[tuple(np.asarray(a) for a in zip(*new_cell_maxima_locs))] = new_cell_maxima[tuple(np.asarray(a) for a in zip(*new_cell_maxima_locs))]


        ### If more than 1 cell is detected per channel then try to use the cell
        ### that is spatially closest to the previously detected one:
        # Check red channel:
        if len(np.unique(cell_areas[:,:,channel1])) > 2:
            # Find which spot is closer to the most recent spots detected:
            closest_cell_max_red = nearest_cell_maxima(red_maxima_coords, np.where(new_cell_maxima_means[:,:,channel1]))
            # If you find a spot that is closer (which you should)...
            if np.array(closest_cell_max_red).any():
                # Then find what the spots label is...
                val = new_cell_maxima_means[closest_cell_max_red[0], closest_cell_max_red[1], channel1]
                # and reset the red channel of new_wscan_based_maxima_means
                # so that only the closest spot is kept:
                new_cell_maxima_means[:,:,channel1] = np.zeros(new_cell_maxima_means[:,:,channel1].shape, dtype=int)
                new_cell_maxima_means[closest_cell_max_red[0], closest_cell_max_red[1], channel1] = val
                ### Then remove the regions that aren't away from the 
                # cell_areas:
                remove_region = [np.logical_not(i in np.unique(new_cell_maxima_means[:,:,channel1])) for i in np.unique(cell_areas[:,:,channel1])]
                cell_areas[:,:,channel1] = mh.labeled.remove_regions(cell_areas[:,:,channel1], np.unique(cell_areas[:,:,channel1])[np.where(remove_region)])
            else:
                sizes = mh.labeled.labeled_size(cell_areas[:,:,channel1])
                iterate = sizes
                n = 2
                secondlargestregion = nth_max(n, iterate)
                too_small = np.where(sizes < secondlargestregion)
                cell_areas[:,:,channel1] = mh.labeled.remove_regions(cell_areas[:,:,channel1], too_small)
            
        # Check blue channel:
        if len(np.unique(cell_areas[:,:,channel2])) > 2:
            closest_cell_max_blu = nearest_cell_maxima(blu_maxima_coords, np.where(new_cell_maxima_means[:,:,channel2]))
            if np.array(closest_cell_max_blu).any():
                val = new_cell_maxima_means[closest_cell_max_blu[0], closest_cell_max_blu[1], channel2]
                new_cell_maxima_means[:,:,channel2] = np.zeros(new_cell_maxima_means[:,:,channel2].shape, dtype=int)
                new_cell_maxima_means[closest_cell_max_blu[0], closest_cell_max_blu[1], channel2] = val            
                ### Remove the regions that were further away:
                remove_region = [np.logical_not(i in np.unique(new_cell_maxima_means[:,:,channel2])) for i in np.unique(cell_areas[:,:,channel2])]
                cell_areas[:,:,channel2] = mh.labeled.remove_regions(cell_areas[:,:,channel2], np.unique(cell_areas[:,:,channel2])[np.where(remove_region)])
            else:
                sizes = mh.labeled.labeled_size(cell_areas[:,:,channel2])
                iterate = sizes
                n = 2
                secondlargestregion = nth_max(n, iterate)
                too_small = np.where(sizes < secondlargestregion)
                cell_areas[:,:,channel2] = mh.labeled.remove_regions(cell_areas[:,:,channel2], too_small)


###

        ### Make sure that the cells meet the minimum size threshold:
        red_cell_props = measure.regionprops(cell_areas[:,:,0])
        grn_cell_props = measure.regionprops(cell_areas[:,:,1])
        blu_cell_props = measure.regionprops(cell_areas[:,:,2])
        
        size_red = np.array([x.area for x in red_cell_props])
        size_grn = np.array([x.area for x in grn_cell_props])
        size_blu = np.array([x.area for x in blu_cell_props])
        # size_red = mh.labeled.labeled_size(cell_areas[:,:,0])
        # size_grn = mh.labeled.labeled_size(cell_areas[:,:,1])
        # size_blu = mh.labeled.labeled_size(cell_areas[:,:,2])
        labels_red = np.array([x.label for x in red_cell_props])
        labels_grn = np.array([x.label for x in grn_cell_props])
        labels_blu = np.array([x.label for x in blu_cell_props])

        too_small_red = labels_red[size_red < size_filter]
        too_small_grn = labels_grn[size_grn < size_filter]
        too_small_blu = labels_blu[size_blu < size_filter]
        
        cell_areas[:,:,0] = mh.labeled.remove_regions(cell_areas[:,:,0], too_small_red)
        cell_areas[:,:,1] = mh.labeled.remove_regions(cell_areas[:,:,1], too_small_grn)
        cell_areas[:,:,2] = mh.labeled.remove_regions(cell_areas[:,:,2], too_small_blu)


        # Fill in any holes in each region that might be present:
        cell_areas[:,:,0] = np.add.reduce([scp_morph.binary_fill_holes(cell_areas[:,:,0] == x) * x for x in np.unique(cell_areas[:,:,0])[1:]])
        cell_areas[:,:,1] = np.add.reduce([scp_morph.binary_fill_holes(cell_areas[:,:,1] == x) * x for x in np.unique(cell_areas[:,:,1])[1:]])
        cell_areas[:,:,2] = np.add.reduce([scp_morph.binary_fill_holes(cell_areas[:,:,2] == x) * x for x in np.unique(cell_areas[:,:,2])[1:]])



        # Refine spots_red to be only your red cell of interest:
        spots_red_of_interest = spots_red.astype(bool) * cell_areas[:,:,0]
        # Also do this for the blue spots:
        spots_blu_of_interest = spots_blu.astype(bool) * cell_areas[:,:,2]


###
        ### Because we do measurements of contact surface between cells and
        ### measurements of LifeAct intensity using the borders we will 
        ### eventually need to make the borders of the cell_areas and these
        ### borders will need to be made sequential in order to count the
        ### surface length or do directed line scans. In order to do this
        ### we need to thin the borders so that when considering a connectivity
        ### of 2 (or 8 depending on your nomenclature of choice) each pixel in
        ### the border is only connected to 2 other pixels. This needs to be 
        ### true the whole way around the cell areas for measuring fluorescence
        ### profiles, but when measuring contact surface or contact angle, the
        ### corners of the contact must be left untouched (even if they are
        ### connected to 3 other pixels (eg. an "L" shape)), this is because
        ### the if these corners are removed, then you can accidentally "open
        ### up" the contact and make it smaller than it really is, sometimes
        ### this even leads to the program missing cells that are definitely
        ### in contact.
     #--#
     #--#
     #--#
     #--#
        ### Having said all that, we need to make 2 copies of cleaned up 
        ### cell_areas and take their borders -- cleanborders will be used for
        ### measuring contact diameter and contact angle, and redseg_clean
        ### and grnseg_clean will be used to measure fluroescence profiles:
        
        
        
        # # And use the labels of these spots to pick out your cells of interest
        # # from the redseg and the bluseg:
        # if np.unique(spots_red_of_interest).any():
        #     redseg_clean_rough = redseg == np.unique(spots_red_of_interest)[np.nonzero(np.unique(spots_red_of_interest))]
        # else:
        #     redseg_clean_rough = np.zeros(redseg.shape)
        # if np.any(redseg_clean_rough):
        #     pass
        # else:
        #     redseg_clean_rough = np.zeros(redseg.shape)

        
        
        # Also, buff out sharp corners from redseg_clean as these will prevent 
        # clean_up_borders from working properly on both the inner and outer 
        # edges:
        redseg_clean = morphology.binary_closing(cell_areas[:,:,0])

        # Buffing out sharp corners leads to unnatural roundness, which is seen
        # as single pixels in borders, but the problems regions show up as multiple 
        # connected pixels. Therefore label the pixels added from binary_closing
        # and remove any regions with a size of 1:
        polish_pixels = np.logical_xor(cell_areas[:,:,0], redseg_clean)
        polish_pixels = morphology.label(polish_pixels)
        polish_px_bad_areas = [x.area == 1 for x in measure.regionprops(polish_pixels)]
        polish_px_bad_labels = np.asarray(polish_px_bad_areas) * np.asarray([x.label for x in measure.regionprops(polish_pixels)])
        polish_px_bad_indices = np.nonzero(polish_px_bad_labels)
        polish_px_bad_labels = polish_px_bad_labels[polish_px_bad_indices]
        polish_px_good = mh.labeled.remove_regions(polish_pixels, polish_px_bad_labels)
        polish_px_bad = np.logical_xor(polish_pixels, polish_px_good)
        
        redseg_clean = redseg_clean * ~polish_px_bad

        # Make sure that the redseg_clean is not a donut, as the program can't
        # analyze donuts:
        redseg_clean = scp_morph.binary_fill_holes(redseg_clean)
        
        # Also, make sure to remove any redseg_clean regions that are touching
        # the borders, as the program can't analyze those cells either:
        redseg_clean = segmentation.clear_border(redseg_clean)

        # And remove any redseg_clean regions that are less than the 
        # size_filter:
        redseg_clean = morphology.remove_small_objects(redseg_clean, 
                                                       min_size=size_filter)


###
        
        # Use the spots_red_of_interest to pick out your cell of interest from
        # the green channel:
        if np.unique(spots_red_of_interest).any():
            grnseg_clean_rough = grnseg == np.unique(spots_red_of_interest)[np.nonzero(np.unique(spots_red_of_interest))]
        else:
            grnseg_clean_rough = np.zeros(grnseg.shape)
        if np.any(grnseg_clean_rough):
            pass
        else:
            grnseg_clean_rough = np.zeros(grnseg.shape)


        # Also, buff out sharp corners from grnseg_clean as these will prevent 
        # clean_up_borders from working properly on both the inner and outer 
        # edges:
        grnseg_clean = morphology.binary_closing(grnseg_clean_rough)

        # Buffing out sharp corners leads to unnatural roundness, which is seen
        # as single pixels, but the problems regions show up as multiple 
        # connected pixels. Therefore label the pixels added from binary_closing
        # and remove any regions with a size of 1:
        polish_pixels = np.logical_xor(grnseg_clean_rough, grnseg_clean)
        polish_pixels = morphology.label(polish_pixels)
        polish_px_bad_areas = [x.area == 1 for x in measure.regionprops(polish_pixels)]
        polish_px_bad_labels = np.asarray(polish_px_bad_areas) * np.asarray([x.label for x in measure.regionprops(polish_pixels)])
        polish_px_bad_indices = np.nonzero(polish_px_bad_labels)
        polish_px_bad_labels = polish_px_bad_labels[polish_px_bad_indices]
        polish_px_good = mh.labeled.remove_regions(polish_pixels, polish_px_bad_labels)
        polish_px_bad = np.logical_xor(polish_pixels, polish_px_good)
        
        grnseg_clean = grnseg_clean * ~polish_px_bad

        # Make sure that the grnseg_clean is not a donut, as the program can't
        # analyze donuts:
        grnseg_clean = scp_morph.binary_fill_holes(grnseg_clean)
        
        # Also, make sure to remove any grnseg_clean regions that are touching
        # the borders, as the program can't analyze those cells either:
        grnseg_clean = segmentation.clear_border(grnseg_clean)
        
        # And remove any grnseg_clean regions that are less than the 
        # size_filter:
        grnseg_clean = morphology.remove_small_objects(grnseg_clean, 
                                                       min_size=size_filter)



###


        # Also, buff out sharp corners from redseg_clean as these will prevent 
        # clean_up_borders from working properly on both the inner and outer 
        # edges:
        # redseg_clean = morphology.binary_closing(redseg_clean_rough)
        # bluseg_clean = morphology.binary_closing(cell_areas[:,:,channel2])
        # We are going to try padding the image with a 1px-wide border made up
        # of zeros because binary_closing connects regions to the edge if the
        # cell is almost touching the borders in the non-padded image (i.e. if
        # the cell is only 1px away from the edge), we will then trim padding
        bluseg_clean = morphology.binary_closing(pad_img_edges(cell_areas[:,:,channel2], 0, 1))
        bluseg_clean = bluseg_clean[1:-1, 1:-1]

        # Buffing out sharp corners leads to unnatural roundness, which is seen
        # as single pixels in borders, but the problems regions show up as multiple 
        # connected pixels. Therefore label the pixels added from binary_closing
        # and remove any regions with a size of 1:
        polish_pixels = np.logical_xor(cell_areas[:,:,channel2], bluseg_clean)
        polish_pixels = morphology.label(polish_pixels)
        polish_px_bad_areas = [x.area == 1 for x in measure.regionprops(polish_pixels)]
        polish_px_bad_labels = np.asarray(polish_px_bad_areas) * np.asarray([x.label for x in measure.regionprops(polish_pixels)])
        polish_px_bad_indices = np.nonzero(polish_px_bad_labels)
        polish_px_bad_labels = polish_px_bad_labels[polish_px_bad_indices]
        polish_px_good = mh.labeled.remove_regions(polish_pixels, polish_px_bad_labels)
        polish_px_bad = np.logical_xor(polish_pixels, polish_px_good)
        
        bluseg_clean = bluseg_clean * ~polish_px_bad

        # Make sure that the bluseg_clean is not a donut, as the program can't
        # analyze donuts:
        bluseg_clean = scp_morph.binary_fill_holes(bluseg_clean)
        
        # Also, make sure to remove any redseg_clean regions that are touching
        # the borders, as the program can't analyze those cells either:
        bluseg_clean = segmentation.clear_border(bluseg_clean)
        
        # And remove any redseg_clean regions that are less than the 
        # size_filter:
        bluseg_clean = morphology.remove_small_objects(bluseg_clean, 
                                                       min_size=size_filter)

###
        
        ### Use the refined cell segmentations to create the cell borders:
        new_area_red = cell_areas[:,:,0].astype(bool) * 1
        cell_borders_red = mh.borders(new_area_red.astype(bool)) * new_area_red
        cell_borders_red = mh.thin(cell_borders_red).astype(bool) * 1
        
        new_area_grn = cell_areas[:,:,1].astype(bool) * 2
        cell_borders_grn = mh.borders(new_area_grn.astype(bool)) * new_area_grn
        cell_borders_grn = mh.thin(cell_borders_grn).astype(bool) * 2
        
        new_area_blu = cell_areas[:,:,2].astype(bool) * 3
        cell_borders_blu = mh.borders(new_area_blu.astype(bool)) * new_area_blu
        cell_borders_blu = mh.thin(cell_borders_blu).astype(bool) * 3
        
        curveborders = np.dstack((cell_borders_red, cell_borders_grn, cell_borders_blu))
###


        ### Now that we have the borders of the cells, it's time to clean up 
        ### the borders so the algorithm used to count pixels and get 
        ### sequential info from the borders doesn't get lost:
        cleanborders = clean_up_borders(curveborders)
        
        ### Also use these cleaned up borders to define the cleaned cell areas:
        cleanareas = np.zeros(cleanborders.shape)
        cleanareas[:,:,0] = np.add.reduce([scp_morph.binary_fill_holes(cleanborders[:,:,0] == x) * x for x in np.unique(cleanborders[:,:,0])[1:]])
        cleanareas[:,:,1] = np.add.reduce([scp_morph.binary_fill_holes(cleanborders[:,:,1] == x) * x for x in np.unique(cleanborders[:,:,1])[1:]])
        cleanareas[:,:,2] = np.add.reduce([scp_morph.binary_fill_holes(cleanborders[:,:,2] == x) * x for x in np.unique(cleanborders[:,:,2])[1:]])

        ### Create an overlay of the original image and the borders for
        ### checking during coding:
        cleanoverlay = mh.overlay(tlchan, red=cleanborders[:,:,0], green=cleanborders[:,:,1], blue=cleanborders[:,:,2])
        
        cell_spots = spots_red_of_interest + spots_blu_of_interest
        cell_max_rcd = ([np.where(cell_spots.astype(bool))[0], np.where(cell_spots.astype(bool))[1]], [np.where(cell_spots.astype(bool))[0], np.where(cell_spots.astype(bool))[1]])
        bg_max_rcd = ([np.where(bg_max)[0], np.where(bg_max)[1]], [np.where(bg_max)[0], np.where(bg_max)[1]])



        ### Plot the cells and any parameters so far:
        fig = plt.figure(num=str(filenumber+1)+'_'+os.path.basename(filename), figsize=(16., 10.))
        plt.suptitle(str(filenumber+1)+' - '+os.path.basename(filename), verticalalignment='center', x=0.5, y=0.5, fontsize=14)
        ax1 = fig.add_subplot(241)
        ax1.yaxis.set_visible(False)
        ax1.xaxis.set_visible(False)
        ax1.imshow(tlchan, cmap='gray')
        
        ax2 = fig.add_subplot(242)
        ax2.yaxis.set_visible(False)
        ax2.xaxis.set_visible(False)
        #figcellborderred = ax2.imshow(mh.overlay(gray = mh.stretch(tlchan), red = cleanborders[:,:,0]))
        #figcellred = ax2.imshow(mh.overlay(gray = mh.stretch(redchan), red = cleanborders[:,:,0]))
        fig_noncont_redcell = ax2.imshow(mh.overlay(mh.stretch(redchan), red=mh.borders(redseg_clean)))
        #fig_noncont_redcell = ax2.imshow(mh.stretch(redchan), cmap='gray')
        ax2.axes.autoscale(False)
        
        ax3 = fig.add_subplot(243)
        ax3.yaxis.set_visible(False)
        ax3.xaxis.set_visible(False)
        #figcellborderblu = ax3.imshow(mh.overlay(gray = mh.stretch(tlchan), blue = cleanborders[:,:,2]))
        fig_noncont_grncell = ax3.imshow(mh.overlay(mh.stretch(grnchan), green=mh.borders(grnseg_clean)))
        #fig_noncont_grncell = ax3.imshow(mh.stretch(grnchan), cmap='gray')
        ax3.axes.autoscale(False)
        
        ax4 = fig.add_subplot(246)
        ax4.yaxis.set_visible(False)
        ax4.xaxis.set_visible(False)
        #ax4.imshow(tlchan, cmap='gray')
        fig_cont_redcell = ax4.imshow(mh.stretch(redchan), cmap='gray')
        ax4.axes.autoscale(False)
        
        ax5 = fig.add_subplot(247)
        ax5.yaxis.set_visible(False)
        ax5.xaxis.set_visible(False)
        #ax5.imshow(tlchan, cmap='gray')
        fig_cont_grncell = ax5.imshow(mh.stretch(grnchan), cmap='gray')
        ax5.axes.autoscale(False)

        ax6 = fig.add_subplot(245)
        ax6.yaxis.set_visible(False)
        ax6.xaxis.set_visible(False)
        ax6.imshow(mh.overlay(gray = mh.stretch(tlchan), red = cleanborders[:,:,0], blue=cleanborders[:,:,2]))
        ax6.axes.autoscale(False)
        ax6.scatter(cell_max_rcd[0][1], cell_max_rcd[0][0], color='m', marker='o')
        ax6.scatter(cell_max_rcd[1][1], cell_max_rcd[1][0], color='m', marker='o')
        ax6.scatter(bg_max_rcd[0][1], bg_max_rcd[0][0], color='m', marker='x')
        ax6.scatter(bg_max_rcd[1][1], bg_max_rcd[1][0], color='m', marker='x')

        ax7 = fig.add_subplot(244)
        ax7.yaxis.set_visible(False)
        ax7.xaxis.set_visible(False)
        # ax_noncont_blucell = ax7.imshow(mh.overlay(mh.stretch(bluchan), blue=cleanborders[:,:,2]))
        ax_noncont_blucell = ax7.imshow(mh.overlay(mh.stretch(bluchan), blue=mh.borders(bluseg_clean)))
        ax7.axes.autoscale(False)
        
        ax8 = fig.add_subplot(248)
        ax8.yaxis.set_visible(False)
        ax8.xaxis.set_visible(False)
        fig_cont_blucell = ax8.imshow(mh.stretch(bluchan), cmap='gray')
        ax8.axes.autoscale(False)


        plt.tight_layout()
        plt.draw()



        ### Tell me the filenumber and filename of what you're working on:
        print('\n',(filenumber+1), filename)

        ### Skip the file if there is only 1 cell in the picture (sometimes the
        ### microscope skips a channel):
#        if len(mh.labeled.labeled_size(cleanareas)) < 3:
        if len(np.unique(cleanborders)) < 3:
            msg = "There aren't 2 cells in this image."
            print(msg)
            messages.append(msg)
            # skip = 'y'
            
        if np.any(redseg_clean):
            pass
        else:
            msg = "Watershed in red channel failed."
            print(msg)
            messages.append(msg)
            skip = 'y'
            
        if np.any(grnseg_clean):
            pass
        else:
            msg = "Watershed in green channel failed."
            print(msg)
            messages.append(msg)
            skip = 'y'

        ### Create an initial guess at the borders of the cells that are in
        ### contact with one another:
        rawcontactsurface = np.zeros(cleanareas.shape, np.dtype(np.int32))
        bg_id, red_id, grn_id, blu_id = 0, 1, 2, 3 #These must correspond to the channel index + 1
        # as a result of this, you can only have 1 cell of interest per channel.
        for (i,j,k), value in np.ndenumerate(cleanborders):
            if (bg_id in cleanborders[sub(i,1):i+2, sub(j,1):j+2, :]) and (red_id in cleanborders[sub(i,1):i+2, sub(j,1):j+2, :]) and (blu_id in cleanborders[sub(i,1):i+2, sub(j,1):j+2, :]):
                rawcontactsurface[i,j,k] = cleanborders[i,j,k]

        ### If this initial guess has too few pixels in contact then consider
        ### the cells to not be in contact:
        if np.count_nonzero(rawcontactsurface) <= 6:
            print('These cells are not in contact.')
            messages.append('These cells are not in contact.')
            cells_touching = False
        elif np.count_nonzero(rawcontactsurface) > 6:
            cells_touching = True

        ### If there aren't 2 cells in the image just skip it.
        ### If either redseg_clean or grnseg_clean failed then skip the image:
        if skip == 'y':
            listupdate()
            continue



        ### If the cells are not in contact then set a bunch of values that will
        ### be in the table to be null and proceed to bin and estimate their 
        ### curvatures:
        if cells_touching == False and skip == 'n':

            ### Give the pixels of the cell borders sequential positional info
            ### starting from an arbitrary starting point:
            ordered_border_indices_red, ordered_border_indices_blu, numbered_border_indices_red, numbered_border_indices_blu, perimeter_cell1, perimeter_cell2 = count_pixels_around_borders_sequentially(cleanborders, red_id, blu_id, 0, 2)

            ### For 2 cells that are not in contact:
            ### Need to start by calculating smallest distance between the red and green borders
            if cleanborders[:,:,0].any() and cleanborders[:,:,2].any():
                closest_redborder_indices_rc, closest_bluborder_indices_rc, mean_closest_indice_loc_red, mean_closest_indice_loc_blu = find_closest_pixels_indices_between_cell_borders(cleanborders[:,:,0], cleanborders[:,:,2], ordered_border_indices_red, ordered_border_indices_blu, mean_closest_indice_loc_red_old, mean_closest_indice_loc_blu_old)
            else:
                closest_redborder_indices_rc, closest_bluborder_indices_rc, mean_closest_indice_loc_red, mean_closest_indice_loc_blu = np.array([ordered_border_indices_red[0][:2]]), np.array([ordered_border_indices_blu[0][:2]]), np.array([ordered_border_indices_red[0][:2]]), np.array([ordered_border_indices_blu[0][:2]])                
            mean_closest_indice_loc_red_old = np.copy(mean_closest_indice_loc_red)
            mean_closest_indice_loc_blu_old = np.copy(mean_closest_indice_loc_blu)

            if cleanborders[:,:,0].any() and cleanborders[:,:,2].any():
                ### Find the bin closest to the other cell and remove it:
                num_bins = num_bins
                far_borders_red, size_list_of_other_bins_red, closest_bin_size_red, closest_bin_indices_red = create_and_remove_bin_closest_to_other_cell(cleanborders[:,:,0].astype(bool), ordered_border_indices_red, mean_closest_indice_loc_red, num_bins)
                
                ### Do the same for the green borders:
                far_borders_blu, size_list_of_other_bins_blu, closest_bin_size_blu, closest_bin_indices_blu = create_and_remove_bin_closest_to_other_cell(cleanborders[:,:,2].astype(bool), ordered_border_indices_blu, mean_closest_indice_loc_blu, num_bins)
    
                ### Combine the 2 borders into an RGB image and assign red to be 1 
                ### and green to be 2:
                borders_without_closest_bin = np.dstack((far_borders_red * red_id, np.zeros(tlchan.shape), far_borders_blu * blu_id))
                closest_bin_red_and_blu = np.zeros(np.shape(cleanareas))
                #closest_bin_red_and_blu[list(zip(*closest_bin_indices_red))] = red_id ###This method of indexing is going to become deprecated, see below for alternative:
                #closest_bin_red_and_blu[list(zip(*closest_bin_indices_blu))] = blu_id
                closest_bin_red_and_blu[tuple(np.asarray(a) for a in zip(*closest_bin_indices_red))] = red_id
                closest_bin_red_and_blu[tuple(np.asarray(a) for a in zip(*closest_bin_indices_blu))] = blu_id
                reordered_closest_bin_indices_red, reordered_closest_bin_indices_blu, closest_bin_len_red, closest_bin_len_blu = make_open_border_indices_sequential(closest_bin_red_and_blu, red_id, blu_id, 0, 2)
    
                ### Re-define the sequencec of pixels for the borders_without_closest_bin:
                reordered_border_indices_red, reordered_border_indices_blu, perimeter_of_bins_1to7_cell1, perimeter_of_bins_1to7_cell2 = make_open_border_indices_sequential(borders_without_closest_bin, red_id, blu_id, 0, 2)
                
                renumbered_border_indices_red = [p for p in enumerate(reordered_border_indices_red)]
                renumbered_border_indices_blu = [p for p in enumerate(reordered_border_indices_blu)]
                
                ### Next I need to use these renumbered border indices red and grn to define the
                ### other 7 bins and find their curvatures:
                num_bins = num_bins-1
                bin_indices_red, bin_start_red, bin_stop_red = sort_pixels_into_bins(reordered_border_indices_red, size_list_of_other_bins_red, closest_bin_indices_red, num_bins)
                bin_indices_blu, bin_start_blu, bin_stop_blu = sort_pixels_into_bins(reordered_border_indices_blu, size_list_of_other_bins_blu, closest_bin_indices_blu, num_bins)
                
                bin_start_red_indices = np.array(reordered_border_indices_red)[bin_start_red]            
                bin_start_red_indices = np.row_stack((bin_start_red_indices, reordered_closest_bin_indices_red[0]))
                bin_stop_red_indices = np.array(reordered_border_indices_red)[np.array(bin_stop_red)-1]
                bin_stop_red_indices = np.row_stack((bin_stop_red_indices, reordered_closest_bin_indices_red[-1]))
    
                bin_start_blu_indices = np.array(reordered_border_indices_blu)[bin_start_blu]            
                bin_start_blu_indices = np.row_stack((bin_start_blu_indices, reordered_closest_bin_indices_blu[0]))
                bin_stop_blu_indices = np.array(reordered_border_indices_blu)[np.array(bin_stop_blu)-1]
                bin_stop_blu_indices = np.row_stack((bin_stop_blu_indices, reordered_closest_bin_indices_blu[-1]))
                
    
                ### Finally, fit arcs to the binned borders and plot and return the
                ### the details to me:
                bin_curvature_details_red, all_fits_red_binned = fit_curves_to_bins(bin_indices_red, cleanborders[:,:,0], odr_circle_fit_iterations)
                bin_curvature_details_blu, all_fits_blu_binned = fit_curves_to_bins(bin_indices_blu, cleanborders[:,:,2], odr_circle_fit_iterations)
    
                ### While we're at it, also give me the curvature fit to the entire
                ### region of the cell borders not in contact with another cell:
                whole_curvature_details_red, all_fits_red_whole = fit_curves_to_bins([zip(*np.where(cleanborders[:,:,0]))], cleanborders[:,:,0], odr_circle_fit_iterations)
                whole_curvature_details_blu, all_fits_blu_whole = fit_curves_to_bins([zip(*np.where(cleanborders[:,:,2]))], cleanborders[:,:,2], odr_circle_fit_iterations)



            ### Determine the noncontactborders of the redseg:
            redseg_border_inner = np.dstack(( mh.borders(redseg_clean) * redseg_clean, np.zeros(redseg_clean.shape), np.zeros(redseg_clean.shape) ))
            redseg_border_clean_inner = clean_up_borders(redseg_border_inner)[:,:,0]
            
            redseg_border_outer = np.dstack(( mh.borders(redseg_clean) * ~redseg_clean, np.zeros(redseg_clean.shape), np.zeros(redseg_clean.shape) ))
            redseg_border_clean_outer = clean_up_borders(redseg_border_outer)[:,:,0]

            redseg_noncontactborders = np.logical_or(redseg_border_clean_inner, redseg_border_clean_outer)
            redseg_noncontactborders_filled = morphology.remove_small_holes(redseg_noncontactborders)

            ### Determine the noncontactborders of the grnseg:
            grnseg_border_inner = np.dstack(( mh.borders(grnseg_clean) * grnseg_clean, np.zeros(grnseg_clean.shape), np.zeros(grnseg_clean.shape) ))
            grnseg_border_clean_inner = clean_up_borders(grnseg_border_inner)[:,:,0]
            
            grnseg_border_outer = np.dstack(( mh.borders(grnseg_clean) * ~grnseg_clean, np.zeros(grnseg_clean.shape), np.zeros(grnseg_clean.shape) ))
            grnseg_border_clean_outer = clean_up_borders(grnseg_border_outer)[:,:,0]

            grnseg_noncontactborders = np.logical_or(grnseg_border_clean_inner, grnseg_border_clean_outer)
            grnseg_noncontactborders_filled = morphology.remove_small_holes(grnseg_noncontactborders)
            
            ### Determine the noncontactborders of the bluseg:
            bluseg_border_inner = np.dstack(( mh.borders(bluseg_clean) * bluseg_clean, np.zeros(bluseg_clean.shape), np.zeros(bluseg_clean.shape) ))
            bluseg_border_clean_inner = clean_up_borders(bluseg_border_inner)[:,:,0]# 0 because of the np.dstack line above
            
            bluseg_border_outer = np.dstack(( mh.borders(bluseg_clean) * ~bluseg_clean, np.zeros(bluseg_clean.shape), np.zeros(bluseg_clean.shape) ))
            bluseg_border_clean_outer = clean_up_borders(bluseg_border_outer)[:,:,0]

            bluseg_noncontactborders = np.logical_or(bluseg_border_clean_inner, bluseg_border_clean_outer)
            bluseg_noncontactborders_filled = morphology.remove_small_holes(bluseg_noncontactborders)


            # Since there is no redseg_contactsurface for this scenario find
            # the pixel closest to the other cell and remove it:
            # Since we already have the mean_closest_indice_loc_red just find 
            # the redseg_noncontactborders pixel closest to that:

            if redseg_clean.any():
                # redseg_noncontact_closest_px_distances_outer = (np.asarray(np.where(redseg_border_clean_outer)) - mean_closest_indice_loc_red[0].T)**2
                redseg_noncontact_closest_px_distances_outer = (np.asarray(np.where(redseg_border_clean_outer)) - closest_redborder_indices_rc.T)**2
                redseg_noncontact_closest_px_loc_outer = np.argmin(np.sum(redseg_noncontact_closest_px_distances_outer, axis=0))
                redseg_noncontact_closest_px_outer = list(zip(*np.where(redseg_border_clean_outer)))[redseg_noncontact_closest_px_loc_outer]
    
                # redseg_noncontact_closest_px_distances_inner = (np.asarray(np.where(redseg_border_clean_inner)) - mean_closest_indice_loc_red[0].T)**2
                redseg_noncontact_closest_px_distances_inner = (np.asarray(np.where(redseg_border_clean_inner)) - closest_redborder_indices_rc.T)**2
                redseg_noncontact_closest_px_loc_inner = np.argmin(np.sum(redseg_noncontact_closest_px_distances_inner, axis=0))
                redseg_noncontact_closest_px_inner = list(zip(*np.where(redseg_border_clean_inner)))[redseg_noncontact_closest_px_loc_inner]
    
    
                redseg_noncontactborders_outer_opened = np.copy(redseg_border_clean_outer)
                redseg_noncontactborders_outer_opened[redseg_noncontact_closest_px_outer] = False
                redseg_noncontactborders_outer_opened = np.dstack((redseg_noncontactborders_outer_opened, np.zeros(redseg_noncontactborders_outer_opened.shape), np.zeros(redseg_noncontactborders_outer_opened.shape)))
                redseg_noncontactborders_inner_opened = np.copy(redseg_border_clean_inner)
                redseg_noncontactborders_inner_opened[redseg_noncontact_closest_px_inner] = False
                redseg_noncontactborders_inner_opened = np.dstack((redseg_noncontactborders_inner_opened, np.zeros(redseg_noncontactborders_inner_opened.shape), np.zeros(redseg_noncontactborders_inner_opened.shape)))
                
    
                # Return the redseg_noncontactborders and redseg_contactsurface 
                # indices in sequential order from contactcorner to contactcorner:
                redseg_noncontact_borders_sequential_outer, blueseg_unused_outer, redchan_unused_perim_outer, bluchan_unused_perim_outer = make_open_border_indices_sequential(redseg_noncontactborders_outer_opened, red_id, blu_id, 0, 2)
                redseg_noncontact_borders_sequential_inner, blueseg_unused_inner, redchan_unused_perim_inner, bluchan_unused_perim_inner = make_open_border_indices_sequential(redseg_noncontactborders_inner_opened, red_id, blu_id, 0, 2)
    
                # Also include the closest pixel for inner and outer borders as the
                # first indice of the sequentially ordered borders:
                redseg_noncontact_borders_sequential_outer.insert(0, tuple(list(redseg_noncontact_closest_px_outer) + [0]))
                redseg_noncontact_borders_sequential_inner.insert(0, tuple(list(redseg_noncontact_closest_px_inner) + [0]))
                # Give the fluorescence for the noncontactborders of redchan and
                # grnchan in sequential order:
                redchan_fluor_outside_contact_outer = redchan[tuple(np.asarray(a) for a in zip(*redseg_noncontact_borders_sequential_outer))[:2]]
                redchan_fluor_outside_contact_inner = redchan[tuple(np.asarray(a) for a in zip(*redseg_noncontact_borders_sequential_inner))[:2]]
            
            
            
            # Since there is no grnseg_contactsurface for this scenario find
            # the pixel closest to the other cell and remove it:
            # Since we already have the mean_closest_indice_loc_red just find 
            # the grnseg_noncontactborders pixel closest to that:

            if grnseg_clean.any():
                # grnseg_noncontact_closest_px_distances_outer = (np.asarray(np.where(grnseg_border_clean_outer)) - mean_closest_indice_loc_red[0].T)**2
                grnseg_noncontact_closest_px_distances_outer = (np.asarray(np.where(grnseg_border_clean_outer)) - closest_redborder_indices_rc.T)**2
                grnseg_noncontact_closest_px_loc_outer = np.argmin(np.sum(grnseg_noncontact_closest_px_distances_outer, axis=0))
                grnseg_noncontact_closest_px_outer = list(zip(*np.where(grnseg_border_clean_outer)))[grnseg_noncontact_closest_px_loc_outer]
    
                # grnseg_noncontact_closest_px_distances_inner = (np.asarray(np.where(grnseg_border_clean_inner)) - mean_closest_indice_loc_red[0].T)**2
                grnseg_noncontact_closest_px_distances_inner = (np.asarray(np.where(grnseg_border_clean_inner)) - closest_redborder_indices_rc.T)**2
                grnseg_noncontact_closest_px_loc_inner = np.argmin(np.sum(grnseg_noncontact_closest_px_distances_inner, axis=0))
                grnseg_noncontact_closest_px_inner = list(zip(*np.where(grnseg_border_clean_inner)))[grnseg_noncontact_closest_px_loc_inner]
    
    
                grnseg_noncontactborders_outer_opened = np.copy(grnseg_border_clean_outer)
                grnseg_noncontactborders_outer_opened[grnseg_noncontact_closest_px_outer] = False
                grnseg_noncontactborders_outer_opened = np.dstack((grnseg_noncontactborders_outer_opened, np.zeros(grnseg_noncontactborders_outer_opened.shape), np.zeros(grnseg_noncontactborders_outer_opened.shape)))
                grnseg_noncontactborders_inner_opened = np.copy(grnseg_border_clean_inner)
                grnseg_noncontactborders_inner_opened[grnseg_noncontact_closest_px_inner] = False
                grnseg_noncontactborders_inner_opened = np.dstack((grnseg_noncontactborders_inner_opened, np.zeros(grnseg_noncontactborders_inner_opened.shape), np.zeros(grnseg_noncontactborders_inner_opened.shape)))
                
    
                # Return the redseg_noncontactborders and redseg_contactsurface 
                # indices in sequential order from contactcorner to contactcorner:
                grnseg_noncontact_borders_sequential_outer, blueseg_unused_outer, grnchan_unused_perim_outer, bluchan_unused_perim_outer = make_open_border_indices_sequential(grnseg_noncontactborders_outer_opened, red_id, blu_id, 0, 2) #grn_id
                grnseg_noncontact_borders_sequential_inner, blueseg_unused_inner, grnchan_unused_perim_inner, bluchan_unused_perim_inner = make_open_border_indices_sequential(grnseg_noncontactborders_inner_opened, red_id, blu_id, 0, 2) #grn_id
    
                # Also include the closest pixel for inner and outer borders as the
                # first indice of the sequentially ordered borders:
                grnseg_noncontact_borders_sequential_outer.insert(0, tuple(list(grnseg_noncontact_closest_px_outer) + [0]))
                grnseg_noncontact_borders_sequential_inner.insert(0, tuple(list(grnseg_noncontact_closest_px_inner) + [0]))
                # Give the fluorescence for the noncontactborders of redchan and
                # grnchan in sequential order:
                grnchan_fluor_outside_contact_outer = grnchan[tuple(np.asarray(a) for a in zip(*grnseg_noncontact_borders_sequential_outer))[:2]]
                grnchan_fluor_outside_contact_inner = grnchan[tuple(np.asarray(a) for a in zip(*grnseg_noncontact_borders_sequential_inner))[:2]]
    

            # Since there is no bluseg_contactsurface for this scenario find
            # the pixel closest to the other cell and remove it:
            # Since we already have the mean_closest_indice_loc_blu just find 
            # the bluseg_noncontactborders pixel closest to that:

            if bluseg_clean.any():
                bluseg_noncontact_closest_px_distances_outer = (np.asarray(np.where(bluseg_border_clean_outer)) - closest_bluborder_indices_rc.T)**2
                bluseg_noncontact_closest_px_loc_outer = np.argmin(np.sum(bluseg_noncontact_closest_px_distances_outer, axis=0))
                bluseg_noncontact_closest_px_outer = list(zip(*np.where(bluseg_border_clean_outer)))[bluseg_noncontact_closest_px_loc_outer]
    
                bluseg_noncontact_closest_px_distances_inner = (np.asarray(np.where(bluseg_border_clean_inner)) - closest_bluborder_indices_rc.T)**2
                bluseg_noncontact_closest_px_loc_inner = np.argmin(np.sum(bluseg_noncontact_closest_px_distances_inner, axis=0))
                bluseg_noncontact_closest_px_inner = list(zip(*np.where(bluseg_border_clean_inner)))[bluseg_noncontact_closest_px_loc_inner]
    
    
                bluseg_noncontactborders_outer_opened = np.copy(bluseg_border_clean_outer)
                bluseg_noncontactborders_outer_opened[bluseg_noncontact_closest_px_outer] = False
                bluseg_noncontactborders_outer_opened = np.dstack((bluseg_noncontactborders_outer_opened, np.zeros(bluseg_noncontactborders_outer_opened.shape), np.zeros(bluseg_noncontactborders_outer_opened.shape)))
                bluseg_noncontactborders_inner_opened = np.copy(bluseg_border_clean_inner)
                bluseg_noncontactborders_inner_opened[bluseg_noncontact_closest_px_inner] = False
                bluseg_noncontactborders_inner_opened = np.dstack((bluseg_noncontactborders_inner_opened, np.zeros(bluseg_noncontactborders_inner_opened.shape), np.zeros(bluseg_noncontactborders_inner_opened.shape)))
                
    
                # Return the redseg_noncontactborders and redseg_contactsurface 
                # indices in sequential order from contactcorner to contactcorner:
                bluseg_noncontact_borders_sequential_outer, redseg_noncontact_borders_sequential_outer, redchan_unused_perim_outer, bluchan_unused_perim_outer = make_open_border_indices_sequential(bluseg_noncontactborders_outer_opened, red_id, blu_id, 0, 2)
                bluseg_noncontact_borders_sequential_inner, redseg_noncontact_borders_sequential_inner, redchan_unused_perim_inner, bluchan_unused_perim_inner = make_open_border_indices_sequential(bluseg_noncontactborders_inner_opened, red_id, blu_id, 0, 2)
    
                # Also include the closest pixel for inner and outer borders as the
                # first indice of the sequentially ordered borders:
                bluseg_noncontact_borders_sequential_outer.insert(0, tuple(list(bluseg_noncontact_closest_px_outer) + [0]))
                bluseg_noncontact_borders_sequential_inner.insert(0, tuple(list(bluseg_noncontact_closest_px_inner) + [0]))
                # Give the fluorescence for the noncontactborders of bluchan in
                # sequential order:
                bluchan_fluor_outside_contact_outer = bluchan[tuple(np.asarray(a) for a in zip(*bluseg_noncontact_borders_sequential_outer))[:2]]
                bluchan_fluor_outside_contact_inner = bluchan[tuple(np.asarray(a) for a in zip(*bluseg_noncontact_borders_sequential_inner))[:2]]
    
            
            # As the sequentially ordered borders sometimes miss pixels that are
            # in between the inner and outer borders, also just give the
            # unordered fluorescences:
            redchan_fluor_outside_contact = redchan[np.where(redseg_noncontactborders_filled)]
            grnchan_fluor_outside_contact = grnchan[np.where(grnseg_noncontactborders_filled)]
            bluchan_fluor_outside_contact = bluchan[np.where(bluseg_noncontactborders_filled)]
            
            
            # Also get un-ordered cytoplasmic fluorescence values:
            redseg_cytopl_area = np.logical_xor(redseg_border_clean_inner, scp_morph.binary_fill_holes(redseg_border_clean_inner))
            grnseg_cytopl_area = np.logical_xor(grnseg_border_clean_inner, scp_morph.binary_fill_holes(grnseg_border_clean_inner))
            bluseg_cytopl_area = np.logical_xor(bluseg_border_clean_inner, scp_morph.binary_fill_holes(bluseg_border_clean_inner))

            redchan_fluor_cytoplasmic = redchan[redseg_cytopl_area]
            grnchan_fluor_cytoplasmic = grnchan[grnseg_cytopl_area]
            bluchan_fluor_cytoplasmic = bluchan[bluseg_cytopl_area]



            ### Define some more default values:
            area_cell_1 = len(np.where(cleanareas==red_id)[0])
            area_cell_2 = len(np.where(cleanareas==blu_id)[0])
            noncontact_perimeter_cell1 = perimeter_cell1
            noncontact_perimeter_cell2 = perimeter_cell2
            noncontactborders = np.copy(cleanborders)

            degree_angle1 = 0.0
            degree_angle2 = 0.0
            average_angle = 0.0
            red_halfangle1 = 0.0
            red_halfangle2 = 0.0
            blu_halfangle1 = 0.0
            blu_halfangle2 = 0.0
            contactsurface_len = 0.0
            
            
            ### Plot the cells and any parameters so far:
            fig.clf()
            fig = plt.figure(num=str(filenumber+1)+'_'+os.path.basename(filename), figsize=(16., 10.))
            plt.suptitle(str(filenumber+1)+' - '+os.path.basename(filename), verticalalignment='center', x=0.5, y=0.5, fontsize=14)
            ax1 = fig.add_subplot(241)
            ax1.yaxis.set_visible(False)
            ax1.xaxis.set_visible(False)
            ax1.imshow(tlchan, cmap='gray')
            
            ax2 = fig.add_subplot(242)
            ax2.yaxis.set_visible(False)
            ax2.xaxis.set_visible(False)
            #figcellborderred = ax2.imshow(mh.overlay(gray = mh.stretch(tlchan), red = cleanborders[:,:,0]))
            #figcellred = ax2.imshow(mh.overlay(gray = mh.stretch(redchan), red = cleanborders[:,:,0]))
            fig_noncont_redcell = ax2.imshow(mh.overlay(mh.stretch(redchan), red=redseg_noncontactborders))
            #fig_noncont_redcell = ax2.imshow(mh.stretch(redchan), cmap='gray')
            ax2.axes.autoscale(False)
            
            ax3 = fig.add_subplot(243)
            ax3.yaxis.set_visible(False)
            ax3.xaxis.set_visible(False)
            #figcellborderblu = ax3.imshow(mh.overlay(gray = mh.stretch(tlchan), blue = cleanborders[:,:,2]))
            fig_noncont_grncell = ax3.imshow(mh.overlay(mh.stretch(grnchan), green=grnseg_noncontactborders))
            #fig_noncont_grncell = ax3.imshow(mh.stretch(grnchan), cmap='gray')
            ax3.axes.autoscale(False)
            
            ax4 = fig.add_subplot(246)
            ax4.yaxis.set_visible(False)
            ax4.xaxis.set_visible(False)
            #ax4.imshow(tlchan, cmap='gray')
            fig_cont_redcell = ax4.imshow(mh.stretch(redchan), cmap='gray')
            ax4.axes.autoscale(False)
            
            ax5 = fig.add_subplot(247)
            ax5.yaxis.set_visible(False)
            ax5.xaxis.set_visible(False)
            #ax5.imshow(tlchan, cmap='gray')
            fig_cont_grncell = ax5.imshow(mh.stretch(grnchan), cmap='gray')
            ax5.axes.autoscale(False)
    
            ax6 = fig.add_subplot(245)
            ax6.yaxis.set_visible(False)
            ax6.xaxis.set_visible(False)
            ax6.imshow(mh.overlay(gray = mh.stretch(tlchan), red = cleanborders[:,:,0], blue=cleanborders[:,:,2]))
            ax6.axes.autoscale(False)
            ax6.scatter(cell_max_rcd[0][1], cell_max_rcd[0][0], color='m', marker='o')
            ax6.scatter(cell_max_rcd[1][1], cell_max_rcd[1][0], color='m', marker='o')
            ax6.scatter(bg_max_rcd[0][1], bg_max_rcd[0][0], color='m', marker='x')
            ax6.scatter(bg_max_rcd[1][1], bg_max_rcd[1][0], color='m', marker='x')

            ax7 = fig.add_subplot(244)
            ax7.yaxis.set_visible(False)
            ax7.xaxis.set_visible(False)
            # ax_noncont_blucell = ax7.imshow(mh.overlay(mh.stretch(bluchan), blue=cleanborders[:,:,2]))
            ax_noncont_blucell = ax7.imshow(mh.overlay(mh.stretch(bluchan), blue=bluseg_noncontactborders))
            ax7.axes.autoscale(False)
            
            ax8 = fig.add_subplot(248)
            ax8.yaxis.set_visible(False)
            ax8.xaxis.set_visible(False)
            fig_cont_blucell = ax8.imshow(mh.stretch(bluchan), cmap='gray')
            ax8.axes.autoscale(False)

    
            plt.tight_layout()
            plt.draw()


            ### Update the data table:
            listupdate()
            
            ### Move on to the next picture:
            continue


        ### If the cells ARE in contact, then proceed to calculate adhesiveness
        ### parameters:
        if cells_touching == True and skip == 'n':
            
            ### Clean up the contacts:
            contactcorners, contactsurface, noncontactborders = clean_up_contact_borders(rawcontactsurface, cleanborders, red_id, blu_id)
            
            ### Check to make sure that the contact corneres were properly
            ### detected:
            if len(np.where(contactcorners)[0]) == 4:
                print("Found the contact corners.")
            ### Or else just skip this image and fill any unfinished analyses
            ### with null data:
            else:
                print ("I failed to find the contact corners :c "
                       "\n Please review the processing of this file to see what "
                       "went wrong."
                       "\n The file is %s" % filename,
                       "\n The parameters used were: \n"
                       "\nRed gaussian filter value:", gaussian_filter_value_red,
                       "\nGreen gaussian filter value:", gaussian_filter_value_grn,
                       #"\nRed thresholding value:", redthreshval,
                       #"\nGreen thresholding value:", grnthreshval,
                       )
                messages.append("I failed to find the contact corners.")
                listupdate()
                continue

            ### Now that we have defined the corners of the contact, the contact
            ### surface, and the parts of the cell border that are not in
            ### contact, we can find the contact angles:
            temp = find_contact_angles(contactcorners, contactsurface, noncontactborders, cell1_id = red_id, cell2_id = blu_id)
            degree_angle1, degree_angle2, average_angle = temp[0], temp[1], temp[2]
            vector1_start, vector1 = temp[3], temp[4]
            vector2_start, vector2 = temp[5], temp[6]
            vector3_start, vector3 = temp[7], temp[8]
            vector4_start, vector4 = temp[9], temp[10]
            pathfinder = temp[11]
            angles_angle_labeled = temp[12]
            print('The angles are:', degree_angle1, 'and', degree_angle2)

            ### Let's also find the half-angles of these angles. First some
            ### preliminary steps: Find the vector that divides the contact angles:
            rawcontact_vector_start = (np.asarray(vector1_start[::-1]) + np.asarray(vector2_start[::-1])) / 2.0
            rawcontact_vector_end = (np.asarray(vector3_start[::-1]) + np.array(vector4_start[::-1])) / 2.0
            rawcontact_vector = rawcontact_vector_start - rawcontact_vector_end
                
            ### Convert the raw contact vector in to (x,y) format:
            contact_vector = np.zeros((2))
            contact_vector[0] = rawcontact_vector[1]
            contact_vector[1] = rawcontact_vector[0]
            perpendicular2contact_vector = rawcontact_vector[:2]
            perpendicular2contact_vector *=[-1.0,1.0]
            if np.dot(contact_vector,perpendicular2contact_vector) != 0:
                raise(ValueError)
                print("Linescan of gap is not perpendicular to contact surface!")
                messages.append("Linescan of gap is not perpendicular to contact surface!")
            n_contactsurface_px = int(len(np.where(contactsurface)[0])/2)
            cx,cy = np.linspace(rawcontact_vector_start[1], rawcontact_vector_end[1], num=n_contactsurface_px), np.linspace(rawcontact_vector_start[0], rawcontact_vector_end[0], num=n_contactsurface_px)
            imgform_contact_points = zip(cy,cx)
        
        
            ### Determine the half-angle for each of the red and green cells:
            red_halfangle1_rad = math.acos( np.round( ((np.dot(vector1,contact_vector)) / (((vector1[0]**2 + vector1[1]**2)**0.5)*((contact_vector[0]**2 + contact_vector[1]**2)**0.5)) ), decimals=10) )
            red_halfangle2_rad = math.acos( np.round( ((np.dot(vector3,contact_vector)) / (((vector3[0]**2 + vector3[1]**2)**0.5)*((contact_vector[0]**2 + contact_vector[1]**2)**0.5)) ), decimals=10) )
            blu_halfangle1_rad = math.acos( np.round( ((np.dot(vector2,contact_vector)) / (((vector2[0]**2 + vector2[1]**2)**0.5)*((contact_vector[0]**2 + contact_vector[1]**2)**0.5)) ), decimals=10) )
            blu_halfangle2_rad = math.acos( np.round( ((np.dot(vector4,contact_vector)) / (((vector4[0]**2 + vector4[1]**2)**0.5)*((contact_vector[0]**2 + contact_vector[1]**2)**0.5)) ), decimals=10) )
            ### and convert to degrees:
            red_halfangle1 = math.degrees(red_halfangle1_rad)
            red_halfangle2 = 180-math.degrees(red_halfangle2_rad)
            blu_halfangle1 = math.degrees(blu_halfangle1_rad)
            blu_halfangle2 = 180-math.degrees(blu_halfangle2_rad)



            ### Find the mean fluorescence intensity value that the red and grn
            ### linescans of the contact intersect at as an indirect measure of
            ### how close the 2 cells are:
            #red_gap_scanx, red_gap_scany, grn_gap_scanx, grn_gap_scany, blu_gap_scanx, blu_gap_scany = perform_linescan(contactsurface, perpendicular2contact_vector, linescan_length, fluor_chans, red_id, blu_id)

            ### Find the perimeter of the red and green cells:
            ordered_border_indices_red, ordered_border_indices_blu, numbered_border_indices_red, numbered_border_indices_blu, perimeter_cell1, perimeter_cell2 = count_pixels_around_borders_sequentially(cleanborders, red_id, blu_id, 0, 2)

            ### Give the noncontactborders sequential information:
            reordered_border_indices_red, reordered_border_indices_blu, noncontact_perimeter_cell1, noncontact_perimeter_cell2 = make_open_border_indices_sequential(noncontactborders+pathfinder, red_id, blu_id, 0, 2)


            renumbered_border_indices_red = [p for p in enumerate(reordered_border_indices_red)]
            renumbered_border_indices_blu = [p for p in enumerate(reordered_border_indices_blu)]

            ### Bin the noncontactborders into 7 bins:
            num_bins = num_bins-1
            bin_size_list_red = create_border_bin_sizes((noncontactborders[:,:,0]+pathfinder[:,:,0]).astype(bool), reordered_border_indices_red, num_bins)
            bin_size_list_blu = create_border_bin_sizes((noncontactborders[:,:,2]+pathfinder[:,:,2]).astype(bool), reordered_border_indices_blu, num_bins)

            ### Make the 8th bin the contactsurface:
#            contact_bin_indices_red = zip(*np.where(contactsurface==1))
#            contact_bin_indices_grn = zip(*np.where(contactsurface==2))
            reordered_contact_surface_red, reordered_contact_surface_blu, contactsurface_len1, contactsurface_len2 = make_open_border_indices_sequential(contactsurface, red_id, blu_id, 0, 2)
            contactsurface_len = (contactsurface_len1 + contactsurface_len2) / 2.0
            print("The length of the contact surface is %.3f pixels" %contactsurface_len)

            ### Return the bin-sorted indices for each pixel:
#            bin_indices_red, bin_start_red, bin_stop_red = sort_pixels_into_bins(reordered_border_indices_red, bin_size_list_red, contact_bin_indices_red, num_bins)
#            bin_indices_grn, bin_start_grn, bin_stop_grn = sort_pixels_into_bins(reordered_border_indices_grn, bin_size_list_grn, contact_bin_indices_grn, num_bins)
            bin_indices_red, bin_start_red, bin_stop_red = sort_pixels_into_bins(reordered_border_indices_red, bin_size_list_red, reordered_contact_surface_red, num_bins)
            bin_indices_blu, bin_start_blu, bin_stop_blu = sort_pixels_into_bins(reordered_border_indices_blu, bin_size_list_blu, reordered_contact_surface_blu, num_bins)

            bin_start_red_indices = np.array(reordered_border_indices_red)[bin_start_red]            
            bin_start_red_indices = np.row_stack((bin_start_red_indices, reordered_contact_surface_red[0]))
            bin_stop_red_indices = np.array(reordered_border_indices_red)[np.array(bin_stop_red)-1]
            bin_stop_red_indices = np.row_stack((bin_stop_red_indices, reordered_contact_surface_red[-1]))

            bin_start_blu_indices = np.array(reordered_border_indices_blu)[bin_start_blu]            
            bin_start_blu_indices = np.row_stack((bin_start_blu_indices, reordered_contact_surface_blu[0]))
            bin_stop_blu_indices = np.array(reordered_border_indices_blu)[np.array(bin_stop_blu)-1]
            bin_stop_blu_indices = np.row_stack((bin_stop_blu_indices, reordered_contact_surface_blu[-1]))



            ### Add the len of the 8th bin to the bin_size_list_red (and grn):
#            bin_size_list_red.append(len(bin_indices_red[7]))
#            bin_size_list_grn.append(len(bin_indices_grn[7]))

            bin_start_red.append(bin_indices_red[7][0])
            bin_stop_red.append(bin_indices_red[7][-1])
            bin_start_blu.append(bin_indices_blu[7][0])
            bin_stop_blu.append(bin_indices_blu[7][-1])

            ### Find the binned curvatures:
            bin_curvature_details_red, all_fits_red_binned = fit_curves_to_bins(bin_indices_red, cleanborders[:,:,0], odr_circle_fit_iterations)
            bin_curvature_details_blu, all_fits_blu_binned = fit_curves_to_bins(bin_indices_blu, cleanborders[:,:,2], odr_circle_fit_iterations)
                 
            ### While we're at it, also give me the curvature fit to the entire
            ### region of the cell borders not in contact with another cell:
            whole_curvature_details_red, all_fits_red_whole = fit_curves_to_bins([zip(*np.where((noncontactborders[:,:,0]+pathfinder[:,:,0]).astype(bool)))], (noncontactborders[:,:,0]+pathfinder[:,:,0]).astype(bool), odr_circle_fit_iterations)
            whole_curvature_details_blu, all_fits_blu_whole = fit_curves_to_bins([zip(*np.where((noncontactborders[:,:,2]+pathfinder[:,:,2]).astype(bool)))], (noncontactborders[:,:,2]+pathfinder[:,:,2]).astype(bool), odr_circle_fit_iterations)

            ### Define some more default values:
            area_cell_1 = len(np.where(cleanareas==red_id)[0])
            area_cell_2 = len(np.where(cleanareas==blu_id)[0])




            # Find redseg inner and outer border pixels that are closest to the
            # contact corners of red_areas:
            redseg_border_inner = np.dstack(( mh.borders(redseg_clean) * redseg_clean, np.zeros(redseg_clean.shape), np.zeros(redseg_clean.shape) ))
            redseg_border_clean_inner = clean_up_borders(redseg_border_inner)[:,:,0]
            
            redseg_border_outer = np.dstack(( mh.borders(redseg_clean) * ~redseg_clean, np.zeros(redseg_clean.shape), np.zeros(redseg_clean.shape) ))
            redseg_border_clean_outer = clean_up_borders(redseg_border_outer)[:,:,0]
            
            redseg_inner_corner_locs = []
            redseg_outer_corner_locs = []
            for corner in zip(*np.where(contactcorners[:, :, channel1])):
                redseg_inner_distances = np.ma.array((np.where(redseg_border_clean_inner)[0] - corner[0]) ** 2 + (np.where(redseg_border_clean_inner)[1] - corner[1]) ** 2)
                redseg_inner_closest_iloc = np.argmin(redseg_inner_distances)               
                redseg_inner_corner_indice = list(zip(*np.where(redseg_border_clean_inner)))[redseg_inner_closest_iloc]
                if redseg_inner_corner_indice in redseg_inner_corner_locs:
                    #print("This pixels has already been recorded!")
                    redseg_inner_distances[redseg_inner_closest_iloc] = np.ma.masked
                    redseg_inner_closest_iloc = np.argmin(redseg_inner_distances)               
                    redseg_inner_corner_indice = list(zip(*np.where(redseg_border_clean_inner)))[redseg_inner_closest_iloc]
                redseg_inner_corner_locs.append(redseg_inner_corner_indice)
                
                
                redseg_outer_distances = np.ma.array((np.where(redseg_border_clean_outer)[0] - corner[0]) ** 2 + (np.where(redseg_border_clean_outer)[1] - corner[1]) ** 2)
                redseg_outer_closest_iloc = np.argmin(redseg_outer_distances) 
                redseg_outer_corner_indice = list(zip(*np.where(redseg_border_clean_outer)))[redseg_outer_closest_iloc]
                if redseg_outer_corner_indice in redseg_outer_corner_locs:
                    #print("This pixels has already been recorded!")
                    redseg_outer_distances[redseg_outer_closest_iloc] = np.ma.masked
                    redseg_outer_closest_iloc = np.argmin(redseg_outer_distances) 
                    redseg_outer_corner_indice = list(zip(*np.where(redseg_border_clean_outer)))[redseg_outer_closest_iloc]
                redseg_outer_corner_locs.append(redseg_outer_corner_indice)

            redseg_inner_corners = np.zeros(redseg_clean.shape)
            redseg_inner_corners[tuple(np.asarray(a) for a in zip(*redseg_inner_corner_locs))] = True
            redseg_outer_corners = np.zeros(redseg_clean.shape)
            redseg_outer_corners[tuple(np.asarray(a) for a in zip(*redseg_outer_corner_locs))] = True
            
            redseg_inner_contactsurface = np.copy(redseg_border_clean_inner)
            redseg_inner_noncontactborders = np.copy(redseg_border_clean_inner)
            # Break the redseg_border_clean_inner at the contact corners:
            redseg_inner_contactsurface[np.where(redseg_inner_corners)] = 0
            redseg_inner_noncontactborders[np.where(redseg_inner_corners)] = 0
            
            # Label the resulting 2 pieces so that they can be classified as either
            # contactsurface or noncontactborders:
            redseg_inner_contactsurface = morphology.label(redseg_inner_contactsurface, connectivity=2)
            if len(np.unique(redseg_inner_contactsurface)) <= 3:
                pass
            else:
                redseg_inner_contactsurface = np.zeros(redseg_inner_contactsurface.shape)
            
            # When the contactsurface is small, sometimes the redseg_contactsurface
            # consists only of the redseg_contactcorners. In this case there will
            # be only 1 piece (the noncontactborders) when the contactcorners are
            # removed. Therefore add the redseg_contactcorners to each possible
            # surface:
            borders1 = (redseg_inner_contactsurface == 1) + redseg_inner_corners.astype(bool)
            borders2 = (redseg_inner_contactsurface == 2) + redseg_inner_corners.astype(bool)
            
            # Compare similarity between label 1 and label 2:
            #ssim_borders1 = structural_similarity((redseg_inner_contactsurface == 1).astype(bool), contactsurface[:,:,0].astype(bool))
            #ssim_borders2 = structural_similarity((redseg_inner_contactsurface == 2).astype(bool), contactsurface[:,:,0].astype(bool))
            ssim_borders1 = structural_similarity(borders1, contactsurface[:,:,0].astype(bool))
            ssim_borders2 = structural_similarity(borders2, contactsurface[:,:,0].astype(bool))
            # Classify the more similar label as being contactsurface,
            # put the noncontactborders in a new array and remove the 
            # noncontactborders from the contactsurface array:
            if ssim_borders2 > ssim_borders1:
                redseg_inner_noncontactborders[np.where(borders2)] = False
                redseg_inner_noncontactborders = redseg_inner_noncontactborders.astype(bool)
                redseg_inner_contactsurface[np.where(borders1)] = False
                redseg_inner_contactsurface = redseg_inner_contactsurface.astype(bool)
                redseg_inner_contactsurface[np.where(redseg_inner_corners)] = True
            elif ssim_borders1 > ssim_borders2:
                redseg_inner_noncontactborders[np.where(borders1)] = False
                redseg_inner_noncontactborders = redseg_inner_noncontactborders.astype(bool)
                redseg_inner_contactsurface[np.where(borders2)] = False
                redseg_inner_contactsurface = redseg_inner_contactsurface.astype(bool)
                redseg_inner_contactsurface[np.where(redseg_inner_corners)] = True
            else:
                redseg_inner_contactsurface = np.zeros(redseg_inner_contactsurface.shape)
                redseg_inner_noncontactborders = np.zeros(redseg_inner_contactsurface.shape)
            
            # Do the same for the outer borders:
            redseg_outer_contactsurface = np.copy(redseg_border_clean_outer)
            redseg_outer_noncontactborders = np.copy(redseg_border_clean_outer)
            
            redseg_outer_contactsurface[np.where(redseg_outer_corners)] = False
            redseg_outer_noncontactborders[np.where(redseg_outer_corners)] = False

            redseg_outer_contactsurface = morphology.label(redseg_outer_contactsurface, connectivity=2)
            if len(np.unique(redseg_outer_contactsurface)) <= 3:
                pass
            else:
                redseg_outer_contactsurface = np.zeros(redseg_outer_contactsurface.shape)
            
            # When the contactsurface is small, sometimes the redseg_contactsurface
            # consists only of the redseg_contactcorners. In this case there will
            # be only 1 piece (the noncontactborders) when the contactcorners are
            # removed. Therefore add the redseg_contactcorners to each possible
            # surface:
            borders1 = (redseg_outer_contactsurface == 1) + redseg_outer_corners.astype(bool)
            borders2 = (redseg_outer_contactsurface == 2) + redseg_outer_corners.astype(bool)


            # Compare similarity between label 1 and label 2:
            ssim_borders1 = structural_similarity(borders1, contactsurface[:,:,0].astype(bool))
            ssim_borders2 = structural_similarity(borders2.astype(bool), contactsurface[:,:,0].astype(bool))
            # Classify the more similar label as being contactsurface,
            # put the noncontactborders in a new array and remove the 
            # noncontactborders from the contactsurface array:
            if ssim_borders2 > ssim_borders1:
                redseg_outer_noncontactborders[np.where(borders2)] = False
                redseg_outer_noncontactborders = redseg_outer_noncontactborders.astype(bool)
                redseg_outer_contactsurface[np.where(borders1)] = False
                redseg_outer_contactsurface = redseg_outer_contactsurface.astype(bool)
                redseg_outer_contactsurface[np.where(redseg_outer_corners)] = True
            elif ssim_borders1 > ssim_borders2:
                redseg_outer_noncontactborders[np.where(borders1)] = False
                redseg_outer_noncontactborders = redseg_outer_noncontactborders.astype(bool)
                redseg_outer_contactsurface[np.where(borders2)] = False
                redseg_outer_contactsurface = redseg_outer_contactsurface.astype(bool)
                redseg_outer_contactsurface[np.where(redseg_outer_corners)] = True
            else:
                redseg_outer_contactsurface = np.zeros(redseg_outer_contactsurface.shape)
                redseg_outer_noncontactborders = np.zeros(redseg_outer_contactsurface.shape)

            # Combine the inner and outer contactsurface back together.
            # Also do the same for inner and outer noncontactborders.
            redseg_contactsurface = np.logical_or(redseg_inner_contactsurface, redseg_outer_contactsurface)
            redseg_noncontactborders = np.logical_or(redseg_inner_noncontactborders, redseg_outer_noncontactborders)

            # Restore any removed pixels from the clean_up_borders function to
            # either the contactsurface or noncontactborders that end up being
            # right in between the inner and outer borders:
            redseg_contactsurface_filled = morphology.remove_small_holes(redseg_contactsurface)
            redseg_noncontactborders_filled = morphology.remove_small_holes(redseg_noncontactborders)
            
            
            # Return pixels in sequential order for inner and outer contactsurface:
            redseg_inner_contactsurface = np.dstack((redseg_inner_contactsurface, np.zeros(redseg_inner_contactsurface.shape), np.zeros(redseg_inner_contactsurface.shape)))
            redseg_inner_contactsurface_seq, blu_unsused, red_unused_perim, blu_unused_perim = make_open_border_indices_sequential(redseg_inner_contactsurface, red_id, blu_id, 0, 2)
            redseg_outer_contactsurface = np.dstack((redseg_outer_contactsurface, np.zeros(redseg_outer_contactsurface.shape), np.zeros(redseg_outer_contactsurface.shape)))
            redseg_outer_contactsurface_seq, blu_unused, red_unused_perim, blue_unused_perim = make_open_border_indices_sequential(redseg_outer_contactsurface, red_id, blu_id, 0, 2)
            
            # Also do same for inner and outer noncontactborders:
            redseg_inner_noncontactborders = np.dstack((redseg_inner_noncontactborders, np.zeros(redseg_inner_noncontactborders.shape), np.zeros(redseg_inner_noncontactborders.shape)))
            redseg_inner_noncontactborders_seq, blu_unused, red_unused_perim, blu_unused_perim = make_open_border_indices_sequential(redseg_inner_noncontactborders, red_id, blu_id, 0, 2)
            redseg_outer_noncontactborders = np.dstack((redseg_outer_noncontactborders, np.zeros(redseg_outer_noncontactborders.shape), np.zeros(redseg_outer_noncontactborders.shape)))
            redseg_outer_noncontactborders_seq, blu_unused, red_unused_perim, blu_unused_perim = make_open_border_indices_sequential(redseg_outer_noncontactborders, red_id, blu_id, 0, 2)
            
            
            
            # Find grnseg inner and outer border pixels that are closest to the
            # contact corners of grn_areas:
            grnseg_border_inner = np.dstack(( mh.borders(grnseg_clean) * grnseg_clean, np.zeros(grnseg_clean.shape), np.zeros(grnseg_clean.shape) ))
            grnseg_border_clean_inner = clean_up_borders(grnseg_border_inner)[:,:,0]
            
            grnseg_border_outer = np.dstack(( mh.borders(grnseg_clean) * ~grnseg_clean, np.zeros(grnseg_clean.shape), np.zeros(grnseg_clean.shape) ))
            grnseg_border_clean_outer = clean_up_borders(grnseg_border_outer)[:,:,0]
            
            grnseg_inner_corner_locs = []
            grnseg_outer_corner_locs = []
            for corner in zip(*np.where(contactcorners[:, :, channel1])):
                grnseg_inner_distances  = np.ma.array((np.where(grnseg_border_clean_inner)[0] - corner[0]) ** 2 + (np.where(grnseg_border_clean_inner)[1] - corner[1]) ** 2)
                grnseg_inner_closest_iloc = np.argmin(grnseg_inner_distances) 
                grnseg_inner_corner_indice = list(zip(*np.where(grnseg_border_clean_inner)))[grnseg_inner_closest_iloc]
                if grnseg_inner_corner_indice in grnseg_inner_corner_locs:
                    #print("This pixels has already been recorded!")
                    grnseg_inner_distances[grnseg_inner_closest_iloc] = np.ma.masked
                    grnseg_inner_closest_iloc = np.argmin(grnseg_inner_distances) 
                    grnseg_inner_corner_indice = list(zip(*np.where(grnseg_border_clean_inner)))[grnseg_inner_closest_iloc]
                grnseg_inner_corner_locs.append(grnseg_inner_corner_indice)
                
                grnseg_outer_distances = np.ma.array((np.where(grnseg_border_clean_outer)[0] - corner[0]) ** 2 + (np.where(grnseg_border_clean_outer)[1] - corner[1]) ** 2)
                grnseg_outer_closest_iloc = np.argmin(grnseg_outer_distances) 
                grnseg_outer_corner_indice = list(zip(*np.where(grnseg_border_clean_outer)))[grnseg_outer_closest_iloc]
                if grnseg_outer_corner_indice in grnseg_outer_corner_locs:
                    #print("This pixels has already been recorded!")
                    grnseg_outer_distances[grnseg_outer_closest_iloc] = np.ma.masked
                    grnseg_outer_closest_iloc = np.argmin(grnseg_outer_distances) 
                    grnseg_outer_corner_indice = list(zip(*np.where(grnseg_border_clean_outer)))[grnseg_outer_closest_iloc]
                grnseg_outer_corner_locs.append(grnseg_outer_corner_indice)

            grnseg_inner_corners = np.zeros(grnseg_clean.shape)
            grnseg_inner_corners[tuple(np.asarray(a) for a in zip(*grnseg_inner_corner_locs))] = True
            grnseg_outer_corners = np.zeros(grnseg_clean.shape)
            grnseg_outer_corners[tuple(np.asarray(a) for a in zip(*grnseg_outer_corner_locs))] = True
            
            grnseg_inner_contactsurface = np.copy(grnseg_border_clean_inner)
            grnseg_inner_noncontactborders = np.copy(grnseg_border_clean_inner)
            # Break the redseg_border_clean_inner at the contact corners:
            grnseg_inner_contactsurface[np.where(grnseg_inner_corners)] = 0
            grnseg_inner_noncontactborders[np.where(grnseg_inner_corners)] = 0
            
            # Label the resulting 2 pieces so that they can be classified as either
            # contactsurface or noncontactborders:
            grnseg_inner_contactsurface = morphology.label(grnseg_inner_contactsurface, connectivity=2)
            if len(np.unique(grnseg_inner_contactsurface)) <= 3:
                pass
            else:
                grnseg_inner_contactsurface = np.zeros(grnseg_inner_contactsurface.shape)
            
            # When the contactsurface is small, sometimes the redseg_contactsurface
            # consists only of the redseg_contactcorners. In this case there will
            # be only 1 piece (the noncontactborders) when the contactcorners are
            # removed. Therefore add the redseg_contactcorners to each possible
            # surface:
            borders1 = (grnseg_inner_contactsurface == 1) + grnseg_inner_corners.astype(bool)
            borders2 = (grnseg_inner_contactsurface == 2) + grnseg_inner_corners.astype(bool)
            
            # Compare similarity between label 1 and label 2:
            #ssim_borders1 = structural_similarity((redseg_inner_contactsurface == 1).astype(bool), contactsurface[:,:,0].astype(bool))
            #ssim_borders2 = structural_similarity((redseg_inner_contactsurface == 2).astype(bool), contactsurface[:,:,0].astype(bool))
            ssim_borders1 = structural_similarity(borders1, contactsurface[:,:,0].astype(bool))
            ssim_borders2 = structural_similarity(borders2, contactsurface[:,:,0].astype(bool))
            # Classify the more similar label as being contactsurface,
            # put the noncontactborders in a new array and remove the 
            # noncontactborders from the contactsurface array:
            if ssim_borders2 > ssim_borders1:
                grnseg_inner_noncontactborders[np.where(borders2)] = False
                grnseg_inner_noncontactborders = grnseg_inner_noncontactborders.astype(bool)
                grnseg_inner_contactsurface[np.where(borders1)] = False
                grnseg_inner_contactsurface = grnseg_inner_contactsurface.astype(bool)
                grnseg_inner_contactsurface[np.where(grnseg_inner_corners)] = True
            elif ssim_borders1 > ssim_borders2:
                grnseg_inner_noncontactborders[np.where(borders1)] = False
                grnseg_inner_noncontactborders = grnseg_inner_noncontactborders.astype(bool)
                grnseg_inner_contactsurface[np.where(borders2)] = False
                grnseg_inner_contactsurface = grnseg_inner_contactsurface.astype(bool)
                grnseg_inner_contactsurface[np.where(grnseg_inner_corners)] = True
            else:
                grnseg_inner_contactsurface = np.zeros(grnseg_inner_contactsurface.shape)
                grnseg_inner_noncontactborders = np.zeros(grnseg_inner_contactsurface.shape)
            
            # Do the same for the outer borders:
            grnseg_outer_contactsurface = np.copy(grnseg_border_clean_outer)
            grnseg_outer_noncontactborders = np.copy(grnseg_border_clean_outer)
            
            grnseg_outer_contactsurface[np.where(grnseg_outer_corners)] = False
            grnseg_outer_noncontactborders[np.where(grnseg_outer_corners)] = False

            grnseg_outer_contactsurface = morphology.label(grnseg_outer_contactsurface, connectivity=2)
            if len(np.unique(grnseg_outer_contactsurface)) <= 3:
                pass
            else:
                grnseg_outer_contactsurface = np.zeros(grnseg_outer_contactsurface.shape)
            
            # When the contactsurface is small, sometimes the redseg_contactsurface
            # consists only of the redseg_contactcorners. In this case there will
            # be only 1 piece (the noncontactborders) when the contactcorners are
            # removed. Therefore add the redseg_contactcorners to each possible
            # surface:
            borders1 = (grnseg_outer_contactsurface == 1) + grnseg_outer_corners.astype(bool)
            borders2 = (grnseg_outer_contactsurface == 2) + grnseg_outer_corners.astype(bool)


            # Compare similarity between label 1 and label 2:
            ssim_borders1 = structural_similarity(borders1, contactsurface[:,:,0].astype(bool))
            ssim_borders2 = structural_similarity(borders2.astype(bool), contactsurface[:,:,0].astype(bool))
            # Classify the more similar label as being contactsurface,
            # put the noncontactborders in a new array and remove the 
            # noncontactborders from the contactsurface array:
            if ssim_borders2 > ssim_borders1:
                grnseg_outer_noncontactborders[np.where(borders2)] = False
                grnseg_outer_noncontactborders = grnseg_outer_noncontactborders.astype(bool)
                grnseg_outer_contactsurface[np.where(borders1)] = False
                grnseg_outer_contactsurface = grnseg_outer_contactsurface.astype(bool)
                grnseg_outer_contactsurface[np.where(grnseg_outer_corners)] = True
            elif ssim_borders1 > ssim_borders2:
                grnseg_outer_noncontactborders[np.where(borders1)] = False
                grnseg_outer_noncontactborders = grnseg_outer_noncontactborders.astype(bool)
                grnseg_outer_contactsurface[np.where(borders2)] = False
                grnseg_outer_contactsurface = grnseg_outer_contactsurface.astype(bool)
                grnseg_outer_contactsurface[np.where(grnseg_outer_corners)] = True
            else:
                grnseg_outer_contactsurface = np.zeros(grnseg_outer_contactsurface.shape)
                grnseg_outer_noncontactborders = np.zeros(grnseg_outer_contactsurface.shape)

            # Combine the inner and outer contactsurface back together.
            # Also do the same for inner and outer noncontactborders.
            grnseg_contactsurface = np.logical_or(grnseg_inner_contactsurface, grnseg_outer_contactsurface)
            grnseg_noncontactborders = np.logical_or(grnseg_inner_noncontactborders, grnseg_outer_noncontactborders)

            # Restore any removed pixels from the clean_up_borders function to
            # either the contactsurface or noncontactborders that end up being
            # right in between the inner and outer borders:
            grnseg_contactsurface_filled = morphology.remove_small_holes(grnseg_contactsurface)
            grnseg_noncontactborders_filled = morphology.remove_small_holes(grnseg_noncontactborders)
            
            
            # Return pixels in sequential order for inner and outer contactsurface:
            grnseg_inner_contactsurface = np.dstack((grnseg_inner_contactsurface, np.zeros(grnseg_inner_contactsurface.shape), np.zeros(grnseg_inner_contactsurface.shape)))
            grnseg_inner_contactsurface_seq, blu_unsused, grn_unused_perim, blu_unused_perim = make_open_border_indices_sequential(grnseg_inner_contactsurface, red_id, blu_id, 0, 2) #grn_id
            grnseg_outer_contactsurface = np.dstack((grnseg_outer_contactsurface, np.zeros(grnseg_outer_contactsurface.shape), np.zeros(grnseg_outer_contactsurface.shape)))
            grnseg_outer_contactsurface_seq, blu_unused, grn_unused_perim, blue_unused_perim = make_open_border_indices_sequential(grnseg_outer_contactsurface, red_id, blu_id, 0, 2) #grn_id
            
            # Also do same for inner and outer noncontactborders:
            grnseg_inner_noncontactborders = np.dstack((grnseg_inner_noncontactborders, np.zeros(grnseg_inner_noncontactborders.shape), np.zeros(grnseg_inner_noncontactborders.shape)))
            grnseg_inner_noncontactborders_seq, blu_unused, grn_unused_perim, blu_unused_perim = make_open_border_indices_sequential(grnseg_inner_noncontactborders, red_id, blu_id, 0, 2) #grn_id
            grnseg_outer_noncontactborders = np.dstack((grnseg_outer_noncontactborders, np.zeros(grnseg_outer_noncontactborders.shape), np.zeros(grnseg_outer_noncontactborders.shape)))
            grnseg_outer_noncontactborders_seq, blu_unused, grn_unused_perim, blu_unused_perim = make_open_border_indices_sequential(grnseg_outer_noncontactborders, red_id, blu_id, 0, 2) #grn_id
      
            
            
            # Find bluseg inner and outer border pixels that are closest to the
            # contact corners of blu_areas:
            bluseg_border_inner = np.dstack(( mh.borders(bluseg_clean) * bluseg_clean, np.zeros(bluseg_clean.shape), np.zeros(bluseg_clean.shape) ))
            bluseg_border_clean_inner = clean_up_borders(bluseg_border_inner)[:,:,0]
            
            bluseg_border_outer = np.dstack(( mh.borders(bluseg_clean) * ~bluseg_clean, np.zeros(bluseg_clean.shape), np.zeros(bluseg_clean.shape) ))
            bluseg_border_clean_outer = clean_up_borders(bluseg_border_outer)[:,:,0]
            
            bluseg_inner_corner_locs = []
            bluseg_outer_corner_locs = []
            for corner in zip(*np.where(contactcorners[:, :, channel2])):
                bluseg_inner_distances  = np.ma.array((np.where(bluseg_border_clean_inner)[0] - corner[0]) ** 2 + (np.where(bluseg_border_clean_inner)[1] - corner[1]) ** 2)
                bluseg_inner_closest_iloc = np.argmin(bluseg_inner_distances) 
                bluseg_inner_corner_indice = list(zip(*np.where(bluseg_border_clean_inner)))[bluseg_inner_closest_iloc]
                if bluseg_inner_corner_indice in bluseg_inner_corner_locs:
                    #print("This pixels has already been recorded!")
                    bluseg_inner_distances[bluseg_inner_closest_iloc] = np.ma.masked
                    bluseg_inner_closest_iloc = np.argmin(bluseg_inner_distances) 
                    bluseg_inner_corner_indice = list(zip(*np.where(bluseg_border_clean_inner)))[bluseg_inner_closest_iloc]
                bluseg_inner_corner_locs.append(bluseg_inner_corner_indice)
                
                bluseg_outer_distances = np.ma.array((np.where(bluseg_border_clean_outer)[0] - corner[0]) ** 2 + (np.where(bluseg_border_clean_outer)[1] - corner[1]) ** 2)
                bluseg_outer_closest_iloc = np.argmin(bluseg_outer_distances) 
                bluseg_outer_corner_indice = list(zip(*np.where(bluseg_border_clean_outer)))[bluseg_outer_closest_iloc]
                if bluseg_outer_corner_indice in bluseg_outer_corner_locs:
                    #print("This pixels has already been recorded!")
                    bluseg_outer_distances[bluseg_outer_closest_iloc] = np.ma.masked
                    bluseg_outer_closest_iloc = np.argmin(bluseg_outer_distances) 
                    bluseg_outer_corner_indice = list(zip(*np.where(bluseg_border_clean_outer)))[bluseg_outer_closest_iloc]
                bluseg_outer_corner_locs.append(bluseg_outer_corner_indice)

            bluseg_inner_corners = np.zeros(bluseg_clean.shape)
            bluseg_inner_corners[tuple(np.asarray(a) for a in zip(*bluseg_inner_corner_locs))] = True
            bluseg_outer_corners = np.zeros(bluseg_clean.shape)
            bluseg_outer_corners[tuple(np.asarray(a) for a in zip(*bluseg_outer_corner_locs))] = True
            
            bluseg_inner_contactsurface = np.copy(bluseg_border_clean_inner)
            bluseg_inner_noncontactborders = np.copy(bluseg_border_clean_inner)
            # Break the bluseg_border_clean_inner at the contact corners:
            bluseg_inner_contactsurface[np.where(bluseg_inner_corners)] = 0
            bluseg_inner_noncontactborders[np.where(bluseg_inner_corners)] = 0
            
            # Label the resulting 2 pieces so that they can be classified as either
            # contactsurface or noncontactborders:
            bluseg_inner_contactsurface = morphology.label(bluseg_inner_contactsurface, connectivity=2)
            if len(np.unique(bluseg_inner_contactsurface)) <= 3:
                pass
            else:
                bluseg_inner_contactsurface = np.zeros(bluseg_inner_contactsurface.shape)
            
            # When the contactsurface is small, sometimes the redseg_contactsurface
            # consists only of the redseg_contactcorners. In this case there will
            # be only 1 piece (the noncontactborders) when the contactcorners are
            # removed. Therefore add the redseg_contactcorners to each possible
            # surface:
            borders1 = (bluseg_inner_contactsurface == 1) + bluseg_inner_corners.astype(bool)
            borders2 = (bluseg_inner_contactsurface == 2) + bluseg_inner_corners.astype(bool)
            
            # Compare similarity between label 1 and label 2:
            #ssim_borders1 = structural_similarity((redseg_inner_contactsurface == 1).astype(bool), contactsurface[:,:,0].astype(bool))
            #ssim_borders2 = structural_similarity((redseg_inner_contactsurface == 2).astype(bool), contactsurface[:,:,0].astype(bool))
            ssim_borders1 = structural_similarity(borders1, contactsurface[:,:,channel2].astype(bool))
            ssim_borders2 = structural_similarity(borders2, contactsurface[:,:,channel2].astype(bool))
            # Classify the more similar label as being contactsurface,
            # put the noncontactborders in a new array and remove the 
            # noncontactborders from the contactsurface array:
            if ssim_borders2 > ssim_borders1:
                bluseg_inner_noncontactborders[np.where(borders2)] = False
                bluseg_inner_noncontactborders = bluseg_inner_noncontactborders.astype(bool)
                bluseg_inner_contactsurface[np.where(borders1)] = False
                bluseg_inner_contactsurface = bluseg_inner_contactsurface.astype(bool)
                bluseg_inner_contactsurface[np.where(bluseg_inner_corners)] = True
            elif ssim_borders1 > ssim_borders2:
                bluseg_inner_noncontactborders[np.where(borders1)] = False
                bluseg_inner_noncontactborders = bluseg_inner_noncontactborders.astype(bool)
                bluseg_inner_contactsurface[np.where(borders2)] = False
                bluseg_inner_contactsurface = bluseg_inner_contactsurface.astype(bool)
                bluseg_inner_contactsurface[np.where(bluseg_inner_corners)] = True
            else:
                bluseg_inner_contactsurface = np.zeros(bluseg_inner_contactsurface.shape)
                bluseg_inner_noncontactborders = np.zeros(bluseg_inner_contactsurface.shape)
            
            # Do the same for the outer borders:
            bluseg_outer_contactsurface = np.copy(bluseg_border_clean_outer)
            bluseg_outer_noncontactborders = np.copy(bluseg_border_clean_outer)
            
            bluseg_outer_contactsurface[np.where(bluseg_outer_corners)] = False
            bluseg_outer_noncontactborders[np.where(bluseg_outer_corners)] = False

            bluseg_outer_contactsurface = morphology.label(bluseg_outer_contactsurface, connectivity=2)
            if len(np.unique(bluseg_outer_contactsurface)) <= 3:
                pass
            else:
                bluseg_outer_contactsurface = np.zeros(bluseg_outer_contactsurface.shape)
            
            # When the contactsurface is small, sometimes the redseg_contactsurface
            # consists only of the redseg_contactcorners. In this case there will
            # be only 1 piece (the noncontactborders) when the contactcorners are
            # removed. Therefore add the redseg_contactcorners to each possible
            # surface:
            borders1 = (bluseg_outer_contactsurface == 1) + bluseg_outer_corners.astype(bool)
            borders2 = (bluseg_outer_contactsurface == 2) + bluseg_outer_corners.astype(bool)


            # Compare similarity between label 1 and label 2:
            ssim_borders1 = structural_similarity(borders1, contactsurface[:,:,channel2].astype(bool))
            ssim_borders2 = structural_similarity(borders2.astype(bool), contactsurface[:,:,channel2].astype(bool))
            # Classify the more similar label as being contactsurface,
            # put the noncontactborders in a new array and remove the 
            # noncontactborders from the contactsurface array:
            if ssim_borders2 > ssim_borders1:
                bluseg_outer_noncontactborders[np.where(borders2)] = False
                bluseg_outer_noncontactborders = bluseg_outer_noncontactborders.astype(bool)
                bluseg_outer_contactsurface[np.where(borders1)] = False
                bluseg_outer_contactsurface = bluseg_outer_contactsurface.astype(bool)
                bluseg_outer_contactsurface[np.where(bluseg_outer_corners)] = True
            elif ssim_borders1 > ssim_borders2:
                bluseg_outer_noncontactborders[np.where(borders1)] = False
                bluseg_outer_noncontactborders = bluseg_outer_noncontactborders.astype(bool)
                bluseg_outer_contactsurface[np.where(borders2)] = False
                bluseg_outer_contactsurface = bluseg_outer_contactsurface.astype(bool)
                bluseg_outer_contactsurface[np.where(bluseg_outer_corners)] = True
            else:
                bluseg_outer_contactsurface = np.zeros(bluseg_outer_contactsurface.shape)
                bluseg_outer_noncontactborders = np.zeros(bluseg_outer_contactsurface.shape)

            # Combine the inner and outer contactsurface back together.
            # Also do the same for inner and outer noncontactborders.
            bluseg_contactsurface = np.logical_or(bluseg_inner_contactsurface, bluseg_outer_contactsurface)
            bluseg_noncontactborders = np.logical_or(bluseg_inner_noncontactborders, bluseg_outer_noncontactborders)

            # Restore any removed pixels from the clean_up_borders function to
            # either the contactsurface or noncontactborders that end up being
            # right in between the inner and outer borders:
            bluseg_contactsurface_filled = morphology.remove_small_holes(bluseg_contactsurface)
            bluseg_noncontactborders_filled = morphology.remove_small_holes(bluseg_noncontactborders)
            
            
            # Return pixels in sequential order for inner and outer contactsurface:
            bluseg_inner_contactsurface = np.dstack((bluseg_inner_contactsurface, np.zeros(bluseg_inner_contactsurface.shape), np.zeros(bluseg_inner_contactsurface.shape)))
            bluseg_inner_contactsurface_seq, blu_unsused, grn_unused_perim, blu_unused_perim = make_open_border_indices_sequential(bluseg_inner_contactsurface, red_id, blu_id, 0, 2) #grn_id
            bluseg_outer_contactsurface = np.dstack((bluseg_outer_contactsurface, np.zeros(bluseg_outer_contactsurface.shape), np.zeros(bluseg_outer_contactsurface.shape)))
            bluseg_outer_contactsurface_seq, blu_unused, grn_unused_perim, blue_unused_perim = make_open_border_indices_sequential(bluseg_outer_contactsurface, red_id, blu_id, 0, 2) #grn_id
            
            # Also do same for inner and outer noncontactborders:
            bluseg_inner_noncontactborders = np.dstack((bluseg_inner_noncontactborders, np.zeros(bluseg_inner_noncontactborders.shape), np.zeros(bluseg_inner_noncontactborders.shape)))
            bluseg_inner_noncontactborders_seq, blu_unused, grn_unused_perim, blu_unused_perim = make_open_border_indices_sequential(bluseg_inner_noncontactborders, red_id, blu_id, 0, 2) #grn_id
            bluseg_outer_noncontactborders = np.dstack((bluseg_outer_noncontactborders, np.zeros(bluseg_outer_noncontactborders.shape), np.zeros(bluseg_outer_noncontactborders.shape)))
            bluseg_outer_noncontactborders_seq, blu_unused, grn_unused_perim, blu_unused_perim = make_open_border_indices_sequential(bluseg_outer_noncontactborders, red_id, blu_id, 0, 2) #grn_id

            
            
            # Return fluorescence in red and grn channel sequentially:
            redchan_fluor_at_contact_inner = redchan[tuple(np.asarray(a) for a in zip(*redseg_inner_contactsurface_seq))[:2]]
            redchan_fluor_at_contact_outer = redchan[tuple(np.asarray(a) for a in zip(*redseg_outer_contactsurface_seq))[:2]]
            grnchan_fluor_at_contact_inner = grnchan[tuple(np.asarray(a) for a in zip(*grnseg_inner_contactsurface_seq))[:2]]
            grnchan_fluor_at_contact_outer = grnchan[tuple(np.asarray(a) for a in zip(*grnseg_outer_contactsurface_seq))[:2]]
            bluchan_fluor_at_contact_inner = bluchan[tuple(np.asarray(a) for a in zip(*bluseg_inner_contactsurface_seq))[:2]]
            bluchan_fluor_at_contact_outer = bluchan[tuple(np.asarray(a) for a in zip(*bluseg_outer_contactsurface_seq))[:2]]
            
            redchan_fluor_outside_contact_inner = redchan[tuple(np.asarray(a) for a in zip(*redseg_inner_noncontactborders_seq))[:2]]
            redchan_fluor_outside_contact_outer = redchan[tuple(np.asarray(a) for a in zip(*redseg_outer_noncontactborders_seq))[:2]]
            grnchan_fluor_outside_contact_inner = grnchan[tuple(np.asarray(a) for a in zip(*grnseg_inner_noncontactborders_seq))[:2]]
            grnchan_fluor_outside_contact_outer = grnchan[tuple(np.asarray(a) for a in zip(*grnseg_outer_noncontactborders_seq))[:2]]
            bluchan_fluor_outside_contact_inner = bluchan[tuple(np.asarray(a) for a in zip(*bluseg_inner_noncontactborders_seq))[:2]]
            bluchan_fluor_outside_contact_outer = bluchan[tuple(np.asarray(a) for a in zip(*bluseg_outer_noncontactborders_seq))[:2]]
            
            # Sequentially ordered borders miss pixels between the inner and 
            # outer borders, so also take fluorescence in red and grn channel
            # the unordered borders which have those pixels:
            redchan_fluor_at_contact = redchan[np.where(redseg_contactsurface_filled)]
            grnchan_fluor_at_contact = grnchan[np.where(grnseg_contactsurface_filled)]
            bluchan_fluor_at_contact = bluchan[np.where(bluseg_contactsurface_filled)]
            
            redchan_fluor_outside_contact = redchan[np.where(redseg_noncontactborders_filled)]
            grnchan_fluor_outside_contact = grnchan[np.where(grnseg_noncontactborders_filled)]
            bluchan_fluor_outside_contact = bluchan[np.where(bluseg_noncontactborders_filled)]
            
            
            # Also get un-ordered cytoplasmic fluorescence values:
            redseg_cytopl_area = np.logical_xor(redseg_border_clean_inner, scp_morph.binary_fill_holes(redseg_border_clean_inner))
            grnseg_cytopl_area = np.logical_xor(grnseg_border_clean_inner, scp_morph.binary_fill_holes(grnseg_border_clean_inner))
            bluseg_cytopl_area = np.logical_xor(bluseg_border_clean_inner, scp_morph.binary_fill_holes(bluseg_border_clean_inner))

            redchan_fluor_cytoplasmic = redchan[redseg_cytopl_area]
            grnchan_fluor_cytoplasmic = grnchan[grnseg_cytopl_area]
            bluchan_fluor_cytoplasmic = bluchan[bluseg_cytopl_area]


            # Sample the intensity values at the contactsurface and at the
            # noncontactborders:
            #plt.hist(redchan[np.where(redseg_contactsurface_filled)], bins=redchan[np.where(redseg_contactsurface_filled)].max(), color='r', alpha=0.5)
            #plt.hist(redchan[np.where(redseg_noncontactborders_filled)], bins=redchan[np.where(redseg_noncontactborders_filled)].max(), color='r', alpha=0.5)
            #print( np.mean(redchan[np.where(redseg_contactsurface_filled)]) )
            #print( np.mean(redchan[np.where(redseg_noncontactborders_filled)]) )


            
            ### Now also sample the green channel:
            #plt.hist(grnchan[np.where(redseg_contactsurface_filled)], bins=grnchan[np.where(redseg_contactsurface_filled)].max(), color='g', alpha=0.5)
            #plt.hist(grnchan[np.where(redseg_noncontactborders_filled)], bins=grnchan[np.where(redseg_noncontactborders_filled)].max(), color='g', alpha=0.5)
            #print( np.mean(grnchan[np.where(redseg_contactsurface_filled)]) )
            #print( np.mean(grnchan[np.where(redseg_noncontactborders_filled)]) )

            # The fluorescence values of the contactsurface and noncontactsurface
            # for the red and green channels should be placed in their own
            # columns in the data table at the end (4 columns total):
            
            
            # Also you can just get rid of the user input prompt for the minimum
            # background threshold and objects of interest




            ### Re-draw the figure with the contact angles, contact surface, etc.

            ### Draw the angles over top of the clean overlay in a figure:
            ### need to collect and organize all the corresponding vector start x values,
            ### y values, and vector x values, y values for usage in ax.quiver:
            vector_start_x_vals = (vector1_start[0], vector2_start[0], vector3_start[0], vector4_start[0])
            vector_start_y_vals = (vector1_start[1], vector2_start[1], vector3_start[1], vector4_start[1])
            vector_x_vals = (vector1[0], vector2[0], vector3[0], vector4[0])
            vector_y_vals = (vector1[1], vector2[1], vector3[1], vector4[1])
            
            ang1_annot = ((vector1 + vector2)/2.0)        
            ang1_annot_mag = np.sqrt(ang1_annot.dot(ang1_annot))
            if ang1_annot_mag == 0:
                ang1_annot_mag = 10**(-12)
    
            ang2_annot = ((vector3 + vector4)/2.0)
            ang2_annot_mag = np.sqrt(ang2_annot.dot(ang2_annot))
            if ang2_annot_mag == 0:
                ang2_annot_mag = 10**(-12)
    
            ### Draw a line between the contact corners of the red cell:
            contact_surface_line_red_rcd = [(vector1_start[1], vector3_start[1]), (vector1_start[0], vector3_start[0])]
            ### Draw a line between the contact corners of the green cell:
            contact_surface_line_blu_rcd = [(vector2_start[1], vector4_start[1]), (vector2_start[0], vector4_start[0])]
            ### draw a line that is perpendicular to the contact surface line, in the middle
            ### of the contact surface:
            temp1_rcd = (vector1_start[::-1] + vector2_start[::-1])/2.0
            temp2_rcd = (vector3_start[::-1] + vector4_start[::-1])/2.0
            mean_contact_surface_line_rcd = (temp1_rcd + temp2_rcd)/2.0
            center_of_contact_surface_lines_rc = mean_contact_surface_line_rcd[:2]

            ### Create an overlay of the original image and the borders for
            ### checking during coding:
            cleanoverlay = mh.overlay(tlchan, red=cleanborders[:,:,0], green=cleanborders[:,:,1], blue=cleanborders[:,:,2])

            ### Draw what you've found over the images of the cells:
            fig.clf()
            fig = plt.figure(num=str(filenumber+1)+'_'+os.path.basename(filename), figsize=(16., 10.))
            plt.suptitle(str(filenumber+1)+' - '+os.path.basename(filename), verticalalignment='center', x=0.5, y=0.5, fontsize=14)
            ax1 = fig.add_subplot(241)
            ax1.yaxis.set_visible(False)
            ax1.xaxis.set_visible(False)
            ax1.imshow(tlchan, cmap='gray')
            
            ax2 = fig.add_subplot(242)
            ax2.yaxis.set_visible(False)
            ax2.xaxis.set_visible(False)
            #figcellborderred = ax2.imshow(mh.overlay(gray = mh.stretch(tlchan), red = cleanborders[:,:,0]))
            fig_noncont_redcell = ax2.imshow(mh.overlay(gray = mh.stretch(redchan), red = redseg_noncontactborders))
            ax2.axes.autoscale(False)
            #ax2.scatter(cell_max_rcd[0][1], cell_max_rcd[0][0], color='m', marker='o')
            
            ax3 = fig.add_subplot(243)
            ax3.yaxis.set_visible(False)
            ax3.xaxis.set_visible(False)
            #figcellborderblu = ax3.imshow(mh.overlay(gray = mh.stretch(tlchan), blue = cleanborders[:,:,2]))
            fig_noncont_grncell = ax3.imshow(mh.overlay(gray = mh.stretch(grnchan), green = grnseg_noncontactborders))
            ax3.axes.autoscale(False)
            #ax3.scatter(cell_max_rcd[1][1], cell_max_rcd[1][0], color='m', marker='o')
            
            ax4 = fig.add_subplot(246)
            ax4.yaxis.set_visible(False)
            ax4.xaxis.set_visible(False)
            #ax4.imshow(tlchan, cmap='gray')
            fig_cont_redcell = ax4.imshow(mh.overlay(gray = mh.stretch(redchan), red = redseg_contactsurface))
            ax4.axes.autoscale(False)
            #drawn_arrows = ax4.quiver(vector_start_x_vals, vector_start_y_vals, vector_x_vals, vector_y_vals, units='xy',scale_units='xy', angles='xy', scale=0.33, color='c', width = 1.0)
            #angle1_annotations = ax4.annotate('%.1f' % degree_angle1, vector1_start, vector1_start + 30*(ang1_annot / ang1_annot_mag)*(1,1), color='c', fontsize='24', ha='center', va='center')
            #angle2_annotations = ax4.annotate('%.1f' % degree_angle2, vector3_start, vector3_start + 30*(ang2_annot / ang2_annot_mag)*(1,1), color='c', fontsize='24', ha='center', va='center')
            #redcell_contactvect = ax4.plot(contact_surface_line_red_rcd[1], contact_surface_line_red_rcd[0], color='c', lw=1)
            #grncell_contactvect = ax4.plot(contact_surface_line_blu_rcd[1], contact_surface_line_blu_rcd[0], color='c', lw=1)
            #contactvect = ax4.plot([mean_contact_surface_line_rcd[1]-2*contact_vector[0], mean_contact_surface_line_rcd[1]+2*contact_vector[0]], [mean_contact_surface_line_rcd[0]-2*contact_vector[1], mean_contact_surface_line_rcd[0]+2*contact_vector[1]], color='m')
            
            ax5 = fig.add_subplot(247)
            ax5.yaxis.set_visible(False)
            ax5.xaxis.set_visible(False)
            #ax5.imshow(mh.overlay(gray = mh.stretch(tlchan), red=contactsurface[:,:,0], blue=contactsurface[:,:,2]))
            fig_cont_grncell = ax5.imshow(mh.overlay(gray = mh.stretch(grnchan), green = grnseg_contactsurface))
            ax5.axes.autoscale(False)
    
            ax6 = fig.add_subplot(245)
            ax6.yaxis.set_visible(False)
            ax6.xaxis.set_visible(False)
            ax6.imshow(cleanoverlay)
            ax6.axes.autoscale(False)
            ax6.scatter(cell_max_rcd[1][1], cell_max_rcd[1][0], color='m', marker='o')
            ax6.scatter(cell_max_rcd[0][1], cell_max_rcd[0][0], color='m', marker='o')
            ax6.scatter(bg_max_rcd[0][1], bg_max_rcd[0][0], color='m', marker='x')
            ax6.scatter(bg_max_rcd[1][1], bg_max_rcd[1][0], color='m', marker='x')
            drawn_arrows = ax6.quiver(vector_start_x_vals, vector_start_y_vals, vector_x_vals, vector_y_vals, units='xy',scale_units='xy', angles='xy', scale=0.33, color='c', width = 1.0)
            angle1_annotations = ax6.annotate('%.1f' % degree_angle1, vector1_start, vector1_start + 30*(ang1_annot / ang1_annot_mag)*(1,1), color='c', fontsize='24', ha='center', va='center')
            angle2_annotations = ax6.annotate('%.1f' % degree_angle2, vector3_start, vector3_start + 30*(ang2_annot / ang2_annot_mag)*(1,1), color='c', fontsize='24', ha='center', va='center')
            redcell_contactvect = ax6.plot(contact_surface_line_red_rcd[1], contact_surface_line_red_rcd[0], color='c', lw=1)
            grncell_contactvect = ax6.plot(contact_surface_line_blu_rcd[1], contact_surface_line_blu_rcd[0], color='c', lw=1)
            contactvect = ax6.plot([mean_contact_surface_line_rcd[1]-2*contact_vector[0], mean_contact_surface_line_rcd[1]+2*contact_vector[0]], [mean_contact_surface_line_rcd[0]-2*contact_vector[1], mean_contact_surface_line_rcd[0]+2*contact_vector[1]], color='m')

            ax7 = fig.add_subplot(244)
            ax7.yaxis.set_visible(False)
            ax7.xaxis.set_visible(False)
            ax_noncont_blucell = ax7.imshow(mh.overlay(mh.stretch(bluchan), blue=bluseg_noncontactborders))
            ax7.axes.autoscale(False)
            
            ax8 = fig.add_subplot(248)
            ax8.yaxis.set_visible(False)
            ax8.xaxis.set_visible(False)
            fig_cont_blucell = ax8.imshow(mh.overlay(mh.stretch(bluchan), blue=bluseg_contactsurface))
            ax8.axes.autoscale(False)


            plt.tight_layout()
            plt.draw
            #plt.show('block' == False)
    
            listupdate()



### Make a scatter plot of your angles:
contact_angle_fig = plt.figure('Contact Angles', facecolor='white', figsize=(10,10))
ax_contact_angle = contact_angle_fig.add_subplot(1,1,1)
contact_angle_scatter = plt.scatter(cell_pair_num_list, average_angle_list)
ax_contact_angle.set_xlabel('Cell Pair Number', fontsize = 24)
ax_contact_angle.set_ylabel('Average Angle', fontsize = 24)
ax_contact_angle.set_ylim(0)
ax_contact_angle.set_xlim(0, cell_pair_num_list[-1] + 1)
plt.tight_layout()

### Make a scatter plot of your contact lengths:
contact_len_fig = plt.figure('Contact Lengths', facecolor='white', figsize=(10,10))
ax_contact_len = contact_len_fig.add_subplot(111)
contact_len_scatter = plt.scatter(cell_pair_num_list, contactsurface_list)
ax_contact_len.set_xlabel('Cell Pair Number', fontsize = 24)
ax_contact_len.set_ylabel('Diameter of Contact Surface', fontsize = 24)
ax_contact_len.set_ylim(0)
ax_contact_len.set_xlim(0, cell_pair_num_list[-1] + 1)
plt.tight_layout()

### Make a plot of fluorescence at contact vs fluorescence at free surface for
### the green channel:
fig_fluor_raw, ax_fluor_raw = plt.subplots(nrows=1, ncols=1, figsize=(10,10))
#rel_adhesiveness_fig = plt.figure('Fluorescence at cortex in green channel', facecolor='white', figsize=(10,10))
#ax_rel_adhesiveness = rel_adhesiveness_fig.add_subplot(111)
noncontact_tension_scatter = ax_fluor_raw.errorbar(cell_pair_num_list, [np.mean(x) for x in grnchan_fluor_outside_contact_list], yerr=[np.std(x) for x in grnchan_fluor_outside_contact_list], c='g', ecolor='g', marker='+', alpha=0.5)
noncontact_tension_plot = ax_fluor_raw.plot(cell_pair_num_list, [np.mean(x) for x in grnchan_fluor_outside_contact_list], c='g')
contact_tension_scatter = ax_fluor_raw.errorbar(cell_pair_num_list, [np.mean(x) for x in grnchan_fluor_at_contact_list], yerr=[np.std(x) for x in grnchan_fluor_at_contact_list], c='c', ecolor='c', marker='x', alpha=0.5)
contact_tension_plot = ax_fluor_raw.plot(cell_pair_num_list, [np.mean(x) for x in grnchan_fluor_at_contact_list], c='c')

noncontact_tension_scatter = ax_fluor_raw.errorbar(cell_pair_num_list, [np.mean(x) for x in redchan_fluor_outside_contact_list], yerr=[np.std(x) for x in redchan_fluor_outside_contact_list], c='r', ecolor='r', marker='+', alpha=0.5)
noncontact_tension_plot = ax_fluor_raw.plot(cell_pair_num_list, [np.mean(x) for x in redchan_fluor_outside_contact_list], c='r')
contact_tension_scatter = ax_fluor_raw.errorbar(cell_pair_num_list, [np.mean(x) for x in redchan_fluor_at_contact_list], yerr=[np.std(x) for x in redchan_fluor_at_contact_list], c='m', ecolor='m', marker='x', alpha=0.5)
contact_tension_plot = ax_fluor_raw.plot(cell_pair_num_list, [np.mean(x) for x in redchan_fluor_at_contact_list], c='m')
ax_fluor_raw.set_xlabel('Cell Pair Number', fontsize = 24)
ax_fluor_raw.set_ylabel('Fluorescence (photons)', fontsize = 24)
ax_fluor_raw.set_ylim(0)
ax_fluor_raw.set_xlim(0, cell_pair_num_list[-1] + 1)

#rel_adhesiveness_scatter = plt.scatter(cell_pair_num_list, )
#ax_rel_adhesiveness.set_xlabel('Cell Pair Number', fontsize = 24)
#ax_rel_adhesiveness.set_ylabel('LifeActGFP Fluorescence (A.U.)', fontsize = 24)
#ax_rel_adhesiveness.set_ylim(0)
#ax_rel_adhesiveness.set_xlim(0, cell_pair_num_list[-1] + 1)
plt.tight_layout()

### And also do the same for the red channel:
#rel_adhesiveness_fig = plt.figure('Fluorescence at cortex in green channel', facecolor='white', figsize=(10,10))
#ax_rel_adhesiveness = rel_adhesiveness_fig.add_subplot(111)
#noncontact_tension_scatter = ax_rel_adhesiveness.scatter(cell_pair_num_list, [np.mean(x) for x in redchan_fluor_outside_contact_list], c='r', marker='+')
#noncontact_tension_plot = ax_rel_adhesiveness.plot(cell_pair_num_list, [np.mean(x) for x in redchan_fluor_outside_contact_list], c='r')
#contact_tension_scatter = ax_rel_adhesiveness.scatter(cell_pair_num_list, [np.mean(x) for x in redchan_fluor_at_contact_list], c='m', marker='x')
#contact_tension_plot = ax_rel_adhesiveness.plot(cell_pair_num_list, [np.mean(x) for x in redchan_fluor_at_contact_list], c='m')
#ax_rel_adhesiveness.set_xlabel('Cell Pair Number', fontsize = 24)
#ax_rel_adhesiveness.set_ylabel('Fluorescence (photons)', fontsize = 24)
#ax_rel_adhesiveness.set_ylim(0)
#ax_rel_adhesiveness.set_xlim(0, cell_pair_num_list[-1] + 1)
#plt.tight_layout()

### Plot normalized fluorescence and relative fluorescence:
mean_rel_fluor_red = [np.mean(redchan_fluor_at_contact_list[i-1]) / np.mean(redchan_fluor_outside_contact_list[i-1]) for i in cell_pair_num_list]
mean_rel_fluor_grn = [np.mean(grnchan_fluor_at_contact_list[i-1]) / np.mean(grnchan_fluor_outside_contact_list[i-1]) for i in cell_pair_num_list]
norm_mean_rel_fluor_grn = np.asarray(mean_rel_fluor_grn) / np.asarray(mean_rel_fluor_red)

median_rel_fluor_red = [np.median(redchan_fluor_at_contact_list[i-1]) / np.median(redchan_fluor_outside_contact_list[i-1]) for i in cell_pair_num_list]
median_rel_fluor_grn = [np.median(grnchan_fluor_at_contact_list[i-1]) / np.median(grnchan_fluor_outside_contact_list[i-1]) for i in cell_pair_num_list]
norm_median_rel_fluor_grn = np.asarray(median_rel_fluor_grn) / np.asarray(median_rel_fluor_red)


norm_rel_adhesiveness_fig = plt.figure( facecolor='white', figsize=(10,10))
ax_norm_rel_adhesiveness = norm_rel_adhesiveness_fig.add_subplot(111)
cont_over_noncont_plot = ax_norm_rel_adhesiveness.plot(cell_pair_num_list, mean_rel_fluor_grn, c='g')
cont_over_noncont_scatter = ax_norm_rel_adhesiveness.scatter(cell_pair_num_list, mean_rel_fluor_grn, c='g', marker='.')
cont_over_noncont_plot = ax_norm_rel_adhesiveness.plot(cell_pair_num_list, mean_rel_fluor_red, c='r')
cont_over_noncont_scatter = ax_norm_rel_adhesiveness.scatter(cell_pair_num_list, mean_rel_fluor_red, c='r', marker='.')
cont_over_noncont_plot = ax_norm_rel_adhesiveness.plot(cell_pair_num_list, norm_mean_rel_fluor_grn, c='k')
cont_over_noncont_scatter = ax_norm_rel_adhesiveness.scatter(cell_pair_num_list, norm_mean_rel_fluor_grn, c='k', marker='.')
ax_norm_rel_adhesiveness.set_xlabel('Cell Pair Number', fontsize = 24)
ax_norm_rel_adhesiveness.set_ylabel('Normalized Mean Relative Fluorescence (A.U.)', fontsize = 24)
ax_norm_rel_adhesiveness.set_ylim(0)
ax_norm_rel_adhesiveness.set_xlim(0, cell_pair_num_list[-1] + 1)
plt.tight_layout()

norm_median_rel_adhesiveness_fig = plt.figure( facecolor='white', figsize=(10,10))
ax_norm_median_rel_adhesiveness = norm_median_rel_adhesiveness_fig.add_subplot(111)
median_cont_over_noncont_plot = ax_norm_median_rel_adhesiveness.plot(cell_pair_num_list, median_rel_fluor_grn, c='g')
median_cont_over_noncont_scatter = ax_norm_median_rel_adhesiveness.scatter(cell_pair_num_list, median_rel_fluor_grn, c='g', marker='.')
median_cont_over_noncont_plot = ax_norm_median_rel_adhesiveness.plot(cell_pair_num_list, median_rel_fluor_red, c='r')
median_cont_over_noncont_scatter = ax_norm_median_rel_adhesiveness.scatter(cell_pair_num_list, median_rel_fluor_red, c='r', marker='.')
median_cont_over_noncont_plot = ax_norm_median_rel_adhesiveness.plot(cell_pair_num_list, norm_median_rel_fluor_grn, c='k')
median_cont_over_noncont_scatter = ax_norm_median_rel_adhesiveness.scatter(cell_pair_num_list, norm_median_rel_fluor_grn, c='k', marker='.')
ax_norm_median_rel_adhesiveness.set_xlabel('Cell Pair Number', fontsize = 24)
ax_norm_median_rel_adhesiveness.set_ylabel('Normalized Median Relative Fluorescence (A.U.)', fontsize = 24)
ax_norm_median_rel_adhesiveness.set_ylim(0)
ax_norm_median_rel_adhesiveness.set_xlim(0, cell_pair_num_list[-1] + 1)
plt.tight_layout()


while True:
    save_table = input("Would you like to save the output?"
                           "(Y/N)?\n>"
                           )
    if save_table == 'y' or save_table == 'Y' or save_table == 'yes':
        
        num_bins = 8
        ### Turn the curvature details into arrays:
        bin_curvature_details_array_red = np.asarray(bin_curvature_details_red_list, dtype='object')
        bin_curvature_details_array_blu = np.asarray(bin_curvature_details_blu_list, dtype='object')
        whole_curvature_details_red_array = np.asarray(whole_cell_curvature_details_red_list, dtype='object')
        whole_curvature_details_blu_array = np.asarray(whole_cell_curvature_details_blu_list, dtype='object')

        ### Give me a list of the curvature value columns:
        bin_curv_val_red_list = [list(bin_curvature_details_array_red[:,i,2]) for i in range(1,bin_curvature_details_array_red.shape[1])]
        bin_curv_val_blu_list = [list(bin_curvature_details_array_blu[:,i,2]) for i in range(1,bin_curvature_details_array_blu.shape[1])]
        whole_curv_val_red_list = list(whole_curvature_details_red_array[:,1,2])
        whole_curv_val_blu_list = list(whole_curvature_details_blu_array[:,1,2])
        
        ### Give me a list of the curvature residuals columns:
        bin_curv_residu_red_list = [list(bin_curvature_details_array_red[:,i,9]) for i in range(1,bin_curvature_details_array_red.shape[1])]
        bin_curv_residu_blu_list = [list(bin_curvature_details_array_blu[:,i,9]) for i in range(1,bin_curvature_details_array_blu.shape[1])]
        whole_curv_residu_red_list = list(whole_curvature_details_red_array[:,1,9])
        whole_curv_residu_blu_list = list(whole_curvature_details_blu_array[:,1,9])

        ### Give me a list of the arclength of the fitted curvature:
        bin_curv_arclen_red_list = [list(bin_curvature_details_array_red[:,i,16]) for i in range(1,bin_curvature_details_array_red.shape[1])]
        bin_curv_arclen_blu_list = [list(bin_curvature_details_array_blu[:,i,16]) for i in range(1,bin_curvature_details_array_blu.shape[1])]
        whole_curv_arclen_red_list = list(whole_curvature_details_red_array[:,1,16])
        whole_curv_arclen_blu_list = list(whole_curvature_details_blu_array[:,1,16])
        
        ### Give me a list of the arc angle of the fitted curvatures (in radians):
        bin_curv_arcanglerad_red_list = [list(bin_curvature_details_array_red[:,i,15]) for i in range(1,bin_curvature_details_array_red.shape[1])]
        bin_curv_arcanglerad_blu_list = [list(bin_curvature_details_array_blu[:,i,15]) for i in range(1,bin_curvature_details_array_blu.shape[1])]
        whole_curv_arcanglerad_red_list = list(whole_curvature_details_red_array[:,1,15])
        whole_curv_arcanglerad_blu_list = list(whole_curvature_details_blu_array[:,1,15])

        ### Give your columns headers:
        time_list.insert(0, 'Time')
        cell_pair_num_list.insert(0, 'Cell Pair Number')
        angle_1_list.insert(0, 'Angle 1 (deg)')
        angle_2_list.insert(0, 'Angle 2 (deg)')
        average_angle_list.insert( 0, 'Average Angle (deg)' )
        contactsurface_list.insert( 0, 'Contact Surface Length (px)')
        contact_corner1_list.insert(0, 'Cell 1 (Red) contact corner locations (x,y)')
        contact_corner2_list.insert(0, 'Cell 2 (Blue) contact corner locations (x,y)')
        corner_corner_length1_list.insert(0, 'Length between contact corners for cell 1 (Red) (px)')
        corner_corner_length2_list.insert(0, 'Length between contact corners for cell 2 (Blue) (px)')        
        noncontactsurface1_list.insert(0, 'Cell 1 (Red) Non-Contact Surface Length (px)')
        noncontactsurface2_list.insert(0, 'Cell 2  (Blue) Non-Contact Surface Length (px)')
        cell1_perimeter_list.insert(0, 'Cell 1 (Red) Perimeter Length (px)')
        cell2_perimeter_list.insert(0, 'Cell 2 (Blue) Perimeter Length (px)')
        filename_list.insert( 0, 'Filename')
        #struct_elem_list.insert( 0, 'Structuring Element Size')
        Nth_pixel_list.insert(0, '# of pixels away from contact corners (px)')
        disk_radius_list.insert(0, 'Structuring element disk radius (px)')
        area_cell_1_list.insert(0, 'Area of cell 1 (Red) (px**2)')
        area_cell_2_list.insert(0, 'Area of cell 2 (Blue) (px**2)')
        cell_pair_area_list.insert(0, 'Area of the cell pair (px**2)')
        #linescan_length_list.insert(0, 'Linescan Length (px)')
        #curve_length_list.insert(0, 'Curve Length (px)')
        processing_time_list.insert(0, "Image Processing Time (seconds)")
        mask_list.insert(0, "Mask? (manually defined by user who checks quality of cell borders)")
        redcellborders_coords.insert(0, "Red cellborders coordinates as list of (row, col) (ie. (y,x))")
        redcontactsurface_coords.insert(0, "Red contact surface coordinates as list of (row, col) (ie. (y,x))")
        rednoncontact_coords.insert(0, "Red non-contact border coordinates as list of (row, col) (ie. (y,x))")
        blucellborders_coords.insert(0, "Blue cellborders coordinates as list of (row, col) (ie. (y,x))")
        blucontactsurface_coords.insert(0, "Blue contact surface coordinates as list of (row, col) (ie. (y,x))")
        blunoncontact_coords.insert(0, "Blue non-contact border coordinates as list of (row, col) (ie. (y,x))")
        redcontactvector1_start.insert(0, "Contact angle 1 vector start point from red cell (x,y)") ### I'm pretty sure this is in (x,y) coordinates.
        blucontactvector1_start.insert(0, "Contact angle 1 vector start point from blue cell (x,y)") ### I'm pretty sure this is in (x,y) coordinates.
        redcontactvector2_start.insert(0, "Contact angle 2 vector start point from red cell (x,y)") ### I'm pretty sure this is in (x,y) coordinates.
        blucontactvector2_start.insert(0, "Contact angle 2 vector start point from blue cell (x,y)") ### I'm pretty sure this is in (x,y) coordinates.
        redcontactvector1.insert(0, "Contact angle 1 vector from red cell (x,y)") ### I'm pretty sure this is in (x,y) coordinates.
        blucontactvector1.insert(0, "Contact angle 1 vector from blue cell (x,y)") ### I'm pretty sure this is in (x,y) coordinates.
        redcontactvector2.insert(0, "Contact angle 2 vector from red cell (x,y)") ### I'm pretty sure this is in (x,y) coordinates.
        blucontactvector2.insert(0, "Contact angle 2 vector from blue cell (x,y)") ### I'm pretty sure this is in (x,y) coordinates.
        red_maxima_coords.insert(0, "Red cell seed coordinates (x,y)")
        blu_maxima_coords.insert(0, "Blue cell seed coordinates (x,y)")
        contact_vectors_list.insert(0, "Contact vector from one pair of contact corners to the other (x,y)") ### I'm pretty sure this is in (x,y) coordinates.
        perpendicular2contact_vectors_list.insert(0, "Perpendicular to contact vector (x,y)") ### I'm pretty sure this is in (x,y) coordinates.
        red_halfangle1_list.insert(0, "Red cell's half-angle of angle 1 (deg)")
        red_halfangle2_list.insert(0, "Red cell's half-angle of angle 2 (deg)")
        blu_halfangle1_list.insert(0, "Blue cell's half-angle of angle 1 (deg)")
        blu_halfangle2_list.insert(0, "Blue cell's half-angle of angle 2 (deg)")
        contactsurface_len_red_list.insert(0, 'Contact surface length of red cell (px)')
        contactsurface_len_blu_list.insert(0, 'Contact surface length of blue cell (px)')
        numbered_border_indices_red_list.insert(0, 'Sequential red cell border coordinates (sequence, (i,j,k))')
        numbered_border_indices_blu_list.insert(0, 'Sequential blue cell border coordinates (sequence, (i,j,k))')        




        [bin_curv_val_red_list[i].insert(0, 'bin_%s_curv_val_red'%i) for i in list(range(num_bins))]
        [bin_curv_val_blu_list[i].insert(0, 'bin_%s_curv_val_blu'%i) for i in list(range(num_bins))]
        whole_curv_val_red_list.insert(0, 'whole_cell_curv_val_red')
        whole_curv_val_blu_list.insert(0, 'whole_cell_curv_val_blu')
        
        [bin_curv_residu_red_list[i].insert(0, 'bin_%s_curv_residu_red'%i) for i in list(range(num_bins))]
        [bin_curv_residu_blu_list[i].insert(0, 'bin_%s_curv_residu_blu'%i) for i in list(range(num_bins))]
        whole_curv_residu_red_list.insert(0, 'whole_cell_curv_residu_red')
        whole_curv_residu_blu_list.insert(0, 'whole_cell_curv_residu_blu')
        
        [bin_curv_arclen_red_list[i].insert(0, 'bin_%s_curv_arclen_red'%i) for i in list(range(num_bins))]
        [bin_curv_arclen_blu_list[i].insert(0, 'bin_%s_curv_arclen_blu'%i) for i in list(range(num_bins))]
        whole_curv_arclen_red_list.insert(0, 'whole_curv_arclen_red')
        whole_curv_arclen_blu_list.insert(0, 'whole_curv_arclen_blu')
        
        [bin_curv_arcanglerad_red_list[i].insert(0, 'bin_%s_curv_arcanglerad_red'%i) for i in list(range(num_bins))]
        [bin_curv_arcanglerad_blu_list[i].insert(0, 'bin_%s_curv_arcanglerad_blu'%i) for i in list(range(num_bins))]
        whole_curv_arcanglerad_red_list.insert(0, 'whole_curve_arcanglerad_red')
        whole_curv_arcanglerad_blu_list.insert(0, 'whole_curve_arcanglerad_blu')

        redchan_fluor_at_contact_list_inner.insert(0, 'redchan_fluor_at_contact_inner')
        grnchan_fluor_at_contact_list_inner.insert(0, 'grnchan_fluor_at_contact_inner')
        bluchan_fluor_at_contact_list_inner.insert(0, 'bluchan_fluor_at_contact_inner')
        redchan_fluor_at_contact_list_outer.insert(0, 'redchan_fluor_at_contact_outer')
        grnchan_fluor_at_contact_list_outer.insert(0, 'grnchan_fluor_at_contact_outer')
        bluchan_fluor_at_contact_list_outer.insert(0, 'bluchan_fluor_at_contact_outer')        
        redchan_fluor_outside_contact_list_inner.insert(0, 'redchan_fluor_outside_contact_inner')
        grnchan_fluor_outside_contact_list_inner.insert(0, 'grnchan_fluor_outside_contact_inner')
        bluchan_fluor_outside_contact_list_inner.insert(0, 'bluchan_fluor_outside_contact_inner')
        redchan_fluor_outside_contact_list_outer.insert(0, 'redchan_fluor_outside_contact_outer')
        grnchan_fluor_outside_contact_list_outer.insert(0, 'grnchan_fluor_outside_contact_outer')
        bluchan_fluor_outside_contact_list_outer.insert(0, 'bluchan_fluor_outside_contact_outer')
        
        
        redchan_fluor_at_contact_list.insert(0, 'redchan_fluor_at_contact')
        grnchan_fluor_at_contact_list.insert(0, 'grnchan_fluor_at_contact')
        bluchan_fluor_at_contact_list.insert(0, 'bluchan_fluor_at_contact')
        redchan_fluor_outside_contact_list.insert(0, 'redchan_fluor_outside_contact')
        grnchan_fluor_outside_contact_list.insert(0, 'grnchan_fluor_outside_contact')
        bluchan_fluor_outside_contact_list.insert(0, 'bluchan_fluor_outside_contact')
        
        redchan_fluor_cytoplasmic_list.insert(0, 'redchan_fluor_cytoplasmic')
        grnchan_fluor_cytoplasmic_list.insert(0, 'grnchan_fluor_cytoplasmic')
        bluchan_fluor_cytoplasmic_list.insert(0, 'bluchan_fluor_cytoplasmic')


        cells_touching_list.insert(0, 'cells touching?')
        
        script_version_list.insert(0, 'script_version')

        msg_list.insert(0, 'messages')
        ### Table of fluorescence intensities:
        ### Headers:
        #red_intensity_mean_headers = list(range(linescan_length))
        #for i in red_intensity_mean_headers:
        #    red_intensity_mean_headers[i] = 'Mean red intensity at line position %i' %i
            
        #grn_intensity_mean_headers = list(range(linescan_length))
        #for i in grn_intensity_mean_headers:
        #    grn_intensity_mean_headers[i] = 'Mean green intensity at line position %i' %i
        
        #master_red_linescan_mean.insert(0, red_intensity_mean_headers)
        #master_grn_linescan_mean.insert(0, grn_intensity_mean_headers)
        #master_red_linescanx.insert(0, 'Red linescan x values')
        #master_red_linescany.insert(0, 'Red linescan y values')
        #master_grn_linescanx.insert(0, 'Green linescan x values')
        #master_grn_linescany.insert(0, 'Green linescan y values')
        
        ### For the intensity data:
        #mean_intensity_table = []
        #mean_intensity_table = np.c_[cell_pair_num_list, master_red_linescan_mean, master_grn_linescan_mean]
        
        #redx_cellpair_num = []
        #redy_cellpair_num = []
        #grnx_cellpair_num = []
        #grny_cellpair_num = []
        #for i in range(filenumber):
        #    redx_cellpair_num.append('Cell pair %i red x linescan values' %(i+1))
        #    redy_cellpair_num.append('Cell pair %i red y linescan values' %(i+1))
        #    grnx_cellpair_num.append('Cell pair %i grn x linescan values' %(i+1))
        #    grny_cellpair_num.append('Cell pair %i grn y linescan values' %(i+1))
        #for i,v in enumerate(redx_cellpair_num):
        #    master_red_linescanx[i].insert(0, v)
        #for i,v in enumerate(redy_cellpair_num):
        #    master_red_linescany[i].insert(0, v)
        #for i,v in enumerate(grnx_cellpair_num):
        #    master_grn_linescanx[i].insert(0, v)
        #for i,v in enumerate(grny_cellpair_num):
        #    master_grn_linescany[i].insert(0, v)
        
        #intensity_table = [v for i in zip(master_red_linescanx, master_red_linescany, master_grn_linescanx, master_grn_linescany) for v in i]

        ### Collect the elements in your lists so that the first element of each list
        ### is on the first row, the the second on the second, etc.:

        excel_table = list(zip( time_list, cell_pair_num_list,
                          angle_1_list, angle_2_list, average_angle_list,
                          red_halfangle1_list, red_halfangle2_list, 
                          blu_halfangle1_list, blu_halfangle2_list, 
                          contactsurface_list, noncontactsurface1_list, noncontactsurface2_list, 
                          cell1_perimeter_list, cell2_perimeter_list,
                          area_cell_1_list, area_cell_2_list, cell_pair_area_list,
                          corner_corner_length1_list, corner_corner_length2_list,
                          redchan_fluor_at_contact_list_inner, grnchan_fluor_at_contact_list_inner, bluchan_fluor_at_contact_list_inner, 
                          redchan_fluor_at_contact_list_outer, grnchan_fluor_at_contact_list_outer, bluchan_fluor_at_contact_list_outer, 
                          redchan_fluor_outside_contact_list_inner, grnchan_fluor_outside_contact_list_inner, bluchan_fluor_outside_contact_list_inner, 
                          redchan_fluor_outside_contact_list_outer, grnchan_fluor_outside_contact_list_outer, bluchan_fluor_outside_contact_list_outer, 
                          redchan_fluor_at_contact_list, grnchan_fluor_at_contact_list, bluchan_fluor_at_contact_list, 
                          redchan_fluor_outside_contact_list, grnchan_fluor_outside_contact_list, bluchan_fluor_outside_contact_list, 
                          redchan_fluor_cytoplasmic_list, grnchan_fluor_cytoplasmic_list, bluchan_fluor_cytoplasmic_list, 
                          *bin_curv_val_red_list, whole_curv_val_red_list, *bin_curv_val_blu_list, whole_curv_val_blu_list,
                          *bin_curv_residu_red_list, whole_curv_residu_red_list, *bin_curv_residu_blu_list, whole_curv_residu_blu_list,
                          *bin_curv_arclen_red_list, whole_curv_arclen_red_list, *bin_curv_arclen_blu_list, whole_curv_arclen_blu_list,
                          *bin_curv_arcanglerad_red_list, whole_curv_arcanglerad_red_list, *bin_curv_arcanglerad_blu_list, whole_curv_arcanglerad_blu_list,
                          filename_list, cells_touching_list, mask_list,
                          Nth_pixel_list, 
                          disk_radius_list, 
                          processing_time_list, 
                          redcellborders_coords, redcontactsurface_coords, rednoncontact_coords, 
                          blucellborders_coords, blucontactsurface_coords, blunoncontact_coords, 
                          redcontactvector1_start, blucontactvector1_start, 
                          redcontactvector2_start, blucontactvector2_start, 
                          redcontactvector1, blucontactvector1,
                          redcontactvector2, blucontactvector2, 
                          contact_corner1_list, contact_corner2_list, 
                          red_maxima_coords, blu_maxima_coords,
                          contact_vectors_list, perpendicular2contact_vectors_list,
                          numbered_border_indices_red_list, numbered_border_indices_blu_list, 
                          contactsurface_len_red_list, contactsurface_len_blu_list, 
                          script_version_list, msg_list))

        for i,v in enumerate(excel_table):
            excel_table[i] = list(v)
### The following block of code can be used to replace np.float('nan') with a
### blank space -- useful if you are plotting things in excel since nan will
### show up as a 0 in excel:
#        for i,v in enumerate(excel_table):
#            for x,y in enumerate(v):
#                if type(y) == float or type(y) == np.float64:
#                    if np.isnan(y):
#                        excel_table[i][x] = ''

### Does replacing nan's with '' stop me from distinguishing real 0's, real seperations
### and skipped images?                        
        for i,v in enumerate(excel_table):
            excel_table[i] = tuple(v)
        
        ### Next we write the table in to a .csv file:
        while True:
            csv_filename = input('What would you like to name your .csv file?\nThe file extension is not necessary.\n[default:%s]\n> ' %os.path.basename(path)) or os.path.basename(path)
            if os.path.exists(csv_filename+'.csv') == True:
                print ("That file already exists in the folder, would you like to"
                       " overwrite it?\nY (overwrite it) / N (don't overwrite it)"
                       )
                overwrite = input(">")
                if overwrite == 'y' or overwrite == 'Y' or overwrite == 'yes':
                    # pass
                    break
                
                elif overwrite == 'n' or overwrite == 'N' or overwrite == 'no':
                    copy_count = 1
                    csv_filename = [csv_filename, ' (', str(copy_count), ')']
                    while os.path.exists(''.join(csv_filename)+'.csv') == True:
                        copy_count +=1
                        csv_filename[-2] = str(copy_count)
                    break
                        
                else:
                    print("Sorry I didn't understand that, please tell me again:")
                    continue
                
            elif os.path.exists(csv_filename+'.csv') == False:
                break
                
        csv_filename = ''.join(csv_filename)
        ### Actually write the table in to a .csv file now:
        with open(csv_filename+'.csv', 'w', newline='') as csv_file:
            out = csv.writer(csv_file, dialect='excel')
            out.writerows(excel_table)
        excel_table_arr = np.array(excel_table, dtype=object)
        ### "dialect='excel' allows excel to read the table correctly. If you don't 
        ### have that part then whats inside each tuple (the values in brackets in 
        ### excel_table) will show up in 1 cell and each tuple will show up in a new 
        ### column. If you use "w" (for write) instead of "wb" (for write buffer) then
        ### your excel file will have spaces between each row. "wb" gets rid of that.
        

         
        multipdf()


        
        curveborders1_all_array = np.dstack([x for x in curveborders1_protoarray])
        curveborders2_all_array = np.dstack([x for x in curveborders2_protoarray])
        np.save(path+'\\%s curveborders1_all_array' %csv_filename, curveborders1_all_array)
        np.save(path+'\\%s curveborders2_all_array' %csv_filename, curveborders2_all_array)
        
        noncontactborders1_all_array = np.dstack([x for x in noncontactborders1_protoarray])
        noncontactborders2_all_array = np.dstack([x for x in noncontactborders2_protoarray])
        np.save(path+'\\%s noncontactborders1_all_array' %csv_filename, noncontactborders1_all_array)
        np.save(path+'\\%s noncontactborders2_all_array' %csv_filename, noncontactborders2_all_array)

        contactsurface1_all_array = np.dstack([x for x in contactsurface1_protoarray])
        contactsurface2_all_array = np.dstack([x for x in contactsurface2_protoarray])
        np.save(path+'\\%s contactsurface1_all_array' %csv_filename, contactsurface1_all_array)
        np.save(path+'\\%s contactsurface2_all_array' %csv_filename, contactsurface2_all_array)
        
        pathfinder1_all_array = np.dstack([x for x in pathfinder1_protoarray])
        pathfinder2_all_array = np.dstack([x for x in pathfinder2_protoarray])
        np.save(path+'\\%s pathfinder1_all_array' %csv_filename, pathfinder1_all_array)
        np.save(path+'\\%s pathfinder2_all_array' %csv_filename, pathfinder2_all_array)

        angles_angle_labeled_red_all_array = np.dstack([x for x in angles_angle_labeled_red_protoarray])
        angles_angle_labeled_blu_all_array = np.dstack([x for x in angles_angle_labeled_blu_protoarray])
        np.save(path+'\\%s angles_angle_labeled_red_all_array' %csv_filename, angles_angle_labeled_red_all_array)
        np.save(path+'\\%s angles_angle_labeled_blu_all_array' %csv_filename, angles_angle_labeled_blu_all_array)


        ### Save the red binned curvature estsimates in to a separate array file:       
        np.save(path+'\\%s bin_curvature_details_array_red' %csv_filename, bin_curvature_details_array_red)
        
        ### Save the grn binned curvature estsimates in to a separate array file:        
        np.save(path+'\\%s bin_curvature_details_array_blu' %csv_filename, bin_curvature_details_array_blu)

        ### Save the bin_indices_red and _grn as arrays they can be opened later:
        bin_indices_red_array = np.asarray(bin_indices_red_list, dtype='object')
        np.save(path+'\\%s bin_indices_red_array' %csv_filename, bin_indices_red_array)

        bin_indices_blu_array = np.asarray(bin_indices_blu_list, dtype='object')
        np.save(path+'\\%s bin_indices_blu_array' %csv_filename, bin_indices_blu_array)

        ### Save the bin_start_red_list and bin_stop_red_list:
        bin_start_red_list_array = np.asarray(bin_start_red_list, dtype='object')
        np.save(path+'\\%s bin_start_red_list_array' %csv_filename, bin_start_red_list_array)
        bin_stop_red_list_array = np.asarray(bin_stop_red_list, dtype='object')
        np.save(path+'\\%s bin_stop_red_list_array' %csv_filename, bin_stop_red_list_array)
        ### Do the same for the green:
        bin_start_blu_list_array = np.asarray(bin_start_blu_list, dtype='object')
        np.save(path+'\\%s bin_start_blu_list_array' %csv_filename, bin_start_blu_list_array)
        bin_stop_blu_list_array = np.asarray(bin_stop_blu_list, dtype='object')
        np.save(path+'\\%s bin_stop_blu_list_array' %csv_filename, bin_stop_blu_list_array)

        ### Save the whole-cell curvature estimate red in a separate array file:
        np.save(path+'\\%s whole_curvature_details_array_red' %csv_filename, whole_curvature_details_red_array)
        
        ### Save the whole-cell curvature estimate grn in a separate array file:
        np.save(path+'\\%s whole_curvature_details_array_blu' %csv_filename, whole_curvature_details_blu_array)





        break
    
    elif save_table == 'n' or save_table == 'N' or save_table == 'no':
        confirm_no_table_save = input("Are you sure you don't want to save your data as an excel table?"
                                          "\nY (I don't want to save it) / N (I changed my mind, let me save it!)\n>")
        if confirm_no_table_save == 'y' or confirm_no_table_save == 'Y' or confirm_no_table_save == 'yes':
            break
        elif confirm_no_table_save == 'n' or confirm_no_table_save == 'N' or confirm_no_table_save == 'no':
            continue
        else:
            print("Sorry I didn't understand that, please tell me again:")
            continue
        
    else:
        print("Sorry I didn't understand that, please tell me again:")
        continue
        

print("I'm finished with \n%s\nnow!"%path)
while True:    
    finish = input("Press enter to quit.\n>")
    break

        
        

        
















