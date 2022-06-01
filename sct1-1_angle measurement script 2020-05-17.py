    # -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 21:55:37 2019

@author: Serge
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 14:59:38 2018

@author: Serge
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 14:10:04 2017

@author: Serge
    """

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 11:10:03 2015

@author: Serge Parent
"""

### Welcoming message and explanation of script:
print ("Hello! Welcome to my cell pair angle measurement script. "
       "This script will find the angle between any 2 cells that are labeled "
       "with green or red fluorescent dyes. It can also be tweaked to work "
       "with blue dyes if needed. In order for this script to work as intended "
       "you will need a folder of cropped images where a pair of cells is in "
       "the middle of the image. The algorithm used to segment cells has "
       "trouble with separating cells in cell aggregates, so try to avoid "
       "using cells attached to cell aggregates or double check their "
       "segmentations. Any cells that touch the borders of the images are "
       "removed so make sure that your cell pair of interest isn't doing so.")
       
       
    
### Import what you need.
import numpy as np

#import torch

### Axes3D will be used to plot the line scans over time (ie. for each cell
### pair timepoint)
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection

### Import Pdfpages to save figures in to a multi-page PDF file:
from matplotlib.backends.backend_pdf import PdfPages

# NumPy is a core Python package and can create arrays of any dimension, images
# that are read in to Python are interpreted as arrays (so for example an RGB 
# image has the x and y axis)
import mahotas as mh

### From PIL import Image to read image files (PIL can handle TIFF tags)
from PIL import Image

### Import imread to deal with multi-page .tif images:
#import imread

# Mahotas is basically a collection of functions/algorithms useful for
# analyzing micrographs or other images found in biology.
from matplotlib import pyplot as plt
from matplotlib import colors

### Import heapq to find the nth biggest number when cleaning up regions using 
### mahotas.
import heapq

### Import math so that you can calculate the angle between 2 vectors by using
### the cross product:
import math

### Import the os so you can access the file.
import os

### Import sys because you will need it later in case the script needs to exit
### before running all the way through:
#import sys

### import glob so that you can look through a folder for files with a pre-
### defined file extension (eg. only .png files or only .jpg files)
import glob

### Import csv so that you can create comma style value files from your data.
### You can open csv files with excel.
import csv

### Import interpolate and optimization from scipy to interpolate intensity
### data and fit model functions/solve functions respectively:
import scipy.interpolate as interpolate
import scipy.optimize as optimization
import scipy.ndimage.morphology as scp_morph
import scipy.spatial as scp_spatial
from scipy.signal import argrelextrema
### Import the garbage collector - maybe things iterations will happen more quickly:
import gc
### It didn't help.

import time

### Import orthogonal distance regression to make curvature estimates based on
### bins:
from scipy import  odr
from scipy import ndimage as ndi

### Import various computervision packages:
from skimage import feature
from skimage import segmentation
from skimage import measure
from skimage import morphology
from skimage import filters


### Make a function that will subtract b from a (ie. a - b) unless a-b
### is less than zero, in which case just give me 0. This will allow me to take
### a slice from a to b (ie. [a:b]) from one end of an image to the other as a
### sliding window. 
def sub(abra,cadabra):
    if abra-cadabra < 0:
        return 0
    else:
        return abra-cadabra




### Define an iterated implementation of a dilation function:
def iter_dilate(image, iterations, selem=None, out=None, shift_x=False, shift_y=False):
    image = image.astype(bool)
    i = 0
    while i < iterations:
        image = morphology.dilation(image, selem=selem, out=out, shift_x=shift_x, shift_y=shift_y)
        i += 1
    return image




def iter_erode(image, iterations, selem=None, out=None, shift_x=False, shift_y=False):
    image = image.astype(bool)
    i = 0
    while i < iterations:
        image = morphology.erosion(image, selem=selem, out=out, shift_x=shift_x, shift_y=shift_y)
        i += 1
    return image




### Function that returns a copy of a 2D image but with all 4 edges set to 0.
### Useful for stopping mh.distance from having high values that go right to edge:
def set_img_edges_vals(image, value, blackout_px_width=1):
    image_copy = np.copy(image)
    image_copy[0:blackout_px_width,:] = value
    image_copy[-1 * blackout_px_width:,:] = value
    image_copy[:,0:blackout_px_width] = value
    image_copy[:,-1 * blackout_px_width:] = value
    return image_copy




### Function to add padding around edge of image:
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

    

### This function was taken from skimage.filters source code because it wasn't
### present in the package for some reason. It works very similarly to the 
### Canny edge detector, but with regions:
def apply_hysteresis_threshold(image, low, high):
    """Apply hysteresis thresholding to `image`.
    This algorithm finds regions where `image` is greater than `high`
    OR `image` is greater than `low` *and* that region is connected to
    a region greater than `high`.
    Parameters
    ----------
    image : array, shape (M,[ N, ..., P])
        Grayscale input image.
    low : float, or array of same shape as `image`
        Lower threshold.
    high : float, or array of same shape as `image`
        Higher threshold.
    Returns
    -------
    thresholded : array of bool, same shape as `image`
        Array in which `True` indicates the locations where `image`
        was above the hysteresis threshold.
    Examples
    --------
    >>> image = np.array([1, 2, 3, 2, 1, 2, 1, 3, 2])
    >>> apply_hysteresis_threshold(image, 1.5, 2.5).astype(int)
    array([0, 1, 1, 1, 0, 0, 0, 1, 1])
    References
    ----------
    .. [1] J. Canny. A computational approach to edge detection.
           IEEE Transactions on Pattern Analysis and Machine Intelligence.
           1986; vol. 8, pp.679-698.
           DOI: 10.1109/TPAMI.1986.4767851
    """
    low = np.clip(low, a_min=None, a_max=high)  # ensure low always below high
    mask_low = image > low
    mask_high = image > high
    # Connected components of mask_low
    labels_low, num_labels = ndi.label(mask_low, structure = np.ones((3,3)))
    # I tried an 8-connectivity structuring element; original code is below:
    #labels_low, num_labels = ndi.label(mask_low)
    
    # Check which connected components contain pixels from mask_high
    sums = ndi.sum(mask_high, labels_low, np.arange(num_labels + 1))
    connected_to_high = sums > 0
    thresholded = connected_to_high[labels_low]
    return thresholded




### Define a function to find the 3rd largest region (the first largest is 
### probably the background, the second largest would be the first cell, the
### third largest would be the second cell). "n" is the number of iteration
### used and "iter" is an iterable object.
def nth_max(n, iter):
    return heapq.nlargest(n, iter)[-1]




def calc_R(x, y, xc, yc):
    """ calculate the distance of each data point from the center (xc, yc) """
    return np.sqrt((x-xc)**2 + (y-yc)**2)




#def f_lin(beta, x):
#    """ implicit definition of the line """
#    return (x[0]-beta[0]) + (x[1]-beta[1])




def f_3(beta, x):
    """ implicit definition of the circle """
    return (x[0]-beta[0])**2 + (x[1]-beta[1])**2 -beta[2]**2
    



### Define a function to round a number down to the nearest even value:
def round_down_to_even(integer):
    if type(integer) == int:# or type(integer) == long #this part was for Python2.7:
        if integer%2 == 0:
            rounded_value = integer
        elif integer%2 == 1:
            rounded_value = integer-1
    else:
        print("The number to be rounded needs to be and integer!")
        rounded_value = float('nan')
    return rounded_value




### Define a function to save a multi-page PDF:
def multipdf():
    pdf = PdfPages('%s Overlay & Intensity Plots.pdf' %myfile)
    figs = [plt.figure(f) for f in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pdf, format='pdf' , dpi=80)
    pdf.close()
    
    
    
### Curve_definitions_red is now a list with a number of elements equal to the 
### number of pixels in the red curveborders. Each element in this list 
### contains 3 (y,x) points, the first of which is the pixel for which the
### curvature will be estimated. These 3 points will be used to analytically
### solve for a circle that passes through all 3 points. The inverse of the 
### radius of this circle will then be the estimate of the curvature.
###     The circumcircle of 3 points can be defined by the following system of 
###     equations (Pedoe, D. Circles: A Mathematical View, rev. ed. Washington,
###     DC: Math. Assoc. Amer., 1995. pp. xii-xiii):
###             DET | x**2 + y**2    x    y    1 |
###                 | x1**2+y1**2    x1   y1   1 |
###                 | x2**2+y2**2    x2   y2   1 |
###                 | x3**2+y3**2    x3   y3   1 |
###                             = 0

###     The circumcenter can be defined as: x0 = -(bx/2a)
###                                         y0 = -(by/2a)
###
###     The circumradius can be defined as: r = sqrt(bx**2 + by**2 - 4ac) / 2*magnitude(a)
###
###     Where 
###         a = DET| x1   y1   1 |
###                | x2   y2   1 |
###                | x3   y3   1 |

###         bx = -DET | x1**2+y1**2   y1   1 |
###                   | x2**2+y2**2   y2   1 |
###                   | x3**2+y3**2   y3   1 |

###         by = +DET | x1**2+y1**2   x1   1 |     (Note the plus sign!)
###                   | x2**2+y2**2   x2   1 |
###                   | x3**2+y3**2   x3   1 |

###         c = -DET | x1**2+y1**2   x1   y1 |
###                  | x2**2+y2**2   x2   y2 |
###                  | x3**2+y3**2   x3   y3 |

###         This is taken from:
###         Weisstein, Eric W. "Circumcircle." From MathWorld--A Wolfram Web Resource. http://mathworld.wolfram.com/Circumcircle.html 

### Now to implement this in code:

### For the array_of_3_x_values and array_of_3_y_values, the first point MUST be
### the point in between the other 2!
def curvature_from_3points(array_of_3_x_values, array_of_3_y_values):
    """Returns the radius (which is the inverse of the curvature), and the x 
    and y coordinates of the center of the circle as a (radius, x0, y0) tuple."""
    if len(array_of_3_x_values) == 3 and len(array_of_3_y_values) == 3:
        
#        curve_definitions_y_temp, curve_definitions_x_temp = np.asarray(zip(*curve_definitions_red[0])[0]), np.asarray(zip(*curve_definitions_red[0])[1])
        
        a_array = np.array(list(zip(array_of_3_x_values, array_of_3_y_values, (1,1,1))))
        if np.isnan(a_array).any() == False:
            # If the determinant of a_array = 0 then the x and y values fall on
            # a straight line, so when finding the the x and y coords of the 
            # circles center it will throw a dividebyzero error if not suppressed.
            a = np.linalg.det(a_array)

            
            bx_array = np.array(list(zip(np.array(array_of_3_x_values)**2.0 + np.array(array_of_3_y_values)**2.0, np.array(array_of_3_y_values), np.array((1,1,1)))) )
            bx = -1.0 * np.linalg.det(bx_array)
            
            by_array = np.array(list(zip(np.array(array_of_3_x_values)**2.0 + np.array(array_of_3_y_values)**2.0, np.array(array_of_3_x_values), np.array((1,1,1)))) )
            by = 1.0 * np.linalg.det(by_array)
            
            c_array = np.array(list(zip(np.array(array_of_3_x_values)**2.0 + np.array(array_of_3_y_values)**2.0, np.array(array_of_3_x_values), np.array(array_of_3_y_values))) )
            c = -1.0 * np.linalg.det(c_array)
            
            # Suppress the warnings for divide by zero, since we actually want
            # infinity to be returned:
            with np.errstate(divide='ignore'):
                x0 = -(bx / (2.0 * a) )
                y0 = -(by / (2.0 * a) )
                circumcircle_radius = ((bx**2.0 + by**2.0 - 4.0*a*c)**0.5) / (2.0 * ((a**2.0)**0.5))
                
    #        theta = np.linspace(0,2*np.pi, 360)
    #        x = r * np.cos(theta) + x0
    #        y = r * np.sin(theta) + y0
            
            #plt.scatter(x0,y0)
            #plt.plot(x,y)
            #plt.scatter(curve_definitions_x_temp, curve_definitions_y_temp)
            ### Assign this curvature to it's central pixel:
            
            return_val = (circumcircle_radius, x0, y0)
        else:
            return_val = (float('nan'), float('nan'), float('nan'))
    else:
        print("Either the x array or the y array do not have 3 values! I need 3 values!")
        return_val = (float('nan'), float('nan'), float('nan'))
    return return_val




def curvature_sign(array_of_3_x_values, array_of_3_y_values, curvature_definitions):
    """Returns the sign of the curvature based on whether or not the rasterized
       version of a straight line draw from one end of the curve to the other is
       mostly (ie. more than 50%) contained in a filled version of set of 
       curvature points."""
    if len(array_of_3_x_values) == 3 and len(array_of_3_y_values) == 3:
        
        if np.isnan(array_of_3_x_values).any() == False and np.isnan(array_of_3_y_values).any() == False:
            curve_chord_len = int(round(np.sqrt((array_of_3_x_values[2] - array_of_3_x_values[1])**2 + (array_of_3_y_values[2] - array_of_3_y_values[1])**2))+1)
            x = np.linspace(array_of_3_x_values[1], array_of_3_x_values[2], num=curve_chord_len, endpoint=True)
            y = np.linspace(array_of_3_y_values[1], array_of_3_y_values[2], num=curve_chord_len, endpoint=True)
            x, y = np.round(x), np.round(y)
            x, y = x.astype(np.int), y.astype(np.int)
        #        y_x = zip(y,x)
            
            
            curvature_definitions = np.asarray(curvature_definitions)
            curvature_definitions = np.ma.masked_array(curvature_definitions, np.isnan(curvature_definitions)).astype(int)
            clean_curve_area = np.zeros( (curvature_definitions[:,0,0].max()+10, curvature_definitions[:,0,1].max()+10) )
            clean_curve_area[curvature_definitions[:,0,0], curvature_definitions[:,0,1]] = 1
            clean_curve_area = scp_morph.binary_fill_holes(clean_curve_area)
            clean_curve_dist = mh.distance(clean_curve_area)
            mean_chord_intens = np.mean(clean_curve_dist[y,x])
            mean_curve_intens = np.mean(clean_curve_dist[array_of_3_y_values, array_of_3_x_values])
            
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
    else:
        print("Either the x array or the y array do not have 3 values! I need 3 values!")
        return_val = float('nan')
    return return_val
        


        
def check_for_signal(image_of_cellpair_as_array):
    ### If the signal in a channel has a standard deviation less than 1 in any 
    ### channel, skip it because the signal to noise ratio won't be good enough
    ### for proper edge detection and segmentation:
    redchan = image_of_cellpair_as_array[:,:,0]
    grnchan = image_of_cellpair_as_array[:,:,1]
    bluchan = image_of_cellpair_as_array[:,:,2]

    if np.std(redchan) < 1:
        redchan = redchan*0
        print ("There is no signal in the red channel or the signal-to-noise"
               " ratio in the red channel is too low."
               )
    if np.std(grnchan) < 1:
        grnchan = grnchan*0
        print ("There is no signal in the green channel or the signal-to-noise"
               " ratio in the green channel is too low."
               )
    if np.std(bluchan) < 1:
        bluchan = bluchan*0
        print ("There is no signal in the blue channel or the signal-to-noise"
               " ratio in the blue channel is too low."
               )
    image_of_cellpair_as_array_clean = np.array(np.dstack((redchan, grnchan, bluchan)))
    return image_of_cellpair_as_array_clean




### Define a function to segment your image:
def get_masked_canny(image, particle_filter, initial_mask, sigma, hist_percentile_upper_cutoff, low_threshold=0, high_threshold=0):
    fullhist_redsobel = mh.fullhistogram(image)
    hist_redsobel_upperthresh = np.sum(fullhist_redsobel[1:]) * hist_percentile_upper_cutoff
    hist_redsobel_cumulative_sum = np.cumsum(fullhist_redsobel[1:])
    hist_dist_redsobel = (hist_redsobel_cumulative_sum - hist_redsobel_upperthresh)**2
    try:
        redcanny_upperthresh = np.where(hist_dist_redsobel == np.min(hist_dist_redsobel))#.astype(np.uint8)
        if len(redcanny_upperthresh[0]) > 1:
            redcanny_upperthresh = redcanny_upperthresh[0][0]
    except ValueError:
        redcanny_upperthresh = 255
   
    if hist_percentile_upper_cutoff == 0:
        high_threshold = high_threshold
        low_threshold = low_threshold
    else:
        high_threshold = redcanny_upperthresh
        low_threshold = mh.otsu(image)*0.2
    
    init_canny = feature.canny(image, low_threshold = low_threshold, high_threshold = high_threshold, sigma=sigma)
    init_canny_fill = scp_morph.binary_fill_holes(init_canny)
    init_canny_labels = mh.label(init_canny_fill)[0]
    init_canny_spots = np.logical_xor(init_canny, init_canny_fill)
    init_canny_spots_labeled = init_canny_spots * init_canny_labels
    sizes = mh.labeled.labeled_size(init_canny_spots_labeled)
    too_small = np.where( np.logical_and(sizes < particle_filter, 0 < sizes) )
    mask = np.zeros(init_canny.shape)
    mask = np.copy(initial_mask)
    mask += np.logical_or.reduce( [init_canny_labels == i for i in np.unique(too_small)] )
#    mask = np.logical_or(mask, reddist_rough_bool)
    mask_red = np.logical_not(mask)
    canny_masked_red = feature.canny(image, mask = mask_red, low_threshold = low_threshold, high_threshold = high_threshold, sigma=sigma)
#    redareas_rough_in = (mh.cwatershed(mask_red * (mh.sobel(edge_enh_cells_red, just_filter = True)), cellspotsflat))
    redsob = mh.stretch(mh.sobel(image, just_filter = True))
    redsob[np.where(canny_masked_red)] = 255

    return redsob, canny_masked_red, init_canny, mask_red




def watershed_on_cannyweighted_sobel_based_segment_and_isolate_center_cells(image_of_cellpair_as_array, gaussian_filter_value_red, gaussian_filter_value_grn, gaussian_filter_value_blu, structuring_element_to_find_maxima, background_threshold, object_of_interest_brightness_cutoff_red, object_of_interest_brightness_cutoff_grn):

    # Set some basic variables and parameters:
    # Normalize each channel of the image:
    image_of_cellpair_as_array[:,:,0] = mh.stretch(image_of_cellpair_as_array[:,:,0])
    image_of_cellpair_as_array[:,:,1] = mh.stretch(image_of_cellpair_as_array[:,:,1])
    image_of_cellpair_as_array[:,:,2] = mh.stretch(image_of_cellpair_as_array[:,:,2])
    
    img_flat = mh.stretch( np.max(image_of_cellpair_as_array, axis=2) )
  
    # Define a 3 dimensional strucutring element:
    Bc3D = np.ones((3,3,3))
    
    # Ridge weight:
    #ridge_weight = 52
    
    # particle_filter = 200#20 #the 20 filter is used for the 2 circle simulation
    # The particle filter is defined globally much further down
###############################################################################


### Do a gaussian blur of the red, green, and blue channels to remove noise,
    #these blurred images will be used frequently to identify coarse structure:
    redfilt = mh.stretch( mh.gaussian_filter(image_of_cellpair_as_array[:,:,0], (gaussian_filter_value_red, gaussian_filter_value_red)) )
    grnfilt = mh.stretch( mh.gaussian_filter(image_of_cellpair_as_array[:,:,1], (gaussian_filter_value_grn, gaussian_filter_value_grn)) )
    #blufilt = mh.stretch( mh.gaussian_filter(image_of_cellpair_as_array[:,:,2], (gaussian_filter_value_blu, gaussian_filter_value_blu)) )

###############################################################################

  ### This block seems to work pretty well for detecting my cells:
    
    redscharr = filters.scharr(image_of_cellpair_as_array[:,:,0])
    grnscharr = filters.scharr(image_of_cellpair_as_array[:,:,1])
    flatscharr = filters.scharr(img_flat)
    
    allscharr = flatscharr + redscharr + grnscharr
    allscharr = set_img_edges_vals(allscharr, allscharr.max())
    # Set a minimum brightness cutoff of 5% of the dtypes max:


    # Define a structuring element for using filters.rank.otsu:
    rank_otsu_struc_elem = np.copy(structuring_element_to_find_maxima)
    
    # For the red channel:
    # Use local otsu thresholding to determine seed regions for watershed:
    red_rank_otsu = redchan > filters.rank.otsu(redchan, selem=rank_otsu_struc_elem) #selem=maxima_Bc
#    red_rank_otsu_filt = redfilt >= filters.rank.otsu(redfilt, selem=rank_otsu_struc_elem)
    
    # and label the regions:
#red_markers = morphology.label(red_rank_otsu)

    # Use a scharr filter to create an elevation map for the watershed to work
    # on:
    redscharr = filters.scharr(image_of_cellpair_as_array[:,:,0])
    # because the scharr filter does not work on the very edges of the image,
    # set those to be equal to the max value of the scharr filter:
    redscharr = set_img_edges_vals(redscharr, redscharr.max())
    

    
    # For the green channel:
    # Use local otsu thresholding to determine seed regions for watershed:
    grn_rank_otsu = grnchan > filters.rank.otsu(grnchan, selem=rank_otsu_struc_elem)
#    grn_rank_otsu_filt = grnfilt >= filters.rank.otsu(grnfilt, selem=rank_otsu_struc_elem)
    # and label the regions:
#grn_markers = morphology.label(grn_rank_otsu)

    # Use a scharr filter to create an elevation map for the watershed to work
    # on:
    grnscharr = filters.scharr(image_of_cellpair_as_array[:,:,1])
    # because the scharr filter does not work on the very edges of the image,
    # set those to be equal to the max value of the scharr filter:
    grnscharr = set_img_edges_vals(grnscharr, grnscharr.max())
    
    
###############################################################################

    bg_red = image_of_cellpair_as_array[:,:,0] < background_threshold
    #fg_red_filt = ~bg_red_filt * red_rank_otsu_filt
    fg_red = np.copy(red_rank_otsu)
    
    bg_red = np.logical_xor(bg_red, fg_red) * bg_red
    fg_red = np.logical_xor(bg_red, fg_red) * fg_red
    
    bg_red_markers = mh.label(bg_red)[0]
    fg_red_markers = mh.label(fg_red)[0]
    
    fg_red_markers = fg_red_markers + bg_red_markers.max() + 1
    fg_red_markers[np.where(fg_red_markers == (bg_red_markers.max() + 1))] = 0
    
    markers_red = bg_red_markers + fg_red_markers
    
    not_fg_dist_red = ~mh.stretch(mh.distance(fg_red, 'euclidean') ** 0.25)
    
    bg_fg_seg_red = segmentation.watershed(redscharr * not_fg_dist_red, markers_red)
    #bg_fg_seg_red = segmentation.watershed(allscharr * not_fg_dist_red, markers_red)
    
    red_seg = np.copy(bg_fg_seg_red)
    
        # Create a list of arrays where each array is a boolean image of each 
    # segment from grn_seg
    red_seg_bools = [red_seg == x for x in np.unique(red_seg)]
    
    # Find the median of each segment in grn_seg:
    red_seg_vals = [image_of_cellpair_as_array[:,:,0][np.where(x)] for x in red_seg_bools]
    red_seg_vals_median = [np.median(x) for x in red_seg_vals]
    
    # Take only the segments whose median brightness exceeds a pre-defined
    # brightness cutoff:
    red_seg_bright_nums = np.unique(red_seg)[np.asarray(red_seg_vals_median) >= background_threshold]
    red_seg_brighter_nums = np.unique(red_seg)[np.asarray(red_seg_vals_median) >= object_of_interest_brightness_cutoff_red]
    
    # Create a list of boolean arrays where each array is one of the bright
    # regions that you kept:
    red_seg_bright = [(red_seg == x)*x for x in red_seg_bright_nums]
    red_seg_brighter = [(red_seg == x)*x for x in red_seg_brighter_nums]
    # Create a flattened version of this list:
    if np.any(red_seg_bright):
        red_seg_bright_flat = np.logical_or.reduce(red_seg_bright)
    else:
        red_seg_bright_flat = np.zeros(image_of_cellpair_as_array[:,:,0].shape, dtype='bool')
    
    if np.any(red_seg_brighter):
        red_seg_brighter_flat = np.logical_or.reduce(red_seg_brighter)
    else:
        red_seg_brighter_flat = np.zeros(image_of_cellpair_as_array[:,:,0].shape, dtype='bool')

    red_seg_dif = np.logical_xor(red_seg_bright_flat, red_seg_brighter_flat)
    # If removing a piece of the red_seg_dif creates a hole, then keep it.
    # Our cells don't have holes in them.
    red_seg_brighter_filled = morphology.remove_small_holes(red_seg_brighter_flat)
    red_seg_dif_donut_holes = np.logical_and(red_seg_brighter_filled, red_seg_dif)
    red_seg_dif_lrg = morphology.remove_small_objects(red_seg_dif, min_size=particle_filter*2.) #min_size=particle_filter/2.
    red_seg_dif_clean = np.logical_or(red_seg_dif_lrg, red_seg_dif_donut_holes)
    # This is a redefinition of the previous version of red_seg_bright_flat
    # to remove small objects that are between the 2 absolute fluorescence
    # thresholds:
    red_seg_bright_flat = np.logical_or(red_seg_dif_clean, red_seg_brighter_flat)
    #plt.imshow(grn_seg_bright_flat)


    bg_grn = image_of_cellpair_as_array[:,:,1] < background_threshold
    #fg_grn_filt = ~bg_grn_filt * grn_rank_otsu_filt
    fg_grn = np.copy(grn_rank_otsu)
    
    bg_grn = np.logical_xor(bg_grn, fg_grn) * bg_grn
    fg_grn = np.logical_xor(bg_grn, fg_grn) * fg_grn
    
    bg_grn_markers = mh.label(bg_grn)[0]
    fg_grn_markers = mh.label(fg_grn)[0]
    
    fg_grn_markers = fg_grn_markers + bg_grn_markers.max() + 1
    fg_grn_markers[np.where(fg_grn_markers == (bg_grn_markers.max() + 1))] = 0
    
    markers_grn = bg_grn_markers + fg_grn_markers
    
    not_fg_dist_grn = ~mh.stretch(mh.distance(fg_grn, 'euclidean') ** 0.25)
    
    bg_fg_seg_grn = segmentation.watershed(grnscharr * not_fg_dist_grn, markers_grn)
    #bg_fg_seg_grn = segmentation.watershed(allscharr * not_fg_dist_grn, markers_grn)
    
    grn_seg = np.copy(bg_fg_seg_grn)
    
    # Create a list of arrays where each array is a boolean image of each 
    # segment from grn_seg
    grn_seg_bools = [grn_seg == x for x in np.unique(grn_seg)]
    
    # Find the median of each segment in grn_seg:
    grn_seg_vals = [image_of_cellpair_as_array[:,:,1][np.where(x)] for x in grn_seg_bools]
    grn_seg_vals_median = [np.median(x) for x in grn_seg_vals]
    
    # Take only the segments whose median brightness exceeds a pre-defined
    # brightness cutoff:
    grn_seg_bright_nums = np.unique(grn_seg)[np.asarray(grn_seg_vals_median) >= background_threshold]
    grn_seg_brighter_nums = np.unique(grn_seg)[np.asarray(grn_seg_vals_median) >= object_of_interest_brightness_cutoff_grn]
    
    # Create a list of boolean arrays where each array is one of the bright
    # regions that you kept:
    grn_seg_bright = [(grn_seg == x)*x for x in grn_seg_bright_nums]
    grn_seg_brighter = [(grn_seg == x)*x for x in grn_seg_brighter_nums]
    # Create a flattened version of this list:
    if np.any(grn_seg_bright):
        grn_seg_bright_flat = np.logical_or.reduce(grn_seg_bright)
    else:
        grn_seg_bright_flat = np.zeros(image_of_cellpair_as_array[:,:,1].shape, dtype='bool')
    
    if np.any(grn_seg_brighter):
        grn_seg_brighter_flat = np.logical_or.reduce(grn_seg_brighter)
    else:
        grn_seg_brighter_flat = np.zeros(image_of_cellpair_as_array[:,:,1].shape, dtype='bool')
    
    #morphology.label(np.logical_xor(grn_seg_bright_flat, grn_seg_brighter_flat))
    grn_seg_dif = np.logical_xor(grn_seg_bright_flat, grn_seg_brighter_flat)
    # If removing a piece of the red_seg_dif creates a hole, then keep it.
    # Our cells don't have holes in them.
    grn_seg_brighter_filled = morphology.remove_small_holes(grn_seg_brighter_flat)
    grn_seg_dif_donut_holes = np.logical_and(grn_seg_brighter_filled, grn_seg_dif)
    grn_seg_dif_lrg = morphology.remove_small_objects(grn_seg_dif, min_size=particle_filter*2.) #min_size=particle_filter/2.
    grn_seg_dif_clean = np.logical_or(grn_seg_dif_lrg, grn_seg_dif_donut_holes)
    # This is a redefinition of the previous version of grn_seg_bright_flat
    # to remove small objects that are between the 2 absolute fluorescence
    # thresholds:
    grn_seg_bright_flat = np.logical_or(grn_seg_dif_clean, grn_seg_brighter_flat)
    
    
    #plt.imshow(grn_seg_bright_flat)


    # These next 2 lines are to exclude red autofluorescence in a green cell
    # or vice-verse:
    red_seg_brightest = (redfilt >= grnfilt) * red_seg_bright_flat
    grn_seg_brightest = (grnfilt >= redfilt) * grn_seg_bright_flat
    
#########################################

    # This part should separate cells from the same channel:
    #dist_red = mh.distance(red_seg_brightest)
    #dist_red = mh.stretch(mh.distance(np.pad(red_seg_brightest, 1, mode='constant')))
    #dist_red = dist_red[1:-1, 1:-1]
    #max_red = mh.regmax(dist_red, Bc=structuring_element_to_find_maxima)
    dist_red = mh.stretch(mh.distance(red_seg_brightest))
    max_red = feature.peak_local_max(dist_red, footprint=structuring_element_to_find_maxima, exclude_border=False, indices=False)
    # If there is more than 1 peak within the footprint then take the brightest
    # one based on the original fluorescence:
    max_red = feature.peak_local_max(max_red * image_of_cellpair_as_array[:,:,0], footprint=structuring_element_to_find_maxima, exclude_border=False, indices=False)
    # In the event that they are both equally bright take the last one:
    max_red = morphology.label(max_red)
    max_red = feature.peak_local_max(max_red, footprint=structuring_element_to_find_maxima, exclude_border=False, indices=False)
    spots_red = morphology.label(max_red)

    # This part should separate cells from the same channel:
    #dist_grn = mh.stretch(mh.distance(grn_seg_brightest))
    #dist_grn = mh.stretch(mh.distance(np.pad(grn_seg_brightest, 1, mode='constant')))
    #dist_grn = dist_grn[1:-1, 1:-1]
    #max_grn = mh.regmax(dist_grn, Bc=structuring_element_to_find_maxima)
    dist_grn = mh.stretch(mh.distance(grn_seg_brightest))
    max_grn = feature.peak_local_max(dist_grn, footprint=structuring_element_to_find_maxima, exclude_border=False, indices=False)
    # If there is more than 1 peak within the footprint then take the brightest
    # one based on the original fluorescence:
    max_grn = feature.peak_local_max(max_grn * image_of_cellpair_as_array[:,:,1], footprint=structuring_element_to_find_maxima, exclude_border=False, indices=False)
    # In the event that they are both equally bright take the last one:
    max_grn = morphology.label(max_grn)
    max_grn = feature.peak_local_max(max_grn, footprint=structuring_element_to_find_maxima, exclude_border=False, indices=False)
    spots_grn = morphology.label(max_grn)


    max_all = max_red + max_grn
    spots_all = morphology.label(max_all)

    #spots_red = morphology.label(max_red)

    #spots_grn = morphology.label(max_grn)


    red_grad_raw = ~mh.stretch(filters.rank.gradient(redfilt, selem=morphology.disk(1)))/255
    red_grad = red_grad_raw * ((~redfilt).astype(float)**2)
    red_grad = mh.stretch(red_grad)
    red_ridges_ax0 = np.zeros(red_grad.shape)
    red_ridges_ax1 = np.zeros(red_grad.shape)
    red_ridges_ax0[argrelextrema(red_grad, np.greater, axis=0, order=15)] = 1
    red_ridges_ax1[argrelextrema(red_grad, np.greater, axis=1, order=15)] = 2
    #red_ridges_ax0[argrelextrema(red_grad, np.greater, axis=0, order=5)] = 1
    #red_ridges_ax1[argrelextrema(red_grad, np.greater, axis=1, order=5)] = 2
    red_ridges = morphology.binary_closing(np.logical_or(red_ridges_ax0, red_ridges_ax1))
    #plt.imshow(mh.overlay(redchan, red=red_seg_brightest * red_ridges))
    
    ### The use of a gradient looks better for separating out similar cells than
    ### the use of a laplace filter.

    red_fg_bg_edges = mh.borders(red_seg_bright_flat) * red_seg_bright_flat
    
###############################################################################
    # This part is computationally intensive sooo.... Maybe don't keep it if it
    # doesn't improve segmentation:
#    testrange = np.arange(52,256)
#    testcansflat = []
#    for i in testrange:
        #testcansred.append(get_masked_canny(~redfilt, particle_filter, initial_mask_red, 0, 0, low_threshold = 26, high_threshold = i)[1])
        #testcansred.append(get_masked_canny(~redfilt, particle_filter, initial_mask_red, 1, 0, low_threshold = 26, high_threshold = i)[1])
        #testcansred.append(get_masked_canny(~image_of_cellpair_as_array[:,:,0], particle_filter, initial_mask_red, 1, 0, low_threshold = 26, high_threshold = i)[1])
        #testcansred.append(get_masked_canny(~image_of_cellpair_as_array[:,:,0], particle_filter, initial_mask_red, 2, 0, low_threshold = 26, high_threshold = i)[1])
#        testcansflat.append(feature.canny(~filt_flat, sigma = 0, high_threshold = i))
#        testcansflat.append(feature.canny(~filt_flat, sigma = 1, high_threshold = i))
#        testcansflat.append(feature.canny(~img_flat, sigma = 1, high_threshold = i))
#        testcansflat.append(feature.canny(~img_flat, sigma = 2, high_threshold = i))
#    testflat = np.add.reduce(testcansflat)
#    testflat *= ~filt_flat ### Added this in to try and remove edges in middle of cell from being detected

#    n_spots = len(np.unique(spots_all)[1:-1])

### Alternate way of getting testred4:
    ### Give all disconnected edges from testred a different label:
#    labeled_can_flat = mh.label(testflat.astype(bool), Bc=np.ones((3,3)))[0]
    ### Return the maximum value for each disconnected edge (but skip the first
    ### region which is where labeled_can_red == 0):
#    lab_can_maxs_flat = [np.max(testflat[np.where(labeled_can_flat == x)]) for x in np.unique(labeled_can_flat)[1:]]
    ### Use these maximas to find the n+1 most common edge, where n is the 
    ### number of spots (or seed points) used for segmentation:
#    if testflat.any():
#        unique_edge_labels_flat = np.unique(labeled_can_flat)[1:]
#        sorted_edge_labels_flat = unique_edge_labels_flat[np.argsort(lab_can_maxs_flat)]
#        edges_to_use_flat = sorted_edge_labels_flat[-1*(n_spots+1):]
#        testflat4 = np.logical_or.reduce([labeled_can_flat == x for x in edges_to_use_flat])
#        img_frame = np.zeros(testflat4.shape)
#        img_frame = set_img_edges_vals(img_frame, True)
#        missing_edge_px = np.logical_and(img_frame, mh.dilate(testflat4))
#        testflat4 += missing_edge_px
#    else:
#        testflat4 = np.zeros(testflat.shape)

    
    # Ridges for flattened image:
#    flat_grad_raw = ~mh.stretch(filters.rank.gradient(filt_flat, selem=morphology.disk(1)))/255
#    flat_grad = flat_grad_raw * ((~filt_flat).astype(float)**2)
#    flat_grad = mh.stretch(flat_grad)
#    flat_ridges_ax0 = np.zeros(flat_grad.shape)
#    flat_ridges_ax1 = np.zeros(flat_grad.shape)
#    flat_ridges_ax0[argrelextrema(flat_grad, np.greater, axis=0, order=15)] = 1
#    flat_ridges_ax1[argrelextrema(flat_grad, np.greater, axis=1, order=15)] = 2
#    flat_ridges = morphology.binary_closing(np.logical_or(flat_ridges_ax0, flat_ridges_ax1))

###############################################################################
    
    ### Try to find edges using a canny edge detector:
    # Set a mask to exclude small inclusions in the cell:
    red_canny_edges = feature.canny(image_of_cellpair_as_array[:,:,0])
    mask_red = ~np.logical_xor(red_canny_edges, morphology.remove_small_holes(red_canny_edges) )
    
    testrange = np.arange(52,256)
    testcansred = []
    for i in testrange:
        #testcansred.append(get_masked_canny(~redfilt, particle_filter, initial_mask_red, 0, 0, low_threshold = 26, high_threshold = i)[1])
        #testcansred.append(get_masked_canny(~redfilt, particle_filter, initial_mask_red, 1, 0, low_threshold = 26, high_threshold = i)[1])
        #testcansred.append(get_masked_canny(~image_of_cellpair_as_array[:,:,0], particle_filter, initial_mask_red, 1, 0, low_threshold = 26, high_threshold = i)[1])
        #testcansred.append(get_masked_canny(~image_of_cellpair_as_array[:,:,0], particle_filter, initial_mask_red, 2, 0, low_threshold = 26, high_threshold = i)[1])
        testcansred.append(feature.canny(~redfilt, sigma = 0, high_threshold = i, mask = mask_red))
        testcansred.append(feature.canny(~redfilt, sigma = 1, high_threshold = i, mask = mask_red))
        testcansred.append(feature.canny(~image_of_cellpair_as_array[:,:,0], sigma = 1, high_threshold = i, mask = mask_red))
        testcansred.append(feature.canny(~image_of_cellpair_as_array[:,:,0], sigma = 2, high_threshold = i, mask = mask_red))
    testred = np.add.reduce(testcansred)
    testred *= ~redfilt ### Added this in to try and remove edges in middle of cell from being detected

#    red_edge_hist = mh.fullhistogram(mh.stretch(testred))[1:]
#    red_edge_hist = red_edge_hist * np.arange(1, len(red_edge_hist)+1)
    n_spots = len(np.unique(spots_red)[1:-1])
#    second_tallest_peak_indice = np.sort(red_edge_hist)[-1*(2*n_spots):]
#    second_tallest_peak = np.where(np.logical_or.reduce([red_edge_hist == i for i in second_tallest_peak_indice]))
#    if testred.any():
#        testred4 = apply_hysteresis_threshold(mh.stretch(testred), low=0, high=np.min(second_tallest_peak))
#    else:
#        testred4 = np.zeros(testred.shape)
#    ### Use testred4 to filter out very blurry edges:
#    testred *= testred4
    
### Alternate way of getting testred4:
    ### Give all disconnected edges from testred a different label:
    labeled_can_red = mh.label(testred.astype(bool), Bc=np.ones((3,3)))[0]
    ### Return the maximum value for each disconnected edge (but skip the first
    ### region which is where labeled_can_red == 0):
    lab_can_maxs_red = [np.max(testred[np.where(labeled_can_red == x)]) for x in np.unique(labeled_can_red)[1:]]
    ### Use these maximas to find the n+1 most common edge, where n is the 
    ### number of spots (or seed points) used for segmentation:
    if testred.any():
        unique_edge_labels_red = np.unique(labeled_can_red)[1:]
        sorted_edge_labels_red = unique_edge_labels_red[np.argsort(lab_can_maxs_red)]
        edges_to_use_red = sorted_edge_labels_red[-1*(n_spots+1):]
        testred4 = np.logical_or.reduce([labeled_can_red == x for x in edges_to_use_red])
        img_frame = np.zeros(testred4.shape)
        img_frame = set_img_edges_vals(img_frame, True)
        missing_edge_px = np.logical_and(img_frame, mh.dilate(testred4))
        testred4 += missing_edge_px
    else:
        testred4 = np.zeros(testred.shape)
    ###

    #red_canny_edges = np.copy(testred4)
    
    red_edges = np.max([mh.stretch(allscharr), mh.stretch(red_fg_bg_edges), mh.stretch(red_ridges * red_seg_bright_flat)], axis=0)
    
    ### More testing:
    red_can_fgbg = np.logical_or(red_canny_edges, red_fg_bg_edges)
    red_can_fgbg_lab = morphology.label(red_can_fgbg)
    connect_labels = np.unique(red_fg_bg_edges * red_can_fgbg_lab)
    connect_labels = connect_labels[connect_labels > 0]
    unconnect_labels = np.unique(red_can_fgbg_lab)[~np.isin(np.unique(red_can_fgbg_lab), connect_labels)]
    red_can_fgbg_lab_clean = mh.labeled.remove_regions(red_can_fgbg_lab, unconnect_labels)
    #red_can_clean_only = np.logical_xor(red_can_fgbg_lab_clean.astype(bool), red_fg_bg_edges)
    #weighted_can_edges = mh.stretch( mh.stretch(redscharr**2) * red_can_clean_only )
    #weighted_ridges = mh.stretch( mh.stretch((redfilt).astype(float)**2)*red_seg_bright_flat*red_ridges )
    
    #ctest = mh.stretch(red_can_fgbg_lab_clean.astype(bool) * redscharr)/255 * 3
    #dtest = red_fg_bg_edges*0
    #etest = red_ridges * red_seg_brightest * 1
    #ftest = ctest + dtest + etest
    #ftest_blur = mh.gaussian_filter(ftest, sigma=(2,2))
    
    # Handle case where testred.max() == 0:
    btest = mh.gaussian_filter( testred/testred.max(), sigma=(2,2))
    # Normalize:
    btest = btest / btest.max()
    
    ctest = mh.gaussian_filter( mh.stretch(red_can_fgbg_lab_clean.astype(bool) * redscharr)/255 * 2, sigma=(2,2))
    dtest = mh.gaussian_filter( red_fg_bg_edges*0, sigma=(2,2))
    
    etest = mh.gaussian_filter( red_ridges * 1, sigma=(2,2))
    # Normalize:
    etest = etest / etest.max() * 0.5
    
    ftest2 = ctest + dtest + etest
    ftest2 = btest + etest

    ### These green edges seem to look very promising!
    #red_edges = np.copy(ftest_blur)
    red_edges = np.copy(ftest2)
    #grn_edges = np.max([mh.stretch(flatscharr), mh.stretch(ftest_blur)], axis=0)
    red_ridge_weight = mh.stretch(filters.gaussian(testred, sigma = 8))#disk_radius / 2))
    red_ridges_weighted = red_ridges * red_ridge_weight
    #red_edges = mh.stretch(morphology.closing(testred, selem=morphology.disk( 1 )))
    red_edges = mh.stretch(testred)
    red_edges[np.where(red_ridges_weighted)] = red_ridges_weighted[np.where(red_ridges_weighted)]
    #rank_mean_red_edges = filters.rank.mean(red_edges.astype(np.uint16), selem=morphology.disk(5))
    red_edges_gauss = mh.stretch(filters.gaussian(red_edges, sigma=5))
    ###

    
    #red_seg_rw = segmentation.random_walker(red_edges, spots_all - ~red_seg_brightest*1)
    #red_seg = segmentation.watershed(rank_mean_red_edges, spots_all, mask=red_seg_bright_flat)
    #red_seg = segmentation.watershed(red_edges_gauss, spots_all, mask=red_seg_bright_flat)
    red_seg = segmentation.watershed(red_edges_gauss, spots_all, mask=red_seg_bright_flat)
    #red_seg = segmentation.watershed(red_edges, spots_all, mask=red_seg_bright_flat)
    #red_seg_join = segmentation.join_segmentations(red_seg_rw, red_seg)
    #plt.imshow(mh.overlay(image_of_cellpair_as_array[:,:,0], red=red_fg_bg_edges, green=red_ridges*red_seg_bright_flat))

    ### If any of the spots in spots_all give rise to a region that is smaller
    ### than particle_fitler*0.1 then remove that spots from spots_all and try again:
    # Determine size of labeled regions:
    sizes_red = np.array([np.count_nonzero(red_seg == x) for x in np.unique(red_seg)])
    # Identify which regions are too small:
    too_small_red = np.unique(red_seg)[sizes_red < particle_filter * 0.1]
    # Identify which seeds in spots_all are red:
    red_seeds = np.unique(spots_all * max_red)
    # Return only the regions that are too small and that are part of the red seeds:
    too_small_red = too_small_red[np.nonzero(too_small_red * [x in red_seeds for x in too_small_red])]
    # Remove the red seeds that are too small:
    spots_all_clean = mh.labeled.remove_regions(spots_all, too_small_red)
    # And try segmentation again:
    red_seg = segmentation.watershed(red_edges_gauss, spots_all_clean, mask=red_seg_bright_flat)

    #grn_lapl_raw = filters.laplace(filters.gaussian(grnchan, sigma=1))
    #grn_lapl = grn_lapl_raw * grn_seg_bright_flat
    #grn_grad_raw = ~mh.stretch(grn_lapl)
    #grn_auto = filters.rank.autolevel(grnchan, selem=morphology.disk(5))
    #grn_grad_raw = ~mh.stretch(filters.rank.gradient(grnchan, selem=morphology.disk(1)))/255
    grn_grad_raw = ~mh.stretch(filters.rank.gradient(grnfilt, selem=morphology.disk(1)))/255
    grn_grad = grn_grad_raw * ((~grnfilt).astype(float)**2)
    grn_grad = mh.stretch(grn_grad)
    grn_ridges_ax0 = np.zeros(grn_grad.shape)
    grn_ridges_ax1 = np.zeros(grn_grad.shape)
    grn_ridges_ax0[argrelextrema(grn_grad, np.greater, axis=0, order=15)] = 1
    grn_ridges_ax1[argrelextrema(grn_grad, np.greater, axis=1, order=15)] = 2
    #grn_ridges_ax0[argrelextrema(grn_grad, np.greater, axis=0, order=5)] = 1
    #grn_ridges_ax1[argrelextrema(grn_grad, np.greater, axis=1, order=5)] = 2
    grn_ridges = morphology.binary_closing(np.logical_or(grn_ridges_ax0, grn_ridges_ax1))
    #plt.imshow(mh.overlay(grnchan, green=grn_seg_brightest * grn_ridges))
    


    grn_fg_bg_edges = mh.borders(grn_seg_bright_flat) * grn_seg_bright_flat
    
    
    
    grn_canny_edges = feature.canny(image_of_cellpair_as_array[:,:,1])
    mask_grn = ~np.logical_xor(grn_canny_edges, morphology.remove_small_holes(grn_canny_edges) )
    
    testrange = np.arange(52,256)
    testcansgrn = []
    for i in testrange:
        testcansgrn.append(feature.canny(~grnfilt, sigma = 0, high_threshold = i, mask = mask_grn))
        testcansgrn.append(feature.canny(~grnfilt, sigma = 1, high_threshold = i, mask = mask_grn))
        testcansgrn.append(feature.canny(~image_of_cellpair_as_array[:,:,1], sigma = 1, high_threshold = i, mask = mask_grn))
        testcansgrn.append(feature.canny(~image_of_cellpair_as_array[:,:,1], sigma = 2, high_threshold = i, mask = mask_grn))
        #testcansgrn.append(get_masked_canny(~grnfilt, particle_filter, initial_mask_grn, 0, 0, low_threshold = 26, high_threshold = i)[1])
        #testcansgrn.append(get_masked_canny(~grnfilt, particle_filter, initial_mask_grn, 1, 0, low_threshold = 26, high_threshold = i)[1])
        #testcansgrn.append(get_masked_canny(~image_of_cellpair_as_array[:,:,1], particle_filter, initial_mask_grn, 1, 0, low_threshold = 26, high_threshold = i)[1])
        #testcansgrn.append(get_masked_canny(~image_of_cellpair_as_array[:,:,1], particle_filter, initial_mask_grn, 2, 0, low_threshold = 26, high_threshold = i)[1])
    testgrn = np.add.reduce(testcansgrn)
    testgrn *= ~grnfilt

#    grn_edge_hist = mh.fullhistogram(mh.stretch(testgrn))[1:]
#    grn_edge_hist = grn_edge_hist * np.arange(1, len(grn_edge_hist)+1)
    n_spots = len(np.unique(spots_grn)[1:-1])
#    second_tallest_peak_indice = np.sort(grn_edge_hist)[-1*(2*n_spots):]
#    second_tallest_peak = np.where(np.logical_or.reduce([grn_edge_hist == i for i in second_tallest_peak_indice]))
#    if testgrn.any():
#        testgrn4 = apply_hysteresis_threshold(mh.stretch(testgrn), low=0, high=np.min(second_tallest_peak))
#    else:
#        testgrn4 = np.zeros(testgrn.shape)
#    ### Use testgrn4 to filter out very blurry edges:
#    testgrn *= testgrn4

### Alternate way of getting testgrn4:
    ### Give all disconnected edges from testgrn a different label:
    labeled_can_grn = mh.label(testgrn.astype(bool), Bc=np.ones((3,3)))[0]
    ### Return the maximum value for each disconnected edge (but skip the first
    ### region which is where labeled_can_grn == 0):
    lab_can_maxs_grn = [np.max(testgrn[np.where(labeled_can_grn == x)]) for x in np.unique(labeled_can_grn)[1:]]
    ### Use these maximas to find the n+1 most common edge, where n is the 
    ### number of spots (or seed points) used for segmentation:
    if testgrn.any():
        unique_edge_labels_grn = np.unique(labeled_can_grn)[1:]
        sorted_edge_labels_grn = unique_edge_labels_grn[np.argsort(lab_can_maxs_grn)]
        edges_to_use_grn = sorted_edge_labels_grn[-1*(n_spots+1):]
        testgrn4 = np.logical_or.reduce([labeled_can_grn == x for x in edges_to_use_grn])
        img_frame = np.zeros(testgrn4.shape)
        img_frame = set_img_edges_vals(img_frame, True)
        missing_edge_px = np.logical_and(img_frame, mh.dilate(testgrn4))
        testgrn4 += missing_edge_px
    else:
        testgrn4 = np.zeros(testgrn.shape)
    ###

    #grn_canny_edges = np.copy(testgrn4)

    #grn_edges = np.max([mh.stretch(allscharr), mh.stretch(grn_fg_bg_edges), mh.stretch(grn_ridges * grn_seg_bright_flat)], axis=0)
    grn_edges = np.max([mh.stretch(allscharr), mh.stretch(grn_fg_bg_edges), mh.stretch(mh.gaussian_filter(grn_ridges * grn_seg_bright_flat, sigma=(2,2)))], axis=0)


    ### More testing:
    grn_can_fgbg = np.logical_or(grn_canny_edges, grn_fg_bg_edges)
    grn_can_fgbg_lab = morphology.label(grn_can_fgbg)
    connect_labels = np.unique(grn_fg_bg_edges * grn_can_fgbg_lab)
    connect_labels = connect_labels[connect_labels > 0]
    unconnect_labels = np.unique(grn_can_fgbg_lab)[~np.isin(np.unique(grn_can_fgbg_lab), connect_labels)]
    grn_can_fgbg_lab_clean = mh.labeled.remove_regions(grn_can_fgbg_lab, unconnect_labels)
    #grn_can_clean_only = np.logical_xor(grn_can_fgbg_lab_clean.astype(bool), grn_fg_bg_edges)
    #weighted_can_edges = mh.stretch( mh.stretch(grnscharr**2) * grn_can_clean_only )
    #weighted_ridges = mh.stretch( mh.stretch((grnfilt).astype(float)**2)*grn_seg_bright_flat*grn_ridges )
    
    #ctest = mh.stretch(grn_can_fgbg_lab_clean.astype(bool) * grnscharr)/255 * 3
    #dtest = grn_fg_bg_edges*0
    #etest = grn_ridges * grn_seg_brightest * 1
    #ftest = ctest + dtest + etest
    #ftest_blur = mh.gaussian_filter(ftest, sigma=(2,2))
    
    btest = mh.gaussian_filter( testgrn/testgrn.max(), sigma=(5,5))
    # Normalize:
    btest = btest / btest.max()
    
    ctest = mh.gaussian_filter( mh.stretch(grn_can_fgbg_lab_clean.astype(bool) * grnscharr)/255 * 2, sigma=(2,2))
    dtest = mh.gaussian_filter( grn_fg_bg_edges*0, sigma=(2,2))
    
    etest = mh.gaussian_filter( grn_ridges * 1, sigma=(2,2))
    # Normalize:
    etest = (etest / etest.max()) * 0.5
    
    ftest2 = ctest + dtest + etest
    ftest2 = btest + etest

    ### These green edges seem to look very promising!
    #grn_edges = np.copy(ftest_blur)
    grn_edges = np.copy(ftest2)
    #grn_edges = np.max([mh.stretch(flatscharr), mh.stretch(ftest_blur)], axis=0)
    grn_ridge_weight = mh.stretch(filters.gaussian(testgrn, sigma = 8))#disk_radius / 2))
    grn_ridges_weighted = grn_ridges * grn_ridge_weight
    #grn_edges = mh.stretch(morphology.closing(testgrn, selem=morphology.disk( 1 )))
    grn_edges = mh.stretch(testgrn)
    grn_edges[np.where(grn_ridges_weighted)] = grn_ridges_weighted[np.where(grn_ridges_weighted)]
    #rank_mean_grn_edges = filters.rank.mean(grn_edges.astype(np.uint16), selem=morphology.disk(5))
    grn_edges_gauss = mh.stretch(filters.gaussian(grn_edges, sigma=5))
    ###
    
    #grn_seg_rw = segmentation.random_walker(grn_edges, spots_all - ~grn_seg_brightest*1)
    #grn_seg = segmentation.watershed(rank_mean_grn_edges, spots_all, mask=grn_seg_bright_flat)
    grn_seg = segmentation.watershed(grn_edges_gauss, spots_all, mask=grn_seg_bright_flat)
    #grn_seg = segmentation.watershed(grn_edges, spots_all, mask=grn_seg_bright_flat)
    #grn_seg_join = segmentation.join_segmentations(grn_seg_rw, grn_seg)
    #plt.imshow(mh.overlay(image_of_cellpair_as_array[:,:,1], red=grn_fg_bg_edges, green=grn_ridges*grn_seg_bright_flat))
    #plt.imshow(mh.overlay(image_of_cellpair_as_array[:,:,1], red=mh.borders(grn_seg)))

    ### If any of the spots in spots_all give rise to a region that is smaller
    ### than particle_fitler*0.1 then remove that spots from spots_all and try again:
    # Determine size of labeled regions:
    sizes_grn = np.array([np.count_nonzero(grn_seg == x) for x in np.unique(grn_seg)])
    # Identify which regions are too small:
    too_small_grn = np.unique(grn_seg)[sizes_grn < particle_filter * 0.1]
    # Identify which seeds in spots_all are grn:
    grn_seeds = np.unique(spots_all * max_grn)
    # Return only the regions that are too small and that are part of the grn seeds:
    too_small_grn = too_small_grn[np.nonzero(too_small_grn * [x in grn_seeds for x in too_small_grn])]
    # Remove the grn seeds that are too small:
    spots_all_clean = mh.labeled.remove_regions(spots_all_clean, too_small_grn)
    # And try segmentation again:
    grn_seg = segmentation.watershed(grn_edges_gauss, spots_all_clean, mask=grn_seg_bright_flat)

###############################################################################



    ### Fill in any holes that might be present in each region:
    # Return the numerical identifier of each region:
    red_regions = np.unique(red_seg)
    # only consider non-zero regions:
    red_regions = red_regions[np.nonzero(red_regions)]
    # Do the hole-filling for each region:
    red_seg_filled_list = [scp_morph.binary_fill_holes(red_seg == x) * x for x in red_regions]
    # Turn the list back in to a flattened array that is 2D:
    red_seg_filled = np.add.reduce(red_seg_filled_list, axis=0)

    #### Remove regions that touch the borders:
    red_seg_clean = mh.labeled.remove_bordering(red_seg_filled)
    ### Remove the regions that are smaller than the size_filter:
    # Determine size of labeled regions:
    sizes_red = np.array([np.count_nonzero(red_seg_clean == x) for x in np.unique(red_seg_clean)])
    # Identify which regions are too small:
    too_small_red = np.unique(red_seg_clean)[sizes_red < size_filter]
    # Remove the regions that are too small
    red_seg_clean = mh.labeled.remove_regions(red_seg_clean, too_small_red)
    
    ### Remove red regions with higher median green brightness than the median
    ### red brightness:
    # Return the numerical identifier of the regions that are still left:
    red_regions_clean = np.unique(red_seg_clean)
    # only consider non-zero regions:
    red_regions_clean = red_regions_clean[np.nonzero(red_regions_clean)]
    # return the median brightness for each region in the red and grn channels
    # as 1D arrays:
    red_seg_clean_medians_red = np.asarray([np.median(redfilt[np.where(red_seg_clean == x)]) for x in red_regions_clean])
    red_seg_clean_medians_grn = np.asarray([np.median(grnfilt[np.where(red_seg_clean == x)]) for x in red_regions_clean])
    # Find which regions have a brighter median in the green than the red chan:
    too_dim_red = red_regions_clean * \
        (red_seg_clean_medians_grn >= red_seg_clean_medians_red)
    # Remove zeros:
    too_dim_red = too_dim_red[np.nonzero(too_dim_red)]
    # Return only the regions with a median fluorescence brighter than what is
    # found in the other channel:
    red_seg_cleaner = mh.labeled.remove_regions(red_seg_clean, too_dim_red)
    
    # Repeat the filling procedure:
    red_regions_cleaner = np.unique(red_seg_cleaner)
    red_regions_cleaner = red_regions_cleaner[np.nonzero(red_regions_cleaner)]
    red_seg_cleaner_filled_list = [scp_morph.binary_fill_holes(red_seg_cleaner == x) * x for x in red_regions_cleaner]
    red_seg_cleaner_filled = np.add.reduce(red_seg_cleaner_filled_list, axis=0)
    # Handle the case where nothing showed up in the red channel:
    if red_seg_cleaner_filled.shape == image_of_cellpair_as_array[:,:,0].shape:
        red_seg_cleaner = np.copy(red_seg_cleaner_filled)
    else:
        red_seg_cleaner = np.zeros(image_of_cellpair_as_array[:,:,0].shape, dtype=np.uint8)

    # Also return all of the removed regions in a separate array:
    red_seg_removed = np.logical_xor(red_seg_cleaner.astype(bool), red_seg.astype(bool)) * red_seg

    
    
###############################################################################
    
    ### Find the centre of each cell for later use:
    # Return the numerical identifier of the regions that are still left:
    red_regions_cleaner = np.unique(red_seg_cleaner)
    # only consider non-zero regions:
    red_regions_cleaner = red_regions_cleaner[np.nonzero(red_regions_cleaner)]
    # return a list of an array containing only 1 region for each region:
    red_seg_cleaner_list = [red_seg_cleaner == x for x in red_regions_cleaner]
    # return a list of an array containing the distance map of only 1 region 
    # for each region:
    red_seg_cleaner_dist = [mh.distance(x) for x in red_seg_cleaner_list]
    # return a list of an array containing the max of only 1 region for each 
    # region:
    red_seg_cleaner_max = [mh.regmax(x, Bc=structuring_element_to_find_maxima) for x in red_seg_cleaner_dist]
    # flatten this list into a 2D image:
    if np.any(red_seg_cleaner_max):
        red_seg_cleaner_max = np.logical_or.reduce(red_seg_cleaner_max)
        red_seg_cleaner_spots = morphology.label(red_seg_cleaner_max)
        #red_seg_cleaner_spots = red_seg_cleaner_max * 1
    else:
        red_seg_cleaner_max = np.zeros(redfilt.shape)
        red_seg_cleaner_spots = np.zeros(redfilt.shape)


###############################################################################
    


    ### Fill in any holes that might be present in each region:
    # Return the numerical identifier of each region:
    grn_regions = np.unique(grn_seg)
    # only consider non-zero regions:
    grn_regions = grn_regions[np.nonzero(grn_regions)]
    # Do the hole-filling for each region:
    grn_seg_filled_list = [scp_morph.binary_fill_holes(grn_seg == x) * x for x in grn_regions]
    # Turn the list back in to a flattened array that is 2D:
    grn_seg_filled = np.add.reduce(grn_seg_filled_list, axis=0)
    
    ### Remove regions that touch the borders:
    grn_seg_clean = mh.labeled.remove_bordering(grn_seg_filled)
    ### Remove the regions that are smaller than the size_filter:
    # Determine size of labeled regions:
    sizes_grn = np.array([np.count_nonzero(grn_seg_clean == x) for x in np.unique(grn_seg_clean)])
    # Identify which regions are too small:
    too_small_grn = np.unique(grn_seg_clean)[sizes_grn < size_filter]
    # Remove the regions that are too small
    grn_seg_clean = mh.labeled.remove_regions(grn_seg_clean, too_small_grn)
    
    ### Remove green regions with higher median red brightness than the median
    ### green brightness:
    # Return the numerical identifier of the regions that are still left:
    grn_regions_clean = np.unique(grn_seg_clean)
    # only consider non-zero regions:
    grn_regions_clean = grn_regions_clean[np.nonzero(grn_regions_clean)]
    # return the median brightness for each region in the red and grn channels
    # as 1D arrays:
    grn_seg_clean_medians_red = np.asarray([np.median(redfilt[np.where(grn_seg_clean == x)]) for x in grn_regions_clean])
    grn_seg_clean_medians_grn = np.asarray([np.median(grnfilt[np.where(grn_seg_clean == x)]) for x in grn_regions_clean])
    # Find which regions have a brighter median in the red than the green chan:
    too_dim_grn = grn_regions_clean * \
        (grn_seg_clean_medians_red >= grn_seg_clean_medians_grn)
    # Remove zeros:
    too_dim_grn = too_dim_grn[np.nonzero(too_dim_grn)]
    # Return only the regions with a median fluorescence brighter than what is
    # found in the other channel:
    grn_seg_cleaner = mh.labeled.remove_regions(grn_seg_clean, too_dim_grn)
    
    # Repeat the filling procedure:
    grn_regions_cleaner = np.unique(grn_seg_cleaner)
    grn_regions_cleaner = grn_regions_cleaner[np.nonzero(grn_regions_cleaner)]
    grn_seg_cleaner_filled_list = [scp_morph.binary_fill_holes(grn_seg_cleaner == x) * x for x in grn_regions_cleaner]
    grn_seg_cleaner_filled = np.add.reduce(grn_seg_cleaner_filled_list, axis=0)
    
    # Handle the case where there was nothing in the green channel:
    if grn_seg_cleaner_filled.shape == image_of_cellpair_as_array[:,:,1].shape:
        grn_seg_cleaner = np.copy(grn_seg_cleaner_filled)
    else:
        grn_seg_cleaner = np.zeros(image_of_cellpair_as_array[:,:,1].shape, dtype=np.uint8)

    # Also return all of the removed regions in a separate array:
    grn_seg_removed = np.logical_xor(grn_seg_cleaner.astype(bool), grn_seg.astype(bool)) * grn_seg
    
    
    
###############################################################################
    
    ### Find the centre of each cell for later use:
    # Return the numerical identifier of the regions that are still left:
    grn_regions_cleaner = np.unique(grn_seg_cleaner)
    # only consider non-zero regions:
    grn_regions_cleaner = grn_regions_cleaner[np.nonzero(grn_regions_cleaner)]
    # return a list of an array containing only 1 region for each region:
    grn_seg_cleaner_list = [grn_seg_cleaner == x for x in grn_regions_cleaner]
    # return a list of an array containing the distance map of only 1 region 
    # for each region:
    grn_seg_cleaner_dist = [mh.distance(x) for x in grn_seg_cleaner_list]
    # return a list of an array containing the max of only 1 region for each 
    # region:
    grn_seg_cleaner_max = [mh.regmax(x, Bc=structuring_element_to_find_maxima) for x in grn_seg_cleaner_dist]
    # flatten this list into a 2D image:
    if np.any(grn_seg_cleaner_max):
        grn_seg_cleaner_max = np.logical_or.reduce(grn_seg_cleaner_max)
        grn_seg_cleaner_spots = morphology.label(grn_seg_cleaner_max)
        #grn_seg_cleaner_spots = grn_seg_cleaner_max * 2
    else:
        grn_seg_cleaner_max = np.zeros(grnfilt.shape)
        grn_seg_cleaner_spots = np.zeros(grnfilt.shape)







    blu_rank_otsu = np.zeros(image_of_cellpair_as_array[:,:,2].shape)
    blu_seg_cleaner = np.zeros(image_of_cellpair_as_array[:,:,2].shape)
    blu_seg_removed = np.zeros(image_of_cellpair_as_array[:,:,2].shape)
    blu_seg_cleaner_spots = np.zeros(image_of_cellpair_as_array[:,:,2].shape)
    blu_edges = np.zeros(image_of_cellpair_as_array[:,:,2].shape)

    cellthresh = np.dstack([red_rank_otsu, grn_rank_otsu, blu_rank_otsu])
    cellareas_rough_in = np.dstack([red_seg_cleaner, grn_seg_cleaner, blu_seg_cleaner])
    cellareas_rough_out = np.dstack([red_seg_removed, grn_seg_removed, blu_seg_removed])
    #spots = np.dstack([red_seg_cleaner_spots, grn_seg_cleaner_spots, blu_seg_cleaner_spots])
    spots = np.dstack([red_seg_cleaner_spots, grn_seg_cleaner_spots, blu_seg_cleaner_spots])
    edge_map = np.dstack([red_edges, grn_edges, blu_edges])

    return cellthresh, cellareas_rough_out, cellareas_rough_in, spots, Bc3D, edge_map
 


    
def separate_red_and_green_cells(rgb_image_of_cells, rgb_image_of_central_cells_areas_as_bool, structuring_element_to_find_maxima, edge_map):
    
    surf_mask = rgb_image_of_central_cells_areas_as_bool.astype(bool)
    surf_mask = np.logical_or(*[surf_mask[:,:,i] for i in range(surf_mask.shape[2])])
    surf_mask = ~scp_morph.binary_fill_holes(surf_mask)
    surf = ~rgb_image_of_cells
    surf_flat = surf[:,:,0].astype(np.uint64) + surf[:,:,1].astype(np.uint64) + surf[:,:,2].astype(np.uint64)
    masked_surf = mh.stretch(np.copy(surf_flat))#.astype(int)
    masked_surf[np.where(surf_mask)] = 255
    redfilt = mh.gaussian_filter(rgb_image_of_cells[:,:,0], (gaussian_filter_value_red, gaussian_filter_value_red))
    grnfilt = mh.gaussian_filter(rgb_image_of_cells[:,:,1], (gaussian_filter_value_grn, gaussian_filter_value_grn))
    
    overlapping_areas = np.logical_and(rgb_image_of_central_cells_areas_as_bool[:,:,0], rgb_image_of_central_cells_areas_as_bool[:,:,1])
    overlap_redareas = overlapping_areas * (redfilt > grnfilt)
    overlap_grnareas = overlapping_areas * (redfilt < grnfilt)
    uncertain_areas = overlapping_areas* (redfilt == grnfilt)
    
    redareas = np.logical_xor(rgb_image_of_central_cells_areas_as_bool[:,:,0], overlap_grnareas)
    grnareas = np.logical_xor(rgb_image_of_central_cells_areas_as_bool[:,:,1], overlap_redareas)
    # Handle case where a hole exists in redareas or grnareas:
    redareas_filled = scp_morph.binary_fill_holes(redareas)
    hole_in_redareas = np.logical_xor(redareas, redareas_filled)
    grnareas_filled = scp_morph.binary_fill_holes(grnareas)
    hole_in_grnareas = np.logical_xor(grnareas, grnareas_filled)
    #hole_in_both = np.logical_and(hole_in_redareas, hole_in_grnareas)
    ### Actually, a hole in both is not possible.
    
    # Handle case where there is a hole in one region but not another:
    # Because it is not possible for there to be a hole in both regions in the
    # Same spot, it does not matter what order the following 2 lines are done.
    redareas_filled[np.where(hole_in_grnareas)] = False
    grnareas_filled[np.where(hole_in_redareas)] = False
    
    # Handle case of uncertain_areas (where redfilt == grnfiflt):
    # If any part of uncertain_areas is now covered by the areas_filled, then
    # remove that part from uncertain_areas:
    uncertain_areas[np.where(redareas_filled)] = False
    uncertain_areas[np.where(grnareas_filled)] = False
    # For any leftover uncertain areas, just use the watershed algorithm:
    
    # Assign seedpoints based on geometric centre of redareas or grnareas:
    dist_red = mh.distance(rgb_image_of_central_cells_areas_as_bool[:,:,0])
    if dist_red.any() == True:
        max_red_locs = np.unravel_index(np.argmax(dist_red), dist_red.shape)
        max_red = np.zeros(dist_red.shape, dtype='bool')
        max_red[max_red_locs] = True
    else:
        max_red = np.zeros(dist_red.shape, dtype='bool')
        
        
    dist_grn = mh.distance(rgb_image_of_central_cells_areas_as_bool[:,:,1])
    if dist_grn.any() == True:
        max_grn_locs = np.unravel_index(np.argmax(dist_grn), dist_grn.shape)
        max_grn = np.zeros(dist_grn.shape, dtype='bool')
        max_grn[max_grn_locs] = True
    else:
        max_grn = np.zeros(dist_grn.shape, dtype='bool')
        
        
    dist_blu = mh.distance(rgb_image_of_central_cells_areas_as_bool[:,:,2])
    if dist_blu.any() == True:
        max_blu_locs = np.unravel_index(np.argmax(dist_blu), dist_blu.shape)
        max_blu = np.zeros(dist_blu.shape, dtype='bool')
        max_blu[max_blu_locs] = True
    else:
        max_blu = np.zeros(dist_blu.shape, dtype='bool')
    
    # Assign a value to each channel's seed point:
    spots = max_red*1 + max_grn*2 + max_blu*3
    # Create an elevation map for the watershed to work on based on edges:
    #elevation_map = filters.scharr(surf_flat.astype(float))
    elevation_map = mh.stretch( filters.gaussian(edge_map[:,:,0] / 2. + edge_map[:,:,1] / 2., sigma=5) )
    # Segment cells:
    flat_seg = segmentation.watershed(elevation_map, spots, mask = (~surf_mask))
    # Assign any leftover uncertain_areas regions based on the segmentation:
    probably_red = (flat_seg * uncertain_areas).astype(bool)
    probably_grn = (flat_seg * uncertain_areas).astype(bool)
    
    # Assign the probable labels to the red and grn areas:
    redareas_filled += probably_red
    grnareas_filled += probably_grn
    bluareas = np.zeros(rgb_image_of_cells[:,:,2].shape, dtype=int)
    
    # This version of cellareas gives preferential weight to image brightness
    # when deciding if part of a cell is likely to be red or green:
    cellareas = np.dstack( (redareas_filled*1, grnareas_filled*2, bluareas) )
    
    # While this version of cellareas just does a segmentation using the edge
    # map that was primarily derived from multiple canny edge detections:
    #cellareas2 = np.dstack( ( (flat_seg == 1) * 1, (flat_seg == 2) * 2, bluareas ) )
    
    
    cell_maxima = np.dstack((max_red, max_grn, max_blu))
    

    return cellareas, cell_maxima




def remove_bordering_and_small_regions(segmented_image_of_cellareas):
    ### To remove cells touching the edges of the image:                
    no_border_cells_red = mh.labeled.remove_bordering(segmented_image_of_cellareas[:,:,0])
    no_border_cells_red = mh.label(no_border_cells_red)[0]
    no_border_cells_grn = mh.labeled.remove_bordering(segmented_image_of_cellareas[:,:,1])
    no_border_cells_grn = mh.label(no_border_cells_grn)[0]
    no_border_cells_blu = mh.labeled.remove_bordering(segmented_image_of_cellareas[:,:,2])
    no_border_cells_blu = mh.label(no_border_cells_blu)[0]
    no_border_cells = np.dstack((no_border_cells_red, no_border_cells_grn, no_border_cells_blu))       
    
    ### Remove any areas that are too small
    sizes = mh.labeled.labeled_size(no_border_cells)
    iter = sizes
    n = 3
    thirdlargestregion = nth_max(n, iter)
    too_small = np.where(sizes < thirdlargestregion)
    no_border_cells = mh.labeled.remove_regions(no_border_cells, too_small)

    sizes = mh.labeled.labeled_size(no_border_cells[:,:,0])
    iter = sizes
    n = 2
    secondlargestregion = nth_max(n, iter)
    too_small = np.where(sizes < secondlargestregion)
    no_border_cells[:,:,0] = mh.labeled.remove_regions(no_border_cells[:,:,0], too_small)
#        too_small_still = np.where(sizes < size_filter)
#        no_border_cells[:,:,0] = mh.labeled.remove_regions(no_border_cells[:,:,0], too_small_still)


    sizes = mh.labeled.labeled_size(no_border_cells[:,:,1])
    iter = sizes
    n = 2
    secondlargestregion = nth_max(n, iter)
    too_small = np.where(sizes < secondlargestregion)
    no_border_cells[:,:,1] = mh.labeled.remove_regions(no_border_cells[:,:,1], too_small)
#        too_small_still = np.where(sizes < size_filter)
#        no_border_cells[:,:,1] = mh.labeled.remove_regions(no_border_cells[:,:,1], too_small_still)


    sizes = mh.labeled.labeled_size(no_border_cells[:,:,2])
    iter = sizes
    n = 2
    secondlargestregion = nth_max(n, iter)
    too_small = np.where(sizes < secondlargestregion)
    no_border_cells[:,:,2] = mh.labeled.remove_regions(no_border_cells[:,:,2], too_small)
#        too_small_still = np.where(sizes < size_filter)
#        no_border_cells[:,:,2] = mh.labeled.remove_regions(no_border_cells[:,:,2], too_small_still)
    
    return no_border_cells




def clean_up_borders(borders_of_interest):
#    temp_pad_magnitude = 1
#    original_borders_of_interest = pad_img_edges(np.copy(borders_of_interest).astype(int), 0, temp_pad_magnitude)
    original_borders_of_interest = np.copy(borders_of_interest).astype(int)
    borders_of_interest = np.zeros(original_borders_of_interest.shape, dtype=int)
    cell_of_interest_red = scp_morph.binary_fill_holes(original_borders_of_interest[:,:,0])
    inner_area_red = np.logical_xor(cell_of_interest_red, original_borders_of_interest[:,:,0].astype(bool))
    inner_area_red = mh.label(inner_area_red)[0]
    sizes = mh.labeled.labeled_size(inner_area_red)
    iter = sizes
    n = 2
    secondlargestregion = nth_max(n, iter)
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
    iter = sizes
    n = 2
    secondlargestregion = nth_max(n, iter)
    too_small = np.where(sizes < secondlargestregion)
    inner_area_grn = mh.labeled.remove_regions(inner_area_grn, too_small)
    temp_Bc = mh.get_structuring_elem(inner_area_grn, 4)
#    temp_Bc = mh.get_structuring_elem(inner_area_grn, 8)
    for val in zip(*np.where(inner_area_grn)):
        if 2 in original_borders_of_interest[sub(val[0],1):val[0]+2, sub(val[1],1):val[1]+2, 1] * temp_Bc:
            borders_of_interest[sub(val[0],1):val[0]+2, sub(val[1],1):val[1]+2, 1] += original_borders_of_interest[sub(val[0],1):val[0]+2, sub(val[1],1):val[1]+2, 1] * temp_Bc
    borders_of_interest[:,:,1] = borders_of_interest[:,:,1].astype(bool) * 2

### Make sure that cleaning up the borders this way doesn't accidentally make
### red and green cell borders overlap:
#    for (i,j,k), val in np.ndenumerate(borders_of_interest):
#        if np.count_nonzero(borders_of_interest[i,j,:]) > 1:
#            borders_of_interest[i,j,:] = original_borders_of_interest[i,j,:]
           
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
    while num_cleanings < 4:
    ### Remove any excess corner pixels:
        extra_type1 = np.c_[[0,1,0], [1,1,0], [0,0,0]]
        extra_type2 = np.c_[[0,1,0], [0,1,1], [0,0,0]]
        extra_type3 = np.c_[[0,0,0], [1,1,0], [0,1,0]]
        extra_type4 = np.c_[[0,0,0], [0,1,1], [0,1,0]]
    
    
#        curve_excess_bord = np.zeros(borders_of_interest.shape, np.dtype(np.int32))
        for (i,j,k), surfval in np.ndenumerate(borders_of_interest):
            if surfval != 0:
                if (len(np.where(extra_type1 * borders_of_interest[sub(i,1):i+2, sub(j,1):j+2, k] == surfval)[0]) > 2 or
                    len(np.where(extra_type2 * borders_of_interest[sub(i,1):i+2, sub(j,1):j+2, k] == surfval)[0]) > 2 or
                    len(np.where(extra_type3 * borders_of_interest[sub(i,1):i+2, sub(j,1):j+2, k] == surfval)[0]) > 2 or
                    len(np.where(extra_type4 * borders_of_interest[sub(i,1):i+2, sub(j,1):j+2, k] == surfval)[0]) > 2
                    ):
                        curve_excess_bord[i,j,k] += borders_of_interest[i,j,k]
                        borders_of_interest[i,j,k] = 0
    
    ### Removing excess corners can make the cellborder discontinuous. To fix it:
    #    count = 0
        count = len(list(zip(*np.where(curve_excess_bord))))
    #    while count < 8:
        while count > 0:        
            for (i,j,k), surfval in np.ndenumerate(borders_of_interest):
            # If you're looking at a cleanborder pixel and had excess corners - 
                if surfval != 0 and curve_excess_bord.any():
                    # then if it's right next to the site of discontinuity:
                    if (
                        (len(np.where(borders_of_interest[sub(i,1):i+2, sub(j,1):j+2, k] == surfval)[0]) == 2)# and
                        ):
#                        print i,j,k
                        try:
                            if ( ((i,j) in np.transpose(np.where(extra_type1 * borders_of_interest[sub(np.where(curve_excess_bord)[0][0],1):np.where(curve_excess_bord)[0][0]+2, sub(np.where(curve_excess_bord)[1][0],1):np.where(curve_excess_bord)[1][0]+2, k]))+np.transpose(np.where(curve_excess_bord))[:,:2][0]-1) and
                                (np.count_nonzero(extra_type1 * borders_of_interest[sub(np.where(curve_excess_bord)[0][0],1):np.where(curve_excess_bord)[0][0]+2, sub(np.where(curve_excess_bord)[1][0],1):np.where(curve_excess_bord)[1][0]+2, k]) == 2 )
                                ):
                                borders_of_interest[sub(i,1):i+2, sub(j,1):j+2, k] += curve_excess_bord[sub(i,1):i+2, sub(j,1):j+2, k]
                                borders_of_interest[i,j,k] = 0
    #                                    excess_bord_arr[np.transpose(np.where(excess_bord_arr))[:,:2][0][0], np.transpose(np.where(excess_bord_arr))[:,:2][0][1], k] = 0
                                print(count, "curve border extra_type1", (i,j,k))
                                break
                            else:
    #                            actual_excess += 1
                                print(count, (i,j,k))
                                pass
                        except ValueError:
                            print('fail')
                            pass
                        try:
                            if ( ((i,j) in np.transpose(np.where(extra_type2 * borders_of_interest[sub(np.where(curve_excess_bord)[0][0],1):np.where(curve_excess_bord)[0][0]+2, sub(np.where(curve_excess_bord)[1][0],1):np.where(curve_excess_bord)[1][0]+2, k]))+np.transpose(np.where(curve_excess_bord))[:,:2][0]-1) and
                                (np.count_nonzero(extra_type2 * borders_of_interest[sub(np.where(curve_excess_bord)[0][0],1):np.where(curve_excess_bord)[0][0]+2, sub(np.where(curve_excess_bord)[1][0],1):np.where(curve_excess_bord)[1][0]+2, k]) == 2 )
                                ):
                                borders_of_interest[sub(i,1):i+2, sub(j,1):j+2, k] += curve_excess_bord[sub(i,1):i+2, sub(j,1):j+2, k]
                                borders_of_interest[i,j,k] = 0
    #                                    excess_bord_arr[np.transpose(np.where(excess_bord_arr))[:,:2][0][0], np.transpose(np.where(excess_bord_arr))[:,:2][0][1], k] = 0
                                print(count, "curve border extra_type2", (i,j,k))
                                break
                            else:
    #                            actual_excess += 1
                                print(count, (i,j,k))
                                pass
                        except ValueError:
                            print('fail')
                            pass
                        try:
                            if ( ((i,j) in np.transpose(np.where(extra_type3 * borders_of_interest[sub(np.where(curve_excess_bord)[0][0],1):np.where(curve_excess_bord)[0][0]+2, sub(np.where(curve_excess_bord)[1][0],1):np.where(curve_excess_bord)[1][0]+2, k]))+np.transpose(np.where(curve_excess_bord))[:,:2][0]-1) and
                                (np.count_nonzero(extra_type3 * borders_of_interest[sub(np.where(curve_excess_bord)[0][0],1):np.where(curve_excess_bord)[0][0]+2, sub(np.where(curve_excess_bord)[1][0],1):np.where(curve_excess_bord)[1][0]+2, k]) == 2 )
                                ):
                                borders_of_interest[sub(i,1):i+2, sub(j,1):j+2, k] += curve_excess_bord[sub(i,1):i+2, sub(j,1):j+2, k]
                                borders_of_interest[i,j,k] = 0
    #                                    excess_bord_arr[np.transpose(np.where(excess_bord_arr))[:,:2][0][0], np.transpose(np.where(excess_bord_arr))[:,:2][0][1], k] = 0
                                print(count, "curve border extra_type3", (i,j,k))
                                break
                            else:
    #                            actual_excess += 1
                                print(count, (i,j,k))
                                pass
                        except ValueError:
                            print('fail')
                            pass
                        try:
                            if ( ((i,j) in np.transpose(np.where(extra_type4 * borders_of_interest[sub(np.where(curve_excess_bord)[0][0],1):np.where(curve_excess_bord)[0][0]+2, sub(np.where(curve_excess_bord)[1][0],1):np.where(curve_excess_bord)[1][0]+2, k]))+np.transpose(np.where(curve_excess_bord))[:,:2][0]-1) and
                                (np.count_nonzero(extra_type4 * borders_of_interest[sub(np.where(curve_excess_bord)[0][0],1):np.where(curve_excess_bord)[0][0]+2, sub(np.where(curve_excess_bord)[1][0],1):np.where(curve_excess_bord)[1][0]+2, k]) == 2 )
                                ):
                                borders_of_interest[sub(i,1):i+2, sub(j,1):j+2, k] += curve_excess_bord[sub(i,1):i+2, sub(j,1):j+2, k]
                                borders_of_interest[i,j,k] = 0
    #                                    excess_bord_arr[np.transpose(np.where(excess_bord_arr))[:,:2][0][0], np.transpose(np.where(excess_bord_arr))[:,:2][0][1], k] = 0
                                print(count, "curve border extra_type4", (i,j,k))
                                break
                            else:
    #                            actual_excess += 1
                                print(count, (i,j,k))
                                pass
                        except ValueError:
                            print('fail')
                            pass
    #                                if (((i,j) in np.transpose(np.where(extra_type1 * cleanborders[sub(np.where(excess_bord_arr)[0][0],1):np.where(excess_bord_arr)[0][0]+2, sub(np.where(excess_bord_arr)[1][0],1):np.where(excess_bord_arr)[1][0]+2, k]))+np.transpose(np.where(excess_bord_arr))[:,:2]-1) or
    #                                    ((i,j) in np.transpose(np.where(extra_type2 * cleanborders[sub(np.where(excess_bord_arr)[0][0],1):np.where(excess_bord_arr)[0][0]+2, sub(np.where(excess_bord_arr)[1][0],1):np.where(excess_bord_arr)[1][0]+2, k]))+np.transpose(np.where(excess_bord_arr))[:,:2]-1) or
    #                                    ((i,j) in np.transpose(np.where(extra_type3 * cleanborders[sub(np.where(excess_bord_arr)[0][0],1):np.where(excess_bord_arr)[0][0]+2, sub(np.where(excess_bord_arr)[1][0],1):np.where(excess_bord_arr)[1][0]+2, k]))+np.transpose(np.where(excess_bord_arr))[:,:2]-1) or
    #                                    ((i,j) in np.transpose(np.where(extra_type4 * cleanborders[sub(np.where(excess_bord_arr)[0][0],1):np.where(excess_bord_arr)[0][0]+2, sub(np.where(excess_bord_arr)[1][0],1):np.where(excess_bord_arr)[1][0]+2, k]))+np.transpose(np.where(excess_bord_arr))[:,:2]-1)
    #                                    ):
    #                                    cleanborders[sub(i,1):i+2, sub(j,1):j+2, k:k+1] += excess_bord_arr[sub(i,1):i+2, sub(j,1):j+2, k:k+1]
    #                                    cleanborders[i,j,k] = 0
    
                            #print i,j,k
    #                            if actual_excess == 8:
    #                                print count, "actual_excess:%d"%actual_excess, (i,j,k), 
    #                                excess_bord_arr[np.transpose(np.where(excess_bord_arr))[:,:2][0][0], np.transpose(np.where(excess_bord_arr))[:,:2][0][1], k] = 0
    #        curve_excess_bord[np.transpose(np.where(curve_excess_bord))[:,:2][0][0], np.transpose(np.where(curve_excess_bord))[:,:2][0][1], k] = 0
    
            if curve_excess_bord.any():
                curve_excess_bord[tuple(np.transpose(np.where(curve_excess_bord))[0])] = 0
    #        else:
    #            break
    #        count += 1
            count -= 1
    #        if curve_excess_bord.any():            
    #            curve_excess_bord[tuple(np.transpose(np.where(curve_excess_bord))[0])] = 0
        num_cleanings += 1
        
### This may need to be removed: ##############################################
### Remove excess corners again:
    extra_type1 = np.c_[[0,1,0], [1,1,0], [0,0,0]]
    extra_type2 = np.c_[[0,1,0], [0,1,1], [0,0,0]]
    extra_type3 = np.c_[[0,0,0], [1,1,0], [0,1,0]]
    extra_type4 = np.c_[[0,0,0], [0,1,1], [0,1,0]]       
#    curve_excess_bord = np.zeros(borders_of_interest.shape, np.dtype(np.int32))
#    
#    for (i,j,k), surfval in np.ndenumerate(borders_of_interest):
#        if surfval != 0:
#            if (len(np.where(extra_type1 * borders_of_interest[sub(i,1):i+2, sub(j,1):j+2, k] == surfval)[0]) > 2 or
#                len(np.where(extra_type2 * borders_of_interest[sub(i,1):i+2, sub(j,1):j+2, k] == surfval)[0]) > 2 or
#                len(np.where(extra_type3 * borders_of_interest[sub(i,1):i+2, sub(j,1):j+2, k] == surfval)[0]) > 2 or
#                len(np.where(extra_type4 * borders_of_interest[sub(i,1):i+2, sub(j,1):j+2, k] == surfval)[0]) > 2
#                ):
#                    curve_excess_bord[i,j,k] += borders_of_interest[i,j,k]
#                    borders_of_interest[i,j,k] = 0
###############################################################################
            
    ### Remove any pixels that are discontinuous with the cell border or that
    ### result in a cusp:
    for (i,j,k), surfval in np.ndenumerate(borders_of_interest):
        if surfval != 0:
            if (len(np.where(borders_of_interest[sub(i,1):i+2, sub(j,1):j+2, sub(k,1):k+2] == surfval)[0]) < 3
                ):
                    borders_of_interest[i,j,k] = 0
 
#    return pad_img_edges(borders_of_interest, 0, -1*temp_pad_magnitude)                   
    return borders_of_interest



def clean_up_contact_borders(rawcontactsurface, cleanborders):
    ### In the event that the contact surface is discontinuous (which happens
    ### occasionally believe it or not) we will fill in any holes by copying
    ### the cellborders values in to the window where the contactsurface exists:
    ###imin1 = min(np.where(rawcontactsurface==1)[0])-2 change to -5 as of 2018-06-21.

    imin1 = min(np.where(rawcontactsurface==1)[0])-5
    imax1 = max(np.where(rawcontactsurface==1)[0])+5

    jmin1 = min(np.where(rawcontactsurface==1)[1])-5
    jmax1 = max(np.where(rawcontactsurface==1)[1])+5

    #kmin1 = min(np.where(rawcontactsurface==1)[2])-2
    #kmax1 = max(np.where(rawcontactsurface==1)[2])+2
 

    imin2 = min(np.where(rawcontactsurface==2)[0])-5
    imax2 = max(np.where(rawcontactsurface==2)[0])+5

    jmin2 = min(np.where(rawcontactsurface==2)[1])-5
    jmax2 = max(np.where(rawcontactsurface==2)[1])+5

    #kmin2 = min(np.where(rawcontactsurface==2)[2])-2
    #kmax2 = max(np.where(rawcontactsurface==2)[2])+2

    ### The part below is taking a really long time with XL Feb 27 crop 4 t023 ###
    ### Try instead to use np.where(cleanborders[imin1:imax1+1, jmin1:jmax1+1,:] == 1)
    ### to move all parts of the cleancellborders btwn imin:imax, jmin:jmax to contactsurface
    ### The part below will fill in the holes in each contact surface:
    filledcontactsurface = np.zeros(rawcontactsurface.shape, np.dtype(np.int32)) + rawcontactsurface
    for (i,j,k), surfval in np.ndenumerate(filledcontactsurface[imin1:imax1+1, jmin1:jmax1+1,:]):
        for (p,q,r), bordval in np.ndenumerate(cleanborders[imin1:imax1+1, jmin1:jmax1+1,:]):
            if ((i,j,k) == (p,q,r) and surfval != bordval and bordval==1):
                filledcontactsurface[i+imin1,j+jmin1,k] = cleanborders[p+imin1,q+jmin1,r]
    for (i,j,k), surfval in np.ndenumerate(filledcontactsurface[imin2:imax2+1, jmin2:jmax2+1,:]):
        for (p,q,r), bordval in np.ndenumerate(cleanborders[imin2:imax2+1, jmin2:jmax2+1,:]):
            if ((i,j,k) == (p,q,r) and surfval != bordval and bordval==2):
                filledcontactsurface[i+imin2,j+jmin2,k] = cleanborders[p+imin2,q+jmin2,r]

    ### If the contact surface is particularly curved this can result in the
    ### contact surface being incorrectly extended for one cell, so chomp away at
    ### end of the contact surface if this is the case:
    false_contacts = np.zeros(filledcontactsurface.shape, np.dtype(np.int32))
    for (i,j,k), surfval in np.ndenumerate(filledcontactsurface):
        if surfval != 0:
            if (len(np.where(filledcontactsurface[sub(i,1):i+2, sub(j,1):j+2, sub(k,1):k+2] == surfval)[0]) <= 2 and
            ((0 in filledcontactsurface[sub(i,1):i+2, sub(j,1):j+2, sub(k,1):k+2]) and 
            (1 in filledcontactsurface[sub(i,1):i+2, sub(j,1):j+2, sub(k,1):k+2]) and 
            (2 in filledcontactsurface[sub(i,1):i+2, sub(j,1):j+2, sub(k,1):k+2])) == False
            ):
                false_contacts[i,j,k] = surfval
                filledcontactsurface[i,j,k] = 0
    
    while false_contacts.any():
        false_contacts = np.zeros(filledcontactsurface.shape)
        for (i,j,k), surfval in np.ndenumerate(filledcontactsurface):
            if surfval != 0:
                if (len(np.where(filledcontactsurface[sub(i,1):i+2, sub(j,1):j+2, sub(k,1):k+2] == surfval)[0]) < 3 and
                ((0 in filledcontactsurface[sub(i,1):i+2, sub(j,1):j+2, sub(k,1):k+2]) and 
                (1 in filledcontactsurface[sub(i,1):i+2, sub(j,1):j+2, sub(k,1):k+2]) and 
                (2 in filledcontactsurface[sub(i,1):i+2, sub(j,1):j+2, sub(k,1):k+2])) == False
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
                if (len(np.where(extra_type1 * contactsurface[sub(i,1):i+2, sub(j,1):j+2, sub(k,1):k+2] == surfval)[0]) > 2 or
                    len(np.where(extra_type2 * contactsurface[sub(i,1):i+2, sub(j,1):j+2, sub(k,1):k+2] == surfval)[0]) > 2 or
                    len(np.where(extra_type3 * contactsurface[sub(i,1):i+2, sub(j,1):j+2, sub(k,1):k+2] == surfval)[0]) > 2 or
                    len(np.where(extra_type4 * contactsurface[sub(i,1):i+2, sub(j,1):j+2, sub(k,1):k+2] == surfval)[0]) > 2
                    ):
                        excess_arr[i,j,k] += contactsurface[i,j,k]
                        contactsurface[i,j,k] = 0
                        
    ### This can result in contactsurface ends that are not actually in contact 
    ### with the other cell, so we need to remove those and replace the falsely
    ### excess pixel: 
        for (i,j,k), surfval in np.ndenumerate(contactsurface):
#            for (i,j,k), surfval in np.ndenumerate(excess_arr):
            if surfval != 0:       
                if ((len(np.where(contactsurface[sub(i,1):i+2, sub(j,1):j+2, sub(k,1):k+2] == surfval)[0]) == 2) and
                    (((0 in contactsurface[sub(i,1):i+2, sub(j,1):j+2, sub(k,1):k+2]) and 
                    (1 in contactsurface[sub(i,1):i+2, sub(j,1):j+2, sub(k,1):k+2]) and 
                    (2 in contactsurface[sub(i,1):i+2, sub( j,1):j+2, sub(k,1):k+2])) == False) and
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
                if ((len(np.where(contactsurface[sub(i,1):i+2, sub(j,1):j+2, sub(k,1):k+2] == surfval)[0]) == 2) and
                    (0 in contactsurface[sub(i,1):i+2, sub(j,1):j+2, sub(k,1):k+2]) and 
                    (1 in contactsurface[sub(i,1):i+2, sub(j,1):j+2, sub(k,1):k+2]) and 
                    (2 in contactsurface[sub(i,1):i+2, sub( j,1):j+2, sub(k,1):k+2]) and
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

    ### This should be the last part and I hope to god that this process will robustly find corners:
    for (i,j,k), value in np.ndenumerate(contactsurface):
        if value != 0:
            if len(np.where(contactsurface[sub(i,1):i+2, sub(j,1):j+2, sub(k,1):k+2] == value)[0]) == 2:
                contactcorners[i,j,k] = contactsurface[i,j,k]

    return contactcorners, contactsurface, noncontactborders



def pixel_by_pixel_curvatures(single_channel_of_curveborders_array, corresponding_indices_of_curveborder_pixels, curvecoords_name):
    ### Create a dictionary of the arguments and values because I want the name
    ### of the list used for the second argument:
    
    ### Lists of curvature points as (curve origin, point1, point2)
    curve_definitions = []
    
    ### Coordinates "x" are represented as (row, column)
    for i,x in enumerate(corresponding_indices_of_curveborder_pixels):
        temp_curveborders = np.copy(single_channel_of_curveborders_array)
        curvepathfinder = np.zeros(np.shape(temp_curveborders), np.dtype(np.uint32))
        curvepathdestroyer = np.zeros(np.shape(temp_curveborders), np.dtype(np.uint32))
        # "cut" open the curveborders:
        temp_curveborders[x] = 0

        
        # count curve length / 2 pixels away on either side of the cut site of
        # temp_curveborders:
        count = 0
        while count < (curve_length-1)/2:
            # for (row,col), val in np.ndenumerate(temp_curveborders):
            #     if val != 0:
            for row,col in zip(*np.where(temp_curveborders)):
                if np.count_nonzero(temp_curveborders[sub(row,1):row+2, sub(col,1):col+2]) == 2:
                    curvepathdestroyer[row,col] = temp_curveborders[row, col]
                    print("filenumber:%s"%(filenumber+1), '/', total_files,',', '%s:'%curvecoords_name+'%.2f %% -'%(100*float(i)/float(len(corresponding_indices_of_curveborder_pixels)-1)), x, row, col, temp_curveborders[row,col])
            temp_curveborders -= curvepathdestroyer
            curvepathfinder += curvepathdestroyer
            curvepathdestroyer = np.zeros(np.shape(temp_curveborders), np.dtype(np.uint32))
            count +=1
        
        curvepathfinder[x] = 1
        test = []
        # for (row,col), val in np.ndenumerate(curvepathfinder):
        for row,col in zip(*np.where(curvepathfinder)):
                # if val != 0:
                if np.count_nonzero(curvepathfinder[sub(row,1):row+2, sub(col,1):col+2]) == 2:
                    test.append((row,col))
                elif np.count_nonzero(curvepathfinder[sub(row,1):row+2, sub(col,1):col+2]) == 3:
                    pass
                else:
                    print("These borders are a problem child.")
        if len(test) == 0:
            curve_definitions.append( (x, (float('nan'), float('nan')), (float('nan'), float('nan'))) )
        else:
#                curve_definitions_red.append((x,test[0],test[1]))
            try:
                curve_definitions.append((x,test[0],test[1]))
            except IndexError:
                curve_definitions.append((x,(float('nan'), float('nan')), (float('nan'), float('nan'))))

    curve_vals = []
    for x in curve_definitions:
        ### Append the pixel location (row, col) and the circumcircle 
        ### radius, the circumcircle center x coordinate and y coordinate
        ### as a single list of 5 elements:
        curve_vals.append( [x[0][1]] + [x[0][0]] + list(curvature_from_3points(np.asarray(list(zip(*x))[1]), np.asarray(list(zip(*x))[0]))) + [curvature_sign(np.asarray(list(zip(*x))[1]), np.asarray(list(zip(*x))[0]), curve_definitions)] )
    return curve_definitions, curve_vals



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



def count_pixels_around_borders_sequentially(curveborders):

    diag_struct_elem = np.array([[0,1,0], [1,0,1], [0,1,0]])
    diag_struct_elem = np.reshape( diag_struct_elem, (3,3))#,1))
    
    cross_struct_elem = np.array([[1,0,1], [0,0,0], [1,0,1]])
    cross_struct_elem = np.reshape( cross_struct_elem, (3,3))#,1))
    
    perimeter_cell1 = 0
    perimeter_cell2 = 0
    perimeter_arr = np.copy(curveborders).astype(float)
    ### Remove 1 pixel from the borders so that the perimeter tracker has somewhere to start
    if 1 in perimeter_arr:
        first_indice = list(zip(*np.where(perimeter_arr == 1)))[0]
        first_indice_arr = perimeter_arr[sub(first_indice[0],1):first_indice[0]+2, sub(first_indice[1],1):first_indice[1]+2, first_indice[2]]
        perimeter_cell1 += len(np.where(first_indice_arr * cross_struct_elem)[0]==1) * 1.0 + len(np.where(first_indice_arr * diag_struct_elem)[0]==1) * math.sqrt(2)
        perimeter_arr[first_indice] = 0
        for (i,j,k), value in np.ndenumerate(perimeter_arr):
            if value == 1 and (np.count_nonzero(perimeter_arr[sub(i,1):i+2, sub(j,1):j+2, k]) == 2):
                second_indice = (i,j,k)
                break
        perimeter_arr[second_indice[0], second_indice[1], second_indice[2]] *= -1
        ordered_border_indices_red = [(first_indice)]
    else:
        ordered_border_indices_red = [float('nan')]
        
    if 2 in perimeter_arr:
        first_indice = list(zip(*np.where(perimeter_arr == 2)))[0]
        first_indice_arr = perimeter_arr[sub(first_indice[0],1):first_indice[0]+2, sub(first_indice[1],1):first_indice[1]+2, first_indice[2]]
        perimeter_cell2 += len(np.where(first_indice_arr * cross_struct_elem)[0]==2) * 1.0 + len(np.where(first_indice_arr * diag_struct_elem)[0]==2) * math.sqrt(2)
        perimeter_arr[first_indice] = 0
        for (i,j,k), value in np.ndenumerate(perimeter_arr):
            if value == 2 and (np.count_nonzero(perimeter_arr[sub(i,1):i+2, sub(j,1):j+2, k]) == 2):
                second_indice = (i,j,k)
                break
        perimeter_arr[second_indice[0], second_indice[1], second_indice[2]] *= -1
        ordered_border_indices_grn = [(first_indice)]
    else:
        ordered_border_indices_grn = [float('nan')]
        
    count = 0
    while (np.count_nonzero(perimeter_arr[:,:,0]) > 1 or np.count_nonzero(perimeter_arr[:,:,1]) > 1):
        for (i,j,k), value in np.ndenumerate(perimeter_arr):
            if value != 0 and (np.count_nonzero(perimeter_arr[sub(i,1):i+2, sub(j,1):j+2, k]) == 2):
                if value == -1.0:
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
                elif value == -2.0:
                    if np.count_nonzero(perimeter_arr[sub(i,1):i+2, sub(j,1):j+2, k] * cross_struct_elem) > 0:
                        perimeter_arr[i,j,k] = 0
                        perimeter_arr[sub(i,1):i+2, sub(j,1):j+2, k] *= -1.0 * cross_struct_elem.astype(float)
                        perimeter_cell2 += math.sqrt(2)
                        ordered_border_indices_grn.append((i,j,k))
                    elif np.count_nonzero(perimeter_arr[sub(i,1):i+2, sub(j,1):j+2, k] * diag_struct_elem) > 0:
                        perimeter_arr[i,j,k] = 0
                        perimeter_arr[sub(i,1):i+2, sub(j,1):j+2, k] *= -1.0 * diag_struct_elem.astype(float)
                        perimeter_cell2 += 1.0
                        ordered_border_indices_grn.append((i,j,k))
        count += 1
        print("filenumber:%s"%(filenumber+1), '/', total_files, ',', "count:%s"%count)
        if count > 400:
            print("I couldn't resolve the perimeter of the cell.")
            perimeter_cell1 = np.float('nan')
            perimeter_cell2 = np.float('nan')
            break

    if len(np.where(perimeter_arr == -1)[0]) == 1:
        ordered_border_indices_red.append(list(zip(*np.where(perimeter_arr == -1)))[0])
    if len(np.where(perimeter_arr == -2)[0]) == 1:
        ordered_border_indices_grn.append(list(zip(*np.where(perimeter_arr == -2)))[0])
        
        
    numbered_border_indices_red = [x for x in enumerate(ordered_border_indices_red)]
    numbered_border_indices_grn = [x for x in enumerate(ordered_border_indices_grn)]
    
    return ordered_border_indices_red, ordered_border_indices_grn, numbered_border_indices_red, numbered_border_indices_grn, perimeter_cell1, perimeter_cell2



def find_closest_pixels_indices_between_cell_borders(redborder, grnborder, ordered_border_indices_red, ordered_border_indices_grn, mean_closest_indice_loc_red_old, mean_closest_indice_loc_grn_old):
    redborder_xy = list(zip(*np.where(redborder)))
    grnborder_x, grnborder_y = np.where(grnborder)

    dif_x = []
    dif_y = []
    for i in redborder_xy:
        dif_x.append([(j - i[0])**2 for j in grnborder_x])
        dif_y.append([(j - i[1])**2 for j in grnborder_y])
    dist_red_grn = np.asarray(dif_x) + np.asarray(dif_y)
    closest_redborder_indices_loc = np.where(dist_red_grn == np.min(dist_red_grn))[0].astype(int)
    closest_grnborder_indices_loc = np.where(dist_red_grn == np.min(dist_red_grn))[1].astype(int)
    closest_redborder_indices_rc = np.asarray(redborder_xy)[closest_redborder_indices_loc]
    closest_grnborder_indices_rc = np.column_stack((np.asarray(grnborder_x)[closest_grnborder_indices_loc], np.asarray(grnborder_y)[closest_grnborder_indices_loc]))
    
    ###

    ordered_border_indices_red_arr = np.asarray(ordered_border_indices_red)
    ordered_border_indices_grn_arr = np.asarray(ordered_border_indices_grn)
    
    ordered_border_red_closest_index_first = np.where(np.all(ordered_border_indices_red_arr[:,0:2] == closest_redborder_indices_rc[0], axis=1))[0]
    ordered_border_grn_closest_index_first = np.where(np.all(ordered_border_indices_grn_arr[:,0:2] == closest_grnborder_indices_rc[0], axis=1))[0]
    
    ordered_border_red_closest_index_last = np.where(np.all(ordered_border_indices_red_arr[:,0:2] == closest_redborder_indices_rc[-1], axis=1))[0]
    ordered_border_grn_closest_index_last = np.where(np.all(ordered_border_indices_grn_arr[:,0:2] == closest_grnborder_indices_rc[-1], axis=1))[0]
    
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
                ### otherwise, the mean indexx will be overestimated by 1.
    if mean_closest_indice_loc_red % 1 == 0:
        mean_closest_indice_loc_red = np.array([mean_closest_indice_loc_red])
    elif mean_closest_indice_loc_red != 0:
        mean_closest_indice_loc_red = np.array([mean_closest_indice_loc_red - 0.5, mean_closest_indice_loc_red + 0.5])
    
    arr_grn = np.sort(np.array((0, ordered_border_grn_closest_index_first, ordered_border_grn_closest_index_last, len(ordered_border_indices_grn))))
    if np.diff(arr_grn)[1] < (np.diff(arr_grn)[0] + np.diff(arr_grn)[2]):
        mean_closest_indice_loc_grn = np.mean(arr_grn[1:3])
    elif np.diff(arr_grn)[1] > (np.diff(arr_grn)[0] + np.diff(arr_grn)[2]):
        difference_grn = (np.diff(arr_grn)[0]+np.diff(arr_grn)[2])
        mean_closest_indice_loc_grn = arr_grn[2] + difference_grn / 2
        if mean_closest_indice_loc_grn > len(ordered_border_indices_grn) - 1:
            mean_closest_indice_loc_grn -= ((len(ordered_border_indices_grn) - 1) - 1)
                ### The extra -1 above is because 0 is a viable index when you loop,
                ### otherwise, the mean indexx will be overestimated by 1.
    if mean_closest_indice_loc_grn % 1 == 0:
        mean_closest_indice_loc_grn = np.array([mean_closest_indice_loc_grn])
    elif mean_closest_indice_loc_grn != 0:
        mean_closest_indice_loc_grn = np.array([mean_closest_indice_loc_grn - 0.5, mean_closest_indice_loc_grn + 0.5])


    return closest_redborder_indices_rc, closest_grnborder_indices_rc, mean_closest_indice_loc_red, mean_closest_indice_loc_grn



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
    closest_bin_centre = int(mean_closest_indice_loc.astype(int)[-1])
    
    ### The following 2 if statements say what to do if the bin edges need to go
    ### outside of the available indices - ie. need to loop:
    if closest_bin_centre - closest_bin_size//2 >= 0:
        if closest_bin_centre + closest_bin_size - closest_bin_size//2 < np.count_nonzero( curveborder_array == True):
            closest_bin_indices = ordered_border_indices[(closest_bin_centre - closest_bin_size//2):(closest_bin_centre + closest_bin_size - closest_bin_size//2)]
            closest_bin_indices_locs = list(range((closest_bin_centre - closest_bin_size//2), (closest_bin_centre + closest_bin_size - closest_bin_size//2)))
        elif closest_bin_centre + closest_bin_size - closest_bin_size//2 >= np.count_nonzero( curveborder_array == True):
            closest_bin_indices = ordered_border_indices[(closest_bin_centre - closest_bin_size//2):(np.count_nonzero( curveborder_array == True))]
            closest_bin_indices += ordered_border_indices[0:(closest_bin_centre + closest_bin_size - closest_bin_size//2 - np.count_nonzero( curveborder_array == True))]
            closest_bin_indices_locs = list(range((closest_bin_centre - closest_bin_size//2), (np.count_nonzero( curveborder_array == True))))
            closest_bin_indices_locs += list(range(0, (closest_bin_centre + closest_bin_size - closest_bin_size//2 - np.count_nonzero( curveborder_array == True))))
    elif closest_bin_centre - closest_bin_size//2 < 0:
        if closest_bin_centre + closest_bin_size - closest_bin_size//2 < np.count_nonzero( curveborder_array == True):
            closest_bin_indices = ordered_border_indices[(np.count_nonzero(curveborder_array==True) + closest_bin_centre - closest_bin_size//2):(np.count_nonzero(curveborder_array==True))]
            closest_bin_indices += ordered_border_indices[0:(closest_bin_centre + closest_bin_size - closest_bin_size//2)]
            closest_bin_indices_locs = list(range((np.count_nonzero(curveborder_array==True) + closest_bin_centre - closest_bin_size//2), (np.count_nonzero(curveborder_array==True))))
            closest_bin_indices_locs += list(range(0, (closest_bin_centre + closest_bin_size - closest_bin_size//2)))
    curveborder_array[np.asarray(closest_bin_indices)[:,0], np.asarray(closest_bin_indices)[:,1]] = 0
    
    return curveborder_array, size_list_of_other_bins, closest_bin_size, closest_bin_indices



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
    
    

def make_open_border_indices_sequential(open_borders_arr):
    diag_struct_elem = np.array([[0,1,0], [1,0,1], [0,1,0]])
    diag_struct_elem = np.reshape( diag_struct_elem, (3,3))#,1))
    
    cross_struct_elem = np.array([[1,0,1], [0,0,0], [1,0,1]])
    cross_struct_elem = np.reshape( cross_struct_elem, (3,3))#,1))

    perimeter_arr = np.copy(open_borders_arr).astype(float)
    perimeter_cell1 = 0
    perimeter_cell2 = 0
    if 1 in perimeter_arr:
        for (i,j,k), value in np.ndenumerate(perimeter_arr):
            if value == 1 and (np.count_nonzero(perimeter_arr[sub(i,1):i+2, sub(j,1):j+2, k]) == 2):
                second_indice = (i,j,k)
                break
        perimeter_arr[second_indice[0], second_indice[1], second_indice[2]] *= -1
        reordered_border_indices_red = []
    if 2 in perimeter_arr:
        for (i,j,k), value in np.ndenumerate(perimeter_arr):
            if value == 2 and (np.count_nonzero(perimeter_arr[sub(i,1):i+2, sub(j,1):j+2, k]) == 2):
                second_indice = (i,j,k)
                break
        perimeter_arr[second_indice[0], second_indice[1], second_indice[2]] *= -1
        reordered_border_indices_grn = []
    
    count = 0
    while (np.count_nonzero(perimeter_arr[:,:,0]) > 1 or np.count_nonzero(perimeter_arr[:,:,1]) > 1):
        for (i,j,k), value in np.ndenumerate(perimeter_arr):
            if value != 0 and (np.count_nonzero(perimeter_arr[sub(i,1):i+2, sub(j,1):j+2, k]) == 2):
                if value == -1.0:
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
                elif value == -2.0:
                    if np.count_nonzero(perimeter_arr[sub(i,1):i+2, sub(j,1):j+2, k] * cross_struct_elem) > 0:
                        perimeter_arr[i,j,k] = 0
                        perimeter_arr[sub(i,1):i+2, sub(j,1):j+2, k] *= -1.0 * cross_struct_elem.astype(float)
                        perimeter_cell2 += math.sqrt(2)
                        reordered_border_indices_grn.append((i,j,k))
                    elif np.count_nonzero(perimeter_arr[sub(i,1):i+2, sub(j,1):j+2, k] * diag_struct_elem) > 0:
                        perimeter_arr[i,j,k] = 0
                        perimeter_arr[sub(i,1):i+2, sub(j,1):j+2, k] *= -1.0 * diag_struct_elem.astype(float)
                        perimeter_cell2 += 1.0
                        reordered_border_indices_grn.append((i,j,k))
        count += 1
        print("filenumber:%s"%(filenumber+1), '/', total_files, ',', "count:%s"%count)
        if count > 400:
            print("I couldn't resolve the perimeter of the cell.")
            perimeter_cell1 = np.float('nan')
            perimeter_cell2 = np.float('nan')
            break
    
    if len(np.where(perimeter_arr == -1)[0]) == 1:
        reordered_border_indices_red.append(list(zip(*np.where(perimeter_arr == -1)))[0])
    if len(np.where(perimeter_arr == -2)[0]) == 1:
        reordered_border_indices_grn.append(list(zip(*np.where(perimeter_arr == -2)))[0])
    return reordered_border_indices_red, reordered_border_indices_grn, perimeter_cell1, perimeter_cell2



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


### The following line is global because it's used later on in the script:
bin_curvature_details_headers = ['bin_#','bin_color', '1 * curvature_sign / R_3', '1 / R_3', 'curvature_sign', 'xc_3', 'yc_3', 'R_3', 'Ri_3', 'residu_3', 'size_of_bin', 'x_fit_arc_start', 'y_fit_arc_start', 'x_fit_arc_stop', 'y_fit_arc_stop', 'arc_angle (in radians)', 'fitted curvature arclength']
def fit_curves_to_bins(bin_indices, one_channel_of_curveborders, odr_circle_fit_iterations):
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



def find_contact_angles(contactcorners, contactsurface, noncontactborders):
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
                if ((np.count_nonzero(noncontactborders[sub(i,1):i+2, sub(j,1):j+2, sub(k,1):k+2]) == 2)):
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
            if ((np.count_nonzero(raw_vectors_1_2[sub(i,1):i+2, sub(j,1):j+2, sub(k,1):k+2]) == 2)):
                vectors_1_2_endpoints.append([[i,j,k], value])
                
### Define vector 1 and 2's startpoints (ie. the corresponding contactcorners):
    for (i,j,k), value in np.ndenumerate(raw_vectors_1_2_startpoints):
        if value != 0:
            vectors_1_2_startpoints.append([[i,j,k], value])
    
### Do the same for vectors 3 and 4:            
    for (i,j,k), value in np.ndenumerate(raw_vectors_3_4):
        if value != 0:
            if ((np.count_nonzero(raw_vectors_3_4[sub(i,1):i+2, sub(j,1):j+2, sub(k,1):k+2]) == 2)):
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
            if value_end == value_start and value_end == 1:
                vector1 = np.asarray([x,y]) - np.asarray([a,b])
#                    vector1[1] *= -1
            elif value_end == value_start and value_end == 2:
                vector2 = np.asarray([x,y]) - np.asarray([a,b])
#                    vector2[1] *= -1
    
### ... and also vectors 3 and 4:
    for (y,x,z), value_end in vectors_3_4_endpoints:
        for (b,a,c), value_start in vectors_3_4_startpoints:    
            if value_end == value_start and value_end == 1:
                vector3 = np.asarray([x,y]) - np.asarray([a,b])
#                    vector3[1] *= -1
            elif value_end == value_start and value_end == 2:
                vector4 = np.asarray([x,y]) - np.asarray([a,b])
#                    vector4[1] *= -1
    
### Determine the x,y coordinates of the vector start locations for later usage
### (see section below on plotting angles in a figure or search up ax.quiver):
    for (b,a,c), value_start in vectors_1_2_startpoints:
        if value_start == 1:
            vector1_start = np.asarray([a,b])
    for (b,a,c), value_start in vectors_1_2_startpoints:
        if value_start == 2:
            vector2_start = np.asarray([a,b])
    for (b,a,c), value_start in vectors_3_4_startpoints:
        if value_start == 1:
            vector3_start = np.asarray([a,b])
    for (b,a,c), value_start in vectors_3_4_startpoints:
        if value_start == 2:
            vector4_start = np.asarray([a,b])
    
### Determine the angle by using the cross-product of the two vectors:
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



def find_contactsurface_len(contactsurface):
    ### Count the length of the contact surface in pixel lengths
    diag_struct_elem = np.array([[0,1,0], [1,0,1], [0,1,0]])
    diag_struct_elem = np.reshape( diag_struct_elem, (3,3))#,1))

    cross_struct_elem = np.array([[1,0,1], [0,0,0], [1,0,1]])
    cross_struct_elem = np.reshape( cross_struct_elem, (3,3))#,1))
    
    contactsurface_len1 = 0
    contactsurface_len2 = 0
    contactlength_arr = np.copy(contactsurface)        
    while (len(np.where(contactlength_arr == 1)[0]) > 1 or len(np.where(contactlength_arr == 2)[0]) > 1):
        for (i,j,k), value in np.ndenumerate(contactlength_arr):
            if value == 1:
                if ((len(np.where(contactlength_arr[sub(i,1):i+2, sub(j,1):j+2, k] == value)[0]) == 2)):
                    if len(np.where(contactlength_arr[sub(i,1):i+2, sub(j,1):j+2, k] * cross_struct_elem)[0]) > 0:
                        contactlength_arr[i,j,k] = 0
                        contactsurface_len1 += math.sqrt(2)
                    elif len(np.where(contactlength_arr[sub(i,1):i+2, sub(j,1):j+2, k] * diag_struct_elem)[0]) > 0:
                        contactlength_arr[i,j,k] = 0
                        contactsurface_len1 += 1.0   
            elif value == 2:
                if ((len(np.where(contactlength_arr[sub(i,1):i+2, sub(j,1):j+2, k] == value)[0]) == 2)):
                    if len(np.where(contactlength_arr[sub(i,1):i+2, sub(j,1):j+2, k] * cross_struct_elem)[0]) > 0:
                        contactlength_arr[i,j,k] = 0
                        contactsurface_len2 += math.sqrt(2)
                    elif len(np.where(contactlength_arr[sub(i,1):i+2, sub(j,1):j+2, k] * diag_struct_elem)[0]) > 0:
                        contactlength_arr[i,j,k] = 0
                        contactsurface_len2 += 1.0

    contactsurface_len = (contactsurface_len1 + contactsurface_len2)/2.0       

    return contactsurface_len, contactsurface_len1, contactsurface_len2



def perform_linescan(contactsurface, perpendicular2contact_vector, linescan_length):
    ### funcdiff will find find the difference between 2 functions at a given x:
    def funcdiff(x):
        return redpolyfunc(x) - grnpolyfunc(x)

    ### Extend the vector that is perpendicular to the contact line by the user-
    ### specified length "linescan_length":        
    linescan = linescan_length / np.sqrt(np.dot(perpendicular2contact_vector, perpendicular2contact_vector)) * perpendicular2contact_vector

    ### Convert the linescan indexing in to image format (row,col) (ie. (y,x)):
    imgform_linescan = np.zeros((2))
    imgform_linescan[0] = linescan[1]
    imgform_linescan[1] = linescan[0]

    ### Create a list of contactsurface pixel x coords and y coords:
    red_contact_indicesyx = []
    grn_contact_indicesyx = []
    red_linescan_values = []
    grn_linescan_values = []
    intersection_values_per_cell = []

    ### Store which cellpair_num was analyzed:
    intersection_cellpair_num = (filenumber+1)
    
    
    ### Return all the indices of pixels of each cell border that is considered
    ### to be in contact with the other cell:
    for i,v in np.ndenumerate(contactsurface):
        if v == 1:
            red_contact_indicesyx.append(i)
        elif v == 2:
            grn_contact_indicesyx.append(i)
    for i,v in enumerate(red_contact_indicesyx):
        red_contact_indicesyx[i] = list(v)
    for i,v in enumerate(grn_contact_indicesyx):
        grn_contact_indicesyx[i] = list(v)

    # This next part does a linescan of the desired length (linescan_length) on
    # each pixel in the red contactsurface:
    ### For each pixel in the red cells contactsurface...
    for i,v in enumerate(red_contact_indicesyx):        

        ### Define linescan start and end locations based on the desired length
        ### of the linescan...
        linescan_start = v[:2] - (imgform_linescan/2.0)
        linescan_end = v[:2] + (imgform_linescan/2.0)
        ### And create a list of equally spaced point between the start and end:
        x,y = np.linspace(linescan_start[1], linescan_end[1], num=linescan_length, endpoint=False), np.linspace(linescan_start[0], linescan_end[0], num=linescan_length, endpoint=False)
        ### And then rasterize those points so that they are equivalent to indices
        ### in the image:
        for j,k in enumerate(x):
            x[j] = round(k)
        for j,k in enumerate(y):
            y[j] = round(k)
        ### Then use that rasterized line to do a linescan on the green and red
        ### channels for that pixel...
        red_linescan = justcellpair[y.astype(np.int),x.astype(np.int),0]
        grn_linescan = justcellpair[y.astype(np.int),x.astype(np.int),1]
        
        ### And also append this same information to a list containing all the
        ### other linescans for the other pixels in this particular cellpair
        ### at this particular time:
        red_linescan_values.append(justcellpair[y.astype(np.int),x.astype(np.int),0])
        grn_linescan_values.append(justcellpair[y.astype(np.int),x.astype(np.int),1])
        
        ### Then turn the linescans into a polynomial piecewise function by
        ### performing interpolation on the red_linescan and grn_linescan:
        redpolyfunc = interpolate.interp1d(np.arange(linescan_length), red_linescan, fill_value="extrapolate")
        grnpolyfunc = interpolate.interp1d(np.arange(linescan_length), grn_linescan, fill_value="extrapolate")
        # The 2 linescans can be plotted as follows:
        #plt.plot(red_linescan, c='r')
        #plt.plot(grn_linescan, c='g')


        # Now it's time to find where the red and green linescans intersect:
        ### Create a list of the x values of your red and green linescan.
        ### This ends up just being each number from 0 to your linescan length,
        ### twice (one for red linescan, and one for green).
        xs = np.r_[range(linescan_length), range(linescan_length)]
        ### Sort these numbers from lowest to highest:
        xs.sort()
        ### Find the lower and upper bounds of your your data along the x axis:
        x_min = xs.min()
        x_max = xs.max()
        ### Take the midpoint between each pixel (ie. each point in 'xs'):
        x_mid = xs[:-1]+np.diff(xs)/2.0
        ### Make an empty set to store the zeros you find (funcdiff will take
        ### the difference between 2 functions at a given x, in this case your
        ### redpolyfunc and grnpolyfunc, which if there is an intersection will
        ### be 0):
        intersects = set()
        ### Find the x values for which the difference between redpolyfunc and 
        ### grnpolyfunc is 0 (this is where their y values are equal and they 
        ### therefore intersect):
        for val in x_mid:
            intersect,infodict,ier,mesg = optimization.fsolve(funcdiff,val,full_output=True)
            # ier==1 indicates a root has been found
            if ier == 1 and x_min < intersect < x_max:
                intersects.add(intersect[0])
            else:
                intersects.add(float('nan'))
        ### Convert your intersection points in to a list:
        intersectsx = list(intersects)
        ### Use the x intersects to find the fluorescence values of the 
        ### redpolyfunc and grnpolyfunc at the point where they intersect:
        intersectsy1 = redpolyfunc(intersectsx)
        intersectsy2 = grnpolyfunc(intersectsx)
        ### Then average the 2 fluorescence intensities together as they might
        ### differ slightly:
        intersectsy = (intersectsy1 + intersectsy2) / 2.0
        
        ### Occasionally more than 1  intersection may be found. This may be 
        ### because of floating point errors. If that happens, take the mean:
        if len(np.where(~np.isnan(intersectsx))[0]) == 1:
            intersectsx = np.nanmean(intersectsx)
        elif len(np.where(~np.isnan(intersectsx))[0]) > 1:
            if np.isclose(np.nanmax(intersectsx), np.nanmin(intersectsx), rtol = 1e-4) == True:
                intersectsx = np.nanmean(intersectsx)
            else:
                intersectsx = np.nanmean(intersectsx)
                
        if len(np.where(~np.isnan(intersectsy))[0]) == 1:
            intersectsy = np.nanmean(intersectsy)
        elif len(np.where(~np.isnan(intersectsy))[0]) > 1:
            if np.isclose(np.nanmax(intersectsy), np.nanmin(intersectsy), rtol = 1e-4) == True:
                intersectsy = np.nanmean(intersectsy)
            else:
                intersectsy = np.nanmean(intersectsy)
        elif np.isnan(intersectsy).all():
            intersectsy = np.mean(intersectsy)

        intersection_values_per_cell.append(intersectsy)
        intersection_values_per_cell_middle_list_index = int(np.count_nonzero(intersection_values_per_cell) / 2)


        
###############################################################################
    
    ### Create an array containing the many linescans from the red channel all 
    ### aligned:
    red_gap_scan = np.reshape(red_linescan_values, (len(red_contact_indicesyx),linescan_length))
    red_gap_scan = red_gap_scan.T                

    ### Create an array containing the many linescans from the green channel all 
    ### aligned:
    grn_gap_scan = np.reshape(grn_linescan_values, (len(red_contact_indicesyx),linescan_length))
    grn_gap_scan = grn_gap_scan.T        
    
    ### Take the average intensity at each position along the many linescans 
    ### for each channel:
    red_gap_scanmean = np.mean(red_gap_scan, 1)
    grn_gap_scanmean = np.mean(grn_gap_scan, 1)
    
    ### Make a list containing the position along the linescan of the fluorescence
    ### value (gap_scanx), and the fluorescence value at that point (gap_scany):
    red_gap_scanx = []
    red_gap_scany = []
    for (i,j), l in np.ndenumerate(red_gap_scan):
        red_gap_scanx.append(i)
        red_gap_scany.append(l)
    
    grn_gap_scanx = []
    grn_gap_scany = []
    for (i,j), l in np.ndenumerate(grn_gap_scan):
        grn_gap_scanx.append(i)
        grn_gap_scany.append(l)

    
    ### Originally I tried to fit a Gaussian function to the data but seeing as
    ### the Gaussian appears to be a poor fit, find the intersection of the 2
    ### intensity curves assuming that they are described well enough by a 
    ### piecewise polynomial:
    redpolyfunc = interpolate.interp1d(np.arange(linescan_length), red_gap_scanmean, fill_value="extrapolate")
    grnpolyfunc = interpolate.interp1d(np.arange(linescan_length), grn_gap_scanmean, fill_value="extrapolate")
    ### Create a list of the x values of your red and green linescan.
    ### This ends up just being each number from 0 to your linescan length,
    ### twice (one for red linescan, and one for green).
    xs = np.r_[range(linescan_length), range(linescan_length)]
    ### Sort these numbers from lowest to highest.        
    xs.sort()
    ### Find the lower and upper bounds of your your data along the x axis:
    x_min = xs.min()
    x_max = xs.max()
    ### Take the midpoint between each pixel (ie. each point in 'xs')
    x_mid = xs[:-1]+np.diff(xs)/2.0  
    ### Make an empty set to store the zeros you find (funcdiff will take
    ### the difference between 2 functions at a given x, in this case your
    ### redpolyfunc and grnpolyfunc, which if there is an intersection will
    ### be 0):
    #test2 = []
    intersects = set()
    ### Find the x values for which the difference between redpolyfunc and 
    ### grnpolyfunc is 0 (this is where their y values are equal and they 
    ### therefore intersect):
    for val in x_mid:
        intersect,infodict,ier,mesg = optimization.fsolve(funcdiff,val,full_output=True)
       # ier==1 indicates a root has been found
        if ier == 1 and x_min < intersect < x_max:
            intersects.add(intersect[0])
        else:
            intersects.add(float('nan'))
    ### Convert your intersection points in to a list:        
    intersectsx = list(intersects)
    #test2.append(intersectsx)
    intersectsy1 = redpolyfunc(intersectsx)
    intersectsy2 = grnpolyfunc(intersectsx)
    intersectsy = (intersectsy1 + intersectsy2) / 2.0
    if len(np.where(~np.isnan(intersectsx))[0]) == 1:
        intersectsx = np.nanmean(intersectsx)
    elif len(np.where(~np.isnan(intersectsx))[0]) > 1:
        if np.isclose(np.nanmax(intersectsx), np.nanmin(intersectsx), rtol = 1e-4) == True:
            print("Found %s x intersects" %len(intersectsx))
            print("They differ by less than 1e-4, so took the average.")
            intersectsx = np.nanmean(intersectsx)
        else:
            print("Found %s x intersects" %len(intersectsx))
            print("They differ by MORE than 1e-4, took the average anyways.")
            print("They are: %s" %intersectsx)
            intersectsx = np.nanmean(intersectsx)
    if len(np.where(~np.isnan(intersectsy))[0]) == 1:
        intersectsy = np.nanmean(intersectsy)
    elif len(np.where(~np.isnan(intersectsy))[0]) > 1:
        if np.isclose(np.nanmax(intersectsy), np.nanmin(intersectsy), rtol = 1e-4) == True:
            print("Found %s y intersects" %len(intersectsy))
            print("They differ by less than 1e-4, so took the average.")
            intersectsy = np.nanmean(intersectsy)
            print("Y intersect: %s"%intersectsy)

        else:
            print("Found %s y intersects" %len(intersectsy))
            print("They differ by MORE than 1e-4, took the average anyways.")
            print("They are: %s" %intersectsy)
            intersectsy = np.nanmean(intersectsy)
            
    elif np.isnan(intersectsy).all():
        intersectsy = np.mean(intersectsy)
    
    return  intersectsx, intersectsy, intersection_values_per_cell, intersection_values_per_cell_middle_list_index, intersection_cellpair_num, red_gap_scanx, red_gap_scany, grn_gap_scanx, grn_gap_scany



### Since I have so many lists I'm going to define a function to update them
### all for the sake of simplicity:
def listupdate():
    master_red_linescan_mean.append(red_gap_scanmean)
    master_grn_linescan_mean.append(grn_gap_scanmean)     
    master_red_linescanx.append(red_gap_scanx)
    master_red_linescany.append(red_gap_scany)
    master_grn_linescanx.append(grn_gap_scanx)
    master_grn_linescany.append(grn_gap_scany)
    time_list.append('')
    cell_pair_num_list.append((filenumber+1))
    angle_1_list.append((degree_angle1))
    angle_2_list.append((degree_angle2))
    average_angle_list.append((average_angle))
    contactsurface_len_red_list.append(contactsurface_len1)
    contactsurface_len_grn_list.append(contactsurface_len2)
    contactsurface_list.append(contactsurface_len)
    contact_corner1_list.append((vector1_start,vector3_start))
    contact_corner2_list.append((vector2_start,vector4_start))
    corner_corner_length1_list.append(((vector1_start - vector3_start)[0]**2 + (vector1_start - vector3_start)[1]**2)**0.5)
    corner_corner_length2_list.append(((vector2_start - vector4_start)[0]**2 + (vector2_start - vector4_start)[1]**2)**0.5)
    noncontactsurface1_list.append((noncontact_perimeter_cell1))
    noncontactsurface2_list.append((noncontact_perimeter_cell2))
    filename_list.append((os.path.basename(filename)))
    object_of_interest_brightness_cutoff_percent_list.append(object_of_interest_brightness_cutoff_percent)
    otsu_fraction_list.append(otsu_fraction)
    red_gauss_list.append(gaussian_filter_value_red)
    grn_gauss_list.append(gaussian_filter_value_grn)
    Nth_pixel_list.append(Nth_pixel)
    disk_radius_list.append(disk_radius)
    background_threshold_list.append(background_threshold)
    object_of_interest_brightness_cutoff_red_list.append(object_of_interest_brightness_cutoff_red)
    object_of_interest_brightness_cutoff_grn_list.append(object_of_interest_brightness_cutoff_grn)
    area_cell_1_list.append(area_cell_1)
    area_cell_2_list.append(area_cell_2)
    cell_pair_area_list.append((area_cell_1 + area_cell_2))
    cell1_perimeter_list.append(perimeter_cell1)
    cell2_perimeter_list.append(perimeter_cell2)
    struct_elem_list.append(Bc3D.shape)
    linescan_length_list.append(linescan_length)
    curve_length_list.append(curve_length)
    intersect_pos.append(intersectsx)
    intens_at_intersect.append(intersectsy)
    otsu_list.append(otsu_thresholding)
    master_intersection_points.append(intersection_points)
    master_intersection_values_list.append(intersection_values_per_cell)
    master_intersection_values_per_cell_middle_list_index.append(intersection_values_per_cell_middle_list_index)
    master_intersection_cellpair_num.append(intersection_cellpair_num)
    
    redcellborders_coords.append(list(zip(*np.where(cleanborders==1))))
    redcontactsurface_coords.append(list(zip(*np.where(contactsurface==1))))
    rednoncontact_coords.append(list(zip(*np.where((noncontactborders+pathfinder)==1))))
    
    grncellborders_coords.append(list(zip(*np.where(cleanborders==2))))
    grncontactsurface_coords.append(list(zip(*np.where(contactsurface==2))))
    grnnoncontact_coords.append(list(zip(*np.where((noncontactborders+pathfinder)==2))))
    
    redcontactvector1_start.append(list(vector1_start))
    grncontactvector1_start.append(list(vector2_start))
    redcontactvector2_start.append(list(vector3_start))
    grncontactvector2_start.append(list(vector4_start))

    redcontactvector1.append(list(vector1))
    grncontactvector1.append(list(vector2))
    redcontactvector2.append(list(vector3))
    grncontactvector2.append(list(vector4))
    try:
#        red_maxima_coords.append((np.where(newcellmaxima==1)[1], np.where(newcellmaxima==1)[0]))
        red_maxima_coords.append( np.where(newcellmaxima[:,:,0]) )
#    newcellmaxima_rcd
    except IndexError:
        red_maxima_coords.append( (np.array(float('nan')), np.array(float('nan'))) )
    try:
#        grn_maxima_coords.append((np.where(newcellmaxima==2)[1], np.where(newcellmaxima==2)[0]))
        grn_maxima_coords.append( np.where(newcellmaxima[:,:,1]) )
#    newcellmaxima_rcd
    except IndexError:
        grn_maxima_coords.append( (np.array(float('nan')), np.array(float('nan'))) )
    
    contact_vectors_list.append(list(contact_vector))
    perpendicular2contact_vectors_list.append(list(perpendicular2contact_vector))
    
    red_halfangle1_list.append(red_halfangle1)
    red_halfangle2_list.append(red_halfangle2)
    grn_halfangle1_list.append(grn_halfangle1)
    grn_halfangle2_list.append(grn_halfangle2)
    curvature_coords_red_list.append(list(zip(*np.where(curveborders_curvature_labeled[:,:,0]))))
    curvature_coords_grn_list.append(list(zip(*np.where(curveborders_curvature_labeled[:,:,1]))))
    numbered_border_indices_red_list.append(numbered_border_indices_red)
    numbered_border_indices_grn_list.append(numbered_border_indices_grn)
    numbered_noncontactborder_indices_red_list.append(numbered_noncontactborder_indices_red)
    numbered_noncontactborder_indices_grn_list.append(numbered_noncontactborder_indices_grn)

    
    mask_list.append(skip=='y')
    masked_curvatures1_protoarray.append(curveborders[:,:,0]==0)
    masked_curvatures2_protoarray.append(curveborders[:,:,1]==0)
    curvatures1_protoarray.append(curveborders_curvature_labeled[:,:,0])
    curvatures2_protoarray.append(curveborders_curvature_labeled[:,:,1])
    
    curveborders1_protoarray.append(curveborders[:,:,0])
    curveborders2_protoarray.append(curveborders[:,:,1])
    noncontactborders1_protoarray.append(noncontactborders[:,:,0])
    noncontactborders2_protoarray.append(noncontactborders[:,:,1])
    contactsurface1_protoarray.append(contactsurface[:,:,0])
    contactsurface2_protoarray.append(contactsurface[:,:,1])
    pathfinder1_protoarray.append(pathfinder[:,:,0])
    pathfinder2_protoarray.append(pathfinder[:,:,1])
    
    angles_angle_labeled_red_protoarray.append(angles_angle_labeled[:,:,0])
    angles_angle_labeled_grn_protoarray.append(angles_angle_labeled[:,:,1])
    
    curvature_values_red_list.append(list(np.round(curveborders_curvature_labeled[np.where(curveborders_curvature_labeled[:,:,0])[0],np.where(curveborders_curvature_labeled[:,:,0])[1],0], decimals=10)))
    mean_curvature_curveborders_list_red.append(mean_curvature_curveborders_red)
    median_curvature_curveborders_list_red.append(median_curvature_curveborders_red)
    sem_curvature_curveborders_list_red.append(sem_curvature_curveborders_red)
    std_curvature_curveborders_list_red.append(std_curvature_curveborders_red)

    all_curvatures_noncontactborders_list_red.append(list(all_curvatures_noncontactborders_red))
    mean_curvature_noncontactborders_list_red.append(mean_curvature_noncontactborders_red)
    median_curvature_noncontactborders_list_red.append(median_curvature_noncontactborders_red)
    sem_curvature_noncontactborders_list_red.append(sem_curvature_noncontactborders_red)
    std_curvature_noncontactborders_list_red.append(std_curvature_noncontactborders_red)

    all_curvatures_pathfinder_list_red.append(list(curvature_pathfinder_red))
    mean_curvature_pathfinder_list_red.append(mean_curvature_pathfinder_red)
    median_curvature_pathfinder_list_red.append(median_curvature_pathfinder_red)
    sem_curvature_pathfinder_list_red.append(sem_curvature_pathfinder_red)
    std_curvature_pathfinder_list_red.append(std_curvature_pathfinder_red)

    all_curvatures_contactsurface_list_red.append(list(curvature_contactsurface_red))
    mean_curvature_contactsurface_list_red.append(mean_curvature_contactsurface_red)
    median_curvature_contactsurface_list_red.append(median_curvature_contactsurface_red)
    sem_curvature_contactsurface_list_red.append(sem_curvature_contactsurface_red)
    std_curvature_contactsurface_list_red.append(std_curvature_contactsurface_red)
    

    curvature_values_grn_list.append(list(np.round(curveborders_curvature_labeled[np.where(curveborders_curvature_labeled[:,:,1])[0],np.where(curveborders_curvature_labeled[:,:,1])[1],1], decimals=10)))
    mean_curvature_curveborders_list_grn.append(mean_curvature_curveborders_grn)
    median_curvature_curveborders_list_grn.append(median_curvature_curveborders_grn)
    sem_curvature_curveborders_list_grn.append(sem_curvature_curveborders_grn)
    std_curvature_curveborders_list_grn.append(std_curvature_curveborders_grn)

    all_curvatures_noncontactborders_list_grn.append(list(all_curvatures_noncontactborders_grn))
    mean_curvature_noncontactborders_list_grn.append(mean_curvature_noncontactborders_grn)
    median_curvature_noncontactborders_list_grn.append(median_curvature_noncontactborders_grn)
    sem_curvature_noncontactborders_list_grn.append(sem_curvature_noncontactborders_grn)
    std_curvature_noncontactborders_list_grn.append(std_curvature_noncontactborders_grn)

    all_curvatures_pathfinder_list_grn.append(list(curvature_pathfinder_grn))
    mean_curvature_pathfinder_list_grn.append(mean_curvature_pathfinder_grn)
    median_curvature_pathfinder_list_grn.append(median_curvature_pathfinder_grn)
    sem_curvature_pathfinder_list_grn.append(sem_curvature_pathfinder_grn)
    std_curvature_pathfinder_list_grn.append(std_curvature_pathfinder_grn)

    all_curvatures_contactsurface_list_grn.append(list(curvature_contactsurface_grn))
    mean_curvature_contactsurface_list_grn.append(mean_curvature_contactsurface_grn)
    median_curvature_contactsurface_list_grn.append(median_curvature_contactsurface_grn)
    sem_curvature_contactsurface_list_grn.append(sem_curvature_contactsurface_grn)
    std_curvature_contactsurface_list_grn.append(std_curvature_contactsurface_grn)

    all_curvatures_pathfinder_list.append(all_curvatures_pathfinder)
    mean_curvature_pathfinder_list.append(mean_curvature_pathfinder)
    median_curvature_pathfinder_list.append(median_curvature_pathfinder)
    sem_curvature_pathfinder_list.append(sem_curvature_pathfinder)
    std_curvature_pathfinder_list.append(std_curvature_pathfinder)
    
    all_curvatures_pathfinder_list_angle1.append(curvature_pathfinder_angle1)
    mean_curvature_pathfinder_list_angle1.append(mean_curvature_pathfinder_angle1)
    median_curvature_pathfinder_list_angle1.append(median_curvature_pathfinder_angle1)
    sem_curvature_pathfinder_list_angle1.append(sem_curvature_pathfinder_angle1)
    std_curvature_pathfinder_list_angle1.append(std_curvature_pathfinder_angle1)
    
    all_curvatures_pathfinder_list_angle2.append(curvature_pathfinder_angle2)
    mean_curvature_pathfinder_list_angle2.append(mean_curvature_pathfinder_angle2)
    median_curvature_pathfinder_list_angle2.append(median_curvature_pathfinder_angle2)
    sem_curvature_pathfinder_list_angle2.append(sem_curvature_pathfinder_angle2)
    std_curvature_pathfinder_list_angle2.append(std_curvature_pathfinder_angle2)

    all_curvatures_pathfinder_list_grn_angle1.append(all_curvatures_pathfinder_grn_angle1)
    mean_curvature_pathfinder_list_grn_angle1.append(mean_curvature_pathfinder_grn_angle1)
    median_curvature_pathfinder_list_grn_angle1.append(median_curvature_pathfinder_grn_angle1)
    sem_curvature_pathfinder_list_grn_angle1.append(sem_curvature_pathfinder_grn_angle1)
    std_curvature_pathfinder_list_grn_angle1.append(std_curvature_pathfinder_grn_angle1)
    
    all_curvatures_pathfinder_list_grn_angle2.append(all_curvatures_pathfinder_grn_angle2)
    mean_curvature_pathfinder_list_grn_angle2.append(mean_curvature_pathfinder_grn_angle2)
    median_curvature_pathfinder_list_grn_angle2.append(median_curvature_pathfinder_grn_angle2)
    sem_curvature_pathfinder_list_grn_angle2.append(sem_curvature_pathfinder_grn_angle2)
    std_curvature_pathfinder_list_grn_angle2.append(std_curvature_pathfinder_grn_angle2)
    
    all_curvatures_pathfinder_list_red_angle1.append(all_curvatures_pathfinder_red_angle1)
    mean_curvature_pathfinder_list_red_angle1.append(mean_curvature_pathfinder_red_angle1)
    median_curvature_pathfinder_list_red_angle1.append(median_curvature_pathfinder_red_angle1)
    sem_curvature_pathfinder_list_red_angle1.append(sem_curvature_pathfinder_red_angle1)
    std_curvature_pathfinder_list_red_angle1.append(std_curvature_pathfinder_red_angle1)
    
    all_curvatures_pathfinder_list_red_angle2.append(all_curvatures_pathfinder_red_angle2)
    mean_curvature_pathfinder_list_red_angle2.append(mean_curvature_pathfinder_red_angle2)
    median_curvature_pathfinder_list_red_angle2.append(median_curvature_pathfinder_red_angle2)
    sem_curvature_pathfinder_list_red_angle2.append(sem_curvature_pathfinder_red_angle2)
    std_curvature_pathfinder_list_red_angle2.append(std_curvature_pathfinder_red_angle2)
    
    bin_curvature_details_red_list.append(bin_curvature_details_red)
    bin_curvature_details_grn_list.append(bin_curvature_details_grn)
    
    whole_cell_curvature_details_red_list.append(whole_curvature_details_red)
    whole_cell_curvature_details_grn_list.append(whole_curvature_details_grn)

    bin_indices_red_list.append(bin_indices_red)
    bin_indices_grn_list.append(bin_indices_grn)
    
    bin_start_red_list.append(bin_start_red_indices)
    bin_stop_red_list.append(bin_stop_red_indices)
    
    bin_start_grn_list.append(bin_start_grn_indices)
    bin_stop_grn_list.append(bin_stop_grn_indices)
    
    cells_touching_list.append(cells_touching)
    
    script_version_list.append(script_version)
    
    gc.collect()
    end_time = time.perf_counter()
    print("This image took %.3f seconds to process." %(end_time - start_time))
    processing_time_list.append(end_time - start_time)

    return






# plt.ion()

diag_struct_elem = np.array([[0,1,0], [1,0,1], [0,1,0]])
diag_struct_elem = np.reshape( diag_struct_elem, (3,3,1))

cross_struct_elem = np.array([[1,0,1], [0,0,0], [1,0,1]])
cross_struct_elem = np.reshape( cross_struct_elem, (3,3,1))

### Assign the path to your folder containing your cropped cell pair images.
while True:
    try:
        path = input('Please select the path to your cropped cell pair images.\n>')
        # path = r"C:\Users\Serge\Desktop\problematic images\broken"
        os.chdir(path)
#        ext = raw_input('What types of files will you be working with?\n(eg. ".jpg", ".tif")\n>')
        ext = ".tif"
        if len(glob.glob(path+'/*'+ext)) == 0:
#            print 'This folder contains no files of the specified file type!'
            print('This folder contains no .tif files!')
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
filename_list = []
red_gauss_list = []
grn_gauss_list = []
#gauss_list = []
red_thresh_list = []
grn_thresh_list = []
struct_elem_list = []
Nth_pixel_list = []
disk_radius_list = [] # DONE #Replace sobel_list with this
background_threshold_list = [] # DONE #Replace gauss_list with this
otsu_fraction_list = []
object_of_interest_brightness_cutoff_percent_list = []
object_of_interest_brightness_cutoff_red_list = [] # DONE #Replace red_otsu_fraction_list with this
object_of_interest_brightness_cutoff_grn_list = [] # DONE #Replace grn_otsu_fraction_list
area_cell_1_list = []
area_cell_2_list = []
cell_pair_area_list = []
cell1_perimeter_list = []
cell2_perimeter_list = []
master_red_linescan_mean = []
master_grn_linescan_mean = []
master_red_linescanx = []
master_red_linescany = []     
master_grn_linescanx = []
master_grn_linescany = []
#peak_peak_dist = []
#fwhm_fwhm_dist = []
#intersect_position_on_linescan = []
intersect_pos = []
intens_at_intersect = []
#intens_at_gauss_intersect = []
linescan_length_list = []
curve_length_list = []
#table_data = []
#sobel_list = []
otsu_list = []
master_intersection_values_list = []
master_intersection_values_per_cell_middle_list_index = []
master_intersection_cellpair_num = []
master_intersection_points = []
processing_time_list = []
mask_list = []

redcellborders_coords = []
redcontactsurface_coords = []
rednoncontact_coords = []

grncellborders_coords = []
grncontactsurface_coords = []
grnnoncontact_coords = []

redcontactvector1_start = []
grncontactvector1_start = []
redcontactvector2_start = []
grncontactvector2_start = []

redcontactvector1 = []
grncontactvector1 = []
redcontactvector2 = []
grncontactvector2 = []

red_maxima_coords = []
grn_maxima_coords = []

contact_vectors_list = []
perpendicular2contact_vectors_list = []

red_halfangle1_list = []
red_halfangle2_list = []
grn_halfangle1_list = []
grn_halfangle2_list = []

curvature_coords_red_list = []
curvature_coords_grn_list = []

numbered_border_indices_red_list = []
numbered_border_indices_grn_list = []
numbered_noncontactborder_indices_red_list = []
numbered_noncontactborder_indices_grn_list = []

contactsurface_len_red_list = []
contactsurface_len_grn_list = []

curvatures1_protoarray = []
curvatures2_protoarray = []

masked_curvatures1_protoarray = []
masked_curvatures2_protoarray = []

curveborders1_protoarray = []
curveborders2_protoarray = []
noncontactborders1_protoarray = []
noncontactborders2_protoarray = []
contactsurface1_protoarray = []
contactsurface2_protoarray = []
pathfinder1_protoarray = []
pathfinder2_protoarray = []
angles_angle_labeled_red_protoarray = []
angles_angle_labeled_grn_protoarray = []

curvature_values_red_list = []
mean_curvature_curveborders_list_red = []
median_curvature_curveborders_list_red = []
sem_curvature_curveborders_list_red = []
std_curvature_curveborders_list_red = []

all_curvatures_noncontactborders_list_red = []
mean_curvature_noncontactborders_list_red = []
median_curvature_noncontactborders_list_red = []
sem_curvature_noncontactborders_list_red = []
std_curvature_noncontactborders_list_red = []

all_curvatures_pathfinder_list_red = []
mean_curvature_pathfinder_list_red = []
median_curvature_pathfinder_list_red = []
sem_curvature_pathfinder_list_red = []
std_curvature_pathfinder_list_red = []

all_curvatures_contactsurface_list_red = []
mean_curvature_contactsurface_list_red = []
median_curvature_contactsurface_list_red = []
sem_curvature_contactsurface_list_red = []
std_curvature_contactsurface_list_red = []


curvature_values_grn_list = []
mean_curvature_curveborders_list_grn = []
median_curvature_curveborders_list_grn = []
sem_curvature_curveborders_list_grn = []
std_curvature_curveborders_list_grn = []

all_curvatures_noncontactborders_list_grn = []
mean_curvature_noncontactborders_list_grn = []
median_curvature_noncontactborders_list_grn = []
sem_curvature_noncontactborders_list_grn = []
std_curvature_noncontactborders_list_grn = []

all_curvatures_pathfinder_list_grn = []
mean_curvature_pathfinder_list_grn = []
median_curvature_pathfinder_list_grn = []
sem_curvature_pathfinder_list_grn = []
std_curvature_pathfinder_list_grn = []

all_curvatures_contactsurface_list_grn = []
mean_curvature_contactsurface_list_grn = []
median_curvature_contactsurface_list_grn = []
sem_curvature_contactsurface_list_grn = []
std_curvature_contactsurface_list_grn = []


all_curvatures_pathfinder_list = []
mean_curvature_pathfinder_list = []
median_curvature_pathfinder_list = []
sem_curvature_pathfinder_list = []
std_curvature_pathfinder_list = []


all_curvatures_pathfinder_list_angle1 = []
mean_curvature_pathfinder_list_angle1 = []
median_curvature_pathfinder_list_angle1 = []
sem_curvature_pathfinder_list_angle1 = []
std_curvature_pathfinder_list_angle1 = []


all_curvatures_pathfinder_list_angle2 = []
mean_curvature_pathfinder_list_angle2 = []
median_curvature_pathfinder_list_angle2 = []
sem_curvature_pathfinder_list_angle2 = []
std_curvature_pathfinder_list_angle2 = []

all_curvatures_pathfinder_list_grn_angle1 = []
mean_curvature_pathfinder_list_grn_angle1 = []
median_curvature_pathfinder_list_grn_angle1 = []
sem_curvature_pathfinder_list_grn_angle1 = []
std_curvature_pathfinder_list_grn_angle1 = []

all_curvatures_pathfinder_list_grn_angle2 = []
mean_curvature_pathfinder_list_grn_angle2 = []
median_curvature_pathfinder_list_grn_angle2 = []
sem_curvature_pathfinder_list_grn_angle2 = []
std_curvature_pathfinder_list_grn_angle2 = []

all_curvatures_pathfinder_list_red_angle1 = []
mean_curvature_pathfinder_list_red_angle1 = []
median_curvature_pathfinder_list_red_angle1 = []
sem_curvature_pathfinder_list_red_angle1 = []
std_curvature_pathfinder_list_red_angle1 = []

all_curvatures_pathfinder_list_red_angle2 = []
mean_curvature_pathfinder_list_red_angle2 = []
median_curvature_pathfinder_list_red_angle2 = []
sem_curvature_pathfinder_list_red_angle2 = []
std_curvature_pathfinder_list_red_angle2 = []

bin_curvature_details_red_list = []
bin_curvature_details_grn_list = []

whole_cell_curvature_details_red_list = []
whole_cell_curvature_details_grn_list = []

bin_indices_red_list = []
bin_indices_grn_list = []

bin_start_red_list = []
bin_stop_red_list = []

bin_start_grn_list = []
bin_stop_grn_list = []

cells_touching_list = []

script_version_list = []


total_files = len(sorted(glob.glob((path)+'/*'+ext)))
for filenumber, filename in enumerate(sorted(glob.glob((path)+'/*'+ext))):
    if filenumber == 0: 
        start_time = time.perf_counter()

#    ### Tell me filenumber and filename of what you're currently working on!
#        print '\n',(filenumber+1), filename  
        cell_raw = Image.open(filename)
        ### You should note that PIl will automatically close the file when it's not needed
        cell_raw = np.array(cell_raw)

    ### Check if the signal:noise ratio is too low in any 1 channel. If it is
    ### then set that channel to be black:
        cell = check_for_signal(cell_raw)
        
        ### Split the file up in to it's RGB 3-colour-based arrays (ie. 3 channels):
        redchan = mh.stretch(cell[:,:,0])
        grnchan = mh.stretch(cell[:,:,1])
        bluchan = mh.stretch(cell[:,:,2])

    ### How many pixels do you want to use to create vectors used to measure angles?
        while True:
            try:
                Nth_pixel = input("How many pixels would you like to use to measure"
                              " each arm of each angle? (I suggest 9 for Xenopus"
                              " and 5 for zebrafish)\n>")
                # Nth_pixel = r'9'
                if type(eval(Nth_pixel)) == float or type(eval(Nth_pixel)) == int:
                    Nth_pixel = eval(Nth_pixel)
                    break
                else:
                    print("Sorry I didn't understand that. Please answer again.")
                    continue
            except (ValueError, NameError, SyntaxError, TypeError):
                print("Sorry I didn't understand that. Please answer again.")
                continue
    
    ### How many pixels do you want to use for the linescan that measure fluorescence
    ### across the cell-cell contacts?
        while True:
            try:
                linescan_length = input("How many pixels long would you like to make "
                                    "each linescan of the gap?\n>")
                # linescan_length = r'12'
                if type(eval(linescan_length)) == float or type(eval(linescan_length)) == int:
                    linescan_length = eval(linescan_length)
                    break
                else:
                    print("Sorry I didn't understand that. Please answer again.")
                    continue          
            except (ValueError, NameError, SyntaxError, TypeError):
                print("Sorry I didn't understand that. Please answer again.")
                continue
            
    ### How many pixels do you want to use for the first method of curvature
    ### calculation?:
        while True:
            try:
                curve_length = input("How many pixels would you like to use to " 
                                          "estimate curvatures at each pixel? (even "
                                          "numbers will be rounded up to the nearest "
                                          "odd number)\n>")
                # curve_length = r'9'
                if type(eval(curve_length)) == float or type(eval(curve_length)) == int:
                    curve_length = eval(curve_length)
                    break
                else:
                    print ("Sorry I didn't understand that. Please answer again "
                           "with an integer.")
                    continue
            except (ValueError, NameError, SyntaxError, TypeError):
                print("Sorry I didn't understand that. Please answer again.")
                continue
    
    ### Approximately how many pixels is the radius of your cell?:
        while True:
            try:
                # Use 6.67% of the hypotenuse of the sides of the image
                #disk_radius = np.hypot(*np.asarray(np.shape(redchan))) * 0.0667
                
                # Alternatively use 16 as the initial cell radius:
                disk_radius = 16
                
                                         
                disk_radius = input('About how many pixels is the radius of your ' 
                                          'cell? [default/recommended: %.1f]' 
                                          '\n>' %(disk_radius)
                                          ) or disk_radius
                # disk_radius = disk_radius
                                         
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


    ### What do you consider definitely background fluorescence?:
        while True:
            try:
                # Use 10% of the images dtype max as the default threshold for
                # determining whether a segmented area should be kept or removed:
                background_threshold_percent = 0.1
                background_threshold_percent = input('What % (from 0-1) of your images data type can '
                                                              'be considered definitely background? [default: 10%]'
                                                              '\n>') or background_threshold_percent

                background_threshold_percent = float(background_threshold_percent)
                # background_threshold_cutoff_red = background_threshold_percent * np.iinfo(cell.dtype).max
                # background_threshold_cutoff_grn = background_threshold_percent * np.iinfo(cell.dtype).max
                background_threshold = background_threshold_percent * np.iinfo(cell.dtype).max
                break
            except (ValueError, NameError, SyntaxError, TypeError):
                print("Sorry I didn't understand that. Please answer again "
                      "with a decimal number.")
                continue

    ### What is the dimmest object of interest?:
        while True:
            try:
                # Use 20% of the images dtype max as the default threshold for
                # determining whether a segmented area should be kept or removed:
                object_of_interest_brightness_cutoff_percent = 0.2#0.05
                otsu_fraction = float('nan')
                object_of_interest_brightness_cutoff_percent = input('What % (from 0-1) of your images data type can '
                                                              'be considered possibly of interest? [default: 20%] [can also enter "otsu" to have it auto select for each image]'
                                                              '\n>') or object_of_interest_brightness_cutoff_percent
                # object_of_interest_brightness_cutoff_percent = object_of_interest_brightness_cutoff_percent
                if object_of_interest_brightness_cutoff_percent == 'otsu':
                    otsu_fraction = input('What % (from 0-1) of the otsu threshold would you like to use? [default: 0.8 , ie. 80%]'
                                          '\n>') or 0.8
                                          ### Make the otsu_fraction a part of the
                                          # script and then include it in the table
                    otsu_fraction = float(otsu_fraction)
                    if redchan.any():
                        object_of_interest_brightness_cutoff_red = filters.threshold_otsu(redchan) * otsu_fraction
                    else:
                        object_of_interest_brightness_cutoff_red = background_threshold
                    if grnchan.any():
                        object_of_interest_brightness_cutoff_grn = filters.threshold_otsu(grnchan) * otsu_fraction
                    else:
                        object_of_interest_brightness_cutoff_grn = background_threshold
                    # The above lines are applied to each image, so this is 
                    # done within the for image_number>1 loop as well.
                    break
                else:
                    object_of_interest_brightness_cutoff_percent = float(object_of_interest_brightness_cutoff_percent)
                    object_of_interest_brightness_cutoff_red = object_of_interest_brightness_cutoff_percent * np.iinfo(cell.dtype).max
                    object_of_interest_brightness_cutoff_grn = object_of_interest_brightness_cutoff_percent * np.iinfo(cell.dtype).max
                    break
                #else:
                #    print ("Sorry I didn't understand that. Please answer again "
                #           "with a decimal number.")
                #    continue
                #if disk_radius == '':
                #    # If no input then just use the default disk_radius
                    
                #    break
                #else:
                #    disk_radius = float(disk_radius)
                #    break
            except (ValueError, NameError, SyntaxError, TypeError):
                print("Sorry I didn't understand that. Please answer again "
                      "with a decimal number.")
                continue


    ###############################################################################
    ### A lot of figures are going to be opened before they are saved in to a
    ### multi-page pdf, so increase the max figure open warning:
        plt.rcParams.update({'figure.max_open_warning':300})

    ### Define some default values that will be used in the script:
        script_version = os.path.basename(__file__)
        sobel = 'Y'
        sobel_filt = 'True'
        gauss = 'Y'
        gaussian_filter_value_red = 2.0
        gaussian_filter_value_grn = 2.0
        gaussian_filter_value_blu = 2.0
        #gaussian_filter_value_flat = 2.0
        if object_of_interest_brightness_cutoff_percent == 'otsu':
            otsu_thresholding = 'Y'
        else:
            otsu_thresholding = 'N'
        red_otsu_fraction = 0.5#0.7#0.9#1.0
        grn_otsu_fraction = 0.5#0.9#1.0
        blu_otsu_fraction = 0.5#0.9#1.0
        hist_percentile_cutoff = 0.9
        odr_circle_fit_iterations = 30
        num_bins = 8
        skip = 'n'
        
    ### Some deprecated legacy values:
        #redthreshval = np.float('nan')
        #grnthreshval = np.float('nan')

###
        ### This is a threshold that determines how dim something in the image
        ### has to be to be considered background, it's set to 10% of the 
        ### images dtype max:
        # background_threshold = np.iinfo(cell.dtype).max * 0.1#0.05


###
        ### The roundness_thresh determines how round a region has to be to be
        ### recognized as a cell:
        roundness_thresh = 1.3 ##1.25 ##1.2
        
        ### The size_filter determines how many px^2 the area of a region has
        ### to be in order to be recognized as a cell:
        size_filter = 1000#20 #the 20 filter is used for the 2 circles simulation
        #size_filter = 20#20 #the 20 filter is used for the 2 circles simulation
        
        particle_filter = 250#20 #the 20 filter is used for the 2 circle simulation
#        particle_filter = 20#20 #the 20 filter is used for the 2 circle simulation
    
        ### Create a maxima-specific structuring element to try and reduce the
        ### problem of watershed oversegmentation: was 24, 24 before Aug. 24 2017
        ### problem of undersegmentation: was 48,48 before Feb. 02, 2018
        #maxima_Bc = morphology.disk(12.5) #15.5 = diameter of 30
        maxima_Bc = morphology.disk(disk_radius) #15.5 = diameter of 30

#        maxima_Bc = np.ones((48,48))
#        maxima_Bc = np.ones((60,60))
###

    ### Define the default values for the the data that will populate the table
    ### that will be output at the end:
        red_gap_scanmean = np.asarray([float('nan')]*linescan_length)
        grn_gap_scanmean = np.asarray([float('nan')]*linescan_length)
        red_gap_scanx = [float('nan')]
        red_gap_scany = [float('nan')]
        grn_gap_scanx = [float('nan')]
        grn_gap_scany = [float('nan')]
        degree_angle1 = float('nan')
        degree_angle2 = float('nan')
        average_angle = float('nan')
        contactsurface_len = float('nan')
        noncontactsurface_len1 = float('nan')
        noncontactsurface_len2 = float('nan')
        area_cell_1 = float('nan')
        area_cell_2 = float('nan')
        perimeter_cell1 = float('nan')
        perimeter_cell2 = float('nan')
        noncontact_perimeter_cell1 = float('nan')
        noncontact_perimeter_cell2 = float('nan')
        intersectsx = float('nan')
        intersectsy = float('nan')
        intersection_points = [(float('nan'), float('nan'))]
        intersection_values_per_cell = [float('nan')]
        intersection_values_per_cell_middle_list_index = int(0)
        intersection_cellpair_num = filenumber+1
        cleanborders = np.zeros(cell.shape)
        contactsurface = np.zeros(cell.shape)
        noncontactborders = np.zeros(cell.shape)
        curveborders_curvature_labeled = np.zeros(cell.shape)
        pathfinder = np.zeros(cell.shape)
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
        grn_halfangle1_rad = float('nan')
        grn_halfangle2_rad = float('nan')
        red_halfangle1 = float('nan')
        red_halfangle2 = float('nan')
        grn_halfangle1 = float('nan')
        grn_halfangle2 = float('nan')
        contactsurface_len1 = float('nan')
        contactsurface_len2 = float('nan')
        numbered_border_indices_red = [float('nan'), (float('nan'),float('nan'),float('nan'))]
        numbered_border_indices_grn = [float('nan'), (float('nan'),float('nan'),float('nan'))]
        numbered_noncontactborder_indices_red = [float('nan'), (float('nan'),float('nan'),float('nan'))]
        numbered_noncontactborder_indices_grn = [float('nan'), (float('nan'),float('nan'),float('nan'))]
        
        angles_angle_labeled = np.zeros(cell.shape)
        
        mean_curvature_curveborders_red = float('nan')
        median_curvature_curveborders_red = float('nan')
        sem_curvature_curveborders_red = float('nan')
        std_curvature_curveborders_red = float('nan')
        
        all_curvatures_noncontactborders_red = [float('nan')]
        mean_curvature_noncontactborders_red = float('nan')
        median_curvature_noncontactborders_red = float('nan')
        sem_curvature_noncontactborders_red = float('nan')
        std_curvature_noncontactborders_red = float('nan')
        
        curvature_pathfinder_red = [float('nan')]
        mean_curvature_pathfinder_red = float('nan')
        median_curvature_pathfinder_red = float('nan')
        sem_curvature_pathfinder_red = float('nan')
        std_curvature_pathfinder_red = float('nan')
        
        curvature_contactsurface_red = [float('nan')]
        mean_curvature_contactsurface_red = float('nan')
        median_curvature_contactsurface_red = float('nan')
        sem_curvature_contactsurface_red = float('nan')
        std_curvature_contactsurface_red = float('nan')
        
        
        mean_curvature_curveborders_grn = float('nan')
        median_curvature_curveborders_grn = float('nan')
        sem_curvature_curveborders_grn = float('nan')
        std_curvature_curveborders_grn = float('nan')
        
        all_curvatures_noncontactborders_grn = [float('nan')]
        mean_curvature_noncontactborders_grn = float('nan')
        median_curvature_noncontactborders_grn = float('nan')
        sem_curvature_noncontactborders_grn = float('nan')
        std_curvature_noncontactborders_grn = float('nan')
        
        curvature_pathfinder_grn = [float('nan')]
        mean_curvature_pathfinder_grn = float('nan')
        median_curvature_pathfinder_grn = float('nan')
        sem_curvature_pathfinder_grn = float('nan')
        std_curvature_pathfinder_grn = float('nan')
        
        curvature_contactsurface_grn = [float('nan')]
        mean_curvature_contactsurface_grn = float('nan')
        median_curvature_contactsurface_grn = float('nan')
        sem_curvature_contactsurface_grn = float('nan')
        std_curvature_contactsurface_grn = float('nan')
        
        all_curvatures_pathfinder = [float('nan')]
        mean_curvature_pathfinder = float('nan')
        median_curvature_pathfinder = float('nan')
        sem_curvature_pathfinder = float('nan')
        std_curvature_pathfinder = float('nan')
        
        curvature_pathfinder_angle1 = [float('nan')]
        mean_curvature_pathfinder_angle1 = float('nan')
        median_curvature_pathfinder_angle1 = float('nan')
        sem_curvature_pathfinder_angle1 = float('nan')
        std_curvature_pathfinder_angle1 = float('nan')
        
        curvature_pathfinder_angle2 = [float('nan')]
        mean_curvature_pathfinder_angle2 = float('nan')
        median_curvature_pathfinder_angle2 = float('nan')
        sem_curvature_pathfinder_angle2 = float('nan')
        std_curvature_pathfinder_angle2 = float('nan')
        
        all_curvatures_pathfinder_grn_angle1 = [float('nan')]
        mean_curvature_pathfinder_grn_angle1 = float('nan')
        median_curvature_pathfinder_grn_angle1 = float('nan')
        sem_curvature_pathfinder_grn_angle1 = float('nan')
        std_curvature_pathfinder_grn_angle1 = float('nan')
        
        all_curvatures_pathfinder_grn_angle2 = [float('nan')]
        mean_curvature_pathfinder_grn_angle2 = float('nan')
        median_curvature_pathfinder_grn_angle2 = float('nan')
        sem_curvature_pathfinder_grn_angle2 = float('nan')
        std_curvature_pathfinder_grn_angle2 = float('nan')
        
        all_curvatures_pathfinder_red_angle1 = [float('nan')]
        mean_curvature_pathfinder_red_angle1 = float('nan')
        median_curvature_pathfinder_red_angle1 = float('nan')
        sem_curvature_pathfinder_red_angle1 = float('nan')
        std_curvature_pathfinder_red_angle1 = float('nan')
        
        all_curvatures_pathfinder_red_angle2 = [float('nan')]
        mean_curvature_pathfinder_red_angle2 = float('nan')
        median_curvature_pathfinder_red_angle2 = float('nan')
        sem_curvature_pathfinder_red_angle2 = float('nan')
        std_curvature_pathfinder_red_angle2 = float('nan')

        bin_curvature_details_red = [[float('nan')] * len(bin_curvature_details_headers)] * (num_bins+1)
        bin_curvature_details_grn = [[float('nan')] * len(bin_curvature_details_headers)] * (num_bins+1)
        
        whole_curvature_details_red = [[float('nan')] * len(bin_curvature_details_headers)] * 2
        whole_curvature_details_grn = [[float('nan')] * len(bin_curvature_details_headers)] * 2
        
        bin_indices_red = [[(float('nan'), float('nan'), float('nan'))]] * num_bins
        bin_indices_grn = [[(float('nan'), float('nan'), float('nan'))]] * num_bins
        
#        bin_start_red = [float('nan')] * num_bins
#        bin_stop_red = [float('nan')] * num_bins
        
#        bin_start_grn = [float('nan')] * num_bins
#        bin_stop_grn = [float('nan')] * num_bins

        bin_start_red_indices = np.empty((num_bins, 3))
        bin_start_red_indices[:] = np.nan
        
        bin_stop_red_indices = np.empty((num_bins, 3))
        bin_stop_red_indices[:] = np.nan

        bin_start_grn_indices = np.empty((num_bins, 3))
        bin_start_grn_indices[:] = np.nan

        bin_stop_grn_indices = np.empty((num_bins, 3))
        bin_stop_grn_indices[:] = np.nan

###
        ### These 2 variable assignments are unique to the first file in the
        ### folder:
        mean_closest_indice_loc_red_old = float('nan')
        mean_closest_indice_loc_grn_old = float('nan')
        

        ### Segment your cells:
        thresh_based_areas, wscan_based_areas_out, wscan_based_areas_in, wscan_based_maxima, Bc3D, edge_map = watershed_on_cannyweighted_sobel_based_segment_and_isolate_center_cells(cell, gaussian_filter_value_red, gaussian_filter_value_grn, gaussian_filter_value_blu, maxima_Bc, background_threshold, object_of_interest_brightness_cutoff_red, object_of_interest_brightness_cutoff_grn)
  
        justcellpair = np.copy(cell)
        justcellpair[np.where(wscan_based_areas_out)] = 0
        justcellpair[:,:,0] = mh.stretch(justcellpair[:,:,0])
        justcellpair[:,:,1] = mh.stretch(justcellpair[:,:,1])
        justcellpair[:,:,2] = mh.stretch(justcellpair[:,:,2])
        
        wscan_border_red = mh.borders(wscan_based_areas_in[:,:,0]) * wscan_based_areas_in[:,:,0]
        borders_red = mh.thin(wscan_border_red) * 1
        
        
        
        wscan_border_grn = mh.borders(wscan_based_areas_in[:,:,1]) * wscan_based_areas_in[:,:,1]

        borders_grn = mh.thin(wscan_border_grn) * 2

        wscan_border_blu = mh.borders(wscan_based_areas_in[:,:,2]) * wscan_based_areas_in[:,:,2]

        borders_blu = mh.thin(wscan_border_blu) * 3

        curveborders = np.dstack((borders_red, borders_grn, borders_blu))
        curveborders = clean_up_borders(curveborders)
        borders_red, borders_grn, borders_blu = np.squeeze(np.dsplit(curveborders, 3))
        semifinalflatcellareas = np.dstack(( scp_morph.binary_fill_holes(borders_red), scp_morph.binary_fill_holes(borders_grn), scp_morph.binary_fill_holes(borders_blu) ))

### The part below is currently taking the inner edges, can take the outer by
### adding a "~"
        borders_red = mh.borders(semifinalflatcellareas[:,:,0]) * semifinalflatcellareas[:,:,0]
        borders_grn = mh.borders(semifinalflatcellareas[:,:,1]) * semifinalflatcellareas[:,:,1]
        borders_blu = mh.borders(semifinalflatcellareas[:,:,2]) * semifinalflatcellareas[:,:,2]
        semifinalflatcellareas = np.dstack(( scp_morph.binary_fill_holes(borders_red), scp_morph.binary_fill_holes(borders_grn), scp_morph.binary_fill_holes(borders_blu) ))

#        semifinalflatcellareas = np.logical_or(thresh_based_areas_in, wscan_based_areas).astype(int)
#        semifinalflatcellareas = np.copy(wscan_based_areas)
#        finalflatcellareas, cell_maxima = separate_red_and_green_cells(cell, semifinalflatcellareas, maxima_Bc)
        finalflatcellareas, cell_maxima = separate_red_and_green_cells(justcellpair, semifinalflatcellareas, maxima_Bc, edge_map)

        ### Clean up the segmented image of any regions (ie. "cells") that are too
        ### small or that are touching the borders:
        finalflatcellareas = remove_bordering_and_small_regions(finalflatcellareas)
###
        ### Do a check of the finalflatcellareas before trying to resolve and 
        ### clean the borders of the cells:
        
        ### Make sure that the cells meet the minimum size threshold:
        size_red = mh.labeled.labeled_size(finalflatcellareas[:,:,0])
        size_grn = mh.labeled.labeled_size(finalflatcellareas[:,:,1])
        #size_blu = mh.labeled.labeled_size(finalflatcellareas[:,:,2])
        
        finalflatcellareas[:,:,0] = mh.labeled.remove_regions(finalflatcellareas[:,:,0], size_red < size_filter)
        finalflatcellareas[:,:,1] = mh.labeled.remove_regions(finalflatcellareas[:,:,1], size_grn < size_filter)
        
###
        ### Clean up the segmented image of any regions (ie. "cells") that are too
        ### small or that are touching the borders:
#        no_border_cells = remove_bordering_and_small_regions(cellareas)
###

        ### Need to do a second iteration of image thresholding and segmentation
        ### in a slightly different way on the array of the image of just the 
        ### cells in the middle to better resolve the edges of the cells:
#        finalflatcellareas, newcellspotsflat, justcellpair = refine_center_cells_segmentation_and_isolation(no_border_cells, cellthresh, gaussian_filter_value_red, red_otsu_fraction, gaussian_filter_value_grn, grn_otsu_fraction, gaussian_filter_value_blu, blu_otsu_fraction, maxima_Bc)
#        finalflatcellareas = cellareas[:,:,0] + cellareas[:,:,1] + cellareas[:,:,2]
###

        ### Use the refined cell segmentations to create the cell borders:
#        newredarea = (finalflatcellareas == 1) * 1
        newredarea = finalflatcellareas[:,:,0] * 1
        newredcellborders = mh.borders(newredarea.astype(bool)) * newredarea
#        newredcellborders = mh.borders(newredarea.astype(bool)) * ~newredarea.astype(bool)
        newredcellborders = mh.thin(newredcellborders).astype(bool) * 1
#        newgrnarea = (finalflatcellareas == 2) * 2
        newgrnarea = finalflatcellareas[:,:,1] * 2
        newgrncellborders = mh.borders(newgrnarea.astype(bool)) * newgrnarea
#        newgrncellborders = mh.borders(newgrnarea.astype(bool)) * ~newgrnarea.astype(bool)
        newgrncellborders = mh.thin(newgrncellborders).astype(bool) * 2
        newbluarea = np.zeros((finalflatcellareas[:,:,2].shape))
        newblucellborders = np.zeros((finalflatcellareas[:,:,2].shape))
        
        curveborders = np.dstack((newredcellborders, newgrncellborders, newblucellborders))
###


###
        ### Clean the borders used for curves so the algorithm used to count
        ### pixels and get sequential info from the borders doesn't get lost:
        curveborders = clean_up_borders(curveborders)
        
#        padded_cell = pad_img_edges(cell, 0)
#        cleanoverlay = np.copy(padded_cell)
        cleanoverlay = np.copy(cell)
        cleanoverlay[:,:,0] = mh.stretch(cleanoverlay[:,:,0])
        cleanoverlay[:,:,1] = mh.stretch(cleanoverlay[:,:,1])
        cleanoverlay[:,:,2] = mh.stretch(cleanoverlay[:,:,2])
        cleanoverlay[np.where(curveborders)] = 255
        cleanoverlay = mh.overlay(np.max(cell, axis=2), red=curveborders[:,:,0], green=curveborders[:,:,1])

        ### Return the indices of the curveborder pixels based on colour:
        curvecoordsred = list(zip(*np.where(curveborders[:,:,0])))
        curvecoordsred_name = 'curvecoordsred'
        curvecoordsgrn = list(zip(*np.where(curveborders[:,:,1])))
        curvecoordsgrn_name = 'curvecoordsgrn'
        curvecoordsblu = list(zip(*np.where(curveborders[:,:,2])))
        curvecoordsblu_name = 'curvecoordsblu'

        ### Use n pixels to the left and right of a given pixel to define a circle
        ### and therefore the curvature at that pixel:
        curve_definitions_red, curve_vals_red = pixel_by_pixel_curvatures(curveborders[:,:,0], curvecoordsred, curvecoordsred_name)
        curve_definitions_grn, curve_vals_grn = pixel_by_pixel_curvatures(curveborders[:,:,1], curvecoordsgrn, curvecoordsgrn_name)
        curve_definitions_blu, curve_vals_blu = pixel_by_pixel_curvatures(curveborders[:,:,2], curvecoordsblu, curvecoordsblu_name)

        ### Assign the curvature value of each pixel to it's location in curveborders:
        curveborders_curvature_labeled = np.zeros(curveborders.shape)
        for x in curve_vals_red:
            curveborders_curvature_labeled[x[1], x[0], 0] = (1.0/x[2]) * x[5]
        for x in curve_vals_grn:
            curveborders_curvature_labeled[x[1], x[0], 1] = (1.0/x[2]) * x[5]
        for x in curve_vals_blu:
            curveborders_curvature_labeled[x[1], x[0], 2] = (1.0/x[2]) * x[5]

        ### Mask the zeros that are not part of the borders:
        masked = np.ma.array(data = curveborders_curvature_labeled, mask = curveborders==0)

        ### Make a copy of the curveborders because we will need to clean them 
        ### up slightly differently for cells that are in contact:
        cleanborders = np.copy(curveborders)        
        cleanareas = np.dstack((newredarea, newgrnarea, newbluarea))
        ### Isolate the watershed segmentations seed points so that you only have the 2
        ### seed points that detect the cells:
        newcellmaxima = np.copy(cell_maxima)
#        newcellmaxima = np.dstack([mh.distance(cleanareas[:,:,i]) for i in range(cleanareas.shape[2])])
#        newcellmaxima = np.expand_dims(newcellspotsflat, axis=2).astype(bool)*cleanareas
        ### Return the seed point locations (in  (row,col,depth) format):
#        newcellmaxima_rcd = ([np.mean(np.where(newcellmaxima[:,:,0])[0]), np.mean(np.where(newcellmaxima[:,:,0])[1])], [np.mean(np.where(newcellmaxima[:,:,1])[0]), np.mean(np.where(newcellmaxima[:,:,1])[1])])
        newcellmaxima_rcd = ([np.where(newcellmaxima[:,:,0])[0], np.where(newcellmaxima[:,:,0])[1]], [np.where(newcellmaxima[:,:,1])[0], np.where(newcellmaxima[:,:,1])[1]])

#        cleanoverlay[:,:,0] = mh.stretch(np.copy(cell[:,:,0]))
#        cleanoverlay[:,:,1] = mh.stretch(np.copy(cell[:,:,1]))
#        cleanoverlay[:,:,2] = mh.stretch(np.copy(cell[:,:,2]))
        ### Plot the cells and any parameters so far:
        fig = plt.figure(num=str(filenumber+1)+'_'+os.path.basename(filename), figsize=(12., 10.))
        plt.suptitle(str(filenumber+1)+' - '+os.path.basename(filename), verticalalignment='center', x=0.5, y=0.5, fontsize=14)
        ax1 = fig.add_subplot(231)
        ax1.yaxis.set_visible(False)
        ax1.xaxis.set_visible(False)
        ax1.imshow(mh.stretch(cell_raw))
        
        ax2 = fig.add_subplot(232)
        ax2.yaxis.set_visible(False)
        ax2.xaxis.set_visible(False)
#        figcellborderred = ax2.imshow(mh.overlay(gray = justcellpair[:,:,0], red = cleanborders[:,:,0]))
#        figcellborderred = ax2.imshow(mh.overlay(gray = mh.stretch(padded_cell[:,:,0]), red = cleanborders[:,:,0]))
        figcellborderred = ax2.imshow(mh.overlay(gray = mh.stretch(cell[:,:,0]), red = cleanborders[:,:,0]))
        ax2.axes.autoscale(False)
        ax2.scatter(newcellmaxima_rcd[0][1], newcellmaxima_rcd[0][0], color='m', marker='o')
        
        ax3 = fig.add_subplot(233)
        ax3.yaxis.set_visible(False)
        ax3.xaxis.set_visible(False)
#        figcellbordergrn = ax3.imshow(mh.overlay(gray = justcellpair[:,:,1], green = cleanborders[:,:,1]))
#        figcellbordergrn = ax3.imshow(mh.overlay(gray = mh.stretch(padded_cell[:,:,1]), green = cleanborders[:,:,1]))
        figcellbordergrn = ax3.imshow(mh.overlay(gray = mh.stretch(cell[:,:,1]), green = cleanborders[:,:,1]))
        ax3.axes.autoscale(False)
        ax3.scatter(newcellmaxima_rcd[1][1], newcellmaxima_rcd[1][0], color='m', marker='o')
        
        ax4 = fig.add_subplot(235)
        ax4.yaxis.set_visible(False)
        ax4.xaxis.set_visible(False)
        ax4.imshow(justcellpair)
#        ax4.imshow(mh.stretch(cell))
        ax4.axes.autoscale(False)
        
        ax5 = fig.add_subplot(236)
        ax5.yaxis.set_visible(False)
        ax5.xaxis.set_visible(False)
        ax5.imshow(justcellpair)
#        ax5.imshow(mh.stretch(cell))
        ax5.axes.autoscale(False)

        ax6 = fig.add_subplot(234)
        ax6.yaxis.set_visible(False)
        ax6.xaxis.set_visible(False)
        ax6.imshow(cleanoverlay)
        ax6.axes.autoscale(False)

        plt.tight_layout()
        plt.draw()
#        plt.show('block'== False)


        ### Tell me the filenumber and filename of what you're working on:
        print('\n',(filenumber+1), filename)

        ### Skip the file if there is only 1 cell in the picture (sometimes the
        ### microscope skips a channel):
#        if len(mh.labeled.labeled_size(cleanareas)) < 3:
        if len(np.unique(cleanborders)) < 3:
            msg = "There aren't 2 cells in this image."
            print(msg)
            skip = 'y'
            
        ### Create an initial guess at the borders of the cells that are in
        ### contact with one another:
        rawcontactsurface = np.zeros(cleanareas.shape, np.dtype(np.int32))
        for (i,j,k), value in np.ndenumerate(cleanborders):
            if (0 in cleanborders[sub(i,1):i+2, sub(j,1):j+2, sub(k,1):k+2]) and (1 in cleanborders[sub(i,1):i+2, sub(j,1):j+2, sub(k,1):k+2]) and (2 in cleanborders[sub(i,1):i+2, sub(j,1):j+2, sub(k,1):k+2]):
                rawcontactsurface[i,j,k] = cleanborders[i,j,k]

        ### If this initial guess has too few pixels in contact then consider
        ### the cells to not be in contact:
        if np.count_nonzero(rawcontactsurface) <= 6:
            print('These cells are not in contact.')
            cells_touching = False
        elif np.count_nonzero(rawcontactsurface) > 6:
            cells_touching = True

        ### If there aren't 2 cells in the image just skip it.
        if skip == 'y':
            listupdate()
            continue

        ### If the cells are not in contact then set a bunch of values that will
        ### be in the table to be null and proceed to bin and estimate their 
        ### curvatures:
        if cells_touching == False and skip == 'n':

            ### Give the pixels of the cell borders sequential positional info
            ### starting from an arbitrary starting point:
            ordered_border_indices_red, ordered_border_indices_grn, numbered_border_indices_red, numbered_border_indices_grn, perimeter_cell1, perimeter_cell2 = count_pixels_around_borders_sequentially(curveborders)

            ### For 2 cells that are not in contact:
            ### Need to start by calculating smallest distance between the red and green borders
            closest_redborder_indices_rc, closest_grnborder_indices_rc, mean_closest_indice_loc_red, mean_closest_indice_loc_grn = find_closest_pixels_indices_between_cell_borders(curveborders[:,:,0], curveborders[:,:,1], ordered_border_indices_red, ordered_border_indices_grn, mean_closest_indice_loc_red_old, mean_closest_indice_loc_grn_old)
            mean_closest_indice_loc_red_old = np.copy(mean_closest_indice_loc_red)
            mean_closest_indice_loc_grn_old = np.copy(mean_closest_indice_loc_grn)

            ### Find the bin closest to the other cell and remove it:
            num_bins = num_bins
            far_borders_red, size_list_of_other_bins_red, closest_bin_size_red, closest_bin_indices_red = create_and_remove_bin_closest_to_other_cell(curveborders[:,:,0].astype(bool), ordered_border_indices_red, mean_closest_indice_loc_red, num_bins)
            
            ### Do the same for the green borders:
            far_borders_grn, size_list_of_other_bins_grn, closest_bin_size_grn, closest_bin_indices_grn = create_and_remove_bin_closest_to_other_cell(curveborders[:,:,1].astype(bool), ordered_border_indices_grn, mean_closest_indice_loc_grn, num_bins)

            ### Combine the 2 borders into an RGB image and assign red to be 1 
            ### and green to be 2:
            borders_without_closest_bin = np.dstack((far_borders_red*1, far_borders_grn*2, np.zeros((cell.shape))))
            closest_bin_red_and_grn = np.zeros(np.shape(cell))
            # closest_bin_red_and_grn[list(zip(*closest_bin_indices_red))] = 1
            # closest_bin_red_and_grn[list(zip(*closest_bin_indices_grn))] = 2
            closest_bin_red_and_grn[tuple(np.asarray(closest_bin_indices_red).T)] = 1
            closest_bin_red_and_grn[tuple(np.asarray(closest_bin_indices_grn).T)] = 2
            reordered_closest_bin_indices_red, reordered_closest_bin_indices_grn, closest_bin_len_red, closest_bin_len_grn = make_open_border_indices_sequential(closest_bin_red_and_grn)

            ### Re-define the sequence of pixels for the borders_without_closest_bin:
            reordered_border_indices_red, reordered_border_indices_grn, perimeter_of_bins_1to7_cell1, perimeter_of_bins_1to7_cell2 = make_open_border_indices_sequential(borders_without_closest_bin)
            
            renumbered_border_indices_red = [p for p in enumerate(reordered_border_indices_red)]
            renumbered_border_indices_grn = [p for p in enumerate(reordered_border_indices_grn)]
            
            ### Next I need to use these renumbered border indices red and grn to define the
            ### other 7 bins and find their curvatures:
            num_bins = num_bins-1
            bin_indices_red, bin_start_red, bin_stop_red = sort_pixels_into_bins(reordered_border_indices_red, size_list_of_other_bins_red, closest_bin_indices_red, num_bins)
            bin_indices_grn, bin_start_grn, bin_stop_grn = sort_pixels_into_bins(reordered_border_indices_grn, size_list_of_other_bins_grn, closest_bin_indices_grn, num_bins)
            
            bin_start_red_indices = np.array(reordered_border_indices_red)[bin_start_red]            
            bin_start_red_indices = np.row_stack((bin_start_red_indices, reordered_closest_bin_indices_red[0]))
            bin_stop_red_indices = np.array(reordered_border_indices_red)[np.array(bin_stop_red)-1]
            bin_stop_red_indices = np.row_stack((bin_stop_red_indices, reordered_closest_bin_indices_red[-1]))

            bin_start_grn_indices = np.array(reordered_border_indices_grn)[bin_start_grn]            
            bin_start_grn_indices = np.row_stack((bin_start_grn_indices, reordered_closest_bin_indices_grn[0]))
            bin_stop_grn_indices = np.array(reordered_border_indices_grn)[np.array(bin_stop_grn)-1]
            bin_stop_grn_indices = np.row_stack((bin_stop_grn_indices, reordered_closest_bin_indices_grn[-1]))
            

            ### Finally, fit arcs to the binned borders and plot and return the
            ### the details to me:
            bin_curvature_details_red, all_fits_red_binned = fit_curves_to_bins(bin_indices_red, curveborders[:,:,0], odr_circle_fit_iterations)
            bin_curvature_details_grn, all_fits_grn_binned = fit_curves_to_bins(bin_indices_grn, curveborders[:,:,1], odr_circle_fit_iterations)

            ### While we're at it, also give me the curvature fit to the entire
            ### region of the cell borders not in contact with another cell:
            whole_curvature_details_red, all_fits_red_whole = fit_curves_to_bins([zip(*np.where(curveborders[:,:,0]))], curveborders[:,:,0], odr_circle_fit_iterations)
            whole_curvature_details_grn, all_fits_grn_whole = fit_curves_to_bins([zip(*np.where(curveborders[:,:,1]))], curveborders[:,:,1], odr_circle_fit_iterations)

            ### Record all the pixel_by_pixel curvature information you have: 
            mean_curvature_curveborders_red = np.mean(masked[list(zip(*ordered_border_indices_red))])
            median_curvature_curveborders_red = np.median(masked[list(zip(*ordered_border_indices_red))])
            sem_curvature_curveborders_red = np.std(masked[list(zip(*ordered_border_indices_red))]) / len(np.where(masked[list(zip(*ordered_border_indices_red))])[0])
            std_curvature_curveborders_red = np.std(masked[list(zip(*ordered_border_indices_red))])

            ### Define some more default values:
            area_cell_1 = len(np.where(cleanareas==1)[0])
            area_cell_2 = len(np.where(cleanareas==2)[0])
            noncontact_perimeter_cell1 = perimeter_cell1
            noncontact_perimeter_cell2 = perimeter_cell2
            noncontactborders = np.copy(curveborders)

            degree_angle1 = 0.0
            degree_angle2 = 0.0
            average_angle = 0.0
            red_halfangle1 = 0.0
            red_halfangle2 = 0.0
            grn_halfangle1 = 0.0
            grn_halfangle2 = 0.0
            contactsurface_len = 0.0

            ### Update the data table:
            listupdate()
            
            ### Move on to the next picture:
            continue


        ### If the cells ARE in contact, then proceed to calculate adhesiveness
        ### parameters:
        if cells_touching == True and skip == 'n':
            
            ### Clean up the contacts:
            contactcorners, contactsurface, noncontactborders = clean_up_contact_borders(rawcontactsurface, cleanborders)
            
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
                       "\nStructuring element size:", Bc3D.shape
                       )
                listupdate()
                continue

            ### Now that we have defined the corners of the contact, the contact
            ### surface, and the parts of the cell border that are not in
            ### contact, we can find the contact angles:
            temp = find_contact_angles(contactcorners, contactsurface, noncontactborders)
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
            
            n_contactsurface_px = int(len(np.where(contactsurface)[0])/2)
            cx,cy = np.linspace(rawcontact_vector_start[1], rawcontact_vector_end[1], num=n_contactsurface_px), np.linspace(rawcontact_vector_start[0], rawcontact_vector_end[0], num=n_contactsurface_px)
            imgform_contact_points = zip(cy,cx)
        
        
            ### Determine the half-angle for each of the red and green cells:
            red_halfangle1_rad = math.acos( np.round( ((np.dot(vector1,contact_vector)) / (((vector1[0]**2 + vector1[1]**2)**0.5)*((contact_vector[0]**2 + contact_vector[1]**2)**0.5)) ), decimals=10) )
            red_halfangle2_rad = math.acos( np.round( ((np.dot(vector3,contact_vector)) / (((vector3[0]**2 + vector3[1]**2)**0.5)*((contact_vector[0]**2 + contact_vector[1]**2)**0.5)) ), decimals=10) )
            grn_halfangle1_rad = math.acos( np.round( ((np.dot(vector2,contact_vector)) / (((vector2[0]**2 + vector2[1]**2)**0.5)*((contact_vector[0]**2 + contact_vector[1]**2)**0.5)) ), decimals=10) )
            grn_halfangle2_rad = math.acos( np.round( ((np.dot(vector4,contact_vector)) / (((vector4[0]**2 + vector4[1]**2)**0.5)*((contact_vector[0]**2 + contact_vector[1]**2)**0.5)) ), decimals=10) )
            ### and convert to degrees:
            red_halfangle1 = math.degrees(red_halfangle1_rad)
            red_halfangle2 = 180-math.degrees(red_halfangle2_rad)
            grn_halfangle1 = math.degrees(grn_halfangle1_rad)
            grn_halfangle2 = 180-math.degrees(grn_halfangle2_rad)


            ### Count the length of the contact surface in pixel lengths
#            contactsurface_len, contactsurface_len1, contactsurface_len2 = find_contactsurface_len(contactsurface)
#            print "The length of the contact surface is %.3f pixels" %contactsurface_len   

            ### Find the mean fluorescence intensity value that the red and grn
            ### linescans of the contact intersect at as an indirect measure of
            ### how close the 2 cells are:
            intersectsx, intersectsy, intersection_values_per_cell, intersection_values_per_cell_middle_list_index, intersection_cellpair_num, red_gap_scanx, red_gap_scany, grn_gap_scanx, grn_gap_scany = perform_linescan(contactsurface, perpendicular2contact_vector, linescan_length)

            ### Find the perimeter of the red and green cells:
            ordered_border_indices_red, ordered_border_indices_grn, numbered_border_indices_red, numbered_border_indices_grn, perimeter_cell1, perimeter_cell2 = count_pixels_around_borders_sequentially(curveborders)

            ### Give the noncontactborders sequential information:
            reordered_border_indices_red, reordered_border_indices_grn, noncontact_perimeter_cell1, noncontact_perimeter_cell2 = make_open_border_indices_sequential(noncontactborders+pathfinder)


            renumbered_border_indices_red = [p for p in enumerate(reordered_border_indices_red)]
            renumbered_border_indices_grn = [p for p in enumerate(reordered_border_indices_grn)]

            ### Bin the noncontactborders into 7 bins:
            num_bins = num_bins-1
            bin_size_list_red = create_border_bin_sizes((noncontactborders[:,:,0]+pathfinder[:,:,0]).astype(bool), reordered_border_indices_red, num_bins)
            bin_size_list_grn = create_border_bin_sizes((noncontactborders[:,:,1]+pathfinder[:,:,1]).astype(bool), reordered_border_indices_grn, num_bins)

            ### Make the 8th bin the contactsurface:
#            contact_bin_indices_red = zip(*np.where(contactsurface==1))
#            contact_bin_indices_grn = zip(*np.where(contactsurface==2))
            reordered_contact_surface_red, reordered_contact_surface_grn, contactsurface_len1, contactsurface_len2 = make_open_border_indices_sequential(contactsurface)
            contactsurface_len = (contactsurface_len1 + contactsurface_len2) / 2.0
            print("The length of the contact surface is %.3f pixels" %contactsurface_len)

            ### Return the bin-sorted indices for each pixel:
#            bin_indices_red, bin_start_red, bin_stop_red = sort_pixels_into_bins(reordered_border_indices_red, bin_size_list_red, contact_bin_indices_red, num_bins)
#            bin_indices_grn, bin_start_grn, bin_stop_grn = sort_pixels_into_bins(reordered_border_indices_grn, bin_size_list_grn, contact_bin_indices_grn, num_bins)
            bin_indices_red, bin_start_red, bin_stop_red = sort_pixels_into_bins(reordered_border_indices_red, bin_size_list_red, reordered_contact_surface_red, num_bins)
            bin_indices_grn, bin_start_grn, bin_stop_grn = sort_pixels_into_bins(reordered_border_indices_grn, bin_size_list_grn, reordered_contact_surface_grn, num_bins)

            bin_start_red_indices = np.array(reordered_border_indices_red)[bin_start_red]            
            bin_start_red_indices = np.row_stack((bin_start_red_indices, reordered_contact_surface_red[0]))
            bin_stop_red_indices = np.array(reordered_border_indices_red)[np.array(bin_stop_red)-1]
            bin_stop_red_indices = np.row_stack((bin_stop_red_indices, reordered_contact_surface_red[-1]))

            bin_start_grn_indices = np.array(reordered_border_indices_grn)[bin_start_grn]            
            bin_start_grn_indices = np.row_stack((bin_start_grn_indices, reordered_contact_surface_grn[0]))
            bin_stop_grn_indices = np.array(reordered_border_indices_grn)[np.array(bin_stop_grn)-1]
            bin_stop_grn_indices = np.row_stack((bin_stop_grn_indices, reordered_contact_surface_grn[-1]))



            ### Add the len of the 8th bin to the bin_size_list_red (and grn):
#            bin_size_list_red.append(len(bin_indices_red[7]))
#            bin_size_list_grn.append(len(bin_indices_grn[7]))

            bin_start_red.append(bin_indices_red[7][0])
            bin_stop_red.append(bin_indices_red[7][-1])
            bin_start_grn.append(bin_indices_grn[7][0])
            bin_stop_grn.append(bin_indices_grn[7][-1])

            ### Find the binned curvatures:
            bin_curvature_details_red, all_fits_red_binned = fit_curves_to_bins(bin_indices_red, curveborders[:,:,0], odr_circle_fit_iterations)
            bin_curvature_details_grn, all_fits_grn_binned = fit_curves_to_bins(bin_indices_grn, curveborders[:,:,1], odr_circle_fit_iterations)
                 
            ### While we're at it, also give me the curvature fit to the entire
            ### region of the cell borders not in contact with another cell:
            whole_curvature_details_red, all_fits_red_whole = fit_curves_to_bins([zip(*np.where((noncontactborders[:,:,0]+pathfinder[:,:,0]).astype(bool)))], (noncontactborders[:,:,0]+pathfinder[:,:,0]).astype(bool), odr_circle_fit_iterations)
            whole_curvature_details_grn, all_fits_grn_whole = fit_curves_to_bins([zip(*np.where((noncontactborders[:,:,1]+pathfinder[:,:,1]).astype(bool)))], (noncontactborders[:,:,1]+pathfinder[:,:,1]).astype(bool), odr_circle_fit_iterations)

            ### Define some more default values:
            area_cell_1 = len(np.where(cleanareas==1)[0])
            area_cell_2 = len(np.where(cleanareas==2)[0])



            ### Record all the pixel_by_pixel curvature information:
            curvature_pathfinder_red = masked[np.where(pathfinder==1)]
            curvature_pathfinder_grn = masked[np.where(pathfinder==2)]
            curvature_pathfinder_angle1 = masked[np.where(angles_angle_labeled == 1)]
            curvature_pathfinder_angle2 = masked[np.where(angles_angle_labeled == 2)]
            all_curvatures_pathfinder = list(curvature_pathfinder_red)+list(curvature_pathfinder_grn)
            all_curvatures_pathfinder_grn_angle1 = masked[np.where(angles_angle_labeled[:,:,1] == 1)]
            all_curvatures_pathfinder_grn_angle1 = all_curvatures_pathfinder_grn_angle1[:,1]
            all_curvatures_pathfinder_grn_angle2 = masked[np.where(angles_angle_labeled[:,:,1] == 2)]
            all_curvatures_pathfinder_grn_angle2 = all_curvatures_pathfinder_grn_angle2[:,1]
            all_curvatures_pathfinder_red_angle1 = masked[np.where(angles_angle_labeled[:,:,0] == 1)]
            all_curvatures_pathfinder_red_angle1 = all_curvatures_pathfinder_red_angle1[:,0]
            all_curvatures_pathfinder_red_angle2 = masked[np.where(angles_angle_labeled[:,:,0] == 2)]
            all_curvatures_pathfinder_red_angle2 = all_curvatures_pathfinder_red_angle2[:,0]
            curvature_contactsurface_red = masked[np.where(contactsurface == 1)]
            curvature_contactsurface_grn = masked[np.where(contactsurface == 2)]

            mean_curvature_curveborders_red = np.mean(masked[list(zip(*ordered_border_indices_red))])
            median_curvature_curveborders_red = np.median(masked[list(zip(*ordered_border_indices_red))])
            sem_curvature_curveborders_red = np.std(masked[list(zip(*ordered_border_indices_red))]) / len(np.where(masked[list(zip(*ordered_border_indices_red))])[0])
            std_curvature_curveborders_red = np.std(masked[list(zip(*ordered_border_indices_red))])
            
            all_curvatures_noncontactborders_red = masked[list(zip(*reordered_border_indices_red))]
            mean_curvature_noncontactborders_red = np.mean(masked[list(zip(*reordered_border_indices_red))])
            median_curvature_noncontactborders_red = np.median(masked[list(zip(*reordered_border_indices_red))])
            sem_curvature_noncontactborders_red = np.std(masked[list(zip(*reordered_border_indices_red))]) / len(np.where(masked[list(zip(*ordered_border_indices_red))])[0])
            std_curvature_noncontactborders_red = np.std(masked[list(zip(*reordered_border_indices_red))])
            
            mean_curvature_pathfinder_red = np.mean(curvature_pathfinder_red)
            median_curvature_pathfinder_red = np.median(curvature_pathfinder_red)
            sem_curvature_pathfinder_red = np.std(curvature_pathfinder_red) / len(np.where(curvature_pathfinder_red)[0])
            std_curvature_pathfinder_red = np.std(curvature_pathfinder_red)
            
            mean_curvature_contactsurface_red = np.mean(curvature_contactsurface_red)
            median_curvature_contactsurface_red = np.median(curvature_contactsurface_red)
            sem_curvature_contactsurface_red = np.std(curvature_contactsurface_red) / len(np.where(curvature_contactsurface_red)[0])
            std_curvature_contactsurface_red = np.std(curvature_contactsurface_red)
            
            mean_curvature_curveborders_grn = np.mean(masked[list(zip(*ordered_border_indices_grn))])
            median_curvature_curveborders_grn = np.median(masked[list(zip(*ordered_border_indices_grn))])
            sem_curvature_curveborders_grn = np.std(masked[list(zip(*ordered_border_indices_grn))]) / len(np.where(masked[list(zip(*ordered_border_indices_grn))])[0])
            std_curvature_curveborders_grn = np.std(masked[list(zip(*ordered_border_indices_grn))])
            
            all_curvatures_noncontactborders_grn = masked[list(zip(*reordered_border_indices_grn))]
            mean_curvature_noncontactborders_grn = np.mean(masked[list(zip(*reordered_border_indices_grn))])
            median_curvature_noncontactborders_grn = np.median(masked[list(zip(*reordered_border_indices_grn))])
            sem_curvature_noncontactborders_grn = np.std(masked[list(zip(*reordered_border_indices_grn))]) / len(np.where(masked[list(zip(*ordered_border_indices_grn))])[0])
            std_curvature_noncontactborders_grn = np.std(masked[list(zip(*reordered_border_indices_grn))])
            
            mean_curvature_pathfinder_grn = np.mean(curvature_pathfinder_grn)
            median_curvature_pathfinder_grn = np.median(curvature_pathfinder_grn)
            sem_curvature_pathfinder_grn = np.std(curvature_pathfinder_grn) / len(np.where(curvature_pathfinder_grn)[0])
            std_curvature_pathfinder_grn = np.std(curvature_pathfinder_grn)
            
            mean_curvature_contactsurface_grn = np.mean(curvature_contactsurface_grn)
            median_curvature_contactsurface_grn = np.median(curvature_contactsurface_grn)
            sem_curvature_contactsurface_grn = np.std(curvature_contactsurface_grn) / len(np.where(curvature_contactsurface_grn)[0])
            std_curvature_contactsurface_grn = np.std(curvature_contactsurface_grn)
            
            mean_curvature_pathfinder = np.mean(all_curvatures_pathfinder)
            median_curvature_pathfinder = np.median(all_curvatures_pathfinder)
            sem_curvature_pathfinder = np.std(all_curvatures_pathfinder) / len(np.where(all_curvatures_pathfinder)[0])
            std_curvature_pathfinder = np.std(all_curvatures_pathfinder)
            
            mean_curvature_pathfinder_angle1 = np.mean(curvature_pathfinder_angle1)
            median_curvature_pathfinder_angle1 = np.median(curvature_pathfinder_angle1)
            sem_curvature_pathfinder_angle1 = np.std(curvature_pathfinder_angle1) / len(np.where(curvature_pathfinder_angle1)[0])
            std_curvature_pathfinder_angle1 = np.std(curvature_pathfinder_angle1)
            
            mean_curvature_pathfinder_angle2 = np.mean(curvature_pathfinder_angle2)
            median_curvature_pathfinder_angle2 = np.median(curvature_pathfinder_angle2)
            sem_curvature_pathfinder_angle2 = np.std(curvature_pathfinder_angle2) / len(np.where(curvature_pathfinder_angle2)[0])
            std_curvature_pathfinder_angle2 = np.std(curvature_pathfinder_angle2)
            
            mean_curvature_pathfinder_grn_angle1 = np.mean(all_curvatures_pathfinder_grn_angle1)
            median_curvature_pathfinder_grn_angle1 = np.median(all_curvatures_pathfinder_grn_angle1)
            sem_curvature_pathfinder_grn_angle1 = np.std(all_curvatures_pathfinder_grn_angle1) / len(np.where(all_curvatures_pathfinder_grn_angle1)[0])
            std_curvature_pathfinder_grn_angle1 = np.std(all_curvatures_pathfinder_grn_angle1)
            
            mean_curvature_pathfinder_grn_angle2 = np.mean(all_curvatures_pathfinder_grn_angle2)
            median_curvature_pathfinder_grn_angle2 = np.median(all_curvatures_pathfinder_grn_angle2)
            sem_curvature_pathfinder_grn_angle2 = np.std(all_curvatures_pathfinder_grn_angle2) / len(np.where(all_curvatures_pathfinder_grn_angle2)[0])
            std_curvature_pathfinder_grn_angle2 = np.std(all_curvatures_pathfinder_grn_angle2)
            
            mean_curvature_pathfinder_red_angle1 = np.mean(all_curvatures_pathfinder_red_angle1)
            median_curvature_pathfinder_red_angle1 = np.median(all_curvatures_pathfinder_red_angle1)
            sem_curvature_pathfinder_red_angle1 = np.std(all_curvatures_pathfinder_red_angle1) / len(np.where(all_curvatures_pathfinder_red_angle1)[0])
            std_curvature_pathfinder_red_angle1 = np.std(all_curvatures_pathfinder_red_angle1)
            
            mean_curvature_pathfinder_red_angle2 = np.mean(all_curvatures_pathfinder_red_angle2)
            median_curvature_pathfinder_red_angle2 = np.median(all_curvatures_pathfinder_red_angle2)
            sem_curvature_pathfinder_red_angle2 = np.std(all_curvatures_pathfinder_red_angle2) / len(np.where(all_curvatures_pathfinder_red_angle2)[0])
            std_curvature_pathfinder_red_angle2 = np.std(all_curvatures_pathfinder_red_angle2)



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
            contact_surface_line_grn_rcd = [(vector2_start[1], vector4_start[1]), (vector2_start[0], vector4_start[0])]
            ### draw a line that is perpendicular to the contact surface line, in the middle
            ### of the contact surface:
            temp1_rcd = (vector1_start[::-1] + vector2_start[::-1])/2.0
            temp2_rcd = (vector3_start[::-1] + vector4_start[::-1])/2.0
            mean_contact_surface_line_rcd = (temp1_rcd + temp2_rcd)/2.0
            center_of_contact_surface_lines_rc = mean_contact_surface_line_rcd[:2]

#            cleanoverlay = np.copy(padded_cell)
            cleanoverlay = np.copy(cell)
            cleanoverlay[:,:,0] = mh.stretch(cleanoverlay[:,:,0])
            cleanoverlay[:,:,1] = mh.stretch(cleanoverlay[:,:,1])
            cleanoverlay[:,:,2] = mh.stretch(cleanoverlay[:,:,2])
            cleanoverlay[np.where(curveborders)] = 255
            cleanoverlay = mh.overlay(np.max(cell, axis=2), red=curveborders[:,:,0], green=curveborders[:,:,1])
            ### Draw what you've found over the images of the cells:
            fig = plt.figure(num=str(filenumber+1)+'_'+os.path.basename(filename), figsize=(12., 10.))
            plt.suptitle(str(filenumber+1)+' - '+os.path.basename(filename), verticalalignment='center', x=0.5, y=0.5, fontsize=14)
            ax1 = fig.add_subplot(231)
            ax1.yaxis.set_visible(False)
            ax1.xaxis.set_visible(False)
            ax1.imshow(mh.stretch(cell_raw))
            
            ax2 = fig.add_subplot(232)
            ax2.yaxis.set_visible(False)
            ax2.xaxis.set_visible(False)
#            figcellborderred = ax2.imshow(mh.overlay(gray = justcellpair[:,:,0], red = cleanborders[:,:,0]))
#            figcellborderred = ax2.imshow(mh.overlay(gray = mh.stretch(padded_cell[:,:,0]), red = cleanborders[:,:,0]))
            figcellborderred = ax2.imshow(mh.overlay(gray = mh.stretch(cell[:,:,0]), red = cleanborders[:,:,0]))
            ax2.axes.autoscale(False)
            ax2.scatter(newcellmaxima_rcd[0][1], newcellmaxima_rcd[0][0], color='m', marker='o')
            
            ax3 = fig.add_subplot(233)
            ax3.yaxis.set_visible(False)
            ax3.xaxis.set_visible(False)
#            figcellbordergrn = ax3.imshow(mh.overlay(gray = justcellpair[:,:,1], green = cleanborders[:,:,1]))
#            figcellbordergrn = ax3.imshow(mh.overlay(gray = mh.stretch(padded_cell[:,:,1]), green = cleanborders[:,:,1]))
            figcellbordergrn = ax3.imshow(mh.overlay(gray = mh.stretch(cell[:,:,1]), green = cleanborders[:,:,1]))
            ax3.axes.autoscale(False)
            ax3.scatter(newcellmaxima_rcd[1][1], newcellmaxima_rcd[1][0], color='m', marker='o')
            
            ax4 = fig.add_subplot(235)
            ax4.yaxis.set_visible(False)
            ax4.xaxis.set_visible(False)
            ax4.imshow(justcellpair)
#            ax4.imshow(mh.stretch(cell))
            ax4.axes.autoscale(False)
            drawn_arrows = ax4.quiver(vector_start_x_vals, vector_start_y_vals, vector_x_vals, vector_y_vals, units='xy',scale_units='xy', angles='xy', scale=0.33, color='c', width = 1.0)
            angle1_annotations = ax4.annotate('%.1f' % degree_angle1, vector1_start, vector1_start + 30*(ang1_annot / ang1_annot_mag)*(1,1), color='c', fontsize='24', ha='center', va='center')
            angle2_annotations = ax4.annotate('%.1f' % degree_angle2, vector3_start, vector3_start + 30*(ang2_annot / ang2_annot_mag)*(1,1), color='c', fontsize='24', ha='center', va='center')
            redcell_contactvect = ax4.plot(contact_surface_line_red_rcd[1], contact_surface_line_red_rcd[0], color='c', lw=1)
            grncell_contactvect = ax4.plot(contact_surface_line_grn_rcd[1], contact_surface_line_grn_rcd[0], color='c', lw=1)
            contactvect = ax4.plot([mean_contact_surface_line_rcd[1]-2*contact_vector[0], mean_contact_surface_line_rcd[1]+2*contact_vector[0]], [mean_contact_surface_line_rcd[0]-2*contact_vector[1], mean_contact_surface_line_rcd[0]+2*contact_vector[1]], color='m')
            
            ax5 = fig.add_subplot(236)
            ax5.yaxis.set_visible(False)
            ax5.xaxis.set_visible(False)
            ax5.imshow( (~(~justcellpair*~contactsurface.astype(np.bool))) )
#            ax5.imshow( (~(~mh.stretch(cell)*~contactsurface.astype(np.bool))) )
            ax5.axes.autoscale(False)
    
            ax6 = fig.add_subplot(234)
            ax6.yaxis.set_visible(False)
            ax6.xaxis.set_visible(False)
            ax6.imshow(cleanoverlay)
            ax6.axes.autoscale(False)
#            contactvect2 = ax6.plot([mean_contact_surface_line_rcd[1]-2*contact_vector[0], mean_contact_surface_line_rcd[1]+2*contact_vector[0]], [mean_contact_surface_line_rcd[0]-2*contact_vector[1], mean_contact_surface_line_rcd[0]+2*contact_vector[1]], color='m')
#            perp2contactvect = ax6.plot([center_of_contact_surface_lines_rc[1]-perpendicular2contact_vector[0], center_of_contact_surface_lines_rc[1]+perpendicular2contact_vector[0]], [center_of_contact_surface_lines_rc[0]-perpendicular2contact_vector[1], center_of_contact_surface_lines_rc[0]+perpendicular2contact_vector[1]], color='b')
            
            plt.tight_layout()
            plt.draw
            #plt.show('block' == False)
    
            listupdate()




    ### Everything in the following elif block is almost identical to the process
    ### from the if filenumber == 1 block above (ie. it will just repeat the above
    ### process for all files in a folder).
    elif filenumber > 0: 
        start_time = time.perf_counter()

#    ### Tell me filenumber and filename of what you're currently working on!
#        print '\n',(filenumber+1), filename  
        cell_raw = Image.open(filename)
        ### You should note that PIl will automatically close the file when it's not needed
        cell_raw = np.array(cell_raw)

    ### Check if the signal:noise ratio is too low in any 1 channel. If it is
    ### then set that channel to be black:
        cell = check_for_signal(cell_raw)
        
        ### Split the file up in to it's RGB 3-colour-based arrays (ie. 3 channels):
        redchan = mh.stretch(cell[:,:,0])
        grnchan = mh.stretch(cell[:,:,1])
        bluchan = mh.stretch(cell[:,:,2])
        
    ###############################################################################
        
        ### Define some default values:
        #sobel = 'Y'
        #sobel_filt = 'True'
        #gauss = 'Y'
        #gaussian_filter_value_red = 2.0
        #gaussian_filter_value_grn = 2.0
        #gaussian_filter_value_blu = 2.0
        #Otsu_Thresholding = 'Y'
        #red_otsu_fraction = 0.75#0.5#0.7#0.9#1.0
        #grn_otsu_fraction = 0.75#0.5#0.9#1.0
        #blu_otsu_fraction = 0.75#0.5#0.9#1.0
#        odr_circle_fit_iterations = 30
        num_bins = 8
        skip = 'n'
    
        # Define the red and green brightness cutoffs for objects of interest:
        if object_of_interest_brightness_cutoff_percent == 'otsu':
            if redchan.any():
                object_of_interest_brightness_cutoff_red = filters.threshold_otsu(redchan) * otsu_fraction
            else:
                object_of_interest_brightness_cutoff_red = background_threshold 
            if grnchan.any():
                object_of_interest_brightness_cutoff_grn = filters.threshold_otsu(grnchan) * otsu_fraction
            else:
                object_of_interest_brightness_cutoff_grn = background_threshold

###
        ### The roundness_thresh determines how round a region has to be to be
        ### recognized as a cell:
#        roundness_thresh = 1.25
        
        ### The size_filter determines how many px^2 the area of a region has
        ### to be in order to be recognized as a cell:
#        size_filter = 500
        
        ### Create a maxima-specific structuring element to try and reduce the
        ### problem of watershed oversegmentation: was 24, 24 before Aug. 24 2017
        ### problem of undersegmentation: was 48,48 before Feb. 02, 2018
#        maxima_Bc = np.ones((36,36))
###

    ### Define the default values for the the data that will populate the table
    ### that will be output at the end:
        red_gap_scanmean = np.asarray([float('nan')]*linescan_length)
        grn_gap_scanmean = np.asarray([float('nan')]*linescan_length)
        red_gap_scanx = [float('nan')]
        red_gap_scany = [float('nan')]
        grn_gap_scanx = [float('nan')]
        grn_gap_scany = [float('nan')]
        degree_angle1 = float('nan')
        degree_angle2 = float('nan')
        average_angle = float('nan')
        contactsurface_len = float('nan')
        noncontactsurface_len1 = float('nan')
        noncontactsurface_len2 = float('nan')
        area_cell_1 = float('nan')
        area_cell_2 = float('nan')
        perimeter_cell1 = float('nan')
        perimeter_cell2 = float('nan')
        noncontact_perimeter_cell1 = float('nan')
        noncontact_perimeter_cell2 = float('nan')
        intersectsx = float('nan')
        intersectsy = float('nan')
        intersection_points = [(float('nan'), float('nan'))]
        intersection_values_per_cell = [float('nan')]
        intersection_values_per_cell_middle_list_index = int(0)
        intersection_cellpair_num = filenumber+1
        cleanborders = np.zeros(cell.shape)
        contactsurface = np.zeros(cell.shape)
        noncontactborders = np.zeros(cell.shape)
        curveborders_curvature_labeled = np.zeros(cell.shape)
        pathfinder = np.zeros(cell.shape)
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
        grn_halfangle1_rad = float('nan')
        grn_halfangle2_rad = float('nan')
        red_halfangle1 = float('nan')
        red_halfangle2 = float('nan')
        grn_halfangle1 = float('nan')
        grn_halfangle2 = float('nan')
        contactsurface_len1 = float('nan')
        contactsurface_len2 = float('nan')
        numbered_border_indices_red = [float('nan'), (float('nan'),float('nan'),float('nan'))]
        numbered_border_indices_grn = [float('nan'), (float('nan'),float('nan'),float('nan'))]
        numbered_noncontactborder_indices_red = [float('nan'), (float('nan'),float('nan'),float('nan'))]
        numbered_noncontactborder_indices_grn = [float('nan'), (float('nan'),float('nan'),float('nan'))]
        
        angles_angle_labeled = np.zeros(cell.shape)
        
        mean_curvature_curveborders_red = float('nan')
        median_curvature_curveborders_red = float('nan')
        sem_curvature_curveborders_red = float('nan')
        std_curvature_curveborders_red = float('nan')
        
        all_curvatures_noncontactborders_red = [float('nan')]
        mean_curvature_noncontactborders_red = float('nan')
        median_curvature_noncontactborders_red = float('nan')
        sem_curvature_noncontactborders_red = float('nan')
        std_curvature_noncontactborders_red = float('nan')
        
        curvature_pathfinder_red = [float('nan')]
        mean_curvature_pathfinder_red = float('nan')
        median_curvature_pathfinder_red = float('nan')
        sem_curvature_pathfinder_red = float('nan')
        std_curvature_pathfinder_red = float('nan')
        
        curvature_contactsurface_red = [float('nan')]
        mean_curvature_contactsurface_red = float('nan')
        median_curvature_contactsurface_red = float('nan')
        sem_curvature_contactsurface_red = float('nan')
        std_curvature_contactsurface_red = float('nan')
        
        
        mean_curvature_curveborders_grn = float('nan')
        median_curvature_curveborders_grn = float('nan')
        sem_curvature_curveborders_grn = float('nan')
        std_curvature_curveborders_grn = float('nan')
        
        all_curvatures_noncontactborders_grn = [float('nan')]
        mean_curvature_noncontactborders_grn = float('nan')
        median_curvature_noncontactborders_grn = float('nan')
        sem_curvature_noncontactborders_grn = float('nan')
        std_curvature_noncontactborders_grn = float('nan')
        
        curvature_pathfinder_grn = [float('nan')]
        mean_curvature_pathfinder_grn = float('nan')
        median_curvature_pathfinder_grn = float('nan')
        sem_curvature_pathfinder_grn = float('nan')
        std_curvature_pathfinder_grn = float('nan')
        
        curvature_contactsurface_grn = [float('nan')]
        mean_curvature_contactsurface_grn = float('nan')
        median_curvature_contactsurface_grn = float('nan')
        sem_curvature_contactsurface_grn = float('nan')
        std_curvature_contactsurface_grn = float('nan')
        
        all_curvatures_pathfinder = [float('nan')]
        mean_curvature_pathfinder = float('nan')
        median_curvature_pathfinder = float('nan')
        sem_curvature_pathfinder = float('nan')
        std_curvature_pathfinder = float('nan')
        
        curvature_pathfinder_angle1 = [float('nan')]
        mean_curvature_pathfinder_angle1 = float('nan')
        median_curvature_pathfinder_angle1 = float('nan')
        sem_curvature_pathfinder_angle1 = float('nan')
        std_curvature_pathfinder_angle1 = float('nan')
        
        curvature_pathfinder_angle2 = [float('nan')]
        mean_curvature_pathfinder_angle2 = float('nan')
        median_curvature_pathfinder_angle2 = float('nan')
        sem_curvature_pathfinder_angle2 = float('nan')
        std_curvature_pathfinder_angle2 = float('nan')
        
        all_curvatures_pathfinder_grn_angle1 = [float('nan')]
        mean_curvature_pathfinder_grn_angle1 = float('nan')
        median_curvature_pathfinder_grn_angle1 = float('nan')
        sem_curvature_pathfinder_grn_angle1 = float('nan')
        std_curvature_pathfinder_grn_angle1 = float('nan')
        
        all_curvatures_pathfinder_grn_angle2 = [float('nan')]
        mean_curvature_pathfinder_grn_angle2 = float('nan')
        median_curvature_pathfinder_grn_angle2 = float('nan')
        sem_curvature_pathfinder_grn_angle2 = float('nan')
        std_curvature_pathfinder_grn_angle2 = float('nan')
        
        all_curvatures_pathfinder_red_angle1 = [float('nan')]
        mean_curvature_pathfinder_red_angle1 = float('nan')
        median_curvature_pathfinder_red_angle1 = float('nan')
        sem_curvature_pathfinder_red_angle1 = float('nan')
        std_curvature_pathfinder_red_angle1 = float('nan')
        
        all_curvatures_pathfinder_red_angle2 = [float('nan')]
        mean_curvature_pathfinder_red_angle2 = float('nan')
        median_curvature_pathfinder_red_angle2 = float('nan')
        sem_curvature_pathfinder_red_angle2 = float('nan')
        std_curvature_pathfinder_red_angle2 = float('nan')

        bin_curvature_details_red = [[float('nan')] * len(bin_curvature_details_headers)] * (num_bins+1)
        bin_curvature_details_grn = [[float('nan')] * len(bin_curvature_details_headers)] * (num_bins+1)

        whole_curvature_details_red = [[float('nan')] * len(bin_curvature_details_headers)] * 2
        whole_curvature_details_grn = [[float('nan')] * len(bin_curvature_details_headers)] * 2

        bin_indices_red = [[(float('nan'), float('nan'), float('nan'))]] * num_bins
        bin_indices_grn = [[(float('nan'), float('nan'), float('nan'))]] * num_bins

#        bin_start_red = [float('nan')] * num_bins
#        bin_stop_red = [float('nan')] * num_bins
        
#        bin_start_grn = [float('nan')] * num_bins
#        bin_stop_grn = [float('nan')] * num_bins

        bin_start_red_indices = np.empty((num_bins, 3))
        bin_start_red_indices[:] = np.nan

        bin_stop_red_indices = np.empty((num_bins, 3))
        bin_stop_red_indices[:] = np.nan

        bin_start_grn_indices = np.empty((num_bins, 3))
        bin_start_grn_indices[:] = np.nan

        bin_stop_grn_indices = np.empty((num_bins, 3))
        bin_stop_grn_indices[:] = np.nan


        ### Segment your cells:
        thresh_based_areas, wscan_based_areas_out, wscan_based_areas_in, wscan_based_maxima, Bc3D, edge_map = watershed_on_cannyweighted_sobel_based_segment_and_isolate_center_cells(cell, gaussian_filter_value_red, gaussian_filter_value_grn, gaussian_filter_value_blu, maxima_Bc, background_threshold, object_of_interest_brightness_cutoff_red, object_of_interest_brightness_cutoff_grn)

        justcellpair = np.copy(cell)
        justcellpair[np.where(wscan_based_areas_out)] = 0
        justcellpair[:,:,0] = mh.stretch(justcellpair[:,:,0])
        justcellpair[:,:,1] = mh.stretch(justcellpair[:,:,1])
        justcellpair[:,:,2] = mh.stretch(justcellpair[:,:,2])

        # Make a copy of the cell maxima that uses their current labeling scheme:
        new_wscan_based_maxima = wscan_based_areas_in.astype(bool) * wscan_based_areas_in
        # If any label in spots exist as more than 1 pixel, then use the mean
        # location of that label:
        new_wscan_based_maxima_locs = []
        new_wscan_based_maxima_locs.extend([np.append((np.mean(np.where(new_wscan_based_maxima[:,:,0] == i), axis=1)).astype(int), 0) for i in np.unique(new_wscan_based_maxima[:,:,0])[1:]])
        new_wscan_based_maxima_locs.extend([np.append((np.mean(np.where(new_wscan_based_maxima[:,:,1] == i), axis=1)).astype(int), 1) for i in np.unique(new_wscan_based_maxima[:,:,1])[1:]])
        # and assign that pixel as the new spot in a new array:
        new_wscan_based_maxima_means = np.zeros(wscan_based_maxima.shape, dtype=int)
        new_wscan_based_maxima_means[tuple(np.asarray(new_wscan_based_maxima_locs).T)] = new_wscan_based_maxima[tuple(np.asarray(new_wscan_based_maxima_locs).T)]
        

        
        ### If more than 1 cell is detected per channel then try to use the cell
        ### that is spatially closest to the previously detected one:
        if len(np.unique(wscan_based_areas_in[:,:,0])) > 2:
            # Find which spot is closer to the most recent spots detected:
            closest_cell_max_red = nearest_cell_maxima(red_maxima_coords, np.where(new_wscan_based_maxima_means[:,:,0]))
            # If you find a spot that is closer (which you should)...
            if np.array(closest_cell_max_red).any():
                # Then find what the spots label is...
                val = new_wscan_based_maxima_means[closest_cell_max_red[0], closest_cell_max_red[1], 0]
                # and reset the red channel of new_wscan_based_maxima_means
                # so that only the closest spot is kept:
                new_wscan_based_maxima_means[:,:,0] = np.zeros(new_wscan_based_maxima_means[:,:,0].shape, dtype=int)
                new_wscan_based_maxima_means[closest_cell_max_red[0], closest_cell_max_red[1], 0] = val
                ### Then remove the regions that aren't away from the 
                # wscan_based_areas_in:
                remove_region = [np.logical_not(i in np.unique(new_wscan_based_maxima_means[:,:,0])) for i in np.unique(wscan_based_areas_in[:,:,0])]
                wscan_based_areas_in[:,:,0] = mh.labeled.remove_regions(wscan_based_areas_in[:,:,0], np.unique(wscan_based_areas_in[:,:,0])[np.where(remove_region)])
                # and fill in any tiny holes that might be created from this:
                if np.unique(wscan_based_areas_in[wscan_based_areas_in!=0]).any():
                    wscan_based_areas_in[:,:,0] = np.add.reduce([morphology.remove_small_holes(wscan_based_areas_in[:,:,0]==lab) for lab in np.unique(wscan_based_areas_in[wscan_based_areas_in!=0])])
                else:
                    wscan_based_areas_in[:,:,0] = np.zeros(wscan_based_areas_in[:,:,0].shape)
            else:
                sizes = mh.labeled.labeled_size(wscan_based_areas_in[:,:,0])
                iter = sizes
                n = 2
                secondlargestregion = nth_max(n, iter)
                too_small = np.where(sizes < secondlargestregion)
                wscan_based_areas_in[:,:,0] = mh.labeled.remove_regions(wscan_based_areas_in[:,:,0], too_small)
            
        if len(np.unique(wscan_based_areas_in[:,:,1])) > 2:
            closest_cell_max_grn = nearest_cell_maxima(grn_maxima_coords, np.where(new_wscan_based_maxima_means[:,:,1]))
            if np.array(closest_cell_max_grn).any():
                val = new_wscan_based_maxima_means[closest_cell_max_grn[0], closest_cell_max_grn[1], 1]
                new_wscan_based_maxima_means[:,:,1] = np.zeros(new_wscan_based_maxima_means[:,:,1].shape, dtype=int)
                new_wscan_based_maxima_means[closest_cell_max_grn[0], closest_cell_max_grn[1], 1] = val            
                ### Remove the regions that were further away:
                remove_region = [np.logical_not(i in np.unique(new_wscan_based_maxima_means[:,:,1])) for i in np.unique(wscan_based_areas_in[:,:,1])]
                wscan_based_areas_in[:,:,1] = mh.labeled.remove_regions(wscan_based_areas_in[:,:,1], np.unique(wscan_based_areas_in[:,:,1])[np.where(remove_region)])
                # and fill in any tiny holes that might be created from this:
                if np.unique(wscan_based_areas_in[wscan_based_areas_in!=0]).any():
                    wscan_based_areas_in[:,:,1] = np.add.reduce([morphology.remove_small_holes(wscan_based_areas_in[:,:,1]==lab) for lab in np.unique(wscan_based_areas_in[wscan_based_areas_in!=0])])
                else:
                    wscan_based_areas_in[:,:,1] = np.zeros(wscan_based_areas_in[:,:,1].shape)
            else:
                sizes = mh.labeled.labeled_size(wscan_based_areas_in[:,:,1])
                iter = sizes
                n = 2
                secondlargestregion = nth_max(n, iter)
                too_small = np.where(sizes < secondlargestregion)
                wscan_based_areas_in[:,:,1] = mh.labeled.remove_regions(wscan_based_areas_in[:,:,1], too_small)
                
        wscan_border_red = mh.borders(wscan_based_areas_in[:,:,0]) * wscan_based_areas_in[:,:,0]
        borders_red = mh.thin(wscan_border_red) * 1
        
        wscan_border_grn = mh.borders(wscan_based_areas_in[:,:,1]) * wscan_based_areas_in[:,:,1]

        borders_grn = mh.thin(wscan_border_grn) * 2

        wscan_border_blu = mh.borders(wscan_based_areas_in[:,:,2]) * wscan_based_areas_in[:,:,2]

        borders_blu = mh.thin(wscan_border_blu) * 3

        curveborders = np.dstack((borders_red, borders_grn, borders_blu))
        curveborders = clean_up_borders(curveborders)
        borders_red, borders_grn, borders_blu = np.squeeze(np.dsplit(curveborders, 3))
        semifinalflatcellareas = np.dstack(( scp_morph.binary_fill_holes(borders_red), scp_morph.binary_fill_holes(borders_grn), scp_morph.binary_fill_holes(borders_blu) ))

        borders_red = mh.borders(semifinalflatcellareas[:,:,0]) * semifinalflatcellareas[:,:,0]
        borders_grn = mh.borders(semifinalflatcellareas[:,:,1]) * semifinalflatcellareas[:,:,1]
        borders_blu = mh.borders(semifinalflatcellareas[:,:,2]) * semifinalflatcellareas[:,:,2]
        semifinalflatcellareas = np.dstack(( scp_morph.binary_fill_holes(borders_red), scp_morph.binary_fill_holes(borders_grn), scp_morph.binary_fill_holes(borders_blu) ))

        finalflatcellareas, cell_maxima = separate_red_and_green_cells(justcellpair, semifinalflatcellareas, maxima_Bc, edge_map)

        ### Clean up the segmented image of any regions (ie. "cells") that are too
        ### small or that are touching the borders:
        finalflatcellareas = remove_bordering_and_small_regions(finalflatcellareas)
     
###
        ### Make sure that the cells meet the minimum size threshold:
        size_red = mh.labeled.labeled_size(finalflatcellareas[:,:,0])
        size_grn = mh.labeled.labeled_size(finalflatcellareas[:,:,1])
        #size_blu = mh.labeled.labeled_size(finalflatcellareas[:,:,2])
        
        finalflatcellareas[:,:,0] = mh.labeled.remove_regions(finalflatcellareas[:,:,0], size_red < size_filter)
        finalflatcellareas[:,:,1] = mh.labeled.remove_regions(finalflatcellareas[:,:,1], size_grn < size_filter)
        
###
        ### Clean up the segmented image of any regions (ie. "cells") that are too
        ### small or that are touching the borders:
#        no_border_cells = remove_bordering_and_small_regions(cellareas)
###

        ### Need to do a second iteration of image thresholding and segmentation
        ### in a slightly different way on the array of the image of just the 
        ### cells in the middle to better resolve the edges of the cells:
#        finalflatcellareas = cellareas[:,:,0] + cellareas[:,:,1] + cellareas[:,:,2]
###

        ### Use the refined cell segmentations to create the cell borders:
#        newredarea = (finalflatcellareas == 1) * 1
        newredarea = finalflatcellareas[:,:,0] * 1
        newredcellborders = mh.borders(newredarea.astype(bool)) * newredarea
        newredcellborders = mh.thin(newredcellborders).astype(bool) * 1
#        newgrnarea = (finalflatcellareas == 2) * 2
        newgrnarea = finalflatcellareas[:,:,1] * 2
        newgrncellborders = mh.borders(newgrnarea.astype(bool)) * newgrnarea
        newgrncellborders = mh.thin(newgrncellborders).astype(bool) * 2
        newbluarea = np.zeros((finalflatcellareas[:,:,2].shape))
        newblucellborders = np.zeros((finalflatcellareas[:,:,2].shape))
        
        curveborders = np.dstack((newredcellborders, newgrncellborders, newblucellborders))
###

        ### Clean the borders used for curves so the algorithm used to count
        ### pixels and get sequential info from the borders doesn't get lost:
        curveborders = clean_up_borders(curveborders)
        
#        padded_cell = pad_img_edges(cell, 0)
#        cleanoverlay = np.copy(padded_cell)
        cleanoverlay = np.copy(cell)
        cleanoverlay[:,:,0] = mh.stretch(cleanoverlay[:,:,0])
        cleanoverlay[:,:,1] = mh.stretch(cleanoverlay[:,:,1])
        cleanoverlay[:,:,2] = mh.stretch(cleanoverlay[:,:,2])
        cleanoverlay[np.where(curveborders)] = 255
        cleanoverlay = mh.overlay(np.max(cell, axis=2), red=curveborders[:,:,0], green=curveborders[:,:,1])
    
        ### Return the indices of the curveborder pixels based on colour:
        curvecoordsred = list(zip(*np.where(curveborders[:,:,0])))
        curvecoordsred_name = 'curvecoordsred'
        curvecoordsgrn = list(zip(*np.where(curveborders[:,:,1])))
        curvecoordsgrn_name = 'curvecoordsgrn'
        curvecoordsblu = list(zip(*np.where(curveborders[:,:,2])))
        curvecoordsblu_name = 'curvecoordsblu'

        ### Use n pixels to the left and right of a given pixel to define a circle
        ### and therefore the curvature at that pixel:
        curve_definitions_red, curve_vals_red = pixel_by_pixel_curvatures(curveborders[:,:,0], curvecoordsred, curvecoordsred_name)
        curve_definitions_grn, curve_vals_grn = pixel_by_pixel_curvatures(curveborders[:,:,1], curvecoordsgrn, curvecoordsgrn_name)
        curve_definitions_blu, curve_vals_blu = pixel_by_pixel_curvatures(curveborders[:,:,2], curvecoordsblu, curvecoordsblu_name)

        ### Assign the curvature value of each pixel to it's location in curveborders:
        curveborders_curvature_labeled = np.zeros(curveborders.shape)
        for x in curve_vals_red:
            curveborders_curvature_labeled[x[1], x[0], 0] = (1.0/x[2]) * x[5]
        for x in curve_vals_grn:
            curveborders_curvature_labeled[x[1], x[0], 1] = (1.0/x[2]) * x[5]
        for x in curve_vals_blu:
            curveborders_curvature_labeled[x[1], x[0], 2] = (1.0/x[2]) * x[5]

        ### Mask the zeros that are not part of the borders:
        masked = np.ma.array(data = curveborders_curvature_labeled, mask = curveborders==0)



        ### Make a copy of the curveborders because we will need to clean them 
        ### up slightly differently for cells that are in contact:
        cleanborders = np.copy(curveborders)
        cleanareas = np.dstack((newredarea, newgrnarea, newbluarea))
        ### Isolate the watershed segmentations seed points so that you only have the 2
        ### seed points that detect the cells:
        newcellmaxima = np.copy(cell_maxima)
#        newcellmaxima = np.expand_dims(newcellspotsflat, axis=2)*cleanareas
        ### Return the seed point locations (in  (row,col,depth) format):
#        newcellmaxima_rcd = ([np.mean(np.where(newcellmaxima[:,:,0])[0]), np.mean(np.where(newcellmaxima[:,:,0])[1])], [np.mean(np.where(newcellmaxima[:,:,1])[0]), np.mean(np.where(newcellmaxima[:,:,1])[1])])
        newcellmaxima_rcd = ([np.where(newcellmaxima[:,:,0])[0], np.where(newcellmaxima[:,:,0])[1]], [np.where(newcellmaxima[:,:,1])[0], np.where(newcellmaxima[:,:,1])[1]])


#        cleanoverlay[:,:,0] = mh.stretch(np.copy(cell[:,:,0]))
#        cleanoverlay[:,:,1] = mh.stretch(np.copy(cell[:,:,1]))
#        cleanoverlay[:,:,2] = mh.stretch(np.copy(cell[:,:,2]))
        ### Plot the cells and any parameters so far:
        fig = plt.figure(num=str(filenumber+1)+'_'+os.path.basename(filename), figsize=(12., 10.))
        plt.suptitle(str(filenumber+1)+' - '+os.path.basename(filename), verticalalignment='center', x=0.5, y=0.5, fontsize=14)
        ax1 = fig.add_subplot(231)
        ax1.yaxis.set_visible(False)
        ax1.xaxis.set_visible(False)
        ax1.imshow(mh.stretch(cell_raw))
        
        ax2 = fig.add_subplot(232)
        ax2.yaxis.set_visible(False)
        ax2.xaxis.set_visible(False)
#        figcellborderred = ax2.imshow(mh.overlay(gray = justcellpair[:,:,0], red = cleanborders[:,:,0]))
#        figcellborderred = ax2.imshow(mh.overlay(gray = mh.stretch(padded_cell[:,:,0]), red = cleanborders[:,:,0]))
        figcellborderred = ax2.imshow(mh.overlay(gray = mh.stretch(cell[:,:,0]), red = cleanborders[:,:,0]))
        ax2.axes.autoscale(False)
        ax2.scatter(newcellmaxima_rcd[0][1], newcellmaxima_rcd[0][0], color='m', marker='o')
        
        ax3 = fig.add_subplot(233)
        ax3.yaxis.set_visible(False)
        ax3.xaxis.set_visible(False)
#        figcellbordergrn = ax3.imshow(mh.overlay(gray = justcellpair[:,:,1], green = cleanborders[:,:,1]))
#        figcellbordergrn = ax3.imshow(mh.overlay(gray = mh.stretch(padded_cell[:,:,1]), green = cleanborders[:,:,1]))
        figcellbordergrn = ax3.imshow(mh.overlay(gray = mh.stretch(cell[:,:,1]), green = cleanborders[:,:,1]))
        ax3.axes.autoscale(False)
        ax3.scatter(newcellmaxima_rcd[1][1], newcellmaxima_rcd[1][0], color='m', marker='o')
        
        ax4 = fig.add_subplot(235)
        ax4.yaxis.set_visible(False)
        ax4.xaxis.set_visible(False)
        ax4.imshow(justcellpair)
#        ax4.imshow(mh.stretch(cell))
        ax4.axes.autoscale(False)
        
        ax5 = fig.add_subplot(236)
        ax5.yaxis.set_visible(False)
        ax5.xaxis.set_visible(False)
        ax5.imshow(justcellpair)
#        ax5.imshow(mh.stretch(cell))
        ax5.axes.autoscale(False)

        ax6 = fig.add_subplot(234)
        ax6.yaxis.set_visible(False)
        ax6.xaxis.set_visible(False)
        ax6.imshow(cleanoverlay)
        ax6.axes.autoscale(False)

        plt.tight_layout()
        plt.draw()
#        plt.show('block'== False)


        ### Tell me the filenumber and filename of what you're working on:
        print('\n',(filenumber+1), filename)

        ### Skip the file if there is only 1 cell in the picture (sometimes the
        ### microscope skips a channel):
#        if len(mh.labeled.labeled_size(cleanareas)) < 3:
        if len(np.unique(cleanborders)) < 3:
            msg = "There aren't 2 cells in this image."
            print(msg)
            skip = 'y'
            
        ### Create an initial guess at the borders of the cells that are in
        ### contact with one another:
        rawcontactsurface = np.zeros(cleanareas.shape, np.dtype(np.int32))
        for (i,j,k), value in np.ndenumerate(cleanborders):
            if (0 in cleanborders[sub(i,1):i+2, sub(j,1):j+2, sub(k,1):k+2]) and (1 in cleanborders[sub(i,1):i+2, sub(j,1):j+2, sub(k,1):k+2]) and (2 in cleanborders[sub(i,1):i+2, sub(j,1):j+2, sub(k,1):k+2]):
                rawcontactsurface[i,j,k] = cleanborders[i,j,k]

        ### If this initial guess has too few pixels in contact then consider
        ### the cells to not be in contact:
        if np.count_nonzero(rawcontactsurface) <= 6:
            print('These cells are not in contact.')
            cells_touching = False
        elif np.count_nonzero(rawcontactsurface) > 6:
            cells_touching = True

        ### If there aren't 2 cells in the image just skip it.
        if skip == 'y':
            listupdate()
            continue

        ### If the cells are not in contact then set a bunch of values that will
        ### be in the table to be null and proceed to bin and estimate their 
        ### curvatures:
        if cells_touching == False and skip == 'n':

            ### Give the pixels of the cell borders sequential positional info
            ### starting from an arbitrary starting point:
            ordered_border_indices_red, ordered_border_indices_grn, numbered_border_indices_red, numbered_border_indices_grn, perimeter_cell1, perimeter_cell2 = count_pixels_around_borders_sequentially(curveborders)

            ### For 2 cells that are not in contact:
            ### Need to start by calculating smallest distance between the red and green borders
            closest_redborder_indices_rc, closest_grnborder_indices_rc, mean_closest_indice_loc_red, mean_closest_indice_loc_grn = find_closest_pixels_indices_between_cell_borders(curveborders[:,:,0], curveborders[:,:,1], ordered_border_indices_red, ordered_border_indices_grn, mean_closest_indice_loc_red_old, mean_closest_indice_loc_grn_old)
            mean_closest_indice_loc_red_old = np.copy(mean_closest_indice_loc_red)
            mean_closest_indice_loc_grn_old = np.copy(mean_closest_indice_loc_grn)

            ### Find the bin closest to the other cell and remove it:
            num_bins = num_bins
            far_borders_red, size_list_of_other_bins_red, closest_bin_size_red, closest_bin_indices_red = create_and_remove_bin_closest_to_other_cell(curveborders[:,:,0].astype(bool), ordered_border_indices_red, mean_closest_indice_loc_red, num_bins)
            
            ### Do the same for the green borders:
            far_borders_grn, size_list_of_other_bins_grn, closest_bin_size_grn, closest_bin_indices_grn = create_and_remove_bin_closest_to_other_cell(curveborders[:,:,1].astype(bool), ordered_border_indices_grn, mean_closest_indice_loc_grn, num_bins)

            ### Combine the 2 borders into an RGB image and assign red to be 1 
            ### and green to be 2:
            borders_without_closest_bin = np.dstack((far_borders_red*1, far_borders_grn*2, np.zeros((cell.shape))))
            closest_bin_red_and_grn = np.zeros(np.shape(cell))
            # closest_bin_red_and_grn[list(zip(*closest_bin_indices_red))] = 1
            # closest_bin_red_and_grn[list(zip(*closest_bin_indices_grn))] = 2
            closest_bin_red_and_grn[tuple(np.asarray(closest_bin_indices_red).T)] = 1
            closest_bin_red_and_grn[tuple(np.asarray(closest_bin_indices_grn).T)] = 2
            reordered_closest_bin_indices_red, reordered_closest_bin_indices_grn, closest_bin_len_red, closest_bin_len_grn = make_open_border_indices_sequential(closest_bin_red_and_grn)
            
            ### Re-define the sequencec of pixels for the borders_without_closest_bin:
            reordered_border_indices_red, reordered_border_indices_grn, perimeter_of_bins_1to7_cell1, perimeter_of_bins_1to7_cell2 = make_open_border_indices_sequential(borders_without_closest_bin)
            
            renumbered_border_indices_red = [p for p in enumerate(reordered_border_indices_red)]
            renumbered_border_indices_grn = [p for p in enumerate(reordered_border_indices_grn)]
            
            ### Next I need to use these renumbered border indices red and grn to define the
            ### other 7 bins and find their curvatures:
            num_bins = num_bins-1
            bin_indices_red, bin_start_red, bin_stop_red = sort_pixels_into_bins(reordered_border_indices_red, size_list_of_other_bins_red, closest_bin_indices_red, num_bins)
            bin_indices_grn, bin_start_grn, bin_stop_grn = sort_pixels_into_bins(reordered_border_indices_grn, size_list_of_other_bins_grn, closest_bin_indices_grn, num_bins)

            bin_start_red_indices = np.array(reordered_border_indices_red)[bin_start_red]            
            bin_start_red_indices = np.row_stack((bin_start_red_indices, reordered_closest_bin_indices_red[0]))
            bin_stop_red_indices = np.array(reordered_border_indices_red)[np.array(bin_stop_red)-1]
            bin_stop_red_indices = np.row_stack((bin_stop_red_indices, reordered_closest_bin_indices_red[-1]))

            bin_start_grn_indices = np.array(reordered_border_indices_grn)[bin_start_grn]            
            bin_start_grn_indices = np.row_stack((bin_start_grn_indices, reordered_closest_bin_indices_grn[0]))
            bin_stop_grn_indices = np.array(reordered_border_indices_grn)[np.array(bin_stop_grn)-1]
            bin_stop_grn_indices = np.row_stack((bin_stop_grn_indices, reordered_closest_bin_indices_grn[-1]))


            
            ### Finally, fit arcs to the binned borders and plot and return the
            ### the details to me:
            bin_curvature_details_red, all_fits_red_binned = fit_curves_to_bins(bin_indices_red, curveborders[:,:,0], odr_circle_fit_iterations)
            bin_curvature_details_grn, all_fits_grn_binned = fit_curves_to_bins(bin_indices_grn, curveborders[:,:,1], odr_circle_fit_iterations)

            ### While we're at it, also give me the curvature fit to the entire
            ### region of the cell borders not in contact with another cell:
            whole_curvature_details_red, all_fits_red_whole = fit_curves_to_bins([zip(*np.where(curveborders[:,:,0]))], curveborders[:,:,0], odr_circle_fit_iterations)
            whole_curvature_details_grn, all_fits_grn_whole = fit_curves_to_bins([zip(*np.where(curveborders[:,:,1]))], curveborders[:,:,1], odr_circle_fit_iterations)

            ### Record all the pixel_by_pixel curvature information you have: 
            mean_curvature_curveborders_red = np.mean(masked[list(zip(*ordered_border_indices_red))])
            median_curvature_curveborders_red = np.median(masked[len(list(zip(*ordered_border_indices_red)))])
            sem_curvature_curveborders_red = np.std(masked[list(zip(*ordered_border_indices_red))]) / len(np.where(masked[list(zip(*ordered_border_indices_red))])[0])
            std_curvature_curveborders_red = np.std(masked[list(zip(*ordered_border_indices_red))])

            ### Define some more default values:
            area_cell_1 = len(np.where(cleanareas==1)[0])
            area_cell_2 = len(np.where(cleanareas==2)[0])
            noncontact_perimeter_cell1 = perimeter_cell1
            noncontact_perimeter_cell2 = perimeter_cell2
            noncontactborders = np.copy(curveborders)

            degree_angle1 = 0.0
            degree_angle2 = 0.0
            average_angle = 0.0
            red_halfangle1 = 0.0
            red_halfangle2 = 0.0
            grn_halfangle1 = 0.0
            grn_halfangle2 = 0.0
            contactsurface_len = 0.0


            ### Update the data table:
            listupdate()
            
            ### Move on to the next picture:
            continue


        ### If the cells ARE in contact, then proceed to calculate adhesiveness
        ### parameters:
        if cells_touching == True and skip == 'n':
            
            ### Clean up the contacts:
            contactcorners, contactsurface, noncontactborders = clean_up_contact_borders(rawcontactsurface, cleanborders)
            
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
                       "\nStructuring element size:", Bc3D.shape
                       )
                listupdate()
                continue

            ### Now that we have defined the corners of the contact, the contact
            ### surface, and the parts of the cell border that are not in
            ### contact, we can find the contact angles:
            temp = find_contact_angles(contactcorners, contactsurface, noncontactborders)
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
            
            n_contactsurface_px = int(len(np.where(contactsurface)[0])/2)
            cx,cy = np.linspace(rawcontact_vector_start[1], rawcontact_vector_end[1], num=n_contactsurface_px), np.linspace(rawcontact_vector_start[0], rawcontact_vector_end[0], num=n_contactsurface_px)
            imgform_contact_points = zip(cy,cx)
        
        
            ### Determine the half-angle for each of the red and green cells:
            red_halfangle1_rad = math.acos( np.round( ((np.dot(vector1,contact_vector)) / (((vector1[0]**2 + vector1[1]**2)**0.5)*((contact_vector[0]**2 + contact_vector[1]**2)**0.5)) ), decimals=10) )
            red_halfangle2_rad = math.acos( np.round( ((np.dot(vector3,contact_vector)) / (((vector3[0]**2 + vector3[1]**2)**0.5)*((contact_vector[0]**2 + contact_vector[1]**2)**0.5)) ), decimals=10) )
            grn_halfangle1_rad = math.acos( np.round( ((np.dot(vector2,contact_vector)) / (((vector2[0]**2 + vector2[1]**2)**0.5)*((contact_vector[0]**2 + contact_vector[1]**2)**0.5)) ), decimals=10) )
            grn_halfangle2_rad = math.acos( np.round( ((np.dot(vector4,contact_vector)) / (((vector4[0]**2 + vector4[1]**2)**0.5)*((contact_vector[0]**2 + contact_vector[1]**2)**0.5)) ), decimals=10) )
            ### and convert to degrees:
            red_halfangle1 = math.degrees(red_halfangle1_rad)
            red_halfangle2 = 180-math.degrees(red_halfangle2_rad)
            grn_halfangle1 = math.degrees(grn_halfangle1_rad)
            grn_halfangle2 = 180-math.degrees(grn_halfangle2_rad)


            ### Count the length of the contact surface in pixel lengths
#            contactsurface_len, contactsurface_len1, contactsurface_len2 = find_contactsurface_len(contactsurface)
#            print "The length of the contact surface is %.3f pixels" %contactsurface_len   

            ### Find the mean fluorescence intensity value that the red and grn
            ### linescans of the contact intersect at as an indirect measure of
            ### how close the 2 cells are:
            intersectsx, intersectsy, intersection_values_per_cell, intersection_values_per_cell_middle_list_index, intersection_cellpair_num, red_gap_scanx, red_gap_scany, grn_gap_scanx, grn_gap_scany = perform_linescan(contactsurface, perpendicular2contact_vector, linescan_length)

            ### Find the perimeter of the red and green cells:
            ordered_border_indices_red, ordered_border_indices_grn, numbered_border_indices_red, numbered_border_indices_grn, perimeter_cell1, perimeter_cell2 = count_pixels_around_borders_sequentially(curveborders)

            ### Give the noncontactborders sequential information:
            reordered_border_indices_red, reordered_border_indices_grn, noncontact_perimeter_cell1, noncontact_perimeter_cell2 = make_open_border_indices_sequential(noncontactborders+pathfinder)

            renumbered_border_indices_red = [p for p in enumerate(reordered_border_indices_red)]
            renumbered_border_indices_grn = [p for p in enumerate(reordered_border_indices_grn)]

            ### Bin the noncontactborders into 7 bins:
            num_bins = num_bins-1
            bin_size_list_red = create_border_bin_sizes((noncontactborders[:,:,0]+pathfinder[:,:,0]).astype(bool), reordered_border_indices_red, num_bins)
            bin_size_list_grn = create_border_bin_sizes((noncontactborders[:,:,1]+pathfinder[:,:,1]).astype(bool), reordered_border_indices_grn, num_bins)

            ### Make the 8th bin the contactsurface:
#            contact_bin_indices_red = zip(*np.where(contactsurface==1))
#            contact_bin_indices_grn = zip(*np.where(contactsurface==2))
            reordered_contact_surface_red, reordered_contact_surface_grn, contactsurface_len1, contactsurface_len2 = make_open_border_indices_sequential(contactsurface)
            contactsurface_len = (contactsurface_len1 + contactsurface_len2) / 2.0
            print("The length of the contact surface is %.3f pixels" %contactsurface_len)

            ### Return the bin-sorted indices for each pixel:
#            bin_indices_red, bin_start_red, bin_stop_red = sort_pixels_into_bins(reordered_border_indices_red, bin_size_list_red, contact_bin_indices_red, num_bins)
#            bin_indices_grn, bin_start_grn, bin_stop_grn = sort_pixels_into_bins(reordered_border_indices_grn, bin_size_list_grn, contact_bin_indices_grn, num_bins)
            bin_indices_red, bin_start_red, bin_stop_red = sort_pixels_into_bins(reordered_border_indices_red, bin_size_list_red, reordered_contact_surface_red, num_bins)
            bin_indices_grn, bin_start_grn, bin_stop_grn = sort_pixels_into_bins(reordered_border_indices_grn, bin_size_list_grn, reordered_contact_surface_grn, num_bins)

            bin_start_red_indices = np.array(reordered_border_indices_red)[bin_start_red]            
            bin_start_red_indices = np.row_stack((bin_start_red_indices, reordered_contact_surface_red[0]))
            bin_stop_red_indices = np.array(reordered_border_indices_red)[np.array(bin_stop_red)-1]
            bin_stop_red_indices = np.row_stack((bin_stop_red_indices, reordered_contact_surface_red[-1]))

            bin_start_grn_indices = np.array(reordered_border_indices_grn)[bin_start_grn]            
            bin_start_grn_indices = np.row_stack((bin_start_grn_indices, reordered_contact_surface_grn[0]))
            bin_stop_grn_indices = np.array(reordered_border_indices_grn)[np.array(bin_stop_grn)-1]
            bin_stop_grn_indices = np.row_stack((bin_stop_grn_indices, reordered_contact_surface_grn[-1]))



            ### Add the len of the 8th bin to the bin_size_list_red (and grn):
#            bin_size_list_red.append(len(bin_indices_red[7]))
#            bin_size_list_grn.append(len(bin_indices_grn[7]))

            ### Find the binned curvatures:
            bin_curvature_details_red, all_fits_red_binned = fit_curves_to_bins(bin_indices_red, curveborders[:,:,0], odr_circle_fit_iterations)
            bin_curvature_details_grn, all_fits_grn_binned = fit_curves_to_bins(bin_indices_grn, curveborders[:,:,1], odr_circle_fit_iterations)

            ### While we're at it, also give me the curvature fit to the entire
            ### region of the cell borders not in contact with another cell:
            whole_curvature_details_red, all_fits_red_whole = fit_curves_to_bins([zip(*np.where((noncontactborders[:,:,0]+pathfinder[:,:,0]).astype(bool)))], (noncontactborders[:,:,0]+pathfinder[:,:,0]).astype(bool), odr_circle_fit_iterations)
            whole_curvature_details_grn, all_fits_grn_whole = fit_curves_to_bins([zip(*np.where((noncontactborders[:,:,1]+pathfinder[:,:,1]).astype(bool)))], (noncontactborders[:,:,1]+pathfinder[:,:,1]).astype(bool), odr_circle_fit_iterations)

            ### Define some more default values:
            area_cell_1 = len(np.where(cleanareas==1)[0])
            area_cell_2 = len(np.where(cleanareas==2)[0])



            ### Record all the pixel_by_pixel curvature information:
            curvature_pathfinder_red = masked[np.where(pathfinder==1)]
            curvature_pathfinder_grn = masked[np.where(pathfinder==2)]
            curvature_pathfinder_angle1 = masked[np.where(angles_angle_labeled == 1)]
            curvature_pathfinder_angle2 = masked[np.where(angles_angle_labeled == 2)]
            all_curvatures_pathfinder = list(curvature_pathfinder_red)+list(curvature_pathfinder_grn)
            all_curvatures_pathfinder_grn_angle1 = masked[np.where(angles_angle_labeled[:,:,1] == 1)]
            all_curvatures_pathfinder_grn_angle1 = all_curvatures_pathfinder_grn_angle1[:,1]
            all_curvatures_pathfinder_grn_angle2 = masked[np.where(angles_angle_labeled[:,:,1] == 2)]
            all_curvatures_pathfinder_grn_angle2 = all_curvatures_pathfinder_grn_angle2[:,1]
            all_curvatures_pathfinder_red_angle1 = masked[np.where(angles_angle_labeled[:,:,0] == 1)]
            all_curvatures_pathfinder_red_angle1 = all_curvatures_pathfinder_red_angle1[:,0]
            all_curvatures_pathfinder_red_angle2 = masked[np.where(angles_angle_labeled[:,:,0] == 2)]
            all_curvatures_pathfinder_red_angle2 = all_curvatures_pathfinder_red_angle2[:,0]
            curvature_contactsurface_red = masked[np.where(contactsurface == 1)]
            curvature_contactsurface_grn = masked[np.where(contactsurface == 2)]

            mean_curvature_curveborders_red = np.mean(masked[list(zip(*ordered_border_indices_red))])
            median_curvature_curveborders_red = np.median(masked[list(zip(*ordered_border_indices_red))])
            sem_curvature_curveborders_red = np.std(masked[list(zip(*ordered_border_indices_red))]) / len(np.where(masked[list(zip(*ordered_border_indices_red))])[0])
            std_curvature_curveborders_red = np.std(masked[list(zip(*ordered_border_indices_red))])
            
            all_curvatures_noncontactborders_red = masked[list(zip(*reordered_border_indices_red))]
            mean_curvature_noncontactborders_red = np.mean(masked[list(zip(*reordered_border_indices_red))])
            median_curvature_noncontactborders_red = np.median(masked[list(zip(*reordered_border_indices_red))])
            sem_curvature_noncontactborders_red = np.std(masked[list(zip(*reordered_border_indices_red))]) / len(np.where(masked[list(zip(*ordered_border_indices_red))])[0])
            std_curvature_noncontactborders_red = np.std(masked[list(zip(*reordered_border_indices_red))])
            
            mean_curvature_pathfinder_red = np.mean(curvature_pathfinder_red)
            median_curvature_pathfinder_red = np.median(curvature_pathfinder_red)
            sem_curvature_pathfinder_red = np.std(curvature_pathfinder_red) / len(np.where(curvature_pathfinder_red)[0])
            std_curvature_pathfinder_red = np.std(curvature_pathfinder_red)
            
            mean_curvature_contactsurface_red = np.mean(curvature_contactsurface_red)
            median_curvature_contactsurface_red = np.median(curvature_contactsurface_red)
            sem_curvature_contactsurface_red = np.std(curvature_contactsurface_red) / len(np.where(curvature_contactsurface_red)[0])
            std_curvature_contactsurface_red = np.std(curvature_contactsurface_red)
            
            mean_curvature_curveborders_grn = np.mean(masked[list(zip(*ordered_border_indices_grn))])
            median_curvature_curveborders_grn = np.median(masked[list(zip(*ordered_border_indices_grn))])
            sem_curvature_curveborders_grn = np.std(masked[list(zip(*ordered_border_indices_grn))]) / len(np.where(masked[list(zip(*ordered_border_indices_grn))])[0])
            std_curvature_curveborders_grn = np.std(masked[list(zip(*ordered_border_indices_grn))])
            
            all_curvatures_noncontactborders_grn = masked[list(zip(*reordered_border_indices_grn))]
            mean_curvature_noncontactborders_grn = np.mean(masked[list(zip(*reordered_border_indices_grn))])
            median_curvature_noncontactborders_grn = np.median(masked[list(zip(*reordered_border_indices_grn))])
            sem_curvature_noncontactborders_grn = np.std(masked[list(zip(*reordered_border_indices_grn))]) / len(np.where(masked[list(zip(*ordered_border_indices_grn))])[0])
            std_curvature_noncontactborders_grn = np.std(masked[list(zip(*reordered_border_indices_grn))])
            
            mean_curvature_pathfinder_grn = np.mean(curvature_pathfinder_grn)
            median_curvature_pathfinder_grn = np.median(curvature_pathfinder_grn)
            sem_curvature_pathfinder_grn = np.std(curvature_pathfinder_grn) / len(np.where(curvature_pathfinder_grn)[0])
            std_curvature_pathfinder_grn = np.std(curvature_pathfinder_grn)
            
            mean_curvature_contactsurface_grn = np.mean(curvature_contactsurface_grn)
            median_curvature_contactsurface_grn = np.median(curvature_contactsurface_grn)
            sem_curvature_contactsurface_grn = np.std(curvature_contactsurface_grn) / len(np.where(curvature_contactsurface_grn)[0])
            std_curvature_contactsurface_grn = np.std(curvature_contactsurface_grn)
            
            mean_curvature_pathfinder = np.mean(all_curvatures_pathfinder)
            median_curvature_pathfinder = np.median(all_curvatures_pathfinder)
            sem_curvature_pathfinder = np.std(all_curvatures_pathfinder) / len(np.where(all_curvatures_pathfinder)[0])
            std_curvature_pathfinder = np.std(all_curvatures_pathfinder)
            
            mean_curvature_pathfinder_angle1 = np.mean(curvature_pathfinder_angle1)
            median_curvature_pathfinder_angle1 = np.median(curvature_pathfinder_angle1)
            sem_curvature_pathfinder_angle1 = np.std(curvature_pathfinder_angle1) / len(np.where(curvature_pathfinder_angle1)[0])
            std_curvature_pathfinder_angle1 = np.std(curvature_pathfinder_angle1)
            
            mean_curvature_pathfinder_angle2 = np.mean(curvature_pathfinder_angle2)
            median_curvature_pathfinder_angle2 = np.median(curvature_pathfinder_angle2)
            sem_curvature_pathfinder_angle2 = np.std(curvature_pathfinder_angle2) / len(np.where(curvature_pathfinder_angle2)[0])
            std_curvature_pathfinder_angle2 = np.std(curvature_pathfinder_angle2)
            
            mean_curvature_pathfinder_grn_angle1 = np.mean(all_curvatures_pathfinder_grn_angle1)
            median_curvature_pathfinder_grn_angle1 = np.median(all_curvatures_pathfinder_grn_angle1)
            sem_curvature_pathfinder_grn_angle1 = np.std(all_curvatures_pathfinder_grn_angle1) / len(np.where(all_curvatures_pathfinder_grn_angle1)[0])
            std_curvature_pathfinder_grn_angle1 = np.std(all_curvatures_pathfinder_grn_angle1)
            
            mean_curvature_pathfinder_grn_angle2 = np.mean(all_curvatures_pathfinder_grn_angle2)
            median_curvature_pathfinder_grn_angle2 = np.median(all_curvatures_pathfinder_grn_angle2)
            sem_curvature_pathfinder_grn_angle2 = np.std(all_curvatures_pathfinder_grn_angle2) / len(np.where(all_curvatures_pathfinder_grn_angle2)[0])
            std_curvature_pathfinder_grn_angle2 = np.std(all_curvatures_pathfinder_grn_angle2)
            
            mean_curvature_pathfinder_red_angle1 = np.mean(all_curvatures_pathfinder_red_angle1)
            median_curvature_pathfinder_red_angle1 = np.median(all_curvatures_pathfinder_red_angle1)
            sem_curvature_pathfinder_red_angle1 = np.std(all_curvatures_pathfinder_red_angle1) / len(np.where(all_curvatures_pathfinder_red_angle1)[0])
            std_curvature_pathfinder_red_angle1 = np.std(all_curvatures_pathfinder_red_angle1)
            
            mean_curvature_pathfinder_red_angle2 = np.mean(all_curvatures_pathfinder_red_angle2)
            median_curvature_pathfinder_red_angle2 = np.median(all_curvatures_pathfinder_red_angle2)
            sem_curvature_pathfinder_red_angle2 = np.std(all_curvatures_pathfinder_red_angle2) / len(np.where(all_curvatures_pathfinder_red_angle2)[0])
            std_curvature_pathfinder_red_angle2 = np.std(all_curvatures_pathfinder_red_angle2)



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
            contact_surface_line_grn_rcd = [(vector2_start[1], vector4_start[1]), (vector2_start[0], vector4_start[0])]
            ### draw a line that is perpendicular to the contact surface line, in the middle
            ### of the contact surface:
            temp1_rcd = (vector1_start[::-1] + vector2_start[::-1])/2.0
            temp2_rcd = (vector3_start[::-1] + vector4_start[::-1])/2.0
            mean_contact_surface_line_rcd = (temp1_rcd + temp2_rcd)/2.0
            center_of_contact_surface_lines_rc = mean_contact_surface_line_rcd[:2]
    
#            cleanoverlay = np.copy(padded_cell)
            cleanoverlay = np.copy(cell)
            cleanoverlay[:,:,0] = mh.stretch(cleanoverlay[:,:,0])
            cleanoverlay[:,:,1] = mh.stretch(cleanoverlay[:,:,1])
            cleanoverlay[:,:,2] = mh.stretch(cleanoverlay[:,:,2])
            cleanoverlay[np.where(curveborders)] = 255
            cleanoverlay = mh.overlay(np.max(cell, axis=2), red=curveborders[:,:,0], green=curveborders[:,:,1])
            
            ### Draw what you've found over the images of the cells:
            fig = plt.figure(num=str(filenumber+1)+'_'+os.path.basename(filename), figsize=(12., 10.))
            plt.suptitle(str(filenumber+1)+' - '+os.path.basename(filename), verticalalignment='center', x=0.5, y=0.5, fontsize=14)
            ax1 = fig.add_subplot(231)
            ax1.yaxis.set_visible(False)
            ax1.xaxis.set_visible(False)
            ax1.imshow(mh.stretch(cell_raw))
            
            ax2 = fig.add_subplot(232)
            ax2.yaxis.set_visible(False)
            ax2.xaxis.set_visible(False)
#            figcellborderred = ax2.imshow(mh.overlay(gray = justcellpair[:,:,0], red = cleanborders[:,:,0]))
#            figcellborderred = ax2.imshow(mh.overlay(gray = mh.stretch(padded_cell[:,:,0]), red = cleanborders[:,:,0]))
            figcellborderred = ax2.imshow(mh.overlay(gray = mh.stretch(cell[:,:,0]), red = cleanborders[:,:,0]))
            ax2.axes.autoscale(False)
            ax2.scatter(newcellmaxima_rcd[0][1], newcellmaxima_rcd[0][0], color='m', marker='o')
            
            ax3 = fig.add_subplot(233)
            ax3.yaxis.set_visible(False)
            ax3.xaxis.set_visible(False)
#            figcellbordergrn = ax3.imshow(mh.overlay(gray = justcellpair[:,:,1], green = cleanborders[:,:,1]))
#            figcellbordergrn = ax3.imshow(mh.overlay(gray = mh.stretch(paddedcell[:,:,1]), green = cleanborders[:,:,1]))
            figcellbordergrn = ax3.imshow(mh.overlay(gray = mh.stretch(cell[:,:,1]), green = cleanborders[:,:,1]))
            ax3.axes.autoscale(False)
            ax3.scatter(newcellmaxima_rcd[1][1], newcellmaxima_rcd[1][0], color='m', marker='o')
            
            ax4 = fig.add_subplot(235)
            ax4.yaxis.set_visible(False)
            ax4.xaxis.set_visible(False)
            ax4.imshow(justcellpair)
#            ax4.imshow(mh.stretch(cell))
            ax4.axes.autoscale(False)
            drawn_arrows = ax4.quiver(vector_start_x_vals, vector_start_y_vals, vector_x_vals, vector_y_vals, units='xy',scale_units='xy', angles='xy', scale=0.33, color='c', width = 1.0)
            angle1_annotations = ax4.annotate('%.1f' % degree_angle1, vector1_start, vector1_start + 30*(ang1_annot / ang1_annot_mag)*(1,1), color='c', fontsize='24', ha='center', va='center')
            angle2_annotations = ax4.annotate('%.1f' % degree_angle2, vector3_start, vector3_start + 30*(ang2_annot / ang2_annot_mag)*(1,1), color='c', fontsize='24', ha='center', va='center')
            redcell_contactvect = ax4.plot(contact_surface_line_red_rcd[1], contact_surface_line_red_rcd[0], color='c', lw=1)
            grncell_contactvect = ax4.plot(contact_surface_line_grn_rcd[1], contact_surface_line_grn_rcd[0], color='c', lw=1)
            contactvect = ax4.plot([mean_contact_surface_line_rcd[1]-2*contact_vector[0], mean_contact_surface_line_rcd[1]+2*contact_vector[0]], [mean_contact_surface_line_rcd[0]-2*contact_vector[1], mean_contact_surface_line_rcd[0]+2*contact_vector[1]], color='m')
            
            ax5 = fig.add_subplot(236)
            ax5.yaxis.set_visible(False)
            ax5.xaxis.set_visible(False)
            ax5.imshow( (~(~justcellpair*~contactsurface.astype(np.bool))) )
#            ax5.imshow( (~(~mh.stretch(cell)*~contactsurface.astype(np.bool))) )
            ax5.axes.autoscale(False)
    
            ax6 = fig.add_subplot(234)
            ax6.yaxis.set_visible(False)
            ax6.xaxis.set_visible(False)
            ax6.imshow(cleanoverlay)
            ax6.axes.autoscale(False)
#            contactvect2 = ax6.plot([mean_contact_surface_line_rcd[1]-2*contact_vector[0], mean_contact_surface_line_rcd[1]+2*contact_vector[0]], [mean_contact_surface_line_rcd[0]-2*contact_vector[1], mean_contact_surface_line_rcd[0]+2*contact_vector[1]], color='m')
#            perp2contactvect = ax6.plot([center_of_contact_surface_lines_rc[1]-perpendicular2contact_vector[0], center_of_contact_surface_lines_rc[1]+perpendicular2contact_vector[0]], [center_of_contact_surface_lines_rc[0]-perpendicular2contact_vector[1], center_of_contact_surface_lines_rc[0]+perpendicular2contact_vector[1]], color='b')
            
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

### Plot the intensities:
### Organize data in to arrays for to be plotted in 3D
red_intens_mean_arr = np.asarray(master_red_linescan_mean)
grn_intens_mean_arr = np.asarray(master_grn_linescan_mean)

redx, redy = np.meshgrid(range(np.size(red_intens_mean_arr, 1)), range(np.size(red_intens_mean_arr, 0)))
grnx, grny = np.meshgrid(range(np.size(grn_intens_mean_arr, 1)), range(np.size(grn_intens_mean_arr, 0)))

### Actually graph the intensities:
#mean_intens_fig = plt.figure("Mean Intensities")
#ax_mean_int = mean_intens_fig.add_subplot(111, projection='3d')
#ax_mean_int.plot_wireframe(redx, redy, master_red_linescan_mean, color='r')
#ax_mean_int.plot_wireframe(grnx, grny, master_grn_linescan_mean, color='g')
#ax_mean_int.set_zlim(0,255)
#ax_mean_int.set_zlabel('Channel-Normalized Mean Fluorescence Intensity', fontsize = 12)
#ax_mean_int.set_xlabel('Position Along Intensity Plot (px)', fontsize = 12)
#ax_mean_int.set_ylabel('Time (minutes)', fontsize = 12)

poly_fit_intersect_fig = plt.figure('Intensities at intersections of polynomial fitted curves')
ax_poly_inters = poly_fit_intersect_fig.add_subplot(111)
ax_poly_inters.plot(cell_pair_num_list, intens_at_intersect, 'k')
ax_poly_inters.plot(cell_pair_num_list, intens_at_intersect, 'k.')
ax_poly_inters.set_xlabel('Time (minutes)', fontsize = 18)
ax_poly_inters.set_ylabel('Average Fluorescence at Intersection', fontsize = 18)
### The theoretical y-max is 255, but that would be if you had overlap of the 
### two brightest point in the cell:
ax_poly_inters.set_ylim(0,255)
ax_poly_inters.set_xlim(0)



intersection_table = np.c_[master_intersection_cellpair_num, master_intersection_values_per_cell_middle_list_index, master_intersection_values_list]
intersection_table = intersection_table.T

### Extract contact midpoint intersection data:
midpoint_intersect_fluor_values = [master_intersection_values_list[i][x] for i,x in enumerate(master_intersection_values_per_cell_middle_list_index)]

### Plot contact midpoint intersection data:
contact_midpoint_intersects = plt.figure("Intersection Fluorescence at Contact Midpoints")
ax_midpoint_intersect = contact_midpoint_intersects.add_subplot(111)
ax_midpoint_intersect.plot(master_intersection_cellpair_num, midpoint_intersect_fluor_values, 'k')
ax_midpoint_intersect.plot(master_intersection_cellpair_num, midpoint_intersect_fluor_values, 'k.')
ax_midpoint_intersect.set_xlim(0)
ax_midpoint_intersect.set_ylim(0, 255)

    
plt.draw()
### If you want to see the plots as they are being plotted then you can uncomment
### the line below, but you will also have to uncomment plt.ion() if you are running
### the script in an external console so that the program continues running 
### without requiring you to close all of the plots.
### console.
# plt.show()


while True:
    save_table = input("Would you like to save the output?"
                            "(Y/N)?\n>"
                            )
    # save_table = 'y'
    if save_table == 'y' or save_table == 'Y' or save_table == 'yes':
        
        num_bins = 8
        ### Turn the curvature details into arrays:
        bin_curvature_details_array_red = np.asarray(bin_curvature_details_red_list, dtype='object')
        bin_curvature_details_array_grn = np.asarray(bin_curvature_details_grn_list, dtype='object')
        whole_curvature_details_red_array = np.asarray(whole_cell_curvature_details_red_list, dtype='object')
        whole_curvature_details_grn_array = np.asarray(whole_cell_curvature_details_grn_list, dtype='object')

        ### Give me a list of the curvature value columns:
        bin_curv_val_red_list = [list(bin_curvature_details_array_red[:,i,2]) for i in range(1,bin_curvature_details_array_red.shape[1])]
        bin_curv_val_grn_list = [list(bin_curvature_details_array_grn[:,i,2]) for i in range(1,bin_curvature_details_array_grn.shape[1])]
        whole_curv_val_red_list = list(whole_curvature_details_red_array[:,1,2])
        whole_curv_val_grn_list = list(whole_curvature_details_grn_array[:,1,2])
        
        ### Give me a list of the curvature residuals columns:
        bin_curv_residu_red_list = [list(bin_curvature_details_array_red[:,i,9]) for i in range(1,bin_curvature_details_array_red.shape[1])]
        bin_curv_residu_grn_list = [list(bin_curvature_details_array_grn[:,i,9]) for i in range(1,bin_curvature_details_array_grn.shape[1])]
        whole_curv_residu_red_list = list(whole_curvature_details_red_array[:,1,9])
        whole_curv_residu_grn_list = list(whole_curvature_details_grn_array[:,1,9])

        ### Give me a list of the arclength of the fitted curvature:
        bin_curv_arclen_red_list = [list(bin_curvature_details_array_red[:,i,16]) for i in range(1,bin_curvature_details_array_red.shape[1])]
        bin_curv_arclen_grn_list = [list(bin_curvature_details_array_grn[:,i,16]) for i in range(1,bin_curvature_details_array_grn.shape[1])]
        whole_curv_arclen_red_list = list(whole_curvature_details_red_array[:,1,16])
        whole_curv_arclen_grn_list = list(whole_curvature_details_grn_array[:,1,16])
        
        ### Give me a list of the arc angle of the fitted curvatures (in radians):
        bin_curv_arcanglerad_red_list = [list(bin_curvature_details_array_red[:,i,15]) for i in range(1,bin_curvature_details_array_red.shape[1])]
        bin_curv_arcanglerad_grn_list = [list(bin_curvature_details_array_grn[:,i,15]) for i in range(1,bin_curvature_details_array_grn.shape[1])]
        whole_curv_arcanglerad_red_list = list(whole_curvature_details_red_array[:,1,15])
        whole_curv_arcanglerad_grn_list = list(whole_curvature_details_grn_array[:,1,15])
    #ftest = list(zip(*bin_curv_val_red_list, whole_curv_val_red_list))

        ### Give your columns headers:
        time_list.insert(0, 'Time')
        cell_pair_num_list.insert(0, 'Cell Pair Number')
        angle_1_list.insert(0, 'Angle 1 (deg)')
        angle_2_list.insert(0, 'Angle 2 (deg)')
        average_angle_list.insert( 0, 'Average Angle (deg)' )
        contactsurface_list.insert( 0, 'Contact Surface Length (px)')
        contact_corner1_list.insert(0, 'Cell 1 (Red) contact corner locations (x,y)')
        contact_corner2_list.insert(0, 'Cell 2 (Green) contact corner locations (x,y)')
        corner_corner_length1_list.insert(0, 'Length between contact corners for cell 1 (Red) (px)')
        corner_corner_length2_list.insert(0, 'Length between contact corners for cell 2 (Green) (px)')        
        noncontactsurface1_list.insert(0, 'Cell 1 (Red) Non-Contact Surface Length (px)')
        noncontactsurface2_list.insert(0, 'Cell 2  (Green) Non-Contact Surface Length (px)')
        cell1_perimeter_list.insert(0, 'Cell 1 (Red) Perimeter Length (px)')
        cell2_perimeter_list.insert(0, 'Cell 2 (Green) Perimeter Length (px)')
        filename_list.insert( 0, 'Filename')
        red_gauss_list.insert( 0, 'Red Gaussian Filter Value')
        grn_gauss_list.insert( 0, 'Green Gaussian Filter Value')
        object_of_interest_brightness_cutoff_percent_list.insert( 0, 'object of interest brightness cutoff percent')
        otsu_fraction_list.insert( 0, 'otsu threshold percent')
        struct_elem_list.insert( 0, 'Structuring Element Size')
        Nth_pixel_list.insert(0, '# of pixels away from contact corners (px)')
        disk_radius_list.insert(0, 'Structuring element disk radius (px)')
        background_threshold_list.insert(0, 'Background threshold (range is %1.f to %1.f)' %(np.iinfo(cell.dtype).min, np.iinfo(cell.dtype).max))
        object_of_interest_brightness_cutoff_red_list.insert(0, 'Red segmented regions to keep mean brightness threshold (range is %1.f to %1.f)' %(np.iinfo(cell.dtype).min, np.iinfo(cell.dtype).max))
        object_of_interest_brightness_cutoff_grn_list.insert(0, 'Green segmented regions to keep mean brightness threshold (range is %1.f to %1.f)' %(np.iinfo(cell.dtype).min, np.iinfo(cell.dtype).max))
        area_cell_1_list.insert(0, 'Area of cell 1 (Red) (px**2)')
        area_cell_2_list.insert(0, 'Area of cell 2 (Green) (px**2)')
        cell_pair_area_list.insert(0, 'Area of the cell pair (px**2)')
        intersect_pos.insert(0, 'Location on linescan of Polyline fit intersections (px)')
        intens_at_intersect.insert(0, 'Mean Fluorescence intensity at intersection of Polyline fits (a.u.)')
        linescan_length_list.insert(0, 'Linescan Length (px)')
        curve_length_list.insert(0, 'Curve Length (px)')
        otsu_list.insert(0, 'Otsu Thresholding?')
        master_intersection_values_list.insert(0, 'Fluorescence Intensity at Intersection (a.u.)')
        master_intersection_values_per_cell_middle_list_index.insert(0, 'Position Along Contact Surface (px relative to centre)')
        master_intersection_cellpair_num.insert(0, 'Cell Pair Number')
        midpoint_intersect_fluor_values.insert(0, "Intersection Fluorescence at Contact Midpoints (a.u.)")
        processing_time_list.insert(0, "Image Processing Time (seconds)")
        mask_list.insert(0, "Mask? (manually defined by user who checks quality of cell borders)")
        redcellborders_coords.insert(0, "Red cellborders coordinates as list of (row, col) (ie. (y,x))")
        redcontactsurface_coords.insert(0, "Red contact surface coordinates as list of (row, col) (ie. (y,x))")
        rednoncontact_coords.insert(0, "Red non-contact border coordinates as list of (row, col) (ie. (y,x))")
        grncellborders_coords.insert(0, "Green cellborders coordinates as list of (row, col) (ie. (y,x))")
        grncontactsurface_coords.insert(0, "Green contact surface coordinates as list of (row, col) (ie. (y,x))")
        grnnoncontact_coords.insert(0, "Green non-contact border coordinates as list of (row, col) (ie. (y,x))")
        redcontactvector1_start.insert(0, "Contact angle 1 vector start point from red cell (x,y)") ### I'm pretty sure this is in (x,y) coordinates.
        grncontactvector1_start.insert(0, "Contact angle 1 vector start point from green cell (x,y)") ### I'm pretty sure this is in (x,y) coordinates.
        redcontactvector2_start.insert(0, "Contact angle 2 vector start point from red cell (x,y)") ### I'm pretty sure this is in (x,y) coordinates.
        grncontactvector2_start.insert(0, "Contact angle 2 vector start point from green cell (x,y)") ### I'm pretty sure this is in (x,y) coordinates.
        redcontactvector1.insert(0, "Contact angle 1 vector from red cell (x,y)") ### I'm pretty sure this is in (x,y) coordinates.
        grncontactvector1.insert(0, "Contact angle 1 vector from green cell (x,y)") ### I'm pretty sure this is in (x,y) coordinates.
        redcontactvector2.insert(0, "Contact angle 2 vector from red cell (x,y)") ### I'm pretty sure this is in (x,y) coordinates.
        grncontactvector2.insert(0, "Contact angle 2 vector from green cell (x,y)") ### I'm pretty sure this is in (x,y) coordinates.
        red_maxima_coords.insert(0, "Red cell seed coordinates (x,y)")
        grn_maxima_coords.insert(0, "Green cell seed coordinates (x,y)")
        contact_vectors_list.insert(0, "Contact vector from one pair of contact corners to the other (x,y)") ### I'm pretty sure this is in (x,y) coordinates.
        perpendicular2contact_vectors_list.insert(0, "Perpendicular to contact vector (x,y)") ### I'm pretty sure this is in (x,y) coordinates.
        red_halfangle1_list.insert(0, "Red cell's half-angle of angle 1 (deg)")
        red_halfangle2_list.insert(0, "Red cell's half-angle of angle 2 (deg)")
        grn_halfangle1_list.insert(0, "Green cell's half-angle of angle 1 (deg)")
        grn_halfangle2_list.insert(0, "Green cell's half-angle of angle 2 (deg)")
        curvature_coords_red_list.insert(0, 'Red curvature coordinates (row,col)')
        curvature_coords_grn_list.insert(0, 'Green curvature coordinates (row,col)')
        curvature_values_red_list.insert(0, 'Red curvature values (1.0/(radius in pixels))')
        curvature_values_grn_list.insert(0, 'Green curvature values (1.0/(radius in pixels))')
        contactsurface_len_red_list.insert(0, 'Contact surface length of red cell (px)')
        contactsurface_len_grn_list.insert(0, 'Contact surface length of green cell (px)')
        numbered_border_indices_red_list.insert(0, 'Sequential red cell border coordinates (sequence, (i,j,k))')
        numbered_border_indices_grn_list.insert(0, 'Sequential green cell border coordinates (sequence, (i,j,k))')        



        mean_curvature_curveborders_list_red.insert(0, 'Mean curvature of red borders')
        median_curvature_curveborders_list_red.insert(0, 'Median curvature of red borders')
        sem_curvature_curveborders_list_red.insert(0, 'SEM of mean curvature of red borders')
        std_curvature_curveborders_list_red.insert(0, 'STD of mean curvature of red borders')

        all_curvatures_noncontactborders_list_red.insert(0, 'All curvatures of red non-contact borders')
        mean_curvature_noncontactborders_list_red.insert(0, 'Mean curvature of red non-contact borders')
        median_curvature_noncontactborders_list_red.insert(0, 'Median curvature of red non-contact borders')
        sem_curvature_noncontactborders_list_red.insert(0, 'SEM of mean curvature of red non-contact borders')
        std_curvature_noncontactborders_list_red.insert(0, 'STD of mean curvature of red non-contact borders')

        all_curvatures_pathfinder_list_red.insert(0, 'All curvatures of pixels used to make red contact vectors')
        mean_curvature_pathfinder_list_red.insert(0, 'Mean curvature of pixels used to make red contact vectors')
        median_curvature_pathfinder_list_red.insert(0, 'Median curvature of pixels used to make red contact vectors')
        sem_curvature_pathfinder_list_red.insert(0, 'SEM of mean curvature of pixels used to make red contact vectors')
        std_curvature_pathfinder_list_red.insert(0, 'STD of mean curvature of pixels used to make red contact vectors')

        all_curvatures_contactsurface_list_red.insert(0, 'All curvatures of red contact diameter')
        mean_curvature_contactsurface_list_red.insert(0, 'Mean curvature of red contact diameter')
        median_curvature_contactsurface_list_red.insert(0, 'Median curvature of red contact diameter')
        sem_curvature_contactsurface_list_red.insert(0, 'SEM of mean curvature of red contact diameter')
        std_curvature_contactsurface_list_red.insert(0, 'STD of mean curvature of red contact diameter')


        mean_curvature_curveborders_list_grn.insert(0, 'Mean curvature of green borders')
        median_curvature_curveborders_list_grn.insert(0, 'Median curvature of green borders')
        sem_curvature_curveborders_list_grn.insert(0, 'SEM of mean curvature of green borders')
        std_curvature_curveborders_list_grn.insert(0, 'STD of mean curvature of green borders')

        all_curvatures_noncontactborders_list_grn.insert(0, 'All curvatures of green non-contact borders')
        mean_curvature_noncontactborders_list_grn.insert(0, 'Mean curvature of green non-contact borders')
        median_curvature_noncontactborders_list_grn.insert(0, 'Median curvature of green non-contact borders')
        sem_curvature_noncontactborders_list_grn.insert(0, 'SEM of mean curvature of green non-contact borders')
        std_curvature_noncontactborders_list_grn.insert(0, 'STD of mean curvature of green non-contact borders')
        
        all_curvatures_pathfinder_list_grn.insert(0, 'All curvatures of pixels used to make green contact vectors')
        mean_curvature_pathfinder_list_grn.insert(0, 'Mean curvature of pixels used to make green contact vectors')
        median_curvature_pathfinder_list_grn.insert(0, 'Median curvature of pixels used to make green contact vectors')
        sem_curvature_pathfinder_list_grn.insert(0, 'SEM of mean curvature of pixels used to make green contact vectors')
        std_curvature_pathfinder_list_grn.insert(0, 'STD of mean curvature of pixels used to make green contact vectors')
        
        all_curvatures_contactsurface_list_grn.insert(0, 'All curvatures of green contact diameter')        
        mean_curvature_contactsurface_list_grn.insert(0, 'Mean curvature of green contact diameter')
        median_curvature_contactsurface_list_grn.insert(0, 'Median curvature of green contact diameter')
        sem_curvature_contactsurface_list_grn.insert(0, 'SEM of mean curvature of green contact diameter')
        std_curvature_contactsurface_list_grn.insert(0, 'STD of mean curvature of green contact diameter')
        
        all_curvatures_pathfinder_list.insert(0, 'All curvatures of pixels used to make all contact vectors')
        mean_curvature_pathfinder_list.insert(0, 'Mean curvature of pixels used to make all contact vectors')
        median_curvature_pathfinder_list.insert(0, 'Median curvature of pixels used to make all contact vectors')
        sem_curvature_pathfinder_list.insert(0, 'SEM of mean curvature of pixels used to make all contact vectors')
        std_curvature_pathfinder_list.insert(0, 'STD of mean curvature of pixels used to make all contact vectors')

        all_curvatures_pathfinder_list_angle1.insert(0, 'All curvatures of pixels used to make angle1')
        mean_curvature_pathfinder_list_angle1.insert(0, 'Mean curvature of pixels used to make angle1')
        median_curvature_pathfinder_list_angle1.insert(0, 'Median curvature of pixels used to make angle1')
        sem_curvature_pathfinder_list_angle1.insert(0, 'SEM of mean curvature of pixels used to make angle1')
        std_curvature_pathfinder_list_angle1.insert(0, 'STD of mean curvature of pixels used to make angle1')
        
        all_curvatures_pathfinder_list_angle2.insert(0, 'All curvatures of pixels used to make angle2')
        mean_curvature_pathfinder_list_angle2.insert(0, 'Mean curvature of pixels used to make angle2')
        median_curvature_pathfinder_list_angle2.insert(0, 'Median curvature of pixels used to make angle2')
        sem_curvature_pathfinder_list_angle2.insert(0, 'SEM of mean curvature of pixels used to make angle2')
        std_curvature_pathfinder_list_angle2.insert(0, 'STD of mean curvature of pixels used to make angle2')
        
        all_curvatures_pathfinder_list_grn_angle1.insert(0, 'All curvatures of pixels used to make green angle1')
        mean_curvature_pathfinder_list_grn_angle1.insert(0, 'Mean curvature of pixels used to make green angle1')
        median_curvature_pathfinder_list_grn_angle1.insert(0, 'Median curvature of pixels used to make green angle1')
        sem_curvature_pathfinder_list_grn_angle1.insert(0, 'SEM of mean curvature of pixels used to make green angle1')
        std_curvature_pathfinder_list_grn_angle1.insert(0, 'STD of mean curvature of pixels used to make green angle1')
        
        all_curvatures_pathfinder_list_grn_angle2.insert(0, 'All curvatures of pixels used to make green angle2')
        mean_curvature_pathfinder_list_grn_angle2.insert(0, 'Mean curvature of pixels used to make green angle2')
        median_curvature_pathfinder_list_grn_angle2.insert(0, 'Median curvature of pixels used to maked green angle2')
        sem_curvature_pathfinder_list_grn_angle2.insert(0, 'SEM of mean curvature of pixels used to make green angle2')
        std_curvature_pathfinder_list_grn_angle2.insert(0, 'STD of mean curvature of pixels used to make green angle2')
        
        all_curvatures_pathfinder_list_red_angle1.insert(0, 'All curvatures of pixels used to make red angle1')
        mean_curvature_pathfinder_list_red_angle1.insert(0, 'Mean curvature of pixels used to make red angle1')
        median_curvature_pathfinder_list_red_angle1.insert(0, 'Median curvature of pixels used to make red angle1')
        sem_curvature_pathfinder_list_red_angle1.insert(0, 'SEM of mean curvature of pixels used to make red angle1')
        std_curvature_pathfinder_list_red_angle1.insert(0, 'STD of mean curvature of pixels used to make red angle1')
        
        all_curvatures_pathfinder_list_red_angle2.insert(0, 'All curvatures of pixels used to make red angle2')
        mean_curvature_pathfinder_list_red_angle2.insert(0, 'Mean curvature of pixels used to make red angle2')
        median_curvature_pathfinder_list_red_angle2.insert(0, 'Median curvature of pixels used to make red angle2')
        sem_curvature_pathfinder_list_red_angle2.insert(0, 'SEM of mean curvature of pixels used to make red angle2')
        std_curvature_pathfinder_list_red_angle2.insert(0, 'STD of mean curvature of pixels used to make red angle2')

        [bin_curv_val_red_list[i].insert(0, 'bin_%s_curv_val_red'%i) for i in list(range(num_bins))]
        [bin_curv_val_grn_list[i].insert(0, 'bin_%s_curv_val_grn'%i) for i in list(range(num_bins))]
        whole_curv_val_red_list.insert(0, 'whole_cell_curv_val_red')
        whole_curv_val_grn_list.insert(0, 'whole_cell_curv_val_grn')
        
        [bin_curv_residu_red_list[i].insert(0, 'bin_%s_curv_residu_red'%i) for i in list(range(num_bins))]
        [bin_curv_residu_grn_list[i].insert(0, 'bin_%s_curv_residu_grn'%i) for i in list(range(num_bins))]
        whole_curv_residu_red_list.insert(0, 'whole_cell_curv_residu_red')
        whole_curv_residu_grn_list.insert(0, 'whole_cell_curv_residu_grn')
        
        [bin_curv_arclen_red_list[i].insert(0, 'bin_%s_curv_arclen_red'%i) for i in list(range(num_bins))]
        [bin_curv_arclen_grn_list[i].insert(0, 'bin_%s_curv_arclen_grn'%i) for i in list(range(num_bins))]
        whole_curv_arclen_red_list.insert(0, 'whole_curv_arclen_red')
        whole_curv_arclen_grn_list.insert(0, 'whole_curv_arclen_grn')
        
        [bin_curv_arcanglerad_red_list[i].insert(0, 'bin_%s_curv_arcanglerad_red'%i) for i in list(range(num_bins))]
        [bin_curv_arcanglerad_grn_list[i].insert(0, 'bin_%s_curv_arcanglerad_grn'%i) for i in list(range(num_bins))]
        whole_curv_arcanglerad_red_list.insert(0, 'whole_curve_arcanglerad_red')
        whole_curv_arcanglerad_grn_list.insert(0, 'whole_curve_arcanglerad_grn')

        cells_touching_list.insert(0, 'cells touching?')
        
        script_version_list.insert(0, 'script_version')

        ### Table of fluorescence intensities:
        ### Headers:
        red_intensity_mean_headers = list(range(linescan_length))
        for i in red_intensity_mean_headers:
            red_intensity_mean_headers[i] = 'Mean red intensity at line position %i' %i
            
        grn_intensity_mean_headers = list(range(linescan_length))
        for i in grn_intensity_mean_headers:
            grn_intensity_mean_headers[i] = 'Mean green intensity at line position %i' %i
        
        master_red_linescan_mean.insert(0, red_intensity_mean_headers)
        master_grn_linescan_mean.insert(0, grn_intensity_mean_headers)
        #master_red_linescanx.insert(0, 'Red linescan x values')
        #master_red_linescany.insert(0, 'Red linescan y values')
        #master_grn_linescanx.insert(0, 'Green linescan x values')
        #master_grn_linescany.insert(0, 'Green linescan y values')
        
        ### For the intensity data:
        mean_intensity_table = []
        mean_intensity_table = np.c_[cell_pair_num_list, master_red_linescan_mean, master_grn_linescan_mean]
        
        redx_cellpair_num = []
        redy_cellpair_num = []
        grnx_cellpair_num = []
        grny_cellpair_num = []
        for i in range(filenumber):
            redx_cellpair_num.append('Cell pair %i red x linescan values' %(i+1))
            redy_cellpair_num.append('Cell pair %i red y linescan values' %(i+1))
            grnx_cellpair_num.append('Cell pair %i grn x linescan values' %(i+1))
            grny_cellpair_num.append('Cell pair %i grn y linescan values' %(i+1))
        for i,v in enumerate(redx_cellpair_num):
            master_red_linescanx[i].insert(0, v)
        for i,v in enumerate(redy_cellpair_num):
            master_red_linescany[i].insert(0, v)
        for i,v in enumerate(grnx_cellpair_num):
            master_grn_linescanx[i].insert(0, v)
        for i,v in enumerate(grny_cellpair_num):
            master_grn_linescany[i].insert(0, v)
        
        intensity_table = [v for i in zip(master_red_linescanx, master_red_linescany, master_grn_linescanx, master_grn_linescany) for v in i]

        ### Collect the elements in your lists so that the first element of each list
        ### is on the first row, the the second on the second, etc.:

        excel_table = list(zip( time_list, cell_pair_num_list,
                          angle_1_list, angle_2_list, average_angle_list,
                          red_halfangle1_list, red_halfangle2_list, 
                          grn_halfangle1_list, grn_halfangle2_list, 
                          contactsurface_list, noncontactsurface1_list, noncontactsurface2_list, 
                          cell1_perimeter_list, cell2_perimeter_list,
                          area_cell_1_list, area_cell_2_list, cell_pair_area_list,
                          corner_corner_length1_list, corner_corner_length2_list,
                          intersect_pos, intens_at_intersect, midpoint_intersect_fluor_values,
                          *bin_curv_val_red_list, whole_curv_val_red_list, *bin_curv_val_grn_list, whole_curv_val_grn_list,
                          *bin_curv_residu_red_list, whole_curv_residu_red_list, *bin_curv_residu_grn_list, whole_curv_residu_grn_list,
                          *bin_curv_arclen_red_list, whole_curv_arclen_red_list, *bin_curv_arclen_grn_list, whole_curv_arclen_grn_list,
                          *bin_curv_arcanglerad_red_list, whole_curv_arcanglerad_red_list, *bin_curv_arcanglerad_grn_list, whole_curv_arcanglerad_grn_list,
                          filename_list, cells_touching_list, mask_list,
                          Nth_pixel_list, linescan_length_list, curve_length_list,
                          disk_radius_list, otsu_list, otsu_fraction_list, 
                          background_threshold_list, object_of_interest_brightness_cutoff_percent_list, 
                          object_of_interest_brightness_cutoff_red_list, object_of_interest_brightness_cutoff_grn_list,
                          processing_time_list, red_gauss_list, grn_gauss_list, struct_elem_list,
                          redcellborders_coords, redcontactsurface_coords, rednoncontact_coords, 
                          grncellborders_coords, grncontactsurface_coords, grnnoncontact_coords, 
                          redcontactvector1_start, grncontactvector1_start, 
                          redcontactvector2_start, grncontactvector2_start, 
                          redcontactvector1, grncontactvector1,
                          redcontactvector2, grncontactvector2, 
                          contact_corner1_list, contact_corner2_list, 
                          red_maxima_coords, grn_maxima_coords,
                          contact_vectors_list, perpendicular2contact_vectors_list,
                          curvature_coords_red_list, curvature_coords_grn_list, 
                          numbered_border_indices_red_list, numbered_border_indices_grn_list, 
                          contactsurface_len_red_list, contactsurface_len_grn_list, 
                          curvature_values_red_list, mean_curvature_curveborders_list_red, median_curvature_curveborders_list_red, sem_curvature_curveborders_list_red, std_curvature_curveborders_list_red,
                          all_curvatures_noncontactborders_list_red, mean_curvature_noncontactborders_list_red, median_curvature_noncontactborders_list_red, sem_curvature_noncontactborders_list_red, std_curvature_noncontactborders_list_red,
                          all_curvatures_contactsurface_list_red, mean_curvature_contactsurface_list_red, median_curvature_contactsurface_list_red, sem_curvature_contactsurface_list_red, std_curvature_contactsurface_list_red,
                          curvature_values_grn_list, mean_curvature_curveborders_list_grn, median_curvature_curveborders_list_grn, sem_curvature_curveborders_list_grn, std_curvature_curveborders_list_grn,
                          all_curvatures_noncontactborders_list_grn, mean_curvature_noncontactborders_list_grn, median_curvature_noncontactborders_list_grn, sem_curvature_noncontactborders_list_grn, std_curvature_noncontactborders_list_grn,
                          all_curvatures_contactsurface_list_grn, mean_curvature_contactsurface_list_grn, median_curvature_contactsurface_list_grn, sem_curvature_contactsurface_list_grn, std_curvature_contactsurface_list_grn,
                          all_curvatures_pathfinder_list_red, mean_curvature_pathfinder_list_red, median_curvature_pathfinder_list_red, sem_curvature_pathfinder_list_red, std_curvature_pathfinder_list_red,
                          all_curvatures_pathfinder_list_grn, mean_curvature_pathfinder_list_grn, median_curvature_pathfinder_list_grn, sem_curvature_pathfinder_list_grn, std_curvature_pathfinder_list_grn,
                          all_curvatures_pathfinder_list_angle1, mean_curvature_pathfinder_list_angle1, median_curvature_pathfinder_list_angle1, sem_curvature_pathfinder_list_angle1, std_curvature_pathfinder_list_angle1,
                          all_curvatures_pathfinder_list_angle2, mean_curvature_pathfinder_list_angle2, median_curvature_pathfinder_list_angle2, sem_curvature_pathfinder_list_angle2, std_curvature_pathfinder_list_angle2,
                          all_curvatures_pathfinder_list, mean_curvature_pathfinder_list, median_curvature_pathfinder_list, sem_curvature_pathfinder_list, std_curvature_pathfinder_list,
                          all_curvatures_pathfinder_list_red_angle1, mean_curvature_pathfinder_list_red_angle1, median_curvature_pathfinder_list_red_angle1, sem_curvature_pathfinder_list_red_angle1, std_curvature_pathfinder_list_red_angle1,
                          all_curvatures_pathfinder_list_red_angle2, mean_curvature_pathfinder_list_red_angle2, median_curvature_pathfinder_list_red_angle2, sem_curvature_pathfinder_list_red_angle2, std_curvature_pathfinder_list_red_angle2,
                          all_curvatures_pathfinder_list_grn_angle1, mean_curvature_pathfinder_list_grn_angle1, median_curvature_pathfinder_list_grn_angle1, sem_curvature_pathfinder_list_grn_angle1, std_curvature_pathfinder_list_grn_angle1,
                          all_curvatures_pathfinder_list_grn_angle2, mean_curvature_pathfinder_list_grn_angle2, median_curvature_pathfinder_list_grn_angle2, sem_curvature_pathfinder_list_grn_angle2, std_curvature_pathfinder_list_grn_angle2,
                          script_version_list))

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
        myfile = input('What would you like to name your .csv file?\nThe file extension is not necessary.\n[default:%s]\n> ' %os.path.basename(path)) or os.path.basename(path)
        # myfile = os.path.basename(path)
        if os.path.exists(myfile+'.csv') == True:
            print ("That file already exists in the folder, would you like to"
                   " overwrite it?\nY (overwrite it) / N (don't overwrite it)"
                   )
            overwrite = input(">")
            # overwrite = 'y'
            if overwrite == 'y' or overwrite == 'Y' or overwrite == 'yes':
                pass
            
            elif overwrite == 'n' or overwrite == 'N' or overwrite == 'no':
                copy_count = 1
                myfile = [myfile, ' (', str(copy_count), ')']
                while os.path.exists(''.join(myfile)+'.csv') == True:
                    copy_count +=1
                    myfile[-2] = str(copy_count)
                    
            else:
                print("Sorry I didn't understand that, please tell me again:")
                continue
        myfile = ''.join(myfile)        
        ### Actually write the table in to a .csv file now:
        out = csv.writer(open(myfile+'.csv',"w",newline=''), dialect='excel')
        out.writerows(excel_table)
        excel_table_arr = np.array(excel_table, dtype=object)
#        np.save(path+'\\%s excel_table' %myfile, excel_table)
        ### "dialect='excel' allows excel to read the table correctly. If you don't 
        ### have that part then whats inside each tuple (the values in brackets in 
        ### excel_table) will show up in 1 cell and each tuple will show up in a new 
        ### column. If you use "w" (for write) instead of "wb" (for write buffer) then
        ### your excel file will have spaces between each row. "wb" gets rid of that.
        meanintensityfile = '%s mean intensity data' %myfile
        meanintensityfile = ''.join(meanintensityfile)
        out = csv.writer(open(meanintensityfile+'.csv',"w",newline=''), dialect='excel')
        out.writerows(mean_intensity_table)
        
        intensityfile = '%s intensity data' %myfile
        intensityfile = ''.join(intensityfile)
        out = csv.writer(open(intensityfile+'.csv',"w",newline=''), dialect='excel')
        out.writerows(intensity_table)
        
        intersectionfile = '%s intersection data' %myfile
        intersectionfile = ''.join(intersectionfile)
        out = csv.writer(open(intersectionfile+'.csv',"w",newline=''), dialect='excel')
        out.writerows(intersection_table)        


         
        multipdf()



        
        if os.path.exists(path+'\\%s Cell 1 curvature borders' %myfile) == False:
            cell1_curvature_pics = os.mkdir(path+'\\%s Cell 1 curvature borders' %myfile)
        if os.path.exists(path+'\\%s Cell 2 curvature borders' %myfile) == False:
            cell2_curvature_pics = os.mkdir(path+'\\%s Cell 2 curvature borders' %myfile)

        masked_curvatures1_array_all = np.dstack([x for x in curvatures1_protoarray])
        masked_curvatures2_array_all = np.dstack([x for x in curvatures2_protoarray])
        np.save(path+'\\%s masked_curvatures1_array_all' %myfile, masked_curvatures1_array_all)
        np.save(path+'\\%s masked_curvatures2_array_all' %myfile, masked_curvatures2_array_all)
        
        masked1_all_array = np.dstack([x for x in masked_curvatures1_protoarray])
        masked2_all_array = np.dstack([x for x in masked_curvatures2_protoarray])
        np.save(path+'\\%s masked1_all_array' %myfile, masked1_all_array)
        np.save(path+'\\%s masked2_all_array' %myfile, masked2_all_array)

        curveborders1_all_array = np.dstack([x for x in curveborders1_protoarray])
        curveborders2_all_array = np.dstack([x for x in curveborders2_protoarray])
        np.save(path+'\\%s curveborders1_all_array' %myfile, curveborders1_all_array)
        np.save(path+'\\%s curveborders2_all_array' %myfile, curveborders2_all_array)
        
        noncontactborders1_all_array = np.dstack([x for x in noncontactborders1_protoarray])
        noncontactborders2_all_array = np.dstack([x for x in noncontactborders2_protoarray])
        np.save(path+'\\%s noncontactborders1_all_array' %myfile, noncontactborders1_all_array)
        np.save(path+'\\%s noncontactborders2_all_array' %myfile, noncontactborders2_all_array)

        contactsurface1_all_array = np.dstack([x for x in contactsurface1_protoarray])
        contactsurface2_all_array = np.dstack([x for x in contactsurface2_protoarray])
        np.save(path+'\\%s contactsurface1_all_array' %myfile, contactsurface1_all_array)
        np.save(path+'\\%s contactsurface2_all_array' %myfile, contactsurface2_all_array)
        
        pathfinder1_all_array = np.dstack([x for x in pathfinder1_protoarray])
        pathfinder2_all_array = np.dstack([x for x in pathfinder2_protoarray])
        np.save(path+'\\%s pathfinder1_all_array' %myfile, pathfinder1_all_array)
        np.save(path+'\\%s pathfinder2_all_array' %myfile, pathfinder2_all_array)

        angles_angle_labeled_red_all_array = np.dstack([x for x in angles_angle_labeled_red_protoarray])
        angles_angle_labeled_grn_all_array = np.dstack([x for x in angles_angle_labeled_grn_protoarray])
        np.save(path+'\\%s angles_angle_labeled_red_all_array' %myfile, angles_angle_labeled_red_all_array)
        np.save(path+'\\%s angles_angle_labeled_grn_all_array' %myfile, angles_angle_labeled_grn_all_array)


        ### Save the red binned curvature estsimates in to a separate array file:       
        np.save(path+'\\%s bin_curvature_details_array_red' %myfile, bin_curvature_details_array_red)
        
        ### Save the grn binned curvature estsimates in to a separate array file:        
        np.save(path+'\\%s bin_curvature_details_array_grn' %myfile, bin_curvature_details_array_grn)

        ### Save the bin_indices_red and _grn as arrays they can be opened later:
        bin_indices_red_array = np.asarray(bin_indices_red_list, dtype='object')
        np.save(path+'\\%s bin_indices_red_array' %myfile, bin_indices_red_array)

        bin_indices_grn_array = np.asarray(bin_indices_grn_list, dtype='object')
        np.save(path+'\\%s bin_indices_grn_array' %myfile, bin_indices_grn_array)

        ### Save the bin_start_red_list and bin_stop_red_list:
        bin_start_red_list_array = np.asarray(bin_start_red_list, dtype='object')
        np.save(path+'\\%s bin_start_red_list_array' %myfile, bin_start_red_list_array)
        bin_stop_red_list_array = np.asarray(bin_stop_red_list, dtype='object')
        np.save(path+'\\%s bin_stop_red_list_array' %myfile, bin_stop_red_list_array)
        ### Do the same for the green:
        bin_start_grn_list_array = np.asarray(bin_start_grn_list, dtype='object')
        np.save(path+'\\%s bin_start_grn_list_array' %myfile, bin_start_grn_list_array)
        bin_stop_grn_list_array = np.asarray(bin_stop_grn_list, dtype='object')
        np.save(path+'\\%s bin_stop_grn_list_array' %myfile, bin_stop_grn_list_array)

        ### Save the whole-cell curvature estimate red in a separate array file:
        np.save(path+'\\%s whole_curvature_details_array_red' %myfile, whole_curvature_details_red_array)
        
        ### Save the whole-cell curvature estimate grn in a separate array file:
        np.save(path+'\\%s whole_curvature_details_array_grn' %myfile, whole_curvature_details_grn_array)



        masked_curvatures1_array_all = np.ma.array(masked_curvatures1_array_all, mask=masked1_all_array)
        masked_curvatures2_array_all = np.ma.array(masked_curvatures2_array_all, mask=masked2_all_array)

        curvatures1_max = np.max(masked_curvatures1_array_all)
        curvatures1_min = np.min(masked_curvatures1_array_all)

        curvatures2_max = np.max(masked_curvatures2_array_all)
        curvatures2_min = np.min(masked_curvatures2_array_all)
        
        
        curvatures_max = max([abs(curvatures1_max), abs(curvatures2_max), abs(curvatures1_min), abs(curvatures2_min)])
        curvatures_min = -1.0*curvatures_max

        for x in range(total_files):
            cmap = plt.cm.seismic
            cmap.set_bad(color='black')
            norm = plt.Normalize(vmin = curvatures_min, vmax = curvatures_max)
            image = cmap(norm(masked_curvatures1_array_all[:,:,x]))
            plt.imsave(path+'\\'+myfile+' Cell 1 curvature borders'+'\\masked_curvatures1_array_all_%s.tiff' %x, image)
            plt.close()

        for x in range(total_files):
            cmap = plt.cm.seismic
            cmap.set_bad(color='black')
            norm = plt.Normalize(vmin = curvatures_min, vmax = curvatures_max)
            image = cmap(norm(masked_curvatures2_array_all[:,:,x]))
            plt.imsave(path+'\\'+myfile+' Cell 2 curvature borders'+'\\masked_curvatures2_array_all_%s.tiff' %x, image)
            plt.close()

        break
    
    elif save_table == 'n' or save_table == 'N' or save_table == 'no':
        confirm_no_table_save = input("Are you sure you don't want to save your data as an excel table?"
                                          "\nY (I don't want to save it) / N (I changed my mind, let me save it!)\n>")
        # confirm_no_table_save = 'y'
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
    # finish = None
    break





### To Do:
    # - msg_list.append(list(various print lines and error codes if possible))

###############################################################################

### Miscellaneous:

#    plt.savefig(path+'//curvefit_red test.png', dpi=300)



### Below numbers the borders sequentially in a picture for easy visualization:
#def sequential_borders_as_array(picture_of_borders_of_interest_as_array, numbered_border_indices):
#    test = np.zeros((picture_of_borders_of_interest_as_array.shape))
#    for i in numbered_border_indices:
#        test[i[1]] = i[0]   
#    return test
#sequential_borders_red = sequential_borders_as_array(borders_without_closest_bin, renumbered_border_indices_red)
#sequential_borders_grn = sequential_borders_as_array(borders_without_closest_bin, renumbered_border_indices_grn)
#test = sequential_borders_red[:,:,0] + sequential_borders_grn[:,:,1]
#edges = feature.canny(newgrnfilt, low_threshold = mh.otsu(mh.stretch(newgrnfilt))/2, high_threshold = mh.otsu(mh.stretch(newgrnfilt)))
#plt.imshow(feature.canny(newgrnfilt)*(255-mh.stretch(mh.distance(newgrnthresh)))*newgrnthresh)
#canny = feature.canny(newgrnfilt)*(255-mh.stretch(mh.distance(newgrnthresh)))*newgrnthresh
#testsob = mh.stretch(mh.sobel(newgrnfilt, just_filter=True))
#testsob[np.where(canny)] = canny[np.where(canny)]
#plt.imshow(testsob)