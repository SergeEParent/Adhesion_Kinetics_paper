# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 16:56:48 2019

@author: Serge
"""

### This script is to simulate cross-sections of 2 perfect spheres with a 
### constant volume spreading on one another.

import numpy as np
#import seaborn as sns
from matplotlib import pyplot as plt
from scipy.ndimage.morphology import binary_fill_holes
#from skimage import morphology
from skimage import segmentation
from skimage import feature
from skimage.morphology import disk
import mahotas as mh
from skimage.draw import circle_perimeter, line, line_aa
from PIL import Image
import os



### The function below relates the contact angle "theta" to the ratio of the 
### contact_radius / cell_radius_initial, ie. the theoretical Contact Surface Length (N.D.)
def f(theta):
    """Takes an angle (in degrees) and returns a ratio of contact_radius / cell_radius_initial:"""
    return (4. * np.sin(np.deg2rad(theta))**3 / (4. - (1 - np.cos(np.deg2rad(theta)))**2 * (2 + np.cos(np.deg2rad(theta))) ))**(1/3)

### radius of the cell as a function of the original radius as the contact angle
### assuming constant volume:
def R_Vconst(R_init, theta):
    return ( 4.**(1/3) * R_init ) / ( 4 - (1 - np.cos(np.deg2rad(theta)))**2 * (2 + np.cos(np.deg2rad(theta))) )**(1/3)

### Normally rounding function will round 0.5 to the nearest even value
### but I want the expected behaviour of always being rounded up, so a function
### to do so is:
def simple_round(arr):
    return (np.array(arr)%1 >= 0.5).astype(int) + np.floor(arr)

def subpx2px(flt_tup): #can be 1D, 2D, or 3D; ie. (x), (x,y), or (x,y,z)
    floors = [np.floor(x) for x in flt_tup]
    ceilings = [np.ceil(x) for x in flt_tup]
    px_idx_arr = np.meshgrid(*list(zip(floors, ceilings)))
    px_idx_rc = list(zip(*[np.ravel(x).astype(int) for x in px_idx_arr]))
    
    wgt_flr = [1-x%1 for x in flt_tup]
    wgt_ceil = [x%1 for x in flt_tup]
    px_wgt_arr = np.meshgrid(*list(zip(wgt_flr, wgt_ceil)))
    px_weights = np.ravel(np.multiply.reduce(px_wgt_arr,axis=0)).tolist()

    return px_idx_rc, px_weights #Note that the weights are a bit off due to floating point rounding errors

### Define a function to draw circles that progressively move closer to each
### other but have their overlapping regions removed:
def draw_circles(circle1_center_c, circle1_center_r, circle2_center_c, circle2_center_r, radius_of_cell, circ_circ_intersec_pts, angle_theoretical, array_shape, circle1_channels, circle2_channels, filled_channels):
    img_list = list()
    dist_inv_list = list()
    dist_list = list()
    dist_flat_inv_list = list()
    dist_flat_list = list()
    peaks_list = list()
    cell_mask_list = list()
    seg_rgb_list = list()
    
    for i in range(len(angle_theoretical)):
        img = np.zeros(array_shape, dtype=bool)
        # use np.round(arr).astype(int) or int(float)?
        ### Round cell centers to whole numbers:
        ### We have to do it this way because the function used to draw circles (circle_perimeter)
        ### will only accept a single pixel as the circle center.
        circle1_center_r_as_int = simple_round(circle1_center_r).astype(int)[i] # int(circle1_center_r[i])
        circle1_center_c_as_int = simple_round(circle1_center_c).astype(int)[i] # int(circle1_center_c[i])
        circle2_center_r_as_int = simple_round(circle2_center_r).astype(int)[i] # int(circle2_center_r[i])
        circle2_center_c_as_int = simple_round(circle2_center_c).astype(int)[i] # int(circle2_center_c[i]+1)
        # Round the cell radii to whole numbers:
        radius_of_cell_as_int = simple_round(radius_of_cell).astype(int)[i] # np.round(radius_of_cell).astype(int)

        # Draw some circles in the appropriate channels:
        rr1, cc1 = circle_perimeter(circle1_center_r_as_int, circle1_center_c_as_int, radius_of_cell_as_int)
        rr2, cc2 = circle_perimeter(circle2_center_r_as_int, circle2_center_c_as_int, radius_of_cell_as_int)

        for channel in circle1_channels:
            img[rr1, cc1, channel] = True
        for channel in circle2_channels:
            img[rr2, cc2, channel] = True
        for channel in filled_channels:
            img[:,:,channel] = binary_fill_holes(img[:,:,channel])


        inters1, inters2 = circ_circ_intersec_pts
        if np.all(np.array([[x.dtype for x in inters1],
                            [x.dtype for x in inters2]]) == float):
            # Round circle-circle intersection points to whole numbers:
            # inters1_r_as_int, inters1_c_as_int = simple_round(inters1).astype(int) # np.round(inters1).astype(int)
            # inters2_r_as_int, inters2_c_as_int = simple_round(inters2).astype(int) # np.round(inters2).astype(int)
            # inters1_r, inters1_c = inters1
            # inters2_r, inters2_c = inters2
            inters1_px, inters1_wgt = subpx2px(list(zip(*inters1))[i])
            inters2_px, inters2_wgt = subpx2px(list(zip(*inters2))[i])
            
            # Create a rasterized line between the intersection points:
            # line_r, line_c = line(inters1_r_as_int[i], inters1_c_as_int[i], inters2_r_as_int[i], inters2_c_as_int[i])
            
            # inters1_r, inters1_c = [np.asarray(x) for x in list(zip(*inters1_px))]
            # inters2_r, inters2_c = [np.asarray(x) for x in list(zip(*inters2_px))]
            line_pts = [np.ravel(x) for x in list(zip(*[inters1_px, inters2_px]))]
            lines = [line(*x) for x in line_pts]
            # lines = [line_aa(*x) for x in line_pts]
            
        elif np.all(np.array([[x.dtype for x in inters1],
                              [x.dtype for x in inters2]]) == 'O'):
            # line_r, line_c = np.array([], dtype=int), np.array([], dtype=int)
            
            # inters1_px, inters1_wgt = list(), list()
            # inters2_px, inters2_wgt = list(), list()
            
            line_pts = list()
            lines = list()
        else:
            raise TypeError('Intersection points are not integers or NoneType')
        
        img = img.astype(int)
        img_list.append(img)

        if i <= 90:
            cell_mask = np.logical_or(*[img[...,channel] for channel in (circle1_channels, circle2_channels)])#.squeeze()
        elif i > 90: # Please note that I haven't tested this for angles greater than 90
            cell_mask = np.logical_and(*[img[...,channel] for channel in (circle1_channels, circle2_channels)])#.squeeze()
        else:
            raise TypeError('Check that the theoretical angle is a number.')
        
        # Use a watershed algorithm to separate the 2 circles, using the 
        # intersection line as a boundary:
        dist = np.copy(img)#.astype(bool)
        # Draw the line in to the img:
        line_rast =  np.zeros(img.shape, dtype=bool)
        # line_rast =  np.zeros(img.shape, dtype=float)
        for (r,c) in lines:
            line_rast[r,c,:2] = True
        # for (r,c,val) in lines:
        #     line_rast[r,c,:2] = line_rast[r,c,:2] + np.array(val, ndmin=line_rast[r,c,:2].ndim).T
        # line_rast /= len(lines)
        # line_rast = line_rast.astype(bool)
        # line_rast[line_r, line_c,:2] = True
        dist = dist * ~line_rast
        dist = dist * cell_mask
        ### Create a 2D structuring element with connectivity-8 for labeling regions:
        struc_elem = np.dstack(( np.zeros((3,3)), disk(1), np.zeros((3,3)) ))
        ### Label the circle regions:
        dist_labeled = mh.labeled.label(dist, Bc=struc_elem)[0]
        ### Return theoretical circle centers as pixels:
        circ1px_idx, circ1px_wgt = subpx2px((circle1_center_r[i], circle1_center_c[i]))
        circ1_r, circ1_c = [np.asarray(idx) for idx in list(zip(*circ1px_idx))]
        circ2px_idx, circ2px_wgt = subpx2px((circle2_center_r[i], circle2_center_c[i]))
        circ2_r, circ2_c = [np.asarray(idx) for idx in list(zip(*circ2px_idx))]
        ### Return nonzero region label that contains the centers as px:
        circ1_label = np.unique(dist_labeled[circ1_r, circ1_c, circle1_channels][dist_labeled[circ1_r, circ1_c, circle1_channels] != 0])
        x = 0
        while True:
            if circ1_label.any() == False:
                x += 1
                circ1px_idx, circ1px_wgt = subpx2px((circle1_center_r[i-x], circle1_center_c[i-x]))
                circ1_r, circ1_c = [np.asarray(idx) for idx in list(zip(*circ1px_idx))]
                circ1_label = np.unique(dist_labeled[circ1_r, circ1_c, circle1_channels][dist_labeled[circ1_r, circ1_c, circle1_channels] != 0])
            else:
                break
        circ2_label = np.unique(dist_labeled[circ2_r, circ2_c, circle2_channels][dist_labeled[circ2_r, circ2_c, circle2_channels] != 0])
        x = 0
        while True:
            if circ2_label.any() == False:
                x += 1
                circ2px_idx, circ2px_wgt = subpx2px((circle2_center_r[i-x], circle2_center_c[i-x]))
                circ2_r, circ2_c = [np.asarray(idx) for idx in list(zip(*circ2px_idx))]
                circ2_label = np.unique(dist_labeled[circ2_r, circ2_c, circle2_channels][dist_labeled[circ2_r, circ2_c, circle2_channels] != 0])
            else:
                break
        ### Take the region that contains the circle center (only works for theoretical angle <= 90deg):
        dist[:,:,circle1_channels] = dist_labeled[:,:,circle1_channels] == circ1_label
        dist[:,:,circle2_channels] = dist_labeled[:,:,circle2_channels] == circ2_label


        ### Start the segmentation process:        
        for channel in circle1_channels + circle2_channels:
            # dist[:,:,channel] = binary_fill_holes(img[:,:,channel])
            dist[:,:,channel] = mh.distance(dist[:,:,channel])
        dist_flat = np.max(dist, axis=2)
        dist_flat_inv = dist_flat.max() - dist_flat
        dist_inv = np.copy(img)
        for channel in circle1_channels + circle2_channels:
            dist_inv[:,:,channel] = dist[:,:,channel].max() - dist[:,:,channel]
        dist_inv_list.append(dist_inv)
        dist_list.append(dist)
        dist_flat_list.append(dist_flat)
        dist_flat_inv_list.append(dist_flat_inv)
    
        peaks = np.zeros(array_shape)
        for channel in circle1_channels + circle2_channels:
            # peaks[:,:,channel] = feature.peak_local_max(dist[:,:,channel], min_distance=radius_of_cell_as_int, indices=False)
            # peaks_locs = feature.peak_local_max(dist[:,:,channel], min_distance=int(radius_of_cell_as_int*0.2), indices=True)
            peaks_locs = feature.peak_local_max(dist[:,:,channel], min_distance=radius_of_cell_as_int, indices=True)
            peaks_r, peaks_c = simple_round(np.mean(peaks_locs, axis=0)).astype(int)
            peaks[peaks_r, peaks_c, channel] = True
        peaks_list.append(peaks)
            
        labels = np.copy(peaks).astype(int)
        for channel in circle1_channels + circle2_channels:
            labels[:,:,channel] *= (channel + 1)
        labels_flat = np.max(np.zeros(array_shape, dtype=int), axis=2)
        if circle1_channels:
            labels_flat += labels[:,:,circle1_channels[0]]
        if circle2_channels:
            labels_flat += labels[:,:,circle2_channels[0]]


        cell_mask = cell_mask.squeeze()
        seg = segmentation.watershed(dist_flat_inv, labels_flat, compactness=0, connectivity=2, mask=cell_mask)
        seg = np.add.reduce([mh.borders(seg==region) * (seg==region) * seg for region in np.unique(seg[seg!=0])])
        
        circle1_seg_val = np.unique(labels[:,:,circle1_channels[0]][labels[:,:,circle1_channels[0]] != 0])
        circle2_seg_val = np.unique(labels[:,:,circle2_channels[0]][labels[:,:,circle2_channels[0]] != 0])

        seg_rgb = np.zeros(array_shape, dtype=np.uint8)
        for channel in circle1_channels:
            seg_rgb[:,:,channel] = mh.stretch(seg == circle1_seg_val)
        for channel in circle2_channels:
            seg_rgb[:,:,channel] = mh.stretch(seg == circle2_seg_val)
            
        for channel in filled_channels:
            seg_rgb[:,:,channel] = mh.stretch(binary_fill_holes(seg_rgb[:,:,channel]))
            
        seg_rgb_list.append(seg_rgb)
        cell_mask_list.append(cell_mask)

    return seg_rgb_list, cell_mask_list
    




### Define some base variables:
radius_initial = 27 #27 used for ecto-ecto, 30 used for ecto-ecto 1xMBS_LifeAct
# What RGB channels do you want to draw circles in? (An RGB image corresponds 
# to an M x N x 3 array, where the third dimension can be either 0,1,2 for
# red, green, blue respectively.)
circle1_channels = [0]
circle2_channels = [1]
filled_channels = [0,1]
# The top 3 channels simulate a cell with a membrane marker in the red and grn
# channels, a cytoplasmic marker in the blu channel, and an initial cell radius
# of 30 pixels.
array_shape = (radius_initial*5, radius_initial*5, 3)
angle_theoretical = np.linspace(0, 90, 91)

approach_angle = [0, 15, 30, 45] #[0] #angle is the intersection between the line perpendicular to the contact plane and the x-axis (ie. the row indices)
## Going to want to do approach angle of [0, 15, 30, 45] to start with.

for appr_ang in approach_angle:

    radius_of_cell = R_Vconst(radius_initial, angle_theoretical) #this radius will be used to draw circles
    
    center_to_contact_distance = radius_of_cell * np.cos(np.deg2rad(angle_theoretical))
    
    # Construct circles that have the adhesiveness specified by angle_theoretical
    # and the radius of radius_of_cell_as_int.
    # to do this you need to find the center points of each circle 
    # to find that you need to find the center-to-center distance between each
    # circle
    circle1_center_c = array_shape[1] / 2 - center_to_contact_distance * np.cos(np.deg2rad(appr_ang))
    circle1_center_r = array_shape[0] / 2 - center_to_contact_distance * np.sin(np.deg2rad(appr_ang))
    circle2_center_c = array_shape[1] / 2 + center_to_contact_distance * np.cos(np.deg2rad(appr_ang))
    circle2_center_r = array_shape[0] / 2 + center_to_contact_distance * np.sin(np.deg2rad(appr_ang))
    
    # Get the line of intersection of the circles:
    # Find where line intersects circles: https://mathworld.wolfram.com/Circle-CircleIntersection.html
    d = 2 * center_to_contact_distance #center-to-center distance between the 2 circles
    R = radius_of_cell.copy() #radius of cell1
    r = radius_of_cell.copy() #radius of cell2
    a = 1 / d * np.sqrt(4 * d**2 * R**2 - (d**2 - r**2 + R**2)**2) # chord length of circle-circle intersection; ie. contact length
    dx = a / 2 * np.sin(np.deg2rad(appr_ang)) # shift in col location of point on line due to approach angle
    dy = a / 2 * np.cos(np.deg2rad(appr_ang)) # shift in row location of point on line due to approach angle
    
    
    line_inters1_c = array_shape[1] / 2 - dx #col location of a point on the line
    line_inters1_r = array_shape[0] / 2 + dy #row location of a point on the line
    line_inters2_c = array_shape[1] / 2 + dx #col location of another point on the line
    line_inters2_r = array_shape[0] / 2 - dy #col location of another point on the line
    
    circ_circ_intersec_pts = ((line_inters1_r, line_inters1_c), 
                              (line_inters2_r, line_inters2_c))
    
    # Draw the circles:
    cell_spreading_sim, cell_mask_list = draw_circles(circle1_center_c, circle1_center_r, circle2_center_c, circle2_center_r, radius_of_cell, circ_circ_intersec_pts, angle_theoretical, array_shape, circle1_channels, circle2_channels, filled_channels)

    
    
    # Save the resulting images:
    if os.path.exists(os.getcwd()+'\\approach_%.1fdeg'%appr_ang) == False:
        os.mkdir(os.getcwd()+'\\approach_%.1fdeg'%appr_ang)
    else:
        pass
    
    for i,sim in enumerate(cell_spreading_sim):
        circles_img = Image.fromarray(sim)
        circles_img.save(os.getcwd()+'\\approach_%.1fdeg'%appr_ang+'\\2 circles angle %03d.tif' %angle_theoretical[i])
        #plt.imshow(x)
        #plt.show()
    
    if os.path.exists(os.getcwd()+'\\approach_%.1fdeg'%appr_ang+'\\masks') == False:
        os.mkdir(os.getcwd()+'\\approach_%.1fdeg'%appr_ang+'\\masks')
    else:
        pass
    
    for i,mask in enumerate(cell_mask_list):
        tl_sim_img = Image.fromarray(mask)
        tl_sim_img.save(os.getcwd()+'\\approach_%.1fdeg'%appr_ang+'\\masks\\2 circles angle %03d.tif' %angle_theoretical[i])
    
    
    

    # Also construct circles that are separated by 2 and 1 pixels:
    circ_circ_intersec_pts = ((np.array(None), np.array(None)), 
                              (np.array(None), np.array(None)))

    circle1_center_c_sep2px = np.asarray([circle1_center_c[0] - 1])
    circle1_center_r_sep2px = np.asarray([circle1_center_r[0]])
    circle2_center_c_sep2px = np.asarray([circle2_center_c[0] + 1])
    circle2_center_r_sep2px = np.asarray([circle2_center_r[0]])
    radius_of_cell_sep2px = np.asarray([radius_of_cell[0]])
    angle_theoretical_sep2px = np.asarray([angle_theoretical[0]])
    cell_spreading_sep2px, cell_mask_list_sep2px = draw_circles(circle1_center_c_sep2px, circle1_center_r_sep2px, circle2_center_c_sep2px, circle2_center_r_sep2px, radius_of_cell_sep2px, circ_circ_intersec_pts, angle_theoretical_sep2px, array_shape, circle1_channels, circle2_channels, filled_channels)
    for i,sim in enumerate(cell_spreading_sep2px):
        circles_img = Image.fromarray(sim)
        circles_img.save(os.getcwd()+'\\approach_%.1fdeg'%appr_ang+'\\2 circles angle 0.tif')
    for i,mask in enumerate(cell_mask_list_sep2px):
        tl_sim_img = Image.fromarray(mask)
        tl_sim_img.save(os.getcwd()+'\\approach_%.1fdeg'%appr_ang+'\\masks\\2 circles angle 0.tif')
    
    
    circle1_center_c_sep1px = np.asarray([circle1_center_c[0] - 1])
    circle1_center_r_sep1px = np.asarray([circle1_center_r[0]])
    circle2_center_c_sep1px = np.asarray([circle2_center_c[0]])
    circle2_center_r_sep1px = np.asarray([circle2_center_r[0]])
    radius_of_cell_sep1px = np.asarray([radius_of_cell[0]])
    angle_theoretical_sep1px = np.asarray([angle_theoretical[0]])
    cell_spreading_sep1px, cell_mask_list_sep1px = draw_circles(circle1_center_c_sep1px, circle1_center_r_sep1px, circle2_center_c_sep1px, circle2_center_r_sep1px, radius_of_cell_sep1px, circ_circ_intersec_pts, angle_theoretical_sep1px, array_shape, circle1_channels, circle2_channels, filled_channels)
    for i,sim in enumerate(cell_spreading_sep1px):
        circles_img = Image.fromarray(sim)
        circles_img.save(os.getcwd()+'\\approach_%.1fdeg'%appr_ang+'\\2 circles angle 00.tif')
    for i,mask in enumerate(cell_mask_list_sep1px):
        tl_sim_img = Image.fromarray(mask)
        tl_sim_img.save(os.getcwd()+'\\approach_%.1fdeg'%appr_ang+'\\masks\\2 circles angle 00.tif')





### Stuff below is for out of my own curiosity/for sim validation:

### This shows the overlay of the circle-circle interserction points with the 
### pixelated images:
# for i,img in enumerate(cell_spreading_sim):
#     fig, ax = plt.subplots()
#     ax.imshow(img)
#     ax.plot(int(line_inters1_c[i]), int(line_inters1_r[i]), marker='.')
#     ax.plot(int(line_inters2_c[i]), int(line_inters2_r[i]), marker='.')
#     # plt.gca().invert_yaxis()
#     ax.set_ylim(array_shape[1], 0)
#     ax.set_xlim(0, array_shape[0])
#     ax.set_aspect('equal')
#     ax.set_adjustable('box')
#     plt.show()

### This shows the overlay of the circle-circle interserction points with the 
### theoretical circles:
# for i,img in enumerate(cell_spreading_sim):
#     fig, ax = plt.subplots()
#     circle1 = plt.Circle((circle1_center_c[i], circle1_center_r[i]), radius_of_cell[i], color='g', fill=False)
#     circle2 = plt.Circle((circle2_center_c[i], circle2_center_r[i]), radius_of_cell[i], color='g', fill=False)
#     ax.plot(line_inters1_c[i], line_inters1_r[i], marker='.')
#     ax.plot(line_inters2_c[i], line_inters2_r[i], marker='.')
#     # plt.gca().invert_yaxis()
#     ax.set_ylim(array_shape[1], 0)
#     ax.set_xlim(0, array_shape[0])
#     ax.set_aspect('equal')
#     ax.set_adjustable('box')
    
#     ax.add_patch(circle1)
#     ax.add_patch(circle2)
#     plt.show()












