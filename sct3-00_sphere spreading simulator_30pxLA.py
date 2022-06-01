# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 16:56:48 2019

@author: Serge
"""

### This script is to simulate cross-sections of 2 perfect spheres with a 
### constant volume spreading on one another.

import numpy as np
#import seaborn as sns
#from matplotlib import pyplot as plt
from scipy.ndimage.morphology import binary_fill_holes
#from skimage import morphology
from skimage import segmentation
from skimage import feature
import mahotas as mh
from skimage.draw import circle_perimeter
from PIL import Image
import os



### The function below relates the contact angle "theta" to the ratio of the 
### contact_radius / cell_radius_initial, ie. the theoretical Contact Surface Length (N.D.)
def f(theta):
    """Takes an angle (in degrees) and returns a ratio of contact_radius / cell_radius_initial:"""
    return (4. * np.sin(np.deg2rad(theta))**3 / (4. - (1 - np.cos(np.deg2rad(theta)))**2 * (2 + np.cos(np.deg2rad(theta))) ))**(1/3)

### radius of the cell as a function of the original radius as the contact angle
### assuming constant volume:
def R(R_init, theta):
    return ( 4.**(1/3) * R_init ) / ( 4 - (1 - np.cos(np.deg2rad(theta)))**2 * (2 + np.cos(np.deg2rad(theta))) )**(1/3)

### Define a function to draw circles that progressively move closer to each
### other but have their overlapping regions removed:
def draw_circles(circle1_center_c, circle1_center_r, circle2_center_c, circle2_center_r, radius_of_cell_as_int, angle_theoretical, array_shape, circle1_channels, circle2_channels, filled_channels):
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
        rr1, cc1 = circle_perimeter(int(circle1_center_r[i]), int(circle1_center_c[i]), radius_of_cell_as_int[i])
        rr2, cc2 = circle_perimeter(int(circle2_center_r[i]), int(circle2_center_c[i]+1), radius_of_cell_as_int[i])
        for channel in circle1_channels:
            img[rr1, cc1, channel] = True
        for channel in circle2_channels:
            img[rr2, cc2, channel] = True
        for channel in filled_channels:
            img[:,:,channel] = binary_fill_holes(img[:,:,channel])

        img = img.astype(int)
        img_list.append(img)
        cell_mask = np.max(np.dstack([binary_fill_holes(img[...,channel]) 
                                      for channel in range(img.shape[-1])]),
                           axis=2)
        cell_mask_list.append(cell_mask)
        
        dist = np.copy(img)
        for channel in circle1_channels + circle2_channels:
            dist[:,:,channel] = binary_fill_holes(img[:,:,channel])
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
            peaks[:,:,channel] = feature.peak_local_max(dist[:,:,channel], indices=False)
            # peaks[:,:,1] = feature.peak_local_max(dist[:,:,1], indices=False)
        peaks_list.append(peaks)
            
        labels = np.copy(peaks).astype(int)
        for channel in circle1_channels + circle2_channels:
            labels[:,:,channel] *= (channel + 1)
        labels_flat = np.max(np.zeros(array_shape, dtype=int), axis=2)
        if circle1_channels:
            labels_flat += labels[:,:,circle1_channels[0]]
        if circle2_channels:
            labels_flat += labels[:,:,circle2_channels[0]]


        seg = segmentation.watershed(dist_flat_inv, labels_flat, mask=cell_mask)
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

    return seg_rgb_list, cell_mask_list
    




### Define some base variables:
radius_initial = 30 #27 used for ecto-ecto, 30 used for ecto-ecto 1xMBS_LifeAct
# What RGB channels do you want to draw circles in? (An RGB image corresponds 
# to an M x N x 3 array, where the third dimension can be either 0,1,2 for
# red, green, blue respectively.)
circle1_channels = [0,1]
circle2_channels = [2]
filled_channels = [2]
# The top 3 channels simulate a cell with a membrane marker in the red and grn
# channels, a cytoplasmic marker in the blu channel, and an initial cell radius
# of 30 pixels.
array_shape = (radius_initial*3, radius_initial*5, 3)
angle_theoretical = np.linspace(0, 90, 91)


### Start the script proper:
#cont_radius_theoretical = f(angle_theoretical)
radius_of_cell = R(radius_initial, angle_theoretical) #this radius will be used to draw circles
#plt.plot(radius_of_cell)

# Round the cell radii to whole numbers:
radius_of_cell_as_int = np.round(radius_of_cell).astype(int)
# angle_theoretical = np.linspace(0, 90, 91)
# contact_radius = radius_of_cell * np.sin(np.deg2rad(angle_theoretical))
center_to_contact_distance = radius_of_cell * np.cos(np.deg2rad(angle_theoretical))

# Construct circles that have the adhesiveness specified by angle_theoretical
# and the radius of radius_of_cell_as_int.
# to do this you need to find the center points of each circle 
# to find that you need to find the center-to-center distance between each
# circle
circle1_center_c = array_shape[1] / 2 - center_to_contact_distance
circle1_center_r = np.array(array_shape[0] / 2).repeat(len(circle1_center_c))
circle2_center_c = array_shape[1] / 2 + center_to_contact_distance
circle2_center_r = np.array(array_shape[0] / 2).repeat(len(circle2_center_c))

# Draw the circles:
cell_spreading_sim, cell_mask_list = draw_circles(circle1_center_c, circle1_center_r, circle2_center_c, circle2_center_r, radius_of_cell_as_int, angle_theoretical, array_shape, circle1_channels, circle2_channels, filled_channels)

# Save the resulting images:
for i,sim in enumerate(cell_spreading_sim):
    circles_img = Image.fromarray(sim)
    circles_img.save(os.getcwd()+'\\2 circles angle %03d.tif' %angle_theoretical[i])
    #plt.imshow(x)
    #plt.show()

if os.path.exists(os.getcwd()+'\\masks') == False:
    os.mkdir(os.getcwd()+'\\masks')
else:
    pass

for i,mask in enumerate(cell_mask_list):
    tl_sim_img = Image.fromarray(mask)
    tl_sim_img.save(os.getcwd()+'\\masks\\2 circles angle %03d.tif' %angle_theoretical[i])




# Also construct circles that are separated by 2 and 1 pixels:
circle1_center_c_sep2px = np.asarray([circle1_center_c[0] - 1])
circle1_center_r_sep2px = np.asarray([circle1_center_r[0]])
circle2_center_c_sep2px = np.asarray([circle2_center_c[0] + 1])
circle2_center_r_sep2px = np.asarray([circle2_center_r[0]])
radius_of_cell_as_int_sep2px = np.asarray([radius_of_cell_as_int[0]])
angle_theoretical_sep2px = np.asarray([angle_theoretical[0]])
cell_spreading_sep2px, cell_mask_list_sep2px = draw_circles(circle1_center_c_sep2px, circle1_center_r_sep2px, circle2_center_c_sep2px, circle2_center_r_sep2px, radius_of_cell_as_int_sep2px, angle_theoretical_sep2px, array_shape, circle1_channels, circle2_channels, filled_channels)
for i,sim in enumerate(cell_spreading_sep2px):
    circles_img = Image.fromarray(sim)
    circles_img.save(os.getcwd()+'\\2 circles angle -002.tif')
for i,mask in enumerate(cell_mask_list_sep2px):
    tl_sim_img = Image.fromarray(mask)
    tl_sim_img.save(os.getcwd()+'\\masks\\2 circles angle -002.tif')


circle1_center_c_sep1px = np.asarray([circle1_center_c[0] - 1])
circle1_center_r_sep1px = np.asarray([circle1_center_r[0]])
circle2_center_c_sep1px = np.asarray([circle2_center_c[0]])
circle2_center_r_sep1px = np.asarray([circle2_center_r[0]])
radius_of_cell_as_int_sep1px = np.asarray([radius_of_cell_as_int[0]])
angle_theoretical_sep1px = np.asarray([angle_theoretical[0]])
cell_spreading_sep1px, cell_mask_list_sep1px = draw_circles(circle1_center_c_sep1px, circle1_center_r_sep1px, circle2_center_c_sep1px, circle2_center_r_sep1px, radius_of_cell_as_int_sep1px, angle_theoretical_sep1px, array_shape, circle1_channels, circle2_channels, filled_channels)
for i,sim in enumerate(cell_spreading_sep1px):
    circles_img = Image.fromarray(sim)
    circles_img.save(os.getcwd()+'\\2 circles angle -001.tif')
for i,mask in enumerate(cell_mask_list_sep1px):
    tl_sim_img = Image.fromarray(mask)
    tl_sim_img.save(os.getcwd()+'\\masks\\2 circles angle -001.tif')



