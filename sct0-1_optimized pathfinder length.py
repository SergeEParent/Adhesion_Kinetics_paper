# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 18:27:46 2018

@author: Serge
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
import os

cell_radius = 40.0
pathfinder_length = np.linspace(2, 181, 180 )


### The function describing the loss of accuracy with increasing pathfinder 
### length:
half_angle = -1.0*np.rad2deg(np.arctan( (np.cos(pathfinder_length / cell_radius) - 1.0) / (np.sin(pathfinder_length / cell_radius)) ))

### The function describing the gain of accuracy with increasing pathfinder
### length:
resolution_angle = np.rad2deg(np.arctan( 1.0 / (pathfinder_length - 1.0) ))

total_measured_angle_error = abs( resolution_angle - half_angle )
min_measured_angle_error = min(total_measured_angle_error)
optimal_pathfinder_length = pathfinder_length[np.where(total_measured_angle_error == min_measured_angle_error)]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(pathfinder_length, half_angle, marker='.', c='b')
ax.plot(pathfinder_length, resolution_angle, marker='.', c='g')
ax.plot(pathfinder_length, total_measured_angle_error, marker='.', c='r')
ax.scatter(optimal_pathfinder_length, min_measured_angle_error, marker = '.', color='k', zorder=10)
ml = MultipleLocator(1)
ax.xaxis.set_minor_locator(ml)
ax.axis((0,20,0,45))
plt.draw
plt.show('block'==False)
###############################################################################
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator

#cell_radius_var = np.linspace(10, 189, 180)
#pathfinder_length_var = np.linspace(2, 181, 180 )

cell_radius_var = np.linspace(10, 60, 51)
pathfinder_length_var = np.linspace(2, 20, 19 )
cell_area_var = np.pi*cell_radius_var**2

pathfinder_length_grid, cell_radius_grid = np.meshgrid(pathfinder_length_var, cell_radius_var)
pathfinder_length_grid, cell_area_grid = np.meshgrid(pathfinder_length_var, cell_area_var)


half_angle_grid = -1.0*np.rad2deg(np.arctan( (np.cos(pathfinder_length_grid / cell_radius_grid) - 1.0) / (np.sin(pathfinder_length_grid / cell_radius_grid)) ))
resolution_angle_grid = np.rad2deg(np.arctan( 1.0 / (pathfinder_length_grid - 1.0) ))

total_measured_angle_error_grid = abs( resolution_angle_grid - half_angle_grid )

min_measured_angle_error_list = np.min(total_measured_angle_error_grid, axis=1)
min_measured_angle_error_grid = np.expand_dims(min_measured_angle_error_list, axis=1)
optimal_pathfinder_length_list = pathfinder_length_grid[np.where(total_measured_angle_error_grid == min_measured_angle_error_grid)]

###
fig2 = plt.figure()
ax2 = fig2.gca(projection='3d')
###ax2.plot_surface(pathfinder_length_grid, cell_radius_grid, resolution_angle_grid, color='g')
###ax2.plot_wireframe(pathfinder_length_grid, cell_radius_grid, half_angle_grid, color='b')
surf = ax2.plot_surface(pathfinder_length_grid, cell_radius_grid, total_measured_angle_error_grid, cmap=cm.coolwarm, linewidth=0, antialiased=False, rstride=1, cstride=1)
line = ax2.plot(optimal_pathfinder_length_list, cell_radius_var, min_measured_angle_error_list, color='k')

### Customize the axes.
ax2.set_zlim(0, 45)
ax2.set_xlim(0, 20)
ax2.set_ylim(10, 60)

ax2.view_init(azim=270, elev=90)
### Set axis titles.
ax2.set_xlabel('pathfinder_length_grid')
ax2.set_ylabel('cell_radius_grid')
ax2.set_zlabel('total_measured_angle_error_grid')

### Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show('block'==False)
###

fig3 = plt.figure(figsize=(6,3))
ax3 = fig3.add_subplot(111)

heatmap = ax3.pcolor(cell_radius_grid, pathfinder_length_grid, total_measured_angle_error_grid, cmap=cm.viridis, edgecolors='grey', linewidths=0.5)
### or
#heatmap = ax3.pcolor(cell_area_grid, pathfinder_length_grid, total_measured_angle_error_grid, cmap=cm.viridis, edgecolors='grey', linewidths=0.5)
error_mins = ax3.plot(cell_radius_var, optimal_pathfinder_length_list, color='r', linewidth=1)
### or
#error_mins = ax3.plot(cell_area_var, optimal_pathfinder_length_list, color='r', linewidth=1)

ax3.set_xlim(10, 60)
unique_optimal_pathfinder_values = np.unique(optimal_pathfinder_length_list)
unique_optimal_pathfinder_value_indices = [np.where(optimal_pathfinder_length_list == x) for x in unique_optimal_pathfinder_values]
first_optimal_pathfinder_value_index = [x[0][0] for x in unique_optimal_pathfinder_value_indices]
last_optimal_pathfinder_value_index = [x[0][-1] for x in unique_optimal_pathfinder_value_indices]
first_optimal_pathfinder_radius = cell_radius_var[first_optimal_pathfinder_value_index]
last_optimal_pathfinder_radius = cell_radius_var[last_optimal_pathfinder_value_index]
first_optimal_pathfinder_area = cell_area_var[first_optimal_pathfinder_value_index]
last_optimal_pathfinder_area = cell_area_var[last_optimal_pathfinder_value_index]
zip(unique_optimal_pathfinder_values, first_optimal_pathfinder_area.round(), last_optimal_pathfinder_area.round())

##cell_area_var_list = [np.pi*10**2, np.pi*20**2, np.pi*30**2, np.pi*40**2, np.pi*50**2, np.pi*60**2]
##cell_area_var_list = ['%.d'%x for x in cell_area_var_list]
first_optimal_pathfinder_area_labels = ['%.d'%x for x in first_optimal_pathfinder_area]
### To show areas instead
ax3.set_xticks(list(first_optimal_pathfinder_radius))
ax3.set_xticklabels(first_optimal_pathfinder_area_labels, rotation=45, ha='right')
##ax3.set_xticklabels(cell_area_var_list, rotation=45, ha='right')
for x in first_optimal_pathfinder_radius:
    ax3.axvline(x, color='m', linewidth = 1, ls=':')

ax3.set_ylim(2, 20)
ax3.tick_params(axis='both', which='major', labelsize=10)
###ax3.set_xlabel('cell_radius_grid')
ax3.set_xlabel('Cell radius (px)', fontsize=10)
### To show areas instead
ax3.set_xlabel('Cell area ($\mathregular{px{^2}}$)', fontsize=10)
###ax3.set_ylabel('pathfinder_length_grid')
ax3.set_ylabel('Pathfinder length (px)', fontsize=10)
ax3.set_aspect(aspect='equal')
plt.tight_layout()
current_dir = os.getcwd()

fig3.savefig(current_dir+'\\optimized pathfinder length fig.pdf', format='pdf', dpi=1000)
plt.show()

