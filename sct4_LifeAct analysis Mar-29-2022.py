# -*- coding: utf-8 -*-
"""
Created on Thu May 30 11:21:19 2019

@author: Serge
"""

# For loading csv files / saving graphs:
from pathlib import Path

# For manipulating data:
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import broyden1 # solves nonlinear equations
# For propagating errors (eg. means with standard devations):
# from uncertainties import ufloat
from uncertainties import unumpy as unp

# For creating graphs
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
from matplotlib import colors as mcol
import matplotlib.cm
import matplotlib.colors
from mpl_toolkits.mplot3d import Axes3D


import seaborn as sns
# For making colorbars:
from mpl_toolkits.axes_grid1 import make_axes_locatable

# For creating 3D plots:
# from mpl_toolkits.mplot3d import Axes3D

# Needed for multipdf function to create pdf files:
from matplotlib.backends.backend_pdf import PdfPages




### First define funtions:

def cm2inch(num):
    return num/2.54




def data_import_and_clean(path):
    """This function opens a bunch of .csv files using pandas and appends 2 
    columns containing: 1) the filename, 2) the folder name that the .csv files
    are kept in."""
    
    data_list = []
    for filename in path.glob('*.csv'):
        data_list.append(pd.read_csv(filename))
        data_list[-1]['Source_File'] = pd.Series(
                data=[filename.name]*data_list[-1].shape[0])
        data_list[-1]['Experimental_Condition'] = pd.Series(
                data=[path.name]*data_list[-1].shape[0])    
    data_list_long = pd.concat([x for x in data_list])
    
    return data_list, data_list_long




def clean_up_linescan(dataframe, col_labels):
    """ This function just turns a stringified python list back into a list
    type object. """
    df = dataframe.copy()
    for col_lab in col_labels:
        # clean_col = [np.asarray([val.strip() for val in row[1:-1].split(' ') if val], dtype=float) for row in df[col_lab]]
        clean_col = [np.asarray([val.strip() for val in row[1:-1].split(',') if val], dtype=float) for row in df[col_lab]]
        df[col_lab] = clean_col
    
    return df




# def multipdf(filename):
#     """ This function saves a multi-page PDF of all open figures. """
#     pdf = PdfPages(filename)
#     figs = [plt.figure(f) for f in plt.get_fignums()]
#     for fig in figs:
#         fig.savefig(pdf, format='pdf' , dpi=80)
#     pdf.close()
def multipdf(filename, tight=False, dpi=80):
    """ This function saves a multi-page PDF of all open figures. """
    if tight == True:
        bbox = 'tight'
    else:
        bbox = None
    pdf = PdfPages(filename)
    figs = [plt.figure(f) for f in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pdf, format='pdf', bbox_inches=bbox, dpi=dpi)
    pdf.close()




def nondim_data(list_of_dfs, df_of_dfs, norm_to_cell=None):
    """  Non-dimensionalizes the contact diameters and cell areas. 
    cell_to_norm_to can be one of: None, 'red', 'blu'. If 'None' then it will 
    normalize to the mean first_valid_area of the 2 cells."""

    
    first_valid_area = [(df['Area of the cell pair (px**2)'].dropna() / 2.).iloc[0] for df in list_of_dfs]
    first_valid_area_red = [(df['Area of cell 1 (Red) (px**2)'].dropna()).iloc[0] for df in list_of_dfs]
    first_valid_area_blu = [(df['Area of cell 2 (Blue) (px**2)'].dropna()).iloc[0] for df in list_of_dfs]
    
    
    if norm_to_cell == 'red':
        first_valid_radius = [np.sqrt(area / np.pi) for area in first_valid_area_red]
    elif norm_to_cell == 'blu':
        first_valid_radius = [np.sqrt(area / np.pi) for area in first_valid_area_blu]
    else:
        first_valid_radius = [np.sqrt(area / np.pi) for area in first_valid_area]
    
    
    
    nondim_df = df_of_dfs.copy()
    contact_diam_non_dim = [list_of_dfs[i]['Contact Surface Length (px)'] / \
                            first_valid_radius[i] for i in \
                            np.arange(len(first_valid_radius))]
    
    area_cellpair_non_dim = [list_of_dfs[i]['Area of the cell pair (px**2)'] / \
                             first_valid_area[i] for i in \
                             np.arange(len(first_valid_area))]
    
    area_red_non_dim = [list_of_dfs[i]['Area of cell 1 (Red) (px**2)'] / \
                        first_valid_area_red[i] for i in \
                        np.arange(len(first_valid_area_red))]
    
    area_blu_non_dim = [list_of_dfs[i]['Area of cell 2 (Blue) (px**2)'] / \
                        first_valid_area_blu[i] for i in \
                        np.arange(len(first_valid_area_blu))]
    
        
    contact_diam_non_dim = pd.concat(contact_diam_non_dim)
    nondim_df['Contact Surface Length (px)'] = contact_diam_non_dim
    
    area_cellpair_non_dim = pd.concat(area_cellpair_non_dim)
    nondim_df['Area of the cell pair (px**2)'] = area_cellpair_non_dim
    
    area_red_non_dim = pd.concat(area_red_non_dim)
    nondim_df['Area of cell 1 (Red) (px**2)'] = area_red_non_dim
    
    area_grn_non_dim = pd.concat(area_blu_non_dim)
    nondim_df['Area of cell 2 (Blue) (px**2)'] = area_grn_non_dim
    
    nondim_df = nondim_df.rename(index=str, columns={'Contact Surface Length (px)': 'Contact Surface Length (N.D.)'})
    nondim_df = nondim_df.rename(index=str, columns={'Area of the cell pair (px**2)': 'Area of the cell pair (N.D.)'})
    nondim_df = nondim_df.rename(index=str, columns={'Area of cell 1 (Red) (px**2)': 'Area of cell 1 (Red) (N.D.)'})
    nondim_df = nondim_df.rename(index=str, columns={'Area of cell 2 (Blue) (px**2)': 'Area of cell 2 (Blue) (N.D.)'})
    
    nondim_df_list = [nondim_df[nondim_df['Source_File'] == x] for x in np.unique(nondim_df['Source_File'])]

    
    return nondim_df_list, nondim_df, first_valid_area_red, first_valid_area_blu




def expected_reduced_cortical_tension(contact_angle, nondim_contact_radius):
    """Returns the expected reduced cortical tension for a given contact angle 
    and non-dimensionalized contact radius (which is assumed to be half of the 
    contact surface length!). As the expected reduced cortical tension for a 
    given non-dimensionalized contact radius is not given by an analytical 
    solution, the closest non-dimensionalized contact radius is also returned
    for comparison with the inputted non-dimensionalized contact radius."""
    beta_star_div_beta_angle = np.cos(np.deg2rad(contact_angle))
    beta_star_div_beta_contnd_range = np.linspace(0, 1, 100001)
    nondim_contact_radius_range = ( (4 * ( 1 - (beta_star_div_beta_contnd_range)**2 )**(3/2)) / (4 - ((1 - beta_star_div_beta_contnd_range)**2) * (2 + beta_star_div_beta_contnd_range)) )**(1/3)
    closest_nondim_contact_radius = [nondim_contact_radius_range[np.argmin(abs(nondim_contact_radius_range - (x)))] for x in nondim_contact_radius]
    beta_star_div_beta_contnd = np.asarray([beta_star_div_beta_contnd_range[np.argmin(abs(nondim_contact_radius_range - (x)))] for x in nondim_contact_radius])
    invalid = np.asarray([not(np.isfinite(x)) for x in nondim_contact_radius], dtype=bool)
    beta_star_div_beta_contnd[invalid] = np.nan

    return beta_star_div_beta_angle, beta_star_div_beta_contnd, closest_nondim_contact_radius




def expected_spreading_sphere_params(theta, radius_initial):
    """Takes a contact angle (in degrees) and an initial radius and returns the 
    various expected variables. Please note that the contact angle theta is HALF
    of the angle measured when drawing 2 lines that are tangent to the cell
    surface at the 2 points that are the edges of the cell-cell contacts.
    Returns: (Rs, Vsphere, SAsphere, Asphere), (Rc, Vcap, SAcap, Acap, ContactLength, SAcontact)"""
    ### The lines below describe how various variables change as a sphere of 
    ### constant volume is squished:
    # Rs is the radius of the sphere, or initial radius
    Rs = radius_initial
    # Phi is the contact angle, or half the measured angle:
    phi = np.deg2rad(theta)
    #CA = 2.0 * phi
    
    #h = np.linspace(0, 2*Rs, 101)
    ### Assuming constant volume:
    #Rc = (1.0/3.0) * ((4.0 * (Rs**3.0) / ((h**2.0))) + h)
    # or if phi changes linearly (which means h does not):
    # Rc = ( (4 * (Rs**3.0)) / ((2.0 - np.cos(phi)) * ((np.cos(phi) + 1.0)**2.0)) )**(1.0/3.0)
    Rc = ( (4 * (Rs**3.0)) / (4 - ((1 - np.cos(phi))**2.0) * (2.0 + np.cos(phi))) )**(1.0/3.0)
    # The above 2 lines describing Rc are equivalent, but in the manuscript I 
    # have shown the work for the derivation of the lower one.

    ### Assuming constant surface area:
    #Rc = 2 * (Rs**2) / h
    # or if phi changes linearly (which means h does not):
    #Rc = ( (2.0 * (Rs**2.0)) / (np.cos(phi) + 1.0) )**0.5
    
    h = Rc * (1 + np.cos(phi))

    # The volume of the sphere before any deformations take place:
    Vsphere = (4.0/3.0) * np.pi * (Rs**3.0)
    
    # The volume of the sphere after deformations take place, ie. a spherical cap:
    Vcap = (1.0/3.0) * np.pi * (h**2.0) * ((3.0 * Rc) - h)
    
    # The surface area of the sphere before any deformations:
    SAsphere = 4.0 * np.pi * Rs**2.0
    
    # The surface area of the sphere after deformations (ie. spherical cap), 
    # please note that this surface area does NOT include the flat portion of 
    # the spherical cap -- it only includes the curved surface:
    SAcap = 2.0 * np.pi * Rc * h
    
    # Cross-sectional area of the sphere before deformations:
    Asphere = np.pi * Rs**2
    
    # Cross-sectional area of the sphere after deformations. Cross-section is
    # taken through the middle of the cap such that it cuts the flattened 
    # surface in half:
    Acap = (Rc**2.0) * np.arcsin( (h - Rc)/Rc ) + (h - Rc) * ((2.0 * Rc * h) - (h**2.0))**0.5 + (np.pi * (Rc**2.0) / 2.0)
    
    # The diameter of the flattened surface of the sphere:
    ContactLength = 2.0 * (h * (2.0 * Rc - h))**0.5

    # The surface area of the contact surface. Note that the ContactLength / 2
    # is equal to the contact radius. The total surface area of the cap is
    # equal to SAcap + SAcontact:
    SAcontact = np.pi * (ContactLength / 2)**2
    

    return (Rs, Vsphere, SAsphere, Asphere), (Rc, Vcap, SAcap, Acap, ContactLength, SAcontact)




def angle2reltension(angle_in_deg):
    """ Takes a numpy array. Outputs B* / B."""
    angle_in_rad = np.deg2rad(angle_in_deg)
    betastar_div_beta = np.cos(angle_in_rad)
    
    return betastar_div_beta

def reltension2angle(betastar_div_beta):
    """ Takes a numpy array. Outputs contact angle in deg (note that this is
    half of the angle measured by the measurement script)."""
    angle_rad = np.arccos(betastar_div_beta)
    angle_deg = np.rad2deg(angle_rad)
    
    return angle_deg




# def contnd2reltension(contnd):
#     """ Takes a numpy array. Outputs B* / B."""
#     betastar_div_beta = 

#     return betastar_div_beta

def reltension2contnd(betastar_div_beta):
    """ Takes a numpy array. Outputs non-dimensional contact radius (note that 
    this is half of the contact surface length (ie. contact diameter) measured 
    by the measurement script)."""
    contnd = ( (4 * ( 1 - (betastar_div_beta)**2 )**(3/2)) / (4 - ((1 - betastar_div_beta)**2) * (2 + betastar_div_beta)) )**(1/3)
    
    return contnd




def round_up_to_multiple(int_or_float, multiple):
    """ Will round a number up to given multiple.
        Eg. round_up_to_multiple(13, 2) will return 14.
        Eg. round_up_to_multiple(13, 10) will return 20."""
    whole_divisions = int(int_or_float / multiple)
    leftover = int_or_float % multiple
    if leftover:
        rounded = (whole_divisions + 1) * multiple
    else:
        rounded = whole_divisions * multiple
    
    return rounded




def colorcoded_arrowplot(xs, ys, ax, head_length=0.4, head_width=0.2, color='k', alpha=1, use_colormap=False, cmap='jet'):
    """ xs,ys = arrow base coords. xs and ys must be the same length and shape. 
    For other arguments see 'matplotlib.pyplot.arrow'"""
    
    # should also have something that checks that xs and ys are the same length
    if use_colormap == False:
        colors = [color] * len(xs[:-1])
    elif use_colormap == True:
        cmap_adjusted = matplotlib.cm.get_cmap(cmap, len(xs[:-1]))
        colors = [cmap_adjusted(i) for i in range(cmap_adjusted.N)]
        
    tails = (xs[:-1], ys[:-1])
    heads = (xs[:-1] + np.diff(xs), ys[:-1] + np.diff(ys))
    
    
    arrow_coords = list(zip(*tails, *heads, colors))
    
    
    
    # fig, ax = plt.subplots()
    for xt,yt,xh,yh,cs in arrow_coords:
        arrowprops = dict(arrowstyle='-|>, head_length=%s, head_width=%s'%(head_length, head_width),
                          shrinkA=0, shrinkB=0, color=cs, alpha=alpha)
        ax.annotate('', xy=(xh,yh), xytext=(xt,yt), xycoords='data', 
                    arrowprops=arrowprops)
        # ax.set_xlim(xs.min()-0.1, xs.max()+0.1)
        # ax.set_ylim(ys.min()-0.1, ys.max()+0.1)
    
        
    return 




def colorcoded_lineplot_3d(xs, ys, zs, ax, color='k', alpha=1, use_colormap=False, cmap='jet'):
    """ xs,ys = arrow base coords. xs and ys must be the same length and shape. 
    For other arguments see 'matplotlib.pyplot.arrow'"""
    
    # should also have something that checks that xs and ys are the same length
    if use_colormap == False:
        colors = [color] * len(xs[:-1])
    elif use_colormap == True:
        cmap_adjusted = matplotlib.cm.get_cmap(cmap, len(xs[:-1]))
        colors = [cmap_adjusted(i) for i in range(cmap_adjusted.N)]
        
    tails = (xs[:-1], ys[:-1], zs[:-1])
    heads = (np.diff(xs), np.diff(ys), np.diff(zs))
    # heads = (xs[:-1] + np.diff(xs), ys[:-1] + np.diff(ys), zs[:-1] + np.diff(zs))
    
    
    arrow_coords = list(zip(*tails, *heads, colors))
    
    
    
    # fig = plt.figure() # don't forget to comment out this line before defining function
    # ax = fig.add_subplot(projection='3d') # don't forget to comment out this line before defining function
    for xt,yt,zt,xh,yh,zh,c in arrow_coords:
        # ax.quiver(xt,yt,zt, xh,yh,zh, pivot='tail', normalize=True, color=c, alpha=alpha)
        ax.plot((xt,xt+xh), (yt,yt+yh), (zt,zt+zh), 
                color=c, alpha=alpha)
        # ax.set_xlim(xs.min()-0.1, xs.max()+0.1)
        # ax.set_ylim(ys.min()-0.1, ys.max()+0.1)
    
    return 





### Next define some global variables:
    
### A lot of figures are going to be opened before they are saved in to a
### multi-page pdf, so increase the max figure open warning:
plt.rcParams.update({'figure.max_open_warning':300})

# Set figure widths (and heights) according to journal preferences (in cm) and
# initial parameters of matplotlib:
# The aspect ratio in [width, height]:
aspect_ratio = np.asarray(plt.rcParams["figure.figsize"]) / plt.rcParams["figure.figsize"][0]
# 1 column (preview; or 3 column formats):
width_1col_prev = cm2inch(5.5)
height_1col_prev = width_1col_prev * aspect_ratio[1]
# 1 column (2 column format):
width_1col = cm2inch(8.5)
height_1col = width_1col * aspect_ratio[1]
# 1.5 columns (2 column format):
width_1pt5col = cm2inch(11.4)
height_1pt5col = width_1pt5col * aspect_ratio[1]
# 2 columns (2 column format):
width_2col = cm2inch(17.4)
height_2col = width_2col * aspect_ratio[1]

dpi_graph = 600
dpi_LineArt = 1000


### X, Y resolution of the acquired timelapses (same for all movies):
um_per_pixel = 0.5069335


### Time resolution of the acquired timelapses (same for all movies):
### time_res = minutes/timeframe
time_res = 0.5






# Get the working directory of the script:
current_script_dir = Path(__file__).parent.absolute()

# Data can be found in the following directory:
data_dir = Path.joinpath(current_script_dir, r"data_ecto-ecto_lifeact")
adh_kin_figs_out_dir = Path.joinpath(current_script_dir, r"adhesion_analysis_figs_out")

ee1xMBS_path = Path.joinpath(data_dir, r"ee1xMBS")
ee1xDB_path = Path.joinpath(data_dir, r"ee1xDB")
radius30px_cell_sim_path = Path.joinpath(data_dir, "radius30px_cell_sim")
radius30px_cell_sim_LA_path = Path.joinpath(data_dir, "radius30px_cell_sim_LA")
ee1xDBsc_path = Path.joinpath(data_dir, r"ee1xDBsc")

# Graphs/other analysis will be output to the following directory:
out_dir = Path.joinpath(current_script_dir, r"lifeact_analysis_figs_out")





### Now load in data and start the analysis:

# For ecto-ecto in 1xMBS load in your data and give your time column units:
# Note to future self: the data_import_and_clean function will create an
# 'Experimental_Condition' column with the folder that the csv files
# are in for values.
lifeact_orig, lifeact_long_orig = data_import_and_clean(ee1xMBS_path)
[x.rename({'Time':'Time (minutes)'}, axis='columns', inplace=True) for x in lifeact_orig]
lifeact_long_orig.rename({'Time':'Time (minutes)'}, axis='columns', inplace=True)

# For ecto-ecto in 1xDB load in your data and give your time column units:
lifeact_db_orig, lifeact_db_long_orig = data_import_and_clean(ee1xDB_path)
[x.rename({'Time':'Time (minutes)'}, axis='columns', inplace=True) for x in lifeact_db_orig]
lifeact_db_long_orig.rename({'Time':'Time (minutes)'}, axis='columns', inplace=True)

# Load ecto-ecto in 1xDB single cell examples:
lifeact_dbsc_orig, lifeact_dbsc_long_orig = data_import_and_clean(ee1xDBsc_path)
[x.rename({'Time':'Time (minutes)'}, axis='columns', inplace=True) for x in lifeact_dbsc_orig]
lifeact_dbsc_long_orig.rename({'Time':'Time (minutes)'}, axis='columns', inplace=True)


### Also load in your simulation of a cell with constant volume spreading 
radius30px_cell_sim_list_original, radius30px_cell_sim_list_long = data_import_and_clean(radius30px_cell_sim_path)

# Since this file originated from the red-green combinations the headers of some
# columns refer to a "Green" cell, but we have red-blue combinations, so change
# those headers to say "Blue" instead of "Green":
for col_name in radius30px_cell_sim_list_long.head(0):
    if 'Green' in col_name:
        radius30px_cell_sim_list_long.rename({col_name: col_name.replace('Green', 'Blue')}, 
                                              axis='columns', inplace=True)

for col_name in radius30px_cell_sim_list_long.head(0):
    if 'grn' in col_name:
        radius30px_cell_sim_list_long.rename({col_name: col_name.replace('grn', 'blu')},
                                              axis='columns', inplace=True)

for df in radius30px_cell_sim_list_original:
    for col_name in df.head(0):
        if 'Green' in col_name:
            df.rename({col_name: col_name.replace('Green', 'Blue')}, 
                      axis='columns', inplace=True)

for df in radius30px_cell_sim_list_original:
    for col_name in df.head(0):
        if 'grn' in col_name:
            df.rename({col_name: col_name.replace('grn', 'blu')}, 
                      axis='columns', inplace=True)

# Each timepoint in the simulation is associated with a theoretical 
# contact angle and therefore a theoretical contact diameter which was then 
# drawn as pixels and measured. The values in the table for angle and contact
# diameter are how that theoretical angle appears under our imaging conditions.
# Therefore we will make a list of the theoretical angles and contact diameters
theor_ang = np.concatenate(([0,0], np.linspace(0,90,91)))
theor_contnd = expected_spreading_sphere_params(theor_ang, 1)[1][4]
theor_relten = angle2reltension(theor_ang)


### Load in simulation of cell with constant volume spreading when analyzed with
# the LifeAct measurement script:
radius30px_cell_sim_LA_list_original, radius30px_cell_sim_LA_list_long = data_import_and_clean(radius30px_cell_sim_LA_path)

# Similarly for the LifeAct simulations each timepoint has a theoretical angle:
theor_ang_LA = np.concatenate(([0,0], np.linspace(0,90,91)))
theor_contnd_LA = expected_spreading_sphere_params(theor_ang_LA, 1)[1][4]
theor_relten_LA = angle2reltension(theor_ang_LA)


# For simplicity let's combine the treatments into a single list of dataframes
# and then also a single dataframe:
lifeact = lifeact_orig + lifeact_db_orig + lifeact_dbsc_orig
lifeact_long = pd.concat((lifeact_long_orig, lifeact_db_long_orig, lifeact_dbsc_long_orig))

# Mask the rows that are not valid:
lifeact = [x.mask(x['Mask? (manually defined by user who checks quality of cell borders)']) for x in lifeact]
lifeact_long = lifeact_long.mask(lifeact_long['Mask? (manually defined by user who checks quality of cell borders)'])

# Drop masked entries:
lifeact = [x.dropna(axis=0, subset=['Source_File']) for x in lifeact]
lifeact_long = lifeact_long.dropna(axis=0, subset=['Source_File'])


# Non-dimensionalize the contact diameters and cell areas:
la_contnd_df_list, la_contnd_df, first_valid_area_red, first_valid_area_blu = nondim_data(lifeact, lifeact_long, 
                                                                                          norm_to_cell='red')

r30sim_contnd_df_list, r30sim_contnd_df, _, _ = nondim_data(radius30px_cell_sim_list_original, 
                                                            radius30px_cell_sim_list_long, 
                                                            norm_to_cell='red')

r30simLA_contnd_df_list, r30simLA_contnd_df, _, _ = nondim_data(radius30px_cell_sim_LA_list_original, 
                                                                radius30px_cell_sim_LA_list_long, 
                                                                norm_to_cell='red')

# Clean up the linescan data that has been stringified so that it is returned
# back to array objects:
cols_to_clean = ['grnchan_fluor_at_contact', 'grnchan_fluor_outside_contact', 'grnchan_fluor_cytoplasmic',
                 'redchan_fluor_at_contact', 'redchan_fluor_outside_contact', 'redchan_fluor_cytoplasmic',
                 'bluchan_fluor_at_contact', 'bluchan_fluor_outside_contact', 'bluchan_fluor_cytoplasmic']
la_contnd_df_list = [clean_up_linescan(df, cols_to_clean) for df in la_contnd_df_list]
la_contnd_df = clean_up_linescan(la_contnd_df, cols_to_clean)




# Create a new column with the relative lifeact values (ie. Bc / B):
# Calculate means and SEMs:
la_contnd_df['grnchan_fluor_at_contact_mean'] = la_contnd_df['grnchan_fluor_at_contact'].apply(np.mean)
la_contnd_df['grnchan_fluor_at_contact_sem'] = la_contnd_df['grnchan_fluor_at_contact'].apply(stats.sem)

la_contnd_df['grnchan_fluor_outside_contact_mean'] = la_contnd_df['grnchan_fluor_outside_contact'].apply(np.mean)
la_contnd_df['grnchan_fluor_outside_contact_sem'] = la_contnd_df['grnchan_fluor_outside_contact'].apply(stats.sem)

la_contnd_df['grnchan_fluor_cytoplasmic_mean'] = la_contnd_df['grnchan_fluor_cytoplasmic'].apply(np.mean)
la_contnd_df['grnchan_fluor_cytoplasmic_sem'] = la_contnd_df['grnchan_fluor_cytoplasmic'].apply(stats.sem)

# Generate uncertain floats to do error propagation with:
la_fluor_at_cont = unp.uarray(la_contnd_df['grnchan_fluor_at_contact_mean'], la_contnd_df['grnchan_fluor_at_contact_sem'])
la_fluor_out_cont = unp.uarray(la_contnd_df['grnchan_fluor_outside_contact_mean'], la_contnd_df['grnchan_fluor_outside_contact_sem'])
la_fluor_cytopl = unp.uarray(la_contnd_df['grnchan_fluor_cytoplasmic_mean'], la_contnd_df['grnchan_fluor_cytoplasmic_sem'])
la_fluor_normed = la_fluor_at_cont / la_fluor_out_cont
la_cytopl_fluor_normed = la_fluor_cytopl / la_fluor_out_cont

la_contnd_df['lifeact_fluor_normed_mean'] = unp.nominal_values(la_fluor_normed)
la_contnd_df['lifeact_fluor_normed_sem'] = unp.std_devs(la_fluor_normed) # It's actually the SEM
la_contnd_df['lifeact_cytoplasmic_fluor_normed_mean'] = unp.nominal_values(la_cytopl_fluor_normed)
la_contnd_df['lifeact_cytoplasmic_fluor_normed_sem'] = unp.std_devs(la_cytopl_fluor_normed)


# Do same for mbrfp:
la_contnd_df['redchan_fluor_at_contact_mean'] = la_contnd_df['redchan_fluor_at_contact'].apply(np.mean)
la_contnd_df['redchan_fluor_at_contact_sem'] = la_contnd_df['redchan_fluor_at_contact'].apply(stats.sem)

la_contnd_df['redchan_fluor_outside_contact_mean'] = la_contnd_df['redchan_fluor_outside_contact'].apply(np.mean)
la_contnd_df['redchan_fluor_outside_contact_sem'] = la_contnd_df['redchan_fluor_outside_contact'].apply(stats.sem)

la_contnd_df['redchan_fluor_cytoplasmic_mean'] = la_contnd_df['redchan_fluor_cytoplasmic'].apply(np.mean)
la_contnd_df['redchan_fluor_cytoplasmic_sem'] = la_contnd_df['redchan_fluor_cytoplasmic'].apply(stats.sem)

# Generate uncertain floats to do error propagation with:
mb_fluor_at_cont = unp.uarray(la_contnd_df['redchan_fluor_at_contact_mean'], la_contnd_df['redchan_fluor_at_contact_sem'])
mb_fluor_out_cont = unp.uarray(la_contnd_df['redchan_fluor_outside_contact_mean'], la_contnd_df['redchan_fluor_outside_contact_sem'])
mb_fluor_cytopl = unp.uarray(la_contnd_df['redchan_fluor_cytoplasmic_mean'], la_contnd_df['redchan_fluor_cytoplasmic_sem'])
mb_fluor_normed = mb_fluor_at_cont / mb_fluor_out_cont
mb_cytopl_fluor_normed = mb_fluor_cytopl / mb_fluor_out_cont

la_contnd_df['mbrfp_fluor_normed_mean'] = unp.nominal_values(mb_fluor_normed)
la_contnd_df['mbrfp_fluor_normed_sem'] = unp.std_devs(mb_fluor_normed)
la_contnd_df['mbrfp_cytoplasmic_fluor_normed_mean'] = unp.nominal_values(mb_cytopl_fluor_normed)
la_contnd_df['mbrfp_cytoplasmic_fluor_normed_sem'] = unp.std_devs(mb_cytopl_fluor_normed)


# Do same for AlexaFluor647 Dextran:
la_contnd_df['bluchan_fluor_at_contact_mean'] = la_contnd_df['bluchan_fluor_at_contact'].apply(np.mean)
la_contnd_df['bluchan_fluor_at_contact_sem'] = la_contnd_df['bluchan_fluor_at_contact'].apply(stats.sem)

la_contnd_df['bluchan_fluor_outside_contact_mean'] = la_contnd_df['bluchan_fluor_outside_contact'].apply(np.mean)
la_contnd_df['bluchan_fluor_outside_contact_sem'] = la_contnd_df['bluchan_fluor_outside_contact'].apply(stats.sem)

la_contnd_df['bluchan_fluor_cytoplasmic_mean'] = la_contnd_df['bluchan_fluor_cytoplasmic'].apply(np.mean)
la_contnd_df['bluchan_fluor_cytoplasmic_sem'] = la_contnd_df['bluchan_fluor_cytoplasmic'].apply(stats.sem)

# Generate uncertain floats to do error propagation with:
dx_fluor_at_cont = unp.uarray(la_contnd_df['bluchan_fluor_at_contact_mean'], la_contnd_df['bluchan_fluor_at_contact_sem'])
dx_fluor_out_cont = unp.uarray(la_contnd_df['bluchan_fluor_outside_contact_mean'], la_contnd_df['bluchan_fluor_outside_contact_sem'])
dx_fluor_cytopl = unp.uarray(la_contnd_df['bluchan_fluor_cytoplasmic_mean'], la_contnd_df['bluchan_fluor_cytoplasmic_sem'])
dx_fluor_normed = dx_fluor_at_cont / dx_fluor_out_cont
dx_cytopl_fluor_normed = dx_fluor_cytopl / dx_fluor_out_cont

la_contnd_df['af647dx_fluor_normed_mean'] = unp.nominal_values(dx_fluor_normed)
la_contnd_df['af647dx_fluor_normed_sem'] = unp.std_devs(dx_fluor_normed)
la_contnd_df['af647dx_cytoplasmic_fluor_normed_mean'] = unp.nominal_values(dx_cytopl_fluor_normed)
la_contnd_df['af647dx_cytoplasmic_fluor_normed_sem'] = unp.std_devs(dx_cytopl_fluor_normed)



# Try plotting the lifeact data in a 3d scatterplot:
x = la_contnd_df[la_contnd_df['Experimental_Condition'] == 'ee1xMBS']['Average Angle (deg)']
y = la_contnd_df[la_contnd_df['Experimental_Condition'] == 'ee1xMBS']['Contact Surface Length (N.D.)']
z = la_contnd_df[la_contnd_df['Experimental_Condition'] == 'ee1xMBS']['lifeact_fluor_normed_mean']
w = la_contnd_df[la_contnd_df['Experimental_Condition'] == 'ee1xMBS']['mbrfp_fluor_normed_mean']
t = la_contnd_df[la_contnd_df['Experimental_Condition'] == 'ee1xMBS']['Time (minutes)']

xdb = la_contnd_df[la_contnd_df['Experimental_Condition'] == 'ee1xDB']['Average Angle (deg)']
ydb = la_contnd_df[la_contnd_df['Experimental_Condition'] == 'ee1xDB']['Contact Surface Length (N.D.)']
zdb = la_contnd_df[la_contnd_df['Experimental_Condition'] == 'ee1xDB']['lifeact_fluor_normed_mean']
wdb = la_contnd_df[la_contnd_df['Experimental_Condition'] == 'ee1xDB']['mbrfp_fluor_normed_mean']
tdb = la_contnd_df[la_contnd_df['Experimental_Condition'] == 'ee1xDB']['Time (minutes)']


# Predicted:
# We're just going to give B*/B a linear distribution for the moment:
beta_star_div_beta_contnd_range = np.linspace(0, 1, 100001)
# The contact angle is related to B* / B by the following equation (in theory):
angle_predict_deg = reltension2angle(beta_star_div_beta_contnd_range)
# Note that angle_predict_deg is half the angle measured in data.

# The contact diameter is related to B* / B by the following equation (in theory):
contnd_predict = reltension2contnd(beta_star_div_beta_contnd_range)
# Note that the contnd_predict is also half of the measured contnd (its equivalent to the contact radius).


# We also want to predict what the relative tension should be for each 
# timepoint in the simulated spreading cells:
r30sim_angle_predict_beta_star_div_beta = angle2reltension(r30sim_contnd_df['Average Angle (deg)']/2)
r30sim_contnd_df['lifeact_fluor_normed_mean'] = r30sim_angle_predict_beta_star_div_beta

r30simLA_angle_predict_beta_star_div_beta = angle2reltension(r30simLA_contnd_df['Average Angle (deg)']/2)
r30simLA_contnd_df['lifeact_fluor_normed_mean'] = r30simLA_angle_predict_beta_star_div_beta

# We want to return the expected maximum contact diameter
expect_max_contnd = expected_spreading_sphere_params(90, 1)[1][4]


# Do 2D plots of Bc/B and mbRFP, measured angle, contact diameter vs each other:

# 2D plots for each timelapse separately:
cmap_name = 'jet'
cmap = plt.cm.get_cmap(cmap_name)
for name, group in la_contnd_df.groupby('Source_File'):
    
    if np.all(group['Experimental_Condition'] == 'ee1xDB'):
        lim_ang = (0, 200)#(0, 100)
        lim_contnd = (0, 3)#(0, 1.5)
    else:
        lim_ang = (0, 200)
        lim_contnd = (0, 3)

    cont = group['Contact Surface Length (N.D.)']
    ang = group['Average Angle (deg)']
    la_normed = group['lifeact_fluor_normed_mean']
    mbrfp_normed = group['mbrfp_fluor_normed_mean']
    minutes = group['Time (minutes)']
    # print('name: %s'%name,' / ','minutes.min: %f'%minutes.min(),' / ','minutes.max: %f'%minutes.max() )
    
    fig, (ax1,ax2,ax3) = plt.subplots(figsize=(13,4), nrows=1, ncols=3)
    fig.suptitle(name)
    graph = ax1.scatter(ang, cont, c=minutes, cmap=cmap, marker='.', s=5, 
                        alpha=1, vmin=minutes.min(), vmax=minutes.max(), zorder=10)
    ax1.plot(ang, cont, c='grey', lw=1, zorder=2, alpha=0.5)
    colorcoded_arrowplot(ang, cont, ax1, alpha=0.7, use_colormap=True, cmap=cmap_name)
    ax1.plot(angle_predict_deg*2, contnd_predict*2, c='k', lw=1, ls='--', zorder=1)
    ax1.plot(r30simLA_contnd_df['Average Angle (deg)'], 
             r30simLA_contnd_df['Contact Surface Length (N.D.)'], 
             c='k', marker='.', markersize=3, lw=1, ls='-', zorder=2)
    ax1.set_xlim(*lim_ang)
    ax1.set_ylim(*lim_contnd)
    ax1.axvline(180, c='grey', lw=1, zorder=0)
    ax1.axhline(expect_max_contnd, c='grey', lw=1, zorder=0)
    ax1.set_xlabel('Average Angle (deg)')
    ax1.set_ylabel('Contact Diameter (N.D.)')
    ax1.set_adjustable('box')

    ax2.scatter(ang, la_normed, c=minutes, cmap=cmap, marker='.', s=5, 
                alpha=1, vmin=minutes.min(), vmax=minutes.max(), zorder=10)
    ax2.plot(ang, la_normed, c='g', lw=1, zorder=2)
    ax2.scatter(ang, mbrfp_normed, c=minutes, cmap=cmap, marker='.', s=5, 
                alpha=1, vmin=minutes.min(), vmax=minutes.max(), zorder=10)
    ax2.plot(ang, mbrfp_normed, c='r', lw=1, zorder=2, alpha=0.5)
    ax2.plot(angle_predict_deg*2, beta_star_div_beta_contnd_range, c='k', lw=1, ls='--', zorder=1)
    ax2.set_xlim(*lim_ang)
    ax2.set_ylim(0,2)
    ax2.axvline(180, c='grey', lw=1, zorder=0)
    # ax2.axhline()
    ax2.set_xlabel('Average Angle (deg)')
    ax2.set_ylabel('Contact fluor. / Non-contact fluor.')
    ax2.set_adjustable('box')

    ax3.scatter(cont, la_normed, c=minutes, cmap=cmap, marker='.', s=5, 
                alpha=1, vmin=minutes.min(), vmax=minutes.max(), zorder=10)
    ax3.plot(cont, la_normed, c='g', lw=1, zorder=2, alpha=0.5)
    ax3.scatter(cont, mbrfp_normed, c=minutes, cmap=cmap, marker='.', s=5, 
                alpha=1, vmin=minutes.min(), vmax=minutes.max(), zorder=10)
    ax3.plot(cont, mbrfp_normed, c='r', lw=1, zorder=2, alpha=0.5)
    ax3.plot(contnd_predict*2, beta_star_div_beta_contnd_range, c='k', lw=1, ls='--', zorder=1)
    ax3.set_xlim(*lim_contnd)
    ax3.set_ylim(0,2)
    ax3.axvline(expect_max_contnd, c='grey', lw=1, zorder=0)
    ax3.set_xlabel('Contact Diameter (N.D.)')
    ax3.set_ylabel('Contact fluor. / Non-contact fluor.')
    ax3.set_adjustable('box')
    
    # Include a colorbar for the timepoints
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(graph, cax=cax)

    plt.subplots_adjust(wspace=0.3, hspace=0.3)

multipdf(Path.joinpath(out_dir, r"cont_vs_angle_vs_Bc_mbrfp.pdf"))
plt.close('all')



### Pick a couple of graphs of interest to save as separate images:
files_of_interest = ['2019_09_17-series001_crop2_z1.csv',
                     '2019_09_20-series001_crop1_z3.csv',
                     '2019_09_20-series001_crop2_z1.csv',
                     '2019_10_22-series001_crop1_z2.csv',
                     '2019_10_22-series001_crop2_z3.csv',
                     '2020_01_17-series003_crop1_z2.csv',
                     '2020_01_17-series003_crop4_z1.csv'] 
cmap = plt.cm.get_cmap(cmap_name)
for name, group in la_contnd_df.groupby('Source_File'):
    if name in files_of_interest:
        if Path.exists(Path.joinpath(out_dir, 'ang_cont_fluor_%s'%name)) == False:
            Path.mkdir(Path.joinpath(out_dir, 'ang_cont_fluor_%s'%name))
        else:
            pass
        
        if np.all(group['Experimental_Condition'] == 'ee1xDB'):
            # print(name)
            lim_ang = (0, 200)#(0, 100)
            lim_contnd = (0, 3)#(0, 1.5)
        else:
            lim_ang = (0, 200)
            lim_contnd = (0, 3)
            
        cont = group['Contact Surface Length (N.D.)']
        ang = group['Average Angle (deg)']
        la_normed = group['lifeact_fluor_normed_mean']
        mbrfp_normed = group['mbrfp_fluor_normed_mean']
        minutes = group['Time (minutes)']
        # print('name: %s'%name,' / ','minutes.min: %f'%minutes.min(),' / ','minutes.max: %f'%minutes.max() )
        
        fig, ax1 = plt.subplots(figsize=(width_1pt5col/3,width_1pt5col/3), nrows=1, ncols=1)
        # fig.suptitle(name)
        graph = ax1.scatter(ang, cont, c=minutes, cmap=cmap, marker='.', s=5, 
                            alpha=1, vmin=minutes.min(), vmax=minutes.max(), zorder=10)
        ax1.plot(ang, cont, c='grey', lw=1, zorder=2, alpha=0.5)
        # colorcoded_arrowplot(ang, cont, ax1, use_colormap=True)
        # ax1.plot(ang, cont, c=minutes, lw=1, zorder=2, alpha=0.5)
        ax1.plot(angle_predict_deg*2, contnd_predict*2, c='k', lw=1, ls='--', zorder=1)
        ax1.plot(r30simLA_contnd_df['Average Angle (deg)'], 
                 r30simLA_contnd_df['Contact Surface Length (N.D.)'], 
                 c='k', marker='.', markersize=3, lw=1, ls='-', zorder=2)
        ax1.set_xlim(*lim_ang)
        ax1.set_ylim(*lim_contnd)
        ax1.axvline(180, c='grey', lw=1, zorder=0)
        ax1.axhline(expect_max_contnd, c='grey', lw=1, zorder=0)
        # ax1.set_xlabel('Average Angle (deg)')
        # ax1.set_ylabel('Contact Diameter (N.D.)')
        ax1.set_adjustable('box')
        fig.savefig(Path.joinpath(out_dir, 'ang_cont_fluor_%s/ang_cont.tif'%name), 
                    dpi=dpi_graph, bbox_inches='tight')
    
        fig, ax2 = plt.subplots(figsize=(width_1pt5col/3,width_1pt5col/3), nrows=1, ncols=1)
        ax2.scatter(ang, la_normed, c=minutes, cmap=cmap, marker='.', s=5, 
                    alpha=1, vmin=minutes.min(), vmax=minutes.max(), zorder=10)
        ax2.plot(ang, la_normed, c='g', lw=1, zorder=2, alpha=0.5)
        # ax2.plot(ang, la_normed, c=minutes, lw=1, zorder=2)
        # ax2.scatter(ang, mbrfp_normed, c=minutes, cmap=cmap, marker='.', s=5, 
        #             alpha=1, vmin=minutes.min(), vmax=minutes.max(), zorder=10)
        # ax2.plot(ang, mbrfp_normed, c='r', lw=1, zorder=2, alpha=0.5)
        ax2.plot(angle_predict_deg*2, beta_star_div_beta_contnd_range, c='k', lw=1, ls='--', zorder=1)
        ax2.set_xlim(*lim_ang)
        ax2.set_ylim(0,2)
        ax2.axvline(180, c='grey', lw=1, zorder=0)
        # ax2.axhline()
        # ax2.set_xlabel('Average Angle (deg)')
        # ax2.set_ylabel('Contact fluor. / Non-contact fluor.')
        ax2.set_adjustable('box')
        fig.savefig(Path.joinpath(out_dir, 'ang_cont_fluor_%s/ang_lifeact.tif'%name), 
                    dpi=dpi_graph, bbox_inches='tight')

        fig, ax3 = plt.subplots(figsize=(width_1pt5col/3,width_1pt5col/3), nrows=1, ncols=1)
        ax3.scatter(cont, la_normed, c=minutes, cmap=cmap, marker='.', s=5, 
                    alpha=1, vmin=minutes.min(), vmax=minutes.max(), zorder=10)
        ax3.plot(cont, la_normed, c='g', lw=1, zorder=2, alpha=0.5)
        # ax3.plot(cont, la_normed, c=minutes, lw=1, zorder=2, alpha=0.5)
        # ax3.scatter(cont, mbrfp_normed, c=minutes, cmap=cmap, marker='.', s=5, 
        #             alpha=1, vmin=minutes.min(), vmax=minutes.max(), zorder=10)
        # ax3.plot(cont, mbrfp_normed, c='r', lw=1, zorder=2, alpha=0.5)
        ax3.plot(contnd_predict*2, beta_star_div_beta_contnd_range, c='k', lw=1, ls='--', zorder=1)
        ax3.set_xlim(*lim_contnd)
        ax3.set_xticks([0,1,2,3])
        # ax3.set_xticklabels([0,1,2,3])
        ax3.set_ylim(0,2)
        ax3.axvline(expect_max_contnd, c='grey', lw=1, zorder=0)
        # ax3.set_xlabel('Contact Diameter (N.D.)')
        # ax3.set_ylabel('Contact fluor. / Non-contact fluor.')
        ax3.set_adjustable('box')
        fig.savefig(Path.joinpath(out_dir, 'ang_cont_fluor_%s/cont_lifeact.tif'%name), 
                    dpi=dpi_graph, bbox_inches='tight')


        fig, ax2 = plt.subplots(figsize=(width_1pt5col/3,width_1pt5col/3), nrows=1, ncols=1)
        # ax2.scatter(ang, la_normed, c=minutes, cmap=cmap, marker='.', s=5, 
        #             alpha=1, vmin=minutes.min(), vmax=minutes.max(), zorder=10)
        # ax2.plot(ang, la_normed, c='g', lw=1, zorder=2)
        # ax2.plot(ang, la_normed, c=minutes, lw=1, zorder=2)
        ax2.scatter(ang, mbrfp_normed, c=minutes, cmap=cmap, marker='.', s=5, 
                    alpha=1, vmin=minutes.min(), vmax=minutes.max(), zorder=10)
        ax2.plot(ang, mbrfp_normed, c='r', lw=1, zorder=2, alpha=0.5)
        # ax2.plot(ang, mbrfp_normed, c=minutes, lw=1, zorder=2, alpha=0.5)
        ax2.plot(angle_predict_deg*2, beta_star_div_beta_contnd_range, c='k', lw=1, ls='--', zorder=1)
        ax2.set_xlim(*lim_ang)
        ax2.set_ylim(0,2)
        ax2.axvline(180, c='grey', lw=1, zorder=0)
        # ax2.axhline()
        # ax2.set_xlabel('Average Angle (deg)')
        # ax2.set_ylabel('Contact fluor. / Non-contact fluor.')
        ax2.set_adjustable('box')
        fig.savefig(Path.joinpath(out_dir, 'ang_cont_fluor_%s/ang_mbrfp.tif'%name), 
                    dpi=dpi_graph, bbox_inches='tight')

        fig, ax3 = plt.subplots(figsize=(width_1pt5col/3,width_1pt5col/3), nrows=1, ncols=1)
        # ax3.scatter(cont, la_normed, c=minutes, cmap=cmap, marker='.', s=5, 
        #             alpha=1, vmin=minutes.min(), vmax=minutes.max(), zorder=10)
        # ax3.plot(cont, la_normed, c='g', lw=1, zorder=2, alpha=0.5)
        # ax3.plot(cont, la_normed, c=minutes, lw=1, zorder=2, alpha=0.5)
        ax3.scatter(cont, mbrfp_normed, c=minutes, cmap=cmap, marker='.', s=5, 
                    alpha=1, vmin=minutes.min(), vmax=minutes.max(), zorder=10)
        ax3.plot(cont, mbrfp_normed, c='r', lw=1, zorder=2, alpha=0.5)
        # ax3.plot(cont, mbrfp_normed, c=minutes, lw=1, zorder=2, alpha=0.5)
        ax3.plot(contnd_predict*2, beta_star_div_beta_contnd_range, c='k', lw=1, ls='--', zorder=1)
        ax3.set_xlim(*lim_contnd)
        ax3.set_xticks([0,1,2,3])
        # ax3.set_xticklabels([0,1,2,3])
        ax3.set_ylim(0,2)
        ax3.axvline(expect_max_contnd, c='grey', lw=1, zorder=0)
        # ax3.set_xlabel('Contact Diameter (N.D.)')
        # ax3.set_ylabel('Contact fluor. / Non-contact fluor.')
        ax3.set_adjustable('box')
        fig.savefig(Path.joinpath(out_dir, 'ang_cont_fluor_%s/cont_mbrfp.tif'%name), 
                    dpi=dpi_graph, bbox_inches='tight')


        
        ### Include a standalone colorbar for the timepoints:
        fig, ax4 = plt.subplots(figsize=(width_1pt5col/3*0.05,width_1pt5col/3), nrows=1, ncols=1)
        fig.patch.set_alpha(0)
        cmap = matplotlib.cm.jet
        norm = mcol.Normalize(vmin=minutes.min(), vmax=minutes.max())
        cbar = matplotlib.colorbar.ColorbarBase(ax4, cmap=cmap, norm=norm, orientation='vertical')
        # cbar.set_label('Time (minutes)', rotation=270, va='bottom', ha='center')
        fig.savefig(Path.joinpath(out_dir, 'ang_cont_fluor_%s/cbar.tif'%name), 
                    dpi=dpi_graph, bbox_inches='tight')
        
        
        ### Make plots of angle and contnd over time separately:
        fig, ax2 = plt.subplots(figsize=(width_1pt5col/2,width_1pt5col/2), nrows=1, ncols=1)
        # ax2.scatter(ang, la_normed, c=minutes, cmap=cmap, marker='.', s=5, 
        #             alpha=1, vmin=minutes.min(), vmax=minutes.max(), zorder=10)
        # ax2.plot(ang, la_normed, c='g', lw=1, zorder=2)
        # ax2.plot(ang, la_normed, c=minutes, lw=1, zorder=2)
        ax2.scatter(minutes, ang, c='k', marker='.', s=5, 
                    alpha=1, zorder=10)
        # ax2.plot(ang, mbrfp_normed, c='r', lw=1, zorder=2, alpha=0.5)
        # ax2.plot(ang, mbrfp_normed, c=minutes, lw=1, zorder=2, alpha=0.5)
        # ax2.plot(angle_predict_deg*2, c='grey', lw=1, ls='--', zorder=1)
        ax2.set_ylim(*lim_ang)
        ax2.set_xlim(0,30)
        ax2.axhline(180, c='grey', lw=1, zorder=0)
        # ax2.axhline()
        # ax2.set_xlabel('Average Angle (deg)')
        # ax2.set_ylabel('Contact fluor. / Non-contact fluor.')
        ax2.set_adjustable('box')
        fig.savefig(Path.joinpath(out_dir, 'ang_cont_fluor_%s/ang_time.tif'%name), 
                    dpi=dpi_graph, bbox_inches='tight')

        fig, ax3 = plt.subplots(figsize=(width_1pt5col/3,width_1pt5col/3), nrows=1, ncols=1)
        # ax3.scatter(cont, la_normed, c=minutes, cmap=cmap, marker='.', s=5, 
        #             alpha=1, vmin=minutes.min(), vmax=minutes.max(), zorder=10)
        # ax3.plot(cont, la_normed, c='g', lw=1, zorder=2, alpha=0.5)
        # ax3.plot(cont, la_normed, c=minutes, lw=1, zorder=2, alpha=0.5)
        ax3.scatter(minutes, cont, c='k', marker='.', s=5, 
                    alpha=1, zorder=10)
        # ax3.plot(cont, mbrfp_normed, c='r', lw=1, zorder=2, alpha=0.5)
        # ax3.plot(cont, mbrfp_normed, c=minutes, lw=1, zorder=2, alpha=0.5)
        # ax3.plot(contnd_predict*2, c='grey', lw=1, ls='--', zorder=1)
        ax3.set_ylim(*lim_contnd)
        ax3.set_yticks([0,1,2,3])
        # ax3.set_xticklabels([0,1,2,3])
        ax3.set_xlim(0,30)
        ax3.axhline(expect_max_contnd, c='grey', lw=1, zorder=0)
        # ax3.set_xlabel('Contact Diameter (N.D.)')
        # ax3.set_ylabel('Contact fluor. / Non-contact fluor.')
        ax3.set_adjustable('box')
        fig.savefig(Path.joinpath(out_dir, 'ang_cont_fluor_%s/contnd_time.tif'%name), 
                    dpi=dpi_graph, bbox_inches='tight')
        plt.close('all')




# Try some 3D plots: want to see if dissociation buffer data 
# of angle vs contnd goes in circles, but too much overlap,
# so what if time plotted on third axis?
# cmap_name = 'jet'
# cmap = plt.cm.get_cmap(cmap_name)
# for name, group in la_contnd_df.groupby('Source_File'):
    
#     if np.all(group['Experimental_Condition'] == 'ee1xDB'):
#     #     lim_ang = (0, 200)#(0, 100)
#     #     lim_contnd = (0, 3)#(0, 1.5)
#     # else:
#     #     lim_ang = (0, 200)
#     #     lim_contnd = (0, 3)
        
#         lim_ang = (0, 200)
#         lim_contnd = (0, 3)
    
#         cont = group['Contact Surface Length (N.D.)']
#         ang = group['Average Angle (deg)']
#         la_normed = group['lifeact_fluor_normed_mean']
#         mbrfp_normed = group['mbrfp_fluor_normed_mean']
#         minutes = group['Time (minutes)']
#         # print('name: %s'%name,' / ','minutes.min: %f'%minutes.min(),' / ','minutes.max: %f'%minutes.max() )
        
#         fig = plt.figure(figsize=(8,8))
#         fig.suptitle(name)
#         ax = fig.add_subplot(projection='3d')
#         ax.scatter(ang, cont, minutes, marker='.', s=5, alpha=1, zorder=10)
#         colorcoded_lineplot_3d(ang, cont, minutes, ax, use_colormap=True)
#         # ax.plot(ang, cont, minutes, lw=1, zorder=2, alpha=0.5)
#         # colorcoded_arrowplot(ang, cont, ax, alpha=0.7, use_colormap=True, cmap=cmap_name)
#         # ax.plot(angle_predict_deg*2, contnd_predict*2, c='k', lw=1, ls='--', zorder=1)
#         # ax.plot(r30simLA_contnd_df['Average Angle (deg)'], 
#         #          r30simLA_contnd_df['Contact Surface Length (N.D.)'], 
#         #          c='k', marker='.', markersize=3, lw=1, ls='-', zorder=2)
#         # ax.set_xlim(*lim_ang)
#         # ax.set_ylim(*lim_contnd)
#         # ax1.axvline(180, c='grey', lw=1, zorder=0)
#         # ax1.axhline(expect_max_contnd, c='grey', lw=1, zorder=0)
#         ax.set_xlabel('Average Angle (deg)')
#         ax.set_ylabel('Contact Diameter (N.D.)')
#         ax.set_zlabel('Time (minutes)')
#         # plt.draw()
#         plt.show()
    
    
#     # ax.set_adjustable('box')
# multipdf(Path.joinpath(out_dir, r"cont_vs_angle_vs_time_3d.pdf"))
# plt.close('all')













# Do 2D plots of Bc/B, measured angle, contact diameter vs each other:

# 2D plots for each timelapse separately:
cmap = plt.cm.get_cmap(cmap_name)
for name, group in la_contnd_df.groupby('Source_File'):
    
    if np.all(group['Experimental_Condition'] == 'ee1xDB'):
        lim_ang = (0, 200)#(0, 100)
        lim_contnd = (0, 3)#(0, 1.5)
    else:
        lim_ang = (0, 200)
        lim_contnd = (0, 3)
    
    cont = group['Contact Surface Length (N.D.)']
    ang = group['Average Angle (deg)']
    la_normed = group['lifeact_fluor_normed_mean']
    # mbrfp_normed = group['mbrfp_fluor_normed_mean']
    minutes = group['Time (minutes)']
    # print('name: %s'%name,' / ','minutes.min: %f'%minutes.min(),' / ','minutes.max: %f'%minutes.max() )
    
    fig, (ax1,ax2,ax3) = plt.subplots(figsize=(13,4), nrows=1, ncols=3)
    fig.suptitle(name)
    graph = ax1.scatter(ang, cont, c=minutes, cmap=cmap, marker='.', s=5, 
                        alpha=1, vmin=minutes.min(), vmax=minutes.max(), zorder=10)
    # ax1.plot(ang, cont, c='grey', lw=1, zorder=2)
    colorcoded_arrowplot(ang, cont, ax1, alpha=0.7, use_colormap=True, cmap=cmap_name)
    ax1.plot(angle_predict_deg*2, contnd_predict*2, c='k', lw=1, ls='--', zorder=1)
    ax1.set_xlim(*lim_ang)
    ax1.set_ylim(*lim_contnd)
    ax1.axvline(180, c='grey', lw=1, zorder=0)
    ax1.axhline(expect_max_contnd, c='grey', lw=1, zorder=0)
    ax1.set_xlabel('Average Angle (deg)')
    ax1.set_ylabel('Contact Diameter (N.D.)')
    ax1.set_adjustable('box')

    ax2.scatter(ang, la_normed, c=minutes, cmap=cmap, marker='.', s=5, 
                alpha=1, vmin=minutes.min(), vmax=minutes.max(), zorder=10)
    ax2.plot(ang, la_normed, c='grey', lw=1, zorder=2)
    # ax2.scatter(ang, mbrfp_normed, c=minutes, cmap=cmap, marker='.', s=5, 
    #             alpha=1, vmin=minutes.min(), vmax=minutes.max(), zorder=10)
    # ax2.plot(ang, mbrfp_normed, c='r', lw=1, zorder=2)
    ax2.plot(angle_predict_deg*2, beta_star_div_beta_contnd_range, c='k', lw=1, ls='--', zorder=1)
    ax2.set_xlim(*lim_ang)
    ax2.set_ylim(0,2)
    ax2.axvline(180, c='grey', lw=1, zorder=0)
    # ax2.axhline()
    ax2.set_xlabel('Average Angle (deg)')
    ax2.set_ylabel(r'$\beta_c/\beta$')
    ax2.set_adjustable('box')

    ax3.scatter(cont, la_normed, c=minutes, cmap=cmap, marker='.', s=5, 
                alpha=1, vmin=minutes.min(), vmax=minutes.max(), zorder=10)
    ax3.plot(cont, la_normed, c='grey', lw=1, zorder=2)
    # ax3.plot(cont, la_normed, c=minutes, lw=1, zorder=2)
    # ax3.scatter(cont, mbrfp_normed, c=minutes, cmap=cmap, marker='.', s=5, 
    #             alpha=1, vmin=minutes.min(), vmax=minutes.max(), zorder=10)
    # ax3.plot(cont, mbrfp_normed, c='r', lw=1, zorder=2)
    ax3.plot(contnd_predict*2, beta_star_div_beta_contnd_range, c='k', lw=1, ls='--', zorder=1)
    ax3.set_xlim(*lim_contnd)
    ax3.set_ylim(0,2)
    ax3.axvline(expect_max_contnd, c='grey', lw=1, zorder=0)
    ax3.set_xlabel('Contact Diameter (N.D.)')
    ax3.set_ylabel(r'$\beta_c/\beta$')
    ax3.set_adjustable('box')
    
    # Include a colorbar for the timepoints
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(graph, cax=cax)

    plt.subplots_adjust(wspace=0.3, hspace=0.3)

multipdf(Path.joinpath(out_dir, r"cont_vs_angle_vs_Bc.pdf"))
plt.close('all')



# Do 2D plots of mbRFP equivalent of Bc/B, measured angle, 
# contact diameter vs each other:

# 2D plots for each timelapse separately:
cmap = plt.cm.get_cmap(cmap_name)
for name, group in la_contnd_df.groupby('Source_File'):
    
    if np.all(group['Experimental_Condition'] == 'ee1xDB'):
        lim_ang = (0, 200)#(0, 100)
        lim_contnd = (0, 3)#(0, 1.5)
    else:
        lim_ang = (0, 200)
        lim_contnd = (0, 3)

    cont = group['Contact Surface Length (N.D.)']
    ang = group['Average Angle (deg)']
    mbrfp_normed = group['mbrfp_fluor_normed_mean']
    minutes = group['Time (minutes)']
    # print('name: %s'%name,' / ','minutes.min: %f'%minutes.min(),' / ','minutes.max: %f'%minutes.max() )
    
    fig, (ax1,ax2,ax3) = plt.subplots(figsize=(13,4), nrows=1, ncols=3)
    fig.suptitle(name)
    graph = ax1.scatter(ang, cont, c=minutes, cmap=cmap, marker='.', s=5, 
                        alpha=1, vmin=minutes.min(), vmax=minutes.max(), zorder=10)
    # ax1.plot(ang, cont, c='grey', lw=1, zorder=2)
    colorcoded_arrowplot(ang, cont, ax1, alpha=0.7, use_colormap=True, cmap=cmap_name)
    ax1.plot(angle_predict_deg*2, contnd_predict*2, c='k', lw=1, ls='--', zorder=1)
    ax1.set_xlim(*lim_ang)
    ax1.set_ylim(*lim_contnd)
    ax1.axvline(180, c='grey', lw=1, zorder=0)
    ax1.axhline(expect_max_contnd, c='grey', lw=1, zorder=0)
    ax1.set_xlabel('Average Angle (deg)')
    ax1.set_ylabel('Contact Diameter (N.D.)')
    ax1.set_adjustable('box')

    ax2.scatter(ang, mbrfp_normed, c=minutes, cmap=cmap, marker='.', s=5, 
                alpha=1, vmin=minutes.min(), vmax=minutes.max(), zorder=10)
    ax2.plot(ang, mbrfp_normed, c='grey', lw=1, zorder=2)
    ax2.plot(angle_predict_deg*2, beta_star_div_beta_contnd_range, c='k', lw=1, ls='--', zorder=1)
    ax2.set_xlim(*lim_ang)
    ax2.set_ylim(0,2)
    ax2.axvline(180, c='grey', lw=1, zorder=0)
    ax2.set_xlabel('Average Angle (deg)')
    ax2.set_ylabel(r'$\beta_c/\beta$')
    ax2.set_adjustable('box')

    ax3.scatter(cont, mbrfp_normed, c=minutes, cmap=cmap, marker='.', s=5, 
                alpha=1, vmin=minutes.min(), vmax=minutes.max(), zorder=10)
    ax3.plot(cont, mbrfp_normed, c='grey', lw=1, zorder=2)
    ax3.plot(contnd_predict*2, beta_star_div_beta_contnd_range, c='k', lw=1, ls='--', zorder=1)
    ax3.set_xlim(*lim_contnd)
    ax3.set_ylim(0,2)
    ax3.axvline(expect_max_contnd, c='grey', lw=1, zorder=0)
    ax3.set_xlabel('Contact Diameter (N.D.)')
    ax3.set_ylabel(r'$\beta_c/\beta$')
    ax3.set_adjustable('box')
    
    # Include a colorbar for the timepoints
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(graph, cax=cax)

    plt.subplots_adjust(wspace=0.3, hspace=0.3)

multipdf(Path.joinpath(out_dir, r"cont_vs_angle_vs_mbRFP.pdf"))
plt.close('all')



# Plot LA-GFP vs mbRFP for each timelapse to see if they form relative straight 
# lines. Then we can try running correlation tests to each:
### Several assumptions are required for a Pearson correlation test:
### 1) variables are continuous : true
### 2) variables are paired : true
### 3) independence of cases : questionnable, I'm leaning on no...
### 4) relationship between data is linear : plots suggest otherwise for many timelapses
### 5) both variables have normal distributions : pretty sure that this is false
### 6) data is homoscedastic : data looks more heteroscedastic
### 7) no univariate or multivariate outliers : definitely have some 

pearson_r = list()
pearson_pval = list()
spearman_rho = list()
spearman_pval = list()
kendall_tau = list()
kendall_pval = list()

pearson_r_cont_la = list()
pearson_pval_cont_la = list()
spearman_rho_cont_la = list()
spearman_pval_cont_la = list()
kendall_tau_cont_la = list()
kendall_pval_cont_la = list()

pearson_r_cont_mb = list()
pearson_pval_cont_mb = list()
spearman_rho_cont_mb = list()
spearman_pval_cont_mb = list()
kendall_tau_cont_mb = list()
kendall_pval_cont_mb = list()

for name, group in la_contnd_df.groupby('Source_File'):
    # Missing/Invalid values will mess up correlation coefficients; drop them:
    group = group.mask(pd.isna(group).any(axis=1)).dropna()
    
    cont = group['Contact Surface Length (N.D.)']
    ang = group['Average Angle (deg)']
    la_cont = group['grnchan_fluor_at_contact_mean']
    la_sem = group['grnchan_fluor_at_contact_sem']
    mbrfp_cont = group['redchan_fluor_at_contact_mean']
    mbrfp_sem = group['redchan_fluor_at_contact_sem']
    minutes = group['Time (minutes)']
    # print('name: %s'%name,' / ','minutes.min: %f'%minutes.min(),' / ','minutes.max: %f'%minutes.max() )
    
    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(18,6), nrows=1, ncols=3)
    fig.suptitle(name)
    graph = ax1.errorbar(mbrfp_cont, la_cont, xerr=mbrfp_sem, yerr=la_sem, 
                         marker='.', markersize=5, ls='', 
                         alpha=1, color='k', ecolor='lightgrey', 
                         #min=minutes.min(), vmax=minutes.max(), zorder=10
                         )
    # ax.plot(ang, cont, c='grey', lw=1, zorder=2, alpha=0.5)
    # ax.plot(angle_predict_deg*2, contnd_predict*2, c='k', lw=1, ls='--', zorder=1)
    try:
        ax1.set_xlim(0,mbrfp_cont.max()+mbrfp_sem.max()+10)
        ax1.set_ylim(0,la_cont.max()+la_sem.max()+10)
    except(ValueError):
        ax1.set_xlim(0, 10)
        ax1.set_ylim(0, 10)
    # ax.axhline(expect_max_contnd, c='grey', lw=1, zorder=0)
    ax1.set_xlabel('mbRFP_fluorescence_mean (photons)')
    ax1.set_ylabel('LAGFP_fluorescence_mean (photons)')
    ax1.set_adjustable('box')

    graph = ax2.errorbar(cont, la_cont, xerr=None, yerr=la_sem, 
                         marker='.', markersize=5, ls='', 
                         alpha=1, color='k', ecolor='lightgrey', 
                         #min=minutes.min(), vmax=minutes.max(), zorder=10
                         )
    # ax.plot(ang, cont, c='grey', lw=1, zorder=2, alpha=0.5)
    # ax.plot(angle_predict_deg*2, contnd_predict*2, c='k', lw=1, ls='--', zorder=1)
    try:
        ax2.set_xlim(0, cont.max()+0.1)
        ax2.set_ylim(0, la_cont.max()+la_sem.max()+10)
    except(ValueError):
        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, 10)
    # ax.axvline(180, c='grey', lw=1, zorder=0)
    # ax.axhline(expect_max_contnd, c='grey', lw=1, zorder=0)
    ax2.set_xlabel('Contact Surface Length (N.D.)')
    ax2.set_ylabel('LAGFP_fluorescence_mean (photons)')
    ax2.set_adjustable('box')

    graph = ax3.errorbar(cont, mbrfp_cont, xerr=None, yerr=mbrfp_sem, 
                         marker='.', markersize=5, ls='', 
                         alpha=1, color='k', ecolor='lightgrey', 
                         #min=minutes.min(), vmax=minutes.max(), zorder=10
                         )
    # ax.plot(ang, cont, c='grey', lw=1, zorder=2, alpha=0.5)
    # ax.plot(angle_predict_deg*2, contnd_predict*2, c='k', lw=1, ls='--', zorder=1)
    try:
        ax3.set_xlim(0, cont.max()+0.1)
        ax3.set_ylim(0, mbrfp_cont.max()+mbrfp_sem.max()+10)
    except(ValueError):
        ax3.set_xlim(0, 0.1)
        ax3.set_ylim(0, 0.1)
    # ax.axvline(180, c='grey', lw=1, zorder=0)
    # ax.axhline(expect_max_contnd, c='grey', lw=1, zorder=0)
    ax3.set_xlabel('Contact Surface Length (N.D.)')
    ax3.set_ylabel('mbRFP_fluorescence_mean (photons)')
    ax3.set_adjustable('box')

    
    # Include a colorbar for the timepoints
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    # cbar = plt.colorbar(graph, cax=cax)

    # plt.subplots_adjust(wspace=0.3, hspace=0.3)
    
    # Calculate some correlation measures:
    # Pearson:
    try:
        prsn_la_mb = stats.pearsonr(mbrfp_cont, la_cont)
        sprmn_la_mb = stats.spearmanr(mbrfp_cont, la_cont)
        kndll_la_mb = stats.kendalltau(mbrfp_cont, la_cont)
        ax1.text(0.05, 0.93, 'r=%.3f'%prsn_la_mb[0], ha='left', transform=ax1.transAxes)
        ax1.text(0.05, 0.9, 'pval=%.3E'%prsn_la_mb[1], ha='left', transform=ax1.transAxes)
        ax1.text(0.5, 0.93, r'$\rho$=%.3f'%sprmn_la_mb[0], ha='center', transform=ax1.transAxes)
        ax1.text(0.5, 0.9, 'pval=%.3E'%sprmn_la_mb[1], ha='center', transform=ax1.transAxes)
        ax1.text(0.95, 0.93, r'$\tau$=%.3f'%kndll_la_mb[0], ha='right', transform=ax1.transAxes)
        ax1.text(0.95, 0.9, 'pval=%.3E'%kndll_la_mb[1], ha='right', transform=ax1.transAxes)
        
        prsn_cont_la = stats.pearsonr(cont, la_cont)
        sprmn_cont_la = stats.spearmanr(cont, la_cont)
        kndll_cont_la = stats.kendalltau(cont, la_cont)
        ax2.text(0.05, 0.93, 'r=%.3f'%prsn_cont_la[0], ha='left', transform=ax2.transAxes)
        ax2.text(0.05, 0.9, 'pval=%.3E'%prsn_cont_la[1], ha='left', transform=ax2.transAxes)
        ax2.text(0.5, 0.93, r'$\rho$=%.3f'%sprmn_cont_la[0], ha='center', transform=ax2.transAxes)
        ax2.text(0.5, 0.9, 'pval=%.3E'%sprmn_cont_la[1], ha='center', transform=ax2.transAxes)
        ax2.text(0.95, 0.93, r'$\tau$=%.3f'%kndll_cont_la[0], ha='right', transform=ax2.transAxes)
        ax2.text(0.95, 0.9, 'pval=%.3E'%kndll_cont_la[1], ha='right', transform=ax2.transAxes)
    
        prsn_cont_mb = stats.pearsonr(cont, mbrfp_cont)
        sprmn_cont_mb = stats.spearmanr(cont, mbrfp_cont)
        kndll_cont_mb = stats.kendalltau(cont, mbrfp_cont)
        ax3.text(0.05, 0.93, 'r=%.3f'%prsn_cont_mb[0], ha='left', transform=ax3.transAxes)
        ax3.text(0.05, 0.9, 'pval=%.3E'%prsn_cont_mb[1], ha='left', transform=ax3.transAxes)
        ax3.text(0.5, 0.93, r'$\rho$=%.3f'%sprmn_cont_mb[0], ha='center', transform=ax3.transAxes)
        ax3.text(0.5, 0.9, 'pval=%.3E'%sprmn_cont_mb[1], ha='center', transform=ax3.transAxes)
        ax3.text(0.95, 0.93, r'$\tau$=%.3f'%kndll_cont_mb[0], ha='right', transform=ax3.transAxes)
        ax3.text(0.95, 0.9, 'pval=%.3E'%kndll_cont_mb[1], ha='right', transform=ax3.transAxes)
    except(ValueError):
        prsn_la_mb = [np.nan, np.nan]
        sprmn_la_mb = [np.nan, np.nan]
        kndll_la_mb = [np.nan, np.nan]
    
    # Add the correlation data to a dataframe:
    pearson_r.append(prsn_la_mb[0])
    pearson_pval.append(prsn_la_mb[1])
    spearman_rho.append(sprmn_la_mb[0])
    spearman_pval.append(sprmn_la_mb[1])
    kendall_tau.append(kndll_la_mb[0])
    kendall_pval.append(kndll_la_mb[1])

    pearson_r_cont_la.append(prsn_cont_la[0])
    pearson_pval_cont_la.append(prsn_cont_la[1])
    spearman_rho_cont_la.append(sprmn_cont_la[0])
    spearman_pval_cont_la.append(sprmn_cont_la[1])
    kendall_tau_cont_la.append(kndll_cont_la[0])
    kendall_pval_cont_la.append(kndll_cont_la[1])

    pearson_r_cont_mb.append(prsn_cont_mb[0])
    pearson_pval_cont_mb.append(prsn_cont_mb[1])
    spearman_rho_cont_mb.append(sprmn_cont_mb[0])
    spearman_pval_cont_mb.append(sprmn_cont_mb[1])
    kendall_tau_cont_mb.append(kndll_cont_mb[0])
    kendall_pval_cont_mb.append(kndll_cont_mb[1])




multipdf(Path.joinpath(out_dir, r"correlations.pdf"))
plt.close('all')

# Add the correlation data to the la_contnd_df dataframe:
sf = [name for (name,df) in la_contnd_df.groupby('Source_File')]
ec = [df['Experimental_Condition'].iloc[-1] for (name,df) in la_contnd_df.groupby('Source_File')]

df = {'pearson_r_la_mb': pearson_r, 'pearson_pval_la_mb': pearson_pval, 
      'spearman_rho_la_mb': spearman_rho, 'spearman_pval_la_mb': spearman_pval, 
      'kendall_tau_la_mb': kendall_tau, 'kendall_pval_la_mb': kendall_pval, 
      'pearson_r_cont_la': pearson_r_cont_la, 'pearson_pval_cont_la': pearson_pval_cont_la, 
      'spearman_rho_cont_la': spearman_rho_cont_la, 'spearman_pval_cont_la': spearman_pval_cont_la, 
      'kendall_tau_cont_la': kendall_tau_cont_la, 'kendall_pval_cont_la': kendall_pval_cont_la, 
      'pearson_r_cont_mb': pearson_r_cont_mb, 'pearson_pval_cont_mb': pearson_pval_cont_mb, 
      'spearman_rho_cont_mb': spearman_rho_cont_mb, 'spearman_pval_cont_mb': spearman_pval_cont_mb, 
      'kendall_tau_cont_mb': kendall_tau_cont_mb, 'kendall_pval_cont_mb': kendall_pval_cont_mb,       
      'Source_File': sf, 
      'Experimental_Condition': ec
      }

corr_df = pd.DataFrame(data=df)

fig, ((ax1,ax2,ax3), (ax4,ax5,ax6), (ax7,ax8,ax9)) = plt.subplots(nrows=3, ncols=3, 
                                                                  figsize=(width_1col*4, width_1col*4))

#
prsn_la_mb_corr = sns.pointplot(y='Source_File', x='pearson_r_la_mb', 
                                data=corr_df[corr_df['Experimental_Condition'] == 'ee1xMBS'], 
                                color='r', markers='+', linestyles='', ax=ax1)
ax1.set_xlim(-1,1)
[ax1.axvline(val, c='lightgrey', lw=1, zorder=0) for val in [-0.5, -0.3, -0.1, 0, 0.1, 0.3, 0.5]]
ax1.set_xlabel('')
ax1b = ax1.twiny()
prsn_la_mb_pval = sns.pointplot(y='Source_File', x='pearson_pval_la_mb', 
                                data=corr_df[corr_df['Experimental_Condition'] == 'ee1xMBS'], 
                                color='k', markers='x', linestyles='', ax=ax1b)
ax1b.set_xscale('log')

prsn_cont_la_corr = sns.pointplot(y='Source_File', x='pearson_r_cont_la', 
                                  data=corr_df[corr_df['Experimental_Condition'] == 'ee1xMBS'], 
                                  color='r', markers='+', linestyles='', ax=ax2)
ax2.set_xlim(-1,1)
[ax2.axvline(val, c='lightgrey', lw=1, zorder=0) for val in [-0.5, -0.3, -0.1, 0, 0.1, 0.3, 0.5]]
ax2.set_yticklabels('')
ax2.set_ylabel('')
ax2.set_xlabel('')
ax2b = ax2.twiny()
prsn_cont_la_pval = sns.pointplot(y='Source_File', x='pearson_pval_cont_la', 
                                  data=corr_df[corr_df['Experimental_Condition'] == 'ee1xMBS'], 
                                  color='k', markers='x', linestyles='', ax=ax2b)
ax2b.set_xscale('log')
ax2b.set_yticklabels('')
ax2b.set_ylabel('')

prsn_cont_mb_corr = sns.pointplot(y='Source_File', x='pearson_r_cont_mb', 
                                  data=corr_df[corr_df['Experimental_Condition'] == 'ee1xMBS'], 
                                  color='r', markers='+', linestyles='', ax=ax3)
ax3.set_xlim(-1,1)
[ax3.axvline(val, c='lightgrey', lw=1, zorder=0) for val in [-0.5, -0.3, -0.1, 0, 0.1, 0.3, 0.5]]
ax3.set_yticklabels('')
ax3.set_ylabel('')
ax3.set_xlabel('')
ax3b = ax3.twiny()
prsn_cont_mb_pval = sns.pointplot(y='Source_File', x='pearson_pval_cont_mb', 
                                  data=corr_df[corr_df['Experimental_Condition'] == 'ee1xMBS'], 
                                  color='k', markers='x', linestyles='', ax=ax3b)
ax3b.set_xscale('log')
ax3b.set_yticklabels('')
ax3b.set_ylabel('')

#

sprmn_la_mb_corr = sns.pointplot(y='Source_File', x='spearman_rho_la_mb', 
                                 data=corr_df[corr_df['Experimental_Condition'] == 'ee1xMBS'], 
                                 color='r', markers='+', linestyles='', ax=ax4)
ax4.set_xlim(-1,1)
[ax4.axvline(val, c='lightgrey', lw=1, zorder=0) for val in [-0.5, -0.3, -0.1, 0, 0.1, 0.3, 0.5]]
ax4.set_xlabel('')
ax4b = ax4.twiny()
sprmn_la_mb_pval = sns.pointplot(y='Source_File', x='spearman_pval_la_mb', 
                                 data=corr_df[corr_df['Experimental_Condition'] == 'ee1xMBS'], 
                     color='k', markers='x', linestyles='', ax=ax4b)
ax4b.set_xscale('log')
ax4b.set_xlabel('')


sprmn_cont_la_corr = sns.pointplot(y='Source_File', x='spearman_rho_cont_la', 
                                   data=corr_df[corr_df['Experimental_Condition'] == 'ee1xMBS'], 
                                   color='r', markers='+', linestyles='', ax=ax5)
ax5.set_xlim(-1,1)
[ax5.axvline(val, c='lightgrey', lw=1, zorder=0) for val in [-0.5, -0.3, -0.1, 0, 0.1, 0.3, 0.5]]
ax5.set_yticklabels('')
ax5.set_ylabel('')
ax5.set_xlabel('')
ax5b = ax5.twiny()
sprmn_cont_la_pval = sns.pointplot(y='Source_File', x='spearman_pval_cont_la', 
                                   data=corr_df[corr_df['Experimental_Condition'] == 'ee1xMBS'], 
                                   color='k', markers='x', linestyles='', ax=ax5b)
ax5b.set_xscale('log')
ax5b.set_yticklabels('')
ax5b.set_ylabel('')
ax5b.set_xlabel('')


sprmn_cont_mb_corr = sns.pointplot(y='Source_File', x='spearman_rho_cont_mb', 
                                   data=corr_df[corr_df['Experimental_Condition'] == 'ee1xMBS'], 
                                   color='r', markers='+', linestyles='', ax=ax6)
ax6.set_xlim(-1,1)
[ax6.axvline(val, c='lightgrey', lw=1, zorder=0) for val in [-0.5, -0.3, -0.1, 0, 0.1, 0.3, 0.5]]
ax6.set_yticklabels('')
ax6.set_ylabel('')
ax6.set_xlabel('')
ax6b = ax6.twiny()
sprmn_cont_mb_pval = sns.pointplot(y='Source_File', x='spearman_pval_cont_mb', 
                                   data=corr_df[corr_df['Experimental_Condition'] == 'ee1xMBS'], 
                                   color='k', markers='x', linestyles='', ax=ax6b)
ax6b.set_xscale('log')
ax6b.set_yticklabels('')
ax6b.set_ylabel('')
ax6b.set_xlabel('')

#

kndll_la_mb_corr = sns.pointplot(y='Source_File', x='kendall_tau_la_mb', 
                                 data=corr_df[corr_df['Experimental_Condition'] == 'ee1xMBS'], 
                                 color='r', markers='+', linestyles='', ax=ax7)
ax7.set_xlim(-1,1)
[ax7.axvline(val, c='lightgrey', lw=1, zorder=0) for val in [-0.5, -0.3, -0.1, 0, 0.1, 0.3, 0.5]]
ax7b = ax7.twiny()
kndll_la_mb_pval = sns.pointplot(y='Source_File', x='kendall_pval_la_mb', 
                                 data=corr_df[corr_df['Experimental_Condition'] == 'ee1xMBS'], 
                                 color='k', markers='x', linestyles='', ax=ax7b)
ax7b.set_xscale('log')
ax7b.set_xlabel('')

kndll_cont_la_corr = sns.pointplot(y='Source_File', x='kendall_tau_cont_la', 
                                   data=corr_df[corr_df['Experimental_Condition'] == 'ee1xMBS'], 
                                   color='r', markers='+', linestyles='', ax=ax8)
ax8.set_xlim(-1,1)
[ax8.axvline(val, c='lightgrey', lw=1, zorder=0) for val in [-0.5, -0.3, -0.1, 0, 0.1, 0.3, 0.5]]
ax8.set_yticklabels('')
ax8.set_ylabel('')
ax8b = ax8.twiny()
kndll_cont_la_pval = sns.pointplot(y='Source_File', x='kendall_pval_cont_la', 
                                   data=corr_df[corr_df['Experimental_Condition'] == 'ee1xMBS'], 
                                   color='k', markers='x', linestyles='', ax=ax8b)
ax8b.set_xscale('log')
ax8b.set_yticklabels('')
ax8b.set_ylabel('')
ax8b.set_xlabel('')


kndll_cont_mb_corr = sns.pointplot(y='Source_File', x='kendall_tau_cont_mb', 
                                   data=corr_df[corr_df['Experimental_Condition'] == 'ee1xMBS'], 
                                   color='r', markers='+', linestyles='', ax=ax9)
ax9.set_xlim(-1,1)
[ax9.axvline(val, c='lightgrey', lw=1, zorder=0) for val in [-0.5, -0.3, -0.1, 0, 0.1, 0.3, 0.5]]
ax9.set_yticklabels('')
ax9.set_ylabel('')
ax9b = ax9.twiny()
kndll_cont_mb_pval = sns.pointplot(y='Source_File', x='kendall_pval_cont_mb', 
                                   data=corr_df[corr_df['Experimental_Condition'] == 'ee1xMBS'], 
                                   color='k', markers='x', linestyles='', ax=ax9b)
ax9b.set_xscale('log')
ax9b.set_yticklabels('')
ax9b.set_ylabel('')
ax9b.set_xlabel('')
#


leg_lines = [mlines.Line2D([], [], color='r', marker='+', linestyle='', label='corr_coeff'), 
             mlines.Line2D([], [], color='k', marker='x', linestyle='', label='p_val')]
plt.legend(handles=leg_lines, bbox_to_anchor=(-3.5, 3.4, 1, .05), 
           loc='lower right', mode='expand', ncol=2)
# plt.tight_layout()
fig.savefig(Path.joinpath(out_dir, r'correlations_mbs.tif'), dpi=dpi_LineArt, bbox_inches='tight')
plt.close(fig)










fig, ((ax1,ax2,ax3), (ax4,ax5,ax6), (ax7,ax8,ax9)) = plt.subplots(nrows=3, ncols=3, 
                                                                  figsize=(width_1col*4, width_1col*4))

#
prsn_la_mb_corr = sns.pointplot(y='Source_File', x='pearson_r_la_mb', 
                                data=corr_df[corr_df['Experimental_Condition'] == 'ee1xDB'], 
                                color='r', markers='+', linestyles='', ax=ax1)
ax1.set_xlim(-1,1)
[ax1.axvline(val, c='lightgrey', lw=1, zorder=0) for val in [-0.5, -0.3, -0.1, 0, 0.1, 0.3, 0.5]]
ax1.set_xlabel('')
ax1b = ax1.twiny()
prsn_la_mb_pval = sns.pointplot(y='Source_File', x='pearson_pval_la_mb', 
                                data=corr_df[corr_df['Experimental_Condition'] == 'ee1xDB'], 
                                color='k', markers='x', linestyles='', ax=ax1b)
ax1b.set_xscale('log')

prsn_cont_la_corr = sns.pointplot(y='Source_File', x='pearson_r_cont_la', 
                                  data=corr_df[corr_df['Experimental_Condition'] == 'ee1xDB'], 
                                  color='r', markers='+', linestyles='', ax=ax2)
ax2.set_xlim(-1,1)
[ax2.axvline(val, c='lightgrey', lw=1, zorder=0) for val in [-0.5, -0.3, -0.1, 0, 0.1, 0.3, 0.5]]
ax2.set_yticklabels('')
ax2.set_ylabel('')
ax2.set_xlabel('')
ax2b = ax2.twiny()
prsn_cont_la_pval = sns.pointplot(y='Source_File', x='pearson_pval_cont_la', 
                                  data=corr_df[corr_df['Experimental_Condition'] == 'ee1xDB'], 
                                  color='k', markers='x', linestyles='', ax=ax2b)
ax2b.set_xscale('log')
ax2b.set_yticklabels('')
ax2b.set_ylabel('')

prsn_cont_mb_corr = sns.pointplot(y='Source_File', x='pearson_r_cont_mb', 
                                  data=corr_df[corr_df['Experimental_Condition'] == 'ee1xDB'], 
                                  color='r', markers='+', linestyles='', ax=ax3)
ax3.set_xlim(-1,1)
[ax3.axvline(val, c='lightgrey', lw=1, zorder=0) for val in [-0.5, -0.3, -0.1, 0, 0.1, 0.3, 0.5]]
ax3.set_yticklabels('')
ax3.set_ylabel('')
ax3.set_xlabel('')
ax3b = ax3.twiny()
prsn_cont_mb_pval = sns.pointplot(y='Source_File', x='pearson_pval_cont_mb', 
                                  data=corr_df[corr_df['Experimental_Condition'] == 'ee1xDB'], 
                                  color='k', markers='x', linestyles='', ax=ax3b)
ax3b.set_xscale('log')
ax3b.set_yticklabels('')
ax3b.set_ylabel('')

#

sprmn_la_mb_corr = sns.pointplot(y='Source_File', x='spearman_rho_la_mb', 
                                 data=corr_df[corr_df['Experimental_Condition'] == 'ee1xDB'], 
                                 color='r', markers='+', linestyles='', ax=ax4)
ax4.set_xlim(-1,1)
[ax4.axvline(val, c='lightgrey', lw=1, zorder=0) for val in [-0.5, -0.3, -0.1, 0, 0.1, 0.3, 0.5]]
ax4.set_xlabel('')
ax4b = ax4.twiny()
sprmn_la_mb_pval = sns.pointplot(y='Source_File', x='spearman_pval_la_mb', 
                                 data=corr_df[corr_df['Experimental_Condition'] == 'ee1xDB'], 
                     color='k', markers='x', linestyles='', ax=ax4b)
ax4b.set_xscale('log')
ax4b.set_xlabel('')


sprmn_cont_la_corr = sns.pointplot(y='Source_File', x='spearman_rho_cont_la', 
                                   data=corr_df[corr_df['Experimental_Condition'] == 'ee1xDB'], 
                                   color='r', markers='+', linestyles='', ax=ax5)
ax5.set_xlim(-1,1)
[ax5.axvline(val, c='lightgrey', lw=1, zorder=0) for val in [-0.5, -0.3, -0.1, 0, 0.1, 0.3, 0.5]]
ax5.set_yticklabels('')
ax5.set_ylabel('')
ax5.set_xlabel('')
ax5b = ax5.twiny()
sprmn_cont_la_pval = sns.pointplot(y='Source_File', x='spearman_pval_cont_la', 
                                   data=corr_df[corr_df['Experimental_Condition'] == 'ee1xDB'], 
                                   color='k', markers='x', linestyles='', ax=ax5b)
ax5b.set_xscale('log')
ax5b.set_yticklabels('')
ax5b.set_ylabel('')
ax5b.set_xlabel('')


sprmn_cont_mb_corr = sns.pointplot(y='Source_File', x='spearman_rho_cont_mb', 
                                   data=corr_df[corr_df['Experimental_Condition'] == 'ee1xDB'], 
                                   color='r', markers='+', linestyles='', ax=ax6)
ax6.set_xlim(-1,1)
[ax6.axvline(val, c='lightgrey', lw=1, zorder=0) for val in [-0.5, -0.3, -0.1, 0, 0.1, 0.3, 0.5]]
ax6.set_yticklabels('')
ax6.set_ylabel('')
ax6.set_xlabel('')
ax6b = ax6.twiny()
sprmn_cont_mb_pval = sns.pointplot(y='Source_File', x='spearman_pval_cont_mb', 
                                   data=corr_df[corr_df['Experimental_Condition'] == 'ee1xDB'], 
                                   color='k', markers='x', linestyles='', ax=ax6b)
ax6b.set_xscale('log')
ax6b.set_yticklabels('')
ax6b.set_ylabel('')
ax6b.set_xlabel('')

#

kndll_la_mb_corr = sns.pointplot(y='Source_File', x='kendall_tau_la_mb', 
                                 data=corr_df[corr_df['Experimental_Condition'] == 'ee1xDB'], 
                                 color='r', markers='+', linestyles='', ax=ax7)
ax7.set_xlim(-1,1)
[ax7.axvline(val, c='lightgrey', lw=1, zorder=0) for val in [-0.5, -0.3, -0.1, 0, 0.1, 0.3, 0.5]]
ax7b = ax7.twiny()
kndll_la_mb_pval = sns.pointplot(y='Source_File', x='kendall_pval_la_mb', 
                                 data=corr_df[corr_df['Experimental_Condition'] == 'ee1xDB'], 
                                 color='k', markers='x', linestyles='', ax=ax7b)
ax7b.set_xscale('log')
ax7b.set_xlabel('')

kndll_cont_la_corr = sns.pointplot(y='Source_File', x='kendall_tau_cont_la', 
                                   data=corr_df[corr_df['Experimental_Condition'] == 'ee1xDB'], 
                                   color='r', markers='+', linestyles='', ax=ax8)
ax8.set_xlim(-1,1)
[ax8.axvline(val, c='lightgrey', lw=1, zorder=0) for val in [-0.5, -0.3, -0.1, 0, 0.1, 0.3, 0.5]]
ax8.set_yticklabels('')
ax8.set_ylabel('')
ax8b = ax8.twiny()
kndll_cont_la_pval = sns.pointplot(y='Source_File', x='kendall_pval_cont_la', 
                                   data=corr_df[corr_df['Experimental_Condition'] == 'ee1xDB'], 
                                   color='k', markers='x', linestyles='', ax=ax8b)
ax8b.set_xscale('log')
ax8b.set_yticklabels('')
ax8b.set_ylabel('')
ax8b.set_xlabel('')


kndll_cont_mb_corr = sns.pointplot(y='Source_File', x='kendall_tau_cont_mb', 
                                   data=corr_df[corr_df['Experimental_Condition'] == 'ee1xDB'], 
                                   color='r', markers='+', linestyles='', ax=ax9)
ax9.set_xlim(-1,1)
[ax9.axvline(val, c='lightgrey', lw=1, zorder=0) for val in [-0.5, -0.3, -0.1, 0, 0.1, 0.3, 0.5]]
ax9.set_yticklabels('')
ax9.set_ylabel('')
ax9b = ax9.twiny()
kndll_cont_mb_pval = sns.pointplot(y='Source_File', x='kendall_pval_cont_mb', 
                                   data=corr_df[corr_df['Experimental_Condition'] == 'ee1xDB'], 
                                   color='k', markers='x', linestyles='', ax=ax9b)
ax9b.set_xscale('log')
ax9b.set_yticklabels('')
ax9b.set_ylabel('')
ax9b.set_xlabel('')
#


leg_lines = [mlines.Line2D([], [], color='r', marker='+', linestyle='', label='corr_coeff'), 
             mlines.Line2D([], [], color='k', marker='x', linestyle='', label='p_val')]
plt.legend(handles=leg_lines, bbox_to_anchor=(-3.5, 3.4, 1, .05), 
           loc='lower right', mode='expand', ncol=2)
# plt.tight_layout()
fig.savefig(Path.joinpath(out_dir, r'correlations_db.tif'), dpi=dpi_LineArt, bbox_inches='tight')
plt.close(fig)





# fig, ax = plt.subplots(figsize=(width_1col, width_1col))
# coeff = sns.pointplot(y='Source_File', x='pearson_r', data=corr_df[corr_df['Experimental_Condition'] == 'ee1xDB'], 
#                       color='r', markers='+', linestyles='', ax=ax)
# ax.set_xlim(-1,1)
# ax2 = ax.twiny()
# pval = sns.pointplot(y='Source_File', x='pearson_pval', data=corr_df[corr_df['Experimental_Condition'] == 'ee1xDB'], 
#                      color='k', markers='x', linestyles='', ax=ax2)
# ax2.set_xscale('log')
# # plt.tight_layout()
# leg_lines = [mlines.Line2D([], [], color='r', marker='+', linestyle='', label='r'), 
#              mlines.Line2D([], [], color='k', marker='x', linestyle='', label='pval')]
# plt.legend(handles=leg_lines, bbox_to_anchor=(-1.0, 1.0, 1, .05), 
#            loc='lower right', mode='expand', ncol=2)
# fig.savefig(Path.joinpath(out_dir, r'pearson_db.tif'), dpi=dpi_LineArt, bbox_inches='tight')
# plt.close(fig)






# fig, ax = plt.subplots(figsize=(width_1col, width_1col))
# coeff = sns.pointplot(y='Source_File', x='spearman_rho', data=corr_df[corr_df['Experimental_Condition'] == 'ee1xDB'], 
#                       color='r', markers='+', linestyles='', ax=ax)
# ax.set_xlim(-1,1)
# ax2 = ax.twiny()
# pval = sns.pointplot(y='Source_File', x='spearman_pval', data=corr_df[corr_df['Experimental_Condition'] == 'ee1xDB'], 
#                      color='k', markers='x', linestyles='', ax=ax2)
# ax2.set_xscale('log')
# # plt.tight_layout()
# leg_lines = [mlines.Line2D([], [], color='r', marker='+', linestyle='', label='rho'), 
#              mlines.Line2D([], [], color='k', marker='x', linestyle='', label='pval')]
# plt.legend(handles=leg_lines, bbox_to_anchor=(-1.0, 1.0, 1, .05), 
#            loc='lower right', mode='expand', ncol=2)
# fig.savefig(Path.joinpath(out_dir, r'spearman_db.tif'), dpi=dpi_LineArt, bbox_inches='tight')
# plt.close(fig)





# fig, ax = plt.subplots(figsize=(width_1col, width_1col))
# coeff = sns.pointplot(y='Source_File', x='kendall_tau', data=corr_df[corr_df['Experimental_Condition'] == 'ee1xDB'], 
#                       color='r', markers='+', linestyles='', ax=ax)
# ax.set_xlim(-1,1)
# pval = sns.pointplot(y='Source_File', x='kendall_pval', data=corr_df[corr_df['Experimental_Condition'] == 'ee1xDB'], 
#                      color='k', markers='x', linestyles='', ax=ax2)
# ax2 = ax.twiny()
# ax2.set_xscale('log')
# # plt.tight_layout()
# leg_lines = [mlines.Line2D([], [], color='r', marker='+', linestyle='', label='tau'), 
#              mlines.Line2D([], [], color='k', marker='x', linestyle='', label='pval')]
# plt.legend(handles=leg_lines, bbox_to_anchor=(-1.0, 1.0, 1, .05), 
#            loc='lower right', mode='expand', ncol=2)
# fig.savefig(Path.joinpath(out_dir, r'kendall_db.tif'), dpi=dpi_LineArt, bbox_inches='tight')
# plt.close(fig)





# exit

### We're going to do a linear regression on the angle vs. contact fluor and
### contact diam vs contact fluor pairs of data for the ee1xDB exptl condition:
temp = la_contnd_df[la_contnd_df['Experimental_Condition'] == 'ee1xDB'][['Average Angle (deg)', 'Contact Surface Length (N.D.)', 'lifeact_fluor_normed_mean', 'Source_File']]
temp_cln = temp.dropna(axis=0)
temp_ang = temp_cln['Average Angle (deg)']
temp_cont = temp_cln['Contact Surface Length (N.D.)']
temp_fluor = temp_cln['lifeact_fluor_normed_mean']

## perform the actual regression:
linreg_angvfluor_overall = stats.linregress(temp_ang, temp_fluor)
linreg_contvfluor_overall = stats.linregress(temp_cont, temp_fluor)

linreg_angvfluor_ls = list()
linreg_contvfluor_ls = list()
for nm,grp in temp_cln.groupby('Source_File'):
    ang = grp['Average Angle (deg)']
    cont = grp['Contact Surface Length (N.D.)']
    fluor = grp['lifeact_fluor_normed_mean']
    
    linreg_angvfluor = stats.linregress(ang, fluor)
    linreg_contvfluor = stats.linregress(cont, fluor)
    
    ## we're going to make a table here of the different results from stats.linregress
    interc = ('intercept',[linreg_angvfluor.intercept])
    pval = ('pvalue',[linreg_angvfluor.pvalue])
    rval = ('rvalue',[linreg_angvfluor.rvalue])
    slp = ('slope',[linreg_angvfluor.slope])
    serr = ('stderr',[linreg_angvfluor.stderr])
    sf = ('Source_File',[nm])
    cols=dict([interc, pval, rval, slp, serr, sf])
    linreg_angvfluor_df = pd.DataFrame(data=cols)
    
    interc = ('intercept',[linreg_contvfluor.intercept])
    pval = ('pvalue',[linreg_contvfluor.pvalue])
    rval = ('rvalue',[linreg_contvfluor.rvalue])
    slp = ('slope',[linreg_contvfluor.slope])
    serr = ('stderr',[linreg_contvfluor.stderr])
    sf = ('Source_File',[nm])
    cols=dict([interc, pval, rval, slp, serr, sf])
    linreg_contvfluor_df = pd.DataFrame(data=cols)
    
    linreg_angvfluor_ls.append(linreg_angvfluor_df)
    linreg_contvfluor_ls.append(linreg_contvfluor_df)
    
linreg_angvfluor_df = pd.concat(linreg_angvfluor_ls)
linreg_angvfluor_df_long = linreg_angvfluor_df.melt(var_name='result_name', value_name='result_value')
linreg_contvfluor_df = pd.concat(linreg_contvfluor_ls)
linreg_contvfluor_df_long = linreg_contvfluor_df.melt(var_name='result_name', value_name='result_value')

# sns.catplot(x='result_name', y='result_value', data=linreg_contvfluor_df_long)

# np.count_nonzero(linreg_angvfluor_df.pvalue < 0.05)
# len(linreg_angvfluor_df)
### only 4 / 14 have linreg_angvfluor_df.pvalue < 0.05

# np.count_nonzero(linreg_angvfluor_df.slope < 0)
# np.count_nonzero(linreg_angvfluor_df.rvalue < 0)
# len(linreg_angvfluor_df)
### most (11 / 14) examples have negative slopes and rvalues despite most 
### R^2 values being statistically not different from a slope of 0

### try centering the different timelapses about 0?
# linreg_angvfluor_ls = list()
# linreg_contvfluor_ls = list()
temp_cln_centered = list()
for nm,grp in temp_cln.groupby('Source_File'):
    grp['Average Angle (deg)'] = grp['Average Angle (deg)'] - grp['Average Angle (deg)'].mean()
    grp['Contact Surface Length (N.D.)'] = grp['Contact Surface Length (N.D.)'] - grp['Average Angle (deg)'].mean()
    grp['lifeact_fluor_normed_mean'] = grp['lifeact_fluor_normed_mean'] - grp['lifeact_fluor_normed_mean'].mean()
    grp['Source_File'] = [nm] * len(grp)
    
    temp_cln_centered.append(grp)
    
    # linreg_angvfluor = stats.linregress(ang, fluor)
    # linreg_contvfluor = stats.linregress(cont, fluor)
    
    # ## we're going to make a table here of the different results from stats.linregress
    # interc = ('intercept',[linreg_angvfluor.intercept])
    # pval = ('pvalue',[linreg_angvfluor.pvalue])
    # rval = ('rvalue',[linreg_angvfluor.rvalue])
    # slp = ('slope',[linreg_angvfluor.slope])
    # serr = ('stderr',[linreg_angvfluor.stderr])
    # cols=dict([interc, pval, rval, slp, serr])
    # linreg_angvfluor_df = pd.DataFrame(data=cols)
    
    # interc = ('intercept',[linreg_contvfluor.intercept])
    # pval = ('pvalue',[linreg_contvfluor.pvalue])
    # rval = ('rvalue',[linreg_contvfluor.rvalue])
    # slp = ('slope',[linreg_contvfluor.slope])
    # serr = ('stderr',[linreg_contvfluor.stderr])
    # cols=dict([interc, pval, rval, slp, serr])
    # linreg_contvfluor_df = pd.DataFrame(data=cols)
    
    # linreg_angvfluor_ls.append(linreg_angvfluor_df)
    # linreg_contvfluor_ls.append(linreg_contvfluor_df)
    
# linreg_angvfluor_df = pd.concat(linreg_angvfluor_ls)
# linreg_angvfluor_df_long = linreg_angvfluor_df.melt(var_name='result_name', value_name='result_value')
# linreg_contvfluor_df = pd.concat(linreg_contvfluor_ls)
# linreg_contvfluor_df_long = linreg_contvfluor_df.melt(var_name='result_name', value_name='result_value')


### We're centering these files to correct for any translational shifts that
### might be present in the data:
temp_cln_centered_df = pd.concat(temp_cln_centered)

sns.scatterplot(temp_cln_centered_df['Average Angle (deg)'], temp_cln_centered_df['lifeact_fluor_normed_mean'], marker='.', size=5, alpha=0.5)
plt.xlim(-100,100)
plt.ylim(-1,1)

linreg_angvfluor_centered = stats.linregress(temp_cln_centered_df['Average Angle (deg)'], temp_cln_centered_df['lifeact_fluor_normed_mean'])
linreg_contvfluor_centered = stats.linregress(temp_cln_centered_df['Contact Surface Length (N.D.)'], temp_cln_centered_df['lifeact_fluor_normed_mean'])


interc = ('intercept',[linreg_angvfluor_centered.intercept])
pval = ('pvalue',[linreg_angvfluor_centered.pvalue])
rval = ('rvalue',[linreg_angvfluor_centered.rvalue])
slp = ('slope',[linreg_angvfluor_centered.slope])
serr = ('stderr',[linreg_angvfluor_centered.stderr])
sf = ('Source_File',['all_files_centered'])
cols=dict([interc, pval, rval, slp, serr, sf])
linreg_angvfluor_centered_df = pd.DataFrame(data=cols)

linreg_angvfluor_df = pd.concat([linreg_angvfluor_df, linreg_angvfluor_centered_df])


interc = ('intercept',[linreg_contvfluor_centered.intercept])
pval = ('pvalue',[linreg_contvfluor_centered.pvalue])
rval = ('rvalue',[linreg_contvfluor_centered.rvalue])
slp = ('slope',[linreg_contvfluor_centered.slope])
serr = ('stderr',[linreg_contvfluor_centered.stderr])
sf = ('Source_File',['all_files_centered'])
cols=dict([interc, pval, rval, slp, serr, sf])
linreg_contvfluor_centered_df = pd.DataFrame(data=cols)

linreg_contvfluor_df = pd.concat([linreg_contvfluor_df, linreg_contvfluor_centered_df])


linreg_angvfluor_df.to_csv(Path.joinpath(out_dir, r'linreg_angvfluor.csv'))
linreg_contvfluor_df.to_csv(Path.joinpath(out_dir, r'linreg_contvfluor.csv'))




## a line: y = slope * x + yintercept
# linreg_angvfluor_x = np.linspace(0,180, 181)
# linreg_angvfluor_y = linreg_angvfluor.slope * linreg_angvfluor_x + linreg_angvfluor.intercept

# linreg_contvfluor_x = np.linspace(0,1.3, 131)
# linreg_contvfluor_y = linreg_contvfluor.slope * linreg_contvfluor_x + linreg_contvfluor.intercept


# plt.scatter(temp_ang, temp_fluor, marker='.')
# plt.plot(linreg_angvfluor_x, linreg_angvfluor_y)
# plt.xlim(0,200)
# plt.ylim(0,2.0)


# plt.scatter(temp_cont, temp_fluor, marker='.')
# plt.plot(linreg_contvfluor_x, linreg_contvfluor_y)
# plt.xlim(0,2)
# plt.ylim(0,2.0)



# 2D plots for all timelapses together for Bc/B:
fig, ax = plt.subplots(figsize=(width_1col, width_1col))
# ax.scatter(x, y, c=t, cmap=cmap, marker='.', s=3, alpha=0.5, zorder=10)
# ax.scatter(x, y, c='g', marker='.', s=3, alpha=0.5, zorder=10)
sns.scatterplot('Average Angle (deg)', 'Contact Surface Length (N.D.)', hue='Source_File',
                data=la_contnd_df[la_contnd_df['Experimental_Condition'] == 'ee1xMBS'], 
                palette='bright', s=5, linewidth=0, alpha=0.5, ax=ax, legend=False, zorder=10)
# ax.scatter(xdb, ydb, c='grey', marker='.', s=3, alpha=0.5, zorder=9)
ax.plot(theor_ang*2, theor_contnd, c='k', lw=1, ls='--', zorder=1)
ax.plot(r30sim_contnd_df['Average Angle (deg)'], 
        r30sim_contnd_df['Contact Surface Length (N.D.)'], 
        c='k', marker='.', markersize=3, lw=1, ls='--', zorder=2)
ax.plot(r30simLA_contnd_df['Average Angle (deg)'], 
        r30simLA_contnd_df['Contact Surface Length (N.D.)'], 
        c='k', marker='.', markersize=3, lw=1, ls='-', zorder=2)
ax.set_xlim(0,200)
ax.set_ylim(0,3)
ax.axhline(expect_max_contnd, c='grey', lw=1, zorder=0)
ax.axvline(180, c='grey', lw=1, zorder=0)
# ax.set_xlabel('Average Angle (deg)')
# ax.set_ylabel('Contact Diameter (N.D.)')
ax.set_xlabel('')
ax.set_ylabel('')
# plt.tight_layout()
fig.savefig(Path.joinpath(out_dir, r'angle_vs_diam_lagfp_mbs.tif'), dpi=dpi_graph, bbox_inches='tight')
plt.close(fig)

fig, ax = plt.subplots(figsize=(width_1col, width_1col))
# ax.scatter(x, y, c=t, cmap=cmap, marker='.', s=3, alpha=0.5, zorder=10)
# ax.scatter(x, y, c='g', marker='.', s=3, alpha=0.5, zorder=10)
# ax.scatter(xdb, ydb, c='grey', marker='.', s=3, alpha=0.5, zorder=9)
sns.scatterplot('Average Angle (deg)', 'Contact Surface Length (N.D.)', hue='Source_File',
                data=la_contnd_df[la_contnd_df['Experimental_Condition'] == 'ee1xDB'], 
                palette='bright', s=5, linewidth=0, alpha=0.5, ax=ax, legend=False, zorder=9)
ax.plot(theor_ang*2, theor_contnd, c='k', lw=1, ls='--', zorder=1)
ax.plot(r30sim_contnd_df['Average Angle (deg)'], 
        r30sim_contnd_df['Contact Surface Length (N.D.)'], 
        c='k', marker='.', markersize=3, lw=1, ls='--', zorder=2)
ax.plot(r30simLA_contnd_df['Average Angle (deg)'], 
        r30simLA_contnd_df['Contact Surface Length (N.D.)'], 
        c='k', marker='.', markersize=3, lw=1, ls='-', zorder=2)
ax.set_xlim(0,200)
ax.set_ylim(0,3)
ax.axhline(expect_max_contnd, c='grey', lw=1, zorder=0)
ax.axvline(180, c='grey', lw=1, zorder=0)
# ax.set_xlabel('Average Angle (deg)')
# ax.set_ylabel('Contact Diameter (N.D.)')
ax.set_xlabel('')
ax.set_ylabel('')
# plt.tight_layout()
fig.savefig(Path.joinpath(out_dir, r'angle_vs_diam_lagfp_db.tif'), dpi=dpi_graph, bbox_inches='tight')
plt.close(fig)





fig, ax = plt.subplots(figsize=(width_1col, width_1col))
# ax.scatter(y, z, c=t, cmap=cmap, marker='.', s=3, alpha=0.5, zorder=10)
# ax.scatter(y, z, c='g', marker='.', s=3, alpha=0.5, zorder=10)
sns.scatterplot('Contact Surface Length (N.D.)', 'lifeact_fluor_normed_mean', hue='Source_File',
                data=la_contnd_df[la_contnd_df['Experimental_Condition'] == 'ee1xMBS'], 
                palette='bright', s=5, linewidth=0, alpha=0.5, ax=ax, legend=False, zorder=10)
# ax.scatter(ydb, zdb, c='grey', marker='.', s=3, alpha=0.5, zorder=9)
ax.plot(contnd_predict*2, beta_star_div_beta_contnd_range, c='k', lw=1, ls='--', zorder=1)
# ax.plot(r30simLA_contnd_df['Contact Surface Length (N.D.)'], 
#         r30simLA_contnd_df['lifeact_fluor_normed_mean'], 
#         c='k', marker='.', markersize=3, lw=1, ls='--', zorder=2)
ax.set_xlim(0,3)
ax.axvline(expect_max_contnd, c='grey', lw=1, zorder=0)
ax.set_ylim(0,2)
# ax.set_xlabel('Contact Diameter (N.D.)')
# ax.set_ylabel(r'$\beta_c/\beta$')
ax.set_xlabel('')
ax.set_ylabel('')
# plt.tight_layout()
fig.savefig(Path.joinpath(out_dir, r'diam_vs_fluor_lagfp_mbs.tif'), dpi=dpi_graph, bbox_inches='tight')
plt.close(fig)

fig, ax = plt.subplots(figsize=(width_1col, width_1col))
# ax.scatter(y, z, c=t, cmap=cmap, marker='.', s=3, alpha=0.5, zorder=10)
# ax.scatter(y, z, c='g', marker='.', s=3, alpha=0.5, zorder=10)
# ax.scatter(ydb, zdb, c='grey', marker='.', s=3, alpha=0.5, zorder=9)
sns.scatterplot('Contact Surface Length (N.D.)', 'lifeact_fluor_normed_mean', hue='Source_File',
                data=la_contnd_df[la_contnd_df['Experimental_Condition'] == 'ee1xDB'], 
                palette='bright', s=5, linewidth=0, alpha=0.5, ax=ax, legend=False, zorder=10)
ax.plot(contnd_predict*2, beta_star_div_beta_contnd_range, c='k', lw=1, ls='--', zorder=1)
# ax.plot(r30simLA_contnd_df['Contact Surface Length (N.D.)'], 
#         r30simLA_contnd_df['lifeact_fluor_normed_mean'], 
#         c='k', marker='.', markersize=3, lw=1, ls='--', zorder=2)
ax.set_xlim(0,3)
ax.axvline(expect_max_contnd, c='grey', lw=1, zorder=0)
ax.set_ylim(0,2)
# ax.set_xlabel('Contact Diameter (N.D.)')
# ax.set_ylabel(r'$\beta_c/\beta$')
ax.set_xlabel('')
ax.set_ylabel('')
# plt.tight_layout()
fig.savefig(Path.joinpath(out_dir, r'diam_vs_fluor_lagfp_db.tif'), dpi=dpi_graph, bbox_inches='tight')
plt.close(fig)





fig, ax = plt.subplots(figsize=(width_1col, width_1col))
# ax.scatter(x, z, c=t, cmap=cmap, marker='.', s=3, alpha=0.5, zorder=10)
# ax.scatter(x, z, c='g', marker='.', s=3, alpha=0.5, zorder=10)
sns.scatterplot('Average Angle (deg)', 'lifeact_fluor_normed_mean', hue='Source_File',
                data=la_contnd_df[la_contnd_df['Experimental_Condition'] == 'ee1xMBS'], 
                palette='bright', s=5, linewidth=0, alpha=0.5, ax=ax, legend=False, zorder=10)
# ax.scatter(xdb, zdb, c='grey', marker='.', s=3, alpha=0.5, zorder=9)
ax.plot(angle_predict_deg*2, beta_star_div_beta_contnd_range, c='k', lw=1, ls='--', zorder=1)
# ax.plot(r30simLA_contnd_df['Average Angle (deg)'], 
#         r30simLA_contnd_df['lifeact_fluor_normed_mean'], 
#         c='k', marker='.', markersize=3, lw=1, ls='--', zorder=2)
ax.set_xlim(0,200)
ax.axvline(180, c='grey', lw=1, zorder=0)
ax.set_ylim(0,2)
# ax.set_xlabel('Average Angle (deg)')
# ax.set_ylabel(r'$\beta_c/\beta$')
ax.set_xlabel('')
ax.set_ylabel('')
# plt.tight_layout()
fig.savefig(Path.joinpath(out_dir, r'angle_vs_fluor_lagfp_mbs.tif'), dpi=dpi_graph, bbox_inches='tight')
plt.close(fig)

## Also save the data from angle_vs_fluor_lagfp_mbs.tif 
## because I want to re-use it in the simulation script 
## (sct5_viscous_shell_model) to compare against models
la_contnd_df.to_pickle(Path.joinpath(out_dir, 'la_contnd_df.pkl'))



fig, ax = plt.subplots(figsize=(width_1col, width_1col))
# ax.scatter(x, z, c=t, cmap=cmap, marker='.', s=3, alpha=0.5, zorder=10)
# ax.scatter(x, z, c='g', marker='.', s=3, alpha=0.5, zorder=10)
# ax.scatter(xdb, zdb, c='grey', marker='.', s=3, alpha=0.5, zorder=9)
sns.scatterplot('Average Angle (deg)', 'lifeact_fluor_normed_mean', hue='Source_File',
                data=la_contnd_df[la_contnd_df['Experimental_Condition'] == 'ee1xDB'], 
                palette='bright', s=5, linewidth=0, alpha=0.5, ax=ax, legend=False, zorder=10)
ax.plot(angle_predict_deg*2, beta_star_div_beta_contnd_range, c='k', lw=1, ls='--', zorder=1)
# ax.plot(r30sim_contnd_df['Average Angle (deg)'], 
#         r30sim_contnd_df['lifeact_fluor_normed_mean'], 
#         c='k', marker='.', markersize=3, lw=1, ls='--', zorder=2)
ax.set_xlim(0,200)
ax.axvline(180, c='grey', lw=1, zorder=0)
ax.set_ylim(0,2)
# ax.set_xlabel('Average Angle (deg)')
# ax.set_ylabel(r'$\beta_c/\beta$')
ax.set_xlabel('')
ax.set_ylabel('')
# plt.tight_layout()
fig.savefig(Path.joinpath(out_dir, r'angle_vs_fluor_lagfp_db.tif'), dpi=dpi_graph, bbox_inches='tight')
plt.close(fig)





# 2D plots for all timelapses together for mbRFP equivalent of Bc/B:
fig, ax = plt.subplots(figsize=(width_1col, width_1col))
# ax.scatter(x, y, c=t, cmap=cmap, marker='.', s=3, alpha=0.5, zorder=10)
# ax.scatter(x, y, c='r', marker='.', s=3, alpha=0.5, zorder=10)
sns.scatterplot('Average Angle (deg)', 'Contact Surface Length (N.D.)', hue='Source_File',
                data=la_contnd_df[la_contnd_df['Experimental_Condition'] == 'ee1xMBS'], 
                palette='bright', s=5, linewidth=0, alpha=0.5, ax=ax, legend=False, zorder=10)
# ax.scatter(xdb, ydb, c='grey', marker='.', s=3, alpha=0.5, zorder=9)
ax.plot(angle_predict_deg*2, contnd_predict*2, c='k', lw=1, ls='--', zorder=1)
ax.plot(r30sim_contnd_df['Average Angle (deg)'], 
        r30sim_contnd_df['Contact Surface Length (N.D.)'], 
        c='k', marker='.', markersize=3, lw=1, ls='--', zorder=2)
ax.plot(r30simLA_contnd_df['Average Angle (deg)'], 
        r30simLA_contnd_df['Contact Surface Length (N.D.)'], 
        c='k', marker='.', markersize=3, lw=1, ls='-', zorder=2)
ax.set_xlim(0,200)
ax.set_ylim(0,3)
ax.axvline(180, c='grey', lw=1, zorder=0)
ax.axhline(expect_max_contnd, c='grey', lw=1, zorder=0)
# ax.set_xlabel('Average Angle (deg)')
# ax.set_ylabel('Contact Diameter (N.D.)')
ax.set_xlabel('')
ax.set_ylabel('')
# plt.tight_layout()
fig.savefig(Path.joinpath(out_dir, r'angle_vs_diam_mbrfp_mbs.tif'), dpi=dpi_graph, bbox_inches='tight')
plt.close(fig)

# 2D plots for all timelapses together for mbRFP equivalent of Bc/B:
fig, ax = plt.subplots(figsize=(width_1col, width_1col))
# ax.scatter(x, y, c=t, cmap=cmap, marker='.', s=3, alpha=0.5, zorder=10)
# ax.scatter(x, y, c='r', marker='.', s=3, alpha=0.5, zorder=10)
# ax.scatter(xdb, ydb, c='grey', marker='.', s=3, alpha=0.5, zorder=9)
sns.scatterplot('Average Angle (deg)', 'Contact Surface Length (N.D.)', hue='Source_File',
                data=la_contnd_df[la_contnd_df['Experimental_Condition'] == 'ee1xDB'], 
                palette='bright', s=5, linewidth=0, alpha=0.5, ax=ax, legend=False, zorder=10)
ax.plot(angle_predict_deg*2, contnd_predict*2, c='k', lw=1, ls='--', zorder=1)
ax.plot(r30sim_contnd_df['Average Angle (deg)'], 
        r30sim_contnd_df['Contact Surface Length (N.D.)'], 
        c='k', marker='.', markersize=3, lw=1, ls='--', zorder=2)
ax.plot(r30simLA_contnd_df['Average Angle (deg)'], 
        r30simLA_contnd_df['Contact Surface Length (N.D.)'], 
        c='k', marker='.', markersize=3, lw=1, ls='-', zorder=2)
ax.set_xlim(0,200)
ax.set_ylim(0,3)
ax.axvline(180, c='grey', lw=1, zorder=0)
ax.axhline(expect_max_contnd, c='grey', lw=1, zorder=0)
# ax.set_xlabel('Average Angle (deg)')
# ax.set_ylabel('Contact Diameter (N.D.)')
ax.set_xlabel('')
ax.set_ylabel('')
# plt.tight_layout()
fig.savefig(Path.joinpath(out_dir, r'angle_vs_diam_mbrfp_db.tif'), dpi=dpi_graph, bbox_inches='tight')
plt.close(fig)





fig, ax = plt.subplots(figsize=(width_1col, width_1col))
# ax.scatter(y, z, c=t, cmap=cmap, marker='.', s=3, alpha=0.5, zorder=10)
# ax.scatter(y, w, c='r', marker='.', s=3, alpha=0.5, zorder=10)
sns.scatterplot('Contact Surface Length (N.D.)', 'mbrfp_fluor_normed_mean', hue='Source_File',
                data=la_contnd_df[la_contnd_df['Experimental_Condition'] == 'ee1xMBS'], 
                palette='bright', s=5, linewidth=0, alpha=0.5, ax=ax, legend=False, zorder=10)
# ax.scatter(ydb, wdb, c='grey', marker='.', s=3, alpha=0.5, zorder=9)
ax.plot(contnd_predict*2, beta_star_div_beta_contnd_range, c='k', lw=1, ls='--', zorder=1)
# ax.plot(r30sim_contnd_df['Contact Surface Length (N.D.)'], 
#         r30sim_contnd_df['lifeact_fluor_normed_mean'], 
#         c='k', marker='.', markersize=3, lw=1, ls='--', zorder=2)
ax.set_xlim(0,3)
ax.set_ylim(0,2)
ax.axvline(expect_max_contnd, c='grey', lw=1, zorder=0)
# ax.set_xlabel('Contact Diameter (N.D.)')
# ax.set_ylabel(r'$\beta_c/\beta$')
ax.set_xlabel('')
ax.set_ylabel('')
# plt.tight_layout()
fig.savefig(Path.joinpath(out_dir, r'diam_vs_fluor_mbrfp_mbs.tif'), dpi=dpi_graph, bbox_inches='tight')
plt.close(fig)

fig, ax = plt.subplots(figsize=(width_1col, width_1col))
# ax.scatter(y, z, c=t, cmap=cmap, marker='.', s=3, alpha=0.5, zorder=10)
# ax.scatter(y, w, c='r', marker='.', s=3, alpha=0.5, zorder=10)
# ax.scatter(ydb, wdb, c='grey', marker='.', s=3, alpha=0.5, zorder=9)
sns.scatterplot('Contact Surface Length (N.D.)', 'mbrfp_fluor_normed_mean', hue='Source_File',
                data=la_contnd_df[la_contnd_df['Experimental_Condition'] == 'ee1xDB'], 
                palette='bright', s=5, linewidth=0, alpha=0.5, ax=ax, legend=False, zorder=10)
ax.plot(contnd_predict*2, beta_star_div_beta_contnd_range, c='k', lw=1, ls='--', zorder=1)
# ax.plot(r30sim_contnd_df['Contact Surface Length (N.D.)'], 
#         r30sim_contnd_df['lifeact_fluor_normed_mean'], 
#         c='k', marker='.', markersize=3, lw=1, ls='--', zorder=2)
ax.set_xlim(0,3)
ax.set_ylim(0,2)
ax.axvline(expect_max_contnd, c='grey', lw=1, zorder=0)
# ax.set_xlabel('Contact Diameter (N.D.)')
# ax.set_ylabel(r'$\beta_c/\beta$')
ax.set_xlabel('')
ax.set_ylabel('')
# plt.tight_layout()
fig.savefig(Path.joinpath(out_dir, r'diam_vs_fluor_mbrfp_db.tif'), dpi=dpi_graph, bbox_inches='tight')
plt.close(fig)




fig, ax = plt.subplots(figsize=(width_1col, width_1col))
# ax.scatter(x, z, c=t, cmap=cmap, marker='.', s=3, alpha=0.5, zorder=10)
# ax.scatter(x, w, c='r', marker='.', s=3, alpha=0.5, zorder=10)
sns.scatterplot('Average Angle (deg)', 'mbrfp_fluor_normed_mean', hue='Source_File',
                data=la_contnd_df[la_contnd_df['Experimental_Condition'] == 'ee1xMBS'], 
                palette='bright', s=5, linewidth=0, alpha=0.5, ax=ax, legend=False, zorder=10)
# ax.scatter(xdb, wdb, c='grey', marker='.', s=3, alpha=0.5, zorder=9)
ax.plot(angle_predict_deg*2, beta_star_div_beta_contnd_range, c='k', lw=1, ls='--', zorder=1)
# ax.plot(r30sim_contnd_df['Average Angle (deg)'], 
#         r30sim_contnd_df['lifeact_fluor_normed_mean'], 
#         c='k', marker='.', markersize=3, lw=1, ls='--', zorder=2)
ax.set_xlim(0,200)
ax.set_ylim(0,2)
ax.axvline(180, c='grey', lw=1, zorder=0)
# ax.set_xlabel('Average Angle (deg)')
# ax.set_ylabel(r'$\beta_c/\beta$')
ax.set_xlabel('')
ax.set_ylabel('')
# plt.tight_layout()
fig.savefig(Path.joinpath(out_dir, r'angle_vs_fluor_mbrfp_mbs.tif'), dpi=dpi_graph, bbox_inches='tight')
plt.close(fig)

fig, ax = plt.subplots(figsize=(width_1col, width_1col))
# ax.scatter(x, z, c=t, cmap=cmap, marker='.', s=3, alpha=0.5, zorder=10)
# ax.scatter(x, w, c='r', marker='.', s=3, alpha=0.5, zorder=10)
# ax.scatter(xdb, wdb, c='grey', marker='.', s=3, alpha=0.5, zorder=9)
sns.scatterplot('Average Angle (deg)', 'mbrfp_fluor_normed_mean', hue='Source_File',
                data=la_contnd_df[la_contnd_df['Experimental_Condition'] == 'ee1xDB'], 
                palette='bright', s=5, linewidth=0, alpha=0.5, ax=ax, legend=False, zorder=10)
ax.plot(angle_predict_deg*2, beta_star_div_beta_contnd_range, c='k', lw=1, ls='--', zorder=1)
# ax.plot(r30sim_contnd_df['Average Angle (deg)'], 
#         r30sim_contnd_df['lifeact_fluor_normed_mean'], 
#         c='k', marker='.', markersize=3, lw=1, ls='--', zorder=2)
ax.set_xlim(0,200)
ax.set_ylim(0,2)
ax.axvline(180, c='grey', lw=1, zorder=0)
# ax.set_xlabel('Average Angle (deg)')
# ax.set_ylabel(r'$\beta_c/\beta$')
ax.set_xlabel('')
ax.set_ylabel('')
# plt.tight_layout()
fig.savefig(Path.joinpath(out_dir, r'angle_vs_fluor_mbrfp_db.tif'), dpi=dpi_graph, bbox_inches='tight')
plt.close(fig)



# Do 3D plots of Bc/B, measured angle, contact diameter vs each other:
# %matplotlib qt

# 1xMBS scatterplot:
# fig = plt.figure(figsize = (12,12))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x, y, z, c=t, cmap=cmap)
# ax.plot(angle_predict_deg*2, contnd_predict*2, beta_star_div_beta_contnd_range)
# ax.set_xlabel('Average Angle (deg)')
# ax.set_ylabel('Contact Surface Length (N.D.)')
# ax.set_zlabel('lifeact_fluor_normed')
# ax.set_xlim(0, 200)
# ax.set_ylim(0, 3)
# ax.set_zlim(0, 2)
# plt.tight_layout()
#plt.close()

# 1xDB scatterplot:
# fig = plt.figure(figsize = (12,12))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(xdb, ydb, zdb, c=tdb, cmap=cmap)
# ax.plot(angle_predict_deg*2, contnd_predict*2, beta_star_div_beta_contnd_range)
# ax.set_xlabel('Average Angle (deg)')
# ax.set_ylabel('Contact Surface Length (N.D.)')
# ax.set_zlabel('lifeact_fluor_normed')
# ax.set_xlim(0, 200)
# ax.set_ylim(0, 3)
# ax.set_zlim(0, 2)
# plt.tight_layout()
#plt.close()

# Plotting them together:
# fig = plt.figure(figsize = (12,12))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x, y, z, c='g')
# ax.scatter(xdb, ydb, zdb, c='grey')
# ax.plot(angle_predict_deg*2, contnd_predict*2, beta_star_div_beta_contnd_range)
# ax.set_xlabel('Average Angle (deg)')
# ax.set_ylabel('Contact Surface Length (N.D.)')
# ax.set_zlabel('lifeact_fluor_normed')
# ax.set_xlim(0, 200)
# ax.set_ylim(0, 3)
# ax.set_zlim(0, 2)
# plt.tight_layout()
# plt.close()





# Since we will be creating a multi-page pdf, make sure that no figures are 
# open before plotting the graphs we want to add to the pdf:
plt.close('all')

# Plot the non-dimensionalized contact length, contact angle, and Bc/B over 
# time for each timelapse separately and save into a single PDF:
for i,(sourcefile,df) in enumerate(la_contnd_df.groupby('Source_File')):
    fig, (ax1, ax2) = plt.subplots(figsize=(6,6), 
                                   num='#%d - '%i + sourcefile, 
                                   nrows=2, sharex=True)
    plt.suptitle('File #%d - '%i + sourcefile, verticalalignment='center', x=0.5, y=0.5, fontsize=14)
    cont = ax1.plot(df['Time (minutes)'], df['Contact Surface Length (N.D.)'], c='k', ls=':', marker='.', label='Contact Surface Length (N.D.)')
    ax1.set_ylim(0, 3)#la_contnd_df['Contact Surface Length (N.D.)'].max())
    
    ax1b = ax1.twinx()
    ang = ax1b.plot(df['Time (minutes)'], df['Average Angle (deg)'], c='k', ls='-', marker='^', label='Average Angle (deg)')
    ax1b.set_ylim(0, 200)
    ax1.set_ylabel('Contact Diameter (N.D.)')
    ax1b.set_ylabel('Average Angle (deg)', rotation=270, va='bottom', ha='center')
    leg_lines = cont + ang
    leg_labels = [lab.get_label() for lab in leg_lines]
    plt.legend(leg_lines, leg_labels, bbox_to_anchor=(0, 1.0, 1, .1), 
               loc='lower left', mode='expand', ncol=2)
    
    ### This version of the graphs uses "stick-style" error bars:
    # ax2.plot(df['Time (minutes)'], lifeact_rel_fluor[i], c='g', marker='.')
    # ax2.plot(df['Time (minutes)'], mbRFP_rel_fluor[i], c='r', marker='.')
    # ax2.errorbar(df['Time (minutes)'], df['lifeact_fluor_normed_mean'], 
    #              yerr=df['lifeact_fluor_normed_sem'], c='g', marker='.', 
    #              label='LifeAct', alpha=0.5, zorder=9)
    # ax2.errorbar(df['Time (minutes)'], df['mbrfp_fluor_normed_mean'], 
    #              yerr=df['mbrfp_fluor_normed_sem'], c='r', marker='.', 
    #              label='mbRFP', alpha=0.5, zorder=8)
    ###
    
    
    
    ### This version of the graphs uses "band-filled" errors:
    mean_la = df['lifeact_fluor_normed_mean']
    sem_la = df['lifeact_fluor_normed_sem']
    mean_cytopl_la = df['lifeact_cytoplasmic_fluor_normed_mean']
    sem_cytopl_la = df['lifeact_cytoplasmic_fluor_normed_sem']
    ax2.plot(df['Time (minutes)'], mean_la, c='g', marker='.', zorder=9, label='LifeAct')
    ax2.fill_between(df['Time (minutes)'], mean_la - sem_la, mean_la + sem_la, 
                     color='g', alpha=0.5, zorder=6)
    ax2.plot(df['Time (minutes)'], mean_cytopl_la, c='orange')
    ax2.fill_between(df['Time (minutes)'], mean_cytopl_la - sem_cytopl_la, mean_cytopl_la + sem_cytopl_la, 
                     color='orange', alpha=0.5, zorder=3)
    
    mean_mb = df['mbrfp_fluor_normed_mean']
    sem_mb = df['mbrfp_fluor_normed_sem']
    mean_cytopl_mb = df['mbrfp_cytoplasmic_fluor_normed_mean']
    sem_cytopl_mb = df['mbrfp_cytoplasmic_fluor_normed_sem']
    ax2.plot(df['Time (minutes)'], mean_mb, c='r', marker='.', zorder=8, label='mbRFP')
    ax2.fill_between(df['Time (minutes)'], mean_mb - sem_mb, mean_mb + sem_mb, 
                     color='r', alpha=0.5, zorder=5)
    ax2.plot(df['Time (minutes)'], mean_cytopl_mb, c='magenta')
    ax2.fill_between(df['Time (minutes)'], mean_cytopl_mb - sem_cytopl_mb, mean_cytopl_mb + sem_cytopl_mb, 
                     color='magenta', alpha=0.5, zorder=2)
    
    mean_dx = df['af647dx_fluor_normed_mean']
    sem_dx = df['af647dx_fluor_normed_sem']
    mean_cytopl_dx = df['af647dx_cytoplasmic_fluor_normed_mean']
    sem_cytopl_dx = df['af647dx_cytoplasmic_fluor_normed_sem']
    ax2.plot(df['Time (minutes)'], mean_dx, c='b', marker='.', zorder=7, label='Dextran')
    ax2.fill_between(df['Time (minutes)'], mean_dx - sem_dx, mean_dx + sem_dx, 
                     color='b', alpha=0.5, zorder=4)
    ax2.plot(df['Time (minutes)'], mean_cytopl_dx, c='cyan')
    ax2.fill_between(df['Time (minutes)'], mean_cytopl_dx - sem_cytopl_dx, mean_cytopl_dx + sem_cytopl_dx, 
                     color='cyan', alpha=0.5, zorder=1)
    ###

    ax2.axhline(1, c='k', ls='--', zorder=10)
    ax2.set_ylim(0,2)
    xmin, xmax = -1, round_up_to_multiple(la_contnd_df['Time (minutes)'].max(), 5)
    ax2.set_xlim(xmin, xmax)
    ax2.set_ylabel('Norm. Contact Fluorescence (N.D.)')
    ax2.set_xlabel ('Time (minutes)')
    ax2.legend(loc='upper right')
    # ax2.plot(df['Time (minutes)'], df[], c='r', marker='.')
    
    # ax2b = ax2.twinx()
    # ax2b.plot(df['Time (minutes)'], lifeact_rel_fluor_norm[i], c='k', marker='.')
    # ax2b.spines['left'].set_color('red')
    # ax2.tick_params(axis='y', colors='green')
    
    
    #plt.show()
multipdf(Path.joinpath(out_dir, r"ee_lifeact_contnd_angles.pdf"))
plt.close('all')



### Pick a couple of graphs of interest to save as separate images:
## Also make the graphs simpler so that they are easier to interpret:
graphs_of_interest = ['2019_09_13-series002_crop1_z3.csv',
                      '2019_09_17-series001_crop1_z2.csv',
                      '2019_09_17-series001_crop2_z1.csv',
                      '2019_09_17-series001_crop4_z2.csv',
                      '2019_09_20-series001_crop3_z2.csv',
                      '2020_01_17-series003_crop4_z1.csv']
# graphs_of_interest = [2,3,5,10,20]
for sourcefile,df in la_contnd_df.groupby('Source_File'):
    if sourcefile in graphs_of_interest:
        fig, (ax1, ax2) = plt.subplots(figsize=(width_1col, width_1col), 
                                       # num='#%d - '%i + sourcefile, 
                                       nrows=2, sharex=True)
        # plt.suptitle('File #%d - '%i + sourcefile, verticalalignment='center', x=0.5, y=0.5, fontsize=14)
        cont = ax1.plot(df['Time (minutes)'], df['Contact Surface Length (N.D.)'], c='k', ls=':', marker='.', alpha=1, label='Diam. (N.D.)')
        ax1.set_ylim(0, 3)#la_contnd_df['Contact Surface Length (N.D.)'].max())
        
        ax1b = ax1.twinx()
        ang = ax1b.plot(df['Time (minutes)'], df['Average Angle (deg)'], c='grey', ls='-', marker='.', alpha=0.5, label='Angle (deg)')
        ax1b.set_ylim(0, 200)
        ax1.set_ylabel('Cont. Diam. (N.D.)')
        ax1b.set_ylabel('Cont. Angle (deg)', rotation=270, va='bottom', ha='center')
        leg_lines = cont + ang
        leg_labels = [lab.get_label() for lab in leg_lines]
        # plt.legend(leg_lines, leg_labels, bbox_to_anchor=(0, 1.0, 1, .1), 
        #            loc='lower left', mode='expand', ncol=2)
        ax1.legend(leg_lines, leg_labels)
        
        ### This version of the graphs uses "stick-style" error bars:
        # ax2.plot(df['Time (minutes)'], lifeact_rel_fluor[i], c='g', marker='.')
        # ax2.plot(df['Time (minutes)'], mbRFP_rel_fluor[i], c='r', marker='.')
        # ax2.errorbar(df['Time (minutes)'], df['lifeact_fluor_normed_mean'], 
        #              yerr=df['lifeact_fluor_normed_sem'], c='g', marker='.', 
        #              label='LifeAct', alpha=0.5, zorder=9)
        # ax2.errorbar(df['Time (minutes)'], df['mbrfp_fluor_normed_mean'], 
        #              yerr=df['mbrfp_fluor_normed_sem'], c='r', marker='.', 
        #              label='mbRFP', alpha=0.5, zorder=8)
        ###
        
        
        
        ### This version of the graphs uses "band-filled" errors:
        mean_la = df['lifeact_fluor_normed_mean']
        sem_la = df['lifeact_fluor_normed_sem']
        # mean_cytopl_la = df['lifeact_cytoplasmic_fluor_normed_mean']
        # sem_cytopl_la = df['lifeact_cytoplasmic_fluor_normed_sem']
        ax2.plot(df['Time (minutes)'], mean_la, c='g', marker='.', alpha=0.5, zorder=9, label='LifeAct')
        ax2.fill_between(df['Time (minutes)'], mean_la - sem_la, mean_la + sem_la, 
                         color='g', alpha=0.25, zorder=6)
        # ax2.plot(df['Time (minutes)'], mean_cytopl_la, c='orange')
        # ax2.fill_between(df['Time (minutes)'], mean_cytopl_la - sem_cytopl_la, mean_cytopl_la + sem_cytopl_la, 
        #                  color='orange', alpha=0.5, zorder=3)
        
        mean_mb = df['mbrfp_fluor_normed_mean']
        sem_mb = df['mbrfp_fluor_normed_sem']
        # mean_cytopl_mb = df['mbrfp_cytoplasmic_fluor_normed_mean']
        # sem_cytopl_mb = df['mbrfp_cytoplasmic_fluor_normed_sem']
        ax2.plot(df['Time (minutes)'], mean_mb, c='r', marker='.', alpha=0.5, zorder=8, label='mbRFP')
        ax2.fill_between(df['Time (minutes)'], mean_mb - sem_mb, mean_mb + sem_mb, 
                         color='r', alpha=0.25, zorder=5)
        # ax2.plot(df['Time (minutes)'], mean_cytopl_mb, c='magenta')
        # ax2.fill_between(df['Time (minutes)'], mean_cytopl_mb - sem_cytopl_mb, mean_cytopl_mb + sem_cytopl_mb, 
        #                  color='magenta', alpha=0.5, zorder=2)
        
        # mean_dx = df['af647dx_fluor_normed_mean']
        # sem_dx = df['af647dx_fluor_normed_sem']
        # mean_cytopl_dx = df['af647dx_cytoplasmic_fluor_normed_mean']
        # sem_cytopl_dx = df['af647dx_cytoplasmic_fluor_normed_sem']
        # ax2.plot(df['Time (minutes)'], mean_dx, c='b', marker='.', zorder=7, label='Dextran')
        # ax2.fill_between(df['Time (minutes)'], mean_dx - sem_dx, mean_dx + sem_dx, 
        #                  color='b', alpha=0.5, zorder=4)
        # ax2.plot(df['Time (minutes)'], mean_cytopl_dx, c='cyan')
        # ax2.fill_between(df['Time (minutes)'], mean_cytopl_dx - sem_cytopl_dx, mean_cytopl_dx + sem_cytopl_dx, 
        #                  color='cyan', alpha=0.5, zorder=1)
        ###
    
        ax2.axhline(1, c='k', ls='--', zorder=10)
        ax2.set_ylim(0,2)
        # xmin, xmax = -1, round_up_to_multiple(la_contnd_df['Time (minutes)'].max(), 5)
        # ax2.set_xlim(xmin, xmax)
        ax2.set_ylabel('Norm. Cont. Fluor.')
        ax2.set_xlabel ('Time (minutes)')
        ax2.legend(ncol=2, loc='upper right')
        # ax2.plot(df['Time (minutes)'], df[], c='r', marker='.')
        
        # ax2b = ax2.twinx()
        # ax2b.plot(df['Time (minutes)'], lifeact_rel_fluor_norm[i], c='k', marker='.')
        # ax2b.spines['left'].set_color('red')
        # ax2.tick_params(axis='y', colors='green')
        
        
        #plt.show()
        # fig.savefig(Path.joinpath(out_dir, r"ee_lifeact_contnd_angles_%d.tif"%i), dpi=dpi_graph, bbox_inches='tight')
        fig.savefig(Path.joinpath(out_dir, r"%s_kinetics.tif"%sourcefile), dpi=dpi_graph, bbox_inches='tight')
        plt.close('all')








# Return the dimensionalized values for the contact diameters of the 1xDB
# ecto-ecto cells out of curiousity:
dim_cont_diam_1xDB = lifeact_long[lifeact_long['Experimental_Condition'] == 'ee1xDB']['Contact Surface Length (px)'] * um_per_pixel
# and plot them in a histogram:
# sns.distplot(dim_cont_diam_1xDB[dim_cont_diam_1xDB != 0])
ee1xDBstats = dim_cont_diam_1xDB[dim_cont_diam_1xDB != 0].describe()
ee1xDBstats['median'] = dim_cont_diam_1xDB[dim_cont_diam_1xDB != 0].median()







# Now we're going to plot kymographs of ordered contactsurface linescans for 
# the red and the green channels:

redchan_fluor_at_contact_inner = list()
redchan_fluor_at_contact_outer = list()
grnchan_fluor_at_contact_inner = list()
grnchan_fluor_at_contact_outer = list()

redchan_fluor_outside_contact_inner = list()
redchan_fluor_outside_contact_outer = list()
grnchan_fluor_outside_contact_inner = list()
grnchan_fluor_outside_contact_outer = list()


for z in lifeact:
    redchan_fluor_at_contact_inner.append([np.asarray([y.strip() for y in x[1:-1].split(',') if y], dtype=float) for x in z['redchan_fluor_at_contact_inner']])
    redchan_fluor_at_contact_outer.append([np.asarray([y.strip() for y in x[1:-1].split(',') if y], dtype=float) for x in z['redchan_fluor_at_contact_outer']])
    grnchan_fluor_at_contact_inner.append([np.asarray([y.strip() for y in x[1:-1].split(',') if y], dtype=float) for x in z['grnchan_fluor_at_contact_inner']])
    grnchan_fluor_at_contact_outer.append([np.asarray([y.strip() for y in x[1:-1].split(',') if y], dtype=float) for x in z['grnchan_fluor_at_contact_outer']])

    redchan_fluor_outside_contact_inner.append([np.asarray([y.strip() for y in x[1:-1].split(',') if y], dtype=float) for x in z['redchan_fluor_outside_contact_inner']])
    redchan_fluor_outside_contact_outer.append([np.asarray([y.strip() for y in x[1:-1].split(',') if y], dtype=float) for x in z['redchan_fluor_outside_contact_outer']])
    grnchan_fluor_outside_contact_inner.append([np.asarray([y.strip() for y in x[1:-1].split(',') if y], dtype=float) for x in z['grnchan_fluor_outside_contact_inner']])
    grnchan_fluor_outside_contact_outer.append([np.asarray([y.strip() for y in x[1:-1].split(',') if y], dtype=float) for x in z['grnchan_fluor_outside_contact_outer']])


longest_len_red_fluor_at_cont_inner = [max([len(y) for y in x]) for x in redchan_fluor_at_contact_inner]
longest_len_red_fluor_at_cont_outer = [max([len(y) for y in x]) for x in redchan_fluor_at_contact_outer]
longest_len_grn_fluor_at_cont_inner = [max([len(y) for y in x]) for x in grnchan_fluor_at_contact_inner]
longest_len_grn_fluor_at_cont_outer = [max([len(y) for y in x]) for x in grnchan_fluor_at_contact_outer]

longest_len_red_fluor_outside_cont_inner = [max([len(y) for y in x]) for x in redchan_fluor_outside_contact_inner]
longest_len_red_fluor_outside_cont_outer = [max([len(y) for y in x]) for x in redchan_fluor_outside_contact_outer]
longest_len_grn_fluor_outside_cont_inner = [max([len(y) for y in x]) for x in grnchan_fluor_outside_contact_inner]
longest_len_grn_fluor_outside_cont_outer = [max([len(y) for y in x]) for x in grnchan_fluor_outside_contact_outer]

# Above line should give you the contact lengths, the longest of which will be one of the dimensions in the kymograph,
# the other one should be given by:
n_timepoints_red_fluor_at_cont_inner = [len(x) for x in redchan_fluor_at_contact_inner]
n_timepoints_red_fluor_at_cont_outer = [len(x) for x in redchan_fluor_at_contact_outer]
n_timepoints_grn_fluor_at_cont_inner = [len(x) for x in grnchan_fluor_at_contact_inner]
n_timepoints_grn_fluor_at_cont_outer = [len(x) for x in grnchan_fluor_at_contact_outer]

n_timepoints_red_fluor_outside_cont_inner = [len(x) for x in redchan_fluor_outside_contact_inner]
n_timepoints_red_fluor_outside_cont_outer = [len(x) for x in redchan_fluor_outside_contact_outer]
n_timepoints_grn_fluor_outside_cont_inner = [len(x) for x in grnchan_fluor_outside_contact_inner]
n_timepoints_grn_fluor_outside_cont_outer = [len(x) for x in grnchan_fluor_outside_contact_outer]

# which is giving the number of timepoints in the kymograph.
# All of the shorter contact lengths need to be padded such that the actual data
# sits in the middle of the kymograph.

pad_lengths = list()
for i in range(len(longest_len_red_fluor_at_cont_inner)):
    pad_lengths.append([longest_len_red_fluor_at_cont_inner[i] - len(x) for x in redchan_fluor_at_contact_inner[i]])
redchan_fluor_at_contact_inner_padded = list()
for i in range(len(redchan_fluor_at_contact_inner)):
    padded = list()
    for j in range(len(redchan_fluor_at_contact_inner[i])):
        padded.append(np.pad(redchan_fluor_at_contact_inner[i][j], (int(pad_lengths[i][j]/2), int(pad_lengths[i][j]/2) + pad_lengths[i][j]%2), 'constant', constant_values=np.float('nan')))
    redchan_fluor_at_contact_inner_padded.append(padded)
    

pad_lengths = list()
for i in range(len(longest_len_red_fluor_at_cont_outer)):
    pad_lengths.append([longest_len_red_fluor_at_cont_outer[i] - len(x) for x in redchan_fluor_at_contact_outer[i]])
redchan_fluor_at_contact_outer_padded = list()
for i in range(len(redchan_fluor_at_contact_outer)):
    padded = list()
    for j in range(len(redchan_fluor_at_contact_outer[i])):
        padded.append(np.pad(redchan_fluor_at_contact_outer[i][j], (int(pad_lengths[i][j]/2), int(pad_lengths[i][j]/2) + pad_lengths[i][j]%2), 'constant', constant_values=np.float('nan')))
    redchan_fluor_at_contact_outer_padded.append(padded)


pad_lengths = list()
for i in range(len(longest_len_grn_fluor_at_cont_inner)):
    pad_lengths.append([longest_len_grn_fluor_at_cont_inner[i] - len(x) for x in grnchan_fluor_at_contact_inner[i]])
grnchan_fluor_at_contact_inner_padded = list()
for i in range(len(grnchan_fluor_at_contact_inner)):
    padded = list()
    for j in range(len(grnchan_fluor_at_contact_inner[i])):
        padded.append(np.pad(grnchan_fluor_at_contact_inner[i][j], (int(pad_lengths[i][j]/2), int(pad_lengths[i][j]/2) + pad_lengths[i][j]%2), 'constant', constant_values=np.float('nan')))
    grnchan_fluor_at_contact_inner_padded.append(padded)


pad_lengths = list()
for i in range(len(longest_len_grn_fluor_at_cont_outer)):
    pad_lengths.append([longest_len_grn_fluor_at_cont_outer[i] - len(x) for x in grnchan_fluor_at_contact_outer[i]])
grnchan_fluor_at_contact_outer_padded = list()
for i in range(len(grnchan_fluor_at_contact_outer)):
    padded = list()
    for j in range(len(grnchan_fluor_at_contact_outer[i])):
        padded.append(np.pad(grnchan_fluor_at_contact_outer[i][j], (int(pad_lengths[i][j]/2), int(pad_lengths[i][j]/2) + pad_lengths[i][j]%2), 'constant', constant_values=np.float('nan')))
    grnchan_fluor_at_contact_outer_padded.append(padded)

#-----------------------------------------------------------------------------#

pad_lengths = list()
for i in range(len(longest_len_red_fluor_outside_cont_inner)):
    pad_lengths.append([longest_len_red_fluor_outside_cont_inner[i] - len(x) for x in redchan_fluor_outside_contact_inner[i]])
redchan_fluor_outside_contact_inner_padded = list()
for i in range(len(redchan_fluor_outside_contact_inner)):
    padded = list()
    for j in range(len(redchan_fluor_outside_contact_inner[i])):
        padded.append(np.pad(redchan_fluor_outside_contact_inner[i][j], (int(pad_lengths[i][j]/2), int(pad_lengths[i][j]/2) + pad_lengths[i][j]%2), 'constant', constant_values=np.float('nan')))
    redchan_fluor_outside_contact_inner_padded.append(padded)
    

pad_lengths = list()
for i in range(len(longest_len_red_fluor_outside_cont_outer)):
    pad_lengths.append([longest_len_red_fluor_outside_cont_outer[i] - len(x) for x in redchan_fluor_outside_contact_outer[i]])
redchan_fluor_outside_contact_outer_padded = list()
for i in range(len(redchan_fluor_outside_contact_outer)):
    padded = list()
    for j in range(len(redchan_fluor_outside_contact_outer[i])):
        padded.append(np.pad(redchan_fluor_outside_contact_outer[i][j], (int(pad_lengths[i][j]/2), int(pad_lengths[i][j]/2) + pad_lengths[i][j]%2), 'constant', constant_values=np.float('nan')))
    redchan_fluor_outside_contact_outer_padded.append(padded)


pad_lengths = list()
for i in range(len(longest_len_grn_fluor_outside_cont_inner)):
    pad_lengths.append([longest_len_grn_fluor_outside_cont_inner[i] - len(x) for x in grnchan_fluor_outside_contact_inner[i]])
grnchan_fluor_outside_contact_inner_padded = list()
for i in range(len(grnchan_fluor_outside_contact_inner)):
    padded = list()
    for j in range(len(grnchan_fluor_outside_contact_inner[i])):
        padded.append(np.pad(grnchan_fluor_outside_contact_inner[i][j], (int(pad_lengths[i][j]/2), int(pad_lengths[i][j]/2) + pad_lengths[i][j]%2), 'constant', constant_values=np.float('nan')))
    grnchan_fluor_outside_contact_inner_padded.append(padded)


pad_lengths = list()
for i in range(len(longest_len_grn_fluor_outside_cont_outer)):
    pad_lengths.append([longest_len_grn_fluor_outside_cont_outer[i] - len(x) for x in grnchan_fluor_outside_contact_outer[i]])
grnchan_fluor_outside_contact_outer_padded = list()
for i in range(len(grnchan_fluor_outside_contact_outer)):
    padded = list()
    for j in range(len(grnchan_fluor_outside_contact_outer[i])):
        padded.append(np.pad(grnchan_fluor_outside_contact_outer[i][j], (int(pad_lengths[i][j]/2), int(pad_lengths[i][j]/2) + pad_lengths[i][j]%2), 'constant', constant_values=np.float('nan')))
    grnchan_fluor_outside_contact_outer_padded.append(padded)





plt.close('all')
for i in range(len(redchan_fluor_at_contact_inner_padded)):
    if i > 0:
        fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2, ncols=2)#, sharey='row')
        graph1 = ax1.imshow(redchan_fluor_at_contact_inner_padded[i])
        graph2 = ax2.imshow(redchan_fluor_outside_contact_inner_padded[i])
        graph3 = ax3.imshow(redchan_fluor_at_contact_outer_padded[i])
        graph4 = ax4.imshow(redchan_fluor_outside_contact_outer_padded[i])
        ax1.set_title('contact inner redchan')
        ax2.set_title('noncontact inner redchan')
        ax3.set_title('contact outer redchan')
        ax4.set_title('noncontact outer redchan')
        ax1.set_aspect('auto')
        ax2.set_aspect('auto')
        ax3.set_aspect('auto')
        ax4.set_aspect('auto')
        ax1.set_yticklabels(ax1.get_yticks() * time_res)
        ax2.set_yticklabels(ax2.get_yticks() * time_res)
        ax3.set_yticklabels(ax3.get_yticks() * time_res)
        ax4.set_yticklabels(ax4.get_yticks() * time_res)
        fig.text(-0.02, 0.5, 'Time point (minutes * 2)', va='center', rotation='vertical', fontsize=24)
        fig.text(0.5, -0.03, '%s'%lifeact[i]['Source_File'].iloc[0], ha='center', fontsize=24)
        divider1 = make_axes_locatable(ax1)
        divider2 = make_axes_locatable(ax2)
        divider3 = make_axes_locatable(ax3)
        divider4 = make_axes_locatable(ax4)
        cax1 = divider1.append_axes("right", size="5%", pad=0.05)
        cax2 = divider2.append_axes("right", size="5%", pad=0.05)
        cax3 = divider3.append_axes("right", size="5%", pad=0.05)
        cax4 = divider4.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(graph1, cax=cax1)
        plt.colorbar(graph2, cax=cax2)
        plt.colorbar(graph3, cax=cax3)
        plt.colorbar(graph4, cax=cax4)
        plt.tight_layout()
        #plt.show()
multipdf(Path.joinpath(out_dir, r"ee_redchan_kymos.pdf"))
plt.close('all')

 

for i in range(len(grnchan_fluor_at_contact_inner_padded)):
    if i > 0:
        fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2, ncols=2)
        graph1 = ax1.imshow(grnchan_fluor_at_contact_inner_padded[i])
        graph2 = ax2.imshow(grnchan_fluor_outside_contact_inner_padded[i])
        graph3 = ax3.imshow(grnchan_fluor_at_contact_outer_padded[i])
        graph4 = ax4.imshow(grnchan_fluor_outside_contact_outer_padded[i])
        ax1.set_title('contact inner grnchan')
        ax2.set_title('noncontact inner grnchan')
        ax3.set_title('contact outer grnchan')
        ax4.set_title('noncontact outer grnchan')
        ax1.set_aspect('auto')
        ax2.set_aspect('auto')
        ax3.set_aspect('auto')
        ax4.set_aspect('auto')
        ax1.set_yticklabels(ax1.get_yticks() * time_res)
        ax2.set_yticklabels(ax2.get_yticks() * time_res)
        ax3.set_yticklabels(ax3.get_yticks() * time_res)
        ax4.set_yticklabels(ax4.get_yticks() * time_res)
        fig.text(0.05, 0.5, 'Time (minutes)', va='center', rotation='vertical', fontsize=24)
        fig.text(0.5, 0.03, '%s'%lifeact[i]['Source_File'].iloc[0], ha='center', fontsize=24)
        divider1 = make_axes_locatable(ax1)
        divider2 = make_axes_locatable(ax2)
        divider3 = make_axes_locatable(ax3)
        divider4 = make_axes_locatable(ax4)
        cax1 = divider1.append_axes("right", size="5%", pad=0.05)
        cax2 = divider2.append_axes("right", size="5%", pad=0.05)
        cax3 = divider3.append_axes("right", size="5%", pad=0.05)
        cax4 = divider4.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(graph1, cax=cax1)
        plt.colorbar(graph2, cax=cax2)
        plt.colorbar(graph3, cax=cax3)
        plt.colorbar(graph4, cax=cax4)
        # plt.tight_layout()
        #plt.show()
multipdf(Path.joinpath(out_dir, r"ee_grnchan_kymos.pdf"))


plt.show()
plt.close('all')



### Check fluorescence of regions immediately on either side of the contact?
### Like... the first 5 pixels on either side? Is one side of comparable
### fluor to the contact? If you only look at the cases where fluor at contact
### is down relative to the regions immediately flanking the contact in eedb
### experimental condition do you see different behaviour for fluor vs cont diam?

cols_to_clean = ['grnchan_fluor_at_contact_inner', 'grnchan_fluor_outside_contact_inner', 
                 'grnchan_fluor_at_contact_outer', 'grnchan_fluor_outside_contact_outer']
la_contnd_df = clean_up_linescan(la_contnd_df, cols_to_clean)

flank_dist = 5 #pixels
la_contnd_df['grnchan_fluor_outside_contact_flnk%sl'%flank_dist] = [ls[:flank_dist] for ls in la_contnd_df['grnchan_fluor_outside_contact']]
la_contnd_df['grnchan_fluor_outside_contact_flnk%sr'%flank_dist] = [ls[::-1][:flank_dist] for ls in la_contnd_df['grnchan_fluor_outside_contact']]


la_contnd_df['grnchan_fluor_outside_contact_flnk%slmean'%flank_dist] = la_contnd_df['grnchan_fluor_outside_contact_flnk%sl'%flank_dist].apply(np.mean)
la_contnd_df['grnchan_fluor_outside_contact_flnk%srmean'%flank_dist] = la_contnd_df['grnchan_fluor_outside_contact_flnk%sr'%flank_dist].apply(np.mean)

la_contnd_df['grnchan_fluor_outside_contact_flnk%smeanmin'%flank_dist] = np.min(list(zip(la_contnd_df['grnchan_fluor_outside_contact_flnk%slmean'%flank_dist], 
                                                                                                la_contnd_df['grnchan_fluor_outside_contact_flnk%srmean'%flank_dist])),
                                                                                      axis=1)
la_contnd_df['grnchan_fluor_normed_mean_flnk%s'%flank_dist] = la_contnd_df['grnchan_fluor_at_contact_mean'] / la_contnd_df['grnchan_fluor_outside_contact_flnk%smeanmin'%flank_dist]


fig, ax = plt.subplots()
sns.scatterplot(x='Contact Surface Length (N.D.)', y='grnchan_fluor_normed_mean_flnk%s'%flank_dist, 
                data=la_contnd_df.query('Experimental_Condition == "ee1xDB"'), 
                palette='tab:blue', marker='o', alpha=0.7, ax=ax)#, hue='Experimental_Condition')
          
sns.scatterplot(x='Contact Surface Length (N.D.)', y='lifeact_fluor_normed_mean', 
                data=la_contnd_df.query('Experimental_Condition == "ee1xDB"'), 
                palette='tab:orange', marker='o', alpha=0.7, ax=ax)#, hue='Experimental_Condition')

ax.set_ylim(-0.1,2)
plt.close('all')
### Slightly shifts fluor vs contnd upwards and increases variation,
### but otherwise doesn't really seem to have an effect on the 
### fluor vs contnd... Still appears that one can get large
### changes in relative fluroescence without a corresponding
### change in contact diameter... 





### The LA-GFP (green channel) and the mbRFP (red channel) look like they track
### very closely together in some cases. The following section checks how well
### they correlate:

# First start by grouping the columns of interest based on which file/timelapse
# the data is from:
dladf_dt = la_contnd_df[['Average Angle (deg)', 'Contact Surface Length (N.D.)',
                         'grnchan_fluor_at_contact_mean', 'redchan_fluor_at_contact_mean',
                         'Source_File', 'Experimental_Condition']]

key_cols = ['Source_File', 'Experimental_Condition']
data_cols = ['Average Angle (deg)', 'Contact Surface Length (N.D.)',
             'grnchan_fluor_at_contact_mean', 'redchan_fluor_at_contact_mean']

dladf_dt = dladf_dt.groupby(key_cols)

# Second calculate the discrete difference for each of these groups/timelapses:
n=1 # This is level of discrete difference to take; think of it like being the nth derivative.
dladf_dt_data = pd.concat([group[data_cols].apply(lambda x: np.diff(x, n=n)) for (name, group) in dladf_dt]).reset_index(drop=True)
dladf_dt_keys = pd.concat([group[key_cols].iloc[n:] for (name, group) in dladf_dt]).reset_index(drop=True)

dladf_dt = pd.concat([dladf_dt_data, dladf_dt_keys], axis=1)

# Third we will create some qualitative correlations (-ve, +ve, or neither):
dladf_dt['qual_dcontnd_dt'] = dladf_dt['Contact Surface Length (N.D.)'] / abs(dladf_dt['Contact Surface Length (N.D.)'])
dladf_dt['qual_dcontnd_dt'][np.isnan(dladf_dt['qual_dcontnd_dt'])] = 0

dladf_dt['qual_dgrnchan_dt'] = dladf_dt['grnchan_fluor_at_contact_mean'] / abs(dladf_dt['grnchan_fluor_at_contact_mean'])
dladf_dt['qual_dgrnchan_dt'][np.isnan(dladf_dt['qual_dgrnchan_dt'])] = 0

dladf_dt['qual_dredchan_dt'] = dladf_dt['redchan_fluor_at_contact_mean'] / abs(dladf_dt['redchan_fluor_at_contact_mean'])
dladf_dt['qual_dredchan_dt'][np.isnan(dladf_dt['qual_dredchan_dt'])] = 0
# If the change is positive, then the column is +1, if -ve, then it's -1, and 
# if no change then it's 0.

# Now count how often pairs of columns have the same type of change:
dladf_dt['contnd_vs_grnchan'] = dladf_dt['qual_dcontnd_dt'] == dladf_dt['qual_dgrnchan_dt']
dladf_dt['contnd_vs_redchan'] = dladf_dt['qual_dcontnd_dt'] == dladf_dt['qual_dredchan_dt']
dladf_dt['grnchan_vs_redchan'] = dladf_dt['qual_dgrnchan_dt'] == dladf_dt['qual_dredchan_dt']

compar_cols = ['contnd_vs_grnchan', 'contnd_vs_redchan', 'grnchan_vs_redchan']
num_match = dladf_dt.groupby(key_cols)[compar_cols].agg([np.count_nonzero, np.size])

# Convert this to a percentage:
perc_match = dladf_dt.groupby(key_cols)[compar_cols].agg(lambda x: np.count_nonzero(x) / np.size(x)).reset_index()
# Note that if 2 sets of data are completely unrelated, then we their directions
# of change to match up 50% of the time.

# Now plot and save each comparison:
fig, ax = plt.subplots(figsize=(width_2col, height_2col))#(9,6))
grn = sns.pointplot(y='Source_File', x='contnd_vs_grnchan', 
                    # hue='Experimental_Condition', 
                    data=perc_match[perc_match['Experimental_Condition']=='ee1xMBS'], 
                    color='g', 
                    # palette=['g','g'], markers=['^','o'], 
                    linestyles=['',''],
                    scale=0.75, ax=ax)
red = sns.pointplot(y='Source_File', x='contnd_vs_redchan', 
                    # hue='Experimental_Condition', 
                    data=perc_match[perc_match['Experimental_Condition']=='ee1xMBS'], 
                    color='r', 
                    # palette=['r','r'], markers=['^','o'], 
                    linestyles=['',''], 
                    scale=0.75, ax=ax)
plt.setp(grn.collections, alpha=0.5) # Set alpha so that you can see overlaps
plt.setp(red.collections, alpha=0.5) # Set alpha so that you can see overlaps
# ax.set_xlabel('Norm. Fluor. variations matching Contact Diameter (N.D.) variations (%)')
ax.set_xlabel('Matching Variations (%)')
ax.set_xlim(0,1)
ax.axvline(0.5, c='k', ls=':')
ax.tick_params(axis='x', labelsize='20')
# ax.legend_.remove()
plt.tight_layout()
fig.savefig(Path.joinpath(out_dir, r'perc_matching_change_cont_vs_fluor_mbs.tif'), dpi=dpi_LineArt)
plt.close(fig)



fig, ax = plt.subplots(figsize=(width_2col, height_2col))#(9,6))
grn = sns.pointplot(y='Source_File', x='contnd_vs_grnchan', 
                    # hue='Experimental_Condition', 
                    data=perc_match[perc_match['Experimental_Condition']=='ee1xDB'], 
                    color='g', 
                    # palette=['g','g'], markers=['^','o'], 
                    linestyles=['',''],
                    scale=0.75, ax=ax)
red = sns.pointplot(y='Source_File', x='contnd_vs_redchan', 
                    # hue='Experimental_Condition', 
                    data=perc_match[perc_match['Experimental_Condition']=='ee1xDB'], 
                    color='r', 
                    # palette=['r','r'], markers=['^','o'], 
                    linestyles=['',''], 
                    scale=0.75, ax=ax)
plt.setp(grn.collections, alpha=0.5) # Set alpha so that you can see overlaps
plt.setp(red.collections, alpha=0.5) # Set alpha so that you can see overlaps
# ax.set_xlabel('Norm. Fluor. variations matching Contact Diameter (N.D.) variations (%)')
ax.set_xlabel('Matching Variations (%)')
ax.set_xlim(0,1)
ax.axvline(0.5, c='k', ls=':')
ax.tick_params(axis='x', labelsize='20')
# ax.legend_.remove()
plt.tight_layout()
fig.savefig(Path.joinpath(out_dir, r'perc_matching_change_cont_vs_fluor_db.tif'), dpi=dpi_LineArt)
plt.close(fig)



fig, ax = plt.subplots(figsize=(width_2col, height_2col))#(9,6))
grn = sns.pointplot(y='Source_File', x='grnchan_vs_redchan', 
                    # hue='Experimental_Condition', 
                    data=perc_match[perc_match['Experimental_Condition']=='ee1xMBS'], 
                    color='grey', 
                    # palette=['grey','grey'], markers=['^','o'], 
                    linestyles=['',''], 
                    scale=0.75, ax=ax)
ax.set_xlim(0,1)
ax.axvline(0.5, c='k', ls=':')
ax.set_xlabel('LifeAct-GFP vs. mbRFP Matching Variation (%)')
ax.tick_params(axis='x', labelsize='20')
plt.tight_layout()
fig.savefig(Path.joinpath(out_dir, r'perc_matching_change_grnchan_vs_redchan_mbs.tif'), dpi=dpi_LineArt)
plt.close(fig)



fig, ax = plt.subplots(figsize=(width_2col, height_2col))#(9,6))
grn = sns.pointplot(y='Source_File', x='grnchan_vs_redchan', 
                    # hue='Experimental_Condition', 
                    data=perc_match[perc_match['Experimental_Condition']=='ee1xDB'], 
                    color='grey', 
                    # palette=['grey','grey'], markers=['^','o'], 
                    linestyles=['',''], 
                    scale=0.75, ax=ax)
ax.set_xlim(0,1)
ax.axvline(0.5, c='k', ls=':')
ax.set_xlabel('LifeAct-GFP vs. mbRFP Matching Variation (%)')
ax.tick_params(axis='x', labelsize='20')

plt.tight_layout()
fig.savefig(Path.joinpath(out_dir, r'perc_matching_change_grnchan_vs_redchan_db.tif'), dpi=dpi_LineArt)
plt.close(fig)


### It looked like a couple of the Bc/B kinetics showed relative lifeact 
### following somewhat closely with predictions, though almost all of them did
### not. Therefore I also want to plot the areas of each red cell to see if the
### Bc/B kinetics that were close were of smaller areas (and therfore less
### light loss from imaging through less of the yolky cell):

# The Source_Files in question are the 2 that follow:
query_files = ['2019_10_02-crop2_z3.csv', '2019_10_22-series001_crop2_z3.csv']

### There were also some eedb condition cells that showed relative lifeact 
### following somewhat closely with predctions, so add those to query_files
### as well:
query_files += (['2020_01_17-series003_crop4_z1.csv', 
                 '2020_01_17-series003_crop5_z2.csv', 
                 '2020_02_27-series001_crop2_z2.csv', 
                 '2020_02_27-series001_crop4_z3.csv', 
                 '2020_02_27-series006_crop1_z3.csv'])

# Summarize the areas for detailed perusing:
area_summaries = lifeact_long.groupby(['Source_File', 'Experimental_Condition'])['Area of cell 1 (Red) (px**2)'].describe().reset_index()

# Plot the areas for a general overview:
fig, ax = plt.subplots(figsize=(width_2col, height_2col))
sns.boxplot(y='Source_File', x='Area of cell 1 (Red) (px**2)', 
            # hue='Experimental_Condition', 
            dodge=False,
            data=lifeact_long[lifeact_long['Experimental_Condition']=='ee1xMBS'], 
            ax=ax, color='grey', zorder=10)
# draw has to be called before trying to pull tick label positions or else the
# tick label positions will all = 0 for some reason.
plt.draw()
highlights = [x for x in ax.get_yticklabels() if x.get_text() in query_files]
hl_pos = [x.get_position()[1] for x in highlights]
[ax.axhline(x, c='k', ls=':', zorder=1) for x in hl_pos]
axb = ax.twiny()
axb_xticks = np.sqrt(ax.get_xticks() / np.pi)
axb_xticklabels = np.around(axb_xticks, decimals=2)
axb.set_xlim(ax.get_xlim())
ax.minorticks_on()
axb.minorticks_on()
axb.set_xticklabels(axb_xticklabels, fontdict={'fontsize':20})
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', which='minor', left=False)
ax.set_xlabel('LifeAct-GFP Cell Area (px$^2$)')
axb.set_xlabel('LifeAct-GFP Cell Radius (px)')
fig.savefig(Path.joinpath(out_dir, r'redcell_mean_areas_mbs.tif'), 
            dpi=dpi_LineArt, bbox_inches='tight')
plt.close(fig)



fig, ax = plt.subplots(figsize=(width_2col, height_2col))
sns.boxplot(y='Source_File', x='Area of cell 1 (Red) (px**2)', 
            # hue='Experimental_Condition', 
            dodge=False,
            data=lifeact_long[lifeact_long['Experimental_Condition']=='ee1xDB'], 
            ax=ax, color='grey', zorder=10)
# draw has to be called before trying to pull tick label positions or else the
# tick label positions will all = 0 for some reason.
plt.draw()
highlights = [x for x in ax.get_yticklabels() if x.get_text() in query_files]
hl_pos = [x.get_position()[1] for x in highlights]
[ax.axhline(x, c='k', ls=':', zorder=1) for x in hl_pos]
axb = ax.twiny()
axb_xticks = np.sqrt(ax.get_xticks() / np.pi)
axb_xticklabels = np.around(axb_xticks, decimals=2)
axb.set_xlim(ax.get_xlim())
# axb.set_xticklabels(axb_xticklabels)
ax.minorticks_on()
axb.minorticks_on()
axb.set_xticklabels(axb_xticklabels, fontdict={'fontsize':20})
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', which='minor', left=False)
ax.set_xlabel('LifeAct-GFP Cell Area (px$^2$)')
axb.set_xlabel('LifeAct-GFP Cell Radius (px)')
fig.savefig(Path.joinpath(out_dir, r'redcell_mean_areas_db.tif'), 
            dpi=dpi_LineArt, bbox_inches='tight')
plt.close(fig)





### This figure plots the cortical tension due to actin vs contact diameter for
### different relative values of energy of adhesion (eg. if the energy of 
### adhesion is 0.5 times the free cortical tension):
low_plat_peaks_fp = Path.joinpath(adh_kin_figs_out_dir, r"eedbFix_vs_eeLowPlat_peaks_contnd.tsv")
low_plat_peaks = pd.read_csv(low_plat_peaks_fp, delimiter='\t')
gen_log_plats_fp = Path.joinpath(adh_kin_figs_out_dir, r"gen_logistic_param_summ_contnd.csv")
gen_log_plats = pd.read_csv(gen_log_plats_fp)

ee_low_plat_peak = low_plat_peaks[low_plat_peaks['Experimental_Condition'] == 'ecto-ecto low plateau']['contnd_val'].iloc[0]
ee_gen_log_low_plat_peak = gen_log_plats[gen_log_plats['Experimental_Condition'] == 'ecto-ecto']['Low Plateau'].iloc[0]
ee_gen_log_high_plat_peak = gen_log_plats[gen_log_plats['Experimental_Condition'] == 'ecto-ecto']['High Plateau'].iloc[0]
gamma_half_div_beta_min, gamma_half_div_beta_max, step_size = -1, 2., 0.25
gamma_half_div_beta = np.arange(gamma_half_div_beta_min, gamma_half_div_beta_max, step_size)
betac_div_beta = np.arange(0, 2.01, 0.0001)
betastar_div_beta = [betac_div_beta - val for val in gamma_half_div_beta]
relcont = [reltension2contnd(val)*2 for val in betastar_div_beta]

# This chunk allows you to pick a color from a colormap, normalized to some data:
# Choose a colormap:
cmap = matplotlib.cm.coolwarm
# Normalize data:
norm = mcol.Normalize(vmin=gamma_half_div_beta.max()*-1, 
                      vmax=gamma_half_div_beta.max()
                      )
bounds = np.arange(gamma_half_div_beta_max*-1+step_size/2, gamma_half_div_beta_max+step_size/2, step_size)
norm = mcol.BoundaryNorm(bounds, cmap.N)


fig, ax = plt.subplots(figsize=(width_1col, width_1col))
[ax.plot(fx, betac_div_beta, c=cmap(norm(gamma)), zorder=3) for (fx, gamma) in zip(relcont, gamma_half_div_beta)]
ax.axvline(ee_low_plat_peak, ls='--', c='grey', zorder=2)
ax.axvline(ee_gen_log_low_plat_peak, ls='--', c='grey', zorder=2)
ax.axvline(ee_gen_log_high_plat_peak, ls='--', c='grey', zorder=2)
ax.axvline(expect_max_contnd, c='grey', zorder=2)
ax.axhline(1, c='grey', zorder=1)
ax.set_xlim(0, 3)
ax.set_ylim(0, 2)
[spine.set_zorder(10) for (k, spine) in ax.spines.items()]
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
# cax.set_yticklabels()
cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
cbar.set_ticks(gamma_half_div_beta)
# plt.cm.ScalarMappable.set_clim(ax, vmin=-1, vmax=1.75)
fig.savefig(Path.joinpath(out_dir, r'contnd vs different gamma_halfs.tif'), 
                          dpi=dpi_graph, bbox_inches='tight')
plt.close(fig)


# Create a separate colorbar for the gamma_half_div_beta



gg, bb = np.meshgrid(gamma_half_div_beta, betac_div_beta)
betastar_div_beta_grid = bb - gg
relcont_grid = reltension2contnd(betastar_div_beta_grid)




def contnd2reltension(contnd, initial_guess):#, Bc, B):
    """ Takes a numpy array. Outputs non-dimensional contact radius (note that 
    this is half of the contact surface length (ie. contact diameter) measured 
    by the measurement script)."""
    #betastar_div_beta = Bc/B - Gamma/(2B)
    def F(betastar_div_beta):
        return ( (4 * ( 1 - (betastar_div_beta)**2 )**(3/2)) / (4 - ((1 - betastar_div_beta)**2) * (2 + betastar_div_beta)) )**(1/3) - contnd
        
    betastar_div_beta = broyden1(F, initial_guess) 
    
    return betastar_div_beta

Bstar_div_B_low_plat = contnd2reltension(ee_gen_log_low_plat_peak/2, 1)
Bc_div_B = 1
Gamma_half_div_B_low_plat = Bc_div_B - Bstar_div_B_low_plat

Bstar_div_B_high_plat = contnd2reltension(ee_gen_log_high_plat_peak/2, 1)
Gamma_half_div_B_high_plat = Bc_div_B - Bstar_div_B_high_plat


betac_div_beta = np.arange(0, 2.01, 1e-4)
betastar_div_beta_low = betac_div_beta - Gamma_half_div_B_low_plat
# relcont_low_plat = [reltension2contnd(val)*2 for val in betastar_div_beta]
relcont_low_plat = reltension2contnd(betastar_div_beta_low)*2

#betac_div_beta = np.arange(0, 2.01, 0.01)
betastar_div_beta_high = betac_div_beta - Gamma_half_div_B_high_plat
# relcont_high_plat = [reltension2contnd(val)*2 for val in betastar_div_beta]
relcont_high_plat = reltension2contnd(betastar_div_beta_high)*2


fig, ax = plt.subplots(figsize=(width_1col, width_1col))
ax.plot(relcont_low_plat, betac_div_beta, c='k', zorder=3)
ax.plot(relcont_high_plat, betac_div_beta, c='k', zorder=3)
ax.axvline(ee_low_plat_peak, ls='--', c='grey', zorder=2)
ax.axvline(ee_gen_log_low_plat_peak, ls='--', c='grey', zorder=2)
ax.axvline(ee_gen_log_high_plat_peak, ls='--', c='grey', zorder=2)
ax.axvline(expect_max_contnd, c='grey', zorder=2)
ax.axhline(1, c='grey', zorder=1)
ax.set_xlim(0, 3)
ax.set_ylim(0, 2)
[spine.set_zorder(10) for (k, spine) in ax.spines.items()]




### Plot the raw values of the contact and non-contact actin and mbRFP fluors:
# cmap = plt.cm.get_cmap('jet')
for name, group in la_contnd_df.groupby('Source_File'):
    cont = group['Contact Surface Length (N.D.)']
    ang = group['Average Angle (deg)']
    lagfp_cont = group['grnchan_fluor_at_contact_mean']
    lagfp_cont_err = group['grnchan_fluor_at_contact_sem']
    lagfp_noncont = group['grnchan_fluor_outside_contact_mean']
    lagfp_noncont_err = group['grnchan_fluor_outside_contact_sem']
    mbrfp_cont = group['redchan_fluor_at_contact_mean']
    mbrfp_cont_err = group['redchan_fluor_at_contact_sem']
    mbrfp_noncont = group['redchan_fluor_outside_contact_mean']
    mbrfp_noncont_err = group['redchan_fluor_outside_contact_sem']
    minutes = group['Time (minutes)']
    
    ylim_gfp1 = max([la_contnd_df['grnchan_fluor_at_contact_mean'].max(), 
                la_contnd_df['grnchan_fluor_outside_contact_mean'].max()])
    ylim_gfp2 = max([la_contnd_df['grnchan_fluor_at_contact_sem'].max(), 
                la_contnd_df['grnchan_fluor_outside_contact_sem'].max()])
    ylim_rfp1 = max([la_contnd_df['redchan_fluor_at_contact_mean'].max(), 
                la_contnd_df['redchan_fluor_outside_contact_mean'].max()])
    ylim_rfp2 = max([la_contnd_df['redchan_fluor_at_contact_sem'].max(), 
                la_contnd_df['redchan_fluor_outside_contact_sem'].max()])
    # print('name: %s'%name,' / ','minutes.min: %f'%minutes.min(),' / ','minutes.max: %f'%minutes.max() )
    
    fig, (ax1,ax2) = plt.subplots(figsize=(9,4), nrows=1, ncols=2)
    fig.suptitle(name)
    ax1.errorbar(minutes, lagfp_cont, yerr=lagfp_cont_err, 
                 color='g', ecolor='lime', marker='.', markersize=5, alpha=1, 
                 zorder=10)
    ax1.errorbar(minutes, lagfp_noncont, yerr=lagfp_noncont_err, 
                 color='g', ecolor='lime', marker='.', markersize=5, alpha=1, 
                 zorder=10)
    # ax1.plot(minutes, cont, c='grey', lw=1, zorder=2, alpha=0.5)
    # ax1.plot(angle_predict_deg*2, contnd_predict*2, c='k', lw=1, ls='--', zorder=1)
    ax1.set_xlim(minutes.min()-1, minutes.max()+1)
    ax1.set_ylim(0, ylim_gfp1 + ylim_gfp2)
    # ax1.axvline(180, c='grey', lw=1, zorder=0)
    # ax1.axhline(expect_max_contnd, c='grey', lw=1, zorder=0)
    ax1.set_xlabel('Time (minutes)')
    ax1.set_ylabel('Fluorescence (photons)')
    ax1.set_adjustable('box')

    ax2.errorbar(minutes, mbrfp_cont, yerr=mbrfp_cont_err, 
                 color='r', ecolor='tomato', marker='.', markersize=5, alpha=1, 
                 zorder=10)
    ax2.errorbar(minutes, mbrfp_noncont, yerr=mbrfp_noncont_err, 
                 color='r', ecolor='tomato', marker='.', markersize=5, alpha=1, 
                 zorder=10)
    # ax2.plot(ang, la_normed, c='g', lw=1, zorder=2)
    # ax2.plot(ang, mbrfp_normed, c='r', lw=1, zorder=2, alpha=0.5)
    # ax2.plot(angle_predict_deg*2, beta_star_div_beta_contnd_range, c='k', lw=1, ls='--', zorder=1)
    ax2.set_xlim(minutes.min()-1, minutes.max()+1)
    ax2.set_ylim(0, ylim_rfp1 + ylim_rfp2)
    # ax2.axvline(180, c='grey', lw=1, zorder=0)
    # ax2.axhline()
    ax2.set_xlabel('Time (minutes)')
    ax2.set_ylabel('Fluorescence (photons)')
    ax2.set_adjustable('box')

    
    # plt.subplots_adjust(wspace=0.3, hspace=0.3)

multipdf(Path.joinpath(out_dir, r"fluor raw vals.pdf"))
plt.close('all')







fig, ax = plt.subplots()
sns.lineplot(x='Time (minutes)', y='Contact Surface Length (N.D.)', data=la_contnd_df[la_contnd_df['Experimental_Condition'] == 'ee1xMBS'], hue='Source_File', legend=False, marker='o', lw='1', ax=ax)
ax.loglog()
plt.close('all')




### Let's look at the distribution of fluoresnce values in single cells relative
### to the mean of all those values at each given timepoint:
lifeact_dbsc_clean = la_contnd_df[la_contnd_df['Experimental_Condition']=='ee1xDBsc'].copy()

# For the green channel:
la_fluor_out_cont = unp.uarray(lifeact_dbsc_clean['grnchan_fluor_outside_contact_mean'], lifeact_dbsc_clean['grnchan_fluor_outside_contact_sem'])
la_fluor_normed = lifeact_dbsc_clean['grnchan_fluor_outside_contact'] / la_fluor_out_cont
lifeact_dbsc_clean['lifeact_noncontact_fluor_normed_mean'] = [unp.nominal_values(x) for x in la_fluor_normed]
lifeact_dbsc_clean['lifeact_noncontact_fluor_normed_sem'] = [unp.std_devs(x) for x in la_fluor_normed] # It's actually the SEM

# For the red channel:
mbrfp_fluor_out_cont = unp.uarray(lifeact_dbsc_clean['redchan_fluor_outside_contact_mean'], lifeact_dbsc_clean['redchan_fluor_outside_contact_sem'])
mbrfp_fluor_normed = lifeact_dbsc_clean['redchan_fluor_outside_contact'] / mbrfp_fluor_out_cont
lifeact_dbsc_clean['mbrfp_noncontact_fluor_normed_mean'] = [unp.nominal_values(x) for x in mbrfp_fluor_normed]
lifeact_dbsc_clean['mbrfp_noncontact_fluor_normed_sem'] = [unp.std_devs(x) for x in mbrfp_fluor_normed] # It's actually the SEM


#la_fluor_flat = [(name, pd.Series(data=np.concatenate(group['lifeact_noncontact_fluor_normed_mean']), name='lifeact_noncontact_fluor_normed_mean')) for name,group in lifeact_dbsc_clean.groupby('Source_File')]
la_fluor_grouped_dbsc = [(name, 'LifeAct', np.concatenate(group['lifeact_noncontact_fluor_normed_mean'])) for name,group in lifeact_dbsc_clean.groupby('Source_File')]
la_fluor_df_dbsc = pd.DataFrame(data=la_fluor_grouped_dbsc, columns=['Source_File', 'Fluorophore', 'fluor_normed_mean'])

la_fluor_df_long_dbsc = pd.concat([pd.DataFrame(data={'Experimental_Condition':['ee1xDBsc']*len(fluor), 'Source_File':[name]*len(fluor), 'Fluorophore':[f_phore]*len(fluor), 'fluor_normed_mean':fluor}) for name,f_phore,fluor in la_fluor_grouped_dbsc])#.reset_index()
ee1xDBsc_summary_la = la_fluor_df_long_dbsc.groupby('Source_File').describe()

#mbrfp_fluor_flat = [(name, pd.Series(data=np.concatenate(group['mbrfp_noncontact_fluor_normed_mean']), name='mbrfp_noncontact_fluor_normed_mean')) for name,group in lifeact_dbsc_clean.groupby('Source_File')]
mbrfp_fluor_grouped_dbsc = [(name, 'mbRFP', np.concatenate(group['mbrfp_noncontact_fluor_normed_mean'])) for name,group in lifeact_dbsc_clean.groupby('Source_File')]
mbrfp_fluor_df_dbsc = pd.DataFrame(data=mbrfp_fluor_grouped_dbsc, columns=['Source_File', 'Fluorophore', 'fluor_normed_mean'])

mbrfp_fluor_df_long_dbsc = pd.concat([pd.DataFrame(data={'Experimental_Condition':['ee1xDBsc']*len(fluor), 'Source_File':[name]*len(fluor), 'Fluorophore':[f_phore]*len(fluor), 'fluor_normed_mean':fluor}) for name,f_phore,fluor in mbrfp_fluor_grouped_dbsc])#.reset_index()
ee1xDBsc_summary_mbrfp = mbrfp_fluor_df_long_dbsc.groupby('Source_File').describe()



### Also repeat the same for the ee1xDB (just to be fair):
lifeact_db_clean = la_contnd_df[la_contnd_df['Experimental_Condition']=='ee1xDB'].copy()
## Non-contact:
# For the green channel:
la_fluor_noncont = unp.uarray(lifeact_db_clean['grnchan_fluor_outside_contact_mean'], lifeact_db_clean['grnchan_fluor_outside_contact_sem'])
la_fluor_normed_noncont = lifeact_db_clean['grnchan_fluor_outside_contact'] / la_fluor_noncont
lifeact_db_clean['lifeact_noncontact_fluor_normed_mean'] = [unp.nominal_values(x) for x in la_fluor_normed_noncont]
lifeact_db_clean['lifeact_noncontact_fluor_normed_sem'] = [unp.std_devs(x) for x in la_fluor_normed_noncont] # It's actually the SEM

# For the red channel:
mbrfp_fluor_noncont = unp.uarray(lifeact_db_clean['redchan_fluor_outside_contact_mean'], lifeact_db_clean['redchan_fluor_outside_contact_sem'])
mbrfp_fluor_normed_noncont = lifeact_db_clean['redchan_fluor_outside_contact'] / mbrfp_fluor_noncont
lifeact_db_clean['mbrfp_noncontact_fluor_normed_mean'] = [unp.nominal_values(x) for x in mbrfp_fluor_normed_noncont]
lifeact_db_clean['mbrfp_noncontact_fluor_normed_sem'] = [unp.std_devs(x) for x in mbrfp_fluor_normed_noncont] # It's actually the SEM

la_fluor_grouped_db_noncont = [(name, 'LifeAct', np.concatenate(group['lifeact_noncontact_fluor_normed_mean'])) for name,group in lifeact_db_clean.groupby('Source_File')]
la_fluor_df_long_db_noncont = pd.concat([pd.DataFrame(data={'Experimental_Condition':['ee1xDB']*len(fluor), 'Source_File':[name]*len(fluor), 'Fluorophore':[f_phore]*len(fluor), 'noncontact_fluor_normed_mean':fluor}) for name,f_phore,fluor in la_fluor_grouped_db_noncont])#.reset_index()
ee1xDB_summary_la_noncont = la_fluor_df_long_db_noncont.groupby('Source_File', sort=False).describe()

mbrfp_fluor_grouped_db_noncont = [(name, 'mbRFP', np.concatenate(group['mbrfp_noncontact_fluor_normed_mean'])) for name,group in lifeact_db_clean.groupby('Source_File')]
mbrfp_fluor_df_long_db_noncont = pd.concat([pd.DataFrame(data={'Experimental_Condition':['ee1xDB']*len(fluor), 'Source_File':[name]*len(fluor), 'Fluorophore':[f_phore]*len(fluor), 'noncontact_fluor_normed_mean':fluor}) for name,f_phore,fluor in mbrfp_fluor_grouped_db_noncont])#.reset_index()
ee1xDB_summary_mbrfp_noncont = mbrfp_fluor_df_long_db_noncont.groupby('Source_File', sort=False).describe()


## Contact:
# For the green channel:
# la_fluor_cont = unp.uarray(lifeact_db_clean['grnchan_fluor_at_contact_mean'], lifeact_db_clean['grnchan_fluor_at_contact_sem'])
la_fluor_normed_cont = lifeact_db_clean['grnchan_fluor_at_contact'] / la_fluor_noncont
lifeact_db_clean['lifeact_contact_fluor_normed_mean'] = [unp.nominal_values(x) for x in la_fluor_normed_cont]
lifeact_db_clean['lifeact_contact_fluor_normed_sem'] = [unp.std_devs(x) for x in la_fluor_normed_cont] # It's actually the SEM

# For the red channel:
# mbrfp_fluor_cont = unp.uarray(lifeact_db_clean['redchan_fluor_at_contact_mean'], lifeact_db_clean['redchan_fluor_at_contact_sem'])
mbrfp_fluor_normed_cont = lifeact_db_clean['redchan_fluor_at_contact'] / mbrfp_fluor_noncont
lifeact_db_clean['mbrfp_contact_fluor_normed_mean'] = [unp.nominal_values(x) for x in mbrfp_fluor_normed_cont]
lifeact_db_clean['mbrfp_contact_fluor_normed_sem'] = [unp.std_devs(x) for x in mbrfp_fluor_normed_cont] # It's actually the SEM

la_fluor_grouped_db_cont = [(name, 'LifeAct', np.concatenate(group['lifeact_contact_fluor_normed_mean'])) for name,group in lifeact_db_clean.groupby('Source_File')]
la_fluor_df_long_db_cont = pd.concat([pd.DataFrame(data={'Experimental_Condition':['ee1xDB']*len(fluor), 'Source_File':[name]*len(fluor), 'Fluorophore':[f_phore]*len(fluor), 'contact_fluor_normed_mean':fluor}) for name,f_phore,fluor in la_fluor_grouped_db_cont])#.reset_index()
ee1xDB_summary_la_cont = la_fluor_df_long_db_cont.groupby('Source_File', sort=False).describe()

mbrfp_fluor_grouped_db_cont = [(name, 'mbRFP', np.concatenate(group['mbrfp_contact_fluor_normed_mean'])) for name,group in lifeact_db_clean.groupby('Source_File')]
mbrfp_fluor_df_long_db_cont = pd.concat([pd.DataFrame(data={'Experimental_Condition':['ee1xDB']*len(fluor), 'Source_File':[name]*len(fluor), 'Fluorophore':[f_phore]*len(fluor), 'contact_fluor_normed_mean':fluor}) for name,f_phore,fluor in mbrfp_fluor_grouped_db_cont])#.reset_index()
ee1xDB_summary_mbrfp_cont = mbrfp_fluor_df_long_db_cont.groupby('Source_File', sort=False).describe()


## Altogether:
# For the green channel:
la_fluor = [np.concatenate([arr, lifeact_db_clean['grnchan_fluor_at_contact'][i]]) for i,arr in enumerate(lifeact_db_clean['grnchan_fluor_outside_contact'])]
la_fluor_normed_all = la_fluor / la_fluor_noncont
lifeact_db_clean['lifeact_fluor_normed_mean'] = [unp.nominal_values(x) for x in la_fluor_normed_all]
lifeact_db_clean['lifeact_fluor_normed_sem'] = [unp.std_devs(x) for x in la_fluor_normed_all] # It's actually the SEM

# For the red channel:
mbrfp_fluor = [np.concatenate([arr, lifeact_db_clean['redchan_fluor_at_contact'][i]]) for i,arr in enumerate(lifeact_db_clean['redchan_fluor_outside_contact'])]
mbrfp_fluor_normed_all = mbrfp_fluor / mbrfp_fluor_noncont
lifeact_db_clean['mbrfp_fluor_normed_mean'] = [unp.nominal_values(x) for x in mbrfp_fluor_normed_all]
lifeact_db_clean['mbrfp_fluor_normed_sem'] = [unp.std_devs(x) for x in mbrfp_fluor_normed_all] # It's actually the SEM

la_fluor_grouped_db_all = [(name, 'LifeAct', np.concatenate(group['lifeact_fluor_normed_mean'])) for name,group in lifeact_db_clean.groupby('Source_File')]
la_fluor_df_long_db_all = pd.concat([pd.DataFrame(data={'Experimental_Condition':['ee1xDB']*len(fluor), 'Source_File':[name]*len(fluor), 'Fluorophore':[f_phore]*len(fluor), 'fluor_normed_mean':fluor}) for name,f_phore,fluor in la_fluor_grouped_db_all])#.reset_index()
ee1xDB_summary_la_all = la_fluor_df_long_db_all.groupby('Source_File', sort=False).describe()

mbrfp_fluor_grouped_db_all = [(name, 'mbRFP', np.concatenate(group['mbrfp_fluor_normed_mean'])) for name,group in lifeact_db_clean.groupby('Source_File')]
mbrfp_fluor_df_long_db_all = pd.concat([pd.DataFrame(data={'Experimental_Condition':['ee1xDB']*len(fluor), 'Source_File':[name]*len(fluor), 'Fluorophore':[f_phore]*len(fluor), 'fluor_normed_mean':fluor}) for name,f_phore,fluor in mbrfp_fluor_grouped_db_all])#.reset_index()
ee1xDB_summary_mbrfp_all = mbrfp_fluor_df_long_db_all.groupby('Source_File', sort=False).describe()





# put the single cell dissociation buffer data into a single long dataframe:
ee1xDBsc_fluor_df_long = pd.concat([la_fluor_df_long_dbsc, mbrfp_fluor_df_long_dbsc])
# also return mins and maxs to be plotted on the same plot:
la_mins_dbsc = ee1xDBsc_summary_la['fluor_normed_mean']['min'].reset_index()
la_maxs_dbsc = ee1xDBsc_summary_la['fluor_normed_mean']['max'].reset_index()
mbrfp_mins_dbsc = ee1xDBsc_summary_mbrfp['fluor_normed_mean']['min'].reset_index()
mbrfp_maxs_dbsc = ee1xDBsc_summary_mbrfp['fluor_normed_mean']['max'].reset_index()


# put the cell pair dissociation buffer noncontact data into a single long dataframe:
ee1xDB_fluor_df_long_noncont = pd.concat([la_fluor_df_long_db_noncont, mbrfp_fluor_df_long_db_noncont])
# also return mins and maxs to be plotted on the same plot:
la_mins_db_noncont = ee1xDB_summary_la_noncont['noncontact_fluor_normed_mean']['min'].reset_index()
la_maxs_db_noncont = ee1xDB_summary_la_noncont['noncontact_fluor_normed_mean']['max'].reset_index()
mbrfp_mins_db_noncont = ee1xDB_summary_mbrfp_noncont['noncontact_fluor_normed_mean']['min'].reset_index()
mbrfp_maxs_db_noncont = ee1xDB_summary_mbrfp_noncont['noncontact_fluor_normed_mean']['max'].reset_index()

# put the cell pair dissociation buffer contact data into a single long dataframe:
ee1xDB_fluor_df_long_cont = pd.concat([la_fluor_df_long_db_cont, mbrfp_fluor_df_long_db_cont])
# also return mins and maxs to be plotted on the same plot:
la_mins_db_cont = ee1xDB_summary_la_cont['contact_fluor_normed_mean']['min'].reset_index()
la_maxs_db_cont = ee1xDB_summary_la_cont['contact_fluor_normed_mean']['max'].reset_index()
mbrfp_mins_db_cont = ee1xDB_summary_mbrfp_cont['contact_fluor_normed_mean']['min'].reset_index()
mbrfp_maxs_db_cont = ee1xDB_summary_mbrfp_cont['contact_fluor_normed_mean']['max'].reset_index()

# put the cell pair dissociation buffer data into a single long dataframe:
ee1xDB_fluor_df_long = pd.concat([la_fluor_df_long_db_all, mbrfp_fluor_df_long_db_all])
# also return mins and maxs to be plotted on the same plot:
la_mins_db_all = ee1xDB_summary_la_all['fluor_normed_mean']['min'].reset_index()
la_maxs_db_all = ee1xDB_summary_la_all['fluor_normed_mean']['max'].reset_index()
mbrfp_mins_db_all = ee1xDB_summary_mbrfp_all['fluor_normed_mean']['min'].reset_index()
mbrfp_maxs_db_all = ee1xDB_summary_mbrfp_all['fluor_normed_mean']['max'].reset_index()





# make one big dataframe:
fluor_df_long = pd.concat([la_fluor_df_long_dbsc, mbrfp_fluor_df_long_dbsc, 
                                   la_fluor_df_long_db_all, mbrfp_fluor_df_long_db_all])
# also return mins and maxs to be plotted on the same plot:
summary = fluor_df_long.groupby(['Experimental_Condition', 'Source_File', 'Fluorophore'], sort=False).describe().droplevel(level=0,axis=1).reset_index()
# summary = fluor_df_long.groupby('Source_File', sort=False).describe()
# summary_mbrfp = fluor_df_long.groupby('Source_File').describe()

la_mins = summary[summary['Fluorophore'] == 'LifeAct'][['Experimental_Condition', 'Source_File', 'min']].reset_index()
la_maxs = summary[summary['Fluorophore'] == 'LifeAct'][['Experimental_Condition', 'Source_File', 'max']].reset_index()
mbrfp_mins = summary[summary['Fluorophore'] == 'mbRFP'][['Experimental_Condition', 'Source_File', 'min']].reset_index()
mbrfp_maxs = summary[summary['Fluorophore'] == 'mbRFP'][['Experimental_Condition', 'Source_File', 'max']].reset_index()

la_means = summary[summary['Fluorophore'] == 'LifeAct'][['Experimental_Condition', 'Source_File', 'mean']].reset_index()
la_25 = summary[summary['Fluorophore'] == 'LifeAct'][['Experimental_Condition', 'Source_File', '25%']].reset_index()
la_50 = summary[summary['Fluorophore'] == 'LifeAct'][['Experimental_Condition', 'Source_File', '50%']].reset_index()
la_75 = summary[summary['Fluorophore'] == 'LifeAct'][['Experimental_Condition', 'Source_File', '75%']].reset_index()






fig, ax = plt.subplots(figsize=(6.5, 6.5))
sns.violinplot(x='fluor_normed_mean', y='Source_File', hue='Fluorophore', data=fluor_df_long,
               palette=['c', 'm'], split=True, inner='quartile', linewidth=0.5, ax=ax)
ax.scatter(la_mins['min'], la_mins['Source_File'], c='c', marker='|')
ax.scatter(la_maxs['max'], la_maxs['Source_File'], c='c', marker='|')
ax.scatter(mbrfp_mins['min'], mbrfp_mins['Source_File'], c='m', marker='|')
ax.scatter(mbrfp_maxs['max'], mbrfp_maxs['Source_File'], c='m', marker='|')
ax.set_ylabel('Source File', fontsize=10)
ax.set_xlabel('Fluorescence (rel. total mean)', fontsize=10)
ax.tick_params(axis='both', labelsize=10)
fig.savefig(Path.joinpath(out_dir, r'ee1xDB_DBsc_fluor_distrib.tif'), 
            dpi=dpi_LineArt, bbox_inches='tight')
plt.close('fig')


fig, ax = plt.subplots(figsize=(6.5, 13))
sns.violinplot(x='fluor_normed_mean', y='Source_File', hue='Fluorophore', data=fluor_df_long,
               palette=['c', 'm'], split=True, inner='quartile', linewidth=0.5, ax=ax)
ax.scatter(la_mins['min'], la_mins['Source_File'], c='c', marker='|')
ax.scatter(la_maxs['max'], la_maxs['Source_File'], c='c', marker='|')
ax.scatter(mbrfp_mins['min'], mbrfp_mins['Source_File'], c='m', marker='|')
ax.scatter(mbrfp_maxs['max'], mbrfp_maxs['Source_File'], c='m', marker='|')
ax.set_xlim(0,4)
ax.set_ylabel('Source File', fontsize=10)
ax.set_xlabel('Fluorescence (rel. total mean)', fontsize=10)
ax.tick_params(axis='both', labelsize=10)
fig.savefig(Path.joinpath(out_dir, r'ee1xDB_DBsc_fluor_distrib_zoom.tif'), 
            dpi=dpi_LineArt, bbox_inches='tight')
plt.close('fig')


fig, ax = plt.subplots(figsize=(6.5, 3.3))
sns.violinplot(x='fluor_normed_mean', y='Source_File', hue='Fluorophore', data=ee1xDBsc_fluor_df_long,
               palette=['c', 'm'], split=True, inner='quartile', linewidth=0.5, ax=ax)
ax.scatter(la_mins_dbsc['min'], la_mins_dbsc['Source_File'], c='c', marker='|')
ax.scatter(la_maxs_dbsc['max'], la_maxs_dbsc['Source_File'], c='c', marker='|')
ax.scatter(mbrfp_mins_dbsc['min'], mbrfp_mins_dbsc['Source_File'], c='m', marker='|')
ax.scatter(mbrfp_maxs_dbsc['max'], mbrfp_maxs_dbsc['Source_File'], c='m', marker='|')
ax.set_ylabel('Source File', fontsize=10)
ax.set_xlabel('Fluorescence (rel. total mean)', fontsize=10)
ax.tick_params(axis='both', labelsize=10)
fig.savefig(Path.joinpath(out_dir, r'ee1xDBsc_fluor_distrib.tif'), 
            dpi=dpi_LineArt, bbox_inches='tight')


fig, ax = plt.subplots(figsize=(6.5, 3.3))
sns.violinplot(x='noncontact_fluor_normed_mean', y='Source_File', hue='Fluorophore', data=ee1xDB_fluor_df_long_noncont,
               palette=['c', 'm'], split=True, inner='quartile', linewidth=0.5, ax=ax)
ax.scatter(la_mins_db_noncont['min'], la_mins_db_noncont['Source_File'], c='c', marker='|')
ax.scatter(la_maxs_db_noncont['max'], la_maxs_db_noncont['Source_File'], c='c', marker='|')
ax.scatter(mbrfp_mins_db_noncont['min'], mbrfp_mins_db_noncont['Source_File'], c='m', marker='|')
ax.scatter(mbrfp_maxs_db_noncont['max'], mbrfp_maxs_db_noncont['Source_File'], c='m', marker='|')
db_noncont_xlims = ax.get_xlim()
ax.set_ylabel('Source File', fontsize=10)
ax.set_xlabel('Fluorescence (rel. total mean)', fontsize=10)
ax.tick_params(axis='both', labelsize=10)
fig.savefig(Path.joinpath(out_dir, r'ee1xDB_fluor_distrib_noncont.tif'), 
            dpi=dpi_LineArt, bbox_inches='tight')
plt.close('fig')


fig, ax = plt.subplots(figsize=(6.5, 3.3))
sns.violinplot(x='contact_fluor_normed_mean', y='Source_File', hue='Fluorophore', data=ee1xDB_fluor_df_long_cont,
               palette=['c', 'm'], split=True, inner='quartile', linewidth=0.5, ax=ax)
ax.scatter(la_mins_db_cont['min'], la_mins_db_cont['Source_File'], c='c', marker='|')
ax.scatter(la_maxs_db_cont['max'], la_maxs_db_cont['Source_File'], c='c', marker='|')
ax.scatter(mbrfp_mins_db_cont['min'], mbrfp_mins_db_cont['Source_File'], c='m', marker='|')
ax.scatter(mbrfp_maxs_db_cont['max'], mbrfp_maxs_db_cont['Source_File'], c='m', marker='|')
ax.set_xlim(db_noncont_xlims)
ax.set_ylabel('Source File', fontsize=10)
ax.set_xlabel('Fluorescence (rel. total mean)', fontsize=10)
ax.tick_params(axis='both', labelsize=10)
fig.savefig(Path.joinpath(out_dir, r'ee1xDB_fluor_distrib_cont.tif'), 
            dpi=dpi_LineArt, bbox_inches='tight')
plt.close('fig')


fig, ax = plt.subplots(figsize=(6.5, 3.3))
sns.violinplot(x='fluor_normed_mean', y='Source_File', hue='Fluorophore', data=ee1xDB_fluor_df_long,
               palette=['c', 'm'], split=True, inner='quartile', linewidth=0.5, ax=ax)
ax.scatter(la_mins_db_all['min'], la_mins_db_all['Source_File'], c='c', marker='|')
ax.scatter(la_maxs_db_all['max'], la_maxs_db_all['Source_File'], c='c', marker='|')
ax.scatter(mbrfp_mins_db_all['min'], mbrfp_mins_db_all['Source_File'], c='m', marker='|')
ax.scatter(mbrfp_maxs_db_all['max'], mbrfp_maxs_db_all['Source_File'], c='m', marker='|')
ax.set_ylabel('Source File', fontsize=10)
ax.set_xlabel('Fluorescence (rel. total mean)', fontsize=10)
ax.tick_params(axis='both', labelsize=10)
fig.savefig(Path.joinpath(out_dir, r'ee1xDB_fluor_distrib_all.tif'), 
            dpi=dpi_LineArt, bbox_inches='tight')
plt.close('fig')


#sns.violinplot(y='Experimental_Condition', x='25%', data=summary_noncont, inner='point')
#sns.violinplot(y='Experimental_Condition', x='50%', data=summary_noncont, inner='point')
#sns.violinplot(y='Experimental_Condition', x='75%', data=summary_noncont, inner='point')


print('Done.')


### ToDo:
#
# - check curvatures of ee1xmbs data

la_contnd_df_list_ee1xMBS = [df[df['Experimental_Condition'] == 'ee1xMBS'] for df in la_contnd_df_list]
indices = np.where([not(df.empty) for df in la_contnd_df_list_ee1xMBS])[0]
la_contnd_df_list_ee1xMBS = [la_contnd_df_list_ee1xMBS[i] for i in indices]
la_contnd_df_list_ee1xDB = [df[df['Experimental_Condition'] == 'ee1xDB'] for df in la_contnd_df_list]
indices = np.where([not(df.empty) for df in la_contnd_df_list_ee1xDB])[0]
la_contnd_df_list_ee1xDB = [la_contnd_df_list_ee1xDB[i] for i in indices]


### Plotting curvatures of 
red_curvature_value_headers = ['bin_0_curv_val_red', 'bin_1_curv_val_red', 'bin_2_curv_val_red', 
                               'bin_3_curv_val_red', 'bin_4_curv_val_red', 'bin_5_curv_val_red', 
                               'bin_6_curv_val_red', 'bin_7_curv_val_red', 'whole_cell_curv_val_red']

red_curvature_value_whole_header = ['whole_cell_curv_val_red']

blu_curvature_value_headers = ['bin_0_curv_val_blu', 'bin_1_curv_val_blu', 'bin_2_curv_val_blu', 
                               'bin_3_curv_val_blu', 'bin_4_curv_val_blu', 'bin_5_curv_val_blu', 
                               'bin_6_curv_val_blu', 'bin_7_curv_val_blu', 'whole_cell_curv_val_blu']

blu_curvature_value_whole_header = ['whole_cell_curv_val_blu']

bin_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'silver', 'purple']


plt.close('all') # Close all open figures so our PDF only contains the graphs from this loop:
for i,df in enumerate(la_contnd_df_list_ee1xMBS[:]):
    fig, ax = plt.subplots(figsize=(13.5,5))
    #[sns.lineplot(x='Time (minutes)', y=bin_num, data=df, marker='.', legend='full', ax=ax) for bin_num in red_curvature_value_headers]
    [ax.plot(df['Time (minutes)'], df[bin_num], marker='.', c=bin_colors[j]) for j,bin_num in enumerate(red_curvature_value_headers)]
    leg_lines = [mlines.Line2D([], [], color=clr, marker='', linestyle='-') for clr in [x.get_color() for x in ax.lines]]
    ax.legend(handles=leg_lines, labels=red_curvature_value_headers, ncol=5)
    # ax.set_ylim(-0.2, 3)
    ax.set_ylim(0)
    ax.axvline(0, c='lightgrey')
    ax.axhline(0, c='lightgrey')
    ax.set_title('%s'%df['Source_File'].iloc[0])
    # plt.show()
multipdf(Path.joinpath(out_dir, r"octant_curvatures_LAGFP_cell_ee1xMBS.pdf"))
plt.close('all')

for i,df in enumerate(la_contnd_df_list_ee1xDB[:]):
    fig, ax = plt.subplots(figsize=(13.5,5))
    #[sns.lineplot(x='Time (minutes)', y=bin_num, data=df, marker='.', legend='full', ax=ax) for bin_num in red_curvature_value_headers]
    [ax.plot(df['Time (minutes)'], df[bin_num], marker='.', c=bin_colors[j]) for j,bin_num in enumerate(red_curvature_value_headers)]
    leg_lines = [mlines.Line2D([], [], color=clr, marker='', linestyle='-') for clr in [x.get_color() for x in ax.lines]]
    ax.legend(handles=leg_lines, labels=red_curvature_value_headers, ncol=5)
    # ax.set_ylim(-0.2, 3)
    ax.set_ylim(0)
    ax.axvline(0, c='lightgrey')
    ax.axhline(0, c='lightgrey')
    ax.set_title('%s'%df['Source_File'].iloc[0])
    # plt.show()
multipdf(Path.joinpath(out_dir, r"octant_curvatures_LAGFP_cell_ee1xDB.pdf"))
plt.close('all')


# Plot only the curvature of the non-contact borders of all files on one graph:
# fig, ax = plt.subplots(figsize=(15,5))
# for i,df in enumerate(ee_list_dropna_time_corr_cpu_cellpair_only[:]):
#     [ax.plot(df['Time (minutes)'], df[bin_num]/ee_first_valid_curvature_red_cellpair_only.iloc[i], marker='.') for bin_num in red_curvature_value_whole_header]

mean_bin_curvatures = list()#red_curvature_value_headers)
for i,df in enumerate(la_contnd_df_list_ee1xMBS[:]):
    mean_bin_curvatures.append([df[bin_num].mean() for bin_num in red_curvature_value_headers] + [df['Source_File'].iloc[0], df['Experimental_Condition'].iloc[0]])
mean_bin_curvatures_ee1xMBS = pd.DataFrame(data=mean_bin_curvatures, columns=red_curvature_value_headers + ['Source_File', 'Experimental_Condition'])
mean_bin_curvatures_ee1xMBS_long = pd.melt(mean_bin_curvatures_ee1xMBS, id_vars=['Source_File', 'Experimental_Condition'], value_vars=red_curvature_value_headers, var_name='bin_label', value_name='mean_curvature')

mean_bin_curvatures = list()#red_curvature_value_headers)
for i,df in enumerate(la_contnd_df_list_ee1xDB[:]):
    mean_bin_curvatures.append([df[bin_num].mean() for bin_num in red_curvature_value_headers] + [df['Source_File'].iloc[0], df['Experimental_Condition'].iloc[0]])
mean_bin_curvatures_ee1xDB = pd.DataFrame(data=mean_bin_curvatures, columns=red_curvature_value_headers + ['Source_File', 'Experimental_Condition'])
mean_bin_curvatures_ee1xDB_long = pd.melt(mean_bin_curvatures_ee1xDB, id_vars=['Source_File', 'Experimental_Condition'], value_vars=red_curvature_value_headers, var_name='bin_label', value_name='mean_curvature')



### Plot the mean curvatures per bin for all timepoints for each file:
y_precis = 0.01
y_min = 0
y_max = round_up_to_multiple(max([mean_bin_curvatures_ee1xMBS_long['mean_curvature'].max(), 
                                  mean_bin_curvatures_ee1xDB_long['mean_curvature'].max()]), 
                             y_precis)

fig, ax = plt.subplots()
# sns.stripplot(x='bin_label', y='mean_curvature', data=mean_bin_curvatures_ee1xMBS_long, hue='Source_File', palette='Spectral', ax=ax)
sns.lineplot(x='bin_label', y='mean_curvature', data=mean_bin_curvatures_ee1xMBS_long, hue='Source_File', palette='Spectral', marker='o', ax=ax)
ax.set_xticklabels(red_curvature_value_headers, rotation=45, ha='right')
ax.legend(loc='best', bbox_to_anchor=(1,1), ncol=1)
ax.set_ylim(0, y_max)
ax.set_title('ee1xMBS')
plt.tight_layout()
fig.savefig(Path.joinpath(out_dir, 'octant_curvatures_means_LAGFP_cell_ee1xMBS.pdf'), 
            dpi=dpi_LineArt, bbox_inches='tight')


fig, ax = plt.subplots()
# sns.stripplot(x='bin_label', y='mean_curvature', data=mean_bin_curvatures_ee1xDB_long, hue='Source_File', palette='Spectral', ax=ax)
sns.lineplot(x='bin_label', y='mean_curvature', data=mean_bin_curvatures_ee1xDB_long, hue='Source_File', palette='Spectral', marker='o', ax=ax)
ax.set_xticklabels(red_curvature_value_headers, rotation=45, ha='right')
ax.legend(loc='best', bbox_to_anchor=(1,1), ncol=1)
ax.set_ylim(0, y_max)
ax.set_title('ee1xDB')
plt.tight_layout()
fig.savefig(Path.joinpath(out_dir, 'octant_curvatures_means_LAGFP_cell_ee1xDB.pdf'), 
            dpi=dpi_LineArt, bbox_inches='tight')


# seaborn violinplot in addition to mean curvature lineplots:

bin_curvatures_ee1xMBS = pd.concat(la_contnd_df_list_ee1xMBS)[red_curvature_value_headers[:-1] + ['Source_File', 'Experimental_Condition']]
bin_curvatures_ee1xMBS = pd.melt(bin_curvatures_ee1xMBS, id_vars=['Source_File', 'Experimental_Condition'], value_vars=red_curvature_value_headers[:-1], var_name='bin_label', value_name='curvature')
bin_curvatures_ee1xMBS_list = [(name, group) for name, group in bin_curvatures_ee1xMBS.groupby('Source_File')]

bin_curvatures_ee1xDB = pd.concat(la_contnd_df_list_ee1xDB)[blu_curvature_value_headers[:-1] + ['Source_File', 'Experimental_Condition']]
bin_curvatures_ee1xDB = pd.melt(bin_curvatures_ee1xDB, id_vars=['Source_File', 'Experimental_Condition'], value_vars=blu_curvature_value_headers[:-1], var_name='bin_label', value_name='curvature')
bin_curvatures_ee1xDB_list = [(name, group) for name, group in bin_curvatures_ee1xDB.groupby('Source_File')]



plt.close('all') # Close all open figures so our PDF only contains the graphs from this loop:
for name, df in bin_curvatures_ee1xMBS_list:
    fig, ax = plt.subplots(figsize=(13.5,6))
    sns.violinplot('bin_label', 'curvature', data=df, color='g', ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    # ax.set_ylim(0)
    # ax.axvline(0, c='lightgrey')
    ax.axhline(0, c='lightgrey')
    ax.set_title('%s'%df['Source_File'].iloc[0])
    plt.tight_layout()
    # plt.show()
multipdf(Path.joinpath(out_dir, r"octant_curvatures_LAGFP_cell_ee1xMBS_distrib.pdf"))
plt.close('all')

for name, df in bin_curvatures_ee1xDB_list:
    fig, ax = plt.subplots(figsize=(13.5,6))
    sns.violinplot('bin_label', 'curvature', data=df, color='g', ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    # ax.set_ylim(0)
    # ax.axvline(0, c='lightgrey')
    ax.axhline(0, c='lightgrey')
    ax.set_title('%s'%df['Source_File'].iloc[0])
    plt.tight_layout()
    # plt.show()
multipdf(Path.joinpath(out_dir, r"octant_curvatures_LAGFP_cell_ee1xDB_distrib.pdf"))
plt.close('all')


fig, ax = plt.subplots()
sns.violinplot(x='bin_label', y='curvature', data=bin_curvatures_ee1xMBS, color='steelblue', ax=ax)
ax.set_ylim(-0.005, 0.2)


fig, ax = plt.subplots()
sns.violinplot(x='bin_label', y='curvature', data=bin_curvatures_ee1xDB, color='steelblue', ax=ax)
ax.set_ylim(-0.005, 0.2)


# Are the means of all the bin curvatures the same?
curvature_stat_list = list()
for name, df in bin_curvatures_ee1xMBS_list:
    curvature_stat_list.append(stats.f_oneway(*[group['curvature'].dropna() for label, group in df.groupby('bin_label')]))
# count the number of datasets with significance < 0.05:
curv_num_signif = np.count_nonzero([f_result.pvalue < 0.05 for f_result in curvature_stat_list])
# count the number of datasets with significance < 0.01:
curv_num_very_signif = np.count_nonzero([f_result.pvalue < 0.01 for f_result in curvature_stat_list])



# Are the means of all the noncontact bin curvatures the same?
noncont_curv_stat_list = list()
for name, df in bin_curvatures_ee1xMBS_list:
    noncont_curv_stat_list.append(stats.f_oneway(*[group['curvature'].dropna() for label, group in df.groupby('bin_label')][:-1]))
# count the number of datasets with significance < 0.05:
noncontcurv_num_signif = np.count_nonzero([f_result.pvalue < 0.05 for f_result in noncont_curv_stat_list])
# count the number of datasets with significance < 0.01:
noncontcurv_num_very_signif = np.count_nonzero([f_result.pvalue < 0.01 for f_result in noncont_curv_stat_list])


# Are the regions flanking the contact site different from the other regions?
flankcurv_stat_list = list()
for name, df in bin_curvatures_ee1xMBS_list:
    flanking_curv_data = df[np.logical_or(df['bin_label'] == 'bin_0_curv_val_red', 
                                          df['bin_label'] == 'bin_6_curv_val_red')]
    other_noncont_curv_data = df[np.logical_or.reduce((df['bin_label'] == 'bin_1_curv_val_red', 
                                                       df['bin_label'] == 'bin_2_curv_val_red', 
                                                       df['bin_label'] == 'bin_3_curv_val_red', 
                                                       df['bin_label'] == 'bin_4_curv_val_red', 
                                                       df['bin_label'] == 'bin_5_curv_val_red'))]
    flankcurv_stat_list.append(stats.f_oneway(flanking_curv_data['curvature'].dropna(), other_noncont_curv_data['curvature'].dropna()))
# count the number of datasets with significance < 0.05:
flankcurv_num_signif = np.count_nonzero([f_result.pvalue < 0.05 for f_result in flankcurv_stat_list])
# count the number of datasets with significance < 0.01:
flankcurv_num_very_signif = np.count_nonzero([f_result.pvalue < 0.01 for f_result in flankcurv_stat_list])




# extract just bin medians:
median_bin_curvatures = list()#red_curvature_value_headers)
for i,df in enumerate(la_contnd_df_list_ee1xMBS[:]):
    median_bin_curvatures.append([df[bin_num].mean() for bin_num in red_curvature_value_headers] + [df['Source_File'].iloc[0], df['Experimental_Condition'].iloc[0]])
median_bin_curvatures_ee1xMBS = pd.DataFrame(data=median_bin_curvatures, columns=red_curvature_value_headers + ['Source_File', 'Experimental_Condition'])
median_bin_curvatures_ee1xMBS_long = pd.melt(median_bin_curvatures_ee1xMBS, id_vars=['Source_File', 'Experimental_Condition'], value_vars=red_curvature_value_headers, var_name='bin_label', value_name='median_curvature')


ee1xMBS_curv = pd.concat(la_contnd_df_list_ee1xMBS)

bin_curvatures_ee1xMBS_red = ee1xMBS_curv[red_curvature_value_headers[:-1] + ['Source_File', 'Experimental_Condition']]
bin_curvatures_ee1xMBS_red = pd.melt(bin_curvatures_ee1xMBS_red, id_vars=['Source_File', 'Experimental_Condition'], value_vars=red_curvature_value_headers[:-1], var_name='bin_label', value_name='curvature')
bin_curvatures_ee1xMBS_red_list = [(name, group) for name, group in bin_curvatures_ee1xMBS_red.groupby('Source_File')]

bin_curvatures_ee1xMBS_blu = ee1xMBS_curv[blu_curvature_value_headers[:-1] + ['Source_File', 'Experimental_Condition']]
bin_curvatures_ee1xMBS_blu = pd.melt(bin_curvatures_ee1xMBS_blu, id_vars=['Source_File', 'Experimental_Condition'], value_vars=blu_curvature_value_headers[:-1], var_name='bin_label', value_name='curvature')
bin_curvatures_ee1xMBS_blu_list = [(name, group) for name, group in bin_curvatures_ee1xMBS_blu.groupby('Source_File')]


bin_curvs_red_median = bin_curvatures_ee1xMBS_red.groupby(['Source_File', 'bin_label']).median().reset_index()
bin_curvs_blu_median = bin_curvatures_ee1xMBS_blu.groupby(['Source_File', 'bin_label']).median().reset_index()




curv_medians_list = [bin_curvs_red_median, bin_curvs_blu_median, ]
curv_medians_name_list = ['red curvature medians', 'blue curvature medians']
# curv_medians_phase_list = ['low', 'low', 'grow', 'grow', 'high', 'high']
curv_medians_cell_list = ['red', 'blu']


plt.close('all') # close all previously opened plots before creating a multipdf:
descending_curvature_medians = list()
for i,curv_median_df in enumerate(curv_medians_list):
    desc_curvs = list()
    for name,df in curv_median_df.groupby('Source_File'):
        df.insert(len(df.keys()), 'bin_color', bin_colors[:-1]) # add a column with the bin_label color
        # add a column that has bin_label as integers: bin_0_curv_val -> 0, bin_1_curv_val -> 1, ... etc.
        df.insert(len(df.keys()), 'bin_label_num', np.arange(0,8)) 
        # df.insert(len(df.keys()), 'phase', [curv_medians_phase_list[i]]*8) # add a column for phase
        df.insert(len(df.keys()), 'cell_color', [curv_medians_cell_list[i]]*8) # add a column for if red or grn cell
        desc_curvs.append(df.sort_values('curvature', ascending=False))
    
    
    # Assign a number to the sort curvature values (1 is the largest (very curved) and 8 is the smallest 
    # (straight line) curvature):
    ranks = np.arange(1,9)
    [df.insert(len(df.keys()), 'rank', ranks) for df in desc_curvs]
    desc_curvs_df = pd.concat(desc_curvs)
    descending_curvature_medians.append(desc_curvs_df)
    heatmap = desc_curvs_df.pivot('Source_File', 'rank', 'bin_label_num')
    
    # fix the colorbar so that the labels are centered over the colors:
    cbar_bounds = np.arange(0,len(bin_colors[:-1])+1)-0.5
    cbar_norm = mpl.colors.BoundaryNorm(cbar_bounds, len(bin_colors[:-1]))
    cbar_ticks = np.arange(0,8)
    cmap = mpl.colors.ListedColormap(bin_colors[:-1])
    # cmap_qual = mpl.colors.ListedColormap(qual_bin_colors[:-1])
    
    
    fig, ax = plt.subplots(figsize=(width_2col, width_1pt5col))
    sns.heatmap(heatmap, cmap=bin_colors[:-1], annot=True, cbar=False, ax=ax)#, cbar_kws={'ticks':bin_colors[:-1]})
    fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=cbar_norm), ticks=cbar_ticks)
    ax.set_title(curv_medians_name_list[i])
    fig.tight_layout
multipdf(Path.joinpath(out_dir, r"octant_curvatures_ee1xMBS_median_heatmaps.pdf"), tight=True, dpi=dpi_graph)


curv_medians_rank_df = pd.concat(descending_curvature_medians)


fig, ax = plt.subplots(figsize=(width_1pt5col,height_1col))
# sns.violinplot(x='bin_label_num', y='rank', color='w', data=curv_medians_rank_df, ax=ax, #)
#                linewidth=1, inner='box', cut=0)
sns.violinplot(x='bin_label_num', y='rank', hue='cell_color', palette=['tab:red','tab:blue'], data=curv_medians_rank_df, ax=ax, #)
               linewidth=1, inner='box', cut=0)
# sns.swarmplot(x='bin_label_num', y='rank', color='r', size=1, alpha=0.3, data=curv_medians_rank_df, ax=ax)
ax.set_yticks([0,2,4,6,8])
ax.set_ylabel('curvature rank')
ax.set_xlabel('bin number')
ax.legend(list(ax.legend_.get_patches()), ['red cell', 'blue cell'], loc='upper left', bbox_to_anchor=(0.0, 1.0), borderpad=0.03, ncol=2)
# ax.legend([],[],frameon=False)
plt.savefig(Path.joinpath(out_dir, r"octant_curv_medians_ee1xMBS_rel_distrib.tif"), 
            dpi=dpi_graph, bbox_inches='tight')

fig, ax = plt.subplots(figsize=(width_1pt5col,height_1col))
sns.swarmplot(x='bin_label_num', y='rank', color='k', size=2, alpha=0.33, data=curv_medians_rank_df, ax=ax)
ax.set_yticks([0,2,4,6,8])
ax.set_ylabel('curvature rank')
ax.set_xlabel('bin number')
# ax.legend(list(ax.legend_.get_patches()), ['low', 'growth', 'high'], loc='lower center', bbox_to_anchor=(0.5, -0.03), borderpad=0.03, ncol=3)
# ax.legend([],[],frameon=False)
plt.savefig(Path.joinpath(out_dir, r"octant_curv_medians_ee1xMBS_rel_distrib_v2.tif"), 
            dpi=dpi_graph, bbox_inches='tight')

fig, ax = plt.subplots(figsize=(width_1pt5col,height_1col))
sns.violinplot(x='bin_label_num', y='rank', color='w', data=curv_medians_rank_df, ax=ax, #)
               linewidth=1, inner=None, cut=0)
sns.swarmplot(x='bin_label_num', y='rank', color='r', size=2, alpha=0.5, data=curv_medians_rank_df, ax=ax)
ax.set_yticks([0,2,4,6,8])
ax.set_ylabel('curvature rank')
ax.set_xlabel('bin number')
# ax.legend(list(ax.legend_.get_patches()), ['low', 'growth', 'high'], loc='lower center', bbox_to_anchor=(0.5, -0.03), borderpad=0.03, ncol=3)
# ax.legend([],[],frameon=False)
plt.savefig(Path.joinpath(out_dir, r"octant_curv_medians_ee1xMBS_rel_distrib_v3.tif"), 
            dpi=dpi_graph, bbox_inches='tight')

# fig, ax = plt.subplots(figsize=(width_1pt5col,height_1col))
# sns.violinplot(x='bin_label_num', y='rank', color='w', data=curv_medians_rank_df, ax=ax, #)
#                linewidth=1, inner=None, cut=0)
# sns.swarmplot(x='bin_label_num', y='rank', hue='Source_File', size=2, alpha=0.99, data=curv_medians_rank_df, ax=ax)
# ax.set_yticks([0,2,4,6,8])
# ax.set_ylabel('curvature rank')
# ax.set_xlabel('bin number')
# # ax.legend(list(ax.legend_.get_patches()), ['low', 'growth', 'high'], loc='lower center', bbox_to_anchor=(0.5, -0.03), borderpad=0.03, ncol=3)
# # ax.legend([],[],frameon=False)
# plt.savefig(Path.joinpath(out_dir, r"octant_curv_medians_ee1xMBS_rel_distrib_v4.tif"), 
#             dpi=dpi_graph, bbox_inches='tight')

# for name,df in curv_medians_rank_df.groupby('Source_File'):
#     sns.lineplot(x='bin_label_num', y='rank', data=df)
#     plt.show()

for name,df in bin_curvatures_ee1xMBS_red.groupby('Source_File'):
    fig, ax = plt.subplots(figsize=(width_1pt5col, width_1pt5col))
    sns.lineplot(x='bin_label', y='curvature', data=df, marker='o', ax=ax)
    ax.set_xticklabels(red_curvature_value_headers, rotation=45, ha='right')
    # fig.tight_layout
    # plt.show()
multipdf(Path.joinpath(out_dir, r"octant_curvatures_ee1xMBS_lineplots.pdf"), tight=True, dpi=dpi_graph)



#####################################################################################################

### Some other stuff to look at regarding errors and theoretical lines:
fig, ax = plt.subplots()
ax.plot(theor_ang*2, theor_contnd)
ax.plot(r30sim_contnd_df['Average Angle (deg)'], theor_contnd, c='orange')
ax.scatter(r30sim_contnd_df['Average Angle (deg)'], theor_contnd, c='orange')
ax.plot(r30sim_contnd_df['Average Angle (deg)'], r30sim_contnd_df['Contact Surface Length (N.D.)'], c='g')
ax.scatter(r30sim_contnd_df['Average Angle (deg)'], r30sim_contnd_df['Contact Surface Length (N.D.)'], c='g')
ax.plot(theor_ang*2, r30sim_contnd_df['Contact Surface Length (N.D.)'], c='r')
ax.scatter(theor_ang*2, r30sim_contnd_df['Contact Surface Length (N.D.)'], c='r')




# How does contact diameter and contact angle change as the Z-position moves away from the center of
# the cell? (derivations can be found in my blue spiral-bound notebook).

cell_rad_init = 30
ang_real = np.linspace(0,90,91)# the real contact angle; radians
cell_rad_real = expected_spreading_sphere_params(ang_real, cell_rad_init)[1][0] # the real cell radius ; 30um
# cell_rad_appar = # the apparent cell radius
cont_real = expected_spreading_sphere_params(ang_real, cell_rad_init)[1][4] / 2 # the real contact radius
delta_z = np.linspace(0, 2, 3) # the deviation along the z axis from the center of the cell, contact; um


ang_vs_cont_dif_zs = list()
for dz in delta_z:
    cont_appar = np.sqrt( cont_real**2 - dz**2 ) # the apparent contact radius
    ang_appar = np.arcsin( np.sqrt( (cont_real**2 - dz**2) / (cell_rad_real**2 - dz**2) ) ) # the apparent contact angle; radians
    
    ang_vs_cont_dif_zs.append( (ang_appar, cont_appar) )

# i = 3
fig, ax = plt.subplots()
for (ang, cont) in ang_vs_cont_dif_zs:
    # ax.plot(np.rad2deg(ang), cont*2)
    ax.scatter(np.rad2deg(ang)*2, cont*2)
    ax.plot(ang_real*2, cont_real*2, c='k', zorder=0)
    # ax.scatter(np.rad2deg(ang)[i]*2, cont[i]*2)
    # ax.set_xlim(0,5)
    # ax.set_ylim(0,5)

# Ans: they all still fall on the same line. I thought that the apparent angle would increase as the 
# z position moved away from the center for some reason...



###############################################################################

### Need to output a file with some means and errors of angle data, contact 
### diameter data, and actin fluorescence data:
    
    
    

# test = la_contnd_df.query('Experimental_Condition == "ee1xMBS"')#['grnchan_fluor_at_contact_inner']
# test2 = test.groupby('Source_File')['grnchan_fluor_at_contact_inner']
# test3 = [(name, group) for (name,group) in test2]
# test4 = test.groupby('Source_File')['Time (minutes)']
# test5 = [(name, group) for (name,group) in test4]
# mid_pt_fluor_list = list()
# for x in test3:
#     mid_pt_fluor_list.append([y[len(y)//2] for y in x[1]])
    
# mid_pt_time_list = list()
# for x in test5:
#     mid_pt_time_list.append([y for y in x[1]])

# for i,x in enumerate(mid_pt_fluor_list):
#     plt.plot(mid_pt_time_list[i], x)
#     plt.draw()
#     plt.show()


