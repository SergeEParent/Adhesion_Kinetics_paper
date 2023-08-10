# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 16:44:46 2022

@author: Serge
"""


### Import some libraries:

### Needed to find directories, folders, files:
from pathlib import Path
### Used for splitting strings and working with regular expressions:
import re

### Import csv so that you can create comma style value files from your data.
### You can open csv files with excel. Pandas can open .csv files, so this is
### probably not needed anymore. 
# import csv

### Some data processing libraries:
### numpy of course:
import numpy as np
### For propagating errors (eg. means with standard devations):
from uncertainties import unumpy as unp
### pandas for more easily working with tables and DataFrames. Also, you can
### easily open comma separated value (.csv) files with pandas:
import pandas as pd
### For statistical tests and distributions:
from scipy import stats

### Some data plotting libraries:
from matplotlib import pyplot as plt
### Seaborn makes some nice graphs fairly easily, but it's mostly here for
### violin plots, univariate scatter plots, and plotting from dataframes more
### easily:
import seaborn as sns


### Import Pdfpages to save figures in to a multi-page PDF file:
from matplotlib.backends.backend_pdf import PdfPages




## Getting a RuntimeWarning somewhere and can't pinpoint it:
# import warnings
# warnings.simplefilter("error")
## RuntimeWarning is showing up when stats.sem is being applied to la_contnd_df



###############################################################################
###############################################################################
###############################################################################




### Define some functions:

def data_import_and_clean(path):
    """This function opens a bunch of .csv files using pandas and appends 4 
    columns containing: 1) the filename, 2) the folder name that the .csv files
    are kept in, and 3) the label of the z-position, and 4) the relative 
    z-position. Beware that the z-position is automatically retrieved from the 
    filename based on the following rule: the last instance of the letter 'z' 
    followed by a (possibly multidigit) number is taken as the z-label."""
    
    data_list = []
    for filename in path.glob('*.csv'):
        data_list.append(pd.read_csv(filename))
        
        data_list[-1]['Source_File'] = pd.Series(
                data=[filename.name]*data_list[-1].shape[0])
        
        data_list[-1]['Experimental_Condition'] = pd.Series(
                data=[path.name]*data_list[-1].shape[0])
        ## Look through the filename for the z-position signifier. I named the
        ## files such that this is at the end:
        if re.findall('z\d{1,}', filename.name): ## if you find a matching string:
            ## take final match as the z-label:
            z_label = re.findall('z\d{1,}', filename.name)[-1] 
            ## also record the numerical value of that z-label. Will be used to
            ## construct the relative z-position later based on the z_stepsize:
            z_num = re.search('\d{1,}', z_label).group(0)
        else:
            raise ValueError('z-position not written in filename as "_zn", '
                             'where "n" is a a number.')
        ## create the column for z_label:
        data_list[-1]['z_label'] = pd.Series(
                data=[z_label]*data_list[-1].shape[0])
        ## create a column for z_num (which is later replaced by rel_z_pos):
        data_list[-1]['z_num'] = pd.Series(
                data=[z_num]*data_list[-1].shape[0], dtype=float)
    data_list_long = pd.concat([x for x in data_list])
    
    return data_list, data_list_long



def cm2inch(num):
    """This function should be pretty self-explanatory, but it will take a 
    number in centimeters and convert it to inches."""
    return num/2.54



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
    nondim_df['Contact Surface Length (N.D.)'] = contact_diam_non_dim
    
    area_cellpair_non_dim = pd.concat(area_cellpair_non_dim)
    nondim_df['Area of the cell pair (N.D.)'] = area_cellpair_non_dim
    
    area_red_non_dim = pd.concat(area_red_non_dim)
    nondim_df['Area of cell 1 (Red) (N.D.)'] = area_red_non_dim
    
    area_grn_non_dim = pd.concat(area_blu_non_dim)
    nondim_df['Area of cell 2 (Blue) (N.D.)'] = area_grn_non_dim
    
    # nondim_df = nondim_df.rename(index=str, columns={'Contact Surface Length (px)': 'Contact Surface Length (N.D.)'})
    # nondim_df = nondim_df.rename(index=str, columns={'Area of the cell pair (px**2)': 'Area of the cell pair (N.D.)'})
    # nondim_df = nondim_df.rename(index=str, columns={'Area of cell 1 (Red) (px**2)': 'Area of cell 1 (Red) (N.D.)'})
    # nondim_df = nondim_df.rename(index=str, columns={'Area of cell 2 (Blue) (px**2)': 'Area of cell 2 (Blue) (N.D.)'})
    
    nondim_df_list = [nondim_df[nondim_df['Source_File'] == x] for x in np.unique(nondim_df['Source_File'])]

    
    return nondim_df_list, nondim_df, first_valid_area_red, first_valid_area_blu




def clean_up_linescan(dataframe, col_labels):
    """ This function just turns a stringified python list back into a list
    type object. """
    df = dataframe.copy()
    for col_lab in col_labels:
        # clean_col = [np.asarray([val.strip() for val in row[1:-1].split(' ') if val], dtype=float) for row in df[col_lab]]
        clean_col = [np.asarray([val.strip() for val in row[1:-1].split(',') if val], dtype=float) if type(row) == str else row for row in df[col_lab]]
        df[col_lab] = clean_col
    
    return df




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
    
    return




def line(x, slope, x0=0, y0=0):
    y = slope * (x - x0) + y0
    return y




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




def fluor_correction(fluor, z, correction_const):
    """This function applies a linear correction to measured fluorescence 
    values using the equation: 
    corrected_fluor * (1 - 1um * 0.0425) = measured_fluor.
    Returns the corrected fluorescence. """
    ## e.g. corrected_fluor * (1 + 1um * -0.0425) = measured_fluor ## the correction constant is negative here -> means 1 + ...
    corrected_fluor = fluor / (1 + z * correction_const)
        
    return corrected_fluor




### Also, define some settings that will be used everywhere:
        
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
um_per_px = 0.5069335

### Z resolution of the acquired timelapses (same for all movies):
# um_per_vx = 

### Step-size between optical sections (i.e. along the Z axis) in microns:
z_stepsize = 2

### Relation between z_labels and z_positions based on z_stepsize:


### Time resolution of the acquired timelapses (same for all movies):
### time_res = minutes/timeframe
time_res = 0.5




###############################################################################
###############################################################################
###############################################################################




### Create path objects describing the locations of the data and where the
### plots and analyses should be output to:

## Get the working directory of the script:
curr_dir = Path(__file__).parent.absolute()

## Data can be found in the following directory:
data_dir = Path.joinpath(curr_dir, r"data_ecto-ecto_lifeact/z_depth_analysis")
adh_kin_figs_out_dir = Path.joinpath(curr_dir, r"adhesion_analysis_figs_out")

## The folders for each experimental condition are:
ee1xMBS_path = Path.joinpath(data_dir, r"ee1xMBS")
ee1xDB_path = Path.joinpath(data_dir, r"ee1xDB")
ee1xDBsc_path = Path.joinpath(data_dir, r"ee1xDBsc")

## I want to know what z position was most in focus (i.e. closest to the middle
## of the cell), and include that information in the analysis of light loss. 
## So first, get the directories to those files:
best_focus_dir = Path.joinpath(curr_dir, r"data_ecto-ecto_lifeact")
ee1xMBS_bestfocus_path = Path.joinpath(best_focus_dir, r"ee1xMBS")
ee1xDB_bestfocus_path = Path.joinpath(best_focus_dir, r"ee1xDB")
ee1xDBsc_bestfocus_path = Path.joinpath(best_focus_dir, r"ee1xDBsc")

## Graphs/other analysis will be output to the following directory:
out_dir = Path.joinpath(curr_dir, r"lifeact_Zdepth_analysis_out")





### Open and clean up the data in the .csv files that was output from 
### "sct3_LifeAct measurement script":

## For ecto-ecto in 1xMBS load in your data and give your time column units:
## Note to future self: the data_import_and_clean function will create an
## 'Experimental_Condition' column with the folder that the csv files
## are in for values.
lifeact_orig, lifeact_long_orig = data_import_and_clean(ee1xMBS_path)
[x.rename({'Time':'Time (minutes)'}, axis='columns', inplace=True) for x in lifeact_orig]
lifeact_long_orig.rename({'Time':'Time (minutes)'}, axis='columns', inplace=True)


## For ecto-ecto in 1xDB load in your data and give your time column units:
lifeact_db_orig, lifeact_db_long_orig = data_import_and_clean(ee1xDB_path)
[x.rename({'Time':'Time (minutes)'}, axis='columns', inplace=True) for x in lifeact_db_orig]
lifeact_db_long_orig.rename({'Time':'Time (minutes)'}, axis='columns', inplace=True)


## Load ecto-ecto in 1xDB single cell examples:
lifeact_dbsc_orig, lifeact_dbsc_long_orig = data_import_and_clean(ee1xDBsc_path)
[x.rename({'Time':'Time (minutes)'}, axis='columns', inplace=True) for x in lifeact_dbsc_orig]
lifeact_dbsc_long_orig.rename({'Time':'Time (minutes)'}, axis='columns', inplace=True)



## For simplicity let's combine the treatments into a single list of dataframes
## and then also a single dataframe:
lifeact = lifeact_orig + lifeact_db_orig + lifeact_dbsc_orig
lifeact_long = pd.concat((lifeact_long_orig, lifeact_db_long_orig, lifeact_dbsc_long_orig))

## We're going to make a column with the filenames without the z-label info
## so that we can group all z-labels according to the filename. This 'root'
## filename will be called the timelapse filename, or 'timelapse_fname':
lifeact_long['TimeLapse_fname'] = lifeact_long['Source_File'].apply(lambda sf: re.search('((?!_z\d{1,}.csv).)*', sf).group(0))
## this line is going through each entry in the column 'Source_File' in the 
## dataframe lifeact_long and applying a function to each entry. The function 
## is regular expression that is searching for the pattern '_z\d{1,}.csv', and
## taking whatever is _not_ part of that pattern (this is called a negative 
## look-ahead, and is indicated by prefixing what you want to match with '?!').
## This all has to be enclosed by brackets so that the pattern is one "thing".
## This negative look-ahead pattern is then applied to each element in the 
## string, which is denoted by the '.' that comes after the negative look-ahead
## pattern. That dot is sometimes called a wildcard. The '*' tells us to repeat
## this pattern as often as needed. 
## So to summarize: '(?!_z\d{1,}.csv)' is telling us to look at the elements in
## the string ahead of where we currently are, and try to find the pattern
## '_z\d{1,}.csv', and take everything that is before this pattern as a match,
## '.' says to try this pattern on any character, 
## '*' says to repeat this as many times as needed until you've tried it on
## all the characters in the string. 




## Now I will make a column that tells me which z positions were thought to
## have the best focus. I went through each timelapse and picked the z_label 
## that best represents the equatorial z position:

## First open the .csv file with the recorded equatorial z_labels:
equatorial_z_path = Path.joinpath(data_dir, r"estimated_equatorial_z.csv")
equatorial_z_df = pd.read_csv(equatorial_z_path)



## Make a dictionary where each TimeLapse_fname has a corresponding equatorial 
## z position:
names_dict = equatorial_z_df.set_index('TimeLapse_fname').to_dict()['equatorial_z']



## Check that all TimeLapses in the Z_depth analysis have a corresponding 
## best-focus file:
best_focus_timelapse_fnames = list(lifeact_long.groupby('TimeLapse_fname').groups.keys())
z_analysis_timelapse_fnames = list(names_dict.keys())

assert best_focus_timelapse_fnames.sort() == z_analysis_timelapse_fnames.sort()



## Assign each timelapse its best-focus z_label:
lifeact_long['best_focus_z_num'] = lifeact_long['TimeLapse_fname'].map(names_dict)





# paths = [ee1xMBS_bestfocus_path, ee1xDB_bestfocus_path, ee1xDBsc_bestfocus_path]

# names_dict = {}

# ## Iterate through the paths and the filenames:
# for path in paths:
#     for filename in path.glob('*.csv'):
    
#         ## Look through the filename for the z-position signifier. I named the
#         ## files such that this is at the end:
#         if re.findall('z\d{1,}', filename.name): ## if you find a matching string:
#             ## take final match as the z-label:
#             z_label = re.findall('z\d{1,}', filename.name)[-1]
#             ## also get the timelapse name:
#             timelapse_fname = re.search('((?!_z\d{1,}.csv).)*', filename.name).group(0)
            
#         names_dict[timelapse_fname] = z_label


# ## Check that all TimeLapses in the Z_depth analysis have a corresponding 
# ## best-focus file:
# best_focus_timelapse_fnames = list(lifeact_long.groupby('TimeLapse_fname').groups.keys())
# z_analysis_timelapse_fnames = list(names_dict.keys())

# assert best_focus_timelapse_fnames.sort() == z_analysis_timelapse_fnames.sort()


# ## Assign each timelapse its best-focus z_label:
# lifeact_long['best_focus_z_num'] = lifeact_long['TimeLapse_fname'].map(names_dict)








    
    
## Define the relative optical section separation distances: 
    
## The data was collected starting from further away from the objective to
## closer towards the objective, therefore I will set the last z_num as 
## position 0 and every step before that becomes a more negative distance.

## First, convert the z_number to a physical distance. 
## The '- df['z_num'].max()' is there because I want the z_num closest to the
## objective to have a relative position of 0:
lifeact_long['z_num'] = (lifeact_long['z_num'] - lifeact_long['z_num'].max()) * z_stepsize

# ## Next, shift the physical distances of z_num so that the largest z_num is
# ## 0 and the smallest one is the largest number:
# lifeact_long['z_num'] -= lifeact_long['z_num'].max()

## Since the z-stack was acquired starting from further away from the objective
## and ending closer towards the objective, this means that the z-direction is
## inverted (i.e. z numbers get larger the closer they get to the objective),
## but what I want is for the z position to become more positive the _further_
## away from the objective we are, therefore we multiply by -1:
lifeact_long['z_num'] *= -1


## Now just rename the column from 'z_num' to 'rel_z_pos':
lifeact_long.rename({'z_num':'rel_z_pos'}, axis='columns', inplace=True)




# ## Mask the rows that are not valid:
# ## The fact that 'Source_File' is also being masked as 'nan' is causing a
# ## problem for the function 'nondim_data', therefore we will restore masked
# ## values for 'Source_File' as well as 'Experimental_Condition'...
# ## copy those two columns:
# sf_og = [df.Source_File for df in lifeact]
# ec_og = [df.Experimental_Condition for df in lifeact]

# sf_og_long = lifeact_long.Source_File
# ec_og_long = lifeact_long.Experimental_Condition

# lifeact = [x.mask(x['Mask? (manually defined by user who checks quality of cell borders)']) for x in lifeact]
# lifeact_long = lifeact_long.mask(lifeact_long['Mask? (manually defined by user who checks quality of cell borders)'])





# lifeact_temp = list()
# for i,df in enumerate(lifeact):
#     df['Source_File'] = sf_og[i]
#     df['Experimental_Condition'] = ec_og[i]
#     lifeact_temp.append(df)
# lifeact = lifeact_temp
# del(lifeact_temp)

# lifeact_long['Source_File'] = sf_og_long
# lifeact_long['Experimental_Condition'] = ec_og_long


### It might actually be better to mask the dataframes right before plotting
### only, unless I can find an easy option to restrict which columns are (or 
### are not) masked. One idea is to find the data columns that will be used and
### make a list containing those column names so that I can have only those
### columns masked. 


## Drop masked entries (because they often don't play well with some functions
## used for analyzing data):
# lifeact = [x.dropna(axis=0, subset=['Source_File']) for x in lifeact]
# lifeact_long = lifeact_long.dropna(axis=0, subset=['Source_File'])
## Can't drop masked entries yet, because then the dataframes will have 
## different lengths. In particular, this is a problem when trying to normalize
## different z planes to the lowest z plane.
# lifeact = [x.dropna(axis=0, subset=['Mask? (manually defined by user who checks quality of cell borders)']) for x in lifeact]
# lifeact_long = lifeact_long.dropna(axis=0, subset=['Mask? (manually defined by user who checks quality of cell borders)'])




## Non-dimensionalize the contact diameters and cell areas:
la_contnd_df_list, la_contnd_df, first_valid_area_red, first_valid_area_blu = nondim_data(lifeact, lifeact_long, 
                                                                                          norm_to_cell='red')


## The dimensional contact surface is missing from this data, so:
for df in la_contnd_df_list:
    df['Contact Surface Length (um)'] = df['Contact Surface Length (px)'] * um_per_px
la_contnd_df['Contact Surface Length (um)'] = la_contnd_df['Contact Surface Length (px)'] * um_per_px


## Clean up the linescan data that has been stringified so that it is returned
## back to array objects:
cols_to_clean = ['grnchan_fluor_at_contact', 'grnchan_fluor_outside_contact', 'grnchan_fluor_cytoplasmic',
                 'redchan_fluor_at_contact', 'redchan_fluor_outside_contact', 'redchan_fluor_cytoplasmic',
                 'bluchan_fluor_at_contact', 'bluchan_fluor_outside_contact', 'bluchan_fluor_cytoplasmic']
la_contnd_df_list = [clean_up_linescan(df, cols_to_clean) for df in la_contnd_df_list]
la_contnd_df = clean_up_linescan(la_contnd_df, cols_to_clean)



## Create a new column with the relative lifeact values (ie. Bc / B):
## Calculate means and SEMs:
la_contnd_df['grnchan_fluor_at_contact_mean'] = la_contnd_df['grnchan_fluor_at_contact'].apply(np.mean)
la_contnd_df['grnchan_fluor_at_contact_sem'] = la_contnd_df['grnchan_fluor_at_contact'].apply(stats.sem)
la_contnd_df['grnchan_fluor_at_contact_std'] = la_contnd_df['grnchan_fluor_at_contact'].apply(np.std)

la_contnd_df['grnchan_fluor_outside_contact_mean'] = la_contnd_df['grnchan_fluor_outside_contact'].apply(np.mean)
la_contnd_df['grnchan_fluor_outside_contact_sem'] = la_contnd_df['grnchan_fluor_outside_contact'].apply(stats.sem)
la_contnd_df['grnchan_fluor_outside_contact_std'] = la_contnd_df['grnchan_fluor_outside_contact'].apply(np.std)

la_contnd_df['grnchan_fluor_cytoplasmic_mean'] = la_contnd_df['grnchan_fluor_cytoplasmic'].apply(np.mean)
la_contnd_df['grnchan_fluor_cytoplasmic_sem'] = la_contnd_df['grnchan_fluor_cytoplasmic'].apply(stats.sem)
la_contnd_df['grnchan_fluor_cytoplasmic_std'] = la_contnd_df['grnchan_fluor_cytoplasmic'].apply(np.std)

## Generate uncertain floats to do error propagation with:
la_fluor_at_cont = unp.uarray(la_contnd_df['grnchan_fluor_at_contact_mean'], la_contnd_df['grnchan_fluor_at_contact_sem'])
la_fluor_out_cont = unp.uarray(la_contnd_df['grnchan_fluor_outside_contact_mean'], la_contnd_df['grnchan_fluor_outside_contact_sem'])
la_fluor_cytopl = unp.uarray(la_contnd_df['grnchan_fluor_cytoplasmic_mean'], la_contnd_df['grnchan_fluor_cytoplasmic_sem'])
la_fluor_normed = la_fluor_at_cont / la_fluor_out_cont
la_cytopl_fluor_normed = la_fluor_cytopl / la_fluor_out_cont
la_fluor_normed_sub = (la_fluor_at_cont - la_fluor_cytopl) / (la_fluor_out_cont - la_fluor_cytopl)

la_contnd_df['lifeact_fluor_normed_mean'] = unp.nominal_values(la_fluor_normed)
la_contnd_df['lifeact_fluor_normed_sem'] = unp.std_devs(la_fluor_normed) # It's actually the SEM
la_contnd_df['lifeact_cytoplasmic_fluor_normed_mean'] = unp.nominal_values(la_cytopl_fluor_normed)
la_contnd_df['lifeact_cytoplasmic_fluor_normed_sem'] = unp.std_devs(la_cytopl_fluor_normed)
la_contnd_df['lifeact_fluor_normed_sub_mean'] = unp.nominal_values(la_fluor_normed_sub)
la_contnd_df['lifeact_fluor_normed_sub_sem'] = unp.std_devs(la_fluor_normed_sub)



## Do same for mbrfp:
la_contnd_df['redchan_fluor_at_contact_mean'] = la_contnd_df['redchan_fluor_at_contact'].apply(np.mean)
la_contnd_df['redchan_fluor_at_contact_sem'] = la_contnd_df['redchan_fluor_at_contact'].apply(stats.sem)
la_contnd_df['redchan_fluor_at_contact_std'] = la_contnd_df['redchan_fluor_at_contact'].apply(np.std)

la_contnd_df['redchan_fluor_outside_contact_mean'] = la_contnd_df['redchan_fluor_outside_contact'].apply(np.mean)
la_contnd_df['redchan_fluor_outside_contact_sem'] = la_contnd_df['redchan_fluor_outside_contact'].apply(stats.sem)
la_contnd_df['redchan_fluor_outside_contact_std'] = la_contnd_df['redchan_fluor_outside_contact'].apply(np.std)

la_contnd_df['redchan_fluor_cytoplasmic_mean'] = la_contnd_df['redchan_fluor_cytoplasmic'].apply(np.mean)
la_contnd_df['redchan_fluor_cytoplasmic_sem'] = la_contnd_df['redchan_fluor_cytoplasmic'].apply(stats.sem)
la_contnd_df['redchan_fluor_cytoplasmic_std'] = la_contnd_df['redchan_fluor_cytoplasmic'].apply(np.std)

## Generate uncertain floats to do error propagation with:
mb_fluor_at_cont = unp.uarray(la_contnd_df['redchan_fluor_at_contact_mean'], la_contnd_df['redchan_fluor_at_contact_sem'])
mb_fluor_out_cont = unp.uarray(la_contnd_df['redchan_fluor_outside_contact_mean'], la_contnd_df['redchan_fluor_outside_contact_sem'])
mb_fluor_cytopl = unp.uarray(la_contnd_df['redchan_fluor_cytoplasmic_mean'], la_contnd_df['redchan_fluor_cytoplasmic_sem'])
mb_fluor_normed = mb_fluor_at_cont / mb_fluor_out_cont
mb_cytopl_fluor_normed = mb_fluor_cytopl / mb_fluor_out_cont
mb_fluor_normed_sub = (mb_fluor_at_cont - mb_fluor_cytopl) / (mb_fluor_out_cont - mb_fluor_cytopl)

la_contnd_df['mbrfp_fluor_normed_mean'] = unp.nominal_values(mb_fluor_normed)
la_contnd_df['mbrfp_fluor_normed_sem'] = unp.std_devs(mb_fluor_normed)
la_contnd_df['mbrfp_cytoplasmic_fluor_normed_mean'] = unp.nominal_values(mb_cytopl_fluor_normed)
la_contnd_df['mbrfp_cytoplasmic_fluor_normed_sem'] = unp.std_devs(mb_cytopl_fluor_normed)
la_contnd_df['mbrfp_fluor_normed_sub_mean'] = unp.nominal_values(mb_fluor_normed_sub)
la_contnd_df['mbrfp_fluor_normed_sub_sem'] = unp.std_devs(mb_fluor_normed_sub)



## Do same for AlexaFluor647 Dextran:
la_contnd_df['bluchan_fluor_at_contact_mean'] = la_contnd_df['bluchan_fluor_at_contact'].apply(np.mean)
la_contnd_df['bluchan_fluor_at_contact_sem'] = la_contnd_df['bluchan_fluor_at_contact'].apply(stats.sem)
la_contnd_df['bluchan_fluor_at_contact_std'] = la_contnd_df['bluchan_fluor_at_contact'].apply(np.std)

la_contnd_df['bluchan_fluor_outside_contact_mean'] = la_contnd_df['bluchan_fluor_outside_contact'].apply(np.mean)
la_contnd_df['bluchan_fluor_outside_contact_sem'] = la_contnd_df['bluchan_fluor_outside_contact'].apply(stats.sem)
la_contnd_df['bluchan_fluor_outside_contact_std'] = la_contnd_df['bluchan_fluor_outside_contact'].apply(np.std)

la_contnd_df['bluchan_fluor_cytoplasmic_mean'] = la_contnd_df['bluchan_fluor_cytoplasmic'].apply(np.mean)
la_contnd_df['bluchan_fluor_cytoplasmic_sem'] = la_contnd_df['bluchan_fluor_cytoplasmic'].apply(stats.sem)
la_contnd_df['bluchan_fluor_cytoplasmic_std'] = la_contnd_df['bluchan_fluor_cytoplasmic'].apply(np.std)

## Generate uncertain floats to do error propagation with:
dx_fluor_at_cont = unp.uarray(la_contnd_df['bluchan_fluor_at_contact_mean'], la_contnd_df['bluchan_fluor_at_contact_sem'])
dx_fluor_out_cont = unp.uarray(la_contnd_df['bluchan_fluor_outside_contact_mean'], la_contnd_df['bluchan_fluor_outside_contact_sem'])
dx_fluor_cytopl = unp.uarray(la_contnd_df['bluchan_fluor_cytoplasmic_mean'], la_contnd_df['bluchan_fluor_cytoplasmic_sem'])
dx_fluor_normed = dx_fluor_at_cont / dx_fluor_out_cont
dx_cytopl_fluor_normed = dx_fluor_cytopl / dx_fluor_out_cont
dx_fluor_normed_sub = (dx_fluor_at_cont - dx_fluor_cytopl) / (dx_fluor_out_cont - dx_fluor_cytopl)

la_contnd_df['af647dx_fluor_normed_mean'] = unp.nominal_values(dx_fluor_normed)
la_contnd_df['af647dx_fluor_normed_sem'] = unp.std_devs(dx_fluor_normed)
la_contnd_df['af647dx_cytoplasmic_fluor_normed_mean'] = unp.nominal_values(dx_cytopl_fluor_normed)
la_contnd_df['af647dx_cytoplasmic_fluor_normed_sem'] = unp.std_devs(dx_cytopl_fluor_normed)
la_contnd_df['af647dx_fluor_normed_sub_mean'] = unp.nominal_values(dx_fluor_normed_sub)
la_contnd_df['af647dx_fluor_normed_sub_sem'] = unp.std_devs(dx_fluor_normed_sub)



data_cols = [['grnchan_fluor_at_contact_mean', 'grnchan_fluor_at_contact_sem'], 
             ['grnchan_fluor_outside_contact_mean', 'grnchan_fluor_outside_contact_sem'], 
             ['grnchan_fluor_cytoplasmic_mean', 'grnchan_fluor_cytoplasmic_sem'], 
             ['redchan_fluor_at_contact_mean', 'redchan_fluor_at_contact_sem'], 
             ['redchan_fluor_outside_contact_mean', 'redchan_fluor_outside_contact_sem'], 
             ['redchan_fluor_cytoplasmic_mean', 'redchan_fluor_cytoplasmic_sem'], 
             ['bluchan_fluor_at_contact_mean', 'bluchan_fluor_at_contact_sem'], 
             ['bluchan_fluor_outside_contact_mean', 'bluchan_fluor_outside_contact_sem'], 
             ['bluchan_fluor_cytoplasmic_mean', 'bluchan_fluor_cytoplasmic_sem']]


## This will mask only the data columns that we will be working with:
for col in data_cols:
    la_contnd_df[col[0]].mask(la_contnd_df['Mask? (manually defined by user who checks quality of cell borders)'], inplace=True)
    la_contnd_df[col[1]].mask(la_contnd_df['Mask? (manually defined by user who checks quality of cell borders)'], inplace=True)
    


## The filename found in source_file actually contains the zlabel, therefore
## sf_subset is a list of dataframes where each dataframe is for 1 z-label,
## but what we actually want is 1 dataframe containing all 3 z-labels for a 
## given timelapse (so if the source file is '2019_09_13-series002_crop1_z1.csv')
## then we want to combine any dataframes with '2019_09_13-series002_crop1' in
## the filename. So:
timelapse_subset = list(la_contnd_df.groupby('TimeLapse_fname'))

## Now I want to normalize the mean fluorescences for each z position to the
## data for z1, for each timelapse file:
## first set up a list that we will later populate with the new normed data:
timelapse_subset_w_znorms = list()

## now start iterating through and normalizing data:
for fname, df in timelapse_subset:
    ## We need access to data from each z slice all at once, therefore we can't
    ## iterate through this, instead we will Pandas '.groupby' and '.apply':
    ## Group data according to z_label:
    z_grps = df.groupby('z_label', sort=False)
    
    ## Find the rel_z_pos for different z_labels:
    z_mins = z_grps['rel_z_pos'].apply(np.min)

    ## Find which of these z_labels is closest to the coverslip:
    closest_z = z_mins.idxmin()
    
    ## Retrieve the data from that z_label:    
    closest_df = z_grps.get_group(closest_z)

    ## Normalize the mean fluorescence values for each z_label to the fluor.
    ## of the z_label closest to the coverslip:
    

        
    ## You can --probably-- delete the commented out chunk of code below:
    
    # # df_temp = df.copy()
    # for col in data_cols:
    #     ## This line below is reversing the column name splitting it at the 
    #     ## first underscore, taking only the second part of the split (i.e. the
    #     ## first part of the string is in its original order), and then 
    #     ## re-reversing it so it is in hte correct order again, and lastly 
    #     ## appending '_unp' to the end to designate the entries are uncertainty 
    #     ## arrays:
    #     unp_col_name = col[0][::-1].split('_', maxsplit=1)[1][::-1]+'_unp'
    #     norm_col_name = col[0][::-1].split('_', maxsplit=1)[1][::-1]+'_unp_norm'
        
    #     ## This line is turning measurement entries into uncertainty arrays and
    #     ## then normalizing them to the measurements from the closest z slice:
        
    #     z_grps_temp = []
    #     for z_lab, grp in z_grps:
    #         ## Create column with uncertainty arrays:
    #         grp_temp = grp.copy()
    #         grp_temp[unp_col_name] = unp.uarray(grp_temp[col[0]], grp_temp[col[1]])

    #         ## Normalize uncertainty column of intensity data to the 
    #         ## z_label that is closest to the objective (which is also 
    #         ## presented as an uncertainty array):
    #         grp_temp[norm_col_name] = grp_temp[unp_col_name] / unp.uarray(closest_df[col[0]], closest_df[col[1]])
            
    #         z_grps_temp.append(grp_temp)
            
    #     # df_temp.append(z_grps_temp)
        
    #     # ## The lines below probably would have worked too, but I'm not 100%
    #     # ## sure that the order of the measurements is preserved when doing
    #     # ## .apply followed by .explode:
    #     # norm_data = z_grps.apply(lambda x: unp.uarray(x[col[0]], x[col[1]]) / unp.uarray(closest_df[col[0]], closest_df[col[1]]))
        
    #     # ## .explode will expand back out the results of .apply and a new
    #     # ## column called norm_col_name will be added to the dataframe
    #     # ## consisting of those results:
    #     # df[norm_col_name] = norm_data.explode().values

    #     # ## We should double check that all of the unp.uarray values are the
    #     # ## same as the ones in the data_cols to make sure that the order is
    #     # ## correct, just for the sake of due diligence:



    z_grps_temp = []
    for z_lab, grp in z_grps:
        ## Create a copy of the group to avoid a SettingWithCopy Warning:
        grp_temp = grp.copy()
        
        for col in data_cols:
            ## This line below is reversing the column name splitting it at the 
            ## first underscore, taking only the second part of the split (i.e. the
            ## first part of the string is in its original order), and then 
            ## re-reversing it so it is in hte correct order again, and lastly 
            ## appending '_unp' to the end to designate the entries are uncertainty 
            ## arrays:
            unp_col_name = col[0][::-1].split('_', maxsplit=1)[1][::-1]+'_unp'
            norm_col_name = col[0][::-1].split('_', maxsplit=1)[1][::-1]+'_unp_norm'
            
            ## This line is turning measurement entries into uncertainty arrays and
            ## then normalizing them to the measurements from the closest z slice:
            grp_temp[unp_col_name] = unp.uarray(grp_temp[col[0]], grp_temp[col[1]])
    
            ## Normalize uncertainty column of intensity data to the 
            ## z_label that is closest to the objective (which is also 
            ## presented as an uncertainty array):
            grp_temp[norm_col_name] = grp_temp[unp_col_name] / unp.uarray(closest_df[col[0]], closest_df[col[1]])
        
        ## Remake the new z groups:
        z_grps_temp.append(grp_temp)


    ## Remake the new dataframe:
    df_temp = pd.concat(z_grps_temp)          
        
    ## We should double check that all of the unp.uarray values are the
    ## same as the ones in the data_cols to make sure that the order is
    ## correct, just for the sake of due diligence:
    ## The following will raise an AssertionError if there are any
    ## differences between the two dataframes.
    pd.testing.assert_frame_equal(df, df_temp[list(df.keys())])
    ## If this check passes, then we will replace df with df_temp by updating
    ## the new list:

    # exit() 
    timelapse_subset_w_znorms.append((fname, df_temp))




### Alright... I think that cleaning up and preparing the DataFrames is done.
# exit()



###############################################################################
###############################################################################
###############################################################################




### Time to plot some graphs and look at the data in different ways:

## First mask the rows from timepoints that are had poor quality cell outlines:
timelapse_subset_w_znorms = [(fname, df.mask(df['Mask? (manually defined by user who checks quality of cell borders)'])) for fname, df in timelapse_subset_w_znorms]

## Drop the masked entries (unnecessary?):
timelapse_subset_w_znorms = [(fname, df.dropna(axis=0, subset=['Source_File'])) for fname, df in timelapse_subset_w_znorms]








for fname, df in timelapse_subset_w_znorms:
    
    expt_cond = df['Experimental_Condition'].iloc[0]
    equatorial_z = df['best_focus_z_num'].iloc[0]

    ## associate different z-labels with different colours:
    zlabs = [zlab for zlab,df in list(df.groupby('z_label'))]
    zlab_colors = dict([(zlab, sns.color_palette('tab10')[i]) for i,zlab in enumerate(zlabs)])  


    fig, ((ax1, ax4, ax7), (ax2, ax5, ax8), (ax3, ax6, ax9)) = plt.subplots(figsize=(15,12), 
                                                                            ncols=3, nrows=3)



    for zlab, zlab_df in list(df.groupby('z_label')):
        
        # print(zlab)
  
        ax1.errorbar(x=zlab_df['Time (minutes)'], y=zlab_df['grnchan_fluor_at_contact_mean'],
                     yerr=zlab_df['grnchan_fluor_at_contact_sem'], 
                     color=zlab_colors[zlab], label=zlab)

        ax2.errorbar(x=zlab_df['Time (minutes)'], y=zlab_df['grnchan_fluor_outside_contact_mean'],
                     yerr=zlab_df['grnchan_fluor_outside_contact_sem'], 
                     color=zlab_colors[zlab], label=zlab)
    
        ax3.errorbar(x=zlab_df['Time (minutes)'], y=zlab_df['grnchan_fluor_cytoplasmic_mean'],
                     yerr=zlab_df['grnchan_fluor_cytoplasmic_sem'], 
                     color=zlab_colors[zlab], label=zlab)
    
    
        ax4.errorbar(x=zlab_df['Time (minutes)'], y=zlab_df['redchan_fluor_at_contact_mean'],
                     yerr=zlab_df['redchan_fluor_at_contact_sem'], 
                     color=zlab_colors[zlab], label=zlab)

        ax5.errorbar(x=zlab_df['Time (minutes)'], y=zlab_df['redchan_fluor_outside_contact_mean'],
                     yerr=zlab_df['redchan_fluor_outside_contact_sem'], 
                     color=zlab_colors[zlab], label=zlab)
    
        ax6.errorbar(x=zlab_df['Time (minutes)'], y=zlab_df['redchan_fluor_cytoplasmic_mean'],
                     yerr=zlab_df['redchan_fluor_cytoplasmic_sem'], 
                     color=zlab_colors[zlab], label=zlab)

    
        ax7.errorbar(x=zlab_df['Time (minutes)'], y=zlab_df['bluchan_fluor_at_contact_mean'],
                     yerr=zlab_df['bluchan_fluor_at_contact_sem'], 
                     color=zlab_colors[zlab], label=zlab)

        ax8.errorbar(x=zlab_df['Time (minutes)'], y=zlab_df['bluchan_fluor_outside_contact_mean'],
                     yerr=zlab_df['bluchan_fluor_outside_contact_sem'], 
                     color=zlab_colors[zlab], label=zlab)
    
        ax9.errorbar(x=zlab_df['Time (minutes)'], y=zlab_df['bluchan_fluor_cytoplasmic_mean'],
                     yerr=zlab_df['bluchan_fluor_cytoplasmic_sem'], 
                     color=zlab_colors[zlab], label=zlab)
    
        # plt.draw()
    
    ax1.set_ylabel('Contact Intensity')
    ax1.set_xlabel('')
    ax1.set_title('LifeAct-GFP', color='g')
    [ax1.spines[edge].set_color('g') for edge in ax1.spines.keys()]
    
    ax2.set_ylabel('Non-contact Intensity')
    ax2.set_xlabel('')
    [ax2.spines[edge].set_color('g') for edge in ax2.spines.keys()]

    ax3.set_ylabel('Cytoplasmic Intensity')
    ax3.set_xlabel('Time (minutes)')
    [ax3.spines[edge].set_color('g') for edge in ax3.spines.keys()]

    ax4.set_ylabel('')
    ax4.set_xlabel('')
    ax4.set_title('mbRFP', color='m')
    [ax4.spines[edge].set_color('m') for edge in ax4.spines.keys()]

    ax5.set_ylabel('')
    ax5.set_xlabel('')
    [ax5.spines[edge].set_color('m') for edge in ax5.spines.keys()]

    ax6.set_ylabel('')
    ax6.set_xlabel('Time (minutes)')
    [ax6.spines[edge].set_color('m') for edge in ax6.spines.keys()]
    
    ax7.set_ylabel('')
    ax7.set_xlabel('')
    ax7.set_title('AF647-dextran', color='c')
    [ax7.spines[edge].set_color('c') for edge in ax7.spines.keys()]

    ax8.set_ylabel('')
    ax8.set_xlabel('')
    [ax8.spines[edge].set_color('c') for edge in ax8.spines.keys()]

    ax9.set_ylabel('')
    ax9.set_xlabel('Time (minutes)')
    [ax9.spines[edge].set_color('c') for edge in ax9.spines.keys()]
  
    
    
    fig.suptitle('%s: %s (equatorial z: %s)'%(expt_cond, fname, equatorial_z), x=0.5, y=0.92)#, ha='right')
    
    handles, labels = fig.gca().get_legend_handles_labels()
    unique_label_handle = dict(zip(labels, handles))
    fig.legend(unique_label_handle.values(), unique_label_handle.keys(), 
               ncol=3, loc='center right', bbox_to_anchor=(0.9, 0.92))
    
    
    
    
    # plt.tight_layout()

    
    

if out_dir.exists():    
    multipdf(Path.joinpath(out_dir,'dif_z_mean_fluors.pdf'))
else:
    out_dir.mkdir()
    multipdf(Path.joinpath(out_dir,'dif_z_mean_fluors.pdf'))

plt.close('all')








## Now plot some normalized data:
        
for fname, df in timelapse_subset_w_znorms:
    
    expt_cond = df['Experimental_Condition'].iloc[0]
    equatorial_z = df['best_focus_z_num'].iloc[0]


    ## associate different z-labels with different colours:
    zlabs = [zlab for zlab,df in list(df.groupby('z_label'))]
    zlab_colors = dict([(zlab, sns.color_palette('tab10')[i]) for i,zlab in enumerate(zlabs)])  


    fig, ((ax1, ax4, ax7), (ax2, ax5, ax8), (ax3, ax6, ax9)) = plt.subplots(figsize=(15,12), 
                                                                            ncols=3, nrows=3)



    for zlab, zlab_df in list(df.groupby('z_label')):
        
        # print(zlab)
  
        ax1.errorbar(x=zlab_df['Time (minutes)'], y=unp.nominal_values(zlab_df['grnchan_fluor_at_contact_unp_norm']),
                     yerr=unp.std_devs(zlab_df['grnchan_fluor_at_contact_unp_norm']), 
                     color=zlab_colors[zlab], label=zlab)

        ax2.errorbar(x=zlab_df['Time (minutes)'], y=unp.nominal_values(zlab_df['grnchan_fluor_outside_contact_unp_norm']),
                     yerr=unp.std_devs(zlab_df['grnchan_fluor_outside_contact_unp_norm']), 
                     color=zlab_colors[zlab], label=zlab)
    
        ax3.errorbar(x=zlab_df['Time (minutes)'], y=unp.nominal_values(zlab_df['grnchan_fluor_cytoplasmic_unp_norm']),
                     yerr=unp.std_devs(zlab_df['grnchan_fluor_cytoplasmic_unp_norm']), 
                     color=zlab_colors[zlab], label=zlab)
    
    
        ax4.errorbar(x=zlab_df['Time (minutes)'], y=unp.nominal_values(zlab_df['redchan_fluor_at_contact_unp_norm']),
                     yerr=unp.std_devs(zlab_df['redchan_fluor_at_contact_unp_norm']), 
                     color=zlab_colors[zlab], label=zlab)

        ax5.errorbar(x=zlab_df['Time (minutes)'], y=unp.nominal_values(zlab_df['redchan_fluor_outside_contact_unp_norm']),
                     yerr=unp.std_devs(zlab_df['redchan_fluor_outside_contact_unp_norm']), 
                     color=zlab_colors[zlab], label=zlab)
    
        ax6.errorbar(x=zlab_df['Time (minutes)'], y=unp.nominal_values(zlab_df['redchan_fluor_cytoplasmic_unp_norm']),
                     yerr=unp.std_devs(zlab_df['redchan_fluor_cytoplasmic_unp_norm']), 
                     color=zlab_colors[zlab], label=zlab)

    
        ax7.errorbar(x=zlab_df['Time (minutes)'], y=unp.nominal_values(zlab_df['bluchan_fluor_at_contact_unp_norm']),
                     yerr=unp.std_devs(zlab_df['bluchan_fluor_at_contact_unp_norm']), 
                     color=zlab_colors[zlab], label=zlab)

        ax8.errorbar(x=zlab_df['Time (minutes)'], y=unp.nominal_values(zlab_df['bluchan_fluor_outside_contact_unp_norm']),
                     yerr=unp.std_devs(zlab_df['bluchan_fluor_outside_contact_unp_norm']), 
                     color=zlab_colors[zlab], label=zlab)
    
        ax9.errorbar(x=zlab_df['Time (minutes)'], y=unp.nominal_values(zlab_df['bluchan_fluor_cytoplasmic_unp_norm']),
                     yerr=unp.std_devs(zlab_df['bluchan_fluor_cytoplasmic_unp_norm']), 
                     color=zlab_colors[zlab], label=zlab)
    
        # plt.draw()
    
    ax1.set_ylabel('Rel. Contact Intensity')
    ax1.set_xlabel('')
    ax1.set_title('LifeAct-GFP', color='g')
    [ax1.spines[edge].set_color('g') for edge in ax1.spines.keys()]
    
    ax2.set_ylabel('Rel. Non-contact Intensity')
    ax2.set_xlabel('')
    [ax2.spines[edge].set_color('g') for edge in ax2.spines.keys()]

    ax3.set_ylabel('Rel. Cytoplasmic Intensity')
    ax3.set_xlabel('Time (minutes)')
    [ax3.spines[edge].set_color('g') for edge in ax3.spines.keys()]

    ax4.set_ylabel('')
    ax4.set_xlabel('')
    ax4.set_title('mbRFP', color='m')
    [ax4.spines[edge].set_color('m') for edge in ax4.spines.keys()]

    ax5.set_ylabel('')
    ax5.set_xlabel('')
    [ax5.spines[edge].set_color('m') for edge in ax5.spines.keys()]

    ax6.set_ylabel('')
    ax6.set_xlabel('Time (minutes)')
    [ax6.spines[edge].set_color('m') for edge in ax6.spines.keys()]
    
    ax7.set_ylabel('')
    ax7.set_xlabel('')
    ax7.set_title('AF647-dextran', color='c')
    [ax7.spines[edge].set_color('c') for edge in ax7.spines.keys()]

    ax8.set_ylabel('')
    ax8.set_xlabel('')
    [ax8.spines[edge].set_color('c') for edge in ax8.spines.keys()]

    ax9.set_ylabel('')
    ax9.set_xlabel('Time (minutes)')
    [ax9.spines[edge].set_color('c') for edge in ax9.spines.keys()]
  
    
    
    fig.suptitle('%s: %s (equatorial z: %s)'%(expt_cond, fname, equatorial_z), x=0.5, y=0.92)#, ha='right')
    
    handles, labels = fig.gca().get_legend_handles_labels()
    unique_label_handle = dict(zip(labels, handles))
    fig.legend(unique_label_handle.values(), unique_label_handle.keys(), 
               ncol=3, loc='center right', bbox_to_anchor=(0.9, 0.92))
    
    
    
    
    # plt.tight_layout()

    
    

if out_dir.exists():    
    multipdf(Path.joinpath(out_dir,'dif_z_mean_fluors_znorm.pdf'))
else:
    out_dir.mkdir()
    multipdf(Path.joinpath(out_dir,'dif_z_mean_fluors_znorm.pdf'))

plt.close('all')





## Plot the cytoplasmic fluorescence loss vs the non-contact fluorescence loss:
## For each timelapse:

for fname, df in timelapse_subset_w_znorms:
    
    expt_cond = df['Experimental_Condition'].iloc[0]
    equatorial_z = df['best_focus_z_num'].iloc[0]

    ## associate different z-labels with different colours:
    zlabs = [zlab for zlab,df in list(df.groupby('z_label'))]
    zlab_colors = dict([(zlab, sns.color_palette('tab10')[i]) for i,zlab in enumerate(zlabs)])  


    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(15,5), ncols=3, nrows=1)
    
    for zlab, zdf in list(df.groupby('z_label')):

        ax1.errorbar(x=unp.nominal_values(zdf['grnchan_fluor_cytoplasmic_unp_norm']), 
                     y=unp.nominal_values(zdf['grnchan_fluor_outside_contact_unp_norm']), 
                     xerr=unp.std_devs(zdf['grnchan_fluor_cytoplasmic_unp_norm']), 
                     yerr=unp.std_devs(zdf['grnchan_fluor_outside_contact_unp_norm']),
                     ls='', marker='.', alpha=0.6, ecolor=np.array(zlab_colors[zlab])*0.5, 
                     color=zlab_colors[zlab], label=zlab)
        
        ax2.errorbar(x=unp.nominal_values(zdf['redchan_fluor_cytoplasmic_unp_norm']), 
                     y=unp.nominal_values(zdf['redchan_fluor_outside_contact_unp_norm']), 
                     xerr=unp.std_devs(zdf['redchan_fluor_cytoplasmic_unp_norm']), 
                     yerr=unp.std_devs(zdf['redchan_fluor_outside_contact_unp_norm']),
                     ls='', marker='.', alpha=0.6, ecolor=np.array(zlab_colors[zlab])*0.5, 
                     color=zlab_colors[zlab], label=zlab)
        
        ax3.errorbar(x=unp.nominal_values(zdf['bluchan_fluor_cytoplasmic_unp_norm']), 
                     y=unp.nominal_values(zdf['bluchan_fluor_outside_contact_unp_norm']), 
                     xerr=unp.std_devs(zdf['bluchan_fluor_cytoplasmic_unp_norm']), 
                     yerr=unp.std_devs(zdf['bluchan_fluor_outside_contact_unp_norm']),
                     ls='', marker='.', alpha=0.6, ecolor=np.array(zlab_colors[zlab])*0.5, 
                     color=zlab_colors[zlab], label=zlab)


    # ax1.set_xlim(0.4, 1.2)
    # ax1.set_ylim(0.4, 1.2)
    ax1.set_ylabel('Non-contact Fluor. (w.r.t. z3)')
    ax1.set_xlabel('Cytoplasmic Fluor. (w.r.t. z3)')
    ax1.set_title('LifeAct-GFP', color='g')
    [ax1.spines[edge].set_color('g') for edge in ax1.spines.keys()]
    ax1.set_aspect('equal')
    # ax1.set_adjustable('box')

    # ax2.set_xlim(0.4, 1.2)
    # ax2.set_ylim(0.4, 1.2)
    ax2.set_ylabel('')
    ax2.set_xlabel('Cytoplasmic Fluor. (w.r.t. z3)')
    ax2.set_title('mbRFP', color='m')
    [ax2.spines[edge].set_color('m') for edge in ax2.spines.keys()]
    ax2.set_aspect('equal')
    # ax2.set_adjustable('box')

    # ax3.set_xlim(0.4, 1.2)
    # ax3.set_ylim(0.4, 1.2)
    ax3.set_ylabel('')
    ax3.set_xlabel('Cytoplasmic Fluor. (w.r.t. z3)')
    ax3.set_title('AF647-dextran', color='c')
    [ax3.spines[edge].set_color('c') for edge in ax3.spines.keys()]
    ax3.set_aspect('equal')
    # ax3.set_adjustable('box')


    ## We want the plots to be squares, so we will set the limits to be the same
    ## for both x and y axes:
    ax1_lim_ls = ax1.get_xlim() + ax1.get_ylim()
    ax2_lim_ls = ax2.get_xlim() + ax2.get_ylim()
    ax3_lim_ls = ax3.get_xlim() + ax3.get_ylim()
    
    ax1_xlims = (min(ax1_lim_ls), max(ax1_lim_ls))
    ax1_ylims = (min(ax1_lim_ls), max(ax1_lim_ls))
    ax2_xlims = (min(ax2_lim_ls), max(ax2_lim_ls))
    ax2_ylims = (min(ax2_lim_ls), max(ax2_lim_ls))
    ax3_xlims = (min(ax3_lim_ls), max(ax3_lim_ls))
    ax3_ylims = (min(ax3_lim_ls), max(ax3_lim_ls))

    ax1.plot(ax1_xlims, ax1_ylims, ls='--', c='k', zorder=9)
    ax2.plot(ax2_xlims, ax2_ylims, ls='--', c='k', zorder=9)
    ax3.plot(ax3_xlims, ax3_ylims, ls='--', c='k', zorder=9)


    fig.suptitle('%s: %s (equatorial z: %s)'%(expt_cond, fname, equatorial_z), x=0.5, y=0.98)#, ha='right')
    
    handles, labels = fig.gca().get_legend_handles_labels()
    unique_label_handle = dict(zip(labels, handles))
    fig.legend(unique_label_handle.values(), unique_label_handle.keys(), 
               ncol=3, loc='center right', bbox_to_anchor=(0.9, 0.97))




if out_dir.exists():    
    multipdf(Path.joinpath(out_dir,'cyto_vs_noncont_mean_fluors_znorm.pdf'))
else:
    out_dir.mkdir()
    multipdf(Path.joinpath(out_dir,'cyto_vs_noncont_mean_fluors_znorm.pdf'))

plt.close('all')



## For the data according to Experimental_Condtion:
timelapse_subset_w_znorms_df = pd.concat([df for fname, df in timelapse_subset_w_znorms])
    
for name, df in timelapse_subset_w_znorms_df.groupby('Experimental_Condition'):

    expt_cond = df['Experimental_Condition'].iloc[0]
    equatorial_z = df['best_focus_z_num'].iloc[0]

    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(15,5), ncols=3, nrows=1)
    
    for zlab, zdf in list(df.groupby('z_label')):
    
        ax1.errorbar(x=unp.nominal_values(zdf['grnchan_fluor_cytoplasmic_unp_norm']), 
                     y=unp.nominal_values(zdf['grnchan_fluor_outside_contact_unp_norm']), 
                     xerr=unp.std_devs(zdf['grnchan_fluor_cytoplasmic_unp_norm']), 
                     yerr=unp.std_devs(zdf['grnchan_fluor_outside_contact_unp_norm']),
                     ls='', marker='.', alpha=0.3, ecolor=np.array(zlab_colors[zlab])*0.5, 
                     color=zlab_colors[zlab], label=zlab)
        
        ax2.errorbar(x=unp.nominal_values(zdf['redchan_fluor_cytoplasmic_unp_norm']), 
                     y=unp.nominal_values(zdf['redchan_fluor_outside_contact_unp_norm']), 
                     xerr=unp.std_devs(zdf['redchan_fluor_cytoplasmic_unp_norm']), 
                     yerr=unp.std_devs(zdf['redchan_fluor_outside_contact_unp_norm']),
                     ls='', marker='.', alpha=0.3, ecolor=np.array(zlab_colors[zlab])*0.5, 
                     color=zlab_colors[zlab], label=zlab)
        
        ax3.errorbar(x=unp.nominal_values(zdf['bluchan_fluor_cytoplasmic_unp_norm']), 
                     y=unp.nominal_values(zdf['bluchan_fluor_outside_contact_unp_norm']), 
                     xerr=unp.std_devs(zdf['bluchan_fluor_cytoplasmic_unp_norm']), 
                     yerr=unp.std_devs(zdf['bluchan_fluor_outside_contact_unp_norm']),
                     ls='', marker='.', alpha=0.3, ecolor=np.array(zlab_colors[zlab])*0.5, 
                     color=zlab_colors[zlab], label=zlab)
    
    
    # ax1.set_xlim(0.4, 1.2)
    # ax1.set_ylim(0.4, 1.2)
    ax1.set_ylabel('Non-contact Fluor. (w.r.t. z3)')
    ax1.set_xlabel('Cytoplasmic Fluor. (w.r.t. z3)')
    ax1.set_title('LifeAct-GFP', color='g')
    [ax1.spines[edge].set_color('g') for edge in ax1.spines.keys()]
    ax1.set_aspect('equal')
    # ax1.set_adjustable('box')
    
    # ax2.set_xlim(0.4, 1.2)
    # ax2.set_ylim(0.4, 1.2)
    ax2.set_ylabel('')
    ax2.set_xlabel('Cytoplasmic Fluor. (w.r.t. z3)')
    ax2.set_title('mbRFP', color='m')
    [ax2.spines[edge].set_color('m') for edge in ax2.spines.keys()]
    ax2.set_aspect('equal')
    # ax2.set_adjustable('box')
    
    # ax3.set_xlim(0.4, 1.2)
    # ax3.set_ylim(0.4, 1.2)
    ax3.set_ylabel('')
    ax3.set_xlabel('Cytoplasmic Fluor. (w.r.t. z3)')
    ax3.set_title('AF647-dextran', color='c')
    [ax3.spines[edge].set_color('c') for edge in ax3.spines.keys()]
    ax3.set_aspect('equal')
    # ax3.set_adjustable('box')
    
    
    ## We want the plots to be squares, so we will set the limits to be the same
    ## for both x and y axes:
    ax1_lim_ls = ax1.get_xlim() + ax1.get_ylim()
    ax2_lim_ls = ax2.get_xlim() + ax2.get_ylim()
    ax3_lim_ls = ax3.get_xlim() + ax3.get_ylim()
    
    ax1_xlims = (min(ax1_lim_ls), max(ax1_lim_ls))
    ax1_ylims = (min(ax1_lim_ls), max(ax1_lim_ls))
    ax2_xlims = (min(ax2_lim_ls), max(ax2_lim_ls))
    ax2_ylims = (min(ax2_lim_ls), max(ax2_lim_ls))
    ax3_xlims = (min(ax3_lim_ls), max(ax3_lim_ls))
    ax3_ylims = (min(ax3_lim_ls), max(ax3_lim_ls))
    
    ax1.plot(ax1_xlims, ax1_ylims, ls='--', c='k', zorder=9)
    ax2.plot(ax2_xlims, ax2_ylims, ls='--', c='k', zorder=9)
    ax3.plot(ax3_xlims, ax3_ylims, ls='--', c='k', zorder=9)
    
    
    
    fig.suptitle('%s: all fluorescence values normalized to z3'%expt_cond, x=0.5, y=0.98)#, ha='right')
    
    handles, labels = fig.gca().get_legend_handles_labels()
    unique_label_handle = dict(zip(labels, handles))
    fig.legend(unique_label_handle.values(), unique_label_handle.keys(), 
               ncol=3, loc='center right', bbox_to_anchor=(0.9, 0.97))
    




if out_dir.exists():    
    multipdf(Path.joinpath(out_dir,'cyto_vs_noncont_mean_fluors_znorm_groupby_expt.pdf'))
else:
    out_dir.mkdir()
    multipdf(Path.joinpath(out_dir,'cyto_vs_noncont_mean_fluors_znorm_groupby_expt.pdf'))

plt.close('all')



## For the data collectively:
fig, (ax1, ax2, ax3) = plt.subplots(figsize=(15,5), ncols=3, nrows=1)

for zlab, zdf in list(timelapse_subset_w_znorms_df.groupby('z_label')):

    ax1.errorbar(x=unp.nominal_values(zdf['grnchan_fluor_cytoplasmic_unp_norm']), 
                 y=unp.nominal_values(zdf['grnchan_fluor_outside_contact_unp_norm']), 
                 xerr=unp.std_devs(zdf['grnchan_fluor_cytoplasmic_unp_norm']), 
                 yerr=unp.std_devs(zdf['grnchan_fluor_outside_contact_unp_norm']),
                 ls='', marker='.', alpha=0.3, ecolor=np.array(zlab_colors[zlab])*0.5, 
                 color=zlab_colors[zlab], label=zlab)
    
    ax2.errorbar(x=unp.nominal_values(zdf['redchan_fluor_cytoplasmic_unp_norm']), 
                 y=unp.nominal_values(zdf['redchan_fluor_outside_contact_unp_norm']), 
                 xerr=unp.std_devs(zdf['redchan_fluor_cytoplasmic_unp_norm']), 
                 yerr=unp.std_devs(zdf['redchan_fluor_outside_contact_unp_norm']),
                 ls='', marker='.', alpha=0.3, ecolor=np.array(zlab_colors[zlab])*0.5, 
                 color=zlab_colors[zlab], label=zlab)
    
    ax3.errorbar(x=unp.nominal_values(zdf['bluchan_fluor_cytoplasmic_unp_norm']), 
                 y=unp.nominal_values(zdf['bluchan_fluor_outside_contact_unp_norm']), 
                 xerr=unp.std_devs(zdf['bluchan_fluor_cytoplasmic_unp_norm']), 
                 yerr=unp.std_devs(zdf['bluchan_fluor_outside_contact_unp_norm']),
                 ls='', marker='.', alpha=0.3, ecolor=np.array(zlab_colors[zlab])*0.5, 
                 color=zlab_colors[zlab], label=zlab)


# ax1.set_xlim(0.4, 1.2)
# ax1.set_ylim(0.4, 1.2)
ax1.set_ylabel('Non-contact Fluor. (w.r.t. z3)')
ax1.set_xlabel('Cytoplasmic Fluor. (w.r.t. z3)')
ax1.set_title('LifeAct-GFP', color='g')
[ax1.spines[edge].set_color('g') for edge in ax1.spines.keys()]
ax1.set_aspect('equal')
# ax1.set_adjustable('box')

# ax2.set_xlim(0.4, 1.2)
# ax2.set_ylim(0.4, 1.2)
ax2.set_ylabel('')
ax2.set_xlabel('Cytoplasmic Fluor. (w.r.t. z3)')
ax2.set_title('mbRFP', color='m')
[ax2.spines[edge].set_color('m') for edge in ax2.spines.keys()]
ax2.set_aspect('equal')
# ax2.set_adjustable('box')

# ax3.set_xlim(0.4, 1.2)
# ax3.set_ylim(0.4, 1.2)
ax3.set_ylabel('')
ax3.set_xlabel('Cytoplasmic Fluor. (w.r.t. z3)')
ax3.set_title('AF647-dextran', color='c')
[ax3.spines[edge].set_color('c') for edge in ax3.spines.keys()]
ax3.set_aspect('equal')
# ax3.set_adjustable('box')


## We want the plots to be squares, so we will set the limits to be the same
## for both x and y axes:
ax1_lim_ls = ax1.get_xlim() + ax1.get_ylim()
ax2_lim_ls = ax2.get_xlim() + ax2.get_ylim()
ax3_lim_ls = ax3.get_xlim() + ax3.get_ylim()

ax1_xlims = (min(ax1_lim_ls), max(ax1_lim_ls))
ax1_ylims = (min(ax1_lim_ls), max(ax1_lim_ls))
ax2_xlims = (min(ax2_lim_ls), max(ax2_lim_ls))
ax2_ylims = (min(ax2_lim_ls), max(ax2_lim_ls))
ax3_xlims = (min(ax3_lim_ls), max(ax3_lim_ls))
ax3_ylims = (min(ax3_lim_ls), max(ax3_lim_ls))

ax1.plot(ax1_xlims, ax1_ylims, ls='--', c='k', zorder=9)
ax2.plot(ax2_xlims, ax2_ylims, ls='--', c='k', zorder=9)
ax3.plot(ax3_xlims, ax3_ylims, ls='--', c='k', zorder=9)



fig.suptitle('All experimental conditions: all fluorescence values normalized to z3', x=0.5, y=0.98)#, ha='right')

handles, labels = fig.gca().get_legend_handles_labels()
unique_label_handle = dict(zip(labels, handles))
fig.legend(unique_label_handle.values(), unique_label_handle.keys(), 
           ncol=3, loc='center right', bbox_to_anchor=(0.9, 0.97))


fig.savefig(Path.joinpath(out_dir, 'cyto_vs_noncont_mean_fluors_znorm_all.tif'), dpi=dpi_graph)

plt.close(fig)











###############################################################################
###############################################################################
###############################################################################





# exit()
#### Below is a WIP:

def V_cap(h, a):
    vol = (1/6) * np.pi * h * (3 * a**2 + h**2)
    return vol

def V_seg(h, a, b):
    vol = (1/6) * np.pi * h * (3 * a**2 + 3 * b**2 + h**2)
    return vol




R = 12.5 #um

h = np.linspace(0, 2*R, 501)

a = np.sqrt(h * (2 * R - h))


h_lg = R 
h_sm = R - np.linspace(0, R, 501) # it's the difference between h_lg and some distance between a_lg and a_sm
a_sm = np.sqrt(h_sm * (2 * R - h_sm))
a_lg = R

vol_cap = V_cap(h, a)



## Spherical segment = bigger_spherical_cap_volume - smaller_spherical_cap_volume
vol_cap_lg = V_cap(h_lg, a_lg)
vol_cap_sm = V_cap(h_sm, a_sm)

vol_seg = vol_cap_lg - vol_cap_sm



# plt.plot(h, vol_cap)
plt.plot(h_lg - h_sm, vol_seg)

plt.close('all')

## We can see from this plot that the volume of a sphere that we have to get 
## through to reach the upper surface of the sphere after crossing its equator,
## if coming from below the sphere, increases approximately linearly at first.
## To reach the bottom half of the spheres surface we obviously don't need to
## get through any of the spheres volume. 




## DONE: Plot how the top half of the cell has its volume change
## bottom half of cell cortex is not affect by cytoplasmic diffusion of signal,
## but top half is. 
## Cytoplasmic fluorescence is lost due to both the distance between plane and
## objective as well as scattering from cytoplasm, while cortical fluorescence
## is only affected by fluorescence loss from distance between plane and 
## objective. <- for the bottom half.
## Difference between cytoplasmic and cortical fluorescence loss in bottom half 
## shows the rate of fluorescence loss due to cytoplasmic material.
## In upper half of cell fluorescence loss in cortex and cytoplasm should occur
## at same rate. Would explain the "upwards shifted line with same slope"
## observed in the graphs. 




## Now we have a column with the equatorial z, so we will fit a line to the 
## z-normalized fluorescence data for (z3, z2) and for (z2, z1), and compare 
## those linear regressions to a slope of 1:

## If the equatorial z is:
## z3 -> then all other z-planes are above the equator, so both cytoplasmic and
##          cortical fluorescence should change in the same way for (z3,z2) and
##          for (z2, z1)
## z2 -> then z1 is above the equator, so the line between (z2, z1) should have
##          a slope of 1, and (z3, z2) may or may not have a different slope,
##          depending on if cortical fluorescence changes differently or not
## z1 -> then all other z-planes are below the equator, and both (z3, z2) and
##          (z2, z1) may or may not have a different slope, depending on if 
##          cortical fluorescence changes differently or not



## LETS JUST SAY... THEORETICALLY SPEAKING... THAT THE SLOPES ARE DIFFERENT
## FROM 1... HOW WOULD YOU USE THIS INFORMATION TO DETERMINE FLUORESCENCE LOSS
## FROM CYTOPLASMIC DIFFUSION OF LIGHT, AND HOW WOULD THIS HELP YOU CORRECT 
## YOUR FLOURESCENCE DATA?

## If z3 is equatorial, then we can only use (z3), so we learn nothing. 
## If z2 is equatorial, then we can use (z3, z2)
## if z1 is equatorial, then we can include (z3, z2, z1)

## so basically, we have to filter our data, and then try a linear regression?

## Let's say that we have a reference slope y = x, and the data shows a slope 
## y = 0.8x, what does this then say about the data? How is this related to the
## fact that the z stepsize = 2um?

## After filtering data: could plot the z positions (in um) on the x axis and 
## the z-normalized fluorescences on the y axes. Then compare the rates of loss
## between cortical and cytoplasmic populations?

## the slope for cortical fluorescence represents the rate of instensity loss
## from just imaging depth...

## the slope for cytoplasmic fluorescence represents the rate of intensity loss
## from both imaging depth and cytoplasmic diffusion...

## the difference between these 2 represents the rate of intensity loss from 
## cytoplasmic diffusion


## TODO: clean up the commented code below here -----vvv-----


## Subset the data so we're only looking at z slices that are physically lower
## than or equal to the cells equator:
filtered_timelapse_subsets = [(fname, df.query('z_label >= best_focus_z_num')) for fname, df in timelapse_subset_w_znorms]

## Go through the different timelapses and...:
for fname, df in filtered_timelapse_subsets:

    expt_cond = df['Experimental_Condition'].iloc[0]
    equatorial_z = df['best_focus_z_num'].iloc[0]

    ## associate different z-labels with different colours:
    zlabs = [zlab for zlab,df in list(df.groupby('z_label'))]
    zlab_colors = dict([(zlab, sns.color_palette('tab10')[i]) for i,zlab in enumerate(zlabs)])  


    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(15,5), ncols=3, nrows=1)
    
    for zlab, zdf in list(df.groupby('z_label')):

    ## catplot or stripplot would probably be better here:
        ax1.errorbar(x=unp.nominal_values(zdf['rel_z_pos']), 
                     y=unp.nominal_values(zdf['grnchan_fluor_cytoplasmic_unp_norm']), 
                     yerr=unp.std_devs(zdf['grnchan_fluor_cytoplasmic_unp_norm']),
                     ls='', marker='.', alpha=0.6, ecolor='grey', 
                     color='k', label='cytoplasmic')
        ax1.errorbar(x=unp.nominal_values(zdf['rel_z_pos']), 
                     y=unp.nominal_values(zdf['grnchan_fluor_outside_contact_unp_norm']), 
                     yerr=unp.std_devs(zdf['grnchan_fluor_outside_contact_unp_norm']),
                     ls='', marker='.', alpha=0.6, ecolor='lime', 
                     color='g', label='cortical')
    
        
        ax2.errorbar(x=unp.nominal_values(zdf['rel_z_pos']), 
                     y=unp.nominal_values(zdf['redchan_fluor_cytoplasmic_unp_norm']), 
                     yerr=unp.std_devs(zdf['redchan_fluor_cytoplasmic_unp_norm']),
                     ls='', marker='.', alpha=0.6, ecolor='grey', 
                     color='k', label='cytoplasmic')
        ax2.errorbar(x=unp.nominal_values(zdf['rel_z_pos']), 
                     y=unp.nominal_values(zdf['redchan_fluor_outside_contact_unp_norm']), 
                     yerr=unp.std_devs(zdf['redchan_fluor_outside_contact_unp_norm']),
                     ls='', marker='.', alpha=0.6, ecolor='m', 
                     color='r', label='cortical')
    
        
        ax3.errorbar(x=unp.nominal_values(zdf['rel_z_pos']), 
                     y=unp.nominal_values(zdf['bluchan_fluor_cytoplasmic_unp_norm']), 
                     yerr=unp.std_devs(zdf['bluchan_fluor_cytoplasmic_unp_norm']),
                     ls='', marker='.', alpha=0.6, ecolor='grey', 
                     color='k', label='cytoplasmic')
        ax3.errorbar(x=unp.nominal_values(zdf['rel_z_pos']), 
                     y=unp.nominal_values(zdf['bluchan_fluor_outside_contact_unp_norm']), 
                     yerr=unp.std_devs(zdf['bluchan_fluor_outside_contact_unp_norm']),
                     ls='', marker='.', alpha=0.6, ecolor='c', 
                     color='b', label='cortical')


    # ax1.set_xlim(0.4, 1.2)
    # ax1.set_ylim(0.4, 1.2)
    ax1.set_ylabel('Fluor. (w.r.t. z3)')
    ax1.set_xlabel('z position')
    ax1.set_title('LifeAct-GFP', color='g')
    [ax1.spines[edge].set_color('g') for edge in ax1.spines.keys()]
    # ax1.set_aspect('equal')
    # ax1.set_adjustable('box')

    # ax2.set_xlim(0.4, 1.2)
    # ax2.set_ylim(0.4, 1.2)
    ax2.set_ylabel('')
    ax2.set_xlabel('z position')
    ax2.set_title('mbRFP', color='m')
    [ax2.spines[edge].set_color('m') for edge in ax2.spines.keys()]
    # ax2.set_aspect('equal')
    # ax2.set_adjustable('box')

    # ax3.set_xlim(0.4, 1.2)
    # ax3.set_ylim(0.4, 1.2)
    ax3.set_ylabel('')
    ax3.set_xlabel('z position')
    ax3.set_title('AF647-dextran', color='c')
    [ax3.spines[edge].set_color('c') for edge in ax3.spines.keys()]
    # ax3.set_aspect('equal')
    # ax3.set_adjustable('box')


    # ## We want the plots to be squares, so we will set the limits to be the same
    # ## for both x and y axes:
    # ax1_lim_ls = ax1.get_xlim() + ax1.get_ylim()
    # ax2_lim_ls = ax2.get_xlim() + ax2.get_ylim()
    # ax3_lim_ls = ax3.get_xlim() + ax3.get_ylim()
    
    # ax1_xlims = (min(ax1_lim_ls), max(ax1_lim_ls))
    # ax1_ylims = (min(ax1_lim_ls), max(ax1_lim_ls))
    # ax2_xlims = (min(ax2_lim_ls), max(ax2_lim_ls))
    # ax2_ylims = (min(ax2_lim_ls), max(ax2_lim_ls))
    # ax3_xlims = (min(ax3_lim_ls), max(ax3_lim_ls))
    # ax3_ylims = (min(ax3_lim_ls), max(ax3_lim_ls))

    # ax1.plot(ax1_xlims, ax1_ylims, ls='--', c='k', zorder=9)
    # ax2.plot(ax2_xlims, ax2_ylims, ls='--', c='k', zorder=9)
    # ax3.plot(ax3_xlims, ax3_ylims, ls='--', c='k', zorder=9)


    fig.suptitle('%s: %s (equatorial z: %s)'%(expt_cond, fname, equatorial_z), x=0.5, y=0.98)#, ha='right')
    
    handles, labels = fig.gca().get_legend_handles_labels()
    unique_label_handle = dict(zip(labels, handles))
    fig.legend(unique_label_handle.values(), unique_label_handle.keys(), 
               ncol=2, loc='center right', bbox_to_anchor=(0.9, 0.97))




if out_dir.exists():    
    multipdf(Path.joinpath(out_dir,'fluors_znorm_vs_zpos.pdf'))
else:
    out_dir.mkdir()
    multipdf(Path.joinpath(out_dir,'fluors_znorm_vs_zpos.pdf'))

plt.close('all')



filtered_timelapse_subsets_df = pd.concat([df for fname, df in filtered_timelapse_subsets])

# filtered_timelapse_subsets_df['Contact Surface Length (um)'] = filtered_timelapse_subsets_df['Contact Surface Length (px)'] * um_per_px

## These non-contact fluorescences are normalized to the bottom z position:
filtered_timelapse_subsets_df['grnchan_fluor_outside_contact_mean_norm'] = unp.nominal_values(filtered_timelapse_subsets_df['grnchan_fluor_outside_contact_unp_norm'])
filtered_timelapse_subsets_df['redchan_fluor_outside_contact_mean_norm'] = unp.nominal_values(filtered_timelapse_subsets_df['redchan_fluor_outside_contact_unp_norm'])
filtered_timelapse_subsets_df['bluchan_fluor_outside_contact_mean_norm'] = unp.nominal_values(filtered_timelapse_subsets_df['bluchan_fluor_outside_contact_unp_norm'])

filtered_timelapse_subsets_df['grnchan_fluor_cytoplasmic_mean_norm'] = unp.nominal_values(filtered_timelapse_subsets_df['grnchan_fluor_cytoplasmic_unp_norm'])
filtered_timelapse_subsets_df['redchan_fluor_cytoplasmic_mean_norm'] = unp.nominal_values(filtered_timelapse_subsets_df['redchan_fluor_cytoplasmic_unp_norm'])
filtered_timelapse_subsets_df['bluchan_fluor_cytoplasmic_mean_norm'] = unp.nominal_values(filtered_timelapse_subsets_df['bluchan_fluor_cytoplasmic_unp_norm'])


# grnchan_df = filtered_timelapse_subsets_df.melt(id_vars=['rel_z_pos'], 
#                                                 value_vars=['grnchan_fluor_outside_contact_mean_norm', 'grnchan_fluor_cytoplasmic_mean_norm'], 
#                                                 var_name='fluor_loc', value_name='fluor_wrt_z3')
# redchan_df = filtered_timelapse_subsets_df.melt(id_vars=['rel_z_pos'], 
#                                                 value_vars=['redchan_fluor_outside_contact_mean_norm', 'redchan_fluor_cytoplasmic_mean_norm'], 
#                                                 var_name='fluor_loc', value_name='fluor_wrt_z3')
# bluchan_df = filtered_timelapse_subsets_df.melt(id_vars=['rel_z_pos'], 
#                                                 value_vars=['bluchan_fluor_outside_contact_mean_norm', 'bluchan_fluor_cytoplasmic_mean_norm'], 
#                                                 var_name='fluor_loc', value_name='fluor_wrt_z3')


# expt_cond = df['Experimental_Condition'].iloc[0]
# equatorial_z = df['best_focus_z_num'].iloc[0]

# ## associate different z-labels with different colours:
# zlabs = [zlab for zlab,df in list(df.groupby('z_label'))]
# zlab_colors = dict([(zlab, sns.color_palette('tab10')[i]) for i,zlab in enumerate(zlabs)])  


fig, (ax1, ax2, ax3) = plt.subplots(figsize=(15,5), ncols=3, nrows=1)

for zlab, zdf in list(filtered_timelapse_subsets_df.groupby('z_label')):

## catplot or stripplot would probably be better here:
    ax1.errorbar(x=unp.nominal_values(zdf['rel_z_pos']), 
                 y=unp.nominal_values(zdf['grnchan_fluor_cytoplasmic_unp_norm']), 
                 yerr=unp.std_devs(zdf['grnchan_fluor_cytoplasmic_unp_norm']),
                 ls='', marker='.', alpha=0.6, ecolor='grey', 
                 color='k', label='cytoplasmic')
    ax1.errorbar(x=unp.nominal_values(zdf['rel_z_pos']), 
                 y=unp.nominal_values(zdf['grnchan_fluor_outside_contact_unp_norm']), 
                 yerr=unp.std_devs(zdf['grnchan_fluor_outside_contact_unp_norm']),
                 ls='', marker='.', alpha=0.6, ecolor='lime', 
                 color='g', label='cortical')

    
    ax2.errorbar(x=unp.nominal_values(zdf['rel_z_pos']), 
                 y=unp.nominal_values(zdf['redchan_fluor_cytoplasmic_unp_norm']), 
                 yerr=unp.std_devs(zdf['redchan_fluor_cytoplasmic_unp_norm']),
                 ls='', marker='.', alpha=0.6, ecolor='grey', 
                 color='k', label='cytoplasmic')
    ax2.errorbar(x=unp.nominal_values(zdf['rel_z_pos']), 
                 y=unp.nominal_values(zdf['redchan_fluor_outside_contact_unp_norm']), 
                 yerr=unp.std_devs(zdf['redchan_fluor_outside_contact_unp_norm']),
                 ls='', marker='.', alpha=0.6, ecolor='m', 
                 color='r', label='cortical')

    
    ax3.errorbar(x=unp.nominal_values(zdf['rel_z_pos']), 
                 y=unp.nominal_values(zdf['bluchan_fluor_cytoplasmic_unp_norm']), 
                 yerr=unp.std_devs(zdf['bluchan_fluor_cytoplasmic_unp_norm']),
                 ls='', marker='.', alpha=0.6, ecolor='grey', 
                 color='k', label='cytoplasmic')
    ax3.errorbar(x=unp.nominal_values(zdf['rel_z_pos']), 
                 y=unp.nominal_values(zdf['bluchan_fluor_outside_contact_unp_norm']), 
                 yerr=unp.std_devs(zdf['bluchan_fluor_outside_contact_unp_norm']),
                 ls='', marker='.', alpha=0.6, ecolor='c', 
                 color='b', label='cortical')


# ax1.set_xlim(0.4, 1.2)
# ax1.set_ylim(0.4, 1.2)
ax1.set_ylabel('Fluor. (w.r.t. z3)')
ax1.set_xlabel('z position')
ax1.set_title('LifeAct-GFP', color='g')
[ax1.spines[edge].set_color('g') for edge in ax1.spines.keys()]
# ax1.set_aspect('equal')
# ax1.set_adjustable('box')

# ax2.set_xlim(0.4, 1.2)
# ax2.set_ylim(0.4, 1.2)
ax2.set_ylabel('')
ax2.set_xlabel('z position')
ax2.set_title('mbRFP', color='m')
[ax2.spines[edge].set_color('m') for edge in ax2.spines.keys()]
# ax2.set_aspect('equal')
# ax2.set_adjustable('box')

# ax3.set_xlim(0.4, 1.2)
# ax3.set_ylim(0.4, 1.2)
ax3.set_ylabel('')
ax3.set_xlabel('z position')
ax3.set_title('AF647-dextran', color='c')
[ax3.spines[edge].set_color('c') for edge in ax3.spines.keys()]
# ax3.set_aspect('equal')
# ax3.set_adjustable('box')


# ## We want the plots to be squares, so we will set the limits to be the same
# ## for both x and y axes:
# ax1_lim_ls = ax1.get_xlim() + ax1.get_ylim()
# ax2_lim_ls = ax2.get_xlim() + ax2.get_ylim()
# ax3_lim_ls = ax3.get_xlim() + ax3.get_ylim()

# ax1_xlims = (min(ax1_lim_ls), max(ax1_lim_ls))
# ax1_ylims = (min(ax1_lim_ls), max(ax1_lim_ls))
# ax2_xlims = (min(ax2_lim_ls), max(ax2_lim_ls))
# ax2_ylims = (min(ax2_lim_ls), max(ax2_lim_ls))
# ax3_xlims = (min(ax3_lim_ls), max(ax3_lim_ls))
# ax3_ylims = (min(ax3_lim_ls), max(ax3_lim_ls))

# ax1.plot(ax1_xlims, ax1_ylims, ls='--', c='k', zorder=9)
# ax2.plot(ax2_xlims, ax2_ylims, ls='--', c='k', zorder=9)
# ax3.plot(ax3_xlims, ax3_ylims, ls='--', c='k', zorder=9)


fig.suptitle('%s: %s (equatorial z: %s)'%(expt_cond, fname, equatorial_z), x=0.5, y=0.98)#, ha='right')

handles, labels = fig.gca().get_legend_handles_labels()
unique_label_handle = dict(zip(labels, handles))
fig.legend(unique_label_handle.values(), unique_label_handle.keys(), 
           ncol=2, loc='center right', bbox_to_anchor=(0.9, 0.97))


## Some savefig command here.


plt.close('all')











## Clean up the mean fluorescences of nans so that linregress can work:

grnchan_cytopl_df = filtered_timelapse_subsets_df.dropna(axis=0, subset=['grnchan_fluor_cytoplasmic_mean_norm'])
grnchan_cortic_df = filtered_timelapse_subsets_df.dropna(axis=0, subset=['grnchan_fluor_outside_contact_mean_norm'])

redchan_cytopl_df = filtered_timelapse_subsets_df.dropna(axis=0, subset=['redchan_fluor_cytoplasmic_mean_norm'])
redchan_cortic_df = filtered_timelapse_subsets_df.dropna(axis=0, subset=['redchan_fluor_outside_contact_mean_norm'])

bluchan_cytopl_df = filtered_timelapse_subsets_df.dropna(axis=0, subset=['bluchan_fluor_cytoplasmic_mean_norm'])
bluchan_cortic_df = filtered_timelapse_subsets_df.dropna(axis=0, subset=['bluchan_fluor_outside_contact_mean_norm'])


grnchan_cytopl_linreg = stats.linregress(grnchan_cytopl_df['rel_z_pos'], grnchan_cytopl_df['grnchan_fluor_cytoplasmic_mean_norm'])
grnchan_cortic_linreg = stats.linregress(grnchan_cortic_df['rel_z_pos'], grnchan_cortic_df['grnchan_fluor_outside_contact_mean_norm'])

redchan_cytopl_linreg = stats.linregress(redchan_cytopl_df['rel_z_pos'], redchan_cytopl_df['redchan_fluor_cytoplasmic_mean_norm'])
redchan_cortic_linreg = stats.linregress(redchan_cortic_df['rel_z_pos'], redchan_cortic_df['redchan_fluor_outside_contact_mean_norm'])

bluchan_cytopl_linreg = stats.linregress(bluchan_cytopl_df['rel_z_pos'], bluchan_cytopl_df['bluchan_fluor_cytoplasmic_mean_norm'])
bluchan_cortic_linreg = stats.linregress(bluchan_cortic_df['rel_z_pos'], bluchan_cortic_df['bluchan_fluor_outside_contact_mean_norm'])


print(grnchan_cytopl_linreg.slope, redchan_cytopl_linreg.slope, bluchan_cytopl_linreg.slope)
print(grnchan_cortic_linreg.slope, redchan_cortic_linreg.slope, bluchan_cortic_linreg.slope)

## so it looks like lifeact is lost at a rate of -0.0398 (3.98%) per um as the z plane
## moves further away purely due to focal depth (cortical line), and then the 
## cytoplasmic portion shows a loss of around -0.0823 (8.23%) per um.
## The difference between the two is:
grn_slope_diff = grnchan_cytopl_linreg.slope - grnchan_cortic_linreg.slope
print('LifeAct-GFP cell bulk-dependent rate of change:', grn_slope_diff)
## Roughly a 4% loss per um? Oof. 

## The other channels:
red_slope_diff = redchan_cytopl_linreg.slope - redchan_cortic_linreg.slope
print('mbRFP cell bulk-dependent rate of change:', red_slope_diff)

blu_slope_diff = bluchan_cytopl_linreg.slope - bluchan_cortic_linreg.slope
print('AlexaFluor647 cell bulk-dependent rate of change:', blu_slope_diff)





## Make some plots:




fig, (ax1, ax2, ax3) = plt.subplots(figsize=(15,5), ncols=3, nrows=1)


# sns.violinplot(x='rel_z_pos', y='fluor_wrt_z3', hue='fluor_loc', data=grnchan_df, split=False, ax=ax1)
# sns.violinplot(x='rel_z_pos', y='fluor_wrt_z3', hue='fluor_loc', data=redchan_df, split=False, ax=ax2)
# sns.violinplot(x='rel_z_pos', y='fluor_wrt_z3', hue='fluor_loc', data=bluchan_df, split=False, ax=ax3)

## Ideally would use violinplot or something together with the linear regressions
## but when a category plot is used then the x axis is categorical and its numbers
## correspond to 1 integer for each category, so line and violin do not line up

sns.scatterplot(x=grnchan_cytopl_df['rel_z_pos'], y=grnchan_cytopl_df['grnchan_fluor_cytoplasmic_mean_norm'], ax=ax1)
sns.scatterplot(x=grnchan_cortic_df['rel_z_pos'], y=grnchan_cortic_df['grnchan_fluor_outside_contact_mean_norm'], ax=ax1)
sns.scatterplot(x=redchan_cytopl_df['rel_z_pos'], y=redchan_cytopl_df['redchan_fluor_cytoplasmic_mean_norm'], ax=ax2)
sns.scatterplot(x=redchan_cortic_df['rel_z_pos'], y=redchan_cortic_df['redchan_fluor_outside_contact_mean_norm'], ax=ax2)
sns.scatterplot(x=bluchan_cytopl_df['rel_z_pos'], y=bluchan_cytopl_df['bluchan_fluor_cytoplasmic_mean_norm'], ax=ax3)
sns.scatterplot(x=bluchan_cortic_df['rel_z_pos'], y=bluchan_cortic_df['bluchan_fluor_outside_contact_mean_norm'], ax=ax3)

ax1.plot([0, 4], [grnchan_cytopl_linreg.intercept, line(4, grnchan_cytopl_linreg.slope, 0, grnchan_cytopl_linreg.intercept)], c='k', ls='--', zorder=10)
ax1.plot([0, 4], [grnchan_cortic_linreg.intercept, line(4, grnchan_cortic_linreg.slope, 0, grnchan_cortic_linreg.intercept)], c='k', ls='--')

ax2.plot([0, 4], [redchan_cytopl_linreg.intercept, line(4, redchan_cytopl_linreg.slope, 0, redchan_cytopl_linreg.intercept)], c='k', ls='--', zorder=10)
ax2.plot([0, 4], [redchan_cortic_linreg.intercept, line(4, redchan_cortic_linreg.slope, 0, redchan_cortic_linreg.intercept)], c='k', ls='--')

ax3.plot([0, 4], [bluchan_cytopl_linreg.intercept, line(4, bluchan_cytopl_linreg.slope, 0, bluchan_cytopl_linreg.intercept)], c='k', ls='--', zorder=10)
ax3.plot([0, 4], [bluchan_cortic_linreg.intercept, line(4, bluchan_cortic_linreg.slope, 0, bluchan_cortic_linreg.intercept)], c='k', ls='--')


# fig, (ax1, ax2, ax3) = plt.subplots(figsize=(15,5), ncols=3, nrows=1)

# sns.lineplot(x='rel_z_pos', y='fluor_wrt_z3', hue='fluor_loc', data=grnchan_df, ax=ax1)
# sns.lineplot(x='rel_z_pos', y='fluor_wrt_z3', hue='fluor_loc', data=redchan_df, ax=ax2)
# sns.lineplot(x='rel_z_pos', y='fluor_wrt_z3', hue='fluor_loc', data=bluchan_df, ax=ax3)

ax1.axhline(1, c='k')
ax2.axhline(1, c='k')
ax3.axhline(1,c ='k')

# ax1.set_ylim(0, 1.05)
# ax2.set_ylim(0, 1.05)
# ax3.set_ylim(0, 1.05)

ax1.set_ylim(0)
ax2.set_ylim(0)
ax3.set_ylim(0)


# plt.close('all')














## Now you need to find a way to correct the fluorescence measurements with 
## your ~4.25%/um loss in LifeAct fluorescence from cytoplasm-induced diffusion

## ...   ...   ....... hmm...









# sns.lmplot(x='rel_z_pos', y='fluor_wrt_z3', hue='fluor_loc', data=grnchan_df)














    
    
# I guess also do this for the data collectively


# At some point we will also need to estimate how cytoplasm must be penetrated 
# for a given z slice that was quantified. 
# This will probably involve fitting a circle to the 6 contact corners (3 pairs;
# one pair for each z slice).


    



## Could try pandas.explode to extract all of the normalized fluorescence info...?
## maybe not... because what about the added uncertainty from normalization?

# sns.pointplot()
    








# Each timepoint in the simulation is associated with a theoretical 
# contact angle and therefore a theoretical contact diameter which was then 
# drawn as pixels and measured. The values in the table for angle and contact
# diameter are how that theoretical angle appears under our imaging conditions.
# Therefore we will make a list of the theoretical angles and contact diameters
theor_ang = np.concatenate(([0,0], np.linspace(0,90,91)))
theor_contnd = expected_spreading_sphere_params(theor_ang, 1)[1][4]
theor_relten = angle2reltension(theor_ang)

# # Similarly for the LifeAct simulations each timepoint has a theoretical angle:
# theor_ang_LA = np.concatenate(([0,0], np.linspace(0,90,91)))
# theor_contnd_LA = expected_spreading_sphere_params(theor_ang_LA, 1)[1][4]
# theor_relten_LA = angle2reltension(theor_ang_LA)

# We want to return the expected maximum contact diameter
expect_max_contnd = expected_spreading_sphere_params(90, 1)[1][4]








# data_cols = [['grnchan_fluor_at_contact_mean', 'grnchan_fluor_at_contact_sem'], 
#              ['grnchan_fluor_outside_contact_mean', 'grnchan_fluor_outside_contact_sem'], 
#              ['grnchan_fluor_cytoplasmic_mean', 'grnchan_fluor_cytoplasmic_sem'], 
#              ['redchan_fluor_at_contact_mean', 'redchan_fluor_at_contact_sem'], 
#              ['redchan_fluor_outside_contact_mean', 'redchan_fluor_outside_contact_sem'], 
#              ['redchan_fluor_cytoplasmic_mean', 'redchan_fluor_cytoplasmic_sem'], 
#              ['bluchan_fluor_at_contact_mean', 'bluchan_fluor_at_contact_sem'], 
#              ['bluchan_fluor_outside_contact_mean', 'bluchan_fluor_outside_contact_sem'], 
#              ['bluchan_fluor_cytoplasmic_mean', 'bluchan_fluor_cytoplasmic_sem']]

# norm_fluor_cols = {'lifeact_fluor_normed_mean': grn_slope_diff, 
#                    'lifeact_fluor_normed_sem': grn_slope_diff, 
#                    'mbrfp_fluor_normed_mean': red_slope_diff, 
#                    'mbrfp_fluor_normed_sem': red_slope_diff, 
#                    'af647dx_fluor_normed_mean': blu_slope_diff, 
#                    'af647dx_fluor_normed_sem': blu_slope_diff, 
#                    }

fluor_cols_to_corr = {'lifeact_fluor_normed_sub_mean': grn_slope_diff, 
                      'lifeact_fluor_normed_sub_sem': grn_slope_diff, 
                      'mbrfp_fluor_normed_sub_mean': red_slope_diff, 
                      'mbrfp_fluor_normed_sub_sem': red_slope_diff, 
                      'af647dx_fluor_normed_mean': blu_slope_diff, 
                      'af647dx_fluor_normed_sem': blu_slope_diff, 
                       }





## Use the difference between slopes of cytoplasmic and cortical fluorescence 
## loss to correct the fluorescence values measured at the contact:
for col_name in fluor_cols_to_corr:
    
    corr_col_name = col_name+'_corr'

    timelapse_subset_w_znorms_df[corr_col_name] = timelapse_subset_w_znorms_df.apply(lambda x: fluor_correction(x[col_name], x['Contact Surface Length (um)']/2, fluor_cols_to_corr[col_name]), axis=1)



### TODO: decide if sub-zero values should be kept or set to 0:
## And also mask any corrected fluorescence intensity data that is negative?
## In such a case it should be 0... 
# corrected_mean_fluor_cols = ['lifeact_fluor_normed_sub_mean_corr', 
#                               'mbrfp_fluor_normed_sub_mean_corr', 
#                               'af647dx_fluor_normed_mean_corr', 
#                               'lifeact_fluor_normed_sub_sem_corr', 
#                               'mbrfp_fluor_normed_sub_sem_corr',
#                               'af647dx_fluor_normed_sem_corr']
# for col_name in corrected_mean_fluor_cols:
#     # filtered_timelapse_subsets_df[col_name] = filtered_timelapse_subsets_df.mask(filtered_timelapse_subsets_df[col_name] < 0, other=0)[col_name]
#     timelapse_subset_w_znorms_df[col_name] = timelapse_subset_w_znorms_df.mask(timelapse_subset_w_znorms_df[col_name] < 0, other=0)[col_name]





# for expt_cond_name, df in filtered_timelapse_subsets_df.groupby('Experimental_Condition'):
for expt_cond_name, df in timelapse_subset_w_znorms_df.groupby('Experimental_Condition'):

    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(15,5), ncols=3, nrows=1)
    
    ax1.scatter(x=df['Contact Surface Length (N.D.)'], y=df['lifeact_fluor_normed_mean'], s=2, marker='.', c='tab:green', alpha=1, label='uncorrected')
    # ax1.scatter(x=df['Contact Surface Length (N.D.)'], y=df['lifeact_fluor_normed_mean_corr'], s=2, marker='.', c='k', alpha=1, label='corrected')
    ax1.scatter(x=df['Contact Surface Length (N.D.)'], y=df['lifeact_fluor_normed_sub_mean_corr'], s=2, marker='.', c='k', alpha=1, label='corrected')
    ax1.plot(theor_contnd, theor_relten, c='k', ls='--')    
    ax1.axhline(0, c='k', ls='-')

    ax2.scatter(x=df['Contact Surface Length (N.D.)'], y=df['mbrfp_fluor_normed_mean'], s=2, marker='.', c='tab:red', alpha=1, label='uncorrected')
    # ax2.scatter(x=df['Contact Surface Length (N.D.)'], y=df['mbrfp_fluor_normed_mean_corr'], s=2, marker='.', c='k', alpha=1, label='corrected')
    ax2.scatter(x=df['Contact Surface Length (N.D.)'], y=df['mbrfp_fluor_normed_sub_mean_corr'], s=2, marker='.', c='k', alpha=1, label='corrected')
    ax2.plot(theor_contnd, theor_relten, c='k', ls='--')    
    ax2.axhline(1, c='k', ls='--')
    ax2.axhline(0, c='k', ls='-')
    
    ## We don't use the subbed version of AlexaFluor647-dextran because the 
    ## signal is actually cytoplasmic.
    ax3.scatter(x=df['Contact Surface Length (N.D.)'], y=df['af647dx_fluor_normed_mean'], s=2, marker='.', c='tab:blue', alpha=1, label='uncorrected')
    ax3.scatter(x=df['Contact Surface Length (N.D.)'], y=df['af647dx_fluor_normed_mean_corr'], s=2, marker='.', c='k', alpha=1, label='corrected')
    ax3.axhline(1, c='k', ls='--')
    ax3.axhline(0, c='k', ls='-')

    ## Note that some data goes beyond these y-limits
    # ax1.set_ylim(-0.05, 2.05)
    ax1.set_title('LifeAct')
    
    # ax2.set_ylim(-0.05, 2.05)
    ax2.set_title('mbRFP')

    # ax3.set_ylim(-0.05, 2.05)
    ax3.set_title('AF647-dextran')    
    ## Note that some data goes beyond these y-limits

    
    handles, labels = fig.gca().get_legend_handles_labels()
    unique_label_handle = dict(zip(labels, handles))
    fig.legend(unique_label_handle.values(), unique_label_handle.keys(), 
               ncol=2)#, loc='center right')#, bbox_to_anchor=(0.9, 0.92))
    
    
    fig.suptitle(expt_cond_name)

    fig.savefig(Path.joinpath(out_dir, 'corrected_contnd_vs_fluor_%s.tif'%expt_cond_name), dpi=dpi_graph)
    
    plt.close(fig)






for expt_cond_name, df in timelapse_subset_w_znorms_df.groupby('Experimental_Condition'):

    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(15,5), ncols=3, nrows=1)
       
    # ax1.scatter(x=df['Contact Surface Length (N.D.)'], y=df['lifeact_fluor_normed_mean_corr'], s=2, marker='.', c='tab:green', alpha=1)#, label='corrected')
    # ax1.errorbar(x=df['Contact Surface Length (N.D.)'], y=df['lifeact_fluor_normed_mean_corr'], yerr=df['lifeact_fluor_normed_sem_corr'], ms=2, marker='.', lw=0, c='tab:green', elinewidth=0.5, ecolor='grey', alpha=1)#, label='corrected')
    ax1.errorbar(x=df['Contact Surface Length (N.D.)'], y=df['lifeact_fluor_normed_sub_mean_corr'], yerr=df['lifeact_fluor_normed_sub_sem_corr'], ms=2, marker='.', lw=0, c='tab:green', elinewidth=0.5, ecolor='grey', alpha=1)#, label='corrected')
    ax1.plot(theor_contnd, theor_relten, c='k', ls='--')    
    ax1.axhline(0, c='k', ls='-')

    # ax2.scatter(x=df['Contact Surface Length (N.D.)'], y=df['mbrfp_fluor_normed_mean_corr'], s=2, marker='.', c='tab:red', alpha=1)#, label='corrected')
    # ax2.errorbar(x=df['Contact Surface Length (N.D.)'], y=df['mbrfp_fluor_normed_mean_corr'], yerr=df['mbrfp_fluor_normed_sem_corr'], ms=2, marker='.', lw=0, c='tab:red', elinewidth=0.5, ecolor='grey', alpha=1)#, label='corrected')
    ax2.errorbar(x=df['Contact Surface Length (N.D.)'], y=df['mbrfp_fluor_normed_sub_mean_corr'], yerr=df['mbrfp_fluor_normed_sub_sem_corr'], ms=2, marker='.', lw=0, c='tab:red', elinewidth=0.5, ecolor='grey', alpha=1)#, label='corrected')
    ax2.plot(theor_contnd, theor_relten, c='k', ls='--')
    ax2.axhline(1, c='k', ls='--')
    ax2.axhline(0, c='k', ls='-')

    ## We don't use the subbed version of AlexaFluor647-dextran because the 
    ## signal is actually cytoplasmic.
    # ax3.scatter(x=df['Contact Surface Length (N.D.)'], y=df['af647dx_fluor_normed_mean_corr'], s=2, marker='.', c='tab:blue', alpha=1)#, label='corrected')
    ax3.errorbar(x=df['Contact Surface Length (N.D.)'], y=df['af647dx_fluor_normed_mean_corr'], yerr=df['af647dx_fluor_normed_sem_corr'], ms=2, marker='.', lw=0, c='tab:blue', elinewidth=0.5, ecolor='grey', alpha=1)#, label='corrected')
    ax3.axhline(1, c='k', ls='--')
    ax3.axhline(0, c='k', ls='-')


    ## Note that some data goes beyond these y-limits
    # ax1.set_ylim(-0.05, 2.05)
    ax1.set_title('LifeAct')
    
    # ax2.set_ylim(-0.05, 2.05)
    ax2.set_title('mbRFP')

    # ax3.set_ylim(-0.05, 2.05)
    ax3.set_title('AF647-dextran')    
    ## Note that some data goes beyond these y-limits

    
    # handles, labels = fig.gca().get_legend_handles_labels()
    # unique_label_handle = dict(zip(labels, handles))
    # fig.legend(unique_label_handle.values(), unique_label_handle.keys(), 
    #            ncol=2)#, loc='center right')#, bbox_to_anchor=(0.9, 0.92))
    
    
    fig.suptitle(expt_cond_name)

    fig.savefig(Path.joinpath(out_dir, 'corrected_contnd_vs_fluor_%s_2.tif'%expt_cond_name), dpi=dpi_graph)

    plt.close(fig)




# df = filtered_timelapse_subsets_df.groupby('Experimental_Condition').get_group('ee1xMBS')
# fig, ax = plt.subplots(figsize=(3,3))
# ax.errorbar(x=df['Contact Surface Length (N.D.)'].values, y=df['lifeact_fluor_normed_mean_corr'].values, yerr=df['lifeact_fluor_normed_sem_corr'].values, ms=2, marker='.', lw=0, c='tab:green', elinewidth=0.5, ecolor='grey')#, label='corrected')
# ax.set_ylabel('Corrected Normalized \n Contact Fluorescence (a.u.)')
# fig.savefig(Path.joinpath(out_dir, 'corrected_contnd_vs_fluor_ee1xMBS_lifeact.tif'), dpi=dpi_graph)








## For use in sct5: I manually checked the rough starting points where the 
## LifeAct signal seems to bottom out and put them in a table, so add that
## info the the DataFrame:
z_plat_start_times = pd.read_csv(Path.joinpath(data_dir, 'approx_lifeact_bottoms.csv'))

## Make a dictionary where each Source_File has a corresponding 
## lifeact_plat_start_time:
sf_zplat_dict = z_plat_start_times.set_index('Source_File').to_dict()['bottom_start_time']

## Map these start times onto a new column in the main DataFrame:
timelapse_subset_w_znorms_df['lifeact_plat_start_time'] = timelapse_subset_w_znorms_df['Source_File'].map(sf_zplat_dict)



## Also save the dataframe because I want to re-use it in the simulation script 
## (sct5_viscous_shell_model) to compare against models:
timelapse_subset_w_znorms_df.to_pickle(Path.joinpath(out_dir, 'zdepth_la_contnd_df.pkl'))












## Plot the corrected fluorescence values over time similar to what was done in
## sct4:

# Since we will be creating a multi-page pdf, make sure that no figures are 
# open before plotting the graphs we want to add to the pdf:
plt.close('all')

# Plot the non-dimensionalized contact length, contact angle, and Bc/B over 
# time for each timelapse separately and save into a single PDF:
for i,(sourcefile,df) in enumerate(timelapse_subset_w_znorms_df.groupby('Source_File')):
    fig, (ax1, ax2) = plt.subplots(figsize=(6,6), 
                                   num='#%d - '%i + sourcefile, 
                                   nrows=2, sharex=True)
    if (df['z_label'] == df['best_focus_z_num']).all():
        plt.suptitle('File #%d - '%i + sourcefile + ' (best focus)', verticalalignment='center', x=0.5, y=0.5, fontsize=12)#, fontweight='bold')
    else:
        plt.suptitle('File #%d - '%i + sourcefile, verticalalignment='center', x=0.5, y=0.5, fontsize=12)
    cont = ax1.plot(df['Time (minutes)'], df['Contact Surface Length (N.D.)'], c='k', ls=':', marker='.', label='Contact Surface Length (N.D.)')
    ax1.set_ylim(0, timelapse_subset_w_znorms_df['Contact Surface Length (N.D.)'].max()) #3)
    
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
    mean_la = df['lifeact_fluor_normed_sub_mean_corr'].astype(float)
    sem_la = df['lifeact_fluor_normed_sub_sem_corr'].astype(float)
    # mean_cytopl_la = df['lifeact_cytoplasmic_fluor_normed_mean']
    # sem_cytopl_la = df['lifeact_cytoplasmic_fluor_normed_sem']
    ax2.plot(df['Time (minutes)'], mean_la, c='g', marker='.', zorder=9, label='LifeAct')
    ax2.fill_between(df['Time (minutes)'], mean_la - sem_la, mean_la + sem_la, 
                     color='g', alpha=0.5, zorder=6)
    # ax2.errorbar(x=df['Time (minutes)'], y=mean_la, yerr=sem_la, c=
    # ax2.plot(df['Time (minutes)'], mean_cytopl_la, c='orange')
    # ax2.fill_between(df['Time (minutes)'], mean_cytopl_la - sem_cytopl_la, mean_cytopl_la + sem_cytopl_la, 
    #                  color='orange', alpha=0.5, zorder=3)
    
    mean_mb = df['mbrfp_fluor_normed_sub_mean_corr'].astype(float)
    sem_mb = df['mbrfp_fluor_normed_sub_sem_corr'].astype(float)
    # mean_cytopl_mb = df['mbrfp_cytoplasmic_fluor_normed_mean']
    # sem_cytopl_mb = df['mbrfp_cytoplasmic_fluor_normed_sem']
    ax2.plot(df['Time (minutes)'], mean_mb, c='r', marker='.', zorder=8, label='mbRFP')
    ax2.fill_between(df['Time (minutes)'], mean_mb - sem_mb, mean_mb + sem_mb, 
                     color='r', alpha=0.5, zorder=5)
    # ax2.plot(df['Time (minutes)'], mean_cytopl_mb, c='magenta')
    # ax2.fill_between(df['Time (minutes)'], mean_cytopl_mb - sem_cytopl_mb, mean_cytopl_mb + sem_cytopl_mb, 
    #                  color='magenta', alpha=0.5, zorder=2)
    
    mean_dx = df['af647dx_fluor_normed_mean_corr'].astype(float)
    sem_dx = df['af647dx_fluor_normed_sem_corr'].astype(float)
    # mean_cytopl_dx = df['af647dx_cytoplasmic_fluor_normed_mean']
    # sem_cytopl_dx = df['af647dx_cytoplasmic_fluor_normed_sem']
    ax2.plot(df['Time (minutes)'], mean_dx, c='b', marker='.', zorder=7, label='Dextran')
    ax2.fill_between(df['Time (minutes)'], mean_dx - sem_dx, mean_dx + sem_dx, 
                     color='b', alpha=0.5, zorder=4)
    # ax2.plot(df['Time (minutes)'], mean_cytopl_dx, c='cyan')
    # ax2.fill_between(df['Time (minutes)'], mean_cytopl_dx - sem_cytopl_dx, mean_cytopl_dx + sem_cytopl_dx, 
    #                  color='cyan', alpha=0.5, zorder=1)
    ###

    ax2.axhline(1, c='k', ls='--', zorder=10)
    ax2.set_ylim(0,2)
    xmin, xmax = -1, round_up_to_multiple(timelapse_subset_w_znorms_df['Time (minutes)'].max(), 5)
    ax2.set_xlim(xmin, xmax)
    ax2.set_ylabel('Corrected Norm. \nContact Fluor. (N.D.)')
    ax2.set_xlabel ('Time (minutes)')
    ax2.legend(loc='upper right')
    # ax2.plot(df['Time (minutes)'], df[], c='r', marker='.')
    
    # ax2b = ax2.twinx()
    # ax2b.plot(df['Time (minutes)'], lifeact_rel_fluor_norm[i], c='k', marker='.')
    # ax2b.spines['left'].set_color('red')
    # ax2.tick_params(axis='y', colors='green')
    
    
    #plt.show()
multipdf(Path.joinpath(out_dir, r"corr_ee_lifeact_contnd_angles.pdf"))
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
for sourcefile,df in timelapse_subset_w_znorms_df.groupby('Source_File'):
    if sourcefile in graphs_of_interest:
        fig, (ax1, ax2) = plt.subplots(figsize=(width_1col, width_1col), 
                                       # num='#%d - '%i + sourcefile, 
                                       nrows=2, sharex=True)
        # plt.suptitle('File #%d - '%i + sourcefile, verticalalignment='center', x=0.5, y=0.5, fontsize=14)
        cont = ax1.plot(df['Time (minutes)'], df['Contact Surface Length (N.D.)'], c='k', ls=':', marker='.', alpha=1, label='Diam. (N.D.)')
        ax1.set_ylim(0, timelapse_subset_w_znorms_df['Contact Surface Length (N.D.)'].max()) #3)
        
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
        mean_la = df['lifeact_fluor_normed_sub_mean_corr'].astype(float)
        sem_la = df['lifeact_fluor_normed_sub_sem_corr'].astype(float)
        # mean_cytopl_la = df['lifeact_cytoplasmic_fluor_normed_mean']
        # sem_cytopl_la = df['lifeact_cytoplasmic_fluor_normed_sem']
        ax2.plot(df['Time (minutes)'], mean_la, c='g', marker='.', alpha=0.5, zorder=9, label='LifeAct')
        ax2.fill_between(df['Time (minutes)'], mean_la - sem_la, mean_la + sem_la, 
                         color='g', alpha=0.25, zorder=6)
        # ax2.plot(df['Time (minutes)'], mean_cytopl_la, c='orange')
        # ax2.fill_between(df['Time (minutes)'], mean_cytopl_la - sem_cytopl_la, mean_cytopl_la + sem_cytopl_la, 
        #                  color='orange', alpha=0.5, zorder=3)
        
        mean_mb = df['mbrfp_fluor_normed_sub_mean_corr'].astype(float)
        sem_mb = df['mbrfp_fluor_normed_sub_sem_corr'].astype(float)
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
        # ax2.set_ylim(0,2)
        ax2.axhline(0, c='k')
        # xmin, xmax = -1, round_up_to_multiple(filtered_timelapse_subsets_df['Time (minutes)'].max(), 5)
        # ax2.set_xlim(xmin, xmax)
        ax2.set_ylabel('Corr. Norm. \nCont. Fluor. (N.D.)')
        ax2.set_xlabel ('Time (minutes)')
        ax2.legend(ncol=2, loc='upper right')
        # ax2.plot(df['Time (minutes)'], df[], c='r', marker='.')
        
        # ax2b = ax2.twinx()
        # ax2b.plot(df['Time (minutes)'], lifeact_rel_fluor_norm[i], c='k', marker='.')
        # ax2b.spines['left'].set_color('red')
        # ax2.tick_params(axis='y', colors='green')
        
        
        #plt.show()
        # fig.savefig(Path.joinpath(out_dir, r"ee_lifeact_contnd_angles_%d.tif"%i), dpi=dpi_graph, bbox_inches='tight')
        fig.savefig(Path.joinpath(out_dir, r"corr_%s_kinetics.tif"%sourcefile), dpi=dpi_graph, bbox_inches='tight')
        plt.close('all')








# data = la_contnd_df[la_contnd_df['Experimental_Condition'] == 'ee1xMBS']
data = timelapse_subset_w_znorms_df.query('Experimental_Condition == "ee1xMBS"')



### Plot the uncorrected LifeAct-GFP data altogether:
fig, ax = plt.subplots(figsize=(width_1col*0.7, width_1col*0.7)) # set up the figure. 
# data = la_contnd_df[la_contnd_df['Experimental_Condition'] == 'ee1xMBS']
sns.scatterplot(x='Average Angle (deg)', y='lifeact_fluor_normed_mean', hue='Source_File', 
                data=data, 
                s=3, linewidth=0, alpha=0.5, ax=ax, legend=False, zorder=2)
ax.errorbar(x=data['Average Angle (deg)'], y=data['lifeact_fluor_normed_mean'], 
            yerr=data['lifeact_fluor_normed_sem'], color='grey', marker='.', 
            ms=0, lw=0, elinewidth=0.5, ecolor='lightgrey', alpha=1, zorder=1)
ax.plot(theor_ang*2, theor_relten, c='k', ls='--', zorder=0) # the "*2" is because each cell in the cellpair has an angle associated with it's spherical cap geometry

# ax.axhline(z_plat_uncorr, c='grey', ls='-', zorder=0)
ax.axvline(180, c='grey', ls='-', zorder=0)
    
ax.set_ylim(0, 2)
ax.set_yticks(np.linspace(0, 2, 21), minor=True)
ax.set_xlim(0, 200)
ax.set_xticks([0, 50, 100, 150, 200], minor=True)
ax.set_xlabel('Contact Angle (deg)')
ax.set_ylabel('Norm. Contact Fluor. (N.D.)')
# save the resulting plot:
# fig.legend()
fig.savefig(Path.joinpath(out_dir, r'angle_vs_fluor_lagfp_uncorr.tif'), dpi=dpi_LineArt, bbox_inches='tight')
plt.close(fig)


fig, ax = plt.subplots(figsize=(width_1col*0.7, width_1col*0.7)) # set up the figure. 
# data = la_contnd_df[la_contnd_df['Experimental_Condition'] == 'ee1xMBS']
sns.scatterplot(x='Contact Surface Length (N.D.)', y='lifeact_fluor_normed_mean', hue='Source_File', 
                data=data, 
                s=3, linewidth=0, alpha=0.5, ax=ax, legend=False, zorder=2)
ax.errorbar(x=data['Contact Surface Length (N.D.)'], y=data['lifeact_fluor_normed_mean'], 
            yerr=data['lifeact_fluor_normed_sem'], color='grey', marker='.', 
            ms=0, lw=0, elinewidth=0.5, ecolor='lightgrey', alpha=1, zorder=1)
ax.plot(theor_contnd, theor_relten, c='k', ls='--', zorder=0)   

# ax.axhline(z_plat_uncorr, c='grey', ls='-', zorder=0)
ax.axvline(max(theor_contnd), c='grey', ls='-', zorder=0)
    
ax.set_ylim(0, 2)
ax.set_yticks(np.linspace(0, 2, 21), minor=True)
ax.set_xlim(0, 3.0)
ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], minor=True)
ax.set_xlabel('Contact Diameter (N.D.)')
ax.set_ylabel('Norm. Contact Fluor. (N.D.)')
# save the resulting plot:
# fig.legend()
fig.savefig(Path.joinpath(out_dir, r'contnd_vs_fluor_lagfp_uncorr.tif'), dpi=dpi_LineArt, bbox_inches='tight')
plt.close(fig)



### Plot the corrected LifeAct-GFP data altogether but with a limited Y-axis to
### so the trends aren't obscured by outliers:
fig, ax = plt.subplots(figsize=(width_1col*0.7, width_1col*0.7)) # set up the figure. 
# data = la_contnd_df[la_contnd_df['Experimental_Condition'] == 'ee1xMBS']
sns.scatterplot(x='Average Angle (deg)', y='lifeact_fluor_normed_sub_mean_corr', hue='Source_File', 
                data=data, 
                s=3, linewidth=0, alpha=0.5, ax=ax, legend=False, zorder=2)
ax.errorbar(x=data['Average Angle (deg)'], y=data['lifeact_fluor_normed_sub_mean_corr'], 
            yerr=data['lifeact_fluor_normed_sub_sem_corr'], color='grey', marker='.', 
            ms=0, lw=0, elinewidth=0.5, ecolor='lightgrey', alpha=1, zorder=1)
ax.plot(theor_ang*2, theor_relten, c='k', ls='--', zorder=0) # the "*2" is because each cell in the cellpair has an angle associated with it's spherical cap geometry

# ax.axhline(z_plat_corr, c='grey', ls='-', zorder=0)
ax.axvline(180, c='grey', ls='-', zorder=0)
    
ax.set_ylim(-0.2, 2)
ax.set_yticks(np.linspace(-0.2, 2, 23), minor=True)
ax.set_xlim(0, 200)
ax.set_xticks([0, 50, 100, 150, 200], minor=True)
ax.set_xlabel('Contact Angle (deg)')
ax.set_ylabel('Norm. Contact Fluor. (N.D.)')
# save the resulting plot:
# fig.legend()
fig.savefig(Path.joinpath(out_dir, r'angle_vs_fluor_lagfp_corr.tif'), dpi=dpi_LineArt, bbox_inches='tight')
plt.close(fig)


fig, ax = plt.subplots(figsize=(width_1col*0.7, width_1col*0.7)) # set up the figure. 
# data = la_contnd_df[la_contnd_df['Experimental_Condition'] == 'ee1xMBS']
sns.scatterplot(x='Contact Surface Length (N.D.)', y='lifeact_fluor_normed_sub_mean_corr', hue='Source_File', 
                data=data, 
                s=3, linewidth=0, alpha=0.5, ax=ax, legend=False, zorder=2)
ax.errorbar(x=data['Contact Surface Length (N.D.)'], y=data['lifeact_fluor_normed_sub_mean_corr'], 
            yerr=data['lifeact_fluor_normed_sub_sem_corr'], color='grey', marker='.', 
            ms=0, lw=0, elinewidth=0.5, ecolor='lightgrey', alpha=1, zorder=1)
ax.plot(theor_contnd, theor_relten, c='k', ls='--', zorder=0)   

# ax.axhline(z_plat_corr, c='grey', ls='-', zorder=0)
ax.axvline(max(theor_contnd), c='grey', ls='-', zorder=0)
    
ax.set_ylim(-0.2, 2)
ax.set_yticks(np.linspace(-0.2, 2, 23), minor=True)
ax.set_xlim(0, 3.0)
ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], minor=True)
ax.set_xlabel('Contact Diameter (N.D.)')
ax.set_ylabel('Norm. Contact Fluor. (N.D.)')
# save the resulting plot:
# fig.legend()
fig.savefig(Path.joinpath(out_dir, r'contnd_vs_fluor_lagfp_corr.tif'), dpi=dpi_LineArt, bbox_inches='tight')
plt.close(fig)




### now do the same plots as above but without limiting the Y-axis:
fig, ax = plt.subplots(figsize=(width_1col*0.7, width_1col*0.7)) # set up the figure. 
# data = la_contnd_df[la_contnd_df['Experimental_Condition'] == 'ee1xMBS']
sns.scatterplot(x='Average Angle (deg)', y='lifeact_fluor_normed_sub_mean_corr', hue='Source_File', 
                data=data, 
                s=3, linewidth=0, alpha=0.5, ax=ax, legend=False, zorder=2)
ax.errorbar(x=data['Average Angle (deg)'], y=data['lifeact_fluor_normed_sub_mean_corr'], 
            yerr=data['lifeact_fluor_normed_sub_sem_corr'], color='grey', marker='.', 
            ms=0, lw=0, elinewidth=0.5, ecolor='lightgrey', alpha=1, zorder=1)
ax.plot(theor_ang*2, theor_relten, c='k', ls='--', zorder=0) # the "*2" is because each cell in the cellpair has an angle associated with it's spherical cap geometry

# ax.axhline(z_plat_corr, c='grey', ls='-', zorder=0)
ax.axvline(180, c='grey', ls='-', zorder=0)
    
# ax.set_ylim(0, 2)
# ax.set_yticks(np.linspace(0, 2, 21), minor=True)
ax.set_xlim(0, 200)
ax.set_xticks([0, 50, 100, 150, 200], minor=True)
ax.set_xlabel('Contact Angle (deg)')
ax.set_ylabel('Norm. Contact Fluor. (N.D.)')
# save the resulting plot:
# fig.legend()
fig.savefig(Path.joinpath(out_dir, r'angle_vs_fluor_lagfp_corr_noYlim.tif'), dpi=dpi_LineArt, bbox_inches='tight')
plt.close(fig)


fig, ax = plt.subplots(figsize=(width_1col*0.7, width_1col*0.7)) # set up the figure. 
# data = la_contnd_df[la_contnd_df['Experimental_Condition'] == 'ee1xMBS']
sns.scatterplot(x='Contact Surface Length (N.D.)', y='lifeact_fluor_normed_sub_mean_corr', hue='Source_File', 
                data=data, 
                s=3, linewidth=0, alpha=0.5, ax=ax, legend=False, zorder=2)
ax.errorbar(x=data['Contact Surface Length (N.D.)'], y=data['lifeact_fluor_normed_sub_mean_corr'], 
            yerr=data['lifeact_fluor_normed_sub_sem_corr'], color='grey', marker='.', 
            ms=0, lw=0, elinewidth=0.5, ecolor='lightgrey', alpha=1, zorder=1)
ax.plot(theor_contnd, theor_relten, c='k', ls='--', zorder=0)   

# ax.axhline(z_plat_corr, c='grey', ls='-', zorder=0)
ax.axvline(max(theor_contnd), c='grey', ls='-', zorder=0)
    
# ax.set_ylim(0, 2)
# ax.set_yticks(np.linspace(0, 2, 21), minor=True)
ax.set_xlim(0, 3.0)
ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], minor=True)
ax.set_xlabel('Contact Diameter (N.D.)')
ax.set_ylabel('Norm. Contact Fluor. (N.D.)')
# save the resulting plot:
# fig.legend()
fig.savefig(Path.joinpath(out_dir, r'contnd_vs_fluor_lagfp_corr_noYlim.tif'), dpi=dpi_LineArt, bbox_inches='tight')
plt.close(fig)





### Now plot the mbRFP signal:
### Plot the uncorrected mbRFP data altogether:
fig, ax = plt.subplots(figsize=(width_1col*0.7, width_1col*0.7)) # set up the figure. 
# data = la_contnd_df[la_contnd_df['Experimental_Condition'] == 'ee1xMBS']
sns.scatterplot(x='Average Angle (deg)', y='mbrfp_fluor_normed_mean', hue='Source_File', 
                data=data, 
                s=3, linewidth=0, alpha=0.5, ax=ax, legend=False, zorder=2)
ax.errorbar(x=data['Average Angle (deg)'], y=data['mbrfp_fluor_normed_mean'], 
            yerr=data['mbrfp_fluor_normed_sem'], color='grey', marker='.', 
            ms=0, lw=0, elinewidth=0.5, ecolor='lightgrey', alpha=1, zorder=1)
ax.plot(theor_ang*2, theor_relten, c='k', ls='--', zorder=0) # the "*2" is because each cell in the cellpair has an angle associated with it's spherical cap geometry

# ax.axhline(z_plat_uncorr, c='grey', ls='-', zorder=0)
ax.axvline(180, c='grey', ls='-', zorder=0)
    
ax.set_ylim(0, 2)
ax.set_yticks(np.linspace(0, 2, 21), minor=True)
ax.set_xlim(0, 200)
ax.set_xticks([0, 50, 100, 150, 200], minor=True)
ax.set_xlabel('Contact Angle (deg)')
ax.set_ylabel('Norm. Contact Fluor. (N.D.)')
# save the resulting plot:
# fig.legend()
fig.savefig(Path.joinpath(out_dir, r'angle_vs_fluor_mbRFP_uncorr.tif'), dpi=dpi_LineArt, bbox_inches='tight')
plt.close(fig)


fig, ax = plt.subplots(figsize=(width_1col*0.7, width_1col*0.7)) # set up the figure. 
# data = la_contnd_df[la_contnd_df['Experimental_Condition'] == 'ee1xMBS']
sns.scatterplot(x='Contact Surface Length (N.D.)', y='mbrfp_fluor_normed_mean', hue='Source_File', 
                data=data, 
                s=3, linewidth=0, alpha=0.5, ax=ax, legend=False, zorder=2)
ax.errorbar(x=data['Contact Surface Length (N.D.)'], y=data['mbrfp_fluor_normed_mean'], 
            yerr=data['mbrfp_fluor_normed_sem'], color='grey', marker='.', 
            ms=0, lw=0, elinewidth=0.5, ecolor='lightgrey', alpha=1, zorder=1)
ax.plot(theor_contnd, theor_relten, c='k', ls='--', zorder=0)   

# ax.axhline(z_plat_uncorr, c='grey', ls='-', zorder=0)
ax.axvline(max(theor_contnd), c='grey', ls='-', zorder=0)
    
ax.set_ylim(0, 2)
ax.set_yticks(np.linspace(0, 2, 21), minor=True)
ax.set_xlim(0, 3.0)
ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], minor=True)
ax.set_xlabel('Contact Diameter (N.D.)')
ax.set_ylabel('Norm. Contact Fluor. (N.D.)')
# save the resulting plot:
# fig.legend()
fig.savefig(Path.joinpath(out_dir, r'contnd_vs_fluor_mbRFP_uncorr.tif'), dpi=dpi_LineArt, bbox_inches='tight')
plt.close(fig)



### Plot the corrected LifeAct-GFP data altogether but with a limited Y-axis to
### so the trends aren't obscured by outliers:
fig, ax = plt.subplots(figsize=(width_1col*0.7, width_1col*0.7)) # set up the figure. 
# data = la_contnd_df[la_contnd_df['Experimental_Condition'] == 'ee1xMBS']
sns.scatterplot(x='Average Angle (deg)', y='mbrfp_fluor_normed_sub_mean_corr', hue='Source_File', 
                data=data, 
                s=3, linewidth=0, alpha=0.5, ax=ax, legend=False, zorder=2)
ax.errorbar(x=data['Average Angle (deg)'], y=data['mbrfp_fluor_normed_sub_mean_corr'], 
            yerr=data['mbrfp_fluor_normed_sub_sem_corr'], color='grey', marker='.', 
            ms=0, lw=0, elinewidth=0.5, ecolor='lightgrey', alpha=1, zorder=1)
ax.plot(theor_ang*2, theor_relten, c='k', ls='--', zorder=0) # the "*2" is because each cell in the cellpair has an angle associated with it's spherical cap geometry

# ax.axhline(z_plat_corr, c='grey', ls='-', zorder=0)
ax.axvline(180, c='grey', ls='-', zorder=0)
    
ax.set_ylim(-0.2, 2)
ax.set_yticks(np.linspace(-0.2, 2, 23), minor=True)
ax.set_xlim(0, 200)
ax.set_xticks([0, 50, 100, 150, 200], minor=True)
ax.set_xlabel('Contact Angle (deg)')
ax.set_ylabel('Norm. Contact Fluor. (N.D.)')
# save the resulting plot:
# fig.legend()
fig.savefig(Path.joinpath(out_dir, r'angle_vs_fluor_mbRFP_corr.tif'), dpi=dpi_LineArt, bbox_inches='tight')
plt.close(fig)


fig, ax = plt.subplots(figsize=(width_1col*0.7, width_1col*0.7)) # set up the figure. 
# data = la_contnd_df[la_contnd_df['Experimental_Condition'] == 'ee1xMBS']
sns.scatterplot(x='Contact Surface Length (N.D.)', y='mbrfp_fluor_normed_sub_mean_corr', hue='Source_File', 
                data=data, 
                s=3, linewidth=0, alpha=0.5, ax=ax, legend=False, zorder=2)
ax.errorbar(x=data['Contact Surface Length (N.D.)'], y=data['mbrfp_fluor_normed_sub_mean_corr'], 
            yerr=data['mbrfp_fluor_normed_sub_sem_corr'], color='grey', marker='.', 
            ms=0, lw=0, elinewidth=0.5, ecolor='lightgrey', alpha=1, zorder=1)
ax.plot(theor_contnd, theor_relten, c='k', ls='--', zorder=0)   

# ax.axhline(z_plat_corr, c='grey', ls='-', zorder=0)
ax.axvline(max(theor_contnd), c='grey', ls='-', zorder=0)
    
ax.set_ylim(-0.2, 2)
ax.set_yticks(np.linspace(-0.2, 2, 23), minor=True)
ax.set_xlim(0, 3.0)
ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], minor=True)
ax.set_xlabel('Contact Diameter (N.D.)')
ax.set_ylabel('Norm. Contact Fluor. (N.D.)')
# save the resulting plot:
# fig.legend()
fig.savefig(Path.joinpath(out_dir, r'contnd_vs_fluor_mbRFP_corr.tif'), dpi=dpi_LineArt, bbox_inches='tight')
plt.close(fig)




### now do the same plots as above but without limiting the Y-axis:
fig, ax = plt.subplots(figsize=(width_1col*0.7, width_1col*0.7)) # set up the figure. 
# data = la_contnd_df[la_contnd_df['Experimental_Condition'] == 'ee1xMBS']
sns.scatterplot(x='Average Angle (deg)', y='mbrfp_fluor_normed_sub_mean_corr', hue='Source_File', 
                data=data, 
                s=3, linewidth=0, alpha=0.5, ax=ax, legend=False, zorder=2)
ax.errorbar(x=data['Average Angle (deg)'], y=data['mbrfp_fluor_normed_sub_mean_corr'], 
            yerr=data['mbrfp_fluor_normed_sub_sem_corr'], color='grey', marker='.', 
            ms=0, lw=0, elinewidth=0.5, ecolor='lightgrey', alpha=1, zorder=1)
ax.plot(theor_ang*2, theor_relten, c='k', ls='--', zorder=0) # the "*2" is because each cell in the cellpair has an angle associated with it's spherical cap geometry

# ax.axhline(z_plat_corr, c='grey', ls='-', zorder=0)
ax.axvline(180, c='grey', ls='-', zorder=0)
    
# ax.set_ylim(0, 2)
# ax.set_yticks(np.linspace(0, 2, 21), minor=True)
ax.set_xlim(0, 200)
ax.set_xticks([0, 50, 100, 150, 200], minor=True)
ax.set_xlabel('Contact Angle (deg)')
ax.set_ylabel('Norm. Contact Fluor. (N.D.)')
# save the resulting plot:
# fig.legend()
fig.savefig(Path.joinpath(out_dir, r'angle_vs_fluor_mbRFP_corr_noYlim.tif'), dpi=dpi_LineArt, bbox_inches='tight')
plt.close(fig)


fig, ax = plt.subplots(figsize=(width_1col*0.7, width_1col*0.7)) # set up the figure. 
# data = la_contnd_df[la_contnd_df['Experimental_Condition'] == 'ee1xMBS']
sns.scatterplot(x='Contact Surface Length (N.D.)', y='mbrfp_fluor_normed_sub_mean_corr', hue='Source_File', 
                data=data, 
                s=3, linewidth=0, alpha=0.5, ax=ax, legend=False, zorder=2)
ax.errorbar(x=data['Contact Surface Length (N.D.)'], y=data['mbrfp_fluor_normed_sub_mean_corr'], 
            yerr=data['mbrfp_fluor_normed_sub_sem_corr'], color='grey', marker='.', 
            ms=0, lw=0, elinewidth=0.5, ecolor='lightgrey', alpha=1, zorder=1)
ax.plot(theor_contnd, theor_relten, c='k', ls='--', zorder=0)   

# ax.axhline(z_plat_corr, c='grey', ls='-', zorder=0)
ax.axvline(max(theor_contnd), c='grey', ls='-', zorder=0)
    
# ax.set_ylim(0, 2)
# ax.set_yticks(np.linspace(0, 2, 21), minor=True)
ax.set_xlim(0, 3.0)
ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], minor=True)
ax.set_xlabel('Contact Diameter (N.D.)')
ax.set_ylabel('Norm. Contact Fluor. (N.D.)')
# save the resulting plot:
# fig.legend()
fig.savefig(Path.joinpath(out_dir, r'contnd_vs_fluor_mbRFP_corr_noYlim.tif'), dpi=dpi_LineArt, bbox_inches='tight')
plt.close(fig)




## 
## Some Rough Stuff:
## 
# exit()
# for zlab, zdf in timelapse_subset_w_znorms_df.groupby('z_label'):
#     print(zlab)
#     # break

# # sns.scatterplot(x='Contact Surface Length (N.D.)', y='grnchan_fluor_cytoplasmic_mean', hue='z_label', data=timelapse_subset_w_znorms_df, legend=False, marker='.', size=2)

# data = timelapse_subset_w_znorms_df[timelapse_subset_w_znorms_df.z_label == timelapse_subset_w_znorms_df.best_focus_z_num]
# data = data.query("Experimental_Condition == 'ee1xMBS'")
# data = data.dropna(axis=0, subset=['grnchan_fluor_at_contact_mean'])
# data['grnchan_fluor_at_contact_mean_tnorm'] = data.groupby('Source_File').apply(lambda x: x['grnchan_fluor_at_contact_mean'] / x.iloc[0]['grnchan_fluor_at_contact_mean']).reset_index()['grnchan_fluor_at_contact_mean']
# data['grnchan_fluor_at_contact_mean_maxnorm'] = data.groupby('Source_File').apply(lambda x: x['grnchan_fluor_at_contact_mean'] / x['grnchan_fluor_at_contact_mean'].max()).reset_index()['grnchan_fluor_at_contact_mean']

# data['grnchan_fluor_cytoplasmic_mean_tnorm'] = data.groupby('Source_File').apply(lambda x: x['grnchan_fluor_cytoplasmic_mean'] / x.iloc[0]['grnchan_fluor_cytoplasmic_mean']).reset_index()['grnchan_fluor_cytoplasmic_mean']


# fig, ax = plt.subplots()
# # sns.scatterplot(x='Average Angle (deg)', y='grnchan_fluor_at_contact_mean_tnorm', hue='Source_File', data=data, legend=False, marker='o', s=10)
# sns.scatterplot(x='Contact Surface Length (um)', 
#                 y='grnchan_fluor_at_contact_mean_tnorm', 
#                 hue='Source_File', 
#                 data=data, legend=False, marker='o', s=10, ax=ax)
# ax.set_xlim(0,)




# for sf, df in data.groupby('Source_File'):
#     fig, ax = plt.subplots()
#     # sns.scatterplot(x='Average Angle (deg)', y='grnchan_fluor_at_contact_mean_tnorm', hue='Source_File', data=data, legend=False, marker='o', s=10)
#     sns.scatterplot(x='Contact Surface Length (um)', 
#                     # y='grnchan_fluor_at_contact_mean_tnorm', 
#                     y='grnchan_fluor_at_contact_mean', 
#                     data=df, legend=False, marker='o', s=10, ax=ax)
#     ax.set_xlim(0,data['Contact Surface Length (um)'].max())
#     plt.show()



# fig, ax = plt.subplots()
# sns.scatterplot(x='Contact Surface Length (um)', 
#                 y='grnchan_fluor_cytoplasmic_mean_tnorm', 
#                 hue='Source_File', 
#                 data=data, legend=False, marker='o', s=10, ax=ax)
# ax.set_xlim(0,)

