# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 12:49:49 2018

@author: Serge Parent
"""


### To Do:
#

import warnings
#import os
#import glob
from pathlib import Path
import csv

from collections import namedtuple #this is used only for creating a F_onewayResult that has nans

import numpy as np
import pandas as pd

from scipy import stats
from scipy import signal
import scipy.fftpack
from scipy.optimize import curve_fit
from scipy.optimize import OptimizeWarning
import scipy.interpolate as interpolate

#from skimage import morphology

import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator
import matplotlib.colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_pdf import PdfPages

# Silence the warning for opening too many figures:
plt.rcParams.update({'figure.max_open_warning': 0})
# Silence the SettingWithCopy warning from Pandas:
pd.options.mode.chained_assignment = None
# scipy.optimize.curve_fit throws an OptimizeWarning if the data can't be fitted
# well with the proposed function, but this causes the script to stop, so silence
# this warning and I will then look into what data didn't have a fit applied:
warnings.simplefilter("error", OptimizeWarning)
# the logistic functions are throwing "RuntimeWarning: overflow in exp" when
# in the user-defined logistic-function. If a number is so large as to throw
# an error over that then it may as well be infinite for biological purposes,
# therefore the warning is going to be ignored:
np.seterr(over='ignore')
# Silence warnings because a deprecation warning is flooding the console:
warnings.filterwarnings("error")




# Get the working directory of the script:
current_script_dir = Path(__file__).parent.absolute()

# Data can be found in the following directory:
data_dir = Path.joinpath(current_script_dir, r"data")

# After running this program once I checked the validity of the sigmoidal fits
# to the data for each cell-cell adhesion timelapse. They are kept in the
# following folder:
fit_validity_dir = Path.joinpath(current_script_dir, r"man_chkd_fit_tables")

# I also made a subset of data where only data with cells that were initially
# separate and then initiated contact had a sigmoid fit to them, and these fits
# were also checked. The tables are found in the following directory:
grow_to_low_fit_valid_dir = Path.joinpath(current_script_dir, r"man_chkd_low_grow_fit_tables")

# Output files and figures to the following directory; if it doesn't exist
# then make it:
output_dir = Path.joinpath(current_script_dir, r"adhesion_analysis_figs_out")
if output_dir.exists():
    pass
else:
    Path.mkdir(output_dir)



# Need a function to convert these to inches since matplotlib doesn't play well
# with metric:
def cm2inch(num):
    return num/2.54



# Define a function to find the order of magnitude of a number:
def order_of_mag(number):
    order_of_magnitude = np.floor(np.log10(number))
    return order_of_magnitude



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



def num_contig_true(list_or_1Darray, mask=None):
    """This function returns the number of consecutive TRUE statements in a 
    list or 1Darray as a list. If there are no TRUE statements then a list 
    containing the number 0 is returned instead of an empty list, since an
    empty list was throwing errors when using certain functions. """
    arr = np.ma.masked_array(data=list_or_1Darray, mask=mask)
    consecutive_true = []
    count = 0
    for x in arr:
        if x == True:
            count += 1
        elif x == False and count == 0:
            pass
        elif x == False and count != 0:
            consecutive_true.append(count)
            count = 0
    if count > 0:
        consecutive_true.append(count)
    if np.ma.any(arr):
        first_element_in_series = np.ma.where(arr)[0][0]
    else:
        first_element_in_series = np.float('nan')
    # If there are no TRUE statement in the input and consecutive_true is empty
    # then append a 0 to the list and be done with it:
    if not consecutive_true:
        consecutive_true.append(0)
    return consecutive_true, first_element_in_series



def time_correct_data(list_of_df_dropna, indiv_logistic_angle_fits_df, indiv_logistic_fit_validity_manual_check):
    """ 
    Required inputs: \n
    list_of_df_dropna = a list of Pandas.DataFrames (all with the same column headers) 
    where each dataframe is the data from a timelapse. \n
    indiv_logistic_angle_fits_df = a single dataframe containing the
    optimal fit parameters used for each logistic fit to each timelapse. \n
    indiv_logistic_fit_validity_manual_check = a dataframe containing true/false 
    values for the whether or not the low plateau, growth phase, and high plateau 
    parameters of each logistic fit seemed to reasonably represent the data they 
    were being fitted to. \n
    Outputs: \n
    time_corrected_df_with_valid_growth_phases = a dataframe containing all timelapses 
    with valid growth phases where the time has been redefined such that 
    time = 0 is the point where growth is fastest according to each individual 
    logistic fit. 
    df_without_valid_growth_phases = a dataframe containing all timelapses without 
    a valid growth phase. time = 0 is typically the first timepoint in the timelapse.
    """
    time_corrected_list_of_df_with_valid_growth_phases = [x.copy() for x in list_of_df_dropna]
    list_of_df_without_valid_growth_phases = list()
    for i,x in enumerate(time_corrected_list_of_df_with_valid_growth_phases):
        try:
            x['Time (minutes)'] = x['Time (minutes)'].apply(lambda y: y - indiv_logistic_angle_fits_df.iloc[i]['popt_x2'])
            x['Time (minutes)'] = x['Time (minutes)'].apply(lambda y: round(y * 2) / 2)
        except ValueError:
            x['Time (minutes)'] = x['Time (minutes)'].apply(lambda y: y * 1)
            
    # Separate the timelapses that can be time corrected from the ones that
    # can't be time corrected (ie. those with a valid growth phase from those
    # without a valid growth phase):
    for x in indiv_logistic_fit_validity_manual_check.query('growth_phase == False').index[::-1]:
        list_of_df_without_valid_growth_phases.append(time_corrected_list_of_df_with_valid_growth_phases.pop(x))
    
    # Collect the lists into a single dataframe:
    try:
        time_corrected_df_with_valid_growth_phases = pd.concat(time_corrected_list_of_df_with_valid_growth_phases)
    except(ValueError):
        time_corrected_df_with_valid_growth_phases = pd.DataFrame(columns=time_corrected_list_of_df_with_valid_growth_phases[0].columns)
    try:
        df_without_valid_growth_phases = pd.concat(list_of_df_without_valid_growth_phases)
    except(ValueError):
        df_without_valid_growth_phases = pd.DataFrame(np.nan, index=[0], columns=time_corrected_list_of_df_with_valid_growth_phases[0].columns)
        df_without_valid_growth_phases['Time (minutes)'] = 0
    
    return time_corrected_df_with_valid_growth_phases, df_without_valid_growth_phases



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



###############################################################################
### The following code is for fitting logistical curves to the ecto-ecto and
### meso-meso data:
### The cell-cell adhesion appears to follow a generalized logistic curve for
### ecto-ecto and meso-meso adhesions:
# k = growth rate
# nu controls where on the x-axis the point of inflection is:
def logistic(x, x0, k, upper_plateau, lower_plateau):
    y = lower_plateau + ( (upper_plateau - lower_plateau) / (1 + np.exp(-k*(x-x0))) )
    return y

def logistic_diffeq(x, x0, k, upper_plateau, lower_plateau):
    with np.errstate(invalid='ignore'):
    # y = lower_plateau + ( (upper_plateau - lower_plateau) / (1 + np.exp(-k*(x-x0))) )
        dy_dx = k * (upper_plateau - lower_plateau) * (np.exp(-k*(x-x0))) / (1 + np.exp(-k*(x-x0)))**2
    return dy_dx

# Instead try a general logistic:
def general_logistic(x, x0, k, upper_plateau, lower_plateau, nu):
    with np.errstate(over='ignore', invalid='ignore'):        
        # The equation given by Tjorve and Tjorve 2010 (combo of eqn8 and eqn9)
        y = lower_plateau + (upper_plateau - lower_plateau) * (1 + (nu - 1) * np.exp( (-k * (x - x0)) / (nu**(nu/(1-nu))) ))**(1/(1-nu))

    return y


def general_logistic_wiki(x, x0, k, upper_plateau, lower_plateau, nu):
    with np.errstate(over='ignore', invalid='ignore'):
        C = 1
        Q = 1# lower_plateau
        y = lower_plateau + ( (upper_plateau - lower_plateau) / (C + Q * np.exp(-k*(x-x0)))**(1/nu) )
        
    return y


def dbl_logistic(x, x1, x2, k1, k2, high_plateau, mid_plateau):#, low_plateau):
#    y = low_plateau + ( (mid_plateau - low_plateau) / (1 + np.exp(-k1*(x-x1))) ) + ( (high_plateau - mid_plateau) / (1 + np.exp(-k2*(x-x2))) )
    ### Assuming the low plateau is equal to 0:
    y = 0 + ( (mid_plateau - 0) / (1 + np.exp(-k1*(x-x1))) ) + ( (high_plateau - mid_plateau) / (1 + np.exp(-k2*(x-x2))) )
    return y

#def janoschek_func(x, x0, k, upper_plateau, lower_plateau, nu):
#    y = upper_plateau - (upper_plateau - lower_plateau) * np.exp(-k * (x - x0)**nu)
#    return y

### The cell-cell adhesion appears to be fairly linear for ecto-ecto cells
### kept in dissociation buffer and ecto-meso cells.
def line(x, m, b):
    y = m * x + b
    return y



### If something is related to the circumference of the contact radius then it
### would follow some variation of the equation y = 1/r + high_plateau
#def inverse_power1(radius, high plateau):



### If something is related to a surface area of the cell (either free surface
### area or area of the contact) then it would follow some variation of the
### equation y = 1/(r**2) + high_plateau
#def inverse_power2:
    

    
### If something is related to the volume of the cell then it would follow some
### variation of the equation y = 1/(r**3) + high_plateau
#def inverse_power3:




### The 2 functions below relate the contact angle "theta" to the ratio of the 
### contact_radius / cell_radius_initial, ie. the theoretical Contact Surface Length (N.D.)
def f_cont_radius(theta):
    """Takes an angle (in degrees) and returns a ratio of contact_radius / cell_radius_initial:"""
    return (4. * np.sin(np.deg2rad(theta))**3 / (4. - (1 - np.cos(np.deg2rad(theta)))**2 * (2 + np.cos(np.deg2rad(theta))) ))**(1/3)
    

def f(theta, radius_initial):
    """Takes a contact angle (in degrees) and an initial radius and returns the 
    various expected variables. Please note that the contact angle theta is HALF
    of the angle measured when drawing 2 lines that are tangent to the cell
    surface at the 2 points that are the edges of the cell-cell contacts.
    Returns: (Rs, Vsphere, SAsphere, Asphere), (Rc, Vcap, SAcap, Acap, ContactLength)"""
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
    Rc = ( (4 * (Rs**3.0)) / ((2.0 - np.cos(phi)) * ((np.cos(phi) + 1.0)**2.0)) )**(1.0/3.0)
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
    
    # The surface area of the sphere after deformations (ie. spherical cap):
    SAcap = 2.0 * np.pi * Rc * h
    
    # Cross-sectional area of the sphere before deformations:
    Asphere = np.pi * Rs**2
    
    # Cross-sectional area of the sphere after deformations. Cross-section is
    # taken through the middle of the cap such that it cuts the flattened 
    # surface in half:
    Acap = (Rc**2.0) * np.arcsin( (h - Rc)/Rc ) + (h - Rc) * ((2.0 * Rc * h) - (h**2.0))**0.5 + (np.pi * (Rc**2.0) / 2.0)
    
    # The diameter of the flattened surface of the sphere:
    ContactLength = 2.0 * (h * (2.0 * Rc - h))**0.5

    return (Rs, Vsphere, SAsphere, Asphere), (Rc, Vcap, SAcap, Acap, ContactLength)




def subset_low_growth_plat(list_of_dataframes, list_of_original_dataframes, grow_to_low_validity):
    """ This function will take a subset that I defined manually from the 
    contact kinetics data and return that set as a list of dataframes (ie. in
    the same format as the contact kinetics data). I manually picked the last 
    data point to use in the subset with the help of an interactive graph made 
    by the pickable_scatter function and put that data in a .CSV file which is
    read in as a Pandas Dataframe and becomes the 'grow_to_low_validity' 
    variable. The first timepoint in the subset is based on where the 
    'cells_touching' column switches from False to True (ie. contact initiation)
    in the list of original dataframes (which are used because they haven't had
    timepoints where the cells don't touch dropped yet). Only the contact 
    initiation closest to the aforementioned 'last datapoint to use' is used
    to define the subset where growth to the low plateau occurs. """
    
    ### Make sure that the index of list_of_dataframes and 
    ### list_of_original_dataframes are both of dtype int64:
    for df in list_of_dataframes:
        df.index = df.index.astype(int)
    
    for df in list_of_original_dataframes:
        df.index = df.index.astype(int)
    

    
    data_list = list()
    cols = ['Source_File', 'first_low_plat_timepoint', 'last_low_plat_timepoint', 'first_low_plat_timepoint_idx', 'last_low_plat_timepoint_idx', 'last_low_plat_raw_timepoint_idx', 'dropna_idx == raw_idx', 'cont_init_idx']
    for i,df in enumerate(list_of_dataframes):
        source_file = df['Source_File'].iloc[0]
        last_low_plat_timepoint = grow_to_low_validity[grow_to_low_validity['Source_File'] == source_file]['last_low_plat_timeframe (minutes; t = ...)']
        last_low_plat_timepoint_idx = df[df['Time (minutes)'] == float(last_low_plat_timepoint.values)].index.values.astype(int)
        last_low_plat_raw_timepoint_idx = list_of_original_dataframes[i][list_of_original_dataframes[i]['Time (minutes)'] == float(last_low_plat_timepoint.values)].index.values.astype(int)
        if last_low_plat_timepoint_idx.any() and last_low_plat_raw_timepoint_idx.any():
            dropna_raw_equal = last_low_plat_timepoint_idx == last_low_plat_raw_timepoint_idx

            cont_init = list_of_original_dataframes[i]['cells touching?'].diff() == 1
            # Some of the timelapses have cells that round up after adhering, and 
            # one of them proceeded to divide, so that data was removed between the
            # original and cleaned up versions of the list of dataframes. As a 
            # result any contact initiation events that happen after the last time
            # point in the list_of_dataframes will need to be exluded:
            cont_init.mask(list_of_original_dataframes[i]['cells touching?'].index >= float(list_of_dataframes[i]['cells touching?'].index.max()), other=False, inplace=True)
    
            # If there is at least one contact initiation that occurs before the
            # last low plateau timepoint I decided to use, then pass:
            if (cont_init[cont_init == True].index.values < last_low_plat_timepoint_idx).any():
                pass
            
            else:
                first_idx = cont_init.index.min()
                cont_init[first_idx] = True # In the event that no contact initiations are observed, the cells must have started off in contact, so use the first timepoint as the start.
            cont_init_idx = cont_init[cont_init == True].index.values
            
            #if cont_init.any():
            valid_low_plat_starts = (last_low_plat_timepoint_idx - cont_init_idx) > 0
            first_low_plat_timepoint_idx = cont_init_idx[valid_low_plat_starts].max()
            first_low_plat_timepoint = list_of_original_dataframes[i]['Time (minutes)'].loc[first_low_plat_timepoint_idx]
            
            #else:                
            #    valid_low_plat_starts = np.asarray([], dtype=bool)
            #    first_low_plat_timepoint_idx = np.nan
            #    first_low_plat_timepoint = np.nan
                
        else:
            dropna_raw_equal = np.array([False])
            cont_init = np.asarray([], dtype=bool)
            cont_init_idx = np.asarray([], dtype=int)
            valid_low_plat_starts = np.asarray([], dtype=bool)
            first_low_plat_timepoint_idx = np.nan
            first_low_plat_timepoint = np.nan
        
        
        data = [source_file, first_low_plat_timepoint, float(last_low_plat_timepoint.values), first_low_plat_timepoint_idx, last_low_plat_timepoint_idx, last_low_plat_raw_timepoint_idx, bool(dropna_raw_equal.astype), cont_init_idx]
        data_list.append(data)
    
    
    details = pd.DataFrame(data = data_list, columns = cols)
    details_clean = details.mask(~details['dropna_idx == raw_idx']).dropna() # If any of the original and processed timepoints don't match, mask that data. Also remove any data where the last timepoint was not define.
    details_clean['first_low_plat_timepoint'] = details_clean['first_low_plat_timepoint'].astype(float)
    details_clean['first_low_plat_timepoint_idx'] = details_clean['first_low_plat_timepoint_idx'].astype(int)
    details_clean['last_low_plat_timepoint_idx'] = details_clean['last_low_plat_timepoint_idx'].astype(int)
    details_clean['last_low_plat_raw_timepoint_idx'] = details_clean['last_low_plat_raw_timepoint_idx'].astype(int)
    
    
    growth_to_low_dynamics_df_list = list()
    for i in details_clean.index:
        growth_to_low_dynamics_df_list.append(list_of_dataframes[i].loc[details_clean['first_low_plat_timepoint_idx'].loc[i] : details_clean['last_low_plat_timepoint_idx'].loc[i]])

    
    return growth_to_low_dynamics_df_list, details





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


### X, Y resolution of the Zeiss Axiovert200M using our 20X objective:
# This was used for all reaggregation assays except the endoderm-endoderm ones.
um_per_pixel = 0.506675079794729

### X, Y resolution of the Zeiss Axiovert200M using our 10X objective:
# This was used for the endoderm-endoderm reaggregation assays.
um_per_pixel_nn = 1.03280106727569
###

### These 2 parameters below are used for deciding on the x limits of the curve
# fitting and also for deciding how many finely the time is fitted for the curves:
fit_time_min, fit_time_max = -50, 160
step_size = (fit_time_max - fit_time_min) * 20 + 1


bounds_genlog_contnd = ([-50, 0, 1, 0, 0], [100, 100, 3, 1, 1000])
bounds_genlog_cont = ([-50, 0, 20, 0, 0], [100, 100, 100, 30, 1000])
bounds_genlog_angle = ([-50, 0, 1, 0, 0], [100, 100, 3, 1, 1000])

# Initial parameters and bounds for logistic fits for ecto-ecto, meso-meso,
# endo-endo, eeCcadMO, eeRcadMO, and eeRCcadMO data:
p0_angle = (10,1,120,0)
#bounds_angle = ([-50,0,60,0], [100,100, 180,60]) # This one worked, was used for committee meeting report Aug 2019
bounds_angle = ([-50,0,60,-180], [100,100, 180,60]) # This one is a test to help logistic functions fit exponential looking data better
p0_cont = (10,1,50, 0)
#bounds_cont = ([-50,0,30,0], [100,100,100,30]) # This one worked, was used for committee meeting report Aug 2019
bounds_cont = ([-50,0,30,-100], [100,100,100,30]) # This one is a test to help logistic functions fit exponential looking data better
p0_contnd = (10,1,2,0)
#bounds_contnd = ([-50,0,1,0], [100,100,3,1]) # This one worked, was used for committee meeting report Aug 2019
bounds_contnd = ([-50,0,1,-3], [100,100,3,1]) # This one is a test to help logistic functions fit exponential looking data better


# Initial parameters and bounds for logistic fits for ecto-meso data:
#bounds_genlog_angle_em = ([-50, 0, 0, 0, 0], [100, 100, 180, 180, 1000])
p0_angle_em = (0, 5, 60, 20)#5 was 0.01 before
p0_angle_em_genlog = (0,0,60,20)
#bounds_genlog_cont_em = ([-50, 0, 20, 0, 0], [100, 100, 100, 100, 1000])
p0_cont_em = (0, 5, 30, 10)#5 was 0.01 before
#bounds_genlog_contnd_em = ([-50, 0, 0, 0, 0], [100, 100, 4, 4, 1000])
p0_contnd_em = (0, 0.15, 1.0, 0.5)#0.15 was 0.01 before

bounds_angle_em = ([-50,0,60,0], [100,100, 180,60])
bounds_cont_em = ([-50,0,30,0], [100,100,100,30])
bounds_contnd_em = ([-50,0,1,0], [100,100,3,1])

# Initial parameters and bounds for logistic fits for ecto-meso data:
p0_angle_eedb = (-100, 0.01, 60, 30)
p0_contnd_angleBased_eedb = (-100, 0.01, 1.0, 0.5)

bounds_angle_eedb = ([-50,0,60,0], [100,100, 180,60])
bounds_cont_eedb = ([-50,0,30,0], [100,100,100,30])
bounds_contnd_eedb = ([-50,0,1,0], [100,100,3,1])

nu = 2


# Committee member wanted plots to restrict view to n>1:
n_size_thresh_for_xlim = 1 #1

###############################################################################
### Open all of your files into long-form dataframes:
# Set the location of the files of validated adhesion measurements:    
ee_path = Path.joinpath(data_dir, r"ecto-ecto")
# Import this data:
ee_list_original, ee_list_long = data_import_and_clean(ee_path)
# I'm just going to rename the 'Time' column to 'Time (minutes)' so that the 
# header has units.
[x.rename({'Time':'Time (minutes)'}, axis='columns', inplace=True) for x in ee_list_original]
ee_list_long.rename({'Time':'Time (minutes)'}, axis='columns', inplace=True)

# I'm going to mask and remove any rows where the 'Mask?' column is "TRUE":
ee_list = [x.mask(x['Mask? (manually defined by user who checks quality of cell borders)']) for x in ee_list_original]
ee_list = [x.dropna(axis=0, subset=['Source_File']) for x in ee_list]
ee_list_long = ee_list_long.mask(ee_list_long['Mask? (manually defined by user who checks quality of cell borders)'])
ee_list_long = ee_list_long.dropna(axis=0, subset=['Source_File'])

mm_path = Path.joinpath(data_dir, r"meso-meso")
mm_list_original, mm_list_long = data_import_and_clean(mm_path)
# I'm just going to rename the 'Time' column to 'Time (minutes)' so that the 
# header has units.
[x.rename({'Time':'Time (minutes)'}, axis='columns', inplace=True) for x in mm_list_original]
mm_list_long.rename({'Time':'Time (minutes)'}, axis='columns', inplace=True)

# And also I'm going to mask and remove any rows where the 'Mask?' column is "TRUE":
mm_list = [x.mask(x['Mask? (manually defined by user who checks quality of cell borders)']) for x in mm_list_original]
mm_list = [x.dropna(axis=0, subset=['Source_File']) for x in mm_list]
mm_list_long = mm_list_long.mask(mm_list_long['Mask? (manually defined by user who checks quality of cell borders)'])
mm_list_long = mm_list_long.dropna(axis=0, subset=['Source_File'])

em_path = Path.joinpath(data_dir, r"ecto-meso")
em_list_original, em_list_long = data_import_and_clean(em_path)
# I'm just going to rename the 'Time' column to 'Time (minutes)' so that the 
# header has units.
[x.rename({'Time':'Time (minutes)'}, axis='columns', inplace=True) for x in em_list_original]
em_list_long.rename({'Time':'Time (minutes)'}, axis='columns', inplace=True)

em_list = [x.mask(x['Mask? (manually defined by user who checks quality of cell borders)']) for x in em_list_original]
em_list = [x.dropna(axis=0, subset=['Source_File']) for x in em_list]
em_list_long = em_list_long.mask(em_list_long['Mask? (manually defined by user who checks quality of cell borders)'])
em_list_long = em_list_long.dropna(axis=0, subset=['Source_File'])

eedb_path = Path.joinpath(data_dir, r"ecto-ecto 1xDB")
eedb_list_original, eedb_list_long = data_import_and_clean(eedb_path)
# I'm just going to rename the 'Time' column to 'Time (minutes)' so that the 
# header has units.
[x.rename({'Time':'Time (minutes)'}, axis='columns', inplace=True) for x in eedb_list_original]
eedb_list_long.rename({'Time':'Time (minutes)'}, axis='columns', inplace=True)

eedb_list = [x.mask(x['Mask? (manually defined by user who checks quality of cell borders)']) for x in eedb_list_original]
eedb_list = [x.dropna(axis=0, subset=['Source_File']) for x in eedb_list]
# Since masking values as nans converts column dtypes, will need to change them
# back for 'cells touching?' and the 'mask' column:
#eedb_list_original[0].dtypes[eedb_list[0].dtypes != eedb_list_original[0].dtypes]
for df in eedb_list:
    df['cells touching?'] = df['cells touching?'].astype(bool)
    df['Mask? (manually defined by user who checks quality of cell borders)'] = df['Mask? (manually defined by user who checks quality of cell borders)'].astype(bool)
eedb_list_long = eedb_list_long.mask(eedb_list_long['Mask? (manually defined by user who checks quality of cell borders)'])
eedb_list_long = eedb_list_long.dropna(axis=0, subset=['Source_File'])
eedb_list_long['cells touching?'] = eedb_list_long['cells touching?'].astype(bool)
eedb_list_long['Mask? (manually defined by user who checks quality of cell borders)'] = eedb_list_long['Mask? (manually defined by user who checks quality of cell borders)'].astype(bool)

eedb_fix_path = Path.joinpath(data_dir, r"ecto-ecto 1xDB_fix")
eedb_fix_list_original, eedb_fix_list_long = data_import_and_clean(eedb_fix_path)
# I'm just going to rename the 'Time' column to 'Time (minutes)' so that the 
# header has units.
[x.rename({'Time':'Time (minutes)'}, axis='columns', inplace=True) for x in eedb_fix_list_original]
eedb_fix_list_long.rename({'Time':'Time (minutes)'}, axis='columns', inplace=True)
# eedb_fix_list = [x.mask(x['Mask? (manually defined by user who checks quality of cell borders)']) for x in eedb_fix_list_original]
# eedb_fix_list_long = eedb_fix_list_long.mask(eedb_fix_list_long['Mask? (manually defined by user who checks quality of cell borders)'])
eedb_fix_list = [x for x in eedb_fix_list_original]

eeCcadMO_path = Path.joinpath(data_dir, r"ecto-ecto CcadMO")
eeCcadMO_list_original, eeCcadMO_list_long = data_import_and_clean(eeCcadMO_path)
# I'm just going to rename the 'Time' column to 'Time (minutes)' so that the 
# header has units.
[x.rename({'Time':'Time (minutes)'}, axis='columns', inplace=True) for x in eeCcadMO_list_original]
eeCcadMO_list_long.rename({'Time':'Time (minutes)'}, axis='columns', inplace=True)

eeCcadMO_list = [x.mask(x['Mask? (manually defined by user who checks quality of cell borders)']) for x in eeCcadMO_list_original]
eeCcadMO_list = [x.dropna(axis=0, subset=['Source_File']) for x in eeCcadMO_list]
eeCcadMO_list_long = eeCcadMO_list_long.mask(eeCcadMO_list_long['Mask? (manually defined by user who checks quality of cell borders)'])
eeCcadMO_list_long = eeCcadMO_list_long.dropna(axis=0, subset=['Source_File'])


eeRcadMO_path = Path.joinpath(data_dir, r"ecto-ecto RcadMO")
eeRcadMO_list_original, eeRcadMO_list_long = data_import_and_clean(eeRcadMO_path)
# I'm just going to rename the 'Time' column to 'Time (minutes)' so that the 
# header has units.
[x.rename({'Time':'Time (minutes)'}, axis='columns', inplace=True) for x in eeRcadMO_list_original]
eeRcadMO_list_long.rename({'Time':'Time (minutes)'}, axis='columns', inplace=True)

eeRcadMO_list = [x.mask(x['Mask? (manually defined by user who checks quality of cell borders)']) for x in eeRcadMO_list_original]
eeRcadMO_list = [x.dropna(axis=0, subset=['Source_File']) for x in eeRcadMO_list]
eeRcadMO_list_long = eeRcadMO_list_long.mask(eeRcadMO_list_long['Mask? (manually defined by user who checks quality of cell borders)'])
eeRcadMO_list_long = eeRcadMO_list_long.dropna(axis=0, subset=['Source_File'])


eeRCcadMO_path = Path.joinpath(data_dir, r"ecto-ecto RCcadMO")
eeRCcadMO_list_original, eeRCcadMO_list_long = data_import_and_clean(eeRCcadMO_path)
# I'm just going to rename the 'Time' column to 'Time (minutes)' so that the 
# header has units.
[x.rename({'Time':'Time (minutes)'}, axis='columns', inplace=True) for x in eeRCcadMO_list_original]
eeRCcadMO_list_long.rename({'Time':'Time (minutes)'}, axis='columns', inplace=True)

eeRCcadMO_list = [x.mask(x['Mask? (manually defined by user who checks quality of cell borders)']) for x in eeRCcadMO_list_original]
eeRCcadMO_list = [x.dropna(axis=0, subset=['Source_File']) for x in eeRCcadMO_list]
eeRCcadMO_list_long = eeRCcadMO_list_long.mask(eeRCcadMO_list_long['Mask? (manually defined by user who checks quality of cell borders)'])
eeRCcadMO_list_long = eeRCcadMO_list_long.dropna(axis=0, subset=['Source_File'])



# ee4MU_path = Path.joinpath(data_dir, r"ecto-ecto 4MU")
# ee4MU_list_original, ee4MU_list_long = data_import_and_clean(ee4MU_path)
# # I'm just going to rename the 'Time' column to 'Time (minutes)' so that the 
# # header has units.
# [x.rename({'Time':'Time (minutes)'}, axis='columns', inplace=True) for x in ee4MU_list_original]
# ee4MU_list_long.rename({'Time':'Time (minutes)'}, axis='columns', inplace=True)

# ee4MU_list = [x.mask(x['Mask? (manually defined by user who checks quality of cell borders)']) for x in ee4MU_list_original]
# ee4MU_list = [x.dropna(axis=0, subset=['Source_File']) for x in ee4MU_list]
# ee4MU_list_long = ee4MU_list_long.mask(ee4MU_list_long['Mask? (manually defined by user who checks quality of cell borders)'])
# ee4MU_list_long = ee4MU_list_long.dropna(axis=0, subset=['Source_File'])

nn_path = Path.joinpath(data_dir, r"endo-endo")
nn_list_original, nn_list_long = data_import_and_clean(nn_path)
# I'm just going to rename the 'Time' column to 'Time (minutes)' so that the 
# header has units.
[x.rename({'Time':'Time (minutes)'}, axis='columns', inplace=True) for x in nn_list_original]
nn_list_long.rename({'Time':'Time (minutes)'}, axis='columns', inplace=True)

nn_list = [x.mask(x['Mask? (manually defined by user who checks quality of cell borders)']) for x in nn_list_original]
nn_list = [x.dropna(axis=0, subset=['Source_File']) for x in nn_list]
nn_list_long = nn_list_long.mask(nn_list_long['Mask? (manually defined by user who checks quality of cell borders)'])
nn_list_long = nn_list_long.dropna(axis=0, subset=['Source_File'])

# contact angles and diameters of cells that are just touching at a point for varying cell radii:
two_circ_path = Path.joinpath(data_dir, r"2_circles")
two_circ_list_original, two_circ_list_long = data_import_and_clean(two_circ_path)

two_circ_list = [x.mask(x['Mask? (manually defined by user who checks quality of cell borders)']) for x in two_circ_list_original]
two_circ_list = [x.dropna(axis=0, subset=['Source_File']) for x in two_circ_list]
two_circ_list_long = two_circ_list_long.mask(two_circ_list_long['Mask? (manually defined by user who checks quality of cell borders)'])
two_circ_list_long = two_circ_list_long.dropna(axis=0, subset=['Source_File'])

# simulation of 2 perfectly spherical cells of constant volume adhering:
radius27px0deg_cell_sim_path = Path.joinpath(data_dir, "radius27px0deg_cell_sim")
radius27px0deg_cell_sim_list_original, radius27px0deg_cell_sim_list_long = data_import_and_clean(radius27px0deg_cell_sim_path)

radius27px15deg_cell_sim_path = Path.joinpath(data_dir, "radius27px15deg_cell_sim")
radius27px15deg_cell_sim_list_original, radius27px15deg_cell_sim_list_long = data_import_and_clean(radius27px15deg_cell_sim_path)

radius27px30deg_cell_sim_path = Path.joinpath(data_dir, "radius27px30deg_cell_sim")
radius27px30deg_cell_sim_list_original, radius27px30deg_cell_sim_list_long = data_import_and_clean(radius27px30deg_cell_sim_path)

radius27px45deg_cell_sim_path = Path.joinpath(data_dir, "radius27px45deg_cell_sim")
radius27px45deg_cell_sim_list_original, radius27px45deg_cell_sim_list_long = data_import_and_clean(radius27px45deg_cell_sim_path)


radius27px_cell_sim_list_original = radius27px0deg_cell_sim_list_original + radius27px15deg_cell_sim_list_original + radius27px30deg_cell_sim_list_original + radius27px45deg_cell_sim_list_original
radius27px_cell_sim_list_long = pd.concat([radius27px0deg_cell_sim_list_long, 
                                           radius27px15deg_cell_sim_list_long, 
                                           radius27px30deg_cell_sim_list_long, 
                                           radius27px45deg_cell_sim_list_long])

###############################################################################
### Non-dimensionalize the contact diameter data:

#ee_first_valid_area = [(df['Area of the cell pair (px**2)'].dropna() / 2.).mean() for df in ee_list]
#ee_first_valid_area_red = [(df['Area of cell 1 (Red) (px**2)'].dropna()).mean() for df in ee_list]
#ee_first_valid_area_grn = [(df['Area of cell 2 (Green) (px**2)'].dropna()).mean() for df in ee_list]

ee_first_valid_area = [(df['Area of the cell pair (px**2)'].dropna() / 2.).iloc[0] for df in ee_list]
ee_first_valid_area_red = [(df['Area of cell 1 (Red) (px**2)'].dropna()).iloc[0] for df in ee_list]
ee_first_valid_area_grn = [(df['Area of cell 2 (Green) (px**2)'].dropna()).iloc[0] for df in ee_list]

ee_first_valid_radius = [np.sqrt(area / np.pi) for area in ee_first_valid_area]
ee_first_valid_radius_red = [np.sqrt(area / np.pi) for area in ee_first_valid_area_red]
ee_first_valid_radius_grn = [np.sqrt(area / np.pi) for area in ee_first_valid_area_grn]
for i,val in enumerate(ee_first_valid_radius):
    ee_list[i]['first_valid_radius'] = val
    ee_list_original[i]['first_valid_radius'] = val
    # ee_list[i]['first_valid_radius_red'] = ee_first_valid_radius_red[i]
    # ee_list[i]['first_valid_radius_grn'] = ee_first_valid_radius_grn[i]
ee_first_valid_radius_sf = pd.DataFrame({'first_valid_radius (um)':np.array(ee_first_valid_radius) * um_per_pixel, 'Source_File':[x.Source_File.iloc[-1] for x in ee_list]})


ee_contact_diam_non_dim_df = ee_list_long.copy()
ee_contact_diam_non_dim = [ee_list[i]['Contact Surface Length (px)'] / \
                           ee_first_valid_radius[i] for i in \
                           np.arange(len(ee_first_valid_radius))]

ee_area_cellpair_non_dim = [ee_list[i]['Area of the cell pair (px**2)'] / \
                            ee_first_valid_area[i] for i in \
                            np.arange(len(ee_first_valid_area))]

ee_area_red_non_dim = [ee_list[i]['Area of cell 1 (Red) (px**2)'] / \
                       ee_first_valid_area_red[i] for i in \
                       np.arange(len(ee_first_valid_area_red))]

ee_area_grn_non_dim = [ee_list[i]['Area of cell 2 (Green) (px**2)'] / \
                       ee_first_valid_area_grn[i] for i in \
                       np.arange(len(ee_first_valid_area_grn))]

ee_contact_diam_non_dim = pd.concat(ee_contact_diam_non_dim)
ee_contact_diam_non_dim_df['Contact Surface Length (px)'] = ee_contact_diam_non_dim

ee_area_cellpair_non_dim = pd.concat(ee_area_cellpair_non_dim)
ee_contact_diam_non_dim_df['Area of the cell pair (px**2)'] = ee_area_cellpair_non_dim

ee_area_red_non_dim = pd.concat(ee_area_red_non_dim)
ee_contact_diam_non_dim_df['Area of cell 1 (Red) (px**2)'] = ee_area_red_non_dim

ee_area_grn_non_dim = pd.concat(ee_area_grn_non_dim)
ee_contact_diam_non_dim_df['Area of cell 2 (Green) (px**2)'] = ee_area_grn_non_dim

ee_contact_diam_non_dim_df = ee_contact_diam_non_dim_df.rename(index=str, columns={'Contact Surface Length (px)': 'Contact Surface Length (N.D.)'})
ee_contact_diam_non_dim_df = ee_contact_diam_non_dim_df.rename(index=str, columns={'Area of the cell pair (px**2)': 'Area of the cell pair (N.D.)'})
ee_contact_diam_non_dim_df = ee_contact_diam_non_dim_df.rename(index=str, columns={'Area of cell 1 (Red) (px**2)': 'Area of cell 1 (Red) (N.D.)'})
ee_contact_diam_non_dim_df = ee_contact_diam_non_dim_df.rename(index=str, columns={'Area of cell 2 (Green) (px**2)': 'Area of cell 2 (Green) (N.D.)'})

ee_contact_diam_non_dim_df_list = [ee_contact_diam_non_dim_df[ee_contact_diam_non_dim_df['Source_File'] == x] for x in np.unique(ee_contact_diam_non_dim_df['Source_File'])]

#eelow_first_valid_area = [(df['Area of the cell pair (px**2)'].dropna() / 2.).mean() for df in eelow_list]
#eelow_first_valid_radius = [np.sqrt(area / np.pi) for area in eelow_first_valid_area]
#eelow_contact_diam_non_dim_df = eelow_list_long.copy()
#eelow_contact_diam_non_dim = [eelow_list[i]['Contact Surface Length (px)'] / \
#                           eelow_first_valid_radius[i] for i in \
#                           np.arange(len(eelow_first_valid_radius))]
#eelow_contact_diam_non_dim = pd.concat(eelow_contact_diam_non_dim)
#eelow_contact_diam_non_dim_df['Contact Surface Length (px)'] = eelow_contact_diam_non_dim
#eelow_contact_diam_non_dim_df = eelow_contact_diam_non_dim_df.rename(index=str, columns={'Contact Surface Length (px)': 'Contact Surface Length (N.D.)'})

#mm_first_valid_area = [(df['Area of the cell pair (px**2)'].dropna() / 2.).mean() for df in mm_list]
#mm_first_valid_area_red = [(df['Area of cell 1 (Red) (px**2)'].dropna()).mean() for df in mm_list]
#mm_first_valid_area_grn = [(df['Area of cell 2 (Green) (px**2)'].dropna()).mean() for df in mm_list]

mm_first_valid_area = [(df['Area of the cell pair (px**2)'].dropna() / 2.).iloc[0] for df in mm_list]
mm_first_valid_area_red = [(df['Area of cell 1 (Red) (px**2)'].dropna()).iloc[0] for df in mm_list]
mm_first_valid_area_grn = [(df['Area of cell 2 (Green) (px**2)'].dropna()).iloc[0] for df in mm_list]


mm_first_valid_radius = [np.sqrt(area / np.pi) for area in mm_first_valid_area]
for i,val in enumerate(mm_first_valid_radius):
    mm_list[i]['first_valid_radius'] = val
    mm_list_original[i]['first_valid_radius'] = val
    # mm_list[i]['first_valid_radius_red'] = mm_first_valid_radius_red[i]
    # mm_list[i]['first_valid_radius_grn'] = mm_first_valid_radius_grn[i]
mm_first_valid_radius_sf = pd.DataFrame({'first_valid_radius (um)':np.array(mm_first_valid_radius) * um_per_pixel, 'Source_File':[x.Source_File.iloc[-1] for x in mm_list]})


mm_contact_diam_non_dim_df = mm_list_long.copy()
mm_contact_diam_non_dim = [mm_list[i]['Contact Surface Length (px)'] / \
                           mm_first_valid_radius[i] for i in \
                           np.arange(len(mm_first_valid_radius))]

mm_area_cellpair_non_dim = [mm_list[i]['Area of the cell pair (px**2)'] / \
                            mm_first_valid_area[i] for i in \
                            np.arange(len(mm_first_valid_area))]

mm_area_red_non_dim = [mm_list[i]['Area of cell 1 (Red) (px**2)'] / \
                       mm_first_valid_area_red[i] for i in \
                       np.arange(len(mm_first_valid_area_red))]

mm_area_grn_non_dim = [mm_list[i]['Area of cell 2 (Green) (px**2)'] / \
                       mm_first_valid_area_grn[i] for i in \
                       np.arange(len(mm_first_valid_area_grn))]

mm_contact_diam_non_dim = pd.concat(mm_contact_diam_non_dim)
mm_contact_diam_non_dim_df['Contact Surface Length (px)'] = mm_contact_diam_non_dim.values

mm_area_cellpair_non_dim = pd.concat(mm_area_cellpair_non_dim)
mm_contact_diam_non_dim_df['Area of the cell pair (px**2)'] = mm_area_cellpair_non_dim

mm_area_red_non_dim = pd.concat(mm_area_red_non_dim)
mm_contact_diam_non_dim_df['Area of cell 1 (Red) (px**2)'] = mm_area_red_non_dim

mm_area_grn_non_dim = pd.concat(mm_area_grn_non_dim)
mm_contact_diam_non_dim_df['Area of cell 2 (Green) (px**2)'] = mm_area_grn_non_dim

mm_contact_diam_non_dim_df = mm_contact_diam_non_dim_df.rename(index=str, columns={'Contact Surface Length (px)': 'Contact Surface Length (N.D.)'})
mm_contact_diam_non_dim_df = mm_contact_diam_non_dim_df.rename(index=str, columns={'Area of the cell pair (px**2)': 'Area of the cell pair (N.D.)'})
mm_contact_diam_non_dim_df = mm_contact_diam_non_dim_df.rename(index=str, columns={'Area of cell 1 (Red) (px**2)': 'Area of cell 1 (Red) (N.D.)'})
mm_contact_diam_non_dim_df = mm_contact_diam_non_dim_df.rename(index=str, columns={'Area of cell 2 (Green) (px**2)': 'Area of cell 2 (Green) (N.D.)'})

mm_contact_diam_non_dim_df_list = [mm_contact_diam_non_dim_df[mm_contact_diam_non_dim_df['Source_File'] == x] for x in np.unique(mm_contact_diam_non_dim_df['Source_File'])]

#em_first_valid_area = [(df['Area of the cell pair (px**2)'].dropna() / 2.).mean() for df in em_list]
#em_first_valid_area_red = [(df['Area of cell 1 (Red) (px**2)'].dropna()).mean() for df in em_list]
#em_first_valid_area_grn = [(df['Area of cell 2 (Green) (px**2)'].dropna()).mean() for df in em_list]

em_first_valid_area = [(df['Area of the cell pair (px**2)'].dropna() / 2.).iloc[0] for df in em_list]
em_first_valid_area_red = [(df['Area of cell 1 (Red) (px**2)'].dropna()).iloc[0] for df in em_list]
em_first_valid_area_grn = [(df['Area of cell 2 (Green) (px**2)'].dropna()).iloc[0] for df in em_list]


em_first_valid_radius = [np.sqrt(area / np.pi) for area in em_first_valid_area]
for i,val in enumerate(em_first_valid_radius):
    em_list[i]['first_valid_radius'] = val
    em_list_original[i]['first_valid_radius'] = val
    # em_list[i]['first_valid_radius_red'] = em_first_valid_radius_red[i]
    # em_list[i]['first_valid_radius_grn'] = em_first_valid_radius_grn[i]
em_first_valid_radius_sf = pd.DataFrame({'first_valid_radius (um)':np.array(em_first_valid_radius) * um_per_pixel, 'Source_File':[x.Source_File.iloc[-1] for x in em_list]})


em_contact_diam_non_dim_df = em_list_long.copy()
em_contact_diam_non_dim = [em_list[i]['Contact Surface Length (px)'] / \
                           em_first_valid_radius[i] for i in \
                           np.arange(len(em_first_valid_radius))]

em_area_cellpair_non_dim = [em_list[i]['Area of the cell pair (px**2)'] / \
                            em_first_valid_area[i] for i in \
                            np.arange(len(em_first_valid_area))]

em_area_red_non_dim = [em_list[i]['Area of cell 1 (Red) (px**2)'] / \
                       em_first_valid_area_red[i] for i in \
                       np.arange(len(em_first_valid_area_red))]

em_area_grn_non_dim = [em_list[i]['Area of cell 2 (Green) (px**2)'] / \
                       em_first_valid_area_grn[i] for i in \
                       np.arange(len(em_first_valid_area_grn))]

em_contact_diam_non_dim = pd.concat(em_contact_diam_non_dim)
em_contact_diam_non_dim_df['Contact Surface Length (px)'] = em_contact_diam_non_dim

em_area_cellpair_non_dim = pd.concat(em_area_cellpair_non_dim)
em_contact_diam_non_dim_df['Area of the cell pair (px**2)'] = em_area_cellpair_non_dim

em_area_red_non_dim = pd.concat(em_area_red_non_dim)
em_contact_diam_non_dim_df['Area of cell 1 (Red) (px**2)'] = em_area_red_non_dim

em_area_grn_non_dim = pd.concat(em_area_grn_non_dim)
em_contact_diam_non_dim_df['Area of cell 2 (Green) (px**2)'] = em_area_grn_non_dim

em_contact_diam_non_dim_df = em_contact_diam_non_dim_df.rename(index=str, columns={'Contact Surface Length (px)': 'Contact Surface Length (N.D.)'})
em_contact_diam_non_dim_df = em_contact_diam_non_dim_df.rename(index=str, columns={'Area of the cell pair (px**2)': 'Area of the cell pair (N.D.)'})
em_contact_diam_non_dim_df = em_contact_diam_non_dim_df.rename(index=str, columns={'Area of cell 1 (Red) (px**2)': 'Area of cell 1 (Red) (N.D.)'})
em_contact_diam_non_dim_df = em_contact_diam_non_dim_df.rename(index=str, columns={'Area of cell 2 (Green) (px**2)': 'Area of cell 2 (Green) (N.D.)'})

em_contact_diam_non_dim_df_list = [em_contact_diam_non_dim_df[em_contact_diam_non_dim_df['Source_File'] == x] for x in np.unique(em_contact_diam_non_dim_df['Source_File'])]


#eedb_first_valid_area = [(df['Area of the cell pair (px**2)'].dropna() / 2.).mean() for df in eedb_list]
#eedb_first_valid_area_red = [(df['Area of cell 1 (Red) (px**2)'].dropna()).mean() for df in eedb_list]
#eedb_first_valid_area_grn = [(df['Area of cell 2 (Green) (px**2)'].dropna()).mean() for df in eedb_list]

eedb_first_valid_area = [(df['Area of the cell pair (px**2)'].dropna() / 2.).iloc[0] for df in eedb_list]
eedb_first_valid_area_red = [(df['Area of cell 1 (Red) (px**2)'].dropna()).iloc[0] for df in eedb_list]
eedb_first_valid_area_grn = [(df['Area of cell 2 (Green) (px**2)'].dropna()).iloc[0] for df in eedb_list]


eedb_first_valid_radius = [np.sqrt(area / np.pi) for area in eedb_first_valid_area]
for i,val in enumerate(eedb_first_valid_radius):
    eedb_list[i]['first_valid_radius'] = val
    eedb_list_original[i]['first_valid_radius'] = val
    # eedb_list[i]['first_valid_radius_red'] = eedb_first_valid_radius_red[i]
    # eedb_list[i]['first_valid_radius_grn'] = eedb_first_valid_radius_grn[i]
eedb_first_valid_radius_sf = pd.DataFrame({'first_valid_radius (um)':np.array(eedb_first_valid_radius) * um_per_pixel, 'Source_File':[x.Source_File.iloc[-1] for x in eedb_list]})


eedb_contact_diam_non_dim_df = eedb_list_long.copy()
eedb_contact_diam_non_dim = [eedb_list[i]['Contact Surface Length (px)'] / \
                           eedb_first_valid_radius[i] for i in \
                           np.arange(len(eedb_first_valid_radius))]

eedb_area_cellpair_non_dim = [eedb_list[i]['Area of the cell pair (px**2)'] / \
                              eedb_first_valid_area[i] for i in \
                              np.arange(len(eedb_first_valid_area))]

eedb_area_red_non_dim = [eedb_list[i]['Area of cell 1 (Red) (px**2)'] / \
                         eedb_first_valid_area_red[i] for i in \
                         np.arange(len(eedb_first_valid_area_red))]

eedb_area_grn_non_dim = [eedb_list[i]['Area of cell 2 (Green) (px**2)'] / \
                         eedb_first_valid_area_grn[i] for i in \
                         np.arange(len(eedb_first_valid_area_grn))]

eedb_contact_diam_non_dim = pd.concat(eedb_contact_diam_non_dim)
eedb_contact_diam_non_dim_df['Contact Surface Length (px)'] = eedb_contact_diam_non_dim

eedb_area_cellpair_non_dim = pd.concat(eedb_area_cellpair_non_dim)
eedb_contact_diam_non_dim_df['Area of the cell pair (px**2)'] = eedb_area_cellpair_non_dim

eedb_area_red_non_dim = pd.concat(eedb_area_red_non_dim)
eedb_contact_diam_non_dim_df['Area of cell 1 (Red) (px**2)'] = eedb_area_red_non_dim

eedb_area_grn_non_dim = pd.concat(eedb_area_grn_non_dim)
eedb_contact_diam_non_dim_df['Area of cell 2 (Green) (px**2)'] = eedb_area_grn_non_dim

eedb_contact_diam_non_dim_df = eedb_contact_diam_non_dim_df.rename(index=str, columns={'Contact Surface Length (px)': 'Contact Surface Length (N.D.)'})
eedb_contact_diam_non_dim_df = eedb_contact_diam_non_dim_df.rename(index=str, columns={'Area of the cell pair (px**2)': 'Area of the cell pair (N.D.)'})
eedb_contact_diam_non_dim_df = eedb_contact_diam_non_dim_df.rename(index=str, columns={'Area of cell 1 (Red) (px**2)': 'Area of cell 1 (Red) (N.D.)'})
eedb_contact_diam_non_dim_df = eedb_contact_diam_non_dim_df.rename(index=str, columns={'Area of cell 2 (Green) (px**2)': 'Area of cell 2 (Green) (N.D.)'})

eedb_contact_diam_non_dim_df_list = [eedb_contact_diam_non_dim_df[eedb_contact_diam_non_dim_df['Source_File'] == x] for x in np.unique(eedb_contact_diam_non_dim_df['Source_File'])]

#eeCcadMO_first_valid_area = [(df['Area of the cell pair (px**2)'].dropna() / 2.).mean() for df in eeCcadMO_list]
#eeCcadMO_first_valid_area_red = [(df['Area of cell 1 (Red) (px**2)'].dropna()).mean() for df in eeCcadMO_list]
#eeCcadMO_first_valid_area_grn = [(df['Area of cell 2 (Green) (px**2)'].dropna()).mean() for df in eeCcadMO_list]

eeCcadMO_first_valid_area = [(df['Area of the cell pair (px**2)'].dropna() / 2.).iloc[0] for df in eeCcadMO_list]
eeCcadMO_first_valid_area_red = [(df['Area of cell 1 (Red) (px**2)'].dropna()).iloc[0] for df in eeCcadMO_list]
eeCcadMO_first_valid_area_grn = [(df['Area of cell 2 (Green) (px**2)'].dropna()).iloc[0] for df in eeCcadMO_list]


eeCcadMO_first_valid_radius = [np.sqrt(area / np.pi) for area in eeCcadMO_first_valid_area]
for i,val in enumerate(eeCcadMO_first_valid_radius):
    eeCcadMO_list[i]['first_valid_radius'] = val
    eeCcadMO_list_original[i]['first_valid_radius'] = val
    # eeCcadMO_list[i]['first_valid_radius_red'] = eeCcadMO_first_valid_radius_red[i]
    # eeCcadMO_list[i]['first_valid_radius_grn'] = eeCcadMO_first_valid_radius_grn[i]
eeCcadMO_first_valid_radius_sf = pd.DataFrame({'first_valid_radius (um)':np.array(eeCcadMO_first_valid_radius) * um_per_pixel, 'Source_File':[x.Source_File.iloc[-1] for x in eeCcadMO_list]})


eeCcadMO_contact_diam_non_dim_df = eeCcadMO_list_long.copy()
eeCcadMO_contact_diam_non_dim = [eeCcadMO_list[i]['Contact Surface Length (px)'] / \
                           eeCcadMO_first_valid_radius[i] for i in \
                           np.arange(len(eeCcadMO_first_valid_radius))]

eeCcadMO_area_cellpair_non_dim = [eeCcadMO_list[i]['Area of the cell pair (px**2)'] / \
                                  eeCcadMO_first_valid_area[i] for i in \
                                  np.arange(len(eeCcadMO_first_valid_area))]

eeCcadMO_area_red_non_dim = [eeCcadMO_list[i]['Area of cell 1 (Red) (px**2)'] / \
                             eeCcadMO_first_valid_area_red[i] for i in \
                             np.arange(len(eeCcadMO_first_valid_area_red))]

eeCcadMO_area_grn_non_dim = [eeCcadMO_list[i]['Area of cell 2 (Green) (px**2)'] / \
                             eeCcadMO_first_valid_area_grn[i] for i in \
                             np.arange(len(eeCcadMO_first_valid_area_grn))]

eeCcadMO_contact_diam_non_dim = pd.concat(eeCcadMO_contact_diam_non_dim)
eeCcadMO_contact_diam_non_dim_df['Contact Surface Length (px)'] = eeCcadMO_contact_diam_non_dim

eeCcadMO_area_cellpair_non_dim = pd.concat(eeCcadMO_area_cellpair_non_dim)
eeCcadMO_contact_diam_non_dim_df['Area of the cell pair (px**2)'] = eeCcadMO_area_cellpair_non_dim

eeCcadMO_area_red_non_dim = pd.concat(eeCcadMO_area_red_non_dim)
eeCcadMO_contact_diam_non_dim_df['Area of cell 1 (Red) (px**2)'] = eeCcadMO_area_red_non_dim

eeCcadMO_area_grn_non_dim = pd.concat(eeCcadMO_area_grn_non_dim)
eeCcadMO_contact_diam_non_dim_df['Area of cell 2 (Green) (px**2)'] = eeCcadMO_area_grn_non_dim

eeCcadMO_contact_diam_non_dim_df = eeCcadMO_contact_diam_non_dim_df.rename(index=str, columns={'Contact Surface Length (px)': 'Contact Surface Length (N.D.)'})
eeCcadMO_contact_diam_non_dim_df = eeCcadMO_contact_diam_non_dim_df.rename(index=str, columns={'Area of the cell pair (px**2)': 'Area of the cell pair (N.D.)'})
eeCcadMO_contact_diam_non_dim_df = eeCcadMO_contact_diam_non_dim_df.rename(index=str, columns={'Area of cell 1 (Red) (px**2)': 'Area of cell 1 (Red) (N.D.)'})
eeCcadMO_contact_diam_non_dim_df = eeCcadMO_contact_diam_non_dim_df.rename(index=str, columns={'Area of cell 2 (Green) (px**2)': 'Area of cell 2 (Green) (N.D.)'})

eeCcadMO_contact_diam_non_dim_df_list = [eeCcadMO_contact_diam_non_dim_df[eeCcadMO_contact_diam_non_dim_df['Source_File'] == x] for x in np.unique(eeCcadMO_contact_diam_non_dim_df['Source_File'])]


#eeRcadMO_first_valid_area = [(df['Area of the cell pair (px**2)'].dropna() / 2.).mean() for df in eeRcadMO_list]
#eeRcadMO_first_valid_area_red = [(df['Area of cell 1 (Red) (px**2)'].dropna()).mean() for df in eeRcadMO_list]
#eeRcadMO_first_valid_area_grn = [(df['Area of cell 2 (Green) (px**2)'].dropna()).mean() for df in eeRcadMO_list]

eeRcadMO_first_valid_area = [(df['Area of the cell pair (px**2)'].dropna() / 2.).iloc[0] for df in eeRcadMO_list]
eeRcadMO_first_valid_area_red = [(df['Area of cell 1 (Red) (px**2)'].dropna()).iloc[0] for df in eeRcadMO_list]
eeRcadMO_first_valid_area_grn = [(df['Area of cell 2 (Green) (px**2)'].dropna()).iloc[0] for df in eeRcadMO_list]

eeRcadMO_first_valid_radius = [np.sqrt(area / np.pi) for area in eeRcadMO_first_valid_area]
for i,val in enumerate(eeRcadMO_first_valid_radius):
    eeRcadMO_list[i]['first_valid_radius'] = val
    eeRcadMO_list_original[i]['first_valid_radius'] = val
    # eeRcadMO_list[i]['first_valid_radius_red'] = eeRcadMO_first_valid_radius_red[i]
    # eeRcadMO_list[i]['first_valid_radius_grn'] = eeRcadMO_first_valid_radius_grn[i]
eeRcadMO_first_valid_radius_sf = pd.DataFrame({'first_valid_radius (um)':np.array(eeRcadMO_first_valid_radius) * um_per_pixel, 'Source_File':[x.Source_File.iloc[-1] for x in eeRcadMO_list]})


eeRcadMO_contact_diam_non_dim_df = eeRcadMO_list_long.copy()
eeRcadMO_contact_diam_non_dim = [eeRcadMO_list[i]['Contact Surface Length (px)'] / \
                           eeRcadMO_first_valid_radius[i] for i in \
                           np.arange(len(eeRcadMO_first_valid_radius))]

eeRcadMO_area_cellpair_non_dim = [eeRcadMO_list[i]['Area of the cell pair (px**2)'] / \
                                  eeRcadMO_first_valid_area[i] for i in \
                                  np.arange(len(eeRcadMO_first_valid_area))]

eeRcadMO_area_red_non_dim = [eeRcadMO_list[i]['Area of cell 1 (Red) (px**2)'] / \
                             eeRcadMO_first_valid_area_red[i] for i in \
                             np.arange(len(eeRcadMO_first_valid_area_red))]

eeRcadMO_area_grn_non_dim = [eeRcadMO_list[i]['Area of cell 2 (Green) (px**2)'] / \
                             eeRcadMO_first_valid_area_grn[i] for i in \
                             np.arange(len(eeRcadMO_first_valid_area_grn))]

eeRcadMO_contact_diam_non_dim = pd.concat(eeRcadMO_contact_diam_non_dim)
eeRcadMO_contact_diam_non_dim_df['Contact Surface Length (px)'] = eeRcadMO_contact_diam_non_dim

eeRcadMO_area_cellpair_non_dim = pd.concat(eeRcadMO_area_cellpair_non_dim)
eeRcadMO_contact_diam_non_dim_df['Area of the cell pair (px**2)'] = eeRcadMO_area_cellpair_non_dim

eeRcadMO_area_red_non_dim = pd.concat(eeRcadMO_area_red_non_dim)
eeRcadMO_contact_diam_non_dim_df['Area of cell 1 (Red) (px**2)'] = eeRcadMO_area_red_non_dim

eeRcadMO_area_grn_non_dim = pd.concat(eeRcadMO_area_grn_non_dim)
eeRcadMO_contact_diam_non_dim_df['Area of cell 2 (Green) (px**2)'] = eeRcadMO_area_grn_non_dim

eeRcadMO_contact_diam_non_dim_df = eeRcadMO_contact_diam_non_dim_df.rename(index=str, columns={'Contact Surface Length (px)': 'Contact Surface Length (N.D.)'})
eeRcadMO_contact_diam_non_dim_df = eeRcadMO_contact_diam_non_dim_df.rename(index=str, columns={'Area of the cell pair (px**2)': 'Area of the cell pair (N.D.)'})
eeRcadMO_contact_diam_non_dim_df = eeRcadMO_contact_diam_non_dim_df.rename(index=str, columns={'Area of cell 1 (Red) (px**2)': 'Area of cell 1 (Red) (N.D.)'})
eeRcadMO_contact_diam_non_dim_df = eeRcadMO_contact_diam_non_dim_df.rename(index=str, columns={'Area of cell 2 (Green) (px**2)': 'Area of cell 2 (Green) (N.D.)'})

eeRcadMO_contact_diam_non_dim_df_list = [eeRcadMO_contact_diam_non_dim_df[eeRcadMO_contact_diam_non_dim_df['Source_File'] == x] for x in np.unique(eeRcadMO_contact_diam_non_dim_df['Source_File'])]




#eeRCcadMO_first_valid_area = [(df['Area of the cell pair (px**2)'].dropna() / 2.).mean() for df in eeRCcadMO_list]
#eeRCcadMO_first_valid_area_red = [(df['Area of cell 1 (Red) (px**2)'].dropna()).mean() for df in eeRCcadMO_list]
#eeRCcadMO_first_valid_area_grn = [(df['Area of cell 2 (Green) (px**2)'].dropna()).mean() for df in eeRCcadMO_list]

eeRCcadMO_first_valid_area = [(df['Area of the cell pair (px**2)'].dropna() / 2.).iloc[0] for df in eeRCcadMO_list]
eeRCcadMO_first_valid_area_red = [(df['Area of cell 1 (Red) (px**2)'].dropna()).iloc[0] for df in eeRCcadMO_list]
eeRCcadMO_first_valid_area_grn = [(df['Area of cell 2 (Green) (px**2)'].dropna()).iloc[0] for df in eeRCcadMO_list]


eeRCcadMO_first_valid_radius = [np.sqrt(area / np.pi) for area in eeRCcadMO_first_valid_area]
for i,val in enumerate(eeRCcadMO_first_valid_radius):
    eeRCcadMO_list[i]['first_valid_radius'] = val
    eeRCcadMO_list_original[i]['first_valid_radius'] = val    
    # eeRCcadMO_list[i]['first_valid_radius_red'] = eeRCcadMO_first_valid_radius_red[i]
    # eeRCcadMO_list[i]['first_valid_radius_grn'] = eeRCccadMO_first_valid_radius_grn[i]
eeRCcadMO_first_valid_radius_sf = pd.DataFrame({'first_valid_radius (um)':np.array(eeRCcadMO_first_valid_radius) * um_per_pixel, 'Source_File':[x.Source_File.iloc[-1] for x in eeRCcadMO_list]})


eeRCcadMO_contact_diam_non_dim_df = eeRCcadMO_list_long.copy()
eeRCcadMO_contact_diam_non_dim = [eeRCcadMO_list[i]['Contact Surface Length (px)'] / \
                           eeRCcadMO_first_valid_radius[i] for i in \
                           np.arange(len(eeRCcadMO_first_valid_radius))]

eeRCcadMO_area_cellpair_non_dim = [eeRCcadMO_list[i]['Area of the cell pair (px**2)'] / \
                                  eeRCcadMO_first_valid_area[i] for i in \
                                  np.arange(len(eeRCcadMO_first_valid_area))]

eeRCcadMO_area_red_non_dim = [eeRCcadMO_list[i]['Area of cell 1 (Red) (px**2)'] / \
                             eeRCcadMO_first_valid_area_red[i] for i in \
                             np.arange(len(eeRCcadMO_first_valid_area_red))]

eeRCcadMO_area_grn_non_dim = [eeRCcadMO_list[i]['Area of cell 2 (Green) (px**2)'] / \
                             eeRCcadMO_first_valid_area_grn[i] for i in \
                             np.arange(len(eeRCcadMO_first_valid_area_grn))]

eeRCcadMO_contact_diam_non_dim = pd.concat(eeRCcadMO_contact_diam_non_dim)
eeRCcadMO_contact_diam_non_dim_df['Contact Surface Length (px)'] = eeRCcadMO_contact_diam_non_dim

eeRCcadMO_area_cellpair_non_dim = pd.concat(eeRCcadMO_area_cellpair_non_dim)
eeRCcadMO_contact_diam_non_dim_df['Area of the cell pair (px**2)'] = eeRCcadMO_area_cellpair_non_dim

eeRCcadMO_area_red_non_dim = pd.concat(eeRCcadMO_area_red_non_dim)
eeRCcadMO_contact_diam_non_dim_df['Area of cell 1 (Red) (px**2)'] = eeRCcadMO_area_red_non_dim

eeRCcadMO_area_grn_non_dim = pd.concat(eeRCcadMO_area_grn_non_dim)
eeRCcadMO_contact_diam_non_dim_df['Area of cell 2 (Green) (px**2)'] = eeRCcadMO_area_grn_non_dim

eeRCcadMO_contact_diam_non_dim_df = eeRCcadMO_contact_diam_non_dim_df.rename(index=str, columns={'Contact Surface Length (px)': 'Contact Surface Length (N.D.)'})
eeRCcadMO_contact_diam_non_dim_df = eeRCcadMO_contact_diam_non_dim_df.rename(index=str, columns={'Area of the cell pair (px**2)': 'Area of the cell pair (N.D.)'})
eeRCcadMO_contact_diam_non_dim_df = eeRCcadMO_contact_diam_non_dim_df.rename(index=str, columns={'Area of cell 1 (Red) (px**2)': 'Area of cell 1 (Red) (N.D.)'})
eeRCcadMO_contact_diam_non_dim_df = eeRCcadMO_contact_diam_non_dim_df.rename(index=str, columns={'Area of cell 2 (Green) (px**2)': 'Area of cell 2 (Green) (N.D.)'})

eeRCcadMO_contact_diam_non_dim_df_list = [eeRCcadMO_contact_diam_non_dim_df[eeRCcadMO_contact_diam_non_dim_df['Source_File'] == x] for x in np.unique(eeRCcadMO_contact_diam_non_dim_df['Source_File'])]


#nn_first_valid_area = [(df['Area of the cell pair (px**2)'].dropna() / 2.).mean() for df in nn_list]
#nn_first_valid_area_red = [(df['Area of cell 1 (Red) (px**2)'].dropna()).mean() for df in nn_list]
#nn_first_valid_area_grn = [(df['Area of cell 2 (Green) (px**2)'].dropna()).mean() for df in nn_list]

nn_first_valid_area = [(df['Area of the cell pair (px**2)'].dropna() / 2.).iloc[0] for df in nn_list]
nn_first_valid_area_red = [(df['Area of cell 1 (Red) (px**2)'].dropna()).iloc[0] for df in nn_list]
nn_first_valid_area_grn = [(df['Area of cell 2 (Green) (px**2)'].dropna()).iloc[0] for df in nn_list]


nn_first_valid_radius = [np.sqrt(area / np.pi) for area in nn_first_valid_area]
for i,val in enumerate(nn_first_valid_radius):
    nn_list[i]['first_valid_radius'] = val
    nn_list_original[i]['first_valid_radius'] = val
    # nn_list[i]['first_valid_radius_red'] = nn_first_valid_radius_red[i]
    # nn_list[i]['first_valid_radius_grn'] = nn_first_valid_radius_grn[i]
nn_first_valid_radius_sf = pd.DataFrame({'first_valid_radius (um)':np.array(nn_first_valid_radius) * um_per_pixel_nn, 'Source_File':[x.Source_File.iloc[-1] for x in nn_list]})


nn_contact_diam_non_dim_df = nn_list_long.copy()
nn_contact_diam_non_dim = [nn_list[i]['Contact Surface Length (px)'] / \
                           nn_first_valid_radius[i] for i in \
                           np.arange(len(nn_first_valid_radius))]

nn_area_cellpair_non_dim = [nn_list[i]['Area of the cell pair (px**2)'] / \
                            nn_first_valid_area[i] for i in \
                            np.arange(len(nn_first_valid_area))]

nn_area_red_non_dim = [nn_list[i]['Area of cell 1 (Red) (px**2)'] / \
                       nn_first_valid_area_red[i] for i in \
                       np.arange(len(nn_first_valid_area_red))]

nn_area_grn_non_dim = [nn_list[i]['Area of cell 2 (Green) (px**2)'] / \
                       nn_first_valid_area_grn[i] for i in \
                       np.arange(len(nn_first_valid_area_grn))]

nn_contact_diam_non_dim = pd.concat(nn_contact_diam_non_dim)
nn_contact_diam_non_dim_df['Contact Surface Length (px)'] = nn_contact_diam_non_dim

nn_area_cellpair_non_dim = pd.concat(nn_area_cellpair_non_dim)
nn_contact_diam_non_dim_df['Area of the cell pair (px**2)'] = nn_area_cellpair_non_dim

nn_area_red_non_dim = pd.concat(nn_area_red_non_dim)
nn_contact_diam_non_dim_df['Area of cell 1 (Red) (px**2)'] = nn_area_red_non_dim

nn_area_grn_non_dim = pd.concat(nn_area_grn_non_dim)
nn_contact_diam_non_dim_df['Area of cell 2 (Green) (px**2)'] = nn_area_grn_non_dim

nn_contact_diam_non_dim_df = nn_contact_diam_non_dim_df.rename(index=str, columns={'Contact Surface Length (px)': 'Contact Surface Length (N.D.)'})
nn_contact_diam_non_dim_df = nn_contact_diam_non_dim_df.rename(index=str, columns={'Area of the cell pair (px**2)': 'Area of the cell pair (N.D.)'})
nn_contact_diam_non_dim_df = nn_contact_diam_non_dim_df.rename(index=str, columns={'Area of cell 1 (Red) (px**2)': 'Area of cell 1 (Red) (N.D.)'})
nn_contact_diam_non_dim_df = nn_contact_diam_non_dim_df.rename(index=str, columns={'Area of cell 2 (Green) (px**2)': 'Area of cell 2 (Green) (N.D.)'})

nn_contact_diam_non_dim_df_list = [nn_contact_diam_non_dim_df[nn_contact_diam_non_dim_df['Source_File'] == x] for x in np.unique(nn_contact_diam_non_dim_df['Source_File'])]


#ee4MU_first_valid_area = [(df['Area of the cell pair (px**2)'].dropna() / 2.).mean() for df in ee4MU_list]
#ee4MU_first_valid_area_red = [(df['Area of cell 1 (Red) (px**2)'].dropna()).mean() for df in ee4MU_list]
#ee4MU_first_valid_area_grn = [(df['Area of cell 2 (Green) (px**2)'].dropna()).mean() for df in ee4MU_list]

# ee4MU_first_valid_area = [(df['Area of the cell pair (px**2)'].dropna() / 2.).iloc[0] for df in ee4MU_list]
# ee4MU_first_valid_area_red = [(df['Area of cell 1 (Red) (px**2)'].dropna()).iloc[0] for df in ee4MU_list]
# ee4MU_first_valid_area_grn = [(df['Area of cell 2 (Green) (px**2)'].dropna()).iloc[0] for df in ee4MU_list]


# ee4MU_first_valid_radius = [np.sqrt(area / np.pi) for area in ee4MU_first_valid_area]
# for i,val in enumerate(ee4MU_first_valid_radius):
#     ee4MU_list[i]['first_valid_radius'] = val
#     ee4MU_list_original[i]['first_valid_radius'] = val
#     # ee4MU_list[i]['first_valid_radius_red'] = ee4MU_first_valid_radius_red[i]
#     # ee4MU_list[i]['first_valid_radius_grn'] = ee4MU_first_valid_radius_grn[i]

# ee4MU_contact_diam_non_dim_df = ee4MU_list_long.copy()
# ee4MU_contact_diam_non_dim = [ee4MU_list[i]['Contact Surface Length (px)'] / \
#                               ee4MU_first_valid_radius[i] for i in \
#                               np.arange(len(ee4MU_first_valid_radius))]

# ee4MU_area_cellpair_non_dim = [ee4MU_list[i]['Area of the cell pair (px**2)'] / \
#                                ee4MU_first_valid_area[i] for i in \
#                                np.arange(len(ee4MU_first_valid_area))]

# ee4MU_area_red_non_dim = [ee4MU_list[i]['Area of cell 1 (Red) (px**2)'] / \
#                           ee4MU_first_valid_area_red[i] for i in \
#                           np.arange(len(ee4MU_first_valid_area_red))]

# ee4MU_area_grn_non_dim = [ee4MU_list[i]['Area of cell 2 (Green) (px**2)'] / \
#                           ee4MU_first_valid_area_grn[i] for i in \
#                            np.arange(len(ee4MU_first_valid_area_grn))]

# ee4MU_contact_diam_non_dim = pd.concat(ee4MU_contact_diam_non_dim)
# ee4MU_contact_diam_non_dim_df['Contact Surface Length (px)'] = ee4MU_contact_diam_non_dim

# ee4MU_area_cellpair_non_dim = pd.concat(ee4MU_area_cellpair_non_dim)
# ee4MU_contact_diam_non_dim_df['Area of the cell pair (px**2)'] = ee4MU_area_cellpair_non_dim

# ee4MU_area_red_non_dim = pd.concat(ee4MU_area_red_non_dim)
# ee4MU_contact_diam_non_dim_df['Area of cell 1 (Red) (px**2)'] = ee4MU_area_red_non_dim

# ee4MU_area_grn_non_dim = pd.concat(ee4MU_area_grn_non_dim)
# ee4MU_contact_diam_non_dim_df['Area of cell 2 (Green) (px**2)'] = ee4MU_area_grn_non_dim

# ee4MU_contact_diam_non_dim_df = ee4MU_contact_diam_non_dim_df.rename(index=str, columns={'Contact Surface Length (px)': 'Contact Surface Length (N.D.)'})
# ee4MU_contact_diam_non_dim_df = ee4MU_contact_diam_non_dim_df.rename(index=str, columns={'Area of the cell pair (px**2)': 'Area of the cell pair (N.D.)'})
# ee4MU_contact_diam_non_dim_df = ee4MU_contact_diam_non_dim_df.rename(index=str, columns={'Area of cell 1 (Red) (px**2)': 'Area of cell 1 (Red) (N.D.)'})
# ee4MU_contact_diam_non_dim_df = ee4MU_contact_diam_non_dim_df.rename(index=str, columns={'Area of cell 2 (Green) (px**2)': 'Area of cell 2 (Green) (N.D.)'})

# ee4MU_contact_diam_non_dim_df_list = [ee4MU_contact_diam_non_dim_df[ee4MU_contact_diam_non_dim_df['Source_File'] == x] for x in np.unique(ee4MU_contact_diam_non_dim_df['Source_File'])]


#eedb_fix_first_valid_area = [(df['Area of the cell pair (px**2)'].dropna() / 2.).mean() for df in eedb_fix_list]
#eedb_fix_first_valid_area_red = [(df['Area of cell 1 (Red) (px**2)'].dropna()).mean() for df in eedb_fix_list]
#eedb_fix_first_valid_area_grn = [(df['Area of cell 2 (Green) (px**2)'].dropna()).mean() for df in eedb_fix_list]

# eedb_fix_first_valid_area = [(df['Area of the cell pair (px**2)'].dropna() / 2.).iloc[0] for df in eedb_fix_list]
# eedb_fix_first_valid_area_red = [(df['Area of cell 1 (Red) (px**2)'].dropna()).iloc[0] for df in eedb_fix_list]
# eedb_fix_first_valid_area_grn = [(df['Area of cell 2 (Green) (px**2)'].dropna()).iloc[0] for df in eedb_fix_list]

eedb_fix_first_valid_area = [(df['Area of the cell pair (px**2)'] / 2.).iloc[0] for df in eedb_fix_list]
eedb_fix_first_valid_area_red = [(df['Area of cell 1 (Red) (px**2)']).iloc[0] for df in eedb_fix_list]
eedb_fix_first_valid_area_grn = [(df['Area of cell 2 (Green) (px**2)']).iloc[0] for df in eedb_fix_list]

eedb_fix_first_valid_radius = [np.sqrt(area / np.pi) for area in eedb_fix_first_valid_area]
for i,val in enumerate(eedb_fix_first_valid_radius):
    eedb_fix_list[i]['first_valid_radius'] = val
    eedb_fix_list_original[i]['first_valid_radius'] = val
    # eedb_fix_list[i]['first_valid_radius_red'] = eedb_fix_first_valid_radius_red[i]
    # eedb_fix_list[i]['first_valid_radius_grn'] = eedb_fix_first_valid_radius_grn[i]
eedb_fix_first_valid_radius_sf = pd.DataFrame({'first_valid_radius (um)':np.array(eedb_fix_first_valid_radius) * um_per_pixel, 'Source_File':[x.Source_File.iloc[-1] for x in eedb_fix_list]})


eedb_fix_contact_diam_non_dim_df = eedb_fix_list_long.copy()
eedb_fix_contact_diam_non_dim = [eedb_fix_list[i]['Contact Surface Length (px)'] / \
                                 eedb_fix_first_valid_radius[i] for i in \
                                 np.arange(len(eedb_fix_first_valid_radius))]

eedb_fix_area_cellpair_non_dim = [eedb_fix_list[i]['Area of the cell pair (px**2)'] / \
                                  eedb_fix_first_valid_area[i] for i in \
                                  np.arange(len(eedb_fix_first_valid_area))]

eedb_fix_area_red_non_dim = [eedb_fix_list[i]['Area of cell 1 (Red) (px**2)'] / \
                             eedb_fix_first_valid_area_red[i] for i in \
                             np.arange(len(eedb_fix_first_valid_area_red))]

eedb_fix_area_grn_non_dim = [eedb_fix_list[i]['Area of cell 2 (Green) (px**2)'] / \
                             eedb_fix_first_valid_area_grn[i] for i in \
                             np.arange(len(eedb_fix_first_valid_area_grn))]

eedb_fix_contact_diam_non_dim = pd.concat(eedb_fix_contact_diam_non_dim)
eedb_fix_contact_diam_non_dim_df['Contact Surface Length (px)'] = eedb_fix_contact_diam_non_dim

eedb_fix_area_cellpair_non_dim = pd.concat(eedb_fix_area_cellpair_non_dim)
eedb_fix_contact_diam_non_dim_df['Area of the cell pair (px**2)'] = eedb_fix_area_cellpair_non_dim

eedb_fix_area_red_non_dim = pd.concat(eedb_fix_area_red_non_dim)
eedb_fix_contact_diam_non_dim_df['Area of cell 1 (Red) (px**2)'] = eedb_fix_area_red_non_dim

eedb_fix_area_grn_non_dim = pd.concat(eedb_fix_area_grn_non_dim)
eedb_fix_contact_diam_non_dim_df['Area of cell 2 (Green) (px**2)'] = eedb_fix_area_grn_non_dim

eedb_fix_contact_diam_non_dim_df = eedb_fix_contact_diam_non_dim_df.rename(index=str, columns={'Contact Surface Length (px)': 'Contact Surface Length (N.D.)'})
eedb_fix_contact_diam_non_dim_df = eedb_fix_contact_diam_non_dim_df.rename(index=str, columns={'Area of the cell pair (px**2)': 'Area of the cell pair (N.D.)'})
eedb_fix_contact_diam_non_dim_df = eedb_fix_contact_diam_non_dim_df.rename(index=str, columns={'Area of cell 1 (Red) (px**2)': 'Area of cell 1 (Red) (N.D.)'})
eedb_fix_contact_diam_non_dim_df = eedb_fix_contact_diam_non_dim_df.rename(index=str, columns={'Area of cell 2 (Green) (px**2)': 'Area of cell 2 (Green) (N.D.)'})

eedb_fix_contact_diam_non_dim_df_list = [eedb_fix_contact_diam_non_dim_df[eedb_fix_contact_diam_non_dim_df['Source_File'] == x] for x in np.unique(eedb_fix_contact_diam_non_dim_df['Source_File'])]

# Non-dimensionalize the touching two circles test:
two_circ_list_original_nd = two_circ_list[0]['Contact Surface Length (px)'] / (two_circ_list_original[0]['Cell Pair Number'] + 9) #The radius is equal to the cell pair number + 9
two_circ_list_original_nd_df = two_circ_list[0].copy()
two_circ_list_original_nd_df['Contact Surface Length (px)'] = two_circ_list_original_nd
two_circ_list_original_nd_df.rename(columns={'Contact Surface Length (px)': 'Contact Surface Length (N.D.)'}, inplace=True)

# Non-dimensionalize the cell simulation data:
radius27px_first_valid_area = [(df['Area of the cell pair (px**2)'].dropna() / 2.).iloc[0] for df in radius27px_cell_sim_list_original]
radius27px_first_valid_area_red = [(df['Area of cell 1 (Red) (px**2)'].dropna()).iloc[0] for df in radius27px_cell_sim_list_original]
radius27px_first_valid_area_grn = [(df['Area of cell 2 (Green) (px**2)'].dropna()).iloc[0] for df in radius27px_cell_sim_list_original]

radius27px_first_valid_radius = [np.sqrt(area / np.pi) for area in radius27px_first_valid_area]
for i,val in enumerate(radius27px_first_valid_radius):
    radius27px_cell_sim_list_original[i]['first_valid_radius'] = val
    # eedb_fix_list[i]['first_valid_radius_red'] = eedb_fix_first_valid_radius_red[i]
    # eedb_fix_list[i]['first_valid_radius_grn'] = eedb_fix_first_valid_radius_grn[i]
radius27px_first_valid_radius_sf = pd.DataFrame({'first_valid_radius (um)':np.array(radius27px_first_valid_radius) * um_per_pixel, 'Source_File':[x.Source_File.iloc[-1] for x in radius27px_cell_sim_list_original]})


radius27px_cell_sim_list_original_nd_df = radius27px_cell_sim_list_long.copy()
radius27px_contact_diam_non_dim = [radius27px_cell_sim_list_original[i]['Contact Surface Length (px)'] / \
                                   radius27px_first_valid_radius[i] for i in \
                                   np.arange(len(radius27px_first_valid_radius))]

radius27px_area_cellpair_non_dim = [radius27px_cell_sim_list_original[i]['Area of the cell pair (px**2)'] / \
                                    radius27px_first_valid_area[i] for i in \
                                    np.arange(len(radius27px_first_valid_area))]

radius27px_area_red_non_dim = [radius27px_cell_sim_list_original[i]['Area of cell 1 (Red) (px**2)'] / \
                               radius27px_first_valid_area_red[i] for i in \
                               np.arange(len(radius27px_first_valid_area_red))]

radius27px_area_grn_non_dim = [radius27px_cell_sim_list_original[i]['Area of cell 2 (Green) (px**2)'] / \
                               radius27px_first_valid_area_grn[i] for i in \
                               np.arange(len(radius27px_first_valid_area_grn))]

radius27px_contact_diam_non_dim = pd.concat(radius27px_contact_diam_non_dim)
radius27px_cell_sim_list_original_nd_df['Contact Surface Length (px)'] = radius27px_contact_diam_non_dim

radius27px_area_cellpair_non_dim = pd.concat(radius27px_area_cellpair_non_dim)
radius27px_cell_sim_list_original_nd_df['Area of the cell pair (px**2)'] = radius27px_area_cellpair_non_dim

radius27px_area_red_non_dim = pd.concat(radius27px_area_red_non_dim)
radius27px_cell_sim_list_original_nd_df['Area of cell 1 (Red) (px**2)'] = radius27px_area_red_non_dim

radius27px_area_grn_non_dim = pd.concat(radius27px_area_grn_non_dim)
radius27px_cell_sim_list_original_nd_df['Area of cell 2 (Green) (px**2)'] = radius27px_area_grn_non_dim

radius27px_cell_sim_list_original_nd_df = radius27px_cell_sim_list_original_nd_df.rename(index=str, columns={'Contact Surface Length (px)': 'Contact Surface Length (N.D.)'})
radius27px_cell_sim_list_original_nd_df = radius27px_cell_sim_list_original_nd_df.rename(index=str, columns={'Area of the cell pair (px**2)': 'Area of the cell pair (N.D.)'})
radius27px_cell_sim_list_original_nd_df = radius27px_cell_sim_list_original_nd_df.rename(index=str, columns={'Area of cell 1 (Red) (px**2)': 'Area of cell 1 (Red) (N.D.)'})
radius27px_cell_sim_list_original_nd_df = radius27px_cell_sim_list_original_nd_df.rename(index=str, columns={'Area of cell 2 (Green) (px**2)': 'Area of cell 2 (Green) (N.D.)'})

# radius27px_cell_sim_list_original_nd_df_list = [radius27px_cell_sim_list_original_nd_df[radius27px_cell_sim_list_original_nd_df['Source_File'] == x] for x in np.unique(radius27px_cell_sim_list_original_nd_df['Source_File'])]

#radius27px_cell_sim_list_original_nd = radius27px_cell_sim_list_original[0]['Contact Surface Length (px)'] / np.sqrt(radius27px_cell_sim_list_original[0]['Area of cell 1 (Red) (px**2)'][0]/np.pi)

#radius27px_cell_sim_list_original_nd_df = radius27px_cell_sim_list_original[0].copy()
#radius27px_cell_sim_list_original_nd_df['Contact Surface Length (px)'] = radius27px_cell_sim_list_original_nd
#radius27px_cell_sim_list_original_nd_df.rename(columns={'Contact Surface Length (px)': 'Contact Surface Length (N.D.)'}, inplace=True)

radius27px_cell_sim_list_original_nd_df_list = [df for grp_nm,df in radius27px_cell_sim_list_original_nd_df.groupby('Experimental_Condition')]
radius27px0deg_cell_sim_list_original_nd_df = radius27px_cell_sim_list_original_nd_df_list[0]
radius27px15deg_cell_sim_list_original_nd_df = radius27px_cell_sim_list_original_nd_df_list[1]
radius27px30deg_cell_sim_list_original_nd_df = radius27px_cell_sim_list_original_nd_df_list[2]
radius27px45deg_cell_sim_list_original_nd_df = radius27px_cell_sim_list_original_nd_df_list[3]






### Trying to fit logistic curve to each timelapse:
### for ecto-ecto:
ee_list_dropna = [x.dropna(axis=0, subset=['Contact Surface Length (px)']) for x in ee_list]

# The following samples decreased their adhesiveness later in the timelapse,
# so I am going to remove them during logistic curve fitting:
files_with_dec_adh = ['2017-08-10 xl wt ecto-fdx wt ecto-rda 20.5C crop2.csv',
                      '2018-03-15 xl wt-ecto-fdx wt-ecto-rdx 20C #2 crop2.csv',
                      '2018-06-27 xl wt-ecto-fdx wt-ecto-rdx 19C crop2.csv',
                      '2018-06-28 xl wt-ecto-fdx wt-ecto-rdx 19C crop1.1.csv',
                      '2018-06-28 xl wt-ecto-fdx wt-ecto-rdx 19C crop6.csv',
                      '2018-08-10 xl wt-ecto-fdx wt-ecto-rdx 21C crop4.csv',
                      '2018-08-10 xl wt-ecto-fdx wt-ecto-rdx 21C crop7.csv']
i = list()
for j,k in enumerate(ee_list_dropna):
    if k['Source_File'].iloc[0] in files_with_dec_adh:
        i.append(j)
#i = [10, 30, 33, 36, 41, 46, 49]
x = [138, 65, 212, 176, 130, 96, 110]
for j,k in enumerate(i):
    ee_list_dropna[k].drop(axis=0, index=ee_list_dropna[k][ee_list_dropna[k]['Cell Pair Number'] >= x[j]].index, inplace=True)
# Also drop zero values from Contact Surface Length so that the logistic fit
# better reflects the low plateau:
for x in ee_list_dropna:
    x.drop(axis=0, index=x[x['Contact Surface Length (px)'] == 0].index, inplace=True)

ee_contact_diam_non_dim_df_list_dropna = [x.dropna(axis=0, subset=['Contact Surface Length (N.D.)']) for x in ee_contact_diam_non_dim_df_list]
# The following samples decreased their adhesiveness later in the timelapse,
# so I am going to remove them during logistic curve fitting:
#i, x = [10, 30, 33, 36, 41, 46, 49], [138, 65, 212, 176, 130, 96, 110]
x = [138, 65, 212, 176, 130, 96, 110]
for j,k in enumerate(i):
    ee_contact_diam_non_dim_df_list_dropna[k].drop(axis=0, index=ee_contact_diam_non_dim_df_list_dropna[k][ee_contact_diam_non_dim_df_list_dropna[k]['Cell Pair Number'] >= x[j]].index, inplace=True)
# Also, try removing zero values, as they interfere with determining the low
# plateau:
for x in ee_contact_diam_non_dim_df_list_dropna:
    x.drop(axis=0, index=x[x['Contact Surface Length (N.D.)'] == 0].index, inplace=True)


ee_indiv_logistic_cont_fits_mesg = []
ee_indiv_logistic_cont_popt_pcov = []
ee_indiv_logistic_x_cont = []
ee_indiv_logistic_y_cont = []
ee_indiv_logistic_cont = []
# For logistic function where nu=1 (initial conditions and bounds are defined
# after function definitions at the top of the document:

for x in ee_list_dropna:
    with warnings.catch_warnings():
        warnings.simplefilter('error', OptimizeWarning)
        try:
            ee_popt_contact, ee_pcov_contact = curve_fit(logistic, x['Time (minutes)'], x['Contact Surface Length (px)'], p0 = (*p0_cont,), bounds=bounds_cont)
            ee_x_contact = np.linspace(fit_time_min, fit_time_max, step_size)
            ee_y_contact = logistic(ee_x_contact, *ee_popt_contact)
                   
            ee_indiv_logistic_cont_fits_mesg.append(("Success!", x['Source_File'].iloc[0]))
            ee_indiv_logistic_cont_popt_pcov.append((ee_popt_contact, ee_pcov_contact, x['Source_File'].iloc[0]))
            ee_indiv_logistic_x_cont.append(ee_x_contact)
            ee_indiv_logistic_y_cont.append(ee_y_contact)
            ee_indiv_logistic_cont.append(("Success", x['Source_File'].iloc[0], bounds_cont, *p0_cont, *ee_popt_contact, ee_pcov_contact, ee_x_contact, ee_y_contact))
        except (RuntimeError, ValueError, OptimizeWarning) as e:
            ee_popt_contact, ee_pcov_contact = np.asarray([np.float('nan')]  * len(p0_cont)), np.asarray([np.float('nan')])
            ee_x_contact = np.linspace(fit_time_min, fit_time_max, step_size)
            ee_y_contact = [np.float('nan')] * len(ee_x_contact)
            
            ee_indiv_logistic_cont_fits_mesg.append((e, x['Source_File'].iloc[0]))
            rt_err_popt_placeholder = np.empty((4,))
            rt_err_popt_placeholder[:] = np.nan
            rt_err_pcov_placeholder = np.empty((4,4))
            rt_err_pcov_placeholder[:] = np.nan
            ee_indiv_logistic_cont_popt_pcov.append((rt_err_popt_placeholder, rt_err_pcov_placeholder, x['Source_File'].iloc[0]))
            ee_indiv_logistic_x_cont.append([np.float('nan')])
            ee_indiv_logistic_y_cont.append([np.float('nan')])
            ee_indiv_logistic_cont.append((e, x['Source_File'].iloc[0], bounds_cont, *p0_cont, *ee_popt_contact, ee_pcov_contact, ee_x_contact, ee_y_contact))
# For logistic function where nu=1:
ee_indiv_logistic_cont_df = pd.DataFrame(data=ee_indiv_logistic_cont, columns=['mesg', 'Source_File', 'bounds', 'p0_x2', 'p0_k2', 'p0_high_plateau', 'p0_mid_plateau', 'popt_x2', 'popt_k2', 'popt_high_plateau', 'popt_mid_plateau', 'pcov', 'logistic_x', 'logistic_y'])

ee_indiv_logistic_angle_fits_mesg = []
ee_indiv_logistic_angle_popt_pcov = []
ee_indiv_logistic_x_angle = []
ee_indiv_logistic_y_angle = []
ee_indiv_logistic_angle = []
# For logistic function where nu=1 (initial conditions and bounds are defined
# after function definitions at the top of the document:

for x in ee_list_dropna:
    with warnings.catch_warnings():
        warnings.simplefilter('error', OptimizeWarning)
        try:       
            ee_popt_angle, ee_pcov_angle = curve_fit(logistic, x['Time (minutes)'], x['Average Angle (deg)'], p0 = (*p0_angle,), bounds=bounds_angle)
            ee_x_angle = np.linspace(fit_time_min, fit_time_max, step_size)
            ee_y_angle = logistic(ee_x_angle, *ee_popt_angle)        
            
            ee_indiv_logistic_angle_fits_mesg.append(("Success!", x['Source_File'].iloc[0]))
            ee_indiv_logistic_angle_popt_pcov.append((ee_popt_angle, ee_pcov_angle, x['Source_File'].iloc[0]))
            ee_indiv_logistic_x_angle.append(ee_x_angle)
            ee_indiv_logistic_y_angle.append(ee_y_angle)
            ee_indiv_logistic_angle.append(("Success", x['Source_File'].iloc[0], bounds_angle, *p0_angle, *ee_popt_angle, ee_pcov_angle, ee_x_angle, ee_y_angle))
        except (RuntimeError, ValueError, OptimizeWarning) as e:
            ee_popt_angle, ee_pcov_angle = np.asarray([np.float('nan')]  * len(p0_angle)), np.asarray([np.float('nan')])
            ee_x_angle = np.linspace(fit_time_min, fit_time_max, step_size)
            ee_y_angle = [np.float('nan')] * len(ee_x_angle)
            
            ee_indiv_logistic_angle_fits_mesg.append((e, x['Source_File'].iloc[0]))
            rt_err_popt_placeholder = np.empty((4,))
            rt_err_popt_placeholder[:] = np.nan
            rt_err_pcov_placeholder = np.empty((4,4))
            rt_err_pcov_placeholder[:] = np.nan
            ee_indiv_logistic_angle_popt_pcov.append((rt_err_popt_placeholder, rt_err_pcov_placeholder, x['Source_File'].iloc[0]))
            ee_indiv_logistic_x_angle.append([np.float('nan')])
            ee_indiv_logistic_y_angle.append([np.float('nan')])
            ee_indiv_logistic_angle.append((e, x['Source_File'].iloc[0], bounds_angle, *p0_angle, *ee_popt_angle, ee_pcov_angle, ee_x_angle, ee_y_angle))
# For logistic function where nu=1:
ee_indiv_logistic_angle_df = pd.DataFrame(data=ee_indiv_logistic_angle, columns=['mesg', 'Source_File', 'bounds', 'p0_x2', 'p0_k2', 'p0_high_plateau', 'p0_mid_plateau', 'popt_x2', 'popt_k2', 'popt_high_plateau', 'popt_mid_plateau', 'pcov', 'logistic_x', 'logistic_y'])


ee_indiv_logistic_contnd_fits_mesg = []
ee_indiv_logistic_contnd_popt_pcov = []
ee_indiv_logistic_x_contnd = []
ee_indiv_logistic_y_contnd = []
ee_indiv_logistic_contnd = []
# For logistic function where nu=1 (initial conditions and bounds are defined
# after function definitions at the top of the document:

for x in ee_contact_diam_non_dim_df_list_dropna:
    with warnings.catch_warnings():
        warnings.simplefilter('error', OptimizeWarning)
        try:
            ee_popt_contnd, ee_pcov_contnd = curve_fit(logistic, x['Time (minutes)'], x['Contact Surface Length (N.D.)'], p0 = (*p0_contnd,), bounds=bounds_contnd)
            ee_x_contnd = np.linspace(fit_time_min, fit_time_max, step_size)
            ee_y_contnd = logistic(ee_x_contnd, *ee_popt_contnd) 
            
            ee_indiv_logistic_contnd_fits_mesg.append(("Success!", x['Source_File'].iloc[0]))
            ee_indiv_logistic_contnd_popt_pcov.append((ee_popt_contnd, ee_pcov_contnd, x['Source_File'].iloc[0]))
            ee_indiv_logistic_x_contnd.append(ee_x_contnd)
            ee_indiv_logistic_y_contnd.append(ee_y_contnd)
            ee_indiv_logistic_contnd.append(("Success", x['Source_File'].iloc[0], bounds_contnd, *p0_contnd, *ee_popt_contnd, ee_pcov_contnd, ee_x_contnd, ee_y_contnd))
        except (RuntimeError, ValueError, OptimizeWarning) as e:
            ee_popt_contnd, ee_pcov_contnd = np.asarray([np.float('nan')] * len(p0_contnd)), np.asarray([np.float('nan')])
            ee_x_contnd = np.linspace(fit_time_min, fit_time_max, step_size)
            ee_y_contnd = [np.float('nan')] * len(ee_x_contnd)
            
            ee_indiv_logistic_contnd_fits_mesg.append((e, x['Source_File'].iloc[0]))
            rt_err_popt_placeholder = np.empty((4,))
            rt_err_popt_placeholder[:] = np.nan
            rt_err_pcov_placeholder = np.empty((4,4))
            rt_err_pcov_placeholder[:] = np.nan
            ee_indiv_logistic_contnd_popt_pcov.append((rt_err_popt_placeholder, rt_err_pcov_placeholder, x['Source_File'].iloc[0]))
            ee_indiv_logistic_x_contnd.append([np.float('nan')])
            ee_indiv_logistic_y_contnd.append([np.float('nan')])
            ee_indiv_logistic_contnd.append((e, x['Source_File'].iloc[0], bounds_contnd, *p0_contnd, *ee_popt_contnd, ee_pcov_contnd, ee_x_contnd, ee_y_contnd))
# For logistic function where nu=1:
ee_indiv_logistic_contnd_df = pd.DataFrame(data=ee_indiv_logistic_contnd, columns=['mesg', 'Source_File', 'bounds', 'p0_x2', 'p0_k2', 'p0_high_plateau', 'p0_mid_plateau', 'popt_x2', 'popt_k2', 'popt_high_plateau', 'popt_mid_plateau', 'pcov', 'logistic_x', 'logistic_y'])







### Trying to fit individual logistic functions to each timelapse:
### For meso-meso:
mm_list_dropna = [x.dropna(axis=0, subset=['Contact Surface Length (px)']) for x in mm_list]

# Also drop zero values from Contact Surface Length so that the logistic fit
# better reflects the low plateau:
for x in mm_list_dropna:
    x.drop(axis=0, index=x[x['Contact Surface Length (px)'] == 0].index, inplace=True)

mm_contact_diam_non_dim_df_list_dropna = [x.dropna(axis=0, subset=['Contact Surface Length (N.D.)']) for x in mm_contact_diam_non_dim_df_list]
# Also, try removing zero values, as they interfere with determining the low
# plateau:
for x in mm_contact_diam_non_dim_df_list_dropna:
    x.drop(axis=0, index=x[x['Contact Surface Length (N.D.)'] == 0].index, inplace=True)


# Now try to fit logistic functions to each timelapse in meso-meso data:
mm_indiv_logistic_cont_fits_mesg = []
mm_indiv_logistic_cont_popt_pcov = []
mm_indiv_logistic_x_cont = []
mm_indiv_logistic_y_cont = []
mm_indiv_logistic_cont = []
# For logistic function where nu=1:
#p0_cont = (10,1,50, 0)
#bounds_cont = ([-50,0,30,0], [100,100,100,30])

for x in mm_list_dropna:
    with warnings.catch_warnings():
        warnings.simplefilter('error', OptimizeWarning)
        try:
            mm_popt_contact, mm_pcov_contact = curve_fit(logistic, x['Time (minutes)'], x['Contact Surface Length (px)'], p0 = (*p0_cont,), bounds=bounds_cont)
            mm_x_contact = np.linspace(fit_time_min, fit_time_max, step_size)
            mm_y_contact = logistic(mm_x_contact, *mm_popt_contact)
                   
            mm_indiv_logistic_cont_fits_mesg.append(("Success!", x['Source_File'].iloc[0]))
            mm_indiv_logistic_cont_popt_pcov.append((mm_popt_contact, mm_pcov_contact, x['Source_File'].iloc[0]))
            mm_indiv_logistic_x_cont.append(mm_x_contact)
            mm_indiv_logistic_y_cont.append(mm_y_contact)
            mm_indiv_logistic_cont.append(("Success", x['Source_File'].iloc[0], bounds_cont, *p0_cont, *mm_popt_contact, mm_pcov_contact, mm_x_contact, mm_y_contact))
        except (RuntimeError, ValueError, OptimizeWarning) as e:
            mm_popt_contact, mm_pcov_contact = np.asarray([np.float('nan')]  * len(p0_cont)), np.asarray([np.float('nan')])
            mm_x_contact = np.linspace(fit_time_min, fit_time_max, step_size)
            mm_y_contact = [np.float('nan')] * len(mm_x_contact)
            
            mm_indiv_logistic_cont_fits_mesg.append((e, x['Source_File'].iloc[0]))
            rt_err_popt_placeholder = np.empty((4,))
            rt_err_popt_placeholder[:] = np.nan
            rt_err_pcov_placeholder = np.empty((4,4))
            rt_err_pcov_placeholder[:] = np.nan
            mm_indiv_logistic_cont_popt_pcov.append((rt_err_popt_placeholder, rt_err_pcov_placeholder, x['Source_File'].iloc[0]))
            mm_indiv_logistic_x_cont.append([np.float('nan')])
            mm_indiv_logistic_y_cont.append([np.float('nan')])
            mm_indiv_logistic_cont.append((e, x['Source_File'].iloc[0], bounds_cont, *p0_cont, *mm_popt_contact, mm_pcov_contact, mm_x_contact, mm_y_contact))
# For logistic function where nu=1:
mm_indiv_logistic_cont_df = pd.DataFrame(data=mm_indiv_logistic_cont, columns=['mesg', 'Source_File', 'bounds', 'p0_x2', 'p0_k2', 'p0_high_plateau', 'p0_mid_plateau', 'popt_x2', 'popt_k2', 'popt_high_plateau', 'popt_mid_plateau', 'pcov', 'logistic_x', 'logistic_y'])

mm_indiv_logistic_angle_fits_mesg = []
mm_indiv_logistic_angle_popt_pcov = []
mm_indiv_logistic_x_angle = []
mm_indiv_logistic_y_angle = []
mm_indiv_logistic_angle = []
# For logistic function where nu=1:
#p0_angle = (10,1,120,0)
#bounds_angle = ([-50,0,60,0], [100,100, 180,60])

for x in mm_list_dropna:
    with warnings.catch_warnings():
        warnings.simplefilter('error', OptimizeWarning)
        try:       
            mm_popt_angle, mm_pcov_angle = curve_fit(logistic, x['Time (minutes)'], x['Average Angle (deg)'], p0 = (*p0_angle,), bounds=bounds_angle)
            mm_x_angle = np.linspace(fit_time_min, fit_time_max, step_size)
            mm_y_angle = logistic(mm_x_angle, *mm_popt_angle)        
            
            mm_indiv_logistic_angle_fits_mesg.append(("Success!", x['Source_File'].iloc[0]))
            mm_indiv_logistic_angle_popt_pcov.append((mm_popt_angle, mm_pcov_angle, x['Source_File'].iloc[0]))
            mm_indiv_logistic_x_angle.append(mm_x_angle)
            mm_indiv_logistic_y_angle.append(mm_y_angle)
            mm_indiv_logistic_angle.append(("Success", x['Source_File'].iloc[0], bounds_angle, *p0_angle, *mm_popt_angle, mm_pcov_angle, mm_x_angle, mm_y_angle))
        except (RuntimeError, ValueError, OptimizeWarning) as e:
            mm_popt_angle, mm_pcov_angle = np.asarray([np.float('nan')]  * len(p0_angle)), np.asarray([np.float('nan')])
            mm_x_angle = np.linspace(fit_time_min, fit_time_max, step_size)
            mm_y_angle = [np.float('nan')] * len(mm_x_angle)
            
            mm_indiv_logistic_angle_fits_mesg.append((e, x['Source_File'].iloc[0]))
            rt_err_popt_placeholder = np.empty((4,))
            rt_err_popt_placeholder[:] = np.nan
            rt_err_pcov_placeholder = np.empty((4,4))
            rt_err_pcov_placeholder[:] = np.nan
            mm_indiv_logistic_angle_popt_pcov.append((rt_err_popt_placeholder, rt_err_pcov_placeholder, x['Source_File'].iloc[0]))
            mm_indiv_logistic_x_angle.append([np.float('nan')])
            mm_indiv_logistic_y_angle.append([np.float('nan')])
            mm_indiv_logistic_angle.append((e, x['Source_File'].iloc[0], bounds_angle, *p0_angle, *mm_popt_angle, mm_pcov_angle, mm_x_angle, mm_y_angle))
# For logistic function where nu=1:
mm_indiv_logistic_angle_df = pd.DataFrame(data=mm_indiv_logistic_angle, columns=['mesg', 'Source_File', 'bounds', 'p0_x2', 'p0_k2', 'p0_high_plateau', 'p0_mid_plateau', 'popt_x2', 'popt_k2', 'popt_high_plateau', 'popt_mid_plateau', 'pcov', 'logistic_x', 'logistic_y'])


mm_indiv_logistic_contnd_fits_mesg = []
mm_indiv_logistic_contnd_popt_pcov = []
mm_indiv_logistic_x_contnd = []
mm_indiv_logistic_y_contnd = []
mm_indiv_logistic_contnd = []
# For logistic function where nu=1:
#p0_contnd = [10,1,2,0]
#bounds_contnd = ([-50,0,1,0], [100,100,3,1])

for x in mm_contact_diam_non_dim_df_list_dropna:
    with warnings.catch_warnings():
        warnings.simplefilter('error', OptimizeWarning)
        try:
            mm_popt_contnd, mm_pcov_contnd = curve_fit(logistic, x['Time (minutes)'], x['Contact Surface Length (N.D.)'], p0 = (*p0_contnd,), bounds=bounds_contnd)
            mm_x_contnd = np.linspace(fit_time_min, fit_time_max, step_size)
            mm_y_contnd = logistic(mm_x_contnd, *mm_popt_contnd) 
                    
            mm_indiv_logistic_contnd_fits_mesg.append(("Success!", x['Source_File'].iloc[0]))
            mm_indiv_logistic_contnd_popt_pcov.append((mm_popt_contnd, mm_pcov_contnd, x['Source_File'].iloc[0]))
            mm_indiv_logistic_x_contnd.append(mm_x_contnd)
            mm_indiv_logistic_y_contnd.append(mm_y_contnd)
            mm_indiv_logistic_contnd.append(("Success", x['Source_File'].iloc[0], bounds_contnd, *p0_contnd, *mm_popt_contnd, mm_pcov_contnd, mm_x_contnd, mm_y_contnd))
        except (RuntimeError, ValueError, OptimizeWarning) as e:
            mm_popt_contnd, mm_pcov_contnd = np.asarray([np.float('nan')]  * len(p0_contnd)), np.asarray([np.float('nan')])
            mm_x_contnd = np.linspace(fit_time_min, fit_time_max, step_size)
            mm_y_contnd = [np.float('nan')] * len(mm_x_contnd)
            
            mm_indiv_logistic_contnd_fits_mesg.append((e, x['Source_File'].iloc[0]))
            rt_err_popt_placeholder = np.empty((4,))
            rt_err_popt_placeholder[:] = np.nan
            rt_err_pcov_placeholder = np.empty((4,4))
            rt_err_pcov_placeholder[:] = np.nan
            mm_indiv_logistic_contnd_popt_pcov.append((rt_err_popt_placeholder, rt_err_pcov_placeholder, x['Source_File'].iloc[0]))
            mm_indiv_logistic_x_contnd.append([np.float('nan')])
            mm_indiv_logistic_y_contnd.append([np.float('nan')])
            mm_indiv_logistic_contnd.append((e, x['Source_File'].iloc[0], bounds_contnd, *p0_contnd, *mm_popt_contnd, mm_pcov_contnd, mm_x_contnd, mm_y_contnd))
# For logistic function where nu=1:
mm_indiv_logistic_contnd_df = pd.DataFrame(data=mm_indiv_logistic_contnd, columns=['mesg', 'Source_File', 'bounds', 'p0_x2', 'p0_k2', 'p0_high_plateau', 'p0_mid_plateau', 'popt_x2', 'popt_k2', 'popt_high_plateau', 'popt_mid_plateau', 'pcov', 'logistic_x', 'logistic_y'])



###############################################################################



### Trying to fit individual logistic functions to each timelapse:
### For ecto-meso:
em_list_dropna = [x.dropna(axis=0, subset=['Contact Surface Length (px)']) for x in em_list]

# Also drop zero values from Contact Surface Length so that the logistic fit
# better reflects the low plateau:
for x in em_list_dropna:
    x.drop(axis=0, index=x[x['Contact Surface Length (px)'] == 0].index, inplace=True)

em_contact_diam_non_dim_df_list_dropna = [x.dropna(axis=0, subset=['Contact Surface Length (N.D.)']) for x in em_contact_diam_non_dim_df_list]
# Also, try removing zero values, as they interfere with determining the low
# plateau:
for x in em_contact_diam_non_dim_df_list_dropna:
    x.drop(axis=0, index=x[x['Contact Surface Length (N.D.)'] == 0].index, inplace=True)


# Now try to fit logistic functions to each timelapse in ecto-meso data:
em_indiv_logistic_cont_fits_mesg = []
em_indiv_logistic_cont_popt_pcov = []
em_indiv_logistic_x_cont = []
em_indiv_logistic_y_cont = []
em_indiv_logistic_cont = []
# For logistic function where nu=1:
#p0_cont = (10,1,50, 0)
#bounds_cont = ([-50,0,30,0], [100,100,100,30])

for x in em_list_dropna:
    with warnings.catch_warnings():
        warnings.simplefilter('error', OptimizeWarning)
        try:
            em_popt_contact, em_pcov_contact = curve_fit(logistic, x['Time (minutes)'], x['Contact Surface Length (px)'], p0 = (*p0_cont_em,), bounds=bounds_cont_em)
            em_x_contact = np.linspace(fit_time_min, fit_time_max, step_size)
            em_y_contact = logistic(em_x_contact, *em_popt_contact)
                   
            em_indiv_logistic_cont_fits_mesg.append(("Success!", x['Source_File'].iloc[0]))
            em_indiv_logistic_cont_popt_pcov.append((em_popt_contact, em_pcov_contact, x['Source_File'].iloc[0]))
            em_indiv_logistic_x_cont.append(em_x_contact)
            em_indiv_logistic_y_cont.append(em_y_contact)
            em_indiv_logistic_cont.append(("Success", x['Source_File'].iloc[0], bounds_cont, *p0_cont_em, *em_popt_contact, em_pcov_contact, em_x_contact, em_y_contact))
        except (RuntimeError, ValueError, OptimizeWarning) as e:
            em_popt_contact, em_pcov_contact = np.asarray([np.float('nan')]  * len(p0_cont_em)), np.asarray([np.float('nan')])
            em_x_contact = np.linspace(fit_time_min, fit_time_max, step_size)
            em_y_contact = [np.float('nan')] * len(em_x_contact)
            
            em_indiv_logistic_cont_fits_mesg.append((e, x['Source_File'].iloc[0]))
            rt_err_popt_placeholder = np.empty((4,))
            rt_err_popt_placeholder[:] = np.nan
            rt_err_pcov_placeholder = np.empty((4,4))
            rt_err_pcov_placeholder[:] = np.nan
            em_indiv_logistic_cont_popt_pcov.append((rt_err_popt_placeholder, rt_err_pcov_placeholder, x['Source_File'].iloc[0]))
            em_indiv_logistic_x_cont.append([np.float('nan')])
            em_indiv_logistic_y_cont.append([np.float('nan')])
            em_indiv_logistic_cont.append((e, x['Source_File'].iloc[0], bounds_cont, *p0_cont_em, *em_popt_contact, em_pcov_contact, em_x_contact, em_y_contact))
# For logistic function where nu=1:
em_indiv_logistic_cont_df = pd.DataFrame(data=em_indiv_logistic_cont, columns=['mesg', 'Source_File', 'bounds', 'p0_x2', 'p0_k2', 'p0_high_plateau', 'p0_mid_plateau', 'popt_x2', 'popt_k2', 'popt_high_plateau', 'popt_mid_plateau', 'pcov', 'logistic_x', 'logistic_y'])

em_indiv_logistic_angle_fits_mesg = []
em_indiv_logistic_angle_popt_pcov = []
em_indiv_logistic_x_angle = []
em_indiv_logistic_y_angle = []
em_indiv_logistic_angle = []
# For logistic function where nu=1:
#p0_angle = (10,1,120,0)
#bounds_angle = ([-50,0,60,0], [100,100, 180,60])

for x in em_list_dropna:
    with warnings.catch_warnings():
        warnings.simplefilter('error', OptimizeWarning)
        try:       
            em_popt_angle, em_pcov_angle = curve_fit(logistic, x['Time (minutes)'], x['Average Angle (deg)'], p0 = (*p0_angle_em,), bounds=bounds_angle_em)
            em_x_angle = np.linspace(fit_time_min, fit_time_max, step_size)
            em_y_angle = logistic(em_x_angle, *em_popt_angle)        
            
            em_indiv_logistic_angle_fits_mesg.append(("Success!", x['Source_File'].iloc[0]))
            em_indiv_logistic_angle_popt_pcov.append((em_popt_angle, em_pcov_angle, x['Source_File'].iloc[0]))
            em_indiv_logistic_x_angle.append(em_x_angle)
            em_indiv_logistic_y_angle.append(em_y_angle)
            em_indiv_logistic_angle.append(("Success", x['Source_File'].iloc[0], bounds_angle, *p0_angle_em, *em_popt_angle, em_pcov_angle, em_x_angle, em_y_angle))
        except (RuntimeError, ValueError, OptimizeWarning) as e:
            em_popt_angle, em_pcov_angle = np.asarray([np.float('nan')]  * len(p0_angle_em)), np.asarray([np.float('nan')])
            em_x_angle = np.linspace(fit_time_min, fit_time_max, step_size)
            em_y_angle = [np.float('nan')] * len(em_x_angle)
            
            em_indiv_logistic_angle_fits_mesg.append((e, x['Source_File'].iloc[0]))
            rt_err_popt_placeholder = np.empty((4,))
            rt_err_popt_placeholder[:] = np.nan
            rt_err_pcov_placeholder = np.empty((4,4))
            rt_err_pcov_placeholder[:] = np.nan
            em_indiv_logistic_angle_popt_pcov.append((rt_err_popt_placeholder, rt_err_pcov_placeholder, x['Source_File'].iloc[0]))
            em_indiv_logistic_x_angle.append([np.float('nan')])
            em_indiv_logistic_y_angle.append([np.float('nan')])
            em_indiv_logistic_angle.append((e, x['Source_File'].iloc[0], bounds_angle, *p0_angle_em, *em_popt_angle, em_pcov_angle, em_x_angle, em_y_angle))
# For logistic function where nu=1:
em_indiv_logistic_angle_df = pd.DataFrame(data=em_indiv_logistic_angle, columns=['mesg', 'Source_File', 'bounds', 'p0_x2', 'p0_k2', 'p0_high_plateau', 'p0_mid_plateau', 'popt_x2', 'popt_k2', 'popt_high_plateau', 'popt_mid_plateau', 'pcov', 'logistic_x', 'logistic_y'])


em_indiv_logistic_contnd_fits_mesg = []
em_indiv_logistic_contnd_popt_pcov = []
em_indiv_logistic_x_contnd = []
em_indiv_logistic_y_contnd = []
em_indiv_logistic_contnd = []
# For logistic function where nu=1:
#p0_contnd = [10,1,2,0]
#bounds_contnd = ([-50,0,1,0], [100,100,3,1])

for x in em_contact_diam_non_dim_df_list_dropna:
    with warnings.catch_warnings():
        warnings.simplefilter('error', OptimizeWarning)
        try:
            em_popt_contnd, em_pcov_contnd = curve_fit(logistic, x['Time (minutes)'], x['Contact Surface Length (N.D.)'], p0 = (*p0_contnd_em,), bounds=bounds_contnd_em) #take away contnd?
            em_x_contnd = np.linspace(fit_time_min, fit_time_max, step_size)
            em_y_contnd = logistic(em_x_contnd, *em_popt_contnd) 
                    
            em_indiv_logistic_contnd_fits_mesg.append(("Success!", x['Source_File'].iloc[0]))
            em_indiv_logistic_contnd_popt_pcov.append((em_popt_contnd, em_pcov_contnd, x['Source_File'].iloc[0]))
            em_indiv_logistic_x_contnd.append(em_x_contnd)
            em_indiv_logistic_y_contnd.append(em_y_contnd)
            em_indiv_logistic_contnd.append(("Success", x['Source_File'].iloc[0], bounds_contnd, *p0_contnd_em, *em_popt_contnd, em_pcov_contnd, em_x_contnd, em_y_contnd))
        except (RuntimeError, ValueError, OptimizeWarning) as e:
            em_popt_contnd, em_pcov_contnd = np.asarray([np.float('nan')]  * len(p0_contnd_em)), np.asarray([np.float('nan')])
            em_x_contnd = np.linspace(fit_time_min, fit_time_max, step_size)
            em_y_contnd = [np.float('nan')] * len(em_x_contnd)
            
            em_indiv_logistic_contnd_fits_mesg.append((e, x['Source_File'].iloc[0]))
            rt_err_popt_placeholder = np.empty((4,))
            rt_err_popt_placeholder[:] = np.nan
            rt_err_pcov_placeholder = np.empty((4,4))
            rt_err_pcov_placeholder[:] = np.nan
            em_indiv_logistic_contnd_popt_pcov.append((rt_err_popt_placeholder, rt_err_pcov_placeholder, x['Source_File'].iloc[0]))
            em_indiv_logistic_x_contnd.append([np.float('nan')])
            em_indiv_logistic_y_contnd.append([np.float('nan')])
            em_indiv_logistic_contnd.append((e, x['Source_File'].iloc[0], bounds_contnd, *p0_contnd_em, *em_popt_contnd, em_pcov_contnd, em_x_contnd, em_y_contnd))
# For logistic function where nu=1:
em_indiv_logistic_contnd_df = pd.DataFrame(data=em_indiv_logistic_contnd, columns=['mesg', 'Source_File', 'bounds', 'p0_x2', 'p0_k2', 'p0_high_plateau', 'p0_mid_plateau', 'popt_x2', 'popt_k2', 'popt_high_plateau', 'popt_mid_plateau', 'pcov', 'logistic_x', 'logistic_y'])


###############################################################################
### Trying to fit individual logistic functions to each timelapse:
### For endo-endo:
nn_list_dropna = [x.dropna(axis=0, subset=['Contact Surface Length (px)']) for x in nn_list]

# Also drop zero values from Contact Surface Length so that the logistic fit
# better reflects the low plateau:
for x in nn_list_dropna:
    x.drop(axis=0, index=x[x['Contact Surface Length (px)'] == 0].index, inplace=True)

nn_contact_diam_non_dim_df_list_dropna = [x.dropna(axis=0, subset=['Contact Surface Length (N.D.)']) for x in nn_contact_diam_non_dim_df_list]
# Also, try removing zero values, as they interfere with determining the low
# plateau:
for x in nn_contact_diam_non_dim_df_list_dropna:
    x.drop(axis=0, index=x[x['Contact Surface Length (N.D.)'] == 0].index, inplace=True)


# Now try to fit logistic functions to each timelapse in endo-endo data:
nn_indiv_logistic_cont_fits_mesg = []
nn_indiv_logistic_cont_popt_pcov = []
nn_indiv_logistic_x_cont = []
nn_indiv_logistic_y_cont = []
nn_indiv_logistic_cont = []
# For logistic function where nu=1:
#p0_cont = (10,1,50, 0)
#bounds_cont = ([0,0,30,0], [100,100,100,30])

for x in nn_list_dropna:
    with warnings.catch_warnings():
        warnings.simplefilter('error', OptimizeWarning)
        try:
            nn_popt_contact, nn_pcov_contact = curve_fit(logistic, x['Time (minutes)'], x['Contact Surface Length (px)'], p0 = (*p0_cont,), bounds=bounds_cont)
            nn_x_contact = np.linspace(fit_time_min, fit_time_max, step_size)
            nn_y_contact = logistic(nn_x_contact, *nn_popt_contact)
                   
            nn_indiv_logistic_cont_fits_mesg.append(("Success!", x['Source_File'].iloc[0]))
            nn_indiv_logistic_cont_popt_pcov.append((nn_popt_contact, nn_pcov_contact, x['Source_File'].iloc[0]))
            nn_indiv_logistic_x_cont.append(nn_x_contact)
            nn_indiv_logistic_y_cont.append(nn_y_contact)
            nn_indiv_logistic_cont.append(("Success", x['Source_File'].iloc[0], bounds_cont, *p0_cont, *nn_popt_contact, nn_pcov_contact, nn_x_contact, nn_y_contact))
        except (RuntimeError, ValueError, OptimizeWarning) as e:
            nn_popt_contact, nn_pcov_contact = np.asarray([np.float('nan')]  * len(p0_cont)), np.asarray([np.float('nan')])
            nn_x_contact = np.linspace(fit_time_min, fit_time_max, step_size)
            nn_y_contact = [np.float('nan')] * len(nn_x_contact)
            
            nn_indiv_logistic_cont_fits_mesg.append((e, x['Source_File'].iloc[0]))
            rt_err_popt_placeholder = np.empty((4,))
            rt_err_popt_placeholder[:] = np.nan
            rt_err_pcov_placeholder = np.empty((4,4))
            rt_err_pcov_placeholder[:] = np.nan
            nn_indiv_logistic_cont_popt_pcov.append((rt_err_popt_placeholder, rt_err_pcov_placeholder, x['Source_File'].iloc[0]))
            nn_indiv_logistic_x_cont.append([np.float('nan')])
            nn_indiv_logistic_y_cont.append([np.float('nan')])
            nn_indiv_logistic_cont.append((e, x['Source_File'].iloc[0], bounds_cont, *p0_cont, *nn_popt_contact, nn_pcov_contact, nn_x_contact, nn_y_contact))
# For logistic function where nu=1:
nn_indiv_logistic_cont_df = pd.DataFrame(data=nn_indiv_logistic_cont, columns=['mesg', 'Source_File', 'bounds', 'p0_x2', 'p0_k2', 'p0_high_plateau', 'p0_mid_plateau', 'popt_x2', 'popt_k2', 'popt_high_plateau', 'popt_mid_plateau', 'pcov', 'logistic_x', 'logistic_y'])


nn_indiv_logistic_angle_fits_mesg = []
nn_indiv_logistic_angle_popt_pcov = []
nn_indiv_logistic_x_angle = []
nn_indiv_logistic_y_angle = []
nn_indiv_logistic_angle = []
# For logistic function where nu=1:
#p0_angle = (10,1,120,0)
#bounds_angle = ([0,0,60,0], [100,100, 180,60])

for x in nn_list_dropna:
    with warnings.catch_warnings():
        warnings.simplefilter('error', OptimizeWarning)
        try:       
            nn_popt_angle, nn_pcov_angle = curve_fit(logistic, x['Time (minutes)'], x['Average Angle (deg)'], p0 = (*p0_angle,), bounds=bounds_angle)
            nn_x_angle = np.linspace(fit_time_min, fit_time_max, step_size)
            nn_y_angle = logistic(nn_x_angle, *nn_popt_angle)        
            
            nn_indiv_logistic_angle_fits_mesg.append(("Success!", x['Source_File'].iloc[0]))
            nn_indiv_logistic_angle_popt_pcov.append((nn_popt_angle, nn_pcov_angle, x['Source_File'].iloc[0]))
            nn_indiv_logistic_x_angle.append(nn_x_angle)
            nn_indiv_logistic_y_angle.append(nn_y_angle)
            nn_indiv_logistic_angle.append(("Success", x['Source_File'].iloc[0], bounds_angle, *p0_angle, *nn_popt_angle, nn_pcov_angle, nn_x_angle, nn_y_angle))
        except (RuntimeError, ValueError, OptimizeWarning) as e:
            nn_popt_angle, nn_pcov_angle = np.asarray([np.float('nan')]  * len(p0_angle)), np.asarray([np.float('nan')])
            nn_x_angle = np.linspace(fit_time_min, fit_time_max, step_size)
            nn_y_angle = [np.float('nan')] * len(nn_x_angle)
            
            nn_indiv_logistic_angle_fits_mesg.append((e, x['Source_File'].iloc[0]))
            rt_err_popt_placeholder = np.empty((4,))
            rt_err_popt_placeholder[:] = np.nan
            rt_err_pcov_placeholder = np.empty((4,4))
            rt_err_pcov_placeholder[:] = np.nan
            nn_indiv_logistic_angle_popt_pcov.append((rt_err_popt_placeholder, rt_err_pcov_placeholder, x['Source_File'].iloc[0]))
            nn_indiv_logistic_x_angle.append([np.float('nan')])
            nn_indiv_logistic_y_angle.append([np.float('nan')])
            nn_indiv_logistic_angle.append((e, x['Source_File'].iloc[0], bounds_angle, *p0_angle, *nn_popt_angle, nn_pcov_angle, nn_x_angle, nn_y_angle))
# For logistic function where nu=1:
nn_indiv_logistic_angle_df = pd.DataFrame(data=nn_indiv_logistic_angle, columns=['mesg', 'Source_File', 'bounds', 'p0_x2', 'p0_k2', 'p0_high_plateau', 'p0_mid_plateau', 'popt_x2', 'popt_k2', 'popt_high_plateau', 'popt_mid_plateau', 'pcov', 'logistic_x', 'logistic_y'])


nn_indiv_logistic_contnd_fits_mesg = []
nn_indiv_logistic_contnd_popt_pcov = []
nn_indiv_logistic_x_contnd = []
nn_indiv_logistic_y_contnd = []
nn_indiv_logistic_contnd = []
#p0_contnd = [60,1,2,0.5]
# For logistic function where nu=1:
#p0_contnd_nn = [10,1,2,0]
#bounds_contnd_nn = ([0,0,1,0], [100,100,3,1])

for x in nn_contact_diam_non_dim_df_list_dropna:
    with warnings.catch_warnings():
        warnings.simplefilter('error', OptimizeWarning)
        try:
            nn_popt_contnd, nn_pcov_contnd = curve_fit(logistic, x['Time (minutes)'], x['Contact Surface Length (N.D.)'], p0 = (*p0_contnd,), bounds=bounds_contnd)
            nn_x_contnd = np.linspace(fit_time_min, fit_time_max, step_size)
            nn_y_contnd = logistic(nn_x_contnd, *nn_popt_contnd) 
                    
            nn_indiv_logistic_contnd_fits_mesg.append(("Success!", x['Source_File'].iloc[0]))
            nn_indiv_logistic_contnd_popt_pcov.append((nn_popt_contnd, nn_pcov_contnd, x['Source_File'].iloc[0]))
            nn_indiv_logistic_x_contnd.append(nn_x_contnd)
            nn_indiv_logistic_y_contnd.append(nn_y_contnd)
            nn_indiv_logistic_contnd.append(("Success", x['Source_File'].iloc[0], bounds_contnd, *p0_contnd, *nn_popt_contnd, nn_pcov_contnd, nn_x_contnd, nn_y_contnd))
        except (RuntimeError, ValueError, OptimizeWarning) as e:
            nn_popt_contnd, nn_pcov_contnd = np.asarray([np.float('nan')]  * len(p0_contnd)), np.asarray([np.float('nan')])
            nn_x_contnd = np.linspace(fit_time_min, fit_time_max, step_size)
            nn_y_contnd = [np.float('nan')] * len(nn_x_contnd)
            
            nn_indiv_logistic_contnd_fits_mesg.append((e, x['Source_File'].iloc[0]))
            rt_err_popt_placeholder = np.empty((4,))
            rt_err_popt_placeholder[:] = np.nan
            rt_err_pcov_placeholder = np.empty((4,4))
            rt_err_pcov_placeholder[:] = np.nan
            nn_indiv_logistic_contnd_popt_pcov.append((rt_err_popt_placeholder, rt_err_pcov_placeholder, x['Source_File'].iloc[0]))
            nn_indiv_logistic_x_contnd.append([np.float('nan')])
            nn_indiv_logistic_y_contnd.append([np.float('nan')])
            nn_indiv_logistic_contnd.append((e, x['Source_File'].iloc[0], bounds_contnd, *p0_contnd, *nn_popt_contnd, nn_pcov_contnd, nn_x_contnd, nn_y_contnd))
# For logistic function where nu=1:
nn_indiv_logistic_contnd_df = pd.DataFrame(data=nn_indiv_logistic_contnd, columns=['mesg', 'Source_File', 'bounds', 'p0_x2', 'p0_k2', 'p0_high_plateau', 'p0_mid_plateau', 'popt_x2', 'popt_k2', 'popt_high_plateau', 'popt_mid_plateau', 'pcov', 'logistic_x', 'logistic_y'])


###############################################################################



### Trying to fit individual logistic functions to each timelapse:
### For ecto-ecto in dissociation buffer:
eedb_list_dropna = [x.dropna(axis=0, subset=['Contact Surface Length (px)']) for x in eedb_list]

# Also drop zero values from Contact Surface Length so that the logistic fit
# better reflects the low plateau:
for x in eedb_list_dropna:
    x.drop(axis=0, index=x[x['Contact Surface Length (px)'] == 0].index, inplace=True)
eedb_df_list_long_dropna = pd.concat(eedb_list_dropna)

eedb_contact_diam_non_dim_df_list_dropna = [x.dropna(axis=0, subset=['Contact Surface Length (N.D.)']) for x in eedb_contact_diam_non_dim_df_list]
# Also, try removing zero values, as they interfere with determining the low
# plateau:
for x in eedb_contact_diam_non_dim_df_list_dropna:
    x.drop(axis=0, index=x[x['Contact Surface Length (N.D.)'] == 0].index, inplace=True)
eedb_contact_diam_non_dim_df_list_long_dropna = pd.concat(eedb_contact_diam_non_dim_df_list_dropna)
# There are some eedb cells that reach a N.D. contact diameter > 1, I want to
# see if they are responsible for the long tail of the eedb half-life and also
# if the minor peak on the right goes away if they are gone:
files_to_exclude = ['2018-01-25 xl wt-ecto-fdx wt-ecto-rdx 1X DB 20C crop3.csv',
                    '2018-01-25 xl wt-ecto-fdx wt-ecto-rdx 1X DB 20C crop7.csv']
excl_bool = eedb_contact_diam_non_dim_df_list_long_dropna['Source_File'].apply(lambda x: not(x in files_to_exclude))
eedb_contact_diam_non_dim_df_list_long_dropna_no_high_adh = eedb_contact_diam_non_dim_df_list_long_dropna[excl_bool]
excl_bool = eedb_df_list_long_dropna['Source_File'].apply(lambda x: not(x in files_to_exclude))
eedb_df_list_long_dropna_no_high_adh = eedb_df_list_long_dropna[excl_bool]


# Now try to fit logistic functions to each timelapse in ecto-ecto in
# dissociation buffer data:
eedb_indiv_logistic_cont_fits_mesg = []
eedb_indiv_logistic_cont_popt_pcov = []
eedb_indiv_logistic_x_cont = []
eedb_indiv_logistic_y_cont = []
eedb_indiv_logistic_cont = []
# For logistic function where nu=1:
#p0_cont = (10,1,50, 0)
#bounds_cont = ([-50,0,30,0], [100,100,100,30])

for x in eedb_list_dropna:
    with warnings.catch_warnings():
        warnings.simplefilter('error', OptimizeWarning)
        try:
            eedb_popt_contact, eedb_pcov_contact = curve_fit(logistic, x['Time (minutes)'], x['Contact Surface Length (px)'], p0 = (*p0_cont,), bounds=bounds_cont_eedb)
            eedb_x_contact = np.linspace(fit_time_min, fit_time_max, step_size)
            eedb_y_contact = logistic(eedb_x_contact, *eedb_popt_contact)
                   
            eedb_indiv_logistic_cont_fits_mesg.append(("Success!", x['Source_File'].iloc[0]))
            eedb_indiv_logistic_cont_popt_pcov.append((eedb_popt_contact, eedb_pcov_contact, x['Source_File'].iloc[0]))
            eedb_indiv_logistic_x_cont.append(eedb_x_contact)
            eedb_indiv_logistic_y_cont.append(eedb_y_contact)
            eedb_indiv_logistic_cont.append(("Success", x['Source_File'].iloc[0], bounds_cont, *p0_cont, *eedb_popt_contact, eedb_pcov_contact, eedb_x_contact, eedb_y_contact))
        except (RuntimeError, ValueError, OptimizeWarning) as e:
            eedb_popt_contact, eedb_pcov_contact = np.asarray([np.float('nan')]  * len(p0_cont)), np.asarray([np.float('nan')])
            eedb_x_contact = np.linspace(fit_time_min, fit_time_max, step_size)
            eedb_y_contact = [np.float('nan')] * len(eedb_x_contact)
            
            eedb_indiv_logistic_cont_fits_mesg.append((e, x['Source_File'].iloc[0]))
            rt_err_popt_placeholder = np.empty((4,))
            rt_err_popt_placeholder[:] = np.nan
            rt_err_pcov_placeholder = np.empty((4,4))
            rt_err_pcov_placeholder[:] = np.nan
            eedb_indiv_logistic_cont_popt_pcov.append((rt_err_popt_placeholder, rt_err_pcov_placeholder, x['Source_File'].iloc[0]))
            eedb_indiv_logistic_x_cont.append([np.float('nan')])
            eedb_indiv_logistic_y_cont.append([np.float('nan')])
            eedb_indiv_logistic_cont.append((e, x['Source_File'].iloc[0], bounds_cont, *p0_cont, *eedb_popt_contact, eedb_pcov_contact, eedb_x_contact, eedb_y_contact))
# For logistic function where nu=1:
eedb_indiv_logistic_cont_df = pd.DataFrame(data=eedb_indiv_logistic_cont, columns=['mesg', 'Source_File', 'bounds', 'p0_x2', 'p0_k2', 'p0_high_plateau', 'p0_mid_plateau', 'popt_x2', 'popt_k2', 'popt_high_plateau', 'popt_mid_plateau', 'pcov', 'logistic_x', 'logistic_y'])

eedb_indiv_logistic_angle_fits_mesg = []
eedb_indiv_logistic_angle_popt_pcov = []
eedb_indiv_logistic_x_angle = []
eedb_indiv_logistic_y_angle = []
eedb_indiv_logistic_angle = []
# For logistic function where nu=1:
#p0_angle = (10,1,120,0)
#bounds_angle = ([-50,0,60,0], [100,100, 180,60])

for x in eedb_list_dropna:
    with warnings.catch_warnings():
        warnings.simplefilter('error', OptimizeWarning)
        try:       
            eedb_popt_angle, eedb_pcov_angle = curve_fit(logistic, x['Time (minutes)'], x['Average Angle (deg)'], p0 = (*p0_angle,), bounds=bounds_angle_eedb)
            eedb_x_angle = np.linspace(fit_time_min, fit_time_max, step_size)
            eedb_y_angle = logistic(eedb_x_angle, *eedb_popt_angle)        
            
            eedb_indiv_logistic_angle_fits_mesg.append(("Success!", x['Source_File'].iloc[0]))
            eedb_indiv_logistic_angle_popt_pcov.append((eedb_popt_angle, eedb_pcov_angle, x['Source_File'].iloc[0]))
            eedb_indiv_logistic_x_angle.append(eedb_x_angle)
            eedb_indiv_logistic_y_angle.append(eedb_y_angle)
            eedb_indiv_logistic_angle.append(("Success", x['Source_File'].iloc[0], bounds_angle, *p0_angle, *eedb_popt_angle, eedb_pcov_angle, eedb_x_angle, eedb_y_angle))
        except (RuntimeError, ValueError, OptimizeWarning) as e:
            eedb_popt_angle, eedb_pcov_angle = np.asarray([np.float('nan')]  * len(p0_angle)), np.asarray([np.float('nan')])
            eedb_x_angle = np.linspace(fit_time_min, fit_time_max, step_size)
            eedb_y_angle = [np.float('nan')] * len(eedb_x_angle)
            
            eedb_indiv_logistic_angle_fits_mesg.append((e, x['Source_File'].iloc[0]))
            rt_err_popt_placeholder = np.empty((4,))
            rt_err_popt_placeholder[:] = np.nan
            rt_err_pcov_placeholder = np.empty((4,4))
            rt_err_pcov_placeholder[:] = np.nan
            eedb_indiv_logistic_angle_popt_pcov.append((rt_err_popt_placeholder, rt_err_pcov_placeholder, x['Source_File'].iloc[0]))
            eedb_indiv_logistic_x_angle.append([np.float('nan')])
            eedb_indiv_logistic_y_angle.append([np.float('nan')])
            eedb_indiv_logistic_angle.append((e, x['Source_File'].iloc[0], bounds_angle, *p0_angle, *eedb_popt_angle, eedb_pcov_angle, eedb_x_angle, eedb_y_angle))
# For logistic function where nu=1:
eedb_indiv_logistic_angle_df = pd.DataFrame(data=eedb_indiv_logistic_angle, columns=['mesg', 'Source_File', 'bounds', 'p0_x2', 'p0_k2', 'p0_high_plateau', 'p0_mid_plateau', 'popt_x2', 'popt_k2', 'popt_high_plateau', 'popt_mid_plateau', 'pcov', 'logistic_x', 'logistic_y'])


eedb_indiv_logistic_contnd_fits_mesg = []
eedb_indiv_logistic_contnd_popt_pcov = []
eedb_indiv_logistic_x_contnd = []
eedb_indiv_logistic_y_contnd = []
eedb_indiv_logistic_contnd = []
# For logistic function where nu=1:
#p0_contnd = [10,1,2,0]
#bounds_contnd = ([-50,0,1,0], [100,100,3,1])

for x in eedb_contact_diam_non_dim_df_list_dropna:
    with warnings.catch_warnings():
        warnings.simplefilter('error', OptimizeWarning)
        try:
            eedb_popt_contnd, eedb_pcov_contnd = curve_fit(logistic, x['Time (minutes)'], x['Contact Surface Length (N.D.)'], p0 = (*p0_contnd,), bounds=bounds_contnd_eedb)
            eedb_x_contnd = np.linspace(fit_time_min, fit_time_max, step_size)
            eedb_y_contnd = logistic(eedb_x_contnd, *eedb_popt_contnd) 
                    
            eedb_indiv_logistic_contnd_fits_mesg.append(("Success!", x['Source_File'].iloc[0]))
            eedb_indiv_logistic_contnd_popt_pcov.append((eedb_popt_contnd, eedb_pcov_contnd, x['Source_File'].iloc[0]))
            eedb_indiv_logistic_x_contnd.append(eedb_x_contnd)
            eedb_indiv_logistic_y_contnd.append(eedb_y_contnd)
            eedb_indiv_logistic_contnd.append(("Success", x['Source_File'].iloc[0], bounds_contnd, *p0_contnd, *eedb_popt_contnd, eedb_pcov_contnd, eedb_x_contnd, eedb_y_contnd))
        except (RuntimeError, ValueError, OptimizeWarning) as e:
            eedb_popt_contnd, eedb_pcov_contnd = np.asarray([np.float('nan')]  * len(p0_contnd)), np.asarray([np.float('nan')])
            eedb_x_contnd = np.linspace(fit_time_min, fit_time_max, step_size)
            eedb_y_contnd = [np.float('nan')] * len(eedb_x_contnd)
            
            eedb_indiv_logistic_contnd_fits_mesg.append((e, x['Source_File'].iloc[0]))
            rt_err_popt_placeholder = np.empty((4,))
            rt_err_popt_placeholder[:] = np.nan
            rt_err_pcov_placeholder = np.empty((4,4))
            rt_err_pcov_placeholder[:] = np.nan
            eedb_indiv_logistic_contnd_popt_pcov.append((rt_err_popt_placeholder, rt_err_pcov_placeholder, x['Source_File'].iloc[0]))
            eedb_indiv_logistic_x_contnd.append([np.float('nan')])
            eedb_indiv_logistic_y_contnd.append([np.float('nan')])
            eedb_indiv_logistic_contnd.append((e, x['Source_File'].iloc[0], bounds_contnd, *p0_contnd, *eedb_popt_contnd, eedb_pcov_contnd, eedb_x_contnd, eedb_y_contnd))
# For logistic function where nu=1:
eedb_indiv_logistic_contnd_df = pd.DataFrame(data=eedb_indiv_logistic_contnd, columns=['mesg', 'Source_File', 'bounds', 'p0_x2', 'p0_k2', 'p0_high_plateau', 'p0_mid_plateau', 'popt_x2', 'popt_k2', 'popt_high_plateau', 'popt_mid_plateau', 'pcov', 'logistic_x', 'logistic_y'])





### Trying to fit individual logistic functions to each timelapse:
### For eeCcadMO:
eeCcadMO_list_dropna = [x.dropna(axis=0, subset=['Contact Surface Length (px)']) for x in eeCcadMO_list]

# Also drop zero values from Contact Surface Length so that the logistic fit
# better reflects the low plateau:
for x in eeCcadMO_list_dropna:
    x.drop(axis=0, index=x[x['Contact Surface Length (px)'] == 0].index, inplace=True)

eeCcadMO_contact_diam_non_dim_df_list_dropna = [x.dropna(axis=0, subset=['Contact Surface Length (N.D.)']) for x in eeCcadMO_contact_diam_non_dim_df_list]
# Also, try removing zero values, as they interfere with determining the low
# plateau:
for x in eeCcadMO_contact_diam_non_dim_df_list_dropna:
    x.drop(axis=0, index=x[x['Contact Surface Length (N.D.)'] == 0].index, inplace=True)


# Now try to fit logistic functions to each timelapse in meso-meso data:
eeCcadMO_indiv_logistic_cont_fits_mesg = []
eeCcadMO_indiv_logistic_cont_popt_pcov = []
eeCcadMO_indiv_logistic_x_cont = []
eeCcadMO_indiv_logistic_y_cont = []
eeCcadMO_indiv_logistic_cont = []
# For logistic function where nu=1:
#p0_cont = (10,1,50, 0)
#bounds_cont = ([-50,0,30,0], [100,100,100,30])

for x in eeCcadMO_list_dropna:
    with warnings.catch_warnings():
        warnings.simplefilter('error', OptimizeWarning)
        try:
            eeCcadMO_popt_contact, eeCcadMO_pcov_contact = curve_fit(logistic, x['Time (minutes)'], x['Contact Surface Length (px)'], p0 = (*p0_cont,), bounds=bounds_cont)
            eeCcadMO_x_contact = np.linspace(fit_time_min, fit_time_max, step_size)
            eeCcadMO_y_contact = logistic(eeCcadMO_x_contact, *eeCcadMO_popt_contact)
                   
            eeCcadMO_indiv_logistic_cont_fits_mesg.append(("Success!", x['Source_File'].iloc[0]))
            eeCcadMO_indiv_logistic_cont_popt_pcov.append((eeCcadMO_popt_contact, eeCcadMO_pcov_contact, x['Source_File'].iloc[0]))
            eeCcadMO_indiv_logistic_x_cont.append(eeCcadMO_x_contact)
            eeCcadMO_indiv_logistic_y_cont.append(eeCcadMO_y_contact)
            eeCcadMO_indiv_logistic_cont.append(("Success", x['Source_File'].iloc[0], bounds_cont, *p0_cont, *eeCcadMO_popt_contact, eeCcadMO_pcov_contact, eeCcadMO_x_contact, eeCcadMO_y_contact))
        except (RuntimeError, ValueError, OptimizeWarning) as e:
            eeCcadMO_popt_contact, eeCcadMO_pcov_contact = np.asarray([np.float('nan')]  * len(p0_cont)), np.asarray([np.float('nan')])
            eeCcadMO_x_contact = np.linspace(fit_time_min, fit_time_max, step_size)
            eeCcadMO_y_contact = [np.float('nan')] * len(eeCcadMO_x_contact)
            
            eeCcadMO_indiv_logistic_cont_fits_mesg.append((e, x['Source_File'].iloc[0]))
            rt_err_popt_placeholder = np.empty((4,))
            rt_err_popt_placeholder[:] = np.nan
            rt_err_pcov_placeholder = np.empty((4,4))
            rt_err_pcov_placeholder[:] = np.nan
            eeCcadMO_indiv_logistic_cont_popt_pcov.append((rt_err_popt_placeholder, rt_err_pcov_placeholder, x['Source_File'].iloc[0]))
            eeCcadMO_indiv_logistic_x_cont.append([np.float('nan')])
            eeCcadMO_indiv_logistic_y_cont.append([np.float('nan')])
            eeCcadMO_indiv_logistic_cont.append((e, x['Source_File'].iloc[0], bounds_cont, *p0_cont, *eeCcadMO_popt_contact, eeCcadMO_pcov_contact, eeCcadMO_x_contact, eeCcadMO_y_contact))
# For logistic function where nu=1:
eeCcadMO_indiv_logistic_cont_df = pd.DataFrame(data=eeCcadMO_indiv_logistic_cont, columns=['mesg', 'Source_File', 'bounds', 'p0_x2', 'p0_k2', 'p0_high_plateau', 'p0_mid_plateau', 'popt_x2', 'popt_k2', 'popt_high_plateau', 'popt_mid_plateau', 'pcov', 'logistic_x', 'logistic_y'])


eeCcadMO_indiv_logistic_angle_fits_mesg = []
eeCcadMO_indiv_logistic_angle_popt_pcov = []
eeCcadMO_indiv_logistic_x_angle = []
eeCcadMO_indiv_logistic_y_angle = []
eeCcadMO_indiv_logistic_angle = []
# For logistic function where nu=1:
#p0_angle = (10,1,120,0)
#bounds_angle = ([-50,0,60,0], [100,100, 180,60])

for x in eeCcadMO_list_dropna:
    with warnings.catch_warnings():
        warnings.simplefilter('error', OptimizeWarning)
        try:       
            eeCcadMO_popt_angle, eeCcadMO_pcov_angle = curve_fit(logistic, x['Time (minutes)'], x['Average Angle (deg)'], p0 = (*p0_angle,), bounds=bounds_angle)
            eeCcadMO_x_angle = np.linspace(fit_time_min, fit_time_max, step_size)
            eeCcadMO_y_angle = logistic(eeCcadMO_x_angle, *eeCcadMO_popt_angle)        
            
            eeCcadMO_indiv_logistic_angle_fits_mesg.append(("Success!", x['Source_File'].iloc[0]))
            eeCcadMO_indiv_logistic_angle_popt_pcov.append((eeCcadMO_popt_angle, eeCcadMO_pcov_angle, x['Source_File'].iloc[0]))
            eeCcadMO_indiv_logistic_x_angle.append(eeCcadMO_x_angle)
            eeCcadMO_indiv_logistic_y_angle.append(eeCcadMO_y_angle)
            eeCcadMO_indiv_logistic_angle.append(("Success", x['Source_File'].iloc[0], bounds_angle, *p0_angle, *eeCcadMO_popt_angle, eeCcadMO_pcov_angle, eeCcadMO_x_angle, eeCcadMO_y_angle))
        except (RuntimeError, ValueError, OptimizeWarning) as e:
            eeCcadMO_popt_angle, eeCcadMO_pcov_angle = np.asarray([np.float('nan')]  * len(p0_angle)), np.asarray([np.float('nan')])
            eeCcadMO_x_angle = np.linspace(fit_time_min, fit_time_max, step_size)
            eeCcadMO_y_angle = [np.float('nan')] * len(eeCcadMO_x_angle)
            
            eeCcadMO_indiv_logistic_angle_fits_mesg.append((e, x['Source_File'].iloc[0]))
            rt_err_popt_placeholder = np.empty((4,))
            rt_err_popt_placeholder[:] = np.nan
            rt_err_pcov_placeholder = np.empty((4,4))
            rt_err_pcov_placeholder[:] = np.nan
            eeCcadMO_indiv_logistic_angle_popt_pcov.append((rt_err_popt_placeholder, rt_err_pcov_placeholder, x['Source_File'].iloc[0]))
            eeCcadMO_indiv_logistic_x_angle.append([np.float('nan')])
            eeCcadMO_indiv_logistic_y_angle.append([np.float('nan')])
            eeCcadMO_indiv_logistic_angle.append((e, x['Source_File'].iloc[0], bounds_angle, *p0_angle, *eeCcadMO_popt_angle, eeCcadMO_pcov_angle, eeCcadMO_x_angle, eeCcadMO_y_angle))
# For logistic function where nu=1:
eeCcadMO_indiv_logistic_angle_df = pd.DataFrame(data=eeCcadMO_indiv_logistic_angle, columns=['mesg', 'Source_File', 'bounds', 'p0_x2', 'p0_k2', 'p0_high_plateau', 'p0_mid_plateau', 'popt_x2', 'popt_k2', 'popt_high_plateau', 'popt_mid_plateau', 'pcov', 'logistic_x', 'logistic_y'])


eeCcadMO_indiv_logistic_contnd_fits_mesg = []
eeCcadMO_indiv_logistic_contnd_popt_pcov = []
eeCcadMO_indiv_logistic_x_contnd = []
eeCcadMO_indiv_logistic_y_contnd = []
eeCcadMO_indiv_logistic_contnd = []
# For logistic function where nu=1:
#p0_contnd = [10,1,2,0]
#bounds_contnd = ([-50,0,1,0], [100,100,3,1])

for x in eeCcadMO_contact_diam_non_dim_df_list_dropna:
    with warnings.catch_warnings():
        warnings.simplefilter('error', OptimizeWarning)
        try:
            eeCcadMO_popt_contnd, eeCcadMO_pcov_contnd = curve_fit(logistic, x['Time (minutes)'], x['Contact Surface Length (N.D.)'], p0 = (*p0_contnd,), bounds=bounds_contnd)
            eeCcadMO_x_contnd = np.linspace(fit_time_min, fit_time_max, step_size)
            eeCcadMO_y_contnd = logistic(eeCcadMO_x_contnd, *eeCcadMO_popt_contnd) 
                    
            eeCcadMO_indiv_logistic_contnd_fits_mesg.append(("Success!", x['Source_File'].iloc[0]))
            eeCcadMO_indiv_logistic_contnd_popt_pcov.append((eeCcadMO_popt_contnd, eeCcadMO_pcov_contnd, x['Source_File'].iloc[0]))
            eeCcadMO_indiv_logistic_x_contnd.append(eeCcadMO_x_contnd)
            eeCcadMO_indiv_logistic_y_contnd.append(eeCcadMO_y_contnd)
            eeCcadMO_indiv_logistic_contnd.append(("Success", x['Source_File'].iloc[0], bounds_contnd, *p0_contnd, *eeCcadMO_popt_contnd, eeCcadMO_pcov_contnd, eeCcadMO_x_contnd, eeCcadMO_y_contnd))
        except (RuntimeError, ValueError, OptimizeWarning) as e:
            eeCcadMO_popt_contnd, eeCcadMO_pcov_contnd = np.asarray([np.float('nan')]  * len(p0_contnd)), np.asarray([np.float('nan')])
            eeCcadMO_x_contnd = np.linspace(fit_time_min, fit_time_max, step_size)
            eeCcadMO_y_contnd = [np.float('nan')] * len(eeCcadMO_x_contnd)
            
            eeCcadMO_indiv_logistic_contnd_fits_mesg.append((e, x['Source_File'].iloc[0]))
            rt_err_popt_placeholder = np.empty((4,))
            rt_err_popt_placeholder[:] = np.nan
            rt_err_pcov_placeholder = np.empty((4,4))
            rt_err_pcov_placeholder[:] = np.nan
            eeCcadMO_indiv_logistic_contnd_popt_pcov.append((rt_err_popt_placeholder, rt_err_pcov_placeholder, x['Source_File'].iloc[0]))
            eeCcadMO_indiv_logistic_x_contnd.append([np.float('nan')])
            eeCcadMO_indiv_logistic_y_contnd.append([np.float('nan')])
            eeCcadMO_indiv_logistic_contnd.append((e, x['Source_File'].iloc[0], bounds_contnd, *p0_contnd, *eeCcadMO_popt_contnd, eeCcadMO_pcov_contnd, eeCcadMO_x_contnd, eeCcadMO_y_contnd))
# For logistic function where nu=1:
eeCcadMO_indiv_logistic_contnd_df = pd.DataFrame(data=eeCcadMO_indiv_logistic_contnd, columns=['mesg', 'Source_File', 'bounds', 'p0_x2', 'p0_k2', 'p0_high_plateau', 'p0_mid_plateau', 'popt_x2', 'popt_k2', 'popt_high_plateau', 'popt_mid_plateau', 'pcov', 'logistic_x', 'logistic_y'])






### Trying to fit individual logistic functions to each timelapse:
### For eeRcadMO:
eeRcadMO_list_dropna = [x.dropna(axis=0, subset=['Contact Surface Length (px)']) for x in eeRcadMO_list]

# Also drop zero values from Contact Surface Length so that the logistic fit
# better reflects the low plateau:
for x in eeRcadMO_list_dropna:
    x.drop(axis=0, index=x[x['Contact Surface Length (px)'] == 0].index, inplace=True)

eeRcadMO_contact_diam_non_dim_df_list_dropna = [x.dropna(axis=0, subset=['Contact Surface Length (N.D.)']) for x in eeRcadMO_contact_diam_non_dim_df_list]
# Also, try removing zero values, as they interfere with determining the low
# plateau:
for x in eeRcadMO_contact_diam_non_dim_df_list_dropna:
    x.drop(axis=0, index=x[x['Contact Surface Length (N.D.)'] == 0].index, inplace=True)


# Now try to fit logistic functions to each timelapse in meso-meso data:
eeRcadMO_indiv_logistic_cont_fits_mesg = []
eeRcadMO_indiv_logistic_cont_popt_pcov = []
eeRcadMO_indiv_logistic_x_cont = []
eeRcadMO_indiv_logistic_y_cont = []
eeRcadMO_indiv_logistic_cont = []
# For logistic function where nu=1:
#p0_cont = (10,1,50, 0)
#bounds_cont = ([-50,0,30,0], [100,100,100,30])

for x in eeRcadMO_list_dropna:
    with warnings.catch_warnings():
        warnings.simplefilter('error', OptimizeWarning)
        try:
            eeRcadMO_popt_contact, eeRcadMO_pcov_contact = curve_fit(logistic, x['Time (minutes)'], x['Contact Surface Length (px)'], p0 = (*p0_cont,), bounds=bounds_cont)
            eeRcadMO_x_contact = np.linspace(fit_time_min, fit_time_max, step_size)
            eeRcadMO_y_contact = logistic(eeRcadMO_x_contact, *eeRcadMO_popt_contact)
                   
            eeRcadMO_indiv_logistic_cont_fits_mesg.append(("Success!", x['Source_File'].iloc[0]))
            eeRcadMO_indiv_logistic_cont_popt_pcov.append((eeRcadMO_popt_contact, eeRcadMO_pcov_contact, x['Source_File'].iloc[0]))
            eeRcadMO_indiv_logistic_x_cont.append(eeRcadMO_x_contact)
            eeRcadMO_indiv_logistic_y_cont.append(eeRcadMO_y_contact)
            eeRcadMO_indiv_logistic_cont.append(("Success", x['Source_File'].iloc[0], bounds_cont, *p0_cont, *eeRcadMO_popt_contact, eeRcadMO_pcov_contact, eeRcadMO_x_contact, eeRcadMO_y_contact))
        except (RuntimeError, ValueError, OptimizeWarning) as e:
            eeRcadMO_popt_contact, eeRcadMO_pcov_contact = np.asarray([np.float('nan')]  * len(p0_cont)), np.asarray([np.float('nan')])
            eeRcadMO_x_contact = np.linspace(fit_time_min, fit_time_max, step_size)
            eeRcadMO_y_contact = [np.float('nan')] * len(eeRcadMO_x_contact)
            
            eeRcadMO_indiv_logistic_cont_fits_mesg.append((e, x['Source_File'].iloc[0]))
            rt_err_popt_placeholder = np.empty((4,))
            rt_err_popt_placeholder[:] = np.nan
            rt_err_pcov_placeholder = np.empty((4,4))
            rt_err_pcov_placeholder[:] = np.nan
            eeRcadMO_indiv_logistic_cont_popt_pcov.append((rt_err_popt_placeholder, rt_err_pcov_placeholder, x['Source_File'].iloc[0]))
            eeRcadMO_indiv_logistic_x_cont.append([np.float('nan')])
            eeRcadMO_indiv_logistic_y_cont.append([np.float('nan')])
            eeRcadMO_indiv_logistic_cont.append((e, x['Source_File'].iloc[0], bounds_cont, *p0_cont, *eeRcadMO_popt_contact, eeRcadMO_pcov_contact, eeRcadMO_x_contact, eeRcadMO_y_contact))
# For logistic function where nu=1:
eeRcadMO_indiv_logistic_cont_df = pd.DataFrame(data=eeRcadMO_indiv_logistic_cont, columns=['mesg', 'Source_File', 'bounds', 'p0_x2', 'p0_k2', 'p0_high_plateau', 'p0_mid_plateau', 'popt_x2', 'popt_k2', 'popt_high_plateau', 'popt_mid_plateau', 'pcov', 'logistic_x', 'logistic_y'])


eeRcadMO_indiv_logistic_angle_fits_mesg = []
eeRcadMO_indiv_logistic_angle_popt_pcov = []
eeRcadMO_indiv_logistic_x_angle = []
eeRcadMO_indiv_logistic_y_angle = []
eeRcadMO_indiv_logistic_angle = []
# For logistic function where nu=1:
#p0_angle = (10,1,120,0)
#bounds_angle = ([-50,0,60,0], [100,100, 180,60])

for x in eeRcadMO_list_dropna:
    with warnings.catch_warnings():
        warnings.simplefilter('error', OptimizeWarning)
        try:
            eeRcadMO_popt_angle, eeRcadMO_pcov_angle = curve_fit(logistic, x['Time (minutes)'], x['Average Angle (deg)'], p0 = (*p0_angle,), bounds=bounds_angle)
            eeRcadMO_x_angle = np.linspace(fit_time_min, fit_time_max, step_size)
            eeRcadMO_y_angle = logistic(eeRcadMO_x_angle, *eeRcadMO_popt_angle)        
            
            eeRcadMO_indiv_logistic_angle_fits_mesg.append(("Success!", x['Source_File'].iloc[0]))
            eeRcadMO_indiv_logistic_angle_popt_pcov.append((eeRcadMO_popt_angle, eeRcadMO_pcov_angle, x['Source_File'].iloc[0]))
            eeRcadMO_indiv_logistic_x_angle.append(eeRcadMO_x_angle)
            eeRcadMO_indiv_logistic_y_angle.append(eeRcadMO_y_angle)
            eeRcadMO_indiv_logistic_angle.append(("Success", x['Source_File'].iloc[0], bounds_angle, *p0_angle, *eeRcadMO_popt_angle, eeRcadMO_pcov_angle, eeRcadMO_x_angle, eeRcadMO_y_angle))
        except (RuntimeError, ValueError, OptimizeWarning) as e:
            eeRcadMO_popt_angle, eeRcadMO_pcov_angle = np.asarray([np.float('nan')]  * len(p0_angle)), np.asarray([np.float('nan')])
            eeRcadMO_x_angle = np.linspace(fit_time_min, fit_time_max, step_size)
            eeRcadMO_y_angle = [np.float('nan')] * len(eeRcadMO_x_angle)
            
            eeRcadMO_indiv_logistic_angle_fits_mesg.append((e, x['Source_File'].iloc[0]))
            rt_err_popt_placeholder = np.empty((4,))
            rt_err_popt_placeholder[:] = np.nan
            rt_err_pcov_placeholder = np.empty((4,4))
            rt_err_pcov_placeholder[:] = np.nan
            eeRcadMO_indiv_logistic_angle_popt_pcov.append((rt_err_popt_placeholder, rt_err_pcov_placeholder, x['Source_File'].iloc[0]))
            eeRcadMO_indiv_logistic_x_angle.append([np.float('nan')])
            eeRcadMO_indiv_logistic_y_angle.append([np.float('nan')])
            eeRcadMO_indiv_logistic_angle.append((e, x['Source_File'].iloc[0], bounds_angle, *p0_angle, *eeRcadMO_popt_angle, eeRcadMO_pcov_angle, eeRcadMO_x_angle, eeRcadMO_y_angle))
# For logistic function where nu=1:
eeRcadMO_indiv_logistic_angle_df = pd.DataFrame(data=eeRcadMO_indiv_logistic_angle, columns=['mesg', 'Source_File', 'bounds', 'p0_x2', 'p0_k2', 'p0_high_plateau', 'p0_mid_plateau', 'popt_x2', 'popt_k2', 'popt_high_plateau', 'popt_mid_plateau', 'pcov', 'logistic_x', 'logistic_y'])


eeRcadMO_indiv_logistic_contnd_fits_mesg = []
eeRcadMO_indiv_logistic_contnd_popt_pcov = []
eeRcadMO_indiv_logistic_x_contnd = []
eeRcadMO_indiv_logistic_y_contnd = []
eeRcadMO_indiv_logistic_contnd = []
# For logistic function where nu=1:
#p0_contnd = [10,1,2,0]
#bounds_contnd = ([-50,0,1,0], [100,100,3,1])

for x in eeRcadMO_contact_diam_non_dim_df_list_dropna:
    with warnings.catch_warnings():
        warnings.simplefilter('error', OptimizeWarning)
        try:
            eeRcadMO_popt_contnd, eeRcadMO_pcov_contnd = curve_fit(logistic, x['Time (minutes)'], x['Contact Surface Length (N.D.)'], p0 = (*p0_contnd,), bounds=bounds_contnd)
            eeRcadMO_x_contnd = np.linspace(fit_time_min, fit_time_max, step_size)
            eeRcadMO_y_contnd = logistic(eeRcadMO_x_contnd, *eeRcadMO_popt_contnd) 
                    
            eeRcadMO_indiv_logistic_contnd_fits_mesg.append(("Success!", x['Source_File'].iloc[0]))
            eeRcadMO_indiv_logistic_contnd_popt_pcov.append((eeRcadMO_popt_contnd, eeRcadMO_pcov_contnd, x['Source_File'].iloc[0]))
            eeRcadMO_indiv_logistic_x_contnd.append(eeRcadMO_x_contnd)
            eeRcadMO_indiv_logistic_y_contnd.append(eeRcadMO_y_contnd)
            eeRcadMO_indiv_logistic_contnd.append(("Success", x['Source_File'].iloc[0], bounds_contnd, *p0_contnd, *eeRcadMO_popt_contnd, eeRcadMO_pcov_contnd, eeRcadMO_x_contnd, eeRcadMO_y_contnd))
        except (RuntimeError, ValueError, OptimizeWarning) as e:
            eeRcadMO_popt_contnd, eeRcadMO_pcov_contnd = np.asarray([np.float('nan')]  * len(p0_contnd)), np.asarray([np.float('nan')])
            eeRcadMO_x_contnd = np.linspace(fit_time_min, fit_time_max, step_size)
            eeRcadMO_y_contnd = [np.float('nan')] * len(eeRcadMO_x_contnd)
            
            eeRcadMO_indiv_logistic_contnd_fits_mesg.append((e, x['Source_File'].iloc[0]))
            rt_err_popt_placeholder = np.empty((4,))
            rt_err_popt_placeholder[:] = np.nan
            rt_err_pcov_placeholder = np.empty((4,4))
            rt_err_pcov_placeholder[:] = np.nan
            eeRcadMO_indiv_logistic_contnd_popt_pcov.append((rt_err_popt_placeholder, rt_err_pcov_placeholder, x['Source_File'].iloc[0]))
            eeRcadMO_indiv_logistic_x_contnd.append([np.float('nan')])
            eeRcadMO_indiv_logistic_y_contnd.append([np.float('nan')])
            eeRcadMO_indiv_logistic_contnd.append((e, x['Source_File'].iloc[0], bounds_contnd, *p0_contnd, *eeRcadMO_popt_contnd, eeRcadMO_pcov_contnd, eeRcadMO_x_contnd, eeRcadMO_y_contnd))
# For logistic function where nu=1:
eeRcadMO_indiv_logistic_contnd_df = pd.DataFrame(data=eeRcadMO_indiv_logistic_contnd, columns=['mesg', 'Source_File', 'bounds', 'p0_x2', 'p0_k2', 'p0_high_plateau', 'p0_mid_plateau', 'popt_x2', 'popt_k2', 'popt_high_plateau', 'popt_mid_plateau', 'pcov', 'logistic_x', 'logistic_y'])





### Trying to fit individual logistic functions to each timelapse:
### For eeRCcadMO:
eeRCcadMO_list_dropna = [x.dropna(axis=0, subset=['Contact Surface Length (px)']) for x in eeRCcadMO_list]

# Also drop zero values from Contact Surface Length so that the logistic fit
# better reflects the low plateau:
for x in eeRCcadMO_list_dropna:
    x.drop(axis=0, index=x[x['Contact Surface Length (px)'] == 0].index, inplace=True)

eeRCcadMO_contact_diam_non_dim_df_list_dropna = [x.dropna(axis=0, subset=['Contact Surface Length (N.D.)']) for x in eeRCcadMO_contact_diam_non_dim_df_list]
# Also, try removing zero values, as they interfere with determining the low
# plateau:
for x in eeRCcadMO_contact_diam_non_dim_df_list_dropna:
    x.drop(axis=0, index=x[x['Contact Surface Length (N.D.)'] == 0].index, inplace=True)


# Now try to fit logistic functions to each timelapse in meso-meso data:
eeRCcadMO_indiv_logistic_cont_fits_mesg = []
eeRCcadMO_indiv_logistic_cont_popt_pcov = []
eeRCcadMO_indiv_logistic_x_cont = []
eeRCcadMO_indiv_logistic_y_cont = []
eeRCcadMO_indiv_logistic_cont = []
# For logistic function where nu=1:
#p0_cont = (10,1,50, 0)
#bounds_cont = ([-50,0,30,0], [100,100,100,30])

for x in eeRCcadMO_list_dropna:
    with warnings.catch_warnings():
        warnings.simplefilter('error', OptimizeWarning)
        try:
            eeRCcadMO_popt_contact, eeRCcadMO_pcov_contact = curve_fit(logistic, x['Time (minutes)'], x['Contact Surface Length (px)'], p0 = (*p0_cont,), bounds=bounds_cont)
            eeRCcadMO_x_contact = np.linspace(fit_time_min, fit_time_max, step_size)
            eeRCcadMO_y_contact = logistic(eeRCcadMO_x_contact, *eeRCcadMO_popt_contact)
                   
            eeRCcadMO_indiv_logistic_cont_fits_mesg.append(("Success!", x['Source_File'].iloc[0]))
            eeRCcadMO_indiv_logistic_cont_popt_pcov.append((eeRCcadMO_popt_contact, eeRCcadMO_pcov_contact, x['Source_File'].iloc[0]))
            eeRCcadMO_indiv_logistic_x_cont.append(eeRCcadMO_x_contact)
            eeRCcadMO_indiv_logistic_y_cont.append(eeRCcadMO_y_contact)
            eeRCcadMO_indiv_logistic_cont.append(("Success", x['Source_File'].iloc[0], bounds_cont, *p0_cont, *eeRCcadMO_popt_contact, eeRCcadMO_pcov_contact, eeRCcadMO_x_contact, eeRCcadMO_y_contact))
        except (RuntimeError, ValueError, OptimizeWarning) as e:
            eeRCcadMO_popt_contact, eeRCcadMO_pcov_contact = np.asarray([np.float('nan')]  * len(p0_cont)), np.asarray([np.float('nan')])
            eeRCcadMO_x_contact = np.linspace(fit_time_min, fit_time_max, step_size)
            eeRCcadMO_y_contact = [np.float('nan')] * len(eeRCcadMO_x_contact)
            
            eeRCcadMO_indiv_logistic_cont_fits_mesg.append((e, x['Source_File'].iloc[0]))
            rt_err_popt_placeholder = np.empty((4,))
            rt_err_popt_placeholder[:] = np.nan
            rt_err_pcov_placeholder = np.empty((4,4))
            rt_err_pcov_placeholder[:] = np.nan
            eeRCcadMO_indiv_logistic_cont_popt_pcov.append((rt_err_popt_placeholder, rt_err_pcov_placeholder, x['Source_File'].iloc[0]))
            eeRCcadMO_indiv_logistic_x_cont.append([np.float('nan')])
            eeRCcadMO_indiv_logistic_y_cont.append([np.float('nan')])
            eeRCcadMO_indiv_logistic_cont.append((e, x['Source_File'].iloc[0], bounds_cont, *p0_cont, *eeRCcadMO_popt_contact, eeRCcadMO_pcov_contact, eeRCcadMO_x_contact, eeRCcadMO_y_contact))
# For logistic function where nu=1:
eeRCcadMO_indiv_logistic_cont_df = pd.DataFrame(data=eeRCcadMO_indiv_logistic_cont, columns=['mesg', 'Source_File', 'bounds', 'p0_x2', 'p0_k2', 'p0_high_plateau', 'p0_mid_plateau', 'popt_x2', 'popt_k2', 'popt_high_plateau', 'popt_mid_plateau', 'pcov', 'logistic_x', 'logistic_y'])


eeRCcadMO_indiv_logistic_angle_fits_mesg = []
eeRCcadMO_indiv_logistic_angle_popt_pcov = []
eeRCcadMO_indiv_logistic_x_angle = []
eeRCcadMO_indiv_logistic_y_angle = []
eeRCcadMO_indiv_logistic_angle = []
# For logistic function where nu=1:
#p0_angle = (10,1,120,0)
#bounds_angle = ([-50,0,60,0], [100,100, 180,60])

for x in eeRCcadMO_list_dropna:
    with warnings.catch_warnings():
        warnings.simplefilter('error', OptimizeWarning)
        try:
            eeRCcadMO_popt_angle, eeRCcadMO_pcov_angle = curve_fit(logistic, x['Time (minutes)'], x['Average Angle (deg)'], p0 = (*p0_angle,), bounds=bounds_angle)
            eeRCcadMO_x_angle = np.linspace(fit_time_min, fit_time_max, step_size)
            eeRCcadMO_y_angle = logistic(eeRCcadMO_x_angle, *eeRCcadMO_popt_angle)        
            
            eeRCcadMO_indiv_logistic_angle_fits_mesg.append(("Success!", x['Source_File'].iloc[0]))
            eeRCcadMO_indiv_logistic_angle_popt_pcov.append((eeRCcadMO_popt_angle, eeRCcadMO_pcov_angle, x['Source_File'].iloc[0]))
            eeRCcadMO_indiv_logistic_x_angle.append(eeRCcadMO_x_angle)
            eeRCcadMO_indiv_logistic_y_angle.append(eeRCcadMO_y_angle)
            eeRCcadMO_indiv_logistic_angle.append(("Success", x['Source_File'].iloc[0], bounds_angle, *p0_angle, *eeRCcadMO_popt_angle, eeRCcadMO_pcov_angle, eeRCcadMO_x_angle, eeRCcadMO_y_angle))
        except (RuntimeError, ValueError, OptimizeWarning) as e:
            eeRCcadMO_popt_angle, eeRCcadMO_pcov_angle = np.asarray([np.float('nan')]  * len(p0_angle)), np.asarray([np.float('nan')])
            eeRCcadMO_x_angle = np.linspace(fit_time_min, fit_time_max, step_size)
            eeRCcadMO_y_angle = [np.float('nan')] * len(eeRCcadMO_x_angle)
            
            eeRCcadMO_indiv_logistic_angle_fits_mesg.append((e, x['Source_File'].iloc[0]))
            rt_err_popt_placeholder = np.empty((4,))
            rt_err_popt_placeholder[:] = np.nan
            rt_err_pcov_placeholder = np.empty((4,4))
            rt_err_pcov_placeholder[:] = np.nan
            eeRCcadMO_indiv_logistic_angle_popt_pcov.append((rt_err_popt_placeholder, rt_err_pcov_placeholder, x['Source_File'].iloc[0]))
            eeRCcadMO_indiv_logistic_x_angle.append([np.float('nan')])
            eeRCcadMO_indiv_logistic_y_angle.append([np.float('nan')])
            eeRCcadMO_indiv_logistic_angle.append((e, x['Source_File'].iloc[0], bounds_angle, *p0_angle, *eeRCcadMO_popt_angle, eeRCcadMO_pcov_angle, eeRCcadMO_x_angle, eeRCcadMO_y_angle))
# For logistic function where nu=1:
eeRCcadMO_indiv_logistic_angle_df = pd.DataFrame(data=eeRCcadMO_indiv_logistic_angle, columns=['mesg', 'Source_File', 'bounds', 'p0_x2', 'p0_k2', 'p0_high_plateau', 'p0_mid_plateau', 'popt_x2', 'popt_k2', 'popt_high_plateau', 'popt_mid_plateau', 'pcov', 'logistic_x', 'logistic_y'])


eeRCcadMO_indiv_logistic_contnd_fits_mesg = []
eeRCcadMO_indiv_logistic_contnd_popt_pcov = []
eeRCcadMO_indiv_logistic_x_contnd = []
eeRCcadMO_indiv_logistic_y_contnd = []
eeRCcadMO_indiv_logistic_contnd = []
# For logistic function where nu=1:
#p0_contnd = [10,1,2,0]
#bounds_contnd = ([-50,0,1,0], [100,100,3,1])

for x in eeRCcadMO_contact_diam_non_dim_df_list_dropna:
    with warnings.catch_warnings():
        warnings.simplefilter('error', OptimizeWarning)
        try:
            eeRCcadMO_popt_contnd, eeRCcadMO_pcov_contnd = curve_fit(logistic, x['Time (minutes)'], x['Contact Surface Length (N.D.)'], p0 = (*p0_contnd,), bounds=bounds_contnd)
            eeRCcadMO_x_contnd = np.linspace(fit_time_min, fit_time_max, step_size)
            eeRCcadMO_y_contnd = logistic(eeRCcadMO_x_contnd, *eeRCcadMO_popt_contnd) 
                    
            eeRCcadMO_indiv_logistic_contnd_fits_mesg.append(("Success!", x['Source_File'].iloc[0]))
            eeRCcadMO_indiv_logistic_contnd_popt_pcov.append((eeRCcadMO_popt_contnd, eeRCcadMO_pcov_contnd, x['Source_File'].iloc[0]))
            eeRCcadMO_indiv_logistic_x_contnd.append(eeRCcadMO_x_contnd)
            eeRCcadMO_indiv_logistic_y_contnd.append(eeRCcadMO_y_contnd)
            eeRCcadMO_indiv_logistic_contnd.append(("Success", x['Source_File'].iloc[0], bounds_contnd, *p0_contnd, *eeRCcadMO_popt_contnd, eeRCcadMO_pcov_contnd, eeRCcadMO_x_contnd, eeRCcadMO_y_contnd))
        except (RuntimeError, ValueError, OptimizeWarning) as e:
            eeRCcadMO_popt_contnd, eeRCcadMO_pcov_contnd = np.asarray([np.float('nan')]  * len(p0_contnd)), np.asarray([np.float('nan')])
            eeRCcadMO_x_contnd = np.linspace(fit_time_min, fit_time_max, step_size)
            eeRCcadMO_y_contnd = [np.float('nan')] * len(eeRCcadMO_x_contnd)
            
            eeRCcadMO_indiv_logistic_contnd_fits_mesg.append((e, x['Source_File'].iloc[0]))
            rt_err_popt_placeholder = np.empty((4,))
            rt_err_popt_placeholder[:] = np.nan
            rt_err_pcov_placeholder = np.empty((4,4))
            rt_err_pcov_placeholder[:] = np.nan
            eeRCcadMO_indiv_logistic_contnd_popt_pcov.append((rt_err_popt_placeholder, rt_err_pcov_placeholder, x['Source_File'].iloc[0]))
            eeRCcadMO_indiv_logistic_x_contnd.append([np.float('nan')])
            eeRCcadMO_indiv_logistic_y_contnd.append([np.float('nan')])
            eeRCcadMO_indiv_logistic_contnd.append((e, x['Source_File'].iloc[0], bounds_contnd, *p0_contnd, *eeRCcadMO_popt_contnd, eeRCcadMO_pcov_contnd, eeRCcadMO_x_contnd, eeRCcadMO_y_contnd))
# For logistic function where nu=1:
eeRCcadMO_indiv_logistic_contnd_df = pd.DataFrame(data=eeRCcadMO_indiv_logistic_contnd, columns=['mesg', 'Source_File', 'bounds', 'p0_x2', 'p0_k2', 'p0_high_plateau', 'p0_mid_plateau', 'popt_x2', 'popt_k2', 'popt_high_plateau', 'popt_mid_plateau', 'pcov', 'logistic_x', 'logistic_y'])




###############################################################################
### The following code is used to produce individual graphs of the contact
### diameters:
### Save all of the individual contact diameter figures:

#ee_figs = [sns.lmplot('Cell Pair Number', 'Contact Surface Length (px)', data = x, fit_reg=False, markers='.') for x in ee_list]
#for i, fig in enumerate(ee_figs):
#    fig.savefig(ee_path+'/ee_fig_%d.pdf'%i, dpi=300)
#    print('figure', i)
#nrows, ncols = 4,3
#for j in range( (len(ee_list) // (nrows * ncols)) + 1 ):
#    fig, axarr = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8,10.5))
#    for i,x in enumerate(ee_list[j * (nrows*ncols) : (j+1) * (nrows*ncols)]):
#        row = i // ncols
#        col = i % ncols
#        axarr[row,col].scatter(x['Time (minutes)'], x['Contact Surface Length (px)'], marker='.', c='k')
#        axarr[row,col].plot(ee_indiv_logistic_x_cont[j * (nrows*ncols) + i], ee_indiv_logistic_y_cont[j * (nrows*ncols) + i])
##        axarr[row,col].plot(ee_indiv_dbl_logistic_x_cont[j * (nrows*ncols) + i], ee_indiv_dbl_logistic_y_cont[j * (nrows*ncols) + i], ls='--')
#        axarr[row,col].axhline()
#        axarr[row,col].set_title(x['Source_File'].iloc[0][:11]+x['Source_File'].iloc[0][-14:-4])
#        print(j * (nrows*ncols) + i)
#    fig.tight_layout()
#    plt.draw()
#    plt.show()
#    fig.savefig(ee_path+'/ee_figs_cont_%d.pdf'%j, dpi=300)
#
#for j in range( (len(ee_list) // (nrows * ncols)) + 1 ):
#    fig, axarr = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8,10.5))
#    for i,x in enumerate(ee_list[j * (nrows*ncols) : (j+1) * (nrows*ncols)]):
#        row = i // ncols
#        col = i % ncols
#        axarr[row,col].scatter(x['Time (minutes)'], x['Average Angle (deg)'], marker='.', c='k')
#        axarr[row,col].plot(ee_indiv_logistic_x_angle[j * (nrows*ncols) + i], ee_indiv_logistic_y_angle[j * (nrows*ncols) + i])
##        axarr[row,col].plot(ee_indiv_dbl_logistic_x_angle[j * (nrows*ncols) + i], ee_indiv_dbl_logistic_y_angle[j * (nrows*ncols) + i], ls='--')
#        axarr[row,col].axhline()
#        axarr[row,col].set_title(x['Source_File'].iloc[0][:11]+x['Source_File'].iloc[0][-14:-4])
#        print(j * (nrows*ncols) + i)
#    fig.tight_layout()
#    plt.draw()
#    plt.show()
#    fig.savefig(ee_path+'/ee_figs_angle_%d.pdf'%j, dpi=300)
#
#
#for j in range( (len(ee_contact_diam_non_dim_df_list) // (nrows * ncols)) + 1 ):
#    fig, axarr = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8,10.5))
#    for i,x in enumerate(ee_contact_diam_non_dim_df_list[j * (nrows*ncols) : (j+1) * (nrows*ncols)]):
#        row = i // ncols
#        col = i % ncols
#        axarr[row,col].scatter(x['Time (minutes)'], x['Contact Surface Length (N.D.)'], marker='.', c='k')
#        axarr[row,col].plot(ee_indiv_logistic_x_contnd[j * (nrows*ncols) + i], ee_indiv_logistic_y_contnd[j * (nrows*ncols) + i])
##        axarr[row,col].plot(ee_indiv_dbl_logistic_x_contnd[j * (nrows*ncols) + i], ee_indiv_dbl_logistic_y_contnd[j * (nrows*ncols) + i], ls='--')
#        axarr[row,col].axhline()
#        axarr[row,col].set_title(x['Source_File'].iloc[0][:11]+x['Source_File'].iloc[0][-14:-4])
#        print(j * (nrows*ncols) + i)
#    fig.tight_layout()
#    plt.draw()
#    plt.show()
#    fig.savefig(ee_path+'/ee_figs_contnd_%d.pdf'%j, dpi=300)
#
### I'm not using these figures anymore but the code in general could be useful
### for a later project. I should move it to a general function file.

###############################################################################

###############################################################################

###############################################################################
# Also drop zero values from Contact Surface Length so that the logistic fit
# better reflects the low plateau:
eedb_fix_list_dropna = eedb_fix_list.copy()
for x in eedb_fix_list_dropna:
    x.drop(axis=0, index=x[x['Contact Surface Length (px)'] == 0].index, inplace=True)

eedb_fix_list_long_dropna = eedb_fix_list_long.copy().reset_index()
eedb_fix_list_long_dropna.drop(columns='index', inplace=True)
eedb_fix_list_long_dropna.drop(axis=0, index=eedb_fix_list_long_dropna[eedb_fix_list_long_dropna['Contact Surface Length (px)'] == 0].index, inplace=True)
    
eedb_fix_contact_diam_non_dim_df_list_dropna = [x.dropna(axis=0, subset=['Contact Surface Length (N.D.)']) for x in eedb_fix_contact_diam_non_dim_df_list]
#eedb_fix_contact_diam_non_dim_df_list_dropna = eedb_fix_contact_diam_non_dim_df_list.copy()
for x in eedb_fix_contact_diam_non_dim_df_list_dropna:
    x.drop(axis=0, index=x[x['Contact Surface Length (N.D.)'] == 0].index, inplace=True)
eedb_fix_contact_diam_non_dim_df_list_long_dropna = pd.concat(eedb_fix_contact_diam_non_dim_df_list_dropna)





### Plot all individual fits and overlay them with respective data to check
### quality of fits:
plt.close('all')
indiv_fits = list()
x_fit = np.arange(-240, 241, step=0.1)#The stepsize was originally left blank, ie. it was 1
for i,x in enumerate(ee_list_dropna):
    print(i,x['Source_File'].iloc[0])
    print('popt_x2 , popt_k2 , popt_high_plateau , popt_mid_plateau')
    print(ee_indiv_logistic_cont_df.iloc[i]['popt_x2'],',', ee_indiv_logistic_cont_df.iloc[i]['popt_k2'],',', ee_indiv_logistic_cont_df.iloc[i]['popt_high_plateau'],',', ee_indiv_logistic_cont_df.iloc[i]['popt_mid_plateau'])
    fig, ax = plt.subplots()
    ax.scatter(x['Time (minutes)'], x['Contact Surface Length (px)'], marker='.')
    #plt.plot(x_fit, logistic(x_fit, ee_indiv_logistic_cont_df.iloc[i]['popt_x2'], ee_indiv_logistic_cont_df.iloc[i]['popt_k2'], ee_indiv_logistic_cont_df.iloc[i]['popt_high_plateau'], ee_indiv_logistic_cont_df.iloc[i]['popt_mid_plateau'], ee_indiv_logistic_cont_df.iloc[i]['popt_nu']), c='r')
    ax.plot(x_fit, logistic(x_fit, ee_indiv_logistic_cont_df.iloc[i]['popt_x2'], ee_indiv_logistic_cont_df.iloc[i]['popt_k2'], ee_indiv_logistic_cont_df.iloc[i]['popt_high_plateau'], ee_indiv_logistic_cont_df.iloc[i]['popt_mid_plateau']), c='r')
    ax.axhline(bounds_cont[0][3], c='k')
    ax.axhline(bounds_cont[0][2], c='k')
    ax.axhline(bounds_cont[1][3], c='k')
    ax.axhline(bounds_cont[1][2], c='k')
    ax.set_xlim((-50,120))
    ax.set_ylim((0, ee_list_long['Contact Surface Length (px)'].max()))
    ax.set_title('No.%d_'%i + x['Source_File'].iloc[0], fontdict={'size':10})
    plt.draw()
    #plt.show()
    #indiv_fits.append( logistic(x_fit, ee_indiv_logistic_cont_df.iloc[i]['popt_x2'], ee_indiv_logistic_cont_df.iloc[i]['popt_k2'], ee_indiv_logistic_cont_df.iloc[i]['popt_high_plateau'], ee_indiv_logistic_cont_df.iloc[i]['popt_mid_plateau'], ee_indiv_logistic_cont_df.iloc[i]['popt_nu']) )
    indiv_fits.append( logistic(x_fit, ee_indiv_logistic_cont_df.iloc[i]['popt_x2'], ee_indiv_logistic_cont_df.iloc[i]['popt_k2'], ee_indiv_logistic_cont_df.iloc[i]['popt_high_plateau'], ee_indiv_logistic_cont_df.iloc[i]['popt_mid_plateau']) )
multipdf(Path.joinpath(output_dir, r"ee_indiv_fits_cont_px.pdf"))
fig.clf()
plt.close('all')

indiv_fits_nd = list()
for i,x in enumerate(ee_contact_diam_non_dim_df_list_dropna):
    print(i,x['Source_File'].iloc[0])
    print('popt_x2 , popt_k2 , popt_high_plateau , popt_mid_plateau')
    fig, ax = plt.subplots()
    print(ee_indiv_logistic_contnd_df.iloc[i]['popt_x2'],',', ee_indiv_logistic_contnd_df.iloc[i]['popt_k2'],',', ee_indiv_logistic_contnd_df.iloc[i]['popt_high_plateau'],',', ee_indiv_logistic_contnd_df.iloc[i]['popt_mid_plateau'])
    ax.scatter(x['Time (minutes)'], x['Contact Surface Length (N.D.)'], marker='.')
    #plt.plot(x_fit, logistic(x_fit, ee_indiv_logistic_contnd_df.iloc[i]['popt_x2'], ee_indiv_logistic_contnd_df.iloc[i]['popt_k2'], ee_indiv_logistic_contnd_df.iloc[i]['popt_high_plateau'], ee_indiv_logistic_contnd_df.iloc[i]['popt_mid_plateau'], ee_indiv_logistic_contnd_df.iloc[i]['popt_nu']), c='r')
    ax.plot(x_fit, logistic(x_fit, ee_indiv_logistic_contnd_df.iloc[i]['popt_x2'], ee_indiv_logistic_contnd_df.iloc[i]['popt_k2'], ee_indiv_logistic_contnd_df.iloc[i]['popt_high_plateau'], ee_indiv_logistic_contnd_df.iloc[i]['popt_mid_plateau']), c='r')
    ax.axhline(bounds_contnd[0][3], c='k')
    ax.axhline(bounds_contnd[0][2], c='k')
    ax.axhline(bounds_contnd[1][3], c='k')
    ax.axhline(bounds_contnd[1][2], c='k')
    ax.set_xlim((-50,120))
    ax.set_ylim((0, ee_contact_diam_non_dim_df['Contact Surface Length (N.D.)'].max()))
    ax.set_title('No.%d_'%i + x['Source_File'].iloc[0], fontdict={'size':10})
    plt.draw()
    #plt.show()
    #indiv_fits_nd.append( logistic(x_fit, ee_indiv_logistic_contnd_df.iloc[i]['popt_x2'], ee_indiv_logistic_contnd_df.iloc[i]['popt_k2'], ee_indiv_logistic_contnd_df.iloc[i]['popt_high_plateau'], ee_indiv_logistic_contnd_df.iloc[i]['popt_mid_plateau'], ee_indiv_logistic_contnd_df.iloc[i]['popt_nu']) )
    indiv_fits_nd.append( logistic(x_fit, ee_indiv_logistic_contnd_df.iloc[i]['popt_x2'], ee_indiv_logistic_contnd_df.iloc[i]['popt_k2'], ee_indiv_logistic_contnd_df.iloc[i]['popt_high_plateau'], ee_indiv_logistic_contnd_df.iloc[i]['popt_mid_plateau']) )
multipdf(Path.joinpath(output_dir, r"ee_indiv_fits_cont_nd.pdf"))
fig.clf()
plt.close('all')


indiv_fits_angle = list()
for i,x in enumerate(ee_list_dropna):
    print(i,x['Source_File'].iloc[0])
    print('popt_x2 , popt_k2 , popt_high_plateau , popt_mid_plateau')
    print(ee_indiv_logistic_angle_df.iloc[i]['popt_x2'],',', ee_indiv_logistic_angle_df.iloc[i]['popt_k2'],',', ee_indiv_logistic_angle_df.iloc[i]['popt_high_plateau'],',', ee_indiv_logistic_angle_df.iloc[i]['popt_mid_plateau'])
    fig, ax = plt.subplots()
    ax.scatter(x['Time (minutes)'], x['Average Angle (deg)'], marker='.')
    #plt.plot(x_fit, logistic(x_fit, ee_indiv_logistic_angle_df.iloc[i]['popt_x2'], ee_indiv_logistic_angle_df.iloc[i]['popt_k2'], ee_indiv_logistic_angle_df.iloc[i]['popt_high_plateau'], ee_indiv_logistic_angle_df.iloc[i]['popt_mid_plateau'], ee_indiv_logistic_angle_df.iloc[i]['popt_nu']), c='r')
    ax.plot(x_fit, logistic(x_fit, ee_indiv_logistic_angle_df.iloc[i]['popt_x2'], ee_indiv_logistic_angle_df.iloc[i]['popt_k2'], ee_indiv_logistic_angle_df.iloc[i]['popt_high_plateau'], ee_indiv_logistic_angle_df.iloc[i]['popt_mid_plateau']), c='r')
    ax.axhline(bounds_angle[0][3], c='k')
    ax.axhline(bounds_angle[0][2], c='k')
    ax.axhline(bounds_angle[1][3], c='k')
    ax.axhline(bounds_angle[1][2], c='k')
    ax.set_xlim((-50,120))
    ax.set_ylim((0,180))
    ax.set_title('No.%d_'%i + x['Source_File'].iloc[0], fontdict={'size':10})
    plt.draw()
    #plt.show()
    #indiv_fits_angle.append( logistic(x_fit, ee_indiv_logistic_angle_df.iloc[i]['popt_x2'], ee_indiv_logistic_angle_df.iloc[i]['popt_k2'], ee_indiv_logistic_angle_df.iloc[i]['popt_high_plateau'], ee_indiv_logistic_angle_df.iloc[i]['popt_mid_plateau'], ee_indiv_logistic_angle_df.iloc[i]['popt_nu']) )
    indiv_fits_angle.append( logistic(x_fit, ee_indiv_logistic_angle_df.iloc[i]['popt_x2'], ee_indiv_logistic_angle_df.iloc[i]['popt_k2'], ee_indiv_logistic_angle_df.iloc[i]['popt_high_plateau'], ee_indiv_logistic_angle_df.iloc[i]['popt_mid_plateau']) )
multipdf(Path.joinpath(output_dir, r"ee_indiv_fits_angles.pdf"))
fig.clf()
plt.close('all')


### Also for meso-meso data plot all individual fits and overlay them with
### respective data to check quality of fits:
indiv_fits_mm = list()
#x_fit = np.arange(-240, 241, step=0.1)#The stepsize was originally left blank, ie. it was 1
for i,x in enumerate(mm_list_dropna):
    print(i,x['Source_File'].iloc[0])
    print('popt_x2 , popt_k2 , popt_high_plateau , popt_mid_plateau')
    print(mm_indiv_logistic_cont_df.iloc[i]['popt_x2'],',', mm_indiv_logistic_cont_df.iloc[i]['popt_k2'],',', mm_indiv_logistic_cont_df.iloc[i]['popt_high_plateau'],',', mm_indiv_logistic_cont_df.iloc[i]['popt_mid_plateau'])
    fig, ax = plt.subplots()
    ax.scatter(x['Time (minutes)'], x['Contact Surface Length (px)'], marker='.')
    #plt.plot(x_fit, logistic(x_fit, mm_indiv_logistic_cont_df.iloc[i]['popt_x2'], mm_indiv_logistic_cont_df.iloc[i]['popt_k2'], mm_indiv_logistic_cont_df.iloc[i]['popt_high_plateau'], mm_indiv_logistic_cont_df.iloc[i]['popt_mid_plateau'], mm_indiv_logistic_cont_df.iloc[i]['popt_nu']), c='r')
    ax.plot(x_fit, logistic(x_fit, mm_indiv_logistic_cont_df.iloc[i]['popt_x2'], mm_indiv_logistic_cont_df.iloc[i]['popt_k2'], mm_indiv_logistic_cont_df.iloc[i]['popt_high_plateau'], mm_indiv_logistic_cont_df.iloc[i]['popt_mid_plateau']), c='r')
    ax.axhline(bounds_cont[0][3], c='k')
    ax.axhline(bounds_cont[0][2], c='k')
    ax.axhline(bounds_cont[1][3], c='k')
    ax.axhline(bounds_cont[1][2], c='k')
    ax.set_xlim((-50,120))
    ax.set_ylim((0, mm_list_long['Contact Surface Length (px)'].max()))
    ax.set_title('No.%d_'%i + x['Source_File'].iloc[0], fontdict={'size':10})
    plt.draw()
    #plt.show()
    #indiv_fits_mm.append( logistic(x_fit, mm_indiv_logistic_cont_df.iloc[i]['popt_x2'], mm_indiv_logistic_cont_df.iloc[i]['popt_k2'], mm_indiv_logistic_cont_df.iloc[i]['popt_high_plateau'], mm_indiv_logistic_cont_df.iloc[i]['popt_mid_plateau'], mm_indiv_logistic_cont_df.iloc[i]['popt_nu']) )
    indiv_fits_mm.append( logistic(x_fit, mm_indiv_logistic_cont_df.iloc[i]['popt_x2'], mm_indiv_logistic_cont_df.iloc[i]['popt_k2'], mm_indiv_logistic_cont_df.iloc[i]['popt_high_plateau'], mm_indiv_logistic_cont_df.iloc[i]['popt_mid_plateau']) )
multipdf(Path.joinpath(output_dir, r"mm_indiv_fits_cont_px.pdf"))
fig.clf()
plt.close('all')


indiv_fits_mm_nd = list()
for i,x in enumerate(mm_contact_diam_non_dim_df_list_dropna):
    print(i,x['Source_File'].iloc[0])
    print('popt_x2 , popt_k2 , popt_high_plateau , popt_mid_plateau')
    print(mm_indiv_logistic_contnd_df.iloc[i]['popt_x2'],',', mm_indiv_logistic_contnd_df.iloc[i]['popt_k2'],',', mm_indiv_logistic_contnd_df.iloc[i]['popt_high_plateau'],',', mm_indiv_logistic_contnd_df.iloc[i]['popt_mid_plateau'])
    fig, ax = plt.subplots()
    ax.scatter(x['Time (minutes)'], x['Contact Surface Length (N.D.)'], marker='.')
    #plt.plot(x_fit, logistic(x_fit, mm_indiv_logistic_contnd_df.iloc[i]['popt_x2'], mm_indiv_logistic_contnd_df.iloc[i]['popt_k2'], mm_indiv_logistic_contnd_df.iloc[i]['popt_high_plateau'], mm_indiv_logistic_contnd_df.iloc[i]['popt_mid_plateau'], mm_indiv_logistic_contnd_df.iloc[i]['popt_nu']), c='r')
    ax.plot(x_fit, logistic(x_fit, mm_indiv_logistic_contnd_df.iloc[i]['popt_x2'], mm_indiv_logistic_contnd_df.iloc[i]['popt_k2'], mm_indiv_logistic_contnd_df.iloc[i]['popt_high_plateau'], mm_indiv_logistic_contnd_df.iloc[i]['popt_mid_plateau']), c='r')
    ax.axhline(bounds_contnd[0][3], c='k')
    ax.axhline(bounds_contnd[0][2], c='k')
    ax.axhline(bounds_contnd[1][3], c='k')
    ax.axhline(bounds_contnd[1][2], c='k')
    ax.set_xlim((-50,120))
    ax.set_ylim((0, mm_contact_diam_non_dim_df['Contact Surface Length (N.D.)'].max()))
    ax.set_title('No.%d_'%i + x['Source_File'].iloc[0], fontdict={'size':10})
    plt.draw()
    #plt.show()
    #indiv_fits_mm_nd.append( logistic(x_fit, mm_indiv_logistic_contnd_df.iloc[i]['popt_x2'], mm_indiv_logistic_contnd_df.iloc[i]['popt_k2'], mm_indiv_logistic_contnd_df.iloc[i]['popt_high_plateau'], mm_indiv_logistic_contnd_df.iloc[i]['popt_mid_plateau'], mm_indiv_logistic_contnd_df.iloc[i]['popt_nu']) )
    indiv_fits_mm_nd.append( logistic(x_fit, mm_indiv_logistic_contnd_df.iloc[i]['popt_x2'], mm_indiv_logistic_contnd_df.iloc[i]['popt_k2'], mm_indiv_logistic_contnd_df.iloc[i]['popt_high_plateau'], mm_indiv_logistic_contnd_df.iloc[i]['popt_mid_plateau']) )
multipdf(Path.joinpath(output_dir, r"mm_indiv_fits_cont_nd.pdf"))
fig.clf()
plt.close('all')


indiv_fits_mm_angle = list()
for i,x in enumerate(mm_list_dropna):
    print(i,x['Source_File'].iloc[0])
    print('popt_x2 , popt_k2 , popt_high_plateau , popt_mid_plateau')
    print(mm_indiv_logistic_angle_df.iloc[i]['popt_x2'],',', mm_indiv_logistic_angle_df.iloc[i]['popt_k2'],',', mm_indiv_logistic_angle_df.iloc[i]['popt_high_plateau'],',', mm_indiv_logistic_angle_df.iloc[i]['popt_mid_plateau'])
    fig, ax = plt.subplots()
    ax.scatter(x['Time (minutes)'], x['Average Angle (deg)'], marker='.')
    #plt.plot(x_fit, logistic(x_fit, mm_indiv_logistic_angle_df.iloc[i]['popt_x2'], mm_indiv_logistic_angle_df.iloc[i]['popt_k2'], mm_indiv_logistic_angle_df.iloc[i]['popt_high_plateau'], mm_indiv_logistic_angle_df.iloc[i]['popt_mid_plateau'], mm_indiv_logistic_angle_df.iloc[i]['popt_nu']), c='r')
    ax.plot(x_fit, logistic(x_fit, mm_indiv_logistic_angle_df.iloc[i]['popt_x2'], mm_indiv_logistic_angle_df.iloc[i]['popt_k2'], mm_indiv_logistic_angle_df.iloc[i]['popt_high_plateau'], mm_indiv_logistic_angle_df.iloc[i]['popt_mid_plateau']), c='r')
    ax.axhline(bounds_angle[0][3], c='k')
    ax.axhline(bounds_angle[0][2], c='k')
    ax.axhline(bounds_angle[1][3], c='k')
    ax.axhline(bounds_angle[1][2], c='k')
    ax.set_xlim((-50,120))
    ax.set_ylim((0,180))
    ax.set_title('No.%d_'%i + x['Source_File'].iloc[0], fontdict={'size':10})
    plt.draw()
    #plt.show()
    #indiv_fits_mm_angle.append( logistic(x_fit, mm_indiv_logistic_angle_df.iloc[i]['popt_x2'], mm_indiv_logistic_angle_df.iloc[i]['popt_k2'], mm_indiv_logistic_angle_df.iloc[i]['popt_high_plateau'], mm_indiv_logistic_angle_df.iloc[i]['popt_mid_plateau'], mm_indiv_logistic_angle_df.iloc[i]['popt_nu']) )
    indiv_fits_mm_angle.append( logistic(x_fit, mm_indiv_logistic_angle_df.iloc[i]['popt_x2'], mm_indiv_logistic_angle_df.iloc[i]['popt_k2'], mm_indiv_logistic_angle_df.iloc[i]['popt_high_plateau'], mm_indiv_logistic_angle_df.iloc[i]['popt_mid_plateau']) )
multipdf(Path.joinpath(output_dir, r"mm_indiv_fits_angles.pdf"))
fig.clf()
plt.close('all')



### Also for endo-endo data plot all individual fits and overlay them with
### respective data to check quality of fits:
indiv_fits_nn = list()
#x_fit = np.arange(-240, 241, step=0.1)#The stepsize was originally left blank, ie. it was 1
for i,x in enumerate(nn_list_dropna):
    print(i,x['Source_File'].iloc[0])
    print('popt_x2 , popt_k2 , popt_high_plateau , popt_mid_plateau')
    print(nn_indiv_logistic_cont_df.iloc[i]['popt_x2'],',', nn_indiv_logistic_cont_df.iloc[i]['popt_k2'],',', nn_indiv_logistic_cont_df.iloc[i]['popt_high_plateau'],',', nn_indiv_logistic_cont_df.iloc[i]['popt_mid_plateau'])
    fig, ax = plt.subplots()
    ax.scatter(x['Time (minutes)'], x['Contact Surface Length (px)'], marker='.')
    #plt.plot(x_fit, logistic(x_fit, nn_indiv_logistic_cont_df.iloc[i]['popt_x2'], nn_indiv_logistic_cont_df.iloc[i]['popt_k2'], nn_indiv_logistic_cont_df.iloc[i]['popt_high_plateau'], nn_indiv_logistic_cont_df.iloc[i]['popt_mid_plateau'], nn_indiv_logistic_cont_df.iloc[i]['popt_nu']), c='r')
    ax.plot(x_fit, logistic(x_fit, nn_indiv_logistic_cont_df.iloc[i]['popt_x2'], nn_indiv_logistic_cont_df.iloc[i]['popt_k2'], nn_indiv_logistic_cont_df.iloc[i]['popt_high_plateau'], nn_indiv_logistic_cont_df.iloc[i]['popt_mid_plateau']), c='r')
    ax.axhline(bounds_contnd[0][3], c='k')
    ax.axhline(bounds_cont[0][2], c='k')
    ax.axhline(bounds_cont[1][3], c='k')
    ax.axhline(bounds_cont[1][2], c='k')
    ax.set_xlim((-50,120))
    ax.set_ylim((0, nn_list_long['Contact Surface Length (px)'].max()))
    ax.set_title('No.%d_'%i + x['Source_File'].iloc[0], fontdict={'size':10})
    plt.draw()
    #plt.show()
    #indiv_fits_nn.append( logistic(x_fit, nn_indiv_logistic_cont_df.iloc[i]['popt_x2'], nn_indiv_logistic_cont_df.iloc[i]['popt_k2'], nn_indiv_logistic_cont_df.iloc[i]['popt_high_plateau'], nn_indiv_logistic_cont_df.iloc[i]['popt_mid_plateau'], nn_indiv_logistic_cont_df.iloc[i]['popt_nu']) )
    indiv_fits_nn.append( logistic(x_fit, nn_indiv_logistic_cont_df.iloc[i]['popt_x2'], nn_indiv_logistic_cont_df.iloc[i]['popt_k2'], nn_indiv_logistic_cont_df.iloc[i]['popt_high_plateau'], nn_indiv_logistic_cont_df.iloc[i]['popt_mid_plateau']) )
multipdf(Path.joinpath(output_dir, r"nn_indiv_fits_cont_px.pdf"))
fig.clf()
plt.close('all')


indiv_fits_nn_nd = list()
for i,x in enumerate(nn_contact_diam_non_dim_df_list_dropna):
    print(i,x['Source_File'].iloc[0])
    print('popt_x2 , popt_k2 , popt_high_plateau , popt_mid_plateau')
    print(nn_indiv_logistic_contnd_df.iloc[i]['popt_x2'],',', nn_indiv_logistic_contnd_df.iloc[i]['popt_k2'],',', nn_indiv_logistic_contnd_df.iloc[i]['popt_high_plateau'],',', nn_indiv_logistic_contnd_df.iloc[i]['popt_mid_plateau'])
    fig, ax = plt.subplots()
    ax.scatter(x['Time (minutes)'], x['Contact Surface Length (N.D.)'], marker='.')
    #plt.plot(x_fit, logistic(x_fit, nn_indiv_logistic_contnd_df.iloc[i]['popt_x2'], nn_indiv_logistic_contnd_df.iloc[i]['popt_k2'], nn_indiv_logistic_contnd_df.iloc[i]['popt_high_plateau'], nn_indiv_logistic_contnd_df.iloc[i]['popt_mid_plateau'], nn_indiv_logistic_contnd_df.iloc[i]['popt_nu']), c='r')
    ax.plot(x_fit, logistic(x_fit, nn_indiv_logistic_contnd_df.iloc[i]['popt_x2'], nn_indiv_logistic_contnd_df.iloc[i]['popt_k2'], nn_indiv_logistic_contnd_df.iloc[i]['popt_high_plateau'], nn_indiv_logistic_contnd_df.iloc[i]['popt_mid_plateau']), c='r')
    ax.axhline(bounds_contnd[0][3], c='k')
    ax.axhline(bounds_contnd[0][2], c='k')
    ax.axhline(bounds_contnd[1][3], c='k')
    ax.axhline(bounds_contnd[1][2], c='k')
    ax.set_xlim((-50,120))
    ax.set_ylim((0, nn_contact_diam_non_dim_df['Contact Surface Length (N.D.)'].max()))
    ax.set_title('No.%d_'%i + x['Source_File'].iloc[0], fontdict={'size':10})
    plt.draw()
    #plt.show()
    #indiv_fits_nn_nd.append( logistic(x_fit, nn_indiv_logistic_contnd_df.iloc[i]['popt_x2'], nn_indiv_logistic_contnd_df.iloc[i]['popt_k2'], nn_indiv_logistic_contnd_df.iloc[i]['popt_high_plateau'], nn_indiv_logistic_contnd_df.iloc[i]['popt_mid_plateau'], nn_indiv_logistic_contnd_df.iloc[i]['popt_nu']) )
    indiv_fits_nn_nd.append( logistic(x_fit, nn_indiv_logistic_contnd_df.iloc[i]['popt_x2'], nn_indiv_logistic_contnd_df.iloc[i]['popt_k2'], nn_indiv_logistic_contnd_df.iloc[i]['popt_high_plateau'], nn_indiv_logistic_contnd_df.iloc[i]['popt_mid_plateau']) )
multipdf(Path.joinpath(output_dir, r"nn_indiv_fits_cont_nd.pdf"))
fig.clf()
plt.close('all')


indiv_fits_nn_angle = list()
for i,x in enumerate(nn_list_dropna):
    print(i,x['Source_File'].iloc[0])
    print('popt_x2 , popt_k2 , popt_high_plateau , popt_mid_plateau')
    print(nn_indiv_logistic_angle_df.iloc[i]['popt_x2'],',', nn_indiv_logistic_angle_df.iloc[i]['popt_k2'],',', nn_indiv_logistic_angle_df.iloc[i]['popt_high_plateau'],',', nn_indiv_logistic_angle_df.iloc[i]['popt_mid_plateau'])
    fig, ax = plt.subplots()
    ax.scatter(x['Time (minutes)'], x['Average Angle (deg)'], marker='.')
    #plt.plot(x_fit, logistic(x_fit, nn_indiv_logistic_angle_df.iloc[i]['popt_x2'], nn_indiv_logistic_angle_df.iloc[i]['popt_k2'], nn_indiv_logistic_angle_df.iloc[i]['popt_high_plateau'], nn_indiv_logistic_angle_df.iloc[i]['popt_mid_plateau'], nn_indiv_logistic_angle_df.iloc[i]['popt_nu']), c='r')
    ax.plot(x_fit, logistic(x_fit, nn_indiv_logistic_angle_df.iloc[i]['popt_x2'], nn_indiv_logistic_angle_df.iloc[i]['popt_k2'], nn_indiv_logistic_angle_df.iloc[i]['popt_high_plateau'], nn_indiv_logistic_angle_df.iloc[i]['popt_mid_plateau']), c='r')
    ax.axhline(bounds_angle[0][3], c='k')
    ax.axhline(bounds_angle[0][2], c='k')
    ax.axhline(bounds_angle[1][3], c='k')
    ax.axhline(bounds_angle[1][2], c='k')
    ax.set_xlim((-50,120))
    ax.set_ylim((0,180))
    ax.set_title('No.%d_'%i + x['Source_File'].iloc[0], fontdict={'size':10})
    plt.draw()
    #plt.show()
    #indiv_fits_nn_angle.append( logistic(x_fit, nn_indiv_logistic_angle_df.iloc[i]['popt_x2'], nn_indiv_logistic_angle_df.iloc[i]['popt_k2'], nn_indiv_logistic_angle_df.iloc[i]['popt_high_plateau'], nn_indiv_logistic_angle_df.iloc[i]['popt_mid_plateau'], nn_indiv_logistic_angle_df.iloc[i]['popt_nu']) )
    indiv_fits_nn_angle.append( logistic(x_fit, nn_indiv_logistic_angle_df.iloc[i]['popt_x2'], nn_indiv_logistic_angle_df.iloc[i]['popt_k2'], nn_indiv_logistic_angle_df.iloc[i]['popt_high_plateau'], nn_indiv_logistic_angle_df.iloc[i]['popt_mid_plateau']) )
multipdf(Path.joinpath(output_dir, r"nn_indiv_fits_angles.pdf"))
fig.clf()
plt.close('all')



### Plot all individual fits and overlay them with respective data to check
### quality of fits:
indiv_fits_eeCcadMO = list()
#x_fit = np.arange(-240, 241, step=0.1)#The stepsize was originally left blank, ie. it was 1
for i,x in enumerate(eeCcadMO_list_dropna):
    print(i,x['Source_File'].iloc[0])
    print('popt_x2 , popt_k2 , popt_high_plateau , popt_mid_plateau')
    print(eeCcadMO_indiv_logistic_cont_df.iloc[i]['popt_x2'],',', eeCcadMO_indiv_logistic_cont_df.iloc[i]['popt_k2'],',', eeCcadMO_indiv_logistic_cont_df.iloc[i]['popt_high_plateau'],',', eeCcadMO_indiv_logistic_cont_df.iloc[i]['popt_mid_plateau'])
    fig, ax = plt.subplots()
    ax.scatter(x['Time (minutes)'], x['Contact Surface Length (px)'], marker='.')
    #plt.plot(x_fit, logistic(x_fit, eeCcadMO_indiv_logistic_cont_df.iloc[i]['popt_x2'], eeCcadMO_indiv_logistic_cont_df.iloc[i]['popt_k2'], eeCcadMO_indiv_logistic_cont_df.iloc[i]['popt_high_plateau'], eeCcadMO_indiv_logistic_cont_df.iloc[i]['popt_mid_plateau'], eeCcadMO_indiv_logistic_cont_df.iloc[i]['popt_nu']), c='r')
    ax.plot(x_fit, logistic(x_fit, eeCcadMO_indiv_logistic_cont_df.iloc[i]['popt_x2'], eeCcadMO_indiv_logistic_cont_df.iloc[i]['popt_k2'], eeCcadMO_indiv_logistic_cont_df.iloc[i]['popt_high_plateau'], eeCcadMO_indiv_logistic_cont_df.iloc[i]['popt_mid_plateau']), c='r')
    ax.axhline(bounds_cont[0][3], c='k')
    ax.axhline(bounds_cont[0][2], c='k')
    ax.axhline(bounds_cont[1][3], c='k')
    ax.axhline(bounds_cont[1][2], c='k')
    ax.set_xlim((-50,120))
    ax.set_ylim((0, eeCcadMO_list_long['Contact Surface Length (px)'].max()))
    ax.set_title('No.%d_'%i + x['Source_File'].iloc[0], fontdict={'size':10})
    plt.draw()
    #plt.show()
    #indiv_fits_eeCcadMO.append( logistic(x_fit, eeCcadMO_indiv_logistic_cont_df.iloc[i]['popt_x2'], eeCcadMO_indiv_logistic_cont_df.iloc[i]['popt_k2'], eeCcadMO_indiv_logistic_cont_df.iloc[i]['popt_high_plateau'], eeCcadMO_indiv_logistic_cont_df.iloc[i]['popt_mid_plateau'], eeCcadMO_indiv_logistic_cont_df.iloc[i]['popt_nu']) )
    indiv_fits_eeCcadMO.append( logistic(x_fit, eeCcadMO_indiv_logistic_cont_df.iloc[i]['popt_x2'], eeCcadMO_indiv_logistic_cont_df.iloc[i]['popt_k2'], eeCcadMO_indiv_logistic_cont_df.iloc[i]['popt_high_plateau'], eeCcadMO_indiv_logistic_cont_df.iloc[i]['popt_mid_plateau']) )
multipdf(Path.joinpath(output_dir, r"eeCcadMO_indiv_fits_cont_px.pdf"))
fig.clf()
plt.close('all')


indiv_fits_eeCcadMO_nd = list()
for i,x in enumerate(eeCcadMO_contact_diam_non_dim_df_list_dropna):
    print(i,x['Source_File'].iloc[0])
    print('popt_x2 , popt_k2 , popt_high_plateau , popt_mid_plateau')
    print(eeCcadMO_indiv_logistic_contnd_df.iloc[i]['popt_x2'],',', eeCcadMO_indiv_logistic_contnd_df.iloc[i]['popt_k2'],',', eeCcadMO_indiv_logistic_contnd_df.iloc[i]['popt_high_plateau'],',', eeCcadMO_indiv_logistic_contnd_df.iloc[i]['popt_mid_plateau'])
    fig, ax = plt.subplots()
    ax.scatter(x['Time (minutes)'], x['Contact Surface Length (N.D.)'], marker='.')
    #plt.plot(x_fit, logistic(x_fit, eeCcadMO_indiv_logistic_contnd_df.iloc[i]['popt_x2'], eeCcadMO_indiv_logistic_contnd_df.iloc[i]['popt_k2'], eeCcadMO_indiv_logistic_contnd_df.iloc[i]['popt_high_plateau'], eeCcadMO_indiv_logistic_contnd_df.iloc[i]['popt_mid_plateau'], eeCcadMO_indiv_logistic_contnd_df.iloc[i]['popt_nu']), c='r')
    ax.plot(x_fit, logistic(x_fit, eeCcadMO_indiv_logistic_contnd_df.iloc[i]['popt_x2'], eeCcadMO_indiv_logistic_contnd_df.iloc[i]['popt_k2'], eeCcadMO_indiv_logistic_contnd_df.iloc[i]['popt_high_plateau'], eeCcadMO_indiv_logistic_contnd_df.iloc[i]['popt_mid_plateau']), c='r')
    ax.axhline(bounds_contnd[0][3], c='k')
    ax.axhline(bounds_contnd[0][2], c='k')
    ax.axhline(bounds_contnd[1][3], c='k')
    ax.axhline(bounds_contnd[1][2], c='k')
    ax.set_xlim((-50,120))
    ax.set_ylim((0, eeCcadMO_contact_diam_non_dim_df['Contact Surface Length (N.D.)'].max()))
    ax.set_title('No.%d_'%i + x['Source_File'].iloc[0], fontdict={'size':10})
    plt.draw()
    #plt.show()
    #indiv_fits_eeCcadMO_nd.append( logistic(x_fit, eeCcadMO_indiv_logistic_contnd_df.iloc[i]['popt_x2'], eeCcadMO_indiv_logistic_contnd_df.iloc[i]['popt_k2'], eeCcadMO_indiv_logistic_contnd_df.iloc[i]['popt_high_plateau'], eeCcadMO_indiv_logistic_contnd_df.iloc[i]['popt_mid_plateau'], eeCcadMO_indiv_logistic_contnd_df.iloc[i]['popt_nu']) )
    indiv_fits_eeCcadMO_nd.append( logistic(x_fit, eeCcadMO_indiv_logistic_contnd_df.iloc[i]['popt_x2'], eeCcadMO_indiv_logistic_contnd_df.iloc[i]['popt_k2'], eeCcadMO_indiv_logistic_contnd_df.iloc[i]['popt_high_plateau'], eeCcadMO_indiv_logistic_contnd_df.iloc[i]['popt_mid_plateau']) )
multipdf(Path.joinpath(output_dir, r"eeCcadMO_indiv_fits_cont_nd.pdf"))
fig.clf()
plt.close('all')


indiv_fits_eeCcadMO_angle = list()
for i,x in enumerate(eeCcadMO_list_dropna):
    print(i,x['Source_File'].iloc[0])
    print('popt_x2 , popt_k2 , popt_high_plateau , popt_mid_plateau')
    print(eeCcadMO_indiv_logistic_angle_df.iloc[i]['popt_x2'],',', eeCcadMO_indiv_logistic_angle_df.iloc[i]['popt_k2'],',', eeCcadMO_indiv_logistic_angle_df.iloc[i]['popt_high_plateau'],',', eeCcadMO_indiv_logistic_angle_df.iloc[i]['popt_mid_plateau'])
    fig, ax = plt.subplots()
    ax.scatter(x['Time (minutes)'], x['Average Angle (deg)'], marker='.')
    #plt.plot(x_fit, logistic(x_fit, eeCcadMO_indiv_logistic_angle_df.iloc[i]['popt_x2'], eeCcadMO_indiv_logistic_angle_df.iloc[i]['popt_k2'], eeCcadMO_indiv_logistic_angle_df.iloc[i]['popt_high_plateau'], eeCcadMO_indiv_logistic_angle_df.iloc[i]['popt_mid_plateau'], eeCcadMO_indiv_logistic_angle_df.iloc[i]['popt_nu']), c='r')
    ax.plot(x_fit, logistic(x_fit, eeCcadMO_indiv_logistic_angle_df.iloc[i]['popt_x2'], eeCcadMO_indiv_logistic_angle_df.iloc[i]['popt_k2'], eeCcadMO_indiv_logistic_angle_df.iloc[i]['popt_high_plateau'], eeCcadMO_indiv_logistic_angle_df.iloc[i]['popt_mid_plateau']), c='r')
    ax.axhline(bounds_angle[0][3], c='k')
    ax.axhline(bounds_angle[0][2], c='k')
    ax.axhline(bounds_angle[1][3], c='k')
    ax.axhline(bounds_angle[1][2], c='k')
    ax.set_xlim((-50,120))
    ax.set_ylim((0,180))
    ax.set_title('No.%d_'%i + x['Source_File'].iloc[0], fontdict={'size':10})
    plt.draw()
    #plt.show()
    #indiv_fits_eeCcadMO_angle.append( logistic(x_fit, eeCcadMO_indiv_logistic_angle_df.iloc[i]['popt_x2'], eeCcadMO_indiv_logistic_angle_df.iloc[i]['popt_k2'], eeCcadMO_indiv_logistic_angle_df.iloc[i]['popt_high_plateau'], eeCcadMO_indiv_logistic_angle_df.iloc[i]['popt_mid_plateau'], eeCcadMO_indiv_logistic_angle_df.iloc[i]['popt_nu']) )
    indiv_fits_eeCcadMO_angle.append( logistic(x_fit, eeCcadMO_indiv_logistic_angle_df.iloc[i]['popt_x2'], eeCcadMO_indiv_logistic_angle_df.iloc[i]['popt_k2'], eeCcadMO_indiv_logistic_angle_df.iloc[i]['popt_high_plateau'], eeCcadMO_indiv_logistic_angle_df.iloc[i]['popt_mid_plateau']) )
multipdf(Path.joinpath(output_dir, r"eeCcadMO_indiv_fits_angles.pdf"))
fig.clf()
plt.close('all')





### Plot all individual fits and overlay them with respective data to check
### quality of fits:
indiv_fits_eeRcadMO = list()
#x_fit = np.arange(-240, 241, step=0.1)#The stepsize was originally left blank, ie. it was 1
for i,x in enumerate(eeRcadMO_list_dropna):
    print(i,x['Source_File'].iloc[0])
    print('popt_x2 , popt_k2 , popt_high_plateau , popt_mid_plateau')
    print(eeRcadMO_indiv_logistic_cont_df.iloc[i]['popt_x2'],',', eeRcadMO_indiv_logistic_cont_df.iloc[i]['popt_k2'],',', eeRcadMO_indiv_logistic_cont_df.iloc[i]['popt_high_plateau'],',', eeRcadMO_indiv_logistic_cont_df.iloc[i]['popt_mid_plateau'])
    fig, ax = plt.subplots()
    ax.scatter(x['Time (minutes)'], x['Contact Surface Length (px)'], marker='.')
    #plt.plot(x_fit, logistic(x_fit, eeRcadMO_indiv_logistic_cont_df.iloc[i]['popt_x2'], eeRcadMO_indiv_logistic_cont_df.iloc[i]['popt_k2'], eeRcadMO_indiv_logistic_cont_df.iloc[i]['popt_high_plateau'], eeRcadMO_indiv_logistic_cont_df.iloc[i]['popt_mid_plateau'], eeRcadMO_indiv_logistic_cont_df.iloc[i]['popt_nu']), c='r')
    ax.plot(x_fit, logistic(x_fit, eeRcadMO_indiv_logistic_cont_df.iloc[i]['popt_x2'], eeRcadMO_indiv_logistic_cont_df.iloc[i]['popt_k2'], eeRcadMO_indiv_logistic_cont_df.iloc[i]['popt_high_plateau'], eeRcadMO_indiv_logistic_cont_df.iloc[i]['popt_mid_plateau']), c='r')
    ax.axhline(bounds_cont[0][3], c='k')
    ax.axhline(bounds_cont[0][2], c='k')
    ax.axhline(bounds_cont[1][3], c='k')
    ax.axhline(bounds_cont[1][2], c='k')
    ax.set_xlim((-50,120))
    ax.set_ylim((0, eeRcadMO_list_long['Contact Surface Length (px)'].max()))
    ax.set_title('No.%d_'%i + x['Source_File'].iloc[0], fontdict={'size':10})
    plt.draw()
    #plt.show()
    #indiv_fits_eeRcadMO.append( logistic(x_fit, eeRcadMO_indiv_logistic_cont_df.iloc[i]['popt_x2'], eeRcadMO_indiv_logistic_cont_df.iloc[i]['popt_k2'], eeRcadMO_indiv_logistic_cont_df.iloc[i]['popt_high_plateau'], eeRcadMO_indiv_logistic_cont_df.iloc[i]['popt_mid_plateau'], eeRcadMO_indiv_logistic_cont_df.iloc[i]['popt_nu']) )
    indiv_fits_eeRcadMO.append( logistic(x_fit, eeRcadMO_indiv_logistic_cont_df.iloc[i]['popt_x2'], eeRcadMO_indiv_logistic_cont_df.iloc[i]['popt_k2'], eeRcadMO_indiv_logistic_cont_df.iloc[i]['popt_high_plateau'], eeRcadMO_indiv_logistic_cont_df.iloc[i]['popt_mid_plateau']) )
multipdf(Path.joinpath(output_dir, r"eeRcadMO_indiv_fits_cont_px.pdf"))
fig.clf()
plt.close('all')


indiv_fits_eeRcadMO_nd = list()
for i,x in enumerate(eeRcadMO_contact_diam_non_dim_df_list_dropna):
    print(i,x['Source_File'].iloc[0])
    print('popt_x2 , popt_k2 , popt_high_plateau , popt_mid_plateau')
    print(eeRcadMO_indiv_logistic_contnd_df.iloc[i]['popt_x2'],',', eeRcadMO_indiv_logistic_contnd_df.iloc[i]['popt_k2'],',', eeRcadMO_indiv_logistic_contnd_df.iloc[i]['popt_high_plateau'],',', eeRcadMO_indiv_logistic_contnd_df.iloc[i]['popt_mid_plateau'])
    fig, ax = plt.subplots()
    ax.scatter(x['Time (minutes)'], x['Contact Surface Length (N.D.)'], marker='.')
    #plt.plot(x_fit, logistic(x_fit, eeRcadMO_indiv_logistic_contnd_df.iloc[i]['popt_x2'], eeRcadMO_indiv_logistic_contnd_df.iloc[i]['popt_k2'], eeRcadMO_indiv_logistic_contnd_df.iloc[i]['popt_high_plateau'], eeRcadMO_indiv_logistic_contnd_df.iloc[i]['popt_mid_plateau'], eeRcadMO_indiv_logistic_contnd_df.iloc[i]['popt_nu']), c='r')
    ax.plot(x_fit, logistic(x_fit, eeRcadMO_indiv_logistic_contnd_df.iloc[i]['popt_x2'], eeRcadMO_indiv_logistic_contnd_df.iloc[i]['popt_k2'], eeRcadMO_indiv_logistic_contnd_df.iloc[i]['popt_high_plateau'], eeRcadMO_indiv_logistic_contnd_df.iloc[i]['popt_mid_plateau']), c='r')
    ax.axhline(bounds_contnd[0][3], c='k')
    ax.axhline(bounds_contnd[0][2], c='k')
    ax.axhline(bounds_contnd[1][3], c='k')
    ax.axhline(bounds_contnd[1][2], c='k')
    ax.set_xlim((-50,120))
    ax.set_ylim((0, eeRcadMO_contact_diam_non_dim_df['Contact Surface Length (N.D.)'].max()))
    ax.set_title('No.%d_'%i + x['Source_File'].iloc[0], fontdict={'size':10})
    plt.draw()
    #plt.show()
    #indiv_fits_eeRcadMO_nd.append( logistic(x_fit, eeRcadMO_indiv_logistic_contnd_df.iloc[i]['popt_x2'], eeRcadMO_indiv_logistic_contnd_df.iloc[i]['popt_k2'], eeRcadMO_indiv_logistic_contnd_df.iloc[i]['popt_high_plateau'], eeRcadMO_indiv_logistic_contnd_df.iloc[i]['popt_mid_plateau'], eeRcadMO_indiv_logistic_contnd_df.iloc[i]['popt_nu']) )
    indiv_fits_eeRcadMO_nd.append( logistic(x_fit, eeRcadMO_indiv_logistic_contnd_df.iloc[i]['popt_x2'], eeRcadMO_indiv_logistic_contnd_df.iloc[i]['popt_k2'], eeRcadMO_indiv_logistic_contnd_df.iloc[i]['popt_high_plateau'], eeRcadMO_indiv_logistic_contnd_df.iloc[i]['popt_mid_plateau']) )
multipdf(Path.joinpath(output_dir, r"eeRcadMO_indiv_fits_cont_nd.pdf"))
fig.clf()
plt.close('all')


indiv_fits_eeRcadMO_angle = list()
for i,x in enumerate(eeRcadMO_list_dropna):
    print(i,x['Source_File'].iloc[0])
    print('popt_x2 , popt_k2 , popt_high_plateau , popt_mid_plateau')
    print(eeRcadMO_indiv_logistic_angle_df.iloc[i]['popt_x2'],',', eeRcadMO_indiv_logistic_angle_df.iloc[i]['popt_k2'],',', eeRcadMO_indiv_logistic_angle_df.iloc[i]['popt_high_plateau'],',', eeRcadMO_indiv_logistic_angle_df.iloc[i]['popt_mid_plateau'])
    fig, ax = plt.subplots()
    ax.scatter(x['Time (minutes)'], x['Average Angle (deg)'], marker='.')
    #plt.plot(x_fit, logistic(x_fit, eeRcadMO_indiv_logistic_angle_df.iloc[i]['popt_x2'], eeRcadMO_indiv_logistic_angle_df.iloc[i]['popt_k2'], eeRcadMO_indiv_logistic_angle_df.iloc[i]['popt_high_plateau'], eeRcadMO_indiv_logistic_angle_df.iloc[i]['popt_mid_plateau'], eeRcadMO_indiv_logistic_angle_df.iloc[i]['popt_nu']), c='r')
    ax.plot(x_fit, logistic(x_fit, eeRcadMO_indiv_logistic_angle_df.iloc[i]['popt_x2'], eeRcadMO_indiv_logistic_angle_df.iloc[i]['popt_k2'], eeRcadMO_indiv_logistic_angle_df.iloc[i]['popt_high_plateau'], eeRcadMO_indiv_logistic_angle_df.iloc[i]['popt_mid_plateau']), c='r')
    ax.axhline(bounds_angle[0][3], c='k')
    ax.axhline(bounds_angle[0][2], c='k')
    ax.axhline(bounds_angle[1][3], c='k')
    ax.axhline(bounds_angle[1][2], c='k')
    ax.set_xlim((-50,120))
    ax.set_ylim((0,180))
    ax.set_title('No.%d_'%i + x['Source_File'].iloc[0], fontdict={'size':10})
    plt.draw()
    #plt.show()
    #indiv_fits_eeRcadMO_angle.append( logistic(x_fit, eeRcadMO_indiv_logistic_angle_df.iloc[i]['popt_x2'], eeRcadMO_indiv_logistic_angle_df.iloc[i]['popt_k2'], eeRcadMO_indiv_logistic_angle_df.iloc[i]['popt_high_plateau'], eeRcadMO_indiv_logistic_angle_df.iloc[i]['popt_mid_plateau'], eeRcadMO_indiv_logistic_angle_df.iloc[i]['popt_nu']) )
    indiv_fits_eeRcadMO_angle.append( logistic(x_fit, eeRcadMO_indiv_logistic_angle_df.iloc[i]['popt_x2'], eeRcadMO_indiv_logistic_angle_df.iloc[i]['popt_k2'], eeRcadMO_indiv_logistic_angle_df.iloc[i]['popt_high_plateau'], eeRcadMO_indiv_logistic_angle_df.iloc[i]['popt_mid_plateau']) )
multipdf(Path.joinpath(output_dir, r"eeRcadMO_indiv_fits_angles.pdf"))
fig.clf()
plt.close('all')





### Plot all individual fits and overlay them with respective data to check
### quality of fits:
indiv_fits_eeRCcadMO = list()
#x_fit = np.arange(-240, 241, step=0.1)#The stepsize was originally left blank, ie. it was 1
for i,x in enumerate(eeRCcadMO_list_dropna):
    print(i,x['Source_File'].iloc[0])
    print('popt_x2 , popt_k2 , popt_high_plateau , popt_mid_plateau')
    print(eeRCcadMO_indiv_logistic_cont_df.iloc[i]['popt_x2'],',', eeRCcadMO_indiv_logistic_cont_df.iloc[i]['popt_k2'],',', eeRCcadMO_indiv_logistic_cont_df.iloc[i]['popt_high_plateau'],',', eeRCcadMO_indiv_logistic_cont_df.iloc[i]['popt_mid_plateau'])
    fig, ax = plt.subplots()
    ax.scatter(x['Time (minutes)'], x['Contact Surface Length (px)'], marker='.')
    #plt.plot(x_fit, logistic(x_fit, eeRCcadMO_indiv_logistic_cont_df.iloc[i]['popt_x2'], eeRCcadMO_indiv_logistic_cont_df.iloc[i]['popt_k2'], eeRCcadMO_indiv_logistic_cont_df.iloc[i]['popt_high_plateau'], eeRCcadMO_indiv_logistic_cont_df.iloc[i]['popt_mid_plateau'], eeRCcadMO_indiv_logistic_cont_df.iloc[i]['popt_nu']), c='r')
    ax.plot(x_fit, logistic(x_fit, eeRCcadMO_indiv_logistic_cont_df.iloc[i]['popt_x2'], eeRCcadMO_indiv_logistic_cont_df.iloc[i]['popt_k2'], eeRCcadMO_indiv_logistic_cont_df.iloc[i]['popt_high_plateau'], eeRCcadMO_indiv_logistic_cont_df.iloc[i]['popt_mid_plateau']), c='r')
    ax.axhline(bounds_cont[0][3], c='k')
    ax.axhline(bounds_cont[0][2], c='k')
    ax.axhline(bounds_cont[1][3], c='k')
    ax.axhline(bounds_cont[1][2], c='k')
    ax.set_xlim((-50,120))
    ax.set_ylim((0, eeRCcadMO_list_long['Contact Surface Length (px)'].max()))
    ax.set_title('No.%d_'%i + x['Source_File'].iloc[0], fontdict={'size':10})
    plt.draw()
    #plt.show()
    #indiv_fits_eeRCcadMO.append( logistic(x_fit, eeRCcadMO_indiv_logistic_cont_df.iloc[i]['popt_x2'], eeRCcadMO_indiv_logistic_cont_df.iloc[i]['popt_k2'], eeRCcadMO_indiv_logistic_cont_df.iloc[i]['popt_high_plateau'], eeRCcadMO_indiv_logistic_cont_df.iloc[i]['popt_mid_plateau'], eeRCcadMO_indiv_logistic_cont_df.iloc[i]['popt_nu']) )
    indiv_fits_eeRCcadMO.append( logistic(x_fit, eeRCcadMO_indiv_logistic_cont_df.iloc[i]['popt_x2'], eeRCcadMO_indiv_logistic_cont_df.iloc[i]['popt_k2'], eeRCcadMO_indiv_logistic_cont_df.iloc[i]['popt_high_plateau'], eeRCcadMO_indiv_logistic_cont_df.iloc[i]['popt_mid_plateau']) )
multipdf(Path.joinpath(output_dir, r"eeRCcadMO_indiv_fits_cont_px.pdf"))
fig.clf()
plt.close('all')


indiv_fits_eeRCcadMO_nd = list()
for i,x in enumerate(eeRCcadMO_contact_diam_non_dim_df_list_dropna):
    print(i,x['Source_File'].iloc[0])
    print('popt_x2 , popt_k2 , popt_high_plateau , popt_mid_plateau')
    print(eeRCcadMO_indiv_logistic_contnd_df.iloc[i]['popt_x2'],',', eeRCcadMO_indiv_logistic_contnd_df.iloc[i]['popt_k2'],',', eeRCcadMO_indiv_logistic_contnd_df.iloc[i]['popt_high_plateau'],',', eeRCcadMO_indiv_logistic_contnd_df.iloc[i]['popt_mid_plateau'])
    fig, ax = plt.subplots()
    ax.scatter(x['Time (minutes)'], x['Contact Surface Length (N.D.)'], marker='.')
    #plt.plot(x_fit, logistic(x_fit, eeRCcadMO_indiv_logistic_contnd_df.iloc[i]['popt_x2'], eeRCcadMO_indiv_logistic_contnd_df.iloc[i]['popt_k2'], eeRCcadMO_indiv_logistic_contnd_df.iloc[i]['popt_high_plateau'], eeRCcadMO_indiv_logistic_contnd_df.iloc[i]['popt_mid_plateau'], eeRCcadMO_indiv_logistic_contnd_df.iloc[i]['popt_nu']), c='r')
    ax.plot(x_fit, logistic(x_fit, eeRCcadMO_indiv_logistic_contnd_df.iloc[i]['popt_x2'], eeRCcadMO_indiv_logistic_contnd_df.iloc[i]['popt_k2'], eeRCcadMO_indiv_logistic_contnd_df.iloc[i]['popt_high_plateau'], eeRCcadMO_indiv_logistic_contnd_df.iloc[i]['popt_mid_plateau']), c='r')
    ax.axhline(bounds_contnd[0][3], c='k')
    ax.axhline(bounds_contnd[0][2], c='k')
    ax.axhline(bounds_contnd[1][3], c='k')
    ax.axhline(bounds_contnd[1][2], c='k')
    ax.set_xlim((-50,120))
    ax.set_ylim((0, eeRCcadMO_contact_diam_non_dim_df['Contact Surface Length (N.D.)'].max()))
    ax.set_title('No.%d_'%i + x['Source_File'].iloc[0], fontdict={'size':10})
    plt.draw()
    #plt.show()
    #indiv_fits_eeRCcadMO_nd.append( logistic(x_fit, eeRCcadMO_indiv_logistic_contnd_df.iloc[i]['popt_x2'], eeRCcadMO_indiv_logistic_contnd_df.iloc[i]['popt_k2'], eeRCcadMO_indiv_logistic_contnd_df.iloc[i]['popt_high_plateau'], eeRCcadMO_indiv_logistic_contnd_df.iloc[i]['popt_mid_plateau'], eeRCcadMO_indiv_logistic_contnd_df.iloc[i]['popt_nu']) )
    indiv_fits_eeRCcadMO_nd.append( logistic(x_fit, eeRCcadMO_indiv_logistic_contnd_df.iloc[i]['popt_x2'], eeRCcadMO_indiv_logistic_contnd_df.iloc[i]['popt_k2'], eeRCcadMO_indiv_logistic_contnd_df.iloc[i]['popt_high_plateau'], eeRCcadMO_indiv_logistic_contnd_df.iloc[i]['popt_mid_plateau']) )
multipdf(Path.joinpath(output_dir, r"eeRCcadMO_indiv_fits_cont_nd.pdf"))
fig.clf()
plt.close('all')


indiv_fits_eeRCcadMO_angle = list()
for i,x in enumerate(eeRCcadMO_list_dropna):
    print(i,x['Source_File'].iloc[0])
    print('popt_x2 , popt_k2 , popt_high_plateau , popt_mid_plateau')
    print(eeRCcadMO_indiv_logistic_angle_df.iloc[i]['popt_x2'],',', eeRCcadMO_indiv_logistic_angle_df.iloc[i]['popt_k2'],',', eeRCcadMO_indiv_logistic_angle_df.iloc[i]['popt_high_plateau'],',', eeRCcadMO_indiv_logistic_angle_df.iloc[i]['popt_mid_plateau'])
    fig, ax = plt.subplots()
    ax.scatter(x['Time (minutes)'], x['Average Angle (deg)'], marker='.')
    #plt.plot(x_fit, logistic(x_fit, eeRCcadMO_indiv_logistic_angle_df.iloc[i]['popt_x2'], eeRCcadMO_indiv_logistic_angle_df.iloc[i]['popt_k2'], eeRCcadMO_indiv_logistic_angle_df.iloc[i]['popt_high_plateau'], eeRCcadMO_indiv_logistic_angle_df.iloc[i]['popt_mid_plateau'], eeRCcadMO_indiv_logistic_angle_df.iloc[i]['popt_nu']), c='r')
    ax.plot(x_fit, logistic(x_fit, eeRCcadMO_indiv_logistic_angle_df.iloc[i]['popt_x2'], eeRCcadMO_indiv_logistic_angle_df.iloc[i]['popt_k2'], eeRCcadMO_indiv_logistic_angle_df.iloc[i]['popt_high_plateau'], eeRCcadMO_indiv_logistic_angle_df.iloc[i]['popt_mid_plateau']), c='r')
    ax.axhline(bounds_angle[0][3], c='k')
    ax.axhline(bounds_angle[0][2], c='k')
    ax.axhline(bounds_angle[1][3], c='k')
    ax.axhline(bounds_angle[1][2], c='k')
    ax.set_xlim((-50,120))
    ax.set_ylim((0,180))
    ax.set_title('No.%d_'%i + x['Source_File'].iloc[0], fontdict={'size':10})
    plt.draw()
    #plt.show()
    #indiv_fits_eeRCcadMO_angle.append( logistic(x_fit, eeRCcadMO_indiv_logistic_angle_df.iloc[i]['popt_x2'], eeRCcadMO_indiv_logistic_angle_df.iloc[i]['popt_k2'], eeRCcadMO_indiv_logistic_angle_df.iloc[i]['popt_high_plateau'], eeRCcadMO_indiv_logistic_angle_df.iloc[i]['popt_mid_plateau'], eeRCcadMO_indiv_logistic_angle_df.iloc[i]['popt_nu']) )
    indiv_fits_eeRCcadMO_angle.append( logistic(x_fit, eeRCcadMO_indiv_logistic_angle_df.iloc[i]['popt_x2'], eeRCcadMO_indiv_logistic_angle_df.iloc[i]['popt_k2'], eeRCcadMO_indiv_logistic_angle_df.iloc[i]['popt_high_plateau'], eeRCcadMO_indiv_logistic_angle_df.iloc[i]['popt_mid_plateau']) )
multipdf(Path.joinpath(output_dir, r"eeRCcadMO_indiv_fits_angles.pdf"))
fig.clf()
plt.close('all')





### Plot all individual fits and overlay them with respective data to check
### quality of fits:
plt.close('all')
indiv_fits_em = list()
x_fit = np.arange(-240, 241, step=0.1)#The stepsize was originally left blank, ie. it was 1
for i,x in enumerate(em_list_dropna):
    print(i,x['Source_File'].iloc[0])
    print('popt_x2 , popt_k2 , popt_high_plateau , popt_mid_plateau')
    print(em_indiv_logistic_cont_df.iloc[i]['popt_x2'],',', em_indiv_logistic_cont_df.iloc[i]['popt_k2'],',', em_indiv_logistic_cont_df.iloc[i]['popt_high_plateau'],',', em_indiv_logistic_cont_df.iloc[i]['popt_mid_plateau'])
    fig, ax = plt.subplots()
    ax.scatter(x['Time (minutes)'], x['Contact Surface Length (px)'], marker='.')
    ax.plot(x_fit, logistic(x_fit, em_indiv_logistic_cont_df.iloc[i]['popt_x2'], em_indiv_logistic_cont_df.iloc[i]['popt_k2'], em_indiv_logistic_cont_df.iloc[i]['popt_high_plateau'], em_indiv_logistic_cont_df.iloc[i]['popt_mid_plateau']), c='r')
    ax.axhline(bounds_cont[0][3], c='k')
    ax.axhline(bounds_cont[0][2], c='k')
    ax.axhline(bounds_cont[1][3], c='k')
    ax.axhline(bounds_cont[1][2], c='k')
    ax.set_xlim((-50,120))
    ax.set_ylim((0, em_list_long['Contact Surface Length (px)'].max()))
    ax.set_title('No.%d_'%i + x['Source_File'].iloc[0], fontdict={'size':10})
    plt.draw()
    #plt.show()
    indiv_fits_em.append( logistic(x_fit, em_indiv_logistic_cont_df.iloc[i]['popt_x2'], em_indiv_logistic_cont_df.iloc[i]['popt_k2'], em_indiv_logistic_cont_df.iloc[i]['popt_high_plateau'], em_indiv_logistic_cont_df.iloc[i]['popt_mid_plateau']) )
multipdf(Path.joinpath(output_dir, r"em_indiv_fits_cont_px.pdf"))
fig.clf()
plt.close('all')

indiv_fits_em_nd = list()
for i,x in enumerate(em_contact_diam_non_dim_df_list_dropna):
    print(i,x['Source_File'].iloc[0])
    print('popt_x2 , popt_k2 , popt_high_plateau , popt_mid_plateau')
    fig, ax = plt.subplots()
    print(em_indiv_logistic_contnd_df.iloc[i]['popt_x2'],',', em_indiv_logistic_contnd_df.iloc[i]['popt_k2'],',', em_indiv_logistic_contnd_df.iloc[i]['popt_high_plateau'],',', em_indiv_logistic_contnd_df.iloc[i]['popt_mid_plateau'])
    ax.scatter(x['Time (minutes)'], x['Contact Surface Length (N.D.)'], marker='.')
    #plt.plot(x_fit, logistic(x_fit, ee_indiv_logistic_contnd_df.iloc[i]['popt_x2'], ee_indiv_logistic_contnd_df.iloc[i]['popt_k2'], ee_indiv_logistic_contnd_df.iloc[i]['popt_high_plateau'], ee_indiv_logistic_contnd_df.iloc[i]['popt_mid_plateau'], ee_indiv_logistic_contnd_df.iloc[i]['popt_nu']), c='r')
    ax.plot(x_fit, logistic(x_fit, em_indiv_logistic_contnd_df.iloc[i]['popt_x2'], em_indiv_logistic_contnd_df.iloc[i]['popt_k2'], em_indiv_logistic_contnd_df.iloc[i]['popt_high_plateau'], em_indiv_logistic_contnd_df.iloc[i]['popt_mid_plateau']), c='r')
    ax.axhline(bounds_contnd[0][3], c='k')
    ax.axhline(bounds_contnd[0][2], c='k')
    ax.axhline(bounds_contnd[1][3], c='k')
    ax.axhline(bounds_contnd[1][2], c='k')
    ax.set_xlim((-50,120))
    ax.set_ylim((0, em_contact_diam_non_dim_df['Contact Surface Length (N.D.)'].max()))
    ax.set_title('No.%d_'%i + x['Source_File'].iloc[0], fontdict={'size':10})
    plt.draw()
    #plt.show()
    #indiv_fits_nd.append( logistic(x_fit, ee_indiv_logistic_contnd_df.iloc[i]['popt_x2'], ee_indiv_logistic_contnd_df.iloc[i]['popt_k2'], ee_indiv_logistic_contnd_df.iloc[i]['popt_high_plateau'], ee_indiv_logistic_contnd_df.iloc[i]['popt_mid_plateau'], ee_indiv_logistic_contnd_df.iloc[i]['popt_nu']) )
    indiv_fits_em_nd.append( logistic(x_fit, em_indiv_logistic_contnd_df.iloc[i]['popt_x2'], em_indiv_logistic_contnd_df.iloc[i]['popt_k2'], em_indiv_logistic_contnd_df.iloc[i]['popt_high_plateau'], em_indiv_logistic_contnd_df.iloc[i]['popt_mid_plateau']) )
multipdf(Path.joinpath(output_dir, r"em_indiv_fits_cont_nd.pdf"))
fig.clf()
plt.close('all')


indiv_fits_em_angle = list()
for i,x in enumerate(em_list_dropna):
    print(i,x['Source_File'].iloc[0])
    print('popt_x2 , popt_k2 , popt_high_plateau , popt_mid_plateau')
    print(em_indiv_logistic_angle_df.iloc[i]['popt_x2'],',', em_indiv_logistic_angle_df.iloc[i]['popt_k2'],',', em_indiv_logistic_angle_df.iloc[i]['popt_high_plateau'],',', em_indiv_logistic_angle_df.iloc[i]['popt_mid_plateau'])
    fig, ax = plt.subplots()
    ax.scatter(x['Time (minutes)'], x['Average Angle (deg)'], marker='.')
    ax.plot(x_fit, logistic(x_fit, em_indiv_logistic_angle_df.iloc[i]['popt_x2'], em_indiv_logistic_angle_df.iloc[i]['popt_k2'], em_indiv_logistic_angle_df.iloc[i]['popt_high_plateau'], em_indiv_logistic_angle_df.iloc[i]['popt_mid_plateau']), c='r')
    ax.axhline(bounds_angle[0][3], c='k')
    ax.axhline(bounds_angle[0][2], c='k')
    ax.axhline(bounds_angle[1][3], c='k')
    ax.axhline(bounds_angle[1][2], c='k')
    ax.set_xlim((-50,120))
    ax.set_ylim((0,180))
    ax.set_title('No.%d_'%i + x['Source_File'].iloc[0], fontdict={'size':10})
    plt.draw()
    #plt.show()
    indiv_fits_em_angle.append( logistic(x_fit, em_indiv_logistic_angle_df.iloc[i]['popt_x2'], em_indiv_logistic_angle_df.iloc[i]['popt_k2'], em_indiv_logistic_angle_df.iloc[i]['popt_high_plateau'], em_indiv_logistic_angle_df.iloc[i]['popt_mid_plateau']) )
multipdf(Path.joinpath(output_dir, r"em_indiv_fits_angles.pdf"))
fig.clf()
plt.close('all')




### Plot all individual fits and overlay them with respective data to check
### quality of fits:
plt.close('all')
indiv_fits_eedb = list()
x_fit = np.arange(-240, 241, step=0.1)#The stepsize was originally left blank, ie. it was 1
for i,x in enumerate(eedb_list_dropna):
    print(i,x['Source_File'].iloc[0])
    print('popt_x2 , popt_k2 , popt_high_plateau , popt_mid_plateau')
    print(eedb_indiv_logistic_cont_df.iloc[i]['popt_x2'],',', eedb_indiv_logistic_cont_df.iloc[i]['popt_k2'],',', eedb_indiv_logistic_cont_df.iloc[i]['popt_high_plateau'],',', eedb_indiv_logistic_cont_df.iloc[i]['popt_mid_plateau'])
    fig, ax = plt.subplots()
    ax.scatter(x['Time (minutes)'], x['Contact Surface Length (px)'], marker='.')
    ax.plot(x_fit, logistic(x_fit, eedb_indiv_logistic_cont_df.iloc[i]['popt_x2'], eedb_indiv_logistic_cont_df.iloc[i]['popt_k2'], eedb_indiv_logistic_cont_df.iloc[i]['popt_high_plateau'], eedb_indiv_logistic_cont_df.iloc[i]['popt_mid_plateau']), c='r')
    ax.axhline(bounds_cont[0][3], c='k')
    ax.axhline(bounds_cont[0][2], c='k')
    ax.axhline(bounds_cont[1][3], c='k')
    ax.axhline(bounds_cont[1][2], c='k')
    ax.set_xlim((-50,120))
    ax.set_ylim((0, eedb_list_long['Contact Surface Length (px)'].max()))
    ax.set_title('No.%d_'%i + x['Source_File'].iloc[0], fontdict={'size':10})
    plt.draw()
    #plt.show()
    indiv_fits_eedb.append( logistic(x_fit, eedb_indiv_logistic_cont_df.iloc[i]['popt_x2'], eedb_indiv_logistic_cont_df.iloc[i]['popt_k2'], eedb_indiv_logistic_cont_df.iloc[i]['popt_high_plateau'], eedb_indiv_logistic_cont_df.iloc[i]['popt_mid_plateau']) )
multipdf(Path.joinpath(output_dir, r"eedb_indiv_fits_cont_px.pdf"))
fig.clf()
plt.close('all')

indiv_fits_eedb_nd = list()
for i,x in enumerate(eedb_contact_diam_non_dim_df_list_dropna):
    print(i,x['Source_File'].iloc[0])
    print('popt_x2 , popt_k2 , popt_high_plateau , popt_mid_plateau')
    fig, ax = plt.subplots()
    print(eedb_indiv_logistic_contnd_df.iloc[i]['popt_x2'],',', eedb_indiv_logistic_contnd_df.iloc[i]['popt_k2'],',', eedb_indiv_logistic_contnd_df.iloc[i]['popt_high_plateau'],',', eedb_indiv_logistic_contnd_df.iloc[i]['popt_mid_plateau'])
    ax.scatter(x['Time (minutes)'], x['Contact Surface Length (N.D.)'], marker='.')
    ax.plot(x_fit, logistic(x_fit, eedb_indiv_logistic_contnd_df.iloc[i]['popt_x2'], eedb_indiv_logistic_contnd_df.iloc[i]['popt_k2'], eedb_indiv_logistic_contnd_df.iloc[i]['popt_high_plateau'], eedb_indiv_logistic_contnd_df.iloc[i]['popt_mid_plateau']), c='r')
    ax.axhline(bounds_contnd[0][3], c='k')
    ax.axhline(bounds_contnd[0][2], c='k')
    ax.axhline(bounds_contnd[1][3], c='k')
    ax.axhline(bounds_contnd[1][2], c='k')
    ax.set_xlim((-50,120))
    ax.set_ylim((0, eedb_contact_diam_non_dim_df['Contact Surface Length (N.D.)'].max()))
    ax.set_title('No.%d_'%i + x['Source_File'].iloc[0], fontdict={'size':10})
    plt.draw()
    #plt.show()
    indiv_fits_eedb_nd.append( logistic(x_fit, eedb_indiv_logistic_contnd_df.iloc[i]['popt_x2'], eedb_indiv_logistic_contnd_df.iloc[i]['popt_k2'], eedb_indiv_logistic_contnd_df.iloc[i]['popt_high_plateau'], eedb_indiv_logistic_contnd_df.iloc[i]['popt_mid_plateau']) )
multipdf(Path.joinpath(output_dir, r"eedb_indiv_fits_cont_nd.pdf"))
fig.clf()
plt.close('all')


indiv_fits_eedb_angle = list()
for i,x in enumerate(eedb_list_dropna):
    print(i,x['Source_File'].iloc[0])
    print('popt_x2 , popt_k2 , popt_high_plateau , popt_mid_plateau')
    print(eedb_indiv_logistic_angle_df.iloc[i]['popt_x2'],',', eedb_indiv_logistic_angle_df.iloc[i]['popt_k2'],',', eedb_indiv_logistic_angle_df.iloc[i]['popt_high_plateau'],',', eedb_indiv_logistic_angle_df.iloc[i]['popt_mid_plateau'])
    fig, ax = plt.subplots()
    ax.scatter(x['Time (minutes)'], x['Average Angle (deg)'], marker='.')
    ax.plot(x_fit, logistic(x_fit, eedb_indiv_logistic_angle_df.iloc[i]['popt_x2'], eedb_indiv_logistic_angle_df.iloc[i]['popt_k2'], eedb_indiv_logistic_angle_df.iloc[i]['popt_high_plateau'], eedb_indiv_logistic_angle_df.iloc[i]['popt_mid_plateau']), c='r')
    ax.axhline(bounds_angle[0][3], c='k')
    ax.axhline(bounds_angle[0][2], c='k')
    ax.axhline(bounds_angle[1][3], c='k')
    ax.axhline(bounds_angle[1][2], c='k')
    ax.set_xlim((-50,120))
    ax.set_ylim((0,180))
    ax.set_title('No.%d_'%i + x['Source_File'].iloc[0], fontdict={'size':10})
    plt.draw()
    #plt.show()
    indiv_fits_eedb_angle.append( logistic(x_fit, eedb_indiv_logistic_angle_df.iloc[i]['popt_x2'], eedb_indiv_logistic_angle_df.iloc[i]['popt_k2'], eedb_indiv_logistic_angle_df.iloc[i]['popt_high_plateau'], eedb_indiv_logistic_angle_df.iloc[i]['popt_mid_plateau']) )
multipdf(Path.joinpath(output_dir, r"eedb_indiv_fits_angles.pdf"))
fig.clf()
plt.close('all')





###############################################################################

## For ecto-ecto make a table where parameters of the logistic fits are 
## considered good (True) or bad (False). Then can plot only the parameters that
## actually fit well:
ee_list_dropna_fit_validity_df = pd.read_csv(Path.joinpath(fit_validity_dir, r"ee plateau table contnd.csv"))
ee_list_dropna_fit_validity_df = ee_list_dropna_fit_validity_df[['Source_File', 'low_plat', 'growth_phase', 'high_plat']]
# The 1 line below is only used for determining contact dynamics:
ee_Source_File_valid_low_plat = ee_list_dropna_fit_validity_df.query('low_plat==True')['Source_File']

ee_list_dropna_angle_fit_validity_df = pd.read_csv(Path.joinpath(fit_validity_dir, r"ee plateau table angles.csv"))
ee_list_dropna_angle_fit_validity_df = ee_list_dropna_angle_fit_validity_df[['Source_File', 'low_plat', 'growth_phase', 'high_plat']]
# The 1 line below is only used for determining contact dynamics:
ee_Source_File_angle_valid_low_plat = ee_list_dropna_angle_fit_validity_df.query('low_plat==True')['Source_File']


# Time correct data for ee:
# For dimensional contact diameter:
ee_df_dropna_time_corr_cpu, ee_df_dropna_cant_time_corr = time_correct_data(ee_list_dropna, ee_indiv_logistic_cont_df, ee_list_dropna_fit_validity_df)
time_grouped_ee_df_time_corr_cpu = ee_df_dropna_time_corr_cpu.groupby(['Time (minutes)'])
time_grouped_ee_df_cant_time_corr = ee_df_dropna_cant_time_corr.groupby(['Time (minutes)'])
ee_contact_mean = time_grouped_ee_df_time_corr_cpu['Contact Surface Length (px)'].describe().reset_index()
ee_contact_mean_cant_time_corr = time_grouped_ee_df_cant_time_corr['Contact Surface Length (px)'].describe().reset_index()

# For contact angle:
ee_df_dropna_angle_time_corr_cpu, ee_df_dropna_angle_cant_time_corr = time_correct_data(ee_contact_diam_non_dim_df_list_dropna, ee_indiv_logistic_angle_df, ee_list_dropna_angle_fit_validity_df)
time_grouped_ee_df_angle_time_corr_cpu = ee_df_dropna_angle_time_corr_cpu.groupby(['Time (minutes)'])
time_grouped_ee_df_angle_cant_time_corr = ee_df_dropna_angle_cant_time_corr.groupby(['Time (minutes)'])
ee_angle_mean = time_grouped_ee_df_angle_time_corr_cpu['Average Angle (deg)'].describe().reset_index()
ee_angle_mean_cant_time_corr = time_grouped_ee_df_angle_cant_time_corr['Average Angle (deg)'].describe().reset_index()

# For non-dimensional contact diameter:
ee_fit_df_dropna_time_corr_cpu, ee_fit_df_dropna_cant_time_corr = time_correct_data(ee_contact_diam_non_dim_df_list_dropna, ee_indiv_logistic_contnd_df, ee_list_dropna_fit_validity_df)
time_grouped_ee_fit_df_time_corr_cpu = ee_fit_df_dropna_time_corr_cpu.groupby(['Time (minutes)'])
time_grouped_ee_fit_df_cant_time_corr = ee_fit_df_dropna_cant_time_corr.groupby(['Time (minutes)'])
ee_fit_contact_mean = time_grouped_ee_fit_df_time_corr_cpu['Contact Surface Length (N.D.)'].describe().reset_index()
ee_fit_contact_mean_cant_time_corr = time_grouped_ee_fit_df_cant_time_corr['Contact Surface Length (N.D.)'].describe().reset_index()



# Fit general logistic curve to mean angles:
sig_angle = 1./np.sqrt(ee_angle_mean['count'])**2
ee_genlog_popt_angle, ee_genlog_pcov_angle = curve_fit(general_logistic, ee_angle_mean['Time (minutes)'], ee_angle_mean['mean'], sigma=sig_angle, p0 = (*p0_angle, nu), maxfev=2000)#, bounds=bounds_genlog_contnd)
# ee_genlog_popt_angle_w, ee_genlog_pcov_angle_w = curve_fit(general_logistic_wiki, ee_angle_mean['Time (minutes)'], ee_angle_mean['mean'], sigma=sig_angle, p0 = (*p0_angle, nu), maxfev=2000)#, bounds=bounds_genlog_contnd)
with np.errstate(over='ignore'):
    ee_genlog_y_angle = general_logistic(x_fit, *ee_genlog_popt_angle)
    # ee_genlog_y_angle_w = general_logistic_wiki(x_fit, *ee_genlog_popt_angle_w)

# Can wind up with nans when fitting the general_logsitic, therefore need to 
# mask them if present:
# ee_genlog_y_angle_diff = np.diff(data=ee_genlog_y_angle)
ee_genlog_y_angle_diff = np.diff(np.ma.masked_array(data=ee_genlog_y_angle, mask=~np.isfinite(ee_genlog_y_angle)))
percent90_high = np.argmin(abs(np.cumsum(ee_genlog_y_angle_diff) - np.cumsum(ee_genlog_y_angle_diff).max()*0.95))
percent90_low = np.argmin(abs(np.cumsum(ee_genlog_y_angle_diff) - np.cumsum(ee_genlog_y_angle_diff).max()*0.05))
percent95_high = np.argmin(abs(np.cumsum(ee_genlog_y_angle_diff) - np.cumsum(ee_genlog_y_angle_diff).max()*0.975))
percent95_low = np.argmin(abs(np.cumsum(ee_genlog_y_angle_diff) - np.cumsum(ee_genlog_y_angle_diff).max()*0.025))
#growth_phase_end_ee_genlog_angle = x_fit[percent90_high]
#growth_phase_end_ee_genlog_angle_val = ee_genlog_y_angle[percent90_high]
#growth_phase_start_ee_genlog_angle = x_fit[percent90_low]
#growth_phase_start_ee_genlog_angle_val = ee_genlog_y_angle[percent90_low]
growth_phase_end_ee_angle = x_fit[percent90_high]
growth_phase_end_ee_angle_val = ee_genlog_y_angle[percent90_high]
growth_phase_start_ee_angle = x_fit[percent90_low]
growth_phase_start_ee_angle_val = ee_genlog_y_angle[percent90_low]

growth_phase_end_ee_angle_strict = x_fit[percent95_high]
growth_phase_end_ee_angle_strict_val = ee_genlog_y_angle[percent95_high]
growth_phase_start_ee_angle_strict = x_fit[percent95_low]
growth_phase_start_ee_angle_strict_val = ee_genlog_y_angle[percent95_high]



# Fit general logistic curve to mean dimensional diameters:
sig_cont = 1./np.sqrt(ee_contact_mean['count'])**2
# nu is the last parameter in the general logistic equation -- affects where
# fastest rate of growth (ie. inflection point) is on x-axis:
ee_genlog_popt_cont, ee_genlog_pcov_cont = curve_fit(general_logistic, ee_contact_mean['Time (minutes)'], ee_contact_mean['mean'], sigma=sig_cont, p0 = (*p0_cont, nu), maxfev=2000)#, bounds=bounds_genlog_cont)
with np.errstate(over='ignore'):
    ee_genlog_y_cont = general_logistic(x_fit, *ee_genlog_popt_cont)

ee_genlog_y_cont_diff = np.diff(np.ma.masked_array(data=ee_genlog_y_cont, mask=~np.isfinite(ee_genlog_y_cont)))
#percent95 = np.argmin(abs(np.cumsum(ee_y_contnd_diff) - np.cumsum(ee_y_contnd_diff).max()*0.95))
#percent50 = np.argmin(abs(np.cumsum(ee_y_contnd_diff) - np.cumsum(ee_y_contnd_diff).max()*0.50))
percent90_high = np.argmin(abs(np.cumsum(ee_genlog_y_cont_diff) - np.cumsum(ee_genlog_y_cont_diff).max()*0.95))
percent90_low = np.argmin(abs(np.cumsum(ee_genlog_y_cont_diff) - np.cumsum(ee_genlog_y_cont_diff).max()*0.05))
#growth_phase_end_ee_genlog_contnd = x_fit[percent90_high]
#growth_phase_end_ee_genlog_contnd_val = ee_genlog_y_contnd[percent90_high]
#growth_phase_start_ee_genlog_contnd = x_fit[percent90_low]
#growth_phase_start_ee_genlog_contnd_val = ee_genlog_y_contnd[percent90_low]
growth_phase_end_ee_cont = x_fit[percent90_high]
growth_phase_end_ee_cont_val = ee_genlog_y_cont[percent90_high]
growth_phase_start_ee_cont = x_fit[percent90_low]
growth_phase_start_ee_cont_val = ee_genlog_y_cont[percent90_low]






# Fit general logistic functions to the time corrected data:
# Fit general logistic curve to mean non-dimensionalized diameters:
sig_contnd = 1./np.sqrt(ee_fit_contact_mean['count'])**2
# nu is the last parameter in the general logistic equation -- affects where
# fastest rate of growth (ie. inflection point) is on x-axis:
ee_genlog_popt_contnd, ee_genlog_pcov_contnd = curve_fit(general_logistic, ee_fit_contact_mean['Time (minutes)'], ee_fit_contact_mean['mean'], sigma=sig_contnd, p0 = (*p0_contnd, nu), maxfev=2000)#, bounds=bounds_genlog_contnd)
# ee_genlog_popt_contnd_weightless, ee_genlog_pcov_contnd_weightless = curve_fit(general_logistic, ee_fit_contact_mean['Time (minutes)'], ee_fit_contact_mean['mean'], p0 = (*p0_contnd, nu), maxfev=2000)#, bounds=bounds_genlog_contnd)
with np.errstate(over='ignore'):
    ee_genlog_y_contnd = general_logistic(x_fit, *ee_genlog_popt_contnd)
    # ee_genlog_y_contnd_weightless = general_logistic(x_fit, *ee_genlog_popt_contnd_weightless)

ee_genlog_y_contnd_diff = np.diff(np.ma.masked_array(data=ee_genlog_y_contnd, mask=~np.isfinite(ee_genlog_y_contnd)))
### Allow one of: 10%, 5%, 2%, or 0.2% growth in the plateau regions (evenly
### distributed between the low and high plateaus) to define the growth phase
### of the logistic fits with varying strictness

#percent95 = np.argmin(abs(np.cumsum(ee_y_contnd_diff) - np.cumsum(ee_y_contnd_diff).max()*0.95))
#percent50 = np.argmin(abs(np.cumsum(ee_y_contnd_diff) - np.cumsum(ee_y_contnd_diff).max()*0.50))
percent90_high = np.argmin(abs(np.cumsum(ee_genlog_y_contnd_diff) - np.cumsum(ee_genlog_y_contnd_diff).max()*0.95))
percent90_low = np.argmin(abs(np.cumsum(ee_genlog_y_contnd_diff) - np.cumsum(ee_genlog_y_contnd_diff).max()*0.05))
percent95_high = np.argmin(abs(np.cumsum(ee_genlog_y_contnd_diff) - np.cumsum(ee_genlog_y_contnd_diff).max()*0.975))
percent95_low = np.argmin(abs(np.cumsum(ee_genlog_y_contnd_diff) - np.cumsum(ee_genlog_y_contnd_diff).max()*0.025))
percent98_high = np.argmin(abs(np.cumsum(ee_genlog_y_contnd_diff) - np.cumsum(ee_genlog_y_contnd_diff).max()*0.99))
percent98_low = np.argmin(abs(np.cumsum(ee_genlog_y_contnd_diff) - np.cumsum(ee_genlog_y_contnd_diff).max()*0.01))
percent100_high = np.argmin(abs(np.cumsum(ee_genlog_y_contnd_diff) - np.cumsum(ee_genlog_y_contnd_diff).max()*.999))
percent100_low = np.argmin(abs(np.cumsum(ee_genlog_y_contnd_diff) - np.cumsum(ee_genlog_y_contnd_diff).max()*0.001))
#growth_phase_end_ee_genlog_contnd = x_fit[percent90_high]
#growth_phase_end_ee_genlog_contnd_val = ee_genlog_y_contnd[percent90_high]
#growth_phase_start_ee_genlog_contnd = x_fit[percent90_low]
#growth_phase_start_ee_genlog_contnd_val = ee_genlog_y_contnd[percent90_low]
growth_phase_end_ee_contnd = x_fit[percent90_high]
growth_phase_end_ee_contnd_val = ee_genlog_y_contnd[percent90_high]
growth_phase_start_ee_contnd = x_fit[percent90_low]
growth_phase_start_ee_contnd_val = ee_genlog_y_contnd[percent90_low]

growth_phase_end_ee_contnd_strict = x_fit[percent95_high]
growth_phase_end_ee_contnd_strict_val = ee_genlog_y_contnd[percent95_high]
growth_phase_start_ee_contnd_strict = x_fit[percent95_low]
growth_phase_start_ee_contnd_strict_val = ee_genlog_y_contnd[percent95_high]

growth_phase_end_ee_contnd_v_strict = x_fit[percent98_high]
growth_phase_end_ee_contnd_v_strict_val = ee_genlog_y_contnd[percent98_high]
growth_phase_start_ee_contnd_v_strict = x_fit[percent98_low]
growth_phase_start_ee_contnd_v_strict_val = ee_genlog_y_contnd[percent98_high]

growth_phase_end_ee_contnd_absolute = x_fit[percent100_high]
growth_phase_end_ee_contnd_absolute_val = ee_genlog_y_contnd[percent100_high]
growth_phase_start_ee_contnd_absolute = x_fit[percent100_low]
growth_phase_start_ee_contnd_absolute_val = ee_genlog_y_contnd[percent100_high]




# Also return the angle mean for the contact-based fit and vice-versa:
ee_angle_mean_for_contnd_fit = time_grouped_ee_fit_df_time_corr_cpu['Average Angle (deg)'].describe().reset_index()
# Fit general logistic curve to mean angles aligned according to contnd:
sig_angle = 1./np.sqrt(ee_angle_mean_for_contnd_fit['count'])**2
ee_genlog_popt_angle_contndfit, ee_genlog_pcov_angle_contndfit = curve_fit(general_logistic, ee_angle_mean_for_contnd_fit['Time (minutes)'], ee_angle_mean_for_contnd_fit['mean'], sigma=sig_angle, p0 = (*p0_angle, nu), maxfev=2000)#, bounds=bounds_genlog_contnd)
with np.errstate(over='ignore'):
    ee_genlog_y_angle_contndfit = general_logistic(x_fit, *ee_genlog_popt_angle_contndfit)



ee_contnd_mean_for_angle_fit = time_grouped_ee_df_angle_time_corr_cpu['Contact Surface Length (N.D.)'].describe().reset_index()
# Fit general logistic curve to mean non-dimensionalized contact diameters
# aligned according to angle data:
sig_contnd = 1./np.sqrt(ee_contnd_mean_for_angle_fit['count'])**2
# nu is the last parameter in the general logistic equation -- affects where
# fastest rate of growth (ie. inflection point) is on x-axis:
ee_genlog_popt_contnd_anglefit, ee_genlog_pcov_contnd_anglefit = curve_fit(general_logistic, ee_contnd_mean_for_angle_fit['Time (minutes)'], ee_contnd_mean_for_angle_fit['mean'], sigma=sig_contnd, p0 = (*p0_contnd, nu), maxfev=2000)#, bounds=bounds_genlog_contnd)
with np.errstate(over='ignore'):
    ee_genlog_y_contnd_anglefit = general_logistic(x_fit, *ee_genlog_popt_contnd_anglefit)


 

# Length of growth phase:
ee_angle_growth_dur = growth_phase_end_ee_angle - growth_phase_start_ee_angle
ee_cont_growth_dur = growth_phase_end_ee_cont - growth_phase_start_ee_cont
ee_contnd_growth_dur = growth_phase_end_ee_contnd - growth_phase_start_ee_contnd


# Magnitude of change during growth phase:
ee_angle_growth_mag = growth_phase_end_ee_angle_val - growth_phase_start_ee_angle_val
ee_cont_growth_mag = growth_phase_end_ee_cont_val - growth_phase_start_ee_cont_val
ee_contnd_growth_mag = growth_phase_end_ee_contnd_val - growth_phase_start_ee_contnd_val


# Average rate of change during growth phase:
ee_growth_rate_lin_angle = ee_angle_growth_mag / ee_angle_growth_dur
ee_growth_rate_lin_cont = ee_cont_growth_mag / ee_cont_growth_dur
ee_growth_rate_lin_contnd = ee_contnd_growth_mag / ee_contnd_growth_dur




    
###############################################################################  
### Find rate of change of fits at the inflection point of each curve:
temps_long = pd.concat([x[['temperature_degC', 'Source_File']] for x in ee_list])
temps = temps_long.groupby('Source_File').mean()
temps = [x['temperature_degC'].iloc[0] for x in ee_list]
growth_rates = [np.diff(x) for x in indiv_fits]
max_growth_rates = [max(x) for x in growth_rates]
max_growth_rates_dropna = pd.DataFrame({'max_growth_rates':max_growth_rates}).mask(~ee_list_dropna_fit_validity_df['growth_phase'])
max_growth_rates_dropna.dropna(inplace=True)

growth_temps_df = pd.DataFrame(data=list(zip(temps, max_growth_rates)), columns=['Temperature (degC)','Max Growth Rates'])
growth_temps_df_dropna = growth_temps_df.mask(~ee_list_dropna_fit_validity_df['growth_phase'])
growth_temps_df_dropna.dropna(inplace=True)
### Show how these fastest rates of change are distributed
fig, kde = plt.subplots(figsize=(width_1col,height_1col))
kde = sns.distplot(max_growth_rates_dropna['max_growth_rates'], bins=120)
kde_x = kde.lines[0].get_xdata()
kde_y = kde.lines[0].get_ydata()
kde_maxid = signal.find_peaks(kde_y, distance=30)[0]
print(kde_x[kde_maxid])
fig.savefig(Path.joinpath(output_dir, r"ee_cont_fastest growth.tif"), dpi=dpi_graph)

growth_rates_nd = [np.diff(x) for x in indiv_fits_nd]
max_growth_rates_nd = [max(x) for x in growth_rates_nd]
### Show how these fastest rates of change are distributed
fig, kde_nd = plt.subplots(figsize=(width_1col,height_1col))
kde_nd = sns.distplot(max_growth_rates_nd, bins=120)
kde_nd_x = kde_nd.lines[0].get_xdata()
kde_nd_y = kde_nd.lines[0].get_ydata()
kde_nd_maxid = signal.find_peaks(kde_nd_y)[0]
print(kde_nd_x[kde_nd_maxid])
fig.savefig(Path.joinpath(output_dir, r"ee_contnd_fastest growth.tif"), dpi=dpi_graph)

growth_rates_angle = [np.diff(x) for x in indiv_fits_angle]
max_growth_rates_angle = [max(x) for x in growth_rates_angle]
### Show how these fastest rates of change are distributed
fig, kde_angle = plt.subplots(figsize=(width_1col,height_1col))
kde_angle = sns.distplot(max_growth_rates_angle, bins=120)
kde_angle_x = kde_angle.lines[0].get_xdata()
kde_angle_y = kde_angle.lines[0].get_ydata()
kde_angle_maxid = signal.find_peaks(kde_angle_y)[0]
print(kde_angle_x[kde_angle_maxid])
fig.savefig(Path.joinpath(output_dir, r"ee_angle_fastest growth.tif"), dpi=dpi_graph)


###############################################################################
# Example of sigmoidal kinetics in individual adhering cells:

# A nice logistic example:
plt.clf()
fig10, ax10 = plt.subplots(figsize=(width_1col,height_1col))
ax10.plot(ee_list[11]['Time (minutes)'], ee_list[11]['Contact Surface Length (px)']/ee_first_valid_radius[11], c='lightgrey')
ax10.scatter(ee_list[11]['Time (minutes)'], ee_list[11]['Contact Surface Length (px)']/ee_first_valid_radius[11], marker='.', c='k', zorder=10)
ax10.set_ylim(0,4.0)
ax10.draw
fig10.savefig(Path.joinpath(output_dir, r"ee_contnd_logistic_example.tif"), dpi=dpi_graph)
fig10.clf()

fig11, ax11 = plt.subplots(figsize=(width_1col,height_1col))
ax11.plot(ee_list[46]['Time (minutes)'], ee_list[46]['Contact Surface Length (px)']/ee_first_valid_radius[46], c='lightgrey')
ax11.scatter(ee_list[46]['Time (minutes)'], ee_list[46]['Contact Surface Length (px)']/ee_first_valid_radius[46], marker='.', c='k', zorder=10)
ax11.set_ylim(0,4.0)
ax11.draw
fig11.savefig(Path.joinpath(output_dir, r"ee_contnd_rapid_inc_example.tif"), dpi=dpi_graph)
fig11.clf()

fig12, ax12 = plt.subplots(figsize=(width_1col,height_1col))
ax12.plot(ee_list[12]['Time (minutes)'], ee_list[12]['Contact Surface Length (px)']/ee_first_valid_radius[12], c='lightgrey')
ax12.scatter(ee_list[12]['Time (minutes)'], ee_list[12]['Contact Surface Length (px)']/ee_first_valid_radius[12], marker='.', c='k', zorder=10)
ax12.set_ylim(0,4.0)
ax12.draw
fig12.savefig(Path.joinpath(output_dir, r"ee_contnd_lowplat_inc_example.tif"), dpi=dpi_graph)
fig12.clf()

fig13, ax13 = plt.subplots(figsize=(width_1col,height_1col))
ax13.plot(ee_list[48]['Time (minutes)'], ee_list[48]['Contact Surface Length (px)']/ee_first_valid_radius[48], c='lightgrey')
ax13.scatter(ee_list[48]['Time (minutes)'], ee_list[48]['Contact Surface Length (px)']/ee_first_valid_radius[48], marker='.', c='k', zorder=10)
ax13.set_ylim(0,4.0)
ax13.draw
fig13.savefig(Path.joinpath(output_dir, r"ee_contnd_lowplat_inc_example2.tif"), dpi=dpi_graph)
fig13.clf()

# An exponential example:
#2017-08-10 xl wt ecto-fdx wt ecto-rda 20.5C crop2.csv
fig, ax = plt.subplots(figsize=(width_1col,height_1col))
ax.plot(ee_list[17]['Time (minutes)'], ee_list[17]['Contact Surface Length (px)']/ee_first_valid_radius[17], c='lightgrey')
ax.scatter(ee_list[17]['Time (minutes)'], ee_list[17]['Contact Surface Length (px)']/ee_first_valid_radius[17], marker='.', c='k', zorder=10)
ax.set_ylim(0,4.0)
ax.set_xlim(-10,50)
ax.draw
fig.savefig(Path.joinpath(output_dir, r"ee_contnd_exp_inc_example.tif"), dpi=dpi_graph)
plt.close(fig)


# 12 or 48 for increase to low plateau followed by another increase

# An example of fast increase with no low plateau:
fig14, ax14 = plt.subplots(figsize=(width_1col,height_1col))
ax14.plot(ee_list[2]['Time (minutes)'], ee_list[2]['Contact Surface Length (px)']/ee_first_valid_radius[2], c='lightgrey')
ax14.scatter(ee_list[2]['Time (minutes)'], ee_list[2]['Contact Surface Length (px)']/ee_first_valid_radius[2], marker='.', c='k', zorder=10)
ax14.set_ylim(0,4.0)
ax14.draw
fig14.savefig(Path.joinpath(output_dir, r"ee_contnd_rapid_inc_example2.tif"), dpi=dpi_graph)
fig14.clf()





###############################################################################
## For meso-meso load a table where parameters of the logistic fits are 
## considered good (True) or bad (False). Then separate data with good growth 
## parameters from those without:

mm_list_dropna_fit_validity_df = pd.read_csv(Path.joinpath(fit_validity_dir, r"mm plateau table contnd.csv"))
mm_list_dropna_fit_validity_df = mm_list_dropna_fit_validity_df[['Source_File', 'low_plat', 'growth_phase', 'high_plat']]
# The 1 line below would only used for determining contact dynamics:
mm_Source_File_valid_low_plat = mm_list_dropna_fit_validity_df.query('low_plat==True')['Source_File']

mm_list_dropna_angle_fit_validity_df = pd.read_csv(Path.joinpath(fit_validity_dir, r"mm plateau table angles.csv"))
mm_list_dropna_angle_fit_validity_df = mm_list_dropna_angle_fit_validity_df[['Source_File', 'low_plat', 'growth_phase', 'high_plat']]
# The 1 line below would only used for determining contact dynamics:
mm_Source_File_angle_valid_low_plat = mm_list_dropna_angle_fit_validity_df.query('low_plat==True')['Source_File']


# Time correct data for mm:
# For dimensional contact diameter:
mm_df_dropna_time_corr_cpu, mm_df_dropna_cant_time_corr = time_correct_data(mm_list_dropna, mm_indiv_logistic_cont_df, mm_list_dropna_fit_validity_df)
time_grouped_mm_df_time_corr_cpu = mm_df_dropna_time_corr_cpu.groupby(['Time (minutes)'])
time_grouped_mm_df_cant_time_corr = mm_df_dropna_cant_time_corr.groupby(['Time (minutes)'])
mm_contact_mean = time_grouped_mm_df_time_corr_cpu['Contact Surface Length (px)'].describe().reset_index()
mm_contact_mean_cant_time_corr = time_grouped_mm_df_cant_time_corr['Contact Surface Length (px)'].describe().reset_index()

# For contact angle:
mm_df_dropna_angle_time_corr_cpu, mm_df_dropna_angle_cant_time_corr = time_correct_data(mm_contact_diam_non_dim_df_list_dropna, mm_indiv_logistic_angle_df, mm_list_dropna_angle_fit_validity_df)
time_grouped_mm_df_angle_time_corr_cpu = mm_df_dropna_angle_time_corr_cpu.groupby(['Time (minutes)'])
time_grouped_mm_df_angle_cant_time_corr = mm_df_dropna_angle_cant_time_corr.groupby(['Time (minutes)'])
mm_angle_mean = time_grouped_mm_df_angle_time_corr_cpu['Average Angle (deg)'].describe().reset_index()
mm_angle_mean_cant_time_corr = time_grouped_mm_df_angle_cant_time_corr['Average Angle (deg)'].describe().reset_index()

# For non-dimensional contact diameter:
mm_fit_df_dropna_time_corr_cpu, mm_fit_df_dropna_cant_time_corr = time_correct_data(mm_contact_diam_non_dim_df_list_dropna, mm_indiv_logistic_contnd_df, mm_list_dropna_fit_validity_df)
time_grouped_mm_fit_df_time_corr_cpu = mm_fit_df_dropna_time_corr_cpu.groupby(['Time (minutes)'])
time_grouped_mm_fit_df_cant_time_corr = mm_fit_df_dropna_cant_time_corr.groupby(['Time (minutes)'])
mm_fit_contact_mean = time_grouped_mm_fit_df_time_corr_cpu['Contact Surface Length (N.D.)'].describe().reset_index()
mm_fit_contact_mean_cant_time_corr = time_grouped_mm_fit_df_cant_time_corr['Contact Surface Length (N.D.)'].describe().reset_index()



# Fit general logistic functions to the time corrected data:
# Fit general logistic curve to mean angles:
#bounds_genlog_angle = ([-50, 0, 1, 0, 0], [100, 100, 3, 1, 1000])
sig_angle = 1./np.sqrt(mm_angle_mean['count'])**2
# nu is the last parameter in the general logistic equation -- affects where
# fastest rate of growth (ie. inflection point) is on x-axis:
mm_genlog_popt_angle, mm_genlog_pcov_angle = curve_fit(general_logistic, mm_angle_mean['Time (minutes)'], mm_angle_mean['mean'], sigma=sig_angle, p0 = (*p0_angle, nu), maxfev=2000)#, bounds=bounds_genlog_contnd)
with np.errstate(over='ignore'):
    mm_genlog_y_angle = general_logistic(x_fit, *mm_genlog_popt_angle)

mm_genlog_y_angle_diff = np.diff(np.ma.masked_array(data=mm_genlog_y_angle, mask=~np.isfinite(mm_genlog_y_angle)))


percent90_high = np.argmin(abs(np.cumsum(mm_genlog_y_angle_diff) - np.cumsum(mm_genlog_y_angle_diff).max()*0.95))
percent90_low = np.argmin(abs(np.cumsum(mm_genlog_y_angle_diff) - np.cumsum(mm_genlog_y_angle_diff).max()*0.05))
#growth_phase_end_mm_genlog_contnd = x_fit[percent90_high]
#growth_phase_end_mm_genlog_contnd_val = mm_genlog_y_contnd[percent90_high]
#growth_phase_start_mm_genlog_contnd = x_fit[percent90_low]
#growth_phase_start_mm_genlog_contnd_val = mm_genlog_y_contnd[percent90_low]
growth_phase_end_mm_angle = x_fit[percent90_high]
growth_phase_end_mm_angle_val = mm_genlog_y_angle[percent90_high]
growth_phase_start_mm_angle = x_fit[percent90_low]
growth_phase_start_mm_angle_val = mm_genlog_y_angle[percent90_low]



# Fit general logistic curve to mean non-dimensionalized diameters:
#bounds_genlog_contnd = ([-50, 0, 1, 0, 0], [100, 100, 3, 1, 1000])
sig_contnd = 1./np.sqrt(mm_fit_contact_mean['count'])**2
# nu is the last parameter in the general logistic equation -- affects where
# fastest rate of growth (ie. inflection point) is on x-axis:
mm_genlog_popt_contnd, mm_genlog_pcov_contnd = curve_fit(general_logistic, mm_fit_contact_mean['Time (minutes)'], mm_fit_contact_mean['mean'], sigma=sig_contnd, p0 = (*p0_contnd, nu), maxfev=2000)#, bounds=bounds_genlog_contnd)
with np.errstate(over='ignore'):
    mm_genlog_y_contnd = general_logistic(x_fit, *mm_genlog_popt_contnd)
    
mm_genlog_y_contnd_diff = np.diff(np.ma.masked_array(data=mm_genlog_y_contnd, mask=~np.isfinite(mm_genlog_y_contnd)))


percent90_high = np.argmin(abs(np.cumsum(mm_genlog_y_contnd_diff) - np.cumsum(mm_genlog_y_contnd_diff).max()*0.95))
percent90_low = np.argmin(abs(np.cumsum(mm_genlog_y_contnd_diff) - np.cumsum(mm_genlog_y_contnd_diff).max()*0.05))
#growth_phase_end_mm_genlog_contnd = x_fit[percent90_high]
#growth_phase_end_mm_genlog_contnd_val = mm_genlog_y_contnd[percent90_high]
#growth_phase_start_mm_genlog_contnd = x_fit[percent90_low]
#growth_phase_start_mm_genlog_contnd_val = mm_genlog_y_contnd[percent90_low]
growth_phase_end_mm_contnd = x_fit[percent90_high]
growth_phase_end_mm_contnd_val = mm_genlog_y_contnd[percent90_high]
growth_phase_start_mm_contnd = x_fit[percent90_low]
growth_phase_start_mm_contnd_val = mm_genlog_y_contnd[percent90_low]





### Find rate of change of fits at the inflection point of each curve:
growth_rates_mm = [np.diff(x) for x in indiv_fits_mm]
max_growth_rates_mm = [max(x) for x in growth_rates_mm]
max_growth_rates_dropna_mm = pd.DataFrame({'max_growth_rates':max_growth_rates_mm}).mask(~mm_list_dropna_fit_validity_df['growth_phase'])
max_growth_rates_dropna_mm.dropna(inplace=True)
### Show how these fastest rates of change are distributed
fig, kde = plt.subplots(figsize=(width_1col,height_1col))
kde = sns.distplot(max_growth_rates_dropna_mm['max_growth_rates'], bins=60, hist_kws={'edgecolor':'k'})
kde_x = kde.lines[0].get_xdata()
kde_y = kde.lines[0].get_ydata()
kde_maxid = signal.find_peaks(kde_y)[0]
print(kde_x[kde_maxid])
fig.savefig(Path.joinpath(output_dir, r"mm_cont_fastest growth.tif"), dpi=dpi_graph)
fig.clf()



# Also return the angle mean for the contact-based fit and vice-versa:
mm_angle_mean_for_contnd_fit = time_grouped_mm_fit_df_time_corr_cpu['Average Angle (deg)'].describe().reset_index()
# Fit general logistic curve to mean angles aligned according to contnd:
sig_angle = 1./np.sqrt(mm_angle_mean_for_contnd_fit['count'])**2
mm_genlog_popt_angle_contndfit, mm_genlog_pcov_angle_contndfit = curve_fit(general_logistic, mm_angle_mean_for_contnd_fit['Time (minutes)'], mm_angle_mean_for_contnd_fit['mean'], sigma=sig_angle, p0 = (*p0_angle, nu), maxfev=2000)#, bounds=bounds_genlog_contnd)
with np.errstate(over='ignore'):
    mm_genlog_y_angle_contndfit = general_logistic(x_fit, *mm_genlog_popt_angle_contndfit)



mm_contnd_mean_for_angle_fit = time_grouped_mm_df_angle_time_corr_cpu['Contact Surface Length (N.D.)'].describe().reset_index()
# Fit general logistic curve to mean non-dimensionalized contact diameters
# aligned according to angle data:
sig_contnd = 1./np.sqrt(mm_contnd_mean_for_angle_fit['count'])**2
# nu is the last parameter in the general logistic equation -- affects where
# fastest rate of growth (ie. inflection point) is on x-axis:
mm_genlog_popt_contnd_anglefit, mm_genlog_pcov_contnd_anglefit = curve_fit(general_logistic, mm_contnd_mean_for_angle_fit['Time (minutes)'], mm_contnd_mean_for_angle_fit['mean'], sigma=sig_contnd, p0 = (*p0_contnd, nu), maxfev=2000)#, bounds=bounds_genlog_contnd)
with np.errstate(over='ignore'):
    mm_genlog_y_contnd_anglefit = general_logistic(x_fit, *mm_genlog_popt_contnd_anglefit)




# Length of growth phase:
mm_angle_growth_dur = growth_phase_end_mm_angle - growth_phase_start_mm_angle
#mm_cont_growth_dur = growth_phase_end_mm_cont - growth_phase_start_mm_cont
mm_contnd_growth_dur = growth_phase_end_mm_contnd - growth_phase_start_mm_contnd


# Magnitude of change during growth phase:
mm_angle_growth_mag = growth_phase_end_mm_angle_val - growth_phase_start_mm_angle_val
#mm_cont_growth_mag = growth_phase_end_mm_cont_val - growth_phase_start_mm_cont_val
mm_contnd_growth_mag = growth_phase_end_mm_contnd_val - growth_phase_start_mm_contnd_val


# Average rate of change during growth phase:
mm_growth_rate_lin_angle = mm_angle_growth_mag / mm_angle_growth_dur
#mm_growth_rate_lin_cont = mm_cont_growth_mag / mm_cont_growth_dur
mm_growth_rate_lin_contnd = mm_contnd_growth_mag / mm_contnd_growth_dur




###############################################################################
## For endo-endo load a table where parameters of the logistic fits are 
## considered good (True) or bad (False). Then separate data with good growth 
## parameters from those without:

nn_list_dropna_fit_validity_df = pd.read_csv(Path.joinpath(fit_validity_dir, r"nn plateau table contnd.csv"))
nn_list_dropna_fit_validity_df = nn_list_dropna_fit_validity_df[['Source_File', 'low_plat', 'growth_phase', 'high_plat']]
# The 1 line below would only used for determining contact dynamics:
nn_Source_File_valid_low_plat = nn_list_dropna_fit_validity_df.query('low_plat==True')['Source_File']

nn_list_dropna_angle_fit_validity_df = pd.read_csv(Path.joinpath(fit_validity_dir, r"nn plateau table angles.csv"))
nn_list_dropna_angle_fit_validity_df = nn_list_dropna_angle_fit_validity_df[['Source_File', 'low_plat', 'growth_phase', 'high_plat']]
# The 1 line below would only used for determining contact dynamics:
nn_Source_File_angle_valid_low_plat = nn_list_dropna_angle_fit_validity_df.query('low_plat==True')['Source_File']


# Time correct data for nn:
# For dimensional contact diameter:
nn_df_dropna_time_corr_cpu, nn_df_dropna_cant_time_corr = time_correct_data(nn_list_dropna, nn_indiv_logistic_cont_df, nn_list_dropna_fit_validity_df)
time_grouped_nn_df_time_corr_cpu = nn_df_dropna_time_corr_cpu.groupby(['Time (minutes)'])
time_grouped_nn_df_cant_time_corr = nn_df_dropna_cant_time_corr.groupby(['Time (minutes)'])
nn_contact_mean = time_grouped_nn_df_time_corr_cpu['Contact Surface Length (px)'].describe().reset_index()
nn_contact_mean_cant_time_corr = time_grouped_nn_df_cant_time_corr['Contact Surface Length (px)'].describe().reset_index()

# For contact angle:
nn_df_dropna_angle_time_corr_cpu, nn_df_dropna_angle_cant_time_corr = time_correct_data(nn_contact_diam_non_dim_df_list_dropna, nn_indiv_logistic_angle_df, nn_list_dropna_angle_fit_validity_df)
time_grouped_nn_df_angle_time_corr_cpu = nn_df_dropna_angle_time_corr_cpu.groupby(['Time (minutes)'])
time_grouped_nn_df_angle_cant_time_corr = nn_df_dropna_angle_cant_time_corr.groupby(['Time (minutes)'])
nn_angle_mean = time_grouped_nn_df_angle_time_corr_cpu['Average Angle (deg)'].describe().reset_index()
nn_angle_mean_cant_time_corr = time_grouped_nn_df_angle_cant_time_corr['Average Angle (deg)'].describe().reset_index()

# For non-dimensional contact diameter:
nn_fit_df_dropna_time_corr_cpu, nn_fit_df_dropna_cant_time_corr = time_correct_data(nn_contact_diam_non_dim_df_list_dropna, nn_indiv_logistic_contnd_df, nn_list_dropna_fit_validity_df)
time_grouped_nn_fit_df_time_corr_cpu = nn_fit_df_dropna_time_corr_cpu.groupby(['Time (minutes)'])
time_grouped_nn_fit_df_cant_time_corr = nn_fit_df_dropna_cant_time_corr.groupby(['Time (minutes)'])
nn_fit_contact_mean = time_grouped_nn_fit_df_time_corr_cpu['Contact Surface Length (N.D.)'].describe().reset_index()
nn_fit_contact_mean_cant_time_corr = time_grouped_nn_fit_df_cant_time_corr['Contact Surface Length (N.D.)'].describe().reset_index()


# Fit general logistic functions to the time corrected data:
#bounds_genlog_contnd = ([-50, 0, 1, 0, 0], [100, 100, 3, 1, 1000])
sig_angle = 1./np.sqrt(nn_angle_mean['count'])**2
# nu is the last parameter in the general logistic equation -- affects where
# fastest rate of growth (ie. inflection point) is on x-axis:
nn_genlog_popt_angle, nn_genlog_pcov_angle = curve_fit(general_logistic, nn_angle_mean['Time (minutes)'], nn_angle_mean['mean'], sigma=sig_angle, p0 = (*p0_angle, nu), maxfev=2000)
with np.errstate(over='ignore'):
    nn_genlog_y_angle = general_logistic(x_fit, *nn_genlog_popt_angle)

nn_genlog_y_angle_diff = np.diff(np.ma.masked_array(data=nn_genlog_y_angle, mask=~np.isfinite(nn_genlog_y_angle)))


percent90_high = np.argmin(abs(np.cumsum(nn_genlog_y_angle_diff) - np.cumsum(nn_genlog_y_angle_diff).max()*0.95))
percent90_low = np.argmin(abs(np.cumsum(nn_genlog_y_angle_diff) - np.cumsum(nn_genlog_y_angle_diff).max()*0.05))
growth_phase_end_nn_angle = x_fit[percent90_high]
growth_phase_end_nn_angle_val = nn_genlog_y_angle[percent90_high]
growth_phase_start_nn_angle = x_fit[percent90_low]
growth_phase_start_nn_angle_val = nn_genlog_y_angle[percent90_low]
growth_phase_start_nn_angle_manual = -24



#bounds_genlog_contnd = ([-50, 0, 1, 0, 0], [100, 100, 3, 1, 1000])
sig_contnd = 1./np.sqrt(nn_fit_contact_mean['count'])**2
# nu is the last parameter in the general logistic equation -- affects where
# fastest rate of growth (ie. inflection point) is on x-axis:
nn_genlog_popt_contnd, nn_genlog_pcov_contnd = curve_fit(general_logistic, nn_fit_contact_mean['Time (minutes)'], nn_fit_contact_mean['mean'], sigma=sig_contnd, p0 = (*p0_contnd, nu), maxfev=2000)#, bounds=bounds_genlog_contnd)
with np.errstate(over='ignore'):
    nn_genlog_y_contnd = general_logistic(x_fit, *nn_genlog_popt_contnd)

nn_genlog_y_contnd_diff = np.diff(np.ma.masked_array(data=nn_genlog_y_contnd, mask=~np.isfinite(nn_genlog_y_contnd)))


percent90_high = np.argmin(abs(np.cumsum(nn_genlog_y_contnd_diff) - np.cumsum(nn_genlog_y_contnd_diff).max()*0.95))
percent90_low = np.argmin(abs(np.cumsum(nn_genlog_y_contnd_diff) - np.cumsum(nn_genlog_y_contnd_diff).max()*0.05))
growth_phase_end_nn_contnd = x_fit[percent90_high]
growth_phase_end_nn_contnd_val = nn_genlog_y_contnd[percent90_high]
growth_phase_start_nn_contnd = x_fit[percent90_low]
growth_phase_start_nn_contnd_val = nn_genlog_y_contnd[percent90_low]
#growth_phase_start_nn_contnd_manual = -24





### Find rate of change of fits at the inflection point of each curve:
growth_rates_nn = [np.diff(x) for x in indiv_fits_nn]
max_growth_rates_nn = [max(x) for x in growth_rates_nn]
max_growth_rates_dropna_nn = pd.DataFrame({'max_growth_rates':max_growth_rates_nn}).mask(~nn_list_dropna_fit_validity_df['growth_phase'])
max_growth_rates_dropna_nn.dropna(inplace=True)
### Show how these fastest rates of change are distributed
fig, kde = plt.subplots(figsize=(width_1col,height_1col))
kde = sns.distplot(max_growth_rates_dropna_nn['max_growth_rates'], bins=60, hist_kws={'edgecolor':'k'})
kde_x = kde.lines[0].get_xdata()
kde_y = kde.lines[0].get_ydata()
kde_maxid = signal.find_peaks(kde_y)[0]
print(kde_x[kde_maxid])
fig.savefig(Path.joinpath(output_dir, r"nn_cont_fastest growth.tif"), dpi=dpi_graph)
fig.clf()



# Also return the angle mean for the contact-based fit and vice-versa:
nn_angle_mean_for_contnd_fit = time_grouped_nn_fit_df_time_corr_cpu['Average Angle (deg)'].describe().reset_index()
# Fit general logistic curve to mean angles aligned according to contnd:
sig_angle = 1./np.sqrt(nn_angle_mean_for_contnd_fit['count'])**2
nn_genlog_popt_angle_contndfit, nn_genlog_pcov_angle_contndfit = curve_fit(general_logistic, nn_angle_mean_for_contnd_fit['Time (minutes)'], nn_angle_mean_for_contnd_fit['mean'], sigma=sig_angle, p0 = (*p0_angle, nu), maxfev=2000)#, bounds=bounds_genlog_contnd)
with np.errstate(over='ignore'):
    nn_genlog_y_angle_contndfit = general_logistic(x_fit, *nn_genlog_popt_angle_contndfit)



nn_contnd_mean_for_angle_fit = time_grouped_nn_df_angle_time_corr_cpu['Contact Surface Length (N.D.)'].describe().reset_index()
# Fit general logistic curve to mean non-dimensionalized contact diameters
# aligned according to angle data:
sig_contnd = 1./np.sqrt(nn_contnd_mean_for_angle_fit['count'])**2
# nu is the last parameter in the general logistic equation -- affects where
# fastest rate of growth (ie. inflection point) is on x-axis:
nn_genlog_popt_contnd_anglefit, nn_genlog_pcov_contnd_anglefit = curve_fit(general_logistic, nn_contnd_mean_for_angle_fit['Time (minutes)'], nn_contnd_mean_for_angle_fit['mean'], sigma=sig_contnd, p0 = (*p0_contnd, nu), maxfev=2000)#, bounds=bounds_genlog_contnd)
with np.errstate(over='ignore'):
    nn_genlog_y_contnd_anglefit = general_logistic(x_fit, *nn_genlog_popt_contnd_anglefit)





# Length of growth phase:
nn_angle_growth_dur = growth_phase_end_nn_angle - growth_phase_start_nn_angle
#nn_cont_growth_dur = growth_phase_end_nn_cont - growth_phase_start_nn_cont
nn_contnd_growth_dur = growth_phase_end_nn_contnd - growth_phase_start_nn_contnd


# Magnitude of change during growth phase:
nn_angle_growth_mag = growth_phase_end_nn_angle_val - growth_phase_start_nn_angle_val
#nn_cont_growth_mag = growth_phase_end_nn_cont_val - growth_phase_start_nn_cont_val
nn_contnd_growth_mag = growth_phase_end_nn_contnd_val - growth_phase_start_nn_contnd_val


# Average rate of change during growth phase:
nn_growth_rate_lin_angle = nn_angle_growth_mag / nn_angle_growth_dur
#nn_growth_rate_lin_cont = nn_cont_growth_mag / nn_cont_growth_dur
nn_growth_rate_lin_contnd = nn_contnd_growth_mag / nn_contnd_growth_dur



###############################################################################
# Repeat for ecto-meso:
em_list_dropna_fit_validity_df = pd.read_csv(Path.joinpath(fit_validity_dir, r"em plateau table contnd.csv"))
em_list_dropna_fit_validity_df = em_list_dropna_fit_validity_df[['Source_File', 'low_plat', 'growth_phase', 'high_plat']]
# The 1 line below would only used for determining contact dynamics:
em_Source_File_valid_low_plat = em_list_dropna_fit_validity_df.query('low_plat==True')['Source_File']

em_list_dropna_angle_fit_validity_df = pd.read_csv(Path.joinpath(fit_validity_dir, r"em plateau table angles.csv"))
em_list_dropna_angle_fit_validity_df = em_list_dropna_angle_fit_validity_df[['Source_File', 'low_plat', 'growth_phase', 'high_plat']]
# The 1 line below would only used for determining contact dynamics:
em_Source_File_angle_valid_low_plat = em_list_dropna_angle_fit_validity_df.query('low_plat==True')['Source_File']


# Time correct data for em:
# For dimensional contact diameter:
em_df_dropna_time_corr_cpu, em_df_dropna_cant_time_corr = time_correct_data(em_list_dropna, em_indiv_logistic_cont_df, em_list_dropna_fit_validity_df)
time_grouped_em_df_time_corr_cpu = em_df_dropna_time_corr_cpu.groupby(['Time (minutes)'])
time_grouped_em_df_cant_time_corr = em_df_dropna_cant_time_corr.groupby(['Time (minutes)'])
em_contact_mean = time_grouped_em_df_time_corr_cpu['Contact Surface Length (px)'].describe().reset_index()
em_contact_mean_cant_time_corr = time_grouped_em_df_cant_time_corr['Contact Surface Length (px)'].describe().reset_index()

# For contact angle:
em_df_dropna_angle_time_corr_cpu, em_df_dropna_angle_cant_time_corr = time_correct_data(em_contact_diam_non_dim_df_list_dropna, em_indiv_logistic_angle_df, em_list_dropna_angle_fit_validity_df)
time_grouped_em_df_angle_time_corr_cpu = em_df_dropna_angle_time_corr_cpu.groupby(['Time (minutes)'])
time_grouped_em_df_angle_cant_time_corr = em_df_dropna_angle_cant_time_corr.groupby(['Time (minutes)'])
em_angle_mean = time_grouped_em_df_angle_time_corr_cpu['Average Angle (deg)'].describe().reset_index()
em_angle_mean_cant_time_corr = time_grouped_em_df_angle_cant_time_corr['Average Angle (deg)'].describe().reset_index()

# For non-dimensional contact diameter:
em_fit_df_dropna_time_corr_cpu, em_fit_df_dropna_cant_time_corr = time_correct_data(em_contact_diam_non_dim_df_list_dropna, em_indiv_logistic_contnd_df, em_list_dropna_fit_validity_df)
time_grouped_em_fit_df_time_corr_cpu = em_fit_df_dropna_time_corr_cpu.groupby(['Time (minutes)'])
time_grouped_em_fit_df_cant_time_corr = em_fit_df_dropna_cant_time_corr.groupby(['Time (minutes)'])
em_fit_contact_mean = time_grouped_em_fit_df_time_corr_cpu['Contact Surface Length (N.D.)'].describe().reset_index()
em_fit_contact_mean_cant_time_corr = time_grouped_em_fit_df_cant_time_corr['Contact Surface Length (N.D.)'].describe().reset_index()


# Fit general logistic functions to the time corrected data:
#bounds_genlog_contnd = ([-50, 0, 1, 0, 0], [100, 100, 3, 1, 1000])
sig_angle = 1./np.sqrt(em_angle_mean['count'])**2
# nu is the last parameter in the general logistic equation -- affects where
# fastest rate of growth (ie. inflection point) is on x-axis:
em_genlog_popt_angle, em_genlog_pcov_angle = curve_fit(general_logistic, em_angle_mean['Time (minutes)'], em_angle_mean['mean'], sigma=sig_angle, p0 = (*p0_angle_em_genlog, nu), maxfev=2000)
with np.errstate(over='ignore'):
    em_genlog_y_angle = general_logistic(x_fit, *em_genlog_popt_angle)



#bounds_genlog_contnd = ([-50, 0, 1, 0, 0], [100, 100, 3, 1, 1000])
sig_contnd = 1./np.sqrt(em_fit_contact_mean['count'])**2
# nu is the last parameter in the general logistic equation -- affects where
# fastest rate of growth (ie. inflection point) is on x-axis:
em_genlog_popt_contnd, em_genlog_pcov_contnd = curve_fit(general_logistic, em_fit_contact_mean['Time (minutes)'], em_fit_contact_mean['mean'], sigma=sig_contnd, p0 = (*p0_contnd_em, nu), maxfev=2000)#, bounds=bounds_genlog_contnd)
with np.errstate(over='ignore'):
    em_genlog_y_contnd = general_logistic(x_fit, *em_genlog_popt_contnd)



#bounds_genlog_contnd = ([-50, 0, 1, 0, 0], [100, 100, 3, 1, 1000])
sig_cont = 1./np.sqrt(em_contact_mean['count'])**2
# nu is the last parameter in the general logistic equation -- affects where
# fastest rate of growth (ie. inflection point) is on x-axis:
em_genlog_popt_cont, em_genlog_pcov_cont = curve_fit(general_logistic, em_contact_mean['Time (minutes)'], em_contact_mean['mean'], sigma=sig_cont, p0 = (*p0_cont_em, nu), maxfev=2000)#, bounds=bounds_genlog_contnd)
with np.errstate(over='ignore'):
    em_genlog_y_cont = general_logistic(x_fit, *em_genlog_popt_cont)






# # Also return the angle mean for the contact-based fit and vice-versa:
# em_angle_mean_for_contnd_fit = time_grouped_em_fit_df_time_corr_cpu['Average Angle (deg)'].describe().reset_index()
# # Fit general logistic curve to mean angles aligned according to contnd:
# sig_angle = 1./np.sqrt(em_angle_mean_for_contnd_fit['count'])**2
# em_genlog_popt_angle_contndfit, em_genlog_pcov_angle_contndfit = curve_fit(general_logistic, em_angle_mean_for_contnd_fit['Time (minutes)'], em_angle_mean_for_contnd_fit['mean'], sigma=sig_angle, p0 = (*p0_angle_em_genlog, nu), maxfev=2000)#, bounds=bounds_genlog_contnd)
# with np.errstate(over='ignore'):
#     em_genlog_y_angle_contndfit = general_logistic(x_fit, *em_genlog_popt_angle_contndfit)



# em_contnd_mean_for_angle_fit = time_grouped_em_df_angle_time_corr_cpu['Contact Surface Length (N.D.)'].describe().reset_index()
# # Fit general logistic curve to mean non-dimensionalized contact diameters
# # aligned according to angle data:
# sig_contnd = 1./np.sqrt(em_contnd_mean_for_angle_fit['count'])**2
# # nu is the last parameter in the general logistic equation -- affects where
# # fastest rate of growth (ie. inflection point) is on x-axis:
# em_genlog_popt_contnd_anglefit, em_genlog_pcov_contnd_anglefit = curve_fit(general_logistic, em_contnd_mean_for_angle_fit['Time (minutes)'], em_contnd_mean_for_angle_fit['mean'], sigma=sig_contnd, p0 = (*p0_contnd_em, nu), maxfev=2000)#, bounds=bounds_genlog_contnd)
# with np.errstate(over='ignore'):
#     em_genlog_y_contnd_anglefit = general_logistic(x_fit, *em_genlog_popt_contnd_anglefit)




# Length of growth phase:
#em_angle_growth_dur = growth_phase_end_em_angle - growth_phase_start_em_angle
#em_cont_growth_dur = growth_phase_end_em_cont - growth_phase_start_em_cont
#em_contnd_growth_dur = growth_phase_end_em_contnd - growth_phase_start_em_contnd
em_angle_growth_dur = np.nan
em_contnd_growth_dur = np.nan

# Magnitude of change during growth phase:
#em_angle_growth_mag = growth_phase_end_em_angle_val - growth_phase_start_em_angle_val
#em_cont_growth_mag = growth_phase_end_em_cont_val - growth_phase_start_em_cont_val
#em_contnd_growth_mag = growth_phase_end_em_contnd_val - growth_phase_start_em_contnd_val
em_angle_growth_mag = np.nan
em_contnd_growth_mag = np.nan

# Average rate of change during growth phase:
em_growth_rate_lin_angle = em_angle_growth_mag / em_angle_growth_dur
#em_growth_rate_lin_cont = em_cont_growth_mag / em_cont_growth_dur
em_growth_rate_lin_contnd = em_contnd_growth_mag / em_contnd_growth_dur



###############################################################################
# Repeat for ecto-ecto in dissociation buffer:
eedb_list_dropna_fit_validity_df = pd.read_csv(Path.joinpath(fit_validity_dir, r"eedb plateau table contnd.csv"))
eedb_list_dropna_fit_validity_df = eedb_list_dropna_fit_validity_df[['Source_File', 'low_plat', 'growth_phase', 'high_plat']]
# The 1 line below would only used for determining contact dynamics:
eedb_Source_File_valid_low_plat = eedb_list_dropna_fit_validity_df.query('low_plat==True')['Source_File']

eedb_list_dropna_angle_fit_validity_df = pd.read_csv(Path.joinpath(fit_validity_dir, r"eedb plateau table angles.csv"))
eedb_list_dropna_angle_fit_validity_df = eedb_list_dropna_angle_fit_validity_df[['Source_File', 'low_plat', 'growth_phase', 'high_plat']]
# The 1 line below would only used for determining contact dynamics:
eedb_Source_File_angle_valid_low_plat = eedb_list_dropna_angle_fit_validity_df.query('low_plat==True')['Source_File']


# Time correct data for eedb:
# For dimensional contact diameter:
eedb_df_dropna_time_corr_cpu, eedb_df_dropna_cant_time_corr = time_correct_data(eedb_list_dropna, eedb_indiv_logistic_cont_df, eedb_list_dropna_fit_validity_df)
time_grouped_eedb_df_time_corr_cpu = eedb_df_dropna_time_corr_cpu.groupby(['Time (minutes)'])
time_grouped_eedb_df_cant_time_corr = eedb_df_dropna_cant_time_corr.groupby(['Time (minutes)'])
eedb_contact_mean = time_grouped_eedb_df_time_corr_cpu['Contact Surface Length (px)'].describe().reset_index()
eedb_contact_mean_cant_time_corr = time_grouped_eedb_df_cant_time_corr['Contact Surface Length (px)'].describe().reset_index()

# For contact angle:
eedb_df_dropna_angle_time_corr_cpu, eedb_df_dropna_angle_cant_time_corr = time_correct_data(eedb_contact_diam_non_dim_df_list_dropna, eedb_indiv_logistic_angle_df, eedb_list_dropna_angle_fit_validity_df)
time_grouped_eedb_df_angle_time_corr_cpu = eedb_df_dropna_angle_time_corr_cpu.groupby(['Time (minutes)'])
time_grouped_eedb_df_angle_cant_time_corr = eedb_df_dropna_angle_cant_time_corr.groupby(['Time (minutes)'])
eedb_angle_mean = time_grouped_eedb_df_angle_time_corr_cpu['Average Angle (deg)'].describe().reset_index()
eedb_angle_mean_cant_time_corr = time_grouped_eedb_df_angle_cant_time_corr['Average Angle (deg)'].describe().reset_index()

# For non-dimensional contact diameter:
eedb_fit_df_dropna_time_corr_cpu, eedb_fit_df_dropna_cant_time_corr = time_correct_data(eedb_contact_diam_non_dim_df_list_dropna, eedb_indiv_logistic_contnd_df, eedb_list_dropna_fit_validity_df)
time_grouped_eedb_fit_df_time_corr_cpu = eedb_fit_df_dropna_time_corr_cpu.groupby(['Time (minutes)'])
time_grouped_eedb_fit_df_cant_time_corr = eedb_fit_df_dropna_cant_time_corr.groupby(['Time (minutes)'])
eedb_fit_contact_mean = time_grouped_eedb_fit_df_time_corr_cpu['Contact Surface Length (N.D.)'].describe().reset_index()
eedb_fit_contact_mean_cant_time_corr = time_grouped_eedb_fit_df_cant_time_corr['Contact Surface Length (N.D.)'].describe().reset_index()


# Fit general logistic functions to the time corrected data:
#bounds_genlog_contnd = ([-50, 0, 1, 0, 0], [100, 100, 3, 1, 1000])
sig_angle = 1./np.sqrt(eedb_angle_mean['count'])**2
# nu is the last parameter in the general logistic equation -- affects where
# fastest rate of growth (ie. inflection point) is on x-axis:
eedb_genlog_popt_angle, eedb_genlog_pcov_angle = curve_fit(general_logistic, eedb_angle_mean['Time (minutes)'], eedb_angle_mean['mean'], sigma=sig_angle, p0 = (*p0_angle_eedb, nu), maxfev=2000)
with np.errstate(over='ignore'):
    eedb_genlog_y_angle = general_logistic(x_fit, *eedb_genlog_popt_angle)


#bounds_genlog_contnd = ([-50, 0, 1, 0, 0], [100, 100, 3, 1, 1000])
sig_contnd = 1./np.sqrt(eedb_fit_contact_mean['count'])**2
# nu is the last parameter in the general logistic equation -- affects where
# fastest rate of growth (ie. inflection point) is on x-axis:
eedb_genlog_popt_contnd, eedb_genlog_pcov_contnd = curve_fit(general_logistic, eedb_fit_contact_mean['Time (minutes)'], eedb_fit_contact_mean['mean'], sigma=sig_contnd, p0 = (*p0_contnd, nu), maxfev=2000)#, bounds=bounds_genlog_contnd)
with np.errstate(over='ignore'):
    eedb_genlog_y_contnd = general_logistic(x_fit, *eedb_genlog_popt_contnd)


#bounds_genlog_contnd = ([-50, 0, 1, 0, 0], [100, 100, 3, 1, 1000])
sig_cont = 1./np.sqrt(eedb_contact_mean['count'])**2
# nu is the last parameter in the general logistic equation -- affects where
# fastest rate of growth (ie. inflection point) is on x-axis:
eedb_genlog_popt_cont, eedb_genlog_pcov_cont = curve_fit(general_logistic, eedb_contact_mean['Time (minutes)'], eedb_contact_mean['mean'], sigma=sig_cont, p0 = (*p0_cont, nu), maxfev=2000)#, bounds=bounds_genlog_contnd)
with np.errstate(over='ignore'):
    eedb_genlog_y_cont = general_logistic(x_fit, *eedb_genlog_popt_cont)





# Also return the angle mean for the contact-based fit and vice-versa:
eedb_angle_mean_for_contnd_fit = time_grouped_eedb_fit_df_time_corr_cpu['Average Angle (deg)'].describe().reset_index()
# Fit general logistic curve to mean angles aligned according to contnd:
sig_angle = 1./np.sqrt(eedb_angle_mean_for_contnd_fit['count'])**2
eedb_genlog_popt_angle_contndfit, eedb_genlog_pcov_angle_contndfit = curve_fit(general_logistic, eedb_angle_mean_for_contnd_fit['Time (minutes)'], eedb_angle_mean_for_contnd_fit['mean'], sigma=sig_angle, p0 = (*p0_angle, nu), maxfev=2000)#, bounds=bounds_genlog_contnd)
with np.errstate(over='ignore'):
    eedb_genlog_y_angle_contndfit = general_logistic(x_fit, *eedb_genlog_popt_angle_contndfit)


eedb_contnd_mean_for_angle_fit = time_grouped_eedb_df_angle_time_corr_cpu['Contact Surface Length (N.D.)'].describe().reset_index()
# Fit general logistic curve to mean non-dimensionalized contact diameters
# aligned according to angle data:
sig_contnd = 1./np.sqrt(eedb_contnd_mean_for_angle_fit['count'])**2
# nu is the last parameter in the general logistic equation -- affects where
# fastest rate of growth (ie. inflection point) is on x-axis:
eedb_genlog_popt_contnd_anglefit, eedb_genlog_pcov_contnd_anglefit = curve_fit(general_logistic, eedb_contnd_mean_for_angle_fit['Time (minutes)'], eedb_contnd_mean_for_angle_fit['mean'], sigma=sig_contnd, p0 = (*p0_contnd_angleBased_eedb, nu), maxfev=2000)#, bounds=bounds_genlog_contnd)
with np.errstate(over='ignore'):
    eedb_genlog_y_contnd_anglefit = general_logistic(x_fit, *eedb_genlog_popt_contnd_anglefit)




# Length of growth phase:
#eedb_angle_growth_dur = growth_phase_end_eedb_angle - growth_phase_start_eedb_angle
#eedb_cont_growth_dur = growth_phase_end_eedb_cont - growth_phase_start_eedb_cont
#eedb_contnd_growth_dur = growth_phase_end_eedb_contnd - growth_phase_start_eedb_contnd
eedb_angle_growth_dur = np.nan
eedb_contnd_growth_dur = np.nan

# Magnitude of change during growth phase:
#eedb_angle_growth_mag = growth_phase_end_eedb_angle_val - growth_phase_start_eedb_angle_val
#eedb_cont_growth_mag = growth_phase_end_eedb_cont_val - growth_phase_start_eedb_cont_val
#eedb_contnd_growth_mag = growth_phase_end_eedb_contnd_val - growth_phase_start_eedb_contnd_val
eedb_angle_growth_mag = np.nan
eedb_contnd_growth_mag = np.nan

# Average rate of change during growth phase:
eedb_growth_rate_lin_angle = eedb_angle_growth_mag / eedb_angle_growth_dur
#eedb_growth_rate_lin_cont = eedb_cont_growth_mag / eedb_cont_growth_dur
eedb_growth_rate_lin_contnd = eedb_contnd_growth_mag / eedb_contnd_growth_dur


###############################################################################
# Repeat for ectoCcadMO-ectoCcadMO data:

eeCcadMO_list_dropna_fit_validity_df = pd.read_csv(Path.joinpath(fit_validity_dir, r"eeCcadMO plateau table contnd.csv"))
eeCcadMO_list_dropna_fit_validity_df = eeCcadMO_list_dropna_fit_validity_df[['Source_File', 'low_plat', 'growth_phase', 'high_plat']]
# The 1 line below would only used for determining contact dynamics:
eeCcadMO_Source_File_valid_low_plat = eeCcadMO_list_dropna_fit_validity_df.query('low_plat==True')['Source_File']

eeCcadMO_list_dropna_angle_fit_validity_df = pd.read_csv(Path.joinpath(fit_validity_dir, r"eeCcadMO plateau table angles.csv"))
eeCcadMO_list_dropna_angle_fit_validity_df = eeCcadMO_list_dropna_angle_fit_validity_df[['Source_File', 'low_plat', 'growth_phase', 'high_plat']]
# The 1 line below would only used for determining contact dynamics:
eeCcadMO_Source_File_angle_valid_low_plat = eeCcadMO_list_dropna_angle_fit_validity_df.query('low_plat==True')['Source_File']


# Time correct data for eeCcadMO-eeCcadMO:
# For dimensional contact diameter:
eeCcadMO_df_dropna_time_corr_cpu, eeCcadMO_df_dropna_cant_time_corr = time_correct_data(eeCcadMO_list_dropna, eeCcadMO_indiv_logistic_cont_df, eeCcadMO_list_dropna_fit_validity_df)
time_grouped_eeCcadMO_df_time_corr_cpu = eeCcadMO_df_dropna_time_corr_cpu.groupby(['Time (minutes)'])
time_grouped_eeCcadMO_df_cant_time_corr = eeCcadMO_df_dropna_cant_time_corr.groupby(['Time (minutes)'])
eeCcadMO_contact_mean = time_grouped_eeCcadMO_df_time_corr_cpu['Contact Surface Length (px)'].describe().reset_index()
eeCcadMO_contact_mean_cant_time_corr = time_grouped_eeCcadMO_df_cant_time_corr['Contact Surface Length (px)'].describe().reset_index()

# For contact angle:
eeCcadMO_df_dropna_angle_time_corr_cpu, eeCcadMO_df_dropna_angle_cant_time_corr = time_correct_data(eeCcadMO_contact_diam_non_dim_df_list_dropna, eeCcadMO_indiv_logistic_angle_df, eeCcadMO_list_dropna_angle_fit_validity_df)
time_grouped_eeCcadMO_df_angle_time_corr_cpu = eeCcadMO_df_dropna_angle_time_corr_cpu.groupby(['Time (minutes)'])
time_grouped_eeCcadMO_df_angle_cant_time_corr = eeCcadMO_df_dropna_angle_cant_time_corr.groupby(['Time (minutes)'])
eeCcadMO_angle_mean = time_grouped_eeCcadMO_df_angle_time_corr_cpu['Average Angle (deg)'].describe().reset_index()
eeCcadMO_angle_mean_cant_time_corr = time_grouped_eeCcadMO_df_angle_cant_time_corr['Average Angle (deg)'].describe().reset_index()

# For non-dimensional contact diameter:
eeCcadMO_fit_df_dropna_time_corr_cpu, eeCcadMO_fit_df_dropna_cant_time_corr = time_correct_data(eeCcadMO_contact_diam_non_dim_df_list_dropna, eeCcadMO_indiv_logistic_contnd_df, eeCcadMO_list_dropna_fit_validity_df)
time_grouped_eeCcadMO_fit_df_time_corr_cpu = eeCcadMO_fit_df_dropna_time_corr_cpu.groupby(['Time (minutes)'])
time_grouped_eeCcadMO_fit_df_cant_time_corr = eeCcadMO_fit_df_dropna_cant_time_corr.groupby(['Time (minutes)'])
eeCcadMO_fit_contact_mean = time_grouped_eeCcadMO_fit_df_time_corr_cpu['Contact Surface Length (N.D.)'].describe().reset_index()
eeCcadMO_fit_contact_mean_cant_time_corr = time_grouped_eeCcadMO_fit_df_cant_time_corr['Contact Surface Length (N.D.)'].describe().reset_index()



# Fit general logistic curve to mean angles:
sig_angle = 1./np.sqrt(eeCcadMO_angle_mean['count'])**2

eeCcadMO_genlog_popt_angle, eeCcadMO_genlog_pcov_angle = curve_fit(general_logistic, eeCcadMO_angle_mean['Time (minutes)'], eeCcadMO_angle_mean['mean'], sigma=sig_angle, p0 = (*p0_angle, nu), maxfev=2000)#, bounds=bounds_genlog_contnd)
with np.errstate(over='ignore'):
    eeCcadMO_genlog_y_angle = general_logistic(x_fit, *eeCcadMO_genlog_popt_angle)

eeCcadMO_genlog_y_angle_diff = np.diff(np.ma.masked_array(data=eeCcadMO_genlog_y_angle, mask=~np.isfinite(eeCcadMO_genlog_y_angle)))


percent90_high = np.argmin(abs(np.cumsum(eeCcadMO_genlog_y_angle_diff) - np.cumsum(eeCcadMO_genlog_y_angle_diff).max()*0.95))
percent90_low = np.argmin(abs(np.cumsum(eeCcadMO_genlog_y_angle_diff) - np.cumsum(eeCcadMO_genlog_y_angle_diff).max()*0.05))
percent99_high = np.argmin(abs(np.cumsum(eeCcadMO_genlog_y_angle_diff) - np.cumsum(eeCcadMO_genlog_y_angle_diff).max()*0.99))
percent99_low = np.argmin(abs(np.cumsum(eeCcadMO_genlog_y_angle_diff) - np.cumsum(eeCcadMO_genlog_y_angle_diff).max()*0.01))
#growth_phase_end_ee_genlog_angle = x_fit[percent90_high]
#growth_phase_end_ee_genlog_angle_val = ee_genlog_y_angle[percent90_high]
#growth_phase_start_ee_genlog_angle = x_fit[percent90_low]
#growth_phase_start_ee_genlog_angle_val = ee_genlog_y_angle[percent90_low]
growth_phase_end_eeCcadMO_angle = x_fit[percent90_high]
growth_phase_end_eeCcadMO_angle_val = eeCcadMO_genlog_y_angle[percent90_high]
growth_phase_start_eeCcadMO_angle = x_fit[percent90_low]
growth_phase_start_eeCcadMO_angle_val = eeCcadMO_genlog_y_angle[percent90_low]

growth_phase_end_eeCcadMO_angle_strict = x_fit[percent99_high]
growth_phase_end_eeCcadMO_angle_strict_val = eeCcadMO_genlog_y_angle[percent99_high]
growth_phase_start_eeCcadMO_angle_strict = x_fit[percent99_low]
growth_phase_start_eeCcadMO_angle_strict_val = eeCcadMO_genlog_y_angle[percent99_high]




# Fit general logistic curve to mean non-dimensionalized diameters:
sig_contnd = 1./np.sqrt(eeCcadMO_fit_contact_mean['count'])**2
# nu is the last parameter in the general logistic equation -- affects where
# fastest rate of growth (ie. inflection point) is on x-axis:
eeCcadMO_genlog_popt_contnd, eeCcadMO_genlog_pcov_contnd = curve_fit(general_logistic, eeCcadMO_fit_contact_mean['Time (minutes)'], eeCcadMO_fit_contact_mean['mean'], sigma=sig_contnd, p0 = (*p0_contnd, nu), maxfev=2000)#, bounds=bounds_genlog_contnd)
with np.errstate(over='ignore'):
    eeCcadMO_genlog_y_contnd = general_logistic(x_fit, *eeCcadMO_genlog_popt_contnd)

eeCcadMO_genlog_y_contnd_diff = np.diff(np.ma.masked_array(data=eeCcadMO_genlog_y_contnd, mask=~np.isfinite(eeCcadMO_genlog_y_contnd)))


#percent95 = np.argmin(abs(np.cumsum(eeCcadMO_y_contnd_diff) - np.cumsum(eeCcadMO_y_contnd_diff).max()*0.95))
#percent50 = np.argmin(abs(np.cumsum(eeCcadMO_y_contnd_diff) - np.cumsum(eeCcadMO_y_contnd_diff).max()*0.50))
percent90_high = np.argmin(abs(np.cumsum(eeCcadMO_genlog_y_contnd_diff) - np.cumsum(eeCcadMO_genlog_y_contnd_diff).max()*0.95))
percent90_low = np.argmin(abs(np.cumsum(eeCcadMO_genlog_y_contnd_diff) - np.cumsum(eeCcadMO_genlog_y_contnd_diff).max()*0.05))
percent99_high = np.argmin(abs(np.cumsum(eeCcadMO_genlog_y_contnd_diff) - np.cumsum(eeCcadMO_genlog_y_contnd_diff).max()*0.995))
percent99_low = np.argmin(abs(np.cumsum(eeCcadMO_genlog_y_contnd_diff) - np.cumsum(eeCcadMO_genlog_y_contnd_diff).max()*0.005))
#growth_phase_end_eeCcadMO_genlog_contnd = x_fit[percent90_high]
#growth_phase_end_eeCcadMO_genlog_contnd_val = eeCcadMO_genlog_y_contnd[percent90_high]
#growth_phase_start_eeCcadMO_genlog_contnd = x_fit[percent90_low]
#growth_phase_start_eeCcadMO_genlog_contnd_val = eeCcadMO_genlog_y_contnd[percent90_low]
growth_phase_end_eeCcadMO_contnd = x_fit[percent90_high]
growth_phase_end_eeCcadMO_contnd_val = eeCcadMO_genlog_y_contnd[percent90_high]
growth_phase_start_eeCcadMO_contnd = x_fit[percent90_low]
growth_phase_start_eeCcadMO_contnd_val = eeCcadMO_genlog_y_contnd[percent90_low]

growth_phase_end_eeCcadMO_contnd_strict = x_fit[percent99_high]
growth_phase_end_eeCcadMO_contnd_strict_val = eeCcadMO_genlog_y_contnd[percent99_high]
growth_phase_start_eeCcadMO_contnd_strict = x_fit[percent99_low]
growth_phase_start_eeCcadMO_contnd_strict_val = eeCcadMO_genlog_y_contnd[percent99_high]




#bounds_genlog_contnd = ([-50, 0, 1, 0, 0], [100, 100, 3, 1, 1000])
sig_cont = 1./np.sqrt(eeCcadMO_contact_mean['count'])**2
# nu is the last parameter in the general logistic equation -- affects where
# fastest rate of growth (ie. inflection point) is on x-axis:
eeCcadMO_genlog_popt_cont, eeCcadMO_genlog_pcov_cont = curve_fit(general_logistic, eeCcadMO_contact_mean['Time (minutes)'], eeCcadMO_contact_mean['mean'], sigma=sig_cont, p0 = (*p0_cont, nu), maxfev=2000)#, bounds=bounds_genlog_contnd)
with np.errstate(over='ignore'):
    eeCcadMO_genlog_y_cont = general_logistic(x_fit, *eeCcadMO_genlog_popt_cont)




# Also return the angle mean for the contact-based fit and vice-versa:
eeCcadMO_angle_mean_for_contnd_fit = time_grouped_eeCcadMO_fit_df_time_corr_cpu['Average Angle (deg)'].describe().reset_index()
# Fit general logistic curve to mean angles aligned according to contnd:
sig_angle = 1./np.sqrt(eeCcadMO_angle_mean_for_contnd_fit['count'])**2
eeCcadMO_genlog_popt_angle_contndfit, eeCcadMO_genlog_pcov_angle_contndfit = curve_fit(general_logistic, eeCcadMO_angle_mean_for_contnd_fit['Time (minutes)'], eeCcadMO_angle_mean_for_contnd_fit['mean'], sigma=sig_angle, p0 = (*p0_angle, nu), maxfev=2000)#, bounds=bounds_genlog_contnd)
with np.errstate(over='ignore'):
    eeCcadMO_genlog_y_angle_contndfit = general_logistic(x_fit, *eeCcadMO_genlog_popt_angle_contndfit)



eeCcadMO_contnd_mean_for_angle_fit = time_grouped_eeCcadMO_df_angle_time_corr_cpu['Contact Surface Length (N.D.)'].describe().reset_index()
# Fit general logistic curve to mean non-dimensionalized contact diameters
# aligned according to angle data:
sig_contnd = 1./np.sqrt(eeCcadMO_contnd_mean_for_angle_fit['count'])**2
# nu is the last parameter in the general logistic equation -- affects where
# fastest rate of growth (ie. inflection point) is on x-axis:
eeCcadMO_genlog_popt_contnd_anglefit, eeCcadMO_genlog_pcov_contnd_anglefit = curve_fit(general_logistic, eeCcadMO_contnd_mean_for_angle_fit['Time (minutes)'], eeCcadMO_contnd_mean_for_angle_fit['mean'], sigma=sig_contnd, p0 = (*p0_contnd, nu), maxfev=2000)#, bounds=bounds_genlog_contnd)
with np.errstate(over='ignore'):
    eeCcadMO_genlog_y_contnd_anglefit = general_logistic(x_fit, *eeCcadMO_genlog_popt_contnd_anglefit)






# Length of growth phase:
eeCcadMO_angle_growth_dur = growth_phase_end_eeCcadMO_angle - growth_phase_start_eeCcadMO_angle
#eeCcadMO_cont_growth_dur = growth_phase_end_eeCcadMO_cont - growth_phase_start_eeCcadMO_cont
eeCcadMO_contnd_growth_dur = growth_phase_end_eeCcadMO_contnd - growth_phase_start_eeCcadMO_contnd


# Magnitude of change during growth phase:
eeCcadMO_angle_growth_mag = growth_phase_end_eeCcadMO_angle_val - growth_phase_start_eeCcadMO_angle_val
#eeCcadMO_cont_growth_mag = growth_phase_end_eeCcadMO_cont_val - growth_phase_start_eeCcadMO_cont_val
eeCcadMO_contnd_growth_mag = growth_phase_end_eeCcadMO_contnd_val - growth_phase_start_eeCcadMO_contnd_val

# Average rate of change during growth phase:
eeCcadMO_growth_rate_lin_angle = eeCcadMO_angle_growth_mag / eeCcadMO_angle_growth_dur
#eeCcadMO_growth_rate_lin_cont = eeCcadMO_cont_growth_mag / eeCcadMO_cont_growth_dur
eeCcadMO_growth_rate_lin_contnd = eeCcadMO_contnd_growth_mag / eeCcadMO_contnd_growth_dur




###############################################################################
### Repeat for RcadMO:
eeRcadMO_list_dropna_fit_validity_df = pd.read_csv(Path.joinpath(fit_validity_dir, r"eeRcadMO plateau table contnd.csv"))
eeRcadMO_list_dropna_fit_validity_df = eeRcadMO_list_dropna_fit_validity_df[['Source_File', 'low_plat', 'growth_phase', 'high_plat']]
# The 1 line below would only used for determining contact dynamics:
eeRcadMO_Source_File_valid_low_plat = eeRcadMO_list_dropna_fit_validity_df.query('low_plat==True')['Source_File']

eeRcadMO_list_dropna_angle_fit_validity_df = pd.read_csv(Path.joinpath(fit_validity_dir, r"eeRcadMO plateau table angles.csv"))
eeRcadMO_list_dropna_angle_fit_validity_df = eeRcadMO_list_dropna_angle_fit_validity_df[['Source_File', 'low_plat', 'growth_phase', 'high_plat']]
# The 1 line below would only used for determining contact dynamics:
eeRcadMO_Source_File_angle_valid_low_plat = eeRcadMO_list_dropna_angle_fit_validity_df.query('low_plat==True')['Source_File']


# Time correct data for eeRcadMO-eeRcadMO:
# For dimensional contact diameters:
eeRcadMO_df_dropna_time_corr_cpu, eeRcadMO_df_dropna_cant_time_corr = time_correct_data(eeRcadMO_list_dropna, eeRcadMO_indiv_logistic_cont_df, eeRcadMO_list_dropna_fit_validity_df)
time_grouped_eeRcadMO_df_time_corr_cpu = eeRcadMO_df_dropna_time_corr_cpu.groupby(['Time (minutes)'])
time_grouped_eeRcadMO_df_cant_time_corr = eeRcadMO_df_dropna_cant_time_corr.groupby(['Time (minutes)'])
eeRcadMO_contact_mean = time_grouped_eeRcadMO_df_time_corr_cpu['Contact Surface Length (px)'].describe().reset_index()
eeRcadMO_contact_mean_cant_time_corr = time_grouped_eeRcadMO_df_cant_time_corr['Contact Surface Length (px)'].describe().reset_index()

# For contact angles:
eeRcadMO_df_dropna_angle_time_corr_cpu, eeRcadMO_df_dropna_angle_cant_time_corr = time_correct_data(eeRcadMO_contact_diam_non_dim_df_list_dropna, eeRcadMO_indiv_logistic_angle_df, eeRcadMO_list_dropna_angle_fit_validity_df)
time_grouped_eeRcadMO_df_angle_time_corr_cpu = eeRcadMO_df_dropna_angle_time_corr_cpu.groupby(['Time (minutes)'])
time_grouped_eeRcadMO_df_angle_cant_time_corr = eeRcadMO_df_dropna_angle_cant_time_corr.groupby(['Time (minutes)'])
eeRcadMO_angle_mean = time_grouped_eeRcadMO_df_angle_time_corr_cpu['Average Angle (deg)'].describe().reset_index()
eeRcadMO_angle_mean_cant_time_corr = time_grouped_eeRcadMO_df_angle_cant_time_corr['Average Angle (deg)'].describe().reset_index()

# For non-dimensional contact diameters:
eeRcadMO_fit_df_dropna_time_corr_cpu, eeRcadMO_fit_df_dropna_cant_time_corr = time_correct_data(eeRcadMO_contact_diam_non_dim_df_list_dropna, eeRcadMO_indiv_logistic_contnd_df, eeRcadMO_list_dropna_fit_validity_df)
time_grouped_eeRcadMO_fit_df_time_corr_cpu = eeRcadMO_fit_df_dropna_time_corr_cpu.groupby(['Time (minutes)'])
time_grouped_eeRcadMO_fit_df_cant_time_corr = eeRcadMO_fit_df_dropna_cant_time_corr.groupby(['Time (minutes)'])
eeRcadMO_fit_contact_mean = time_grouped_eeRcadMO_fit_df_time_corr_cpu['Contact Surface Length (N.D.)'].describe().reset_index()
eeRcadMO_fit_contact_mean_cant_time_corr = time_grouped_eeRcadMO_fit_df_cant_time_corr['Contact Surface Length (N.D.)'].describe().reset_index()








# Fit general logistic curve to mean angles:
sig_angle = 1./np.sqrt(eeRcadMO_angle_mean['count'])**2

eeRcadMO_genlog_popt_angle, eeRcadMO_genlog_pcov_angle = curve_fit(general_logistic, eeRcadMO_angle_mean['Time (minutes)'], eeRcadMO_angle_mean['mean'], sigma=sig_angle, p0 = (*p0_angle, nu), maxfev=2000)#, bounds=bounds_genlog_contnd)
with np.errstate(over='ignore'):
    eeRcadMO_genlog_y_angle = general_logistic(x_fit, *eeRcadMO_genlog_popt_angle)

eeRcadMO_genlog_y_angle_diff = np.diff(np.ma.masked_array(data=eeRcadMO_genlog_y_angle, mask=~np.isfinite(eeRcadMO_genlog_y_angle)))


percent90_high = np.argmin(abs(np.cumsum(eeRcadMO_genlog_y_angle_diff) - np.cumsum(eeRcadMO_genlog_y_angle_diff).max()*0.95))
percent90_low = np.argmin(abs(np.cumsum(eeRcadMO_genlog_y_angle_diff) - np.cumsum(eeRcadMO_genlog_y_angle_diff).max()*0.05))
percent99_high = np.argmin(abs(np.cumsum(eeRcadMO_genlog_y_angle_diff) - np.cumsum(eeRcadMO_genlog_y_angle_diff).max()*0.99))
percent99_low = np.argmin(abs(np.cumsum(eeRcadMO_genlog_y_angle_diff) - np.cumsum(eeRcadMO_genlog_y_angle_diff).max()*0.01))
#growth_phase_end_eeRcadMO_genlog_angle = x_fit[percent90_high]
#growth_phase_end_eeRcadMO_genlog_angle_val = eeRcadMO_genlog_y_angle[percent90_high]
#growth_phase_start_eeRcadMO_genlog_angle = x_fit[percent90_low]
#growth_phase_start_eeRcadMO_genlog_angle_val = eeRcadMO_genlog_y_angle[percent90_low]
growth_phase_end_eeRcadMO_angle = x_fit[percent90_high]
growth_phase_end_eeRcadMO_angle_val = eeRcadMO_genlog_y_angle[percent90_high]
growth_phase_start_eeRcadMO_angle = x_fit[percent90_low]
growth_phase_start_eeRcadMO_angle_val = eeRcadMO_genlog_y_angle[percent90_low]

growth_phase_end_eeRcadMO_angle_strict = x_fit[percent99_high]
growth_phase_end_eeRcadMO_angle_strict_val = eeRcadMO_genlog_y_angle[percent99_high]
growth_phase_start_eeRcadMO_angle_strict = x_fit[percent99_low]
growth_phase_start_eeRcadMO_angle_strict_val = eeRcadMO_genlog_y_angle[percent99_high]




# Fit general logistic curve to mean non-dimensionalized diameters:
sig_contnd = 1./np.sqrt(eeRcadMO_fit_contact_mean['count'])**2
# nu is the last parameter in the general logistic equation -- affects where
# fastest rate of growth (ie. inflection point) is on x-axis:
eeRcadMO_genlog_popt_contnd, eeRcadMO_genlog_pcov_contnd = curve_fit(general_logistic, eeRcadMO_fit_contact_mean['Time (minutes)'], eeRcadMO_fit_contact_mean['mean'], sigma=sig_contnd, p0 = (*p0_contnd, nu), maxfev=2000)#, bounds=bounds_genlog_contnd)
with np.errstate(over='ignore'):
    eeRcadMO_genlog_y_contnd = general_logistic(x_fit, *eeRcadMO_genlog_popt_contnd)

eeRcadMO_genlog_y_contnd_diff = np.diff(np.ma.masked_array(data=eeRcadMO_genlog_y_contnd, mask=~np.isfinite(eeRcadMO_genlog_y_contnd)))


#percent95 = np.argmin(abs(np.cumsum(eeRcadMO_y_contnd_diff) - np.cumsum(eeRcadMO_y_contnd_diff).max()*0.95))
#percent50 = np.argmin(abs(np.cumsum(eeRcadMO_y_contnd_diff) - np.cumsum(eeRcadMO_y_contnd_diff).max()*0.50))
percent90_high = np.argmin(abs(np.cumsum(eeRcadMO_genlog_y_contnd_diff) - np.cumsum(eeRcadMO_genlog_y_contnd_diff).max()*0.95))
percent90_low = np.argmin(abs(np.cumsum(eeRcadMO_genlog_y_contnd_diff) - np.cumsum(eeRcadMO_genlog_y_contnd_diff).max()*0.05))
percent99_high = np.argmin(abs(np.cumsum(eeRcadMO_genlog_y_contnd_diff) - np.cumsum(eeRcadMO_genlog_y_contnd_diff).max()*0.995))
percent99_low = np.argmin(abs(np.cumsum(eeRcadMO_genlog_y_contnd_diff) - np.cumsum(eeRcadMO_genlog_y_contnd_diff).max()*0.005))
#growth_phase_end_eeRcadMO_genlog_contnd = x_fit[percent90_high]
#growth_phase_end_eeRcadMO_genlog_contnd_val = eeRcadMO_genlog_y_contnd[percent90_high]
#growth_phase_start_eeRcadMO_genlog_contnd = x_fit[percent90_low]
#growth_phase_start_eeRcadMO_genlog_contnd_val = eeRcadMO_genlog_y_contnd[percent90_low]
growth_phase_end_eeRcadMO_contnd = x_fit[percent90_high]
growth_phase_end_eeRcadMO_contnd_val = eeRcadMO_genlog_y_contnd[percent90_high]
growth_phase_start_eeRcadMO_contnd = x_fit[percent90_low]
growth_phase_start_eeRcadMO_contnd_val = eeRcadMO_genlog_y_contnd[percent90_low]

growth_phase_end_eeRcadMO_contnd_strict = x_fit[percent99_high]
growth_phase_end_eeRcadMO_contnd_strict_val = eeRcadMO_genlog_y_contnd[percent99_high]
growth_phase_start_eeRcadMO_contnd_strict = x_fit[percent99_low]
growth_phase_start_eeRcadMO_contnd_strict_val = eeRcadMO_genlog_y_contnd[percent99_high]


#bounds_genlog_contnd = ([-50, 0, 1, 0, 0], [100, 100, 3, 1, 1000])
sig_cont = 1./np.sqrt(eeRcadMO_contact_mean['count'])**2
# nu is the last parameter in the general logistic equation -- affects where
# fastest rate of growth (ie. inflection point) is on x-axis:
eeRcadMO_genlog_popt_cont, eeRcadMO_genlog_pcov_cont = curve_fit(general_logistic, eeRcadMO_contact_mean['Time (minutes)'], eeRcadMO_contact_mean['mean'], sigma=sig_cont, p0 = (*p0_cont, nu), maxfev=2000)#, bounds=bounds_genlog_contnd)
with np.errstate(over='ignore'):
    eeRcadMO_genlog_y_cont = general_logistic(x_fit, *eeRcadMO_genlog_popt_cont)




# Also return the angle mean for the contact-based fit and vice-versa:
eeRcadMO_angle_mean_for_contnd_fit = time_grouped_eeRcadMO_fit_df_time_corr_cpu['Average Angle (deg)'].describe().reset_index()
# Fit general logistic curve to mean angles aligned according to contnd:
sig_angle = 1./np.sqrt(eeRcadMO_angle_mean_for_contnd_fit['count'])**2
eeRcadMO_genlog_popt_angle_contndfit, eeRcadMO_genlog_pcov_angle_contndfit = curve_fit(general_logistic, eeRcadMO_angle_mean_for_contnd_fit['Time (minutes)'], eeRcadMO_angle_mean_for_contnd_fit['mean'], sigma=sig_angle, p0 = (*p0_angle, nu), maxfev=2000)#, bounds=bounds_genlog_contnd)
with np.errstate(over='ignore'):
    eeRcadMO_genlog_y_angle_contndfit = general_logistic(x_fit, *eeRcadMO_genlog_popt_angle_contndfit)



eeRcadMO_contnd_mean_for_angle_fit = time_grouped_eeRcadMO_df_angle_time_corr_cpu['Contact Surface Length (N.D.)'].describe().reset_index()
# Fit general logistic curve to mean non-dimensionalized contact diameters
# aligned according to angle data:
sig_contnd = 1./np.sqrt(eeRcadMO_contnd_mean_for_angle_fit['count'])**2
# nu is the last parameter in the general logistic equation -- affects where
# fastest rate of growth (ie. inflection point) is on x-axis:
eeRcadMO_genlog_popt_contnd_anglefit, eeRcadMO_genlog_pcov_contnd_anglefit = curve_fit(general_logistic, eeRcadMO_contnd_mean_for_angle_fit['Time (minutes)'], eeRcadMO_contnd_mean_for_angle_fit['mean'], sigma=sig_contnd, p0 = (*p0_contnd, nu), maxfev=2000)#, bounds=bounds_genlog_contnd)
with np.errstate(over='ignore'):
    eeRcadMO_genlog_y_contnd_anglefit = general_logistic(x_fit, *eeRcadMO_genlog_popt_contnd_anglefit)




# Length of growth phase:
eeRcadMO_angle_growth_dur = growth_phase_end_eeRcadMO_angle - growth_phase_start_eeRcadMO_angle
#eeRcadMO_cont_growth_dur = growth_phase_end_eeRcadMO_cont - growth_phase_start_eeRcadMO_cont
eeRcadMO_contnd_growth_dur = growth_phase_end_eeRcadMO_contnd - growth_phase_start_eeRcadMO_contnd


# Magnitude of change during growth phase:
eeRcadMO_angle_growth_mag = growth_phase_end_eeRcadMO_angle_val - growth_phase_start_eeRcadMO_angle_val
#eeRcadMO_cont_growth_mag = growth_phase_end_eeRcadMO_cont_val - growth_phase_start_eeRcadMO_cont_val
eeRcadMO_contnd_growth_mag = growth_phase_end_eeRcadMO_contnd_val - growth_phase_start_eeRcadMO_contnd_val

# Average rate of change during growth phase:
eeRcadMO_growth_rate_lin_angle = eeRcadMO_angle_growth_mag / eeRcadMO_angle_growth_dur
#eeRcadMO_growth_rate_lin_cont = eeRcadMO_cont_growth_mag / eeRcadMO_cont_growth_dur
eeRcadMO_growth_rate_lin_contnd = eeRcadMO_contnd_growth_mag / eeRcadMO_contnd_growth_dur






###############################################################################
### Repeat for RCcadMO:
eeRCcadMO_list_dropna_fit_validity_df = pd.read_csv(Path.joinpath(fit_validity_dir, r"eeRCcadMO plateau table contnd.csv"))
eeRCcadMO_list_dropna_fit_validity_df = eeRCcadMO_list_dropna_fit_validity_df[['Source_File', 'low_plat', 'growth_phase', 'high_plat']]
# The 1 line below would only used for determining contact dynamics:
eeRCcadMO_Source_File_valid_low_plat = eeRCcadMO_list_dropna_fit_validity_df.query('low_plat==True')['Source_File']

eeRCcadMO_list_dropna_angle_fit_validity_df = pd.read_csv(Path.joinpath(fit_validity_dir, r"eeRCcadMO plateau table angles.csv"))
eeRCcadMO_list_dropna_angle_fit_validity_df = eeRCcadMO_list_dropna_angle_fit_validity_df[['Source_File', 'low_plat', 'growth_phase', 'high_plat']]
# The 1 line below would only used for determining contact dynamics:
eeRCcadMO_Source_File_angle_valid_low_plat = eeRCcadMO_list_dropna_angle_fit_validity_df.query('low_plat==True')['Source_File']


# Time correct data for eeRCcadMO-eeRCcadMO:
# For dimensional contact diameters:
eeRCcadMO_df_dropna_time_corr_cpu, eeRCcadMO_df_dropna_cant_time_corr = time_correct_data(eeRCcadMO_list_dropna, eeRCcadMO_indiv_logistic_cont_df, eeRCcadMO_list_dropna_fit_validity_df)
time_grouped_eeRCcadMO_df_time_corr_cpu = eeRCcadMO_df_dropna_time_corr_cpu.groupby(['Time (minutes)'])
time_grouped_eeRCcadMO_df_cant_time_corr = eeRCcadMO_df_dropna_cant_time_corr.groupby(['Time (minutes)'])
eeRCcadMO_contact_mean = time_grouped_eeRCcadMO_df_time_corr_cpu['Contact Surface Length (px)'].describe().reset_index()
eeRCcadMO_contact_mean_cant_time_corr = time_grouped_eeRCcadMO_df_cant_time_corr['Contact Surface Length (px)'].describe().reset_index()

# For contact angles:
eeRCcadMO_df_dropna_angle_time_corr_cpu, eeRCcadMO_df_dropna_angle_cant_time_corr = time_correct_data(eeRCcadMO_contact_diam_non_dim_df_list_dropna, eeRCcadMO_indiv_logistic_angle_df, eeRCcadMO_list_dropna_angle_fit_validity_df)
time_grouped_eeRCcadMO_df_angle_time_corr_cpu = eeRCcadMO_df_dropna_angle_time_corr_cpu.groupby(['Time (minutes)'])
time_grouped_eeRCcadMO_df_angle_cant_time_corr = eeRCcadMO_df_dropna_angle_cant_time_corr.groupby(['Time (minutes)'])
eeRCcadMO_angle_mean = time_grouped_eeRCcadMO_df_angle_time_corr_cpu['Average Angle (deg)'].describe().reset_index()
eeRCcadMO_angle_mean_cant_time_corr = time_grouped_eeRCcadMO_df_angle_cant_time_corr['Average Angle (deg)'].describe().reset_index()

# For non-dimensional contact diameters:
eeRCcadMO_fit_df_dropna_time_corr_cpu, eeRCcadMO_fit_df_dropna_cant_time_corr = time_correct_data(eeRCcadMO_contact_diam_non_dim_df_list_dropna, eeRCcadMO_indiv_logistic_contnd_df, eeRCcadMO_list_dropna_fit_validity_df)
time_grouped_eeRCcadMO_fit_df_time_corr_cpu = eeRCcadMO_fit_df_dropna_time_corr_cpu.groupby(['Time (minutes)'])
time_grouped_eeRCcadMO_fit_df_cant_time_corr = eeRCcadMO_fit_df_dropna_cant_time_corr.groupby(['Time (minutes)'])
eeRCcadMO_fit_contact_mean = time_grouped_eeRCcadMO_fit_df_time_corr_cpu['Contact Surface Length (N.D.)'].describe().reset_index()
eeRCcadMO_fit_contact_mean_cant_time_corr = time_grouped_eeRCcadMO_fit_df_cant_time_corr['Contact Surface Length (N.D.)'].describe().reset_index()








# Fit general logistic curve to mean angles:
sig_angle = 1./np.sqrt(eeRCcadMO_angle_mean['count'])**2

eeRCcadMO_genlog_popt_angle, eeRCcadMO_genlog_pcov_angle = curve_fit(general_logistic, eeRCcadMO_angle_mean['Time (minutes)'], eeRCcadMO_angle_mean['mean'], sigma=sig_angle, p0 = (*p0_angle, nu), maxfev=2000)#, bounds=bounds_genlog_contnd)
with np.errstate(over='ignore'):
    eeRCcadMO_genlog_y_angle = general_logistic(x_fit, *eeRCcadMO_genlog_popt_angle)

eeRCcadMO_genlog_y_angle_diff = np.diff(np.ma.masked_array(data=eeRCcadMO_genlog_y_angle, mask=~np.isfinite(eeRCcadMO_genlog_y_angle)))


percent90_high = np.argmin(abs(np.cumsum(eeRCcadMO_genlog_y_angle_diff) - np.cumsum(eeRCcadMO_genlog_y_angle_diff).max()*0.95))
percent90_low = np.argmin(abs(np.cumsum(eeRCcadMO_genlog_y_angle_diff) - np.cumsum(eeRCcadMO_genlog_y_angle_diff).max()*0.05))
percent99_high = np.argmin(abs(np.cumsum(eeRCcadMO_genlog_y_angle_diff) - np.cumsum(eeRCcadMO_genlog_y_angle_diff).max()*0.99))
percent99_low = np.argmin(abs(np.cumsum(eeRCcadMO_genlog_y_angle_diff) - np.cumsum(eeRCcadMO_genlog_y_angle_diff).max()*0.01))
#growth_phase_end_eeRCcadMO_genlog_angle = x_fit[percent90_high]
#growth_phase_end_eeRCcadMO_genlog_angle_val = eeRCcadMO_genlog_y_angle[percent90_high]
#growth_phase_start_eeRCcadMO_genlog_angle = x_fit[percent90_low]
#growth_phase_start_eeRCcadMO_genlog_angle_val = eeRCcadMO_genlog_y_angle[percent90_low]
growth_phase_end_eeRCcadMO_angle = x_fit[percent90_high]
growth_phase_end_eeRCcadMO_angle_val = eeRCcadMO_genlog_y_angle[percent90_high]
growth_phase_start_eeRCcadMO_angle = x_fit[percent90_low]
growth_phase_start_eeRCcadMO_angle_val = eeRCcadMO_genlog_y_angle[percent90_low]

growth_phase_end_eeRCcadMO_angle_strict = x_fit[percent99_high]
growth_phase_end_eeRCcadMO_angle_strict_val = eeRCcadMO_genlog_y_angle[percent99_high]
growth_phase_start_eeRCcadMO_angle_strict = x_fit[percent99_low]
growth_phase_start_eeRCcadMO_angle_strict_val = eeRCcadMO_genlog_y_angle[percent99_high]




# Fit general logistic curve to mean non-dimensionalized diameters:
sig_contnd = 1./np.sqrt(eeRCcadMO_fit_contact_mean['count'])**2
# nu is the last parameter in the general logistic equation -- affects where
# fastest rate of growth (ie. inflection point) is on x-axis:
eeRCcadMO_genlog_popt_contnd, eeRCcadMO_genlog_pcov_contnd = curve_fit(general_logistic, eeRCcadMO_fit_contact_mean['Time (minutes)'], eeRCcadMO_fit_contact_mean['mean'], sigma=sig_contnd, p0 = (*p0_contnd, nu), maxfev=2000)#, bounds=bounds_genlog_contnd)
with np.errstate(over='ignore'):
    eeRCcadMO_genlog_y_contnd = general_logistic(x_fit, *eeRCcadMO_genlog_popt_contnd)

eeRCcadMO_genlog_y_contnd_diff = np.diff(np.ma.masked_array(data=eeRCcadMO_genlog_y_contnd, mask=~np.isfinite(eeRCcadMO_genlog_y_contnd)))


#percent95 = np.argmin(abs(np.cumsum(eeRCcadMO_y_contnd_diff) - np.cumsum(eeRCcadMO_y_contnd_diff).max()*0.95))
#percent50 = np.argmin(abs(np.cumsum(eeRCcadMO_y_contnd_diff) - np.cumsum(eeRCcadMO_y_contnd_diff).max()*0.50))
percent90_high = np.argmin(abs(np.cumsum(eeRCcadMO_genlog_y_contnd_diff) - np.cumsum(eeRCcadMO_genlog_y_contnd_diff).max()*0.95))
percent90_low = np.argmin(abs(np.cumsum(eeRCcadMO_genlog_y_contnd_diff) - np.cumsum(eeRCcadMO_genlog_y_contnd_diff).max()*0.05))
percent99_high = np.argmin(abs(np.cumsum(eeRCcadMO_genlog_y_contnd_diff) - np.cumsum(eeRCcadMO_genlog_y_contnd_diff).max()*0.995))
percent99_low = np.argmin(abs(np.cumsum(eeRCcadMO_genlog_y_contnd_diff) - np.cumsum(eeRCcadMO_genlog_y_contnd_diff).max()*0.005))
#growth_phase_end_eeRCcadMO_genlog_contnd = x_fit[percent90_high]
#growth_phase_end_eeRCcadMO_genlog_contnd_val = eeRCcadMO_genlog_y_contnd[percent90_high]
#growth_phase_start_eeRCcadMO_genlog_contnd = x_fit[percent90_low]
#growth_phase_start_eeRCcadMO_genlog_contnd_val = eeRCcadMO_genlog_y_contnd[percent90_low]
growth_phase_end_eeRCcadMO_contnd = x_fit[percent90_high]
growth_phase_end_eeRCcadMO_contnd_val = eeRCcadMO_genlog_y_contnd[percent90_high]
growth_phase_start_eeRCcadMO_contnd = x_fit[percent90_low]
growth_phase_start_eeRCcadMO_contnd_val = eeRCcadMO_genlog_y_contnd[percent90_low]

growth_phase_end_eeRCcadMO_contnd_strict = x_fit[percent99_high]
growth_phase_end_eeRCcadMO_contnd_strict_val = eeRCcadMO_genlog_y_contnd[percent99_high]
growth_phase_start_eeRCcadMO_contnd_strict = x_fit[percent99_low]
growth_phase_start_eeRCcadMO_contnd_strict_val = eeRCcadMO_genlog_y_contnd[percent99_high]


#bounds_genlog_contnd = ([-50, 0, 1, 0, 0], [100, 100, 3, 1, 1000])
sig_cont = 1./np.sqrt(eeRCcadMO_contact_mean['count'])**2
# nu is the last parameter in the general logistic equation -- affects where
# fastest rate of growth (ie. inflection point) is on x-axis:
eeRCcadMO_genlog_popt_cont, eeRCcadMO_genlog_pcov_cont = curve_fit(general_logistic, eeRCcadMO_contact_mean['Time (minutes)'], eeRCcadMO_contact_mean['mean'], sigma=sig_cont, p0 = (*p0_cont, nu), maxfev=2000)#, bounds=bounds_genlog_contnd)
with np.errstate(over='ignore'):
    eeRCcadMO_genlog_y_cont = general_logistic(x_fit, *eeRCcadMO_genlog_popt_cont)




# Also return the angle mean for the contact-based fit and vice-versa:
eeRCcadMO_angle_mean_for_contnd_fit = time_grouped_eeRCcadMO_fit_df_time_corr_cpu['Average Angle (deg)'].describe().reset_index()
# Fit general logistic curve to mean angles aligned according to contnd:
sig_angle = 1./np.sqrt(eeRCcadMO_angle_mean_for_contnd_fit['count'])**2
eeRCcadMO_genlog_popt_angle_contndfit, eeRCcadMO_genlog_pcov_angle_contndfit = curve_fit(general_logistic, eeRCcadMO_angle_mean_for_contnd_fit['Time (minutes)'], eeRCcadMO_angle_mean_for_contnd_fit['mean'], sigma=sig_angle, p0 = (*p0_angle, nu), maxfev=2000)#, bounds=bounds_genlog_contnd)
with np.errstate(over='ignore'):
    eeRCcadMO_genlog_y_angle_contndfit = general_logistic(x_fit, *eeRCcadMO_genlog_popt_angle_contndfit)



eeRCcadMO_contnd_mean_for_angle_fit = time_grouped_eeRCcadMO_df_angle_time_corr_cpu['Contact Surface Length (N.D.)'].describe().reset_index()
# Fit general logistic curve to mean non-dimensionalized contact diameters
# aligned according to angle data:
sig_contnd = 1./np.sqrt(eeRCcadMO_contnd_mean_for_angle_fit['count'])**2
# nu is the last parameter in the general logistic equation -- affects where
# fastest rate of growth (ie. inflection point) is on x-axis:
eeRCcadMO_genlog_popt_contnd_anglefit, eeRCcadMO_genlog_pcov_contnd_anglefit = curve_fit(general_logistic, eeRCcadMO_contnd_mean_for_angle_fit['Time (minutes)'], eeRCcadMO_contnd_mean_for_angle_fit['mean'], sigma=sig_contnd, p0 = (*p0_contnd, nu), maxfev=2000)#, bounds=bounds_genlog_contnd)
with np.errstate(over='ignore'):
    eeRCcadMO_genlog_y_contnd_anglefit = general_logistic(x_fit, *eeRCcadMO_genlog_popt_contnd_anglefit)




# Length of growth phase:
eeRCcadMO_angle_growth_dur = growth_phase_end_eeRCcadMO_angle - growth_phase_start_eeRCcadMO_angle
#eeRCcadMO_cont_growth_dur = growth_phase_end_eeRCcadMO_cont - growth_phase_start_eeRCcadMO_cont
eeRCcadMO_contnd_growth_dur = growth_phase_end_eeRCcadMO_contnd - growth_phase_start_eeRCcadMO_contnd


# Magnitude of change during growth phase:
eeRCcadMO_angle_growth_mag = growth_phase_end_eeRCcadMO_angle_val - growth_phase_start_eeRCcadMO_angle_val
#eeRCcadMO_cont_growth_mag = growth_phase_end_eeRCcadMO_cont_val - growth_phase_start_eeRCcadMO_cont_val
eeRCcadMO_contnd_growth_mag = growth_phase_end_eeRCcadMO_contnd_val - growth_phase_start_eeRCcadMO_contnd_val

# Average rate of change during growth phase:
eeRCcadMO_growth_rate_lin_angle = eeRCcadMO_angle_growth_mag / eeRCcadMO_angle_growth_dur
#eeRCcadMO_growth_rate_lin_cont = eeRCcadMO_cont_growth_mag / eeRCcadMO_cont_growth_dur
eeRCcadMO_growth_rate_lin_contnd = eeRCcadMO_contnd_growth_mag / eeRCcadMO_contnd_growth_dur





### Now that you have acquired all of the general logistic parameters and
### and growth periods, output them into a csv file for convenient access:
# for the angle data:
def get_fit_Nsizes(validity_df):
    n_fit_good = np.count_nonzero(validity_df['growth_phase'])
    n_fit_bad = np.count_nonzero(validity_df['growth_phase']== False)
    n_fit_total = n_fit_good + n_fit_bad
    return n_fit_good, n_fit_bad, n_fit_total

col_labels = [('Experimental_Condition', 'Low Plateau', 'High Plateau', 
              r'Logistic Growth Rate ($\kappa$)', 'Maximum Growth Rate (deg/min)', r'$\nu$',
              'Growth Phase Duration (minutes)', 'Average Growth Rate (deg/min)',
              'good_fit_n', 'bad_fit_n', 'total_n')]
rows = [('ecto-ecto', *ee_genlog_popt_angle[3:1:-1], 
         ee_genlog_popt_angle[1], (ee_genlog_popt_angle[2] - ee_genlog_popt_angle[3])*ee_genlog_popt_angle[1],
         ee_genlog_popt_angle[4], ee_angle_growth_dur, ee_growth_rate_lin_angle, 
         *get_fit_Nsizes(ee_list_dropna_angle_fit_validity_df),), 
        ('meso-meso', *mm_genlog_popt_angle[3:1:-1], 
         mm_genlog_popt_angle[1], (mm_genlog_popt_angle[2] - mm_genlog_popt_angle[3])*mm_genlog_popt_angle[1], 
         mm_genlog_popt_angle[4], mm_angle_growth_dur, mm_growth_rate_lin_angle, 
         *get_fit_Nsizes(mm_list_dropna_angle_fit_validity_df),), 
        ('endo-endo', *nn_genlog_popt_angle[3:1:-1], 
         nn_genlog_popt_angle[1], (nn_genlog_popt_angle[2] - nn_genlog_popt_angle[3])*nn_genlog_popt_angle[1], 
         nn_genlog_popt_angle[4], nn_angle_growth_dur, nn_growth_rate_lin_angle, 
         *get_fit_Nsizes(nn_list_dropna_angle_fit_validity_df),), 
        ('ecto-meso', *em_genlog_popt_angle[3:1:-1], 
         em_genlog_popt_angle[1], (em_genlog_popt_angle[2] - em_genlog_popt_angle[3])*em_genlog_popt_angle[1], 
         em_genlog_popt_angle[4], em_angle_growth_dur, em_growth_rate_lin_angle, 
         *get_fit_Nsizes(em_list_dropna_angle_fit_validity_df),), 
        ('ecto-ecto in 1xDB', *eedb_genlog_popt_angle[3:1:-1], 
         eedb_genlog_popt_angle[1], (eedb_genlog_popt_angle[2] - eedb_genlog_popt_angle[3])*eedb_genlog_popt_angle[1], 
         eedb_genlog_popt_angle[4], eedb_angle_growth_dur, eedb_growth_rate_lin_angle, 
         *get_fit_Nsizes(eedb_list_dropna_angle_fit_validity_df),), 
        ('ecto-ecto CcadMO', *eeCcadMO_genlog_popt_angle[3:1:-1], 
         eeCcadMO_genlog_popt_angle[1], (eeCcadMO_genlog_popt_angle[2] - eeCcadMO_genlog_popt_angle[3])*eeCcadMO_genlog_popt_angle[1], 
         eeCcadMO_genlog_popt_angle[4], eeCcadMO_angle_growth_dur, eeCcadMO_growth_rate_lin_angle, 
         *get_fit_Nsizes(eeCcadMO_list_dropna_angle_fit_validity_df),), 
        ('ecto-ecto RcadMO', *eeRcadMO_genlog_popt_angle[3:1:-1], 
         eeRcadMO_genlog_popt_angle[1], (eeRcadMO_genlog_popt_angle[2] - eeRcadMO_genlog_popt_angle[3])*eeRcadMO_genlog_popt_angle[1], 
         eeRcadMO_genlog_popt_angle[4], eeRcadMO_angle_growth_dur, eeRcadMO_growth_rate_lin_angle, 
         *get_fit_Nsizes(eeRcadMO_list_dropna_angle_fit_validity_df),), 
        ('ecto-ecto CcadMO + RcadMO', *eeRCcadMO_genlog_popt_angle[3:1:-1], 
         eeRCcadMO_genlog_popt_angle[1], (eeRCcadMO_genlog_popt_angle[2] - eeRCcadMO_genlog_popt_angle[3])*eeRCcadMO_genlog_popt_angle[1], 
         eeRCcadMO_genlog_popt_angle[4], eeRCcadMO_angle_growth_dur, eeRCcadMO_growth_rate_lin_angle, 
         *get_fit_Nsizes(eeRCcadMO_list_dropna_angle_fit_validity_df),), ]
genlog_param_summary_angles = col_labels + rows



# and for the contnd data:
col_labels = [('Experimental_Condition', 'Low Plateau', 'High Plateau', 
              r'Logistic Growth Rate ($\kappa$)', 'Maximum Growth Rate (N.D./min)', r'$\nu$',
              'Growth Phase Duration (minutes)', 'Average Growth Rate (N.D./min)',
              'good_fit_n', 'bad_fit_n', 'total_n')]
rows = [('ecto-ecto', *ee_genlog_popt_contnd[3:1:-1], 
         ee_genlog_popt_contnd[1], (ee_genlog_popt_contnd[2] - ee_genlog_popt_contnd[3])*ee_genlog_popt_contnd[1], 
         ee_genlog_popt_contnd[4], ee_contnd_growth_dur, ee_growth_rate_lin_contnd,
         *get_fit_Nsizes(ee_list_dropna_fit_validity_df),),
        ('meso-meso', *mm_genlog_popt_contnd[3:1:-1], 
         mm_genlog_popt_contnd[1], (mm_genlog_popt_contnd[2] - mm_genlog_popt_contnd[3])*mm_genlog_popt_contnd[1], 
         mm_genlog_popt_contnd[4], mm_contnd_growth_dur, mm_growth_rate_lin_contnd, 
         *get_fit_Nsizes(mm_list_dropna_fit_validity_df),),
        ('endo-endo', *nn_genlog_popt_contnd[3:1:-1], 
         nn_genlog_popt_contnd[1], (nn_genlog_popt_contnd[2] - nn_genlog_popt_contnd[3])*nn_genlog_popt_contnd[1], 
         nn_genlog_popt_contnd[4], nn_contnd_growth_dur, nn_growth_rate_lin_contnd, 
         *get_fit_Nsizes(nn_list_dropna_fit_validity_df),),
        ('ecto-meso', *em_genlog_popt_contnd[3:1:-1], 
         em_genlog_popt_contnd[1], (em_genlog_popt_contnd[2] - em_genlog_popt_contnd[3])*em_genlog_popt_contnd[1], 
         em_genlog_popt_contnd[4], em_contnd_growth_dur, em_growth_rate_lin_contnd, 
         *get_fit_Nsizes(em_list_dropna_fit_validity_df),),
        ('ecto-ecto in 1xDB', *eedb_genlog_popt_contnd[3:1:-1], 
         eedb_genlog_popt_contnd[1], (eedb_genlog_popt_contnd[2] - eedb_genlog_popt_contnd[3])*eedb_genlog_popt_contnd[1], 
         eedb_genlog_popt_contnd[4], eedb_contnd_growth_dur, eedb_growth_rate_lin_contnd, 
         *get_fit_Nsizes(eedb_list_dropna_fit_validity_df),),
        ('ecto-ecto CcadMO', *eeCcadMO_genlog_popt_contnd[3:1:-1], 
         eeCcadMO_genlog_popt_contnd[1], (eeCcadMO_genlog_popt_contnd[2] - eeCcadMO_genlog_popt_contnd[3])*eeCcadMO_genlog_popt_contnd[1], 
         eeCcadMO_genlog_popt_contnd[4], eeCcadMO_contnd_growth_dur, eeCcadMO_growth_rate_lin_contnd, 
         *get_fit_Nsizes(eeCcadMO_list_dropna_fit_validity_df),),
        ('ecto-ecto RcadMO', *eeRcadMO_genlog_popt_contnd[3:1:-1], 
         eeRcadMO_genlog_popt_contnd[1], (eeRcadMO_genlog_popt_contnd[2] - eeRcadMO_genlog_popt_contnd[3])*eeRcadMO_genlog_popt_contnd[1], 
         eeRcadMO_genlog_popt_contnd[4], eeRcadMO_contnd_growth_dur, eeRcadMO_growth_rate_lin_contnd, 
         *get_fit_Nsizes(eeRcadMO_list_dropna_fit_validity_df),),
        ('ecto-ecto CcadMO + RcadMO', *eeRCcadMO_genlog_popt_contnd[3:1:-1], 
         eeRCcadMO_genlog_popt_contnd[1], (eeRCcadMO_genlog_popt_contnd[2] - eeRCcadMO_genlog_popt_contnd[3])*eeRCcadMO_genlog_popt_contnd[1], 
         eeRCcadMO_genlog_popt_contnd[4], eeRCcadMO_contnd_growth_dur, eeRCcadMO_growth_rate_lin_contnd, 
         *get_fit_Nsizes(eeRCcadMO_list_dropna_fit_validity_df),), ]
genlog_param_summary_contnd = col_labels + rows




# Output the general logistic parameters to a .csv:
with open(Path.joinpath(output_dir, r'gen_logistic_param_summ_angles.csv'), 'w', 
          newline='') as file_out:
    csv_obj = csv.writer(file_out)#, delimiter=',')
    csv_obj.writerows(genlog_param_summary_angles)

with open(Path.joinpath(output_dir, r'gen_logistic_param_summ_contnd.csv'), 'w', 
          newline='') as file_out:
    csv_obj = csv.writer(file_out)#, delimiter=',')
    csv_obj.writerows(genlog_param_summary_contnd)




###############################################################################

### We also want to check what the growth rates to the low plateas are:

# Calculating growth rates of low plateaus: (all manually done)
grow_to_low_valid_angles = pd.read_csv(Path.joinpath(grow_to_low_fit_valid_dir, r"angle_growth_to_low_plateau_validity.csv"))
grow_to_low_valid_contnd = pd.read_csv(Path.joinpath(grow_to_low_fit_valid_dir, r"contnd_growth_to_low_plateau_validity.csv"))

##############################################
### contND variables that have been completed:
#ee_contact_diam_non_dim_df_list_dropna
#ee_indiv_logistic_contnd_df

#mm_contact_diam_non_dim_df_list_dropna
#mm_indiv_logistic_contnd_df

#nn_contact_diam_non_dim_df_list_dropna
#nn_indiv_logistic_contnd_df

#em_contact_diam_non_dim_df_list_dropna
#em_indiv_logistic_contnd_df

#eedb_contact_diam_non_dim_df_list_dropna
#eedb_indiv_logistic_contnd_df

#eeCcadMO_contact_diam_non_dim_df_list_dropna
#eeCcadMO_indiv_logistic_contnd_df

#eeRcadMO_contact_diam_non_dim_df_list_dropna
#eeRcadMO_indiv_logistic_contnd_df

#eeRCcadMO_contact_diam_non_dim_df_list_dropna
#eeRCcadMO_indiv_logistic_contnd_df



##############################################
### Angle variables that have been completed: 
#ee_contact_diam_non_dim_df_list_dropna
#ee_indiv_logistic_angle_df

#mm_contact_diam_non_dim_df_list_dropna
#mm_indiv_logistic_angle_df

#nn_contact_diam_non_dim_df_list_dropna
#nn_indiv_logistic_angle_df

#em_contact_diam_non_dim_df_list_dropna
#em_indiv_logistic_angle_df

#eedb_contact_diam_non_dim_df_list_dropna
#eedb_indiv_logistic_angle_df

#eeCcadMO_contact_diam_non_dim_df_list_dropna
#eeCcadMO_indiv_logistic_angle_df

#eeRcadMO_contact_diam_non_dim_df_list_dropna
#eeRcadMO_indiv_logistic_angle_df

#eeRcadMO_contact_diam_non_dim_df_list_dropna
#eeRcadMO_indiv_logistic_angle_df



###############################################################################
### The following is used only to help me manually select where the low plateau
### ends for each timelapse of a cell-cell adhesion event.
### BTW: all lines that start with a single "#" should be un-commented up to 
### the next full line of "#"

def pickable_scatter(fig, ax, x, y):
    def on_click(event):
        ind = event.ind
        print('\nind:', ind, '\ntime:', x[ind], '\ny_val:', y[ind])
    ax.scatter(x, y, picker=True)
    cid = fig.canvas.mpl_connect('pick_event', on_click)


### For the following interactive plots to work, you need to switch matplotlib
### from 'inline' to the 'qt' backend (ie. the plots won't show up in the 
### console anymore). This, however, won't work in the script so you need to 
### type it in the console yourself:
# %matplotlib qt
### Contact Diameters:
# data_list = eeRCcadMO_contact_diam_non_dim_df_list_dropna
# fitted_logistic = eeRCcadMO_indiv_logistic_contnd_df
# data_list = ee_contact_diam_non_dim_df_list_dropna
# fitted_logistic = ee_indiv_logistic_contnd_df
# i = 49

# fig, ax = plt.subplots()
# x, y = data_list[i]['Time (minutes)'].values, data_list[i]['Contact Surface Length (N.D.)'].values
# sf = data_list[i]['Source_File'].iloc[0]
# pickable_scatter(fig, ax, x, y)
# ax.plot(x, y, c='lightgrey', zorder=0)
# ax.plot(x_fit, logistic(x_fit, fitted_logistic.iloc[i]['popt_x2'], fitted_logistic.iloc[i]['popt_k2'], fitted_logistic.iloc[i]['popt_high_plateau'], fitted_logistic.iloc[i]['popt_mid_plateau']), c='r')
# title = ax.set_title('%s -- '%i + sf + ' -- contND')
# xmin, xmax = ax.set_xlim(x.min() - 10, x.max() + 10)
# ymin = ax.set_ylim(0)



### Angles:
# data_list = eeRCcadMO_contact_diam_non_dim_df_list_dropna
# fitted_logistic = eeRCcadMO_indiv_logistic_angle_df
data_list = ee_contact_diam_non_dim_df_list_dropna
fitted_logistic = ee_indiv_logistic_angle_df

i = 1

fig, ax = plt.subplots()
x, y = data_list[i]['Time (minutes)'].values, data_list[i]['Average Angle (deg)'].values
sf = data_list[i]['Source_File'].iloc[0]
pickable_scatter(fig, ax, x, y)
ax.plot(x, y, c='lightgrey', zorder=0)
ax.plot(x_fit, logistic(x_fit, fitted_logistic.iloc[i]['popt_x2'], fitted_logistic.iloc[i]['popt_k2'], fitted_logistic.iloc[i]['popt_high_plateau'], fitted_logistic.iloc[i]['popt_mid_plateau']), c='r')
title = ax.set_title('%s -- '%i + sf + ' -- Angles')
xmin, xmax = ax.set_xlim(x.min() - 10, x.max() + 10)
ymin = ax.set_ylim(0)





###############################################################################

### Get the growth to the low plateau subsets from each of the cell 
### combinations:
low_growth_contnd_ee_original, low_growth_contnd_ee_details = subset_low_growth_plat(ee_list_original, ee_list_original, grow_to_low_valid_contnd)
ee_grow_to_low_validity_contnd = grow_to_low_valid_contnd[grow_to_low_valid_contnd['Experimental_Condition'] == 'ecto-ecto']
ee_grow_to_low_valid_contnd = [ee_list_original[i] for i in np.where(np.isfinite(ee_grow_to_low_validity_contnd['last_low_plat_timeframe (minutes; t = ...)']).values)[0]]
ee_low_adh_contnd_max_t = [x['Time (minutes)'].max() for x in low_growth_contnd_ee_original]
# The variable 'ee_low_adh_contnd' contains a subset of 'ee_list_original' that
# only contains low adhesion data that I have manually set cutoffs for:
ee_low_adh_contnd  = [ee_grow_to_low_valid_contnd[i][ee_grow_to_low_valid_contnd[i]['Time (minutes)'] <= t_max] for i, t_max in enumerate(ee_low_adh_contnd_max_t)]

low_growth_angle_ee_original, low_growth_angle_ee_details = subset_low_growth_plat(ee_list_original, ee_list_original, grow_to_low_valid_angles)
ee_grow_to_low_validity_angle = grow_to_low_valid_angles[grow_to_low_valid_angles['Experimental_Condition'] == 'ecto-ecto']
ee_grow_to_low_valid_angle = [ee_list_original[i] for i in np.where(np.isfinite(ee_grow_to_low_validity_angle['last_low_plat_timeframe (minutes; t = ...)']).values)[0]]
ee_low_adh_angle_max_t = [x['Time (minutes)'].max() for x in low_growth_angle_ee_original]
# The variable 'ee_low_adh_angle' contains a subset of 'ee_list_original' that
# only contains low adhesion data that I have manually set cutoffs for:
ee_low_adh_angle = [ee_grow_to_low_valid_angle[i][ee_grow_to_low_valid_angle[i]['Time (minutes)'] <= t_max] for i, t_max in enumerate(ee_low_adh_angle_max_t)]

low_growth_cont_ee, low_growth_cont_ee_details = subset_low_growth_plat(ee_list_dropna, ee_list, grow_to_low_valid_contnd)
low_growth_contnd_ee, low_growth_contnd_ee_details = subset_low_growth_plat(ee_contact_diam_non_dim_df_list_dropna, ee_list, grow_to_low_valid_contnd)
low_growth_angle_ee, low_growth_angle_ee_details = subset_low_growth_plat(ee_contact_diam_non_dim_df_list_dropna, ee_list, grow_to_low_valid_angles)

low_growth_contnd_mm, low_growth_contnd_mm_details = subset_low_growth_plat(mm_contact_diam_non_dim_df_list_dropna, mm_list, grow_to_low_valid_contnd)
low_growth_angle_mm, low_growth_angle_mm_details = subset_low_growth_plat(mm_contact_diam_non_dim_df_list_dropna, mm_list, grow_to_low_valid_angles)

low_growth_contnd_nn, low_growth_contnd_nn_details = subset_low_growth_plat(nn_contact_diam_non_dim_df_list_dropna, nn_list, grow_to_low_valid_contnd)
low_growth_angle_nn, low_growth_angle_nn_details = subset_low_growth_plat(nn_contact_diam_non_dim_df_list_dropna, nn_list, grow_to_low_valid_angles)

low_growth_contnd_em, low_growth_contnd_em_details = subset_low_growth_plat(em_contact_diam_non_dim_df_list_dropna, em_list, grow_to_low_valid_contnd)
low_growth_angle_em, low_growth_angle_em_details = subset_low_growth_plat(em_contact_diam_non_dim_df_list_dropna, em_list, grow_to_low_valid_angles)

low_growth_contnd_eedb, low_growth_contnd_eedb_details = subset_low_growth_plat(eedb_contact_diam_non_dim_df_list_dropna, eedb_list, grow_to_low_valid_contnd)
low_growth_angle_eedb, low_growth_angle_eedb_details = subset_low_growth_plat(eedb_contact_diam_non_dim_df_list_dropna, eedb_list, grow_to_low_valid_angles)

low_growth_contnd_eeCcadMO, low_growth_contnd_eeCcadMO_details = subset_low_growth_plat(eeCcadMO_contact_diam_non_dim_df_list_dropna, eeCcadMO_list, grow_to_low_valid_contnd)
low_growth_angle_eeCcadMO, low_growth_angle_eeCcadMO_details = subset_low_growth_plat(eeCcadMO_contact_diam_non_dim_df_list_dropna, eeCcadMO_list, grow_to_low_valid_angles)

low_growth_contnd_eeRcadMO, low_growth_contnd_eeRcadMO_details = subset_low_growth_plat(eeRcadMO_contact_diam_non_dim_df_list_dropna, eeRcadMO_list, grow_to_low_valid_contnd)
low_growth_angle_eeRcadMO, low_growth_angle_eeRcadMO_details = subset_low_growth_plat(eeRcadMO_contact_diam_non_dim_df_list_dropna, eeRcadMO_list, grow_to_low_valid_angles)

low_growth_contnd_eeRCcadMO, low_growth_contnd_eeRCcadMO_details = subset_low_growth_plat(eeRCcadMO_contact_diam_non_dim_df_list_dropna, eeRCcadMO_list, grow_to_low_valid_contnd)
low_growth_angle_eeRCcadMO, low_growth_angle_eeRCcadMO_details = subset_low_growth_plat(eeRCcadMO_contact_diam_non_dim_df_list_dropna, eeRCcadMO_list, grow_to_low_valid_angles)



### Fit individual logistic functions to each of the low plateau curves:
## First define the initial parameters to use in the curve_fit function:
# parameter order is: (inflection_point, logistic_growth_rate, high_plateau, low_plateau)

p0_angle_low = (10,0.5,60,-60)
bounds_angle_low = ([-50,0,0,-120], [100,100, 120,0]) #([set of lower bounds], [set of upper bounds])

p0_contnd_low = (10,0.5,0.5,-0.5)
bounds_contnd_low = ([-50,0,0,-2], [100,100,2,0]) #([set of lower bounds], [set of upper bounds])


## Now perform the logistic fit on the individual curves:
# Define some lists of optimization outcomes:
def fit_logistic_to_data(list_of_dataframes, x_col, y_col, initial_parameters, fit_x_min, fit_x_max, step_size, bounds=(-np.inf, np.inf), estim_init_x=False):
    
    indiv_logistic_fits_mesg = []
    indiv_logistic_popt_pcov = []
    indiv_logistic_x = []
    indiv_logistic_y = []
    indiv_logistic = []
    
    # Fitting curves:
    for df in list_of_dataframes:
        with warnings.catch_warnings():
            warnings.simplefilter('error', OptimizeWarning)
            try:
                if estim_init_x == True:
                    initial_parameters = list(initial_parameters)
                    initial_parameters[0] = df[x_col].min()
                else:
                    pass
                popt, pcov = curve_fit(logistic, df[x_col], df[y_col], p0 = (*initial_parameters,), bounds=bounds)
                x = np.linspace(fit_x_min, fit_x_max, step_size)
                y = logistic(x, *popt)        
                
                indiv_logistic_fits_mesg.append(("Success!", df['Source_File'].iloc[0]))
                indiv_logistic_popt_pcov.append((popt, pcov, df['Source_File'].iloc[0]))
                indiv_logistic_x.append(x)
                indiv_logistic_y.append(y)
                indiv_logistic.append(("Success", df['Source_File'].iloc[0], bounds, *initial_parameters, *popt, pcov, x, y))
            except (RuntimeError, ValueError, OptimizeWarning) as e:
                popt, pcov = np.asarray([np.float('nan')]  * len(initial_parameters)), np.asarray([np.float('nan')])
                x = np.linspace(fit_x_min, fit_x_max, step_size)
                y = [np.float('nan')] * len(x)
                
                indiv_logistic_fits_mesg.append((e, df['Source_File'].iloc[0]))
                rt_err_popt_placeholder = np.empty((4,))
                rt_err_popt_placeholder[:] = np.nan
                rt_err_pcov_placeholder = np.empty((4,4))
                rt_err_pcov_placeholder[:] = np.nan
                indiv_logistic_popt_pcov.append((rt_err_popt_placeholder, rt_err_pcov_placeholder, df['Source_File'].iloc[0]))
                indiv_logistic_x.append([np.float('nan')])
                indiv_logistic_y.append([np.float('nan')])
                indiv_logistic.append((e, df['Source_File'].iloc[0], bounds, *initial_parameters, *popt, pcov, x, y))
    
    indiv_logistic_df = pd.DataFrame(data=indiv_logistic, columns=['mesg', 'Source_File', 'bounds', 'p0_x2', 'p0_k2', 'p0_high_plateau', 'p0_mid_plateau', 'popt_x2', 'popt_k2', 'popt_high_plateau', 'popt_mid_plateau', 'pcov', 'logistic_x', 'logistic_y'])

    return indiv_logistic_df



indiv_logistic_df_low_growth_ee_angle = fit_logistic_to_data(low_growth_angle_ee, 'Time (minutes)', 'Average Angle (deg)', p0_angle_low, fit_time_min, fit_time_max, step_size, bounds_angle_low, estim_init_x=True)
indiv_logistic_df_low_growth_ee_contnd = fit_logistic_to_data(low_growth_contnd_ee, 'Time (minutes)', 'Contact Surface Length (N.D.)', p0_contnd_low, fit_time_min, fit_time_max, step_size, bounds_contnd_low, estim_init_x=True)

indiv_logistic_df_low_growth_mm_angle = fit_logistic_to_data(low_growth_angle_mm, 'Time (minutes)', 'Average Angle (deg)', p0_angle_low, fit_time_min, fit_time_max, step_size, bounds_angle_low, estim_init_x=True)
indiv_logistic_df_low_growth_mm_contnd = fit_logistic_to_data(low_growth_contnd_mm, 'Time (minutes)', 'Contact Surface Length (N.D.)', p0_contnd_low, fit_time_min, fit_time_max, step_size, bounds_contnd_low, estim_init_x=True)

indiv_logistic_df_low_growth_nn_angle = fit_logistic_to_data(low_growth_angle_nn, 'Time (minutes)', 'Average Angle (deg)', p0_angle_low, fit_time_min, fit_time_max, step_size, bounds_angle_low, estim_init_x=True)
indiv_logistic_df_low_growth_nn_contnd = fit_logistic_to_data(low_growth_contnd_nn, 'Time (minutes)', 'Contact Surface Length (N.D.)', p0_contnd_low, fit_time_min, fit_time_max, step_size, bounds_contnd_low, estim_init_x=True)

indiv_logistic_df_low_growth_em_angle = fit_logistic_to_data(low_growth_angle_em, 'Time (minutes)', 'Average Angle (deg)', p0_angle_low, fit_time_min, fit_time_max, step_size, bounds_angle_low, estim_init_x=True)
indiv_logistic_df_low_growth_em_contnd = fit_logistic_to_data(low_growth_contnd_em, 'Time (minutes)', 'Contact Surface Length (N.D.)', p0_contnd_low, fit_time_min, fit_time_max, step_size, bounds_contnd_low, estim_init_x=True)

indiv_logistic_df_low_growth_eedb_angle = fit_logistic_to_data(low_growth_angle_eedb, 'Time (minutes)', 'Average Angle (deg)', p0_angle_low, fit_time_min, fit_time_max, step_size, bounds_angle_low, estim_init_x=True)
indiv_logistic_df_low_growth_eedb_contnd = fit_logistic_to_data(low_growth_contnd_eedb, 'Time (minutes)', 'Contact Surface Length (N.D.)', p0_contnd_low, fit_time_min, fit_time_max, step_size, bounds_contnd_low, estim_init_x=True)

indiv_logistic_df_low_growth_eeCcadMO_angle = fit_logistic_to_data(low_growth_angle_eeCcadMO, 'Time (minutes)', 'Average Angle (deg)', p0_angle_low, fit_time_min, fit_time_max, step_size, bounds_angle_low, estim_init_x=True)
indiv_logistic_df_low_growth_eeCcadMO_contnd = fit_logistic_to_data(low_growth_contnd_eeCcadMO, 'Time (minutes)', 'Contact Surface Length (N.D.)', p0_contnd_low, fit_time_min, fit_time_max, step_size, bounds_contnd_low, estim_init_x=True)

indiv_logistic_df_low_growth_eeRcadMO_angle = fit_logistic_to_data(low_growth_angle_eeRcadMO, 'Time (minutes)', 'Average Angle (deg)', p0_angle_low, fit_time_min, fit_time_max, step_size, bounds_angle_low, estim_init_x=True)
indiv_logistic_df_low_growth_eeRcadMO_contnd = fit_logistic_to_data(low_growth_contnd_eeRcadMO, 'Time (minutes)', 'Contact Surface Length (N.D.)', p0_contnd_low, fit_time_min, fit_time_max, step_size, bounds_contnd_low, estim_init_x=True)

indiv_logistic_df_low_growth_eeRCcadMO_angle = fit_logistic_to_data(low_growth_angle_eeRCcadMO, 'Time (minutes)', 'Average Angle (deg)', p0_angle_low, fit_time_min, fit_time_max, step_size, bounds_angle_low, estim_init_x=True)
indiv_logistic_df_low_growth_eeRCcadMO_contnd = fit_logistic_to_data(low_growth_contnd_eeRCcadMO, 'Time (minutes)', 'Contact Surface Length (N.D.)', p0_contnd_low, fit_time_min, fit_time_max, step_size, bounds_contnd_low, estim_init_x=True)



indiv_logistic_all_angle = pd.concat([indiv_logistic_df_low_growth_ee_angle, 
                                      indiv_logistic_df_low_growth_mm_angle, 
                                      indiv_logistic_df_low_growth_nn_angle, 
                                      indiv_logistic_df_low_growth_em_angle, 
                                      indiv_logistic_df_low_growth_eedb_angle, 
                                      indiv_logistic_df_low_growth_eeCcadMO_angle, 
                                      indiv_logistic_df_low_growth_eeRcadMO_angle,
                                      indiv_logistic_df_low_growth_eeRCcadMO_angle]).reset_index()

indiv_logistic_all_contnd = pd.concat([indiv_logistic_df_low_growth_ee_contnd, 
                                       indiv_logistic_df_low_growth_mm_contnd, 
                                       indiv_logistic_df_low_growth_nn_contnd, 
                                       indiv_logistic_df_low_growth_em_contnd, 
                                       indiv_logistic_df_low_growth_eedb_contnd, 
                                       indiv_logistic_df_low_growth_eeCcadMO_contnd, 
                                       indiv_logistic_df_low_growth_eeRcadMO_contnd,
                                       indiv_logistic_df_low_growth_eeRCcadMO_contnd]).reset_index()



# Create master lists of optimal parameters for the low to high growth phases:
indiv_logistic_all_angle_low2high = pd.concat([ee_indiv_logistic_angle_df,
                                               mm_indiv_logistic_angle_df,
                                               nn_indiv_logistic_angle_df,
                                               #em_indiv_logistic_angle_df,
                                               #eedb_indiv_logisti_angle_df,
                                               eeCcadMO_indiv_logistic_angle_df,
                                               eeRcadMO_indiv_logistic_angle_df,
                                               eeRCcadMO_indiv_logistic_angle_df]).reset_index()
good_growth_fits_angle_low2high = pd.concat([ee_list_dropna_angle_fit_validity_df,
                                             mm_list_dropna_angle_fit_validity_df,
                                             nn_list_dropna_angle_fit_validity_df,
                                             eeCcadMO_list_dropna_angle_fit_validity_df,
                                             eeRcadMO_list_dropna_angle_fit_validity_df,
                                             eeRCcadMO_list_dropna_angle_fit_validity_df]).reset_index()


indiv_logistic_all_contnd_low2high = pd.concat([ee_indiv_logistic_contnd_df,
                                                mm_indiv_logistic_contnd_df,
                                                nn_indiv_logistic_contnd_df,
                                                #em_indiv_logistic_contnd_df,
                                                #eedb_indiv_logisti_contnd_df,
                                                eeCcadMO_indiv_logistic_contnd_df,
                                                eeRcadMO_indiv_logistic_contnd_df,
                                                eeRCcadMO_indiv_logistic_contnd_df]).reset_index()
good_growth_fits_contnd_low2high = pd.concat([ee_list_dropna_fit_validity_df,
                                              mm_list_dropna_fit_validity_df,
                                              nn_list_dropna_fit_validity_df,
                                              eeCcadMO_list_dropna_fit_validity_df,
                                              eeRcadMO_list_dropna_fit_validity_df,
                                              eeRCcadMO_list_dropna_fit_validity_df])


###############################################################################

#list_of_df = low_growth_contnd_ee
#list_of_assoc_fits = indiv_logistic_df_low_growth_ee_contnd
#i = 0
#y_fit = logistic(x_fit, list_of_assoc_fits.iloc[i]['popt_x2'], list_of_assoc_fits.iloc[i]['popt_k2'], list_of_assoc_fits.iloc[i]['popt_high_plateau'], list_of_assoc_fits.iloc[i]['popt_mid_plateau'])
#fig, ax = plt.subplots()
#ax.scatter(x['Time (minutes)'], x['Contact Surface Length (N.D.)'], marker='.')
#ax.plot(x_fit, y_fit, c='r')
#ax.set_xlim((-50, 120))
#ax.set_ylim((-0.1, 2.1))
#plt.show()
###############################################################################

### Plot the growth to low plateau data with the corresponding logistic fits:
def plot_and_save_logistic_fits_on_contnd(list_of_df, list_of_assoc_fits, output_filename):
    """ Be sure to call plt.clf() before using this function so that you don't
    get unwanted/unexpected plots being saved in the pdf file with the others."""
    y_max = max([df['Contact Surface Length (N.D.)'].max() for df in list_of_df])
    
    y_fit_list = list()
    for i,x in enumerate(list_of_df):
        y_fit = logistic(x_fit, list_of_assoc_fits.iloc[i]['popt_x2'], list_of_assoc_fits.iloc[i]['popt_k2'], list_of_assoc_fits.iloc[i]['popt_high_plateau'], list_of_assoc_fits.iloc[i]['popt_mid_plateau'])
        fig, ax = plt.subplots()
        ax.scatter(x['Time (minutes)'], x['Contact Surface Length (N.D.)'], marker='.')
        ax.plot(x['Time (minutes)'], x['Contact Surface Length (N.D.)'], c='b', ls='-', alpha=0.5)
        ax.plot(x_fit, y_fit, c='r')
        ax.axhline(list_of_assoc_fits.iloc[i]['bounds'][0][3], c='k')
        ax.axhline(list_of_assoc_fits.iloc[i]['bounds'][0][2], c='k')
        ax.axhline(list_of_assoc_fits.iloc[i]['bounds'][1][3], ls=':', c='k')
        ax.axhline(list_of_assoc_fits.iloc[i]['bounds'][1][2], ls=':', c='k')
        ax.set_xlim((x['Time (minutes)'].min()-5,x['Time (minutes)'].max()+5))
        ax.set_ylim((-0.05*y_max, y_max + y_max*0.05))
        ax.set_title('No.%d_'%i + x['Source_File'].iloc[0], fontdict={'size':10})
        ax.text(0.01, 0.99, 
                '[popt_x2 , popt_k2 , popt_high_plateau , popt_mid_plateau]',
                horizontalalignment='left', verticalalignment='top',
                transform=ax.transAxes)
        ax.text(0.01, 0.96,
                str([np.format_float_scientific(x, 3) for x in 
                     [list_of_assoc_fits.iloc[i]['popt_x2'], 
                      list_of_assoc_fits.iloc[i]['popt_k2'], 
                      list_of_assoc_fits.iloc[i]['popt_high_plateau'], 
                      list_of_assoc_fits.iloc[i]['popt_mid_plateau']]]), 
                horizontalalignment='left', verticalalignment='top',
                transform=ax.transAxes)
        plt.draw()
        #plt.show()
        y_fit_list.append(y_fit)
    multipdf(Path.joinpath(output_dir, r"%s.pdf"%output_filename))
    fig.clf()

    return



def plot_and_save_logistic_fits_on_angles(list_of_df, list_of_assoc_fits, output_filename):
    """ Be sure to call plt.clf() before using this function so that you don't
    get unwanted/unexpected plots being saved in the pdf file with the others."""
    y_max = max([df['Average Angle (deg)'].max() for df in list_of_df])
    
    y_fit_list = list()
    for i,x in enumerate(list_of_df):
        y_fit = logistic(x_fit, list_of_assoc_fits.iloc[i]['popt_x2'], list_of_assoc_fits.iloc[i]['popt_k2'], list_of_assoc_fits.iloc[i]['popt_high_plateau'], list_of_assoc_fits.iloc[i]['popt_mid_plateau'])
        fig, ax = plt.subplots()
        ax.scatter(x['Time (minutes)'], x['Average Angle (deg)'], marker='.', c='b')
        ax.plot(x['Time (minutes)'], x['Average Angle (deg)'], c='b', ls='-', alpha=0.5)
        ax.plot(x_fit, y_fit, c='r')
        ax.axhline(list_of_assoc_fits.iloc[i]['bounds'][0][3], c='k')
        ax.axhline(list_of_assoc_fits.iloc[i]['bounds'][0][2], c='k')
        ax.axhline(list_of_assoc_fits.iloc[i]['bounds'][1][3], ls=':', c='k')
        ax.axhline(list_of_assoc_fits.iloc[i]['bounds'][1][2], ls=':', c='k')
        ax.set_xlim((x['Time (minutes)'].min()-5,x['Time (minutes)'].max()+5))
        ax.set_ylim((-0.05*y_max, y_max + y_max*0.05))
        ax.set_title('No.%d_'%i + x['Source_File'].iloc[0], fontdict={'size':10})
        ax.text(0.01, 0.99, 
                '[popt_x2 , popt_k2 , popt_high_plateau , popt_mid_plateau]',
                horizontalalignment='left', verticalalignment='top',
                transform=ax.transAxes)
        ax.text(0.01, 0.96,
                str([np.format_float_scientific(x, 3) for x in 
                     [list_of_assoc_fits.iloc[i]['popt_x2'], 
                      list_of_assoc_fits.iloc[i]['popt_k2'], 
                      list_of_assoc_fits.iloc[i]['popt_high_plateau'], 
                      list_of_assoc_fits.iloc[i]['popt_mid_plateau']]]), 
                horizontalalignment='left', verticalalignment='top',
                transform=ax.transAxes)
        plt.draw()
        #plt.show()
        y_fit_list.append(y_fit)
    multipdf(Path.joinpath(output_dir, r"%s.pdf"%output_filename))
    fig.clf()

    return




plt.close('all')
plot_and_save_logistic_fits_on_contnd(low_growth_contnd_ee, 
                                      indiv_logistic_df_low_growth_ee_contnd, 
                                      r"ee_contnd_low_growth_fits")
plt.close('all')
plot_and_save_logistic_fits_on_angles(low_growth_angle_ee, 
                                      indiv_logistic_df_low_growth_ee_angle, 
                                      r"ee_angle_low_growth_fits")

### We also want some individual examples of good ee_contnd_low_growth_fits:
# multipage PDF containing all individual examples:
plt.close('all')
for df in ee_low_adh_contnd:
    fig, ax = plt.subplots(figsize=(width_1col,height_1col))
    ax.plot(df['Time (minutes)'], df['Contact Surface Length (px)']/df['first_valid_radius'], c='lightgrey')
    ax.scatter(df['Time (minutes)'], df['Contact Surface Length (px)']/df['first_valid_radius'], marker='.', c='k', zorder=10)
    ax.set_title(np.unique(df['Source_File']), fontdict={'fontsize':6})
    ax.set_ylim(-0.05,2.0)
    ax.set_ylabel('Cont. Diam. (N.D.)')
    ax.set_xlabel('Time (minutes)')
    plt.tight_layout()
multipdf(Path.joinpath(output_dir, r"ee_low_adh_contnd.pdf"), tight=True, dpi=dpi_graph)

plt.close('all')
for df in ee_low_adh_angle:
    fig, ax = plt.subplots(figsize=(width_1col,height_1col))
    ax.plot(df['Time (minutes)'], df['Average Angle (deg)'], c='lightgrey')
    ax.scatter(df['Time (minutes)'], df['Average Angle (deg)'], marker='.', c='k', zorder=10)
    ax.set_title(np.unique(df['Source_File']), fontdict={'fontsize':6})
    ax.set_ylim(-1,180)
    ax.set_ylabel('Cont. Angle (deg)')
    ax.set_xlabel('Time (minutes)')
    plt.tight_layout()
multipdf(Path.joinpath(output_dir, r"ee_low_adh_angle.pdf"), tight=True, dpi=dpi_graph)
plt.close('all')

# Source_Files for nice growth profile:
#2018-08-10 xl wt-ecto-fdx wt-ecto-rdx 21C crop3.csv
file_index = np.where([np.any('2018-08-10 xl wt-ecto-fdx wt-ecto-rdx 21C crop3.csv' == df['Source_File']) for df in ee_low_adh_contnd])[0].astype(int)

fig, ax = plt.subplots(figsize=(width_1col,height_1col))
[ax.plot(ee_low_adh_contnd[i]['Time (minutes)'], ee_low_adh_contnd[i]['Contact Surface Length (px)']/ee_low_adh_contnd[i]['first_valid_radius'], c='lightgrey') for i in file_index]
[ax.scatter(ee_low_adh_contnd[i]['Time (minutes)'], ee_low_adh_contnd[i]['Contact Surface Length (px)']/ee_low_adh_contnd[i]['first_valid_radius'], marker='.', c='k', zorder=10) for i in file_index]
ax.set_ylim(0,2.0)
ax.draw
fig.savefig(Path.joinpath(output_dir, r'ee_nice_low_adh_example.tif'), dpi=dpi_graph)
plt.close(fig)

# frequent attach-detach low plateau:
#2018-06-28 xl wt-ecto-fdx wt-ecto-rdx 19C crop6.csv
file_index = np.where([np.any('2018-06-28 xl wt-ecto-fdx wt-ecto-rdx 19C crop6.csv' == df['Source_File']) for df in ee_low_adh_contnd])[0].astype(int)

fig, ax = plt.subplots(figsize=(width_1col,height_1col))
[ax.plot(ee_low_adh_contnd[i]['Time (minutes)'], ee_low_adh_contnd[i]['Contact Surface Length (px)']/ee_low_adh_contnd[i]['first_valid_radius'], c='lightgrey') for i in file_index]
[ax.scatter(ee_low_adh_contnd[i]['Time (minutes)'], ee_low_adh_contnd[i]['Contact Surface Length (px)']/ee_low_adh_contnd[i]['first_valid_radius'], marker='.', c='k', zorder=10) for i in file_index]
ax.set_ylim(0,2.0)
ax.draw
fig.savefig(Path.joinpath(output_dir, r'ee_unstable_low_adh_example.tif'), dpi=dpi_graph)
plt.close(fig)

# instant:
#2018-08-10 xl wt-ecto-fdx wt-ecto-rdx 21C crop7
file_index = np.where([np.any('2018-08-10 xl wt-ecto-fdx wt-ecto-rdx 21C crop7.csv' == df['Source_File']) for df in ee_low_adh_contnd])[0].astype(int)

fig, ax = plt.subplots(figsize=(width_1col,height_1col))
[ax.plot(ee_low_adh_contnd[i]['Time (minutes)'], ee_low_adh_contnd[i]['Contact Surface Length (px)']/ee_low_adh_contnd[i]['first_valid_radius'], c='lightgrey') for i in file_index]
[ax.scatter(ee_low_adh_contnd[i]['Time (minutes)'], ee_low_adh_contnd[i]['Contact Surface Length (px)']/ee_low_adh_contnd[i]['first_valid_radius'], marker='.', c='k', zorder=10) for i in file_index]
ax.set_ylim(0,2.0)
ax.draw
fig.savefig(Path.joinpath(output_dir, r'ee_instant_low_adh_example.tif'), dpi=dpi_graph)
plt.close(fig)



plt.close('all')
plot_and_save_logistic_fits_on_contnd(low_growth_contnd_mm, 
                                      indiv_logistic_df_low_growth_mm_contnd, 
                                      r"mm_contnd_low_growth_fits")
plt.close('all')
plot_and_save_logistic_fits_on_angles(low_growth_angle_mm, 
                                      indiv_logistic_df_low_growth_mm_angle, 
                                      r"mm_angle_low_growth_fits")

plt.close('all')
plot_and_save_logistic_fits_on_contnd(low_growth_contnd_nn, 
                                      indiv_logistic_df_low_growth_nn_contnd, 
                                      r"nn_contnd_low_growth_fits")
plt.close('all')
plot_and_save_logistic_fits_on_angles(low_growth_angle_nn, 
                                      indiv_logistic_df_low_growth_nn_angle, 
                                      r"nn_angle_low_growth_fits")

plt.close('all')
plot_and_save_logistic_fits_on_contnd(low_growth_contnd_em, 
                                      indiv_logistic_df_low_growth_em_contnd, 
                                      r"em_contnd_low_growth_fits")
plt.close('all')
plot_and_save_logistic_fits_on_angles(low_growth_angle_em, 
                                      indiv_logistic_df_low_growth_em_angle, 
                                      r"em_angle_low_growth_fits")

plt.close('all')
plot_and_save_logistic_fits_on_contnd(low_growth_contnd_eedb, 
                                      indiv_logistic_df_low_growth_eedb_contnd, 
                                      r"eedb_contnd_low_growth_fits")
plt.close('all')
plot_and_save_logistic_fits_on_angles(low_growth_angle_eedb, 
                                      indiv_logistic_df_low_growth_eedb_angle, 
                                      r"eedb_angle_low_growth_fits")

plt.close('all')
plot_and_save_logistic_fits_on_contnd(low_growth_contnd_eeCcadMO, 
                                      indiv_logistic_df_low_growth_eeCcadMO_contnd, 
                                      r"eeCcadMO_contnd_low_growth_fits")
plt.close('all')
plot_and_save_logistic_fits_on_angles(low_growth_angle_eeCcadMO, 
                                      indiv_logistic_df_low_growth_eeCcadMO_angle, 
                                      r"eeCcadMO_angle_low_growth_fits")

plt.close('all')
plot_and_save_logistic_fits_on_contnd(low_growth_contnd_eeRcadMO, 
                                      indiv_logistic_df_low_growth_eeRcadMO_contnd, 
                                      r"eeRcadMO_contnd_low_growth_fits")
plt.close('all')
plot_and_save_logistic_fits_on_angles(low_growth_angle_eeRcadMO, 
                                      indiv_logistic_df_low_growth_eeRcadMO_angle, 
                                      r"eeRcadMO_angle_low_growth_fits")

plt.close('all')
plot_and_save_logistic_fits_on_contnd(low_growth_contnd_eeRCcadMO, 
                                      indiv_logistic_df_low_growth_eeRCcadMO_contnd, 
                                      r"eeRCcadMO_contnd_low_growth_fits")
plt.close('all')
plot_and_save_logistic_fits_on_angles(low_growth_angle_eeRCcadMO, 
                                      indiv_logistic_df_low_growth_eeRCcadMO_angle, 
                                      r"eeRCcadMO_angle_low_growth_fits")


def cont_init_valid(list_of_list_of_dfs):
    """ Function will separate data based on if first timepoint that cells are
    in contact in the low growth kinetics corresponds to the first timepoint in
    the timelapse. This is because if the imaging of cells started off when 
    they were touching and they were already at the low plateau then it is not
    possible to measure the speed of growth to the low plateau and assuming that
    it happened in less than the time resolution of the timelapse would be 
    wrong. However in some cases the growth to the low plateau may indeed have
    been so fast that at one timepoint the cells were separate and at the next
    the cells were at the low plateau."""
    
    list_of_list_of_valid_dfs = list()
    list_of_list_of_maybe_invalid = list()
    for cell_combo in list_of_list_of_dfs:
        list_of_valid_dfs = [cell_combo[i] for i in np.ravel(np.where( [df['Cell Pair Number'].iloc[0] > 1 for df in cell_combo] ))]
        list_of_list_of_valid_dfs.append(list_of_valid_dfs)
        list_of_maybe_invalid = [cell_combo[i] for i in np.ravel(np.where( [df['Cell Pair Number'].iloc[0] == 1 for df in cell_combo] ))]
        list_of_list_of_maybe_invalid.append(list_of_maybe_invalid)
        
    return list_of_list_of_valid_dfs, list_of_list_of_maybe_invalid




low_growth_contnd_all = [low_growth_contnd_ee, low_growth_contnd_mm, low_growth_contnd_nn, 
                         low_growth_contnd_em, low_growth_contnd_eedb, 
                         low_growth_contnd_eeCcadMO, low_growth_contnd_eeRcadMO,
                         low_growth_contnd_eeRCcadMO]
contnd_init_valid_all, contnd_init_maybe_invalid_all = cont_init_valid(low_growth_contnd_all)
#low_growth_contnd_all_sf = [df[['Source_File', 'Experimental_Condition']].iloc[[0]] for list_of_dfs in low_growth_contnd_all for df in list_of_dfs]#.reset_index()
low_growth_contnd_all_sf = pd.concat([df.iloc[[0]] for list_of_dfs in low_growth_contnd_all for df in list_of_dfs])[['Source_File', 'Experimental_Condition']].reset_index()
maybe_valid_contnd = pd.concat([df.iloc[0] for list_of_dfs in contnd_init_maybe_invalid_all for df in list_of_dfs])['Source_File']

low_growth_angle_all = [low_growth_angle_ee, low_growth_angle_mm, low_growth_angle_nn, 
                         low_growth_angle_em, low_growth_angle_eedb, 
                         low_growth_angle_eeCcadMO, low_growth_angle_eeRcadMO,
                         low_growth_angle_eeRCcadMO]
angle_init_valid_all, angle_init_maybe_invalid_all = cont_init_valid(low_growth_angle_all)
#low_growth_angle_all_sf = [df[['Source_File', 'Experimental_Condition']].iloc[[0]] for list_of_dfs in low_growth_angle_all for df in list_of_dfs]#.reset_index()
low_growth_angle_all_sf = pd.concat([df.iloc[[0]] for list_of_dfs in low_growth_angle_all for df in list_of_dfs])[['Source_File', 'Experimental_Condition']].reset_index()
maybe_valid_angle = pd.concat([df.iloc[0] for list_of_dfs in angle_init_maybe_invalid_all for df in list_of_dfs])['Source_File']


### For the maybe_valid_contnd:
## Valid: These can be included still in the low_growth dataset
#maybe_valid_contnd_confirmed = ['2018-03-08 xl wt-ecto-fdx wt-ecto-rdx 20C crop3.csv',
#                                '2018-03-15 xl wt-ecto-fdx wt-ecto-rdx 20C #1 crop1.csv',
#                                '2018-03-15 xl wt-ecto-fdx wt-ecto-rdx 20C #1 crop8.csv',
#                                '2017-10-26 endo-endo crop1.csv',
#                                '2018-01-25 endo-endo crop1.csv',
#                                '2018-01-25 endo-endo crop2.csv',
#                                '2018-05-31 endo-endo crop1.csv',
#                                '2014-02-27 bgsub xl wt meso rda wt ecto fda crop 2.csv',
#                                '2018-02-27 xl CcadMO-ecto-fdx CcadMO-ecto-rdx 21C crop3.csv',
#                                '2018-02-27 xl CcadMO-ecto-fdx CcadMO-ecto-rdx 21C crop6.csv']
## Not valid: these need to be dropped from the low_growth dataset
#maybe_valid_contnd_denied = ['2018-03-15 xl wt-ecto-fdx wt-ecto-rdx 20C #1 crop11.csv',
#                             '2018-03-15 xl wt-ecto-fdx wt-ecto-rdx 20C #1 crop3.csv',
#                             '2018-03-15 xl wt-ecto-fdx wt-ecto-rdx 20C #1 crop5.csv',
#                             '2018-08-10 xl wt-ecto-fdx wt-ecto-rdx 21C crop4.csv',
#                             '2018-08-10 xl wt-ecto-fdx wt-ecto-rdx 21C crop6.csv',
#                             '2019-01-09_XL_WTectoFDX_WTectoRDX_#2_19C_dt1m crop1.csv',
#                             '2017-11-09 endo-endo crop1.csv',
#                             '2018-01-18 endo-endo crop1.csv',
#                             '2018-07-11 endo-endo crop2.csv',
#                             '2014-03-07 xl wt ecto-fda wt meso-rda 1 min interv crop4.csv',
#                             '2018-01-25 xl wt-ecto-fdx wt-ecto-rdx 1X DB 20C crop3.csv',
#                             '2018-06-22 xl wt-ecto-fdx wt-ecto-rdx in 1Xdb 19C crop5.csv',
#                             '2018-07-11 xl RcadMO-ecto-fdx RcadMO-ecto-rdx 19C crop 2.csv',
#                             '2018-07-19 xl RcadMO-ecto-fdx RcadMO-ecto-rdx 19C crop 1.csv']

### For the maybe_valid_angle:
## Valid: These can be included still in the low_growth dataset
#maybe_valid_angle_confirmed = ['2018-03-15 xl wt-ecto-fdx wt-ecto-rdx 20C #1 crop1.csv',
#                               '2018-01-18 endo-endo crop1.csv',
#                               '2018-01-25 endo-endo crop1.csv',
#                               '2018-01-25 endo-endo crop2.csv',
#                               '2018-05-31 endo-endo crop1.csv',
#                               '2018-01-25 xl wt-ecto-fdx wt-ecto-rdx 1X DB 20C crop3.csv',
#                               '2018-02-27 xl CcadMO-ecto-fdx CcadMO-ecto-rdx 21C crop2.csv',
#                               '2018-02-27 xl CcadMO-ecto-fdx CcadMO-ecto-rdx 21C crop3.csv',
#                               '2018-02-27 xl CcadMO-ecto-fdx CcadMO-ecto-rdx 21C crop6.csv',
#                               '2018-07-11 xl RcadMO-ecto-fdx RcadMO-ecto-rdx 19C crop 2.csv']
## Not valid: these need to be dropped from the low_growth dataset
#maybe_valid_angle_denied = ['2018-03-08 xl wt-ecto-fdx wt-ecto-rdx 20C crop3.csv',
#                            '2018-03-15 xl wt-ecto-fdx wt-ecto-rdx 20C #1 crop11.csv',
#                            '2018-03-15 xl wt-ecto-fdx wt-ecto-rdx 20C #1 crop2.csv',
#                            '2018-03-15 xl wt-ecto-fdx wt-ecto-rdx 20C #1 crop5.csv',
#                            '2018-08-10 xl wt-ecto-fdx wt-ecto-rdx 21C crop6.csv',
#                            '2019-01-09_XL_WTectoFDX_WTectoRDX_#2_19C_dt1m crop1.csv',
#                            '2016 Dec 8 xl wt meso-rdx wt meso-fdx 19C #2 crop2 bgsub.csv',
#                            '2017-10-26 endo-endo crop1.csv',
#                            '2017-11-09 endo-endo crop1.csv',
#                            '2018-07-11 endo-endo crop2.csv',
#                            '2014-02-27 bgsub xl wt meso rda wt ecto fda crop 2.csv',
#                            '2014-03-07 xl wt ecto-fda wt meso-rda 1 min interv crop4.csv',
#                            '2018-07-19 xl RcadMO-ecto-fdx RcadMO-ecto-rdx 19C crop 1.csv']

### The above lists have now been turned into columns in .csv files.
# Use these lists of valid and not valid timelapses in combination with lists 
# of good or poor data fits to remove all timelapses that are no good:
low_fit_angle_validity = pd.read_csv(Path.joinpath(grow_to_low_fit_valid_dir, r"grow_to_low_angle_fit_validity.csv"))
low_fit_contnd_validity = pd.read_csv(Path.joinpath(grow_to_low_fit_valid_dir, r"grow_to_low_contnd_fit_validity.csv"))
# Note that for the above tables grow_to_low_angle_fit_validity.csv and 
# grow_to_low_contnd_fit_validity.csv, the filenumber should match up with
# low_growth_angle_all_sf and low_growth_contnd_all_sf, respectively.

inadmissible = low_fit_angle_validity[['1st_timeframe_contact?', '1st_timeframe_contact_valid?']] == [True, False]
inadmissible_low_growth_angle = low_fit_angle_validity.iloc[inadmissible[inadmissible.all(axis=1)].index]['Source_File']

inadmissible = low_fit_contnd_validity[['1st_timeframe_contact?', '1st_timeframe_contact_valid?']] == [True, False]
inadmissible_low_growth_contnd = low_fit_contnd_validity.iloc[inadmissible[inadmissible.all(axis=1)].index]['Source_File']

# Take the indices of low_growth_angle_all that contain files that are bad:
#low_fit_indices_angle_bad = pd.concat([low_growth_angle_all_sf[(low_growth_angle_all_sf['Source_File']==sf)] for sf in inadmissible_low_growth_angle]).index
low_growth_angle_all_sf_bad = low_growth_angle_all_sf[np.logical_or.reduce([low_growth_angle_all_sf['Source_File']==sf for sf in inadmissible_low_growth_angle])]['Source_File']
low_growth_angle_all_sf_good = low_growth_angle_all_sf[~np.logical_or.reduce([low_growth_angle_all_sf['Source_File']==sf for sf in inadmissible_low_growth_angle])]['Source_File']
#low_fit_indices_contnd_bad = pd.concat([low_growth_contnd_all_sf[(low_growth_contnd_all_sf['Source_File']==sf)] for sf in inadmissible_low_growth_contnd]).index
low_growth_contnd_all_sf_bad = low_growth_contnd_all_sf[np.logical_or.reduce([low_growth_contnd_all_sf['Source_File']==sf for sf in inadmissible_low_growth_contnd])]['Source_File']
low_growth_contnd_all_sf_good = low_growth_contnd_all_sf[~np.logical_or.reduce([low_growth_contnd_all_sf['Source_File']==sf for sf in inadmissible_low_growth_contnd])]['Source_File']


# make a copy of low_growth_angle_all and low_growth_contnd_all that only 
# contains the useable data:
low_growth_angle_all_good_data = list()
for sublist in low_growth_angle_all:
    for df in sublist:
        if df['Source_File'].iloc[0] in low_growth_angle_all_sf_good.to_numpy():
            low_growth_angle_all_good_data.append(df)
            
low_growth_contnd_all_good_data = list()
for sublist in low_growth_contnd_all:
    for df in sublist:
        if df['Source_File'].iloc[0] in low_growth_contnd_all_sf_good.to_numpy():
            low_growth_contnd_all_good_data.append(df)

# Separate the useable data into data with good fits to the growth curves, and
# data where cells jumped from being separate to being at the low plateau:
# Data with good growth fits:
admissible_fits = low_fit_angle_validity[['growth_data?', 'growth_fit_valid?']] == [True, True]
good_growth_fits_angle = low_fit_angle_validity.iloc[admissible_fits[admissible_fits.all(axis=1)].index]['Source_File']
low_growth_angle_all_growth_fit = list()
for df in low_growth_angle_all_good_data:
    if df['Source_File'].iloc[0] in good_growth_fits_angle.to_numpy():
        low_growth_angle_all_growth_fit.append(df)

admissible_fits = low_fit_contnd_validity[['growth_data?', 'growth_fit_valid?']] == [True, True]
good_growth_fits_contnd = low_fit_contnd_validity.iloc[admissible_fits[admissible_fits.all(axis=1)].index]['Source_File']
low_growth_contnd_all_growth_fit = list()
for df in low_growth_contnd_all_good_data:
    if df['Source_File'].iloc[0] in good_growth_fits_contnd.to_numpy():
        low_growth_contnd_all_growth_fit.append(df)

# Data where cells go from separate to low_plateau faster than the time 
# resolution of the timelapse:
admissible_init_cont = low_fit_angle_validity[['growth_data?', 'growth_fit_valid?', '1st_timeframe_contact_valid?']] == [False, False, True]
good_init_cont_angle = low_fit_angle_validity.iloc[admissible_init_cont[admissible_init_cont.all(axis=1)].index]['Source_File']
low_growth_angle_all_instant_low_plat = list()
for df in low_growth_angle_all_good_data:
    if df['Source_File'].iloc[0] in good_init_cont_angle.to_numpy():
        low_growth_angle_all_instant_low_plat.append(df)

admissible_init_cont = low_fit_contnd_validity[['growth_data?', 'growth_fit_valid?', '1st_timeframe_contact_valid?']] == [False, False, True]
good_init_cont_contnd = low_fit_contnd_validity.iloc[admissible_init_cont[admissible_init_cont.all(axis=1)].index]['Source_File']
low_growth_contnd_all_instant_low_plat = list()
for df in low_growth_contnd_all_good_data:
    if df['Source_File'].iloc[0] in good_init_cont_contnd.to_numpy():
        low_growth_contnd_all_instant_low_plat.append(df)



# Get the slowest average growth rates of the instant low plat data:
# N.B. that these growth rates are simple growth rates (eg. contact length/min)
master_list = ee_list + mm_list + nn_list + em_list + eedb_list + eeCcadMO_list + eeRcadMO_list + eeRCcadMO_list
low_growth_t_interv = [(np.diff(df['Time (minutes)']).min(), df['Source_File'].iloc[0], df['Experimental_Condition'].iloc[0]) for df in master_list]
low_growth_t_interv_df = pd.DataFrame(data=low_growth_t_interv, columns=('t_int', 'Source_File', 'Experimental_Condition'))
low_growth_instant_low_plat_t_interv_angle_df = pd.concat([low_growth_t_interv_df[low_growth_t_interv_df['Source_File'] == sf] for sf in good_init_cont_angle])
low_growth_instant_low_plat_t_interv_contnd_df = pd.concat([low_growth_t_interv_df[low_growth_t_interv_df['Source_File'] == sf] for sf in good_init_cont_contnd])


# Give me the minimum average growth rate based on the fitted popt_high_plateau
# parameter of the low plateau data:
# Note that the correct fitted parameter to ask for is the popt_high_plateau 
# since we fitted curves to the growth to the low plateau, therefore the 
# parameter popt_low_plateau is equal to or less than 0 (the cells start
# off separated, therefore they cannot have a positive popt_low_plateau)
slowest_instant_t_inst_angle = pd.concat([indiv_logistic_all_angle[['popt_high_plateau', 'Source_File']][indiv_logistic_all_angle['Source_File'] == sf] for sf in low_growth_instant_low_plat_t_interv_angle_df['Source_File']])
slowest_instant_growth_rates_angle_curvefit = pd.merge(low_growth_instant_low_plat_t_interv_angle_df, slowest_instant_t_inst_angle, on='Source_File')
slowest_instant_growth_rates_angle_curvefit['avg_growth_rate'] = slowest_instant_growth_rates_angle_curvefit['popt_high_plateau'] / slowest_instant_growth_rates_angle_curvefit['t_int']

slowest_instant_t_inst_contnd = pd.concat([indiv_logistic_all_contnd[['popt_high_plateau', 'Source_File']][indiv_logistic_all_contnd['Source_File'] == sf] for sf in low_growth_instant_low_plat_t_interv_contnd_df['Source_File']])
slowest_instant_growth_rates_contnd_curvefit = pd.merge(low_growth_instant_low_plat_t_interv_contnd_df, slowest_instant_t_inst_contnd, on='Source_File')
slowest_instant_growth_rates_contnd_curvefit['avg_growth_rate'] = slowest_instant_growth_rates_contnd_curvefit['popt_high_plateau'] / slowest_instant_growth_rates_contnd_curvefit['t_int']


# Give me the minimum average growth rate based on the mean of the low plateau
# data:
instant_low_plat_mean_angle = [(df['Average Angle (deg)'].mean(), df['Source_File'].iloc[0]) for df in low_growth_angle_all_instant_low_plat]
instant_low_plat_mean_angle_df = pd.DataFrame(data=instant_low_plat_mean_angle, columns=('Mean Angle (deg)', 'Source_File'))
slowest_instant_growth_rates_angle_mean = pd.merge(low_growth_instant_low_plat_t_interv_angle_df, instant_low_plat_mean_angle_df, on='Source_File')
slowest_instant_growth_rates_angle_mean['avg_growth_rate'] = slowest_instant_growth_rates_angle_mean['Mean Angle (deg)'] / slowest_instant_growth_rates_angle_mean['t_int']

instant_low_plat_mean_contnd = [(df['Contact Surface Length (N.D.)'].mean(), df['Source_File'].iloc[0]) for df in low_growth_contnd_all_instant_low_plat]
instant_low_plat_mean_contnd_df = pd.DataFrame(data=instant_low_plat_mean_contnd, columns=('Mean Contact Surface Length (N.D.)', 'Source_File'))
slowest_instant_growth_rates_contnd_mean = pd.merge(low_growth_instant_low_plat_t_interv_contnd_df, instant_low_plat_mean_contnd_df, on='Source_File')
slowest_instant_growth_rates_contnd_mean['avg_growth_rate'] = slowest_instant_growth_rates_contnd_mean['Mean Contact Surface Length (N.D.)'] / slowest_instant_growth_rates_contnd_mean['t_int']



# Get the low growth rates of the data with good fits to growth data:
# N.B. these growth rates are equivalent to the logistic growth rate 'k'
# in other words they are growth rates per capita.
logistic_low_growth_rates_angle = pd.concat([indiv_logistic_all_angle[['logistic_y', 'logistic_x', 'popt_x2', 'popt_k2', 'popt_high_plateau', 'popt_mid_plateau', 'Source_File']][indiv_logistic_all_angle['Source_File'] == sf] for sf in good_growth_fits_angle])
# logistic_low_growth_rates_angle = pd.concat([indiv_logistic_all_angle[['popt_k2', 'Source_File']][indiv_logistic_all_angle['Source_File'] == sf] for sf in good_growth_fits_angle])
logistic_low_growth_rates_angle_t_int = pd.concat([low_growth_t_interv_df[low_growth_t_interv_df['Source_File'] == sf] for sf in good_growth_fits_angle])
logistic_low_growth_rates_angle = pd.merge(logistic_low_growth_rates_angle, logistic_low_growth_rates_angle_t_int, on='Source_File')

logistic_low_growth_rates_contnd = pd.concat([indiv_logistic_all_contnd[['logistic_y', 'logistic_x', 'popt_x2', 'popt_k2', 'popt_high_plateau', 'popt_mid_plateau', 'Source_File']][indiv_logistic_all_contnd['Source_File'] == sf] for sf in good_growth_fits_contnd])
# logistic_low_growth_rates_contnd = pd.concat([indiv_logistic_all_contnd[['popt_k2', 'Source_File']][indiv_logistic_all_contnd['Source_File'] == sf] for sf in good_growth_fits_contnd])
logistic_low_growth_rates_contnd_t_int = pd.concat([low_growth_t_interv_df[low_growth_t_interv_df['Source_File'] == sf] for sf in good_growth_fits_contnd])
logistic_low_growth_rates_contnd = pd.merge(logistic_low_growth_rates_contnd, logistic_low_growth_rates_contnd_t_int, on='Source_File')


### Repeat for the low_plat-to-high_plat growth rates:
growth_t_interv = [(np.diff(df['Time (minutes)']).min(), df['Source_File'].iloc[0], df['Experimental_Condition'].iloc[0]) for df in master_list]
growth_t_interv_df = pd.DataFrame(data=growth_t_interv, columns=('t_int', 'Source_File', 'Experimental_Condition'))

good_growth = good_growth_fits_angle_low2high[good_growth_fits_angle_low2high['growth_phase'] == True]['Source_File']
logistic_highplat_growth_rates_angle = pd.concat([indiv_logistic_all_angle_low2high[['logistic_y', 'logistic_x', 'popt_x2', 'popt_k2', 'popt_high_plateau', 'popt_mid_plateau', 'Source_File']][indiv_logistic_all_angle_low2high['Source_File'] == sf] for sf in good_growth])
# logistic_highplat_growth_rates_angle = pd.concat([indiv_logistic_all_angle_low2high[['popt_k2', 'Source_File']][indiv_logistic_all_angle_low2high['Source_File'] == sf] for sf in good_growth])
logistic_highplat_growth_rates_angle_t_int = pd.concat([growth_t_interv_df[growth_t_interv_df['Source_File'] == sf] for sf in good_growth])
logistic_highplat_growth_rates_angle = pd.merge(logistic_highplat_growth_rates_angle, logistic_highplat_growth_rates_angle_t_int, on='Source_File')

good_growth = good_growth_fits_contnd_low2high[good_growth_fits_contnd_low2high['growth_phase'] == True]['Source_File']
logistic_highplat_growth_rates_contnd = pd.concat([indiv_logistic_all_contnd_low2high[['logistic_y', 'logistic_x', 'popt_x2', 'popt_k2', 'popt_high_plateau', 'popt_mid_plateau', 'Source_File']][indiv_logistic_all_contnd_low2high['Source_File'] == sf] for sf in good_growth])
# logistic_highplat_growth_rates_contnd = pd.concat([indiv_logistic_all_contnd_low2high[['popt_k2', 'Source_File']][indiv_logistic_all_contnd_low2high['Source_File'] == sf] for sf in good_growth])
logistic_highplat_growth_rates_contnd_t_int = pd.concat([growth_t_interv_df[growth_t_interv_df['Source_File'] == sf] for sf in good_growth])
logistic_highplat_growth_rates_contnd = pd.merge(logistic_highplat_growth_rates_contnd, logistic_highplat_growth_rates_contnd_t_int, on='Source_File')


### Finally we can start plotting some low plateau growth rates:
graph_angle_low_logistic_growth_rate = sns.catplot(x='Experimental_Condition', y='popt_k2', data=logistic_low_growth_rates_angle, hue='t_int', palette=['grey', 'k'])
graph_angle_low_logistic_growth_rate.set_xticklabels(ha='right', rotation='45')
graph_angle_low_logistic_growth_rate.fig.suptitle('angle_low_popt_k2')
graph_angle_low_logistic_growth_rate.savefig(Path.joinpath(output_dir, r"low_grow_speed_angle_low_logistic_growth_rate"), dpi=dpi_graph)

graph_contnd_low_logistic_growth_rate = sns.catplot(x='Experimental_Condition', y='popt_k2', data=logistic_low_growth_rates_contnd, hue='t_int', palette=['grey', 'k'])
graph_contnd_low_logistic_growth_rate.set_xticklabels(ha='right', rotation='45')
graph_contnd_low_logistic_growth_rate.fig.suptitle('angle_low_popt_k2')
graph_contnd_low_logistic_growth_rate.savefig(Path.joinpath(output_dir, r"low_grow_speed_contnd_low_logistic_growth_rate"), dpi=dpi_graph)


graph_angle_low_fit = sns.catplot(x='Experimental_Condition', y='avg_growth_rate', data=slowest_instant_growth_rates_angle_curvefit, hue='t_int', palette=['grey', 'k'])
graph_angle_low_fit.set_xticklabels(ha='right', rotation='45')
graph_angle_low_fit.fig.suptitle('angle_low_fit')
graph_angle_low_fit.savefig(Path.joinpath(output_dir, r"low_grow_speed_angle_low_fit"), dpi=dpi_graph)

graph_contnd_low_fit = sns.catplot(x='Experimental_Condition', y='avg_growth_rate', data=slowest_instant_growth_rates_contnd_curvefit, hue='t_int', palette=['grey', 'k'])
graph_contnd_low_fit.set_xticklabels(ha='right', rotation='45')
graph_contnd_low_fit.fig.suptitle('contnd_low_fit')
graph_contnd_low_fit.savefig(Path.joinpath(output_dir, r"low_grow_speed_contnd_low_fit"), dpi=dpi_graph)


graph_angle_mean = sns.catplot(x='Experimental_Condition', y='avg_growth_rate', data=slowest_instant_growth_rates_angle_mean, hue='t_int', palette=['grey', 'k'])
graph_angle_mean.set_xticklabels(ha='right', rotation='45')
graph_angle_mean.fig.suptitle('angle_mean')
graph_angle_mean.savefig(Path.joinpath(output_dir, r"low_grow_speed_angle_mean"), dpi=dpi_graph)

graph_contnd_mean = sns.catplot(x='Experimental_Condition', y='avg_growth_rate', data=slowest_instant_growth_rates_contnd_mean, hue='t_int', palette=['grey', 'k'])
graph_contnd_mean.set_xticklabels(ha='right', rotation='45')
graph_contnd_mean.fig.suptitle('contnd_mean')
graph_contnd_mean.savefig(Path.joinpath(output_dir, r"low_grow_speed_contnd_mean"), dpi=dpi_graph)




#### The following is to find the smallest popt_k2 for a given
# instant_low_grow timelapse (Please note that it underestimates the logistic
# growth rate, so the growth rates 'k' are faster than these numbers suggest):
def fit_logistic_to_piecewise_instant_low_plat(df, x, target_growth_dur_col_label, upper_plat_col_label, k_init=1, lin_growth=True):
    """Returns df but with an extra column containing the growth rate 'k' of 
    the fitted logistic functions."""
    
    # popt_k_list = list()
    y_fit_list = list()
    x_fit_list = list()
    y_sim_list = list()
    x_sim_list = list()
    popt_list = list()
    for i in range(len(df)):
        target_growth_duration = df.iloc[i][target_growth_dur_col_label] 
        
        x_sim = x.copy()
        # x_sim = np.arange(-240, 241, step=0.01)
        x0 = 0
        lower_plateau = 0 # -1*df.iloc[i][upper_plat_col_label] # 0
        upper_plateau = df.iloc[i][upper_plat_col_label]
        
        # Create piecewise function of upper and lower plateau with linear 
        # growth inbetween:
        growth_start = 0 - (target_growth_duration/2)
        growth_end = 0 + (target_growth_duration/2)
        y_sim = np.zeros(x.shape)
        y_sim[x <= growth_start] = lower_plateau
        y_sim[x >= growth_end] = upper_plateau
        growth_indices = np.logical_and((x >= growth_start), (x < growth_end))
        if lin_growth == True:
            y_sim[growth_indices] = np.linspace(lower_plateau, upper_plateau, np.count_nonzero(growth_indices))
        else:
            y_sim = y_sim[np.where(~growth_indices)]
            x_sim = x[np.where(~growth_indices)]
            
        # Fit logistic function to piecewise function:
        popt, pcov = curve_fit(logistic, x_sim, y_sim, p0=(x0, k_init, upper_plateau, lower_plateau))
        y_fit = logistic(x, *popt,)
        
        y_fit_list.append(y_fit)
        x_fit_list.append(x)
        y_sim_list.append(y_sim)
        x_sim_list.append(x_sim)
        popt_list.append(popt)
        # popt_list.append(popt)
        
        
    # df['popt_k for "%s"'%upper_plat_col_label] = popt_list
    # df[['y_sim', 'x_sim', 'popt_x2', 'popt_k2', 'popt_high_plateau', 'popt_mid_plateau'] = np.array([y_sim_list, x_sim_list , *list(zip(*popt_list))]).T
    cols_to_add = pd.DataFrame({'logistic_y':y_fit_list, 'logistic_x':x_fit_list, 
                                'popt_x2':list(zip(*popt_list))[0], 'popt_k2':list(zip(*popt_list))[1], 
                                'popt_high_plateau':list(zip(*popt_list))[2], 
                                'popt_mid_plateau':list(zip(*popt_list))[3]})
    df = pd.concat([df, cols_to_add], axis=1)
       
    return df, (x_fit_list, y_fit_list), (x_sim_list, y_sim_list)

x_fit_more_subdiv = np.arange(-120, 121, step=0.01)
# instant_low_growth_rates_angle_fitted, xy_fit_list_angle, xy_sim_list_angle = fit_logistic_to_piecewise_instant_low_plat(slowest_instant_growth_rates_angle_mean, x_fit, 't_int', 'Mean Angle (deg)')
# instant_low_growth_rates_contnd_fitted, xy_fit_list_contnd, xy_sim_list_contnd = fit_logistic_to_piecewise_instant_low_plat(slowest_instant_growth_rates_contnd_mean, x_fit, 't_int', 'Mean Contact Surface Length (N.D.)')
instant_low_growth_rates_angle_fitted, xy_fit_list_angle, xy_sim_list_angle = fit_logistic_to_piecewise_instant_low_plat(slowest_instant_growth_rates_angle_mean, x_fit_more_subdiv, 't_int', 'Mean Angle (deg)')
instant_low_growth_rates_contnd_fitted, xy_fit_list_contnd, xy_sim_list_contnd = fit_logistic_to_piecewise_instant_low_plat(slowest_instant_growth_rates_contnd_mean, x_fit_more_subdiv, 't_int', 'Mean Contact Surface Length (N.D.)')

plt.close('all')
for i in range(len(list(zip(*xy_fit_list_contnd)))):
    fig,ax = plt.subplots()
    ax.set_title(instant_low_growth_rates_contnd_fitted['Source_File'].iloc[i])
    ax.scatter(xy_sim_list_contnd[0][i], xy_sim_list_contnd[1][i], marker='.')
    ax.plot(xy_fit_list_contnd[0][i], xy_fit_list_contnd[1][i], c='k')
    ax.axhline(instant_low_growth_rates_contnd_fitted['Mean Contact Surface Length (N.D.)'].iloc[i], c='tab:orange')
    ax.set_ylim(-0.01, instant_low_growth_rates_contnd_fitted['Mean Contact Surface Length (N.D.)'].max() * 1.02)
    ax.set_xlim(-2,2)
    plt.tight_layout()
multipdf(Path.joinpath(output_dir, r"instant_low_growth_fits_contnd.pdf"), tight=True, dpi=dpi_graph)
plt.close('all')

for i in range(len(list(zip(*xy_fit_list_angle)))):
    fig,ax = plt.subplots()
    ax.set_title(instant_low_growth_rates_angle_fitted['Source_File'].iloc[i])
    ax.scatter(xy_sim_list_angle[0][i], xy_sim_list_angle[1][i], marker='.')
    ax.plot(xy_fit_list_angle[0][i], xy_fit_list_angle[1][i], c='k')
    ax.axhline(instant_low_growth_rates_angle_fitted['Mean Angle (deg)'].iloc[i], c='tab:orange')
    ax.set_ylim(-0.01, instant_low_growth_rates_angle_fitted['Mean Angle (deg)'].max() * 1.02)
    ax.set_xlim(-2,2)
    plt.tight_layout()
multipdf(Path.joinpath(output_dir, r"instant_low_growth_fits_angle.pdf"), tight=True, dpi=dpi_graph)
plt.close('all')

#instant_low_growth_rates_contnd_fitted_no_lin_growth, xy_fit_list_contnd_no_lin_growth, xy_sim_list_contnd_no_lin_growth = fit_logistic_to_piecewise_instant_low_plat(slowest_instant_growth_rates_contnd_mean, x_fit, 't_int', 'Mean Contact Surface Length (N.D.)', lin_growth=False)
#sns.distplot(instant_low_growth_rates_contnd_fitted['popt_k for "Mean Contact Surface Length (N.D.)"'])
#sns.distplot(instant_low_growth_rates_contnd_fitted_no_lin_growth['popt_k for "Mean Contact Surface Length (N.D.)"'])
# Based on the above 2 distplots, whether or not you include linear growth has 
# no effect on the optimally fitted logistic growth rates.
#(instant_low_growth_rates_contnd_fitted['popt_k for "Mean Contact Surface Length (N.D.)"'] == instant_low_growth_rates_contnd_fitted_no_lin_growth['popt_k for "Mean Contact Surface Length (N.D.)"']).all()
# The above line confirms it.

plt.close('all')
# Compare the angle growth rates between ecto-ecto low to high, and ecto-ecto
# 0 to low:
pt_size = 5
jit=0.25
conditions = ['ecto-ecto']
data_max = max(logistic_highplat_growth_rates_angle['popt_k2'].max(), 
               logistic_low_growth_rates_angle['popt_k2'].max(), 
               instant_low_growth_rates_angle_fitted['popt_k2'].max())
data_min = min(logistic_highplat_growth_rates_angle['popt_k2'].min(), 
               logistic_low_growth_rates_angle['popt_k2'].min(), 
               instant_low_growth_rates_angle_fitted['popt_k2'].min())
fig, ax1 = plt.subplots(figsize=(width_1col,height_2col))
sns.stripplot(x='Experimental_Condition', y='popt_k2', 
              # data=logistic_highplat_growth_rates_angle, 
              data=logistic_highplat_growth_rates_angle[np.logical_or.reduce([logistic_highplat_growth_rates_angle['Experimental_Condition'] == x for x in conditions])],
              hue='t_int', palette=['grey', 'k'], jitter=jit, size=pt_size, ax=ax1)
ax1.semilogy()
ax1.set_ylim(10**order_of_mag(data_min), 10**(order_of_mag(data_max)+1))
ax1.grid(b=True, which='both', axis='y', c='lightgrey')
ax1.set_xticklabels(labels='')
# ax1.set_xticklabels(labels=ax1.get_xticklabels(), ha='right', rotation='45')
# ax1.set_ylabel('logistic growth rate')
ax1.set_ylabel('')
ax1.set_xlabel('')
# ax1.set_title('valid_growth_data_low_to_high')
plt.tight_layout()
fig.savefig(Path.joinpath(output_dir, r"logistic_growth_rate_comparison_angle_valid_growth_data_low_to_high"), dpi=dpi_graph)

plt.close('all')
fig, ax2 = plt.subplots(figsize=(width_1col,height_2col))
sns.stripplot(x='Experimental_Condition', y='popt_k2', 
              # data=logistic_low_growth_rates_angle, 
              data=logistic_low_growth_rates_angle[np.logical_or.reduce([logistic_low_growth_rates_angle['Experimental_Condition'] == x for x in conditions])],
              hue='t_int', palette=['grey', 'k'], jitter=jit, size=pt_size, ax=ax2)
ax2.semilogy()
ax2.set_ylim(10**order_of_mag(data_min), 10**(order_of_mag(data_max)+1))
ax2.grid(b=True, which='both', axis='y', c='lightgrey')
ax2.set_xticklabels(labels='')
# ax2.set_xticklabels(labels=ax2.get_xticklabels(), ha='right', rotation='45')
# ax2.set_ylabel('logistic growth rate')
ax2.set_ylabel('')
ax2.set_xlabel('')
# ax2.set_title('valid_growth_data_low_grow')
plt.tight_layout()
fig.savefig(Path.joinpath(output_dir, r"logistic_growth_rate_comparison_angle_valid_growth_data_low_grow"), dpi=dpi_graph)

plt.close('all')
fig, ax3 = plt.subplots(figsize=(width_1col,height_2col))
sns.stripplot(x='Experimental_Condition', y='popt_k2', 
              # data=instant_low_growth_rates_angle_fitted, 
              data=instant_low_growth_rates_angle_fitted[np.logical_or.reduce([instant_low_growth_rates_angle_fitted['Experimental_Condition'] == x for x in conditions])],
              hue='t_int', palette=['grey', 'k'], jitter=jit, size=pt_size, ax=ax3)
ax3.semilogy()
ax3.set_ylim(10**order_of_mag(data_min), 10**(order_of_mag(data_max)+1))
ax3.grid(b=True, which='both', axis='y', c='lightgrey')
ax3.set_xticklabels(labels='')
# ax3.set_xticklabels(labels=ax3.get_xticklabels(), ha='right', rotation='45')
# ax3.set_ylabel('logistic growth rate')
ax3.set_ylabel('')
ax3.set_xlabel('')
# ax3.set_title('instant_growth_low_grow')
plt.tight_layout()
fig.savefig(Path.joinpath(output_dir, r"logistic_growth_rate_comparison_angle_instant_growth_low_grow"), dpi=dpi_graph)



plt.close('all')
# Compare the angle growth rates between ecto-ecto low to high, and ecto-ecto
# 0 to low:
conditions = ['ecto-ecto']
logistic_highplat_growth_rates_angle['growth_rate_type'] = ['low to high plateau'] * len(logistic_highplat_growth_rates_angle)
logistic_low_growth_rates_angle['growth_rate_type'] = ['slow growth to low plateau'] * len(logistic_low_growth_rates_angle)
instant_low_growth_rates_angle_fitted['growth_rate_type'] = ['fast growth to low plateau'] * len(instant_low_growth_rates_angle_fitted)
# instant_low_growth_rates_angle_fittedv2 = instant_low_growth_rates_angle_fitted.rename(columns={'popt_k2':'popt_k2'})
data = pd.concat([logistic_highplat_growth_rates_angle, logistic_low_growth_rates_angle, instant_low_growth_rates_angle_fitted])
data = data.query('Experimental_Condition == "ecto-ecto"')
data_min, data_max = min(data['popt_k2']), max(data['popt_k2'])

fig, ax = plt.subplots(figsize=(width_2col*2/3*0.9,height_1pt5col*0.9))
sns.stripplot(x='growth_rate_type', y='popt_k2', data=data, 
              hue='t_int', palette=['grey', 'k'], jitter=jit, 
              size=pt_size, ax=ax)
ax.semilogy()
ax.set_ylim(10**order_of_mag(data_min), 10**(order_of_mag(data_max)+1))
ax.grid(b=True, which='both', axis='y', c='lightgrey')
ax.set_xticklabels(labels='')
ax.set_xlabel('')
ax.set_ylabel('')
handles, labels = ax.get_legend_handles_labels()
labels = [float(x)*60 for x in labels]
ax.legend(title='timestep\n(seconds)', handles=handles, labels=['30s', '60s'])

plt.tight_layout()
fig.savefig(Path.joinpath(output_dir, r"logistic_growth_rate_comparison_angle_low_grow.tif"), dpi=dpi_graph, bbox_inches='tight')



plt.close('all')
# Compare the contnd growth rates between ecto-ecto low to high, and ecto-ecto
# 0 to low:
data_max = max(logistic_highplat_growth_rates_contnd['popt_k2'].max(), 
               logistic_low_growth_rates_contnd['popt_k2'].max(), 
               instant_low_growth_rates_contnd_fitted['popt_k2'].max())
data_min = min(logistic_highplat_growth_rates_contnd['popt_k2'].min(), 
               logistic_low_growth_rates_contnd['popt_k2'].min(), 
               instant_low_growth_rates_contnd_fitted['popt_k2'].min())
fig, ax1 = plt.subplots(figsize=(width_1col,height_2col))
sns.stripplot(x='Experimental_Condition', y='popt_k2', 
              # data=logistic_highplat_growth_rates_contnd,
              data=logistic_highplat_growth_rates_contnd[np.logical_or.reduce([logistic_highplat_growth_rates_contnd['Experimental_Condition'] == x for x in conditions])],
              hue='t_int', palette=['grey', 'k'], jitter=jit, size=pt_size, ax=ax1)
ax1.semilogy()
ax1.set_ylim(10**order_of_mag(data_min), 10**(order_of_mag(data_max)+1))
ax1.grid(b=True, which='both', axis='y', c='lightgrey')
ax1.set_xticklabels(labels='')
# ax1.set_xticklabels(labels=ax1.get_xticklabels(), ha='right', rotation='45')
# ax1.set_ylabel('logistic growth rate')
ax1.set_ylabel('')
ax1.set_xlabel('')
# ax1.set_title('valid_growth_data_low_to_high')
plt.tight_layout()
fig.savefig(Path.joinpath(output_dir, r"logistic_growth_rate_comparison_contnd_valid_growth_data_low_to_high"), dpi=dpi_graph)

plt.close('all')
fig, ax2 = plt.subplots(figsize=(width_1col,height_2col))
sns.stripplot(x='Experimental_Condition', y='popt_k2', 
              # data=logistic_low_growth_rates_contnd, 
              data=logistic_low_growth_rates_contnd[np.logical_or.reduce([logistic_low_growth_rates_contnd['Experimental_Condition'] == x for x in conditions])],
              hue='t_int', palette=['grey', 'k'], jitter=jit, size=pt_size, ax=ax2)
ax2.semilogy(10**order_of_mag(data_min))
ax2.set_ylim(10**order_of_mag(data_min), 10**(order_of_mag(data_max)+1))
ax2.grid(b=True, which='both', axis='y', c='lightgrey')
ax2.set_xticklabels(labels='')
# ax2.set_xticklabels(labels=ax2.get_xticklabels(), ha='right', rotation='45')
# ax2.set_ylabel('logistic growth rate')
ax2.set_ylabel('')
ax2.set_xlabel('')
# ax2.set_title('valid_growth_data_low_grow')
plt.tight_layout()
fig.savefig(Path.joinpath(output_dir, r"logistic_growth_rate_comparison_contnd_valid_growth_data_low_grow"), dpi=dpi_graph)

plt.close('all')
fig, ax3 = plt.subplots(figsize=(width_1col,height_2col))
sns.stripplot(x='Experimental_Condition', 
              y='popt_k2', 
              # data=instant_low_growth_rates_contnd_fitted, 
              data=instant_low_growth_rates_contnd_fitted[np.logical_or.reduce([instant_low_growth_rates_contnd_fitted['Experimental_Condition'] == x for x in conditions])],
              hue='t_int', palette=['grey', 'k'], jitter=jit, size=pt_size, ax=ax3)
ax3.semilogy()
ax3.set_ylim(10**order_of_mag(data_min), 10**(order_of_mag(data_max)+1))
ax3.grid(b=True, which='both', axis='y', c='lightgrey')
ax3.set_xticklabels(labels='')
# ax3.set_xticklabels(labels=ax3.get_xticklabels(), ha='right', rotation='45')
# ax3.set_ylabel('logistic growth rate')
ax3.set_ylabel('')
ax3.set_xlabel('')
# ax3.set_title('instant_growth_low_grow')
plt.tight_layout()
fig.savefig(Path.joinpath(output_dir, r"logistic_growth_rate_comparison_contnd_instant_growth_low_grow"), dpi=dpi_graph)



plt.close('all')
# Compare the contnd growth rates between ecto-ecto low to high, and ecto-ecto
# 0 to low:
# max_highplat_growth_rates_contnd = logistic_highplat_growth_rates_contnd.copy(deep=True)

# max_highplat_growth_rates_contnd = ['low to high plateau'] * len(max_highplat_growth_rates_contnd)
logistic_highplat_growth_rates_contnd['growth_rate_type'] = ['low to high plateau'] * len(logistic_highplat_growth_rates_contnd)
logistic_highplat_growth_rates_contnd['max_growth_rate'] = logistic_highplat_growth_rates_contnd.apply(lambda x: np.nanmax(logistic_diffeq(*x[['logistic_x', 'popt_x2','popt_k2', 'popt_high_plateau', 'popt_mid_plateau']])), axis=1)

logistic_low_growth_rates_contnd['growth_rate_type'] = ['slow growth to low plateau'] * len(logistic_low_growth_rates_contnd)
logistic_low_growth_rates_contnd['max_growth_rate'] = logistic_low_growth_rates_contnd.apply(lambda x: np.nanmax(logistic_diffeq(*x[['logistic_x', 'popt_x2','popt_k2', 'popt_high_plateau', 'popt_mid_plateau']])), axis=1)

instant_low_growth_rates_contnd_fitted['growth_rate_type'] = ['fast growth to low plateau'] * len(instant_low_growth_rates_contnd_fitted)
instant_low_growth_rates_contnd_fitted['max_growth_rate'] = instant_low_growth_rates_contnd_fitted.apply(lambda x: np.nanmax(logistic_diffeq(*x[['logistic_x', 'popt_x2','popt_k2', 'popt_high_plateau', 'popt_mid_plateau']])), axis=1)

# instant_low_growth_rates_contnd_fittedv2 = instant_low_growth_rates_contnd_fitted.rename(columns={'popt_k for "Mean Contact Surface Length (N.D.)"':'popt_k2'})
data = pd.concat([logistic_highplat_growth_rates_contnd, logistic_low_growth_rates_contnd, instant_low_growth_rates_contnd_fitted])
data = data.query('Experimental_Condition == "ecto-ecto"')
data_min, data_max = min(data['popt_k2']), max(data['popt_k2'])

fig, ax = plt.subplots(figsize=(width_2col*2/3 *0.9,height_1pt5col*0.9))
sns.stripplot(x='growth_rate_type', y='popt_k2', data=data, 
              hue='t_int', palette=['grey', 'k'], 
              jitter=jit, dodge=False, size=pt_size, ax=ax)
ax.semilogy()
ax.set_ylim(10**order_of_mag(data_min), 10**(order_of_mag(data_max)+1))
ax.grid(b=True, which='both', axis='y', c='lightgrey')
ax.set_xticklabels(labels='')
ax.set_ylabel('')
ax.set_xlabel('')
handles, labels = ax.get_legend_handles_labels()
labels = [float(x)*60 for x in labels]
ax.legend(title='timestep\n(seconds)', handles=handles, labels=['30s', '60s'])

plt.tight_layout()
fig.savefig(Path.joinpath(output_dir, r"logistic_growth_rate_comparison_contnd_low_grow.tif"), dpi=dpi_graph, bbox_inches='tight')


# fig, ax = plt.subplots(figsize=(width_2col*2/3 *0.9,height_1pt5col*0.9))
# sns.stripplot(x='growth_rate_type', y='Mean Contact Surface Length (N.D.)', data=data, 
#               hue='t_int', palette=['grey', 'k'], 
#               jitter=jit, dodge=False, size=pt_size, ax=ax)
# ax.semilogy()
# ax.set_ylim(10**order_of_mag(data_min), 10**(order_of_mag(data_max)+1))
# ax.grid(b=True, which='both', axis='y', c='lightgrey')
# ax.set_xticklabels(labels='')
# ax.set_ylabel('')
# ax.set_xlabel('')
# handles, labels = ax.get_legend_handles_labels()
# labels = [float(x)*60 for x in labels]
# ax.legend(title='timestep\n(seconds)', handles=handles, labels=['30s', '60s'])

# plt.tight_layout()
# fig.savefig(Path.joinpath(output_dir, r"mean_growth_rate_comparison_contnd_low_grow.tif"), dpi=dpi_graph, bbox_inches='tight')



# Compare the contnd growth rates between ecto-ecto low to high, and ecto-ecto
# 0 to low:
logistic_highplat_growth_rates_contnd['growth_rate_type'] = ['low to high plateau'] * len(logistic_highplat_growth_rates_contnd)
logistic_highplat_growth_rates_contnd['max_growth_rate'] = logistic_highplat_growth_rates_contnd.apply(lambda x: np.nanmax(logistic_diffeq(*x[['logistic_x', 'popt_x2','popt_k2', 'popt_high_plateau', 'popt_mid_plateau']])), axis=1)

logistic_low_growth_rates_contnd['growth_rate_type'] = ['slow growth to low plateau'] * len(logistic_low_growth_rates_contnd)
logistic_low_growth_rates_contnd['max_growth_rate'] = logistic_low_growth_rates_contnd.apply(lambda x: np.nanmax(logistic_diffeq(*x[['logistic_x', 'popt_x2','popt_k2', 'popt_high_plateau', 'popt_mid_plateau']])), axis=1)

instant_low_growth_rates_contnd_fitted['growth_rate_type'] = ['fast growth to low plateau'] * len(instant_low_growth_rates_contnd_fitted)
instant_low_growth_rates_contnd_fitted['max_growth_rate'] = instant_low_growth_rates_contnd_fitted.apply(lambda x: np.nanmax(logistic_diffeq(*x[['logistic_x', 'popt_x2','popt_k2', 'popt_high_plateau', 'popt_mid_plateau']])), axis=1)


# instant_low_growth_rates_contnd_fittedv2 = instant_low_growth_rates_contnd_fitted.rename(columns={'popt_k for "Mean Contact Surface Length (N.D.)"':'popt_k2'})
data = pd.concat([logistic_highplat_growth_rates_contnd, logistic_low_growth_rates_contnd, instant_low_growth_rates_contnd_fitted])
data = data.query('Experimental_Condition == "ecto-ecto"')
data_min, data_max = min(data['max_growth_rate']), max(data['max_growth_rate'])

plt.close('all')
fig, ax = plt.subplots(figsize=(width_2col*2/3 *0.9,height_1pt5col*0.9))
sns.stripplot(x='growth_rate_type', y='max_growth_rate', data=data, 
              hue='t_int', palette=['grey', 'k'], 
              jitter=jit, dodge=False, size=pt_size, ax=ax)
ax.semilogy()
ax.set_ylim(10**order_of_mag(data_min), 10**(order_of_mag(data_max)+1))
ax.grid(b=True, which='both', axis='y', c='lightgrey')
ax.set_xticklabels(labels='')
ax.set_ylabel('')
ax.set_xlabel('')
handles, labels = ax.get_legend_handles_labels()
labels = [float(x)*60 for x in labels]
ax.legend(title='timestep\n(seconds)', handles=handles, labels=['30s', '60s'])

plt.tight_layout()
fig.savefig(Path.joinpath(output_dir, r"max_growth_rate_comparison_contnd_low_grow.tif"), dpi=dpi_graph, bbox_inches='tight')



## We will also produce the dimensionalized data for 
## logistic_growth_rate_comparison_contnd_low_grow.tif:
first_valid_radius_sf_mlist = pd.concat([ee_first_valid_radius_sf, 
                                         mm_first_valid_radius_sf, 
                                         nn_first_valid_radius_sf, 
                                         em_first_valid_radius_sf, 
                                         eedb_first_valid_radius_sf, 
                                         eeCcadMO_first_valid_radius_sf, 
                                         eeRcadMO_first_valid_radius_sf,
                                         eeRCcadMO_first_valid_radius_sf]) # 'mlist' for 'master list'


data_cont = np.hstack([logistic_highplat_growth_rates_contnd.query('Source_File == "%s"'%sf)['max_growth_rate'].values * first_valid_radius_sf_mlist.query('Source_File == "%s"'%sf)['first_valid_radius (um)'].values for sf in logistic_highplat_growth_rates_contnd.Source_File])
logistic_highplat_growth_rates_cont = logistic_highplat_growth_rates_contnd.copy(deep=True)
logistic_highplat_growth_rates_cont['max_growth_rate'] = data_cont


data_cont = np.hstack([logistic_low_growth_rates_contnd.query('Source_File == "%s"'%sf)['max_growth_rate'].values * first_valid_radius_sf_mlist.query('Source_File == "%s"'%sf)['first_valid_radius (um)'].values for sf in logistic_low_growth_rates_contnd.Source_File])
logistic_low_growth_rates_cont = logistic_low_growth_rates_contnd.copy(deep=True)
logistic_low_growth_rates_cont['max_growth_rate'] = data_cont


data_cont = np.hstack([instant_low_growth_rates_contnd_fitted.query('Source_File == "%s"'%sf)['max_growth_rate'].values * first_valid_radius_sf_mlist.query('Source_File == "%s"'%sf)['first_valid_radius (um)'].values for sf in instant_low_growth_rates_contnd_fitted.Source_File])
instant_low_growth_rates_cont_fitted = instant_low_growth_rates_contnd_fitted.copy(deep=True)
instant_low_growth_rates_cont_fitted['max_growth_rate'] = data_cont


data = pd.concat([logistic_highplat_growth_rates_cont, logistic_low_growth_rates_cont, instant_low_growth_rates_cont_fitted])
data = data.query('Experimental_Condition == "ecto-ecto"')
data_min, data_max = min(data['max_growth_rate']), max(data['max_growth_rate'])

plt.close('all')
fig, ax = plt.subplots(figsize=(width_2col*2/3 *0.9,height_1pt5col*0.9))
# sns.stripplot(x='growth_rate_type', y='max_growth_rate', data=data, 
#               hue='t_int', palette=['grey', 'k'], 
#               jitter=jit, dodge=False, size=pt_size, ax=ax)
sns.swarmplot(x='growth_rate_type', y='max_growth_rate', data=data, 
              hue='t_int', palette=['grey', 'k'], 
              dodge=False, size=pt_size, ax=ax)
sns.boxplot(x='growth_rate_type', y='max_growth_rate', data=data,
            orient='v', medianprops=dict(color='red'), 
            boxprops=dict(facecolor='w'), ax=ax)
            # boxprops=dict(color='w'),
# ax.semilogy()
# ax.set_ylim(10**order_of_mag(data_min), 10**(order_of_mag(data_max)+1))
ax.set_ylim(0, data_max+1)
ax.grid(b=True, which='both', axis='y', c='lightgrey')
ax.set_xticklabels(labels='')
ax.set_ylabel('')
ax.set_xlabel('')
handles, labels = ax.get_legend_handles_labels()
labels = [float(x)*60 for x in labels]
ax.legend(title='timestep', handles=handles, labels=['30s', '60s'], ncol=1, loc=2)

plt.tight_layout()
fig.savefig(Path.joinpath(output_dir, r"max_growth_rate_comparison_cont_low_grow.tif"), dpi=dpi_graph, bbox_inches='tight')





# Now to produce the mean growth rates for the low plateau:
# indiv_logistic_df_low_growth_ee_contnd.keys() # 


# Get the logistic fits to the low growth where 5% of growth 
# is left to the low adhesion plateau and figure out the mean
# growth rate from 0 to that point (if the fit is good). 
# If there is no growth data to the low plateau (ie. for 
# grow_to_low_instant data) then take the mean growth rate 
# from 0 to the mean value of that low plateau. 

# Check the 2nd bookmark down from here, that is where the 
# low plateau histograms are. Which data sets did they use?
# Want to double check which source files are used for 
# instant low growth and logistic low growth data





###############################################################################
# Extract dynamics of cell-cell contacts from the adhesion kinetics profiles
# for whole timelapses (low plateau, growth phase, and high plateau together):
ee_source_files = np.array(list(ee_list_long.groupby('Source_File').groups.keys()))
#exptl_cond = np.array(list(ee_list_long.groupby('Source_File').agg(np.max)['Experimental_Condition']))
exptl_cond = np.array(list(ee_list_long.groupby('Source_File').agg(pd.Series.mode)['Experimental_Condition']))
ee_time_intervals = [float(np.unique(np.diff(x['Time (minutes)']))) for x in ee_list_original]
num_touch_events_ee = np.array([len(num_contig_true(x['cells touching?'], x['Mask? (manually defined by user who checks quality of cell borders)'])[0]) for x in ee_list_original])
first_touch_event_ee = np.array([num_contig_true(x['cells touching?'], x['Mask? (manually defined by user who checks quality of cell borders)'])[1] for x in ee_list_original])
contig_touch_ee = [num_contig_true(x['cells touching?'])[0] for x in ee_list_original]
num_separation_events_ee = np.array([len(num_contig_true(~x['cells touching?'], x['Mask? (manually defined by user who checks quality of cell borders)'])[0]) for x in ee_list_original])
first_separation_event_ee = np.array([num_contig_true(~x['cells touching?'], x['Mask? (manually defined by user who checks quality of cell borders)'])[1] for x in ee_list_original])
cols = ['Experimental_Condition', 'Source_File', 'time_intervals', 'num_touch_events', 'first_touch_events', 'contig_touch', 'num_separation_events', 'first_separation_event']
data = [exptl_cond, ee_source_files, ee_time_intervals, num_touch_events_ee, first_touch_event_ee, contig_touch_ee, num_separation_events_ee, first_separation_event_ee]
ee_contact_dynamics = pd.DataFrame(data = dict(zip(cols,data)) )

ee_low_plat_original = [x[x['Time (minutes)'] < growth_phase_start_ee_contnd] for x in ee_list_original]
ee_source_files = np.array(list(ee_list_long.groupby('Source_File').groups.keys()))
exptl_cond = np.asarray(len(ee_low_plat_original) * ['ecto-ecto low plateau'])
num_touch_events_ee = np.array([len(num_contig_true(x['cells touching?'], x['Mask? (manually defined by user who checks quality of cell borders)'])[0]) for x in ee_low_plat_original])
first_touch_event_ee = np.array([num_contig_true(x['cells touching?'], x['Mask? (manually defined by user who checks quality of cell borders)'])[1] for x in ee_low_plat_original])
contig_touch_ee = [num_contig_true(x['cells touching?'])[0] for x in ee_low_plat_original]
num_separation_events_ee = np.array([len(num_contig_true(~x['cells touching?'], x['Mask? (manually defined by user who checks quality of cell borders)'])[0]) for x in ee_low_plat_original])
first_separation_event_ee = np.array([num_contig_true(~x['cells touching?'], x['Mask? (manually defined by user who checks quality of cell borders)'])[1] for x in ee_low_plat_original])
cols = ['Experimental_Condition', 'Source_File', 'num_touch_events', 'first_touch_events', 'contig_touch', 'num_separation_events', 'first_separation_event']
data = [exptl_cond, ee_source_files, num_touch_events_ee, first_touch_event_ee, contig_touch_ee, num_separation_events_ee, first_separation_event_ee]
ee_low_plat_contact_dynamics = pd.DataFrame(data = dict(zip(cols,data)) )


mm_low_plat_original = [x[x['Time (minutes)'] < growth_phase_start_mm_contnd] for x in mm_list_original]
mm_source_files = list(mm_list_long.groupby('Source_File').groups.keys())
exptl_cond = np.asarray(len(mm_low_plat_original) * ['meso-meso low plateau'])
num_touch_events_mm = [len(num_contig_true(x['cells touching?'], x['Mask? (manually defined by user who checks quality of cell borders)'])[0]) for x in mm_low_plat_original]
first_touch_event_mm = [num_contig_true(x['cells touching?'], x['Mask? (manually defined by user who checks quality of cell borders)'])[1] for x in mm_low_plat_original]
contig_touch_mm = [num_contig_true(x['cells touching?'])[0] for x in mm_low_plat_original]
num_separation_events_mm = [len(num_contig_true(~x['cells touching?'], x['Mask? (manually defined by user who checks quality of cell borders)'])[0]) for x in mm_low_plat_original]
first_separation_event_mm = [num_contig_true(~x['cells touching?'], x['Mask? (manually defined by user who checks quality of cell borders)'])[1] for x in mm_low_plat_original]
cols = ['Experimental_Condition', 'Source_File', 'num_touch_events', 'first_touch_events', 'contig_touch', 'num_separation_events', 'first_separation_event']
data = [exptl_cond, mm_source_files, num_touch_events_mm, first_touch_event_mm, contig_touch_mm, num_separation_events_mm, first_separation_event_mm]
mm_low_plat_contact_dynamics = pd.DataFrame(data = dict(zip(cols,data)) )

mm_source_files = list(mm_list_long.groupby('Source_File').groups.keys())
#exptl_cond = np.array(list(mm_list_long.groupby('Source_File').agg(np.max)['Experimental_Condition']))
exptl_cond = np.array(list(mm_list_long.groupby('Source_File').agg(pd.Series.mode)['Experimental_Condition']))
num_touch_events_mm = [len(num_contig_true(x['cells touching?'], x['Mask? (manually defined by user who checks quality of cell borders)'])[0]) for x in mm_list_original]
first_touch_event_mm = [num_contig_true(x['cells touching?'], x['Mask? (manually defined by user who checks quality of cell borders)'])[1] for x in mm_list_original]
contig_touch_mm = [num_contig_true(x['cells touching?'])[0] for x in mm_list_original]
num_separation_events_mm = [len(num_contig_true(~x['cells touching?'], x['Mask? (manually defined by user who checks quality of cell borders)'])[0]) for x in mm_list_original]
first_separation_event_mm = [num_contig_true(~x['cells touching?'], x['Mask? (manually defined by user who checks quality of cell borders)'])[1] for x in mm_list_original]
cols = ['Experimental_Condition', 'Source_File', 'num_touch_events', 'first_touch_events', 'contig_touch', 'num_separation_events', 'first_separation_event']
data = [exptl_cond, mm_source_files, num_touch_events_mm, first_touch_event_mm, contig_touch_mm, num_separation_events_mm, first_separation_event_mm]
mm_contact_dynamics = pd.DataFrame(data = dict(zip(cols,data)) )

em_source_files = list(em_list_long.groupby('Source_File').groups.keys())
#exptl_cond = np.array(list(em_list_long.groupby('Source_File').agg(np.max)['Experimental_Condition']))
exptl_cond = np.array(list(em_list_long.groupby('Source_File').agg(pd.Series.mode)['Experimental_Condition']))
num_touch_events_em = [len(num_contig_true(x['cells touching?'], x['Mask? (manually defined by user who checks quality of cell borders)'])[0]) for x in em_list_original]
first_touch_event_em = [num_contig_true(x['cells touching?'], x['Mask? (manually defined by user who checks quality of cell borders)'])[1] for x in em_list_original]
contig_touch_em = [num_contig_true(x['cells touching?'])[0] for x in em_list_original]
num_separation_events_em = [len(num_contig_true(~x['cells touching?'], x['Mask? (manually defined by user who checks quality of cell borders)'])[0]) for x in em_list_original]
first_separation_event_em = [num_contig_true(~x['cells touching?'], x['Mask? (manually defined by user who checks quality of cell borders)'])[1] for x in em_list_original]
cols = ['Experimental_Condition', 'Source_File', 'num_touch_events', 'first_touch_events', 'contig_touch', 'num_separation_events', 'first_separation_event']
data = [exptl_cond, em_source_files, num_touch_events_em, first_touch_event_em, contig_touch_em, num_separation_events_em, first_separation_event_em]
em_contact_dynamics = pd.DataFrame(data = dict(zip(cols,data)) )

eedb_source_files = list(eedb_list_long.groupby('Source_File').groups.keys())
#exptl_cond = np.array(list(eedb_list_long.groupby('Source_File').agg(np.max)['Experimental_Condition']))
exptl_cond = np.array(list(eedb_list_long.groupby('Source_File').agg(pd.Series.mode)['Experimental_Condition']))
num_touch_events_eedb = [len(num_contig_true(x['cells touching?'], x['Mask? (manually defined by user who checks quality of cell borders)'])[0]) for x in eedb_list_original]
first_touch_event_eedb = [num_contig_true(x['cells touching?'], x['Mask? (manually defined by user who checks quality of cell borders)'])[1] for x in eedb_list_original]
contig_touch_eedb = [num_contig_true(x['cells touching?'])[0] for x in eedb_list_original]
num_separation_events_eedb = [len(num_contig_true(~x['cells touching?'], x['Mask? (manually defined by user who checks quality of cell borders)'])[0]) for x in eedb_list_original]
first_separation_event_eedb = [num_contig_true(~x['cells touching?'], x['Mask? (manually defined by user who checks quality of cell borders)'])[1] for x in eedb_list_original]
cols = ['Experimental_Condition', 'Source_File', 'num_touch_events', 'first_touch_events', 'contig_touch', 'num_separation_events', 'first_separation_event']
data = [exptl_cond, eedb_source_files, num_touch_events_eedb, first_touch_event_eedb, contig_touch_eedb, num_separation_events_eedb, first_separation_event_eedb]
eedb_contact_dynamics = pd.DataFrame(data = dict(zip(cols,data)) )

eeCcadMO_source_files = list(eeCcadMO_list_long.groupby('Source_File').groups.keys())
#exptl_cond = np.array(list(eeCcadMO_list_long.groupby('Source_File').agg(np.max)['Experimental_Condition']))
exptl_cond = np.array(list(eeCcadMO_list_long.groupby('Source_File').agg(pd.Series.mode)['Experimental_Condition']))
num_touch_events_eeCcadMO = [len(num_contig_true(x['cells touching?'], x['Mask? (manually defined by user who checks quality of cell borders)'])[0]) for x in eeCcadMO_list_original]
first_touch_event_eeCcadMO = [num_contig_true(x['cells touching?'], x['Mask? (manually defined by user who checks quality of cell borders)'])[1] for x in eeCcadMO_list_original]
contig_touch_eeCcadMO = [num_contig_true(x['cells touching?'])[0] for x in eeCcadMO_list_original]
num_separation_events_eeCcadMO = [len(num_contig_true(~x['cells touching?'], x['Mask? (manually defined by user who checks quality of cell borders)'])[0]) for x in eeCcadMO_list_original]
first_separation_event_eeCcadMO = [num_contig_true(~x['cells touching?'], x['Mask? (manually defined by user who checks quality of cell borders)'])[1] for x in eeCcadMO_list_original]
cols = ['Experimental_Condition', 'Source_File', 'num_touch_events', 'first_touch_events', 'contig_touch', 'num_separation_events', 'first_separation_event']
data = [exptl_cond, eeCcadMO_source_files, num_touch_events_eeCcadMO, first_touch_event_eeCcadMO, contig_touch_eeCcadMO, num_separation_events_eeCcadMO, first_separation_event_eeCcadMO]
eeCcadMO_contact_dynamics = pd.DataFrame(data = dict(zip(cols,data)) )

eeRcadMO_source_files = list(eeRcadMO_list_long.groupby('Source_File').groups.keys())
#exptl_cond = np.array(list(eeRcadMO_list_long.groupby('Source_File').agg(np.max)['Experimental_Condition']))
exptl_cond = np.array(list(eeRcadMO_list_long.groupby('Source_File').agg(pd.Series.mode)['Experimental_Condition']))
num_touch_events_eeRcadMO = [len(num_contig_true(x['cells touching?'], x['Mask? (manually defined by user who checks quality of cell borders)'])[0]) for x in eeRcadMO_list_original]
first_touch_event_eeRcadMO = [num_contig_true(x['cells touching?'], x['Mask? (manually defined by user who checks quality of cell borders)'])[1] for x in eeRcadMO_list_original]
contig_touch_eeRcadMO = [num_contig_true(x['cells touching?'])[0] for x in eeRcadMO_list_original]
num_separation_events_eeRcadMO = [len(num_contig_true(~x['cells touching?'], x['Mask? (manually defined by user who checks quality of cell borders)'])[0]) for x in eeRcadMO_list_original]
first_separation_event_eeRcadMO = [num_contig_true(~x['cells touching?'], x['Mask? (manually defined by user who checks quality of cell borders)'])[1] for x in eeRcadMO_list_original]
cols = ['Experimental_Condition', 'Source_File', 'num_touch_events', 'first_touch_events', 'contig_touch', 'num_separation_events', 'first_separation_event']
data = [exptl_cond, eeRcadMO_source_files, num_touch_events_eeRcadMO, first_touch_event_eeRcadMO, contig_touch_eeRcadMO, num_separation_events_eeRcadMO, first_separation_event_eeRcadMO]
eeRcadMO_contact_dynamics = pd.DataFrame(data = dict(zip(cols,data)) )

nn_source_files = list(nn_list_long.groupby('Source_File').groups.keys())
#exptl_cond = np.array(list(nn_list_long.groupby('Source_File').agg(np.max)['Experimental_Condition']))
exptl_cond = np.array(list(nn_list_long.groupby('Source_File').agg(pd.Series.mode)['Experimental_Condition']))
num_touch_events_nn = [len(num_contig_true(x['cells touching?'], x['Mask? (manually defined by user who checks quality of cell borders)'])[0]) for x in nn_list_original]
first_touch_event_nn = [num_contig_true(x['cells touching?'], x['Mask? (manually defined by user who checks quality of cell borders)'])[1] for x in nn_list_original]
contig_touch_nn = [num_contig_true(x['cells touching?'])[0] for x in nn_list_original]
num_separation_events_nn = [len(num_contig_true(~x['cells touching?'], x['Mask? (manually defined by user who checks quality of cell borders)'])[0]) for x in nn_list_original]
first_separation_event_nn = [num_contig_true(~x['cells touching?'], x['Mask? (manually defined by user who checks quality of cell borders)'])[1] for x in nn_list_original]
cols = ['Experimental_Condition', 'Source_File', 'num_touch_events', 'first_touch_events', 'contig_touch', 'num_separation_events', 'first_separation_event']
data = [exptl_cond, nn_source_files, num_touch_events_nn, first_touch_event_nn, contig_touch_nn, num_separation_events_nn, first_separation_event_nn]
nn_contact_dynamics = pd.DataFrame(data = dict(zip(cols,data)) )

eeRCcadMO_source_files = list(eeRCcadMO_list_long.groupby('Source_File').groups.keys())
#exptl_cond = np.array(list(eeRCcadMO_list_long.groupby('Source_File').agg(np.max)['Experimental_Condition']))
exptl_cond = np.array(list(eeRCcadMO_list_long.groupby('Source_File').agg(pd.Series.mode)['Experimental_Condition']))
num_touch_events_eeRCcadMO = [len(num_contig_true(x['cells touching?'], x['Mask? (manually defined by user who checks quality of cell borders)'])[0]) for x in eeRCcadMO_list_original]
first_touch_event_eeRCcadMO = [num_contig_true(x['cells touching?'], x['Mask? (manually defined by user who checks quality of cell borders)'])[1] for x in eeRCcadMO_list_original]
contig_touch_eeRCcadMO = [num_contig_true(x['cells touching?'])[0] for x in eeRCcadMO_list_original]
num_separation_events_eeRCcadMO = [len(num_contig_true(~x['cells touching?'], x['Mask? (manually defined by user who checks quality of cell borders)'])[0]) for x in eeRCcadMO_list_original]
first_separation_event_eeRCcadMO = [num_contig_true(~x['cells touching?'], x['Mask? (manually defined by user who checks quality of cell borders)'])[1] for x in eeRCcadMO_list_original]
cols = ['Experimental_Condition', 'Source_File', 'num_touch_events', 'first_touch_events', 'contig_touch', 'num_separation_events', 'first_separation_event']
data = [exptl_cond, eeRCcadMO_source_files, num_touch_events_eeRCcadMO, first_touch_event_eeRCcadMO, contig_touch_eeRCcadMO, num_separation_events_eeRCcadMO, first_separation_event_eeRCcadMO]
eeRCcadMO_contact_dynamics = pd.DataFrame(data = dict(zip(cols,data)) )



###############################################################################
# Extract the contact dynamics of the low plateau for ecto-ecto and meso-meso:
# ee_low_plat_original = [x.copy() for x in ee_list_original]
# for i,x in enumerate(ee_low_plat_original):
#     try:
#         x['Time (minutes)'] = x['Time (minutes)'].apply(lambda y: y-ee_indiv_logistic_cont_df.iloc[i]['popt_x2'])
#         x['Time (minutes)'] = x['Time (minutes)'].apply(lambda y: round(y*2)/2)
#     except ValueError:
#         x['Time (minutes)'] = x['Time (minutes)'].apply(lambda y: y * 1)        


# ee_low_plat = [x[x['Time (minutes)'] < growth_phase_start_ee_contnd] for x in ee_low_plat_original]
# ee_source_files = np.array(list(ee_list_long.groupby('Source_File').groups.keys()))
# ee_time_intervals = [float(np.unique(np.diff(x['Time (minutes)']))) for x in ee_list_original]
# exptl_cond = np.asarray(len(ee_low_plat) * ['ecto-ecto low plateau'])
# num_touch_events_ee = np.array([len(num_contig_true(x['cells touching?'], x['Mask? (manually defined by user who checks quality of cell borders)'])[0]) for x in ee_low_plat])
# first_touch_event_ee = np.array([num_contig_true(x['cells touching?'], x['Mask? (manually defined by user who checks quality of cell borders)'])[1] for x in ee_low_plat])
# contig_touch_ee = [num_contig_true(x['cells touching?'])[0] for x in ee_low_plat]
# num_separation_events_ee = np.array([len(num_contig_true(~x['cells touching?'], x['Mask? (manually defined by user who checks quality of cell borders)'])[0]) for x in ee_low_plat])
# first_separation_event_ee = np.array([num_contig_true(~x['cells touching?'], x['Mask? (manually defined by user who checks quality of cell borders)'])[1] for x in ee_low_plat])
# cols = ['Experimental_Condition', 'Source_File', 'time_intervals', 'num_touch_events', 'first_touch_events', 'contig_touch', 'num_separation_events', 'first_separation_event']
# data = [exptl_cond, ee_source_files, ee_time_intervals, num_touch_events_ee, first_touch_event_ee, contig_touch_ee, num_separation_events_ee, first_separation_event_ee]
# ee_low_plat_contact_dynamics = pd.DataFrame(data = dict(zip(cols,data)) )



###############################################################################
# Extract the contact dynamics of the low plateau for ecto-ecto using only the
# manually curated low plateaus:
ee_low_plat_angle = [x.copy() for x in ee_low_adh_angle]
ee_low_plat = [x.copy() for x in ee_low_adh_contnd]
# for i,x in enumerate(ee_low_plat):
#     try:
#         x['Time (minutes)'] = x['Time (minutes)'].apply(lambda y: y-ee_indiv_logistic_cont_df.iloc[i]['popt_x2'])
#         x['Time (minutes)'] = x['Time (minutes)'].apply(lambda y: round(y*2)/2)
#     except ValueError:
#         x['Time (minutes)'] = x['Time (minutes)'].apply(lambda y: y * 1)        
ee_low_plat_long = pd.concat([df for df in ee_low_plat])


ee_source_files = np.array(list(ee_low_plat_long.groupby('Source_File').groups.keys()))
ee_time_intervals = [float(np.unique(np.diff(x['Time (minutes)']))) for x in ee_low_adh_contnd]
exptl_cond = np.asarray(len(ee_low_plat) * ['ecto-ecto low plateau'])
num_touch_events_ee = np.array([len(num_contig_true(x['cells touching?'], x['Mask? (manually defined by user who checks quality of cell borders)'])[0]) for x in ee_low_plat])
first_touch_event_ee = np.array([num_contig_true(x['cells touching?'], x['Mask? (manually defined by user who checks quality of cell borders)'])[1] for x in ee_low_plat])
contig_touch_ee = [num_contig_true(x['cells touching?'])[0] for x in ee_low_plat]
num_separation_events_ee = np.array([len(num_contig_true(~x['cells touching?'], x['Mask? (manually defined by user who checks quality of cell borders)'])[0]) for x in ee_low_plat])
first_separation_event_ee = np.array([num_contig_true(~x['cells touching?'], x['Mask? (manually defined by user who checks quality of cell borders)'])[1] for x in ee_low_plat])
cols = ['Experimental_Condition', 'Source_File', 'time_intervals', 'num_touch_events', 'first_touch_events', 'contig_touch', 'num_separation_events', 'first_separation_event']
data = [exptl_cond, ee_source_files, ee_time_intervals, num_touch_events_ee, first_touch_event_ee, contig_touch_ee, num_separation_events_ee, first_separation_event_ee]
ee_low_plat_contact_dynamics = pd.DataFrame(data = dict(zip(cols,data)) )


# mm_low_plat_original = [x.copy() for x in mm_list_original]
# for i,x in enumerate(mm_low_plat_original):
#     try:
#         x['Time (minutes)'] = x['Time (minutes)'].apply(lambda y: y-mm_indiv_logistic_cont_df.iloc[i]['popt_x2'])
#         x['Time (minutes)'] = x['Time (minutes)'].apply(lambda y: round(y*2)/2)
#     except ValueError:
#         x['Time (minutes)'] = x['Time (minutes)'].apply(lambda y: y * 1)        


# mm_low_plat = [x[x['Time (minutes)'] < growth_phase_start_mm_contnd] for x in mm_low_plat_original]
# mm_source_files = list(mm_list_long.groupby('Source_File').groups.keys())
# mm_time_intervals = [float(np.unique(np.diff(x['Time (minutes)']))) for x in mm_list_original]
# exptl_cond = np.asarray(len(mm_low_plat) * ['meso-meso low plateau'])
# num_touch_events_mm = [len(num_contig_true(x['cells touching?'], x['Mask? (manually defined by user who checks quality of cell borders)'])[0]) for x in mm_low_plat]
# first_touch_event_mm = [num_contig_true(x['cells touching?'], x['Mask? (manually defined by user who checks quality of cell borders)'])[1] for x in mm_low_plat]
# contig_touch_mm = [num_contig_true(x['cells touching?'])[0] for x in mm_low_plat]
# num_separation_events_mm = [len(num_contig_true(~x['cells touching?'], x['Mask? (manually defined by user who checks quality of cell borders)'])[0]) for x in mm_low_plat]
# first_separation_event_mm = [num_contig_true(~x['cells touching?'], x['Mask? (manually defined by user who checks quality of cell borders)'])[1] for x in mm_low_plat]
# cols = ['Experimental_Condition', 'Source_File', 'time_intervals', 'num_touch_events', 'first_touch_events', 'contig_touch', 'num_separation_events', 'first_separation_event']
# data = [exptl_cond, mm_source_files, mm_time_intervals, num_touch_events_mm, first_touch_event_mm, contig_touch_mm, num_separation_events_mm, first_separation_event_mm]
# mm_low_plat_contact_dynamics = pd.DataFrame(data = dict(zip(cols,data)) )

# the eedb adhesion kinetics
eedb_source_files = np.array(list(eedb_list_long.groupby('Source_File').groups.keys()))
eedb_time_intervals = [float(np.unique(np.diff(x['Time (minutes)']))) for x in eedb_list_original]
exptl_cond = np.asarray(len(eedb_list) * ['ecto-ecto low plateau'])
num_touch_events_eedb = np.array([len(num_contig_true(x['cells touching?'], x['Mask? (manually defined by user who checks quality of cell borders)'])[0]) for x in eedb_list])
first_touch_event_eedb = np.array([num_contig_true(x['cells touching?'], x['Mask? (manually defined by user who checks quality of cell borders)'])[1] for x in eedb_list])
contig_touch_eedb = [num_contig_true(x['cells touching?'])[0] for x in eedb_list]
num_separation_events_eedb = np.array([len(num_contig_true(~x['cells touching?'], x['Mask? (manually defined by user who checks quality of cell borders)'])[0]) for x in eedb_list])
first_separation_event_eedb = np.array([num_contig_true(~x['cells touching?'], x['Mask? (manually defined by user who checks quality of cell borders)'])[1] for x in eedb_list])
cols = ['Experimental_Condition', 'Source_File', 'time_intervals', 'num_touch_events', 'first_touch_events', 'contig_touch', 'num_separation_events', 'first_separation_event']
data = [exptl_cond, eedb_source_files, eedb_time_intervals, num_touch_events_eedb, first_touch_event_eedb, contig_touch_eedb, num_separation_events_eedb, first_separation_event_eedb]
eedb_contact_dynamics = pd.DataFrame(data = dict(zip(cols,data)) )

        
###############################################################################


ee_low_plat_contnd = [x.copy() for x in ee_contact_diam_non_dim_df_list_dropna]
for i,x in enumerate(ee_low_plat_contnd):
    try:
        x['Time (minutes)'] = x['Time (minutes)'].apply(lambda y: y-ee_indiv_logistic_cont_df.iloc[i]['popt_x2'])
        x['Time (minutes)'] = x['Time (minutes)'].apply(lambda y: round(y*2)/2)
    except ValueError:
        x['Time (minutes)'] = x['Time (minutes)'].apply(lambda y: y * 1)        


ee_low_plat_contnd_generous = [x[x['Time (minutes)'] < growth_phase_start_ee_contnd] for x in ee_low_plat_contnd]
ee_low_plat_contnd_df = pd.concat(ee_low_plat_contnd_generous)

ee_low_plat_contnd_strict = [x[x['Time (minutes)'] < growth_phase_start_ee_contnd_strict] for x in ee_low_plat_contnd]
ee_low_plat_contnd_strict_df = pd.concat(ee_low_plat_contnd_strict)


ee_low_plat_contnd_verified = [x[x['Time (minutes)'] < growth_phase_start_ee_contnd] for x in ee_low_plat_contnd]
ee_low_plat_contnd_verified = [ee_low_plat_contnd_verified[i] for i in np.where(ee_list_dropna_fit_validity_df['low_plat'])[0]]
ee_low_plat_contnd_verified_df = pd.concat(ee_low_plat_contnd_verified)

ee_low_plat_contnd_verified_strict = [x[x['Time (minutes)'] < growth_phase_start_ee_contnd_strict] for x in ee_low_plat_contnd]
ee_low_plat_contnd_verified_strict = [ee_low_plat_contnd_verified_strict[i] for i in np.where(ee_list_dropna_fit_validity_df['low_plat'])[0]]
ee_low_plat_contnd_verified_strict_df = pd.concat(ee_low_plat_contnd_verified_strict)

ee_low_plat_contnd_verified_v_strict = [x[x['Time (minutes)'] < growth_phase_start_ee_contnd_v_strict] for x in ee_low_plat_contnd]
ee_low_plat_contnd_verified_v_strict = [ee_low_plat_contnd_verified_v_strict[i] for i in np.where(ee_list_dropna_fit_validity_df['low_plat'])[0]]
ee_low_plat_contnd_verified_v_strict_df = pd.concat(ee_low_plat_contnd_verified_v_strict)

ee_low_plat_contnd_verified_absolute = [x[x['Time (minutes)'] < growth_phase_start_ee_contnd_absolute] for x in ee_low_plat_contnd]
ee_low_plat_contnd_verified_absolute = [ee_low_plat_contnd_verified_absolute[i] for i in np.where(ee_list_dropna_fit_validity_df['low_plat'])[0]]
ee_low_plat_contnd_verified_absolute_df = pd.concat(ee_low_plat_contnd_verified_absolute)


# Subset the timelapses that I thought had low adhesion events (note that these
# are the whole timelapses still, not just the low adhesion events), note that
# this is different from timelapses that had good fits for the low plateau 
# parameters of the individual logistic fits:
ee_low_plat_contnd_data_manual = [ee_contact_diam_non_dim_df_list_dropna[i] for i in np.where(np.isfinite(ee_grow_to_low_validity_contnd['last_low_plat_timeframe (minutes; t = ...)']).values)[0]]
# Now subset just the low adhesion events:
ee_low_plat_contnd_manual_bool_idx = [df['Time (minutes)'] <= ee_low_plat[i]['Time (minutes)'].max() for i,df in enumerate(ee_low_plat_contnd_data_manual)]
ee_low_plat_contnd_manual = [ee_low_plat_contnd_data_manual[i][bool_idx] for i,bool_idx in enumerate(ee_low_plat_contnd_manual_bool_idx)]
# and make a dataframe out of the list of subset data:
ee_low_plat_contnd_manual_absolute_df = pd.concat([df for df in ee_low_plat_contnd_manual])

# ee_first_valid_area_low_only = [ee_first_valid_area[i] for i in np.where(ee_list_dropna_fit_validity_df['low_plat'])[0]]
ee_first_valid_area_low_only = [ee_first_valid_area[i] for i in np.where(np.isfinite(ee_grow_to_low_validity_contnd['last_low_plat_timeframe (minutes; t = ...)']).values)[0]]
# ee_low_plat_cont_manual_absolute_diams = [ee_low_plat_contnd_manual[i]['Contact Surface Length (N.D.)'] * np.sqrt(x/np.pi) for i,x in enumerate(ee_first_valid_area_low_only)]
# ee_low_plat_cont_manual_absolute_diams_df = pd.concat([series for series in ee_low_plat_cont_manual_absolute_diams])


# Repeat for angle data:
ee_low_plat_angle_data_manual = [ee_contact_diam_non_dim_df_list_dropna[i] for i in np.where(np.isfinite(ee_grow_to_low_validity_angle['last_low_plat_timeframe (minutes; t = ...)']).values)[0]]
ee_low_plat_angle_manual_bool_idx = [df['Time (minutes)'] <= ee_low_plat_angle[i]['Time (minutes)'].max() for i,df in enumerate(ee_low_plat_angle_data_manual)]
ee_low_plat_angle_manual = [ee_low_plat_angle_data_manual[i][bool_idx] for i,bool_idx in enumerate(ee_low_plat_angle_manual_bool_idx)]
ee_low_plat_angle_manual_absolute_df = pd.concat([df for df in ee_low_plat_angle_manual])

ee_first_valid_area_low_only_angle = [ee_first_valid_area[i] for i in np.where(np.isfinite(ee_grow_to_low_validity_angle['last_low_plat_timeframe (minutes; t = ...)']).values)[0]]





contact_dynamics = pd.concat( [ee_low_plat_contact_dynamics] )

        
# Return max & min in each list of contig_touch when 30second and 1minute t_int
# are pooled together:
# df_maxs = contact_dynamics.groupby('Source_File')['contig_touch'].apply(lambda x: max(x.iloc[0])).reset_index()
# df_mins = contact_dynamics.groupby('Source_File')['contig_touch'].apply(lambda x: min(x.iloc[0])).reset_index()
# # Flatten all of the lists based on Experimental_Condition:
# df_flattened = contact_dynamics.groupby('Experimental_Condition')['contig_touch'].apply(lambda x: [item for sublist in x for item in sublist]).reset_index()
# lengths = [len(item) for item in df_flattened['contig_touch']]
# df_flattened_expanded = pd.DataFrame( {"Experimental_Condition":np.repeat(df_flattened['Experimental_Condition'].values,lengths), "contig_touch":np.concatenate(df_flattened['contig_touch'])} )
# df_fl_ex_summary = df_flattened_expanded.groupby("Experimental_Condition").describe().reset_index()

# df_flattened_sf = contact_dynamics.groupby('Source_File')#['contig_touch', 'Experimental_Condition']#.apply(lambda x: [item for sublist in x for item in sublist]).reset_index()
# df_flattened_sf = contact_dynamics.copy().reset_index()
# lengths = [len(item) for item in df_flattened_sf['contig_touch']]
# df_flattened_expanded_sf = pd.DataFrame( {"Experimental_Condition":np.repeat(df_flattened_sf['Experimental_Condition'].values,lengths), "Source_File":np.repeat(df_flattened_sf['Source_File'].values,lengths), "contig_touch":np.concatenate(df_flattened_sf['contig_touch'])} )
# df_fl_ex_sf_summary = df_flattened_expanded_sf.groupby(['Experimental_Condition','Source_File']).describe().reset_index()







###############################################################################
### Plot the contact dynamics without separating out the 30second intervals and
# 1minute intervals together in the same :
# category_order = contact_dynamics.groupby('Experimental_Condition', sort=False).count().reset_index()['Experimental_Condition']


# plt.close('all')
# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(height_1col/2,height_1col))
# sns.swarmplot(x='Experimental_Condition', y=('contig_touch','max'), data=df_fl_ex_sf_summary, dodge=True, color='k', size=3, order=category_order, ax=ax)
# ax.axhline(1, c='k', lw=1, ls='--')
# ax.semilogy()
# #ax.set_ylabel('Longest Contact \nDuration (timeframes)')
# ax.set_ylabel('')
# ax.set_xlabel('')
# ax.set_xticklabels('')
# [plt.axvline(x=val, lw=0.5, c='grey') for val in list(ax.get_xticks()[:-1] + np.diff(ax.get_xticks())/2.)]
# [plt.axhline(y=val, lw=1, c='lightgrey') for val in list(range(2, 11, 1))]
# plt.tight_layout()
# plt.draw()
# fig.savefig(Path.joinpath(output_dir, r"max_contig_touch.tif"), dpi=dpi_graph)
# fig.clf()



# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(width_1col,height_1col))
# sns.swarmplot(x='Experimental_Condition', y=('contig_touch','min'), data=df_fl_ex_sf_summary, dodge=False, color='k', size=3, order=category_order, ax=ax)
# ax.axhline(1, c='k', lw=1, ls='--')
# ax.set_xticklabels(labels=ax.get_xticklabels(), rotation = 45, ha='right')
# ax.semilogy()
# #ax.set_ylim(0.9)
# ax.set_ylabel('Shortest Contact \nDuration (timeframes)')
# ax.set_xlabel('')
# [plt.axvline(x=val, lw=0.5, c='grey') for val in list(ax.get_xticks()[:-1] + np.diff(ax.get_xticks())/2.)]
# [plt.axhline(y=val, lw=1, c='lightgrey') for val in list(range(2, 11, 1))]
# plt.tight_layout()
# plt.draw()
#fig.savefig(Path.joinpath(output_dir, r"min contig touches.pdf"), dpi=300)

# fig = plt.figure(figsize=(width_1col,height_1col))
# ax = fig.add_subplot(111)
# sns.swarmplot(x='Experimental_Condition', y='num_touch_events', data=contact_dynamics, dodge=False, color='k', size=3, ax=ax)
# ax.set_xticklabels(labels=ax.get_xticklabels(), rotation = 45, ha='right')
# ax.set_ylabel("Number of Touch Events")
# ax.set_yticks(list(range(0, 22, 2)))
# ax.set_xlabel('')
# [plt.axvline(x=val, lw=0.5, c='grey') for val in list(ax.get_xticks()[:-1] + np.diff(ax.get_xticks())/2.)]
# [plt.axhline(y=val, lw=1, c='lightgrey') for val in list(range(0, 22, 2))]
# plt.tight_layout()
# plt.draw()
#fig.savefig(Path.joinpath(output_dir, r"num_touch_events.pdf"), dpi=300)

#fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4,3))
#sns.swarmplot(x='Experimental_Condition', y=('contig_touch','count'), data=df_fl_ex_sf_summary, dodge=True, ax=ax)
#ax.axhline(np.arange(0,21), c='k', lw=1, ls='--')
#ax.set_yticks(np.linspace(0,df_fl_ex_sf_summary[('contig_touch','count')].max(), num=5, endpoint=True, dtype=int))
#ax.minorticks_on()
#ax.grid(b=True, color='lightgrey', which='minor', axis='y', linewidth=1)
#ax.grid(b=True, color='k', which='major', axis='y', linewidth=1)
#ax.set_xticklabels(labels=ax.get_xticklabels(), rotation = 45, ha='right')
#plt.tight_layout()
#fig.savefig(Path.joinpath(output_dir, r"num touch events.pdf"), dpi=300)



### Plot a histogram of grouped 30second and 1minute t_int data:
# ml = MultipleLocator(5)
# num_bins = df_flattened_expanded_sf.loc[df_flattened_expanded_sf[df_flattened_expanded_sf['Experimental_Condition']=='ecto-ecto low plateau'].index]['contig_touch'].max()
# fig, ax = plt.subplots(figsize=(width_1col,height_1col))
# ax = sns.distplot(df_flattened_expanded_sf.loc[df_flattened_expanded_sf[df_flattened_expanded_sf['Experimental_Condition']=='ecto-ecto low plateau'].index]['contig_touch'], kde=False, bins=num_bins, hist_kws=dict(edgecolor="k", linewidth=1), ax=ax)
# ax.set_xlabel('')
# ax.xaxis.set_minor_locator(ml)
# ax.xaxis.set_tick_params(which='minor')
# fig.savefig(Path.joinpath(output_dir, r"ee_contig_touch_overall.tif"), dpi=dpi_graph)
# fig.clf()



###############################################################################
# Repeat the contact dynamics analysis but based on whether the time interval
# between timepoints is 30seconds or 1minute:
# Return max & min in each list of contig_touch:
df_maxs = contact_dynamics.groupby('Source_File')['contig_touch'].apply(lambda x: max(x.iloc[0])).reset_index()
df_mins = contact_dynamics.groupby('Source_File')['contig_touch'].apply(lambda x: min(x.iloc[0])).reset_index()
df_flattened = contact_dynamics.groupby('time_intervals')['contig_touch'].apply(lambda x: [item for sublist in x for item in sublist]).reset_index()
lengths = [len(item) for item in df_flattened['contig_touch']]
df_flattened_expanded = pd.DataFrame( {"time_intervals":np.repeat(df_flattened['time_intervals'].values,lengths), "contig_touch":np.concatenate(df_flattened['contig_touch'])} )
#df_flattened_expanded = pd.DataFrame( {"Experimental_Condition":np.repeat(df_flattened['Experimental_Condition'].values,lengths), "time_intervals":np.repeat(df_flattened['time_intervals'].values,lengths), "contig_touch":np.concatenate(df_flattened['contig_touch'])} )
df_fl_ex_summary = df_flattened_expanded.groupby("time_intervals").describe().reset_index()


# Using only data that I have manually selected to be before growth phases of
# adhesion, ie. with 0% growth counted in the low adhesiveness (this data
# corresponds to the data that had a statistical test run on it -
# "ee_low_plat_contnd_manual_absolute_df" and was plotted as a histogram - 
# "ee_contig_touch_overall_tintervals"):
df_flattened_sf = contact_dynamics.groupby('Source_File')#['contig_touch', 'Experimental_Condition']#.apply(lambda x: [item for sublist in x for item in sublist]).reset_index()
df_flattened_sf = contact_dynamics.copy().reset_index()
lengths = [len(item) for item in df_flattened_sf['contig_touch']]
df_flattened_expanded_sf = pd.DataFrame( {"Experimental_Condition":np.repeat(df_flattened_sf['Experimental_Condition'].values,lengths),
                                          "time_intervals":np.repeat(df_flattened_sf['time_intervals'].values,lengths),
                                          "Source_File":np.repeat(df_flattened_sf['Source_File'].values,lengths),
                                          "contig_touch":np.concatenate(df_flattened_sf['contig_touch'])} )
df_fl_ex_sf_summary = df_flattened_expanded_sf.groupby(['time_intervals','Source_File']).describe().reset_index()


# Using only data that I have manually selected to be before growth phases of
# adhesion, ie. with 0% growth counted in the low adhesiveness, and also only
# the data where individual logistic fits to the whole timelapse data had a 
# good fit for the low plateau:
df_flattened_expanded_sf_low_plat_only_vals = [x in ee_Source_File_valid_low_plat.values for x in df_flattened_expanded_sf['Source_File'].values]
df_flattened_expanded_sf_low_plat_only = df_flattened_expanded_sf.copy()
df_flattened_expanded_sf_low_plat_only['valid_low_plat'] = df_flattened_expanded_sf_low_plat_only_vals
df_flattened_expanded_sf_low_plat_only = df_flattened_expanded_sf_low_plat_only.query('valid_low_plat==True').reset_index(drop=True)
df_fl_ex_sf_low_plat_only_summary = df_flattened_expanded_sf_low_plat_only.groupby(['time_intervals','Source_File']).describe().reset_index()




### For eedb:
# Repeat the contact dynamics analysis but based on whether the time interval
# between timepoints is 30seconds or 1minute:
# Return max & min in each list of contig_touch:
df_maxs_eedb = eedb_contact_dynamics.groupby('Source_File')['contig_touch'].apply(lambda x: max(x.iloc[0])).reset_index()
df_mins_eedb = eedb_contact_dynamics.groupby('Source_File')['contig_touch'].apply(lambda x: min(x.iloc[0])).reset_index()
df_flattened_eedb = eedb_contact_dynamics.groupby('time_intervals')['contig_touch'].apply(lambda x: [item for sublist in x for item in sublist]).reset_index()
lengths_eedb = [len(item) for item in df_flattened_eedb['contig_touch']]
df_flattened_expanded_eedb = pd.DataFrame( {"time_intervals":np.repeat(df_flattened_eedb['time_intervals'].values,lengths_eedb), "contig_touch":np.concatenate(df_flattened_eedb['contig_touch'])} )
#df_flattened_expanded = pd.DataFrame( {"Experimental_Condition":np.repeat(df_flattened['Experimental_Condition'].values,lengths), "time_intervals":np.repeat(df_flattened['time_intervals'].values,lengths), "contig_touch":np.concatenate(df_flattened['contig_touch'])} )
df_fl_ex_summary_eedb = df_flattened_expanded_eedb.groupby("time_intervals").describe().reset_index()


# Using only data that I have manually selected to be before growth phases of
# adhesion, ie. with 0% growth counted in the low adhesiveness (this data
# corresponds to the data that had a statistical test run on it -
# "ee_low_plat_contnd_manual_absolute_df" and was plotted as a histogram - 
# "ee_contig_touch_overall_tintervals"):
df_flattened_sf_eedb = eedb_contact_dynamics.groupby('Source_File')#['contig_touch', 'Experimental_Condition']#.apply(lambda x: [item for sublist in x for item in sublist]).reset_index()
df_flattened_sf_eedb = eedb_contact_dynamics.copy().reset_index()
lengths_eedb = [len(item) for item in df_flattened_sf_eedb['contig_touch']]
df_flattened_expanded_sf_eedb = pd.DataFrame( {"Experimental_Condition":np.repeat(df_flattened_sf_eedb['Experimental_Condition'].values,lengths_eedb),
                                          "time_intervals":np.repeat(df_flattened_sf_eedb['time_intervals'].values,lengths_eedb),
                                          "Source_File":np.repeat(df_flattened_sf_eedb['Source_File'].values,lengths_eedb),
                                          "contig_touch":np.concatenate(df_flattened_sf_eedb['contig_touch'])} )
df_fl_ex_sf_summary_eedb = df_flattened_expanded_sf_eedb.groupby(['time_intervals','Source_File']).describe().reset_index()

# There are some eedb cells that reach a N.D. contact diameter > 1, I want to
# see if they are responsible for the long tail of the eedb half-life and also
# if the minor peak on the right goes away if they are gone:
files_to_exclude = ['2018-01-25 xl wt-ecto-fdx wt-ecto-rdx 1X DB 20C crop3.csv',
                    '2018-01-25 xl wt-ecto-fdx wt-ecto-rdx 1X DB 20C crop7.csv']
excl_bool = df_flattened_expanded_sf_eedb['Source_File'].apply(lambda x: not(x in files_to_exclude))
df_flattened_expanded_sf_eedb_no_high_adh = df_flattened_expanded_sf_eedb[excl_bool]


### Plot the contact dynamics:
category_order = contact_dynamics.groupby('time_intervals', sort=False).count().reset_index()['time_intervals']



# Manually determined low plateaus only:
plt.close('all')
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(height_1col/2,height_1col))
sns.swarmplot(x='time_intervals', y=('contig_touch','max'), data=df_fl_ex_sf_summary, dodge=True, color='k', size=3, order=category_order, ax=ax)
ax.axhline(1, c='k', lw=1, ls='--')
ax.semilogy()
#ax.set_ylabel('Longest Contact \nDuration (timeframes)')
ax.set_ylabel('')
ax.set_xlabel('')
ax.set_xticklabels('')
[plt.axvline(x=val, lw=0.5, c='grey') for val in list(ax.get_xticks()[:-1] + np.diff(ax.get_xticks())/2.)]
[plt.axhline(y=val, lw=1, c='lightgrey') for val in list(range(2, 11, 1))]
plt.tight_layout()
plt.draw()
fig.savefig(Path.joinpath(output_dir, r"max_contig_touch_tintervals.tif"), dpi=dpi_graph)
fig.clf()




# Manually determined with good low plateau logistic fit parameters only:
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(height_1col/2,height_1col))
sns.swarmplot(x='time_intervals', y=('contig_touch','max'), data=df_fl_ex_sf_low_plat_only_summary, dodge=True, color='k', size=3, order=category_order, ax=ax)
ax.axhline(1, c='k', lw=1, ls='--')
ax.semilogy()
#ax.set_ylabel('Longest Contact \nDuration (timeframes)')
ax.set_ylabel('')
ax.set_xlabel('')
ax.set_xticklabels('')
[plt.axvline(x=val, lw=0.5, c='grey') for val in list(ax.get_xticks()[:-1] + np.diff(ax.get_xticks())/2.)]
[plt.axhline(y=val, lw=1, c='lightgrey') for val in list(range(2, 11, 1))]
plt.tight_layout()
plt.draw()
fig.savefig(Path.joinpath(output_dir, r"max_contig_touch_tintervals_low_plat_and_good_fits.tif"), dpi=dpi_graph)
fig.clf()




# For eedb data:
plt.close('all')
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(height_1col/2,height_1col))
sns.swarmplot(x='time_intervals', y=('contig_touch','max'), data=df_fl_ex_sf_summary_eedb, dodge=True, color='k', size=3, ax=ax)
ax.axhline(1, c='k', lw=1, ls='--')
ax.semilogy()
#ax.set_ylabel('Longest Contact \nDuration (timeframes)')
ax.set_ylabel('')
ax.set_xlabel('')
ax.set_xticklabels('')
[plt.axvline(x=val, lw=0.5, c='grey') for val in list(ax.get_xticks()[:-1] + np.diff(ax.get_xticks())/2.)]
[plt.axhline(y=val, lw=1, c='lightgrey') for val in list(range(2, 11, 1))]
plt.tight_layout()
plt.draw()
fig.savefig(Path.joinpath(output_dir, r"max_contig_touch_tintervals_eedb.tif"), dpi=dpi_graph)
fig.clf()



# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(width_1col,height_1col))
# sns.swarmplot(x='time_intervals', y=('contig_touch','min'), data=df_fl_ex_sf_summary, dodge=False, color='k', size=3, order=category_order, ax=ax)
# ax.axhline(1, c='k', lw=1, ls='--')
# ax.set_xticklabels(labels=ax.get_xticklabels(), rotation = 45, ha='right')
# ax.semilogy()
# #ax.set_ylim(0.9)
# ax.set_ylabel('Shortest Contact \nDuration (timeframes)')
# ax.set_xlabel('')
# [plt.axvline(x=val, lw=0.5, c='grey') for val in list(ax.get_xticks()[:-1] + np.diff(ax.get_xticks())/2.)]
# [plt.axhline(y=val, lw=1, c='lightgrey') for val in list(range(2, 11, 1))]
# plt.tight_layout()
# plt.draw()
#fig.savefig(Path.joinpath(output_dir, r"min contig touches.pdf"), dpi=300)

# fig = plt.figure(figsize=(width_1col,height_1col))
# ax = fig.add_subplot(111)
# sns.swarmplot(x='time_intervals', y='num_touch_events', data=contact_dynamics, dodge=False, color='k', size=3, ax=ax)
# ax.set_xticklabels(labels=ax.get_xticklabels(), rotation = 45, ha='right')
# ax.set_ylabel("Number of Touch Events")
# ax.set_yticks(list(range(0, 22, 2)))
# ax.set_xlabel('')
# [plt.axvline(x=val, lw=0.5, c='grey') for val in list(ax.get_xticks()[:-1] + np.diff(ax.get_xticks())/2.)]
# [plt.axhline(y=val, lw=1, c='lightgrey') for val in list(range(0, 22, 2))]
# plt.tight_layout()
# plt.draw()
#fig.savefig(Path.joinpath(output_dir, r"num_touch_events.pdf"), dpi=300)


fig = plt.figure(figsize=(width_2col/3,height_2col/3))
ax = fig.add_subplot(111)
ml_minor = MultipleLocator(1)
xlim_l, xlim_h, step = 1, contact_dynamics['num_touch_events'].max()+1, 1
bins = np.arange(xlim_l,xlim_h+1,step) -0.5
sns.distplot(contact_dynamics.query('time_intervals == 0.5')['num_touch_events'], 
             kde=False, bins=bins, label='30s',  
             hist_kws={'edgecolor':'k', 'linewidth':'1'}, ax=ax)
sns.distplot(contact_dynamics.query('time_intervals == 1.0')['num_touch_events'], 
             kde=False, bins=bins, label='60s', 
             hist_kws={'edgecolor':'k', 'linewidth':'1'}, ax=ax)
# ax.set_xlabel("Number of Touch Events")
ax.xaxis.set_minor_locator(ml_minor)
ax.set_xticks(np.arange(xlim_l,xlim_h,2))
ax.set_yticks(np.arange(0,13,4))
ax.set_ylim(0,13)
# ax.set_xticklabels(np.arange(xlim_l-1,xlim_h-1,step))
# ax.set_ylabel('Count')
ax.set_xlabel('')
ax.set_ylabel('')
ax.legend(title='timestep\n(seconds)')
# [plt.axvline(x=val, lw=0.5, c='grey') for val in list(ax.get_xticks()[:-1] + np.diff(ax.get_xticks())/2.)]
# [plt.axhline(y=val, lw=1, c='lightgrey') for val in list(range(0, 22, 2))]
# plt.tight_layout()
plt.draw()
fig.savefig(Path.joinpath(output_dir, r"ee_low_adh_num_touch_events.tif"), dpi=dpi_graph, bbox_inches='tight')


#fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(width_1col,height_1col))
#sns.swarmplot(x='Experimental_Condition', y=('contig_touch','count'), data=df_fl_ex_sf_summary, dodge=True, ax=ax)
#ax.axhline(np.arange(0,21), c='k', lw=1, ls='--')
#ax.set_yticks(np.linspace(0,df_fl_ex_sf_summary[('contig_touch','count')].max(), num=5, endpoint=True, dtype=int))
#ax.minorticks_on()
#ax.grid(b=True, color='lightgrey', which='minor', axis='y', linewidth=1)
#ax.grid(b=True, color='k', which='major', axis='y', linewidth=1)
#ax.set_xticklabels(labels=ax.get_xticklabels(), rotation = 45, ha='right')
#plt.tight_layout()
#fig.savefig(Path.joinpath(output_dir, r"num touch events.pdf"), dpi=300)



###############################################################################
ml = MultipleLocator(5)
ml_minor = MultipleLocator(1)
ml_minor2 = MultipleLocator(2)


# Manually determined low plateaus only:
bin_min, bin_max, bin_step = df_flattened_expanded_sf['contig_touch'].min(), df_flattened_expanded_sf['contig_touch'].max() + 1, 1
bins = np.arange(bin_min, bin_max+1, bin_step) - 0.5
# bins = df_flattened_expanded_sf.loc[df_flattened_expanded_sf[df_flattened_expanded_sf['time_intervals']==0.5].index]['contig_touch'].max() - 1
# bins_1 = df_flattened_expanded_sf.loc[df_flattened_expanded_sf[df_flattened_expanded_sf['time_intervals']==1.0].index]['contig_touch'].max() - 1
fig, ax = plt.subplots( figsize=(width_1col,height_1col) )
dist_pt5 = ax.hist(df_flattened_expanded_sf.loc[df_flattened_expanded_sf[df_flattened_expanded_sf['time_intervals']==0.5].index]['contig_touch'],
                   bins=bins, edgecolor="k", linewidth=1, label='t_int = 30s', alpha=0.5)
dist_1 = ax.hist(df_flattened_expanded_sf.loc[df_flattened_expanded_sf[df_flattened_expanded_sf['time_intervals']==1.0].index]['contig_touch'],
                 bins=bins, edgecolor="k", linewidth=1, label='t_int = 60s', alpha=0.5)
ax.legend()
ax.set_xlabel('')
ax.set_xlim(0,35) # Note that one data point is greater than this x limit. Its value is 85.
ax.xaxis.set_major_locator(ml)
ax.xaxis.set_minor_locator(ml_minor)
ax.xaxis.set_tick_params(which='both')
fig.savefig(Path.joinpath(output_dir, r"ee_contig_touch_overall_tintervals.tif"), dpi=dpi_graph)
fig.clf()




# Manually determined with good low plateau logistic fit parameters only:
bin_min, bin_max, bin_step = df_flattened_expanded_sf_low_plat_only['contig_touch'].min(), df_flattened_expanded_sf_low_plat_only['contig_touch'].max() + 1, 1
bins = np.arange(bin_min, bin_max+1, bin_step) - 0.5
# bins = df_flattened_expanded_sf_low_plat_only.loc[df_flattened_expanded_sf_low_plat_only[df_flattened_expanded_sf_low_plat_only['time_intervals']==0.5].index]['contig_touch'].max() - 1
# bins_1 = df_flattened_expanded_sf_low_plat_only.loc[df_flattened_expanded_sf_low_plat_only[df_flattened_expanded_sf_low_plat_only['time_intervals']==1.0].index]['contig_touch'].max() - 1
fig, ax = plt.subplots( figsize=(width_1col,height_1col) )
dist_pt5 = ax.hist(df_flattened_expanded_sf_low_plat_only.loc[df_flattened_expanded_sf_low_plat_only[df_flattened_expanded_sf_low_plat_only['time_intervals']==0.5].index]['contig_touch'],
                   bins=bins, edgecolor="k", linewidth=1, label='t_int = 30s', alpha=0.5)
dist_1 = ax.hist(df_flattened_expanded_sf_low_plat_only.loc[df_flattened_expanded_sf_low_plat_only[df_flattened_expanded_sf_low_plat_only['time_intervals']==1.0].index]['contig_touch'],
                 bins=bins, edgecolor="k", linewidth=1, label='t_int = 60s', alpha=0.5)
ax.legend()
ax.set_xlabel('')
ax.xaxis.set_major_locator(ml)
ax.xaxis.set_minor_locator(ml_minor)
ax.xaxis.set_tick_params(which='both')
fig.savefig(Path.joinpath(output_dir, r"ee_contig_touch_overall_tintervals_low_plat_only.tif"), dpi=dpi_graph)
fig.clf()



# For eedb data:
bin_min, bin_max, bin_step = df_flattened_expanded_sf_eedb['contig_touch'].min(), df_flattened_expanded_sf_eedb['contig_touch'].max() + 1, 1
bins = np.arange(bin_min, bin_max+1, bin_step) - 0.5
# bins = df_flattened_expanded_sf_eedb.loc[df_flattened_expanded_sf_eedb[df_flattened_expanded_sf_eedb['time_intervals']==0.5].index]['contig_touch'].max() - 1
# num_bins_1 = df_flattened_expanded_sf_eedb.loc[df_flattened_expanded_sf_eedb[df_flattened_expanded_sf_eedb['time_intervals']==1.0].index]['contig_touch'].max() - 1
fig, ax = plt.subplots( figsize=(width_1col,height_1col) )
dist_pt5 = ax.hist(df_flattened_expanded_sf_eedb.loc[df_flattened_expanded_sf_eedb[df_flattened_expanded_sf_eedb['time_intervals']==0.5].index]['contig_touch'],
                   bins=bins, edgecolor="k", linewidth=1, label='t_int = 30s', alpha=0.5)
# dist_1 = ax.hist(df_flattened_expanded_sf_eedb.loc[df_flattened_expanded_sf_eedb[df_flattened_expanded_sf_eedb['time_intervals']==1.0].index]['contig_touch'],
#                  bins=num_bins_1, edgecolor="k", linewidth=1, label='t_int = 60s', alpha=0.5)
ax.legend()
ax.set_xlabel('')
# ax.xaxis.set_major_locator(ml)
# ax.xaxis.set_minor_locator(ml_minor)
# ax.xaxis.set_tick_params(which='both')
fig.savefig(Path.joinpath(output_dir, r"eedb_contig_touch_overall_tintervals.tif"), dpi=dpi_graph)
fig.clf()



# Plot a survival curve of % contacts that are still touching after x number of
# timeframes:
sorted_pt5 = np.sort(df_flattened_expanded_sf.loc[df_flattened_expanded_sf[df_flattened_expanded_sf['time_intervals']==0.5].index]['contig_touch'])
sorted_pt5dim = sorted_pt5*0.5
sorted_1 = np.sort(df_flattened_expanded_sf.loc[df_flattened_expanded_sf[df_flattened_expanded_sf['time_intervals']==1.0].index]['contig_touch'])
sorted_1dim = sorted_1.copy()
sorted_dim_all = np.sort(np.concatenate((sorted_pt5dim, sorted_1dim)))
time_bins = 0.5
max_time = sorted_dim_all.max() + time_bins
range_time = np.arange(0,max_time, 0.5)
survival = [np.count_nonzero(sorted_dim_all >= x) for x in range_time]
survival_pt5 = [np.count_nonzero(sorted_pt5dim >= x) for x in range_time]
survival_1 = [np.count_nonzero(sorted_1 >= x) for x in range_time]
with np.errstate(divide='raise'):
    survival_log = np.log10(survival)
    survival_ln = np.log(survival)
line_model_abs = curve_fit(line, range_time, survival_ln)
# decay_abs_line_fit = line(range_time, *line_model_abs[0],)
line_reg_abs = scipy.stats.linregress(range_time, survival_ln)
decay_abs_line_fit = line(range_time, line_reg_abs[0], line_reg_abs[1])
half_life = np.log(2) / (-1 * line_reg_abs[0])

survival_prct = [np.count_nonzero(sorted_dim_all >= x) / len(sorted_dim_all) for x in range_time]
survival_pt5_prct = [np.count_nonzero(sorted_pt5dim >= x) / len(sorted_pt5dim) for x in range_time]
survival_1_prct = [np.count_nonzero(sorted_1 >= x) / len(sorted_1) for x in range_time]
with np.errstate(divide='raise'):
    survival_prct_log = np.log10(survival_prct)
    survival_prct_ln = np.log(survival_prct)
line_model_prct = curve_fit(line, range_time, survival_prct_ln)
# decay_prct_line_fit = line(range_time, *line_model_prct[0],)
line_reg_prct = scipy.stats.linregress(range_time, survival_prct_ln)
decay_prct_line_fit = line(range_time, line_reg_prct[0], line_reg_prct[1])



# Survival for eedb:
sorted_pt5_eedb = np.sort(df_flattened_expanded_sf_eedb.loc[df_flattened_expanded_sf_eedb[df_flattened_expanded_sf_eedb['time_intervals']==0.5].index]['contig_touch'])
sorted_pt5_eedb_no_high_adh = np.sort(df_flattened_expanded_sf_eedb_no_high_adh.loc[df_flattened_expanded_sf_eedb_no_high_adh[df_flattened_expanded_sf_eedb_no_high_adh['time_intervals']==0.5].index]['contig_touch'])
sorted_dim_all_eedb = sorted_pt5_eedb*0.5 # provides duration of contact in minutes
sorted_dim_no_high_adh = sorted_pt5_eedb_no_high_adh*0.5
# sorted_1_eedb = np.sort(df_flattened_expanded_sf_eedb.loc[df_flattened_expanded_sf_eedb[df_flattened_expanded_sf_eedb['time_intervals']==1.0].index]['contig_touch'])
# sorted_1dim_eedb = sorted_1_eedb.copy()
# sorted_dim_all_eedb = np.sort(np.concatenate((sorted_pt5dim_eedb, sorted_1dim_eedb)))
time_bins = 0.5
max_time_eedb = sorted_dim_all_eedb.max() + time_bins
range_time_eedb = np.arange(0,max_time_eedb, 0.5)
survival_eedb = [np.count_nonzero(sorted_dim_all_eedb >= x) for x in range_time_eedb]
range_time_eedb_no_high_adh = np.arange(0,sorted_dim_no_high_adh.max() + time_bins, 0.5)
survival_eedb_no_high_adh = [np.count_nonzero(sorted_dim_no_high_adh >= x) for x in range_time_eedb_no_high_adh]
with np.errstate(divide='raise'):
    survival_log_eedb = np.log10(survival_eedb)
    survival_ln_eedb = np.log(survival_eedb)
    survival_ln_eedb_no_high_adh = np.log(survival_eedb_no_high_adh)
# line_model_abs_eedb = curve_fit(line, range_time_eedb, survival_ln_eedb)
# decay_abs_line_fit = line(range_time, *line_model_abs[0],)
line_reg_abs_eedb = scipy.stats.linregress(range_time_eedb, survival_ln_eedb)
decay_abs_line_fit_eedb = line(range_time_eedb, line_reg_abs_eedb[0], line_reg_abs_eedb[1])
half_life_eedb = np.log(2) / (-1 * line_reg_abs_eedb[0])

line_reg_abs_eedb_no_high_adh = scipy.stats.linregress(range_time_eedb_no_high_adh, survival_ln_eedb_no_high_adh)
decay_abs_line_fit_eedb_no_high_adh = line(range_time_eedb_no_high_adh, line_reg_abs_eedb_no_high_adh[0], line_reg_abs_eedb_no_high_adh[1])
half_life_eedb_no_high_adh = np.log(2) / (-1 * line_reg_abs_eedb_no_high_adh[0])


survival_prct_eedb = [np.count_nonzero(sorted_dim_all_eedb >= x) / len(sorted_dim_all_eedb) for x in range_time_eedb]
survival_prct_eedb_no_high_adh = [np.count_nonzero(sorted_dim_no_high_adh >= x) / len(sorted_dim_no_high_adh) for x in range_time_eedb_no_high_adh]
# survival_pt5_eedb = [np.count_nonzero(sorted_pt5dim_eedb >= x) for x in range_time_eedb]
# survival_1_eedb = [np.count_nonzero(sorted_1_eedb >= x) for x in range_time_eedb]
with np.errstate(divide='raise'):
    survival_log_eedb = np.log10(survival_eedb)
    survival_ln_eedb = np.log(survival_eedb)
    survival_reciprocal_eedb = 1/np.asarray(survival_eedb)
    
# line_model_abs_eedb = curve_fit(line, range_time_eedb, survival_ln_eedb)
# decay_abs_line_fit = line(range_time, *line_model_abs[0],)
line_reg_abs_eedb = scipy.stats.linregress(range_time_eedb, survival_ln_eedb)
# line_reg_abs_eedb = scipy.stats.linregress(range_time_eedb, survival_reciprocal_eedb)
decay_abs_line_fit_eedb = line(range_time_eedb, line_reg_abs_eedb[0], line_reg_abs_eedb[1])



fig, ax = plt.subplots(figsize=(width_1col, width_1col))
ax.semilogy()#np.exp(1))
[ax.axhline(x, c='lightgrey', lw=1) for x in range(1,6)]
ax.xaxis.set_minor_locator(ml_minor)
ax.xaxis.set_tick_params(which='both')
ax.plot(range_time, survival_pt5, c='tab:orange', marker='.', zorder=8, label='30s acq. interv.')
ax.plot(range_time, survival_1, c='b', marker='.', zorder=9, label='60s acq. interv.')
ax.plot(range_time, survival, c='k', marker='.', zorder=10, label='30s & 60s')
ax.plot(range_time, np.exp(decay_abs_line_fit), c='r', label='line fit')
ax.set_xlim(0, range_time.max())
ax.set_ylim(1)
ax.legend()
fig.savefig(Path.joinpath(output_dir, r"ee_contig_touch_all_survival_abs_y.tif"), dpi=dpi_graph)
plt.close(fig)

# fig, ax = plt.subplots(figsize=(width_1col, width_1col))
# ax.semilogy()#np.exp(1))
# ax.xaxis.set_minor_locator(ml_minor)
# ax.xaxis.set_tick_params(which='both')
# ax.plot(range_time, survival_pt5_prct, c='orange', marker='.', zorder=8, label='30s acq. interv.')
# ax.plot(range_time, survival_1_prct, c='b', marker='.', zorder=9, label='60s acq. interv.')
# ax.plot(range_time, survival_prct, c='k', marker='.', zorder=10, label='30s & 60s')
# ax.plot(range_time, np.exp(decay_prct_line_fit), c='g', label='line fit')
# ax.set_xlim(0, range_time.max())
# ax.set_ylim(1e-2, 1)
# ax.legend()
# fig.savefig(Path.joinpath(output_dir, r"ee_contig_touch_all_survival_prct_y.tif"), dpi=dpi_graph)
# plt.close(fig)

# Decay plot for eedb:
fig, ax = plt.subplots(figsize=(width_1col, width_1col))
ax.semilogy()#np.exp(1))
[ax.axhline(x, c='lightgrey', lw=1) for x in range(1,6)]
ax.xaxis.set_minor_locator(ml_minor)
ax.xaxis.set_tick_params(which='both')
ax.plot(range_time_eedb, survival_eedb, c='k', marker='.', zorder=10, label='eedb all')
ax.plot(range_time_eedb, np.exp(decay_abs_line_fit_eedb), c='r', ls='-', label='line fit')
ax.plot(range_time_eedb_no_high_adh, survival_eedb_no_high_adh, c='grey', marker='.', zorder=9, label='eedb no high adh.')
ax.plot(range_time_eedb_no_high_adh, np.exp(decay_abs_line_fit_eedb_no_high_adh), c='m', ls='-', label='no high adh. line fit')
ax.set_ylim(1)
# ax.plot(range_time_eedb, survival_reciprocal_eedb, c='k', marker='.', zorder=10, label='30s acq. interv.')
# ax.plot(range_time_eedb, decay_abs_line_fit_eedb, c='g', label='line fit')
# ax.set_ylim(0)
ax.set_xlim(0, range_time_eedb.max())
ax.legend()
fig.savefig(Path.joinpath(output_dir, r"eedb_contig_touch_all_survival_abs_y.tif"), dpi=dpi_graph)
plt.close(fig)


# Decay plot for both ee_low_plat and eedb:
fig, ax = plt.subplots(figsize=(width_2col/3*0.98, width_2col/3*0.98))
ax.semilogy()#np.exp(1))
[ax.axhline(x, c='lightgrey', lw=1) for x in range(1,6)]
ax.xaxis.set_minor_locator(ml_minor2)
ax.xaxis.set_tick_params(which='both')
ax.plot(range_time_eedb, survival_eedb, c='k', marker='.', zorder=10, label='no Ca\N{SUPERSCRIPT two}\N{superscript plus sign}')
ax.plot(range_time_eedb, np.exp(decay_abs_line_fit_eedb), c='r', ls='-', label='fit no Ca\N{SUPERSCRIPT two}\N{superscript plus sign}')
# ax.plot(range_time_eedb_no_high_adh, survival_eedb_no_high_adh, c='k', marker='.', zorder=10, label='no Ca\N{SUPERSCRIPT two}\N{superscript plus sign}')
# ax.plot(range_time_eedb_no_high_adh, np.exp(decay_abs_line_fit_eedb_no_high_adh), c='r', ls='-', label='fit no Ca\N{SUPERSCRIPT two}\N{superscript plus sign}')
ax.plot(range_time, survival, c='g', marker='.', zorder=10, label='Ca\N{SUPERSCRIPT two}\N{superscript plus sign}')
ax.plot(range_time, np.exp(decay_abs_line_fit), c='tab:orange', label='fit Ca\N{SUPERSCRIPT two}\N{superscript plus sign}')
ax.set_ylim(1)
ax.set_xlim(0, range_time_eedb.max())
ax.legend()
fig.savefig(Path.joinpath(output_dir, r"ee_eedb_contig_touch_all_survival_abs_y.tif"), dpi=dpi_graph, bbox_inches='tight')
plt.close(fig)




# Output the regression information:
decay_linregress_out = [['Experimental_Condition', 'slope', 'intercept', 'rvalue', 'pvalue', 'stderr', 'half-life']]
decay_linregress_out.append(['ecto-ecto low plateau'] + list(line_reg_abs) + [half_life])
decay_linregress_out.append(['eedb'] + list(line_reg_abs_eedb) + [half_life_eedb])
decay_linregress_out.append(['eedb no high adh.'] + list(line_reg_abs_eedb_no_high_adh) + [half_life_eedb_no_high_adh])
with open(Path.joinpath(output_dir, r'eelowplat_and_eedb_contact_decay.tsv'), 'w', 
          newline='') as file_out:
    tsv_obj = csv.writer(file_out, delimiter='\t')
    tsv_obj.writerows(decay_linregress_out)



### When are more than 90% of the contacts broken?



###############################################################################

### Run some statistical tests to see if there is statistical evidence that the
### living cell low plateau data is different from the fixed low plateau data:
# Will also be outputting a list of p-values:
low_plat_stat_contnd = [('percent growth allowed', 'statistical test of contnd of eeLowPlat vs eedb_fix')]
low_plat_stat_angle = [('percent growth allowed', 'statistical test of angle of eeLowPlat vs eedb_fix')]


# Is the low plateau growth significantly different from the eedb fixed sample?:
Foneway_eeLowPlat_vs_eedb_contnd = stats.f_oneway(eedb_fix_contact_diam_non_dim_df_list_long_dropna['Contact Surface Length (N.D.)'], ee_low_plat_contnd_df['Contact Surface Length (N.D.)'])
Foneway_eeLowPlat_vs_eedb_angle = stats.f_oneway(eedb_fix_contact_diam_non_dim_df_list_long_dropna['Average Angle (deg)'], ee_low_plat_contnd_df['Average Angle (deg)'])
# Answer: yes.

# If you only allow 2.5% growth in the low plateau is the effect still significant?:
Foneway_eeLowPlat_vs_eedb_contnd_strict = stats.f_oneway(eedb_fix_contact_diam_non_dim_df_list_long_dropna['Contact Surface Length (N.D.)'], ee_low_plat_contnd_strict_df['Contact Surface Length (N.D.)'])
Foneway_eeLowPlat_vs_eedb_angle_strict = stats.f_oneway(eedb_fix_contact_diam_non_dim_df_list_long_dropna['Average Angle (deg)'], ee_low_plat_contnd_strict_df['Average Angle (deg)'])
# Answer: yes.

# Is the low plateau growth significantly different from the eedb fixed sample
# when you are only using verified low plateaus and allowing 5% growth on 
# either end of the fitted curve?:
Foneway_eeLowPlatVerified_vs_eedb_contnd = stats.f_oneway(eedb_fix_contact_diam_non_dim_df_list_long_dropna['Contact Surface Length (N.D.)'], ee_low_plat_contnd_verified_df['Contact Surface Length (N.D.)'])
Foneway_eeLowPlatVerified_vs_eedb_angle = stats.f_oneway(eedb_fix_contact_diam_non_dim_df_list_long_dropna['Average Angle (deg)'], ee_low_plat_contnd_verified_df['Average Angle (deg)'])
low_plat_stat_contnd.append(('5_percent', Foneway_eeLowPlatVerified_vs_eedb_contnd))
low_plat_stat_angle.append(('5_percent', Foneway_eeLowPlatVerified_vs_eedb_angle))
# Answer: yes.

# Is the low plateau growth significantly different from the eedb fixed sample
# when you are only using verified low plateaus AND allowing only 2.5% growth
# on either end of the fitted curve?:
Foneway_eeLowPlatVerified_vs_eedb_contnd_strict = stats.f_oneway(eedb_fix_contact_diam_non_dim_df_list_long_dropna['Contact Surface Length (N.D.)'], ee_low_plat_contnd_verified_strict_df['Contact Surface Length (N.D.)'])
Foneway_eeLowPlatVerified_vs_eedb_angle_strict = stats.f_oneway(eedb_fix_contact_diam_non_dim_df_list_long_dropna['Average Angle (deg)'], ee_low_plat_contnd_verified_strict_df['Average Angle (deg)'])
low_plat_stat_contnd.append(('2.5_percent', Foneway_eeLowPlatVerified_vs_eedb_contnd_strict))
low_plat_stat_angle.append(('2.5_percent', Foneway_eeLowPlatVerified_vs_eedb_angle_strict))
# Answer: yes.

# Is the low plateau growth significantly different from the eedb fixed sample
# when you are only using verified low plateaus AND allowing only 1.0% growth
# on either end of the fitted curve?:
Foneway_eeLowPlatVerified_vs_eedb_contnd_v_strict = stats.f_oneway(eedb_fix_contact_diam_non_dim_df_list_long_dropna['Contact Surface Length (N.D.)'], ee_low_plat_contnd_verified_v_strict_df['Contact Surface Length (N.D.)'])
Foneway_eeLowPlatVerified_vs_eedb_angle_v_strict = stats.f_oneway(eedb_fix_contact_diam_non_dim_df_list_long_dropna['Average Angle (deg)'], ee_low_plat_contnd_verified_v_strict_df['Average Angle (deg)'])
low_plat_stat_contnd.append(('1_percent', Foneway_eeLowPlatVerified_vs_eedb_contnd_v_strict))
low_plat_stat_angle.append(('1_percent', Foneway_eeLowPlatVerified_vs_eedb_angle_v_strict))
# Answer: yes.

# Is the low plateau growth significantly different from the eedb fixed sample
# when you are only using verified low plateaus AND allowing only 0.1% growth
# (this is less than the smallest upward increment, therefore there is no real
# growth)
# on either end of the fitted curve?:
Foneway_eeLowPlatVerified_vs_eedb_contnd_absolute = stats.f_oneway(eedb_fix_contact_diam_non_dim_df_list_long_dropna['Contact Surface Length (N.D.)'], ee_low_plat_contnd_verified_absolute_df['Contact Surface Length (N.D.)'])
Foneway_eeLowPlatVerified_vs_eedb_angle_absolute = stats.f_oneway(eedb_fix_contact_diam_non_dim_df_list_long_dropna['Average Angle (deg)'], ee_low_plat_contnd_verified_absolute_df['Average Angle (deg)'])
low_plat_stat_contnd.append(('0.1_percent', Foneway_eeLowPlatVerified_vs_eedb_contnd_absolute))
low_plat_stat_angle.append(('0.1_percent', Foneway_eeLowPlatVerified_vs_eedb_angle_absolute))
# Answer: yes.

# Is the low plateau growth significantly different from the eedb fixed sample
# when you are only using manually defined low plateaus (selected so that there
# is no growth):
Foneway_eeLowPlatVerified_vs_eedb_contnd_absolute = stats.f_oneway(eedb_fix_contact_diam_non_dim_df_list_long_dropna['Contact Surface Length (N.D.)'], ee_low_plat_contnd_manual_absolute_df['Contact Surface Length (N.D.)'])
Foneway_eeLowPlatVerified_vs_eedb_angle_absolute = stats.f_oneway(eedb_fix_contact_diam_non_dim_df_list_long_dropna['Average Angle (deg)'], ee_low_plat_angle_manual_absolute_df['Average Angle (deg)'])
low_plat_stat_contnd.append(('0_percent (manual)', Foneway_eeLowPlatVerified_vs_eedb_contnd_absolute))
low_plat_stat_angle.append(('0_percent (manual)', Foneway_eeLowPlatVerified_vs_eedb_angle_absolute))
# Answer: yes.



# Output the statistical results into a .tsv:
with open(Path.joinpath(output_dir, r'eedbFix_vs_eeLowPlat_stats_contnd.tsv'), 'w', 
          newline='') as file_out:
    tsv_obj = csv.writer(file_out, delimiter='\t')
    tsv_obj.writerows(low_plat_stat_contnd)

with open(Path.joinpath(output_dir, r'eedbFix_vs_eeLowPlat_stats_angle.tsv'), 'w', 
          newline='') as file_out:
    tsv_obj = csv.writer(file_out, delimiter='\t')
    tsv_obj.writerows(low_plat_stat_angle)



###############################################################################


### Show how the initial radii for ecto-ecto are distributed:    
two_circ_interp_angle = interpolate.interp1d(two_circ_list_long['Area of cell 1 (Red) (px**2)'], two_circ_list_long['Average Angle (deg)'], fill_value="extrapolate")
two_circ_interp_cont = interpolate.interp1d(two_circ_list_long['Area of cell 1 (Red) (px**2)'], two_circ_list_long['Contact Surface Length (px)'], fill_value="extrapolate")
two_circ_interp_contnd = interpolate.interp1d(two_circ_list_original_nd_df['Area of cell 1 (Red) (px**2)'], two_circ_list_original_nd_df['Contact Surface Length (N.D.)'], fill_value="extrapolate")


# Plot the cleaned up low plateaus, high plateaus, and logistic growth constants
# for the angles:




# Plot the cleaned up low plateaus for the non-dimensionalized contacts when 
# allowing 0% growth (using human judgement) in low adhesion data:
plt.clf()
kde_low_angle = sns.distplot(ee_low_plat_angle_manual_absolute_df['Average Angle (deg)'], kde=True, hist_kws=dict(edgecolor="k", linewidth=2), rug=True)
kde_low_angle_x = kde_low_angle.lines[0].get_xdata()
kde_low_angle_y = kde_low_angle.lines[0].get_ydata()
kde_low_angle_maxid = signal.find_peaks(kde_low_angle_y)[0]
kde_low_angle_peaks_y = kde_low_angle_y[kde_low_angle_maxid]
kde_low_angle_peaks = kde_low_angle_x[kde_low_angle_maxid]
plt.clf()

angle = sns.distplot(eedb_fix_contact_diam_non_dim_df_list_long_dropna['Average Angle (deg)'], kde=True, hist_kws=dict(edgecolor="k", linewidth=2), rug=True)#, ax=kde_low)#, kde_kws=dict(bw=5))
angle_x = angle.lines[0].get_xdata()
angle_y = angle.lines[0].get_ydata()
angle_maxid = signal.find_peaks(angle_y)[0]
eedb_fix_angle_peaks_y = angle_y[angle_maxid]
eedb_fix_angle_peaks = angle_x[angle_maxid]
plt.clf()

eedb_no_high_adh_angle = sns.distplot(eedb_contact_diam_non_dim_df_list_long_dropna_no_high_adh['Average Angle (deg)'], kde=True, hist_kws=dict(edgecolor="k", linewidth=2), rug=True)#, ax=kde_low)#, kde_kws=dict(bw=5))
eedb_no_high_adh_angle_x = eedb_no_high_adh_angle.lines[0].get_xdata()
eedb_no_high_adh_angle_y = eedb_no_high_adh_angle.lines[0].get_ydata()
eedb_no_high_adh_angle_maxid = signal.find_peaks(eedb_no_high_adh_angle_y)[0]
eedb_no_high_adh_angle_peaks_y = eedb_no_high_adh_angle_y[eedb_no_high_adh_angle_maxid]
eedb_no_high_adh_angle_peaks = eedb_no_high_adh_angle_x[eedb_no_high_adh_angle_maxid]
plt.clf()

try:
    kde_radius_init_2_circ_angle = sns.distplot(two_circ_interp_angle(ee_first_valid_area_low_only_angle), kde=True)#, hist=False)
    kde_radius_init_2_circ_angle_x = kde_radius_init_2_circ_angle.lines[0].get_xdata()
    kde_radius_init_2_circ_angle_y = kde_radius_init_2_circ_angle.lines[0].get_ydata()
    kde_radius_init_2_circ_angle_maxid = signal.find_peaks(kde_radius_init_2_circ_angle_y)[0]
    kde_radius_init_2_circ_angle_peaks_y = kde_radius_init_2_circ_angle_y[kde_radius_init_2_circ_angle_maxid]
    kde_radius_init_2_circ_angle_peaks = kde_radius_init_2_circ_angle_x[kde_radius_init_2_circ_angle_maxid]
except(RuntimeError): 
    # the angles have a larger step-size between values, and as a result the 
    # low adhesion expected values are (almost) all the same. In such a case 
    # you can't get a KDE so instead we will just take the mode and plot that
    kde_radius_init_2_circ_angle_peaks = stats.mode(two_circ_interp_angle(ee_first_valid_area_low_only_angle)).mode
plt.clf()


distrib_ee_low_plat_angle_mean = np.mean(ee_low_plat_angle_manual_absolute_df['Average Angle (deg)'])
distrib_ee_low_plat_angle_median = np.median(ee_low_plat_angle_manual_absolute_df['Average Angle (deg)'])
distrib_ee_low_plat_angle_std = np.std(ee_low_plat_angle_manual_absolute_df['Average Angle (deg)'])

distrib_eedbfix_angle_mean = np.mean(eedb_fix_contact_diam_non_dim_df_list_long_dropna['Average Angle (deg)'])
distrib_eedbfix_angle_median = np.mean(eedb_fix_contact_diam_non_dim_df_list_long_dropna['Average Angle (deg)'])
distrib_eedbfix_angle_std = np.std(eedb_fix_contact_diam_non_dim_df_list_long_dropna['Average Angle (deg)'])

distrib_eedb_angle_mean = np.mean(eedb_contact_diam_non_dim_df_list_long_dropna_no_high_adh['Average Angle (deg)'])
distrib_eedb_angle_median = np.mean(eedb_contact_diam_non_dim_df_list_long_dropna_no_high_adh['Average Angle (deg)'])
distrib_eedb_angle_std = np.std(eedb_contact_diam_non_dim_df_list_long_dropna_no_high_adh['Average Angle (deg)'])

distrib_two_circ_angle_mean = np.mean(two_circ_interp_angle(ee_first_valid_area_low_only_angle))
distrib_two_circ_angle_median = np.mean(two_circ_interp_angle(ee_first_valid_area_low_only_angle))
distrib_two_circ_angle_std = np.std(two_circ_interp_angle(ee_first_valid_area_low_only_angle))

xlim = max(ee_low_plat_angle_manual_absolute_df['Average Angle (deg)'].max(), 
           eedb_fix_contact_diam_non_dim_df_list_long_dropna['Average Angle (deg)'].max(), 
           eedb_contact_diam_non_dim_df_list_long_dropna_no_high_adh['Average Angle (deg)'].max(), 
           two_circ_interp_angle(ee_first_valid_area_low_only_angle).max())
xlim = round_up_to_multiple(xlim, 10)

fig_kde_low_angle, (kde_low_angle_top, kde_low_angle_bot) = plt.subplots(figsize=(width_1col*2/3,height_1col*2/3), nrows=2, ncols=1, sharex=True)
kde_low_angle_top = sns.distplot(ee_low_plat_angle_manual_absolute_df['Average Angle (deg)'], kde=True, hist_kws=dict(color='g', edgecolor='g', linewidth=0.5), kde_kws=dict(color='g', linewidth=0.5), ax=kde_low_angle_top)#, rug=True, rug_kws=dict(color='k', alpha=0.5))
angle_eedb_fix = sns.distplot(eedb_fix_contact_diam_non_dim_df_list_long_dropna['Average Angle (deg)'], kde=True, hist_kws=dict(color='orange', edgecolor='orange', linewidth=0.5), kde_kws=dict(color='orange', linewidth=0.5), ax=kde_low_angle_top)#, rug=True, rug_kws=dict(alpha=0.5))#, ax=kde_low)#, kde_kws=dict(bw=5))
# angle_eedb = sns.distplot(eedb_contact_diam_non_dim_df_list_long_dropna['Average Angle (deg)'], kde=True, hist_kws=dict(color='k', edgecolor='k', linewidth=2), kde_kws=dict(color='k', linewidth=0.5))
angle_eedb_no_high_adh = sns.distplot(eedb_contact_diam_non_dim_df_list_long_dropna_no_high_adh['Average Angle (deg)'], kde=True, hist_kws=dict(color='k', edgecolor='k', linewidth=0.5), kde_kws=dict(color='k', linewidth=0.5), ax=kde_low_angle_bot)
kde_radius_init_2_circ_angle = kde_low_angle_bot.twinx()
## Try violinplot instead?
# kde_low_angle_top = sns.boxplot(ee_low_plat_angle_manual_absolute_df['Average Angle (deg)'], color='g', edgecolor='g', linewidth=2), kde_kws=dict(color='g'), ax=kde_low_angle_top)#, rug=True, rug_kws=dict(color='k', alpha=0.5))
# angle_eedb_fix = sns.distplot(eedb_fix_contact_diam_non_dim_df_list_long_dropna['Average Angle (deg)'], kde=True, hist_kws=dict(color='orange', edgecolor='orange', linewidth=2), kde_kws=dict(color='orange'), ax=kde_low_angle_top)#, rug=True, rug_kws=dict(alpha=0.5))#, ax=kde_low)#, kde_kws=dict(bw=5))
try:
    sns.distplot(two_circ_interp_angle(ee_first_valid_area_low_only_angle), hist=False, ax = kde_radius_init_2_circ_angle, kde_kws=dict(color='r', linewidth=0.5))
except(RuntimeError):
    sns.distplot(two_circ_interp_angle(ee_first_valid_area_low_only_angle), hist=True, kde=False, ax = kde_radius_init_2_circ_angle, hist_kws=dict(color='r', edgecolor='r', linewidth=0.5))
    # pass # skip plotting the distribution if there is no KDE
# kde_low_angle_top.axvline(distrib_ee_low_plat_angle_mean, c='g', ls='-', zorder=10)
# kde_low_angle_top.axvline(distrib_eedbfix_angle_mean, c='orange', ls='-', zorder=10)
# kde_low_angle_bot.axvline(distrib_eedb_angle_mean, c='k', ls='-', zorder=10)
# kde_low_angle_bot.axvline(distrib_two_circ_angle_mean, c='r', ls='-', zorder=10)

# kde_low_nd_top.scatter(kde_low_cont_nd_peaks, kde_low_cont_nd_peaks_y, c='g', marker='.', s=50, zorder=10)
# kde_low_nd_bot.scatter(eedb_fix_cont_nd_peaks, eedb_fix_cont_nd_peaks_y, c='orange', marker='.', s=50, zorder=10)
# kde_low_nd_top.scatter(eedb_no_high_adh_cont_nd_peaks, eedb_no_high_adh_cont_nd_peaks_y, c='k', marker='.', s=50, zorder=10)
# [kde_low_nd_top.axvline(x, c='k', ls=':', zorder=10) for x in eedb_no_high_adh_cont_nd_peaks]
# [kde_low_angle_top.axvline(x, c='g', ls=':', zorder=9) for x in kde_low_angle_peaks]
# [kde_low_angle_top.axvline(x, c='orange', ls=':', zorder=10) for x in eedb_fix_angle_peaks]
# [kde_low_nd_top.axvline(x, c='r', zorder=9) for x in kde_radius_init_2_circ_nd_peaks]
# [kde_low_angle_bot.axvline(x, c='k', ls=':', zorder=10) for x in eedb_no_high_adh_angle_peaks]
# [kde_low_nd_bot.axvline(x, c='g', zorder=9) for x in kde_low_cont_nd_peaks]
# [kde_low_nd_bot.axvline(x, c='orange', ls=':', zorder=10) for x in eedb_fix_cont_nd_peaks]
# [kde_low_angle_bot.axvline(x, c='r', ls=':', zorder=9) for x in kde_radius_init_2_circ_angle_peaks]
kde_low_angle_bot.set_xlabel('')
kde_low_angle_top.set_xlabel('')
kde_low_angle_bot.set_yticks([0,0.01, 0.02])
kde_radius_init_2_circ_angle.set_yticks([0,15,30])
kde_low_angle_top.set_yticks([0,0.01, 0.02])
# kde_low_nd_top.set_x
#kde_radius_init_2_circ.yaxis.label.set_color('green')
kde_radius_init_2_circ_angle.tick_params(axis='y', colors='red')
kde_radius_init_2_circ_angle.set_ylim(0)
kde_low_angle_top.set_xlim(0, xlim)
kde_low_angle_top.set_ylim(0)
kde_low_angle_bot.set_ylim(0)
fig_kde_low_angle.draw
fig_kde_low_angle.savefig(Path.joinpath(output_dir, r"ee_angle_data_less_than_growth_start_manual_low_data_no_high_adh.tif"), 
                          dpi=dpi_graph, bbox_inches='tight')
# print('ecto-ecto (black): ', kde_low_cont_nd_peaks)
# print('ecto-ecto_fix (orange): ', eedb_fix_cont_nd_peaks)
#print('2_circles (red): ', kde_radius_init_2_circ_nd_peaks)
plt.close(fig_kde_low_angle)



fig_kde_low_angle, (kde_low_angle_top, kde_low_angle_bot) = plt.subplots(figsize=(width_1col*2/3,height_1col*2/3), nrows=2, ncols=1, sharex=True)
kde_low_angle_top = sns.distplot(ee_low_plat_angle_manual_absolute_df['Average Angle (deg)'], kde=True, hist_kws=dict(color='g', edgecolor='g', linewidth=0.5), kde_kws=dict(color='g', linewidth=0.5), ax=kde_low_angle_top)#, rug=True, rug_kws=dict(color='k', alpha=0.5))
angle_eedb_fix = sns.distplot(eedb_fix_contact_diam_non_dim_df_list_long_dropna['Average Angle (deg)'], kde=True, hist_kws=dict(color='orange', edgecolor='orange', linewidth=0.5), kde_kws=dict(color='orange', linewidth=0.5), ax=kde_low_angle_top)#, rug=True, rug_kws=dict(alpha=0.5))#, ax=kde_low)#, kde_kws=dict(bw=5))
angle_eedb = sns.distplot(eedb_contact_diam_non_dim_df_list_long_dropna['Average Angle (deg)'], kde=True, hist_kws=dict(color='k', edgecolor='k', linewidth=0.5), kde_kws=dict(color='k', linewidth=0.5), ax=kde_low_angle_bot)
# angle_eedb_no_high_adh = sns.distplot(eedb_contact_diam_non_dim_df_list_long_dropna_no_high_adh['Average Angle (deg)'], kde=True, hist_kws=dict(color='k', edgecolor='k', linewidth=0.5), kde_kws=dict(color='k', linewidth=0.5), ax=kde_low_angle_bot)
kde_radius_init_2_circ_angle = kde_low_angle_bot.twinx()
## Try violinplot instead?
# kde_low_angle_top = sns.boxplot(ee_low_plat_angle_manual_absolute_df['Average Angle (deg)'], color='g', edgecolor='g', linewidth=2), kde_kws=dict(color='g'), ax=kde_low_angle_top)#, rug=True, rug_kws=dict(color='k', alpha=0.5))
# angle_eedb_fix = sns.distplot(eedb_fix_contact_diam_non_dim_df_list_long_dropna['Average Angle (deg)'], kde=True, hist_kws=dict(color='orange', edgecolor='orange', linewidth=2), kde_kws=dict(color='orange'), ax=kde_low_angle_top)#, rug=True, rug_kws=dict(alpha=0.5))#, ax=kde_low)#, kde_kws=dict(bw=5))
try:
    sns.distplot(two_circ_interp_angle(ee_first_valid_area_low_only_angle), hist=False, ax = kde_radius_init_2_circ_angle, kde_kws=dict(color='r', linewidth=0.5))
except(RuntimeError):
    sns.distplot(two_circ_interp_angle(ee_first_valid_area_low_only_angle), hist=True, kde=False, ax = kde_radius_init_2_circ_angle, hist_kws=dict(color='r', edgecolor='r', linewidth=0.5))
    # pass # skip plotting the distribution if there is no KDE
# kde_low_angle_top.axvline(distrib_ee_low_plat_angle_mean, c='g', ls='-', zorder=10)
# kde_low_angle_top.axvline(distrib_eedbfix_angle_mean, c='orange', ls='-', zorder=10)
# kde_low_angle_bot.axvline(distrib_eedb_angle_mean, c='k', ls='-', zorder=10)
# kde_low_angle_bot.axvline(distrib_two_circ_angle_mean, c='r', ls='-', zorder=10)

# kde_low_nd_top.scatter(kde_low_cont_nd_peaks, kde_low_cont_nd_peaks_y, c='g', marker='.', s=50, zorder=10)
# kde_low_nd_bot.scatter(eedb_fix_cont_nd_peaks, eedb_fix_cont_nd_peaks_y, c='orange', marker='.', s=50, zorder=10)
# kde_low_nd_top.scatter(eedb_no_high_adh_cont_nd_peaks, eedb_no_high_adh_cont_nd_peaks_y, c='k', marker='.', s=50, zorder=10)
# [kde_low_nd_top.axvline(x, c='k', ls=':', zorder=10) for x in eedb_no_high_adh_cont_nd_peaks]
# [kde_low_angle_top.axvline(x, c='g', ls=':', zorder=9) for x in kde_low_angle_peaks]
# [kde_low_angle_top.axvline(x, c='orange', ls=':', zorder=10) for x in eedb_fix_angle_peaks]
# [kde_low_nd_top.axvline(x, c='r', zorder=9) for x in kde_radius_init_2_circ_nd_peaks]
# [kde_low_angle_bot.axvline(x, c='k', ls=':', zorder=10) for x in eedb_no_high_adh_angle_peaks]
# [kde_low_nd_bot.axvline(x, c='g', zorder=9) for x in kde_low_cont_nd_peaks]
# [kde_low_nd_bot.axvline(x, c='orange', ls=':', zorder=10) for x in eedb_fix_cont_nd_peaks]
# [kde_low_angle_bot.axvline(x, c='r', ls=':', zorder=9) for x in kde_radius_init_2_circ_angle_peaks]
kde_low_angle_bot.set_xlabel('')
kde_low_angle_top.set_xlabel('')
kde_low_angle_bot.set_yticks([0,0.01, 0.02])
kde_radius_init_2_circ_angle.set_yticks([0,15,30])
kde_low_angle_top.set_yticks([0,0.01, 0.02])
# kde_low_nd_top.set_x
#kde_radius_init_2_circ.yaxis.label.set_color('green')
kde_radius_init_2_circ_angle.tick_params(axis='y', colors='red')
kde_radius_init_2_circ_angle.set_ylim(0)
kde_low_angle_top.set_xlim(0, xlim)
kde_low_angle_top.set_ylim(0)
kde_low_angle_bot.set_ylim(0)
fig_kde_low_angle.draw
fig_kde_low_angle.savefig(Path.joinpath(output_dir, r"ee_angle_data_less_than_growth_start_manual_low_data.tif"), 
                          dpi=dpi_graph, bbox_inches='tight')
# print('ecto-ecto (black): ', kde_low_cont_nd_peaks)
# print('ecto-ecto_fix (orange): ', eedb_fix_cont_nd_peaks)
#print('2_circles (red): ', kde_radius_init_2_circ_nd_peaks)
plt.close(fig_kde_low_angle)







# Output the statistical results into a .tsv:
low_adh_stats_headers = ('sample 1', 'sample 2', 'F_onewayResult')
eelow_vs_eedb = ('ee_low_plat', 'eedb',  
                 stats.f_oneway(ee_low_plat_angle_manual_absolute_df['Average Angle (deg)'], eedb_contact_diam_non_dim_df_list_long_dropna_no_high_adh['Average Angle (deg)']))
eelow_vs_eedbfix = ('ee_low_plat', 'eedb_fix', 
                    stats.f_oneway(ee_low_plat_angle_manual_absolute_df['Average Angle (deg)'], eedb_fix_contact_diam_non_dim_df_list_long_dropna['Average Angle (deg)']))
eelow_vs_twocirc = ('ee_low_plat', 'two_circle_interp', 
                    stats.f_oneway(ee_low_plat_angle_manual_absolute_df['Average Angle (deg)'], two_circ_interp_angle(ee_first_valid_area_low_only_angle)))
eedb_vs_eedbfix = ('eedb', 'eedb_fix', 
                   stats.f_oneway(eedb_contact_diam_non_dim_df_list_long_dropna_no_high_adh['Average Angle (deg)'], eedb_fix_contact_diam_non_dim_df_list_long_dropna['Average Angle (deg)']))
eedbfix_vs_twocirc = ('eedb_fix', 'two_circ_interp', 
                      stats.f_oneway(eedb_fix_contact_diam_non_dim_df_list_long_dropna['Average Angle (deg)'], two_circ_interp_angle(ee_first_valid_area_low_only_angle)))
low_adh_stats_angle = [low_adh_stats_headers, eelow_vs_eedb, 
                       eelow_vs_eedbfix, eelow_vs_twocirc, 
                       eedb_vs_eedbfix, eedbfix_vs_twocirc]


with open(Path.joinpath(output_dir, r'ee_angle_data_less_than_growth_manual_stats.tsv'), 'w', 
          newline='') as file_out:
    tsv_obj = csv.writer(file_out, delimiter='\t')
    tsv_obj.writerows(low_adh_stats_angle)




# test = pd.concat(low_growth_angle_ee)['Average Angle (deg)']
# kde_low_angle_top = sns.distplot(ee_low_plat_angle_manual_absolute_df['Average Angle (deg)'], kde=True, hist_kws=dict(color='g', edgecolor='g', linewidth=2), kde_kws=dict(color='g'))#, rug=True, rug_kws=dict(color='k', alpha=0.5))
# angle_eedb = sns.distplot(eedb_contact_diam_non_dim_df_list_long_dropna['Average Angle (deg)'], kde=True, hist_kws=dict(color='k', edgecolor='k', linewidth=2), kde_kws=dict(color='k'))
# angle_eedb_no_high_adh = sns.distplot(eedb_contact_diam_non_dim_df_list_long_dropna_no_high_adh['Average Angle (deg)'], kde=True, hist_kws=dict(color='r', edgecolor='r', linewidth=2), kde_kws=dict(color='r'))
# sns.distplot(test, kde=True, hist_kws=dict(color='b', edgecolor='b', linewidth=2), kde_kws=dict(color='b'))



# I noticed that the end values are excluded from signal.find_peaks, therefore 
# I will need to pad the array with angles in order to get the true peak value:
# Also, I checked the other measures, and luckily they are distributed in a 
# much more gaussian fashion, so their peaks are in the middle of the dataset.
# i = 1
# kde_radius_init_2_circ_angle = sns.distplot(two_circ_interp_angle(ee_first_valid_area), kde=False, norm_hist=False)
# #kde_radius_init_2_circ_angle_x = kde_radius_init_2_circ_angle.lines[0].get_xdata()
# #kde_radius_init_2_circ_angle_x = np.pad(kde_radius_init_2_circ_angle_x, pad_width = i, mode='constant', constant_values=0)
# #kde_radius_init_2_circ_angle_y = kde_radius_init_2_circ_angle.lines[0].get_ydata()
# #kde_radius_init_2_circ_angle_y = np.pad(kde_radius_init_2_circ_angle_y, pad_width = i, mode='constant', constant_values=0)
# #kde_radius_init_2_circ_angle_maxid = signal.find_peaks(kde_radius_init_2_circ_angle_y, distance=200)[0]
# #kde_radius_init_2_circ_angle_peaks_y = kde_radius_init_2_circ_angle_y[kde_radius_init_2_circ_angle_maxid]
# #kde_radius_init_2_circ_angle_peaks = kde_radius_init_2_circ_angle_x[kde_radius_init_2_circ_angle_maxid]
# plt.clf()


# fig_kde_low_angle, kde_low_angle = plt.subplots(figsize=(width_1col,height_1col))
# kde_low_angle = sns.distplot(ee_low_plat_contnd_df['Average Angle (deg)'], kde=True, hist_kws=dict(edgecolor="k", linewidth=2), rug=True, bins=np.linspace(0,180,37))
# angle = sns.distplot(eedb_fix_list_long_dropna['Average Angle (deg)'], kde=True, hist_kws=dict(edgecolor="k", linewidth=2), rug=True, bins=np.linspace(0,180,37))#, ax=kde_low)#, kde_kws=dict(bw=5))
# kde_radius_init_2_circ_angle = kde_low_angle.twinx()
# sns.distplot(two_circ_interp_angle(ee_first_valid_area), hist=True, kde=False, ax = kde_radius_init_2_circ_angle, color='g', bins=np.linspace(0,180,37))
# kde_low_angle.plot(kde_low_angle_peaks, kde_low_angle_peaks_y, 'bo', ms=4)
# kde_low_angle.plot(eedb_fix_angle_peaks, eedb_fix_angle_peaks_y, 'ro', ms=4)
# #kde_radius_init_2_circ_angle.plot(kde_radius_init_2_circ_angle_peaks, kde_radius_init_2_circ_angle_peaks_y, 'go', ms=4)
# #kde_low.axvline(eedb_fix_cont_peak, c='r')
# kde_low_angle.set_xlim(0)
# kde_low_angle.set_xlabel('Contact Angle (deg)')
# #kde_radius_init_2_circ.yaxis.label.set_color('green')
# kde_radius_init_2_circ_angle.tick_params(axis='y', colors='green')
# kde_radius_init_2_circ_angle.set_ylim(0)
# fig_kde_low_angle.draw
# fig_kde_low_angle.savefig(Path.joinpath(output_dir, r"ee_angle_raw_data_less_than_growth_start.tif"), dpi=dpi_graph)

# print('kde plot peaks:')
# print('ecto-ecto (blue): ', kde_low_angle_peaks)
# print('ecto-ecto_fix (orange): ', eedb_fix_angle_peaks)
# #print('2_circles (green): ', kde_radius_init_2_circ_angle_peaks)







# Plot the cleaned up low plateaus, high plateaus, and logistic growth constants
# for the non-dimensionalized contacts when allowing 5% growth on either end:



# plt.clf()
# kde_low_nd = sns.distplot(ee_low_plat_contnd_df['Contact Surface Length (N.D.)'], kde=True, hist_kws=dict(edgecolor="k", linewidth=2), rug=True)
# kde_low_nd_x = kde_low_nd.lines[0].get_xdata()
# kde_low_nd_y = kde_low_nd.lines[0].get_ydata()
# kde_low_nd_maxid = signal.find_peaks(kde_low_nd_y)[0]
# kde_low_cont_nd_peaks_y = kde_low_nd_y[kde_low_nd_maxid]
# kde_low_cont_nd_peaks = kde_low_nd_x[kde_low_nd_maxid]
# plt.clf()

# cont_nd = sns.distplot(eedb_fix_contact_diam_non_dim_df_list_long_dropna['Contact Surface Length (N.D.)'], kde=True, hist_kws=dict(edgecolor="k", linewidth=2), rug=True)#, ax=kde_low)#, kde_kws=dict(bw=5))
# cont_nd_x = cont_nd.lines[0].get_xdata()
# cont_nd_y = cont_nd.lines[0].get_ydata()
# cont_nd_maxid = signal.find_peaks(cont_nd_y)[0]
# eedb_fix_cont_nd_peaks_y = cont_nd_y[cont_nd_maxid]
# eedb_fix_cont_nd_peaks = cont_nd_x[cont_nd_maxid]
# plt.clf()

# kde_radius_init_2_circ_nd = sns.distplot(two_circ_interp_contnd(ee_first_valid_area), kde=False, norm_hist=False)
# #kde_radius_init_2_circ_nd_x = kde_radius_init_2_circ_nd.lines[0].get_xdata()
# #kde_radius_init_2_circ_nd_y = kde_radius_init_2_circ_nd.lines[0].get_ydata()
# #kde_radius_init_2_circ_nd_maxid = signal.find_peaks(kde_radius_init_2_circ_nd_y)[0]
# #kde_radius_init_2_circ_nd_peaks_y = kde_radius_init_2_circ_nd_y[kde_radius_init_2_circ_nd_maxid]
# #kde_radius_init_2_circ_nd_peaks = kde_radius_init_2_circ_nd_x[kde_radius_init_2_circ_nd_maxid]
# plt.clf()


# fig_kde_low_nd, kde_low_nd = plt.subplots(figsize=(width_1col,height_1col))
# kde_low_nd = sns.distplot(ee_low_plat_contnd_df['Contact Surface Length (N.D.)'], kde=True, hist_kws=dict(color='k', edgecolor="k", linewidth=2), kde_kws=dict(color='k'))#, rug=True, rug_kws=dict(color='k', alpha=0.5))
# cont = sns.distplot(eedb_fix_contact_diam_non_dim_df_list_long_dropna['Contact Surface Length (N.D.)'], kde=True, hist_kws=dict(edgecolor="k", linewidth=2))#, rug=True, rug_kws=dict(alpha=0.5))#, ax=kde_low)#, kde_kws=dict(bw=5))
# kde_radius_init_2_circ_nd = kde_low_nd.twinx()
# sns.distplot(two_circ_interp_contnd(ee_first_valid_area), hist=False, ax = kde_radius_init_2_circ_nd, color='r')
# kde_low_nd.plot(kde_low_cont_nd_peaks, kde_low_cont_nd_peaks_y, 'ko', ms=4)
# kde_low_nd.plot(eedb_fix_cont_nd_peaks, eedb_fix_cont_nd_peaks_y, c='orange', marker='o', ms=4)
# kde_low_nd.set_xlim(0)
# kde_low_nd.set_xlabel('')
# #kde_radius_init_2_circ.yaxis.label.set_color('green')
# kde_radius_init_2_circ_nd.tick_params(axis='y', colors='red')
# kde_radius_init_2_circ_nd.set_ylim(0)
# fig_kde_low_nd.draw
# fig_kde_low_nd.savefig(Path.joinpath(output_dir, r"ee_contnd_raw_data_less_than_growth_start.tif"), dpi=dpi_graph)
# # print('ecto-ecto (black): ', kde_low_cont_nd_peaks)
# # print('ecto-ecto_fix (orange): ', eedb_fix_cont_nd_peaks)
# #print('2_circles (red): ', kde_radius_init_2_circ_nd_peaks)
# plt.clf()



# # Plot the cleaned up low plateaus for the non-dimensionalized contacts:
# plt.clf()
# kde_low_nd = sns.distplot(ee_low_plat_contnd_strict_df['Contact Surface Length (N.D.)'], kde=True, hist_kws=dict(edgecolor="k", linewidth=2), rug=True)
# kde_low_nd_x = kde_low_nd.lines[0].get_xdata()
# kde_low_nd_y = kde_low_nd.lines[0].get_ydata()
# kde_low_nd_maxid = signal.find_peaks(kde_low_nd_y)[0]
# kde_low_cont_nd_peaks_y = kde_low_nd_y[kde_low_nd_maxid]
# kde_low_cont_nd_peaks = kde_low_nd_x[kde_low_nd_maxid]
# plt.clf()

# cont_nd = sns.distplot(eedb_fix_contact_diam_non_dim_df_list_long_dropna['Contact Surface Length (N.D.)'], kde=True, hist_kws=dict(edgecolor="k", linewidth=2), rug=True)#, ax=kde_low)#, kde_kws=dict(bw=5))
# cont_nd_x = cont_nd.lines[0].get_xdata()
# cont_nd_y = cont_nd.lines[0].get_ydata()
# cont_nd_maxid = signal.find_peaks(cont_nd_y)[0]
# eedb_fix_cont_nd_peaks_y = cont_nd_y[cont_nd_maxid]
# eedb_fix_cont_nd_peaks = cont_nd_x[cont_nd_maxid]
# plt.clf()

# kde_radius_init_2_circ_nd = sns.distplot(two_circ_interp_contnd(ee_first_valid_area), kde=False, norm_hist=False)
# #kde_radius_init_2_circ_nd_x = kde_radius_init_2_circ_nd.lines[0].get_xdata()
# #kde_radius_init_2_circ_nd_y = kde_radius_init_2_circ_nd.lines[0].get_ydata()
# #kde_radius_init_2_circ_nd_maxid = signal.find_peaks(kde_radius_init_2_circ_nd_y)[0]
# #kde_radius_init_2_circ_nd_peaks_y = kde_radius_init_2_circ_nd_y[kde_radius_init_2_circ_nd_maxid]
# #kde_radius_init_2_circ_nd_peaks = kde_radius_init_2_circ_nd_x[kde_radius_init_2_circ_nd_maxid]
# plt.clf()

# fig_kde_low_nd, kde_low_nd = plt.subplots(figsize=(width_1col,height_1col))
# kde_low_nd = sns.distplot(ee_low_plat_contnd_strict_df['Contact Surface Length (N.D.)'], kde=True, hist_kws=dict(color='k', edgecolor="k", linewidth=2), kde_kws=dict(color='k'))#, rug=True, rug_kws=dict(color='k', alpha=0.5))
# cont = sns.distplot(eedb_fix_contact_diam_non_dim_df_list_long_dropna['Contact Surface Length (N.D.)'], kde=True, hist_kws=dict(edgecolor="k", linewidth=2))#, rug=True, rug_kws=dict(alpha=0.5))#, ax=kde_low)#, kde_kws=dict(bw=5))
# kde_radius_init_2_circ_nd = kde_low_nd.twinx()
# sns.distplot(two_circ_interp_contnd(ee_first_valid_area), hist=False, ax = kde_radius_init_2_circ_nd, color='r')
# kde_low_nd.plot(kde_low_cont_nd_peaks, kde_low_cont_nd_peaks_y, 'ko', ms=4)
# kde_low_nd.plot(eedb_fix_cont_nd_peaks, eedb_fix_cont_nd_peaks_y, c='orange', marker='o', ms=4)
# #kde_radius_init_2_circ_nd.plot(kde_radius_init_2_circ_nd_peaks, kde_radius_init_2_circ_nd_peaks_y, 'ro', ms=4)
# #kde_low_nd.axvline(ee_low_plat_contnd_verified_df['Contact Surface Length (N.D.)'].mean(), c='k')
# #kde_low_nd.axvline(eedb_fix_contact_diam_non_dim_df_list_long_dropna['Contact Surface Length (N.D.)'].mean(), c='orange')
# #kde_low_nd.axvline(two_circ_interp_contnd(ee_first_valid_area).mean(), c='r')
# #kde_low.axvline(eedb_fix_cont_peak, c='r')
# kde_low_nd.set_xlim(0)
# kde_low_nd.set_xlabel('')
# #kde_radius_init_2_circ.yaxis.label.set_color('green')
# kde_radius_init_2_circ_nd.tick_params(axis='y', colors='red')
# kde_radius_init_2_circ_nd.set_ylim(0)
# fig_kde_low_nd.draw
# fig_kde_low_nd.savefig(Path.joinpath(output_dir, r"ee_contnd_raw_data_less_than_growth_start_strict.tif"), dpi=dpi_graph)
# # print('ecto-ecto (black): ', kde_low_cont_nd_peaks)
# # print('ecto-ecto_fix (orange): ', eedb_fix_cont_nd_peaks)
# #print('2_circles (red): ', kde_radius_init_2_circ_nd_peaks)
# plt.clf()





# Plot the cleaned up low plateaus for the non-dimensionalized contacts when 
# allowing 5% growth on either end and using only verified low plateaus:
# plt.clf()
# kde_low_nd = sns.distplot(ee_low_plat_contnd_verified_df['Contact Surface Length (N.D.)'], kde=True, hist_kws=dict(edgecolor="k", linewidth=2), rug=True)
# kde_low_nd_x = kde_low_nd.lines[0].get_xdata()
# kde_low_nd_y = kde_low_nd.lines[0].get_ydata()
# kde_low_nd_maxid = signal.find_peaks(kde_low_nd_y)[0]
# kde_low_cont_nd_peaks_y = kde_low_nd_y[kde_low_nd_maxid]
# kde_low_cont_nd_peaks = kde_low_nd_x[kde_low_nd_maxid]
# plt.clf()

# cont_nd = sns.distplot(eedb_fix_contact_diam_non_dim_df_list_long_dropna['Contact Surface Length (N.D.)'], kde=True, hist_kws=dict(edgecolor="k", linewidth=2), rug=True)#, ax=kde_low)#, kde_kws=dict(bw=5))
# cont_nd_x = cont_nd.lines[0].get_xdata()
# cont_nd_y = cont_nd.lines[0].get_ydata()
# cont_nd_maxid = signal.find_peaks(cont_nd_y)[0]
# eedb_fix_cont_nd_peaks_y = cont_nd_y[cont_nd_maxid]
# eedb_fix_cont_nd_peaks = cont_nd_x[cont_nd_maxid]
# plt.clf()

# kde_radius_init_2_circ_nd = sns.distplot(two_circ_interp_contnd(ee_first_valid_area_low_only), kde=False, norm_hist=False)
# #kde_radius_init_2_circ_nd_x = kde_radius_init_2_circ_nd.lines[0].get_xdata()
# #kde_radius_init_2_circ_nd_y = kde_radius_init_2_circ_nd.lines[0].get_ydata()
# #kde_radius_init_2_circ_nd_maxid = signal.find_peaks(kde_radius_init_2_circ_nd_y)[0]
# #kde_radius_init_2_circ_nd_peaks_y = kde_radius_init_2_circ_nd_y[kde_radius_init_2_circ_nd_maxid]
# #kde_radius_init_2_circ_nd_peaks = kde_radius_init_2_circ_nd_x[kde_radius_init_2_circ_nd_maxid]
# plt.clf()



# fig_kde_low_nd, kde_low_nd = plt.subplots(figsize=(width_1col,height_1col))
# kde_low_nd = sns.distplot(ee_low_plat_contnd_verified_df['Contact Surface Length (N.D.)'], kde=True, hist_kws=dict(color='k', edgecolor="k", linewidth=2), kde_kws=dict(color='k'))#, rug=True, rug_kws=dict(color='k', alpha=0.5))
# cont = sns.distplot(eedb_fix_contact_diam_non_dim_df_list_long_dropna['Contact Surface Length (N.D.)'], kde=True, hist_kws=dict(edgecolor="k", linewidth=2))#, rug=True, rug_kws=dict(alpha=0.5))#, ax=kde_low)#, kde_kws=dict(bw=5))
# kde_radius_init_2_circ_nd = kde_low_nd.twinx()
# sns.distplot(two_circ_interp_contnd(ee_first_valid_area_low_only), hist=False, ax = kde_radius_init_2_circ_nd, color='r')
# kde_low_nd.plot(kde_low_cont_nd_peaks, kde_low_cont_nd_peaks_y, 'ko', ms=4)
# kde_low_nd.plot(eedb_fix_cont_nd_peaks, eedb_fix_cont_nd_peaks_y, c='orange', marker='o', ms=4)
# kde_low_nd.set_xlim(0)
# #kde_low_nd.set_xlabel('Contact Surface Length (N.D.)')
# kde_low_nd.set_xlabel('')
# #kde_radius_init_2_circ.yaxis.label.set_color('green')
# kde_radius_init_2_circ_nd.tick_params(axis='y', colors='red')
# kde_radius_init_2_circ_nd.set_ylim(0)
# fig_kde_low_nd.draw
# fig_kde_low_nd.savefig(Path.joinpath(output_dir, r"ee_contnd_raw_data_less_than_growth_start_low_plat_only.tif"), dpi=dpi_graph)
# # print('ecto-ecto (black): ', kde_low_cont_nd_peaks)
# # print('ecto-ecto_fix (orange): ', eedb_fix_cont_nd_peaks)
# #print('2_circles (red): ', kde_radius_init_2_circ_nd_peaks)
# plt.clf()



# # Plot the cleaned up low plateaus for the non-dimensionalized contacts when 
# # allowing 2.5% growth on either end and using only verified low plateaus:
# plt.clf()
# kde_low_nd = sns.distplot(ee_low_plat_contnd_verified_strict_df['Contact Surface Length (N.D.)'], kde=True, hist_kws=dict(edgecolor="k", linewidth=2), rug=True)
# kde_low_nd_x = kde_low_nd.lines[0].get_xdata()
# kde_low_nd_y = kde_low_nd.lines[0].get_ydata()
# kde_low_nd_maxid = signal.find_peaks(kde_low_nd_y)[0]
# kde_low_cont_nd_peaks_y = kde_low_nd_y[kde_low_nd_maxid]
# kde_low_cont_nd_peaks = kde_low_nd_x[kde_low_nd_maxid]
# plt.clf()

# cont_nd = sns.distplot(eedb_fix_contact_diam_non_dim_df_list_long_dropna['Contact Surface Length (N.D.)'], kde=True, hist_kws=dict(edgecolor="k", linewidth=2), rug=True)#, ax=kde_low)#, kde_kws=dict(bw=5))
# cont_nd_x = cont_nd.lines[0].get_xdata()
# cont_nd_y = cont_nd.lines[0].get_ydata()
# cont_nd_maxid = signal.find_peaks(cont_nd_y)[0]
# eedb_fix_cont_nd_peaks_y = cont_nd_y[cont_nd_maxid]
# eedb_fix_cont_nd_peaks = cont_nd_x[cont_nd_maxid]
# plt.clf()

# kde_radius_init_2_circ_nd = sns.distplot(two_circ_interp_contnd(ee_first_valid_area_low_only), kde=False, norm_hist=False)
# #kde_radius_init_2_circ_nd_x = kde_radius_init_2_circ_nd.lines[0].get_xdata()
# #kde_radius_init_2_circ_nd_y = kde_radius_init_2_circ_nd.lines[0].get_ydata()
# #kde_radius_init_2_circ_nd_maxid = signal.find_peaks(kde_radius_init_2_circ_nd_y)[0]
# #kde_radius_init_2_circ_nd_peaks_y = kde_radius_init_2_circ_nd_y[kde_radius_init_2_circ_nd_maxid]
# #kde_radius_init_2_circ_nd_peaks = kde_radius_init_2_circ_nd_x[kde_radius_init_2_circ_nd_maxid]
# plt.clf()



# fig_kde_low_nd, kde_low_nd = plt.subplots(figsize=(width_1col,height_1col))
# kde_low_nd = sns.distplot(ee_low_plat_contnd_verified_strict_df['Contact Surface Length (N.D.)'], kde=True, hist_kws=dict(color='k', edgecolor="k", linewidth=2), kde_kws=dict(color='k'))#, rug=True, rug_kws=dict(color='k', alpha=0.5))
# cont = sns.distplot(eedb_fix_contact_diam_non_dim_df_list_long_dropna['Contact Surface Length (N.D.)'], kde=True, hist_kws=dict(edgecolor="k", linewidth=2))#, rug=True, rug_kws=dict(alpha=0.5))#, ax=kde_low)#, kde_kws=dict(bw=5))
# kde_radius_init_2_circ_nd = kde_low_nd.twinx()
# sns.distplot(two_circ_interp_contnd(ee_first_valid_area_low_only), hist=False, ax = kde_radius_init_2_circ_nd, color='r')
# kde_low_nd.plot(kde_low_cont_nd_peaks, kde_low_cont_nd_peaks_y, 'ko', ms=4)
# kde_low_nd.plot(eedb_fix_cont_nd_peaks, eedb_fix_cont_nd_peaks_y, c='orange', marker='o', ms=4)
# kde_low_nd.set_xlim(0)
# #kde_low_nd.set_xlabel('Contact Surface Length (N.D.)')
# kde_low_nd.set_xlabel('')
# #kde_radius_init_2_circ.yaxis.label.set_color('green')
# kde_radius_init_2_circ_nd.tick_params(axis='y', colors='red')
# kde_radius_init_2_circ_nd.set_ylim(0)
# fig_kde_low_nd.draw
# fig_kde_low_nd.savefig(Path.joinpath(output_dir, r"ee_contnd_raw_data_less_than_growth_start_strict_low_plat_only.tif"), dpi=dpi_graph)
# # print('ecto-ecto (black): ', kde_low_cont_nd_peaks)
# # print('ecto-ecto_fix (orange): ', eedb_fix_cont_nd_peaks)
# #print('2_circles (red): ', kde_radius_init_2_circ_nd_peaks)
# plt.clf()




# # Plot the cleaned up low plateaus for the non-dimensionalized contacts when 
# # allowing 1% growth on either end and using only verified low plateaus:
# plt.clf()
# kde_low_nd = sns.distplot(ee_low_plat_contnd_verified_v_strict_df['Contact Surface Length (N.D.)'], kde=True, hist_kws=dict(edgecolor="k", linewidth=2), rug=True)
# kde_low_nd_x = kde_low_nd.lines[0].get_xdata()
# kde_low_nd_y = kde_low_nd.lines[0].get_ydata()
# kde_low_nd_maxid = signal.find_peaks(kde_low_nd_y)[0]
# kde_low_cont_nd_peaks_y = kde_low_nd_y[kde_low_nd_maxid]
# kde_low_cont_nd_peaks = kde_low_nd_x[kde_low_nd_maxid]
# plt.clf()

# cont_nd = sns.distplot(eedb_fix_contact_diam_non_dim_df_list_long_dropna['Contact Surface Length (N.D.)'], kde=True, hist_kws=dict(edgecolor="k", linewidth=2), rug=True)#, ax=kde_low)#, kde_kws=dict(bw=5))
# cont_nd_x = cont_nd.lines[0].get_xdata()
# cont_nd_y = cont_nd.lines[0].get_ydata()
# cont_nd_maxid = signal.find_peaks(cont_nd_y)[0]
# eedb_fix_cont_nd_peaks_y = cont_nd_y[cont_nd_maxid]
# eedb_fix_cont_nd_peaks = cont_nd_x[cont_nd_maxid]
# plt.clf()

# kde_radius_init_2_circ_nd = sns.distplot(two_circ_interp_contnd(ee_first_valid_area_low_only), kde=False, norm_hist=False)
# #kde_radius_init_2_circ_nd_x = kde_radius_init_2_circ_nd.lines[0].get_xdata()
# #kde_radius_init_2_circ_nd_y = kde_radius_init_2_circ_nd.lines[0].get_ydata()
# #kde_radius_init_2_circ_nd_maxid = signal.find_peaks(kde_radius_init_2_circ_nd_y)[0]
# #kde_radius_init_2_circ_nd_peaks_y = kde_radius_init_2_circ_nd_y[kde_radius_init_2_circ_nd_maxid]
# #kde_radius_init_2_circ_nd_peaks = kde_radius_init_2_circ_nd_x[kde_radius_init_2_circ_nd_maxid]
# plt.clf()



# fig_kde_low_nd, kde_low_nd = plt.subplots(figsize=(width_1col,height_1col))
# kde_low_nd = sns.distplot(ee_low_plat_contnd_verified_v_strict_df['Contact Surface Length (N.D.)'], kde=True, hist_kws=dict(color='k', edgecolor="k", linewidth=2), kde_kws=dict(color='k'))#, rug=True, rug_kws=dict(color='k', alpha=0.5))
# cont = sns.distplot(eedb_fix_contact_diam_non_dim_df_list_long_dropna['Contact Surface Length (N.D.)'], kde=True, hist_kws=dict(edgecolor="k", linewidth=2))#, rug=True, rug_kws=dict(alpha=0.5))#, ax=kde_low)#, kde_kws=dict(bw=5))
# kde_radius_init_2_circ_nd = kde_low_nd.twinx()
# sns.distplot(two_circ_interp_contnd(ee_first_valid_area_low_only), hist=False, ax = kde_radius_init_2_circ_nd, color='r')
# kde_low_nd.plot(kde_low_cont_nd_peaks, kde_low_cont_nd_peaks_y, 'ko', ms=4)
# kde_low_nd.plot(eedb_fix_cont_nd_peaks, eedb_fix_cont_nd_peaks_y, c='orange', marker='o', ms=4)
# kde_low_nd.set_xlim(0)
# #kde_low_nd.set_xlabel('Contact Surface Length (N.D.)')
# kde_low_nd.set_xlabel('')
# #kde_radius_init_2_circ.yaxis.label.set_color('green')
# kde_radius_init_2_circ_nd.tick_params(axis='y', colors='red')
# kde_radius_init_2_circ_nd.set_ylim(0)
# fig_kde_low_nd.draw
# fig_kde_low_nd.savefig(Path.joinpath(output_dir, r"ee_contnd_raw_data_less_than_growth_start_v_strict_low_plat_only.tif"), dpi=dpi_graph)
# # print('ecto-ecto (black): ', kde_low_cont_nd_peaks)
# # print('ecto-ecto_fix (orange): ', eedb_fix_cont_nd_peaks)
# #print('2_circles (red): ', kde_radius_init_2_circ_nd_peaks)
# plt.clf()



# # Plot the cleaned up low plateaus for the non-dimensionalized contacts when 
# # allowing 0.1% growth on either end and using only verified low plateaus:
# plt.clf()
# kde_low_nd = sns.distplot(ee_low_plat_contnd_verified_absolute_df['Contact Surface Length (N.D.)'], kde=True, hist_kws=dict(edgecolor="k", linewidth=2), rug=True)
# kde_low_nd_x = kde_low_nd.lines[0].get_xdata()
# kde_low_nd_y = kde_low_nd.lines[0].get_ydata()
# kde_low_nd_maxid = signal.find_peaks(kde_low_nd_y)[0]
# kde_low_cont_nd_peaks_y = kde_low_nd_y[kde_low_nd_maxid]
# kde_low_cont_nd_peaks = kde_low_nd_x[kde_low_nd_maxid]
# plt.clf()

# cont_nd = sns.distplot(eedb_fix_contact_diam_non_dim_df_list_long_dropna['Contact Surface Length (N.D.)'], kde=True, hist_kws=dict(edgecolor="k", linewidth=2), rug=True)#, ax=kde_low)#, kde_kws=dict(bw=5))
# cont_nd_x = cont_nd.lines[0].get_xdata()
# cont_nd_y = cont_nd.lines[0].get_ydata()
# cont_nd_maxid = signal.find_peaks(cont_nd_y)[0]
# eedb_fix_cont_nd_peaks_y = cont_nd_y[cont_nd_maxid]
# eedb_fix_cont_nd_peaks = cont_nd_x[cont_nd_maxid]
# plt.clf()

# kde_radius_init_2_circ_nd = sns.distplot(two_circ_interp_contnd(ee_first_valid_area_low_only), kde=False, norm_hist=False)
# #kde_radius_init_2_circ_nd_x = kde_radius_init_2_circ_nd.lines[0].get_xdata()
# #kde_radius_init_2_circ_nd_y = kde_radius_init_2_circ_nd.lines[0].get_ydata()
# #kde_radius_init_2_circ_nd_maxid = signal.find_peaks(kde_radius_init_2_circ_nd_y)[0]
# #kde_radius_init_2_circ_nd_peaks_y = kde_radius_init_2_circ_nd_y[kde_radius_init_2_circ_nd_maxid]
# #kde_radius_init_2_circ_nd_peaks = kde_radius_init_2_circ_nd_x[kde_radius_init_2_circ_nd_maxid]
# plt.clf()



# fig_kde_low_nd, kde_low_nd = plt.subplots(figsize=(width_1col,height_1col))
# kde_low_nd = sns.distplot(ee_low_plat_contnd_verified_absolute_df['Contact Surface Length (N.D.)'], kde=True, hist_kws=dict(color='k', edgecolor="k", linewidth=2), kde_kws=dict(color='k'))#, rug=True, rug_kws=dict(color='k', alpha=0.5))
# cont = sns.distplot(eedb_fix_contact_diam_non_dim_df_list_long_dropna['Contact Surface Length (N.D.)'], kde=True, hist_kws=dict(edgecolor="k", linewidth=2))#, rug=True, rug_kws=dict(alpha=0.5))#, ax=kde_low)#, kde_kws=dict(bw=5))
# kde_radius_init_2_circ_nd = kde_low_nd.twinx()
# sns.distplot(two_circ_interp_contnd(ee_first_valid_area_low_only), hist=False, ax = kde_radius_init_2_circ_nd, color='r')
# kde_low_nd.plot(kde_low_cont_nd_peaks, kde_low_cont_nd_peaks_y, 'ko', ms=4)
# kde_low_nd.plot(eedb_fix_cont_nd_peaks, eedb_fix_cont_nd_peaks_y, c='orange', marker='o', ms=4)
# kde_low_nd.set_xlabel('')
# #kde_radius_init_2_circ.yaxis.label.set_color('green')
# kde_radius_init_2_circ_nd.tick_params(axis='y', colors='red')
# kde_radius_init_2_circ_nd.set_ylim(0)
# fig_kde_low_nd.draw
# fig_kde_low_nd.savefig(Path.joinpath(output_dir, r"ee_contnd_raw_data_less_than_growth_start_pt1percent_low_plat_only.tif"), dpi=dpi_graph)
# # print('ecto-ecto (black): ', kde_low_cont_nd_peaks)
# # print('ecto-ecto_fix (orange): ', eedb_fix_cont_nd_peaks)
# #print('2_circles (red): ', kde_radius_init_2_circ_nd_peaks)
# plt.clf()



# Plot the cleaned up low plateaus for the non-dimensionalized contacts when 
# allowing 0% growth (using human judgement) in low adhesion data:
plt.clf()
kde_low_nd = sns.distplot(ee_low_plat_contnd_manual_absolute_df['Contact Surface Length (N.D.)'], kde=True, hist_kws=dict(edgecolor="k", linewidth=2), rug=True)
kde_low_nd_x = kde_low_nd.lines[0].get_xdata()
kde_low_nd_y = kde_low_nd.lines[0].get_ydata()
kde_low_nd_maxid = signal.find_peaks(kde_low_nd_y)[0]
kde_low_cont_nd_peaks_y = kde_low_nd_y[kde_low_nd_maxid]
kde_low_cont_nd_peaks = kde_low_nd_x[kde_low_nd_maxid]
plt.clf()

cont_nd = sns.distplot(eedb_fix_contact_diam_non_dim_df_list_long_dropna['Contact Surface Length (N.D.)'], kde=True, hist_kws=dict(edgecolor="k", linewidth=2), rug=True)#, ax=kde_low)#, kde_kws=dict(bw=5))
cont_nd_x = cont_nd.lines[0].get_xdata()
cont_nd_y = cont_nd.lines[0].get_ydata()
cont_nd_maxid = signal.find_peaks(cont_nd_y)[0]
eedb_fix_cont_nd_peaks_y = cont_nd_y[cont_nd_maxid]
eedb_fix_cont_nd_peaks = cont_nd_x[cont_nd_maxid]
plt.clf()

kde_radius_init_2_circ_nd = sns.distplot(two_circ_interp_contnd(ee_first_valid_area_low_only), kde=True)#, hist=False)
kde_radius_init_2_circ_nd_x = kde_radius_init_2_circ_nd.lines[0].get_xdata()
kde_radius_init_2_circ_nd_y = kde_radius_init_2_circ_nd.lines[0].get_ydata()
kde_radius_init_2_circ_nd_maxid = signal.find_peaks(kde_radius_init_2_circ_nd_y)[0]
kde_radius_init_2_circ_nd_peaks_y = kde_radius_init_2_circ_nd_y[kde_radius_init_2_circ_nd_maxid]
kde_radius_init_2_circ_nd_peaks = kde_radius_init_2_circ_nd_x[kde_radius_init_2_circ_nd_maxid]
plt.clf()

eedb_cont_nd = sns.distplot(eedb_contact_diam_non_dim_df_list_long_dropna['Contact Surface Length (N.D.)'], kde=True, hist_kws=dict(edgecolor="k", linewidth=2), rug=True)#, ax=kde_low)#, kde_kws=dict(bw=5))
eedb_cont_nd_x = eedb_cont_nd.lines[0].get_xdata()
eedb_cont_nd_y = eedb_cont_nd.lines[0].get_ydata()
eedb_cont_nd_maxid = signal.find_peaks(eedb_cont_nd_y)[0]
eedb_cont_nd_peaks_y = eedb_cont_nd_y[eedb_cont_nd_maxid]
eedb_cont_nd_peaks = eedb_cont_nd_x[eedb_cont_nd_maxid]
plt.clf()

eedb_no_high_adh_cont_nd = sns.distplot(eedb_contact_diam_non_dim_df_list_long_dropna_no_high_adh['Contact Surface Length (N.D.)'], kde=True, hist_kws=dict(edgecolor="k", linewidth=2), rug=True)#, ax=kde_low)#, kde_kws=dict(bw=5))
eedb_no_high_adh_cont_nd_x = eedb_no_high_adh_cont_nd.lines[0].get_xdata()
eedb_no_high_adh_cont_nd_y = eedb_no_high_adh_cont_nd.lines[0].get_ydata()
eedb_no_high_adh_cont_nd_maxid = signal.find_peaks(eedb_no_high_adh_cont_nd_y)[0]
eedb_no_high_adh_cont_nd_peaks_y = eedb_no_high_adh_cont_nd_y[eedb_no_high_adh_cont_nd_maxid]
eedb_no_high_adh_cont_nd_peaks = eedb_no_high_adh_cont_nd_x[eedb_no_high_adh_cont_nd_maxid]
plt.clf()





distrib_ee_low_plat_contnd_mean = np.mean(ee_low_plat_contnd_manual_absolute_df['Contact Surface Length (N.D.)'])
distrib_ee_low_plat_contnd_median = np.median(ee_low_plat_contnd_manual_absolute_df['Contact Surface Length (N.D.)'])
distrib_ee_low_plat_contnd_std = np.std(ee_low_plat_contnd_manual_absolute_df['Contact Surface Length (N.D.)'])

distrib_eedbfix_contnd_mean = np.mean(eedb_fix_contact_diam_non_dim_df_list_long_dropna['Contact Surface Length (N.D.)'])
distrib_eedbfix_contnd_median = np.median(eedb_fix_contact_diam_non_dim_df_list_long_dropna['Contact Surface Length (N.D.)'])
distrib_eedbfix_contnd_std = np.std(eedb_fix_contact_diam_non_dim_df_list_long_dropna['Contact Surface Length (N.D.)'])

distrib_eedb_contnd_mean = np.mean(eedb_contact_diam_non_dim_df_list_long_dropna_no_high_adh['Contact Surface Length (N.D.)'])
distrib_eedb_contnd_median = np.median(eedb_contact_diam_non_dim_df_list_long_dropna_no_high_adh['Contact Surface Length (N.D.)'])
distrib_eedb_contnd_std = np.std(eedb_contact_diam_non_dim_df_list_long_dropna_no_high_adh['Contact Surface Length (N.D.)'])

distrib_two_circ_contnd_mean = np.mean(two_circ_interp_contnd(ee_first_valid_area_low_only))
distrib_two_circ_contnd_median = np.median(two_circ_interp_contnd(ee_first_valid_area_low_only))
distrib_two_circ_contnd_std = np.std(two_circ_interp_contnd(ee_first_valid_area_low_only))


xlim = max(ee_low_plat_contnd_manual_absolute_df['Contact Surface Length (N.D.)'].max(), 
           eedb_fix_contact_diam_non_dim_df_list_long_dropna['Contact Surface Length (N.D.)'].max(), 
           eedb_contact_diam_non_dim_df_list_long_dropna_no_high_adh['Contact Surface Length (N.D.)'].max(), 
           two_circ_interp_contnd(ee_first_valid_area_low_only).max())
xlim = round_up_to_multiple(xlim, 0.5)


fig_kde_low_nd, (kde_low_nd_top, kde_low_nd_bot) = plt.subplots(figsize=(width_1col*2/3,height_1col*2/3), nrows=2, ncols=1, sharex=True)
kde_low_nd_top = sns.distplot(ee_low_plat_contnd_manual_absolute_df['Contact Surface Length (N.D.)'], kde=True, hist_kws=dict(color='g', edgecolor='g', linewidth=0.5), kde_kws=dict(color='g', linewidth=0.5), ax=kde_low_nd_top)#, rug=True, rug_kws=dict(color='k', alpha=0.5))
contnd_eedb_fix = sns.distplot(eedb_fix_contact_diam_non_dim_df_list_long_dropna['Contact Surface Length (N.D.)'], kde=True, hist_kws=dict(color='orange', edgecolor='orange', linewidth=0.5), kde_kws=dict(color='orange', linewidth=0.5), ax=kde_low_nd_top)#, rug=True, rug_kws=dict(alpha=0.5))#, ax=kde_low)#, kde_kws=dict(bw=5))
# contnd_eedb = sns.distplot(eedb_contact_diam_non_dim_df_list_long_dropna['Contact Surface Length (N.D.)'], kde=True, hist_kws=dict(color='k', edgecolor='k', linewidth=2), kde_kws=dict(color='k', linewidth=0.5))
contnd_eedb_no_high_adh = sns.distplot(eedb_contact_diam_non_dim_df_list_long_dropna_no_high_adh['Contact Surface Length (N.D.)'], kde=True, hist_kws=dict(color='k', edgecolor='k', linewidth=0.5), kde_kws=dict(color='k', linewidth=0.5), ax=kde_low_nd_bot)
kde_radius_init_2_circ_nd = kde_low_nd_bot.twinx()
# sns.distplot(two_circ_interp_contnd(ee_first_valid_area_low_only), hist=False, ax = kde_radius_init_2_circ_nd, color='r')
sns.distplot(two_circ_interp_contnd(ee_first_valid_area_low_only), hist=True, ax = kde_radius_init_2_circ_nd, hist_kws=dict(color='r', edgecolor='r', linewidth=0.5), kde_kws=dict(color='r', linewidth=0.5),)
# kde_low_nd_top.axvline(distrib_ee_low_plat_contnd_median, c='g', ls='-', zorder=10)
# kde_low_nd_top.axvline(distrib_eedbfix_contnd_median, c='orange', ls='-', zorder=10)
# kde_low_nd_bot.axvline(distrib_eedb_contnd_median, c='k', ls='-', zorder=10)
# kde_low_nd_bot.axvline(distrib_two_circ_contnd_median, c='r', ls='-', zorder=10)

# kde_low_nd_top.scatter(kde_low_cont_nd_peaks, kde_low_cont_nd_peaks_y, c='g', marker='.', s=50, zorder=10)
# kde_low_nd_bot.scatter(eedb_fix_cont_nd_peaks, eedb_fix_cont_nd_peaks_y, c='orange', marker='.', s=50, zorder=10)
# kde_low_nd_top.scatter(eedb_no_high_adh_cont_nd_peaks, eedb_no_high_adh_cont_nd_peaks_y, c='k', marker='.', s=50, zorder=10)
# [kde_low_nd_top.axvline(x, c='k', ls=':', zorder=10) for x in eedb_no_high_adh_cont_nd_peaks]
# [kde_low_nd_top.axvline(x, c='g', ls=':', zorder=9) for x in kde_low_cont_nd_peaks]
# [kde_low_nd_top.axvline(x, c='orange', ls=':', zorder=10) for x in eedb_fix_cont_nd_peaks]
# [kde_low_nd_top.axvline(x, c='r', zorder=9) for x in kde_radius_init_2_circ_nd_peaks]
# [kde_low_nd_bot.axvline(x, c='k', ls=':', zorder=10) for x in eedb_no_high_adh_cont_nd_peaks]
# [kde_low_nd_bot.axvline(x, c='g', zorder=9) for x in kde_low_cont_nd_peaks]
# [kde_low_nd_bot.axvline(x, c='orange', ls=':', zorder=10) for x in eedb_fix_cont_nd_peaks]
# [kde_low_nd_bot.axvline(x, c='r', ls=':', zorder=9) for x in kde_radius_init_2_circ_nd_peaks]
kde_low_nd_bot.set_xlabel('')
kde_low_nd_top.set_xlabel('')
kde_low_nd_bot.set_yticks([0,1,2])
kde_radius_init_2_circ_nd.set_yticks([0,10,20])
kde_low_nd_top.set_yticks([0,1,2])
# kde_low_nd_top.set_x
#kde_radius_init_2_circ.yaxis.label.set_color('green')
kde_radius_init_2_circ_nd.tick_params(axis='y', colors='red')
kde_radius_init_2_circ_nd.set_ylim(0)
kde_low_nd_top.set_xlim(0, xlim)
kde_low_nd_top.set_ylim(0)
kde_low_nd_bot.set_ylim(0)
fig_kde_low_nd.draw
fig_kde_low_nd.savefig(Path.joinpath(output_dir, r"ee_contnd_data_less_than_growth_start_manual_low_data_no_high_adh.tif"), 
                       dpi=dpi_graph, bbox_inches='tight')
# print('ecto-ecto (black): ', kde_low_cont_nd_peaks)
# print('ecto-ecto_fix (orange): ', eedb_fix_cont_nd_peaks)
#print('2_circles (red): ', kde_radius_init_2_circ_nd_peaks)
ylims_bot = kde_low_nd_bot.get_ylim()
plt.close(fig_kde_low_nd)


fig_kde_low_nd, (kde_low_nd_top, kde_low_nd_bot) = plt.subplots(figsize=(width_1col*2/3,height_1col*2/3), nrows=2, ncols=1, sharex=True)
kde_low_nd_top = sns.distplot(ee_low_plat_contnd_manual_absolute_df['Contact Surface Length (N.D.)'], kde=True, hist_kws=dict(color='g', edgecolor='g', linewidth=0.5), kde_kws=dict(color='g', linewidth=0.5), ax=kde_low_nd_top)#, rug=True, rug_kws=dict(color='k', alpha=0.5))
contnd_eedb_fix = sns.distplot(eedb_fix_contact_diam_non_dim_df_list_long_dropna['Contact Surface Length (N.D.)'], kde=True, hist_kws=dict(color='orange', edgecolor='orange', linewidth=0.5), kde_kws=dict(color='orange', linewidth=0.5), ax=kde_low_nd_top)#, rug=True, rug_kws=dict(alpha=0.5))#, ax=kde_low)#, kde_kws=dict(bw=5))
contnd_eedb = sns.distplot(eedb_contact_diam_non_dim_df_list_long_dropna['Contact Surface Length (N.D.)'], kde=True, hist_kws=dict(color='k', edgecolor='k', linewidth=0.5), kde_kws=dict(color='k', linewidth=0.5), ax=kde_low_nd_bot)
# contnd_eedb_no_high_adh = sns.distplot(eedb_contact_diam_non_dim_df_list_long_dropna_no_high_adh['Contact Surface Length (N.D.)'], kde=True, hist_kws=dict(color='k', edgecolor='k', linewidth=0.5), kde_kws=dict(color='k', linewidth=0.5), ax=kde_low_nd_bot)
kde_radius_init_2_circ_nd = kde_low_nd_bot.twinx()
# sns.distplot(two_circ_interp_contnd(ee_first_valid_area_low_only), hist=False, ax = kde_radius_init_2_circ_nd, color='r')
sns.distplot(two_circ_interp_contnd(ee_first_valid_area_low_only), hist=True, ax = kde_radius_init_2_circ_nd, hist_kws=dict(color='r', edgecolor='r', linewidth=0.5), kde_kws=dict(color='r', linewidth=0.5),)
# kde_low_nd_top.axvline(distrib_ee_low_plat_contnd_median, c='g', ls='-', zorder=10)
# kde_low_nd_top.axvline(distrib_eedbfix_contnd_median, c='orange', ls='-', zorder=10)
# kde_low_nd_bot.axvline(distrib_eedb_contnd_median, c='k', ls='-', zorder=10)
# kde_low_nd_bot.axvline(distrib_two_circ_contnd_median, c='r', ls='-', zorder=10)

# kde_low_nd_top.scatter(kde_low_cont_nd_peaks, kde_low_cont_nd_peaks_y, c='g', marker='.', s=50, zorder=10)
# kde_low_nd_bot.scatter(eedb_fix_cont_nd_peaks, eedb_fix_cont_nd_peaks_y, c='orange', marker='.', s=50, zorder=10)
# kde_low_nd_top.scatter(eedb_no_high_adh_cont_nd_peaks, eedb_no_high_adh_cont_nd_peaks_y, c='k', marker='.', s=50, zorder=10)
# [kde_low_nd_top.axvline(x, c='k', ls=':', zorder=10) for x in eedb_no_high_adh_cont_nd_peaks]
# [kde_low_nd_top.axvline(x, c='g', ls=':', zorder=9) for x in kde_low_cont_nd_peaks]
# [kde_low_nd_top.axvline(x, c='orange', ls=':', zorder=10) for x in eedb_fix_cont_nd_peaks]
# [kde_low_nd_top.axvline(x, c='r', zorder=9) for x in kde_radius_init_2_circ_nd_peaks]
# [kde_low_nd_bot.axvline(x, c='k', ls=':', zorder=10) for x in eedb_no_high_adh_cont_nd_peaks]
# [kde_low_nd_bot.axvline(x, c='g', zorder=9) for x in kde_low_cont_nd_peaks]
# [kde_low_nd_bot.axvline(x, c='orange', ls=':', zorder=10) for x in eedb_fix_cont_nd_peaks]
# [kde_low_nd_bot.axvline(x, c='r', ls=':', zorder=9) for x in kde_radius_init_2_circ_nd_peaks]
kde_low_nd_bot.set_xlabel('')
kde_low_nd_top.set_xlabel('')
kde_low_nd_bot.set_yticks([0,1,2])
kde_radius_init_2_circ_nd.set_yticks([0,10,20])
kde_low_nd_top.set_yticks([0,1,2])
# kde_low_nd_top.set_x
#kde_radius_init_2_circ.yaxis.label.set_color('green')
kde_radius_init_2_circ_nd.tick_params(axis='y', colors='red')
kde_radius_init_2_circ_nd.set_ylim(0)
kde_low_nd_top.set_xlim(0, xlim)
kde_low_nd_top.set_ylim(0)
kde_low_nd_bot.set_ylim(ylims_bot)
fig_kde_low_nd.draw
fig_kde_low_nd.savefig(Path.joinpath(output_dir, r"ee_contnd_data_less_than_growth_start_manual_low_data.tif"), 
                       dpi=dpi_graph, bbox_inches='tight')
# print('ecto-ecto (black): ', kde_low_cont_nd_peaks)
# print('ecto-ecto_fix (orange): ', eedb_fix_cont_nd_peaks)
#print('2_circles (red): ', kde_radius_init_2_circ_nd_peaks)
plt.close(fig_kde_low_nd)


# Output the statistical results into a .tsv:
low_adh_stats_headers = ('sample 1', 'sample 2', 'F_onewayResult')
eelow_vs_eedb = ('ee_low_plat', 'eedb',  
                 stats.f_oneway(ee_low_plat_contnd_manual_absolute_df['Contact Surface Length (N.D.)'], eedb_contact_diam_non_dim_df_list_long_dropna_no_high_adh['Contact Surface Length (N.D.)']))
eelow_vs_eedbfix = ('ee_low_plat', 'eedb_fix', 
                    stats.f_oneway(ee_low_plat_contnd_manual_absolute_df['Contact Surface Length (N.D.)'], eedb_fix_contact_diam_non_dim_df_list_long_dropna['Contact Surface Length (N.D.)']))
eelow_vs_twocirc = ('ee_low_plat', 'two_circle_interp', 
                    stats.f_oneway(ee_low_plat_contnd_manual_absolute_df['Contact Surface Length (N.D.)'], two_circ_interp_contnd(ee_first_valid_area_low_only)))
eedb_vs_eedbfix = ('eedb', 'eedb_fix', 
                   stats.f_oneway(eedb_contact_diam_non_dim_df_list_long_dropna_no_high_adh['Contact Surface Length (N.D.)'], eedb_fix_contact_diam_non_dim_df_list_long_dropna['Contact Surface Length (N.D.)']))
eedbfix_vs_twocirc = ('eedb_fix', 'two_circ_interp', 
                      stats.f_oneway(eedb_fix_contact_diam_non_dim_df_list_long_dropna['Contact Surface Length (N.D.)'], two_circ_interp_contnd(ee_first_valid_area_low_only)))
low_adh_stats_contnd = [low_adh_stats_headers, eelow_vs_eedb, 
                        eelow_vs_eedbfix, eelow_vs_twocirc, 
                        eedb_vs_eedbfix, eedbfix_vs_twocirc]


with open(Path.joinpath(output_dir, r'ee_contnd_data_less_than_growth_manual_stats.tsv'), 'w', 
          newline='') as file_out:
    tsv_obj = csv.writer(file_out, delimiter='\t')
    tsv_obj.writerows(low_adh_stats_contnd)






fig_kde_low_nd, kde_low_nd = plt.subplots(figsize=(width_1col,height_1col))
# kde_low_nd = sns.distplot(ee_low_plat_contnd_manual_absolute_df['Contact Surface Length (N.D.)'], kde=True, hist_kws=dict(color='g', edgecolor='g', linewidth=2), kde_kws=dict(color='g'))#, rug=True, rug_kws=dict(color='k', alpha=0.5))
# contnd_eedb_fix = sns.distplot(eedb_fix_contact_diam_non_dim_df_list_long_dropna['Contact Surface Length (N.D.)'], kde=True, hist_kws=dict(color='orange', edgecolor='orange', linewidth=2), kde_kws=dict(color='orange'))#, rug=True, rug_kws=dict(alpha=0.5))#, ax=kde_low)#, kde_kws=dict(bw=5))
contnd_eedb = sns.distplot(eedb_contact_diam_non_dim_df_list_long_dropna['Contact Surface Length (N.D.)'], kde=True, hist_kws=dict(color='r', edgecolor='k', linewidth=2), kde_kws=dict(color='r'), label='eedb')
contnd_eedb_no_high_adh = sns.distplot(eedb_contact_diam_non_dim_df_list_long_dropna_no_high_adh['Contact Surface Length (N.D.)'], kde=True, hist_kws=dict(color='k', edgecolor='k', linewidth=2), kde_kws=dict(color='k'), label='eedb no \nhigh adh.')
# kde_radius_init_2_circ_nd = kde_low_nd.twinx()
# sns.distplot(two_circ_interp_contnd(ee_first_valid_area_low_only), hist=False, ax = kde_radius_init_2_circ_nd, color='r')
# kde_low_nd.scatter(kde_low_cont_nd_peaks, kde_low_cont_nd_peaks_y, c='g', marker='o', s=3)
# kde_low_nd.scatter(eedb_fix_cont_nd_peaks, eedb_fix_cont_nd_peaks_y, c='orange', marker='o', s=3)
kde_low_nd.scatter(eedb_cont_nd_peaks, eedb_cont_nd_peaks_y, c='r', marker='.', s=50, zorder=10)
kde_low_nd.scatter(eedb_no_high_adh_cont_nd_peaks, eedb_no_high_adh_cont_nd_peaks_y, c='k', marker='.', s=50, zorder=10)
kde_low_nd.set_xlabel('')
#kde_radius_init_2_circ.yaxis.label.set_color('green')
# kde_radius_init_2_circ_nd.tick_params(axis='y', colors='red')
# kde_radius_init_2_circ_nd.set_ylim(0)
kde_low_nd.set_ylim(0)
fig_kde_low_nd.draw
kde_low_nd.legend()
fig_kde_low_nd.savefig(Path.joinpath(output_dir, r"eedb_all_vs_eedb_no_high_adh.tif"), dpi=dpi_graph)
# print('ecto-ecto (black): ', kde_low_cont_nd_peaks)
# print('ecto-ecto_fix (orange): ', eedb_fix_cont_nd_peaks)
#print('2_circles (red): ', kde_radius_init_2_circ_nd_peaks)
plt.close(fig_kde_low_nd)




low_plat_peaks_contnd = [['Experimental_Condition', 'contnd_val', 'peak_height']]
[low_plat_peaks_contnd.append(['ecto-ecto low plateau',*x,]) for x in zip(kde_low_cont_nd_peaks, kde_low_cont_nd_peaks_y)]
[low_plat_peaks_contnd.append(['eedb_fix',*x,]) for x in zip(eedb_fix_cont_nd_peaks, eedb_fix_cont_nd_peaks_y)]
[low_plat_peaks_contnd.append(['two_circles',*x,]) for x in zip(kde_radius_init_2_circ_nd_peaks, kde_radius_init_2_circ_nd_peaks_y)]
[low_plat_peaks_contnd.append(['eedb',*x,]) for x in zip(eedb_cont_nd_peaks, eedb_cont_nd_peaks_y)]
[low_plat_peaks_contnd.append(['eedb no high adh',*x,]) for x in zip(eedb_no_high_adh_cont_nd_peaks, eedb_no_high_adh_cont_nd_peaks_y)]


with open(Path.joinpath(output_dir, r'eedbFix_vs_eeLowPlat_peaks_contnd.tsv'), 'w', 
          newline='') as file_out:
    tsv_obj = csv.writer(file_out, delimiter='\t')
    tsv_obj.writerows(low_plat_peaks_contnd)



###############################################################################        
# Plot all ecto-ecto in one graph after time-correcting them:
fig6, ax6 = plt.subplots(figsize=(width_1col, height_1col))
for i,x in enumerate(ee_contact_diam_non_dim_df_list_dropna):
    if ee_list_dropna_fit_validity_df['growth_phase'].iloc[i]:
        ax6.scatter(x['Time (minutes)']-ee_indiv_logistic_contnd_df.iloc[i]['popt_x2'], x['Contact Surface Length (N.D.)'], marker='.', s=1)
ax6.axvline(growth_phase_start_ee_contnd, c='k', ls=':', lw=1)
ax6.axvline(growth_phase_end_ee_contnd, c='k', ls=':', lw=1)
fig6.savefig(Path.joinpath(output_dir, r"ee_contactnds.tif"), dpi=dpi_graph)
fig6.clf()

# Plot all ecto-ecto in one graph after time-correcting them:
fig4, ax4 = plt.subplots(figsize=(width_1col, height_1col))
for i,x in enumerate(ee_list_dropna):
    if ee_list_dropna_fit_validity_df['growth_phase'].iloc[i]:
        ax4.scatter(x['Time (minutes)']-ee_indiv_logistic_cont_df.iloc[i]['popt_x2'], x['Contact Surface Length (px)']*um_per_pixel, marker='.', s=1)
#ax4.axvline(growth_phase_start_ee_contnd)
#ax4.axvline(growth_phase_end_ee_contnd)
fig4.savefig(Path.joinpath(output_dir, r"ee_contacts.tif"), dpi=dpi_graph)
fig4.clf()

# Plot all ecto-ecto in one graph after time-correcting them:
fig5, ax5 = plt.subplots(figsize=(width_1col, height_1col))
for i,x in enumerate(ee_list_dropna):
    if ee_list_dropna_angle_fit_validity_df['growth_phase'].iloc[i]:
        ax5.scatter(x['Time (minutes)']-ee_indiv_logistic_angle_df.iloc[i]['popt_x2'], x['Average Angle (deg)'], marker='.', s=1)
#ax5.axvline(growth_phase_start_ee_contnd)
#ax5.axvline(growth_phase_end_ee_contnd)
fig5.savefig(Path.joinpath(output_dir, r"ee_angles.tif"), dpi=dpi_graph)
fig5.clf()

#Also plot the data that could not be time corrected (if any):
fig6, ax6 = plt.subplots(figsize=(width_1col, height_1col))
for i,x in enumerate(ee_contact_diam_non_dim_df_list_dropna):
    if ee_list_dropna_fit_validity_df['growth_phase'].iloc[i] == False:
        ax6.scatter(x['Time (minutes)']-ee_indiv_logistic_contnd_df.iloc[i]['popt_x2'], x['Contact Surface Length (N.D.)'], marker='.', s=5)
        ax6.plot(x['Time (minutes)']-ee_indiv_logistic_contnd_df.iloc[i]['popt_x2'], x['Contact Surface Length (N.D.)'], lw=1)
ax6.set_ylim(0, 4.0)
#ax6.axvline(growth_phase_start_ee_contnd, c='k', ls=':', lw=1)
#ax6.axvline(growth_phase_end_ee_contnd, c='k', ls=':', lw=1)
fig6.savefig(Path.joinpath(output_dir, r"ee_contactnds_bad_fit.tif"), dpi=dpi_graph)
fig6.clf()

fig4, ax4 = plt.subplots(figsize=(width_1col, height_1col))
for i,x in enumerate(ee_list_dropna):
    if ee_list_dropna_fit_validity_df['growth_phase'].iloc[i] == False:
        ax4.scatter(x['Time (minutes)']-ee_indiv_logistic_cont_df.iloc[i]['popt_x2'], x['Contact Surface Length (px)']*um_per_pixel, marker='.', s=5)
        ax4.plot(x['Time (minutes)']-ee_indiv_logistic_cont_df.iloc[i]['popt_x2'], x['Contact Surface Length (px)']*um_per_pixel, lw=1)
#ax4.axvline(growth_phase_start_ee_contnd)
#ax4.axvline(growth_phase_end_ee_contnd)
fig4.savefig(Path.joinpath(output_dir, r"ee_contacts_bad_fit.tif"), dpi=dpi_graph)
fig4.clf()

fig5, ax5 = plt.subplots(figsize=(width_1col, height_1col))
for i,x in enumerate(ee_list_dropna):
    if ee_list_dropna_angle_fit_validity_df['growth_phase'].iloc[i] == False:
        ax5.scatter(x['Time (minutes)']-ee_indiv_logistic_angle_df.iloc[i]['popt_x2'], x['Average Angle (deg)'], marker='.', s=5)
        ax5.plot(x['Time (minutes)']-ee_indiv_logistic_angle_df.iloc[i]['popt_x2'], x['Average Angle (deg)'], lw=1)
#ax5.axvline(growth_phase_start_ee_contnd)
#ax5.axvline(growth_phase_end_ee_contnd)
ax5.set_ylim(0, 220)
fig5.savefig(Path.joinpath(output_dir, r"ee_angles_bad_fit.tif"), dpi=dpi_graph)
fig5.clf()



# Return the timepoint where the n-size is no longer 2 or more:
#n_size_thresh_for_xlim = 1
if np.any(ee_fit_contact_mean):
    mlx = MultipleLocator(10)
    mly = MultipleLocator(0.5)
    last_multi_n = len((ee_fit_contact_mean['count'] > n_size_thresh_for_xlim).values) - 1 - np.argmax((ee_fit_contact_mean['count'] > n_size_thresh_for_xlim).values[::-1])
    first_multi_n = np.argmax((ee_fit_contact_mean['count'] > n_size_thresh_for_xlim).values)
    fig, ax = plt.subplots(figsize=(width_1pt5col, height_1pt5col))
    n_size = ee_fit_contact_mean['count']
    cmap = plt.cm.get_cmap('jet')#, n_size.max())
    #norm = mpl.colors.LogNorm(vmin=n_size.min()-0.5, vmax=n_size.max()+0.5)    
    ax.errorbar(ee_fit_contact_mean['Time (minutes)'], ee_fit_contact_mean['mean'], ee_fit_contact_mean['std'], marker='None', ls='None', elinewidth=1, ecolor='grey')
    graph = ax.scatter(ee_fit_contact_mean['Time (minutes)'], ee_fit_contact_mean['mean'], marker='.', s=10, c=n_size, cmap=cmap, zorder=9, norm=matplotlib.colors.LogNorm())
    ax.plot(x_fit, ee_genlog_y_contnd, zorder=10, c='k')
    # ax.plot(x_fit, ee_genlog_y_contnd_weightless, zorder=10, c='k')
    # ax.set_xlim(ee_fit_contact_mean['Time (minutes)'].min()-1, ee_fit_contact_mean['Time (minutes)'].max()+1)
    ax.set_xlim(ee_fit_contact_mean['Time (minutes)'].iloc[first_multi_n]-1, ee_fit_contact_mean['Time (minutes)'].iloc[last_multi_n]+1)
    #ax.axvline(growth_phase_start_ee_genlog_contnd)
    #ax.axvline(growth_phase_end_ee_genlog_contnd)
    #ax.scatter(growth_phase_start_ee_contnd, growth_phase_start_ee_contnd_val, marker='x', c='r', zorder=10)
    #ax.scatter(growth_phase_end_ee_contnd, growth_phase_end_ee_contnd_val, marker='x', c='r', zorder=10)
    ax.axvline(growth_phase_start_ee_contnd)
    ax.axvline(growth_phase_end_ee_contnd)
    ax.axvline(growth_phase_start_ee_contnd_strict, c='k', ls='--')
    ax.axvline(growth_phase_end_ee_contnd_strict, c='k', ls='--')
    ax.axhline(eedb_fix_contact_diam_non_dim_df_list_long_dropna['Contact Surface Length (N.D.)'].mean(), c='k', ls='--', lw=0.5)
    ax.axhline(two_circ_interp_contnd(ee_first_valid_area).mean(), c='r', ls='--', lw=0.5)
    ax.set_ylim(0, 4.0)
    ax.xaxis.set_minor_locator(mlx)
    ax.yaxis.set_minor_locator(mly)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(graph, cax=cax)
    #cbar_ticks = np.unique([int(x) for x in cbar.get_ticks()])
    #cax.clear()
    #cbar = plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax, ticks=cbar_ticks)
    ax.draw
    fig.savefig(Path.joinpath(output_dir, r"ee_contactnds_gen_logistic.tif"), dpi=dpi_graph)
    fig.clf()
    
    
if np.any(ee_contact_mean):
    last_multi_n = len((ee_contact_mean['count'] > n_size_thresh_for_xlim).values) - 1 - np.argmax((ee_contact_mean['count'] > n_size_thresh_for_xlim).values[::-1])
    first_multi_n = np.argmax((ee_contact_mean['count'] > n_size_thresh_for_xlim).values)
    fig, ax = plt.subplots(figsize=(width_1pt5col, height_1pt5col))
    n_size = ee_contact_mean['count']
    cmap = plt.cm.get_cmap('jet')#, n_size.max())
    #norm = mpl.colors.Normalize(vmin=n_size.min()-0.5, vmax=n_size.max()+0.5)    
    ax.errorbar(ee_contact_mean['Time (minutes)'], ee_contact_mean['mean'], ee_contact_mean['std'], marker='None', ls='None', elinewidth=1, ecolor='grey')
    graph = ax.scatter(ee_contact_mean['Time (minutes)'], ee_contact_mean['mean'], marker='.', s=10, c=n_size, cmap=cmap, zorder=9, norm=matplotlib.colors.LogNorm())
    ax.plot(x_fit, ee_genlog_y_cont, zorder=10, c='k')
    ax.set_xlim(ee_contact_mean['Time (minutes)'].iloc[first_multi_n]-1, ee_contact_mean['Time (minutes)'].iloc[last_multi_n]+1)
    # ax.set_xlim(ee_contact_mean['Time (minutes)'].min()-1, ee_contact_mean['Time (minutes)'].max()+1)
    ax.axvline(growth_phase_start_ee_cont)
    ax.axvline(growth_phase_end_ee_cont)
    #ax.axhline(eedb_fix_contact_diam_non_dim_df_list_long_dropna['Contact Surface Length (N.D.)'].mean(), c='k', ls='--', lw=0.5)
    #ax.axhline(two_circ_interp_contnd(ee_first_valid_area).mean(), c='r', ls='--', lw=0.5)
    ax.set_ylim(0)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(graph, cax=cax)
    #cbar_ticks = np.unique([int(x) for x in cbar.get_ticks()])
    #cax.clear()
    #cbar = plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax, ticks=cbar_ticks)
    ax.draw
    fig.savefig(Path.joinpath(output_dir, r"ee_contact_gen_logistic.tif"), dpi=dpi_graph)
    fig.clf()
    
    
if np.any(ee_angle_mean):
    mlx = MultipleLocator(10)
    mly = MultipleLocator(10)
    last_multi_n = len((ee_angle_mean['count'] > n_size_thresh_for_xlim).values) - 1 - np.argmax((ee_angle_mean['count'] > n_size_thresh_for_xlim).values[::-1])
    first_multi_n = np.argmax((ee_angle_mean['count'] > n_size_thresh_for_xlim).values)
    fig2, ax2 = plt.subplots(figsize=(width_1pt5col, height_1pt5col))
    n_size = ee_angle_mean['count']
    cmap = plt.cm.get_cmap('jet')#, n_size.max())
    #norm = mpl.colors.Normalize(vmin=n_size.min()-0.5, vmax=n_size.max()+0.5)    
    ax2.errorbar(ee_angle_mean['Time (minutes)'], ee_angle_mean['mean'], ee_angle_mean['std'], marker='None', ls='None', elinewidth=1, ecolor='grey')
    graph = ax2.scatter(ee_angle_mean['Time (minutes)'], ee_angle_mean['mean'], marker='.', s=10, c=n_size, cmap=cmap, zorder=9, norm=matplotlib.colors.LogNorm())
    ax2.plot(x_fit, ee_genlog_y_angle, zorder=10, c='k')
    #ax2.scatter(growth_phase_start_ee_angle, growth_phase_start_ee_angle_val, marker='x', c='r', zorder=10)
    #ax2.scatter(growth_phase_end_ee_angle, growth_phase_end_ee_angle_val, marker='x', c='r', zorder=10)
    ax2.set_xlim(ee_angle_mean['Time (minutes)'].iloc[first_multi_n]-1, ee_angle_mean['Time (minutes)'].iloc[last_multi_n]+1)
    # ax2.set_xlim(ee_angle_mean['Time (minutes)'].min()-1, ee_angle_mean['Time (minutes)'].max()+1)
    #ax2.axvline(growth_phase_start_ee_genlog_angle)
    #ax2.axvline(growth_phase_end_ee_genlog_angle)
    ax2.axvline(growth_phase_start_ee_angle)
    ax2.axvline(growth_phase_end_ee_angle)
    ax2.axvline(growth_phase_start_ee_angle_strict, c='k', ls='--')
    ax2.axvline(growth_phase_end_ee_angle_strict, c='k', ls='--')
    ax2.axhline(eedb_fix_contact_diam_non_dim_df_list_long_dropna['Average Angle (deg)'].mean(), c='k', ls='--', lw=0.5)
    ax2.axhline(two_circ_interp_angle(ee_first_valid_area).mean(),c='r', ls='--', lw=0.5)
    ax2.yaxis.set_ticks(np.arange(0, 210, 30))
    ax2.xaxis.set_minor_locator(mlx)
    ax2.yaxis.set_minor_locator(mly)
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(graph, cax=cax)
    #cbar_ticks = np.unique([int(x) for x in cbar.get_ticks()])
    #cax.clear()
    #cbar = plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax, ticks=cbar_ticks)
    ax2.draw
    fig2.savefig(Path.joinpath(output_dir, r"ee_angles_gen_logistic.tif"), dpi=dpi_graph)
    fig2.clf()
    
    
if np.any(ee_angle_mean_for_contnd_fit):
    last_multi_n = len((ee_angle_mean_for_contnd_fit['count'] > n_size_thresh_for_xlim).values) - 1 - np.argmax((ee_angle_mean_for_contnd_fit['count'] > n_size_thresh_for_xlim).values[::-1])
    first_multi_n = np.argmax((ee_angle_mean_for_contnd_fit['count'] > n_size_thresh_for_xlim).values)
    fig, ax = plt.subplots(figsize=(width_2col, height_2col))
    n_size = ee_angle_mean_for_contnd_fit['count']
    cmap = plt.cm.get_cmap('jet')#, n_size.max())
    #norm = mpl.colors.Normalize(vmin=n_size.min()-0.5, vmax=n_size.max()+0.5)    
    ax.errorbar(ee_angle_mean_for_contnd_fit['Time (minutes)'], ee_angle_mean_for_contnd_fit['mean'], ee_angle_mean_for_contnd_fit['std'], marker='None', ls='None', elinewidth=1, ecolor='grey')
    graph = ax.scatter(ee_angle_mean_for_contnd_fit['Time (minutes)'], ee_angle_mean_for_contnd_fit['mean'], marker='.', s=10, c=n_size, cmap=cmap, zorder=9, norm=matplotlib.colors.LogNorm())
    ax.plot(x_fit, ee_genlog_y_angle_contndfit, zorder=10, c='k')
    ax.set_xlim(ee_angle_mean_for_contnd_fit['Time (minutes)'].min()-1, ee_angle_mean_for_contnd_fit['Time (minutes)'].max()+1)
    #ax.axvline(growth_phase_start_ee_angle)
    #ax.axvline(growth_phase_end_ee_angle)
    #ax.axvline(growth_phase_start_ee_angle_strict, c='k', ls='--')
    #ax.axvline(growth_phase_end_ee_angle_strict, c='k', ls='--')
    ax.axhline(eedb_fix_contact_diam_non_dim_df_list_long_dropna['Average Angle (deg)'].mean(), c='k', ls='--', lw=0.5)
    ax.axhline(two_circ_interp_angle(ee_first_valid_area).mean(),c='r', ls='--', lw=0.5)
    ax.yaxis.set_ticks(np.arange(0, 220, 20))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(graph, cax=cax)
    #cbar_ticks = np.unique([int(x) for x in cbar.get_ticks()])
    #cax.clear()
    #cbar = plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax, ticks=cbar_ticks)
    ax.draw
    fig.savefig(Path.joinpath(output_dir, r"ee_angle_mean_for_contnd_fit.tif"), dpi=dpi_graph)
    fig.clf()
    
    
if np.any(ee_contnd_mean_for_angle_fit):
    last_multi_n = len((ee_contnd_mean_for_angle_fit['count'] > n_size_thresh_for_xlim).values) - 1 - np.argmax((ee_contnd_mean_for_angle_fit['count'] > n_size_thresh_for_xlim).values[::-1])
    first_multi_n = np.argmax((ee_contnd_mean_for_angle_fit['count'] > n_size_thresh_for_xlim).values)
    fig2, ax2 = plt.subplots(figsize=(width_2col, height_2col))
    n_size = ee_contnd_mean_for_angle_fit['count']
    cmap = plt.cm.get_cmap('jet')#, n_size.max())
    #norm = mpl.colors.Normalize(vmin=n_size.min()-0.5, vmax=n_size.max()+0.5)    
    ax2.errorbar(ee_contnd_mean_for_angle_fit['Time (minutes)'], ee_contnd_mean_for_angle_fit['mean'], ee_contnd_mean_for_angle_fit['std'], marker='None', ls='None', elinewidth=1, ecolor='grey')
    graph = ax2.scatter(ee_contnd_mean_for_angle_fit['Time (minutes)'], ee_contnd_mean_for_angle_fit['mean'], marker='.', s=10, c=n_size, cmap=cmap, zorder=9, norm=matplotlib.colors.LogNorm())
    ax2.plot(x_fit, ee_genlog_y_contnd_anglefit, zorder=10, c='k')
    ax2.set_xlim(ee_contnd_mean_for_angle_fit['Time (minutes)'].min()-1, ee_contnd_mean_for_angle_fit['Time (minutes)'].max()+1)
    #ax2.axvline(growth_phase_start_ee_contnd)
    #ax2.axvline(growth_phase_end_ee_contnd)
    #ax2.axvline(growth_phase_start_ee_contnd_strict, c='k', ls='--')
    #ax2.axvline(growth_phase_end_ee_contnd_strict, c='k', ls='--')
    ax2.axhline(eedb_fix_contact_diam_non_dim_df_list_long_dropna['Contact Surface Length (N.D.)'].mean(), c='k', ls='--', lw=0.5)
    ax2.axhline(two_circ_interp_contnd(ee_first_valid_area).mean(), c='r', ls='--', lw=0.5)
    ax2.set_ylim(0, 4.0)
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(graph, cax=cax)
    #cbar_ticks = np.unique([int(x) for x in cbar.get_ticks()])
    #cax.clear()
    #cbar = plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax, ticks=cbar_ticks)
    ax2.draw
    fig2.savefig(Path.joinpath(output_dir, r"ee_contnd_mean_for_angle_fit.tif"), dpi=dpi_graph)
    fig2.clf()
    
    
    
if np.any(ee_angle_mean_cant_time_corr):
    last_multi_n = len((ee_angle_mean_cant_time_corr['count'] > n_size_thresh_for_xlim).values) - 1 - np.argmax((ee_angle_mean_cant_time_corr['count'] > n_size_thresh_for_xlim).values[::-1])
    first_multi_n = np.argmax((ee_angle_mean_cant_time_corr['count'] > n_size_thresh_for_xlim).values)
    fig, ax = plt.subplots(figsize=(width_2col, height_2col))
    n_size = ee_angle_mean_cant_time_corr['count']
    cmap = plt.cm.get_cmap('jet')#, n_size.max())
    #norm = mpl.colors.Normalize(vmin=n_size.min()-0.5, vmax=n_size.max()+0.5)    
    ax.errorbar(ee_angle_mean_cant_time_corr['Time (minutes)'], ee_angle_mean_cant_time_corr['mean'], ee_angle_mean_cant_time_corr['std'], ls='None', elinewidth=1, ecolor='grey')
    graph = ax.scatter(ee_angle_mean_cant_time_corr['Time (minutes)'], ee_angle_mean_cant_time_corr['mean'], marker='.', s=10, c=n_size, cmap=cmap, zorder=9)#, norm=matplotlib.colors.LogNorm()
    ax.set_xlim(ee_angle_mean_cant_time_corr['Time (minutes)'].min()-1, ee_angle_mean_cant_time_corr['Time (minutes)'].max()+1)
    ax.axhline(ee_genlog_popt_angle[3], c='g', ls='--')
    ax.axhline(ee_genlog_popt_angle[2], c='g', ls='--')
    ax.axhline(two_circ_interp_angle(ee_first_valid_area).mean(), c='g', ls='-', lw=0.5)
    #ax.axhline(two_circ_interp_angle(mm_first_valid_area).mean(), c='r', ls='--', lw=0.5)
    #ax.axhline(two_circ_interp_angle(nn_first_valid_area).mean(), c='b', ls='--', lw=0.5)
    ax.set_ylim(0, 220)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(graph, cax=cax)
    #cbar_ticks = np.unique([int(x) for x in cbar.get_ticks()])
    #cax.clear()
    #cbar = plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax, ticks=cbar_ticks)
    ax.draw
    fig.savefig(Path.joinpath(output_dir, r"ee_angle_mean_cant_time_corr.tif"), dpi=dpi_graph)
    fig.clf()
    
    
if np.any(ee_fit_contact_mean_cant_time_corr):
    last_multi_n = len((ee_fit_contact_mean_cant_time_corr['count'] > n_size_thresh_for_xlim).values) - 1 - np.argmax((ee_fit_contact_mean_cant_time_corr['count'] > n_size_thresh_for_xlim).values[::-1])
    first_multi_n = np.argmax((ee_fit_contact_mean_cant_time_corr['count'] > n_size_thresh_for_xlim).values)
    fig, ax = plt.subplots(figsize=(width_2col, height_2col))
    n_size = ee_fit_contact_mean_cant_time_corr['count']
    cmap = plt.cm.get_cmap('jet')#, n_size.max())
    #norm = mpl.colors.Normalize(vmin=n_size.min()-0.5, vmax=n_size.max()+0.5)    
    ax.errorbar(ee_fit_contact_mean_cant_time_corr['Time (minutes)'], ee_fit_contact_mean_cant_time_corr['mean'], ee_fit_contact_mean_cant_time_corr['std'], ls='None', elinewidth=1, ecolor='grey')
    graph = ax.scatter(ee_fit_contact_mean_cant_time_corr['Time (minutes)'], ee_fit_contact_mean_cant_time_corr['mean'], marker='.', s=10, c=n_size, cmap=cmap, zorder=9)#, norm=matplotlib.colors.LogNorm()
    ax.set_xlim(ee_fit_contact_mean_cant_time_corr['Time (minutes)'].min()-1, ee_fit_contact_mean_cant_time_corr['Time (minutes)'].max()+1)
    ax.axhline(ee_genlog_popt_contnd[3], c='g', ls='--')
    ax.axhline(ee_genlog_popt_contnd[2], c='g', ls='--')
    ax.axhline(two_circ_interp_contnd(ee_first_valid_area).mean(), c='g', ls='-', lw=0.5)
    #ax.axhline(two_circ_interp_contnd(mm_first_valid_area).mean(), c='r', ls='--', lw=0.5)
    #ax.axhline(two_circ_interp_contnd(nn_first_valid_area).mean(), c='b', ls='--', lw=0.5)
    ax.set_ylim(0, 4.0)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(graph, cax=cax)
    #cbar_ticks = np.unique([int(x) for x in cbar.get_ticks()])
    #cax.clear()
    #cbar = plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax, ticks=cbar_ticks)
    ax.draw
    fig.savefig(Path.joinpath(output_dir, r"ee_fit_contact_mean_cant_time_corr.tif"), dpi=dpi_graph)
    fig.clf()





###############################################################################

# Plot all meso-meso in one graph after time-correcting them:
fig6, ax6 = plt.subplots(figsize=(width_1col, height_1col))
for i,x in enumerate(mm_contact_diam_non_dim_df_list_dropna):
    if mm_list_dropna_fit_validity_df['growth_phase'].iloc[i]:
        ax6.scatter(x['Time (minutes)']-mm_indiv_logistic_contnd_df.iloc[i]['popt_x2'], x['Contact Surface Length (N.D.)'], marker='.', s=1)
fig6.savefig(Path.joinpath(output_dir, r"mm_contactnds.tif"), dpi=dpi_graph)
fig6.clf()

# Plot all meso-meso in one graph after time-correcting them:
fig4, ax4 = plt.subplots(figsize=(width_1col, height_1col))
for i,x in enumerate(mm_list_dropna):
    if mm_list_dropna_fit_validity_df['growth_phase'].iloc[i]:
        ax4.scatter(x['Time (minutes)']-mm_indiv_logistic_cont_df.iloc[i]['popt_x2'], x['Contact Surface Length (px)']*um_per_pixel, marker='.', s=1)
fig4.savefig(Path.joinpath(output_dir, r"mm_contacts.tif"), dpi=dpi_graph)
fig4.clf()

# Plot all meso-meso in one graph after time-correcting them:
fig5, ax5 = plt.subplots(figsize=(width_1col, height_1col))
for i,x in enumerate(mm_list_dropna):
    if mm_list_dropna_angle_fit_validity_df['growth_phase'].iloc[i]:
        ax5.scatter(x['Time (minutes)']-mm_indiv_logistic_angle_df.iloc[i]['popt_x2'], x['Average Angle (deg)'], marker='.', s=1)
fig5.savefig(Path.joinpath(output_dir, r"mm_angles.tif"), dpi=dpi_graph)
fig5.clf()

#Also plot the data that could not be time corrected (if any):
fig6, ax6 = plt.subplots(figsize=(width_1col, height_1col))
for i,x in enumerate(mm_contact_diam_non_dim_df_list_dropna):
    if mm_list_dropna_fit_validity_df['growth_phase'].iloc[i] == False:
        ax6.scatter(x['Time (minutes)']-mm_indiv_logistic_contnd_df.iloc[i]['popt_x2'], x['Contact Surface Length (N.D.)'], marker='.', s=5)
        ax6.plot(x['Time (minutes)']-mm_indiv_logistic_contnd_df.iloc[i]['popt_x2'], x['Contact Surface Length (N.D.)'], lw=1)
ax6.set_ylim(0, 4.0)
fig6.savefig(Path.joinpath(output_dir, r"mm_contactnds_bad_fit.tif"), dpi=dpi_graph)
fig6.clf()

fig4, ax4 = plt.subplots(figsize=(width_1col, height_1col))
for i,x in enumerate(mm_list_dropna):
    if mm_list_dropna_fit_validity_df['growth_phase'].iloc[i] == False:
        ax4.scatter(x['Time (minutes)']-mm_indiv_logistic_cont_df.iloc[i]['popt_x2'], x['Contact Surface Length (px)']*um_per_pixel, marker='.', s=5)
        ax4.plot(x['Time (minutes)']-mm_indiv_logistic_cont_df.iloc[i]['popt_x2'], x['Contact Surface Length (px)']*um_per_pixel, lw=1)
fig4.savefig(Path.joinpath(output_dir, r"mm_contacts_bad_fit.tif"), dpi=dpi_graph)
fig4.clf()

fig5, ax5 = plt.subplots(figsize=(width_1col, height_1col))
for i,x in enumerate(mm_list_dropna):
    if mm_list_dropna_angle_fit_validity_df['growth_phase'].iloc[i] == False:
        ax5.scatter(x['Time (minutes)']-mm_indiv_logistic_angle_df.iloc[i]['popt_x2'], x['Average Angle (deg)'], marker='.', s=5)
        ax5.plot(x['Time (minutes)']-mm_indiv_logistic_angle_df.iloc[i]['popt_x2'], x['Average Angle (deg)'], lw=1)
ax5.set_ylim(0, 220)
fig5.savefig(Path.joinpath(output_dir, r"mm_angles_bad_fit.tif"), dpi=dpi_graph)
fig5.clf()



# Plot the meso-meso with the data points color coded according to N size:
#n_size_thresh_for_xlim = 1
if np.any(mm_fit_contact_mean):
    mlx = MultipleLocator(10)
    mly = MultipleLocator(0.5)
    last_multi_n = len((mm_fit_contact_mean['count'] > n_size_thresh_for_xlim).values) - 1 - np.argmax((mm_fit_contact_mean['count'] > n_size_thresh_for_xlim).values[::-1])
    first_multi_n = np.argmax((mm_fit_contact_mean['count'] > n_size_thresh_for_xlim).values)
    fig, ax = plt.subplots(figsize=(width_1col, height_1col))
    n_size = mm_fit_contact_mean['count']
    cmap = plt.cm.get_cmap('jet', n_size.max())
    norm = mpl.colors.Normalize(vmin=n_size.min()-0.5, vmax=n_size.max()+0.5)    
    ax.errorbar(mm_fit_contact_mean['Time (minutes)'], mm_fit_contact_mean['mean'], mm_fit_contact_mean['std'], marker='None', ls='None', elinewidth=1, ecolor='grey')
    graph = ax.scatter(mm_fit_contact_mean['Time (minutes)'], mm_fit_contact_mean['mean'], marker='.', s=10, c=n_size, cmap=cmap, zorder=9)#, norm=matplotlib.colors.LogNorm()
    ax.plot(x_fit, mm_genlog_y_contnd, zorder=10, c='k')
    ax.set_xlim(mm_fit_contact_mean['Time (minutes)'].iloc[first_multi_n]-1, mm_fit_contact_mean['Time (minutes)'].iloc[last_multi_n]+1)
    # ax.set_xlim(mm_fit_contact_mean['Time (minutes)'].min()-1, mm_fit_contact_mean['Time (minutes)'].max()+1)
    ax.axvline(growth_phase_start_mm_contnd)
    ax.axvline(growth_phase_end_mm_contnd)
    ax.axhline(ee_genlog_popt_contnd[3], c='g', ls='--')
    ax.axhline(ee_genlog_popt_contnd[2], c='g', ls='--')
    ax.axhline(two_circ_interp_contnd(ee_first_valid_area).mean(), c='g', ls='-', lw=0.5)
    ax.axhline(two_circ_interp_contnd(mm_first_valid_area).mean(), c='r', ls='--', lw=0.5)
    ax.set_ylim(0, 4.0)
    ax.xaxis.set_minor_locator(mlx)
    ax.yaxis.set_minor_locator(mly)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(graph, cax=cax)
    cbar_ticks = np.unique([int(x) for x in cbar.get_ticks()])
    cax.clear()
    cbar = plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax, ticks=cbar_ticks)
    ax.draw
    fig.savefig(Path.joinpath(output_dir, r"mm_contactnds_gen_logistic.tif"), dpi=dpi_graph)
    fig.clf()


if np.any(mm_angle_mean):
    last_multi_n = len((mm_angle_mean['count'] > n_size_thresh_for_xlim).values) - 1 - np.argmax((mm_angle_mean['count'] > n_size_thresh_for_xlim).values[::-1])
    first_multi_n = np.argmax((mm_angle_mean['count'] > n_size_thresh_for_xlim).values)
    fig2, ax2 = plt.subplots(figsize=(width_2col, height_2col))
    n_size = mm_angle_mean['count']
    cmap = plt.cm.get_cmap('jet', n_size.max())
    norm = mpl.colors.Normalize(vmin=n_size.min()-0.5, vmax=n_size.max()+0.5)    
    ax2.errorbar(mm_angle_mean['Time (minutes)'], mm_angle_mean['mean'], mm_angle_mean['std'], marker='None', ls='None', elinewidth=1, ecolor='grey')
    graph = ax2.scatter(mm_angle_mean['Time (minutes)'], mm_angle_mean['mean'], marker='.', s=10, c=n_size, cmap=cmap, zorder=9)#, norm=matplotlib.colors.LogNorm()
    ax2.plot(x_fit, mm_genlog_y_angle, zorder=10, c='k')
    ax2.set_xlim(mm_angle_mean['Time (minutes)'].iloc[first_multi_n]-1, mm_angle_mean['Time (minutes)'].iloc[last_multi_n]+1)
    # ax2.set_xlim(mm_angle_mean['Time (minutes)'].min()-1, mm_angle_mean['Time (minutes)'].max()+1)
    ax2.axvline(growth_phase_start_mm_angle)
    ax2.axvline(growth_phase_end_mm_angle)
    ax2.axhline(ee_genlog_popt_angle[3], c='g', ls='--')
    ax2.axhline(ee_genlog_popt_angle[2], c='g', ls='--')
    ax2.axhline(two_circ_interp_angle(ee_first_valid_area).mean(), c='g', ls='-', lw=0.5)
    ax2.axhline(two_circ_interp_angle(mm_first_valid_area).mean(), c='r', ls='--', lw=0.5)
    ax2.yaxis.set_ticks(np.arange(0, 220, 20))
    ax2.set_ylim(0,220)
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(graph, cax=cax)
    cbar_ticks = np.unique([int(x) for x in cbar.get_ticks()])
    cax.clear()
    cbar = plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax, ticks=cbar_ticks)
    ax2.draw
    fig2.savefig(Path.joinpath(output_dir, r"mm_angles_gen_logistic.tif"), dpi=dpi_graph)
    fig2.clf()


if np.any(mm_angle_mean_for_contnd_fit):
    last_multi_n = len((mm_angle_mean_for_contnd_fit['count'] > n_size_thresh_for_xlim).values) - 1 - np.argmax((mm_angle_mean_for_contnd_fit['count'] > n_size_thresh_for_xlim).values[::-1])
    first_multi_n = np.argmax((mm_angle_mean_for_contnd_fit['count'] > n_size_thresh_for_xlim).values)
    fig, ax = plt.subplots(figsize=(width_2col, height_2col))
    n_size = mm_angle_mean_for_contnd_fit['count']
    cmap = plt.cm.get_cmap('jet', n_size.max())
    norm = mpl.colors.Normalize(vmin=n_size.min()-0.5, vmax=n_size.max()+0.5)    
    ax.errorbar(mm_angle_mean_for_contnd_fit['Time (minutes)'], mm_angle_mean_for_contnd_fit['mean'], mm_angle_mean_for_contnd_fit['std'], marker='None', ls='None', elinewidth=1, ecolor='grey')
    graph = ax.scatter(mm_angle_mean_for_contnd_fit['Time (minutes)'], mm_angle_mean_for_contnd_fit['mean'], marker='.', s=10, c=n_size, cmap=cmap, zorder=9)#, norm=matplotlib.colors.LogNorm()
    ax.plot(x_fit, mm_genlog_y_angle_contndfit, zorder=10, c='k')
    ax.set_xlim(mm_angle_mean_for_contnd_fit['Time (minutes)'].iloc[first_multi_n]-1, mm_angle_mean_for_contnd_fit['Time (minutes)'].iloc[last_multi_n]+1)
    # ax.set_xlim(mm_angle_mean_for_contnd_fit['Time (minutes)'].min()-1, mm_angle_mean_for_contnd_fit['Time (minutes)'].max()+1)
    #ax.axvline(growth_phase_start_mm_angle)
    #ax.axvline(growth_phase_end_mm_angle)
    ax.axhline(ee_genlog_popt_angle[3], c='g', ls='--')
    ax.axhline(ee_genlog_popt_angle[2], c='g', ls='--')
    ax.axhline(two_circ_interp_angle(ee_first_valid_area).mean(), c='g', ls='-', lw=0.5)
    ax.axhline(two_circ_interp_angle(mm_first_valid_area).mean(), c='r', ls='--', lw=0.5)
    ax.yaxis.set_ticks(np.arange(0, 220, 20))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(graph, cax=cax)
    cbar_ticks = np.unique([int(x) for x in cbar.get_ticks()])
    cax.clear()
    cbar = plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax, ticks=cbar_ticks)
    ax.draw
    fig.savefig(Path.joinpath(output_dir, r"mm_angle_mean_for_contnd_fit.tif"), dpi=dpi_graph)
    fig.clf()


if np.any(mm_contnd_mean_for_angle_fit):
    last_multi_n = len((mm_contnd_mean_for_angle_fit['count'] > n_size_thresh_for_xlim).values) - 1 - np.argmax((mm_contnd_mean_for_angle_fit['count'] > n_size_thresh_for_xlim).values[::-1])
    first_multi_n = np.argmax((mm_contnd_mean_for_angle_fit['count'] > n_size_thresh_for_xlim).values)
    fig2, ax2 = plt.subplots(figsize=(width_2col, height_2col))
    n_size = mm_contnd_mean_for_angle_fit['count']
    cmap = plt.cm.get_cmap('jet', n_size.max())
    norm = mpl.colors.Normalize(vmin=n_size.min()-0.5, vmax=n_size.max()+0.5)    
    ax2.errorbar(mm_contnd_mean_for_angle_fit['Time (minutes)'], mm_contnd_mean_for_angle_fit['mean'], mm_contnd_mean_for_angle_fit['std'], marker='None', ls='None', elinewidth=1, ecolor='grey')
    graph = ax2.scatter(mm_contnd_mean_for_angle_fit['Time (minutes)'], mm_contnd_mean_for_angle_fit['mean'], marker='.', s=10, c=n_size, cmap=cmap, zorder=9)#, norm=matplotlib.colors.LogNorm()
    ax2.plot(x_fit, mm_genlog_y_contnd_anglefit, zorder=10, c='k')
    ax2.set_xlim(mm_contnd_mean_for_angle_fit['Time (minutes)'].iloc[first_multi_n]-1, mm_contnd_mean_for_angle_fit['Time (minutes)'].iloc[last_multi_n]+1)
    # ax2.set_xlim(mm_contnd_mean_for_angle_fit['Time (minutes)'].min()-1, mm_contnd_mean_for_angle_fit['Time (minutes)'].max()+1)
    #ax2.axvline(growth_phase_start_mm_contnd)
    #ax2.axvline(growth_phase_end_mm_contnd)
    ax2.axhline(ee_genlog_popt_contnd[3], c='g', ls='--')
    ax2.axhline(ee_genlog_popt_contnd[2], c='g', ls='--')
    ax2.axhline(two_circ_interp_contnd(ee_first_valid_area).mean(), c='g', ls='-', lw=0.5)
    ax2.axhline(two_circ_interp_contnd(mm_first_valid_area).mean(), c='r', ls='--', lw=0.5)
    ax2.set_ylim(0, 4.0)
    ax2.set_ylim(0,220)
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(graph, cax=cax)
    cbar_ticks = np.unique([int(x) for x in cbar.get_ticks()])
    cax.clear()
    cbar = plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax, ticks=cbar_ticks)
    ax2.draw
    fig2.savefig(Path.joinpath(output_dir, r"mm_contnd_mean_for_angle_fit.tif"), dpi=dpi_graph)
    fig2.clf()



if np.any(mm_angle_mean_cant_time_corr):
    last_multi_n = len((mm_angle_mean_cant_time_corr['count'] > n_size_thresh_for_xlim).values) - 1 - np.argmax((mm_angle_mean_cant_time_corr['count'] > n_size_thresh_for_xlim).values[::-1])
    first_multi_n = np.argmax((mm_angle_mean_cant_time_corr['count'] > n_size_thresh_for_xlim).values)
    fig, ax = plt.subplots(figsize=(width_2col, height_2col))
    n_size = mm_angle_mean_cant_time_corr['count']
    cmap = plt.cm.get_cmap('jet', n_size.max())
    norm = mpl.colors.Normalize(vmin=n_size.min()-0.5, vmax=n_size.max()+0.5)    
    ax.errorbar(mm_angle_mean_cant_time_corr['Time (minutes)'], mm_angle_mean_cant_time_corr['mean'], mm_angle_mean_cant_time_corr['std'], ls='None', elinewidth=1, ecolor='grey')
    graph = ax.scatter(mm_angle_mean_cant_time_corr['Time (minutes)'], mm_angle_mean_cant_time_corr['mean'], marker='.', s=10, c=n_size, cmap=cmap, zorder=9)#, norm=matplotlib.colors.LogNorm()
    ax.set_xlim(mm_angle_mean_cant_time_corr['Time (minutes)'].min()-1, mm_angle_mean_cant_time_corr['Time (minutes)'].max()+1)
    ax.axhline(ee_genlog_popt_angle[3], c='g', ls='--')
    ax.axhline(ee_genlog_popt_angle[2], c='g', ls='--')
    ax.axhline(two_circ_interp_angle(ee_first_valid_area).mean(), c='g', ls='-', lw=0.5)
    ax.axhline(two_circ_interp_angle(mm_first_valid_area).mean(), c='r', ls='--', lw=0.5)
    #ax.axhline(two_circ_interp_angle(nn_first_valid_area).mean(), c='b', ls='--', lw=0.5)
    ax.set_ylim(0, 220)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(graph, cax=cax)
    cbar_ticks = np.unique([int(x) for x in cbar.get_ticks()])
    cax.clear()
    cbar = plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax, ticks=cbar_ticks)
    ax.draw
    fig.savefig(Path.joinpath(output_dir, r"mm_angle_mean_cant_time_corr.tif"), dpi=dpi_graph)
    fig.clf()


if np.any(mm_fit_contact_mean_cant_time_corr):
    last_multi_n = len((mm_fit_contact_mean_cant_time_corr['count'] > n_size_thresh_for_xlim).values) - 1 - np.argmax((mm_fit_contact_mean_cant_time_corr['count'] > n_size_thresh_for_xlim).values[::-1])
    first_multi_n = np.argmax((mm_fit_contact_mean_cant_time_corr['count'] > n_size_thresh_for_xlim).values)
    fig, ax = plt.subplots(figsize=(width_2col, height_2col))
    n_size = mm_fit_contact_mean_cant_time_corr['count']
    cmap = plt.cm.get_cmap('jet', n_size.max())
    norm = mpl.colors.Normalize(vmin=n_size.min()-0.5, vmax=n_size.max()+0.5)    
    ax.errorbar(mm_fit_contact_mean_cant_time_corr['Time (minutes)'], mm_fit_contact_mean_cant_time_corr['mean'], mm_fit_contact_mean_cant_time_corr['std'], ls='None', elinewidth=1, ecolor='grey')
    graph = ax.scatter(mm_fit_contact_mean_cant_time_corr['Time (minutes)'], mm_fit_contact_mean_cant_time_corr['mean'], marker='.', s=10, c=n_size, cmap=cmap, zorder=9)#, norm=matplotlib.colors.LogNorm()
    ax.set_xlim(mm_fit_contact_mean_cant_time_corr['Time (minutes)'].min()-1, mm_fit_contact_mean_cant_time_corr['Time (minutes)'].max()+1)
    ax.axhline(ee_genlog_popt_contnd[3], c='g', ls='--')
    ax.axhline(ee_genlog_popt_contnd[2], c='g', ls='--')
    ax.axhline(two_circ_interp_contnd(ee_first_valid_area).mean(), c='g', ls='-', lw=0.5)
    ax.axhline(two_circ_interp_contnd(mm_first_valid_area).mean(), c='r', ls='--', lw=0.5)
    #ax.axhline(two_circ_interp_contnd(nn_first_valid_area).mean(), c='b', ls='--', lw=0.5)
    ax.set_ylim(0, 4.0)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(graph, cax=cax)
    cbar_ticks = np.unique([int(x) for x in cbar.get_ticks()])
    cax.clear()
    cbar = plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax, ticks=cbar_ticks)
    ax.draw
    fig.savefig(Path.joinpath(output_dir, r"mm_fit_contact_mean_cant_time_corr.tif"), dpi=dpi_graph)
    fig.clf()





###############################################################################

# Plot all endo-endo in one graph after time-correcting them:
fig6, ax6 = plt.subplots(figsize=(width_1col, height_1col))
for i,x in enumerate(nn_contact_diam_non_dim_df_list_dropna):
    if nn_list_dropna_fit_validity_df['growth_phase'].iloc[i]:
        ax6.scatter(x['Time (minutes)']-nn_indiv_logistic_contnd_df.iloc[i]['popt_x2'], x['Contact Surface Length (N.D.)'], marker='.', s=1)
#ax6.axhline(ee_popt_contnd[3], c='g', ls='--')
#ax6.axhline(ee_popt_contnd[2], c='g', ls='--')
#ax6.set_ylim(0, 4.0)
fig6.savefig(Path.joinpath(output_dir, r"nn_contactnds.tif"), dpi=dpi_graph)
fig6.clf()

# Plot all endo-endo in one graph after time-correcting them:
fig4, ax4 = plt.subplots(figsize=(width_1col, height_1col))
for i,x in enumerate(nn_list_dropna):
    if nn_list_dropna_fit_validity_df['growth_phase'].iloc[i]:
        ax4.scatter(x['Time (minutes)']-nn_indiv_logistic_cont_df.iloc[i]['popt_x2'], x['Contact Surface Length (px)']*um_per_pixel_nn, marker='.', s=1)
fig4.savefig(Path.joinpath(output_dir, r"nn_contacts.tif"), dpi=dpi_graph)
fig4.clf()

# Plot all endo-endo in one graph after time-correcting them:
fig5, ax5 = plt.subplots(figsize=(width_1col, height_1col))
for i,x in enumerate(nn_list_dropna):
    if nn_list_dropna_angle_fit_validity_df['growth_phase'].iloc[i]:
        ax5.scatter(x['Time (minutes)']-nn_indiv_logistic_angle_df.iloc[i]['popt_x2'], x['Average Angle (deg)'], marker='.', s=1)
fig5.savefig(Path.joinpath(output_dir, r"nn_angles.tif"), dpi=dpi_graph)
fig5.clf()

#Also plot the data that could not be time corrected (if any):
fig6, ax6 = plt.subplots(figsize=(width_1col, height_1col))
for i,x in enumerate(nn_contact_diam_non_dim_df_list_dropna):
    if nn_list_dropna_fit_validity_df['growth_phase'].iloc[i] == False:
        ax6.scatter(x['Time (minutes)']-nn_indiv_logistic_contnd_df.iloc[i]['popt_x2'], x['Contact Surface Length (N.D.)'], marker='.', s=5)
        ax6.plot(x['Time (minutes)']-nn_indiv_logistic_contnd_df.iloc[i]['popt_x2'], x['Contact Surface Length (N.D.)'], lw=1)
#ax6.axhline(ee_popt_contnd[3], c='g', ls='--')
#ax6.axhline(ee_popt_contnd[2], c='g', ls='--')
ax6.set_ylim(0, 4.0)
fig6.savefig(Path.joinpath(output_dir, r"nn_contactnds_bad_fit.tif"), dpi=dpi_graph)
fig6.clf()

fig4, ax4 = plt.subplots(figsize=(width_1col, height_1col))
for i,x in enumerate(nn_list_dropna):
    if nn_list_dropna_fit_validity_df['growth_phase'].iloc[i] == False:
        ax4.scatter(x['Time (minutes)']-nn_indiv_logistic_cont_df.iloc[i]['popt_x2'], x['Contact Surface Length (px)']*um_per_pixel_nn, marker='.', s=5)
        ax4.plot(x['Time (minutes)']-nn_indiv_logistic_cont_df.iloc[i]['popt_x2'], x['Contact Surface Length (px)']*um_per_pixel_nn, lw=1)
fig4.savefig(Path.joinpath(output_dir, r"nn_contacts_bad_fit.tif"), dpi=dpi_graph)
fig4.clf()

fig5, ax5 = plt.subplots(figsize=(width_1col, height_1col))
for i,x in enumerate(nn_list_dropna):
    if nn_list_dropna_angle_fit_validity_df['growth_phase'].iloc[i] == False:
        ax5.scatter(x['Time (minutes)']-nn_indiv_logistic_angle_df.iloc[i]['popt_x2'], x['Average Angle (deg)'], marker='.', s=5)
        ax5.plot(x['Time (minutes)']-nn_indiv_logistic_angle_df.iloc[i]['popt_x2'], x['Average Angle (deg)'], lw=1)
ax5.set_ylim(0, 220)
fig5.savefig(Path.joinpath(output_dir, r"nn_angles_bad_fit.tif"), dpi=dpi_graph)
fig5.clf()



# Plot the endo-endo with the data points color coded according to N size:
#n_size_thresh_for_xlim = 1
if np.any(nn_fit_contact_mean):
    mlx = MultipleLocator(10)
    mly = MultipleLocator(0.5)
    last_multi_n = len((nn_fit_contact_mean['count'] > n_size_thresh_for_xlim).values) - 1 - np.argmax((nn_fit_contact_mean['count'] > n_size_thresh_for_xlim).values[::-1])
    first_multi_n = np.argmax((nn_fit_contact_mean['count'] > n_size_thresh_for_xlim).values)
    fig, ax = plt.subplots(figsize=(width_1col, height_1col))
    n_size = nn_fit_contact_mean['count']
    cmap = plt.cm.get_cmap('jet', n_size.max())
    norm = mpl.colors.Normalize(vmin=n_size.min()-0.5, vmax=n_size.max()+0.5)    
    ax.errorbar(nn_fit_contact_mean['Time (minutes)'], nn_fit_contact_mean['mean'], nn_fit_contact_mean['std'], marker='None', ls='None', elinewidth=1, ecolor='grey')
    graph = ax.scatter(nn_fit_contact_mean['Time (minutes)'], nn_fit_contact_mean['mean'], marker='.', s=10, c=n_size, cmap=cmap, zorder=9)#, norm=matplotlib.colors.LogNorm()
    ax.plot(x_fit, nn_genlog_y_contnd, zorder=10, c='k')
    ax.set_xlim(nn_fit_contact_mean['Time (minutes)'].iloc[first_multi_n]-1, nn_fit_contact_mean['Time (minutes)'].iloc[last_multi_n]+1)
    # ax.set_xlim(nn_fit_contact_mean['Time (minutes)'].min()-1, nn_fit_contact_mean['Time (minutes)'].max()+1)
    ax.axvline(growth_phase_start_nn_contnd)
    #ax.axvline(growth_phase_start_nn_contnd_manual, c='k', ls='--')
    ax.axvline(growth_phase_end_nn_contnd)
    ax.axhline(ee_genlog_popt_contnd[3], c='g', ls='--')
    ax.axhline(ee_genlog_popt_contnd[2], c='g', ls='--')
    ax.axhline(two_circ_interp_contnd(ee_first_valid_area).mean(), c='g', ls='-', lw=0.5)
    ax.axhline(two_circ_interp_contnd(mm_first_valid_area).mean(), c='r', ls='--', lw=0.5)
    ax.axhline(two_circ_interp_contnd(nn_first_valid_area).mean(), c='b', ls='--', lw=0.5)
    ax.set_ylim(0, 4.0)
    ax.xaxis.set_minor_locator(mlx)
    ax.yaxis.set_minor_locator(mly)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(graph, cax=cax)
    cbar_ticks = np.unique([int(x) for x in cbar.get_ticks()])
    cax.clear()
    cbar = plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax, ticks=cbar_ticks)
    plt.draw
    plt.savefig(Path.joinpath(output_dir, r"nn_contactnds_gen_logistic.tif"), dpi=dpi_graph)
    plt.clf()
    
    
if np.any(nn_angle_mean):
    last_multi_n = len((nn_angle_mean['count'] > n_size_thresh_for_xlim).values) - 1 - np.argmax((nn_angle_mean['count'] > n_size_thresh_for_xlim).values[::-1])
    first_multi_n = np.argmax((nn_angle_mean['count'] > n_size_thresh_for_xlim).values)
    fig2, ax2 = plt.subplots(figsize=(width_2col, height_2col))
    n_size = nn_angle_mean['count']
    cmap = plt.cm.get_cmap('jet', n_size.max())
    norm = mpl.colors.Normalize(vmin=n_size.min()-0.5, vmax=n_size.max()+0.5)
    ax2.errorbar(nn_angle_mean['Time (minutes)'], nn_angle_mean['mean'], nn_angle_mean['std'], marker='None', ls='None', elinewidth=1, ecolor='grey')
    graph = ax2.scatter(nn_angle_mean['Time (minutes)'], nn_angle_mean['mean'], marker='.', s=10, c=n_size, cmap=cmap, zorder=9)#, norm=matplotlib.colors.LogNorm()
    ax2.plot(x_fit, nn_genlog_y_angle, zorder=10, c='k')
    ax2.set_xlim(nn_angle_mean['Time (minutes)'].iloc[first_multi_n]-1, nn_angle_mean['Time (minutes)'].iloc[last_multi_n]+1)
    # ax2.set_xlim(nn_angle_mean['Time (minutes)'].min()-1, nn_angle_mean['Time (minutes)'].max()+1)
    #ax2.axvline(growth_phase_start_nn_genlog_angle)
    #ax2.axvline(growth_phase_end_nn_genlog_angle)
    ax2.axvline(growth_phase_start_nn_angle)
    ax2.axvline(growth_phase_start_nn_angle_manual, c='k', ls='--')
    ax2.axvline(growth_phase_end_nn_angle)
    ax2.axhline(ee_genlog_popt_angle[3], c='g', ls='--')
    ax2.axhline(ee_genlog_popt_angle[2], c='g', ls='--')
    ax2.axhline(two_circ_interp_angle(ee_first_valid_area).mean(), c='g', ls='-', lw=0.5)
    ax2.axhline(two_circ_interp_angle(mm_first_valid_area).mean(), c='r', ls='--', lw=0.5)
    ax2.axhline(two_circ_interp_angle(nn_first_valid_area).mean(), c='b', ls='--', lw=0.5)
    ax2.yaxis.set_ticks(np.arange(0, 220, 20))
    ax2.set_ylim(0,220)
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(graph, cax=cax)
    cbar_ticks = np.unique([int(x) for x in cbar.get_ticks()])
    cax.clear()
    cbar = plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax, ticks=cbar_ticks)
    ax2.draw
    fig2.savefig(Path.joinpath(output_dir, r"nn_angles_gen_logistic.tif"), dpi=dpi_graph)
    fig2.clf()
    
    
if np.any(nn_angle_mean_for_contnd_fit):
    last_multi_n = len((nn_angle_mean_for_contnd_fit['count'] > n_size_thresh_for_xlim).values) - 1 - np.argmax((nn_angle_mean_for_contnd_fit['count'] > n_size_thresh_for_xlim).values[::-1])
    first_multi_n = np.argmax((nn_angle_mean_for_contnd_fit['count'] > n_size_thresh_for_xlim).values)
    fig, ax = plt.subplots(figsize=(width_2col, height_2col))
    n_size = nn_angle_mean_for_contnd_fit['count']
    cmap = plt.cm.get_cmap('jet', n_size.max())
    norm = mpl.colors.Normalize(vmin=n_size.min()-0.5, vmax=n_size.max()+0.5)    
    ax.errorbar(nn_angle_mean_for_contnd_fit['Time (minutes)'], nn_angle_mean_for_contnd_fit['mean'], nn_angle_mean_for_contnd_fit['std'], marker='None', ls='None', elinewidth=1, ecolor='grey')
    graph = ax.scatter(nn_angle_mean_for_contnd_fit['Time (minutes)'], nn_angle_mean_for_contnd_fit['mean'], marker='.', s=10, c=n_size, cmap=cmap, zorder=9)#, norm=matplotlib.colors.LogNorm()
    ax.plot(x_fit, nn_genlog_y_angle_contndfit, zorder=10, c='k')
    ax.set_xlim(nn_angle_mean_for_contnd_fit['Time (minutes)'].iloc[first_multi_n]-1, nn_angle_mean_for_contnd_fit['Time (minutes)'].iloc[last_multi_n]+1)
    # ax.set_xlim(nn_angle_mean_for_contnd_fit['Time (minutes)'].min()-1, nn_angle_mean_for_contnd_fit['Time (minutes)'].max()+1)
    #ax.axvline(growth_phase_start_nn_genlog_angle)
    #ax.axvline(growth_phase_end_nn_genlog_angle)
    #ax.axvline(growth_phase_start_nn_angle)
    #ax.axvline(growth_phase_start_nn_angle_manual, c='k', ls='--')
    #ax.axvline(growth_phase_end_nn_angle)
    ax.axhline(ee_genlog_popt_angle[3], c='g', ls='--')
    ax.axhline(ee_genlog_popt_angle[2], c='g', ls='--')
    ax.axhline(two_circ_interp_angle(ee_first_valid_area).mean(), c='g', ls='-', lw=0.5)
    ax.axhline(two_circ_interp_angle(mm_first_valid_area).mean(), c='r', ls='--', lw=0.5)
    ax.axhline(two_circ_interp_angle(nn_first_valid_area).mean(), c='b', ls='--', lw=0.5)
    ax.yaxis.set_ticks(np.arange(0, 220, 20))
    ax.set_ylim(0,220)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(graph, cax=cax)
    cbar_ticks = np.unique([int(x) for x in cbar.get_ticks()])
    cax.clear()
    cbar = plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax, ticks=cbar_ticks)
    plt.draw
    plt.savefig(Path.joinpath(output_dir, r"nn_angle_mean_for_contnd_fit.tif"), dpi=dpi_graph)
    plt.clf()
    
    
if np.any(nn_contnd_mean_for_angle_fit):
    last_multi_n = len((nn_contnd_mean_for_angle_fit['count'] > n_size_thresh_for_xlim).values) - 1 - np.argmax((nn_contnd_mean_for_angle_fit['count'] > n_size_thresh_for_xlim).values[::-1])
    first_multi_n = np.argmax((nn_contnd_mean_for_angle_fit['count'] > n_size_thresh_for_xlim).values)
    fig2, ax2 = plt.subplots(figsize=(width_2col, height_2col))
    n_size = nn_contnd_mean_for_angle_fit['count']
    cmap = plt.cm.get_cmap('jet', n_size.max())
    norm = mpl.colors.Normalize(vmin=n_size.min()-0.5, vmax=n_size.max()+0.5)    
    ax2.errorbar(nn_contnd_mean_for_angle_fit['Time (minutes)'], nn_contnd_mean_for_angle_fit['mean'], nn_contnd_mean_for_angle_fit['std'], marker='None', ls='None', elinewidth=1, ecolor='grey')
    graph = ax2.scatter(nn_contnd_mean_for_angle_fit['Time (minutes)'], nn_contnd_mean_for_angle_fit['mean'], marker='.', s=10, c=n_size, cmap=cmap, zorder=9)#, norm=matplotlib.colors.LogNorm()
    ax2.plot(x_fit, nn_genlog_y_contnd_anglefit, zorder=10, c='k')
    ax2.set_xlim(nn_contnd_mean_for_angle_fit['Time (minutes)'].iloc[first_multi_n]-1, nn_contnd_mean_for_angle_fit['Time (minutes)'].iloc[last_multi_n]+1)
    # ax2.set_xlim(nn_contnd_mean_for_angle_fit['Time (minutes)'].min()-1, nn_contnd_mean_for_angle_fit['Time (minutes)'].max()+1)
    #ax2.axvline(growth_phase_start_nn_contnd)
    #ax2.axvline(growth_phase_end_nn_contnd)
    ax2.axhline(ee_genlog_popt_contnd[3], c='g', ls='--')
    ax2.axhline(ee_genlog_popt_contnd[2], c='g', ls='--')
    ax2.axhline(two_circ_interp_contnd(ee_first_valid_area).mean(), c='g', ls='-', lw=0.5)
    ax2.axhline(two_circ_interp_contnd(mm_first_valid_area).mean(), c='r', ls='--', lw=0.5)
    ax2.axhline(two_circ_interp_contnd(nn_first_valid_area).mean(), c='b', ls='--', lw=0.5)
    ax2.set_ylim(0, 4.0)
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(graph, cax=cax)
    cbar_ticks = np.unique([int(x) for x in cbar.get_ticks()])
    cax.clear()
    cbar = plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax, ticks=cbar_ticks)
    ax2.draw
    fig2.savefig(Path.joinpath(output_dir, r"nn_contnd_mean_for_angle_fit.tif"), dpi=dpi_graph)
    fig2.clf()
    
    
    
if np.any(nn_angle_mean_cant_time_corr):
    last_multi_n = len((nn_angle_mean_cant_time_corr['count'] > n_size_thresh_for_xlim).values) - 1 - np.argmax((nn_angle_mean_cant_time_corr['count'] > n_size_thresh_for_xlim).values[::-1])
    first_multi_n = np.argmax((nn_angle_mean_cant_time_corr['count'] > n_size_thresh_for_xlim).values)
    fig, ax = plt.subplots(figsize=(width_2col, height_2col))
    n_size = nn_angle_mean_cant_time_corr['count']
    cmap = plt.cm.get_cmap('jet', n_size.max())
    norm = mpl.colors.Normalize(vmin=n_size.min()-0.5, vmax=n_size.max()+0.5)    
    ax.errorbar(nn_angle_mean_cant_time_corr['Time (minutes)'], nn_angle_mean_cant_time_corr['mean'], nn_angle_mean_cant_time_corr['std'], ls='None', elinewidth=1, ecolor='grey')
    graph = ax.scatter(nn_angle_mean_cant_time_corr['Time (minutes)'], nn_angle_mean_cant_time_corr['mean'], marker='.', s=10, c=n_size, cmap=cmap, zorder=9)#, norm=matplotlib.colors.LogNorm()
    ax.set_xlim(nn_angle_mean_cant_time_corr['Time (minutes)'].min()-1, nn_angle_mean_cant_time_corr['Time (minutes)'].max()+1)
    ax.axhline(ee_genlog_popt_angle[3], c='g', ls='--')
    ax.axhline(ee_genlog_popt_angle[2], c='g', ls='--')
    ax.axhline(two_circ_interp_angle(ee_first_valid_area).mean(), c='g', ls='-', lw=0.5)
    ax.axhline(two_circ_interp_angle(mm_first_valid_area).mean(), c='r', ls='--', lw=0.5)
    ax.axhline(two_circ_interp_angle(nn_first_valid_area).mean(), c='b', ls='--', lw=0.5)
    ax.set_ylim(0, 220)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(graph, cax=cax)
    cbar_ticks = np.unique([int(x) for x in cbar.get_ticks()])
    cax.clear()
    cbar = plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax, ticks=cbar_ticks)
    ax.draw
    fig.savefig(Path.joinpath(output_dir, r"nn_angle_mean_cant_time_corr.tif"), dpi=dpi_graph)
    fig.clf()
    
    
if np.any(nn_fit_contact_mean_cant_time_corr):
    last_multi_n = len((nn_fit_contact_mean_cant_time_corr['count'] > n_size_thresh_for_xlim).values) - 1 - np.argmax((nn_fit_contact_mean_cant_time_corr['count'] > n_size_thresh_for_xlim).values[::-1])
    first_multi_n = np.argmax((nn_fit_contact_mean_cant_time_corr['count'] > n_size_thresh_for_xlim).values)
    fig, ax = plt.subplots(figsize=(width_2col, height_2col))
    n_size = nn_fit_contact_mean_cant_time_corr['count']
    cmap = plt.cm.get_cmap('jet', n_size.max())
    norm = mpl.colors.Normalize(vmin=n_size.min()-0.5, vmax=n_size.max()+0.5)    
    ax.errorbar(nn_fit_contact_mean_cant_time_corr['Time (minutes)'], nn_fit_contact_mean_cant_time_corr['mean'], nn_fit_contact_mean_cant_time_corr['std'], ls='None', elinewidth=1, ecolor='grey')
    graph = ax.scatter(nn_fit_contact_mean_cant_time_corr['Time (minutes)'], nn_fit_contact_mean_cant_time_corr['mean'], marker='.', s=10, c=n_size, cmap=cmap, zorder=9)#, norm=matplotlib.colors.LogNorm()
    ax.set_xlim(nn_fit_contact_mean_cant_time_corr['Time (minutes)'].min()-1, nn_fit_contact_mean_cant_time_corr['Time (minutes)'].max()+1)
    ax.axhline(ee_genlog_popt_contnd[3], c='g', ls='--')
    ax.axhline(ee_genlog_popt_contnd[2], c='g', ls='--')
    ax.axhline(two_circ_interp_contnd(ee_first_valid_area).mean(), c='g', ls='-', lw=0.5)
    ax.axhline(two_circ_interp_contnd(mm_first_valid_area).mean(), c='r', ls='--', lw=0.5)
    ax.axhline(two_circ_interp_contnd(nn_first_valid_area).mean(), c='b', ls='--', lw=0.5)
    ax.set_ylim(0, 4.0)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(graph, cax=cax)
    cbar_ticks = np.unique([int(x) for x in cbar.get_ticks()])
    cax.clear()
    cbar = plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax, ticks=cbar_ticks)
    ax.draw
    fig.savefig(Path.joinpath(output_dir, r"nn_fit_contact_mean_cant_time_corr.tif"), dpi=dpi_graph)
    fig.clf()





###############################################################################

# Plot all ecto-meso data in one graph after time-correcting them:
fig6, ax6 = plt.subplots(figsize=(width_1col, height_1col))
for i,x in enumerate(em_contact_diam_non_dim_df_list_dropna):
    if em_list_dropna_fit_validity_df['growth_phase'].iloc[i]:
        ax6.scatter(x['Time (minutes)']-em_indiv_logistic_contnd_df.iloc[i]['popt_x2'], x['Contact Surface Length (N.D.)'], marker='.', s=1)
#ax6.axhline(ee_popt_contnd[3], c='g', ls='--')
#ax6.axhline(ee_popt_contnd[2], c='g', ls='--')
#ax6.set_ylim(0, 4.0)
fig6.savefig(Path.joinpath(output_dir, r"em_contactnds.tif"), dpi=dpi_graph)
fig6.clf()

# Plot all ecto-meso data in one graph after time-correcting them:
fig4, ax4 = plt.subplots(figsize=(width_1col, height_1col))
for i,x in enumerate(em_list_dropna):
    if em_list_dropna_fit_validity_df['growth_phase'].iloc[i]:
        ax4.scatter(x['Time (minutes)']-em_indiv_logistic_cont_df.iloc[i]['popt_x2'], x['Contact Surface Length (px)']*um_per_pixel, marker='.', s=1)
fig4.savefig(Path.joinpath(output_dir, r"em_contacts.tif"), dpi=dpi_graph)
fig4.clf()

# Plot all ecto-meso data in one graph after time-correcting them:
fig5, ax5 = plt.subplots(figsize=(width_1col, height_1col))
for i,x in enumerate(em_list_dropna):
    if em_list_dropna_angle_fit_validity_df['growth_phase'].iloc[i]:
        ax5.scatter(x['Time (minutes)']-em_indiv_logistic_angle_df.iloc[i]['popt_x2'], x['Average Angle (deg)'], marker='.', s=1)
fig5.savefig(Path.joinpath(output_dir, r"em_angles.tif"), dpi=dpi_graph)
fig5.clf()

#Also plot the data that could not be time corrected (if any):
fig6, ax6 = plt.subplots(figsize=(width_1col, height_1col))
for i,x in enumerate(em_contact_diam_non_dim_df_list_dropna):
    if em_list_dropna_fit_validity_df['growth_phase'].iloc[i] == False:
        ax6.scatter(x['Time (minutes)']-em_indiv_logistic_contnd_df.iloc[i]['popt_x2'], x['Contact Surface Length (N.D.)'], marker='.', s=5)
        ax6.plot(x['Time (minutes)']-em_indiv_logistic_contnd_df.iloc[i]['popt_x2'], x['Contact Surface Length (N.D.)'], lw=1)
#ax6.axhline(ee_popt_contnd[3], c='g', ls='--')
#ax6.axhline(ee_popt_contnd[2], c='g', ls='--')
ax6.set_ylim(0, 4.0)
fig6.savefig(Path.joinpath(output_dir, r"em_contactnds_bad_fit.tif"), dpi=dpi_graph)
fig6.clf()

fig4, ax4 = plt.subplots(figsize=(width_1col, height_1col))
for i,x in enumerate(em_list_dropna):
    if em_list_dropna_fit_validity_df['growth_phase'].iloc[i] == False:
        ax4.scatter(x['Time (minutes)']-em_indiv_logistic_cont_df.iloc[i]['popt_x2'], x['Contact Surface Length (px)']*um_per_pixel, marker='.', s=5)
        ax4.plot(x['Time (minutes)']-em_indiv_logistic_cont_df.iloc[i]['popt_x2'], x['Contact Surface Length (px)']*um_per_pixel, lw=1)
fig4.savefig(Path.joinpath(output_dir, r"em_contacts_bad_fit.tif"), dpi=dpi_graph)
fig4.clf()

fig5, ax5 = plt.subplots(figsize=(width_1col, height_1col))
for i,x in enumerate(em_list_dropna):
    if em_list_dropna_angle_fit_validity_df['growth_phase'].iloc[i] == False:
        ax5.scatter(x['Time (minutes)']-em_indiv_logistic_angle_df.iloc[i]['popt_x2'], x['Average Angle (deg)'], marker='.', s=5)
        ax5.plot(x['Time (minutes)']-em_indiv_logistic_angle_df.iloc[i]['popt_x2'], x['Average Angle (deg)'], lw=1)
ax5.set_ylim(0, 220)
fig5.savefig(Path.joinpath(output_dir, r"em_angles_bad_fit.tif"), dpi=dpi_graph)
fig5.clf()



# Plot em data:
if np.any(em_fit_contact_mean):
    mlx = MultipleLocator(10)
    mly = MultipleLocator(0.5)
    last_multi_n = len((em_fit_contact_mean['count'] > n_size_thresh_for_xlim).values) - 1 - np.argmax((em_fit_contact_mean['count'] > n_size_thresh_for_xlim).values[::-1])
    first_multi_n = np.argmax((em_fit_contact_mean['count'] > n_size_thresh_for_xlim).values)
    fig8, ax8 = plt.subplots(figsize=(width_1col, height_1col))
    n_size = em_fit_contact_mean['count']
    cmap = plt.cm.get_cmap('jet', n_size.max())
    norm = mpl.colors.Normalize(vmin=n_size.min()-0.5, vmax=n_size.max()+0.5)    
    ax8.errorbar(em_fit_contact_mean['Time (minutes)'], em_fit_contact_mean['mean'], em_fit_contact_mean['std'], marker='None', ls='None', elinewidth=1, ecolor='grey')
    graph = ax8.scatter(em_fit_contact_mean['Time (minutes)'], em_fit_contact_mean['mean'], marker='.', s=10, c=n_size, cmap=cmap, zorder=9)
    #ax8.axhline(ee_popt_contnd[3], c='g', ls='--')
    #ax8.axhline(ee_popt_contnd[2], c='g', ls='--')
    ax8.plot(x_fit, em_genlog_y_contnd, c='k')
    ax8.axhline(ee_genlog_popt_contnd[3], c='g', ls='--')
    ax8.axhline(ee_genlog_popt_contnd[2], c='g', ls='--')
    #[plt.axhline(x, c='g', ls='-', lw=0.5) for x in kde_radius_init_2_circ_nd_peaks]
    #[plt.axhline(x, c='r', ls='--', lw=0.5) for x in em_kde_radius_init_2_circ_nd_peaks]
    ax8.axhline(two_circ_interp_contnd(ee_first_valid_area).mean(), c='g', ls='-', lw=0.5)
    ax8.axhline(two_circ_interp_contnd(em_first_valid_area).mean(), c='orange', ls='--', lw=0.5)
    ax8.set_ylim(0, 4.0)
    ax8.set_xlim(em_fit_contact_mean['Time (minutes)'].iloc[first_multi_n]-1, em_fit_contact_mean['Time (minutes)'].iloc[last_multi_n]+1)
    # ax8.set_xlim(em_fit_contact_mean['Time (minutes)'].min()-1, em_fit_contact_mean['Time (minutes)'].max()+1)
    ax8.xaxis.set_minor_locator(mlx)
    ax8.yaxis.set_minor_locator(mly)
    divider = make_axes_locatable(ax8)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(graph, cax=cax)
    cbar_ticks = np.unique([int(x) for x in cbar.get_ticks()])
    cax.clear()
    cbar = plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax, ticks=cbar_ticks)
    ax8.draw
    fig8.savefig(Path.joinpath(output_dir, r"em_contnd_mean.tif"), dpi=dpi_graph)
    fig8.clf()
    
    
if np.any(em_contact_mean):
    last_multi_n = len((em_contact_mean['count'] > n_size_thresh_for_xlim).values) - 1 - np.argmax((em_contact_mean['count'] > n_size_thresh_for_xlim).values[::-1])
    first_multi_n = np.argmax((em_contact_mean['count'] > n_size_thresh_for_xlim).values)
    fig6, ax6 = plt.subplots(figsize=(width_2col, height_2col))
    n_size = em_contact_mean['count']
    cmap = plt.cm.get_cmap('jet', n_size.max())
    norm = mpl.colors.Normalize(vmin=n_size.min()-0.5, vmax=n_size.max()+0.5)    
    ax6.errorbar(em_contact_mean['Time (minutes)'], em_contact_mean['mean'], em_contact_mean['std'], marker='None', ls='None', elinewidth=1, ecolor='grey')
    graph = ax6.scatter(em_contact_mean['Time (minutes)'], em_contact_mean['mean'], marker='.', s=10, c=n_size, cmap=cmap, zorder=9)
    ax6.plot(x_fit, em_genlog_y_cont, c='k')
    ax6.axhline(ee_genlog_popt_cont[3], c='g', ls='--')
    ax6.axhline(ee_genlog_popt_cont[2], c='g', ls='--')
    ax6.axhline(two_circ_interp_cont(ee_first_valid_area).mean(), c='g', ls='-', lw=0.5)
    ax6.axhline(two_circ_interp_cont(em_first_valid_area).mean(), c='orange', ls='--', lw=0.5)
    ax6.set_xlim(em_contact_mean['Time (minutes)'].iloc[first_multi_n]-1, em_contact_mean['Time (minutes)'].iloc[last_multi_n]+1)
    # ax6.set_xlim(em_contact_mean['Time (minutes)'].min()-1, em_contact_mean['Time (minutes)'].max()+1)
    divider = make_axes_locatable(ax6)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(graph, cax=cax)
    cbar_ticks = np.unique([int(x) for x in cbar.get_ticks()])
    cax.clear()
    cbar = plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax, ticks=cbar_ticks)
    ax6.draw
    fig6.savefig(Path.joinpath(output_dir, r"em_contact_mean.tif"), dpi=dpi_graph)
    fig6.clf()
    
    
if np.any(em_angle_mean):
    last_multi_n = len((em_angle_mean['count'] > n_size_thresh_for_xlim).values) - 1 - np.argmax((em_angle_mean['count'] > n_size_thresh_for_xlim).values[::-1])
    first_multi_n = np.argmax((em_angle_mean['count'] > n_size_thresh_for_xlim).values)
    fig7, ax7 = plt.subplots(figsize=(width_2col, height_2col))
    n_size = em_angle_mean['count']
    cmap = plt.cm.get_cmap('jet', n_size.max())
    norm = mpl.colors.Normalize(vmin=n_size.min()-0.5, vmax=n_size.max()+0.5)    
    ax7.errorbar(em_angle_mean['Time (minutes)'], em_angle_mean['mean'], em_angle_mean['std'], marker='None', ls='None', elinewidth=1, ecolor='grey')
    graph = ax7.scatter(em_angle_mean['Time (minutes)'], em_angle_mean['mean'], marker='.', s=10, c=n_size, cmap=cmap, zorder=9)
    ax7.plot(x_fit, em_genlog_y_angle, c='k')
    ax7.axhline(ee_genlog_popt_angle[3], c='g', ls='--')
    ax7.axhline(ee_genlog_popt_angle[2], c='g', ls='--')
    ax7.axhline(two_circ_interp_angle(ee_first_valid_area).mean(), c='g', ls='-', lw=0.5)
    ax7.axhline(two_circ_interp_angle(em_first_valid_area).mean(), c='orange', ls='--', lw=0.5)
    ax7.set_ylim(0, 220)
    ax7.set_xlim(em_angle_mean['Time (minutes)'].iloc[first_multi_n]-1, em_angle_mean['Time (minutes)'].iloc[last_multi_n]+1)
    # ax7.set_xlim(em_angle_mean['Time (minutes)'].min()-1, em_angle_mean['Time (minutes)'].max()+1)
    divider = make_axes_locatable(ax7)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(graph, cax=cax)
    cbar_ticks = np.unique([int(x) for x in cbar.get_ticks()])
    cax.clear()
    cbar = plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax, ticks=cbar_ticks)
    ax7.draw
    fig7.savefig(Path.joinpath(output_dir, r"em_angle_mean.tif"), dpi=dpi_graph)
    fig7.clf()
    
    
# if np.any(em_angle_mean_for_contnd_fit):
#     last_multi_n = len((em_angle_mean_for_contnd_fit['count'] > n_size_thresh_for_xlim).values) - 1 - np.argmax((em_angle_mean_for_contnd_fit['count'] > n_size_thresh_for_xlim).values[::-1])
#     first_multi_n = np.argmax((em_angle_mean_for_contnd_fit['count'] > n_size_thresh_for_xlim).values)
#     fig8, ax8 = plt.subplots(figsize=(width_2col, height_2col))
#     n_size = em_angle_mean_for_contnd_fit['count']
#     cmap = plt.cm.get_cmap('jet', n_size.max())
#     norm = mpl.colors.Normalize(vmin=n_size.min()-0.5, vmax=n_size.max()+0.5)    
#     ax8.errorbar(em_angle_mean_for_contnd_fit['Time (minutes)'], em_angle_mean_for_contnd_fit['mean'], em_angle_mean_for_contnd_fit['std'], marker='None', ls='None', elinewidth=1, ecolor='grey')
#     graph = ax8.scatter(em_angle_mean_for_contnd_fit['Time (minutes)'], em_angle_mean_for_contnd_fit['mean'], marker='.', s=10, c=n_size, cmap=cmap, zorder=9)
#     ax8.plot(x_fit, em_genlog_y_angle_contndfit, c='k')
#     ax8.set_xlim(em_angle_mean_for_contnd_fit['Time (minutes)'].min()-1, em_angle_mean_for_contnd_fit['Time (minutes)'].max()+1)
#     ax8.axhline(ee_genlog_popt_angle[3], c='g', ls='--')
#     ax8.axhline(ee_genlog_popt_angle[2], c='g', ls='--')
#     ax8.axhline(two_circ_interp_angle(ee_first_valid_area).mean(), c='g', ls='-', lw=0.5)
#     ax8.axhline(two_circ_interp_angle(em_first_valid_area).mean(), c='orange', ls='--', lw=0.5)
#     ax8.set_ylim(0, 220)
#     divider = make_axes_locatable(ax8)
#     cax = divider.append_axes('right', size='5%', pad=0.05)
#     cbar = plt.colorbar(graph, cax=cax)
#     cbar_ticks = np.unique([int(x) for x in cbar.get_ticks()])
#     cax.clear()
#     cbar = plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax, ticks=cbar_ticks)
#     ax8.draw
#     fig8.savefig(Path.joinpath(output_dir, r"em_angle_mean_for_contnd_fit.tif"), dpi=dpi_graph)
#     fig8.clf()
    
    
# if np.any(em_contnd_mean_for_angle_fit):
#     last_multi_n = len((em_contnd_mean_for_angle_fit['count'] > n_size_thresh_for_xlim).values) - 1 - np.argmax((em_contnd_mean_for_angle_fit['count'] > n_size_thresh_for_xlim).values[::-1])
#     first_multi_n = np.argmax((em_contnd_mean_for_angle_fit['count'] > n_size_thresh_for_xlim).values)
#     fig7, ax7 = plt.subplots(figsize=(width_2col, height_2col))
#     n_size = em_contnd_mean_for_angle_fit['count']
#     cmap = plt.cm.get_cmap('jet', n_size.max())
#     norm = mpl.colors.Normalize(vmin=n_size.min()-0.5, vmax=n_size.max()+0.5)    
#     ax7.errorbar(em_contnd_mean_for_angle_fit['Time (minutes)'], em_contnd_mean_for_angle_fit['mean'], em_contnd_mean_for_angle_fit['std'], marker='None', ls='None', elinewidth=1, ecolor='grey')
#     graph = ax7.scatter(em_contnd_mean_for_angle_fit['Time (minutes)'], em_contnd_mean_for_angle_fit['mean'], marker='.', s=10, c=n_size, cmap=cmap, zorder=9)
#     ax7.plot(x_fit, em_genlog_y_contnd_anglefit, c='k')
#     ax7.set_xlim(em_contnd_mean_for_angle_fit['Time (minutes)'].min()-1, em_contnd_mean_for_angle_fit['Time (minutes)'].max()+1)
#     ax7.axhline(ee_genlog_popt_contnd[3], c='g', ls='--')
#     ax7.axhline(ee_genlog_popt_contnd[2], c='g', ls='--')
#     #[plt.axhline(x, c='g', ls='-', lw=0.5) for x in kde_radius_init_2_circ_nd_peaks]
#     #[plt.axhline(x, c='r', ls='--', lw=0.5) for x in em_kde_radius_init_2_circ_nd_peaks]
#     ax7.axhline(two_circ_interp_contnd(ee_first_valid_area).mean(), c='g', ls='-', lw=0.5)
#     ax7.axhline(two_circ_interp_contnd(em_first_valid_area).mean(), c='orange', ls='--', lw=0.5)
#     ax7.set_ylim(0, 4.0)
#     divider = make_axes_locatable(ax7)
#     cax = divider.append_axes('right', size='5%', pad=0.05)
#     cbar = plt.colorbar(graph, cax=cax)
#     cbar_ticks = np.unique([int(x) for x in cbar.get_ticks()])
#     cax.clear()
#     cbar = plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax, ticks=cbar_ticks)
#     ax7.draw
#     fig7.savefig(Path.joinpath(output_dir, r"em_contnd_mean_for_angle_fit.tif"), dpi=dpi_graph)
#     fig7.clf()
    
    
    
if np.any(em_angle_mean_cant_time_corr):
    last_multi_n = len((em_angle_mean_cant_time_corr['count'] > n_size_thresh_for_xlim).values) - 1 - np.argmax((em_angle_mean_cant_time_corr['count'] > n_size_thresh_for_xlim).values[::-1])
    first_multi_n = np.argmax((em_angle_mean_cant_time_corr['count'] > n_size_thresh_for_xlim).values)
    fig, ax = plt.subplots(figsize=(width_2col, height_2col))
    n_size = em_angle_mean_cant_time_corr['count']
    cmap = plt.cm.get_cmap('jet', n_size.max())
    norm = mpl.colors.Normalize(vmin=n_size.min()-0.5, vmax=n_size.max()+0.5)    
    ax.errorbar(em_angle_mean_cant_time_corr['Time (minutes)'], em_angle_mean_cant_time_corr['mean'], em_angle_mean_cant_time_corr['std'], ls='None', elinewidth=1, ecolor='grey')
    graph = ax.scatter(em_angle_mean_cant_time_corr['Time (minutes)'], em_angle_mean_cant_time_corr['mean'], marker='.', s=10, c=n_size, cmap=cmap, zorder=9)#, norm=matplotlib.colors.LogNorm()
    ax.set_xlim(em_angle_mean_cant_time_corr['Time (minutes)'].min()-1, em_angle_mean_cant_time_corr['Time (minutes)'].max()+1)
    ax.axhline(ee_genlog_popt_angle[3], c='g', ls='--')
    ax.axhline(ee_genlog_popt_angle[2], c='g', ls='--')
    ax.axhline(two_circ_interp_angle(ee_first_valid_area).mean(), c='g', ls='-', lw=0.5)
    #ax.axhline(two_circ_interp_angle(mm_first_valid_area).mean(), c='r', ls='--', lw=0.5)
    #ax.axhline(two_circ_interp_angle(nn_first_valid_area).mean(), c='b', ls='--', lw=0.5)
    ax.axhline(two_circ_interp_angle(em_first_valid_area).mean(), c='c', ls='--', lw=0.5)
    ax.set_ylim(0, 220)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(graph, cax=cax)
    cbar_ticks = np.unique([int(x) for x in cbar.get_ticks()])
    cax.clear()
    cbar = plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax, ticks=cbar_ticks)
    ax.draw
    fig.savefig(Path.joinpath(output_dir, r"em_angle_mean_cant_time_corr.tif"), dpi=dpi_graph)
    fig.clf()
    
    
if np.any(em_fit_contact_mean_cant_time_corr):
    last_multi_n = len((em_fit_contact_mean_cant_time_corr['count'] > n_size_thresh_for_xlim).values) - 1 - np.argmax((em_fit_contact_mean_cant_time_corr['count'] > n_size_thresh_for_xlim).values[::-1])
    first_multi_n = np.argmax((em_fit_contact_mean_cant_time_corr['count'] > n_size_thresh_for_xlim).values)
    fig, ax = plt.subplots(figsize=(width_2col, height_2col))
    n_size = em_fit_contact_mean_cant_time_corr['count']
    cmap = plt.cm.get_cmap('jet', n_size.max())
    norm = mpl.colors.Normalize(vmin=n_size.min()-0.5, vmax=n_size.max()+0.5)    
    ax.errorbar(em_fit_contact_mean_cant_time_corr['Time (minutes)'], em_fit_contact_mean_cant_time_corr['mean'], em_fit_contact_mean_cant_time_corr['std'], ls='None', elinewidth=1, ecolor='grey')
    graph = ax.scatter(em_fit_contact_mean_cant_time_corr['Time (minutes)'], em_fit_contact_mean_cant_time_corr['mean'], marker='.', s=10, c=n_size, cmap=cmap, zorder=9)#, norm=matplotlib.colors.LogNorm()
    ax.set_xlim(em_fit_contact_mean_cant_time_corr['Time (minutes)'].min()-1, em_fit_contact_mean_cant_time_corr['Time (minutes)'].max()+1)
    ax.axhline(ee_genlog_popt_contnd[3], c='g', ls='--')
    ax.axhline(ee_genlog_popt_contnd[2], c='g', ls='--')
    ax.axhline(two_circ_interp_contnd(ee_first_valid_area).mean(), c='g', ls='-', lw=0.5)
    #ax.axhline(two_circ_interp_contnd(mm_first_valid_area).mean(), c='r', ls='--', lw=0.5)
    #ax.axhline(two_circ_interp_contnd(nn_first_valid_area).mean(), c='b', ls='--', lw=0.5)
    ax.axhline(two_circ_interp_contnd(em_first_valid_area).mean(), c='c', ls='--', lw=0.5)
    ax.set_ylim(0, 4.0)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(graph, cax=cax)
    cbar_ticks = np.unique([int(x) for x in cbar.get_ticks()])
    cax.clear()
    cbar = plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax, ticks=cbar_ticks)
    ax.draw
    fig.savefig(Path.joinpath(output_dir, r"em_fit_contact_mean_cant_time_corr.tif"), dpi=dpi_graph)
    fig.clf()





###############################################################################

# Plot all ecto-ecto in dissociation buffer data in one graph after 
# time-correcting them:
fig6, ax6 = plt.subplots(figsize=(width_2col/3, height_2col/3))
for i,x in enumerate(eedb_contact_diam_non_dim_df_list_dropna):
    if eedb_list_dropna_fit_validity_df['growth_phase'].iloc[i]:
        ax6.scatter(x['Time (minutes)']-eedb_indiv_logistic_contnd_df.iloc[i]['popt_x2'], x['Contact Surface Length (N.D.)'], marker='.', s=1)
#ax6.axhline(ee_popt_contnd[3], c='g', ls='--')
#ax6.axhline(ee_popt_contnd[2], c='g', ls='--')
#ax6.set_ylim(0, 4.0)
ax6.minorticks_on()
fig6.savefig(Path.joinpath(output_dir, r"eedb_contactnds.tif"), dpi=dpi_graph, bbox_inches='tight')
fig6.clf()

# Plot all eedb data in one graph after time-correcting them:
fig4, ax4 = plt.subplots(figsize=(width_2col/3, height_2col/3))
for i,x in enumerate(eedb_list_dropna):
    if eedb_list_dropna_fit_validity_df['growth_phase'].iloc[i]:
        ax4.scatter(x['Time (minutes)']-eedb_indiv_logistic_cont_df.iloc[i]['popt_x2'], x['Contact Surface Length (px)']*um_per_pixel, marker='.', s=1)
ax4.minorticks_on()
fig4.savefig(Path.joinpath(output_dir, r"eedb_contacts.tif"), dpi=dpi_graph, bbox_inches='tight')
fig4.clf()

# Plot all eedb data in one graph after time-correcting them:
fig5, ax5 = plt.subplots(figsize=(width_2col/3, height_2col/3))
for i,x in enumerate(eedb_list_dropna):
    if eedb_list_dropna_angle_fit_validity_df['growth_phase'].iloc[i]:
        ax5.scatter(x['Time (minutes)']-eedb_indiv_logistic_angle_df.iloc[i]['popt_x2'], x['Average Angle (deg)'], marker='.', s=1)
ax5.minorticks_on()
fig5.savefig(Path.joinpath(output_dir, r"eedb_angles.tif"), dpi=dpi_graph, bbox_inches='tight')
fig5.clf()

#Also plot the data that could not be time corrected (if any):
fig6, ax6 = plt.subplots(figsize=(width_2col/3, height_2col/3))
for i,x in enumerate(eedb_contact_diam_non_dim_df_list_dropna):
    if eedb_list_dropna_fit_validity_df['growth_phase'].iloc[i] == False:
        ax6.scatter(x['Time (minutes)']-eedb_indiv_logistic_contnd_df.iloc[i]['popt_x2'], x['Contact Surface Length (N.D.)'], marker='.', s=5)
        ax6.plot(x['Time (minutes)']-eedb_indiv_logistic_contnd_df.iloc[i]['popt_x2'], x['Contact Surface Length (N.D.)'], lw=1)
#ax6.axhline(ee_popt_contnd[3], c='g', ls='--')
#ax6.axhline(ee_popt_contnd[2], c='g', ls='--')
ax6.set_ylim(0, 4.0)
ax6.minorticks_on()
fig6.savefig(Path.joinpath(output_dir, r"eedb_contactnds_bad_fit.tif"), dpi=dpi_graph, bbox_inches='tight')
fig6.clf()

fig4, ax4 = plt.subplots(figsize=(width_2col/3, height_2col/3))
for i,x in enumerate(eedb_list_dropna):
    if eedb_list_dropna_fit_validity_df['growth_phase'].iloc[i] == False:
        ax4.scatter(x['Time (minutes)']-eedb_indiv_logistic_cont_df.iloc[i]['popt_x2'], x['Contact Surface Length (px)']*um_per_pixel, marker='.', s=5)
        ax4.plot(x['Time (minutes)']-eedb_indiv_logistic_cont_df.iloc[i]['popt_x2'], x['Contact Surface Length (px)']*um_per_pixel, lw=1)
ax4.minorticks_on()
fig4.savefig(Path.joinpath(output_dir, r"eedb_contacts_bad_fit.tif"), dpi=dpi_graph, bbox_inches='tight')
fig4.clf()

fig5, ax5 = plt.subplots(figsize=(width_2col/3, height_2col/3))
for i,x in enumerate(eedb_list_dropna):
    if eedb_list_dropna_angle_fit_validity_df['growth_phase'].iloc[i] == False:
        ax5.scatter(x['Time (minutes)']-eedb_indiv_logistic_angle_df.iloc[i]['popt_x2'], x['Average Angle (deg)'], marker='.', s=5)
        ax5.plot(x['Time (minutes)']-eedb_indiv_logistic_angle_df.iloc[i]['popt_x2'], x['Average Angle (deg)'], lw=1)
ax5.set_ylim(0, 220)
ax5.minorticks_on()
fig5.savefig(Path.joinpath(output_dir, r"eedb_angles_bad_fit.tif"), dpi=dpi_graph, bbox_inches='tight')
fig5.clf()



# Plot eedb data:
if np.any(eedb_fit_contact_mean):
    mlx = MultipleLocator(10)
    mly = MultipleLocator(0.5)
    last_multi_n = len((eedb_fit_contact_mean['count'] > n_size_thresh_for_xlim).values) - 1 - np.argmax((eedb_fit_contact_mean['count'] > n_size_thresh_for_xlim).values[::-1])
    first_multi_n = np.argmax((eedb_fit_contact_mean['count'] > n_size_thresh_for_xlim).values)
    fig8, ax8 = plt.subplots(figsize=(width_2col/3, height_2col/3))
    n_size = eedb_fit_contact_mean['count']
    cmap = plt.cm.get_cmap('jet', n_size.max())
    norm = mpl.colors.Normalize(vmin=n_size.min()-0.5, vmax=n_size.max()+0.5)    
    ax8.errorbar(eedb_fit_contact_mean['Time (minutes)'], eedb_fit_contact_mean['mean'], eedb_fit_contact_mean['std'], marker='None', ls='None', elinewidth=1, ecolor='grey')
    graph = ax8.scatter(eedb_fit_contact_mean['Time (minutes)'], eedb_fit_contact_mean['mean'], marker='.', s=10, c=n_size, cmap=cmap, zorder=9)
    #ax8.axhline(ee_popt_contnd[3], c='g', ls='--')
    #ax8.axhline(ee_popt_contnd[2], c='g', ls='--')
    ax8.plot(x_fit, eedb_genlog_y_contnd, c='k')
    ax8.axhline(ee_genlog_popt_contnd[3], c='g', ls='--')
    ax8.axhline(ee_genlog_popt_contnd[2], c='g', ls='--')
    #[plt.axhline(x, c='g', ls='-', lw=0.5) for x in kde_radius_init_2_circ_nd_peaks]
    #[plt.axhline(x, c='r', ls='--', lw=0.5) for x in em_kde_radius_init_2_circ_nd_peaks]
    ax8.axhline(two_circ_interp_contnd(ee_first_valid_area).mean(), c='g', ls='-', lw=0.5)
    ax8.axhline(two_circ_interp_contnd(eedb_first_valid_area).mean(), c='orange', ls='--', lw=0.5)
    ax8.set_ylim(0, 4.0)
    ax8.minorticks_on()
    ax8.set_xlim(eedb_fit_contact_mean['Time (minutes)'].iloc[first_multi_n]-1, eedb_fit_contact_mean['Time (minutes)'].iloc[last_multi_n]+1)
    # ax8.set_xlim(eedb_fit_contact_mean['Time (minutes)'].min()-1, eedb_fit_contact_mean['Time (minutes)'].max()+1)
    ax8.xaxis.set_minor_locator(mlx)
    ax8.yaxis.set_minor_locator(mly)
    divider = make_axes_locatable(ax8)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(graph, cax=cax)
    cbar_ticks = np.unique([int(x) for x in cbar.get_ticks()])
    cax.clear()
    cbar = plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax, ticks=cbar_ticks)
    ax8.draw
    fig8.savefig(Path.joinpath(output_dir, r"eedb_contnd_mean.tif"), dpi=dpi_graph, bbox_inches='tight')
    fig8.clf()
    
    
if np.any(eedb_contact_mean):
    last_multi_n = len((eedb_contact_mean['count'] > n_size_thresh_for_xlim).values) - 1 - np.argmax((eedb_contact_mean['count'] > n_size_thresh_for_xlim).values[::-1])
    first_multi_n = np.argmax((eedb_contact_mean['count'] > n_size_thresh_for_xlim).values)
    fig6, ax6 = plt.subplots(figsize=(width_2col/3, height_2col/3))
    n_size = eedb_contact_mean['count']
    cmap = plt.cm.get_cmap('jet', n_size.max())
    norm = mpl.colors.Normalize(vmin=n_size.min()-0.5, vmax=n_size.max()+0.5)
    ax6.errorbar(eedb_contact_mean['Time (minutes)'], eedb_contact_mean['mean'], eedb_contact_mean['std'], marker='None', ls='None', elinewidth=1, ecolor='grey')
    graph = ax6.scatter(eedb_contact_mean['Time (minutes)'], eedb_contact_mean['mean'], marker='.', s=10, c=n_size, cmap=cmap, zorder=9)
    ax6.plot(x_fit, eedb_genlog_y_cont, c='k')
    ax6.axhline(ee_genlog_popt_cont[3], c='g', ls='--')
    ax6.axhline(ee_genlog_popt_cont[2], c='g', ls='--')
    ax6.axhline(two_circ_interp_cont(ee_first_valid_area).mean(), c='g', ls='-', lw=0.5)
    ax6.axhline(two_circ_interp_cont(eedb_first_valid_area).mean(), c='orange', ls='--', lw=0.5)
    ax6.set_xlim(eedb_contact_mean['Time (minutes)'].iloc[first_multi_n]-1, eedb_contact_mean['Time (minutes)'].iloc[last_multi_n]+1)
    # ax6.set_xlim(eedb_contact_mean['Time (minutes)'].min()-1, eedb_contact_mean['Time (minutes)'].max()+1)
    ax6.minorticks_on()
    divider = make_axes_locatable(ax6)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(graph, cax=cax)
    cbar_ticks = np.unique([int(x) for x in cbar.get_ticks()])
    cax.clear()
    cbar = plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax, ticks=cbar_ticks)
    ax6.draw
    fig6.savefig(Path.joinpath(output_dir, r"eedb_contact_mean.tif"), dpi=dpi_graph, bbox_inches='tight')
    fig6.clf()
    
    
if np.any(eedb_angle_mean):
    last_multi_n = len((eedb_angle_mean['count'] > n_size_thresh_for_xlim).values) - 1 - np.argmax((eedb_angle_mean['count'] > n_size_thresh_for_xlim).values[::-1])
    first_multi_n = np.argmax((eedb_angle_mean['count'] > n_size_thresh_for_xlim).values)
    fig7, ax7 = plt.subplots(figsize=(width_2col/3, height_2col/3))
    n_size = eedb_angle_mean['count']
    cmap = plt.cm.get_cmap('jet', n_size.max())
    norm = mpl.colors.Normalize(vmin=n_size.min()-0.5, vmax=n_size.max()+0.5)    
    ax7.errorbar(eedb_angle_mean['Time (minutes)'], eedb_angle_mean['mean'], eedb_angle_mean['std'], marker='None', ls='None', elinewidth=1, ecolor='grey')
    graph = ax7.scatter(eedb_angle_mean['Time (minutes)'], eedb_angle_mean['mean'], marker='.', s=10, c=n_size, cmap=cmap, zorder=9)
    ax7.plot(x_fit, eedb_genlog_y_angle, c='k')
    ax7.axhline(ee_genlog_popt_angle[3], c='g', ls='--')
    ax7.axhline(ee_genlog_popt_angle[2], c='g', ls='--')
    ax7.axhline(two_circ_interp_angle(ee_first_valid_area).mean(), c='g', ls='-', lw=0.5)
    ax7.axhline(two_circ_interp_angle(eedb_first_valid_area).mean(), c='orange', ls='--', lw=0.5)
    ax7.set_ylim(0, 220)
    ax7.minorticks_on()
    ax7.set_xlim(eedb_angle_mean['Time (minutes)'].iloc[first_multi_n]-1, eedb_angle_mean['Time (minutes)'].iloc[last_multi_n]+1)
    # ax7.set_xlim(eedb_angle_mean['Time (minutes)'].min()-1, eedb_angle_mean['Time (minutes)'].max()+1)
    divider = make_axes_locatable(ax7)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(graph, cax=cax)
    cbar_ticks = np.unique([int(x) for x in cbar.get_ticks()])
    cax.clear()
    cbar = plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax, ticks=cbar_ticks)
    ax7.draw
    fig7.savefig(Path.joinpath(output_dir, r"eedb_angle_mean.tif"), dpi=dpi_graph, bbox_inches='tight')
    fig7.clf()
    
    
    
if np.any(eedb_angle_mean_for_contnd_fit):
    last_multi_n = len((eedb_angle_mean_for_contnd_fit['count'] > n_size_thresh_for_xlim).values) - 1 - np.argmax((eedb_angle_mean_for_contnd_fit['count'] > n_size_thresh_for_xlim).values[::-1])
    first_multi_n = np.argmax((eedb_angle_mean_for_contnd_fit['count'] > n_size_thresh_for_xlim).values)
    fig6, ax6 = plt.subplots(figsize=(width_2col/3, height_2col/3))
    n_size = eedb_angle_mean_for_contnd_fit['count']
    cmap = plt.cm.get_cmap('jet', n_size.max())
    norm = mpl.colors.Normalize(vmin=n_size.min()-0.5, vmax=n_size.max()+0.5)    
    ax6.errorbar(eedb_angle_mean_for_contnd_fit['Time (minutes)'], eedb_angle_mean_for_contnd_fit['mean'], eedb_angle_mean_for_contnd_fit['std'], marker='None', ls='None', elinewidth=1, ecolor='grey')
    graph = ax6.scatter(eedb_angle_mean_for_contnd_fit['Time (minutes)'], eedb_angle_mean_for_contnd_fit['mean'], marker='.', s=10, c=n_size, cmap=cmap, zorder=9)
    ax6.plot(x_fit, eedb_genlog_y_angle_contndfit, c='k')
    ax6.set_xlim(eedb_angle_mean_for_contnd_fit['Time (minutes)'].iloc[first_multi_n]-1, eedb_angle_mean_for_contnd_fit['Time (minutes)'].iloc[last_multi_n]+1)
    # ax6.set_xlim(eedb_angle_mean_for_contnd_fit['Time (minutes)'].min()-1, eedb_angle_mean_for_contnd_fit['Time (minutes)'].max()+1)
    ax6.axhline(ee_genlog_popt_angle[3], c='g', ls='--')
    ax6.axhline(ee_genlog_popt_angle[2], c='g', ls='--')
    ax6.axhline(two_circ_interp_angle(ee_first_valid_area).mean(), c='g', ls='-', lw=0.5)
    ax6.axhline(two_circ_interp_angle(eedb_first_valid_area).mean(), c='orange', ls='--', lw=0.5)
    ax6.set_ylim(0, 220)
    ax6.minorticks_on()
    divider = make_axes_locatable(ax6)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(graph, cax=cax)
    cbar_ticks = np.unique([int(x) for x in cbar.get_ticks()])
    cax.clear()
    cbar = plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax, ticks=cbar_ticks)
    ax6.draw
    fig6.savefig(Path.joinpath(output_dir, r"eedb_angle_mean_for_contnd_fit.tif"), dpi=dpi_graph, bbox_inches='tight')
    fig6.clf()
    
    
if np.any(eedb_contnd_mean_for_angle_fit):
    last_multi_n = len((eedb_contnd_mean_for_angle_fit['count'] > n_size_thresh_for_xlim).values) - 1 - np.argmax((eedb_contnd_mean_for_angle_fit['count'] > n_size_thresh_for_xlim).values[::-1])
    first_multi_n = np.argmax((eedb_contnd_mean_for_angle_fit['count'] > n_size_thresh_for_xlim).values)
    fig7, ax7 = plt.subplots(figsize=(width_2col/3, height_2col/3))
    n_size = eedb_contnd_mean_for_angle_fit['count']
    cmap = plt.cm.get_cmap('jet', n_size.max())
    norm = mpl.colors.Normalize(vmin=n_size.min()-0.5, vmax=n_size.max()+0.5)    
    ax7.errorbar(eedb_contnd_mean_for_angle_fit['Time (minutes)'], eedb_contnd_mean_for_angle_fit['mean'], eedb_contnd_mean_for_angle_fit['std'], marker='None', ls='None', elinewidth=1, ecolor='grey')
    graph = ax7.scatter(eedb_contnd_mean_for_angle_fit['Time (minutes)'], eedb_contnd_mean_for_angle_fit['mean'], marker='.', s=10, c=n_size, cmap=cmap, zorder=9)
    ax7.plot(x_fit, eedb_genlog_y_contnd_anglefit, c='k')
    ax7.set_xlim(eedb_contnd_mean_for_angle_fit['Time (minutes)'].iloc[first_multi_n]-1, eedb_contnd_mean_for_angle_fit['Time (minutes)'].iloc[last_multi_n]+1)
    # ax7.set_xlim(eedb_contnd_mean_for_angle_fit['Time (minutes)'].min()-1, eedb_contnd_mean_for_angle_fit['Time (minutes)'].max()+1)
    ax7.axhline(ee_genlog_popt_contnd[3], c='g', ls='--')
    ax7.axhline(ee_genlog_popt_contnd[2], c='g', ls='--')
    ax7.axhline(two_circ_interp_contnd(ee_first_valid_area).mean(), c='g', ls='-', lw=0.5)
    ax7.axhline(two_circ_interp_contnd(eedb_first_valid_area).mean(), c='orange', ls='--', lw=0.5)
    ax7.set_ylim(0, 4.0)
    ax7.minorticks_on()
    divider = make_axes_locatable(ax7)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(graph, cax=cax)
    cbar_ticks = np.unique([int(x) for x in cbar.get_ticks()])
    cax.clear()
    cbar = plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax, ticks=cbar_ticks)
    ax7.draw
    fig7.savefig(Path.joinpath(output_dir, r"eedb_contnd_mean_for_angle_fit.tif"), dpi=dpi_graph, bbox_inches='tight')
    fig7.clf()
    
    
    
if np.any(eedb_angle_mean_cant_time_corr):
    last_multi_n = len((eedb_angle_mean_cant_time_corr['count'] > n_size_thresh_for_xlim).values) - 1 - np.argmax((eedb_angle_mean_cant_time_corr['count'] > n_size_thresh_for_xlim).values[::-1])
    first_multi_n = np.argmax((eedb_angle_mean_cant_time_corr['count'] > n_size_thresh_for_xlim).values)
    fig, ax = plt.subplots(figsize=(width_2col/3, height_2col/3))
    n_size = eedb_angle_mean_cant_time_corr['count']
    cmap = plt.cm.get_cmap('jet', n_size.max())
    norm = mpl.colors.Normalize(vmin=n_size.min()-0.5, vmax=n_size.max()+0.5)    
    ax.errorbar(eedb_angle_mean_cant_time_corr['Time (minutes)'], eedb_angle_mean_cant_time_corr['mean'], eedb_angle_mean_cant_time_corr['std'], ls='None', elinewidth=1, ecolor='grey')
    graph = ax.scatter(eedb_angle_mean_cant_time_corr['Time (minutes)'], eedb_angle_mean_cant_time_corr['mean'], marker='.', s=10, c=n_size, cmap=cmap, zorder=9)#, norm=matplotlib.colors.LogNorm()
    ax.set_xlim(eedb_angle_mean_cant_time_corr['Time (minutes)'].min()-1, eedb_angle_mean_cant_time_corr['Time (minutes)'].max()+1)
    ax.axhline(ee_genlog_popt_angle[3], c='g', ls='--')
    ax.axhline(ee_genlog_popt_angle[2], c='g', ls='--')
    ax.axhline(two_circ_interp_angle(ee_first_valid_area).mean(), c='g', ls='-', lw=0.5)
    #ax.axhline(two_circ_interp_angle(mm_first_valid_area).mean(), c='r', ls='--', lw=0.5)
    #ax.axhline(two_circ_interp_angle(nn_first_valid_area).mean(), c='b', ls='--', lw=0.5)
    ax.axhline(two_circ_interp_angle(eedb_first_valid_area).mean(), c='c', ls='--', lw=0.5)
    ax.set_ylim(0, 220)
    ax.minorticks_on()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(graph, cax=cax)
    cbar_ticks = np.unique([int(x) for x in cbar.get_ticks()])
    cax.clear()
    cbar = plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax, ticks=cbar_ticks)
    ax.draw
    fig.savefig(Path.joinpath(output_dir, r"eedb_angle_mean_cant_time_corr.tif"), dpi=dpi_graph, bbox_inches='tight')
    fig.clf()
    
    
if np.any(eedb_fit_contact_mean_cant_time_corr):
    mlx = MultipleLocator(10)
    mly = MultipleLocator(0.5)
    last_multi_n = len((eedb_fit_contact_mean_cant_time_corr['count'] > n_size_thresh_for_xlim).values) - 1 - np.argmax((eedb_fit_contact_mean_cant_time_corr['count'] > n_size_thresh_for_xlim).values[::-1])
    first_multi_n = np.argmax((eedb_fit_contact_mean_cant_time_corr['count'] > n_size_thresh_for_xlim).values)
    fig, ax = plt.subplots(figsize=(width_2col/3, height_2col/3))
    n_size = eedb_fit_contact_mean_cant_time_corr['count']
    cmap = plt.cm.get_cmap('jet', n_size.max())
    norm = mpl.colors.Normalize(vmin=n_size.min()-0.5, vmax=n_size.max()+0.5)    
    ax.errorbar(eedb_fit_contact_mean_cant_time_corr['Time (minutes)'], eedb_fit_contact_mean_cant_time_corr['mean'], eedb_fit_contact_mean_cant_time_corr['std'], ls='None', elinewidth=1, ecolor='grey')
    graph = ax.scatter(eedb_fit_contact_mean_cant_time_corr['Time (minutes)'], eedb_fit_contact_mean_cant_time_corr['mean'], marker='.', s=10, c=n_size, cmap=cmap, zorder=9)#, norm=matplotlib.colors.LogNorm()
    ax.set_xlim(eedb_fit_contact_mean_cant_time_corr['Time (minutes)'].min()-1, eedb_fit_contact_mean_cant_time_corr['Time (minutes)'].max()+1)
    ax.axhline(ee_genlog_popt_contnd[3], c='g', ls='--')
    ax.axhline(ee_genlog_popt_contnd[2], c='g', ls='--')
    ax.axhline(two_circ_interp_contnd(ee_first_valid_area).mean(), c='g', ls='-', lw=0.5)
    #ax.axhline(two_circ_interp_contnd(mm_first_valid_area).mean(), c='r', ls='--', lw=0.5)
    #ax.axhline(two_circ_interp_contnd(nn_first_valid_area).mean(), c='b', ls='--', lw=0.5)
    ax.axhline(two_circ_interp_contnd(eedb_first_valid_area).mean(), c='c', ls='--', lw=0.5)
    ax.set_ylim(0, 4.0)
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(mlx)
    ax.yaxis.set_minor_locator(mly)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(graph, cax=cax)
    cbar_ticks = np.unique([int(x) for x in cbar.get_ticks()])
    cax.clear()
    cbar = plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax, ticks=cbar_ticks)
    ax.draw
    fig.savefig(Path.joinpath(output_dir, r"eedb_fit_contact_mean_cant_time_corr.tif"), dpi=dpi_graph, bbox_inches='tight')
    fig.clf()





###############################################################################

# Plot all eeCcadMO-eeCcadMO data in one graph after time-correcting them:
fig6, ax6 = plt.subplots(figsize=(width_2col/3, height_2col/3))
for i,x in enumerate(eeCcadMO_contact_diam_non_dim_df_list_dropna):
    if eeCcadMO_list_dropna_fit_validity_df['growth_phase'].iloc[i]:
        ax6.scatter(x['Time (minutes)']-eeCcadMO_indiv_logistic_contnd_df.iloc[i]['popt_x2'], x['Contact Surface Length (N.D.)'], marker='.', s=1)
#ax6.axhline(ee_popt_contnd[3], c='g', ls='--')
#ax6.axhline(ee_popt_contnd[2], c='g', ls='--')
#ax6.set_ylim(0, 4.0)
ax6.minorticks_on()
fig6.savefig(Path.joinpath(output_dir, r"eeCcadMO_contactnds.tif"), dpi=dpi_graph, bbox_inches='tight')
fig6.clf()

# Plot all eeCcadMO-eeCcadMO in one graph after time-correcting them:
fig4, ax4 = plt.subplots(figsize=(width_2col/3, height_2col/3))
for i,x in enumerate(eeCcadMO_list_dropna):
    if eeCcadMO_list_dropna_fit_validity_df['growth_phase'].iloc[i]:
        ax4.scatter(x['Time (minutes)']-eeCcadMO_indiv_logistic_cont_df.iloc[i]['popt_x2'], x['Contact Surface Length (px)']*um_per_pixel, marker='.', s=1)
ax4.minorticks_on()
fig4.savefig(Path.joinpath(output_dir, r"eeCcadMO_contacts.tif"), dpi=dpi_graph, bbox_inches='tight')
fig4.clf()

# Plot all eeCcadMO-eeCcadMO in one graph after time-correcting them:
fig5, ax5 = plt.subplots(figsize=(width_2col/3, height_2col/3))
for i,x in enumerate(eeCcadMO_list_dropna):
    if eeCcadMO_list_dropna_angle_fit_validity_df['growth_phase'].iloc[i]:
        ax5.scatter(x['Time (minutes)']-eeCcadMO_indiv_logistic_angle_df.iloc[i]['popt_x2'], x['Average Angle (deg)'], marker='.', s=1)
ax5.minorticks_on()
fig5.savefig(Path.joinpath(output_dir, r"eeCcadMO_angles.tif"), dpi=dpi_graph, bbox_inches='tight')
fig5.clf()

#Also plot the data that could not be time corrected (if any):
fig6, ax6 = plt.subplots(figsize=(width_2col/3, height_2col/3))
for i,x in enumerate(eeCcadMO_contact_diam_non_dim_df_list_dropna):
    if eeCcadMO_list_dropna_fit_validity_df['growth_phase'].iloc[i] == False:
        ax6.scatter(x['Time (minutes)']-eeCcadMO_indiv_logistic_contnd_df.iloc[i]['popt_x2'], x['Contact Surface Length (N.D.)'], marker='.', s=5)
        ax6.plot(x['Time (minutes)']-eeCcadMO_indiv_logistic_contnd_df.iloc[i]['popt_x2'], x['Contact Surface Length (N.D.)'], lw=1)
#ax6.axhline(ee_popt_contnd[3], c='g', ls='--')
#ax6.axhline(ee_popt_contnd[2], c='g', ls='--')
ax6.set_ylim(0, 4.0)
ax6.minorticks_on()
fig6.savefig(Path.joinpath(output_dir, r"eeCcadMO_contactnds_bad_fit.tif"), dpi=dpi_graph, bbox_inches='tight')
fig6.clf()

fig4, ax4 = plt.subplots(figsize=(width_2col/3, height_2col/3))
for i,x in enumerate(eeCcadMO_list_dropna):
    if eeCcadMO_list_dropna_fit_validity_df['growth_phase'].iloc[i] == False:
        ax4.scatter(x['Time (minutes)']-eeCcadMO_indiv_logistic_cont_df.iloc[i]['popt_x2'], x['Contact Surface Length (px)']*um_per_pixel, marker='.', s=5)
        ax4.plot(x['Time (minutes)']-eeCcadMO_indiv_logistic_cont_df.iloc[i]['popt_x2'], x['Contact Surface Length (px)']*um_per_pixel, lw=1)
fig4.savefig(Path.joinpath(output_dir, r"eeCcadMO_contacts_bad_fit.tif"), dpi=dpi_graph, bbox_inches='tight')
fig4.clf()

fig5, ax5 = plt.subplots(figsize=(width_2col/3, height_2col/3))
for i,x in enumerate(eeCcadMO_list_dropna):
    if eeCcadMO_list_dropna_angle_fit_validity_df['growth_phase'].iloc[i] == False:
        ax5.scatter(x['Time (minutes)']-eeCcadMO_indiv_logistic_angle_df.iloc[i]['popt_x2'], x['Average Angle (deg)'], marker='.', s=5)
        ax5.plot(x['Time (minutes)']-eeCcadMO_indiv_logistic_angle_df.iloc[i]['popt_x2'], x['Average Angle (deg)'], lw=1)
ax5.set_ylim(0, 220)
ax5.minorticks_on()
fig5.savefig(Path.joinpath(output_dir, r"eeCcadMO_angles_bad_fit.tif"), dpi=dpi_graph, bbox_inches='tight')
fig5.clf()



# Plot the eeCcadMO-eeCcadMO with the data points color coded according to N size:
#n_size_thresh_for_xlim = 1
if np.any(eeCcadMO_fit_contact_mean):
    mlx = MultipleLocator(10)
    mly = MultipleLocator(0.5)
    last_multi_n = len((eeCcadMO_fit_contact_mean['count'] > n_size_thresh_for_xlim).values) - 1 - np.argmax((eeCcadMO_fit_contact_mean['count'] > n_size_thresh_for_xlim).values[::-1])
    first_multi_n = np.argmax((eeCcadMO_fit_contact_mean['count'] > n_size_thresh_for_xlim).values)
    fig, ax = plt.subplots(figsize=(width_2col/3, height_2col/3))
    n_size = eeCcadMO_fit_contact_mean['count']
    cmap = plt.cm.get_cmap('jet', n_size.max())
    norm = mpl.colors.Normalize(vmin=n_size.min()-0.5, vmax=n_size.max()+0.5)    
    ax.errorbar(eeCcadMO_fit_contact_mean['Time (minutes)'], eeCcadMO_fit_contact_mean['mean'], eeCcadMO_fit_contact_mean['std'], marker='None', ls='None', elinewidth=1, ecolor='grey')
    graph = ax.scatter(eeCcadMO_fit_contact_mean['Time (minutes)'], eeCcadMO_fit_contact_mean['mean'], marker='.', s=10, c=n_size, cmap=cmap, zorder=9)#, norm=matplotlib.colors.LogNorm()
    ax.plot(x_fit, eeCcadMO_genlog_y_contnd, zorder=10, c='k')
    ax.set_xlim(eeCcadMO_fit_contact_mean['Time (minutes)'].iloc[first_multi_n]-1, eeCcadMO_fit_contact_mean['Time (minutes)'].iloc[last_multi_n]+1)
    # ax.set_xlim(eeCcadMO_fit_contact_mean['Time (minutes)'].min()-1, eeCcadMO_fit_contact_mean['Time (minutes)'].max()+1)
    ax.axvline(growth_phase_start_eeCcadMO_contnd)
    #ax.axvline(growth_phase_start_eeCcadMO_contnd_manual, c='k', ls='--')
    ax.axvline(growth_phase_end_eeCcadMO_contnd)
    ax.axhline(ee_genlog_popt_contnd[3], c='g', ls='--')
    ax.axhline(ee_genlog_popt_contnd[2], c='g', ls='--')
    ax.axhline(two_circ_interp_contnd(ee_first_valid_area).mean(), c='g', ls='-', lw=0.5)
    ax.axhline(two_circ_interp_contnd(mm_first_valid_area).mean(), c='r', ls='--', lw=0.5)
    ax.axhline(two_circ_interp_contnd(nn_first_valid_area).mean(), c='b', ls='--', lw=0.5)
    ax.axhline(two_circ_interp_contnd(eeCcadMO_first_valid_area).mean(), c='c', ls='--', lw=0.5)
    ax.set_ylim(0, 4.0)
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(mlx)
    ax.yaxis.set_minor_locator(mly)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(graph, cax=cax)
    cbar_ticks = np.unique([int(x) for x in cbar.get_ticks()])
    cax.clear()
    cbar = plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax, ticks=cbar_ticks)
    plt.draw
    plt.savefig(Path.joinpath(output_dir, r"eeCcadMO_contactnds_gen_logistic.tif"), dpi=dpi_graph, bbox_inches='tight')
    plt.clf()
    
    
if np.any(eeCcadMO_angle_mean):
    last_multi_n = len((eeCcadMO_angle_mean['count'] > n_size_thresh_for_xlim).values) - 1 - np.argmax((eeCcadMO_angle_mean['count'] > n_size_thresh_for_xlim).values[::-1])
    first_multi_n = np.argmax((eeCcadMO_angle_mean['count'] > n_size_thresh_for_xlim).values)
    fig2, ax2 = plt.subplots(figsize=(width_2col/3, height_2col/3))
    n_size = eeCcadMO_angle_mean['count']
    cmap = plt.cm.get_cmap('jet', n_size.max())
    norm = mpl.colors.Normalize(vmin=n_size.min()-0.5, vmax=n_size.max()+0.5)    
    ax2.errorbar(eeCcadMO_angle_mean['Time (minutes)'], eeCcadMO_angle_mean['mean'], eeCcadMO_angle_mean['std'], marker='None', ls='None', elinewidth=1, ecolor='grey')
    graph = ax2.scatter(eeCcadMO_angle_mean['Time (minutes)'], eeCcadMO_angle_mean['mean'], marker='.', s=10, c=n_size, cmap=cmap, zorder=9)#, norm=matplotlib.colors.LogNorm()
    ax2.plot(x_fit, eeCcadMO_genlog_y_angle, zorder=10, c='k')
    ax2.set_xlim(eeCcadMO_angle_mean['Time (minutes)'].iloc[first_multi_n]-1, eeCcadMO_angle_mean['Time (minutes)'].iloc[last_multi_n]+1)
    # ax2.set_xlim(eeCcadMO_angle_mean['Time (minutes)'].min()-1, eeCcadMO_angle_mean['Time (minutes)'].max()+1)
    #ax2.axvline(growth_phase_start_eeCcadMO_genlog_angle)
    #ax2.axvline(growth_phase_end_eeCcadMO_genlog_angle)
    ax2.axvline(growth_phase_start_eeCcadMO_angle)
    #ax2.axvline(growth_phase_start_eeCcadMO_angle_manual, c='k', ls='--')
    ax2.axvline(growth_phase_end_eeCcadMO_angle)
    ax2.axhline(ee_genlog_popt_angle[3], c='g', ls='--')
    ax2.axhline(ee_genlog_popt_angle[2], c='g', ls='--')
    ax2.axhline(two_circ_interp_angle(ee_first_valid_area).mean(), c='g', ls='-', lw=0.5)
    ax2.axhline(two_circ_interp_angle(mm_first_valid_area).mean(), c='r', ls='--', lw=0.5)
    ax2.axhline(two_circ_interp_angle(nn_first_valid_area).mean(), c='b', ls='--', lw=0.5)
    ax2.axhline(two_circ_interp_angle(eeCcadMO_first_valid_area).mean(), c='c', ls='--', lw=0.5)
    ax2.yaxis.set_ticks(np.arange(0, 220, 20))
    ax2.set_ylim(0,220)
    ax2.minorticks_on()
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(graph, cax=cax)
    cbar_ticks = np.unique([int(x) for x in cbar.get_ticks()])
    cax.clear()
    cbar = plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax, ticks=cbar_ticks)
    ax2.draw
    fig2.savefig(Path.joinpath(output_dir, r"eeCcadMO_angles_gen_logistic.tif"), dpi=dpi_graph, bbox_inches='tight')
    fig2.clf()
    
    
    
if np.any(eeCcadMO_angle_mean_for_contnd_fit):
    last_multi_n = len((eeCcadMO_angle_mean_for_contnd_fit['count'] > n_size_thresh_for_xlim).values) - 1 - np.argmax((eeCcadMO_angle_mean_for_contnd_fit['count'] > n_size_thresh_for_xlim).values[::-1])
    first_multi_n = np.argmax((eeCcadMO_angle_mean_for_contnd_fit['count'] > n_size_thresh_for_xlim).values)
    fig, ax = plt.subplots(figsize=(width_2col/3, height_2col/3))
    n_size = eeCcadMO_angle_mean_for_contnd_fit['count']
    cmap = plt.cm.get_cmap('jet', n_size.max())
    norm = mpl.colors.Normalize(vmin=n_size.min()-0.5, vmax=n_size.max()+0.5)    
    ax.errorbar(eeCcadMO_angle_mean_for_contnd_fit['Time (minutes)'], eeCcadMO_angle_mean_for_contnd_fit['mean'], eeCcadMO_angle_mean_for_contnd_fit['std'], marker='None', ls='None', elinewidth=1, ecolor='grey')
    graph = ax.scatter(eeCcadMO_angle_mean_for_contnd_fit['Time (minutes)'], eeCcadMO_angle_mean_for_contnd_fit['mean'], marker='.', s=10, c=n_size, cmap=cmap, zorder=9)#, norm=matplotlib.colors.LogNorm()
    ax.plot(x_fit, eeCcadMO_genlog_y_angle_contndfit, zorder=10, c='k')
    ax.set_xlim(eeCcadMO_angle_mean_for_contnd_fit['Time (minutes)'].iloc[first_multi_n]-1, eeCcadMO_angle_mean_for_contnd_fit['Time (minutes)'].iloc[last_multi_n]+1)
    # ax.set_xlim(eeCcadMO_angle_mean_for_contnd_fit['Time (minutes)'].min()-1, eeCcadMO_angle_mean_for_contnd_fit['Time (minutes)'].max()+1)
    #ax.axvline(growth_phase_start_eeCcadMO_genlog_angle)
    #ax.axvline(growth_phase_end_eeCcadMO_genlog_angle)
    #ax.axvline(growth_phase_start_eeCcadMO_angle)
    #ax.axvline(growth_phase_start_eeCcadMO_angle_manual, c='k', ls='--')
    #ax.axvline(growth_phase_end_eeCcadMO_angle)
    ax.axhline(ee_genlog_popt_angle[3], c='g', ls='--')
    ax.axhline(ee_genlog_popt_angle[2], c='g', ls='--')
    ax.axhline(two_circ_interp_angle(ee_first_valid_area).mean(), c='g', ls='-', lw=0.5)
    ax.axhline(two_circ_interp_angle(mm_first_valid_area).mean(), c='r', ls='--', lw=0.5)
    ax.axhline(two_circ_interp_angle(nn_first_valid_area).mean(), c='b', ls='--', lw=0.5)
    ax.axhline(two_circ_interp_angle(eeCcadMO_first_valid_area).mean(), c='c', ls='--', lw=0.5)
    ax.yaxis.set_ticks(np.arange(0, 220, 20))
    ax.set_ylim(0,220)
    ax.minorticks_on()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(graph, cax=cax)
    cbar_ticks = np.unique([int(x) for x in cbar.get_ticks()])
    cax.clear()
    cbar = plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax, ticks=cbar_ticks)
    plt.draw
    plt.savefig(Path.joinpath(output_dir, r"eeCcadMO_angle_mean_for_contnd_fit.tif"), dpi=dpi_graph, bbox_inches='tight')
    plt.clf()
    
    
if np.any(eeCcadMO_contnd_mean_for_angle_fit):
    last_multi_n = len((eeCcadMO_contnd_mean_for_angle_fit['count'] > n_size_thresh_for_xlim).values) - 1 - np.argmax((eeCcadMO_contnd_mean_for_angle_fit['count'] > n_size_thresh_for_xlim).values[::-1])
    first_multi_n = np.argmax((eeCcadMO_contnd_mean_for_angle_fit['count'] > n_size_thresh_for_xlim).values)
    fig2, ax2 = plt.subplots(figsize=(width_2col/3, height_2col/3))
    n_size = eeCcadMO_contnd_mean_for_angle_fit['count']
    cmap = plt.cm.get_cmap('jet', n_size.max())
    norm = mpl.colors.Normalize(vmin=n_size.min()-0.5, vmax=n_size.max()+0.5)    
    ax2.errorbar(eeCcadMO_contnd_mean_for_angle_fit['Time (minutes)'], eeCcadMO_contnd_mean_for_angle_fit['mean'], eeCcadMO_contnd_mean_for_angle_fit['std'], marker='None', ls='None', elinewidth=1, ecolor='grey')
    graph = ax2.scatter(eeCcadMO_contnd_mean_for_angle_fit['Time (minutes)'], eeCcadMO_contnd_mean_for_angle_fit['mean'], marker='.', s=10, c=n_size, cmap=cmap, zorder=9)#, norm=matplotlib.colors.LogNorm()
    ax2.plot(x_fit, eeCcadMO_genlog_y_contnd_anglefit, zorder=10, c='k')
    ax2.set_xlim(eeCcadMO_contnd_mean_for_angle_fit['Time (minutes)'].iloc[first_multi_n]-1, eeCcadMO_contnd_mean_for_angle_fit['Time (minutes)'].iloc[last_multi_n]+1)
    # ax2.set_xlim(eeCcadMO_contnd_mean_for_angle_fit['Time (minutes)'].min()-1, eeCcadMO_contnd_mean_for_angle_fit['Time (minutes)'].max()+1)
    #ax2.axvline(growth_phase_start_eeCcadMO_contnd)
    #ax2.axvline(growth_phase_start_eeCcadMO_contnd_manual, c='k', ls='--')
    #ax2.axvline(growth_phase_end_eeCcadMO_contnd)
    ax2.axhline(ee_genlog_popt_contnd[3], c='g', ls='--')
    ax2.axhline(ee_genlog_popt_contnd[2], c='g', ls='--')
    ax2.axhline(two_circ_interp_contnd(ee_first_valid_area).mean(), c='g', ls='-', lw=0.5)
    ax2.axhline(two_circ_interp_contnd(mm_first_valid_area).mean(), c='r', ls='--', lw=0.5)
    ax2.axhline(two_circ_interp_contnd(nn_first_valid_area).mean(), c='b', ls='--', lw=0.5)
    ax2.axhline(two_circ_interp_contnd(eeCcadMO_first_valid_area).mean(), c='c', ls='--', lw=0.5)
    ax2.set_ylim(0, 4.0)
    ax2.minorticks_on()
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(graph, cax=cax)
    cbar_ticks = np.unique([int(x) for x in cbar.get_ticks()])
    cax.clear()
    cbar = plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax, ticks=cbar_ticks)
    ax2.draw
    fig2.savefig(Path.joinpath(output_dir, r"eeCcadMO_contnd_mean_for_angle_fit.tif"), dpi=dpi_graph, bbox_inches='tight')
    fig2.clf()
    
    
    
if np.any(eeCcadMO_angle_mean_cant_time_corr):
    last_multi_n = len((eeCcadMO_angle_mean_cant_time_corr['count'] > n_size_thresh_for_xlim).values) - 1 - np.argmax((eeCcadMO_angle_mean_cant_time_corr['count'] > n_size_thresh_for_xlim).values[::-1])
    first_multi_n = np.argmax((eeCcadMO_angle_mean_cant_time_corr['count'] > n_size_thresh_for_xlim).values)
    fig, ax = plt.subplots(figsize=(width_2col/3, height_2col/3))
    n_size = eeCcadMO_angle_mean_cant_time_corr['count']
    cmap = plt.cm.get_cmap('jet', n_size.max())
    norm = mpl.colors.Normalize(vmin=n_size.min()-0.5, vmax=n_size.max()+0.5)    
    ax.errorbar(eeCcadMO_angle_mean_cant_time_corr['Time (minutes)'], eeCcadMO_angle_mean_cant_time_corr['mean'], eeCcadMO_angle_mean_cant_time_corr['std'], ls='None', elinewidth=1, ecolor='grey')
    graph = ax.scatter(eeCcadMO_angle_mean_cant_time_corr['Time (minutes)'], eeCcadMO_angle_mean_cant_time_corr['mean'], marker='.', s=10, c=n_size, cmap=cmap, zorder=9)#, norm=matplotlib.colors.LogNorm()
    ax.set_xlim(eeCcadMO_angle_mean_cant_time_corr['Time (minutes)'].min()-1, eeCcadMO_angle_mean_cant_time_corr['Time (minutes)'].max()+1)
    ax.axhline(ee_genlog_popt_angle[3], c='g', ls='--')
    ax.axhline(ee_genlog_popt_angle[2], c='g', ls='--')
    ax.axhline(two_circ_interp_angle(ee_first_valid_area).mean(), c='g', ls='-', lw=0.5)
    ax.axhline(two_circ_interp_angle(mm_first_valid_area).mean(), c='r', ls='--', lw=0.5)
    ax.axhline(two_circ_interp_angle(nn_first_valid_area).mean(), c='b', ls='--', lw=0.5)
    ax.axhline(two_circ_interp_angle(eeCcadMO_first_valid_area).mean(), c='c', ls='--', lw=0.5)
    ax.set_ylim(0, 220)
    ax.minorticks_on()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(graph, cax=cax)
    cbar_ticks = np.unique([int(x) for x in cbar.get_ticks()])
    cax.clear()
    cbar = plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax, ticks=cbar_ticks)
    ax.draw
    fig.savefig(Path.joinpath(output_dir, r"eeCcadMO_angle_mean_cant_time_corr.tif"), dpi=dpi_graph, bbox_inches='tight')
    fig.clf()
    
    
if np.any(eeCcadMO_fit_contact_mean_cant_time_corr):
    mlx = MultipleLocator(10)
    mly = MultipleLocator(0.5)
    last_multi_n = len((eeCcadMO_fit_contact_mean_cant_time_corr['count'] > n_size_thresh_for_xlim).values) - 1 - np.argmax((eeCcadMO_fit_contact_mean_cant_time_corr['count'] > n_size_thresh_for_xlim).values[::-1])
    first_multi_n = np.argmax((eeCcadMO_fit_contact_mean_cant_time_corr['count'] > n_size_thresh_for_xlim).values)
    fig, ax = plt.subplots(figsize=(width_2col/3, height_2col/3))
    n_size = eeCcadMO_fit_contact_mean_cant_time_corr['count']
    cmap = plt.cm.get_cmap('jet', n_size.max())
    norm = mpl.colors.Normalize(vmin=n_size.min()-0.5, vmax=n_size.max()+0.5)    
    ax.errorbar(eeCcadMO_fit_contact_mean_cant_time_corr['Time (minutes)'], eeCcadMO_fit_contact_mean_cant_time_corr['mean'], eeCcadMO_fit_contact_mean_cant_time_corr['std'], ls='None', elinewidth=1, ecolor='grey')
    graph = ax.scatter(eeCcadMO_fit_contact_mean_cant_time_corr['Time (minutes)'], eeCcadMO_fit_contact_mean_cant_time_corr['mean'], marker='.', s=10, c=n_size, cmap=cmap, zorder=9)#, norm=matplotlib.colors.LogNorm()
    ax.set_xlim(eeCcadMO_fit_contact_mean_cant_time_corr['Time (minutes)'].min()-1, eeCcadMO_fit_contact_mean_cant_time_corr['Time (minutes)'].max()+1)
    ax.axhline(ee_genlog_popt_contnd[3], c='g', ls='--')
    ax.axhline(ee_genlog_popt_contnd[2], c='g', ls='--')
    ax.axhline(two_circ_interp_contnd(ee_first_valid_area).mean(), c='g', ls='-', lw=0.5)
    ax.axhline(two_circ_interp_contnd(mm_first_valid_area).mean(), c='r', ls='--', lw=0.5)
    ax.axhline(two_circ_interp_contnd(nn_first_valid_area).mean(), c='b', ls='--', lw=0.5)
    ax.axhline(two_circ_interp_contnd(eeCcadMO_first_valid_area).mean(), c='c', ls='--', lw=0.5)
    ax.set_ylim(0, 4.0)
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(mlx)
    ax.yaxis.set_minor_locator(mly)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(graph, cax=cax)
    cbar_ticks = np.unique([int(x) for x in cbar.get_ticks()])
    cax.clear()
    cbar = plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax, ticks=cbar_ticks)
    ax.draw
    fig.savefig(Path.joinpath(output_dir, r"eeCcadMO_fit_contact_mean_cant_time_corr.tif"), dpi=dpi_graph, bbox_inches='tight')
    fig.clf()





###############################################################################

# Plot all eeRcadMO-eeRcadMO data in one graph after time-correcting them:
fig6, ax6 = plt.subplots(figsize=(width_2col/3, height_2col/3))
for i,x in enumerate(eeRcadMO_contact_diam_non_dim_df_list_dropna):
    if eeRcadMO_list_dropna_fit_validity_df['growth_phase'].iloc[i]:
        ax6.scatter(x['Time (minutes)']-eeRcadMO_indiv_logistic_contnd_df.iloc[i]['popt_x2'], x['Contact Surface Length (N.D.)'], marker='.', s=1)
#ax6.axhline(ee_popt_contnd[3], c='g', ls='--')
#ax6.axhline(ee_popt_contnd[2], c='g', ls='--')
#ax6.set_ylim(0, 4.0)
ax6.minorticks_on()
fig6.savefig(Path.joinpath(output_dir, r"eeRcadMO_contactnds.tif"), dpi=dpi_graph, bbox_inches='tight')
fig6.clf()

# Plot all eeRcadMO-eeRcadMO in one graph after time-correcting them:
fig4, ax4 = plt.subplots(figsize=(width_2col/3, height_2col/3))
for i,x in enumerate(eeRcadMO_list_dropna):
    if eeRcadMO_list_dropna_fit_validity_df['growth_phase'].iloc[i]:
        ax4.scatter(x['Time (minutes)']-eeRcadMO_indiv_logistic_cont_df.iloc[i]['popt_x2'], x['Contact Surface Length (px)']*um_per_pixel, marker='.', s=1)
ax4.minorticks_on()
fig4.savefig(Path.joinpath(output_dir, r"eeRcadMO_contacts.tif"), dpi=dpi_graph, bbox_inches='tight')
fig4.clf()

# Plot all eeRcadMO-eeRcadMO in one graph after time-correcting them:
fig5, ax5 = plt.subplots(figsize=(width_2col/3, height_2col/3))
for i,x in enumerate(eeRcadMO_list_dropna):
    if eeRcadMO_list_dropna_angle_fit_validity_df['growth_phase'].iloc[i]:
        ax5.scatter(x['Time (minutes)']-eeRcadMO_indiv_logistic_angle_df.iloc[i]['popt_x2'], x['Average Angle (deg)'], marker='.', s=1)
ax5.minorticks_on()
fig5.savefig(Path.joinpath(output_dir, r"eeRcadMO_angles.tif"), dpi=dpi_graph, bbox_inches='tight')
fig5.clf()

#Also plot the data that could not be time corrected (if any):
fig6, ax6 = plt.subplots(figsize=(width_2col/3, height_2col/3))
for i,x in enumerate(eeRcadMO_contact_diam_non_dim_df_list_dropna):
    if eeRcadMO_list_dropna_fit_validity_df['growth_phase'].iloc[i] == False:
        ax6.scatter(x['Time (minutes)']-eeRcadMO_indiv_logistic_contnd_df.iloc[i]['popt_x2'], x['Contact Surface Length (N.D.)'], marker='.', s=5)
        ax6.plot(x['Time (minutes)']-eeRcadMO_indiv_logistic_contnd_df.iloc[i]['popt_x2'], x['Contact Surface Length (N.D.)'], lw=1)
#ax6.axhline(ee_popt_contnd[3], c='g', ls='--')
#ax6.axhline(ee_popt_contnd[2], c='g', ls='--')
ax6.set_ylim(0, 4.0)
ax6.minorticks_on()
fig6.savefig(Path.joinpath(output_dir, r"eeRcadMO_contactnds_bad_fit.tif"), dpi=dpi_graph, bbox_inches='tight')
fig6.clf()

fig4, ax4 = plt.subplots(figsize=(width_2col/3, height_2col/3))
for i,x in enumerate(eeRcadMO_list_dropna):
    if eeRcadMO_list_dropna_fit_validity_df['growth_phase'].iloc[i] == False:
        ax4.scatter(x['Time (minutes)']-eeRcadMO_indiv_logistic_cont_df.iloc[i]['popt_x2'], x['Contact Surface Length (px)']*um_per_pixel, marker='.', s=5)
        ax4.plot(x['Time (minutes)']-eeRcadMO_indiv_logistic_cont_df.iloc[i]['popt_x2'], x['Contact Surface Length (px)']*um_per_pixel, lw=1)
fig4.savefig(Path.joinpath(output_dir, r"eeRcadMO_contacts_bad_fit.tif"), dpi=dpi_graph)
fig4.clf()

fig5, ax5 = plt.subplots(figsize=(width_2col/3, height_2col/3))
for i,x in enumerate(eeRcadMO_list_dropna):
    if eeRcadMO_list_dropna_angle_fit_validity_df['growth_phase'].iloc[i] == False:
        ax5.scatter(x['Time (minutes)']-eeRcadMO_indiv_logistic_angle_df.iloc[i]['popt_x2'], x['Average Angle (deg)'], marker='.', s=5)
        ax5.plot(x['Time (minutes)']-eeRcadMO_indiv_logistic_angle_df.iloc[i]['popt_x2'], x['Average Angle (deg)'], lw=1)
ax5.set_ylim(0, 220)
ax5.minorticks_on()
fig5.savefig(Path.joinpath(output_dir, r"eeRcadMO_angles_bad_fit.tif"), dpi=dpi_graph, bbox_inches='tight')
fig5.clf()



# Plot the eeRcadMO-eeRcadMO with the data points color coded according to N size:
#n_size_thresh_for_xlim = 1
if np.any(eeRcadMO_fit_contact_mean):
    mlx = MultipleLocator(10)
    mly = MultipleLocator(0.5)
    last_multi_n = len((eeRcadMO_fit_contact_mean['count'] > n_size_thresh_for_xlim).values) - 1 - np.argmax((eeRcadMO_fit_contact_mean['count'] > n_size_thresh_for_xlim).values[::-1])
    first_multi_n = np.argmax((eeRcadMO_fit_contact_mean['count'] > n_size_thresh_for_xlim).values)
    fig, ax = plt.subplots(figsize=(width_2col/3, height_2col/3))
    n_size = eeRcadMO_fit_contact_mean['count']
    cmap = plt.cm.get_cmap('jet', n_size.max())
    norm = mpl.colors.Normalize(vmin=n_size.min()-0.5, vmax=n_size.max()+0.5)    
    ax.errorbar(eeRcadMO_fit_contact_mean['Time (minutes)'], eeRcadMO_fit_contact_mean['mean'], eeRcadMO_fit_contact_mean['std'], marker='None', ls='None', elinewidth=1, ecolor='grey')
    graph = ax.scatter(eeRcadMO_fit_contact_mean['Time (minutes)'], eeRcadMO_fit_contact_mean['mean'], marker='.', s=10, c=n_size, cmap=cmap, zorder=9)#, norm=matplotlib.colors.LogNorm()
    ax.plot(x_fit, eeRcadMO_genlog_y_contnd, zorder=10, c='k')
    ax.set_xlim(eeRcadMO_fit_contact_mean['Time (minutes)'].iloc[first_multi_n]-1, eeRcadMO_fit_contact_mean['Time (minutes)'].iloc[last_multi_n]+1)
    # ax.set_xlim(eeRcadMO_fit_contact_mean['Time (minutes)'].min()-1, eeRcadMO_fit_contact_mean['Time (minutes)'].max()+1)
    ax.axvline(growth_phase_start_eeRcadMO_contnd)
    #ax.axvline(growth_phase_start_eeRcadMO_contnd_manual, c='k', ls='--')
    ax.axvline(growth_phase_end_eeRcadMO_contnd)
    ax.axhline(ee_genlog_popt_contnd[3], c='g', ls='--')
    ax.axhline(ee_genlog_popt_contnd[2], c='g', ls='--')
    ax.axhline(two_circ_interp_contnd(ee_first_valid_area).mean(), c='g', ls='-', lw=0.5)
    ax.axhline(two_circ_interp_contnd(mm_first_valid_area).mean(), c='r', ls='--', lw=0.5)
    ax.axhline(two_circ_interp_contnd(nn_first_valid_area).mean(), c='b', ls='--', lw=0.5)
    ax.axhline(two_circ_interp_contnd(eeRcadMO_first_valid_area).mean(), c='m', ls='--', lw=0.5)
    ax.set_ylim(0, 4.0)
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(mlx)
    ax.yaxis.set_minor_locator(mly)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(graph, cax=cax)
    cbar_ticks = np.unique([int(x) for x in cbar.get_ticks()])
    cax.clear()
    cbar = plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax, ticks=cbar_ticks)
    plt.draw
    plt.savefig(Path.joinpath(output_dir, r"eeRcadMO_contactnds_gen_logistic.tif"), dpi=dpi_graph, bbox_inches='tight')
    plt.clf()
    
    
if np.any(eeRcadMO_angle_mean):
    last_multi_n = len((eeRcadMO_angle_mean['count'] > n_size_thresh_for_xlim).values) - 1 - np.argmax((eeRcadMO_angle_mean['count'] > n_size_thresh_for_xlim).values[::-1])
    first_multi_n = np.argmax((eeRcadMO_angle_mean['count'] > n_size_thresh_for_xlim).values)
    fig2, ax2 = plt.subplots(figsize=(width_2col/3, height_2col/3))
    n_size = eeRcadMO_angle_mean['count']
    cmap = plt.cm.get_cmap('jet', n_size.max())
    norm = mpl.colors.Normalize(vmin=n_size.min()-0.5, vmax=n_size.max()+0.5)    
    ax2.errorbar(eeRcadMO_angle_mean['Time (minutes)'], eeRcadMO_angle_mean['mean'], eeRcadMO_angle_mean['std'], marker='None', ls='None', elinewidth=1, ecolor='grey')
    graph = ax2.scatter(eeRcadMO_angle_mean['Time (minutes)'], eeRcadMO_angle_mean['mean'], marker='.', s=10, c=n_size, cmap=cmap, zorder=9)#, norm=matplotlib.colors.LogNorm()
    ax2.plot(x_fit, eeRcadMO_genlog_y_angle, zorder=10, c='k')
    ax2.set_xlim(eeRcadMO_angle_mean['Time (minutes)'].iloc[first_multi_n]-1, eeRcadMO_angle_mean['Time (minutes)'].iloc[last_multi_n]+1)
    # ax2.set_xlim(eeRcadMO_angle_mean['Time (minutes)'].min()-1, eeRcadMO_angle_mean['Time (minutes)'].max()+1)
    #ax2.axvline(growth_phase_start_eeRcadMO_genlog_angle)
    #ax2.axvline(growth_phase_end_eeRcadMO_genlog_angle)
    ax2.axvline(growth_phase_start_eeRcadMO_angle)
    #ax2.axvline(growth_phase_start_eeRcadMO_angle_manual, c='k', ls='--')
    ax2.axvline(growth_phase_end_eeRcadMO_angle)
    ax2.axhline(ee_genlog_popt_angle[3], c='g', ls='--')
    ax2.axhline(ee_genlog_popt_angle[2], c='g', ls='--')
    ax2.axhline(two_circ_interp_angle(ee_first_valid_area).mean(), c='g', ls='-', lw=0.5)
    ax2.axhline(two_circ_interp_angle(mm_first_valid_area).mean(), c='r', ls='--', lw=0.5)
    ax2.axhline(two_circ_interp_angle(nn_first_valid_area).mean(), c='b', ls='--', lw=0.5)
    ax2.axhline(two_circ_interp_angle(eeRcadMO_first_valid_area).mean(), c='m', ls='--', lw=0.5)
    ax2.yaxis.set_ticks(np.arange(0, 220, 20))
    ax2.set_ylim(0,220)
    ax2.minorticks_on()
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(graph, cax=cax)
    cbar_ticks = np.unique([int(x) for x in cbar.get_ticks()])
    cax.clear()
    cbar = plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax, ticks=cbar_ticks)
    ax2.draw
    fig2.savefig(Path.joinpath(output_dir, r"eeRcadMO_angles_gen_logistic.tif"), dpi=dpi_graph, bbox_inches='tight')
    fig2.clf()
    
    
    
if np.any(eeRcadMO_angle_mean_for_contnd_fit):
    last_multi_n = len((eeRcadMO_angle_mean_for_contnd_fit['count'] > n_size_thresh_for_xlim).values) - 1 - np.argmax((eeRcadMO_angle_mean_for_contnd_fit['count'] > n_size_thresh_for_xlim).values[::-1])
    first_multi_n = np.argmax((eeRcadMO_angle_mean_for_contnd_fit['count'] > n_size_thresh_for_xlim).values)
    fig, ax = plt.subplots(figsize=(width_2col/3, height_2col/3))
    n_size = eeRcadMO_angle_mean_for_contnd_fit['count']
    cmap = plt.cm.get_cmap('jet', n_size.max())
    norm = mpl.colors.Normalize(vmin=n_size.min()-0.5, vmax=n_size.max()+0.5)    
    ax.errorbar(eeRcadMO_angle_mean_for_contnd_fit['Time (minutes)'], eeRcadMO_angle_mean_for_contnd_fit['mean'], eeRcadMO_angle_mean_for_contnd_fit['std'], marker='None', ls='None', elinewidth=1, ecolor='grey')
    graph = ax.scatter(eeRcadMO_angle_mean_for_contnd_fit['Time (minutes)'], eeRcadMO_angle_mean_for_contnd_fit['mean'], marker='.', s=10, c=n_size, cmap=cmap, zorder=9)#, norm=matplotlib.colors.LogNorm()
    ax.plot(x_fit, eeRcadMO_genlog_y_angle_contndfit, zorder=10, c='k')
    ax.set_xlim(eeRcadMO_angle_mean_for_contnd_fit['Time (minutes)'].iloc[first_multi_n]-1, eeRcadMO_angle_mean_for_contnd_fit['Time (minutes)'].iloc[last_multi_n]+1)
    # ax.set_xlim(eeRcadMO_angle_mean_for_contnd_fit['Time (minutes)'].min()-1, eeRcadMO_angle_mean_for_contnd_fit['Time (minutes)'].max()+1)
    #ax.axvline(growth_phase_start_eeRcadMO_genlog_angle)
    #ax.axvline(growth_phase_end_eeRcadMO_genlog_angle)
    #ax.axvline(growth_phase_start_eeRcadMO_angle)
    #ax.axvline(growth_phase_start_eeRcadMO_angle_manual, c='k', ls='--')
    #ax.axvline(growth_phase_end_eeRcadMO_angle)
    ax.axhline(ee_genlog_popt_angle[3], c='g', ls='--')
    ax.axhline(ee_genlog_popt_angle[2], c='g', ls='--')
    ax.axhline(two_circ_interp_angle(ee_first_valid_area).mean(), c='g', ls='-', lw=0.5)
    ax.axhline(two_circ_interp_angle(mm_first_valid_area).mean(), c='r', ls='--', lw=0.5)
    ax.axhline(two_circ_interp_angle(nn_first_valid_area).mean(), c='b', ls='--', lw=0.5)
    ax.axhline(two_circ_interp_angle(eeRcadMO_first_valid_area).mean(), c='m', ls='--', lw=0.5)
    ax.yaxis.set_ticks(np.arange(0, 220, 20))
    ax.set_ylim(0,220)
    ax.minorticks_on()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(graph, cax=cax)
    cbar_ticks = np.unique([int(x) for x in cbar.get_ticks()])
    cax.clear()
    cbar = plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax, ticks=cbar_ticks)
    plt.draw
    plt.savefig(Path.joinpath(output_dir, r"eeRcadMO_angle_mean_for_contnd_fit.tif"), dpi=dpi_graph, bbox_inches='tight')
    plt.clf()
    
    
if np.any(eeRcadMO_contnd_mean_for_angle_fit):
    last_multi_n = len((eeRcadMO_contnd_mean_for_angle_fit['count'] > n_size_thresh_for_xlim).values) - 1 - np.argmax((eeRcadMO_contnd_mean_for_angle_fit['count'] > n_size_thresh_for_xlim).values[::-1])
    first_multi_n = np.argmax((eeRcadMO_contnd_mean_for_angle_fit['count'] > n_size_thresh_for_xlim).values)
    fig2, ax2 = plt.subplots(figsize=(width_2col/3, height_2col/3))
    n_size = eeRcadMO_contnd_mean_for_angle_fit['count']
    cmap = plt.cm.get_cmap('jet', n_size.max())
    norm = mpl.colors.Normalize(vmin=n_size.min()-0.5, vmax=n_size.max()+0.5)    
    ax2.errorbar(eeRcadMO_contnd_mean_for_angle_fit['Time (minutes)'], eeRcadMO_contnd_mean_for_angle_fit['mean'], eeRcadMO_contnd_mean_for_angle_fit['std'], marker='None', ls='None', elinewidth=1, ecolor='grey')
    graph = ax2.scatter(eeRcadMO_contnd_mean_for_angle_fit['Time (minutes)'], eeRcadMO_contnd_mean_for_angle_fit['mean'], marker='.', s=10, c=n_size, cmap=cmap, zorder=9)#, norm=matplotlib.colors.LogNorm()
    ax2.plot(x_fit, eeRcadMO_genlog_y_contnd_anglefit, zorder=10, c='k')
    ax2.set_xlim(eeRcadMO_contnd_mean_for_angle_fit['Time (minutes)'].iloc[first_multi_n]-1, eeRcadMO_contnd_mean_for_angle_fit['Time (minutes)'].iloc[last_multi_n]+1)
    # ax2.set_xlim(eeRcadMO_contnd_mean_for_angle_fit['Time (minutes)'].min()-1, eeRcadMO_contnd_mean_for_angle_fit['Time (minutes)'].max()+1)
    #ax2.axvline(growth_phase_start_eeRcadMO_contnd)
    #ax2.axvline(growth_phase_start_eeRcadMO_contnd_manual, c='k', ls='--')
    #ax2.axvline(growth_phase_end_eeRcadMO_contnd)
    ax2.axhline(ee_genlog_popt_contnd[3], c='g', ls='--')
    ax2.axhline(ee_genlog_popt_contnd[2], c='g', ls='--')
    ax2.axhline(two_circ_interp_contnd(ee_first_valid_area).mean(), c='g', ls='-', lw=0.5)
    ax2.axhline(two_circ_interp_contnd(mm_first_valid_area).mean(), c='r', ls='--', lw=0.5)
    ax2.axhline(two_circ_interp_contnd(nn_first_valid_area).mean(), c='b', ls='--', lw=0.5)
    ax2.axhline(two_circ_interp_contnd(eeRcadMO_first_valid_area).mean(), c='m', ls='--', lw=0.5)
    ax2.set_ylim(0, 4.0)
    ax2.minorticks_on()
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(graph, cax=cax)
    cbar_ticks = np.unique([int(x) for x in cbar.get_ticks()])
    cax.clear()
    cbar = plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax, ticks=cbar_ticks)
    ax2.draw
    fig2.savefig(Path.joinpath(output_dir, r"eeRcadMO_contnd_mean_for_angle_fit.tif"), dpi=dpi_graph, bbox_inches='tight')
    fig2.clf()
    
    
    
if np.any(eeRcadMO_angle_mean_cant_time_corr):
    last_multi_n = len((eeRcadMO_angle_mean_cant_time_corr['count'] > n_size_thresh_for_xlim).values) - 1 - np.argmax((eeRcadMO_angle_mean_cant_time_corr['count'] > n_size_thresh_for_xlim).values[::-1])
    first_multi_n = np.argmax((eeRcadMO_angle_mean_cant_time_corr['count'] > n_size_thresh_for_xlim).values)
    fig, ax = plt.subplots(figsize=(width_2col/3, height_2col/3))
    n_size = eeRcadMO_angle_mean_cant_time_corr['count']
    cmap = plt.cm.get_cmap('jet', n_size.max())
    norm = mpl.colors.Normalize(vmin=n_size.min()-0.5, vmax=n_size.max()+0.5)    
    ax.errorbar(eeRcadMO_angle_mean_cant_time_corr['Time (minutes)'], eeRcadMO_angle_mean_cant_time_corr['mean'], eeRcadMO_angle_mean_cant_time_corr['std'], ls='None', elinewidth=1, ecolor='grey')
    graph = ax.scatter(eeRcadMO_angle_mean_cant_time_corr['Time (minutes)'], eeRcadMO_angle_mean_cant_time_corr['mean'], marker='.', s=10, c=n_size, cmap=cmap, zorder=9)#, norm=matplotlib.colors.LogNorm()
    ax.set_xlim(eeRcadMO_angle_mean_cant_time_corr['Time (minutes)'].min()-1, eeRcadMO_angle_mean_cant_time_corr['Time (minutes)'].max()+1)
    ax.axhline(ee_genlog_popt_angle[3], c='g', ls='--')
    ax.axhline(ee_genlog_popt_angle[2], c='g', ls='--')
    ax.axhline(two_circ_interp_angle(ee_first_valid_area).mean(), c='g', ls='-', lw=0.5)
    ax.axhline(two_circ_interp_angle(mm_first_valid_area).mean(), c='r', ls='--', lw=0.5)
    ax.axhline(two_circ_interp_angle(nn_first_valid_area).mean(), c='b', ls='--', lw=0.5)
    ax.axhline(two_circ_interp_angle(eeRcadMO_first_valid_area).mean(), c='m', ls='--', lw=0.5)
    ax.set_ylim(0, 220)
    ax.minorticks_on()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(graph, cax=cax)
    cbar_ticks = np.unique([int(x) for x in cbar.get_ticks()])
    cax.clear()
    cbar = plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax, ticks=cbar_ticks)
    ax.draw
    fig.savefig(Path.joinpath(output_dir, r"eeRcadMO_angle_mean_cant_time_corr.tif"), dpi=dpi_graph, bbox_inches='tight')
    fig.clf()
    
    
if np.any(eeRcadMO_fit_contact_mean_cant_time_corr):
    last_multi_n = len((eeRcadMO_fit_contact_mean_cant_time_corr['count'] > n_size_thresh_for_xlim).values) - 1 - np.argmax((eeRcadMO_fit_contact_mean_cant_time_corr['count'] > n_size_thresh_for_xlim).values[::-1])
    first_multi_n = np.argmax((eeRcadMO_fit_contact_mean_cant_time_corr['count'] > n_size_thresh_for_xlim).values)
    fig, ax = plt.subplots(figsize=(width_2col/3, height_2col/3))
    n_size = eeRcadMO_fit_contact_mean_cant_time_corr['count']
    cmap = plt.cm.get_cmap('jet', n_size.max())
    norm = mpl.colors.Normalize(vmin=n_size.min()-0.5, vmax=n_size.max()+0.5)
    ax.errorbar(eeRcadMO_fit_contact_mean_cant_time_corr['Time (minutes)'], eeRcadMO_fit_contact_mean_cant_time_corr['mean'], eeRcadMO_fit_contact_mean_cant_time_corr['std'], ls='None', elinewidth=1, ecolor='grey')
    graph = ax.scatter(eeRcadMO_fit_contact_mean_cant_time_corr['Time (minutes)'], eeRcadMO_fit_contact_mean_cant_time_corr['mean'], marker='.', s=10, c=n_size, cmap=cmap, zorder=9)#, norm=matplotlib.colors.LogNorm())
    ax.set_xlim(eeRcadMO_fit_contact_mean_cant_time_corr['Time (minutes)'].min()-1, eeRcadMO_fit_contact_mean_cant_time_corr['Time (minutes)'].max()+1)
    ax.axhline(ee_genlog_popt_contnd[3], c='g', ls='--')
    ax.axhline(ee_genlog_popt_contnd[2], c='g', ls='--')
    ax.axhline(two_circ_interp_contnd(ee_first_valid_area).mean(), c='g', ls='-', lw=0.5)
    ax.axhline(two_circ_interp_contnd(mm_first_valid_area).mean(), c='r', ls='--', lw=0.5)
    ax.axhline(two_circ_interp_contnd(nn_first_valid_area).mean(), c='b', ls='--', lw=0.5)
    ax.axhline(two_circ_interp_contnd(eeRcadMO_first_valid_area).mean(), c='m', ls='--', lw=0.5)
    ax.set_ylim(0, 4.0)
    ax.minorticks_on()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(graph, cax=cax)
    cbar_ticks = np.unique([int(x) for x in cbar.get_ticks()])
    cax.clear()
    cbar = plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax, ticks=cbar_ticks)
    ax.draw
    fig.savefig(Path.joinpath(output_dir, r"eeRcadMO_fit_contact_mean_cant_time_corr.tif"), dpi=dpi_graph, bbox_inches='tight')
    fig.clf()





###############################################################################

# Plot all eeRCcadMO-eeRCcadMO data in one graph after time-correcting them:
fig6, ax6 = plt.subplots(figsize=(width_2col/3, height_2col/3))
for i,x in enumerate(eeRCcadMO_contact_diam_non_dim_df_list_dropna):
    if eeRCcadMO_list_dropna_fit_validity_df['growth_phase'].iloc[i]:
        ax6.scatter(x['Time (minutes)']-eeRCcadMO_indiv_logistic_contnd_df.iloc[i]['popt_x2'], x['Contact Surface Length (N.D.)'], marker='.', s=1)
#ax6.axhline(ee_popt_contnd[3], c='g', ls='--')
#ax6.axhline(ee_popt_contnd[2], c='g', ls='--')
#ax6.set_ylim(0, 4.0)
ax6.minorticks_on()
fig6.savefig(Path.joinpath(output_dir, r"eeRCcadMO_contactnds.tif"), dpi=dpi_graph, bbox_inches='tight')
fig6.clf()

# Plot all eeRCcadMO-eeRCcadMO in one graph after time-correcting them:
fig4, ax4 = plt.subplots(figsize=(width_2col/3, height_2col/3))
for i,x in enumerate(eeRCcadMO_list_dropna):
    if eeRCcadMO_list_dropna_fit_validity_df['growth_phase'].iloc[i]:
        ax4.scatter(x['Time (minutes)']-eeRCcadMO_indiv_logistic_cont_df.iloc[i]['popt_x2'], x['Contact Surface Length (px)']*um_per_pixel, marker='.', s=1)
ax4.minorticks_on()
fig4.savefig(Path.joinpath(output_dir, r"eeRCcadMO_contacts.tif"), dpi=dpi_graph, bbox_inches='tight')
fig4.clf()

# Plot all eeRCcadMO-eeRCcadMO in one graph after time-correcting them:
fig5, ax5 = plt.subplots(figsize=(width_2col/3, height_2col/3))
for i,x in enumerate(eeRCcadMO_list_dropna):
    if eeRCcadMO_list_dropna_angle_fit_validity_df['growth_phase'].iloc[i]:
        ax5.scatter(x['Time (minutes)']-eeRCcadMO_indiv_logistic_angle_df.iloc[i]['popt_x2'], x['Average Angle (deg)'], marker='.', s=1)
ax5.minorticks_on()
fig5.savefig(Path.joinpath(output_dir, r"eeRCcadMO_angles.tif"), dpi=dpi_graph, bbox_inches='tight')
fig5.clf()

#Also plot the data that could not be time corrected (if any):
fig6, ax6 = plt.subplots(figsize=(width_2col/3, height_2col/3))
# mlx = MultipleLocator(10)
# mly = MultipleLocator(0.5)
for i,x in enumerate(eeRCcadMO_contact_diam_non_dim_df_list_dropna):
    if eeRCcadMO_list_dropna_fit_validity_df['growth_phase'].iloc[i] == False:
        ax6.scatter(x['Time (minutes)']-eeRCcadMO_indiv_logistic_contnd_df.iloc[i]['popt_x2'], x['Contact Surface Length (N.D.)'], marker='.', s=5)
        ax6.plot(x['Time (minutes)']-eeRCcadMO_indiv_logistic_contnd_df.iloc[i]['popt_x2'], x['Contact Surface Length (N.D.)'], lw=1)
#ax6.axhline(ee_popt_contnd[3], c='g', ls='--')
#ax6.axhline(ee_popt_contnd[2], c='g', ls='--')
ax6.set_ylim(0, 4.0)
ax6.minorticks_on()
# ax6.xaxis.set_minor_locator(mlx)
# ax6.yaxis.set_minor_locator(mly)
fig6.savefig(Path.joinpath(output_dir, r"eeRCcadMO_contactnds_bad_fit.tif"), dpi=dpi_graph, bbox_inches='tight')
fig6.clf()

fig4, ax4 = plt.subplots(figsize=(width_2col/3, height_2col/3))
for i,x in enumerate(eeRCcadMO_list_dropna):
    if eeRCcadMO_list_dropna_fit_validity_df['growth_phase'].iloc[i] == False:
        ax4.scatter(x['Time (minutes)']-eeRCcadMO_indiv_logistic_cont_df.iloc[i]['popt_x2'], x['Contact Surface Length (px)']*um_per_pixel, marker='.', s=5)
        ax4.plot(x['Time (minutes)']-eeRCcadMO_indiv_logistic_cont_df.iloc[i]['popt_x2'], x['Contact Surface Length (px)']*um_per_pixel, lw=1)
fig4.savefig(Path.joinpath(output_dir, r"eeRCcadMO_contacts_bad_fit.tif"), dpi=dpi_graph)
fig4.clf()

fig5, ax5 = plt.subplots(figsize=(width_2col/3, height_2col/3))
for i,x in enumerate(eeRCcadMO_list_dropna):
    if eeRCcadMO_list_dropna_angle_fit_validity_df['growth_phase'].iloc[i] == False:
        ax5.scatter(x['Time (minutes)']-eeRCcadMO_indiv_logistic_angle_df.iloc[i]['popt_x2'], x['Average Angle (deg)'], marker='.', s=5)
        ax5.plot(x['Time (minutes)']-eeRCcadMO_indiv_logistic_angle_df.iloc[i]['popt_x2'], x['Average Angle (deg)'], lw=1)
ax5.set_ylim(0, 220)
ax5.minorticks_on()
fig5.savefig(Path.joinpath(output_dir, r"eeRCcadMO_angles_bad_fit.tif"), dpi=dpi_graph, bbox_inches='tight')
fig5.clf()



# Plot the eeRCcadMO-eeRCcadMO with the data points color coded according to N size:
#n_size_thresh_for_xlim = 1
if np.any(eeRCcadMO_fit_contact_mean):
    mlx = MultipleLocator(10)
    mly = MultipleLocator(0.5)
    last_multi_n = len((eeRCcadMO_fit_contact_mean['count'] > n_size_thresh_for_xlim).values) - 1 - np.argmax((eeRCcadMO_fit_contact_mean['count'] > n_size_thresh_for_xlim).values[::-1])
    first_multi_n = np.argmax((eeRCcadMO_fit_contact_mean['count'] > n_size_thresh_for_xlim).values)
    fig, ax = plt.subplots(figsize=(width_2col/3, height_2col/3))
    n_size = eeRCcadMO_fit_contact_mean['count']
    cmap = plt.cm.get_cmap('jet', n_size.max())
    norm = mpl.colors.Normalize(vmin=n_size.min()-0.5, vmax=n_size.max()+0.5)    
    ax.errorbar(eeRCcadMO_fit_contact_mean['Time (minutes)'], eeRCcadMO_fit_contact_mean['mean'], eeRCcadMO_fit_contact_mean['std'], marker='None', ls='None', elinewidth=1, ecolor='grey')
    graph = ax.scatter(eeRCcadMO_fit_contact_mean['Time (minutes)'], eeRCcadMO_fit_contact_mean['mean'], marker='.', s=10, c=n_size, cmap=cmap, zorder=9)#, norm=matplotlib.colors.LogNorm()
    ax.plot(x_fit, eeRCcadMO_genlog_y_contnd, zorder=10, c='k')
    ax.set_xlim(eeRCcadMO_fit_contact_mean['Time (minutes)'].iloc[first_multi_n]-1, eeRCcadMO_fit_contact_mean['Time (minutes)'].iloc[last_multi_n]+1)
    # ax.set_xlim(eeRCcadMO_fit_contact_mean['Time (minutes)'].min()-1, eeRCcadMO_fit_contact_mean['Time (minutes)'].max()+1)
    ax.axvline(growth_phase_start_eeRCcadMO_contnd)
    #ax.axvline(growth_phase_start_eeRCcadMO_contnd_manual, c='k', ls='--')
    ax.axvline(growth_phase_end_eeRCcadMO_contnd)
    ax.axhline(ee_genlog_popt_contnd[3], c='g', ls='--')
    ax.axhline(ee_genlog_popt_contnd[2], c='g', ls='--')
    ax.axhline(two_circ_interp_contnd(ee_first_valid_area).mean(), c='g', ls='-', lw=0.5)
    ax.axhline(two_circ_interp_contnd(mm_first_valid_area).mean(), c='r', ls='--', lw=0.5)
    ax.axhline(two_circ_interp_contnd(nn_first_valid_area).mean(), c='b', ls='--', lw=0.5)
    ax.axhline(two_circ_interp_contnd(eeRCcadMO_first_valid_area).mean(), c='y', ls='--', lw=0.5)
    ax.set_ylim(0, 4.0)
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(mlx)
    ax.yaxis.set_minor_locator(mly)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(graph, cax=cax)
    cbar_ticks = np.unique([int(x) for x in cbar.get_ticks()])
    cax.clear()
    cbar = plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax, ticks=cbar_ticks)
    plt.draw
    plt.savefig(Path.joinpath(output_dir, r"eeRCcadMO_contactnds_gen_logistic.tif"), dpi=dpi_graph, bbox_inches='tight')
    plt.clf()
    
    
if np.any(eeRCcadMO_angle_mean):
    last_multi_n = len((eeRCcadMO_angle_mean['count'] > n_size_thresh_for_xlim).values) - 1 - np.argmax((eeRCcadMO_angle_mean['count'] > n_size_thresh_for_xlim).values[::-1])
    first_multi_n = np.argmax((eeRCcadMO_angle_mean['count'] > n_size_thresh_for_xlim).values)
    fig2, ax2 = plt.subplots(figsize=(width_2col/3, height_2col/3))
    n_size = eeRCcadMO_angle_mean['count']
    cmap = plt.cm.get_cmap('jet', n_size.max())
    norm = mpl.colors.Normalize(vmin=n_size.min()-0.5, vmax=n_size.max()+0.5)    
    ax2.errorbar(eeRCcadMO_angle_mean['Time (minutes)'], eeRCcadMO_angle_mean['mean'], eeRCcadMO_angle_mean['std'], marker='None', ls='None', elinewidth=1, ecolor='grey')
    graph = ax2.scatter(eeRCcadMO_angle_mean['Time (minutes)'], eeRCcadMO_angle_mean['mean'], marker='.', s=10, c=n_size, cmap=cmap, zorder=9)#, norm=matplotlib.colors.LogNorm()
    ax2.plot(x_fit, eeRCcadMO_genlog_y_angle, zorder=10, c='k')
    ax2.set_xlim(eeRCcadMO_angle_mean['Time (minutes)'].iloc[first_multi_n]-1, eeRCcadMO_angle_mean['Time (minutes)'].iloc[last_multi_n]+1)
    # ax2.set_xlim(eeRCcadMO_angle_mean['Time (minutes)'].min()-1, eeRCcadMO_angle_mean['Time (minutes)'].max()+1)
    #ax2.axvline(growth_phase_start_eeRCcadMO_genlog_angle)
    #ax2.axvline(growth_phase_end_eeRCcadMO_genlog_angle)
    ax2.axvline(growth_phase_start_eeRCcadMO_angle)
    #ax2.axvline(growth_phase_start_eeRCcadMO_angle_manual, c='k', ls='--')
    ax2.axvline(growth_phase_end_eeRCcadMO_angle)
    ax2.axhline(ee_genlog_popt_angle[3], c='g', ls='--')
    ax2.axhline(ee_genlog_popt_angle[2], c='g', ls='--')
    ax2.axhline(two_circ_interp_angle(ee_first_valid_area).mean(), c='g', ls='-', lw=0.5)
    ax2.axhline(two_circ_interp_angle(mm_first_valid_area).mean(), c='r', ls='--', lw=0.5)
    ax2.axhline(two_circ_interp_angle(nn_first_valid_area).mean(), c='b', ls='--', lw=0.5)
    ax2.axhline(two_circ_interp_angle(eeRCcadMO_first_valid_area).mean(), c='y', ls='--', lw=0.5)
    ax2.yaxis.set_ticks(np.arange(0, 220, 20))
    ax2.set_ylim(0,220)
    ax2.minorticks_on()
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(graph, cax=cax)
    cbar_ticks = np.unique([int(x) for x in cbar.get_ticks()])
    cax.clear()
    cbar = plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax, ticks=cbar_ticks)
    ax2.draw
    fig2.savefig(Path.joinpath(output_dir, r"eeRCcadMO_angles_gen_logistic.tif"), dpi=dpi_graph, bbox_inches='tight')
    fig2.clf()
    
    
    
if np.any(eeRCcadMO_angle_mean_for_contnd_fit):
    last_multi_n = len((eeRCcadMO_angle_mean_for_contnd_fit['count'] > n_size_thresh_for_xlim).values) - 1 - np.argmax((eeRCcadMO_angle_mean_for_contnd_fit['count'] > n_size_thresh_for_xlim).values[::-1])
    first_multi_n = np.argmax((eeRCcadMO_angle_mean_for_contnd_fit['count'] > n_size_thresh_for_xlim).values)
    fig, ax = plt.subplots(figsize=(width_2col/3, height_2col/3))
    n_size = eeRCcadMO_angle_mean_for_contnd_fit['count']
    cmap = plt.cm.get_cmap('jet', n_size.max())
    norm = mpl.colors.Normalize(vmin=n_size.min()-0.5, vmax=n_size.max()+0.5)    
    ax.errorbar(eeRCcadMO_angle_mean_for_contnd_fit['Time (minutes)'], eeRCcadMO_angle_mean_for_contnd_fit['mean'], eeRCcadMO_angle_mean_for_contnd_fit['std'], marker='None', ls='None', elinewidth=1, ecolor='grey')
    graph = ax.scatter(eeRCcadMO_angle_mean_for_contnd_fit['Time (minutes)'], eeRCcadMO_angle_mean_for_contnd_fit['mean'], marker='.', s=10, c=n_size, cmap=cmap, zorder=9)#, norm=matplotlib.colors.LogNorm()
    ax.plot(x_fit, eeRCcadMO_genlog_y_angle_contndfit, zorder=10, c='k')
    ax.set_xlim(eeRCcadMO_angle_mean_for_contnd_fit['Time (minutes)'].iloc[first_multi_n]-1, eeRCcadMO_angle_mean_for_contnd_fit['Time (minutes)'].iloc[last_multi_n]+1)
    # ax.set_xlim(eeRCcadMO_angle_mean_for_contnd_fit['Time (minutes)'].min()-1, eeRCcadMO_angle_mean_for_contnd_fit['Time (minutes)'].max()+1)
    #ax.axvline(growth_phase_start_eeRCcadMO_genlog_angle)
    #ax.axvline(growth_phase_end_eeRCcadMO_genlog_angle)
    #ax.axvline(growth_phase_start_eeRCcadMO_angle)
    #ax.axvline(growth_phase_start_eeRCcadMO_angle_manual, c='k', ls='--')
    #ax.axvline(growth_phase_end_eeRCcadMO_angle)
    ax.axhline(ee_genlog_popt_angle[3], c='g', ls='--')
    ax.axhline(ee_genlog_popt_angle[2], c='g', ls='--')
    ax.axhline(two_circ_interp_angle(ee_first_valid_area).mean(), c='g', ls='-', lw=0.5)
    ax.axhline(two_circ_interp_angle(mm_first_valid_area).mean(), c='r', ls='--', lw=0.5)
    ax.axhline(two_circ_interp_angle(nn_first_valid_area).mean(), c='b', ls='--', lw=0.5)
    ax.axhline(two_circ_interp_angle(eeRCcadMO_first_valid_area).mean(), c='y', ls='--', lw=0.5)
    ax.yaxis.set_ticks(np.arange(0, 220, 20))
    ax.set_ylim(0,220)
    ax.minorticks_on()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(graph, cax=cax)
    cbar_ticks = np.unique([int(x) for x in cbar.get_ticks()])
    cax.clear()
    cbar = plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax, ticks=cbar_ticks)
    plt.draw
    plt.savefig(Path.joinpath(output_dir, r"eeRCcadMO_angle_mean_for_contnd_fit.tif"), dpi=dpi_graph, bbox_inches='tight')
    plt.clf()
    
    
if np.any(eeRCcadMO_contnd_mean_for_angle_fit):
    last_multi_n = len((eeRCcadMO_contnd_mean_for_angle_fit['count'] > n_size_thresh_for_xlim).values) - 1 - np.argmax((eeRCcadMO_contnd_mean_for_angle_fit['count'] > n_size_thresh_for_xlim).values[::-1])
    first_multi_n = np.argmax((eeRCcadMO_contnd_mean_for_angle_fit['count'] > n_size_thresh_for_xlim).values)
    fig2, ax2 = plt.subplots(figsize=(width_2col/3, height_2col/3))
    n_size = eeRCcadMO_contnd_mean_for_angle_fit['count']
    cmap = plt.cm.get_cmap('jet', n_size.max())
    norm = mpl.colors.Normalize(vmin=n_size.min()-0.5, vmax=n_size.max()+0.5)    
    ax2.errorbar(eeRCcadMO_contnd_mean_for_angle_fit['Time (minutes)'], eeRCcadMO_contnd_mean_for_angle_fit['mean'], eeRCcadMO_contnd_mean_for_angle_fit['std'], marker='None', ls='None', elinewidth=1, ecolor='grey')
    graph = ax2.scatter(eeRCcadMO_contnd_mean_for_angle_fit['Time (minutes)'], eeRCcadMO_contnd_mean_for_angle_fit['mean'], marker='.', s=10, c=n_size, cmap=cmap, zorder=9)#, norm=matplotlib.colors.LogNorm()
    ax2.plot(x_fit, eeRCcadMO_genlog_y_contnd_anglefit, zorder=10, c='k')
    ax2.set_xlim(eeRCcadMO_contnd_mean_for_angle_fit['Time (minutes)'].iloc[first_multi_n]-1, eeRCcadMO_contnd_mean_for_angle_fit['Time (minutes)'].iloc[last_multi_n]+1)
    # ax2.set_xlim(eeRCcadMO_contnd_mean_for_angle_fit['Time (minutes)'].min()-1, eeRCcadMO_contnd_mean_for_angle_fit['Time (minutes)'].max()+1)
    #ax2.axvline(growth_phase_start_eeRCcadMO_contnd)
    #ax2.axvline(growth_phase_start_eeRCcadMO_contnd_manual, c='k', ls='--')
    #ax2.axvline(growth_phase_end_eeRCcadMO_contnd)
    ax2.axhline(ee_genlog_popt_contnd[3], c='g', ls='--')
    ax2.axhline(ee_genlog_popt_contnd[2], c='g', ls='--')
    ax2.axhline(two_circ_interp_contnd(ee_first_valid_area).mean(), c='g', ls='-', lw=0.5)
    ax2.axhline(two_circ_interp_contnd(mm_first_valid_area).mean(), c='r', ls='--', lw=0.5)
    ax2.axhline(two_circ_interp_contnd(nn_first_valid_area).mean(), c='b', ls='--', lw=0.5)
    ax2.axhline(two_circ_interp_contnd(eeRCcadMO_first_valid_area).mean(), c='y', ls='--', lw=0.5)
    ax2.set_ylim(0, 4.0)
    ax2.minorticks_on()
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(graph, cax=cax)
    cbar_ticks = np.unique([int(x) for x in cbar.get_ticks()])
    cax.clear()
    cbar = plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax, ticks=cbar_ticks)
    ax2.draw
    fig2.savefig(Path.joinpath(output_dir, r"eeRCcadMO_contnd_mean_for_angle_fit.tif"), dpi=dpi_graph, bbox_inches='tight')
    fig2.clf()
    
    
    
if np.any(eeRCcadMO_angle_mean_cant_time_corr):
    last_multi_n = len((eeRCcadMO_angle_mean_cant_time_corr['count'] > n_size_thresh_for_xlim).values) - 1 - np.argmax((eeRCcadMO_angle_mean_cant_time_corr['count'] > n_size_thresh_for_xlim).values[::-1])
    first_multi_n = np.argmax((eeRCcadMO_angle_mean_cant_time_corr['count'] > n_size_thresh_for_xlim).values)
    fig, ax = plt.subplots(figsize=(width_2col/3, height_2col/3))
    n_size = eeRCcadMO_angle_mean_cant_time_corr['count']
    cmap = plt.cm.get_cmap('jet', n_size.max())
    norm = mpl.colors.Normalize(vmin=n_size.min()-0.5, vmax=n_size.max()+0.5)    
    ax.errorbar(eeRCcadMO_angle_mean_cant_time_corr['Time (minutes)'], eeRCcadMO_angle_mean_cant_time_corr['mean'], eeRCcadMO_angle_mean_cant_time_corr['std'], ls='None', elinewidth=1, ecolor='grey')
    graph = ax.scatter(eeRCcadMO_angle_mean_cant_time_corr['Time (minutes)'], eeRCcadMO_angle_mean_cant_time_corr['mean'], marker='.', s=10, c=n_size, cmap=cmap, zorder=9)#, norm=matplotlib.colors.LogNorm()
    ax.set_xlim(eeRCcadMO_angle_mean_cant_time_corr['Time (minutes)'].min()-1, eeRCcadMO_angle_mean_cant_time_corr['Time (minutes)'].max()+1)
    ax.axhline(ee_genlog_popt_angle[3], c='g', ls='--')
    ax.axhline(ee_genlog_popt_angle[2], c='g', ls='--')
    ax.axhline(two_circ_interp_angle(ee_first_valid_area).mean(), c='g', ls='-', lw=0.5)
    ax.axhline(two_circ_interp_angle(mm_first_valid_area).mean(), c='r', ls='--', lw=0.5)
    ax.axhline(two_circ_interp_angle(nn_first_valid_area).mean(), c='b', ls='--', lw=0.5)
    ax.axhline(two_circ_interp_angle(eeRCcadMO_first_valid_area).mean(), c='y', ls='--', lw=0.5)
    ax.set_ylim(0, 220)
    ax.minorticks_on()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(graph, cax=cax)
    cbar_ticks = np.unique([int(x) for x in cbar.get_ticks()])
    cax.clear()
    cbar = plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax, ticks=cbar_ticks)
    ax.draw
    fig.savefig(Path.joinpath(output_dir, r"eeRCcadMO_angle_mean_cant_time_corr.tif"), dpi=dpi_graph, bbox_inches='tight')
    fig.clf()
    
    
if np.any(eeRCcadMO_fit_contact_mean_cant_time_corr):
    last_multi_n = len((eeRCcadMO_fit_contact_mean_cant_time_corr['count'] > n_size_thresh_for_xlim).values) - 1 - np.argmax((eeRCcadMO_fit_contact_mean_cant_time_corr['count'] > n_size_thresh_for_xlim).values[::-1])
    first_multi_n = np.argmax((eeRCcadMO_fit_contact_mean_cant_time_corr['count'] > n_size_thresh_for_xlim).values)
    fig, ax = plt.subplots(figsize=(width_2col/3, height_2col/3))
    n_size = eeRCcadMO_fit_contact_mean_cant_time_corr['count']
    cmap = plt.cm.get_cmap('jet', n_size.max())
    norm = mpl.colors.Normalize(vmin=n_size.min()-0.5, vmax=n_size.max()+0.5)
    ax.errorbar(eeRCcadMO_fit_contact_mean_cant_time_corr['Time (minutes)'], eeRCcadMO_fit_contact_mean_cant_time_corr['mean'], eeRCcadMO_fit_contact_mean_cant_time_corr['std'], ls='None', elinewidth=1, ecolor='grey')
    graph = ax.scatter(eeRCcadMO_fit_contact_mean_cant_time_corr['Time (minutes)'], eeRCcadMO_fit_contact_mean_cant_time_corr['mean'], marker='.', s=10, c=n_size, cmap=cmap, zorder=9)#, norm=matplotlib.colors.LogNorm())
    ax.set_xlim(eeRCcadMO_fit_contact_mean_cant_time_corr['Time (minutes)'].min()-1, eeRCcadMO_fit_contact_mean_cant_time_corr['Time (minutes)'].max()+1)
    ax.axhline(ee_genlog_popt_contnd[3], c='g', ls='--')
    ax.axhline(ee_genlog_popt_contnd[2], c='g', ls='--')
    ax.axhline(two_circ_interp_contnd(ee_first_valid_area).mean(), c='g', ls='-', lw=0.5)
    ax.axhline(two_circ_interp_contnd(mm_first_valid_area).mean(), c='r', ls='--', lw=0.5)
    ax.axhline(two_circ_interp_contnd(nn_first_valid_area).mean(), c='b', ls='--', lw=0.5)
    ax.axhline(two_circ_interp_contnd(eeRCcadMO_first_valid_area).mean(), c='y', ls='--', lw=0.5)
    ax.set_ylim(0, 4.0)
    ax.minorticks_on()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(graph, cax=cax)
    cbar_ticks = np.unique([int(x) for x in cbar.get_ticks()])
    cax.clear()
    cbar = plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax, ticks=cbar_ticks)
    ax.draw
    fig.savefig(Path.joinpath(output_dir, r"eeRCcadMO_fit_contact_mean_cant_time_corr.tif"), dpi=dpi_graph, bbox_inches='tight')
    fig.clf()





###############################################################################

### Plot the contact angle vs. the contact diameter for each cell combination:

angle_theoretical = np.linspace(0, 90, 91)
#angle_experimental_ee =  ee_angle_mean['mean'] / 2.
#angle_fit_ee = ee_y_angle
cont_radius_theoretical = f_cont_radius(angle_theoretical)
cont_radius_theoretical2 = f(angle_theoretical, 1)[1][4]/2

area_cell_theoretical = f(angle_theoretical, 1)[1][3]
area_cell_initial_theor = f(angle_theoretical, 1)[0][3]
area_cell_theoretical_nd = area_cell_theoretical / area_cell_initial_theor

# Just checking that the functions for cont_radius theoretical and 
# cont_radius_theoretical2 agree and to what extend:
#decim = 12
#(np.round(cont_radius_theoretical2[1][-1]/2, decimals=decim) == np.round(cont_radius_theoretical, decimals=decim)).all()
# Answer: Yes. To 12 decimal points.

area_theoretical = f(angle_theoretical, 1)[1][3]

#cont_radius_experimental_ee =  ee_fit_contact_mean['mean'] / 2.
#cont_radius_nd_fit = 



### Ecto-ecto
fig, ax = plt.subplots(figsize=(width_1col,width_1col))
#ax.scatter(angle_theoretical, cont_radius_theoretical, marker='.')
ax.plot(angle_theoretical * 2, cont_radius_theoretical2 * 2, c='k', ls=':')
#ax.scatter(ee_y_angle/2., ee_y_contnd/2., marker='.', zorder=10)
#ax.scatter(ee_genlog_y_angle, ee_genlog_y_contnd, zorder=10, marker='.', s=5, c='k')
ax.plot(ee_genlog_y_angle, ee_genlog_y_contnd_anglefit, zorder=10, c='k', ls='-', lw=1)
ax.plot(ee_genlog_y_angle_contndfit, ee_genlog_y_contnd, zorder=10, c='k', ls='--', lw=1)
sns.scatterplot(y=ee_contact_diam_non_dim_df['Contact Surface Length (N.D.)'], x=ee_contact_diam_non_dim_df['Average Angle (deg)'], data=ee_contact_diam_non_dim_df, legend=False, marker='.', color='green', linewidth=0, alpha=0.3, ax=ax)#, hue='Source_File')
#sns.scatterplot(y=radius27px0deg_cell_sim_list_original_nd_df['Contact Surface Length (N.D.)']/2, x=radius27px0deg_cell_sim_list_original_nd_df['Average Angle (deg)']/2, data=radius27px0deg_cell_sim_list_original_nd_df, marker='.', color='r', legend=False, alpha=0.3, ax=ax)
ax.plot(radius27px0deg_cell_sim_list_original_nd_df['Average Angle (deg)'], radius27px0deg_cell_sim_list_original_nd_df['Contact Surface Length (N.D.)'], c='k', ls=':')
ax.axvline(growth_phase_start_ee_angle_val, ls='-', c='grey', lw=1)
ax.axhline(growth_phase_start_ee_contnd_val, ls='-', c='grey', lw=1)
ax.axvline(growth_phase_end_ee_angle_val, ls='-', c='grey', lw=1)
ax.axhline(growth_phase_end_ee_contnd_val, ls='-', c='grey', lw=1)
ax.set_xlim(-5, 220)
ax.set_ylim(-0.05, 4)
ax.axvline(0, ls='--', c='grey')
ax.axvline(180, ls='--', c='grey')
ax.axhline(0, ls='--', c='grey')
ax.axhline(cont_radius_theoretical2.max() * 2, ls='--', c='grey')
#ax.set_xlabel('Contact Angle (deg)')
ax.set_xlabel('')
ax.set_ylabel('')
#ax.tick_params(axis='both', labelsize=14)
fig.savefig(Path.joinpath(output_dir, r"ee_cont_len vs avg_ang.tif"), dpi=dpi_graph)
fig.clf()
### How well do the 2 logistical fits to the overall data fit?:
#plt.plot(ee_y_angle / ee_y_angle.max())
#plt.plot(ee_y_contnd / ee_y_contnd.max())



### Meso-meso
fig, ax = plt.subplots(figsize=(width_1col,width_1col))
#ax.scatter(angle_theoretical, cont_radius_theoretical, marker='.')
ax.plot(angle_theoretical * 2, cont_radius_theoretical2 * 2, c='k', ls=':')
#ax.scatter(mm_genlog_y_angle, mm_genlog_y_contnd, zorder=10, marker='.', s=5, c='k')
ax.plot(mm_genlog_y_angle, mm_genlog_y_contnd_anglefit, zorder=10, c='k', ls='-', lw=1)
ax.plot(mm_genlog_y_angle_contndfit, mm_genlog_y_contnd, zorder=10, c='k', ls='--', lw=1)
sns.scatterplot(y=mm_contact_diam_non_dim_df['Contact Surface Length (N.D.)'], x=mm_contact_diam_non_dim_df['Average Angle (deg)'], data=mm_contact_diam_non_dim_df, legend=False, marker='.', color='red', linewidth=0, alpha=0.3, ax=ax)#, hue='Source_File')
#sns.scatterplot(y=radius27px0deg_cell_sim_list_original_nd_df['Contact Surface Length (N.D.)']/2, x=radius27px0deg_cell_sim_list_original_nd_df['Average Angle (deg)']/2, data=radius27px0deg_cell_sim_list_original_nd_df, marker='.', color='r', legend=False, alpha=0.3, ax=ax)
ax.plot(radius27px0deg_cell_sim_list_original_nd_df['Average Angle (deg)'], radius27px0deg_cell_sim_list_original_nd_df['Contact Surface Length (N.D.)'], c='k', ls=':')
#ax.set_ylim(-0.05,2)
#ax.set_xlim(-5,95)
ax.axvline(growth_phase_start_mm_angle_val, ls='-', c='grey', lw=1)
ax.axhline(growth_phase_start_mm_contnd_val, ls='-', c='grey', lw=1)
ax.axvline(growth_phase_end_mm_angle_val, ls='-', c='grey', lw=1)
ax.axhline(growth_phase_end_mm_contnd_val, ls='-', c='grey', lw=1)
ax.set_xlim(-5, 220)
ax.set_ylim(-0.05, 4)
ax.axvline(0, ls='--', c='grey')
ax.axvline(180, ls='--', c='grey')
ax.axhline(0, ls='--', c='grey')
ax.axhline(cont_radius_theoretical2.max() * 2, ls='--', c='grey')
#ax.set_xlabel('Contact Angle (deg)')
ax.set_xlabel('')
ax.set_ylabel('')
fig.savefig(Path.joinpath(output_dir, r"mm_cont_len vs avg_ang.tif"), dpi=dpi_graph)
fig.clf()



### Ecto-meso
fig, ax = plt.subplots(figsize=(width_1col,width_1col))
#ax.scatter(angle_theoretical, cont_radius_theoretical, marker='.')
ax.plot(angle_theoretical * 2, cont_radius_theoretical2 * 2, c='k', ls=':')
#ax.scatter(ee_y_angle/2., ee_y_contnd/2., marker='.', zorder=10)
#ax.scatter(em_y_angle/2., em_y_contnd/2., zorder=10, marker='.', s=5, c='k')
#ax.plot(em_y_angle/2., em_y_contnd/2., zorder=10, c='k', ls='-', lw=1)
# ax.plot(em_genlog_y_angle, em_genlog_y_contnd_anglefit, zorder=10, c='k', ls='-', lw=1)
# ax.plot(em_genlog_y_angle_contndfit, em_genlog_y_contnd, zorder=10, c='k', ls='--', lw=1)
sns.scatterplot(y=em_contact_diam_non_dim_df['Contact Surface Length (N.D.)'], x=em_contact_diam_non_dim_df['Average Angle (deg)'], data=em_contact_diam_non_dim_df, legend=False, marker='.', color='orange', linewidth=0, alpha=0.3, ax=ax)#, hue='Source_File')
#sns.scatterplot(y=radius27px0deg_cell_sim_list_original_nd_df['Contact Surface Length (N.D.)']/2, x=radius27px0deg_cell_sim_list_original_nd_df['Average Angle (deg)']/2, data=radius27px0deg_cell_sim_list_original_nd_df, marker='.', color='r', legend=False, alpha=0.3, ax=ax)
ax.plot(radius27px0deg_cell_sim_list_original_nd_df['Average Angle (deg)'], radius27px0deg_cell_sim_list_original_nd_df['Contact Surface Length (N.D.)'], c='k', ls=':')
#ax.set_ylim(-0.05,2)
#ax.set_xlim(-5,95)
ax.set_xlim(-5, 220)
ax.set_ylim(-0.05, 4)
ax.axvline(0, ls='--', c='grey')
ax.axvline(180, ls='--', c='grey')
ax.axhline(0, ls='--', c='grey')
ax.axhline(cont_radius_theoretical2.max() * 2, ls='--', c='grey')
#ax.set_xlabel('Contact Angle (deg)')
ax.set_xlabel('')
ax.set_ylabel('')
fig.savefig(Path.joinpath(output_dir, r"em_cont_len vs avg_ang.tif"), dpi=dpi_graph)
fig.clf()



### Endo-endo
fig, ax = plt.subplots(figsize=(width_1col,width_1col))
#ax.scatter(angle_theoretical, cont_radius_theoretical, marker='.')
ax.plot(angle_theoretical * 2, cont_radius_theoretical2 * 2, c='k', ls=':')
#ax.scatter(ee_y_angle/2., ee_y_contnd/2., marker='.', zorder=10)
#ax.scatter(nn_y_angle, nn_y_contnd, zorder=10, marker='.', s=5, c='k')
ax.plot(nn_genlog_y_angle, nn_genlog_y_contnd_anglefit, zorder=10, c='k', ls='-', lw=1)
ax.plot(nn_genlog_y_angle_contndfit, nn_genlog_y_contnd, zorder=10, c='k', ls='--', lw=1)
sns.scatterplot(y=nn_contact_diam_non_dim_df['Contact Surface Length (N.D.)'], x=nn_contact_diam_non_dim_df['Average Angle (deg)'], data=nn_contact_diam_non_dim_df, legend=False, marker='.', color='blue', linewidth=0, alpha=0.3, ax=ax)#, hue='Source_File')
#sns.scatterplot(y=radius27px0deg_cell_sim_list_original_nd_df['Contact Surface Length (N.D.)']/2, x=radius27px0deg_cell_sim_list_original_nd_df['Average Angle (deg)']/2, data=radius27px0deg_cell_sim_list_original_nd_df, marker='.', color='r', legend=False, alpha=0.3, ax=ax)
ax.plot(radius27px0deg_cell_sim_list_original_nd_df['Average Angle (deg)'], radius27px0deg_cell_sim_list_original_nd_df['Contact Surface Length (N.D.)'], c='k', ls=':')
#ax.set_ylim(-0.05,2)
#ax.set_xlim(-5,95)
ax.axvline(growth_phase_start_nn_angle_val, ls='-', c='grey', lw=1)
ax.axhline(growth_phase_start_nn_contnd_val, ls='-', c='grey', lw=1)
ax.axvline(growth_phase_end_nn_angle_val, ls='-', c='grey', lw=1)
ax.axhline(growth_phase_end_nn_contnd_val, ls='-', c='grey', lw=1)
ax.set_xlim(-5, 220)
ax.set_ylim(-0.05, 4)
ax.axvline(0, ls='--', c='grey')
ax.axvline(180, ls='--', c='grey')
ax.axhline(0, ls='--', c='grey')
ax.axhline(cont_radius_theoretical2.max() * 2, ls='--', c='grey')
#ax.set_xlabel('Contact Angle (deg)')
ax.set_xlabel('')
ax.set_ylabel('')
fig.savefig(Path.joinpath(output_dir, r"nn_cont_len vs avg_ang.tif"), dpi=dpi_graph)
fig.clf()




### Ecto-ecto in dissociation buffer
fig, ax = plt.subplots(figsize=(width_1col,width_1col))
#ax.scatter(angle_theoretical, cont_radius_theoretical, marker='.')
ax.plot(angle_theoretical * 2, cont_radius_theoretical2 * 2, c='k', ls=':')
#ax.scatter(ee_y_angle/2., ee_y_contnd/2., marker='.', zorder=10)
#ax.scatter(ee_genlog_y_angle, ee_genlog_y_contnd, zorder=10, marker='.', s=5, c='k')
ax.plot(eedb_genlog_y_angle, eedb_genlog_y_contnd_anglefit, zorder=10, c='k', ls='-', lw=1)
ax.plot(eedb_genlog_y_angle_contndfit, eedb_genlog_y_contnd, zorder=10, c='k', ls='--', lw=1)
sns.scatterplot(y=eedb_contact_diam_non_dim_df['Contact Surface Length (N.D.)'], x=eedb_contact_diam_non_dim_df['Average Angle (deg)'], data=eedb_contact_diam_non_dim_df, legend=False, marker='.', color='lightgrey', linewidth=0, alpha=0.3, ax=ax)#, hue='Source_File')
#sns.scatterplot(y=radius27px0deg_cell_sim_list_original_nd_df['Contact Surface Length (N.D.)']/2, x=radius27px0deg_cell_sim_list_original_nd_df['Average Angle (deg)']/2, data=radius27px0deg_cell_sim_list_original_nd_df, marker='.', color='r', legend=False, alpha=0.3, ax=ax)
ax.plot(radius27px0deg_cell_sim_list_original_nd_df['Average Angle (deg)'], radius27px0deg_cell_sim_list_original_nd_df['Contact Surface Length (N.D.)'], c='k', ls=':')
#ax.axvline(growth_phase_start_ee_angle_val, ls='-', c='grey', lw=1)
#ax.axhline(growth_phase_start_ee_contnd_val, ls='-', c='grey', lw=1)
#ax.axvline(growth_phase_end_ee_angle_val, ls='-', c='grey', lw=1)
#ax.axhline(growth_phase_end_ee_contnd_val, ls='-', c='grey', lw=1)
ax.set_xlim(-5, 220)
ax.set_ylim(-0.05, 4)
ax.axvline(0, ls='--', c='grey')
ax.axvline(180, ls='--', c='grey')
ax.axhline(0, ls='--', c='grey')
ax.axhline(cont_radius_theoretical2.max() * 2, ls='--', c='grey')
#ax.set_xlabel('Contact Angle (deg)')
ax.set_xlabel('')
ax.set_ylabel('')
#ax.tick_params(axis='both', labelsize=14)
fig.savefig(Path.joinpath(output_dir, r"eedb_cont_len vs avg_ang.tif"), dpi=dpi_graph)
fig.clf()
### How well do the 2 logistical fits to the overall data fit?:
#plt.plot(ee_y_angle / ee_y_angle.max())
#plt.plot(ee_y_contnd / ee_y_contnd.max())




### Ecto-ecto in dissocation buffer fixed:
fig, ax = plt.subplots(figsize=(width_1col,width_1col))
#ax.scatter(angle_theoretical, cont_radius_theoretical, marker='.')
ax.plot(angle_theoretical * 2, cont_radius_theoretical2 * 2, c='k', ls=':', zorder=1)
#ax.scatter(ee_y_angle/2., ee_y_contnd/2., marker='.', zorder=10)
#ax.scatter(em_y_angle/2., em_y_contnd/2., zorder=10, marker='.', s=5, c='k')
#ax.plot(em_y_angle/2., em_y_contnd/2., zorder=10, c='k', ls='-', lw=1)
#ax.plot(eedb_fix_genlog_y_angle, eedbfix_genlog_y_contnd_anglefit, zorder=10, c='k', ls='-', lw=1)
#ax.plot(eedbfix_genlog_y_angle_contndfit, eedbfix_genlog_y_contnd, zorder=10, c='k', ls='--', lw=1)
sns.scatterplot(y=eedb_fix_contact_diam_non_dim_df['Contact Surface Length (N.D.)'], x=eedb_fix_contact_diam_non_dim_df['Average Angle (deg)'], data=eedb_fix_contact_diam_non_dim_df, legend=False, marker='.', facecolors='k', edgecolor='k', linewidth=0, alpha=0.5, ax=ax, zorder=10)#, hue='Source_File')
sns.scatterplot(y=eedb_contact_diam_non_dim_df['Contact Surface Length (N.D.)'], x=eedb_contact_diam_non_dim_df['Average Angle (deg)'], data=eedb_contact_diam_non_dim_df, legend=False, marker='.', facecolors='lightgrey', edgecolor='lightgrey', linewidth=1, alpha=0.3, ax=ax, zorder=9)#, hue='Source_File')
ax.plot(radius27px0deg_cell_sim_list_original_nd_df['Average Angle (deg)'], radius27px0deg_cell_sim_list_original_nd_df['Contact Surface Length (N.D.)'], c='k', ls=':', zorder=2)
#ax.set_ylim(-0.05,2)
#ax.set_xlim(-5,95)
ax.set_xlim(-5, 220)
ax.set_ylim(-0.05, 4)
ax.axvline(0, ls='--', c='grey')
ax.axvline(180, ls='--', c='grey')
ax.axhline(0, ls='--', c='grey')
ax.axhline(cont_radius_theoretical2.max() * 2, ls='--', c='grey')
#ax.set_xlabel('Contact Angle (deg)')
ax.set_xlabel('')
ax.set_ylabel('')
fig.savefig(Path.joinpath(output_dir, r"eedb_fix_cont_len vs avg_ang.tif"), dpi=dpi_graph)
fig.clf()




### eeCcadMO-eeCcadMO
fig, ax = plt.subplots(figsize=(width_1col,width_1col))
#ax.scatter(angle_theoretical, cont_radius_theoretical, marker='.')
ax.plot(angle_theoretical * 2, cont_radius_theoretical2 * 2, c='k', ls=':')
#ax.scatter(ee_y_angle/2., ee_y_contnd/2., marker='.', zorder=10)
#ax.scatter(ee_genlog_y_angle, ee_genlog_y_contnd, zorder=10, marker='.', s=5, c='k')
ax.plot(eeCcadMO_genlog_y_angle, eeCcadMO_genlog_y_contnd_anglefit, zorder=10, c='k', ls='-', lw=1)
ax.plot(eeCcadMO_genlog_y_angle_contndfit, eeCcadMO_genlog_y_contnd, zorder=10, c='k', ls='--', lw=1)
sns.scatterplot(y=eeCcadMO_contact_diam_non_dim_df['Contact Surface Length (N.D.)'], x=eeCcadMO_contact_diam_non_dim_df['Average Angle (deg)'], data=eeCcadMO_contact_diam_non_dim_df, legend=False, marker='.', color='cyan', linewidth=0, alpha=0.3, ax=ax)#, hue='Source_File')
#sns.scatterplot(y=radius27px0deg_cell_sim_list_original_nd_df['Contact Surface Length (N.D.)']/2, x=radius27px0deg_cell_sim_list_original_nd_df['Average Angle (deg)']/2, data=radius27px0deg_cell_sim_list_original_nd_df, marker='.', color='r', legend=False, alpha=0.3, ax=ax)
ax.plot(radius27px0deg_cell_sim_list_original_nd_df['Average Angle (deg)'], radius27px0deg_cell_sim_list_original_nd_df['Contact Surface Length (N.D.)'], c='k', ls=':')
#ax.axvline(growth_phase_start_ee_angle_val, ls='-', c='grey', lw=1)
#ax.axhline(growth_phase_start_ee_contnd_val, ls='-', c='grey', lw=1)
#ax.axvline(growth_phase_end_ee_angle_val, ls='-', c='grey', lw=1)
#ax.axhline(growth_phase_end_ee_contnd_val, ls='-', c='grey', lw=1)
ax.set_xlim(-5, 220)
ax.set_ylim(-0.05, 4)
ax.axvline(0, ls='--', c='grey')
ax.axvline(180, ls='--', c='grey')
ax.axhline(0, ls='--', c='grey')
ax.axhline(cont_radius_theoretical2.max() * 2, ls='--', c='grey')
#ax.set_xlabel('Contact Angle (deg)')
ax.set_xlabel('')
ax.set_ylabel('')
#ax.tick_params(axis='both', labelsize=14)
fig.savefig(Path.joinpath(output_dir, r"eeCcadMO_cont_len vs avg_ang.tif"), dpi=dpi_graph)
fig.clf()
### How well do the 2 logistical fits to the overall data fit?:
#plt.plot(ee_y_angle / ee_y_angle.max())
#plt.plot(ee_y_contnd / ee_y_contnd.max())




### eeRcadMO-eeRcadMO
fig, ax = plt.subplots(figsize=(width_1col,width_1col))
#ax.scatter(angle_theoretical, cont_radius_theoretical, marker='.')
ax.plot(angle_theoretical * 2, cont_radius_theoretical2 * 2, c='k', ls=':')
#ax.scatter(ee_y_angle/2., ee_y_contnd/2., marker='.', zorder=10)
#ax.scatter(ee_genlog_y_angle, ee_genlog_y_contnd, zorder=10, marker='.', s=5, c='k')
ax.plot(eeRcadMO_genlog_y_angle, eeRcadMO_genlog_y_contnd_anglefit, zorder=10, c='k', ls='-', lw=1)
ax.plot(eeRcadMO_genlog_y_angle_contndfit, eeRcadMO_genlog_y_contnd, zorder=10, c='k', ls='--', lw=1)
sns.scatterplot(y=eeRcadMO_contact_diam_non_dim_df['Contact Surface Length (N.D.)'], x=eeRcadMO_contact_diam_non_dim_df['Average Angle (deg)'], data=eeRcadMO_contact_diam_non_dim_df, legend=False, marker='.', color='magenta', linewidth=0, alpha=0.3, ax=ax)#, hue='Source_File')
#sns.scatterplot(y=radius27px0deg_cell_sim_list_original_nd_df['Contact Surface Length (N.D.)']/2, x=radius27px0deg_cell_sim_list_original_nd_df['Average Angle (deg)']/2, data=radius27px0deg_cell_sim_list_original_nd_df, marker='.', color='r', legend=False, alpha=0.3, ax=ax)
ax.plot(radius27px0deg_cell_sim_list_original_nd_df['Average Angle (deg)'], radius27px0deg_cell_sim_list_original_nd_df['Contact Surface Length (N.D.)'], c='k', ls=':')
#ax.axvline(growth_phase_start_ee_angle_val, ls='-', c='grey', lw=1)
#ax.axhline(growth_phase_start_ee_contnd_val, ls='-', c='grey', lw=1)
#ax.axvline(growth_phase_end_ee_angle_val, ls='-', c='grey', lw=1)
#ax.axhline(growth_phase_end_ee_contnd_val, ls='-', c='grey', lw=1)
ax.set_xlim(-5, 220)
ax.set_ylim(-0.05, 4)
ax.axvline(0, ls='--', c='grey')
ax.axvline(180, ls='--', c='grey')
ax.axhline(0, ls='--', c='grey')
ax.axhline(cont_radius_theoretical2.max() * 2, ls='--', c='grey')
#ax.set_xlabel('Contact Angle (deg)')
ax.set_xlabel('')
ax.set_ylabel('')
#ax.tick_params(axis='both', labelsize=14)
fig.savefig(Path.joinpath(output_dir, r"eeRcadMO_cont_len vs avg_ang.tif"), dpi=dpi_graph)
fig.clf()
### How well do the 2 logistical fits to the overall data fit?:
#plt.plot(ee_y_angle / ee_y_angle.max())
#plt.plot(ee_y_contnd / ee_y_contnd.max())




### eeRCcadMO-eeRCcadMO
fig, ax = plt.subplots(figsize=(width_1col,width_1col))
#ax.scatter(angle_theoretical, cont_radius_theoretical, marker='.')
ax.plot(angle_theoretical * 2, cont_radius_theoretical2 * 2, c='k', ls=':')
#ax.scatter(ee_y_angle/2., ee_y_contnd/2., marker='.', zorder=10)
#ax.scatter(ee_genlog_y_angle, ee_genlog_y_contnd, zorder=10, marker='.', s=5, c='k')
ax.plot(eeRCcadMO_genlog_y_angle, eeRCcadMO_genlog_y_contnd_anglefit, zorder=10, c='k', ls='-', lw=1)
ax.plot(eeRCcadMO_genlog_y_angle_contndfit, eeRCcadMO_genlog_y_contnd, zorder=10, c='k', ls='--', lw=1)
sns.scatterplot(y=eeRCcadMO_contact_diam_non_dim_df['Contact Surface Length (N.D.)'], x=eeRCcadMO_contact_diam_non_dim_df['Average Angle (deg)'], data=eeRCcadMO_contact_diam_non_dim_df, legend=False, marker='.', color='y', linewidth=0, alpha=0.3, ax=ax)#, hue='Source_File')
#sns.scatterplot(y=radius27px0deg_cell_sim_list_original_nd_df['Contact Surface Length (N.D.)']/2, x=radius27px0deg_cell_sim_list_original_nd_df['Average Angle (deg)']/2, data=radius27px0deg_cell_sim_list_original_nd_df, marker='.', color='r', legend=False, alpha=0.3, ax=ax)
ax.plot(radius27px0deg_cell_sim_list_original_nd_df['Average Angle (deg)'], radius27px0deg_cell_sim_list_original_nd_df['Contact Surface Length (N.D.)'], c='k', ls=':')
#ax.axvline(growth_phase_start_ee_angle_val, ls='-', c='grey', lw=1)
#ax.axhline(growth_phase_start_ee_contnd_val, ls='-', c='grey', lw=1)
#ax.axvline(growth_phase_end_ee_angle_val, ls='-', c='grey', lw=1)
#ax.axhline(growth_phase_end_ee_contnd_val, ls='-', c='grey', lw=1)
ax.set_xlim(-5, 220)
ax.set_ylim(-0.05, 4)
ax.axvline(0, ls='--', c='grey')
ax.axvline(180, ls='--', c='grey')
ax.axhline(0, ls='--', c='grey')
ax.axhline(cont_radius_theoretical2.max() * 2, ls='--', c='grey')
#ax.set_xlabel('Contact Angle (deg)')
ax.set_xlabel('')
ax.set_ylabel('')
#ax.tick_params(axis='both', labelsize=14)
fig.savefig(Path.joinpath(output_dir, r"eeRCcadMO_cont_len vs avg_ang.tif"), dpi=dpi_graph)
fig.clf()
### How well do the 2 logistical fits to the overall data fit?:
#plt.plot(ee_y_angle / ee_y_angle.max())
#plt.plot(ee_y_contnd / ee_y_contnd.max())




### Calculate the expected reduction in cortical tension for angles and contnd:
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

beta_star_div_beta_angle_ee, beta_star_div_beta_contnd_ee, closest_nondim_contact_radius_ee = expected_reduced_cortical_tension(ee_genlog_y_angle/2, ee_genlog_y_contnd/2)
beta_star_div_beta_angle_mm, beta_star_div_beta_contnd_mm, closest_nondim_contact_radius_mm = expected_reduced_cortical_tension(mm_genlog_y_angle/2, mm_genlog_y_contnd/2)
beta_star_div_beta_angle_em, beta_star_div_beta_contnd_em, closest_nondim_contact_radius_em = expected_reduced_cortical_tension(em_contact_diam_non_dim_df['Average Angle (deg)']/2, em_contact_diam_non_dim_df['Contact Surface Length (N.D.)']/2)
with np.errstate(invalid='ignore'):
    beta_star_div_beta_angle_nn, beta_star_div_beta_contnd_nn, closest_nondim_contact_radius_nn = expected_reduced_cortical_tension(nn_genlog_y_angle/2, nn_genlog_y_contnd/2)


### Plot the expected reduction in cortical tension for angles and contnd:
fig, ax = plt.subplots(figsize=(width_1col,width_1col))
ax.plot(x_fit, beta_star_div_beta_angle_ee, ls='-', c='g', alpha=0.5, zorder=10)
ax.plot(x_fit, beta_star_div_beta_contnd_ee, ls='--', c='g', alpha=0.5, zorder=9)
ax.plot(x_fit, beta_star_div_beta_angle_mm, ls='-', c='r', alpha=0.5, zorder=8)
ax.plot(x_fit, beta_star_div_beta_contnd_mm, ls='--', c='r', alpha=0.5, zorder=7)
#ax.plot(beta_star_div_beta_angle_em, ls='-', c='orange')
#ax.plot(beta_star_div_beta_contnd_em, ls='--', c='orange')
ax.plot(x_fit, beta_star_div_beta_angle_nn, ls='-', c='b', alpha=0.5, zorder=6)
ax.plot(x_fit, beta_star_div_beta_contnd_nn, ls='--', c='b', alpha=0.5, zorder=5)
ax.set_ylim(0, 1.0)
ax.set_ylabel(r'$\beta^*/\beta$')
ax.set_xlabel('Time (minutes)')
fig.savefig(Path.joinpath(output_dir, r"expected_cortical_reduction.tif"), 
            dpi=dpi_graph, bbox_inches='tight')
fig.clf()

### Is there a temperature dependence in the growth rates of ecto-ecto?
fig, ax = plt.subplots(figsize=(width_1col,height_1col))
sns.swarmplot(x='Temperature (degC)', y='Max Growth Rates', data=growth_temps_df_dropna, marker='.', color='k', ax=ax)
# ax.set_xlabel(r'Temperature $\degree$C')
ax.set_xlabel(r'')
ax.set_ylabel(r'')
plt.tight_layout()
fig.savefig(Path.joinpath(output_dir, r"ee_temp_dependence_of_growth.tif"), 
            dpi=dpi_graph, bbox_inches='tight')
fig.clf()

fig, ax = plt.subplots()
sns.swarmplot(x='Temperature (degC)', y='Max Growth Rates', data=growth_temps_df_dropna, marker='.', color='k', ax=ax)
ax.semilogy()
fig.savefig(Path.joinpath(output_dir, r"ee_temp_dependence_of_growth_v2.tif"), 
            dpi=dpi_graph, bbox_inches='tight')
fig.clf()

# Test the significance of temperature on growth rates:
growth_temps_grouped = growth_temps_df_dropna.groupby(['Temperature (degC)'])
growth_temps_grouped_values = [col for col_name, col in growth_temps_grouped['Max Growth Rates']]
#stats.f_oneway(*growth_temps_grouped_values,)
eeGrowthRates_kruskal = stats.kruskal(*growth_temps_grouped_values,)
table = [['Condition', 'Statistic', 'pvalue'],
         ['ecto-ecto', *eeGrowthRates_kruskal]]

with open(Path.joinpath(output_dir, r'ee_temp_dep_growth_kruskal.tsv'), 
          'w', newline='') as file_out:
            tsv_obj = csv.writer(file_out, delimiter='\t')
            tsv_obj.writerows(table)
###############################################################################

### Report the binned and whole curvature values:
# But first take only the true cell pairs:
is_cellpair = np.array([True, True, False, False, False,   # 1--5
                        False, False, True, True, False,   # 6--10
                        True, True, True, True, False,     # 11--15
                        False, False, False, False, False, # 16--20
                        False, False, False, False, False, # 21--25
                        False, False, False, False, False,  # 26--30
                        False, False, False, True, True,   # 31--35
                        False, True, False, False, True,    # 36--40
                        False, False, False, False, True,  # 41--45
                        False, True, True, False, True,    # 46--50
                        False, False, False, True, False,  # 51--55
                        True, False]) # 56--57
    

is_cellpair_filename_order = np.array([
        '2015-06-03 bgsub Xl wt ecto fdx wt ecto rdx crop 1.csv', #1
        '2015-06-03 bgsub xl wt ecto fdx wt ecto rdx crop 2.csv', 
        '2015-08-27 bgsub xl wt ecto fdx wt ecto rdx crop 1.csv', 
        '2015-08-27 bgsub xl wt ecto fdx wt ecto rdx crop 2.csv',
        '2015-08-27 bgsub xl wt ecto fdx wt ecto rdx crop 3 sm.csv',
        '2015-08-27 bgsub xl wt ecto fdx wt ecto rdx crop 4.csv', #6
        '2015-08-27 bgsub xl wt ecto fdx wt ecto rdx crop 5.csv',
        '2015-08-28 bgsub xl wt ecto fdx wt ecto rdx crop 1.csv',
        '2015-09-03 bgsub xl wt ecto fdx wt ecto rdx crop 1.csv',
        '2015-09-03 bgsub xl wt ecto fdx wt ecto rdx crop 2.csv',
        '2015-09-03 bgsub xl wt ecto fdx wt ecto rdx crop 3.csv', #11
        '2016-06-22 bgsub xl wt ecto-rdx wt ecto-fdx #3 crop1.csv',
        '2017-08-10 xl wt ecto-fdx wt ecto-rda 20.5C crop1.csv',
        '2017-08-10 xl wt ecto-fdx wt ecto-rda 20.5C crop2.csv',
        '2018-03-08 xl wt-ecto-fdx wt-ecto-rdx 20C crop1.csv',
        '2018-03-08 xl wt-ecto-fdx wt-ecto-rdx 20C crop2.csv', #16
        '2018-03-08 xl wt-ecto-fdx wt-ecto-rdx 20C crop3.csv',
        '2018-03-08 xl wt-ecto-fdx wt-ecto-rdx 20C crop4.csv',
        '2018-03-08 xl wt-ecto-fdx wt-ecto-rdx 20C crop5.2.csv',
        '2018-03-08 xl wt-ecto-fdx wt-ecto-rdx 20C crop7.csv',
        '2018-03-08 xl wt-ecto-fdx wt-ecto-rdx 20C crop8.csv', #21
        '2018-03-15 xl wt-ecto-fdx wt-ecto-rdx 20C #1 crop1.csv',
        '2018-03-15 xl wt-ecto-fdx wt-ecto-rdx 20C #1 crop10.csv',
        '2018-03-15 xl wt-ecto-fdx wt-ecto-rdx 20C #1 crop11.csv',
        '2018-03-15 xl wt-ecto-fdx wt-ecto-rdx 20C #1 crop2.csv',
        '2018-03-15 xl wt-ecto-fdx wt-ecto-rdx 20C #1 crop3.csv', #26
        '2018-03-15 xl wt-ecto-fdx wt-ecto-rdx 20C #1 crop4.csv',
        '2018-03-15 xl wt-ecto-fdx wt-ecto-rdx 20C #1 crop5.csv',
        '2018-03-15 xl wt-ecto-fdx wt-ecto-rdx 20C #1 crop6.csv',
        '2018-03-15 xl wt-ecto-fdx wt-ecto-rdx 20C #1 crop7.csv',
        '2018-03-15 xl wt-ecto-fdx wt-ecto-rdx 20C #1 crop8.csv', #31
        '2018-03-15 xl wt-ecto-fdx wt-ecto-rdx 20C #1 crop9.csv',
        '2018-03-15 xl wt-ecto-fdx wt-ecto-rdx 20C #2 crop1.csv',
        '2018-03-15 xl wt-ecto-fdx wt-ecto-rdx 20C #2 crop2.csv',
        '2018-03-15 xl wt-ecto-fdx wt-ecto-rdx 20C #2 crop3.csv',
        '2018-06-27 xl wt-ecto-fdx wt-ecto-rdx 19C crop1.csv', #36
        '2018-06-27 xl wt-ecto-fdx wt-ecto-rdx 19C crop2.csv',
        '2018-06-27 xl wt-ecto-fdx wt-ecto-rdx 19C crop5.csv',
        '2018-06-27 xl wt-ecto-fdx wt-ecto-rdx 19C crop6.csv',
        '2018-06-28 xl wt-ecto-fdx wt-ecto-rdx 19C crop1.1.csv',
        '2018-06-28 xl wt-ecto-fdx wt-ecto-rdx 19C crop2.csv', #41
        '2018-06-28 xl wt-ecto-fdx wt-ecto-rdx 19C crop3.csv',
        '2018-06-28 xl wt-ecto-fdx wt-ecto-rdx 19C crop4.csv',
        '2018-06-28 xl wt-ecto-fdx wt-ecto-rdx 19C crop5.csv',
        '2018-06-28 xl wt-ecto-fdx wt-ecto-rdx 19C crop6.csv',
        '2018-06-28 xl wt-ecto-fdx wt-ecto-rdx 19C crop7.csv', #46
        '2018-08-10 xl wt-ecto-fdx wt-ecto-rdx 21C crop1.csv',
        '2018-08-10 xl wt-ecto-fdx wt-ecto-rdx 21C crop2.csv',
        '2018-08-10 xl wt-ecto-fdx wt-ecto-rdx 21C crop3.csv',
        '2018-08-10 xl wt-ecto-fdx wt-ecto-rdx 21C crop4.csv',
        '2018-08-10 xl wt-ecto-fdx wt-ecto-rdx 21C crop5.csv', #51
        '2018-08-10 xl wt-ecto-fdx wt-ecto-rdx 21C crop6.csv',
        '2018-08-10 xl wt-ecto-fdx wt-ecto-rdx 21C crop7.csv',
        '2019-01-09_XL_WTectoFDX_WTectoRDX_#2_19C_dt1m crop1.csv',
        '2019-01-09_XL_WTectoFDX_WTectoRDX_#2_19C_dt1m crop3.csv',
        '2019-01-09_XL_WTectoFDX_WTectoRDX_#2_19C_dt1m crop4.csv', #56
        '2019-01-09_XL_WTectoFDX_WTectoRDX_#2_19C_dt1m crop5.csv'])

columns = {'Source_File':is_cellpair_filename_order, 'is_cellpair':is_cellpair}
is_cellpair_df = pd.DataFrame(data=columns)

# Now that we have a dataframe of what data conists of a cell pair, we can 
# select at the adhesion kinetics, areas and curvatures of just that data:
cellpair_filenames = is_cellpair_df['Source_File'].mask(~is_cellpair_df['is_cellpair']).dropna()
not_cellpair_filenames = is_cellpair_df['Source_File'].mask(is_cellpair_df['is_cellpair']).dropna()
ok_values = ee_fit_df_dropna_time_corr_cpu['Source_File'].isin(cellpair_filenames)
ee_df_dropna_time_corr_cpu_cellpair_only = ee_fit_df_dropna_time_corr_cpu.mask(~ok_values)
ee_df_dropna_time_corr_cpu_cellpair_only.dropna(axis=0, subset=['Source_File'], inplace=True)
ee_list_dropna_time_corr_cpu_cellpair_only = list(ee_df_dropna_time_corr_cpu_cellpair_only.groupby(['Source_File']))
ee_list_dropna_time_corr_cpu_cellpair_only = [x[1] for x in ee_list_dropna_time_corr_cpu_cellpair_only]

# While we're at it we may as well create a list of the first valid radii for
# ecto-ecto true cell pairs only:
ee_first_valid_area_cellpair_only = ee_first_valid_area * is_cellpair
columns = {'Source_File':is_cellpair_filename_order, 'ee_first_valid_area_cellpair_only':ee_first_valid_area_cellpair_only}
ee_first_valid_area_cellpair_only_df = pd.DataFrame(data=columns)

ee_first_valid_area_red_cellpair_only = ee_first_valid_area_red * is_cellpair
columns = {'Source_File':is_cellpair_filename_order, 'ee_first_valid_area_cellpair_only':ee_first_valid_area_cellpair_only}
ee_first_valid_area_red_cellpair_only_df = pd.DataFrame(data=columns)

ee_first_valid_area_grn_cellpair_only = ee_first_valid_area_grn * is_cellpair
columns = {'Source_File':is_cellpair_filename_order, 'ee_first_valid_area_cellpair_only':ee_first_valid_area_cellpair_only}
ee_first_valid_area_grn_cellpair_only_df = pd.DataFrame(data=columns)


ee_first_valid_radius_cellpair_only = ee_first_valid_radius * is_cellpair
columns = {'Source_File':is_cellpair_filename_order, 'ee_first_valid_radius_cellpair_only':ee_first_valid_radius_cellpair_only}
ee_first_valid_radius_cellpair_only_df = pd.DataFrame(data=columns)

ee_first_valid_radius_red_cellpair_only = ee_first_valid_radius_red * is_cellpair
columns = {'Source_File':is_cellpair_filename_order, 'ee_first_valid_radius_cellpair_only':ee_first_valid_radius_cellpair_only}
ee_first_valid_radius_red_cellpair_only_df = pd.DataFrame(data=columns)

ee_first_valid_radius_grn_cellpair_only = ee_first_valid_radius_grn * is_cellpair
columns = {'Source_File':is_cellpair_filename_order, 'ee_first_valid_radius_cellpair_only':ee_first_valid_radius_cellpair_only}
ee_first_valid_radius_grn_cellpair_only_df = pd.DataFrame(data=columns)

ee_first_valid_curvature_cellpair_only = (1/ee_first_valid_radius_cellpair_only_df['ee_first_valid_radius_cellpair_only']).replace([np.inf, -np.inf],np.nan)#.dropna()
ee_first_valid_radius_cellpair_only_df['first_valid_curvature'] = ee_first_valid_curvature_cellpair_only
ee_first_valid_radius_cellpair_only_df.dropna(inplace=True)

ee_first_valid_curvature_red_cellpair_only = (1/ee_first_valid_radius_red_cellpair_only_df['ee_first_valid_radius_cellpair_only']).replace([np.inf, -np.inf],np.nan)#.dropna()
ee_first_valid_radius_red_cellpair_only_df['first_valid_curvature'] = ee_first_valid_curvature_red_cellpair_only
ee_first_valid_radius_red_cellpair_only_df.dropna(inplace=True)

ee_first_valid_curvature_grn_cellpair_only = (1/ee_first_valid_radius_grn_cellpair_only_df['ee_first_valid_radius_cellpair_only']).replace([np.inf, -np.inf],np.nan)#.dropna()
ee_first_valid_radius_grn_cellpair_only_df['first_valid_curvature'] = ee_first_valid_curvature_grn_cellpair_only
ee_first_valid_radius_grn_cellpair_only_df.dropna(inplace=True)

# ee_first_valid_curvature_cellpair_only.rename('ee_first_valid_curvature_cellpair_only', inplace=True)
# ee_first_valid_curvature_red_cellpair_only.rename('ee_first_valid_curvature_cellpair_only', inplace=True)
# ee_first_valid_curvature_grn_cellpair_only.rename('ee_first_valid_curvature_cellpair_only', inplace=True)



# Take the time-average of the cell pair only cells:
# For the cellpairs alignment is only being done according to contnd individual fits:
ee_time_corr_avg_cellpair_only = ee_df_dropna_time_corr_cpu_cellpair_only.groupby(['Time (minutes)'])
ee_time_corr_avg_cellpair_only_cont_nd = ee_time_corr_avg_cellpair_only['Contact Surface Length (N.D.)'].describe().reset_index()
ee_time_corr_avg_cellpair_only_angle = ee_time_corr_avg_cellpair_only['Average Angle (deg)'].describe().reset_index()


# Fit logistic curve to mean angles:
# sig_angle is the weighting factors used when fitting the logistic regression:
sig_angle = 1./np.sqrt(ee_time_corr_avg_cellpair_only_angle['count'])**2
eecellpairs_popt_angle, eecellpairs_pcov_angle = curve_fit(general_logistic, ee_time_corr_avg_cellpair_only_angle['Time (minutes)'], ee_time_corr_avg_cellpair_only_angle['mean'], sigma=sig_angle, p0 = (*p0_angle, nu))#, bounds=bounds_genlog_angle)
eecellpairs_y_angle = general_logistic(x_fit, *eecellpairs_popt_angle)
# Fit logistic curve to mean non-dimensionalized diameters:
# sig_contnd is the weighting factors used when fitting the logistic regression:
sig_contnd = 1./np.sqrt(ee_time_corr_avg_cellpair_only_cont_nd['count'])**2
eecellpairs_popt_contnd, eecellpairs_pcov_contnd = curve_fit(general_logistic, ee_time_corr_avg_cellpair_only_cont_nd['Time (minutes)'], ee_time_corr_avg_cellpair_only_cont_nd['mean'], sigma=sig_contnd, p0 = (*p0_contnd, nu))#, bounds=bounds_genlog_contnd)
eecellpairs_y_contnd = general_logistic(x_fit, *eecellpairs_popt_contnd)


# Determine the growth phase for the cellpair general logistic fits:
eecellpairs_y_contnd_diff = np.diff(np.ma.masked_array(data=eecellpairs_y_contnd, mask=~np.isfinite(eecellpairs_y_contnd)))
### Allow one of: 10%, 5%, 2%, or 0.2% growth in the plateau regions (evenly
### distributed between the low and high plateaus) to define the growth phase
### of the logistic fits with varying strictness
percent90_high = np.argmin(abs(np.cumsum(eecellpairs_y_contnd_diff) - np.cumsum(eecellpairs_y_contnd_diff).max()*0.95))
percent90_low = np.argmin(abs(np.cumsum(eecellpairs_y_contnd_diff) - np.cumsum(eecellpairs_y_contnd_diff).max()*0.05))

growth_phase_end_eecellpair_contnd = x_fit[percent90_high]
growth_phase_end_eecellpair_contnd_val = eecellpairs_y_contnd[percent90_high]
growth_phase_start_eecellpair_contnd = x_fit[percent90_low]
growth_phase_start_eecellpair_contnd_val = eecellpairs_y_contnd[percent90_low]


# Length of growth phase:
# eecellpair_angle_growth_dur = growth_phase_end_eecellpair_angle - growth_phase_start_eecellpair_angle
# eecellpair_cont_growth_dur = growth_phase_end_eecellpair_cont - growth_phase_start_eecellpair_cont
eecellpair_contnd_growth_dur = growth_phase_end_eecellpair_contnd - growth_phase_start_eecellpair_contnd

# Magnitude of change during growth phase:
# eecellpair_angle_growth_mag = growth_phase_end_eecellpair_angle_val - growth_phase_start_eecellpair_angle_val
# eecellpair_cont_growth_mag = growth_phase_end_eecellpair_cont_val - growth_phase_start_eecellpair_cont_val
eecellpair_contnd_growth_mag = growth_phase_end_eecellpair_contnd_val - growth_phase_start_eecellpair_contnd_val

# Average rate of change during growth phase:
# eecellpair_growth_rate_lin_angle = eecellpair_angle_growth_mag / eecellpair_angle_growth_dur
# eecellpair_growth_rate_lin_cont = eecellpair_cont_growth_mag / eecellpair_cont_growth_dur
eecellpair_growth_rate_lin_contnd = eecellpair_contnd_growth_mag / eecellpair_contnd_growth_dur


# pull bin curvature data from ee_df_dropna_time_corr_cpu_cellpair_only 
# according to 'source_file'
eecp_low_plat_bool = ee_df_dropna_time_corr_cpu_cellpair_only['Time (minutes)'] < growth_phase_start_eecellpair_contnd
eecp_growth_bool = np.logical_and((ee_df_dropna_time_corr_cpu_cellpair_only['Time (minutes)'] >= growth_phase_start_eecellpair_contnd),
                                  (ee_df_dropna_time_corr_cpu_cellpair_only['Time (minutes)'] <= growth_phase_end_eecellpair_contnd))
eecp_high_plat_bool = (ee_df_dropna_time_corr_cpu_cellpair_only['Time (minutes)'] > growth_phase_end_eecellpair_contnd)





### Make a convenient variable with the column names of the curvature values:
red_curvature_value_headers = ['bin_0_curv_val_red', 'bin_1_curv_val_red', 'bin_2_curv_val_red', 
                               'bin_3_curv_val_red', 'bin_4_curv_val_red', 'bin_5_curv_val_red', 
                               'bin_6_curv_val_red', 'bin_7_curv_val_red', 'whole_cell_curv_val_red']

red_curvature_value_whole_header = ['whole_cell_curv_val_red']

grn_curvature_value_headers = ['bin_0_curv_val_grn', 'bin_1_curv_val_grn', 'bin_2_curv_val_grn', 
                               'bin_3_curv_val_grn', 'bin_4_curv_val_grn', 'bin_5_curv_val_grn', 
                               'bin_6_curv_val_grn', 'bin_7_curv_val_grn', 'whole_cell_curv_val_grn']

grn_curvature_value_whole_header = ['whole_cell_curv_val_grn']



red_curvature_residu_headers = ['bin_0_curv_residu_red', 'bin_1_curv_residu_red', 'bin_2_curv_residu_red', 
                                'bin_3_curv_residu_red', 'bin_4_curv_residu_red', 'bin_5_curv_residu_red', 
                                'bin_6_curv_residu_red', 'bin_7_curv_residu_red', 'whole_cell_curv_residu_red']

grn_curvature_residu_headers = ['bin_0_curv_residu_grn', 'bin_1_curv_residu_grn', 'bin_2_curv_residu_grn', 
                                'bin_3_curv_residu_grn', 'bin_4_curv_residu_grn', 'bin_5_curv_residu_grn', 
                                'bin_6_curv_residu_grn', 'bin_7_curv_residu_grn', 'whole_cell_curv_residu_grn']







bin_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'silver', 'purple'] # [bin0, bin1, bin2, ..., bin7, whole_noncontact]

qual_bin_colors = ['b', 'c', 'g', 'k', 'g', 'c', 'b', 'silver', 'purple'] # [bin0, bin1, bin2, ..., bin7, whole_noncontact]



### Try non-dimensionalizing the curvatures:
# There are some cellpair_filenames entries that are not found in ee_df_dropna_time_corr_cpu_cellpair_only
# because they didn't have good logistic data fits, so we will need to make these consistent:
fitted_cellpairs = cellpair_filenames.isin(ee_df_dropna_time_corr_cpu_cellpair_only['Source_File'])
ee_first_valid_radius_red_cellpair_only_df = ee_first_valid_radius_red_cellpair_only_df[fitted_cellpairs]
ee_first_valid_radius_grn_cellpair_only_df = ee_first_valid_radius_grn_cellpair_only_df[fitted_cellpairs]


# ee_df_dropna_time_corr_cpu_cellpair_only_curvnd
curvnd = list()
for name,group in ee_df_dropna_time_corr_cpu_cellpair_only.groupby('Source_File'):

    sf_row_red = ee_first_valid_radius_red_cellpair_only_df[ee_first_valid_radius_red_cellpair_only_df['Source_File'] == name]
    group[red_curvature_value_headers] /= float(sf_row_red['first_valid_curvature'])    

    sf_row_grn = ee_first_valid_radius_grn_cellpair_only_df[ee_first_valid_radius_grn_cellpair_only_df['Source_File'] == name]
    group[grn_curvature_value_headers] /= float(sf_row_grn['first_valid_curvature'])
    
    curvnd.append(group)

### You can comment out the below line to keep the curvatures in their dimensional units:
ee_df_dropna_time_corr_cpu_cellpair_only = pd.concat(curvnd)


# should also check if the whole_cell_curv_val_red and _grn fit less and less
# well to a circle as you go from low to grow to high phase of adhesion, b/c
# that would also serve as evidence that cell is becoming progressively flattened





eecp_low_plat_curv = ee_df_dropna_time_corr_cpu_cellpair_only[eecp_low_plat_bool]
eecp_growth_curv = ee_df_dropna_time_corr_cpu_cellpair_only[eecp_growth_bool]
eecp_high_plat_curv = ee_df_dropna_time_corr_cpu_cellpair_only[eecp_high_plat_bool]





    
bin_curvatures_eecp_low_red = eecp_low_plat_curv[red_curvature_value_headers[:-1] + ['Source_File', 'Experimental_Condition']]
bin_curvatures_eecp_grow_red = eecp_growth_curv[red_curvature_value_headers[:-1] + ['Source_File', 'Experimental_Condition']]
bin_curvatures_eecp_high_red = eecp_high_plat_curv[red_curvature_value_headers[:-1] + ['Source_File', 'Experimental_Condition']]

bin_curvatures_eecp_low_red = pd.melt(bin_curvatures_eecp_low_red, id_vars=['Source_File', 'Experimental_Condition'], value_vars=red_curvature_value_headers[:-1], var_name='bin_label', value_name='curvature')
bin_curvatures_eecp_grow_red = pd.melt(bin_curvatures_eecp_grow_red, id_vars=['Source_File', 'Experimental_Condition'], value_vars=red_curvature_value_headers[:-1], var_name='bin_label', value_name='curvature')
bin_curvatures_eecp_high_red = pd.melt(bin_curvatures_eecp_high_red, id_vars=['Source_File', 'Experimental_Condition'], value_vars=red_curvature_value_headers[:-1], var_name='bin_label', value_name='curvature')

bin_curvatures_eecp_low_red_list = [(name, group) for name, group in bin_curvatures_eecp_low_red.groupby('Source_File')]
bin_curvatures_eecp_grow_red_list = [(name, group) for name, group in bin_curvatures_eecp_grow_red.groupby('Source_File')]
bin_curvatures_eecp_high_red_list = [(name, group) for name, group in bin_curvatures_eecp_high_red.groupby('Source_File')]


bin_curvatures_eecp_low_grn = eecp_low_plat_curv[grn_curvature_value_headers[:-1] + ['Source_File', 'Experimental_Condition']]
bin_curvatures_eecp_grow_grn = eecp_growth_curv[grn_curvature_value_headers[:-1] + ['Source_File', 'Experimental_Condition']]
bin_curvatures_eecp_high_grn = eecp_high_plat_curv[grn_curvature_value_headers[:-1] + ['Source_File', 'Experimental_Condition']]

bin_curvatures_eecp_low_grn = pd.melt(bin_curvatures_eecp_low_grn, id_vars=['Source_File', 'Experimental_Condition'], value_vars=grn_curvature_value_headers[:-1], var_name='bin_label', value_name='curvature')
bin_curvatures_eecp_grow_grn = pd.melt(bin_curvatures_eecp_grow_grn, id_vars=['Source_File', 'Experimental_Condition'], value_vars=grn_curvature_value_headers[:-1], var_name='bin_label', value_name='curvature')
bin_curvatures_eecp_high_grn = pd.melt(bin_curvatures_eecp_high_grn, id_vars=['Source_File', 'Experimental_Condition'], value_vars=grn_curvature_value_headers[:-1], var_name='bin_label', value_name='curvature')

bin_curvatures_eecp_low_grn_list = [(name, group) for name, group in bin_curvatures_eecp_low_grn.groupby('Source_File')]
bin_curvatures_eecp_grow_grn_list = [(name, group) for name, group in bin_curvatures_eecp_grow_grn.groupby('Source_File')]
bin_curvatures_eecp_high_grn_list = [(name, group) for name, group in bin_curvatures_eecp_high_grn.groupby('Source_File')]



# Plot the violin plots for red cells:
plt.close('all') # Close all open figures so our PDF only contains the graphs from this loop:
for name, df in bin_curvatures_eecp_low_red_list:
    fig, ax = plt.subplots(figsize=(width_2col,height_2col))
    sns.violinplot('bin_label', 'curvature', data=df, color='springgreen', ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    # ax.set_ylim(0)
    # ax.axvline(0, c='lightgrey')
    ax.axhline(0, c='lightgrey')
    ax.set_title('%s'%df['Source_File'].iloc[0])
    plt.tight_layout()
    # plt.show()
multipdf(Path.joinpath(output_dir, r"octant_curvatures_eecp_low_red_distrib.pdf"))
plt.close('all')


plt.close('all') # Close all open figures so our PDF only contains the graphs from this loop:
for name, df in bin_curvatures_eecp_grow_red_list:
    fig, ax = plt.subplots(figsize=(width_2col,height_2col))
    sns.violinplot('bin_label', 'curvature', data=df, color='springgreen', ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    # ax.set_ylim(0)
    # ax.axvline(0, c='lightgrey')
    ax.axhline(0, c='lightgrey')
    ax.set_title('%s'%df['Source_File'].iloc[0])
    plt.tight_layout()
    # plt.show()
multipdf(Path.joinpath(output_dir, r"octant_curvatures_eecp_grow_red_distrib.pdf"))
plt.close('all')


plt.close('all') # Close all open figures so our PDF only contains the graphs from this loop:
for name, df in bin_curvatures_eecp_high_red_list:
    fig, ax = plt.subplots(figsize=(width_2col,height_2col))
    sns.violinplot('bin_label', 'curvature', data=df, color='springgreen', ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    # ax.set_ylim(0)
    # ax.axvline(0, c='lightgrey')
    ax.axhline(0, c='lightgrey')
    ax.set_title('%s'%df['Source_File'].iloc[0])
    plt.tight_layout()
    # plt.show()
multipdf(Path.joinpath(output_dir, r"octant_curvatures_eecp_high_red_distrib.pdf"))
plt.close('all')



# Plot the violin plots for green cells:
plt.close('all') # Close all open figures so our PDF only contains the graphs from this loop:
for name, df in bin_curvatures_eecp_low_grn_list:
    fig, ax = plt.subplots(figsize=(width_2col,height_2col))
    sns.violinplot('bin_label', 'curvature', data=df, color='springgreen', ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    # ax.set_ylim(0)
    # ax.axvline(0, c='lightgrey')
    ax.axhline(0, c='lightgrey')
    ax.set_title('%s'%df['Source_File'].iloc[0])
    plt.tight_layout()
    # plt.show()
multipdf(Path.joinpath(output_dir, r"octant_curvatures_eecp_low_grn_distrib.pdf"))
plt.close('all')


plt.close('all') # Close all open figures so our PDF only contains the graphs from this loop:
for name, df in bin_curvatures_eecp_grow_grn_list:
    fig, ax = plt.subplots(figsize=(width_2col,height_2col))
    sns.violinplot('bin_label', 'curvature', data=df, color='springgreen', ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    # ax.set_ylim(0)
    # ax.axvline(0, c='lightgrey')
    ax.axhline(0, c='lightgrey')
    ax.set_title('%s'%df['Source_File'].iloc[0])
    plt.tight_layout()
    # plt.show()
multipdf(Path.joinpath(output_dir, r"octant_curvatures_eecp_grow_grn_distrib.pdf"))
plt.close('all')


plt.close('all') # Close all open figures so our PDF only contains the graphs from this loop:
for name, df in bin_curvatures_eecp_high_grn_list:
    fig, ax = plt.subplots(figsize=(width_2col,height_2col))
    sns.violinplot('bin_label', 'curvature', data=df, color='springgreen', ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    # ax.set_ylim(0)
    # ax.axvline(0, c='lightgrey')
    ax.axhline(0, c='lightgrey')
    ax.set_title('%s'%df['Source_File'].iloc[0])
    plt.tight_layout()
    # plt.show()
multipdf(Path.joinpath(output_dir, r"octant_curvatures_eecp_high_grn_distrib.pdf"))
plt.close('all')



### Are any of the non-contact bins significantly different from one another?
### low plateau data
noncont_curv_stat_low_red_list = list()
for name, df in bin_curvatures_eecp_low_red_list:        
    try: # if all bins have only 1 data point then a warning will come up, so handle that:
        noncont_curv_stat_low_red_list.append((name, 'low plateau', 'red', 
                                               stats.f_oneway(*[group['curvature'].dropna() for label, group in df.groupby('bin_label')][:-1])))
    except RuntimeWarning:
        F_onewayResult = namedtuple('F_onewayResult', ('statistic', 'pvalue'))
        noncont_curv_stat_low_red_list.append((name, 'low plateau', 'red', 
                                               F_onewayResult(np.nan, np.nan)))
# count the number of datasets with significance < 0.05:
noncontcurv_num_signif_low_red = np.count_nonzero([f_result[-1].pvalue < 0.05 for f_result in noncont_curv_stat_low_red_list])
# count the number of datasets with significance < 0.01:
noncontcurv_num_very_signif_low_red = np.count_nonzero([f_result[-1].pvalue < 0.01 for f_result in noncont_curv_stat_low_red_list])
noncontcurv_num_total_low_red = len(noncont_curv_stat_low_red_list)


noncont_curv_stat_low_grn_list = list()
for name, df in bin_curvatures_eecp_low_grn_list:        
    try: # if all bins have only 1 data point then a warning will come up, so handle that:
        noncont_curv_stat_low_grn_list.append((name, 'low plateau', 'green', 
                                               stats.f_oneway(*[group['curvature'].dropna() for label, group in df.groupby('bin_label')][:-1])))
    except RuntimeWarning:
        F_onewayResult = namedtuple('F_onewayResult', ('statistic', 'pvalue'))
        noncont_curv_stat_low_grn_list.append((name, 'low plateau', 'green', 
                                               F_onewayResult(np.nan, np.nan)))
# count the number of datasets with significance < 0.05:
noncontcurv_num_signif_low_grn = np.count_nonzero([f_result[-1].pvalue < 0.05 for f_result in noncont_curv_stat_low_grn_list])
# count the number of datasets with significance < 0.01:
noncontcurv_num_very_signif_low_grn = np.count_nonzero([f_result[-1].pvalue < 0.01 for f_result in noncont_curv_stat_low_grn_list])
noncontcurv_num_total_low_grn = len(noncont_curv_stat_low_grn_list)


### growth phase data
noncont_curv_stat_grow_red_list = list()
for name, df in bin_curvatures_eecp_grow_red_list:        
    try: # if all bins have only 1 data point then a warning will come up, so handle that:
        noncont_curv_stat_grow_red_list.append((name, 'growth phase', 'red', 
                                                stats.f_oneway(*[group['curvature'].dropna() for label, group in df.groupby('bin_label')][:-1])))
    except RuntimeWarning:
        F_onewayResult = namedtuple('F_onewayResult', ('statistic', 'pvalue'))
        noncont_curv_stat_grow_red_list.append((name, 'growth phase', 'red', 
                                                F_onewayResult(np.nan, np.nan)))
# count the number of datasets with significance < 0.05:
noncontcurv_num_signif_grow_red = np.count_nonzero([f_result[-1].pvalue < 0.05 for f_result in noncont_curv_stat_grow_red_list])
# count the number of datasets with significance < 0.01:
noncontcurv_num_very_signif_grow_red = np.count_nonzero([f_result[-1].pvalue < 0.01 for f_result in noncont_curv_stat_grow_red_list])
noncontcurv_num_total_grow_red = len(noncont_curv_stat_grow_red_list)


noncont_curv_stat_grow_grn_list = list()
for name, df in bin_curvatures_eecp_grow_grn_list:        
    try: # if all bins have only 1 data point then a warning will come up, so handle that:
        noncont_curv_stat_grow_grn_list.append((name, 'growth phase', 'green', 
                                                stats.f_oneway(*[group['curvature'].dropna() for label, group in df.groupby('bin_label')][:-1])))
    except RuntimeWarning:
        F_onewayResult = namedtuple('F_onewayResult', ('statistic', 'pvalue'))
        noncont_curv_stat_grow_grn_list.append((name, 'growth phase', 'green', 
                                                F_onewayResult(np.nan, np.nan)))
# count the number of datasets with significance < 0.05:
noncontcurv_num_signif_grow_grn = np.count_nonzero([f_result[-1].pvalue < 0.05 for f_result in noncont_curv_stat_grow_grn_list])
# count the number of datasets with significance < 0.01:
noncontcurv_num_very_signif_grow_grn = np.count_nonzero([f_result[-1].pvalue < 0.01 for f_result in noncont_curv_stat_grow_grn_list])
noncontcurv_num_total_grow_grn = len(noncont_curv_stat_grow_grn_list)


### high plateau data
noncont_curv_stat_high_red_list = list()
for name, df in bin_curvatures_eecp_high_red_list:        
    try: # if all bins have only 1 data point then a warning will come up, so handle that:
        noncont_curv_stat_high_red_list.append((name, 'high plateau', 'red', 
                                                stats.f_oneway(*[group['curvature'].dropna() for label, group in df.groupby('bin_label')][:-1])))
    except RuntimeWarning:
        F_onewayResult = namedtuple('F_onewayResult', ('statistic', 'pvalue'))
        noncont_curv_stat_high_red_list.append((name, 'high plateau', 'red', 
                                                F_onewayResult(np.nan, np.nan)))
# count the number of datasets with significance < 0.05:
noncontcurv_num_signif_high_red = np.count_nonzero([f_result[-1].pvalue < 0.05 for f_result in noncont_curv_stat_high_red_list])
# count the number of datasets with significance < 0.01:
noncontcurv_num_very_signif_high_red = np.count_nonzero([f_result[-1].pvalue < 0.01 for f_result in noncont_curv_stat_high_red_list])
noncontcurv_num_total_high_red = len(noncont_curv_stat_high_red_list)


noncont_curv_stat_high_grn_list = list()
for name, df in bin_curvatures_eecp_high_grn_list:        
    try: # if all bins have only 1 data point then a warning will come up, so handle that:
        noncont_curv_stat_high_grn_list.append((name, 'high plateau', 'green', 
                                                stats.f_oneway(*[group['curvature'].dropna() for label, group in df.groupby('bin_label')][:-1])))
    except RuntimeWarning:
        F_onewayResult = namedtuple('F_onewayResult', ('statistic', 'pvalue'))
        noncont_curv_stat_high_grn_list.append((name, 'high plateau', 'green', 
                                                F_onewayResult(np.nan, np.nan)))
# count the number of datasets with significance < 0.05:
noncontcurv_num_signif_high_grn = np.count_nonzero([f_result[-1].pvalue < 0.05 for f_result in noncont_curv_stat_high_grn_list])
# count the number of datasets with significance < 0.01:
noncontcurv_num_very_signif_high_grn = np.count_nonzero([f_result[-1].pvalue < 0.01 for f_result in noncont_curv_stat_high_grn_list])
noncontcurv_num_total_high_grn = len(noncont_curv_stat_high_grn_list)


# Save the statistics as a .csv:
stats_data_list = [noncont_curv_stat_low_red_list, noncont_curv_stat_low_grn_list, 
                   noncont_curv_stat_grow_red_list, noncont_curv_stat_grow_grn_list, 
                   noncont_curv_stat_high_red_list, noncont_curv_stat_high_grn_list]

df_list = list()
for ls in stats_data_list:
    data = {'Source_File': [f[0] for f in ls],
            'statistic': [f[-1].statistic for f in ls], 
            'pvalue': [f[-1].pvalue for f in ls], 
            'phase': [f[1] for f in ls], 
            'cell_color': [f[2] for f in ls]}
    df_list.append(pd.DataFrame(data))

noncont_curv_stats_df = pd.concat(df_list)
noncont_curv_stats_df['is_signif'] = noncont_curv_stats_df['pvalue'] < 0.05
noncont_curv_stats_df['is_very_signif'] = noncont_curv_stats_df['pvalue'] < 0.01

# output the stats data as a .csv:
noncont_curv_stats_df.to_csv(Path.joinpath(output_dir, 'octant_curvatures_eecp_stats.csv'), index=False)
# give me the proportions of timelapses with curvature bins that have at least
# one bin that differs significantly:
[(name, np.count_nonzero(df['is_signif']), len(df['is_signif']), np.count_nonzero(df['is_signif']) / len(df['is_signif'])) 
 for name,df in noncont_curv_stats_df.groupby('phase')]
[(name, np.count_nonzero(df['is_very_signif']), len(df['is_very_signif']), np.count_nonzero(df['is_very_signif']) / len(df['is_very_signif'])) 
 for name,df in noncont_curv_stats_df.groupby('phase')]
# Ans: about half have at least 1 bin that is significantly different among the low plateau data




# bin_curvatures_eecp_low_red
# Shows all data in bins:
# sns.catplot(x='bin_label', y='curvature', hue='Source_File', data=bin_curvatures_eecp_low_red, legend=False, marker='.')


# extract just bin means:
bin_curvs_low_red_mean = bin_curvatures_eecp_low_red.groupby(['Source_File', 'bin_label']).mean().reset_index()
bin_curvs_grow_red_mean = bin_curvatures_eecp_grow_red.groupby(['Source_File', 'bin_label']).mean().reset_index()
bin_curvs_high_red_mean = bin_curvatures_eecp_high_red.groupby(['Source_File', 'bin_label']).mean().reset_index()

bin_curvs_low_grn_mean = bin_curvatures_eecp_low_grn.groupby(['Source_File', 'bin_label']).mean().reset_index()
bin_curvs_grow_grn_mean = bin_curvatures_eecp_grow_grn.groupby(['Source_File', 'bin_label']).mean().reset_index()
bin_curvs_high_grn_mean = bin_curvatures_eecp_high_grn.groupby(['Source_File', 'bin_label']).mean().reset_index()


# Shows just means in bins:
# sns.catplot(x='bin_label', y='curvature', hue='Source_File', data=bin_curvs_low_red_mean, legend=False, marker='.')

curv_means_list = [bin_curvs_low_red_mean, bin_curvs_low_grn_mean, 
                   bin_curvs_grow_red_mean, bin_curvs_grow_grn_mean, 
                   bin_curvs_high_red_mean, bin_curvs_high_grn_mean]
curv_means_name_list = ['low plateau red means', 'low plateau green means', 
                        'growth phase red means', 'growth phase green means', 
                        'high plateau red means', 'high plateau green means']
curv_means_phase_list = ['low', 'low', 'grow', 'grow', 'high', 'high']
curv_means_cell_list = ['red', 'grn', 'red', 'grn', 'red', 'grn']

plt.close('all') # close all previously opened plots before creating a multipdf:
descending_curvature_means = list()
for i,curv_mean_df in enumerate(curv_means_list):
    desc_curvs = list()
    for name,df in curv_mean_df.groupby('Source_File'):
        df.insert(len(df.keys()), 'bin_color', bin_colors[:-1]) # add a column with the bin_label color
        # add a column that has bin_label as integers: bin_0_curv_val -> 0, bin_1_curv_val -> 1, ... etc.
        df.insert(len(df.keys()), 'bin_label_num', np.arange(0,8)) 
        df.insert(len(df.keys()), 'phase', [curv_means_phase_list[i]]*8) # add a column for phase
        df.insert(len(df.keys()), 'cell_color', [curv_means_cell_list[i]]*8) # add a column for if red or grn cell
        desc_curvs.append(df.sort_values('curvature', ascending=False))
        
    
    
    # Assign a number to the sort curvature values (1 is the largest (very curved) and 8 is the smallest 
    # (straight line) curvature):
    ranks = np.arange(1,9)
    [df.insert(len(df.keys()), 'rank', ranks) for df in desc_curvs]
    desc_curvs_df = pd.concat(desc_curvs)
    descending_curvature_means.append(desc_curvs_df)
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
    ax.set_title(curv_means_name_list[i])
    fig.tight_layout
multipdf(Path.joinpath(output_dir, r"octant_curvatures_eecp_mean_heatmaps.pdf"), tight=True, dpi=dpi_graph)



plt.close('all') # close all previously opened plots before creating a multipdf:
for i,curv_mean_df in enumerate(curv_means_list):
    desc_curvs = list()
    for name,df in curv_mean_df.groupby('Source_File'):
        df.insert(len(df.keys()), 'bin_color', bin_colors[:-1]) # add a column with the bin_label color
        # add a column that has bin_label as integers: bin_0_curv_val -> 0, bin_1_curv_val -> 1, ... etc.
        df.insert(len(df.keys()), 'bin_label_num', np.arange(0,8)) 
        df.insert(len(df.keys()), 'phase', [curv_means_phase_list[i]]*8) # add a column for phase
        df.insert(len(df.keys()), 'cell_color', [curv_means_cell_list[i]]*8) # add a column for if red or grn cell
        desc_curvs.append(df.sort_values('curvature', ascending=False))
    
    
    # Assign a number to the sort curvature values (1 is the largest (very curved) and 8 is the smallest 
    # (straight line) curvature):
    ranks = np.arange(1,9)
    [df.insert(len(df.keys()), 'rank', ranks) for df in desc_curvs]
    desc_curvs_df = pd.concat(desc_curvs)
    heatmap = desc_curvs_df.pivot('Source_File', 'rank', 'bin_label_num')
    
    # fix the colorbar so that the labels are centered over the colors:
    cbar_bounds = np.arange(0,len(qual_bin_colors[:-1])+1)-0.5
    cbar_norm = mpl.colors.BoundaryNorm(cbar_bounds, len(bin_colors[:-1]))
    cbar_ticks = np.arange(0,8)
    # cmap = mpl.colors.ListedColormap(bin_colors[:-1])
    cmap_qual = mpl.colors.ListedColormap(qual_bin_colors[:-1])
    
    
    fig, ax = plt.subplots(figsize=(width_2col, width_1pt5col))
    sns.heatmap(heatmap, cmap=qual_bin_colors[:-1], annot=True, cbar=False, ax=ax)#, cbar_kws={'ticks':bin_colors[:-1]})
    fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap_qual, norm=cbar_norm), ticks=cbar_ticks)
    ax.set_title(curv_means_name_list[i])
    fig.tight_layout
multipdf(Path.joinpath(output_dir, r"octant_curvatures_eecp_mean_heatmaps_qual.pdf"), tight=True, dpi=dpi_graph)
plt.close('all')




# extract just bin medians:
bin_curvs_low_red_median = bin_curvatures_eecp_low_red.groupby(['Source_File', 'bin_label']).median().reset_index()
bin_curvs_grow_red_median = bin_curvatures_eecp_grow_red.groupby(['Source_File', 'bin_label']).median().reset_index()
bin_curvs_high_red_median = bin_curvatures_eecp_high_red.groupby(['Source_File', 'bin_label']).median().reset_index()

bin_curvs_low_grn_median = bin_curvatures_eecp_low_grn.groupby(['Source_File', 'bin_label']).median().reset_index()
bin_curvs_grow_grn_median = bin_curvatures_eecp_grow_grn.groupby(['Source_File', 'bin_label']).median().reset_index()
bin_curvs_high_grn_median = bin_curvatures_eecp_high_grn.groupby(['Source_File', 'bin_label']).median().reset_index()


# Shows just means in bins:
# sns.catplot(x='bin_label', y='curvature', hue='Source_File', data=bin_curvs_low_red_mean, legend=False, marker='.')

curv_medians_list = [bin_curvs_low_red_median, bin_curvs_low_grn_median, 
                     bin_curvs_grow_red_median, bin_curvs_grow_grn_median, 
                     bin_curvs_high_red_median, bin_curvs_high_grn_median]
curv_medians_name_list = ['low plateau red medians', 'low plateau green medians', 
                        'growth phase red medians', 'growth phase green medians', 
                        'high plateau red medians', 'high plateau green medians']
curv_medians_phase_list = ['low', 'low', 'grow', 'grow', 'high', 'high']
curv_medians_cell_list = ['red', 'grn', 'red', 'grn', 'red', 'grn']


plt.close('all') # close all previously opened plots before creating a multipdf:
descending_curvature_medians = list()
for i,curv_median_df in enumerate(curv_medians_list):
    desc_curvs = list()
    for name,df in curv_median_df.groupby('Source_File'):
        df.insert(len(df.keys()), 'bin_color', bin_colors[:-1]) # add a column with the bin_label color
        # add a column that has bin_label as integers: bin_0_curv_val -> 0, bin_1_curv_val -> 1, ... etc.
        df.insert(len(df.keys()), 'bin_label_num', np.arange(0,8)) 
        df.insert(len(df.keys()), 'phase', [curv_medians_phase_list[i]]*8) # add a column for phase
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
multipdf(Path.joinpath(output_dir, r"octant_curvatures_eecp_median_heatmaps.pdf"), tight=True, dpi=dpi_graph)



plt.close('all') # close all previously opened plots before creating a multipdf:
for i,curv_median_df in enumerate(curv_medians_list):
    desc_curvs = list()
    for name,df in curv_median_df.groupby('Source_File'):
        df.insert(len(df.keys()), 'bin_color', bin_colors[:-1]) # add a column with the bin_label color
        # add a column that has bin_label as integers: bin_0_curv_val -> 0, bin_1_curv_val -> 1, ... etc.
        df.insert(len(df.keys()), 'bin_label_num', np.arange(0,8)) 
        df.insert(len(df.keys()), 'phase', [curv_medians_phase_list[i]]*8) # add a column for phase
        df.insert(len(df.keys()), 'cell_color', [curv_medians_cell_list[i]]*8) # add a column for if red or grn cell
        desc_curvs.append(df.sort_values('curvature', ascending=False))
    
    
    # Assign a number to the sort curvature values (1 is the largest (very curved) and 8 is the smallest 
    # (straight line) curvature):
    ranks = np.arange(1,9)
    [df.insert(len(df.keys()), 'rank', ranks) for df in desc_curvs]
    desc_curvs_df = pd.concat(desc_curvs)
    heatmap = desc_curvs_df.pivot('Source_File', 'rank', 'bin_label_num')
    
    # fix the colorbar so that the labels are centered over the colors:
    cbar_bounds = np.arange(0,len(qual_bin_colors[:-1])+1)-0.5
    cbar_norm = mpl.colors.BoundaryNorm(cbar_bounds, len(bin_colors[:-1]))
    cbar_ticks = np.arange(0,8)
    # cmap = mpl.colors.ListedColormap(bin_colors[:-1])
    cmap_qual = mpl.colors.ListedColormap(qual_bin_colors[:-1])
    
    
    fig, ax = plt.subplots(figsize=(width_2col, width_1pt5col))
    sns.heatmap(heatmap, cmap=qual_bin_colors[:-1], annot=True, cbar=False, ax=ax)#, cbar_kws={'ticks':bin_colors[:-1]})
    fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap_qual, norm=cbar_norm), ticks=cbar_ticks)
    ax.set_title(curv_medians_name_list[i])
    fig.tight_layout
multipdf(Path.joinpath(output_dir, r"octant_curvatures_eecp_median_heatmaps_qual.pdf"), tight=True, dpi=dpi_graph)
plt.close('all')



# Plot category plots of the mean and median curvatures for each bin:
curv_means_df = pd.concat(descending_curvature_means).reset_index()
cont = curv_means_df['bin_label_num'] == 7
cont_flanks = np.logical_or(curv_means_df['bin_label_num'] == 0, curv_means_df['bin_label_num'] == 6)
sides = np.logical_or(curv_means_df['bin_label_num'] == 1, curv_means_df['bin_label_num'] == 5)
rear_flanks = np.logical_or(curv_means_df['bin_label_num'] == 2, curv_means_df['bin_label_num'] == 4)
rear = curv_means_df['bin_label_num'] == 3

curv_means_df['bin_label_qual'] = [np.nan] * len(curv_means_df)
curv_means_df['bin_label_qual'][cont] = 'contact'
curv_means_df['bin_label_qual'][cont_flanks] = 'contact adjacent'
curv_means_df['bin_label_qual'][sides] = 'sides'
curv_means_df['bin_label_qual'][rear_flanks] = 'opposite adjacent'
curv_means_df['bin_label_qual'][rear] = 'opposite contact'



curv_means_list = [(name,df) for name,df in curv_means_df.groupby('phase')]
curv_means_df_list = [(name, df.groupby(['bin_label_num', 'rank'])['Source_File'].count().reset_index()) 
                      for name,df in curv_means_list]
curv_means_qual_list = [(name, df.groupby(['bin_label_qual', 'rank'])['Source_File'].count().reset_index()) 
                        for name,df in curv_means_list]


hm_grow_means, hm_high_means, hm_low_means = [(name, df.pivot('bin_label_num', 'rank', 'Source_File')) 
                                              for name,df in curv_means_df_list]

hm_low_means[1][~np.isfinite(hm_low_means[1])] = 0
hm_grow_means[1][~np.isfinite(hm_grow_means[1])] = 0
hm_high_means[1][~np.isfinite(hm_high_means[1])] = 0


plt.close('all')
fig, ax = plt.subplots(figsize=(width_1col, width_1col))
sns.heatmap(hm_low_means[1], cmap='Greys', annot=False, ax=ax, cbar_kws={'format':'%d'})

fig, ax = plt.subplots(figsize=(width_1col, width_1col))
sns.heatmap(hm_grow_means[1], cmap='Greys', annot=False, ax=ax, cbar_kws={'format':'%d'})

fig, ax = plt.subplots(figsize=(width_1col, width_1col))
sns.heatmap(hm_high_means[1], cmap='Greys', annot=False, ax=ax, cbar_kws={'format':'%d'})
multipdf(Path.joinpath(output_dir, r"octant_curvatures_eecp_mean_heatmaps_counts.pdf"), tight=True, dpi=dpi_graph)
plt.close('all')


hm_grow_means_qual, hm_high_means_qual, hm_low_means_qual = [(name, df.pivot('bin_label_qual', 'rank', 'Source_File')) 
                                                             for name,df in curv_means_qual_list]

hm_low_means_qual[1][~np.isfinite(hm_low_means_qual[1])] = 0
hm_grow_means_qual[1][~np.isfinite(hm_grow_means_qual[1])] = 0
hm_high_means_qual[1][~np.isfinite(hm_high_means_qual[1])] = 0

# the contact adjacent, sides, and opposite adjacent regions need to have their 
# counts divided by 2 since they are composed of 2 bins:
hm_grow_means_qual[1].loc['contact adjacent'] /= 2
hm_grow_means_qual[1].loc['sides'] /= 2
hm_grow_means_qual[1].loc['opposite adjacent'] /= 2

plt.close('all')
fig, ax = plt.subplots(figsize=(width_1col, width_1col))
sns.heatmap(hm_low_means_qual[1], cmap='Greys', annot=False, ax=ax, cbar_kws={'format':'%d'})

fig, ax = plt.subplots(figsize=(width_1col, width_1col))
sns.heatmap(hm_grow_means_qual[1], cmap='Greys', annot=False, ax=ax, cbar_kws={'format':'%d'})

fig, ax = plt.subplots(figsize=(width_1col, width_1col))
sns.heatmap(hm_high_means_qual[1], cmap='Greys', annot=False, ax=ax, cbar_kws={'format':'%d'})
multipdf(Path.joinpath(output_dir, r"octant_curvatures_eecp_mean_heatmaps_counts_qual.pdf"), tight=True, dpi=dpi_graph)
plt.close('all')





curv_medians_df = pd.concat(descending_curvature_medians)
cont = curv_medians_df['bin_label_num'] == 7
cont_flanks = np.logical_or(curv_medians_df['bin_label_num'] == 0, curv_medians_df['bin_label_num'] == 6)
sides = np.logical_or(curv_medians_df['bin_label_num'] == 1, curv_medians_df['bin_label_num'] == 5)
rear_flanks = np.logical_or(curv_medians_df['bin_label_num'] == 2, curv_medians_df['bin_label_num'] == 4)
rear = curv_medians_df['bin_label_num'] == 3

curv_medians_df['bin_label_qual'] = [np.nan] * len(curv_means_df)
curv_medians_df['bin_label_qual'][cont] = 'contact'
curv_medians_df['bin_label_qual'][cont_flanks] = 'contact adjacent'
curv_medians_df['bin_label_qual'][sides] = 'sides'
curv_medians_df['bin_label_qual'][rear_flanks] = 'opposite adjacent'
curv_medians_df['bin_label_qual'][rear] = 'opposite contact'




curv_medians_list = [(name,df) for name,df in curv_medians_df.groupby('phase')]
curv_medians_df_list = [(name, df.groupby(['bin_label_num', 'rank'])['Source_File'].count().reset_index()) 
                        for name,df in curv_medians_list]
curv_medians_qual_list = [(name, df.groupby(['bin_label_qual', 'rank'])['Source_File'].count().reset_index()) 
                          for name,df in curv_medians_list]
# cont_adj = [(name,df['bin_label_qual'] == 'contact adjacent') for name,df in curv_medians_qual_list]
# # for name,df in cont_adj:
# #     curv_medians_qual_list
# sides = [(name,df['bin_label_qual'] == 'sides') for name,df in curv_medians_qual_list]
# opp_adj = [(name,df['bin_label_qual'] == 'opposite adjacent') for name,df in curv_medians_qual_list]

hm_grow_medians, hm_high_medians, hm_low_medians = [(name, df.pivot('bin_label_num', 'rank', 'Source_File')) 
                                                    for name,df in curv_medians_df_list]

hm_low_medians[1][~np.isfinite(hm_low_medians[1])] = 0
hm_grow_medians[1][~np.isfinite(hm_grow_medians[1])] = 0
hm_high_medians[1][~np.isfinite(hm_high_medians[1])] = 0


plt.close('all')
fig, ax = plt.subplots(figsize=(width_1col, width_1col))
sns.heatmap(hm_low_medians[1], cmap='Greys', annot=False, ax=ax, cbar_kws={'format':'%d'})

fig, ax = plt.subplots(figsize=(width_1col, width_1col))
sns.heatmap(hm_grow_medians[1], cmap='Greys', annot=False, ax=ax, cbar_kws={'format':'%d'})

fig, ax = plt.subplots(figsize=(width_1col, width_1col))
sns.heatmap(hm_high_medians[1], cmap='Greys', annot=False, ax=ax, cbar_kws={'format':'%d'})
multipdf(Path.joinpath(output_dir, r"octant_curvatures_eecp_median_heatmaps_counts.pdf"), tight=True, dpi=dpi_graph)
plt.close('all')


hm_grow_medians_qual, hm_high_medians_qual, hm_low_medians_qual = [(name, df.pivot('bin_label_qual', 'rank', 'Source_File')) 
                                                                   for name,df in curv_medians_qual_list]

hm_low_medians_qual[1][~np.isfinite(hm_low_medians_qual[1])] = 0
hm_grow_medians_qual[1][~np.isfinite(hm_grow_medians_qual[1])] = 0
hm_high_medians_qual[1][~np.isfinite(hm_high_medians_qual[1])] = 0

# the contact adjacent, sides, and opposite adjacent regions need to have their 
# counts divided by 2 since they are composed of 2 bins:
hm_grow_medians_qual[1].loc['contact adjacent'] /= 2
hm_grow_medians_qual[1].loc['sides'] /= 2
hm_grow_medians_qual[1].loc['opposite adjacent'] /= 2

plt.close('all')
fig, ax = plt.subplots(figsize=(width_1col, width_1col))
sns.heatmap(hm_low_medians_qual[1], cmap='Greys', annot=False, ax=ax, cbar_kws={'format':'%d'})

fig, ax = plt.subplots(figsize=(width_1col, width_1col))
sns.heatmap(hm_grow_medians_qual[1], cmap='Greys', annot=False, ax=ax, cbar_kws={'format':'%d'})

fig, ax = plt.subplots(figsize=(width_1col, width_1col))
sns.heatmap(hm_high_medians_qual[1], cmap='Greys', annot=False, ax=ax, cbar_kws={'format':'%d'})
multipdf(Path.joinpath(output_dir, r"octant_curvatures_eecp_median_heatmaps_counts_qual.pdf"), tight=True, dpi=dpi_graph)
plt.close('all')



fig, ax = plt.subplots(figsize=(width_2col,width_1col))
sns.violinplot(x='bin_label_num', y='rank', hue='phase', data=curv_medians_df, ax=ax)
ax.set_yticks([0,2,4,6,8])
ax.set_ylabel('curvature rank')
ax.set_xlabel('bin number')
ax.legend(loc='lower center', ncol=3)
plt.savefig(Path.joinpath(output_dir, r"octant_curv_means_eecp_rel_distrib.tif"), dpi=dpi_graph)


fig, ax = plt.subplots(figsize=(width_1pt5col,height_1col))
sns.violinplot(x='bin_label_num', y='rank', hue='phase', data=curv_medians_df, ax=ax, #)
               linewidth=1, inner='box', cut=0)
ax.set_yticks([0,2,4,6,8])
ax.set_ylabel('curvature rank')
ax.set_xlabel('bin number')
ax.legend(list(ax.legend_.get_patches()), ['low', 'growth', 'high'], loc='lower center', bbox_to_anchor=(0.5, -0.03), borderpad=0.03, ncol=3)
# ax.legend([],[],frameon=False)
plt.savefig(Path.joinpath(output_dir, r"octant_curv_medians_eecp_rel_distrib.tif"), 
            dpi=dpi_graph, bbox_inches='tight')


fig, ax = plt.subplots(figsize=(width_2col,width_1col))
sns.violinplot(x='bin_label_num', y='curvature', hue='phase', data=curv_medians_df, ax=ax)
# ax.set_yticks([0,2,4,6,8])
ax.set_ylabel('curvature (N.D.)')
ax.set_xlabel('bin number')
ax.legend(loc='lower center', ncol=3)
plt.savefig(Path.joinpath(output_dir, r"octant_curv_medians_eecp_distrib.tif"), dpi=dpi_graph)








eecp_low_plat_curv['phase'] = ['low'] * len(eecp_low_plat_curv)
eecp_growth_curv['phase'] = ['grow'] * len(eecp_growth_curv)
eecp_high_plat_curv['phase'] = ['high'] * len(eecp_high_plat_curv)
eecp_df = pd.concat([eecp_low_plat_curv, eecp_growth_curv, eecp_high_plat_curv])
eecp_df_long = pd.melt(eecp_df, id_vars=['Source_File', 'Experimental_Condition', 'phase', ], value_vars=red_curvature_value_headers+grn_curvature_value_headers, var_name='bin_label', value_name='curvature')
eecp_curv_residu = pd.melt(eecp_df, id_vars=['Source_File', 'Experimental_Condition', 'phase', ], value_vars=red_curvature_residu_headers+grn_curvature_residu_headers, var_name='bin_label', value_name='curvature residues')
# eecp_df_long['cell_color']
bin_label_num_vals = [0,1,2,3,4,5,6,7,8] # 8 == 'all non-contact'


bin_label_num_list = list()
for i in bin_label_num_vals:
    red_loc = eecp_df_long['bin_label'] == red_curvature_value_headers[i]
    grn_loc = eecp_df_long['bin_label'] == grn_curvature_value_headers[i]
    bin_label_num_list.append(np.logical_or(red_loc, grn_loc) * i)
bin_label_num_data = np.add.reduce(bin_label_num_list)
eecp_df_long['bin_label_num'] = bin_label_num_data



bin_label_num_list = list()
for i in bin_label_num_vals:
    red_loc = eecp_curv_residu['bin_label'] == red_curvature_residu_headers[i]
    grn_loc = eecp_curv_residu['bin_label'] == grn_curvature_residu_headers[i]
    bin_label_num_list.append(np.logical_or(red_loc, grn_loc) * i)
bin_label_num_data = np.add.reduce(bin_label_num_list)
eecp_curv_residu['bin_label_num'] = bin_label_num_data


# could also add a column specifying cell color



fig, ax = plt.subplots(figsize=(width_2col,width_1col))
sns.violinplot(x='bin_label_num', y='curvature', hue='phase', data=eecp_df_long, ax=ax)
# ax.set_yticks([0,2,4,6,8])
ax.set_ylim(0,3)
ax.set_ylabel('curvature (N.D.)')
ax.set_xlabel('bin number')
ax.legend(loc='upper center', ncol=3)
plt.savefig(Path.joinpath(output_dir, r"octant_curv_eecp_distrib.tif"), dpi=dpi_graph)



fig, ax = plt.subplots(figsize=(width_2col,width_1col))
# sns.violinplot(x='bin_label_num', y='curvature residues', hue='phase', data=eecp_curv_residu, ax=ax)
# sns.stripplot(x='bin_label_num', y='curvature residues', hue='phase', data=eecp_curv_residu, 
#               dodge=True, marker='.', ax=ax)
sns.boxplot(x='bin_label_num', y='curvature residues', hue='phase', data=eecp_curv_residu, 
              dodge=True, ax=ax)

# ax.set_yticks([0,2,4,6,8])
# ax.set_ylim(0,3)
# ax.set_xlim(0,7.5)
ax.semilogy()
ax.set_ylabel('curvature residues')
ax.set_xlabel('bin number')
ax.legend(loc='upper center', ncol=3)
plt.savefig(Path.joinpath(output_dir, r"octant_curv_residu_eecp_distrib.tif"), dpi=dpi_graph)


fig, ax = plt.subplots(figsize=(width_2col,width_1col))
# all data except overall non-cont data and contact data:
data = eecp_curv_residu[eecp_curv_residu['bin_label_num'].isin(bin_label_num_vals[:-2])] 
sns.violinplot(x='bin_label_num', y='curvature residues', hue='phase', data=data, ax=ax)
# ax.set_yticks([0,2,4,6,8])
# ax.set_ylim(0,3)
# ax.semilogy()
# ax.set_ylabel('curvature residues')
ax.set_xlabel('bin number')
ax.legend(loc='upper center', ncol=3)
plt.savefig(Path.joinpath(output_dir, r"octant_curv_residu_eecp_distrib_2.tif"), dpi=dpi_graph)

fig, ax = plt.subplots(figsize=(width_2col,width_1col))
# all data except overall non-cont data and contact data:
data = eecp_curv_residu[eecp_curv_residu['bin_label_num'].isin(bin_label_num_vals[:-2])] 
sns.violinplot(x='bin_label_num', y='curvature residues', hue='phase', data=data, ax=ax)
# ax.set_yticks([0,2,4,6,8])
# ax.set_ylim(0,5)
# ax.semilogy()
# ax.set_ylabel('curvature residues')
ax.set_xlabel('bin number')
ax.legend(loc='upper center', ncol=3)
plt.savefig(Path.joinpath(output_dir, r"octant_curv_residu_eecp_distrib_2_zoom.tif"), dpi=dpi_graph)


fig, ax = plt.subplots(figsize=(width_2col,width_1col))
data = eecp_curv_residu[eecp_curv_residu['bin_label_num']==8] # just the overall non-cont data
sns.violinplot(x='bin_label_num', y='curvature residues', hue='phase', data=data, ax=ax)
# ax.set_yticks([0,2,4,6,8])
# ax.set_ylim(0,3)
# ax.semilogy()
# ax.set_ylabel('curvature residues')
ax.set_xlabel('bin number')
ax.legend(loc='upper center', ncol=3)
plt.savefig(Path.joinpath(output_dir, r"octant_curv_residu_eecp_distrib_3.tif"), dpi=dpi_graph)





# # begin temp:
# bin_curvs_low_red_median = bin_curvatures_eecp_low_red.groupby(['Source_File', 'bin_label']).median().reset_index()
# bin_curvs_grow_red_median = bin_curvatures_eecp_grow_red.groupby(['Source_File', 'bin_label']).median().reset_index()
# bin_curvs_high_red_median = bin_curvatures_eecp_high_red.groupby(['Source_File', 'bin_label']).median().reset_index()

# bin_curvs_low_grn_median = bin_curvatures_eecp_low_grn.groupby(['Source_File', 'bin_label']).median().reset_index()
# bin_curvs_grow_grn_median = bin_curvatures_eecp_grow_grn.groupby(['Source_File', 'bin_label']).median().reset_index()
# bin_curvs_high_grn_median = bin_curvatures_eecp_high_grn.groupby(['Source_File', 'bin_label']).median().reset_index()
# # end temp.

### red cell lineplots showing mean curvatures for each adhesion:
plt.close('all')
for name,df in bin_curvatures_eecp_low_red.groupby('Source_File'):
    fig, ax = plt.subplots(figsize=(width_1pt5col, width_1pt5col))
    sns.lineplot(x='bin_label', y='curvature', data=df, marker='o', ax=ax)
    ax.set_xticklabels(red_curvature_value_headers, rotation=45, ha='right')
    # fig.tight_layout
    # plt.show()
multipdf(Path.joinpath(output_dir, r"octant_curvatures_eecp_low_red_lineplots.pdf"), tight=True, dpi=dpi_graph)

plt.close('all')
for name,df in bin_curvatures_eecp_grow_red.groupby('Source_File'):
    fig, ax = plt.subplots(figsize=(width_1pt5col, width_1pt5col))
    sns.lineplot(x='bin_label', y='curvature', data=df, marker='o', ax=ax)
    ax.set_xticklabels(red_curvature_value_headers, rotation=45, ha='right')
    # fig.tight_layout
    # plt.show()
multipdf(Path.joinpath(output_dir, r"octant_curvatures_eecp_grow_red_lineplots.pdf"), tight=True, dpi=dpi_graph)

plt.close('all')
for name,df in bin_curvatures_eecp_high_red.groupby('Source_File'):
    fig, ax = plt.subplots(figsize=(width_1pt5col, width_1pt5col))
    sns.lineplot(x='bin_label', y='curvature', data=df, marker='o', ax=ax)
    ax.set_xticklabels(red_curvature_value_headers, rotation=45, ha='right')
    # fig.tight_layout
    # plt.show()
multipdf(Path.joinpath(output_dir, r"octant_curvatures_eecp_high_red_lineplots.pdf"), tight=True, dpi=dpi_graph)



### green cell lineplots showing mean curvatures for each adhesion:
plt.close('all')
for name,df in bin_curvatures_eecp_low_grn.groupby('Source_File'):
    fig, ax = plt.subplots(figsize=(width_1pt5col, width_1pt5col))
    sns.lineplot(x='bin_label', y='curvature', data=df, marker='o', ax=ax)
    ax.set_xticklabels(grn_curvature_value_headers, rotation=45, ha='right')
    # fig.tight_layout
    # plt.show()
multipdf(Path.joinpath(output_dir, r"octant_curvatures_eecp_low_grn_lineplots.pdf"), tight=True, dpi=dpi_graph)

plt.close('all')
for name,df in bin_curvatures_eecp_grow_grn.groupby('Source_File'):
    fig, ax = plt.subplots(figsize=(width_1pt5col, width_1pt5col))
    sns.lineplot(x='bin_label', y='curvature', data=df, marker='o', ax=ax)
    ax.set_xticklabels(grn_curvature_value_headers, rotation=45, ha='right')
    # fig.tight_layout
    # plt.show()
multipdf(Path.joinpath(output_dir, r"octant_curvatures_eecp_grow_grn_lineplots.pdf"), tight=True, dpi=dpi_graph)

plt.close('all')
for name,df in bin_curvatures_eecp_high_grn.groupby('Source_File'):
    fig, ax = plt.subplots(figsize=(width_1pt5col, width_1pt5col))
    sns.lineplot(x='bin_label', y='curvature', data=df, marker='o', ax=ax)
    ax.set_xticklabels(grn_curvature_value_headers, rotation=45, ha='right')
    # fig.tight_layout
    # plt.show()
multipdf(Path.joinpath(output_dir, r"octant_curvatures_eecp_high_grn_lineplots.pdf"), tight=True, dpi=dpi_graph)






# fig, ax = plt.subplots(figsize=(width_2col,width_1col))
# sns.stripplot(x='bin_label_num', y='rank', hue='phase', data=curv_medians_df, dodge=True, jitter=True, alpha=0.25, ax=ax)
# ax.set_ylabel('curvature rank')
# ax.set_xlabel('bin number')


# # Are the means of all the bin curvatures the same?
# curvature_stat_list = list()
# for name, df in bin_curvatures_ee1xMBS_list:
#     curvature_stat_list.append(stats.f_oneway(*[group['curvature'].dropna() for label, group in df.groupby('bin_label')]))
# # count the number of datasets with significance < 0.05:
# curv_num_signif = np.count_nonzero([f_result.pvalue < 0.05 for f_result in curvature_stat_list])
# # count the number of datasets with significance < 0.01:
# curv_num_very_signif = np.count_nonzero([f_result.pvalue < 0.01 for f_result in curvature_stat_list])




# # Are the regions flanking the contact site different from the other regions?
# flankcurv_stat_list = list()
# for name, df in bin_curvatures_ee1xMBS_list:
#     flanking_curv_data = df[np.logical_or(df['bin_label'] == 'bin_0_curv_val_red', 
#                                           df['bin_label'] == 'bin_6_curv_val_red')]
#     other_noncont_curv_data = df[np.logical_or.reduce((df['bin_label'] == 'bin_1_curv_val_red', 
#                                                        df['bin_label'] == 'bin_2_curv_val_red', 
#                                                        df['bin_label'] == 'bin_3_curv_val_red', 
#                                                        df['bin_label'] == 'bin_4_curv_val_red', 
#                                                        df['bin_label'] == 'bin_5_curv_val_red'))]
#     flankcurv_stat_list.append(stats.f_oneway(flanking_curv_data['curvature'].dropna(), other_noncont_curv_data['curvature'].dropna()))
# # count the number of datasets with significance < 0.05:
# flankcurv_num_signif = np.count_nonzero([f_result.pvalue < 0.05 for f_result in flankcurv_stat_list])
# # count the number of datasets with significance < 0.01:
# flankcurv_num_very_signif = np.count_nonzero([f_result.pvalue < 0.01 for f_result in flankcurv_stat_list])











# Plot the adhesion kinetics for just cell pairs:
fig, ax = plt.subplots(figsize=(width_1col,height_1col))
sns.scatterplot(x='Time (minutes)', y='Contact Surface Length (N.D.)', hue='Source_File', data=ee_df_dropna_time_corr_cpu_cellpair_only, marker='.', legend=False, ax=ax)
plt.savefig(Path.joinpath(output_dir, r"ee_cellpair_only_contnd_indiv.tif"), dpi=dpi_graph)
plt.clf()

last_multi_n = len((ee_time_corr_avg_cellpair_only_cont_nd['count'] > n_size_thresh_for_xlim).values) - 1 - np.argmax((ee_time_corr_avg_cellpair_only_cont_nd['count'] > n_size_thresh_for_xlim).values[::-1])
first_multi_n = np.argmax((ee_time_corr_avg_cellpair_only_cont_nd['count'] > n_size_thresh_for_xlim).values)
fig, ax = plt.subplots(figsize=(width_2col, height_2col))
cm = plt.cm.get_cmap('jet')
n_size = ee_time_corr_avg_cellpair_only_cont_nd['count']
ax.errorbar(x=ee_time_corr_avg_cellpair_only_cont_nd['Time (minutes)'], y=ee_time_corr_avg_cellpair_only_cont_nd['mean'], yerr=ee_time_corr_avg_cellpair_only_cont_nd['std'], marker='None', ls='None', elinewidth=1, ecolor='grey')
graph = ax.scatter(ee_time_corr_avg_cellpair_only_cont_nd['Time (minutes)'], ee_time_corr_avg_cellpair_only_cont_nd['mean'], marker='.', s=10, c=n_size, cmap=cm, norm=matplotlib.colors.LogNorm(), zorder=9)
ax.plot(x_fit, eecellpairs_y_contnd, c='k', zorder=10)
#ax.set_xlim(ee_time_corr_avg_cellpair_only_cont_nd['Time (minutes)'].iloc[first_multi_n]-1, ee_time_corr_avg_cellpair_only_cont_nd['Time (minutes)'].iloc[last_multi_n]+1)
ax.axvline(growth_phase_start_eecellpair_contnd)
ax.axvline(growth_phase_end_eecellpair_contnd)
#ax.axvline(growth_phase_start_ee_contnd_strict, c='k', ls='--')
#ax.axvline(growth_phase_end_ee_contnd_strict, c='k', ls='--')
#ax.axhline(eedb_fix_contact_diam_non_dim_df_list_long_dropna['Contact Surface Length (N.D.)'].mean(), c='k', ls='--', lw=0.5)
#ax.axhline(two_circ_interp_contnd(ee_first_valid_area).mean(), c='r', ls='--', lw=0.5)
plt.axhline(ee_popt_contnd[3], c='g', ls='--')
plt.axhline(ee_popt_contnd[2], c='g', ls='--')
ax.set_ylim(0, 4)
ax.set_xlim(ee_time_corr_avg_cellpair_only_cont_nd['Time (minutes)'].min()-1, ee_time_corr_avg_cellpair_only_cont_nd['Time (minutes)'].max()+1)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(graph, cax=cax)
ax.draw
plt.savefig(Path.joinpath(output_dir, r"ee_cellpair_only_contnd_time_avg.tif"), dpi=dpi_graph)
plt.clf()



last_multi_n = len((ee_time_corr_avg_cellpair_only_angle['count'] > n_size_thresh_for_xlim).values) - 1 - np.argmax((ee_time_corr_avg_cellpair_only_angle['count'] > n_size_thresh_for_xlim).values[::-1])
first_multi_n = np.argmax((ee_time_corr_avg_cellpair_only_angle['count'] > n_size_thresh_for_xlim).values)
fig, ax = plt.subplots(figsize=(width_2col, height_2col))
cm = plt.cm.get_cmap('jet')
n_size = ee_time_corr_avg_cellpair_only_angle['count']
ax.errorbar(x=ee_time_corr_avg_cellpair_only_angle['Time (minutes)'], y=ee_time_corr_avg_cellpair_only_angle['mean'], yerr=ee_time_corr_avg_cellpair_only_angle['std'], marker='None', ls='None', elinewidth=1, ecolor='grey')
graph = ax.scatter(ee_time_corr_avg_cellpair_only_angle['Time (minutes)'], ee_time_corr_avg_cellpair_only_angle['mean'], marker='.', s=10, c=n_size, cmap=cm, norm=matplotlib.colors.LogNorm(), zorder=9)
ax.plot(x_fit, eecellpairs_y_angle, c='k', zorder=10)
ax.set_xlim(ee_time_corr_avg_cellpair_only_angle['Time (minutes)'].iloc[first_multi_n]-1, ee_time_corr_avg_cellpair_only_angle['Time (minutes)'].iloc[last_multi_n]+1)
#ax.axvline(growth_phase_start_ee_contnd)
#ax.axvline(growth_phase_end_ee_contnd)
#ax.axvline(growth_phase_start_ee_contnd_strict, c='k', ls='--')
#ax.axvline(growth_phase_end_ee_contnd_strict, c='k', ls='--')
#ax.axhline(eedb_fix_contact_diam_non_dim_df_list_long_dropna['Contact Surface Length (N.D.)'].mean(), c='k', ls='--', lw=0.5)
#ax.axhline(two_circ_interp_contnd(ee_first_valid_area).mean(), c='r', ls='--', lw=0.5)
plt.axhline(ee_genlog_popt_angle[3], c='g', ls='--')
plt.axhline(ee_genlog_popt_angle[2], c='g', ls='--')
ax.set_ylim(0, 200)
# ax.set_xlim(ee_time_corr_avg_cellpair_only_angle['Time (minutes)'].min()-1, ee_time_corr_avg_cellpair_only_angle['Time (minutes)'].max()+1)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(graph, cax=cax)
ax.draw
plt.savefig(Path.joinpath(output_dir, r"ee_cellpair_only_angle_time_avg.tif"), dpi=dpi_graph)
plt.clf()





# Plot the diameter vs angle for just cell pairs:
fig, ax = plt.subplots(figsize=(width_1col,width_1col))
# sns.scatterplot(x=ee_fit_df_dropna_time_corr_cpu['Average Angle (deg)'], y=ee_fit_df_dropna_time_corr_cpu['Contact Surface Length (N.D.)'], marker='.', size=4, color='m', edgecolor=None, alpha=0.3, legend=False, ax=ax)
sns.scatterplot(x=ee_df_dropna_time_corr_cpu_cellpair_only['Average Angle (deg)'], y=ee_df_dropna_time_corr_cpu_cellpair_only['Contact Surface Length (N.D.)'], marker='.', size=5, color='g', edgecolor=None, alpha=0.3, legend=False, ax=ax)
ax.plot(radius27px0deg_cell_sim_list_original_nd_df['Average Angle (deg)'], radius27px0deg_cell_sim_list_original_nd_df['Contact Surface Length (N.D.)'], c='k', ls='--')
ax.plot(angle_theoretical * 2, cont_radius_theoretical2 * 2, c='k', ls=':')
#ax.scatter(eecellpairs_y_angle, eecellpairs_y_contnd, zorder=10, marker='.', s=5, c='k')
ax.plot(eecellpairs_y_angle, eecellpairs_y_contnd, zorder=10, c='k', ls='--', lw=1)
# ax.plot(ee_genlog_y_angle_contndfit, ee_genlog_y_contnd, zorder=10, c='grey', ls='-', lw=1)
ax.set_xlim(-5, 220)
ax.set_ylim(-0.05, 4)
ax.axvline(0, ls='--', c='grey')
ax.axvline(180, ls='--', c='grey')
ax.axhline(0, ls='--', c='grey')
ax.axhline(cont_radius_theoretical2.max() * 2, ls='--', c='grey')
ax.set_xlabel('')
ax.set_ylabel('')
plt.savefig(Path.joinpath(output_dir, r"ee_cellpair_only_cont_len vs avg_ang.tif"), dpi=dpi_graph)


plt.close('all')
# Also plot the diameter vs angle for individual cell pair timelapses:
for name,df in ee_df_dropna_time_corr_cpu_cellpair_only.groupby('Source_File'):
    fig, ax = plt.subplots(figsize=(width_2col,width_2col))
    ax.set_title(name)
    # sns.scatterplot(x=ee_fit_df_dropna_time_corr_cpu['Average Angle (deg)'], y=ee_fit_df_dropna_time_corr_cpu['Contact Surface Length (N.D.)'], marker='.', size=4, color='m', edgecolor=None, alpha=0.3, legend=False, ax=ax)
    sns.scatterplot(x='Average Angle (deg)', y='Contact Surface Length (N.D.)', hue='Time (minutes)', 
                    palette='jet', data=df, marker='.', size=5, edgecolor=None, alpha=1, 
                    legend=False, ax=ax)
    ax.plot(radius27px0deg_cell_sim_list_original_nd_df['Average Angle (deg)'], radius27px0deg_cell_sim_list_original_nd_df['Contact Surface Length (N.D.)'], c='k', ls='--')
    ax.plot(angle_theoretical * 2, cont_radius_theoretical2 * 2, c='k', ls=':')
    #ax.scatter(eecellpairs_y_angle, eecellpairs_y_contnd, zorder=10, marker='.', s=5, c='k')
    # ax.plot(eecellpairs_y_angle, eecellpairs_y_contnd, zorder=10, c='k', ls='--', lw=1)
    # ax.plot(ee_genlog_y_angle_contndfit, ee_genlog_y_contnd, zorder=10, c='grey', ls='-', lw=1)
    ax.set_xlim(-5, 220)
    ax.set_ylim(-0.05, 4)
    ax.axvline(0, ls='--', c='grey')
    ax.axvline(180, ls='--', c='grey')
    ax.axhline(0, ls='--', c='grey')
    ax.axhline(cont_radius_theoretical2.max() * 2, ls='--', c='grey')
    # ax.set_xlabel('')
    # ax.set_ylabel('')
multipdf(Path.joinpath(output_dir, 'eecp_ang_vs_diam.pdf'), tight=True, dpi=dpi_graph)
plt.close('all')




red_curvature_value_headers = ['bin_0_curv_val_red', 'bin_1_curv_val_red', 'bin_2_curv_val_red', 
                               'bin_3_curv_val_red', 'bin_4_curv_val_red', 'bin_5_curv_val_red', 
                               'bin_6_curv_val_red', 'bin_7_curv_val_red', 'whole_cell_curv_val_red']

red_curvature_value_whole_header = ['whole_cell_curv_val_red']

grn_curvature_value_headers = ['bin_0_curv_val_grn', 'bin_1_curv_val_grn', 'bin_2_curv_val_grn', 
                               'bin_3_curv_val_grn', 'bin_4_curv_val_grn', 'bin_5_curv_val_grn', 
                               'bin_6_curv_val_grn', 'bin_7_curv_val_grn', 'whole_cell_curv_val_grn']

grn_curvature_value_whole_header = ['whole_cell_curv_val_grn']

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'silver', 'purple']

#fig, ax = plt.subplots()
# for i,df in enumerate(ee_list_dropna_time_corr_cpu_cellpair_only[:]):
#     fig, ax = plt.subplots(figsize=(15,5))
#     #[sns.lineplot(x='Time (minutes)', y=bin_num, data=df, marker='.', legend='full', ax=ax) for bin_num in red_curvature_value_headers]
#     [ax.plot(df['Time (minutes)'], df[bin_num]/ee_first_valid_curvature_red_cellpair_only.iloc[i], marker='.', c=colors[j]) for j,bin_num in enumerate(red_curvature_value_headers)]
#     ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#     ax.set_ylim(-0.2, 3)
#     plt.show()

# # Plot only the curvature of the non-contact borders of all files on one graph:
# fig, ax = plt.subplots(figsize=(15,5))
# for i,df in enumerate(ee_list_dropna_time_corr_cpu_cellpair_only[:]):
#     [ax.plot(df['Time (minutes)'], df[bin_num]/ee_first_valid_curvature_red_cellpair_only.iloc[i], marker='.') for bin_num in red_curvature_value_whole_header]
    
    
#sns.catplot(y=red_curvature_value_headers, data=ee_df_dropna_time_corr_cpu, ax=ax)
#ax.set_yticklabels(labels=ax.get_xticklabels(), rotation = 45, ha='right')

#colors = ['k', 'r', 'b', 'b', 'b', 'b', 'k', 'silver', 'purple']
#for df in ee_list_dropna_time_corr_cpu_cellpair_only[:]:
#    fig, ax = plt.subplots(figsize=(15,5))
#    #[sns.lineplot(x='Time (minutes)', y=bin_num, data=df, marker='.', legend='full', ax=ax) for bin_num in red_curvature_value_headers]
#    [ax.plot(df['Time (minutes)'], df[bin_num], marker='.', c=colors[i]) for i,bin_num in enumerate(red_curvature_value_headers)]
#    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#    plt.show()



colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'silver', 'purple']

#fig, ax = plt.subplots(figsize = (15,5))
# for i,df in enumerate(ee_list_dropna_time_corr_cpu_cellpair_only[:]):
#     fig, ax = plt.subplots(figsize=(15,5))
#     #[sns.lineplot(x='Time (minutes)', y=bin_num, data=df, marker='.', legend='full', ax=ax) for bin_num in grn_curvature_value_headers]
#     [ax.plot(df['Time (minutes)'], df[bin_num]/ee_first_valid_curvature_grn_cellpair_only.iloc[i], marker='.', c=colors[j]) for j,bin_num in enumerate(grn_curvature_value_headers)]
#     ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#     ax.set_ylim(-0.2, 3)
#     plt.show()

# # Plot only the curvature of the non-contact borders of all files on one graph:
# fig, ax = plt.subplots(figsize=(15,5))
# for i,df in enumerate(ee_list_dropna_time_corr_cpu_cellpair_only[:]):
#     [ax.plot(df['Time (minutes)'], df[bin_num]/ee_first_valid_curvature_grn_cellpair_only.iloc[i], marker='.') for bin_num in grn_curvature_value_whole_header]

#sns.catplot(y=grn_curvature_value_headers, data=ee_df_dropna_time_corr_cpu, ax=ax)
#ax.set_xticklabels(labels=ax.get_xticklabels(), rotation = 45, ha='right')



#####################
### Plotting the cell areas of the red and green cells for just cell pairs:

# Plot the cell areas of the red and green cells for just cell pairs over time:
plt.clf()
fig, (ax1, ax2) = plt.subplots(figsize=(7,7), nrows=2, sharex=True)
sns.scatterplot(x='Time (minutes)', y='Area of cell 1 (Red) (N.D.)', hue='Source_File', data=ee_df_dropna_time_corr_cpu_cellpair_only, marker='.', legend=False, ax=ax1)
sns.scatterplot(x='Time (minutes)', y='Area of cell 2 (Green) (N.D.)', hue='Source_File', data=ee_df_dropna_time_corr_cpu_cellpair_only, marker='.', legend=False, ax=ax2)
plt.show()
#plt.savefig)

plt.clf()
fig, ax = plt.subplots(figsize=(10,5))
sns.scatterplot(x='Time (minutes)', y='Area of the cell pair (N.D.)', hue='Source_File', data=ee_df_dropna_time_corr_cpu_cellpair_only, marker='.', legend=False, ax=ax)
plt.show()



    
# Plot the cell areas of the red and green cells for just cell pairs relative
# to the Contact angles:
plt.clf()
fig, (ax1, ax2) = plt.subplots(figsize=(7,7), nrows=2, sharex=True)
sns.scatterplot(x='Average Angle (deg)', y='Area of cell 1 (Red) (N.D.)', hue='Source_File', data=ee_df_dropna_time_corr_cpu_cellpair_only, marker='.', legend=False, ax=ax1)
sns.scatterplot(x='Average Angle (deg)', y='Area of cell 2 (Green) (N.D.)', hue='Source_File', data=ee_df_dropna_time_corr_cpu_cellpair_only, marker='.', legend=False, ax=ax2)
#ax1.set_ylabel('')
#ax2.set_ylabel('')
#ax2.set_xlabel('')
plt.savefig(Path.joinpath(output_dir, r"ee_cellpair_only_angle_vs_area_indiv.tif"), dpi=dpi_graph)
plt.clf()

fig, (ax1, ax2) = plt.subplots(figsize=(7,7), nrows=2, sharex=True)
ax1.errorbar(x=ee_time_corr_avg_cellpair_only_angle['mean'].values, y=ee_time_corr_avg_cellpair_only['Area of cell 1 (Red) (N.D.)'].describe().reset_index()['mean'].values, xerr=ee_time_corr_avg_cellpair_only_angle['std'].values, yerr=ee_time_corr_avg_cellpair_only['Area of cell 1 (Red) (N.D.)'].describe().reset_index()['std'].values, c='r', ecolor='grey', ls='None', marker='.')
ax2.errorbar(x=ee_time_corr_avg_cellpair_only_angle['mean'].values, y=ee_time_corr_avg_cellpair_only['Area of cell 2 (Green) (N.D.)'].describe().reset_index()['mean'].values, xerr=ee_time_corr_avg_cellpair_only_angle['std'].values, yerr=ee_time_corr_avg_cellpair_only['Area of cell 2 (Green) (N.D.)'].describe().reset_index()['std'].values, c='g', ecolor='grey', ls='None', marker='.')
ax1.plot(radius27px0deg_cell_sim_list_original_nd_df['Average Angle (deg)'], radius27px0deg_cell_sim_list_original_nd_df['Area of cell 1 (Red) (N.D.)'], c='k', ls='--', zorder=10)
ax2.plot(radius27px0deg_cell_sim_list_original_nd_df['Average Angle (deg)'], radius27px0deg_cell_sim_list_original_nd_df['Area of cell 1 (Red) (N.D.)'], c='k', ls='--', zorder=10)
ax1.plot(angle_theoretical*2, area_cell_theoretical_nd, c='k', zorder=10)
ax2.plot(angle_theoretical*2, area_cell_theoretical_nd, c='k', zorder=10)
ax1.axvline(180, c='k', ls=':')
ax2.axvline(180, c='k', ls=':')
ax2.set_xlim(0,190)
ax2.xaxis.set_major_locator(ticker.MultipleLocator(20))
#ax1.set_ylabel('Area of cell 1 (Red) (N.D.)')
#ax2.set_ylabel('Area of cell 2 (Green) (N.D.)')
#ax2.set_xlabel('Average Angle (deg)')
ax1.set_ylabel('')
ax2.set_ylabel('')
ax2.set_xlabel('')
plt.savefig(Path.joinpath(output_dir, r"ee_cellpair_only_angle_vs_area_time_avg.tif"), dpi=dpi_graph)
plt.clf()

# Plot the cell areas of the red and green cells for just cell pairs relative
# to the Contact surface length:
fig, (ax1, ax2) = plt.subplots(figsize=(7,7), nrows=2, sharex=True)
sns.scatterplot(x='Contact Surface Length (N.D.)', y='Area of cell 1 (Red) (N.D.)', hue='Source_File', data=ee_df_dropna_time_corr_cpu_cellpair_only, marker='.', legend=False, ax=ax1)
sns.scatterplot(x='Contact Surface Length (N.D.)', y='Area of cell 2 (Green) (N.D.)', hue='Source_File', data=ee_df_dropna_time_corr_cpu_cellpair_only, marker='.', legend=False, ax=ax2)
plt.savefig(Path.joinpath(output_dir, r"ee_cellpair_only_contnd_vs_area_indiv.tif"), dpi=dpi_graph)
plt.clf()

fig, (ax1, ax2) = plt.subplots(figsize=(7,7), nrows=2, sharex=True)
ax1.errorbar(x=ee_time_corr_avg_cellpair_only_cont_nd['mean'].values, y=ee_time_corr_avg_cellpair_only['Area of cell 1 (Red) (N.D.)'].describe().reset_index()['mean'].values, xerr=ee_time_corr_avg_cellpair_only_cont_nd['std'].values, yerr=ee_time_corr_avg_cellpair_only['Area of cell 1 (Red) (N.D.)'].describe().reset_index()['std'].values, c='r', ecolor='grey', ls='None', marker='.')
ax2.errorbar(x=ee_time_corr_avg_cellpair_only_cont_nd['mean'].values, y=ee_time_corr_avg_cellpair_only['Area of cell 2 (Green) (N.D.)'].describe().reset_index()['mean'].values, xerr=ee_time_corr_avg_cellpair_only_cont_nd['std'].values, yerr=ee_time_corr_avg_cellpair_only['Area of cell 2 (Green) (N.D.)'].describe().reset_index()['std'].values, c='g', ecolor='grey', ls='None', marker='.')
ax1.plot(radius27px0deg_cell_sim_list_original_nd_df['Contact Surface Length (N.D.)'], radius27px0deg_cell_sim_list_original_nd_df['Area of cell 1 (Red) (N.D.)'], c='k', ls='--', zorder=10)
ax2.plot(radius27px0deg_cell_sim_list_original_nd_df['Contact Surface Length (N.D.)'], radius27px0deg_cell_sim_list_original_nd_df['Area of cell 1 (Red) (N.D.)'], c='k', ls='--', zorder=10)
ax1.plot(cont_radius_theoretical2*2, area_cell_theoretical_nd, c='k', zorder=10)
ax2.plot(cont_radius_theoretical2*2, area_cell_theoretical_nd, c='k', zorder=10)
ax1.axvline(f(90, 1)[1][4], c='k', ls=':')
ax2.axvline(f(90, 1)[1][4], c='k', ls=':')
ax2.set_xlim(0,4)
#ax2.xaxis.set_major_locator(ticker.MultipleLocator(20))
#ax1.set_ylabel('Area of cell 1 (Red) (N.D.)')
#ax2.set_ylabel('Area of cell 2 (Green) (N.D.)')
#ax2.set_xlabel('Contact Surface Length (N.D.)')
ax1.set_ylabel('')
ax2.set_ylabel('')
ax2.set_xlabel('')
plt.savefig(Path.joinpath(output_dir, r"ee_cellpair_only_contnd_vs_area_time_avg.tif"), dpi=dpi_graph)
plt.clf()









###############################################################################




# This is the proportion of touch events from the figure 
# "ee_contig_touch_overall_tintervals" that are in contact for only a single 
# timeframe for 30sec and 1min time intervals:
# Touch events in contact for only a single timeframe:
num_contig_touch_events_with_1_timeframe_tint30s = np.count_nonzero(df_flattened_expanded_sf.loc[df_flattened_expanded_sf[df_flattened_expanded_sf['time_intervals']==0.5].index]['contig_touch'] == 1)
num_contig_touch_events_with_1_timeframe_tint1min = np.count_nonzero(df_flattened_expanded_sf.loc[df_flattened_expanded_sf[df_flattened_expanded_sf['time_intervals']==1].index]['contig_touch'] == 1)

# and also for only 2 timeframes:
num_contig_touch_events_with_2_timeframe_tint30s = np.count_nonzero(df_flattened_expanded_sf.loc[df_flattened_expanded_sf[df_flattened_expanded_sf['time_intervals']==0.5].index]['contig_touch'] == 2)
num_contig_touch_events_with_2_timeframe_tint1min = np.count_nonzero(df_flattened_expanded_sf.loc[df_flattened_expanded_sf[df_flattened_expanded_sf['time_intervals']==1].index]['contig_touch'] == 2)

# and also for 1 or 2 timeframes:
num_contig_touch_1_or_2_tfs_tint30s = num_contig_touch_events_with_1_timeframe_tint30s + num_contig_touch_events_with_2_timeframe_tint30s
num_contig_touch_1_or_2_tfs_tint60s = num_contig_touch_events_with_1_timeframe_tint1min + num_contig_touch_events_with_2_timeframe_tint1min

# total touch events
num_tot_contig_touch_tint30s = np.count_nonzero(df_flattened_expanded_sf.loc[df_flattened_expanded_sf[df_flattened_expanded_sf['time_intervals']==0.5].index]['contig_touch'])
num_tot_contig_touch_tint60s = np.count_nonzero(df_flattened_expanded_sf.loc[df_flattened_expanded_sf[df_flattened_expanded_sf['time_intervals']==1].index]['contig_touch'])


prop_contig_touch = [('touch duration (timeframes)', 'tint_30s', 'tint_60s'),
                     ('1', num_contig_touch_events_with_1_timeframe_tint30s, num_contig_touch_events_with_1_timeframe_tint1min), 
                     ('2', num_contig_touch_events_with_2_timeframe_tint30s, num_contig_touch_events_with_2_timeframe_tint1min), 
                     ('1 or 2', num_contig_touch_1_or_2_tfs_tint30s, num_contig_touch_1_or_2_tfs_tint60s),
                     ('any', num_tot_contig_touch_tint30s, num_tot_contig_touch_tint60s)]

with open(Path.joinpath(output_dir, r'prop_contig_touch.tsv'), 
          'w', newline='') as file_out:
            tsv_obj = csv.writer(file_out, delimiter='\t')
            tsv_obj.writerows(prop_contig_touch)



# What proportion of timelapses that were manually selected for their low 
# adhesion have a their maximum contiguously touching time of 2 or less
# timeframes? (related to "max_contig_touch_tintervals")
max_contig_touch_tint_t30s, max_contig_touch_tint_t60s = [df for group,df in 
                                                          df_fl_ex_sf_summary.groupby('time_intervals')]
max_contig_touch_tint_t30s_eedb, = [df for group,df in df_fl_ex_sf_summary_eedb.groupby('time_intervals')]


# number of ee cellpair timelapses with a longest contiguous touching event 
# during the low plateau of only 1 frame:
max_contig_touch_tint_n1_t30s = np.count_nonzero(max_contig_touch_tint_t30s[('contig_touch','max')] == 1)
max_contig_touch_tint_n1_t60s = np.count_nonzero(max_contig_touch_tint_t60s[('contig_touch','max')] == 1)
max_contig_touch_tint_n1_t30s_eedb = np.count_nonzero(max_contig_touch_tint_t30s_eedb[('contig_touch','max')] == 1)

# number of ee cellpair timelapses with a longest contiguous touching event 
# during the low plateau of 2 frames:
max_contig_touch_tint_n2_t30s = np.count_nonzero(max_contig_touch_tint_t30s[('contig_touch','max')] == 2)
max_contig_touch_tint_n2_t60s = np.count_nonzero(max_contig_touch_tint_t60s[('contig_touch','max')] == 2)
max_contig_touch_tint_n2_t30s_eedb = np.count_nonzero(max_contig_touch_tint_t30s_eedb[('contig_touch','max')] == 2)

# number of ee cellpair timelapses with a longest contiguous touching event 
# during the low plateau of 1 or 2 frames:
max_contig_touch_tint_n1or2_t30s = max_contig_touch_tint_n1_t30s + max_contig_touch_tint_n2_t30s
max_contig_touch_tint_n1or2_t60s = max_contig_touch_tint_n1_t60s + max_contig_touch_tint_n2_t60s
max_contig_touch_tint_n1or2_t30s_eedb = max_contig_touch_tint_n1_t30s_eedb + max_contig_touch_tint_n2_t30s_eedb

# Total number of ee cellpair timelapses analyzed:
max_contig_touch_tint_ntot_t30s = len(max_contig_touch_tint_t30s)
max_contig_touch_tint_ntot_t60s = len(max_contig_touch_tint_t60s)
max_contig_touch_tint_ntot_t30s_eedb = len(max_contig_touch_tint_t30s_eedb)



prop_contig_touch_maxes = [('max touch duration (timeframes)', 'tint_30s', 'tint_60s', 'tint_30s_eedb'),
                           ('1', max_contig_touch_tint_n1_t30s, max_contig_touch_tint_n1_t60s, max_contig_touch_tint_n1_t30s_eedb), 
                           ('2', max_contig_touch_tint_n2_t30s, max_contig_touch_tint_n2_t60s, max_contig_touch_tint_n2_t30s_eedb), 
                           ('1 or 2', max_contig_touch_tint_n1or2_t30s, max_contig_touch_tint_n1or2_t60s, max_contig_touch_tint_n1or2_t30s_eedb),
                           ('all timeframes', max_contig_touch_tint_ntot_t30s, max_contig_touch_tint_ntot_t60s, max_contig_touch_tint_ntot_t30s_eedb)]

with open(Path.joinpath(output_dir, r'prop_contig_touch_maxes.tsv'), 
          'w', newline='') as file_out:
            tsv_obj = csv.writer(file_out, delimiter='\t')
            tsv_obj.writerows(prop_contig_touch_maxes)


# If the contact angle increases logisticially, do you also expect the contact
# diameter to increase logistically for a sphere of constant volume?
plt.clf()
plt.plot(x_fit, f(ee_genlog_y_angle, 1)[1][-1] / f(ee_genlog_y_angle, 1)[1][-1].max())
plt.plot(x_fit, ee_genlog_y_angle / ee_genlog_y_angle.max())
# Answer: Yes.



# Just plotting the ee_contactnds_gen_log as an example:
#fig, ax = plt.subplots()
#ax.plot(x_fit, ee_genlog_y_contnd, c='k')
#ax.scatter(x_fit[np.argmax(ee_genlog_y_contnd_diff)], ee_genlog_y_contnd[np.argmax(ee_genlog_y_contnd_diff)], c='r', marker='.', zorder=10)
#ax.set_ylim(0,2.5)
#ax.set_xlim(-50, 75)
#ax.set_xticklabels('')
#ax.set_yticklabels('')
#fig.savefig(Path.joinpath(output_dir, r"example_gen_log.tif"), dpi=dpi_graph)
###







### Plot histograms of parameters for individual logistic fits:
# x0 = ee_indiv_logistic_contnd_popt_pcov[i][0][0]
# k = ee_indiv_logistic_contnd_popt_pcov[i][0][1]
# upper_plateau = ee_indiv_logistic_contnd_popt_pcov[i][0][2]
# lower_plateau = ee_indiv_logistic_contnd_popt_pcov[i][0][3]

# For ecto-ecto:
plt.clf()
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(width_1col,height_1col))
# plot individual parameters of all timelapses as green:
# the point of inflection of each logistic fit (ie. x0):
#sns.swarmplot(pd.Series([ee_indiv_logistic_contnd_popt_pcov[i][0][0] for i in range(len(ee_indiv_logistic_contnd_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax1, color='g', alpha=0.5)
sns.swarmplot(pd.Series([ee_indiv_logistic_contnd_popt_pcov[i][0][0] for i in ee_list_dropna_fit_validity_df.query('growth_phase == False').index]).dropna(inplace=False), orient='v', ax=ax1, marker='.', color='k', alpha=0.5) # only valid growth phase timelapses could be aligned, therefore only those x0's could be used.
sns.swarmplot(pd.Series([ee_indiv_logistic_contnd_popt_pcov[i][0][0] for i in ee_list_dropna_fit_validity_df.query('growth_phase == True').index]).dropna(inplace=False), orient='v', ax=ax1, marker='.', color='g', alpha=0.5) # only valid growth phase timelapses could be aligned, therefore only those x0's could be used.
sns.violinplot(pd.Series([ee_indiv_logistic_contnd_popt_pcov[i][0][0] for i in range(len(ee_indiv_logistic_contnd_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax1, color='lightgrey', alpha=1)
# the logistic growth rate (ie. k):
#sns.swarmplot(pd.Series([ee_indiv_logistic_contnd_popt_pcov[i][0][1] for i in range(len(ee_indiv_logistic_contnd_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax2, color='g', alpha=0.5)
sns.swarmplot(pd.Series([ee_indiv_logistic_contnd_popt_pcov[i][0][1] for i in ee_list_dropna_fit_validity_df.query('growth_phase == False').index]).dropna(inplace=False), orient='v', ax=ax2, marker='.', color='k', alpha=0.5)
sns.swarmplot(pd.Series([ee_indiv_logistic_contnd_popt_pcov[i][0][1] for i in ee_list_dropna_fit_validity_df.query('growth_phase == True').index]).dropna(inplace=False), orient='v', ax=ax2, marker='.', color='g', alpha=0.5)
sns.violinplot(pd.Series([ee_indiv_logistic_contnd_popt_pcov[i][0][1] for i in range(len(ee_indiv_logistic_contnd_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax2, color='lightgrey', alpha=1)
# the high plateau:
#sns.swarmplot(pd.Series([ee_indiv_logistic_contnd_popt_pcov[i][0][2] for i in range(len(ee_indiv_logistic_contnd_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax3, color='g', alpha=0.5)
sns.swarmplot(pd.Series([ee_indiv_logistic_contnd_popt_pcov[i][0][2] for i in ee_list_dropna_fit_validity_df.query('high_plat == False').index]).dropna(inplace=False), orient='v', ax=ax3, marker='.', color='k', alpha=0.5)
sns.swarmplot(pd.Series([ee_indiv_logistic_contnd_popt_pcov[i][0][2] for i in ee_list_dropna_fit_validity_df.query('high_plat == True').index]).dropna(inplace=False), orient='v', ax=ax3, marker='.', color='g', alpha=0.5)
sns.violinplot(pd.Series([ee_indiv_logistic_contnd_popt_pcov[i][0][2] for i in range(len(ee_indiv_logistic_contnd_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax3, color='lightgrey', alpha=1)
# the low plateau
#sns.swarmplot(pd.Series([ee_indiv_logistic_contnd_popt_pcov[i][0][3] for i in range(len(ee_indiv_logistic_contnd_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax4, color='g', alpha=0.5)
sns.swarmplot(pd.Series([ee_indiv_logistic_contnd_popt_pcov[i][0][3] for i in ee_list_dropna_fit_validity_df.query('low_plat == False').index]).dropna(inplace=False), orient='v', ax=ax4, marker='.', color='k', alpha=0.5)
sns.swarmplot(pd.Series([ee_indiv_logistic_contnd_popt_pcov[i][0][3] for i in ee_list_dropna_fit_validity_df.query('low_plat == True').index]).dropna(inplace=False), orient='v', ax=ax4, marker='.', color='g', alpha=0.5)
sns.violinplot(pd.Series([ee_indiv_logistic_contnd_popt_pcov[i][0][3] for i in range(len(ee_indiv_logistic_contnd_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax4, color='lightgrey', alpha=1)
# plot the manually curated parameters as black dots:
#sns.swarmplot(pd.Series([ee_indiv_logistic_contnd_popt_pcov[i][0][0] for i in ee_list_dropna_fit_validity_df.query('growth_phase == True').index]).dropna(inplace=False), orient='v', ax=ax1, marker='.', color='k', alpha=0.5) # only valid growth phase timelapses could be aligned, therefore only those x0's could be used.
#sns.swarmplot(pd.Series([ee_indiv_logistic_contnd_popt_pcov[i][0][1] for i in ee_list_dropna_fit_validity_df.query('growth_phase == True').index]).dropna(inplace=False), orient='v', ax=ax2, marker='.', color='k', alpha=0.5)
#sns.swarmplot(pd.Series([ee_indiv_logistic_contnd_popt_pcov[i][0][2] for i in ee_list_dropna_fit_validity_df.query('high_plat == True').index]).dropna(inplace=False), orient='v', ax=ax3, marker='.', color='k', alpha=0.5)
#sns.swarmplot(pd.Series([ee_indiv_logistic_contnd_popt_pcov[i][0][3] for i in ee_list_dropna_fit_validity_df.query('low_plat == True').index]).dropna(inplace=False), orient='v', ax=ax4, marker='.', color='k', alpha=0.5)

ax3.axhline(bounds_contnd[0][2], c='k')
ax3.axhline(bounds_contnd[1][2], c='k')
ax4.axhline(bounds_contnd[0][3], c='k')
ax4.axhline(bounds_contnd[1][3], c='k')
#ax1.set_xlabel(r'x$_\theta$')
#ax2.set_xlabel('logistic growth rate "k"')
#ax3.set_xlabel('upper_plateau')
#ax4.set_xlabel('lower_plateau')
#plt.suptitle('contnd', fontsize=18, x=0.5, y=1.05)
plt.tight_layout()
plt.savefig(Path.joinpath(output_dir, r"ee_contnd_popt_distrib.tif"), dpi=dpi_graph)
#print(bounds_contnd)

plt.clf()
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(width_1col,height_1col))
# plot individual parameters of all timelapses as green:
#sns.swarmplot(pd.Series([ee_indiv_logistic_cont_popt_pcov[i][0][0] for i in range(len(ee_indiv_logistic_cont_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax1, color='g', alpha=0.5)
sns.swarmplot(pd.Series([ee_indiv_logistic_cont_popt_pcov[i][0][0] for i in ee_list_dropna_fit_validity_df.query('growth_phase == False').index]).dropna(inplace=False), orient='v', ax=ax1, marker='.', color='k', alpha=0.5) # only valid growth phase timelapses could be aligned, therefore only those x0's could be used.
sns.swarmplot(pd.Series([ee_indiv_logistic_cont_popt_pcov[i][0][0] for i in ee_list_dropna_fit_validity_df.query('growth_phase == True').index]).dropna(inplace=False), orient='v', ax=ax1, marker='.', color='g', alpha=0.5) # only valid growth phase timelapses could be aligned, therefore only those x0's could be used.
sns.violinplot(pd.Series([ee_indiv_logistic_cont_popt_pcov[i][0][0] for i in range(len(ee_indiv_logistic_cont_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax1, color='lightgrey', alpha=1)

#sns.swarmplot(pd.Series([ee_indiv_logistic_cont_popt_pcov[i][0][1] for i in range(len(ee_indiv_logistic_cont_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax2, color='g', alpha=0.5)
sns.swarmplot(pd.Series([ee_indiv_logistic_cont_popt_pcov[i][0][1] for i in ee_list_dropna_fit_validity_df.query('growth_phase == False').index]).dropna(inplace=False), orient='v', ax=ax2, marker='.', color='k', alpha=0.5)
sns.swarmplot(pd.Series([ee_indiv_logistic_cont_popt_pcov[i][0][1] for i in ee_list_dropna_fit_validity_df.query('growth_phase == True').index]).dropna(inplace=False), orient='v', ax=ax2, marker='.', color='g', alpha=0.5)
sns.violinplot(pd.Series([ee_indiv_logistic_cont_popt_pcov[i][0][1] for i in range(len(ee_indiv_logistic_cont_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax2, color='lightgrey', alpha=1)

#sns.swarmplot(pd.Series([ee_indiv_logistic_cont_popt_pcov[i][0][2] for i in range(len(ee_indiv_logistic_cont_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax3, color='g', alpha=0.5)
sns.swarmplot(pd.Series([ee_indiv_logistic_cont_popt_pcov[i][0][2] for i in ee_list_dropna_fit_validity_df.query('high_plat == False').index]).dropna(inplace=False), orient='v', ax=ax3, marker='.', color='k', alpha=0.5)
sns.swarmplot(pd.Series([ee_indiv_logistic_cont_popt_pcov[i][0][2] for i in ee_list_dropna_fit_validity_df.query('high_plat == True').index]).dropna(inplace=False), orient='v', ax=ax3, marker='.', color='g', alpha=0.5)
sns.violinplot(pd.Series([ee_indiv_logistic_cont_popt_pcov[i][0][2] for i in range(len(ee_indiv_logistic_cont_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax3, color='lightgrey', alpha=1)

#sns.swarmplot(pd.Series([ee_indiv_logistic_cont_popt_pcov[i][0][3] for i in range(len(ee_indiv_logistic_cont_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax4, color='g', alpha=0.5)
sns.swarmplot(pd.Series([ee_indiv_logistic_cont_popt_pcov[i][0][3] for i in ee_list_dropna_fit_validity_df.query('low_plat == False').index]).dropna(inplace=False), orient='v', ax=ax4, marker='.', color='k', alpha=0.5)
sns.swarmplot(pd.Series([ee_indiv_logistic_cont_popt_pcov[i][0][3] for i in ee_list_dropna_fit_validity_df.query('low_plat == True').index]).dropna(inplace=False), orient='v', ax=ax4, marker='.', color='g', alpha=0.5)
sns.violinplot(pd.Series([ee_indiv_logistic_cont_popt_pcov[i][0][3] for i in range(len(ee_indiv_logistic_cont_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax4, color='lightgrey', alpha=1)
# plot the manually selected timelapses with a low plateau as black dots:
#sns.swarmplot(pd.Series([ee_indiv_logistic_cont_popt_pcov[i][0][0] for i in ee_list_dropna_fit_validity_df.query('growth_phase == True').index]).dropna(inplace=False), orient='v', ax=ax1, marker='.', color='k', alpha=0.5) # only valid growth phase timelapses could be aligned, therefore only those x0's could be used.
#sns.swarmplot(pd.Series([ee_indiv_logistic_cont_popt_pcov[i][0][1] for i in ee_list_dropna_fit_validity_df.query('growth_phase == True').index]).dropna(inplace=False), orient='v', ax=ax2, marker='.', color='k', alpha=0.5)
#sns.swarmplot(pd.Series([ee_indiv_logistic_cont_popt_pcov[i][0][2] for i in ee_list_dropna_fit_validity_df.query('high_plat == True').index]).dropna(inplace=False), orient='v', ax=ax3, marker='.', color='k', alpha=0.5)
#sns.swarmplot(pd.Series([ee_indiv_logistic_cont_popt_pcov[i][0][3] for i in ee_list_dropna_fit_validity_df.query('low_plat == True').index]).dropna(inplace=False), orient='v', ax=ax4, marker='.', color='k', alpha=0.5)

ax3.axhline(bounds_cont[0][2], c='k')
ax3.axhline(bounds_cont[1][2], c='k')
ax4.axhline(bounds_cont[0][3], c='k')
ax4.axhline(bounds_cont[1][3], c='k')
#ax1.set_xlabel(r'x$_\theta$')
#ax2.set_xlabel('logistic growth rate "k"')
#ax3.set_xlabel('upper_plateau')
#ax4.set_xlabel('lower_plateau')
#plt.suptitle('cont', fontsize=18, x=0.5, y=1.05)
plt.tight_layout()
plt.savefig(Path.joinpath(output_dir, r"ee_cont_popt_distrib.tif"), dpi=dpi_graph)
#print(bounds_cont)

plt.clf()
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(width_1col,height_1col))
# plot individual parameters of all timelapses as green:
#sns.swarmplot(pd.Series([ee_indiv_logistic_angle_popt_pcov[i][0][0] for i in range(len(ee_indiv_logistic_angle_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax1, color='g', alpha=0.5)
sns.swarmplot(pd.Series([ee_indiv_logistic_angle_popt_pcov[i][0][0] for i in ee_list_dropna_angle_fit_validity_df.query('growth_phase == False').index]).dropna(inplace=False), orient='v', ax=ax1, marker='.', color='k', alpha=0.5) # only valid growth phase timelapses could be aligned, therefore only those x0's could be used.
sns.swarmplot(pd.Series([ee_indiv_logistic_angle_popt_pcov[i][0][0] for i in ee_list_dropna_angle_fit_validity_df.query('growth_phase == True').index]).dropna(inplace=False), orient='v', ax=ax1, marker='.', color='g', alpha=0.5) # only valid growth phase timelapses could be aligned, therefore only those x0's could be used.
sns.violinplot(pd.Series([ee_indiv_logistic_angle_popt_pcov[i][0][0] for i in range(len(ee_indiv_logistic_angle_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax1, color='lightgrey', alpha=1)

#sns.swarmplot(pd.Series([ee_indiv_logistic_angle_popt_pcov[i][0][1] for i in range(len(ee_indiv_logistic_angle_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax2, color='g', alpha=0.5)
sns.swarmplot(pd.Series([ee_indiv_logistic_angle_popt_pcov[i][0][1] for i in ee_list_dropna_angle_fit_validity_df.query('growth_phase == False').index]).dropna(inplace=False), orient='v', ax=ax2, marker='.', color='k', alpha=0.5)
sns.swarmplot(pd.Series([ee_indiv_logistic_angle_popt_pcov[i][0][1] for i in ee_list_dropna_angle_fit_validity_df.query('growth_phase == True').index]).dropna(inplace=False), orient='v', ax=ax2, marker='.', color='g', alpha=0.5)
sns.violinplot(pd.Series([ee_indiv_logistic_angle_popt_pcov[i][0][1] for i in range(len(ee_indiv_logistic_angle_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax2, color='lightgrey', alpha=1)

#sns.swarmplot(pd.Series([ee_indiv_logistic_angle_popt_pcov[i][0][2] for i in range(len(ee_indiv_logistic_angle_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax3, color='g', alpha=0.5)
sns.swarmplot(pd.Series([ee_indiv_logistic_angle_popt_pcov[i][0][2] for i in ee_list_dropna_angle_fit_validity_df.query('high_plat == False').index]).dropna(inplace=False), orient='v', ax=ax3, marker='.', color='k', alpha=0.5)
sns.swarmplot(pd.Series([ee_indiv_logistic_angle_popt_pcov[i][0][2] for i in ee_list_dropna_angle_fit_validity_df.query('high_plat == True').index]).dropna(inplace=False), orient='v', ax=ax3, marker='.', color='g', alpha=0.5)
sns.violinplot(pd.Series([ee_indiv_logistic_angle_popt_pcov[i][0][2] for i in range(len(ee_indiv_logistic_angle_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax3, color='lightgrey', alpha=1)

#sns.swarmplot(pd.Series([ee_indiv_logistic_angle_popt_pcov[i][0][3] for i in range(len(ee_indiv_logistic_angle_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax4, color='g', alpha=0.5)
sns.swarmplot(pd.Series([ee_indiv_logistic_angle_popt_pcov[i][0][3] for i in ee_list_dropna_angle_fit_validity_df.query('low_plat == False').index]).dropna(inplace=False), orient='v', ax=ax4, marker='.', color='k', alpha=0.5)
sns.swarmplot(pd.Series([ee_indiv_logistic_angle_popt_pcov[i][0][3] for i in ee_list_dropna_angle_fit_validity_df.query('low_plat == True').index]).dropna(inplace=False), orient='v', ax=ax4, marker='.', color='g', alpha=0.5)
sns.violinplot(pd.Series([ee_indiv_logistic_angle_popt_pcov[i][0][3] for i in range(len(ee_indiv_logistic_angle_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax4, color='lightgrey', alpha=1)
# plot the manually selected timelapses with a low plateau as black dots:
#sns.swarmplot(pd.Series([ee_indiv_logistic_angle_popt_pcov[i][0][0] for i in ee_list_dropna_angle_fit_validity_df.query('growth_phase == True').index]).dropna(inplace=False), orient='v', ax=ax1, marker='.', color='k', alpha=0.5) # only valid growth phase timelapses could be aligned, therefore only those x0's could be used.
#sns.swarmplot(pd.Series([ee_indiv_logistic_angle_popt_pcov[i][0][1] for i in ee_list_dropna_angle_fit_validity_df.query('growth_phase == True').index]).dropna(inplace=False), orient='v', ax=ax2, marker='.', color='k', alpha=0.5)
#sns.swarmplot(pd.Series([ee_indiv_logistic_angle_popt_pcov[i][0][2] for i in ee_list_dropna_angle_fit_validity_df.query('high_plat == True').index]).dropna(inplace=False), orient='v', ax=ax3, marker='.', color='k', alpha=0.5)
#sns.swarmplot(pd.Series([ee_indiv_logistic_angle_popt_pcov[i][0][3] for i in ee_list_dropna_angle_fit_validity_df.query('low_plat == True').index]).dropna(inplace=False), orient='v', ax=ax4, marker='.', color='k', alpha=0.5)

ax3.axhline(bounds_angle[0][2], c='k')
ax3.axhline(bounds_angle[1][2], c='k')
ax4.axhline(bounds_angle[0][3], c='k')
ax4.axhline(bounds_angle[1][3], c='k')
#ax1.set_xlabel(r'x$_\theta$')
#ax2.set_xlabel('logistic growth rate "k"')
#ax3.set_xlabel('upper_plateau')
#ax4.set_xlabel('lower_plateau')
#plt.suptitle('angle', fontsize=18, x=0.5, y=1.05)
plt.tight_layout()
plt.savefig(Path.joinpath(output_dir, r"ee_angle_popt_distrib.tif"), dpi=dpi_graph)
#print(bounds_angle)


# For meso-meso:
plt.clf()
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(width_2col,height_2col))
# plot individual parameters of all timelapses as red:
#sns.swarmplot(pd.Series([mm_indiv_logistic_contnd_popt_pcov[i][0][0] for i in range(len(mm_indiv_logistic_contnd_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax1, color='r', alpha=0.5)
sns.swarmplot(pd.Series([mm_indiv_logistic_contnd_popt_pcov[i][0][0] for i in mm_list_dropna_fit_validity_df.query('growth_phase == False').index], dtype=float).dropna(inplace=False), orient='v', ax=ax1, marker='o', color='k', alpha=0.5) # only valid growth phase timelapses could be aligned, therefore only those x0's could be used.
sns.swarmplot(pd.Series([mm_indiv_logistic_contnd_popt_pcov[i][0][0] for i in mm_list_dropna_fit_validity_df.query('growth_phase == True').index], dtype=float).dropna(inplace=False), orient='v', ax=ax1, marker='o', color='r', alpha=0.5) # only valid growth phase timelapses could be aligned, therefore only those x0's could be used.
sns.violinplot(pd.Series([mm_indiv_logistic_contnd_popt_pcov[i][0][0] for i in range(len(mm_indiv_logistic_contnd_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax1, color='lightgrey', alpha=1)

#sns.swarmplot(pd.Series([mm_indiv_logistic_contnd_popt_pcov[i][0][1] for i in range(len(mm_indiv_logistic_contnd_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax2, color='r', alpha=0.5)
sns.swarmplot(pd.Series([mm_indiv_logistic_contnd_popt_pcov[i][0][1] for i in mm_list_dropna_fit_validity_df.query('growth_phase == False').index], dtype=float).dropna(inplace=False), orient='v', ax=ax2, marker='o', color='k', alpha=0.5)
sns.swarmplot(pd.Series([mm_indiv_logistic_contnd_popt_pcov[i][0][1] for i in mm_list_dropna_fit_validity_df.query('growth_phase == True').index], dtype=float).dropna(inplace=False), orient='v', ax=ax2, marker='o', color='r', alpha=0.5)
sns.violinplot(pd.Series([mm_indiv_logistic_contnd_popt_pcov[i][0][1] for i in range(len(mm_indiv_logistic_contnd_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax2, color='lightgrey', alpha=1)

#sns.swarmplot(pd.Series([mm_indiv_logistic_contnd_popt_pcov[i][0][2] for i in range(len(mm_indiv_logistic_contnd_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax3, color='r', alpha=0.5)
sns.swarmplot(pd.Series([mm_indiv_logistic_contnd_popt_pcov[i][0][2] for i in mm_list_dropna_fit_validity_df.query('high_plat == False').index], dtype=float).dropna(inplace=False), orient='v', ax=ax3, marker='o', color='k', alpha=0.5)
sns.swarmplot(pd.Series([mm_indiv_logistic_contnd_popt_pcov[i][0][2] for i in mm_list_dropna_fit_validity_df.query('high_plat == True').index], dtype=float).dropna(inplace=False), orient='v', ax=ax3, marker='o', color='r', alpha=0.5)
sns.violinplot(pd.Series([mm_indiv_logistic_contnd_popt_pcov[i][0][2] for i in range(len(mm_indiv_logistic_contnd_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax3, color='lightgrey', alpha=1)

#sns.swarmplot(pd.Series([mm_indiv_logistic_contnd_popt_pcov[i][0][3] for i in range(len(mm_indiv_logistic_contnd_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax4, color='r', alpha=0.5)
sns.swarmplot(pd.Series([mm_indiv_logistic_contnd_popt_pcov[i][0][3] for i in mm_list_dropna_fit_validity_df.query('low_plat == False').index], dtype=float).dropna(inplace=False), orient='v', ax=ax4, marker='o', color='k', alpha=0.5)
sns.swarmplot(pd.Series([mm_indiv_logistic_contnd_popt_pcov[i][0][3] for i in mm_list_dropna_fit_validity_df.query('low_plat == True').index], dtype=float).dropna(inplace=False), orient='v', ax=ax4, marker='o', color='r', alpha=0.5)
sns.violinplot(pd.Series([mm_indiv_logistic_contnd_popt_pcov[i][0][3] for i in range(len(mm_indiv_logistic_contnd_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax4, color='lightgrey', alpha=1)
# plot the manually selected timelapses with a low plateau as black dots:
#sns.swarmplot(pd.Series([mm_indiv_logistic_contnd_popt_pcov[i][0][0] for i in mm_list_dropna_fit_validity_df.query('growth_phase == True').index]).dropna(inplace=False), orient='v', ax=ax1, marker='.', color='k', alpha=0.5) # only valid growth phase timelapses could be aligned, therefore only those x0's could be used.
#sns.swarmplot(pd.Series([mm_indiv_logistic_contnd_popt_pcov[i][0][1] for i in mm_list_dropna_fit_validity_df.query('growth_phase == True').index]).dropna(inplace=False), orient='v', ax=ax2, marker='.', color='k', alpha=0.5)
#sns.swarmplot(pd.Series([mm_indiv_logistic_contnd_popt_pcov[i][0][2] for i in mm_list_dropna_fit_validity_df.query('high_plat == True').index]).dropna(inplace=False), orient='v', ax=ax3, marker='.', color='k', alpha=0.5)
#sns.swarmplot(pd.Series([mm_indiv_logistic_contnd_popt_pcov[i][0][3] for i in mm_list_dropna_fit_validity_df.query('low_plat == True').index]).dropna(inplace=False), orient='v', ax=ax4, marker='.', color='k', alpha=0.5)

ax3.axhline(bounds_contnd[0][2], c='k')
ax3.axhline(bounds_contnd[1][2], c='k')
ax4.axhline(bounds_contnd[0][3], c='k')
ax4.axhline(bounds_contnd[1][3], c='k')
ax1.set_xlabel(r'x$_\theta$')
ax2.set_xlabel('logistic growth rate "k"')
ax3.set_xlabel('upper_plateau')
ax4.set_xlabel('lower_plateau')
#plt.suptitle('contnd', fontsize=18, x=0.5, y=1.05)
plt.tight_layout()
plt.savefig(Path.joinpath(output_dir, r"mm_contnd_popt_distrib.tif"), dpi=dpi_graph)
#print(bounds_contnd)

plt.clf()
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(width_2col,height_2col))
# plot individual parameters of all timelapses as red:
#sns.swarmplot(pd.Series([mm_indiv_logistic_cont_popt_pcov[i][0][0] for i in range(len(mm_indiv_logistic_cont_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax1, color='r', alpha=0.5)
sns.swarmplot(pd.Series([mm_indiv_logistic_cont_popt_pcov[i][0][0] for i in mm_list_dropna_fit_validity_df.query('growth_phase == False').index], dtype=float).dropna(inplace=False), orient='v', ax=ax1, marker='o', color='k', alpha=0.5) # only valid growth phase timelapses could be aligned, therefore only those x0's could be used.
sns.swarmplot(pd.Series([mm_indiv_logistic_cont_popt_pcov[i][0][0] for i in mm_list_dropna_fit_validity_df.query('growth_phase == True').index], dtype=float).dropna(inplace=False), orient='v', ax=ax1, marker='o', color='r', alpha=0.5) # only valid growth phase timelapses could be aligned, therefore only those x0's could be used.
sns.violinplot(pd.Series([mm_indiv_logistic_cont_popt_pcov[i][0][0] for i in range(len(mm_indiv_logistic_cont_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax1, color='lightgrey', alpha=1)

#sns.swarmplot(pd.Series([mm_indiv_logistic_cont_popt_pcov[i][0][1] for i in range(len(mm_indiv_logistic_cont_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax2, color='r', alpha=0.5)
sns.swarmplot(pd.Series([mm_indiv_logistic_cont_popt_pcov[i][0][1] for i in mm_list_dropna_fit_validity_df.query('growth_phase == False').index], dtype=float).dropna(inplace=False), orient='v', ax=ax2, marker='o', color='k', alpha=0.5)
sns.swarmplot(pd.Series([mm_indiv_logistic_cont_popt_pcov[i][0][1] for i in mm_list_dropna_fit_validity_df.query('growth_phase == True').index], dtype=float).dropna(inplace=False), orient='v', ax=ax2, marker='o', color='r', alpha=0.5)
sns.violinplot(pd.Series([mm_indiv_logistic_cont_popt_pcov[i][0][1] for i in range(len(mm_indiv_logistic_cont_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax2, color='lightgrey', alpha=1)

#sns.swarmplot(pd.Series([mm_indiv_logistic_cont_popt_pcov[i][0][2] for i in range(len(mm_indiv_logistic_cont_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax3, color='r', alpha=0.5)
sns.swarmplot(pd.Series([mm_indiv_logistic_cont_popt_pcov[i][0][2] for i in mm_list_dropna_fit_validity_df.query('high_plat == False').index], dtype=float).dropna(inplace=False), orient='v', ax=ax3, marker='o', color='k', alpha=0.5)
sns.swarmplot(pd.Series([mm_indiv_logistic_cont_popt_pcov[i][0][2] for i in mm_list_dropna_fit_validity_df.query('high_plat == True').index], dtype=float).dropna(inplace=False), orient='v', ax=ax3, marker='o', color='r', alpha=0.5)
sns.violinplot(pd.Series([mm_indiv_logistic_cont_popt_pcov[i][0][2] for i in range(len(mm_indiv_logistic_cont_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax3, color='lightgrey', alpha=1)

#sns.swarmplot(pd.Series([mm_indiv_logistic_cont_popt_pcov[i][0][3] for i in range(len(mm_indiv_logistic_cont_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax4, color='r', alpha=0.5)
sns.swarmplot(pd.Series([mm_indiv_logistic_cont_popt_pcov[i][0][3] for i in mm_list_dropna_fit_validity_df.query('low_plat == False').index], dtype=float).dropna(inplace=False), orient='v', ax=ax4, marker='o', color='k', alpha=0.5)
sns.swarmplot(pd.Series([mm_indiv_logistic_cont_popt_pcov[i][0][3] for i in mm_list_dropna_fit_validity_df.query('low_plat == True').index], dtype=float).dropna(inplace=False), orient='v', ax=ax4, marker='o', color='r', alpha=0.5)
sns.violinplot(pd.Series([mm_indiv_logistic_cont_popt_pcov[i][0][3] for i in range(len(mm_indiv_logistic_cont_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax4, color='lightgrey', alpha=1)
# plot the manually selected timelapses with a low plateau as black dots:
#sns.swarmplot(pd.Series([mm_indiv_logistic_cont_popt_pcov[i][0][0] for i in mm_list_dropna_fit_validity_df.query('growth_phase == True').index]).dropna(inplace=False), orient='v', ax=ax1, marker='.', color='k', alpha=0.5) # only valid growth phase timelapses could be aligned, therefore only those x0's could be used.
#sns.swarmplot(pd.Series([mm_indiv_logistic_cont_popt_pcov[i][0][1] for i in mm_list_dropna_fit_validity_df.query('growth_phase == True').index]).dropna(inplace=False), orient='v', ax=ax2, marker='.', color='k', alpha=0.5)
#sns.swarmplot(pd.Series([mm_indiv_logistic_cont_popt_pcov[i][0][2] for i in mm_list_dropna_fit_validity_df.query('high_plat == True').index]).dropna(inplace=False), orient='v', ax=ax3, marker='.', color='k', alpha=0.5)
#sns.swarmplot(pd.Series([mm_indiv_logistic_cont_popt_pcov[i][0][3] for i in mm_list_dropna_fit_validity_df.query('low_plat == True').index]).dropna(inplace=False), orient='v', ax=ax4, marker='.', color='k', alpha=0.5)

ax3.axhline(bounds_cont[0][2], c='k')
ax3.axhline(bounds_cont[1][2], c='k')
ax4.axhline(bounds_cont[0][3], c='k')
ax4.axhline(bounds_cont[1][3], c='k')
ax1.set_xlabel(r'x$_\theta$')
ax2.set_xlabel('logistic growth rate "k"')
ax3.set_xlabel('upper_plateau')
ax4.set_xlabel('lower_plateau')
#plt.suptitle('cont', fontsize=18, x=0.5, y=1.05)
plt.tight_layout()
plt.savefig(Path.joinpath(output_dir, r"mm_cont_popt_distrib.tif"), dpi=dpi_graph)
#print(bounds_cont)

plt.clf()
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(width_2col,height_2col))
# plot individual parameters of all timelapses as red:
#sns.swarmplot(pd.Series([mm_indiv_logistic_angle_popt_pcov[i][0][0] for i in range(len(mm_indiv_logistic_angle_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax1, color='r', alpha=0.5)
sns.swarmplot(pd.Series([mm_indiv_logistic_angle_popt_pcov[i][0][0] for i in mm_list_dropna_angle_fit_validity_df.query('growth_phase == False').index], dtype=float).dropna(inplace=False), orient='v', ax=ax1, marker='o', color='k', alpha=0.5) # only valid growth phase timelapses could be aligned, therefore only those x0's could be used.
sns.swarmplot(pd.Series([mm_indiv_logistic_angle_popt_pcov[i][0][0] for i in mm_list_dropna_angle_fit_validity_df.query('growth_phase == True').index], dtype=float).dropna(inplace=False), orient='v', ax=ax1, marker='o', color='r', alpha=0.5) # only valid growth phase timelapses could be aligned, therefore only those x0's could be used.
sns.violinplot(pd.Series([mm_indiv_logistic_angle_popt_pcov[i][0][0] for i in range(len(mm_indiv_logistic_angle_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax1, color='lightgrey', alpha=1)

#sns.swarmplot(pd.Series([mm_indiv_logistic_angle_popt_pcov[i][0][1] for i in range(len(mm_indiv_logistic_angle_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax2, color='r', alpha=0.5)
sns.swarmplot(pd.Series([mm_indiv_logistic_angle_popt_pcov[i][0][1] for i in mm_list_dropna_angle_fit_validity_df.query('growth_phase == False').index], dtype=float).dropna(inplace=False), orient='v', ax=ax2, marker='o', color='k', alpha=0.5)
sns.swarmplot(pd.Series([mm_indiv_logistic_angle_popt_pcov[i][0][1] for i in mm_list_dropna_angle_fit_validity_df.query('growth_phase == True').index], dtype=float).dropna(inplace=False), orient='v', ax=ax2, marker='o', color='r', alpha=0.5)
sns.violinplot(pd.Series([mm_indiv_logistic_angle_popt_pcov[i][0][1] for i in range(len(mm_indiv_logistic_angle_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax2, color='lightgrey', alpha=1)

#sns.swarmplot(pd.Series([mm_indiv_logistic_angle_popt_pcov[i][0][2] for i in range(len(mm_indiv_logistic_angle_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax3, color='r', alpha=0.5)
sns.swarmplot(pd.Series([mm_indiv_logistic_angle_popt_pcov[i][0][2] for i in mm_list_dropna_angle_fit_validity_df.query('high_plat == False').index], dtype=float).dropna(inplace=False), orient='v', ax=ax3, marker='o', color='k', alpha=0.5)
sns.swarmplot(pd.Series([mm_indiv_logistic_angle_popt_pcov[i][0][2] for i in mm_list_dropna_angle_fit_validity_df.query('high_plat == True').index], dtype=float).dropna(inplace=False), orient='v', ax=ax3, marker='o', color='r', alpha=0.5)
sns.violinplot(pd.Series([mm_indiv_logistic_angle_popt_pcov[i][0][2] for i in range(len(mm_indiv_logistic_angle_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax3, color='lightgrey', alpha=1)

#sns.swarmplot(pd.Series([mm_indiv_logistic_angle_popt_pcov[i][0][3] for i in range(len(mm_indiv_logistic_angle_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax4, color='r', alpha=0.5)
sns.swarmplot(pd.Series([mm_indiv_logistic_angle_popt_pcov[i][0][3] for i in mm_list_dropna_angle_fit_validity_df.query('low_plat == False').index], dtype=float).dropna(inplace=False), orient='v', ax=ax4, marker='o', color='k', alpha=0.5)
sns.swarmplot(pd.Series([mm_indiv_logistic_angle_popt_pcov[i][0][3] for i in mm_list_dropna_angle_fit_validity_df.query('low_plat == True').index], dtype=float).dropna(inplace=False), orient='v', ax=ax4, marker='o', color='r', alpha=0.5)
sns.violinplot(pd.Series([mm_indiv_logistic_angle_popt_pcov[i][0][3] for i in range(len(mm_indiv_logistic_angle_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax4, color='lightgrey', alpha=1)
# plot the manually selected timelapses with a low plateau as black dots:
#sns.swarmplot(pd.Series([mm_indiv_logistic_angle_popt_pcov[i][0][0] for i in mm_list_dropna_angle_fit_validity_df.query('growth_phase == True').index]).dropna(inplace=False), orient='v', ax=ax1, marker='.', color='k', alpha=0.5) # only valid growth phase timelapses could be aligned, therefore only those x0's could be used.
#sns.swarmplot(pd.Series([mm_indiv_logistic_angle_popt_pcov[i][0][1] for i in mm_list_dropna_angle_fit_validity_df.query('growth_phase == True').index]).dropna(inplace=False), orient='v', ax=ax2, marker='.', color='k', alpha=0.5)
#sns.swarmplot(pd.Series([mm_indiv_logistic_angle_popt_pcov[i][0][2] for i in mm_list_dropna_angle_fit_validity_df.query('high_plat == True').index]).dropna(inplace=False), orient='v', ax=ax3, marker='.', color='k', alpha=0.5)
#sns.swarmplot(pd.Series([mm_indiv_logistic_angle_popt_pcov[i][0][3] for i in mm_list_dropna_angle_fit_validity_df.query('low_plat == True').index]).dropna(inplace=False), orient='v', ax=ax4, marker='.', color='k', alpha=0.5)

ax3.axhline(bounds_angle[0][2], c='k')
ax3.axhline(bounds_angle[1][2], c='k')
ax4.axhline(bounds_angle[0][3], c='k')
ax4.axhline(bounds_angle[1][3], c='k')
ax1.set_xlabel(r'x$_\theta$')
ax2.set_xlabel('logistic growth rate "k"')
ax3.set_xlabel('upper_plateau')
ax4.set_xlabel('lower_plateau')
#plt.suptitle('angle', fontsize=18, x=0.5, y=1.05)
plt.tight_layout()
plt.savefig(Path.joinpath(output_dir, r"mm_angle_popt_distrib.tif"), dpi=dpi_graph)
#print(bounds_angle)


# For endo-endo:
plt.clf()
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(width_2col,height_2col))
# plot individual parameters of all timelapses as blue:
#sns.swarmplot(pd.Series([nn_indiv_logistic_contnd_popt_pcov[i][0][0] for i in range(len(nn_indiv_logistic_contnd_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax1, color='b', alpha=0.5)
sns.swarmplot(pd.Series([nn_indiv_logistic_contnd_popt_pcov[i][0][0] for i in nn_list_dropna_fit_validity_df.query('growth_phase == False').index]).dropna(inplace=False), orient='v', ax=ax1, marker='o', color='k', alpha=0.5) # only valid growth phase timelapses could be aligned, therefore only those x0's could be used.
sns.swarmplot(pd.Series([nn_indiv_logistic_contnd_popt_pcov[i][0][0] for i in nn_list_dropna_fit_validity_df.query('growth_phase == True').index]).dropna(inplace=False), orient='v', ax=ax1, marker='o', color='b', alpha=0.5) # only valid growth phase timelapses could be aligned, therefore only those x0's could be used.
sns.violinplot(pd.Series([nn_indiv_logistic_contnd_popt_pcov[i][0][0] for i in range(len(nn_indiv_logistic_contnd_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax1, color='lightgrey', alpha=1)

#sns.swarmplot(pd.Series([nn_indiv_logistic_contnd_popt_pcov[i][0][1] for i in range(len(nn_indiv_logistic_contnd_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax2, color='b', alpha=0.5)
sns.swarmplot(pd.Series([nn_indiv_logistic_contnd_popt_pcov[i][0][1] for i in nn_list_dropna_fit_validity_df.query('growth_phase == False').index]).dropna(inplace=False), orient='v', ax=ax2, marker='o', color='k', alpha=0.5)
sns.swarmplot(pd.Series([nn_indiv_logistic_contnd_popt_pcov[i][0][1] for i in nn_list_dropna_fit_validity_df.query('growth_phase == True').index]).dropna(inplace=False), orient='v', ax=ax2, marker='o', color='b', alpha=0.5)
sns.violinplot(pd.Series([nn_indiv_logistic_contnd_popt_pcov[i][0][1] for i in range(len(nn_indiv_logistic_contnd_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax2, color='lightgrey', alpha=1)

#sns.swarmplot(pd.Series([nn_indiv_logistic_contnd_popt_pcov[i][0][2] for i in range(len(nn_indiv_logistic_contnd_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax3, color='b', alpha=0.5)
sns.swarmplot(pd.Series([nn_indiv_logistic_contnd_popt_pcov[i][0][2] for i in nn_list_dropna_fit_validity_df.query('high_plat == False').index]).dropna(inplace=False), orient='v', ax=ax3, marker='o', color='k', alpha=0.5)
sns.swarmplot(pd.Series([nn_indiv_logistic_contnd_popt_pcov[i][0][2] for i in nn_list_dropna_fit_validity_df.query('high_plat == True').index]).dropna(inplace=False), orient='v', ax=ax3, marker='o', color='b', alpha=0.5)
sns.violinplot(pd.Series([nn_indiv_logistic_contnd_popt_pcov[i][0][2] for i in range(len(nn_indiv_logistic_contnd_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax3, color='lightgrey', alpha=1)

#sns.swarmplot(pd.Series([nn_indiv_logistic_contnd_popt_pcov[i][0][3] for i in range(len(nn_indiv_logistic_contnd_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax4, color='b', alpha=0.5)
sns.swarmplot(pd.Series([nn_indiv_logistic_contnd_popt_pcov[i][0][3] for i in nn_list_dropna_fit_validity_df.query('low_plat == False').index]).dropna(inplace=False), orient='v', ax=ax4, marker='o', color='k', alpha=0.5)
sns.swarmplot(pd.Series([nn_indiv_logistic_contnd_popt_pcov[i][0][3] for i in nn_list_dropna_fit_validity_df.query('low_plat == True').index]).dropna(inplace=False), orient='v', ax=ax4, marker='o', color='b', alpha=0.5)
sns.violinplot(pd.Series([nn_indiv_logistic_contnd_popt_pcov[i][0][3] for i in range(len(nn_indiv_logistic_contnd_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax4, color='lightgrey', alpha=1)
# plot the manually selected timelapses with a low plateau as black dots:
#sns.swarmplot(pd.Series([nn_indiv_logistic_contnd_popt_pcov[i][0][0] for i in nn_list_dropna_fit_validity_df.query('growth_phase == True').index]).dropna(inplace=False), orient='v', ax=ax1, marker='.', color='k', alpha=0.5) # only valid growth phase timelapses could be aligned, therefore only those x0's could be used.
#sns.swarmplot(pd.Series([nn_indiv_logistic_contnd_popt_pcov[i][0][1] for i in nn_list_dropna_fit_validity_df.query('growth_phase == True').index]).dropna(inplace=False), orient='v', ax=ax2, marker='.', color='k', alpha=0.5)
#sns.swarmplot(pd.Series([nn_indiv_logistic_contnd_popt_pcov[i][0][2] for i in nn_list_dropna_fit_validity_df.query('high_plat == True').index]).dropna(inplace=False), orient='v', ax=ax3, marker='.', color='k', alpha=0.5)
#sns.swarmplot(pd.Series([nn_indiv_logistic_contnd_popt_pcov[i][0][3] for i in nn_list_dropna_fit_validity_df.query('low_plat == True').index]).dropna(inplace=False), orient='v', ax=ax4, marker='.', color='k', alpha=0.5)

ax3.axhline(bounds_contnd[0][2], c='k')
ax3.axhline(bounds_contnd[1][2], c='k')
ax4.axhline(bounds_contnd[0][3], c='k')
ax4.axhline(bounds_contnd[1][3], c='k')
ax1.set_xlabel(r'x$_\theta$')
ax2.set_xlabel('logistic growth rate "k"')
ax3.set_xlabel('upper_plateau')
ax4.set_xlabel('lower_plateau')
#plt.suptitle('contnd', fontsize=18, x=0.5, y=1.05)
plt.tight_layout()
plt.savefig(Path.joinpath(output_dir, r"nn_contnd_popt_distrib.tif"), dpi=dpi_graph)
#print(bounds_contnd)

plt.clf()
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(width_2col,height_2col))
# plot individual parameters of all timelapses as blue:
#sns.swarmplot([nn_indiv_logistic_cont_popt_pcov[i][0][0] for i in range(len(nn_indiv_logistic_cont_popt_pcov))], orient='v', ax=ax1, color='b', alpha=0.5)
sns.swarmplot(pd.Series([nn_indiv_logistic_cont_popt_pcov[i][0][0] for i in nn_list_dropna_fit_validity_df.query('growth_phase == False').index]).dropna(inplace=False), orient='v', ax=ax1, marker='o', color='k', alpha=0.5) # only valid growth phase timelapses could be aligned, therefore only those x0's could be used.
sns.swarmplot(pd.Series([nn_indiv_logistic_cont_popt_pcov[i][0][0] for i in nn_list_dropna_fit_validity_df.query('growth_phase == True').index]).dropna(inplace=False), orient='v', ax=ax1, marker='o', color='b', alpha=0.5) # only valid growth phase timelapses could be aligned, therefore only those x0's could be used.
sns.violinplot([nn_indiv_logistic_cont_popt_pcov[i][0][0] for i in range(len(nn_indiv_logistic_cont_popt_pcov))], orient='v', ax=ax1, color='lightgrey', alpha=1)

#sns.swarmplot([nn_indiv_logistic_cont_popt_pcov[i][0][1] for i in range(len(nn_indiv_logistic_cont_popt_pcov))], orient='v', ax=ax2, color='b', alpha=0.5)
sns.swarmplot(pd.Series([nn_indiv_logistic_cont_popt_pcov[i][0][1] for i in nn_list_dropna_fit_validity_df.query('growth_phase == False').index]).dropna(inplace=False), orient='v', ax=ax2, marker='o', color='k', alpha=0.5)
sns.swarmplot(pd.Series([nn_indiv_logistic_cont_popt_pcov[i][0][1] for i in nn_list_dropna_fit_validity_df.query('growth_phase == True').index]).dropna(inplace=False), orient='v', ax=ax2, marker='o', color='b', alpha=0.5)
sns.violinplot([nn_indiv_logistic_cont_popt_pcov[i][0][1] for i in range(len(nn_indiv_logistic_cont_popt_pcov))], orient='v', ax=ax2, color='lightgrey', alpha=1)

#sns.swarmplot([nn_indiv_logistic_cont_popt_pcov[i][0][2] for i in range(len(nn_indiv_logistic_cont_popt_pcov))], orient='v', ax=ax3, color='b', alpha=0.5)
sns.swarmplot(pd.Series([nn_indiv_logistic_cont_popt_pcov[i][0][2] for i in nn_list_dropna_fit_validity_df.query('high_plat == False').index]).dropna(inplace=False), orient='v', ax=ax3, marker='o', color='k', alpha=0.5)
sns.swarmplot(pd.Series([nn_indiv_logistic_cont_popt_pcov[i][0][2] for i in nn_list_dropna_fit_validity_df.query('high_plat == True').index]).dropna(inplace=False), orient='v', ax=ax3, marker='o', color='b', alpha=0.5)
sns.violinplot([nn_indiv_logistic_cont_popt_pcov[i][0][2] for i in range(len(nn_indiv_logistic_cont_popt_pcov))], orient='v', ax=ax3, color='lightgrey', alpha=1)

#sns.swarmplot([nn_indiv_logistic_cont_popt_pcov[i][0][3] for i in range(len(nn_indiv_logistic_cont_popt_pcov))], orient='v', ax=ax4, color='b', alpha=0.5)
sns.swarmplot(pd.Series([nn_indiv_logistic_cont_popt_pcov[i][0][3] for i in nn_list_dropna_fit_validity_df.query('low_plat == False').index]).dropna(inplace=False), orient='v', ax=ax4, marker='o', color='k', alpha=0.5)
sns.swarmplot(pd.Series([nn_indiv_logistic_cont_popt_pcov[i][0][3] for i in nn_list_dropna_fit_validity_df.query('low_plat == True').index]).dropna(inplace=False), orient='v', ax=ax4, marker='o', color='b', alpha=0.5)
sns.violinplot([nn_indiv_logistic_cont_popt_pcov[i][0][3] for i in range(len(nn_indiv_logistic_cont_popt_pcov))], orient='v', ax=ax4, color='lightgrey', alpha=1)
# plot the manually selected timelapses with a low plateau as black dots:
#sns.swarmplot(pd.Series([nn_indiv_logistic_cont_popt_pcov[i][0][0] for i in nn_list_dropna_fit_validity_df.query('growth_phase == True').index]).dropna(inplace=False), orient='v', ax=ax1, marker='.', color='k', alpha=0.5) # only valid growth phase timelapses could be aligned, therefore only those x0's could be used.
#sns.swarmplot(pd.Series([nn_indiv_logistic_cont_popt_pcov[i][0][1] for i in nn_list_dropna_fit_validity_df.query('growth_phase == True').index]).dropna(inplace=False), orient='v', ax=ax2, marker='.', color='k', alpha=0.5)
#sns.swarmplot(pd.Series([nn_indiv_logistic_cont_popt_pcov[i][0][2] for i in nn_list_dropna_fit_validity_df.query('high_plat == True').index]).dropna(inplace=False), orient='v', ax=ax3, marker='.', color='k', alpha=0.5)
#sns.swarmplot(pd.Series([nn_indiv_logistic_cont_popt_pcov[i][0][3] for i in nn_list_dropna_fit_validity_df.query('low_plat == True').index]).dropna(inplace=False), orient='v', ax=ax4, marker='.', color='k', alpha=0.5)

ax3.axhline(bounds_cont[0][2], c='k')
ax3.axhline(bounds_cont[1][2], c='k')
ax4.axhline(bounds_cont[0][3], c='k')
ax4.axhline(bounds_cont[1][3], c='k')
ax1.set_xlabel(r'x$_\theta$')
ax2.set_xlabel('logistic growth rate "k"')
ax3.set_xlabel('upper_plateau')
ax4.set_xlabel('lower_plateau')
#plt.suptitle('cont', fontsize=18, x=0.5, y=1.05)
plt.tight_layout()
plt.savefig(Path.joinpath(output_dir, r"nn_cont_popt_distrib.tif"), dpi=dpi_graph)
#print(bounds_cont)

plt.clf()
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(width_2col,height_2col))
# plot individual parameters of all timelapses as blue:
#sns.swarmplot(pd.Series([nn_indiv_logistic_angle_popt_pcov[i][0][0] for i in range(len(nn_indiv_logistic_angle_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax1, color='b', alpha=0.5)
sns.swarmplot(pd.Series([nn_indiv_logistic_angle_popt_pcov[i][0][0] for i in nn_list_dropna_angle_fit_validity_df.query('growth_phase == False').index]).dropna(inplace=False), orient='v', ax=ax1, marker='o', color='k', alpha=0.5) # only valid growth phase timelapses could be aligned, therefore only those x0's could be used.
sns.swarmplot(pd.Series([nn_indiv_logistic_angle_popt_pcov[i][0][0] for i in nn_list_dropna_angle_fit_validity_df.query('growth_phase == True').index]).dropna(inplace=False), orient='v', ax=ax1, marker='o', color='b', alpha=0.5) # only valid growth phase timelapses could be aligned, therefore only those x0's could be used.
sns.violinplot(pd.Series([nn_indiv_logistic_angle_popt_pcov[i][0][0] for i in range(len(nn_indiv_logistic_angle_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax1, color='lightgrey', alpha=1)

#sns.swarmplot(pd.Series([nn_indiv_logistic_angle_popt_pcov[i][0][1] for i in range(len(nn_indiv_logistic_angle_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax2, color='b', alpha=0.5)
sns.swarmplot(pd.Series([nn_indiv_logistic_angle_popt_pcov[i][0][1] for i in nn_list_dropna_angle_fit_validity_df.query('growth_phase == False').index]).dropna(inplace=False), orient='v', ax=ax2, marker='o', color='k', alpha=0.5)
sns.swarmplot(pd.Series([nn_indiv_logistic_angle_popt_pcov[i][0][1] for i in nn_list_dropna_angle_fit_validity_df.query('growth_phase == True').index]).dropna(inplace=False), orient='v', ax=ax2, marker='o', color='b', alpha=0.5)
sns.violinplot(pd.Series([nn_indiv_logistic_angle_popt_pcov[i][0][1] for i in range(len(nn_indiv_logistic_angle_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax2, color='lightgrey', alpha=1)

#sns.swarmplot(pd.Series([nn_indiv_logistic_angle_popt_pcov[i][0][2] for i in range(len(nn_indiv_logistic_angle_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax3, color='b', alpha=0.5)
sns.swarmplot(pd.Series([nn_indiv_logistic_angle_popt_pcov[i][0][2] for i in nn_list_dropna_angle_fit_validity_df.query('high_plat == False').index]).dropna(inplace=False), orient='v', ax=ax3, marker='o', color='k', alpha=0.5)
sns.swarmplot(pd.Series([nn_indiv_logistic_angle_popt_pcov[i][0][2] for i in nn_list_dropna_angle_fit_validity_df.query('high_plat == True').index]).dropna(inplace=False), orient='v', ax=ax3, marker='o', color='b', alpha=0.5)
sns.violinplot(pd.Series([nn_indiv_logistic_angle_popt_pcov[i][0][2] for i in range(len(nn_indiv_logistic_angle_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax3, color='lightgrey', alpha=1)

#sns.swarmplot(pd.Series([nn_indiv_logistic_angle_popt_pcov[i][0][3] for i in range(len(nn_indiv_logistic_angle_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax4, color='b', alpha=0.5)
sns.swarmplot(pd.Series([nn_indiv_logistic_angle_popt_pcov[i][0][3] for i in nn_list_dropna_angle_fit_validity_df.query('low_plat == False').index]).dropna(inplace=False), orient='v', ax=ax4, marker='o', color='k', alpha=0.5)
sns.swarmplot(pd.Series([nn_indiv_logistic_angle_popt_pcov[i][0][3] for i in nn_list_dropna_angle_fit_validity_df.query('low_plat == True').index]).dropna(inplace=False), orient='v', ax=ax4, marker='o', color='b', alpha=0.5)
sns.violinplot(pd.Series([nn_indiv_logistic_angle_popt_pcov[i][0][3] for i in range(len(nn_indiv_logistic_angle_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax4, color='lightgrey', alpha=1)
# plot the manually selected timelapses with a low plateau as black dots:
#sns.swarmplot(pd.Series([nn_indiv_logistic_angle_popt_pcov[i][0][0] for i in nn_list_dropna_angle_fit_validity_df.query('growth_phase == True').index]).dropna(inplace=False), orient='v', ax=ax1, marker='.', color='k', alpha=0.5) # only valid growth phase timelapses could be aligned, therefore only those x0's could be used.
#sns.swarmplot(pd.Series([nn_indiv_logistic_angle_popt_pcov[i][0][1] for i in nn_list_dropna_angle_fit_validity_df.query('growth_phase == True').index]).dropna(inplace=False), orient='v', ax=ax2, marker='.', color='k', alpha=0.5)
#sns.swarmplot(pd.Series([nn_indiv_logistic_angle_popt_pcov[i][0][2] for i in nn_list_dropna_angle_fit_validity_df.query('high_plat == True').index]).dropna(inplace=False), orient='v', ax=ax3, marker='.', color='k', alpha=0.5)
#sns.swarmplot(pd.Series([nn_indiv_logistic_angle_popt_pcov[i][0][3] for i in nn_list_dropna_angle_fit_validity_df.query('low_plat == True').index]).dropna(inplace=False), orient='v', ax=ax4, marker='.', color='k', alpha=0.5)

ax3.axhline(bounds_angle[0][2], c='k')
ax3.axhline(bounds_angle[1][2], c='k')
ax4.axhline(bounds_angle[0][3], c='k')
ax4.axhline(bounds_angle[1][3], c='k')
ax1.set_xlabel(r'x$_\theta$')
ax2.set_xlabel('logistic growth rate "k"')
ax3.set_xlabel('upper_plateau')
ax4.set_xlabel('lower_plateau')
#plt.suptitle('angle', fontsize=18, x=0.5, y=1.05)
plt.tight_layout()
plt.savefig(Path.joinpath(output_dir, r"nn_angle_popt_distrib.tif"), dpi=dpi_graph)
#print(bounds_angle)


# For ecto-ecto_Ccad-morpholino injected / eeCcadMO:
plt.clf()
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(width_2col,height_2col))
# plot individual parameters of all timelapses as cyan:
#sns.swarmplot(pd.Series([eeCcadMO_indiv_logistic_contnd_popt_pcov[i][0][0] for i in range(len(eeCcadMO_indiv_logistic_contnd_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax1, color='c', alpha=0.5)
sns.swarmplot(pd.Series([eeCcadMO_indiv_logistic_contnd_popt_pcov[i][0][0] for i in eeCcadMO_list_dropna_fit_validity_df.query('growth_phase == False').index]).dropna(inplace=False), orient='v', ax=ax1, marker='o', color='k', alpha=0.5) # only valid growth phase timelapses could be aligned, therefore only those x0's could be used.
sns.swarmplot(pd.Series([eeCcadMO_indiv_logistic_contnd_popt_pcov[i][0][0] for i in eeCcadMO_list_dropna_fit_validity_df.query('growth_phase == True').index]).dropna(inplace=False), orient='v', ax=ax1, marker='o', color='c', alpha=0.5) # only valid growth phase timelapses could be aligned, therefore only those x0's could be used.
sns.violinplot(pd.Series([eeCcadMO_indiv_logistic_contnd_popt_pcov[i][0][0] for i in range(len(eeCcadMO_indiv_logistic_contnd_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax1, color='lightgrey', alpha=1)

#sns.swarmplot(pd.Series([eeCcadMO_indiv_logistic_contnd_popt_pcov[i][0][1] for i in range(len(eeCcadMO_indiv_logistic_contnd_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax2, color='c', alpha=0.5)
sns.swarmplot(pd.Series([eeCcadMO_indiv_logistic_contnd_popt_pcov[i][0][1] for i in eeCcadMO_list_dropna_fit_validity_df.query('growth_phase == False').index]).dropna(inplace=False), orient='v', ax=ax2, marker='o', color='k', alpha=0.5)
sns.swarmplot(pd.Series([eeCcadMO_indiv_logistic_contnd_popt_pcov[i][0][1] for i in eeCcadMO_list_dropna_fit_validity_df.query('growth_phase == True').index]).dropna(inplace=False), orient='v', ax=ax2, marker='o', color='c', alpha=0.5)
sns.violinplot(pd.Series([eeCcadMO_indiv_logistic_contnd_popt_pcov[i][0][1] for i in range(len(eeCcadMO_indiv_logistic_contnd_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax2, color='lightgrey', alpha=1)

#sns.swarmplot(pd.Series([eeCcadMO_indiv_logistic_contnd_popt_pcov[i][0][2] for i in range(len(eeCcadMO_indiv_logistic_contnd_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax3, color='c', alpha=0.5)
sns.swarmplot(pd.Series([eeCcadMO_indiv_logistic_contnd_popt_pcov[i][0][2] for i in eeCcadMO_list_dropna_fit_validity_df.query('high_plat == False').index]).dropna(inplace=False), orient='v', ax=ax3, marker='o', color='k', alpha=0.5)
sns.swarmplot(pd.Series([eeCcadMO_indiv_logistic_contnd_popt_pcov[i][0][2] for i in eeCcadMO_list_dropna_fit_validity_df.query('high_plat == True').index]).dropna(inplace=False), orient='v', ax=ax3, marker='o', color='c', alpha=0.5)
sns.violinplot(pd.Series([eeCcadMO_indiv_logistic_contnd_popt_pcov[i][0][2] for i in range(len(eeCcadMO_indiv_logistic_contnd_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax3, color='lightgrey', alpha=1)

#sns.swarmplot(pd.Series([eeCcadMO_indiv_logistic_contnd_popt_pcov[i][0][3] for i in range(len(eeCcadMO_indiv_logistic_contnd_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax4, color='c', alpha=0.5)
sns.swarmplot(pd.Series([eeCcadMO_indiv_logistic_contnd_popt_pcov[i][0][3] for i in eeCcadMO_list_dropna_fit_validity_df.query('low_plat == False').index]).dropna(inplace=False), orient='v', ax=ax4, marker='o', color='k', alpha=0.5)
sns.swarmplot(pd.Series([eeCcadMO_indiv_logistic_contnd_popt_pcov[i][0][3] for i in eeCcadMO_list_dropna_fit_validity_df.query('low_plat == True').index]).dropna(inplace=False), orient='v', ax=ax4, marker='o', color='c', alpha=0.5)
sns.violinplot(pd.Series([eeCcadMO_indiv_logistic_contnd_popt_pcov[i][0][3] for i in range(len(eeCcadMO_indiv_logistic_contnd_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax4, color='lightgrey', alpha=1)
# plot the manually selected timelapses with a low plateau as black dots:
#sns.swarmplot(pd.Series([eeCcadMO_indiv_logistic_contnd_popt_pcov[i][0][0] for i in eeCcadMO_list_dropna_fit_validity_df.query('growth_phase == True').index]).dropna(inplace=False), orient='v', ax=ax1, marker='.', color='k', alpha=0.5) # only valid growth phase timelapses could be aligned, therefore only those x0's could be used.
#sns.swarmplot(pd.Series([eeCcadMO_indiv_logistic_contnd_popt_pcov[i][0][1] for i in eeCcadMO_list_dropna_fit_validity_df.query('growth_phase == True').index]).dropna(inplace=False), orient='v', ax=ax2, marker='.', color='k', alpha=0.5)
#sns.swarmplot(pd.Series([eeCcadMO_indiv_logistic_contnd_popt_pcov[i][0][2] for i in eeCcadMO_list_dropna_fit_validity_df.query('high_plat == True').index]).dropna(inplace=False), orient='v', ax=ax3, marker='.', color='k', alpha=0.5)
#sns.swarmplot(pd.Series([eeCcadMO_indiv_logistic_contnd_popt_pcov[i][0][3] for i in eeCcadMO_list_dropna_fit_validity_df.query('low_plat == True').index]).dropna(inplace=False), orient='v', ax=ax4, marker='.', color='k', alpha=0.5)

ax3.axhline(bounds_contnd[0][2], c='k')
ax3.axhline(bounds_contnd[1][2], c='k')
ax4.axhline(bounds_contnd[0][3], c='k')
ax4.axhline(bounds_contnd[1][3], c='k')
ax1.set_xlabel(r'x$_\theta$')
ax2.set_xlabel('logistic growth rate "k"')
ax3.set_xlabel('upper_plateau')
ax4.set_xlabel('lower_plateau')
#plt.suptitle('contnd', fontsize=18, x=0.5, y=1.05)
plt.tight_layout()
plt.savefig(Path.joinpath(output_dir, r"eeCcadMO_contnd_popt_distrib.tif"), dpi=dpi_graph)
#print(bounds_contnd)

plt.clf()
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(width_2col,height_2col))
# plot individual parameters of all timelapses as cyan:
#sns.swarmplot(pd.Series([eeCcadMO_indiv_logistic_cont_popt_pcov[i][0][0] for i in range(len(eeCcadMO_indiv_logistic_cont_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax1, color='c', alpha=0.5)
sns.swarmplot(pd.Series([eeCcadMO_indiv_logistic_cont_popt_pcov[i][0][0] for i in eeCcadMO_list_dropna_fit_validity_df.query('growth_phase == False').index]).dropna(inplace=False), orient='v', ax=ax1, marker='o', color='k', alpha=0.5) # only valid growth phase timelapses could be aligned, therefore only those x0's could be used.
sns.swarmplot(pd.Series([eeCcadMO_indiv_logistic_cont_popt_pcov[i][0][0] for i in eeCcadMO_list_dropna_fit_validity_df.query('growth_phase == True').index]).dropna(inplace=False), orient='v', ax=ax1, marker='o', color='c', alpha=0.5) # only valid growth phase timelapses could be aligned, therefore only those x0's could be used.
sns.violinplot(pd.Series([eeCcadMO_indiv_logistic_cont_popt_pcov[i][0][0] for i in range(len(eeCcadMO_indiv_logistic_cont_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax1, color='lightgrey', alpha=1)

#sns.swarmplot(pd.Series([eeCcadMO_indiv_logistic_cont_popt_pcov[i][0][1] for i in range(len(eeCcadMO_indiv_logistic_cont_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax2, color='c', alpha=0.5)
sns.swarmplot(pd.Series([eeCcadMO_indiv_logistic_cont_popt_pcov[i][0][1] for i in eeCcadMO_list_dropna_fit_validity_df.query('growth_phase == False').index]).dropna(inplace=False), orient='v', ax=ax2, marker='o', color='k', alpha=0.5)
sns.swarmplot(pd.Series([eeCcadMO_indiv_logistic_cont_popt_pcov[i][0][1] for i in eeCcadMO_list_dropna_fit_validity_df.query('growth_phase == True').index]).dropna(inplace=False), orient='v', ax=ax2, marker='o', color='c', alpha=0.5)
sns.violinplot(pd.Series([eeCcadMO_indiv_logistic_cont_popt_pcov[i][0][1] for i in range(len(eeCcadMO_indiv_logistic_cont_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax2, color='lightgrey', alpha=1)

#sns.swarmplot(pd.Series([eeCcadMO_indiv_logistic_cont_popt_pcov[i][0][2] for i in range(len(eeCcadMO_indiv_logistic_cont_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax3, color='c', alpha=0.5)
sns.swarmplot(pd.Series([eeCcadMO_indiv_logistic_cont_popt_pcov[i][0][2] for i in eeCcadMO_list_dropna_fit_validity_df.query('high_plat == False').index]).dropna(inplace=False), orient='v', ax=ax3, marker='o', color='k', alpha=0.5)
sns.swarmplot(pd.Series([eeCcadMO_indiv_logistic_cont_popt_pcov[i][0][2] for i in eeCcadMO_list_dropna_fit_validity_df.query('high_plat == True').index]).dropna(inplace=False), orient='v', ax=ax3, marker='o', color='c', alpha=0.5)
sns.violinplot(pd.Series([eeCcadMO_indiv_logistic_cont_popt_pcov[i][0][2] for i in range(len(eeCcadMO_indiv_logistic_cont_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax3, color='lightgrey', alpha=1)

#sns.swarmplot(pd.Series([eeCcadMO_indiv_logistic_cont_popt_pcov[i][0][3] for i in range(len(eeCcadMO_indiv_logistic_cont_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax4, color='c', alpha=0.5)
sns.swarmplot(pd.Series([eeCcadMO_indiv_logistic_cont_popt_pcov[i][0][3] for i in eeCcadMO_list_dropna_fit_validity_df.query('low_plat == False').index]).dropna(inplace=False), orient='v', ax=ax4, marker='o', color='k', alpha=0.5)
sns.swarmplot(pd.Series([eeCcadMO_indiv_logistic_cont_popt_pcov[i][0][3] for i in eeCcadMO_list_dropna_fit_validity_df.query('low_plat == True').index]).dropna(inplace=False), orient='v', ax=ax4, marker='o', color='c', alpha=0.5)
sns.violinplot(pd.Series([eeCcadMO_indiv_logistic_cont_popt_pcov[i][0][3] for i in range(len(eeCcadMO_indiv_logistic_cont_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax4, color='lightgrey', alpha=1)
# plot the manually selected timelapses with a low plateau as black dots:
#sns.swarmplot(pd.Series([eeCcadMO_indiv_logistic_cont_popt_pcov[i][0][0] for i in eeCcadMO_list_dropna_fit_validity_df.query('growth_phase == True').index]).dropna(inplace=False), orient='v', ax=ax1, marker='.', color='k', alpha=0.5) # only valid growth phase timelapses could be aligned, therefore only those x0's could be used.
#sns.swarmplot(pd.Series([eeCcadMO_indiv_logistic_cont_popt_pcov[i][0][1] for i in eeCcadMO_list_dropna_fit_validity_df.query('growth_phase == True').index]).dropna(inplace=False), orient='v', ax=ax2, marker='.', color='k', alpha=0.5)
#sns.swarmplot(pd.Series([eeCcadMO_indiv_logistic_cont_popt_pcov[i][0][2] for i in eeCcadMO_list_dropna_fit_validity_df.query('high_plat == True').index]).dropna(inplace=False), orient='v', ax=ax3, marker='.', color='k', alpha=0.5)
#sns.swarmplot(pd.Series([eeCcadMO_indiv_logistic_cont_popt_pcov[i][0][3] for i in eeCcadMO_list_dropna_fit_validity_df.query('low_plat == True').index]).dropna(inplace=False), orient='v', ax=ax4, marker='.', color='k', alpha=0.5)

ax3.axhline(bounds_cont[0][2], c='k')
ax3.axhline(bounds_cont[1][2], c='k')
ax4.axhline(bounds_cont[0][3], c='k')
ax4.axhline(bounds_cont[1][3], c='k')
ax1.set_xlabel(r'x$_\theta$')
ax2.set_xlabel('logistic growth rate "k"')
ax3.set_xlabel('upper_plateau')
ax4.set_xlabel('lower_plateau')
#plt.suptitle('cont', fontsize=18, x=0.5, y=1.05)
plt.tight_layout()
plt.savefig(Path.joinpath(output_dir, r"eeCcadMO_cont_popt_distrib.tif"), dpi=dpi_graph)
#print(bounds_cont)

plt.clf()
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(width_2col,height_2col))
# plot individual parameters of all timelapses as cyan:
#sns.swarmplot(pd.Series([eeCcadMO_indiv_logistic_angle_popt_pcov[i][0][0] for i in range(len(eeCcadMO_indiv_logistic_angle_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax1, color='c', alpha=0.5)
sns.swarmplot(pd.Series([eeCcadMO_indiv_logistic_angle_popt_pcov[i][0][0] for i in eeCcadMO_list_dropna_angle_fit_validity_df.query('growth_phase == False').index]).dropna(inplace=False), orient='v', ax=ax1, marker='o', color='k', alpha=0.5) # only valid growth phase timelapses could be aligned, therefore only those x0's could be used.
sns.swarmplot(pd.Series([eeCcadMO_indiv_logistic_angle_popt_pcov[i][0][0] for i in eeCcadMO_list_dropna_angle_fit_validity_df.query('growth_phase == True').index]).dropna(inplace=False), orient='v', ax=ax1, marker='o', color='c', alpha=0.5) # only valid growth phase timelapses could be aligned, therefore only those x0's could be used.
sns.violinplot(pd.Series([eeCcadMO_indiv_logistic_angle_popt_pcov[i][0][0] for i in range(len(eeCcadMO_indiv_logistic_angle_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax1, color='lightgrey', alpha=1)

#sns.swarmplot(pd.Series([eeCcadMO_indiv_logistic_angle_popt_pcov[i][0][1] for i in range(len(eeCcadMO_indiv_logistic_angle_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax2, color='c', alpha=0.5)
sns.swarmplot(pd.Series([eeCcadMO_indiv_logistic_angle_popt_pcov[i][0][1] for i in eeCcadMO_list_dropna_angle_fit_validity_df.query('growth_phase == False').index]).dropna(inplace=False), orient='v', ax=ax2, marker='o', color='k', alpha=0.5)
sns.swarmplot(pd.Series([eeCcadMO_indiv_logistic_angle_popt_pcov[i][0][1] for i in eeCcadMO_list_dropna_angle_fit_validity_df.query('growth_phase == True').index]).dropna(inplace=False), orient='v', ax=ax2, marker='o', color='c', alpha=0.5)
sns.violinplot(pd.Series([eeCcadMO_indiv_logistic_angle_popt_pcov[i][0][1] for i in range(len(eeCcadMO_indiv_logistic_angle_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax2, color='lightgrey', alpha=1)

#sns.swarmplot(pd.Series([eeCcadMO_indiv_logistic_angle_popt_pcov[i][0][2] for i in range(len(eeCcadMO_indiv_logistic_angle_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax3, color='c', alpha=0.5)
sns.swarmplot(pd.Series([eeCcadMO_indiv_logistic_angle_popt_pcov[i][0][2] for i in eeCcadMO_list_dropna_angle_fit_validity_df.query('high_plat == False').index]).dropna(inplace=False), orient='v', ax=ax3, marker='o', color='k', alpha=0.5)
sns.swarmplot(pd.Series([eeCcadMO_indiv_logistic_angle_popt_pcov[i][0][2] for i in eeCcadMO_list_dropna_angle_fit_validity_df.query('high_plat == True').index]).dropna(inplace=False), orient='v', ax=ax3, marker='o', color='c', alpha=0.5)
sns.violinplot(pd.Series([eeCcadMO_indiv_logistic_angle_popt_pcov[i][0][2] for i in range(len(eeCcadMO_indiv_logistic_angle_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax3, color='lightgrey', alpha=1)

#sns.swarmplot(pd.Series([eeCcadMO_indiv_logistic_angle_popt_pcov[i][0][3] for i in range(len(eeCcadMO_indiv_logistic_angle_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax4, color='c', alpha=0.5)
sns.swarmplot(pd.Series([eeCcadMO_indiv_logistic_angle_popt_pcov[i][0][3] for i in eeCcadMO_list_dropna_angle_fit_validity_df.query('low_plat == False').index]).dropna(inplace=False), orient='v', ax=ax4, marker='o', color='k', alpha=0.5)
sns.swarmplot(pd.Series([eeCcadMO_indiv_logistic_angle_popt_pcov[i][0][3] for i in eeCcadMO_list_dropna_angle_fit_validity_df.query('low_plat == True').index]).dropna(inplace=False), orient='v', ax=ax4, marker='o', color='c', alpha=0.5)
sns.violinplot(pd.Series([eeCcadMO_indiv_logistic_angle_popt_pcov[i][0][3] for i in range(len(eeCcadMO_indiv_logistic_angle_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax4, color='lightgrey', alpha=1)
# plot the manually selected timelapses with a low plateau as black dots:
#sns.swarmplot(pd.Series([eeCcadMO_indiv_logistic_angle_popt_pcov[i][0][0] for i in eeCcadMO_list_dropna_angle_fit_validity_df.query('growth_phase == True').index]).dropna(inplace=False), orient='v', ax=ax1, marker='.', color='k', alpha=0.5) # only valid growth phase timelapses could be aligned, therefore only those x0's could be used.
#sns.swarmplot(pd.Series([eeCcadMO_indiv_logistic_angle_popt_pcov[i][0][1] for i in eeCcadMO_list_dropna_angle_fit_validity_df.query('growth_phase == True').index]).dropna(inplace=False), orient='v', ax=ax2, marker='.', color='k', alpha=0.5)
#sns.swarmplot(pd.Series([eeCcadMO_indiv_logistic_angle_popt_pcov[i][0][2] for i in eeCcadMO_list_dropna_angle_fit_validity_df.query('high_plat == True').index]).dropna(inplace=False), orient='v', ax=ax3, marker='.', color='k', alpha=0.5)
#sns.swarmplot(pd.Series([eeCcadMO_indiv_logistic_angle_popt_pcov[i][0][3] for i in eeCcadMO_list_dropna_angle_fit_validity_df.query('low_plat == True').index]).dropna(inplace=False), orient='v', ax=ax4, marker='.', color='k', alpha=0.5)

ax3.axhline(bounds_angle[0][2], c='k')
ax3.axhline(bounds_angle[1][2], c='k')
ax4.axhline(bounds_angle[0][3], c='k')
ax4.axhline(bounds_angle[1][3], c='k')
ax1.set_xlabel(r'x$_\theta$')
ax2.set_xlabel('logistic growth rate "k"')
ax3.set_xlabel('upper_plateau')
ax4.set_xlabel('lower_plateau')
#plt.suptitle('angle', fontsize=18, x=0.5, y=1.05)
plt.tight_layout()
plt.savefig(Path.joinpath(output_dir, r"eeCcadMO_angle_popt_distrib.tif"), dpi=dpi_graph)
#print(bounds_angle)


# For ecto-ecto_Rcad-morpholino injected / eeRcadMO:
plt.clf()
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(width_2col,height_2col))
# plot individual parameters of all timelapses as magenta:
#sns.swarmplot(pd.Series([eeRcadMO_indiv_logistic_contnd_popt_pcov[i][0][0] for i in range(len(eeRcadMO_indiv_logistic_contnd_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax1, color='m', alpha=0.5)
sns.swarmplot(pd.Series([eeRcadMO_indiv_logistic_contnd_popt_pcov[i][0][0] for i in eeRcadMO_list_dropna_fit_validity_df.query('growth_phase == False').index], dtype=float).dropna(inplace=False), orient='v', ax=ax1, marker='o', color='k', alpha=0.5) # only valid growth phase timelapses could be aligned, therefore only those x0's could be used.
sns.swarmplot(pd.Series([eeRcadMO_indiv_logistic_contnd_popt_pcov[i][0][0] for i in eeRcadMO_list_dropna_fit_validity_df.query('growth_phase == True').index], dtype=float).dropna(inplace=False), orient='v', ax=ax1, marker='o', color='m', alpha=0.5) # only valid growth phase timelapses could be aligned, therefore only those x0's could be used.
sns.violinplot(pd.Series([eeRcadMO_indiv_logistic_contnd_popt_pcov[i][0][0] for i in range(len(eeRcadMO_indiv_logistic_contnd_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax1, color='lightgrey', alpha=1)

#sns.swarmplot(pd.Series([eeRcadMO_indiv_logistic_contnd_popt_pcov[i][0][1] for i in range(len(eeRcadMO_indiv_logistic_contnd_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax2, color='m', alpha=0.5)
sns.swarmplot(pd.Series([eeRcadMO_indiv_logistic_contnd_popt_pcov[i][0][1] for i in eeRcadMO_list_dropna_fit_validity_df.query('growth_phase == False').index], dtype=float).dropna(inplace=False), orient='v', ax=ax2, marker='o', color='k', alpha=0.5)
sns.swarmplot(pd.Series([eeRcadMO_indiv_logistic_contnd_popt_pcov[i][0][1] for i in eeRcadMO_list_dropna_fit_validity_df.query('growth_phase == True').index], dtype=float).dropna(inplace=False), orient='v', ax=ax2, marker='o', color='m', alpha=0.5)
sns.violinplot(pd.Series([eeRcadMO_indiv_logistic_contnd_popt_pcov[i][0][1] for i in range(len(eeRcadMO_indiv_logistic_contnd_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax2, color='lightgrey', alpha=1)

#sns.swarmplot(pd.Series([eeRcadMO_indiv_logistic_contnd_popt_pcov[i][0][2] for i in range(len(eeRcadMO_indiv_logistic_contnd_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax3, color='m', alpha=0.5)
sns.swarmplot(pd.Series([eeRcadMO_indiv_logistic_contnd_popt_pcov[i][0][2] for i in eeRcadMO_list_dropna_fit_validity_df.query('high_plat == False').index], dtype=float).dropna(inplace=False), orient='v', ax=ax3, marker='o', color='k', alpha=0.5)
sns.swarmplot(pd.Series([eeRcadMO_indiv_logistic_contnd_popt_pcov[i][0][2] for i in eeRcadMO_list_dropna_fit_validity_df.query('high_plat == True').index], dtype=float).dropna(inplace=False), orient='v', ax=ax3, marker='o', color='m', alpha=0.5)
sns.violinplot(pd.Series([eeRcadMO_indiv_logistic_contnd_popt_pcov[i][0][2] for i in range(len(eeRcadMO_indiv_logistic_contnd_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax3, color='lightgrey', alpha=1)

#sns.swarmplot(pd.Series([eeRcadMO_indiv_logistic_contnd_popt_pcov[i][0][3] for i in range(len(eeRcadMO_indiv_logistic_contnd_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax4, color='m', alpha=0.5)
sns.swarmplot(pd.Series([eeRcadMO_indiv_logistic_contnd_popt_pcov[i][0][3] for i in eeRcadMO_list_dropna_fit_validity_df.query('low_plat == False').index], dtype=float).dropna(inplace=False), orient='v', ax=ax4, marker='o', color='k', alpha=0.5)
sns.swarmplot(pd.Series([eeRcadMO_indiv_logistic_contnd_popt_pcov[i][0][3] for i in eeRcadMO_list_dropna_fit_validity_df.query('low_plat == True').index], dtype=float).dropna(inplace=False), orient='v', ax=ax4, marker='o', color='m', alpha=0.5)
sns.violinplot(pd.Series([eeRcadMO_indiv_logistic_contnd_popt_pcov[i][0][3] for i in range(len(eeRcadMO_indiv_logistic_contnd_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax4, color='lightgrey', alpha=1)
# plot the manually selected timelapses with a low plateau as black dots:
#sns.swarmplot(pd.Series([eeRcadMO_indiv_logistic_contnd_popt_pcov[i][0][0] for i in eeRcadMO_list_dropna_fit_validity_df.query('growth_phase == True').index]).dropna(inplace=False), orient='v', ax=ax1, marker='.', color='k', alpha=0.5) # only valid growth phase timelapses could be aligned, therefore only those x0's could be used.
#sns.swarmplot(pd.Series([eeRcadMO_indiv_logistic_contnd_popt_pcov[i][0][1] for i in eeRcadMO_list_dropna_fit_validity_df.query('growth_phase == True').index]).dropna(inplace=False), orient='v', ax=ax2, marker='.', color='k', alpha=0.5)
#sns.swarmplot(pd.Series([eeRcadMO_indiv_logistic_contnd_popt_pcov[i][0][2] for i in eeRcadMO_list_dropna_fit_validity_df.query('high_plat == True').index]).dropna(inplace=False), orient='v', ax=ax3, marker='.', color='k', alpha=0.5)
#sns.swarmplot(pd.Series([eeRcadMO_indiv_logistic_contnd_popt_pcov[i][0][3] for i in eeRcadMO_list_dropna_fit_validity_df.query('low_plat == True').index]).dropna(inplace=False), orient='v', ax=ax4, marker='.', color='k', alpha=0.5)

ax3.axhline(bounds_contnd[0][2], c='k')
ax3.axhline(bounds_contnd[1][2], c='k')
ax4.axhline(bounds_contnd[0][3], c='k')
ax4.axhline(bounds_contnd[1][3], c='k')
ax1.set_xlabel(r'x$_\theta$')
ax2.set_xlabel('logistic growth rate "k"')
ax3.set_xlabel('upper_plateau')
ax4.set_xlabel('lower_plateau')
#plt.suptitle('contnd', fontsize=18, x=0.5, y=1.05)
plt.tight_layout()
plt.savefig(Path.joinpath(output_dir, r"eeRcadMO_contnd_popt_distrib.tif"), dpi=dpi_graph)
#print(bounds_contnd)

plt.clf()
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(width_2col,height_2col))
# plot individual parameters of all timelapses as magenta:
#sns.swarmplot(pd.Series([eeRcadMO_indiv_logistic_cont_popt_pcov[i][0][0] for i in range(len(eeRcadMO_indiv_logistic_cont_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax1, color='m', alpha=0.5)
sns.swarmplot(pd.Series([eeRcadMO_indiv_logistic_cont_popt_pcov[i][0][0] for i in eeRcadMO_list_dropna_fit_validity_df.query('growth_phase == False').index], dtype=float).dropna(inplace=False), orient='v', ax=ax1, marker='o', color='k', alpha=0.5) # only valid growth phase timelapses could be aligned, therefore only those x0's could be used.
sns.swarmplot(pd.Series([eeRcadMO_indiv_logistic_cont_popt_pcov[i][0][0] for i in eeRcadMO_list_dropna_fit_validity_df.query('growth_phase == True').index], dtype=float).dropna(inplace=False), orient='v', ax=ax1, marker='o', color='m', alpha=0.5) # only valid growth phase timelapses could be aligned, therefore only those x0's could be used.
sns.violinplot(pd.Series([eeRcadMO_indiv_logistic_cont_popt_pcov[i][0][0] for i in range(len(eeRcadMO_indiv_logistic_cont_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax1, color='lightgrey', alpha=1)

#sns.swarmplot(pd.Series([eeRcadMO_indiv_logistic_cont_popt_pcov[i][0][1] for i in range(len(eeRcadMO_indiv_logistic_cont_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax2, color='m', alpha=0.5)
sns.swarmplot(pd.Series([eeRcadMO_indiv_logistic_cont_popt_pcov[i][0][1] for i in eeRcadMO_list_dropna_fit_validity_df.query('growth_phase == False').index], dtype=float).dropna(inplace=False), orient='v', ax=ax2, marker='o', color='k', alpha=0.5)
sns.swarmplot(pd.Series([eeRcadMO_indiv_logistic_cont_popt_pcov[i][0][1] for i in eeRcadMO_list_dropna_fit_validity_df.query('growth_phase == True').index], dtype=float).dropna(inplace=False), orient='v', ax=ax2, marker='o', color='m', alpha=0.5)
sns.violinplot(pd.Series([eeRcadMO_indiv_logistic_cont_popt_pcov[i][0][1] for i in range(len(eeRcadMO_indiv_logistic_cont_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax2, color='lightgrey', alpha=1)

#sns.swarmplot(pd.Series([eeRcadMO_indiv_logistic_cont_popt_pcov[i][0][2] for i in range(len(eeRcadMO_indiv_logistic_cont_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax3, color='m', alpha=0.5)
sns.swarmplot(pd.Series([eeRcadMO_indiv_logistic_cont_popt_pcov[i][0][2] for i in eeRcadMO_list_dropna_fit_validity_df.query('high_plat == False').index], dtype=float).dropna(inplace=False), orient='v', ax=ax3, marker='o', color='k', alpha=0.5)
sns.swarmplot(pd.Series([eeRcadMO_indiv_logistic_cont_popt_pcov[i][0][2] for i in eeRcadMO_list_dropna_fit_validity_df.query('high_plat == True').index], dtype=float).dropna(inplace=False), orient='v', ax=ax3, marker='o', color='m', alpha=0.5)
sns.violinplot(pd.Series([eeRcadMO_indiv_logistic_cont_popt_pcov[i][0][2] for i in range(len(eeRcadMO_indiv_logistic_cont_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax3, color='lightgrey', alpha=1)

#sns.swarmplot(pd.Series([eeRcadMO_indiv_logistic_cont_popt_pcov[i][0][3] for i in range(len(eeRcadMO_indiv_logistic_cont_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax4, color='m', alpha=0.5)
sns.swarmplot(pd.Series([eeRcadMO_indiv_logistic_cont_popt_pcov[i][0][3] for i in eeRcadMO_list_dropna_fit_validity_df.query('low_plat == False').index], dtype=float).dropna(inplace=False), orient='v', ax=ax4, marker='o', color='k', alpha=0.5)
sns.swarmplot(pd.Series([eeRcadMO_indiv_logistic_cont_popt_pcov[i][0][3] for i in eeRcadMO_list_dropna_fit_validity_df.query('low_plat == True').index], dtype=float).dropna(inplace=False), orient='v', ax=ax4, marker='o', color='m', alpha=0.5)
sns.violinplot(pd.Series([eeRcadMO_indiv_logistic_cont_popt_pcov[i][0][3] for i in range(len(eeRcadMO_indiv_logistic_cont_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax4, color='lightgrey', alpha=1)
# plot the manually selected timelapses with a low plateau as black dots:
#sns.swarmplot(pd.Series([eeRcadMO_indiv_logistic_cont_popt_pcov[i][0][0] for i in eeRcadMO_list_dropna_fit_validity_df.query('growth_phase == True').index]).dropna(inplace=False), orient='v', ax=ax1, marker='.', color='k', alpha=0.5) # only valid growth phase timelapses could be aligned, therefore only those x0's could be used.
#sns.swarmplot(pd.Series([eeRcadMO_indiv_logistic_cont_popt_pcov[i][0][1] for i in eeRcadMO_list_dropna_fit_validity_df.query('growth_phase == True').index]).dropna(inplace=False), orient='v', ax=ax2, marker='.', color='k', alpha=0.5)
#sns.swarmplot(pd.Series([eeRcadMO_indiv_logistic_cont_popt_pcov[i][0][2] for i in eeRcadMO_list_dropna_fit_validity_df.query('high_plat == True').index]).dropna(inplace=False), orient='v', ax=ax3, marker='.', color='k', alpha=0.5)
#sns.swarmplot(pd.Series([eeRcadMO_indiv_logistic_cont_popt_pcov[i][0][3] for i in eeRcadMO_list_dropna_fit_validity_df.query('low_plat == True').index]).dropna(inplace=False), orient='v', ax=ax4, marker='.', color='k', alpha=0.5)

ax3.axhline(bounds_cont[0][2], c='k')
ax3.axhline(bounds_cont[1][2], c='k')
ax4.axhline(bounds_cont[0][3], c='k')
ax4.axhline(bounds_cont[1][3], c='k')
ax1.set_xlabel(r'x$_\theta$')
ax2.set_xlabel('logistic growth rate "k"')
ax3.set_xlabel('upper_plateau')
ax4.set_xlabel('lower_plateau')
#plt.suptitle('cont', fontsize=18, x=0.5, y=1.05)
plt.tight_layout()
plt.savefig(Path.joinpath(output_dir, r"eeRcadMO_cont_popt_distrib.tif"), dpi=dpi_graph)
#print(bounds_cont)

plt.clf()
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(width_2col,height_2col))
# plot individual parameters of all timelapses as magenta:
#sns.swarmplot(pd.Series([eeRcadMO_indiv_logistic_angle_popt_pcov[i][0][0] for i in range(len(eeRcadMO_indiv_logistic_angle_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax1, color='m', alpha=0.5)
sns.swarmplot(pd.Series([eeRcadMO_indiv_logistic_angle_popt_pcov[i][0][0] for i in eeRcadMO_list_dropna_angle_fit_validity_df.query('growth_phase == False').index], dtype=float).dropna(inplace=False), orient='v', ax=ax1, marker='o', color='k', alpha=0.5) # only valid growth phase timelapses could be aligned, therefore only those x0's could be used.
sns.swarmplot(pd.Series([eeRcadMO_indiv_logistic_angle_popt_pcov[i][0][0] for i in eeRcadMO_list_dropna_angle_fit_validity_df.query('growth_phase == True').index], dtype=float).dropna(inplace=False), orient='v', ax=ax1, marker='o', color='m', alpha=0.5) # only valid growth phase timelapses could be aligned, therefore only those x0's could be used.
sns.violinplot(pd.Series([eeRcadMO_indiv_logistic_angle_popt_pcov[i][0][0] for i in range(len(eeRcadMO_indiv_logistic_angle_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax1, color='lightgrey', alpha=1)

#sns.swarmplot(pd.Series([eeRcadMO_indiv_logistic_angle_popt_pcov[i][0][1] for i in range(len(eeRcadMO_indiv_logistic_angle_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax2, color='m', alpha=0.5)
sns.swarmplot(pd.Series([eeRcadMO_indiv_logistic_angle_popt_pcov[i][0][1] for i in eeRcadMO_list_dropna_angle_fit_validity_df.query('growth_phase == False').index], dtype=float).dropna(inplace=False), orient='v', ax=ax2, marker='o', color='k', alpha=0.5)
sns.swarmplot(pd.Series([eeRcadMO_indiv_logistic_angle_popt_pcov[i][0][1] for i in eeRcadMO_list_dropna_angle_fit_validity_df.query('growth_phase == True').index], dtype=float).dropna(inplace=False), orient='v', ax=ax2, marker='o', color='m', alpha=0.5)
sns.violinplot(pd.Series([eeRcadMO_indiv_logistic_angle_popt_pcov[i][0][1] for i in range(len(eeRcadMO_indiv_logistic_angle_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax2, color='lightgrey', alpha=1)

#sns.swarmplot(pd.Series([eeRcadMO_indiv_logistic_angle_popt_pcov[i][0][2] for i in range(len(eeRcadMO_indiv_logistic_angle_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax3, color='m', alpha=0.5)
sns.swarmplot(pd.Series([eeRcadMO_indiv_logistic_angle_popt_pcov[i][0][2] for i in eeRcadMO_list_dropna_angle_fit_validity_df.query('high_plat == False').index], dtype=float).dropna(inplace=False), orient='v', ax=ax3, marker='o', color='k', alpha=0.5)
sns.swarmplot(pd.Series([eeRcadMO_indiv_logistic_angle_popt_pcov[i][0][2] for i in eeRcadMO_list_dropna_angle_fit_validity_df.query('high_plat == True').index], dtype=float).dropna(inplace=False), orient='v', ax=ax3, marker='o', color='m', alpha=0.5)
sns.violinplot(pd.Series([eeRcadMO_indiv_logistic_angle_popt_pcov[i][0][2] for i in range(len(eeRcadMO_indiv_logistic_angle_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax3, color='lightgrey', alpha=1)

#sns.swarmplot(pd.Series([eeRcadMO_indiv_logistic_angle_popt_pcov[i][0][3] for i in range(len(eeRcadMO_indiv_logistic_angle_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax4, color='m', alpha=0.5)
sns.swarmplot(pd.Series([eeRcadMO_indiv_logistic_angle_popt_pcov[i][0][3] for i in eeRcadMO_list_dropna_angle_fit_validity_df.query('low_plat == False').index], dtype=float).dropna(inplace=False), orient='v', ax=ax4, marker='o', color='k', alpha=0.5)
sns.swarmplot(pd.Series([eeRcadMO_indiv_logistic_angle_popt_pcov[i][0][3] for i in eeRcadMO_list_dropna_angle_fit_validity_df.query('low_plat == True').index], dtype=float).dropna(inplace=False), orient='v', ax=ax4, marker='o', color='m', alpha=0.5)
sns.violinplot(pd.Series([eeRcadMO_indiv_logistic_angle_popt_pcov[i][0][3] for i in range(len(eeRcadMO_indiv_logistic_angle_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax4, color='lightgrey', alpha=1)
# plot the manually selected timelapses with a low plateau as black dots:
#sns.swarmplot(pd.Series([eeRcadMO_indiv_logistic_angle_popt_pcov[i][0][0] for i in eeRcadMO_list_dropna_angle_fit_validity_df.query('growth_phase == True').index]).dropna(inplace=False), orient='v', ax=ax1, marker='.', color='k', alpha=0.5) # only valid growth phase timelapses could be aligned, therefore only those x0's could be used.
#sns.swarmplot(pd.Series([eeRcadMO_indiv_logistic_angle_popt_pcov[i][0][1] for i in eeRcadMO_list_dropna_angle_fit_validity_df.query('growth_phase == True').index]).dropna(inplace=False), orient='v', ax=ax2, marker='.', color='k', alpha=0.5)
#sns.swarmplot(pd.Series([eeRcadMO_indiv_logistic_angle_popt_pcov[i][0][2] for i in eeRcadMO_list_dropna_angle_fit_validity_df.query('high_plat == True').index]).dropna(inplace=False), orient='v', ax=ax3, marker='.', color='k', alpha=0.5)
#sns.swarmplot(pd.Series([eeRcadMO_indiv_logistic_angle_popt_pcov[i][0][3] for i in eeRcadMO_list_dropna_angle_fit_validity_df.query('low_plat == True').index]).dropna(inplace=False), orient='v', ax=ax4, marker='.', color='k', alpha=0.5)

ax3.axhline(bounds_angle[0][2], c='k')
ax3.axhline(bounds_angle[1][2], c='k')
ax4.axhline(bounds_angle[0][3], c='k')
ax4.axhline(bounds_angle[1][3], c='k')
ax1.set_xlabel(r'x$_\theta$')
ax2.set_xlabel('logistic growth rate "k"')
ax3.set_xlabel('upper_plateau')
ax4.set_xlabel('lower_plateau')
#plt.suptitle('angle', fontsize=18, x=0.5, y=1.05)
plt.tight_layout()
plt.savefig(Path.joinpath(output_dir, r"eeRcadMO_angle_popt_distrib.tif"), dpi=dpi_graph)
#print(bounds_angle)


plt.clf()


# For ecto-ecto_Rcad-morpholino + Ccad-morpholino double-injected / eeRCcadMO:
plt.clf()
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(width_2col,height_2col))
# plot individual parameters of all timelapses as yellow:
#sns.swarmplot(pd.Series([eeRCcadMO_indiv_logistic_contnd_popt_pcov[i][0][0] for i in range(len(eeRCcadMO_indiv_logistic_contnd_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax1, color='m', alpha=0.5)
sns.swarmplot(pd.Series([eeRCcadMO_indiv_logistic_contnd_popt_pcov[i][0][0] for i in eeRCcadMO_list_dropna_fit_validity_df.query('growth_phase == False').index], dtype=float).dropna(inplace=False), orient='v', ax=ax1, marker='o', color='k', alpha=0.5) # only valid growth phase timelapses could be aligned, therefore only those x0's could be used.
sns.swarmplot(pd.Series([eeRCcadMO_indiv_logistic_contnd_popt_pcov[i][0][0] for i in eeRCcadMO_list_dropna_fit_validity_df.query('growth_phase == True').index], dtype=float).dropna(inplace=False), orient='v', ax=ax1, marker='o', color='y', alpha=0.5) # only valid growth phase timelapses could be aligned, therefore only those x0's could be used.
sns.violinplot(pd.Series([eeRCcadMO_indiv_logistic_contnd_popt_pcov[i][0][0] for i in range(len(eeRCcadMO_indiv_logistic_contnd_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax1, color='lightgrey', alpha=1)

#sns.swarmplot(pd.Series([eeRCcadMO_indiv_logistic_contnd_popt_pcov[i][0][1] for i in range(len(eeRCcadMO_indiv_logistic_contnd_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax2, color='m', alpha=0.5)
sns.swarmplot(pd.Series([eeRCcadMO_indiv_logistic_contnd_popt_pcov[i][0][1] for i in eeRCcadMO_list_dropna_fit_validity_df.query('growth_phase == False').index], dtype=float).dropna(inplace=False), orient='v', ax=ax2, marker='o', color='k', alpha=0.5)
sns.swarmplot(pd.Series([eeRCcadMO_indiv_logistic_contnd_popt_pcov[i][0][1] for i in eeRCcadMO_list_dropna_fit_validity_df.query('growth_phase == True').index], dtype=float).dropna(inplace=False), orient='v', ax=ax2, marker='o', color='y', alpha=0.5)
sns.violinplot(pd.Series([eeRCcadMO_indiv_logistic_contnd_popt_pcov[i][0][1] for i in range(len(eeRCcadMO_indiv_logistic_contnd_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax2, color='lightgrey', alpha=1)

#sns.swarmplot(pd.Series([eeRCcadMO_indiv_logistic_contnd_popt_pcov[i][0][2] for i in range(len(eeRCcadMO_indiv_logistic_contnd_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax3, color='m', alpha=0.5)
sns.swarmplot(pd.Series([eeRCcadMO_indiv_logistic_contnd_popt_pcov[i][0][2] for i in eeRCcadMO_list_dropna_fit_validity_df.query('high_plat == False').index], dtype=float).dropna(inplace=False), orient='v', ax=ax3, marker='o', color='k', alpha=0.5)
sns.swarmplot(pd.Series([eeRCcadMO_indiv_logistic_contnd_popt_pcov[i][0][2] for i in eeRCcadMO_list_dropna_fit_validity_df.query('high_plat == True').index], dtype=float).dropna(inplace=False), orient='v', ax=ax3, marker='o', color='y', alpha=0.5)
sns.violinplot(pd.Series([eeRCcadMO_indiv_logistic_contnd_popt_pcov[i][0][2] for i in range(len(eeRCcadMO_indiv_logistic_contnd_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax3, color='lightgrey', alpha=1)

#sns.swarmplot(pd.Series([eeRCcadMO_indiv_logistic_contnd_popt_pcov[i][0][3] for i in range(len(eeRCcadMO_indiv_logistic_contnd_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax4, color='m', alpha=0.5)
sns.swarmplot(pd.Series([eeRCcadMO_indiv_logistic_contnd_popt_pcov[i][0][3] for i in eeRCcadMO_list_dropna_fit_validity_df.query('low_plat == False').index], dtype=float).dropna(inplace=False), orient='v', ax=ax4, marker='o', color='k', alpha=0.5)
sns.swarmplot(pd.Series([eeRCcadMO_indiv_logistic_contnd_popt_pcov[i][0][3] for i in eeRCcadMO_list_dropna_fit_validity_df.query('low_plat == True').index], dtype=float).dropna(inplace=False), orient='v', ax=ax4, marker='o', color='y', alpha=0.5)
sns.violinplot(pd.Series([eeRCcadMO_indiv_logistic_contnd_popt_pcov[i][0][3] for i in range(len(eeRCcadMO_indiv_logistic_contnd_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax4, color='lightgrey', alpha=1)
# plot the manually selected timelapses with a low plateau as black dots:
#sns.swarmplot(pd.Series([eeRCcadMO_indiv_logistic_contnd_popt_pcov[i][0][0] for i in eeRCcadMO_list_dropna_fit_validity_df.query('growth_phase == True').index]).dropna(inplace=False), orient='v', ax=ax1, marker='.', color='k', alpha=0.5) # only valid growth phase timelapses could be aligned, therefore only those x0's could be used.
#sns.swarmplot(pd.Series([eeRCcadMO_indiv_logistic_contnd_popt_pcov[i][0][1] for i in eeRCcadMO_list_dropna_fit_validity_df.query('growth_phase == True').index]).dropna(inplace=False), orient='v', ax=ax2, marker='.', color='k', alpha=0.5)
#sns.swarmplot(pd.Series([eeRCcadMO_indiv_logistic_contnd_popt_pcov[i][0][2] for i in eeRCcadMO_list_dropna_fit_validity_df.query('high_plat == True').index]).dropna(inplace=False), orient='v', ax=ax3, marker='.', color='k', alpha=0.5)
#sns.swarmplot(pd.Series([eeRCcadMO_indiv_logistic_contnd_popt_pcov[i][0][3] for i in eeRCcadMO_list_dropna_fit_validity_df.query('low_plat == True').index]).dropna(inplace=False), orient='v', ax=ax4, marker='.', color='k', alpha=0.5)

ax3.axhline(bounds_contnd[0][2], c='k')
ax3.axhline(bounds_contnd[1][2], c='k')
ax4.axhline(bounds_contnd[0][3], c='k')
ax4.axhline(bounds_contnd[1][3], c='k')
ax1.set_xlabel(r'x$_\theta$')
ax2.set_xlabel('logistic growth rate "k"')
ax3.set_xlabel('upper_plateau')
ax4.set_xlabel('lower_plateau')
#plt.suptitle('contnd', fontsize=18, x=0.5, y=1.05)
plt.tight_layout()
plt.savefig(Path.joinpath(output_dir, r"eeRCcadMO_contnd_popt_distrib.tif"), dpi=dpi_graph)
#print(bounds_contnd)

plt.clf()
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(width_2col,height_2col))
# plot individual parameters of all timelapses as magenta:
#sns.swarmplot(pd.Series([eeRCcadMO_indiv_logistic_cont_popt_pcov[i][0][0] for i in range(len(eeRCcadMO_indiv_logistic_cont_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax1, color='m', alpha=0.5)
sns.swarmplot(pd.Series([eeRCcadMO_indiv_logistic_cont_popt_pcov[i][0][0] for i in eeRCcadMO_list_dropna_fit_validity_df.query('growth_phase == False').index], dtype=float).dropna(inplace=False), orient='v', ax=ax1, marker='o', color='k', alpha=0.5) # only valid growth phase timelapses could be aligned, therefore only those x0's could be used.
sns.swarmplot(pd.Series([eeRCcadMO_indiv_logistic_cont_popt_pcov[i][0][0] for i in eeRCcadMO_list_dropna_fit_validity_df.query('growth_phase == True').index], dtype=float).dropna(inplace=False), orient='v', ax=ax1, marker='o', color='y', alpha=0.5) # only valid growth phase timelapses could be aligned, therefore only those x0's could be used.
sns.violinplot(pd.Series([eeRCcadMO_indiv_logistic_cont_popt_pcov[i][0][0] for i in range(len(eeRCcadMO_indiv_logistic_cont_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax1, color='lightgrey', alpha=1)

#sns.swarmplot(pd.Series([eeRCcadMO_indiv_logistic_cont_popt_pcov[i][0][1] for i in range(len(eeRCcadMO_indiv_logistic_cont_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax2, color='m', alpha=0.5)
sns.swarmplot(pd.Series([eeRCcadMO_indiv_logistic_cont_popt_pcov[i][0][1] for i in eeRCcadMO_list_dropna_fit_validity_df.query('growth_phase == False').index], dtype=float).dropna(inplace=False), orient='v', ax=ax2, marker='o', color='k', alpha=0.5)
sns.swarmplot(pd.Series([eeRCcadMO_indiv_logistic_cont_popt_pcov[i][0][1] for i in eeRCcadMO_list_dropna_fit_validity_df.query('growth_phase == True').index], dtype=float).dropna(inplace=False), orient='v', ax=ax2, marker='o', color='y', alpha=0.5)
sns.violinplot(pd.Series([eeRCcadMO_indiv_logistic_cont_popt_pcov[i][0][1] for i in range(len(eeRCcadMO_indiv_logistic_cont_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax2, color='lightgrey', alpha=1)

#sns.swarmplot(pd.Series([eeRCcadMO_indiv_logistic_cont_popt_pcov[i][0][2] for i in range(len(eeRCcadMO_indiv_logistic_cont_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax3, color='m', alpha=0.5)
sns.swarmplot(pd.Series([eeRCcadMO_indiv_logistic_cont_popt_pcov[i][0][2] for i in eeRCcadMO_list_dropna_fit_validity_df.query('high_plat == False').index], dtype=float).dropna(inplace=False), orient='v', ax=ax3, marker='o', color='k', alpha=0.5)
sns.swarmplot(pd.Series([eeRCcadMO_indiv_logistic_cont_popt_pcov[i][0][2] for i in eeRCcadMO_list_dropna_fit_validity_df.query('high_plat == True').index], dtype=float).dropna(inplace=False), orient='v', ax=ax3, marker='o', color='y', alpha=0.5)
sns.violinplot(pd.Series([eeRCcadMO_indiv_logistic_cont_popt_pcov[i][0][2] for i in range(len(eeRCcadMO_indiv_logistic_cont_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax3, color='lightgrey', alpha=1)

#sns.swarmplot(pd.Series([eeRCcadMO_indiv_logistic_cont_popt_pcov[i][0][3] for i in range(len(eeRCcadMO_indiv_logistic_cont_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax4, color='m', alpha=0.5)
sns.swarmplot(pd.Series([eeRCcadMO_indiv_logistic_cont_popt_pcov[i][0][3] for i in eeRCcadMO_list_dropna_fit_validity_df.query('low_plat == False').index], dtype=float).dropna(inplace=False), orient='v', ax=ax4, marker='o', color='k', alpha=0.5)
sns.swarmplot(pd.Series([eeRCcadMO_indiv_logistic_cont_popt_pcov[i][0][3] for i in eeRCcadMO_list_dropna_fit_validity_df.query('low_plat == True').index], dtype=float).dropna(inplace=False), orient='v', ax=ax4, marker='o', color='y', alpha=0.5)
sns.violinplot(pd.Series([eeRCcadMO_indiv_logistic_cont_popt_pcov[i][0][3] for i in range(len(eeRCcadMO_indiv_logistic_cont_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax4, color='lightgrey', alpha=1)
# plot the manually selected timelapses with a low plateau as black dots:
#sns.swarmplot(pd.Series([eeRCcadMO_indiv_logistic_cont_popt_pcov[i][0][0] for i in eeRCcadMO_list_dropna_fit_validity_df.query('growth_phase == True').index]).dropna(inplace=False), orient='v', ax=ax1, marker='.', color='k', alpha=0.5) # only valid growth phase timelapses could be aligned, therefore only those x0's could be used.
#sns.swarmplot(pd.Series([eeRCcadMO_indiv_logistic_cont_popt_pcov[i][0][1] for i in eeRCcadMO_list_dropna_fit_validity_df.query('growth_phase == True').index]).dropna(inplace=False), orient='v', ax=ax2, marker='.', color='k', alpha=0.5)
#sns.swarmplot(pd.Series([eeRCcadMO_indiv_logistic_cont_popt_pcov[i][0][2] for i in eeRCcadMO_list_dropna_fit_validity_df.query('high_plat == True').index]).dropna(inplace=False), orient='v', ax=ax3, marker='.', color='k', alpha=0.5)
#sns.swarmplot(pd.Series([eeRCcadMO_indiv_logistic_cont_popt_pcov[i][0][3] for i in eeRCcadMO_list_dropna_fit_validity_df.query('low_plat == True').index]).dropna(inplace=False), orient='v', ax=ax4, marker='.', color='k', alpha=0.5)

ax3.axhline(bounds_cont[0][2], c='k')
ax3.axhline(bounds_cont[1][2], c='k')
ax4.axhline(bounds_cont[0][3], c='k')
ax4.axhline(bounds_cont[1][3], c='k')
ax1.set_xlabel(r'x$_\theta$')
ax2.set_xlabel('logistic growth rate "k"')
ax3.set_xlabel('upper_plateau')
ax4.set_xlabel('lower_plateau')
#plt.suptitle('cont', fontsize=18, x=0.5, y=1.05)
plt.tight_layout()
plt.savefig(Path.joinpath(output_dir, r"eeRCcadMO_cont_popt_distrib.tif"), dpi=dpi_graph)
#print(bounds_cont)

plt.clf()
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(width_2col,height_2col))
# plot individual parameters of all timelapses as magenta:
#sns.swarmplot(pd.Series([eeRCcadMO_indiv_logistic_angle_popt_pcov[i][0][0] for i in range(len(eeRCcadMO_indiv_logistic_angle_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax1, color='m', alpha=0.5)
sns.swarmplot(pd.Series([eeRCcadMO_indiv_logistic_angle_popt_pcov[i][0][0] for i in eeRCcadMO_list_dropna_angle_fit_validity_df.query('growth_phase == False').index], dtype=float).dropna(inplace=False), orient='v', ax=ax1, marker='o', color='k', alpha=0.5) # only valid growth phase timelapses could be aligned, therefore only those x0's could be used.
sns.swarmplot(pd.Series([eeRCcadMO_indiv_logistic_angle_popt_pcov[i][0][0] for i in eeRCcadMO_list_dropna_angle_fit_validity_df.query('growth_phase == True').index], dtype=float).dropna(inplace=False), orient='v', ax=ax1, marker='o', color='y', alpha=0.5) # only valid growth phase timelapses could be aligned, therefore only those x0's could be used.
sns.violinplot(pd.Series([eeRCcadMO_indiv_logistic_angle_popt_pcov[i][0][0] for i in range(len(eeRCcadMO_indiv_logistic_angle_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax1, color='lightgrey', alpha=1)

#sns.swarmplot(pd.Series([eeRCcadMO_indiv_logistic_angle_popt_pcov[i][0][1] for i in range(len(eeRCcadMO_indiv_logistic_angle_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax2, color='m', alpha=0.5)
sns.swarmplot(pd.Series([eeRCcadMO_indiv_logistic_angle_popt_pcov[i][0][1] for i in eeRCcadMO_list_dropna_angle_fit_validity_df.query('growth_phase == False').index], dtype=float).dropna(inplace=False), orient='v', ax=ax2, marker='o', color='k', alpha=0.5)
sns.swarmplot(pd.Series([eeRCcadMO_indiv_logistic_angle_popt_pcov[i][0][1] for i in eeRCcadMO_list_dropna_angle_fit_validity_df.query('growth_phase == True').index], dtype=float).dropna(inplace=False), orient='v', ax=ax2, marker='o', color='y', alpha=0.5)
sns.violinplot(pd.Series([eeRCcadMO_indiv_logistic_angle_popt_pcov[i][0][1] for i in range(len(eeRCcadMO_indiv_logistic_angle_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax2, color='lightgrey', alpha=1)

#sns.swarmplot(pd.Series([eeRCcadMO_indiv_logistic_angle_popt_pcov[i][0][2] for i in range(len(eeRCcadMO_indiv_logistic_angle_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax3, color='m', alpha=0.5)
sns.swarmplot(pd.Series([eeRCcadMO_indiv_logistic_angle_popt_pcov[i][0][2] for i in eeRCcadMO_list_dropna_angle_fit_validity_df.query('high_plat == False').index], dtype=float).dropna(inplace=False), orient='v', ax=ax3, marker='o', color='k', alpha=0.5)
sns.swarmplot(pd.Series([eeRCcadMO_indiv_logistic_angle_popt_pcov[i][0][2] for i in eeRCcadMO_list_dropna_angle_fit_validity_df.query('high_plat == True').index], dtype=float).dropna(inplace=False), orient='v', ax=ax3, marker='o', color='y', alpha=0.5)
sns.violinplot(pd.Series([eeRCcadMO_indiv_logistic_angle_popt_pcov[i][0][2] for i in range(len(eeRCcadMO_indiv_logistic_angle_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax3, color='lightgrey', alpha=1)

#sns.swarmplot(pd.Series([eeRCcadMO_indiv_logistic_angle_popt_pcov[i][0][3] for i in range(len(eeRCcadMO_indiv_logistic_angle_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax4, color='m', alpha=0.5)
sns.swarmplot(pd.Series([eeRCcadMO_indiv_logistic_angle_popt_pcov[i][0][3] for i in eeRCcadMO_list_dropna_angle_fit_validity_df.query('low_plat == False').index], dtype=float).dropna(inplace=False), orient='v', ax=ax4, marker='o', color='k', alpha=0.5)
sns.swarmplot(pd.Series([eeRCcadMO_indiv_logistic_angle_popt_pcov[i][0][3] for i in eeRCcadMO_list_dropna_angle_fit_validity_df.query('low_plat == True').index], dtype=float).dropna(inplace=False), orient='v', ax=ax4, marker='o', color='y', alpha=0.5)
sns.violinplot(pd.Series([eeRCcadMO_indiv_logistic_angle_popt_pcov[i][0][3] for i in range(len(eeRCcadMO_indiv_logistic_angle_popt_pcov))]).dropna(inplace=False), orient='v', ax=ax4, color='lightgrey', alpha=1)
# plot the manually selected timelapses with a low plateau as black dots:
#sns.swarmplot(pd.Series([eeRCcadMO_indiv_logistic_angle_popt_pcov[i][0][0] for i in eeRCcadMO_list_dropna_angle_fit_validity_df.query('growth_phase == True').index]).dropna(inplace=False), orient='v', ax=ax1, marker='.', color='k', alpha=0.5) # only valid growth phase timelapses could be aligned, therefore only those x0's could be used.
#sns.swarmplot(pd.Series([eeRCcadMO_indiv_logistic_angle_popt_pcov[i][0][1] for i in eeRCcadMO_list_dropna_angle_fit_validity_df.query('growth_phase == True').index]).dropna(inplace=False), orient='v', ax=ax2, marker='.', color='k', alpha=0.5)
#sns.swarmplot(pd.Series([eeRCcadMO_indiv_logistic_angle_popt_pcov[i][0][2] for i in eeRCcadMO_list_dropna_angle_fit_validity_df.query('high_plat == True').index]).dropna(inplace=False), orient='v', ax=ax3, marker='.', color='k', alpha=0.5)
#sns.swarmplot(pd.Series([eeRCcadMO_indiv_logistic_angle_popt_pcov[i][0][3] for i in eeRCcadMO_list_dropna_angle_fit_validity_df.query('low_plat == True').index]).dropna(inplace=False), orient='v', ax=ax4, marker='.', color='k', alpha=0.5)

ax3.axhline(bounds_angle[0][2], c='k')
ax3.axhline(bounds_angle[1][2], c='k')
ax4.axhline(bounds_angle[0][3], c='k')
ax4.axhline(bounds_angle[1][3], c='k')
ax1.set_xlabel(r'x$_\theta$')
ax2.set_xlabel('logistic growth rate "k"')
ax3.set_xlabel('upper_plateau')
ax4.set_xlabel('lower_plateau')
#plt.suptitle('angle', fontsize=18, x=0.5, y=1.05)
plt.tight_layout()
plt.savefig(Path.joinpath(output_dir, r"eeRCcadMO_angle_popt_distrib.tif"), dpi=dpi_graph)
#print(bounds_angle)


plt.clf()








###############################################################################

plt.close('all')




###############################################################################

### Frequency analysis test code:


from astropy.timeseries import LombScargle
def plot_lombscargle_freq_analysis(list_of_timelapse_lists, x_label, y_label, logistic_params_df, pdf_filename_prefix):
    """ The input is Pandas dataframe data of timelapses that are collected
    into lists based on their Experimental_Condition (ee, mm, nn, etc.), and
    then these lists are themselves collected into a master list. The inputs
    x_label and y_label are strings that will be passed to the dataframes and 
    are used to identify which column to pull data from (eg. 
    df['Time (minutes)']). The x_label is assumed to be time.
    Returns the peaks as a dataframe and saves a .pdf for each experimental 
    condition."""
    list_of_periods_peaks = list()
    list_of_freqs_peaks = list()
    list_of_power_peaks = list()
    list_of_freq_false_pos = list()
    list_of_t_max = list()
    list_of_t_int = list()
    list_of_sf = list()
    list_of_expt_cond = list()
    data_zerod_successfully_list = list()
    
    for timelapse_list in list_of_timelapse_lists:

        plt.close('all')
        for i in range(len(timelapse_list)):
            
            timepoints = timelapse_list[i][x_label]
            logistic_fit_at_timepoints = logistic(timepoints, 
                                                  logistic_params_df.iloc[i]['popt_x2'], 
                                                  logistic_params_df.iloc[i]['popt_k2'], 
                                                  logistic_params_df.iloc[i]['popt_high_plateau'],
                                                  logistic_params_df.iloc[i]['popt_mid_plateau'])
            
            try:
                zerod_data = timelapse_list[i][y_label] - logistic_fit_at_timepoints
                data_zerod_successfully = True
            except ValueError:
                zerod_data = timelapse_list[i][y_label]
                data_zerod_successfully = False
                
            t_int = np.diff(timepoints).min()
            t_max = timelapse_list[i][x_label].max()
            freq_max = 1 / (4 * t_int)
            freq_min = 1 / timepoints.max()
            #n_freqs = ((1/freq_min) - (1/freq_max)) / t_int
            #freqs_arr = np.linspace(freq_min, freq_max, int(n_freqs))

            ls = LombScargle(timepoints, zerod_data)#, normalization='log')
            #power = ls.power(freqs_arr)
            freqs_arr, power = ls.autopower(minimum_frequency=freq_min, maximum_frequency=freq_max)#, samples_per_peak=500)
            peaks_idx = signal.find_peaks(power)[0]
            peaks_p = power[peaks_idx]
            peaks_f = freqs_arr[peaks_idx]
            prob_false_positive = ls.false_alarm_probability(peaks_p)
            
            list_of_periods_peaks += ([1/pk for pk in peaks_f])
            list_of_freqs_peaks += ([pk for pk in peaks_f])
            list_of_power_peaks += ([pk for pk in peaks_p])
            list_of_freq_false_pos += ([prob for prob in prob_false_positive])
            list_of_t_max += [t_max] * len(peaks_f)
            list_of_t_int += [t_int] * len(peaks_f)
            list_of_sf += [timelapse_list[i]['Source_File'].iloc[0]] * len(peaks_f)
            list_of_expt_cond += [timelapse_list[0]['Experimental_Condition'].iloc[0]] * len(peaks_f)
            data_zerod_successfully_list += ([data_zerod_successfully] * len(peaks_f))

            fig, ax = plt.subplots()
            ax.plot(freqs_arr, power)
            ax.scatter(peaks_f, peaks_p, marker='.', c='k', zorder=10)
            [ax.annotate('%.3f'%peaks_f[i] + ' (= %.3f)'%(1/peaks_f[i]), xy=(peaks_f[i],peaks_p[i]),
                         xytext=(0,15), textcoords='offset pixels',
                         horizontalalignment='left', verticalalignment='top') 
             for i in range(len(peaks_f))]
            ax.text(0.01, 0.04, '[t_int, t_max]', horizontalalignment='left', 
                    verticalalignment='bottom', transform=ax.transAxes)
            ax.text(0.01, 0.01, str([t_int, t_max]),
                    horizontalalignment='left', verticalalignment='bottom', 
                    transform=ax.transAxes)   
            ax.set_xlabel('frequency (1/minutes)')
            ax.set_ylabel('normalized amplitude')
            ax.set_title('Lomb-Scargle Periodogram' + ' , ' + timelapse_list[i]['Source_File'].iloc[0])
            plt.draw()

        multipdf(Path.joinpath(output_dir, pdf_filename_prefix+'_%s_lomb-scargle_periodogram.pdf'%timelapse_list[0]['Experimental_Condition'].iloc[0]))
        plt.close('all')

    cols_headers = ['period_peak_val', 'frequency_peak_val', 'normalized_amplitude', 'percent_probability_to_find_in_nonperiodic_data', 't_max', 't_int', 'Source_File', 'Experimental_Condition', 'data_zerod_successfully']
    data = [list_of_periods_peaks, list_of_freqs_peaks, list_of_power_peaks, list_of_freq_false_pos, list_of_t_max, list_of_t_int, list_of_sf, list_of_expt_cond, data_zerod_successfully_list]
    data = zip(*data,)
    peak_info_df = pd.DataFrame(data=data, columns=cols_headers)
    
    return peak_info_df


master_non_dim_list = [ee_contact_diam_non_dim_df_list, 
                       mm_contact_diam_non_dim_df_list, 
                       nn_contact_diam_non_dim_df_list, 
                       eedb_contact_diam_non_dim_df_list, 
                       eeCcadMO_contact_diam_non_dim_df_list, 
                       eeRcadMO_contact_diam_non_dim_df_list,
                       eeRCcadMO_contact_diam_non_dim_df_list]

master_logistic_params_df = pd.concat([ee_indiv_logistic_contnd_df,
                                       mm_indiv_logistic_contnd_df,
                                       nn_indiv_logistic_contnd_df,
                                       eedb_indiv_logistic_contnd_df,
                                       eeCcadMO_indiv_logistic_contnd_df,
                                       eeRcadMO_indiv_logistic_contnd_df,
                                       eeRCcadMO_indiv_logistic_contnd_df])



### Run the frequency analysis on the contnd data:
peaks_contnd = plot_lombscargle_freq_analysis(master_non_dim_list,
                                              'Time (minutes)',
                                              'Contact Surface Length (N.D.)',
                                              master_logistic_params_df,
                                              pdf_filename_prefix='contnd')

### Run the frequency analysis on the angle data:
peaks_angle = plot_lombscargle_freq_analysis(master_non_dim_list,
                                             'Time (minutes)',
                                             'Average Angle (deg)',
                                             master_logistic_params_df,
                                             pdf_filename_prefix='angle')

possible_true_peaks_contnd = peaks_contnd[peaks_contnd['percent_probability_to_find_in_nonperiodic_data'] < 0.05]
possible_true_peaks_angle = peaks_angle[peaks_angle['percent_probability_to_find_in_nonperiodic_data'] < 0.05]


### When looking at the possibly true peaks, do any of them have a period that
### coincides with the total time of the timelapse? Because that would not be
### a period that we could actually say is true...
#sns.scatterplot(x='percent_probability_to_find_in_nonperiodic_data', 
#                y='period_peak_val', data = possible_true_peaks_contnd,
#                hue='t_int', palette=['grey', 'k'])
fig, ax = plt.subplots(figsize=(width_2col, width_2col))
color = possible_true_peaks_contnd['percent_probability_to_find_in_nonperiodic_data']
cm = 'jet'
period_pk = ax.scatter(possible_true_peaks_contnd['t_max'], 
                       possible_true_peaks_contnd['period_peak_val'], 
                       marker='.', c=color, cmap=cm)

# This colorbar shows the probability of a peak being real, only peaks with a
# probability less than 0.05 are considered:
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(period_pk, cax=cax)
ax.set_xlabel('max time of data (minutes)')
ax.set_ylabel('period peak from frequency analysis (minutes)')

# The line plot below shows the upper limit for where data points can exist:
ax.plot([0,100], [0,100], c='lightgrey', zorder=0)
# The line plot below shows the 
ax.plot([0,200], [0,100], c='lightgrey', zorder=0)

ax.plot([0,200], [0,50], c='lightgrey', zorder=0)
plt.close('all')



# Still not sure about the following summary plots...
#sns.violinplot(x='Experimental_Condition', y='frequency_peak_val', data=peaks_contnd, hue='t_int')
#sns.swarmplot(x=peaks_contnd['Experimental_Condition'], y=np.log10(1/(peaks_contnd['frequency_peak_val'])), hue=peaks_contnd['t_int'])

plt.close('all')
fig, ax = plt.subplots()
sns.violinplot(x=peaks_contnd['Experimental_Condition'], y=(peaks_contnd['frequency_peak_val']), hue=peaks_contnd['t_int'], palette=['grey','k'], ax=ax)
ax.axhline(1/(4*0.5), c='grey')
ax.axhline(1/(4*1.0), c='k')
#ax.set_yscale('log') 
### When looking at all peaks it doesn't look like any frequency values stand
### out...
plt.close('all')



# Probably need to remove the shortest frequencies from this plot as I don't 
# think that they are valid and could be obscuring trends in the data?
    
    
    
###############################################################################

### Is there a correlation between low and high plateaus for ee data?

great_fits_contnd = [np.all(ee_list_dropna_fit_validity_df.iloc[i]) for i in range(len(ee_list_dropna_fit_validity_df))]
great_fits_angle = [np.all(ee_list_dropna_angle_fit_validity_df.iloc[i]) for i in range(len(ee_list_dropna_angle_fit_validity_df))]

great_fits_params_contnd = np.asarray(ee_indiv_logistic_contnd_popt_pcov)[great_fits_contnd][:,0]
great_fits_params_contnd = np.row_stack(great_fits_params_contnd)
great_fits_params_angle = np.asarray(ee_indiv_logistic_angle_popt_pcov)[great_fits_angle][:,0]
great_fits_params_angle = np.row_stack(great_fits_params_angle)

fig, ax = plt.subplots(figsize=(width_1col, width_1col))
ax.set_adjustable('box')
# ax.set_aspect('equal')
ax.scatter(great_fits_params_contnd[:,3], great_fits_params_contnd[:,2])
ax.set_xlim(0,3)
ax.set_ylim(0,3)
plt.close(fig)

fig, ax = plt.subplots(figsize=(width_1col, width_1col))
ax.set_adjustable('box')
# ax.set_aspect('equal')
ax.scatter(great_fits_params_angle[:,3], great_fits_params_angle[:,2])
ax.set_xlim(0,180)
ax.set_ylim(0,180)
plt.close(fig)


# Ans. Not really.



###############################################################################

### Does changing the approach angle affect the error in simulations?

fig, ax = plt.subplots(figsize=(width_1col, width_1col))
ax.plot(radius27px0deg_cell_sim_list_original_nd_df['Average Angle (deg)'], radius27px0deg_cell_sim_list_original_nd_df['Contact Surface Length (N.D.)'], c='k', alpha=0.75, label='0\N{DEGREE SIGN} approach')
ax.plot(radius27px15deg_cell_sim_list_original_nd_df['Average Angle (deg)'], radius27px15deg_cell_sim_list_original_nd_df['Contact Surface Length (N.D.)'], c='k', alpha=0.75, ls='--', label='15\N{DEGREE SIGN} appr.')
ax.plot(radius27px30deg_cell_sim_list_original_nd_df['Average Angle (deg)'], radius27px30deg_cell_sim_list_original_nd_df['Contact Surface Length (N.D.)'], c='grey', alpha=0.75, label='30\N{DEGREE SIGN} appr.')
ax.plot(radius27px45deg_cell_sim_list_original_nd_df['Average Angle (deg)'], radius27px45deg_cell_sim_list_original_nd_df['Contact Surface Length (N.D.)'], c='grey', alpha=0.75, ls='--', label='45\N{DEGREE SIGN} appr.')
ax.set_xlabel('Average Angle (deg)')
ax.set_ylabel('Contact Surface Length (N.D.)')
ax.legend()#loc='upper left')
fig.savefig(Path.joinpath(output_dir, r"cell_sim_dif_approach_angles.tif"), dpi=dpi_graph, bbox_inches='tight')
plt.close(fig)

# Ans. Yes, an angle of 0deg seems to have the highest error at low theoretical angles and an angle
# of 45deg seems to have the highest error at high theoretical angles.



###############################################################################

### Return to me the dimensional areas distributions of each homotypic cell 
### combination:

ee_areas = (ee_list_long['Area of the cell pair (px**2)'] * um_per_pixel**2) / 2
mm_areas = (mm_list_long['Area of the cell pair (px**2)'] * um_per_pixel**2) / 2
nn_areas = (nn_list_long['Area of the cell pair (px**2)'] * um_per_pixel_nn**2) / 2
em_areas = (em_list_long['Area of the cell pair (px**2)'] * um_per_pixel**2) / 2
eedb_areas = (eedb_list_long['Area of the cell pair (px**2)'] * um_per_pixel**2) / 2
eeCcadMO_areas = (eeCcadMO_list_long['Area of the cell pair (px**2)'] * um_per_pixel**2) / 2
eeRcadMO_areas = (eeRcadMO_list_long['Area of the cell pair (px**2)'] * um_per_pixel**2) / 2
eeRCcadMO_areas = (eeRCcadMO_list_long['Area of the cell pair (px**2)'] * um_per_pixel**2) / 2


ee_radius_mean_um = np.sqrt(ee_areas.mean() / np.pi)
mm_radius_mean_um = np.sqrt(mm_areas.mean() / np.pi)
nn_radius_mean_um = np.sqrt(nn_areas.mean() / np.pi)
em_radius_mean_um = np.sqrt(em_areas.mean() / np.pi)
eedb_radius_mean_um = np.sqrt(eedb_areas.mean() / np.pi)
eeCcadMO_radius_mean_um = np.sqrt(eeCcadMO_areas.mean() / np.pi)
eeRcadMO_radius_mean_um = np.sqrt(eeRcadMO_areas.mean() / np.pi)
eeRCcadMO_radius_mean_um = np.sqrt(eeRCcadMO_areas.mean() / np.pi)

area_means = open(Path.joinpath(output_dir, 'cell_radius_means.txt'), 'w')
lines = ['ee_radius_mean_um = %s\n'%(ee_radius_mean_um), 
         'mm_radius_mean_um = %s\n'%(mm_radius_mean_um),
         'nn_radius_mean_um = %s\n'%(nn_radius_mean_um),
         'em_radius_mean_um = %s\n'%(em_radius_mean_um),
         'eedb_radius_mean_um = %s\n'%(eedb_radius_mean_um),
         'eeCcadMO_radius_mean_um = %s\n'%(eeCcadMO_radius_mean_um),
         'eeRcadMO_radius_mean_um = %s\n'%(eeRcadMO_radius_mean_um),
         'eeRCcadMO_radius_mean_um = %s\n'%(eeRCcadMO_radius_mean_um)]
area_means.writelines(lines)
area_means.close()

# sns.distplot(np.sqrt(ee_areas / np.pi))
# sns.distplot(np.sqrt(mm_areas / np.pi))
# sns.distplot(np.sqrt(nn_areas / np.pi))


###############################################################################

### Collect low plateau and high plateau data means + S.D. into a single .tsv


ee_ang_low_all = ee_df_dropna_angle_time_corr_cpu['Average Angle (deg)'][ee_df_dropna_angle_time_corr_cpu['Time (minutes)'] <= growth_phase_start_ee_angle]
ee_ang_low_mean = np.mean(ee_ang_low_all)
ee_ang_low_std = np.std(ee_ang_low_all)
ee_ang_low_nsize = len(ee_ang_low_all)

mm_ang_low_all = mm_df_dropna_angle_time_corr_cpu['Average Angle (deg)'][mm_df_dropna_angle_time_corr_cpu['Time (minutes)'] <= growth_phase_start_mm_angle]
mm_ang_low_mean = np.mean(mm_ang_low_all)
mm_ang_low_std = np.std(mm_ang_low_all)
mm_ang_low_nsize = len(mm_ang_low_all)

nn_ang_low_all = nn_df_dropna_angle_time_corr_cpu['Average Angle (deg)'][nn_df_dropna_angle_time_corr_cpu['Time (minutes)'] <= growth_phase_start_nn_angle]
nn_ang_low_mean = np.mean(nn_ang_low_all)
nn_ang_low_std = np.std(nn_ang_low_all)
nn_ang_low_nsize = len(nn_ang_low_all)

# em_ang_low_all = em_df_dropna_angle_time_corr_cpu['Average Angle (deg)'][em_df_dropna_angle_time_corr_cpu['Time (minutes)'] <= growth_phase_start_em_angle]
# em_ang_low_mean = np.mean(em_ang_low_all)
# em_ang_low_std = np.std(em_ang_low_all)
# em_ang_low_nsize = len(em_ang_low_all)

# eedb_ang_low_all = eedb_df_dropna_angle_time_corr_cpu['Average Angle (deg)'][eedb_df_dropna_angle_time_corr_cpu['Time (minutes)'] <= growth_phase_start_eedb_angle]
# eedb_ang_low_mean = np.mean(eedb_ang_low_all)
# eedb_ang_low_std = np.std(eedb_ang_low_all)
# eedb_ang_low_nsize = len(eedb_ang_low_all)

eeCcadMO_ang_low_all = eeCcadMO_df_dropna_angle_time_corr_cpu['Average Angle (deg)'][eeCcadMO_df_dropna_angle_time_corr_cpu['Time (minutes)'] <= growth_phase_start_eeCcadMO_angle]
eeCcadMO_ang_low_mean = np.mean(eeCcadMO_ang_low_all)
eeCcadMO_ang_low_std = np.std(eeCcadMO_ang_low_all)
eeCcadMO_ang_low_nsize = len(eeCcadMO_ang_low_all)

eeRcadMO_ang_low_all = eeRcadMO_df_dropna_angle_time_corr_cpu['Average Angle (deg)'][eeRcadMO_df_dropna_angle_time_corr_cpu['Time (minutes)'] <= growth_phase_start_eeRcadMO_angle]
eeRcadMO_ang_low_mean = np.mean(eeRcadMO_ang_low_all)
eeRcadMO_ang_low_std = np.std(eeRcadMO_ang_low_all)
eeRcadMO_ang_low_nsize = len(eeRcadMO_ang_low_all)

eeRCcadMO_ang_low_all = eeRCcadMO_df_dropna_angle_time_corr_cpu['Average Angle (deg)'][eeRCcadMO_df_dropna_angle_time_corr_cpu['Time (minutes)'] <= growth_phase_start_eeRCcadMO_angle]
eeRCcadMO_ang_low_mean = np.mean(eeRCcadMO_ang_low_all)
eeRCcadMO_ang_low_std = np.std(eeRCcadMO_ang_low_all)
eeRCcadMO_ang_low_nsize = len(eeRCcadMO_ang_low_all)




ee_ang_high_all = ee_df_dropna_angle_time_corr_cpu['Average Angle (deg)'][ee_df_dropna_angle_time_corr_cpu['Time (minutes)'] >= growth_phase_end_ee_angle]
ee_ang_high_mean = np.mean(ee_ang_high_all)
ee_ang_high_std = np.std(ee_ang_high_all)
ee_ang_high_nsize = len(ee_ang_high_all)

mm_ang_high_all = mm_df_dropna_angle_time_corr_cpu['Average Angle (deg)'][mm_df_dropna_angle_time_corr_cpu['Time (minutes)'] >= growth_phase_end_mm_angle]
mm_ang_high_mean = np.mean(mm_ang_high_all)
mm_ang_high_std = np.std(mm_ang_high_all)
mm_ang_high_nsize = len(mm_ang_high_all)

nn_ang_high_all = nn_df_dropna_angle_time_corr_cpu['Average Angle (deg)'][nn_df_dropna_angle_time_corr_cpu['Time (minutes)'] >= growth_phase_end_nn_angle]
nn_ang_high_mean = np.mean(nn_ang_high_all)
nn_ang_high_std = np.std(nn_ang_high_all)
nn_ang_high_nsize = len(nn_ang_high_all)

# em_ang_high_all = em_df_dropna_angle_time_corr_cpu['Average Angle (deg)'][em_df_dropna_angle_time_corr_cpu['Time (minutes)'] >= growth_phase_end_em_angle]
# em_ang_high_mean = np.mean(em_ang_high_all)
# em_ang_high_std = np.std(em_ang_high_all)
# em_ang_high_nsize = len(em_ang_high_all)

# eedb_ang_high_all = eedb_df_dropna_angle_time_corr_cpu['Average Angle (deg)'][eedb_df_dropna_angle_time_corr_cpu['Time (minutes)'] >= growth_phase_end_eedb_angle]
# eedb_ang_high_mean = np.mean(eedb_ang_high_all)
# eedb_ang_high_std = np.std(eedb_ang_high_all)
# eedb_ang_high_nsize = len(eedb_ang_high_all)

eeCcadMO_ang_high_all = eeCcadMO_df_dropna_angle_time_corr_cpu['Average Angle (deg)'][eeCcadMO_df_dropna_angle_time_corr_cpu['Time (minutes)'] >= growth_phase_end_eeCcadMO_angle]
eeCcadMO_ang_high_mean = np.mean(eeCcadMO_ang_high_all)
eeCcadMO_ang_high_std = np.std(eeCcadMO_ang_high_all)
eeCcadMO_ang_high_nsize = len(eeCcadMO_ang_high_all)

eeRcadMO_ang_high_all = eeRcadMO_df_dropna_angle_time_corr_cpu['Average Angle (deg)'][eeRcadMO_df_dropna_angle_time_corr_cpu['Time (minutes)'] >= growth_phase_end_eeRcadMO_angle]
eeRcadMO_ang_high_mean = np.mean(eeRcadMO_ang_high_all)
eeRcadMO_ang_high_std = np.std(eeRcadMO_ang_high_all)
eeRcadMO_ang_high_nsize = len(eeRcadMO_ang_high_all)

eeRCcadMO_ang_high_all = eeRCcadMO_df_dropna_angle_time_corr_cpu['Average Angle (deg)'][eeRCcadMO_df_dropna_angle_time_corr_cpu['Time (minutes)'] >= growth_phase_end_eeRCcadMO_angle]
eeRCcadMO_ang_high_mean = np.mean(eeRCcadMO_ang_high_all)
eeRCcadMO_ang_high_std = np.std(eeRCcadMO_ang_high_all)
eeRCcadMO_ang_high_nsize = len(eeRCcadMO_ang_high_all)




em_ang_all = em_df_dropna_angle_time_corr_cpu['Average Angle (deg)'].copy()
em_ang_all_mean = np.mean(em_ang_all)
em_ang_all_std = np.std(em_ang_all)
em_ang_all_nsize = len(em_ang_all)

eedb_ang_all = eedb_df_dropna_angle_time_corr_cpu['Average Angle (deg)'].copy()
eedb_ang_all_mean = np.mean(eedb_ang_all)
eedb_ang_all_std = np.std(eedb_ang_all)
eedb_ang_all_nsize = len(eedb_ang_all)


 

headers = ['Experimental_Condition', 'raw_angles', 'mean_angles', 'std_dev', 
           'n_size', 'time_cutoff']

ee_angle_high_plat = ['ecto_ecto_95-100pct_growth', list(ee_ang_high_all), ee_ang_high_mean, 
                      ee_ang_high_std, ee_ang_high_nsize, growth_phase_end_ee_angle]
mm_angle_high_plat = ['meso_meso_95-100pct_growth', list(mm_ang_high_all), mm_ang_high_mean, 
                      mm_ang_high_std, mm_ang_high_nsize, growth_phase_end_mm_angle]
nn_angle_high_plat = ['endo_endo_95-100pct_growth', list(nn_ang_high_all), nn_ang_high_mean, 
                      nn_ang_high_std, nn_ang_high_nsize, growth_phase_end_nn_angle]
# em_angle_high_plat = ['ecto_meso_95-100pct_growth', list(em_ang_high_all), em_ang_high_mean, 
#                       em_ang_high_std, em_ang_high_nsize, growth_phase_end_em_angle]
# eedb_angle_high_plat = ['ecto_ecto_dissoc_buff_95-100pct_growth', list(eedb_ang_high_all), eedb_ang_high_mean, 
#                         eedb_ang_high_std, eedb_ang_high_nsize, growth_phase_end_eedb_angle]
eeCcadMO_angle_high_plat = ['eeCcadMO_95-100pct_growth', list(eeCcadMO_ang_high_all), eeCcadMO_ang_high_mean, 
                            eeCcadMO_ang_high_std, eeCcadMO_ang_high_nsize, growth_phase_end_eeCcadMO_angle]
eeRcadMO_angle_high_plat = ['eeRcadMO_95-100pct_growth', list(eeRcadMO_ang_high_all), eeRcadMO_ang_high_mean, 
                            eeRcadMO_ang_high_std, eeRcadMO_ang_high_nsize, growth_phase_end_eeRcadMO_angle]
eeRCcadMO_angle_high_plat = ['eeRCcadMO_95-100pct_growth', list(eeRCcadMO_ang_high_all), eeRCcadMO_ang_high_mean, 
                             eeRCcadMO_ang_high_std, eeRCcadMO_ang_high_nsize, growth_phase_end_eeRCcadMO_angle]

ee_angle_low_plat = ['ecto_ecto_0-5pct_growth', list(ee_ang_low_all), ee_ang_low_mean, 
                     ee_ang_low_std, ee_ang_low_nsize, growth_phase_start_ee_angle]
mm_angle_low_plat = ['meso_meso_0-5pct_growth', list(mm_ang_low_all), mm_ang_low_mean, 
                     mm_ang_low_std, mm_ang_low_nsize, growth_phase_start_mm_angle]
nn_angle_low_plat = ['endo_endo_0-5pct_growth', list(nn_ang_low_all), nn_ang_low_mean, 
                     nn_ang_low_std, nn_ang_low_nsize, growth_phase_start_nn_angle]
# em_angle_low_plat = ['ecto_meso_0-5pct_growth', list(em_ang_low_all), em_ang_low_mean, 
#                      em_ang_Low_std, em_ang_low_nsize, growth_phase_start_em_angle]
# eedb_angle_low_plat = ['ecto_ecto_dissoc_buff_0-5pct_growth', list(eedb_ang_low_all), eedb_ang_low_mean, 
#                        eedb_ang_low_std, eedb_ang_low_nsize, growth_phase_start_eedb_angle]
eeCcadMO_angle_low_plat = ['eeCcadMO_0-5pct_growth', list(eeCcadMO_ang_low_all), eeCcadMO_ang_low_mean, 
                           eeCcadMO_ang_low_std, eeCcadMO_ang_low_nsize, growth_phase_start_eeCcadMO_angle]
eeRcadMO_angle_low_plat = ['eeRcadMO_0-5pct_growth', list(eeRcadMO_ang_low_all), eeRcadMO_ang_low_mean, 
                           eeRcadMO_ang_low_std, eeRcadMO_ang_low_nsize, growth_phase_start_eeRcadMO_angle]
eeRCcadMO_angle_low_plat = ['eeRCcadMO_0-5pct_growth', list(eeRCcadMO_ang_low_all), eeRCcadMO_ang_low_mean, 
                            eeRCcadMO_ang_low_std, eeRCcadMO_ang_low_nsize, growth_phase_start_eeRCcadMO_angle]

em_angle_all = ['ecto_meso_0-100pct_growth', list(em_ang_all), em_ang_all_mean, 
                em_ang_all_std, em_ang_all_nsize, np.nan]
eedb_angle_all = ['ecto_ecto_dissoc_buff_0-100pct_growth', list(eedb_ang_all), eedb_ang_all_mean, 
                  eedb_ang_all_std, eedb_ang_all_nsize, np.nan]


### Construct the table:
table = [headers, ee_angle_high_plat, mm_angle_high_plat, nn_angle_high_plat, 
         eeCcadMO_angle_high_plat, eeRcadMO_angle_high_plat, eeRCcadMO_angle_high_plat, 
         ee_angle_low_plat, mm_angle_low_plat, nn_angle_low_plat, 
         eeCcadMO_angle_low_plat, eeRcadMO_angle_low_plat, eeRCcadMO_angle_low_plat, 
         em_angle_all, eedb_angle_all]

### and save the table as a .tsv:
with open(Path.joinpath(output_dir, r'angle_vals_summary.tsv'), 'w', 
          newline='') as file_out:
    tsv_obj = csv.writer(file_out, delimiter='\t')
    tsv_obj.writerows(table)



### turn the table into a dataframe to facilitate mathematical operations:
table_df = pd.DataFrame(data=table[1:], columns=table[0])


from uncertainties import unumpy as unp


def sigma_div_beta_err(t):
    return 1 - unp.cos(unp.radians(t))


angles_with_err = unp.uarray(table_df['mean_angles'], table_df['std_dev'])
table_df['sigma_div_beta'] = sigma_div_beta_err(angles_with_err/2)

missing_data = 1e6 # the uncertainties package doesn't seem to like np.nan,
                   # so we are just going to use a very large number that won't
                   # show up normally.

sigma_david_2014 = unp.uarray([0.57, 0.15, 0.05, 0.09, missing_data, missing_data,
                               0.57, 0.15, 0.05, 0.09, missing_data, missing_data,
                               missing_data, 0.57], 
                              [0.26, 0.06, 0.06, 0.04, missing_data, missing_data,
                               0.26, 0.06, 0.06, 0.04, missing_data, missing_data,
                               missing_data, 0.26])

beta_david_2014 = np.array([0.86, 0.24, 0.09, 0.24, missing_data, missing_data,
                            0.86, 0.24, 0.09, 0.24, missing_data, missing_data,
                            missing_data, 0.86])
    

table_df['sigma'] = table_df['sigma_div_beta'] * beta_david_2014
table_df['beta'] = 1 / table_df['sigma_div_beta'] * sigma_david_2014


print('Done!')



