# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 23:32:29 2023

@author: Serge
"""

### Needed to find directories, folders, files:
from pathlib import Path
### Used for splitting strings and working with regular expressions:
import re

### Some data processing libraries:
### numpy of course:
import numpy as np
### For propagating errors (eg. means with standard devations):
# from uncertainties import unumpy as unp
### pandas for more easily working with tables and DataFrames. Also, you can
### easily open comma separated value (.csv) files with pandas:
import pandas as pd

### Some data plotting libraries:
from matplotlib import pyplot as plt
import seaborn as sns

### For statistical tests and distributions:
from scipy import stats
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy import ndimage as ndi

### Import Pdfpages to save figures in to a multi-page PDF file:
from matplotlib.backends.backend_pdf import PdfPages
### A lot of figures are going to be opened before they are saved in to a
### multi-page pdf, so increase the max figure open warning:
plt.rcParams.update({'figure.max_open_warning':300})
plt.rcParams.update({'font.size':10})





def cm2inch(num):
    return num/2.54



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








### Create path objects describing the locations of the data and where the
### plots and analyses should be output to:

## Get the working directory of the script:
curr_dir = Path(__file__).parent.absolute()

## Data can be found in the following directory:
path_results = Path.joinpath(curr_dir, r"Jasplak_data/results_raw")
path_grpkeys = Path.joinpath(curr_dir, r"Jasplak_data/group_keys")
path_init_area = Path.joinpath(curr_dir, r"Jasplak_data/initial_areas")

# adh_kin_figs_out_dir = Path.joinpath(curr_dir, r"adhesion_analysis_figs_out")

## cleaned up data will be output to the following directory:
out_dir = Path.joinpath(curr_dir, r"Jasplak_analysis_out")

## Make the directory where we will save our plots and analyses if it doesn't
## exist yet:
if out_dir.exists():    
    pass
else:
    out_dir.mkdir()




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




## Images were acquired at 1 frame every 10 seconds:
t_res = 10 # 10 seconds per timeframe

## The microscopes have slightly different pixel resolutions:
xy_res_DMSO = 0.5045296 # um per pixel (x and y are the same resolution)
xy_res_jasp = 0.5066751 # um per pixel (x and y are the same resolution)

xy_res_dict = {'DMSO':xy_res_DMSO, 'Jasplak100nM':xy_res_jasp, 'Jasplak200nM':xy_res_jasp}


## I want this information in the master table:
# column_heads = ['Time (minutes)', 'Timeframe', 'Contact Diameter (px)', 
#                 'Contact Diameter (um)', 'Contact Diameter (N.D.)', 
#                 'Initial Avg Cell Area (px**2)', 'Initial Avg Cell Radius (px)', 
#                 Initial Avg Cell Radius (um), 'Contact Number', 
#                 'Source File', 'Experimental Condition', 'Measurement Name']      


## To start:
## open the raw data for results and group keys and turn each into a dataframe:
data_list = list()
grps_list = list()
init_cell_area_list = list()


for filename in path_init_area.glob('*.csv'):
    df = pd.read_csv(filename)

    csv_filename = filename.name
    ## Add the csv filename as a column:
    df['CSV Filename'] = [csv_filename] * len(df)
    
    ## split the filenames apart. Contains the date, experimental condition, 
    ## and other info:
    fn_info = df['CSV Filename'].apply(lambda fn: re.split('_', fn))
    fn_dates = list(zip(*fn_info))[0]
    
    ## split the labels apart. They contain the source file, measurement name, 
    ## type of stack (e.g. z or t stack), and stack position, separated by a
    ## colon, in that order:
    meas_info = df['Label'].apply(lambda meas_label: re.split(':', meas_label))
    meas_sf, meas_nm, stk_type, meas_pos = zip(*meas_info)
    
    ## lastly, I also want the experimental condition (is it DMSO? Jasplak?)
    ## this is specified in the filename. The possibilities are 'DMSO', 
    ## 'Jasplak100nM', and 'Jasplak200nM':
    
    ## For DMSO treated cells:
    dmso_cond = list(map(lambda x: 'DMSO' if 'DMSO' in x else '', meas_sf))
    
    ## For cells incubated in 100nM of Jasplakinolide:
    jasp100_cond = list(map(lambda x: 'Jasplak100nM' if 'Jasplak100nM' in x and 'DMSO' not in x else '', meas_sf))
    
    ## For cells incubated in 200nM of Jasplakinolide:
    jasp200_cond = list(map(lambda x: 'Jasplak200nM' if 'Jasplak200nM' in x and 'DMSO' not in x else '', meas_sf))
    
    ## connect these 3 posssible experimental condition lists:
    expt_cond_pre = list(zip(*(dmso_cond, jasp100_cond, jasp200_cond)))
    ## and remove any empty lists:
    expt_cond = [list(filter(None, x)) for x in expt_cond_pre]
    ## if the list has a single entry (it should), convert it into a single string:
    expt_cond = list(map(lambda x: ''.join(x) if len(x) == 1 else x, expt_cond))
    
    ## make sure that all entries specify only one experimental condition:
    assert(np.all([type(x)==str for x in expt_cond]))

    ## populate the dataframe with desired columns of cleaned up data:
    df['Source File'] = meas_sf
    df['Measurement Name'] = meas_nm
    df['Experimental Condition'] = expt_cond
    df['Acquisition Date'] = fn_dates
    
    
    ## take the average area for each group:
    area_means = df.groupby('Group')['Area'].mean()
    ## convert to a dictionary:
    grp_to_avg_area_dict = area_means.to_dict()
    ## apply the dictionary to the dataframe to populate it with the mean areas:
    df['Initial Avg Cell Area (px**2)'] = df['Group'].map(grp_to_avg_area_dict)

    ## convert to initial radii:
    df['Initial Avg Cell Radius (px)'] = df['Initial Avg Cell Area (px**2)'].map(lambda x: np.sqrt(x/np.pi))
    df['Initial Avg Cell Radius (um)'] = df['Initial Avg Cell Radius (px)'] * df['Experimental Condition'].map(xy_res_dict)


    init_cell_area_list.append(df)

## turn the list of dataframes into a single long dataframe:
init_cell_area_df = pd.concat(init_cell_area_list)
## group the initial average cell radii according to the source file and group number, and then turn that into a dictionary:
grp_to_avg_radius_dict = dict([(sf_grp_pair, float(df['Initial Avg Cell Radius (px)'].unique())) for sf_grp_pair, df in list(init_cell_area_df.groupby(['Source File', 'Group']))])
grp_to_avg_area_dict = dict([(sf_grp_pair, float(df['Initial Avg Cell Area (px**2)'].unique())) for sf_grp_pair, df in list(init_cell_area_df.groupby(['Source File', 'Group']))])





## next we are going to create a dictionary that associates each measurement
## name with a group number:
for filename in path_grpkeys.glob('*.csv'):
    df = pd.read_csv(filename)
    
    csv_filename = filename.name
    ## Add the csv filename as a column:
    df['CSV Filename'] = [csv_filename] * len(df)

    
    ## there are several unused columns here, which we will drop:
    df.drop(columns=['Type', 'X', 'Y', 'Width', 'Height', 'Points', 'Fill', 
                     'LWidth', 'C', 'Z', 'T', 'Index', ' '], inplace=True)

    ## check for and remove duplicate entries:
    duplic_mask = [np.count_nonzero(df['Name'] == x) > 1 for x in df['Name'].unique()]
    duplic_vals = df['Name'].unique()[duplic_mask]
    ## remove the first of the duplicate values, since I must have re-measured:
    ## get the index of the first instance of each duplicate:
    duplic_indices_to_drop = [df.query('Name == "%s"'%val).index[0] for val in duplic_vals]
    ## remove that index from the dataframe:
    df_no_dupes = df.drop(index=duplic_indices_to_drop)
    
    ## double-check that all measurement names are unique:
    assert(len(df['Name'].unique()) == len(df_no_dupes['Name']))


    ## split the filenames apart. Contains the date, experimental condition, 
    ## and other info:
    fn_info = df_no_dupes['CSV Filename'].apply(lambda fn: re.split('_', fn))
    fn_dates = list(zip(*fn_info))[0]
    
    ## get the experimental condition (is it DMSO? Jasplak?)
    ## this is specified in the filename. The possibilities are 'DMSO', 
    ## 'Jasplak100nM', and 'Jasplak200nM':
    
    ## For DMSO treated cells:
    dmso_cond = list(map(lambda x: 'DMSO' if 'DMSO' in x else '', df_no_dupes['CSV Filename']))
    
    ## For cells incubated in 100nM of Jasplakinolide:
    jasp100_cond = list(map(lambda x: 'Jasplak100nM' if 'Jasplak100nM' in x and 'DMSO' not in x else '', df_no_dupes['CSV Filename']))
    
    ## For cells incubated in 200nM of Jasplakinolide:
    jasp200_cond = list(map(lambda x: 'Jasplak200nM' if 'Jasplak200nM' in x and 'DMSO' not in x else '', df_no_dupes['CSV Filename']))
    
    ## connect these 3 posssible experimental condition lists:
    expt_cond_pre = list(zip(*(dmso_cond, jasp100_cond, jasp200_cond)))
    ## and remove any empty lists:
    expt_cond = [list(filter(None, x)) for x in expt_cond_pre]
    ## if the list has a single entry (it should), convert it into a single string:
    expt_cond = list(map(lambda x: ''.join(x) if len(x) == 1 else x, expt_cond))
    
    ## make sure that all entries specify only one experimental condition:
    assert(np.all([type(x)==str for x in expt_cond]))
    
    ## populate the dataframe with desired columns of cleaned up data:
    df_no_dupes['Acquisition Date'] = fn_dates
    df_no_dupes.rename({'Name':'Measurement Name'}, axis='columns', inplace=True)
    df_no_dupes['Experimental Condition'] = expt_cond


    grps_list.append(df_no_dupes)
    
grps_df = pd.concat(grps_list)
name_to_grp_dict = dict([(grp_name, int(grp_df['Group'].values)) for grp_name, grp_df in list(grps_df.groupby(['Acquisition Date', 'Experimental Condition', 'Measurement Name']))])




for filename in path_results.glob('*.csv'):
    ## open the .csv file:
    df = pd.read_csv(filename)
    
    csv_filename = filename.name
    ## Add the csv filename as a column:
    df['CSV Filename'] = [csv_filename] * len(df)
    
    ## split the filenames apart. Contains the date, experimental condition, 
    ## and other info:
    fn_info = df['CSV Filename'].apply(lambda fn: re.split('_', fn))
    fn_dates = list(zip(*fn_info))[0]


    ## the raw data has a measurement called "angle", which is not useful, so
    ## drop it:
    df.drop(columns=['Angle', ' '], inplace=True)

    ## split the labels apart. They contain the source file, measurement name, 
    ## type of stack (e.g. z or t stack), and stack position, separated by a
    ## colon, in that order:
    meas_info = df['Label'].apply(lambda meas_label: re.split(':', meas_label))
    meas_sf, meas_nm, stk_type, meas_pos = zip(*meas_info)
    ## the measurement position has a lot of extraneous info such as the size 
    ## of the stack and the filename, so pull out just the stack position:
    t_pos, _ = list(zip(*[re.split('/', pos) for pos in meas_pos]))
    t_pos = [int(x) for x in t_pos]
    
    ## lastly, I also want the experimental condition (is it DMSO? Jasplak?)
    ## this is specified in the filename. The possibilities are 'DMSO', 
    ## 'Jasplak100nM', and 'Jasplak200nM':
    
    ## For DMSO treated cells:
    dmso_cond = list(map(lambda x: 'DMSO' if 'DMSO' in x else '', meas_sf))
    
    ## For cells incubated in 100nM of Jasplakinolide:
    jasp100_cond = list(map(lambda x: 'Jasplak100nM' if 'Jasplak100nM' in x and 'DMSO' not in x else '', meas_sf))
    
    ## For cells incubated in 200nM of Jasplakinolide:
    jasp200_cond = list(map(lambda x: 'Jasplak200nM' if 'Jasplak200nM' in x and 'DMSO' not in x else '', meas_sf))
    
    ## connect these 3 posssible experimental condition lists:
    expt_cond_pre = list(zip(*(dmso_cond, jasp100_cond, jasp200_cond)))
    ## and remove any empty lists:
    expt_cond = [list(filter(None, x)) for x in expt_cond_pre]
    ## if the list has a single entry (it should), convert it into a single string:
    expt_cond = list(map(lambda x: ''.join(x) if len(x) == 1 else x, expt_cond))
    
    ## make sure that all entries specify only one experimental condition:
    assert(np.all([type(x)==str for x in expt_cond]))
    
    
    
    
    ## populate the dataframe with desired columns of cleaned up data:
    df['Source File'] = meas_sf
    df['Measurement Name'] = meas_nm
    df['Experimental Condition'] = expt_cond
    df['Acquisition Date'] = fn_dates


    ## Use the dictionary 'name_to_grp_dict' to connect the measurements to 
    ## their group IDs, and include those in the dataframe:
    ## '.map' only works on 'Series' objects, not 'DataFrames', so make column
    ## with our 3 required criteria for assigning the group number to a given
    ## measurement name:
    df['Group Criterion'] = df.apply(lambda x: tuple(x[['Acquisition Date', 'Experimental Condition', 'Measurement Name']]), axis=1)
    ## ...finally we can assign the group numbers to the measurement names:
    df['Group'] = df['Group Criterion'].map(name_to_grp_dict).astype(int)


    ## Use the dictionary 'grp_to_avg_radius_dict' to connect the different
    ## groups within a an experimental condition and date to measurement names:
    df['init_radius_criterion'] = df.apply(lambda x: tuple(x[['Source File', 'Group']]), axis=1)
    df['Initial Avg Cell Area (px**2)'] = df['init_radius_criterion'].map(grp_to_avg_area_dict)
    df['Initial Avg Cell Radius (px)'] = df['init_radius_criterion'].map(grp_to_avg_radius_dict)
    df['Initial Avg Cell Radius (um)'] = df['Initial Avg Cell Radius (px)'] * df['Experimental Condition'].map(xy_res_dict)


    ## Include the timeframe numbers and time (in seconds):
    df['Timeframe'] = t_pos
    df['Time (seconds)'] = df['Timeframe'].apply(lambda x: x * t_res)

    
    ## Clean up and expand upon the contact diameter measurements:
    df.rename({'Length':'Contact Diameter (px)'}, axis='columns', inplace=True)
    
    ## I think that my potential measurement error in the contact lengths was about
    ## +/-3 pixels (this is my conservative estimate, I'd like to think that I did 
    ## a more accurate job than that though...)
    df['Estimated Error in Cont. Diam. (px)'] = [float(3)] * len(df)
    
    
    df['Contact Diameter (um)'] = df['Contact Diameter (px)'] * df['Experimental Condition'].map(xy_res_dict)
    
    ## Non-dimensionalize the contact diameters:
    df['Contact Diameter (N.D.)'] = df['Contact Diameter (px)'] / df['Initial Avg Cell Radius (px)']

    ## we need to sort each group in df according to the timeframe so that we
    ## can get the rate of change of the contact diameter, and then put df back
    ## together:
    grp_list = []
    for nm, grp in df.groupby('Group'):
        ## Sort data in ascending order according to the Timeframe number:
        grp = grp.sort_values('Timeframe')
        ## Lastly, create a column with the rate of change in contact size since the  
        ## previous timepoint:
        dt = np.diff(grp['Time (seconds)']).astype(float)
        dr = np.diff(grp['Contact Diameter (N.D.)']).astype(float)
        ## len(dt) and len(dr) are 1 item shorter than len(jasplak_results_df), this is
        ## because there is no entry before the first one to take the difference of.
        ## Therefore, we will make this first entry np.nan so that dt and dr line up 
        ## with jasplak_results_df['First Contact Measurement?']:
        dt = np.insert(dt, 0, np.nan)
        dr = np.insert(dr, 0, np.nan)
        ## add dr and dt to jasplak_results_df:
        grp['dt'] = dt
        grp['dr'] = dr
        grp['Cont. Diam. Rate of Change (N.D./sec)'] = dr/dt
        grp['Smoothed Contact Diameter (N.D.)'] = ndi.gaussian_filter1d(grp['Contact Diameter (N.D.)'], sigma=10)
        
        dr_smooth = np.diff(grp['Smoothed Contact Diameter (N.D.)']).astype(float)
        dr_smooth = np.insert(dr_smooth, 0, np.nan)
        grp['Smoothed Cont. Diam. Rate of Change (N.D./sec)'] = dr_smooth/dt

        grp_list.append(grp)
    ## put the df back together
    df = pd.concat(grp_list)
    # df = pd.concat([grp.sort_values('Timeframe') for nm, grp in df.groupby('Group')])


    ## update the data list:
    data_list.append(df)

    # #### The following is for debugging:
    # if csv_filename == '2023-01-20_ecto-ecto_DMSO_dt10s_19C_forJasp200_results.csv':
    #     exit()
    #     ## the timeframes were measured out of order here...:
    #     nm, grp = list(zip(*df.groupby('Group')))
    #     grp[-2][:20]['Time (seconds)']
    #     ## how to sort them so that they are in the correct order? Need to sort
    #     ## within each group within each source file...
    # #### end of debugging code

## Create the final DataFrame with all of the useful, juicy data:
jasplak_results_df = pd.concat(data_list).reset_index()

## Record the index of the first timepoint for each forming contact:
nms, grps = list(zip(*jasplak_results_df.groupby(['Source File', 'Group'])))
first_cont_indices = [df.query('`Time (seconds)` == "%s"'%min(df['Time (seconds)'])).index for df in grps]
## Make a column where the first contact measurements are marked as True so 
## that they can be easily masked:
first_cont_cols = pd.concat([df['Time (seconds)'] == min(df['Time (seconds)']) for df in grps])
jasplak_results_df['First Contact Measurement?'] = first_cont_cols



## The column currently called 'Group' has what I want to call the 'Contact 
## Number':
jasplak_results_df.rename({'Group':'Contact Number'}, axis='columns', inplace=True)

## I also want to drop unnecessary columns:
jasplak_results_df.drop(columns=['Group Criterion', 'init_radius_criterion', 
                                 'Label', 'CSV Filename', 'index'], inplace=True)


## I want this information in the final, master DataFrame:
# column_heads = ['Time (seconds)', 'Timeframe', 'Contact Diameter (px)', 
#                 'Contact Diameter (um)', 'Contact Diameter (N.D.)', 
#                 'Initial Avg Cell Area (px**2)', 'Initial Avg Cell Radius (px)', 
#                 Initial Avg Cell Radius (um), 'Contact Number', 
#                 'Source File', 'Experimental Condition', 'Measurement Name',
#                 'Experiment ID']




## The groups overall seem very different (p < 0.001), but there are thousands 
## of data points... how do individual experiments compare to one another?
## (i.e. Jasplak dose vs DMSO, since those cells are derived from the same 
## tissue explants):



## First need to connect DMSO source files to Jasplak source files:

# make a column called 'Experiment ID', and have one for each DMSO-Jasplak. pair
# Expt. No. should go up to 5 (3 DMSO-Jasplak200nM pairs, 2 DMSO-Jasplak100nM 
# pairs):

## use info in source file names and the Acquisition Date column to create the
## Experiment ID column.

def get_expt_id(x):
    acq_date, expt_details = re.split('_', x)[0], ''.join(re.split('_', x)[1:])
    
    if 'Jasp' and '100' in expt_details:
        expt_id = 'Jasplak100nM : ' + '%s'%acq_date
    elif 'Jasp' and '200' in expt_details:
        expt_id = 'Jasplak200nM : ' + '%s'%acq_date
    else:
        expt_id = 'np.nan'
        
    return expt_id



expt_ids = jasplak_results_df['Source File'].apply(lambda x: get_expt_id(x))

jasplak_results_df['Experimental ID'] = expt_ids


## The following is a final check to make sure that the data is contiguous and
## clean:
val_chk = np.unique(jasplak_results_df['dt'])

idx2chk = np.array(np.squeeze(np.where(np.logical_and((jasplak_results_df['dt'] != t_res), 
                                                      np.isfinite(jasplak_results_df['dt'])))), ndmin=1)

data2chk_list = list()
for i in idx2chk:
    print(jasplak_results_df.iloc[i-1:i+2])
    data2chk_list.append(jasplak_results_df.iloc[i-1:i+2])

## If everything looks good, then data2chk_list should be an empty list:
try:
    assert(not data2chk_list)
except(AssertionError):
    print('There are some data that are not contiguous, meaning that some '
          'cell-cell contacts are measured for a while, and then one or more '
          ' timeframes are skipped before measurement for that contact '
          'resumes. Please make each contact have a single, continuous set of '
          'measurements.')


## Now that everything seems okay, save the dataframe as a .csv file:
jasplak_results_df.to_csv(Path.joinpath(out_dir, 'jasplak_data_clean.csv'))













###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################



## now we can start plotting things
# exit()


## plot each contact over time and place in two multi-page pdf files (one for 
## DMSO treatment and one for each Jasplak concentration treatment):
for expt_cond_nm, expt_cond_grps in jasplak_results_df.groupby('Experimental Condition'):

    ax_xlims = (min([0, jasplak_results_df['Time (seconds)'].min()]), jasplak_results_df['Time (seconds)'].max() + t_res)
    ## For the DMSO treatment:
    if expt_cond_nm == 'DMSO':
        for name, single_contact_df in expt_cond_grps.groupby(['Source File', 'Contact Number']):
        
        
            fig, ax = plt.subplots(figsize=(8,6))
            
            
            sns.lineplot(x='Time (seconds)', y='Contact Diameter (N.D.)', 
                         data=single_contact_df, marker='.', color='k', 
                         linewidth=1, markeredgewidth=0.3, ax=ax)
            
            sns.lineplot(x='Time (seconds)', y='Smoothed Contact Diameter (N.D.)', 
                         data=single_contact_df, marker='.', color='r', 
                         linewidth=1, markeredgewidth=0.3, ax=ax)
            ax.set_title('%s : Contact Number %d'%(*name,))
            ax.set_xlim(ax_xlims)
            ax.set_ylim(0,3)

        multipdf(Path.joinpath(out_dir,'DMSO_indiv_contact_kinetics.pdf'))
        
        plt.close('all')





    ## For the Jasplakinolide100nM treatment:
    if expt_cond_nm == 'Jasplak100nM':
        for name, single_contact_df in expt_cond_grps.groupby(['Source File', 'Contact Number']):
    
            
            fig, ax = plt.subplots(figsize=(8,6))
            
            sns.lineplot(x='Time (seconds)', y='Contact Diameter (N.D.)', 
                         data=single_contact_df, marker='.', color='k', 
                         linewidth=1, markeredgewidth=0.3, ax=ax)
            
            sns.lineplot(x='Time (seconds)', y='Smoothed Contact Diameter (N.D.)', 
                         data=single_contact_df, marker='.', color='r', 
                         linewidth=1, markeredgewidth=0.3, ax=ax)
            ax.set_title('%s : Contact Number %d'%(*name,))
            ax.set_xlim(ax_xlims)
            ax.set_ylim(0,3)
    
        multipdf(Path.joinpath(out_dir,'jasplak100nM_indiv_contact_kinetics.pdf'))
        
        plt.close('all')



    ## For the Jasplakinolide200nM treatment:
    if expt_cond_nm == 'Jasplak200nM':
        for name, single_contact_df in expt_cond_grps.groupby(['Source File', 'Contact Number']):
    
            
            fig, ax = plt.subplots(figsize=(8,6))
            
            sns.lineplot(x='Time (seconds)', y='Contact Diameter (N.D.)', 
                         data=single_contact_df, marker='.', color='k', 
                         linewidth=1, markeredgewidth=0.3, ax=ax)
            
            sns.lineplot(x='Time (seconds)', y='Smoothed Contact Diameter (N.D.)', 
                         data=single_contact_df, marker='.', color='r', 
                         linewidth=1, markeredgewidth=0.3, ax=ax)
            ax.set_title('%s : Contact Number %d'%(*name,))
            ax.set_xlim(ax_xlims)

    
        multipdf(Path.joinpath(out_dir,'jasplak200nM_indiv_contact_kinetics.pdf'))
        
        plt.close('all')



# exit()















sf_grp, data_list = list(zip(*jasplak_results_df.groupby(['Source File', 'Contact Number'])))

###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
### The following is used only to help me manually select where the low plateau
### ends for each timelapse of a cell-cell adhesion event.
### BTW: all lines that start with a single "#" should be un-commented up to 
### the next full line of "#"

# if True:
#     pass
#     def pickable_scatter(fig, ax, x, y, smth_x, smth_y):
#         def on_click(event):
#             ind = event.ind
#             print('\nind:', ind, '\ntime:', x[ind], '\ny_val:', y[ind])
#         # ax.scatter(x, y, picker=True)
#         ax.scatter(smth_x, smth_y, marker='.', s=1, c='r', picker=True)
#         # ax.plot(x, y, c='k', marker='.', ls='-', lw=1, ms=2)
#         cid = fig.canvas.mpl_connect('pick_event', on_click)
        
        
#     ### For the following interactive plots to work, you need to switch matplotlib
#     ### from 'inline' to the 'qt' backend (ie. the plots won't show up in the 
#     ### console anymore). This, however, won't work in the script so you need to 
#     ### type it in the console yourself:
#     %matplotlib qt
#     # plt.rcParams.update()
#     plt.rcParams.update({'font.size':10})
    
    
#     ### Contact Diameters:
    
#     # i = 14
#     # description = sf_grp[i]
#     # x, y = data_list[i]['Time (seconds)'].values, data_list[i]['Contact Diameter (N.D.)'].values
#     # smth_x, smth_y = data_list[i]['Time (seconds)'].values, data_list[i]['Smoothed Contact Diameter (N.D.)'].values
    
#     # fig, ax = plt.subplots(figsize=(9,4.5))
#     # pickable_scatter(fig, ax, x, y, smth_x, smth_y)
#     # ax.plot(x, y, c='k', marker='.', ls='-', lw=0.5, ms=1, zorder=0)
#     # ax.plot(smth_x, smth_y, c='r', ls='-', lw=0.5, zorder=1)
#     # ax.set_title('i=%d: %s: Contact Number %d'%(i,*description,))
#     # ax.set_xlim(ax_xlims)
#     # ax.set_ylim(0,3)

        
#     plt.rcParams.update({'font.size':2})

#     i = 33
#     description = sf_grp[i]
#     x, y = data_list[i]['Time (seconds)'].values, data_list[i]['Contact Diameter (N.D.)'].values
#     smth_x, smth_y = data_list[i]['Time (seconds)'].values, data_list[i]['Smoothed Contact Diameter (N.D.)'].values
    
#     fig, ax = plt.subplots(figsize=(2.1,1.4))
#     pickable_scatter(fig, ax, x, y, smth_x, smth_y)
#     ax.plot(x, y, c='k', marker='.', ls='-', lw=0.1, ms=0.2, zorder=0)
#     ax.plot(smth_x, smth_y, c='r', ls='-', lw=0.1, ms=0.2, zorder=1)
#     ax.set_title('i=%d: %s: Contact Number %d'%(i,*description,))
#     ax.set_xlim(ax_xlims)
#     ax.set_ylim(0,3)

###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################




## Conclusions are still obfuscated by a lot of data that does not reflect 
## contact growth (e.g. a contact remaining at a low plateau for a while), so
## using the interactive plot above and by adjusting i to go through the whole
## of data_list, I've picked out the periods of contact growth as follows:
growth_periods = (#('i', 'Growth Start Time', 'Growth End Time'),
                  ## This portion is for Jasplak 200nM:
                  (0, 10, 1570), 
                  (1, 540, 1570), 
                  (2, 620, 1400), 
                  (3, 840, 1610), 
                  (4, 440, 1360), 
                  (5, 860, 1750), 
                  ## This portion is for Jasplak200nM (2023-01-20):
                  (6, 910, 1370), 
                  (7, 1040, 2200), 
                  (8, 710, 1890), 
                  (9, 1660, 2440), 
                  (10, 920, 2100), 
                  (11, 1460, 2750), 
                  (12, 140, 2740), 
                  ## This portion is for DMSO (2023-01-27):
                  (13, 4670, 5300), 
                  (14, 1620, 2740),
                  (15, 2680, 3980), 
                  (16, 2960, 4380), 
                  (17, 4700, 5410), 
                  (18, 3660, 4480), 
                  (19, 2000, 3520), 
                  ## This portion is for Jasplak200nM (2023-01-27):
                  (20, 1880, 3650), 
                  (21, 2680, 4310), 
                  (22, 1310, 3260),
                  (23, 2750, 4150),
                  (24, 2650, 4620), 
                  (25, 1100, 4400), 
                  (26, 3090, 3960), 
                  (27, 2020, 3060), 
                  (28, 1640, 5400), 
                  ## This portion is for DMSO (2023-02-16):
                  (29, 1610, 3280), 
                  (30, 2480, 5280), 
                  (31, 2560, 3100), 
                  (32, 10, 2600), 
                  (33, 980, 1900), 
                  (34, 1370, 1870), 
                  (35, 1170, 2010), 
                  (36, 1040, 2750), 
                  ## This portion is for Jasplak200nM (2023-02-16):
                  (37, 3080, 3500), 
                  (38, 680, 1240), 
                  (39, 1040, 3010), 
                  (40, 2700, 3760), 
                  (41, 420, 4510), 
                  (42, 1330, 2180), 
                  (43, 10, 2210), 
                  (44, 570, 2320), 
                  (45, 800, 3500), 
                  )
    


# growth_per_dict = dict(zip(growth_periods[0], zip(*growth_periods[1:])))


for i,df in enumerate(data_list):
    description = sf_grp[i]
    
    fig, ax = plt.subplots(figsize=(8,6))
    
    
    sns.lineplot(x='Time (seconds)', y='Contact Diameter (N.D.)', 
                 data=df, marker='.', color='k', 
                 linewidth=1, markeredgewidth=0.3, ax=ax)
    
    sns.lineplot(x='Time (seconds)', y='Smoothed Contact Diameter (N.D.)', 
                 data=df, marker='.', color='r', 
                 linewidth=1, markeredgewidth=0.3, ax=ax)
    ax.axvline(growth_periods[i][1], lw=1, ls='--')
    ax.axvline(growth_periods[i][2], lw=1, ls='--')
    ax.axhline(1.76, c='tab:green', lw=1)
    ax.set_title('i=%d: %s: Contact Number %d'%(i,*description,))
    ax.set_xlim(ax_xlims)
    ax.set_ylim(0,3)
    
    plt.plot()


multipdf(Path.joinpath(out_dir,'growth_periods.pdf'))

plt.close('all')


growth_phase_cont_roc_ls = list()
for i,df in enumerate(data_list):
    name = sf_grp[i]
    df = df[np.logical_and(df['Time (seconds)'] >= growth_periods[i][1], df['Time (seconds)'] <= growth_periods[i][2])]
    
    ## I also want to have a column with Time normalized to the start of the growth
    ## period:
    df['Time rel. to Growth Start (seconds)'] = df['Time (seconds)'] - growth_periods[i][1]

    growth_phase_cont_roc_ls.append(df)

growth_phase_cont_roc_df = pd.concat(growth_phase_cont_roc_ls)


## The VESM model only makes predictions about contact growth, not shrinkage,
## therefore we must only consider positive growth rates:
## This is for whole timecourses:
pos_smooth_cont_roc_df = jasplak_results_df[jasplak_results_df['Smoothed Cont. Diam. Rate of Change (N.D./sec)'] > 0]
pos_cont_roc_df = jasplak_results_df[jasplak_results_df['Cont. Diam. Rate of Change (N.D./sec)'] > 0]

## this is for just the growth phases:
pos_smooth_growth_phase_cont_roc_df = growth_phase_cont_roc_df[growth_phase_cont_roc_df['Smoothed Cont. Diam. Rate of Change (N.D./sec)'] > 0]
pos_growth_phase_cont_roc_df = growth_phase_cont_roc_df[growth_phase_cont_roc_df['Cont. Diam. Rate of Change (N.D./sec)'] > 0]

## Cannot have any np.nans in columns that are compared with statistical tests:
jasplak_results_nona = pos_growth_phase_cont_roc_df.dropna(subset=['Cont. Diam. Rate of Change (N.D./sec)'])
smooth_jasplak_results_nona = pos_smooth_growth_phase_cont_roc_df.dropna(subset=['Cont. Diam. Rate of Change (N.D./sec)'])


### Normalize positive contact growth rate data to DMSO median for each
### experiment before plotting as box plots:
## Return the median magnitude in the contact rate of change of the DMSO group 
## from each experiment as a dictionary:
pos_DMSO_medians_per_exptID = smooth_jasplak_results_nona.query('`Experimental Condition` == "DMSO"').groupby('Experimental ID')['Smoothed Cont. Diam. Rate of Change (N.D./sec)'].median().to_dict()

## Normalize the experimental condition from each experiment according to these
## median values and put the result in a new column:
# norm_mag_cont_roc = growth_phase_abs_nona.groupby('Experimental ID').apply(lambda x: x['Mag. Cont. Diam. Rate of Change (N.D./sec)'] / x['Experimental ID'].map(DMSO_means_per_exptID))
norm_pos_cont_roc = pd.concat([grp['Smoothed Cont. Diam. Rate of Change (N.D./sec)'] / grp['Experimental ID'].map(pos_DMSO_medians_per_exptID) for nm,grp in smooth_jasplak_results_nona.groupby('Experimental ID')])
smooth_jasplak_results_nona['Norm. Smoothed Cont. Diam. Rate of Change (N.D./sec)'] = norm_pos_cont_roc








## Provide some summary statistics:
dcontdt_summary = pos_growth_phase_cont_roc_df.groupby('Experimental Condition')['Cont. Diam. Rate of Change (N.D./sec)'].describe()
dcontdt_summary.to_csv(Path.joinpath(out_dir, 'dcontdt_summary.csv'))

dcontdt_smooth_summary = pos_smooth_growth_phase_cont_roc_df.groupby('Experimental Condition')['Smoothed Cont. Diam. Rate of Change (N.D./sec)'].describe()
dcontdt_smooth_summary.to_csv(Path.joinpath(out_dir, 'dcontdt_smooth_summary.csv'))

dcontdt_norm_smooth_summary = smooth_jasplak_results_nona.groupby('Experimental Condition')['Norm. Smoothed Cont. Diam. Rate of Change (N.D./sec)'].describe()
dcontdt_norm_smooth_summary.to_csv(Path.joinpath(out_dir, 'dcontdt_norm_smooth_summary.csv'))



## Save the medians in a .txt file:
nms, norm_pos_smooth_grps = list(zip(*smooth_jasplak_results_nona.groupby('Experimental Condition')['Norm. Smoothed Cont. Diam. Rate of Change (N.D./sec)']))

norm_pos_smooth_grps_medians = [x.median() for x in norm_pos_smooth_grps]
out_file = Path.joinpath(out_dir, 'ExptCond_medians_norm_smooth_pos.txt')
with open(out_file, 'w') as f:
    f.writelines(', '.join([str(x) for x in norm_pos_smooth_grps_medians]))
f.close()



## Run statistical test on grouped data:
pos_dmsoJ100J200_ftest = stats.f_oneway(*norm_pos_smooth_grps)

# ## --Tukey posthoc test for each combination of 2 groups:
# pos_tukey = pairwise_tukeyhsd(
#                 endog=smooth_jasplak_results_nona['Norm. Smoothed Cont. Diam. Rate of Change (N.D./sec)'], 
#                 groups=smooth_jasplak_results_nona['Experimental Condition'], 
#                 alpha=0.05)

# print(pos_tukey)
# out_file = Path.joinpath(out_dir, 'ExptCond_stats_norm_smooth_pos.txt')
# with open(out_file, 'w') as f:
#     f.writelines(str(pos_tukey))
# f.close()




## plot the rates of change as violinplots overall and compare different 
## experimental conditions (e.g. Jasplak 100nM vs. corresponding DMSO control):
fig, ax = plt.subplots(figsize=(5,3.75))
# sns.violinplot(x='Experimental Condition', 
sns.boxplot(x='Experimental Condition', 
            y='Norm. Smoothed Cont. Diam. Rate of Change (N.D./sec)', 
            data=smooth_jasplak_results_nona, ax=ax)
plt.tight_layout()
fig.savefig(Path.joinpath(out_dir, 'norm_smooth_pos_cont_roc.tif'), bbox='tight')


fig, ax = plt.subplots(figsize=(5,3.75))
# sns.violinplot(x='Experimental Condition', 
sns.boxplot(x='Experimental Condition', 
            y='Smoothed Cont. Diam. Rate of Change (N.D./sec)', 
            data=smooth_jasplak_results_nona, ax=ax)
plt.tight_layout()
fig.savefig(Path.joinpath(out_dir, 'smooth_pos_cont_roc.tif'), bbox='tight')


fig, ax = plt.subplots(figsize=(5,3.75))
# sns.violinplot(x='Experimental Condition', 
sns.boxplot(x='Experimental Condition', 
            y='Cont. Diam. Rate of Change (N.D./sec)', 
            data=jasplak_results_nona, ax=ax)
plt.tight_layout()
fig.savefig(Path.joinpath(out_dir, 'pos_cont_roc.tif'), bbox='tight')





## Group by experimental condition:
nms, grps = list(zip(*jasplak_results_nona.groupby('Experimental Condition')['Cont. Diam. Rate of Change (N.D./sec)']))
nms, smooth_grps = list(zip(*smooth_jasplak_results_nona.groupby('Experimental Condition')['Smoothed Cont. Diam. Rate of Change (N.D./sec)']))

## Run statistical test on grouped data:
print(stats.f_oneway(*grps))
print(stats.f_oneway(*smooth_grps))




## Out of curiousity this is for the contacts including non-positive rates of 
## change:
growth_phase_cont_roc_df_nona = growth_phase_cont_roc_df.dropna(subset=['Cont. Diam. Rate of Change (N.D./sec)'])
smooth_growth_phase_cont_roc_df_nona = growth_phase_cont_roc_df.dropna(subset=['Cont. Diam. Rate of Change (N.D./sec)'])

nms_test, grps_test = list(zip(*growth_phase_cont_roc_df_nona.groupby('Experimental Condition')['Cont. Diam. Rate of Change (N.D./sec)']))
nms_test, smooth_grps_test = list(zip(*growth_phase_cont_roc_df_nona.groupby('Experimental Condition')['Smoothed Cont. Diam. Rate of Change (N.D./sec)']))

dcontdt_summary_test = growth_phase_cont_roc_df.groupby('Experimental Condition')['Cont. Diam. Rate of Change (N.D./sec)'].describe()
dcontdt_smooth_summary_test = growth_phase_cont_roc_df.groupby('Experimental Condition')['Smoothed Cont. Diam. Rate of Change (N.D./sec)'].describe()
dcontdt_summary_test_sf = growth_phase_cont_roc_df_nona.groupby('Source File')['Cont. Diam. Rate of Change (N.D./sec)'].describe()
dcontdt_smooth_summary_test_sf = growth_phase_cont_roc_df_nona.groupby('Source File')['Smoothed Cont. Diam. Rate of Change (N.D./sec)'].describe()
growth_phase_cont_roc_df_nona.groupby('Source File')['Cont. Diam. Rate of Change (N.D./sec)'].median()
growth_phase_cont_roc_df_nona.groupby('Source File')['Smoothed Cont. Diam. Rate of Change (N.D./sec)'].median()

print(stats.f_oneway(*grps_test))
print(stats.f_oneway(*smooth_grps_test))


## The groups overall seem very different (p < 0.001), but there are thousands 
## of data points... how do individual experiments compare to one another?
## (i.e. Jasplak dose vs DMSO, since those cells are derived from the same 
## tissue explants):



## First need to connect DMSO source files to Jasplak source files:

# make a column called 'Expt. No.', and have one for each DMSO-Jasplak. pair
# Expt. No. should go up to 5 (3 DMSO-Jasplak200nM pairs, 2 DMSO-Jasplak100nM 
# pairs):

plt.close('all')
for expt_id, expt_id_grp in growth_phase_cont_roc_df.groupby('Experimental ID'):
    # if expt_id == 'Jasplak100nM : 2023-01-20': # test which will be commented out
    fig, ax = plt.subplots()
    # for expt_cond, expt_cond_grp in expt_id_grp.groupby('Experimental Condition'):
    #     print(expt_cond)
    #     if expt_cond == 'DMSO':
    #         # pal = 'GnBu'
    #         pal = 'Blues'
    #         # pal = sns.color_palette("flare", as_cmap=True)
    #         # pal = "crest"
    #     else:
    #         # pal = 'OrRd'
    #         pal = 'Reds'
    #         # pal = "flare"
    #     sns.lineplot(x='Time rel. to Growth Start (seconds)', y='Contact Diameter (N.D.)', 
    #                  hue='Contact Number', palette=pal, data=expt_cond_grp, 
    #                  legend=False, ax=ax)
    sns.lineplot(x='Time rel. to Growth Start (seconds)', y='Contact Diameter (N.D.)', 
                 units='Contact Number', hue='Experimental Condition', data=expt_id_grp, 
                 palette=['tab:grey', 'tab:red'], alpha=0.7, 
                 estimator=None, legend='brief', ax=ax)        
    ax.set_title(expt_id)
    # fig.savefig(Path.joinpath(out_dir, 'rel_timecourses_%s.tif'
    #                           %'_'.join(re.split(' : ',expt_id))), bbox='tight')
## Save the plots in a multipge PDF:
multipdf(Path.joinpath(out_dir,'rel_timecourses.pdf'))
plt.close('all')




plt.close('all')
for expt_id, expt_id_grp in growth_phase_cont_roc_df.groupby('Experimental ID'):
    fig, ax = plt.subplots()
    # expt_id_grp['Smoothed Contact Diameter (N.D.) zeroed to initial'] = expt_id_grp.groupby(['Source File', 'Contact Number'])['Smoothed Contact Diameter (N.D.)'].apply(lambda x: x - x.iloc[0])
    # sns.lineplot(x='Time rel. to Growth Start (seconds)', y='Smoothed Contact Diameter (N.D.) zeroed to initial', 
    sns.lineplot(x='Time rel. to Growth Start (seconds)', y='Smoothed Contact Diameter (N.D.)', 
                 units='Contact Number', hue='Experimental Condition', data=expt_id_grp, 
                 palette=['tab:grey', 'tab:red'], alpha=0.7, 
                 estimator=None, legend='brief', ax=ax)        
    ax.set_title(expt_id)
    # fig.savefig(Path.joinpath(out_dir, 'smooth_rel_timecourses_%s.tif'
    #                           %'_'.join(re.split(' : ',expt_id))), bbox='tight')
## Save the plots in a multipge PDF:
multipdf(Path.joinpath(out_dir,'smooth_rel_timecourses.pdf'))
plt.close('all')



plt.close('all')
for expt_id, expt_id_grp in growth_phase_cont_roc_df.groupby('Experimental ID'):
    fig, ax = plt.subplots()
    sns.lineplot(x='Time (seconds)', y='Contact Diameter (N.D.)', 
                 units='Contact Number', hue='Experimental Condition', data=expt_id_grp, 
                 palette=['tab:grey', 'tab:red'], alpha=0.7, 
                 estimator=None, legend='brief', ax=ax)        
    ax.set_title(expt_id)
    # fig.savefig(Path.joinpath(out_dir, 'timecourses_%s.tif'
    #                           %'_'.join(re.split(' : ',expt_id))), bbox='tight')
## Save the plots in a multipge PDF:
multipdf(Path.joinpath(out_dir,'timecourses.pdf'))
plt.close('all')



plt.close('all')
for expt_id, expt_id_grp in growth_phase_cont_roc_df.groupby('Experimental ID'):
    fig, ax = plt.subplots()
    sns.lineplot(x='Time (seconds)', y='Smoothed Contact Diameter (N.D.)', 
                 units='Contact Number', hue='Experimental Condition', data=expt_id_grp, 
                 palette=['tab:grey', 'tab:red'], alpha=0.7, 
                 estimator=None, legend='brief', ax=ax)        
    ax.set_title(expt_id)
    # fig.savefig(Path.joinpath(out_dir, 'smooth_timecourses_%s.tif'
    #                           %'_'.join(re.split(' : ',expt_id))), bbox='tight')
## Save the plots in a multipge PDF:
multipdf(Path.joinpath(out_dir,'smooth_timecourses.pdf'))
plt.close('all')



## Plot violin plots of the contact rates of change over time for each expt:
plt.close('all')
print('\nRaw:')
indiv_expt_stats_raw = list()
for expt_id, expt_id_grp in jasplak_results_nona.groupby('Experimental ID'):
    fig, ax = plt.subplots()
    # sns.violinplot(x='Experimental Condition', y='Cont. Diam. Rate of Change (N.D./sec)', 
    sns.boxplot(x='Experimental Condition', 
                   y='Cont. Diam. Rate of Change (N.D./sec)', 
                   data=expt_id_grp, palette=['tab:grey', 'tab:red'], 
                   ax=ax)        
    ax.set_title(expt_id)
    # fig.savefig(Path.joinpath(out_dir, 'cont_roc_%s.tif'
    #                           %'_'.join(re.split(' : ',expt_id))), bbox='tight')
    
    ## Run some stats on each experiment:
    expt_id_grp_pos = expt_id_grp[expt_id_grp['Cont. Diam. Rate of Change (N.D./sec)'] > 0]
    expt_id_grp_pos_nona = expt_id_grp.dropna(subset=['Cont. Diam. Rate of Change (N.D./sec)'])
    nms, grps = list(zip(*expt_id_grp_pos_nona.groupby('Experimental Condition')['Cont. Diam. Rate of Change (N.D./sec)']))
    indiv_expt_stats_raw.append(' = '.join([expt_id, str(stats.f_oneway(*grps))]) + '\n')
    print(expt_id, stats.f_oneway(*grps))
## Save the plots in a multipge PDF:
multipdf(Path.joinpath(out_dir,'cont_roc.pdf'))
plt.close('all')

## Save the stats in a .txt file:    
out_file = Path.joinpath(out_dir, 'Individual experiment stats positive.txt')
with open(out_file, 'w') as f:
    f.writelines(indiv_expt_stats_raw)
f.close()



## Plot violin plots of the smoothed contact rates of change over time for each expt:
plt.close('all')
print('\nSmooth:')
indiv_expt_stats_smooth = list()
for expt_id, expt_id_grp in smooth_jasplak_results_nona.groupby('Experimental ID'):
    fig, ax = plt.subplots()
    # sns.violinplot(x='Experimental Condition', 
    sns.boxplot(x='Experimental Condition', 
                   y='Smoothed Cont. Diam. Rate of Change (N.D./sec)', 
                   data=expt_id_grp, palette=['tab:grey', 'tab:red'], 
                   ax=ax)        
    ax.set_title(expt_id)
    # fig.savefig(Path.joinpath(out_dir, 'smooth_cont_roc_%s.tif'
    #                           %'_'.join(re.split(' : ',expt_id))), bbox='tight')
    
    ## Run some stats on each experiment:
    expt_id_grp_pos = expt_id_grp[expt_id_grp['Smoothed Cont. Diam. Rate of Change (N.D./sec)'] > 0]
    expt_id_grp_pos_nona = expt_id_grp_pos.dropna(subset=['Smoothed Cont. Diam. Rate of Change (N.D./sec)'])
    nms, grps = list(zip(*expt_id_grp_pos_nona.groupby('Experimental Condition')['Smoothed Cont. Diam. Rate of Change (N.D./sec)']))
    indiv_expt_stats_smooth.append(' = '.join([expt_id, str(stats.f_oneway(*grps))]) + '\n')
    print(expt_id, stats.f_oneway(*grps))
## Save the plots in a multipge PDF:
multipdf(Path.joinpath(out_dir,'smooth_cont_roc.pdf'))
plt.close('all')

## Save the stats in a .txt file:
out_file = Path.joinpath(out_dir, 'Individual experiment stats smooth positive.txt')
with open(out_file, 'w') as f:
    f.writelines(indiv_expt_stats_smooth)
f.close()


plt.close('all')
print('\nNorm. Smooth:')
indiv_expt_stats_smooth = list()
for expt_id, expt_id_grp in smooth_jasplak_results_nona.groupby('Experimental ID'):
    fig, ax = plt.subplots()
    # sns.violinplot(x='Experimental Condition', 
    sns.boxplot(x='Experimental Condition', 
                   y='Norm. Smoothed Cont. Diam. Rate of Change (N.D./sec)', 
                   data=expt_id_grp, palette=['tab:grey', 'tab:red'], 
                   ax=ax)        
    ax.set_title(expt_id)
    # fig.savefig(Path.joinpath(out_dir, 'smooth_cont_roc_%s.tif'
    #                           %'_'.join(re.split(' : ',expt_id))), bbox='tight')
    
    ## Run some stats on each experiment:
    expt_id_grp_pos = expt_id_grp[expt_id_grp['Norm. Smoothed Cont. Diam. Rate of Change (N.D./sec)'] > 0]
    expt_id_grp_pos_nona = expt_id_grp_pos.dropna(subset=['Norm. Smoothed Cont. Diam. Rate of Change (N.D./sec)'])
    nms, grps = list(zip(*expt_id_grp_pos_nona.groupby('Experimental Condition')['Smoothed Cont. Diam. Rate of Change (N.D./sec)']))
    indiv_expt_stats_smooth.append(' = '.join([expt_id, str(stats.f_oneway(*grps))]) + '\n')
    print(expt_id, stats.f_oneway(*grps))
## Save the plots in a multipge PDF:
multipdf(Path.joinpath(out_dir,'smooth_cont_roc.pdf'))
plt.close('all')

## Save the stats in a .txt file:
out_file = Path.joinpath(out_dir, 'Individual experiment stats norm smooth positive.txt')
with open(out_file, 'w') as f:
    f.writelines(indiv_expt_stats_smooth)
f.close()





## Compare magnitude of dcontdt when including both pos and neg values
growth_phase_abs_nona = growth_phase_cont_roc_df.dropna(subset=['Cont. Diam. Rate of Change (N.D./sec)'])
growth_phase_abs_nona['Mag. Cont. Diam. Rate of Change (N.D./sec)'] = growth_phase_cont_roc_df['Cont. Diam. Rate of Change (N.D./sec)'].abs()
growth_phase_abs_nona['Mag. Smoothed Cont. Diam. Rate of Change (N.D./sec)'] = growth_phase_cont_roc_df['Smoothed Cont. Diam. Rate of Change (N.D./sec)'].abs()


## Return the mean magnitude in the contact rate of change of the DMSO group 
## from each experiment as a dictionary:
DMSO_medians_per_exptID = growth_phase_abs_nona.query('`Experimental Condition` == "DMSO"').groupby('Experimental ID')['Mag. Smoothed Cont. Diam. Rate of Change (N.D./sec)'].median().to_dict()
# DMSO_means_per_exptID = growth_phase_abs_nona.query('`Experimental Condition` == "DMSO"').groupby('Experimental ID')['Mag. Smoothed Cont. Diam. Rate of Change (N.D./sec)'].mean().to_dict()


## Normalize the experimental condition from each experiment according to these
## mean values and put the result in a new column:
# norm_mag_cont_roc = growth_phase_abs_nona.groupby('Experimental ID').apply(lambda x: x['Mag. Cont. Diam. Rate of Change (N.D./sec)'] / x['Experimental ID'].map(DMSO_means_per_exptID))
norm_mag_cont_roc = pd.concat([grp['Mag. Smoothed Cont. Diam. Rate of Change (N.D./sec)'] / grp['Experimental ID'].map(DMSO_medians_per_exptID) for nm,grp in growth_phase_abs_nona.groupby('Experimental ID')])
# norm_mag_cont_roc = pd.concat([grp['Mag. Smoothed Cont. Diam. Rate of Change (N.D./sec)'] / grp['Experimental ID'].map(DMSO_means_per_exptID) for nm,grp in growth_phase_abs_nona.groupby('Experimental ID')])
growth_phase_abs_nona['Norm. Mag. Smoothed Cont. Diam. Rate of Change (N.D./sec)'] = norm_mag_cont_roc




fig, ax = plt.subplots(figsize=(5,3.75))
# sns.violinplot(x=growth_phase_abs_nona['Experimental Condition'], 
sns.boxplot(x=growth_phase_abs_nona['Experimental Condition'], 
               y=growth_phase_abs_nona['Norm. Mag. Smoothed Cont. Diam. Rate of Change (N.D./sec)'], 
               ax=ax)
ax.set_ylabel('Norm. Mag. Smoothed Cont. Diam. \nRate of Change (N.D./sec)')
# ax.set_xticklabels(['DMSO', 'Jasplakinolide (200nM)'])
ax.set_xticklabels(['DMSO', 'Jasplak. (100nM)', 'Jasplak. (200nM)'])
plt.tight_layout()
fig.savefig(Path.joinpath(out_dir, 'norm_smooth_mag_cont_roc.tif'))


fig, ax = plt.subplots(figsize=(5,3.75))
# sns.violinplot(x=growth_phase_abs_nona['Experimental Condition'], 
sns.boxplot(x=growth_phase_abs_nona['Experimental Condition'], 
               y=growth_phase_abs_nona['Mag. Smoothed Cont. Diam. Rate of Change (N.D./sec)'], 
               ax=ax)
ax.set_ylabel('Mag. Smoothed Cont. Diam. \nRate of Change (N.D./sec)')
# ax.set_xticklabels(['DMSO', 'Jasplakinolide (200nM)'])
ax.set_xticklabels(['DMSO', 'Jasplak. (100nM)', 'Jasplak. (200nM)'])
plt.tight_layout()
fig.savefig(Path.joinpath(out_dir, 'smooth_mag_cont_roc.tif'))


fig, ax = plt.subplots(figsize=(5,3.75))
# sns.violinplot(x=growth_phase_abs_nona['Experimental Condition'], 
sns.boxplot(x=growth_phase_abs_nona['Experimental Condition'], 
               y=growth_phase_abs_nona['Mag. Cont. Diam. Rate of Change (N.D./sec)'], 
               ax=ax)
ax.set_ylabel('Mag. Cont. Diam. \nRate of Change (N.D./sec)')
# ax.set_xticklabels(['DMSO', 'Jasplakinolide (200nM)'])
ax.set_xticklabels(['DMSO', 'Jasplak. (100nM)', 'Jasplak. (200nM)'])
plt.tight_layout()
fig.savefig(Path.joinpath(out_dir, 'mag_cont_roc.tif'))





## Group by experimental condition:
nms, mag_grps = list(zip(*growth_phase_abs_nona.groupby('Experimental Condition')['Mag. Cont. Diam. Rate of Change (N.D./sec)']))
nms, mag_smooth_grps = list(zip(*growth_phase_abs_nona.groupby('Experimental Condition')['Mag. Smoothed Cont. Diam. Rate of Change (N.D./sec)']))
nms, norm_mag_smooth_grps = list(zip(*growth_phase_abs_nona.groupby('Experimental Condition')['Norm. Mag. Smoothed Cont. Diam. Rate of Change (N.D./sec)']))

## Run statistical test on grouped data:
print(stats.kruskal(*mag_grps))
print(stats.kruskal(*mag_smooth_grps))
dmsoJ100J200_ftest = stats.kruskal(*norm_mag_smooth_grps)


# ## Run Tukey posthoc test on each pair of groups:
# tukey = pairwise_tukeyhsd(
#     endog=growth_phase_abs_nona['Norm. Mag. Smoothed Cont. Diam. Rate of Change (N.D./sec)'], 
#     groups=growth_phase_abs_nona['Experimental Condition'], 
#     alpha=0.05)

# print(tukey)
# out_file = Path.joinpath(out_dir, 'ExptCond_stats_norm_smooth_mag.txt')
# with open(out_file, 'w') as f:
#     f.writelines(str(tukey))
# f.close()


# dmsoJ100_ftest = stats.ttest_1samp(growth_phase_abs_nona.query('`Experimental Condition` == "Jasplak100nM"')['Norm. Mag. Smoothed Cont. Diam. Rate of Change (N.D./sec)'], 1)
# dmsoJ200_ftest = stats.ttest_1samp(growth_phase_abs_nona.query('`Experimental Condition` == "Jasplak200nM"')['Norm. Mag. Smoothed Cont. Diam. Rate of Change (N.D./sec)'], 1)
# J100J200_ftest = stats.ttest_ind(growth_phase_abs_nona.query('`Experimental Condition` == "Jasplak100nM"')['Norm. Mag. Smoothed Cont. Diam. Rate of Change (N.D./sec)'], 
#                                  growth_phase_abs_nona.query('`Experimental Condition` == "Jasplak200nM"')['Norm. Mag. Smoothed Cont. Diam. Rate of Change (N.D./sec)'], 
#                                  equal_var=False)




## Save the medians in a .txt file:
norm_mag_smooth_grps_medians = [x.median() for x in norm_mag_smooth_grps]
out_file = Path.joinpath(out_dir, 'ExptCond_medians_norm_smooth_mag.txt')
with open(out_file, 'w') as f:
    f.writelines(', '.join([str(x) for x in norm_mag_smooth_grps_medians]))
f.close()




print('median: ', growth_phase_abs_nona.groupby('Experimental Condition')['Mag. Cont. Diam. Rate of Change (N.D./sec)'].median())
print('mean: ', growth_phase_abs_nona.groupby('Experimental Condition')['Mag. Cont. Diam. Rate of Change (N.D./sec)'].mean())

print('median: ', growth_phase_abs_nona.groupby(['Experimental Condition', 'Experimental ID'])['Mag. Cont. Diam. Rate of Change (N.D./sec)'].median())
print('mean: ', growth_phase_abs_nona.groupby(['Experimental Condition', 'Experimental ID'])['Mag. Cont. Diam. Rate of Change (N.D./sec)'].mean())

growth_phase_abs_nona.groupby('Experimental Condition')['Mag. Cont. Diam. Rate of Change (N.D./sec)'].median().to_csv(Path.joinpath(out_dir, 'overall_mag_medians.csv'))
growth_phase_abs_nona.groupby('Experimental Condition')['Mag. Cont. Diam. Rate of Change (N.D./sec)'].mean().to_csv(Path.joinpath(out_dir, 'overall_mag_means.csv'))

growth_phase_abs_nona.groupby(['Experimental Condition', 'Experimental ID'])['Mag. Cont. Diam. Rate of Change (N.D./sec)'].median().to_csv(Path.joinpath(out_dir, 'mag_medians_by_expt.csv'))
growth_phase_abs_nona.groupby(['Experimental Condition', 'Experimental ID'])['Mag. Cont. Diam. Rate of Change (N.D./sec)'].mean().to_csv(Path.joinpath(out_dir, 'mag_means_by_expt.csv'))


growth_phase_abs_nona.groupby('Experimental Condition')['Norm. Mag. Smoothed Cont. Diam. Rate of Change (N.D./sec)'].median().to_csv(Path.joinpath(out_dir, 'overall_norm_mag_medians.csv'))
growth_phase_abs_nona.groupby('Experimental Condition')['Norm. Mag. Smoothed Cont. Diam. Rate of Change (N.D./sec)'].mean().to_csv(Path.joinpath(out_dir, 'overall_norm_mag_means.csv'))

growth_phase_abs_nona.groupby(['Experimental Condition', 'Experimental ID'])['Norm. Mag. Smoothed Cont. Diam. Rate of Change (N.D./sec)'].median().to_csv(Path.joinpath(out_dir, 'norm_mag_medians_by_expt.csv'))
growth_phase_abs_nona.groupby(['Experimental Condition', 'Experimental ID'])['Norm. Mag. Smoothed Cont. Diam. Rate of Change (N.D./sec)'].mean().to_csv(Path.joinpath(out_dir, 'norm_mag_means_by_expt.csv'))




## Plot the DMSO vs. Jasplak. magnitudes of change for each experiment against 
## each other:
plt.close('all')
indiv_expt_stats_smooth_mag = list()
for expt_id, expt_id_grp in growth_phase_abs_nona.groupby('Experimental ID'):
    fig, ax = plt.subplots()
    # expt_id_grp['Smoothed Contact Diameter (N.D.) zeroed to initial'] = expt_id_grp.groupby(['Source File', 'Contact Number'])['Smoothed Contact Diameter (N.D.)'].apply(lambda x: x - x.iloc[0])
    # sns.lineplot(x='Time rel. to Growth Start (seconds)', y='Smoothed Contact Diameter (N.D.) zeroed to initial', 
    sns.violinplot(x='Experimental Condition', 
    # sns.boxplot(x='Experimental Condition', 
                   y='Norm. Mag. Smoothed Cont. Diam. Rate of Change (N.D./sec)', 
                   data=expt_id_grp, 
                   palette=['tab:grey', 'tab:red'], 
                   ax=ax)        
    ax.set_title(expt_id)
    # fig.savefig(Path.joinpath(out_dir, 'smooth_mag_rel_timecourses_%s.tif'
    #                           %'_'.join(re.split(' : ',expt_id))), bbox='tight')
    ## Run some stats on each experiment:
    expt_id_grp_mag = expt_id_grp[expt_id_grp['Norm. Mag. Smoothed Cont. Diam. Rate of Change (N.D./sec)'] > 0]
    expt_id_grp_mag_nona = expt_id_grp_mag.dropna(subset=['Norm. Mag. Smoothed Cont. Diam. Rate of Change (N.D./sec)'])
    nms, grps = list(zip(*expt_id_grp_mag_nona.groupby('Experimental Condition')['Norm. Mag. Smoothed Cont. Diam. Rate of Change (N.D./sec)']))
    indiv_expt_stats_smooth_mag.append(' = '.join([expt_id, str(stats.f_oneway(*grps))]) + '\n')
    print(expt_id, stats.f_oneway(*grps))
## Save the plots in a multipge PDF:
multipdf(Path.joinpath(out_dir,'norm_smooth_mag_rel_timecourses.pdf'))
plt.close('all')

## Save the stats in a .txt file:
out_file = Path.joinpath(out_dir, 'Individual experiment stats norm smooth mag.txt')
with open(out_file, 'w') as f:
    f.writelines(indiv_expt_stats_smooth_mag)
f.close()





cont_num_means = growth_phase_abs_nona.groupby(['Contact Number', 'Experimental Condition', 'Experimental ID'])['Mag. Smoothed Cont. Diam. Rate of Change (N.D./sec)'].apply(np.median).reset_index()
# cont_num_means = smooth_jasplak_results_nona.groupby(['Contact Number', 'Experimental Condition', 'Experimental ID'])['Smoothed Cont. Diam. Rate of Change (N.D./sec)'].apply(np.mean).reset_index()
# cont_num_means = growth_phase_cont_roc_df.groupby(['Contact Number', 'Experimental Condition', 'Experimental ID'])['Smoothed Cont. Diam. Rate of Change (N.D./sec)'].apply(np.mean).reset_index()
# cont_num_means = growth_phase_abs_nona.groupby(['Contact Number', 'Experimental Condition', 'Experimental ID'])['Norm. Mag. Smoothed Cont. Diam. Rate of Change (N.D./sec)'].apply(np.median).reset_index()

# DMSO_means_per_exptID = cont_num_means.query('`Experimental Condition` == "DMSO"').groupby('Experimental ID')['Mag. Smoothed Cont. Diam. Rate of Change (N.D./sec)'].mean().to_dict()
# norm_mag_cont_roc = pd.concat([grp['Mag. Smoothed Cont. Diam. Rate of Change (N.D./sec)'] / grp['Experimental ID'].map(DMSO_means_per_exptID) for nm,grp in cont_num_means.groupby('Experimental ID')])
# cont_num_means['Norm. Mag. Smoothed Cont. Diam. Rate of Change (N.D./sec)'] = norm_mag_cont_roc
# DMSO_means_per_exptID = cont_num_means.query('`Experimental Condition` == "DMSO"').groupby('Experimental ID')['Smoothed Cont. Diam. Rate of Change (N.D./sec)'].median().to_dict()
# norm_mag_cont_roc = pd.concat([grp['Smoothed Cont. Diam. Rate of Change (N.D./sec)'] / grp['Experimental ID'].map(DMSO_means_per_exptID) for nm,grp in cont_num_means.groupby('Experimental ID')])
# cont_num_means['Norm. Smoothed Cont. Diam. Rate of Change (N.D./sec)'] = norm_mag_cont_roc
DMSO_means_per_exptID = cont_num_means.query('`Experimental Condition` == "DMSO"').groupby('Experimental ID')['Mag. Smoothed Cont. Diam. Rate of Change (N.D./sec)'].median().to_dict()
norm_mag_cont_roc = pd.concat([grp['Mag. Smoothed Cont. Diam. Rate of Change (N.D./sec)'] / grp['Experimental ID'].map(DMSO_means_per_exptID) for nm,grp in cont_num_means.groupby('Experimental ID')])
cont_num_means['Norm. Mag. Smoothed Cont. Diam. Rate of Change (N.D./sec)'] = norm_mag_cont_roc



fig, ax = plt.subplots(figsize=(height_1col, width_1col))
sns.boxplot(x='Experimental Condition', 
            y='Norm. Mag. Smoothed Cont. Diam. Rate of Change (N.D./sec)', 
            # y='Mag. Smoothed Cont. Diam. Rate of Change (N.D./sec)', 
            # y='Smoothed Cont. Diam. Rate of Change (N.D./sec)', 
            # y='Norm. Smoothed Cont. Diam. Rate of Change (N.D./sec)', 
            data=cont_num_means, ax=ax)
sns.swarmplot(x='Experimental Condition', 
               y='Norm. Mag. Smoothed Cont. Diam. Rate of Change (N.D./sec)', 
              # y='Mag. Smoothed Cont. Diam. Rate of Change (N.D./sec)', 
              # y='Smoothed Cont. Diam. Rate of Change (N.D./sec)', 
              # y='Norm. Smoothed Cont. Diam. Rate of Change (N.D./sec)', 
              marker='.', size=7, 
              data=cont_num_means, 
              palette=['k',], ax=ax)
ax.set_ylabel('Magnitude of Contact Diameter\nRate of Change (N.D./sec)')
ax.set_xticklabels(['DMSO', 'Jasplak.\n(200nM)'], )
## This next line will take the existing yticks and use only the integers:
ax.set_yticks(ax.get_yticks()[ax.get_yticks()%1 == 0].astype(int))
# ax.set_xlabel('Experimental\nCondition')
ax.set_xlabel('')
ax.minorticks_on()
ax.tick_params(axis='x', which='minor', bottom=False)
plt.tight_layout()
fig.savefig(Path.joinpath(out_dir, 'mean_norm_mag_cont_roc.tif'), bbox='tight')



nms, norm_mag_smooth_grps = list(zip(*cont_num_means.groupby('Experimental Condition')['Norm. Mag. Smoothed Cont. Diam. Rate of Change (N.D./sec)']))
# nms, norm_mag_smooth_grps = list(zip(*cont_num_means.groupby('Experimental Condition')['Mag. Smoothed Cont. Diam. Rate of Change (N.D./sec)']))
# nms, norm_mag_smooth_grps = list(zip(*cont_num_means.groupby('Experimental Condition')['Smoothed Cont. Diam. Rate of Change (N.D./sec)']))
# stats.f_oneway(*norm_mag_smooth_grps)
mean_norm_smooth_mag_roc_kruskal_stats = stats.kruskal(*norm_mag_smooth_grps)

## Save the stats in a .txt file:
out_file = Path.joinpath(out_dir, 'mean_norm_smooth_mag_roc_stats.txt')
with open(out_file, 'w') as f:
    f.writelines(' = '.join([expt_id, str(mean_norm_smooth_mag_roc_kruskal_stats)]) + '\n')
f.close()

n_cont_num_means = cont_num_means.groupby('Experimental Condition').describe()

print(n_cont_num_means)

# tukey = pairwise_tukeyhsd(
#     endog=cont_num_means['Norm. Mag. Smoothed Cont. Diam. Rate of Change (N.D./sec)'], 
#     # endog=cont_num_means['Mag. Smoothed Cont. Diam. Rate of Change (N.D./sec)'], 
#     # endog=cont_num_means['Smoothed Cont. Diam. Rate of Change (N.D./sec)'], 
#     groups=cont_num_means['Experimental Condition'], 
#     alpha=0.05)

# print(tukey)







## If you start or end with negative growth in the growth phase, that 
## should be excluded from the counts... 
## To do that group your data per contact number and take only the data between
## the first False value in the series and the the last False value in series:

smooth_growth_phase_cont_roc_df_nona['Neg. Smoothed Cont. Diam. Rate of Change'] = smooth_growth_phase_cont_roc_df_nona.groupby('Experimental Condition')['Smoothed Cont. Diam. Rate of Change (N.D./sec)'].apply(lambda x: x < 0)

smooth_growth_phase_cont_roc_df_nona_cln = smooth_growth_phase_cont_roc_df_nona.groupby(['Experimental Condition', 'Source File', 'Contact Number'])['Neg. Smoothed Cont. Diam. Rate of Change'].apply(lambda srs: srs[np.where(srs==False)[0][0]: np.where(srs==False)[0][-1]]).reset_index()

cont_has_neg_df = smooth_growth_phase_cont_roc_df_nona_cln.groupby(['Experimental Condition', 'Source File', 'Contact Number'])['Neg. Smoothed Cont. Diam. Rate of Change'].max().reset_index()




## Number of contacts that go negative during their growth per Expt_Cond:
num_cont_w_neg = cont_has_neg_df.groupby('Experimental Condition')['Neg. Smoothed Cont. Diam. Rate of Change'].sum()
num_cont_w_neg.name = 'Negative'
num_cont_w_pos = cont_has_neg_df.groupby('Experimental Condition').apply(lambda df: df['Neg. Smoothed Cont. Diam. Rate of Change'].count() - df['Neg. Smoothed Cont. Diam. Rate of Change'].sum())
num_cont_w_pos.name = 'Positive'

## Total number of contacts per Expt_Cond:
total_cont = cont_has_neg_df.groupby('Experimental Condition')['Neg. Smoothed Cont. Diam. Rate of Change'].count()
total_cont.name = 'Total'



## Percent of contacts that are negative\positive:
prct_neg = 100 * cont_has_neg_df.groupby('Experimental Condition').apply(lambda df: df['Neg. Smoothed Cont. Diam. Rate of Change'].sum() / df['Neg. Smoothed Cont. Diam. Rate of Change'].count())
prct_neg.name = 'Negative'
prct_pos = 100 * cont_has_neg_df.groupby('Experimental Condition').apply(lambda df: 1 - df['Neg. Smoothed Cont. Diam. Rate of Change'].sum() / df['Neg. Smoothed Cont. Diam. Rate of Change'].count())
prct_pos.name = 'Positive'


prct_cont_df = pd.concat([prct_neg, prct_pos], axis=1)

num_cont_df = pd.concat({num_cont_w_neg.name: num_cont_w_neg, 
                         num_cont_w_pos.name: num_cont_w_pos, 
                         total_cont.name: total_cont, 
                         'Percent Negative': prct_neg}, axis=1).reset_index()
num_cont_df.to_csv(Path.joinpath(out_dir, 'num_cont_df.csv'))


## Horizontal bar graph:
fig, ax = plt.subplots(figsize=(width_1col, width_1col*0.5))
## I want DMSO to be at the top of the bar graph so I will invert the order of 
## the rows in the dataframe with the '[::-1]'
prct_cont_df[::-1].plot(kind='barh', stacked=True, color=['tab:red', 'tab:blue'], 
                  legend=False, ax=ax)
## these next 2 lines are just to make sure that the labels on the data aren't
## incorrect since I will be renaming Jasplak200nM to fit better in the plot:
true_ylabels = [label.get_text() for label in ax.get_yticklabels()]
assert true_ylabels == ['Jasplak200nM', 'DMSO']
ax.set_yticklabels(['Jasplak.\n(200nM)', 'DMSO'], rotation=0)
ax.set_ylabel('')
ax.set_xlabel('Contacts Interrupted by\nNegative Growth (%)')
plt.tight_layout()
fig.savefig(Path.joinpath(out_dir, 'num_cont_w_neg_pos_HA.tif'), dpi=dpi_graph, bbox='tight')


## Vertical bar graph:
fig, ax = plt.subplots(figsize=(width_1col*0.5, width_1col))
## I want DMSO to be at the top of the bar graph so I will invert the order of 
## the rows in the dataframe with the '[::-1]'
prct_cont_df.plot(kind='bar', stacked=True, color=['tab:red', 'tab:blue'], 
                  legend=False, ax=ax)
## these next 2 lines are just to make sure that the labels on the data aren't
## incorrect since I will be renaming Jasplak200nM to fit better in the plot:
true_xlabels = [label.get_text() for label in ax.get_xticklabels()]
assert true_xlabels == ['DMSO', 'Jasplak200nM']
ax.set_xticklabels(['DMSO', 'Jasplak.\n(200nM)'], rotation=90)#, ha='right')
ax.set_xlabel('')
ax.set_ylabel('Contacts Interrupted by\nNegative Growth (%)')
plt.tight_layout()
fig.savefig(Path.joinpath(out_dir, 'num_cont_w_neg_pos_VA.tif'), dpi=dpi_graph, bbox='tight')



# print(num_cont_df)

num_cont_w_neg_pos = pd.concat([num_cont_w_neg, num_cont_w_pos],axis=1).values.astype(int)
num_cont_w_neg_fisher = stats.fisher_exact(num_cont_w_neg_pos)




## Amount of time spent with negative growth
time_in_neg_growth = smooth_growth_phase_cont_roc_df_nona_cln.groupby(['Experimental Condition', 'Source File', 'Contact Number'])['Neg. Smoothed Cont. Diam. Rate of Change'].sum()
time_in_neg_growth.name = 'Number of Negative Timeframes'
time_in_neg_growth = time_in_neg_growth.reset_index()
time_in_neg_growth['Time in Negative Growth (seconds)'] = t_res * time_in_neg_growth['Number of Negative Timeframes']

fig, ax = plt.subplots(figsize=(height_1col, width_1col))
sns.boxplot(x='Experimental Condition', y='Time in Negative Growth (seconds)', 
            data=time_in_neg_growth, ax=ax)
sns.swarmplot(x='Experimental Condition', y='Time in Negative Growth (seconds)', 
              color='k', data=time_in_neg_growth, ax=ax)
plt.tight_layout()
fig.savefig(Path.joinpath(out_dir, 'Time in Negative Growth.tif'), dpi=dpi_graph, bbox='tight')

time_in_neg_growth_krusk = stats.kruskal(*list(zip(*time_in_neg_growth.groupby('Experimental Condition')['Time in Negative Growth (seconds)']))[1],)




## Find consecutive True values in a list so you can count the number of times
## that contact growth stalls.
## Number of times contact enters negative growth:
cont_num_grps = smooth_growth_phase_cont_roc_df_nona_cln.groupby(['Experimental Condition', 'Source File', 'Contact Number'])['Neg. Smoothed Cont. Diam. Rate of Change']
## check that none of the groups start off with negative growth:
assert [x[1]==False for x in cont_num_grps]

neg_growth_events = cont_num_grps.apply(lambda x: np.diff(x.astype(int)))
num_neg_growth_events = neg_growth_events.apply(lambda x: np.count_nonzero(x == 1))
num_neg_growth_events.name = 'Number of Negative Growth Events'
num_neg_growth_events = num_neg_growth_events.reset_index()


## Horizontal bar graph:
fig, ax = plt.subplots(figsize=(width_1col, width_1col*0.5))
sns.boxplot(x='Number of Negative Growth Events', y='Experimental Condition', 
            data=num_neg_growth_events, ax=ax)
sns.swarmplot(x='Number of Negative Growth Events', y='Experimental Condition', 
              color='k', data=num_neg_growth_events, ax=ax)
## This next line will take the existing yticks and use only the integers:
ax.set_xticks(ax.get_xticks()[ax.get_xticks()%1 == 0].astype(int))
ax.set_xlim(0)
## these next 2 lines are just to make sure that the labels on the data aren't
## incorrect since I will be renaming Jasplak200nM to fit better in the plot:
true_ylabels = [label.get_text() for label in ax.get_yticklabels()]
assert true_ylabels == ['DMSO', 'Jasplak200nM']
## Rename Jasplak200nM to fit better in the plot:
ax.set_yticklabels(['DMSO', 'Jasplak.\n(200nM)'], )
ax.set_ylabel('')
plt.tight_layout()
fig.savefig(Path.joinpath(out_dir, 'Number of Negative Growth Events HA.tif'), dpi=dpi_graph, bbox='tight')

## Vertical bar graph:
fig, ax = plt.subplots(figsize=(width_1col*0.5, width_1col))
sns.boxplot(x='Experimental Condition', y='Number of Negative Growth Events', 
            data=num_neg_growth_events, ax=ax)
sns.swarmplot(x='Experimental Condition', y='Number of Negative Growth Events', 
              color='k', data=num_neg_growth_events, ax=ax)
## This next line will take the existing yticks and use only the integers:
ax.set_yticks(ax.get_yticks()[ax.get_yticks()%1 == 0].astype(int))
ax.set_ylim(0)
## these next 2 lines are just to make sure that the labels on the data aren't
## incorrect since I will be renaming Jasplak200nM to fit better in the plot:
true_xlabels = [label.get_text() for label in ax.get_xticklabels()]
assert true_xlabels == ['DMSO', 'Jasplak200nM']
## Rename Jasplak200nM to fit better in the plot:
ax.set_xticklabels(['DMSO', 'Jasplak.\n(200nM)'], rotation=90)
ax.set_xlabel('')
plt.tight_layout()
fig.savefig(Path.joinpath(out_dir, 'Number of Negative Growth Events VA.tif'), dpi=dpi_graph, bbox='tight')


num_neg_growth_krusk = stats.kruskal(*list(zip(*num_neg_growth_events.groupby('Experimental Condition')['Number of Negative Growth Events']))[1],)


## Save the stats in a .txt file:
out_file = Path.joinpath(out_dir, 'neg_growth_stats.txt')
with open(out_file, 'w') as f:
    f.writelines([
        '(associated plot or data file = statistic or odds ratio, p-value)\n', 
        ' = '.join(['num_cont_w_neg_pos.tif', 'FisherExact%s'%str(num_cont_w_neg_fisher)]) + '\n',
        ' = '.join(['Time in Negative Growth.tif', str(time_in_neg_growth_krusk)]) + '\n',
        ' = '.join(['Number of Negative Growth Events.tif', str(num_neg_growth_krusk)]) + '\n'
        ])
f.close()





## Figures of interest: 
## ((date, expt_cond, cont_num), ...)
exmpl_kinetics = (('2023-01-20', 'DMSO', '1'), 
                  ('2023-01-20', 'DMSO', '3'), 
                  ('2023-01-20', 'DMSO', '6'), 
                  ('2023-01-27', 'DMSO', '7'), 
                  ('2023-01-27', 'Jasplak200nM', '1'), 
                  ('2023-01-27', 'Jasplak200nM', '2'), 
                  ('2023-01-27', 'Jasplak200nM', '6'), 
                  ('2023-01-27', 'Jasplak200nM', '9'), 
                  )
## We will save these examples in a folder later, but first check that the 
## folder exists, and if it doesn't then make it:
if Path.exists(Path.joinpath(out_dir, 'selected_kinetics')):
    pass
else:
    Path.mkdir(Path.joinpath(out_dir, 'selected_kinetics'))
exmpl_kinetics_dir = Path.joinpath(out_dir, 'selected_kinetics')


for i,df in enumerate(data_list):
    ## return the (date, expt_cond, cont_num):
    expt_id = df[['Acquisition Date', 'Experimental Condition', 'Contact Number']].iloc[0]
    ## turn it in to a tuple of strings to be consistent with exmpl_kinetics:
    expt_id = tuple([str(x) for x in expt_id])
    
    ## check if this contact kinetics measurement is in exmpl_kinetics:
    if expt_id in exmpl_kinetics:
        
        ## if it is, then plot the figure:
        fig, ax = plt.subplots(figsize=(width_1col, height_1col))
        sns.lineplot(x='Time (seconds)', y='Contact Diameter (N.D.)', 
                     data=df, marker='.', color='k', 
                     linewidth=1, markeredgewidth=0, ax=ax)
        
        sns.lineplot(x='Time (seconds)', y='Smoothed Contact Diameter (N.D.)', 
                     data=df, marker='', color='r', 
                     linewidth=1, ax=ax)
        ax.axvline(growth_periods[i][1], lw=1, ls='--')
        ax.axvline(growth_periods[i][2], lw=1, ls='--')
        ax.axhline(1.76, c='tab:green', lw=1)
        ax.set_xlim(ax_xlims)
        ax.set_ylim(0,3)
        ax.minorticks_on()
        ax.set_ylabel('Contact Diameter (N.D.)')
        plt.tight_layout()
        
        ## and then save the figure in a folder:            
        date, expt_cond, cont_num = *expt_id,
        fig.savefig(Path.joinpath(exmpl_kinetics_dir, '%s_%s_%s.tif'%(date, expt_cond, cont_num)), dpi=dpi_graph, bbox='tight')
        
        # plt.plot()
        plt.close(fig)



## What are the growth periods like between DMSO and Jasplakinolide cells?:
_, growth_start, growth_end = list(zip(*growth_periods))

data_list_new = list()
for i,df in enumerate(data_list):
    df['growth_start'] = growth_start[i]
    df['growth_end'] = growth_end[i]
    df['growth_duration'] = growth_end[i] - growth_start[i]
    
    data_list_new.append(df)

growth_periods_df = pd.concat([df.iloc[:1] for df in data_list_new])

growth_periods_df_summ = growth_periods_df.groupby(['Acquisition Date', 'Experimental Condition', 'Contact Number'])


fig, ax = plt.subplots()
sns.boxplot(x='Experimental Condition', y='growth_duration', 
            data=growth_periods_df, ax=ax)
sns.swarmplot(x='Experimental Condition', y='growth_duration', 
              data=growth_periods_df, color='k', ax=ax)

growth_periods_df.groupby('Experimental Condition')['growth_duration'].describe()
growth_periods_df.groupby('Experimental Condition')['growth_duration'].mean()
growth_periods_df.groupby('Experimental Condition')['growth_duration'].median()



