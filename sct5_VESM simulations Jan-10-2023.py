# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 17:34:55 2020

@author: Serge
"""



## TODO list: nothing for now.
    ## - but coule use multiprocessing to slightly speed up VESM solutions




from pathlib import Path
import pickle
import numpy as np
# from uncertainties import unumpy as unp
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size':24})
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
import matplotlib.cm # used to get a colormap
# from mpl_toolkits.mplot3d import Axes3D


from scipy.integrate import solve_ivp
from scipy.stats import truncnorm
from scipy import interpolate

# from multiprocessing import Process

import time as tm

## Needed for multipdf function to create pdf files:
from matplotlib.backends.backend_pdf import PdfPages


### A lot of figures are going to be opened before they are saved in to a
### multi-page pdf, so increase the max figure open warning:
plt.rcParams.update({'figure.max_open_warning':300})
plt.rcParams.update({'font.size':10})


np.seterr(divide='raise', invalid='raise')


## Define some generally useful functions:
def cm2inch(num):
    return num/2.54




def multipdf(filename):
    """ This function saves a multi-page PDF of all open figures. """
    pdf = PdfPages(filename)
    figs = [plt.figure(f) for f in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pdf, format='pdf' , dpi=80)
    pdf.close()



## a function to find the order of magnitude of a number:
def order_of_mag(number):
    order_of_magnitude = np.floor(np.log10(number))
    return order_of_magnitude




### Set your output directory to be the same location as this script:
## Get the working directory of the script:
current_script_dir = Path(__file__).parent.absolute()

## Set the output directory where analyses will be saved to:
out_dir = Path.joinpath(current_script_dir, r"viscoelast_shell_model_out")
if Path.exists(out_dir) == False:
    Path.mkdir(out_dir)
else:
    pass

## Empirical data can be found in the following directory:
lifeact_analysis_dir = Path.joinpath(current_script_dir, r"lifeact_analysis_figs_out")
zdepth_lifeact_analysis_dir = Path.joinpath(current_script_dir, r"lifeact_Zdepth_analysis_out")


## Create directories to save figures to:
man_plots_dir = Path.joinpath(out_dir, 'manuscript_plots')
if Path.exists(man_plots_dir) == False:
    Path.mkdir(man_plots_dir)
else:
    pass

## make a sub-directory for the z-corrected data:
zcorr_plots_dir = Path.joinpath(man_plots_dir, 'z_depth_corrected_plots')
if Path.exists(zcorr_plots_dir) == False:
    Path.mkdir(zcorr_plots_dir)
else:
    pass





### Set some parameters for saving graphs:
## Set figure widths (and heights) according to journal preferences (in cm) and
## initial parameters of matplotlib:
## The aspect ratio in [width, height]:
aspect_ratio = np.asarray(plt.rcParams["figure.figsize"]) / plt.rcParams["figure.figsize"][0]
## 1 column (preview; or 3 column formats):
width_1col_prev = cm2inch(5.5)
height_1col_prev = width_1col_prev * aspect_ratio[1]
## 1 column (2 column format):
width_1col = cm2inch(8.5)
height_1col = width_1col * aspect_ratio[1]
## 1.5 columns (2 column format):
width_1pt5col = cm2inch(11.4)
height_1pt5col = width_1pt5col * aspect_ratio[1]
## 2 columns (2 column format):
width_2col = cm2inch(17.4)
height_2col = width_2col * aspect_ratio[1]

dpi_graph = 600
dpi_LineArt = 1000

cmap = matplotlib.cm.tab10


# exit()


#############################################################
###
### Rudi's stuff:
###
##
#



# # eq_4
# def phi(theta_t):
#      return ((1 + np.cos(theta_t)) * np.sin(theta_t)) / ( (1 - np.cos(theta_t)) * (2 + np.cos(theta_t)) )
   

# # eq_21
# def Ca(t, state):
#     # R, theta_t = state
#     theta_t = state
#     phi_t = phi(theta_t)
#     # V = (np.pi * R**3) / (3 * phi_t)
#     return (np.cos(theta_eq) - np.cos(theta_t)) / (sigma + 2 * phi_t * np.log(3 * V * phi_t / (np.pi * a**3))) #- capillary_number

# # variation of eq_3:
# def contact_radius(theta_t):
#     return np.cbrt((3 / np.pi) * phi(theta_t) * V)


# relation between contact angle and contact radius according to my paper (eq. S7):
def cont_radius(theta, R_s):
    """ R_s is the radius of the cell were it a perfect sphere with the same volume as the spherical
        cap. Think of it as the 'initial' radius."""
    r_c = R_s * ((4 * (1 - np.cos(theta)**2 )**(3/2)) / (4 - (1 - np.cos(theta))**2 * (2 + np.cos(theta))))**(1/3)
    return r_c

# # eq_20
# def dR_t(t, state):
#     R_t, theta_t = state
#     phi_t = phi(theta_t)
#     return (beta * (np.cos(theta_eq) - np.cos(theta_t))) / (zeta + 6 * eta * phi_t * np.log(R_t / a))






# Rudi's equation 4:
def theta_from_Bc_dec(t, t_p, B, B_p, k):
    theta = np.arccos(1 - ((1 - B_p / B) / t_p ** k) * t ** k) 
    return theta

t_p = 10
t = np.linspace(0, 10, 1001)
B = 1
B_p = 0.25
k = np.linspace(1, 3, 3)
tt, kk = np.meshgrid(t, k)
theta = theta_from_Bc_dec(tt, t_p, B, B_p, kk)
theta_p = theta_from_Bc_dec(t_p, t_p, B, B_p, k)

fig, ax = plt.subplots()
[ax.plot(t, np.rad2deg(th), alpha=0.5) for th in theta]
[ax.plot([t_p, 1.5*t_p], [np.rad2deg(th), np.rad2deg(th)]) for th in theta_p]
leg_lines = [mlines.Line2D([], [], color=[x.get_color() for x in ax.lines][0], marker='', linestyle='-', label='k = 1'), 
             mlines.Line2D([], [], color=[x.get_color() for x in ax.lines][1], marker='', linestyle='-', label='k = 2'), 
             mlines.Line2D([], [], color=[x.get_color() for x in ax.lines][2], marker='', linestyle='-', label='k = 3')]
plt.legend(handles=leg_lines)
ax.set_ylabel('Angle (Deg)', fontsize=16)
ax.set_xlabel('Time', fontsize=16)
ax.set_title('Equation 4', fontsize=20)

fig.savefig(Path.joinpath(out_dir, 'Equation 4.pdf'), 
            dpi=dpi_LineArt, bbox_inches='tight')


# Rudi's equation 10:
def theta_from_Bc_dec_w_deform(t, v_max, r_cp, B, B_p, k):
    with np.errstate(invalid='warn'):
        theta = np.arccos( (B_p / B) + (k / B) * (1 - v_max * t / r_cp) )
        
    return theta




# # Some constants:
# R_sphere = 1 # radius of droplet if not spread
# # R_0 = 0 # initial spherical cap base radius
# theta_0 = np.deg2rad(179)
# beta = 1
# beta_star = 0.25
# theta_eq = np.pi - np.arccos(beta_star / beta) # contact angle at equilibrium
# zeta = 0 # friction coefficient
# eta = 20e-3 # viscosity in Pa*s
# sigma = zeta / eta
# a = 0.1 # lower cutoff value of rho (radial distance from axisymmetric axis)
# V = 4/3 * np.pi * R_sphere**3 # assume constant volume equal to the volume of a sphere
# # capillary_number = eta * contact_line_speed / beta



t = np.linspace(0, 100, 10001)
B = 1
B_p = 0.25
k = np.linspace(1, 3, 3)

R_s = 1 #30
theta_eq = np.arccos(B_p / B)
r_cp = cont_radius(theta_eq, R_s)
v_max = 1/15 #2
vs = np.linspace(0.05, 0.25, 5)

tt_k, kk = np.meshgrid(t, k)
theta_ks = theta_from_Bc_dec_w_deform(tt_k, v_max, r_cp, B, B_p, kk)

tt_v, vv = np.meshgrid(t, vs)
theta_vs = theta_from_Bc_dec_w_deform(tt_v, vv, r_cp, B, B_p, 1)

theta_p = np.arccos(B_p / B)

fig, ax = plt.subplots()
[ax.plot(t, np.rad2deg(th), alpha=0.5) for th in theta_ks]
ax.axhline(np.rad2deg(theta_p), c='k')
leg_lines = [mlines.Line2D([], [], color=[x.get_color() for x in ax.lines][0], marker='', linestyle='-', label='k = 1'), 
             mlines.Line2D([], [], color=[x.get_color() for x in ax.lines][1], marker='', linestyle='-', label='k = 2'), 
             mlines.Line2D([], [], color=[x.get_color() for x in ax.lines][2], marker='', linestyle='-', label='k = 3')]
plt.legend(handles=leg_lines)
ax.set_ylabel('Angle (Deg)', fontsize=16)
ax.set_xlabel('Time', fontsize=16)
ax.set_title('Equation 10', fontsize=20)

fig.savefig(Path.joinpath(out_dir, 'Equation 10 var_k.pdf'), 
            dpi=dpi_LineArt, bbox_inches='tight')




fig, ax = plt.subplots()
[ax.plot(t, np.rad2deg(th), alpha=0.5) for th in theta_vs]
ax.axhline(np.rad2deg(theta_p), c='k')
leg_lines = [mlines.Line2D([], [], color=[x.get_color() for x in ax.lines][0], marker='', linestyle='-', label='v_max = %0.2f'%vs[0]), 
             mlines.Line2D([], [], color=[x.get_color() for x in ax.lines][1], marker='', linestyle='-', label='v_max = %0.2f'%vs[1]), 
             mlines.Line2D([], [], color=[x.get_color() for x in ax.lines][2], marker='', linestyle='-', label='v_max = %0.2f'%vs[2]),
             mlines.Line2D([], [], color=[x.get_color() for x in ax.lines][1], marker='', linestyle='-', label='v_max = %0.2f'%vs[3]), 
             mlines.Line2D([], [], color=[x.get_color() for x in ax.lines][2], marker='', linestyle='-', label='v_max = %0.2f'%vs[4])]
plt.legend(handles=leg_lines)
ax.set_ylabel('Angle (Deg)', fontsize=16)
ax.set_xlabel('Time', fontsize=16)
ax.set_title('Equation 10', fontsize=20)

fig.savefig(Path.joinpath(out_dir, 'Equation 10 var_vmax.pdf'), 
            dpi=dpi_LineArt, bbox_inches='tight')













# exit()

#############################################################
###
### Serge's stuff: Solving ODEs of contact angle change
###
##
#


### Set some parameters :
t_init, t_end = 0, 14400 # endpoint is 4hrs of simulated time

dt = 1e-1 # small amount to move time forward by if theta = theta_eq
max_step = np.inf #5e-2 #1e-1
# t_end is repeated a lot, but I kind of like how that makes it clear what it is being used for

theta_min = np.deg2rad(0.1) #0.1 degrees is the minimum angle, 
## assume that this angle is the floor and cells will not seperate, so according to the JKR theory of
## small adhesions between spheres, the contact will never get smaller than this. 






### Create a function describing the equilibrium model for contact size:
def angle2reltension(angle_in_rad):
    """ Takes a numpy array. Outputs "B* / B" (i.e. the relative cortical tension
    in the near-equilibrium case)."""
    # angle_in_rad = np.deg2rad(angle_in_deg)
    betastar_div_beta = np.cos(angle_in_rad)
    
    return betastar_div_beta




# What is the Reynold's Number for an ectoderm cell?
# Re = rho * N * D**2 / mu
# rho = density (kg/m^3): for ectoderm cells is ~ 1.186g/cm^3 according to my napkin math
# N = rotational speed (rad/s): varies over time
# D = diameter of stirring device (m): need a different way to think about it for spherical objects?
# mu = dynamic viscosity (Pa*s): according to Valentine et al 2005 is ~10-30mPa*s for Xenopus cytoplasmic egg extract

def Re(rho, N, D, mu):
    return rho * N * D**2 / mu

rho = 1186 #1.186g/cm**3 = kg/m**3 
N = np.deg2rad(np.linspace(0, 200, 21))# According to my graphs from earlier on angular acceleration this can be 0-10 deg/s for the very damped curves or as high as 250 for the low eta curves
D = 12.5e-6 #25um = 25e-6m ; maybe I should use the radius of the cell isntead?
mu = 20e-3 #pick a halfway point 20mPa*s = 20e-3Pa*s

reynolds = Re(rho, N, D, mu)

# Re ranges from 0 - ~3.23e-5 (even at very high angular velocities)
# This seems to me to be pretty low. Then we can assume that inertial forces are
# negligible, therefore Bnet ~= 0

### Therefore we are assuming negligible inertial forces.






### The following are some functions used to describe how the density at the 
### contact relative to the non-contact surface of the actin cortex changes 
### over time. 
## An exponential decay function (looks similar to the LifeAct data at the contact):
def z_decay(t, z_init, t_half, z_plat):
    tau = t_half / np.log(2)
    z = np.asarray((z_init - z_plat) * np.exp(-1 * t / tau) + z_plat)
    z[t < 0] = z_init
    dz_dt = np.asarray(1/tau * z)
    return z, dz_dt




## A random distribution function for points with a similar description as our
## LifeAct data in dissociation buffer (1xDB):
data_acquisition_rate = 1/30 # 1 data point acquired every 30 seconds
acquisition_duration = t_end*10 - t_init # t_end - t_init was set to be 3600 seconds

def z_rand(t, mean, std, data_acquis_rate, acquis_dur, z_init=None, random_state=None):
    ## Be aware that z_init will skew the random distribution slightly, but if
    ## there are many points the change should be inconsequential. 
    
    num_datapts = int(data_acquisition_rate * acquisition_duration) + 1 # the +1 is so that t_end is included
    
    ## The empirical data of ecto-ecto cells in ca2+-free medium looked roughly
    ## normally distributed (or maybe log-normal distribution...), so let us create
    ## a normal distribution of data points with the same mean, standard devation, 
    ## and sampling rate as our empirical data. 
    ## A truncated normal distribution is used here to stop negative z values
    ## from appearing, as they cause trouble for solve_ivp. 
    ## The z values correspond to lifeact fluorescence intensity values, so
    ## negative values do not make sense.
    low, high = 0, 10 # fluorescence floor of the distribution = 0, ceiling = 10 times the free surface fluor.
    norm_distrib_data = truncnorm.rvs((low - mean)/std, (high - mean)/std, loc=mean, scale=std, size=num_datapts, random_state=random_state)
    if z_init:
        norm_distrib_data[0] = z_init # sets the first random point in the series to be equal to the initial z (i.e. initial cortex thickness)
    else:
        pass
    ## Assign each data point a time based on its position:
    norm_distrib_time = np.arange(num_datapts) * 1/data_acquisition_rate
    
    ## Now interpolate between the different data points so that this normal 
    ## distribution can be used as a function in a similar way to 'z_cont'. 
    ## Cubic interpolation makes the function less "jagged", but doesn't seem
    ## to provide any noticeable performance bonus. Does allow us to get the
    ## first derivative w.r.t. time though:
    ## k=3 -> cubic interpolation, s=0 -> no smoothing (interpolation line goes through every point)
    z_rand = interpolate.UnivariateSpline(norm_distrib_time, norm_distrib_data, k=3, s=0, ext='const')
    ## Would probably actually want to make the derivative it's own function?
    ## No. Instead just have this function return 2 values (z and dz_dt), and 
    ## then you can unpack it later in the VESM.

    z = z_rand(t)
    dz_dt = z_rand.derivative(n=1)(t)

    return z, dz_dt



def z_const(t, val):
    """Returns the value and the derivative of a constant function."""
    return val, 0



def z_const_new(t, val):
    return val, 0




# This chunk below is just a test to make sure that the interpolation function
# worked correctly; it's for validation purposes:
# t_min, t_max = norm_distrib_time.min(), norm_distrib_time.max()
# t_step = int((t_max - t_min) * 10 + 1)
# t = np.linspace(t_min, t_max, t_step)
# y = rand_distrib_over_time(t)
# plt.scatter(norm_distrib_time, norm_distrib_data)



















### The angular velocity viscoelastic model has 2 events that you need to be 
### aware of: 
### 1) An equilibrium event where the contact angle at a particular time
###    is equal to the equilibrium contact angle given the relative cortex density
###    at that particular time.
### 2) A zero angle event, where the contact angle at a particular time is 
###    predicted to be 0 or negative. 
### Both of these events are not defined in the model, and so are handled with
### extra code to define them explicitly. 
##
### **The angular acceleration viscoelastic model does not have these same
### limitations. 



class SpherCapGeom:#(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, EA, eta_A, EN, eta_N, eta_m, Gamma, E_r, z_cont_func, z_cont_args, z_cm_func, z_cm_args):
    
    """ Returns some useful geometric relations for a spherical cap with a 
        constant volume but variable contact angle. """
    
    def __init__(self, theta, R_s):
        
        self.R_c = (4**(1/3) * R_s) / (4 - (1 - np.cos(theta))**2 * (2 + np.cos(theta)))**(1/3)
        self.dRc_dtheta = (-4**(1/3) * R_s * 3 * np.sin(theta) * (np.cos(theta)**2 - 1)) / (3 * (4 - (1 - np.cos(theta))**2 * (2 + np.cos(theta)))**(4/3))
    
        self.Ss = 4 * np.pi * R_s**2
        
        self.Scm = 2 * np.pi * self.R_c**2 * (1 + np.cos(theta))
        self.dScm_dtheta = 2 * np.pi * (2 * self.R_c * (1 + np.cos(theta)) * self.dRc_dtheta - (self.R_c**2 * np.sin(theta)))
    
        self.Scc = np.pi * (self.R_c * np.sin(theta))**2
        self.dScc_dtheta = 2 * np.pi * self.R_c * np.sin(theta) * (np.sin(theta) * self.dRc_dtheta + self.R_c * np.cos(theta))
    
        self.SN = self.Scc + self.Scm
        self.dSN_dtheta = self.dScc_dtheta + self.dScm_dtheta
        
        self.SA = self.Scc + self.Scm
        self.dSA_dtheta = self.dScc_dtheta + self.dScm_dtheta
        
        self.r_c = R_s * ( (4 * (1 - (np.cos(theta))**2 )**(3/2)) / (4 - (1 - np.cos(theta))**2 * (2 + np.cos(theta))) )**(1/3)



def theta_dot_rolling_condition(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args):
    
    # theta_stretch, Scc0, Scm0, SNcc0, SNcm0, theta = syst
    # should theta in the rest of the function actually be theta_roll?
    
    # for the peeling test:
    # theta_stretch, Scc0, Scm0, SNcc0, SNcm0, theta, theta_peel = syst
    # theta, Scc0, Scm0, SNcc0, SNcm0, theta_roll, theta_peel = syst
    theta, Scc0, Scm0, SNcc0, SNcm0 = syst


    SCap = SpherCapGeom(theta, R_s)
    
    r_c, dScc_dtheta, Scm, dScm_dtheta = SCap.r_c, SCap.dScc_dtheta, SCap.Scm, SCap.dScm_dtheta


    # z_cc, dzcc_dt = z_cont_func(t, *z_cont_args)
    z_cm, dzcm_dt = z_cm_func(t, *z_cm_args)
    # z_ccG = z_cm
    # z_N = z_cm
    z_Ncm = z_cm
    # z_Ncc = z_cm
    
    
    ## This is the contribution to contact expansion from the adhesion energy:
    adh_nrg_contrib = Gamma / (2*z_cm)
    
    ## Viscous culture medium contribution that resists changes in contact size. 
    ## The assumption here is that we focus only on what is occurring locally
    ## around the contact zone. An actual analysis of fluid flow and drag on a 
    ## cell as it spreads on another cell seems unecessary since the contribution
    ## from this is small relative to the cortical tensions. 
    env_drag_contrib = (etam/(2*np.pi*r_c*z_cm)) * dScc_dtheta
    
    ## nu is the Poisson Ratio ; a value of 0.5 means we are assuming 
    ## incompressibility:
    nu_A = 0.5 # cortical actin Poisson ratio
    nu_B = 0.5 # cytoplasm Poisson ratio
    nu_N = 0.5 # cortical non-actin Poisson ratio
    ## Current literature suggests that the cell interior is roughly a very 
    ## weak elastic solid body, therefore it should resist deformation as if it
    ## were a very weak elastic sphere. We are using Hertzian contact mechanics
    ## to estimate how the elastic force resisting deformation relates to the
    ## contact radius, but we know that this is not an ideal approximation
    ## since Hertzian mechanics applies to small deformations and the cell can
    ## deform quite substantially. It is the best we can come up with right now
    ## and also the weak nature of the elasticity of the cytoplasm means that 
    ## the contributed force will be small anyways. 
    bulk_elast = (4 * E_B * r_c**2) / (6 * (1 - nu_B**2) * R_s * z_cm)

    
    
    
    ## This is the rolling equation with the viscosity incorporated correctly:
    actin_elast_contrib = E_A/(1 - nu_A) * (Scm / Scm0 - 1)
    actin_visc_remodel_contrib = eta_A * np.log(2) / t_Apt5 * (1 - Scm / Scm0) * (Scm / Scm0)
    nonact_elast_contrib = E_N/(1 - nu_N) * (Scm / SNcm0 - 1)
    nonact_visc_remodel_contrib = eta_N * np.log(2) / t_Npt5 * (1 - Scm / SNcm0) * (Scm / SNcm0)
    denom = (1 - np.cos(theta)) * ((eta_A / Scm0) * dScm_dtheta + eta_A * dScc_dtheta * (Scm / Scm0**2) + (z_Ncm / z_cm) * (eta_N / SNcm0) * dScm_dtheta + (z_Ncm / z_cm) * eta_N * dScc_dtheta * (Scm / SNcm0**2)) + env_drag_contrib

    theta_dot = [ (np.cos(theta) - 1) * (sigma_a + actin_elast_contrib + actin_visc_remodel_contrib) / denom +
                 ((np.cos(theta) - 1) * (z_Ncm / z_cm) * (nonact_elast_contrib + nonact_visc_remodel_contrib) - bulk_elast + adh_nrg_contrib) / denom][-1]

    return theta_dot

def theta_dot_peeling_condition(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args):
# def get_theta_dot_rolling(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args):
    # theta_stretch, Scc0, Scm0, SNcc0, SNcm0, theta_roll, theta = syst
    # theta, Scc0, Scm0, SNcc0, SNcm0, theta_roll, theta_peel = syst
    theta, Scc0, Scm0, SNcc0, SNcm0 = syst
    # should theta in the rest of the function actually be theta_roll?

    SCap = SpherCapGeom(theta, R_s)
    # SCap = SpherCapGeom(theta_stretch, R_s)

    r_c, Scc, dScc_dtheta = SCap.r_c, SCap.Scc, SCap.dScc_dtheta


    z_cc, dzcc_dt = z_cont_func(t, *z_cont_args)
    z_cm, dzcm_dt = z_cm_func(t, *z_cm_args)
    # z_ccG = z_cm
    # z_N = z_cm
    # z_Ncm = z_cm
    z_Ncc = z_cm
    
    

    
    ## This is the contribution to contact expansion from the adhesion energy:
    adh_nrg_contrib = Gamma / (2*z_cc)
    
    ## Viscous culture medium contribution that resists changes in contact size. 
    ## The assumption here is that we focus only on what is occurring locally
    ## around the contact zone. An actual analysis of fluid flow and drag on a 
    ## cell as it spreads on another cell seems unecessary since the contribution
    ## from this is small relative to the cortical tensions. 
    env_drag_contrib = (etam/(2*np.pi*r_c*z_cc)) * dScc_dtheta
    
    ## nu is the Poisson Ratio ; a value of 0.5 means we are assuming 
    ## incompressibility:
    nu_A = 0.5 # cortical actin Poisson ratio
    nu_B = 0.5 # cytoplasm Poisson ratio
    nu_N = 0.5 # cortical non-actin Poisson ratio
    ## Current literature suggests that the cell interior is roughly a very 
    ## weak elastic solid body, therefore it should resist deformation as if it
    ## were a very weak elastic sphere. We are using Hertzian contact mechanics
    ## to estimate how the elastic force resisting deformation relates to the
    ## contact radius, but we know that this is not an ideal approximation
    ## since Hertzian mechanics applies to small deformations and the cell can
    ## deform quite substantially. It is the best we can come up with right now
    ## and also the weak nature of the elasticity of the cytoplasm means that 
    ## the contributed force will be small anyways. 
    bulk_elast = (4 * E_B * r_c**2) / (6 * (1 - nu_B**2) * R_s * z_cc)

    
    
    ## This is the peeling equation with the viscosity incorporated correctly:
    actin_elast_contrib = E_A/(1 - nu_A) * (Scc / Scc0 - 1)
    actin_visc_remodel_contrib = eta_A * np.log(2) / t_Apt5 * (1 - Scc / Scc0) * (Scc / Scc0)
    nonact_elast_contrib = E_N/(1 - nu_N) * (Scc / SNcc0 - 1)
    nonact_visc_remodel_contrib = eta_N * np.log(2) / t_Npt5 * (1 - Scc / SNcc0) * (Scc / SNcc0)
    denom = (1 - np.cos(theta)) * ((eta_A / Scc0) * dScc_dtheta - eta_A * dScc_dtheta * (Scc / Scc0**2) + (z_Ncc / z_cc) * (eta_N / SNcc0) * dScc_dtheta - (z_Ncc / z_cc) * eta_N * dScc_dtheta * (Scc / SNcc0**2)) + env_drag_contrib

    theta_dot = [ (np.cos(theta) - 1) * (sigma_a + actin_elast_contrib + actin_visc_remodel_contrib) / denom +
                 ((np.cos(theta) - 1) * (z_Ncc / z_cc) * (nonact_elast_contrib + nonact_visc_remodel_contrib) - bulk_elast + adh_nrg_contrib) / denom][-1]
        
    return theta_dot



def get_theta_dot_rolling(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args):

    theta, Scc0, Scm0, SNcc0, SNcm0 = syst

        
    SCap = SpherCapGeom(theta, R_s)

    r_c, Scc, dScc_dtheta, Scm, dScm_dtheta = SCap.r_c, SCap.Scc, SCap.dScc_dtheta, SCap.Scm, SCap.dScm_dtheta


    # z_cc = z_cont_func(t, *z_cont_args)
    z_cc, dzcc_dt = z_cont_func(t, *z_cont_args)
    z_cm, dzcm_dt = z_cm_func(t, *z_cm_args)
    # z_N = z_cm
    z_Ncm = z_cm
    z_Ncc = z_cm
    
    
    theta_dot_roll = theta_dot_rolling_condition(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args)


    dScm0_dt = -dScc_dtheta * theta_dot_roll - np.log(2)/t_Apt5 * (Scm0 - Scm)
    dScc0_dt = dScc_dtheta * theta_dot_roll - np.log(2)/t_Apt5 * (Scc0 - Scc)
    dSNcm0_dt = -dScc_dtheta * theta_dot_roll - np.log(2)/t_Npt5 * (SNcm0 - Scm)
    dSNcc0_dt = dScc_dtheta * theta_dot_roll - np.log(2)/t_Npt5 * (SNcc0 - Scc)

    
    # E_r = sigma_a * z_cm -> bulk_elast = E_r/(sigma_a*z_cm)*((Scm + Scc)/Ss - 1)
    ## This is the contribution to contact expansion from the adhesion energy:
    adh_nrg_contrib = Gamma / (2*z_cm)
    
    ## Viscous culture medium contribution that resists changes in contact size. 
    ## The assumption here is that we focus only on what is occurring locally
    ## around the contact zone. An actual analysis of fluid flow and drag on a 
    ## cell as it spreads on another cell seems unecessary since the contribution
    ## from this is small relative to the cortical tensions. 
    env_drag_contrib = (etam/(2*np.pi*r_c*z_cm)) * dScc_dtheta
    
    ## nu is the Poisson Ratio ; a value of 0.5 means we are assuming 
    ## incompressibility:
    nu_A = 0.5 # cortical actin Poisson ratio
    nu_B = 0.5 # cytoplasm Poisson ratio
    nu_N = 0.5 # cortical non-actin Poisson ratio
    ## Current literature suggests that the cell interior is roughly a very 
    ## weak elastic solid body, therefore it should resist deformation as if it
    ## were a very weak elastic sphere. We are using Hertzian contact mechanics
    ## to estimate how the elastic force resisting deformation relates to the
    ## contact radius, but we know that this is not an ideal approximation
    ## since Hertzian mechanics applies to small deformations and the cell can
    ## deform quite substantially. It is the best we can come up with right now
    ## and also the weak nature of the elasticity of the cytoplasm means that 
    ## the contributed force will be small anyways. 
    bulk_elast = (4 * E_B * r_c**2) / (6 * (1 - nu_B**2) * R_s * z_cm)
    
    
    
    theta_dot = [( (np.cos(theta) * (sigma_a + (E_A/(1 - nu_A))*(Scm / Scm0 - 1) - eta_A*dScm0_dt*(Scm/Scm0**2))) - 
                    ((z_cc / z_cm) * (sigma_a + (E_A/(1 - nu_A))*(Scc / Scc0 - 1) - eta_A*dScc0_dt*(Scc/Scc0**2))) ) / 
                  ( env_drag_contrib - 
                    (eta_A / Scm0 * dScm_dtheta * np.cos(theta)) + 
                    (eta_A / Scc0 * (z_cc / z_cm) * dScc_dtheta) - 
                    ((z_Ncm/z_cm) * eta_N / SNcm0 * dScm_dtheta * np.cos(theta)) +
                    ((z_Ncc/z_cm) * eta_N / SNcc0 * dScc_dtheta) ) + 
                  ( (np.cos(theta) * (z_Ncm / z_cm) * ((E_N/(1 - nu_N))*(Scm / SNcm0 - 1) - eta_N * (Scm/SNcm0**2) * dSNcm0_dt )) - 
                    ((z_Ncc / z_cm) * ((E_N/(1 - nu_N)) * (Scc/SNcc0 - 1) - eta_N * (Scc/SNcc0**2) * dSNcc0_dt)) ) / 
                  ( env_drag_contrib - 
                    (eta_A / Scm0 * dScm_dtheta * np.cos(theta)) + 
                    (eta_A / Scc0 * (z_cc / z_cm) * dScc_dtheta) - 
                    ((z_Ncm/z_cm) * eta_N / SNcm0 * dScm_dtheta * np.cos(theta)) +
                    ((z_Ncc/z_cm) * eta_N / SNcc0 * dScc_dtheta) ) + 
                  ( adh_nrg_contrib - bulk_elast ) / 
                  ( env_drag_contrib - 
                    (eta_A / Scm0 * dScm_dtheta * np.cos(theta)) + 
                    (eta_A / Scc0 * (z_cc / z_cm) * dScc_dtheta) - 
                    ((z_Ncm/z_cm) * eta_N / SNcm0 * dScm_dtheta * np.cos(theta)) +
                    ((z_Ncc/z_cm) * eta_N / SNcc0 * dScc_dtheta) )
                  ][-1]    
     
    return theta_dot

def syst_eq_rolling(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args):
    
    theta, Scc0, Scm0, SNcc0, SNcm0 = syst


    theta_dot_roll = theta_dot_rolling_condition(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args)


    SCap = SpherCapGeom(theta, R_s)
    Scc, Scm, dScc_dtheta = SCap.Scc, SCap.Scm, SCap.dScc_dtheta
    
    
    dScm0_dt = -dScc_dtheta * theta_dot_roll - np.log(2)/t_Apt5 * (Scm0 - Scm)
    dScc0_dt = dScc_dtheta * theta_dot_roll - np.log(2)/t_Apt5 * (Scc0 - Scc)
    dSNcm0_dt = -dScc_dtheta * theta_dot_roll - np.log(2)/t_Npt5 * (SNcm0 - Scm)
    dSNcc0_dt = dScc_dtheta * theta_dot_roll - np.log(2)/t_Npt5 * (SNcc0 - Scc)


    theta_dot = get_theta_dot_rolling(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args)


    syst = [theta_dot, dScc0_dt, dScm0_dt, dSNcc0_dt, dSNcm0_dt]
    
    return syst




def get_theta_dot_stretching(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args):

    theta, Scc0, Scm0, SNcc0, SNcm0 = syst

    
    
    SCap = SpherCapGeom(theta, R_s)

    r_c, Scc, dScc_dtheta, Scm, dScm_dtheta = SCap.r_c, SCap.Scc, SCap.dScc_dtheta, SCap.Scm, SCap.dScm_dtheta


    # z_cc = z_cont_func(t, *z_cont_args)
    z_cc, dzcc_dt = z_cont_func(t, *z_cont_args)
    z_cm, dzcm_dt = z_cm_func(t, *z_cm_args)
    # z_N = z_cm
    z_Ncm = z_cm
    z_Ncc = z_cm
    
    

    dScm0_dt = -np.log(2)/t_Apt5 * (Scm0 - Scm)
    dScc0_dt = -np.log(2)/t_Apt5 * (Scc0 - Scc)
    dSNcm0_dt = -np.log(2)/t_Npt5 * (SNcm0 - Scm)
    dSNcc0_dt = -np.log(2)/t_Npt5 * (SNcc0 - Scc)

    
    
    # E_r = sigma_a * z_cm -> bulk_elast = E_r/(sigma_a*z_cm)*((Scm + Scc)/Ss - 1)
    ## This is the contribution to contact expansion from the adhesion energy:
    adh_nrg_contrib = Gamma / (2*z_cm)
    
    ## Viscous culture medium contribution that resists changes in contact size. 
    ## The assumption here is that we focus only on what is occurring locally
    ## around the contact zone. An actual analysis of fluid flow and drag on a 
    ## cell as it spreads on another cell seems unecessary since the contribution
    ## from this is small relative to the cortical tensions. 
    env_drag_contrib = (etam/(2*np.pi*r_c*z_cm)) * dScc_dtheta
    
    ## nu is the Poisson Ratio ; a value of 0.5 means we are assuming 
    ## incompressibility:
    nu_A = 0.5 # cortical actin Poisson ratio
    nu_B = 0.5 # cytoplasm Poisson ratio
    nu_N = 0.5 # cortical non-actin Poisson ratio
    ## Current literature suggests that the cell interior is roughly a very 
    ## weak elastic solid body, therefore it should resist deformation as if it
    ## were a very weak elastic sphere. We are using Hertzian contact mechanics
    ## to estimate how the elastic force resisting deformation relates to the
    ## contact radius, but we know that this is not an ideal approximation
    ## since Hertzian mechanics applies to small deformations and the cell can
    ## deform quite substantially. It is the best we can come up with right now
    ## and also the weak nature of the elasticity of the cytoplasm means that 
    ## the contributed force will be small anyways. 
    bulk_elast = (4 * E_B * r_c**2) / (6 * (1 - nu_B**2) * R_s * z_cm)
    
    
    
    theta_dot = [( (np.cos(theta) * (sigma_a + (E_A/(1 - nu_A))*(Scm / Scm0 - 1) - eta_A*dScm0_dt*(Scm/Scm0**2))) - 
                    ((z_cc / z_cm) * (sigma_a + (E_A/(1 - nu_A))*(Scc / Scc0 - 1) - eta_A*dScc0_dt*(Scc/Scc0**2))) ) / 
                  ( env_drag_contrib - 
                    (eta_A / Scm0 * dScm_dtheta * np.cos(theta)) + 
                    (eta_A / Scc0 * (z_cc / z_cm) * dScc_dtheta) - 
                    ((z_Ncm/z_cm) * eta_N / SNcm0 * dScm_dtheta * np.cos(theta)) +
                    ((z_Ncc/z_cm) * eta_N / SNcc0 * dScc_dtheta) ) + 
                  ( (np.cos(theta) * (z_Ncm / z_cm) * ((E_N/(1 - nu_N))*(Scm / SNcm0 - 1) - eta_N * (Scm/SNcm0**2) * dSNcm0_dt )) - 
                    ((z_Ncc / z_cm) * ((E_N/(1 - nu_N)) * (Scc/SNcc0 - 1) - eta_N * (Scc/SNcc0**2) * dSNcc0_dt)) ) / 
                  ( env_drag_contrib - 
                    (eta_A / Scm0 * dScm_dtheta * np.cos(theta)) + 
                    (eta_A / Scc0 * (z_cc / z_cm) * dScc_dtheta) - 
                    ((z_Ncm/z_cm) * eta_N / SNcm0 * dScm_dtheta * np.cos(theta)) +
                    ((z_Ncc/z_cm) * eta_N / SNcc0 * dScc_dtheta) ) + 
                  ( adh_nrg_contrib - bulk_elast ) / 
                  ( env_drag_contrib - 
                    (eta_A / Scm0 * dScm_dtheta * np.cos(theta)) + 
                    (eta_A / Scc0 * (z_cc / z_cm) * dScc_dtheta) - 
                    ((z_Ncm/z_cm) * eta_N / SNcm0 * dScm_dtheta * np.cos(theta)) +
                    ((z_Ncc/z_cm) * eta_N / SNcc0 * dScc_dtheta) )
                  ][-1]    
     
    return theta_dot

def syst_eq_stretching(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args):
    
    theta, Scc0, Scm0, SNcc0, SNcm0 = syst


    SCap = SpherCapGeom(theta, R_s)
    Scc, Scm = SCap.Scc, SCap.Scm
    
    
    dScm0_dt = -np.log(2)/t_Apt5 * (Scm0 - Scm)
    dScc0_dt = -np.log(2)/t_Apt5 * (Scc0 - Scc)
    dSNcm0_dt = -np.log(2)/t_Npt5 * (SNcm0 - Scm)
    dSNcc0_dt = -np.log(2)/t_Npt5 * (SNcc0 - Scc)


    theta_dot = get_theta_dot_stretching(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args)


    syst = [theta_dot, dScc0_dt, dScm0_dt, dSNcc0_dt, dSNcm0_dt]
    
    return syst




def get_theta_dot_peeling(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args):

    theta, Scc0, Scm0, SNcc0, SNcm0 = syst

    
    
    SCap = SpherCapGeom(theta, R_s)

    r_c, Scc, dScc_dtheta, Scm, dScm_dtheta = SCap.r_c, SCap.Scc, SCap.dScc_dtheta, SCap.Scm, SCap.dScm_dtheta


    # z_cc = z_cont_func(t, *z_cont_args)
    z_cc, dzcc_dt = z_cont_func(t, *z_cont_args)
    z_cm, dzcm_dt = z_cm_func(t, *z_cm_args)
    # z_N = z_cm
    z_Ncm = z_cm
    z_Ncc = z_cm
    
    
    theta_dot_peel = theta_dot_peeling_condition(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args)



    dScm0_dt = -dScc_dtheta * theta_dot_peel - np.log(2)/t_Apt5 * (Scm0 - Scm)
    dScc0_dt = dScc_dtheta * theta_dot_peel - np.log(2)/t_Apt5 * (Scc0 - Scc)
    dSNcm0_dt = -dScc_dtheta * theta_dot_peel - np.log(2)/t_Npt5 * (SNcm0 - Scm)
    dSNcc0_dt = dScc_dtheta * theta_dot_peel - np.log(2)/t_Npt5 * (SNcc0 - Scc)

    
    
    # E_r = sigma_a * z_cm -> bulk_elast = E_r/(sigma_a*z_cm)*((Scm + Scc)/Ss - 1)
    ## This is the contribution to contact expansion from the adhesion energy:
    adh_nrg_contrib = Gamma / (2*z_cm)
    
    ## Viscous culture medium contribution that resists changes in contact size. 
    ## The assumption here is that we focus only on what is occurring locally
    ## around the contact zone. An actual analysis of fluid flow and drag on a 
    ## cell as it spreads on another cell seems unecessary since the contribution
    ## from this is small relative to the cortical tensions. 
    env_drag_contrib = (etam/(2*np.pi*r_c*z_cm)) * dScc_dtheta
    
    ## nu is the Poisson Ratio ; a value of 0.5 means we are assuming 
    ## incompressibility:
    nu_A = 0.5 # cortical actin Poisson ratio
    nu_B = 0.5 # cytoplasm Poisson ratio
    nu_N = 0.5 # cortical non-actin Poisson ratio
    ## Current literature suggests that the cell interior is roughly a very 
    ## weak elastic solid body, therefore it should resist deformation as if it
    ## were a very weak elastic sphere. We are using Hertzian contact mechanics
    ## to estimate how the elastic force resisting deformation relates to the
    ## contact radius, but we know that this is not an ideal approximation
    ## since Hertzian mechanics applies to small deformations and the cell can
    ## deform quite substantially. It is the best we can come up with right now
    ## and also the weak nature of the elasticity of the cytoplasm means that 
    ## the contributed force will be small anyways. 
    bulk_elast = (4 * E_B * r_c**2) / (6 * (1 - nu_B**2) * R_s * z_cm)
    
    
    
    theta_dot = [( (np.cos(theta) * (sigma_a + (E_A/(1 - nu_A))*(Scm / Scm0 - 1) - eta_A*dScm0_dt*(Scm/Scm0**2))) - 
                    ((z_cc / z_cm) * (sigma_a + (E_A/(1 - nu_A))*(Scc / Scc0 - 1) - eta_A*dScc0_dt*(Scc/Scc0**2))) ) / 
                  ( env_drag_contrib - 
                    (eta_A / Scm0 * dScm_dtheta * np.cos(theta)) + 
                    (eta_A / Scc0 * (z_cc / z_cm) * dScc_dtheta) - 
                    ((z_Ncm/z_cm) * eta_N / SNcm0 * dScm_dtheta * np.cos(theta)) +
                    ((z_Ncc/z_cm) * eta_N / SNcc0 * dScc_dtheta) ) + 
                  ( (np.cos(theta) * (z_Ncm / z_cm) * ((E_N/(1 - nu_N))*(Scm / SNcm0 - 1) - eta_N * (Scm/SNcm0**2) * dSNcm0_dt )) - 
                    ((z_Ncc / z_cm) * ((E_N/(1 - nu_N)) * (Scc/SNcc0 - 1) - eta_N * (Scc/SNcc0**2) * dSNcc0_dt)) ) / 
                  ( env_drag_contrib - 
                    (eta_A / Scm0 * dScm_dtheta * np.cos(theta)) + 
                    (eta_A / Scc0 * (z_cc / z_cm) * dScc_dtheta) - 
                    ((z_Ncm/z_cm) * eta_N / SNcm0 * dScm_dtheta * np.cos(theta)) +
                    ((z_Ncc/z_cm) * eta_N / SNcc0 * dScc_dtheta) ) + 
                  ( adh_nrg_contrib - bulk_elast ) / 
                  ( env_drag_contrib - 
                    (eta_A / Scm0 * dScm_dtheta * np.cos(theta)) + 
                    (eta_A / Scc0 * (z_cc / z_cm) * dScc_dtheta) - 
                    ((z_Ncm/z_cm) * eta_N / SNcm0 * dScm_dtheta * np.cos(theta)) +
                    ((z_Ncc/z_cm) * eta_N / SNcc0 * dScc_dtheta) )
                  ][-1]    
     
    return theta_dot

def syst_eq_peeling(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args):
    
    theta, Scc0, Scm0, SNcc0, SNcm0 = syst

    
    theta_dot_peel = theta_dot_peeling_condition(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args)

    SCap = SpherCapGeom(theta, R_s)
    Scc, Scm, dScc_dtheta = SCap.Scc, SCap.Scm, SCap.dScc_dtheta
    
    
    dScm0_dt = -dScc_dtheta * theta_dot_peel - np.log(2)/t_Apt5 * (Scm0 - Scm)
    dScc0_dt = dScc_dtheta * theta_dot_peel - np.log(2)/t_Apt5 * (Scc0 - Scc)
    dSNcm0_dt = -dScc_dtheta * theta_dot_peel - np.log(2)/t_Npt5 * (SNcm0 - Scm)
    dSNcc0_dt = dScc_dtheta * theta_dot_peel - np.log(2)/t_Npt5 * (SNcc0 - Scc)


    theta_dot = get_theta_dot_peeling(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args)


    syst = [theta_dot, dScc0_dt, dScm0_dt, dSNcc0_dt, dSNcm0_dt]
    
    return syst



def syst_eq_quicksolve(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args):
    """This is just to quickly get a solve_ivp object that can be modified to 
    have its last timepoint be np.inf. This is done to make some later parts 
    cleaner/shorter. """
    theta, Scc0, Scm0, SNcc0, SNcm0 = syst
    
    SCap = SpherCapGeom(theta, R_s)
    Scc, Scm = SCap.Scc, SCap.Scm

    dScc0_dt = -np.log(2)/t_Apt5 * (Scc0 - Scc)
    dScm0_dt = -np.log(2)/t_Apt5 * (Scm0 - Scm)
    dSNcc0_dt = -np.log(2)/t_Npt5 * (SNcc0 - Scc)
    dSNcm0_dt = -np.log(2)/t_Npt5 * (SNcm0 - Scm)
    
    theta_dot = 0
    
    syst = [theta_dot, dScc0_dt, dScm0_dt, dSNcc0_dt, dSNcm0_dt]

    return syst



def syst_eq_theta_min(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args):
    theta, Scc0, Scm0, SNcc0, SNcm0 = syst
     
    SCap = SpherCapGeom(theta, R_s)
    Scc, Scm = SCap.Scc, SCap.Scm

    dScc0_dt = -np.log(2)/t_Apt5 * (Scc0 - Scc)
    dScm0_dt = -np.log(2)/t_Apt5 * (Scm0 - Scm)
    dSNcc0_dt = -np.log(2)/t_Npt5 * (SNcc0 - Scc)
    dSNcm0_dt = -np.log(2)/t_Npt5 * (SNcm0 - Scm)

    theta_dot = 0

    syst = [theta_dot, dScc0_dt, dScm0_dt, dSNcc0_dt, dSNcm0_dt]

    return syst




class Events:
    
    def rolling_start(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args):
        theta_dot = theta_dot_rolling_condition(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args)
        return theta_dot
    rolling_start.direction = 1 # trigger when theta_dot increases above 0
    rolling_start.terminal = True
    
    def rolling_stops(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args):
        theta_dot = theta_dot_rolling_condition(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args)
        return theta_dot
    rolling_stops.direction = -1 # trigger when theta_dot decreases below 0
    rolling_stops.terminal = True



    def peeling_start(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args):
        theta_dot = theta_dot_peeling_condition(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args)
        return theta_dot
    peeling_start.direction = -1 # trigger when theta_dot decreases below 0
    peeling_start.terminal = True

    def peeling_stops(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args):
        theta_dot = theta_dot_peeling_condition(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args)
        return theta_dot
    peeling_stops.direction = 1 # trigger when theta_dot increases above 0
    peeling_stops.terminal = True



    def theta_min_start(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args):
        theta, Scc0, Scm0, SNcc0, SNcm0 = syst
        return theta - theta_min
    theta_min_start.direction = -1 # only trigger if the angle is going from positive to negative.
    theta_min_start.terminal = True



    def stretching_start(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args):
        theta_dot = get_theta_dot_stretching(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args)
        return theta_dot
    stretching_start.direction = 1 # only trigger if the angle starts growing.
    stretching_start.terminal = True




def check_adh_conditions(t, syst, args, calcium, Gamma):
    ## This function helps make the VESM less verbose when determining the case_id.
    
    ## Has the minimum angle been exceeded?
    theta, Scc0, Scm0, SNcc0, SNcm0 = syst
    
    if theta < theta_min:
        theta_min_exceeded = False
    else:
        theta_min_exceeded = True

    ### Is rolling adhesion possible?:
    rolling_vel = theta_dot_rolling_condition(t, syst, *args)
    
    if (rolling_vel > 0) and Gamma:
        rolling_pos = True
    else:
        rolling_pos = False
    
    ## Is peeling deadhesion possible?
    peeling_vel = theta_dot_peeling_condition(t, syst, *args)
    
    if (peeling_vel < 0) and (calcium == False) and Gamma:
        peeling_neg = True
    else:
        peeling_neg = False

    return theta_min_exceeded, rolling_pos, peeling_neg



def VESM(t_span, syst, args, solve_ivp_method, solve_ivp_max_step, solve_ivp_atol, solve_ivp_rtol, strain_lim_event_stol, calcium=True):

    ## These lines below are only used for convenience while testing/debugging:
    ## in the presence of calcium:
    # calcium = True
    # Gamma, strain_lim_actin, t_Apt5, EA, eta_A, strain_lim_nonactin, t_Npt5, EN, eta_N, E_r = *params,
    # consts = (R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, EA, eta_A, EN, eta_N, eta_m_real, Gamma, E_r, z_cc_func, z_cc_func_args, z_cm_func, z_cm_func_args)

    # args = consts
    # # syst_init = theta_init, Scc0_init, Scm0_init, SA0_init, SNcc0_init, SNcm0_init, SN0_init
    
    # # theta_roll_init = theta_init
    # # theta_peel_init = theta_init
    # # syst_init = theta_init, Scc0_init, Scm0_init, SNcc0_init, SNcm0_init, theta_roll_init, theta_peel_init
    # syst_init = theta_init, Scc0_init, Scm0_init, SNcc0_init, SNcm0_init
    
    # syst = syst_init
    # t_span = t_init, t_end
    
    
    ## For debugging in the no-calcium case:
    # calcium = False
    # Gamma, strain_lim_actin, t_Apt5, EA, eta_A, strain_lim_nonactin, t_Npt5, EN, eta_N, E_r = *params,
    
    # ## Specifically for testing fig8G - random fluctuations in z_cm:
    # z_cm_func, z_cm_func_args = z_cc_func, z_cc_func_args
    # ##
    
    # consts = (R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, EA, eta_A, EN, eta_N, eta_m_real, Gamma, E_r, z_cc_func, z_cc_func_args, z_cm_func, z_cm_func_args)

    # args = consts
    # syst_init = theta_init, Scc0_init, Scm0_init, SNcc0_init, SNcm0_init
    
    # syst = syst_init
    # t_span = t_init, t_end_rand
    
    
    ### end of debugging lines.


    
    ### First check which scenario we are starting with:
    t, t_end = t_span
    theta, Scc0, Scm0, SNcc0, SNcm0 = syst
    
    
    R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args = args


    case_id = None # used to identify why a particular solve_ivp solution has stopped.
    case_info = None
    

    ## Determine the initial adhesion conditions:
    ## Due to (very small) imprecision when solving the systems of equations,
    ## the solver may stop from an event, but the adhesion conditions may not 
    ## change. For example, the rolling_stop event may occur, but the solver 
    ## stops ever so slightly before the condition. As a result, when 
    ## rolling_pos is checked in the next iteration, it will still return True,
    ## despite the event being called and therefore it should have been False.
    ## To get around this I convert these conditions to "flags" where if an 
    ## event occurs, this "flag" will be changed from e.g. True to False. 
    ## The inequalities are only used to set the flags during the initial
    ## conditions.
    
    theta_min_exceeded, rolling_pos, peeling_neg = check_adh_conditions(t, syst, args, calcium, Gamma)
        


    ## The VESM will solve different equations based on whether rolling 
    ## adhesion is occurring or not, and these different solutions have certain
    ## conditions which will halt the solver because a different equation needs
    ## to be solved from that point onwards. 
    ## For example, rolling adhesion may be occurring initially, but eventually
    ## stops, at which point the solver switches over to stretching adhesion. 
        ## The solution to this example is then a system of equations followed
        ## by a different system of equations, which are combined together.
    ## The empty lists below will be populated with solutions from consecutive  
    ## time periods. The many solutions will then be concatenated into a single
    ## solution.
    ts = list()
    solns = list()
    messages = list()
    soln_statuses = list()
    successes = list()
    t_events = list()
    
    theta_min_exceeded_status_list = list()
    rolling_pos_status_list = list()
    peeling_neg_status_list = list()
    adh_mode_list = list()

    color_list = list()
    solver_iteration_list = list()
    
    if np.isfinite(solve_ivp_max_step):
        count_max = (t_end - t) / solve_ivp_max_step
    else:
        count_max = np.inf
    
    
    
    count = 0
    while count < count_max:
        ## This will keep track of the clock time.         
        clock = tm.localtime()
        
        ## Determine the adhesion mode:
        if theta_min_exceeded == False:
            adh_mode = None
        elif rolling_pos and (peeling_neg == False):
            adh_mode = 'rolling'
        elif peeling_neg:
            adh_mode = 'peeling'
        else:
            adh_mode = 'stretching'
            
            
        ## If there is no 'Gamma' (i.e. adhesion energy), then make sure that 
        ## the adhesion mode is never 'rolling':
        if not Gamma:
            assert(adh_mode != 'rolling')
        
        
        print('case_id=%s: %s'%(case_id, case_info)) ## this case_id is from the previous count, thus the next print statement starts with a newline to group the print statements more intuitively
        print('\n count:%d , t=%.16f, adhesion mode=%s, theta_min_exceeded=%s, rolling_pos=%s, peeling_neg=%s'%(count,t,adh_mode,theta_min_exceeded,rolling_pos,peeling_neg))
            
        
        if theta_min_exceeded == False: 
            ## it doesn't matter what anything else is if
            ## theta_min_exceeded is False, because theta will
            ## then be set to equal theta_min...
            color = 'tab:red'
            
            ## If theta_min is not exceeded, then let the solution be equal to
            ## theta_min until either rolling adhesion becomes positive
            ## (denoted by the event 'theta_min_event_end_rolling'):
            if Gamma:
                soln_tstart = tm.perf_counter()
                print('Solving for rolling adhesion... (clock: %d:%d:%d)'%(clock.tm_hour, clock.tm_min, clock.tm_sec))
                soln_rolling_starts = solve_ivp(syst_eq_theta_min, (t, t_end), syst, args=args, method=solve_ivp_method, events=Events.rolling_start, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
                soln_tstop = tm.perf_counter()
                soln_time = soln_tstop - soln_tstart
                print('... solved after %f seconds.'%soln_time)
            else:
                soln_rolling_starts = solve_ivp(syst_eq_quicksolve, (t, t_end), syst, args=args, method=solve_ivp_method, events=None, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
                soln_rolling_starts.t[-1] = np.inf

            ## ... or until stretching adhesion becomes positive (denoted by 
            ## the event 'theta_min_event_end_stretching'):
            soln_tstart = tm.perf_counter()
            print('Solving for stretching adhesion... (clock: %d:%d:%d)'%(clock.tm_hour, clock.tm_min, clock.tm_sec))
            soln_stretching_starts = solve_ivp(syst_eq_theta_min, (t, t_end), syst, args=args, method=solve_ivp_method, events=Events.stretching_start, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
            soln_tstop = tm.perf_counter()
            soln_time = soln_tstop - soln_tstart
            print('... solved after %f seconds.'%soln_time)
            
            
            ## Inadmissible solutions:
            # soln_rolling_starts = solve_ivp(syst_eq_quicksolve, (t, t_end), syst, args=args, method=solve_ivp_method, events=None, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
            # soln_rolling_starts.t[-1] = np.inf

            # soln_stretching_starts = solve_ivp(syst_eq_quicksolve, (t, t_end), syst, args=args, method=solve_ivp_method, events=None, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
            # soln_stretching_starts.t[-1] = np.inf
                        
            soln_rolling_stops = solve_ivp(syst_eq_quicksolve, (t, t_end), syst, args=args, method=solve_ivp_method, events=None, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
            soln_rolling_stops.t[-1] = np.inf

            soln_rolling_switch2peeling = solve_ivp(syst_eq_quicksolve, (t, t_end), syst, args=args, method=solve_ivp_method, events=None, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
            soln_rolling_switch2peeling.t[-1] = np.inf
                
            soln_peeling_theta_min = solve_ivp(syst_eq_quicksolve, (t, t_end), syst, args=args, method=solve_ivp_method, events=None, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
            soln_peeling_theta_min.t[-1] = np.inf

            soln_peeling_stops = solve_ivp(syst_eq_quicksolve, (t, t_end), syst, args=args, method=solve_ivp_method, events=None, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
            soln_peeling_stops.t[-1] = np.inf
            
            soln_stretching_theta_min = solve_ivp(syst_eq_quicksolve, (t, t_end), syst, args=args, method=solve_ivp_method, events=None, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
            soln_stretching_theta_min.t[-1] = np.inf

            soln_stretching_switch2rolling = solve_ivp(syst_eq_quicksolve, (t, t_end), syst, args=args, method=solve_ivp_method, events=None, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
            soln_stretching_switch2rolling.t[-1] = np.inf
            
            soln_stretching_switch2peeling = solve_ivp(syst_eq_quicksolve, (t, t_end), syst, args=args, method=solve_ivp_method, events=None, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
            soln_stretching_switch2peeling.t[-1] = np.inf

            
            ### This chunk was test code when trying to get peeling to work:
            # ## the solver can get stuck from events, so if solve_ivp 
            # ## returns a first and last time point that are equal, then
            # ## the answer is inadmissible, so set the time for that 
            # ## solution to be inf so it's not selected as the case_id
            # if soln_rolling_starts.t[0] == soln_rolling_starts.t[-1]:
            #     soln_rolling_starts.t[-1] = np.inf
            # if soln_stretching_starts.t[0] == soln_stretching_starts.t[-1]:
            #     soln_stretching_starts.t[-1] = np.inf
            

        elif adh_mode == 'rolling':
            ## Now that theta_min has been exceeded, adhesion can proceed via 
            ## either the rolling or stretching mode of contact growth:
            color = 'tab:green'
            
            
            ## Rolling adhesion can proceed until syst_eq_rolling 
            ## is no longer positive:
            soln_tstart = tm.perf_counter()
            print('Solving for rolling adhesion and checking if rolling stops... (clock: %d:%d:%d)'%(clock.tm_hour, clock.tm_min, clock.tm_sec))
            soln_rolling_stops = solve_ivp(syst_eq_rolling, (t, t_end), syst, args=args, method=solve_ivp_method, events=Events.rolling_stops, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
            soln_tstop = tm.perf_counter()
            soln_time = soln_tstop - soln_tstart
            print('... solved after %f seconds.'%soln_time)

            
            if (calcium == False) and Gamma:
                ## or until peeling starts:
                soln_tstart = tm.perf_counter()
                print('Solving for rolling adhesion and checking if peeling starts... (clock: %d:%d:%d)'%(clock.tm_hour, clock.tm_min, clock.tm_sec))
                if soln_rolling_stops.t[-1] < (t_end - 1e-3):
                    soln_rolling_switch2peeling = solve_ivp(syst_eq_rolling, (t, soln_rolling_stops.t[-1]+1e-3), syst, args=args, method=solve_ivp_method, events=Events.peeling_start, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
                else:
                    soln_rolling_switch2peeling = solve_ivp(syst_eq_rolling, (t, t_end), syst, args=args, method=solve_ivp_method, events=Events.peeling_start, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
                soln_tstop = tm.perf_counter()
                soln_time = soln_tstop - soln_tstart
                print('... solved after %f seconds.'%soln_time)
            else:
                soln_rolling_switch2peeling = solve_ivp(syst_eq_quicksolve, (t, t_end), syst, args=args, method=solve_ivp_method, events=None, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
                soln_rolling_switch2peeling.t[-1] = np.inf
                

            ## Inadmissible solutions:
            soln_rolling_starts = solve_ivp(syst_eq_quicksolve, (t, t_end), syst, args=args, method=solve_ivp_method, events=None, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
            soln_rolling_starts.t[-1] = np.inf

            soln_stretching_starts = solve_ivp(syst_eq_quicksolve, (t, t_end), syst, args=args, method=solve_ivp_method, events=None, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
            soln_stretching_starts.t[-1] = np.inf
                        
            # soln_rolling_stops = solve_ivp(syst_eq_quicksolve, (t, t_end), syst, args=args, method=solve_ivp_method, events=None, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
            # soln_rolling_stops.t[-1] = np.inf

            # soln_rolling_switch2peeling = solve_ivp(syst_eq_quicksolve, (t, t_end), syst, args=args, method=solve_ivp_method, events=None, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
            # soln_rolling_switch2peeling.t[-1] = np.inf
         
            soln_peeling_theta_min = solve_ivp(syst_eq_quicksolve, (t, t_end), syst, args=args, method=solve_ivp_method, events=None, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
            soln_peeling_theta_min.t[-1] = np.inf

            soln_peeling_stops = solve_ivp(syst_eq_quicksolve, (t, t_end), syst, args=args, method=solve_ivp_method, events=None, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
            soln_peeling_stops.t[-1] = np.inf
            
            soln_stretching_theta_min = solve_ivp(syst_eq_quicksolve, (t, t_end), syst, args=args, method=solve_ivp_method, events=None, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
            soln_stretching_theta_min.t[-1] = np.inf

            soln_stretching_switch2rolling = solve_ivp(syst_eq_quicksolve, (t, t_end), syst, args=args, method=solve_ivp_method, events=None, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
            soln_stretching_switch2rolling.t[-1] = np.inf
            
            soln_stretching_switch2peeling = solve_ivp(syst_eq_quicksolve, (t, t_end), syst, args=args, method=solve_ivp_method, events=None, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
            soln_stretching_switch2peeling.t[-1] = np.inf
            
            
            ### This chunk was test code when trying to get peeling to work:
            # ## the solver can get stuck from events, so if solve_ivp 
            # ## returns a first and last time point that are equal, then
            # ## the answer is inadmissible, so set the time for that 
            # ## solution to be inf so it's not selected as the case_id
            # if soln_rolling_stops.t[0] == soln_rolling_stops.t[-1]:
            #     soln_rolling_stops.t[-1] = np.inf
            # if soln_rolling_switch2peeling.t[0] == soln_rolling_switch2peeling.t[-1]:
            #     soln_rolling_switch2peeling.t[-1] = np.inf

            
    
        elif adh_mode == 'peeling':
            color = 'tab:orange'
            
            soln_tstart = tm.perf_counter()
            print('Solving for peeling de-adhesion and checking if peeling stops... (clock: %d:%d:%d)'%(clock.tm_hour, clock.tm_min, clock.tm_sec))
            soln_peeling_stops = solve_ivp(syst_eq_peeling, (t, t_end), syst, args=args, method=solve_ivp_method, events=Events.peeling_stops, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
            soln_tstop = tm.perf_counter()
            soln_time = soln_tstop - soln_tstart
            print('... solved after %f seconds.'%soln_time)


            ## Solving for small angles (when theta_min would be reached) often
            ## takes a very, very long time, and peeling often stops first.
            ## So to speed things up we check if the theta_min event happens 
            ## before peeling, and if it doesn't, then we can safely skip 
            ## trying to solve until theta_min is hit:
            soln_tstart = tm.perf_counter()
            print('Solving for peeling de-adhesion and checking if theta_min is hit... (clock: %d:%d:%d)'%(clock.tm_hour, clock.tm_min, clock.tm_sec))
            if soln_peeling_stops.t[-1] < (t_end - 1e-3):
                soln_peeling_theta_min = solve_ivp(syst_eq_peeling, (t, soln_peeling_stops.t[-1]+1e-3), syst, args=args, method=solve_ivp_method, events=Events.theta_min_start, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
            else:
                soln_peeling_theta_min = solve_ivp(syst_eq_peeling, (t, t_end), syst, args=args, method=solve_ivp_method, events=Events.theta_min_start, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
            soln_tstop = tm.perf_counter()
            soln_time = soln_tstop - soln_tstart
            print('... solved after %f seconds.'%soln_time)



            ## Inadmissible solutions:
            soln_rolling_starts = solve_ivp(syst_eq_quicksolve, (t, t_end), syst, args=args, method=solve_ivp_method, events=None, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
            soln_rolling_starts.t[-1] = np.inf

            soln_stretching_starts = solve_ivp(syst_eq_quicksolve, (t, t_end), syst, args=args, method=solve_ivp_method, events=None, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
            soln_stretching_starts.t[-1] = np.inf
                        
            soln_rolling_stops = solve_ivp(syst_eq_quicksolve, (t, t_end), syst, args=args, method=solve_ivp_method, events=None, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
            soln_rolling_stops.t[-1] = np.inf

            soln_rolling_switch2peeling = solve_ivp(syst_eq_quicksolve, (t, t_end), syst, args=args, method=solve_ivp_method, events=None, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
            soln_rolling_switch2peeling.t[-1] = np.inf
         
            # soln_peeling_theta_min = solve_ivp(syst_eq_quicksolve, (t, t_end), syst, args=args, method=solve_ivp_method, events=None, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
            # soln_peeling_theta_min.t[-1] = np.inf

            # soln_peeling_stops = solve_ivp(syst_eq_quicksolve, (t, t_end), syst, args=args, method=solve_ivp_method, events=None, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
            # soln_peeling_stops.t[-1] = np.inf
            
            soln_stretching_theta_min = solve_ivp(syst_eq_quicksolve, (t, t_end), syst, args=args, method=solve_ivp_method, events=None, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
            soln_stretching_theta_min.t[-1] = np.inf

            soln_stretching_switch2rolling = solve_ivp(syst_eq_quicksolve, (t, t_end), syst, args=args, method=solve_ivp_method, events=None, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
            soln_stretching_switch2rolling.t[-1] = np.inf
            
            soln_stretching_switch2peeling = solve_ivp(syst_eq_quicksolve, (t, t_end), syst, args=args, method=solve_ivp_method, events=None, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
            soln_stretching_switch2peeling.t[-1] = np.inf

            ### This chunk was test code when trying to get peeling to work:
            # ## the solver can get stuck from events, so if solve_ivp 
            # ## returns a first and last time point that are equal, then
            # ## the answer is inadmissible, so set the time for that 
            # ## solution to be inf so it's not selected as the case_id
            # if soln_peeling_stops.t[0] == soln_peeling_stops.t[-1]:
            #     soln_peeling_stops.t[-1] = np.inf
            # if soln_peeling_theta_min.t[0] == soln_peeling_theta_min.t[-1]:
            #     soln_peeling_theta_min.t[-1] = np.inf


        elif adh_mode == 'stretching':
            color = 'k'
            
            if (calcium == False) and Gamma:
                ## or until peeling starts:
                soln_tstart = tm.perf_counter()
                print('Solving for stretching adhesion and checking if peeling starts... (clock: %d:%d:%d)'%(clock.tm_hour, clock.tm_min, clock.tm_sec))
                # soln_stretching_switch2peeling = solve_ivp(syst_eq_stretching, (t, t_end), syst, args=args, method=solve_ivp_method, events=PeelingEvents.peeling_start, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
                soln_stretching_switch2peeling = solve_ivp(syst_eq_stretching, (t, t_end), syst, args=args, method=solve_ivp_method, events=Events.peeling_start, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
                ## similar to switch2rolling, switch2peeling is 
                ## inadmissible if theta_dot_peeling is positive:
                # peeling_neg = get_theta_dot_peeling(soln_stretching_switch2peeling.t[-1], soln_stretching_switch2peeling.y.T[-1], *args) < 0
                # if not peeling_neg:
                #     soln_stretching_switch2peeling.t[-1] = np.inf
                soln_tstop = tm.perf_counter()
                soln_time = soln_tstop - soln_tstart
                print('... solved after %f seconds.'%soln_time)
            else:
                soln_stretching_switch2peeling = solve_ivp(syst_eq_quicksolve, (t, t_end), syst, args=args, method=solve_ivp_method, events=None, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
                soln_stretching_switch2peeling.t[-1] = np.inf

            if Gamma:
                if np.isfinite(soln_stretching_switch2peeling.t[-1]) and (soln_stretching_switch2peeling.t[-1] < (t_end - 1e-3)):
                    soln_tstart = tm.perf_counter()
                    print('Solving for stretching adhesion and checking if rolling starts... (clock: %d:%d:%d)'%(clock.tm_hour, clock.tm_min, clock.tm_sec))
                    soln_stretching_switch2rolling = solve_ivp(syst_eq_stretching, (t, soln_stretching_switch2peeling.t[-1]+1e-3), syst, args=args, method=solve_ivp_method, events=Events.rolling_start, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
                    # ## Also, switch2rolling is inadmissible if theta_dot_rolling
                    # ## is negative:
                    # rolling_pos = get_theta_dot_rolling(soln_stretching_switch2rolling.t[-1], soln_stretching_switch2rolling.y.T[-1], *args) > 0
                    # if not rolling_pos:
                        # soln_stretching_switch2rolling.t[-1] = np.inf
                    soln_tstop = tm.perf_counter()
                    soln_time = soln_tstop - soln_tstart
                    print('... solved after %f seconds.'%soln_time)
                else:
                    soln_tstart = tm.perf_counter()
                    print('Solving for stretching adhesion and checking if rolling starts... (clock: %d:%d:%d)'%(clock.tm_hour, clock.tm_min, clock.tm_sec))
                    soln_stretching_switch2rolling = solve_ivp(syst_eq_stretching, (t, t_end), syst, args=args, method=solve_ivp_method, events=Events.rolling_start, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
                    # ## Also, switch2rolling is inadmissible if theta_dot_rolling
                    # ## is negative:
                    # rolling_pos = get_theta_dot_rolling(soln_stretching_switch2rolling.t[-1], soln_stretching_switch2rolling.y.T[-1], *args) > 0
                    # if not rolling_pos:
                        # soln_stretching_switch2rolling.t[-1] = np.inf
                    soln_tstop = tm.perf_counter()
                    soln_time = soln_tstop - soln_tstart
                    print('... solved after %f seconds.'%soln_time)
            else:
                soln_stretching_switch2rolling = solve_ivp(syst_eq_quicksolve, (t, t_end), syst, args=args, method=solve_ivp_method, events=None, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
                soln_stretching_switch2rolling.t[-1] = np.inf
                    

            if np.isfinite(soln_stretching_switch2peeling.t[-1]) and (soln_stretching_switch2peeling.t[-1] < (t_end - 1e-3)):
                soln_tstart = tm.perf_counter()
                print('Solving for stretching adhesion and checking if theta_min is hit... (clock: %d:%d:%d)'%(clock.tm_hour, clock.tm_min, clock.tm_sec))
                soln_stretching_theta_min = solve_ivp(syst_eq_stretching, (t, soln_stretching_switch2peeling.t[-1]+1e-3), syst, args=args, method=solve_ivp_method, events=Events.theta_min_start, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
                soln_tstop = tm.perf_counter()
                soln_time = soln_tstop - soln_tstart
                print('... solved after %f seconds.'%soln_time)
            elif np.isfinite(soln_stretching_switch2rolling.t[-1]) and (soln_stretching_switch2rolling.t[-1] < (t_end - 1e-3)):
                soln_tstart = tm.perf_counter()
                print('Solving for stretching adhesion and checking if theta_min is hit... (clock: %d:%d:%d)'%(clock.tm_hour, clock.tm_min, clock.tm_sec))
                soln_stretching_theta_min = solve_ivp(syst_eq_stretching, (t, soln_stretching_switch2rolling.t[-1]+1e-3), syst, args=args, method=solve_ivp_method, events=Events.theta_min_start, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
                soln_tstop = tm.perf_counter()
                soln_time = soln_tstop - soln_tstart
                print('... solved after %f seconds.'%soln_time)
            else:
                soln_tstart = tm.perf_counter()
                print('Solving for stretching adhesion and checking if theta_min is hit... (clock: %d:%d:%d)'%(clock.tm_hour, clock.tm_min, clock.tm_sec))
                soln_stretching_theta_min = solve_ivp(syst_eq_stretching, (t, t_end), syst, args=args, method=solve_ivp_method, events=Events.theta_min_start, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
                soln_tstop = tm.perf_counter()
                soln_time = soln_tstop - soln_tstart
                print('... solved after %f seconds.'%soln_time)
            
    
        
            ## Inadmissible solutions:
            soln_rolling_starts = solve_ivp(syst_eq_quicksolve, (t, t_end), syst, args=args, method=solve_ivp_method, events=None, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
            soln_rolling_starts.t[-1] = np.inf

            soln_stretching_starts = solve_ivp(syst_eq_quicksolve, (t, t_end), syst, args=args, method=solve_ivp_method, events=None, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
            soln_stretching_starts.t[-1] = np.inf
                        
            soln_rolling_stops = solve_ivp(syst_eq_quicksolve, (t, t_end), syst, args=args, method=solve_ivp_method, events=None, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
            soln_rolling_stops.t[-1] = np.inf

            soln_rolling_switch2peeling = solve_ivp(syst_eq_quicksolve, (t, t_end), syst, args=args, method=solve_ivp_method, events=None, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
            soln_rolling_switch2peeling.t[-1] = np.inf
         
            soln_peeling_theta_min = solve_ivp(syst_eq_quicksolve, (t, t_end), syst, args=args, method=solve_ivp_method, events=None, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
            soln_peeling_theta_min.t[-1] = np.inf

            soln_peeling_stops = solve_ivp(syst_eq_quicksolve, (t, t_end), syst, args=args, method=solve_ivp_method, events=None, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
            soln_peeling_stops.t[-1] = np.inf
            
            # soln_stretching_theta_min = solve_ivp(syst_eq_quicksolve, (t, t_end), syst, args=args, method=solve_ivp_method, events=None, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
            # soln_stretching_theta_min.t[-1] = np.inf

            # soln_stretching_switch2rolling = solve_ivp(syst_eq_quicksolve, (t, t_end), syst, args=args, method=solve_ivp_method, events=None, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
            # soln_stretching_switch2rolling.t[-1] = np.inf
            
            # soln_stretching_switch2peeling = solve_ivp(syst_eq_quicksolve, (t, t_end), syst, args=args, method=solve_ivp_method, events=None, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
            # soln_stretching_switch2peeling.t[-1] = np.inf


            ### This chunk was test code when trying to get peeling to work:
            # ## the solver can get stuck from events, so if solve_ivp 
            # ## returns a first and last time point that are equal, then
            # ## the answer is inadmissible, so set the time for that 
            # ## solution to be inf so it's not selected as the case_id
            # if soln_stretching_switch2peeling.t[0] == soln_stretching_switch2peeling.t[-1]:
            #     soln_stretching_switch2peeling.t[-1] = np.inf
            # if soln_stretching_theta_min.t[0] == soln_stretching_theta_min.t[-1]:
            #     soln_stretching_theta_min.t[-1] = np.inf
            # if soln_stretching_theta_min.t[0] == soln_stretching_theta_min.t[-1]:
            #     soln_stretching_theta_min.t[-1] = np.inf
            # ## if the solver immediately switches back to rolling, then
            # ## the stretching mode of adhesion must be accumulating stress
            # ## too quickly when it starts, and therefore rolling adhesion
            # ## should continue, so here the rolling solution is valid
            # if soln_stretching_switch2rolling.t[0] == soln_stretching_switch2rolling.t[-1]:
            #     soln_stretching_switch2rolling.t[-1] = np.inf
            #     # pass
            # ## if the solver immediately switches back to rolling, then
            # ## the stretching mode of adhesion must be accumulating stress
            # ## too quickly when it starts, and therefore rolling adhesion
            # ## should continue, so here the rolling solution is valid
            # if soln_stretching_switch2rolling.t[0] == soln_stretching_switch2rolling.t[-1]:
            #     soln_stretching_switch2rolling.t[-1] = np.inf
            #     # pass


        case_id = np.argmin([soln_rolling_starts.t[-1], soln_stretching_starts.t[-1], # from theta_min_exceeded == False
                             soln_rolling_stops.t[-1], soln_rolling_switch2peeling.t[-1], # from adh_mode == 'rolling'
                             soln_peeling_theta_min.t[-1], soln_peeling_stops.t[-1], # from adh_mode == 'peeling'
                             soln_stretching_theta_min.t[-1], soln_stretching_switch2rolling.t[-1], soln_stretching_switch2peeling.t[-1]]) # from adh_mode == 'stretching'

        case_info_list = ['theta_min -> rolling starts', 'theta_min -> stretching starts', 
                          'rolling -> rolling stops', 'rolling -> peeling starts', 
                          'peeling -> theta_min starts', 'peeling -> peeling stops', 
                          'stretching -> theta_min starts', 'stretching -> rolling starts', 'stretching -> peeling starts']
        case_info = case_info_list[case_id]
        
        ## case_id:
        ## 0 = contact angle stays at theta_min until rolling adhesion starts
        ## 1 = contact angle stays at theta_min until stretching adhesion starts
        ## 2 = rolling adhesion occurs until adhesion energy is no longer enough to promote contact growth
        ## 3 = rolling adhesion occurs until peeling de-adhesion starts
        ## 4 = peeling de-adhesion occurs until theta_min is reached
        ## 5 = peeling de-adhesion occurs until the angle can be supported by adhesion energy
        ## 6 = stretching adbesion occurs until theta_min is hit
        ## 7 = stretching adhesion occurs until rolling adhesion starts
        ## 8 = stretching adhesion occurs until peeling starts



        if case_id == 0:
            ## set the solution:
            soln = soln_rolling_starts
            ## set the conditions from which the model will be resumed:
            t = soln.t[-1]
            syst = soln.y.T[-1]
            
            ## Check the adhesion conditions first, and then set event-based flags:
            theta_min_exceeded, rolling_pos, peeling_neg = check_adh_conditions(t, syst, args, calcium, Gamma)
            
            ## set event-based flags that determine how the model will proceed
            ## when it resumes in the next iteration:
            theta_min_exceeded = True
            rolling_pos = True
            peeling_neg = False
            
                        
        elif case_id == 1:
            ## set the solution:
            soln = soln_stretching_starts
            ## set the conditions from which the model will be resumed:
            t = soln.t[-1]
            syst = soln.y.T[-1]

            ## Check the adhesion conditions first, and then set event-based flags:
            theta_min_exceeded, rolling_pos, peeling_neg = check_adh_conditions(t, syst, args, calcium, Gamma)

            ## set event-based flags that determine how the model will proceed
            ## when it resumes in the next iteration:
            theta_min_exceeded = True
            rolling_pos = False


        elif case_id == 2:
            ## set the solution:
            soln = soln_rolling_stops
            ## set the conditions from which the model will be resumed:
            t = soln.t[-1]
            syst = soln.y.T[-1]

            ## Check the adhesion conditions first, and then set event-based flags:
            theta_min_exceeded, rolling_pos, peeling_neg = check_adh_conditions(t, syst, args, calcium, Gamma)

            ## set event-based flags that determine how the model will proceed
            ## when it resumes in the next iteration:
            theta_min_exceeded = True # this stays True.
            rolling_pos = False
                   
        
        elif case_id == 3:
            ## set the solution:
            soln = soln_rolling_switch2peeling
            ## set the conditions from which the model will be resumed:
            t = soln.t[-1]
            syst = soln.y.T[-1]
            
            ## Check the adhesion conditions first, and then set event-based flags:
            theta_min_exceeded, rolling_pos, peeling_neg = check_adh_conditions(t, syst, args, calcium, Gamma)

            ## set event-based flags that determine how the model will proceed
            ## when it resumes in the next iteration:
            peeling_neg = True
                        

        elif case_id == 4:
            ## set the solution:
            soln = soln_peeling_theta_min
            ## set the conditions from which the model will be resumed:
            t = soln.t[-1]
            syst = soln.y.T[-1]

            ## Check the adhesion conditions first, and then set event-based flags:
            theta_min_exceeded, rolling_pos, peeling_neg = check_adh_conditions(t, syst, args, calcium, Gamma)

            ## set event-based flags that determine how the model will proceed
            ## when it resumes in the next iteration:
            theta_min_exceeded = False
                    
        
        elif case_id == 5:
            ## set the solution:
            soln = soln_peeling_stops
            ## set the conditions from which the model will be resumed:
            t = soln.t[-1]
            syst = soln.y.T[-1]

            ## Check the adhesion conditions first, and then set event-based flags:
            theta_min_exceeded, rolling_pos, peeling_neg = check_adh_conditions(t, syst, args, calcium, Gamma)

            ## set event-based flags that determine how the model will proceed
            ## when it resumes in the next iteration:
            peeling_neg = False
                        

        elif case_id == 6:
            ## set the solution:
            soln = soln_stretching_theta_min
            ## set the conditions from which the model will be resumed:
            t = soln.t[-1]
            syst = soln.y.T[-1]

            ## Check the adhesion conditions first, and then set event-based flags:
            theta_min_exceeded, rolling_pos, peeling_neg = check_adh_conditions(t, syst, args, calcium, Gamma)

            ## set event-based flags that determine how the model will proceed
            ## when it resumes in the next iteration:
            theta_min_exceeded = False
            
            
        elif case_id == 7:
            ## set the solution:
            soln = soln_stretching_switch2rolling
            ## set the conditions from which the model will be resumed:
            t = soln.t[-1]
            syst = soln.y.T[-1]

            ## Check the adhesion conditions first, and then set event-based flags:
            theta_min_exceeded, rolling_pos, peeling_neg = check_adh_conditions(t, syst, args, calcium, Gamma)

            ## set event-based flags that determine how the model will proceed
            ## when it resumes in the next iteration:
            rolling_pos = True
            peeling_neg = False
            

        elif case_id == 8:
            ## set the solution:
            soln = soln_stretching_switch2peeling
            ## set the conditions from which the model will be resumed:
            t = soln.t[-1]
            syst = soln.y.T[-1]

            ## Check the adhesion conditions first, and then set event-based flags:
            theta_min_exceeded, rolling_pos, peeling_neg = check_adh_conditions(t, syst, args, calcium, Gamma)

            ## set event-based flags that determine how the model will proceed
            ## when it resumes in the next iteration:
            peeling_neg = True
            

        else:
            print('Unexpected behaviour while solving.')
            break                



        ## Update the list of solutions with the most current solution:    
        ts.append(soln.t)
        solns.append(soln.y)
        messages.append([soln.message] * len(soln.t))
        soln_statuses.append([soln.status] * len(soln.t))
        successes.append([soln.success] * len(soln.t))
        if soln.t_events is None:
            soln.t_events = []
        else:
            pass
        soln.t_events = [x for x in soln.t_events if np.any(x)]
        t_events.append([soln.t_events] * len(soln.t))
        adh_mode_list.append([adh_mode] * len(soln.t))
        theta_min_exceeded_status_list.append([theta_min_exceeded] * len(soln.t))
        rolling_pos_status_list.append([rolling_pos] * len(soln.t))
        peeling_neg_status_list.append([peeling_neg] * len(soln.t))
        adh_mode_list.append([adh_mode] * len(soln.t))
        color_list.append([color] * len(soln.t))
        solver_iteration_list.append([count] * len(soln.t))
        
        
        count += 1
    
        if soln.message == 'A termination event occurred.':
            continue
        elif soln.message == 'The solver successfully reached the end of the integration interval.':
            break
        else:            
            print('Unexpected behaviour in angular_veloc_ve_model.')
            break
    
    soln.t = np.concatenate(ts)
    soln.y = np.concatenate(solns, axis=1)
    t_events = [np.ravel(x) for x in t_events]
    soln.t_events = np.concatenate(t_events)
    soln.theta_min_exceeded_status = np.concatenate(theta_min_exceeded_status_list)
    soln.rolling_pos_status = np.concatenate(rolling_pos_status_list)
    soln.peeling_neg_status = np.concatenate(peeling_neg_status_list)
    soln.adh_mode = np.concatenate(adh_mode_list)
    soln.message = np.concatenate(messages)
    soln.success = np.concatenate(successes)
    soln.status = np.concatenate(soln_statuses)
    soln.color = np.concatenate(color_list)
    soln.count = np.concatenate(solver_iteration_list)
    
    return soln
    
    








###############################################################################
###############################################################################
###############################################################################
## 
##
## Load in some data:


## Load in the data from the LifeAct-GFP z-depth analysis:
la_contnd_df = pd.read_pickle(Path.joinpath(zdepth_lifeact_analysis_dir, 'zdepth_la_contnd_df.pkl'))


## Return the ee1xMBS data:
ee1xMBS_z_plat_df = la_contnd_df.query('Experimental_Condition == "ee1xMBS"')
## Return data where 'Time (minutes)' is >= 'lifeact_plat_start_time':
ee1xMBS_z_plat_df = ee1xMBS_z_plat_df.query('`Time (minutes)` >= lifeact_plat_start_time')

## Get the mean and std of the normalized contact LifeAct intensity:
z_plat_uncorr_mean = ee1xMBS_z_plat_df['lifeact_fluor_normed_mean'].mean()
z_plat_uncorr_std = ee1xMBS_z_plat_df['lifeact_fluor_normed_mean'].std()
z_plat_uncorr_mean = 0.2 # this actually looks more accurate than the mean...
## It seems to be because the lifeact_plat_start_time does not always represent
## the high plateau very well.

## Get the mean and std of the corrected normalized contact LifeAct intensity:
z_plat_corr_mean = ee1xMBS_z_plat_df['lifeact_fluor_normed_sub_mean_corr'].mean()
z_plat_corr_std = ee1xMBS_z_plat_df['lifeact_fluor_normed_sub_mean_corr'].std()
z_plat_corr_mean = 0.02 # this actually looks more accurate than the mean...


## Return the ee1xDB data:
ee1xDB_z_plat_df = la_contnd_df.query('Experimental_Condition == "ee1xDB"')
## Return data where 'Time (minutes)' is >= 'lifeact_plat_start_time':
ee1xDB_z_plat_df = ee1xDB_z_plat_df.query('`Time (minutes)` >= lifeact_plat_start_time')

## Get the mean and std of the normalized contact LifeAct intensity:
z_plat_low_uncorr_mean = ee1xDB_z_plat_df['lifeact_fluor_normed_mean'].mean()
z_plat_low_uncorr_std = ee1xDB_z_plat_df['lifeact_fluor_normed_mean'].std()

## Get the mean and std of the corrected normalized contact LifeAct intensity:
# z_plat_low_corr_mean = ee1xDB_z_plat_df['lifeact_fluor_normed_sub_mean_corr'].mean()
# z_plat_low_corr_std = ee1xDB_z_plat_df['lifeact_fluor_normed_sub_mean_corr'].std()


###############################################################################




## Set some technical parameters used by solve_ivp:
solve_ivp_atol = 1e-9#1e-6 # tolerance used in solve_ivp
solve_ivp_rtol = 1e-6#1e-3 # tolerance used in solve_ivp
strain_lim_event_stol = 0 # 1e-4 # allowable tolerance for strain_max_actin and strain_max_nonactin events
solve_ivp_max_step = np.inf #1 #0.1
solve_ivp_method = 'BDF' #'RK45' #'BDF'




## Set constants:
R_s = 15 #um; approximate radius of an ecto cell is ~15um=1.5e-5m
R_s_low = 12 #um
R_s_endo = 34 #um
z_cm = 1 # our Xenopus ectoderm cells have a cortex thickness of ~1um to ~1.5um
# beta = 1#8.6e-4 # according to David et al 2014, ecto cells have a cortical tension of ~0.86mJ/m**2=8.6e-4J/m**2 
# sigma_a = 1 #beta / z_cm
# pr = 0.5 # Value of the Poisson ratio -> 0.5 == incompressible material ; this is assigned in the functions get_theta_dot_stretching/_rolling /_peeling
eta_m_real = 1e-21 #1e-3 J*s/m^3 -> 1e-21 J*s/um^3

## 1 J/m^3 = 1e-18 J/um^3
# beta = 1#8.6e-4 # according to David et al 2014, ecto cells have a cortical 
# tension of ~0.86mJ/m**2 = 8.6e-4J/m**2 = 8.6e-16J/um^2
sigma_a = 8.6e-16 # J/um^3 #sigma_a = beta / z_cm # 8.6e-16J/um^2 / 1um = 8.6e-16J/um^3
# sigma_a = 2.3e-16 # J/um^3 #sigma_a = beta / z_cm # 2.3e-16J/um^2 / 1um = 2.3e-16J/um^3
sigma_a_low = 2.3e-16 # J/um^3 #sigma_a = beta / z_cm # 2.3e-16J/um^2 / 1um = 2.3e-16J/um^3
sigma_a_endo = 0.9e-16 # J/um^3 #sigma_a = beta / z_cm # 0.9e-16J/um^2 / 1um = 0.9e-16J/um^3
sigma_a_YY = 0.5 * sigma_a
Gamma_null = 0 #beta * 0.1 # the case for no adhesion energy "Gamma"
G2pct, G5pct, G10pct, G20pct, G30pct, G220pct = np.array([0.02, 0.05, 0.1, 0.2, 0.3, 2.2]) * (sigma_a * z_cm)
# c = 6 * np.pi * R_s # units of meters; due to lack of information we are picking R_s as the Stokes radius
turnover_rate_vfast = 1 #seconds
turnover_rate_fast = 12 #seconds
turnover_rate_med = 30 #seconds
turnover_rate_slow = 60 #seconds
turnover_rate_vslow = 120 #seconds
turnover_rate_vvslow = 300 #seconds
turnover_rate_vvvslow = 600 #seconds
turnover_rate_negligible = 3600 #seconds


## A range of elasticity values:
E0 = 0 * 1e-18 # J/um^3
E1e0 = 1e0 * 1e-18 # J/um^3
E1e1 = 1e1 * 1e-18 # J/um^3
E1e2 = 1e2 * 1e-18 # J/um^3
E5e2 = 5e2 * 1e-18 # J/um^3
E1e3 = 1e3 * 1e-18 # J/um^3
E1e4 = 1e4 * 1e-18 # J/um^3


## A range of viscosity values:
eta0 = 0 * 1e-18 # J/um^3
eta1e0 = 1e0 * 1e-18 # J*s/um^3
eta1e1 = 1e1 * 1e-18 # J*s/um^3
eta1e2 = 1e2 * 1e-18 # J*s/um^3
eta1e3 = 1e3 * 1e-18 # J*s/um^3
eta1e4 = 1e4 * 1e-18 # J*s/um^3
eta1e5 = 1e5 * 1e-18 # J*s/um^3


## A variety of strain limits. They were intended to be used in solve_ivp 
## events that would stop the integration so you would know that the 
## cytoskeletal network that you are looking at broke, however are not in use
## in this version of the script.
strain_lim_high = 1e3 # a percent represented in decimal format (i.e. 0.2 == 20%)
strain_lim_med = 0.2 # a percent represented in decimal format (i.e. 0.2 == 20%)
strain_lim_low = 0.1 # a percent represented in decimal format (i.e. 0.2 == 20%)
strain_lim_vlow = 0.01 # a percent represented in decimal format (i.e. 0.2 == 20%)
strain_lim_vvlow = 0.001 # a percent represented in decimal format (i.e. 0.2 == 20%)

## From Valentine et al. 2005: elasticity of Xenopus oocyte extract is about 
## 2-10 Pa. We're going to do a conversion since all the simulation stresses
## are in J/um**3
E_B_null = 0 # this ones easy. 
E_B_example = 6e-18 # (6 J/m**3) * (1m**3/1e18um**3) = 6e-18 J/um**3
E_b_low = 2e-18 # (2 J/m**3) * (1m**3/1e18um**3) = 2e-18 J/um**3
# E_b_med = 6e-18 # (6 J/m**3) * (1m**3/1e18um**3) = 6e-18 J/um**3
# E_b_high = 1e-17 # (10 J/m**3) * (1m**3/1e18um**3) = 1e-17 J/um**3






## Set initial values: 
## Contact angle (needs to be non-zero because if zero then get a 
## mathematical singularity which results in a divide by zero error):
theta_init = np.deg2rad(1) #theta cannot be zero or we will get a singularity
## Reference surfaces:
def get_surf_init(R_s):
    SCap_init = SpherCapGeom(theta_init, R_s)
    # Rc_init = SCap_init.R_c #um
    Scc0_init = SCap_init.Scc #um^2
    Scm0_init = SCap_init.Scm #um^2
    SNcc0_init = SCap_init.Scc #um^2
    SNcm0_init = SCap_init.Scm #um^2

    return Scc0_init, Scm0_init, SNcc0_init, SNcm0_init



## Set the initial thickness that the contact cortex starts at:
z_init = z_cm * np.cos(theta_init) # um
## Set a lower bound for the contact cortex (my data is roughly a quarter of 
## the non-contact actin cortex):
z_plat_corr = z_plat_corr_mean * z_cm # um ; based on the corrected data values from my measured data
z_plat_uncorr = z_plat_uncorr_mean * z_cm # um ; based on the data values from my measured data
## Set half-life of cortex thickness decay (my data shows some to be about half
## the brightness after around 2 minutes, though there is variability):
t_half = 120 # seconds

## set cortex thickness at the contact to follow an exponential decay, as is
## roughly seen in the data:
z_cc_func = z_decay
z_cc_func_uncorr_args = (z_init, t_half, z_plat_uncorr)
z_cc_func_corr_args = (z_init, t_half, z_plat_corr)

## set non-contact cortex thickness to be constant:
z_cm_func = z_const
z_cm_func_args = (z_cm,) 





## Some realistic model cases:
## 'H' stands for 'high value', 'L' stands for 'low value', 'M' for 'medium':

# elasticity = 90Pa, Poisson number = 0.5, sigma_a = 860J/m^3
E_actin_H = 90 * 1e-18
E_microtub_H = 15 * 1e-18
E_intermfil_H = 240 * 1e-18
E_multinetwork_L = 100 * 1e-18
E_multinetwork_M = 623 * 1e-18
E_multinetwork_H = 1400 * 1e-18
# E_multinetwork_vH = 42e3 * 1e-18 # yes, that is 42kPa... see Cartagena-Rivera et al 2016

## viscosity = 0Pa.s, 1j/m**3 == 1e-18 J/um**3
eta_actin = 0 * 1e-18
eta_microtub = 0 * 1e-18
eta_intermfil = 0 * 1e-18
eta_multinetwork = 30e-3  * 1e-18




halflife_actin = 8.6 # seconds
halflife_microtub = None # not sure, data for this seems to be missing
halflife_low = 60 # seconds
halflife_intermfil_L = 194 # seconds
halflife_intermfil_M = 490 # seconds
halflife_intermfil_H = 954 # seconds

strain_lim_actin = 0.2 # unitless
strain_lim_microtub = 0.5 # unitless
strain_lim_intermfil = 2.5 # unitless



## I considered changing these adhesion energies to be consistent with 
## sigma_a_low and  sigma_a_endo but since they are based on literature values 
## I think it is wiser to keep them constant: 
adh_energy_Ccad_H = 2.5e-2 * sigma_a # fraction of sigma_a * sigma_a = J/um**2
adh_energy_Ncam_H = 6.4e-2 * sigma_a # fraction of sigma_a * sigma_a = J/um**2
adh_energy_aggrec_hyal_H = 4.9e-2 * sigma_a # fraction of sigma_a * sigma_a = J/um**2
adh_energy_aggrec_coll_noCalcium_H = 1.7e-1 * sigma_a # fraction of sigma_a * sigma_a = J/um**2
adh_energy_aggrec_coll_yesCalcium_H = 2.9e-1 * sigma_a # fraction of sigma_a * sigma_a = J/um**2
adh_energy_total_noCalcium_H = (adh_energy_aggrec_hyal_H + adh_energy_aggrec_coll_noCalcium_H)
adh_energy_total_yesCalcium_H = (adh_energy_Ccad_H + adh_energy_aggrec_hyal_H + adh_energy_aggrec_coll_yesCalcium_H)

adh_energy_Ccad_L = 5e-4 * sigma_a # fraction of sigma_a * sigma_a = J/um**2
adh_energy_Ncam_L = 1.3e-3 * sigma_a # fraction of sigma_a * sigma_a = J/um**2
adh_energy_aggrec_hyal_L = 9.8e-4 * sigma_a # fraction of sigma_a * sigma_a = J/um**2
adh_energy_aggrec_coll_noCalcium_L = 3.5e-3 * sigma_a # fraction of sigma_a * sigma_a = J/um**2
adh_energy_aggrec_coll_yesCalcium_L = 5.8e-3 * sigma_a # fraction of sigma_a * sigma_a = J/um**2
adh_energy_total_noCalcium_L = (adh_energy_aggrec_hyal_L + adh_energy_aggrec_coll_noCalcium_L)
adh_energy_total_yesCalcium_L = (adh_energy_Ccad_L + adh_energy_aggrec_hyal_L + adh_energy_aggrec_coll_yesCalcium_L)

adh_energy_total_noCalcium_M = (adh_energy_total_noCalcium_L + adh_energy_total_noCalcium_H) / 2
adh_energy_total_yesCalcium_M = (adh_energy_total_yesCalcium_L + adh_energy_total_yesCalcium_H) / 2







### Params that I want to talk about in the figures:
## [[Gamma, strain_lim_actin, t_Apt5, EA, eta_A, strain_lim_nonactin, t_Npt5, EN, eta_N, E_r],
##  ...
##  ]

# Fig.S4A: Increasing contact viscosity:
figS4A_params = [[Gamma_null, strain_lim_high, turnover_rate_vfast, 0, eta0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_null], # no adhesion energy, no deformation limit, no viscoelasticity
                  [G20pct, strain_lim_high, turnover_rate_vfast, 0, eta1e2, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], # no adhesion energy, no deformation limit, medium actin viscosity
                  [G20pct, strain_lim_high, turnover_rate_vfast, 0, eta1e3, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], # no adhesion energy, no deformation limit, medium actin viscosity
                  [G20pct, strain_lim_high, turnover_rate_vfast, 0, eta1e4, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], # no adhesion energy, no deformation limit, high actin viscosity
                  [G20pct, strain_lim_high, turnover_rate_vfast, 0, eta1e5, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], # no adhesion energy, no deformation limit, high actin viscosity
                  [Gamma_null, strain_lim_high, turnover_rate_vfast, 0, eta1e5, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_null], # no adhesion energy, no deformation limit, higher actin viscosity
                  ]

figS4B_params = [[Gamma_null, strain_lim_high, turnover_rate_vfast, 0,eta0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_null], # no adhesion energy, no deformation limit, no viscoelasticity
                 [G20pct, strain_lim_high, turnover_rate_slow, 0, eta1e2, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], # no adhesion energy, no deformation limit, medium actin viscosity
                 [G20pct, strain_lim_high, turnover_rate_slow, 0, eta1e3, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], # no adhesion energy, no deformation limit, medium actin viscosity
                 [G20pct, strain_lim_high, turnover_rate_slow, 0, eta1e4, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], # no adhesion energy, no deformation limit, high actin viscosity
                 [G20pct, strain_lim_high, turnover_rate_slow, 0, eta1e5, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], # no adhesion energy, no deformation limit, high actin viscosity
                 [Gamma_null, strain_lim_high, turnover_rate_slow, 0, eta1e5, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_null], # no adhesion energy, no deformation limit, higher actin viscosity
                 ]

# Fig.S4B: rounding energy:
figS4C_params = [[Gamma_null, strain_lim_high, turnover_rate_fast, 0, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_null], 
                 [G20pct, strain_lim_high, turnover_rate_fast, 0, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_null], 
                 [Gamma_null, strain_lim_high, turnover_rate_fast, 0, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], # no adhesion energy, deformation limit similar to initial actin decrease rate
                 [G20pct, strain_lim_high, turnover_rate_fast, 0, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], # no adhesion energy, deformation limit similar to middling actin decrease rate
                 ]

# Fig.3A - fast turnover half-life, increasing elasticity
fig3A_params = [[Gamma_null, strain_lim_high, turnover_rate_vfast, E0, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_null], 
                [G20pct, strain_lim_high, turnover_rate_slow, E0, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], 
                [G20pct, strain_lim_high, turnover_rate_slow, E1e1, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], 
                [G20pct, strain_lim_high, turnover_rate_slow, E1e2, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], 
                [G20pct, strain_lim_high, turnover_rate_slow, E5e2, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], 
                [G20pct, strain_lim_high, turnover_rate_slow, E1e3, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], 
                [Gamma_null, strain_lim_high, turnover_rate_slow, E1e3, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_null], 
                ]

# Fig. 3B - decreasing turnover half-life, mid elasticity
fig3B_params = [[Gamma_null, strain_lim_high, turnover_rate_vfast, E0, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_null], 
                [G20pct, strain_lim_high, turnover_rate_vfast, E5e2, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], 
                [G20pct, strain_lim_high, turnover_rate_fast, E5e2, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], 
                [G20pct, strain_lim_high, turnover_rate_slow, E5e2, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], 
                [G20pct, strain_lim_high, turnover_rate_vvvslow, E5e2, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], 
                [Gamma_null, strain_lim_high, turnover_rate_vvslow, E5e2, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_null], 
                ]

# Fig. 3C - physiological solutions for contact surface stresses:
fig3C_params = [[Gamma_null, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_null], 
                [adh_energy_total_yesCalcium_M, strain_lim_actin, halflife_actin, E_actin_H, eta_actin, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], # actin
                [adh_energy_total_yesCalcium_M, strain_lim_microtub, halflife_actin, E_microtub_H, eta_microtub, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], # microtubules
                [adh_energy_total_yesCalcium_M, strain_lim_intermfil, halflife_intermfil_L, E_intermfil_H, eta_intermfil, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], # intermediate filaments low
                [adh_energy_total_yesCalcium_M, strain_lim_intermfil, halflife_intermfil_H, E_intermfil_H, eta_intermfil, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], # intermediate filaments high
                [adh_energy_total_yesCalcium_M, strain_lim_intermfil, halflife_intermfil_L, E_multinetwork_L, eta_multinetwork, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], # Xenopus multinetwork low
                [adh_energy_total_yesCalcium_M, strain_lim_intermfil, halflife_intermfil_M, E_multinetwork_M, eta_multinetwork, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], # Xenopus multinetwork medium
                [adh_energy_total_yesCalcium_M, strain_lim_intermfil, halflife_intermfil_H, E_multinetwork_H, eta_multinetwork, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], # Xenopus multinetwork high
                ]

# Fig. 3D - just Xenopus-like solutions for overlapping onto lifeact data:
fig3D_params = [[Gamma_null, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_null], 
                [adh_energy_total_yesCalcium_M, strain_lim_intermfil, halflife_intermfil_L, E_multinetwork_L, eta_multinetwork, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], # Xenopus multinetwork low
                [adh_energy_total_yesCalcium_M, strain_lim_intermfil, halflife_intermfil_M, E_multinetwork_L, eta_multinetwork, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], # Xenopus multinetwork medium
                [adh_energy_total_yesCalcium_M, strain_lim_intermfil, halflife_intermfil_L, E_multinetwork_M, eta_multinetwork, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], # Xenopus multinetwork medium
                [adh_energy_total_yesCalcium_M, strain_lim_intermfil, halflife_intermfil_M, E_multinetwork_M, eta_multinetwork, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], # Xenopus multinetwork medium
                [adh_energy_total_yesCalcium_M, strain_lim_intermfil, halflife_intermfil_H, E_multinetwork_H, eta_multinetwork, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], # Xenopus multinetwork high
                ]


# Fig. S4C - physiological solutions for whole cell surface stresses:
figS4D_params = [[Gamma_null, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_null], 
                 [adh_energy_total_yesCalcium_M, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_actin, halflife_actin, E_actin_H, eta_actin, E_B_example], # actin
                 [adh_energy_total_yesCalcium_M, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_intermfil, halflife_intermfil_L, E_intermfil_H, eta_intermfil, E_B_example], # intermediate filaments low
                 [adh_energy_total_yesCalcium_M, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_intermfil, halflife_intermfil_L, E_multinetwork_L, eta_multinetwork, E_B_example], # Xenopus multinetwork low
                 [adh_energy_total_yesCalcium_M, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_intermfil, halflife_intermfil_M, E_multinetwork_M, eta_multinetwork, E_B_example], # Xenopus multinetwork medium
                 ]


# Fig. SY - just like Fig. 3D - physiological solutions for Xenopus ectoderm 
#           cells, but with a reduced cortical tension everywhere (0.5 times 
#           normal levels), as was seen in Yunyun's paper 
#           (Huang et al. 2022; J. Cell Biol.)
figSY_params = [[Gamma_null, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_null], 
                [adh_energy_total_yesCalcium_M, strain_lim_intermfil, halflife_intermfil_L, E_multinetwork_L, eta_multinetwork, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], # Xenopus multinetwork low
                [adh_energy_total_yesCalcium_M, strain_lim_intermfil, halflife_intermfil_M, E_multinetwork_L, eta_multinetwork, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], # Xenopus multinetwork medium
                [adh_energy_total_yesCalcium_M, strain_lim_intermfil, halflife_intermfil_L, E_multinetwork_M, eta_multinetwork, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], # Xenopus multinetwork medium
                [adh_energy_total_yesCalcium_M, strain_lim_intermfil, halflife_intermfil_M, E_multinetwork_M, eta_multinetwork, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], # Xenopus multinetwork medium
                [adh_energy_total_yesCalcium_M, strain_lim_intermfil, halflife_intermfil_H, E_multinetwork_H, eta_multinetwork, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], # Xenopus multinetwork high
                ]


# Fig. SL - just like Fig. 3D - physiological solutions for Xenopus ectoderm 
#           cells, but with a reduced cortical tension everywhere (equal to 
#           about 0.23mJ/m**2), as was seen in David et al. 2014; Development.
figSL_params = [[Gamma_null, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_null], 
                [adh_energy_total_yesCalcium_M, strain_lim_intermfil, halflife_intermfil_L, E_multinetwork_L, eta_multinetwork, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], # Xenopus multinetwork low
                [adh_energy_total_yesCalcium_M, strain_lim_intermfil, halflife_intermfil_M, E_multinetwork_L, eta_multinetwork, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], # Xenopus multinetwork medium
                [adh_energy_total_yesCalcium_M, strain_lim_intermfil, halflife_intermfil_L, E_multinetwork_M, eta_multinetwork, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], # Xenopus multinetwork medium
                [adh_energy_total_yesCalcium_M, strain_lim_intermfil, halflife_intermfil_M, E_multinetwork_M, eta_multinetwork, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], # Xenopus multinetwork medium
                [adh_energy_total_yesCalcium_M, strain_lim_intermfil, halflife_intermfil_H, E_multinetwork_H, eta_multinetwork, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], # Xenopus multinetwork high
                ]


# Fig. SN - just like Fig. 3D - physiological solutions for Xenopus ectoderm 
#           cells, but with a reduced cortical tension everywhere (equal to 
#           about 0.09mJ/m**2), as was seen in David et al. 2014; Development.
# figSN_params = [[Gamma_null, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_null], 
#                 [adh_energy_total_yesCalcium_M*0.25, strain_lim_intermfil, halflife_intermfil_L, E_multinetwork_L, eta_multinetwork, strain_lim_high, turnover_rate_vfast, 0, 0, E_b_low], # Xenopus multinetwork low
#                 [adh_energy_total_yesCalcium_M*0.25, strain_lim_intermfil, halflife_intermfil_M, E_multinetwork_L, eta_multinetwork, strain_lim_high, turnover_rate_vfast, 0, 0, E_b_low], # Xenopus multinetwork medium
#                 [adh_energy_total_yesCalcium_M*0.25, strain_lim_intermfil, halflife_intermfil_L, E_multinetwork_M, eta_multinetwork, strain_lim_high, turnover_rate_vfast, 0, 0, E_b_low], # Xenopus multinetwork medium
#                 [adh_energy_total_yesCalcium_M*0.25, strain_lim_intermfil, halflife_intermfil_M, E_multinetwork_M, eta_multinetwork, strain_lim_high, turnover_rate_vfast, 0, 0, E_b_low], # Xenopus multinetwork medium
#                 [adh_energy_total_yesCalcium_M*0.25, strain_lim_intermfil, halflife_intermfil_H, E_multinetwork_H, eta_multinetwork, strain_lim_high, turnover_rate_vfast, 0, 0, E_b_low], # Xenopus multinetwork high
#                 ]



all_fig_params = [figS4A_params
                  + figS4B_params 
                  + figS4C_params 
                  + figS4D_params 
                  + fig3A_params 
                  + fig3B_params 
                  + fig3C_params
                  + fig3D_params
                  + figSY_params
                  # + figSN_params
                  + figSL_params
                  ][0]

## remove duplicates from all_fig_params before solving the param space
## The above comment line is probably no longer accurate.
params_w_fig_id = {'figS4A':figS4A_params, 
                   'figS4B':figS4B_params,
                   'figS4C':figS4C_params,
                   'figS4D':figS4D_params,
                   'fig3A':fig3A_params,
                   'fig3B':fig3B_params,
                   'fig3C':fig3C_params,
                   'fig3D':fig3D_params,
                   'figSY':figSY_params,
                    # 'figSN':figSN_params,
                    # 'figSL':figSL_params,
                   }








## parameters below are presented as tuples of (linecolor, linestyle, list_of_parameters)
## to facilitate plotting.
## The equilibrium model:
params_eq = ('k', ':', 'near eq.', [Gamma_null, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_null])


### Realisitc parameters assuming post-Gamma contact spreading through stretching:
params_actin_only = ('tab:green', '-', 'actin', [adh_energy_total_yesCalcium_M, strain_lim_actin, halflife_actin, E_actin_H, eta_actin, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example])
# In the absence of any information we will assume that microtubules are comparable to actin turnover rates
params_microtub_only = ('tab:orange', '-', 'MT', [adh_energy_total_yesCalcium_M, strain_lim_microtub, halflife_actin, E_microtub_H, eta_microtub, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example])
params_intermfil_only_L = ('tab:blue', ':', 'IF (low)', [adh_energy_total_yesCalcium_M, strain_lim_intermfil, halflife_intermfil_L, E_intermfil_H, eta_intermfil, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example])
params_intermfil_only_H = ('tab:blue', '-', 'IF (high)', [adh_energy_total_yesCalcium_M, strain_lim_intermfil, halflife_intermfil_H, E_intermfil_H, eta_intermfil, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example])
# params_intermfil_only_vH = ('tab:blue', '-', [adh_energy_total_yesCalcium_M, strain_lim_high, halflife_intermfil_L, E_intermfil_H, eta_intermfil, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example])
## In the absence of multinetwork turnover rate and strain limit information, 
## we will assume that the network can go up to the highest known 
## (ie. the intermediate filament network):
params_multinetwork_L = ('tab:brown', ':', r'$\it{Xen.}$ E_{low}, IF $\lambda$_{low}', [adh_energy_total_yesCalcium_M, strain_lim_intermfil, halflife_intermfil_L, E_multinetwork_L, eta_multinetwork, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example])
params_multinetwork_M = ('tab:brown', '--', r'$\it{Xen.}$ E_{low}, IF $\lambda$_{low}', [adh_energy_total_yesCalcium_M, strain_lim_intermfil, halflife_intermfil_M, E_multinetwork_M, eta_multinetwork, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example])
params_multinetwork_Hh = ('tab:brown', (0, (3,2,1,2,1,2)), r'$\it{Xen.}$ E_{low}, IF $\lambda$_{med}', [adh_energy_total_yesCalcium_M, strain_lim_intermfil, halflife_intermfil_M, E_multinetwork_L, eta_multinetwork, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example])
params_multinetwork_He = ('tab:brown', '-.', r'$\it{Xen.}$ E_{med}, IF $\lambda$_{low}', [adh_energy_total_yesCalcium_M, strain_lim_intermfil, halflife_intermfil_L, E_multinetwork_M, eta_multinetwork, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example])
## the linestyle above "(0, (3,2,1,2,1,2))" specifies a dash-dot-dot (-..) linestyle, the other ones have convenient and obvious names so those are used there
## the numbers (3, 2, 1, 2, 1, 2) mean first line has length 3, then a space of length 2, then a line of length 1, then a space of length 2, etc. and it repeats when the end is reached
params_multinetwork_Heh = ('tab:brown', '-', r'$\it{Xen.}$ E_{high}, IF $\lambda$_{high}', [adh_energy_total_yesCalcium_M, strain_lim_intermfil, halflife_intermfil_H, E_multinetwork_H, eta_multinetwork, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example])

# params_multinetowrk_vvH = ('k', '-', [adh_energy_total_yesCalcium_H, strain_lim_high, halflife_intermfil_H, E_multinetwork_vH, eta_multinetwork, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example])

params_physiol = [params_eq, params_actin_only, params_microtub_only, 
                  params_intermfil_only_L, params_intermfil_only_H, #params_intermfil_only_vH, 
                  params_multinetwork_L, params_multinetwork_M,  
                  params_multinetwork_He, params_multinetwork_Hh, 
                  params_multinetwork_Heh
                  ]

params_physiol_YY = [params_eq, params_actin_only, params_microtub_only, 
                     params_intermfil_only_L, params_intermfil_only_H, #params_intermfil_only_vH, 
                     params_multinetwork_L, params_multinetwork_M,  
                     params_multinetwork_He, params_multinetwork_Hh, 
                     params_multinetwork_Heh
                     ]

# ### Realisitc parameters assuming post-Gamma contact spreading through rolling:
# params_actin_only = ('tab:green', '-', 'actin', [adh_energy_total_yesCalcium_M, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_actin, halflife_actin, E_actin_H, eta_actin, E_B_example])
# # In the absence of any information we will assume that microtubules are comparable to actin turnover rates
# params_microtub_only = ('tab:orange', '-', 'MT', [adh_energy_total_yesCalcium_M, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_microtub, halflife_actin, E_microtub_H, eta_microtub, E_B_example])
# params_intermfil_only_L = ('tab:blue', ':', 'IF (low)', [adh_energy_total_yesCalcium_M, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_intermfil, halflife_intermfil_L, E_intermfil_H, eta_intermfil, E_B_example])
# params_intermfil_only_H = ('tab:blue', '-', 'IF (high)', [adh_energy_total_yesCalcium_M, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_intermfil, halflife_intermfil_H, E_intermfil_H, eta_intermfil, E_B_example])
# # params_intermfil_only_vH = ('tab:blue', '-', [adh_energy_total_yesCalcium_M, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_high, halflife_intermfil_H, E_intermfil_H, eta_intermfil, E_B_example])
# ## In the absence of multinetwork turnover rate and strain limit information, 
# ## we will assume that the network can go up to the highest known 
# ## (ie. the intermediate filament network):
# params_multinetwork_L = ('tab:brown', '--', r'$\it{Xen.}$ E_{low}, IF $\lambda$_{low}', [adh_energy_total_yesCalcium_M, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_intermfil, halflife_intermfil_L, E_multinetwork_H, eta_multinetwork, E_B_example])
# params_multinetwork_M = ('tab:brown', '-.', r'$\it{Xen.}$ E_{low}, IF $\lambda$_{low}', [adh_energy_total_yesCalcium_M, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_intermfil, halflife_intermfil_M, E_multinetwork_M, eta_multinetwork, E_B_example])
# # params_multinetwork_Hh = ('k', '--', [adh_energy_total_yesCalcium_M, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_intermfil, halflife_intermfil_M, E_multinetwork_L, eta_multinetwork, E_B_example])
# # params_multinetwork_He = ('k', '--', [adh_energy_total_yesCalcium_M, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_intermfil, halflife_low, E_multinetwork_M, eta_multinetwork, E_B_example])
# params_multinetwork_Heh = ('tab:brown', '-', r'$\it{Xen.}$ E_{high}, IF $\lambda$_{high}', [adh_energy_total_yesCalcium_M, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_intermfil, halflife_intermfil_H, E_multinetwork_H, eta_multinetwork, E_B_example])
# # params_multinetowrk_vvH = ('k', '-', [adh_energy_total_yesCalcium_M, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_high, halflife_intermfil_H, E_multinetwork_vH, eta_multinetwork, E_B_example])

# params_rolling = [params_eq, params_actin_only, params_microtub_only, 
#                   params_intermfil_only_L, params_intermfil_only_H, #params_intermfil_only_vH, 
#                   params_multinetwork_L, params_multinetwork_M, 
#                   # params_multinetwork_He, 
#                   params_multinetwork_Heh
#                   ]










## For solutions based on uncorrected F-actin levels:
soln_space_fig_params_uncorr = list()
success_list_fig_params_uncorr = list()
process_time_list_fig_params_uncorr = list()
soln_space_fig_ids_uncorr = list()

for fig_id in params_w_fig_id:
    ## unpack parameters:
    for i,params in enumerate(params_w_fig_id[fig_id]):
        Gamma, strain_lim_actin, t_Apt5, EA, eta_A, strain_lim_nonactin, t_Npt5, EN, eta_N, E_r = params        
    
        ## However for figSY, the cortical tension is half of normal levels:
        if fig_id == 'figSY':
            ## Set initials:
            ## Note that sigma_a is half of normal levels in this example:
            t, t_end = t_init, t_end # Set the range over which to solve.
            Scc0_init, Scm0_init, SNcc0_init, SNcm0_init = get_surf_init(R_s)
            syst = [theta_init, Scc0_init, Scm0_init, SNcc0_init, SNcm0_init]
            consts = (R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a_YY, EA, eta_A, EN, eta_N, eta_m_real, Gamma, E_r, z_cc_func, z_cc_func_uncorr_args, z_cm_func, z_cm_func_args)
        elif fig_id == 'figSL':
            ## Set initials:
            ## Note the initial radius and cortical tension for ecto low based
            ## on David et al 2014; Development are different:
            t, t_end = t_init, t_end # Set the range over which to solve.
            Scc0_init, Scm0_init, SNcc0_init, SNcm0_init = get_surf_init(R_s_low)
            syst = [theta_init, Scc0_init, Scm0_init, SNcc0_init, SNcm0_init]
            consts = (R_s_low, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a_low, EA, eta_A, EN, eta_N, eta_m_real, Gamma, E_r, z_cc_func, z_cc_func_uncorr_args, z_cm_func, z_cm_func_args)
        elif fig_id == 'figSN':
            ## Set initials:
            ## Note that the initial radius and cortical tension for endoderm 
            ## are different:
            t, t_end = t_init, t_end # Set the range over which to solve.
            Scc0_init, Scm0_init, SNcc0_init, SNcm0_init = get_surf_init(R_s_endo)
            syst = [theta_init, Scc0_init, Scm0_init, SNcc0_init, SNcm0_init]
            consts = (R_s_endo, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a_endo, EA, eta_A, EN, eta_N, eta_m_real, Gamma, E_r, z_cc_func, z_cc_func_uncorr_args, z_cm_func, z_cm_func_args)
        else: 
            ## Set initials:
            t, t_end = t_init, t_end # Set the range over which to solve.
            Scc0_init, Scm0_init, SNcc0_init, SNcm0_init = get_surf_init(R_s)
            syst = [theta_init, Scc0_init, Scm0_init, SNcc0_init, SNcm0_init]
            ## For almost all models this is the set of constants:
            consts = (R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, EA, eta_A, EN, eta_N, eta_m_real, Gamma, E_r, z_cc_func, z_cc_func_uncorr_args, z_cm_func, z_cm_func_args)

        
        print('Started angular_veloc_ve_model (uncorr): %s -- no. %d/%d'%(fig_id, i+1, len(params_w_fig_id[fig_id])))
        time_start = tm.perf_counter()

        
        
        soln = VESM((t, t_end), syst, consts, solve_ivp_method, solve_ivp_max_step, solve_ivp_atol, solve_ivp_rtol, strain_lim_event_stol, calcium=True)
        soln_endo = VESM((t, t_end), syst, consts, solve_ivp_method, solve_ivp_max_step, solve_ivp_atol, solve_ivp_rtol, strain_lim_event_stol, calcium=True)
        
        ## some debugging code:
        # print(params)
        # plt.plot(soln.t, soln.y[0])
        # plt.show()
        ##
        
        soln.fig_id = fig_id
        
        soln_space_fig_params_uncorr.append(soln)
        success_list_fig_params_uncorr.append(soln.success)
        soln_space_fig_ids_uncorr.append(fig_id)
           
        if np.all(soln.success):
            print('solve_ivp succeeded...')
        else:
            print('solve_ivp failed...')
            
        time_stop = tm.perf_counter()
        processing_time = time_stop - time_start
        process_time_list_fig_params_uncorr.append(processing_time)
        print('Finished angular_veloc_ve_model (uncorr): %s -- no. %d/%d :: processing time = %.2fsec\n\n'%(fig_id, i+1, len(params_w_fig_id[fig_id]), processing_time))

print('Finished solving param_space for manuscript plot params (for uncorrected F-actin values).\n\n')




## For solutions based on corrected F-actin levels:
soln_space_fig_params_corr = list()
success_list_fig_params_corr = list()
process_time_list_fig_params_corr = list()
soln_space_fig_ids_corr = list()

for fig_id in params_w_fig_id:
    ## unpack parameters:
    for i,params in enumerate(params_w_fig_id[fig_id]):
        Gamma, strain_lim_actin, t_Apt5, EA, eta_A, strain_lim_nonactin, t_Npt5, EN, eta_N, E_r = params        
    
        ## However for figSY, the cortical tension is half of normal levels:
        if fig_id == 'figSY':
            ## Set initials:
            ## Note that sigma_a is half of normal levels in this example:
            t, t_end = t_init, t_end # Set the range over which to solve.
            Scc0_init, Scm0_init, SNcc0_init, SNcm0_init = get_surf_init(R_s)
            syst = [theta_init, Scc0_init, Scm0_init, SNcc0_init, SNcm0_init]
            consts = (R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a_YY, EA, eta_A, EN, eta_N, eta_m_real, Gamma, E_r, z_cc_func, z_cc_func_corr_args, z_cm_func, z_cm_func_args)
        elif fig_id == 'figSL':
            ## Set initials:
            ## Note the initial radius and cortical tension for ecto low based
            ## on David et al 2014; Development are different:
            t, t_end = t_init, t_end # Set the range over which to solve.
            Scc0_init, Scm0_init, SNcc0_init, SNcm0_init = get_surf_init(R_s_low)
            syst = [theta_init, Scc0_init, Scm0_init, SNcc0_init, SNcm0_init]
            consts = (R_s_low, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a_low, EA, eta_A, EN, eta_N, eta_m_real, Gamma, E_r, z_cc_func, z_cc_func_corr_args, z_cm_func, z_cm_func_args)
        elif fig_id == 'figSN':
            ## Set initials:
            ## Note that the initial radius and cortical tension for endoderm 
            ## are different:
            t, t_end = t_init, t_end # Set the range over which to solve.
            Scc0_init, Scm0_init, SNcc0_init, SNcm0_init = get_surf_init(R_s_endo)
            syst = [theta_init, Scc0_init, Scm0_init, SNcc0_init, SNcm0_init]
            consts = (R_s_endo, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a_endo, EA, eta_A, EN, eta_N, eta_m_real, Gamma, E_r, z_cc_func, z_cc_func_corr_args, z_cm_func, z_cm_func_args)
        else: 
            ## Set initials:
            t, t_end = t_init, t_end # Set the range over which to solve.
            Scc0_init, Scm0_init, SNcc0_init, SNcm0_init = get_surf_init(R_s)
            syst = [theta_init, Scc0_init, Scm0_init, SNcc0_init, SNcm0_init]
            ## For almost all models this is the set of constants:
            consts = (R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, EA, eta_A, EN, eta_N, eta_m_real, Gamma, E_r, z_cc_func, z_cc_func_corr_args, z_cm_func, z_cm_func_args)

        
        print('Started angular_veloc_ve_model (corr): %s -- no. %d/%d'%(fig_id, i+1, len(params_w_fig_id[fig_id])))
        time_start = tm.perf_counter()

        
        
        soln = VESM((t, t_end), syst, consts, solve_ivp_method, solve_ivp_max_step, solve_ivp_atol, solve_ivp_rtol, strain_lim_event_stol, calcium=True)
        soln_endo = VESM((t, t_end), syst, consts, solve_ivp_method, solve_ivp_max_step, solve_ivp_atol, solve_ivp_rtol, strain_lim_event_stol, calcium=True)
        
        ## some debugging code:
        # print(params)
        # plt.plot(soln.t, soln.y[0])
        # plt.show()
        ##
        
        soln.fig_id = fig_id
        
        soln_space_fig_params_corr.append(soln)
        success_list_fig_params_corr.append(soln.success)
        soln_space_fig_ids_corr.append(fig_id)
           
        if np.all(soln.success):
            print('solve_ivp succeeded...')
        else:
            print('solve_ivp failed...')
            
        time_stop = tm.perf_counter()
        processing_time = time_stop - time_start
        process_time_list_fig_params_corr.append(processing_time)
        print('Finished angular_veloc_ve_model (corr): %s -- no. %d/%d :: processing time = %.2fsec\n\n'%(fig_id, i+1, len(params_w_fig_id[fig_id]), processing_time))

print('Finished solving param_space for manuscript plot params (for corrected F-actin values).\n\n')





### Plot the solutions:

## the low viscosity parameter examples are covering up too much, so we are
## going to make their lines dashed:
low_visc_params = [[G20pct, strain_lim_high, turnover_rate_vfast, 0, eta1e2, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], 
                   [G20pct, strain_lim_high, turnover_rate_slow, 0, eta1e2, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example]]


### Plotting the uncorrected data:
plt.close('all')
for fig_id in params_w_fig_id:

    cmap_count = 0 # each figure starts with a fresh colormap 
    fig, ax = plt.subplots(figsize=(width_1col*0.7, width_1col*0.7))

    for params_of_interest in params_w_fig_id[fig_id][::-1][:-1]: #reverse the list and take all but the last entry of that reversed list (ie. take all but the first entry in the list and reverse their order.)
        # print(params_of_interest) ## some debugging code
        for i, soln in enumerate(soln_space_fig_params_uncorr):
      
            if np.all(all_fig_params[i] == params_of_interest) and (soln_space_fig_ids_uncorr[i] == fig_id):
                ## The following condition just isolates the low viscosity
                ## examples so that they can be plotted with dashed lines,
                ## the other examples all get solid lines
                if params_of_interest in low_visc_params:
                    time = soln.t
                    theta, Scc0, Scm0, SNcc0, SNcm0 = soln.y
                    ax.plot(time, 2*np.rad2deg(theta), lw=1, c=cmap(cmap_count), ls='--')
                    cmap_count += 1 # each line that gets drawn gets a new color from the color map
                else:
                    time = soln.t
                    theta, Scc0, Scm0, SNcc0, SNcm0 = soln.y
                    
                    ax.plot(time, 2*np.rad2deg(theta), lw=1, c=cmap(cmap_count))#, ls='--')
                    cmap_count += 1 # each line that gets drawn gets a new color from the color map
    params_of_interest = params_w_fig_id[fig_id][0]
    for i, soln in enumerate(soln_space_fig_params_uncorr):
        if np.all(all_fig_params[i] == params_of_interest) and (soln_space_fig_ids_uncorr[i] == fig_id):
            time = soln.t
            theta, Scc0, Scm0, SNcc0, SNcm0, = soln.y
            z_cc, dzcc_dt = z_cc_func(time, *z_cc_func_uncorr_args)
            ax.plot(time, 2*np.rad2deg(theta), lw=1, c='k', alpha=1.0, ls=':')
         
    ax.set_xlim(-10, 1800)
    ax.set_xticks([0,600,1200,1800])#,2400])
    ax.set_xticklabels(['0','600','1200','1800'])#,'2400'])
    ax.set_ylabel('Contact Angle (deg)')
    ax.set_xlabel('Time (s)')
            
    time_zcc = np.linspace(0, 3600, 36001)
    z_cc, dzcc_dt = z_cc_func(time, *z_cc_func_uncorr_args)
    z_cm, dzcm_dt = z_cm_func(time, *z_cm_func_args)
    axb = ax.twinx()
    axb.axvline(t_half, c='grey', lw=1, ls=':', zorder=7)
    axb.plot(time, z_cc/z_cm, c='grey', lw=1, zorder=8)
    axb.set_ylabel('Rel. Contact Actin', c='grey', rotation=270, va='bottom', ha='center')
    axb.set_ylim(0, 1)
    # axb.set_xlim(0,200)
    axb.tick_params(axis='y', colors='grey')
            
    ax.set_zorder(axb.get_zorder()+1) # this puts ax in front of axb
    ax.patch.set_visible(False) # this sets the background of ax
    
    # plt.draw()
    # plt.show()
    
    fig.savefig(Path.joinpath(man_plots_dir, r"%s_uncorr_time_course.tif"%fig_id), dpi=dpi_LineArt, bbox_inches='tight')
    plt.close('all')
                
    
    
    fig, ax = plt.subplots(figsize=(width_1col*0.7,width_1col*0.7))
    ax.axhline(z_plat_uncorr, c='grey', ls=':')
    # ax.axhline(z_plat, c='grey', ls=':')
    ax.axvline(180, c='grey', ls=':')

    for params_of_interest in params_w_fig_id[fig_id][::-1][:-1]:
        for i, soln in enumerate(soln_space_fig_params_uncorr):
            if np.all(all_fig_params[i] == params_of_interest) and (soln_space_fig_ids_uncorr[i] == fig_id):
                time = soln.t
                theta, Scc0, Scm0, SNcc0, SNcm0 = soln.y
                
                z_cc, dzcc_dt = z_cc_func(time, *z_cc_func_uncorr_args)
                ax.plot(2*np.rad2deg(theta), z_cc/z_cm, lw=1)
    params_of_interest = params_w_fig_id[fig_id][0]
    for i, soln in enumerate(soln_space_fig_params_uncorr):
        if np.all(all_fig_params[i] == params_of_interest) and (soln_space_fig_ids_uncorr[i] == fig_id):
            time = soln.t
            theta, Scc0, Scm0, SNcc0, SNcm0 = soln.y
            z_cc, dzcc_dt = z_cc_func(time, *z_cc_func_uncorr_args)
            ax.plot(2*np.rad2deg(theta), z_cc/z_cm, lw=1, c='k', alpha=1.0, ls=':')
    
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.linspace(0,1, 11), minor=True)
    ax.set_ylabel('Rel. Contact Actin')
    ax.set_xlim(0, 220)
    ax.set_xticks([0,50,100,150,200], minor=True)
    ax.set_xlabel('Contact Angle (deg)')

    fig.savefig(Path.joinpath(man_plots_dir, r"%s_uncorr.tif"%fig_id), dpi=dpi_LineArt, bbox_inches='tight')
    plt.close('all')



### Plotting the corrected data:
plt.close('all')
for fig_id in params_w_fig_id:

    cmap_count = 0 # each figure starts with a fresh colormap 
    fig, ax = plt.subplots(figsize=(width_1col*0.7, width_1col*0.7))

    for params_of_interest in params_w_fig_id[fig_id][::-1][:-1]: #reverse the list and take all but the last entry of that reversed list (ie. take all but the first entry in the list and reverse their order.)
        # print(params_of_interest) ## some debugging code
        for i, soln in enumerate(soln_space_fig_params_corr):
      
            if np.all(all_fig_params[i] == params_of_interest) and (soln_space_fig_ids_corr[i] == fig_id):
                ## The following condition just isolates the low viscosity
                ## examples so that they can be plotted with dashed lines,
                ## the other examples all get solid lines
                if params_of_interest in low_visc_params:
                    time = soln.t
                    theta, Scc0, Scm0, SNcc0, SNcm0 = soln.y
                    ax.plot(time, 2*np.rad2deg(theta), lw=1, c=cmap(cmap_count), ls='--')
                    cmap_count += 1 # each line that gets drawn gets a new color from the color map
                else:
                    time = soln.t
                    theta, Scc0, Scm0, SNcc0, SNcm0 = soln.y
                    
                    ax.plot(time, 2*np.rad2deg(theta), lw=1, c=cmap(cmap_count))#, ls='--')
                    cmap_count += 1 # each line that gets drawn gets a new color from the color map
    params_of_interest = params_w_fig_id[fig_id][0]
    for i, soln in enumerate(soln_space_fig_params_corr):
        if np.all(all_fig_params[i] == params_of_interest) and (soln_space_fig_ids_corr[i] == fig_id):
            time = soln.t
            theta, Scc0, Scm0, SNcc0, SNcm0, = soln.y
            z_cc, dzcc_dt = z_cc_func(time, *z_cc_func_corr_args)
            ax.plot(time, 2*np.rad2deg(theta), lw=1, c='k', alpha=1.0, ls=':')
         
    ax.set_xlim(-10, 1800)
    ax.set_xticks([0,600,1200,1800])#,2400])
    ax.set_xticklabels(['0','600','1200','1800'])#,'2400'])
    ax.set_ylabel('Contact Angle (deg)')
    ax.set_xlabel('Time (s)')
            
    time_zcc = np.linspace(0, 3600, 36001)
    z_cc, dzcc_dt = z_cc_func(time, *z_cc_func_corr_args)
    z_cm, dzcm_dt = z_cm_func(time, *z_cm_func_args)
    axb = ax.twinx()
    axb.axvline(t_half, c='grey', lw=1, ls=':', zorder=7)
    axb.plot(time, z_cc/z_cm, c='grey', lw=1, zorder=8)
    axb.set_ylabel('Rel. Contact Actin', c='grey', rotation=270, va='bottom', ha='center')
    axb.set_ylim(0, 1)
    # axb.set_xlim(0,200)
    axb.tick_params(axis='y', colors='grey')
            
    ax.set_zorder(axb.get_zorder()+1) # this puts ax in front of axb
    ax.patch.set_visible(False) # this sets the background of ax
    
    # plt.draw()
    # plt.show()
    
    fig.savefig(Path.joinpath(zcorr_plots_dir, r"%s_corr_time_course.tif"%fig_id), dpi=dpi_LineArt, bbox_inches='tight')
    plt.close('all')
                
    
    
    fig, ax = plt.subplots(figsize=(width_1col*0.7,width_1col*0.7))
    ax.axhline(z_plat_corr, c='grey', ls=':')
    # ax.axhline(z_plat, c='grey', ls=':')
    ax.axvline(180, c='grey', ls=':')

    for params_of_interest in params_w_fig_id[fig_id][::-1][:-1]:
        for i, soln in enumerate(soln_space_fig_params_corr):
            if np.all(all_fig_params[i] == params_of_interest) and (soln_space_fig_ids_corr[i] == fig_id):
                time = soln.t
                theta, Scc0, Scm0, SNcc0, SNcm0 = soln.y
                
                z_cc, dzcc_dt = z_cc_func(time, *z_cc_func_corr_args)
                ax.plot(2*np.rad2deg(theta), z_cc/z_cm, lw=1)
    params_of_interest = params_w_fig_id[fig_id][0]
    for i, soln in enumerate(soln_space_fig_params_corr):
        if np.all(all_fig_params[i] == params_of_interest) and (soln_space_fig_ids_corr[i] == fig_id):
            time = soln.t
            theta, Scc0, Scm0, SNcc0, SNcm0 = soln.y
            z_cc, dzcc_dt = z_cc_func(time, *z_cc_func_corr_args)
            ax.plot(2*np.rad2deg(theta), z_cc/z_cm, lw=1, c='k', alpha=1.0, ls=':')
    
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.linspace(0,1, 11), minor=True)
    ax.set_ylabel('Rel. Contact Actin')
    ax.set_xlim(0, 220)
    ax.set_xticks([0,50,100,150,200], minor=True)
    ax.set_xlabel('Contact Angle (deg)')

    fig.savefig(Path.joinpath(zcorr_plots_dir, r"%s_corr.tif"%fig_id), dpi=dpi_LineArt, bbox_inches='tight')
    plt.close('all')









## Solve for each set of parameters:
## using corrected F-actin signals:
solns_physiol_corr = list()
for linecol, linesty, lb, params in params_physiol:
    
    ## unpack the parameters and assign some constants: 
    Gamma, strain_lim_actin, t_Apt5, EA, eta_A, strain_lim_nonactin, t_Npt5, EN, eta_N, E_r = params
    consts = (R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, EA, eta_A, EN, eta_N, eta_m_real, Gamma, E_r, z_cc_func, z_cc_func_corr_args, z_cm_func, z_cm_func_args)
    
    t, t_end = t_init, t_end # Set the range over which to solve.
    Scc0_init, Scm0_init, SNcc0_init, SNcm0_init = get_surf_init(R_s)
    syst = [theta_init, Scc0_init, Scm0_init, SNcc0_init, SNcm0_init]
    
    ## solve the system:
    soln = VESM((t, t_end), syst, consts, solve_ivp_method, solve_ivp_max_step, solve_ivp_atol, solve_ivp_rtol, strain_lim_event_stol, calcium=True)

    solns_physiol_corr.append((linecol, linesty, lb, soln))



## using uncorrected F-actin signals:
solns_physiol_uncorr = list()
for linecol, linesty, lb, params in params_physiol:
    
    ## unpack the parameters and assign some constants: 
    Gamma, strain_lim_actin, t_Apt5, EA, eta_A, strain_lim_nonactin, t_Npt5, EN, eta_N, E_r = params
    consts = (R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, EA, eta_A, EN, eta_N, eta_m_real, Gamma, E_r, z_cc_func, z_cc_func_uncorr_args, z_cm_func, z_cm_func_args)
    
    t, t_end = t_init, t_end # Set the range over which to solve.
    Scc0_init, Scm0_init, SNcc0_init, SNcm0_init = get_surf_init(R_s)
    syst = [theta_init, Scc0_init, Scm0_init, SNcc0_init, SNcm0_init]
    
    ## solve the system:
    soln = VESM((t, t_end), syst, consts, solve_ivp_method, solve_ivp_max_step, solve_ivp_atol, solve_ivp_rtol, strain_lim_event_stol, calcium=True)

    solns_physiol_uncorr.append((linecol, linesty, lb, soln))



## using uncorrected F-actin signals for sigma_a_YY:
solns_physiol_YY = list()
for linecol, linesty, lb, params in params_physiol_YY:
    
    ## unpack the parameters and assign some constants: 
    Gamma, strain_lim_actin, t_Apt5, EA, eta_A, strain_lim_nonactin, t_Npt5, EN, eta_N, E_r = params
    consts = (R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a_YY, EA, eta_A, EN, eta_N, eta_m_real, Gamma, E_r, z_cc_func, z_cc_func_uncorr_args, z_cm_func, z_cm_func_args)
    
    t, t_end = t_init, t_end # Set the range over which to solve.
    Scc0_init, Scm0_init, SNcc0_init, SNcm0_init = get_surf_init(R_s)
    syst = [theta_init, Scc0_init, Scm0_init, SNcc0_init, SNcm0_init]
    
    ## solve the system:
    soln = VESM((t, t_end), syst, consts, solve_ivp_method, solve_ivp_max_step, solve_ivp_atol, solve_ivp_rtol, strain_lim_event_stol, calcium=True)

    solns_physiol_YY.append((linecol, linesty, lb, soln))




# solns_rolling = list()
# for linecol, linesty, lb, params in params_rolling:
    
#     ## unpack the parameters and assign some constants: 
#     Gamma, strain_lim_actin, t_Apt5, EA, eta_A, strain_lim_nonactin, t_Npt5, EN, eta_N, E_r = params
#     consts = (R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, EA, eta_A, EN, eta_N, eta_m_real, Gamma, E_r, z_cc_func, z_cc_func_args, z_cm_func, z_cm_func_args)
    
#     t, t_end = t_init, t_end # Set the range over which to solve.
#     syst = [theta_init, Scc0_init, Scm0_init, SNcc0_init, SNcm0_init]

#     ## solve the system:
#     soln = VESM((t, t_end), syst, consts, solve_ivp_method, solve_ivp_max_step, solve_ivp_atol, solve_ivp_rtol, strain_lim_event_stol, calcium=True)

#     solns_rolling.append((linecol, linesty, lb, soln))



## Plot a timecourse of the results based on uncorrected data:
fig, ax = plt.subplots(figsize=(width_1col*0.7, width_1col*0.7)) # set up the figure. 
for linecol, linesty, lb, soln in solns_physiol_uncorr:
    # unpack the results:
    time = soln.t
    theta, Scc0, Scm0, SNcc0, SNcm0 = soln.y
    z_cc, dzcc_dt = z_cc_func(time, *z_cc_func_uncorr_args)
    z_cm, dzcm_dt = z_cm_func(time, *z_cm_func_args)

    ## plot the results:
    ax.plot(time, 2*np.rad2deg(theta), c=linecol, ls=linesty, lw=1, label=lb)
    ax.set_xlim(-10, 1800)
    ax.set_xticks([0,600,1200,1800])#,2400])
    ax.set_xticklabels(['0','600','1200','1800'])#,'2400'])
    ax.set_ylabel('Contact Angle (deg)')
    ax.set_xlabel('Time (s)')
    
    
    axb = ax.twinx()
    # axb.axvline(t_half, c='grey', lw=1, ls=':', zorder=7)
    axb.plot(time, z_cc/z_cm, c='grey', lw=1, zorder=8)
    # axb.plot(time/60, z_cc/z_cm, c='grey')
    axb.set_ylabel('Rel. Contact Actin', c='grey', rotation=270, va='bottom', ha='center')
    axb.set_ylim(0, 1)
    axb.tick_params(axis='y', colors='grey')
    # axb.set_ylim(0,1)
    
    ax.set_zorder(axb.get_zorder()+1) # this puts ax in front of axb
    ax.patch.set_visible(False) # this sets the background of ax
    
## save the resulting plot:
fig.savefig(Path.joinpath(man_plots_dir, "physiol_vals_uncorr_timecourse_contact_surf.tif"), dpi=dpi_LineArt, bbox_inches='tight')
plt.close('fig')



## Plot a timecourse of the results based on corrected data:
fig, ax = plt.subplots(figsize=(width_1col*0.7, width_1col*0.7)) # set up the figure. 
for linecol, linesty, lb, soln in solns_physiol_corr:
    # unpack the results:
    time = soln.t
    theta, Scc0, Scm0, SNcc0, SNcm0 = soln.y
    z_cc, dzcc_dt = z_cc_func(time, *z_cc_func_corr_args)
    z_cm, dzcm_dt = z_cm_func(time, *z_cm_func_args)

    ## plot the results:
    ax.plot(time, 2*np.rad2deg(theta), c=linecol, ls=linesty, lw=1, label=lb)
    ax.set_xlim(-10, 1800)
    ax.set_xticks([0,600,1200,1800])#,2400])
    ax.set_xticklabels(['0','600','1200','1800'])#,'2400'])
    ax.set_ylabel('Contact Angle (deg)')
    ax.set_xlabel('Time (s)')
    
    
    axb = ax.twinx()
    # axb.axvline(t_half, c='grey', lw=1, ls=':', zorder=7)
    axb.plot(time, z_cc/z_cm, c='grey', lw=1, zorder=8)
    # axb.plot(time/60, z_cc/z_cm, c='grey')
    axb.set_ylabel('Rel. Contact Actin', c='grey', rotation=270, va='bottom', ha='center')
    axb.set_ylim(0, 1)
    axb.tick_params(axis='y', colors='grey')
    # axb.set_ylim(0,1)
    
    ax.set_zorder(axb.get_zorder()+1) # this puts ax in front of axb
    ax.patch.set_visible(False) # this sets the background of ax
    
## save the resulting plot:
fig.savefig(Path.joinpath(zcorr_plots_dir, "physiol_vals_corr_timecourse_contact_surf.tif"), dpi=dpi_LineArt, bbox_inches='tight')
plt.close('fig')



## Plot a timecourse of the results based on uncorrected data with sigma_a_YY:
fig, ax = plt.subplots(figsize=(width_1col*0.7, width_1col*0.7)) # set up the figure. 
for linecol, linesty, lb, soln in solns_physiol_YY:
    # unpack the results:
    time = soln.t
    theta, Scc0, Scm0, SNcc0, SNcm0 = soln.y
    z_cc, dzcc_dt = z_cc_func(time, *z_cc_func_uncorr_args)
    z_cm, dzcm_dt = z_cm_func(time, *z_cm_func_args)

    ## plot the results:
    ax.plot(time, 2*np.rad2deg(theta), c=linecol, ls=linesty, lw=1, label=lb)
    ax.set_xlim(-10, 1800)
    ax.set_xticks([0,600,1200,1800])#,2400])
    ax.set_xticklabels(['0','600','1200','1800'])#,'2400'])
    ax.set_ylabel('Contact Angle (deg)')
    ax.set_xlabel('Time (s)')
    
    
    axb = ax.twinx()
    # axb.axvline(t_half, c='grey', lw=1, ls=':', zorder=7)
    axb.plot(time, z_cc/z_cm, c='grey', lw=1, zorder=8)
    # axb.plot(time/60, z_cc/z_cm, c='grey')
    axb.set_ylabel('Rel. Contact Actin', c='grey', rotation=270, va='bottom', ha='center')
    axb.set_ylim(0, 1)
    axb.tick_params(axis='y', colors='grey')
    # axb.set_ylim(0,1)
    
    ax.set_zorder(axb.get_zorder()+1) # this puts ax in front of axb
    ax.patch.set_visible(False) # this sets the background of ax
    
## save the resulting plot:
fig.savefig(Path.joinpath(man_plots_dir, "physiol_vals_uncorr_timecourse_contact_surf_YY.tif"), dpi=dpi_LineArt, bbox_inches='tight')
plt.close('fig')



# fig, ax = plt.subplots(figsize=(width_1col*0.7, width_1col*0.7)) # set up the figure. 
# for linecol, linesty, lb, soln in solns_rolling:
#     ## unpack the results:
#     time = soln.t
#     theta, Scc0, Scm0, SNcc0, SNcm0 = soln.y
#     z_cc, dzcc_dt = z_cc_func(time, *z_cc_func_args)
#     z_cm = z_cm_func(time, *z_cm_func_args)

#     ## plot the results:
#     ax.plot(time/60, 2*np.rad2deg(theta), c=linecol, ls=linesty)
#     ax.set_xlim(-5/60, 1800/60)
#     ax.set_ylim(0, 200)
    
#     axb = ax.twinx()
#     axb.plot(time/60, z_cc/z_cm, c='grey')
#     axb.tick_params(axis='y', colors='grey')
#     axb.set_ylim(0,1)
    
#     ax.set_zorder(axb.get_zorder()+1) # this puts ax in front of axb
#     ax.patch.set_visible(False) # this sets the background of ax

# ## save the resulting plot:
# fig.savefig(Path.joinpath(man_plots_dir, "physiol_vals_timecourse_whole_surf.tif"), dpi=dpi_LineArt, bbox_inches='tight')
# plt.close('fig')



# ## Plot a the angles against the contact actin:
# fig, ax = plt.subplots(figsize=(width_1col*0.7, width_1col*0.7)) # set up the figure. 
# for linecol, linesty, lb, soln in solns_stretching[::-1]:
#     ## unpack the results:
#     time = soln.t
#     theta, Scc0, Scm0, SNcc0, SNcm0 = soln.y    
#     z_cc, dzcc_dt = z_cc_func(time, *z_cc_func_args)

#     ## plot the results:
#     ax.axhline(z_plat, c='grey', ls=':')
#     ax.axvline(180, c='grey', ls=':')
#     ax.plot(2*np.rad2deg(theta), z_cc, c=linecol, ls=linesty, lw=1, label=lb)
#     # ax.set_xlim(0,200)
#     # ax.set_ylim(0,2)
    
#     ax.set_ylim(0, 1.05)
#     ax.set_yticks(np.linspace(0,1, 11), minor=True)
#     ax.set_ylabel('Rel. Contact Actin')
#     ax.set_xlim(0, 220)
#     ax.set_xticks([0,50,100,150,200], minor=True)
#     ax.set_xlabel('Contact Angle (deg)')
# ## save the resulting plot:
# fig.savefig(Path.joinpath(man_plots_dir, "physiol_vals_angle_vs_actin_contact_surf.tif"), dpi=dpi_LineArt, bbox_inches='tight')
# plt.close('fig')


# fig, ax = plt.subplots(figsize=(width_1col*0.7, width_1col*0.7)) # set up the figure. 
# for linecol, linesty, lb, soln in solns_rolling:
#     ## unpack the results:
#     time = soln.t
#     theta, Scc0, Scm0, SNcc0, SNcm0 = soln.y
#     z_cc, dzcc_dt = z_cc_func(time, *z_cc_func_args)

#     ## plot the results:
#     ax.plot(2*np.rad2deg(theta), z_cc, c=linecol, ls=linesty)
#     ax.set_xlim(0,200)
#     ax.set_ylim(0,2)
# ## save the resulting plot:
# fig.savefig(Path.joinpath(man_plots_dir, "physiol_vals_angle_vs_actin_whole_surf.tif"), dpi=dpi_LineArt, bbox_inches='tight')
# plt.close('fig')




### Some code for looking at individual solutions from the physiological datasets
# i = 3
# plt.plot(solns_stretching[i][-1].t, solns_stretching[i][-1].y[0])
# params = params_stretching[i][-1]




### What is the maximum strain for the whole-cell surface area?
print("How much larger is the cells whole surface than its \
initial spherical surface area?")
theta_full_adh =  np.pi/2
R_c = (4**(1/3) * R_s) / (4 - (1 - np.cos(theta_full_adh))**2 * (2 + np.cos(theta_full_adh)))**(1/3)
Ss = 4 * np.pi * R_s**2    
Scm = 2 * np.pi * R_c**2 * (1 + np.cos(theta_full_adh))
Scc = np.pi * (R_c * np.sin(theta_full_adh))**2

strain_full_adh = (Scm + Scc) / Ss
print("With a spherical radius of %.2f, the strain at full \
adhesion is %.2f"%(R_s,strain_full_adh))

print("Actually, we can see that the R_s terms in \
'(Scm + Scc) / Ss' cancel out, so the max strain is always \
about 1.19, regardless of the cell size.")






### Also want to compare the rate of change of the surface Scc with the rate of
### change of the surface Scm (both with respect to time).
theta_range = np.linspace(0, np.pi/2, 901) # the whole range of realistic contact angles

## The radius of the spherical cap:
Rc_range = (4**(1/3) * R_s) / (4 - (1 - np.cos(theta_range))**2 * (2 + np.cos(theta_range)))**(1/3)

## The rate of change of said radius as the contact angle changes:
dRc_dtheta_range = [(-4**(1/3) * R_s * 3 * np.sin(theta_range) * (np.cos(theta_range)**2 - 1)) 
                    / (3 * (4 - (1 - np.cos(theta_range))**2 * (2 + np.cos(theta_range)))**(4/3))][-1]


## The surface area of the cell-medium interface:
Scm = 2 * np.pi * Rc_range**2 * (1 + np.cos(theta_range))

## Rate of change of said surface with respect to contact angle:
dScm_dtheta = 2 * np.pi * (2 * Rc_range * (1 + np.cos(theta_range)) * dRc_dtheta_range - (Rc_range**2 * np.sin(theta_range)))


## The surface area of the cell-cell interface:
Scc = np.pi * (Rc_range * np.sin(theta_range))**2

## Rate of change of said surface with respect to contact angle:
dScc_dtheta = 2 * np.pi * Rc_range * np.sin(theta_range) * (np.sin(theta_range) * dRc_dtheta_range + Rc_range * np.cos(theta_range))



fig, ax = plt.subplots(figsize=(width_1col, width_1col))
ax.plot(np.rad2deg(theta_range), abs(dScc_dtheta), label=r'$\frac{dS_{cc}}{d\theta}$')
ax.plot(np.rad2deg(theta_range), abs(dScm_dtheta), ls='--', label=r'$\frac{dS_{cm}}{d\theta}$')
ax.set_xlim(0,90)
ax.set_xlabel('Contact Angle (deg)')
ax.set_ylabel('Surface Area rate of change \nw.r.t. Contact Angle ($\mu$$m^2$/deg)')
ax.legend()
fig.savefig(Path.joinpath(out_dir, "dScc_vs_dScm_dtheta.tif"), dpi=dpi_LineArt, bbox_inches='tight')







# ### Now plot the 
# fig, ax = plt.subplots(figsize=(width_1col*0.70, width_1col*0.70))
fig_of_interest = 'fig3D'
# for params in params_w_fig_id[fig_of_interest][1:]:
#     for i, soln in enumerate(soln_space_fig_params):
#         if np.all(all_fig_params[i] == params) and (soln_space_fig_ids[i] == fig_of_interest):
#             time = soln.t
#             # theta, theta0cc, theta0cm, theta0N = soln.y
#             theta, Scc0, Scm0, SNcc0, SNcm0 = soln.y
#             z_cc, dzcc_dt = z_cc_func(time, *z_cc_func_args)
#             z_cm, dzcm_dt = z_cm_func(time, *z_cm_func_args)
#             ax.plot(time/60, 2*np.rad2deg(theta), lw=2, alpha=1.0)
# for i, soln in enumerate(soln_space_fig_params):
#     if np.all(all_fig_params[i] == params_w_fig_id[fig_of_interest][0]) and (soln_space_fig_ids[i] == fig_of_interest):
#         time = soln.t
#         # theta, theta0cc, theta0cm, theta0N = soln.y
#         theta, Scc0, Scm0, SNcc0, SNcm0 = soln.y
#         z_cc, dzcc_dt = z_cc_func(time, *z_cc_func_args)
#         z_cm, dzcm_dt = z_cm_func(time, *z_cm_func_args)
#         ax.plot(time/60, 2*np.rad2deg(theta), lw=1, c='k', alpha=1.0, ls=':')
# ax.set_xlim(0,1800/60)
# ax.set_ylim(0,200)
# ax.set_xlabel('Time (minutes)')
# ax.set_ylabel('Contact Angle (deg)')

# axb = ax.twinx()
# axb.plot(time/60, z_cc/z_cm, c='grey', lw=1, ls='--')
# axb.set_ylim(0,1)
# axb.set_ylabel('Norm. Contact Fluor. (N.D.)', rotation=270, va='bottom', ha='center', color='grey')
# axb.tick_params(colors='grey')
# # plt.tight_layout()
# fig.savefig(Path.joinpath(out_dir, r'angle_vs_time_w_model_%s.tif'%fig_of_interest), dpi=dpi_LineArt, bbox_inches='tight')
# plt.close(fig)






### Plot the physiological models with the LifeAct-GFP data when using 
### uncorrected models and data:

# Each timepoint in the simulation is associated with a theoretical 
# contact angle and therefore a theoretical contact diameter which was then 
# drawn as pixels and measured. The values in the table for angle and contact
# diameter are how that theoretical angle appears under our imaging conditions.
# Therefore we will make a list of the theoretical angles and contact diameters
theor_ang = np.linspace(0, np.pi/2, 91) 
theor_contnd = SpherCapGeom(theor_ang, R_s/R_s).r_c * 2 # the "*2" is to turn the radius into diameter
theor_relten = angle2reltension(theor_ang)



data = la_contnd_df[la_contnd_df['Experimental_Condition'] == 'ee1xMBS']

fig, ax = plt.subplots(figsize=(width_1col*0.7, width_1col*0.7)) # set up the figure. 

for linecol, linesty, lb, soln in solns_physiol_uncorr[::-1]:
    ## unpack the results:
    time = soln.t
    theta, Scc0, Scm0, SNcc0, SNcm0 = soln.y    
    z_cc, dzcc_dt = z_cc_func(time, *z_cc_func_uncorr_args)

    ## plot the results:
    ax.plot(2*np.rad2deg(theta), z_cc, c=linecol, ls=linesty, lw=1, label=lb)

# sns.scatterplot(x='Average Angle (deg)', y='lifeact_fluor_normed_sub_mean_corr', color='grey', 
#                 data=la_contnd_df[la_contnd_df['Experimental_Condition'] == 'ee1xMBS'], 
#                 s=5, linewidth=0, alpha=0.5, ax=ax, legend=False, zorder=1)
ax.errorbar(x=data['Average Angle (deg)'], y=data['lifeact_fluor_normed_mean'], 
            yerr=data['lifeact_fluor_normed_sem'], color='grey', marker='.', 
            ms=1, lw=0, elinewidth=1, ecolor='lightgrey', alpha=1, zorder=1)

ax.axhline(z_plat_uncorr, c='grey', ls='-', zorder=0)
# ax.axhline(z_plat, c='grey', ls='-', zorder=0)
ax.axvline(180, c='grey', ls='-', zorder=0)
    
ax.set_ylim(0, 2)
ax.set_yticks(np.linspace(0, 2, 21), minor=True)
ax.set_xlim(0, 200)
ax.set_xticks([0, 50, 100, 150, 200], minor=True)
ax.set_xlabel('Contact Angle (deg)')
ax.set_ylabel('Norm. Contact Fluor. (N.D.)')
# save the resulting plot:
# fig.legend()
fig.savefig(Path.joinpath(man_plots_dir, r'angle_vs_fluor_lagfp_w_model_uncorr_%s.tif'%fig_of_interest), dpi=dpi_LineArt, bbox_inches='tight')
plt.close(fig)

### also the LifeAct-GFP data alone:
fig, ax = plt.subplots(figsize=(width_1col*0.7, width_1col*0.7)) # set up the figure. 

# data = la_contnd_df[la_contnd_df['Experimental_Condition'] == 'ee1xMBS']
sns.scatterplot(x='Average Angle (deg)', y='lifeact_fluor_normed_mean', hue='Source_File', 
                data=data, 
                s=3, linewidth=0, alpha=0.5, ax=ax, legend=False, zorder=2)
ax.errorbar(x=data['Average Angle (deg)'], y=data['lifeact_fluor_normed_mean'], 
            yerr=data['lifeact_fluor_normed_sem'], color='grey', marker='.', 
            ms=0, lw=0, elinewidth=0.5, ecolor='lightgrey', alpha=1, zorder=1)
ax.plot(np.rad2deg(theor_ang)*2, theor_relten, c='k', ls='--', zorder=0) # the "*2" is because each cell in the cellpair has an angle associated with it's spherical cap geometry

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
fig.savefig(Path.joinpath(man_plots_dir, r'angle_vs_fluor_lagfp_uncorr.tif'), dpi=dpi_LineArt, bbox_inches='tight')
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
fig.savefig(Path.joinpath(man_plots_dir, r'contnd_vs_fluor_lagfp_uncorr.tif'), dpi=dpi_LineArt, bbox_inches='tight')
plt.close(fig)



### and also when using corrected models and data:
fig, ax = plt.subplots(figsize=(width_1col*0.7, width_1col*0.7)) # set up the figure. 

for linecol, linesty, lb, soln in solns_physiol_corr[::-1]:
    ## unpack the results:
    time = soln.t
    theta, Scc0, Scm0, SNcc0, SNcm0 = soln.y    
    z_cc, dzcc_dt = z_cc_func(time, *z_cc_func_corr_args)

    ## plot the results:
    ax.plot(2*np.rad2deg(theta), z_cc, c=linecol, ls=linesty, lw=1, label=lb)

# data = la_contnd_df[la_contnd_df['Experimental_Condition'] == 'ee1xMBS']
# sns.scatterplot(x='Average Angle (deg)', y='lifeact_fluor_normed_sub_mean_corr', color='grey', 
#                 data=la_contnd_df[la_contnd_df['Experimental_Condition'] == 'ee1xMBS'], 
#                 s=5, linewidth=0, alpha=0.5, ax=ax, legend=False, zorder=1)
ax.errorbar(x=data['Average Angle (deg)'], y=data['lifeact_fluor_normed_sub_mean_corr'], 
            yerr=data['lifeact_fluor_normed_sub_sem_corr'], color='grey', marker='.', 
            ms=1, lw=0, elinewidth=1, ecolor='lightgrey', alpha=1, zorder=1)

ax.axhline(z_plat_corr, c='grey', ls='-', zorder=0)
# ax.axhline(z_plat, c='grey', ls='-', zorder=0)
ax.axvline(180, c='grey', ls='-', zorder=0)
    
ax.set_ylim(-0.2, 2)
ax.set_yticks(np.linspace(-0.2, 2, 23), minor=True)
ax.set_xlim(0, 200)
ax.set_xticks([0, 50, 100, 150, 200], minor=True)
ax.set_xlabel('Contact Angle (deg)')
ax.set_ylabel('Norm. Contact Fluor. (N.D.)')
# save the resulting plot:
# fig.legend()
fig.savefig(Path.joinpath(zcorr_plots_dir, r'angle_vs_fluor_lagfp_w_model_corr_%s.tif'%fig_of_interest), dpi=dpi_LineArt, bbox_inches='tight')
plt.close(fig)

### also the LifeAct-GFP data alone:
fig, ax = plt.subplots(figsize=(width_1col*0.7, width_1col*0.7)) # set up the figure. 

# data = la_contnd_df[la_contnd_df['Experimental_Condition'] == 'ee1xMBS']
sns.scatterplot(x='Average Angle (deg)', y='lifeact_fluor_normed_sub_mean_corr', hue='Source_File', 
                data=data, 
                s=3, linewidth=0, alpha=0.5, ax=ax, legend=False, zorder=2)
ax.errorbar(x=data['Average Angle (deg)'], y=data['lifeact_fluor_normed_sub_mean_corr'], 
            yerr=data['lifeact_fluor_normed_sub_sem_corr'], color='grey', marker='.', 
            ms=0, lw=0, elinewidth=0.5, ecolor='lightgrey', alpha=1, zorder=1)
ax.plot(np.rad2deg(theor_ang)*2, theor_relten, c='k', ls='--', zorder=0) # the "*2" is because each cell in the cellpair has an angle associated with it's spherical cap geometry

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
fig.savefig(Path.joinpath(zcorr_plots_dir, r'angle_vs_fluor_lagfp_corr.tif'), dpi=dpi_LineArt, bbox_inches='tight')
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
fig.savefig(Path.joinpath(zcorr_plots_dir, r'contnd_vs_fluor_lagfp_corr.tif'), dpi=dpi_LineArt, bbox_inches='tight')
plt.close(fig)




### and also when using corrected models and data but without limiting Y-axis:
fig, ax = plt.subplots(figsize=(width_1col*0.7, width_1col*0.7)) # set up the figure. 

for linecol, linesty, lb, soln in solns_physiol_corr[::-1]:
    ## unpack the results:
    time = soln.t
    theta, Scc0, Scm0, SNcc0, SNcm0 = soln.y    
    z_cc, dzcc_dt = z_cc_func(time, *z_cc_func_corr_args)

    ## plot the results:
    ax.plot(2*np.rad2deg(theta), z_cc, c=linecol, ls=linesty, lw=1, label=lb)

# data = la_contnd_df[la_contnd_df['Experimental_Condition'] == 'ee1xMBS']
# sns.scatterplot(x='Average Angle (deg)', y='lifeact_fluor_normed_sub_mean_corr', color='grey', 
#                 data=la_contnd_df[la_contnd_df['Experimental_Condition'] == 'ee1xMBS'], 
#                 s=5, linewidth=0, alpha=0.5, ax=ax, legend=False, zorder=1)
ax.errorbar(x=data['Average Angle (deg)'], y=data['lifeact_fluor_normed_sub_mean_corr'], 
            yerr=data['lifeact_fluor_normed_sub_sem_corr'], color='grey', marker='.', 
            ms=1, lw=0, elinewidth=1, ecolor='lightgrey', alpha=1, zorder=1)

ax.axhline(z_plat_corr, c='grey', ls='-', zorder=0)
# ax.axhline(z_plat, c='grey', ls='-', zorder=0)
ax.axvline(180, c='grey', ls='-', zorder=0)
    
# ax.set_ylim(0, 2)
# ax.set_yticks(np.linspace(0, 2, 21), minor=True)
ax.set_xlim(0, 200)
ax.set_xticks([0, 50, 100, 150, 200], minor=True)
ax.set_xlabel('Contact Angle (deg)')
ax.set_ylabel('Norm. Contact Fluor. (N.D.)')
# save the resulting plot:
# fig.legend()
fig.savefig(Path.joinpath(zcorr_plots_dir, r'angle_vs_fluor_lagfp_w_model_corr_%s_noYlim.tif'%fig_of_interest), dpi=dpi_LineArt, bbox_inches='tight')
plt.close(fig)

### also the LifeAct-GFP data alone:
fig, ax = plt.subplots(figsize=(width_1col*0.7, width_1col*0.7)) # set up the figure. 

# data = la_contnd_df[la_contnd_df['Experimental_Condition'] == 'ee1xMBS']
sns.scatterplot(x='Average Angle (deg)', y='lifeact_fluor_normed_sub_mean_corr', hue='Source_File', 
                data=data, 
                s=3, linewidth=0, alpha=0.5, ax=ax, legend=False, zorder=2)
ax.errorbar(x=data['Average Angle (deg)'], y=data['lifeact_fluor_normed_sub_mean_corr'], 
            yerr=data['lifeact_fluor_normed_sub_sem_corr'], color='grey', marker='.', 
            ms=0, lw=0, elinewidth=0.5, ecolor='lightgrey', alpha=1, zorder=1)
ax.plot(np.rad2deg(theor_ang)*2, theor_relten, c='k', ls='--', zorder=0) # the "*2" is because each cell in the cellpair has an angle associated with it's spherical cap geometry

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
fig.savefig(Path.joinpath(zcorr_plots_dir, r'angle_vs_fluor_lagfp_corr_noYlim.tif'), dpi=dpi_LineArt, bbox_inches='tight')
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
fig.savefig(Path.joinpath(zcorr_plots_dir, r'contnd_vs_fluor_lagfp_corr_noYlim.tif'), dpi=dpi_LineArt, bbox_inches='tight')
plt.close(fig)



# From Youssef et al 2011 (which references Sivasankar, Gumbiner, Leckband 2001)
# Cells overexpressing cadherins result in cadherin amounts of 20,000 - 250,000
# per cell (what is the density of this?)

# From Sivasankar, Brieher, Lavrik, Gumbiner, Leckband 1999:
# cadherin binding energy is 4-48 x 10-21 Joules per cadherin bond
# Same paper indicates that cadherin densities are 2000 - 4000 cadherins/um**2
# in artificial cadherin monolayers

# This translates to an energy density of 
# 8,000E-21 J/um**2 = 8E-18 J/um**2
# to 
# 192,000E-21 J/um**2 = 1.92E-16J/um**2

# compare this with the David et al 2014 estimates of cortical tension:
# beta ~ 0.86E-16J/um**2 = 8.6E-17J/um**2

# This corresponds to a value of 
# 8E-18 / 8.6E-17 =~ 0.093 (around 10% of cortical tension)
# to 
# 1.92E-16 / 8.6E-17 = 2.233 (around 223% of cortical tension)

# do these cadherin energy density values correspond to a cadherin ring or just
# a cadherin surface?




# exit()
## The above 'exit' is just a convenient stopping point when developing the 
## script since everything from here onwards takes a long time.



print('\n\nStarting simulation of low adhesion / uncoupled cytoskeletal networks.\n\n')
## Random contact actin levels at the low plateau: 
z_cc_func = z_rand
# z_cc_func_corr_args = (z_plat_low_corr_mean * z_cm, z_plat_low_corr_std * z_cm, data_acquisition_rate, acquisition_duration, z_init, 42)#42) # 42 = the random_state
z_cc_func_uncorr_args = (z_plat_low_uncorr_mean * z_cm, z_plat_low_uncorr_std * z_cm, data_acquisition_rate, acquisition_duration, z_init, 42)#42) # 42 = the random_state

## Non-contact actin remains constant:
z_cm_func_const = z_const
z_cm_func_args_const = (1,)


## The end of the integration interval:
t_end_rand = 1800

## Solving the equilibrium case: 
params = [Gamma_null, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_null]
Gamma, strain_lim_actin, t_Apt5, EA, eta_A, strain_lim_nonactin, t_Npt5, EN, eta_N, E_r = params

t, t_end_rand = t_init, t_end_rand # Set the range over which to solve. 
Scc0_init, Scm0_init, SNcc0_init, SNcm0_init = get_surf_init(R_s)
syst = [theta_init, Scc0_init, Scm0_init, SNcc0_init, SNcm0_init]
consts = (R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, EA, eta_A, EN, eta_N, eta_m_real, Gamma, E_r, z_cc_func, z_cc_func_uncorr_args, z_cm_func, z_cm_func_args)

## Note that we have to set calcium==True here because calcium==False only 
## works for nonzero values of Gamma. 
print('Starting soln_eq...')
# soln_eq = VESM((t, t_end_rand), syst, consts, solve_ivp_method, solve_ivp_max_step, solve_ivp_atol, solve_ivp_rtol, strain_lim_event_stol, calcium=False)
soln_eq = VESM((t, t_end_rand), syst, consts, solve_ivp_method, solve_ivp_max_step, solve_ivp_atol, solve_ivp_rtol, strain_lim_event_stol, calcium=True)
print('Done with soln_eq.')



### Want to get the equilibrium angle for adh_energy_total_noCalcium_M and use 
### that as the initial system:
Gamma_list = [adh_energy_total_noCalcium_M, adh_energy_total_noCalcium_L, adh_energy_total_noCalcium_H]
# Gamma_list = [adh_energy_total_yesCalcium_M, adh_energy_total_yesCalcium_L, adh_energy_total_yesCalcium_H]
# Gamma_list = [adh_energy_total_yesCalcium_M] 

## Solving the viscoelastic case: 
## Xenopus-like solutions for overlapping onto lifeact data:
# fig7G_params = [[G, strain_lim_intermfil, halflife_intermfil_M, E_multinetwork_M, eta_multinetwork, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example] for G in Gamma_list] # Xenopus multinetwork medium
fig7G_params = [[G, strain_lim_intermfil, halflife_intermfil_M, E_multinetwork_M, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example] for G in Gamma_list] # Xenopus multinetwork medium




fig7G_soln_list = list()

for params in fig7G_params:

    Gamma, strain_lim_actin, t_Apt5, EA, eta_A, strain_lim_nonactin, t_Npt5, EN, eta_N, E_r = params

    t, t_end_rand = t_init, t_end_rand # Set the range over which to solve. 
    consts = (R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, EA, eta_A, EN, eta_N, eta_m_real, Gamma, E_r, z_cc_func, z_cc_func_uncorr_args, z_cm_func_const, z_cm_func_args_const)
    
    print('\n\nStarting soln...')
    print(consts)
    # soln = VESM((t, t_end_rand), syst, consts, solve_ivp_method, solve_ivp_max_step, solve_ivp_atol, solve_ivp_rtol, strain_lim_event_stol, calcium=False)
    soln = VESM((t, t_end_rand), syst, consts, solve_ivp_method, solve_ivp_max_step, solve_ivp_atol, solve_ivp_rtol, strain_lim_event_stol, calcium=True)
    print('Done with soln...')
    fig7G_soln_list.append(soln)



## Solving the z_rand case takes a really long time, so save the solutions
## for convenience:
with open(Path.joinpath(out_dir, 'fig7G_soln_list.pkl'), 'wb') as f:    
    pickle.dump(fig7G_soln_list, f)

## May as well save the z_decay situation solutions as well:
with open(Path.joinpath(out_dir, 'soln_space_fig_params_uncorr.pkl'), 'wb') as f:    
    pickle.dump(soln_space_fig_params_uncorr, f)

with open(Path.joinpath(out_dir, 'soln_space_fig_params_corr.pkl'), 'wb') as f:    
    pickle.dump(soln_space_fig_params_corr, f)


## To open the pickled files:
# with open(Path.joinpath(out_dir, 'fig7G_soln_list.pkl'), 'wb') as f:    
#     pickle.load(f)

# with open(Path.joinpath(out_dir, 'soln_space_fig_params.pkl'), 'wb') as f:    
#     pickle.dump(soln_space_fig_params, f)
    
    
    
    
    
    
    
    

## Plot the results: 
# i = 0
time_eq, theta_eq = soln_eq.t, soln_eq.y[0,:]
time = fig7G_soln_list[0].t
z_cc_eq, dzcc_dt_eq = z_cc_func(time_eq, *z_cc_func_uncorr_args)
z_cm, dzcm_dt = z_cm_func_const(time, *z_cm_func_args_const)




fig, ax = plt.subplots(figsize=(width_1col*0.7, width_1col*0.7))
ax.plot(time_eq/60, 2*np.rad2deg(theta_eq), lw=0.7, c='k', alpha=1.0, ls=':')
# ax.axhline(np.mean(2*np.rad2deg(theta_eq)), lw=0.7, c='k', alpha=1.0, ls='--')
# ax.axhline(2*np.rad2deg(theta_Gamma), lw=0.7, c='tab:orange', alpha=1.0, ls='--')
# ax.plot(time, 2*np.rad2deg(theta), lw=0.7, c='r', alpha=1.0)
# ax.plot(time/60, 2*np.rad2deg(theta))
# for soln in fig7G_soln_list:
#     ax.plot(soln.t/60, 2*np.rad2deg(soln.y[0]))
# ax.plot(time/60, 2*np.rad2deg(theta), c='tab:blue', lw=1)
for soln in fig7G_soln_list:
    ax.plot(soln.t/60, 2*np.rad2deg(soln.y[0]), lw=1, alpha=1.0)

# ax.plot(time_const/60, 2*np.rad2deg(theta_const), c='tab:red', lw=1, ls='-')

axb = ax.twinx()
axb.plot(time_eq/60, z_cc_eq/z_cm, c='grey', lw=1)
# axb.plot(time_const/60, z_cc_const/z_cm_const, c='grey', lw=1)
# axb.plot(time_const/60, z_cc_const, c='b', lw=1)
# axb.plot(time_const/60, [z_cm_const]*len(z_cc_const), c='g', lw=1)
# axb.plot(time/60, z_cm, c='grey', lw=1)
# axb.plot(time/60, z_cc, c='k', lw=1, ls='--')
axb.axhline(1, c='lightgrey', lw=0.7, zorder=0)
axb.tick_params(axis='y', colors='grey')
axb.set_ylim(0,2)

ax.set_zorder(axb.get_zorder()+1) # this puts ax in front of axb
ax.patch.set_visible(False) # this sets the background of ax
# ax.set_xlim(-5/60, 5)#1800/60)
# ax.set_ylim(0, 80)#200)

# plt.show()
fig.savefig(Path.joinpath(man_plots_dir, 'low_adh_sim_over_time.tif'), dpi=dpi_LineArt, bbox_inches='tight')
plt.close('all')





data = la_contnd_df[la_contnd_df['Experimental_Condition'] == 'ee1xDB']

fig, ax = plt.subplots(figsize=(width_1col*0.7, width_1col*0.7))
# sns.scatterplot(x='Average Angle (deg)', y='lifeact_fluor_normed_mean', color='c', 
#                 data=la_contnd_df[la_contnd_df['Experimental_Condition'] == 'ee1xMBS'], 
#                 s=5, linewidth=0, alpha=0.5, ax=ax, legend=False, zorder=10)
# sns.scatterplot(x='Average Angle (deg)', y='lifeact_fluor_normed_mean', color='grey', 
#                 data=la_contnd_df[la_contnd_df['Experimental_Condition'] == 'ee1xDB'], 
#                 s=5, linewidth=0, alpha=1, ax=ax, legend=False, zorder=0)

# sns.scatterplot(x='Average Angle (deg)', y='lifeact_fluor_normed_sub_mean_corr', color='grey', 
#                 data=la_contnd_df[la_contnd_df['Experimental_Condition'] == 'ee1xDB'], 
#                 s=5, linewidth=0, alpha=1, ax=ax, legend=False, zorder=0)
ax.errorbar(x=data['Average Angle (deg)'], y=data['lifeact_fluor_normed_sub_mean_corr'], 
            yerr=data['lifeact_fluor_normed_sub_sem_corr'], color='grey', marker='.',
            ms=1, lw=0, elinewidth=1, ecolor='lightgrey', alpha=1, zorder=0)


# sns.scatterplot(x='Average Angle (deg)', y='lifeact_fluor_normed_mean', hue='Source_File', 
#                 data=la_contnd_df[la_contnd_df['Experimental_Condition'] == 'ee1xDB'], 
#                 s=5, linewidth=0, alpha=0.5, ax=ax, legend=False, zorder=10)

for soln in fig7G_soln_list:
    z_cc, dzcc_dt = z_cc_func(soln.t, *z_cc_func_uncorr_args)
    z_cm, dzcm_dt = z_cm_func(soln.t, *z_cm_func_args)
    ax.scatter(2*np.rad2deg(soln.y[0]), z_cc/z_cm, marker='.', alpha=0.4, s=1)
    # ax.draw
# ax.scatter( 2*np.rad2deg(soln_const.y[0]), z_cc_func(soln_const.t, *z_cc_func_args)/z_cm_func_const(soln_const.t, *z_cm_func_args_const), c='r', marker='.', alpha=0.4, s=1)

ax.scatter( 2*np.rad2deg(soln_eq.y[0]), z_cc_eq/z_cm, c='k', marker='.', s=1)
ax.axvline(180, c='grey', lw=1, zorder=10)
ax.axvline(41.1, c='tab:orange', lw=1, ls='--', zorder=0) # the simulated cells from Fig 4E,F have a contact angle of about 41.1 degrees
ax.set_xlim(0,200)
ax.set_ylim(0,2)
ax.set_ylabel('')
ax.set_xlabel('')
plt.tight_layout()

fig.savefig(Path.joinpath(man_plots_dir, 'low_adh_sim.tif'), dpi=dpi_LineArt, bbox_inches='tight')
plt.close('all')



## When you are done, let me know:
print('Done.')







## To address a reviewers concerns:
## They bring up that whether the data is plotted as a disc (circular surface 
## area), ring (circumference), or radius, this will change the shape of the 
## adhesion kinetics curves.
## I will use the contact angles from the solutions in fig3B to plot diameter,
## and surface area of the contact, normalized to initial cell radius and 
## initial non-contact surface area, respectively. I picked fig3B because it
## has a nice assortment of curve shapes.

fig3B_soln_locs = np.where([soln.fig_id == 'fig3B' for soln in soln_space_fig_params_uncorr])
fig3B_solns = [soln_space_fig_params_uncorr[i] for i in np.ravel(fig3B_soln_locs)]


fig, ax = plt.subplots()
axb = ax.twinx()
solns_of_interest = [0, 4, -1]

xlim_upper = 1800

line_colors = list(mcolors.TABLEAU_COLORS)
color_count = 0
for i in solns_of_interest:
    soln = fig3B_solns[i]
    ## there are actually too many solutions here, which makes it difficult to
    ## see individual lines, so we will instead take the first, middle and last
    ## solutions as examples:
    SCap = SpherCapGeom(soln.y[0], R_s)
    
    cont_diam = SCap.r_c * 2 # the diameter
    cont_disc = SCap.Scc # the surface are of the contact
    
    # R_s # the initial cell radius
    SCap_init = SpherCapGeom(0, R_s) # create an initial spherical cell
    surf_init = SCap_init.Scm # the initial surface area of the cell
    
    color = line_colors[color_count]
    # time = soln.t[soln.t < xlim_upper]
    # norm_cont_diam = (cont_diam / (R_s*2))[soln.t < xlim_upper]
    # norm_cont_surf = (cont_disc / surf_init)[soln.t < xlim_upper]
    time = soln.t
    norm_cont_diam = (cont_diam / (R_s*2))
    norm_cont_surf = (cont_disc / surf_init)
    ax.plot(time, norm_cont_diam, ls='-', c=color, label='Norm. Cont. Diam.')
    axb.plot(time, norm_cont_surf, ls='--', c=color, label='Norm. Cont. Surf.')
    
    leg_lines = [mlines.Line2D([], [], color='k', marker='', linestyle='-', label='Norm. Cont. Diam'), 
                 mlines.Line2D([], [], color='k', marker='', linestyle='--', label='Norm. Cont. Surf.')]
    fig.legend(handles=leg_lines, ncol=2, loc='upper center')

    color_count += 1
ax.set_ylabel('Norm. Cont. Diam.')
axb.set_ylabel('Norm. Cont. Surf.')
ax.set_xlabel('Time (seconds)')
plt.xlim(-18, xmax=xlim_upper)

fig.savefig(Path.joinpath(out_dir, 'ContDiam_vs_Disc.tif'), dpi=dpi_LineArt, bbox_inches='tight')
    






###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###
### Rough stuff:
###

## Want to compare the original solutions to figSY, figSL, and figSN:
# plt.close('all')
# figs_of_interest = ['fig3D', 'figSY', 'figSL', 'figSN']
# fig, ax = plt.subplots(figsize=(width_1col*0.7, width_1col*0.7))
# cmap_count = 0 # each figure starts with a fresh colormap 
# for fig_id in figs_of_interest:

#     for params_of_interest in params_w_fig_id[fig_id][::-1][:-1]: #reverse the list and take all but the last entry of that reversed list (ie. take all but the first entry in the list and reverse their order.)
#         # print(params_of_interest) ## some debugging code
#         for i, soln in enumerate(soln_space_fig_params):
      
#             if np.all(all_fig_params[i] == params_of_interest) and (soln_space_fig_ids[i] == fig_id):
#                 time = soln.t
#                 theta, Scc0, Scm0, SNcc0, SNcm0 = soln.y
                
#                 ax.plot(time, 2*np.rad2deg(theta), lw=1, c=cmap(cmap_count))#, ls='--')
                        

#     params_of_interest = params_w_fig_id[fig_id][0]
#     for i, soln in enumerate(soln_space_fig_params):
#         if np.all(all_fig_params[i] == params_of_interest) and (soln_space_fig_ids[i] == fig_id):
#             time = soln.t
#             theta, Scc0, Scm0, SNcc0, SNcm0, = soln.y
#             z_cc, dzcc_dt = z_cc_func(time, *z_cc_func_args)
#             ax.plot(time, 2*np.rad2deg(theta), lw=1, c='k', alpha=1.0, ls=':')
         
#     ax.set_xlim(-10, 1800)
#     ax.set_xticks([0,600,1200,1800])#,2400])
#     ax.set_xticklabels(['0','600','1200','1800'])#,'2400'])
#     ax.set_ylabel('Contact Angle (deg)')
#     ax.set_xlabel('Time (s)')
            
#     time_zcc = np.linspace(0, 3600, 36001)
#     z_cc, dzcc_dt = z_cc_func(time, *z_cc_func_args)
#     z_cm, dzcm_dt = z_cm_func(time, *z_cm_func_args)
#     axb = ax.twinx()
#     axb.axvline(t_half, c='grey', lw=1, ls=':', zorder=7)
#     axb.plot(time, z_cc/z_cm, c='grey', lw=1, zorder=8)
#     axb.set_ylabel('Rel. Contact Actin', c='grey', rotation=270, va='bottom', ha='center')
#     axb.set_ylim(0, 1)
#     # axb.set_xlim(0,200)
#     axb.tick_params(axis='y', colors='grey')
            
#     ax.set_zorder(axb.get_zorder()+1) # this puts ax in front of axb
#     ax.patch.set_visible(False) # this sets the background of ax
    
#     cmap_count += 1 # each line that gets drawn gets a new color from the color map

    
# leg_lines = [mlines.Line2D([], [], color=cmap(i), marker='', linestyle='-', label=figs_of_interest[i]) for i in range(cmap_count)]
#              # mlines.Line2D([], [], color='k', marker='', linestyle='--', label='Norm. Cont. Surf.')]
# fig.legend(handles=leg_lines, ncol=4, loc='upper center', fontsize='x-small')

# # plt.draw()
# # plt.show()

# fig.savefig(Path.joinpath(man_plots_dir, r"fig3D_SY_SL_SN_time_course.tif"), dpi=dpi_LineArt, bbox_inches='tight')
# plt.close('all')






# ## This plots solutions to the VESM with the adhesion modes color coded
# ## (e.g. rolling adhesion occurs initially and is green):

# ## Want to plot the z_decay simulation over time but color-coded according to
# ## the adh_mode:
# z_rand_Gamma_high = fig7G_soln_list[-1]
# # col_names = ['t', 'y', 'color']
# z_rand_Gamma_high_df = pd.DataFrame({'time': z_rand_Gamma_high.t, 
#                                       'angle': z_rand_Gamma_high.y[0], 
#                                       'color': z_rand_Gamma_high.color})


# fig, ax = plt.subplots()#figsize=(width_1col*0.7, width_1col*0.7))
# for c,grp in z_rand_Gamma_high_df.groupby('color'):
#     ax.scatter(grp['time'], 2*np.rad2deg(grp['angle']), s=0.5, alpha=0.2, c=c)
# # ax.plot(time_eq/60, 2*np.rad2deg(theta_eq), lw=1, alpha=0.5, c='m')

# axb = ax.twinx()
# axb.plot(time_eq, z_cc_eq/z_cm, c='m', lw=1)
# axb.axhline(1, c='m', alpha=0.3, lw=0.7, zorder=0)
# axb.tick_params(axis='y', colors='m')
# axb.set_ylim(0,2)

# ax.set_zorder(axb.get_zorder()+1) # this puts ax in front of axb
# ax.patch.set_visible(False) # this sets the background of ax
# plt.close()




## This plots the rolling and peeling condition angular velocities over time:
##
# plt.plot(soln.t, theta_dot_rolling_condition(soln.t, soln.y, *args), lw=1)
# plt.plot(soln.t, theta_dot_peeling_condition(soln.t, soln.y, *args), lw=1)
# plt.axhline(0, c='k', ls=':', lw=1)
# plt.close()
##


# print('\nStarting z_cm random test:')
# ## Can we find solutions to the VESM when z_cm is allowed to change randomly?
# ### Want to get the equilibrium angle for adh_energy_total_noCalcium_M and use 
# ### that as the initial system:
# # Gamma_list = [adh_energy_total_noCalcium_M, adh_energy_total_noCalcium_L, adh_energy_total_noCalcium_H]
# Gamma_list = [adh_energy_total_noCalcium_M] 

# ## Solving the viscoelastic case: 
# ## Xenopus-like solutions for overlapping onto lifeact data:
# # fig7G_params = [[G, strain_lim_intermfil, halflife_intermfil_M, E_multinetwork_M, eta_multinetwork, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example] for G in Gamma_list] # Xenopus multinetwork medium
# fig8G_params = [[G, strain_lim_intermfil, halflife_intermfil_M, E_multinetwork_M, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example] for G in Gamma_list] # Xenopus multinetwork medium



# z_cm_func_eq_zcc = z_cc_func
# z_cm_func_args_eq_zcc = z_cc_func_args

# fig8G_soln_list = list()

# for params in fig8G_params:

#     Gamma, strain_lim_actin, t_Apt5, EA, eta_A, strain_lim_nonactin, t_Npt5, EN, eta_N, E_r = params

#     t, t_end_rand = t_init, t_end_rand # Set the range over which to solve. 
#     consts = (R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, EA, eta_A, EN, eta_N, eta_m_real, Gamma, E_r, z_cc_func, z_cc_func_args, z_cm_func_eq_zcc, z_cm_func_args_eq_zcc)
    
#     print('\n\nStarting soln...')
#     print(consts)
#     soln = VESM((t, t_end_rand), syst, consts, solve_ivp_method, solve_ivp_max_step, solve_ivp_atol, solve_ivp_rtol, strain_lim_event_stol, calcium=False)
#     print('Done with soln...')
#     fig8G_soln_list.append(soln)



# ## Solving the z_rand case takes a really long time, so save the solutions
# ## for convenience:
# with open(Path.joinpath(out_dir, 'fig8G_soln_list.pkl'), 'wb') as f:    
#     pickle.dump(fig8G_soln_list, f)



# time_eq, theta_eq = soln_eq.t, soln_eq.y[0,:]
# time = fig8G_soln_list[0].t
# z_cc_eq, dzcc_dt_eq = z_cc_func(time_eq, *z_cc_func_args)
# z_cm_eq, dzcm_dt = z_cm_func_eq_zcc(time_eq, *z_cm_func_args_eq_zcc)


# fig, ax = plt.subplots(figsize=(width_1col*0.7, width_1col*0.7))
# ax.plot(time_eq/60, 2*np.rad2deg(theta_eq), lw=0.7, c='k', alpha=1.0, ls=':')
# # ax.axhline(np.mean(2*np.rad2deg(theta_eq)), lw=0.7, c='k', alpha=1.0, ls='--')
# # ax.axhline(2*np.rad2deg(theta_Gamma), lw=0.7, c='tab:orange', alpha=1.0, ls='--')
# # ax.plot(time, 2*np.rad2deg(theta), lw=0.7, c='r', alpha=1.0)
# # ax.plot(time/60, 2*np.rad2deg(theta))
# # for soln in fig8G_soln_list:
# #     ax.plot(soln.t/60, 2*np.rad2deg(soln.y[0]))
# # ax.plot(time/60, 2*np.rad2deg(theta), c='tab:blue', lw=1)
# for soln in fig8G_soln_list:
#     ax.plot(soln.t/60, 2*np.rad2deg(soln.y[0]), lw=1, alpha=1.0)

# # ax.plot(time_const/60, 2*np.rad2deg(theta_const), c='tab:red', lw=1, ls='-')

# # axb = ax.twinx()
# # axb.plot(time_eq/60, z_cc_eq/z_cm, c='grey', lw=1)
# # axb.plot(time_const/60, z_cc_const/z_cm_const, c='grey', lw=1)
# # axb.plot(time_const/60, z_cc_const, c='b', lw=1)
# # axb.plot(time_const/60, [z_cm_const]*len(z_cc_const), c='g', lw=1)
# # axb.plot(time/60, z_cm, c='grey', lw=1)
# # axb.plot(time/60, z_cc, c='k', lw=1, ls='--')
# # axb.axhline(1, c='lightgrey', lw=0.7, zorder=0)
# # axb.tick_params(axis='y', colors='grey')
# # axb.set_ylim(0,2)

# ax.set_zorder(axb.get_zorder()+1) # this puts ax in front of axb
# ax.patch.set_visible(False) # this sets the background of ax
# # ax.set_xlim(-5/60, 5)#1800/60)
# # ax.set_ylim(0, 80)#200)

# # plt.show()
# fig.savefig(Path.joinpath(man_plots_dir, 'low_adh_sim_over_time_zcm_rand_8G.tif'), dpi=dpi_LineArt, bbox_inches='tight')
# plt.close('all')


# fig, ax = plt.subplots(figsize=(width_1col*0.7, width_1col*0.7))
# # sns.scatterplot(x='Average Angle (deg)', y='lifeact_fluor_normed_mean', color='c', 
# #                 data=la_contnd_df[la_contnd_df['Experimental_Condition'] == 'ee1xMBS'], 
# #                 s=5, linewidth=0, alpha=0.5, ax=ax, legend=False, zorder=10)
# # sns.scatterplot(x='Average Angle (deg)', y='lifeact_fluor_normed_mean', color='grey', 
# #                 data=la_contnd_df[la_contnd_df['Experimental_Condition'] == 'ee1xDB'], 
# #                 s=5, linewidth=0, alpha=1, ax=ax, legend=False, zorder=0)

# # sns.scatterplot(x='Average Angle (deg)', y='lifeact_fluor_normed_sub_mean_corr', color='grey', 
# #                 data=la_contnd_df[la_contnd_df['Experimental_Condition'] == 'ee1xDB'], 
# #                 s=5, linewidth=0, alpha=1, ax=ax, legend=False, zorder=0)
# ax.errorbar(x=data['Average Angle (deg)'], y=data['lifeact_fluor_normed_sub_mean_corr'], 
#             yerr=data['lifeact_fluor_normed_sub_sem_corr'], color='grey', marker='.',
#             ms=1, lw=0, elinewidth=1, ecolor='lightgrey', alpha=1, zorder=0)


# # sns.scatterplot(x='Average Angle (deg)', y='lifeact_fluor_normed_mean', hue='Source_File', 
# #                 data=la_contnd_df[la_contnd_df['Experimental_Condition'] == 'ee1xDB'], 
# #                 s=5, linewidth=0, alpha=0.5, ax=ax, legend=False, zorder=10)

# for soln in fig8G_soln_list:
#     z_cc, dzcc_dt = z_cc_func(soln.t, *z_cc_func_args)
#     z_cm, dzcm_dt = z_cm_func_eq_zcc(soln.t, *z_cm_func_args_eq_zcc)
#     ax.scatter(2*np.rad2deg(soln.y[0]), z_cm, marker='.', alpha=0.4, s=1)
#     # ax.draw
# # ax.scatter( 2*np.rad2deg(soln_const.y[0]), z_cc_func(soln_const.t, *z_cc_func_args)/z_cm_func_const(soln_const.t, *z_cm_func_args_const), c='r', marker='.', alpha=0.4, s=1)
# # ax.scatter( 2*np.rad2deg(soln.y[0]), z_cc/z_cm, c='k', marker='.', s=1)
# ax.scatter( 2*np.rad2deg(soln_eq.y[0]), z_cm_eq, c='k', marker='.', s=1)
# ax.axvline(180, c='grey', lw=1, zorder=10)
# ax.axvline(41.1, c='tab:orange', lw=1, ls='--', zorder=0) # the simulated cells from Fig 4E,F have a contact angle of about 41.1 degrees
# ax.set_xlim(0,200)
# ax.set_ylim(0,2)
# ax.set_ylabel('')
# ax.set_xlabel('')
# plt.tight_layout()

# fig.savefig(Path.joinpath(man_plots_dir, 'low_adh_sim_8G.tif'), dpi=dpi_LineArt, bbox_inches='tight')
# plt.close('all')







# fig9G_params = [[G, strain_lim_intermfil, halflife_intermfil_M, E_multinetwork_M, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example] for G in Gamma_list] # Xenopus multinetwork medium



# z_cm_func_eq_zcc = z_cc_func
# z_cm_func_args_eq_zcc = z_cc_func_args

# fig9G_soln_list = list()

# for params in fig9G_params:

#     Gamma, strain_lim_actin, t_Apt5, EA, eta_A, strain_lim_nonactin, t_Npt5, EN, eta_N, E_r = params

#     t, t_end_rand = t_init, t_end_rand # Set the range over which to solve. 
#     consts = (R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, EA, eta_A, EN, eta_N, eta_m_real, Gamma, E_r, z_cc_func, z_cc_func_args, z_cm_func_eq_zcc, z_cm_func_args_eq_zcc)
    
#     print('\n\nStarting soln...')
#     print(consts)
#     soln = VESM((t, t_end_rand), syst, consts, solve_ivp_method, solve_ivp_max_step, solve_ivp_atol, solve_ivp_rtol, strain_lim_event_stol, calcium=True)
#     print('Done with soln...')
#     fig9G_soln_list.append(soln)



# ## Solving the z_rand case takes a really long time, so save the solutions
# ## for convenience:
# with open(Path.joinpath(out_dir, 'fig9G_soln_list.pkl'), 'wb') as f:    
#     pickle.dump(fig9G_soln_list, f)



# time_eq, theta_eq = soln_eq.t, soln_eq.y[0,:]
# time = fig9G_soln_list[0].t
# # z_cc_eq, dzcc_dt_eq = z_cc_func(time_eq, *z_cc_func_args)
# # z_cm, dzcm_dt = z_cc_func(time, *z_cm_func_args_const)


# fig, ax = plt.subplots(figsize=(width_1col*0.7, width_1col*0.7))
# ax.plot(time_eq/60, 2*np.rad2deg(theta_eq), lw=0.7, c='k', alpha=1.0, ls=':')
# # ax.axhline(np.mean(2*np.rad2deg(theta_eq)), lw=0.7, c='k', alpha=1.0, ls='--')
# # ax.axhline(2*np.rad2deg(theta_Gamma), lw=0.7, c='tab:orange', alpha=1.0, ls='--')
# # ax.plot(time, 2*np.rad2deg(theta), lw=0.7, c='r', alpha=1.0)
# # ax.plot(time/60, 2*np.rad2deg(theta))
# # for soln in fig9G_soln_list:
# #     ax.plot(soln.t/60, 2*np.rad2deg(soln.y[0]))
# # ax.plot(time/60, 2*np.rad2deg(theta), c='tab:blue', lw=1)
# for soln in fig9G_soln_list:
#     ax.plot(soln.t/60, 2*np.rad2deg(soln.y[0]), lw=1, alpha=1.0)

# # ax.plot(time_const/60, 2*np.rad2deg(theta_const), c='tab:red', lw=1, ls='-')

# # axb = ax.twinx()
# # axb.plot(time_eq/60, z_cc_eq/z_cm, c='grey', lw=1)
# # axb.plot(time_const/60, z_cc_const/z_cm_const, c='grey', lw=1)
# # axb.plot(time_const/60, z_cc_const, c='b', lw=1)
# # axb.plot(time_const/60, [z_cm_const]*len(z_cc_const), c='g', lw=1)
# # axb.plot(time/60, z_cm, c='grey', lw=1)
# # axb.plot(time/60, z_cc, c='k', lw=1, ls='--')
# # axb.axhline(1, c='lightgrey', lw=0.7, zorder=0)
# # axb.tick_params(axis='y', colors='grey')
# # axb.set_ylim(0,2)

# ax.set_zorder(axb.get_zorder()+1) # this puts ax in front of axb
# ax.patch.set_visible(False) # this sets the background of ax
# # ax.set_xlim(-5/60, 5)#1800/60)
# # ax.set_ylim(0, 80)#200)

# # plt.show()
# fig.savefig(Path.joinpath(man_plots_dir, 'low_adh_sim_over_time_zcm_rand_9G.tif'), dpi=dpi_LineArt, bbox_inches='tight')
# plt.close('all')



# fig, ax = plt.subplots(figsize=(width_1col*0.7, width_1col*0.7))
# # sns.scatterplot(x='Average Angle (deg)', y='lifeact_fluor_normed_mean', color='c', 
# #                 data=la_contnd_df[la_contnd_df['Experimental_Condition'] == 'ee1xMBS'], 
# #                 s=5, linewidth=0, alpha=0.5, ax=ax, legend=False, zorder=10)
# # sns.scatterplot(x='Average Angle (deg)', y='lifeact_fluor_normed_mean', color='grey', 
# #                 data=la_contnd_df[la_contnd_df['Experimental_Condition'] == 'ee1xDB'], 
# #                 s=5, linewidth=0, alpha=1, ax=ax, legend=False, zorder=0)

# # sns.scatterplot(x='Average Angle (deg)', y='lifeact_fluor_normed_sub_mean_corr', color='grey', 
# #                 data=la_contnd_df[la_contnd_df['Experimental_Condition'] == 'ee1xDB'], 
# #                 s=5, linewidth=0, alpha=1, ax=ax, legend=False, zorder=0)
# ax.errorbar(x=data['Average Angle (deg)'], y=data['lifeact_fluor_normed_sub_mean_corr'], 
#             yerr=data['lifeact_fluor_normed_sub_sem_corr'], color='grey', marker='.',
#             ms=1, lw=0, elinewidth=1, ecolor='lightgrey', alpha=1, zorder=0)


# for soln in fig9G_soln_list:
#     z_cc, dzcc_dt = z_cc_func(soln.t, *z_cc_func_args)
#     z_cm, dzcm_dt = z_cm_func_eq_zcc(soln.t, *z_cm_func_args_eq_zcc)
#     ax.scatter(2*np.rad2deg(soln.y[0]), z_cm, marker='.', alpha=0.4, s=1)
#     # ax.draw
# # ax.scatter( 2*np.rad2deg(soln_const.y[0]), z_cc_func(soln_const.t, *z_cc_func_args)/z_cm_func_const(soln_const.t, *z_cm_func_args_const), c='r', marker='.', alpha=0.4, s=1)
# # ax.scatter( 2*np.rad2deg(soln.y[0]), z_cc/z_cm, c='k', marker='.', s=1)
# ax.scatter( 2*np.rad2deg(soln_eq.y[0]), z_cm_eq, c='k', marker='.', s=1)
# ax.axvline(180, c='grey', lw=1, zorder=10)
# ax.axvline(41.1, c='tab:orange', lw=1, ls='--', zorder=0) # the simulated cells from Fig 4E,F have a contact angle of about 41.1 degrees
# ax.set_xlim(0,200)
# ax.set_ylim(0,2)
# ax.set_ylabel('')
# ax.set_xlabel('')
# plt.tight_layout()

# fig.savefig(Path.joinpath(man_plots_dir, 'low_adh_sim_9G.tif'), dpi=dpi_LineArt, bbox_inches='tight')
# plt.close('all')







### The following gets stuck switching between rolling and peeling adhesion for
### an unknown amount of time (at least an hour), probably due to the 
### discontinuity in the parameter z between the cell-cell surface and the 
### cell-medium surface (i.e. transition from z_cm to z_cc is not smooth): 
# fig10G_params = [[G, strain_lim_intermfil, halflife_intermfil_M, E_multinetwork_M, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example] for G in Gamma_list] # Xenopus multinetwork medium

# fig10G_soln_list = list()

# for params in fig10G_params:

#     Gamma, strain_lim_actin, t_Apt5, EA, eta_A, strain_lim_nonactin, t_Npt5, EN, eta_N, E_r = params

#     t, t_end_rand = t_init, t_end_rand # Set the range over which to solve. 
#     consts = (R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, EA, eta_A, EN, eta_N, eta_m_real, Gamma, E_r, z_cc_func, z_cc_func_args, z_cm_func_const, z_cm_func_args_const)
    
#     print('\n\nStarting soln...')
#     print(consts)
#     soln = VESM((t, t_end_rand), syst, consts, solve_ivp_method, solve_ivp_max_step, solve_ivp_atol, solve_ivp_rtol, strain_lim_event_stol, calcium=False)
#     ## for some reason the solver gets stuck in an endless mode of switching
#     ## back and forth between rolling and peeling adhesion at about 
#     ## t = 423.844708149083
#     print('Done with soln...')
#     fig10G_soln_list.append(soln)



# ## Solving the z_rand case takes a really long time, so save the solutions
# ## for convenience:
# with open(Path.joinpath(out_dir, 'fig10G_soln_list.pkl'), 'wb') as f:    
#     pickle.dump(fig10G_soln_list, f)


    
# time_eq, theta_eq = soln_eq.t, soln_eq.y[0,:]
# time = fig10G_soln_list[0].t
# # z_cc_eq, dzcc_dt_eq = z_cc_func(time_eq, *z_cc_func_args)
# # z_cm, dzcm_dt = z_cc_func(time, *z_cm_func_args_const)


# fig, ax = plt.subplots(figsize=(width_1col*0.7, width_1col*0.7))
# ax.plot(time_eq/60, 2*np.rad2deg(theta_eq), lw=0.7, c='k', alpha=1.0, ls=':')
# # ax.axhline(np.mean(2*np.rad2deg(theta_eq)), lw=0.7, c='k', alpha=1.0, ls='--')
# # ax.axhline(2*np.rad2deg(theta_Gamma), lw=0.7, c='tab:orange', alpha=1.0, ls='--')
# # ax.plot(time, 2*np.rad2deg(theta), lw=0.7, c='r', alpha=1.0)
# # ax.plot(time/60, 2*np.rad2deg(theta))
# # for soln in fig10G_soln_list:
# #     ax.plot(soln.t/60, 2*np.rad2deg(soln.y[0]))
# # ax.plot(time/60, 2*np.rad2deg(theta), c='tab:blue', lw=1)
# for soln in fig10G_soln_list:
#     ax.plot(soln.t/60, 2*np.rad2deg(soln.y[0]), lw=1, alpha=1.0)

# # ax.plot(time_const/60, 2*np.rad2deg(theta_const), c='tab:red', lw=1, ls='-')

# # axb = ax.twinx()
# # axb.plot(time_eq/60, z_cc_eq/z_cm, c='grey', lw=1)
# # axb.plot(time_const/60, z_cc_const/z_cm_const, c='grey', lw=1)
# # axb.plot(time_const/60, z_cc_const, c='b', lw=1)
# # axb.plot(time_const/60, [z_cm_const]*len(z_cc_const), c='g', lw=1)
# # axb.plot(time/60, z_cm, c='grey', lw=1)
# # axb.plot(time/60, z_cc, c='k', lw=1, ls='--')
# # axb.axhline(1, c='lightgrey', lw=0.7, zorder=0)
# # axb.tick_params(axis='y', colors='grey')
# # axb.set_ylim(0,2)

# ax.set_zorder(axb.get_zorder()+1) # this puts ax in front of axb
# ax.patch.set_visible(False) # this sets the background of ax
# # ax.set_xlim(-5/60, 5)#1800/60)
# # ax.set_ylim(0, 80)#200)

# # plt.show()
# fig.savefig(Path.joinpath(man_plots_dir, 'low_adh_sim_over_time_zcm_rand_10G.tif'), dpi=dpi_LineArt, bbox_inches='tight')
# plt.close('all')









