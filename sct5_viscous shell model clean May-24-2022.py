# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 17:34:55 2020

@author: Serge
"""







from pathlib import Path
import numpy as np
import pandas as pd
# import sympy as sp
import seaborn as sns
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size':24})
import matplotlib.lines as mlines
import matplotlib.cm # used to get a colormap
# from mpl_toolkits.mplot3d import Axes3D


from scipy.integrate import solve_ivp
# from scipy.integrate import solve_bvp
from scipy.stats import distributions as distrib
from scipy import interpolate

# from multiprocessing import Process

import time as tm

# Needed for multipdf function to create pdf files:
from matplotlib.backends.backend_pdf import PdfPages


### A lot of figures are going to be opened before they are saved in to a
### multi-page pdf, so increase the max figure open warning:
plt.rcParams.update({'figure.max_open_warning':300})
plt.rcParams.update({'font.size':10})




def cm2inch(num):
    return num/2.54




def multipdf(filename):
    """ This function saves a multi-page PDF of all open figures. """
    pdf = PdfPages(filename)
    figs = [plt.figure(f) for f in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pdf, format='pdf' , dpi=80)
    pdf.close()



# Define a function to find the order of magnitude of a number:
def order_of_mag(number):
    order_of_magnitude = np.floor(np.log10(number))
    return order_of_magnitude




### Set your output directory to be the same location as this script:
# Get the working directory of the script:
current_script_dir = Path(__file__).parent.absolute()
# Set output directory:
out_dir = Path.joinpath(current_script_dir, r"viscoelast_shell_model_out")




### Set some parameters for saving graphs:
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

cmap = matplotlib.cm.tab10





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















#############################################################
###
### Serge's stuff: Solving ODEs of contact angle change
###
##
#

# Note: Use multiprocessing to speed up identifying the solution space?



# Get the working directory of the script:
current_script_dir = Path(__file__).parent.absolute()

# Data can be found in the following directory:
lifeact_analysis_dir = Path.joinpath(current_script_dir, r"lifeact_analysis_figs_out")





### Set some parameters :
# atol = 1e-6 # used in solve_ivp
# rtol = 1e-3 # used in solve_ivp
# strain_lim_event_stol = 1e-3 # used in the events that split different usages of solve_ivp

t_init, t_end = 0, 14400 # endpoint is 4hrs of simulated time

dt = 1e-1 # small amount to move time forward by if theta = theta_eq
max_step = np.inf #5e-2 #1e-1
# t_end is repeated a lot later, but I kind of like how that makes it clear what it is being used for

theta_min = np.deg2rad(0.1) #0.1 degrees is the minimum angle, 
## assume that this angle is the floor and cells will not seperate, so according to the JKR theory of
## small adhesions between spheres, the contact will never get smaller than this. 

theta_max = np.deg2rad(179.9) 
## assume that this is the upper limit to how far cells can spread, and that
## any further spreading would cause cell death







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

### Assuming negligible inertial forces:






### The following are some functions used to describe how the density at the 
### contact relative to the non-contact surface of the actin cortex changes 
### over time. So far there are 3 functions: 
### 1) an exponential decay where any time point less than 0 has a density 
### equal to the initial value, and an exponential decay from the initial value
### to some plateau starting at 0. This function reflects the data of cells as
### they adhere to an upper adhesion plateau. 
### 2) a piecewise function built from random points generated from a normal
### distribution. This is similar to the data of cells sticking together in 
### calcium-free medium. 
### 3) a sinusoidal function that is perhaps a crude estimate of function (2),
### but it is a continuous function, and so may behave better during numerical
### integration.

# An exponential decay function (looks similar to the LifeAct data at the contact):

def z_decay(t, z_init, t_half, z_plat):
    tau = t_half / np.log(2)
    z = np.asarray((z_init - z_plat) * np.exp(-1 * t / tau) + z_plat)
    z[t < 0] = z_init
    return z


# def dzcont_dt_decay(t, z_init, t_half, z_plat):
#     # z = (z_init - z_plat) * np.exp(-1 * t * np.log(2) / t_half) + z_plat
#     dz_dt = (z_init - z_plat) * np.exp(-1 * t * np.log(2) / t_half) * -1 * np.log(2) / t_half
#     return dz_dt




# A piecewise linear function connecting randomly distributed points:
    
### def z_cont_rand
### If we randomly distribute fluorescence intensity values in a manner similar
### to that of our data, are variations in contact angle on a similar scale to
### our empirical observations?: 
# mean = 0.80
# std = 0.26
# # The above mean and standard deviation are approximately equal to that of the
# # lifeact fluorescence at the cell-cell contact relative to the cell-medium
# # interface from ectoderm-ectoderm cells cultured in calcium-free medium
# data_acquisition_rate = 1/30 # 1 data point acquired every 30 seconds
# acquisition_duration = t_end*10 - t_init # t_end - t_init was set to be 3600 seconds
# num_datapts = int(data_acquisition_rate * acquisition_duration) + 1 # the +1 is so that t_end is included

# # The empirical data of ecto-ecto cells in ca2+-free medium looked roughly
# # normally distributed (or maybe log-normal distribution...), so let us create
# # a normal distribution of data points with the same mean, standard devation, 
# # and sampling rate as our empirical data for an imaging session of 3600 sec
# # or 1 hour:
# norm_distrib_data = distrib.norm.rvs(mean, std, size=num_datapts, random_state=42)
# # keep the random state constant so that this is repeatable.

# # Assign each data point a time based on its position:
# norm_distrib_time = np.arange(num_datapts) * 1/data_acquisition_rate


# # Now interpolate between the different data points so that this normal 
# # distribution can be used as a function in a similar way to 'z_cont':
# z_cont_rand = interpolate.interp1d(norm_distrib_time, norm_distrib_data, bounds_error=False, fill_value=np.nan)
# z_cont_rand_args = ()


# mean = 1.0
# std = 0.26
# # The above mean and standard deviation are approximately equal to that of the
# # lifeact fluorescence at the cell-cell contact relative to the cell-medium
# # interface from ectoderm-ectoderm cells cultured in calcium-free medium
# data_acquisition_rate = 1/30 # 1 data point acquired every 30 seconds
# acquisition_duration = t_end*10 - t_init # t_end - t_init was set to be 3600 seconds
# num_datapts = int(data_acquisition_rate * acquisition_duration) + 1 # the +1 is so that t_end is included

# # The empirical data of ecto-ecto cells in ca2+-free medium looked roughly
# # normally distributed (or maybe log-normal distribution...), so let us create
# # a normal distribution of data points with the same mean, standard devation, 
# # and sampling rate as our empirical data for an imaging session of 3600 sec
# # or 1 hour:
# norm_distrib_data = distrib.norm.rvs(mean, std, size=num_datapts, random_state=24)
# # keep the random state constant so that this is repeatable.

# # Assign each data point a time based on its position:
# norm_distrib_time = np.arange(num_datapts) * 1/data_acquisition_rate


# # Now interpolate between the different data points so that this normal 
# # distribution can be used as a function in a similar way to 'z_cont':
# z_cm_rand = interpolate.interp1d(norm_distrib_time, norm_distrib_data, bounds_error=False, fill_value=np.nan)
z_cm_rand_args = ()


# these would be defined for either z_cc or z_cm later
# mean = 1.0
# std = 0.26
data_acquisition_rate = 1/30 # 1 data point acquired every 30 seconds
acquisition_duration = t_end*10 - t_init # t_end - t_init was set to be 3600 seconds

def z_rand(t, mean, std, data_acquis_rate, acquis_dur, z_init=None, random_state=None):
    ## Be aware that z_init will skew the random distribution slightly, but if
    ## there are many points the change should be inconsequential. 
    
    num_datapts = int(data_acquisition_rate * acquisition_duration) + 1 # the +1 is so that t_end is included
    
    norm_distrib_data = distrib.norm.rvs(mean, std, size=num_datapts, random_state=random_state)
    if z_init:
        norm_distrib_data[0] = z_init
    else:
        pass
    norm_distrib_time = np.arange(num_datapts) * 1/data_acquisition_rate
    z_rand = interpolate.interp1d(norm_distrib_time, norm_distrib_data, bounds_error=False, fill_value=np.nan)
    z = z_rand(t)

    return z


def z_const(t, val):
    return val


# This chunk below is just a test to make sure that the interpolation function
# worked correctly; it's for validation purposes:
# t_min, t_max = norm_distrib_time.min(), norm_distrib_time.max()
# t_step = int((t_max - t_min) * 10 + 1)
# t = np.linspace(t_min, t_max, t_step)
# y = rand_distrib_over_time(t)
# plt.scatter(norm_distrib_time, norm_distrib_data)




# A periodically fluctuating function:

# def z_cont_sin(t, initial, period, amplitude):
#     z_cc = -1 * amplitude * np.sin(2 * np.pi * t / period - np.pi/2) + initial
#     # z_cc = amplitude * np.sin(2 * np.pi * t / period - np.pi/2) + initial
#     return z_cc

# # def z_cont_sin_v2(t, initial, period, amplitude, vectorized=False):
# #     # return 0 * t + initial # for a constant contact actin
# #     beta_c = -1 * amplitude * np.sin(2 * np.pi * t / period - np.pi/2) + initial
# #     if vectorized == True:
# #         beta_c[beta_c > initial] = initial
        
# #     elif vectorized == False:
# #         if beta_c > initial:
# #             beta_c = initial
# #     return beta_c

# # per = 120 # seconds
# # amp = (b_init - b_plat)

# z_cont_sin_args = (0.80, 120, 0.26)








### These are some other functions that I was playing around with, but they
### are not used in the script:

# Can I get a combination of an exponential decay with a sin wave for the contact actin decrease?
# def z_cont_decay_sin(t, b_init, t_half, b_plat, period, amplitude):
#     wave = -1 * amplitude * np.sin(2 * np.pi * t / period - np.pi/2) + b_plat
#     tau = t_half / np.log(2)
#     b = (b_init - b_plat) * np.exp(-1 * t / tau) + wave
#     return b

# # b_init = beta
# # t_half = 150
# # b_plat = beta * 0.25
# # per = 120 #seconds
# # amp = (b_init - b_plat) * 0.05


# desired_mean = 0.80
# desired_std = 0.26
# per = 120
# timepoints = np.linspace(t_init, t_end, (t_end - t_init)*10 + 1)

# def distrib2sin(t, desired_mean, desired_std, periodicity, tol=1e-12, niter_max=1e4):
#     amp = desired_std
    
#     count = 0
#     while count < niter_max:
#         z = z_cont_sin(timepoints, desired_mean, periodicity, amp)
#         z.std()
        
#         if abs(z.std() - desired_std) > tol and (z.std() - desired_mean) < 0:
#             amp = amp - (z.std() - desired_std) * 0.5
         
#         elif abs(z.std() - desired_std) > tol and (z.std() - desired_mean) > 0:
#             amp = amp + (z.std() - desired_std) * 0.5
            
#         elif abs(z.std() - desired_std) <= tol:
#             break
#         count += 1

#     return (z, desired_mean, amp, periodicity)

# z, sin_center, amp, periodicity = distrib2sin(timepoints, desired_mean, desired_std, per)

# print(z.mean(), z.std())
# print(sin_center, amp)














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



class spher_cap_geom:#(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, k_EA, k_eta_A, k_EN, k_eta_N, k_etam, Gamma, E_r, z_cont_func, z_cont_args, z_cm_func, z_cm_args):
    
    """ Returns some useful geometric relations for a spherical cap with a 
        constant volume but variable contact angle. """
    
    # def __init__(self, t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, k_EA, k_eta_A, k_EN, k_eta_N, k_etam, Gamma, E_r, z_cont_func, z_cont_args, z_cm_func, z_cm_args):
    def __init__(self, theta, R_s):
    
        # theta, Scc0, Scm0, SA0, SNcc0, SNcm0, SN0 = syst
    
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



    


def get_theta_dot_stretching(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args):
    theta, Scc0, Scm0, SA0, SNcc0, SNcm0, SN0 = syst
    
    SCap = spher_cap_geom(theta, R_s)
    # r_c, Scc, dScc_dtheta, Scm, dScm_dtheta, SN, dSN_dtheta = SCap.r_c, SCap.Scc, SCap.dScc_dtheta, SCap.Scm, SCap.dScm_dtheta, SCap.SN, SCap.dSN_dtheta
    r_c, Scc, dScc_dtheta, Scm, dScm_dtheta = SCap.r_c, SCap.Scc, SCap.dScc_dtheta, SCap.Scm, SCap.dScm_dtheta



    z_cc = z_cont_func(t, *z_cont_args)
    z_cm = z_cm_func(t, *z_cm_args)
    # z_N = z_cm
    z_Ncm = z_cm
    z_Ncc = z_cm
    
    
    dScc0_dt = -np.log(2)/t_Apt5 * (Scc0 - Scc)
    dScm0_dt = -np.log(2)/t_Apt5 * (Scm0 - Scm)
    dSNcc0_dt = -np.log(2)/t_Npt5 * (SNcc0 - Scc)
    dSNcm0_dt = -np.log(2)/t_Npt5 * (SNcm0 - Scm)
    # dSN0_dt = -np.log(2)/t_Npt5 * (SN0 - SN)    

    
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
    
    
    # theta_dot = [( (np.cos(theta) * (1 + k_EA*(Scm / Scm0 - 1) - k_eta_A*dScm0_dt*(Scm/Scm0**2))) - 
    #                 ((z_cc / z_cm) * (1 + k_EA*(Scc / Scc0 - 1) - k_eta_A*dScc0_dt*(Scc/Scc0**2))) ) / 
    #               ( env_drag_contrib - 
    #                 (k_eta_A / Scm0 * dScm_dtheta * np.cos(theta)) + 
    #                 (k_eta_A / Scc0 * (z_cc / z_cm) * dScc_dtheta) - 
    #                 ((z_N/z_cm) * k_eta_N / SN0 * dSN_dtheta * np.cos(theta)) +
    #                 ((z_N/z_cm) * k_eta_N / SN0 * dSN_dtheta) ) + 
    #               ( (np.cos(theta) * (z_N / z_cm) * (k_EN*(SN / SN0 - 1) - k_eta_N * (SN/SN0**2) * dSN0_dt )) - 
    #                 ((z_N / z_cm) * (k_EN * (SN/SN0 - 1) - k_eta_N * (SN/SN0**2) * dSN0_dt)) ) / 
    #               ( env_drag_contrib - 
    #                 (k_eta_A / Scm0 * dScm_dtheta * np.cos(theta)) + 
    #                 (k_eta_A / Scc0 * (z_cc / z_cm) * dScc_dtheta) - 
    #                 ((z_N/z_cm) * k_eta_N / SN0 * dSN_dtheta * np.cos(theta)) +
    #                 ((z_N/z_cm) * k_eta_N / SN0 * dSN_dtheta) ) + 
    #               ( adh_nrg_contrib - bulk_elast ) / 
    #               ( env_drag_contrib - 
    #                 (k_eta_A / Scm0 * dScm_dtheta * np.cos(theta)) + 
    #                 (k_eta_A / Scc0 * (z_cc / z_cm) * dScc_dtheta) - 
    #                 ((z_N/z_cm) * k_eta_N / SN0 * dSN_dtheta * np.cos(theta)) +
    #                 ((z_N/z_cm) * k_eta_N / SN0 * dSN_dtheta) )
    #               ][-1]
    
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




def get_theta_dot_rolling_inc(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args):
    theta, Scc0, Scm0, SA0, SNcc0, SNcm0, SN0 = syst

    SCap = spher_cap_geom(theta, R_s)
    r_c, dScc_dtheta, SA, dSA_dtheta, SN, dSN_dtheta = SCap.r_c, SCap.dScc_dtheta, SCap.SA, SCap.dSA_dtheta, SCap.SN, SCap.dSN_dtheta


    # z_cc = z_cont_func(t, *z_cont_args)
    z_cm = z_cm_func(t, *z_cm_args)
    z_ccG = z_cm
    # z_N = z_cm
    z_Ncm = z_cm
    z_Ncc = z_cm
    
    

    
    dSN0_dt = -np.log(2)/t_Npt5 * (SN0 - SN)
    dSA0_dt = -np.log(2)/t_Apt5 * (SA0 - SA)
    
    
    
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

    
    ## This is the angular velocity ie. the rate at which the contact angle
    ## is changing:
    # theta_dot = [( (np.cos(theta) * (1 + k_EA*(SA / SA0 - 1) - k_eta_A*dSA0_dt*(SA/SA0**2))) - 
    #               ((z_ccG / z_cm) * (1 + k_EA*(SA / SA0 - 1) - k_eta_A*dSA0_dt*(SA/SA0**2))) ) / 
    #               ( env_drag_contrib - 
    #               (k_eta_A / SA0 * dSA_dtheta * np.cos(theta)) + 
    #               (k_eta_A / SA0 * (z_ccG / z_cm) * dSA_dtheta) - 
    #               ((z_N/z_cm) * k_eta_N / SN0 * dSN_dtheta * np.cos(theta)) +
    #               ((z_N/z_cm) * k_eta_N / SN0 * dSN_dtheta) ) + 
    #               ( (np.cos(theta) * (z_N / z_cm) * (k_EN*(SN / SN0 - 1) - k_eta_N * (SN/SN0**2) * dSN0_dt )) - 
    #               ((z_N / z_cm) * (k_EN * (SN/SN0 - 1) - k_eta_N * (SN/SN0**2) * dSN0_dt)) ) / 
    #               ( env_drag_contrib - 
    #               (k_eta_A / SA0 * dSA_dtheta * np.cos(theta)) + 
    #               (k_eta_A / SA0 * (z_ccG / z_cm) * dSA_dtheta) - 
    #               ((z_N/z_cm) * k_eta_N / SN0 * dSN_dtheta * np.cos(theta)) +
    #               ((z_N/z_cm) * k_eta_N / SN0 * dSN_dtheta) ) + 
    #               ( adh_nrg_contrib - bulk_elast ) / 
    #               ( env_drag_contrib - 
    #               (k_eta_A / SA0 * dSA_dtheta * np.cos(theta)) + 
    #               (k_eta_A / SA0 * (z_ccG / z_cm) * dSA_dtheta) - 
    #               ((z_N/z_cm) * k_eta_N / SN0 * dSN_dtheta * np.cos(theta)) +
    #               ((z_N/z_cm) * k_eta_N / SN0 * dSN_dtheta) )
    #               ][-1]

    theta_dot = [( (np.cos(theta) * (sigma_a + (E_A/(1 - nu_A))*(SA / SA0 - 1) - eta_A*dSA0_dt*(SA/SA0**2))) - 
                  ((z_ccG / z_cm) * (sigma_a + (E_A/(1 - nu_A))*(SA / SA0 - 1) - eta_A*dSA0_dt*(SA/SA0**2))) ) / 
                  ( env_drag_contrib - 
                  (eta_A / SA0 * dSA_dtheta * np.cos(theta)) + 
                  (eta_A / SA0 * (z_ccG / z_cm) * dSA_dtheta) - 
                  ((z_Ncm/z_cm) * eta_N / SN0 * dSN_dtheta * np.cos(theta)) +
                  ((z_Ncc/z_cm) * eta_N / SN0 * dSN_dtheta) ) + 
                  ( (np.cos(theta) * (z_Ncm / z_cm) * ((E_N/(1 - nu_N))*(SN / SN0 - 1) - eta_N * (SN/SN0**2) * dSN0_dt )) - 
                  ((z_Ncc / z_cm) * ((E_N/(1 - nu_N)) * (SN/SN0 - 1) - eta_N * (SN/SN0**2) * dSN0_dt)) ) / 
                  ( env_drag_contrib - 
                  (eta_A / SA0 * dSA_dtheta * np.cos(theta)) + 
                  (eta_A / SA0 * (z_ccG / z_cm) * dSA_dtheta) - 
                  ((z_Ncm/z_cm) * eta_N / SN0 * dSN_dtheta * np.cos(theta)) +
                  ((z_Ncc/z_cm) * eta_N / SN0 * dSN_dtheta) ) + 
                  ( adh_nrg_contrib - bulk_elast ) / 
                  ( env_drag_contrib - 
                  (eta_A / SA0 * dSA_dtheta * np.cos(theta)) + 
                  (eta_A / SA0 * (z_ccG / z_cm) * dSA_dtheta) - 
                  ((z_Ncm/z_cm) * eta_N / SN0 * dSN_dtheta * np.cos(theta)) +
                  ((z_Ncc/z_cm) * eta_N / SN0 * dSN_dtheta) ) 
                  ][-1]

    
    return theta_dot
    
    
def get_theta_dot_rolling_dec(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args):
    theta, Scc0, Scm0, SA0, SNcc0, SNcm0, SN0 = syst

    SCap = spher_cap_geom(theta, R_s)
    r_c, dScc_dtheta, SA, dSA_dtheta, SN, dSN_dtheta = SCap.r_c, SCap.dScc_dtheta, SCap.SA, SCap.dSA_dtheta, SCap.SN, SCap.dSN_dtheta

    z_cc = z_cont_func(t, *z_cont_args)
    z_cm = z_cm_func(t, *z_cm_args)
    z_cmG = z_cc
    # z_N = z_cm
    z_Ncm = z_cm
    z_Ncc = z_cm

    
    dSN0_dt = -np.log(2)/t_Npt5 * (SN0 - SN)
    dSA0_dt = -np.log(2)/t_Apt5 * (SA0 - SA)
    
    
    
    ## This is the contribution to contact expansion from the adhesion energy:
    adh_nrg_contrib = Gamma / (2*z_cmG)
    
    ## Viscous culture medium contribution that resists changes in contact size. 
    ## The assumption here is that we focus only on what is occurring locally
    ## around the contact zone. An actual analysis of fluid flow and drag on a 
    ## cell as it spreads on another cell seems unecessary since the contribution
    ## from this is small relative to the cortical tensions. 
    env_drag_contrib = (etam/(2*np.pi*r_c*z_cmG)) * dScc_dtheta
    
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
    
    ## This is the angular velocity ie. the rate at which the contact angle
    ## is changing:
    # theta_dot = [( (np.cos(theta) * (1 + k_EA*(SA / SA0 - 1) - k_eta_A*dSA0_dt*(SA/SA0**2))) - 
    #               ((z_cc / z_cmG) * (1 + k_EA*(SA / SA0 - 1) - k_eta_A*dSA0_dt*(SA/SA0**2))) ) / 
    #               ( env_drag_contrib - 
    #               (k_eta_A / SA0 * dSA_dtheta * np.cos(theta)) + 
    #               (k_eta_A / SA0 * (z_cc / z_cmG) * dSA_dtheta) - 
    #               ((z_N/z_cmG) * k_eta_N / SN0 * dSN_dtheta * np.cos(theta)) +
    #               ((z_N/z_cmG) * k_eta_N / SN0 * dSN_dtheta) ) + 
    #               ( (np.cos(theta) * (z_N / z_cmG) * (k_EN*(SN / SN0 - 1) - k_eta_N * (SN/SN0**2) * dSN0_dt )) - 
    #               ((z_N / z_cmG) * (k_EN * (SN/SN0 - 1) - k_eta_N * (SN/SN0**2) * dSN0_dt)) ) / 
    #               ( env_drag_contrib - 
    #               (k_eta_A / SA0 * dSA_dtheta * np.cos(theta)) + 
    #               (k_eta_A / SA0 * (z_cc / z_cmG) * dSA_dtheta) - 
    #               ((z_N/z_cmG) * k_eta_N / SN0 * dSN_dtheta * np.cos(theta)) +
    #               ((z_N/z_cmG) * k_eta_N / SN0 * dSN_dtheta) ) + 
    #               ( adh_nrg_contrib - bulk_elast ) / 
    #               ( env_drag_contrib - 
    #               (k_eta_A / SA0 * dSA_dtheta * np.cos(theta)) + 
    #               (k_eta_A / SA0 * (z_cc / z_cmG) * dSA_dtheta) - 
    #               ((z_N/z_cmG) * k_eta_N / SN0 * dSN_dtheta * np.cos(theta)) +
    #               ((z_N/z_cmG) * k_eta_N / SN0 * dSN_dtheta) )
    #               ][-1]
    theta_dot = [( (np.cos(theta) * (sigma_a + (E_A/(1 - nu_A))*(SA / SA0 - 1) - eta_A*dSA0_dt*(SA/SA0**2))) - 
                  ((1) * (sigma_a + (E_A/(1 - nu_A))*(SA / SA0 - 1) - eta_A*dSA0_dt*(SA/SA0**2))) ) / 
                  ( env_drag_contrib - 
                  (eta_A / SA0 * dSA_dtheta * np.cos(theta)) + 
                  (eta_A / SA0 * (1) * dSA_dtheta) - 
                  ((z_Ncm/z_cmG) * eta_N / SN0 * dSN_dtheta * np.cos(theta)) +
                  ((z_Ncc/z_cmG) * eta_N / SN0 * dSN_dtheta) ) + 
                  ( (np.cos(theta) * (z_Ncm / z_cmG) * ((E_N/(1 - nu_N))*(SN / SN0 - 1) - eta_N * (SN/SN0**2) * dSN0_dt )) - 
                  ((z_Ncc / z_cmG) * ((E_N/(1 - nu_N)) * (SN/SN0 - 1) - eta_N * (SN/SN0**2) * dSN0_dt)) ) / 
                  ( env_drag_contrib - 
                  (eta_A / SA0 * dSA_dtheta * np.cos(theta)) + 
                  (eta_A / SA0 * (1) * dSA_dtheta) - 
                  ((z_Ncm/z_cmG) * eta_N / SN0 * dSN_dtheta * np.cos(theta)) +
                  ((z_Ncc/z_cmG) * eta_N / SN0 * dSN_dtheta) ) + 
                  ( adh_nrg_contrib - bulk_elast ) / 
                  ( env_drag_contrib - 
                  (eta_A / SA0 * dSA_dtheta * np.cos(theta)) + 
                  (eta_A / SA0 * (1) * dSA_dtheta) - 
                  ((z_Ncm/z_cmG) * eta_N / SN0 * dSN_dtheta * np.cos(theta)) +
                  ((z_Ncc/z_cmG) * eta_N / SN0 * dSN_dtheta) ) 
                  ][-1]

    return theta_dot
    
    
    
def get_theta_dot_strain_lim_cc(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args):
    theta, Scc0, Scm0, SA0, SNcc0, SNcm0, SN0 = syst

    SCap = spher_cap_geom(theta, R_s)
    Scc, dScc_dtheta = SCap.Scc, SCap.dScc_dtheta

    dScc0_dt = -np.log(2)/t_Apt5 * (Scc0 - Scc)
    
    strain_cc = Scc/Scc0 - 1
    strain_max_actin = strain_lim_actin * np.sign(strain_cc)
    
    theta_dot_strainlim_cc = dScc0_dt * (strain_max_actin + 1) * (dScc_dtheta)**-1

    return theta_dot_strainlim_cc



def get_theta_dot_strain_lim_wholesurf(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args):
    theta, Scc0, Scm0, SA0, SNcc0, SNcm0, SN0 = syst

    SCap = spher_cap_geom(theta, R_s)
    SN, dSN_dtheta = SCap.SN, SCap.dSN_dtheta

    dSN0_dt = -np.log(2)/t_Npt5 * (SN0 - SN)    

    strain_N = SN/SN0 - 1
    strain_max_nonactin = strain_lim_nonactin * np.sign(strain_N)

    theta_dot_strainlim_N = dSN0_dt * (strain_max_nonactin + 1) * (dSN_dtheta)**-1    

    return theta_dot_strainlim_N






def strain_max_actin_event_start(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args):
    theta, Scc0, Scm0, SA0, SNcc0, SNcm0, SN0 = syst

    SCap = spher_cap_geom(theta, R_s)
    Scc = SCap.Scc
    
    strain_cc = Scc/Scc0 - 1
    dist_to_strain_max = abs(strain_cc) - strain_lim_actin
    
    return dist_to_strain_max
strain_max_actin_event_start.direction = 1 # only trigger event if abs(strain_cc) - strain_lim_actin is going from negative to positive
strain_max_actin_event_start.terminal = True # if event is triggered, stop integration



def strain_max_actin_event_end(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args):
    theta_dot = get_theta_dot_stretching(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args)
    theta_dot_strainlim_cc = get_theta_dot_strain_lim_cc(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args)
        
    dist_to_ang_veloc_max = theta_dot - theta_dot_strainlim_cc
    ## when strain is limited the magnitude of theta_dot_strainlim is less than
    ## the magnitude of theta_dot, however when theta_dot starts to reach its
    ## equilibrium angle then the magnitude of theta_dot will decrease, 
    ## eventually becoming smaller than theta_dot_strainlim, so the
    ## dist_to_ang_veloc_max will approach 0 
    return dist_to_ang_veloc_max
strain_max_actin_event_end.terminal = True # if event is triggered, stop integration




def strain_max_nonactin_event_start(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args):
    theta, Scc0, Scm0, SA0, SNcc0, SNcm0, SN0 = syst
    
    SCap = spher_cap_geom(theta, R_s)
    SN = SCap.SN

    strain_N = SN/SN0 - 1
    dist_to_strain_max = abs(strain_N) - strain_lim_nonactin #+ strain_lim_event_stol
    
    return dist_to_strain_max
strain_max_nonactin_event_start.direction = 1 # only trigger event if abs(strain_cc) - strain_lim_actin is going from negative to positive
strain_max_nonactin_event_start.terminal = True # if event is triggered, stop integration



def strain_max_nonactin_event_end(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args):
    theta_dot = get_theta_dot_stretching(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args)
    theta_dot_strainlim_N = get_theta_dot_strain_lim_wholesurf(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args)
    
    dist_to_ang_veloc_max = theta_dot - theta_dot_strainlim_N
    ## when strain is limited the magnitude of theta_dot_strainlim is less than
    ## the magnitude of theta_dot, however when theta_dot starts to reach its
    ## equilibrium angle then the magnitude of theta_dot will decrease, 
    ## eventually becoming smaller than theta_dot_strainlim, so the
    ## dist_to_ang_veloc_max will approach 0 
    return dist_to_ang_veloc_max
strain_max_nonactin_event_end.terminal = True # if event is triggered, stop integration













def theta_exceeds_Gamma_inc_event_start(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args):
    theta_dot = get_theta_dot_stretching(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args)
    ## This is the rate at which the rolling method of contact proceeds. If it
    ## is positive then the two cells will continue to roll on to each other
    ## to spread, but once it becomes 0 then rolling can no longer proceed:
    theta_Gamma_dot = get_theta_dot_rolling_inc(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args)
    
    
    ## This is the rate that the angle is actually increasing at. It incorporates
    ## both rolling and stretching modes of contact expansion:
    # theta_dot = get_theta_dot_rolling_and_stretching(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, k_EA, k_eta_A, k_EN, k_eta_N, k_etam, Gamma, E_r, z_cont_func, z_cont_args, z_cm_func, z_cm_args)
    # return theta_Gamma_dot # when this is 0 the event will be triggered
    return theta_Gamma_dot - theta_dot # when this is 0 the event will be triggered
theta_exceeds_Gamma_inc_event_start.direction = -1 # trigger when theta_Gamma_dot becomes negative; ie. theta_dot increases to become larger than theta_Gamma_dot
theta_exceeds_Gamma_inc_event_start.terminal = True



# def theta_exceeds_Gamma_inc_event_end(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, k_EA, k_eta_A, k_EN, k_eta_N, k_etam, Gamma, E_r, z_cont_func, z_cont_args, z_cm_func, z_cm_args):
#     ## This one is how the contact angle is currently changing (ie. through the
#     ## stretching mechanism): 
#     # theta_dot = get_theta_dot_stretching(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, k_EA, k_eta_A, k_EN, k_eta_N, k_etam, Gamma, E_r, z_cont_func, z_cont_args, z_cm_func, z_cm_args)

#     ## This one accounts for both rolling and stretching, and we use it instead
#     ## of the one that accounts only for rolling since when this event is  
#     ## triggered the Gamma_inc function will use this diff eq.
#     # theta_Gamma_dot = get_theta_dot_rolling_and_stretching(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, k_EA, k_eta_A, k_EN, k_eta_N, k_etam, Gamma, E_r, z_cont_func, z_cont_args, z_cm_func, z_cm_args)
#     theta_Gamma_dot = get_theta_dot_rolling_inc(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, k_EA, k_eta_A, k_EN, k_eta_N, k_etam, Gamma, E_r, z_cont_func, z_cont_args, z_cm_func, z_cm_args)
#     # return theta_Gamma_dot - theta_dot # when this is 0 the event will be triggered
#     return theta_Gamma_dot # when this is 0 the event will be triggered
# theta_exceeds_Gamma_inc_event_end.direction = 1 # trigger when theta_dot decreases to become smaller than theta_Gamma_dot
# theta_exceeds_Gamma_inc_event_end.terminal = True




# def theta_exceeds_Gamma_dec_event_start(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, k_EA, k_eta_A, k_EN, k_eta_N, k_etam, Gamma, E_r, z_cont_func, z_cont_args, z_cm_func, z_cm_args):    
#     ## This is how the contact angle is currently changing:
#     # theta_dot = get_theta_dot_stretching(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, k_EA, k_eta_A, k_EN, k_eta_N, k_etam, Gamma, E_r, z_cont_func, z_cont_args, z_cm_func, z_cm_args)
    
#     ## When this becomes positive the contact would start the rolling adhesion 
#     ## growth mechanism again because Gamma is sufficiently high:
#     theta_Gamma_dot = get_theta_dot_rolling_dec(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, k_EA, k_eta_A, k_EN, k_eta_N, k_etam, Gamma, E_r, z_cont_func, z_cont_args, z_cm_func, z_cm_args)
#     return theta_Gamma_dot # when this is 0 the event will be triggered
# theta_exceeds_Gamma_dec_event_start.direction = 1 # trigger when theta_Gamma_dot increases to become larger than theta_dot
# theta_exceeds_Gamma_dec_event_start.terminal = True



# def theta_exceeds_Gamma_dec_event_end(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, k_EA, k_eta_A, k_EN, k_eta_N, k_etam, Gamma, E_r, z_cont_func, z_cont_args, z_cm_func, z_cm_args):    
#     ## This is how the contact angle is currently changing:
#     # theta_dot = get_theta_dot_stretching(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, k_EA, k_eta_A, k_EN, k_eta_N, k_etam, Gamma, E_r, z_cont_func, z_cont_args, z_cm_func, z_cm_args)
    
#     ## When this becomes positive the contact would start the rolling adhesion 
#     ## growth mechanism again because Gamma is sufficiently high:
#     theta_Gamma_dot = get_theta_dot_rolling_inc(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, k_EA, k_eta_A, k_EN, k_eta_N, k_etam, Gamma, E_r, z_cont_func, z_cont_args, z_cm_func, z_cm_args)
#     return theta_Gamma_dot # when this is 0 the event will be triggered
# theta_exceeds_Gamma_dec_event_end.direction = 1 # trigger when theta_Gamma_dot increases to become larger than theta_dot
# theta_exceeds_Gamma_dec_event_end.terminal = True



# def angular_veloc_Gamma_inc_zero_event(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, k_EA, k_eta_A, k_EN, k_eta_N, k_etam, Gamma, E_r, z_cont_func, z_cont_args, z_cm_func, z_cm_args):
#     theta_Gamma_dot = get_theta_dot_rolling_inc(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, k_EA, k_eta_A, k_EN, k_eta_N, k_etam, Gamma, E_r, z_cont_func, z_cont_args, z_cm_func, z_cm_args)
#     return theta_Gamma_dot
# angular_veloc_Gamma_inc_zero_event.direction = -1
# angular_veloc_Gamma_inc_zero_event.terminal = True



# def angular_veloc_Gamma_dec_zero_event(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, k_EA, k_eta_A, k_EN, k_eta_N, k_etam, Gamma, E_r, z_cont_func, z_cont_args, z_cm_func, z_cm_args):
#     theta_Gamma_dot = get_theta_dot_rolling_dec(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, k_EA, k_eta_A, k_EN, k_eta_N, k_etam, Gamma, E_r, z_cont_func, z_cont_args, z_cm_func, z_cm_args)
#     return theta_Gamma_dot
# angular_veloc_Gamma_dec_zero_event.direction = 1
# angular_veloc_Gamma_dec_zero_event.terminal = True



# def angular_veloc_Gamma_inc_event_start(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, k_EA, k_eta_A, k_EN, k_eta_N, k_etam, Gamma, E_r, z_cont_func, z_cont_args, z_cm_func, z_cm_args):
#     theta_Gamma_dot = get_theta_dot_rolling_inc(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, k_EA, k_eta_A, k_EN, k_eta_N, k_etam, Gamma, E_r, z_cont_func, z_cont_args, z_cm_func, z_cm_args)
#     # theta_Gamma_dot_cc = get_theta_dot_rolling_dec(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, k_EA, k_eta_A, k_EN, k_eta_N, k_etam, Gamma, E_r, z_cont_func, z_cont_args, z_cm_func, z_cm_args)
#     return theta_Gamma_dot
# # angular_veloc_Gamma_inc_start_event.direction = 1 # trigger when theta_Gamma_dot increases above 0
# angular_veloc_Gamma_inc_event_start.direction = -1 # trigger when theta_Gamma_dot decreases below 0
# ## this event direction might seem counter-intuitive, but when the contact is 
# ## shrinking, then get_theta_dot_rolling_inc is increasing (because the contact
# ## is smaller than what the adhesion energy predicts it should be based on 
# ## bringing the cell-medium cortical tension together, but since peeling occurs
# ## when the cell-cell tension is greater than the cell-medium tension, it's 
# ## only when get_theta_dot_rolling_inc becomes negative, ie. the contact has
# ## grown above the equilibrium angle determined by that function, that the 
# ## rolling increase function can proceed.)
# ## In order for the contact to grow when the contact cortical tension is
# ## stronger than the free surface cortical tension, the contact cortical 
# ## tension has to decrease (ie. it is the limiting/determining factor in
# ## contact growth).
# angular_veloc_Gamma_inc_event_start.terminal = True


def angular_veloc_Gamma_inc_event_start(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args):
    theta_Gamma_dot = get_theta_dot_rolling_inc(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args)
    return theta_Gamma_dot
angular_veloc_Gamma_inc_event_start.direction = 1 # trigger when theta_Gamma_dot decreases below 0
angular_veloc_Gamma_inc_event_start.terminal = True

def angular_veloc_Gamma_inc_event_end(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args):
    theta_Gamma_dot = get_theta_dot_rolling_inc(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args)
    return theta_Gamma_dot
angular_veloc_Gamma_inc_event_end.direction = -1 # trigger when theta_Gamma_dot decreases below 0
angular_veloc_Gamma_inc_event_end.terminal = True



def angular_veloc_Gamma_dec_event_start(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args):
    theta_Gamma_dot = get_theta_dot_rolling_dec(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args)
    return theta_Gamma_dot
angular_veloc_Gamma_dec_event_start.direction = -1 # trigger when theta_Gamma_dot decreases below 0
angular_veloc_Gamma_dec_event_start.terminal = True

def angular_veloc_Gamma_dec_event_end(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args):
    theta_Gamma_dot = get_theta_dot_rolling_dec(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args)
    return theta_Gamma_dot
angular_veloc_Gamma_dec_event_end.direction = 1 # trigger when theta_Gamma_dot decreases below 0
angular_veloc_Gamma_dec_event_end.terminal = True






def theta_max_event(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args):
    theta, Scc0, Scm0, SA0, SNcc0, SNcm0, SN0 = syst

    return theta - theta_max
theta_max_event.terminal = True



def theta_min_event_start(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args):
    theta, Scc0, Scm0, SA0, SNcc0, SNcm0, SN0 = syst

    return theta - theta_min
theta_min_event_start.direction = -1 # only trigger if the angle is going from positive to negative.
theta_min_event_start.terminal = True



def theta_min_event_end_G(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args):
    theta_Gamma_dot = get_theta_dot_rolling_inc(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args)
    return theta_Gamma_dot
theta_min_event_end_G.direction = 1 # only trigger if the angle starts growing.
theta_min_event_end_G.terminal = True



def theta_min_event_end_A(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args):
    theta_dot = get_theta_dot_strain_lim_cc(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args)
    return theta_dot
theta_min_event_end_A.direction = 1 # only trigger if the angle starts growing.
theta_min_event_end_A.terminal = True

def theta_min_event_end_N(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args):
    theta_dot = get_theta_dot_strain_lim_wholesurf(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args)
    return theta_dot
theta_min_event_end_N.direction = 1 # only trigger if the angle starts growing.
theta_min_event_end_N.terminal = True


def theta_min_event_end(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args):
    theta_dot = get_theta_dot_stretching(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args)
    return theta_dot
theta_min_event_end.direction = 1 # only trigger if the angle starts growing.
theta_min_event_end.terminal = True





def angular_veloc_less_theta_min(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args):
    theta, Scc0, Scm0, SA0, SNcc0, SNcm0, SN0 = syst
     
    SCap = spher_cap_geom(theta, R_s)
    Scc, Scm, SN, SA = SCap.Scc, SCap.Scm, SCap.SN, SCap.SA

    dScc0_dt = -np.log(2)/t_Apt5 * (Scc0 - Scc)
    dScm0_dt = -np.log(2)/t_Apt5 * (Scm0 - Scm)
    dSA0_dt = -np.log(2)/t_Apt5 * (SA0 - SA)
    dSNcc0_dt = -np.log(2)/t_Npt5 * (SNcc0 - Scc)
    dSNcm0_dt = -np.log(2)/t_Npt5 * (SNcm0 - Scm)
    dSN0_dt = -np.log(2)/t_Npt5 * (SN0 - SN)

    theta_dot = 0

    syst = [theta_dot, dScc0_dt, dScm0_dt, dSA0_dt, dSNcc0_dt, dSNcm0_dt, dSN0_dt, ]
    return syst



def angular_veloc_Gamma_inc(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args):
    ## would apply Gamma causes contact to grow:
    theta, Scc0, Scm0, SA0, SNcc0, SNcm0, SN0 = syst
    
    SCap = spher_cap_geom(theta, R_s)
    Scc, Scm, SN, SA = SCap.Scc, SCap.Scm, SCap.SN, SCap.SA

    dScc0_dt = -np.log(2)/t_Apt5 * (Scc0 - Scc)
    dScm0_dt = -np.log(2)/t_Apt5 * (Scm0 - Scm)
    dSA0_dt = -np.log(2)/t_Apt5 * (SA0 - SA)
    dSNcc0_dt = -np.log(2)/t_Npt5 * (SNcc0 - Scc)
    dSNcm0_dt = -np.log(2)/t_Npt5 * (SNcm0 - Scm)
    dSN0_dt = -np.log(2)/t_Npt5 * (SN0 - SN)

    theta_dot = get_theta_dot_rolling_inc(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args)

    syst = [theta_dot, dScc0_dt, dScm0_dt, dSA0_dt, dSNcc0_dt, dSNcm0_dt, dSN0_dt, ]
    return syst

def angular_veloc_Gamma_dec(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args):
    ## would apply Gamma causes contact to shrink:
    theta, Scc0, Scm0, SA0, SNcc0, SNcm0, SN0 = syst
    
    SCap = spher_cap_geom(theta, R_s)
    Scc, Scm, SN, SA = SCap.Scc, SCap.Scm, SCap.SN, SCap.SA

    dScc0_dt = -np.log(2)/t_Apt5 * (Scc0 - Scc)
    dScm0_dt = -np.log(2)/t_Apt5 * (Scm0 - Scm)
    dSA0_dt = -np.log(2)/t_Apt5 * (SA0 - SA)
    dSNcc0_dt = -np.log(2)/t_Npt5 * (SNcc0 - Scc)
    dSNcm0_dt = -np.log(2)/t_Npt5 * (SNcm0 - Scm)
    dSN0_dt = -np.log(2)/t_Npt5 * (SN0 - SN)

    theta_dot = get_theta_dot_rolling_dec(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args)
        
    syst = [theta_dot, dScc0_dt, dScm0_dt, dSA0_dt, dSNcc0_dt, dSNcm0_dt, dSN0_dt, ]
    return syst


  
    


def quicksolve(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args):
    """This is just to quickly get a solve_ivp object that can be modified to 
    have its last timepoint be np.inf. This is done to make some later parts 
    cleaner/shorter. """
    theta, Scc0, Scm0, SA0, SNcc0, SNcm0, SN0 = syst
    
    SCap = spher_cap_geom(theta, R_s)
    Scc, Scm, SN, SA = SCap.Scc, SCap.Scm, SCap.SN, SCap.SA

    dScc0_dt = -np.log(2)/t_Apt5 * (Scc0 - Scc)
    dScm0_dt = -np.log(2)/t_Apt5 * (Scm0 - Scm)
    dSA0_dt = -np.log(2)/t_Apt5 * (SA0 - SA)
    dSNcc0_dt = -np.log(2)/t_Npt5 * (SNcc0 - Scc)
    dSNcm0_dt = -np.log(2)/t_Npt5 * (SNcm0 - Scm)
    dSN0_dt = -np.log(2)/t_Npt5 * (SN0 - SN)
    
    theta_dot = 0
    
    syst = [theta_dot, dScc0_dt, dScm0_dt, dSA0_dt, dSNcc0_dt, dSNcm0_dt, dSN0_dt, ]
    return syst




def angular_veloc(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args):
    theta, Scc0, Scm0, SA0, SNcc0, SNcm0, SN0 = syst

    SCap = spher_cap_geom(theta, R_s)
    Scc, Scm, SN, SA = SCap.Scc, SCap.Scm, SCap.SN, SCap.SA

    dScc0_dt = -np.log(2)/t_Apt5 * (Scc0 - Scc)
    dScm0_dt = -np.log(2)/t_Apt5 * (Scm0 - Scm)
    dSA0_dt = -np.log(2)/t_Apt5 * (SA0 - SA)
    dSNcc0_dt = -np.log(2)/t_Npt5 * (SNcc0 - Scc)
    dSNcm0_dt = -np.log(2)/t_Npt5 * (SNcm0 - Scm)
    dSN0_dt = -np.log(2)/t_Npt5 * (SN0 - SN)

    theta_dot = get_theta_dot_stretching(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args)
    
    syst = [theta_dot, dScc0_dt, dScm0_dt, dSA0_dt, dSNcc0_dt, dSNcm0_dt, dSN0_dt, ]
    return syst



def angular_veloc_strainlim_actin(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args):
    theta, Scc0, Scm0, SA0, SNcc0, SNcm0, SN0 = syst
    
    SCap = spher_cap_geom(theta, R_s)
    Scc, Scm, SN, SA = SCap.Scc, SCap.Scm, SCap.SN, SCap.SA

    dScc0_dt = -np.log(2)/t_Apt5 * (Scc0 - Scc)
    dScm0_dt = -np.log(2)/t_Apt5 * (Scm0 - Scm)
    dSA0_dt = -np.log(2)/t_Apt5 * (SA0 - SA)
    dSNcc0_dt = -np.log(2)/t_Npt5 * (SNcc0 - Scc)
    dSNcm0_dt = -np.log(2)/t_Npt5 * (SNcm0 - Scm)
    dSN0_dt = -np.log(2)/t_Npt5 * (SN0 - SN)
    
    theta_dot = get_theta_dot_strain_lim_cc(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args)
    
    syst = [theta_dot, dScc0_dt, dScm0_dt, dSA0_dt, dSNcc0_dt, dSNcm0_dt, dSN0_dt, ]
    return syst




def angular_veloc_strainlim_nonactin(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args):
    theta, Scc0, Scm0, SA0, SNcc0, SNcm0, SN0 = syst
    
    SCap = spher_cap_geom(theta, R_s)
    Scc, Scm, SN, SA = SCap.Scc, SCap.Scm, SCap.SN, SCap.SA

    dScc0_dt = -np.log(2)/t_Apt5 * (Scc0 - Scc)
    dScm0_dt = -np.log(2)/t_Apt5 * (Scm0 - Scm)
    dSA0_dt = -np.log(2)/t_Apt5 * (SA0 - SA)
    dSNcc0_dt = -np.log(2)/t_Npt5 * (SNcc0 - Scc)
    dSNcm0_dt = -np.log(2)/t_Npt5 * (SNcm0 - Scm)
    dSN0_dt = -np.log(2)/t_Npt5 * (SN0 - SN)

    theta_dot = get_theta_dot_strain_lim_wholesurf(t, syst, R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args)

    syst = [theta_dot, dScc0_dt, dScm0_dt, dSA0_dt, dSNcc0_dt, dSNcm0_dt, dSN0_dt, ]
    return syst












def angular_veloc_ve_model(t_span, syst, args, solve_ivp_method, solve_ivp_max_step, solve_ivp_atol, solve_ivp_rtol, strain_lim_event_stol, calcium=True):
    ### the 'calcium' keyword is used to decide if cortices of cells can be 
    ### coupled. If calcium == True then it is assumed that cadherins can bind
    ### and therefore link the cortical networks of each cell together. As a 
    ### consequence Gamma_exceeded can be either 'True' or 'False'
    ### if calcium == False, then Gamma_exceeded can only ever be False as well.
    
      
    
    ## These lines below are only used for convenience while testing/debugging:
    # calcium = True
    # Gamma, strain_lim_actin, t_Apt5, k_EA, k_eta_A, strain_lim_nonactin, t_Npt5, k_EN, k_eta_N, E_r = *params,
    # consts = (R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, k_EA, k_eta_A, k_EN, k_eta_N, k_etam_real, Gamma, E_r, z_cc_func, z_cc_func_args, z_cm_func, z_cm_func_args)

    # args = consts
    # syst_init = theta_init, Scc0_init, Scm0_init, SA0_init, SN0_init
    
    # syst = syst_init
    # t_span = t_init, t_end
    
    
    # calcium = False
    # Gamma, strain_lim_actin, t_Apt5, k_EA, k_eta_A, strain_lim_nonactin, t_Npt5, k_EN, k_eta_N, E_r = *params,
    # # consts = (R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, k_EA, k_eta_A, k_EN, k_eta_N, k_etam_real, Gamma, E_r, z_cc_func, z_cc_func_args, z_cm_func_rand, z_cm_func_args_rand)

    # args = consts
    # syst_init = theta_init, Scc0_init, Scm0_init, SA0_init, SN0_init
    
    # syst = syst_init
    # t_span = t_init, t_end_rand
    ### end of debugging lines.


    ### First check which scenario we are starting with:
    t, t_end = t_span
    theta, Scc0, Scm0, SA0, SNcc0, SNcm0, SN0 = syst
    R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, E_A, eta_A, E_N, eta_N, etam, Gamma, E_B, z_cont_func, z_cont_args, z_cm_func, z_cm_args = args

    
    theta_min_exceeded = None
    Gamma_exceeded = None
    strain_max_status_actin = None
    strain_max_status_nonactin = None
    theta_max_status = False
    theta_Gamma_inc_status = True # determines if soln_Gamma_peeling_start should be calculated
    theta_Gamma_dec_status = True # determines if soln_Gamma_rolling_start should be calculated
    case_id = None
    


    ### First, is stretching or rolling adhesion occurring?:
    ### If stretching, then set Gamma_exceeded == True.
    ## Rolling contact growth:
    # theta_dot = get_theta_dot_stretching(t, syst, *args)
    theta_Gamma_dot_inc = get_theta_dot_rolling_inc(t, syst, *args)
    # if theta_Gamma_dot_inc < theta_dot:
    #     Gamma_exceeded = True
    # else:
    #     Gamma_exceeded = False
    
    if theta_Gamma_dot_inc > 0:
        Gamma_exceeded = False
        theta_Gamma_inc_status = True
        theta_Gamma_dec_status = False
        
    elif theta_Gamma_dot_inc < 0:
        Gamma_exceeded = True
        theta_Gamma_inc_status = False
    
    elif theta_Gamma_dot_inc == 0:
        Gamma_exceeded = True
        theta_Gamma_inc_status = False
    
    ## Rolling contact shrinkage:
    theta_Gamma_dot_dec = get_theta_dot_rolling_dec(t, syst, *args)

    ## The contact can only peel apart if calcium == False (ie. cadherins are
    ## disengaged and cytoskeletal networks are not coupled to each other):
    if calcium == True:
        theta_Gamma_dec_status = False
        
    else:
        if theta_Gamma_dot_dec < 0:
            theta_Gamma_dec_status = True
            theta_Gamma_inc_status = False
            
        elif theta_Gamma_dot_dec > 0:
            theta_Gamma_dec_status = False
        
        elif theta_Gamma_dot_dec == 0:
            theta_Gamma_dec_status = False
    
    

    SCap = spher_cap_geom(theta, R_s)
    Scc, SN = SCap.Scc, SCap.SN
    
    strain_actin = Scc/Scc0 - 1
    strain_nonactin = SN/SN0 - 1
    
    
    if theta < theta_min:
        theta_min_exceeded = False
    elif theta >= theta_min:
        theta_min_exceeded = True
        
    if abs(strain_actin) <= (strain_lim_actin + strain_lim_event_stol):
        strain_max_status_actin = False
    elif abs(strain_actin) >= (strain_lim_actin - strain_lim_event_stol):
        strain_max_status_actin = True
        
    if abs(strain_nonactin) <= (strain_lim_nonactin + strain_lim_event_stol):
        strain_max_status_nonactin = False
    elif abs(strain_nonactin) >= (strain_lim_nonactin - strain_lim_event_stol):
        strain_max_status_nonactin = True
    
    
    
    ts = list()
    solns = list() # give separate lists to theta, theta0cc, theta0cm, theta0N?
    messages = list()
    soln_statuses = list()
    successes = list()
    t_events = list()
    theta_min_exceeded_status_list = list()
    Gamma_exceeded_status_list = list()
    strain_max_status_actin_list = list()
    strain_max_status_nonactin_list = list()
    theta_max_status_list = list()
    color_list = list()
    
    if np.isfinite(solve_ivp_max_step):
        count_max = (t_end - t) / solve_ivp_max_step
    else:
        count_max = np.inf
    
    count = 0
    while count < count_max:
        # if calcium == False:
        #     Gamma_exceeded = False

        print('case_id=%s'%case_id)
        print('\n count:%d , t=%.12f, theta_min_exceeded=%s, Gamma_exceeded=%s, strain_max_status_actin=%s, strain_max_status_nonactin=%s, theta_Gamma_inc_status=%s, theta_Gamma_dec_status=%s'%(count,t,theta_min_exceeded,Gamma_exceeded,strain_max_status_actin,strain_max_status_nonactin,theta_Gamma_inc_status,theta_Gamma_dec_status))
                
        if theta_min_exceeded == False: 
            # it doesn't matter what anything else is if
            # theta_min_exceeded is False, because theta will
            # then be set to equal theta_min...
            color = 'tab:red'
            
            soln_G = solve_ivp(angular_veloc_less_theta_min, (t, t_end), syst, args=args, method=solve_ivp_method, events=theta_min_event_end_G, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
            soln_A = solve_ivp(angular_veloc_less_theta_min, (t, t_end), syst, args=args, method=solve_ivp_method, events=theta_min_event_end, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
            soln_A.t[-1] = np.inf # this is inf only because I'm not using the strain-limited solutions as part of the model anymore
            soln_N = solve_ivp(angular_veloc_less_theta_min, (t, t_end), syst, args=args, method=solve_ivp_method, events=theta_min_event_end, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
            soln_N.t[-1] = np.inf # this is inf only because I'm not using the strain-limited solutions as part of the model anymore
            soln_default = solve_ivp(angular_veloc_less_theta_min, (t, t_end), syst, args=args, method=solve_ivp_method, events=theta_min_event_end, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)

            
            case_id = np.argmin([soln_G.t[-1], soln_A.t[-1], soln_N.t[-1], soln_default.t[-1]])
            
            if case_id == 0:
                soln = soln_G
                
            elif case_id == 1:
                soln = soln_A
                
            elif case_id == 2:
                soln = soln_N
                
            elif case_id == 3:
                soln = soln_default
                
            else:
                print('Unexpected behaviour while solving.')
                break                

            t = soln.t[-1]
            syst = soln.y[:,-1]
            
            theta, Scc0, Scm0, SA0, SNcc0, SNcm0, SN0 = syst

            SCap = spher_cap_geom(theta, R_s)
            Scc, SN = SCap.Scc, SCap.SN
            
            strain_actin = Scc/Scc0 - 1
            strain_nonactin = SN/SN0 - 1


            if soln.message == 'A termination event occurred.':
                theta_min_exceeded = True
            
            
            ## Just double-check the event statuses:
            if abs(strain_actin) <= (strain_lim_actin + strain_lim_event_stol):
                strain_max_status_actin = False
            elif abs(strain_actin) >= (strain_lim_actin - strain_lim_event_stol):
                strain_max_status_actin = True
                
            if abs(strain_nonactin) <= (strain_lim_nonactin + strain_lim_event_stol):
                strain_max_status_nonactin = False
            elif abs(strain_nonactin) >= (strain_lim_nonactin - strain_lim_event_stol):
                strain_max_status_nonactin = True



            
        elif Gamma_exceeded == False and theta_min_exceeded == True:

            # color = 'tab:grey'
            if calcium == True:
                color = 'tab:green'
                soln_theta_default = solve_ivp(angular_veloc_Gamma_inc, (t, t_end), syst, args=args, method=solve_ivp_method, events=None, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
                # soln_stretching_start = solve_ivp(angular_veloc_Gamma_inc, (t, t_end), syst, args=args, method=solve_ivp_method, events=theta_exceeds_Gamma_inc_event_start, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
                soln_stretching_start = solve_ivp(angular_veloc_Gamma_inc, (t, t_end), syst, args=args, method=solve_ivp_method, events=angular_veloc_Gamma_inc_event_end, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
                ## You can't get peeling of the contact when calcium is present /
                ## cytoskeletal networks are linked, so have to use angular_veloc
                ## and not angular_veloc_Gamma_dec:
                soln_theta_min_exceeded_end = solve_ivp(angular_veloc, (t, t_end), syst, args=args, method=solve_ivp_method, events=theta_min_event_start, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
                
                ## this won't be used (the last timpoint is set to inf), so just
                ## solve whatver equation that is fast and easy:
                soln_Gamma_peeling_start = solve_ivp(quicksolve, (t, t_end), syst, args=args, method=solve_ivp_method, events=None, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
                soln_Gamma_peeling_start.t[-1] = np.inf


                ## Since you can't get angular_veloc_Gamma_dec (ie. peeling) 
                ## when cytoskeletal networks are linked, we don't have to worry
                ## about theta_Gamma_dec_status.
                soln_Gamma_rolling_start = solve_ivp(quicksolve, (t, t_end), syst, args=args, method=solve_ivp_method, events=None, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
                soln_Gamma_rolling_start.t[-1] = np.inf

                
            elif calcium == False:
                ## if calcium == False then cytoskeletal cortexes not coupled
                ## therefore angle can never do beyond what Gamma allows:
                soln_theta_default = solve_ivp(quicksolve, (t, t_end), syst, args=args, method=solve_ivp_method, events=theta_max_event, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
                soln_theta_default.t[-1] = np.inf
                soln_stretching_start = solve_ivp(quicksolve, (t, t_end), syst, args=args, method=solve_ivp_method, events=theta_max_event, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
                soln_stretching_start.t[-1] = np.inf

                ## We have to use True/False flags because the solver can get stuck
                ## at the events:
                assert(theta_Gamma_inc_status or theta_Gamma_dec_status) 
                if theta_Gamma_inc_status == True:
                    color = 'tab:green'
                    # soln_stretching_start = solve_ivp(angular_veloc_Gamma_inc, (t, t_end), syst, args=args, method=solve_ivp_method, events=theta_exceeds_Gamma_inc_event_start, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
                    soln_stretching_start = solve_ivp(angular_veloc_Gamma_inc, (t, t_end), syst, args=args, method=solve_ivp_method, events=angular_veloc_Gamma_inc_event_end, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
                    # soln_stretching_start2 = solve_ivp(angular_veloc_Gamma_inc, (t, t_end), syst, args=args, method=solve_ivp_method, events=theta_exceeds_Gamma_inc_event_start, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
                    # soln_Gamma_peeling_start = solve_ivp(angular_veloc_Gamma_inc, (t, t_end), syst, args=args, method=solve_ivp_method, events=angular_veloc_Gamma_dec_event_start, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
                    soln_Gamma_peeling_start = solve_ivp(quicksolve, (t, t_end), syst, args=args, method=solve_ivp_method, events=None, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
                    soln_Gamma_peeling_start.t[-1] = np.inf
                else:
                    ## this won't be used (the last timpoint is set to inf), so just
                    ## solve whatver equation that is fast and easy:
                    soln_Gamma_peeling_start = solve_ivp(quicksolve, (t, t_end), syst, args=args, method=solve_ivp_method, events=None, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
                    soln_Gamma_peeling_start.t[-1] = np.inf
                
                if theta_Gamma_dec_status == True:
                    color = 'tab:red'
                    soln_theta_min_exceeded_end = solve_ivp(angular_veloc_Gamma_dec, (t, t_end), syst, args=args, method=solve_ivp_method, events=theta_min_event_start, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
                    soln_stretching_start = solve_ivp(angular_veloc_Gamma_dec, (t, t_end), syst, args=args, method=solve_ivp_method, events=angular_veloc_Gamma_dec_event_end, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
                    # soln_Gamma_rolling_start = soln_stretching_start
                    # soln_Gamma_rolling_start = solve_ivp(angular_veloc_Gamma_dec, (t, t_end), syst, args=args, method=solve_ivp_method, events=angular_veloc_Gamma_inc_event_start, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
                    ## Need to decide whether stretching or rolling is occurring 
                    ## once peeling stops
                    soln_Gamma_rolling_start = solve_ivp(quicksolve, (t, t_end), syst, args=args, method=solve_ivp_method, events=None, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
                    soln_Gamma_rolling_start.t[-1] = np.inf

                else:
                    soln_theta_min_exceeded_end = solve_ivp(quicksolve, (t, t_end), syst, args=args, method=solve_ivp_method, events=theta_max_event, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
                    soln_theta_min_exceeded_end.t[-1] = np.inf
                    soln_Gamma_rolling_start = solve_ivp(quicksolve, (t, t_end), syst, args=args, method=solve_ivp_method, events=None, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
                    soln_Gamma_rolling_start.t[-1] = np.inf
                    
            
            
            case_id = np.argmin([soln_theta_default.t[-1], soln_stretching_start.t[-1], soln_theta_min_exceeded_end.t[-1], soln_Gamma_peeling_start.t[-1], soln_Gamma_rolling_start.t[-1]])            
            # case_id = np.argmin([soln_theta_default.t[-1], min(soln_stretching_start.t[-1], soln_stretching_start2.t[-1]), soln_theta_min_exceeded_end.t[-1], soln_Gamma_peeling_start.t[-1], soln_Gamma_rolling_start.t[-1]])
            
            # theta_dot = get_theta_dot_stretching(soln_stretching_start.t[-1], soln_stretching_start.y[:,-1], *args)
            # theta_Gamma_dot_inc = get_theta_dot_rolling_inc(soln_stretching_start.t[-1], soln_stretching_start.y[:,-1], *args)
            # if case_id == 1 and theta_Gamma_dec_status == True:
            #     if theta_Gamma_dot_inc > theta_dot:
            #         case_id = 4
            #     else:
            #         pass
                
            ## case_id:
            ## 0 = soln_theta_max_exceeded
            ## 1 = soln_stretching_start
            ## 2 = soln_theta_min_exceeded_end
            ## 3 = soln_Gamma_peeling_start
            ## 4 = soln_Gamma_rolling_start
            
            
            
            # if the events happen at the same time then the
            # argmin will take the first index, therefore 
            # a Gamma_exceeded will change to true, and 
            # theta_min_exceeded was already true. 
            # If everything hits the maximum angle then no other events occurred.
            if case_id == 0:
                # soln = soln_theta_max_exceeded
                soln = soln_theta_default
                if soln.message == 'The solver successfully reached the end of the integration interval.':
                    pass

                elif soln.message == 'A termination event occurred.':
                    theta_max_status = True
                    soln.message = 'The solver successfully reached the end of the integration interval.'

                else:
                    print('Unexpected behaviour while solving.')
                    break
                
                
                
            elif case_id == 1:
                Gamma_exceeded = True
                ### soln_stretching_start can be either angular_veloc_Gamma_inc
                ### or angular_veloc_Gamma_dec, so need to associate color here.
                ## make sure one of these is true
                # assert(theta_Gamma_inc_status or theta_Gamma_dec_status) 
                # if theta_Gamma_inc_status:
                #     color = 'tab:green'
                # elif theta_Gamma_dec_status:
                #     color = 'tab:red'
                
                # soln_id = np.argmin([soln_stretching_start.t[-1], soln_stretching_start2.t[-1]])
                # soln = [soln_stretching_start, soln_stretching_start2][soln_id]

                soln = soln_stretching_start


                t = soln.t[-1]
                syst = soln.y[:,-1]

                theta, Scc0, Scm0, SA0, SNcc0, SNcm0, SN0 = syst
                                
                SCap = spher_cap_geom(theta, R_s)
                Scc, Scm, SA = SCap.Scc, SCap.Scm, SCap.SA
                                
                # # Since we have hit the limit of Gamma-induced growth, we 
                # # must assign the stress from the total surface to the contact
                # # and non-contact surfaces, and then set the reference angle
                # # for each of these surfaces to be consistent with those 
                # # stresses:
                # # SA_elast_stress = k_EA * sigma_a * z_cm * (SA/SA0 - 1) # elastic stress from whole surface:
                # # SA_visc_stress =  k_eta_A * sigma_a * z_cm * (dSA_dtheta * theta_dot / SA0 - dSA0_dt*(SA/SA0**2)) # viscous stress from whole surface:
                # is equal to the stress in the contact zone. Equating the two
                # and through some algebra we get the reference contact surface:
                z_cc = z_cont_func(t, *z_cont_args)
                z_cm = z_cm_func(t, *z_cm_args)
                Scc0 = Scc / ((z_cm/z_cc) * (SA/SA0 - 1) + 1)
                # eqn for reference non-contact surface is almost the same:
                Scm0 = Scm / ((z_cm/z_cm) * (SA/SA0 - 1) + 1)
                
                ## For the non-actin cytoskeleton:
                z_Ncm = z_cm
                z_Ncc = z_cm
                SNcc0 = Scc / ((z_Ncm/z_Ncc) * (SN/SN0 - 1) + 1)
                SNcm0 = Scm / ((z_Ncm/z_Ncc) * (SN/SN0 - 1) + 1)
                
                # # we are omitting the non-actin stresses because that network 
                # # does not separate from whole surface to contact and noncontact.
                    
                syst = [theta, Scc0, Scm0, SA0, SNcc0, SNcm0, SN0]

                
                
                theta, Scc0, Scm0, SA0, SNcc0, SNcm0, SN0 = syst

                SCap = spher_cap_geom(theta, R_s)
                Scc, SN = SCap.Scc, SCap.SN

                strain_actin = Scc/Scc0 - 1
                strain_nonactin = SN/SN0 - 1
                
                ## Just double-check the event statuses:
                if abs(strain_actin) <= (strain_lim_actin + strain_lim_event_stol):
                    strain_max_status_actin = False
                elif abs(strain_actin) >= (strain_lim_actin - strain_lim_event_stol):
                    strain_max_status_actin = True
                    
                if abs(strain_nonactin) <= (strain_lim_nonactin + strain_lim_event_stol):
                    strain_max_status_nonactin = False
                elif abs(strain_nonactin) >= (strain_lim_nonactin - strain_lim_event_stol):
                    strain_max_status_nonactin = True


                
            elif case_id == 2:
                theta_min_exceeded = False
                soln = soln_theta_min_exceeded_end
                
                t = soln.t[-1]
                syst = soln.y[:,-1]
                
                
                
                theta, Scc0, Scm0, SA0, SNcc0, SNcm0, SN0 = syst

                SCap = spher_cap_geom(theta, R_s)
                Scc, SN = SCap.Scc, SCap.SN

                strain_actin = Scc/Scc0 - 1
                strain_nonactin = SN/SN0 - 1
                
                ## Just double-check the event statuses:
                if abs(strain_actin) <= (strain_lim_actin + strain_lim_event_stol):
                    strain_max_status_actin = False
                elif abs(strain_actin) >= (strain_lim_actin - strain_lim_event_stol):
                    strain_max_status_actin = True
                    
                if abs(strain_nonactin) <= (strain_lim_nonactin + strain_lim_event_stol):
                    strain_max_status_nonactin = False
                elif abs(strain_nonactin) >= (strain_lim_nonactin - strain_lim_event_stol):
                    strain_max_status_nonactin = True
                
                
                
            # These 2 are also needed to look at low adhesion behaviour:
            ## These 2 would be needed for low adhesion analysis but it's too complicated so omit:
            elif case_id == 3:
                theta_Gamma_inc_status = False
                theta_Gamma_dec_status = True
                soln = soln_Gamma_peeling_start
                t = soln.t[-1]
                syst = soln.y[:,-1]
                
                # theta, Scc0, Scm0, SA0, SNcc0, SNcm0, SN0 = syst
                
                # SCap = spher_cap_geom(theta, R_s)
                # Scc, Scm, SA = SCap.Scc, SCap.Scm, SCap.SA
                                
                # # # Since we have hit the limit of Gamma-induced growth, we 
                # # # must assign the stress from the total surface to the contact
                # # # and non-contact surfaces, and then set the reference angle
                # # # for each of these surfaces to be consistent with those 
                # # # stresses:
                # # # SA_elast_stress = k_EA * sigma_a * z_cm * (SA/SA0 - 1) # elastic stress from whole surface:
                # # # SA_visc_stress =  k_eta_A * sigma_a * z_cm * (dSA_dtheta * theta_dot / SA0 - dSA0_dt*(SA/SA0**2)) # viscous stress from whole surface:
                # # is equal to the stress in the contact zone. Equating the two
                # # and through some algebra we get the reference contact surface:
                # z_cc = z_cont_func(t, *z_cont_args)
                # z_cm = z_cm_func(t, *z_cm_args)
                # Scc0 = Scc / ((z_cm/z_cc) * (SA/SA0 - 1) + 1)
                # # eqn for reference non-contact surface is almost the same:
                # Scm0 = Scm / ((z_cm/z_cm) * (SA/SA0 - 1) + 1)
                
                # # # we are omitting the non-actin stresses because that network 
                # # # does not separate from whole surface to contact and noncontact.
                    
                # syst = [theta, Scc0, Scm0, SA0, SN0]

                
               
                theta, Scc0, Scm0, SA0, SNcc0, SNcm0, SN0 = syst

                SCap = spher_cap_geom(theta, R_s)
                SA, SN = SCap.SA, SCap.SN

                strain_actin = SA/SA0 - 1
                strain_nonactin = SN/SN0 - 1
                
                ## Just double-check the event statuses:
                if abs(strain_actin) <= (strain_lim_actin + strain_lim_event_stol):
                    strain_max_status_actin = False
                elif abs(strain_actin) >= (strain_lim_actin - strain_lim_event_stol):
                    strain_max_status_actin = True
                    
                if abs(strain_nonactin) <= (strain_lim_nonactin + strain_lim_event_stol):
                    strain_max_status_nonactin = False
                elif abs(strain_nonactin) >= (strain_lim_nonactin - strain_lim_event_stol):
                    strain_max_status_nonactin = True


                
            elif case_id == 4:
                theta_Gamma_dec_status = False
                theta_Gamma_inc_status = True
                soln = soln_Gamma_rolling_start
                t = soln.t[-1]
                syst = soln.y[:,-1]
                
                # theta, Scc0, Scm0, SA0, SNcc0, SNcm0, SN0 = syst
                
                # SCap = spher_cap_geom(theta, R_s)
                # Scc, Scm, SA = SCap.Scc, SCap.Scm, SCap.SA
                                
                # # # Since we have hit the limit of Gamma-induced growth, we 
                # # # must assign the stress from the total surface to the contact
                # # # and non-contact surfaces, and then set the reference angle
                # # # for each of these surfaces to be consistent with those 
                # # # stresses:
                # # # SA_elast_stress = k_EA * sigma_a * z_cm * (SA/SA0 - 1) # elastic stress from whole surface:
                # # # SA_visc_stress =  k_eta_A * sigma_a * z_cm * (dSA_dtheta * theta_dot / SA0 - dSA0_dt*(SA/SA0**2)) # viscous stress from whole surface:
                # # is equal to the stress in the contact zone. Equating the two
                # # and through some algebra we get the reference contact surface:
                # z_cc = z_cont_func(t, *z_cont_args)
                # z_cm = z_cm_func(t, *z_cm_args)
                # Scc0 = Scc / ((z_cm/z_cc) * (SA/SA0 - 1) + 1)
                # # eqn for reference non-contact surface is almost the same:
                # Scm0 = Scm / ((z_cm/z_cm) * (SA/SA0 - 1) + 1)
                
                # # # we are omitting the non-actin stresses because that network 
                # # # does not separate from whole surface to contact and noncontact.
                    
                # syst = [theta, Scc0, Scm0, SA0, SN0]


                
                theta, Scc0, Scm0, SA0, SNcc0, SNcm0, SN0 = syst

                SCap = spher_cap_geom(theta, R_s)
                SA, SN = SCap.SA, SCap.SN

                strain_actin = SA/SA0 - 1
                strain_nonactin = SN/SN0 - 1
                
                ## Just double-check the event statuses:
                if abs(strain_actin) <= (strain_lim_actin + strain_lim_event_stol):
                    strain_max_status_actin = False
                elif abs(strain_actin) >= (strain_lim_actin - strain_lim_event_stol):
                    strain_max_status_actin = True
                    
                if abs(strain_nonactin) <= (strain_lim_nonactin + strain_lim_event_stol):
                    strain_max_status_nonactin = False
                elif abs(strain_nonactin) >= (strain_lim_nonactin - strain_lim_event_stol):
                    strain_max_status_nonactin = True
            
            else:
                print('Unexpected behaviour while solving.')
                break
                
                
    
        elif (Gamma_exceeded == True) and (strain_max_status_actin == True) and (strain_max_status_nonactin == True):

            soln_actin_max = angular_veloc_strainlim_actin(t, syst, *consts,)
            soln_nonactin_max = angular_veloc_strainlim_nonactin(t, syst, *consts,)


            # if both actin and non-actin networks are limiting then check which has the
            # slowest growth rate, and that must be the model to use:
            ang_veloc_actin_max = soln_actin_max[0]
            ang_veloc_nonactin_max = soln_nonactin_max[0]
        
            if ang_veloc_actin_max < ang_veloc_nonactin_max: # then the contact surface is limiting
                color = 'tab:green'
                soln_theta_max_exceeded = solve_ivp(angular_veloc_strainlim_actin, (t, t_end), syst, args=args, method=solve_ivp_method, events=theta_max_event, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
                soln_theta_min_exceeded_end = solve_ivp(angular_veloc_strainlim_actin, (t, t_end), syst, args=args, method=solve_ivp_method, events=theta_min_event_start, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
                soln_strain_max_actin_end = solve_ivp(angular_veloc_strainlim_actin, (t, t_end), syst, args=args, method=solve_ivp_method, events=strain_max_actin_event_end, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
                soln_strain_max_nonactin_start = solve_ivp(angular_veloc_strainlim_actin, (t, t_end), syst, args=args, method=solve_ivp_method, events=strain_max_nonactin_event_start, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
                
                case_id = np.argmin([soln_theta_max_exceeded.t[-1], soln_theta_min_exceeded_end.t[-1], soln_strain_max_actin_end.t[-1], soln_strain_max_nonactin_start.t[-1]])
                
                if case_id == 0:
                    soln = soln_theta_max_exceeded
                    if soln.message == 'The solver successfully reached the end of the integration interval.':
                        pass
    
                    elif soln.message == 'A termination event occurred.':
                        theta_max_status = True
                        soln.message = 'The solver successfully reached the end of the integration interval.'
    
                    else:
                        print('Unexpected behaviour while solving.')
                        break
                
                elif case_id == 1: # then angle dropped below theta_min first:
                    theta_min_exceeded = False
                    soln = soln_theta_min_exceeded_end
                    
                    t = soln.t[-1]
                    syst = soln.y[:,-1]
                    theta, Scc0, Scm0, SA0, SNcc0, SNcm0, SN0 = syst

                    SCap = spher_cap_geom(theta, R_s)
                    Scc, SN = SCap.Scc, SCap.SN
                    
                    strain_actin = Scc/Scc0 - 1
                    strain_nonactin = SN/SN0 - 1
                    
                    ## Just double-check the event statuses:
                    if abs(strain_actin) <= (strain_lim_actin + strain_lim_event_stol):
                        strain_max_status_actin = False
                    elif abs(strain_actin) >= (strain_lim_actin - strain_lim_event_stol):
                        strain_max_status_actin = True
                        
                    if abs(strain_nonactin) <= (strain_lim_nonactin + strain_lim_event_stol):
                        strain_max_status_nonactin = False
                    elif abs(strain_nonactin) >= (strain_lim_nonactin - strain_lim_event_stol):
                        strain_max_status_nonactin = True


                
                elif case_id == 2: # strain_max_actin_end occurred first:       
                    strain_max_status_actin = False
                    soln = soln_strain_max_actin_end
                    
                    t = soln.t[-1]
                    syst = soln.y[:,-1]
                    
                    theta, Scc0, Scm0, SA0, SNcc0, SNcm0, SN0 = syst

                    SCap = spher_cap_geom(theta, R_s)
                    Scc, SN = SCap.Scc, SCap.SN
                    
                    strain_actin = Scc/Scc0 - 1
                    strain_nonactin = SN/SN0 - 1
                            
                    ## Just double-check the event statuses:
                    if theta < theta_min:
                        theta_min_exceeded = False
                    elif theta >= theta_min:
                        theta_min_exceeded = True
                    
                        
                    if abs(strain_nonactin) <= (strain_lim_nonactin + strain_lim_event_stol):
                        strain_max_status_nonactin = False
                    elif abs(strain_nonactin) >= (strain_lim_nonactin - strain_lim_event_stol):
                        strain_max_status_nonactin = True
        
                elif case_id == 3: # strain_max_nonactin_end_occurred first:
                    strain_max_status_nonactin = True
                    soln = soln_strain_max_nonactin_start
                    
                    t = soln.t[-1]
                    syst = soln.y[:,-1]
                    
                    theta, Scc0, Scm0, SA0, SNcc0, SNcm0, SN0 = syst

                    SCap = spher_cap_geom(theta, R_s)
                    Scc, SN = SCap.Scc, SCap.SN
                    
                    strain_actin = Scc/Scc0 - 1
                    strain_nonactin = SN/SN0 - 1
                    
                    
                    ## Just double-check the event statuses:
                    if theta < theta_min:
                        theta_min_exceeded = False
                    elif theta >= theta_min:
                        theta_min_exceeded = True
                    
                        
                    if abs(strain_actin) <= (strain_lim_actin + strain_lim_event_stol):
                        strain_max_status_actin = False
                    elif abs(strain_actin) >= (strain_lim_actin - strain_lim_event_stol):
                        strain_max_status_actin = True
                    
                        
                else:
                        print('Unexpected behaviour while solving.')
                        break
                            
            else: # the whole cell surface network is limiting
                color = 'tab:blue'
                soln_theta_max_exceeded = solve_ivp(angular_veloc_strainlim_nonactin, (t, t_end), syst, args=args, method=solve_ivp_method, events=theta_max_event, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
                soln_theta_min_exceeded_end = solve_ivp(angular_veloc_strainlim_nonactin, (t, t_end), syst, args=args, method=solve_ivp_method, events=theta_min_event_start, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
                soln_strain_max_actin_start = solve_ivp(angular_veloc_strainlim_nonactin, (t, t_end), syst, args=args, method=solve_ivp_method, events=strain_max_actin_event_start, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
                soln_strain_max_nonactin_end = solve_ivp(angular_veloc_strainlim_nonactin, (t, t_end), syst, args=args, method=solve_ivp_method, events=strain_max_nonactin_event_end, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
                
                case_id = np.argmin([soln_theta_max_exceeded.t[-1], soln_theta_min_exceeded_end.t[-1], soln_strain_max_actin_start.t[-1], soln_strain_max_nonactin_end.t[-1]])
                
                if case_id == 0:
                    soln = soln_theta_max_exceeded
                    if soln.message == 'The solver successfully reached the end of the integration interval.':
                        pass
    
                    elif soln.message == 'A termination event occurred.':
                        theta_max_status = True
                        soln.message = 'The solver successfully reached the end of the integration interval.'
    
                    else:
                        print('Unexpected behaviour while solving.')
                        break
                    
                elif case_id == 1: # then angle dropped below theta_min first:
                    theta_min_exceeded = False
                    soln = soln_theta_min_exceeded_end
                    
                    t = soln.t[-1]
                    syst = soln.y[:,-1]

                    theta, Scc0, Scm0, SA0, SNcc0, SNcm0, SN0 = syst

                    SCap = spher_cap_geom(theta, R_s)
                    Scc, SN = SCap.Scc, SCap.SN

                    strain_actin = Scc/Scc0 - 1
                    strain_nonactin = SN/SN0 - 1
                    
                    ## Just double-check the event statuses:
                    if abs(strain_actin) <= (strain_lim_actin + strain_lim_event_stol):
                        strain_max_status_actin = False
                    elif abs(strain_actin) >= (strain_lim_actin - strain_lim_event_stol):
                        strain_max_status_actin = True
                        
                    if abs(strain_nonactin) <= (strain_lim_nonactin + strain_lim_event_stol):
                        strain_max_status_nonactin = False
                    elif abs(strain_nonactin) >= (strain_lim_nonactin - strain_lim_event_stol):
                        strain_max_status_nonactin = True

                
                elif case_id == 2:
                    strain_max_status_actin = True
                    soln = soln_strain_max_actin_start
                    
                    t = soln.t[-1]
                    syst = soln.y[:,-1]
                    
                    theta, Scc0, Scm0, SA0, SNcc0, SNcm0, SN0 = syst

                    SCap = spher_cap_geom(theta, R_s)
                    Scc, SN = SCap.Scc, SCap.SN

                    strain_actin = Scc/Scc0 - 1
                    strain_nonactin = SN/SN0 - 1
                    
                    
                    ## Just double-check the event statuses:
                    if theta < theta_min:
                        theta_min_exceeded = False
                    elif theta >= theta_min:
                        theta_min_exceeded = True
                    
                        
                    if abs(strain_nonactin) <= (strain_lim_nonactin + strain_lim_event_stol):
                        strain_max_status_nonactin = False
                    elif abs(strain_nonactin) >= (strain_lim_nonactin - strain_lim_event_stol):
                        strain_max_status_nonactin = True
                        
                elif case_id == 3:
                    strain_max_status_nonactin = False
                    soln = soln_strain_max_nonactin_end
                    
                    t = soln.t[-1]
                    syst = soln.y[:,-1]
                    
                    theta, Scc0, Scm0, SN0 = syst

                    SCap = spher_cap_geom(theta, R_s)
                    Scc, SN = SCap.Scc, SCap.SN
                    
                    strain_actin = Scc/Scc0 - 1
                    strain_nonactin = SN/SN0 - 1
                    
                    
                    ## Just double-check the event statuses:
                    if theta < theta_min:
                        theta_min_exceeded = False
                    elif theta >= theta_min:
                        theta_min_exceeded = True
                    
                        
                    if abs(strain_actin) <= (strain_lim_actin + strain_lim_event_stol):
                        strain_max_status_actin = False
                    elif abs(strain_actin) >= (strain_lim_actin - strain_lim_event_stol):
                        strain_max_status_actin = True

                        
                else:
                    print('Unexpected behaviour while solving.')
                    break
                
        elif (Gamma_exceeded == True) and (strain_max_status_actin == True) and (strain_max_status_nonactin == False):
            # possible ways that angle can escape strain-limited actin growth:
            # actin is no longer strain-limited, 
            # nonactin network becomes the limiting factor instead
            color = 'tab:green'
            soln_theta_max_exceeded = solve_ivp(angular_veloc_strainlim_actin, (t, t_end), syst, args=args, method=solve_ivp_method, events=[theta_max_event, ], max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
            soln_theta_min_exceeded_end = solve_ivp(angular_veloc_strainlim_actin, (t, t_end), syst, args=args, method=solve_ivp_method, events=[theta_min_event_start,theta_max_event], max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
            soln_strain_max_actin_end = solve_ivp(angular_veloc_strainlim_actin, (t, t_end), syst, args=args, method=solve_ivp_method, events=[strain_max_actin_event_end,theta_max_event], max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
            soln_strain_max_nonactin_start = solve_ivp(angular_veloc_strainlim_actin, (t, t_end), syst, args=args, method=solve_ivp_method, events=[strain_max_nonactin_event_start,theta_max_event], max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
            
            case_id = np.argmin([soln_theta_max_exceeded.t[-1], soln_theta_min_exceeded_end.t[-1], soln_strain_max_actin_end.t[-1], soln_strain_max_nonactin_start.t[-1]])
            
            
            if case_id == 0: # then angle hit the max angle in all cases:
                soln = soln_theta_max_exceeded
                if soln.message == 'The solver successfully reached the end of the integration interval.':
                    pass

                elif soln.message == 'A termination event occurred.':
                    theta_max_status = True
                    soln.message = 'The solver successfully reached the end of the integration interval.'

                else:
                    print('Unexpected behaviour while solving.')
                    break
            
            elif case_id == 1: # then angle dropped below theta_min first:
                theta_min_exceeded = False
                soln = soln_theta_min_exceeded_end
                
                t = soln.t[-1]
                syst = soln.y[:,-1]
                
                
                theta, Scc0, Scm0, SA0, SNcc0, SNcm0, SN0 = syst

                SCap = spher_cap_geom(theta, R_s)
                Scc, SN = SCap.Scc, SCap.SN
                
                strain_actin = Scc/Scc0 - 1
                strain_nonactin = SN/SN0 - 1
                
                ## Just double-check the event statuses:
                if abs(strain_actin) <= (strain_lim_actin + strain_lim_event_stol):
                    strain_max_status_actin = False
                elif abs(strain_actin) >= (strain_lim_actin - strain_lim_event_stol):
                    strain_max_status_actin = True
                    
                if abs(strain_nonactin) <= (strain_lim_nonactin + strain_lim_event_stol):
                    strain_max_status_nonactin = False
                elif abs(strain_nonactin) >= (strain_lim_nonactin - strain_lim_event_stol):
                    strain_max_status_nonactin = True

            
            # if soln_strain_max_actin_end.t[-1] <= soln_strain_max_nonactin_start.t[-1]:
            elif case_id == 2:
                soln = soln_strain_max_actin_end
                strain_max_status_actin = False
                
                t = soln.t[-1]
                syst = soln.y[:,-1]
                
                theta, Scc0, Scm0, SA0, SNcc0, SNcm0, SN0 = syst

                SCap = spher_cap_geom(theta, R_s)
                Scc, SN = SCap.Scc, SCap.SN
                
                strain_actin = Scc/Scc0 - 1
                strain_nonactin = SN/SN0 - 1

            
                ## Just double-check the event statuses:
                if theta < theta_min:
                    theta_min_exceeded = False
                elif theta >= theta_min:
                    theta_min_exceeded = True
                
                    
                if abs(strain_nonactin) <= (strain_lim_nonactin + strain_lim_event_stol):
                    strain_max_status_nonactin = False
                elif abs(strain_nonactin) >= (strain_lim_nonactin - strain_lim_event_stol):
                    strain_max_status_nonactin = True
                    
            # else:
            elif case_id == 3:
                soln = soln_strain_max_nonactin_start
                strain_max_status_nonactin = True
                    
                t = soln.t[-1]
                syst = soln.y[:,-1]
                
                theta, Scc0, Scm0, SA0, SNcc0, SNcm0, SN0 = syst

                SCap = spher_cap_geom(theta, R_s)
                Scc, SN = SCap.Scc, SCap.SN
                
                strain_actin = Scc/Scc0 - 1
                strain_nonactin = SN/SN0 - 1
                
                ## Just double-check the event statuses:
                if theta < theta_min:
                    theta_min_exceeded = False
                elif theta >= theta_min:
                    theta_min_exceeded = True
                
                    
                if abs(strain_actin) <= (strain_lim_actin + strain_lim_event_stol):
                    strain_max_status_actin = False
                elif abs(strain_actin) >= (strain_lim_actin - strain_lim_event_stol):
                    strain_max_status_actin = True
                    
                    
            else:
                print('Unexpected behaviour while solving.')
                break
            
        elif (Gamma_exceeded == True) and (strain_max_status_actin == False) and (strain_max_status_nonactin == True):
            # possible ways that angle can escape strain-limited non-actin growth:
            # non-actin network is no longer strain-limited, 
            # actin network becomes the limiting factor instead
            # soln = solve_ivp(angular_veloc_strainlim_nonactin, (t, t_end), syst, args=args, method=solve_ivp_method, events=(strain_max_actin_event_start, strain_max_nonactin_event_end), max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
            color = 'tab:blue'
            soln_theta_max_exceeded = solve_ivp(angular_veloc_strainlim_nonactin, (t, t_end), syst, args=args, method=solve_ivp_method, events=theta_max_event, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
            soln_theta_min_exceeded_end = solve_ivp(angular_veloc_strainlim_nonactin, (t, t_end), syst, args=args, method=solve_ivp_method, events=theta_min_event_start, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
            soln_strain_max_actin_start = solve_ivp(angular_veloc_strainlim_nonactin, (t, t_end), syst, args=args, method=solve_ivp_method, events=strain_max_actin_event_start, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
            soln_strain_max_nonactin_end = solve_ivp(angular_veloc_strainlim_nonactin, (t, t_end), syst, args=args, method=solve_ivp_method, events=strain_max_nonactin_event_end, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
            
            case_id = np.argmin([soln_theta_max_exceeded.t[-1], soln_theta_min_exceeded_end.t[-1], soln_strain_max_actin_start.t[-1], soln_strain_max_nonactin_end.t[-1]])

            
            if case_id == 0: # then angle hit the max angle in all cases:
                soln = soln_theta_max_exceeded
                if soln.message == 'The solver successfully reached the end of the integration interval.':
                    pass

                elif soln.message == 'A termination event occurred.':
                    theta_max_status = True
                    soln.message = 'The solver successfully reached the end of the integration interval.'

                else:
                    print('Unexpected behaviour while solving.')
                    break
                
            elif case_id == 1: # then angle dropped below theta_min first:
                theta_min_exceeded = False
                soln = soln_theta_min_exceeded_end
                
                t = soln.t[-1]
                syst = soln.y[:,-1]
                
                
                theta, Scc0, Scm0, SA0, SNcc0, SNcm0, SN0 = syst

                SCap = spher_cap_geom(theta, R_s)
                Scc, SN = SCap.Scc, SCap.SN
                
                strain_actin = Scc/Scc0 - 1
                strain_nonactin = SN/SN0 - 1
                
                ## Just double-check the event statuses:
                if abs(strain_actin) <= (strain_lim_actin + strain_lim_event_stol):
                    strain_max_status_actin = False
                elif abs(strain_actin) >= (strain_lim_actin - strain_lim_event_stol):
                    strain_max_status_actin = True
                    
                if abs(strain_nonactin) <= (strain_lim_nonactin + strain_lim_event_stol):
                    strain_max_status_nonactin = False
                elif abs(strain_nonactin) >= (strain_lim_nonactin - strain_lim_event_stol):
                    strain_max_status_nonactin = True

            
            elif case_id == 2: 
                strain_max_status_actin = True
                soln = soln_strain_max_actin_start
                
                t = soln.t[-1]
                syst = soln.y[:,-1]
                
                theta, Scc0, Scm0, SA0, SNcc0, SNcm0, SN0 = syst

                SCap = spher_cap_geom(theta, R_s)
                Scc, SN = SCap.Scc, SCap.SN

                strain_actin = Scc/Scc0 - 1
                strain_nonactin = SN/SN0 - 1
                
                ## Just double-check the event statuses:
                if theta < theta_min:
                    theta_min_exceeded = False
                elif theta >= theta_min:
                    theta_min_exceeded = True
                
                    
                if abs(strain_nonactin) <= (strain_lim_nonactin + strain_lim_event_stol):
                    strain_max_status_nonactin = False
                elif abs(strain_nonactin) >= (strain_lim_nonactin - strain_lim_event_stol):
                    strain_max_status_nonactin = True

            elif case_id == 3: 
                strain_max_status_nonactin = False
                soln = soln_strain_max_nonactin_end
                
                t = soln.t[-1]
                syst = soln.y[:,-1]
                
                theta, Scc0, Scm0, SA0, SNcc0, SNcm0, SN0 = syst

                SCap = spher_cap_geom(theta, R_s)
                Scc, SN = SCap.Scc, SCap.SN
                
                strain_actin = Scc/Scc0 - 1
                strain_nonactin = SN/SN0 - 1
                
                ## Just double-check the event statuses:
                if theta < theta_min:
                    theta_min_exceeded = False
                elif theta >= theta_min:
                    theta_min_exceeded = True
                
                    
                if abs(strain_actin) <= (strain_lim_actin + strain_lim_event_stol):
                    strain_max_status_actin = False
                elif abs(strain_actin) >= (strain_lim_actin - strain_lim_event_stol):
                    strain_max_status_actin = True
                    
                    
            else:
                print('Unexpected behaviour while solving.')
                break

        elif (Gamma_exceeded == True) and (strain_max_status_actin == False) and (strain_max_status_nonactin == False):
            color = 'k'
            soln_theta_max_exceeded = solve_ivp(angular_veloc, (t, t_end), syst, args=args, method=solve_ivp_method, events=theta_max_event, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
            soln_theta_min_exceeded_end = solve_ivp(angular_veloc, (t, t_end), syst, args=args, method=solve_ivp_method, events=theta_min_event_start, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
            soln_strain_max_actin_start = solve_ivp(angular_veloc, (t, t_end), syst, args=args, method=solve_ivp_method, events=strain_max_actin_event_start, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
            soln_strain_max_nonactin_start = solve_ivp(angular_veloc, (t, t_end), syst, args=args, method=solve_ivp_method, events=strain_max_nonactin_event_start, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
            ## This 1 would be needed for low adhesion analysis but it's too complicated so omit:
            ## if Gamma == 0 then the event 'theta_exceeds_Gamma_dec_event_end'
            ## can wind up with errors. soln_Gamma_exceeded_end can only be the
            ## case if Gamma exists in the first place:
            if calcium == True:
                if Gamma:
                    soln_Gamma_rolling_start = solve_ivp(angular_veloc, (t, t_end), syst, args=args, method=solve_ivp_method, events=angular_veloc_Gamma_inc_event_start, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
                else:
                    soln_Gamma_rolling_start = solve_ivp(quicksolve, (t, t_end), syst, args=args, method=solve_ivp_method, events=None, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
                    soln_Gamma_rolling_start.t[-1] = np.inf
                soln_Gamma_peeling_start = solve_ivp(quicksolve, (t, t_end), syst, args=args, method=solve_ivp_method, events=None, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
                soln_Gamma_peeling_start.t[-1] = np.inf
                
            elif calcium == False:
                soln_Gamma_rolling_start = solve_ivp(angular_veloc, (t, t_end), syst, args=args, method=solve_ivp_method, events=angular_veloc_Gamma_inc_event_start, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
                soln_Gamma_peeling_start = solve_ivp(angular_veloc, (t, t_end), syst, args=args, method=solve_ivp_method, events=angular_veloc_Gamma_dec_event_start, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
            
            # if theta_Gamma_inc_status == True:
            #     soln_Gamma_rolling_start = solve_ivp(angular_veloc, (t, t_end), syst, args=args, method=solve_ivp_method, events=angular_veloc_Gamma_inc_event_start, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
            # else:
            #     soln_Gamma_rolling_start = solve_ivp(quicksolve, (t, t_end), syst, args=args, method=solve_ivp_method, events=None, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
            #     soln_Gamma_rolling_start.t[-1] = np.inf
            # if theta_Gamma_dec_status == True:
            #     # soln_Gamma_dec_exceeded_end = solve_ivp(angular_veloc, (t, t_end), syst, args=args, method=solve_ivp_method, events=theta_exceeds_Gamma_dec_event_end, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
            #     soln_Gamma_peeling_start = solve_ivp(angular_veloc, (t, t_end), syst, args=args, method=solve_ivp_method, events=angular_veloc_Gamma_dec_event_start, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
            # else:
            #     soln_Gamma_peeling_start = solve_ivp(quicksolve, (t, t_end), syst, args=args, method=solve_ivp_method, events=None, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
            #     soln_Gamma_peeling_start.t[-1] = np.inf
                
            case_id = np.argmin([soln_theta_max_exceeded.t[-1], soln_theta_min_exceeded_end.t[-1], soln_strain_max_actin_start.t[-1], soln_strain_max_nonactin_start.t[-1], soln_Gamma_rolling_start.t[-1], soln_Gamma_peeling_start.t[-1]])
            ## case_id:
            ## 0 == soln_theta_max_exceeded
            ## 1 == soln_theta_min_exceeded_end
            ## 2 == soln_strain_max_actin_start
            ## 3 == soln_strain_max_nonactin_start
            ## 4 == soln_Gamma_rolling_start
            ## 5 == soln_Gamma_peeling_start
            
            
            
            if case_id == 0: # then angle hit the max angle in all cases:
                soln = soln_theta_max_exceeded
                if soln.message == 'The solver successfully reached the end of the integration interval.':
                    pass

                elif soln.message == 'A termination event occurred.':
                    theta_max_status = True
                    soln.message = 'The solver successfully reached the end of the integration interval.'

                else:
                    print('Unexpected behaviour while solving.')
                    break

            elif case_id == 1:
                theta_min_exceeded = False
                soln = soln_theta_min_exceeded_end
                
                t = soln.t[-1]
                syst = soln.y[:,-1]
                
                
                theta, Scc0, Scm0, SA0, SNcc0, SNcm0, SN0 = syst

                SCap = spher_cap_geom(theta, R_s)
                Scc, SN = SCap.Scc, SCap.SN
                
                strain_actin = Scc/Scc0 - 1
                strain_nonactin = SN/SN0 - 1
                
                ## Just double-check the event statuses:
                if abs(strain_actin) <= (strain_lim_actin + strain_lim_event_stol):
                    strain_max_status_actin = False
                elif abs(strain_actin) >= (strain_lim_actin - strain_lim_event_stol):
                    strain_max_status_actin = True
                    
                if abs(strain_nonactin) <= (strain_lim_nonactin + strain_lim_event_stol):
                    strain_max_status_nonactin = False
                elif abs(strain_nonactin) >= (strain_lim_nonactin - strain_lim_event_stol):
                    strain_max_status_nonactin = True
            

            elif case_id == 2:
                strain_max_status_actin = True
                soln = soln_strain_max_actin_start
                
                t = soln.t[-1]
                syst = soln.y[:,-1]
                
                theta, Scc0, Scm0, SA0, SNcc0, SNcm0, SN0 = syst

                SCap = spher_cap_geom(theta, R_s)
                Scc, SN = SCap.Scc, SCap.SN
                
                strain_actin = Scc/Scc0 - 1
                strain_nonactin = SN/SN0 - 1
                
                ## Just double-check the event statuses:
                if theta < theta_min:
                    theta_min_exceeded = False
                elif theta >= theta_min:
                    theta_min_exceeded = True
                
                    
                if abs(strain_nonactin) <= (strain_lim_nonactin + strain_lim_event_stol):
                    strain_max_status_nonactin = False
                elif abs(strain_nonactin) >= (strain_lim_nonactin - strain_lim_event_stol):
                    strain_max_status_nonactin = True
                    
            elif case_id == 3:
                strain_max_status_nonactin = True
                soln = soln_strain_max_nonactin_start
                
                t = soln.t[-1]
                syst = soln.y[:,-1]
                
                theta, Scc0, Scm0, SA0, SNcc0, SNcm0, SN0 = syst

                SCap = spher_cap_geom(theta, R_s)
                Scc, SN = SCap.Scc, SCap.SN
                
                strain_actin = Scc/Scc0 - 1
                strain_nonactin = SN/SN0 - 1
                
                ## Just double-check the event statuses:
                if theta < theta_min:
                    theta_min_exceeded = False
                elif theta >= theta_min:
                    theta_min_exceeded = True
                
                    
                if abs(strain_actin) <= (strain_lim_actin + strain_lim_event_stol):
                    strain_max_status_actin = False
                elif abs(strain_actin) >= (strain_lim_actin - strain_lim_event_stol):
                    strain_max_status_actin = True
                    
                    
                    
            ## This 1 would be needed for low adhesion analysis but it's too complicated so omit:
            elif case_id == 4 or case_id == 5:
                Gamma_exceeded = False
                
                if case_id == 4:
                    theta_Gamma_inc_status = True
                    theta_Gamma_dec_status = False
                    soln = soln_Gamma_rolling_start
                    
                elif case_id == 5:
                    theta_Gamma_dec_status = True
                    theta_Gamma_inc_status = False
                    soln = soln_Gamma_peeling_start
                
                t = soln.t[-1]
                syst = soln.y[:,-1]
                
                theta, Scc0, Scm0, SA0, SNcc0, SNcm0, SN0 = syst

                SCap = spher_cap_geom(theta, R_s)
                Scc, SN = SCap.Scc, SCap.SN
                
                strain_actin = Scc/Scc0 - 1
                strain_nonactin = SN/SN0 - 1
                
                ## Just double-check the event statuses:
                if abs(strain_actin) <= (strain_lim_actin + strain_lim_event_stol):
                    strain_max_status_actin = False
                elif abs(strain_actin) >= (strain_lim_actin - strain_lim_event_stol):
                    strain_max_status_actin = True
                    
                if abs(strain_nonactin) <= (strain_lim_nonactin + strain_lim_event_stol):
                    strain_max_status_nonactin = False
                elif abs(strain_nonactin) >= (strain_lim_nonactin - strain_lim_event_stol):
                    strain_max_status_nonactin = True
                    
                
                # theta_Gamma_dot_inc = get_theta_dot_rolling_inc(t, syst, *args) # if angular velocity according to Gamma is increasing still then Gamma_exceeded must be False
                # if theta_Gamma_dot_inc > 0:
                #     theta_Gamma_inc_status = True
                #     theta_Gamma_dec_status = False
                # elif theta_Gamma_dot_inc < 0:
                #     theta_Gamma_inc_status = False
                # elif theta_Gamma_dot_inc == 0:
                #     theta_Gamma_inc_status = False
                    
                # theta_Gamma_dot_dec = get_theta_dot_rolling_dec(t, syst, *args) # if angular velocity according to Gamma is increasing still then Gamma_exceeded must be False
                # if calcium == True:
                #     theta_Gamma_dec_status = False
                    
                # else:
                #     if theta_Gamma_dot_dec < 0:
                #         theta_Gamma_dec_status = True
                #         theta_Gamma_inc_status = False
                        
                #     elif theta_Gamma_dot_dec > 0:
                #         theta_Gamma_dec_status = False
                    
                #     elif theta_Gamma_dot_dec == 0:
                #         theta_Gamma_dec_status = False
                
            
            else:
                print('Unexpected behaviour while solving.')
                break


        
        ts.append(soln.t)
        solns.append(soln.y)
        messages.append(soln.message)
        soln_statuses.append(soln.status)
        successes.append(soln.success)
        if soln.t_events is None:
            soln.t_events = []
        else:
            pass
        soln.t_events = [x for x in soln.t_events if np.any(x)]
        t_events.append(soln.t_events)
        theta_min_exceeded_status_list.append([theta_min_exceeded] * len(soln.t))
        Gamma_exceeded_status_list.append([Gamma_exceeded] * len(soln.t))
        strain_max_status_actin_list.append([strain_max_status_actin] * len(soln.t))
        strain_max_status_nonactin_list.append([strain_max_status_nonactin] * len(soln.t))
        theta_max_status_list.append([theta_max_status] * len(soln.t))
        color_list.append([color] * len(soln.t))
        
        
        count += 1
    
        if soln.message == 'A termination event occurred.':
            continue
        elif soln.message == 'The solver successfully reached the end of the integration interval.':
            break
        else:            
            print('Unexpected behaviour in solve_ivp.')
            break
    
    soln.t = np.concatenate(ts)
    soln.y = np.concatenate(solns, axis=1)
    # soln.message = np.concatenate(messages)
    # soln.status = np.concatenate(soln_statuses)
    # soln.success = np.concatenate(successes)
    t_events = [np.ravel(x) for x in t_events]
    soln.t_events = np.concatenate(t_events)
    soln.theta_min_exceeded_status = np.concatenate(theta_min_exceeded_status_list)
    soln.Gamma_exceeded_status = np.concatenate(Gamma_exceeded_status_list)
    soln.strain_status_actin = np.concatenate(strain_max_status_actin_list)
    soln.strain_status_nonactin = np.concatenate(strain_max_status_nonactin_list)
    soln.color = np.concatenate(color_list)
    
    return soln, (messages, soln_statuses, successes, t_events)












###############################################################################
###############################################################################
###############################################################################
    
## Set some technical parameters used by solve_ivp:
solve_ivp_atol = 1e-9#1e-6 # tolerance used in solve_ivp
solve_ivp_rtol = 1e-6#1e-3 # tolerance used in solve_ivp
strain_lim_event_stol = 0 # 1e-4 # allowable tolerance for strain_max_actin and strain_max_nonactin events
solve_ivp_max_step = np.inf #1 #0.1
solve_ivp_method = 'BDF' #'RK45' #'BDF'




## Set constants:
# m_const = 1e2#1e-11 # according to my own estimate of an ectoderm cells mass, it has a mass of ~1e-11kg
R_s = 15#1.5e-5 # approximate radius of an ecto cell is ~15um=1.5e-5m
# beta = 1#8.6e-4 # according to David et al 2014, ecto cells have a cortical tension of ~0.86mJ/m**2=8.6e-4J/m**2 
z_cm = 1 # our Xenopus ectoderm cells have a cortex thickness of ~1um to ~1.5um
# sigma_a = 1 #beta / z_cm
# pr = 0.5 # Value of the Poisson ratio -> 0.5 == incompressible material
eta_m_real = 1e-21 #1e-3 J.s/m^3; 1e-21 J.s/um^3

# k_etam_const = 0.1


### Physiological constants:
## 1 J/m^3 = 1e-18 J/um^3
# R_s_real = 15 #um; approximate radius of an ecto cell is ~15um=1.5e-5m
# beta = 1#8.6e-4 # according to David et al 2014, ecto cells have a cortical 
# tension of ~0.86mJ/m**2 = 8.6e-4J/m**2 = 0.86e-16J/um^2
# z_cm_real = 1 #m our Xenopus ectoderm cells have a cortex thickness of ~1um to ~1.5um
sigma_a = 8.6e-16 # J/um^3 #sigma_a = beta / z_cm # 8.6e-16J/um^2 / 1um = 8.6e-16J/um^3
sigma_a_Jmcubed = 860 # J/m^3
# pr = 0.5 # Value of the Poisson ratio -> 0.5 == incompressible material
# Gamma = 0 #beta * 0.1 # for now asssume no adhesion energy
Gamma_null = 0 #beta * 0.1 # for now asssume no adhesion energy # Gamma
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


# eta, E = 0, 0
# k_E * sigma_a = E / (1 - nu) ; where nu = Poisson Ratio = 0.5
# the values for E here are measured in Pascals, therefore we used sigma_a_Jmcubed
# to figure out the value for kE 
# k_E_null = 0
# kEfor_E0 = 0 / (0.5 * sigma_a_Jmcubed)
# kEfor_E1e0 = 1e0 / (0.5 * sigma_a_Jmcubed)
# kEfor_E1e1 = 1e1 / (0.5 * sigma_a_Jmcubed)
# kEfor_E1e2 = 1e2 / (0.5 * sigma_a_Jmcubed)
# kEfor_E5e2 = 5e2 / (0.5 * sigma_a_Jmcubed)
# kEfor_E1e3 = 1e3 / (0.5 * sigma_a_Jmcubed)
# kEfor_E1e4 = 1e4 / (0.5 * sigma_a_Jmcubed)

kEfor_E0 = 0 * 1e-18
kEfor_E1e0 = 1e0 * 1e-18
kEfor_E1e1 = 1e1 * 1e-18
kEfor_E1e2 = 1e2 * 1e-18
kEfor_E5e2 = 5e2 * 1e-18
kEfor_E1e3 = 1e3 * 1e-18
kEfor_E1e4 = 1e4 * 1e-18




# k_eta_const = eta / sigma_a_real
# the values for eta here are measured in Pascals * seconds, therefore we used
# sigma_a_Jmcubed to figure out the value for k_eta
# k_eta_null = 0
# ketafor_eta0 = 0 / sigma_a_Jmcubed
# ketafor_eta1e0 = 1e0 / sigma_a_Jmcubed
# ketafor_eta1e1 = 1e1 / sigma_a_Jmcubed
# ketafor_eta1e2 = 1e2 / sigma_a_Jmcubed
# ketafor_eta1e3 = 1e3 / sigma_a_Jmcubed
# ketafor_eta1e4 = 1e4 / sigma_a_Jmcubed
# ketafor_eta1e5 = 1e5 / sigma_a_Jmcubed

ketafor_eta0 = 0 * 1e-18
ketafor_eta1e0 = 1e0 * 1e-18
ketafor_eta1e1 = 1e1 * 1e-18
ketafor_eta1e2 = 1e2 * 1e-18
ketafor_eta1e3 = 1e3 * 1e-18
ketafor_eta1e4 = 1e4 * 1e-18
ketafor_eta1e5 = 1e5 * 1e-18


# k_etam_real = c * eta_m_real / sigma_a
# the value for eta_m_real here is in J.s/um^3, therefore we used sigma_a to
# calculate k_etam_real since sigma_a is in units of J/um^3
# k_etam_real = eta_m_real / sigma_a
k_etam_real = eta_m_real

strain_lim_high = 1e3
strain_lim_med = 0.2
strain_lim_low = 0.1
strain_lim_vlow = 0.01
strain_lim_vvlow = 0.001

## From Valentine et al. 2005: elasticity of Xenopus oocyte extract is about 
## 2-10 Pa. We're going to do a conversion since all the simulation stresses
## are in J/um**3
E_B_null = 0 # this ones easy. 
E_B_example = 6e-18 # (3 J/m**3) * (1m**3/1e18um**3) = 3e-18 J/um**3
# E_b_low = 2e-18 # (2 J/m**3) * (1m**3/1e18um**3) = 2e-18 J/um**3
# E_b_med = 6e-18 # (2 J/m**3) * (1m**3/1e18um**3) = 2e-18 J/um**3
# E_b_high = 1e-17 # (2 J/m**3) * (1m**3/1e18um**3) = 2e-18 J/um**3


## From David et al 2014; Development 141 : 
## characteristic velocity was 1.8 +/- 0.9 um/min 
## therefore the characteristic velocity ranges from 0.9 to 2.7 um/min, so...
## 0.81 um**2/min * 1min/60s = 0.0135
## to 3.6 um**2/min * 1min/60s = 0.06
## 7.29 um**2/min * 1min/60s = 0.1215
# vc_low = 0.0135
# vc_mid = 0.06 # 1/1um**2 * 3.6um**2/min * 1min/60s = 0.06sec**-1 
# vc_high = 0.1215

# vc_low = 0.9 * 1/60 / 2 
# vc_mid = 1.8 * 1/60 / 2
# vc_high = 2.7 * 1/60 / 2
# divide by 2 because David et al 2014 used rates of whole contact lengths 
# (ie. diameter), not contact radii, so need to do conversion here. 
# vc_v_high = 1e6 # this is just a value that is so high that the rate of contact growth is sure not to surpass it



# Set initial contact angle (needs to be non-zero because if zero then get a 
# mathematical singularity which results in a divide by zero error):
# theta_eq_min = np.deg2rad(1e0) # theta_eq cannot be zero or we get a singularity
# phi_init = np.deg2rad(0)
theta_init = np.deg2rad(1) #theta cannot be zero or we will get a singularity
# theta0cc_init, theta0cm_init, theta0N_init = theta_init, theta_init, theta_init
Rc_init = (4**(1/3) * R_s) / (4 - (1 - np.cos(theta_init))**2 * (2 + np.cos(theta_init)))**(1/3)
Scc0_init = np.pi * (Rc_init * np.sin(theta_init))**2
Scm0_init = 2 * np.pi * Rc_init**2 * (1 + np.cos(theta_init))
SA0_init = Scc0_init + Scm0_init
SNcc0_init = np.pi * (Rc_init * np.sin(theta_init))**2
SNcm0_init = 2 * np.pi * Rc_init**2 * (1 + np.cos(theta_init))
SN0_init = Scc0_init + Scm0_init



# Set the initial thickness that the contact cortex starts at:
z_init = z_cm * np.cos(theta_init) #+ Gamma_null/(2 * sigma_a)
# Set a lower bound for the contact cortex (my data is roughly a quarter of 
# the non-contact actin cortex):
z_plat = 0.20 * z_cm
# Set half-life of cortex thickness decay (my data shows some to be about half
# the brightness after around 2 minutes, though there is variability):
t_half = 120 # seconds


z_cc_func = z_decay
z_cc_func_args = (z_init, t_half, z_plat)

z_cm_func = z_const
## set cortex thickness to be 1um, constant
z_cm_func_args = (1,) 

# z_cm_func = z_rand
# # ## set cortex thickness to randomly fluctuate about 1um, with a std of 0.26
# # ## data acquisition rate was 1frame/30seconds, acquisition duration was 
# z_cm_func_args = (1.0, 0.26, data_acquisition_rate, acquisition_duration, 24) # 24 equals the random_state
## include z_init as the first datapoint in z_cc_rand and 1 as the first datapoint in z_cm_rand?
## how would I achieve this?






## Some realistic model cases:
## 'H' stands for 'high value', 'L' stands for 'low value', 'M' for 'medium':
# elasticity = 90Pa, Poisson number = 0.5, sigma_a = 860J/m^3
# k_E_actin_H = 90/(1 - 0.5) * (860**-1)
# k_E_microtub_H = 15/(1 - 0.5) * (860**-1)
# k_E_intermfil_H = 240/(1 - 0.5) * (860**-1)
# k_E_multinetwork_L = 100/(1 - 0.5) * (860**-1)
# # k_E_multinetwork_LM = 200/(1 - 0.5) * (860**-1)
# k_E_multinetwork_M = 623/(1 - 0.5) * (860**-1)
# k_E_multinetwork_H = 1400/(1 - 0.5) * (860**-1)
# # k_E_multinetwork_vH = 42e3/(1 - 0.5) * (860**-1) # yes, that is 42kPa... see Cartagena-Rivera et al 2016

# # viscosity = 0Pa.s, sigma_a = 860J/m^3
# k_eta_actin = 0 / 860
# k_eta_microtub = 0 / 860
# k_eta_intermfil = 0 / 860
# k_eta_multinetwork = 30e-3 / 860 # 30mPa.s or 30e-3Pa.s

k_E_actin_H = 90 * 1e-18
k_E_microtub_H = 15 * 1e-18
k_E_intermfil_H = 240 * 1e-18
k_E_multinetwork_L = 100 * 1e-18
# k_E_multinetwork_LM = 200 * 1e-18
k_E_multinetwork_M = 623 * 1e-18
k_E_multinetwork_H = 1400 * 1e-18
# k_E_multinetwork_vH = 42e3/(1 - 0.5) * (860**-1) # yes, that is 42kPa... see Cartagena-Rivera et al 2016

# viscosity = 0Pa.s, sigma_a = 860J/m^3
k_eta_actin = 0 * 1e-18
k_eta_microtub = 0 * 1e-18
k_eta_intermfil = 0 * 1e-18
k_eta_multinetwork = 30e-3  * 1e-18




halflife_actin = 8.6 # seconds
halflife_microtub = None # not sure, data for this is seems to be missing
halflife_low = 60 # seconds
halflife_intermfil_L = 194# seconds
halflife_intermfil_M = 490# seconds
halflife_intermfil_H = 954 # seconds

strain_lim_actin = 0.2 # unitless
strain_lim_microtub = 0.5
strain_lim_intermfil = 2.5#1.0


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
## [[Gamma, strain_lim_actin, t_Apt5, k_EA, k_eta_A, strain_lim_nonactin, t_Npt5, k_EN, k_eta_N, E_r],
##  ...
##  ]

# Fig.S4A: Increasing contact viscosity:
figS4A_params = [[Gamma_null, strain_lim_high, turnover_rate_vfast, 0, ketafor_eta0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_null], # no adhesion energy, no deformation limit, no viscoelasticity
                  [G20pct, strain_lim_high, turnover_rate_vfast, 0, ketafor_eta1e2, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], # no adhesion energy, no deformation limit, medium actin viscosity
                  [G20pct, strain_lim_high, turnover_rate_vfast, 0, ketafor_eta1e3, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], # no adhesion energy, no deformation limit, medium actin viscosity
                  [G20pct, strain_lim_high, turnover_rate_vfast, 0, ketafor_eta1e4, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], # no adhesion energy, no deformation limit, high actin viscosity
                  [G20pct, strain_lim_high, turnover_rate_vfast, 0, ketafor_eta1e5, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], # no adhesion energy, no deformation limit, high actin viscosity
                  [Gamma_null, strain_lim_high, turnover_rate_vfast, 0, ketafor_eta1e5, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_null], # no adhesion energy, no deformation limit, higher actin viscosity
                  ]

figS4B_params = [[Gamma_null, strain_lim_high, turnover_rate_vfast, 0, ketafor_eta0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_null], # no adhesion energy, no deformation limit, no viscoelasticity
                 [G20pct, strain_lim_high, turnover_rate_slow, 0, ketafor_eta1e2, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], # no adhesion energy, no deformation limit, medium actin viscosity
                 [G20pct, strain_lim_high, turnover_rate_slow, 0, ketafor_eta1e3, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], # no adhesion energy, no deformation limit, medium actin viscosity
                 [G20pct, strain_lim_high, turnover_rate_slow, 0, ketafor_eta1e4, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], # no adhesion energy, no deformation limit, high actin viscosity
                 [G20pct, strain_lim_high, turnover_rate_slow, 0, ketafor_eta1e5, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], # no adhesion energy, no deformation limit, high actin viscosity
                 [Gamma_null, strain_lim_high, turnover_rate_slow, 0, ketafor_eta1e5, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_null], # no adhesion energy, no deformation limit, higher actin viscosity
                 ]

# Fig.S4Aii: Increasing actin elasticity:
# figS4Aii_params = [[Gamma_null, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_null], # no adhesion energy, no deformation limit, no viscoelasticity
#                    [Gamma_null, strain_lim_high, turnover_rate_vfast, 0.1, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_null], # no adhesion energy, no deformation limit, medium actin viscosity
#                    [Gamma_null, strain_lim_high, turnover_rate_vfast, 1, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_null], # no adhesion energy, no deformation limit, medium actin viscosity
#                    [Gamma_null, strain_lim_high, turnover_rate_vfast, 10, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_null], # no adhesion energy, no deformation limit, high actin viscosity
#                    [Gamma_null, strain_lim_high, turnover_rate_vfast, 100, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_null], # no adhesion energy, no deformation limit, higher actin viscosity
#                    ]

# Fig.S4B: rounding energy:
figS4C_params = [[Gamma_null, strain_lim_high, turnover_rate_fast, 0, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_null], 
                 [G20pct, strain_lim_high, turnover_rate_fast, 0, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_null], 
                 [Gamma_null, strain_lim_high, turnover_rate_fast, 0, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], # no adhesion energy, deformation limit similar to initial actin decrease rate
                 [G20pct, strain_lim_high, turnover_rate_fast, 0, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], # no adhesion energy, deformation limit similar to middling actin decrease rate
                 # [Gamma_null, strain_lim_high, turnover_rate_fast, 0, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_null], # no adhesion energy, deformation limit similar to initial actin decrease rate
                 ]

# Fig.S4Bii: decreasing actin turnover rate:
# figS4Bii_params = [[Gamma_null, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_null],
#                    [Gamma_null, strain_lim_med, turnover_rate_fast, 0, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_null], # no adhesion energy, deformation limit similar to initial actin decrease rate
#                    [Gamma_null, strain_lim_med, turnover_rate_med, 0, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_null], # no adhesion energy, deformation limit similar to middling actin decrease rate
#                    [Gamma_null, strain_lim_med, turnover_rate_slow, 0, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_null], # no adhesion energy, deformation limit similar to middling actin decrease rate
#                    ]

# Fig.S4C: Realistic Gamma values overcome initial heavy delays in contact growth:
# figS4C_params = [[Gamma_null, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_null], # no adhesion energy, no deformation limit, no viscoelasticity
#                  [G20pct, strain_lim_low, turnover_rate_vfast, 0, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_null], # no adhesion energy, deformation limit similar to initial actin decrease rate
#                  [G20pct, strain_lim_low, turnover_rate_fast, 0, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_null], # no adhesion energy, deformation limit similar to initial actin decrease rate
#                  [G20pct, strain_lim_low, turnover_rate_med, 0, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_null], # no adhesion energy, deformation limit similar to middling actin decrease rate
#                  [G20pct, strain_lim_low, turnover_rate_slow, 0, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_null], # no adhesion energy, deformation limit similar to middling actin decrease rate
#                  ]

# Fig.S4Cii: Realistic Gamma values overcome initial heavy delays in contact growth:
# figS4Cii_params = [[Gamma_null, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_null], # no adhesion energy, no deformation limit, no viscoelasticity
#                    [G20pct, strain_lim_high, turnover_rate_fast, 0, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_null], # no adhesion energy, deformation limit similar to initial actin decrease rate
#                    [G20pct, strain_lim_med, turnover_rate_fast, 0, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_null], # no adhesion energy, deformation limit similar to initial actin decrease rate
#                    [G20pct, strain_lim_low, turnover_rate_fast, 0, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_null], # no adhesion energy, deformation limit similar to middling actin decrease rate
#                    [G20pct, strain_lim_vlow, turnover_rate_fast, 0, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_null], # no adhesion energy, deformation limit similar to middling actin decrease rate
#                    ]

# Fig.S4ZL increasing actin-based viscosity:
# figS4Z_params = [[G20pct, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_null], # no adhesion energy, no deformation limit, medium actin viscosity
#                  [G20pct, strain_lim_high, turnover_rate_vfast, 0, 0.1, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_null], # no adhesion energy, no deformation limit, medium actin viscosity
#                  [G20pct, strain_lim_high, turnover_rate_vfast, 0, 1, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_null], # no adhesion energy, no deformation limit, high actin viscosity
#                  [G20pct, strain_lim_high, turnover_rate_vfast, 0, 10, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_null], # no adhesion energy, no deformation limit, high actin viscosity
#                  [G20pct, strain_lim_high, turnover_rate_vfast, 0, 100, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_null], # no adhesion energy, no deformation limit, higher actin viscosity
#                  ]

# Fig.S4D: rounding energy can return the equilibrium angle to empirical values:
# figS4D_params = [[Gamma_null, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_null], 
#                  [G20pct, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], 
#                  [G20pct, strain_lim_low, turnover_rate_vfast, 0, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], 
#                  [G20pct, strain_lim_low, turnover_rate_fast, 0, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], 
#                  [G20pct, strain_lim_low, turnover_rate_med, 0, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], 
#                  [G20pct, strain_lim_low, turnover_rate_slow, 0, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], 
#                  ]

# Fig.S4E increasing non-actin viscosity can slow growth as well, but must be quite high:
# figS4E_params = [[G20pct, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], 
#                  [G20pct, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_high, turnover_rate_vfast, 0, 10, E_B_example], 
#                  [G20pct, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_high, turnover_rate_vfast, 0, 100, E_B_example], 
#                  [G20pct, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_high, turnover_rate_vfast, 0, 1000, E_B_example], 
#                  [G20pct, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_high, turnover_rate_vfast, 0, 10000, E_B_example], 
#                  ]

# Fig.S4Eii increasing non-actin elasticity:
# figS4Eii_params = [[G20pct, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], 
#                    [G20pct, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_high, turnover_rate_vfast, 10, 0, E_B_example], 
#                    [G20pct, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_high, turnover_rate_vfast, 100, 0, E_B_example], 
#                    [G20pct, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_high, turnover_rate_vfast, 1000, 0, E_B_example], 
#                    ]

# Fig.3A - fast turnover half-life, increasing elasticity
fig3A_params = [[Gamma_null, strain_lim_high, turnover_rate_vfast, kEfor_E0, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_null], 
                # [Gamma_null, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], #use E_B_null?
                [G20pct, strain_lim_high, turnover_rate_slow, kEfor_E0, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], 
                [G20pct, strain_lim_high, turnover_rate_slow, kEfor_E1e1, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], 
                [G20pct, strain_lim_high, turnover_rate_slow, kEfor_E1e2, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], 
                [G20pct, strain_lim_high, turnover_rate_slow, kEfor_E5e2, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], 
                [G20pct, strain_lim_high, turnover_rate_slow, kEfor_E1e3, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], 
                [Gamma_null, strain_lim_high, turnover_rate_slow, kEfor_E1e3, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_null], 
                ]

# Fig. 3B - decreasing turnover half-life, mid elasticity
fig3B_params = [[Gamma_null, strain_lim_high, turnover_rate_vfast, kEfor_E0, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_null], 
                [G20pct, strain_lim_high, turnover_rate_vfast, kEfor_E5e2, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], 
                [G20pct, strain_lim_high, turnover_rate_fast, kEfor_E5e2, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], 
                [G20pct, strain_lim_high, turnover_rate_slow, kEfor_E5e2, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], 
                [G20pct, strain_lim_high, turnover_rate_vvvslow, kEfor_E5e2, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], 
                [Gamma_null, strain_lim_high, turnover_rate_vvslow, kEfor_E5e2, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_null], 
                # [G20pct, strain_lim_high, turnover_rate_fast, 0, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], 
                ]

# Fig. 3C - physiological solutions for contact surface stresses:
fig3C_params = [[Gamma_null, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_null], 
                [adh_energy_total_yesCalcium_M, strain_lim_actin, halflife_actin, k_E_actin_H, k_eta_actin, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], # actin
                [adh_energy_total_yesCalcium_M, strain_lim_microtub, halflife_actin, k_E_microtub_H, k_eta_microtub, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], # microtubules
                [adh_energy_total_yesCalcium_M, strain_lim_intermfil, halflife_intermfil_L, k_E_intermfil_H, k_eta_intermfil, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], # intermediate filaments low
                [adh_energy_total_yesCalcium_M, strain_lim_intermfil, halflife_intermfil_H, k_E_intermfil_H, k_eta_intermfil, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], # intermediate filaments high
                [adh_energy_total_yesCalcium_M, strain_lim_intermfil, halflife_intermfil_L, k_E_multinetwork_L, k_eta_multinetwork, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], # Xenopus multinetwork low
                [adh_energy_total_yesCalcium_M, strain_lim_intermfil, halflife_intermfil_M, k_E_multinetwork_M, k_eta_multinetwork, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], # Xenopus multinetwork medium
                [adh_energy_total_yesCalcium_M, strain_lim_intermfil, halflife_intermfil_H, k_E_multinetwork_H, k_eta_multinetwork, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], # Xenopus multinetwork high
                ]

# Fig. 3D - just Xenopus-like solutions for overlapping onto lifeact data:
fig3D_params = [[Gamma_null, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_null], 
                # [adh_energy_total_yesCalcium_M, strain_lim_actin, halflife_actin, k_E_actin_H, k_eta_actin, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], # actin
                # [adh_energy_total_yesCalcium_M, strain_lim_microtub, halflife_actin, k_E_microtub_H, k_eta_microtub, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], # microtubules
                # [adh_energy_total_yesCalcium_M, strain_lim_intermfil, halflife_intermfil_L, k_E_intermfil_H, k_eta_intermfil, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], # intermediate filaments low
                # [adh_energy_total_yesCalcium_M, strain_lim_intermfil, halflife_intermfil_H, k_E_intermfil_H, k_eta_intermfil, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], # intermediate filaments high
                [adh_energy_total_yesCalcium_M, strain_lim_intermfil, halflife_intermfil_L, k_E_multinetwork_L, k_eta_multinetwork, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], # Xenopus multinetwork low
                # [adh_energy_total_yesCalcium_M, strain_lim_intermfil, 120, k_E_multinetwork_M, k_eta_multinetwork, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], # Xenopus multinetwork medium
                [adh_energy_total_yesCalcium_M, strain_lim_intermfil, halflife_intermfil_M, k_E_multinetwork_L, k_eta_multinetwork, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], # Xenopus multinetwork medium
                [adh_energy_total_yesCalcium_M, strain_lim_intermfil, halflife_intermfil_L, k_E_multinetwork_M, k_eta_multinetwork, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], # Xenopus multinetwork medium
                [adh_energy_total_yesCalcium_M, strain_lim_intermfil, halflife_intermfil_M, k_E_multinetwork_M, k_eta_multinetwork, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], # Xenopus multinetwork medium
                [adh_energy_total_yesCalcium_M, strain_lim_intermfil, halflife_intermfil_H, k_E_multinetwork_H, k_eta_multinetwork, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], # Xenopus multinetwork high
                ]


# Fig. S4C - physiological solutions for whole cell surface stresses:
figS4D_params = [[Gamma_null, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_null], 
                 [adh_energy_total_yesCalcium_M, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_actin, halflife_actin, k_E_actin_H, k_eta_actin, E_B_example], # actin
                 # [adh_energy_total_yesCalcium_M, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_microtub, halflife_actin, k_E_microtub_H, k_eta_microtub, E_B_example], # microtubules
                 [adh_energy_total_yesCalcium_M, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_intermfil, halflife_intermfil_L, k_E_intermfil_H, k_eta_intermfil, E_B_example], # intermediate filaments low
                 # [adh_energy_total_yesCalcium_M, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_intermfil, halflife_intermfil_H, k_E_intermfil_H, k_eta_intermfil, E_B_example], # intermediate filaments high
                 [adh_energy_total_yesCalcium_M, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_intermfil, halflife_intermfil_L, k_E_multinetwork_L, k_eta_multinetwork, E_B_example], # Xenopus multinetwork low
                 [adh_energy_total_yesCalcium_M, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_intermfil, halflife_intermfil_M, k_E_multinetwork_M, k_eta_multinetwork, E_B_example], # Xenopus multinetwork medium
                 # [adh_energy_total_yesCalcium_M, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_intermfil, halflife_intermfil_H, k_E_multinetwork_H, k_eta_multinetwork, E_B_example], # Xenopus multinetwork high
                 ]

# figS4C_params = [[adh_energy_total_yesCalcium_M, strain_lim_actin, turnover_rate_vfast, kEforE0, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], # actin
#                  [adh_energy_total_yesCalcium_M, strain_lim_microtub, turnover_rate_vfast, kEforE500, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], # microtubules
#                  [adh_energy_total_yesCalcium_M, strain_lim_intermfil, turnover_rate_fast, kEforE500, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], # intermediate filaments low
#                  [adh_energy_total_yesCalcium_M, strain_lim_intermfil, turnover_rate_slow, kEforE500, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], # intermediate filaments high
#                  [adh_energy_total_yesCalcium_M, strain_lim_intermfil, turnover_rate_vvvslow, kEforE500, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], # Xenopus multinetwork low
#                  [adh_energy_total_yesCalcium_M, strain_lim_intermfil, turnover_rate_vvvslow, kEforE500, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], # Xenopus multinetwork medium
#                  [adh_energy_total_yesCalcium_M, strain_lim_intermfil, turnover_rate_vvvslow, kEforE500, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], # Xenopus multinetwork high
#                 ]


# fig3Z_params = [[Gamma_null, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_high, turnover_rate_vfast, 0, 1000, E_B_null], 
#                 [G20pct, strain_lim_low, turnover_rate_vfast, 0, 0, strain_lim_high, turnover_rate_vfast, 0, 1000, E_B_example], 
#                 [G20pct, strain_lim_low, turnover_rate_fast, 0, 0, strain_lim_high, turnover_rate_vfast, 0, 1000, E_B_example], 
#                 [G20pct, strain_lim_low, turnover_rate_med, 0, 0, strain_lim_high, turnover_rate_vfast, 0, 1000, E_B_example], 
#                 ]

# Extra figure: varying strain limits and turnover rates of actin network:
# figX_params = [[G20pct, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], 
#                [G20pct, strain_lim_med, turnover_rate_vfast, 0, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], 
#                [G20pct, strain_lim_low, turnover_rate_vfast, 0, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], 
#                [G20pct, strain_lim_vlow, turnover_rate_vfast, 0, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], 
#                # [G20pct, strain_lim_vlow*0.5, turnover_rate_vfast, 0, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], 
#                # [G20pct, strain_lim_vlow*0.1, turnover_rate_vfast, 0, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], 
#                [G20pct, strain_lim_med, turnover_rate_slow, 0, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], 
#                [G20pct, strain_lim_low, turnover_rate_slow, 0, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], 
#                [G20pct, strain_lim_vlow, turnover_rate_slow, 0, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], 
#                # [G20pct, strain_lim_vlow*0.5, turnover_rate_slow, 0, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], 
#                # [G20pct, strain_lim_vlow*0.1, turnover_rate_slow, 0, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], 
#                ]

# Extra figure: decreasing non-actin strain-limits:
# figY_params = [[G20pct, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], 
#                [G20pct, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_low, turnover_rate_med, 0, 0, E_B_example], 
#                [G20pct, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_vlow, turnover_rate_med, 0, 0, E_B_example], 
#                [G20pct, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_vvlow, turnover_rate_med, 0, 0, E_B_example], 
#                ]
     
# Extra figure: varying non-actin network turnover rates with low strain limits:
# figZ_params = [[G20pct, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], 
#                [G20pct, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_vvlow, turnover_rate_fast, 0, 0, E_B_example], 
#                [G20pct, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_vvlow, turnover_rate_med, 0, 0, E_B_example], 
#                [G20pct, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_vvlow, turnover_rate_slow, 0, 0, E_B_example], 
#                [G20pct, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_vvlow, turnover_rate_vslow, 0, 0, E_B_example], 
#                ]


# all_fig_params = np.concatenate(figS4A_params, figS4B_params, ...)
# all_fig_params_full = figS4A_params + figS4B_params + figS4Bii_params + figS4C_params + figS4Cii_params + figS4Z_params + figS4D_params + figS4E_params + fig3A_params + fig3B_params + fig3Z_params + figX_params + figY_params
all_fig_params = [figS4A_params
                  # + figS4Aii_params
                  + figS4B_params 
                  # + figS4Bii_params 
                  + figS4C_params 
                  # + figS4Cii_params 
                  # + figS4Z_params 
                   + figS4D_params 
                  # + figS4E_params 
                  # + figS4Eii_params 
                  + fig3A_params 
                  + fig3B_params 
                  + fig3C_params
                  + fig3D_params
                  # + fig3Z_params 
                  # + figX_params 
                  # + figY_params 
                  # + figZ_params
                  ][0]
# remove duplicates from all_fig_params before solving the param space
# all_fig_params = np.unique(all_fig_params_full, axis=0)
params_w_fig_id = {'figS4A':figS4A_params, 
                   # 'figS4Aii':figS4Aii_params, 
                   'figS4B':figS4B_params,
                   # 'figS4Bii':figS4Bii_params,
                   'figS4C':figS4C_params,
                   # 'figS4Cii':figS4Cii_params,
                   # 'figS4Z':figS4Z_params,
                    'figS4D':figS4D_params,
                   # 'figS4E':figS4E_params,
                   # 'figS4Eii':figS4Eii_params, 
                   'fig3A':fig3A_params,
                   'fig3B':fig3B_params,
                   'fig3C':fig3C_params,
                   'fig3D':fig3D_params,
                   # 'fig3Z':fig3Z_params,
                   # 'figX':figX_params,
                   # 'figY':figY_params,
                   # 'figZ':figZ_params
                   }








## parameters below are presented as tuples of (linecolor, linestyle, list_of_parameters)
## to facilitate plotting.
## The equilibrium model:
params_eq = ('k', ':', 'near eq.', [Gamma_null, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_null])


### Realisitc parameters assuming post-Gamma contact spreading through stretching:
params_actin_only = ('tab:green', '-', 'actin', [adh_energy_total_yesCalcium_M, strain_lim_actin, halflife_actin, k_E_actin_H, k_eta_actin, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example])
# In the absence of any information we will assume that microtubules are comparable to actin turnover rates
params_microtub_only = ('tab:orange', '-', 'MT', [adh_energy_total_yesCalcium_M, strain_lim_microtub, halflife_actin, k_E_microtub_H, k_eta_microtub, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example])
params_intermfil_only_L = ('tab:blue', ':', 'IF (low)', [adh_energy_total_yesCalcium_M, strain_lim_intermfil, halflife_intermfil_L, k_E_intermfil_H, k_eta_intermfil, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example])
params_intermfil_only_H = ('tab:blue', '-', 'IF (high)', [adh_energy_total_yesCalcium_M, strain_lim_intermfil, halflife_intermfil_H, k_E_intermfil_H, k_eta_intermfil, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example])
# params_intermfil_only_vH = ('tab:blue', '-', [adh_energy_total_yesCalcium_M, strain_lim_high, halflife_intermfil_L, k_E_intermfil_H, k_eta_intermfil, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example])
# In the absence of multinetwork turnover rate and strain limit information, we
# will assume that the network can go up to the highest known 
# (ie. the intermediate filament network):
params_multinetwork_L = ('tab:brown', ':', r'$\it{Xen.}$ E_{low}, IF $\lambda$_{low}', [adh_energy_total_yesCalcium_M, strain_lim_intermfil, halflife_intermfil_L, k_E_multinetwork_L, k_eta_multinetwork, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example])
params_multinetwork_M = ('tab:brown', '--', r'$\it{Xen.}$ E_{low}, IF $\lambda$_{low}', [adh_energy_total_yesCalcium_M, strain_lim_intermfil, halflife_intermfil_M, k_E_multinetwork_M, k_eta_multinetwork, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example])
params_multinetwork_Hh = ('tab:brown', (0, (3,2,1,2,1,2)), r'$\it{Xen.}$ E_{low}, IF $\lambda$_{med}', [adh_energy_total_yesCalcium_M, strain_lim_intermfil, halflife_intermfil_M, k_E_multinetwork_L, k_eta_multinetwork, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example])
params_multinetwork_He = ('tab:brown', '-.', r'$\it{Xen.}$ E_{med}, IF $\lambda$_{low}', [adh_energy_total_yesCalcium_M, strain_lim_intermfil, halflife_intermfil_L, k_E_multinetwork_M, k_eta_multinetwork, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example])
## the linestyle above "(0, (3,2,1,2,1,2))" specifies a dash-dot-dot (-..) linestyle, the other ones have convenient and obvious names so those are used there
## the numbers (3, 2, 1, 2, 1, 2) mean first line has length 3, then a space of length 2, then a line of length 1, then a space of length 2, etc. and it repeats when the end is reached
params_multinetwork_Heh = ('tab:brown', '-', r'$\it{Xen.}$ E_{high}, IF $\lambda$_{high}', [adh_energy_total_yesCalcium_M, strain_lim_intermfil, halflife_intermfil_H, k_E_multinetwork_H, k_eta_multinetwork, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example])

# params_multinetowrk_vvH = ('k', '-', [adh_energy_total_yesCalcium_H, strain_lim_high, halflife_intermfil_H, k_E_multinetwork_vH, k_eta_multinetwork, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example])

params_stretching = [params_eq, params_actin_only, params_microtub_only, 
                     params_intermfil_only_L, params_intermfil_only_H, #params_intermfil_only_vH, 
                     params_multinetwork_L, params_multinetwork_M,  
                     params_multinetwork_He, params_multinetwork_Hh, 
                     params_multinetwork_Heh
                     ]



### Realisitc parameters assuming post-Gamma contact spreading through rolling:
params_actin_only = ('tab:green', '-', 'actin', [adh_energy_total_yesCalcium_M, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_actin, halflife_actin, k_E_actin_H, k_eta_actin, E_B_example])
# In the absence of any information we will assume that microtubules are comparable to actin turnover rates
params_microtub_only = ('tab:orange', '-', 'MT', [adh_energy_total_yesCalcium_M, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_microtub, halflife_actin, k_E_microtub_H, k_eta_microtub, E_B_example])
params_intermfil_only_L = ('tab:blue', ':', 'IF (low)', [adh_energy_total_yesCalcium_M, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_intermfil, halflife_intermfil_L, k_E_intermfil_H, k_eta_intermfil, E_B_example])
params_intermfil_only_H = ('tab:blue', '-', 'IF (high)', [adh_energy_total_yesCalcium_M, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_intermfil, halflife_intermfil_H, k_E_intermfil_H, k_eta_intermfil, E_B_example])
# params_intermfil_only_vH = ('tab:blue', '-', [adh_energy_total_yesCalcium_M, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_high, halflife_intermfil_H, k_E_intermfil_H, k_eta_intermfil, E_B_example])
# In the absence of multinetwork turnover rate and strain limit information, we
# will assume that the network can go up to the highest known 
# (ie. the intermediate filament network):
params_multinetwork_L = ('tab:brown', '--', r'$\it{Xen.}$ E_{low}, IF $\lambda$_{low}', [adh_energy_total_yesCalcium_M, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_intermfil, halflife_intermfil_L, k_E_multinetwork_H, k_eta_multinetwork, E_B_example])
params_multinetwork_M = ('tab:brown', '-.', r'$\it{Xen.}$ E_{low}, IF $\lambda$_{low}', [adh_energy_total_yesCalcium_M, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_intermfil, halflife_intermfil_M, k_E_multinetwork_M, k_eta_multinetwork, E_B_example])
# params_multinetwork_Hh = ('k', '--', [adh_energy_total_yesCalcium_M, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_intermfil, halflife_intermfil_M, k_E_multinetwork_L, k_eta_multinetwork, E_B_example])
# params_multinetwork_He = ('k', '--', [adh_energy_total_yesCalcium_M, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_intermfil, halflife_low, k_E_multinetwork_M, k_eta_multinetwork, E_B_example])
params_multinetwork_Heh = ('tab:brown', '-', r'$\it{Xen.}$ E_{high}, IF $\lambda$_{high}', [adh_energy_total_yesCalcium_M, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_intermfil, halflife_intermfil_H, k_E_multinetwork_H, k_eta_multinetwork, E_B_example])
# params_multinetowrk_vvH = ('k', '-', [adh_energy_total_yesCalcium_M, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_high, halflife_intermfil_H, k_E_multinetwork_vH, k_eta_multinetwork, E_B_example])

params_rolling = [params_eq, params_actin_only, params_microtub_only, 
                  params_intermfil_only_L, params_intermfil_only_H, #params_intermfil_only_vH, 
                  params_multinetwork_L, params_multinetwork_M, 
                  # params_multinetwork_He, 
                  params_multinetwork_Heh
                  ]











soln_space_fig_params = list()
success_list_fig_params = list()
process_time_list_fig_params = list()
soln_space_fig_ids = list()

for fig_id in params_w_fig_id:
    # unpack parameters:
    for i,params in enumerate(params_w_fig_id[fig_id]):
        Gamma, strain_lim_actin, t_Apt5, k_EA, k_eta_A, strain_lim_nonactin, t_Npt5, k_EN, k_eta_N, E_r = params
        
        # Set initials:
        # syst = [theta_init, theta_init, theta_init]
        t, t_end = t_init, t_end # Set the range over which to solve.
        # syst = [theta_init, theta0cc_init, theta0cm_init, theta0N_init]
        syst = [theta_init, Scc0_init, Scm0_init, SA0_init, SNcc0_init, SNcm0_init, SN0_init]
        consts = (R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, k_EA, k_eta_A, k_EN, k_eta_N, k_etam_real, Gamma, E_r, z_cc_func, z_cc_func_args, z_cm_func, z_cm_func_args)
    
        # count = 0
        # count_lim = int((t_end - t) / dt) + 1
        
        print('Started angular_veloc_ve_model: %s -- no. %d/%d'%(fig_id, i+1, len(params_w_fig_id[fig_id])))
        time_start = tm.perf_counter()
        
        # consts = (R_s, deform_lim, deform_lim_So, sigma_a, k_E, k_eta, k_Eo, k_eta_o, k_etam_real, Gamma, z_cc_func, z_cc_func_args) #k_eta_null, Gamma_null
        # ts = list()
        # ys = list()
        # while count < count_lim:
        #     if count % int(count_lim*0.02) == 0: # Count every 2% so that I know how far along things are.
        #         print(count, ' / %d'%count_lim + ' -- (%.1f %%)'%(100*count/count_lim))
        #     soln = solve_ivp(angular_veloc_deformlim_actin_and_other_cytoskel, (t, t_end), syst, args=consts, method='BDF', events=None, max_step=max_step, atol=1e-6, rtol=1e-3, dense_output=True)
        #     ts.append(soln.t)
        #     ys.append(soln.y)
        #     if soln.status == -1: 
        #         t = soln.t[-1] + dt # move forward a small amount of time from the last timepoint
        #         syst = soln.y[:, -1].copy() # use the last 
        #     else:
        #         break
        #     count += 1
        # soln.t = np.concatenate(ts)
        # soln.y = np.concatenate(ys, axis=1)
        # plt.plot(soln.t, soln.y.T)
        
        
        soln, soln_details = angular_veloc_ve_model((t, t_end), syst, consts, solve_ivp_method, solve_ivp_max_step, solve_ivp_atol, solve_ivp_rtol, strain_lim_event_stol, calcium=True)
        
        ## some debugging code:
        # print(params)
        # plt.plot(soln.t, soln.y[0])
        # plt.show()
        ##
        
        soln_space_fig_params.append(soln)
        success_list_fig_params.append(soln.success)
        soln_space_fig_ids.append(fig_id)
           
        if soln.success == True:
            print('solve_ivp succeeded...')
        else:
            print('solve_ivp failed...')
            
        time_stop = tm.perf_counter()
        processing_time = time_stop - time_start
        process_time_list_fig_params.append(processing_time)
        print('Finished angular_veloc_ve_model: %s -- no. %d/%d :: processing time = %.2fsec'%(fig_id, i+1, len(params_w_fig_id[fig_id]), processing_time))

print('Finished solving param_space for manuscript plot params.\n\n')



man_plots_dir = Path.joinpath(out_dir, 'manuscript_plots')
if Path.exists(man_plots_dir) == False:
    Path.mkdir(man_plots_dir)
else:
    pass


## the low viscosity parameter examples are covering up too much, so we are
## going to make their lines dashed:
low_visc_params = [[G20pct, strain_lim_high, turnover_rate_vfast, 0, ketafor_eta1e2, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], 
                   [G20pct, strain_lim_high, turnover_rate_slow, 0, ketafor_eta1e2, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example]]

plt.close('all')
for fig_id in params_w_fig_id:

    cmap_count = 0 # each figure starts with a fresh colormap 
    fig, ax = plt.subplots(figsize=(width_1col*0.7, width_1col*0.7))

    for params_of_interest in params_w_fig_id[fig_id][::-1][:-1]: #reverse the list and take all but the last entry of that reversed list (ie. take all but the first entry in the list and reverse their order.)
        # print(params_of_interest) ## some debugging code
        for i, soln in enumerate(soln_space_fig_params):
      
            if np.all(all_fig_params[i] == params_of_interest) and (soln_space_fig_ids[i] == fig_id):
                ## The following condition just isolates the low viscosity
                ## examples so that they can be plotted with dashed lines,
                ## the other examples all get solid lines
                if params_of_interest in low_visc_params:
                    time = soln.t
                    # theta, theta0cc, theta0cm, theta0N = soln.y
                    theta, Scc0, Scm0, SA0, SNcc0, SNcm0, SN0 = soln.y
                    ax.plot(time, 2*np.rad2deg(theta), lw=1, c=cmap(cmap_count), ls='--')
                    cmap_count += 1 # each line that gets drawn gets a new color from the color map
                else:
                    time = soln.t
                    # theta, theta0cc, theta0cm, theta0N = soln.y
                    theta, Scc0, Scm0, SA0, SNcc0, SNcm0, SN0 = soln.y
                    ax.plot(time, 2*np.rad2deg(theta), lw=1, c=cmap(cmap_count))#, ls='--')
                    cmap_count += 1 # each line that gets drawn gets a new color from the color map
    params_of_interest = params_w_fig_id[fig_id][0]
    for i, soln in enumerate(soln_space_fig_params):
        if np.all(all_fig_params[i] == params_of_interest) and (soln_space_fig_ids[i] == fig_id):
            time = soln.t
            # theta, theta0cc, theta0cm, theta0N = soln.y
            theta, Scc0, Scm0, SA0, SNcc0, SNcm0, SN0 = soln.y
            z_cc = z_cc_func(time, *z_cc_func_args)
            ax.plot(time, 2*np.rad2deg(theta), lw=1, c='k', alpha=1.0, ls=':')
         
    ax.set_xlim(-10, 1800)
    ax.set_xticks([0,600,1200,1800])#,2400])
    ax.set_xticklabels(['0','600','1200','1800'])#,'2400'])
    ax.set_ylabel('Contact Angle (deg)')
    ax.set_xlabel('Time (s)')
            
    time_zcc = np.linspace(0, 3600, 36001)
    z_cc = z_cc_func(time, *z_cc_func_args)
    z_cm = z_cm_func(time, *z_cm_func_args)
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
    
    fig.savefig(Path.joinpath(man_plots_dir, r"%s_time_course.tif"%fig_id), dpi=dpi_LineArt, bbox_inches='tight')
    plt.close('all')
                
    
    
    fig, ax = plt.subplots(figsize=(width_1col*0.7,width_1col*0.7))
    ax.axhline(z_plat, c='grey', ls=':')
    ax.axvline(180, c='grey', ls=':')

    for params_of_interest in params_w_fig_id[fig_id][::-1][:-1]:
        for i, soln in enumerate(soln_space_fig_params):
            if np.all(all_fig_params[i] == params_of_interest) and (soln_space_fig_ids[i] == fig_id):
                time = soln.t
                # theta, theta0cc, theta0cm, theta0N = soln.y
                theta, Scc0, Scm0, SA0, SNcc0, SNcm0, SN0 = soln.y
                
                z_cc = z_cc_func(time, *z_cc_func_args)
                ax.plot(2*np.rad2deg(theta), z_cc/z_cm, lw=1)
    params_of_interest = params_w_fig_id[fig_id][0]
    for i, soln in enumerate(soln_space_fig_params):
        if np.all(all_fig_params[i] == params_of_interest) and (soln_space_fig_ids[i] == fig_id):
            time = soln.t
            # theta, theta0cc, theta0cm, theta0N = soln.y
            theta, Scc0, Scm0, SA0, SNcc0, SNcm0, SN0 = soln.y
            z_cc = z_cc_func(time, *z_cc_func_args)
            ax.plot(2*np.rad2deg(theta), z_cc/z_cm, lw=1, c='k', alpha=1.0, ls=':')
    
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.linspace(0,1, 11), minor=True)
    ax.set_ylabel('Rel. Contact Actin')
    ax.set_xlim(0, 220)
    ax.set_xticks([0,50,100,150,200], minor=True)
    ax.set_xlabel('Contact Angle (deg)')

    fig.savefig(Path.joinpath(man_plots_dir, r"%s.tif"%fig_id), dpi=dpi_LineArt, bbox_inches='tight')
    plt.close('all')



















## Solve for each set of parameters:
solns_stretching = list()
for linecol, linesty, lb, params in params_stretching:
    
    # unpack the parameters and assign some constants: 
    Gamma, strain_lim_actin, t_Apt5, k_EA, k_eta_A, strain_lim_nonactin, t_Npt5, k_EN, k_eta_N, E_r = params
    consts = (R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, k_EA, k_eta_A, k_EN, k_eta_N, k_etam_real, Gamma, E_r, z_cc_func, z_cc_func_args, z_cm_func, z_cm_func_args)
    
    t, t_end = t_init, t_end # Set the range over which to solve.
    syst = [theta_init, Scc0_init, Scm0_init, SA0_init, SNcc0_init, SNcm0_init, SN0_init] # set up the initial conditions
    
    # solve the system:
    soln, soln_details = angular_veloc_ve_model((t, t_end), syst, consts, solve_ivp_method, solve_ivp_max_step, solve_ivp_atol, solve_ivp_rtol, strain_lim_event_stol, calcium=True)

    solns_stretching.append((linecol, linesty, lb, soln))



solns_rolling = list()
for linecol, linesty, lb, params in params_rolling:
    
    # unpack the parameters and assign some constants: 
    Gamma, strain_lim_actin, t_Apt5, k_EA, k_eta_A, strain_lim_nonactin, t_Npt5, k_EN, k_eta_N, E_r = params
    consts = (R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, k_EA, k_eta_A, k_EN, k_eta_N, k_etam_real, Gamma, E_r, z_cc_func, z_cc_func_args, z_cm_func, z_cm_func_args)
    
    t, t_end = t_init, t_end # Set the range over which to solve.
    syst = [theta_init, Scc0_init, Scm0_init, SA0_init, SNcc0_init, SNcm0_init, SN0_init] # set up the initial conditions
    
    # solve the system:
    soln, soln_details = angular_veloc_ve_model((t, t_end), syst, consts, solve_ivp_method, solve_ivp_max_step, solve_ivp_atol, solve_ivp_rtol, strain_lim_event_stol, calcium=True)

    solns_rolling.append((linecol, linesty, lb, soln))



## Plot a timecourse of the results:
fig, ax = plt.subplots(figsize=(width_1col*0.7, width_1col*0.7)) # set up the figure. 
for linecol, linesty, lb, soln in solns_stretching:
    # unpack the results:
    time = soln.t
    theta, Scc0, Scm0, SA0, SNcc0, SNcm0, SN0 = soln.y
    z_cc = z_cc_func(time, *z_cc_func_args)
    z_cm = z_cm_func(time, *z_cm_func_args)

    # plot the results:
    # ax.plot(time/60, 2*np.rad2deg(theta), c=linecol, ls=linesty)
    # ax.set_xlim(-5/60, 1800/60)
    # ax.set_ylim(0, 200)
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
    
# save the resulting plot:
fig.savefig(Path.joinpath(man_plots_dir, "physiol_vals_timecourse_contact_surf.tif"), dpi=dpi_LineArt, bbox_inches='tight')
plt.close('fig')


fig, ax = plt.subplots(figsize=(width_1col*0.7, width_1col*0.7)) # set up the figure. 
for linecol, linesty, lb, soln in solns_rolling:
    # unpack the results:
    time = soln.t
    theta, Scc0, Scm0, SA0, SNcc0, SNcm0, SN0 = soln.y
    z_cc = z_cc_func(time, *z_cc_func_args)
    z_cm = z_cm_func(time, *z_cm_func_args)

    # plot the results:
    ax.plot(time/60, 2*np.rad2deg(theta), c=linecol, ls=linesty)
    ax.set_xlim(-5/60, 1800/60)
    ax.set_ylim(0, 200)
    
    axb = ax.twinx()
    axb.plot(time/60, z_cc/z_cm, c='grey')
    axb.tick_params(axis='y', colors='grey')
    axb.set_ylim(0,1)
    
    ax.set_zorder(axb.get_zorder()+1) # this puts ax in front of axb
    ax.patch.set_visible(False) # this sets the background of ax

# save the resulting plot:
fig.savefig(Path.joinpath(man_plots_dir, "physiol_vals_timecourse_whole_surf.tif"), dpi=dpi_LineArt, bbox_inches='tight')
plt.close('fig')



## Plot a the angles against the contact actin:

fig, ax = plt.subplots(figsize=(width_1col*0.7, width_1col*0.7)) # set up the figure. 
for linecol, linesty, lb, soln in solns_stretching[::-1]:
    # unpack the results:
    time = soln.t
    theta, Scc0, Scm0, SA0, SNcc0, SNcm0, SN0 = soln.y    
    z_cc = z_cc_func(time, *z_cc_func_args)

    # plot the results:
    ax.axhline(z_plat, c='grey', ls=':')
    ax.axvline(180, c='grey', ls=':')
    ax.plot(2*np.rad2deg(theta), z_cc, c=linecol, ls=linesty, lw=1, label=lb)
    # ax.set_xlim(0,200)
    # ax.set_ylim(0,2)
    
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.linspace(0,1, 11), minor=True)
    ax.set_ylabel('Rel. Contact Actin')
    ax.set_xlim(0, 220)
    ax.set_xticks([0,50,100,150,200], minor=True)
    ax.set_xlabel('Contact Angle (deg)')
# save the resulting plot:
fig.savefig(Path.joinpath(man_plots_dir, "physiol_vals_angle_vs_actin_contact_surf.tif"), dpi=dpi_LineArt, bbox_inches='tight')
plt.close('fig')


fig, ax = plt.subplots(figsize=(width_1col*0.7, width_1col*0.7)) # set up the figure. 
for linecol, linesty, lb, soln in solns_rolling:
    # unpack the results:
    time = soln.t
    theta, Scc0, Scm0, SA0, SNcc0, SNcm0, SN0 = soln.y    
    z_cc = z_cc_func(time, *z_cc_func_args)

    # plot the results:
    ax.plot(2*np.rad2deg(theta), z_cc, c=linecol, ls=linesty)
    ax.set_xlim(0,200)
    ax.set_ylim(0,2)
# save the resulting plot:
fig.savefig(Path.joinpath(man_plots_dir, "physiol_vals_angle_vs_actin_whole_surf.tif"), dpi=dpi_LineArt, bbox_inches='tight')
plt.close('fig')







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
print("With an spherical radius of %.2f, the strain at full \
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
fig.savefig(Path.joinpath(man_plots_dir, "dScc_vs_dScm_dtheta.tif"), dpi=dpi_LineArt, bbox_inches='tight')












## Load in the data from the LifeAct-GFP analysis:
la_contnd_df = pd.read_pickle(Path.joinpath(lifeact_analysis_dir, 'la_contnd_df.pkl'))











# fig, ax = plt.subplots(figsize=(width_1col*0.7, width_1col*0.7))
# fig_of_interest = 'fig3D'
# for params in params_w_fig_id[fig_of_interest][1:]:
#     for i, soln in enumerate(soln_space_fig_params):
#         if np.all(all_fig_params[i] == params) and (soln_space_fig_ids[i] == fig_of_interest):
#             time = soln.t
#             # theta, theta0cc, theta0cm, theta0N = soln.y
#             theta, Scc0, Scm0, SA0, SN0 = soln.y
#             z_cc = z_cc_func(time, *z_cc_func_args)
#             z_cm = z_cm_func(time, *z_cm_func_args)
#             ax.plot(2*np.rad2deg(theta), z_cc/z_cm, lw=1.5, alpha=1.0)
# for i, soln in enumerate(soln_space_fig_params):
#     if np.all(all_fig_params[i] == params_w_fig_id[fig_of_interest][0]) and (soln_space_fig_ids[i] == fig_of_interest):
#         time = soln.t
#         # theta, theta0cc, theta0cm, theta0N = soln.y
#         theta, Scc0, Scm0, SA0, SN0 = soln.y
#         z_cc = z_cc_func(time, *z_cc_func_args)
#         z_cm = z_cm_func(time, *z_cm_func_args)
#         ax.plot(2*np.rad2deg(theta), z_cc/z_cm, lw=1, c='k', alpha=1.0, ls=':')
# sns.scatterplot(x='Average Angle (deg)', y='lifeact_fluor_normed_mean', color='grey', 
#                 data=la_contnd_df[la_contnd_df['Experimental_Condition'] == 'ee1xMBS'], 
#                 s=5, linewidth=0, alpha=0.5, ax=ax, legend=False, zorder=10)
# ax.set_xlim(0,200)
# ax.axvline(180, c='grey', lw=1, zorder=0)
# ax.set_ylim(0,2)
# ax.set_xlabel('Contact Angle (deg)')
# ax.set_ylabel('Norm. Contact Fluor. (N.D.)')
# # plt.tight_layout()
# fig.savefig(Path.joinpath(man_plots_dir, r'angle_vs_fluor_lagfp_w_model_%s.tif'%fig_of_interest), dpi=dpi_LineArt, bbox_inches='tight')
# plt.close(fig)








fig, ax = plt.subplots(figsize=(width_1col*0.70, width_1col*0.70))
fig_of_interest = 'fig3D'
for params in params_w_fig_id[fig_of_interest][1:]:
    for i, soln in enumerate(soln_space_fig_params):
        if np.all(all_fig_params[i] == params) and (soln_space_fig_ids[i] == fig_of_interest):
            time = soln.t
            # theta, theta0cc, theta0cm, theta0N = soln.y
            theta, Scc0, Scm0, SA0, SNcc0, SNcm0, SN0 = soln.y
            z_cc = z_cc_func(time, *z_cc_func_args)
            z_cm = z_cm_func(time, *z_cm_func_args)
            ax.plot(time/60, 2*np.rad2deg(theta), lw=2, alpha=1.0)
for i, soln in enumerate(soln_space_fig_params):
    if np.all(all_fig_params[i] == params_w_fig_id[fig_of_interest][0]) and (soln_space_fig_ids[i] == fig_of_interest):
        time = soln.t
        # theta, theta0cc, theta0cm, theta0N = soln.y
        theta, Scc0, Scm0, SA0, SNcc0, SNcm0, SN0 = soln.y
        z_cc = z_cc_func(time, *z_cc_func_args)
        z_cm = z_cm_func(time, *z_cm_func_args)
        ax.plot(time/60, 2*np.rad2deg(theta), lw=1, c='k', alpha=1.0, ls=':')
ax.set_xlim(0,1800/60)
ax.set_ylim(0,200)
ax.set_xlabel('Time (minutes)')
ax.set_ylabel('Contact Angle (deg)')

axb = ax.twinx()
axb.plot(time/60, z_cc/z_cm, c='grey', lw=1, ls='--')
axb.set_ylim(0,1)
axb.set_ylabel('Norm. Contact Fluor. (N.D.)', rotation=270, va='bottom', ha='center', color='grey')
axb.tick_params(colors='grey')
# plt.tight_layout()
fig.savefig(Path.joinpath(out_dir, r'angle_vs_time_w_model_%s.tif'%fig_of_interest), dpi=dpi_LineArt, bbox_inches='tight')
plt.close(fig)







fig, ax = plt.subplots(figsize=(width_1col*0.7, width_1col*0.7)) # set up the figure. 

for linecol, linesty, lb, soln in solns_stretching[::-1]:
    # unpack the results:
    time = soln.t
    theta, Scc0, Scm0, SA0, SNcc0, SNcm0, SN0 = soln.y    
    z_cc = z_cc_func(time, *z_cc_func_args)

    # plot the results:
    ax.plot(2*np.rad2deg(theta), z_cc, c=linecol, ls=linesty, lw=1, label=lb)

sns.scatterplot(x='Average Angle (deg)', y='lifeact_fluor_normed_mean', color='grey', 
                data=la_contnd_df[la_contnd_df['Experimental_Condition'] == 'ee1xMBS'], 
                s=5, linewidth=0, alpha=0.5, ax=ax, legend=False, zorder=1)
ax.axhline(z_plat, c='grey', ls='-', zorder=0)
ax.axvline(180, c='grey', ls='-', zorder=0)
    
ax.set_ylim(0, 2)
ax.set_yticks(np.linspace(0,1, 11), minor=True)
ax.set_xlim(0, 200)
ax.set_xticks([0,50,100,150,200], minor=True)
ax.set_xlabel('Contact Angle (deg)')
ax.set_ylabel('Norm. Contact Fluor. (N.D.)')
# save the resulting plot:
# fig.legend()
fig.savefig(Path.joinpath(man_plots_dir, r'angle_vs_fluor_lagfp_w_model_%s.tif'%fig_of_interest), dpi=dpi_LineArt, bbox_inches='tight')
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
## The above is just a convenient stopping point when developing the script 
## since everything after this takes a somewhat substantial amount of time.



print('\n\nStarting simulation of low adhesion / uncoupled cytoskeletal networks.\n\n')

lacontnd_normed_mean = la_contnd_df[la_contnd_df['Experimental_Condition'] == 'ee1xDB']['lifeact_fluor_normed_mean'].mean()
lacontnd_normed_std = la_contnd_df[la_contnd_df['Experimental_Condition'] == 'ee1xDB']['lifeact_fluor_normed_mean'].std()
print('The relative contact actin fluoresence is:\n> mean=%f.2f, std=%.2f'%(lacontnd_normed_mean, lacontnd_normed_std))


## Random low actin levels: 
z_cc_func = z_rand
z_cc_func_args = (lacontnd_normed_mean, lacontnd_normed_std, data_acquisition_rate, acquisition_duration, z_init, 42)#42) # 42 = the random_state
# z_cm_func_rand = z_rand 
# z_cm_func_args_rand = (1.0, 0.1, data_acquisition_rate, acquisition_duration, z_init, 24)#24) # 24 = the random_state
z_cm_func_const = z_const
z_cm_func_args_const = (1,)

# solve_ivp_max_step_low = 1e0 # if we use np.inf then some events are skipped 
## since there are long periods of equilibrium. 


t_end_rand = 1800
# Solving the equilibrium case: 
params = [Gamma_null, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_null]
# params = [G2pct, strain_lim_high, turnover_rate_vfast, 0, 0, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_null]
Gamma, strain_lim_actin, t_Apt5, k_EA, k_eta_A, strain_lim_nonactin, t_Npt5, k_EN, k_eta_N, E_r = params

t, t_end_rand = t_init, t_end_rand # Set the range over which to solve. 
# syst = [theta_init, theta0cc_init, theta0cm_init, theta0N_init]
syst = [theta_init, Scc0_init, Scm0_init, SA0_init, SNcc0_init, SNcm0_init, SN0_init]
consts = (R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, k_EA, k_eta_A, k_EN, k_eta_N, k_etam_real, Gamma, E_r, z_cc_func, z_cc_func_args, z_cm_func, z_cm_func_args)

## Note that we have to set calcium==True here because calcium==False only 
## works for nonzero values of Gamma. 
print('Starting soln_eq...')
soln_eq, soln_details = angular_veloc_ve_model((t, t_end_rand), syst, consts, solve_ivp_method, solve_ivp_max_step, solve_ivp_atol, solve_ivp_rtol, strain_lim_event_stol, calcium=True)
print('Done with soln_eq.')


# Solving the viscoelastic case: 
# Fig. 7G - Xenopus-like solutions for overlapping onto lifeact data:
# fig7G_params = [[adh_energy_total_yesCalcium_M, strain_lim_intermfil, halflife_intermfil_M, k_E_multinetwork_M, k_eta_multinetwork, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], # Xenopus multinetwork medium
#                 [adh_energy_total_yesCalcium_L, strain_lim_intermfil, halflife_intermfil_M, k_E_multinetwork_M, k_eta_multinetwork, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], # Xenopus multinetwork low
#                 [adh_energy_total_yesCalcium_H, strain_lim_intermfil, halflife_intermfil_M, k_E_multinetwork_M, k_eta_multinetwork, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], # Xenopus multinetwork low
#                ] 
fig7G_params = [[Gamma_null, strain_lim_intermfil, halflife_intermfil_M, k_E_multinetwork_M, k_eta_multinetwork, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example], # Xenopus multinetwork medium
               ] 


### Want to get the equilibrium angle for adh_energy_total_noCalcium_M and use 
### that as the initial system:
Gamma, strain_lim_actin, t_Apt5, k_EA, k_eta_A, strain_lim_nonactin, t_Npt5, k_EN, k_eta_N, E_r = fig7G_params[0]
# Gamma = adh_energy_total_noCalcium_M
# Gamma_list = [adh_energy_total_noCalcium_M, adh_energy_total_noCalcium_L, adh_energy_total_noCalcium_H]
Gamma_list = [adh_energy_total_yesCalcium_M, adh_energy_total_yesCalcium_L, adh_energy_total_yesCalcium_H]
Gamma_list = [adh_energy_total_yesCalcium_M]

syst_list = list()
for Gamma in Gamma_list:
    t, t_end = t_init, t_end # Set the range over which to solve.
    syst = [theta_init, Scc0_init, Scm0_init, SA0_init, SNcc0_init, SNcm0_init, SN0_init]
    consts = (R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, k_EA, k_eta_A, k_EN, k_eta_N, k_etam_real, Gamma, E_r, z_cm_func_const, z_cm_func_args_const, z_cm_func_const, z_cm_func_args_const)
    
    soln = solve_ivp(angular_veloc_Gamma_inc, (t, t_end), syst, args=consts, method=solve_ivp_method, events=None, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
    
    # soln = solve_ivp(get_theta_dot_rolling_inc, (t_init, t_end), syst, args=consts, method=solve_ivp_method, events=None, max_step=solve_ivp_max_step, atol=solve_ivp_atol, rtol=solve_ivp_rtol, dense_output=True)
    theta_init_rand = soln.y[0,-1]
    Rc_init_rand = (4**(1/3) * R_s) / (4 - (1 - np.cos(theta_init_rand))**2 * (2 + np.cos(theta_init_rand)))**(1/3)
    Scc0_init_rand = np.pi * (Rc_init_rand * np.sin(theta_init_rand))**2
    Scm0_init_rand = 2 * np.pi * Rc_init_rand**2 * (1 + np.cos(theta_init_rand))
    SA0_init_rand = Scc0_init_rand + Scm0_init_rand
    SNcc0_init_rand = np.pi * (Rc_init_rand * np.sin(theta_init_rand))**2
    SNcm0_init_rand = 2 * np.pi * Rc_init_rand**2 * (1 + np.cos(theta_init_rand))
    SN0_init_rand = Scc0_init_rand + Scm0_init_rand

    syst_list.append([theta_init_rand, Scc0_init_rand, Scm0_init_rand, SA0_init_rand, SNcc0_init_rand, SNcm0_init_rand, SN0_init_rand])



fig7G_soln_list = list()
fig7G_soln_details_list = list()
# for i, params in enumerate(fig7G_params):
for i, syst in enumerate(syst_list):
    Gamma, strain_lim_actin, t_Apt5, k_EA, k_eta_A, strain_lim_nonactin, t_Npt5, k_EN, k_eta_N, E_r = fig7G_params[0]

    t, t_end_rand = t_init, t_end_rand # Set the range over which to solve. 
    # syst = [theta_init_rand, Scc0_init_rand, Scm0_init_rand, SA0_init_rand, SN0_init_rand]
    # syst = syst_list[i]
    # consts = (R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, k_EA, k_eta_A, k_EN, k_eta_N, k_etam_real, Gamma, E_r, z_cc_func, z_cc_func_args, z_cm_func_rand, z_cm_func_args_rand)
    consts = (R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, k_EA, k_eta_A, k_EN, k_eta_N, k_etam_real, Gamma, E_r, z_cc_func, z_cc_func_args, z_cm_func_const, z_cm_func_args_const)
    
    print('\n\nStarting soln...')
    print(consts)
    # soln, soln_details = angular_veloc_ve_model((t, t_end_rand), syst, consts, solve_ivp_method, solve_ivp_max_step, solve_ivp_atol, solve_ivp_rtol, strain_lim_event_stol, calcium=False)
    soln, soln_details = angular_veloc_ve_model((t, t_end_rand), syst, consts, solve_ivp_method, solve_ivp_max_step, solve_ivp_atol, solve_ivp_rtol, strain_lim_event_stol, calcium=True)
    print('Done with soln...')
    fig7G_soln_list.append(soln)
    fig7G_soln_details_list.append(soln_details)

# if Gamma != 0:
#     # soln_theta_Gamma = solve_ivp(angular_veloc_Gamma_inc, (t, t_end_rand*10), syst, args=consts, method=solve_ivp_method, events=angular_veloc_Gamma_inc_zero_event, atol=solve_ivp_atol, rtol=solve_ivp_rtol)
#     soln_theta_Gamma = solve_ivp(angular_veloc_Gamma_inc, (t, t_end_rand*10), syst, args=consts, method=solve_ivp_method, events=angular_veloc_Gamma_inc_zero_event, atol=solve_ivp_atol, rtol=solve_ivp_rtol)
#     if soln_theta_Gamma.message == 'A termination event occurred.':
#         theta_Gamma = soln_theta_Gamma.y[0,-1]


# params = [Gamma_null, strain_lim_intermfil, halflife_intermfil_M, k_E_multinetwork_M, k_eta_multinetwork, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example]
# # params = [adh_energy_total_yesCalcium_M, strain_lim_intermfil, halflife_intermfil_M, k_E_multinetwork_M, k_eta_multinetwork, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example]
# Gamma, strain_lim_actin, t_Apt5, k_EA, k_eta_A, strain_lim_nonactin, t_Npt5, k_EN, k_eta_N, E_r = params

# t_end_rand = 1800
# # t, t_end_rand = t_init, 500# Set the range over which to solve. 
# # syst = [theta_init, Scc0_init, Scm0_init, SA0_init, SNcc0_init, SNcm0_init, SN0_init]    syst = [theta_init_rand, Scc0_init_rand, Scm0_init_rand, SA0_init_rand, SN0_init_rand]
# syst = [theta_init_rand, Scc0_init_rand, Scm0_init_rand, SA0_init_rand, SN0_init_rand]
# consts = (R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, k_EA, k_eta_A, k_EN, k_eta_N, k_etam_real, Gamma, E_r, z_cc_func, z_cc_func_args, z_cm_func_const, z_cm_func_args_const)

# print('\n\nStarting soln...')
# # soln_const, soln_details_const = angular_veloc_ve_model((t, t_end_rand), syst, consts, solve_ivp_method, solve_ivp_max_step, solve_ivp_atol, solve_ivp_rtol, strain_lim_event_stol, calcium=False)
# soln_const, soln_details_const = angular_veloc_ve_model((t, t_end_rand), syst, consts, solve_ivp_method, solve_ivp_max_step, solve_ivp_atol, solve_ivp_rtol, strain_lim_event_stol, calcium=True)
# print('Done with soln...')


# soln_const_df = pd.DataFrame({'t':soln_const.t, 'theta':soln_const.y[0], 'color':soln_const.color})

# fig, ax = plt.subplots()
# [ax.scatter(x=data['t'], y=data['theta'], color=color, marker='.', s=0.7) for color, data, in soln_const_df.groupby('color')]
# axb = ax.twinx()
# axb.plot(soln_const_df['t'], z_cc_func(soln_const_df['t'], *z_cc_func_args) / z_cm_func_const(soln_const_df['t'], *z_cm_func_args_const), c='grey', ls='--', lw='1')
# axb.axhline(1, c='lightgrey', lw=1)
# [ax.axvline(e, c='k', lw=0.5, ls=':', zorder=0) for e in soln_const.t_events if e]
# fig.savefig(Path.joinpath(man_plots_dir, 'test.tif'), dpi=dpi_LineArt, bbox_inches='tight')


## Plot the results: 
# i = 0
time_eq, theta_eq = soln_eq.t, soln_eq.y[0,:]
time = fig7G_soln_list[0].t
# time_const = soln_const.t
# theta, Scc0, Scm0, SA0, SN0 = fig7G_soln_list[i].y
# theta_const, Scc0_const, Scm0_const, SA0_const, SN0_const = soln_const.y
z_cc = z_cc_func(time, *z_cc_func_args)
# z_cm = z_cm_func_rand(time, *z_cm_func_args_rand)
z_cm = z_cm_func_const(time, *z_cm_func_args_const)

# z_cc_const = z_cc_func(time_const, *z_cc_func_args) # there is no z_cc_func_args_const (since contact actin varies)
# z_cm_const = z_cm_func_const(time_const, *z_cm_func_args_const)



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
axb.plot(time/60, z_cc/z_cm, c='grey', lw=1)
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




# fig, ax = plt.subplots(figsize=(width_1col*0.7, width_1col*0.7))
# # ax.plot(time_eq/60, 2*np.rad2deg(theta_eq), lw=0.7, c='k', alpha=1.0, ls=':')
# # ax.axhline(np.mean(2*np.rad2deg(theta_eq)), lw=0.7, c='k', alpha=1.0, ls='--')
# # ax.axhline(2*np.rad2deg(theta_Gamma), lw=0.7, c='tab:orange', alpha=1.0, ls='--')
# # ax.plot(time, 2*np.rad2deg(theta), lw=0.7, c='r', alpha=1.0)
# # ax.plot(time/60, 2*np.rad2deg(theta))
# # for soln in fig7G_soln_list:
# #     ax.plot(soln.t/60, 2*np.rad2deg(soln.y[0]))
# ax.plot(time/60, 2*np.rad2deg(theta), c='tab:blue', lw=0.7)
# ax.plot(time_const/60, 2*np.rad2deg(theta_const), c='tab:red', lw=0.5, ls='-', zorder=0)
# [ax.axvline(event_t/60, lw=0.5, ls=':', c='k') for event_t in fig7G_soln_details_list[0][-1] if event_t.size > 0]

# axb = ax.twinx()
# axb.plot(time/60, z_cc/z_cm, c='grey', lw=0.7)
# # axb.plot(time_const/60, z_cc_const/z_cm_const, c='grey', lw=1)
# # axb.plot(time_const/60, z_cc_const, c='b', lw=1)
# # axb.plot(time_const/60, [z_cm_const]*len(z_cc_const), c='g', lw=1)
# axb.plot(time/60, z_cm, c='green', lw=0.7)
# axb.plot(time/60, z_cc, c='lightgreen', lw=0.7, ls='--')
# axb.axhline(1, c='lightgrey', lw=0.7, zorder=0)
# axb.tick_params(axis='y', colors='grey')
# axb.set_ylim(0,2)

# ax.set_zorder(axb.get_zorder()+1) # this puts ax in front of axb
# ax.patch.set_visible(False) # this sets the background of ax
# # ax.set_xlim(-5/60, 5)#1800/60)
# # ax.set_ylim(0, 80)#200)

# # plt.show()
# fig.savefig(Path.joinpath(man_plots_dir, 'low_adh_sim_over_time2.tif'), dpi=dpi_LineArt, bbox_inches='tight')
# plt.close('all')



# plt.plot(time_eq/60, theta_eq, lw=1, c='k', alpha=1.0, ls='-',zorder=10)
# plt.plot(time_eq/60, z_cont_rand(time_eq), c='g', lw=0.7)
# [plt.axvline(t_event/60, c='r', lw=0.5) for t_event in soln_eq.t_events]
# plt.axhline(1, c='r', lw=0.5)


# plt.scatter(2*np.rad2deg(theta), z_cont_rand(time), marker='.', c='r', alpha=0.5, s=2)
# plt.scatter(2*np.rad2deg(theta_eq), z_cont_rand(time_eq), marker='.', c='grey', alpha=0.5, s=2)


fig, ax = plt.subplots(figsize=(width_1col*0.7, width_1col*0.7))
# sns.scatterplot(x='Average Angle (deg)', y='lifeact_fluor_normed_mean', color='c', 
#                 data=la_contnd_df[la_contnd_df['Experimental_Condition'] == 'ee1xMBS'], 
#                 s=5, linewidth=0, alpha=0.5, ax=ax, legend=False, zorder=10)
sns.scatterplot(x='Average Angle (deg)', y='lifeact_fluor_normed_mean', color='grey', 
                data=la_contnd_df[la_contnd_df['Experimental_Condition'] == 'ee1xDB'], 
                s=5, linewidth=0, alpha=1, ax=ax, legend=False, zorder=0)
# sns.scatterplot(x='Average Angle (deg)', y='lifeact_fluor_normed_mean', hue='Source_File', 
#                 data=la_contnd_df[la_contnd_df['Experimental_Condition'] == 'ee1xDB'], 
#                 s=5, linewidth=0, alpha=0.5, ax=ax, legend=False, zorder=10)

for soln in fig7G_soln_list:
    ax.scatter(2*np.rad2deg(soln.y[0]), z_cc_func(soln.t, *z_cc_func_args)/z_cm_func(soln.t, *z_cm_func_args), marker='.', alpha=0.4, s=1)
    # ax.draw
# ax.scatter( 2*np.rad2deg(soln_const.y[0]), z_cc_func(soln_const.t, *z_cc_func_args)/z_cm_func_const(soln_const.t, *z_cm_func_args_const), c='r', marker='.', alpha=0.4, s=1)

ax.scatter( 2*np.rad2deg(soln_eq.y[0]), z_cc_func(soln_eq.t, *z_cc_func_args)/z_cm_func(soln_eq.t, *z_cm_func_args), c='k', marker='.', s=1)
ax.axvline(180, c='grey', lw=1, zorder=10)
ax.axvline(41.1, c='tab:orange', lw=1, ls='--', zorder=0) # the simulated cells from Fig 4E,F have a contact angle of about 41.1 degrees
ax.set_xlim(0,200)
ax.set_ylim(0,2)
ax.set_ylabel('')
ax.set_xlabel('')
plt.tight_layout()

fig.savefig(Path.joinpath(man_plots_dir, 'low_adh_sim.tif'), dpi=dpi_LineArt, bbox_inches='tight')
plt.close('all')



# When you are done, let me know:
print('Done.')





# z_cc_func = z_decay
# z_cc_func_args = (z_init, t_half, z_plat)
# z_cm_func = z_const
# z_cm_func_args = (1,) 

# params = adh_energy_total_yesCalcium_M, strain_lim_actin, halflife_actin, k_E_actin_H, k_eta_actin, strain_lim_high, turnover_rate_vfast, 0, 0, E_B_example # actin
# Gamma, strain_lim_actin, t_Apt5, k_EA, k_eta_A, strain_lim_nonactin, t_Npt5, k_EN, k_eta_N, E_r = params


# t, t_end_rand = t_init, t_end_rand # Set the range over which to solve. 
# # syst = [theta_init, theta0cc_init, theta0cm_init, theta0N_init]
# syst = [theta_init, Scc0_init, Scm0_init, SA0_init, SNcc0_init, SNcm0_init, SN0_init]
# consts = (R_s, strain_lim_actin, t_Apt5, strain_lim_nonactin, t_Npt5, sigma_a, k_EA, k_eta_A, k_EN, k_eta_N, k_etam_real, Gamma, E_r, z_cc_func, z_cc_func_args, z_cm_func, z_cm_func_args)

# ## Note that we have to set calcium==True here because calcium==False only 
# ## works for nonzero values of Gamma. 
# soln, soln_details = angular_veloc_ve_model((t, t_end_rand), syst, consts, solve_ivp_method, solve_ivp_max_step, solve_ivp_atol, solve_ivp_rtol, strain_lim_event_stol, calcium=True)



# plt.plot(soln.t, 2*np.rad2deg(soln.y[0]))



