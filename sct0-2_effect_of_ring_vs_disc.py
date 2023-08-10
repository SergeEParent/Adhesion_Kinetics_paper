# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 14:28:38 2022

@author: Serge
"""

from matplotlib import pyplot as plt
import numpy as np

def logistic(t, low_plat, high_plat, k, t_0):
    out = low_plat + (high_plat - low_plat)/(1 + np.exp(-k*(t-t_0)))
    return out

def ring(r):
    out = 2 * np.pi * r
    return out

def disc(r):
    out = np.pi * r**2
    return out


t = np.linspace(0,120, 1201)


diam = logistic(t, 0.1,1,1,60)

ring_out = ring(diam/2)

disc_out = disc(diam/2)


fig, ax = plt.subplots()
## Normalize the plotted outputs by dividing by the max value:
ax.plot(t, diam/diam.max(), c='k')
ax.plot(t, ring_out/ring_out.max(), c='tab:orange', ls='--')
ax.plot(t, disc_out/disc_out.max(), c='tab:green', ls=':')

plt.show()