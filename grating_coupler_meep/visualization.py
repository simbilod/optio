"""
Visualization tools
"""

import sys
from functools import partial
from typing import Optional, Tuple

import hashlib
import time
import pathlib
import omegaconf
import pandas as pd

import meep as mp
import numpy as np
import fire

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl

def plotStructure(sim, geometry, sources, waveguide_monitor_port, fiber_monitor_port, wl=1/1.55, cmap=plt.get_cmap('tab10'), z_cut=0):
    """
    Plots the x-y index distribution of a MEEP simulation object with a custom colormap along z=z_cut cut

    sim (meep.simulation): MEEP simulation instance
    geometry ([meep.geometry]): list of MEEP geometric items used to generate the simulation (geometry argument in meep.sim)
    wl (float): wavelength at which to inspect the index
    ns ([float]): list of refractive indices
    colors ([(R,G,B,alpha)]): list of colors in RGBA format (default: )
    z (float): z value at which to plot the x-y plane (default: 0)
    """

    # Initialize simulation
    sim.init_sim()

    # Extract list of indices used to generate the structure
    epsilons = []
    for item in geometry:
        epsilons.append(item.material.epsilon(1/wl))
    epsilons = np.array(epsilons)[:,0,0]
    epsilons_unique = np.unique(epsilons)
    epsilons_unique_sorted = np.sort(epsilons_unique)
    ns_unique_sorted = np.sqrt(epsilons_unique_sorted)

    # Extract simulation region
    eps_array=sim.get_epsilon()
    (x,y,z,w)=sim.get_array_metadata()

    # Prepare custom colormap and intervals
    # See https://matplotlib.org/stable/tutorials/colors/colorbar_only.html

    # From index list to intervals
    ns_intervals = 0.5*(ns_unique_sorted[1:] + ns_unique_sorted[:-1])

    fig, ax = plt.subplots(figsize=(6, 1))
    fig.subplots_adjust(bottom=0.5)

    bounds = []
    bounds.append(np.min(ns_unique_sorted))
    for interval in ns_intervals:
        bounds.append(interval)
    bounds.append(np.max(ns_unique_sorted))
    norm = mpl.colors.BoundaryNorm(bounds, len(ns_unique_sorted))

    # # Manually plot simulation regiond
    fig, ax = plt.subplots(figsize=[10,8])
    im = plt.pcolormesh(x,y,np.transpose(np.sqrt(eps_array)),cmap=mpl.cm.ScalarMappable(cmap=cmap, norm=norm).get_cmap(), norm=norm,shading='gouraud')
    ax.set_aspect('auto')
    fig.colorbar(im, ax=ax)

    # Add monitors
    for source in sources:
        xi = source.center - source.size/2
        xf = source.center + source.size/2
        plt.plot([xi.x, xf.x], [xi.y, xf.y], linewidth=2, color='k')
        #plt.arrow([xi.x, xf.x], [xi.y, xf.y], linewidth=2, color='k')

    for monitor in [waveguide_monitor_port, fiber_monitor_port]:
        xi = monitor.center - monitor.size/2
        xf = monitor.center + monitor.size/2
        plt.plot([xi.x, xf.x], [xi.y, xf.y], linewidth=2, color='gray')

    plt.xlabel(r'x ($\mu$m)')
    plt.ylabel(r'y ($\mu$m)')

    plt.show()