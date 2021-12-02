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

def port_arrow(sim, port_direction, arrow_size=0.1):
    """
    Given a port_direction, returns the dx and dy of the arrow
    Uses the simulation domain to scale the arrows
    Arrow size in % of cell size
    """
    # Get cell size
    sc = sim.cell_size
    scx = sc.x
    scy = sc.y
    # Parse direction
    if type(port_direction) == int:
        if port_direction == 0: # mp.X
            port_direction = mp.Vector3(1,0,0)
        elif port_direction == 1: # mp.Y
            port_direction = mp.Vector3(0,1,0)
    theta = np.arctan(port_direction.y/port_direction.x)
    dx = arrow_size*scx*np.cos(theta)
    dy = arrow_size*scy*np.sin(theta)
    return dx, dy

def plotStructure(sim, geometry, sources, sources_directions, waveguide_monitor_port, waveguide_port_direction, fiber_monitor_port, fiber_port_direction, wl=1/1.55, cmap=plt.get_cmap('tab10'), z_cut=0):
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
    for source, source_direction in zip(sources, sources_directions):
        xi = source.center - source.size/2
        xf = source.center + source.size/2
        plt.plot([xi.x, xf.x], [xi.y, xf.y], linewidth=2, color='k')
        dx, dy = port_arrow(sim, source_direction)
        plt.arrow(source.center.x, source.center.y, dx, dy, width=0.1, color='k')
    for monitor, monitor_direction in [(waveguide_monitor_port, waveguide_port_direction), (fiber_monitor_port, fiber_port_direction)]:
        xi = monitor.center - monitor.size/2
        xf = monitor.center + monitor.size/2
        plt.plot([xi.x, xf.x], [xi.y, xf.y], linewidth=2, color='gray')
        dx, dy = port_arrow(sim, monitor_direction)
        plt.arrow(monitor.center.x, monitor.center.y, dx, dy, width=0.1, color='gray')

    plt.xlabel(r'x ($\mu$m)')
    plt.ylabel(r'y ($\mu$m)')

    plt.show()