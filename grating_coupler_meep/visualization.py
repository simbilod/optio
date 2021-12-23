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
import h5py
from os import listdir
from os.path import isfile, join

import meep as mp
import numpy as np
import fire
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from matplotlib import animation


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
        if port_direction == 0:  # mp.X
            port_direction = mp.Vector3(1, 0, 0)
        elif port_direction == 1:  # mp.Y
            port_direction = mp.Vector3(0, 1, 0)
    theta = np.arctan(port_direction.y / port_direction.x)
    dx = arrow_size * scx * np.cos(theta)
    dy = arrow_size * scy * np.sin(theta)
    return dx, dy


def plotStructure_fromSimulation(
    sim,
    geometry,
    waveguide_monitor_port,
    waveguide_port_direction,
    fiber_monitor_port,
    fiber_port_direction,
    wl=1 / 1.55,
    cmap=plt.get_cmap("tab10"),
    draw_directions=True,
    colorbar=True,
):
    """
    Plots the x-y index distribution of a MEEP simulation object with a custom colormap along z=z_cut cut
    Uses the simulation index data -- see what is used for simulation

    sim (meep.simulation): MEEP simulation instance; used to get the epsilon distribution
    geometry ([meep.geometry]): list of MEEP geometric items used to generate the simulation (geometry argument in meep.sim); used to get the indices level for plotting
    wl (float): wavelength at which to inspect the index
    waveguide_monitor_port (meep.ModeRegion): mode monitor in the waveguide
    waveguide_port_direction (meep.X or meep.Y or meep.Vector3): k-vector of the waveguide monitor
    fiber_monitor_port (meep.ModeRegion): mode monitor in the fiber
    fiber_port_direction (meep.X or meep.Y or meep.Vector3): k-vector of the fiber monitor
    cmap ([float]): list of colors for plotting
    """

    # Inspect simulation
    sources = sim.sources

    # Extract list of indices used to generate the structure
    epsilons = []
    for item in geometry:
        epsilons.append(item.material.epsilon(1 / wl))
    epsilons = np.array(epsilons)[:, 0, 0]
    epsilons_unique = np.unique(epsilons)
    epsilons_unique_sorted = np.sort(epsilons_unique)
    ns_unique_sorted = np.sqrt(epsilons_unique_sorted)

    # Extract simulation region
    eps_array = sim.get_epsilon()
    (x, y, z, w) = sim.get_array_metadata()

    # Prepare custom colormap and intervals
    # See https://matplotlib.org/stable/tutorials/colors/colorbar_only.html

    # From index list to intervals
    ns_intervals = 0.5 * (ns_unique_sorted[1:] + ns_unique_sorted[:-1])

    # fig, ax = plt.subplots(figsize=(6, 1))
    # fig.subplots_adjust(bottom=0.5)

    bounds = []
    bounds.append(np.min(ns_unique_sorted))
    for interval in ns_intervals:
        bounds.append(interval)
    bounds.append(np.max(ns_unique_sorted))
    norm = mpl.colors.BoundaryNorm(bounds, len(ns_unique_sorted))

    # # Manually plot simulation region
    fig, ax = plt.subplots(figsize=[10, 8])
    im = plt.pcolormesh(
        x,
        y,
        np.transpose(np.sqrt(eps_array)),
        cmap=mpl.cm.ScalarMappable(cmap=cmap, norm=norm).get_cmap(),
        norm=norm,
        shading="gouraud",
    )
    ax.set_aspect("auto")
    if colorbar:
        fig.colorbar(im, ax=ax)

    # Add monitors
    for source in sources:
        source_direction = source.direction
        xi = source.center - source.size / 2
        xf = source.center + source.size / 2
        plt.plot([xi.x, xf.x], [xi.y, xf.y], linewidth=2, color="k")
        dx, dy = port_arrow(sim, source_direction)
        plt.arrow(source.center.x, source.center.y, dx, dy, width=0.1, color="k")
    for monitor, monitor_direction in [
        (waveguide_monitor_port, waveguide_port_direction),
        (fiber_monitor_port, fiber_port_direction),
    ]:
        xi = monitor.center - monitor.size / 2
        xf = monitor.center + monitor.size / 2
        plt.plot([xi.x, xf.x], [xi.y, xf.y], linewidth=2, color="gray")
        dx, dy = port_arrow(sim, monitor_direction)
        plt.arrow(monitor.center.x, monitor.center.y, dx, dy, width=0.1, color="gray")

    # Draw PMLs
    dpml = sim.boundary_layers[0].thickness
    plt.fill_between(
        [np.min(x), np.max(x)],
        np.min(y),
        np.min(y) + dpml,
        facecolor="none",
        hatch="X",
        edgecolor="k",
        linewidth=0.0,
    )
    plt.fill_between(
        [np.min(x), np.max(x)],
        np.max(y) - dpml,
        np.max(y),
        facecolor="none",
        hatch="X",
        edgecolor="k",
        linewidth=0.0,
    )
    plt.fill_between(
        [np.min(x), np.min(x) + dpml],
        np.min(y),
        np.max(y),
        facecolor="none",
        hatch="X",
        edgecolor="k",
        linewidth=0.0,
    )
    plt.fill_between(
        [np.max(x) - dpml, np.max(x)],
        np.min(y),
        np.max(y),
        facecolor="none",
        hatch="X",
        edgecolor="k",
        linewidth=0.0,
    )

    plt.xlabel(r"x ($\mu$m)")
    plt.ylabel(r"y ($\mu$m)")

    # plt.show()
    return plt
    

def animateFields(
    sim,
    geometry,
    waveguide_monitor_port,
    waveguide_port_direction,
    fiber_monitor_port,
    fiber_port_direction,
    wl=1 / 1.55,
    cmap=plt.get_cmap("tab10"),
    h5_file="fiber-out/fiber-ez.h5",
):
    """
    Generates a custom MP4 of the simulation, using custom plotStructure and a directory containing the field time slices

    sim (meep.simulation): MEEP simulation instance
    geometry ([meep.geometry]): list of MEEP geometric items used to generate the simulation (geometry argument in meep.sim)
    wl (float): wavelength at which to inspect the index
    waveguide_monitor_port (meep.ModeRegion): mode monitor in the waveguide
    waveguide_port_direction (meep.X or meep.Y or meep.Vector3): k-vector of the waveguide monitor
    fiber_monitor_port (meep.ModeRegion): mode monitor in the fiber
    fiber_port_direction (meep.X or meep.Y or meep.Vector3): k-vector of the fiber monitor
    cmap ([float]): colormap for plotting
    z (float): z value at which to plot the x-y plane (default: 0)
    """
    # Data for animation
    onlyfiles = [f for f in listdir("fiber-out/") if isfile(join("fiber-out/", f))]
    onlyfiles = sorted([w[:-3] for w in onlyfiles])

    # Structure
    (x, y, z, w) = sim.get_array_metadata()

    # # First set up the figure, the axis, and the plot element we want to animate
    # fig = plotStructure_fromSimulation(
    #     sim,
    #     geometry,
    #     waveguide_monitor_port,
    #     waveguide_port_direction,
    #     fiber_monitor_port,
    #     fiber_port_direction,
    #     wl=1 / 1.55,
    #     cmap=plt.get_cmap("tab10"),
    #     draw_directions=True,
    #     colorbar=True,
    # )

    # fig = plotStructure_fromSimulation(
    # sim,
    # geometry,
    # waveguide_monitor_port,
    # waveguide_port_direction,
    # fiber_monitor_port,
    # fiber_port_direction,
    # wl=1 / 1.55,
    # cmap=plt.get_cmap("tab10"),
    # draw_directions=True,
    # colorbar=True,
    # )
    fig = plt.figure()

    a = np.random.random((5, 5))
    im = plt.imshow(a, interpolation="none")

    # initialization function: plot the background of each frame
    def init():
        # im.set_data(np.random.random((5,5)))
        im = plotStructure(
            sim,
            geometry,
            waveguide_monitor_port,
            waveguide_port_direction,
            fiber_monitor_port,
            fiber_port_direction,
            wl=1 / 1.55,
            cmap=plt.get_cmap("tab10"),
            draw_directions=False,
        )
        return [im]

    # animation function.  This is called sequentially
    frames = len(onlyfiles)
    def animate(i):
        print(onlyfiles[i])
        # Load data
        with h5py.File("fiber-out/" + onlyfiles[i] + ".h5", "r") as f:
            # List all groups
            a_group_key = list(f.keys())[0]
            # Get the data
            data = list(f["ez"])
        # Return data
        im = plt.pcolormesh(
            x,
            y,
            np.transpose(data),
        )
        return [im]

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(
        fig, animate, init_func=None, frames=int(frames/10), interval=20, blit=True
    )

    anim.save("basic_animation.mp4", fps=30, extra_args=["-vcodec", "libx264"])

    # plt.show()


if __name__ == "__main__":
    import matplotlib.pyplot as plt