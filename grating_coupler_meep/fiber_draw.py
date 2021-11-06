"""SMF specs from photonics.byu.edu/FiberOpticConnectors.parts/images/smf28.pdf

MFD:

- 10.4 for Cband
- 9.2 for Oband

"""

from functools import partial
from typing import Optional, Tuple

import meep as mp
import numpy as np

nm = 1e-3
nSi = 3.47
nSiO2 = 1.45

Floats = Tuple[float, ...]


def draw_grating_coupler_fiber(
    period: float = 0.68,
    fill_factor: float = 0.5,
    fiber_angle_deg: float = 15.0,
    fiber_xposition: float = 1.0,
    fiber_core_diameter: float = 10.4,
    fiber_numerical_aperture: float = 0.14,
    fiber_nclad: float = nSiO2,
    resolution: int = 64,  # pixels/um
    ncore: float = nSi,
    nclad: float = nSiO2,
    nsubstrate: float = nSi,
    n_periods: int = 30,
    box_thickness: float = 3.0,
    clad_thickness: float = 2.0,
    core_thickness: float = 220 * nm,
    wavelength: float = 1.55,
    etch_depth: float = 70 * nm,
    widths: Optional[Floats] = None,
    gaps: Optional[Floats] = None,
    wavelength_min: float = 1.5,
    wavelength_max: float = 1.6,
    wavelength_points: int = 50,
):
    """Returns simulation
    Draw grating coupler with fiber.
    na**2 = ncore**2 - nclad**2
    ncore = sqrt(na**2 + ncore**2)

    Args:
        period: grating coupler period
        fill_factor:
        fiber_angle_deg: angle fiber in degrees

        widths: overrides n_periods period and fill_factor
        gaps: overrides n_periods period and fill_factor

    """
    wavelengths = np.linspace(wavelength_min, wavelength_max, wavelength_points)
    freqs = 1 / wavelengths
    widths = widths or n_periods * [period * fill_factor]
    gaps = gaps or n_periods * [period * (1 - fill_factor)]

    substrate_thickness = 1.0
    hair = 4
    core_material = mp.Medium(index=ncore)
    clad_material = mp.Medium(index=nclad)

    dtaper = 12
    dbuffer = 0.5
    dpml = 1

    fiber_clad = 120
    fiber_angle = np.radians(fiber_angle_deg)
    hfiber_geom = 100  # Some large number to make fiber extend into PML

    fiber_ncore = (fiber_numerical_aperture ** 2 + fiber_nclad ** 2) ** 0.5
    fiber_clad_material = mp.Medium(index=fiber_nclad)
    fiber_core_material = mp.Medium(index=fiber_ncore)

    # MEEP's computational cell is always centered at (0,0), but code has beginning of grating at (0,0)
    sxy = 2 * dpml + dtaper + period * n_periods + 2 * dbuffer  # sx here
    sz = (
        2 * dbuffer
        + box_thickness
        + core_thickness
        + hair
        + substrate_thickness
        + 2 * dpml
    )  # sy here
    # comp_origin_x = dpml + dbuffer + dtaper
    comp_origin_x = 0

    # meep_origin_x = sxy/2
    # x_offset = meep_origin_x - comp_origin_x
    # x_offset = 0
    # comp_origin_y = dpml + substrate_thickness + box_thickness + core_thickness/2
    # comp_origin_y = 0
    # meep_origin_y = sz/2
    # y_offset = meep_origin_y - comp_origin_y
    y_offset = 0

    # x_offset_vector = mp.Vector3(x_offset,0)
    # offset_vector = mp.Vector3(x_offset, y_offset)
    offset_vector = mp.Vector3(0, 0, 0)

    Si = mp.Medium(index=nsubstrate)

    # We will do x-z plane simulation
    cell_size = mp.Vector3(sxy, sz)

    geometry = []

    # Fiber (defined first to be overridden)

    # Core
    geometry.append(
        mp.Block(
            material=fiber_clad_material,
            center=mp.Vector3(x=fiber_xposition) - offset_vector,
            size=mp.Vector3(fiber_clad, hfiber_geom),
            e1=mp.Vector3(x=1).rotate(mp.Vector3(z=1), -1 * fiber_angle),
            e2=mp.Vector3(y=1).rotate(mp.Vector3(z=1), -1 * fiber_angle),
        )
    )
    geometry.append(
        mp.Block(
            material=fiber_core_material,
            center=mp.Vector3(x=fiber_xposition) - offset_vector,
            size=mp.Vector3(fiber_core_diameter, hfiber_geom),
            e1=mp.Vector3(x=1).rotate(mp.Vector3(z=1), -1 * fiber_angle),
            e2=mp.Vector3(y=1).rotate(mp.Vector3(z=1), -1 * fiber_angle),
        )
    )

    # clad
    geometry.append(
        mp.Block(
            material=clad_material,
            center=mp.Vector3(0, clad_thickness / 2) - offset_vector,
            size=mp.Vector3(mp.inf, clad_thickness),
        )
    )

    # waveguide
    geometry.append(
        mp.Block(
            material=core_material,
            center=mp.Vector3(0, core_thickness / 2) - offset_vector,
            size=mp.Vector3(mp.inf, core_thickness),
        )
    )

    # grating etch
    x = 0
    for width, gap in zip(widths, gaps):
        geometry.append(
            mp.Block(
                material=clad_material,
                center=mp.Vector3(x + gap / 2, core_thickness - etch_depth / 2)
                - offset_vector,
                size=mp.Vector3(gap, etch_depth),
            )
        )
        x += width + gap

    geometry.append(
        mp.Block(
            material=clad_material,
            center=mp.Vector3(sxy - comp_origin_x - 0.5 * (dpml + dbuffer), 0)
            - offset_vector,
            size=mp.Vector3(dpml + dbuffer, core_thickness),
        )
    )

    # BOX
    geometry.append(
        mp.Block(
            material=clad_material,
            center=mp.Vector3(0, -0.5 * box_thickness) - offset_vector,
            size=mp.Vector3(mp.inf, box_thickness),
        )
    )

    # Substrate
    geometry.append(
        mp.Block(
            material=Si,
            center=mp.Vector3(
                0,
                -0.5 * (core_thickness + substrate_thickness + dpml + dbuffer)
                - box_thickness,
            )
            - offset_vector,
            size=mp.Vector3(mp.inf, substrate_thickness + dpml + dbuffer),
        )
    )

    # PMLs
    boundary_layers = [mp.PML(dpml)]

    # mode frequency
    fcen = 1 / wavelength

    waveguide_port_center = mp.Vector3(-1 * dtaper, 0) - offset_vector
    waveguide_port_size = mp.Vector3(0, 2 * clad_thickness - 0.2)
    fiber_port_center = (
        mp.Vector3(
            (0.5 * sz - dpml + y_offset - 1) * np.sin(fiber_angle) + fiber_xposition,
            0.5 * sz - dpml + y_offset - 1,
        )
        - offset_vector
    )
    fiber_port_size = mp.Vector3(sxy * 3 / 5 - 2 * dpml - 2, 0)

    # Waveguide source
    sources = [
        mp.EigenModeSource(
            src=mp.GaussianSource(fcen, fwidth=0.1 * fcen),
            size=waveguide_port_size,
            center=waveguide_port_center,
            eig_band=1,
            direction=mp.X,
            eig_match_freq=True,
            eig_parity=mp.ODD_Z,
        )
    ]

    # symmetries = [mp.Mirror(mp.Y,-1)]
    symmetries = []

    sim = mp.Simulation(
        resolution=resolution,
        cell_size=cell_size,
        boundary_layers=boundary_layers,
        geometry=geometry,
        # geometry_center=mp.Vector3(x_offset, y_offset),
        sources=sources,
        dimensions=2,
        symmetries=symmetries,
        eps_averaging=True,
    )

    # Ports
    waveguide_monitor_port = mp.ModeRegion(
        center=waveguide_port_center + mp.Vector3(x=0.2), size=waveguide_port_size
    )
    waveguide_monitor = sim.add_mode_monitor(
        freqs, 0, 1, waveguide_monitor_port, yee_grid=True
    )
    fiber_monitor_port = mp.ModeRegion(
        center=fiber_port_center - mp.Vector3(y=0.2),
        size=fiber_port_size,
        direction=mp.NO_DIRECTION,
    )
    fiber_monitor = sim.add_mode_monitor(freqs, 0, 1, fiber_monitor_port)
    return sim, fiber_monitor, waveguide_monitor


# remove silicon to clearly see the fiber (for debugging)
draw_grating_coupler_fiber_no_silicon = partial(
    draw_grating_coupler_fiber, ncore=nSiO2, nsubstrate=nSiO2
)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    sim, fiber_monitor, waveguide_monitor = draw_grating_coupler_fiber()
    # sim, fiber_monitor, waveguide_monitor = draw_grating_coupler_fiber_no_silicon()

    sim.plot2D()
    plt.show()
