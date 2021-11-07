"""SMF specs from photonics.byu.edu/FiberOpticConnectors.parts/images/smf28.pdf

MFD:

- 10.4 for Cband
- 9.2 for Oband

"""

from functools import partial
from typing import Optional, Tuple

import hashlib
import time
import pathlib
import omegaconf
import pandas as pd

import meep as mp
import numpy as np

nm = 1e-3
nSi = 3.47
nSiO2 = 1.45

Floats = Tuple[float, ...]


def grating_coupler_fiber(
    period: float = 0.68,
    fill_factor: float = 0.5,
    widths: Optional[Floats] = None,
    gaps: Optional[Floats] = None,
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
    etch_depth: float = 70 * nm,
    wavelength_min: float = 1.5,
    wavelength_max: float = 1.6,
    wavelength_points: int = 50,
    run: bool = True,
    overwrite: bool = False,
):
    """Returns simulation results from grating coupler with fiber.
    na**2 = ncore**2 - nclad**2
    ncore = sqrt(na**2 + ncore**2)

    Args:
        period: grating coupler period
        fill_factor:
        widths: overrides n_periods period and fill_factor
        gaps: overrides n_periods period and fill_factor
        fiber_angle_deg: angle fiber in degrees

    """
    wavelengths = np.linspace(wavelength_min, wavelength_max, wavelength_points)
    wavelength = np.mean(wavelengths)
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
    sxy = 2 * dpml + dtaper + period * n_periods + 2 * dbuffer
    sz = (
        2 * dbuffer
        + box_thickness
        + core_thickness
        + hair
        + substrate_thickness
        + 2 * dpml
    )
    comp_origin_x = 0
    y_offset = 0
    offset_vector = mp.Vector3(0, 0, 0)

    # We will do x-z plane simulation
    cell_size = mp.Vector3(sxy, sz)

    geometry = []
    # Fiber (defined first to be overridden)
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
            material=mp.Medium(index=nsubstrate),
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

    waveguide_port_center = mp.Vector3(-dtaper, 0) - offset_vector
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
        freqs, waveguide_monitor_port, yee_grid=True
    )
    fiber_monitor_port = mp.ModeRegion(
        center=fiber_port_center - mp.Vector3(y=0.2),
        size=fiber_port_size,
        direction=mp.NO_DIRECTION,
    )
    fiber_monitor = sim.add_mode_monitor(freqs, fiber_monitor_port)

    if not run:
        sim.plot2D()
        return pd.DataFrame()

    settings_list = widths + gaps
    settings_string_list = [str(i) for i in settings_list]
    settings_string = "_".join(settings_string_list)
    settings_hash = hashlib.md5(settings_string.encode()).hexdigest()[:8]

    filename = f"fiber_{settings_hash}.yml"
    filepath = pathlib.Path("data") / filename
    filepath_csv = filepath.with_suffix(".csv")

    if filepath_csv.exists() and not overwrite:
        return pd.read_csv(filepath_csv)

    start = time.time()

    # Run simulation
    # sim.run(until=400)
    field_monitor_point = (-dtaper, 0, 0)
    sim.run(
        until_after_sources=mp.stop_when_fields_decayed(
            dt=50, c=mp.Ez, pt=field_monitor_point, decay_by=1e-9
        )
    )

    # Extract mode information
    transmission_waveguide = sim.get_eigenmode_coefficients(
        waveguide_monitor, [1], eig_parity=mp.ODD_Z, direction=mp.X
    )
    kpoint = mp.Vector3(y=-1).rotate(mp.Vector3(z=1), -1 * fiber_angle_deg)
    reflection_fiber = sim.get_eigenmode_coefficients(
        fiber_monitor,
        [1],
        direction=mp.NO_DIRECTION,
        eig_parity=mp.ODD_Z,
        kpoint_func=lambda f, n: kpoint,
    )
    end = time.time()

    settings = dict(period=period, fill_factor=fill_factor)

    simulation = dict(
        settings=settings,
        compute_time_seconds=end - start,
    )
    filepath.write_text(omegaconf.OmegaConf.to_yaml(simulation))

    results = dict(t=transmission_waveguide, r=reflection_fiber)
    df = pd.DataFrame(results, index=wavelengths)
    df.to_csv(filepath_csv, index=False)
    return df


# remove silicon to clearly see the fiber (for debugging)
grating_coupler_fiber_no_silicon = partial(
    grating_coupler_fiber, ncore=nSiO2, nsubstrate=nSiO2
)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    df = grating_coupler_fiber(run=True)
    # df = grating_coupler_fiber_no_silicon()

    print(df)
