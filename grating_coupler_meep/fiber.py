"""SMF specs from photonics.byu.edu/FiberOpticConnectors.parts/images/smf28.pdf

MFD:

- 10.4 for Cband
- 9.2 for Oband

TODO:

- verify with lumerical sims
- get Sparameters
- enable mpi run from python

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

# from grating_coupler_meep.visualization import plotStructure_fromSimulation

# sys.path.append("../../../meep_dev/meep/python/")
# from visualization import plot2D

nm = 1e-3
nSi = 3.48
nSiO2 = 1.44

Floats = Tuple[float, ...]


def dict_to_name(**kwargs) -> str:
    """Returns name from a dict."""
    kv = []

    for key in sorted(kwargs):
        if isinstance(key, str):
            value = kwargs[key]
            if value is not None:
                kv += [f"{key}{to_string(value)}"]
    return "_".join(kv)


def to_string(value):
    if isinstance(value, list):
        settings_string_list = [to_string(i) for i in value]
        return "_".join(settings_string_list)
    if isinstance(value, dict):
        return dict_to_name(**value)
    else:
        return str(value)


def fiber_ncore(fiber_numerical_aperture, fiber_nclad):
    return (fiber_numerical_aperture ** 2 + fiber_nclad ** 2) ** 0.5


def fiber(
    period: float = 0.66,
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
    box_thickness: float = 2.0,
    clad_thickness: float = 2.0,
    core_thickness: float = 220 * nm,
    etch_depth: float = 70 * nm,
    wavelength_min: float = 1.4,
    wavelength_max: float = 1.6,
    wavelength_points: int = 50,
    run: bool = True,
    animate: bool = False,
    overwrite: bool = False,
    dirpath: Optional[str] = None,
    decay_by: float = 1e-3,
    dtaper: float = 1,
    ncores: int = 1,
) -> pd.DataFrame:
    """Returns simulation results from grating coupler with fiber.
    na**2 = ncore**2 - nclad**2
    ncore = sqrt(na**2 + ncore**2)

    Args:
        period: grating coupler period
        fill_factor:
        widths: overrides n_periods period and fill_factor
        gaps: overrides n_periods period and fill_factor
        fiber_angle_deg: angle fiber in degrees
        decay_by: 1e-9

    """
    wavelengths = np.linspace(wavelength_min, wavelength_max, wavelength_points)
    wavelength = np.mean(wavelengths)
    freqs = 1 / wavelengths
    widths = widths or n_periods * [period * fill_factor]
    gaps = gaps or n_periods * [period * (1 - fill_factor)]

    settings = dict(
        period=period,
        fill_factor=fill_factor,
        fiber_angle_deg=fiber_angle_deg,
        fiber_xposition=fiber_xposition,
        fiber_core_diameter=fiber_core_diameter,
        fiber_numerical_aperture=fiber_core_diameter,
        fiber_nclad=fiber_nclad,
        resolution=resolution,
        ncore=ncore,
        nclad=nclad,
        nsubstrate=nsubstrate,
        n_periods=n_periods,
        box_thickness=box_thickness,
        clad_thickness=clad_thickness,
        etch_depth=etch_depth,
        wavelength_min=wavelength_min,
        wavelength_max=wavelength_max,
        wavelength_points=wavelength_points,
        decay_by=decay_by,
        dtaper=dtaper,
        widths=widths,
        gaps=gaps,
        ncores=ncores,
    )
    settings_string = to_string(settings)
    settings_hash = hashlib.md5(settings_string.encode()).hexdigest()[:8]

    filename = f"fiber_{settings_hash}.yml"
    dirpath = dirpath or pathlib.Path(__file__).parent / "data"
    dirpath = pathlib.Path(dirpath)
    dirpath.mkdir(exist_ok=True, parents=True)
    filepath = dirpath / filename
    filepath_csv = filepath.with_suffix(".csv")

    length_grating = np.sum(widths) + np.sum(gaps)

    substrate_thickness = 1.0
    hair = 4
    core_material = mp.Medium(index=ncore)
    clad_material = mp.Medium(index=nclad)
    fiber_angle = np.radians(fiber_angle_deg)

    y_offset = 0

    # Minimally-parametrized computational cell
    # Could be further optimized
    dpml = 1

    # X-domain
    dbufferx = 0.5
    if length_grating + dtaper < 3 * fiber_core_diameter:
        sxy = 3 * fiber_core_diameter + 2 * dbufferx + 2 * dpml
    else:  # Fiber probably to the left
        sxy = 3 / 2 * fiber_core_diameter + length_grating / 2 + 2 * dbufferx + 2 * dpml

    # Useful reference points
    cell_edge_left = -sxy / 2 + dbufferx + dpml
    grating_start = -fiber_xposition

    # Y-domain (using z notation from 3D legacy code)
    dbuffery = 0.5
    sz = (
        2 * dbuffery
        + box_thickness
        + core_thickness
        + hair
        + substrate_thickness
        + 2 * dpml
    )

    # Initialize domain x-z plane simulation
    cell_size = mp.Vector3(sxy, sz)

    # Ports (position, sizes, directions)
    fiber_offset_from_angle = (clad_thickness + core_thickness) * np.tan(fiber_angle)
    fiber_port_center = mp.Vector3(
        (0.5 * sz - dpml + y_offset - 1) * np.sin(fiber_angle)
        + cell_edge_left
        + 3 / 2 * fiber_core_diameter
        - fiber_offset_from_angle,
        0.5 * sz - dpml + y_offset - 1,
    )
    fiber_port_size = mp.Vector3(3 * fiber_core_diameter, 0, 0)
    fiber_port_direction = mp.Vector3(y=-1).rotate(mp.Vector3(z=1), -1 * fiber_angle)

    waveguide_port_center = mp.Vector3(-sxy / 4)  # grating_start - dtaper, 0)
    waveguide_port_size = mp.Vector3(0, 2 * clad_thickness - 0.2)
    waveguide_port_direction = mp.X

    # Geometry
    fiber_clad = 120
    hfiber_geom = 100  # Some large number to make fiber extend into PML

    fiber_ncore = (fiber_numerical_aperture ** 2 + fiber_nclad ** 2) ** 0.5
    fiber_clad_material = mp.Medium(index=fiber_nclad)
    fiber_core_material = mp.Medium(index=fiber_ncore)

    geometry = []
    # Fiber (defined first to be overridden)
    geometry.append(
        mp.Block(
            material=fiber_clad_material,
            center=mp.Vector3(
                x=grating_start + fiber_xposition - fiber_offset_from_angle
            ),
            size=mp.Vector3(fiber_clad, hfiber_geom),
            e1=mp.Vector3(x=1).rotate(mp.Vector3(z=1), -1 * fiber_angle),
            e2=mp.Vector3(y=1).rotate(mp.Vector3(z=1), -1 * fiber_angle),
        )
    )
    geometry.append(
        mp.Block(
            material=fiber_core_material,
            center=mp.Vector3(
                x=grating_start + fiber_xposition - fiber_offset_from_angle
            ),
            size=mp.Vector3(fiber_core_diameter, hfiber_geom),
            e1=mp.Vector3(x=1).rotate(mp.Vector3(z=1), -1 * fiber_angle),
            e2=mp.Vector3(y=1).rotate(mp.Vector3(z=1), -1 * fiber_angle),
        )
    )

    # clad
    geometry.append(
        mp.Block(
            material=clad_material,
            center=mp.Vector3(0, clad_thickness / 2),
            size=mp.Vector3(mp.inf, clad_thickness),
        )
    )
    # BOX
    geometry.append(
        mp.Block(
            material=clad_material,
            center=mp.Vector3(0, -0.5 * box_thickness),
            size=mp.Vector3(mp.inf, box_thickness),
        )
    )

    # waveguide
    geometry.append(
        mp.Block(
            material=core_material,
            center=mp.Vector3(0, core_thickness / 2),
            size=mp.Vector3(mp.inf, core_thickness),
        )
    )

    # grating etch
    x = grating_start
    for width, gap in zip(widths, gaps):
        geometry.append(
            mp.Block(
                material=clad_material,
                center=mp.Vector3(x + gap / 2, core_thickness - etch_depth / 2),
                size=mp.Vector3(gap, etch_depth),
            )
        )
        x += width + gap

    # Substrate
    geometry.append(
        mp.Block(
            material=mp.Medium(index=nsubstrate),
            center=mp.Vector3(
                0,
                -0.5 * (core_thickness + substrate_thickness + dpml + dbuffery)
                - box_thickness,
            ),
            size=mp.Vector3(mp.inf, substrate_thickness + dpml + dbuffery),
        )
    )

    # PMLs
    boundary_layers = [mp.PML(dpml)]

    # mode frequency
    fcen = 1 / wavelength

    # Waveguide source
    sources_directions = [mp.X]
    sources = [
        mp.EigenModeSource(
            src=mp.GaussianSource(fcen, fwidth=0.1 * fcen),
            size=waveguide_port_size,
            center=waveguide_port_center,
            eig_band=1,
            direction=sources_directions[0],
            eig_match_freq=True,
            eig_parity=mp.ODD_Z,
        )
    ]

    # symmetries = [mp.Mirror(mp.Y,-1)]
    symmetries = []

    # Ports
    waveguide_monitor_port = mp.ModeRegion(
        center=waveguide_port_center + mp.Vector3(x=0.2), size=waveguide_port_size
    )
    fiber_monitor_port = mp.ModeRegion(
        center=fiber_port_center - mp.Vector3(y=0.2), size=fiber_port_size
    )

    # Plotting
    epsilons = []
    for item in geometry:
        epsilons.append(item.material.epsilon(1 / 1.55))
    epsilons = np.array(epsilons)[:, 0, 0]

    eps_parameters = {}
    eps_parameters["contour"] = True
    eps_parameters["levels"] = np.unique(epsilons)

    # Running
    if not run:
        sim = mp.Simulation(
            resolution=resolution,
            cell_size=cell_size,
            boundary_layers=boundary_layers,
            geometry=geometry,
            # geometry_center=mp.Vector3(x_offset, y_offset),
            sources=sources,
            dimensions=2,
            symmetries=symmetries,
            eps_averaging=False,  # Turn off subpixel averaging to better look at the geometry
        )
        """
        waveguide_monitor = sim.add_mode_monitor(
            freqs, waveguide_monitor_port, yee_grid=True
        )
        fiber_monitor = sim.add_mode_monitor(freqs, fiber_monitor_port)
        """
        sim.init_sim()
        # plotStructure_fromSimulation(
        #     sim,
        #     geometry,
        #     waveguide_monitor_port,
        #     waveguide_port_direction,
        #     fiber_monitor_port,
        #     fiber_port_direction,
        #     colorbar=False,
        # )
        # sim.plot2D()

        # eps_parameters['discrete'] = True
        # eps_parameters['discrete_eps_levels'] = epsilons
        # eps_parameters['cmap'] = 'viridis'
        # eps_parameters['discrete_eps_colors'] = ['red', 'blue', 'green', 'yellow']
        sim.plot2D(eps_parameters=eps_parameters)

        filepath.write_text(omegaconf.OmegaConf.to_yaml(settings))
        print(f"write {filepath}")
        return pd.DataFrame()

    if filepath_csv.exists() and not overwrite:
        return pd.read_csv(filepath_csv)
    else:
        sim = mp.Simulation(
            resolution=resolution,
            cell_size=cell_size,
            boundary_layers=boundary_layers,
            geometry=geometry,
            sources=sources,
            dimensions=2,
            symmetries=symmetries,
            eps_averaging=True,
        )
        waveguide_monitor = sim.add_mode_monitor(
            freqs, waveguide_monitor_port, yee_grid=True
        )
        fiber_monitor = sim.add_mode_monitor(freqs, fiber_monitor_port)
        start = time.time()
        # Run simulation
        # sim.run(until=400)
        # Location where to monitor fields decay
        field_monitor_point = (0, 0, 0)
        if animate:
            # Run while saving fields
            # sim.use_output_directory()
            animate = mp.Animate2D(
                sim,
                fields=mp.Ez,
                realtime=False,
                normalize=True,
                eps_parameters=eps_parameters,
                field_parameters={
                    "alpha": 0.8,
                    "cmap": "RdBu",
                    "interpolation": "none",
                },
                boundary_parameters={
                    "hatch": "o",
                    "linewidth": 1.5,
                    "facecolor": "y",
                    "edgecolor": "b",
                    "alpha": 0.3,
                },
            )

            sim.run(
                mp.at_every(1, animate),
                until_after_sources=mp.stop_when_fields_decayed(
                    dt=50, c=mp.Ez, pt=field_monitor_point, decay_by=decay_by
                ),
            )
            animate.to_mp4(30, "testvideo.mp4")
            # sim.run(
            #     mp.at_every(0.6, mp.output_efield_z),
            #     until=1
            # until_after_sources=mp.stop_when_fields_decayed(
            #    dt=50, c=mp.Ez, pt=field_monitor_point, decay_by=decay_by
            #    )
            # )
            # Generate MP4 from fields
            # animateFields(
            #     sim,
            #     geometry,
            #     waveguide_monitor_port,
            #     waveguide_port_direction,
            #     fiber_monitor_port,
            #     fiber_port_direction,
            # )
            # Delete fields

        else:
            sim.run(
                until_after_sources=mp.stop_when_fields_decayed(
                    dt=50, c=mp.Ez, pt=field_monitor_point, decay_by=decay_by
                )
            )

        # Extract mode information
        waveguide_mode = sim.get_eigenmode_coefficients(
            waveguide_monitor,
            [1],
            eig_parity=mp.ODD_Z,
            direction=waveguide_port_direction,
        )
        fiber_mode = sim.get_eigenmode_coefficients(
            fiber_monitor,
            [1],
            direction=mp.NO_DIRECTION,
            eig_parity=mp.ODD_Z,
            kpoint_func=lambda f, n: mp.Vector3(0, fcen * 1.45, 0).rotate(
                mp.Vector3(z=1), -1 * np.radians(fiber_angle_deg)
            ),  # Hardcoded index for now, pull from simulation eventually
        )
        end = time.time()

        a1 = waveguide_mode.alpha[:, :, 0].flatten()  # forward wave
        b1 = waveguide_mode.alpha[:, :, 1].flatten()  # backward wave

        # Since waveguide port is oblique, figure out forward and backward direction
        kdom_fiber = fiber_mode.kdom[0]
        idx = 1 - (kdom_fiber.y > 0) * 1

        print(kdom_fiber)
        print(idx)

        a2 = fiber_mode.alpha[:, :, idx].flatten()  # forward wave
        b2 = fiber_mode.alpha[:, :, 1 - idx].flatten()  # backward wave

        s11 = np.squeeze(b1 / a1)
        s12 = np.squeeze(a2 / a1)
        s22 = s11.copy()
        s21 = s12.copy()

        simulation = dict(
            settings=settings,
            compute_time_seconds=end - start,
        )
        filepath.write_text(omegaconf.OmegaConf.to_yaml(simulation))

        r = dict(s11=s11, s12=s12, s21=s21, s22=s22, wavelengths=wavelengths)
        keys = [key for key in r.keys() if key.startswith("s")]
        s = {f"{key}a": list(np.unwrap(np.angle(r[key].flatten()))) for key in keys}
        s.update({f"{key}m": list(np.abs(r[key].flatten())) for key in keys})
        s["wavelength"] = wavelengths

        df = pd.DataFrame(s, index=wavelengths)
        df.to_csv(filepath_csv, index=False)
        return df


# remove silicon to clearly see the fiber (for debugging)
# fiber_no_silicon = partial(fiber, ncore=nSiO2, nsubstrate=nSiO2, run=False)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # import matplotlib
    # matplotlib.use('TkAgg')
    # # fiber_no_silicon()
    # print(fiber_ncore(0.14, nSiO2))
    # fiber(run=False, fiber_xposition=0, )
    # fiber(run=True)
    # fiber_no_silicon()
    # df = fiber(run=True, animate=False, overwrite=True, fiber_xposition=1)

    # fire.Fire(fiber)

    # One way to do it
    fiber(run=True, animate=True, overwrite=True)

    # # Another way:
    # from grating_coupler_meep.get_simulation_fiber import get_GC_simulation
    # from grating_coupler_meep.get_Sparameters_fiber import get_Sparameters_fiber

    # sim_dict = get_GC_simulation(fiber_xposition=1, fiber_angle_deg=15)
    # df = get_Sparameters_fiber(sim_dict, overwrite=True, verbosity=2)
