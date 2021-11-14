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

nm = 1e-3
nSi = 3.47
nSiO2 = 1.45

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


def fiber(
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
    dirpath: Optional[str] = None,
    decay_by: float = 1e-3,
    dtaper: float = 1,
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

    dbuffer = 0.5
    dpml = 1

    fiber_clad = 120
    fiber_angle = np.radians(fiber_angle_deg)
    hfiber_geom = 100  # Some large number to make fiber extend into PML

    fiber_ncore = (fiber_numerical_aperture ** 2 + fiber_nclad ** 2) ** 0.5
    fiber_clad_material = mp.Medium(index=fiber_nclad)
    fiber_core_material = mp.Medium(index=fiber_ncore)

    # MEEP's computational cell is always centered at (0,0), but code has beginning of grating at (0,0)
    sxy = 2 * dpml + dtaper + length_grating + 2 * dbuffer
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
    x = -length_grating / 2
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

    waveguide_port_center = mp.Vector3(-dtaper - length_grating / 2, 0) - offset_vector
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
        filepath.write_text(omegaconf.OmegaConf.to_yaml(settings))
        print(f"write {filepath}")
        return pd.DataFrame()

    if filepath_csv.exists() and not overwrite:
        return pd.read_csv(filepath_csv)

    else:
        start = time.time()
        # Run simulation
        # sim.run(until=400)
        field_monitor_point = (-dtaper, 0, 0)
        sim.run(
            until_after_sources=mp.stop_when_fields_decayed(
                dt=50, c=mp.Ez, pt=field_monitor_point, decay_by=decay_by
            )
        )

        # Extract mode information
        transmission_waveguide = sim.get_eigenmode_coefficients(
            waveguide_monitor, [1], eig_parity=mp.ODD_Z, direction=mp.X
        ).alpha
        kpoint = mp.Vector3(y=-1).rotate(mp.Vector3(z=1), -1 * fiber_angle_deg)
        reflection_fiber = sim.get_eigenmode_coefficients(
            fiber_monitor,
            [1],
            direction=mp.NO_DIRECTION,
            eig_parity=mp.ODD_Z,
            kpoint_func=lambda f, n: kpoint,
        ).alpha
        end = time.time()

        a1 = transmission_waveguide[:, :, 0].flatten()  # forward wave
        b1 = transmission_waveguide[:, :, 1].flatten()  # backward wave
        a2 = reflection_fiber[:, :, 0].flatten()  # forward wave
        b2 = reflection_fiber[:, :, 1].flatten()  # backward wave

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

        df = pd.DataFrame(s, index=wavelengths)
        df.to_csv(filepath_csv, index=True)
        return df


# remove silicon to clearly see the fiber (for debugging)
fiber_no_silicon = partial(fiber, ncore=nSiO2, nsubstrate=nSiO2, run=False)


if __name__ == "__main__":
    # import matplotlib.pyplot as plt

    # fiber(run=False, fiber_xposition=13)
    # plt.show()
    # fiber_no_silicon()

    fire.Fire(fiber)
