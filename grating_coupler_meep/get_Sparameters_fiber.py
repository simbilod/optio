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


def get_Sparameters_fiber(
    sim_dict,
    run: bool = True,
    animate: bool = False,
    overwrite: bool = False,
    dirpath: Optional[str] = None,
    decay_by: float = 1e-3,
    ncores: int = 1,
    verbosity=0,
) -> pd.DataFrame:
    """Returns simulation results from grating coupler with fiber.

    Args:
        decay_by: 1e-3
        ncores: number of cores for MPI (give it here?)

    """

    mp.verbosity(verbosity)

    settings = sim_dict["settings"]
    settings.update(
        {
            "decay_by": decay_by,
            "ncores": ncores,
        }
    )

    settings_string = to_string(settings)
    settings_hash = hashlib.md5(settings_string.encode()).hexdigest()[:8]

    filename = f"fiber_{settings_hash}.yml"
    dirpath = dirpath or pathlib.Path(__file__).parent / "data"
    dirpath = pathlib.Path(dirpath)
    dirpath.mkdir(exist_ok=True, parents=True)
    filepath = dirpath / filename
    filepath_csv = filepath.with_suffix(".csv")
    filepath_mp4 = filepath.with_suffix(".mp4")

    if filepath_csv.exists() and not overwrite:
        return pd.read_csv(filepath_csv)
    else:
        sim = sim_dict["sim"]
        freqs = sim_dict["freqs"]
        start = time.time()
        # Run simulation
        # Locations where to monitor fields decay
        termination = []
        for monitor in [sim_dict["waveguide_monitor"], sim_dict["fiber_monitor"]]:
            termination.append(
                mp.stop_when_fields_decayed(
                    dt=50,
                    c=mp.Ez,
                    pt=monitor.regions[0].center,
                    decay_by=1e-9,
                )
            )
        if animate:
            # Run while saving fields
            # sim.use_output_directory()
            animate = mp.Animate2D(
                sim,
                fields=mp.Ez,
                realtime=False,
                normalize=True,
                eps_parameters={"contour": True},
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
                until_after_sources=termination)
            animate.to_mp4(15, filepath_mp4)

        else:
            sim.run(
                until_after_sources=termination)

        # Extract mode information
        waveguide_monitor = sim_dict["waveguide_monitor"]
        waveguide_port_direction = sim_dict["waveguide_port_direction"]
        fiber_monitor = sim_dict["fiber_monitor"]
        fiber_angle_deg = sim_dict["fiber_angle_deg"]
        fcen = sim_dict["fcen"]
        wavelengths = 1 / freqs

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


if __name__ == "__main__":
    from grating_coupler_meep.get_simulation_fiber import get_simulation_fiber_clean

    fiber_numerical_aperture = float(np.sqrt(1.44427**2 - 1.43482**2))

    sim_dict = get_simulation_fiber_clean(
        # grating parameters
        period = 0.66,
        fill_factor = 0.5,
        n_periods = 50,
        etch_depth = 70 * nm,
        # fiber parameters,
        fiber_angle_deg = 10.0,
        fiber_xposition = 0.0,
        fiber_core_diameter = 9,
        fiber_numerical_aperture = fiber_numerical_aperture,
        fiber_nclad = nSiO2,
        # material parameters
        ncore = nSi,
        ncladtop = nSiO2,
        ncladbottom = nSiO2,
        nsubstrate = nSi,
        # stack parameters
        pml_thickness = 1.0,
        substrate_thickness = 1.0,
        bottom_clad_thickness = 2.0,
        core_thickness = 220 * nm,
        top_clad_thickness = 2.0,
        air_gap_thickness = 1.0,
        fiber_thickness = 2.0,
        # simulation parameters
    )
    df = get_Sparameters_fiber(sim_dict, overwrite=True, verbosity=2)
