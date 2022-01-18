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
from typing import Optional, Tuple, Dict

import hashlib
import time
import pathlib
import omegaconf
import pandas as pd
import subprocess
import shlex
import shutil

import meep as mp
import numpy as np
import fire

from optio.get_simulation_fiber import get_simulation_fiber


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


def get_Sparameters_simulation(
    sim_dict,
    run: bool = True,
    animate: bool = False,
    overwrite: bool = False,
    dirpath: Optional[str] = None,
    decay_by: float = 1e-3,
    verbosity=0,
) -> pd.DataFrame:
    """Returns simulation results from grating coupler with fiber.

    Args:

    """

    mp.verbosity(verbosity)

    settings = sim_dict["settings"]
    settings.update(
        {
            "decay_by": decay_by,
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

            sim.run(mp.at_every(1, animate), until_after_sources=termination)
            animate.to_mp4(15, filepath_mp4)

        else:
            sim.run(until_after_sources=termination)

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


def get_Sparameters_fiber(
    # grating parameters
    period: float = 0.66,
    fill_factor: float = 0.5,
    widths: Optional[Floats] = None,
    gaps: Optional[Floats] = None,
    n_periods: int = 30,
    etch_depth: float = 70 * nm,
    # fiber parameters,
    fiber_angle_deg: float = 20.0,
    fiber_xposition: float = 1.0,
    fiber_core_diameter: float = 10.4,
    fiber_numerical_aperture: float = 0.14,
    fiber_nclad: float = nSiO2,
    # material parameters
    ncore: float = nSi,
    ncladtop: float = nSiO2,
    ncladbottom: float = nSiO2,
    nsubstrate: float = nSi,
    # stack parameters
    pml_thickness: float = 1.0,
    substrate_thickness: float = 1.0,
    bottom_clad_thickness: float = 2.0,
    core_thickness: float = 220 * nm,
    top_clad_thickness: float = 2.0,
    air_gap_thickness: float = 1.0,
    fiber_thickness: float = 2.0,
    # simulation parameters
    res: int = 64,  # pixels/um
    wavelength_min: float = 1.4,
    wavelength_max: float = 1.7,
    wavelength_points: int = 150,
    eps_averaging: bool = False,
    fiber_port_y_offset_from_air: float = 1,
    waveguide_port_x_offset_from_grating_start: float = 10,
    # Calculation settings
    run: bool = True,
    animate: bool = False,
    overwrite: bool = False,
    dirpath: Optional[str] = None,
    decay_by: float = 1e-3,
    ncores: int = 1,
    verbosity=0,
) -> pd.DataFrame:

    print("getting simulation")
    sim_dict = get_simulation_fiber(
        # grating parameters
        period=period,
        fill_factor=fill_factor,
        widths=widths,
        gaps=gaps,
        n_periods=n_periods,
        etch_depth=etch_depth,
        fiber_angle_deg=fiber_angle_deg,
        fiber_xposition=fiber_xposition,
        fiber_core_diameter=fiber_core_diameter,
        fiber_numerical_aperture=fiber_numerical_aperture,
        fiber_nclad=fiber_nclad,
        ncore=ncore,
        ncladtop=ncladtop,
        ncladbottom=ncladbottom,
        nsubstrate=nsubstrate,
        pml_thickness=pml_thickness,
        substrate_thickness=substrate_thickness,
        bottom_clad_thickness=bottom_clad_thickness,
        core_thickness=core_thickness,
        top_clad_thickness=top_clad_thickness,
        air_gap_thickness=air_gap_thickness,
        fiber_thickness=fiber_thickness,
        res=res,  # pixels/um
        wavelength_min=wavelength_min,
        wavelength_max=wavelength_max,
        wavelength_points=wavelength_points,
        eps_averaging=eps_averaging,
        fiber_port_y_offset_from_air=fiber_port_y_offset_from_air,
        waveguide_port_x_offset_from_grating_start=waveguide_port_x_offset_from_grating_start,
    )
    print("computing Sparams")
    df = get_Sparameters_simulation(
        sim_dict,
        run=run,
        animate=animate,
        overwrite=overwrite,
        dirpath=dirpath,
        decay_by=decay_by,
        verbosity=verbosity,
    )
    return df


def write_sparameters_meep_parallel(
    instance: Dict,
    cores: int = 2,
    temp_dir: Optional[str] = None,
    temp_file_str: str = "write_sparameters_meep_parallel",
    verbosity: bool = False,
):
    """
    Given a Dict of write_sparameters_meep keyword arguments (the "instance"), launches a parallel simulation on `cores` cores
    Returns the subprocess Popen object

    Args
        instances (Dict): Dict. The keys must be parameters names of write_sparameters_meep, and entries the values
        cores (int): number of processors
        temp_dir (FilePath): temporary directory to hold simulation files
        temp_file_str (str): names of temporary files in temp_dir
        verbosity (bool): progress messages
    """

    # Save the component object to simulation for later retrieval
    temp_dir = temp_dir or pathlib.Path(__file__).parent / "temp"
    temp_dir = pathlib.Path(temp_dir)
    temp_dir.mkdir(exist_ok=True, parents=True)
    filepath = temp_dir / temp_file_str

    # Write execution file
    script_lines = [
        "from optio.get_Sparameters_fiber import get_Sparameters_fiber\n\n",
        'if __name__ == "__main__":\n\n',
        "\tget_Sparameters_fiber(\n",
    ]
    for key in instance.keys():
        if isinstance(instance[key], str):
            parameter = f'"{instance[key]}"'
        else:
            parameter = instance[key]
        script_lines.append(f"\t\t{key} = {parameter},\n")
    script_lines.append("\t)")
    script_file = filepath.with_suffix(".py")
    script_file_obj = open(script_file, "w")
    script_file_obj.writelines(script_lines)
    script_file_obj.close()

    # Exec string
    command = f"mpirun -np {cores} python {script_file}"

    # Launch simulation
    if verbosity:
        print(f"Launching: {command}")
    proc = subprocess.Popen(
        shlex.split(command),
        shell=False,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    return proc


def write_sparameters_meep_parallel_pools(
    instances: Tuple,
    cores_per_instance: int = 2,
    total_cores: int = 4,
    temp_dir: Optional[str] = None,
    delete_temp_files: bool = False,
    verbosity: bool = False,
):
    """
    Given a tuple of write_sparameters_meep keyword arguments (the "instances"), launches parallel simulations
    Each simulation is assigned "cores_per_instance" cores
    A total of "total_cores" is assumed, if cores_per_instance * len(instances) > total_cores then the overflow will be performed serially

    Args
        instances ([Dict]): list of Dicts. The keys must be parameters names of write_sparameters_meep, and entries the values
        cores_per_instance (int): number of processors to assign to each instance
        total_cores (int): total number of cores to use
        temp_dir (FilePath): temporary directory to hold simulation files
        delete_temp_file (Boolean): whether to delete temp_dir when done
        verbosity: progress messages
    """
    # Save the component object to simulation for later retrieval
    temp_dir = temp_dir or pathlib.Path(__file__).parent / "temp"
    temp_dir = pathlib.Path(temp_dir)
    temp_dir.mkdir(exist_ok=True, parents=True)

    # Setup pools
    num_pools = int(np.ceil(cores_per_instance * len(instances) / total_cores))
    instances_per_pool = int(np.floor(total_cores / cores_per_instance))
    num_tasks = len(instances)

    if verbosity:
        print(f"Running parallel simulations over {num_tasks} instances")
        print(
            f"Using a total of {total_cores} cores with {cores_per_instance} cores per instance"
        )
        print(
            f"Tasks split amongst {num_pools} pools with up to {instances_per_pool} instances each."
        )

    i = 0
    # For each pool
    for j in range(num_pools):
        processes = []
        # For instance in the pool
        for k in range(instances_per_pool):
            # Flag to catch nonfull pools
            if i >= num_tasks:
                continue
            if verbosity:
                print(f"Task {k} of pool {j} is instance {i}")
            # Obtain current instance
            instance = instances[i]

            process = write_sparameters_meep_parallel(
                instance=instance,
                cores=cores_per_instance,
                temp_dir=temp_dir,
                temp_file_str=f"write_sparameters_meep_parallel_{i}",
                verbosity=verbosity,
            )
            processes.append(process)

            # Increment task number
            i += 1

        # Wait for pool to end
        for process in processes:
            process.wait()

    if delete_temp_files:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":

    fiber_numerical_aperture = float(np.sqrt(1.44427 ** 2 - 1.43482 ** 2))

    instance = dict(
        # grating parameters
        period=0.66,
        fill_factor=0.5,
        n_periods=50,
        etch_depth=70 * nm,
        # fiber parameters,
        fiber_angle_deg=10.0,
        fiber_xposition=0.0,
        fiber_core_diameter=9,
        fiber_numerical_aperture=fiber_numerical_aperture,
        fiber_nclad=1.43482,
        # material parameters
        ncore=3.47,
        ncladtop=1.44,
        ncladbottom=1.44,
        nsubstrate=3.47,
        # other stack parameters
        pml_thickness=1.0,
        substrate_thickness=1.0,
        bottom_clad_thickness=2.0,
        core_thickness=220 * nm,
        top_clad_thickness=2.0,
        air_gap_thickness=1.0,
        fiber_thickness=2.0,
        # simulation parameters
        res=20,  # pixels/um
        wavelength_min=1.4,
        wavelength_max=1.7,
        wavelength_points=150,
        fiber_port_y_offset_from_air=1,
        waveguide_port_x_offset_from_grating_start=10,
        # Computation parameters
        overwrite=True,
        verbosity=2,
        decay_by=1e-3,
    )

    write_sparameters_meep_parallel_pools(
        instances=[instance],
        cores_per_instance=4,
        total_cores=4,
        verbosity=True,
    )
