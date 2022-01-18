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

nm = 1E-3

from optio.get_Sparameters_fiber import write_sparameters_meep_parallel_pools

if __name__ == "__main__":

    # For shared parameters across simulations
    base_instance = dict(
        # grating parameters
        period=0.66,
        fill_factor=0.5,
        n_periods=50,
        etch_depth=70 * nm,
        # fiber parameters,
        fiber_angle_deg=10.0,
        fiber_xposition=0.0,
        fiber_core_diameter=9,
        fiber_numerical_aperture=4,
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
        res=80,  # pixels/um
        wavelength_min=1.5,
        wavelength_max=1.6,
        wavelength_points=50,
        fiber_port_y_offset_from_air=1,
        waveguide_port_x_offset_from_grating_start=10,
        # Computation parameters
        overwrite=True,
        verbosity=0,
        decay_by=1e-3,
    )

    instance2 = base_instance.copy()
    instance2["period"] = 0.5

    widths_list = []
    gaps_list = []

    write_sparameters_meep_parallel_pools(
        instances=[instance1, instance2],
        cores_per_instance=5,
        total_cores=50,
        verbosity=True,
        delete_temp_files=True,
    )