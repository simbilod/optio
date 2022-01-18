# Sample Test passing with nose and pytest

import numpy as np
import matplotlib.pyplot as plt
from optio.get_simulation_fiber import get_simulation_fiber 
from optio.get_Sparameters_fiber import get_Sparameters_fiber 


nm = 1e-3

def test_pass():
    assert True, "dummy sample test"

def fiber_ncore(fiber_numerical_aperture, fiber_nclad):
    return (fiber_numerical_aperture ** 2 + fiber_nclad ** 2) ** 0.5

def test_benchmark():
    """
    Compares to Chrostowski and Hochberg Ch. 5
    """

    fiber_numerical_aperture = float(np.sqrt(1.44427**2 - 1.43482**2))

    sim_dict = get_simulation_fiber(
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
        fiber_nclad = 1.43482,
        # material parameters
        ncore = 3.47,
        ncladtop = 1.44,
        ncladbottom = 1.44,
        nsubstrate = 3.47,
        # other stack parameters
        pml_thickness = 1.0,
        substrate_thickness = 1.0,
        bottom_clad_thickness = 2.0,
        core_thickness = 220 * nm,
        top_clad_thickness = 2.0,
        air_gap_thickness = 1.0,
        fiber_thickness = 2.0,
        # simulation parameters
        res = 100,  # pixels/um
        wavelength_min = 1.4,
        wavelength_max = 1.7,
        wavelength_points = 150,
        fiber_port_y_offset_from_air = 1,
        waveguide_port_x_offset_from_grating_start = 10
    )
    df = get_Sparameters_fiber(sim_dict, overwrite=True, verbosity=2)

    plt.plot(df['wavelength'], df['s12m'])
    plt.plot(df['wavelength'], df['s11m'])
    plt.show()



if __name__ == "__main__":
    test_benchmark()