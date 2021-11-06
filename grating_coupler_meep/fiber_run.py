import pickle

import meep as mp
from fiber_draw import draw_grating_coupler_fiber

nm = 1e-3


def run(**kwargs):
    """Draw grating coupler with fiber.

    Args:
        period: grating coupler period

    """

    sim, fiber_monitor, waveguide_monitor = draw_grating_coupler_fiber(**kwargs)

    # Run simulation
    sim.run(until=400)

    # Extract mode information
    res_waveguide = sim.get_eigenmode_coefficients(
        waveguide_monitor, [1], eig_parity=mp.ODD_Z, direction=mp.X
    )
    kpoint = mp.Vector3(y=-1).rotate(mp.Vector3(z=1), -1 * fiber_angle)
    res_fiber = sim.get_eigenmode_coefficients(
        fiber_monitor,
        [1],
        direction=mp.NO_DIRECTION,
        eig_parity=mp.ODD_Z,
        kpoint_func=lambda f, n: kpoint,
    )

    # Save raw data
    params = {
        "a": period,
        "FF": FF,
        "theta": fiber_angle,
        "x": fiber_xposition,
        "source": source,
    }

    filename_dat = "./data/" + filename + ".pickle"

    with open(filename_dat, "wb") as f:
        pickle.dump(
            {"params": params, "res_fiber": res_fiber, "res_waveguide": res_waveguide},
            f,
            pickle.HIGHEST_PROTOCOL,
        )


if __name__ == "__main__":
    # c= run()
    sim, fiber_monitor, waveguide_monitor = draw_grating_coupler_fiber()

    """Run simulation"""
    sim.run(until=400)

    """Extract mode information"""
    res_waveguide = sim.get_eigenmode_coefficients(
        waveguide_monitor, [1], eig_parity=mp.ODD_Z, direction=mp.X
    )
    # kpoint = mp.Vector3(y=-1).rotate(mp.Vector3(z=1), -1 * fiber_angle)
    # res_fiber = sim.get_eigenmode_coefficients(
    #     fiber_monitor,
    #     [1],
    #     direction=mp.NO_DIRECTION,
    #     eig_parity=mp.ODD_Z,
    #     kpoint_func=lambda f, n: kpoint,
    # )

    # # Save raw data
    # params = {
    #     "a": period,
    #     "FF": FF,
    #     "theta": fiber_angle,
    #     "x": fiber_xposition,
    #     "source": source,
    # }
    # filename_dat = "./data/" + filename + ".pickle"
    # with open(filename_dat, "wb") as f:
    #     pickle.dump(
    #         {"params": params, "res_fiber": res_fiber, "res_waveguide": res_waveguide},
    #         f,
    #         pickle.HIGHEST_PROTOCOL,
    #     )
