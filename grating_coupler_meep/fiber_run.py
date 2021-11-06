import hashlib
import pathlib
import omegaconf

import meep as mp
from fiber_draw import draw_grating_coupler_fiber

nm = 1e-3


def clean_value(value) -> str:
    """ """
    if isinstance(value, list):
        value = "_".join(clean_value(v) for v in value)
    return str(value)


def run(fiber_angle_deg: float = 15.0, **kwargs) -> float:
    """Returns grating coupler with fiber coupling efficiency.

    Args:
        fiber_angle_deg: angle fiber in degrees
        kwargs:
    """
    settings = kwargs.copy()
    settings.update(fiber_angle_deg=fiber_angle_deg)

    settings_string_list = [
        f"{key}={clean_value(settings[key])}" for key in sorted(settings.keys())
    ]

    settings_string = "_".join(settings_string_list)
    settings_hash = hashlib.md5(settings_string.encode()).hexdigest()[:8]
    filename = f"fiber_{settings_hash}"
    filepath = pathlib.Path("data" / filename)

    if filepath.exists():
        simulation = omegaconf.OmegaConf.load(filepath)
        return simulation["transmission_waveguide"]

    sim, fiber_monitor, waveguide_monitor = draw_grating_coupler_fiber(
        fiber_angle_deg=fiber_angle_deg, **kwargs
    )

    # Run simulation
    sim.run(until=400)

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
    simulation = dict(
        settings=settings,
        reflection_fiber=reflection_fiber,
        transmission_waveguide=transmission_waveguide,
    )
    filepath.write_text(omegaconf.OmegaConf.to_yaml(simulation))
    return transmission_waveguide


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
