import hashlib
import time
import pathlib
import omegaconf
import numpy as np
import pandas as pd

import meep as mp
from fiber_draw import draw_grating_coupler_fiber

nm = 1e-3


def clean_value(value) -> str:
    """ """
    if isinstance(value, list):
        value = "_".join(clean_value(v) for v in value)
    return str(value)


def run(
    fiber_angle_deg: float = 15.0,
    wavelength_min: float = 1.5,
    wavelength_max: float = 1.6,
    wavelength_points: int = 50,
    overwrite: bool = False,
    **kwargs,
) -> pd.DataFrame:
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
    wavelengths = np.linspace(wavelength_min, wavelength_max, wavelength_points)

    settings_string = "_".join(settings_string_list)
    settings_hash = hashlib.md5(settings_string.encode()).hexdigest()[:8]

    filename = f"fiber_{settings_hash}.yml"
    filepath = pathlib.Path("data") / filename
    filepath_csv = filepath.with_suffix(".csv")

    if filepath_csv.exists() and not overwrite:
        return pd.read_csv(filepath_csv)

    start = time.time()
    sim, fiber_monitor, waveguide_monitor = draw_grating_coupler_fiber(
        fiber_angle_deg=fiber_angle_deg,
        wavelength_min=wavelength_min,
        wavelength_max=wavelength_max,
        wavelength_points=wavelength_points,
        **kwargs,
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
    end = time.time()
    simulation = dict(
        settings=settings,
        compute_time_seconds=end - start,
        reflection_fiber=reflection_fiber,
        transmission_waveguide=transmission_waveguide,
    )
    filepath.write_text(omegaconf.OmegaConf.to_yaml(simulation))

    results = dict(t=transmission_waveguide, r=reflection_fiber)
    df = pd.DataFrame(results, index=wavelengths)
    df.to_csv(filepath_csv, index=False)
    return df


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
