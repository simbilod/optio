import numpy as np
import meep as mp
import math

import argparse
import pickle


def main(args):

    period = args.period
    FF = args.FF
    fiber_angle = args.theta
    fiber_xposition = args.x
    source = args.source
    filename = args.filename

    resolution = 64  # pixels/unit length (1 um)

    hSiN = 0.44
    hSiO2 = 3.2
    hSi = 1
    hair = 4

    dgrat = period * FF
    dgap = period * (1 - FF)
    # dgrat = 0.767723445279288/2
    # dgap = 0.767723445279288/2

    # a = dgrat + dgap
    a = period

    # Some semi-hardcoded values
    N = 16
    N = N + 1
    dtaper = 12

    dbuffer = 0.5
    dpml = 1

    # Fiber; semi-hardcoded
    # Fiber parameters, from SMF-633-4/125-1-L or PMF-633-4/125-0.25-L
    fiber_core = 4
    fiber_clad = 120
    # fiber_angle = 20
    fiber_angle = np.radians(fiber_angle)
    # fiber_xposition = 5
    fiber_air_gap = 1
    hfiber = 3
    haircore = 2
    hfiber_geom = 100  # Some large number to make fiber extend into PML
    # Index from 0.10 @ 633 nm numerical aperture
    # cladding 635 nm real index, From NA, n=1.4535642400664652
    nClad = 1.4535642400664652
    Clad = mp.Medium(index=nClad)
    # Pure fused silica core core 635 nm real index (will be SiO2 below)

    # MEEP's computational cell is always centered at (0,0), but code has beginning of grating at (0,0)
    sxy = 2 * dpml + dtaper + a * N + 2 * dbuffer  # sx here
    sz = 2 * dbuffer + hSiO2 + hSiN + hair + hSi + 2 * dpml  # sy here
    # comp_origin_x = dpml + dbuffer + dtaper
    comp_origin_x = 0
    # meep_origin_x = sxy/2
    # x_offset = meep_origin_x - comp_origin_x
    x_offset = 0
    # comp_origin_y = dpml + hSi + hSiO2 + hSiN/2
    comp_origin_y = 0
    # meep_origin_y = sz/2
    # y_offset = meep_origin_y - comp_origin_y
    y_offset = 0

    # x_offset_vector = mp.Vector3(x_offset,0)
    # offset_vector = mp.Vector3(x_offset, y_offset)
    offset_vector = mp.Vector3(0, 0, 0)

    # Si3N4 635 nm real index
    nSiN = 2.0102
    SiN = mp.Medium(index=nSiN)
    # SiO2 635 nm real index
    nSiO2 = 1.4569
    SiO2 = mp.Medium(index=nSiO2)
    # Si substrate 635 nm complex index, following https://meep.readthedocs.io/en/latest/Materials/#conductivity-and-complex
    # eps = 15.044 + i*0.14910
    Si = mp.Medium(
        epsilon=15.044, D_conductivity=2 * math.pi * 0.635 * 0.14910 / 15.044
    )

    # We will do x-z plane simulation
    cell_size = mp.Vector3(sxy, sz)

    geometry = []

    # Fiber (defined first to be overridden)

    # Core
    # fiber_offset = mp.Vector3(fiber_xposition + extrax, hSiN/2 + hair + haircore + extray) - offset_vector
    geometry.append(
        mp.Block(
            material=Clad,
            center=mp.Vector3(x=fiber_xposition) - offset_vector,
            size=mp.Vector3(fiber_clad, hfiber_geom),
            e1=mp.Vector3(x=1).rotate(mp.Vector3(z=1), -1 * fiber_angle),
            e2=mp.Vector3(y=1).rotate(mp.Vector3(z=1), -1 * fiber_angle),
        )
    )
    geometry.append(
        mp.Block(
            material=SiO2,
            center=mp.Vector3(x=fiber_xposition) - offset_vector,
            size=mp.Vector3(fiber_core, hfiber_geom),
            e1=mp.Vector3(x=1).rotate(mp.Vector3(z=1), -1 * fiber_angle),
            e2=mp.Vector3(y=1).rotate(mp.Vector3(z=1), -1 * fiber_angle),
        )
    )

    # air
    geometry.append(
        mp.Block(
            material=mp.air,
            center=mp.Vector3(0, haircore / 2) - offset_vector,
            size=mp.Vector3(mp.inf, haircore),
        )
    )

    # waveguide
    geometry.append(
        mp.Block(
            material=SiN,
            center=mp.Vector3(0, 0) - offset_vector,
            size=mp.Vector3(mp.inf, hSiN),
        )
    )

    # grating etch
    for n in range(0, N):
        geometry.append(
            mp.Block(
                material=mp.air,
                center=mp.Vector3(n * a + dgap / 2, 0) - offset_vector,
                size=mp.Vector3(dgap, hSiN),
            )
        )

    geometry.append(
        mp.Block(
            material=mp.air,
            center=mp.Vector3(sxy - comp_origin_x - 0.5 * (dpml + dbuffer), 0)
            - offset_vector,
            size=mp.Vector3(dpml + dbuffer, hSiN),
        )
    )

    # BOX
    geometry.append(
        mp.Block(
            material=SiO2,
            center=mp.Vector3(0, -0.5 * (hSiN + hSiO2)) - offset_vector,
            size=mp.Vector3(mp.inf, hSiO2),
        )
    )

    # Substrate
    geometry.append(
        mp.Block(
            material=Si,
            center=mp.Vector3(0, -0.5 * (hSiN + hSi + dpml + dbuffer) - hSiO2)
            - offset_vector,
            size=mp.Vector3(mp.inf, hSi + dpml + dbuffer),
        )
    )

    # PMLs
    boundary_layers = [mp.PML(dpml)]

    # Source

    # mode frequency
    fcen = 1 / 0.635

    waveguide_port_center = mp.Vector3(-1 * dtaper, 0) - offset_vector
    waveguide_port_size = mp.Vector3(0, 2 * haircore - 0.1)
    fiber_port_center = (
        mp.Vector3(
            (0.5 * sz - dpml + y_offset - 1) * np.sin(fiber_angle) + fiber_xposition,
            0.5 * sz - dpml + y_offset - 1,
        )
        - offset_vector
    )
    fiber_port_size = mp.Vector3(sxy * 3 / 5 - 2 * dpml - 2, 0)

    # Waveguide source
    if source == 1:
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
    # Fiber source
    else:
        sources = [
            mp.EigenModeSource(
                src=mp.GaussianSource(fcen, fwidth=0.1 * fcen),
                size=fiber_port_size,
                center=fiber_port_center,
                eig_band=1,
                direction=mp.NO_DIRECTION,
                eig_kpoint=mp.Vector3(y=-1).rotate(mp.Vector3(z=1), -1 * fiber_angle),
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
        eps_averaging=False,
    )

    # Ports
    waveguide_monitor_port = mp.ModeRegion(
        center=waveguide_port_center + mp.Vector3(x=0.2), size=waveguide_port_size
    )
    waveguide_monitor = sim.add_mode_monitor(
        fcen, 0, 1, waveguide_monitor_port, yee_grid=True
    )
    fiber_monitor_port = mp.ModeRegion(
        center=fiber_port_center - mp.Vector3(y=0.2),
        size=fiber_port_size,
        direction=mp.NO_DIRECTION,
    )
    fiber_monitor = sim.add_mode_monitor(fcen, 0, 1, fiber_monitor_port)

    """Run simulation"""
    if source == 1:  # Waveguide is source, monitor at fiber
        # sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, fiber_port_center, 1e-6))
        sim.run(until=400)
    else:  # Opposite
        # sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, waveguide_port_center, 1e-6))
        sim.run(until=400)

    """Extract mode information"""
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-period",
        type=float,
        default=0.767723445279288,
        help="grating period (default: 0.767723445279288 um)",
    )
    parser.add_argument(
        "-FF", type=float, default=0.5, help="Fill factor (default: 0.5 um)"
    )
    parser.add_argument(
        "-theta", type=float, default=8, help="fiber_angle (default: 8 degrees)"
    )
    parser.add_argument(
        "-x", type=float, default=1, help="Fiber position (default: 1 um)"
    )
    parser.add_argument(
        "-source", type=int, default=0, help="1 for waveguide, 0 for fiber"
    )
    parser.add_argument("-filename", type=str, default="default", help="data name")
    args = parser.parse_args()
    main(args)
