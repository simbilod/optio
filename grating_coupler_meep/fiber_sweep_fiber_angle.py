import multiprocessing
from functools import partial
from fiber import fiber


fiber_sweep_angle = partial(fiber, dirpath="data/fiber_sweep_angle_deg")

if __name__ == "__main__":
    p = multiprocessing.Pool(multiprocessing.cpu_count())

    p.starmap(
        fiber_sweep_angle,
        [
            (0.66, 0.5, None, None, fiber_angle_deg)
            for fiber_angle_deg in range(10, 30, 5)
        ],
    )
