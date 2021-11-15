from typing import Optional
import subprocess
import multiprocessing
from functools import partial


def run(
    value: str,
    key: str,
    ncores: int = 6,
):
    command = f"mpirun -np {ncores} python fiber.py --{key}={value}"
    print(command)
    subprocess.call(command, shell=True)


run_fiber_xposition = partial(run, key="fiber_xposition")
run_fiber_angle_deg = partial(run, key="fiber_angle_deg")
run_ncores = partial(run, key="ncores")

if __name__ == "__main__":
    p = multiprocessing.Pool(multiprocessing.cpu_count())
    p.starmap(run_fiber_angle_deg, ncores=[1, 2, 4, 8, 16])

    # p.starmap(run_fiber_angle_deg, fiber_angle_deg=range(10, 30, 5))
    # for fiber_xposition in range(-5, 6):
    #     p = multiprocessing.Process(target=run, args=(fiber_xposition=fiber_xposition))
    #     p.start()
