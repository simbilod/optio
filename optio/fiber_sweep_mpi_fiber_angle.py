import subprocess
import multiprocessing


def run(
    key: str,
    value: float,
    ncores: int = 6,
):
    command = f"mpirun -np {ncores} python fiber.py --{key}={value}"
    print(command)
    subprocess.call(command, shell=True)


if __name__ == "__main__":
    p = multiprocessing.Pool(multiprocessing.cpu_count())
    # p.starmap(
    #     run_fiber_xposition,
    #     [("fiber_xposition", fiber_xposition) for fiber_xposition in range(-5, 5)],
    # )

    p.starmap(
        run,
        [("fiber_angle_deg", fiber_angle_deg) for fiber_angle_deg in range(10, 30, 5)],
    )
