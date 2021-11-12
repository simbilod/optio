from subprocess_tee import run

if __name__ == "__main__":
    for fiber_xposition in range(-5, 6):
        command = f"mpirun -np {6} python fiber.py --fiber_xposition={fiber_xposition} > log.log"
        run(command)
