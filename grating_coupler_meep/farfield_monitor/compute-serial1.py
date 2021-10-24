import numpy as np
import subprocess
from datetime import datetime

# Run commands in parallel
processes = []

num_processors = 10  # number of processors per MEEP execution
periods = np.linspace(0.2, 0.4, 11)
FFs = np.linspace(0.1, 0.9, 17)
xs = [0]
thetas = [0]

# Generate commands
commands = []
for period in periods:
    for FF in FFs:
        for theta in thetas:
            for x in xs:
                for source in [1]:

                    params = {
                        "a": period,
                        "FF": FF,
                        "theta": theta,
                        "x": x,
                        "source": source,
                    }

                    now = datetime.now()
                    dt_string = now.strftime("%d%m%Y_%H:%M:%S")
                    filename = dt_string + "_"
                    for param in params.keys():
                        filename += param
                        filename += "_"
                        filename += "{:1.4f}".format(params[param])
                        filename += "_"

                    command = "mpirun -np {6} python gc_outcoupler2.py -period {0:} -FF {1:} -theta {2:} -x {3:} -source {4:} -filename {5:} > ./logs/{5:}.log".format(
                        period, FF, theta, x, source, filename, num_processors
                    )

                    subprocess.call(command, shell=True)
