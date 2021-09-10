import numpy as np
from string import *
import subprocess
from datetime import datetime

# Run commands in parallel
processes = []

num_processors = 8 # number of processors per MEEP execution
periods = np.linspace(0.6, 0.9, 21)
FFs = np.linspace(0.1, 0.9, 20)
thetas = [8]
xs = np.linspace(0, 5, 6)

# Generate commands
commands = []
for period in periods:
    for FF in FFs:
        for theta in thetas:
            for x in xs:
                for source in [0,1]:

                    params = {'a': period, 
                    'FF': FF, 
                    'theta': theta,
                    'x': x,
                    'source': source}
                    
                    now = datetime.now()
                    dt_string = now.strftime("%d%m%Y_%H:%M:%S")
                    filename = dt_string + '_'
                    for param in params.keys():
                        filename += param
                        filename += '_'
                        filename += '{:1.4f}'.format(params[param])
                        filename += '_'

                    commands.append('mpirun -np {6} python gc_outcoupler.py -period {0:} -FF {1:} -theta {2:} -x {3:} -source {4:} -filename {5:} > ./logs/{5:}.log'.format(period, FF, theta, x, source, filename, num_processors) )

# We have 60 cores, so stagger execution
def chunks(l, n):
    n = max(1, n)
    return (l[i:i+n] for i in range(0, len(l), n))
# Number of simulations
sims = np.shape(periods)[0] * np.shape(FFs)[0] * np.shape(thetas)[0] * np.shape(xs)[0] * 2

# Each chunk fills the cores
commands_chunks = chunks(commands, int(60/num_processors))

for commands_chunk in commands_chunks:
    processes = []
    for command in commands_chunk:
        process = subprocess.Popen(command, shell=True)
        processes.append(process)
    # Collect statuses
    output = [p.wait() for p in processes]
    print(output)
