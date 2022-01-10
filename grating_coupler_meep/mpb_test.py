import meep as mp
import numpy as np
from meep import mpb
import matplotlib.pyplot as plt

# Parameters for the waveguide
w = 5  # core width (um)

# Materials
clad = mp.Medium(index=1.44)
core = mp.Medium(index=3.5)  # 1.446789549312546

sc = 30  # supercell width (um)
resolution = 64  # pixels/um
geometry_lattice = mp.Lattice(size=mp.Vector3(sc, 0, 0))

# define the 2d blocks for the strip and substrate
geometry = [
    mp.Block(size=mp.Vector3(w, mp.inf, mp.inf), center=mp.Vector3(), material=core),
    mp.Block(
        size=mp.Vector3(mp.inf, mp.inf, mp.inf), center=mp.Vector3(), material=clad
    ),
]

# The k (i.e. beta, i.e. propagation constant) points to look at, in
# units of 2*pi/um.  We'll look at num_k points from k_min to k_max.
num_k = 20
k_min = 0.1
k_max = 4.0
k_points = mp.interpolate(num_k, [mp.Vector3(k_min), mp.Vector3(k_max)])

# Increase this to see more modes.  (The guided ones are the ones below the
# light line, i.e. those with frequencies < kmag / 1.45, where kmag
# is the corresponding column in the output if you grep for "freqs:".)
num_bands = 1

# ModeSolver object
ms = mpb.ModeSolver(
    geometry_lattice=geometry_lattice,
    geometry=geometry,
    # Add new things pertaining to simulation
    k_points=k_points,
    resolution=resolution,
    num_bands=num_bands,
)

eps = ms.get_epsilon()

print(eps)
