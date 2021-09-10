# grating_coupler_meep
Sample scripts to perform 2D grating coupler simulation using the open-source FDTD software MEEP

Folder "far field" contains scripts to simulate the radiation pattern of a waveguide + grating.
Folder "fiber" contains scripts to simulate the S-parameters of the fundamental TE mode of the waveguide and a (parametrized) fiber.

USAGE

Within each folder, python file "gc_outcoupler.py" contains an "initialize" function to create the geometry. It returns the MEEP simulation object and monitors. The "main" function takes this object, performs the simulation, and saves it to a timestamped pickle file with results and inputs.
Jupyter notebooks within each folder help : (1) observe the geometry computed by initialize and (2) parse the pickle files to compare results form various geometries.

TODO

* Add a class to define and hold geometry (instead of semi-hardcoding) and share across analyses
* Extend to partial etch processes
* Improve parameter sweep interface
* Add optimization (NLopt)
* Add tests

Clean up

References
Literature :
[1] Chrostowski, L., & Hochberg, M. (2015). Optical I/O. Silicon Photonics Design: From Devices to Systems. Cambridge University Press. doi: 10.1017/CBO9781316084168.006
Software examples :
[1] https://support.lumerical.com/hc/en-us/articles/360042305334-Grating-coupler
[2] http://www.simpetus.com/projects.html#meep_outcoupler
