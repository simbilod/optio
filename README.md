# optio

Sample scripts to perform 2D grating coupler simulation using the open-source FDTD software MEEP

Folder "far field" contains scripts to simulate the radiation pattern of a waveguide + grating.
Folder "fiber" contains scripts to simulate the S-parameters of the fundamental TE mode of the waveguide and a (parametrized) fiber.

USAGE

Example use:
```
# Define and run simulation
df = fiber(run=True, overwrite=True)
# Analyze results
<...>
```
`df` is a Pandas dataframe containing the S-parameters vs wavelength. If `overwrite` is true, a `csv` file is (over)written with the results. Further executions of the function will load the results, allowing easy plotting in other scripts or notebooks.

TODO

* Unittests w/ quantitative comparison to benchmark
* Improve parameter sweep interface
* Add optimization (NLopt)

Clean up

References

- [1] Chrostowski, L., & Hochberg, M. (2015). Optical I/O. Silicon Photonics Design: From Devices to Systems. Cambridge University Press. doi: 10.1017/CBO9781316084168.006
- [2] https://support.lumerical.com/hc/en-us/articles/360042305334-Grating-coupler
- [3] http://www.simpetus.com/projects.html#meep_outcoupler
