# optIO

Open-source simulation and optimization of OPTical IO to integrated photonic chips through MEEP and MPB [1,2]. Currently supports 2D simulations of grating couplers as referenced in [3,4].

## TODO

* Usage documentation
* Unittests w/ quantitative comparison to benchmark
* Material dispersion
* Improve parameter sweep interface
* Add optimization (NLopt)
* Clean up

## References

- [1] https://github.com/NanoComp/meep
- [2] http://www.simpetus.com/projects.html#meep_outcoupler
- [3] Wang, Y., Flueckiger, J., Lin, C., & Chrostowski, L. (2013). Universal grating coupler design. Photonics North 2013. SPIE. doi: 10.1117/12.2042185
- [4] Chrostowski, L., & Hochberg, M. (2015). Optical I/O. Silicon Photonics Design: From Devices to Systems. Cambridge University Press. doi: 10.1017/CBO9781316084168.006
- [5] https://support.lumerical.com/hc/en-us/articles/360042305334-Grating-coupler

## Acknowledgements

* Simon Bilodeau: maintainer
* Joaquin Matres Abril: packagification, basic sweeping and plotting routines
