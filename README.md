# Periodicity

Useful tools for analysis of periodicities in time series data.

__Documentation: https://periodicity.readthedocs.io__

Currently includes:
* Auto-Correlation Function
* Spectral methods:
    * Lomb-Scargle periodogram
    * Wavelet Transform
    * Hilbert-Huang Transform (WIP)
* Phase-folding methods:
    * String Length
    * Phase Dispersion Minimization
    * Analysis of Variance (soon™)
* Gaussian Processes:
    * `george` implementation
    * `celerite` implementation
    * `pymc3` implementation (soon™)

## Quick start
### Installing current release from pypi (v0.1.0b2)
    $ pip install periodicity
### Installing current development version (v1.0b1)
    $ git clone https://github.com/dioph/periodicity.git
    $ cd periodicity
    $ pip install .

### Visualization of this example:

![gp_example](https://raw.githubusercontent.com/dioph/periodicity/master/docs/source/_static/images/example2.png)

![gp_example](https://raw.githubusercontent.com/dioph/periodicity/master/docs/source/_static/images/example1.png)
