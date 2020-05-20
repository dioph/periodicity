# Periodicity

Useful tools for periodicity analysis in time series data.

[![](https://github.com/dioph/periodicity/workflows/periodicity-tests/badge.svg)](https://github.com/dioph/periodicity/actions?query=branch%3Amaster)
[![PyPI version](https://badge.fury.io/py/periodicity.svg)](https://badge.fury.io/py/periodicity)
[![Downloads](https://pepy.tech/badge/periodicity)](https://pepy.tech/project/periodicity)

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
### Installing current release from PyPI (v1.0b1)
    $ pip install periodicity
### Installing current development version (v1.0b2)
    $ git clone https://github.com/dioph/periodicity.git
    $ cd periodicity
    $ pip install .
