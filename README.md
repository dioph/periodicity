# Periodicity

Useful tools for periodicity analysis in time series data.

[![](https://github.com/dioph/periodicity/workflows/CI/badge.svg)](https://github.com/dioph/periodicity/actions?query=branch%3Amaster)
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

## Installation

The latest version is available to download via PyPI: __`pip install periodicity`__.

Alternatively, you can build the current development version from source by cloning this repo (__`git clone https://github.com/dioph/periodicity.git`__) and running __`pip install ./periodicity`__.

