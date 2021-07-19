# Periodicity

Useful tools for periodicity analysis in time series data.

[![](https://github.com/dioph/periodicity/workflows/CI/badge.svg)](https://github.com/dioph/periodicity/actions?query=branch%3Amaster)
[![PyPI version](https://badge.fury.io/py/periodicity.svg)](https://badge.fury.io/py/periodicity)
[![Downloads](https://pepy.tech/badge/periodicity)](https://pepy.tech/project/periodicity)

__Documentation: https://periodicity.readthedocs.io__

Currently includes:
* Auto-Correlation Function (and other general timeseries utilities!)
* Spectral methods:
    * Lomb-Scargle periodogram
    * Bayesian Lomb-Scargle with linear Trend (soon™)
* Time-frequency methods (WIP):
    * Wavelet Transform
    * Hilbert-Huang Transform
* Phase-folding methods:
    * String Length
    * Phase Dispersion Minimization
    * Analysis of Variance (soon™)
* Decomposition methods:
    * Empirical Mode Decomposition
    * Local Mean Decomposition
    * Variational Mode Decomposition (soon™)
* Gaussian Processes:
    * `george` implementation
    * `celerite` implementation
    * `pymc3` implementation (soon™)

## Installation

The latest version is available to download via PyPI: __`pip install periodicity`__.

Alternatively, you can build the current development version from source by cloning this repo (__`git clone https://github.com/dioph/periodicity.git`__) and running __`pip install ./periodicity`__.

## Development

If you're interested in contributing to periodicity, install __pipenv__ and you can setup everything you need with __`pipenv install --dev`__.

To automatically test the project (and also check formatting, coverage, etc.), simply run __`tox`__ within the project's directory.