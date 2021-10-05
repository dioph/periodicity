===========
periodicity
===========

``periodicity`` is a toolkit of period determination methods.
The latest development version can be found `here <https://github.com/dioph/periodicity>`_.

Installation
============
``periodicity`` requires
``xarray`` (for computation on labeled data),
``celerite2`` and ``george`` (for Gaussian Process models),
``emcee`` and  ``pymc3_ext`` (for MCMC sampling Gaussian Processes),
and ``PyWavelets`` (for Wavelet analysis).

Installing the most recent stable version of the package is as easy as::

    pip install periodicity

Changelog
=========

1.0 (2021-00-00)
----------------

Initial beta release

.. toctree::
    :maxdepth: 2
    :caption: User Guide

    guide/core
    guide/spectral
    guide/phase
    guide/gp
    guide/timefrequency
    guide/decomposition

.. toctree::
    :maxdepth: 1
    :caption: Tutorials

    tutorials/fastgp
    tutorials/wavelet
    tutorials/hht


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
