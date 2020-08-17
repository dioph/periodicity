===========
periodicity
===========

``periodicity`` is a toolkit of period determination methods.
The latest development version can be found `here <https://github.com/dioph/periodicity>`_.

Installation
============
``periodicity`` requires
``numpy`` (for array arithmetic),
``scipy`` (for signal processing and optimization),
``astropy`` (for a great Lomb-Scargle periodogram implementation), and
``emcee`` (for MCMC sampling Gaussian Processes)

Installing the most recent stable version of the package is as easy as::

    pip install periodicity

Changelog
=========

1.0 (2020-00-00)
----------------

Initial beta release

.. toctree::
    :maxdepth: 2
    :caption: User Guide

    spectral
    phase
    gp
    api

.. toctree::
    :maxdepth: 1
    :caption: Tutorials

    tutorials/wavelet
    tutorials/fastgp



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
