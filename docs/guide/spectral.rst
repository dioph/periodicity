================
Spectral methods
================

The basics: Fourier analysis
============================

.. math::
    
    \mathcal{F}\{x(t)\} = \int_{-\infty}^{\infty} x(t)\exp(-j2 \pi ft)dt

The Lomb-Scargle periodogram
============================

Wavelet analysis
================

Hilbert-Huang Transform
=======================

Empirical Mode Decomposition (EMD)
----------------------------------

Comparisons between the methods
===============================

A summary of the comparison between Fourier, Wavelet, and HHT analyses is given in the following table:

+----------------------+------------------+-----------------------+-----------------------+
|                      | **Fourier**      | **Continuous**        | **Hilbert**           |
|                      |                  | **Wavelet**           |                       |
+----------------------+------------------+-----------------------+-----------------------+
| **Basis**            | a priori         | a priori              | adaptive              |
+----------------------+------------------+-----------------------+-----------------------+
| **Frequency**        | - convolution    | - convolution         | - differentiation     |
|                      | - global         | - regional            | - local               |
|                      | - uncertainty    | - uncertainty         | - certainty           |
+----------------------+------------------+-----------------------+-----------------------+
| **Presentation**     | energy-frequency | energy-time-frequency | energy-time-frequency |
+----------------------+------------------+-----------------------+-----------------------+
| **Nonlinear**        | no               | no                    | yes                   |
+----------------------+------------------+-----------------------+-----------------------+
| **Non-stationary**   | no               | yes                   | yes                   |
+----------------------+------------------+-----------------------+-----------------------+
| **Feature**          | no               | yes                   | yes                   |
| **extraction**       |                  |                       |                       |
+----------------------+------------------+-----------------------+-----------------------+
| **Theoretical base** | theory complete  | theory complete       | empirical             |
|                      |                  |                       |                       |
+----------------------+------------------+-----------------------+-----------------------+

References and additional reading
=================================

    `Hilbert-Huang Transform on Scholarpedia <http://www.scholarpedia.org/article/Hilbert-Huang_transform#Comparisons_with_other_methods>`_.
   
    `Vanderplas 2018 <https://iopscience.iop.org/article/10.3847/1538-4365/aab766>`_.
    *Understanding the Lomb-Scargle Periodogram.*
    ApJS, 236, 16.


