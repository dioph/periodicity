import os

import numpy as np


def SpottedStar():
    """
    Returns
    -------
    data: ndarray
        KIC 9655172 light curve, used for testing and demonstration.

    Examples
    --------
    >>> import periodicity.data
    >>> t, y, dy = periodicity.data.SpottedStar()
    >>> y.shape == (2148,)
    True
    """
    filename = os.path.join(os.path.dirname(__file__), "spotted_star.npy")
    data = np.load(filename)
    return data


def SunSpots():
    """
    Returns
    -------
    data: ndarray
        Daily total sunspot number time series (WDC-SILSO), non-uniformly
        sampled (~daily) spanning from Jan 1818 to Jun 2021, used for testing
        and demonstration. Bad measurements are marked with -1.

    Examples
    --------
    >>> import periodicity.data
    >>> t, y = periodicity.data.TSI()
    >>> y.shape == (74326,)
    True
    """
    filename = os.path.join(os.path.dirname(__file__), "sunspots.npy")
    data = np.load(filename)
    return data


def TSI():
    """
    Returns
    -------
    data: ndarray
        PMOD Composite Total Solar Irradiance time series, non-uniformly
        sampled (~daily) spanning from Nov 1978 to Mar 2012, used for testing
        and demonstration. Bad measurements are marked with -99.

    Examples
    --------
    >>> import periodicity.data
    >>> t, y = periodicity.data.TSI()
    >>> y.shape == (12187,)
    True
    """
    filename = os.path.join(os.path.dirname(__file__), "tsi.npy")
    data = np.load(filename)
    return data


def BPSK(t_bit, n_bits, f_c, n0_db=-np.inf):
    """
    Parameters
    ----------
    t_bit: int
        Number of samples per bit (sampling rate / bit rate).
    n_bits: int
        Desired number of bits in the generated signal.
    f_c: scalar
        Carrier frequency (normalized units).
    n0_db: scalar, optional
        Noise spectral density (average noise power). Defaults to -inf (zero noise).

    Returns
    -------
    data: ndarray
        Noisy BPSK signal, used for testing and demonstration.

    Examples
    --------
    >>> import periodicity.data
    >>> y = periodicity.data.BPSK(t_bit=10, n_bits=4000, f_c=0.05)
    >>> y.shape == (40_000,)
    True
    """
    t0 = t_bit * n_bits
    sym_seq = np.zeros(t0)
    sym_seq[::t_bit] = np.random.choice([-1, 1], n_bits)
    pulse = np.ones(t_bit)
    baseband = np.convolve(pulse, sym_seq)[:t0]
    signal = baseband * np.exp(1j * 2 * np.pi * f_c * np.arange(t0))
    noise = np.random.randn(t0) + 1j * np.random.randn(t0)
    n0 = 10 ** (n0_db / 10)
    noise *= np.sqrt(n0 / np.var(noise))
    data = signal + noise
    return data


def SustainedPlusGappedPureTones():
    """
    Returns
    -------
    data: ndarray
        Sustained pure tone plus a gapped one with a higher frequency, used for
        testing and demonstration.

    Examples
    --------
    >>> import periodicity.data
    >>> y = periodicity.data.SustainedPlusGappedPureTones()
    >>> y.shape == (1000,)
    True
    """
    t = np.arange(1000)
    data = np.sin(2 * np.pi * 0.065 * t)
    data[500:750] += np.sin(2 * np.pi * 0.255 * (t[500:750] - 500))
    return data


def GaussianAtomsPlusFMSinusoid():
    """
    Returns
    -------
    data: ndarray
        Two Gaussian atoms at different timeshifts, with different amplitudes
        and frequencies, plus an FM sinusoid, used for testing and demonstration.

    Examples
    --------
    >>> import periodicity.data
    >>> y = periodicity.data.GaussianAtomsPlusFMSinusoid()
    >>> y.shape == (2000,)
    True
    """
    n = np.arange(1, 2001)
    fmax = 3 / 32
    fmin = 9 / 128
    phi = -np.arccos((3 * fmin - fmax) / (fmax + fmin))
    x1 = 3 * np.exp(-(((n - 500) / 100) ** 2)) * np.cos(2 * np.pi * 5 / 16 * (n - 1000))
    x2 = np.cos(
        2 * np.pi * (fmax + fmin) / 2 * (n - 1000)
        + (fmax - fmin) / 2 * 1000 * (np.sin(2 * np.pi * n / 1000) + phi - np.sin(phi))
    )
    x3 = np.exp(-(((n - 1000) / 200) ** 2)) * np.cos(2 * np.pi * 7 / 256 * (n - 1000))
    return x1 + x2 + x3


def DuffingWave():
    """
    Returns
    -------
    data: ndarray
        Damped Duffing wave with chirp frequency, used for testing and demonstration.

    Examples
    --------
    >>> import periodicity.data
    >>> y = periodicity.data.DuffingWave()
    >>> y.shape == (1024,)
    True
    """
    t = np.arange(1024)
    data = np.exp(-t / 256) * np.cos(
        (np.pi / 64) * (t ** 2 / 512 + 32)
        + 0.3 * np.sin((np.pi / 32) * (t ** 2 / 512 + 32))
    )
    return data
