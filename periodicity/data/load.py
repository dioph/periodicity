import os

import numpy as np


def spotted_lc():
    """
    Returns
    -------
    data: ndarray
        KIC 9655172 light curve, used for testing and demonstration.

    Examples
    --------
    >>> import periodicity.data
    >>> t, y, dy = periodicity.data.spotted_lc()
    >>> y.shape == (2148,)
    True
    """
    filename = os.path.join(os.path.dirname(__file__), "spotted_lc.npy")
    data = np.load(filename)
    return data


def bpsk(t_bit, n_bits, f_c, n0_db=-np.inf):
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
    >>> t, y = periodicity.data.bpsk(t_bit=10, n_bits=4000, f_c=0.05)
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
