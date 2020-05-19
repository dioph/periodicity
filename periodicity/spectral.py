import numpy as np
from astropy.timeseries import LombScargle
from scipy.signal import hilbert
import pywt

from .utils import find_extrema, find_zero_crossings, get_envelope, fill_gaps

__all__ = ['lombscargle', 'window', 'wavelet', 'emd', 'hht']


def lombscargle(t, x, dx=None, f0=None, fmax=None, n=5,
                fap_method=None, fap_level=None, psd=False):
    """Computes the generalized Lomb-Scargle periodogram of a discrete signal.

    Parameters
    ----------
    t: array-like
        Time array.
    x: array-like
        Signal array.
    dx: array-like, optional
        Measurement uncertainties for each sample.
    f0: float, optional
        Minimum frequency.
        If not given, it will be determined from the time baseline.
    fmax: float, optional
        Maximum frequency.
        If not given, defaults to the pseudo-Nyquist limit.
    n: float, optional
        Samples per peak (default is 5).
    fap_method: {None, 'baluev', 'bootstrap'}
        The approximation method to use for the highest peak FAP and
        false alarm levels. The default is None, in which case the FAP
        is not calculated.
    fap_level: array-like, optional
        List of false alarm probabilities for which you want to calculate
        approximate levels. Can also be passed as a single scalar value.
    psd: bool, optional
        Whether to leave periodogram non-normalized (Fourier Spectral Density).

    Returns
    -------
    ls: astropy.timeseries.LombScargle object
        The full object for the given data.
    f: ndarray
        Frequency array.
    a: ndarray
        Power array.
    fap: float
        If `fap_method` is given, the False Alarm Probability of highest peak.
    fal: float
        If `fap_level` is given, the power level for the given probabilities.
    """
    if psd:
        ls = LombScargle(t, x, dy=dx, normalization='psd')
    else:
        ls = LombScargle(t, x, dy=dx)
    if fmax is None:
        ts = float(np.median(np.diff(t)))
        fs = 1 / ts
        fmax = fs / 2
    f, a = ls.autopower(samples_per_peak=n, minimum_frequency=f0, maximum_frequency=fmax)
    if fap_method is not None:
        if fap_method not in ['baluev', 'bootstrap']:
            raise ValueError(f"Unknown FAP method {fap_method}.")
        fap = ls.false_alarm_probability(a.max(), method=fap_method, minimum_frequency=f0,
                                         maximum_frequency=fmax, samples_per_peak=n)
        if fap_level is not None:
            fal = ls.false_alarm_level(fap_level, method=fap_method, minimum_frequency=f0,
                                       maximum_frequency=fmax, samples_per_peak=n)
            return ls, f, a, fap, fal
        return ls, f, a, fap
    return ls, f, a


def window(t, n=5):
    """Computes the periodogram of the window function.

    Parameters
    ----------
    t: array-like
        Timestamps of the sampling comb window.
    n: float, optional
        Samples per peak (default is 5).
    Returns
    -------
    f: ndarray
        Frequency array.
    a: ndarray
        Power array.
    """
    ls = LombScargle(t, 1, fit_mean=False, center_data=False)
    f, a = ls.autopower(minimum_frequency=0, samples_per_peak=n)
    return f, a


def wavelet(t, x, periods):
    """Wavelet Power Spectrum using Morlet wavelets.

    Parameters
    ----------
    t: array-like
        Time array.
    x: array-like
        Signal array.
    periods: array-like
        Periods to consider, in the same units as `t`.

    Returns
    -------
    power: ndarray[len(t), len(periods)]
        Wavelet Power Spectrum.
    coi: tuple of ndarray
        Time and scale samples for plotting the Cone of Influence boundaries.
    mask_coi: ndarray[len(t), len(periods)]
        Boolean mask with the same shape as `power`; it is True inside the COI.
    """
    dt = float(np.median(np.diff(t)))
    scales = pywt.scale2frequency('morl', 1) * np.asarray(periods) / dt
    conv_complex = len(scales) * len(x)
    n = len(scales) + len(x) - 1
    fft_complex = n * np.log2(n)
    if fft_complex < conv_complex:
        method = 'fft'
    else:
        method = 'conv'
    coefs, freqs = pywt.cwt(x, scales, 'morl', dt, method=method)
    power = np.square(np.abs(coefs))
    # Cone of Influence (COI)
    tmax = np.max(t)
    tmin = np.min(t)
    pmax = np.max(periods)
    pmin = np.min(periods)
    T, P = np.meshgrid(t, periods)
    S = 2 ** .5 * P
    mask_coi = (S < np.minimum(T - tmin, tmax - T))
    p_samples = np.logspace(np.log10(pmin), np.log10(pmax), 100)
    p_samples = p_samples[2 ** .5 * p_samples < (tmax - tmin) / 2]
    t1 = tmin + 2 ** .5 * p_samples
    t2 = tmax - 2 ** .5 * p_samples
    t_samples = np.hstack((t1, t2))
    p_samples = np.hstack((p_samples, p_samples))
    sorted_ids = t_samples.argsort()
    sorted_t_samples = t_samples[sorted_ids]
    sorted_p_samples = p_samples[sorted_ids]
    coi = (sorted_t_samples, sorted_p_samples)
    return power, coi, mask_coi


# TODO: check out Supersmoother (Reimann 1994)


def emd(x, t=None, max_iter=2000, theta_1=0.05, theta_2=0.50,
        alpha=0.05, delta=0., n_rep=2):
    """Empirical Mode Decomposition

    Parameters
    ----------
    x: array-like
        Signal.
    t: array-like, optional
        Signal timestamps. If not given, integer indices will be used.
    max_iter: int, optional
        Maximum number of sifting iterations (the default is 2000).
    theta_1: float, optional
        Lower threshold for the evaluation function (the default is 0.05).
    theta_2: float, optional
        Upper threshold for the evaluation function (usually ``10 * theta_1``).
    alpha: float, optional
        Fraction of total duration where the evaluation function is allowed to
        be ``theta_1 < sigma < theta_2`` (the default is 0.05).
    delta: float, optional
        Peak prominence to be used when searching for local extrema.
    n_rep: int, optional
        Number of extrema to repeat on either side of the signal while
        interpolating envelopes (the default is 2).

    Returns
    -------
    imfs: list of ndarray
        List of intrinsic mode functions obtained through the decomposition.
        The last element corresponds to the monotonic residue.

    Notes
    -----
    This algorithm is described in [#]_.

    References
    ----------
    .. [#] G. Rilling, P. Flandrin, P. Gonçalves, "On Empirical Mode
       Decomposition and its Algorithms," IEEE-EURASIP Workshop on Nonlinear
       Signal and Image Processing, June 2003.
    """
    imfs = []
    n_ext = len(x)
    residue = x.copy()
    while n_ext > 2:
        mode = residue.copy()
        stop = False
        it = 0
        while not stop and it < max_iter:
            maxima, minima = find_extrema(mode, delta=delta)
            zeroes = find_zero_crossings(mode, delta=delta)
            n_ext = len(maxima) + len(minima)
            n_zero = len(zeroes)
            it += 1
            try:
                upper, lower = get_envelope(mode, t, delta=delta, n_rep=n_rep)
                mu = (upper + lower) / 2
                amp = (upper - lower) / 2
                sigma = np.abs(mu / amp)
                # sigma < theta_1 for some prescribed fraction (1-alpha) of the total duration
                stop = (np.mean(sigma > theta_1) < alpha)
                # sigma < theta_2 for the remaining fraction
                stop = stop and np.all(sigma < theta_2)
                # the number of extrema and the number of zero-crossings must differ at most by 1
                stop = stop and (np.abs(n_zero - n_ext) <= 1)
                # if there are not enough extrema, the current mode is a monotonic residue
                stop = stop or (n_ext < 3)
            except TypeError:
                stop = True
                mu = np.zeros_like(mode)
            mode -= mu
        imfs.append(mode)
        residue -= mode
    imfs.append(residue)
    return imfs


def hht(t, x, ts=None, **kwargs):
    """Hilbert-Huang Transform

    Parameters
    ----------
    t: array-like
        Time array.
    x: array-like
        Signal array.
    ts: float, optional
        Sampling period. If omitted, it will be estimated from `t`.
    **kwargs: dict
        Keyword arguments to be used by `emd`.

    Returns
    -------
    f: list of ndarray
        Instantaneous frequency array for each intrinsic mode.
    a: list of ndarray
        Signal amplitude envelope for each intrinsic mode.

    See Also
    --------
    emd: Empirical Mode Decomposition
    """
    f, a = [], []
    t, x = fill_gaps(t, x, ts)
    if ts is None:
        ts = float(np.median(np.diff(t)))
    imfs = emd(x, t, **kwargs)
    for imf in imfs:
        if np.any(imf):
            xa = hilbert(imf)
            amp = np.abs(xa)
            phase = np.unwrap(np.angle(xa))
            freq = np.diff(phase) / (2 * np.pi * ts)
            a.append(amp)
            f.append(freq)
    return f, a
