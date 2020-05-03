import numpy as np
from astropy.timeseries import LombScargle
from scipy.signal import hilbert
from wavelets import WaveletAnalysis

from periodicity.utils import find_extrema, find_zero_crossings, get_envelope, fill_gaps


def lombscargle(t, x, dx=None, f0=0, fmax=None, n=5, fap_method=None, fap_level=None, psd=False):
    """Computes the generalized Lomb-Scargle periodogram of a discrete signal x(t)

    Parameters
    ----------
    t: array-like
        time array
    x: array-like
        signal array
    dx: array-like (optional)
        measurement uncertainties for each sample
    f0: float (optional default=0)
        minimum frequency
    fmax: float (optional)
        maximum frequency
        If None is given, defaults to the pseudo-Nyquist limit
    n: float (optional default=5)
        samples per peak
    fap_method: string {None, 'baluev', 'bootstrap'}
        the approximation method to use for highest peak FAP and false alarm levels
        None by default
    fap_level: array-like (optional)
        false alarm probabilities to approximate heights
    psd: bool (optional)
        whether to leave periodogram unnormalized (Fourier Spectral Density)

    Returns
    -------
    ls: astropy.timeseries.LombScargle object
        the full object for the given dataset
    f: array-like
        frequency array
    a: array-like
        power array
    fap: float
        false alarm probability of highest peak
    fal: float
        false alarm level for a given FAP
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
        assert fap_method in ['baluev', 'bootstrap'], "Unknown FAP method {}".format(fap_method)
        fap = ls.false_alarm_probability(a.max(), method=fap_method, minimum_frequency=f0,
                                         maximum_frequency=fmax, samples_per_peak=n)
        if fap_level is not None:
            fal = ls.false_alarm_level(fap_level, method=fap_method, minimum_frequency=f0,
                                       maximum_frequency=fmax, samples_per_peak=n)
            return ls, f, a, fap, fal
        return ls, f, a, fap
    return ls, f, a


def window(t, n=5):
    """Computes the periodogram of the window function

    Parameters
    ----------
    t: array-like
        times of sampling comb window
    n: float (optional default=5)
        samples per peak
    Returns
    -------
    f: array-like
        frequency array
    a : array-like
        power array
    """
    ls = LombScargle(t, 1, fit_mean=False, center_data=False)
    f, a = ls.autopower(minimum_frequency=0, samples_per_peak=n)
    return f, a


def wavelet(t, x, n=1):
    """Wavelet Power Spectrum using Morlet wavelets

    Parameters
    ----------
    t: array-like
        time array
    x: array-like
        signal array
    n: float (optional default=1)
        multiplier of number of scales considered

    Returns
    -------
    wa: wavelets.WaveletAnalysis object
        the full object for the given dataset
    periods: array-like
        trial periods array
    gwps: array-like
        Global Wavelet Power Spectrum (WPS projected on period axis)
    wps: array-like
        Wavelet Power Spectrum
    """
    dt = float(np.median(np.diff(t)))
    wa = WaveletAnalysis(x, t, dt=dt, mask_coi=True, unbias=True, dj=0.125 / n)
    periods = wa.fourier_periods
    wps = wa.wavelet_power
    gwps = wa.global_wavelet_spectrum
    return wa, periods, gwps, wps


# TODO: check out Supersmoother (Reimann 1994)


def emd(x, t=None, maxiter=2000, theta_1=0.05, theta_2=0.50, alpha=0.05, delta=0., nbsym=2):
    """Empirical Mode Decomposition

    G. Rilling, P. Flandrin, P. Gonçalves, June 2003
    'On Empirical Mode Decomposition and its Algorithms'
    IEEE-EURASIP Workshop on Nonlinear Signal and Image Processing

    Parameters
    ----------
    x: array-like
        signal
    t: array-like, optional
        signal timestamps
    maxiter: int, optional (default=2000)
        maximum number of sifting iterations
    theta_1: float, optional (default=0.05)
        lower threshold for the evaluation function
    theta_2: float, optional (default=0.50)
        upper threshold for the evaluation function (typically 10 * theta_1)
    alpha: float, optional (default=0.05)
        fraction of total duration where the evaluation function is allowed to be theta_1 < sigma < theta_2
    delta: float, optional
        peak prominence to use when searching for local extrema
    nbsym: int, optional (default=2)
        number of extrema to repeat on either side of the signal while interpolating envelopes

    Returns
    -------
    imfs: list of arrays
        list of intrinsic mode functions obtained through the decomposition.
        the last element corresponds to the residue.
    """
    imfs = []
    ner = len(x)
    residue = x.copy()
    while ner > 2:
        mode = residue.copy()
        stop = False
        k = 0
        while not stop and k < maxiter:
            maxima, minima = find_extrema(mode, delta=delta)
            zeroes = find_zero_crossings(mode, delta=delta)
            ner = len(maxima) + len(minima)
            nzm = len(zeroes)
            k += 1
            try:
                upper, lower = get_envelope(mode, t, delta=delta, nbsym=nbsym)
                mu = (upper + lower) / 2
                amp = (upper - lower) / 2
                sigma = np.abs(mu / amp)
                # sigma < theta_1 for some prescribed fraction (1-alpha) of the total duration
                stop = (np.mean(sigma > theta_1) < alpha)
                # sigma < theta_2 for the remaining fraction
                stop = stop and np.all(sigma < theta_2)
                # the number of extrema and the number of zero-crossings must differ at most by 1
                stop = stop and (np.abs(nzm - ner) <= 1)
                # if there are not enough extrema, the current mode is a monotonic residue
                stop = stop or (ner < 3)
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
    t:
    x:
    ts:
    kwargs:

    Returns
    -------
    f:
    a:
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