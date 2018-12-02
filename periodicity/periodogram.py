import numpy as np
from astropy.stats import LombScargle


def lombscargle(t, x, f0=0, fmax=None, n=5, fap_method=None, fap_level=None):
    """Computes the generalized Lomb-Scargle periodogram of a discrete signal x(t)

    Parameters
    ----------
    t: array-like
        time array
    x: array-like
        signal array
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

    Returns
    -------
    f: array-like
        frequency array
    a: array-like
        power array
    fap: float
        false alarm probability of highest peak
    fal: float
        false alarm level for a given FAP
    """
    ls = LombScargle(t, x)
    if fmax is None:
        T = float(np.median(np.diff(t)))
        fs = 1 / T
        fmax = fs / 2
    f, a = ls.autopower(samples_per_peak=n, minimum_frequency=f0, maximum_frequency=fmax)
    if fap_method is not None:
        assert fap_method in ['baluev', 'bootstrap'], "Unknown FAP method {}".format(fap_method)
        fap = ls.false_alarm_probability(a.max(), method=fap_method, minimum_frequency=f0,
                                         maximum_frequency=fmax, samples_per_peak=n)
        if fap_level is not None:
            fal = ls.false_alarm_level(fap_level, method=fap_method, minimum_frequency=f0,
                                       maximum_frequency=fmax, samples_per_peak=n)
            return f, a, fap, fal
        return f, a, fap
    return f, a


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


def wavelet(t, x):
    raise NotImplementedError
    pass

# TODO: check out Supersmoother (Reimann 1994)
