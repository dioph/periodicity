import numpy as np
from astropy.stats import LombScargle


def lombscargle(t, x, f=None, n=5, fap_method=None, fap_level=None):
    """
    Computes the generalized Lomb-Scargle periodogram of a discrete signal x(t)

    Parameters
    ----------
    t: array-like
        time array
    x: array-like
        signal array
    f: array-like (optional)
        frequency array
    n: float (optional default=5)
        samples per peak
    fap_method: string (optional)
        the approximation method to use for highest peak FAP and false alarm levels
    fap_level: array-like (optional)
        false alarm probabilities to approximate heights

    Returns
    -------
    f: array-like
        frequency array
    a: array-like
        power array
    """
    ls = LombScargle(t, x)
    if f is None:
        T = np.median(np.diff(t))
        fs = 1/T
        T0 = t.max() - t.min()
        f0 = 1/T0
        f = np.linspace(f0, fs, n*fs*T0)
    a = ls.power(f)
    if fap_method is not None:
        assert fap_method in ['baluev', 'bootstrap'], "Unknown FAP method {}".format(fap_method)
        fap = ls.false_alarm_probability(a.max(), method=method, minimum_frequency=f0, maximum_frequency=fs, samples_per_peak=n)

    return f, a, fap


def window(t):
    """
    Computes the periodogram of the window function

    Parameters
    ----------

    Returns
    -------

    """
    pass