import numpy as np
from .acf import gaussian, smooth


def stringlength(t, x, dphi=0.1, n_periods=1000, s=0):
    """String Length

    Parameters
    ----------
    t: array-like
        time array
    x: array-like
        signal array
    dphi: float (optional default=0.1)
        factor to multiply (1 / baseline) in order to get frequency separation
    n_periods: int (optional default=1000)
        number of period samples to test
    Returns
    -------
    periods: array-like
        periods tested
    L: array-like
        string length for each period
    """
    # scale x to range from -0.25 to +0.25
    x = (x - np.max(x)) / (2 * (np.max(x) - np.min(x))) - 0.25
    df = dphi / (np.max(t) - np.min(t))
    periods = 1 / np.linspace(df, n_periods*df, n_periods)
    periods.sort()
    L = []
    for period in periods:
        phi = ((t / period) % 1)
        sorted_args = np.argsort(phi)
        phi = phi[sorted_args]
        m = x[sorted_args]
        ll = np.hypot(np.roll(m, -1) - m, np.roll(phi, -1) - phi).sum()
        L.append(ll)
    # TODO: consider flagging false periods for rejection
    L = np.array(L)
    if s > 0:
        kernel = gaussian(mu=0, sd=s)
        h = kernel(np.arange(-(3 * s - 1), 3 * s, 1.))
        L = smooth(L, kernel=h)
    return periods, L

# TODO: Analysis of Variance (Schwarzenberg-Czerny 1989)

# TODO: Phase Dispersion Minimization (Stellingwerf 1978)

# TODO: Gregory-Loredo method (Gregory & Loredo 1992)

# TODO: conditional entropy method (Graham et al. 2013)
