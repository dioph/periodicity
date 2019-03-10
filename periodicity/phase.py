import numpy as np
from .acf import gaussian, smooth


def stringlength(t, x, dphi=0.1, n_periods=1000, s=0):
    """String Length
    (Dworetsky 1983, MNRAS, 203, 917)

    Parameters
    ----------
    t: array-like
        time array
    x: array-like
        signal array
    dphi: float (optional default=0.1)
        factor to multiply (1 / baseline) in order to get frequency separation
    n_periods: int (optional default=1000)
        number of trial periods
    Returns
    -------
    periods: array-like
        trial periods
    L: array-like
        string length for each period
    s: int (optional)
        standard deviation of Gaussian filter used to smooth ACF, measured in samples
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


def pdm(t, x, Nb=5, Nc=2, pmin=.01, pmax=10, n_periods=1000, s=0):
    """Phase Dispersion Minimization
    (Stellingwerf 1978, ApJ, 224, 953)

    Parameters
    ----------
    t: array-like
        time array
    x: array-like
        signal array
    Nb: int (optional default=5)
        number of phase bins
    Nc: int (optional default=2)
        number of covers per bin
    pmin, pmax: floats (optional defaults=0.01 and 10)
        minimum/maximum trial period normalized by the baseline
    n_periods: int (optional default=1000)
        number of trial periods
    s: int (optional)
        standard deviation of Gaussian filter used to smooth ACF, measured in samples

    Returns
    -------
    periods: array-like
        trial periods
    theta: array-like
        phase dispersion statistic as in Eq. 3 of the paper
    """
    t = np.asarray(t)
    x = np.asarray(x)
    sigma = np.var(x, ddof=1)
    t0 = t.max() - t.min()
    theta = []
    periods = np.linspace(pmin*t0, pmax*t0, n_periods)
    M = Nb * Nc
    for period in periods:
        phi = ((t / period) % 1)
        sorted_args = np.argsort(phi)
        phi = phi[sorted_args]
        m = x[sorted_args]
        mj = []
        for k in range(M):
            mask = phi >= k / M
            mask &= phi < (k + Nc) / M
            mask |= phi < (k - (M - Nc)) / M
            mj.append(m[mask])
        sj = np.array([np.var(k, ddof=1) for k in mj])
        nj = np.array([k.size for k in mj])
        s = np.sum((nj - 1) * sj)/(np.sum(nj) - M)
        theta.append(s/sigma)
    theta = np.array(theta)
    if s > 0:
        kernel = gaussian(mu=0, sd=s)
        h = kernel(np.arange(-(3 * s - 1), 3 * s, 1.))
        theta = smooth(theta, kernel=h)
    return periods, theta

# TODO: Analysis of Variance (Schwarzenberg-Czerny 1989)

# TODO: Gregory-Loredo method (Gregory & Loredo 1992)

# TODO: conditional entropy method (Graham et al. 2013)
