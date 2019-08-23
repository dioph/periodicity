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
    s: int (optional)
        standard deviation of Gaussian filter used to smooth, measured in samples

    Returns
    -------
    periods: array-like
        trial periods
    ell: array-like
        string length for each period
    """
    # scale x to range from -0.25 to +0.25
    x = (x - np.max(x)) / (2 * (np.max(x) - np.min(x))) - 0.25
    df = dphi / (np.max(t) - np.min(t))
    periods = 1 / np.linspace(df, n_periods*df, n_periods)
    periods.sort()
    ell = []
    for period in periods:
        phi = ((t / period) % 1)
        sorted_args = np.argsort(phi)
        phi = phi[sorted_args]
        m = x[sorted_args]
        ll = np.hypot(np.roll(m, -1) - m, np.roll(phi, -1) - phi).sum()
        ell.append(ll)
    # TODO: consider flagging false periods for rejection
    ell = np.array(ell)
    if s > 0:
        kernel = gaussian(mu=0, sd=s)
        h = kernel(np.arange(-(3 * s - 1), 3 * s, 1.))
        ell = smooth(ell, kernel=h)
    return periods, ell


def pdm(t, x, nb=5, nc=2, pmin=.01, pmax=10, n_periods=1000, s=0):
    """Phase Dispersion Minimization
    (Stellingwerf 1978, ApJ, 224, 953)

    Parameters
    ----------
    t: array-like
        time array
    x: array-like
        signal array
    nb: int (optional default=5)
        number of phase bins
    nc: int (optional default=2)
        number of covers per bin
    pmin, pmax: floats (optional defaults=0.01 and 10)
        minimum/maximum trial period normalized by the baseline
    n_periods: int (optional default=1000)
        number of trial periods
    s: int (optional)
        standard deviation of Gaussian filter used to smooth, measured in samples

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
    m0 = nb * nc
    for period in periods:
        phi = ((t / period) % 1)
        sorted_args = np.argsort(phi)
        phi = phi[sorted_args]
        m = x[sorted_args]
        mj = []
        for k in range(m0):
            mask = phi >= k / m0
            mask &= phi < (k + nc) / m0
            mask |= phi < (k - (m0 - nc)) / m0
            mj.append(m[mask])
        sj = np.array([np.var(k, ddof=1) for k in mj])
        nj = np.array([k.size for k in mj])
        ss = np.sum((nj - 1) * sj)/(np.sum(nj) - m0)
        theta.append(ss/sigma)
    theta = np.array(theta)
    if s > 0:
        kernel = gaussian(mu=0, sd=s)
        h = kernel(np.arange(-(3 * s - 1), 3 * s, 1.))
        theta = smooth(theta, kernel=h)
    return periods, theta


def pdm2(t, x, pmin=None, pmax=None, n_periods=None, s=0, oversample=10, do_subharmonic=False):
    t = np.asarray(t)
    x = np.asarray(x)
    sigma = np.var(x, ddof=1)
    ne = t.size
    assert x.size == ne, "incompatible array shapes"
    theta_crit = 1. - 11. / ne ** 0.8
    dt = np.median(np.diff(t))
    t0 = t.max() - t.min()
    thetas = []
    if pmax is None:
        pmax = oversample * t0
    if pmin is None:
        pmin = 2 * dt
    if n_periods is None:
        n_periods = int((1 / pmin - 1 / pmax) * oversample * t0 + 1)
    periods = np.linspace(pmax, pmin, n_periods)
    for period in periods:
        phi = ((t - t[0]) / period) % 1
        masks = np.array([np.logical_and(phi < (b + 1) / 10, phi >= b / 10) for b in range(10)])
        sj = np.array([np.var(x[masks[j]], ddof=1) for j in range(10)])
        nj = masks.sum(axis=1)
        good = nj > 1
        ss = np.sum((nj[good] - 1) * sj[good]) / np.sum(nj[good] - 1)
        theta = ss / sigma
        if do_subharmonic and period <= pmax / 2 and theta < theta_crit:
            sub_index = int((n_periods - 1) * (1 - (2 * period - pmin) / (pmax - pmin)) + 0.5)
            theta = (theta + thetas[sub_index]) / 2
        thetas.append(theta)
    thetas = np.array(thetas)[::-1]
    periods = periods[::-1]
    if s > 0:
        kernel = gaussian(mu=0, sd=s)
        h = kernel(np.arange(-(3 * s - 1), 3 * s, 1.))
        thetas = smooth(thetas, kernel=h)
    return periods, thetas

# TODO: Analysis of Variance (Schwarzenberg-Czerny 1989)

# TODO: Gregory-Loredo method (Gregory & Loredo 1992)

# TODO: conditional entropy method (Graham et al. 2013)
