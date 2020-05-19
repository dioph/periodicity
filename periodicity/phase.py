import numpy as np
from scipy.ndimage import gaussian_filter1d

__all__ = ['stringlength', 'pdm', 'pdm2']


def stringlength(t, x, dphi=0.1, n_periods=1000, s=0):
    """String Length [#]_.

    Parameters
    ----------
    t: array-like
        Time array.
    x: array-like
        Signal array.
    dphi: float, optional
        Factor to multiply (1 / baseline) in order to get the frequency
        separation (the default is 0.1).
    n_periods: int, optional
        Number of trial periods (the default is 1000).
    s: int, optional
        Standard deviation of Gaussian filter used to smooth,
        measured in samples.

    Returns
    -------
    periods: ndarray[n_periods]
        Trial periods.
    ell: ndarray[n_periods]
        String length for each period.

    References
    ----------
    .. [#] M. M. Dworetsky, "A period-finding method for sparse randomly spaced
       observations or 'How long is a piece of string ?'," MNRAS, June 1983.
    """
    # scale x to range from -0.25 to +0.25
    x = (x - np.max(x)) / (2 * (np.max(x) - np.min(x))) - 0.25
    df = dphi / (np.max(t) - np.min(t))
    periods = 1 / np.linspace(df, n_periods * df, n_periods)
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
        ell = gaussian_filter1d(ell, sigma=s, truncate=3.0)
    return periods, ell


def pdm(t, x, nb=5, nc=2, pmin=.01, pmax=10, n_periods=1000, s=0):
    """Phase Dispersion Minimization [#]_.

    Parameters
    ----------
    t: array-like
        Time array.
    x: array-like
        Signal array.
    nb: int, optional
        Number of phase bins (the default is 5).
    nc: int, optional
        Number of covers per bin (the default is 2).
    pmin, pmax: float, optional
        Minimum/maximum trial period normalized by the baseline.
    n_periods: int, optional
        Number of trial periods (the default is 1000).
    s: int, optional
        Standard deviation of Gaussian filter used to smooth,
        measured in samples.

    Returns
    -------
    periods: ndarray[n_periods]
        Trial periods.
    theta: ndarray[n_periods]
        Phase dispersion statistic as in Eq. 3 of the paper.

    References
    ----------
    .. [#] R. F. Stellingwerf, "Period determination using phase dispersion
       minimization," ApJ, September 1978.
    """
    t = np.asarray(t)
    x = np.asarray(x)
    sigma = np.var(x, ddof=1)
    t0 = t.max() - t.min()
    theta = []
    periods = np.linspace(pmin * t0, pmax * t0, n_periods)
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
        ss = np.sum((nj - 1) * sj) / (np.sum(nj) - m0)
        theta.append(ss / sigma)
    theta = np.array(theta)
    if s > 0:
        theta = gaussian_filter1d(theta, sigma=s, truncate=3.0)
    return periods, theta


def pdm2(t, x, pmin=None, pmax=None, n_periods=None, s=0,
         oversample=10, do_subharmonic=False):
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
        masks = np.array([np.logical_and(phi < (b + 1) / 10, phi >= b / 10)
                          for b in range(10)])
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
        thetas = gaussian_filter1d(thetas, sigma=s, truncate=3.0)
    return periods, thetas

# TODO: Analysis of Variance (Schwarzenberg-Czerny 1989)

# TODO: Gregory-Loredo method (Gregory & Loredo 1992)

# TODO: conditional entropy method (Graham et al. 2013)
