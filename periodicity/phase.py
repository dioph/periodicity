import numpy as np
from scipy.ndimage import gaussian_filter1d
from multiprocessing import Pool, cpu_count

MAX_CORES = cpu_count()

__all__ = ['stringlength', 'pdm', 'pdm2']


def _stringlength(task):
    """Calculates the string length for a single trial period."""
    t, x, period = task
    phi = ((t / period) % 1)
    sorted_args = np.argsort(phi)
    phi = phi[sorted_args]
    m = x[sorted_args]
    ll = np.hypot(np.roll(m, -1) - m, np.roll(phi, -1) - phi).sum()
    return ll


def stringlength(t, x, dphi=0.1, n_periods=1000, s=0, cores=None):
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
    cores: int, optional
        The number of parallel cores to use.
        By default it will try to use all available CPU cores.

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
    m = (x - np.max(x)) / (2 * (np.max(x) - np.min(x))) - 0.25
    df = dphi / (np.max(t) - np.min(t))
    periods = 1 / np.linspace(n_periods * df, df, n_periods)
    if cores is None or cores > MAX_CORES:
        cores = MAX_CORES
    with Pool(cores) as pool:
        tasks = [(t, m, period) for period in periods]
        ell = pool.map(_stringlength, tasks)
    ell = np.array(ell)
    if s > 0:
        ell = gaussian_filter1d(ell, sigma=s, truncate=3.0)
    return periods, ell


def _pdm(task):
    """Calculates the PDM theta statistic for a single trial period."""
    t, x, nb, nc, sigma, period = task
    m0 = nb * nc
    phi = ((t / period) % 1)
    sorted_args = np.argsort(phi)
    phi = phi[sorted_args]
    m = x[sorted_args]
    mj = []
    m0_good = 0
    for k in range(m0):
        mask = (phi >= k / m0)
        mask &= (phi < (k + nc) / m0)
        mask |= (phi < (k - (m0 - nc)) / m0)
        mk = m[mask]
        if mk.size > 1:
            mj.append(mk)
            m0_good += 1
    sj = np.array([np.var(k, ddof=1) for k in mj])
    nj = np.array([k.size for k in mj])
    ss = np.sum((nj - 1) * sj) / (np.sum(nj) - m0_good)
    theta = ss / sigma
    return theta


def pdm(t, x, nb=5, nc=2, p_min=.01, p_max=10, n_periods=1000, s=0, cores=None):
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
    p_min, p_max: float, optional
        Minimum/maximum trial period normalized by the baseline.
    n_periods: int, optional
        Number of trial periods (the default is 1000).
    s: int, optional
        Standard deviation of Gaussian filter used to smooth,
        measured in samples.
    cores: int, optional
        The number of parallel cores to use.
        By default it will try to use all available CPU cores.

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
    t0 = np.max(t) - np.min(t)
    periods = np.linspace(p_min * t0, p_max * t0, n_periods)
    if cores is None or cores > MAX_CORES:
        cores = MAX_CORES
    with Pool(cores) as pool:
        tasks = [(t, x, nb, nc, sigma, period) for period in periods]
        theta = pool.map(_pdm, tasks)
    theta = np.array(theta)
    if s > 0:
        theta = gaussian_filter1d(theta, sigma=s, truncate=3.0)
    return periods, theta


def _pdm2(task):
    t, m, sigma, period = task
    phi = ((t - t[0]) / period) % 1
    bins = np.arange(0.0, 1.0, 0.1)
    ids = np.digitize(phi, bins)
    unique_ids = np.unique(ids)
    mj = []
    good = 0
    for i in unique_ids:
        mask = (ids == i)
        mi = m[mask]
        if mi.size > 1:
            mj.append(mi)
            good += 1
    sj = np.array([np.var(k, ddof=1) for k in mj])
    nj = np.array([k.size for k in mj])
    ss = np.sum((nj - 1) * sj) / (np.sum(nj) - good)
    theta = ss / sigma
    return theta


def pdm2(t, x, p_min=None, p_max=None, n_periods=None, s=0,
         oversample=10, do_subharmonic=False, cores=None):
    """PDM2 algorithm described in [#]_.

    Parameters
    ----------
    t: array-like
        Time array.
    x: array-like
        Signal array.
    p_min, p_max: float, optional
        Minimum/maximum trial period normalized by the baseline.
    n_periods: int, optional
        Number of trial periods (the default is 1000).
    s: int, optional
        Standard deviation of Gaussian filter used to smooth,
        measured in samples.
    oversample: scalar, optional
        If `p_max` is omitted, the time baseline multiplied by `oversample` will
        be used instead.
    do_subharmonic: bool, optional
        Whether to perform subharmonic averaging. This option looks for a
        significant minimum in :math:`\theta` at both the main period and its
        double. For actual variations, both will be present. For a noise result,
        the double period signal will not be present.
    cores: int, optional
        The number of parallel cores to use.
        By default it will try to use all available CPU cores.

    Returns
    -------
    periods: ndarray[n_periods]
        Trial periods.
    theta: ndarray[n_periods]
        Phase dispersion statistic.

    References
    ----------
    .. [#] R. F. Stellingwerf, "Period Determination of RR Lyrae Stars,"
       RR Lyrae Stars, Metal-Poor Stars and the Galaxy, August 2011.
    """
    t = np.asarray(t)
    x = np.asarray(x)
    sigma = np.var(x, ddof=1)
    ne = t.size
    if x.size != ne:
        raise ValueError("Incompatible array shapes")
    theta_crit = 1. - 11. / ne ** 0.8
    dt = np.median(np.diff(t))
    t0 = np.max(t) - np.min(t)
    if p_max is None:
        p_max = oversample * t0
    if p_min is None:
        p_min = 2 * dt
    if n_periods is None:
        n_periods = int((1 / p_min - 1 / p_max) * oversample * t0 + 1)
    periods = np.linspace(p_min, p_max, n_periods)
    dp = (p_max - p_min) / (n_periods - 1)
    if cores is None or cores > MAX_CORES:
        cores = MAX_CORES
    with Pool(cores) as pool:
        tasks = [(t, x, sigma, period) for period in periods]
        thetas = pool.map(_pdm2, tasks)
    if do_subharmonic:
        can_average, = np.where((thetas < theta_crit) & (periods <= p_max / 2))
        sub_indices = np.round(2 * can_average + p_min / dp).astype(int)
        thetas[can_average] = (thetas[can_average] + thetas[sub_indices]) / 2
    if s > 0:
        thetas = gaussian_filter1d(thetas, sigma=s, truncate=3.0)
    return periods, thetas

# TODO: Analysis of Variance (Schwarzenberg-Czerny 1989)

# TODO: Gregory-Loredo method (Gregory & Loredo 1992)

# TODO: conditional entropy method (Graham et al. 2013)
