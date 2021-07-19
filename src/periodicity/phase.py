from multiprocessing import Pool, cpu_count

import numpy as np

from .core import Periodogram, Timeseries

MAX_CORES = cpu_count()

__all__ = ["StringLength", "PDM"]

# TODO: Analysis of Variance (Schwarzenberg-Czerny 1989)

# TODO: Gregory-Loredo method (Gregory & Loredo 1992)

# TODO: conditional entropy method (Graham et al. 2013)


class StringLength(object):
    """String Length [#]_.

    Parameters
    ----------
    dphi: float, optional
        Factor to multiply (1 / baseline) in order to get the frequency
        separation (the default is 0.1).
    n_periods: int, optional
        Number of trial periods (the default is 1000).
    cores: int, optional
        The number of parallel cores to use.
        By default it will try to use all available CPU cores.

    References
    ----------
    .. [#] M. M. Dworetsky, "A period-finding method for sparse randomly spaced
       observations or 'How long is a piece of string ?'," MNRAS, June 1983.
    """

    def __init__(self, dphi=0.1, n_periods=1000, cores=None):
        self.dphi = dphi
        self.n_periods = n_periods
        if cores is None or cores > MAX_CORES:
            cores = MAX_CORES
        self.cores = cores

    def _stringlength(self, period):
        """Calculates the string length for a single trial period."""
        phi = (self.t / period) % 1
        sorted_args = np.argsort(phi)
        phi = phi[sorted_args]
        m = self.m[sorted_args]
        ll = np.hypot(np.roll(m, -1) - m, np.roll(phi, -1) - phi).sum()
        return ll

    def __call__(self, signal):
        """
        Returns
        -------
        periods: ndarray[n_periods]
            Trial periods.
        ell: ndarray[n_periods]
            String length for each period.
        """
        if not isinstance(signal, Timeseries):
            signal = Timeseries(val=signal)
        self.signal = signal
        # scale x to range from -0.25 to +0.25
        self.t = signal.time
        x = signal.val
        self.m = (x - np.max(x)) / (2 * (np.max(x) - np.min(x))) + 0.25
        df = self.dphi / signal.baseline
        periods = 1 / np.linspace(self.n_periods * df, df, self.n_periods)
        with Pool(self.cores) as pool:
            ell = pool.map(self._stringlength, periods)
        return Periodogram(period=periods, val=ell)


class PDM(object):
    """Phase Dispersion Minimization [#]_.

    Parameters
    ----------
    nb: int, optional
        Number of phase bins (the default is 5).
    nc: int, optional
        Number of covers per bin (the default is 2).
    p_min, p_max: float, optional
        Minimum/maximum trial period normalized by the baseline.
    n_periods: int, optional
        Number of trial periods (the default is 1000).
    oversample: scalar, optional
        If `p_max` is omitted, the time baseline multiplied by `oversample` will
        be used instead.
    do_subharmonic: bool, optional
        Whether to perform subharmonic averaging. This option looks for a
        significant minimum in :math:`\\theta` at both the main period and its
        double. For actual variations, both will be present. For a noise result,
        the double period signal will not be present.
    cores: int, optional
        The number of parallel cores to use.
        By default it will try to use all available CPU cores.

    References
    ----------
    .. [#] R. F. Stellingwerf, "Period determination using phase dispersion
       minimization," ApJ, September 1978.
    .. [#] R. F. Stellingwerf, "Period Determination of RR Lyrae Stars,"
       RR Lyrae Stars, Metal-Poor Stars and the Galaxy, August 2011.
    """

    def __init__(
        self, nb, nc, p_min, p_max, n_periods, oversample, do_subharmonic, cores
    ):
        self.nb = nb
        self.nc = nc
        self.p_min = p_min
        self.p_max = p_max
        self.n_periods = n_periods
        self.oversample = oversample
        self.do_subharmonic = do_subharmonic
        self.cores = cores

    def _pdm(self, period):
        """Calculates the PDM theta statistic for a single trial period."""
        m0 = self.nb * self.nc
        phi = (self.t / period) % 1
        sorted_args = np.argsort(phi)
        phi = phi[sorted_args]
        m = self.x[sorted_args]
        mj = []
        m0_good = 0
        for k in range(m0):
            mask = phi >= k / m0
            mask &= phi < (k + self.nc) / m0
            mask |= phi < (k - (m0 - self.nc)) / m0
            mk = m[mask]
            if mk.size > 1:
                mj.append(mk)
                m0_good += 1
        sj = np.array([np.var(k, ddof=1) for k in mj])
        nj = np.array([k.size for k in mj])
        ss = np.sum((nj - 1) * sj) / (np.sum(nj) - m0_good)
        theta = ss / self.sigma
        return theta

    def __call__(self, signal):
        """
        Returns
        -------
        periods: ndarray[n_periods]
            Trial periods.
        theta: ndarray[n_periods]
            Phase dispersion statistic as in Eq. 3 of the paper.
        """
        if not isinstance(signal, Timeseries):
            signal = Timeseries(val=signal)
        self.t = signal.time
        self.x = signal.val
        self.sigma = np.var(self.x, ddof=1)
        theta_crit = 1.0 - 11.0 / signal.size ** 0.8
        t0 = signal.baseline
        if self.p_min is None:
            p_min = 2 * signal.median_ts
        else:
            p_min = self.p_min
        if self.p_max is None:
            p_max = self.oversample * t0
        else:
            p_max = self.p_max
        if self.n_periods is None:
            n_periods = int((1 / p_min - 1 / p_max) * self.oversample * t0 + 1)
        else:
            n_periods = self.n_periods
        self.periods = np.linspace(p_min, p_max, n_periods)
        dp = self.periods[1] - self.periods[0]
        if self.cores is None or self.cores > MAX_CORES:
            cores = MAX_CORES
        with Pool(cores) as pool:
            thetas = pool.map(self._pdm, self.periods)
        if self.do_subharmonic:
            (can_average,) = np.where(
                (thetas < theta_crit) & (self.periods <= p_max / 2)
            )
            sub_indices = np.round(2 * can_average + p_min / dp).astype(int)
            thetas[can_average] = (thetas[can_average] + thetas[sub_indices]) / 2
        return Periodogram(period=self.periods, val=thetas)
