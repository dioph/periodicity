import copy

import numpy as np

from .core import Periodogram, Timeseries

__all__ = ["GLS", "BGLST"]
# TODO: check out Supersmoother (Reimann 1994)


def _trig_sum(t, w, df, nf, fmin, n=5):
    """
    Computes
        S_j = sum_i w_i * sin(2 pi * f_j * t_i),
        C_j = sum_i w_i * cos(2 pi * f_j * t_i)
    using an FFT-based O[Nlog(N)] method.
    """
    nfft = 1 << int(nf * n - 1).bit_length()
    tmin = t.min()
    w = w * np.exp(2j * np.pi * fmin * (t - tmin))
    tnorm = ((t - tmin) * nfft * df) % nfft
    grid = np.zeros(nfft, dtype=w.dtype)
    integers = tnorm % 1 == 0
    np.add.at(grid, tnorm[integers].astype(int), w[integers])
    tnorm, w = tnorm[~integers], w[~integers]
    ilo = np.clip((tnorm - 2).astype(int), 0, nfft - 4)
    numerator = w * np.prod(tnorm - ilo - np.arange(4)[:, np.newaxis], 0)
    denominator = 6
    for j in range(4):
        if j > 0:
            denominator *= j / (j - 4)
        ind = ilo + (3 - j)
        np.add.at(grid, ind, numerator / (denominator * (tnorm - ind)))
    fftgrid = np.fft.ifft(grid)[:nf]
    if tmin != 0:
        f = fmin + df * np.arange(nf)
        fftgrid *= np.exp(2j * np.pi * tmin * f)
    C = nfft * fftgrid.real
    S = nfft * fftgrid.imag
    return S, C


class GLS(object):
    """
    References
    ----------
    .. [1] Press W.H. and Rybicki, G.B, "Fast algorithm for spectral analysis
        of unevenly sampled data". ApJ 1:338, p277, 1989
    .. [2] M. Zechmeister and M. Kurster, A&A 496, 577-584 (2009)
    .. [3] W. Press et al, Numerical Recipes in C (2002)
    """

    def __init__(self, fmin=None, fmax=None, n=5, psd=False):
        """Computes the generalized Lomb-Scargle periodogram of a discrete signal.

        Parameters
        ----------
        fmin: float, optional
            Minimum frequency.
            If not given, it will be determined from the time baseline.
        fmax: float, optional
            Maximum frequency.
            If not given, defaults to the pseudo-Nyquist limit.
        n: float, optional
            Samples per peak (default is 5).
        psd: bool, optional
            Whether to leave periodogram non-normalized (Fourier Spectral Density).
        """
        self.fmin = fmin
        self.fmax = fmax
        self.n = n
        self.psd = psd

    def __call__(self, signal, err=None, fit_mean=True):
        """Fast implementation of the Lomb-Scargle periodogram.
        Based on the lombscargle_fast implementation of the astropy package.
        Assumes an uniform frequency grid, but the errors can be heteroscedastic.

        Parameters
        ----------
        err: array-like, optional
            Measurement uncertainties for each sample.
        fit_mean: bool, optional
            If True, then let the mean vary with the fit.
        """
        if not isinstance(signal, Timeseries):
            signal = Timeseries(val=signal)
        df = 1.0 / signal.baseline / self.n
        if self.fmin is None:
            fmin = 0.5 * df
        else:
            fmin = self.fmin
        if self.fmax is None:
            fmax = 0.5 / signal.median_ts
        else:
            fmax = self.fmax
        self.frequency = np.arange(fmin, fmax + df, df)
        nf = self.frequency.size
        if err is None:
            err = np.ones_like(signal.val)
        self.err = err
        w = err ** -2.0
        w /= w.sum()
        t = signal.time
        if fit_mean:
            y = signal.val - np.dot(w, signal.val)
        else:
            y = signal.val
        Sh, Ch = _trig_sum(t, w * y, df, nf, fmin)
        S2, C2 = _trig_sum(t, w, 2 * df, nf, 2 * fmin)
        if fit_mean:
            S, C = _trig_sum(t, w, df, nf, fmin)
            tan_2omega_tau = (S2 - 2 * S * C) / (C2 - (C * C - S * S))
        else:
            tan_2omega_tau = S2 / C2
        S2w = tan_2omega_tau / np.sqrt(1 + tan_2omega_tau * tan_2omega_tau)
        C2w = 1 / np.sqrt(1 + tan_2omega_tau * tan_2omega_tau)
        Cw = np.sqrt(0.5) * np.sqrt(1 + C2w)
        Sw = np.sqrt(0.5) * np.sign(S2w) * np.sqrt(1 - C2w)
        YY = np.dot(w, y ** 2)
        YC = Ch * Cw + Sh * Sw
        YS = Sh * Cw - Ch * Sw
        CC = 0.5 * (1 + C2 * C2w + S2 * S2w)
        SS = 0.5 * (1 - C2 * C2w - S2 * S2w)
        if fit_mean:
            CC -= (C * Cw + S * Sw) ** 2
            SS -= (S * Cw - C * Sw) ** 2
        power = YC * YC / CC + YS * YS / SS
        if self.psd:
            power *= 0.5 * (err ** -2.0).sum()
        else:
            power /= YY
        self.signal = signal
        self.periodogram = Periodogram(frequency=self.frequency, val=power)
        return self.periodogram

    def copy(self):
        return copy.deepcopy(self)

    def bootstrap(self, n_bootstraps, random_seed=None):
        rng = np.random.default_rng(random_seed)
        bs_replicates = np.empty(n_bootstraps)
        ndata = len(self.signal)
        gls = self.copy()
        for i in range(n_bootstraps):
            bs_sample = self.signal.copy()
            bs_sample.val = self.signal.val[rng.integers(0, ndata, ndata)]
            bs_replicates[i] = gls(bs_sample).val.max()
        self.bs_replicates = bs_replicates
        return self.bs_replicates

    def fap(self, power):
        """
        fap_level: array-like, optional
            List of false alarm probabilities for which you want to calculate
            approximate levels. Can also be passed as a single scalar value.
        """
        return np.mean(power < self.bs_replicates)

    def fal(self, fap):
        return np.quantile(self.bs_replicates, 1 - fap)

    def window(self):
        gls = self.copy()
        return gls(0.0 * self.signal + 1.0, fit_mean=False)

    def model(self, tf, f0):
        """Compute the Lomb-Scargle model fit at a given frequency

        Parameters
        ----------
        tf: float or array-like
            The times at which the fit should be computed
        f0: float
            The frequency at which to compute the model

        Returns
        -------
        yf: ndarray
            The model fit evaluated at each value of tf
        """
        t = self.signal.time
        y = self.signal.val
        w = self.err ** -2.0
        y_mean = np.dot(y, w) / w.sum()
        y = y - y_mean
        X = (
            np.vstack(
                [
                    np.ones_like(t),
                    np.sin(2 * np.pi * f0 * t),
                    np.cos(2 * np.pi * f0 * t),
                ]
            )
            / self.err
        )
        theta = np.linalg.solve(np.dot(X, X.T), np.dot(X, y / self.err))
        Xf = np.vstack(
            [np.ones_like(tf), np.sin(2 * np.pi * f0 * tf), np.cos(2 * np.pi * f0 * tf)]
        )
        yf = y_mean + np.dot(Xf.T, theta)
        return yf


class BGLST(object):
    pass
