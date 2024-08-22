import warnings

import matplotlib.pyplot as plt
import numpy as np
import pywt
from scipy.signal import hilbert

from .core import TFSeries, TSeries
from .decomposition import EMD, LMD

__all__ = ["WPS", "HHT"]


class HHT(object):
    """Hilbert-Huang Transform"""

    def __init__(
        self,
        frequencies,
        emd=None,
        method="DQ",
        norm_type="spline",
        norm_iter=10,
        smooth_width=None,
    ):
        """
        Parameters
        ----------
        frequencies: array-like
            The frequency grid on which to project the representation.
        emd: callable, optional
            The decomposition used to extract AM-FM modes from the signal.
            It must accept a TSeries object as input and return a list of TSeries.
            If not given, a EMD with default settings is used.
        method: str, optional
            Method of instant frequency determination. Should be one of

            - 'DQ' (default): Direct Quadrature
            - 'NHT': Normalized Hilbert Transform
            - 'TEO': Teager Energy Operator
            - 'HT': Hilbert Transform
        norm_type: str, optional
            Type of amplitude normalization, used in 'DQ' and 'NHT'. Should be one of

            - 'hilbert': absolute value of analytic signal
            - 'spline' (default): cubic spline interpolation of maxima
            - 'lmd': local mean decomposition (single iteration)
        norm_iter: int, optional
            Number of normalization iterations, used to test for convergence.
            The normalized FM mode is clipped to unit amplitude after this many
            iterations if necessary. Default is 10.
        smooth_width: int, optional
            Width (standard deviation) of a gaussian FIR filter used to smooth the
            calculatd instantaneous frequencies and amplitudes.
        """
        self.frequencies = frequencies
        if np.any(np.diff(frequencies) < 0):
            self.frequencies.sort()
        if emd is None:
            emd = EMD()
        self.emd = emd
        if method.upper() not in ["DQ", "NHT", "TEO", "HT"]:
            raise ValueError(f"Method {method} is unknown.")
        self.method = method.upper()
        if norm_type.lower() not in ["hilbert", "spline", "lmd"]:
            raise ValueError(f"Method {norm_type} is unknown.")
        self.norm_type = norm_type.lower()
        self.norm_iter = norm_iter
        self.smooth_width = smooth_width

    def _normalize(self, mode, eps=1e-6, pad_width=2):
        """Huang et al. (2009)"""
        F = mode.copy()
        A = 1.0
        for it in range(self.norm_iter):
            if self.norm_type == "hilbert":
                env = np.abs(hilbert(F.values))
            elif self.norm_type == "spline":
                env, _ = np.abs(F).get_envelope(pad_width=pad_width)
            elif self.norm_type == "lmd":
                lmd = LMD(pad_width=pad_width)
                mu, env = lmd.sift(F)
                F = F - mu
            F = F / env
            A = A * env
            if np.max(np.abs(F)) - 1.0 < eps:
                break
        F.values = np.clip(F.values, -1.0, 1.0)
        return A, F

    def _spectrogram(self, freq_grid, freq, amp):
        tshape = len(freq)
        fshape = len(freq_grid)
        power = np.zeros((fshape, tshape), float)
        f_bins = np.clip(np.searchsorted(freq_grid, freq), 0, fshape - 1)
        power[f_bins, np.arange(tshape)] += amp
        power[[0, -1]] = 0
        return TFSeries(time=self.signal.time, frequency=freq_grid, values=power)

    def __call__(self, signal):
        if not isinstance(signal, TSeries):
            signal = TSeries(values=signal)
        self.signal = signal
        f, a = [], []
        tfs = []
        modes = self.emd(signal)
        for mode in modes:
            if np.any(mode):
                if self.method == "DQ":
                    A, F = self._normalize(mode)
                    amp = A.values
                    phi = np.arctan2(np.sqrt(1 - F.values**2), F.values)
                    corr = np.sign(np.gradient(phi))
                    phi = np.unwrap(phi * corr)
                    freq = np.gradient(phi, F.time)
                    freq /= 2 * np.pi
                elif self.method == "NHT":
                    A, F = self._normalize(mode)
                    amp = A.values
                    phi = np.unwrap(np.angle(hilbert(F.values)))
                    freq = np.gradient(phi, F.time)
                    freq /= 2 * np.pi
                elif self.method == "TEO":
                    teo_x = signal.TEO.values
                    teo_xdot = signal.derivative.TEO.values
                    amp = teo_x / np.sqrt(teo_xdot)
                    freq = np.sqrt(teo_xdot / teo_x)
                    freq /= 2 * np.pi
                elif self.method == "HT":
                    analytic = hilbert(signal.values)
                    amp = np.abs(analytic)
                    phi = np.unwrap(np.angle(analytic))
                    freq = np.gradient(phi, signal.time)
                    freq /= 2 * np.pi
                freq = TSeries(signal.time, freq)
                amp = TSeries(signal.time, amp)
                if self.smooth_width is not None:
                    freq = freq.smooth(self.smooth_width)
                    amp = amp.smooth(self.smooth_width)
                f.append(freq)
                a.append(amp)
                tfs.append(self._spectrogram(self.frequencies, freq.values, amp.values))
        self.modes = modes
        self.instant_fs = f
        self.instant_as = a
        self.tfs = tfs
        self.tf = sum(tfs)
        return self.tf


def denoise(data, family="db4", sigma=None, detrend=False):
    coefs = pywt.wavedec(data, family, mode="per")
    threshold = sigma * np.sqrt(2 * np.log(len(data)))
    new_coefs = coefs.copy()
    if detrend:
        new_coefs[0] = np.zeros_like(new_coefs[0])
    new_coefs[1:] = (pywt.threshold(c, value=threshold, mode="soft") for c in coefs[1:])
    y = pywt.waverec(new_coefs, family, mode="per")
    return y


def reconstruct(coefs, periods, dt, family):
    scales = pywt.scale2frequency(family, 1) * periods / dt
    mwf = pywt.ContinuousWavelet("morl").wavefun()
    y_0 = mwf[0][np.argmin(np.abs(mwf[1]))]
    r_sum = np.transpose(np.sum(np.transpose(coefs) / scales**0.5, axis=-1))
    return r_sum * (1 / y_0)


class WPS(object):
    """Wavelet Power Spectrum using Morlet wavelets.

    Parameters
    ----------
    periods: array-like
        Periods to consider, in the same time units as the input signal.
    """

    def __init__(self, periods):
        self.periods = np.asarray(periods)
        self.frequency = 1.0 / self.periods

    def __call__(self, signal):
        """Computes the WPS of the signal for the given periods.
        The following attributes will then be defined:
        - WPS.signal
        - WPS.time, WPS.scales
        - WPS.power, WPS.spectrum
        - WPS.mask_coi
        - WPS.masked_spectrum
        - WPS.sav, WPS.masked_sav
        - WPS.gwps, WPS.masked_gwps

        Parameters
        ----------
        signal: `Signal` or array-like
            Input signal

        Returns
        -------
        spectrum: ndarray[len(periods), len(signal)]
        """
        if not isinstance(signal, TSeries):
            signal = TSeries(values=signal)

        n_times = signal.size
        n_scales = self.periods.size
        family = "cmor2.0-1.0"
        dt = signal.median_dt
        scales = pywt.scale2frequency(family, 1) * self.periods / dt

        # Chooses the method with the lowest computational complexity
        conv_complex = n_scales * n_times
        fft_complex = (n_scales + n_times - 1) * np.log2(n_scales + n_times - 1)
        if fft_complex < conv_complex:
            method = "fft"
        else:
            method = "conv"
        self.coefs, _ = pywt.cwt(
            signal.values - signal.mean(), scales, family, dt, method=method
        )

        # Defines useful attributes
        power = np.square(np.abs(self.coefs))
        unbiased_power = (power.T / scales).T
        self.signal = signal
        self.time = signal.time
        self.scales = scales
        self.power = TFSeries(time=self.time, frequency=self.frequency, values=power)
        self.spectrum = TFSeries(
            time=self.time, frequency=self.frequency, values=unbiased_power
        )
        self.masked_spectrum = self.spectrum.copy()
        self.masked_spectrum.values[~self.mask_coi] = np.nan
        return self.spectrum

    def coi(self, coi_samples=100):
        """
        coi_samples: int, optional
            Number of samples of the Cone of Influence (COI).
            Used for plotting (default is 100).
        """
        corr = np.exp2(0.5)
        t_max = np.max(self.time)
        t_min = np.min(self.time)
        p_max = np.max(self.periods)
        p_min = np.min(self.periods)
        p_samples = np.logspace(np.log10(p_min), np.log10(p_max), coi_samples)
        p_samples = p_samples[corr * p_samples < (t_max - t_min) / 2]
        t1 = t_min + corr * p_samples
        t2 = t_max - corr * p_samples
        t_samples = np.hstack((t1, t2))
        p_samples = np.hstack((p_samples, p_samples))
        return TSeries(t_samples, p_samples)

    @property
    def mask_coi(self):
        corr = np.exp2(0.5)
        t_max = np.max(self.time)
        t_min = np.min(self.time)
        t_mesh, p_mesh = np.meshgrid(self.time, self.periods)
        return corr * p_mesh < np.minimum(t_mesh - t_min, t_max - t_mesh)

    def sav(self, pmin=None, pmax=None):
        mask = np.ones(len(self.periods), bool)
        if pmin is not None:
            mask &= self.periods >= pmin
        if pmax is not None:
            mask &= self.periods <= pmax
        return self.spectrum[mask].mean("frequency")

    def masked_sav(self, pmin=None, pmax=None):
        mask = np.ones(len(self.periods), bool)
        if pmin is not None:
            mask &= self.periods >= pmin
        if pmax is not None:
            mask &= self.periods <= pmax
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return self.masked_spectrum[mask].mean("frequency")

    def gwps(self, tmin=None, tmax=None):
        mask = np.ones(len(self.time), bool)
        if tmin is not None:
            mask &= self.time >= tmin
        if tmax is not None:
            mask &= self.time <= tmax
        return self.spectrum[:, mask].mean("time")

    def masked_gwps(self, tmin=None, tmax=None):
        mask = np.ones(len(self.time), bool)
        if tmin is not None:
            mask &= self.time >= tmin
        if tmax is not None:
            mask &= self.time <= tmax
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return self.masked_spectrum[:, mask].mean("time")

    def plot_coi(self, coi_samples=100, **kwargs):
        coi = self.coi(coi_samples)
        plt.fill_between(coi.time, coi.values, self.periods.max(), **kwargs)


class CompositeSpectrum(object):
    def __init__(self, periods):
        self.periods = periods
        self.wps = WPS(periods)

    def __call__(self, signal):
        if not isinstance(signal, TSeries):
            signal = TSeries(values=signal)
        wav = self.wps(signal)
        gwps = wav.mean("time")
        gwps /= gwps.amax()
        ryy = signal.fill_gaps().acf()
        cs = gwps * np.interp(gwps.period, ryy.time, ryy.values)
        return cs
