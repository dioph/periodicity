import matplotlib.pyplot as plt
import numpy as np
import pywt
from scipy.signal import hilbert

from .core import Periodogram, Timeseries
from .decomposition import EMD, _lmd_sift

__all__ = ["WPS", "HHT"]


class HHT(object):
    """Hilbert-Huang Transform"""

    def __init__(self, emd=None, method="DQ", norm_type="spline", norm_iter=10):
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

    def _normalize(self, imf, eps=1e-6, n_rep=2):
        """Huang et al. (2009)"""
        F = imf.copy()
        A = 1.0
        for it in range(self.norm_iter):
            if self.norm_type == "hilbert":
                env = np.abs(hilbert(F.val))
            elif self.norm_type == "spline":
                env, _ = F.abs().get_envelope(n_rep=n_rep)
            elif self.norm_type == "lmd":
                mu, env = _lmd_sift(F, n_rep=n_rep)
                F -= mu
            F /= env
            A *= env
            if np.max(np.abs(F.val)) - 1.0 < eps:
                break
        F.val[F.val > 1.0] = 1.0
        F.val[F.val < -1.0] = -1.0
        return A, F

    def __call__(self, signal):
        if not isinstance(signal, Timeseries):
            signal = Timeseries(val=signal)
        f, a = [], []
        imfs = self.emd(signal)
        for imf in imfs:
            if np.any(imf):
                if self.method == "DQ":
                    A, F = self._normalize(imf)
                    amp = A.val
                    phi = np.arctan2(np.sqrt(1 - F.val ** 2), F.val)
                    corr = np.sign(np.gradient(phi))
                    phi = np.unwrap(phi * corr)
                    freq = np.gradient(phi, F.time)
                    freq /= 2 * np.pi
                elif self.method == "NHT":
                    A, F = self._normalize(imf)
                    amp = A.val
                    phi = np.unwrap(np.angle(hilbert(F.val)))
                    freq = np.gradient(phi, F.time)
                    freq /= 2 * np.pi
                elif self.method == "TEO":
                    teo_x = signal.TEO.val
                    teo_xdot = signal.derivative.TEO.val
                    amp = teo_x / np.sqrt(teo_xdot)
                    freq = np.sqrt(teo_xdot / teo_x)
                    freq /= 2 * np.pi
                elif self.method == "HT":
                    analytic = hilbert(self.val)
                    amp = np.abs(analytic)
                    phi = np.unwrap(np.angle(analytic))
                    freq = np.gradient(phi, self.time)
                    freq /= 2 * np.pi
                f.append(Timeseries(signal.time, freq))
                a.append(Timeseries(signal.time, amp))
        self.signal = signal
        self.imfs = imfs
        self.freqs = f
        self.amps = a
        return f, a

    def spectrogram(self, freqs, smooth_freq=None, smooth_amp=None):
        tshape = len(self.signal)
        fshape = len(freqs)
        power = np.zeros((fshape, tshape), float)
        for i in range(len(self.freqs)):
            freq = self.freqs[i]
            amp = self.amps[i]
            if smooth_freq is not None:
                freq = freq.smooth(smooth_freq)
            if smooth_amp is not None:
                amp = amp.smooth(smooth_amp)
            for i in range(tshape):
                j = np.argmin(np.abs(freq[i].val - freqs))
                if 0 < j < fshape - 1:
                    power[j, i] += amp[i].val
        return power


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
    r_sum = np.transpose(np.sum(np.transpose(coefs) / scales ** 0.5, axis=-1))
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
        if not isinstance(signal, Timeseries):
            signal = Timeseries(val=signal)

        n_times = signal.size
        n_scales = self.periods.size
        family = "cmor2.0-1.0"
        dt = signal.median_ts
        scales = pywt.scale2frequency(family, 1) * self.periods / dt

        # Chooses the method with the lowest computational complexity
        conv_complex = n_scales * n_times
        fft_complex = (n_scales + n_times - 1) * np.log2(n_scales + n_times - 1)
        if fft_complex < conv_complex:
            method = "fft"
        else:
            method = "conv"
        self.coefs, _ = pywt.cwt(
            signal.val - signal.val.mean(), scales, family, dt, method=method
        )

        # Defines useful attributes
        self.power = np.square(np.abs(self.coefs))
        self.signal = signal
        self.time = signal.time
        self.scales = scales
        self.spectrum = (self.power.T / self.scales).T
        return self.spectrum

    def coi(self, coi_samples=100):
        """
        coi_samples: int, optional
            Number of samples of the Cone of Influence (COI).
            Used for plotting (default is 100).
        """
        corr = 2 ** 0.5
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
        return Timeseries(t_samples, p_samples)

    @property
    def mask_coi(self):
        corr = 2 ** 0.5
        t_max = np.max(self.time)
        t_min = np.min(self.time)
        t_mesh, p_mesh = np.meshgrid(self.time, self.periods)
        return corr * p_mesh < np.minimum(t_mesh - t_min, t_max - t_mesh)

    @property
    def masked_spectrum(self):
        return np.ma.masked_array(self.spectrum, ~self.mask_coi)

    def sav(self, pmin=None, pmax=None):
        mask = np.ones(len(self.periods), bool)
        if pmin is not None:
            mask &= self.periods >= pmin
        if pmax is not None:
            mask &= self.periods <= pmax
        return Timeseries(self.time, np.mean(self.spectrum[mask], axis=0))

    def masked_sav(self, pmin=None, pmax=None):
        mask = np.ones(len(self.periods), bool)
        if pmin is not None:
            mask &= self.periods >= pmin
        if pmax is not None:
            mask &= self.periods <= pmax
        return Timeseries(self.time, np.mean(self.masked_spectrum[mask], axis=0))

    def gwps(self, tmin=None, tmax=None):
        mask = np.ones(len(self.time), bool)
        if tmin is not None:
            mask &= self.time >= tmin
        if tmax is not None:
            mask &= self.time <= tmax
        return Periodogram(
            period=self.periods, val=np.mean(self.spectrum[:, mask], axis=1)
        )

    def masked_gwps(self, tmin=None, tmax=None):
        mask = np.ones(len(self.time), bool)
        if tmin is not None:
            mask &= self.time >= tmin
        if tmax is not None:
            mask &= self.time <= tmax
        return Periodogram(
            period=self.periods, val=np.mean(self.masked_spectrum[:, mask], axis=1)
        )

    def imshow(self, **kwargs):
        plt.imshow(self.spectrum, aspect="auto", **kwargs)

    def contour(self, **kwargs):
        plt.contourf(self.time, self.periods, self.spectrum, **kwargs)

    def plot_coi(self, coi_samples=100, **kwargs):
        coi = self.coi(coi_samples)
        plt.fill_between(coi.time, coi.val, self.periods.max(), **kwargs)
