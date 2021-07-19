import copy
import warnings

from astropy.convolution import Box1DKernel
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate, signal
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit, minimize

__all__ = ["Signal", "Timeseries", "Periodogram"]
# TODO: Spectrogram (2D methods)


class Signal(object):
    __array_priority__ = 1000

    def __init__(self, val=None):
        """
        val: array-like
            Signal samples.
        """
        if val is None:
            raise ValueError("Values must be given.")
        self.val = np.asarray(val)

    @property
    def size(self):
        return len(self.val)

    def copy(self):
        return copy.deepcopy(self)

    def __len__(self):
        return self.size

    def __add__(self, other):
        result = self.copy()
        if isinstance(other, Signal):
            if len(self) != len(other):
                raise ValueError("Cannot add two Signals with different lengths.")
            result.val = self.val + other.val
        else:
            result.val = self.val + other
        return result

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-1 * other)

    def __rsub__(self, other):
        return (-1 * self).__add__(other)

    def __mul__(self, other):
        result = self.copy()
        if isinstance(other, Signal):
            if len(self) != len(other):
                raise ValueError("Cannot multiply two Signals with different lengths.")
            result.val = self.val * other.val
        else:
            result.val = self.val * other
        return result

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self.__mul__(1.0 / other)

    def __rtruediv__(self, other):
        result = self.copy()
        if isinstance(other, Signal):
            if len(self) != len(other):
                raise ValueError("Cannot divide two Signals with different lengths.")
            result.val = other.val / self.val
        else:
            result.val = other / self.val
        return result

    def __div__(self, other):
        return self.__truediv__(other)

    def __rdiv__(self, other):
        return self.__rtruediv__(other)

    def max(self):
        return self[self.val.argmax()]

    def abs(self):
        result = self.copy()
        result.val = np.abs(result.val)
        return result

    def find_peaks(self, delta=0.0):
        """Finds local maxima and corresponding peak prominences.

        Parameters
        ----------
        delta: float, optional
            Minimum peak prominence.
            Recommended: ``delta >= rms_noise * 5``

        Returns
        -------
        peaks: `Signal`
            [t_max, y_max] for each maximum found.
        heights: ndarray
            Peak prominences for each maximum found.
        """
        maxima, res = signal.find_peaks(self.val, prominence=delta)
        peaks = self[maxima]
        heights = res["prominences"]

        if heights.size == 0 and delta > 1e-6:
            return self.find_peaks(delta=delta / 2)

        return peaks, heights

    def find_extrema(self, include_edges=False, **peak_kwargs):
        """Finds local extrema.

        Parameters
        ----------
        include_edges: bool, optional
            Whether to include first and last elements of the signal as maxima and minima.
        peak_kwargs: optional
            Keyword arguments to pass into signal.find_peaks.

        Returns
        -------
        maxima: `Signal`
            Maximum samples.
        minima: `Signal`
            Minimum samples.
        """
        maxima, _ = signal.find_peaks(self.val, **peak_kwargs)
        minima, _ = signal.find_peaks(-self.val, **peak_kwargs)
        if include_edges:
            maxima = np.hstack([0, maxima, -1])
            minima = np.hstack([0, minima, -1])
        return self[maxima], self[minima]

    def find_zero_crossings(self, height=None, delta=0.0):
        """Finds zero crossing indices.

        Parameters
        ----------
        height: float, optional
            Maximum deviation from zero.
        delta: float, optional
            Prominence value used in `scipy.signal.find_peaks` when `height` is
            specified.

        Returns
        -------
        ind_zer: `Signal`
            Zero-crossing samples.
        """
        if height is None:
            (ind_zer,) = np.where(np.diff(np.signbit(self.val)))
        else:
            ind_zer, _ = signal.find_peaks(
                -np.abs(self.val), height=-height, prominence=delta
            )
        return self[ind_zer]

    def estimate_noise(self, sigma=3.0, n_iter=3):
        """Estimates the standard deviation of a white gaussian noise in the data.

        Parameters
        ----------
        sigma: float, optional
            sigma_clip value (defaults to 3.0).
        n_iter: int, optional
            Number of iterations for k-sigma clipping (defaults to 3).

        Returns
        -------
        noise: float
            Estimate of standard deviation of the noise.
        """
        residue = self.val - signal.medfilt(self.val, 3)
        sd = np.std(residue)
        index = np.arange(residue.size)
        for i in range(n_iter):
            mu = np.mean(residue[index])
            sd = np.std(residue[index])
            (index,) = np.where(np.abs(residue - mu) < sigma * sd)
        noise = sd / 0.893421
        return noise

    def smooth(self, kernel):
        """Time-domain low-pass filtering.

        Parameters
        ----------
        kernel: int or array-like
            Standard deviation of Gaussian filter used to smooth, measured in
            samples. Alternatively, any FIR filter can be passed as an array
            of samples to be convolved.

        Returns
        -------
        smoothed_signal: `Signal`
            Smoothed signal.
        """
        smoothed_signal = self.copy()
        if isinstance(kernel, int):
            xf = gaussian_filter1d(self.val, sigma=kernel, truncate=3.0)
        else:
            w = kernel.shape[0]
            y = np.pad(self.val, w - 1, mode="reflect")
            yf = np.convolve(y, kernel, mode="valid")
            xf = yf[w // 2 : -w // 2 + 1]
        smoothed_signal.val = xf
        return smoothed_signal


class Timeseries(Signal):
    __array_priority__ = 1000

    def __init__(self, time=None, val=None, assume_sorted=False):
        """
        time: array-like
            Signal timestamps.
        val: array-like
            Signal samples.
        # TODO: incorporate measurement uncertainties.
        """
        super(Timeseries, self).__init__(val)
        if time is None:
            time = np.arange(len(val))
        self.time = np.asarray(time)
        if len(self.time) != len(self.val):
            raise ValueError("Input arrays have incompatible lengths.")
        # Avoid sorting if possible
        if not assume_sorted:
            if np.any(np.diff(self.time) < 0):
                sorted_ids = np.argsort(self.time)
                self.time = self.time[sorted_ids]
                self.val = self.val[sorted_ids]

    def __getitem__(self, key):
        result = self.copy()
        result.time = self.time[key]
        result.val = self.val[key]
        return result

    @property
    def baseline(self):
        return self.time[-1] - self.time[0]

    @property
    def median_ts(self):
        return np.median(np.diff(self.time))

    @property
    def ts(self):
        if np.allclose(np.diff(self.time), self.median_ts):
            return self.median_ts
        raise AttributeError(
            "The sampling period is only strictly defined for "
            "uniformly sampled signals. Use median_ts for a median value."
        )

    @property
    def fft(self, oversample=1.0):
        n = int(oversample * self.size)
        freqs = np.fft.rfftfreq(n=n, d=self.ts)
        coefs = np.fft.rfft(self.val, n=n)
        return Periodogram(frequency=freqs, val=coefs)

    @property
    def psd(self):
        power = self.fft
        power.val = np.square(np.abs(power.val))
        return power

    @property
    def derivative(self):
        result = self.copy()
        result.val = np.gradient(self.val, self.time)
        return result

    @property
    def TEO(self):
        """Teager Energy Operator (TEO)

        J. F. Kaiser, "On Teager’s energy algorithm and its generalization to
        continuous signals", Proc. 4th IEEE Signal Processing Workshop, 1990.
        """
        return self.derivative * self.derivative - self * self.derivative.derivative

    def timeshift(self, t0):
        result = self.copy()
        result.time += t0
        return result

    def timescale(self, alpha):
        result = self.copy()
        result.time *= alpha
        return result

    def curve_fit(self, fun, **kwargs):
        if isinstance(fun, int):
            fit = np.poly1d(np.polyfit(self.time, self.val, fun))(self.time)
        else:
            popt, pcov = curve_fit(fun, self.time, self.val, **kwargs)
            fit = fun(self.time, *popt)
        result = self.copy()
        result.val = fit
        return result

    def acf(self, max_lag=None, unbias=False):
        """Auto-Correlation Function implemented using IFFT of the power spectrum.

        Parameters
        ----------
        max_lag: int or float, optional
            Maximum lag to compute ACF.
            If given as a float, will be assumed to be a measure of time and the
            ACF will be computed for lags lower than or equal to `max_lag`.
        unbias: bool, optional
            Whether to correct for the "mask effect" (dividing Ryy by the ACF of a
            signal equal to 1 on the original domain of y and equal to 0 on the
            padding's domain).

        Returns
        -------
        acf: `Signal`
            ACF of input signal.
        """
        if max_lag is None:
            max_lag = self.size // 2
        if type(max_lag) is float:
            max_lag = np.where(self.time - np.min(self.time) <= max_lag)[0][-1] + 1
        f = np.fft.fft(self.val - self.val.mean(), n=2 * self.size)
        ryy = np.fft.ifft(f * np.conjugate(f))
        if unbias:
            mask = np.fft.fft(np.ones_like(self.val), n=2 * self.size)
            correction = np.fft.ifft(mask * np.conjugate(mask))
            ryy /= correction
        ryy = np.real(ryy[:max_lag])
        ryy /= ryy[0]
        lags = self.time[:max_lag] - np.min(self.time[:max_lag])
        return Timeseries(lags, ryy, assume_sorted=True)

    def join(self, other):
        if len(np.intersect1d(self.time, other.time) > 0):
            warnings.warn(
                "There are overlapping timestamps. The corresponding "
                "timestamps in the returned Signal have both samples."
            )
        t_new = np.concatenate([self.time, other.time])
        v_new = np.concatenate([self.val, other.val])
        return Timeseries(t_new, v_new)

    def split(self, max_gap=None):
        if max_gap is None:
            max_gap = 1.5 * self.median_ts
        ids = np.where(np.diff(self.time) > max_gap)[0]
        ids = np.hstack([0, ids + 1, self.size])
        splits = []
        for i in range(len(ids) - 1):
            splits.append(self[ids[i] : ids[i + 1]])
        return splits

    def downsample(self, ts, func=np.nanmean):
        relative_time = self.time - self.time[0]
        relative_bins = np.arange(0, self.baseline, ts)
        bins = relative_bins + self.time[0]
        n_bins = len(bins)
        ids = np.digitize(relative_time, relative_bins) - 1
        unique_ids = np.unique(ids)
        binned_signal = self.copy()
        result = np.repeat(np.nan, n_bins)
        for i in unique_ids:
            mask = ids == i
            result[i] = func(self.val[mask])
        binned_signal.time = bins
        binned_signal.val = result
        binned_signal = binned_signal[np.isfinite(result)]
        return binned_signal

    def resample_uniform(self, ts=None, kind="linear"):
        """Linear interpolation to create a uniformly sampled signal.

        Parameters
        ----------
        ts: float, optional
            Sampling period. If omitted, it will be estimated from `median_ts`.

        Returns
        -------
        uniform_signal: `Signal`
            Interpolated signal.
        """
        if ts is None:
            ts = self.median_ts
        uniform_signal = self.copy()
        t_new = np.arange(np.min(self.time), np.max(self.time), ts)
        v_new = interpolate.interp1d(self.time, self.val)(t_new)
        uniform_signal.time = t_new
        uniform_signal.val = v_new
        return uniform_signal

    def fill_gaps(self, ts=None, kind="linear", k=0.0):
        if ts is None:
            ts = self.median_ts
        filled_signal = self.copy()
        filled_signal = filled_signal[np.isfinite(filled_signal.val)]
        t_old = filled_signal.time
        v_old = filled_signal.val
        t_new = [t_old[0]]
        for t in t_old[1:]:
            prevtime = t_new[-1]
            while (t - prevtime) > 1.2 * ts:
                t_new.append(prevtime + ts)
                prevtime = t_new[-1]
            t_new.append(t)
        t_new = np.asarray(t_new, float)
        is_old = np.isin(t_new, t_old)
        v_new = np.zeros(len(t_new))
        v_new[is_old] = np.copy(v_old)
        if kind == "constant":
            v_new[~is_old] = k
        elif kind == "hold":
            j = 0
            k = v_new[0]
            for i in range(1, len(t_new)):
                if is_old[i]:
                    k = v_new[i]
                    if j < is_old.sum() - 1:
                        v_new[i] = 0.5 * (v_old[j] + v_old[j + 1])
                        j += 1
                else:
                    v_new[i] = k
        elif kind == "linear":
            v_new[~is_old] = np.interp(t_new[~is_old], t_old, v_old)
        elif kind == "random":
            noise = self.estimate_noise()
            v_new[~is_old] = np.random.normal(self.val.mean(), noise, (~is_old).sum())
        elif kind == "mirror":
            ids = np.where(np.diff(is_old))[0] + 1
            n_gaps = ids.size // 2
            for i in range(n_gaps):
                start = ids[2 * i]
                end = ids[2 * i + 1]
                gap_size = end - start
                left_ids = np.arange(start, start + gap_size // 2)
                right_ids = np.arange(end - gap_size // 2, end)
                l_vals = 2 * start - left_ids - 1
                r_vals = 2 * end - right_ids - 1
                v_new[left_ids] = v_new[l_vals]
                v_new[right_ids] = v_new[r_vals]
                if gap_size % 2 == 1:
                    center = (start + end - 1) // 2
                    v_new[center] = (v_new[center - 1] + v_new[center + 1]) / 2
        else:
            raise ValueError("Unknown imputation method '{}'".format(kind))
        filled_signal.time = t_new
        filled_signal.val = v_new
        return filled_signal

    def get_envelope(self, n_rep=0, **peak_kwargs):
        """Interpolates maxima/minima with cubic splines into upper/lower envelopes.

        Parameters
        ----------
        peak_kwargs: float, optional
            Keyword arguments to use in `find_extrema`.
        n_rep: int, optional
            Number of extrema to repeat on either side of the signal.

        Returns
        -------
        upper: `Signal`
            Upper envelope.
        lower: `Signal`
            Lower envelope.
        """
        peaks, dips = self.find_extrema(include_edges=True, **peak_kwargs)
        if peaks.size < (2 + n_rep) or dips.size < (2 + n_rep):
            raise TypeError("Signal doesn't have enough extrema for padding.")
        t_max = np.pad(peaks.time, n_rep, "reflect", reflect_type="odd")
        y_max = np.pad(peaks.val, n_rep, "reflect")
        t_min = np.pad(dips.time, n_rep, "reflect", reflect_type="odd")
        y_min = np.pad(dips.val, n_rep, "reflect")
        t_max = np.delete(t_max, [n_rep, -n_rep - 1])
        y_max = np.delete(y_max, [n_rep, -n_rep - 1])
        t_min = np.delete(t_min, [n_rep, -n_rep - 1])
        y_min = np.delete(y_min, [n_rep, -n_rep - 1])
        if t_max.size < 4 or t_min.size < 4:
            raise TypeError(
                "Signal doesn't have enough extrema for envelope interpolation."
            )
        upper = self.copy()
        lower = self.copy()
        tck = interpolate.splrep(t_max, y_max)
        upper.val = interpolate.splev(upper.time, tck)
        tck = interpolate.splrep(t_min, y_min)
        lower.val = interpolate.splev(lower.time, tck)
        return upper, lower

    def band_pass(self, lo, hi, order=5):
        """Implements a band-pass IIR butterworth filter.

        Parameters
        ----------
        lo: float
            Lower cutoff frequency.
        hi: float
            Higher cutoff frequency.
        order: int, optional
            Order of the butterworth filter. Default is 5.

        Returns
        -------
        filt_signal: `Signal`
            Filtered signal.
        """
        fs = 1.0 / self.median_ts
        nyq = 0.5 * fs
        lo /= nyq
        hi /= nyq
        b, a = signal.butter(N=order, Wn=[lo, hi], btype="band")
        xf = signal.filtfilt(b, a, self.val)
        filt_signal = self.copy()
        filt_signal.val = xf
        return filt_signal

    def acf_period_quality(self, p_min, p_max):
        """Calculates the ACF quality of a band-pass filtered version of the signal.

        Parameters
        ----------
        p_min: float, optional
            Lower cutoff period to filter signal.
        p_max: float, optional
            Higher cutoff period to filter signal.
            Must be between `p_min` and half the baseline.

        Returns
        -------
        best_per: float
            Highest peak (best period) for the ACF of the filtered signal.
        height: float
            Maximum height for the ACF of the filtered signal.
        quality: float
            Quality factor of the best period.
        """
        ml = np.where(self.time - self.time[0] >= 2 * p_max)[0][0]
        rxx = self.band_pass(1 / p_max, 1 / p_min).acf(max_lag=ml)
        if p_max >= 20:
            rxx = rxx.smooth(Box1DKernel(width=p_max // 10))
            rxx /= rxx.max().val
        peaks, heights = rxx.find_peaks()
        best_per = peaks.time[np.argmax(heights)]
        height = np.max(heights)
        tau_max = 20 * p_max / best_per

        def rss(params):
            log_aa, log_tt = params
            aa = np.exp(log_aa)
            tt = np.exp(log_tt)
            acf_model = (
                aa * np.exp(-rxx.time / tt) * np.cos(2 * np.pi * rxx.time / best_per)
            )
            return np.sum(np.square(rxx.val - acf_model))

        results = minimize(fun=rss, x0=[0.0, np.log(best_per * 2)])
        log_amp, log_tau = results.x
        tau = min(np.exp(log_tau), tau_max)
        quality = (tau / best_per) * (ml * height / rss([log_amp, np.log(tau)]))
        return best_per, height, quality

    def plot(self, *args, ax=plt, **kwargs):
        ax.plot(self.time, self.val, *args, **kwargs)


class Periodogram(Signal):
    def __init__(self, frequency=None, period=None, val=None):
        """
        frequency: array-like
            Frequency grid.
        period: array-like
            Period grid.
        val: array-like
            Periodogram samples.
        """
        super(Periodogram, self).__init__(val)
        if period is not None:
            frequency = 1.0 / period
        elif frequency is not None:
            period = 1.0 / frequency
        else:
            raise ValueError("At least one of frequency and period must be given.")
        self.frequency = np.asarray(frequency)
        self.period = np.asarray(period)

    def __getitem__(self, key):
        result = self.copy()
        result.frequency = self.frequency[key]
        result.period = self.period[key]
        result.val = self.val[key]
        return result

    def plot(self, *args, ax=plt, use_frequency=False, **kwargs):
        if use_frequency:
            ax.plot(self.frequency, self.val, *args, **kwargs)
        else:
            ax.plot(self.period, self.val, *args, **kwargs)


class Spectrogram(Signal):
    def __init__(self, frequency=None, period=None, time=None, val=None):
        """"""
        super(Spectrogram, self).__init__(val)
