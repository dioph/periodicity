from numbers import Number
import warnings

import numpy as np
from scipy import interpolate, ndimage, optimize, signal
import xarray as xr

__all__ = ["TSeries", "FSeries", "TFSeries"]

HANDLED_FUNCTIONS = {}


def implements(numpy_function):
    """Register an __array_function__ implementation for Signal objects."""

    def decorator(func):
        HANDLED_FUNCTIONS[numpy_function] = func
        return func

    return decorator


def wrap_reduce(func):
    def wrapped_func(signal, dim=None, **kwargs):
        out = kwargs.get("out", ())
        if out:
            kwargs["out"] = out.values
        if dim is not None:
            kwargs["axis"] = signal.get_axis(dim)
        result = func(signal.values, **kwargs)
        axis = kwargs.get("axis", None)
        dims = signal._get_remaining_axes(axis, result)
        return signal._replace_data_and_dims(dims, result)

    return wrapped_func


@implements(np.full_like)
def full_like(signal, fill_value, **kwargs):
    return signal._replace_data(np.full_like(signal.values, fill_value, **kwargs))


@implements(np.zeros_like)
def zeros_like(signal, **kwargs):
    return np.full_like(signal, 0, **kwargs)


@implements(np.ones_like)
def ones_like(signal, **kwargs):
    return np.full_like(signal, 1, **kwargs)


class Signal(np.lib.mixins.NDArrayOperatorsMixin):
    _HANDLED_TYPES = (Number, np.ndarray, xr.DataArray)
    # TODO: consider using __slots__ (https://stackoverflow.com/questions/472000/usage-of-slots)

    def __init__(self, *args, **kwargs):
        self._array = xr.DataArray(*args, **kwargs)

    @property
    def values(self):
        return self._array.values

    @values.setter
    def values(self, new):
        self._array.values = new

    @property
    def dims(self):
        return self._array.dims

    @property
    def coords(self):
        return self._array._coords

    def get_axis(self, dim):
        try:
            return self.dims.index(dim)
        except ValueError:
            raise ValueError(f"{dim} not found in {self.dims}.")

    def _get_remaining_axes(self, axis, data):
        if data.shape == self.shape:
            return self.dims
        removed_axes = (
            range(self.ndim) if axis is None else np.atleast_1d(axis) % self.ndim
        )
        return [adim for n, adim in enumerate(self.dims) if n not in removed_axes]

    @property
    def index(self):
        return {k: v for k, v in self.coords.items() if isinstance(v, xr.IndexVariable)}

    def __len__(self):
        return len(self._array)

    @property
    def size(self):
        return self._array.size

    @property
    def shape(self):
        return self._array.shape

    @property
    def ndim(self):
        return self._array.ndim

    @property
    def dtype(self):
        return self._array.dtype

    @property
    def attrs(self):
        return self._array.attrs

    def __str__(self):
        return str(self._array).replace("xarray.DataArray", "Signal")

    def __repr__(self):
        return repr(self._array).replace("xarray.DataArray", "Signal")

    def _replace_data(self, data):
        new = type(self)(**self.index, values=data)
        new.attrs.update(self.attrs)
        return new

    def _replace_data_and_dims(self, dims, data):
        if len(dims) != data.ndim:
            raise ValueError("Wrong number of dimensions!")
        if data.ndim == 0:
            return data.item()
        index = {k: v for k, v in self.index.items() if k in dims}
        if data.ndim == 1:
            if "time" in dims:
                return TSeries(**index, values=data)
            if "frequency" in dims:
                return FSeries(**index, values=data)
        if data.ndim == 2:
            if "time" in dims and "frequency" in dims:
                return TFSeries(**index, values=data)
        raise ValueError(f"Unknwon dimension {dims}.")

    def copy(self):
        return self._replace_data(data=self.values.copy())

    def from_xray(self, result):
        return type(self)(result)

    def __array_function__(self, func, types, args, kwargs):
        """"https://numpy.org/doc/stable/reference/arrays.classes.html#numpy.class.__array_function__"""
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        if not all(issubclass(t, Signal) for t in types):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """https://numpy.org/doc/stable/reference/generated/numpy.lib.mixins.NDArrayOperatorsMixin.html"""
        out = kwargs.get("out", ())
        for x in inputs + out:
            if not isinstance(x, self._HANDLED_TYPES + (Signal,)):
                return NotImplemented
        if out:
            kwargs["out"] = tuple(
                x.values if isinstance(x, (Signal, xr.DataArray)) else x for x in out
            )
        n_arrays = sum(isinstance(x, (Signal, xr.DataArray)) for x in inputs)
        if n_arrays > 1:
            inputs = tuple(x._array if isinstance(x, Signal) else x for x in inputs)
            ufunc_method = getattr(ufunc, method)
            result = xr.apply_ufunc(ufunc_method, *inputs, **kwargs)
            result = self.from_xray(result)
        else:
            inputs = tuple(
                x.values if isinstance(x, (Signal, xr.DataArray)) else x for x in inputs
            )
            result = getattr(ufunc, method)(*inputs, **kwargs)
            if method == "__call__":
                result = self._replace_data(result)
            elif method == "__reduce__":
                axis = kwargs.get("axis", None)
                dims = self._get_remaining_axes(axis, result)
                result = self._replace_data_and_dims(dims, result)
        if method == "at":
            return None
        return result

    def to_pandas(self):
        return self._array.to_pandas()

    @implements(np.all)
    @wrap_reduce
    def all(self, **kwargs):
        return np.all(self, **kwargs)

    @implements(np.any)
    @wrap_reduce
    def any(self, **kwargs):
        return np.any(self, **kwargs)

    @implements(np.argmax)
    @wrap_reduce
    def argmax(self, **kwargs):
        return np.nanargmax(self, **kwargs)

    @implements(np.argmin)
    @wrap_reduce
    def argmin(self, **kwargs):
        return np.nanargmin(self, **kwargs)

    @implements(np.amax)
    @wrap_reduce
    def amax(self, **kwargs):
        return np.nanmax(self, **kwargs)

    def max(self):
        idx = np.unravel_index(self.argmax(axis=None), self.shape)
        idx = tuple(slice(i, i + 1) for i in idx)
        return self[idx]

    @implements(np.mean)
    @wrap_reduce
    def mean(self, **kwargs):
        return np.nanmean(self, **kwargs)

    @implements(np.median)
    @wrap_reduce
    def median(self, **kwargs):
        return np.nanmedian(self, **kwargs)

    @implements(np.amin)
    @wrap_reduce
    def amin(self, **kwargs):
        return np.nanmin(self, **kwargs)

    def min(self):
        idx = np.unravel_index(self.argmin(axis=None), self.shape)
        idx = tuple(slice(i, i + 1) for i in idx)
        return self[idx]

    @implements(np.prod)
    @wrap_reduce
    def prod(self, **kwargs):
        return np.nanprod(self, **kwargs)

    @implements(np.sum)
    @wrap_reduce
    def sum(self, **kwargs):
        return np.nansum(self, **kwargs)

    @implements(np.std)
    @wrap_reduce
    def std(self, **kwargs):
        return np.nanstd(self, **kwargs)

    @implements(np.var)
    @wrap_reduce
    def var(self, dim=None, **kwargs):
        return np.nanvar(self, **kwargs)

    @implements(np.roll)
    def roll(self, shift):
        return self._replace_data(np.roll(self.values, shift))

    def isnull(self):
        scalar_type = self.dtype.type
        if issubclass(scalar_type, (np.datetime64, np.timedelta64)):
            return np.isnat(self)
        elif issubclass(scalar_type, np.inexact):
            return np.isnan(self)
        elif issubclass(scalar_type, (np.bool_, np.integer, np.character, np.void)):
            return np.zeros_like(self, dtype=bool)
        else:
            return self != self

    def count(self, axis=None):
        return np.sum(np.logical_not(self.isnull()), axis=axis)

    def hist(self, *args, **kwargs):
        return self._array.plot.hist(*args, **kwargs)

    def find_peaks(self, include_edges=False, prominence=0.0, **peak_kwargs):
        """Finds local maxima and corresponding peak prominences.

        Parameters
        ----------
        include_edges: bool, optional
            Whether to include first and last elements of the signal as maxima.
        prominence: float, optional
            Minimum peak prominence.
            Recommended: ``prominence >= rms_noise * 5``
        peak_kwargs: optional
            Keyword arguments to pass into signal.find_peaks.

        Returns
        -------
        peaks: `Signal`
            [t_max, y_max] for each maximum found. Includes corresponding peak prominences as an attribute.
        """
        if self.ndim != 1:
            raise NotImplementedError("'find_peaks' is only implemented for 1D arrays.")
        maxima, res = signal.find_peaks(
            self.values, prominence=prominence, **peak_kwargs
        )
        if include_edges:
            maxima = np.hstack([0, maxima, -1])
            for key, values in res.items():
                if values.dtype == "float":
                    fillv = np.nan
                else:
                    fillv = -1
                res[key] = np.hstack([fillv, values, fillv])
        res["indices"] = maxima
        peaks = self[maxima]
        peaks.attrs.update(res)
        return peaks

    def find_dips(self, include_edges=False, prominence=0.0, **dip_kwargs):
        """Finds local minima and corresponding dip prominences.

        Parameters
        ----------
        include_edges: bool, optional
            Whether to include first and last elements of the signal as minima.
        prominence: float, optional
            Minimum dip prominence.
            Recommended: ``prominence >= rms_noise * 5``
        dip_kwargs: optional
            Keyword arguments to pass into signal.find_peaks.

        Returns
        -------
        dips: `Signal`
            [t_min, y_min] for each minimum found. Includes corresponding dip prominences as an attribute.
        """
        if self.ndim != 1:
            raise NotImplementedError("'find_dips' is only implemented for 1D arrays.")
        return -(-self).find_peaks(include_edges, prominence, **dip_kwargs)

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
        ind_zer: `ndarray`
            Zero-crossing sample indices.
        """
        if self.ndim != 1:
            raise NotImplementedError(
                "'find_zero_crossings' is only implemented for 1D arrays."
            )
        if height is None:
            (ind_zer,) = np.where(np.diff(np.signbit(self.values)))
        else:
            ind_zer, _ = signal.find_peaks(
                -np.abs(self.values), height=-height, prominence=delta
            )
        return ind_zer

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
        residue = self.values - ndimage.median_filter(self.values, 3)
        sd = np.std(residue)
        index = np.where(np.isfinite(residue))
        for i in range(n_iter):
            mu = np.mean(residue[index])
            sd = np.std(residue[index])
            index = np.where(np.abs(residue - mu) < sigma * sd)
        if self.ndim == 1:
            noise = sd / 0.893421
        elif self.ndim == 2:
            noise = sd / 0.969684
        else:
            raise NotImplementedError(
                "'estimate_noise' is only implemented for 1D or 2D arrays."
            )
        return noise

    def smooth(self, width, kernel="gaussian", **kwargs):
        """Low-pass FIR filter.

        width: int
            Width scaling the kernel size, measured in samples.
            If kernel == 'gaussian', it represents the standard deviation.
            If kernel == 'boxcar' or 'triangle', it represents the shape of all dimensions of the kernel.

        kernel: {'gaussian', 'boxcar', 'triangle'}
            Type of smoothing filter to be used.

        Returns
        -------
        smoothed_signal: `Signal`
            Smoothed signal.
        """
        if kernel == "gaussian":
            xf = ndimage.gaussian_filter(self.values, sigma=width, **kwargs)
        elif kernel == "boxcar":
            if width % 2 == 0:
                weight = np.ones((width + 1,) * self.ndim) / width ** self.ndim
                edges = [slice(None)] * self.ndim
                for i in range(self.ndim):
                    edges[i] = [0, -1]
                    weight[tuple(edges)] /= 2
                    edges[i] = slice(None)
            else:
                weight = np.ones((width,) * self.ndim) / width ** self.ndim
            xf = self.convolve(weight).values
        elif kernel == "triangle":
            half = int(width // 2)
            weight = np.array(list(range(1, half + 2)) + list(range(half, 0, -1)))
            for i in range(self.ndim - 1):
                weight = weight + weight.reshape(weight.shape + (1,)) - 1
            weight = weight / weight.sum()
            xf = self.convolve(weight).values
        else:
            raise ValueError(f"Kernel type '{kernel}'' is unknown.")
        smoothed_signal = self._replace_data(xf)
        return smoothed_signal

    def convolve(self, kernel):
        """FIR filtering.

        Parameters
        ----------
        kernel: array-like
            Any FIR filter passed as an array of samples to be convolved.

        Returns
        -------
        result: `Signal`
            The result of convolution with `kernel`.
        """
        xf = ndimage.convolve(self.values, kernel, mode="mirror")
        result = self._replace_data(xf)
        return result


class TSeries(Signal):
    def __init__(self, time=None, values=None, assume_sorted=False):
        if time is None:
            time = np.arange(len(values))
        if values is None:
            values = np.ones(len(time))
        data = xr.Variable("time", values)
        if not isinstance(time, xr.IndexVariable):
            time = xr.IndexVariable("time", time)
        if time.size != data.size:
            raise ValueError("Input arrays have incompatible lengths.")
        time = dict(time=time)
        super().__init__(data, time, fastpath=True)
        if (
            not assume_sorted
            and not self.coords["time"]._data.array.is_monotonic_increasing
        ):
            self._array = self._array.sortby("time")

    @property
    def time(self):
        return self._array.time.values

    def __str__(self):
        return str(self._array).replace("xarray.DataArray", "TSeries")

    def __repr__(self):
        return repr(self._array).replace("xarray.DataArray", "TSeries")

    def __getitem__(self, key):
        time = self.coords["time"][key]
        values = self.values[key]
        if values.ndim < 1:
            return values.item()
        return TSeries(time, values)

    def from_xray(self, xray, assume_sorted=False):
        if xray.ndim == 0:
            return xray.item()
        elif xray.ndim == 1:
            ts = TSeries(xray._coords["time"], xray.values, assume_sorted)
            ts.attrs.update(xray.attrs)
            return ts

    @property
    def baseline(self):
        return self.time[-1] - self.time[0]

    @property
    def median_dt(self):
        return np.median(np.diff(self.time))

    @property
    def dt(self):
        if np.allclose(np.diff(self.time), self.median_dt):
            return self.median_dt
        raise AttributeError(
            "The sampling period is only strictly defined for "
            "uniformly sampled signals. Use median_dt for a median value."
        )

    def tmax(self):
        return self.max().time.item()

    @property
    def derivative(self):
        return self.from_xray(self._array.differentiate(coord="time"))

    @property
    def TEO(self):
        """Teager Energy Operator (TEO)

        J. F. Kaiser, "On Teager’s energy algorithm and its generalization to
        continuous signals", Proc. 4th IEEE Signal Processing Workshop, 1990.
        """
        return self.derivative * self.derivative - self * self.derivative.derivative

    def timeshift(self, t0):
        return TSeries(self.coords["time"] + t0, self.values)

    def timescale(self, alpha):
        return TSeries(self.coords["time"] * alpha, self.values)

    def fold(self, period, t0=0):
        return TSeries(((self.coords["time"] - t0) / period) % 1, self.values)

    def fft(self, oversample=1.0, dt=None):
        nfft = int(oversample * self.size)
        if dt is None:
            dt = self.dt
        freqs = np.fft.rfftfreq(n=nfft, d=dt)
        coefs = np.fft.rfft(self.values, n=nfft)
        return FSeries(freqs, coefs)

    def psd(self, *args, **kwargs):
        return np.square(np.abs(self.fft(*args, **kwargs)))

    def dropna(self):
        return self.from_xray(self._array.dropna("time"))

    def cov(self, other):
        return self.to_pandas().cov(other.to_pandas())

    def corr(self, other):
        return self.to_pandas().corr(other.to_pandas())

    def polyfit(self, degree):
        coefs = np.polyfit(self.time, self.values, degree)
        fit = self._replace_data(np.poly1d(coefs)(self.time))
        fit.attrs.update(coefficients=coefs)
        return fit

    def curvefit(self, fun, **kwargs):
        popt, pcov = optimize.curve_fit(fun, self.time, self.values, **kwargs)
        fit = self._replace_data(fun(self.time, *popt))
        fit.attrs.update(coefficients=popt, covariance=pcov)
        return fit

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
        acf: `TSeries`
            ACF of input signal.
        """
        if max_lag is None:
            max_lag = self.size // 2
        lags = self.time - self.time.min()
        if isinstance(max_lag, float):
            max_lag = np.searchsorted(lags, max_lag) + 1
        max_lag = min(max_lag, self.size)
        ryy = (self - self.mean()).psd(oversample=2.0, dt=self.median_dt).ifft()
        if unbias:
            correction = (self / self).psd(oversample=2.0, dt=self.median_dt).ifft()
            ryy /= correction
        ryy = ryy[:max_lag] / ryy[0]
        return TSeries(lags[:max_lag], ryy.values, assume_sorted=True)

    def join(self, other, **kwargs):
        if len(np.intersect1d(self.time, other.time) > 0):
            warnings.warn(
                "There are overlapping timestamps. The corresponding "
                "timestamps in the returned TSeries have both samples."
            )
        new_array = xr.concat([self._array, other._array], "time", **kwargs)
        return self.from_xray(new_array)

    def split(self, max_gap=None):
        if max_gap is None:
            max_gap = 1.5 * self.median_dt
        ids = np.where(np.diff(self.time) > max_gap)[0]
        ids = np.hstack([0, ids + 1, self.size])
        splits = []
        for i in range(len(ids) - 1):
            splits.append(self[ids[i] : ids[i + 1]])
        return splits

    def downsample(self, dt, func=np.nanmean):
        labels = np.arange(self.time.min(), self.time.max(), dt)
        binned_array = self._array.groupby_bins(
            "time", bins=labels.size, labels=labels
        ).reduce(func)
        return TSeries(labels, binned_array.values).dropna()

    def interp(self, new_time=None, method="linear", **kwargs):
        """Interpolation onto a new time grid.

        Parameters
        ----------
        new_time: ndarray, optional
            Sampling grid. If omitted, it will be uniform with period `median_dt`.
        method: {'linear', 'spline', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'}
            Interpolation method to be used.
        s: float, optional
            Smoothing condition for the spline. Ignored unless method == "spline".

        Returns
        -------
        uniform_signal: `Signal`
            Interpolated signal.
        """
        if new_time is None:
            new_time = np.arange(np.min(self.time), np.max(self.time), self.median_dt)
        if method == "spline":
            tck = interpolate.splrep(self.time, self.values, **kwargs)
            new_values = interpolate.splev(new_time, tck)
            new_series = TSeries(new_time, new_values)
        else:
            result = self._array.interp(time=new_time, method=method, **kwargs)
            new_series = self.from_xray(result)
        return new_series

    def interpolate_na(self, method="linear", **kwargs):
        if method == "constant":
            k = kwargs.pop("k", 0.0)
            result = self._array.fillna(k)
        elif method == "bfill":
            result = self._array.bfill("time")
        elif method == "ffill":
            result = self._array.ffill("time")
        elif method == "random":
            mu = kwargs.pop("mu", self.mean())
            sd = kwargs.pop("sd", self.estimate_noise())
            random_seed = kwargs.pop("random_seed", None)
            rng = np.random.default_rng(random_seed)
            size = self.size - self.count()
            result = self._array.copy()
            result.values[result.isnull()] = rng.normal(mu, sd, size)
        elif method == "mirror":
            result = self._array.copy()
            ids = np.where(np.diff(self.isnull().values))[0] + 1
            n_gaps = ids.size // 2
            for i in range(n_gaps):
                start = ids[2 * i]
                end = ids[2 * i + 1]
                gap_size = end - start
                left_ids = np.arange(start, start + gap_size // 2)
                right_ids = np.arange(end - gap_size // 2, end)
                l_vals = 2 * start - left_ids - 1
                r_vals = 2 * end - right_ids - 1
                result.values[left_ids] = result.values[l_vals]
                result.values[right_ids] = result.values[r_vals]
                if gap_size % 2 == 1:
                    center = (start + end - 1) // 2
                    result.values[center] = 0.5 * (
                        result.values[center - 1] + result.values[center + 1]
                    )
        else:
            result = self._array.interpolate_na("time", method=method, **kwargs)
        return self.from_xray(result)

    def fill_gaps(self, dt=None, **kwargs):
        if dt is None:
            dt = self.median_dt
        t_new = [self.time[0]]
        for t in self.time[1:]:
            prevtime = t_new[-1]
            while (t - prevtime) > 1.2 * dt:
                t_new.append(prevtime + dt)
                prevtime = t_new[-1]
            t_new.append(t)
        t_new = np.array(t_new)[~np.isin(t_new, self.time)]
        result = self.join(TSeries(t_new, np.full_like(t_new, np.nan)))
        return result.interpolate_na(**kwargs)

    def drop(self, index=None):
        if index is None:
            index = []
        return TSeries(
            np.delete(self.coords["time"], index),
            np.delete(self.values, index),
            assume_sorted=True,
        )

    def pad(self, pad_width, **kwargs):
        time_kwargs = {}
        data_kwargs = {}
        for key, arg in kwargs.items():
            arg = np.asarray(arg)
            if np.size(arg) == 1:
                time_kwargs[key] = arg.item()
                data_kwargs[key] = arg.item()
            else:
                time_kwargs[key] = arg[0]
                data_kwargs[key] = arg[1]
        t_new = np.pad(self.time, pad_width, **time_kwargs)
        v_new = np.pad(self.values, pad_width, **data_kwargs)
        return TSeries(t_new, v_new)

    def get_envelope(self, pad_width=0, **peak_kwargs):
        """Interpolates maxima/minima with cubic splines into upper/lower envelopes.

        Parameters
        ----------
        peak_kwargs: float, optional
            Keyword arguments to use in `find_extrema`.
        pad_width: int, optional
            Number of extrema to repeat on either side of the signal.

        Returns
        -------
        upper: `Signal`
            Upper envelope.
        lower: `Signal`
            Lower envelope.
        """
        peaks = self.find_peaks(include_edges=True, **peak_kwargs)
        dips = self.find_dips(include_edges=True, **peak_kwargs)
        if peaks.size < (2 + pad_width) or dips.size < (2 + pad_width):
            raise ValueError("Signal doesn't have enough extrema for padding.")
        peaks = peaks.pad(pad_width, mode="reflect", reflect_type=["odd", None]).drop(
            [pad_width, -pad_width - 1]
        )
        dips = dips.pad(pad_width, mode="reflect", reflect_type=["odd", None]).drop(
            [pad_width, -pad_width - 1]
        )
        if peaks.size < 4 or dips.size < 4:
            raise ValueError(
                "Signal doesn't have enough extrema for envelope interpolation."
            )
        upper = peaks.interp(new_time=self.time, method="spline")
        lower = dips.interp(new_time=self.time, method="spline")
        return upper, lower

    def butterworth(self, fmin=None, fmax=None, order=5):
        """Implements a IIR butterworth filter.

        Parameters
        ----------
        fmin: float
            Lower cutoff frequency.
        fmax: float
            Higher cutoff frequency.
        order: int, optional
            Order of the butterworth filter. Default is 5.

        Returns
        -------
        filt_signal: `TSeries`
            Filtered signal.
        """
        nyq = 0.5 / self.median_dt
        if fmin is not None and fmax is None:
            Wn = fmin / nyq
            btype = "highpass"
        elif fmin is None and fmax is not None:
            Wn = fmax / nyq
            btype = "lowpass"
        elif fmin is not None and fmax is not None:
            Wn = [fmin / nyq, fmax / nyq]
            btype = "bandpass"
        else:
            raise ValueError("At least one of 'fmin' and 'fmax' must be given!")
        sos = signal.butter(N=order, Wn=Wn, btype=btype, output="sos")
        filt_signal = self._replace_data(signal.sosfiltfilt(sos, self.values))
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
        ml = np.searchsorted(self.time - self.time[0], 2 * p_max)
        rxx = self.butterworth(1 / p_max, 1 / p_min).acf(max_lag=ml)
        if p_max >= 20:
            width = int(p_max // 10)
            rxx = rxx.smooth(width, kernel="boxcar")
            rxx /= rxx.amax()
        peaks = rxx.find_peaks()
        best_per = peaks.time[peaks.attrs["prominences"].argmax()]
        height = peaks.attrs["prominences"].max()
        tau_max = 20 * p_max / best_per

        def rss(params):
            log_aa, log_tt = params
            aa = np.exp(log_aa)
            tt = np.exp(log_tt)
            acf_model = (
                aa * np.exp(-rxx.time / tt) * np.cos(2 * np.pi * rxx.time / best_per)
            )
            return np.sum(np.square(rxx - acf_model))

        results = optimize.minimize(fun=rss, x0=[0.0, np.log(best_per * 2)])
        log_amp, log_tau = results.x
        tau = min(np.exp(log_tau), tau_max)
        quality = (tau / best_per) * (ml * height / rss([log_amp, np.log(tau)]))
        return best_per, height, quality

    def plot(self, *args, **kwargs):
        return self._array.plot.line(*args, **kwargs)


class FSeries(Signal):
    def __init__(self, frequency=None, values=None, assume_sorted=False):
        """
        frequency: array-like
            Frequency grid.
        values: array-like
            Periodogram samples.
        """
        if values is None:
            values = np.ones(len(frequency))
        data = xr.Variable("frequency", values)
        if not isinstance(frequency, xr.IndexVariable):
            frequency = xr.IndexVariable("frequency", frequency)
        if frequency.size != values.size:
            raise ValueError("Input arrays have incompatible lengths.")
        with np.errstate(divide="ignore", invalid="ignore"):
            frequency = dict(frequency=frequency, period=1.0 / frequency)
        super().__init__(data, frequency, fastpath=True)
        if (
            not assume_sorted
            and not self.coords["frequency"]._data.array.is_monotonic_increasing
        ):
            self._array = self._array.sortby("frequency")

    @property
    def frequency(self):
        return self._array.frequency.values

    @property
    def period(self):
        return self._array.period.values

    def __str__(self):
        return str(self._array).replace("xarray.DataArray", "FSeries")

    def __repr__(self):
        return repr(self._array).replace("xarray.DataArray", "FSeries")

    def __getitem__(self, key):
        frequency = self.coords["frequency"][key]
        values = self.values[key]
        if values.ndim < 1:
            return values.item()
        return FSeries(frequency, values)

    def from_xray(self, xray):
        if xray.ndim == 0:
            return xray.item()
        elif xray.ndim == 1:
            fs = FSeries(xray._coords["frequency"], xray.values)
            fs.attrs.update(xray.attrs)
            return fs

    @property
    def median_df(self):
        return np.median(np.diff(self.frequency))

    @property
    def df(self):
        if np.allclose(np.diff(self.frequency), self.median_df):
            return self.median_df
        raise AttributeError(
            "The sampling period is only strictly defined for "
            "uniform frequency grids. Use median_df for a median value."
        )

    @property
    def median_dp(self):
        return -np.median(np.diff(self.period))

    @property
    def dp(self):
        if np.allclose(np.diff(self.period), self.median_dp):
            return self.median_dp
        raise AttributeError(
            "The sampling period is only strictly defined for "
            "uniform period grids. Use median_dp for a median value."
        )

    def fmax(self):
        return self.max().frequency.item()

    def pmax(self):
        return self.max().period.item()

    def psort_by_peak(self):
        peaks = self.find_peaks()
        return peaks.period[peaks.values.argsort()[::-1]]

    def psort_by_prominence(self):
        peaks = self.find_peaks()
        return peaks.period[peaks.attrs["prominences"].argsort()[::-1]]

    @property
    def period_at_highest_peak(self):
        peaks = self.find_peaks()
        return peaks.pmax()

    @property
    def period_at_highest_prominence(self):
        peaks = self.find_peaks()
        prominences = peaks.attrs["prominences"]
        return peaks.period[np.nanargmax(prominences)]

    def periods_at_half_max(self, peak_order=1, use_prominence=False):
        peaks = self.find_peaks()
        indices = peaks.attrs["indices"]
        if use_prominence:
            heights = peaks.attrs["prominences"]
        else:
            heights = peaks.values
        jmax = heights.argsort()[-peak_order]
        idmax = indices[jmax]
        height = heights[jmax]
        half = self[idmax] - height / 2
        hi = (self[:idmax] - half).find_zero_crossings()[-1]
        lo = (self[idmax:] - half).find_zero_crossings()[0]
        upper = self[:idmax].period[hi]
        lower = self[idmax:].period[lo]
        return lower, upper

    def ifft(self, nfft=None):
        coefs = np.fft.irfft(self.values, n=nfft)
        dt = 1 / (coefs.size * self.df)
        time = np.arange(coefs.size) * dt
        return TSeries(time, coefs)

    def dropna(self):
        return self.from_xray(self._array.dropna("frequency"))

    def polyfit(self, degree, use_period=False):
        if use_period:
            xdata = self.period
        else:
            xdata = self.frequency
        coefs = np.polyfit(xdata, self.values, degree)
        fit = self._replace_data(np.poly1d(coefs)(xdata))
        fit.attrs.update(coefficients=coefs)
        return fit

    def curvefit(self, fun, use_period=False, **kwargs):
        if use_period:
            xdata = self.period
        else:
            xdata = self.frequency
        popt, pcov = optimize.curve_fit(fun, xdata, self.values, **kwargs)
        fit = self._replace_data(fun(xdata, *popt))
        fit.attrs.update(coefficients=popt, covariance=pcov)
        return fit

    def downsample(self, df=None, dp=None, func=np.nanmean):
        if df is None and dp is None:
            raise ValueError("At least one of df or dp must be given.")
        if df is not None and dp is not None:
            raise ValueError("Can't make a uniform grid at both frequency and period!")
        if df is not None:
            labels = np.arange(self.frequency.min(), self.frequency.max(), df)
            binned_array = self._array.groupby_bins(
                "frequency", bins=labels.size, labels=labels
            ).reduce(func)
        else:
            labels = 1.0 / np.arange(self.period.min(), self.period.max(), dp)
            binned_array = self._array.groupby_bins(
                "period", bins=labels.size, labels=1 / labels
            ).reduce(func)
        return FSeries(labels, binned_array.values).dropna()

    def plot(self, *args, **kwargs):
        return self._array.plot.line(*args, **kwargs)


class TFSeries(Signal):
    def __init__(self, time=None, frequency=None, values=None):
        """"""
        data = xr.Variable(("frequency", "time"), values)
        if not isinstance(time, xr.IndexVariable):
            time = xr.IndexVariable("time", time)
        if not isinstance(frequency, xr.IndexVariable):
            frequency = xr.IndexVariable("frequency", frequency)
        if time.size != values.shape[1] or frequency.size != values.shape[0]:
            raise ValueError("Input arrays have incompatible lengths.")
        with np.errstate(divide="ignore", invalid="ignore"):
            coords = dict(time=time, frequency=frequency, period=1.0 / frequency)
        super().__init__(data, coords, fastpath=True)

    @property
    def time(self):
        return self._array.time.values

    @property
    def frequency(self):
        return self._array.frequency.values

    @property
    def period(self):
        return self._array.period.values

    def __str__(self):
        return str(self._array).replace("xarray.DataArray", "TFSeries")

    def __repr__(self):
        return repr(self._array).replace("xarray.DataArray", "TFSeries")

    def __getitem__(self, key):
        new_key = xr.core.indexing.expanded_indexer(key, 2)
        k1, k2 = new_key
        frequency = self.coords["frequency"][k1]
        time = self.coords["time"][k2]
        values = self.values[new_key]
        if values.ndim < 1:
            return values.item()
        elif values.ndim == 1:
            if time.ndim == 0:
                return self._replace_data_and_dims(("frequency",), values)
            elif frequency.ndim == 0:
                return self._replace_data_and_dims(("time",), values)
        return TFSeries(time, frequency, values)

    def from_xray(self, xray):
        if xray.ndim == 0:
            return xray.item()
        elif xray.ndim == 1:
            if "time" in xray.dims:
                return TSeries(xray._coords["time"], xray.values)
            if "frequency" in xray.dims:
                return FSeries(xray._coords["frequency"], xray.values)
        elif xray.ndim == 2:
            tf = TFSeries(xray._coords["time"], xray._coords["frequency"], xray.values)
            tf.attrs.update(xray.attrs)
            return tf

    @property
    def median_dt(self):
        return np.median(np.diff(self.time))

    @property
    def dt(self):
        if np.allclose(np.diff(self.time), self.median_dt):
            return self.median_dt
        raise AttributeError(
            "The sampling period is only strictly defined for "
            "uniformly sampled signals. Use median_dt for a median value."
        )

    @property
    def median_df(self):
        return np.median(np.diff(self.frequency))

    @property
    def df(self):
        if np.allclose(np.diff(self.frequency), self.median_df):
            return self.median_df
        raise AttributeError(
            "The sampling period is only strictly defined for "
            "uniform frequency grids. Use median_df for a median value."
        )

    @property
    def median_dp(self):
        return -np.median(np.diff(self.period))

    @property
    def dp(self):
        if np.allclose(np.diff(self.period), self.median_dp):
            return self.median_dp
        raise AttributeError(
            "The sampling period is only strictly defined for "
            "uniform period grids. Use median_dp for a median value."
        )

    def downsample(self, dt=None, df=None, dp=None, func=np.nanmean):
        tlabels = self.time
        flabels = self.frequency
        binned_array = self._array.copy()
        if df is not None and dp is not None:
            raise ValueError("Can't make a uniform grid at both frequency and period!")
        if df is not None:
            flabels = np.arange(self.frequency.min(), self.frequency.max(), df)
            binned_array = (
                binned_array.groupby_bins(
                    "frequency", bins=flabels.size, labels=flabels
                )
                .reduce(func)
                .dropna("frequency_bins")
            )
            flabels = binned_array.frequency_bins.values
        if dp is not None:
            flabels = 1.0 / np.arange(self.period.min(), self.period.max(), dp)
            binned_array = (
                binned_array.groupby_bins(
                    "period", bins=flabels.size, labels=1.0 / flabels
                )
                .reduce(func)
                .dropna("period_bins")
            )
            flabels = 1.0 / binned_array.period_bins.values
        if dt is not None:
            tlabels = np.arange(self.time.min(), self.time.max(), dt)
            binned_array = (
                binned_array.groupby_bins("time", bins=tlabels.size, labels=tlabels)
                .reduce(func)
                .dropna("time_bins")
            )
            tlabels = binned_array.time_bins.values
        return TFSeries(time=tlabels, frequency=flabels, values=binned_array.values)

    def pcolormesh(self, *args, **kwargs):
        return self._array.plot.pcolormesh(*args, **kwargs)

    def imshow(self, *args, **kwargs):
        return self._array.plot.imshow(*args, **kwargs)

    def contour(self, *args, **kwargs):
        return self._array.plot.contour(*args, **kwargs)

    def contourf(self, *args, **kwargs):
        return self._array.plot.contourf(*args, **kwargs)

    def surface(self, *args, **kwargs):
        return self._array.plot.surface(*args, **kwargs)
