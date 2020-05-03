import numpy as np
from astropy.convolution import Box1DKernel
from scipy import interpolate, signal
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize


def acf(y, t=None, maxlag=None, s=0, fill=False):
    """Auto-Correlation Function implemented using IFFT of the power spectrum.

    Parameters
    ----------
    y: array-like
        discrete input signal
    t: array-like (optional)
        time array
    maxlag: int or float (optional)
        Maximum lag to compute ACF. If given as a float, will be assumed to be a measure of time and the ACF will be
        computed for lags lower than or equal to `maxlag`.
    s: int (optional)
        standard deviation of Gaussian filter used to smooth ACF, measured in samples
    fill: bool (optional default=False)
        whether to use linear interpolation to sample signal uniformly

    Returns
    -------
    lags: array-like
        array of lags
    ryy: array-like
        ACF of input signal
    """
    if t is None:
        t = np.arange(len(y))

    if fill:
        t, y = fill_gaps(t, y)

    n = len(y)

    if maxlag is None:
        maxlag = n

    if type(maxlag) is float:
        maxlag = np.where(t - np.min(t) <= maxlag)[0][-1] + 1

    f = np.fft.fft(y - y.mean(), n=2 * n)
    ryy = np.fft.ifft(f * np.conjugate(f))[:maxlag].real

    if s > 0:
        ryy = gaussian_filter1d(ryy, sigma=s, truncate=3.0)

    ryy /= ryy[0]
    lags = t[:maxlag] - np.min(t[:maxlag])

    return lags, ryy


def fill_gaps(t, y, ts=None):
    """Linear interpolation to create a uniformly sampled signal

    Parameters
    ----------
    t: array-like
        signal timestamps
    y: array-like
        signal samples
    ts: float, optional
        sampling period; if timestamps are sorted, can be estimated if omitted.

    Returns
    -------
    tnew:
    ynew:
    """
    if ts is None:
        ts = float(np.median(np.diff(t)))
    tnew = np.arange(np.min(t), np.max(t), ts)
    ynew = interpolate.interp1d(t, y)(tnew)
    return tnew, ynew


def find_peaks(y, t=None, delta=0.):
    """Finds function maxima and the corresponding peak heights

    Parameters
    ----------
    y: array-like
        signal array
    t: array-like (optional)
        time array
        if not given will use indexes
    delta: float (optional)
        minimum difference between a peak and the following points before a peak may be considered a peak.
        default: 0.0
        recommended: delta >= RMSnoise * 5

    Returns
    -------
    peaks: array-like
        [tmax, ymax] for each maximum found
    heights: array-like
        average peak heights for each peak found
    """
    peaks = []
    dips = []
    if t is None:
        t = np.arange(len(y))
    y = np.asarray(y)
    assert len(t) == len(y), "t and y must have same length"

    mn, mx = np.inf, -np.inf
    mnpos, mxpos = np.nan, np.nan
    lookformax = False

    for i in range(len(y)):
        if y[i] > mx:
            mx = y[i]
            mxpos = t[i]
        if y[i] < mn:
            mn = y[i]
            mnpos = t[i]
        if lookformax:
            if y[i] < mx-delta:
                peaks.append((mxpos, mx))
                mn = y[i]
                mnpos = t[i]
                lookformax = False
        else:
            if y[i] > mn+delta and mn != -np.inf:
                dips.append((mnpos, mn))
                mx = y[i]
                mxpos = t[i]
                lookformax = True
    peaks = np.array(peaks)
    dips = np.array(dips)

    heights = []
    for i in range(len(peaks)):
        h1 = peaks[i, 1] - dips[i, 1]
        try:
            h2 = peaks[i, 1] - dips[i+1, 1]
            heights.append((h1+h2)/2)
        except IndexError:
            heights.append(h1)
    heights = np.array(heights)

    if heights.size == 0 and delta > 1e-6:
        return find_peaks(y=y, t=t, delta=delta/2)

    return peaks, heights


def gaussian(mu, sd):
    """Simple 1D Gaussian function generator

    Parameters
    ----------
    mu: float
        mean
    sd: float
        standard deviation

    Returns
    -------
    f: function
        1D Gaussian with given parameters
    """

    def f(x):
        return 1 / (np.sqrt(2 * np.pi) * sd) * np.exp(-.5 * ((x - mu) / sd) ** 2)

    return f


def smooth(y, kernel):
    """Wrap to numpy.convolve

    Parameters
    ----------
    y: array-like
        input noisy signal
    kernel: array-like
        FIR filter to smooth the signal
    Returns
    -------
    yf: array-like
        Smoothed signal
    """
    double_y = np.append(y[::-1], y)
    yf = np.convolve(double_y, kernel, mode='same')[len(y):]
    return yf


def filt(x, lo, hi, fs, order=5):
    """Implements a band-pass IIR butterworth filter

    Parameters
    ----------
    x: array-like
        input signal to be filtered
    lo: float
        lower cutoff frequency
    hi: float
        higher cutoff frequency
    fs: float
        sampling frequency of signal
    order: int (optional default=5)
        order of the butterworth filter

    Returns
    -------
    xf: array-like
        filtered signal
    """
    nyq = .5 * fs
    lo /= nyq
    hi /= nyq
    b, a = signal.butter(N=order, Wn=[lo, hi], btype='band')
    xf = signal.filtfilt(b, a, x)
    return xf


def acf_harmonic_quality(t, x, pmin=None, periods=None, a=1, b=2, n=8):
    """Calculates the quality of the ACF of a band-pass filtered version of the signal

    t: array-like
        time array
    x: array-like
        signal array
    pmin: float (optional)
        lower cutoff period to filter signal
    periods: list (optional)
        list of higher cutoff periods to filter signal
        Will only consider periods between `pmin` and half the baseline
    a, b, n: floats (optional)
        if `periods` is not given then it assumes the first `n` powers of `b` scaled by `a`:
            periods = a * b ** np.arange(n)
        defaults are a=1, b=2, n=8

    Returns
    -------
    ps: list
        highest peaks (best periods) for each filtered version
    hs: list
        maximum heights for each filtered version
    qs: list
        quality factor of each best period
    """
    if periods is None:
        periods = a * b ** np.arange(n)
    fs = 1 / float(np.median(np.diff(t)))
    if pmin is None:
        pmin = max(np.min(periods) / 10, 3 / fs)
    t -= np.min(t)
    periods = np.array([pi for pi in periods if pmin < pi < np.max(t) / 2])
    ps = []
    hs = []
    qs = []
    for pi in periods:
        xf = filt(x, 1 / pi, 1 / pmin, fs)
        ml = np.where(t >= 2 * pi)[0][0]
        lags, rxx = acf(xf, t, maxlag=ml)
        if pi >= 20:
            rxx = smooth(rxx, Box1DKernel(width=pi // 10))
        try:
            peaks, heights = find_peaks(rxx, lags)
            bp_acf = peaks[np.argmax(heights)][0]
        except:
            continue
        ps.append(bp_acf)
        hs.append(np.max(heights))
        tau_max = 20 * pi / bp_acf

        def eps(params):
            acf_model = params[0] * np.exp(-lags / params[1]) * np.cos(2 * np.pi * lags / bp_acf)
            return np.sum(np.square(rxx - acf_model))

        results = minimize(fun=eps, x0=np.array([1., bp_acf * 2]))
        amp, tau = results.x
        tau = min(tau, tau_max)
        ri = eps(results.x)
        qs.append((tau / bp_acf) * (ml * hs[-1] / ri))

    return ps, hs, qs
