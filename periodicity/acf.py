import numpy as np
from scipy import signal
from scipy.optimize import least_squares
from astropy.convolution import Box1DKernel


def acf(y, t=None, maxlag=None, s=0, fill=False):
    """Auto-Correlation Function implemented using IFFT of the power spectrum.

    Parameters
    ----------
    y: array-like
        discrete input signal
    t: array-like (optional)
        time array
    maxlag: int (optional)
        maximum lag to compute ACF
        # TODO: turn maxlag into a measure of time if t is given
    s: int (optional)
        standard deviation of Gaussian filter used to smooth ACF, measured in samples
    fill: bool (optional default=False)
        whether to use linear interpolation to sample signal uniformly

    Returns
    -------
    lags: array-like
        array of lags
    R: array-like
        ACF of input signal
    """
    if t is None:
        t = np.arange(len(y))

    if fill:
        t, y = fill_gaps(t, y)

    N = len(y)

    if maxlag is None:
        maxlag = N

    f = np.fft.fft(y - y.mean(), n=2 * N)
    R = np.fft.ifft(f * np.conjugate(f))[:maxlag].real

    if s > 0:
        kernel = gaussian(mu=0, sd=s)
        h = kernel(np.arange(-(3 * s - 1), 3 * s, 1.))
        R = smooth(R, kernel=h)

    R /= R[0]
    lags = t[:maxlag] - np.min(t[:maxlag])

    return lags, R


def fill_gaps(t, y):
    """Linear interpolation to create a uniformly sampled signal

    Parameters
    ----------
    t: array-like
        time array
    y: array-like
        signal array

    Returns
    -------
    tnew: array-like
        new sampling times uniformly spaced
    ynew: array-like
        signal with gaps filled by linear interpolation
    """
    T = float(np.median(np.diff(t)))
    gaps = np.where(np.diff(t) > 1.5 * T)[0]
    t_gaps = []
    y_gaps = []
    tnew = t
    ynew = y
    for g in gaps:
        t0, t1 = tnew[g:g + 2]
        y0, y1 = ynew[g:g + 2]
        tfill = np.arange(t0 + T, t1, T)
        t_gaps.append(tfill)
        y_gaps.append(y0 + (tfill - t0) * (y1 - y0) / (t1 - t0))
    ids = []
    shift = 1
    for i, tg, yg in zip(gaps, t_gaps, y_gaps):
        idx = i + shift
        tnew = np.insert(tnew, idx, tg)
        ynew = np.insert(ynew, idx, yg)
        n = len(tg)
        ids.append(np.arange(idx, idx + n))
        shift += n
    tnew = np.arange(tnew.size) * T + tnew[0]
    return tnew, ynew


def find_peaks(R, lags):
    """Finds ACF maxima and the corresponding peak heights

    Parameters
    ----------
    R: array-like
        ACF array
    lags: array-like
        time array

    Returns
    -------
    peaks: array-like
        mask with indices of peaks
    heights: array-like
        peak heights for each peak found
    """
    peaks = np.array([i for i in range(1, len(lags) - 1) if R[i - 1] < R[i] and R[i + 1] < R[i]])
    dips = np.array([i for i in range(1, len(lags) - 1) if R[i - 1] > R[i] and R[i + 1] > R[i]])
    if lags[dips[0]] > lags[peaks[0]]:
        peaks = peaks[1:]
    # TODO: Calculate actual peak heights
    heights = R[peaks]
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


def acf_harmonic_quality(t, x, pmin=None, periods=None):
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
        periods = 2 ** np.arange(8)
    fs = 1 / float(np.median(np.diff(t)))
    if pmin is None:
        pmin = max(np.min(periods) / 10, 2 / fs)
    t -= np.min(t)
    periods = np.array([pi for pi in periods if pmin < pi < np.max(t) / 2])
    ps = []
    hs = []
    qs = []
    for pi in periods:
        xf = filt(x, 1/pi, 1/pmin, fs)
        ml = np.where(t >= 2*pi)[0][0]
        lgs, R = acf(xf, t, maxlag=ml)
        if pi >= 20:
            R = smooth(R, Box1DKernel(width=pi//10))
        try:
            peaks, heights = find_peaks(R, lgs)
        except IndexError:
            continue
        bp_acf = lgs[peaks][np.argmax(heights)]
        ps.append(bp_acf)
        hs.append(np.max(heights))
        tau_max = 20 * pi / bp_acf

        def eps(params, lags, r):
            acf_model = params[0] * np.exp(-lags / params[1]) * np.cos(2 * np.pi * lags / bp_acf)
            if params[1] > tau_max:
                return np.ones_like(r) * 1000
            return np.square(r - acf_model)

        results = least_squares(fun=eps, x0=[1, 1], loss='soft_l1', f_scale=0.1, args=(lgs, R))
        A, tau = results.x
        ri = eps(results.x, lgs, R).sum()
        qs.append((tau / ps[-1]) * (ml * hs[-1] / ri))

    return ps, hs, qs
