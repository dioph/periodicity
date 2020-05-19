import numpy as np
from astropy.convolution import Box1DKernel
from scipy import interpolate, signal
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize



def acf(y, t=None, max_lag=None, s=0, fill=False):
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

    if max_lag is None:
        max_lag = n

    if type(max_lag) is float:
        max_lag = np.where(t - np.min(t) <= max_lag)[0][-1] + 1

    f = np.fft.fft(y - y.mean(), n=2 * n)
    ryy = np.fft.ifft(f * np.conjugate(f))[:max_lag].real

    if s > 0:
        ryy = gaussian_filter1d(ryy, sigma=s, truncate=3.0)

    ryy /= ryy[0]
    lags = t[:max_lag] - np.min(t[:max_lag])

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
    t_new = np.arange(np.min(t), np.max(t), ts)
    y_new = interpolate.interp1d(t, y)(t_new)
    return t_new, y_new


def find_peaks(y, t=None, delta=0.):
    """Finds local maxima and corresponding peak prominences

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
    if t is None:
        t = np.arange(len(y))

    maxima, res = signal.find_peaks(y, prominence=delta)
    peaks = np.array([t[maxima], y[maxima]]).T
    heights = res['prominences']

    if heights.size == 0 and delta > 1e-6:
        return find_peaks(y=y, t=t, delta=delta / 2)

    return peaks, heights


def find_extrema(y, delta=0.):
    """Finds local extrema

    Parameters
    ----------
    y: array-like
        signal array
    delta: float, optional
         minimum peak prominence

    Returns
    -------
    peaks: array-like
        maxima indices
    dips: array-like
        minima indices
    """
    maxima, _ = signal.find_peaks(y, prominence=delta)
    minima, _ = signal.find_peaks(-y, prominence=delta)
    return maxima, minima


def find_zero_crossings(y, height=None, delta=0.):
    """Finds zero crossing indices

    Parameters
    ----------
    y: array-like
        signal
    height: float, optional
        maximum deviation from zero
    delta: float, optional
        prominence used in `scipy.signal.find_peaks` when `height` is specified.

    Returns
    -------
    indzer: array-like
        zero-crossing indices
    """
    if height is None:
        ind_zer, = np.where(np.diff(np.signbit(y)))
    else:
        ind_zer, _ = signal.find_peaks(-np.abs(y), height=-height,
                                       prominence=delta)
    return ind_zer


def get_envelope(y, t=None, delta=0., n_rep=0):
    """Interpolates maxima/minima with cubic splines into upper/lower envelopes.

    Parameters
    ----------
    y: array-like
        signal
    t: array-like, optional
        signal timestamps
    delta: float, optional
        prominence to use in `find_extrema`
    nbsym: int, optional
        number of extrema to repeat on either side of the signal

    Returns
    -------
    upper: array-like
        upper envelope
    lower: array-like
        lower envelope
    """
    if t is None:
        t = np.arange(len(y))

    peaks, dips = find_extrema(y, delta)
    if n_rep == 0:
        peaks = np.r_[0, peaks, len(y) - 1]
        t_max = t[peaks]
        y_max = y[peaks]
        dips = np.r_[0, peaks, len(y) - 1]
        t_min = t[dips]
        y_min = y[dips]
    else:
        l_peaks = peaks[:n_rep][::-1]
        r_peaks = peaks[-n_rep:][::-1]
        l_off = 2 * t[0] - t[l_peaks]
        r_off = 2 * t[-1] - t[r_peaks]
        t_max = np.r_[l_off, t[peaks], r_off]
        y_max = np.r_[y[l_peaks], y[peaks], y[r_peaks]]
        l_dips = dips[:n_rep][::-1]
        r_dips = dips[-n_rep:][::-1]
        l_off = 2 * t[0] - t[l_dips]
        r_off = 2 * t[-1] - t[r_dips]
        t_min = np.r_[l_off, t[dips], r_off]
        y_min = np.r_[y[l_dips], y[dips], y[r_dips]]

    tck = interpolate.splrep(t_max, y_max)
    upper = interpolate.splev(t, tck)
    tck = interpolate.splrep(t_min, y_min)
    lower = interpolate.splev(t, tck)
    return upper, lower


def get_noise(y, sigma=3., niter=3):
    """Finds the standard deviation of a white gaussian noise in the data

    Parameters
    ----------
    y: array-like
        signal array
    sigma: float, optional (default=3.0)
        sigma_clip value
    niter: int, optional (default=3)
        number of iterations for k-sigma clipping

    Returns
    -------
    noise: float
        estimate of standard deviation of the noise
    """
    residue = y - signal.medfilt(y, 3)
    sd = np.std(residue)
    index = np.arange(residue.size)
    for i in range(niter):
        mu = np.mean(residue[index])
        sd = np.std(residue[index])
        index, = np.where(np.abs(residue - mu) < sigma * sd)
    noise = sd / .893421
    return noise


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
    def pdf(x):
        z = (x - mu) / sd
        return np.exp(-z * z / 2.0) / np.sqrt(2.0 * np.pi) / sd
    return pdf


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
    w = kernel.shape[0]
    s = np.r_[y[w - 1:0:-1], y, y[-2:-w - 1:-1]]
    sf = np.convolve(s, kernel, mode='valid')
    yf = sf[w // 2 - 1:-w // 2]
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


def acf_harmonic_quality(t, x, p_min=None, periods=None, a=1, b=2, n=8):
    """Calculates the ACF quality of band-pass filtered versions of the signal.

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
    if p_min is None:
        p_min = max(np.min(periods) / 10, 3 / fs)
    t -= np.min(t)
    periods = np.array([pi for pi in periods if p_min < pi < np.max(t) / 2])
    ps = []
    hs = []
    qs = []
    for p_max in periods:
        xf = filt(x, 1 / p_max, 1 / p_min, fs)
        ml = np.where(t >= 2 * p_max)[0][0]
        lags, rxx = acf(xf, t, max_lag=ml)
        if p_max >= 20:
            rxx = smooth(rxx, Box1DKernel(width=p_max // 10))
        try:
            peaks, heights = find_peaks(rxx, lags)
            per = peaks[np.argmax(heights)][0]
        except Exception:
            continue
        ps.append(per)
        hs.append(np.max(heights))
        tau_max = 20 * p_max / per

        def rss(params):
            aa, tt = params
            if tt < 0:
                return 1e300
            acf_model = aa * np.exp(-lags / tt) * np.cos(2 * np.pi * lags / per)
            return np.sum(np.square(rxx - acf_model))

        results = minimize(fun=rss, x0=np.array([1., per * 2]))
        amp, tau = results.x
        tau = min(tau, tau_max)
        ri = rss(results.x)
        qs.append((tau / per) * (ml * hs[-1] / ri))

    return ps, hs, qs
