import numpy as np
from scipy import signal


def acf(y, t=None, maxlag=None, s=0, fill_gaps=False):
    """
    Auto-Correlation Function implemented using IFFT of the power spectrum.

    Parameters
    ----------
    y: array-like
        discrete input signal
    maxlag: int (optional)
        maximum lag to compute ACF
    s: int (optional)
        standard deviation of Gaussian filter used to smooth ACF, measured in samples

    Returns
    -------
    R: array-like
        ACF of input signal
    """
    N = len(y)
    if t is None:
        t = np.arange(N)
    if maxlag is None:
        maxlag = N
    if fill_gaps:
        T = float(np.median(np.diff(t)))
        gaps = np.where(np.diff(t) > 1.5 * T)[0]
        # TODO: fill gaps with linear interpolation
    f = np.fft.fft(y - y.mean(), n=2 * N)
    R = np.fft.ifft(f * np.conjugate(f))[:maxlag].real
    if s > 0:
        h = gaussian(m=0, s=s)
        h = h(np.arange(-(3 * s - 1), 3 * s, 1.))
        R = smooth(R, kernel=h)
    R /= R[0]
    return R


def find_peaks(R, lags):
    """
    Finds ACF maxima and the corresponding peak heights

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


def gaussian(m, s):
    """
    Simple 1D Gaussian function generator

    Parameters
    ----------
    m: float
        mean
    s: float
        standard deviation

    Returns
    -------
    f: function
        1D Gaussian with given parameters
    """
    def f(x):
        return 1 / (np.sqrt(2 * np.pi) * s) * np.exp(-.5 * ((x - m) / s) ** 2)

    return f


def smooth(y, kernel):
    """
    Wrap to numpy.convolve

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
    """
    Implements a band-pass IIR butterworth filter

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
