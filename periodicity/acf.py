import numpy as np
from scipy import signal


def acf(y, maxlag=None, s=0):
    N = len(y)
    if maxlag is None:
        maxlag = N
    f = np.fft.fft(y - y.mean(), n=2 * N)
    R = np.fft.ifft(f * np.conjugate(f))[:maxlag].real
    if s > 0:
        h = gaussian(m=0, s=s)
        h = h(np.arange(-(3 * s - 1), 3 * s, 1.))
        R = smooth(R, kernel=h)
    R /= R[0]
    return R


def find_peaks(R, lags):
    peaks = np.array([i for i in range(1, len(lags) - 1) if R[i - 1] < R[i] and R[i + 1] < R[i]])
    dips = np.array([i for i in range(1, len(lags) - 1) if R[i - 1] > R[i] and R[i + 1] > R[i]])
    if lags[dips[0]] > lags[peaks[0]]:
        peaks = peaks[1:]
    # TODO: Calculate actual peak heights
    heights = R[peaks]
    return peaks, heights


def gaussian(m, s):
    def f(x):
        return 1 / (np.sqrt(2 * np.pi) * s) * np.exp(-.5 * ((x - m) / s) ** 2)

    return f


def smooth(y, kernel):
    double_y = np.append(y[::-1], y)
    return np.convolve(double_y, kernel, mode='same')[len(y):]


def filt(x, lo, hi, fs, order=5):
    nyq = .5 * fs
    lo /= nyq
    hi /= nyq
    b, a = signal.butter(order, [lo, hi], btype='band')
    xf = signal.filtfilt(b, a, x)
    return xf
