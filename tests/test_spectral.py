import numpy as np

from periodicity.core import TSeries
from periodicity.spectral import GLS


def test_gls_default_frequency_grid():
    t0 = 2.5
    ts = 0.1
    fs = 1 / ts
    f0 = 1 / t0
    time = np.arange(0, t0 + ts, ts)
    signal = TSeries(time)
    gls = GLS(n=1)
    ls = gls(signal)
    freq = ls.frequency
    # frequencies are sorted
    assert sorted(freq) == list(freq)
    # minimum frequency corresponds to a half-cycle within the baseline
    assert freq[0] == f0 / 2
    # maximum frequency is half the sampling rate
    assert np.round(freq[-1], 6) == fs / 2
    # uniform grid with spacing equal to f0
    assert np.max(np.abs(np.diff(freq) - f0)) < 1e-10


def test_can_find_periods():
    sine = TSeries(values=np.sin((np.arange(100) / 100) * 20 * np.pi))
    gls = GLS()
    ls = gls(sine)
    assert ls.period_at_highest_peak == 10.0
