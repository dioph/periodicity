import numpy as np

from periodicity.core import Timeseries
from periodicity.spectral import GLS


def test_gls_default_frequency_grid():
    t0 = 2.5
    ts = 0.1
    fs = 1 / ts
    f0 = 1 / t0
    rng = np.random.default_rng()
    time = np.arange(0, t0 + ts, ts)
    signal = Timeseries(time, rng.normal(1, 0.1, time.size))
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
    sine = Timeseries(val=np.sin((np.arange(100) / 100) * 20 * np.pi))
    gls = GLS()
    ls = gls(sine)
    best_period = 1 / ls.frequency[ls.val.argmax()]
    assert best_period == 10.0
