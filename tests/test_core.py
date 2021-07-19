import numpy as np
import pytest

from periodicity.core import Timeseries


def test_time_array_is_always_sorted():
    sig = Timeseries([3, 2, 1], [3, 5, 7])
    assert all(sig.time == [1, 2, 3])
    assert all(sig.val == [7, 5, 3])


def test_input_arrays_with_different_sizes():
    with pytest.raises(ValueError):
        _ = Timeseries([1, 2], [1, 2, 3])


def test_ts_of_nonuniform_samples():
    sig = Timeseries([1, 3, 4], [1, 1, 1])
    assert sig.median_ts == 1.5
    with pytest.raises(AttributeError):
        _ = sig.ts


def test_baseline():
    assert Timeseries(np.arange(10), np.ones(10)).baseline == 9


def test_nonuniform_slice_of_uniform_signal():
    sig = Timeseries(np.arange(10), np.ones(10))
    assert sig.ts == 1.0
    sig_slice = sig[[2, 5, 6]]
    with pytest.raises(AttributeError):
        _ = sig_slice.ts


def test_get_constant_envelope():
    t = np.linspace(0, 100, 1001)
    sig = Timeseries(t, np.sin(t))
    upper1, lower1 = sig.get_envelope()
    assert np.abs(upper1.val - 1).max() < 2e-3
    assert np.abs(lower1.val + 1).max() < 2e-3
    upper2, lower2 = sig.get_envelope(n_rep=2)
    assert np.abs(upper2.val - 1).max() < 2e-3
    assert np.abs(lower2.val + 1).max() < 2e-3
    upper3, lower3 = sig.get_envelope(n_rep=10)
    assert np.allclose(upper2.val, upper3.val)
    assert np.allclose(lower2.val, lower3.val)


def test_teo_of_sine_wave():
    t = np.linspace(0, 100, 100_001)
    sig = Timeseries(t, np.sin(t))
    teo = sig.TEO
    assert np.allclose(teo[:-2].val, 1.0)
