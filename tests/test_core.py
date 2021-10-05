import numpy as np
import pytest

from periodicity.core import TSeries


def test_time_array_is_always_sorted():
    sig = TSeries([3, 2, 1], [3, 5, 7])
    assert all(sig.time == [1, 2, 3])
    assert all(sig.values == [7, 5, 3])


def test_input_arrays_with_different_sizes():
    with pytest.raises(ValueError):
        _ = TSeries([1, 2], [1, 2, 3])


def test_dt_of_nonuniform_samples():
    sig = TSeries([1, 3, 4], [1, 1, 1])
    assert sig.median_dt == 1.5
    with pytest.raises(AttributeError):
        _ = sig.dt


def test_baseline():
    assert TSeries(np.arange(10)).baseline == 9


def test_nonuniform_slice_of_uniform_signal():
    sig = TSeries(np.arange(10))
    assert sig.dt == 1.0
    sig_slice = sig[[2, 5, 6]]
    with pytest.raises(AttributeError):
        _ = sig_slice.dt


def test_get_constant_envelope():
    t = np.linspace(0, 100, 1001)
    sig = TSeries(t, np.sin(t))
    upper1, lower1 = sig.get_envelope()
    assert np.abs(upper1 - 1).max() < 2e-3
    assert np.abs(lower1 + 1).max() < 2e-3
    upper2, lower2 = sig.get_envelope(pad_width=2)
    assert np.abs(upper2 - 1).max() < 2e-3
    assert np.abs(lower2 + 1).max() < 2e-3
    upper3, lower3 = sig.get_envelope(pad_width=10)
    assert np.allclose(upper2.values, upper3.values)
    assert np.allclose(lower2.values, lower3.values)


def test_teo_of_sine_wave():
    t = np.linspace(0, 100, 100_001)
    sig = TSeries(t, np.sin(t))
    teo = sig.TEO
    assert np.allclose(teo[:-2].values, 1.0)
