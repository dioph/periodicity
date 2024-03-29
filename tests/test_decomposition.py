import numpy as np

from periodicity.core import TSeries
from periodicity.data import SustainedPlusGappedPureTones
from periodicity.decomposition import CEEMDAN


def test_two_tones_two_imfs():
    # Test if nothing but the two tones are recovered by CEEMDAN
    x = TSeries(values=SustainedPlusGappedPureTones())
    imfs = CEEMDAN(ensemble_size=50, random_seed=42)(x)
    assert len(imfs) == 2
    # Test if the residual noise in the first mode is close to zero
    left_mse = np.mean(np.square(imfs[0][11:490]))
    right_mse = np.mean(np.square(imfs[0][761:990]))
    assert left_mse < 1e-4
    assert right_mse < 1e-4
    # Test the closeness between the original tones and the recovered IMFs
    s2 = np.sin(2 * np.pi * 0.065 * np.arange(1000))
    s1 = np.zeros_like(s2)
    s1[500:750] += np.sin(2 * np.pi * 0.255 * np.arange(250))
    err1 = (imfs[0] - s1).values[3:-3]
    err2 = (imfs[1] - s2).values[3:-3]
    err = (sum(imfs) - x).values
    rrse_1 = np.linalg.norm(err1) / np.linalg.norm(s1[3:-3])
    rrse_2 = np.linalg.norm(err2) / np.linalg.norm(s2[3:-3])
    rrse_x = np.linalg.norm(err) / np.linalg.norm(x.values)
    assert rrse_1 < 0.10
    assert rrse_2 < 0.05
    assert rrse_x < 1e-16
