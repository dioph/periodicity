import numpy as np

from periodicity.data import SpottedStar
from periodicity.gp import FastGPModeler, StrongGPModeler, make_gaussian_prior


def test_make_gaussian_prior_spotted_lc():
    log_periods = np.linspace(-3, 5, 1000)
    t, y, dy = SpottedStar()
    prior = make_gaussian_prior(t, y)
    prior_prob = prior(log_periods)
    # prior has a maximum at approx 10.7 days
    assert prior_prob.argmax() == 671
    peaks = [
        i
        for i in range(1, len(log_periods) - 1)
        if prior_prob[i - 1] < prior_prob[i] and prior_prob[i + 1] < prior_prob[i]
    ]
    # prior has peaks at 0.4, 0.8, 1.7, 3.5, 5.6, 10.7 and 21.5 days
    assert len(peaks) == 7


def test_fastgp_constructor():
    model = FastGPModeler([1, 2], [3, 4])
    assert model.mu == (-17, -13, 0, 3)


def test_fastgp_spotted_lc_minimize():
    t, y, dy = SpottedStar()
    model = FastGPModeler(t, y)
    _, _, _, v = model.minimize()
    assert np.round(np.exp(v[4]), 3) == 10.618


def test_fastgp_spotted_lc_mcmc():
    np.random.seed(42)
    t, y, dy = SpottedStar()
    model = FastGPModeler(t, y)
    model.prior = make_gaussian_prior(t, y)
    model.minimize()
    samples = model.mcmc(n_walkers=16, n_steps=1000, burn=100)
    assert np.round(np.exp(np.median(samples[:, 4])), 1) == 10.6


def test_stronggp_constructor():
    model = StrongGPModeler([1, 2], [3, 4])
    assert model.mu == (-17, -13, 5, 1.9)
