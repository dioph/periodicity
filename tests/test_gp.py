import numpy as np

from periodicity.core import TSeries
from periodicity.data import SpottedStar
from periodicity.gp import BrownianGP, HarmonicGP, make_gaussian_prior


def test_make_gaussian_prior_spotted_lc():
    log_periods = np.linspace(-3, 5, 1000)
    t, y, dy = SpottedStar()
    prior = make_gaussian_prior(TSeries(t, y))
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


def test_browniangp_spotted_lc_minimize():
    t, y, dy = SpottedStar()
    sig = TSeries(t, y)
    model = BrownianGP(sig, err=dy)
    soln, _ = model.minimize(
        model.gp, options={"disp": True, "eps": 1e-10, "ftol": 1e-10}
    )
    assert soln.fun < -12898
    assert np.all(np.logical_and(soln.x <= 0.99999, soln.x >= 1e-5))


def test_harmonicgp_spotted_lc_minimize():
    t, y, dy = SpottedStar()
    sig = TSeries(t, y)
    model = HarmonicGP(sig, err=dy)
    soln, _ = model.minimize(
        model.gp, options={"disp": True, "eps": 1e-10, "ftol": 1e-10}
    )
    assert soln.fun < -13188
    assert np.all(np.logical_and(soln.x <= 0.99999, soln.x >= 1e-5))


def test_browniangp_spotted_lc_mcmc():
    t, y, dy = SpottedStar()
    sig = TSeries(t, y)
    model = BrownianGP(sig, err=dy)
    trace, tau = model.mcmc(n_walkers=16, n_steps=1000, burn=200, random_seed=42)
    assert trace["period"].shape == (16 * (1000 - 200),)
    assert np.round(np.median(trace["period"]), 0) == 10.0


def test_harmonicgp_spotted_lc_mcmc():
    t, y, dy = SpottedStar()
    sig = TSeries(t, y)
    model = HarmonicGP(sig, err=dy)
    trace, tau = model.mcmc(n_walkers=16, n_steps=1000, burn=200, random_seed=42)
    assert trace["period"].shape == (16 * (1000 - 200),)
    assert np.round(np.median(trace["period"]), 0) == 11.0
