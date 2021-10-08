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
    soln, _ = model.minimize(model.gp, options={'disp': True})
    opt_params = model.prior_transform(soln.x)
    assert np.round(opt_params["period"], 1) == 10.0
    assert np.allclose(soln.x, np.array([0.49532516, 0.50588049, 0.5597264, 0.59941389, 0.99999, 0.0457812]))


def test_harmonicgp_spotted_lc_minimize():
    t, y, dy = SpottedStar()
    sig = TSeries(t, y)
    model = HarmonicGP(sig, err=dy)
    soln, _ = model.minimize(model.gp, options={'disp': True})
    opt_params = model.prior_transform(soln.x)
    assert np.round(opt_params["period"], 1) == 11.0
    assert np.allclose(soln.x, np.array([0.49287794, 0.49067351, 0.60968014, 0.54946591, 0.63684495, 0.1428619, 0.55616662]))


# def test_browniangp_spotted_lc_mcmc():
#     t, y, dy = SpottedStar()
#     sig = TSeries(t, y)
#     model = BrownianGP(sig, err=dy)
#     trace, tau = model.mcmc(n_walkers=16, n_steps=1000, burn=100, random_seed=42)
#     assert np.round(np.median(trace["period"]), 1) == 10.0


# def test_harmonicgp_spotted_lc_mcmc():
#     t, y, dy = SpottedStar()
#     sig = TSeries(t, y)
#     model = HarmonicGP(sig, err=dy)
#     trace, tau = model.mcmc(n_walkers=16, n_steps=1000, burn=100, random_seed=42)
#     assert np.round(np.median(trace["period"]), 1) == 11.0
