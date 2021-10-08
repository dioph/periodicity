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
    assert np.allclose(soln.x, np.array([0.49999587, 0.50236365, 0.5598169, 0.599418, 0.99999, 0.24995572]))


def test_harmonicgp_spotted_lc_minimize():
    t, y, dy = SpottedStar()
    sig = TSeries(t, y)
    model = HarmonicGP(sig, err=dy)
    soln, _ = model.minimize(model.gp, options={'disp': True})
    opt_params = model.prior_transform(soln.x)
    assert np.round(opt_params["period"], 1) == 11.0
    assert np.allclose(soln.x, np.array([0.49999355, 0.47318792, 0.609833, 0.57830117, 0.56661697, 0.48780288, 0.52261041]))


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
