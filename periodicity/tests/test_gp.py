import unittest

import numpy as np

from periodicity.data import spotted_lc
from periodicity.gp import make_gaussian_prior, FastGPModeler, StrongGPModeler


class MakeGaussianPriorTest(unittest.TestCase):
    def test_spotted_lc(self):
        log_periods = np.linspace(-3, 5, 1000)
        t, y, dy = spotted_lc()
        prior = make_gaussian_prior(t, y)
        prior_prob = prior(log_periods)
        # prior has a maximum at approx 10.7 days
        self.assertEqual(671, prior_prob.argmax())
        peaks = [i for i in range(1, len(log_periods) - 1)
                 if prior_prob[i - 1] < prior_prob[i]
                 and prior_prob[i + 1] < prior_prob[i]]
        # prior has peaks at 0.4, 0.8, 1.7, 3.5, 5.6, 10.7 and 21.5 days
        self.assertEqual(7, len(peaks))


class FastGPTest(unittest.TestCase):
    def test_class_constructor(self):
        model = FastGPModeler([1, 2], [3, 4])
        self.assertEqual((-17, -13, 0, 3), model.mu)

    def test_spotted_lc_minimize(self):
        t, y, dy = spotted_lc()
        model = FastGPModeler(t, y)
        _, _, _, v = model.minimize()
        self.assertAlmostEqual(10.618, np.exp(v[4]), places=3)

    def test_spotted_lc_mcmc(self):
        np.random.seed(42)
        t, y, dy = spotted_lc()
        model = FastGPModeler(t, y)
        model.prior = make_gaussian_prior(t, y)
        model.minimize()
        samples = model.mcmc(n_walkers=16, n_steps=1000, burn=100)
        self.assertAlmostEqual(10.6, np.exp(np.median(samples[:, 4])), places=1)


class StrongGPTest(unittest.TestCase):
    def test_class_constructor(self):
        model = StrongGPModeler([1, 2], [3, 4])
        self.assertEqual((-17, -13, 5, 1.9), model.mu)
