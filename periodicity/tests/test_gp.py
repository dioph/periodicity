import unittest

import numpy as np

from periodicity.data import lightcurve1, lightcurve2
from periodicity.gp import make_gaussian_prior, FastGPModeler, StrongGPModeler


class MakeGaussianPriorTest(unittest.TestCase):
    def test_lightcurve_1(self):
        log_periods = np.linspace(-3, 5, 1000)
        t, y, dy = lightcurve1()
        prior = make_gaussian_prior(t, y,
                                    fundamental_height=0.9,
                                    fundamental_width=0.2)
        prior_prob = prior(log_periods)
        # prior has a maximum at approx 24.7 days
        self.assertEqual(775, prior_prob.argmax())
        peaks = [i for i in range(1, len(log_periods) - 1)
                 if prior_prob[i - 1] < prior_prob[i]
                 and prior_prob[i + 1] < prior_prob[i]]
        # prior has peaks at 0.6, 1.1 and 24.7 days
        self.assertEqual(3, len(peaks))

    def test_lightcurve_2(self):
        log_periods = np.linspace(-3, 5, 1000)
        t, y, dy = lightcurve2()
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

    def test_minimize(self):
        t, y, dy = lightcurve2()
        model = FastGPModeler(t, y)
        _, _, _, v = model.minimize()
        self.assertAlmostEqual(10.618, np.exp(v[4]), places=3)

    def test_docs_example(self):
        np.random.seed(42)
        t, y, dy = lightcurve1()
        model = FastGPModeler(t, y)
        model.prior = make_gaussian_prior(t, y)
        model.minimize()
        samples = model.mcmc(n_walkers=32, n_steps=5000, burn=500)
        self.assertAlmostEqual(24., np.exp(np.median(samples[:, 4])), places=0)


class StrongGPTest(unittest.TestCase):
    def test_class_constructor(self):
        model = StrongGPModeler([1, 2], [3, 4])
        self.assertEqual((-17, -13, 5, 1.9), model.mu)
