from autograd import numpy as np
import emcee
from scipy.optimize import minimize
from scipy.stats import linregress
from tqdm import tqdm

from .acf import gaussian, acf_harmonic_quality


class GPModeler(object):
    """Abstract class implementing common functions for a GP Model"""
    def __init__(self, t, x):
        self.t = np.array(t, float)
        self.x = np.array(x, float)

        def uniform_prior(logp):
            window = np.logical_and(self.bounds['log_P'][0] < logp, logp < self.bounds['log_P'][1])
            probs = np.ones_like(logp)
            probs[~window] = 0.0
            return probs

        a, b = linregress(self.t, self.x)[:2]
        self.x -= (a * self.t + b)

        self.prior = uniform_prior
        self.gp = None
        self.mu = ()
        self.bounds = dict()
        self.sd = ()

    def lnlike(self, p):
        self.gp.set_parameter_vector(p)
        self.gp.compute(self.t)
        ll = self.gp.log_likelihood(self.x, quiet=True)
        return ll

    def lnprior(self, p):
        priors = np.append([gaussian(self.mu[i], self.sd[i]) for i in range(len(self.mu))], [self.prior])
        for i, (lo, hi) in enumerate(self.bounds.values()):
            if not(lo < p[i] < hi):
                return -np.inf
        lp = np.sum(np.log(priors[i](p[i])) for i in range(len(p)))
        return lp

    def sample_prior(self, N):
        ndim = len(self.gp)
        samples = np.inf * np.ones((N, ndim))
        m = np.ones(N, dtype=bool)
        nbad = m.sum()
        while nbad > 0:
            r = np.random.randn(N * (ndim - 1)).reshape((N, ndim - 1))
            for i in range(ndim - 1):
                samples[m, i] = r[m, i] * self.sd[i] + self.mu[i]
            samples[m, -1] = self.sample_period(nbad)
            lp = np.array([self.lnprior(p) for p in samples])
            m = ~np.isfinite(lp)
            nbad = m.sum()
        return samples

    def sample_period(self, N):
        logP = np.arange(self.bounds['log_P'][0], self.bounds['log_P'][1], .005)
        probs = self.prior(logP)
        probs /= probs.sum()
        periods = np.random.choice(logP.size, N, p=probs)
        samples = logP[periods]
        return samples

    def lnprob(self, p):
        self.gp.set_parameter_vector(p)
        lp = self.lnprior(p)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.lnlike(p)

    def nll(self, p, x):
        self.gp.set_parameter_vector(p)
        return -self.gp.log_likelihood(x)

    def grad_nll(self, p, x):
        pass

    def minimize(self):
        """Minimizes negative log-likelihood function within bounds

        Returns
        -------
        t: array-like
            5000 uniform time samples within modeler time array
        mu: array-like
            predicted mean function with maximum likelihood hyperparameters
        sd: array-like
            predicted error at each sample with maximum likelihood hyperparameters
        v: list
            maximum likelihood hyperparameters
        """
        assert self.t.size <= 10000, "Don't forget to decimate before minimizing! (N={})".format(self.t.size)
        self.gp.compute(self.t)
        p0 = self.gp.get_parameter_vector()
        results = minimize(fun=self.nll, x0=p0, args=self.x, method='L-BFGS-B',
                           jac=self.grad_nll, bounds=self.bounds.values())
        self.gp.set_parameter_vector(results.x)
        self.gp.compute(self.t)
        t = np.linspace(self.t.min(), self.t.max(), 5000)
        mu, var = self.gp.predict(self.x, self.t, return_var=True)
        sd = np.sqrt(var)
        return t, mu, sd, results.x

    def mcmc(self, nwalkers=50, nsteps=1000, burn=0, useprior=False):
        """Samples the posterior probability distribution with a Markov Chain Monte Carlo simulation

        Parameters
        ----------
        nwalkers: int (optional default=50)
            number of walkers
        nsteps: int (optional default=1000)
            number of steps taken by each walker
        burn: int (optional default=0)
            number of burn-in samples to remove from the beginning of the simulation
        useprior: bool (optional default=False)
            whether to sample from the prior distribution or use a ball centered at the current hyperparameter vector

        Returns
        -------
        samples: array-like
            resulting samples of the posterior distribution of the hyperparameters
        """
        ndim = len(self.gp)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob)
        # TODO: figure out a way to optimize time complexity by parallel computing
        p = self.gp.get_parameter_vector()
        if useprior:
            p0 = self.sample_prior(nwalkers)
        else:
            p0 = p + 1e-5 * np.random.randn(nwalkers, ndim)
        for _ in tqdm(sampler.sample(p0, iterations=nsteps), total=nsteps):
            pass
        samples = sampler.chain[:, burn:, :].reshape(-1, ndim)
        return samples


class FastGPModeler(GPModeler):
    """GP Model based on a sum of exponentials kernel (fast but not so strong)"""
    def __init__(self, t, x, log_sigma=-17, log_B=-13, log_C=0, log_L=3, log_P=2, bounds=None, sd=None):
        import celerite

        class CustomTerm(celerite.terms.Term):
            """Custom sum of exponentials kernel"""
            parameter_names = ("log_B", "log_C", "log_L", "log_P")

            def get_real_coefficients(self, params):
                log_B, log_C, log_L, log_P = params
                a = np.exp(log_B)
                b = np.exp(log_C)
                c = np.exp(-log_L)
                return a * (1.0 + b) / (2.0 + b), c

            def get_complex_coefficients(self, params):
                log_B, log_C, log_L, log_P = params
                a = np.exp(log_B)
                b = np.exp(log_C)
                c = np.exp(-log_L)
                return a / (2.0 + b), 0.0, c, 2 * np.pi * np.exp(-log_P)

        super(FastGPModeler, self).__init__(t, x)
        self.mu = (log_sigma, log_B, log_C, log_L)
        if bounds is None:
            bounds = {'log_sigma': (-20, 0), 'log_B': (-20, 0), 'log_C': (-5, 5),
                      'log_L': (1.5, 5.0), 'log_P': (-0.69, 4.61)}
        self.bounds = bounds
        if sd is None:
            sd = (5.0, 5.7, 2.0, 0.7)
        self.sd = sd
        term = celerite.terms.JitterTerm(log_sigma=log_sigma)
        term += CustomTerm(log_B=log_B, log_C=log_C, log_L=log_L, log_P=log_P, bounds=bounds)
        self.gp = celerite.GP(term)

    def grad_nll(self, p, x):
        self.gp.set_parameter_vector(p)
        return -self.gp.grad_log_likelihood(x)[1]


class StrongGPModeler(GPModeler):
    """GP Model based on Quasi-Periodic kernel (strong but not so fast)"""
    def __init__(self, t, x, log_sigma=-17, log_A=-13, log_L=5, log_G=1.9, log_P=2, bounds=None, sd=None):
        import george

        super(StrongGPModeler, self).__init__(t, x)
        self.mu = (log_sigma, log_A, log_L, log_G)
        if bounds is None:
            bounds = {'log_sigma': (-20, 0), 'log_A': (-20, 0), 'log_L': (2, 8),
                      'log_G': (0, 3), 'log_P': (-0.69, 4.61)}
        self.bounds = bounds
        if sd is None:
            sd = (5.0, 5.7, 1.2, 1.4)
        self.sd = sd
        kernel = george.kernels.ConstantKernel(log_A, bounds=[bounds['log_A']])
        kernel *= george.kernels.ExpSquaredKernel(np.exp(log_L), metric_bounds=[bounds['log_L']])
        kernel *= george.kernels.ExpSine2Kernel(log_G, log_P, bounds=[bounds['log_G'], bounds['log_P']])
        self.gp = george.GP(kernel, solver=george.HODLRSolver, white_noise=log_sigma, fit_white_noise=True)

    def grad_nll(self, p, x):
        self.gp.set_parameter_vector(p)
        return -self.gp.grad_log_likelihood(x)


class TensorGPModeler(GPModeler):
    """GP Model using symbolic computing from PyMC3 and Theano for optimization and enabling GPUs"""
    def __init__(self, t, x, sigma=4e-8, A=2e-6, L=150, inv_G=-0.6, P=7.5):
        import pymc3 as pm

        super(TensorGPModeler, self).__init__(t, x)
        cov = A * pm.gp.cov.ExpQuad(1, L) * pm.gp.cov.Periodic(1, P, inv_G)

        raise NotImplementedError

    def lnlike(self, p):
        pass

    def lnprior(self, p):
        pass

    def sample_prior(self, N):
        pass

    def sample_period(self, N):
        pass

    def lnprob(self, p):
        pass

    def nll(self, p, x):
        pass

    def grad_nll(self, p, x):
        pass

    def minimize(self):
        pass

    def mcmc(self, nwalkers=50, nsteps=1000, burn=0, useprior=False):
        pass


def make_gaussian_prior(t, x, pmin=None, periods=None):
    """Generates a weighted sum of Gaussians as a probability prior on the signal period

    Based on Angus et al. (2018) MNRAS 474, 2094A

    Parameters
    ----------
    t: array-like
        time array
    x: array-like
        input quasi-periodic signal
    pmin: float (optional)
        lower cutoff period to filter signal
    periods: list (optional)
        list of higher cutoff periods to filter signal

    Returns
    -------
    gaussian_prior: function
        prior on logP
    """
    ps, hs, qs = acf_harmonic_quality(t, x, pmin, periods)

    def gaussian_prior(logp):
        tot = 0
        for pi, qi in zip(ps, qs):
            qi = max(qi, 0)
            gaussian1 = gaussian(np.log(pi), .2)
            gaussian2 = gaussian(np.log(pi / 2), .2)
            gaussian3 = gaussian(np.log(2 * pi), .2)
            tot += qi * (.9 * gaussian1(logp) + .05 * gaussian2(logp) + .05 * gaussian3(logp))
        tot /= np.sum(qs)
        return tot

    return gaussian_prior
