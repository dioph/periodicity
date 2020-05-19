import emcee
from autograd import numpy as np
from scipy.optimize import minimize
from scipy.stats import linregress

from .utils import gaussian, acf_harmonic_quality

__all__ = ['GPModeler', 'FastGPModeler', 'StrongGPModeler',
           'make_gaussian_prior']


class GPModeler(object):
    """Abstract class implementing common functions for a GP Model

    Parameters
    ----------
    t: array-like
        Time array.
    x: array-like
        Signal array.

    Attributes
    ----------
    prior: function
        A function with a single parameter `log_p` which returns the prior
        probability on the logarithm of the period.
        By default it is uniform within the specified bounds.
    gp
        Gaussian Process object with `compute`, `log_likelihood`,
        `set_parameter_vector` and `get_parameter_vector` methods implemented.
    mu, sd: tuple
        Means and standard deviations of gaussian priors on the hyperparameters.
    bounds: dict
        Lower and upper bounds on the values of the hyperparameters.
    """
    def __init__(self, t, x):
        self.t = np.array(t, float)
        self.x = np.array(x, float)

        def uniform_prior(log_p):
            window = np.logical_and(self.bounds['log_P'][0] < log_p,
                                    log_p < self.bounds['log_P'][1])
            prob = np.ones_like(log_p)
            prob[~window] = 0.0
            return prob

        a, b = linregress(self.t, self.x)[:2]
        self.x -= (a * self.t + b)

        self.prior = uniform_prior
        self.gp = None
        self.mu = ()
        self.bounds = dict()
        self.sd = ()

    def ln_like(self, p):
        """Log-likelihood function."""
        self.gp.set_parameter_vector(p)
        self.gp.compute(self.t)
        ll = self.gp.log_likelihood(self.x, quiet=True)
        return ll

    def ln_prior(self, p):
        """Log-prior function."""
        priors = [gaussian(self.mu[i], self.sd[i])
                  for i in range(len(self.mu))] + [self.prior]
        for i, (lo, hi) in enumerate(self.bounds.values()):
            if not(lo < p[i] < hi):
                return -np.inf
        lp = np.sum([np.log(priors[i](p[i])) for i in range(len(p))])
        return lp

    def sample_prior(self, n_samples):
        """Sample from the prior distribution."""
        n_dim = len(self.gp)
        samples = np.inf * np.ones((n_samples, n_dim))
        m = np.ones(n_samples, dtype=bool)
        n_bad = m.sum()
        while n_bad > 0:
            samples[m, :-1] = np.random.normal(self.mu, self.sd,
                                               size=(n_bad, n_dim - 1))
            samples[m, -1] = self.sample_period(n_bad)
            lp = np.array([self.ln_prior(p) for p in samples])
            m = ~np.isfinite(lp)
            n_bad = m.sum()
        return samples

    def sample_period(self, n_samples):
        """Sample log-periods from its prior distribution."""
        log_p_min, log_p_max = self.bounds['log_P']
        log_p = np.linspace(log_p_min, log_p_max, 1000)
        prob = self.prior(log_p)
        prob /= prob.sum()
        samples = np.random.choice(log_p, n_samples, p=prob)
        return samples

    def ln_prob(self, p):
        """Log-posterior function."""
        self.gp.set_parameter_vector(p)
        lp = self.ln_prior(p)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.ln_like(p)

    def nll(self, p, x):
        """Negative log-likelihood."""
        self.gp.set_parameter_vector(p)
        return -self.gp.log_likelihood(x)

    def grad_nll(self, p, x):
        """Gradient of the negative log-likelihood."""
        raise NotImplementedError

    def minimize(self):
        """Minimizes negative log-likelihood function within bounds.

        Returns
        -------
        t: ndarray[5000,]
            Uniform time samples within modeler time array.
        mu: ndarray[5000,]
            Predicted mean function with ML hyperparameters.
        sd: ndarray[5000,]
            Predicted error at each sample with ML hyperparameters.
        v: list
            ML (maximum likelihood) hyperparameters.
        """
        if self.t.size > 10_000:
            raise ValueError(f"Don't forget to decimate before minimizing! "
                             f"(N={self.t.size})")
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

    def mcmc(self, n_walkers=50, n_steps=1000, burn=0, use_prior=False):
        """Samples the posterior probability distribution with a Markov Chain
        Monte Carlo simulation.

        Parameters
        ----------
        n_walkers: int, optional
            Number of walkers (the default is 50).
        n_steps: int, optional
            Number of steps taken by each walker (the default is 1000).
        burn: int, optional
            Number of burn-in samples to remove from the beginning of the
            simulation (the default is 0).
        use_prior: bool, optional
            Whether to start walkers by sampling from the prior distribution.
            The default is False, in which case a ball centered
            at the current hyperparameter vector is used.

        Returns
        -------
        samples: ndarray[n_walkers * (n_steps - burn), n_dim]
            Samples of the posterior hyperparameter distribution.
        """
        n_dim = len(self.gp)
        sampler = emcee.EnsembleSampler(n_walkers, n_dim, self.ln_prob)
        # TODO: multi-threading
        p = self.gp.get_parameter_vector()
        if use_prior:
            p0 = self.sample_prior(n_walkers)
        else:
            p0 = p + 1e-5 * np.random.randn(n_walkers, n_dim)
        sampler.run_mcmc(p0, n_steps, progress=True)
        samples = sampler.get_chain(discard=burn, flat=True)
        return samples


class FastGPModeler(GPModeler):
    """Model based on a Sum of Exponentials kernel (fast but not so strong)"""
    def __init__(self, t, x,
                 log_sigma=-17, log_b=-13, log_c=0, log_l=3, log_p=2,
                 bounds=None, sd=None):
        import celerite

        class CustomTerm(celerite.terms.Term):
            """Custom sum of exponentials kernel"""
            parameter_names = ("log_b_", "log_c_", "log_l_", "log_p_")

            def get_real_coefficients(self, params):
                log_b_, log_c_, log_l_, log_p_ = params
                a = np.exp(log_b_)
                b = np.exp(log_c_)
                c = np.exp(-log_l_)
                return a * (1.0 + b) / (2.0 + b), c

            def get_complex_coefficients(self, params):
                log_b_, log_c_, log_l_, log_p_ = params
                a = np.exp(log_b_)
                b = np.exp(log_c_)
                c = np.exp(-log_l_)
                return a / (2.0 + b), 0.0, c, 2 * np.pi * np.exp(-log_p_)

        super(FastGPModeler, self).__init__(t, x)
        self.mu = (log_sigma, log_b, log_c, log_l)
        if bounds is None:
            bounds = {
                'log_sigma': (-20, 0),
                'log_B': (-20, 0),
                'log_C': (-5, 5),
                'log_L': (1.5, 5.0),
                'log_P': (-0.69, 4.61)
            }
        self.bounds = bounds
        if sd is None:
            sd = (5.0, 5.7, 2.0, 0.7)
        self.sd = sd
        term = celerite.terms.JitterTerm(log_sigma=log_sigma)
        term += CustomTerm(log_b_=log_b,
                           log_c_=log_c,
                           log_l_=log_l,
                           log_p_=log_p,
                           bounds=bounds)
        self.gp = celerite.GP(term)

    def grad_nll(self, p, x):
        self.gp.set_parameter_vector(p)
        return -self.gp.grad_log_likelihood(x)[1]


class StrongGPModeler(GPModeler):
    """GP Model based on Quasi-Periodic kernel (strong but not so fast)"""
    def __init__(self, t, x,
                 log_sigma=-17, log_a=-13, log_l=5, log_g=1.9, log_p=2,
                 bounds=None, sd=None):
        import george

        super(StrongGPModeler, self).__init__(t, x)
        self.mu = (log_sigma, log_a, log_l, log_g)
        if bounds is None:
            bounds = {
                'log_sigma': (-20, 0),
                'log_A': (-20, 0),
                'log_L': (2, 8),
                'log_G': (0, 3),
                'log_P': (-0.69, 4.61)
            }
        self.bounds = bounds
        if sd is None:
            sd = (5.0, 5.7, 1.2, 1.4)
        self.sd = sd
        kernel = george.kernels.ConstantKernel(
            log_a,
            bounds=[bounds['log_A']]
        )
        kernel *= george.kernels.ExpSquaredKernel(
            np.exp(log_l),
            metric_bounds=[bounds['log_L']]
        )
        kernel *= george.kernels.ExpSine2Kernel(
            log_g, log_p,
            bounds=[bounds['log_G'], bounds['log_P']]
        )
        self.gp = george.GP(kernel, solver=george.HODLRSolver,
                            white_noise=log_sigma, fit_white_noise=True)

    def grad_nll(self, p, x):
        self.gp.set_parameter_vector(p)
        return -self.gp.grad_log_likelihood(x)


class TensorGPModeler(GPModeler):
    """GP Model using symbolic computing from PyMC3 and Theano for optimization and enabling GPUs"""
    def __init__(self, t, x, sigma=4e-8, A=2e-6, L=150, inv_G=-0.6, P=7.5):
        import pymc3 as pm

        super(TensorGPModeler, self).__init__(t, x)
        cov = A * pm.gp.cov.ExpQuad(1, L) * pm.gp.cov.Periodic(1, P, inv_G)

    def ln_like(self, p):
        pass

    def ln_prior(self, p):
        pass

    def sample_prior(self, N):
        pass

    def sample_period(self, N):
        pass

    def ln_prob(self, p):
        pass

    def nll(self, p, x):
        pass

    def grad_nll(self, p, x):
        pass

    def minimize(self):
        pass

    def mcmc(self, nwalkers=50, nsteps=1000, burn=0, useprior=False):
        pass


def make_gaussian_prior(t, x, p_min=None, periods=None, a=1, b=2, n=8,
                        fundamental_height=0.8, fundamental_width=0.1):
    """Generates a weighted sum of Gaussian PDFs as a probability prior on the
    logarithm of the signal period.

    Based on [#]_

    Parameters
    ----------
    t: array-like
        Time array.
    x: array-like
        Input (quasi-)periodic signal.
    p_min: float, optional
        Lower cutoff period to filter signal.
    periods: list, optional
        List of higher cutoff periods to filter signal.
        Only periods between `p_min` and half the baseline will be considered.
    a, b, n: float, optional
        If `periods` is not given, then the first `n` powers of `b` scaled
        by `a` will be used:

        ``periods = a * b ** np.arange(n)``
    fundamental_height: float, optional
        Weight of the gaussian mixture on the fundamental peak.
        The double and half harmonics both are equally weighted
        ``(1 - fundamental_height) / 2``.
        Defaults to 0.8.
    fundamental_width: float, optional
        Width (standard deviation) of the gaussian PDFs in the prior.
        Defaults to 0.1.

    Returns
    -------
    gaussian_prior: function
        prior on the log-period

    See Also
    --------
    periodicity.utils.acf_harmonic_quality

    References
    ----------
    .. [#] R. Angus, T. Morton, S. Aigrain, D. Foreman-Mackey, V. Rajpaul,
       "Inferring probabilistic stellar rotation periods using Gaussian
       processes," MNRAS, February 2018.
    """
    ps, hs, qs = acf_harmonic_quality(t, x, p_min, periods, a, b, n)

    def gaussian_prior(log_p):
        tot = 0
        fh = fundamental_height
        hh = (1 - fh) / 2
        fw = fundamental_width
        for pi, qi in zip(ps, qs):
            qi = max(qi, 0)
            gaussian1 = gaussian(np.log(pi), fw)
            gaussian2 = gaussian(np.log(pi / 2), fw)
            gaussian3 = gaussian(np.log(2 * pi), fw)
            tot += qi * (fh * gaussian1(log_p) +
                         hh * gaussian2(log_p) +
                         hh * gaussian3(log_p))
        tot /= np.sum(qs)
        return tot

    return gaussian_prior
