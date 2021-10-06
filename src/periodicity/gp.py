import celerite2
import celerite2.theano
import emcee
import george
import numpy as np
import pymc3 as pm
import pymc3_ext as pmx
from scipy.optimize import minimize
from scipy.stats import norm

from .core import TSeries

__all__ = [
    "GeorgeModeler",
    "CeleriteModeler",
    "TheanoModeler",
    "QuasiPeriodicGP",
    "BrownianGP",
    "HarmonicGP",
    "BrownianTheanoGP",
    "HarmonicTheanoGP",
]


def _gaussian(mu, sd):
    """Simple 1D Gaussian function generator.

    Parameters
    ----------
    mu: float
        Mean.
    sd: float
        Standard deviation.

    Returns
    -------
    pdf: function
        1D Gaussian PDF with given parameters.
    """

    def pdf(x):
        z = (x - mu) / sd
        return np.exp(-z * z / 2.0) / np.sqrt(2.0 * np.pi) / sd

    return pdf


def make_ppf(x, pdf):
    cdf = np.cumsum(pdf)
    cdf /= cdf[-1]

    def ppf(q):
        icdf = np.interp(q, cdf, x)
        return icdf

    return ppf


def make_gaussian_prior(
    signal,
    p_min=None,
    periods=None,
    a=1.0,
    b=2.0,
    n=8,
    fundamental_height=0.8,
    fundamental_width=0.1,
):
    """Generates a weighted sum of Gaussian PDFs as a probability prior on the
    logarithm of the signal period.

    Based on [#]_

    Parameters
    ----------
    signal: TSeries or array-like
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
    if not isinstance(signal, TSeries):
        signal = TSeries(data=signal)
    if periods is None:
        periods = a * b ** np.arange(n)
    if p_min is None:
        p_min = max(np.min(periods) / 10, 3 * signal.median_dt)
    periods = np.array([p for p in periods if p_min < p < signal.baseline / 2])
    ps, hs, qs = [], [], []
    for p_max in periods:
        p, h, q = signal.acf_period_quality(p_min, p_max)
        ps.append(p)
        hs.append(h)
        qs.append(q)

    def gaussian_prior(log_p):
        tot = 0
        fh = fundamental_height
        hh = (1 - fh) / 2
        fw = fundamental_width
        for p, q in zip(ps, qs):
            q = max(q, 0)
            gaussian1 = _gaussian(np.log(p), fw)
            gaussian2 = _gaussian(np.log(p / 2), fw)
            gaussian3 = _gaussian(np.log(2 * p), fw)
            tot += q * (
                fh * gaussian1(log_p) + hh * gaussian2(log_p) + hh * gaussian3(log_p)
            )
        tot /= np.sum(qs)
        return tot

    return gaussian_prior


class GeorgeModeler(object):
    def __init__(self, signal, err, init_period=None, period_prior=None):
        if not isinstance(signal, TSeries):
            signal = TSeries(data=signal)
        self.signal = signal
        self.err = err
        self.t = self.signal.time
        self.y = self.signal.values
        self.sigma = np.std(self.y)
        self.jitter = np.min(self.err) ** 2
        self.mean = np.mean(self.y)
        if init_period is None:
            init_period = np.sqrt(signal.size) * signal.median_dt
        if period_prior is None:

            def period_prior(u):
                sigma_period = 0.2 * np.log(signal.size)
                return np.exp(norm.ppf(u, np.log(init_period), sigma_period))

        self.period_prior = period_prior
        self.gp = george.GP(
            self.kernel,
            solver=george.HODLRSolver,
            mean=self.mean,
            fit_mean=True,
            white_noise=np.log(self.jitter),
            fit_white_noise=True,
        )
        self.gp.compute(self.t, yerr=self.err)
        self.ndim = len(self.gp)

    def prior_transform(self, u):
        raise NotImplementedError("subclasses must implement this method")

    def set_params(self, theta, gp):
        gp.set_parameter_vector(theta)
        gp.compute(self.t, yerr=self.err)
        return gp

    def get_prediction(self, time, gp):
        mu, var = gp.predict(self.y, t=time, return_var=True)
        sd = np.sqrt(var)
        return mu, sd

    def nll(self, u, gp):
        """Objective function based on the Negative Log-Likelihood."""
        theta = self.prior_transform(u)
        gp = self.set_params(theta, gp)
        ll = gp.log_likelihood(self.y, quiet=True)
        return -ll if np.isfinite(ll) else 1e25

    def minimize(self, gp):
        """Gradient-based optimization of the objective function within the unit
        hypercube."""
        u0 = np.full(self.ndim, 0.5)
        bounds = [(1e-5, 1 - 1e-5) for x in u0]
        soln = minimize(self.nll, u0, method="L-BFGS-B", args=(gp,), bounds=bounds)
        opt_theta = self.prior_transform(soln.x)
        opt_gp = self.set_params(opt_theta, gp)
        return soln, opt_gp

    def log_prob(self, u, gp):
        """Posterior distribution over the hyperparameters."""
        if any(u >= 1 - 1e-5) or any(u <= 1e-5):
            return -np.inf
        params = self.prior_transform(u)
        gp = self.set_params(params, gp)
        ll = gp.log_likelihood(self.y)
        return ll

    def mcmc(
        self, n_walkers=50, n_steps=1000, burn=0, use_prior=False, random_seed=None
    ):
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
            at the MLE hyperparameter vector is used.

        Returns
        -------
        samples: ndarray[n_dim, n_walkers * (n_steps - burn)]
            Samples of the posterior hyperparameter distribution.
        """
        rng = np.random.default_rng(random_seed)
        np.random.seed(random_seed)
        if use_prior:
            u0 = rng.random((n_walkers, self.ndim))
        else:
            soln, opt_gp = self.minimize(self.gp)
            u0 = soln.x + 1e-5 * rng.standard_normal((n_walkers, self.ndim))
        # TODO: multi-threading
        sampler = emcee.EnsembleSampler(
            n_walkers, self.ndim, self.log_prob, args=(self.gp,)
        )
        sampler.run_mcmc(u0, n_steps, progress=True)
        samples = sampler.get_chain(discard=burn, flat=True)
        tau = sampler.get_autocorr_time(discard=burn, quiet=True)
        trace = np.vstack(self.prior_transform(samples.T))
        self.sampler = sampler
        return trace, tau


class QuasiPeriodicGP(GeorgeModeler):
    def __init__(self, signal, err, init_period=None, period_prior=None):
        kernel = george.kernels.ConstantKernel(np.var(signal))
        kernel *= george.kernels.ExpSquaredKernel(10.0)
        kernel *= george.kernels.ExpSine2Kernel(4.5, 0.0)
        self.kernel = kernel
        super().__init__(signal, err, init_period, period_prior)

    def prior_transform(self, u):
        period = self.period_prior(u[5])
        mean = norm.ppf(u[0], self.mean, 10.0)
        jitter = np.exp(norm.ppf(u[1], np.log(self.jitter), 5.0))
        sigma2 = np.exp(norm.ppf(u[2], 2 * np.log(self.sigma), 5.0))
        tau2 = (10 ** u[3] * period) ** 2
        gamma = np.exp(norm.ppf(u[4], 1.5, 1.5))
        theta = [mean, jitter, sigma2, tau2, gamma, period]
        return theta


class CeleriteModeler(object):
    def __init__(self, signal, err, init_period=None, period_prior=None):
        if not isinstance(signal, TSeries):
            signal = TSeries(data=signal)
        self.signal = signal
        self.err = err
        self.t = self.signal.time
        self.y = self.signal.values
        self.sigma = np.std(self.y)
        self.jitter = np.min(self.err) ** 2
        self.mean = np.mean(self.y)
        if init_period is None:
            init_period = np.sqrt(signal.size) * signal.median_dt
        if period_prior is None:

            def period_prior(u):
                sigma_period = 0.5 * np.log(signal.size)
                return np.exp(norm.ppf(u, np.log(init_period), sigma_period))

        self.period_prior = period_prior
        init_params = self.prior_transform(np.full(self.ndim, 0.5))
        init_params["period"] = init_period
        mean = init_params.pop("mean")
        jitter = init_params.pop("jitter")
        self.gp = celerite2.GaussianProcess(self.kernel(**init_params), mean=mean)
        self.gp.compute(self.t, diag=self.err ** 2 + jitter)

    def prior_transform(self, u):
        raise NotImplementedError("subclasses must implement this method")

    def set_params(self, params, gp):
        gp.mean = params.pop("mean")
        jitter = params.pop("jitter")
        gp.kernel = self.kernel(**params)
        gp.compute(self.t, diag=self.err ** 2 + jitter, quiet=True)
        return gp

    def get_psd(self, frequency, gp):
        return gp.kernel.get_psd(2 * np.pi * frequency)

    def get_prediction(self, time, gp):
        mu, var = gp.predict(self.y, t=time, return_var=True)
        sd = np.sqrt(var)
        return mu, sd

    def nll(self, u, gp):
        """Objective function based on the Negative Log-Likelihood."""
        params = self.prior_transform(u)
        gp = self.set_params(params, gp)
        return -gp.log_likelihood(self.y)

    def minimize(self, gp):
        """Gradient-based optimization of the objective function within the unit
        hypercube."""
        u0 = np.full(self.ndim, 0.5)
        bounds = [(1e-5, 1 - 1e-5) for x in u0]
        soln = minimize(self.nll, u0, method="L-BFGS-B", args=(gp,), bounds=bounds)
        opt_params = self.prior_transform(soln.x)
        opt_gp = self.set_params(opt_params, gp)
        return soln, opt_gp

    def log_prob(self, u, gp):
        if any(u >= 1 - 1e-5) or any(u <= 1e-5):
            return -np.inf
        params = self.prior_transform(u)
        gp = self.set_params(params, gp)
        ll = gp.log_likelihood(self.y)
        return ll

    def mcmc(
        self, n_walkers=50, n_steps=1000, burn=0, use_prior=False, random_seed=None
    ):
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
            at the MLE hyperparameter vector is used.

        Returns
        -------
        trace: dict
            Samples of the posterior hyperparameter distribution.
        tau: ndarray
            Estimated autocorrelation time of MCMC chain for each parameter.
        """
        rng = np.random.default_rng(random_seed)
        np.random.seed(random_seed)
        if use_prior:
            u0 = rng.random((n_walkers, self.ndim))
        else:
            soln, opt_gp = self.minimize(self.gp)
            u0 = soln.x + 1e-5 * rng.standard_normal((n_walkers, self.ndim))
        sampler = emcee.EnsembleSampler(
            n_walkers, self.ndim, self.log_prob, args=(self.gp,)
        )
        sampler.run_mcmc(u0, n_steps, progress=True)
        samples = sampler.get_chain(discard=burn, flat=True)
        tau = sampler.get_autocorr_time(discard=burn, quiet=True)
        trace = self.prior_transform(samples.T)
        self.sampler = sampler
        return trace, tau


class BrownianTerm(celerite2.terms.TermSum):
    def __init__(self, sigma, tau, period, mix):
        Q = 0.01
        sigma_1 = sigma * np.sqrt(mix)
        f = np.sqrt(1 - 4 * Q ** 2)
        w0 = 2 * Q / (tau * (1 - f))
        S0 = (1 - mix) * sigma ** 2 / (0.5 * w0 * Q * (1 + 1 / f))
        super().__init__(
            celerite2.terms.SHOTerm(sigma=sigma_1, tau=tau, rho=period),
            celerite2.terms.SHOTerm(S0=S0, w0=w0, Q=Q),
        )


class BrownianGP(CeleriteModeler):
    def __init__(self, signal, err, init_period=None, period_prior=None):
        self.ndim = 6
        self.kernel = BrownianTerm
        super().__init__(signal, err, init_period, period_prior)

    def prior_transform(self, u):
        period = self.period_prior(u[3])
        params = {
            "mean": norm.ppf(u[0], self.mean, 10.0),
            "sigma": np.exp(norm.ppf(u[1], np.log(self.sigma), 5.0)),
            "tau": period * 10 ** u[2],
            "period": period,
            "mix": u[4] * 0.5,
            "jitter": np.exp(norm.ppf(u[5], np.log(self.jitter), 5.0)),
        }
        return params


class HarmonicGP(CeleriteModeler):
    def __init__(self, signal, err, init_period=None, period_prior=None):
        self.ndim = 7
        self.kernel = celerite2.terms.RotationTerm
        super().__init__(signal, err, init_period, period_prior)

    def prior_transform(self, u):
        period = self.period_prior(u[2])
        params = {
            "mean": norm.ppf(u[0], self.mean, 10.0),
            "sigma": np.exp(norm.ppf(u[1], np.log(self.sigma), 5.0)),
            "period": period,
            "Q0": np.exp(norm.ppf(u[3], 1.0, 5.0)),
            "dQ": np.exp(norm.ppf(u[4], 2.0, 5.0)),
            "f": u[5],
            "jitter": np.exp(norm.ppf(u[6], np.log(self.jitter), 5.0)),
        }
        return params


class TheanoModeler(object):
    def __init__(self, signal, err, init_period=None):
        if not isinstance(signal, TSeries):
            signal = TSeries(data=signal)
        self.signal = signal
        self.err = err
        self.t = self.signal.time
        self.y = self.signal.values
        self.sigma = np.std(self.y)
        self.jitter = np.min(self.err) ** 2
        self.mean = np.mean(self.y)
        if init_period is None:
            init_period = np.sqrt(signal.size) * signal.median_dt
        self.sigma_period = 0.5 * np.log(signal.size)
        self.init_period = init_period

    def mcmc(self, n_walkers=1, n_steps=2000, burn=1000, cores=1):
        with self.model:
            trace = pmx.sample(
                tune=burn,
                draws=n_steps - burn,
                cores=cores,
                chains=n_walkers,
                random_seed=42,
            )
            self.period_samples = trace["period"]
            return trace


class BrownianTheanoGP(TheanoModeler):
    def __init__(self, signal, err, init_period=None, predict_at=None, psd_at=None):
        super().__init__(signal, err, init_period)
        with pm.Model() as model:
            # The mean flux of the time series
            mean = pm.Normal("mean", mu=self.mean, sd=10.0)
            # A jitter term describing excess white noise
            log_jitter = pm.Normal("log_jitter", mu=np.log(self.jitter), sd=5.0)
            # The parameters of the BrownianTerm kernel
            sigma = pm.Lognormal("sigma", mu=np.log(self.sigma), sd=5.0)
            period = pm.Lognormal(
                "period", mu=np.log(self.init_period), sd=self.sigma_period
            )
            log_tau = pm.Uniform("log_tau", lower=0.0, upper=np.log(10))
            tau = pm.math.exp(log_tau) * period
            mix = pm.Uniform("mix", lower=0.0, upper=0.5)
            Q = 0.01
            sigma_1 = sigma * pm.math.sqrt(mix)
            f = pm.math.sqrt(1 - 4 * Q ** 2)
            w0 = 2 * Q / (tau * (1 - f))
            S0 = (1 - mix) * sigma ** 2 / (0.5 * w0 * Q * (1 + 1 / f))
            # Set up the Gaussian Process model
            kernel1 = celerite2.theano.terms.SHOTerm(sigma=sigma_1, tau=tau, rho=period)
            kernel2 = celerite2.theano.terms.SHOTerm(S0=S0, w0=w0, Q=Q)
            kernel = kernel1 + kernel2
            gp = celerite2.theano.GaussianProcess(kernel, mean=mean)
            gp.compute(self.t, diag=self.err ** 2 + pm.math.exp(log_jitter), quiet=True)
            gp.marginal("obs", observed=self.y)
            if predict_at is not None:
                pm.Deterministic("pred", gp.predict(self.y, predict_at))
            if psd_at is not None:
                pm.Deterministic("psd", kernel.get_psd(2 * np.pi * psd_at))
        self.model = model


class HarmonicTheanoGP(TheanoModeler):
    def __init__(self, signal, err, init_period=None, predict_at=None, psd_at=None):
        super().__init__(signal, err, init_period)
        with pm.Model() as model:
            # The mean flux of the time series
            mean = pm.Normal("mean", mu=self.mean, sd=10.0)
            # A jitter term describing excess white noise
            log_jitter = pm.Normal("log_jitter", mu=np.log(self.jitter), sd=5.0)
            # The parameters of the RotationTerm kernel
            sigma = pm.Lognormal("sigma", mu=np.log(self.sigma), sd=5.0)
            period = pm.Lognormal(
                "period", mu=np.log(self.init_period), sd=self.sigma_period
            )
            Q0 = pm.Lognormal("Q0", mu=1.0, sd=5.0)
            dQ = pm.Lognormal("dQ", mu=2.0, sd=5.0)
            f = pm.Uniform("f", lower=0.0, upper=1.0)
            # Set up the Gaussian Process model
            kernel = celerite2.theano.terms.RotationTerm(
                sigma=sigma,
                period=period,
                Q0=Q0,
                dQ=dQ,
                f=f,
            )
            gp = celerite2.theano.GaussianProcess(kernel, mean=mean)
            gp.compute(self.t, diag=self.err ** 2 + pm.math.exp(log_jitter), quiet=True)
            gp.marginal("obs", observed=self.y)
            if predict_at is not None:
                pm.Deterministic("pred", gp.predict(self.y, predict_at))
            if psd_at is not None:
                pm.Deterministic("psd", kernel.get_psd(2 * np.pi * psd_at))
        self.model = model
