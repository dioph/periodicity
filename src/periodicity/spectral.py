import copy

import numpy as np

from .core import FSeries, TSeries

__all__ = ["GLS", "BGLST"]
# TODO: check out Supersmoother (Reimann 1994)


def _trig_sum(t, w, df, nf, fmin, n=5):
    """
    Computes
        S_j = sum_i w_i * sin(2 pi * f_j * t_i),
        C_j = sum_i w_i * cos(2 pi * f_j * t_i)
    using an FFT-based O[Nlog(N)] method.
    """
    nfft = 1 << int(nf * n - 1).bit_length()
    tmin = t.min()
    w = w * np.exp(2j * np.pi * fmin * (t - tmin))
    tnorm = ((t - tmin) * nfft * df) % nfft
    grid = np.zeros(nfft, dtype=w.dtype)
    integers = tnorm % 1 == 0
    np.add.at(grid, tnorm[integers].astype(int), w[integers])
    tnorm, w = tnorm[~integers], w[~integers]
    ilo = np.clip((tnorm - 2).astype(int), 0, nfft - 4)
    numerator = w * np.prod(tnorm - ilo - np.arange(4)[:, np.newaxis], 0)
    denominator = 6
    for j in range(4):
        if j > 0:
            denominator *= j / (j - 4)
        ind = ilo + (3 - j)
        np.add.at(grid, ind, numerator / (denominator * (tnorm - ind)))
    fftgrid = np.fft.ifft(grid)[:nf]
    if tmin != 0:
        f = fmin + df * np.arange(nf)
        fftgrid *= np.exp(2j * np.pi * tmin * f)
    C = nfft * fftgrid.real
    S = nfft * fftgrid.imag
    return S, C


class GLS(object):
    """
    References
    ----------
    .. [1] Press W.H. and Rybicki, G.B, "Fast algorithm for spectral analysis
        of unevenly sampled data". ApJ 1:338, p277, 1989
    .. [2] M. Zechmeister and M. Kurster, A&A 496, 577-584 (2009)
    .. [3] W. Press et al, Numerical Recipes in C (2002)
    """

    def __init__(self, fmin=None, fmax=None, n=5, psd=False):
        """Computes the generalized Lomb-Scargle periodogram of a discrete signal.

        Parameters
        ----------
        fmin: float, optional
            Minimum frequency.
            If not given, it will be determined from the time baseline.
        fmax: float, optional
            Maximum frequency.
            If not given, defaults to the pseudo-Nyquist limit.
        n: float, optional
            Samples per peak (default is 5).
        psd: bool, optional
            Whether to leave periodogram non-normalized (Fourier Spectral Density).
        """
        self.fmin = fmin
        self.fmax = fmax
        self.n = n
        self.psd = psd

    def __call__(self, signal, err=None, fit_mean=True):
        """Fast implementation of the Lomb-Scargle periodogram.
        Based on the lombscargle_fast implementation of the astropy package.
        Assumes an uniform frequency grid, but the errors can be heteroscedastic.

        Parameters
        ----------
        err: array-like, optional
            Measurement uncertainties for each sample.
        fit_mean: bool, optional
            If True, then let the mean vary with the fit.
        """
        if not isinstance(signal, TSeries):
            signal = TSeries(values=signal)
        df = 1.0 / signal.baseline / self.n
        if self.fmin is None:
            fmin = 0.5 * df
        else:
            fmin = self.fmin
        if self.fmax is None:
            fmax = 0.5 / signal.median_dt
        else:
            fmax = self.fmax
        self.frequency = np.arange(fmin, fmax + df, df)
        nf = self.frequency.size
        if err is None:
            err = np.ones_like(signal.values)
        self.err = err
        w = err ** -2.0
        w /= w.sum()
        t = signal.time
        if fit_mean:
            y = signal.values - np.dot(w, signal.values)
        else:
            y = signal.values
        Sh, Ch = _trig_sum(t, w * y, df, nf, fmin)
        S2, C2 = _trig_sum(t, w, 2 * df, nf, 2 * fmin)
        if fit_mean:
            S, C = _trig_sum(t, w, df, nf, fmin)
            tan_2omega_tau = (S2 - 2 * S * C) / (C2 - (C * C - S * S))
        else:
            tan_2omega_tau = S2 / C2
        S2w = tan_2omega_tau / np.sqrt(1 + tan_2omega_tau * tan_2omega_tau)
        C2w = 1 / np.sqrt(1 + tan_2omega_tau * tan_2omega_tau)
        Cw = np.sqrt(0.5) * np.sqrt(1 + C2w)
        Sw = np.sqrt(0.5) * np.sign(S2w) * np.sqrt(1 - C2w)
        YY = np.dot(w, y ** 2)
        YC = Ch * Cw + Sh * Sw
        YS = Sh * Cw - Ch * Sw
        CC = 0.5 * (1 + C2 * C2w + S2 * S2w)
        SS = 0.5 * (1 - C2 * C2w - S2 * S2w)
        if fit_mean:
            CC -= (C * Cw + S * Sw) ** 2
            SS -= (S * Cw - C * Sw) ** 2
        power = YC * YC / CC + YS * YS / SS
        if self.psd:
            power *= 0.5 * (err ** -2.0).sum()
        else:
            power /= YY
        self.signal = signal
        self.periodogram = FSeries(self.frequency, power)
        return self.periodogram

    def copy(self):
        return copy.deepcopy(self)

    def bootstrap(self, n_bootstraps, random_seed=None):
        rng = np.random.default_rng(random_seed)
        bs_replicates = np.empty(n_bootstraps)
        ndata = len(self.signal)
        gls = self.copy()
        for i in range(n_bootstraps):
            bs_sample = self.signal.copy()
            bs_sample.values = self.signal.values[rng.integers(0, ndata, ndata)]
            bs_replicates[i] = gls(bs_sample).max()
        self.bs_replicates = bs_replicates
        return self.bs_replicates

    def fap(self, power):
        """
        fap_level: array-like, optional
            List of false alarm probabilities for which you want to calculate
            approximate levels. Can also be passed as a single scalar value.
        """
        return np.mean(power < self.bs_replicates)

    def fal(self, fap):
        return np.quantile(self.bs_replicates, 1 - fap)

    def window(self):
        gls = self.copy()
        return gls(0.0 * self.signal + 1.0, fit_mean=False)

    def model(self, tf, f0):
        """Compute the Lomb-Scargle model fit at a given frequency

        Parameters
        ----------
        tf: float or array-like
            The times at which the fit should be computed
        f0: float
            The frequency at which to compute the model

        Returns
        -------
        yf: ndarray
            The model fit evaluated at each value of tf
        """
        t = self.signal.time
        y = self.signal.values
        w = self.err ** -2.0
        y_mean = np.dot(y, w) / w.sum()
        y = y - y_mean
        X = (
            np.vstack(
                [
                    np.ones_like(t),
                    np.sin(2 * np.pi * f0 * t),
                    np.cos(2 * np.pi * f0 * t),
                ]
            )
            / self.err
        )
        theta = np.linalg.solve(np.dot(X, X.T), np.dot(X, y / self.err))
        Xf = np.vstack(
            [np.ones_like(tf), np.sin(2 * np.pi * f0 * tf), np.cos(2 * np.pi * f0 * tf)]
        )
        yf = y_mean + np.dot(Xf.T, theta)
        return TSeries(tf, yf)


class BGLST(object):
    pass


'''
class BGLST:
    """Bayesian linear regression with harmonic, offset and trend
    y = A*sin(2*pi*f*t-tau)+B*sin(2*pi*f*t-tau)+alpha*t+beta+epsilon
    """

    def __init__(
        self,
        t,
        y,
        w,
        w_A=0.0,
        A_hat=0.0,
        w_B=0.0,
        B_hat=0.0,
        w_alpha=0.0,
        alpha_hat=0.0,
        w_beta=0.0,
        beta_hat=0.0,
    ):
        """
        Args:
            t (:obj:`array` of :obj:`float`): The array of time moments of the data points
            y (:obj:`array` of :obj:`float`): The array of y-values of the data points
            w (:obj:`array` of :obj:`float`): The array of weights of the data points (i.e. the precisions or inverse of the variances)
            w_A (float): prior weight of A (i.e. the precision or inverse of the variance)
            A_hat (float): prior mean of A
            w_B (float): prior weight of B
            B_hat (float): prior mean of B
            w_alpha (float): prior weight of alpha
            alpha_hat (float): prior mean of alpha
            w_beta (float): prior weight of beta
            beta_hat (float): prior mean of beta

        """
        self.t = t
        self.y = y
        self.w = w
        self.wy_arr = w * y
        self.wt_arr = w * t
        self.yy = sum(self.wy_arr * y)
        self.Y = sum(self.wy_arr) + w_beta * beta_hat
        self.W = sum(w) + w_beta
        self.tt = sum(self.wt_arr * t) + w_alpha
        self.T = sum(self.wt_arr)
        self.yt = sum(self.wy_arr * t) + w_alpha * alpha_hat
        self.two_pi_t = 2.0 * np.pi * t
        self.four_pi_t = 4.0 * np.pi * t
        self.norm_term = sum(np.log(np.sqrt(w)) - np.log(np.sqrt(2.0 * np.pi)))
        self.norm_term_ll = self.norm_term
        if w_A > 0:
            self.norm_term += (
                np.log(np.sqrt(w_A))
                - np.log(np.sqrt(2.0 * np.pi))
                - 0.5 * w_A * A_hat ** 2
            )
        if w_B > 0:
            self.norm_term += (
                np.log(np.sqrt(w_B))
                - np.log(np.sqrt(2.0 * np.pi))
                - 0.5 * w_B * B_hat ** 2
            )
        if w_alpha > 0:
            self.norm_term += (
                np.log(np.sqrt(w_alpha))
                - np.log(np.sqrt(2.0 * np.pi))
                - 0.5 * w_alpha * alpha_hat ** 2
            )
        if w_beta > 0:
            self.norm_term += (
                np.log(np.sqrt(w_beta))
                - np.log(np.sqrt(2.0 * np.pi))
                - 0.5 * w_beta * beta_hat ** 2
            )
        self.w_A = w_A
        self.A_hat = A_hat
        self.w_B = w_B
        self.B_hat = B_hat
        self.alpha_hat = alpha_hat
        self.w_alpha = w_alpha
        self.beta_hat = beta_hat
        self.w_beta = w_beta

    def _linreg(self):
        W = sum(self.w)
        wt_arr = self.w * self.t
        tau = sum(wt_arr) / W
        wy_arr = self.w * self.y

        yt = sum(wy_arr * (self.t - tau))
        Y = sum(wy_arr)
        tt = sum(wt_arr * (self.t - tau))

        sigma_alpha = 1.0 / tt
        mu_alpha = yt * sigma_alpha

        sigma_beta = 1.0 / W
        mu_beta = Y * sigma_beta - mu_alpha * tau

        y_model = self.t * mu_alpha + mu_beta
        loglik = self.norm_term_ll - 0.5 * sum(self.w * (self.y - y_model) ** 2)
        return ((mu_alpha, mu_beta), (sigma_alpha, sigma_beta), y_model, loglik)

    def calc(self, freq):
        s_arr_1 = np.sin(self.two_pi_t * freq)
        c_arr_1 = np.cos(self.two_pi_t * freq)
        return self._calc(s_arr_1, c_arr_1)

    def _calc(self, s_arr_1, c_arr_1):
        s_2_arr = 2.0 * s_arr_1 * c_arr_1
        c_2_arr = c_arr_1 * c_arr_1 - s_arr_1 * s_arr_1
        tau = 0.5 * np.arctan(sum(self.w * s_2_arr) / sum(self.w * c_2_arr))

        cos_tau = np.cos(tau)
        sin_tau = np.sin(tau)
        s_arr = s_arr_1 * cos_tau - c_arr_1 * sin_tau
        c_arr = c_arr_1 * cos_tau + s_arr_1 * sin_tau
        wc_arr = self.w * c_arr
        ws_arr = self.w * s_arr
        c = sum(wc_arr)
        s = sum(ws_arr)
        cc = sum(wc_arr * c_arr) + self.w_A
        ss = sum(ws_arr * s_arr) + self.w_B
        ct = sum(wc_arr * self.t)
        st = sum(ws_arr * self.t)
        yc = sum(self.wy_arr * c_arr) + self.w_A * self.A_hat
        ys = sum(self.wy_arr * s_arr) + self.w_B * self.B_hat
        if ss == 0.0:
            assert cc > 0
            K = (ct ** 2 / cc - self.tt) / 2.0
            assert K < 0
            M = self.yt - yc * ct / cc
            N = ct * c / cc - self.T
            P = (c ** 2 / cc - self.W - N ** 2 / 2.0 / K) / 2.0
            Q = -yc * c / cc + self.Y - M * N / 2.0 / K
            if P == 0.0:
                if Q != 0.0:
                    print("WARNING: Q=", Q)
                assert abs(Q) < 1e-8
                log_prob = (
                    self.norm_term
                    + np.log(2.0 * np.pi ** 2 / np.sqrt(-cc * K))
                    + (yc ** 2 / 2.0 / cc - M ** 2 / 4.0 / K - self.yy / 2.0)
                )
            else:
                assert P < 0
                log_prob = (
                    self.norm_term
                    + np.log(2.0 * np.pi ** 2 / np.sqrt(cc * K * P))
                    + (
                        yc ** 2 / 2.0 / cc
                        - M ** 2 / 4.0 / K
                        - Q ** 2 / 4.0 / P
                        - self.yy / 2.0
                    )
                )
        elif cc == 0.0:
            assert ss > 0
            K = (st ** 2 / ss - self.tt) / 2.0
            assert K < 0
            M = self.yt - ys * st / ss
            N = st * s / ss - self.T
            P = (s ** 2 / ss - self.W - N ** 2 / 2.0 / K) / 2.0
            Q = -ys * s / ss + self.Y - M * N / 2.0 / K
            if P == 0.0:
                if Q != 0.0:
                    print("WARNING: Q=", Q)
                log_prob = (
                    self.norm_term
                    + np.log(2.0 * np.pi ** 2 / np.sqrt(-ss * K))
                    + (ys ** 2 / 2.0 / ss - M ** 2 / 4.0 / K - self.yy / 2.0)
                )
            else:
                assert P < 0
                log_prob = (
                    self.norm_term
                    + np.log(2.0 * np.pi ** 2 / np.sqrt(ss * K * P))
                    + (
                        ys ** 2 / 2.0 / ss
                        - M ** 2 / 4.0 / K
                        - Q ** 2 / 4.0 / P
                        - self.yy / 2.0
                    )
                )
        else:
            assert cc > 0
            assert ss > 0
            K = (ct ** 2 / cc + st ** 2 / ss - self.tt) / 2.0
            assert K < 0
            M = self.yt - yc * ct / cc - ys * st / ss
            N = ct * c / cc + st * s / ss - self.T
            P = (c ** 2 / cc + s ** 2 / ss - self.W - N ** 2 / 2.0 / K) / 2.0
            Q = -yc * c / cc - ys * s / ss + self.Y - M * N / 2.0 / K
            if P == 0.0:
                if Q != 0.0:
                    print("WARNING: Q=", Q)
                log_prob = (
                    self.norm_term
                    + np.log(2.0 * np.pi ** 2 / np.sqrt(-cc * ss * K))
                    + (
                        yc ** 2 / 2.0 / cc
                        + ys ** 2 / 2.0 / ss
                        - M ** 2 / 4.0 / K
                        - self.yy / 2.0
                    )
                )
            elif P > 0 and abs(P) < 1e-6:
                print("ERROR: P=", P)
                if Q != 0.0:
                    print("WARNING: Q=", Q)
                log_prob = (
                    self.norm_term
                    + np.log(2.0 * np.pi ** 2 / np.sqrt(-cc * ss * K))
                    + (
                        yc ** 2 / 2.0 / cc
                        + ys ** 2 / 2.0 / ss
                        - M ** 2 / 4.0 / K
                        - self.yy / 2.0
                    )
                )
            else:
                if P > 0:
                    print("ERROR: P=", P)
                assert P < 0
                log_prob = (
                    self.norm_term
                    + np.log(2.0 * np.pi ** 2 / np.sqrt(cc * ss * K * P))
                    + (
                        yc ** 2 / 2.0 / cc
                        + ys ** 2 / 2.0 / ss
                        - M ** 2 / 4.0 / K
                        - Q ** 2 / 4.0 / P
                        - self.yy / 2.0
                    )
                )
        return (log_prob, (tau, ss, cc, ys, yc, st, ct, s, c, K, M, N, P, Q))

    def step(self, n):
        if n == 0 and all(self.sin_0 == 0):
            # zero frequency
            (_, _, _, loglik) = self._linreg()
            return loglik

        if n > 1:
            sin_n_delta = self.two_cos_delta * self.sin_n_1_delta - self.sin_n_2_delta
            cos_n_delta = self.two_cos_delta * self.cos_n_1_delta - self.cos_n_2_delta
        elif n == 1:
            sin_n_delta = np.sin(self.delta)
            cos_n_delta = np.cos(self.delta)
        else:  # n == 0:
            sin_n_delta = np.zeros(len(self.t))
            cos_n_delta = np.ones(len(self.t))

        self.sin_n_2_delta = self.sin_n_1_delta
        self.cos_n_2_delta = self.cos_n_1_delta
        self.sin_n_1_delta = sin_n_delta
        self.cos_n_1_delta = cos_n_delta

        s_arr_1 = self.sin_0 * cos_n_delta + self.cos_0 * sin_n_delta
        c_arr_1 = self.cos_0 * cos_n_delta - self.sin_0 * sin_n_delta

        log_prob, _ = self._calc(s_arr_1, c_arr_1)
        return log_prob

    def calc_all(self, freq_start, freq_end, count):
        """Calculates the probabilistic spectrum for the given range of frequencies

        Returns: (freqs, probs), where
            freqs is the array of frequencies
            probs is the corresponding array of unnormalized log probabilities
        """
        self.freq_start = freq_start
        delta_freq = (freq_end - freq_start) / count
        self.delta = self.two_pi_t * delta_freq
        self.two_cos_delta = 2.0 * np.cos(self.delta)
        self.sin_n_1_delta = np.zeros(len(self.t))
        self.cos_n_1_delta = np.ones(len(self.t))

        self.sin_0 = np.sin(self.two_pi_t * freq_start)
        self.cos_0 = np.cos(self.two_pi_t * freq_start)

        freqs = np.zeros(count)
        probs = np.zeros(count)
        for n in np.arange(0, count):
            freqs[n] = freq_start + delta_freq * n
            probs[n] = self.step(n)
        probs -= scipy.special.logsumexp(probs) + np.log(delta_freq)
        return (freqs, probs)

    def model(
        self, freq, t=None, w=None, calc_pred_var=False, pred_var_with_obs_noise=False
    ):
        """Calculates the regression model at given time moments and weights using a given frequency.

        Args:
            freq (float): The frequency used in the regression model
            t (:obj:`array` of :obj:`float`): The array of time moments for which the
                model values to calculate. If omitted the values are calculated for the
                time moments of the data.
            w (:obj:`array` of :obj:`float`): The array of weights corresponding to time moments
                in t. If omitted the weights of the data points are used. Otherwise must be of equal length to t.
            calc_pred_var (bool): Whether to calculate the predictive variance

        Returns: (tau, mean, cov, y_model, loglik, pred_cov), where
            tau is the time shift in the model
            mean is the mean vector of the regression coefficients A, B, alpha and beta
            cov is the covariance matrix of the regression coefficients
            y_model is the array of predictive means (for a given fixed frequency)
            loglik is the log likelihood of the data given the model
            y_cov is the array of predictive variances corresponding to y_model (for a given fixed frequency) or None, if calc_pred_var = False

        """
        if freq == 0.0:
            (
                (mu_alpha, mu_beta),
                (sigma_alpha, sigma_beta),
                y_model,
                loglik,
            ) = self._linreg()
            return (
                0.0,
                (0.0, 0.0, mu_alpha, mu_beta),
                (0.0, 0.0, sigma_alpha, sigma_beta),
                y_model,
                loglik,
            )

        _, params = self.calc(freq)
        tau, ss, cc, ys, yc, st, ct, s, c, K, M, N, P, Q = params

        if P == 0.0:
            sigma_beta = 0.0
            mu_beta = 0.0
        else:
            sigma_beta = -0.5 / P
            mu_beta = Q * sigma_beta

        L = M + N * mu_beta

        sigma_alpha_beta = -0.5 * sigma_beta * N / K
        sigma_alpha = -0.5 / K + 0.25 * sigma_beta * N * N / K / K
        mu_alpha = -0.5 * L / K

        BC = yc - mu_alpha * ct - mu_beta * c
        BS = ys - mu_alpha * st - mu_beta * s

        if cc == 0.0:
            sigma_A = 0.0
            mu_A = 0.0
        else:
            sigma_A = 1.0 / cc
            mu_A = BC * sigma_A

        if ss == 0.0:
            sigma_B = 0.0
            mu_B = 0.0
        else:
            sigma_B = 1.0 / ss
            mu_B = BS * sigma_B

        det1 = (sigma_alpha * sigma_beta - sigma_alpha_beta ** 2) / ss
        denom = sigma_alpha_beta ** 2 - sigma_alpha * sigma_beta
        sigma_B_alpha = det1 * (st * sigma_alpha + s * sigma_alpha_beta) / denom
        sigma_B_beta = det1 * (st * sigma_alpha_beta + s * sigma_beta) / denom
        sigma_B = (det1 * (self.tt - ct ** 2 / cc) + sigma_B_beta ** 2) / sigma_beta

        det2 = (sigma_alpha * sigma_beta - sigma_alpha_beta ** 2) / cc
        sigma_A_alpha = det2 * (ct * sigma_alpha + c * sigma_alpha_beta) / denom
        sigma_A_beta = det2 * (ct * sigma_alpha_beta + c * sigma_beta) / denom
        sigma_A = (det2 * (self.tt - st ** 2 / ss) + sigma_A_beta ** 2) / sigma_beta

        term_A_B_1 = sigma_A_alpha * (
            sigma_B_alpha * sigma_beta - sigma_B_beta * sigma_alpha_beta
        )
        term_A_B_2 = sigma_A_beta * (
            sigma_B_beta * sigma_alpha - sigma_B_alpha * sigma_alpha_beta
        )
        sigma_A_B = (term_A_B_1 + term_A_B_2) / (
            sigma_alpha * sigma_beta - sigma_alpha_beta ** 2
        )

        y_model = (
            np.cos(self.t * 2.0 * np.pi * freq - tau) * mu_A
            + np.sin(self.t * 2.0 * np.pi * freq - tau) * mu_B
            + self.t * mu_alpha
            + mu_beta
        )
        loglik = self.norm_term_ll - 0.5 * sum(self.w * (self.y - y_model) ** 2)
        if t is None:
            t = self.t
            w = self.w
        y_model = (
            np.cos(t * 2.0 * np.pi * freq - tau) * mu_A
            + np.sin(t * 2.0 * np.pi * freq - tau) * mu_B
            + t * mu_alpha
            + mu_beta
        )
        mean = np.array([mu_A, mu_B, mu_alpha, mu_beta])
        cov = np.array(
            [
                [sigma_A, sigma_A_B, sigma_A_alpha, sigma_A_beta],
                [sigma_A_B, sigma_B, sigma_B_alpha, sigma_B_beta],
                [sigma_A_alpha, sigma_B_alpha, sigma_alpha, sigma_alpha_beta],
                [sigma_A_beta, sigma_B_beta, sigma_alpha_beta, sigma_beta],
            ]
        )

        pred_var = None
        if calc_pred_var:
            n = len(t)
            X = np.column_stack(
                (
                    np.cos(t * 2.0 * np.pi * freq - tau),
                    np.sin(t * 2.0 * np.pi * freq - tau),
                    t,
                    np.ones(n),
                )
            )
            pred_var = np.einsum("ij,ij->j", X.T, np.dot(cov, X.T))
            if pred_var_with_obs_noise:
                assert n == len(w)
                pred_var += np.ones(n) / w

        return (tau, mean, cov, y_model, loglik, pred_var)

    def fit(self, tau, freq, A, B, alpha, beta, t=None):
        """Fit a model with given parameters to the data
        Args:
            tau (float): Time shift in the model
            freq (float): Frequency used in the regression model
            A, B, alpha, beta (float):  Regression coefficients
            t (:obj:`array` of :obj:`float`): The array of time moments for which the
                model values to calculate. If omitted the values are calculated for the
                time moments of the data.
            calc_pred_cov (bool): Whether to calculate the predictive covariance

        Returns: y_model, loglik, where
            y_model is the predictive mean (for a given fixed frequency)
            loglik is the log likelihood of the data given the model


        """
        y_model = (
            np.cos(self.t * 2.0 * np.pi * freq - tau) * A
            + np.sin(self.t * 2.0 * np.pi * freq - tau) * B
            + self.t * alpha
            + beta
        )
        loglik = self.norm_term_ll - 0.5 * sum(self.w * (self.y - y_model) ** 2)
        if t is None:
            t = self.t
        if np.any(t != self.t):
            y_model = (
                np.cos(t * 2.0 * np.pi * freq - tau) * A
                + np.sin(t * 2.0 * np.pi * freq - tau) * B
                + t * alpha
                + beta
            )
        return y_model, loglik
'''
