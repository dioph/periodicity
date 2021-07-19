from multiprocessing import Pool

import numpy as np
from tqdm.auto import trange

from .core import Timeseries

__all__ = ["EMD", "CEEMDAN", "LMD", "VMD"]
# TODO: Variational Mode Decomposition (Dragomiretskiy & Zosso, 2014)


def _emd_sift(mode, n_rep):
    upper, lower = mode.get_envelope(n_rep=n_rep)
    mu = (upper + lower) / 2
    amp = (upper - lower) / 2
    sigma = np.abs(mu.val / amp.val)
    return mu, sigma


class EMD(object):
    def __init__(self, max_iter=2000, n_rep=2, theta_1=0.05, theta_2=0.50, alpha=0.05):
        """Empirical Mode Decomposition

        Parameters
        ----------
        max_iter: int, optional
            Maximum number of sifting iterations (the default is 2000).
        n_rep: int, optional
            Number of extrema to repeat on either side of the signal while
            interpolating envelopes (the default is 2).
        theta_1: float, optional
            Lower threshold for the evaluation function (the default is 0.05).
        theta_2: float, optional
            Upper threshold for the evaluation function (usually ``10 * theta_1``).
        alpha: float, optional
            Fraction of total duration where the evaluation function is allowed to
            be ``theta_1 < sigma < theta_2`` (the default is 0.05).

        References
        ----------
        .. [#] G. Rilling, P. Flandrin, P. Gonçalves, "On Empirical Mode
        Decomposition and its Algorithms," IEEE-EURASIP Workshop on Nonlinear
        Signal and Image Processing, June 2003.
        """
        self.max_iter = max_iter
        self.n_rep = n_rep
        self.theta_1 = theta_1
        self.theta_2 = theta_2
        self.alpha = alpha

    def __call__(self, signal, max_modes=None):
        """
        Parameters
        ----------
        max_modes: int, optional
            If given, stops the decomposition after this number of modes.
        """
        if not isinstance(signal, Timeseries):
            signal = Timeseries(val=signal)
        if max_modes is None:
            max_modes = np.inf
        imfs = []
        iters = []
        is_monotonic = signal.size < 4
        residue = signal.copy()
        while not is_monotonic and len(imfs) < max_modes:
            # Sifts next mode
            mode = residue.copy()
            is_imf = False
            it = 0
            while True:
                is_imf = False
                maxima, minima = mode.find_extrema()
                zeroes = mode.find_zero_crossings()
                n_ext = maxima.size + minima.size
                n_zero = zeroes.size
                it += 1
                try:
                    upper, lower = mode.get_envelope(n_rep=self.n_rep)
                except TypeError:
                    # If there are not enough extrema, the current mode is a monotonic residue
                    is_monotonic = True
                    break
                mu = (upper + lower) / 2
                amp = (upper - lower) / 2
                sigma = np.abs(mu.val / amp.val)
                # sigma < theta_1 for some prescribed fraction (1-alpha) of the total duration
                is_imf = np.mean(sigma > self.theta_1) < self.alpha
                # sigma < theta_2 for the remaining fraction
                is_imf = is_imf and np.all(sigma < self.theta_2)
                # The number of extrema and zero-crossings must differ at most by 1
                is_imf = is_imf and (np.abs(n_zero - n_ext) <= 1)
                # Checks stopping criteria BEFORE updating the current mode
                if is_imf or it >= self.max_iter:
                    break
                mode -= mu
            # Appends last IMF extracted, unless it was a monotonic residue
            if is_imf or it == self.max_iter:
                imfs.append(mode)
                iters.append(it)
                residue -= mode

        # Defines useful attributes
        self.signal = signal
        self.imfs = imfs
        self.iters = iters
        self.residue = residue
        self.n_imfs = len(imfs)
        return self.imfs


def _lmd_sift(sig, n_rep=0, smooth_iter=12):
    # define padded extrema
    peaks, dips = sig.find_extrema(include_edges=True)
    extrema = peaks[1:-1].join(dips)
    if n_rep > 0:
        t_extr = np.pad(extrema.time, n_rep, "reflect", reflect_type="odd")
        y_extr = np.pad(extrema.val, n_rep + 1, "symmetric", reflect_type="odd")
        t_extr = np.delete(t_extr, [n_rep, -n_rep - 1])
        y_extr = np.delete(y_extr, [n_rep, n_rep + 1, -n_rep - 1, -n_rep - 2])
        extrema = Timeseries(t_extr, y_extr)
    # zero-order hold local mean and envelope
    mu = 0.5 * (extrema[:-1] + extrema[1:])
    mu = mu.join(extrema[-1:]).fill_gaps(ts=sig.ts, kind="hold")
    mu.val[-1] = mu.val[-2]
    env = 0.5 * (extrema[:-1] - extrema[1:]).abs()
    env = env.join(extrema[-1:]).fill_gaps(ts=sig.ts, kind="hold")
    env.val[-1] = env.val[-2]
    # smooth local mean and envelope
    window = np.max(np.diff(extrema.time) / sig.ts) // 3
    window = max(3, window + (1 - window % 2))
    half = int(window // 2)
    weight = np.array(list(range(1, half + 2)) + list(range(half, 0, -1)))
    weight = weight / weight.sum()
    for it in range(smooth_iter):
        mu = mu.smooth(weight)
        if np.all(mu.val[1:] != mu.val[:-1]):
            break
    for it in range(smooth_iter):
        env = env.smooth(weight)
        if np.all(env.val[1:] != env.val[:-1]):
            break
    # remove padding
    start = np.argmin(np.abs(mu.time - sig.time[0]))
    mu = mu[start : start + sig.size].timeshift(sig.time[0] - mu.time[start])
    env = env[start : start + sig.size].timeshift(sig.time[0] - env.time[start])
    return mu, env


class LMD(object):
    def __init__(self, max_iter=10, n_rep=0, smooth_iter=12, eps=1e-6):
        self.max_iter = max_iter
        self.n_rep = n_rep
        self.smooth_iter = smooth_iter
        self.eps = eps

    def __call__(self, signal, max_modes=None):
        if not isinstance(signal, Timeseries):
            signal = Timeseries(val=signal)
        if max_modes is None:
            max_modes = np.inf
        pfs = []
        is_monotonic = signal.size < 4
        residue = signal.copy()
        while not is_monotonic and len(pfs) < max_modes:
            # Sifts next mode
            F = residue.copy()
            A = 1.0
            for it in range(self.max_iter):
                try:
                    mu, env = _lmd_sift(F, self.n_rep, self.smooth_iter)
                except TypeError:
                    is_monotonic = True
                    break
                F = (F - mu) / env
                A = A * env
                if np.max(np.abs(F.val)) - 1.0 < self.eps:
                    break
            F.val[F.val > 1.0] = 1.0
            F.val[F.val < -1.0] = -1.0
            if not is_monotonic:
                pfs.append([A, F])
                residue -= A * F
        # Defines useful attributes
        self.signal = signal
        self.pfs = pfs
        self.residue = residue
        self.n_pfs = len(pfs)
        return self.pfs


class VMD(object):
    pass


class CEEMDAN(object):
    def __init__(
        self,
        epsilon=0.2,
        ensemble_size=50,
        min_energy=0.0,
        random_seed=None,
        cores=None,
        **kwargs,
    ):
        """Complete Ensemble Empirical Mode Decomposition with Adaptive Noise

        Parameters
        ----------
        max_modes: int, optional
            If given, stops the decomposition after this number of modes.
        epsilon: float, optional
            Normalized standard deviation of the added Gaussian noise (the default is 0.2).
        ensemble_size: int, optional
            Number of realizations averaged for each IMF (the default is 50).
        min_energy: float, optional
            Stopping criterium for the energy contribution of the residue.
        random_seed: int, optional
            Seed for generating random numbers, useful for reproducibility of results.

        References
        ----------
        .. [#] M. Colominas, G. Schlotthauer, M. Torres, "Improved complete
        ensemble EMD: A suitable tool for biomedical signal processing,"
        Biomedical Signal Processing and Control, 2014.
        .. [#] M. Torres, M. Colominas, G. Schlotthauer, P. Flandrin, "A Complete
        Ensemble Empirical Mode Decomposition with Adaptive Noise," 36th
        International Conference on Acoustics, Speech and Signal Processing
        ICASSP, 2011.
        """
        self.epsilon = epsilon
        self.ensemble_size = ensemble_size
        self.min_energy = min_energy
        self.cores = cores
        self.emd = EMD(**kwargs)
        self.rng = np.random.default_rng(random_seed)

    def _realization(self, task):
        noise_modes, k, residue = task
        noisy_residue = residue.copy()
        if len(noise_modes) > k:
            beta = self.epsilon * np.std(residue.val)
            if k == 0:
                beta /= np.std(noise_modes[k].val)
            noisy_residue += beta * noise_modes[k]
        try:
            mode = self.emd(noisy_residue, max_modes=1)[0]
        except IndexError:
            # in case noisy_residue happens to be monotonic even though residue was not
            mode = noisy_residue.copy()
        return noisy_residue - mode

    def __call__(self, signal, max_modes=None, progress=False):
        if not isinstance(signal, Timeseries):
            signal = Timeseries(val=signal)
        if max_modes is None:
            max_modes = np.inf
        sigma_x = np.std(signal.val)

        # Creates the noise realizations
        white_noise_modes = []
        if self.cores is not None:
            with Pool(self.cores) as pool:
                white_noise = [
                    Timeseries(signal.time, self.rng.standard_normal(signal.size))
                    for i in range(self.ensemble_size)
                ]
                white_noise_modes = pool.map(self.emd, white_noise)
                if progress:
                    print("White noise: done")
        else:
            if progress:
                ensemble = trange(self.ensemble_size, desc="White noise", leave=True)
            else:
                ensemble = range(self.ensemble_size)
            for i in ensemble:
                white_noise = Timeseries(
                    signal.time, self.rng.standard_normal(signal.size)
                )
                white_noise_modes.append(self.emd(white_noise))

        imfs = []
        residue = signal / sigma_x
        while len(imfs) < max_modes:
            k = len(imfs)

            # Averages the ensemble of trials for the next mode
            mu = 0
            if self.cores is not None:
                with Pool(self.cores) as pool:
                    tasks = [
                        (noise_modes, k, residue) for noise_modes in white_noise_modes
                    ]
                    mus = pool.map(self._realization, tasks)
                mu = sum(mus) / self.ensemble_size
                if progress:
                    print(f"Mode #{k+1}: done")
            else:
                if progress:
                    ensemble = trange(
                        self.ensemble_size, desc=f"Mode #{k+1}", leave=True
                    )
                for i in ensemble:
                    noise_modes = white_noise_modes[i]
                    mu += (
                        self._realization((noise_modes, k, residue))
                        / self.ensemble_size
                    )
            imfs.append(residue - mu)
            residue = mu.copy()

            # Checks stopping criteria (if the residue is an IMF or too small)
            if np.var(residue.val) < self.min_energy:
                break
            residue_imfs = self.emd(residue)
            if len(residue_imfs) <= 1:
                if len(imfs) < max_modes and len(residue_imfs) == 1:
                    imfs.append(residue)
                break

        # Undoes the initial normalization
        for i in range(len(imfs)):
            imfs[i] *= sigma_x
        self.signal = signal
        self.imfs = imfs
        self.residue = signal - sum(imfs)
        self.n_imfs = len(imfs)
        return self.imfs

    def postprocessing(self):
        ck = self.emd(self.imfs[0], max_modes=1)[0]
        c_imfs = [ck]
        qk = self.imfs[0] - ck
        for k in range(1, len(self.imfs)):
            Dk = qk + self.imfs[k]
            modes = self.emd(Dk, max_modes=1)
            if len(modes) > 0:
                ck = modes[0]
            else:
                c_imfs.append(self.imfs[k])
                break
            qk = Dk - ck
            c_imfs.append(ck)
        self.c_residue = sum(self.imfs) + self.residue - sum(c_imfs)
        self.c_imfs = c_imfs

    @property
    def orthogonality_matrix(self):
        orth = np.zeros((self.n_imfs, self.n_imfs), float)
        for i in range(self.n_imfs):
            for j in range(self.n_imfs):
                orth[i, j] = np.corrcoef(self.imfs[i].val, self.imfs[j].val)[0, 1]
        return orth

    @property
    def c_orthogonality_matrix(self):
        orth = np.zeros((len(self.c_imfs), len(self.c_imfs)), float)
        for i in range(len(self.c_imfs)):
            for j in range(len(self.c_imfs)):
                orth[i, j] = np.corrcoef(self.c_imfs[i].val, self.c_imfs[j].val)[0, 1]
        return orth
