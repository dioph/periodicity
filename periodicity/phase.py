# implements phase-folding methods such as:
# - string length (Dworetsky 1983)
# - analysis of variance (Schwarzenberg-Czerny 1989)
# - phase dispersion minimization (Stellingwerf 1978)
# - Gregory-Loredo method (Gregory & Loredo 1992)
# - conditional entropy method (Graham et al. 2013)

import numpy as np


def stringlength(t, x, dphi=0.1, n_periods=1000):
    """String Length

    Parameters
    ----------
    t:

    x:

    dphi: float (optional default=0.1)

    n_periods: int (optional default=1000)

    Returns
    -------
    periods:

    L:

    """
    # scale x to range from -0.25 to +0.25
    x = (x - x.min()) / (2 * (x.max() - x.min())) - 0.25
    df = dphi / (np.max(t) - np.min(t))
    periods = 1 / np.arange(df, n_periods*df+df, df)
    periods.sort()
    L = []
    for period in periods:
        phi = ((t / period) % 1)
        sorted_args = np.argsort(phi)
        phi = phi[sorted_args]
        m = x[sorted_args]
        ll = np.sqrt(np.square(np.append(m[1:], m[0])-m) + np.square(np.append(phi[1:], phi[0])-phi)).sum()
        L.append(ll)
    # TODO: consider flagging false periods for rejection
    L = np.array(L)
    return periods, L

# TODO: Analysis of Variance (Schwarzenberg-Czerny 1989)

# TODO: Phase Dispersion Minimization (Stellingwerf 1978)

# TODO: Gregory-Loredo method (Gregory & Loredo 1992)

# TODO: conditional entropy method (Graham et al. 2013)
