import os

import numpy as np


def lightcurve1():
    """
    Returns
    -------
    data: ndarray
        KIC 9895037 light curve, used for testing and demonstration.

    Examples
    --------
    >>> import periodicity.data
    >>> t, y, dy = periodicity.data.lightcurve1()
    >>> y.shape == (2145,)
    True
    """
    filename = os.path.join(os.path.dirname(__file__), "lightcurve1.npy")
    data = np.load(filename)
    return data


def lightcurve2():
    """
    Returns
    -------
    data: ndarray
        KIC 9655172 light curve, used for testing and demonstration.

    Examples
    --------
    >>> import periodicity.data
    >>> t, y, dy = periodicity.data.lightcurve2()
    >>> y.shape == (2148,)
    True
    """
    filename = os.path.join(os.path.dirname(__file__), "lightcurve2.npy")
    data = np.load(filename)
    return data
