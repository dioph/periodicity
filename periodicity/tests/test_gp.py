from .. import gp
from astropy.io import ascii

lightcurve1 = ascii.read('periodicity/tests/data/lightcurve1.csv')
lightcurve2 = ascii.read('periodicity/tests/data/lightcurve2.csv')


def test_file_format_lightcurve1():
    assert lightcurve1.colnames == ['time', 'flux', 'flux_err']
    assert lightcurve1['flux'].size == lightcurve1['time'].size
    assert lightcurve1['time'].size == 2145


def test_file_format_lightcurve2():
    assert lightcurve2.colnames == ['time', 'flux', 'flux_err']
    assert lightcurve2['flux'].size == lightcurve2['time'].size
    assert lightcurve2['time'].size == 2148


def test_make_gaussian_prior():
    prior = gp.make_gaussian_prior(lightcurve1['time'], lightcurve1['flux'], pmin=2)
    logp = gp.np.linspace(-3, 5, 1000)
    probs = prior(logp)
    assert probs.argmax() == 777
    peaks = [i for i in range(1, len(logp) - 1) if probs[i - 1] < probs[i] and probs[i + 1] < probs[i]]
    assert len(peaks) == 3


def test_class_constructor():
    model = gp.FastGPModeler([1, 2], [3, 4])
    assert model.mu == (-17, -13, 0, 3)


def test_minimize():
    model = gp.FastGPModeler(lightcurve2['time'], lightcurve2['flux'])
    _, _, _, v = model.minimize()
    assert 2.35 < v[4] < 2.37
