# Periodicity

Useful tools for analysis of periodicities in time series data.

## Quick start
`pip install periodicity`

## Example using GP with astronomical data
```python
from periodicity.gp import *
from lightkurve import KeplerLightCurveFile

lc = KeplerLightCurveFile.from_archive(target=9895037, quarter=4).PDCSAP_FLUX.normalize()
lc = lc.append(KeplerLightCurveFile.from_archive(target=9895037, quarter=5).PDCSAP_FLUX.normalize())
lc = lc.remove_nans().remove_outliers().bin(binsize=4)

t, x = lc.time, lc.flux
x = x - x.mean()

model = FastGPModeler(t, x)
model.prior = make_gaussian_prior(t, x, pmin=2)
model.minimize()
samples = model.mcmc(nwalkers=32, nsteps=5000, burn=500)

print('Median period: {:.2f}'.format(np.exp(np.median(samples[:, 4]))))
```
