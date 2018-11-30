# Periodicity

Useful tools for analysis of periodicities in time series data.

Includes:
* Auto-Correlation Function
* Fourier methods:
    * Lomb-Scargle periodogram
    * Wavelet Transform (in progress)
* Phase-folding methods:
    * String Length
    * Analysis of Variance (in progress)
* Gaussian Processes:
    * `george` implementation
    * `celerite` implementation
    * `pymc3` implementation (in progress)

## Quick start
    $ pip install periodicity

## Example using GP with astronomical data
```python
from periodicity.gp import *
from lightkurve import search_lightcurvefile

lcs = search_lightcurvefile(target=9895037, quarter=[4,5]).download_all()
lc = lcs[0].PDCSAP_FLUX.normalize().append(lcs[1].PDCSAP_FLUX.normalize())
lc = lc.remove_nans().remove_outliers().bin(binsize=4)

t, x = lc.time, lc.flux
x = x - x.mean()

model = FastGPModeler(t, x)
model.prior = make_gaussian_prior(t, x, pmin=2)
model.minimize()
samples = model.mcmc(nwalkers=32, nsteps=5000, burn=500)

print('Median period: {:.2f}'.format(np.exp(np.median(samples[:, 4]))))
```

### Visualization of this example:

![gp_example](https://github.com/dioph/periodicity/blob/master/figures/example2.png?raw=True)

![gp_example](https://github.com/dioph/periodicity/blob/master/figures/example1.png?raw=True)
