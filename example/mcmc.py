import sys
sys.path.insert(0, '../')

from retrieval import Planet

import numpy as np
import astropy.units as u
from emcee import EnsembleSampler
from multiprocessing import Pool
import matplotlib.pyplot as plt

example_spectrum = np.load('../retrieval/data/example_spectrum.npy')

planet = Planet(1 * u.M_jup, 1 * u.R_jup, 1e-3 * u.bar, 2.2 * u.u)


def lnprior(theta):
    temperature = theta[0]

    if 500 < temperature < 5000:
        return 0
    return -np.inf


def lnlikelihood(theta):
    temperature = theta[0] * u.K
    model = planet.transit_depth(temperature).flux
    lp = lnprior(theta)
    return lp + -0.5 * np.sum((example_spectrum[:, 1] - model)**2 /
                               example_spectrum[:, 2]**2)

nwalkers = 10
ndim = 1

p0 = [[1500 + 10 * np.random.randn()]
      for i in range(nwalkers)]

with Pool() as pool:
    sampler = EnsembleSampler(nwalkers, ndim, lnlikelihood, pool=pool)
    sampler.run_mcmc(p0, 1000)

plt.hist(sampler.flatchain)
plt.xlabel('Temperature [K]')
plt.show()