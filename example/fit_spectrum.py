import sys
sys.path.insert(0, '../')

from retrieval import Planet

import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import astropy.units as u

example_spectrum = np.load('../retrieval/data/example_spectrum.npy')

planet = Planet(1 * u.M_jup, 1 * u.R_jup, 1e-3 * u.bar, 2.2 * u.u)


def minimize(p):
    temperature = p[0] * u.K
    return np.sum((example_spectrum[:, 1] -
                   planet.transit_depth(temperature).flux)**2 /
                  example_spectrum[:, 2]**2)

initp = [1700]  # K

bestp = fmin_l_bfgs_b(minimize, initp, approx_grad=True,
                      bounds=[[500, 5000]])[0][0] * u.K

print(bestp)