import sys
sys.path.insert(0, '../')

from retrieval import Planet

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

temperature = 1500 * u.K

planet = Planet(1 * u.M_jup, 1 * u.R_jup, 1e-3 * u.bar, 2.2 * u.u)
sp = planet.transit_depth(temperature)

output_path = '../retrieval/data/example_spectrum.npy'

np.save(output_path, np.vstack([sp.wavelength.value, sp.flux.value,
                                sp.flux.mean().value / 100 *
                                np.ones(len(sp.flux))]).T)

ax = sp.plot(label=temperature)

ax.set_xlabel('Wavelength [$\mu$m]')
ax.set_ylabel('Transit depth')
ax.legend()
plt.show()