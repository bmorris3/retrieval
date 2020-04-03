import sys
sys.path.insert(0, '../')

from retrieval import Planet

import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

example_spectrum = np.load('../retrieval/data/example_spectrum.npy')

planet = Planet(1 * u.M_jup, 1 * u.R_jup, 1e-3 * u.bar, 2.2 * u.u)


def distance(theta):
    temperature = theta[0] * u.K
    model = planet.transit_depth(temperature).flux
    return np.sum((example_spectrum[:, 1] - model)**2 /
                  example_spectrum[:, 2]**2) / example_spectrum.shape[0]

init_temp = 1500

distance_chain = [distance([init_temp])]
temperature_chain = [init_temp]

n_steps = 5000

threshold = 1
i = 0
total_steps = 1

while len(temperature_chain) < n_steps:
    total_steps += 1
    trial_temp = temperature_chain[i] + 10 * np.random.randn()
    trial_dist = distance([trial_temp])

    if trial_dist < threshold:
        i += 1
        temperature_chain.append(trial_temp)
        distance_chain.append(trial_dist)

acceptance_rate = i / total_steps
print("acceptance rate = ", acceptance_rate)

plt.hist(temperature_chain)
plt.xlabel('Temperature [K]')
plt.show()