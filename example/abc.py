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
n_steps = 1000

thresholds = [1.5, 1.1, 1.0]

for threshold in thresholds:
    # Create chains for the distance and temperature
    distance_chain = [distance([init_temp])]
    temperature_chain = [init_temp]

    # Set some indices
    i = 0
    total_steps = 1

    # Until the chain is the correct number of steps...
    while len(temperature_chain) < n_steps:
        # Generate a trial temperature
        total_steps += 1
        trial_temp = temperature_chain[i] + 10 * np.random.randn()

        # Measure the distance between the trial step and observations
        trial_dist = distance([trial_temp])

        # If trial step has distance less than some threshold...
        if trial_dist < threshold:
            # Accept the step, add values to the chain
            i += 1
            temperature_chain.append(trial_temp)
            distance_chain.append(trial_dist)

    # Compute the acceptance rate:
    acceptance_rate = i / total_steps
    print(f"h = {threshold}, acceptance rate = {acceptance_rate}")

    plt.hist(temperature_chain, histtype='step', lw=2,
             label=f'h = {threshold}')
plt.legend()
plt.xlabel('Temperature [K]')
plt.show()