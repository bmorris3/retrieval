import sys
sys.path.insert(0, '../')

from retrieval import Planet, get_example_spectrum

import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

np.random.seed(42)

example_spectrum = get_example_spectrum()
wavelength, transit_depth = example_spectrum[:, 0], example_spectrum[:, 1]

planet = Planet(1 * u.M_jup, 1 * u.R_jup, 1e-3 * u.bar, 2.2 * u.u)

on_h_band = np.abs(wavelength - 1.65) < 0.1
off_h_band = np.abs(wavelength - 1.425) < 0.1

depth_on = transit_depth[on_h_band].mean()
depth_off = transit_depth[off_h_band].mean()
depth_difference_observed = (depth_off - depth_on) / depth_off

plt.plot(wavelength, transit_depth)
plt.plot(wavelength[on_h_band], transit_depth[on_h_band])
plt.axhline(depth_on, color='C1', ls='--')
plt.plot(wavelength[off_h_band], transit_depth[off_h_band])
plt.axhline(depth_off, color='C2', ls='--')
plt.xlim([1.25, 1.8])
plt.show()


def distance(theta):
    temperature = theta[0] * u.K
    model = planet.transit_depth(temperature).flux
    depth_difference_simulated = abs((model[off_h_band].mean() -
                                      model[on_h_band].mean()) /
                                     model[off_h_band].mean())
    return abs(depth_difference_simulated - depth_difference_observed)


init_temp = 1500
n_steps = 1500

thresholds = [1e-3, 2e-4, 1e-4]

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