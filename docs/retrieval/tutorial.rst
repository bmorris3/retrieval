Tutorial
--------

:math:`\chi^2` minimization
+++++++++++++++++++++++++++

.. code-block:: python

    from retrieval import Planet, get_example_spectrum

    import numpy as np
    from scipy.optimize import fmin_l_bfgs_b
    import astropy.units as u

    example_spectrum = get_example_spectrum()
    wavelength, transit_depth = example_spectrum[:, 0], example_spectrum[:, 1]

    plt.plot(wavelength, transit_depth)
    plt.xlabel('Wavelength [$\mu$m]')
    plt.ylabel('Transit Depth')
    plt.show()

.. plot::

    from retrieval import Planet, get_example_spectrum

    import numpy as np
    from scipy.optimize import fmin_l_bfgs_b
    import astropy.units as u

    example_spectrum = get_example_spectrum()

    plt.plot(example_spectrum[:, 0], example_spectrum[:, 1])
    plt.xlabel('Wavelength')
    plt.ylabel('Transit Depth')
    plt.show()

.. code-block:: python

    planet = Planet(1 * u.M_jup, 1 * u.R_jup, 1e-3 * u.bar, 2.2 * u.u)


.. code-block:: python

    def chi2(p):
        """
        Compute the chi^2 for the model with parameters `p`
        """
        temperature = p[0] * u.K
        return np.sum((example_spectrum[:, 1] -
                       planet.transit_depth(temperature).flux)**2 /
                      example_spectrum[:, 2]**2)

    initp = [1700]  # K

    bestp = fmin_l_bfgs_b(chi2, initp, approx_grad=True,
                          bounds=[[500, 5000]])[0][0] * u.K

The resulting best-fit temperature is::

    >>> print(bestp)  # doctest: +SKIP
    1509.4660124465638

MCMC with a likelihood
++++++++++++++++++++++

.. code-block:: python

    from emcee import EnsembleSampler


    def lnprior(theta):
        """
        Log-prior
        """
        temperature = theta[0]

        if 500 < temperature < 5000:
            return 0
        return -np.inf


    def lnlikelihood(theta):
        """
        Log-likelihood
        """
        temperature = theta[0] * u.K
        model = planet.transit_depth(temperature).flux
        lp = lnprior(theta)
        return lp + -0.5 * np.sum((example_spectrum[:, 1] - model)**2 /
                                   example_spectrum[:, 2]**2)



.. code-block:: python

    nwalkers = 4
    ndim = 1

    p0 = [[1500 + 10 * np.random.randn()]
          for i in range(nwalkers)]

    with Pool() as pool:
        sampler = EnsembleSampler(nwalkers, ndim, lnlikelihood, pool=pool)
        sampler.run_mcmc(p0, 1000)

    plt.hist(sampler.flatchain)
    plt.xlabel('Temperature [K]')
    plt.show()

.. plot::

    from retrieval import Planet, get_example_spectrum

    import numpy as np
    import astropy.units as u
    import matplotlib.pyplot as plt

    from emcee import EnsembleSampler

    example_spectrum = get_example_spectrum()

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

    sampler = EnsembleSampler(nwalkers, ndim, lnlikelihood)
    sampler.run_mcmc(p0, 1000)

    plt.hist(sampler.flatchain)
    plt.xlabel('Temperature [K]')
    plt.show()


ABC without a likelihood
++++++++++++++++++++++++

.. code-block:: python

    def distance(theta):
        """
        In this example, the distance is the reduced chi^2
        """
        temperature = theta[0] * u.K
        model = planet.transit_depth(temperature).flux
        return np.sum((example_spectrum[:, 1] - model)**2 /
                      example_spectrum[:, 2]**2) / example_spectrum.shape[0]


.. code-block:: python

    init_temp = 1500
    n_steps = 100

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


.. plot::

    from retrieval import Planet, get_example_spectrum

    import numpy as np
    import astropy.units as u
    import matplotlib.pyplot as plt

    example_spectrum = get_example_spectrum()

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

        plt.hist(temperature_chain, histtype='step', lw=2,
                 label=f'h = {threshold}')
    plt.legend()
    plt.xlabel('Temperature [K]')
    plt.show()


