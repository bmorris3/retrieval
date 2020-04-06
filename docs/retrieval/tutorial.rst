Tutorial
========

Background
++++++++++

In this tutorial, we seek to measure the temperature of an exoplanet atmosphere
via transmission spectroscopy. As a transiting planet passes in front of its
host star, the planet will appear largest at wavelengths where the planet's
atmosphere is opaque. For this toy model, we will include water opacity as the
only absorbing species in the atmosphere.

As in `Fisher & Heng (2018)
<https://ui.adsabs.harvard.edu/abs/2018MNRAS.481.4698F/abstract>`_, we compute
the apparent radius of the exoplanet as a function of wavelength given by

.. math::

    R = R_0 + H (\gamma + \ln \tau)

where :math:`R_0` is the fiducial radius of the exoplanet, :math:`H` is the
atmospheric scale height, :math:`\gamma` is the Euler-Mascheroni constant, and
:math:`\tau` is

.. math::

    \tau = \frac{\kappa_\lambda P_0}{g} \sqrt{\frac{2\pi R_0}{H}}

where :math:`\kappa_\lambda` is the wavelength-dependent opacity, :math:`P_0`
is the reference pressure probed by transmission spectroscopy, and :math:`g` is
the surface gravity. The pressure scale height :math:`H` is

.. math::

    H = \frac{k_B T}{\mu g}

where :math:`k_B` is Boltzmann's constant, :math:`T` is the temperature, and
:math:`\mu` is the mean molecular weight.

We have generated an example transmission spectrum with random normal noise
with :math:`T = 1500` K, and in the following examples we will attempt to
"retrieve" that temperature directly from the spectrum. First we'll solve for
the approximate temperature (without uncertainties) using :math:`\chi^2`
minimization. Then we'll solve for the temperature and its uncertainty assuming
Gaussian uncertainties using a likelihood function and Markov Chain Monte Carlo
(MCMC). Finally, we'll use approximate Bayesian computation (ABC) without
defining a proper likelihood.



:math:`\chi^2` minimization
+++++++++++++++++++++++++++

First we will access the pre-generated example spectrum contained within the
``retrieval`` module, like so:

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
    plt.xlabel('Wavelength [$\mu$m]')
    plt.ylabel('Transit Depth')
    plt.show()

The transit depth spectrum spans the near-infrared, where water opacity is
significant, giving rise to the J, H and K bands visible in this transmission
spectrum.

Now we create an instance of the `~retrieval.Planet` object, which requires us
to specify the planet's mass, radius, atmospheric pressure and mean molecular
weight:

.. code-block:: python

    planet = Planet(1 * u.M_jup, 1 * u.R_jup, 1e-3 * u.bar, 2.2 * u.u)


With the exoplanet's properties defined, we can now solve for the temperature
in the atmosphere, using the forward model described in the equations above.
We'll define the :math:`\chi^2` by comparing the example transmission spectrum
with instances of the forward model at different temperatures, varying the
temperature until we find good agreement between the model and observations:

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
                          bounds=[[500, 5000]])[0][0]

The resulting best-fit temperature is::

    >>> print(bestp)  # doctest: +SKIP
    1509.4660124465638

which is close to the temperature used to generate the example spectrum, so we
have demonstrated that the forward model is producing a sufficient approximation
to the observed spectrum.

MCMC with a likelihood
++++++++++++++++++++++

It was straightforward in the example above to fit for the temperature, but it
may take a bit more effort to find the uncertainty on the temperature. One
computationally expensive but easy-to-implement technique for measuring the
uncertainty on the fitting parameter (temperature) is with `Markov Chain Monte
Carlo <https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo>`_. MCMC is a
Bayesian technique, and uses some of the concepts straight out of `Bayes'
theorem <https://en.wikipedia.org/wiki/Bayes%27_theorem>`_,

.. math::

    P\left( \theta \vert D \right) = \frac{ \pi\left( \theta \right) ~{\cal L}\left( D \vert \theta \right) }{ {\cal Z} \left( D \right)}.

The prior distribution, denoted by :math:`\pi(\theta)`, represents your prior
beliefs about the fitting parameters :math:`\theta`.

The likelihood function, denoted by :math:`{\cal L}( D \vert \theta)`, is the
mathematical relationship between the data (:math:`D`), model and
measurement noise. The goal of MCMC is to numerically evaluate the right hand
side of the equation to solve for the posterior distribution
:math:`P\left( \theta \vert D \right)`.

To do so, we must first describe the prior and likelihood, respectively:

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


We've chosen a flat prior that expects the temperature to sit between 500 and
5000 K, which might represent our sensible rough estimates for the minimum and
maximum temperature a planet might have given its orbital distance and host
star's spectral type. The log-likelihood we have chosen for this example is
the sum of the log-prior and :math:`-0.5 \chi^2`. This is a natural choice for
the likelihood given Gaussian, uncorrelated uncertainties for the transit depth
measurements.

We can now sample the posterior distribution with MCMC using
`emcee <https://emcee.readthedocs.io/en/stable/>`_ like so:

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

The algorithm produces a "chain" of posterior samples for the temperature of the
atmosphere, which we see is roughly Gaussian in shape and centered near 1510 K
with an uncertainty of roughly 10 K.

ABC without a likelihood
++++++++++++++++++++++++

There can be situations where the likelihood is expensive or difficult to
compute. In these situations, it can be useful to use `approximate Bayesian
computation <https://en.wikipedia.org/wiki/Approximate_Bayesian_computation>`_,
a technique for estimating posterior distributions without computing a
likelihood.

Central to the idea of

.. code-block:: python

    from sklearn.metrics import r2_score

    def distance(theta):
        temperature = theta[0] * u.K
        model = planet.transit_depth(temperature).flux
        return abs(r2_score(example_spectrum[:, 1], model) - 1)

.. code-block:: python

    thresholds = [0.8, 0.4, 0.25]

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
            trial_temp = temperature_chain[i] + 100 * np.random.randn()

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
    from sklearn.metrics import r2_score

    example_spectrum = get_example_spectrum()

    planet = Planet(1 * u.M_jup, 1 * u.R_jup, 1e-3 * u.bar, 2.2 * u.u)


    def distance(theta):
        temperature = theta[0] * u.K
        model = planet.transit_depth(temperature).flux
        return abs(r2_score(example_spectrum[:, 1], model) - 1)


    init_temp = 1500
    n_steps = 1500

    thresholds = [0.8, 0.4, 0.25]

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
            trial_temp = temperature_chain[i] + 100 * np.random.randn()

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


