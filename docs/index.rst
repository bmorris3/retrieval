retrieval
=========

This is the documentation for ``retrieval``.

Here's a simple example:

.. plot::

    import astropy.units as u
    import matplotlib.pyplot as plt
    import numpy as np

    from retrieval import Planet

    temperatures = np.arange(1000, 3000, 500) * u.K

    planet = Planet(1 * u.M_jup, 1 * u.R_jup, 1e-3 * u.bar, 2.2 * u.u)

    for temperature in temperatures:
        sp = planet.transit_depth(temperature)

        ax = sp.plot(label=temperature)

    ax.set_xlabel('Wavelength [$\mu$m]')
    ax.set_ylabel('Transit depth')
    ax.legend()
    plt.show()


.. toctree::
  :maxdepth: 2

  retrieval/index.rst
