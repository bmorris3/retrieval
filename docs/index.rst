retrieval
=========

This is the documentation for ``retrieval``.

Here's a simple example:

.. plot::

    from retrieval import transit_depth
    import astropy.units as u
    import matplotlib.pyplot as plt
    import numpy as np

    temperatures = np.arange(1000, 3000, 500) * u.K

    for temperature in temperatures:
        sp = transit_depth(temperature)

        ax = sp.plot(label=temperature)

    ax.set_xlabel('Wavelength [$\mu$m]')
    ax.set_ylabel('Transit depth')
    ax.legend()
    plt.show()


.. toctree::
  :maxdepth: 2

  retrieval/index.rst
