import os

import numpy as np
import astropy.units as u
from scipy.interpolate import RegularGridInterpolator

__all__ = ['water_opacity']

water_opacity_path = os.path.join(os.path.dirname(__file__), 'data',
                                  'water_opacity_grid.npy')

wavenumber = np.arange(2000, 13000, 5) / u.cm


class Interpolator(object):
    """
    Lazy loader for the water opacity grid interpolator
    """
    def __init__(self):
        self._interpolator = None

    @property
    def interp(self):
        # Lazy load the interpolator
        if self._interpolator is None:
            opacity_grid = np.load(water_opacity_path)
            temperature_grid = np.concatenate([np.arange(50, 700, 50),
                                               np.arange(700, 1500, 100),
                                               np.arange(1500, 3100,
                                                         200)]) * u.K
            interp = RegularGridInterpolator((wavenumber.value,
                                              temperature_grid.value),
                                             opacity_grid)
            self._interpolator = lambda temp: (interp((wavenumber.value,
                                                       temp.value))
                                               * u.cm ** 2 / u.g)
        return self._interpolator


interpolator = Interpolator()


def water_opacity(temperature):
    """

    Parameters
    ----------
    temperature : `~astropy.units.Quantity`
        Temperature of the exoplanet atmosphere

    Returns
    -------
    wavenumber : `~astropy.units.Quantity`

    opacity : `~astropy.units.Quantity`
    """
    return wavenumber, interpolator.interp(temperature)
