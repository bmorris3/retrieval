import numpy as np
import astropy.units as u
from astropy.constants import G, k_B

from .opacity import water_opacity
from .spectrum import Spectrum


__all__ = ['Planet']

gamma = 0.57721


class Planet(object):
    """
    Properties of an exoplanet.
    """
    def __init__(self, mass, radius, pressure, mu):
        """
        Parameters
        ----------
        mass : `~astropy.unit.Quantity`
            Mass of the transiting planet
        radius : `~astropy.unit.Quantity`
            Radius of the transiting planet
        pressure : `~astropy.unit.Quantity`
            Pressure level probed in transmission
        mu : `~astropy.unit.Quantity`
            Mean molecular weight of the atmosphere
        """
        self.mass = mass
        self.radius = radius
        self.pressure = pressure
        self.mu = mu

    def transit_depth(self, temperature, rstar=1 * u.R_sun):
        """
        Compute the transit depth with wavelength at ``temperature``.

        Parameters
        ----------
        temperature : `~astropy.units.Quantity`
            Temperature of the atmosphere observed in transmission
        rstar : `~astropy.units.Quantity`
            Radius of the star, used to compute the ratio of the planet-to-star
            radius. Default is one solar radius.

        Returns
        -------
        sp : `~retrieval.Spectrum`
            Transit depth spectrum
        """
        wavenumber, kappa = water_opacity(temperature)

        g = G * self.mass / self.radius**2
        P0 = self.pressure

        scale_height = k_B * temperature / self.mu / g
        tau = P0 * kappa / g * np.sqrt(2 * np.pi * self.radius / scale_height)
        r = self.radius + scale_height * (gamma + np.log(tau))

        depth = (r / rstar).decompose() ** 2
        wavelength = wavenumber.to(u.um, u.spectral())

        return Spectrum(wavelength, depth)
