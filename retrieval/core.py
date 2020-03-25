import numpy as np
import astropy.units as u
from astropy.constants import G, k_B, R_jup, M_jup, R_sun

from .opacity import water_opacity
from .spectrum import Spectrum


__all__ = ['transit_depth']

gamma = 0.57721


def transit_depth(temperature):
    """
    Compute the transit depth with wavelength at ``temperature``.

    Parameters
    ----------
    temperature : `~astropy.units.Quantity`

    Returns
    -------
    sp : `~retrieval.Spectrum`
        Transit depth spectrum
    """
    wavenumber, kappa = water_opacity(temperature)

    g = G * M_jup / R_jup**2
    rstar = 1 * R_sun

    R0 = R_jup
    P0 = 1e-3 * u.bar

    mu = 2 * u.u

    scale_height = k_B * temperature / mu / g
    tau = P0 * kappa / g * np.sqrt(2.0 * np.pi * R0 / scale_height)
    r = R0 + scale_height * (gamma + np.log(tau))

    depth = (r / rstar) ** 2
    wavelength = wavenumber.to(u.um, u.spectral())

    return Spectrum(wavelength, depth)
