import matplotlib.pyplot as plt

__all__ = ['Spectrum']


class Spectrum(object):
    """
    A simple spectrum-like object.

    Consists of a wavelength axis plus a "flux" axis, where "flux" is a
    stand-in for anything that's measured as a function of wavelength, including
    flux, transit depth, etc.
    """
    def __init__(self, wavelength, flux):
        self.wavelength = wavelength
        self.flux = flux

    def plot(self, ax=None, **kwargs):
        """
        Plot the wavelength axis against the "flux" axis.
        """
        if ax is None:
            ax = plt.gca()

        ax.plot(self.wavelength, self.flux, **kwargs)

        return ax