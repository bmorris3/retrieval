import astropy.units as u

from ..core import Planet


def test_radius():
    temperature = 1500 * u.K
    planet = Planet(1 * u.M_jup, 1 * u.R_jup, 1e-3 * u.bar, 2.2 * u.u)
    sp = planet.transit_depth(temperature)

    assert abs(sp.flux.mean() - 0.01075) < 1e-5
