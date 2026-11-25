"""Invariants for regional and spatial AOD in the photometer pilots."""

from pathlib import Path

import numpy as np

from lightpollution.models.scattering_orders import _collision, boundary_distance
from lightpollution.physics import EARTH_RADIUS_M, Observer


SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'


def test_regional_aod_integrates_the_bilinear_field(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    from run_cams_dutch_pilot import area_mean_aod

    latitude = np.deg2rad([50., 50.3, 54.])
    longitude = np.deg2rad([4., 6., 10.])
    points = .02 + .04 * latitude[:, None] + .3 * longitude
    expected_latitude = np.diff(latitude * np.sin(latitude) + np.cos(latitude))[0:2].sum()
    expected_latitude /= np.sin(latitude[-1]) - np.sin(latitude[0])
    expected = .02 + .04 * expected_latitude + .3 * (longitude[0] + longitude[-1]) / 2
    actual = area_mean_aod(np.stack((points, np.full_like(points, .1))),
                           np.rad2deg(latitude), np.rad2deg(longitude))
    np.testing.assert_allclose(actual, [expected, .1], rtol=1e-12)


def test_spatial_collision_matches_transmission_along_known_aod_gradient(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    from run_photometer_all_orders import GridAtmosphere

    observer = Observer(52., 4.5, 250.)
    direction = observer.direction(90., 5.)
    atmosphere = GridAtmosphere(latitudes=(50., 55.), longitudes=(4., 5., 6.),
                                grid=((.005, .015, .009), (.005, .015, .009)),
                                rayleigh_multiplier=0.)
    distance, _ = boundary_distance(observer.position[None, :], direction[None, :], atmosphere.top_m)
    path = np.linspace(0, distance.item(), 20001)
    x, y, z = (observer.position + path[:, None] * direction).T
    longitude = np.rad2deg(np.arctan2(y, x))
    height = np.sqrt(x*x+y*y+z*z) - EARTH_RADIUS_M
    aod = np.interp(longitude, [4., 5., 6.], [.005, .015, .009])
    extinction = aod / 1200 * np.exp(-height / 1200)
    expected = np.exp(-np.trapezoid(extinction, x=path))
    count = 150_000
    _, collision, _ = _collision(np.broadcast_to(observer.position, (count, 3)),
                                 np.broadcast_to(direction, (count, 3)),
                                 atmosphere, np.random.default_rng(2024), 1000)
    measured = 1 - collision.mean()
    standard_error = np.sqrt(expected * (1-expected) / count)
    assert abs(measured-expected) < 4 * standard_error
