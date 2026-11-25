"""Invariants for regional and spatial AOD in the photometer pilots."""

from pathlib import Path

import numpy as np

from lightpollution.models.scattering_orders import _collision, boundary_distance
from lightpollution.physics import EARTH_RADIUS_M, Observer, unit_position


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


def test_photometer_boundary_pairs_distances_on_the_selected_raster(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    from lightpollution.data import Bounds
    from run_photometer_all_orders import NearBoundary

    bounds = Bounds(11.5, 45.25, 11.75, 45.5)
    key = 'NearNadir_Composite_Snow_Free'
    raster = np.ones(bounds.shape)
    observer = Observer(bounds.latitude[29], bounds.longitude[29], 337)
    boundary = NearBoundary({key: 2 * raster, key + '_Quality': 0 * raster,
                             key + '_Num': 4 * raster, 'Land_Water_Mask': raster},
                            observer, bounds, (0, 3, 10))
    points = EARTH_RADIUS_M * unit_position(bounds.latitude[[29, 40, 58, 58]],
                                            [bounds.longitude[29]] * 3 + [11.49])
    scores, quality = boundary.sample(points, points / EARTH_RADIUS_M)
    np.testing.assert_array_equal(scores, [[2, 0, 0], [2, 2, 0], [2, 2, 2], [0, 0, 0]])
    np.testing.assert_array_equal(quality, [2, 2, 2, 0])


def test_photometer_percent_error_uses_total_flux_and_one_shared_fit(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    from summarize_photometer_italy import signed_errors

    runs = {'base': 'reference', 'spatial': 'spatial', 'regional': 'regional'}
    rows = [{'observed_relative': 3, 'runs': runs}, {'observed_relative': 6, 'runs': runs}]
    values = {'reference': np.array([[2.], [4.]]),
              'spatial': np.array([[4.], [4.]]), 'regional': np.array([[1.], [1.]])}
    errors, scale = signed_errors(rows, [0], values, 0, 1)
    np.testing.assert_allclose(scale, [1, .5])
    np.testing.assert_allclose(errors[1], 100 * np.array([[.5, 5/6, 2/6], [.5, .5, 1.5/6]]) - 100)
