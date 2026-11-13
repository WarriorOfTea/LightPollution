from dataclasses import replace

import numpy as np
import pytest

from lightpollution.data import Bounds
from lightpollution.models.all_orders import AllOrderSettings, integrate_all_orders
from lightpollution.models.angular_sources import (AngularGridBoundary, bin_horizontal_factor,
    empirical_radiance, eta_from_ratio, horizontal_radiance_factor)
from lightpollution.models.boundary_orders import integrate_boundary_variants, paired_ratio
from lightpollution.models.scattering_orders import LambertianCap
from lightpollution.physics import Atmosphere, EARTH_RADIUS_M, Observer, unit_position


class Caps:
    labels = ('cap', 'twice_cap')
    upper_bounds = np.array([1., 2.])

    def __init__(self, cap):
        self.cap = cap

    def sample(self, points, outgoing):
        mask = self.cap.contains(points)
        return mask[:, None]*np.array([1., 2.]), np.where(mask, 2, 0)


@pytest.mark.parametrize('absorption', ['implicit', 'analog'])
def test_full_boundary_transport_replays_reference_paths_and_joint_statistics(absorption):
    observer = Observer(0, 0, 20)
    cap = LambertianCap(-np.rad2deg(60000/EARTH_RADIUS_M), 0, 40000)
    settings = AllOrderSettings(samples=8192, absorption=absorption)
    old = integrate_all_orders(observer, cap, 180, 10, settings=settings)
    new = integrate_boundary_variants(observer, Caps(cap), 180, 10, settings=settings)
    np.testing.assert_allclose(np.array(new['category_mean_radiance'])[:, 0], old['category_mean_radiance'], rtol=1e-14)
    np.testing.assert_allclose(new['category_mean_covariance_per_variant'][0], old['mean_covariance'], rtol=1e-14)
    assert new['all_order_radiance'][0] == pytest.approx(old['all_order_radiance'], rel=1e-14)
    assert new['all_order_standard_error'][0] == pytest.approx(old['all_order_standard_error'], rel=1e-14)
    assert new['max_scattering_order_seen'] == old['max_scattering_order_seen']
    assert sum(new['path_outcomes'].values()) == settings.samples
    assert sum(new['ground_support_counts'].values()) == new['path_outcomes']['ground']
    np.testing.assert_allclose(np.sum(new['emission_bin_radiance'], axis=0), new['all_order_radiance'], rtol=1e-14)
    r = paired_ratio(new, 'twice_cap', 'cap')
    assert r['ratio'] == 2
    assert r['ratio_standard_error'] < 1e-10
    assert new['all_order_standard_error'][1] == 2*new['all_order_standard_error'][0]
    fewer = integrate_boundary_variants(observer, Caps(cap), 180, 10, settings=replace(settings, recorded_orders=1))
    np.testing.assert_allclose(fewer['all_order_radiance'], new['all_order_radiance'], rtol=1e-14)


def test_view_bin_constraints_hold_for_every_angle_inside_the_measured_intervals():
    near, off = 10., 8.
    for tail in (0., 1., 2.):
        assert empirical_radiance(near, off, np.cos(np.deg2rad(10)), tail) == near
        np.testing.assert_allclose(empirical_radiance(near, off, np.cos(np.deg2rad([40, 50, 60])), tail), off)
        assert empirical_radiance(near, off, 0., tail) == tail*off
        value = empirical_radiance(near, off, np.linspace(0, 1, 1000), tail)
        assert np.all((value >= 0) & (value <= max(near, tail*off, off)))


def test_mixture_fit_uses_radiance_and_retains_incompatible_eta():
    gn, go = bin_horizontal_factor(0, 20), bin_horizontal_factor(40, 60)
    for eta in (0., .1, .6):
        ratio = (1-eta+eta*go)/(1-eta+eta*gn)
        assert float(eta_from_ratio(ratio)) == pytest.approx(eta)
    assert not 0 <= eta_from_ratio(.8) <= 1
    assert horizontal_radiance_factor(.5) == .3125
    for measure in ('solid_angle', 'zenith_angle'):
        assert bin_horizontal_factor(40, 60, measure) > bin_horizontal_factor(0, 20, measure)


def grid():
    bounds = Bounds(0, 0, 2/240, 2/240)
    near, off = np.array([[10., 20.], [30., np.nan]]), np.array([[20., 10.], [30., np.nan]])
    return AngularGridBoundary(bounds, near, off, near, np.array([[True, True], [True, False]]), .2)


def test_cell_lookup_sign_area_support_bounds_and_low_angle_caps():
    b = grid()
    lat = np.array([1.5, 1.5, .5, .5, 1.5])/240
    lon = np.array([.5, 1.5, .5, 1.5, 2.5])/240
    normal = unit_position(lat, lon)
    p = EARTH_RADIUS_M*normal
    value, code = b.sample(p, normal)
    np.testing.assert_allclose(value[:, 0], [10, 20, 30, 0, 0])
    np.testing.assert_array_equal(code, [2, 2, 2, 1, 0])
    # Tangent directions probe the declared radiance bounds at the horizon.
    tangent = np.cross(normal, [0, 0, 1.])
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    value, _ = b.sample(p, tangent)
    assert np.all(np.isfinite(value)) and np.all(value <= b.upper_bounds[None, :]*(1+1e-12))
    assert value[0, 3] == pytest.approx(0., abs=1e-13)
    assert value[0, 4] == 20
    assert value[0, 5] == 40


def test_no_scattering_and_safety_failure_cannot_publish_truncated_light():
    observer = Observer(0, 0, 20)
    cap = LambertianCap(0, 0, 40000)
    r = integrate_boundary_variants(observer, Caps(cap), 180, 45,
        Atmosphere(rayleigh_multiplier=0, aerosol_optical_depth=0), AllOrderSettings(samples=32))
    assert r['all_order_radiance'] == [0., 0.]
    assert r['normal_95_intervals'] == [None, None]
    assert paired_ratio(r, 'cap', 'twice_cap')['ratio'] is None
    with pytest.raises(RuntimeError, match='NOT returned or truncated'):
        integrate_boundary_variants(observer, Caps(cap), 180, 10,
            settings=AllOrderSettings(samples=8192, safety_scattering_events=1))


def test_false_boundary_bound_and_invalid_maps_fail_instead_of_biased_clipping():
    b = grid()
    with pytest.raises(ValueError):
        AngularGridBoundary(b.bounds, b.near, b.off, b.all_angle, b.support, 1.)
    boundary = Caps(LambertianCap(0, 0, np.pi*EARTH_RADIUS_M))
    boundary.upper_bounds = np.array([.1, .1])
    with pytest.raises(ValueError, match='exceeds its declared bound'):
        integrate_boundary_variants(Observer(0, 0, 20), boundary, 180, 10,
            settings=AllOrderSettings(samples=8192))
