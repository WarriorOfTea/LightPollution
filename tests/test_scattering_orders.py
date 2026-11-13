from dataclasses import replace

import numpy as np
import pytest

from lightpollution.models.scattering_orders import (
    LambertianCap, OrderSettings, boundary_distance, integrate_orders, sample_phase,
)
from lightpollution.physics import EARTH_RADIUS_M, Atmosphere, Observer, SolverSettings, integrate_direction


def scene():
    return Observer(0, 0, 20), LambertianCap(-np.rad2deg(60000/EARTH_RADIUS_M), 0, 40000)


def test_first_boundary_blocks_paths_through_the_earth():
    points = np.array([[EARTH_RADIUS_M+20, 0, 0]]*3)
    directions = np.array([[-1., 0, 0], [1., 0, 0], [0., 1., 0]])
    distance, ground = boundary_distance(points, directions, 80000)
    np.testing.assert_array_equal(ground, [True, False, False])
    assert distance[0] == pytest.approx(20, abs=1e-8)
    assert distance[1] == pytest.approx(79980, abs=1e-8)
    assert np.linalg.norm(points[2]+distance[2]*directions[2]) == pytest.approx(EARTH_RADIUS_M+80000)


@pytest.mark.parametrize('fraction,g,expected_mean,expected_second', [
    (1., .8, 0., .4), (0., .8, .8, (1+2*.8**2)/3),
    (0., -.5, -.5, .5), (0., 0., 0., 1/3), (.4, .8, .48, .4*.4+.6*(1+2*.8**2)/3),
])
def test_phase_samples_independent_analytic_moments(fraction, g, expected_mean, expected_second):
    directions = np.broadcast_to([0., 0., 1.], (100000, 3))
    samples = sample_phase(directions, fraction, g, np.random.default_rng(3119))
    np.testing.assert_allclose(np.linalg.norm(samples, axis=1), 1, atol=2e-15)
    assert np.mean(samples[:, 2]) == pytest.approx(expected_mean, abs=.007)
    assert np.mean(samples[:, 2]**2) == pytest.approx(expected_second, abs=.007)
    assert np.mean(samples[:, 0]) == pytest.approx(0., abs=.007)


def test_surface_quadrature_preserves_radiance_and_total_power():
    _, cap = scene()
    sources = cap.quadrature_sources(8, 16)
    area = 4*np.pi*EARTH_RADIUS_M**2*np.sin(cap.radius_m/(2*EARTH_RADIUS_M))**2
    assert sources.power.sum() == pytest.approx(np.pi*cap.radiance*area, rel=2e-14)
    assert np.all(cap.contains(sources.positions_m))


def test_mc_first_order_matches_independent_deterministic_surface_integral():
    observer, cap = scene()
    mc = integrate_orders(observer, cap, 180., 45., settings=OrderSettings(samples=131072, max_order=1))
    deterministic = integrate_direction(observer, cap.quadrature_sources(16, 48), 180, 45,
        settings=SolverSettings(384, 32))
    error = abs(mc['exact_order_radiance'][0]-deterministic)
    assert error < 4*mc['exact_order_standard_error'][0]
    assert sum(mc['path_outcomes'].values()) == 131072


def test_identical_random_prefixes_for_paired_orders_and_replay():
    observer, cap = scene()
    settings = OrderSettings(samples=32768, max_order=3, batch_size=4096)
    three = integrate_orders(observer, cap, 180, 45, settings=settings)
    two = integrate_orders(observer, cap, 180, 45, settings=replace(settings, max_order=2))
    assert three['exact_order_radiance'][:2] == two['exact_order_radiance']
    assert three['exact_order_standard_error'][:2] == two['exact_order_standard_error']
    assert three['luminous_paths_by_order'][:2] == two['luminous_paths_by_order']
    assert np.all(np.diff(three['cumulative_radiance']) >= 0)
    covariance = np.asarray(three['mean_covariance'])
    assert np.linalg.eigvalsh(covariance).min() >= -1e-20
    assert three['cumulative_standard_error'][-1]**2 == pytest.approx(np.sum(covariance))
    double_cap = replace(cap, radiance=2)
    double = integrate_orders(observer, double_cap, 180, 45, settings=settings)
    np.testing.assert_array_equal(double['exact_order_radiance'], 2*np.array(three['exact_order_radiance']))
    assert sum(three['path_outcomes'].values()) == settings.samples


@pytest.mark.parametrize('aod,ssa', [(0., .9), (.2, 0.)])
def test_no_spurious_radiance_in_vacuum_or_pure_absorber(aod, ssa):
    observer, cap = scene()
    atmosphere = Atmosphere(rayleigh_multiplier=0, aerosol_optical_depth=aod,
        aerosol_single_scattering_albedo=ssa)
    result = integrate_orders(observer, cap, 180, 10, atmosphere, OrderSettings(samples=256))
    assert result['exact_order_radiance'] == [0., 0., 0.]
    assert sum(result['path_outcomes'].values()) == 256


@pytest.mark.parametrize('kwargs', [{'samples': 1}, {'max_order': 0}, {'seed': -1}, {'samples': True}, {'max_order': 33}])
def test_invalid_settings_fail(kwargs):
    with pytest.raises(ValueError):
        OrderSettings(**kwargs)


def test_null_collision_limit_must_fail_instead_of_biasing_estimate():
    observer, cap = scene()
    with pytest.raises(RuntimeError, match='Null-collision limit'):
        integrate_orders(observer, cap, 180, 45, settings=OrderSettings(samples=2048, max_null_events=1))


def test_observer_outside_medium_fails():
    observer, cap = scene()
    with pytest.raises(ValueError, match='inside'):
        integrate_orders(observer, cap, 180, 45, Atmosphere(top_m=10))
