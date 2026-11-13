from dataclasses import replace

import numpy as np
import pytest

from lightpollution.models.all_orders import AllOrderSettings, integrate_all_orders, relative_tail
from lightpollution.models.scattering_orders import LambertianCap, OrderSettings, integrate_orders
from lightpollution.physics import EARTH_RADIUS_M, Atmosphere, Observer


def scene():
    return Observer(0, 0, 20), LambertianCap(-np.rad2deg(60000/EARTH_RADIUS_M), 0, 40000)


def test_old_five_orders_replay_but_higher_paths_are_not_discarded():
    observer, cap = scene()
    atm = replace(Atmosphere(), aerosol_optical_depth=.3)
    settings = AllOrderSettings(samples=32768)
    full = integrate_all_orders(observer, cap, 180, 10, atm, settings)
    old = integrate_orders(observer, cap, 180, 10, atm, OrderSettings(samples=settings.samples, max_order=5))
    np.testing.assert_allclose(full['category_mean_radiance'][:5], old['exact_order_radiance'], rtol=2e-14)
    assert full['all_order_radiance'] > full['cumulative_radiance'][4]
    assert full['max_luminous_order_seen'] > 5
    assert full['finite_scattering_order_truncation'] is False
    assert sum(full['path_outcomes'].values()) == settings.samples
    assert sum(r['luminous_paths'] for r in full['luminous_order_histogram'].values()) == sum(full['luminous_paths_by_category'])
    cov = np.array(full['mean_covariance'])
    assert np.linalg.eigvalsh(cov).min() > -1e-18
    assert full['all_order_standard_error']**2 == pytest.approx(float(cov.sum()))
    partial = relative_tail(full, 3)
    assert partial['increment_relative_to_partial'] > 0
    assert partial['missing_fraction_of_full'] == pytest.approx(1-full['cumulative_radiance'][2]/full['all_order_radiance'])


def test_recording_fewer_orders_does_not_change_the_all_order_solution():
    observer, cap = scene()
    settings = AllOrderSettings(samples=16384)
    five = integrate_all_orders(observer, cap, 180, 45, settings=settings)
    one = integrate_all_orders(observer, cap, 180, 45, settings=replace(settings, recorded_orders=1))
    assert five['all_order_radiance'] == pytest.approx(one['all_order_radiance'], rel=2e-14)
    assert five['all_order_standard_error'] == pytest.approx(one['all_order_standard_error'], rel=2e-14)
    assert five['max_scattering_order_seen'] == one['max_scattering_order_seen']
    assert five['path_outcomes'] == one['path_outcomes']
    doubled = integrate_all_orders(observer, replace(cap, radiance=2), 180, 45, settings=settings)
    assert doubled['all_order_radiance'] == 2*five['all_order_radiance']
    assert doubled['all_order_standard_error'] == 2*five['all_order_standard_error']


def test_implicit_and_analog_absorption_are_equivalent_for_conservative_medium():
    observer = Observer(0, 0, 0)
    cap = LambertianCap(0, 0, np.pi*EARTH_RADIUS_M)
    atm = Atmosphere(rayleigh_multiplier=0, aerosol_scale_height_m=1e6, aerosol_optical_depth=1e6,
        aerosol_single_scattering_albedo=1, aerosol_g=0, top_m=1)
    settings = AllOrderSettings(samples=8192)
    a = integrate_all_orders(observer, cap, 0, 90, atm, settings)
    b = integrate_all_orders(observer, cap, 0, 90, atm, replace(settings, absorption='analog'))
    assert a['category_mean_radiance'] == b['category_mean_radiance']
    assert a['path_outcomes'] == b['path_outcomes']


def test_physical_absorption_cross_check_and_bounded_bernoulli_variance():
    observer = Observer(0, 0, 0)
    cap = LambertianCap(0, 0, np.pi*EARTH_RADIUS_M)
    atm = Atmosphere(rayleigh_multiplier=0, aerosol_scale_height_m=1e6, aerosol_optical_depth=1e6,
        aerosol_single_scattering_albedo=.5, aerosol_g=0, top_m=1)
    settings = AllOrderSettings(samples=32768)
    a = integrate_all_orders(observer, cap, 0, 90, atm, settings)
    b = integrate_all_orders(observer, cap, 0, 90, atm, replace(settings, absorption='analog', seed=4091))
    assert abs(a['all_order_radiance']-b['all_order_radiance']) < 4*np.hypot(a['all_order_standard_error'], b['all_order_standard_error'])
    p = b['all_order_radiance']
    assert b['all_order_standard_error']**2 == pytest.approx(p*(1-p)/(settings.samples-1), rel=1e-13)
    assert b['path_outcomes']['absorbed'] > 0


@pytest.mark.parametrize('aod,ssa', [(0.,.9),(.2,0.)])
def test_no_false_radiance_or_zero_width_normal_interval_in_empty_or_absorbing_medium(aod,ssa):
    observer, cap = scene()
    atm = Atmosphere(rayleigh_multiplier=0, aerosol_optical_depth=aod, aerosol_single_scattering_albedo=ssa)
    r = integrate_all_orders(observer, cap, 180, 45, atm, AllOrderSettings(samples=256))
    assert r['all_order_radiance'] == 0
    assert r['all_order_normal_95_interval'] is None
    assert r['all_order_hoeffding_marginal_95_interval'][1] > 0
    assert relative_tail(r, 3)['increment_relative_to_partial'] is None


def test_safety_budget_is_failure_instead_of_a_finite_order_answer():
    observer, cap = scene()
    with pytest.raises(RuntimeError, match='NOT returned or truncated'):
        integrate_all_orders(observer,cap,180,10,settings=AllOrderSettings(samples=8192,safety_scattering_events=1))


@pytest.mark.parametrize('kwargs', [{'samples':1},{'seed':False},{'recorded_orders':0},{'recorded_orders':65},
    {'absorption':'delete_low_weights'},{'safety_scattering_events':0}])
def test_invalid_all_order_settings(kwargs):
    with pytest.raises(ValueError):
        AllOrderSettings(**kwargs)
