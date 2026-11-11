"""Analytic limits, literature coefficients and application support contracts."""

from copy import deepcopy
import json

from fastapi.testclient import TestClient
import numpy as np
import pytest

from lightpollution.application.planner import Planner, compare_results, export_result
from lightpollution.domain.calibration import calibration_key
from lightpollution.domain.scene import DomainError, SceneSpec
from lightpollution.interfaces.commands import offline_example
from lightpollution.interfaces.web import create_app
from lightpollution.models.background import (BackgroundModel, background_config,
    direct_optical_depth, patat_v_magnitude, van_rhijn)
from lightpollution.models.surface import surface_balance, surface_config
from lightpollution.physics import Atmosphere, EARTH_RADIUS_M, Observer
from lightpollution.skyview import horizontal_stars, load_stars


def scene(dated=False, spectral=False):
    data = offline_example()
    data['directions'] = [[0, 90], [180, 50], [0, 45]]
    data['profile'] = 'quick'
    if not dated:
        data.pop('time')
        data.update(mode='undated', moon={'mode': 'none'})
    if spectral:
        data['spectrum'] = {}
    return data


class AngularSolver:
    version = 'analytic-separable-angular-kernel-for-tests'

    def direction(self, observer, sources, az, alt, atmosphere, settings):
        return float(np.sum(sources.power)*(1+settings.horizontal_fraction*(1+az/180)))

    def spectrum(self, observer, sources, az, alt, atmosphere, settings, wavelengths, exponent):
        return self.direction(observer, sources, az, alt, atmosphere, settings)*np.ones_like(wavelengths)


def test_thin_shell_geometry_and_radial_beer_analytic_limits():
    observer, atmosphere = Observer(), Atmosphere()
    assert van_rhijn(90) == 1
    assert van_rhijn(0) == pytest.approx(1/np.sqrt(1-(EARTH_RADIUS_M/(EARTH_RADIUS_M+90000))**2))
    assert van_rhijn(30) > van_rhijn(60) > 1
    h, top = observer.height_m, atmosphere.top_m
    expected = atmosphere.rayleigh_sea_level*atmosphere.rayleigh_scale_height_m*(
        np.exp(-h/atmosphere.rayleigh_scale_height_m)-np.exp(-top/atmosphere.rayleigh_scale_height_m))
    expected += atmosphere.aerosol_optical_depth*(np.exp(-h/atmosphere.aerosol_scale_height_m)
        - np.exp(-top/atmosphere.aerosol_scale_height_m))
    assert direct_optical_depth(90, observer, atmosphere) == pytest.approx(expected, rel=1e-10)
    assert direct_optical_depth(10, observer, atmosphere) > expected


def test_natural_component_sum_zenith_and_vacuum_shape():
    cfg = background_config({})
    model = BackgroundModel(cfg, Observer(), Atmosphere(rayleigh_multiplier=0, aerosol_optical_depth=0), 21.6)
    rows = model.evaluate([0, 0], [90, 20], None)
    q0 = 10**(.16)
    assert rows[0]['natural_relative'] == pytest.approx(q0)
    assert rows[1]['natural_relative'] == pytest.approx(q0*(.4+.6*van_rhijn(20, 90000, 20)))
    for row in rows:
        assert row['natural_relative'] == row['airglow_relative']+row['diffuse_residual_relative']
    assert not model.metadata()['catalogue_stars_included']
    assert not model.metadata()['galactic_and_zodiacal_maps_available']


@pytest.mark.parametrize('depression,expected', [(5, 11.840), (6, 13.301), (9, 17.000),
                                               (12, 19.673), (15, 21.320)])
def test_patat_table1_polynomial_reproduction(depression, expected):
    assert patat_v_magnitude(depression) == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize('depression', [4.999, 15.001, float('nan')])
def test_patat_does_not_extrapolate(depression):
    with pytest.raises(ValueError):
        patat_v_magnitude(depression)


def test_twilight_floor_is_not_counted_twice_or_subtracted_from_local_night():
    cfg = background_config({'twilight': 'patat2006'})
    model = BackgroundModel(cfg, Observer(), Atmosphere(), 21.6)
    rows = model.evaluate([0, 0, 0], [90, 50, 49.999], {'sun': {'altitude_deg': -12.}})
    expected_total = 10**(-.4*(19.673-22))
    assert rows[0]['natural_relative']+rows[0]['twilight_relative'] == pytest.approx(expected_total)
    other = BackgroundModel(cfg, Observer(), Atmosphere(), 22.8)
    assert other.evaluate(0, 90, {'sun': {'altitude_deg': -12}})[0]['twilight_relative'] == rows[0]['twilight_relative']
    assert rows[1]['background_supported'] and not rows[2]['background_supported']
    for sun_alt in [-17.99, -16, -15.01, -4.99, 5]:
        row = model.evaluate(0, 90, {'sun': {'altitude_deg': sun_alt}})[0]
        assert not row['background_supported'] and row['twilight_relative'] is None
    assert model.evaluate(0, 90, {'sun': {'altitude_deg': -18}})[0]['twilight_relative'] == 0


def test_surface_energy_conservation_and_snow_free_limit_on_parameter_grid():
    checked = 0
    for u in [0, .15, .5, 1]:
        for bare in [.01, .15, 1]:
            for snow in [0, .8, 1]:
                for fraction in [0, .5, 1]:
                    cfg = surface_config({'direct_upward_fraction': u, 'bare_albedo': bare,
                                          'snow_albedo': snow, 'snow_fraction': fraction})
                    b = surface_balance(cfg)
                    assert b['direct_fraction_of_lamp_power']+b['reflected_fraction_of_lamp_power']+b['absorbed_fraction_of_lamp_power'] == pytest.approx(1)
                    assert 0 <= b['upward_fraction_of_lamp_power'] <= 1
                    assert b['direct_multiplier_of_reference_upward_power']+b['reflected_multiplier_of_reference_upward_power'] == pytest.approx(b['upward_power_over_snow_free'])
                    assert surface_balance(cfg, reference=True)['upward_power_over_snow_free'] == 1
                    if fraction == 0:
                        assert b['upward_power_over_snow_free'] == 1
                    checked += 1
    assert checked == 108


@pytest.mark.parametrize('key,value', [('surface', {'snow_fraction': 1.1}),
    ('surface', {'bare_albedo': 0, 'direct_upward_fraction': 0}),
    ('surface', {'snow_albedo': True}), ('surface', {'automatic_future_snow': True}),
    ('background', {'airglow_fraction': -1}), ('background', {'twilight': 'extrapolate'}),
    ('background', {'airglow_height_m': 1000})])
def test_invalid_scenarios_are_rejected_before_work(key, value):
    data = scene()
    data[key] = value
    with pytest.raises(DomainError, match='.'):
        SceneSpec.from_mapping(data)


@pytest.mark.parametrize('spectral', [False, True])
def test_fixed_lamp_snow_changes_only_reflected_component_and_preserves_common_scale(tmp_path, spectral):
    planner = Planner(tmp_path, solver=AngularSolver())
    data = scene(spectral=spectral)
    legacy = planner.run(SceneSpec.from_mapping(data)).value
    data['surface'] = {}
    bare = planner.run(SceneSpec.from_mapping(data)).value
    assert bare['calibration']['source_coefficient'] == pytest.approx(legacy['calibration']['source_coefficient'])
    assert [r['artificial_relative'] for r in bare['sky']['directions']] == pytest.approx([r['artificial_relative'] for r in legacy['sky']['directions']])
    data['surface'] = {'snow_fraction': 1}
    snow = planner.run(SceneSpec.from_mapping(data)).value
    assert bare['comparison_basis_id'] == snow['comparison_basis_id']
    assert bare['calibration']['source_coefficient'] == snow['calibration']['source_coefficient']
    expected = (.15+.85*.8)/(.15+.85*.15)
    for a, b in zip(bare['sky']['directions'], snow['sky']['directions'], strict=True):
        assert b['artificial_direct_relative'] == a['artificial_direct_relative']
        assert b['artificial_relative'] == pytest.approx(a['artificial_relative']*expected)
        assert b['artificial_relative'] == pytest.approx(b['artificial_direct_relative']+b['artificial_reflected_relative'])
    assert all(r['supported'] for r in compare_results(bare, snow)['directions'])
    if spectral:
        assert np.asarray(snow['spectral']['directions'][0]['city_radiance_W_m2_sr_nm']) == pytest.approx(np.asarray(bare['spectral']['directions'][0]['city_radiance_W_m2_sr_nm'])*expected)
    assert planner.run(SceneSpec.from_mapping(data)).value['cache']['scene_hit']


def test_angular_components_do_not_reduce_to_one_snow_sky_multiplier(tmp_path):
    planner = Planner(tmp_path, solver=AngularSolver())
    data = scene()
    data['surface'] = {'direct_horizontal_fraction': .5}
    bare = planner.run(SceneSpec.from_mapping(data)).value
    data['surface']['snow_fraction'] = 1
    snow = planner.run(SceneSpec.from_mapping(data)).value
    ratios = [b['artificial_relative']/a['artificial_relative'] for a, b in
              zip(bare['sky']['directions'], snow['sky']['directions'], strict=True)]
    assert max(ratios)-min(ratios) > .1
    changed = deepcopy(data)
    changed['surface']['bare_albedo'] = .2
    other = planner.run(SceneSpec.from_mapping(changed)).value
    with pytest.raises(DomainError, match='.'):
        compare_results(snow, other)
    calibration = {'schema_version': 1, 'scope': bare['calibration']['scope'],
                   'source_coefficient': bare['calibration']['source_coefficient']}
    calibration['id'] = calibration_key(calibration)
    data['photometric_calibration'] = calibration
    assert planner.run(SceneSpec.from_mapping(data)).value['calibration']['origin'] == 'provided_calibration'
    data['surface']['direct_upward_fraction'] = .4
    with pytest.raises(DomainError, match='.'):
        planner.run(SceneSpec.from_mapping(data))


def test_white_night_partial_support_and_export_without_fake_zero(tmp_path):
    data = scene(dated=True)
    data['time'] = '2026-06-21T22:00:00Z'
    data['background'] = {'twilight': 'patat2006'}
    planner = Planner(tmp_path, solver=AngularSolver())
    result = planner.run(SceneSpec.from_mapping(data))
    value = result.value
    assert -15 < value['observation']['geometry']['sun']['altitude_deg'] < -5
    first, middle, last = value['sky']['directions']
    assert first['supported'] and middle['supported']
    assert first['twilight_relative'] > first['natural_relative']
    assert not last['supported'] and last['total_relative'] is None and last['twilight_relative'] is None
    assert last['artificial_relative'] > 0 and last['natural_relative'] > 0
    assert value['stars']['metadata']['exported'] > 1000
    assert 'bounded_empirical_twilight' in value['capabilities']
    export_result(result, tmp_path/'out')
    text = (tmp_path/'out/result.json').read_text()
    assert 'NaN' not in text and 'Infinity' not in text
    assert json.loads(text)['sky']['directions'][-1]['total_relative'] is None
    assert 'twilight_relative' in (tmp_path/'out/directions.csv').read_text()
    # The existing renderer remains a night-only API unless explicitly opted in.
    with pytest.raises(ValueError, match='Sun'):
        horizontal_stars(load_stars()[0], Observer(), data['time'], allow_moon=True)


def test_http_background_validation_and_api_scene_roundtrip(tmp_path):
    app = create_app(tmp_path)
    with TestClient(app) as client:
        data = scene()
        data.update(background={'twilight': 'patat2006'}, surface={'snow_fraction': .7})
        response = client.post('/api/jobs', json=data)
        assert response.status_code == 202
        job = client.get('/api/jobs/' + response.json()['id']).json()
        assert job['request']['surface']['snow_fraction'] == .7
        assert job['request']['background']['airglow_height_m'] == 90000
        data['surface']['snow_fraction'] = -1
        response = client.post('/api/jobs', json=data)
        assert response.status_code == 422 and response.json()['error']['code'] == 'invalid_surface'


def test_moon_and_twilight_restrictions_are_preserved_together(tmp_path, monkeypatch):
    from lightpollution.models import observation as module
    original = module.observation_context
    def context(*args):
        value = original(*args)
        value['sun']['altitude_deg'] = -16.
        value['moon'].update(altitude_deg=45., azimuth_deg=0., possible_lunar_eclipse=False)
        return value
    monkeypatch.setattr(module, 'observation_context', context)
    data = scene(dated=True)
    data['background'] = {'twilight': 'patat2006'}
    value = Planner(tmp_path, solver=AngularSolver()).run(SceneSpec.from_mapping(data)).value
    row = value['sky']['directions'][-1]
    assert row['status'] == 'multiple_model_restrictions'
    assert not row['moon_supported'] and not row['background_supported']
    assert set(row['support_reasons']) == {'moon_separation_below_10deg', 'twilight_solar_domain_gap'}
    assert row['moon_relative'] is None and row['total_relative'] is None
