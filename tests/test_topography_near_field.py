from pathlib import Path

import numpy as np
import pytest

from lightpollution.adapters.terrain import load_terrain
from lightpollution.application.planner import Planner
from lightpollution.data import Bounds
from lightpollution.domain.scene import DomainError, SceneSpec
from lightpollution.interfaces.commands import offline_example
from lightpollution.models.terrain import ElevatedSources, TerrainGrid
from lightpollution.models.topographic_transfer import TopographicSolver
from lightpollution.physics import Atmosphere, EARTH_RADIUS_M, Observer, SolverSettings, Sources, emission, integrate_direction, unit_position
from lightpollution.sources import regional_sources, synthetic_cities


def test_height_bilinear_interpolation_and_coverage():
    lat, lon = np.array([0., .1]), np.array([0., .1])
    t = TerrainGrid(lat, lon, np.array([[10., 20.], [30., 40.]]), {})
    assert t.sample(.05, .05) == pytest.approx(25)
    assert t.sample(.025, .075) == pytest.approx(22.5)
    with pytest.raises(ValueError, match='coverage'):
        t.sample(.11, .05)
    assert t.observer(Observer(.05, .05, 2)).height_m == pytest.approx(27)


def test_unknown_or_missing_terrain_never_becomes_a_flat_surface(tmp_path):
    for name in ['demo-misspelled', 'petersburg-copdem90']:
        with pytest.raises(DomainError) as error:
            load_terrain(name,tmp_path)
        assert error.value.code=='missing_terrain'


def test_ridge_occludes_segments_in_both_directions_and_observer_horizon():
    lat, lon = np.linspace(-.04, .04, 161), np.linspace(-.04, .04, 161)
    yy, xx = np.meshgrid(lat, lon, indexing='ij')
    height = 120*np.exp(-(xx/.001)**2)
    terrain = TerrainGrid(lat, lon, height, {})
    a = (EARTH_RADIUS_M+10)*unit_position(0, -.01)
    b = (EARTH_RADIUS_M+10)*unit_position(0, .01)
    assert not terrain.visible(a, b, 25)
    assert not terrain.visible(b, a, 25)
    high = (EARTH_RADIUS_M+400)*unit_position(0, .01)
    assert terrain.visible(a, high, 25)
    horizon = terrain.horizon(Observer(0, -.01, 10), [90., 270.], 1500, 25)
    assert horizon[0] > horizon[1]


def test_flat_terrain_reproduces_legacy_scattering_and_grey_spectral_agree(tmp_path):
    terrain = load_terrain('demo-flat', tmp_path)
    observer, atmosphere, settings = Observer(), Atmosphere(), SolverSettings(48, 8)
    sources = synthetic_cities()
    # Use exactly zero elevated heights for the analytic flat-sphere limit.
    solver = TopographicSolver(terrain, 200)
    grey = solver.direction(observer, sources, 180, 45, atmosphere, settings)
    legacy = integrate_direction(observer, sources, 180, 45, atmosphere, settings)
    assert grey == pytest.approx(legacy, rel=1e-9)
    spectral = solver.spectrum(observer, sources, 180, 45, atmosphere, settings, np.array([500., 550., 600.]), 1.3)
    assert spectral[1] == pytest.approx(grey, rel=1e-12)


def emitting_pixel():
    bounds = Bounds(30.3, 60.3, 30.3+1/240, 60.3+1/240)
    shape = bounds.shape
    arrays = {'NearNadir_Composite_Snow_Free': np.ones(shape),
        'NearNadir_Composite_Snow_Free_Quality': np.zeros(shape),
        'NearNadir_Composite_Snow_Free_Num': np.full(shape, 10), 'Land_Water_Mask': np.ones(shape)}
    observer = Observer(float(bounds.latitude[0]), float(bounds.longitude[0]), 20)
    return arrays, bounds, observer


def test_ridge_reduces_scattered_city_light_with_identical_emitters():
    latitude = longitude = np.linspace(-.1, .1, 401)
    yy, xx = np.meshgrid(latitude, longitude, indexing='ij')
    ridge = TerrainGrid(latitude, longitude, 120*np.exp(-(xx/.001)**2), {})
    flat = TerrainGrid(latitude, longitude, np.zeros(xx.shape), {})
    observer = Observer(0, -.01, 10)
    sources = ElevatedSources(np.asarray([(EARTH_RADIUS_M+10)*unit_position(0,.01)]), np.asarray([1.]))
    settings = SolverSettings(96,12)
    a = TopographicSolver(flat,25).direction(observer,sources,0,90,Atmosphere(),settings)
    b = TopographicSolver(ridge,25).direction(observer,sources,0,90,Atmosphere(),settings)
    assert 0 < b < a


@pytest.mark.parametrize('height', [2., 20.])
def test_finite_cell_conserves_power_and_matches_rectangular_solid_angle(height):
    arrays, bounds, observer = emitting_pixel()
    observer = Observer(observer.latitude_deg, observer.longitude_deg, height)
    scene = offline_example()
    scene['near_field'] = {}
    defaults = SceneSpec.from_mapping(scene).to_dict()['near_field']
    sources, report = regional_sources(arrays, bounds, observer, exclude_radius_m=0,
        finite_cells=defaults)
    assert sources.power.sum() == pytest.approx(1, rel=1e-12)
    assert report['finite_emitting_pixels'] == 1 and len(sources.power)>100
    point = observer.position
    delta = point-sources.positions_m
    r = np.linalg.norm(delta, axis=1)
    mu = np.sum(delta/r[:, None]*(sources.positions_m/EARTH_RADIUS_M), axis=1)
    actual = np.sum(sources.power*emission(mu)/r**2)
    h = observer.height_m
    a = np.deg2rad(1/480)*EARTH_RADIUS_M*np.cos(np.deg2rad(observer.latitude_deg))
    b = np.deg2rad(1/480)*EARTH_RADIUS_M
    solid_angle = 4*np.arctan(a*b/(h*np.sqrt(h*h+a*a+b*b)))
    expected = solid_angle/(np.pi*4*a*b)
    assert actual == pytest.approx(expected, rel=.002)
    # A single central point source has an artificial large near-field peak.
    central, _ = regional_sources(arrays, bounds, observer, exclude_radius_m=0)
    point_source = 1/(np.pi*h*h)
    assert point_source > actual*10


def test_near_field_retains_previously_excluded_pixels_at_fixed_normalisation():
    arrays, bounds, observer = emitting_pixel()
    with pytest.raises(ValueError, match='No sources'):
        regional_sources(arrays, bounds, observer)
    a, _ = regional_sources(arrays, bounds, observer, exclude_radius_m=0,
        normalisation_radiance_area=200000., finite_cells={'radius_m': 3000., 'minimum_patch_m': 10., 'quadrature_order': 2})
    observer2 = Observer(observer.latitude_deg+.0005, observer.longitude_deg, 20)
    b, _ = regional_sources(arrays, bounds, observer2, exclude_radius_m=0,
        normalisation_radiance_area=200000., finite_cells={'radius_m': 3000., 'minimum_patch_m': 10., 'quadrature_order': 2})
    assert a.power.sum() == pytest.approx(b.power.sum(), rel=1e-12)


@pytest.mark.parametrize('field,value', [('terrain', {'dataset':'unknown'}),
    ('terrain', {'dataset':'demo-flat','visibility_step_m':0}),
    ('near_field', {'minimum_patch_m':0}), ('near_field', {'quadrature_order':2.5})])
def test_invalid_topography_configuration(field, value):
    data = offline_example(); data[field] = value
    with pytest.raises(DomainError):
        SceneSpec.from_mapping(data)


def test_planner_exposes_datum_and_horizon_without_fake_building_reconstruction(tmp_path):
    data = offline_example()
    data.update(terrain={'dataset':'demo-flat'}, near_field={}, directions=[[0,90]], profile='quick')
    result = Planner(tmp_path).run(SceneSpec.from_mapping(data)).value
    assert result['data_provenance']['QA']['local_exclusion_radius_m'] == 0
    assert result['terrain']['source_to_scatter_occlusion']
    assert result['terrain']['scatter_to_observer_occlusion']
    assert not result['terrain']['individual_buildings_or_lamps_reconstructed']
    assert result['sky']['directions'][0]['terrain_obstructed'] is False
    assert 'observer_msl' in result['resolved_scene']
