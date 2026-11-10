from copy import deepcopy
from dataclasses import asdict
import json
import sys

import numpy as np
import pytest

from lightpollution import cli
from lightpollution.moonlight import (REFERENCE_DISTANCE_KM, nanolambert_to_v_mag,
                                     scattered_moonlight, scattering_airmass,
                                     v_mag_to_nanolambert)
from lightpollution.observation import observation_context
from lightpollution.photometry import (KS_V_BAND, PhotometricBand, RelativeBrightness,
                                      RelativeScale, SurfaceBrightness)
from lightpollution.physics import Atmosphere, Observer
from lightpollution.scene import calculate_observation, vertical_extinction_v, write_observation


@pytest.fixture
def scene():
    return {'schema_version':1, 'mode':'undated',
            'observer':asdict(Observer()), 'directions':[[0,90],[0,45],[180,30]],
            'moon':{'mode':'manual','azimuth_deg':180,'altitude_deg':30,'phase_angle_deg':60},
            'natural':{'mode':'uniform','v_mag_arcsec2':21.6,'source':'Test scenario'},
            'artificial':{'mode':'none'}}


def add_grid(scene, path):
    np.savez(path,azimuth_deg=[0,90,180,270],altitude_deg=[0,90],
             radiance=[[1,2,3,2],[1,1,1,1]],
             metadata_json=json.dumps({'observer':scene['observer'],'atmosphere':asdict(Atmosphere())}))
    scene['artificial']={'mode':'grid','sky_grid':path.name,'anchor_direction':[0,90],
                         'anchor_v_mag_arcsec2':20.5,'scale_origin':'scenario','source':'Test anchor',
                         'band_bridge':'effective_550nm_shape_anchored_in_V'}


@pytest.mark.parametrize('phase,expected', [
    (30,[1160,530,437,818]), (60,[541,247,204,381]),
    (90,[217,99,82,153]), (120,[63,29,24,44])])
def test_original_paper_table2_at_published_rounding(phase, expected):
    # PASP 103, p1038 Table 2: Moon Z=60, k=.172; target lies on Moon-zenith great circle.
    rho = np.array([30,60,90,120])
    values = scattered_moonlight(phase,30,90-np.abs(60-rho),rho,.172)
    np.testing.assert_allclose(values,expected,rtol=0,atol=.5)


def test_independent_specsim_published_example_and_paper_conversion():
    # specsim documented example: obs Z20, Moon Z70, rho50, phase45, k=.15.
    brightness = scattered_moonlight(45,20,70,50,.15)
    assert float(nanolambert_to_v_mag(brightness)) == pytest.approx(19.855,abs=.0005)
    np.testing.assert_allclose(v_mag_to_nanolambert(nanolambert_to_v_mag(brightness)),brightness)


def test_moon_physical_limits_phase_and_distance():
    full = scattered_moonlight(0,30,90,60,.172)
    quarter = scattered_moonlight(90,30,90,60,.172)
    assert quarter/full < .15  # Illuminated fraction .5 is NOT half the full-Moon flux.
    assert scattered_moonlight(30,30,90,60,0) == 0
    assert scattered_moonlight(30,-1,90,60,.172) == 0
    np.testing.assert_allclose(scattered_moonlight(0,30,90,60,.172,2*REFERENCE_DISTANCE_KM),full/4)
    np.testing.assert_allclose(scattering_airmass([90,0]),[1,5])
    with pytest.raises(ValueError,match='separation'): scattered_moonlight(30,30,90,1,.172)
    with pytest.raises(ValueError): scattered_moonlight(np.nan,30,90,60,.172)


def test_common_scale_zeros_and_no_implicit_rescaling():
    scale = RelativeScale()
    one = scale.from_magnitudes(SurfaceBrightness(22.,KS_V_BAND))
    zero = RelativeBrightness([0.,0.],scale)
    np.testing.assert_array_equal((one+zero).values,[1,1])
    np.testing.assert_allclose((one+one).magnitudes(),22-2.5*np.log10(2))
    with pytest.raises(ValueError): one + RelativeBrightness([1.],RelativeScale(reference_mag_arcsec2=21))
    with pytest.raises(ValueError): RelativeBrightness([-1],scale)
    with pytest.raises(ValueError): RelativeBrightness([np.nan],scale)
    with pytest.raises(ValueError): scale.from_magnitudes(SurfaceBrightness(22,PhotometricBand('SQM','native','native')))


def test_scene_linear_sum_unknown_near_moon_and_reference_invariance(scene):
    report = calculate_observation(scene)
    first = report['directions'][0]
    assert first['total_relative'] == pytest.approx(first['natural_relative']+first['moon_relative'])
    last = report['directions'][-1]
    assert last['moon_relative'] is None and last['total_relative'] is None
    assert not last['supported']
    scene['reference_v_mag_arcsec2'] = 20.
    other = calculate_observation(scene)['directions'][0]
    assert other['total_v_mag_arcsec2'] == pytest.approx(first['total_v_mag_arcsec2'])
    assert other['moon_relative']/other['natural_relative'] == pytest.approx(first['moon_relative']/first['natural_relative'])
    assert not report['stars_metadata']['available']


def test_sky_grid_calibration_and_input_origin_are_enforced(scene,tmp_path):
    add_grid(scene,tmp_path/'field.npz')
    scene['moon']={'mode':'none'}
    report=calculate_observation(scene,tmp_path)
    first=report['directions'][0]
    assert first['artificial_relative'] == pytest.approx(10**(-.4*(20.5-22)))
    assert report['artificial_provenance']['calibration_independently_verified'] is False
    scene['observer']['height_m'] += 1
    with pytest.raises(ValueError,match='match'): calculate_observation(scene,tmp_path)
    scene['observer']['height_m'] -= 1
    scene['atmosphere']={'aerosol_optical_depth':.2}
    with pytest.raises(ValueError,match='match'): calculate_observation(scene,tmp_path)
    del scene['atmosphere']
    scene['artificial']['band_bridge']='DNB_equals_V'
    with pytest.raises(ValueError,match='bridge'): calculate_observation(scene,tmp_path)
    scene['artificial']['band_bridge']='effective_550nm_shape_anchored_in_V'
    with np.load(tmp_path/'field.npz',allow_pickle=False) as data:
        corrupted={key:np.array(data[key]) for key in data.files}
    corrupted['radiance'][-1,-1]=2
    np.savez(tmp_path/'field.npz',**corrupted)
    with pytest.raises(ValueError,match='zenith'): calculate_observation(scene,tmp_path)


@pytest.mark.parametrize('change',[
    {'mode':'unknown'}, {'cloud_free':False}, {'time':'2025-01-12T21:00:00Z'},
    {'moon':{'mode':'none','phase_angle_deg':30}}, {'moon':{'mode':'manual'}},
    {'moon':{'mode':'manual','azimuth_deg':0,'altitude_deg':30,'phase_angle_deg':181}},
    {'directions':[[0,-1]]}, {'schema_version':True}, {'unknown_option':123},
    {'atmosphere':{'reference_wavelength_nm':600}},
])
def test_invalid_or_ambiguous_scenes_fail_before_any_output(scene,change):
    scene.update(change)
    with pytest.raises(ValueError): calculate_observation(scene)


def test_no_date_requires_explicit_moon_and_dated_requires_time_zone(scene):
    del scene['moon']
    with pytest.raises(ValueError): calculate_observation(scene)
    scene.update(mode='dated',time='2025-01-12T21:00:00')
    with pytest.raises(ValueError,match='timezone'): calculate_observation(scene)
    scene['time']='2025-01-12'
    with pytest.raises(ValueError,match='date AND time'): calculate_observation(scene)
    scene.update(time='2025-01-12T21:00:00Z',moon={'mode':'none'})
    with pytest.raises(ValueError,match='manual'): calculate_observation(scene)


def test_dated_matches_same_manual_geometry_and_exports_stars(scene):
    dated=deepcopy(scene)
    dated.pop('moon')
    dated.update(mode='dated',time='2025-01-13T00:00:00+03:00')
    result=calculate_observation(dated)
    assert result['utc']=='2025-01-12T21:00:00Z'
    moon=result['moon']
    assert moon['altitude_deg']>0 and not moon['possible_lunar_eclipse']
    scene['moon']={'mode':'manual',**{key:moon[key] for key in
                                   ('azimuth_deg','altitude_deg','phase_angle_deg','distance_km')}}
    assert result['directions']==calculate_observation(scene)['directions']
    assert result['stars_metadata']['exported']>1000
    star=result['stars'][0]
    assert star['observed_v_mag']>=star['catalogue_v_mag']
    assert star['flux_relative_to_v_mag0']==pytest.approx(10**(-.4*star['observed_v_mag']))


def test_eclipse_is_excluded_and_twilight_rejected(scene):
    observer=Observer(19.826,-155.47,2800)
    assert observation_context(observer,'2025-03-14T07:00:00Z')['moon']['possible_lunar_eclipse']
    scene.pop('moon')
    scene.update(mode='dated',time='2025-03-14T07:00:00Z',observer=asdict(observer))
    report=calculate_observation(scene)
    assert all(row['total_relative'] is None and row['status']=='possible_lunar_eclipse'
               for row in report['directions'])
    scene.update(observer=asdict(Observer()),time='2025-06-21T00:00:00Z')
    with pytest.raises(ValueError,match='twilight'): calculate_observation(scene)


def test_vertical_extinction_matches_analytic_zero_aerosol_column():
    atmosphere=Atmosphere(aerosol_optical_depth=0)
    observer=Observer(height_m=1000)
    expected=2.5/np.log(10)*atmosphere.rayleigh_sea_level*8000*(np.exp(-1000/8000)-np.exp(-80000/8000))
    assert vertical_extinction_v(observer,atmosphere)==pytest.approx(expected)


def test_cli_strict_json_and_reusing_output_has_no_stale_stars(scene,tmp_path,monkeypatch):
    path=tmp_path/'scene.json'
    output=tmp_path/'output'
    output.mkdir()
    (output/'stars.csv').write_text('stale dated data')
    path.write_text(json.dumps(scene))
    monkeypatch.setattr(sys,'argv',['lightpollution','observe','--scene',str(path),'--output',str(output)])
    cli.main()
    report=json.loads((output/'observation.json').read_text(),parse_constant=lambda value:pytest.fail(value))
    assert report['directions'][-1]['total_relative'] is None
    assert 'stale' not in (output/'stars.csv').read_text()
    assert b'\r\n' not in (output/'directions.csv').read_bytes()
    assert len((output/'stars.csv').read_text().splitlines())==1
    bad=deepcopy(scene)
    bad['cloud_free']=False
    path.write_text(json.dumps(bad))
    monkeypatch.setattr(sys,'argv',['lightpollution','observe','--scene',str(path),'--output',str(tmp_path/'bad')])
    with pytest.raises(SystemExit): cli.main()
    assert not (tmp_path/'bad').exists()
