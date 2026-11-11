from copy import deepcopy
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from lightpollution.application.eye import build_eye_result
from lightpollution.application.planner import Planner,export_result
from lightpollution.domain.scene import DomainError,SceneSpec
from lightpollution.interfaces.commands import offline_example
from lightpollution.interfaces.eye_view import render_eye_view
from lightpollution.models.eye import (REFERENCE_SP,V22_LUMINANCE,PHOTOPIC_V0_LUX,
    point_threshold,evaluate_detection,eye_config)


@pytest.mark.parametrize('factor,expected',[(1.,6.93),(2.,6.18)])
def test_published_Crumey_dark_sky_reference(factor,expected):
    magnitude=-2.5*np.log10(point_threshold(2e-4,factor)/PHOTOPIC_V0_LUX)
    assert magnitude==pytest.approx(expected,abs=.015)


def test_published_equation54_shape_over_its_measured_domain():
    mu=np.linspace(20.01,21.99,101)
    b=PHOTOPIC_V0_LUX*(180*3600/np.pi)**2*10**(-.4*mu)
    exact=-2.5*np.log10(point_threshold(b,2)/PHOTOPIC_V0_LUX)
    published=.3834*mu-1.4400-2.5*np.log10(2)
    assert np.max(abs(exact-published))<.011


def test_background_floor_monotonicity_and_field_factor():
    b=np.geomspace(1e-5,3e-3,50)
    assert np.all(np.diff(point_threshold(b))>0)
    assert point_threshold(0)==point_threshold(1e-5)
    assert point_threshold(2e-4,4)==2*point_threshold(2e-4,2)
    with pytest.raises(ValueError):point_threshold(.00301)
    with pytest.raises(ValueError):point_threshold(-1)


def test_spectral_equivalence_and_single_stellar_transmission():
    b=2e-4;t=float(point_threshold(b,2))
    full=evaluate_detection(b,b*REFERENCE_SP,2*t*REFERENCE_SP,2)
    half=evaluate_detection(b,b*REFERENCE_SP,t*REFERENCE_SP,2)
    assert full['flux_over_threshold']==pytest.approx(2)
    assert half['flux_over_threshold']==pytest.approx(1)
    assert full['margin_mag']-half['margin_mag']==pytest.approx(2.5*np.log10(2))
    red=evaluate_detection(b,b*REFERENCE_SP,.7*t,2)
    blue=evaluate_detection(b,b*REFERENCE_SP,2.8*t,2)
    assert blue['flux_over_threshold']/red['flux_over_threshold']==pytest.approx(4)


def physical_fixture():
    b=2e-4;t=float(point_threshold(b,2))*REFERENCE_SP
    field={'azimuth_deg':[0.,180.],'altitude_deg':[10.,90.],
        'city_photopic_luminance_cd_m2':[[0.,0.],[0.,0.]],
        'city_scotopic_luminance_cd_m2':[[0.,0.],[0.,0.]],
        'unresolved_v_relative':np.full((2,2),b/V22_LUMINANCE).tolist(),
        'physical_supported':[[True,True],[True,True]]}
    result={'request':{'mode':'dated','time':'2026-11-11T21:00:00Z','directions':[[180.,50.]],
        'observer':{'latitude_deg':60.3,'longitude_deg':30.315,'height_m':2.}},
        'observation':{'geometry':{'sun':{'altitude_deg':-30}},'moon':{'altitude_deg':-20}},
        'sky':{'directions':[{'azimuth_deg':180.,'altitude_deg':50.}]},
        'stars':{'rows':[{'hr':i,'name':str(i),'azimuth_deg':180.,'altitude_deg':50.,
            'catalogue_v_mag':5.8,'observed_v_mag':6.,'scotopic_illuminance_lux':e}
            for i,e in [(1,1.1*t),(2,.9*t),(3,None)]]}}
    config=eye_config({'view':{'width':640,'height':480,'horizontal_fov_deg':40.}})
    result['eye']=build_eye_result(result,config,field,PHOTOPIC_V0_LUX*REFERENCE_SP)
    return result


def test_threshold_classification_missing_spectrum_and_no_probability():
    eye=physical_fixture()['eye']
    assert eye['summary']['above_threshold']==1
    assert eye['summary']['below_threshold']==1
    assert eye['summary']['unavailable']==1
    assert eye['stars'][2]['above_threshold'] is None
    assert eye['stars'][2]['star_equivalent_illuminance_lux'] is None
    assert eye['stars'][0]['margin_mag']==pytest.approx(2.5*np.log10(1.1))
    assert eye['summary']['not_a_detection_probability']


@pytest.mark.parametrize('restriction',['sun','moon','horizon'])
def test_no_supported_image_when_global_eye_assumptions_fail(restriction,tmp_path):
    result=physical_fixture()
    if restriction=='sun':result['observation']['geometry']['sun']['altitude_deg']=-10
    if restriction=='moon':result['observation']['moon']['altitude_deg']=20
    if restriction=='horizon':result['integration']={'artificial_horizon_visible':False}
    old=result['eye'];result['eye']=build_eye_result(result,old['config'],old['field'],PHOTOPIC_V0_LUX*REFERENCE_SP)
    assert not result['eye']['available']
    display=render_eye_view(result,tmp_path)
    assert not display['available'] and not (tmp_path/'eye-view.png').exists()


def test_bright_background_not_extrapolated_as_scotopic_vision(tmp_path):
    result=physical_fixture();eye=result['eye']
    field=deepcopy(eye['field']);field['unresolved_v_relative']=np.full((2,2),.02/V22_LUMINANCE).tolist()
    result['eye']=build_eye_result(result,eye['config'],field,PHOTOPIC_V0_LUX*REFERENCE_SP)
    assert all(r['above_threshold'] is None for r in result['eye']['stars'])
    assert all(r['threshold_scotopic_illuminance_lux'] is None for r in result['eye']['directions'])
    assert not render_eye_view(result,tmp_path)['available']


def test_reused_output_directory_never_returns_previous_supported_image(tmp_path):
    result=physical_fixture()
    assert render_eye_view(result,tmp_path)['available']
    result['eye'].update(available=False,global_restrictions=['astronomical_darkness_required'])
    assert not render_eye_view(result,tmp_path)['available']
    assert not (tmp_path/'eye-view.png').exists()
    assert (tmp_path/'eye-view.previous.png').is_file()


def test_pixel_count_does_not_change_the_eye_detection_model(tmp_path):
    result=physical_fixture();a=render_eye_view(result,tmp_path/'small')
    result['eye']['config']['view'].update(width=1280,height=960)
    b=render_eye_view(result,tmp_path/'large')
    assert a['displayed_stars']==b['displayed_stars']==1
    assert a['catalogue_stars_in_view']==b['catalogue_stars_in_view']==3
    assert Image.open(tmp_path/'large/eye-view.png').size==(1280,1130)


@pytest.mark.parametrize('config',[{'observer_factor':0},{'observer_factor':True},
    {'unresolved_background_sp_ratio':float('nan')},{'adaptation':'partial'},
    {'unknown':1},{'view':{'width':1}},{'view':{'width':640.5}},
    {'view':{'altitude_deg':-5}}])
def test_invalid_eye_settings_rejected(config):
    data=offline_example();data['eye']=config
    with pytest.raises(DomainError):SceneSpec.from_mapping(data)


def test_eye_requires_date_and_enables_physical_spectra():
    data=offline_example();data['eye']={}
    assert SceneSpec.from_mapping(data).to_dict()['spectrum']['stars']=='catalogue_continuum'
    data.update(mode='undated',moon={'mode':'none'});data.pop('time')
    with pytest.raises(DomainError):SceneSpec.from_mapping(data)


def test_eye_changes_reuse_transfer_and_never_refit_city_sources(tmp_path):
    class SpectralSolver:
        version='test-constant-spectral-1'
        calls=0
        def spectrum(self,observer,sources,az,alt,atm,settings,wavelengths,angstrom):
            self.calls+=1
            return np.ones(len(wavelengths))
    solver=SpectralSolver();planner=Planner(tmp_path,solver=solver)
    data=offline_example();data.update(profile='quick',directions=[[180,50]],eye={})
    a=planner.run(SceneSpec.from_mapping(data)).value;n=solver.calls
    assert not a['stars']['metadata']['detected_by_eye_not_modelled']
    assert not a['observation']['stars_metadata']['detected_by_eye_not_modelled']
    assert 'eye_detection' not in a['unavailable_capabilities']
    data['eye']['observer_factor']=3.
    b=planner.run(SceneSpec.from_mapping(data)).value
    assert solver.calls==n and b['cache']['transfer_misses']==0
    assert a['calibration']['source_coefficient']==b['calibration']['source_coefficient']
    assert a['comparison_basis_id']==b['comparison_basis_id']
    assert a['stars']['rows']==b['stars']['rows']
    assert a['eye']['summary']['above_threshold']>=b['eye']['summary']['above_threshold']
