from copy import deepcopy
from dataclasses import asdict

from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import pytest

from lightpollution.adapters.spectral_library import load_spectral_library
from lightpollution.application.camera import build_camera_result
from lightpollution.application.planner import Planner,export_result
from lightpollution.domain.scene import DomainError,SceneSpec
from lightpollution.interfaces.camera_view import export_camera,render_camera_arrays
from lightpollution.interfaces.commands import offline_example
from lightpollution.models.camera import (PLANCK,LIGHT_SPEED,camera_config,camera_view,
    response_weights,sample_detector,read_noise_lower_bound)
from lightpollution.models.camera_field import deposit_electrons,project_star,directions,build_camera_patch
from lightpollution.models.spectra import band_integral,blackbody_shape
from lightpollution.physics import Atmosphere,Observer


def camera_fixture():
    observer=Observer(60.3,30.315,20.);library=load_spectral_library()
    result={'request':{'mode':'dated','time':'2026-11-11T21:00:00Z','observer':asdict(observer),
        'spectrum':{'angstrom_exponent':1.3}},'observation':{'observer':asdict(observer),'atmosphere':asdict(Atmosphere())},
        'stars':{'rows':[{'hr':hr,'name':str(hr),'azimuth_deg':180.,'altitude_deg':50.,
            'catalogue_v_mag':mag,'observed_v_mag':mag,'johnson_v_flux_W_m2':library.zero_v*10**(-.4*mag)}
            for hr,mag in [(1,5),(2,9)]]},'calibration':{'source_coefficient':123.}}
    config=camera_config({'profile':{'sensor':{'width_px':320,'height_px':240}},'sampling':{'max_grid_size':3}})
    result['camera']=build_camera_result(result,config,lambda az,alt:[1e12,2e12,1.5,True])
    return result


def test_exact_V_matched_response_does_not_need_invented_spectrum():
    library=load_spectral_library();profile=camera_config({})['profile']
    weights,factor=response_weights(profile,library)
    for shape in [blackbody_shape(library.wavelengths,3500),blackbody_shape(library.wavelengths,10000),np.ones(95)]:
        observed=band_integral(library.wavelengths,shape,weights)
        expected=band_integral(library.wavelengths,shape,library.responses['johnson_v'])*factor
        assert observed==pytest.approx(expected,rel=1e-13)
    qe=weights*PLANCK*LIGHT_SPEED/(library.wavelengths*1e-9)
    assert qe.max()==pytest.approx(.65) and np.all((qe>=0)&(qe<=1))


@pytest.mark.parametrize('phase',[(0.,0.),(.13,.37),(.5,.5),(.91,.87)])
def test_camera_PSF_preserves_electrons_at_subpixel_phases(phase):
    view=camera_view(camera_config({'profile':{'sensor':{'width_px':320,'height_px':240}}}))
    buffer=np.zeros((240,320));position=project_star(view,180,50)
    p=(160+phase[0],120+phase[1],position[2],0.)
    ledger=deposit_electrons(buffer,p,123456.,100)
    assert abs(buffer.sum()/123456-1)<1e-7
    assert ledger['input_e']==ledger['deposited_e']+ledger['outside_frame_or_truncated_e']
    cropped=np.zeros((240,320));edge=deposit_electrons(cropped,(-.4,120,position[2],0.),123456.,100)
    assert .1<edge['deposited_e']/123456<.9


def test_rotation_and_physical_sensor_geometry():
    config=camera_config({});view=camera_view(config)
    expected=np.rad2deg(2*np.arctan(960*3.76e-3/(2*16)))
    assert view['horizontal_fov_deg']==pytest.approx(expected)
    for rotation in [0,37,90,-120]:
        view['rotation_deg']=rotation
        az,alt=directions(view,.27,-.42);px,py,*_=project_star(view,float(az),float(alt))
        assert px==pytest.approx((.27+1)*960/2-.5,abs=1e-9)
        assert py==pytest.approx((1+.42)*640/2-.5,abs=1e-9)


def test_noise_statistics_and_gain_do_not_create_photons():
    config=camera_config({'exposure_s':10.,'seed':846})
    n=200000;sky=np.full(n,100.);star=np.full(n,20.)
    config['profile']['sensor'].update(read_noise_e=5.,dark_current_e_per_s=.1,gain_e_per_adu=1.,bias_adu=1000.)
    sample=sample_detector(star,sky,config);raw=sample['adu'].astype(float)-1000
    assert abs(raw.mean()-121)<.2
    assert abs(raw.var()-(121+25))/(121+25)<.015
    assert np.all(sample['variance_e2_before_clipping']==146)
    config['profile']['sensor']['gain_e_per_adu']=2
    other=sample_detector(star,sky,config)
    assert np.array_equal(other['expected_e'],sample['expected_e'])
    assert np.array_equal(other['variance_e2_before_clipping'],sample['variance_e2_before_clipping'])
    assert np.array_equal(other['adu'],sample_detector(star,sky,config)['adu'])


def test_camera_background_exposure_aperture_and_natural_cos4_falloff():
    result=camera_fixture();a,_=render_camera_arrays(result)
    changed=deepcopy(result);changed['camera']['config']['exposure_s']*=2
    b,_=render_camera_arrays(changed)
    assert np.allclose(b['sky_e'],2*a['sky_e']) and np.allclose(b['star_e'],2*a['star_e'])
    changed=deepcopy(result);changed['camera']['config']['profile']['optics']['aperture_diameter_mm']*=2
    c,_=render_camera_arrays(changed)
    assert np.allclose(c['sky_e'],4*a['sky_e']) and np.allclose(c['star_e'],4*a['star_e'])
    # Constant radiance: gnomonic pixel area cos³(theta), pupil projection cos(theta).
    gain=a['sky_e']/a['sky_e'][120,160]
    cosine=a['projected_pupil_cosine']/a['projected_pupil_cosine'][120,160]
    assert np.max(abs(gain-cosine**4))<1e-6


def test_fullwell_and_ADC_limits_are_distinct():
    config=camera_config({});sensor=config['profile']['sensor'];sensor.update(full_well_e=200.,read_noise_e=0.,gain_e_per_adu=1.,bias_adu=100.)
    sample=sample_detector(np.full(20,10000.),np.zeros(20),config)
    assert np.all(sample['adu']==300) and np.all(sample['fullwell_mask']) and not sample['adc_clipped_mask'].any()
    sensor.update(full_well_e=100000.,bit_depth=8)
    sample=sample_detector(np.full(20,10000.),np.zeros(20),config)
    assert np.all(sample['adu']==255) and np.all(sample['adc_clipped_mask']) and not sample['fullwell_mask'].any()


def test_Ekos_noise_tolerance_criterion_direct_rate():
    b=np.array([.1,1,100]);rn=3.
    t=read_noise_lower_bound(b,rn,5)
    assert np.allclose(np.sqrt((b*t+rn**2)/(b*t)),1.05)
    assert np.isinf(read_noise_lower_bound(0,3))


def test_camera_has_no_eye_threshold_and_FITS_retains_linear_maps(tmp_path):
    result=camera_fixture();result['eye']={'stars':[{'above_threshold':False}]}
    arrays,stars=render_camera_arrays(result)
    assert len(stars)==2 and all(r['input_e']>0 for r in stars)
    assert arrays['star_e'].sum()==pytest.approx(sum(r['input_e'] for r in stars),rel=1e-7)
    metadata=export_camera(result,tmp_path)
    assert metadata['available'] and not result['camera']['eye_threshold_used']
    with fits.open(tmp_path/'camera.fits',checksum=True) as f:
        assert f[0].data.dtype==np.dtype('uint16') and f[0].header['BUNIT']=='ADU'
        assert f[0].header['EXPTIME']==30 and f[0].header['SIMULATE']
        assert f[0].header['DATE-OBS']!=f[0].header['DATE-AVG']
        assert all(h.verify_checksum()==1 for h in f)
        assert np.max(abs(f['MEAN_E'].data-(f['SKY_E'].data+f['STAR_E'].data+f['DARK_E'].data)))<.01
        assert np.max(abs(f['VAR_E2'].data-f['MEAN_E'].data-9))<.01
        assert np.all(f['MASK'].data==1)
        wcs=WCS(f[0].header);centre=wcs.pixel_to_world_values(159.5,119.5)
        assert centre==pytest.approx([f[0].header['CRVAL1'],f[0].header['CRVAL2']],abs=1e-10)


def test_arbitrary_response_requires_background_spectrum_policy(tmp_path):
    result=camera_fixture();config=deepcopy(result['camera']['config'])
    config['profile']['response']={'kind':'tabulated','wavelength_nm':[360,830],'qe':[.5,.5],'transmission':[1,1]}
    result['camera']=build_camera_result(result,config,lambda az,alt:[1e12,2e12,1.,True])
    assert not result['camera']['available']
    assert not export_camera(result,tmp_path)['available'] and not (tmp_path/'camera.fits').exists()
    config['unresolved_spectrum']='flat_energy_V_normalized'
    result['camera']=build_camera_result(result,config,lambda az,alt:[1e12,2e12,1.,True])
    assert result['camera']['available']
    assert 'not_measured' in result['camera']['response']['unresolved_spectrum_status']


def test_preview_default_removes_bias_pedestal_but_RAW_retains_it(tmp_path):
    from PIL import Image
    result=camera_fixture();camera=result['camera'];camera['stars']=[]
    sensor=camera['config']['profile']['sensor'];sensor.update(read_noise_e=0.,dark_current_e_per_s=0.)
    for key in ['city_response_per_m2_sr_s','unresolved_response_per_m2_sr_s']:
        camera['field'][key]=np.zeros_like(camera['field'][key]).tolist()
    assert camera['config']['preview']['black_adu']==sensor['bias_adu']
    export_camera(result,tmp_path)
    with fits.open(tmp_path/'camera.fits') as raw:assert np.all(raw[0].data==sensor['bias_adu'])
    image=np.asarray(Image.open(tmp_path/'camera-preview.png'))
    assert np.max(image[60:60+240])==0
    assert camera_config({'profile':{'sensor':{'bias_adu':1000.}}})['preview']['black_adu']==1000.


def test_heldout_camera_rays_are_not_refinement_data():
    config=camera_config({});view=camera_view(config)
    sampling={**config['sampling'],'validation':'heldout_8'}
    patch=build_camera_patch(view,lambda az,alt:[1e12,2e12,1.,True],sampling)
    assert patch['validation']['supported']==8 and not patch['validation']['used_for_refinement']
    assert patch['validation']['max_relative_error']<1e-12


@pytest.mark.parametrize('change',[{'exposure_s':True},{'seed':1.5},{'exposure_s':121},
    {'profile':{'sensor':{'gain_e_per_adu':0}}},{'profile':{'sensor':{'width_px':2400,'height_px':1600}}},
    {'profile':{'optics':{'aperture_diameter_mm':100}}},{'unresolved_spectrum':'invent_natural_RGB'}])
def test_camera_invalid_inputs_rejected(change):
    with pytest.raises(ValueError):camera_config(change)


def test_planner_camera_exports_and_source_coefficient_stability(tmp_path):
    scene=offline_example();scene.update(profile='quick',directions=[[180,50]],spectrum={})
    baseline=Planner(tmp_path).run(SceneSpec.from_mapping(scene)).value
    scene['camera']={'profile':{'sensor':{'width_px':320,'height_px':240}},'sampling':{'max_grid_size':3}}
    result=Planner(tmp_path).run(SceneSpec.from_mapping(scene))
    assert result.value['calibration']['source_coefficient']==baseline['calibration']['source_coefficient']
    assert result.value['stars']['rows']==baseline['stars']['rows']
    export_result(result,tmp_path/'export')
    assert (tmp_path/'export/camera.fits').is_file() and (tmp_path/'export/camera-preview.png').is_file()


EKOS_XML='''<ExposureCalculatorCameraData><CameraDataClassVersion>1</CameraDataClassVersion>
<CameraId>Synthetic interoperability test</CameraId><SensorType>MONOCHROME</SensorType>
<CameraGainReadMode><GainReadModeNumber>0</GainReadModeNumber><GainReadModeName>Normal</GainReadModeName>
<CameraGainReadNoise><GainReadNoiseValue><Gain>0</Gain><ReadNoise>5</ReadNoise></GainReadNoiseValue>
<GainReadNoiseValue><Gain>100</Gain><ReadNoise>2</ReadNoise></GainReadNoiseValue></CameraGainReadNoise>
</CameraGainReadMode><CameraPixelSize>4.8</CameraPixelSize><CameraQuantumEfficiency>0.8</CameraQuantumEfficiency>
</ExposureCalculatorCameraData>'''


def test_existing_Ekos_XML_reuses_pitch_and_readnoise_without_inventing_QE_curve():
    from lightpollution.adapters.ekos_camera import import_ekos_camera
    base=camera_config({})['profile'];profile=import_ekos_camera(EKOS_XML,base,0,50)
    assert profile['sensor']['pixel_pitch_um']==4.8 and profile['sensor']['read_noise_e']==3.5
    assert profile['sensor']['gain_e_per_adu']==base['sensor']['gain_e_per_adu']
    assert profile['response']==base['response'] and profile['provenance']['ekos_scalar_QE_hint']==.8
    assert profile['origin']=='ekos_partial' and 'QE_wavelength_curve' in profile['provenance']['not_provided']
    with pytest.raises(ValueError):import_ekos_camera(EKOS_XML,base,0,101)
    with pytest.raises(ValueError):import_ekos_camera('<!DOCTYPE test>'+EKOS_XML)


def test_camera_profile_HTTP_roundtrip_and_XML(tmp_path):
    from fastapi.testclient import TestClient
    from lightpollution.interfaces.web import create_app
    camera=camera_config({'view':{'azimuth_deg':123.,'rotation_deg':37.},'exposure_s':12.})
    with TestClient(create_app(tmp_path)) as client:
        exported=client.post('/api/camera-profile/export',json={'camera':camera});assert exported.status_code==200
        profile=exported.json();assert 'view' not in profile['profile'] and 'exposure_s' not in profile['profile']
        profile['profile']['sensor']['read_noise_e']=2.5
        merged=client.post('/api/camera-profile/merge',json={'camera':camera,'profile':profile})
        assert merged.status_code==200 and merged.json()['view']==camera['view'] and merged.json()['exposure_s']==12.
        xml=client.post('/api/camera-profile/ekos',json={'xml':EKOS_XML,'gain':50})
        assert xml.status_code==200 and xml.json()['profile']['sensor']['read_noise_e']==3.5
        assert client.get('/api/jobs').json()==[]


def test_native_camera_catalogue_is_a_midpoint_snapshot_and_changes_only_camera_stars():
    from test_kstars_integration import payload
    from lightpollution.application.kstars import scene_from_kstars
    data=payload();data['scene']['camera']={'catalogue':{'provider':'KStars','utc':data['snapshot']['utc'],
        'magnitude_limit':12.,'truncated':False,'rows':[{'id':'test-star','name':'test','azimuth_deg':180.,
            'altitude_deg':45.,'v_mag':10.,'b_v':.5}]}}
    spec,_=scene_from_kstars(data)
    assert spec.to_dict()['camera']['view']['altitude_deg']==45.
    data['scene']['camera']['catalogue']['utc']='2026-11-10T21:00:00Z'
    with pytest.raises(DomainError):scene_from_kstars(data)
