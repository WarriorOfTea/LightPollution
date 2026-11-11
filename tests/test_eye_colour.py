from copy import deepcopy
from dataclasses import asdict
import csv
import struct
import zlib

import numpy as np
import pytest

from lightpollution.adapters.colour_library import ColourLibrary,load_colour_library
from lightpollution.adapters.spectral_library import load_spectral_library
from lightpollution.application.eye import build_eye_result
from lightpollution.application.eye_appearance import ensure_eye_appearance,export_eye_colour,stellar_colour_table
from lightpollution.interfaces.eye_view import encode_field,render_linear_field,render_eye_view
from lightpollution.models.eye import eye_config
from lightpollution.models.eye_colour import (RGB_LUMINANCE,RGB_TO_XYZ,XYZ_TO_RGB,cone_colour_weight,
    encircled_pool_gain,gamut_preserving_luminance,kirk2011_normalized_lmsr,xyz_unit_rgb)
from lightpollution.models.eye_optics import optical_mixture
from lightpollution.models.eye_profiles import apply_profile,display_geometry,profile_document
from lightpollution.models.spectra import stellar_spectral_observation
from lightpollution.physics import Atmosphere,Observer
from test_eye import physical_fixture


def colour_fixture():
    result=physical_fixture();library=load_spectral_library()
    observer=Observer(60.3,30.315,20.);atmosphere=Atmosphere()
    rows=[{'hr':hr,'name':str(hr),'azimuth_deg':180.+offset,'altitude_deg':50.,
        'catalogue_v_mag':mag,'observed_v_mag':mag}
        for hr,mag,offset in [(7001,1.,0.),(100001,2.,3.),(100002,5.,-3.)]]
    stars,_,_=stellar_spectral_observation(rows,observer,atmosphere,library.wavelengths,library.responses,
        {100001:{'b_v':'1.5'},100002:{'b_v':'0.0'}},library.measured,library.zero_b,library.zero_v,1.3)
    result['stars']['rows']=stars
    result['observation'].update(observer=asdict(observer),atmosphere=asdict(atmosphere))
    result['request']['spectrum']={'angstrom_exponent':1.3}
    config=eye_config({**result['eye']['config'],'appearance':{'mode':'experimental_stellar_colour'}})
    result['eye']=build_eye_result(result,config,result['eye']['field'],1e-6)
    return result


def test_CIE_checksums_domains_D65_and_Y_agreement():
    library=load_colour_library();spectral=load_spectral_library()
    xyz,lms=library.responses(spectral.wavelengths)
    assert xyz.shape==lms.shape==(95,3) and np.all(lms>=0)
    assert np.all(lms[spectral.wavelengths>615,2]==0)
    assert np.all(lms[spectral.wavelengths<390]==0)
    assert np.max(abs(xyz[:,1]-spectral.responses['photopic']))<1e-6
    d65=np.interp(spectral.wavelengths,library.d65[:,0],library.d65[:,1])
    white=np.trapezoid(d65[:,None]*xyz,x=spectral.wavelengths,axis=0)
    rgb=xyz_unit_rgb(white)
    assert np.max(abs(rgb-1))<.001


def test_unknown_CIE_data_inside_valid_domain_is_not_zero_filled(tmp_path):
    import shutil,json,hashlib
    from lightpollution.adapters.colour_library import DIRECTORY
    for p in DIRECTORY.iterdir():
        if p.is_file():shutil.copy(p,tmp_path/p.name)
    values=np.loadtxt(tmp_path/'CIE_lms_cf_2deg.csv',delimiter=',');values[5,1]=np.nan
    np.savetxt(tmp_path/'CIE_lms_cf_2deg.csv',values,delimiter=',')
    p=tmp_path/'provenance.json';meta=json.loads(p.read_text())
    meta['sha256']['CIE_lms_cf_2deg.csv']=hashlib.sha256((tmp_path/'CIE_lms_cf_2deg.csv').read_bytes()).hexdigest()
    p.write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='published wavelength domain'):ColourLibrary(tmp_path)


def test_gamut_keeps_luminance_and_neutral_without_individual_channel_clipping():
    rng=np.random.default_rng(24)
    rgb=rng.normal(.4,1.5,(10000,3))
    delta=rgb-(rgb@RGB_LUMINANCE)[:,None]
    y=rng.uniform(0,1,10000)
    mapped,scale=gamut_preserving_luminance(y[:,None]+delta)
    assert np.all((mapped>=0)&(mapped<=1)) and np.all((scale>=0)&(scale<=1))
    assert np.max(abs(mapped@RGB_LUMINANCE-y))<1e-14
    neutral,_=gamut_preserving_luminance(np.repeat(np.linspace(0,1,20)[:,None],3,axis=1))
    assert np.max(np.ptp(neutral,axis=1))<1e-14
    assert np.max(abs(XYZ_TO_RGB@RGB_TO_XYZ-np.eye(3)))<1e-14


def test_colour_is_suppressed_when_the_same_star_flux_is_lowered():
    gain=encircled_pool_gain(optical_mixture(6),5.)
    flux=np.geomspace(1e-12,1e-3,1000)
    w=cone_colour_weight(2e-4,2.8e-4,flux,2*flux,gain)
    assert np.all(np.diff(w)>=-1e-12) and w[0]<1e-12 and w[-1]>.99
    assert encircled_pool_gain(optical_mixture(6),10)<gain


def test_Kirk_prototype_has_no_implicit_bridge_from_CIE_integrals():
    q=np.array([[1.,2.,3.,0.],[1.,2.,3.,4.]])
    output=kirk2011_normalized_lmsr(q,input_convention='paper_normalized')
    assert np.array_equal(output[0],q[0,:3])
    assert np.all(np.isfinite(output)) and not np.allclose(output[1],q[1,:3])
    with pytest.raises(ValueError):kirk2011_normalized_lmsr(q,input_convention='CIE2006_W_m2')


def test_rebuilt_spectra_preserve_frozen_flux_and_do_not_change_thresholds(tmp_path):
    result=colour_fixture();before=deepcopy(result)
    colours,metadata=stellar_colour_table(result)
    assert sum(r['available'] for r in colours)==3
    assert max(max(r['relative_P_S_error']) for r in colours)<1e-10
    appearance=ensure_eye_appearance(result)
    assert result['stars']==before['stars'] and result['eye']['stars']==before['eye']['stars']
    assert appearance['spectral_metadata']['stellar_atmospheric_transmission_count']==1
    assert not appearance['spectral_metadata']['total_background_LMSR_available']
    assert appearance['experimental'] and not appearance['psychophysical_colour_threshold_validated']
    export_eye_colour(result['eye'],tmp_path)
    rows=list(csv.DictReader((tmp_path/'eye-colour.csv').open()))
    assert float(rows[0]['cone_colour_weight'])==pytest.approx(appearance['rows'][0]['cone_colour_weight'])
    assert float(rows[0]['appearance_B_unitY'])==pytest.approx(appearance['rows'][0]['appearance_unit_Y_rgb'][2])


def test_pool_colour_signal_independent_of_output_resolution():
    result=colour_fixture();a=ensure_eye_appearance(result)
    result['eye']['config']['view'].update(width=1280,height=960)
    b=ensure_eye_appearance(result)
    assert a['rows']==b['rows']


def test_colour_cache_invalidates_on_local_background_change():
    result=colour_fixture();first=deepcopy(ensure_eye_appearance(result))
    result['eye']['stars'][0]['photopic_background_cd_m2']*=20
    result['eye']['stars'][0]['scotopic_background_cd_m2']*=20
    second=ensure_eye_appearance(result)
    assert first['cache_key']!=second['cache_key']
    assert first['rows'][0]['cone_colour_weight']!=second['rows'][0]['cone_colour_weight']
    assert first['rows'][1:]==second['rows'][1:]


def test_portable_profile_HTTP_roundtrip_and_reject_view_override(tmp_path):
    from fastapi.testclient import TestClient
    from lightpollution.interfaces.commands import offline_example
    from lightpollution.interfaces.web import create_app
    scene=offline_example();scene['eye']=eye_config({'appearance':{'mode':'experimental_stellar_colour'},
        'view':{'width':640,'height':480},'display':{'window_width_mm':600,'viewing_distance_mm':600}})
    with TestClient(create_app(tmp_path)) as client:
        exported=client.post('/api/eye-profile/export',json={'eye':scene['eye']})
        assert exported.status_code==200
        profile=exported.json()
        assert 'view' not in profile['eye']
        profile['eye']['observer_factor']=3.
        imported=client.post('/api/eye-profile/apply',json={'scene':scene,'profile':profile})
        assert imported.status_code==200
        result=imported.json()
        assert result['eye']['observer_factor']==3. and result['eye']['view']==scene['eye']['view']
        assert result['observer']==scene['observer'] and result['time']==scene['time']
        assert client.post('/api/eye-profile/merge',json={'eye':scene['eye'],'profile':profile}).json()==result['eye']
        profile['eye']['view']={'azimuth_deg':0}
        bad=client.post('/api/eye-profile/apply',json={'scene':scene,'profile':profile})
        assert bad.status_code==422 and bad.json()['error']['code']=='invalid_eye_profile'
        assert client.get('/api/jobs').json()==[]


def test_RGB_exports_have_true_16bit_channels_and_shared_dither(tmp_path):
    result=colour_fixture();before=deepcopy(result['eye']['summary'])
    display=render_eye_view(result,tmp_path)
    assert display['colour_mode']=='experimental_stellar_colour'
    assert result['eye']['summary']==before
    payload=(tmp_path/'eye-view-16bit.png').read_bytes()
    width,height,depth,kind,*_=struct.unpack('>IIBBBBB',payload[16:29])
    assert depth==16 and kind==2
    pos=8;data=b''
    while pos<len(payload):
        size=struct.unpack('>I',payload[pos:pos+4])[0];name=payload[pos+4:pos+8]
        if name==b'IDAT':data+=payload[pos+8:pos+8+size]
        pos+=12+size
    decoded=zlib.decompress(data)
    rows=np.frombuffer(decoded,dtype=np.uint8).reshape(height,1+width*6)
    assert np.all(rows[:,0]==0)
    pixels=np.frombuffer(rows[:,1:].copy().tobytes(),dtype='>u2').reshape(height,width,3)
    with np.load(tmp_path/'eye-linear.npz',allow_pickle=False) as values:
        encoded,metrics=encode_field(values,result['eye']['config']['display'])
    assert np.max(abs(pixels.astype(float)-65535*encoded))<=.5
    assert metrics['maximum_display_Y_error']<1e-12
    assert (tmp_path/'eye-colour.csv').is_file()


def test_display_geometry_and_profile_roundtrip_keep_scene_view():
    config=eye_config({'view':{'width':1280,'height':960},'display':{
        'window_width_mm':600.,'viewing_distance_mm':600.,'origin':'user_supplied'},
        'observer_profile':{'profile_id':'observer-A','origin':'user_supplied'},
        'appearance':{'mode':'experimental_stellar_colour'}})
    angle=display_geometry(config['display'],60.)
    assert angle['displayed_horizontal_angle_deg']==pytest.approx(53.130102354)
    assert not angle['individual_perceptual_match_validated']
    saved=profile_document(config);target=eye_config({'view':{'azimuth_deg':20.}})
    restored=apply_profile(saved,target)
    assert restored['view']==target['view'] and restored['display']==config['display']
    assert restored['observer_profile']==config['observer_profile']


@pytest.mark.parametrize('data',[{'display':{'window_width_mm':500}},
    {'display':{'viewing_distance_mm':True,'window_width_mm':500}},
    {'display':{'origin':'instrument_measured'}},{'display':{'measured_at':'2026-11-11'}},
    {'observer_profile':{'colour_integration_diameter_arcmin':0}},
    {'observer_profile':{'origin':'validated'}},{'appearance':{'mode':'photorealistic'}}])
def test_invalid_profiles_and_claimed_validation_rejected(data):
    with pytest.raises(ValueError):eye_config(data)
