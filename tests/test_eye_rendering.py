from copy import deepcopy

import numpy as np
import pytest
from PIL import Image

from lightpollution.application.eye import sample_eye_field
from lightpollution.interfaces.eye_view import render_eye_view
from lightpollution.models.eye import eye_config
from lightpollution.models.eye_display import (apply_ward1997, quantize, srgb_encode, tone_linear,
    tvi_threshold, ward1997_curve)
from lightpollution.models.eye_field import (build_view_patch, heldout_coordinates, interpolate_patch,
    patch_support, tangent_coordinates, tangent_directions)
from lightpollution.models.eye_optics import (deposit_star, optical_mixture, pixel_solid_angles, watson_mtf)
from test_eye import physical_fixture

VIEW = {'azimuth_deg':180., 'altitude_deg':75., 'horizontal_fov_deg':60., 'width':1800, 'height':1100}


def test_patch_geometry_is_unique_at_zenith_and_wraps_azimuth():
    x, y, front = tangent_coordinates(VIEW, [0., 90., 180., 270., 360.], [90.]*5)
    assert np.all(front)
    assert np.ptp(x) < 1e-14 and np.ptp(y) < 1e-14
    a, h = tangent_directions(VIEW, x, y)
    assert np.allclose(h, 90)
    a = np.array([359.999, .001, 120., 230.])
    h = np.array([89., 89., 60., 65.])
    x, y, front = tangent_coordinates(VIEW, a, h)
    aa, hh = tangent_directions(VIEW, x, y)
    assert np.max(abs((aa-a+180)%360-180)) < 1e-10
    assert np.max(abs(hh-h)) < 1e-10


def test_positive_pchip_reproduces_nodes_and_constant_and_has_no_overshoot():
    axis = np.array([-1., -.5, 0., .5, 1.])
    patch = {'x':axis, 'y':axis, 'physical_supported':np.ones((5,5),bool)}
    values = (1+axis[:,None])**2+(1+axis[None,:])**2
    x, y = np.meshgrid(axis, axis)
    assert np.allclose(interpolate_patch(patch,values,x,y), values)
    query = np.linspace(-1,1,257)
    xx, yy = np.meshgrid(query, query)
    result = interpolate_patch(patch,values,xx,yy)
    assert np.min(result)>=0 and np.max(result)<=np.max(values)
    assert np.min(np.diff(result,axis=0))>=-1e-12
    assert np.min(np.diff(result,axis=1))>=-1e-12
    assert np.allclose(interpolate_patch(patch, np.full((5,5),3.7), xx, yy),3.7)
    eps = 1e-6
    q = interpolate_patch(patch,values,np.array([.5-eps,.5,.5+eps]),np.full(3,.27))
    assert abs((q[2]-q[1])/eps-(q[1]-q[0])/eps)<1e-4


def test_holdout_never_refines_the_grid_and_support_uses_derivative_stencil():
    def direct(az,alt):
        return [1+np.cos(np.deg2rad(alt))**2, 2.,3.,True]
    a = build_view_patch(VIEW, direct, validation='none')
    b = build_view_patch(VIEW, direct, validation='heldout_24', evaluate_standard=direct)
    assert a['x']==b['x'] and a['y']==b['y'] and a['refinement']==b['refinement']
    assert a['city_photopic_luminance_cd_m2']==b['city_photopic_luminance_cd_m2']
    assert len(b['validation']['rows'])==24 and not b['validation']['used_for_refinement']
    assert sum('standard_P_S_V' in r for r in b['validation']['rows'])==4
    assert np.array_equal(heldout_coordinates(VIEW),heldout_coordinates(VIEW))
    b['physical_supported'][0][0]=False
    assert not patch_support(b,-.99,-.99)


@pytest.mark.parametrize('diameter',[2.,4.,6.])
def test_mean_optics_fit_and_dc_normalization(diameter):
    m=optical_mixture(diameter)
    assert m['normalization_integral']==pytest.approx(1,abs=1e-14)
    assert min(m['weight'])>=0 and min(m['sigma_deg'])>0
    # A positive Gaussian mixture cannot reproduce a hard diffraction cutoff
    # exactly. The 2 mm edge has the largest residual; the default 6 mm is tighter.
    assert m['fit_mtf_max_absolute_error']<.025
    assert m['fit_mtf_rms_absolute_error']<.004
    if diameter==6.:assert m['fit_mtf_max_absolute_error']<.003
    assert watson_mtf(0,diameter)==1 and watson_mtf(10000,diameter)==0
    assert np.all(np.diff(watson_mtf(np.linspace(0,200,1001),diameter))<=0)


@pytest.mark.parametrize('width',[390,900,1800])
@pytest.mark.parametrize('phase',[(0.,0.),(.13,.37),(.5,.5),(.91,.87)])
def test_flux_conservation_across_pixel_scales_subpixel_phase_and_projection(width,phase):
    mixture=optical_mixture(6.)
    for correlation in [0.,.35]:
        buffer=np.zeros((width,width),dtype=np.float64)
        e=1.2345e-8
        ledger=deposit_star(buffer,width/2+phase[0],width/2+phase[1],e,np.deg2rad(60)/width,
                            mixture,correlation=correlation)
        assert buffer.sum()==pytest.approx(e,rel=1e-6)
        assert ledger['deposited_scotopic_illuminance_lux']==pytest.approx(buffer.sum(),rel=1e-12)


def test_crop_is_not_renormalized_and_one_magnitude_has_the_right_flux_ratio():
    a=np.zeros((390,390));b=np.zeros_like(a)
    mixture=optical_mixture(6.)
    deposit_star(a,-.5,195.,1.,np.deg2rad(60)/390,mixture,correlation=.2)
    deposit_star(b,-.5,195.,10**.4,np.deg2rad(60)/390,mixture,correlation=.2)
    assert a.sum()==pytest.approx(.5,rel=1e-6)
    assert np.allclose(b,a*10**.4,rtol=1e-12,atol=1e-15)


def test_solid_angle_is_positive_and_integrates_to_exact_rectangular_cone():
    view={**VIEW,'width':390,'height':238}
    omega=pixel_solid_angles(view)
    ex=np.tan(np.deg2rad(view['horizontal_fov_deg'])/2)
    ey=ex*view['height']/view['width']
    expected=4*np.arctan(ex*ey/np.sqrt(1+ex*ex+ey*ey))
    assert omega.min()>0 and omega.sum()==pytest.approx(expected,rel=1e-13)


def test_explicit_display_keeps_order_and_encodes_srgb_once():
    profile=eye_config({})['display']
    assert tvi_threshold(1e-6,scotopic=True)==pytest.approx(10**-2.86)
    assert tvi_threshold(100.,scotopic=False)==pytest.approx(10**(2-1.255))
    values=np.geomspace(1e-7,1e3,1000)
    image=tone_linear(values,profile)
    assert np.all(np.diff(image)>=0) and image.min()>0 and image.max()<=1
    assert srgb_encode(0.)==0 and srgb_encode(1.)==pytest.approx(1)
    assert srgb_encode(.0031308)==pytest.approx(.040449936)
    assert srgb_encode(.18)==pytest.approx(.46135613)


def test_ward_comparator_is_monotone_and_reports_the_narrow_scene_fallback():
    profile=eye_config({})['display']
    curve=ward1997_curve(np.geomspace(.001,.003,1000),profile)
    assert curve['mode']=='linear_range_fits'
    assert not curve['complete_pcond_implementation']
    curve=ward1997_curve(np.geomspace(1e-4,1e4,4000),profile)
    assert curve['mode']=='histogram_tvi' and curve['converged']
    display=apply_ward1997(np.geomspace(1e-5,1e5,2000),curve,profile)
    assert np.all(np.diff(display)>=0)
    assert display.min()==pytest.approx(0,abs=1e-14) and display.max()==pytest.approx(1)


def test_dither_is_bounded_reproducible_and_unbiased_and_16bit_rounding_bounded():
    values=np.tile(np.linspace(.073,.088,3000),(800,1))
    a=quantize(values);b=quantize(values)
    error=a.astype(float)-255*values
    assert np.array_equal(a,b) and np.max(abs(error))<1
    assert abs(error.mean())<.002
    assert np.max(abs(error.reshape(80,-1).mean(axis=1)))<.01
    assert np.max(abs(quantize(values,16).astype(float)-65535*values))<=.5


def test_new_exports_preserve_detection_and_store_float_photometry(tmp_path):
    result=physical_fixture();summary=deepcopy(result['eye']['summary'])
    config=deepcopy(result['eye']['config'])
    display=render_eye_view(result,tmp_path)
    assert result['eye']['summary']==summary and result['eye']['config']==config
    assert display['displayed_stars']==1 and not display['calibrated_eye_appearance']
    assert all(row['reintegrated_before_tone'] for row in display['previews'])
    with np.load(tmp_path/'eye-linear.npz',allow_pickle=False) as data:
        assert data['background_scotopic_cd_m2'].dtype==np.float32
        assert data['stellar_physical_pixel_illuminance_lux'].sum()>data['stellar_appearance_pixel_illuminance_lux'].sum()
    assert np.asarray(Image.open(tmp_path/'eye-view-16bit.png')).max()>255
    assert Image.open(tmp_path/'eye-mobile.png').width==390
    result['eye']['available']=False
    render_eye_view(result,tmp_path)
    assert not (tmp_path/'eye-linear.npz').exists()
    assert not (tmp_path/'eye-mobile.png').exists()


@pytest.mark.parametrize('settings',[{'optics':{'pupil_diameter_mm':7}}, {'sampling':{'max_grid_size':4}},
    {'sampling':{'max_grid_size':True}}, {'sampling':{'validation':'fit_to_holdout'}},
    {'display':{'world_adaptation_scotopic_cd_m2':0}}, {'display':{'black_cd_m2':20}},
    {'display':{'white_cd_m2':20,'adaptation_cd_m2':20}}])
def test_invalid_new_contract_is_rejected(settings):
    with pytest.raises(ValueError):eye_config(settings)
