from dataclasses import asdict
import json

import numpy as np
import pytest

from lightpollution.atmosphere_config import atmosphere_from_mapping, atmosphere_origin
from lightpollution.calibration import astmon_candidates, fit_and_check, nightly_values
from lightpollution.cli import make_parser, run_simulation
from lightpollution.data import Bounds, COMPOSITES, load_region, save_region
from lightpollution.photometry import KS_V_BAND
from lightpollution.physics import Atmosphere, Observer
from lightpollution.scene import calculate_observation
from lightpollution.seasonal import prepare_seasonal


def regions(tmp_path, factor=2.):
    bounds=Bounds(0,0,2/240,2/240)
    paths=[]
    for month,multiplier in [(1,1.),(7,factor)]:
        arrays={'Land_Water_Mask':np.ones((2,2))}
        for composite in COMPOSITES:
            key=composite+'_Composite_Snow_Free'
            arrays.update({key:multiplier*np.array([[1.,2.],[3.,4.]]),
                           key+'_Quality':np.zeros((2,2)),key+'_Num':np.full((2,2),10.),
                           key+'_Std':np.ones((2,2))})
        path=tmp_path/f'{month}.npz'
        save_region(path,arrays,bounds,{'year':2025,'month':month,'product':'VNP46A3.002'})
        paths.append(path)
    return paths


def test_atmosphere_defaults_and_user_values_have_one_source_of_truth(tmp_path):
    assert atmosphere_from_mapping()==Atmosphere()
    assert not atmosphere_origin()['regional_probability_estimated']
    assert atmosphere_from_mapping({'aerosol_optical_depth':.25}).aerosol_optical_depth==.25
    for invalid in [{'humidity':.8},{'aerosol_g':2},{'aerosol_optical_depth':True}]:
        with pytest.raises(ValueError): atmosphere_from_mapping(invalid)
    path=tmp_path/'atmosphere.json'
    path.write_text(json.dumps({'aerosol_optical_depth':.25,'aerosol_g':.6}))
    args=make_parser().parse_args(['demo','--atmosphere-file',str(path),'--aod','.3','--no-plots',
        '--output',str(tmp_path/'demo'),'--azimuth-step','90','--altitudes','45','90',
        '--ray-segments','8','--path-quadrature','2'])
    report=run_simulation(args)
    assert report['atmosphere']['aerosol_optical_depth']==.3 and report['atmosphere']['aerosol_g']==.6
    assert not (tmp_path/'demo/sky.png').exists()


def test_matched_months_preserve_amplitude_and_refuse_double_preparation(tmp_path):
    paths=regions(tmp_path)
    observer=Observer(1,1,20)
    report=prepare_seasonal(paths,tmp_path/'matched','2025-01',observer,0)
    assert report['common_pixels']==report['reference_land_domain_pixels']==4
    assert report['periods'][1]['source_power_multiplier']==pytest.approx(2.)
    a,_,meta=load_region(tmp_path/'matched/2025-07/source-grid.npz')
    assert meta['source_power_multiplier']==pytest.approx(2)
    with pytest.raises(ValueError,match='twice'):
        prepare_seasonal([tmp_path/'matched/2025-01/source-grid.npz',tmp_path/'matched/2025-07/source-grid.npz'],
                         tmp_path/'double','2025-01',observer,0)


def test_missing_month_pixels_are_not_dark_sources_and_coverage_gate_precedes_output(tmp_path):
    paths=regions(tmp_path)
    a,b,m=load_region(paths[1])
    a['NearNadir_Composite_Snow_Free_Quality'][0,0]=1
    save_region(paths[1],a,b,m)
    with pytest.raises(ValueError,match='Insufficient'):
        prepare_seasonal(paths,tmp_path/'bad','2025-01',exclude_radius_m=0)
    assert not (tmp_path/'bad').exists()
    report=prepare_seasonal(paths,tmp_path/'ok','2025-01',exclude_radius_m=0,minimum_common_fraction=.5)
    assert report['common_pixels']==3 and report['reference_land_domain_pixels']==4
    assert report['periods'][1]['source_power_multiplier']==pytest.approx(2)


def test_simulation_scales_both_source_paths_and_blocks_observer_mismatch(tmp_path):
    observer=Observer(1,1,20)
    report=prepare_seasonal(regions(tmp_path),tmp_path/'matched','2025-01',observer,0)
    outputs=[]
    for month in [1,7]:
        args=make_parser().parse_args(['simulate','--region',str(tmp_path/f'matched/2025-{month:02}/source-grid.npz'),
            '--output',str(tmp_path/f'model-{month}'),'--latitude','1','--longitude','1',
            '--exclude-radius-m','0','--no-plots','--azimuth-step','90','--altitudes','1','45','90',
            '--ray-segments','8','--path-quadrature','2'])
        run_simulation(args)
        with np.load(args.output/'sky-grid.npz') as data:
            outputs.append(data['radiance'])
    np.testing.assert_allclose(outputs[1],2*outputs[0],rtol=1e-12)
    args.latitude=2
    with pytest.raises(ValueError,match='selection'): run_simulation(args)


def test_shared_v_anchor_keeps_the_monthly_amplitude(tmp_path):
    observer=Observer()
    reference=tmp_path/'reference.npz'
    target=tmp_path/'target.npz'
    meta={'observer':asdict(observer),'atmosphere':asdict(Atmosphere()),
          'provenance':{'region_metadata':{'source_power_multiplier':2.,'common_mask_sha256':'same',
                                         'normalisation_reference_period':'2025-01'}}}
    for path,power in [(reference,1.),(target,2.)]:
        np.savez(path,azimuth_deg=[0,180],altitude_deg=[0,90],radiance=np.full((2,2),power),
                 metadata_json=json.dumps(meta))
    scene={'schema_version':1,'mode':'undated','observer':asdict(observer),'directions':[[0,90]],
           'moon':{'mode':'none'},'natural':{'mode':'none','source':'Test'},
           'artificial':{'mode':'grid','sky_grid':str(target),'anchor_v_mag_arcsec2':22.,
                          'scale_origin':'scenario','source':'Test','band_bridge':'effective_550nm_shape_anchored_in_V'}}
    with pytest.raises(ValueError,match='erase'): calculate_observation(scene)
    meta['provenance']['region_metadata'].update(source_power_multiplier=1.,year=2025,month=7)
    np.savez(target,azimuth_deg=[0,180],altitude_deg=[0,90],radiance=np.full((2,2),2.),
             metadata_json=json.dumps(meta))
    with pytest.raises(ValueError,match='erase'): calculate_observation(scene)
    scene['artificial']['reference_sky_grid']=str(reference)
    result=calculate_observation(scene)['directions'][0]
    assert result['artificial_relative']==pytest.approx(2.)


def test_ground_fit_requires_explicit_cloud_qa_and_uses_independent_nights():
    nights=[{'period':p,'night_id':night,'brightness_relative':b,'mag_arcsec2':22-2.5*np.log10(b),
             'samples':10,'cloud_free_confirmed':True}
            for p,night,b in [('2025-01','night1',5),('2025-01','night2',6),('2025-07','night3',10)]]
    fit=fit_and_check(nights,{'2025-01':1.,'2025-07':2.},['2025-01'],natural_mag=22.)
    assert fit['coefficient_city_relative']==pytest.approx(4.5)
    assert fit['held_out']['rmse_mag']==pytest.approx(0,abs=1e-12)
    assert fit['training']['nights']==2 and fit['held_out']['nights']==1
    nights[0]['cloud_free_confirmed']=None
    with pytest.raises(ValueError,match='QA'): fit_and_check(nights,{'2025-01':1,'2025-07':2},['2025-01'])
    assert fit_and_check(nights,{'2025-01':1,'2025-07':2},['2025-01'],allow_unverified_clouds=True)['status'].startswith('provisional')
    nights[0]['cloud_free_confirmed']=False
    with pytest.raises(ValueError,match='Known cloudy'):
        fit_and_check(nights,{'2025-01':1,'2025-07':2},['2025-01'],allow_unverified_clouds=True)


def test_nightly_mean_is_linear_and_cloudy_records_do_not_get_certified():
    rows=[{'period':'2025-01','night_id':'n1','mag_arcsec2':m,'cloud_free_confirmed':None} for m in [20.,22.]]
    result=nightly_values(rows)[0]
    assert result['brightness_relative']==pytest.approx((10**.8+1)/2)
    assert result['cloud_free_confirmed'] is None
    rows[0]['cloud_free_confirmed']=False
    assert nightly_values(rows)==[]


def test_astro_selection_is_not_cloud_classification(tmp_path):
    path=tmp_path/'astmon.csv'
    path.write_text('Datetime_UTC;Filter;Magnitude;Error\n2025-03-29 02:00:00;Johnson-V;18;.02\n')
    rows,report=astmon_candidates(path,Observer(40.45,-3.73,650),2025,[3])
    assert len(rows)==1 and report['certified_cloud_free_candidates']==0
    assert rows[0]['cloud_free_confirmed'] is None
