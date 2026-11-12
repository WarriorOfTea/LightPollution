import csv
from copy import deepcopy

import pytest

from lightpollution.adapters.catalogue_identity import PATH,cross_id
from lightpollution.application.camera import build_camera_result
from lightpollution.models.camera import validate_catalogue
from test_camera import camera_fixture


def vega():
    with PATH.open() as stream:
        reference=next(r for r in csv.DictReader(stream) if r['hr']=='7001')
    return {'id':'Vega','name':'Vega','azimuth_deg':180.,'altitude_deg':50.,'v_mag':.03,'b_v':None,
        'astrometry':{'hd':172167,'ra_j2000_deg':float(reference['ra_deg']),
            'dec_j2000_deg':float(reference['dec_deg']),'pm_ra_cosdec_mas_yr':200.,'pm_dec_mas_yr':286.,'multiple':False}}


def test_unique_HD_and_astrometry_are_both_required_for_spectral_cross_ID():
    star=vega();identity=cross_id(star)
    assert identity['hr']==7001 and identity['separation_j2000_arcsec']==0
    changed=deepcopy(star);changed['astrometry']['ra_j2000_deg']+=.1
    assert cross_id(changed) is None
    changed=deepcopy(star);changed['astrometry']['hd']=0
    assert cross_id(changed) is None
    changed=deepcopy(star);changed['astrometry']['multiple']=True
    assert cross_id(changed) is None
    assert cross_id({'id':'a nearby object'}) is None


def test_cross_ID_reuses_measured_SED_without_merging_nearby_stars():
    result=camera_fixture();config=result['camera']['config']
    star=vega();other=deepcopy(star);other['id']='companion';other['astrometry']['hd']=0
    other['astrometry']['ra_j2000_deg']+=1/3600;other['v_mag']=9;other['b_v']=.7
    config['catalogue']={'provider':'KStars','utc':result['request']['time'],'magnitude_limit':12.,
        'truncated':False,'rows':[star,other]}
    value=build_camera_result(result,config,lambda a,h:[1e12,2e12,1.5,True])
    assert len(value['stars'])==2
    assert value['stars'][0]['spectrum_recipe']=={'kind':'measured_reference_shape','hr':7001}
    assert value['stars'][1]['spectrum_recipe']['kind']=='B_V_fitted_Planck_continuum'
    assert value['catalogue']['BSC_cross_identified_rows']==1
    assert value['source_coefficient']==result['calibration']['source_coefficient']


def test_astrometric_metadata_are_validated():
    catalogue={'provider':'KStars','utc':'2026-11-11T21:00:00Z','magnitude_limit':12.,'truncated':False,'rows':[vega()]}
    validate_catalogue(catalogue)
    catalogue['rows'][0]['astrometry']['multiple']=1
    with pytest.raises(ValueError):validate_catalogue(catalogue)


def test_Tycho_magnitudes_are_explicitly_converted_and_not_used_as_Johnson_unchanged():
    from lightpollution.models.catalogue_photometry import johnson_photometry
    star={'v_mag':11.,'b_v':1.,'photometry':'Tycho_VT_BTminusVT_inferred'}
    v,bv,status=johnson_photometry(star)
    assert v==pytest.approx(10.91) and bv==pytest.approx(.85)
    assert 'scenario' in status
    result=camera_fixture();config=result['camera']['config']
    native={'id':'Tycho','name':'Tycho','azimuth_deg':180.,'altitude_deg':50.,**star}
    red={**native,'id':'red','b_v':1.98}
    missing={**native,'id':'missing','b_v':None}
    config['catalogue']={'provider':'KStars','utc':result['request']['time'],'magnitude_limit':12.,
        'truncated':False,'rows':[native,red,missing]}
    result['observation']['atmosphere'].update(rayleigh_multiplier=0.,aerosol_optical_depth=0.)
    value=build_camera_result(result,config,lambda a,h:[1e12,2e12,1.5,True])
    rows={r['id']:r for r in value['stars']}
    assert rows['Tycho']['observed_v_mag']==pytest.approx(10.91)
    assert rows['Tycho']['spectrum_recipe']['b_v']==pytest.approx(.85)
    assert not rows['red']['available'] and not rows['missing']['available']
    assert value['catalogue']['photometry_unavailable_rows']==2
    assert value['catalogue']['Tycho_to_Johnson_converted_rows']==1
