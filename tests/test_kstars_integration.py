from copy import deepcopy

from fastapi.testclient import TestClient
import pytest

from lightpollution.application.kstars import scene_from_kstars
from lightpollution.application.planner import Planner
from lightpollution.domain.scene import DomainError
from lightpollution.interfaces.commands import offline_example
from lightpollution.interfaces.web import create_app


def payload():
    return {'scene': offline_example(), 'snapshot': {'schema_version': 1,
        'kstars_version': '3.6.2', 'utc': '2026-11-09T21:00:00Z',
        'altitude_convention':'geometric_unrefracted',
        'location': {'latitude_deg':60.3, 'longitude_deg':30.315, 'elevation_m':120},
        'focus':{'azimuth_deg':180, 'altitude_deg':45, 'object_name':'Vega',
                 'catalogue_magnitude':.03}, 'artificial_horizon_visible':True}}


def test_atomic_frame_overrides_manual_geometry_but_preserves_agl_and_physics():
    data=payload();data['scene']['observer']['height_m']=1.7;data['scene']['surface']={'snow_fraction':.7}
    original=deepcopy(data)
    spec, frame=scene_from_kstars(data);scene=spec.to_dict()
    assert scene['observer']['height_m']==1.7
    assert scene['observer']['height_m']!=frame['location']['elevation_m']
    assert scene['directions']==[[180.,45.]]
    assert scene['surface']['snow_fraction']==.7
    assert scene['integration']['provider']=='KStars'
    assert scene['integration']['site_elevation_m']==120
    assert data==original


@pytest.mark.parametrize('change',['timezone','non_utc_offset','refraction','boolean_angle','unknown_frame','invalid_horizon','nan_magnitude','nan_star_colour','invalid_observer'])
def test_invalid_live_context_rejected(change):
    data=payload();s=data['snapshot']
    if change=='timezone':s['utc']='2026-11-09T21:00:00'
    if change=='non_utc_offset':s['utc']='2026-11-09T21:00:00+03:00'
    if change=='refraction':s['altitude_convention']='apparent'
    if change=='boolean_angle':s['focus']['altitude_deg']=True
    if change=='unknown_frame':s['unexpected']=1
    if change=='invalid_horizon':s['artificial_horizon_visible']='false'
    if change=='nan_magnitude':s['focus']['catalogue_magnitude']=float('nan')
    if change=='nan_star_colour':s['focus']['star_b_v']=float('nan')
    if change=='invalid_observer':data['scene']['observer']=None
    with pytest.raises(DomainError):scene_from_kstars(data)


def test_kstars_horizon_does_not_turn_obstructed_ground_into_dark_sky(tmp_path):
    data=payload();data['snapshot']['artificial_horizon_visible']=False
    spec,_=scene_from_kstars(data)
    value=Planner(tmp_path).run(spec).value
    row=value['sky']['directions'][0]
    assert not row['supported'] and row['total_relative'] is None
    assert row['status']=='kstars_artificial_horizon_obstructed'
    assert value['integration']['snapshot_id']==spec.to_dict()['integration']['snapshot_id']


def test_native_http_job_saves_exact_source_frame(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        capabilities=client.get('/api/kstars/capabilities').json()
        assert capabilities['snapshot_schema']==1
        data=payload();response=client.post('/api/kstars/jobs',json=data)
        assert response.status_code==202
        job=response.json()
        assert job['request']['directions']==[[180.,45.]]
        assert (tmp_path/'kstars-frames'/(job['id']+'.json')).exists()
        data['snapshot']['altitude_convention']='unknown'
        assert client.post('/api/kstars/jobs',json=data).status_code==422
