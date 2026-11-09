import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from lightpollution.data import Bounds, COMPOSITES, GRID, audit, decode, good_mask, read_region, save_region, load_region
from lightpollution.download import HDF5_MAGIC, select_files, validate_url, verify_file
from lightpollution.download import download_annual
from lightpollution.physics import Observer
from lightpollution.sources import regional_sources


def test_petersburg_has_four_tiles_and_correct_grid():
    bounds=Bounds()
    assert bounds.tiles()==("h20v02","h21v02","h20v03","h21v03")
    assert bounds.shape==(480,720)
    assert bounds.latitude[0]==pytest.approx(61-1/480)
    assert bounds.longitude[0]==pytest.approx(29+1/480)


def test_three_observations_are_not_high_quality():
    mask=good_mask(np.array([1,1,1,1,np.nan,-1]),np.array([0,0,1,2,0,0]),np.array([3,4,8,8,8,8]))
    np.testing.assert_array_equal(mask,[False,True,False,False,False,False])


def test_fill_mask_precedes_radiance_scaling(tmp_path):
    with h5py.File(tmp_path/'example.h5','w') as h:
        data=h.create_dataset('radiance',data=np.array([[-999.9,10]],dtype='float32'))
        data.attrs.update({'_FillValue':np.float32(-999.9),'scale_factor':0.1,'offset':0.0})
        decoded=decode(data,(slice(None),slice(None)))
        assert np.isnan(decoded[0,0])
        assert decoded[0,1]==1
        data.attrs['offset']=1
        with pytest.raises(ValueError,match='convention'): decode(data,(slice(None),slice(None)))


def make_tile(path: Path,h:int,v:int,value:float,year=2025):
    with h5py.File(path,'w') as f:
        f.attrs.update({'ShortName':'VNP46A4','VersionID':'002','RangeBeginningDate':f'{year}-01-01',
                        'RangeEndingDate':f'{year}-12-31','WestBoundingCoord':-180+10*h,
                        'EastBoundingCoord':-170+10*h,'NorthBoundingCoord':90-10*v,'SouthBoundingCoord':80-10*v})
        fields=f.create_group(GRID)
        for prefix in COMPOSITES:
            for suffix,constant in (('',value),('_Quality',0),('_Num',10),('_Std',0.1)):
                d=fields.create_dataset(prefix+'_Composite_Snow_Free'+suffix,shape=(2400,2400),dtype='f4',fillvalue=constant)
                d.attrs.update({'_FillValue':np.float32(-999.9),'scale_factor':1.,'offset':0.,'units':'nWatts/(cm^2 sr)'})
        d=fields.create_dataset('Land_Water_Mask',shape=(2400,2400),dtype='u1',fillvalue=1)
        d.attrs.update({'_FillValue':255,'scale_factor':1.,'offset':0.})
        fields.create_dataset('lat',data=90-10*v-(np.arange(2400)+0.5)/240)
        fields.create_dataset('lon',data=-180+10*h+(np.arange(2400)+0.5)/240)


@pytest.mark.parametrize('coordinate_convention',['cell_centres','upper_left_edges'])
def test_four_tile_mosaic_is_north_to_south_without_seam(tmp_path,coordinate_convention):
    bounds=Bounds(30-1/240,60-1/240,30+1/240,60+1/240)
    for h,v,value in ((20,2,1),(21,2,2),(20,3,3),(21,3,4)):
        make_tile(tmp_path/f'VNP46A4.A2025001.h{h:02}v{v:02}.002.2026001000000.h5',h,v,value)
    if coordinate_convention=='upper_left_edges':
        for path in tmp_path.glob('*.h5'):
            with h5py.File(path,'r+') as f:
                f[GRID+'/lat'][:]+=1/480
                f[GRID+'/lon'][:]-=1/480
    arrays,metadata=read_region(tmp_path,bounds)
    np.testing.assert_array_equal(arrays['NearNadir_Composite_Snow_Free'],[[1,2],[3,4]])
    assert len(metadata)==4
    assert all(m['coordinate_conventions']['lat']==coordinate_convention for m in metadata.values())
    summary=audit(arrays,bounds)
    assert summary['composites']['NearNadir']['accepted_pixels']==4
    path=tmp_path/'window.npz'
    save_region(path,arrays,bounds,{'year':2025})
    restored,b,meta=load_region(path)
    assert b==bounds and meta['year']==2025
    np.testing.assert_array_equal(restored['longitude'],bounds.longitude)


def test_missing_tile_fails_instead_of_silently_filling_dark(tmp_path):
    with pytest.raises(ValueError,match='pinned file'): read_region(tmp_path,Bounds())


def test_nonstandard_coordinate_shift_is_rejected(tmp_path):
    bounds=Bounds(30,59.5,30+1/240,59.5+1/240)
    path=tmp_path/'VNP46A4.A2025001.h21v03.002.2026001000000.h5'
    make_tile(path,21,3,1)
    with h5py.File(path,'r+') as f:
        f[GRID+'/lat'][:]+=0.001
    with pytest.raises(ValueError,match='Coordinate order'):
        read_region(tmp_path,bounds)


def test_aggregation_preserves_weights_and_uses_correct_water_classes():
    bounds=Bounds(30,59.5,30+4/240,59.5+4/240)
    arrays={}
    for prefix in COMPOSITES:
        key=prefix+'_Composite_Snow_Free'
        for suffix,value in (('',10),('_Quality',0),('_Num',10),('_Std',1)):
            arrays[key+suffix]=np.full((4,4),value,dtype=float)
    arrays['Land_Water_Mask']=np.array([[0,1,5,2]]*4)
    observer=Observer(60.3,30.315,20)
    sources,report=regional_sources(arrays,bounds,observer,block=2,exclude_radius_m=0)
    assert sources.power.sum()==pytest.approx(1)
    assert report['retained_positive_pixels']==12
    assert report['weight_removed_water_fraction']==pytest.approx(0.25)
    assert report['retained_weight_fraction']==pytest.approx(0.75)
    arrays['Land_Water_Mask']=arrays['Land_Water_Mask'].astype(float)
    arrays['Land_Water_Mask'][:,-1]=np.nan
    _,report=regional_sources(arrays,bounds,observer,block=2,exclude_radius_m=0,include_water=True)
    assert report['weight_removed_unknown_surface_fraction']==pytest.approx(0.25)
    assert report['weight_removed_water_fraction']==0
    assert report['retained_weight_fraction']==pytest.approx(0.75)


def test_download_chooses_latest_processing_timestamp():
    old={'name':'VNP46A4.A2025001.h20v02.002.2026001000000.h5'}
    new={'name':'VNP46A4.A2025001.h20v02.002.2026076194841.h5'}
    wrong={'name':'VNP46A4.A2025001.h20v02.001.2026076194841.h5'}
    assert select_files([old,new,wrong],2025,('h20v02',))==[new]
    with pytest.raises(ValueError): select_files([old],2025,('h21v02',))


@pytest.mark.parametrize('url',['http://ladsweb.modaps.eosdis.nasa.gov/file','https://example.com/file',
                              'https://ladsweb.modaps.eosdis.nasa.gov.example.com/file'])
def test_nasa_token_cannot_be_sent_to_other_hosts(url):
    with pytest.raises(ValueError): validate_url(url)


def test_html_truncation_and_checksum_are_not_accepted_as_science_data(tmp_path):
    path=tmp_path/'data.h5'
    path.write_bytes(b'<html>login</html>')
    with pytest.raises(ValueError,match='not HDF5'): verify_file(path,path.stat().st_size)
    path.write_bytes(HDF5_MAGIC+b'example')
    with pytest.raises(ValueError,match='byte count'): verify_file(path,100)
    with pytest.raises(ValueError,match='SHA-256'): verify_file(path,path.stat().st_size,'0'*64)


def test_authorization_failure_preserves_catalogue_without_claiming_a_download(tmp_path,monkeypatch,capsys):
    from lightpollution import download
    name='VNP46A4.A2025001.h20v02.002.2026076194841.h5'
    class Response:
        status_code=303
        headers={'Location':'/profiles/licenses/archive/file'}
        def raise_for_status(self): pass
        def json(self):
            return {'content':[{'name':name,'size':100,'downloadsLink':'https://ladsweb.modaps.eosdis.nasa.gov/archive/'+name}]}
        def __enter__(self): return self
        def __exit__(self,*args): pass
    monkeypatch.setattr(download.requests,'get',lambda *args,**kwargs:Response())
    monkeypatch.setenv('BLACKMARBLE_TOKEN','test-only-credential')
    with pytest.raises(RuntimeError,match='LAADS web sign-in'):
        download_annual(tmp_path,tiles=('h20v02',))
    saved=(tmp_path/'manifest.json').read_text()
    assert json.loads(saved)['files'][0]['download_verified'] is False
    assert 'test-only-credential' not in saved+capsys.readouterr().out
    assert not list(tmp_path.glob('*.h5')) and not list(tmp_path.glob('*.part'))


def test_aggregation_does_not_move_a_local_exclusion_ring_onto_the_observer():
    bounds=Bounds(30,60,30+12/240,60+12/240)
    observer=Observer((bounds.south+bounds.north)/2,(bounds.west+bounds.east)/2,20)
    key='NearNadir_Composite_Snow_Free'
    arrays={key:np.ones((12,12)),key+'_Quality':np.zeros((12,12)),
            key+'_Num':np.full((12,12),10), 'Land_Water_Mask':np.ones((12,12))}
    sources,report=regional_sources(arrays,bounds,observer,block=12,exclude_radius_m=500)
    assert sources.power.sum()==pytest.approx(1)
    assert report['nearest_effective_source_chord_m']>=500-0.001
    assert 0<report['weight_removed_local_fraction']<1


def test_distant_city_is_resolved_when_a_low_sightline_passes_above_it():
    bounds=Bounds(30.30,59.95,30.30+8/240,59.95+8/240)
    observer=Observer(60.30,30.315,20)
    key='NearNadir_Composite_Snow_Free'
    arrays={key:np.ones((8,8)),key+'_Quality':np.zeros((8,8)),
            key+'_Num':np.full((8,8),10),'Land_Water_Mask':np.ones((8,8))}
    coarse,_=regional_sources(arrays,bounds,observer,block=8)
    toward,_=regional_sources(arrays,bounds,observer,block=8,refinement_direction=(180,1))
    away,_=regional_sources(arrays,bounds,observer,block=8,refinement_direction=(0,1))
    assert len(toward.power)>len(coarse.power)==len(away.power)==1
    assert toward.power.sum()==pytest.approx(coarse.power.sum())
