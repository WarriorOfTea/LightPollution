import hashlib
from importlib.resources import files

import numpy as np
import pytest

from lightpollution.physics import Atmosphere, Observer
from lightpollution.skyview import (DisplayScenario, background_ratio, camera_basis, dome_coordinates,
    horizontal_stars, limiting_magnitude, load_stars, sample_sky, stellar_extinction)


def test_catalogue_has_provenance_and_known_stars_not_the_fourteen_nonstellar_entries():
    stars,metadata=load_stars()
    assert len(stars['hr'])==metadata['retained_rows']==9096
    path=files('lightpollution').joinpath('catalogues/bsc5.csv')
    assert hashlib.sha256(path.read_bytes()).hexdigest()==metadata['csv_sha256']
    assert not set(metadata['omitted_hr']) & set(stars['hr'].astype(int).astype(str))
    sirius=np.flatnonzero(stars['hr']==2491)[0]
    assert stars['vmag'][sirius]==-1.46
    assert stars['ra_deg'][sirius]==pytest.approx(101.2871)
    assert stars['dec_deg'][sirius]==pytest.approx(-16.7161)


def test_interpolation_wraps_north_without_creating_new_extrema():
    az=np.arange(0,360,90);alt=np.array([0,45,90])
    field=np.array([[4,8,12,8],[2,4,6,4],[1,1,1,1.]])
    np.testing.assert_allclose(sample_sky(az,alt,field,[-1,359,719],[0,0,0]),[4+4/90]*3)
    np.testing.assert_allclose(sample_sky(az,alt,field,[0,90,360],[90]*3),1)
    assert sample_sky(az,alt,field,45,22.5)==pytest.approx(4.5)
    with pytest.raises(ValueError):sample_sky(az,alt,field,0,-1)


def test_sky_view_is_not_mirrored_and_zenith_collapses():
    x,y=dome_coordinates([0,90,180,270],[0]*4)
    np.testing.assert_allclose(x,[0,-1,0,1],atol=1e-12)
    np.testing.assert_allclose(y,[1,0,-1,0],atol=1e-12)
    np.testing.assert_allclose(dome_coordinates([0,90,180,270],[90]*4),0,atol=1e-12)


def test_south_facing_camera_has_west_on_right_and_an_orthonormal_basis():
    forward,right,up=camera_basis(180,35)
    basis=np.array([forward,right,up])
    np.testing.assert_allclose(basis@basis.T,np.eye(3),atol=1e-12)
    assert right[0]<0  # East-north-up system: camera right points west.
    np.testing.assert_allclose(np.cross(right,forward),up,atol=1e-12)


def test_extinction_matches_analytic_vertical_and_increases_toward_horizon():
    atmosphere=Atmosphere();observer=Observer(height_m=0)
    result=stellar_extinction(np.array([0,15,45,90]),observer,atmosphere)
    tau=(atmosphere.rayleigh_sea_level*atmosphere.rayleigh_scale_height_m*
         (1-np.exp(-atmosphere.top_m/atmosphere.rayleigh_scale_height_m))+
         atmosphere.aerosol_optical_depth*(1-np.exp(-atmosphere.top_m/atmosphere.aerosol_scale_height_m)))
    assert result[-1]==pytest.approx(tau*2.5/np.log(10),rel=1e-8)
    assert np.all(np.diff(result)<0)


def test_more_background_hides_stars_and_adds_flux_rather_than_magnitudes():
    scenario=DisplayScenario(21.6)
    np.testing.assert_allclose(background_ratio([0,1,10],scenario),[1,2,11])
    assert limiting_magnitude(100,scenario)==pytest.approx(scenario.dark_limiting_mag-2.5)
    assert np.all(np.diff(limiting_magnitude(background_ratio([0,1,10],scenario),scenario))<0)


def test_real_astrometry_polaris_is_north_and_clock_changes_other_stars():
    stars,_=load_stars();observer=Observer()
    az,alt,meta=horizontal_stars(stars,observer,'2026-11-09T21:00:00Z')
    polaris=np.flatnonzero(stars['hr']==424)[0]
    assert alt[polaris]==pytest.approx(observer.latitude_deg,abs=1)
    assert min(az[polaris],360-az[polaris])<2
    assert meta['sun_altitude_deg']<-18 and meta['moon_altitude_deg']<0
    az2,alt2,_=horizontal_stars(stars,observer,'2026-11-09T22:00:00Z')
    assert np.max(np.abs(alt2-alt))>5
    with pytest.raises(ValueError,match='UTC'):
        horizontal_stars(stars,observer,'2026-11-09T21:00:00')
    with pytest.raises(ValueError,match='Sun'):
        horizontal_stars(stars,observer,'2026-11-09T10:00:00Z')


def test_render_new_location_does_not_claim_petersburg_or_the_wrong_source_year(tmp_path, monkeypatch):
    import json
    from matplotlib.figure import Figure
    from lightpollution import skyview

    observer = {'latitude_deg':49.532563, 'longitude_deg':6.549911, 'height_m':20}
    metadata = {'observer':observer, 'atmosphere':{},
                'provenance':{'input_kind':'blackmarble', 'region_metadata':{'year':2014}}}
    source = tmp_path/'sky.npz'
    np.savez(source, azimuth_deg=[0,90,180,270], altitude_deg=[0,45,90],
             radiance=[[4,6,8,6],[2,3,4,3],[1,1,1,1]], metadata_json=json.dumps(metadata))
    captions = []
    monkeypatch.setattr(Figure, 'savefig', lambda fig, *args, **kwargs:
                        captions.extend(text.get_text() for text in fig.texts))
    monkeypatch.setattr(skyview, '_perspective', lambda *args, **kwargs: {})
    result = skyview.render_observer(source, tmp_path/'render', '2014-09-14T20:08:00Z',
                                     DisplayScenario(23.5))
    text = '\n'.join(captions)
    assert 'VNP46A4 (2014)' in text
    assert '2025' not in text and 'Петербург' not in text
    assert result['observer'] == observer
    assert result['astrometry']['utc'] == '2014-09-14T20:08:00Z'
