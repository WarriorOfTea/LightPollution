"""Numerical camera QA: frozen input, linear FITS, WCS, detector statistics."""

import argparse
from copy import deepcopy
from pathlib import Path

from astropy import units as u
from astropy.coordinates import AltAz,EarthLocation,SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.utils import iers
from astropy.wcs import WCS
import numpy as np

from lightpollution.adapters.storage import read_json,write_json
from lightpollution.interfaces.camera_view import render_camera_arrays
from lightpollution.models.camera import camera_config,sample_detector
from lightpollution.models.camera_field import directions


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=Path('artifacts/camera-model-20261112'))
    args=parser.parse_args();out=args.output;result=read_json(out/'camera-input.json')
    expected,rows=render_camera_arrays(result);mask=expected['supported'];view=result['camera']['view']
    with fits.open(out/'city-30s/camera.fits.gz',checksum=True) as hdus:
        checks=[h.verify_checksum()==1 for h in hdus];assert all(checks)
        assert np.array_equal(expected['adu'],hdus[0].data)
        assert np.array_equal(mask.astype(np.uint8),hdus['MASK'].data)
        maps={'MEAN_E':'expected_e','SKY_E':'sky_e','CITY_E':'city_e','OTHER_E':'unresolved_e',
            'STAR_E':'star_e','DARK_E':'dark_e','VAR_E2':'variance_e2_before_clipping','OMEGA_SR':'pixel_solid_angle_sr'}
        errors={}
        for name,key in maps.items():
            values=hdus[name].data
            assert np.all(np.isnan(values[~mask]))
            reference=expected[key][mask];error=float(np.max(abs(values[mask]-reference)/np.maximum(abs(reference),1e-25)))
            assert error<1e-6,(name,error)
            errors[name]=error
        wcs=WCS(hdus[0].header)
        w,h=view['width'],view['height'];points=np.array([[0,0],[w-1,0],[0,h-1],[w-1,h-1],[(w-1)/2,(h-1)/2]])
        az,alt=directions(view,2*(points[:,0]+.5)/w-1,1-2*(points[:,1]+.5)/h)
        observer=result['observation']['observer'];location=EarthLocation.from_geodetic(observer['longitude_deg']*u.deg,
            observer['latitude_deg']*u.deg,observer['height_m']*u.m)
        with iers.conf.set_temp('auto_download',False),iers.conf.set_temp('auto_max_age',None):
            truth=SkyCoord(az=az*u.deg,alt=alt*u.deg,frame=AltAz(obstime=Time(result['request']['time']),location=location,pressure=0*u.hPa)).icrs
        predicted=wcs.pixel_to_world(*points.T);separation=predicted.separation(truth).arcsec
        assert float(separation.max())<1.,separation
    # Frozen sky/stars, exact exposure scaling, never an eye threshold.
    longer=deepcopy(result);longer['camera']['config']['exposure_s']*=2
    doubled,_=render_camera_arrays(longer)
    for key in ['star_e','sky_e','expected_e']:assert np.allclose(doubled[key],2*expected[key])
    config=camera_config({'exposure_s':10.,'seed':846});sensor=config['profile']['sensor']
    sensor.update(read_noise_e=5.,dark_current_e_per_s=.1,gain_e_per_adu=1.,bias_adu=1000.)
    sample=sample_detector(np.full(200000,20.),np.full(200000,100.),config)
    measured=sample['adu'].astype(float)-1000;mean=float(measured.mean());variance=float(measured.var())
    assert abs(mean-121)<.2 and abs(variance-146)/146<.015
    # Archive the one initially slow physical run as a performance limit, not a failed detector.
    report={'FITS_HDU_checksums_passed':sum(checks),'FITS_HDU_checksums_checked':len(checks),
        'primary_uint16_ADU_values_matching':int(expected['adu'].size),'primary_ADU_values_checked':int(expected['adu'].size),
        'scientific_maps_relative_float32_rounding_errors':errors,'expected_maps_before_clipping':True,
        'WCS_test_pixels':points.tolist(),'WCS_error_arcsec':separation.tolist(),'WCS_max_error_arcsec':float(separation.max()),
        'WCS_declared_test_tolerance_arcsec':1.,'WCS_is_local_TAN_approximation':True,
        'exposure_doubling_stellar_sky_total_expectations':True,'eye_threshold_used':False,
        'Monte_Carlo':{'synthetic_pixels':200000,'seed':846,'expected_e':121.,'expected_variance_e2_before_quantization':146.,
            'measured_bias_subtracted_mean_ADU':mean,'measured_variance_ADU2':variance,'gain_e_per_ADU':1.,
            'relative_variance_difference_including_quantization':abs(variance-146)/146},
        'new_measured_RAW_frames':0,'ground_camera_validation':False}
    write_json(out/'numerical-verification.json',report)
    print('Camera QA:',sum(checks),'/',len(checks),'HDU checksums;',expected['adu'].size,'ADU values; WCS',float(separation.max()),'arcsec; Monte Carlo',variance)


if __name__=='__main__':main()
