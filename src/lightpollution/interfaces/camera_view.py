"""Linear FITS and fixed-stretch illustration; FITS Viewer supplies interaction."""

import csv
from pathlib import Path

from astropy import units as u
from astropy.coordinates import AltAz,EarthLocation,SkyCoord
from astropy.io import fits
from astropy.time import Time,TimeDelta
from astropy.utils import iers
from PIL import Image,ImageDraw
import numpy as np

from ..adapters.storage import content_key,write_json
from ..models.camera import read_noise_lower_bound,sample_detector
from ..models.camera_field import directions,pixels,project_star,deposit_electrons,sample_camera_patch
from .eye_view import _font

CAMERA_EXPORTS=('camera.fits','camera-preview.png','camera-stars.csv','camera-profile.json','camera-ekos.json')


def render_camera_arrays(result):
    camera=result['camera'];config=camera['config'];view=camera['view']
    sensor=config['profile']['sensor'];optics=config['profile']['optics'];t=config['exposure_s']
    city,residual,supported,omega=sample_camera_patch(camera['field'])
    x,y=pixels(view);az,alt=directions(view,x,y)
    supported&=alt>=10
    if result.get('terrain'):
        terrain=result['terrain'];horizon=np.interp(az,np.r_[terrain['horizon_azimuth_deg'],360.],
            np.r_[terrain['horizon_altitude_deg'],terrain['horizon_altitude_deg'][0]])
        supported&=alt>horizon
    area=np.pi*(optics['aperture_diameter_mm']*.001/2)**2
    collect=area*optics['throughput']
    edge=np.tan(np.deg2rad(view['horizontal_fov_deg'])/2)
    pupil_cosine=1/np.sqrt(1+(x*edge)**2+(y*edge*view['height']/view['width'])**2)
    sky=t*collect*omega*pupil_cosine*(city+residual)
    star_buffer=np.zeros_like(sky);rows=[]
    for star in camera['stars']:
        position=project_star(view,star['azimuth_deg'],star['altitude_deg'])
        if position is None:continue
        px,py,scales,rho=position
        margin=6*np.deg2rad(optics['psf_fwhm_arcsec']/3600)/np.sqrt(8*np.log(2))/min(scales)
        if not (-margin<=px<view['width']+margin and -margin<=py<view['height']+margin):continue
        inside=bool(-.5<=px<view['width']-.5 and -.5<=py<view['height']-.5)
        record={**star,'x_px':float(px),'y_px':float(py),'centre_in_frame':inside,'input_e':None,
            'deposited_e':None,'outside_frame_or_truncated_e':None,'gaussian_tail_upper_bound_e':None,
            'ideal_isolated_aperture_snr':None,'expected_saturation_in_aperture':None}
        if star['available']:
            tx=(2*(px+.5)/view['width']-1)*edge
            ty=(1-2*(py+.5)/view['height'])*edge*view['height']/view['width']
            electrons=t*collect*star['response_per_m2_s']/np.sqrt(1+tx*tx+ty*ty)
            record.update(deposit_electrons(star_buffer,position,electrons,optics['psf_fwhm_arcsec']))
        rows.append(record)
    sky=np.where(supported,sky,0.);star_buffer=np.where(supported,star_buffer,0.)
    detector=sample_detector(star_buffer,sky,config)
    detector['adu'][~supported]=0
    # A simple isolated-aperture diagnostic; saturated/edge apertures are excluded.
    capacity=min(sensor['full_well_e'],(2**sensor['bit_depth']-1-sensor['bias_adu'])*sensor['gain_e_per_adu'])
    for row in rows:
        if not row['available'] or not row['centre_in_frame']:continue
        px,py=row['x_px'],row['y_px'];position=project_star(view,row['azimuth_deg'],row['altitude_deg'])
        scales=position[2];radius=1.5*np.deg2rad(optics['psf_fwhm_arcsec']/3600)/np.sqrt(np.prod(scales))
        x0,x1=int(np.floor(px-radius)),int(np.ceil(px+radius))+1
        y0,y1=int(np.floor(py-radius)),int(np.ceil(py+radius))+1
        if x0<0 or y0<0 or x1>view['width'] or y1>view['height']:continue
        yy,xx=np.mgrid[y0:y1,x0:x1];ap=(xx-px)**2+(yy-py)**2<=radius**2
        if not supported[y0:y1,x0:x1][ap].all():continue
        saturated=bool(np.any(detector['expected_e'][y0:y1,x0:x1][ap]>=capacity))
        row['expected_saturation_in_aperture']=saturated
        if saturated:continue
        signal=row['input_e']*(1-np.exp(-(1.5*np.sqrt(8*np.log(2)))**2/2))
        n=np.pi*radius**2;bp=float(sky[min(view['height']-1,max(0,round(py))),min(view['width']-1,max(0,round(px)))])
        noise=np.sqrt(signal+n*(bp+sensor['dark_current_e_per_s']*t+sensor['read_noise_e']**2))
        row['ideal_isolated_aperture_snr']=float(signal/noise) if noise>0 else None
    return {'city_e':t*collect*omega*pupil_cosine*city,'unresolved_e':t*collect*omega*pupil_cosine*residual,
        'sky_e':sky,'star_e':star_buffer,'supported':supported,'pixel_solid_angle_sr':omega,
        'projected_pupil_cosine':pupil_cosine,
        **detector},rows


def exposure_bridge(camera,arrays):
    config=camera['config'];sensor=config['profile']['sensor'];optics=config['profile']['optics']
    valid=arrays['supported'];t=config['exposure_s'];b=arrays['sky_e'][valid]/t
    minimum=read_noise_lower_bound(b,sensor['read_noise_e'])
    capacity=min(sensor['full_well_e'],sensor['gain_e_per_adu']*(2**sensor['bit_depth']-1-sensor['bias_adu']))
    maximum_rate=float((arrays['expected_e'][valid]/t).max())
    upper=.9*capacity/maximum_rate if maximum_rate>0 else None
    lower=float(minimum.max()) if np.isfinite(minimum).all() else None
    return {'schema_version':1,'kind':'lightpollution_camera_exposure_bridge','profile_id':camera['profile_id'],
        'pixel_pitch_um':sensor['pixel_pitch_um'],'focal_ratio':optics['focal_length_mm']/optics['aperture_diameter_mm'],
        'sky_rate_e_pixel_s_min_median_max':[float(b.min()),float(np.median(b)),float(b.max())],
        'read_noise_e':sensor['read_noise_e'],'noise_increase_percent':5.,
        'read_noise_lower_bound_s_max_over_field':lower,
        'read_noise_lower_bound_s_median':float(np.median(minimum)) if np.isfinite(np.median(minimum)) else None,
        'exposure_upper_bound_s_at_90_percent_capacity':upper,
        'bounds_compatible':bool(lower is not None and upper is not None and lower<=upper),
        'same_noise_criterion_as_Ekos':True,'criterion':'sqrt((b*t+RN²)/(b*t)) <=1.05; sky only',
        'is_unique_optimal_exposure':False,'V_band_is_not_SQM':True,
        'automatic_hardware_or_capture_sequence_changes':False,
        'Ekos_scalar_formula_reference':'OptimalSubExposureCalculator::calculateCFactor; same noise criterion, direct spatial electron rates',
        'profile_is_measured_camera':False}


def fits_header(result):
    camera=result['camera'];view=camera['view'];config=camera['config'];sensor=config['profile']['sensor'];optics=config['profile']['optics']
    observer=result['request']['observer'];model_observer=result['observation']['observer'];mid=Time(result['request']['time'])
    location=EarthLocation.from_geodetic(observer['longitude_deg']*u.deg,observer['latitude_deg']*u.deg,model_observer['height_m']*u.m)
    az,alt=directions(view,np.array([0.,2/view['width'],0.]),np.array([0.,0.,-2/view['height']]))
    with iers.conf.set_temp('auto_download',False),iers.conf.set_temp('auto_max_age',None):
        coords=SkyCoord(az=az*u.deg,alt=alt*u.deg,frame=AltAz(obstime=mid,location=location,pressure=0*u.hPa)).icrs
    rays=coords.cartesian.xyz.value.T;ra,dec=np.deg2rad([coords.ra.deg[0],coords.dec.deg[0]])
    east=np.array([-np.sin(ra),np.cos(ra),0.]);north=np.array([-np.sin(dec)*np.cos(ra),-np.sin(dec)*np.sin(ra),np.cos(dec)])
    plane=np.column_stack([rays@east,rays@north])/(rays@rays[0])[:,None]
    cd=np.rad2deg((plane[1:]-plane[0]).T)
    h=fits.Header();h['BUNIT']='ADU';h['IMAGETYP']='SIMULATED';h['SIMULATE']=True
    start=mid-TimeDelta(config['exposure_s']/2,format='sec')
    h['DATE-OBS']=start.isot;h['MJD-OBS']=start.mjd;h['DATE-AVG']=mid.isot;h['MJD-AVG']=mid.mjd;h['TIMESYS']='UTC'
    h['EXPTIME']=(config['exposure_s'],'s');h['EGAIN']=(sensor['gain_e_per_adu'],'electron/ADU')
    h['RDNOISE']=(sensor['read_noise_e'],'electron RMS');h['DARKCUR']=(sensor['dark_current_e_per_s'],'electron/pixel/s')
    h['FWELLE']=(sensor['full_well_e'],'electron');h['OFFSET']=(sensor['bias_adu'],'ADU');h['ADCBITS']=sensor['bit_depth']
    h['XPIXSZ']=(sensor['pixel_pitch_um'],'um');h['YPIXSZ']=(sensor['pixel_pitch_um'],'um')
    h['FOCALLEN']=(optics['focal_length_mm'],'mm');h['APTDIA']=(optics['aperture_diameter_mm'],'mm')
    h['RNGSEED']=config['seed'];h['CAMPROF']=config['profile']['profile_id'];h['RESPKIND']=config['profile']['response']['kind']
    h['CTYPE1']='RA---TAN';h['CTYPE2']='DEC--TAN';h['CUNIT1']='deg';h['CUNIT2']='deg';h['RADESYS']='ICRS'
    h['CRVAL1']=float(coords.ra.deg[0]);h['CRVAL2']=float(coords.dec.deg[0]);h['CRPIX1']=(view['width']+1)/2;h['CRPIX2']=(view['height']+1)/2
    for i in range(2):
        for j in range(2):h[f'CD{i+1}_{j+1}']=float(cd[i,j])
    h['HISTORY']='Synthetic scenario. UTC is exposure midpoint; geometry frozen. Not a measured RAW.'
    h['HISTORY']='No eye threshold or human tone mapping. Photon Poisson + dark Poisson + Gaussian read noise.'
    h['HISTORY']='Read MASK extension. Expected maps/variance are BEFORE clipping and quantization.'
    return h


def export_camera(result,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);camera=result['camera']
    record={'schema_version':1,'available':False,'reasons':camera['reasons'],'image_is_photograph':False}
    if camera['available']:
        arrays,rows=render_camera_arrays(result);valid=arrays['supported']
        if valid.any():
            config=camera['config'];profile=config['profile'];sensor=profile['sensor'];view=camera['view']
            primary=fits.PrimaryHDU(arrays['adu'],fits_header(result));primary.header['NVALID']=int(valid.sum())
            hdus=[primary]
            for key,name,unit in [('expected_e','MEAN_E','electron'),('sky_e','SKY_E','electron'),
                    ('city_e','CITY_E','electron'),('unresolved_e','OTHER_E','electron'),('star_e','STAR_E','electron'),
                    ('dark_e','DARK_E','electron'),('variance_e2_before_clipping','VAR_E2','electron2'),
                    ('pixel_solid_angle_sr','OMEGA_SR','sr')]:
                h=fits.ImageHDU(np.where(valid,arrays[key],np.nan).astype(np.float32),name=name);h.header['BUNIT']=unit
                hdus.append(h)
            hdus.append(fits.ImageHDU(valid.astype(np.uint8),name='MASK'))
            fits.HDUList(hdus).writeto(output/'camera.fits',overwrite=True,checksum=True)
            bridge=exposure_bridge(camera,arrays);write_json(output/'camera-ekos.json',bridge)
            write_json(output/'camera-profile.json',{'schema_version':1,'kind':'camera_profile','profile':profile})
            with (output/'camera-stars.csv').open('w',newline='') as stream:
                fields=list(rows[0]) if rows else ['id','available','input_e','deposited_e']
                writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(rows)
            preview=config['preview'];linear=np.clip((arrays['adu'].astype(float)-preview['black_adu'])/(preview['white_adu']-preview['black_adu']),0,1)
            tone=np.arcsinh(preview['asinh_scale']*linear)/np.arcsinh(preview['asinh_scale'])
            encoded=np.where(tone<=.0031308,12.92*tone,1.055*tone**(1/2.4)-.055)
            pixels=np.rint(255*encoded).astype(np.uint8);pixels[~valid]=38
            height,width=pixels.shape;image=Image.new('RGB',(width,height+120),(10,13,18));image.paste(Image.fromarray(pixels).convert('RGB'),(0,60))
            draw=ImageDraw.Draw(image);font=_font(max(10,min(18,width//45)))
            draw.text((15,12),'КАМЕРА · ЧИСЛЕННЫЙ МОНОХРОМНЫЙ КАДР',font=font,fill=(210,218,225))
            draw.text((15,34),f"{config['exposure_s']:g} с · f/{bridge['focal_ratio']:g} · {sensor['pixel_pitch_um']:g} мкм · фиксированный stretch ADU",font=font,fill=(155,170,185))
            draw.text((15,height+74),f"{profile['profile_id']} · источник звёзд: {camera['catalogue']['provider']}",font=font,fill=(155,170,185))
            draw.text((15,height+95),'Модель сценария; спектральный отклик и точность камеры требуют проверки.',font=font,fill=(155,170,185))
            image.save(output/'camera-preview.png')
            record.update(available=True,reasons=[],supported_pixels=int(valid.sum()),field_pixels=int(valid.size),
                stars_in_frame=sum(r['centre_in_frame'] for r in rows),star_rows=len(rows),
                stars_with_available_response=sum(r['available'] for r in rows),
                deposited_stellar_e=float(arrays['star_e'][valid].sum()),sky_e_min_median_max=[float(f(arrays['sky_e'][valid])) for f in [np.min,np.median,np.max]],
                fullwell_clipped_pixels=int(np.count_nonzero(arrays['fullwell_mask']&valid)),
                adc_clipped_pixels=int(np.count_nonzero(arrays['adc_clipped_mask']&valid)),
                expected_e_above_capacity=int(np.count_nonzero((arrays['expected_e']>=min(sensor['full_well_e'],sensor['gain_e_per_adu']*(2**sensor['bit_depth']-1-sensor['bias_adu'])))&valid)),
                exposure_bridge=bridge,preview={'operator':'fixed ADU asinh then one sRGB; no automatic per-frame scaling',**preview},
                FITS_extensions=[h.name for h in hdus],linear_FITS=True,
                variance_includes_quantization=False,variance_valid_before_clipping=True)
            record['projected_entrance_pupil_cosine_applied']=True
            record['stellar_flux_ledger_precedes_model_domain_mask']=True
        else:record['reasons']=['camera_field_has_no_supported_sky_pixels']
    if not record['available']:
        for name in CAMERA_EXPORTS:
            p=output/name
            if p.is_file():p.replace(p.with_name(p.stem+'.previous'+p.suffix))
    write_json(output/'camera.json',{'model':camera['model'],'config':camera['config'],'exports':record,
        'response':camera['response'],'catalogue':camera['catalogue'],'assumptions':camera['assumptions'],
        'source_coefficient':camera['source_coefficient'],'ground_camera_validation':False})
    return record
