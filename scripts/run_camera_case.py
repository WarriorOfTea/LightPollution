"""Freeze a native KStars camera case; export repeatable 30/90s and city-off controls."""

import argparse
from copy import deepcopy
import gzip
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from lightpollution.adapters.spectral_library import load_spectral_library
from lightpollution.adapters.storage import content_key,file_hash,read_json,write_json
from lightpollution.interfaces.camera_view import export_camera,render_camera_arrays
from lightpollution.models.camera_field import project_star
from lightpollution.models.spectra import ARCSEC2_PER_SR


def summarize(result,record,rows):
    camera=result['camera'];config=camera['config'];view=camera['view']
    return {'exposure_s':config['exposure_s'],'seed':config['seed'],'profile':config['profile'],
        'view':view,'source_coefficient':camera['source_coefficient'],'catalogue':camera['catalogue'],
        'refinement':camera['field']['refinement'],'heldout_validation':camera['field'].get('validation'),
        'exports':record,'in_frame_stars_with_V_proxy':sum(r['centre_in_frame'] and 'proxy' in r['flux_origin'] for r in rows),
        'available_isolated_aperture_SNR_records':sum(r['ideal_isolated_aperture_snr'] is not None for r in rows),
        'faintest_received_catalogue_V':max((r['v_mag'] for r in config['catalogue']['rows']),default=None)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-directory',type=Path,default=Path('data/camera-model-20261112/native-reviewed'))
    parser.add_argument('--output',type=Path,default=Path('artifacts/camera-model-20261112'))
    parser.add_argument('--workspace',type=Path,default=Path('data/camera-model-20261112/exports'))
    parser.add_argument('--render-saved',action='store_true')
    parser.add_argument('--input',type=Path,default=Path('artifacts/camera-model-20261112/camera-input.json'))
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    if args.render_saved:base=read_json(args.input)
    else:
        path=args.native_directory/'zenith-result.json';full=read_json(path)
        base={'artifact_type':'frozen_camera_input','request':{k:full['request'][k] for k in ['time','observer','spectrum']},
            'observation':{k:full['observation'][k] for k in ['observer','atmosphere']},
            'calibration':full['calibration'],'camera':{k:v for k,v in full['camera'].items() if k!='exports'},
            'terrain':full.get('terrain'),
            'source_result':{'path':str(path),'sha256':file_hash(path),'runtime':full['runtime'],'cache':full['cache']},
            'data_provenance':full['data_provenance'],'new_ground_RAW_frames':0}
        original_in_frame=0
        for star in full['stars']['rows']:
            pos=project_star(full['camera']['view'],star['azimuth_deg'],star['altitude_deg'])
            if pos and -.5<=pos[0]<full['camera']['view']['width']-.5 and -.5<=pos[1]<full['camera']['view']['height']-.5:
                original_in_frame+=1
        base['BSC_stars_in_same_sensor_field']=original_in_frame
        write_json(out/'camera-input.json',base)
    if args.render_saved and args.input.resolve()!=(out/'camera-input.json').resolve():shutil.copyfile(args.input,out/'camera-input.json')
    dark=deepcopy(base);field=dark['camera']['field'];factor=dark['camera']['response']['exact_V_factor_per_J']
    assert factor is not None,'This city-off control uses the exact V-matched response.'
    library=load_spectral_library();reference=library.zero_v*10**(-.4*22)*ARCSEC2_PER_SR
    city=np.asarray(field['city_response_per_m2_sr_s'])
    field['total_V_relative']=np.maximum(0,np.asarray(field['total_V_relative'])-city/(factor*reference)).tolist()
    field['city_response_per_m2_sr_s']=np.zeros_like(city).tolist()
    field['inherited_city_on_interpolation_validation']=field.pop('validation',None)
    dark['camera']['counterfactual_city_emission_multiplier']=0.
    longer=deepcopy(base);longer['camera']['config']['exposure_s']=90.;longer['camera']['config']['seed']+=1
    cases={'city-30s':base,'city-off-30s':dark,'city-90s':longer};summary={};arrays={}
    for name,result in cases.items():
        working=args.workspace/name;record=export_camera(result,working);assert record['available']
        arrays[name],rows=render_camera_arrays(result)
        target=out/name;target.mkdir(exist_ok=True)
        for filename in ['camera-preview.png','camera-profile.json','camera-stars.csv','camera-ekos.json','camera.json']:
            shutil.copyfile(working/filename,target/filename)
        if name=='city-30s':
            with (working/'camera.fits').open('rb') as src,(target/'camera.fits.gz').open('wb') as dest:
                with gzip.GzipFile(filename='',fileobj=dest,mode='wb',mtime=0) as packed:shutil.copyfileobj(src,packed)
        summary[name]=summarize(result,record,rows)
    assert content_key(base['camera']['stars'])==content_key(dark['camera']['stars'])==content_key(longer['camera']['stars'])
    assert all(result['camera']['source_coefficient']==base['camera']['source_coefficient'] for result in cases.values())
    assert np.allclose(arrays['city-90s']['expected_e'],3*arrays['city-30s']['expected_e'])
    # Synthetic target magnitudes, at the observer; no second extinction.
    magnitudes=np.linspace(8,14,121);profile=base['camera']['config']['profile'];optics=profile['optics'];sensor=profile['sensor']
    area=np.pi*(optics['aperture_diameter_mm']*.001/2)**2
    radius=1.5*optics['psf_fwhm_arcsec']/(206.264806*sensor['pixel_pitch_um']/optics['focal_length_mm'])
    npix=np.pi*radius**2;fraction=1-np.exp(-(1.5*np.sqrt(8*np.log(2)))**2/2)
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained');snr_controls={}
    for name,result in cases.items():
        config=result['camera']['config'];t=config['exposure_s'];sky=arrays[name]['sky_e'];h,w=sky.shape
        b=float(sky[h//2,w//2]);signal=t*area*optics['throughput']*factor*library.zero_v*10**(-.4*magnitudes)*fraction
        snr=signal/np.sqrt(signal+npix*(b+sensor['dark_current_e_per_s']*t+sensor['read_noise_e']**2))
        axes[0].semilogy(magnitudes,snr,label=name);snr_controls[name]={'sky_e_central_pixel':b,'SNR':snr.tolist()}
    axes[0].set(xlabel='Observed V magnitude: synthetic isolated target',ylabel='Ideal aperture SNR',title='Same physical stellar input; different sky/exposure')
    axes[0].legend();axes[0].grid(alpha=.2)
    image=axes[1].imshow(arrays['city-30s']['sky_e']/30,origin='upper',cmap='cividis')
    axes[1].set(title='City-on sky rate, electron/pixel/s',xlabel='Sensor x',ylabel='Sensor y');fig.colorbar(image,ax=axes[1])
    fig.savefig(out/'camera-controls.png',dpi=150);plt.close(fig)
    summary.update(source_coefficient_unchanged=True,stellar_input_table_unchanged=True,
        source_input_sha256=content_key(base['camera']['stars']),BSC_stars_in_same_sensor_field=base.get('BSC_stars_in_same_sensor_field'),
        all_camera_star_fluxes_rendered_without_eye_threshold=True,new_ground_RAW_frames=0,new_human_observers=0,
        synthetic_SNR_controls={'observed_V_magnitudes':magnitudes.tolist(),'cases':snr_controls,'npix':npix,
            'not_measured_stars_or_a_detection_probability':True})
    write_json(out/'case-summary.json',summary)
    print('Camera cases: 3/3; stars',summary['city-30s']['exports']['stars_in_frame'],
        '; BSC same field',summary['BSC_stars_in_same_sensor_field'],'; measured RAW0.')


if __name__=='__main__':main()
