"""Reproducible appearance sidecar over the frozen 0.7 physical calculation."""

import argparse
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from lightpollution.adapters.colour_library import load_colour_library
from lightpollution.adapters.spectral_library import load_spectral_library
from lightpollution.adapters.storage import content_key,file_hash,read_json,write_json
from lightpollution.application.eye_appearance import ensure_eye_appearance
from lightpollution.application.eye_validation import audit_observations
from lightpollution.interfaces.eye_view import render_eye_view
from lightpollution.models.eye import eye_config
from lightpollution.models.eye_colour import (cone_colour_weight,encircled_pool_gain,
    gamut_preserving_luminance,kirk2011_normalized_lmsr,xyz_unit_rgb)
from lightpollution.models.eye_display import srgb_encode
from lightpollution.models.eye_optics import optical_mixture
from lightpollution.models.eye_profiles import profile_document
from lightpollution.models.spectra import band_integral,blackbody_shape


def diagnostic(out,profile):
    """Known synthetic spectra/inputs, explicitly not field observations."""
    spectral=load_spectral_library();colour=load_colour_library();axis=spectral.wavelengths
    xyz_response,_=colour.responses(axis)
    gain=encircled_pool_gain(optical_mixture(profile['optics']['pupil_diameter_mm']),
        profile['observer_profile']['colour_integration_diameter_arcmin'])
    mags=np.linspace(-3,8,500);samples=[]
    fig,axes=plt.subplots(1,3,figsize=(15,4.3),layout='constrained')
    for temperature,c in [(3500,'#ad5b1c'),(6000,'#596269'),(10000,'#216aad')]:
        shape=blackbody_shape(axis,temperature)
        shape=shape*(spectral.zero_v/float(band_integral(axis,shape,spectral.responses['johnson_v'])))
        p=683*float(band_integral(axis,shape,spectral.responses['photopic']))
        s=1700*float(band_integral(axis,shape,spectral.responses['scotopic']))
        unit=xyz_unit_rgb(683*np.trapezoid(shape[:,None]*xyz_response,x=axis,axis=0))
        scale=10**(-.4*mags)
        weight=cone_colour_weight(2e-4,2.8e-4,p*scale,s*scale,gain)
        axes[0].plot(mags,weight,c=c,label=f'{temperature} K')
        swatches=[]
        for magnitude in [-2,0,2,4,6,8]:
            factor=10**(-.4*magnitude)
            w=float(cone_colour_weight(2e-4,2.8e-4,p*factor,s*factor,gain))
            rgb,compression=gamut_preserving_luminance(.12*(1+w*(unit-1)))
            swatches.append(srgb_encode(rgb))
            samples.append({'temperature_K':temperature,'observed_V_mag':magnitude,
                'photopic_illuminance_lux':p*factor,'scotopic_illuminance_lux':s*factor,
                'cone_colour_weight':w,'linear_display_Y':.12,'gamut_common_chroma_scale':float(compression),
                'srgb':srgb_encode(rgb).tolist()})
        y={3500:2,6000:1,10000:0}[temperature]
        for x,rgb in enumerate(swatches):axes[1].add_patch(plt.Rectangle((x,y),1,1,color=rgb))
    axes[0].set(xlabel='Observed V magnitude (synthetic spectra)',ylabel='Experimental cone-colour weight',ylim=(-.02,1.02),
        title='Ferwerda TVI control + explicit gate')
    axes[0].legend();axes[0].grid(alpha=.2)
    axes[1].set(xlim=(0,6),ylim=(0,3),xticks=np.arange(6)+.5,xticklabels=['−2','0','2','4','6','8'],
        yticks=[.5,1.5,2.5],yticklabels=['10000 K','6000 K','3500 K'],xlabel='Observed V magnitude',
        title='Equal display Y=0.12; colour only')
    rods=np.r_[0.,np.geomspace(1e-6,1e4,200)]
    inputs=np.column_stack([np.full_like(rods,.8),np.full_like(rods,.6),np.full_like(rods,.3),rods])
    transformed=kirk2011_normalized_lmsr(inputs,input_convention='paper_normalized')
    for index,label in enumerate(['L̂','M̂','Ŝ']):axes[2].plot(rods,transformed[:,index],label=label)
    axes[2].set(xscale='symlog',yscale='symlog',xlabel='Paper-normalized qR (arbitrary control)',
        ylabel='Kirk equations 10–13 output',title='Kirk research prototype; no physical bridge')
    axes[2].legend();axes[2].grid(alpha=.2)
    fig.savefig(out/'colour-controls.png',dpi=160);plt.close(fig)
    record={'synthetic_controls_not_observations':True,'background_P_S_cd_m2':[2e-4,2.8e-4],
        'SED_normalized_at_observed_V':True,'stellar_transmission_in_this_control':'none: already at-eye synthetic SED',
        'ferwerda_control_samples':samples,'kirk_input_domain':'paper_normalized; not our physical CIE2006 integrals',
        'kirk_control_qL_qM_qS':[.8,.6,.3],'kirk_rod_zero_identity_error':float(abs(transformed[0]-inputs[0,:3]).max()),
        'kirk_absolute_conversion_known':False,'comparison_is_not_human_validation':True}
    write_json(out/'colour-controls.json',record)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,default=Path('artifacts/eye-rendering-20261111/physical-result.json'))
    parser.add_argument('--output',type=Path,default=Path('artifacts/eye-colour-20261111'))
    parser.add_argument('--workspace',type=Path,default=Path('data/eye-colour-20261111'))
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    args.workspace.mkdir(parents=True,exist_ok=True)
    base=read_json(args.base);colour=deepcopy(base)
    physical_stars=content_key(base['stars']);coefficient=base['calibration']['source_coefficient']
    config=eye_config({**colour['eye']['config'],'appearance':{'mode':'experimental_stellar_colour'}})
    colour['eye']['config']=config;colour['request']['eye']=deepcopy(config)
    appearance=ensure_eye_appearance(colour)
    display=render_eye_view(colour,args.workspace/'colour')
    assert content_key(colour['stars'])==physical_stars
    assert colour['eye']['stars']==base['eye']['stars']
    assert colour['calibration']['source_coefficient']==coefficient
    mono=deepcopy(base);mono['eye']['config']=eye_config(base['eye']['config'])
    mono_display=render_eye_view(mono,args.workspace/'monochrome')
    for name in ['eye-view.png','eye-preview.png','eye-mobile.png','eye-view-16bit.png','eye-colour.csv','eye-display.json']:
        shutil.copyfile(args.workspace/'colour'/name,out/name)
    shutil.copyfile(args.workspace/'monochrome/eye-preview.png',out/'monochrome-preview.png')
    old=np.load(args.base.parent/'city/eye-linear.npz');new=np.load(args.workspace/'colour/eye-linear.npz')
    physics_arrays=['background_scotopic_cd_m2','photopic_background_cd_m2',
        'stellar_physical_pixel_illuminance_lux','stellar_appearance_pixel_illuminance_lux',
        'appearance_scotopic_cd_m2','pixel_solid_angle_sr','supported']
    array_checks={key:bool(np.array_equal(old[key],new[key])) for key in physics_arrays}
    assert all(array_checks.values())
    old.close();new.close()
    weights=np.asarray([r['cone_colour_weight'] for r in appearance['rows'] if r['available']])
    shown={r['hr'] for r in display['displayed_star_rows']}
    shown_weights=[r['cone_colour_weight'] for r in appearance['rows'] if r['available'] and r['hr'] in shown]
    summary={'completed_at':datetime.now(ZoneInfo('Europe/Berlin')).isoformat(timespec='seconds'),
        'base_result':{'path':str(args.base),'sha256':file_hash(args.base),
            'commit':'e84dc59d613c60b534ede8fd64e785271e32f68d'},
        'source_coefficient':coefficient,'source_coefficient_unchanged':True,
        'physical_stellar_table_hash':physical_stars,'physical_stellar_rows':len(base['stars']['rows']),
        'all_physical_stellar_rows_unchanged':True,'threshold_rows_unchanged':True,
        'physical_float32_array_equality_against_0_7':array_checks,
        'colour_model':{k:v for k,v in appearance.items() if k!='rows'},
        'maximum_P_S_reconstruction_relative_error':max(max(r['relative_P_S_error']) for r in appearance['rows'] if r['available']),
        'cone_colour_weight_available_rows_quantiles':np.quantile(weights,[0,.25,.5,.75,1]).tolist(),
        'shown_colour_rows':len(shown_weights),'shown_weights_over_point_one':sum(w>.1 for w in shown_weights),
        'colour_display':display,'monochrome_display':mono_display,
        'independent_human_observers':0,'ground_observation_nights':0,'calibrated_perceptual_match':False,
        'full_resolution_linear_data_regenerated_at':str(args.workspace/'colour/eye-linear.npz')}
    write_json(out/'case-summary.json',summary)
    write_json(out/'eye-profile.json',profile_document(config))
    empty={'schema_version':1,'kind':'eye_observations','cases':[],'observations':[]}
    write_json(out/'observation-template.json',empty)
    write_json(out/'observation-status.json',audit_observations(empty,{}))
    diagnostic(out,config)
    print('Frozen physical rows:',len(base['stars']['rows']),'unchanged; displayed:',display['displayed_stars'],
        '; colour available:',appearance['colour_available'],'; observers: 0',flush=True)


if __name__=='__main__':main()
