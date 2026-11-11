"""Colour sidecar from frozen physical spectra, never a second eye threshold."""

import csv
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..adapters.colour_library import load_colour_library
from ..adapters.spectral_library import load_spectral_library
from ..adapters.storage import content_key,file_hash
from ..models.eye import eye_config
from ..models.eye_colour import VERSION,cone_colour_weight,encircled_pool_gain,xyz_unit_rgb
from ..models.eye_optics import optical_mixture
from ..models.spectra import band_integral,blackbody_shape
from ..physics import Atmosphere,Observer
from ..skyview import stellar_extinction


def stellar_colour_table(result):
    """Rebuild each documented SED shape with ONE transmission, then audit it.

    The frozen observed V flux is the amplitude authority. Transmission is
    applied to the TOA shape for chromatic shape only, not again to observed
    irradiance. Rebuilt P/S must agree with the existing physical table.
    No transport file or spectral transport-library identity is modified.
    """
    library=load_spectral_library();colour=load_colour_library()
    axis=library.wavelengths;xyz_response,lms_response=colour.responses(axis)
    observer=Observer(**result['observation']['observer'])
    atmosphere=Atmosphere(**result['observation']['atmosphere'])
    rows=result['stars']['rows']
    altitudes=np.asarray([r['altitude_deg'] for r in rows])
    tau_r=stellar_extinction(altitudes,observer,replace(atmosphere,aerosol_optical_depth=0))*np.log(10)/2.5
    tau_a=stellar_extinction(altitudes,observer,replace(atmosphere,rayleigh_multiplier=0))*np.log(10)/2.5
    exponent=result['request']['spectrum']['angstrom_exponent']
    transmission=np.exp(-tau_r[:,None]*(550/axis)**4-tau_a[:,None]*(550/axis)**exponent)
    output=[]
    for index,row in enumerate(rows):
        recipe=row.get('spectrum_recipe') or {}
        record={'hr':row['hr'],'name':row['name'],'available':False,
                'status':'stellar_spectrum_unavailable','spectrum_kind':recipe.get('kind')}
        if recipe.get('kind')=='measured_reference_shape':
            shape=library.measured.get(recipe['hr'])
        elif recipe.get('kind')=='B_V_fitted_Planck_continuum':
            shape=blackbody_shape(axis,recipe['continuum_temperature_K'])
        else:shape=None
        v=row.get('johnson_v_flux_W_m2')
        if shape is None or v is None or v<=0:
            output.append(record);continue
        observed_shape=shape*transmission[index]
        integral=float(band_integral(axis,observed_shape,library.responses['johnson_v']))
        if integral<=0:
            record['status']='zero_reconstructed_spectrum';output.append(record);continue
        observed=observed_shape*(v/integral)
        photopic=float(683*band_integral(axis,observed,library.responses['photopic']))
        scotopic=float(1700*band_integral(axis,observed,library.responses['scotopic']))
        expected=np.array([row['photopic_illuminance_lux'],row['scotopic_illuminance_lux']])
        errors=abs(np.array([photopic,scotopic])-expected)/np.maximum(expected,1e-30)
        if np.max(errors)>1e-6:
            record.update(status='saved_flux_reconstruction_mismatch',relative_P_S_error=errors.tolist())
            output.append(record);continue
        xyz=683*np.trapezoid(observed[:,None]*xyz_response,x=axis,axis=0)
        lms=np.trapezoid(observed[:,None]*lms_response,x=axis,axis=0)
        if xyz[1]<=0:
            record['status']='zero_colour_luminance';output.append(record);continue
        record.update(available=True,status='ok',xyz_1931_lux_scale=xyz.tolist(),
            lms_2006_weighted_W_m2=lms.tolist(),rod_weighted_W_m2=scotopic/1700,
            unit_Y_linear_srgb=xyz_unit_rgb(xyz).tolist(),photopic_illuminance_lux=photopic,
            scotopic_illuminance_lux=scotopic,relative_P_S_error=errors.tolist(),
            CIE_Y_to_existing_photopic_ratio=float(xyz[1]/photopic))
        output.append(record)
    return output,{'colour_library_id':colour.identity,'stellar_library_id':library.identity,
        'physical_flux_authority':'existing observed V; P/S reconstruction audited against frozen rows',
        'stellar_atmospheric_transmission_count':1,'source_coefficient_refitted':False,
        'XYZ_observer':'CIE1931 2deg','LMS_observer':'CIE2006 2deg peak-normalized tables',
        'LMS_units':'weighted W/m2; NOT trolands or Kirk paper-normalized q',
        'rod_units':'weighted W/m2, frozen scotopic lux divided by1700',
        'total_background_LMSR_available':False,'background_colour_inferred':False}


def ensure_eye_appearance(result):
    eye=result['eye'];config=eye_config(eye['config'])
    if config['appearance']['mode']=='monochrome':
        eye.pop('appearance',None)
        return None
    spectral=load_spectral_library();colour_library=load_colour_library()
    cache_key=content_key({'config':{k:config[k] for k in ['optics','observer_profile','appearance']},
        'stars':result['stars']['rows'],'observer':result['observation'].get('observer'),
        'local_backgrounds':[[r['hr'],r['photopic_background_cd_m2'],r['scotopic_background_cd_m2']]
            for r in eye['stars']],
        'atmosphere':result['observation'].get('atmosphere'),'spectrum':result['request'].get('spectrum'),
        'libraries':[spectral.identity,colour_library.identity],
        'code':file_hash(Path(__file__)),'model_code':file_hash(Path(__file__).parents[1]/'models/eye_colour.py'),
        'optics_code':file_hash(Path(__file__).parents[1]/'models/eye_optics.py')})
    if eye.get('appearance',{}).get('cache_key')==cache_key:
        return eye['appearance']
    colours,metadata=stellar_colour_table(result)
    mixture=optical_mixture(config['optics']['pupil_diameter_mm'])
    diameter=config['observer_profile']['colour_integration_diameter_arcmin']
    gain=encircled_pool_gain(mixture,diameter)
    thresholds={row['hr']:row for row in eye['stars']}
    for row in colours:
        threshold=thresholds[row['hr']]
        if row['available']:
            weight=float(cone_colour_weight(threshold['photopic_background_cd_m2'],
                threshold['scotopic_background_cd_m2'],row['photopic_illuminance_lux'],
                row['scotopic_illuminance_lux'],gain))
            row['cone_colour_weight']=weight
            row['appearance_unit_Y_rgb']=(1+weight*(np.asarray(row['unit_Y_linear_srgb'])-1)).tolist()
        else:
            row.update(cone_colour_weight=None,appearance_unit_Y_rgb=None)
    record={'schema_version':1,'model':VERSION,'experimental':True,'cache_key':cache_key,
        'observer_profile':config['observer_profile'],'integration_diameter_arcmin':diameter,
        'optical_pool_mean_gain_per_sr':gain,'rows':colours,'spectral_metadata':metadata,
        'colour_available':sum(r['available'] for r in colours),'catalogue_rows':len(colours),
        'unknown_colour_fallback':'achromatic display of known scotopic flux; not an invented SED',
        'background_display':'achromatic; no natural LMS reconstructed from V or S/P',
        'psychophysical_colour_threshold_validated':False,'independent_human_observers':0,
        'kirk2011_absolute_bridge_available':False}
    record['identity']=content_key({'model':VERSION,'profile':config['observer_profile'],
        'optics':config['optics'],'spectral':metadata,'rows':colours})
    eye['appearance']=record
    return record


def export_eye_colour(eye,output):
    record=eye.get('appearance')
    if not record:return
    output=Path(output)
    fields=['hr','name','available','status','spectrum_kind','X_lux_scale','Y_lux_scale','Z_lux_scale',
        'L_weighted_W_m2','M_weighted_W_m2','S_weighted_W_m2','R_weighted_W_m2',
        'cone_colour_weight','appearance_R_unitY','appearance_G_unitY','appearance_B_unitY']
    with (output/'eye-colour.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for row in record['rows']:
            values={key:row[key] for key in fields if key in row}
            if row['available']:
                values.update(dict(zip(fields[5:8],row['xyz_1931_lux_scale'],strict=True)))
                values.update(dict(zip(fields[8:11],row['lms_2006_weighted_W_m2'],strict=True)))
                values['R_weighted_W_m2']=row['rod_weighted_W_m2']
                values.update(dict(zip(fields[13:16],row['appearance_unit_Y_rgb'],strict=True)))
            writer.writerow(values)
