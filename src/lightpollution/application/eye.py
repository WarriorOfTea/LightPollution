"""Attach bounded visual-threshold results to an existing physical scene."""

import csv
from pathlib import Path

import numpy as np

from ..adapters.storage import write_json
from ..models.eye import (MODEL, REFERENCE_SP, SCOTOPIC_MAXIMUM, V22_LUMINANCE,
                          evaluate_detection)
from ..skyview import sample_sky
from ..models.eye_field import interpolate_patch, patch_support, tangent_coordinates
from ..models.kstars_horizon import horizon_visible


def sample_eye_field(field, azimuth, altitude):
    args=(field['azimuth_deg'],field['altitude_deg'])
    p=sample_sky(*args,field['photopic_luminance_cd_m2'],azimuth,altitude)
    s=sample_sky(*args,field['scotopic_luminance_cd_m2'],azimuth,altitude)
    support=sample_sky(*args,np.asarray(field['physical_supported'],float),azimuth,altitude)>=1-1e-12
    if 'view_patch' in field:
        patch=field['view_patch']
        x,y,front=tangent_coordinates(patch['view'],azimuth,altitude)
        inside=front&(abs(x)<=1)&(abs(y)<=1)
        if np.any(inside):
            # np.where also supports a scalar query; no mutation of caller arrays.
            p=np.where(inside,interpolate_patch(patch,patch['photopic_luminance_cd_m2'],x,y),p)
            s=np.where(inside,interpolate_patch(patch,patch['scotopic_luminance_cd_m2'],x,y),s)
            support=np.where(inside,patch_support(patch,x,y),support)
    return p,s,support


def build_eye_result(result, config, angular_field, reference_star_scotopic_lux_at_v0):
    shape=(len(angular_field['altitude_deg']),len(angular_field['azimuth_deg']))
    # Total V-only residual has an explicit photopic bridge and an S/P
    # scenario. It is never presented as an inferred spectrum of the sky.
    residual=np.asarray(angular_field['unresolved_v_relative']).reshape(shape)
    p=np.asarray(angular_field['city_photopic_luminance_cd_m2'])+residual*V22_LUMINANCE
    s=np.asarray(angular_field['city_scotopic_luminance_cd_m2'])+residual*V22_LUMINANCE*config['unresolved_background_sp_ratio']
    field={**angular_field,'photopic_luminance_cd_m2':p.tolist(),
           'scotopic_luminance_cd_m2':s.tolist(),
           'interpolation':'periodic bilinear in linear light; no inferred sub-grid sky structure'}
    if 'view_patch' in angular_field:
        patch=angular_field['view_patch']
        residual=np.asarray(patch['unresolved_v_relative'])*V22_LUMINANCE
        field['view_patch']={**patch,
            'photopic_luminance_cd_m2':(np.asarray(patch['city_photopic_luminance_cd_m2'])+residual).tolist(),
            'scotopic_luminance_cd_m2':(np.asarray(patch['city_scotopic_luminance_cd_m2'])+
                residual*config['unresolved_background_sp_ratio']).tolist()}
        field['interpolation']='view-local positive tensor PCHIP on unit-sphere gnomonic rays; periodic bilinear outside patch'
    geometry=result['observation'].get('geometry') or {}
    moon=result['observation'].get('moon') or {}
    reasons=[]
    if result['request']['mode']!='dated':reasons.append('date_required')
    if geometry.get('sun',{}).get('altitude_deg',90)>-18:reasons.append('astronomical_darkness_required')
    if moon.get('altitude_deg',-90)>0:reasons.append('direct_moon_glare_and_nonlocal_adaptation_not_modelled')
    origin=result.get('integration',{})
    horizon=origin.get('artificial_horizon')
    field['artificial_horizon']=horizon
    if origin.get('artificial_horizon_visible') is False and horizon is None:
        reasons.append('focused_direction_blocked_by_KStars_horizon')
    available=not reasons

    def threshold_row(az,alt,irradiance):
        bp,bs,physical=sample_eye_field(field,az,alt)
        values=evaluate_detection(bp,bs,irradiance,config['observer_factor'])
        physical=bool(physical) and bool(horizon_visible(horizon,az,alt))
        ok=available and physical and bool(values['supported'])
        row={'azimuth_deg':float(az),'altitude_deg':float(alt),
            'photopic_background_cd_m2':float(bp),'scotopic_background_cd_m2':float(bs),
            'background_equivalent_cd_m2':float(values['background_equivalent_cd_m2']),
            'supported':ok,'status':('ok' if ok else reasons[0] if reasons else
                'physical_sky_unavailable' if not physical else 'outside_scotopic_domain'),
            'background_floor_applied':bool(values['background_floor_applied'])}
        for key in ['threshold_equivalent_illuminance_lux','threshold_scotopic_illuminance_lux',
                    'star_equivalent_illuminance_lux','flux_over_threshold','margin_mag']:
            number=float(values[key]);row[key]=number if ok and np.isfinite(number) else None
        row['above_threshold']=bool(values['above_threshold']) if ok else None
        return row

    directions=[]
    for direction in result['sky']['directions']:
        row=threshold_row(direction['azimuth_deg'],direction['altitude_deg'],0.)
        threshold=row['threshold_scotopic_illuminance_lux']
        row['reference_A0_observed_v_limit']=(-2.5*np.log10(threshold/reference_star_scotopic_lux_at_v0)
                                             if threshold is not None else None)
        directions.append(row)
    stars=[]
    for star in result['stars']['rows']:
        irradiance=star.get('scotopic_illuminance_lux')
        row=threshold_row(star['azimuth_deg'],star['altitude_deg'],irradiance or 0.)
        row.update(hr=star['hr'],name=star['name'],observed_v_mag=star['observed_v_mag'],
            catalogue_v_mag=star['catalogue_v_mag'],scotopic_illuminance_lux=irradiance,
            spectrum_kind=star.get('spectrum_recipe',{}).get('kind'))
        if irradiance is None:
            row.update(supported=False,status='stellar_spectrum_unavailable',above_threshold=None,
                       margin_mag=None,flux_over_threshold=None,star_equivalent_illuminance_lux=None)
        row['star_specific_observed_v_limit']=(star['observed_v_mag']+row['margin_mag']
            if row['margin_mag'] is not None else None)
        row['near_threshold']=abs(row['margin_mag'])<=.5 if row['margin_mag'] is not None else None
        row['bright_star_appearance_modelled']=False
        stars.append(row)
    return {'schema_version':1,'model':MODEL,'available':available,'global_restrictions':reasons,
        'config':config,'field':field,'directions':directions,'stars':stars,
        'summary':{'catalogue_stars_after_terrain_mask':len(stars),
            'threshold_supported_stars':sum(r['supported'] for r in stars),
            'above_threshold':sum(r['above_threshold'] is True for r in stars),
            'below_threshold':sum(r['above_threshold'] is False for r in stars),
            'unavailable':sum(r['above_threshold'] is None for r in stars),
            'not_a_detection_probability':True},
        'spectral_bridge':{'reference_Blackwell_S_P':REFERENCE_SP,
            'V22_photopic_luminance_cd_m2':V22_LUMINANCE,
            'city':'CIE photopic/scotopic integrals of transported lamp-scenario spectra',
            'unresolved':'V-to-photopic proxy from Cox/Crumey; S/P is explicit scenario, no full natural/Moon SED',
            'reference_A0_scotopic_lux_at_observed_V0':reference_star_scotopic_lux_at_v0},
        'assumptions':['completed local adaptation; binocular normal/averted viewing',
            'observer_factor_excludes_atmosphere; stellar transmission applied exactly once',
            'local background approximation, no direct nearby glare or adaptation dynamics',
            'threshold relation does not determine suprathreshold colour or apparent brightness',
            'counts refer to available BSC5P catalogue entries; faint-end completeness and unresolved multiples are not fitted',
            'no individual observer calibration or matched visual field validation'],
        'scotopic_max_background_cd_m2':SCOTOPIC_MAXIMUM,
        'physical_error_bound':None,'display_colour_modelled':False}


def export_eye(eye, output):
    output=Path(output)
    write_json(output/'eye.json',eye)
    for key in ['directions','stars']:
        rows=eye[key]
        if rows:
            with (output/('eye-'+key+'.csv')).open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
                writer.writeheader();writer.writerows(rows)
