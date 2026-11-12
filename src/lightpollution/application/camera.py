"""Connect frozen at-observer light to the camera, without an eye transfer."""

from copy import deepcopy
from collections import Counter
from dataclasses import replace

import numpy as np

from ..adapters.spectral_library import load_spectral_library
from ..adapters.storage import content_key
from ..adapters.catalogue_identity import cross_id,load_cross_identifications
from ..models.camera import MODEL,camera_config,camera_view,response_weights
from ..models.camera_field import build_camera_patch
from ..models.catalogue_photometry import johnson_photometry
from ..models.spectra import ARCSEC2_PER_SR,band_integral,stellar_spectral_observation
from ..physics import Atmosphere,Observer
from ..skyview import stellar_extinction


def camera_response(config,library):
    weights,exact_v_factor=response_weights(config['profile'],library)
    if exact_v_factor is not None:
        unresolved_factor=exact_v_factor;unresolved_status='exact_for_declared_V_energy_matched_response'
    elif config['unresolved_spectrum']=='flat_energy_V_normalized':
        shape=np.ones_like(library.wavelengths)
        unresolved_factor=float(band_integral(library.wavelengths,shape,weights)/
            band_integral(library.wavelengths,shape,library.responses['johnson_v']))
        unresolved_status='explicit_flat_energy_scenario_not_measured_sky_spectrum'
    else:unresolved_factor=None;unresolved_status='missing_unresolved_sky_spectrum'
    return weights,exact_v_factor,unresolved_factor,unresolved_status


def camera_star_table(result,config,weights,exact_v_factor,library):
    observer=Observer(**result['observation']['observer']);atmosphere=Atmosphere(**result['observation']['atmosphere'])
    if 'catalogue' in config:
        catalogue=config['catalogue'];native=catalogue['rows'];rows=[];colours={};names={};identities={};used_hr=set();photometry={};excluded=[]
        cross_index=load_cross_identifications()
        hd_counts=Counter(s.get('astrometry',{}).get('hd') for s in native)
        for index,star in enumerate(native):
            if star['altitude_deg']<10:continue
            hd=star.get('astrometry',{}).get('hd')
            identity=cross_id(star,cross_index) if hd_counts[hd]==1 else None
            hr=identity['hr'] if identity and identity['hr'] not in used_hr else 1_000_000+index
            used_hr.add(hr);names[hr]=star['id'];identities[hr]=identity
            magnitude,colour,policy=johnson_photometry(star)
            photometry[hr]={'input_magnitude':star['v_mag'],'input_colour_index':star['b_v'],
                'input_system':star.get('photometry','Johnson_V_BV'),'policy':policy}
            if magnitude is None:
                excluded.append({'id':star['id'],'name':star['name'],'azimuth_deg':star['azimuth_deg'],
                    'altitude_deg':star['altitude_deg'],'observed_v_mag':None,'available':False,
                    'flux_origin':policy,'BSC_cross_id':identity,'spectrum_recipe':None,
                    'input_photometry':photometry[hr],'status':'catalogue_photometry_unavailable','response_per_m2_s':None})
                continue
            dm=float(stellar_extinction(np.array([star['altitude_deg']]),observer,atmosphere)[0])
            rows.append({'hr':hr,'name':star['name'],'azimuth_deg':star['azimuth_deg'],'altitude_deg':star['altitude_deg'],
                'catalogue_v_mag':magnitude,'observed_v_mag':magnitude+dm})
            if colour is not None:colours[hr]={'b_v':str(colour)}
            elif identity and hr in library.colours:colours[hr]=library.colours[hr]
        if rows:
            rows,_,_=stellar_spectral_observation(rows,observer,atmosphere,library.wavelengths,library.responses,
                colours,library.measured,library.zero_b,library.zero_v,result['request']['spectrum']['angstrom_exponent'])
        origin={'provider':'KStars live StarComponent','magnitude_limit_requested':catalogue['magnitude_limit'],
            'truncated':catalogue['truncated'],'rows_received':len(native),'utc':catalogue['utc'],
            'completeness_established':False,'catalogue_snapshot_sha256':content_key(catalogue),
            'brightest_received_v_mag':min((r['v_mag'] for r in native),default=None),
            'faintest_received_v_mag':max((r['v_mag'] for r in native),default=None),
            'native_colour_rows':sum(r['b_v'] is not None for r in native),
            'BSC_cross_identified_rows':sum(i is not None for i in identities.values()),
            'spectral_rows':sum(r.get('spectral_available',False) for r in rows),
            'provenance':catalogue.get('provenance',{}),
            'native_magnitude_systems':dict(Counter(s.get('photometry','Johnson_V_BV') for s in native)),
            'Tycho_to_Johnson_converted_rows':sum('linear_unreddened' in p['policy'] for p in photometry.values()),
            'photometry_unavailable_rows':len(excluded),
            'native_magnitude_limit_is_not_uniform_Johnson_V':any(s.get('photometry')=='Tycho_VT_BTminusVT_inferred' for s in native),
            'close_pairs_merged_by_distance_only':False}
    else:
        rows=result['stars']['rows'];names={r['hr']:'HR'+str(r['hr']) for r in rows}
        origin={'provider':'BSC5P','completeness_established':False,'rows_received':len(rows),
            'deep_camera_stars_missing':True}
    selected=[r for r in rows if r['altitude_deg']>=10]
    altitudes=np.asarray([r['altitude_deg'] for r in selected]);axis=library.wavelengths
    r_tau=stellar_extinction(altitudes,observer,replace(atmosphere,aerosol_optical_depth=0))*np.log(10)/2.5
    a_tau=stellar_extinction(altitudes,observer,replace(atmosphere,rayleigh_multiplier=0))*np.log(10)/2.5
    exponent=result['request']['spectrum']['angstrom_exponent']
    transmission=np.exp(-r_tau[:,None]*(550/axis)**4-a_tau[:,None]*(550/axis)**exponent)
    output=[]
    for index,row in enumerate(selected):
        record={'id':names[row['hr']],'name':row['name'],'azimuth_deg':row['azimuth_deg'],'altitude_deg':row['altitude_deg'],
            'observed_v_mag':row['observed_v_mag'],'available':True,'flux_origin':'saved_spectral_observed_V'}
        record['BSC_cross_id']=identities.get(row['hr']) if 'catalogue' in config else None
        record['spectrum_recipe']=row.get('spectrum_recipe')
        record['input_photometry']=photometry.get(row['hr']) if 'catalogue' in config else None
        v=row.get('johnson_v_flux_W_m2')
        if v is None:
            v=library.zero_v*10**(-.4*row['observed_v_mag']);record['flux_origin']='effective_550nm_observed_V_proxy_missing_SED'
        if exact_v_factor is not None:rate=v*exact_v_factor
        else:
            recipe=row.get('spectrum_recipe') or {};shape=None
            if recipe.get('kind')=='measured_reference_shape':shape=library.measured.get(recipe['hr'])
            elif recipe.get('kind')=='B_V_fitted_Planck_continuum':
                from ..models.spectra import blackbody_shape
                shape=blackbody_shape(axis,recipe['continuum_temperature_K'])
            if shape is None:
                record.update(available=False,status='stellar_spectrum_unavailable',response_per_m2_s=None)
                output.append(record);continue
            shape=shape*transmission[index];shape*=v/float(band_integral(axis,shape,library.responses['johnson_v']))
            rate=float(band_integral(axis,shape,weights))
        record.update(status='ok',response_per_m2_s=float(rate));output.append(record)
    if 'catalogue' in config:output.extend(excluded)
    return output,origin


def build_camera_result(result,config,evaluate,check=lambda:None):
    config=camera_config(config);library=load_spectral_library();view=camera_view(config)
    weights,exact_v,unresolved,status=camera_response(config,library)
    reasons=[]
    if unresolved is None:reasons.append('missing_unresolved_sky_spectrum_for_this_response')
    origin=result.get('integration',{})
    if origin.get('artificial_horizon_visible') is False and 'artificial_horizon' not in origin:
        reasons.append('KStars_focused_direction_obstructed')
    patch=build_camera_patch(view,evaluate,config['sampling'],check) if not reasons else None
    stars,origin=camera_star_table(result,config,weights,exact_v,library)
    return {'schema_version':1,'model':MODEL,'config':config,'view':view,'available':not reasons,
        'reasons':reasons,'field':patch,'stars':stars,'catalogue':origin,
        'response':{'kind':config['profile']['response']['kind'],'exact_V_factor_per_J':exact_v,
            'unresolved_factor_per_J':unresolved,'unresolved_spectrum_status':status,
            'wavelength_range_nm':[360,830],'out_of_band_response_assumed_zero':True},
        'source_coefficient':result['calibration']['source_coefficient'],'source_coefficient_refitted':False,
        'units':{'node':'detected photons per m2/sr/s before scalar throughput','star':'detected photons per m2/s before scalar throughput'},
        'assumptions':['monochrome linear detector; no CFA/demosaicing/white balance',
            'equatorial tracking; input UTC is exposure midpoint; horizon/atmosphere frozen over <=120s',
            'one effective achromatic Gaussian PSF including seeing/optics/tracking',
            'projected pupil cos(theta) and exact pixel solid angle; no additional mechanical vignetting/PRNU/DSNU/blooming/correlated noise',
            'source coefficient and atmospheric parameters are scenarios unless independently calibrated',
            'catalogue completeness and diffuse residual source separation are not independently measured',
            'temperature is metadata; supplied dark current belongs to that temperature; no universal doubling law',
            'no direct bright-Moon/lamp flare or diffuse deep-sky-object images'],
        'eye_threshold_used':False,'human_tone_operator_used':False,'ground_camera_validation':False,
        'spectral_library_id':library.identity,'profile_id':content_key(config['profile'])}
