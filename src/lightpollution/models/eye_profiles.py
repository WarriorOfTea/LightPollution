"""Display geometry/provenance and portable eye profiles; no implicit calibration."""

from copy import deepcopy
from datetime import datetime

import numpy as np

DISPLAY_DEFAULTS={'profile_id':'nominal-srgb-1','origin':'nominal','window_width_mm':None,
    'viewing_distance_mm':None,'ambient_illuminance_lux':None,'measured_at':None,'instrument':None}
OBSERVER_DEFAULTS={'profile_id':'nominal-observer-1','origin':'nominal',
    'colour_integration_diameter_arcmin':5.,'calibration_id':None}


def _text(value,name,nullable=False):
    if nullable and value is None:return
    if not isinstance(value,str) or not value.strip() or len(value)>160:
        raise ValueError(f'{name} requires nonempty text up to160 characters.')


def validate_profiles(config):
    display=config['display'];observer=config['observer_profile']
    for group in [display,observer]:_text(group['profile_id'],'profile_id')
    if display['origin'] not in ('nominal','user_supplied','instrument_measured'):
        raise ValueError('Unknown display-profile origin.')
    if observer['origin'] not in ('nominal','user_supplied'):
        raise ValueError('Observer calibration must be audited separately, not declared by an input flag.')
    _text(observer['calibration_id'],'calibration_id',True)
    for key,lo,hi in [('window_width_mm',10,3000),('viewing_distance_mm',100,10000),
                      ('ambient_illuminance_lux',0,100000)]:
        value=display[key]
        if value is not None and (isinstance(value,bool) or not isinstance(value,(int,float))
                or not np.isfinite(value) or not lo<=value<=hi):
            raise ValueError(f'{key} must be null or finite in [{lo},{hi}].')
    if (display['window_width_mm'] is None)!=(display['viewing_distance_mm'] is None):
        raise ValueError('Window width and viewing distance must be provided together.')
    diameter=observer['colour_integration_diameter_arcmin']
    if isinstance(diameter,bool) or not isinstance(diameter,(int,float)) or not np.isfinite(diameter) or not .5<=diameter<=30:
        raise ValueError('Colour pool diameter must be .5–30 arcmin.')
    _text(display['instrument'],'instrument',True)
    if display['measured_at'] is not None:
        try:
            instant=datetime.fromisoformat(display['measured_at'].replace('Z','+00:00'))
            if instant.tzinfo is None:raise ValueError()
        except (ValueError,TypeError,AttributeError):
            raise ValueError('Display measurement needs an ISO instant with timezone.') from None
    if display['origin']=='instrument_measured' and (not display['instrument'] or not display['measured_at']):
        raise ValueError('Measured display origin requires instrument and measurement time.')


def display_geometry(profile,scene_fov_deg):
    width,distance=profile['window_width_mm'],profile['viewing_distance_mm']
    angle=float(2*np.rad2deg(np.arctan(width/(2*distance)))) if width is not None else None
    return {'profile_id':profile['profile_id'],'profile_origin':profile['origin'],
        'scene_horizontal_fov_deg':scene_fov_deg,'displayed_horizontal_angle_deg':angle,
        'angular_scale_ratio':angle/scene_fov_deg if angle is not None else None,
        'angular_geometry_known':angle is not None,'measured_display_spectra_available':False,
        'individual_perceptual_match_validated':False}


def profile_document(eye):
    from .eye import eye_config
    normalized=eye_config(eye)
    keys=['observer_factor','unresolved_background_sp_ratio','adaptation','optics','appearance','observer_profile','display']
    return {'schema_version':1,'kind':'eye_view_profile','eye':{k:deepcopy(normalized[k]) for k in keys}}


def apply_profile(document,eye):
    if (not isinstance(document,dict) or set(document)!={'schema_version','kind','eye'}
            or type(document['schema_version']) is not int or document['schema_version']!=1
            or document['kind']!='eye_view_profile' or not isinstance(document['eye'],dict)):
        raise ValueError('Expected eye_view_profile schema1.')
    allowed={'observer_factor','unresolved_background_sp_ratio','adaptation','optics','appearance','observer_profile','display'}
    if set(document['eye'])-allowed:
        raise ValueError('A perception profile cannot change the observer location, time, view or sampling.')
    from .eye import eye_config
    return eye_config({**deepcopy(eye),**deepcopy(document['eye'])})
