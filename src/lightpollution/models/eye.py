"""Blackwell/Crumey point-source threshold at the eye, not display brightness.

The spectrum at the observer already includes atmospheric extinction.
Reference: Crumey 2014, equations 5, 23, 26, 32, 48 and 53.
The observer factor here excludes atmospheric transmission explicitly.
"""

import numpy as np

MODEL = 'crumey2014_scotopic_at_eye_1'
R1, R2 = 6.505e-4, -8.461e-4
REFERENCE_SP = 1.408
BACKGROUND_FLOOR = 1e-5
SCOTOPIC_MAXIMUM = 3e-3
PHOTOPIC_V0_LUX = 2.54e-6
V22_LUMINANCE = PHOTOPIC_V0_LUX * (180*3600/np.pi)**2 * 10**(-.4*22)


def eye_config(value):
    if not isinstance(value, dict) or set(value)-{'observer_factor','unresolved_background_sp_ratio','adaptation','view','sampling','optics','display'}:
        raise ValueError('Unknown eye settings.')
    config = {'observer_factor':2.,'unresolved_background_sp_ratio':REFERENCE_SP,
              'adaptation':'complete', **value}
    for key, lower, upper in [('observer_factor',.5,10.),('unresolved_background_sp_ratio',.2,5.)]:
        x=config[key]
        if isinstance(x,bool) or not isinstance(x,(int,float)) or not np.isfinite(x) or not lower<=x<=upper:
            raise ValueError(f'{key} must be finite in [{lower},{upper}].')
        config[key]=float(x)
    if config['adaptation']!='complete':
        raise ValueError('This eye model requires completed adaptation; no time-course is fitted.')
    for name, defaults in [
            ('sampling', {'target_relative_error':.01, 'max_grid_size':5, 'validation':'none'}),
            ('optics', {'pupil_diameter_mm':6.}),
            ('display', {'white_cd_m2':100., 'black_cd_m2':.05, 'adaptation_cd_m2':20.,
                         'world_adaptation_scotopic_cd_m2':.002})]:
        supplied = config.get(name, {})
        if not isinstance(supplied, dict) or set(supplied)-set(defaults):
            raise ValueError(f'Unknown eye {name} settings.')
        config[name] = {**defaults, **supplied}
    numeric = [('sampling','target_relative_error',.001,.1), ('optics','pupil_diameter_mm',2,6),
        ('display','white_cd_m2',20,500), ('display','black_cd_m2',0,2),
        ('display','adaptation_cd_m2',2,100), ('display','world_adaptation_scotopic_cd_m2',1e-5,.004224)]
    for group,key,lo,hi in numeric:
        x=config[group][key]
        if isinstance(x,bool) or not isinstance(x,(int,float)) or not np.isfinite(x) or not lo<=x<=hi:
            raise ValueError(f'Eye {group}.{key} must be finite in [{lo},{hi}].')
        config[group][key]=float(x)
    if (type(config['sampling']['max_grid_size']) is not int or config['sampling']['max_grid_size'] not in (3,5,9)
            or config['sampling']['validation'] not in ('none','heldout_24')):
        raise ValueError('Eye sampling requires a grid limit 3/5/9 and validation none/heldout_24.')
    if not config['display']['black_cd_m2'] < config['display']['adaptation_cd_m2'] < config['display']['white_cd_m2']:
        raise ValueError('Display black < adaptation < white required.')
    if 'view' in config:
        view=config['view']
        if not isinstance(view,dict) or set(view)-{'azimuth_deg','altitude_deg','horizontal_fov_deg','width','height'}:
            raise ValueError('Unknown eye view settings.')
        view={'azimuth_deg':180.,'altitude_deg':50.,'horizontal_fov_deg':90.,'width':1800,'height':1100,**view}
        for key,lower,upper in [('azimuth_deg',0,360),('altitude_deg',10,90),
                ('horizontal_fov_deg',20,110),('width',640,2400),('height',480,1600)]:
            x=view[key]
            if isinstance(x,bool) or not isinstance(x,(int,float)) or not np.isfinite(x) or not lower<=x<=upper:
                raise ValueError(f'{key} outside supported eye-view range.')
            if key in ('width','height') and int(x)!=x:
                raise ValueError('Image dimensions must be integers.')
        config['view']=view
    return config


def point_threshold(background_equivalent_cd_m2, observer_factor=2.):
    """Required Blackwell-equivalent illuminance at the eye, in lux.

    Lower-background saturation follows the paper's 1e-5 cd/m2 cut-off.
    Caller must separately enforce the astronomical scotopic domain.
    """
    b=np.asarray(background_equivalent_cd_m2,dtype=float)
    if np.any(~np.isfinite(b)) or np.any(b<0) or not np.isfinite(observer_factor) or observer_factor<=0:
        raise ValueError('Finite nonnegative background and positive observer factor required.')
    if np.any(b>SCOTOPIC_MAXIMUM):
        raise ValueError('Background exceeds the adopted astronomical scotopic domain.')
    effective=np.maximum(b,BACKGROUND_FLOOR)
    return observer_factor*(R1*effective**.25+R2*effective**.5)**2


def evaluate_detection(photopic_background_cd_m2, scotopic_background_cd_m2,
                       stellar_scotopic_illuminance_lux, observer_factor=2.):
    """Spectral equivalence to Blackwell's 2850 K laboratory conditions.

    Missing inputs must be handled by the caller. No detection probability,
    pupil-area multiplier or second atmospheric attenuation is introduced.
    """
    p,s,e=np.broadcast_arrays(np.asarray(photopic_background_cd_m2,float),
        np.asarray(scotopic_background_cd_m2,float),np.asarray(stellar_scotopic_illuminance_lux,float))
    if any(np.any(~np.isfinite(x)) or np.any(x<0) for x in [p,s,e]):
        raise ValueError('Eye photometry must be finite and nonnegative.')
    b=s/REFERENCE_SP
    supported=(p<=SCOTOPIC_MAXIMUM)&(b<=SCOTOPIC_MAXIMUM)
    threshold=np.full(b.shape,np.nan)
    threshold[supported]=point_threshold(b[supported],observer_factor)
    equivalent=e/REFERENCE_SP
    ratio=equivalent/threshold
    with np.errstate(divide='ignore',invalid='ignore'):
        margin=2.5*np.log10(ratio)
    return {'supported':supported,'background_equivalent_cd_m2':b,
        'threshold_equivalent_illuminance_lux':threshold,
        'threshold_scotopic_illuminance_lux':threshold*REFERENCE_SP,
        'star_equivalent_illuminance_lux':equivalent,'flux_over_threshold':ratio,
        'margin_mag':margin,'above_threshold':supported&(ratio>=1),
        'background_floor_applied':b<BACKGROUND_FLOOR}
