"""Linear monochrome detector: photons, electrons, independent noise and ADU.

No eye threshold, adaptation or tone operator enters these calculations.
The default response is an explicitly IDEAL V-energy-matched detector;
it needs no invented spectrum for a V-only diffuse background.
"""

from copy import deepcopy

import numpy as np

PLANCK=6.62607015e-34
LIGHT_SPEED=299792458.
MODEL='linear_monochrome_electron_camera_1'

PROFILE={
    'profile_id':'ideal-V-detector-1','origin':'synthetic',
    'sensor':{'width_px':960,'height_px':640,'pixel_pitch_um':3.76,
        'gain_e_per_adu':.8,'read_noise_e':3.,'dark_current_e_per_s':.02,
        'full_well_e':20000.,'bit_depth':16,'bias_adu':500.,'temperature_c':None},
    'optics':{'focal_length_mm':16.,'aperture_diameter_mm':8.,'throughput':.8,
        'psf_fwhm_arcsec':100.},
    'response':{'kind':'johnson_v_energy_matched','qe_scale':.65},
    'provenance':{}}


def _number(value,name,lo,hi,integer=False):
    if (isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value)
            or not lo<=value<=hi or integer and int(value)!=value):
        raise ValueError(f'{name} must be finite in [{lo},{hi}]'+(' and integer.' if integer else '.'))
    return int(value) if integer else float(value)


def camera_config(value):
    if not isinstance(value,dict) or set(value)-{'profile','exposure_s','seed','view','sampling',
            'unresolved_spectrum','preview','catalogue'}:
        raise ValueError('Unknown camera settings.')
    config={'exposure_s':30.,'seed':20261112,'unresolved_spectrum':'unavailable',**deepcopy(value)}
    supplied=config.get('profile',{})
    if not isinstance(supplied,dict) or set(supplied)-set(PROFILE):raise ValueError('Unknown camera profile fields.')
    profile={**deepcopy(PROFILE),**supplied}
    if not isinstance(profile['profile_id'],str) or not 1<=len(profile['profile_id'])<=160:
        raise ValueError('Camera profile needs a short nonempty ID.')
    if profile['origin'] not in ('synthetic','user_supplied','ekos_partial') or not isinstance(profile['provenance'],dict):
        raise ValueError('Camera origin must be synthetic, user_supplied or ekos_partial.')
    for group in ['sensor','optics']:
        supplied=profile[group]
        if not isinstance(supplied,dict) or set(supplied)-set(PROFILE[group]):raise ValueError(f'Unknown camera {group} fields.')
        profile[group]={**PROFILE[group],**supplied}
    ranges={
        'sensor':{'width_px':(64,2400,True),'height_px':(64,1600,True),'pixel_pitch_um':(.5,30),
            'gain_e_per_adu':(.01,100),'read_noise_e':(0,100),'dark_current_e_per_s':(0,1000),
            'full_well_e':(1,1e7),'bit_depth':(8,16,True),'bias_adu':(0,65535)},
        'optics':{'focal_length_mm':(8,5000),'aperture_diameter_mm':(1,2000),'throughput':(0,1),
            'psf_fwhm_arcsec':(.1,1000)}}
    for group,fields in ranges.items():
        for key,args in fields.items():profile[group][key]=_number(profile[group][key],key,*args)
    sensor=profile['sensor'];optics=profile['optics']
    if sensor['width_px']*sensor['height_px']>2_000_000:raise ValueError('Camera supports at most 2 million pixels.')
    if sensor['bias_adu']>=2**sensor['bit_depth']-1:raise ValueError('Bias must leave a nonzero ADC range.')
    if sensor['temperature_c'] is not None:sensor['temperature_c']=_number(sensor['temperature_c'],'temperature_c',-100,60)
    if optics['aperture_diameter_mm']>optics['focal_length_mm']:raise ValueError('This camera approximation requires f-number >=1.')
    response=profile['response']
    if not isinstance(response,dict):raise ValueError('A camera response object is required.')
    if response.get('kind')=='johnson_v_energy_matched':
        if set(response)-{'kind','qe_scale'}:raise ValueError('Unknown V-matched response fields.')
        response['qe_scale']=_number(response.get('qe_scale',.65),'qe_scale',0,1)
    elif response.get('kind')=='tabulated':
        if set(response)!={'kind','wavelength_nm','qe','transmission'}:raise ValueError('Tabulated response needs axis, QE and transmission.')
        try:axis,qe,trans=(np.asarray(response[k],float) for k in ['wavelength_nm','qe','transmission'])
        except (ValueError,TypeError):raise ValueError('Invalid camera response arrays.') from None
        if (axis.ndim!=1 or not 2<=axis.size<=1000 or qe.shape!=axis.shape or trans.shape!=axis.shape
                or any(np.any(~np.isfinite(a)) for a in [axis,qe,trans]) or np.any(np.diff(axis)<=0)
                or axis[0]!=360 or axis[-1]!=830 or np.any((qe<0)|(qe>1)) or np.any((trans<0)|(trans>1))):
            raise ValueError('Response must cover 360–830 nm, increasing; QE/transmission in [0,1].')
        effective=qe*trans
        if effective.max()>0 and np.trapezoid(effective,x=axis)/effective.max()<20:
            raise ValueError('Narrow passbands below20nm effective width need a finer spectral model; unsupported here.')
    else:raise ValueError('Response kind must be johnson_v_energy_matched or tabulated.')
    config['profile']=profile
    config['exposure_s']=_number(config['exposure_s'],'exposure_s',.001,120)
    config['seed']=_number(config['seed'],'seed',0,2**32-1,True)
    if config['unresolved_spectrum'] not in ('unavailable','flat_energy_V_normalized'):
        raise ValueError('Unknown unresolved-background spectrum policy.')
    for key,defaults in [('view',{'azimuth_deg':180.,'altitude_deg':50.,'rotation_deg':0.}),
            ('sampling',{'target_relative_error':.01,'max_grid_size':5,'validation':'none'}),
            ('preview',{'black_adu':sensor['bias_adu'],
                        'white_adu':min(2**sensor['bit_depth']-1,sensor['bias_adu']+5500.),'asinh_scale':10.})]:
        supplied=config.get(key,{})
        if not isinstance(supplied,dict) or set(supplied)-set(defaults):raise ValueError(f'Unknown camera {key} settings.')
        config[key]={**defaults,**supplied}
    for key,lo,hi in [('azimuth_deg',0,360),('altitude_deg',10,90),('rotation_deg',-180,180)]:
        config['view'][key]=_number(config['view'][key],key,lo,hi)
    sampling=config['sampling'];sampling['target_relative_error']=_number(sampling['target_relative_error'],'target_relative_error',.001,.1)
    if type(sampling['max_grid_size']) is not int or sampling['max_grid_size'] not in (3,5,9) or sampling['validation'] not in ('none','heldout_8'):
        raise ValueError('Camera sampling supports grids 3/5/9 and none/heldout_8 validation.')
    for key,lo,hi in [('black_adu',0,65535),('white_adu',1,65535),('asinh_scale',.01,1000)]:
        config['preview'][key]=_number(config['preview'][key],key,lo,hi)
    if config['preview']['black_adu']>=config['preview']['white_adu']:raise ValueError('Preview black must be below white.')
    if 'catalogue' in config:validate_catalogue(config['catalogue'])
    return config


def validate_catalogue(catalogue):
    if (not isinstance(catalogue,dict) or set(catalogue)!={'provider','utc','magnitude_limit','truncated','rows'}
            or catalogue['provider']!='KStars' or not isinstance(catalogue['utc'],str)
            or type(catalogue['truncated']) is not bool or not isinstance(catalogue['rows'],list)
            or len(catalogue['rows'])>20000):raise ValueError('Invalid KStars camera catalogue.')
    _number(catalogue['magnitude_limit'],'magnitude_limit',0,16)
    ids=set()
    for row in catalogue['rows']:
        if (not isinstance(row,dict) or set(row)!={'id','name','azimuth_deg','altitude_deg','v_mag','b_v'}
                or not isinstance(row['id'],str) or not 1<=len(row['id'])<=160 or row['id'] in ids
                or not isinstance(row['name'],str) or len(row['name'])>160):raise ValueError('Invalid or duplicate camera star.')
        ids.add(row['id'])
        for key,lo,hi in [('azimuth_deg',0,360),('altitude_deg',-90,90),('v_mag',-5,25)]:_number(row[key],key,lo,hi)
        if row['b_v'] is not None:_number(row['b_v'],'b_v',-1,4)


def camera_view(config):
    sensor=config['profile']['sensor'];optics=config['profile']['optics']
    angle=np.rad2deg(2*np.arctan(sensor['width_px']*sensor['pixel_pitch_um']*.001/(2*optics['focal_length_mm'])))
    if angle>60:raise ValueError('First camera mode supports horizontal fields <=60 degrees.')
    return {**config['view'],'width':sensor['width_px'],'height':sensor['height_px'],'horizontal_fov_deg':float(angle)}


def response_weights(profile,library):
    axis=library.wavelengths;response=profile['response'];v=library.responses['johnson_v']
    if response['kind']=='johnson_v_energy_matched':
        shape=v*550/axis;qe=response['qe_scale']*shape/shape.max();trans=np.ones_like(axis)
        # This exact proportionality eliminates dependence on unknown SED shape.
        v_factor=response['qe_scale']*550e-9/(shape.max()*PLANCK*LIGHT_SPEED)
    else:
        qe=np.interp(axis,response['wavelength_nm'],response['qe'])
        trans=np.interp(axis,response['wavelength_nm'],response['transmission']);v_factor=None
    weights=qe*trans*(axis*1e-9/(PLANCK*LIGHT_SPEED))
    return weights,v_factor


def sample_detector(star_e,sky_e,config):
    sensor=config['profile']['sensor'];t=config['exposure_s']
    dark=np.full_like(sky_e,sensor['dark_current_e_per_s']*t,dtype=float)
    mean=np.maximum(0,np.asarray(star_e)+np.asarray(sky_e)+dark)
    variance=mean+sensor['read_noise_e']**2
    rng=np.random.default_rng(config['seed'])
    charge=rng.poisson(mean).astype(float)
    fullwell=charge>=sensor['full_well_e']
    clipped=np.minimum(charge,sensor['full_well_e'])
    signal=clipped+rng.normal(0,sensor['read_noise_e'],mean.shape)
    raw=np.rint(signal/sensor['gain_e_per_adu']+sensor['bias_adu'])
    maximum=2**sensor['bit_depth']-1
    return {'adu':np.clip(raw,0,maximum).astype(np.uint16),'expected_e':mean,
        'variance_e2_before_clipping':variance,'dark_e':dark,'fullwell_mask':fullwell,
        'adc_clipped_mask':(raw<0)|(raw>maximum)}


def read_noise_lower_bound(rate_e_per_s,read_noise_e,tolerance_percent=5.):
    """Same fractional noise criterion as Ekos, using a direct electron rate.

    sqrt((b*t+RN²)/(b*t)) <= 1+tolerance/100. This is a lower bound,
    NOT an optimum including object signal, saturation or guiding.
    """
    rate=np.asarray(rate_e_per_s,float)
    if np.any(~np.isfinite(rate)) or np.any(rate<0) or not np.isfinite(read_noise_e) or read_noise_e<0:
        raise ValueError('Finite nonnegative rate/read noise required.')
    _number(tolerance_percent,'tolerance_percent',.01,100)
    factor=(1+tolerance_percent/100)**2-1
    return np.divide(read_noise_e**2,rate*factor,out=np.full_like(rate,np.inf),where=rate>0)
