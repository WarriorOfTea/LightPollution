"""Monochrome screen illustration of calculated eye thresholds.

The screen transfer is deliberately separate from the numerical eye model.
It does not claim calibrated subjective brightness, ocular PSF or star colour.
"""

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import matplotlib
import numpy as np

from ..adapters.storage import write_json
from ..application.eye import sample_eye_field
from ..models.eye import REFERENCE_SP, evaluate_detection
from ..skyview import camera_basis

DISPLAY_VERSION='monochrome_threshold_illustration_1'


def _unavailable(output, record):
    # Reusing a CLI output directory must not expose yesterday's valid PNG
    # as the current unsupported scene. Preserve it under a different name.
    path=output/'eye-view.png'
    if path.is_file():
        path.replace(output/'eye-view.previous.png')
        record['previous_image_retained_as']='eye-view.previous.png'
    write_json(output/'eye-display.json',record)
    return record


def _font(size):
    path=Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans.ttf'
    return ImageFont.truetype(str(path),size)


def projection(view):
    width,height=int(view['width']),int(view['height'])
    forward,right,up=camera_basis(view['azimuth_deg'],view['altitude_deg'])
    edge=np.tan(np.deg2rad(view['horizontal_fov_deg'])/2)
    x=(np.arange(width)+.5-width/2)*(2*edge/width)
    y=(height/2-np.arange(height)-.5)*(2*edge/width)
    rays=forward+x[None,:,None]*right+y[:,None,None]*up
    rays/=np.linalg.norm(rays,axis=-1)[...,None]
    az=np.rad2deg(np.arctan2(rays[...,0],rays[...,1]))%360
    alt=np.rad2deg(np.arcsin(rays[...,2]))
    return az,alt,(forward,right,up,edge)


def _srgb(linear):
    x=np.clip(linear,0,1)
    return np.where(x<=.0031308,12.92*x,1.055*x**(1/2.4)-.055)


def render_eye_view(result, output, *, title='НОЧНОЕ НЕБО · НЕВООРУЖЁННЫЙ ГЛАЗ'):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    eye=result['eye']
    view={'azimuth_deg':180.,'altitude_deg':50.,'horizontal_fov_deg':90.,'width':1800,'height':1100,
          **eye['config'].get('view',{})}
    if 'view' not in eye['config']:
        view['azimuth_deg'],view['altitude_deg']=result['request']['directions'][0]
    record={'schema_version':1,'display_version':DISPLAY_VERSION,'view':view,
        'image_is_photograph':False,'calibrated_eye_appearance':False,
        'colour_mode':'achromatic scotopic illustration; bright-star cone colour not modelled',
        'point_kernel':'Gaussian 3 arcmin FWHM with 0.55 px minimum sigma; illustrative, not measured ocular PSF',
        'brightness_transfer':'fixed logarithmic sky and asinh of positive signal above threshold, then sRGB; not a psychophysical law',
        'screen_reference_scotopic_cd_m2':2e-4*REFERENCE_SP,
        'viewing_assumptions':'no direct lamps or Moon; completed local adaptation, normal binocular/averted viewing'}
    if not eye['available']:
        record.update(available=False,reasons=eye['global_restrictions'])
        return _unavailable(output,record)
    az,alt,basis=projection(view)
    minimum=eye['field']['altitude_deg'][0]
    covered=alt>=minimum
    p,s,physical=sample_eye_field(eye['field'],az,np.clip(alt,minimum,90))
    detection=evaluate_detection(p,s,np.zeros_like(s),eye['config']['observer_factor'])
    supported=covered&physical&detection['supported']
    if result.get('terrain'):
        profile=result['terrain']
        horizon=np.interp(az,np.r_[profile['horizon_azimuth_deg'],360.],
            np.r_[profile['horizon_altitude_deg'],profile['horizon_altitude_deg'][0]])
        supported &= alt>horizon
    width,height=int(view['width']),int(view['height'])
    record.update(available=bool(np.any(supported)),field_pixels=width*height,
                  supported_pixels=int(supported.sum()),supported_fraction=float(supported.mean()))
    if not np.any(supported):
        record['reasons']=['No supported scotopic sightlines in this field.']
        return _unavailable(output,record)
    # These fixed display constants are the same in the real/city-off panels.
    image=.0006+.003*np.log1p(s/(2e-4*REFERENCE_SP))
    forward,right,up,edge=basis
    displayed=[];within=0;unavailable=0
    for row in eye['stars']:
        a,h=np.deg2rad([row['azimuth_deg'],row['altitude_deg']])
        ray=np.array([np.cos(h)*np.sin(a),np.cos(h)*np.cos(a),np.sin(h)])
        depth=float(ray@forward)
        if depth<=0:continue
        x,y=float(ray@right)/depth,float(ray@up)/depth
        px=(x/edge+1)*width/2-.5;py=height/2-y*width/(2*edge)-.5
        if not (0<=px<width and 0<=py<height):continue
        within+=1
        if not row['supported']:
            unavailable+=1;continue
        if not row['above_threshold'] or not supported[int(py),int(px)]:continue
        ratio=row['flux_over_threshold']
        # Angular-to-screen conversion, with an explicit display sampling floor.
        radians_per_pixel=2*edge/(width*math.sqrt(1+x*x+y*y))
        sigma=max(.55,np.deg2rad(3/60)/(2.35482*radians_per_pixel))
        radius=int(np.ceil(4*sigma))
        x0,x1=max(0,int(px)-radius),min(width,int(px)+radius+2)
        y0,y1=max(0,int(py)-radius),min(height,int(py)+radius+2)
        xx,yy=np.meshgrid(np.arange(x0,x1),np.arange(y0,y1))
        amplitude=.08*np.arcsinh(max(ratio-1,0))
        image[y0:y1,x0:x1]+=amplitude*np.exp(-((xx-px)**2+(yy-py)**2)/(2*sigma*sigma))
        displayed.append({'hr':row['hr'],'name':row['name'],'x_px':px,'y_px':py,
            'flux_over_threshold':ratio,'margin_mag':row['margin_mag']})
    pixels=np.uint8(np.round(255*_srgb(image)))
    # Unsupported patches are visibly marked; they do not masquerade as dark sky.
    yy,xx=np.indices(pixels.shape)
    pixels[~supported]=np.where(((xx[~supported]+yy[~supported])//12)%2,38,48)
    frame=Image.new('RGB',(width,height+170),(11,14,18))
    frame.paste(Image.fromarray(pixels).convert('RGB'),(0,80))
    draw=ImageDraw.Draw(frame)
    text_colour=(203,211,220)
    draw.text((28,15),title,font=_font(24),fill=text_colour)
    o=result['request']['observer']
    line=f"{o['latitude_deg']:.4f}° N · {o['longitude_deg']:.4f}° E · {result['request']['time'].replace('T',' ').replace('Z',' UTC')} · AGL {o['height_m']:g} м"
    draw.text((28,48),line,font=_font(17),fill=(144,156,170))
    limits=[r['reference_A0_observed_v_limit'] for r in eye['directions'] if r['supported']]
    detail=f"Азимут {view['azimuth_deg']:g}° · высота {view['altitude_deg']:g}° · поле {view['horizontal_fov_deg']:g}° · звёзд выше порога: {len(displayed)} / {within}"
    draw.text((28,height+94),detail,font=_font(18),fill=text_colour)
    draw.text((28,height+123),'Полная адаптация · монохромный режим · сценарная яркость неба · экранное отображение не калибровано',font=_font(17),fill=(144,156,170))
    if np.any(~supported):
        draw.text((width-390,94),'Штриховка: вне области модели',font=_font(16),fill=(185,185,185))
    frame.save(output/'eye-view.png')
    record.update(displayed_stars=len(displayed),catalogue_stars_in_view=within,
        unavailable_stellar_thresholds_in_view=unavailable,displayed_star_rows=displayed,
        reference_A0_limit_range=[min(limits),max(limits)] if limits else None)
    write_json(output/'eye-display.json',record)
    return record
