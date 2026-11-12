"""Numerical eye-view exports: physical flux, explicit display, bounded encoding."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, PngImagePlugin
import matplotlib
import numpy as np

from ..adapters.storage import write_json
from ..adapters.png16 import write_rgb16
from ..application.eye import sample_eye_field
from ..application.eye_appearance import ensure_eye_appearance,export_eye_colour
from ..models.eye import evaluate_detection, eye_config
from ..models.eye_display import (VERSION as TONE_VERSION, DITHER_SEED, display_scale,
    quantize, srgb_encode, tone_linear)
from ..models.eye_optics import VERSION as OPTICS_VERSION, deposit_star, optical_mixture, pixel_solid_angles
from ..models.eye_colour import RGB_LUMINANCE,gamut_preserving_luminance
from ..models.eye_profiles import display_geometry
from ..skyview import camera_basis

DISPLAY_VERSION = 'flux_integrated_achromatic_eye_2'
COLOUR_DISPLAY_VERSION = 'experimental_spectral_star_colour_3'
IMAGE_EXPORTS = ('eye-view.png', 'eye-preview.png', 'eye-mobile.png', 'eye-view-16bit.png', 'eye-linear.npz','eye-colour.csv')


def _unavailable(output, record):
    for name in IMAGE_EXPORTS:
        path = output/name
        if path.is_file():
            previous = path.with_name(path.stem+'.previous'+path.suffix)
            path.replace(previous)
            if name == 'eye-view.png':
                record['previous_image_retained_as'] = previous.name
    write_json(output/'eye-display.json', record)
    return record


def _font(size):
    path = Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans.ttf'
    return ImageFont.truetype(str(path), size)


def projection(view):
    width, height = int(view['width']), int(view['height'])
    forward, right, up = camera_basis(view['azimuth_deg'], view['altitude_deg'])
    edge = np.tan(np.deg2rad(view['horizontal_fov_deg'])/2)
    x = (np.arange(width)+.5-width/2)*(2*edge/width)
    y = (height/2-np.arange(height)-.5)*(2*edge/width)
    rays = forward+x[None, :, None]*right+y[:, None, None]*up
    rays /= np.linalg.norm(rays, axis=-1)[..., None]
    az = np.rad2deg(np.arctan2(rays[..., 0], rays[..., 1])) % 360
    alt = np.rad2deg(np.arcsin(rays[..., 2]))
    return az, alt, (forward, right, up, edge)


# Retained as a pure encoding helper for the frozen historical 0.6 audit.
_srgb = srgb_encode


def render_linear_field(result, view):
    """Re-integrate sources at the requested pixel scale, BEFORE tone encoding."""
    eye = result['eye']
    config = eye_config(eye['config'])
    az, alt, basis = projection(view)
    minimum = eye['field']['altitude_deg'][0]
    p, s, physical = sample_eye_field(eye['field'], az, np.clip(alt, minimum, 90))
    domain = evaluate_detection(p, s, np.zeros_like(s), config['observer_factor'])['supported']
    from ..models.kstars_horizon import horizon_visible
    supported = (alt >= minimum) & physical & domain
    supported &= horizon_visible(result.get('integration',{}).get('artificial_horizon'),az,alt)
    if result.get('terrain'):
        terrain = result['terrain']
        horizon = np.interp(az, np.r_[terrain['horizon_azimuth_deg'], 360.],
            np.r_[terrain['horizon_altitude_deg'], terrain['horizon_altitude_deg'][0]])
        supported &= alt > horizon
    width, height = int(view['width']), int(view['height'])
    omega = pixel_solid_angles(view)
    physical_flux = np.zeros((height, width))
    appearance_flux = np.zeros_like(physical_flux)
    appearance=ensure_eye_appearance(result)
    colour_rows={r['hr']:r for r in appearance['rows']} if appearance else {}
    colour_flux=np.zeros((height,width,3)) if appearance else None
    mixture = optical_mixture(config['optics']['pupil_diameter_mm'])
    forward, right, up, edge = basis
    step = 2*edge/width
    displayed, ledgers = [], []
    within = unavailable = 0
    for row in eye['stars']:
        if not bool(horizon_visible(result.get('integration',{}).get('artificial_horizon'),
                                    row['azimuth_deg'],row['altitude_deg'])):
            continue
        a, h = np.deg2rad([row['azimuth_deg'], row['altitude_deg']])
        ray = np.array([np.cos(h)*np.sin(a), np.cos(h)*np.cos(a), np.sin(h)])
        depth = float(ray@forward)
        if depth <= 0:
            continue
        x, y = float(ray@right)/depth, float(ray@up)/depth
        wing_angle = np.deg2rad(6*max(mixture['sigma_deg']))
        if (abs(np.arctan(x)) > np.arctan(edge)+wing_angle or
                abs(np.arctan(y)) > np.arctan(edge*height/width)+wing_angle):
            continue
        px, py = (x/edge+1)*width/2-.5, height/2-y*width/(2*edge)-.5
        # Projection Jacobian: covariance = sigma^2 (1+r^2)(I+xx^T).
        scales = step/np.sqrt((1+x*x+y*y)*np.array([1+x*x, 1+y*y]))
        correlation = -x*y/np.sqrt((1+x*x)*(1+y*y))  # pixel y runs downward
        inside = -.5 <= px < width-.5 and -.5 <= py < height-.5
        if inside:
            within += 1
            unavailable += not row['supported']
        # Include the wings of stars just outside the camera boundary.
        margin = 6*np.deg2rad(max(mixture['sigma_deg']))/min(scales)
        if not (-margin <= px < width+margin and -margin <= py < height+margin):
            continue
        irradiance = row.get('scotopic_illuminance_lux')
        if irradiance is None:
            continue
        valid_pixel = inside and supported[min(height-1, max(0, int(py+.5))), min(width-1, max(0, int(px+.5)))]
        visible = bool(row['above_threshold'] and row['supported'] and valid_pixel)
        colour=colour_rows.get(row['hr'],{})
        unit_rgb=colour.get('appearance_unit_Y_rgb') or [1.,1.,1.]
        ledger = deposit_star(physical_flux, px, py, irradiance, scales, mixture,
            correlation=correlation, appearance_buffer=appearance_flux if visible else None,
            colour_buffer=colour_flux if visible else None,unit_Y_rgb=unit_rgb)
        ledgers.append({'hr': row['hr'], 'centre_in_field': bool(inside), 'appearance_included': visible, **ledger})
        if visible:
            displayed.append({'hr': row['hr'], 'name': row['name'], 'x_px': px, 'y_px': py,
                'flux_over_threshold': row['flux_over_threshold'], 'margin_mag': row['margin_mag'],
                'scotopic_illuminance_lux': irradiance})
    background = np.asarray(s, dtype=np.float32)
    scene = (background+appearance_flux/omega).astype(np.float32)
    record={'background_scotopic_cd_m2': background, 'photopic_background_cd_m2': np.asarray(p, np.float32),
        'stellar_physical_pixel_illuminance_lux': physical_flux.astype(np.float32),
        'stellar_appearance_pixel_illuminance_lux': appearance_flux.astype(np.float32),
        'appearance_scotopic_cd_m2': scene, 'pixel_solid_angle_sr': omega.astype(np.float32),
        'supported': supported, 'displayed_star_rows': displayed, 'flux_ledger': ledgers,
        'catalogue_stars_in_view': within, 'unavailable_stellar_thresholds_in_view': int(unavailable),
        'optics': mixture}
    if colour_flux is not None:
        record['appearance_rgb_unitY_carrier']=(background[...,None]+colour_flux/omega[...,None]).astype(np.float32)
    return record


def encode_field(arrays,profile):
    target=tone_linear(arrays['appearance_scotopic_cd_m2'],profile)
    if 'appearance_rgb_unitY_carrier' not in arrays:
        return srgb_encode(target),{'colour_mode':'achromatic','gamut_compressed_pixels':0}
    carrier=np.asarray(arrays['appearance_rgb_unitY_carrier'],dtype=float)
    carrier_y=carrier@RGB_LUMINANCE
    ratio=np.divide(target,carrier_y,out=np.zeros_like(target),where=carrier_y>0)
    linear,factors=gamut_preserving_luminance(carrier*ratio[...,None])
    return srgb_encode(linear),{'colour_mode':'experimental_stellar_colour',
        'gamut_compressed_pixels':int(np.count_nonzero(factors<1-1e-10)),
        'maximum_display_Y_error':float(np.max(abs(linear@RGB_LUMINANCE-target)))}


def _mark_unsupported(codes,supported,step):
    yy,xx=np.indices(supported.shape)
    marked=np.where(((xx[~supported]+yy[~supported])//step)%2,38,48)
    codes[~supported]=marked[:,None] if codes.ndim==3 else marked


def _png_info():
    info = PngImagePlugin.PngInfo()
    info.add(b'sRGB', b'\x00')
    info.add_text('Description', 'Numerical eye model; sRGB encoding; no calibrated human brightness or colour match')
    return info


def _frame(pixels, result, view, displayed, title, *, scale=1.):
    height, width = pixels.shape[:2]
    header, footer = round(80*scale), round(90*scale)
    frame = Image.new('RGB', (width, height+header+footer), (10, 13, 18))
    frame.paste(Image.fromarray(pixels).convert('RGB'), (0, header))
    draw = ImageDraw.Draw(frame)
    def text(x, y, value, size, colour=(203, 211, 220)):
        draw.text((round(x*scale), round(y*scale)), value, font=_font(max(7, round(size*scale))), fill=colour)
    text(28, 15, title, 24)
    observer = result['request']['observer']
    line = (f"{observer['latitude_deg']:.4f}° N · {observer['longitude_deg']:.4f}° E · "
            f"{result['request']['time'].replace('T',' ').replace('Z',' UTC')} · AGL {observer['height_m']:g} м")
    text(28, 48, line, 17, (144, 156, 170))
    text(28, (height+header)/scale+14,
         f"Азимут {view['azimuth_deg']:g}° · высота {view['altitude_deg']:g}° · поле {view['horizontal_fov_deg']:g}° · звёзд выше порога: {displayed}", 18)
    mode=result['eye']['config'].get('appearance',{}).get('mode','monochrome')
    line=('Модель потока и оптики · экспериментальный цвет звёзд · восприятие не калибровано'
          if mode!='monochrome' else
          'Модель потока и оптики · монохромный режим · экранная яркость и восприятие не калиброваны')
    text(28, (height+header)/scale+43,line,17,(144,156,170))
    return frame


def render_eye_view(result, output, *, title='НОЧНОЕ НЕБО · МОДЕЛЬ НЕВООРУЖЁННОГО ГЛАЗА'):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    eye = result['eye']
    config = eye_config(eye['config'])
    view = {'azimuth_deg': 180., 'altitude_deg': 50., 'horizontal_fov_deg': 90., 'width': 1800, 'height': 1100,
            **config.get('view', {})}
    if 'view' not in config:
        view['azimuth_deg'], view['altitude_deg'] = result['request']['directions'][0]
    experimental=config['appearance']['mode']!='monochrome'
    record = {'schema_version': 3, 'display_version': COLOUR_DISPLAY_VERSION if experimental else DISPLAY_VERSION, 'view': view,
        'image_is_photograph': False, 'calibrated_eye_appearance': False,
        'colour_mode': 'experimental_stellar_colour' if experimental else 'achromatic',
        'point_kernel': OPTICS_VERSION, 'brightness_transfer': TONE_VERSION,
        'display_profile': config['display'], 'display_scale': display_scale(config['display']),
        'dither_seed': DITHER_SEED, 'dither_bound_code_8bit': 1., 'rounding_bound_code_16bit': .5,
        'encoding': 'exactly one sRGB transfer; dither only at final 8-bit quantization',
        'viewing_assumptions': 'no direct lamps or Moon; completed local adaptation, binocular/averted viewing',
        'subthreshold_physical_flux_retained_in_npz': True,
        'appearance_gate': 'Crumey total at-eye threshold once, before tone; no probability or calibrated suprathreshold response',
        'unresolved_starlight': 'diffuse residual excludes catalogue point sources by convention; source separation not independently measured',
        'unavailable_spectra': 'not replaced with invented flux',
        'camera_or_glare_model': False}
    record['observer_profile']=config['observer_profile']
    record['display_geometry']=display_geometry(config['display'],view['horizontal_fov_deg'])
    if not eye['available']:
        record.update(available=False, reasons=eye['global_restrictions'])
        return _unavailable(output, record)
    arrays = render_linear_field(result, view)
    supported = arrays['supported']
    record.update(available=bool(np.any(supported)), field_pixels=int(supported.size),
        supported_pixels=int(supported.sum()), supported_fraction=float(supported.mean()))
    if not np.any(supported):
        record['reasons'] = ['No supported scotopic sightlines in this field.']
        return _unavailable(output, record)
    encoded,colour_summary=encode_field(arrays,config['display'])
    record['appearance_encoding']=colour_summary
    pixels = quantize(encoded)
    _mark_unsupported(pixels,supported,12)
    _frame(pixels, result, view, len(arrays['displayed_star_rows']), title).save(output/'eye-view.png', pnginfo=_png_info())
    # Unannotated 16-bit display export; physical values are in the NPZ, not here.
    quantized16 = quantize(encoded, 16, dither=False)
    quantized16[~supported] = 0
    if quantized16.ndim==3:write_rgb16(output/'eye-view-16bit.png',quantized16)
    else:Image.fromarray(quantized16).save(output/'eye-view-16bit.png', pnginfo=_png_info())
    np.savez_compressed(output/'eye-linear.npz', **{k: v for k, v in arrays.items() if isinstance(v, np.ndarray)})
    record['previews'] = []
    for width, filename in [(900, 'eye-preview.png'), (390, 'eye-mobile.png')]:
        preview = {**view, 'width': width, 'height': round(view['height']*width/view['width'])}
        field = render_linear_field(result, preview)
        preview_encoded,preview_summary=encode_field(field,config['display'])
        codes = quantize(preview_encoded)
        _mark_unsupported(codes,field['supported'],6)
        # At 390 px the field is unannotated; UI/chat supplies its provenance.
        image = (_frame(codes, result, preview, len(field['displayed_star_rows']), title, scale=width/view['width'])
                 if width == 900 else Image.fromarray(codes).convert('RGB'))
        image.save(output/filename, pnginfo=_png_info())
        record['previews'].append({'file': filename, 'view': preview, 'reintegrated_before_tone': True,
            'displayed_stars': len(field['displayed_star_rows']),'appearance_encoding':preview_summary,
            'physical_deposited_scotopic_lux': float(field['stellar_physical_pixel_illuminance_lux'].sum(dtype=np.float64))})
    limits = [r['reference_A0_observed_v_limit'] for r in eye['directions'] if r['supported']]
    record.update(displayed_stars=len(arrays['displayed_star_rows']),
        catalogue_stars_in_view=arrays['catalogue_stars_in_view'],
        unavailable_stellar_thresholds_in_view=arrays['unavailable_stellar_thresholds_in_view'],
        displayed_star_rows=arrays['displayed_star_rows'], flux_ledger=arrays['flux_ledger'], optics=arrays['optics'],
        field_crop_offset_y_px=80, field_pixel_max_code=int(pixels[supported].max()),
        saturated_field_pixels=int(np.count_nonzero(
            (np.any(pixels == 255,axis=-1) if pixels.ndim==3 else pixels == 255) & supported)),
        reference_A0_limit_range=[min(limits), max(limits)] if limits else None)
    if eye.get('appearance'):
        record['colour_model']={k:v for k,v in eye['appearance'].items() if k!='rows'}
        export_eye_colour(eye,output)
    elif (output/'eye-colour.csv').is_file():
        (output/'eye-colour.csv').replace(output/'eye-colour.previous.csv')
    write_json(output/'eye-display.json', record)
    return record
