"""Explicit achromatic display contract, independent of the eye threshold.

Ferwerda 1996 rod-to-photopic TVI scaling is followed by a documented smooth
engineering shoulder and one sRGB encoding. This is not a validated brightness
match. The Ward 1997 comparator implements only its histogram/TVI core.
"""

import numpy as np

VERSION = 'ferwerda1996_scotopic_fixed_profile_shoulder_1'
DITHER_SEED = 37013


def tvi_threshold(adaptation_cd_m2, *, scotopic):
    value = np.asarray(adaptation_cd_m2, dtype=float)
    if np.any(~np.isfinite(value)) or np.any(value <= 0):
        raise ValueError('Positive finite adaptation luminance required.')
    x = np.log10(value)
    if scotopic:
        exponent = np.where(x <= -3.94, -2.86, np.where(x >= -1.44, x-.395,
            np.maximum(0, .405*x+1.6)**2.18-2.86))
    else:
        exponent = np.where(x <= -2.6, -.72, np.where(x >= 1.9, x-1.255,
            np.maximum(0, .249*x+.65)**2.7-.72))
    return 10**exponent


def display_scale(profile):
    world = profile['world_adaptation_scotopic_cd_m2']
    screen = profile['adaptation_cd_m2']
    return float(tvi_threshold(screen, scotopic=False)/tvi_threshold(world, scotopic=True))


def tone_linear(luminance_scotopic_cd_m2, profile):
    luminance = np.asarray(luminance_scotopic_cd_m2, dtype=float)
    if np.any(~np.isfinite(luminance)) or np.any(luminance < 0):
        raise ValueError('Finite nonnegative scene luminance required.')
    scale = display_scale(profile)
    span = profile['white_cd_m2']-profile['black_cd_m2']
    # Low-signal derivative scale is preserved; asymptote avoids hard clipping.
    return -np.expm1(-scale*luminance/span)


def srgb_encode(linear):
    x = np.clip(linear, 0, 1)
    return np.where(x <= .0031308, 12.92*x, 1.055*x**(1/2.4)-.055)


def quantize(encoded, bits=8, *, dither=True, seed=DITHER_SEED):
    if bits not in (8, 16):
        raise ValueError('Only 8- and 16-bit PNG exports are supported.')
    scaled = np.clip(np.asarray(encoded, dtype=float), 0, 1)*(2**bits-1)
    if bits == 8 and dither:
        floor = np.floor(scaled)
        shape=(*scaled.shape[:-1],1) if scaled.ndim==3 and scaled.shape[-1]==3 else scaled.shape
        uniform = np.random.default_rng(seed).random(shape)
        codes = floor+(uniform < scaled-floor)
    else:
        codes = np.rint(scaled)
    return codes.astype(np.uint8 if bits == 8 else np.uint16)


def ward1997_curve(adaptation_samples_cd_m2, profile, *, bins=100):
    """Ward Larson et al. 1997 §§4.2–4.5, histogram/TVI core comparator.

    Callers must supply foveal (~1 degree) means, not isolated pixel peaks.
    Glare, acuity and the paper's XYZ colour transform are deliberately absent.
    This is an independently implemented comparator, not a claim of pcond parity.
    """
    samples = np.asarray(adaptation_samples_cd_m2, dtype=float)
    if np.any(~np.isfinite(samples)) or np.any(samples <= 0):
        raise ValueError('Positive finite foveal adaptation samples required.')
    lower, upper = max(1e-4, float(samples.min())), float(samples.max())
    black, white = max(1e-4, profile['black_cd_m2']), profile['white_cd_m2']
    # §4.4: linear fallback when the scene range already fits the display.
    if upper <= lower or np.log(upper/lower) <= np.log(white/black):
        return {'version':'ward1997_histogram_tvi_core_1','mode':'linear_range_fits',
                'scale':white/upper,'foveal_samples':int(samples.size), 'iterations':0,
                'complete_pcond_implementation':False}
    histogram, edges = np.histogram(np.log(np.maximum(samples, lower)), bins=bins,
                                   range=(np.log(lower), np.log(upper)))
    counts = histogram.astype(float)
    centres = (edges[:-1]+edges[1:])/2
    world = np.exp(centres)
    world_tvi = np.where(np.log10(world) < -.0184,
        tvi_threshold(world, scotopic=True), tvi_threshold(world, scotopic=False))
    span, step = np.log(white/black), edges[1]-edges[0]
    tolerance = .025*counts.sum()
    converged = False
    for iteration in range(1000):
        total = counts.sum()
        if total < tolerance:
            return {'version':'ward1997_histogram_tvi_core_1','mode':'linear_histogram_depleted',
                    'scale':white/upper,'foveal_samples':int(samples.size),'iterations':iteration,
                    'complete_pcond_implementation':False}
        cumulative = (np.cumsum(counts)-.5*counts)/total
        display = np.exp(np.log(black)+span*cumulative)
        display_tvi = np.where(np.log10(display) < -.0184,
            tvi_threshold(display, scotopic=True), tvi_threshold(display, scotopic=False))
        ceiling = (display_tvi/world_tvi)*total*step*world/(span*display)
        clipped = np.minimum(counts, ceiling)
        removed = float(np.sum(counts-clipped))
        counts = clipped
        if removed <= tolerance:
            converged = True
            break
    cdf = np.r_[0., np.cumsum(counts)/counts.sum()]
    return {'version':'ward1997_histogram_tvi_core_1','mode':'histogram_tvi',
        'log_world_edges':edges.tolist(), 'log_display_values':(np.log(black)+span*cdf).tolist(),
        'iterations':iteration+1,'converged':converged,'foveal_samples':int(samples.size),
        'complete_pcond_implementation':False}


def apply_ward1997(luminance, curve, profile):
    values = np.maximum(np.asarray(luminance, dtype=float), 1e-30)
    if curve['mode'].startswith('linear'):
        display = values*curve['scale']
    else:
        display = np.exp(np.interp(np.log(values), curve['log_world_edges'], curve['log_display_values']))
    return np.clip((display-profile['black_cd_m2'])/(profile['white_cd_m2']-profile['black_cd_m2']),0,1)
