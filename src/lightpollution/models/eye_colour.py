"""Spectral colour and explicitly experimental low-light appearance controls.

XYZ1931, LMS2006 and scotopic response are separate quantities. The Kirk
operator below accepts paper-normalized input ONLY; no absolute calibration
of its q units from our radiometric data is asserted.
"""

import numpy as np

from .eye_display import tvi_threshold

# Derived from sRGB primaries xy and D65 xy, as in W3C CSS Color4 Appendix A.
RGB_TO_XYZ = np.array([[.4123907992659595,.357584339383878,.1804807884018343],
    [.2126390058715104,.715168678767756,.0721923153607337],
    [.0193308187155919,.119194779794626,.9505321522496606]])
XYZ_TO_RGB = np.linalg.inv(RGB_TO_XYZ)
RGB_LUMINANCE = RGB_TO_XYZ[1]
VERSION = 'ferwerda_star_colour_control_1'


def xyz_unit_rgb(xyz):
    values = np.asarray(xyz,dtype=float)
    if values.shape[-1]!=3 or np.any(~np.isfinite(values)) or np.any(values<0) or np.any(values[...,1]<=0):
        raise ValueError('Finite nonnegative XYZ with positive Y required.')
    # Unit display Y. This does not rescale the physical source irradiance.
    return (values/values[...,1,None])@XYZ_TO_RGB.T


def gamut_preserving_luminance(rgb):
    """Compress along a ray to neutral at fixed Y, never clip channels alone."""
    values = np.asarray(rgb,dtype=float)
    if values.shape[-1]!=3 or np.any(~np.isfinite(values)):
        raise ValueError('Finite linear sRGB triplets required.')
    luminance = values@RGB_LUMINANCE
    if np.any(luminance < -1e-10) or np.any(luminance > 1+1e-10):
        raise ValueError('Display luminance must already be in [0,1].')
    luminance=np.clip(luminance,0,1)
    delta=values-luminance[...,None]
    with np.errstate(divide='ignore',invalid='ignore'):
        bounds=np.where(delta>0,(1-luminance[...,None])/delta,
            np.where(delta<0,-luminance[...,None]/delta,1.))
    factor=np.clip(np.min(bounds,axis=-1),0,1)
    mapped=luminance[...,None]+factor[...,None]*delta
    # Only roundoff correction after common chroma compression.
    return np.clip(mapped,0,1),factor


def encircled_pool_gain(mixture,diameter_arcmin):
    """Mean optical signal in a fixed ANGULAR pool, independent of image pixels."""
    if not np.isfinite(diameter_arcmin) or not .5<=diameter_arcmin<=30:
        raise ValueError('Colour integration diameter must be within .5–30 arcmin.')
    radius=np.deg2rad(diameter_arcmin/120)
    sigma=np.deg2rad(np.asarray(mixture['sigma_deg']))
    enclosed=np.sum(np.asarray(mixture['weight'])*(-np.expm1(-radius*radius/(2*sigma*sigma))))
    return float(enclosed/(np.pi*radius*radius))


def cone_colour_weight(photopic_background,scotopic_background,photopic_star_lux,scotopic_star_lux,
                       pool_gain):
    """Ferwerda TVI control + explicit engineering colour-detection gate.

    This gate is NOT a published star-colour threshold. Its pool size is an
    observer scenario pending visual calibration. Crumey detection is untouched.
    """
    p,s,ep,es=np.broadcast_arrays(*[np.asarray(v,float) for v in
        [photopic_background,scotopic_background,photopic_star_lux,scotopic_star_lux]])
    if any(np.any(~np.isfinite(x)) or np.any(x<0) for x in [p,s,ep,es]) or not np.isfinite(pool_gain) or pool_gain<=0:
        raise ValueError('Finite nonnegative photometry and positive pool gain required.')
    local_p=np.maximum(1e-8,p+ep*pool_gain)
    local_s=np.maximum(1e-8,s+es*pool_gain)
    tp=tvi_threshold(local_p,scotopic=False)
    ts=tvi_threshold(local_s,scotopic=True)
    # Explicit log-linear interpolation over the paper's .01–10 mesopic range.
    rods=np.clip((1-np.log10(local_s))/3,0,1)
    cone=ep/tp
    rod=rods*es/ts
    fraction=np.divide(cone,cone+rod,out=np.zeros_like(cone),where=(cone+rod)>0)
    signal=ep*pool_gain/tp
    gate=-np.expm1(-signal*signal)
    return np.clip(fraction*gate,0,1)


def kirk2011_normalized_lmsr(q,*,input_convention):
    """Equations10–13, fixed fully adapted constants, research-only q space.

    The .637/.392/1.606 denominators must not be silently applied to peak-1
    CIE2006 integrals. This function deliberately has no radiometric adapter.
    """
    if input_convention!='paper_normalized':
        raise ValueError('Kirk prototype requires explicitly paper-normalized LMSR; physical bridge uncalibrated.')
    q=np.asarray(q,dtype=float)
    if q.shape[-1]!=4 or np.any(~np.isfinite(q)) or np.any(q<0):
        raise ValueError('Nonnegative finite LMSR quadruplets required.')
    l,m,s,r=np.moveaxis(q,-1,0)
    gl=(1+.33*(l+.25*r))**-.5/.637
    gm=(1+.33*(m+.25*r))**-.5/.392
    gs=(1+.33*(s+.4*r))**-.5/1.606
    average=.619*gl+.381*gm
    delta_rg=15*.25*(1.111*gm-.939*gl)*r
    delta_by=15*(.4*gs-.15*average)*r
    delta_y=5*average*r
    # Inverse of opponent matrix [-1,1,0; -1,-1,1; 1,1,0].
    delta=np.stack([(delta_y-delta_rg)/2,(delta_y+delta_rg)/2,delta_by+delta_y],axis=-1)
    return q[...,:3]+delta
