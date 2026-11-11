"""Positive, unit-energy approximation to Watson (2013), equations 1–5.

Only the mean radial optical core at pupil 2–6 mm is represented. Retinal
contrast sensitivity, individual aberrations and wide-angle glare are absent.
Gaussian pixel integrals conserve at-eye flux before tone reproduction.
"""

from functools import lru_cache
from math import erf

import numpy as np

VERSION = 'watson2013_positive_gaussian_mixture_1'


def watson_mtf(frequency_cycles_deg, pupil_mm=6.):
    if not np.isfinite(pupil_mm) or not 2 <= pupil_mm <= 6:
        raise ValueError('Watson 2013 supports pupil diameter 2–6 mm.')
    u = np.asarray(frequency_cycles_deg, dtype=float)
    if np.any(~np.isfinite(u)) or np.any(u < 0):
        raise ValueError('Nonnegative finite spatial frequencies required.')
    cutoff = pupil_mm*np.pi*1e6/(555*180)
    q = np.clip(u/cutoff, 0, 1)
    diffraction = (2/np.pi)*(np.arccos(q)-q*np.sqrt(1-q*q))
    u1 = 21.95-5.512*pupil_mm+.3922*pupil_mm**2
    return (1+(u/u1)**2)**(-.62)*np.sqrt(np.maximum(0, diffraction))


@lru_cache(maxsize=32)
def optical_mixture(pupil_mm=6.):
    """Deterministic nonnegative least squares, with an explicit DC constraint.

    The multiplicative update is followed by unit-sum normalization. Residuals
    are exported; neither Gaussian fit nor mean-eye data is personal calibration.
    """
    pupil_mm = float(pupil_mm)
    frequency = np.r_[0., np.geomspace(.03, 200., 320)]
    sigma_deg = np.geomspace(.015/60, 30/60, 48)
    basis = np.exp(-2*np.pi**2*frequency[:, None]**2*sigma_deg[None, :]**2)
    truth = watson_mtf(frequency, pupil_mm)
    # Weight low-frequency and high-frequency ranges equally per log interval.
    basis[0] *= 8
    target = truth.copy()
    target[0] *= 8
    weights = np.full(len(sigma_deg), 1/len(sigma_deg))
    gram, rhs = basis.T@basis, basis.T@target
    for _ in range(8000):
        weights *= rhs/np.maximum(gram@weights, 1e-30)
    weights /= weights.sum()
    predicted = np.exp(-2*np.pi**2*frequency[:, None]**2*sigma_deg[None, :]**2)@weights
    return {'version': VERSION, 'pupil_diameter_mm': pupil_mm,
        'sigma_deg': sigma_deg.tolist(), 'weight': weights.tolist(),
        'fit_frequency_cycles_deg': [float(frequency.min()), float(frequency.max())],
        'fit_mtf_rms_absolute_error': float(np.sqrt(np.mean((predicted-truth)**2))),
        'fit_mtf_max_absolute_error': float(np.max(abs(predicted-truth))),
        'normalization_integral': float(weights.sum()), 'wide_angle_glare_included': False}


def _normal_cdf(values):
    values = np.asarray(values, dtype=float)
    return np.fromiter((.5*(1+erf(float(x)*.7071067811865476)) for x in values),
                       dtype=float, count=values.size)


@lru_cache(maxsize=1)
def _integration_nodes():
    return np.polynomial.legendre.leggauss(12)


def _gaussian_cdf_grid(x, y, correlation):
    """Bivariate normal CDF: integrate its analytic derivative in correlation."""
    xx, yy = np.meshgrid(np.clip(x, -12, 12), np.clip(y, -12, 12))
    cdf = np.outer(_normal_cdf(y), _normal_cdf(x))
    nodes, weights = _integration_nodes()
    for t, weight in zip((nodes+1)*correlation/2, weights*correlation/2, strict=True):
        cdf += weight/(2*np.pi*np.sqrt(1-t*t))*np.exp(-(xx*xx-2*t*xx*yy+yy*yy)/(2*(1-t*t)))
    return cdf


def deposit_star(buffer, x_px, y_px, illuminance_lux, radians_per_pixel, mixture, *,
                 correlation=0., cutoff_sigma=6., appearance_buffer=None):
    """Integrate an elliptical angular PSF into pixel flux (lux), not peaks.

    Local tangent axes and their correlation use the differential metric of the
    gnomonic projection. A local differential replaces the exact PSF over large
    angular distances; core fit and that approximation are separately reported.
    The crop is NEVER re-normalized. Returned ledger separates FOV losses from
    negligible truncated Gaussian wings.
    """
    if illuminance_lux < 0 or not np.isfinite(illuminance_lux):
        raise ValueError('Finite nonnegative at-eye illuminance required.')
    scales = np.asarray(radians_per_pixel, dtype=float)
    if scales.ndim == 0:
        scales = np.repeat(scales, 2)
    if np.any(~np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError('Two positive angular pixel scales required.')
    if not np.isfinite(correlation) or abs(correlation) >= .8:
        raise ValueError('Core projection correlation must be within (-.8,.8).')
    height, width = buffer.shape
    deposited, truncated = 0., 0.
    for sigma_deg, weight in zip(mixture['sigma_deg'], mixture['weight'], strict=True):
        if weight < 1e-12:
            truncated += illuminance_lux*weight
            continue
        sx, sy = np.deg2rad(sigma_deg)/scales
        x0, x1 = max(0, int(np.floor(x_px-cutoff_sigma*sx))), min(width, int(np.ceil(x_px+cutoff_sigma*sx))+1)
        y0, y1 = max(0, int(np.floor(y_px-cutoff_sigma*sy))), min(height, int(np.ceil(y_px+cutoff_sigma*sy))+1)
        if x1 <= x0 or y1 <= y0:
            continue
        xx=(np.arange(x0, x1+1)-.5-x_px)/sx
        yy=(np.arange(y0, y1+1)-.5-y_px)/sy
        if abs(correlation) < 1e-10:
            mass = np.outer(np.maximum(0, np.diff(_normal_cdf(yy))), np.maximum(0, np.diff(_normal_cdf(xx))))
        else:
            mass=np.maximum(0,np.diff(np.diff(_gaussian_cdf_grid(xx,yy,correlation),axis=0),axis=1))
        buffer[y0:y1, x0:x1] += illuminance_lux*weight*mass
        if appearance_buffer is not None:
            appearance_buffer[y0:y1, x0:x1] += illuminance_lux*weight*mass
        deposited += illuminance_lux*weight*float(mass.sum())
        truncated += illuminance_lux*weight*2*(1-erf(cutoff_sigma/np.sqrt(2)))
    return {'input_scotopic_illuminance_lux': float(illuminance_lux),
        'deposited_scotopic_illuminance_lux': float(deposited),
        'outside_field_or_truncated_lux': float(max(0, illuminance_lux-deposited)),
        'gaussian_truncation_upper_bound_lux': float(truncated)}


def pixel_solid_angles(view):
    """Exact spherical quadrilateral area of each gnomonic pixel (steradian)."""
    width, height = int(view['width']), int(view['height'])
    step = 2*np.tan(np.deg2rad(view['horizontal_fov_deg'])/2)/width
    x = (np.arange(width+1)-width/2)*step
    y = (np.arange(height+1)-height/2)*step
    xx, yy = np.meshgrid(x, y)
    primitive = np.arctan2(xx*yy, np.sqrt(1+xx*xx+yy*yy))
    return np.diff(np.diff(primitive, axis=0), axis=1)[::-1].copy()
