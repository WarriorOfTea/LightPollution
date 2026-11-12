"""Exact angular contours from KStars, independent of atmospheric radiance.

The nearest floor above / ceiling below rule matches ArtificialHorizon::isVisible.
No azimuth grid is introduced: each segment is evaluated at the actual ray.
"""

from copy import deepcopy
from math import atan2, cos, sin, sqrt, tan, pi

import numpy as np


def validate_horizon(value):
    if (not isinstance(value, dict) or set(value) != {'kind', 'regions'}
            or value['kind'] not in ('KStars_piecewise_linear_v1', 'KStars_great_circle_v1')
            or not isinstance(value['regions'], list) or len(value['regions']) > 64):
        raise ValueError('Expected KStars piecewise-linear horizon with at most64 regions.')
    count = 0
    for region in value['regions']:
        if (not isinstance(region, dict) or set(region) != {'name', 'ceiling', 'points'}
                or not isinstance(region['name'], str) or len(region['name']) > 256
                or type(region['ceiling']) is not bool or not isinstance(region['points'], list)
                or len(region['points']) < 2):
            raise ValueError('Invalid KStars horizon region.')
        count += len(region['points'])
        for point in region['points']:
            if (not isinstance(point, list) or len(point) != 2
                    or any(isinstance(v, bool) or not isinstance(v, (int, float))
                           or not np.isfinite(v) for v in point)
                    or not 0 <= point[0] <= 360 or not -90 <= point[1] <= 90):
                raise ValueError('Horizon points need finite azimuth/altitude in degrees.')
    if count > 4096:
        raise ValueError('At most4096 horizon vertices are supported.')
    return deepcopy(value)


def _great_circle_level(a0, h0, a1, h1, azimuth):
    """Spherical navigation parameters used by native GreatCircle::altAtAz.

    Use scalar libm for the two vertices, as KStars does; NumPy evaluates the
    image rays in parallel. Exact vertex rays use the same scalar evaluation.
    """
    def radians(value):return value * 2.0 * pi / 360.0
    a0, h0, a1, h1 = map(radians, (a0, h0, a1, h1))
    delta = a1-a0
    alpha1 = atan2(cos(h1)*sin(delta), cos(h0)*sin(h1)-sin(h0)*cos(h1)*cos(delta))
    alpha0 = atan2(sin(alpha1)*cos(h0), sqrt(cos(alpha1)**2+sin(alpha1)**2*sin(h0)**2))
    sigma = 0.0 if h0 == 0 and alpha1 == .5*pi else atan2(tan(h0), cos(alpha1))
    origin = a0-atan2(sin(alpha0)*sin(sigma), cos(sigma))
    angle = np.asarray(azimuth)*2.0*pi/360.0
    def degrees(value):
        value = value*360.0/(2*pi)
        return np.where(value < -90, value+180, np.where(value > 90, value-180, value))
    level = degrees(np.arctan2(np.sin(angle-origin), tan(alpha0)))
    for vertex in (a0, a1):
        level = np.where(angle == vertex, degrees(atan2(sin(vertex-origin), tan(alpha0))), level)
    return level


def horizon_visible(horizon, azimuth, altitude):
    azimuth, altitude = np.broadcast_arrays(np.asarray(azimuth, float), np.asarray(altitude, float))
    visible = np.ones(altitude.shape, dtype=bool)
    if horizon is None:
        return visible
    spherical = horizon['kind'] == 'KStars_great_circle_v1'
    above, below = np.full(altitude.shape, np.inf), np.full(altitude.shape, -np.inf)
    above_ceiling, below_ceiling = np.ones(altitude.shape, bool), np.zeros(altitude.shape, bool)
    for region in horizon['regions']:
        ceiling = region['ceiling']
        limit = np.full(altitude.shape, np.inf if ceiling else -np.inf)
        exists = np.zeros(altitude.shape, bool)
        for (a0, h0), (a1, h1) in zip(region['points'], region['points'][1:]):
            if spherical:
                # Same pole avoidance as ArtificialHorizonEntity.
                h0, h1 = min(89.999, h0), min(89.999, h1)
            # KStars uses the shorter arc, including intervals crossing north.
            delta = abs((a1-a0+180) % 360-180)
            distance = abs((azimuth-a0+180) % 360-180)
            to_end = abs((azimuth-a1+180) % 360-180)
            inside = (distance <= delta) & (to_end <= delta)
            if delta == 0:
                level = np.full(altitude.shape, h1)
            elif spherical:
                level = _great_circle_level(a0, h0, a1, h1, azimuth)
            else:
                level = h0+(h1-h0)*distance/delta
            limit = np.where(inside, np.minimum(limit, level) if ceiling else np.maximum(limit, level), limit)
            exists |= inside
        higher = exists & (limit > altitude) & (limit < above)
        lower = exists & (limit < altitude) & (limit > below)
        above = np.where(higher, limit, above)
        below = np.where(lower, limit, below)
        above_ceiling = np.where(higher, ceiling, above_ceiling)
        below_ceiling = np.where(lower, ceiling, below_ceiling)
    return visible & above_ceiling & ~below_ceiling
