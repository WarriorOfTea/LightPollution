"""Positive view-local reconstruction on a gnomonic patch of the unit sphere.

The periodic all-sky grid remains a fallback outside the patch. PCHIP is a
shape-preserving numerical interpolant, not an additional scattering law.
"""

import numpy as np

from ..skyview import camera_basis


def view_geometry(view):
    basis = np.asarray(camera_basis(view['azimuth_deg'], view['altitude_deg']))
    edge = float(np.tan(np.deg2rad(view['horizontal_fov_deg']) / 2))
    return basis, edge, edge * view['height'] / view['width']


def tangent_directions(view, x, y):
    x, y = np.broadcast_arrays(x, y)
    basis, ex, ey = view_geometry(view)
    rays = basis[0] + (x * ex)[..., None] * basis[1] + (y * ey)[..., None] * basis[2]
    rays /= np.linalg.norm(rays, axis=-1)[..., None]
    return np.rad2deg(np.arctan2(rays[..., 0], rays[..., 1])) % 360, np.rad2deg(np.arcsin(rays[..., 2]))


def tangent_coordinates(view, azimuth, altitude):
    a, h = np.deg2rad(np.broadcast_arrays(azimuth, altitude))
    rays = np.stack([np.cos(h)*np.sin(a), np.cos(h)*np.cos(a), np.sin(h)], axis=-1)
    basis, ex, ey = view_geometry(view)
    depth = rays @ basis[0]
    safe = np.where(depth > 0, depth, 1.)
    return (rays @ basis[1]) / (safe * ex), (rays @ basis[2]) / (safe * ey), depth > 0


def _slopes(axis, values):
    """Fritsch–Butland weighted harmonic derivatives; last axis is sampled."""
    h = np.diff(axis)
    d = np.diff(values, axis=-1) / h
    m = np.zeros_like(values, dtype=float)
    w1, w2 = 2*h[1:] + h[:-1], h[1:] + 2*h[:-1]
    same = (d[..., :-1] * d[..., 1:]) > 0
    with np.errstate(divide='ignore', invalid='ignore'):
        harmonic = (w1+w2) / (w1/d[..., :-1] + w2/d[..., 1:])
    m[..., 1:-1] = np.where(same, harmonic, 0.)
    for end, j, k in [(0, 0, 1), (-1, -1, -2)]:
        trial = ((2*h[j]+h[k])*d[..., j]-h[j]*d[..., k])/(h[j]+h[k])
        trial = np.where(trial*d[..., j] <= 0, 0., trial)
        trial = np.where((d[..., j]*d[..., k] < 0) & (abs(trial) > 3*abs(d[..., j])),
                         3*d[..., j], trial)
        m[..., end] = trial
    return m


def _hermite(axis, values, query, slopes=None, *, per_query=False):
    slopes = _slopes(axis, values) if slopes is None else slopes
    j = np.clip(np.searchsorted(axis, query, side='right')-1, 0, len(axis)-2)
    h = axis[j+1]-axis[j]
    t = np.clip((query-axis[j])/h, 0, 1)
    # Either shared samples (..., n), or one set per query (q, n).
    if per_query:
        v0, v1 = values[np.arange(query.size), j], values[np.arange(query.size), j+1]
        m0, m1 = slopes[np.arange(query.size), j], slopes[np.arange(query.size), j+1]
    else:
        v0, v1, m0, m1 = values[..., j], values[..., j+1], slopes[..., j], slopes[..., j+1]
    return ((2*t**3-3*t*t+1)*v0 + (t**3-2*t*t+t)*h*m0 +
            (-2*t**3+3*t*t)*v1 + (t**3-t*t)*h*m1)


def interpolate_patch(patch, values, x, y):
    x, y = np.broadcast_arrays(x, y)
    shape = x.shape
    x, y = x.ravel(), y.ravel()
    ax, ay = np.asarray(patch['x']), np.asarray(patch['y'])
    values = np.asarray(values, dtype=float)
    slopes = _slopes(ax, values)
    result = np.empty(x.size)
    for start in range(0, x.size, 65536):
        sl = slice(start, start+65536)
        # Transpose so every query has its own set of vertical samples.
        horizontal = _hermite(ax, values, x[sl], slopes).T
        result[sl] = _hermite(ay, horizontal, y[sl], per_query=True)
    return result.reshape(shape)


def patch_support(patch, x, y):
    """Conservative support of the entire PCHIP derivative stencil."""
    ax, ay = np.asarray(patch['x']), np.asarray(patch['y'])
    j = np.clip(np.searchsorted(ax, x, side='right')-1, 0, len(ax)-2)
    i = np.clip(np.searchsorted(ay, y, side='right')-1, 0, len(ay)-2)
    ok = np.ones(np.broadcast(x, y).shape, dtype=bool)
    support = np.asarray(patch['physical_supported'], dtype=bool)
    for dy in [-1, 0, 1, 2]:
        for dx in [-1, 0, 1, 2]:
            ok &= support[np.clip(i+dy, 0, len(ay)-1), np.clip(j+dx, 0, len(ax)-1)]
    return ok


def heldout_coordinates(view):
    """24 geometry-only off-grid probes; never used by mesh refinement."""
    def radical(n, base):
        value, factor = 0., 1./base
        while n:
            value += (n % base)*factor
            n //= base
            factor /= base
        return value
    points = [[1.9*radical(i, 2)-.95, 1.9*radical(i, 3)-.95] for i in range(1, 25)]
    # Explicitly straddle the coordinate singularity when zenith is in view.
    zx, zy, front = tangent_coordinates(view, 0., 90.)
    if front and abs(zx) < .9 and abs(zy) < .9:
        points[:4] = [[float(zx)+dx, float(zy)+dy]
                      for dx, dy in [(-.008, -.006), (.008, .006), (-.015, .016), (.015, -.016)]]
    return np.asarray(points)


def build_view_patch(view, evaluate, *, target=.01, max_grid_size=5, validation='none',
                     evaluate_standard=None, check=lambda: None):
    """Adapt a tensor grid using separate cell-centre probes, then freeze it.

    evaluate returns [city P, city S, residual V, physical support]. Each
    physical solver invocation remains cached by Planner's existing kernel.
    """
    cache = {}
    def at(x, y, evaluator=evaluate):
        az, alt = tangent_directions(view, x, y)
        if float(alt) < 10:
            return [0., 0., 0., False]
        key = (round(float(x), 13), round(float(y), 13))
        if evaluator is evaluate and key in cache:
            return cache[key]
        check()
        row = evaluator(float(az), float(alt))
        if evaluator is evaluate:
            cache[key] = row
        return row
    levels = []
    size = 3
    while True:
        axis = np.linspace(-1., 1., size)
        nodes = np.asarray([[at(x, y) for x in axis] for y in axis], dtype=float)
        patch = {'version': 'gnomonic_tensor_pchip_1', 'view': view, 'x': axis.tolist(), 'y': axis.tolist(),
            'city_photopic_luminance_cd_m2': nodes[..., 0].tolist(),
            'city_scotopic_luminance_cd_m2': nodes[..., 1].tolist(),
            'unresolved_v_relative': nodes[..., 2].tolist(), 'physical_supported': nodes[..., 3].astype(bool).tolist()}
        mid = (axis[:-1]+axis[1:])/2
        points = np.array([[x, y] for y in mid for x in mid])
        truth = np.asarray([at(x, y) for x, y in points])
        estimate = np.stack([interpolate_patch(patch, nodes[..., k], *points.T) for k in range(3)], axis=-1)
        valid = truth[:, 3].astype(bool) & patch_support(patch, *points.T)
        # Absolute near-zero floor is explicit and reported, not hidden in %.
        floors = np.array([1e-12, 1e-12, 1e-8])
        error = abs(estimate-truth[:, :3]) / np.maximum(abs(truth[:, :3]), floors)
        maximum = float(error[valid].max()) if np.any(valid) else None
        levels.append({'grid_size': size, 'refinement_probes': len(points), 'supported_probes': int(valid.sum()),
                       'max_relative_error': maximum})
        if maximum is None or maximum <= target or 2*size-1 > max_grid_size:
            break
        size = 2*size-1
    patch['refinement'] = {'target_relative_error': target, 'levels': levels,
        'target_met_on_refinement_probes': maximum is not None and maximum <= target,
        'max_grid_size': max_grid_size, 'unique_quick_rays': len(cache),
        'near_zero_floors_P_S_V': floors.tolist(), 'source_scale_refitted': False}
    if validation == 'heldout_24':
        points = heldout_coordinates(view)
        # The grid above is already frozen: these rows cannot refine it.
        truth = np.asarray([at(x, y) for x, y in points])
        estimate = np.stack([interpolate_patch(patch, nodes[..., k], *points.T) for k in range(3)], axis=-1)
        rows = []
        for index, ((x, y), direct, predicted) in enumerate(zip(points, truth, estimate, strict=True)):
            az, alt = tangent_directions(view, x, y)
            ok = bool(direct[3] and patch_support(patch, x, y))
            row = {'index': index, 'x': float(x), 'y': float(y), 'azimuth_deg': float(az),
                'altitude_deg': float(alt), 'supported': ok, 'direct_P_S_V': direct[:3].tolist(),
                'interpolated_P_S_V': predicted.tolist(),
                'relative_errors_P_S_V': (abs(predicted-direct[:3])/np.maximum(abs(direct[:3]), floors)).tolist()}
            if evaluate_standard is not None and index in (0, 7, 15, 23):
                standard = np.asarray(at(x, y, evaluate_standard)[:3])
                row['standard_P_S_V'] = standard.tolist()
                row['quick_standard_relative_errors_P_S_V'] = (abs(direct[:3]-standard)/np.maximum(abs(standard), floors)).tolist()
            rows.append(row)
        errors = [r['relative_errors_P_S_V'] for r in rows if r['supported']]
        patch['validation'] = {'protocol': 'heldout_24_geometry_only_1', 'rows': rows,
            'used_for_refinement': False, 'interpolation_target_relative_error': .02,
            'supported_rays': len(errors), 'max_relative_error': float(np.max(errors)) if errors else None,
            'all_supported_rays_within_target': bool(len(errors) == 24 and np.max(errors) <= .02),
            'physical_model_accuracy_established': False, 'source_scale_refitted': False}
    return patch
