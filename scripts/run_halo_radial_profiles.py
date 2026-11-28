"""Fit an energy-neutral city diagram to the radial slopes of multiple VIIRS views."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from datetime import date, datetime
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

from diagnose_halo_satellite import angular_basis, regression, spatial_terms
from lightpollution.adapters.storage import file_hash, read_json, write_json
from lightpollution.data import PIXELS_PER_DEGREE
from lightpollution.models.all_orders import AllOrderSettings
from lightpollution.models.halo_profiles import (
    KAPPA_BOUNDS,
    pool_profile_fits,
    profile_normal_terms,
    redistribution_flux_below,
)
from lightpollution.models.satellite_halo import build_halo_kernel
from lightpollution.physics import Atmosphere, EARTH_RADIUS_M, cell_areas
from run_halo_city_fits import LOCATIONS, responses
from run_viirs_halo_recovery import block_measurements, preflight


RADII = (4, 8, 13, 19, 28, 40)
RINGS = len(RADII) - 1
SECTORS = ('toward', 'right', 'away', 'left')
PROFILE_MODES = ('annular', 'radial', 'sides', 'joint')
NOISE_FLOOR = 0.02
NPP, NOAA = 'VNP46A1', 'VJ146A1'
NOAA_DATES = ('2019-03-31', '2019-04-01', '2019-04-03', '2019-04-07', '2019-04-08')


@dataclass
class Vectors:
    observed: np.ndarray
    design: np.ndarray
    positions: np.ndarray
    edge: np.ndarray
    label: np.ndarray
    block: np.ndarray
    macroblock: np.ndarray
    test: np.ndarray


def radial_footprint(data, azimuth, spacing, threshold=2.0):
    """Four view-relative sectors and distance from the nearest bright city cell."""
    bright = data['city_mask'] & (data['near_radiance'] >= threshold)
    edge = distance_transform_edt(~bright, sampling=spacing) / 1000
    r = np.hypot(data['north'], data['east'])
    theta = np.rad2deg(np.arctan2(data['east'], data['north']))
    quadrant = np.floor(((theta - azimuth + 45) % 360) / 90).astype(int)
    weak = data['weak_surface'] & (r <= 65)
    background = data['masks'][2] & (edge >= RADII[-1])
    profiles = [
        weak & (quadrant == sector) & (edge >= lo) & (edge < hi)
        for sector in range(len(SECTORS))
        for lo, hi in zip(RADII, RADII[1:])
    ]
    return (data['masks'][0], data['masks'][3], background, *profiles), edge


def vectors(data, grid, terms):
    masks, edge = grid
    shape = edge.shape
    y, x, i, j, labels = block_measurements(
        data | {'masks': masks},
        (
            terms[..., 0] + terms[..., 1],
            terms[..., 1] - terms[..., 4],
            terms[..., 2],
            terms[..., 3],
            np.broadcast_to(data['north'], shape),
            np.broadcast_to(data['east'], shape),
            edge,
        ),
    )
    blocks = np.column_stack((i, j))
    macro = (i // 4) * (int(shape[1]) // 16 + 1) + j // 4
    test = ((i // 4) + 2 * (j // 4)) % 3 == 0
    return Vectors(y, x[:, :4], x[:, 4:6], x[:, 6], labels, blocks, macro, test)


def fit_core(data, *, fit_scale=True, use_weak=True, gradient=True):
    """Power and nuisance calibration uses core, bright neighbors and distant dark blocks."""
    use = ~data.test & (data.label < 3)
    y, x, cl = data.observed[use], data.design[use], data.label[use]
    w = np.ones(len(y))
    for i in range(3):
        group = y[cl == i]
        if len(group) < 3:
            raise ValueError('Insufficient independent core or background support.')
        w[cl == i] = 1 / (max(0.10, float(np.median(group))) * np.sqrt(len(group)))
    cols = [x[:, 2]] + ([x[:, 3]] if use_weak else []) + [np.ones_like(y)]
    offset = np.zeros_like(y) if fit_scale else x[:, 0]
    if fit_scale:
        cols.insert(0, x[:, 0])
    pos = data.positions[use] / 65 if gradient else data.positions[use, :0]
    coef, slope = regression(np.column_stack(cols), y - offset, w, pos)
    s = float(coef[0]) if fit_scale else 1.0
    params = coef[1:] if fit_scale else coef
    return {
        'scale': s,
        'bright': float(params[0]),
        'weak': float(params[1]) if use_weak else 0.0,
        'constant': float(params[-1]),
        'gradient': slope.tolist(),
        'train_n': np.bincount(cl, minlength=3).tolist(),
    }


def forward(data, calibration):
    n = (
        data.design[:, 2] * calibration['bright']
        + data.design[:, 3] * calibration['weak']
        + calibration['constant']
    )
    if calibration['gradient']:
        n += data.positions @ np.asarray(calibration['gradient']) / 65
    s = calibration['scale']
    return n, s * data.design[:, 0], s * data.design[:, 1]


def aggregate(data, use, values, *, annular=False):
    """Means over spatial groups, plus a covariance with shared-block terms."""
    mask = use & (data.label >= 3)
    group = data.label - 3
    if annular:
        group = group % RINGS
    all_macros = np.unique(data.macroblock[mask])
    summaries = {}
    for k in np.unique(group[mask]):
        here = mask & (group == k)
        macro = data.macroblock[here]
        unique, index = np.unique(macro, return_inverse=True)
        if len(macro) < 3 or len(unique) < 2:
            continue
        buckets = np.stack(
            [values[here][index == j].mean(axis=0) for j in range(len(unique))]
        )
        mean = buckets.mean(axis=0)
        spread = (buckets[:, 0] - mean[0]) / np.sqrt(len(unique) * (len(unique) - 1))
        noise = np.zeros(len(all_macros))
        noise[np.searchsorted(all_macros, unique)] = spread
        summaries[int(k)] = (mean, noise, len(macro), len(unique))
    return summaries


def difference_rows(summaries, kind):
    keys = sorted(summaries)
    index = {k: i for i, k in enumerate(keys)}
    if kind == 'annular':
        links = [(r, r + 1) for r in range(RINGS - 1)]
    elif kind == 'radial':
        links = [
            (s * RINGS + r, s * RINGS + r + 1)
            for s in range(len(SECTORS))
            for r in range(RINGS - 1)
        ]
    else:
        links = [
            (axis * RINGS + r, (axis + 2) * RINGS + r)
            for axis in (0, 1)
            for r in range(RINGS)
        ]
    rows = []
    for a, b in links:
        if a in index and b in index:
            contrast = np.zeros(len(keys))
            contrast[[index[a], index[b]]] = [1, -1]
            rows.append(contrast)
    return np.array(rows).reshape(len(rows), len(keys))


def normal_equation(summary, mode):
    if not summary:
        return None
    means = np.array([summary[k][0] for k in sorted(summary)])
    noise = np.array([summary[k][1] for k in sorted(summary)])
    covariance = noise @ noise.T + np.eye(len(means)) * NOISE_FLOOR**2
    kinds = ('radial', 'sides') if mode == 'joint' else (mode,)
    contrast = np.concatenate([difference_rows(summary, kind) for kind in kinds])
    if len(contrast) < 2:
        return None
    return profile_normal_terms(
        contrast, covariance, means[:, 0] - means[:, 1], means[:, 2]
    )


def fit_profiles(data, calibration):
    nuisance, base, derivative = forward(data, calibration)
    values = np.column_stack((data.observed, base + nuisance, derivative, data.edge))
    sector_profile = aggregate(data, ~data.test, values)
    annular = aggregate(data, ~data.test, values, annular=True)
    return {
        k: normal_equation(annular if k == 'annular' else sector_profile, k)
        for k in PROFILE_MODES
    }


def score_scene(data, calibration, kappa):
    nuisance, base, derivative = forward(data, calibration)
    pred = base + kappa * derivative + nuisance
    use = data.test & (data.label >= 3)
    core = data.test & (data.label == 0)
    residual = pred - data.observed
    sums = aggregate(
        data,
        data.test,
        np.column_stack(
            (data.observed, pred, data.edge, nuisance, base + kappa * derivative)
        ),
    )
    profiles = []
    for k, (value, _, blocks, groups) in sorted(sums.items()):
        profiles.append(
            {
                'sector': SECTORS[k // RINGS],
                'ring': k % RINGS,
                'values': value.tolist(),
                'blocks': blocks,
                'groups': groups,
            }
        )
    stats = {
        'halo_n': int(use.sum()),
        'halo_mae': float(abs(residual[use]).mean()),
        'halo_bias': float(residual[use].mean()),
        'core_n': int(core.sum()),
        'core_mae': float(abs(residual[core]).mean()) if core.any() else None,
    }
    if sums:
        means = np.array([sums[k][0] for k in sorted(sums)])
        for mode in ('radial', 'sides'):
            d = difference_rows(sums, mode)
            error = d @ (means[:, 1] - means[:, 0])
            if mode == 'radial':
                error /= np.abs(d @ means[:, 2])
            stats[mode + '_n'] = len(d)
            stats[mode + '_mae'] = float(abs(error).mean()) if len(error) else None
    return {'score': stats, 'profiles': profiles}


def angular_statistics(data):
    area = cell_areas(data['bounds'].latitude)[:, None]
    city, near, off = [
        1e-5 * float(np.sum(data[k] * area))
        for k in ('city_flux', 'paired_city_near', 'paired_city_off')
    ]
    t, w = np.polynomial.legendre.leggauss(12)

    def integrate(high):
        breaks = (0, 0.5, np.cos(np.deg2rad(40)), np.cos(np.deg2rad(20)), 1)
        edges = [e for e in breaks if e < high] + [high]
        total = 0.0
        for a, b in zip(edges, edges[1:]):
            mu = a + (t + 1) * (b - a) / 2
            basis = responses(mu)
            L = basis @ np.array([city - np.pi * near, 0, near, off, off])
            total += np.pi * (b - a) * (w @ (mu * L))
        return float(total)

    total = integrate(1)
    return {
        'h2_equivalent_dnb_w': total,
        'fraction_h2': [integrate(np.sin(np.deg2rad(e))) / total for e in (10, 25)],
        'fraction_per_kappa': [
            2 * np.pi * off * redistribution_flux_below(e) / total for e in (10, 25)
        ],
    }


def atmospheres(scene):
    model = Atmosphere(**scene['atmosphere'])
    tucson = scene['city'] == 'Tucson'
    if tucson:
        model = replace(
            model, aerosol_g=float(np.interp(650, [400, 680], [0.68, 0.64]))
        )
    return {
        'nominal': model,
        'shallow': replace(
            model, aerosol_scale_height_m=model.aerosol_scale_height_m * 0.65
        ),
        'deep': replace(
            model, aerosol_scale_height_m=model.aerosol_scale_height_m * 1.8
        ),
        'more_aod': replace(
            model, aerosol_optical_depth=model.aerosol_optical_depth * 1.5
        ),
        'alternative_phase': replace(model, aerosol_g=0.8 if tucson else 0.62),
    }


def evaluate(scene, data, paired, samples):
    dy = EARTH_RADIUS_M * np.pi / (180 * PIXELS_PER_DEGREE)
    dx = dy * np.cos(np.deg2rad(data['lat']))
    weak = np.where(
        (~data['city_mask'])
        & (data['near_radiance'] >= 0.3)
        & (data['near_radiance'] < 2),
        np.pi * data['near_radiance'],
        0,
    )
    grid = radial_footprint(data, scene['azimuth_deg'], (dy, dx))
    angular = angular_statistics(data)
    fits, stress, normals = {}, {}, {}
    settings = AllOrderSettings(
        samples=samples, batch_size=32768, seed=scene['seed'], recorded_orders=1
    )
    for name, air in atmospheres(scene).items():
        kernel = build_halo_kernel(
            scene['zenith_deg'],
            scene['azimuth_deg'],
            dx,
            dy,
            150_000,
            air,
            settings,
            angular_radiance=angular_basis,
        )
        terms = sum(spatial_terms(data, kernel, weak, True))
        v = vectors(data, grid, terms)
        calibration = fit_core(v)
        normal = fit_profiles(v, calibration)
        best = pool_profile_fits([normal['joint']]) if normal['joint'] else None
        stress[name] = {
            'scale': calibration['scale'],
            'joint': best,
            'atmosphere': dict(air.__dict__),
        }
        if name == 'nominal':
            nominal = v
            core = calibration
            normals = normal
            fits['h2'] = score_scene(v, fit_core(v, fit_scale=False), 0)
            fits['core'] = score_scene(v, core, 0)
            for label, kappa in zip(
                ('lower_bound', 'upper_bound'), KAPPA_BOUNDS, strict=True
            ):
                fits[label] = {'kappa': kappa} | score_scene(v, core, kappa)
            for mode, n in normal.items():
                if n:
                    fit = pool_profile_fits([n])
                    fits[mode] = fit | score_scene(v, core, fit['kappa'])
            for variant, options in (
                ('without_weak', {'use_weak': False}),
                ('constant_background', {'gradient': False}),
            ):
                alternative = fit_core(v, **options)
                alt = fit_profiles(v, alternative)['joint']
                if alt:
                    stress[variant] = {
                        'scale': alternative['scale'],
                        'joint': pool_profile_fits([alt]),
                    }
            edge_alt = radial_footprint(
                data, scene['azimuth_deg'], (dy, dx), threshold=4
            )
            other = vectors(data, edge_alt, terms)
            alt_core = fit_core(other)
            alt = fit_profiles(other, alt_core)['joint']
            if alt:
                stress['brighter_edge'] = {
                    'scale': alt_core['scale'],
                    'joint': pool_profile_fits([alt]),
                }
            common = None
            if paired:
                mask, az = paired
                same = data | {'valid': data['valid'] & mask}
                common = vectors(same, radial_footprint(data, az, (dy, dx)), terms)
    core_train = (nominal.label == 0) & ~nominal.test
    design = nominal.design[core_train]
    sensitivity = (
        abs(design[:, 1]) / np.maximum(design[:, 0], 1e-10) * max(KAPPA_BOUNDS)
    )
    result = {
        **{
            k: scene[k]
            for k in (
                'city',
                'night',
                'utc',
                'zenith_deg',
                'azimuth_deg',
                'files',
                'source_template_files',
                'product',
                'solar_altitude_core_deg',
                'lunar_altitude_core_deg',
            )
        },
        'angular': angular,
        'core': core,
        'core_p95_fractional_template_change': float(np.quantile(sensitivity, 0.95)),
        'fit': fits,
        'stress': stress,
        'normals': normals,
    }
    return result, nominal, common


def paired_fit(a, b, ca, cb):
    if not (np.array_equal(a.label, b.label) and np.array_equal(a.block, b.block)):
        raise ValueError('Paired profiles must use the same locations and masks.')
    na, ha, da = forward(a, ca)
    nb, hb, db = forward(b, cb)
    values = np.column_stack(
        (a.observed - b.observed, (ha + na) - (hb + nb), da - db, a.edge)
    )
    training = aggregate(a, ~a.test, values)
    normal = normal_equation(training, 'joint')
    if not normal:
        return None
    fit = pool_profile_fits([normal])
    kappa = fit['kappa']
    use = a.test & (a.label >= 3)
    residual = values[use, 1] - values[use, 0]
    trial = residual + kappa * values[use, 2]
    held = aggregate(
        a,
        a.test,
        np.column_stack(
            (a.observed - na, b.observed - nb, a.edge, ha + kappa * da, hb + kappa * db)
        ),
    )
    profiles = [
        {
            'sector': SECTORS[k // RINGS],
            'ring': k % RINGS,
            'values': value.tolist(),
            'blocks': blocks,
            'groups': groups,
        }
        for k, (value, _, blocks, groups) in sorted(held.items())
    ]
    return fit | {
        'blocks': int(use.sum()),
        'profiles_used': len(training),
        'mae_core': float(abs(residual).mean()),
        'mae_gradient': float(abs(trial).mean()),
        'profiles': profiles,
    }


def add_view_tests(points, records):
    groups = {
        'berlin_two_nights': ('Berlin/2017-03-28', 'Berlin/2017-03-29'),
        'madrid_december': ('Madrid/2013-12-04', 'Madrid/2013-12-29'),
        'tucson_dimmed_april': (
            'Tucson/2019-04-01',
            'Tucson/2019-04-03',
            'Tucson/2019-04-01/NOAA20',
            'Tucson/2019-04-03/NOAA20',
        ),
    }
    same_night = {d: (f'Tucson/{d}', f'Tucson/{d}/NOAA20') for d in NOAA_DATES}
    same_night = {k: v for k, v in same_night.items() if all(w in points for w in v)}
    pairs, joint = {}, {}
    for date_key, (a, b) in same_night.items():
        pa, pb = (points[k] for k in (a, b))
        ta, tb = (datetime.fromisoformat(p['utc']) for p in (pa, pb))
        minutes = abs((ta - tb).total_seconds()) / 60
        if minutes <= 90 and all(7 <= (x.hour + x.minute / 60) < 10 for x in (ta, tb)):
            groups['tucson/' + date_key] = (a, b)
            f = paired_fit(records[a][1], records[b][1], pa['core'], pb['core'])
            if f:
                pairs[date_key] = f | {'keys': [a, b], 'separation_minutes': minutes}
                for k in (a, b):
                    p = points[k]
                    p['fit']['paired_difference'] = {'kappa': f['kappa']} | score_scene(
                        records[k][0], p['core'], f['kappa']
                    )
    for group, names in groups.items():
        selected = [k for k in names if k in points and points[k]['normals']['joint']]
        if len(selected) < 2:
            continue
        result = pool_profile_fits([points[k]['normals']['joint'] for k in selected])
        joint[group] = result | {'keys': selected}
        for key in selected:
            donors = [k for k in selected if k != key]
            mode = 'other_platform' if '/' in group else 'another_night'
            if group == 'tucson_dimmed_april':
                donors = [k for k in donors if k.split('/')[1] != key.split('/')[1]]
            transferred = pool_profile_fits(
                [points[k]['normals']['joint'] for k in donors]
            )
            p = points[key]
            if transferred:
                p['fit'][mode] = (
                    transferred
                    | {'donors': donors}
                    | score_scene(records[key][0], p['core'], transferred['kappa'])
                )
            p.setdefault('shared_groups', {})[group] = score_scene(
                records[key][0], p['core'], result['kappa']
            )
    return pairs, joint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        'daily',
        'noaa_daily',
        'maps',
        'boundaries',
        'previous_satellite',
        'output',
    ):
        parser.add_argument(name, type=Path)
    parser.add_argument('--samples', type=int, default=1_048_576)
    parser.add_argument('--workers', type=int, default=5)
    args = parser.parse_args()
    previous = read_json(args.previous_satellite)
    if file_hash(args.boundaries) != previous['boundaries_sha256']:
        raise ValueError(
            'Expected the same municipality masks as in the earlier study.'
        )
    geometry = read_json(args.boundaries)
    points, records, exclusions = {}, {}, {}
    with ProcessPoolExecutor(args.workers) as pool:
        jobs = {}
        for key, old in previous['scenes'].items():
            if old['status'] != 'evaluated':
                continue
            city, d = key.split('/')
            locations = [(key, args.daily, NPP)]
            if city == 'Tucson' and d in NOAA_DATES:
                locations.append((key + '/NOAA20', args.noaa_daily, NOAA))
            data = {}
            for vkey, path, product in locations:
                try:
                    info, grid = preflight(
                        path,
                        city,
                        date.fromisoformat(d),
                        geometry,
                        args.maps,
                        location=LOCATIONS[city],
                        minimum_class_cells=1,
                        compare_a2=False,
                        a1_product=product,
                    )
                except ValueError as exc:
                    if product == NPP:
                        raise
                    exclusions[vkey] = str(exc)
                    continue
                if grid is None:
                    exclusions[vkey] = info | {
                        'status': 'insufficient_clear_dark_support'
                    }
                    continue
                if (
                    info['source_template_files'] != old['source_template_files']
                    or product == NPP
                    and info['files'] != old['files']
                ):
                    raise ValueError(f'Input provenance differs for {vkey}')
                data[vkey] = (
                    info
                    | {
                        'product': product,
                        'atmosphere': old['atmosphere'],
                        'seed': old['seed'],
                    },
                    grid,
                )
            for view, (scene, grid) in data.items():
                companion = next((v for k, v in data.items() if k != view), None)
                atmosphere_pair = (
                    (companion[1]['valid'], data[key][0]['azimuth_deg'])
                    if companion
                    else None
                )
                jobs[view] = pool.submit(
                    evaluate, scene, grid, atmosphere_pair, args.samples
                )
        for key, job in jobs.items():
            result, nominal, common = job.result()
            points[key], records[key] = result, (nominal, common)
            print(
                key,
                'scale',
                round(result['core']['scale'], 3),
                'shape',
                {
                    k: round(v['kappa'], 3)
                    for k, v in result['fit'].items()
                    if 'kappa' in v
                },
                flush=True,
            )
    pairs, shared = add_view_tests(points, records)
    write_json(
        args.output,
        {
            'scenes': points,
            'paired_views': pairs,
            'shared_shape': shared,
            'excluded': exclusions,
            'ring_edges_km': RADII,
            'kappa_bounds': KAPPA_BOUNDS,
            'noise_floor': NOISE_FLOOR,
            'samples_per_kernel': args.samples,
            'kernels': sum(
                sum('atmosphere' in v for v in p['stress'].values())
                for p in points.values()
            ),
            'previous_satellite_sha256': file_hash(args.previous_satellite),
            'boundaries_sha256': file_hash(args.boundaries),
        },
    )


if __name__ == '__main__':
    main()
