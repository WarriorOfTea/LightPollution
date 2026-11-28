"""Score the satellite radial-profile fits on consecutive views and ground photometry."""

import argparse
from pathlib import Path

import numpy as np

from diagnose_halo_ground import observation_key
from lightpollution.adapters.storage import file_hash, read_json, write_json
from run_halo_ground_validation import ENHANCED, HELD
from run_halo_radial_profiles import PROFILE_MODES
from summarize_halo_diagnosis import scores


def ground_row(row, replay, scene):
    terms = np.sum(replay['single_and_multiple'], axis=0)
    h2 = row['natural'] + terms[0] + terms[ENHANCED]
    np.testing.assert_allclose(h2, row['prediction']['off_h2']['linear'], rtol=3e-11)
    redistrib = terms[ENHANCED] - terms[HELD] - terms[-1]
    s = scene['core']['scale']
    estimates = {
        'h2': h2,
        'core': row['natural'] + terms[0] + s * terms[ENHANCED],
        'old_free_tail': row['prediction']['a1_anchored']['linear'],
    }
    for mode, fit in scene['fit'].items():
        if 'kappa' in fit:
            estimates[mode] = (
                row['natural']
                + terms[0]
                + s * (terms[ENHANCED] + fit['kappa'] * redistrib)
            )
    if 'joint' in scene['fit']:
        estimates['joint_fixed_power'] = h2 + scene['fit']['joint']['kappa'] * redistrib
    for mode, fit in scene['stress'].items():
        if fit['joint']:
            estimates['stress_' + mode] = (
                row['natural']
                + terms[0]
                + fit['scale'] * (terms[ENHANCED] + fit['joint']['kappa'] * redistrib)
            )
    return {
        **{
            k: row[k]
            for k in (
                'key',
                'city',
                'night',
                'site',
                'band',
                'distance_km',
                'calendar_offset_days',
                'city_dominant',
                'fitted_scene',
            )
        },
        'relative_pct': {
            k: 100 * (x / row['observed'] - 1) for k, x in estimates.items()
        },
    }


def satellite_score(scenes, key):
    scores = [v['fit'][key]['score'] for v in scenes.values()]
    result = {'scenes': len(scores)}
    for name in ('halo', 'radial', 'sides', 'core'):
        good = [v for v in scores if v.get(name + '_mae') is not None]
        if good:
            result[name + '_mae'] = float(np.mean([v[name + '_mae'] for v in good]))
            result[name + '_n'] = sum(v[name + '_n'] for v in good)
            result[name + '_scenes'] = len(good)
    result['halo_bias'] = float(np.mean([v['halo_bias'] for v in scores]))
    return result


def compare_satellite(scenes, mode):
    selection = {k: v for k, v in scenes.items() if mode in v['fit']}
    if selection:
        return {
            label: satellite_score(selection, label)
            for label in dict.fromkeys(('h2', 'core', mode))
        }
    return {}


def direction_measurements(scenes):
    """Descriptive residual profiles and comparable inner-to-outer decline."""
    output = {}
    for key, scene in scenes.items():
        profiles = scene['fit']['core']['profiles']
        item = {}
        for sector in sorted({p['sector'] for p in profiles}):
            slices = [p for p in profiles if p['sector'] == sector]
            corrected = [p['values'][0] - p['values'][3] for p in slices]
            slope, raw_slope = None, None
            if len(slices) >= 3:
                d = [p['values'][2] for p in slices]
                slope = -float(np.polyfit(d, corrected, 1)[0])
                raw_slope = -float(
                    np.polyfit(d, [p['values'][0] for p in slices], 1)[0]
                )
            by_ring = {p['ring']: y for p, y in zip(slices, corrected, strict=True)}
            ratio = None
            if by_ring.get(1, 0) > 0.02 and by_ring.get(3, 0) > 0.02:
                ratio = by_ring[3] / by_ring[1]
            item[sector] = {
                'bins': len(slices),
                'decline_nw_per_km': slope,
                'raw_decline_nw_per_km': raw_slope,
                'outer_to_inner_ratio': ratio,
            }
        output[key] = item
    return output


def angular_ranges(scenes):
    out = {}
    for key, s in scenes.items():
        a = s['angular']
        fractions = {
            name: (
                np.asarray(a['fraction_h2'])
                + t['joint']['kappa'] * np.asarray(a['fraction_per_kappa'])
            ).tolist()
            for name, t in s['stress'].items()
            if t['joint']
        }
        out[key] = {
            'power_equivalent_w': a['h2_equivalent_dnb_w'] * s['core']['scale'],
            'case_fractions': fractions,
            'reference_fractions': a['fraction_h2'],
            'cases_at_bound': sum(
                t['joint']['at_bound'] for t in s['stress'].values() if t['joint']
            ),
        }
    return out


def shape_resolution(scenes):
    """Size of the largest permitted model change in observable held-out bin means."""
    results = {}
    for key, s in scenes.items():
        lo, hi = (
            s['fit'][bound]['profiles'] for bound in ('lower_bound', 'upper_bound')
        )
        profile = [
            abs(p['values'][1] - q['values'][1]) for p, q in zip(lo, hi, strict=True)
        ]
        results[key] = {
            'bins': len(profile),
            'maximum_mean_bin_difference': max(profile),
            'typical_bin_difference': float(np.median(profile)),
        }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('profiles', 'previous_ground', 'ground_response', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    profiles = read_json(args.profiles)
    ground = read_json(args.previous_ground)
    saved = read_json(args.ground_response)
    if saved['previous_ground_sha256'] != file_hash(args.previous_ground):
        raise ValueError('Ground response and archive mismatch.')
    replays = {observation_key(v): v for v in saved['rows']}
    scenes = profiles['scenes']
    rows = [
        ground_row(r, replays[observation_key(r)], scenes[r['fitted_scene']])
        for r in ground['rows']
        if r['fitted_scene'] in scenes
    ]
    cases = {
        'all': rows,
        'same_utc_night': [r for r in rows if r['calendar_offset_days'] == 0],
    }
    for city in ('Berlin', 'Madrid', 'Tucson'):
        site_rows = [r for r in rows if r['city'] == city]
        cases[city] = site_rows
        for name, lo, hi in (('near', 0, 15), ('mid', 15, 35), ('far', 35, np.inf)):
            cases[f'{city}/{name}'] = [
                r for r in site_rows if lo <= r['distance_km'] < hi
            ]
        cases[f'{city}/dominant'] = [r for r in site_rows if r['city_dominant']]
    modes = (
        'core',
        *PROFILE_MODES,
        'joint_fixed_power',
        'paired_difference',
        'other_platform',
        'another_night',
    )
    matched = {}
    for m in modes:
        selected = [r for r in rows if m in r['relative_pct']]
        if selected:
            ss = scores(selected)
            matched[m] = {k: ss[k] for k in dict.fromkeys(('h2', 'core', m))}
    write_json(
        args.output,
        {
            'satellite': {
                m: compare_satellite(scenes, m)
                for m in (
                    'core',
                    *PROFILE_MODES,
                    'lower_bound',
                    'upper_bound',
                    'paired_difference',
                    'other_platform',
                    'another_night',
                )
            },
            'ground': {k: scores(v) for k, v in cases.items()},
            'matched_ground': matched,
            'directions': direction_measurements(scenes),
            'fractions': angular_ranges(scenes),
            'shape_resolution': shape_resolution(scenes),
            'rows': rows,
            'ground_n_previous': ground['observations'],
            'no_new_satellite_fit': [
                {'city': c, 'n': v['ground_n'], 'baseline': v['baseline_all']['off_h2']}
                for c, v in ground['metrics'].items()
                if c != 'All' and v['paired_n'] == 0
            ],
            'sources_sha256': {
                p.name: file_hash(p)
                for p in (args.profiles, args.previous_ground, args.ground_response)
            },
        },
    )


if __name__ == '__main__':
    main()
