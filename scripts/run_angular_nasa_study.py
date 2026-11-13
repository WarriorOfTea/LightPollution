"""Audit angular NASA inputs and propagate bounded scenarios through all orders.

No network or credentials are used. Pinned local HDF5 files are independently
read/hashed and compared with the previously prepared regional caches.
"""

import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime
import json
from pathlib import Path
import resource
from zoneinfo import ZoneInfo

import numpy as np

from lightpollution.adapters.storage import file_hash, write_json
from lightpollution.data import COMPOSITES, good_mask, load_region, read_region
from lightpollution.models.all_orders import AllOrderSettings
from lightpollution.models.angular_sources import AngularGridBoundary, bin_horizontal_factor, eta_from_ratio
from lightpollution.models.boundary_orders import integrate_boundary_variants, paired_ratio
from lightpollution.physics import Atmosphere, Observer, cell_areas

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    'madrid-2015-01': ('data/seasonal-ground-20261110/madrid-2015-01/source-grid.npz', 'madrid-2015-01', 2015, 1),
    'madrid-2015-07': ('data/seasonal-ground-20261110/madrid-2015-07/source-grid.npz', 'madrid-2015-07', 2015, 7),
    'madrid-2017-03': ('data/seasonal-ground-20261110/madrid-2017-03/source-grid.npz', 'madrid-2017-03', 2017, 3),
    'madrid-2017-11': ('data/seasonal-ground-20261110/madrid-2017-11/source-grid.npz', 'madrid-2017-11', 2017, 11),
    'petersburg-2025': ('artifacts/petersburg-2025/source-grid.npz', '2025', 2025, None),
    'petersburg-2025-01': ('data/seasonal-ground-20261110/petersburg-2025-01/source-grid.npz', '2025-01', 2025, 1),
    'petersburg-2025-07': ('data/seasonal-ground-20261110/petersburg-2025-07/source-grid.npz', '2025-07', 2025, 7),
    'saarland-2014': ('artifacts/photo-comparison/saarland-2014/source-grid.npz', '2014-saarland', 2014, None),
}
SOURCE_FILES = ['scripts/run_angular_nasa_study.py', 'src/lightpollution/models/angular_sources.py',
    'src/lightpollution/models/boundary_orders.py', 'src/lightpollution/models/all_orders.py',
    'src/lightpollution/models/scattering_orders.py', 'src/lightpollution/physics.py', 'src/lightpollution/data.py']


def statistics(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not len(values):
        return {'count': 0}
    return dict(zip(('p05', 'p25', 'median', 'p75', 'p95'), map(float, np.percentile(values, [5, 25, 50, 75, 95]))), count=int(len(values)), mean=float(values.mean()))


def extract(arrays):
    maps = {p: arrays[p+'_Composite_Snow_Free'] for p in COMPOSITES}
    land = np.isin(arrays['Land_Water_Mask'], [0, 1, 5])
    masks = {p: land & good_mask(*(arrays[p+'_Composite_Snow_Free'+s] for s in ('', '_Quality', '_Num'))) for p in COMPOSITES}
    return maps, land, masks, np.logical_and.reduce(list(masks.values()))


def audit_dataset(name, output):
    cache, folder, year, month = DATASETS[name]
    arrays, bounds, metadata = load_region(ROOT/cache)
    fresh, attrs = read_region(ROOT/'data/blackmarble'/folder, bounds, year, month)
    # Revalidate every decoded field, including std/count/QA and land/water.
    for key, value in fresh.items():
        np.testing.assert_array_equal(value, arrays[key], err_msg=f'{name}:{key}')
    expected = {r['filename']: r['sha256'] for r in metadata['inputs']}
    actual = {r['filename']: r['sha256'] for r in attrs.values()}
    if actual != expected:
        raise ValueError(f'Pinned raw hashes changed for {name}.')
    maps, land, masks, common = extract(arrays)
    area = np.broadcast_to(cell_areas(bounds.latitude)[:, None], bounds.shape)
    n, o, a = [maps[p] for p in COMPOSITES]
    sums = {p: float((maps[p][common]*area[common]).sum()) for p in COMPOSITES}
    report = {'dataset': name, 'cache': cache, 'cache_sha256': file_hash(ROOT/cache),
        'raw_files': [{'filename': k, 'sha256': v} for k, v in actual.items()],
        'bounds': asdict(bounds), 'total_grid_cells': int(common.size), 'known_land_coast_cells': int(land.sum()),
        'land_water_missing_cells': int(np.count_nonzero(~np.isfinite(arrays['Land_Water_Mask']))),
        'accepted_land_coast_cells': {p: int(m.sum()) for p, m in masks.items()},
        'common_three_angle_cells': int(common.sum()), 'radiance_area_sums': sums,
        'area_sum_ratios_to_near': {p: v/sums['NearNadir'] if sums['NearNadir'] else None for p, v in sums.items()},
        'comparisons': {}, 'observations_on_common': {}, 'std_on_common': {},
        'negative_nonfill_land_radiance': {p: int(np.count_nonzero(land & np.isfinite(maps[p]) & (maps[p] < 0))) for p in COMPOSITES},
        'zero_near_positive_off': int(np.count_nonzero(common & (n == 0) & (o > 0))),
        'zero_off_positive_near': int(np.count_nonzero(common & (o == 0) & (n > 0))),
        'all_outside_near_off_envelope': int(np.count_nonzero(common & ((a < np.minimum(n, o)) | (a > np.maximum(n, o))))),
        'all_envelope_denominator': int(common.sum()),
        'raw_cache_fields_verified': len(fresh), 'raw_hashes_verified': len(actual),
        'period': {'year': year, 'month': month},
        'limitations': ['The three composites overlap in observations and contain different time/angle samples.',
            'Pixel ratios include temporal variability, noise, gridding and residual corrections; not a unique angular emission inversion.',
            'Only known land/coast classes 0,1,5; strict QA=0 and Num>3 in every angle.',
            'Pixel sampling is 15 arcsec, not an independent optical footprint; 0.5 nW/(cm2 sr) censoring matters.']}
    for p in COMPOSITES:
        report['observations_on_common'][p] = statistics(arrays[p+'_Composite_Snow_Free_Num'][common])
        report['std_on_common'][p] = statistics(arrays[p+'_Composite_Snow_Free_Std'][common])
    for threshold in (1., 5., 20.):
        mask = common & (n >= threshold) & (o >= threshold)
        ratio = o[mask]/n[mask]
        eta = eta_from_ratio(ratio)
        report['comparisons'][str(threshold)] = {'threshold_both_nW_cm2_sr': threshold,
            'denominator': int(mask.sum()), 'off_over_near': statistics(ratio),
            'off_less_than_near': int(np.count_nonzero(ratio < 1)),
            'off_at_least_20pct_lower': int(np.count_nonzero(ratio < .8)),
            'off_at_least_20pct_higher': int(np.count_nonzero(ratio > 1.2)),
            'eta_outside_0_1': int(np.count_nonzero((eta < 0) | (eta > 1))),
            'eta_within_0_1': statistics(eta[(eta >= 0) & (eta <= 1)])}
    blocks = []
    for block in (4, 12, 24):
        for row in range(0, common.shape[0], block):
            for col in range(0, common.shape[1], block):
                ix = np.s_[row:row+block, col:col+block]
                m, l, ar = common[ix], land[ix], area[ix]
                count, land_count = int(m.sum()), int(l.sum())
                if count < 4 or count < .8*land_count:
                    continue
                total_area = ar[m].sum()
                sn, so, sa = [float((value[ix][m]*ar[m]).sum()) for value in (n, o, a)]
                if min(sn, so)/total_area < 5:
                    continue
                blocks.append({'block_pixels': block, 'row': row, 'col': col, 'common_cells': count,
                    'land_cells': land_count, 'latitude_deg': float(bounds.latitude[row:min(row+block, len(bounds.latitude))].mean()),
                    'longitude_deg': float(bounds.longitude[col:min(col+block, len(bounds.longitude))].mean()),
                    'near_mean': float(sn/total_area), 'off_mean': float(so/total_area),
                    'all_mean': float(sa/total_area), 'off_over_near': so/sn,
                    'eta_diagnostic': float(eta_from_ratio(so/sn))})
    report['blocks'] = {str(block): {'denominator_blocks': len(b := [r for r in blocks if r['block_pixels'] == block]),
        'off_over_near': statistics([r['off_over_near'] for r in b]),
        'off_less_than_near': sum(r['off_over_near'] < 1 for r in b),
        'eta_outside_0_1': sum(not 0 <= r['eta_diagnostic'] <= 1 for r in b)} for block in (4, 12, 24)}
    if blocks:
        with (output/f'{name}-blocks.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(blocks[0]), lineterminator='\n')
            writer.writeheader(); writer.writerows(blocks)
    write_json(output/f'{name}-audit.json', report)
    return arrays, bounds, common, report


def make_boundary(arrays, bounds, support):
    maps, *_ = extract(arrays)
    area = np.broadcast_to(cell_areas(bounds.latitude)[:, None], bounds.shape)
    n, o, a = [maps[p] for p in COMPOSITES]
    ratio = float((o[support]*area[support]).sum()/(n[support]*area[support]).sum())
    eta = float(eta_from_ratio(ratio))
    if not 0 <= eta < 1:
        raise ValueError('The regional mixture is incompatible; it must not be clipped into the model.')
    return AngularGridBoundary(bounds, n, o, a, support, eta), ratio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/angular-nasa-study-20261113')
    parser.add_argument('--samples', type=int, default=2097152)
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--case', help='Run only one named case (resume keeps other files).')
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    (output/'cases').mkdir(exist_ok=True)
    code = {p: file_hash(ROOT/p) for p in SOURCE_FILES}
    retained, audits = {}, {}
    for name in DATASETS:
        arrays, bounds, common, audit = audit_dataset(name, output)
        audits[name] = audit
        if name in ('madrid-2017-03', 'madrid-2017-11', 'petersburg-2025', 'saarland-2014'):
            retained[name] = (arrays, bounds, common)
        print(json.dumps({'audit_completed': name, 'common': audit['common_three_angle_cells'],
            'off_near_ratio': audit['area_sum_ratios_to_near']['OffNadir']}, ensure_ascii=False), flush=True)
    write_json(output/'input-summary.json', audits)
    if args.audit_only:
        return
    madrid_support = retained['madrid-2017-03'][2] & retained['madrid-2017-11'][2]
    boundaries, inputs = {}, {}
    for name, (arrays, bounds, common) in retained.items():
        support = madrid_support if name.startswith('madrid') else common
        boundary, ratio = make_boundary(arrays, bounds, support)
        boundaries[name] = boundary
        inputs[name] = {'common_cells_for_transport': int(support.sum()), 'common_support_sha256': __import__('hashlib').sha256(support.tobytes()).hexdigest(),
            'off_near_area_ratio_on_transport_support': ratio, 'regional_eta_uniform_solid_angle': boundary.regional_eta,
            'regional_eta_uniform_zenith_angle': float(eta_from_ratio(ratio, 'zenith_angle')),
            'normalisation': 'Raw spatial radiance proxy with one scale for all variants; no per-map sum or zenith renormalisation.',
            'maps': DATASETS[name][0], 'maps_sha256': audits[name]['cache_sha256']}
    cases = []
    for dataset in boundaries:
        locations = [('ucm', Observer(40.451111, -3.726111, 20.), 180.),
                     ('north', Observer(40.85, -3.726111, 20.), 180.)] if dataset.startswith('madrid') else (
            [('north', Observer(60.3, 30.315, 20.), 180.)] if dataset.startswith('petersburg') else
            [('saar', Observer(49.532563, 6.549911, 20.), 180.)])
        for label, obs, az in locations:
            for alt in (10., 45., 90.):
                cases.append({'id': f'{dataset}-{label}-alt{int(alt)}', 'dataset': dataset, 'observer': asdict(obs),
                    'azimuth_deg': az, 'altitude_deg': alt, 'aod': .1, 'seed': 9071, 'role': 'primary'})
    template = next(c for c in cases if c['id'] == 'madrid-2017-03-north-alt45')
    for aod in (.02, .3):
        cases.append({**template, 'id': template['id']+f'-aod{aod}', 'aod': aod, 'role': 'atmosphere_sensitivity'})
    for altitude in (10, 45):
        c = next(c for c in cases if c['id'] == f'madrid-2017-03-north-alt{altitude}')
        cases.append({**c, 'id': c['id']+'-seed19073', 'seed': 19073, 'role': 'independent_seed'})
    protocol = {'id': 'LP-NASA-angular-all-orders-1', 'created_at': datetime.now(ZoneInfo('Europe/Berlin')).isoformat(timespec='seconds'),
        'source_code_sha256': code, 'inputs': inputs, 'cases': cases, 'samples_per_case': args.samples,
        'bin_assumptions': {'near_deg': [0, 20], 'off_deg': [40, 60], 'all_angle': 'Overlapping composite, not a third independent angle.',
            'empirical': 'Near plateau 0-20; linear in mu 20-40; off plateau 40-60; linear in mu to 0/off/2off at 90.',
            'mixture': 'Existing k=4 intensity mixture converted to radiance. Uniform-solid-angle bin weights are a scenario.',
            'radiance_cap': 'Mixture radiance frozen above 85 or 88 deg. This explicitly changes the unbounded grazing law.',
            'near_bin_horizontal_factor': bin_horizontal_factor(0, 20), 'off_bin_horizontal_factor': bin_horizontal_factor(40, 60)},
        'shared_atmosphere': asdict(Atmosphere()), 'fixed_spectrum': '550 nm transfer of the DNB spatial proxy; no DNB-to-V spectral inversion.',
        'observer_height': 'All observers 20 m above the same spherical emitting surface; actual terrain/station MSL heights not represented.',
        'not_field_validation': True, 'absolute_calibration': None, 'moon_cloud_natural_ground_reflection_DSM': 'Disabled.',
        'support_policy': 'Intersection of all three angles, land/coast QA0 Num>3. Both Madrid 2017 months additionally share a temporal intersection.',
        'predeclared_checks': ['Replay cap exactly in unit tests for implicit/analog absorption.',
            'Independent seed compatibility in two directions using 4 combined standard errors.',
            'Per-variant positivity/bounds, outcome conservation, category/angular sum closure.',
            'MC marginal intervals withheld if positive paths or effective score samples <100.']}
    write_json(output/'protocol.json', protocol)
    for case in cases:
        if args.case and case['id'] != args.case:
            continue
        path = output/'cases'/f"{case['id']}.json"
        identity = {'case': case, 'samples': args.samples, 'code': code, 'input': inputs[case['dataset']]}
        if path.exists():
            previous = json.loads(path.read_text())
            if previous.get('identity') == identity:
                print(json.dumps({'cached_case': case['id']}), flush=True)
                continue
        result = integrate_boundary_variants(Observer(**case['observer']), boundaries[case['dataset']],
            case['azimuth_deg'], case['altitude_deg'], replace(Atmosphere(), aerosol_optical_depth=case['aod']),
            AllOrderSettings(samples=args.samples, seed=case['seed']))
        result['identity'] = identity
        result['case'] = case
        result['ratios_to_near'] = {label: paired_ratio(result, label, 'near_lambertian') for label in result['labels']}
        result['peak_rss_kib_process'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        write_json(path, result)
        print(json.dumps({'case_completed': case['id'], 'runtime_s': result['runtime_seconds'],
            'ratios': {p: result['ratios_to_near'][p]['ratio'] for p in result['labels']},
            'effective_scores': dict(zip(result['labels'], result['effective_score_samples']))}), flush=True)


if __name__ == '__main__':
    main()
