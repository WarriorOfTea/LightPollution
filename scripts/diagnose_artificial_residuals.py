"""Isolate source-map, emission-angle, and wavelength effects on Veneto SQM residuals."""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

import numpy as np

from lightpollution.adapters.spectral_library import load_spectral_library
from lightpollution.adapters.storage import file_hash, read_json, write_json
from lightpollution.data import Bounds, PIXELS_PER_DEGREE, good_mask, read_region
from lightpollution.models.all_orders import AllOrderSettings
from lightpollution.models.angular_sources import empirical_radiance
from lightpollution.models.boundary_orders import integrate_boundary_variants
from lightpollution.models.spectra import band_integral, wavelength_atmosphere
from lightpollution.physics import Atmosphere, Observer


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts'
SELECTION = ART / 'photometer-aod-selection-20261126' / 'selection-audit.json'
PREVIOUS = ART / 'photometer-three-models-20261126' / 'results.json'
PHOTOMETRY = ART / 'photometer-sqm-residuals-20261126'
HOLDOUTS = ART / 'city-source-calibration-20261126' / 'windows.json'
WIDE = Bounds(7.2, 42.4, 16.4, 49.2)
ANGSTROM = 1.3
COARSE_EDGES = np.array([380., 475., 530., 585., 640., 780.])
SPECTRAL_SAMPLES = 131072
BANDS = ('original', 'outside', 'angle_core', 'horizon_core', 'angle_outside', 'horizon_outside', 'annual_core_gap')
NEAR = 'NearNadir_Composite_Snow_Free'
OFF = 'OffNadir_Composite_Snow_Free'
PICKS = {
    'baseline': [1, 0, 0, 0, 0, 0, 0],
    'expanded': [1, 1, 0, 0, 0, 0, 0],
    'angle_h1': [0, 0, 1, 0, 0, 0, 0],
    'angle_h2': [0, 0, 1, 1, 0, 0, 0],
    'annual_gaps': [1, 0, 0, 0, 0, 0, 1],
    'expanded_angle_h1_gaps': [0, 0, 1, 0, 1, 0, 1],
    'expanded_angle_h2_gaps': [0, 0, 1, 1, 1, 1, 1],
}


def spectral_nodes(edges):
    library = load_spectral_library()
    dnb_wave, dnb = np.loadtxt(PHOTOMETRY / 'viirs-dnb-rsr.dat', unpack=True)
    sqm_wave, sq1, sq2 = np.loadtxt(PHOTOMETRY / 'sqm-response.csv', delimiter=',', skiprows=1, unpack=True)
    axis = np.unique(np.r_[library.wavelengths, dnb_wave, sqm_wave])
    axis = axis[(axis >= 380) & (axis <= 780)]
    dnb_rsr = np.interp(axis, dnb_wave, dnb, left=0, right=0)
    lamps = [np.interp(axis, library.wavelengths, library.templates[key]) for key in ('LED-B2', 'HP1')]
    mixture = sum(lamp / band_integral(axis, lamp, dnb_rsr) for lamp in lamps) / 2
    response = mixture * np.interp(axis, sqm_wave, (sq1 + sq2) / 2)
    nodes = []
    # Integrate the piecewise-linear detected energy from the archived projection.
    # The first moment divides the same energy to obtain the representative wavelength.
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        wave = np.r_[lo, axis[(axis > lo) & (axis < hi)], hi]
        y = np.interp(wave, axis, response)
        w = np.trapezoid(y, wave)
        first_moment = ((np.diff(wave)/6)*(y[:-1]*(2*wave[:-1]+wave[1:])
                                            + y[1:]*(wave[:-1]+2*wave[1:]))).sum()
        nodes.append((float(first_moment/w), float(w)))
    return nodes


def choose_windows(selection, previous, held_out):
    cases = selection['station_nights']
    by_site = defaultdict(list)
    for k, row in previous.items():
        if held_out[k]['split'] == 'holdout':
            by_site[row['site']].append(k)
    chosen, used_nights = {}, set()
    for site, keys in sorted(by_site.items(), key=lambda item: (len(item[1]), item[0])):
        for monthly in (True, False):
            for sign in (1, -1):
                eligible = [k for k in keys if (len(previous[k]['map_period']) == 7) == monthly
                            and previous[k]['night'] not in used_nights]
                key = min(eligible, key=lambda k: (sign*previous[k]['aod_regional'], cases[k]['weather_max_cloud'], k))
                chosen[key] = {'site': site, 'monthly': monthly, 'aod_selection': 'low' if sign == 1 else 'high'}
                used_nights.add(previous[key]['night'])
    return chosen


def load_map(nasa, period, bounds, metadata):
    year, *month = map(int, period.split('-'))
    arrays, tiles = read_region(nasa / period, bounds, year, month[0] if month else None)
    if {k: v['sha256'] for k, v in tiles.items()} != {k: v['sha256'] for k, v in metadata[period]['tiles'].items()}:
        raise ValueError(f'Changed Black Marble input for {period}.')
    land = np.isin(arrays['Land_Water_Mask'], [0, 1, 5])
    near = land & good_mask(arrays[NEAR], arrays[NEAR+'_Quality'], arrays[NEAR+'_Num'])
    common = near & good_mask(arrays[OFF], arrays[OFF+'_Quality'], arrays[OFF+'_Num'])
    return np.where(near, arrays[NEAR], 0.), np.where(common, arrays[OFF], 0.), near, common


class SourceBoundary:
    labels = BANDS

    def __init__(self, near, off, known, common, core, annual):
        self.near, self.off, self.known, self.common = near, off, known, common
        self.core = ((WIDE.latitude[:, None] > core.south) & (WIDE.latitude[:, None] < core.north)
                    & (WIDE.longitude[None, :] > core.west) & (WIDE.longitude[None, :] < core.east))
        self.annual_fill = np.zeros(WIDE.shape)
        if annual is not None:
            self.annual_fill[self.core] = (annual.ravel() * ~known[self.core])
        maximum = max(near.max(), off.max(), self.annual_fill.max())
        self.upper_bounds = np.full(len(self.labels), maximum)
        self.diagnostics = {'source_cells_core': int((known & self.core).sum()), 'source_cells_wide': int(known.sum()),
                            'core_valid_off_fraction_of_near_radiance': float(near[self.core & common].sum()
                                                                            / near[self.core].sum()),
                            'core_filled_positive_cells': int(np.count_nonzero(self.annual_fill)),
                            'annual_fill_available': annual is not None}

    def sample(self, points, outgoing):
        x, y, z = points.T
        latitude = np.rad2deg(np.arctan2(z, np.hypot(x, y)))
        longitude = np.rad2deg(np.arctan2(y, x))
        row = np.floor((WIDE.north - latitude)*PIXELS_PER_DEGREE).astype(int)
        col = np.floor((longitude - WIDE.west)*PIXELS_PER_DEGREE).astype(int)
        n, m = WIDE.shape
        inside = (row >= 0) & (row < n) & (col >= 0) & (col < m)
        row, col = row.clip(0, n-1), col.clip(0, m-1)
        core = inside & self.core[row, col]
        outer = inside & ~core
        near, off = self.near[row, col], self.off[row, col]
        common = inside & self.common[row, col]
        mu = np.clip(np.einsum('ij,ij->i', points, outgoing) / np.linalg.norm(points, axis=1), 0, 1)
        angle = np.where(common, empirical_radiance(near, off, mu, 1.), near)
        slope = np.where(common, off*np.maximum(0., 1-2*mu), 0.)
        fill = self.annual_fill[row, col] * core
        codes = np.where(inside, 1 + (self.known[row, col] | (fill > 0)).astype(int), 0)
        return np.column_stack((near*core, near*outer, angle*core, slope*core, angle*outer, slope*outer, fill)), codes


def run_atmosphere(observer, boundary, atmosphere, samples, seed):
    result = integrate_boundary_variants(observer, boundary, 0., 90., atmosphere,
        AllOrderSettings(samples=samples, seed=seed, recorded_orders=1))
    return {k: result[k] for k in ('all_order_radiance', 'all_order_standard_error', 'mean_covariance',
                                   'effective_score_samples', 'positive_paths', 'max_scattering_order_seen')}


def run_one_period(job):
    period, cases, file_root, maps, nodes, fine_nodes, identity, workspace = job
    cache = workspace / f'{period}.json'
    if cache.is_file():
        previous = read_json(cache)
        if previous['identity'] == identity:
            return previous
    core = Bounds(**maps[period]['bounds'])
    annual_period = period[:4]
    annual = (load_map(file_root, annual_period, core, maps)[0]
              if len(period) == 7 and annual_period in maps else None)
    boundary = SourceBoundary(*load_map(file_root, period, WIDE, maps), core, annual)
    rows = {}
    for key, meta, record, position in cases:
        observer = Observer(*position)
        old = record['artificial']['regional']
        atm = Atmosphere(aerosol_optical_depth=record['aod_regional'])
        base = run_atmosphere(observer, boundary, atm, record['samples'], old['seed'])
        np.testing.assert_allclose(base['all_order_radiance'][0], old['all_order_radiance'][0], atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(base['all_order_standard_error'][0], old['all_order_standard_error'][0], atol=1e-12, rtol=1e-12)
        groups = {'spectral': nodes}
        if meta['refine']:
            groups['spectral_refined'] = fine_nodes
        out = {}
        for group, bands in groups.items():
            runs = []
            for wavelength, weight in bands:
                seed = int.from_bytes(sha256(f'{key}:{group}:{wavelength:.8f}'.encode()).digest()[:4], 'big')
                stats = run_atmosphere(observer, boundary, wavelength_atmosphere(atm, wavelength, ANGSTROM),
                                       SPECTRAL_SAMPLES, seed)
                runs.append({'wavelength_nm': wavelength, 'band_weight': weight, 'seed': seed, 'stats': stats})
            out[group] = runs
        rows[key] = meta | {'night': record['night'], 'utc': record['utc'], 'period': period, 'aod_550': record['aod_regional'],
                            'map_qa': boundary.diagnostics, 'grey_550': base, 'grey_seed': old['seed'],
                            'grey_samples': record['samples'], 'spectral_samples': SPECTRAL_SAMPLES} | out
    batch = {'identity': identity, 'rows': rows}
    write_json(cache, batch)
    return batch


def error_stats(observed, natural, art, uncertainty, baseline):
    ratios = (natural + art) / observed
    pct = 100*(ratios-1)
    return {'n': len(pct), 'mean_bias_pct': float(pct.mean()), 'mae_pct': float(np.abs(pct).mean()),
            'median_artificial_change_pct': float(100*np.median(art/baseline-1)),
            'p10_p90_artificial_change_pct': (100*np.quantile(art/baseline-1, [.1, .9])).tolist(),
            'median_mc_se_pct_of_observed': float(100*np.median(uncertainty/observed)),
            'underprediction_windows': int((pct < 0).sum())}


def summarize(photometry, selection, rows):
    archived = photometry['reference_windows']
    band = photometry['band_projections']['mean']
    flux = band['vega_zero_W_m2_sr']
    sky_conversion = photometry['johnson_v_vega_zero_W_m2_sr']*band['natural_SQM_over_V']
    coefficient = band['artificial_SQM_over_DNB']['equal_DNB_LED-B2_HP1'] * 1e-5
    deficit = {}
    for subset, keys in {'all_148': list(archived), 'bright_proxy': [k for k in archived if
            selection['station_nights'][k]['natural_plus_moon_fraction_proxy']['21.35'] <= .5]}.items():
        obs = np.array([flux*10**(-.4*archived[k]['observed_mag_sqm']) for k in keys])
        nat = np.array([archived[k]['natural_sqm_W_m2_sr'] for k in keys])
        entry = {}
        for model in ('standard', 'regional', 'spatial_vertical'):
            art = np.array([archived[k]['models'][model]['artificial_sqm_W_m2_sr'] for k in keys])
            natural_required = -2.5*np.log10((obs-art)/sky_conversion)
            entry[model] = {'natural_alone_would_require_V_median': float(np.median(natural_required)),
                            'natural_alone_would_need_brighter_than_V20_99': int((natural_required < 20.99).sum()),
                            'artificial_required_div_existing_model': {str(level): np.quantile(
                                (obs-nat*10**(.4*(21.6-level)))/art, [.1, .5, .9]).tolist()
                                for level in (20.99, 21.6, 22.1)}}
        deficit[subset] = {'windows': len(keys), 'nights': len({archived[k]['night'] for k in keys}), 'models': entry}
    output_rows = {}
    for key, row in rows.items():
        observed = archived[key]
        record = {'obs': flux*10**(-.4*observed['observed_mag_sqm']), 'nat': observed['natural_sqm_W_m2_sr'], 'model': {}}
        calculations = record['model']
        for name, counts in PICKS.items():
            means = np.asarray(row['grey_550']['all_order_radiance'])
            covar = np.asarray(row['grey_550']['mean_covariance'])
            calculations[name] = (float(coefficient*(means @ counts)),
                                  float(coefficient*np.sqrt(max(0., counts @ covar @ counts))))
        for spectral_group in ('spectral', 'spectral_refined'):
            if spectral_group not in row:
                continue
            for name in ('baseline', 'expanded_angle_h1_gaps', 'expanded_angle_h2_gaps'):
                weights = np.asarray([b['band_weight'] * 1e-5 for b in row[spectral_group]])
                means = np.asarray([b['stats']['all_order_radiance'] for b in row[spectral_group]])
                cov = np.asarray([b['stats']['mean_covariance'] for b in row[spectral_group]])
                v = np.asarray(PICKS[name])
                calculations[f'{spectral_group}_{name}'] = (float(weights @ means @ v),
                    float(np.sqrt(np.maximum(0, (weights**2) @ (v @ cov @ v)))))
        output_rows[key] = record
    aggregate = {}
    mask = {'pilots': list(rows), 'monthly_with_annual': [k for k, r in rows.items() if r['map_qa']['annual_fill_available']],
            'refinement': [k for k, r in rows.items() if r['refine']]}
    mask.update({f'monthly_{monthly}': [k for k, r in rows.items() if r['monthly'] == monthly] for monthly in (True, False)})
    for subset, keys in mask.items():
        if not keys:
            continue
        stats = {}
        for name in output_rows[keys[0]]['model']:
            if name.startswith('spectral_refined') and subset != 'refinement':
                continue
            if name == 'annual_gaps' and subset != 'monthly_with_annual':
                continue
            observed = np.array([output_rows[k]['obs'] for k in keys])
            natural = np.array([output_rows[k]['nat'] for k in keys])
            art, se = np.array([output_rows[k]['model'][name] for k in keys]).T
            old = np.array([output_rows[k]['model']['baseline'][0] for k in keys])
            stats[name] = error_stats(observed, natural, art, se, old)
        aggregate[subset] = stats
    return {'deficit': deficit, 'aggregate': aggregate, 'projection_by_window': output_rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('maps', type=Path)
    parser.add_argument('workspace', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    args.workspace.mkdir(parents=True, exist_ok=True)
    selection, archive, held_out = (read_json(p) for p in (SELECTION, PREVIOUS, HOLDOUTS))
    photometry = read_json(PHOTOMETRY / 'results.json')
    previous = archive['results']
    chosen = choose_windows(selection, previous, held_out)
    maps = selection['sources']['NASA']['monthly_annual']
    sites = selection['sources']['DARKERSKY4CE']['instruments']
    nodes = spectral_nodes(COARSE_EDGES)
    fine_nodes = spectral_nodes(np.sort(np.r_[COARSE_EDGES, (COARSE_EDGES[1:] + COARSE_EDGES[:-1]) / 2]))
    q = photometry['band_projections']['mean']['artificial_SQM_over_DNB']['equal_DNB_LED-B2_HP1']
    np.testing.assert_allclose([sum(w for _, w in n) for n in (nodes, fine_nodes)], q, rtol=1e-12, atol=1e-12)
    inputs = [SELECTION, PREVIOUS, HOLDOUTS, PHOTOMETRY / 'results.json',
              PHOTOMETRY / 'viirs-dnb-rsr.dat', PHOTOMETRY / 'sqm-response.csv']
    identity = {'runner': file_hash(Path(__file__)), 'inputs': {str(p.relative_to(ROOT)): file_hash(p) for p in inputs},
                'spectral_library': load_spectral_library().identity}
    groups = defaultdict(list)
    for key, meta in chosen.items():
        earlier = previous[key]
        pos = sites[meta['site']]
        groups[earlier['map_period']].append((key, meta | {'refine': meta['monthly'] and meta['aod_selection']=='low'},
            earlier, (pos['latitude'], pos['longitude'], pos['height_m'])))
    with ProcessPoolExecutor(args.workers) as pool:
        jobs = {pool.submit(run_one_period, (p, g, args.maps, maps, nodes, fine_nodes, identity, args.workspace)): p
                for p, g in groups.items()}
        rows = {}
        for task in as_completed(jobs):
            batch = task.result()['rows']
            rows.update(batch)
            print(jobs[task], len(batch), flush=True)
    rows = dict(sorted(rows.items()))
    numbers = summarize(photometry, selection, rows)
    payload = {'inputs': identity, 'wide_bounds': asdict(WIDE), 'angstrom_exponent': ANGSTROM, 'boundary_terms': BANDS,
               'grey_linear_variants': PICKS, 'nodes': nodes, 'refined_nodes': fine_nodes, 'rows': rows} | numbers
    write_json(args.output, payload)


if __name__ == '__main__':
    main()
