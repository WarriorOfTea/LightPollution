"""Compare the Berlin angular laws with clear-night observations in other cities."""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from hashlib import sha256
from itertools import combinations
from pathlib import Path

import numpy as np

from lightpollution.adapters.spectral_library import load_spectral_library
from lightpollution.adapters.storage import file_hash, read_json, write_json
from lightpollution.data import Bounds
from lightpollution.models.all_orders import AllOrderSettings
from lightpollution.models.boundary_orders import integrate_boundary_variants
from lightpollution.models.spectra import band_integral, wavelength_atmosphere
from lightpollution.physics import Atmosphere, Observer
from compare_selected_aod_to_sqm import (
    ARCSEC2_PER_SR, SKY_HASH, VEGA_HASH, VEGA_MAG, load_fits_spectrum,
)
from run_berlin_propagation import BerlinBoundary, FORMS, map_layers, pair_class
from run_city_source_transport import grid_city_indices


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'artifacts' / 'multicity-propagation-20261126'
SPECTRA = ROOT / 'artifacts' / 'photometer-sqm-residuals-20261126'
WAVELENGTH_EDGES = (380., 450., 490., 540., 585., 630., 690., 740., 780.)
APERTURE_RADIUS = {'photopic': 10., 'sqm': 10., 'tess': 8.5}
TERM_NAMES = ('rest', 'city_lambert', 'city_uniform', 'city_off_h1', 'city_off_h2',
              'within_3km', 'within_10km', 'monthly_lambert')
LAMPS = ('HP1', 'LED-B2', 'equal_dnb')
FORMS_WITH_REST = {name: np.array([1, *city, 0, 0, 0]) for name, city in FORMS.items()}


def spectral_inputs(vega_path, sky_path):
    library = load_spectral_library()
    dnb_wave, dnb = np.loadtxt(SPECTRA / 'viirs-dnb-rsr.dat', unpack=True)
    sqm_wave, *sqm = np.loadtxt(SPECTRA / 'sqm-response.csv', delimiter=',', skiprows=1, unpack=True)
    tess_wave, *tess = np.loadtxt(DATA / 'tess-response.csv', delimiter=',', skiprows=1, unpack=True)
    vegas, vega = load_fits_spectrum(vega_path, VEGA_HASH, lambda _, flux: .01*flux)
    skies, sky = load_fits_spectrum(sky_path, SKY_HASH, lambda wave, flux: flux/wave)
    responses = {
        'photopic': (library.wavelengths, library.responses['photopic']),
        'sqm': (sqm_wave, np.mean(sqm, axis=0)),
        'tess': (tess_wave, np.mean(tess, axis=0)),
    }
    wave = np.unique(np.r_[library.wavelengths, dnb_wave, sqm_wave, tess_wave, WAVELENGTH_EDGES])
    wave = wave[(wave >= WAVELENGTH_EDGES[0]) & (wave <= WAVELENGTH_EDGES[-1])]
    dnb_band = np.interp(wave, dnb_wave, dnb, left=0, right=0)
    lamps = {}
    for name in LAMPS[:2]:
        f = np.interp(wave, library.wavelengths, library.templates[name])
        lamps[name] = f/band_integral(wave, f, dnb_band)
    lamps['equal_dnb'] = (lamps['HP1'] + lamps['LED-B2'])/2
    sky_photopic = band_integral(skies, sky, np.interp(skies, *responses['photopic'], left=0, right=0))
    checks = {}
    for name, band in responses.items():
        zeropoint = band_integral(vegas, vega, np.interp(vegas, *band, left=0, right=0))
        natural = band_integral(skies, sky, np.interp(skies, *band, left=0, right=0)) / sky_photopic
        checks[name] = {'vega_zero_W_m2_sr': float(zeropoint*ARCSEC2_PER_SR*10**(.4*VEGA_MAG)),
                        'natural_per_mcd_m2_W_m2_sr': float(natural/683000)}
    ab_wave = np.linspace(350., 1000., 6501)
    ab = 3631e-26 * 299792458/(ab_wave*1e-9)**2*1e-9 * ARCSEC2_PER_SR
    tess_ab = [band_integral(ab_wave, ab, np.interp(ab_wave, tess_wave, curve)) for curve in tess]
    np.testing.assert_allclose(tess_ab, [521.8, 516.9], rtol=.015)
    bins = []
    for lo, hi in zip(WAVELENGTH_EDGES[:-1], WAVELENGTH_EDGES[1:], strict=True):
        chosen = (lo <= wave) & (wave <= hi)
        x = wave[chosen]
        y = lamps['equal_dnb'][chosen]
        node = np.trapezoid(x*y, x)/np.trapezoid(y, x)
        weights = {name: {lamp: float(1e-5*band_integral(x, f[chosen], np.interp(x, *band, left=0, right=0)))
                          for lamp, f in lamps.items()} for name, band in responses.items()}
        bins.append({'wavelength_nm': float(node), 'W_m2_sr_per_dnb': weights})
    return bins, checks, list(map(float, tess_ab))


# Workers share one source map per period and keep independent seeds per site and band.
_WORKER_BOUNDARY = None
_WORKER_BINS = None


def setup_worker(boundary, bins):
    global _WORKER_BOUNDARY, _WORKER_BINS
    _WORKER_BOUNDARY, _WORKER_BINS = boundary, bins


def geometry_key(row):
    return (row['period'], row['atmosphere'], row['latitude'], row['longitude'],
            row['elevation_m'], APERTURE_RADIUS[row['band']])


def run_transport(key, atmosphere_case, reference_m, samples):
    _, _, lat, lon, elevation, aperture = key
    height = elevation - reference_m
    if height <= 0:
        raise ValueError(f'Observer has nonpositive height above source shell: {key}')
    observer = Observer(lat, lon, height)
    params = atmosphere_case['parameters']
    atmosphere = Atmosphere(**params, rayleigh_multiplier=float(np.exp(-reference_m/8000)))
    boundary = _WORKER_BOUNDARY.at(observer)
    bands = []
    for wave in _WORKER_BINS:
        wavelength = wave['wavelength_nm']
        at = wavelength_atmosphere(atmosphere, wavelength, atmosphere_case['angstrom'])
        if 'phase_nodes' in atmosphere_case:
            at = replace(at, aerosol_g=float(np.interp(wavelength, *atmosphere_case['phase_nodes'])))
        seed = int.from_bytes(sha256(f'{key}/{wavelength}'.encode()).digest()[:4], 'little')
        result = integrate_boundary_variants(observer, boundary, 0., 90., at,
            AllOrderSettings(samples=samples, seed=seed, recorded_orders=1), aperture_radius_deg=aperture)
        columns = ('all_order_radiance', 'mean_covariance', 'effective_score_samples', 'max_scattering_order_seen')
        bands.append({name: result[name] for name in columns} | {'seed': seed})
    return bands


def project(row, runs, bins, calibration, background_mcd, lamp='equal_dnb'):
    band = row['band']
    norm = calibration[band]
    radiances = np.array([item['all_order_radiance'] for item in runs])
    covariances = np.array([item['mean_covariance'] for item in runs])
    weights = np.array([item['W_m2_sr_per_dnb'][band][lamp] for item in bins])
    terms = weights @ radiances
    covariance = np.tensordot(weights**2, covariances, axes=1)
    natural = background_mcd*norm['natural_per_mcd_m2_W_m2_sr']
    observed = row['value'] / 683000 if band == 'photopic' else norm['vega_zero_W_m2_sr']*10**(-.4*row['value'])
    results = {}
    for name, coef in FORMS_WITH_REST.items():
        art = float(coef @ terms)
        total = art + natural
        display = total*683000 if band == 'photopic' else -2.5*np.log10(total/norm['vega_zero_W_m2_sr'])
        results[name] = {'prediction': float(display), 'total_W_m2_sr': float(total),
                         'artificial_W_m2_sr': art, 'natural_fraction': float(natural/total),
                         'city_fraction': float((art-terms[0])/art),
                         'se_W_m2_sr': float(np.sqrt(max(0., coef @ covariance @ coef))),
                         'relative_pct': float(100*(total/observed-1))}
    return results, dict(zip(TERM_NAMES, map(float, terms), strict=True)), float(observed)


def summarize(rows, key='prediction'):
    if not rows:
        return None
    matches = [(a,b) for a,b in combinations(rows, 2) if a['night']==b['night'] and a['band']==b['band']
               and a['site']!=b['site']]
    result = {}
    for form in FORMS:
        error = np.array([r[key][form]['relative_pct'] for r in rows])
        pair_values = [(pair_class(a['observed_W_m2_sr'], b['observed_W_m2_sr'], .10),
                        pair_class(a[key][form]['total_W_m2_sr'], b[key][form]['total_W_m2_sr'], .10))
                       for a,b in matches]
        result[form] = {'mae_pct': float(np.mean(np.abs(error))), 'bias_pct': float(np.mean(error)),
                        'n_distinct_pairs': int(sum(a!=0 for a,b in pair_values)),
                        'n_correct_distinct_pairs': int(sum(a!=0 and a==b for a,b in pair_values)),
                        'n_close_pairs': int(sum(a==0 for a,b in pair_values)),
                        'n_correct_close_pairs': int(sum(a==b==0 for a,b in pair_values))}
    return {'n':len(rows), 'n_nights':len(set(r['night'] for r in rows)), 'cases':result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('maps', 'boundaries', 'vega', 'sky', 'output'):
        parser.add_argument(field, type=Path)
    parser.add_argument('--workers', type=int, default=5)
    parser.add_argument('--samples', type=int, default=98304)
    args = parser.parse_args()
    observation_path = DATA/'observations.json'
    boundaries = args.boundaries
    inputs = read_json(observation_path)
    geometry = read_json(boundaries)
    bins, calibration, tess_ab = spectral_inputs(args.vega, args.sky)
    groups = defaultdict(list)
    for row in inputs['rows']:
        groups[row['period']].append(row)
    transport, source_diagnostics, rows = {}, {}, []
    for period, sites in groups.items():
        city = sites[0]['city']; settings = inputs['cities'][city]
        bounds = Bounds(*settings['bounds'])
        year, month = map(int, period.split('-')[-2:])
        annual = f'{city.lower()}-{year}'
        ground_map, hashes = map_layers(args.maps, period, month, bounds=bounds, year=year)
        backup_map, backup_hashes = map_layers(args.maps, annual, None, bounds=bounds, year=year)
        city_mask = grid_city_indices(bounds, {city:geometry[city]}, (city,)) == 1
        boundary = BerlinBoundary(ground_map, backup_map, city_mask, bounds=bounds)
        del ground_map, backup_map
        source_diagnostics[period] = {
            'tile_sha256': hashes | backup_hashes, 'bounds': asdict(bounds),
            **{k.replace('berlin','city'): v for k,v in boundary.map_diagnostics.items()},
        }
        keys = list(dict.fromkeys(geometry_key(row) for row in sites))
        period_runs = {}
        print(period, len(keys), 'distinct fields of view', flush=True)
        with ProcessPoolExecutor(args.workers, initializer=setup_worker, initargs=(boundary,bins)) as pool:
            futures = {pool.submit(run_transport,k,inputs['atmospheres'][k[1]],settings['source_reference_m'],args.samples):k
                       for k in keys}
            for task in as_completed(futures):
                k = futures[task]
                period_runs[k] = task.result()
                print(period,len(period_runs),'/',len(keys),flush=True)
        del boundary
        indexed_runs = {k: f'{period}/{i+1}' for i,k in enumerate(keys)}
        transport.update({indexed_runs[k]: {'geometry':list(k), 'bands':period_runs[k]} for k in keys})
        for row in sites:
            k = geometry_key(row); estimates = period_runs[k]
            baseline, art, measured = project(row, estimates, bins, calibration, .17)
            sensitivity = {key:project(row, estimates, bins, calibration, background, lamp)[0]
                           for key,background,lamp in (('bright_natural',.25,'equal_dnb'),('sodium',.17,'HP1'),('led',.17,'LED-B2'))}
            rows.append(row | {'run':indexed_runs[k], 'prediction':baseline, 'artificial_terms_W_m2_sr':art,
                                'observed_W_m2_sr':measured, 'sensitivity':sensitivity})
    summaries = {}
    for city in inputs['cities']:
        selected = [r for r in rows if r['city']==city]
        dominant = [r for r in selected if all(r['prediction'][k]['city_fraction']>.5 for k in ('lambert','off_h1','off_h2'))]
        summaries[city] = {'all':summarize(selected), 'city_dominant':summarize(dominant),
            'bands':{b:summarize([r for r in selected if r['band']==b]) for b in sorted(set(r['band'] for r in selected))},
            'nights':{night:summarize([r for r in selected if r['night']==night]) for night in sorted(set(r['night'] for r in selected))}}
    hashes = {p.name:file_hash(p) for p in (observation_path,boundaries,DATA/'tess-response.csv',
        SPECTRA/'sqm-response.csv',SPECTRA/'viirs-dnb-rsr.dat',args.vega,args.sky)}
    write_json(args.output, {'samples_per_band':args.samples,'spectral_bands':bins,'calibration':calibration,
        'tess_ab_checks':tess_ab, 'input_sha256':hashes, 'map_diagnostics':source_diagnostics,
        'atmospheres':inputs['atmospheres'],'runs':transport,'rows':rows,'metrics':summaries})


if __name__ == '__main__':
    main()
