"""Compare three atmospheric inputs on frozen Veneto SQM windows."""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import itertools
import json

import numpy as np

from lightpollution.adapters.storage import file_hash, read_json, write_json, write_npz
from lightpollution.data import Bounds, read_region
from lightpollution.models.all_orders import AllOrderSettings
from lightpollution.models.background import BackgroundModel, background_config
from lightpollution.models.boundary_orders import integrate_boundary_variants
from lightpollution.models.observation import vertical_extinction_v
from lightpollution.moonlight import scattered_moonlight, v_mag_to_nanolambert
from lightpollution.observation import observation_context
from lightpollution.physics import Atmosphere, Observer
from prepare_italy_aerosol_profiles import Collection, G, HEIGHT_BINS, V1, V2, profile_areas, remap_xy
from run_cams_dutch_pilot import area_mean_aod
from run_photometer_all_orders import NearBoundary
from run_photometer_vertical_italy import VerticalGridAtmosphere

LABELS = ('standard', 'regional', 'spatial_vertical')
NATIVE_LAT = np.arange(43.5, 47.76, .25)
NATIVE_LON = np.arange(9.5, 14.01, .25)
NATIVE_SLICE = '[534:551][758:776]'


def midtime(case):
    return datetime.fromisoformat(case['start']) + timedelta(minutes=30)


def setup_profiles(selection, cache, destination, workers):
    cache.mkdir(parents=True, exist_ok=True)
    destination.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(4) as pool:
        datasets = dict(zip(V1 + V2, pool.map(lambda name: Collection(cache, name), V1 + V2), strict=True))

    def run(case):
        instant = midtime(case)
        name = instant.isoformat(timespec='minutes').replace(':', '')
        target = destination / (name + '.npz')
        if target.exists():
            return
        v2 = instant.year >= 2026
        chm, met = (datasets[k] for k in (V2 if v2 else V1))
        start = instant.replace(minute=0) if v2 else instant
        levels, count = ('[36:71]', 2) if v2 else ('[0:35]', 1)
        pm = 'pm25_rh35' if v2 else 'pm25_rh35_gcc'
        geo = dict(latitude=NATIVE_LAT, longitude=NATIVE_LON, spatial_slice=NATIVE_SLICE)
        pollution, chm_record = chm.get(start, [(pm, True)], levels, count, **geo)
        terrain, met_record = met.get(start, [('zl', True), ('phis', False)], levels, count, **geo)
        areas = np.empty((len(HEIGHT_BINS)-1, len(NATIVE_LAT), len(NATIVE_LON)))
        for i, j in itertools.product(range(len(NATIVE_LAT)), range(len(NATIVE_LON))):
            areas[:, i, j] = np.mean([
                profile_areas(terrain['zl'][k, ::-1, i, j], pollution[pm][k, ::-1, i, j],
                              terrain['phis'][k, i, j] / G) for k in range(count)], axis=0)
        evidence = {'version': 2 if v2 else 1, 'variable': pm, 'sources': [chm_record, met_record],
                    'timestamps_utc': [(start+timedelta(hours=k)).isoformat()+'Z' for k in range(count)]}
        write_npz(target, areas=areas, provenance=json.dumps(evidence))
        print(f'{instant}: GEOS-CF v{evidence["version"]}', flush=True)

    cases = {midtime(c): c for c in selection['station_nights'].values()}
    failures = {}
    with ThreadPoolExecutor(workers) as pool:
        jobs = {pool.submit(run, case): when for when, case in cases.items()}
        for job in as_completed(jobs):
            try:
                job.result()
            except (RuntimeError, ValueError, KeyError) as exc:
                failures[jobs[job].isoformat()] = str(exc)
                print('GEOS-CF gap', jobs[job], exc, flush=True)
    write_json(destination / 'failures.json', failures)


def aerosol_layers(areas, aod, latitude, longitude):
    remapped = remap_xy(areas, latitude, longitude, source_latitudes=NATIVE_LAT,
                        source_longitudes=NATIVE_LON)
    total = remapped.sum(axis=0)
    if np.any(total <= 0):
        raise ValueError('GEOS-CF gives an empty aerosol column.')
    layers = np.moveaxis(remapped / total * aod, 0, -1)
    np.testing.assert_allclose(layers.sum(axis=-1), aod, atol=1e-10, rtol=0)
    return layers


def run_group(job):
    period, nasa, metadata, group, cams, profiles, output, samples, run_key = job
    path = output / f'{period}.json'
    previous = read_json(path) if path.exists() else {}
    records = previous.get('results', {}) if previous.get('run_key') == run_key else {}
    profile_paths = {key: profiles/(midtime(case).isoformat(timespec='minutes').replace(':', '')+'.npz')
                     for key, case, _ in group}
    profile_hashes = {key: file_hash(p) if p.is_file() else None for key, p in profile_paths.items()}
    pending = [(key, case, pos) for key, case, pos in group
               if key not in records or records[key]['profile_sha256'] != profile_hashes[key]]
    if not pending:
        return records
    year, *month = map(int, period.split('-'))
    bounds = Bounds(**metadata['bounds'])
    arrays, tiles = read_region(nasa / period, bounds, year, month[0] if month else None)
    for tile, record in tiles.items():
        if record['sha256'] != metadata['tiles'][tile]['sha256']:
            raise ValueError(f'Black Marble mismatch in {period} {tile}.')
    lat, lon = [np.asarray(cams[k]) for k in ('latitude_deg', 'longitude_deg')]
    cache = cams['values_by_native_hour_utc']
    boundaries = {}
    standard_results = {row['site']: row['artificial']['standard'] for row in records.values()}
    for key, case, position in pending:
        instant = midtime(case)
        observer = Observer(position['latitude'], position['longitude'], position['height_m'])
        if case['site'] not in boundaries:
            boundaries[case['site']] = NearBoundary(arrays, observer, bounds, (0, 3, 10))
        boundary = boundaries[case['site']]
        t0, t1 = (np.asarray(cache[t]) for t in case['aod_native_times'])
        w = case['aod_time_weight_for_second']
        aod = (1-w)*t0+w*t1
        region = float(area_mean_aod(aod, lat, lon))
        if not np.isclose(region, case['aod_area_mean_midpoint'], atol=1e-10, rtol=0):
            raise ValueError(f'CAMS AOD mismatch on {key}.')
        profile = profile_paths[key]
        previous_case = records.get(key)
        standard = Atmosphere()
        settings = {
            'standard': standard,
            'regional': replace(standard, aerosol_optical_depth=region),
        }
        evidence = None
        if profile_hashes[key] is not None:
            with np.load(profile) as payload:
                layers = aerosol_layers(payload['areas'], aod, lat, lon)
                evidence = json.loads(payload['provenance'].item())
            evidence['prepared_sha256'] = profile_hashes[key]
            settings['spatial_vertical'] = VerticalGridAtmosphere(edges_m=tuple(HEIGHT_BINS),
                tau_layers=layers, latitudes=tuple(lat), longitudes=tuple(lon))
        utc = instant.isoformat()+'Z'
        sky = observation_context(observer, utc, 0, 90)
        if sky['sun']['altitude_deg'] > -18 or sky['moon']['altitude_deg'] > -6:
            raise ValueError(f'Unexpected Sun or Moon position in the primary sample: {key}.')
        background = BackgroundModel(background_config({'twilight': 'patat2006'}), observer, standard, 21.6)
        diffuse = background.evaluate(np.array([0.]), np.array([90.]), sky)[0]
        moon = sky['moon']
        moon_v = float(scattered_moonlight(moon['geocentric_phase_angle_deg'], moon['altitude_deg'], 90.,
            moon['target_separation_deg'], vertical_extinction_v(observer, standard),
            moon['topocentric_distance_km'])/v_mag_to_nanolambert(22.))
        measurements = {}
        for label, atmosphere in settings.items():
            if label == 'regional' and previous_case is not None:
                measurements[label] = previous_case['artificial'][label]
                continue
            if label == 'standard' and case['site'] in standard_results:
                measurements[label] = standard_results[case['site']]
                continue
            seed_scope = case['site']+'/'+period if label == 'standard' else key
            seed = int.from_bytes(hashlib.sha256((seed_scope+'/'+label).encode()).digest()[:4], 'big')
            result = integrate_boundary_variants(observer, boundary, 0, 90, atmosphere,
                AllOrderSettings(samples=samples, seed=seed))
            fields = ('all_order_radiance', 'all_order_standard_error', 'effective_score_samples',
                      'mean_covariance', 'ground_support_counts', 'max_scattering_order_seen')
            measurements[label] = {field: result[field] for field in fields} | {'seed': seed}
            if label == 'standard':
                standard_results[case['site']] = measurements[label]
        records[key] = {'site': case['site'], 'night': case['date'], 'utc': utc, 'map_period': period,
                        'sqm': case['sqm'], 'n_sqm': case['n_sqm'], 'samples': samples, 'aod_regional': region,
                        'natural_v': diffuse, 'moon_v_relative': moon_v, 'geos_cf': evidence,
                        'profile_sha256': profile_hashes[key],
                        'solar_altitude_deg': sky['sun']['altitude_deg'], 'lunar_altitude_deg': sky['moon']['altitude_deg'],
                        'artificial': measurements}
        print(f'{key}: {len(settings)}/3 models', flush=True)
        write_json(path, {'run_key': run_key, 'results': records})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('profiles', 'transport'))
    parser.add_argument('selection', type=Path)
    parser.add_argument('nasa', type=Path)
    parser.add_argument('workspace', type=Path)
    parser.add_argument('--samples', type=int, default=262144)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    source = read_json(args.selection)
    profiles = args.workspace/'profiles'
    if args.operation == 'profiles':
        setup_profiles(source, args.workspace/'geos-cf', profiles, args.workers)
        return
    groups = defaultdict(list)
    instruments = source['sources']['DARKERSKY4CE']['instruments']
    maps = source['sources']['NASA']['monthly_annual']
    groups_path = args.workspace/'groups'
    groups_path.mkdir(parents=True, exist_ok=True)
    for key, case in source['station_nights'].items():
        groups[case['black_marble_spatial_key']].append((key, case, instruments[case['site']]))
    run_key = {'selection': file_hash(args.selection), 'samples': args.samples,
               'runner': file_hash(Path(__file__))}
    jobs = [(period, args.nasa, maps[period], rows, source['sources']['CAMS_550'],
             profiles, groups_path, args.samples, run_key) for period, rows in sorted(groups.items())]
    with ProcessPoolExecutor(args.workers) as pool:
        results = {key: record for group in pool.map(run_group, jobs) for key, record in group.items()}
    write_json(args.workspace/'transport.json', {'selection_sha256': file_hash(args.selection),
                'radii_km': [0, 3, 10], 'models': LABELS, 'results': results})


if __name__ == '__main__':
    main()
