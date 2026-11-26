"""Freeze matching photometry and CAMS dates for supplementary Italian stations."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from lightpollution.adapters.storage import file_hash, read_json, write_json
from lightpollution.data import Bounds
from lightpollution.physics import Atmosphere, Observer
from run_cams_dutch_pilot import area_mean_aod
from run_city_source_transport import CITY_CODES, LEGACY, SELECTION, old_veneto, solve_period


SITES = ('Arcugnano', 'Galzignano', 'Piadera', 'Arson')
NEARBY_CITIES = dict(zip(SITES, ('Vicenza', 'Padova', 'Pordenone', 'Treviso'), strict=True))
FIELD = 'aerosol_optical_depth'


def load_cached_grid(queries, stamps, latitude, longitude):
    found = {}
    coords = [(lat, lon) for lat in latitude for lon in longitude]
    selected = set(stamps)
    for start, end, files in queries:
        required = sorted(t for t in selected - found.keys() if start <= t[:10] <= end)
        if not required:
            continue
        ordered = [files[k] for k in range(0, len(coords), 22)]
        records = [row for path in ordered for row in read_json(path)]
        grids = {t: [] for t in required}
        for expected, record in zip(coords, records, strict=True):
            actual = (round(record['latitude'], 4), round(record['longitude'], 4))
            if expected != actual or record['utc_offset_seconds'] != 0:
                raise ValueError(f'Unexpected grid coordinate or time zone in {ordered[0].name}.')
            series = dict(zip(record['hourly']['time'], record['hourly'][FIELD], strict=True))
            for t in required:
                grids[t].append(series.get(t))
        for time, grid in grids.items():
            values = np.array(grid, dtype=float).reshape(len(latitude), len(longitude))
            if np.all(np.isfinite(values) & (values >= 0)):
                found[time] = (values, ordered)
    return found


def prepare(archive, satellite, previous_selection):
    source = read_json(SELECTION)
    weather = archive / 'flex-weather-passed.json'
    metadata = archive / 'initial.json'
    comparisons = read_json(LEGACY / 'comparison.json')['sites']
    existing = {(site, row['date']) for site, entry in comparisons.items() for row in entry['rows'] if row['weather']['passes']}
    if previous_selection:
        existing.update((row['site'].removeprefix('SQM_IT_'), row['night'])
                        for row in read_json(previous_selection)['cases'].values())
    stations = read_json(metadata)['stations']
    bounds_data = source['sources']['NASA']['monthly_annual']['2022']['bounds']
    bounds = Bounds(**bounds_data)
    cams = source['sources']['CAMS_550']
    lat, lon = (np.asarray(cams[k]) for k in ('latitude_deg', 'longitude_deg'))
    raw_files = defaultdict(dict)
    for path in sorted((archive / 'cams').glob('*.json')):
        parts = path.stem.split('-')
        if len(parts) == 9:
            raw_files[('-'.join(parts[2:5]), '-'.join(parts[5:8]))][int(parts[-1])] = path
    starts = set(range(0, len(lat)*len(lon), 22))
    queries = [(a, b, files) for (a, b), files in sorted(raw_files.items()) if files.keys() == starts]
    rows, stamps, maps, counts = defaultdict(list), set(), {}, {site: Counter() for site in SITES}
    for case in read_json(weather)['rows']:
        site = case['site'].removeprefix('SQM_IT_')
        if site not in counts or case['moon_max'] > -6 or case['sun_max'] > -18:
            continue
        key = (site, case['date'])
        counts[site]['weather_hours'] += 1
        if key in existing:
            continue
        start = datetime.fromisoformat(case['start'])
        options = (case['start'][:7], case['start'][:4])
        period = next((p for p in options if (satellite / p).is_dir() and
                       all(len(list((satellite / p).glob(f'*.{tile}.002.*.h5'))) == 1 for tile in bounds.tiles())), None)
        if period is None:
            counts[site]['no_complete_NASA_map_hours'] += 1
            continue
        lo = start.replace(hour=start.hour//3*3)
        hi = lo + timedelta(hours=3)
        endpoints = [v.isoformat(timespec='minutes') for v in (lo, hi)]
        weight = ((start - lo).total_seconds() + 1800) / 10800
        stamps.update(endpoints)
        rows[key].append((case, period, endpoints, weight))
    grid = load_cached_grid(queries, stamps, lat, lon)
    records, used_cams, used_raw = {}, set(), set()
    for (site, night), entries in sorted(rows.items()):
        entries.sort(key=lambda entry: (len(entry[1]) != 7, not entry[0]['qualified_metar'],
            entry[0]['weather_max_cloud'], (int(entry[0]['start'][11:13]) - 22) % 24, entry[0]['start']))
        for case, period, endpoints, weight in entries:
            if any(t not in grid for t in endpoints):
                continue
            a, b = (grid[t] for t in endpoints)
            used_cams.update(a[1] + b[1])
            if period not in maps:
                tiles = {tile: next((satellite / period).glob(f'*.{tile}.002.*.h5')) for tile in bounds.tiles()}
                maps[period] = {'bounds': bounds_data, 'tiles': {tile: {'filename': p.name, 'sha256': file_hash(p)}
                                                                                for tile, p in tiles.items()}}
            station = stations[case['site']]
            raw, = (archive / 'photometers').rglob(case['file'])
            used_raw.add(raw)
            position = [station['latitude'], station['longitude'], station['height_m']]
            utc = datetime.fromisoformat(case['start']) + timedelta(minutes=30)
            fields = ('sun_max', 'moon_max', 'weather_max_cloud', 'weather_max_precip', 'weather_max_snow',
                      'weather_max_rh', 'qualified_metar', 'spread', 'closest_reference_city', 'reference_city_km')
            records[f'{site}-{night}'] = {k: case[k] for k in fields} | {
                'site': case['site'], 'night': night, 'utc': utc.isoformat() + 'Z', 'position': position,
                'map_period': period, 'band': 'SQM', 'magnitude': case['sqm'], 'readings': case['n_sqm'],
                'cloud_pct': case['weather_max_cloud'], 'natural_v_relative': 10**(.4 * (22 - 21.6)),
                'tier': 'supplementary_clear', 'aod': float(area_mean_aod((1 - weight)*a[0] + weight*b[0], lat, lon)),
                'native_times_utc': endpoints, 'native_time_weight': weight, 'sqm_file': case['file'],
            }
            counts[site]['selected_nights'] += 1
            break
        else:
            counts[site]['nights_without_matching_CAMS'] += 1
    sources = {p.name: file_hash(p) for p in (weather, metadata, previous_selection) if p is not None}
    return {'counts': counts, 'cases': records, 'maps': maps, 'source_sha256': sources,
            'raw_sqm_sha256': {p.name: file_hash(p) for p in sorted(used_raw)},
            'cams_requests': [{'sha256': file_hash(p), 'url': p.with_suffix('.url').read_text().strip()}
                             for p in sorted(used_cams)]}


def annual_fallback(paths):
    nasa = read_json(SELECTION)['sources']['NASA']['monthly_annual']
    cases, maps, unfilled = {}, {}, {}
    for path in paths:
        for month, group in read_json(path).items():
            if len(month) != 7:
                continue
            for key, row in group['records'].items():
                city = NEARBY_CITIES[row['site'].removeprefix('SQM_IT_')]
                qa = group['map'][city]
                near_fraction = qa['near_supported'] / qa['polygon_pixels']
                paired_fraction = qa['common_fraction_of_near_dnb'] or 0.
                if near_fraction >= .5 and paired_fraction >= .5:
                    continue
                year = month[:4]
                diagnostic = {'original_map_period': month, 'near_supported_fraction': near_fraction,
                              'common_fraction_of_near_DNB': paired_fraction, 'original_series': path.parent.name}
                if year not in nasa:
                    unfilled[key] = diagnostic
                    continue
                context = {k: v for k, v in row.items() if k != 'models'}
                cases[key] = context | diagnostic | {'map_period': year}
                maps[year] = nasa[year]
    return {'cases': cases, 'maps': maps, 'unfilled_cases': unfilled,
            'source_sha256': {p.parent.name: file_hash(p) for p in paths}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('satellite', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--previous-selection', type=Path)
    parser.add_argument('--annual-fallback-from', type=Path, nargs='+')
    parser.add_argument('--boundaries', type=Path)
    parser.add_argument('--transport', type=Path)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if (args.transport is None) != (args.boundaries is None):
        parser.error('--boundaries and --transport are used together')
    if not args.transport:
        result = (annual_fallback(args.annual_fallback_from) if args.annual_fallback_from else
                  prepare(args.archive, args.satellite, args.previous_selection))
        write_json(args.output, result)
        print('windows', len(result['cases']), 'unfilled', len(result.get('unfilled_cases', {})), flush=True)
        return
    result = read_json(args.output)
    features = {f['properties']['URAU_CODE']: f['geometry'] for f in read_json(args.boundaries)['features']}
    polygons = {name: features[code] for name, code in CITY_CODES.items() if name != 'Madrid'}
    args.transport.mkdir(parents=True, exist_ok=True)
    from run_city_source_transport import __file__ as transport_path
    identity = {'boundaries': file_hash(args.boundaries), 'runner': file_hash(Path(transport_path)),
                'selection': file_hash(args.output), 'samples': 262144}
    groups = defaultdict(list)
    legacy = {key: models for group in old_veneto()[0].values() for key, _, _, models in group}
    for key, row in result['cases'].items():
        models = legacy.get(key, {'standard': (Atmosphere(), None, None),
                                  'regional': (Atmosphere(aerosol_optical_depth=row['aod']), None, None)})
        groups[row['map_period']].append((key, row, Observer(*row['position']), models))
    with ProcessPoolExecutor(args.workers) as pool:
        jobs = {pool.submit(solve_period, (p, cases, result['maps'], args.satellite, polygons, 262144,
                                         args.transport, identity)): p for p, cases in groups.items()}
        results = {}
        for task in as_completed(jobs):
            period = jobs[task]
            results[period] = task.result()
            print(period, len(results[period]['records']), flush=True)
    write_json(args.transport / 'transport.json', dict(sorted(results.items())))


if __name__ == '__main__':
    main()
