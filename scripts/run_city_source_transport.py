"""Trace the contributions of fixed city polygons and angular emission terms."""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

from matplotlib.path import Path as PolygonPath
import numpy as np

from lightpollution.adapters.storage import file_hash, read_json, write_json
from lightpollution.data import Bounds, PIXELS_PER_DEGREE, good_mask, load_region, read_region
from lightpollution.models.all_orders import AllOrderSettings
from lightpollution.models.angular_sources import empirical_radiance
from lightpollution.models.boundary_orders import integrate_boundary_variants
from lightpollution.physics import EARTH_RADIUS_M, Atmosphere, Observer, unit_position
from run_photometer_all_orders import GridAtmosphere
from run_photometer_vertical_italy import VerticalGridAtmosphere
from run_selected_aod_models import HEIGHT_BINS, LABELS, NATIVE_LAT, NATIVE_LON, aerosol_layers


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts'
SELECTION = ART / 'photometer-aod-selection-20261126' / 'selection-audit.json'
TRANSPORT = ART / 'photometer-three-models-20261126'
LEGACY = ART / 'photometer-italy-20261125'
LEGACY_PROFILES = ART / 'photometer-vertical-aerosol-20261126' / 'profile-input.json'
MADRID = ART / 'photometer-aod-pilot-20261125' / 'madrid-power.json'
MADRID_MAPS = ART / 'angular-nasa-study-20261113' / 'protocol.json'
CITY_CODES = {'Verona': 'IT012C', 'Vicenza': 'IT076C', 'Padova': 'IT027C', 'Brescia': 'IT028C',
              'Treviso': 'IT041C', 'Pordenone': 'IT051C', 'Venezia': 'IT011C', 'Madrid': 'ES001C'}
CITY_NAMES = tuple(CITY_CODES)
BASIS_TERMS = ('lambert', 'off_hold', 'horizon_slope')


def grid_city_indices(bounds, geometries, city_names=CITY_NAMES):
    index = np.zeros(bounds.shape, dtype=np.uint8)
    for name, geometry in geometries.items():
        polygons = geometry['coordinates']
        if geometry['type'] == 'Polygon':
            polygons = [polygons]
        code = city_names.index(name) + 1
        for shell, *holes in polygons:
            shell = np.asarray(shell)
            row, = np.nonzero((bounds.latitude >= shell[:, 1].min()) & (bounds.latitude <= shell[:, 1].max()))
            col, = np.nonzero((bounds.longitude >= shell[:, 0].min()) & (bounds.longitude <= shell[:, 0].max()))
            x, y = np.meshgrid(bounds.longitude[col], bounds.latitude[row])
            points = np.column_stack((x.ravel(), y.ravel()))
            inside = PolygonPath(shell).contains_points(points)
            for hole in holes:
                inside &= ~PolygonPath(hole).contains_points(points)
            block = index[np.ix_(row, col)]
            inside = inside.reshape(block.shape)
            if np.any(inside & (block != 0) & (block != code)):
                raise ValueError('Selected municipal raster masks overlap.')
            block[inside] = code
            index[np.ix_(row, col)] = block
    return index


class CityBoundary:
    """Score the baseline and city bases on the same Monte Carlo histories."""

    def __init__(self, arrays, bounds, city_index, cities, observer):
        self.bounds = bounds
        self.cities = cities
        self.city_index = city_index
        near, off = 'NearNadir_Composite_Snow_Free', 'OffNadir_Composite_Snow_Free'
        self.known = good_mask(arrays[near], arrays[near + '_Quality'], arrays[near + '_Num']) & np.isin(
            arrays['Land_Water_Mask'], [0, 1, 5])
        self.common = self.known & good_mask(arrays[off], arrays[off + '_Quality'], arrays[off + '_Num'])
        self.near = np.where(self.known, arrays[near], 0)
        self.off = np.where(self.common, arrays[off], 0)
        normal = unit_position(bounds.latitude[:, None], bounds.longitude[None, :])
        chord = np.linalg.norm(normal - unit_position(observer.latitude_deg, observer.longitude_deg), axis=-1)
        self.near3 = chord < 2 * np.sin(3000 / (2 * EARTH_RADIUS_M))
        self.near10 = chord < 2 * np.sin(10000 / (2 * EARTH_RADIUS_M))
        self.labels = ('total_lambert', 'inside_3km', 'inside_10km') + tuple(
            f'{name}_{term}' for name in cities for term in BASIS_TERMS)
        maximum = float(self.near.max())
        self.upper_bounds = [maximum] * 3
        self.map_diagnostics = {}
        for name in cities:
            city = city_index == CITY_NAMES.index(name) + 1
            n = self.near[city].max(initial=0)
            o = self.off[city].max(initial=0)
            self.upper_bounds.extend([n, max(n, o), o])
            area_weight = np.cos(np.deg2rad(bounds.latitude))[:, None] * city
            near_sum = float((self.near * area_weight).sum())
            common_near = float((self.near * area_weight * self.common).sum())
            self.map_diagnostics[name] = {
                'polygon_pixels': int(city.sum()), 'near_supported': int((city & self.known).sum()),
                'near_off_supported': int((city & self.common).sum()),
                'common_fraction_of_near_dnb': common_near / near_sum if near_sum else None,
                'off_over_near_common': float((self.off * area_weight).sum()) / common_near if common_near else None,
            }

    def sample(self, points, outgoing):
        x, y, z = points.T
        lat = np.rad2deg(np.arctan2(z, np.hypot(x, y)))
        lon = np.rad2deg(np.arctan2(y, x))
        row = np.floor((self.bounds.north - lat) * PIXELS_PER_DEGREE).astype(int)
        col = np.floor((lon - self.bounds.west) * PIXELS_PER_DEGREE).astype(int)
        n, m = self.known.shape
        inside = (row >= 0) & (row < n) & (col >= 0) & (col < m)
        row, col = row.clip(0, n - 1), col.clip(0, m - 1)
        near = np.where(inside, self.near[row, col], 0.)
        off = self.off[row, col]
        common = inside & self.common[row, col]
        city = np.where(inside, self.city_index[row, col], 0)
        mu = np.clip(np.einsum('ij,ij->i', outgoing, points) / np.linalg.norm(points, axis=1), 0, 1)
        hold = np.where(common, empirical_radiance(near, off, mu, 1.), near)
        slope = np.where(common, off * np.maximum(0, 1 - 2 * mu), 0.)
        values = [near, near * self.near3[row, col], near * self.near10[row, col]]
        for name in self.cities:
            mask = city == CITY_NAMES.index(name) + 1
            values.extend([near * mask, hold * mask, slope * mask])
        return np.column_stack(values), np.where(inside, 1 + self.known[row, col].astype(int), 0)


def main_veneto():
    source = read_json(SELECTION)
    archive = read_json(TRANSPORT / 'results.json')
    if archive['selection_sha256'] != file_hash(SELECTION):
        raise ValueError('Veneto selection and past transport disagree.')
    instruments = source['sources']['DARKERSKY4CE']['instruments']
    maps = source['sources']['NASA']['monthly_annual']
    cams = source['sources']['CAMS_550']
    lat, lon = tuple(cams['latitude_deg']), tuple(cams['longitude_deg'])
    cache = cams['values_by_native_hour_utc']
    with np.load(TRANSPORT / 'geos-cf-layer-integrals.npz') as data:
        np.testing.assert_array_equal(data['latitude_deg'], NATIVE_LAT)
        np.testing.assert_array_equal(data['longitude_deg'], NATIVE_LON)
        areas = dict(zip(data['timestamps_utc'], data['PM25_vertical_integrals'], strict=True))
    groups = defaultdict(list)
    for key, selected in source['station_nights'].items():
        old = archive['results'][key]
        period, site = old['map_period'], old['site']
        location = instruments[site]
        t0, t1 = (np.asarray(cache[t]) for t in selected['aod_native_times'])
        weight = selected['aod_time_weight_for_second']
        layers = aerosol_layers(areas[old['utc']], (1 - weight) * t0 + weight * t1, lat, lon)
        atmosphere = (Atmosphere(), Atmosphere(aerosol_optical_depth=old['aod_regional']),
                      VerticalGridAtmosphere(edges_m=tuple(HEIGHT_BINS), tau_layers=layers, latitudes=lat, longitudes=lon))
        position = Observer(location['latitude'], location['longitude'], location['height_m'])
        context = {'site': site, 'night': old['night'], 'utc': old['utc'], 'map_period': period, 'band': 'SQM',
                   'magnitude': old['sqm'], 'readings': old['n_sqm'], 'cloud_pct': selected['weather_max_cloud'],
                   'natural_v_relative': old['natural_v']['natural_relative'], 'tier': selected['tier'],
                   'aod': old['aod_regional']}
        models = {name: (model, old['artificial'][name]['seed'], old['artificial'][name]['all_order_radiance'][0])
                  for name, model in zip(LABELS, atmosphere, strict=True)}
        groups[period].append((key, context, position, models))
    return groups, maps


def old_veneto():
    profiles = read_json(LEGACY_PROFILES)
    comparisons = read_json(LEGACY / 'comparison.json')['sites']
    groups, maps = defaultdict(list), {}
    lat, lon, edges = [tuple(profiles[k]) for k in ('latitudes', 'longitudes', 'altitude_edges_m')]
    for site, dataset in comparisons.items():
        old = read_json(LEGACY / f'{site.lower()}.json')
        for period, metadata in old['monthly_sources'].items():
            maps[period] = {'bounds': old['bounds'], 'tiles': metadata['tiles']}
        for row in dataset['rows']:
            if not row['weather']['passes']:
                continue
            day = row['date']
            context = {'site': 'SQM_IT_' + site, 'night': day, 'utc': day + 'T22:30:00Z', 'map_period': day[:7],
                       'band': 'SQM', 'magnitude': 22 - 2.5 * np.log10(row['observed_relative']),
                       'readings': row['readings'], 'cloud_pct': row['weather']['maxima']['cloud_cover'],
                       'natural_v_relative': row['natural_relative'], 'tier': 'legacy_clear', 'aod': row['aod_regional']}
            layers = np.asarray(profiles['days'][day]['aod_per_layer'])
            atmosphere = (Atmosphere(), Atmosphere(aerosol_optical_depth=row['aod_regional']),
                          VerticalGridAtmosphere(edges_m=edges, tau_layers=layers, latitudes=lat, longitudes=lon))
            groups[day[:7]].append((site + '-' + day, context, Observer(**old['observer']),
                                    {name: (model, None, None) for name, model in zip(LABELS, atmosphere, strict=True)}))
    return groups, maps


def madrid():
    groups = defaultdict(list)
    maps = read_json(MADRID_MAPS)['inputs']
    for row in read_json(MADRID)['nights']:
        day = row['day']
        axes = {'latitudes': (39.5, 40.5, 41.5), 'longitudes': (-4.5, -3.5, -2.5)}
        models = {'standard': Atmosphere(), 'power_regional': Atmosphere(aerosol_optical_depth=row['aod_region_sea_level']),
                  'power_spatial': GridAtmosphere(grid=tuple(map(tuple, row['aod_grid_sea_level'])), **axes)}
        context = {'site': 'AstMon_UCM', 'night': day, 'utc': day + 'T22:30:00Z', 'map_period': day[:7],
                   'band': 'V', 'magnitude': 22 - 2.5 * np.log10(row['observed_relative']), 'readings': row['samples'],
                   'cloud_pct': None, 'natural_v_relative': 10**(.4 * (22 - 21.6)),
                   'tier': 'urban_power_pilot', 'aod': row['aod_region_sea_level']}
        period = day[:7]
        groups[period].append(('Madrid-' + day, context, Observer(40.451111, -3.726111, 650),
                                {name: (model, None, None) for name, model in models.items()}))
    return groups, {k: maps['madrid-' + k] for k in groups}


def solve_period(job):
    period, cases, maps, source, polygons, samples, output, identity = job
    cached = output / f'{period}.json'
    if cached.exists():
        result = read_json(cached)
        if result['identity'] == identity:
            return result
    metadata = maps[period]
    year, *month = map(int, period.split('-'))
    if 'maps' in metadata:
        path = ROOT / metadata['maps']
        if file_hash(path) != metadata['maps_sha256']:
            raise ValueError('Madrid source grid changed.')
        arrays, bounds, _ = load_region(path)
    else:
        bounds = Bounds(**metadata['bounds'])
        arrays, tiles = read_region(source / period, bounds, year, month[0] if month else None)
        if {k: v['sha256'] for k, v in tiles.items()} != {k: v['sha256'] for k, v in metadata['tiles'].items()}:
            raise ValueError(f'Modified Black Marble tile for {period}.')
    indices = grid_city_indices(bounds, polygons)
    active = tuple(name for name in polygons if np.any(indices == CITY_NAMES.index(name) + 1))
    cache, boundaries, records = {}, {}, {}
    for key, context, observer, models in cases:
        if observer not in boundaries:
            boundaries[observer] = CityBoundary(arrays, bounds, indices, active, observer)
        boundary = boundaries[observer]
        record = context | {'position': [observer.latitude_deg, observer.longitude_deg, observer.height_m], 'models': {}}
        for name, (model, seed, old) in models.items():
            scope = context['site'] + '/' + period if name == 'standard' else key
            if seed is None:
                seed = int.from_bytes(sha256((scope + '/' + name).encode()).digest()[:4], 'big')
            if name != 'standard' or seed not in cache:
                result = integrate_boundary_variants(observer, boundary, 0., 90., model,
                    AllOrderSettings(samples=samples, seed=seed))
                value = {k: result[k] for k in ('all_order_radiance', 'mean_covariance', 'effective_score_samples',
                                                'max_scattering_order_seen')}
                cache[seed] = value | {'seed': seed, 'samples': samples}
            value = cache[seed]
            if old is not None and samples == 262144:
                np.testing.assert_allclose(value['all_order_radiance'][0], old, atol=1e-12, rtol=1e-12)
            record['models'][name] = value
        records[key] = record
    saved = {'identity': identity, 'columns': list(boundary.labels), 'map': boundary.map_diagnostics, 'records': records}
    write_json(cached, saved)
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('series', choices=('veneto', 'legacy', 'madrid'))
    parser.add_argument('boundaries', type=Path, help='GISCO 2024 city GeoJSON in EPSG 4326')
    parser.add_argument('satellite', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--samples', type=int, default=262144)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    loader, inputs = {
        'veneto': (main_veneto, (SELECTION, TRANSPORT / 'results.json', TRANSPORT / 'geos-cf-layer-integrals.npz')),
        'legacy': (old_veneto, (LEGACY / 'comparison.json', LEGACY / 'arcugnano.json', LEGACY / 'galzignano.json', LEGACY_PROFILES)),
        'madrid': (madrid, (MADRID, MADRID_MAPS)),
    }[args.series]
    features = {f['properties']['URAU_CODE']: f['geometry'] for f in read_json(args.boundaries)['features']}
    used_cities = ('Madrid',) if args.series == 'madrid' else CITY_NAMES[:-1]
    polygons = {name: features[CITY_CODES[name]] for name in used_cities}
    groups, maps = loader()
    identity = {'boundaries': file_hash(args.boundaries), 'runner': file_hash(Path(__file__)),
                'inputs': {p.name: file_hash(p) for p in inputs}, 'samples': args.samples}
    with ProcessPoolExecutor(args.workers) as pool:
        jobs = {pool.submit(solve_period, (p, cases, maps, args.satellite, polygons, args.samples, args.output, identity)): p
                for p, cases in groups.items()}
        results = {}
        for task in as_completed(jobs):
            period = jobs[task]
            results[period] = task.result()
            print(period, len(results[period]['records']), flush=True)
    write_json(args.output / 'transport.json', dict(sorted(results.items())))


if __name__ == '__main__':
    main()
