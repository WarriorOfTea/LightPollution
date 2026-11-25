"""Compare fixed, regional and spatial AOD on screened Italian SQM windows."""

import argparse
import csv
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from lightpollution.adapters.storage import file_hash, read_json, write_json
from lightpollution.data import Bounds, read_region
from lightpollution.models.all_orders import AllOrderSettings
from lightpollution.models.background import BackgroundModel, background_config
from lightpollution.models.boundary_orders import integrate_boundary_variants
from lightpollution.observation import observation_context
from lightpollution.physics import Atmosphere, Observer
from run_cams_dutch_pilot import area_mean_aod
from run_photometer_all_orders import GridAtmosphere, NearBoundary

BOUNDS = Bounds(10.0, 43.6, 14.0, 47.2)
SITES = {'Galzignano': Observer(45.31151, 11.69832, 337),
         'Arcugnano': Observer(45.49744, 11.53481, 182)}
BACKGROUND_MAG = 21.6


class PairedMaps:
    """Score distance masks and a fixed-map sensitivity on the same paths."""

    def __init__(self, monthly, fixed):
        self.monthly, self.fixed = monthly, fixed
        self.labels = monthly.labels + ('fixed_outside_3km',)
        self.upper_bounds = np.r_[monthly.upper_bounds, fixed.upper_bounds]

    def sample(self, points, outgoing):
        monthly, support = self.monthly.sample(points, outgoing)
        fixed, _ = self.fixed.sample(points, outgoing)
        return np.column_stack((monthly, fixed)), support


def read_windows(path, site):
    observer = SITES[site]
    with path.open(encoding='utf-8-sig', newline='') as handle:
        rows = [row for row in csv.DictReader(handle)
                if row['site'] == f'SQM_IT_{site}'
                and row['above_twice_nominal_natural'] == 'True']
    for row in rows:
        if (float(row['latitude']), float(row['longitude'])) != (
                observer.latitude_deg, observer.longitude_deg):
            raise ValueError(f'Contradictory coordinates at {site}.')
        if (row['window_start_utc'], row['window_end_utc']) != ('22:00', '23:00'):
            raise ValueError('The CAMS samples must bracket the SQM window.')
    days = sorted(row['date_utc'] for row in rows)
    if not days or len(days) != len(set(days)):
        raise ValueError(f'Missing or duplicate windows at {site}.')
    return sorted(rows, key=lambda row: row['date_utc'])


def read_cams(path, days):
    doc = read_json(path)
    latitudes, longitudes = (np.asarray(doc[k]) for k in ('latitudes', 'longitudes'))
    if (doc['endpoint_suffixes_utc'] != ['T22:00', 'T23:00']
            or not np.array_equal(latitudes, np.linspace(BOUNDS.south, BOUNDS.north, 10).round(1))
            or not np.array_equal(longitudes, np.linspace(BOUNDS.west, BOUNDS.east, 11).round(1))):
        raise ValueError('CAMS grid or timestamps do not match this experiment.')
    grids = np.array([doc['aod_endpoints'][d] for d in days], dtype=float)
    if (grids.shape != (len(days), 2, len(latitudes), len(longitudes))
            or not np.all(np.isfinite(grids) & (grids >= 0))):
        raise ValueError('CAMS requires complete nonnegative endpoint grids.')
    grids = grids.mean(axis=1)
    return latitudes, longitudes, grids, area_mean_aod(grids, latitudes, longitudes)


def summarize_run(result):
    keys = ('settings', 'labels', 'all_order_radiance', 'all_order_standard_error',
            'mean_covariance', 'positive_paths', 'effective_score_samples',
            'category_labels', 'category_mean_radiance', 'ground_support_counts',
            'max_scattering_order_seen', 'finite_scattering_order_truncation', 'runtime_seconds')
    return {key: result[key] for key in keys}


def ephemerides(observer, day, base):
    contexts = [observation_context(observer, f'{day}T{hour}:00Z', 0, 90)
                for hour in ('22:00', '22:30', '23:00')]
    sun = [c['sun']['altitude_deg'] for c in contexts]
    moon = [c['moon']['altitude_deg'] for c in contexts]
    if max(sun) > -18 or max(moon) > -6:
        raise ValueError(f'{day} failed the solar/lunar limits.')
    model = BackgroundModel(background_config({}), observer, base, BACKGROUND_MAG)
    natural = model.evaluate(np.array([0.]), np.array([90.]), contexts[1])[0]['natural_relative']
    return {'sun_altitude_deg': sun, 'moon_altitude_deg': moon, 'natural_relative': natural}


def study(site, satellite, selection, cams_input, samples, annual_year=None):
    windows = read_windows(selection, site)
    observer = SITES[site]
    days = [row['date_utc'] for row in windows]
    lat, lon, grids, regional_aod = read_cams(cams_input, days)
    base = Atmosphere()
    seed = 41900 + 1000 * list(SITES).index(site)
    if annual_year is not None:
        seed += 10_000
    active_period, fixed = None, None
    rows, monthly_sources, runs = [], {}, {}

    def run(label, atmos, index):
        result = integrate_boundary_variants(observer, boundary, 0, 90, atmos,
                                             AllOrderSettings(samples=samples, seed=index))
        runs[label] = summarize_run(result)
        estimate = result['all_order_radiance'][1]
        se = result['all_order_standard_error'][1]
        print(f'{site} {label}: {estimate:.5g} ± {se:.2g}, '
              f"effective={result['effective_score_samples'][1]:.0f}", flush=True)

    for i, (window, grid, average) in enumerate(zip(windows, grids, regional_aod, strict=True)):
        day = window['date_utc']
        period = str(annual_year) if annual_year is not None else day[:7]
        if period != active_period:
            year = int(period[:4])
            month = None if annual_year is not None else int(period[5:])
            arrays, tiles = read_region(satellite / f'italy-{period}', BOUNDS, year, month)
            monthly = NearBoundary(arrays, observer, BOUNDS, (0, 3, 10))
            if fixed is None:
                fixed = NearBoundary(arrays, observer, BOUNDS, (3,))
            boundary = PairedMaps(monthly, fixed)
            monthly_sources[period] = {'tiles': {k: {name: v[name] for name in ('filename', 'sha256')}
                                                 for k, v in tiles.items()},
                                      'retained_cells': int(monthly.known.sum()),
                                      'total_cells': monthly.known.size,
                                      'max_radiance_by_mask': boundary.upper_bounds.tolist()}
            active_period = period
            run(f'{period}/base', base, seed + 3*i)
        regional = replace(base, aerosol_optical_depth=float(average))
        spatial = GridAtmosphere(latitudes=tuple(lat), longitudes=tuple(lon),
                                 grid=tuple(map(tuple, grid)))
        run(f'{day}/regional', regional, seed + 3*i + 1)
        run(f'{day}/spatial', spatial, seed + 3*i + 2)
        local_aod = np.interp(observer.latitude_deg, lat,
                              [np.interp(observer.longitude_deg, lon, g) for g in grid])
        rows.append({'date': day, 'readings': int(window['n_sqm']),
                     'observed_relative': float(window['total_to_nominal_natural_21_6'])
                     * 10**(-.4*(BACKGROUND_MAG-22)),
                     'aod_regional': float(average), 'aod_at_station': float(local_aod),
                     'aod_grid_min': float(grid.min()), 'aod_grid_max': float(grid.max()),
                     'runs': {'base': f'{period}/base',
                              'regional': f'{day}/regional', 'spatial': f'{day}/spatial'},
                     **ephemerides(observer, day, base)})
    return {'site': site, 'observer': asdict(observer), 'bounds': asdict(BOUNDS),
            'azimuth_deg': 0, 'altitude_deg': 90, 'background_mag': BACKGROUND_MAG,
            'reference_map_month': days[0][:7], 'atmosphere': asdict(base),
            **({'annual_map_year': annual_year} if annual_year is not None else {}),
            'selection_sha256': file_hash(selection), 'cams_sha256': file_hash(cams_input),
            'monthly_sources': monthly_sources, 'runs': runs, 'rows': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('site', choices=SITES)
    parser.add_argument('selection', type=Path)
    parser.add_argument('cams', type=Path)
    parser.add_argument('satellite', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--samples', type=int, default=1048576)
    parser.add_argument('--annual-map', type=int, help='Use one VNP46A4 map for every window.')
    args = parser.parse_args()
    write_json(args.output, study(args.site, args.satellite, args.selection, args.cams,
                                  args.samples, args.annual_map))


if __name__ == '__main__':
    main()
