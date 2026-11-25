"""Run three first-order AOD variants on screened Dutch-network SQM windows."""

import argparse
import json
from pathlib import Path

import numpy as np

from lightpollution.adapters.storage import file_hash, write_json
from lightpollution.data import Bounds, read_region
from lightpollution.physics import Atmosphere, Observer, SolverSettings, integrate_direction
from lightpollution.sources import regional_sources
from run_photometer_aod_pilot import spatial_responses

SITES = {'Texel': (53.007, 4.7792), 'Borkum': (53.582, 6.664),
         'Lochem': (52.16, 6.41), 'Gorredijk': (53.01, 6.07),
         'Vlieland-Oost': (53.3006, 5.0774), 'Schiermonnikoog-dorp': (53.481, 6.168),
         'Moddergat': (53.40656, 6.06985), 'Lauwersoog': (53.4, 6.6),
         'Spiekeroog': (53.765, 7.726)}
BOUNDS = Bounds(3.6, 50.8, 8.4, 54.4)


def load_aod(path, days):
    responses = json.loads(path.read_text())
    latitudes = np.unique([round(r['latitude'], 4) for r in responses])
    longitudes = np.unique([round(r['longitude'], 4) for r in responses])
    expected_latitudes = [50.8, 52., 53.2, 54.4]
    expected_longitudes = [3.6, 5.2, 6.8, 8.4]
    if (not np.allclose(latitudes, expected_latitudes, rtol=0, atol=1e-4)
            or not np.allclose(longitudes, expected_longitudes, rtol=0, atol=1e-4)):
        raise ValueError('Unexpected atmospheric grid.')
    grid = np.full((len(days), len(latitudes), len(longitudes)), np.nan)
    for row in responses:
        if row['utc_offset_seconds'] != 0:
            raise ValueError('CAMS timestamps must be UTC.')
        i = np.searchsorted(latitudes, round(row['latitude'], 4))
        j = np.searchsorted(longitudes, round(row['longitude'], 4))
        if np.any(np.isfinite(grid[:, i, j])):
            raise ValueError('Duplicate CAMS cell.')
        series = dict(zip(row['hourly']['time'], row['hourly']['aerosol_optical_depth'], strict=True))
        grid[:, i, j] = [(series[d + 'T22:00'] + series[d + 'T23:00']) / 2 for d in days]
    if not np.all(np.isfinite(grid) & (grid >= 0)):
        raise ValueError('Missing or invalid CAMS AOD.')
    return latitudes, longitudes, grid


def score(rows, variants):
    total = []
    for site in sorted({r['site'] for r in rows}):
        group = [r for r in rows if r['site'] == site and r['near_fraction_3_10km'] <= .2]
        if len(group) < 2:
            continue
        first, *other = sorted(group, key=lambda r: r['day'])
        natural = 10**(-.4*(22-22))
        scale = (first['observed_relative'] - natural) / first['responses']['current']
        if scale <= 0:
            continue
        for row in other:
            residuals = {v: float(-2.5*np.log10((scale*row['responses'][v]+natural)
                                               / row['observed_relative'])) for v in variants}
            total.append({'site': site, 'train_day': first['day'], 'check_day': row['day'], 'residual_mag': residuals})
    mae = {v: float(np.mean([abs(r['residual_mag'][v]) for r in total])) if total else None
           for v in variants}
    return total, mae


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('satellite', type=Path)
    parser.add_argument('selection', type=Path)
    parser.add_argument('cams', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    days = sorted({r['day'] for rows in selection.values() for r in rows})
    lat, lon, grid = load_aod(args.cams, days)
    weights = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], grid.shape)
    regional = np.average(grid, weights=weights, axis=(1, 2))
    arrays, tiles = read_region(args.satellite, BOUNDS, 2024, 1)
    variants = ('current', 'cams_region', 'cams_spatial')
    output = []
    for site, rows in selection.items():
        if not rows:
            continue
        observer = Observer(*SITES[site], 5)
        _, source_description = regional_sources(arrays, BOUNDS, observer, block=24, exclude_radius_m=0)
        normalisation = source_description['retained_radiance_area']
        sources, _ = regional_sources(arrays, BOUNDS, observer, block=24,
                                      normalisation_radiance_area=normalisation)
        distant, _ = regional_sources(arrays, BOUNDS, observer, block=24, exclude_radius_m=10_000,
                                       normalisation_radiance_area=normalisation)
        inds = [days.index(row['day']) for row in rows]
        local = grid[inds]
        cases = np.concatenate([np.full((1, len(lat), len(lon)), .1),
                                np.broadcast_to(regional[inds, None, None], local.shape), local])
        coarse_settings, fine_settings = SolverSettings(96, 12), SolverSettings(192, 24)
        coarse = spatial_responses(observer, sources, cases, coarse_settings, lat, lon)
        fine = spatial_responses(observer, sources, cases, fine_settings, lat, lon)
        baseline = integrate_direction(observer, sources, 0, 90, Atmosphere(), fine_settings)
        if not np.isclose(fine[0], baseline, rtol=1e-10, atol=0):
            raise RuntimeError('Spatial solver does not reproduce the homogeneous baseline.')
        distance_check = integrate_direction(observer, distant, 0, 90, Atmosphere(), fine_settings)
        near_fraction = float(1 - distance_check / baseline)
        for n, (row, idx) in enumerate(zip(rows, inds, strict=True)):
            values = [0, 1 + n, 1 + len(rows) + n]
            output.append({**row, 'site': site, 'aod_area_mean': regional[idx],
                           'near_fraction_3_10km': near_fraction,
                           'observed_relative': 10**(-.4*(row['mean_mag']-22)),
                           'responses': dict(zip(variants, fine[values].tolist(), strict=True)),
                           'refinement_relative': dict(zip(variants, abs(fine[values]/coarse[values]-1).tolist(), strict=True))})
        print(site, len(rows), near_fraction, flush=True)
    checks, mae = score(output, variants)
    write_json(args.output, {'provider': 'Open-Meteo proxy of operational CAMS Global, not EAC4',
                             'domain': vars(BOUNDS), 'grid_latitudes': lat.tolist(), 'grid_longitudes': lon.tolist(),
                             'days': days, 'aod_grid': grid.tolist(), 'station_windows': output,
                             'scoring': 'Effective SQM/V proxy, nominal natural 22; one fixed scale per site from first eligible night; coastal and source QA still preliminary.',
                             'station_check_residuals': checks, 'pilot_mae_mag': mae,
                             'cores': {'hdf5': {k: v['sha256'] for k, v in tiles.items()},
                                       'cams_sha256': file_hash(args.cams), 'selection_sha256': file_hash(args.selection)}})
    print(mae)


if __name__ == '__main__':
    main()
