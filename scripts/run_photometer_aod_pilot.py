"""Run a zenith pilot with spatial AOD JSONs saved from NASA POWER."""

import argparse
import csv
from dataclasses import replace
from pathlib import Path

import numpy as np

from lightpollution.adapters.storage import file_hash, read_json, write_json
from lightpollution.data import load_region
from lightpollution.physics import (
    EARTH_RADIUS_M, Atmosphere, Observer, SolverSettings, emission,
    hg_phase, integrate_direction, quadrature, rayleigh_phase, source_to_point,
)
from lightpollution.sources import regional_sources

ROOT = Path(__file__).resolve().parents[1]
LATITUDES = np.array([39.5, 40.5, 41.5])
LONGITUDES = np.array([-4.5, -3.5, -2.5])


def interpolation(positions, latitudes, longitudes):
    x, y, z = np.moveaxis(positions, -1, 0)
    latitude = np.rad2deg(np.arctan2(z, np.hypot(x, y)))
    longitude = np.rad2deg(np.arctan2(y, x))
    row = np.interp(latitude, latitudes, np.arange(len(latitudes)))
    col = np.interp(longitude, longitudes, np.arange(len(longitudes)))
    return np.maximum(1 - abs(row[..., None, None] - np.arange(len(latitudes))[:, None]), 0) * np.maximum(
        1 - abs(col[..., None, None] - np.arange(len(longitudes))), 0)


def spatial_responses(observer, sources, grids, settings, latitudes=LATITUDES, longitudes=LONGITUDES):
    """Use the existing first-order geometry and integrate grid AOD along both chords."""
    atmosphere = Atmosphere()
    origin = observer.position
    direction = observer.direction(0, 90)
    end = atmosphere.top_m - observer.height_m
    edges = settings.near_ray_scale_m * np.expm1(np.linspace(
        0, np.log1p(end / settings.near_ray_scale_m), settings.ray_segments + 1))
    distance = (edges[1:] + edges[:-1]) / 2
    points = origin + distance[:, None] * direction
    lengths_squared, incoming, mu = source_to_point(sources, points)
    radius2 = np.sum(points * points, axis=1)
    lengths = np.sqrt(lengths_squared)
    nodes, weights = quadrature(settings.path_quadrature)
    rayleigh_tau_source = replace(atmosphere, aerosol_optical_depth=0).optical_depth_radial(
        EARTH_RADIUS_M**2, radius2[:, None], lengths_squared, settings.path_quadrature)
    rayleigh_tau_observer = replace(atmosphere, aerosol_optical_depth=0).optical_depth_radial(
        np.dot(origin, origin), radius2, distance**2, settings.path_quadrature)
    path_factor = np.zeros((*lengths.shape, len(latitudes), len(longitudes)))
    for node, weight in zip(nodes, weights, strict=True):
        chord = (1 - node) * sources.positions_m[None, :, :] + node * points[:, None, :]
        heights = np.linalg.norm(chord, axis=-1) - EARTH_RADIUS_M
        density = np.exp(-np.maximum(heights, 0) / atmosphere.aerosol_scale_height_m)
        path_factor += (weight * density)[..., None, None] * interpolation(chord, latitudes, longitudes)
    path_factor *= (lengths / atmosphere.aerosol_scale_height_m)[..., None, None]
    aod_observer = np.einsum('ij,kij->k', interpolation(origin, latitudes, longitudes), grids)
    aerosol_tau_observer = aod_observer[:, None] * (
        np.exp(-observer.height_m / atmosphere.aerosol_scale_height_m)
        - np.exp(-(observer.height_m + distance) / atmosphere.aerosol_scale_height_m))
    height = np.sqrt(radius2) - EARTH_RADIUS_M
    br, ba_unit, _ = replace(atmosphere, aerosol_optical_depth=1).coefficients(height)
    cosine = -np.sum(incoming * direction, axis=-1)
    rayleigh = br[:, None] * rayleigh_phase(cosine)
    aerosol_unit = ba_unit[:, None] * hg_phase(cosine, atmosphere.aerosol_g)
    source_weight = (sources.power[None, :] * emission(mu, settings.horizontal_fraction)
                     / lengths_squared * np.exp(-rayleigh_tau_source))
    return np.array([
        np.sum(np.sum(source_weight * (rayleigh + aod * aerosol_unit)
                      * np.exp(-np.einsum('pqij,ij->pq', path_factor, grid)), axis=1)
               * np.exp(-rayleigh_tau_observer - attenuation) * np.diff(edges))
        for grid, aod, attenuation in zip(grids, aod_observer, aerosol_tau_observer, strict=True)
    ])


def power_columns(directory, month, dates):
    columns = np.empty((len(dates), 3, 3))
    hashes = {}
    for i, lat in enumerate(LATITUDES):
        for j, lon in enumerate(LONGITUDES):
            path = directory / f'{month}_{lat}_{lon}.json'
            data = read_json(path)
            lon_read, lat_read, height = data['geometry']['coordinates']
            if (lon_read, lat_read) != (lon, lat) or data['header']['sources'] != ['SYN1DEG']:
                raise ValueError('Unexpected product or grid position.')
            aod = data['properties']['parameter']['AOD_55']
            for k, day in enumerate(dates):
                key = day.replace('-', '') + '22'
                t22, t23 = aod[key], aod[key[:-2] + '23']
                if not (0 <= t22 and 0 <= t23):
                    raise ValueError(f'Missing AOD at {key}.')
                # Convert the reported local column to the shared spherical reference.
                columns[k, i, j] = (t22 + t23) * .5 * np.exp(height / Atmosphere().aerosol_scale_height_m)
            hashes[path.name] = file_hash(path)
    return columns, hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('power_directory', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    audit = ROOT / 'artifacts/photometer-cams-preflight-20261125/candidate-audit.json'
    selected = {row['date_utc']: row['eligible_obs_utc'] for row in read_json(audit)['madrid']['days']
                if row['known_benchmark_night'] or row['new_prior_weather_candidate']}
    raw = ROOT / 'scratch/data-search-20261110/AstMon_evol.csv'
    wanted = {stamp for dates in selected.values() for stamp in dates}
    with raw.open() as stream:
        measured = {r['Datetime_UTC'].replace(' ', 'T') + 'Z': float(r['Magnitude'])
                    for r in csv.DictReader(stream, delimiter=';')
                    if r['Filter'] == 'Johnson-V' and r['Pixel_x'] == 'Pos0'
                    and r['Datetime_UTC'].replace(' ', 'T') + 'Z' in wanted}
    if set(measured) != wanted:
        raise ValueError('Photometry times do not match the preflight.')
    observer = Observer(40.451111, -3.726111, 650)
    nights, fingerprints = [], {}
    for month in ('03', '11'):
        days = sorted(day for day in selected if day[5:7] == month)
        grid_path = ROOT / f'data/seasonal-ground-20261110/madrid-matched-2017/2017-{month}/source-grid.npz'
        arrays, bounds, meta = load_region(grid_path)
        sources, _ = regional_sources(arrays, bounds, observer, block=12,
                                      normalisation_radiance_area=meta['reference_radiance_area_sum'])
        local, hashes = power_columns(args.power_directory, month, days)
        area = np.cos(np.deg2rad(LATITUDES))[:, None] * np.ones((3, 3))
        regional = np.average(local, weights=np.broadcast_to(area, local.shape), axis=(1, 2))
        grid = np.concatenate([np.full((1, 3, 3), .10),
                               np.broadcast_to(regional[:, None, None], local.shape), local])
        coarse = spatial_responses(observer, sources, grid, SolverSettings(192, 24))
        fine = spatial_responses(observer, sources, grid, SolverSettings(384, 48))
        homogeneous = integrate_direction(observer, sources, 0, 90, Atmosphere(), SolverSettings(384, 48))
        if not np.isclose(fine[0], homogeneous, rtol=1e-10, atol=0):
            raise RuntimeError('The homogeneous spatial kernel differs from the first-order baseline.')
        for i, day in enumerate(days):
            ids = [0, 1 + i, 1 + len(days) + i]
            response = fine[ids]
            nights.append({'day': day, 'samples': len(selected[day]),
                           'observed_relative': float(np.mean([10**(-.4*(measured[t]-22)) for t in selected[day]])),
                           'aod_region_sea_level': regional[i], 'aod_grid_sea_level': local[i].tolist(),
                           'responses': response.tolist(),
                           'refinement_relative': abs(response / coarse[ids] - 1).tolist()})
        fingerprints[month] = dict(power=hashes, source_grid_sha256=file_hash(grid_path))
        print(month, len(days), 'nights complete', flush=True)
    natural = 10**(-.4*(21.6-22))
    train = next(row for row in nights if row['day'] == '2017-03-19')
    scale = (train['observed_relative'] - natural) / train['responses'][0]
    variants = ['current', 'power_region', 'power_spatial']
    for row in nights:
        predicted = scale * np.array(row['responses']) + natural
        row['model_minus_observed_mag'] = dict(zip(
            variants, (-2.5*np.log10(predicted / row['observed_relative'])).tolist(), strict=True))
    test = [row for row in nights if row['day'] not in ('2017-03-19', '2017-11-19')]
    metrics = {}
    for variant in variants:
        residual = np.array([row['model_minus_observed_mag'][variant] for row in test])
        metrics[variant] = {'bias_mag': float(residual.mean()), 'mae_mag': float(abs(residual).mean()),
                            'rmse_mag': float(np.sqrt(np.mean(residual**2)))}
    write_json(args.output, {'support': 'Madrid urban engineering pilot; preliminary local cloud checks only.',
                             'aod_provider': 'NASA POWER CERES SYN1deg, not CAMS',
                             'utc_reference': '22:30', 'training_night': train['day'], 'scale': scale,
                             'natural_v_mag_arcsec2': 21.6, 'input_hashes': fingerprints,
                             'raw_photometry_sha256': file_hash(raw), 'audit_sha256': file_hash(audit),
                             'metrics_9_candidate_nights': metrics, 'nights': nights})
    print(metrics)


if __name__ == '__main__':
    main()
