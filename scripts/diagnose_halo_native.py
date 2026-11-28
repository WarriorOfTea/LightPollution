"""Check the A1 grid and viewing-angle convention on two overlapping L1B granules."""

import argparse
from datetime import date
from pathlib import Path

import h5py
import numpy as np

from lightpollution.adapters.storage import file_hash, read_json, write_json
from lightpollution.data import PIXELS_PER_DEGREE
from lightpollution.physics import EARTH_RADIUS_M
from run_halo_city_fits import LOCATIONS
from run_viirs_halo_recovery import LABELS, ROOT, preflight


CASES = {'Madrid': '2013-12-29', 'Tucson': '2019-04-03'}
MANIFEST = ROOT / 'artifacts/viirs-halo-airglow-20261127/source-audit.json'


def native_pixels(paths, bounds):
    with h5py.File(paths['VNP03DNB']) as geolocation:
        geo = geolocation['geolocation_data']
        lat, lon = geo['latitude'][:], geo['longitude'][:]
        regional = (
            (lat <= bounds.north)
            & (lat > bounds.south)
            & (lon >= bounds.west)
            & (lon < bounds.east)
        )
        selection = np.where(regional)
        top, left = (int(v.min()) for v in selection)
        bottom, right = (int(v.max()) + 1 for v in selection)
        window = (slice(top, bottom), slice(left, right))
        ix, iy = selection[0] - top, selection[1] - left

        def physical(name):
            dataset = geo[name]
            raw = dataset[window][ix, iy]
            scale = float(np.asarray(dataset.attrs.get('scale_factor', 1)).item())
            offset = float(np.asarray(dataset.attrs.get('add_offset', 0)).item())
            return raw * scale + offset

        height, ray, z, az, moon, sun, quality = (
            physical(v)
            for v in (
                'height',
                'range',
                'sensor_zenith',
                'sensor_azimuth',
                'lunar_zenith',
                'solar_zenith',
                'quality_flag',
            )
        )
    with h5py.File(paths['VNP02DNB']) as data:
        obs = data['observation_data']
        dnb = obs['DNB_observations'][window][ix, iy] * 1e9  # W/cm²/sr → nW/cm²/sr
        flag = obs['DNB_quality_flags'][window][ix, iy]
    select = (
        np.isfinite(dnb)
        & (dnb >= 0)
        & (flag == 0)
        & (quality == 0)
        & (z >= 0)
        & (z < 75)
        & (moon >= 100)
        & (sun >= 108)
    )
    fields = lat[selection], lon[selection], height, ray, z, az, dnb, selection[0] // 16
    return tuple(np.asarray(v[select], dtype=float) for v in fields)


def same_scan_spread(latitude, longitude, height, distance, zenith, azimuth, scan):
    phi, lam, z, az = np.deg2rad([latitude, longitude, zenith, azimuth])
    up = np.array([np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)]).T
    north = np.array(
        [-np.sin(phi) * np.cos(lam), -np.sin(phi) * np.sin(lam), np.cos(phi)]
    ).T
    east = np.array([-np.sin(lam), np.cos(lam), np.zeros_like(lam)]).T
    radial = up * (EARTH_RADIUS_M + height + distance * np.cos(z))[:, None]
    lateral = (north * np.cos(az)[:, None] + east * np.sin(az)[:, None]) * (
        distance * np.sin(z)
    )[:, None]
    result = {}
    for name, points in (
        ('as_written', radial + lateral),
        ('azimuth_reversed', radial - lateral),
    ):
        residuals = []
        for line in np.unique(scan):
            px = points[scan == line]
            if len(px) > 4:
                residuals.extend(np.linalg.norm(px - px.mean(axis=0), axis=1))
        result[name] = {
            'n': len(residuals),
            'median_km': float(np.median(residuals) / 1000),
            'p90_km': float(np.quantile(residuals, 0.90) / 1000),
        }
    return result


def compare(paths, data):
    bounds = data['bounds']
    lat, lon, height, distance, zenith, azimuth, dnb, scan = native_pixels(
        paths, bounds
    )
    row = ((bounds.north - lat) * PIXELS_PER_DEGREE).astype(int)
    col = ((lon - bounds.west) * PIXELS_PER_DEGREE).astype(int)
    grid, valid = data['raw'][row, col], data['valid'][row, col]
    a1_native = {}
    for label, mask in zip(LABELS, data['masks'], strict=True):
        use = valid & mask[row, col]
        y, x = dnb[use], grid[use]
        values = x - y
        if len(values):
            a1_native[label] = {
                'n': len(values),
                'native_mean': float(y.mean()),
                'a1_mean': float(x.mean()),
                'median_difference': float(np.median(values)),
                'mae': float(abs(values).mean()),
                'correlation': float(np.corrcoef(x, y)[0, 1]),
                'native_terrain_p10_p90_m': np.quantile(
                    height[use], [0.10, 0.90]
                ).tolist(),
            }
    return a1_native, same_scan_spread(
        lat, lon, height, distance, zenith, azimuth, scan
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('native', 'daily', 'maps', 'boundaries', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    geometry = read_json(args.boundaries)
    manifest = read_json(MANIFEST)['files']
    cases = {}
    for city, day in CASES.items():
        inputs = [v for v in manifest if v['city'] == city]
        paths = {
            v['filename'].split('.')[0]: args.native / v['filename'] for v in inputs
        }
        for record in inputs:
            if file_hash(args.native / record['filename']) != record['sha256']:
                raise ValueError('Unexpected native granule release.')
        info, data = preflight(
            args.daily,
            city,
            date.fromisoformat(day),
            geometry,
            args.maps,
            location=LOCATIONS[city],
            minimum_class_cells=1,
            compare_a2=False,
        )
        radiances, scan_spread = compare(paths, data)
        cases[city + '/' + day] = {
            'paired_native_centres': radiances,
            'same_scan_satellite_position_spread': scan_spread,
            'files': {v['filename']: v['sha256'] for v in inputs},
            'a1_files': info['files'],
        }
    write_json(args.output, cases)


if __name__ == '__main__':
    main()
