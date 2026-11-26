"""Repeat the Veneto SQM-site comparison with dated regional and spatial profiles."""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

from lightpollution.adapters.storage import file_hash, read_json, write_json
from lightpollution.data import read_region
from lightpollution.models.all_orders import AllOrderSettings
from lightpollution.models.boundary_orders import integrate_boundary_variants
from lightpollution.physics import EARTH_RADIUS_M, Atmosphere
from run_cams_dutch_pilot import area_mean_aod
from run_photometer_all_orders import NearBoundary
from run_photometer_italy import BOUNDS, SITES, summarize_run


@dataclass(frozen=True)
class VerticalGridAtmosphere(Atmosphere):
    """Piecewise-constant extinction in altitude bands and bilinear in position."""

    edges_m: tuple = ()
    tau_layers: np.ndarray | None = None
    latitudes: tuple = ()
    longitudes: tuple = ()

    def __post_init__(self):
        Atmosphere(**{key: getattr(self, key) for key in Atmosphere.__dataclass_fields__})
        edges, layers = np.asarray(self.edges_m), np.asarray(self.tau_layers)
        if (edges.ndim != 1 or len(edges) < 2 or not np.all(np.isfinite(edges))
                or edges[0] != 0 or edges[-1] != self.top_m or np.any(np.diff(edges) <= 0)):
            raise ValueError('Altitude edges must cover the atmosphere in increasing order.')
        if layers.ndim == 3:
            for axis in (self.latitudes, self.longitudes):
                if len(axis) < 2 or not np.all(np.isfinite(axis)) or np.any(np.diff(axis) <= 0):
                    raise ValueError('Invalid horizontal grid axes.')
            expected = (len(self.latitudes), len(self.longitudes), len(edges) - 1)
        else:
            expected = (len(edges) - 1,)
        if layers.shape != expected or not np.all(np.isfinite(layers) & (layers >= 0)):
            raise ValueError('Invalid layer optical depths.')

    @cached_property
    def layer_extinction(self):
        values = np.asarray(self.tau_layers) / np.diff(self.edges_m)
        return np.concatenate([values, np.zeros((*values.shape[:-1], 1))], axis=-1)

    @property
    def extinction_majorant(self):
        return self.rayleigh_sea_level + self.layer_extinction.max()

    def coefficients_at(self, positions_m):
        x, y, z = np.moveaxis(positions_m, -1, 0)
        radius_xy = np.hypot(x, y)
        height = np.maximum(np.hypot(radius_xy, z) - EARTH_RADIUS_M, 0)
        k = np.searchsorted(self.edges_m[1:], height, side='right')
        e = self.layer_extinction
        if e.ndim == 1:
            aerosol = e[k]
        else:
            lat = np.interp(np.rad2deg(np.arctan2(z, radius_xy)), self.latitudes, np.arange(len(self.latitudes)))
            lon = np.interp(np.rad2deg(np.arctan2(y, x)), self.longitudes, np.arange(len(self.longitudes)))
            i = np.minimum(lat.astype(int), len(self.latitudes)-2)
            j = np.minimum(lon.astype(int), len(self.longitudes)-2)
            fy, fx = lat-i, lon-j
            aerosol = ((1-fy)*((1-fx)*e[i,j,k] + fx*e[i,j+1,k])
                       + fy*((1-fx)*e[i+1,j,k] + fx*e[i+1,j+1,k]))
        rayleigh = self.rayleigh_sea_level * np.exp(-height/self.rayleigh_scale_height_m)
        return rayleigh, aerosol*self.aerosol_single_scattering_albedo, rayleigh + aerosol


def regional_profile(layers, latitudes, longitudes):
    return area_mean_aod(np.moveaxis(layers, -1, 0), latitudes, longitudes)


def aod_quantiles(depths, edges_m):
    total = depths.sum()
    if total <= 0:
        raise ValueError('Invalid optical depth for profile quantiles.')
    cumulative = np.r_[0., np.cumsum(depths)] / total
    return np.interp([.5, .9], cumulative, edges_m).tolist()


def calculate_month(job):
    site, month, satellite, source_meta, rows, levels, samples, seed = job
    year, month_number = map(int, month.split('-'))
    arrays, tiles = read_region(satellite / f'italy-{month}', BOUNDS, year, month_number)
    if {k: v['sha256'] for k, v in tiles.items()} != {
            k: v['sha256'] for k, v in source_meta['tiles'].items()}:
        raise ValueError(f'The pinned lights map changed in {month}.')
    observer = SITES[site]
    boundary = NearBoundary(arrays, observer, BOUNDS, (0, 3, 10))
    lat, lon, edges = levels
    outputs = []
    for i, (row, atmosphere_data) in enumerate(rows):
        date = row['date']
        layers = np.asarray(atmosphere_data['aod_per_layer'])
        region = regional_profile(layers, lat, lon)
        if not np.isclose(region.sum(), row['aod_regional'], rtol=0, atol=1e-10):
            raise ValueError(f'The regional CAMS AOD changed at {date}.')
        models = {
            'regional_layered': VerticalGridAtmosphere(edges_m=edges, tau_layers=region),
            'spatial_layered': VerticalGridAtmosphere(edges_m=edges, tau_layers=layers,
                                                       latitudes=lat, longitudes=lon)}
        centre_layers = np.array([
            np.interp(observer.latitude_deg, lat,
                      [np.interp(observer.longitude_deg, lon, line) for line in layer])
            for layer in np.moveaxis(layers, -1, 0)])
        result = {'site': site, 'date': date, 'geos_cf_version': atmosphere_data['geos_cf_version'],
                  'geos_cf_sources': atmosphere_data['geos_cf_sources'],
                  'aod_regional': row['aod_regional'], 'aod_at_station': row['aod_at_station'],
                  'region_z50_z90_m': aod_quantiles(region, edges),
                  'station_z50_z90_m': aod_quantiles(centre_layers, edges),
                  'variants': {}}
        for k, (label, atmosphere) in enumerate(models.items()):
            output = integrate_boundary_variants(observer, boundary, 0, 90, atmosphere,
                AllOrderSettings(samples=samples, seed=seed+2*i+k))
            result['variants'][label] = summarize_run(output)
            result['variants'][label]['extinction_majorant_per_m'] = float(atmosphere.extinction_majorant)
        outputs.append(result)
        print(f'{site} {date}: regional / spatial done', flush=True)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('previous', type=Path)
    parser.add_argument('profiles', type=Path)
    parser.add_argument('satellite', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--samples', type=int, default=524288)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    profiles = read_json(args.profiles)
    cams_path = args.previous / 'cams-input.json'
    if profiles['cams_sha256'] != file_hash(cams_path):
        raise ValueError('Mismatched CAMS AOD sources.')
    cams = read_json(cams_path)
    levels = tuple(tuple(profiles[k]) for k in ('latitudes', 'longitudes', 'altitude_edges_m'))
    if any(not np.array_equal(levels[i], cams[k]) for i, k in enumerate(('latitudes', 'longitudes'))):
        raise ValueError('Mismatched CAMS axes.')
    for day, source in profiles['days'].items():
        actual = np.asarray(source['aod_per_layer']).sum(axis=-1)
        expected = np.asarray(cams['aod_endpoints'][day]).mean(axis=0)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)

    jobs, inputs = [], {'profiles_sha256': file_hash(args.profiles)}
    for site in SITES:
        path = args.previous / (site.lower() + '.json')
        source = read_json(path)
        inputs[site] = file_hash(path)
        groups = defaultdict(list)
        for row in source['rows']:
            groups[row['date'][:7]].append((row, profiles['days'][row['date']]))
        for month, group in groups.items():
            jobs.append((site, month, args.satellite, source['monthly_sources'][month], group,
                         levels, args.samples, 66200+100*len(jobs)))

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for future in as_completed([executor.submit(calculate_month, job) for job in jobs]):
            rows.extend(future.result())
    rows.sort(key=lambda row: (row['site'], row['date']))
    write_json(args.output, {'direction': {'azimuth_deg': 0, 'altitude_deg': 90},
                             'inputs': inputs, 'samples': args.samples, 'rows': rows})


if __name__ == '__main__':
    main()
