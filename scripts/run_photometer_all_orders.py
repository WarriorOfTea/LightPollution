"""Recalculate the Dutch CAMS photometry pilot with all scattering orders."""
import argparse
from dataclasses import dataclass, replace
from itertools import combinations
from pathlib import Path

import numpy as np

from lightpollution.adapters.storage import file_hash, read_json, write_json
from lightpollution.data import PIXELS_PER_DEGREE, good_mask, read_region
from lightpollution.models.all_orders import AllOrderSettings
from lightpollution.models.background import BackgroundModel, background_config
from lightpollution.models.boundary_orders import integrate_boundary_variants
from lightpollution.observation import observation_context
from lightpollution.physics import EARTH_RADIUS_M, Atmosphere, Observer, unit_position
from run_cams_dutch_pilot import BOUNDS, SITES, area_mean_aod


@dataclass(frozen=True)
class GridAtmosphere(Atmosphere):
    latitudes: tuple = ()
    longitudes: tuple = ()
    grid: tuple = ()

    def __post_init__(self):
        Atmosphere(**{k: getattr(self, k) for k in Atmosphere.__dataclass_fields__})
        if (len(self.latitudes) < 2 or len(self.longitudes) < 2
                or not np.all(np.isfinite(self.latitudes)) or not np.all(np.isfinite(self.longitudes))
                or not np.all(np.diff(self.latitudes) > 0) or not np.all(np.diff(self.longitudes) > 0)):
            raise ValueError('The atmospheric axes must be finite and strictly increasing.')
        grid = np.asarray(self.grid)
        if (grid.shape != (len(self.latitudes), len(self.longitudes))
                or not np.all(np.isfinite(grid) & (grid >= 0))):
            raise ValueError('AOD grid must be nonnegative, finite and complete.')

    @property
    def extinction_majorant(self):
        return self.rayleigh_sea_level + np.max(self.grid) / self.aerosol_scale_height_m

    def coefficients_at(self, positions_m):
        x, y, z = np.moveaxis(positions_m, -1, 0)
        radius_xy = np.hypot(x, y)
        height = np.maximum(np.hypot(radius_xy, z) - EARTH_RADIUS_M, 0)
        lat = np.interp(np.rad2deg(np.arctan2(z, radius_xy)), self.latitudes, np.arange(len(self.latitudes)))
        lon = np.interp(np.rad2deg(np.arctan2(y, x)), self.longitudes, np.arange(len(self.longitudes)))
        i = np.minimum(lat.astype(int), len(self.latitudes)-2)
        j = np.minimum(lon.astype(int), len(self.longitudes)-2)
        fy, fx = lat-i, lon-j
        g = np.asarray(self.grid)
        aod = (1-fy)*((1-fx)*g[i,j]+fx*g[i,j+1]) + fy*((1-fx)*g[i+1,j]+fx*g[i+1,j+1])
        molecular = self.rayleigh_sea_level*np.exp(-height/self.rayleigh_scale_height_m)
        aerosol = aod/self.aerosol_scale_height_m*np.exp(-height/self.aerosol_scale_height_m)
        return molecular, aerosol*self.aerosol_single_scattering_albedo, molecular+aerosol


class NearBoundary:
    def __init__(self, arrays, observer, bounds=BOUNDS, radii_km=(3, 10)):
        self.bounds = bounds
        self.labels = tuple(f'outside_{radius}km' for radius in radii_km)
        key = 'NearNadir_Composite_Snow_Free'
        self.known = good_mask(arrays[key], arrays[key+'_Quality'], arrays[key+'_Num']) & np.isin(
            arrays['Land_Water_Mask'], [0, 1, 5])
        radiance = np.where(self.known, arrays[key], 0)
        normals = unit_position(bounds.latitude[:,None], bounds.longitude[None,:])
        chord = np.linalg.norm(normals-unit_position(observer.latitude_deg, observer.longitude_deg), axis=-1)
        distance = 2*EARTH_RADIUS_M*np.arcsin(np.minimum(1, chord/2))
        self.radiance = np.stack([np.where(distance >= radius * 1000, radiance, 0)
                                  for radius in radii_km], axis=-1)
        self.upper_bounds = self.radiance.max(axis=(0, 1))

    def sample(self, points, outgoing):
        x, y, z = points.T
        lat = np.rad2deg(np.arctan2(z,np.hypot(x,y)))
        lon = np.rad2deg(np.arctan2(y,x))
        row = np.floor((self.bounds.north-lat)*PIXELS_PER_DEGREE).astype(int)
        col = np.floor((lon-self.bounds.west)*PIXELS_PER_DEGREE).astype(int)
        n, m = self.known.shape
        inside = (row >= 0) & (row < n) & (col >= 0) & (col < m)
        row, col = row.clip(0,n-1), col.clip(0,m-1)
        return np.where(inside[:,None],self.radiance[row,col],0), np.where(
            inside, 1+self.known[row,col].astype(int), 0)


def estimates(result):
    mean = np.asarray(result['category_mean_radiance'])[0]
    cov = np.asarray(result['category_mean_covariance_per_variant'])[0]
    return {'all': result['all_order_radiance'][0], 'all_se': result['all_order_standard_error'][0],
            'first': float(mean[0]), 'first_se': float(np.sqrt(max(0,cov[0,0]))),
            'all_first_cov': float(cov[0].sum()), 'far': result['all_order_radiance'][1],
            'positive': result['positive_paths'][0], 'effective': result['effective_score_samples'][0],
            'highest_order': result['max_scattering_order_seen'], 'seconds': result['runtime_seconds']}


def score(rows, background):
    checks = []
    for site in ('Schiermonnikoog-dorp', 'Vlieland-Oost'):
        train, test = sorted((r for r in rows if r['site'] == site),key=lambda r:r['day'])
        case = train['variants']['current']
        for order in ('first','all'):
            scale = (train['observed_relative']-background)/case[order]
            if scale <= 0:
                raise ValueError('Nonpositive photometric scale.')
            for name, prediction in test['variants'].items():
                flux = scale*prediction[order]+background
                residual = -2.5*np.log10(flux/test['observed_relative'])
                # The baseline estimate is reused at both times, so its noise cancels.
                flux_se = 0.
                if name != 'current':
                    scale_se = scale*case[order+'_se']/case[order]
                    flux_se = np.hypot(scale*prediction[order+'_se'], prediction[order]*scale_se)
                checks.append({'site': site, 'train': train['day'], 'check': test['day'],
                               'variant': name, 'order': order, 'residual_mag': float(residual),
                               'mc_se_mag': float(2.5/np.log(10)*flux_se/flux)})
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('satellite', type=Path)
    parser.add_argument('pilot', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--samples', type=int, default=524288)
    parser.add_argument('--control-only', action='store_true',
                        help='Refine only the two calibration/check pairs.')
    args = parser.parse_args()
    pilot = read_json(args.pilot)
    regional_aod = area_mean_aod(np.asarray(pilot['aod_grid']), np.asarray(pilot['grid_latitudes']),
                                 np.asarray(pilot['grid_longitudes']))
    arrays, tiles = read_region(args.satellite, BOUNDS, 2024, 1)
    rows, distances, current = [], {}, {}
    for (a, p), (b, q) in combinations(SITES.items(),2):
        chord = np.linalg.norm(unit_position(*p)-unit_position(*q))
        distances[f'{a} / {b}'] = float(2*EARTH_RADIUS_M*np.arcsin(chord/2)/1000)
    atmosphere = Atmosphere()
    runs = 0
    for row in pilot['station_windows']:
        site, day = row['site'], row['day']
        if args.control_only and site not in ('Schiermonnikoog-dorp', 'Vlieland-Oost'):
            continue
        observer = Observer(*SITES[site],5)
        if site not in current:
            boundary = NearBoundary(arrays, observer)
            first = integrate_boundary_variants(observer,boundary,0,90,atmosphere,
                                                AllOrderSettings(samples=args.samples, seed=40971+len(current)
                                                                 + 10000*args.control_only))
            current[site] = boundary, estimates(first)
            runs += 1
        boundary, base = current[site]
        j = pilot['days'].index(day)
        grid = GridAtmosphere(latitudes=tuple(pilot['grid_latitudes']),
                              longitudes=tuple(pilot['grid_longitudes']),
                              grid=tuple(map(tuple,pilot['aod_grid'][j])))
        regional = replace(atmosphere,aerosol_optical_depth=float(regional_aod[j]))
        context = observation_context(observer,day+'T22:30:00Z',0,90)
        natural = BackgroundModel(background_config({}),observer,regional,21.6).evaluate(
            np.array([0]),np.array([90]),context)[0]['natural_relative']
        variants = {'current':base}
        for offset,(name,atm) in enumerate((('cams_region',regional),('cams_spatial',grid))):
            if args.control_only and day != '2024-01-10':
                continue
            values = integrate_boundary_variants(observer,boundary,0,90,atm,AllOrderSettings(
                samples=args.samples,seed=41000+2*len(rows)+offset+10000*args.control_only))
            variants[name] = estimates(values)
            runs += 1
        rows.append({'site':site,'day':day,'n':row['n'],'observed_relative':row['observed_relative'],
                     'natural_relative':natural,'variants':variants})
        print(site,day,flush=True)
    natural = rows[0]['natural_relative']
    if not np.allclose([r['natural_relative'] for r in rows],natural,atol=0,rtol=1e-12):
        raise RuntimeError('Zenith normalization must agree across windows.')
    write_json(args.output,{'samples_per_run':args.samples,'runs':runs,
                           'pilot_sha256':file_hash(args.pilot),'tiles':{k:v['sha256'] for k,v in tiles.items()},
                           'regional_aod':dict(zip(pilot['days'],regional_aod.tolist(),strict=True)),
                           'distances_km':distances,'background_v_mag':21.6,'background_relative':natural,
                           'rows':rows,'checks':score(rows,natural),
                           'background_sensitivity':score(rows,10**(-.4*(22-22)))})


if __name__ == '__main__':
    main()
