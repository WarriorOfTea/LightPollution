"""Compare the spatial decay of one clear Berlin skyglow transect with transport."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from hashlib import sha256
from itertools import combinations
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from lightpollution.adapters.spectral_library import load_spectral_library
from lightpollution.adapters.storage import file_hash, read_json, write_json
from lightpollution.data import Bounds, PIXELS_PER_DEGREE, good_mask, read_region
from lightpollution.models.all_orders import AllOrderSettings
from lightpollution.models.angular_sources import empirical_radiance
from lightpollution.models.boundary_orders import integrate_boundary_variants
from lightpollution.models.spectra import band_integral, wavelength_atmosphere
from lightpollution.observation import observation_context
from lightpollution.physics import EARTH_RADIUS_M, Atmosphere, Observer, unit_position
from run_city_source_transport import grid_city_indices


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / 'artifacts' / 'berlin-propagation-20261126'
OBSERVATIONS = ARTIFACT / 'observations.json'
DNB = ROOT / 'artifacts' / 'photometer-sqm-residuals-20261126' / 'viirs-dnb-rsr.dat'
BOUNDS = Bounds(10.2, 50.1, 16.6, 55.3)
NEAR, OFF = (f'{k}_Composite_Snow_Free' for k in ('NearNadir', 'OffNadir'))
EDGES = (380., 490., 540., 585., 630., 780.)
MONTH, YEAR = 'berlin-2017-03', 'berlin-2017'
TERMS = ('rest', 'berlin_lambert', 'berlin_uniform', 'berlin_off_h1', 'berlin_off_h2',
         'within_3km', 'within_10km', 'monthly_lambert')
FORMS = {'lambert': (1, 0, 0, 0), 'uniform_intensity': (0, 1, 0, 0),
         'half_lambert_half_uniform': (.5, .5, 0, 0), 'off_h1': (0, 0, 1, 0), 'off_h2': (0, 0, 0, 1)}
MU_FLOOR = np.cos(np.deg2rad(85.))
MU_NEAR = np.cos(np.deg2rad(20.))
UNIFORM_NORM = -np.log(MU_NEAR) / (1 - MU_NEAR)


def visible_bands():
    library = load_spectral_library()
    dnb_wave, rsr = np.loadtxt(DNB, unpack=True)
    wave = np.unique(np.r_[library.wavelengths, dnb_wave, EDGES])
    wave = wave[(wave >= 380) & (wave <= 780)]
    dnb = np.interp(wave, dnb_wave, rsr, left=0, right=0)
    v = np.interp(wave, library.wavelengths, library.responses['photopic'], left=0, right=0)
    spectra = {}
    for name in ('HP1', 'LED-B2'):
        sed = np.interp(wave, library.wavelengths, library.templates[name])
        spectra[name] = sed*v/band_integral(wave, sed, dnb)
    spectra['equal_dnb'] = (spectra['HP1'] + spectra['LED-B2']) / 2
    # 1 nW/cm² = 1e-5 W/m². Multiply by 683 lm/W and 1000 mcd/cd.
    conversion = 1e-5 * 683 * 1000
    bins = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:], strict=True):
        selected = (lo <= wave) & (wave <= hi)
        x = wave[selected]
        weights = {name: float(np.trapezoid(s[selected], x)) for name, s in spectra.items()}
        node = np.trapezoid(x*spectra['equal_dnb'][selected], x) / weights['equal_dnb']
        bins.append({'wavelength_nm': float(node), 'mcd_m2_per_dnb_unit': {k: conversion*w for k, w in weights.items()}})
    return bins


def map_layers(source, period, month, *, bounds=BOUNDS, year=2017):
    target = source / period
    manifest = read_json(target / 'manifest.json')
    layers, tiles = read_region(target, bounds, year, month)
    hashes = {record['name']: record['sha256'] for record in manifest['files']}
    if hashes != {tile['filename']: tile['sha256'] for tile in tiles.values()}:
        raise ValueError(f'Unexpected Black Marble release in {period}.')
    land = np.isin(layers['Land_Water_Mask'], [0, 1, 5])
    near = land & good_mask(layers[NEAR], layers[NEAR+'_Quality'], layers[NEAR+'_Num'])
    off = near & good_mask(layers[OFF], layers[OFF+'_Quality'], layers[OFF+'_Num'])
    return (np.where(near, layers[NEAR], 0.), np.where(off, layers[OFF], 0.), near, off), hashes


class BerlinBoundary:
    labels = TERMS

    def __init__(self, monthly, annual, berlin, *, bounds=BOUNDS):
        n, o, n_ok, o_ok = monthly
        y, q, y_ok, q_ok = annual
        self.monthly, self.berlin = n, berlin
        self.bounds = bounds
        self.near = np.where(n_ok, n, y)
        self.off = np.where(n_ok, o, q)
        self.known = n_ok | y_ok
        self.paired = np.where(n_ok, o_ok, q_ok)
        max_near, max_off = self.near.max(), self.off.max()
        self.upper_bounds = np.array([max_near, max_near, max_near/(MU_FLOOR*UNIFORM_NORM),
                                      max(max_near, max_off), max(max_near, 2*max_off), max_near, max_near, n.max()])
        weights = np.cos(np.deg2rad(bounds.latitude))[:, None]*berlin
        sum_paired = float((weights*self.near*self.paired).sum())
        self.map_diagnostics = {'berlin_grid_cells': int(berlin.sum()), 'near_month_cells_berlin': int((berlin & n_ok).sum()),
            'near_combined_cells_berlin': int((berlin & self.known).sum()),
            'near_off_combined_cells_berlin': int((berlin & self.paired).sum()),
            'paired_off_over_near_area_weighted': float((weights*self.off*self.paired).sum()/sum_paired),
            'annual_fill_fraction_of_combined_near_radiance_berlin': float((weights*self.near*~n_ok).sum()/(weights*self.near).sum()),
            'annual_fill_cells_region': int((~n_ok & y_ok).sum())}

    def at(self, observer):
        self.observer_direction = unit_position(observer.latitude_deg, observer.longitude_deg)
        return self

    def sample(self, points, outgoing):
        x, y, z = points.T
        lat = np.rad2deg(np.arctan2(z, np.hypot(x, y)))
        lon = np.rad2deg(np.arctan2(y, x))
        row = np.floor((self.bounds.north-lat)*PIXELS_PER_DEGREE).astype(int)
        col = np.floor((lon-self.bounds.west)*PIXELS_PER_DEGREE).astype(int)
        h, w = self.bounds.shape
        inside = (row >= 0) & (row < h) & (col >= 0) & (col < w)
        row, col = row.clip(0, h-1), col.clip(0, w-1)
        near = np.where(inside, self.near[row, col], 0.)
        off = self.off[row, col]
        berlin = inside & self.berlin[row, col]
        support = inside & self.paired[row, col]
        mu = np.clip(np.einsum('ij,ij->i', points, outgoing)/np.linalg.norm(points, axis=1), 0, 1)
        chord = np.linalg.norm(points/EARTH_RADIUS_M-self.observer_direction, axis=1)
        held = np.where(support, empirical_radiance(near, off, mu, 1.), near)
        slope = np.where(support, off*np.maximum(0, 1-2*mu), 0.)
        values = [near*~berlin, near*berlin, near*berlin/(np.maximum(mu, MU_FLOOR)*UNIFORM_NORM),
                  held*berlin, (held+slope)*berlin, near*(chord < 2*np.sin(3000/(2*EARTH_RADIUS_M))),
                  near*(chord < 2*np.sin(10000/(2*EARTH_RADIUS_M))), self.monthly[row,col]*inside]
        return np.column_stack(values), np.where(inside, 1+self.known[row,col].astype(int), 0)


def solve_stop(job):
    observation, boundary, atmospheres, bins, alpha, aperture, samples = job
    observer = Observer(observation['latitude'], observation['longitude'], observation['observer_height_m'])
    boundary.at(observer)
    cases = {}
    for name, aod in atmospheres.items():
        atmosphere = Atmosphere(aerosol_optical_depth=aod)
        bands = []
        for j, b in enumerate(bins):
            at_wave = wavelength_atmosphere(atmosphere, b['wavelength_nm'], alpha)
            seed = int.from_bytes(sha256(f'berlin/{observation["stop"]}/{name}/{j}'.encode()).digest()[:4], 'little')
            r = integrate_boundary_variants(observer, boundary, 0., 90., at_wave,
                AllOrderSettings(samples=samples, seed=seed, recorded_orders=1), aperture_radius_deg=aperture)
            bands.append({k: r[k] for k in ('all_order_radiance', 'mean_covariance', 'effective_score_samples',
                                          'max_scattering_order_seen')} | {'seed': seed})
        values = np.asarray([band['all_order_radiance'] for band in bands])
        covariance = np.asarray([band['mean_covariance'] for band in bands])
        spectra = {}
        for lamp in bins[0]['mcd_m2_per_dnb_unit']:
            weight = np.array([b['mcd_m2_per_dnb_unit'][lamp] for b in bins])
            total = weight @ values
            cov = np.tensordot(weight**2, covariance, axes=1)
            spectra[lamp] = {'terms': dict(zip(TERMS, total.tolist(), strict=True)), 'covariance': cov.tolist()}
        cases[name] = {'spectral_radiance_runs': bands, 'photopic_mcd_m2': spectra}
    return observation | {'atmospheres': cases}


def projection(row, form, atmosphere, lamp, natural):
    spectrum = row['atmospheres'][atmosphere]['photopic_mcd_m2'][lamp]
    terms = spectrum['terms']
    weights = np.array([1, *FORMS[form], 0, 0, 0])
    city = float(np.dot([terms[name] for name in TERMS[1:5]], FORMS[form]))
    total = terms['rest'] + city
    uncertainty = np.sqrt(max(0., weights @ np.asarray(spectrum['covariance']) @ weights))
    return {'artificial_mcd_m2': total, 'berlin_mcd_m2': city, 'total_mcd_m2': total + natural,
            'city_fraction': city/total, 'mc_standard_error_mcd_m2': float(uncertainty),
            'local_3km_lambert_fraction': terms['within_3km']/(terms['rest']+terms['berlin_lambert']),
            'local_10km_lambert_fraction': terms['within_10km']/(terms['rest']+terms['berlin_lambert']),
            'monthly_original_lambert_total': terms['monthly_lambert'] + natural}


def pair_class(first, second, tolerance):
    relative = (first-second)/((first+second)/2)
    return np.sign(relative) if abs(relative) > tolerance else 0


def metrics(rows, predicted, natural):
    measured = np.array([r['luminance_mcd_m2'] for r in rows])
    calculated = np.array([r['total_mcd_m2'] for r in predicted])
    art_obs = measured - natural
    artificial = calculated - natural
    signed = 100*(calculated/measured - 1.)
    pairs = list(combinations(range(len(rows)), 2))
    order = {}
    for tolerance in (.05, .1, .2):
        expected = np.array([pair_class(measured[i], measured[j], tolerance) for i,j in pairs])
        actual = np.array([pair_class(calculated[i], calculated[j], tolerance) for i,j in pairs])
        order[str(tolerance)] = {'pairs': len(pairs), 'correct_three_classes': int((expected==actual).sum()),
            'observed_near_equal': int((expected==0).sum()), 'correct_near_equal': int(((expected==0)&(actual==0)).sum()),
            'observed_distinct': int((expected!=0).sum()), 'correct_distinct': int(((expected!=0)&(expected==actual)).sum()),
            'reversed_distinct': int((expected*actual<0).sum())}
    shape_difference = [-2.5*np.log10((artificial[i]/artificial[j])/(art_obs[i]/art_obs[j])) for i,j in pairs]
    return {'n': len(rows), 'mean_signed_pct': float(signed.mean()), 'mean_absolute_pct': float(np.abs(signed).mean()),
            'mean_absolute_artificial_pct': float(np.abs(100*(artificial/art_obs - 1)).mean()),
            'pairs_full_luminance': order,
            'mean_absolute_artificial_pair_ratio_error_mag': float(np.abs(shape_difference).mean()) if pairs else None,
            'outer_to_inner_artificial_ratio': {'model': float(artificial[-1]/artificial[0]),
                                                 'observed': float(art_obs[-1]/art_obs[0])}}


def summarize(rows, atmospheres, natural):
    results = {}
    for row in rows:
        row['prediction'] = {name: {lamp: {form: projection(row, form, name, lamp, natural) for form in FORMS}
                                   for lamp in ('HP1','LED-B2','equal_dnb')} for name in atmospheres}
    primary = lambda r: r['prediction']['aeronet']['equal_dnb']
    groups = {'all': rows, 'outside_15_km': [r for r in rows if r['distance_km'] >= 15],
              'berlin_dominates_outside_15_km': [r for r in rows if r['distance_km'] >= 15 and all(
                  primary(r)[shape]['city_fraction'] > .5 for shape in ('lambert','off_h1','off_h2'))],
              'all_berlin_dominant': [r for r in rows if all(primary(r)[shape]['city_fraction'] > .5
                                                            for shape in ('lambert','off_h1','off_h2'))]}
    for group, selection in groups.items():
        if not selection:
            continue
        results[group] = {'stops':[r['stop'] for r in selection], 'metrics': {
            atm: {lamp: {form: metrics(selection, [r['prediction'][atm][lamp][form] for r in selection], natural)
                        for form in FORMS} for lamp in ('HP1','LED-B2','equal_dnb')} for atm in atmospheres}}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('maps', type=Path)
    parser.add_argument('boundaries', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--samples', type=int, default=131072)
    args = parser.parse_args()
    obs = read_json(OBSERVATIONS)
    aod_alpha, offset = np.polyfit(np.log(np.asarray(obs['aeronet_wavelengths_nm']) / 550.), np.log(obs['aeronet_aod']), 1)
    atmosphere = {'aeronet': float(np.exp(offset)), 'standard_010': .1, 'higher_022': .22}
    bins = visible_bands()
    feature, = [f for f in read_json(args.boundaries)['features'] if f['properties']['URAU_CODE'] == 'DE001C']
    city = grid_city_indices(BOUNDS, {'Berlin': feature['geometry']}, ('Berlin',)) == 1
    month, m_sha = map_layers(args.maps, MONTH, 3)
    year, y_sha = map_layers(args.maps, YEAR, None)
    boundary = BerlinBoundary(month, year, city)
    source_rows = []
    tz = ZoneInfo(obs['time_zone'])
    for values in obs['observations']:
        record = dict(zip(obs['columns'], values, strict=True))
        time = datetime.fromisoformat(obs['date_local']+'T'+record['time_local']).replace(tzinfo=tz)
        time = time.astimezone(ZoneInfo('UTC')).isoformat().replace('+00:00', 'Z')
        height = record['elevation_m'] + obs['assumed_tripod_height_m']
        context = observation_context(Observer(record['latitude'], record['longitude'], height), time, 0, 90)
        solar, lunar = (context[x]['altitude_deg'] for x in ('sun','moon'))
        if solar > -18 or lunar > -6:
            raise ValueError(f'Observed stop {record["stop"]} violates the dark, moonless limits.')
        source_rows.append(record | {'utc': time, 'observer_height_m': height,
                                    'sun_altitude_deg': solar, 'moon_altitude_deg': lunar})
    with ProcessPoolExecutor(args.workers) as pool:
        tasks = [pool.submit(solve_stop, (row, boundary, atmosphere, bins, -float(aod_alpha), obs['aperture_radius_deg'], args.samples))
                 for row in source_rows]
        rows = []
        for task in as_completed(tasks):
            solved = task.result()
            rows.append(solved)
            print(solved['stop'], solved['site'], flush=True)
    rows.sort(key=lambda r: r['stop'])
    background = obs['table_2_reference_natural_photopic_mcd_m2']
    summary = summarize(rows, atmosphere, background)
    write_json(args.output, {'observations_sha256': file_hash(OBSERVATIONS), 'dnb_response_sha256': file_hash(DNB),
        'gis_boundaries_sha256': file_hash(args.boundaries), 'gis_city_code': 'DE001C', 'tiles_sha256': m_sha | y_sha,
        'bounds': BOUNDS.__dict__, 'map_diagnostics': boundary.map_diagnostics, 'aperture_radius_deg': obs['aperture_radius_deg'],
        'angstrom_exponent': -float(aod_alpha), 'aod_550': atmosphere, 'mcd_m2_natural': background, 'spectral_bands': bins,
        'spectral_library': load_spectral_library().identity, 'samples_per_run': args.samples, 'source_boundary_terms': TERMS,
        'uniform_intensity_cap_zenith_deg': 85., 'metrics': summary, 'rows': rows})


if __name__ == '__main__':
    main()
