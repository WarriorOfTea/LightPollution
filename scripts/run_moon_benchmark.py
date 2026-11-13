"""Evaluate the unchanged KS1991 block on published SAAO V photometry.

Selection uses geometry and metadata, never residuals or measured brightness.
Default and site extinction share all other inputs. No lunar fit is performed.
"""

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_body, solar_system_ephemeris
from astropy.time import Time
from astropy.utils import iers
import numpy as np

from lightpollution.adapters.storage import file_hash, write_json
from lightpollution.adapters.winkler import read_winkler
from lightpollution.models.observation import vertical_extinction_v
from lightpollution.moonlight import nanolambert_to_v_mag, scattered_moonlight, v_mag_to_nanolambert
from lightpollution.physics import Atmosphere, Observer

ROOT = Path(__file__).resolve().parents[1]
PAPER = 'https://arxiv.org/html/2205.07773'


def metrics(rows, name):
    residual = np.array([r['predictions'][name]['residual_mag_model_minus_observed'] for r in rows])
    nights = sorted({r['night_id'] for r in rows})
    if not len(rows):
        return {'records': 0, 'nights': 0, 'mae_mag': None, 'bias_mag': None, 'rmse_mag': None}
    per_night = []
    for night in nights:
        e = np.array([r['predictions'][name]['residual_mag_model_minus_observed'] for r in rows if r['night_id'] == night])
        per_night.append({'night_id': night, 'records': len(e), 'mae_mag': float(np.mean(abs(e))), 'bias_mag': float(np.mean(e))})
    return {'records': len(rows), 'nights': len(nights), 'mae_mag': float(np.mean(abs(residual))),
        'bias_mag': float(np.mean(residual)), 'rmse_mag': float(np.sqrt(np.mean(residual**2))),
        'median_model_over_observed_flux': float(np.median(10**(-.4*residual))),
        'equal_night_mae_mag': float(np.mean([r['mae_mag'] for r in per_night])), 'per_night': per_night}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'scratch/benchmark-research/responses/winkler-photometry.body')
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/benchmark-baseline-20261113/moon')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    # This is the official 1-m telescope location, used ONLY for coarse
    # twilight screening and lunar distance. Published 0.5-m angles stay inputs.
    observer = Observer(-(32+22/60+46/3600), 20+48/60+38.5/3600, 1798.)
    default_k = vertical_extinction_v(observer, Atmosphere())
    site_k = .138/(.4*np.log(10))
    protocol = {'id': 'LP-SAAO-KS-V-v1', 'created_at': datetime.now(ZoneInfo('Europe/Berlin')).isoformat(timespec='seconds'),
        'source': PAPER, 'source_sha256': file_hash(args.input), 'source_tables': [10, 13, 4],
        'site_for_auxiliary_ephemeris': asdict(observer),
        'site_source': 'https://www.saao.ac.za/astronomers/sutherland/',
        'auxiliary_site_is_not_exact_0_5m_telescope_position': True,
        'phase_angle': 'acos(2*published illuminated fraction-1)',
        'time_policy': 'HJD is not UTC; use HJD approximately as JD for auxiliary calculations only, bracket +/-10 min.',
        'primary_selection': {'illuminated_fraction': [.90, .98], 'moon_altitude_min_deg': 25,
            'target_altitude_min_deg': 30, 'separation_deg': [20, 90], 'abs_galactic_latitude_min_deg': 20,
            'sun_altitude_max_in_full_time_bracket_deg': -18, 'published_coordinates_required': True},
        'default_k_V_mag_airmass': default_k, 'site_k_V_mag_airmass': site_k,
        'site_k_origin': 'Table 4 tauV=.138; standard-star extinction consistent on the observing nights, not a fit to sky brightness.',
        'natural_V_mag_arcsec2': 21.6, 'natural_variation_diagnostic_mag': [21.0, 22.0],
        'fitted_lunar_parameters': [], 'residual_based_selection': False,
        'reference_is_measured_total_sky_not_author_model_subtraction': True,
        'limitations': ['No per-record cloud flag or photometric error bars supplied; conditional published-photometry test.',
            'Uniform natural background omits angular airglow/starlight/zodiacal variation.',
            'Site altitude/total extinction do not identify aerosol microphysics.',
            'This is V photometry, not eye RGB, camera noise or a full NASA scene validation.'],
        'source_code_sha256': {p: file_hash(ROOT/p) for p in ['scripts/run_moon_benchmark.py',
            'src/lightpollution/adapters/winkler.py', 'src/lightpollution/moonlight.py', 'src/lightpollution/models/observation.py']}}
    write_json(args.output/'protocol.json', protocol)
    rows = read_winkler(args.input.read_text())
    location = EarthLocation.from_geodetic(observer.longitude_deg*u.deg, observer.latitude_deg*u.deg, observer.height_m*u.m)
    jd = 2456000 + np.array([r['hjd_minus_2456000'] for r in rows])
    instants = Time(jd[:, None] + np.array([-10, 0, 10])[None, :]/1440, format='jd', scale='utc')
    with iers.conf.set_temp('auto_download', False), iers.conf.set_temp('auto_max_age', None), solar_system_ephemeris.set('builtin'):
        frame = AltAz(obstime=instants, location=location, pressure=0*u.hPa)
        sun_alt = get_body('sun', instants, location=location).transform_to(frame).alt.deg
        moon = get_body('moon', instants, location=location)
        moon_distance = moon.distance.to_value(u.km)
        moon_alt_check = moon.transform_to(frame).alt.deg
    rejected = Counter()
    primary, moonless = [], []
    for index, row in enumerate(rows):
        row['night_id'] = f"HJD-day-{int(row['hjd_minus_2456000'])}"
        row['sun_altitude_max_in_bracket_deg'] = float(sun_alt[index].max())
        row['moon_distance_km'] = float(moon_distance[index, 1])
        row['relative_distance_flux_span_in_time_bracket'] = float((moon_distance[index].max()/moon_distance[index].min())**2-1)
        galactic = None if row['ra_deg'] is None else float(SkyCoord(ra=row['ra_deg']*u.deg, dec=row['dec_deg']*u.deg).galactic.b.deg)
        row['galactic_latitude_deg'] = galactic
        if row['moon_zenith_deg'] is None:
            row['exclusion_reasons'] = ['published_moon_geometry_missing']
            if (np.max(moon_alt_check[index]) <= -5 and sun_alt[index].max() <= -18
                    and galactic is not None and abs(galactic) >= 20):
                moonless.append(row)
            rejected.update(row['exclusion_reasons'])
            continue
        moon_alt, target_alt = 90-row['moon_zenith_deg'], 90-row['target_zenith_deg']
        phase = row['illuminated_fraction']
        conditions = {'phase_outside_0.90_0.98': .9 <= phase <= .98,
            'moon_altitude_below_25': moon_alt >= 25,
            'target_altitude_below_30': target_alt >= 30,
            'separation_outside_20_90': 20 <= row['separation_deg'] <= 90,
            'galactic_plane_or_missing_coordinates': galactic is not None and abs(galactic) >= 20,
            'twilight_in_time_bracket': sun_alt[index].max() <= -18}
        row['exclusion_reasons'] = [name for name, accepted in conditions.items() if not accepted]
        rejected.update(row['exclusion_reasons'])
        if row['exclusion_reasons']:
            continue
        alpha = float(np.rad2deg(np.arccos(2*phase-1)))
        row['phase_angle_deg'] = alpha
        row['predictions'] = {}
        for name, k in [('default_atmosphere', default_k), ('site_extinction', site_k)]:
            lunar = float(scattered_moonlight(alpha, moon_alt, target_alt, row['separation_deg'], k, row['moon_distance_km']))
            natural = float(v_mag_to_nanolambert(21.6))
            predicted = float(nanolambert_to_v_mag(lunar+natural))
            flux_ratio = (lunar+natural)/float(v_mag_to_nanolambert(row['v_mag_arcsec2']))
            row['predictions'][name] = {'v_mag_arcsec2': predicted,
                'residual_mag_model_minus_observed': predicted-row['v_mag_arcsec2'],
                'model_over_observed_flux': flux_ratio, 'lunar_nL': lunar,
                'natural_fraction_of_predicted_total': natural/(lunar+natural),
                'natural_21_22_mag_result_span': [float(nanolambert_to_v_mag(lunar+v_mag_to_nanolambert(m))) for m in [21, 22]]}
        primary.append(row)
    if not primary:
        raise RuntimeError('No cases pass the prespecified protocol; do not loosen selection after seeing residuals.')
    report = {'protocol_sha256': file_hash(args.output/'protocol.json'), 'source_table_records': len(rows),
        'primary_records': len(primary), 'excluded_records': len(rows)-len(primary),
        'rejection_reason_counts_not_mutually_exclusive': dict(rejected),
        'missing_moon_geometry_rows': sum(r['moon_zenith_deg'] is None for r in rows),
        'missing_coordinate_rows': sum(r['ra_deg'] is None for r in rows),
        'default_atmosphere': metrics(primary, 'default_atmosphere'),
        'site_extinction': metrics(primary, 'site_extinction'),
        'max_relative_distance_flux_span_due_time_bracket': max(r['relative_distance_flux_span_in_time_bracket'] for r in primary),
        'max_natural_background_fraction_site_case': max(r['predictions']['site_extinction']['natural_fraction_of_predicted_total'] for r in primary),
        'largest_prediction_span_for_natural_21_22_mag': max(abs(np.diff(r['predictions']['site_extinction']['natural_21_22_mag_result_span'])[0]) for r in primary),
        'moonless_control_records': len(moonless), 'per_record_cloud_mask_available': False,
        'field_accuracy_certified': False, 'production_lunar_model_changed': False,
        'all_108_rows_scored': False}
    write_json(args.output/'selection-audit.json', {'rows': [{key: r[key] for key in ['row_id', 'patch', 'night_id',
        'coordinate_status', 'exclusion_reasons', 'sun_altitude_max_in_bracket_deg', 'galactic_latitude_deg']} for r in rows]})
    write_json(args.output/'primary-results.json', {'attribution': 'Winkler (2022), Tables 10/13; measured numerical inputs and our derived residuals.',
        'source': PAPER, 'records': primary})
    write_json(args.output/'moonless-controls.json', {'role': 'Diagnostic only; NOT fitted to lunar test or used to select residuals.', 'records': moonless})
    write_json(args.output/'summary.json', report)
    print(report)


if __name__ == '__main__':
    main()
