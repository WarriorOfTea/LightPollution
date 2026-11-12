"""Reproduce the bounded aerosol sensitivity example documented in MODELS.

No network, satellite inverse retrieval, fitted scale or regional weather is
involved. All comparisons retain the same one-source upward power.
"""

import argparse
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from lightpollution.adapters.storage import file_hash, write_json
from lightpollution.physics import (EARTH_RADIUS_M, Atmosphere, Observer,
    SolverSettings, Sources, hg_phase, integrate_direction, rayleigh_phase,
    unit_position)


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
        default=ROOT / 'artifacts/scattering-notation-20261112/sensitivity.json')
    args = parser.parse_args()
    atmosphere = Atmosphere()
    observer = Observer(60., 30., 20.)
    distance = 40_000.
    latitude = observer.latitude_deg - np.rad2deg(distance / EARTH_RADIUS_M)
    position = EARTH_RADIUS_M * unit_position(np.array([latitude]), np.array([30.]))
    sources = Sources(position, np.array([1.]))
    altitudes = [10., 45., 90.]
    settings = SolverSettings(192, 24)
    refined = SolverSettings(384, 48)

    def responses(atmosphere, numerical):
        return {str(int(alt)): integrate_direction(observer, sources, 180., alt,
                    atmosphere, numerical) for alt in altitudes}

    reference = responses(atmosphere, settings)
    reference_refined = responses(atmosphere, refined)
    rows = []
    for parameter, values in [('aerosol_optical_depth', [.02, .10, .30]),
                               ('aerosol_g', [.6, .8, .9])]:
        for value in values:
            scenario = replace(atmosphere, **{parameter: value})
            response = responses(scenario, settings)
            check = responses(scenario, refined)
            rows.append({'parameter': parameter, 'value': value,
                'response_by_altitude': response, 'refined_response_by_altitude': check,
                'ratio_to_baseline_by_altitude': {k: response[k] / reference[k] for k in response},
                'refined_ratio_to_baseline_by_altitude': {k: check[k] / reference_refined[k] for k in check},
                'relative_change_on_refinement': {k: abs(response[k] / check[k] - 1) for k in response}})
    molecular, aerosol_scattering, total_extinction = map(float, atmosphere.coefficients(0.))
    record = {'completed_at': datetime.now(ZoneInfo('Europe/Berlin')).isoformat(timespec='seconds'),
        'kind': 'controlled_single_source_sensitivity_not_field_validation',
        'observer': asdict(observer), 'source': {'latitude_deg': float(latitude),
            'longitude_deg': 30., 'surface_arc_distance_m': distance,
            'upward_power_arbitrary_units': 1., 'emission': 'Lambertian'},
        'wavelength_nm': 550., 'azimuth_deg': 180., 'altitudes_deg': altitudes,
        'atmosphere': asdict(atmosphere), 'settings': asdict(settings),
        'refined_settings': asdict(refined), 'source_scale_refitted': False,
        'terrain_or_ground_reflection': False, 'satellite_data_used': False,
        'surface_coefficients_m-1': {'Rayleigh': molecular,
            'aerosol_scattering': aerosol_scattering, 'total_extinction': total_extinction},
        'local_aerosol_to_Rayleigh_scattering_ratio': {
            str(theta): aerosol_scattering * float(hg_phase(np.cos(np.deg2rad(theta)), atmosphere.aerosol_g))
                / (molecular * float(rayleigh_phase(np.cos(np.deg2rad(theta)))))
            for theta in [0, 90, 180]},
        'vertical_Rayleigh_optical_depth_to_infinity': molecular * atmosphere.rayleigh_scale_height_m,
        'vertical_total_scattering_optical_depth_to_infinity': molecular * atmosphere.rayleigh_scale_height_m
            + atmosphere.aerosol_optical_depth * atmosphere.aerosol_single_scattering_albedo,
        'baseline_response_by_altitude': reference, 'scenarios': rows,
        'code_sha256': {name: file_hash(ROOT / name) for name in
            ['scripts/audit_scattering.py', 'src/lightpollution/physics.py']},
        'limitation': 'A sensitivity example, not a measured error, regional probability or multiple-scattering benchmark.'}
    write_json(args.output, record)
    maximum = max(v for row in rows for v in row['relative_change_on_refinement'].values())
    print(f'{len(rows)} scenarios x {len(altitudes)} directions; refinement max relative change {maximum:.6g}')


if __name__ == '__main__':
    main()
