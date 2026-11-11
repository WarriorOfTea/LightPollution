"""Frozen real-data case for eye rendering 0.7; no fitted display target."""

import argparse
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from lightpollution.adapters.storage import read_json, write_json
from lightpollution.application.planner import Planner, export_result
from lightpollution.application.eye import build_eye_result, export_eye
from lightpollution.domain.fields import RunResult
from lightpollution.domain.scene import SceneSpec
from lightpollution.models.eye_field import heldout_coordinates, tangent_directions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('artifacts/eye-rendering-20261111'))
    parser.add_argument('--workspace', type=Path, default=Path('data/eye-model-20261111'))
    parser.add_argument('--compute-only', action='store_true')
    parser.add_argument('--render-saved', action='store_true')
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    scene = read_json(Path('artifacts/eye-model-20261111/city/scene.json'))
    scene['eye']['sampling'] = {'validation': 'heldout_24'}
    spec = SceneSpec.from_mapping(scene)
    points = heldout_coordinates(spec.to_dict()['eye']['view'])
    az, alt = tangent_directions(spec.to_dict()['eye']['view'], *points.T)
    criteria = out/'predeclared-criteria.json'
    if not criteria.exists():
        write_json(criteria, {'declared_at': datetime.now(ZoneInfo('Europe/Berlin')).isoformat(timespec='seconds'),
            'scene': spec.to_dict(), 'heldout_x_y': points.tolist(), 'heldout_az_alt_deg': np.stack([az, alt], axis=-1).tolist(),
            'heldout_used_for_refinement': False, 'interpolation_max_relative_error_target': .02,
            'refinement_target': .01, 'profile_comparison_indices': [0, 7, 15, 23],
            'flux_conservation_relative_error_target': .01, 'resolutions_px': [390, 900, 1800],
            'subpixel_phases': [[0., 0.], [.13, .37], [.5, .5], [.91, .87]],
            'sixteen_bit_rounding_bound_code': .5, 'eight_bit_dither_bound_code': 1.,
            'source_coefficient_refitted': False, 'independent_human_observers': 0,
            'colour_or_camera_claim': False})
    physical_path = out/'physical-result.json'
    if args.render_saved:
        city = read_json(physical_path)
    else:
        result = Planner(args.workspace, raw_directory=Path('data/blackmarble')).run(
            spec, progress=lambda p: print(p, flush=True))
        city = result.to_dict()
        write_json(physical_path, city)
    if args.compute_only:
        print('Physical result saved', physical_path, flush=True)
        return
    export_result(RunResult(city), out/'city')
    city = read_json(out/'city/result.json')
    field = deepcopy(city['eye']['field'])
    for node in [field, field.get('view_patch', {})]:
        for key in ['city_photopic_luminance_cd_m2', 'city_scotopic_luminance_cd_m2']:
            if key in node:
                node[key] = np.zeros_like(node[key]).tolist()
        if 'validation' in node:
            node['inherited_city_on_validation'] = node.pop('validation')
            node['validation_scope_note'] = 'City-on interpolation check; not a second direct transfer experiment for city-off.'
    field['counterfactual_city_emission_multiplier'] = 0.
    dark = deepcopy(city)
    dark['eye'] = build_eye_result(dark, city['eye']['config'], field,
        city['eye']['spectral_bridge']['reference_A0_scotopic_lux_at_observed_V0'])
    from lightpollution.interfaces.eye_view import render_eye_view
    dark['eye']['display'] = render_eye_view(dark, out/'city-off', title='ТА ЖЕ ТОЧКА · ГОРОДСКИЕ ИСТОЧНИКИ УСЛОВНО ОТКЛЮЧЕНЫ')
    export_eye(dark['eye'], out/'city-off')
    write_json(out/'city-off/result.json', {'artifact_type': 'eye_only_counterfactual',
        'request': dark['request'], 'eye': dark['eye'], 'same_stellar_fluxes_and_atmosphere': True,
        'same_display_profile': True, 'not_a_different_measured_site': True})
    write_json(out/'case-summary.json', {'city': city['eye']['summary'], 'city_display': city['eye']['display'],
        'city_off': dark['eye']['summary'], 'city_off_display': dark['eye']['display'],
        'refinement': city['eye']['field']['view_patch']['refinement'],
        'validation': city['eye']['field']['view_patch']['validation'],
        'cache': city['cache'], 'runtime': city['runtime'],
        'source_coefficient': city['calibration']['source_coefficient'], 'field_accuracy_certified': False})
    print('Rendered', city['eye']['display']['displayed_stars'], dark['eye']['display']['displayed_stars'], flush=True)


if __name__ == '__main__':
    main()
