"""Bind measured clear moonless nights to a portable regional source scale."""

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from ..domain.calibration import calibration_key
from ..domain.scene import DomainError
from ..models.shared_photometry import fit_shared_photometry
from ..observation import observation_context
from ..physics import Observer


def fit_dataset(dataset, results):
    if (not isinstance(dataset, dict) or dataset.get('schema_version') != 1
            or dataset.get('band') != 'Johnson-V' or not dataset.get('source')
            or not isinstance(dataset.get('samples'), list)):
        raise DomainError('invalid_photometry', 'Нужны схема 1, полоса Johnson-V, источник и измерения samples.')
    try:
        zone = ZoneInfo(dataset.get('timezone', 'UTC'))
        rows, scope, fingerprints, ids = [], None, {}, set()
        for row in dataset['samples']:
            if row['id'] in ids:
                raise ValueError('Duplicate observation identifier.')
            ids.add(row['id'])
            case = row['case_id']
            value = results[case]
            current_scope = value['calibration']['scope']
            if scope is None:
                scope = current_scope
            if current_scope != scope:
                raise ValueError('Model cases must share region normalisation, response band, solver and physical model.')
            if value['request']['profile'] != 'standard':
                raise ValueError('Ground fits require the standard numerical profile.')
            # Runtime, cache hits and stellar tables must not change the identity
            # of an otherwise identical calibration experiment.
            fingerprints[case] = calibration_key({'scope': current_scope,
                'observer': value['request']['observer'], 'period': value['request']['period'],
                'atmosphere': value['resolved_scene']['atmosphere'],
                'spectrum': value['request'].get('spectrum'),
                'directions': [{k: r[k] for k in ['azimuth_deg', 'altitude_deg', 'raw_transfer_response',
                                                'natural_relative', 'supported']}
                               for r in value['sky']['directions']]})
            az, alt = row.get('direction', [0, 90])
            az = float(az) % 360 if alt != 90 else 0.
            direction = next(r for r in value['sky']['directions']
                             if r['azimuth_deg'] == az and r['altitude_deg'] == float(alt))
            if not direction['supported']:
                raise ValueError('A measured direction is outside the supported scene.')
            instant = datetime.fromisoformat(row['utc'].replace('Z', '+00:00'))
            if instant.tzinfo is None or instant.utcoffset() is None:
                raise ValueError('Measurement time needs an explicit timezone.')
            utc = instant.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')
            observer = Observer(**value['request']['observer'])
            if 'observer' in row and row['observer'] != value['request']['observer']:
                raise ValueError('Measurement and model observers do not match.')
            context = observation_context(observer, utc, az, alt)
            if context['sun']['altitude_deg'] > -18 or context['moon']['altitude_deg'] > -5:
                raise ValueError('Initial calibration accepts only astronomical, moonless nights (Moon <= -5 deg).')
            night = (instant.astimezone(zone)-timedelta(hours=12)).date().isoformat()
            if row.get('night_id', night) != night:
                raise ValueError('Night identifier disagrees with local noon-to-noon grouping.')
            mag = row['observed_v_mag_arcsec2']
            if isinstance(mag, bool) or not isinstance(mag, (float, int)) or not 10 <= mag <= 30:
                raise ValueError('Observed V surface brightness must be in [10,30] mag/arcsec2.')
            # Moon is below -5 deg and its omitted residual is an explicit initial approximation.
            rows.append({'id': row['id'], 'night_id': night, 'site_id': row['site_id'], 'case_id': case,
                'role': row['role'], 'direction': [az, float(alt)], 'raw_response': direction['raw_transfer_response'],
                'observed_v_mag_arcsec2': mag, 'background_relative': direction['natural_relative'],
                'cloud_free_confirmed': row.get('cloud_free_confirmed'), 'qa_source': row.get('qa_source')})
        fit = fit_shared_photometry(rows)
    except (ValueError, KeyError, TypeError, StopIteration) as exc:
        raise DomainError('invalid_photometry', str(exc) or 'Некорректные фотометрические данные.') from exc
    document = {'schema_version': 1, 'kind': 'shared_regional_ground_fit', 'scope': scope,
        **fit, 'measurement_source': dataset['source'], 'measurement_rights': dataset.get('rights'),
        'dataset_sha256': calibration_key(dataset), 'case_result_sha256': fingerprints,
        'assumptions': ['One positive coefficient; natural background and atmospheric scenarios fixed by model cases.',
            'Johnson-V measurements mapped to the declared model V response; exact instrument response unverified.',
            'One clear interval does not validate a region or determine AOD, spectrum and source power independently.',
            'Common hypothetical ground surface; observer height is not a geographic DEM.']}
    return {**document, 'id': calibration_key(document)}
