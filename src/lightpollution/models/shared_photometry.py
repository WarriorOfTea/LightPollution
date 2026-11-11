"""One positive source coefficient, equal night weights, disjoint validation."""

from collections import defaultdict

import numpy as np


def fit_shared_photometry(samples: list[dict], reference_mag: float = 22.) -> dict:
    """Samples contain raw response x, observed V magnitude, and known background.

    Directions/cases are first averaged within a night. Each complete night has
    total weight one even when it contains several directions or stations.
    No sample-error-derived confidence interval or physical accuracy is inferred.
    """
    if not samples:
        raise ValueError('Nonempty photometric samples are required.')
    groups = defaultdict(list)
    roles = {}
    for row in samples:
        role, night = row['role'], row['night_id']
        if role not in ['training', 'validation'] or not night or not row.get('site_id'):
            raise ValueError('Every sample needs role, night_id and site_id.')
        if night in roles and roles[night] != role:
            raise ValueError('Training and validation nights must be disjoint, including across sites.')
        roles[night] = role
        if row.get('cloud_free_confirmed') is not True or not row.get('qa_source'):
            raise ValueError('Every sample needs explicitly confirmed clear-sky QA and its source.')
        values = [row['raw_response'], row['observed_v_mag_arcsec2'], row['background_relative']]
        if not all(np.isfinite(v) for v in values) or values[0] < 0 or values[2] < 0:
            raise ValueError('Photometric data must be finite and responses/backgrounds nonnegative.')
        groups[(role, night, row['site_id'], row.get('case_id'), tuple(row['direction']))].append(row)
    means = []
    for (role, night, site, case, direction), rows in groups.items():
        means.append({'role': role, 'night_id': night, 'site_id': site, 'case_id': case,
            'direction': list(direction), 'samples': len(rows),
            'raw_response': float(np.mean([r['raw_response'] for r in rows])),
            'observed_relative': float(np.mean([10**(-.4*(r['observed_v_mag_arcsec2']-reference_mag)) for r in rows])),
            'background_relative': float(np.mean([r['background_relative'] for r in rows]))})
    train = [r for r in means if r['role'] == 'training']
    check = [r for r in means if r['role'] == 'validation']
    if not train or not check:
        raise ValueError('Nonempty training and independent validation nights are required.')
    counts = {night: sum(r['night_id'] == night for r in train) for night in roles}
    x = np.array([r['raw_response'] for r in train])
    maximum = float(x.max())
    if maximum <= 0:
        raise ValueError('Training responses do not identify a source coefficient.')
    x = x / maximum
    y = np.array([r['observed_relative'] - r['background_relative'] for r in train])
    w = np.array([1 / counts[r['night_id']] for r in train])
    coefficient = float(np.sum(w*x*y) / np.sum(w*x*x) / maximum)
    if not np.isfinite(coefficient) or coefficient <= 0:
        raise ValueError('No positive fitted city component; revisit the background and data.')

    def evaluate(rows):
        details = []
        for row in rows:
            q = coefficient*row['raw_response'] + row['background_relative']
            if q <= 0:
                raise ValueError('Predicted total background must be positive.')
            observed = reference_mag - 2.5*np.log10(row['observed_relative'])
            predicted = reference_mag - 2.5*np.log10(q)
            details.append({**row, 'observed_v_mag_arcsec2': float(observed),
                'predicted_v_mag_arcsec2': float(predicted), 'predicted_relative': float(q),
                'residual_mag_model_minus_observed': float(predicted - observed)})
        nights = sorted({r['night_id'] for r in details})
        weights = np.array([1 / sum(s['night_id'] == r['night_id'] for s in details) / len(nights)
                            for r in details])
        residual = np.array([r['residual_mag_model_minus_observed'] for r in details])
        return {'nights': len(nights), 'samples': sum(r['samples'] for r in details),
            'night_direction_groups': len(details), 'sites': sorted({r['site_id'] for r in details}),
            'bias_mag': float(np.sum(weights*residual)), 'mae_mag': float(np.sum(weights*abs(residual))),
            'rmse_mag': float(np.sqrt(np.sum(weights*residual**2))), 'records': details}
    training, validation = evaluate(train), evaluate(check)
    return {'source_coefficient': coefficient, 'training': training, 'validation': validation,
            'independent_night_split': True,
            'independent_validation_sites': sorted(set(validation['sites']) - set(training['sites'])),
            'field_accuracy_certified': False, 'uncertainty_interval': None,
            'weighting': 'Each night has total weight one; linear-light average within each night/site/case/direction.'}
