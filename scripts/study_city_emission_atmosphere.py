"""Controlled city-emission / vertical-profile study, without production rollout.

Uses the pinned Madrid grid and support from the preceding angular study.
Profiles below are illustrative, not retrieved weather. No network is used.
"""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import resource
from zoneinfo import ZoneInfo

import numpy as np

from lightpollution.adapters.storage import file_hash, write_json
from lightpollution.data import load_region
from lightpollution.models.all_orders import AllOrderSettings
from lightpollution.models.angular_sources import AngularGridBoundary, bin_horizontal_factor, horizontal_radiance_factor
from lightpollution.models.boundary_orders import integrate_boundary_variants, paired_ratio
from lightpollution.physics import Atmosphere, EARTH_RADIUS_M, Observer, cell_areas

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = ROOT/'artifacts/angular-nasa-study-20261113'
CODE = ('scripts/study_city_emission_atmosphere.py', 'src/lightpollution/physics.py',
        'src/lightpollution/models/scattering_orders.py', 'src/lightpollution/models/all_orders.py',
        'src/lightpollution/models/angular_sources.py', 'src/lightpollution/models/boundary_orders.py')


@dataclass(frozen=True)
class LayerStudyAtmosphere(Atmosphere):
    """Research-only piecewise constant aerosol extinction with exact AOD.

    Each tuple is (bottom MSL metres, top MSL metres, fraction of total AOD).
    Collision sampling uses a valid global majorant even for elevated layers.
    The inherited optical-depth quadrature is NOT used in this experiment.
    """
    aerosol_layers: tuple[tuple[float, float, float], ...] = ((0., 1000., 1.),)

    def __post_init__(self):
        # The base validator inspects its whole __dict__, which would include
        # this subclass's structured layer tuples. Validate only base fields.
        Atmosphere(**{name: getattr(self, name) for name in Atmosphere.__dataclass_fields__})
        if not self.aerosol_layers:
            raise ValueError('At least one aerosol layer is required.')
        previous = 0.
        for bottom, top, fraction in self.aerosol_layers:
            if (not np.all(np.isfinite([bottom, top, fraction])) or bottom < previous
                    or not bottom < top <= self.top_m or fraction < 0):
                raise ValueError('Layers must be ordered, disjoint, finite and nonnegative.')
            previous = top
        if not np.isclose(sum(x[2] for x in self.aerosol_layers), 1., atol=1e-12, rtol=0):
            raise ValueError('Layer AOD fractions must sum to one.')

    @property
    def extinction_majorant(self):
        return self.rayleigh_sea_level + max(
            self.aerosol_optical_depth*fraction/(top-bottom)
            for bottom, top, fraction in self.aerosol_layers)

    def coefficients(self, height_m):
        height = np.maximum(height_m, 0.)
        molecular = self.rayleigh_sea_level*np.exp(-height/self.rayleigh_scale_height_m)
        aerosol = np.zeros_like(height, dtype=float)
        for bottom, top, fraction in self.aerosol_layers:
            aerosol += np.where((height >= bottom) & (height < top),
                self.aerosol_optical_depth*fraction/(top-bottom), 0.)
        return molecular, aerosol*self.aerosol_single_scattering_albedo, molecular+aerosol


def hybrid_radiance(near, off, mu, eta):
    """Old fitted intensity mixture only for 0--60deg; measured off proxy above.

    The cutoff is on RADIANCE. A finite jump at 60deg is retained explicitly;
    no smoothing, clipping of eta, or per-variant power normalization is used.
    """
    if not np.isfinite(eta) or not 0 <= eta < 1:
        raise ValueError('The fitted mixture requires finite eta in [0,1).')
    cosine = np.asarray(mu, dtype=float)
    if np.any(~np.isfinite(cosine)) or np.any((cosine < 0) | (cosine > 1)):
        raise ValueError('Cosines must lie in [0,1].')
    denominator = 1-eta+eta*bin_horizontal_factor(0, 20)
    mixture = np.asarray(near)*(1-eta+eta*horizontal_radiance_factor(np.maximum(.5, cosine)))/denominator
    return np.where(cosine <= .5, off, mixture)


class StudyBoundary:
    labels = ('near_lambertian', 'empirical_hold_off', 'fitted_0_60_hold_off')

    def __init__(self, grid):
        self.grid = grid
        near, off = float(grid.near.max(initial=0)), float(grid.off.max(initial=0))
        mixture_max = float(hybrid_radiance(near, off, np.nextafter(.5, 1.), grid.regional_eta))
        self.upper_bounds = np.array([near, max(near, off), max(mixture_max, off)])

    def sample(self, points, outgoing):
        values, codes = self.grid.sample(points, outgoing)
        mu = np.clip(np.einsum('ij,ij->i', outgoing,
            points/np.linalg.norm(points, axis=1)[:, None]), 0, 1)
        hybrid = hybrid_radiance(values[:, 0], values[:, 1], mu, self.grid.regional_eta)
        return np.stack((values[:, 0], values[:, 4], hybrid), axis=1), codes


def angular_integrals(eta, ratio):
    """Analytical integrals, in units of the NearNadir Lambertian power."""
    gn = bin_horizontal_factor(0, 20)
    flux = 2*((.375*(1-eta)+.015625*eta)/(1-eta+eta*gn)+.125*ratio)
    return {'off_over_near': ratio, 'eta': eta,
        'hybrid_power_over_near_lambertian': float(flux),
        'hybrid_xi_25': float(ratio*np.sin(np.deg2rad(25))**2/flux),
        'hybrid_xi_30': float(.25*ratio/flux),
        'mixture_at_60_minus_over_off': float(
            (1-eta+eta*horizontal_radiance_factor(.5))/(1-eta+eta*gn)/ratio),
        'xi_definition': 'Fraction of upward flux emitted below 25/30 deg elevation; not the mixture coefficient.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/city-emission-review-20261113')
    parser.add_argument('--samples', type=int, default=524288)
    parser.add_argument('--case')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'cases').mkdir(exist_ok=True)
    previous = json.loads((PREVIOUS/'protocol.json').read_text())
    arrays_by_period, common = {}, None
    for period in ('madrid-2017-03', 'madrid-2017-11'):
        record = previous['inputs'][period]
        path = ROOT/record['maps']
        if file_hash(path) != record['maps_sha256']:
            raise ValueError('The pinned NASA grid changed.')
        arrays, bounds, _ = load_region(path)
        arrays_by_period[period] = arrays
        support = np.isin(arrays['Land_Water_Mask'], [0, 1, 5])
        for category in ('NearNadir', 'OffNadir', 'AllAngle'):
            from lightpollution.data import good_mask
            name = category+'_Composite_Snow_Free'
            support &= good_mask(arrays[name], arrays[name+'_Quality'], arrays[name+'_Num'])
        common = support if common is None else common & support
    import hashlib
    if hashlib.sha256(common.tobytes()).hexdigest() != previous['inputs']['madrid-2017-03']['common_support_sha256']:
        raise ValueError('The pinned common support changed.')
    record = previous['inputs']['madrid-2017-03']
    arrays = arrays_by_period['madrid-2017-03']
    grid = AngularGridBoundary(bounds, *(arrays[c+'_Composite_Snow_Free']
        for c in ('NearNadir', 'OffNadir', 'AllAngle')), common, record['regional_eta_uniform_solid_angle'])
    boundary = StudyBoundary(grid)
    profiles = {'exponential_1200m': Atmosphere(),
        'mixed_0_1000m': LayerStudyAtmosphere(),
        'elevated_2000_3000m': LayerStudyAtmosphere(aerosol_layers=((2000., 3000., 1.),)),
        'half_low_half_elevated': LayerStudyAtmosphere(aerosol_layers=((0., 1000., .5), (2000., 3000., .5)))}
    seeds = dict(zip(profiles, (9071, 19073, 31013, 47017)))
    positions = {'ucm': Observer(40.451111, -3.726111, 20.), 'north': Observer(40.85, -3.726111, 20.)}
    cases = [{'id': f'{location}-alt{alt}-{profile}', 'location': location,
        'altitude_deg': alt, 'profile': profile, 'seed': seeds[profile]}
        for location in positions for alt in (10, 45, 90) for profile in profiles]
    angular = {name: angular_integrals(r['regional_eta_uniform_solid_angle'],
        r['off_near_area_ratio_on_transport_support']) for name, r in previous['inputs'].items()}
    protocol = {'id': 'LP-city-emission-vertical-profile-1',
        'created_at': datetime.now(ZoneInfo('Europe/Berlin')).isoformat(timespec='seconds'),
        'code_sha256': {name: file_hash(ROOT/name) for name in CODE},
        'source_reference': previous['inputs']['madrid-2017-03'],
        'same_support_cells': int(common.sum()), 'samples_per_case': args.samples,
        'profiles': {name: asdict(atm) for name, atm in profiles.items()},
        'positions': {name: asdict(obs) for name, obs in positions.items()}, 'cases': cases,
        'angular_integrals': angular, 'fixed_scale_no_recalibration': True,
        'baseline_atmosphere_also_stratified': 'Exponential molecular and aerosol profiles already vary with height.',
        'not_field_validation': True, 'production_solver_changed': False,
        'limitations': ['550nm scalar transfer only; source spectrum and actual vertical profiles not measured.',
            'No DSM, clouds, ground re-reflection, moonlight or natural sky.',
            'Full elevated-AOD layer is an intentionally strong sensitivity scenario, not a local forecast.',
            'Profiles use distinct seeds; angular variants share complete paths within each profile.',
            'Between-profile intervals are approximate delta-method MC intervals, not atmospheric or field uncertainty.']}
    write_json(args.output/'protocol.json', protocol)
    for case in cases:
        if args.case and args.case != case['id']:
            continue
        path = args.output/'cases'/(case['id']+'.json')
        identity = {'case': case, 'samples': args.samples, 'code': protocol['code_sha256'],
            'source': protocol['source_reference'], 'atmosphere': asdict(profiles[case['profile']])}
        if path.exists() and json.loads(path.read_text()).get('identity') == identity:
            print(json.dumps({'cached': case['id']}), flush=True)
            continue
        result = integrate_boundary_variants(positions[case['location']], boundary, 180., case['altitude_deg'],
            profiles[case['profile']], AllOrderSettings(samples=args.samples, seed=case['seed']))
        result['identity'], result['case'] = identity, case
        result['ratios_to_near'] = {label: paired_ratio(result, label, boundary.labels[0]) for label in boundary.labels}
        result['peak_rss_kib_process'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        write_json(path, result)
        print(json.dumps({'case': case['id'], 'radiance': result['all_order_radiance'],
            'standard_error': result['all_order_standard_error'], 'effective_samples': result['effective_score_samples'],
            'runtime_s': result['runtime_seconds']}), flush=True)


if __name__ == '__main__':
    main()
