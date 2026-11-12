"""Diffuse V-only background scenarios; no inferred weather or display model.

Thin-shell airglow follows van Rhijn geometry and direct Beer transmission.
Twilight reproduces Patat et al. (2006), Table 1, within its measured domain.
See MODELS.md for the derivation and the explicit gaps.
"""

from dataclasses import dataclass

import numpy as np

from ..physics import EARTH_RADIUS_M, Atmosphere, Observer


def _number(value, name, lower, upper):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not np.isfinite(value) or not lower <= value <= upper):
        raise ValueError(f'{name} must be a finite number in [{lower}, {upper}].')
    return float(value)


def background_config(value):
    if not isinstance(value, dict) or set(value) - {
            'mode', 'airglow_fraction', 'airglow_height_m', 'twilight'}:
        raise ValueError('Unknown natural-background settings.')
    result = {'mode': 'airglow_layer', 'airglow_fraction': .6,
              'airglow_height_m': 90_000., 'twilight': 'none', **value}
    if result['mode'] not in ('uniform', 'airglow_layer'):
        raise ValueError('Natural background must be uniform or airglow_layer.')
    if result['twilight'] not in ('none', 'patat2006'):
        raise ValueError('Twilight must be none or patat2006.')
    result['airglow_fraction'] = _number(result['airglow_fraction'], 'airglow_fraction', 0, 1)
    result['airglow_height_m'] = _number(result['airglow_height_m'], 'airglow_height_m', 80_000, 400_000)
    return result


def van_rhijn(altitude_deg, layer_height_m=90_000., observer_height_m=0.):
    """Geometrical path-length ratio of an optically thin spherical shell."""
    alt = np.asarray(altitude_deg, dtype=float)
    if (not np.all(np.isfinite(alt)) or np.any((alt < 0) | (alt > 90))
            or not np.isfinite(layer_height_m) or not np.isfinite(observer_height_m)
            or not 0 <= observer_height_m < layer_height_m):
        raise ValueError('Invalid thin-shell geometry.')
    ratio = (EARTH_RADIUS_M + observer_height_m)/(EARTH_RADIUS_M + layer_height_m)
    return 1/np.sqrt(1-(ratio*np.cos(np.deg2rad(alt)))**2)


def direct_optical_depth(altitude_deg, observer: Observer, atmosphere: Atmosphere, upper_m=None):
    """Observer-to-boundary optical depth on a spherical straight ray."""
    alt = np.asarray(altitude_deg, dtype=float)
    if not np.all(np.isfinite(alt)) or np.any((alt < 0) | (alt > 90)):
        raise ValueError('Invalid background sightline altitude.')
    upper = atmosphere.top_m if upper_m is None else min(upper_m, atmosphere.top_m)
    if upper <= observer.height_m:
        raise ValueError('Emitting layer must be above observer.')
    r0 = EARTH_RADIUS_M + observer.height_m
    r1 = EARTH_RADIUS_M + upper
    projection = r0*np.sin(np.deg2rad(alt))
    length = (r1*r1-r0*r0)/(np.sqrt(projection*projection+r1*r1-r0*r0)+projection)
    # 96 nodes resolve the low-altitude aerosol layer even on long oblique rays.
    return atmosphere.optical_depth_radial(r0*r0, r1*r1, length*length, order=96)


def patat_v_magnitude(sun_depression_deg):
    """TOTAL natural + twilight V fit, not an isolated solar component."""
    depression = np.asarray(sun_depression_deg, dtype=float)
    if not np.all(np.isfinite(depression)) or np.any((depression < 5) | (depression > 15)):
        raise ValueError('Patat V polynomial is supported only for Sun depression 5–15 deg.')
    x = depression - 5
    return 11.84 + 1.518*x - .057*x*x


@dataclass(frozen=True)
class BackgroundModel:
    config: dict
    observer: Observer
    atmosphere: Atmosphere
    zenith_v_mag_arcsec2: float
    reference_v_mag_arcsec2: float = 22.

    @property
    def allows_twilight(self):
        return self.config['twilight'] == 'patat2006'

    def evaluate(self, azimuth, altitude, context):
        azimuth, altitude = np.broadcast_arrays(azimuth, altitude)
        q0 = 10**(-.4*(self.zenith_v_mag_arcsec2-self.reference_v_mag_arcsec2))
        if self.config['mode'] == 'uniform':
            ag = np.zeros(altitude.shape)
            residual = np.full(altitude.shape, q0)
        else:
            fraction = self.config['airglow_fraction']
            height = self.config['airglow_height_m']
            tau_ag = direct_optical_depth(altitude, self.observer, self.atmosphere, height)
            tau_ag0 = direct_optical_depth(90., self.observer, self.atmosphere, height)
            tau_ext = direct_optical_depth(altitude, self.observer, self.atmosphere)
            tau_ext0 = direct_optical_depth(90., self.observer, self.atmosphere)
            ag = q0*fraction*van_rhijn(altitude, height, self.observer.height_m)*np.exp(tau_ag0-tau_ag)
            residual = q0*(1-fraction)*np.exp(tau_ext0-tau_ext)
        rows = []
        sun_alt = context['sun']['altitude_deg'] if context else None
        for i in range(altitude.size):
            twilight, status, supported, fitted = 0., 'ok', True, None
            if sun_alt is not None and sun_alt > -18:
                if not self.allows_twilight:
                    status, supported, twilight = 'twilight_disabled', False, None
                elif not -15 <= sun_alt <= -5:
                    status, supported, twilight = 'twilight_solar_domain_gap', False, None
                elif altitude.flat[i] < 50:
                    status, supported, twilight = 'twilight_outside_zenith_region', False, None
                else:
                    fitted = float(patat_v_magnitude(-sun_alt))
                    # The fitted total already contains Paranal's night floor.
                    # Subtract THAT fixed floor; never subtract the local scenario.
                    fit_q = 10**(-.4*(fitted-self.reference_v_mag_arcsec2))
                    floor_q = 10**(-.4*(21.6-self.reference_v_mag_arcsec2))
                    twilight = max(0., fit_q-floor_q)
                    status = 'paranal_empirical_scenario_not_local_validation'
            night = float(ag.flat[i]+residual.flat[i])
            rows.append({'natural_relative': night, 'airglow_relative': float(ag.flat[i]),
                'diffuse_residual_relative': float(residual.flat[i]), 'twilight_relative': twilight,
                'background_supported': supported, 'background_status': status,
                'patat_total_v_mag_arcsec2': fitted})
        return rows

    def metadata(self):
        return {'config': self.config, 'zenith_night_v_mag_arcsec2': self.zenith_v_mag_arcsec2,
            'zenith_normalisation': 'observed diffuse night zenith scenario in CURRENT atmosphere',
            'catalogue_stars_included': False, 'unresolved_starlight_in_diffuse_residual': True,
            'galactic_and_zodiacal_maps_available': False, 'spectrum_available': False,
            'airglow': 'van Rhijn thin spherical shell + direct Beer attenuation; scattering into ray omitted',
            'twilight': {'model': self.config['twilight'], 'source': 'Patat et al. 2006 Table 1 V',
                'sun_altitude_domain_deg': [-15., -5.], 'target_altitude_domain_deg': [50., 90.],
                'night_floor_v_mag_arcsec2': 21.6, 'published_scatter_mag': .18,
                'scatter_is_global_accuracy_bound': False, 'local_field_validation_nights': 0,
                'night_solar_cutoff_deg': -18., 'deep_transition_15_to_18_deg_supported': False,
                'atmosphere_or_snow_adjusts_empirical_twilight': False},
            'physical_error_bound': None, 'document': 'MODELS.md'}
