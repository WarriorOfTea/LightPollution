"""First ground reflection at fixed lamp power, relative to snow-free uplight.

An effective uniform cover is a scenario over ILLUMINATED source surfaces.
It is neither snow at the observer nor an inferred satellite snow fraction.
"""

from .background import _number


def surface_config(value):
    if not isinstance(value, dict) or set(value) - {'snow_fraction', 'bare_albedo',
            'snow_albedo', 'direct_upward_fraction', 'direct_horizontal_fraction'}:
        raise ValueError('Unknown surface-reflection settings.')
    result = {'snow_fraction': 0., 'bare_albedo': .15, 'snow_albedo': .8,
              'direct_upward_fraction': .15, 'direct_horizontal_fraction': 0., **value}
    result = {key: _number(v, key, 0, 1) for key, v in result.items()}
    u, rho0 = result['direct_upward_fraction'], result['bare_albedo']
    if u+(1-u)*rho0 <= 0:
        raise ValueError('Snow-free upward reference power must be positive.')
    return result


def surface_balance(config, *, reference=False):
    """All powers are fractions of one lamp's emitted radiant power."""
    u, rho0 = config['direct_upward_fraction'], config['bare_albedo']
    f = 0. if reference else config['snow_fraction']
    rho = (1-f)*rho0 + f*config['snow_albedo']
    reflected = (1-u)*rho
    upward_reference = u+(1-u)*rho0
    return {'effective_albedo': rho, 'direct_fraction_of_lamp_power': u,
        'reflected_fraction_of_lamp_power': reflected,
        'absorbed_fraction_of_lamp_power': (1-u)*(1-rho),
        'upward_fraction_of_lamp_power': u+reflected,
        'snow_free_upward_fraction_of_lamp_power': upward_reference,
        'direct_multiplier_of_reference_upward_power': u/upward_reference,
        'reflected_multiplier_of_reference_upward_power': reflected/upward_reference,
        'upward_power_over_snow_free': (u+reflected)/upward_reference}


def surface_reference(config):
    """Parameters identifying the fixed bare-ground lamp inversion."""
    return {key: config[key] for key in ('bare_albedo', 'direct_upward_fraction',
                                        'direct_horizontal_fraction')}


def surface_metadata(config):
    return {'model': 'fixed_lamp_first_lambertian_ground_reflection_1', 'config': config,
        'balance': surface_balance(config), 'reference_balance': surface_balance(config, reference=True),
        'fraction_domain': 'illumination-weighted fraction of source ground under lamps; uniform regional scenario',
        'source_reference': 'NearNadir_Composite_Snow_Free (or explicit synthetic snow-free proxy)',
        'direct_emission': 'normalised Lambert/low-angle mixture', 'reflected_emission': 'Lambert mu/pi',
        'spectral_albedo': 'grey constant over model band; no snow colour predicted',
        'lamp_power_held_fixed': True, 'additional_total_uplight_multiplier_applied': False,
        'atmosphere_ground_repeated_bounces': False, 'moon_sun_ground_feedback': False,
        'snow_forecast': False, 'snow_observations_loaded': False,
        'clear_sky_field_validation_nights': 0, 'physical_error_bound': None,
        'document': 'docs/NATURAL_TWILIGHT_SNOW.md'}
