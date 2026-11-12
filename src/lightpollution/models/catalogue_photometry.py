"""Explicit Tycho-to-Johnson approximation; never silently equate passbands.

Tycho-2 Guide, equations5/6 (ESA): for unreddened main-sequence stars.
The narrower colour domain below is our conservative operational restriction.
KStars' guessed spectral-class marker only infers the input system.
"""


def johnson_photometry(star):
    system=star.get('photometry','Johnson_V_BV')
    magnitude,colour=star['v_mag'],star['b_v']
    if system=='Johnson_V_BV':
        return magnitude,colour,'Johnson_V_BV_catalogue_or_legacy_assumption'
    if system!='Tycho_VT_BTminusVT_inferred':
        raise ValueError('Unsupported catalogue photometric system.')
    if colour is None:
        return None,None,'Tycho_colour_missing_VT_is_not_Johnson_V'
    if not -.2<=colour<=1.8:
        return None,None,'Tycho_colour_outside_declared_linear_transform_domain'
    return magnitude-.090*colour,.850*colour,'Tycho_to_Johnson_linear_unreddened_main_sequence_scenario'
