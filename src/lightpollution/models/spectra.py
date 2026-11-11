"""Finite-band spectral transport and photometry, with explicit response types.

Wavelengths are nm; energy spectra are per nm. Photopic/scotopic responses do
not implement visual adaptation, detection or display colour. Templates are
scenarios, not an inversion of VIIRS DNB.
"""

from dataclasses import replace

import numpy as np

from ..physics import (EARTH_RADIUS_M, Atmosphere, emission, hg_phase, rayleigh_phase,
                       source_to_point)

ARCSEC2_PER_SR = (180 * 3600 / np.pi)**2
H = 6.62607015e-34
C = 299792458.
K = 1.380649e-23


def checked_axis(wavelengths, values):
    axis, data = np.asarray(wavelengths, dtype=float), np.asarray(values, dtype=float)
    if (axis.ndim != 1 or len(axis) < 2 or np.any(~np.isfinite(axis))
            or axis[0] <= 0 or np.any(np.diff(axis) <= 0) or data.shape[-1:] != axis.shape
            or np.any(~np.isfinite(data)) or np.any(data < 0)):
        raise ValueError('Spectra require an increasing positive wavelength axis and finite nonnegative values.')
    return axis, data


def band_integral(wavelengths, spectra, response, detector='energy'):
    axis, data = checked_axis(wavelengths, spectra)
    _, curve = checked_axis(axis, response)
    if detector == 'energy':
        weight = curve
    elif detector == 'photon':
        weight = curve * axis * 1e-9 / (H*C)
    else:
        raise ValueError('Response type must be energy or photon.')
    return np.trapezoid(data * weight, axis, axis=-1)


def mix_templates(wavelengths, templates, fractions):
    if (not fractions or any(k not in templates for k in fractions)
            or any(not np.isfinite(v) or v < 0 for v in fractions.values())
            or not np.isclose(sum(fractions.values()), 1., rtol=0, atol=1e-9)):
        raise ValueError('Known templates and nonnegative radiant-power fractions summing to one are required.')
    output = np.zeros(len(wavelengths))
    for name, fraction in fractions.items():
        axis, value = checked_axis(wavelengths, templates[name])
        area = np.trapezoid(value, axis)
        if area <= 0:
            raise ValueError('A template has zero integrated power.')
        output += fraction * value / area
    return output


def wavelength_atmosphere(atmosphere, wavelength_nm, angstrom_exponent):
    return replace(atmosphere, reference_wavelength_nm=float(wavelength_nm),
        aerosol_optical_depth=atmosphere.aerosol_optical_depth * (wavelength_nm / 550.)**(-angstrom_exponent))


def integrate_spectral_direction(observer, sources, azimuth, altitude, atmosphere,
                                 settings, wavelengths, angstrom_exponent):
    """Same scalar single scattering, sharing geometry across all wavelengths.

    Gas absorption and wavelength-dependent g/SSA are omitted explicitly.
    Rayleigh scales as lambda^-4, aerosol AOD by a prescribed Angstrom law.
    """
    axis, _ = checked_axis(wavelengths, np.zeros(len(wavelengths)))
    if not np.isfinite(angstrom_exponent) or not 0 <= angstrom_exponent <= 4:
        raise ValueError('Angstrom exponent must be finite and in [0,4].')
    if not len(sources.power) or not np.any(sources.power):
        return np.zeros(len(axis))
    if observer.height_m >= atmosphere.top_m or atmosphere.reference_wavelength_nm != 550:
        raise ValueError('The spectral reference atmosphere must be at 550 nm and above the observer.')
    origin, direction = observer.position, observer.direction(azimuth, altitude)
    projection = float(np.dot(origin, direction))
    end = -projection + np.sqrt(projection**2 + (EARTH_RADIUS_M+atmosphere.top_m)**2 - np.dot(origin, origin))
    edges = settings.near_ray_scale_m*np.expm1(np.linspace(
        0, np.log1p(end/settings.near_ray_scale_m), settings.ray_segments+1))
    s, ds = (edges[1:]+edges[:-1])/2, np.diff(edges)
    points = origin[None, :] + s[:, None]*direction
    distance_squared, incoming, mu = source_to_point(sources, points)
    r2 = np.sum(points*points, axis=-1)
    molecular_atm = replace(atmosphere, aerosol_optical_depth=0.)
    aerosol_atm = replace(atmosphere, rayleigh_multiplier=0.)
    taus = []
    for part in [molecular_atm, aerosol_atm]:
        tau_source = part.optical_depth_radial(EARTH_RADIUS_M**2, r2[:, None],
                                               distance_squared, settings.path_quadrature)
        tau_observer = part.optical_depth_radial(np.dot(origin, origin), r2,
                                                 s*s, settings.path_quadrature)
        taus.append(tau_source + tau_observer[:, None])
    cosine = -np.sum(incoming*direction[None, None, :], axis=-1)
    br, ba, _ = atmosphere.coefficients(np.sqrt(r2)-EARTH_RADIUS_M)
    geometric = sources.power[None, :]*emission(mu, settings.horizontal_fraction)/distance_squared*ds[:, None]
    rayleigh = geometric * br[:, None] * rayleigh_phase(cosine)
    aerosol = geometric * ba[:, None] * hg_phase(cosine, atmosphere.aerosol_g)
    result = []
    for wavelength in axis:
        molecular_factor = (550/wavelength)**4
        aerosol_factor = (550/wavelength)**angstrom_exponent
        attenuation = np.exp(-taus[0]*molecular_factor - taus[1]*aerosol_factor)
        result.append(float(np.sum((rayleigh*molecular_factor + aerosol*aerosol_factor)*attenuation)))
    return np.asarray(result)


def blackbody_shape(wavelengths, temperature):
    if not np.isfinite(temperature) or not 1500 <= temperature <= 50000:
        raise ValueError('Continuum temperature must be in [1500,50000] K.')
    axis, _ = checked_axis(wavelengths, np.zeros(len(wavelengths)))
    metres = axis*1e-9
    shape = 1 / (metres**5 * np.expm1(H*C/(metres*K*temperature)))
    return shape / np.max(shape)


def continuum_from_bv(wavelengths, colour, b_response, v_response, zero_b, zero_v):
    """A two-band continuum approximation, not a measured stellar temperature."""
    if not np.isfinite(colour):
        return None
    def bv(temperature):
        shape = blackbody_shape(wavelengths, temperature)
        return -2.5*np.log10((band_integral(wavelengths, shape, b_response)/zero_b)
                             / (band_integral(wavelengths, shape, v_response)/zero_v))
    low, high = 1500., 50000.
    if not bv(high) <= colour <= bv(low):
        return None
    for _ in range(38):
        middle = (low+high)/2
        if bv(middle) > colour:
            low = middle
        else:
            high = middle
    return blackbody_shape(wavelengths, (low+high)/2), (low+high)/2


def response_summary(wavelengths, spectra, responses, zero_v):
    v = band_integral(wavelengths, spectra, responses['johnson_v'])
    p = 683.*band_integral(wavelengths, spectra, responses['photopic'])
    s = 1700.*band_integral(wavelengths, spectra, responses['scotopic'])
    b = band_integral(wavelengths, spectra, responses['johnson_b'])
    return {'johnson_v_energy': np.asarray(v).tolist(),
            'johnson_b_energy': np.asarray(b).tolist(),
            'photopic': np.asarray(p).tolist(), 'scotopic': np.asarray(s).tolist(),
            'scotopic_photopic_ratio': np.divide(s, p, out=np.zeros_like(np.asarray(s)), where=p>0).tolist(),
            'v_zero_point_energy_W_m2': float(zero_v)}


def stellar_spectral_observation(rows, observer, atmosphere, wavelengths, responses,
                                colours, measured, zero_b, zero_v, angstrom_exponent,
                                continuum=True):
    """Spectral flux and band extinction once, after catalogue V normalisation.

    Measured spectra supply a shape; catalogue V supplies its TOA scale. The
    unscaled measured SED is kept in the reference library for independent checks.
    Missing colours are never silently assigned a white or solar spectrum.
    """
    from ..skyview import stellar_extinction
    if not rows:
        return [], {'available': 0, 'total_catalogue_rows': 0, 'measured': 0,
                    'catalogue_continuum': 0, 'unavailable': 0}, []
    temperatures = np.geomspace(1500., 50000., 2048)
    metres = np.asarray(wavelengths)*1e-9
    templates = 1/(metres[None, :]**5*np.expm1(H*C/(metres[None, :]*K*temperatures[:, None])))
    template_bv = -2.5*np.log10((band_integral(wavelengths, templates, responses['johnson_b'])/zero_b)
                               / (band_integral(wavelengths, templates, responses['johnson_v'])/zero_v))
    shapes, indexes, recipes = [], [], []
    counts = {'measured': 0, 'catalogue_continuum': 0, 'unavailable': 0}
    output = [dict(r) for r in rows]
    for i, row in enumerate(rows):
        hr = row['hr']
        if hr in measured:
            shape = measured[hr]
            recipe = {'kind': 'measured_reference_shape', 'hr': hr}
            counts['measured'] += 1
        else:
            data = colours.get(hr, {})
            colour = float(data['b_v']) if data.get('b_v') else np.nan
            if not continuum or not np.isfinite(colour) or not template_bv[-1] <= colour <= template_bv[0]:
                counts['unavailable'] += 1
                output[i]['spectral_available'] = False
                output[i]['spectral_reason'] = 'no_measured_shape_or_usable_B_V_continuum'
                continue
            temperature = float(np.interp(colour, template_bv[::-1], temperatures[::-1]))
            shape = blackbody_shape(wavelengths, temperature)
            recipe = {'kind': 'B_V_fitted_Planck_continuum', 'continuum_temperature_K': temperature,
                      'b_v': colour, 'temperature_is_not_measured_Teff': True,
                      'bv_uncertainty_flag': data.get('bv_uncertainty_flag'),
                      'variability_id': data.get('variability_id')}
            counts['catalogue_continuum'] += 1
        shapes.append(shape)
        indexes.append(i)
        recipes.append(recipe)
    if not indexes:
        return output, {**counts, 'available': 0, 'total_catalogue_rows': len(rows)}, []
    shapes = np.asarray(shapes)
    magnitudes = np.array([rows[i]['catalogue_v_mag'] for i in indexes])
    flux_toa = shapes * (zero_v*10**(-.4*magnitudes)
                        / band_integral(wavelengths, shapes, responses['johnson_v']))[:, None]
    altitudes = np.array([rows[i]['altitude_deg'] for i in indexes])
    molecular_tau = stellar_extinction(altitudes, observer, replace(atmosphere, aerosol_optical_depth=0))*np.log(10)/2.5
    aerosol_tau = stellar_extinction(altitudes, observer, replace(atmosphere, rayleigh_multiplier=0))*np.log(10)/2.5
    tau = molecular_tau[:, None]*(550/np.asarray(wavelengths))**4
    tau += aerosol_tau[:, None]*(550/np.asarray(wavelengths))**angstrom_exponent
    observed = flux_toa*np.exp(-tau)
    summary = response_summary(wavelengths, observed, responses, zero_v)
    for j, index in enumerate(indexes):
        v = summary['johnson_v_energy'][j]
        magnitude = float(-2.5*np.log10(v/zero_v))
        output[index].update({'spectral_available': True, 'spectrum_recipe': recipes[j],
            'observed_v_mag': magnitude, 'extinction_v_mag': magnitude-rows[index]['catalogue_v_mag'],
            'flux_relative_to_v_mag0': v/zero_v,
            'johnson_v_flux_W_m2': v, 'johnson_b_flux_W_m2': summary['johnson_b_energy'][j],
            'photopic_illuminance_lux': summary['photopic'][j],
            'scotopic_illuminance_lux': summary['scotopic'][j],
            'scotopic_photopic_ratio': summary['scotopic_photopic_ratio'][j]})
    chosen = sorted(range(len(indexes)), key=lambda j: magnitudes[j])[:6]
    chosen += [j for j, recipe in enumerate(recipes) if recipe['kind'] == 'measured_reference_shape' and j not in chosen]
    examples = [{'hr': rows[indexes[j]]['hr'], 'flux_toa_W_m2_nm': flux_toa[j].tolist(),
                 'flux_observed_W_m2_nm': observed[j].tolist(), 'recipe': recipes[j]} for j in chosen]
    return output, {**counts, 'available': len(indexes), 'total_catalogue_rows': len(rows),
        'flux_unit': 'W/(m2 nm) point-source irradiance; not per sr',
        'extinction_applied_once': True, 'zero_point_V_W_m2': zero_v,
        'model': 'Measured SED shapes where available; otherwise B-V Planck continuum scenario.'}, examples
