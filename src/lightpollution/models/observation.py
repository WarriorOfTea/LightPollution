"""Versioned numerical observation scene: dated or explicitly undated.

One effective V-band convention, one common reference, separate components.
No eye threshold, display brightness, camera response or implicit current time.
"""

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone

import numpy as np

from ..atmosphere_config import atmosphere_from_mapping, atmosphere_origin
from ..moonlight import (REFERENCE_DISTANCE_KM, nanolambert_to_v_mag,
                        scattered_moonlight, separation_angle)
from ..observation import observation_context
from ..photometry import KS_V_BAND, RelativeBrightness, RelativeScale, SurfaceBrightness
from ..physics import Atmosphere, Observer
from ..domain.fields import ObservationResources
from ..skyview import horizontal_stars, sample_sky, stellar_extinction
from .background import BackgroundModel


def _object(value, allowed, required=()):
    if not isinstance(value, dict) or set(value)-set(allowed) or set(required)-set(value):
        raise ValueError(f"Invalid object; allowed keys: {sorted(allowed)}; required: {sorted(required)}.")
    return value


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")
    return float(value)


def _direction(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("Each direction is [azimuth_deg, altitude_deg].")
    az, alt = [_finite(v, "direction") for v in value]
    if not 0 <= alt <= 90:
        raise ValueError("Target altitude must be in [0,90].")
    return [az % 360, alt]


def _source(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("A nonempty source/assumption reference is required.")
    return value


def _utc(value):
    if not isinstance(value, str) or "T" not in value:
        raise ValueError("Dated mode requires an ISO date AND time with a timezone.")
    instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if instant.tzinfo is None or instant.utcoffset() is None:
        raise ValueError("Dated mode requires an explicit timezone.")
    return instant.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def vertical_extinction_v(observer: Observer, atmosphere: Atmosphere):
    """kV=(2.5/ln10)*tau550 above the observer, under the effective-grey bridge."""
    h, top = observer.height_m, atmosphere.top_m
    if h >= top:
        raise ValueError("Observer must be below the model atmosphere top.")
    tau = (atmosphere.rayleigh_sea_level * atmosphere.rayleigh_scale_height_m *
           (np.exp(-h/atmosphere.rayleigh_scale_height_m)-np.exp(-top/atmosphere.rayleigh_scale_height_m)))
    tau += atmosphere.aerosol_optical_depth * (
        np.exp(-h/atmosphere.aerosol_scale_height_m)-np.exp(-top/atmosphere.aerosol_scale_height_m))
    return float(2.5/np.log(10)*tau)


def _transfer_grid(resource, observer: Observer, atmosphere: Atmosphere):
    az, alt, raw, metadata = resource.azimuths, resource.altitudes, resource.radiance, resource.metadata
    if (not np.all(np.isfinite(az)) or not np.all(np.isfinite(alt))
            or metadata.get('observer')!=asdict(observer) or metadata.get('atmosphere')!=asdict(atmosphere)):
        raise ValueError('Sky grid observer/atmosphere must exactly match the scene.')
    sample_sky(az,alt,raw,0,90)
    if not np.allclose(raw[-1],raw[-1,0],rtol=1e-10,atol=0):
        raise ValueError('A physical sky grid must have a unique zenith.')
    return az,alt,raw,metadata


def calculate_scene(config: dict, resources: ObservationResources, *,
                    background: BackgroundModel | None = None, sample_directions=None) -> dict:
    """Validate the complete scene before returning strict-JSON-compatible results."""
    config = deepcopy(config)
    _object(config, {'schema_version','mode','time','observer','directions','atmosphere',
                     'cloud_free','moon','reference_v_mag_arcsec2','natural','artificial'},
            {'schema_version','mode','observer','directions','natural','artificial'})
    if type(config['schema_version']) is not int or config['schema_version'] != 1:
        raise ValueError("Only observation schema_version 1 is supported.")
    if config.get('cloud_free', True) is not True:
        raise ValueError("Only cloud_free=true is currently supported.")
    if config['mode'] not in ('dated','undated'):
        raise ValueError("mode must be dated or undated.")
    observer_values = _object(config['observer'], {'latitude_deg','longitude_deg','height_m'},
                              {'latitude_deg','longitude_deg','height_m'})
    observer = Observer(**{key: _finite(value, key) for key,value in observer_values.items()})
    atmosphere_values = config.get('atmosphere', {})
    atmosphere = atmosphere_from_mapping(atmosphere_values)
    if atmosphere.reference_wavelength_nm != 550.:
        raise ValueError("The effective V bridge requires the declared 550 nm transfer scenario.")
    directions = config['directions']
    if not isinstance(directions, list) or not directions:
        raise ValueError("At least one direction is required.")
    directions = np.asarray([_direction(value) for value in directions])
    scale = RelativeScale(reference_mag_arcsec2=_finite(config.get('reference_v_mag_arcsec2',22.), 'reference'))
    k_v = vertical_extinction_v(observer, atmosphere)
    context = None
    warnings = ['effective_550nm_shape_anchored_in_V_not_spectral_integration',
                'no_independent_field_validation', 'no_clouds_refraction_terrain_or_eye_camera_model']
    if config['mode'] == 'dated':
        if 'moon' in config:
            raise ValueError("Dated mode computes the Moon; manual lunar parameters are forbidden.")
        utc = _utc(config.get('time'))
        context = observation_context(observer, utc, *directions[0])
        if not context['astronomical_darkness'] and not (background and background.allows_twilight):
            raise ValueError("Sun is above -18 degrees; twilight is outside this scene model.")
        moon = context['moon']
        lunar = {'mode':'ephemeris', 'azimuth_deg':moon['azimuth_deg'],
                 'altitude_deg':moon['altitude_deg'],
                 'phase_angle_deg':moon['geocentric_phase_angle_deg'],
                 'distance_km':moon['topocentric_distance_km'],
                 'possible_lunar_eclipse':moon['possible_lunar_eclipse']}
    else:
        if 'time' in config:
            raise ValueError("Undated mode must not contain a time.")
        lunar = dict(_object(config.get('moon'), {'mode','azimuth_deg','altitude_deg',
                                                 'phase_angle_deg','distance_km'}, {'mode'}))
        if lunar['mode'] == 'none':
            if set(lunar) != {'mode'}:
                raise ValueError("Moon mode none must not include lunar parameters.")
        elif lunar['mode'] == 'manual':
            for key in ('azimuth_deg','altitude_deg','phase_angle_deg'):
                lunar[key] = _finite(lunar.get(key), key)
            lunar['distance_km'] = _finite(lunar.get('distance_km',REFERENCE_DISTANCE_KM),'distance')
            if (not -90 <= lunar['altitude_deg'] <= 90 or not 0 <= lunar['phase_angle_deg'] <= 180
                    or lunar['distance_km'] <= 0):
                raise ValueError("Invalid manual lunar geometry.")
            lunar['azimuth_deg'] %= 360
            warnings.append('manual_geometry_not_a_prediction_of_a_specific_night; no_eclipse_assumed')
        else:
            raise ValueError("Undated mode requires moon mode none or manual.")
        utc = None
    if lunar['mode'] != 'none':
        lunar['illuminated_fraction'] = float((1+np.cos(np.deg2rad(lunar['phase_angle_deg'])))/2)
        if lunar['altitude_deg'] <= 0:
            warnings.append('moon_below_horizon_set_to_zero; upper_atmosphere_residual_omitted')
        if lunar['phase_angle_deg'] < 7:
            warnings.append('lunar_opposition_surge_omitted')
        if lunar['altitude_deg'] < 10:
            warnings.append('low_moon; empirical_airmass_and_no_refraction')

    natural = _object(config['natural'], {'mode','v_mag_arcsec2','source'}, {'mode','source'})
    _source(natural['source'])
    if natural['mode'] == 'uniform':
        natural_q = float(scale.from_magnitudes(SurfaceBrightness(
            _finite(natural.get('v_mag_arcsec2'),'natural brightness'), KS_V_BAND)).values)
        warnings.append('uniform_natural_background_is_a_scenario')
    elif natural['mode'] == 'none' and set(natural) == {'mode','source'}:
        natural_q = 0.
        warnings.append('natural_background_explicitly_disabled')
    else:
        raise ValueError("Natural mode must be uniform with magnitude, or none.")
    if background is not None:
        warnings = [w for w in warnings if w != 'uniform_natural_background_is_a_scenario']
        warnings += ['natural_diffuse_components_are_scenarios_not_local_measurements',
                     'no_galactic_or_zodiacal_angular_maps',
                     'airglow_scattering_into_line_of_sight_omitted']
        if background.allows_twilight:
            warnings += ['twilight_paranal_empirical_not_local_validation',
                         'twilight_has_explicit_solar_and_angular_domain_gaps',
                         'twilight_not_adjusted_for_current_atmosphere_or_snow']

    artificial = _object(config['artificial'], {'mode','sky_grid','reference_sky_grid','anchor_direction',
                         'anchor_v_mag_arcsec2','scale_origin','source','band_bridge'}, {'mode'})
    grid = None
    grid_provenance = None
    if artificial['mode'] == 'grid':
        if (artificial.get('band_bridge') != 'effective_550nm_shape_anchored_in_V'
                or artificial.get('scale_origin') not in ('scenario','provided_calibration')):
            raise ValueError("Grid needs an explicit effective-grey V bridge and scale origin.")
        _source(artificial.get('source'))
        path = _source(artificial.get('sky_grid'))
        resource = resources.grids[path]
        az,alt,raw,metadata=_transfer_grid(resource,observer,atmosphere)
        anchor = _direction(artificial.get('anchor_direction',[0,90]))
        if resources.source_scale is None:
            reference_provenance=None
            reference_field=(az,alt,raw)
            if 'reference_sky_grid' in artificial:
                reference_path=_source(artificial['reference_sky_grid'])
                reference_resource=resources.grids[reference_path]
                raz,ralt,rraw,rmeta=_transfer_grid(reference_resource,observer,atmosphere)
                current_region=metadata.get('provenance',{}).get('region_metadata',{})
                reference_region=rmeta.get('provenance',{}).get('region_metadata',{})
                if current_region.get('common_mask_sha256') and (
                        current_region.get('common_mask_sha256')!=reference_region.get('common_mask_sha256')
                        or current_region.get('normalisation_reference_period')!=reference_region.get('normalisation_reference_period')
                        or current_region.get('source_composite')!=reference_region.get('source_composite')):
                    raise ValueError('Shared seasonal calibration needs the same common mask and source reference.')
                reference_field=(raz,ralt,rraw)
                reference_provenance={'path':artificial['reference_sky_grid'],
                                      'sha256':reference_resource.sha256}
            else:
                region=metadata.get('provenance',{}).get('region_metadata',{})
                period=f"{region.get('year')}-{region.get('month'):02}" if region.get('month') is not None else None
                if (region.get('source_power_multiplier',1.)!=1. or region.get('normalisation_reference_period')
                        and period!=region['normalisation_reference_period']):
                    raise ValueError('Seasonal amplitude needs reference_sky_grid; per-month anchoring would erase it.')
            anchor_raw = float(sample_sky(*reference_field,*anchor))
            if anchor_raw <= 0:
                raise ValueError("Artificial calibration anchor must have positive raw brightness.")
            anchor_q = float(scale.from_magnitudes(SurfaceBrightness(
                _finite(artificial.get('anchor_v_mag_arcsec2'),'artificial anchor'), KS_V_BAND)).values)
            grid = (az,alt,raw,anchor_q/anchor_raw)
        else:
            fixed = resources.source_scale
            if metadata.get('source_basis_id') != fixed.basis_id:
                raise ValueError('Fixed source scale does not match the source basis.')
            anchor_raw, anchor_q = fixed.anchor_raw, fixed.anchor_relative
            grid = (az, alt, raw, fixed.coefficient)
            reference_provenance = fixed.provenance
            warnings.append('fixed_source_coefficient_from_reference_atmosphere; '
                            + ('provided_ground_fit_with_declared_scope' if artificial['scale_origin'] == 'provided_calibration'
                               else 'scenario_not_new_calibration'))
        if utc:
            warnings.append('source_composite_is_not_an_hourly_lighting_forecast')
        grid_provenance = {'path':artificial['sky_grid'], 'sha256':resource.sha256,
                           'input_metadata':metadata, 'anchor_raw':anchor_raw,
                           'reference_sky_grid':reference_provenance,
                           'anchor_relative':anchor_q, 'scale_origin':artificial['scale_origin'],
                           'source':artificial['source'], 'calibration_independently_verified':False}
    elif artificial['mode'] != 'none' or set(artificial) != {'mode'}:
        raise ValueError("Artificial mode must be grid with calibration, or explicit none.")

    def fields(azimuth, altitude):
        azimuth, altitude = np.broadcast_arrays(azimuth,altitude)
        city = np.zeros(azimuth.shape) if grid is None else sample_sky(*grid[:3],azimuth,altitude)*grid[3]
        natural_values = np.full(azimuth.shape,natural_q)
        moon_q = np.zeros(azimuth.shape)
        separation = np.full(azimuth.shape,np.nan)
        valid = np.ones(azimuth.shape,dtype=bool)
        eclipse = False
        background_rows = background.evaluate(azimuth, altitude, context) if background else None
        twilight_values = np.zeros(azimuth.shape)
        if background_rows is not None:
            natural_values = np.asarray([r['natural_relative'] for r in background_rows]).reshape(azimuth.shape)
            twilight_values = np.asarray([r['twilight_relative'] or 0. for r in background_rows]).reshape(azimuth.shape)
        if lunar['mode'] != 'none':
            separation = separation_angle(azimuth,altitude,lunar['azimuth_deg'],lunar['altitude_deg'])
            if lunar['altitude_deg'] > 0:
                eclipse = lunar.get('possible_lunar_eclipse',False)
                valid = (separation >= 10) & (not eclipse)
                if np.any(valid):
                    nL = scattered_moonlight(lunar['phase_angle_deg'],lunar['altitude_deg'],
                                            altitude[valid],separation[valid],k_v,lunar['distance_km'])
                    nonzero = nL > 0
                    values = np.zeros(nL.shape)
                    values[nonzero] = scale.from_magnitudes(SurfaceBrightness(
                        nanolambert_to_v_mag(nL[nonzero]),KS_V_BAND)).values
                    moon_q[valid] = values
        total = (RelativeBrightness(city,scale) + RelativeBrightness(natural_values,scale)
                 + RelativeBrightness(moon_q,scale) + RelativeBrightness(twilight_values,scale))
        result = []
        for i in range(azimuth.size):
            moon_ok = bool(valid.flat[i])
            background_ok = background_rows is None or background_rows[i]['background_supported']
            ok = moon_ok and background_ok
            q = float(total.values.flat[i])
            result.append({'azimuth_deg':float(azimuth.flat[i]), 'altitude_deg':float(altitude.flat[i]),
                           'artificial_relative':float(city.flat[i]), 'natural_relative':float(natural_values.flat[i]),
                           'moon_relative':float(moon_q.flat[i]) if moon_ok else None,
                           'total_relative':q if ok else None,
                           'total_v_mag_arcsec2':float(scale.reference_mag_arcsec2-2.5*np.log10(q)) if ok and q>0 else None,
                           'moon_separation_deg':float(separation.flat[i]) if np.isfinite(separation.flat[i]) else None,
                           'supported':ok,
                           'status':('possible_lunar_eclipse' if eclipse else 'moon_separation_below_10deg') if not ok else 'ok',
                           'low_altitude_caution':bool(altitude.flat[i]<10)})
            if background_rows is not None:
                result[-1].update(background_rows[i])
                moon_status = ('possible_lunar_eclipse' if eclipse else 'moon_separation_below_10deg') if not moon_ok else 'ok'
                reasons = ([] if moon_ok else [moon_status]) + ([] if background_ok else [background_rows[i]['background_status']])
                result[-1].update(moon_supported=moon_ok, moon_status=moon_status, support_reasons=reasons)
                if len(reasons) > 1:
                    result[-1]['status'] = 'multiple_model_restrictions'
                elif not background_ok:
                    result[-1]['status'] = background_rows[i]['background_status']
                elif ok:
                    result[-1]['status'] = background_rows[i]['background_status']
        return result

    rows = fields(directions[:,0],directions[:,1])
    star_rows = []
    star_metadata = {'available':False, 'reason':'No date: no unique horizontal stellar positions.'}
    if utc is not None:
        if resources.catalogue is None:
            raise ValueError("Dated scenes require an explicitly loaded stellar catalogue.")
        stars, provenance = resources.catalogue
        star_az, star_alt, astrometry = horizontal_stars(stars,observer,utc,allow_moon=True,
            allow_twilight=bool(background and background.allows_twilight))
        selected = np.flatnonzero(star_alt >= 0)
        # A cached grid has a finite altitude domain; never extrapolate its shape.
        covered = np.ones(selected.shape,dtype=bool) if grid is None else star_alt[selected] >= grid[1][0]
        selected = selected[covered]
        extinction = stellar_extinction(star_alt[selected],observer,atmosphere)
        for index, dm in zip(selected,extinction,strict=True):
            observed = float(stars['vmag'][index]+dm)
            star_rows.append({'hr':int(stars['hr'][index]), 'name':str(stars['name'][index]),
                              'azimuth_deg':float(star_az[index]), 'altitude_deg':float(star_alt[index]),
                              'catalogue_v_mag':float(stars['vmag'][index]), 'extinction_v_mag':float(dm),
                              'observed_v_mag':observed, 'flux_relative_to_v_mag0':float(10**(-.4*observed))})
        star_metadata = {'available':True, 'catalogue':provenance, 'astrometry':astrometry,
                         'above_geometric_horizon':int(np.count_nonzero(star_alt>=0)),
                         'outside_grid_altitude_coverage':int(np.count_nonzero(~covered)),
                         'exported':len(star_rows), 'detected_by_eye_not_modelled':True,
                         'flux_reference':'V magnitude 0 point-source flux; NOT the diffuse Bref'}
    output = {'schema_version':1, 'mode':config['mode'], 'utc':utc, 'input_scene':config,
            'observer':asdict(observer), 'atmosphere':asdict(atmosphere),
            'atmosphere_origin':atmosphere_origin(atmosphere_values), 'cloud_free':True,
            'geometry':context, 'moon':lunar,
            'photometry':{'band':asdict(KS_V_BAND), 'reference_v_mag_arcsec2':scale.reference_mag_arcsec2,
                          'diffuse_units':'B/Bref, common reference; linear addition',
                          'extinction_v_mag_airmass':k_v,
                          'band_bridge':'effective 550 nm grey transfer shape anchored in V; not a spectral inversion'},
            'artificial_provenance':grid_provenance, 'directions':rows,
            'stars_metadata':star_metadata, 'stars':star_rows,
            'assumptions_and_limitations':warnings,
            'model':{'moonlight':'Krisciunas-Schaefer 1991 eqs 1,3,8,9,15,21; rho>=10 deg',
                     'distance_correction_reference_km':REFERENCE_DISTANCE_KM,
                     'lunar_opposition_surge':False, 'lunar_eclipse_model':False,
                     'physical_error_bound':None, 'document':'MODELS.md'}}
    if sample_directions is not None:
        samples=np.asarray([_direction(v) for v in sample_directions])
        output['sampled_directions']=fields(samples[:,0],samples[:,1])
    return output
