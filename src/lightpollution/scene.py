"""Versioned numerical observation scene: dated or explicitly undated.

One effective V-band convention, one common reference, separate components.
No eye threshold, display brightness, camera response or implicit current time.
"""

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from .moonlight import (REFERENCE_DISTANCE_KM, nanolambert_to_v_mag,
                        scattered_moonlight, separation_angle)
from .observation import observation_context
from .photometry import KS_V_BAND, RelativeBrightness, RelativeScale, SurfaceBrightness
from .physics import Atmosphere, Observer
from .skyview import horizontal_stars, load_stars, sample_sky, stellar_extinction


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


def calculate_observation(config: dict, base_directory: Path = Path('.')) -> dict:
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
    atmosphere_values = _object(config.get('atmosphere', {}), Atmosphere.__dataclass_fields__)
    atmosphere = Atmosphere(**{key: _finite(value, key) for key,value in atmosphere_values.items()})
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
        if not context['astronomical_darkness']:
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

    artificial = _object(config['artificial'], {'mode','sky_grid','anchor_direction',
                         'anchor_v_mag_arcsec2','scale_origin','source','band_bridge'}, {'mode'})
    grid = None
    grid_provenance = None
    if artificial['mode'] == 'grid':
        if (artificial.get('band_bridge') != 'effective_550nm_shape_anchored_in_V'
                or artificial.get('scale_origin') not in ('scenario','provided_calibration')):
            raise ValueError("Grid needs an explicit effective-grey V bridge and scale origin.")
        _source(artificial.get('source'))
        path = Path(_source(artificial.get('sky_grid')))
        if not path.is_absolute():
            path = base_directory / path
        with np.load(path, allow_pickle=False) as data:
            az, alt, raw = (np.array(data[key], copy=True) for key in ('azimuth_deg','altitude_deg','radiance'))
            metadata = json.loads(str(data['metadata_json'].item()))
        if (not np.all(np.isfinite(az)) or not np.all(np.isfinite(alt))
                or metadata.get('observer') != asdict(observer)
                or metadata.get('atmosphere') != asdict(atmosphere)):
            raise ValueError("Sky grid observer/atmosphere must exactly match the scene.")
        anchor = _direction(artificial.get('anchor_direction',[0,90]))
        anchor_raw = float(sample_sky(az,alt,raw,*anchor))
        if not np.allclose(raw[-1],raw[-1,0],rtol=1e-10,atol=0):
            raise ValueError("A physical sky grid must have a unique zenith.")
        if anchor_raw <= 0:
            raise ValueError("Artificial calibration anchor must have positive raw brightness.")
        anchor_q = float(scale.from_magnitudes(SurfaceBrightness(
            _finite(artificial.get('anchor_v_mag_arcsec2'),'artificial anchor'), KS_V_BAND)).values)
        grid = (az,alt,raw,anchor_q/anchor_raw)
        grid_provenance = {'path':artificial['sky_grid'], 'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                           'input_metadata':metadata, 'anchor_raw':anchor_raw,
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
                 + RelativeBrightness(moon_q,scale))
        result = []
        for i in range(azimuth.size):
            ok = bool(valid.flat[i])
            q = float(total.values.flat[i])
            result.append({'azimuth_deg':float(azimuth.flat[i]), 'altitude_deg':float(altitude.flat[i]),
                           'artificial_relative':float(city.flat[i]), 'natural_relative':float(natural_values.flat[i]),
                           'moon_relative':float(moon_q.flat[i]) if ok else None,
                           'total_relative':q if ok else None,
                           'total_v_mag_arcsec2':float(scale.reference_mag_arcsec2-2.5*np.log10(q)) if ok and q>0 else None,
                           'moon_separation_deg':float(separation.flat[i]) if np.isfinite(separation.flat[i]) else None,
                           'supported':ok,
                           'status':('possible_lunar_eclipse' if eclipse else 'moon_separation_below_10deg') if not ok else 'ok',
                           'low_altitude_caution':bool(altitude.flat[i]<10)})
        return result

    rows = fields(directions[:,0],directions[:,1])
    star_rows = []
    star_metadata = {'available':False, 'reason':'No date: no unique horizontal stellar positions.'}
    if utc is not None:
        stars, provenance = load_stars()
        star_az, star_alt, astrometry = horizontal_stars(stars,observer,utc,allow_moon=True)
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
    return {'schema_version':1, 'mode':config['mode'], 'utc':utc, 'input_scene':config,
            'observer':asdict(observer), 'atmosphere':asdict(atmosphere), 'cloud_free':True,
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
                     'physical_error_bound':None, 'document':'docs/OBSERVATION_MODEL.md'}}


def write_observation(report: dict, output: Path):
    """Validate serialisability before creating outputs; no NaN/Infinity in JSON."""
    import csv
    stars = report['stars']
    summary = {key:value for key,value in report.items() if key != 'stars'}
    serialized = json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    output.mkdir(parents=True,exist_ok=True)
    (output/'observation.json').write_text(serialized)
    for name,rows in [('directions',report['directions']),('stars',stars)]:
        columns = list(rows[0]) if rows else ['hr','name','azimuth_deg','altitude_deg',
                    'catalogue_v_mag','extinction_v_mag','observed_v_mag','flux_relative_to_v_mag0']
        with (output/f'{name}.csv').open('w',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=columns,lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
