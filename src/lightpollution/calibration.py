"""Native-band night-level calibration with explicit cloud QA and held-out nights."""

from datetime import datetime, timedelta, timezone
from dataclasses import asdict
from pathlib import Path
from zoneinfo import ZoneInfo
import csv

from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, get_body, solar_system_ephemeris
from astropy.time import Time
from astropy.utils import iers
import numpy as np

from .download import sha256
from .physics import Observer
from .skyview import sample_sky


def astmon_candidates(path: Path, observer: Observer, year: int, months: list[int],
                       timezone_name: str = 'Europe/Madrid', first_hour: int = 0,
                       last_hour: int = 4) -> tuple[list[dict],dict]:
    """Astronomical/hour selection never certifies cloud-free weather."""
    if not months or any(not 1<=m<=12 for m in months) or not 0<=first_hour<last_hour<=24:
        raise ValueError('Invalid months or local-hour window.')
    local_zone=ZoneInfo(timezone_name)
    records=[]
    native_rows=finite_rows=hour_rows=0
    with path.open(newline='',encoding='utf-8-sig') as stream:
        for row in csv.DictReader(stream,delimiter=';'):
            if row['Filter']!='Johnson-V':
                continue
            instant=datetime.strptime(row['Datetime_UTC'],'%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
            if instant.year!=year or instant.month not in months:
                continue
            native_rows+=1
            mag,error=float(row['Magnitude']),float(row['Error'])
            if not np.isfinite(mag) or not np.isfinite(error) or error<=0:
                continue
            finite_rows+=1
            local=instant.astimezone(local_zone)
            if not first_hour<=local.hour<last_hour:
                continue
            hour_rows+=1
            records.append({'utc':instant.isoformat().replace('+00:00','Z'),
                            'period':instant.strftime('%Y-%m'),'night_id':(local-timedelta(hours=12)).date().isoformat(),
                            'mag_arcsec2':mag,'reported_error_mag':error,
                            'cloud_free_confirmed':None,'cloud_qa_source':None})
    selected=[]
    # Bounded vector batches keep memory and ephemeris transforms predictable.
    location=EarthLocation.from_geodetic(observer.longitude_deg*u.deg,observer.latitude_deg*u.deg,
                                         observer.height_m*u.m)
    for offset in range(0,len(records),512):
        batch=records[offset:offset+512]
        times=Time([r['utc'][:-1] for r in batch],format='isot',scale='utc')
        frame=AltAz(obstime=times,location=location,pressure=0*u.hPa)
        with iers.conf.set_temp('auto_download',False),iers.conf.set_temp('auto_max_age',None),solar_system_ephemeris.set('builtin'):
            sun=get_body('sun',times,location).transform_to(frame).alt.deg
            moon=get_body('moon',times,location).transform_to(frame).alt.deg
        for row,s,m in zip(batch,sun,moon,strict=True):
            if s<=-18 and m<=-5:
                selected.append({**row,'sun_altitude_deg':float(s),'moon_altitude_deg':float(m)})
    report={'dataset':'Madrid AstMon Johnson-V, doi:10.5281/zenodo.4633001',
            'input_sha256':sha256(path),'year':year,'months':months,'selected_native_rows':native_rows,
            'finite_positive_error_rows':finite_rows,'local_hour_rows':hour_rows,
            'astronomical_moonless_candidates':len(selected),
            'astronomical_rejections_after_hour_filter':hour_rows-len(selected),
            'certified_cloud_free_candidates':0,'unknown_cloud_candidates':len(selected),
            'timezone':timezone_name,'local_hour_window':[first_hour,last_hour],
            'sun_altitude_max_deg':-18,'moon_altitude_max_deg':-5,
            'observer':asdict(observer),
            'night_definition':'local noon-to-noon, identified by previous evening date',
            'height_and_atmosphere_are_scenarios':True,'cloud_flags_in_file':False}
    return selected,report


def model_zenith_predictions(paths: list[Path], observer: Observer) -> tuple[dict,dict]:
    import json
    values={}
    common=reference=atmosphere=composite=None
    inputs=[]
    for path in paths:
        with np.load(path,allow_pickle=False) as data:
            meta=json.loads(str(data['metadata_json'].item()))
            region=meta['provenance']['region_metadata']
            period=f"{region['year']:04d}-{region['month']:02d}"
            if meta['observer']!=asdict(observer) or not region.get('common_mask_sha256'):
                raise ValueError('Calibration grids require the same observer and matched monthly support.')
            signature=(region['common_mask_sha256'],region['normalisation_reference_period'],meta['atmosphere'],region.get('source_composite'))
            if common is None:
                common,reference,atmosphere,composite=signature
            elif signature!=(common,reference,atmosphere,composite):
                raise ValueError('Calibration grids have different source support/reference or atmosphere.')
            if period in values:
                raise ValueError('Duplicate model period.')
            values[period]=float(sample_sky(data['azimuth_deg'],data['altitude_deg'],data['radiance'],0,90))
            if not np.allclose(data['radiance'][-1],data['radiance'][-1,0],rtol=1e-10,atol=0):
                raise ValueError('Model zenith must be unique.')
            inputs.append({'path':str(path),'sha256':sha256(path)})
    if reference not in values or any(not np.isfinite(v) or v<=0 for v in values.values()):
        raise ValueError('Positive model responses and the source reference month are required.')
    baseline=values[reference]
    return {key:value/baseline for key,value in values.items()}, {
        'reference_period':reference,'reference_raw_zenith':baseline,'common_mask_sha256':common,
        'observer':asdict(observer),'atmosphere':atmosphere,'composite':composite,'inputs':inputs}


def nightly_values(records: list[dict], reference_mag: float = 22.) -> list[dict]:
    groups={}
    for row in records:
        groups.setdefault((row['period'],row['night_id']),[]).append(row)
    result=[]
    for (period,night),rows in sorted(groups.items()):
        magnitudes=np.asarray([r['mag_arcsec2'] for r in rows],dtype=float)
        if not np.all(np.isfinite(magnitudes)):
            raise ValueError('Nightly aggregation requires finite magnitudes.')
        flux=10**(-.4*(magnitudes-reference_mag))
        if np.any(~np.isfinite(flux)) or np.any(flux<=0):
            raise ValueError('Brightness is outside the supported floating-point range.')
        cloud=all(r.get('cloud_free_confirmed') is True and r.get('cloud_qa_source') for r in rows)
        if any(r.get('cloud_free_confirmed') is False for r in rows):
            continue
        mean=float(np.mean(flux))
        result.append({'period':period,'night_id':night,'samples':len(rows),'brightness_relative':mean,
                       'mag_arcsec2':float(reference_mag-2.5*np.log10(mean)),
                       'cloud_free_confirmed':True if cloud else None,
                       'within_night_relative_std':float(np.std(flux)),
                       'weight':1.})
    return result


def fit_and_check(nights: list[dict], relative_predictions: dict[str,float], training_periods: list[str],
                  natural_mag: float = 21.6, reference_mag: float = 22.,
                  allow_unverified_clouds: bool = False) -> dict:
    """One nonnegative city coefficient; equal weight per night, no test leakage."""
    if not nights or not np.isfinite(reference_mag) or not np.isfinite(natural_mag):
        raise ValueError('Finite magnitude references and nightly data are required.')
    if any(row.get('cloud_free_confirmed') is False for row in nights):
        raise ValueError('Known cloudy nights must be excluded, including from provisional fits.')
    if any(not np.isfinite(row['brightness_relative']) or row['brightness_relative']<=0 for row in nights):
        raise ValueError('Nightly brightness must be positive and finite.')
    if any(not row['cloud_free_confirmed'] for row in nights) and not allow_unverified_clouds:
        raise ValueError('Cloud-free QA is missing. A provisional fit requires explicit opt-in.')
    if any(p not in relative_predictions or not np.isfinite(relative_predictions[p]) or relative_predictions[p]<=0
           for p in {row['period'] for row in nights}):
        raise ValueError('Every observed period needs a positive model prediction.')
    train=[row for row in nights if row['period'] in training_periods]
    check=[row for row in nights if row['period'] not in training_periods]
    if not train or not check or {r['night_id'] for r in train}&{r['night_id'] for r in check}:
        raise ValueError('Training and held-out nights must be nonempty and disjoint.')
    natural=10**(-.4*(natural_mag-reference_mag))
    x=np.array([relative_predictions[row['period']] for row in train])
    y=np.array([row['brightness_relative'] for row in train])
    coefficient=float(np.dot(x,y-natural)/np.dot(x,x))
    if not np.isfinite(coefficient) or coefficient<=0:
        raise ValueError('The fitted artificial component is not positive; revisit data/assumptions.')
    def evaluate(rows):
        details=[]
        for row in rows:
            predicted=coefficient*relative_predictions[row['period']]+natural
            mag=reference_mag-2.5*np.log10(predicted)
            details.append({**row,'predicted_mag_arcsec2':float(mag),
                            'predicted_relative':float(predicted),
                            'residual_mag_model_minus_observed':float(mag-row['mag_arcsec2'])})
        residual=np.array([row['residual_mag_model_minus_observed'] for row in details])
        return {'nights':len(rows),'samples':sum(row['samples'] for row in rows),
                'bias_mag':float(np.mean(residual)),'mae_mag':float(np.mean(abs(residual))),
                'rmse_mag':float(np.sqrt(np.mean(residual**2))),'records':details}
    return {'status':'provisional_unverified_clouds' if any(not r['cloud_free_confirmed'] for r in nights) else 'QA_supplied_externally',
            'coefficient_city_relative':coefficient,
            'reference_city_mag_arcsec2':float(reference_mag-2.5*np.log10(coefficient)),
            'reference_mag_arcsec2':reference_mag,'natural_mag_arcsec2':natural_mag,
            'model_relative_predictions':relative_predictions,
            'training_periods':training_periods,'training':evaluate(train),'held_out':evaluate(check),
            'independent_night_split':True,'field_accuracy_certified':False,
            'assumptions':['Native AstMon Johnson-V treated as effective V; residual response/zero-point uncertainty.',
                           'Fixed natural background and baseline atmosphere; no measured cloud/aerosol reconstruction.',
                           'Equal weight per night; reported per-sample errors are not independent night uncertainties.']}
