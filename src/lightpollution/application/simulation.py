"""Legacy simulation workflow shared by CLI and research scripts."""

from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from ..data import load_region
from ..atmosphere_config import atmosphere_from_mapping, atmosphere_origin
from ..periods import CompositePeriod
from ..physics import Atmosphere, Observer, SolverSettings, Sources, sky_grid
from ..sources import regional_sources, synthetic_cities
from ..visualization import sky_plot

def run_simulation(args) -> dict:
    if not 0<args.azimuth_step<=90 or not np.isclose(360/args.azimuth_step,round(360/args.azimuth_step)):
        raise ValueError("Azimuth step must divide 360 exactly and be in (0,90].")
    if args.altitudes != sorted(set(args.altitudes)) or not (0 <= min(args.altitudes) <= max(args.altitudes) <= 90):
        raise ValueError("Altitudes must be unique, increasing and inside [0,90].")
    observer=Observer(args.latitude,args.longitude,args.height)
    atmospheric_parameters=json.loads(args.atmosphere_file.read_text()) if args.atmosphere_file else {}
    if not isinstance(atmospheric_parameters,dict):
        raise ValueError('Atmosphere file must be a JSON object of physical parameters.')
    if args.aod is not None:
        atmospheric_parameters['aerosol_optical_depth']=args.aod
    atmosphere=atmosphere_from_mapping(atmospheric_parameters)
    settings=SolverSettings(args.ray_segments,args.path_quadrature,args.horizontal_fraction)
    if args.command=='demo':
        source_factory=None
        sources=synthetic_cities(observer)
        provenance={"input_kind":"synthetic", "actual_nasa_pixels":0,
                    "description":"Two specified test cities: 40 km south and 25 km east; total test power 1.25 W.",
                    "source_power_units":"W (synthetic)","effective_sources":len(sources.power)}
        title='Учебный пример: два заданных города. Это не данные NASA'
    else:
        arrays,bounds,metadata=load_region(args.region)
        source_scale=float(metadata.get('source_power_multiplier',1.))
        composite=args.composite or metadata.get('source_composite','NearNadir')
        if metadata.get('source_composite',composite)!=composite:
            raise ValueError('Cannot change the composite after seasonal support matching.')
        if not np.isfinite(source_scale) or source_scale<=0:
            raise ValueError('Source amplitude multiplier must be positive and finite.')
        if 'seasonal_selection' in metadata and metadata['seasonal_selection'] != {
                'observer':asdict(observer),'exclude_radius_m':args.exclude_radius_m,'include_water':args.include_water}:
            raise ValueError('Seasonal source selection must match observer, exclusion radius and water policy.')
        sources,provenance=regional_sources(arrays,bounds,observer,args.block,args.exclude_radius_m,
                                          include_water=args.include_water,composite=composite)
        sources=Sources(sources.positions_m,sources.power*source_scale)
        provenance['source_power_multiplier']=source_scale
        if 'source_power_multiplier' in metadata:
            provenance['normalisation']=f'Total source proxy = {source_scale:.12g} relative to the fixed reference month.'
        provenance.update({"input_kind":"blackmarble", "region_file":str(args.region),"region_metadata":metadata})
        refined_counts=[]
        if not 0<=args.refine_below_altitude<=90:
            raise ValueError('Ray refinement altitude must be in [0,90].')
        def source_factory(az,alt):
            if alt>args.refine_below_altitude:
                return sources
            selected,_=regional_sources(arrays,bounds,observer,args.block,args.exclude_radius_m,
                                         include_water=args.include_water,refinement_direction=(az,alt),composite=composite)
            refined_counts.append(len(selected.power))
            return Sources(selected.positions_m,selected.power*source_scale)
        period_label=CompositePeriod(metadata['year'],metadata.get('month')).label
        title=f"VIIRS {period_label}, {observer.latitude_deg:g}°N {observer.longitude_deg:g}°E: условный сценарий"
    azimuths=np.arange(0,360,args.azimuth_step)
    altitudes=np.asarray(args.altitudes)
    started=time.perf_counter()
    radiance=sky_grid(observer,sources,azimuths,altitudes,atmosphere,settings,source_factory=source_factory)
    if args.command!='demo':
        provenance['ray_refinement']={'at_or_below_altitude_deg':args.refine_below_altitude,
                                      'minimum_effective_sources':min(refined_counts,default=len(sources.power)),
                                      'maximum_effective_sources':max(refined_counts,default=len(sources.power))}
    elapsed=time.perf_counter()-started
    location=np.unravel_index(np.argmax(radiance),radiance.shape)
    summary={"provenance":provenance,"observer":asdict(observer),"atmosphere":asdict(atmosphere),
             "atmosphere_origin":atmosphere_origin(atmospheric_parameters),
             "solver":asdict(settings),"shape":list(radiance.shape),"elapsed_seconds":elapsed,
             "azimuth_convention":"North=0, East=90, clockwise", "raw_units":"W/(m2 sr)" if args.command=='demo' else "m^-2 sr^-1 per unit reference assumed source power",
             "absolute_visible_photometry_calibrated":False,
             "maximum":{"azimuth_deg":float(azimuths[location[1]]),"altitude_deg":float(altitudes[location[0]]),"value":float(radiance[location])},
             "physical_error_bound":None,"model_document":"MODELS.md"}
    for target in (5,15,45):
        if target in altitudes:
            row=int(np.flatnonzero(altitudes==target)[0])
            south,north=radiance[row,np.argmin(abs(azimuths-180))],radiance[row,0]
            summary[f"south_over_north_at_{target}deg"]=float(south/north) if north>0 else None
    args.output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output/'sky-grid.npz',azimuth_deg=azimuths,altitude_deg=altitudes,radiance=radiance,
                        metadata_json=json.dumps(summary,ensure_ascii=False))
    az_mesh,alt_mesh=np.meshgrid(azimuths,altitudes)
    np.savetxt(args.output/'sky-grid.csv',np.column_stack((az_mesh.ravel(),alt_mesh.ravel(),radiance.ravel())),
               delimiter=',',header='azimuth_deg,altitude_deg,radiance',comments='',fmt='%.12g')
    (args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    if not args.no_plots:
        sky_plot(azimuths,altitudes,radiance,args.output/'sky.png',title)
    return summary
