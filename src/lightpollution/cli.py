"""Small reproducible command-line workflow; credentials never enter output files."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import requests

from .data import Bounds, audit, load_region, read_region, save_region
from .download import catalogue, download_composite, write_manifest
from .periods import CompositePeriod
from .physics import Atmosphere, Observer, SolverSettings, sky_grid
from .sources import regional_sources, synthetic_cities
from .visualization import source_map, sky_plot

DEFAULT_ALTITUDES = [1,2,3,5,7.5,10,15,20,30,45,60,75,90]


def make_parser() -> argparse.ArgumentParser:
    parser=argparse.ArgumentParser(description="Black Marble audit and experimental directional skyglow")
    commands=parser.add_subparsers(dest='command',required=True)
    cat=commands.add_parser('catalogue',help='Inspect public NASA file metadata; no authentication needed')
    cat.add_argument('--year',type=int,default=2025)
    cat.add_argument('--month',type=int,choices=range(1,13),help='Monthly VNP46A3; omit for annual VNP46A4')
    cat.add_argument('--output',type=Path)
    fetch=commands.add_parser('download',help='Download four pinned Petersburg HDF5 tiles')
    fetch.add_argument('--year',type=int,default=2025)
    fetch.add_argument('--month',type=int,choices=range(1,13))
    fetch.add_argument('--directory',type=Path)
    fetch.add_argument('--manifest',type=Path)
    check=commands.add_parser('audit',help='Check and extract the 29–32 E, 59–61 N window')
    check.add_argument('--year',type=int,default=2025)
    check.add_argument('--month',type=int,choices=range(1,13))
    check.add_argument('--directory',type=Path)
    check.add_argument('--output',type=Path)
    check.add_argument('--no-plots',action='store_true',help='Export numerical audit only')
    ground=commands.add_parser('audit-ground',help='Audit a downloaded public native-band reference; no rendering')
    ground.add_argument('--kind',required=True,choices=['astmon','galicia'])
    ground.add_argument('--file',type=Path,required=True)
    ground.add_argument('--output',type=Path,required=True)
    context=commands.add_parser('context',help='Sun/Moon geometry; not a prediction of lunar sky brightness')
    context.add_argument('--utc',required=True)
    context.add_argument('--latitude',type=float,default=60.30)
    context.add_argument('--longitude',type=float,default=30.315)
    context.add_argument('--height',type=float,default=20)
    context.add_argument('--look-azimuth',type=float,default=180)
    context.add_argument('--look-altitude',type=float,default=35)
    context.add_argument('--output',type=Path,required=True)
    observe=commands.add_parser('observe',help='Numerical dated/undated V-band scene with separate lunar/city/natural components')
    observe.add_argument('--scene',type=Path,required=True,help='Versioned JSON scene; grid paths relative to this file')
    observe.add_argument('--output',type=Path,required=True)
    view=commands.add_parser('render',help='Stars and computed skyglow as seen from the saved observer')
    view.add_argument('--sky',type=Path,default=Path('artifacts/petersburg-2025/observer-sky/sky-grid.npz'))
    view.add_argument('--output',type=Path,default=Path('artifacts/petersburg-2025/observer-view'))
    view.add_argument('--utc',required=True,help='Explicit UTC time, e.g. 2026-11-09T21:00:00Z')
    view.add_argument('--artificial-zenith-mag',type=float,required=True,
                      help='ASSUMED artificial zenith V brightness in mag/arcsec2; not satellite calibration')
    view.add_argument('--natural-mag',type=float,default=21.6)
    view.add_argument('--dark-limiting-mag',type=float,default=6.5)
    view.add_argument('--kstars-directory',type=Path,help='Also composite the computed field on an exported KStars raster')
    view.add_argument('--look-azimuth',type=float,default=180)
    view.add_argument('--look-altitude',type=float,default=35)
    view.add_argument('--horizontal-fov',type=float,default=110)
    for name,help_text in (('simulate','Calculate from a successfully audited NASA region'),
                           ('demo','Synthetic two-city example; not NASA measurements')):
        run=commands.add_parser(name,help=help_text)
        run.add_argument('--output',type=Path,default=Path('artifacts/'+('synthetic-demo' if name=='demo' else 'petersburg-2025/observer-sky')))
        run.add_argument('--region',type=Path,default=Path('artifacts/petersburg-2025/source-grid.npz'))
        run.add_argument('--latitude',type=float,default=60.30)
        run.add_argument('--longitude',type=float,default=30.315)
        run.add_argument('--height',type=float,default=20)
        run.add_argument('--azimuth-step',type=float,default=10)
        run.add_argument('--altitudes',type=float,nargs='+',default=DEFAULT_ALTITUDES if name=='demo' else [0,.5]+DEFAULT_ALTITUDES)
        run.add_argument('--ray-segments',type=int,default=96)
        run.add_argument('--path-quadrature',type=int,default=12)
        run.add_argument('--aod',type=float,default=0.10)
        run.add_argument('--horizontal-fraction',type=float,default=0)
        run.add_argument('--block',type=int,default=12)
        run.add_argument('--exclude-radius-m',type=float,default=3000)
        run.add_argument('--include-water',action='store_true')
        run.add_argument('--refine-below-altitude',type=float,default=5,
                         help='Refine source patches near sightlines at/below this altitude')
    return parser


def run_simulation(args) -> dict:
    if not 0<args.azimuth_step<=90 or not np.isclose(360/args.azimuth_step,round(360/args.azimuth_step)):
        raise ValueError("Azimuth step must divide 360 exactly and be in (0,90].")
    if args.altitudes != sorted(set(args.altitudes)) or not (0 <= min(args.altitudes) <= max(args.altitudes) <= 90):
        raise ValueError("Altitudes must be unique, increasing and inside [0,90].")
    observer=Observer(args.latitude,args.longitude,args.height)
    atmosphere=Atmosphere(aerosol_optical_depth=args.aod)
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
        sources,provenance=regional_sources(arrays,bounds,observer,args.block,args.exclude_radius_m,include_water=args.include_water)
        provenance.update({"input_kind":"blackmarble", "region_file":str(args.region),"region_metadata":metadata})
        refined_counts=[]
        if not 0<=args.refine_below_altitude<=90:
            raise ValueError('Ray refinement altitude must be in [0,90].')
        def source_factory(az,alt):
            if alt>args.refine_below_altitude:
                return sources
            selected,_=regional_sources(arrays,bounds,observer,args.block,args.exclude_radius_m,
                                         include_water=args.include_water,refinement_direction=(az,alt))
            refined_counts.append(len(selected.power))
            return selected
        period_label=CompositePeriod(metadata['year'],metadata.get('month')).label
        title=f"Петербург, VIIRS {period_label}: условный региональный сценарий"
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
             "solver":asdict(settings),"shape":list(radiance.shape),"elapsed_seconds":elapsed,
             "azimuth_convention":"North=0, East=90, clockwise", "raw_units":"W/(m2 sr)" if args.command=='demo' else "m^-2 sr^-1 per unit total assumed source power",
             "absolute_visible_photometry_calibrated":False,
             "maximum":{"azimuth_deg":float(azimuths[location[1]]),"altitude_deg":float(altitudes[location[0]]),"value":float(radiance[location])},
             "physical_error_bound":None,"model_document":"docs/MODEL.md"}
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
    sky_plot(azimuths,altitudes,radiance,args.output/'sky.png',title)
    return summary


def main():
    parser=make_parser()
    args=parser.parse_args()
    try:
        if args.command=='catalogue':
            period=CompositePeriod(args.year,args.month)
            manifest=catalogue(args.year) if args.month is None else catalogue(args.year,month=args.month)
            write_manifest(args.output or Path(f'artifacts/nasa-{period.label}-catalogue.json'),manifest)
            print(f"Catalogue only: {len(manifest['files'])} files; not downloaded.")
        elif args.command=='download':
            period=CompositePeriod(args.year,args.month)
            download_composite(args.directory or Path(f'data/blackmarble/{period.label}'),args.year,
                               manifest_path=args.manifest,month=args.month)
        elif args.command=='audit':
            period=CompositePeriod(args.year,args.month)
            args.directory=args.directory or Path(f'data/blackmarble/{period.label}')
            args.output=args.output or Path(f'artifacts/petersburg-{period.label}')
            arrays,attributes=read_region(args.directory,Bounds(),args.year,args.month)
            report=audit(arrays,Bounds())
            report['observation_period']=period.metadata()
            args.output.mkdir(parents=True,exist_ok=True)
            (args.output/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
            (args.output/'metadata.json').write_text(json.dumps(attributes,ensure_ascii=False,indent=2)+'\n')
            inputs=[{"filename":value['filename'],"sha256":value['sha256']} for value in attributes.values()]
            save_region(args.output/'source-grid.npz',arrays,Bounds(),
                        {**period.metadata(),"product":period.product+'.002',"inputs":inputs})
            if not args.no_plots:
                source_map(arrays,Bounds(),args.output/'sources.png',year=args.year,month=args.month)
            print(json.dumps(report,ensure_ascii=False,indent=2))
        elif args.command=='audit-ground':
            from .ground import audit_astmon, audit_galicia
            report=(audit_astmon if args.kind=='astmon' else audit_galicia)(args.file)
            write_manifest(args.output,report)
            print(json.dumps({key:report[key] for key in ('dataset','rows','sha256')},ensure_ascii=False,indent=2))
        elif args.command=='context':
            from .observation import observation_context
            report=observation_context(Observer(args.latitude,args.longitude,args.height),args.utc,
                                       args.look_azimuth,args.look_altitude)
            write_manifest(args.output,report)
            print(json.dumps(report,ensure_ascii=False,indent=2))
        elif args.command=='observe':
            from .scene import calculate_observation, write_observation
            report=calculate_observation(json.loads(args.scene.read_text()),args.scene.resolve().parent)
            write_observation(report,args.output)
            print(json.dumps({'mode':report['mode'],'utc':report['utc'],
                              'directions':len(report['directions']),
                              'supported_directions':sum(row['supported'] for row in report['directions']),
                              'stars':len(report['stars']),'output':str(args.output)},ensure_ascii=False))
        elif args.command=='render':
            from .skyview import DisplayScenario, render_observer
            scenario=DisplayScenario(args.artificial_zenith_mag,args.natural_mag,args.dark_limiting_mag)
            if args.kstars_directory:
                kstars_metadata=json.loads((args.kstars_directory/'kstars-summary.json').read_text())
                if kstars_metadata['utc']!=args.utc:
                    raise ValueError('KStars export and observer renderer timestamps differ.')
            summary=render_observer(args.sky,args.output,args.utc,scenario,
                                    args.look_azimuth,args.look_altitude,args.horizontal_fov)
            if args.kstars_directory:
                from .skyview import composite_kstars
                composite_kstars(args.sky,args.kstars_directory,scenario)
            print(json.dumps({key:summary[key] for key in ('observer','astrometry','display_scenario',
                                                         'catalogue_rows','above_horizon','dark_visible','scenario_visible')},
                             ensure_ascii=False,indent=2))
        else:
            print(json.dumps(run_simulation(args),ensure_ascii=False,indent=2))
    except (ValueError,RuntimeError,OSError,requests.RequestException,KeyError) as exc:
        parser.exit(1,f"Error: {exc}\n")


if __name__=='__main__':
    main()
