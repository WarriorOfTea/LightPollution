"""Reproduce the real-photograph comparison without changing the physical solver.

The case is specified in artifacts/photo-comparison/case.json. Individual rays
are checkpointed in scratch; changing inputs or solver code invalidates the cache.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from lightpollution.data import Bounds, audit, load_region, read_region, save_region
from lightpollution.physics import Atmosphere, Observer, SolverSettings, integrate_direction
from lightpollution.sources import regional_sources
from lightpollution.visualization import sky_plot


CONTEXT=None


def initialise(region,observer,atmosphere,settings,block,exclusion):
    global CONTEXT
    arrays,bounds,_=load_region(Path(region))
    obs=Observer(**observer)
    sources,report=regional_sources(arrays,bounds,obs,block,exclusion)
    CONTEXT=(arrays,bounds,obs,Atmosphere(**atmosphere),SolverSettings(**settings),block,exclusion,sources)


def calculate(task):
    row,col,az,alt=task
    arrays,bounds,observer,atmosphere,settings,block,exclusion,sources=CONTEXT
    if alt<=5:
        sources,_=regional_sources(arrays,bounds,observer,block,exclusion,refinement_direction=(az,alt))
    value=integrate_direction(observer,sources,az,alt,atmosphere,settings)
    return row,col,value,len(sources.power)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',type=Path,default=Path('artifacts/photo-comparison/case.json'))
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    case=json.loads(args.case.read_text())
    directory=Path(case['model_directory']);directory.mkdir(parents=True,exist_ok=True)
    bounds=Bounds(**case['bounds'])
    region=directory/'source-grid.npz'
    if not region.exists():
        arrays,attributes=read_region(Path(case['raw_directory']),bounds,case['year'])
        (directory/'audit.json').write_text(json.dumps(audit(arrays,bounds),indent=2)+'\n')
        (directory/'metadata.json').write_text(json.dumps(attributes,indent=2)+'\n')
        inputs=[{'filename':v['filename'],'sha256':v['sha256']} for v in attributes.values()]
        save_region(region,arrays,bounds,{'year':case['year'],'product':'VNP46A4.002','inputs':inputs})
    arrays,stored_bounds,metadata=load_region(region)
    if stored_bounds != bounds or metadata.get('year') != case['year']:
        raise ValueError('Saved region does not match the case bounds/year; select a different model directory.')
    observer=Observer(**case['observer']);atmosphere=Atmosphere(**case['atmosphere'])
    settings=SolverSettings(**case['solver'])
    _,provenance=regional_sources(arrays,bounds,observer,case['block'],case['exclude_radius_m'])
    provenance.update(input_kind='blackmarble',region_file=str(region),region_metadata=metadata)
    az=np.asarray(case['azimuth_deg']);alt=np.asarray(case['altitude_deg'])
    if (az.ndim != 1 or alt.ndim != 1 or len(az)<2 or len(alt)<2
            or az[0]!=0 or az[-1]>=360 or alt[0]!=0 or alt[-1]!=90
            or not np.all(np.isfinite(az)) or not np.all(np.isfinite(alt))
            or np.any(np.diff(az)<=0) or np.any(np.diff(alt)<=0)):
        raise ValueError('The case must specify an ordered complete sky grid.')
    field=np.zeros((len(alt),len(az)))
    core={name:hashlib.sha256(Path('src/lightpollution',name).read_bytes()).hexdigest()
          for name in ('physics.py','sources.py')}
    fingerprint=hashlib.sha256(json.dumps({'case':case,'core':core,
                'region_sha256':hashlib.sha256(region.read_bytes()).hexdigest()},sort_keys=True).encode()).hexdigest()
    checkpoint=Path('scratch/photo-case-rays.jsonl');checkpoint.parent.mkdir(exist_ok=True)
    completed={}
    if checkpoint.exists():
        lines=checkpoint.read_text().splitlines()
        if json.loads(lines[0])['fingerprint']!=fingerprint:
            raise ValueError('Checkpoint belongs to other inputs; move it before rerunning.')
        for line in lines[1:]:
            r,c,value,count=json.loads(line);completed[(r,c)]=(value,count)
    else:checkpoint.write_text(json.dumps({'fingerprint':fingerprint})+'\n')
    tasks=[(r,c,float(a),float(h)) for r,h in enumerate(alt) for c,a in enumerate(az) if h!=90 or c==0]
    for (r,c),(value,count) in completed.items():field[r,c]=value
    counts=[count for value,count in completed.values()]
    pending=[t for t in tasks if (t[0],t[1]) not in completed]
    print(f'{len(tasks)} independent rays, {len(completed)} resumed, base sources {provenance["effective_sources"]}',flush=True)
    started=time.perf_counter()
    arguments=(str(region),asdict(observer),asdict(atmosphere),asdict(settings),case['block'],case['exclude_radius_m'])
    with checkpoint.open('a') as log, ProcessPoolExecutor(max_workers=args.workers,initializer=initialise,initargs=arguments) as executor:
        futures=[executor.submit(calculate,t) for t in pending]
        for n,future in enumerate(as_completed(futures),len(completed)+1):
            r,c,value,count=future.result();field[r,c]=value;counts.append(count)
            log.write(json.dumps([r,c,value,count])+'\n');log.flush()
            if n%72==0:print(f'Rays {n}/{len(tasks)}',flush=True)
    field[-1,:]=field[-1,0]
    provenance['ray_refinement']={'at_or_below_altitude_deg':5,'minimum_sources':min(counts),'maximum_sources':max(counts)}
    if not np.all(np.isfinite(field)) or np.any(field<=0):
        raise ValueError('Every computed sky value must be finite and positive.')
    summary={'provenance':provenance,'observer':asdict(observer),'atmosphere':asdict(atmosphere),
             'solver':asdict(settings),'shape':list(field.shape),
             'timing':{'current_invocation_elapsed_seconds':time.perf_counter()-started,
                       'rays_reused':len(completed),'rays_computed_this_invocation':len(pending),
                       'full_campaign_elapsed_seconds':None,
                       'scope':'Current invocation only; interrupted earlier passes are not timed here.'},
             'raw_units':'m^-2 sr^-1 per unit total assumed source power',
             'absolute_visible_photometry_calibrated':False,'physical_error_bound':None,
             'azimuth_convention':'North=0, East=90, clockwise','model_document':'docs/MODEL.md',
             'photo_case':str(args.case)}
    np.savez_compressed(directory/'sky-grid.npz',azimuth_deg=az,altitude_deg=alt,radiance=field,
                        metadata_json=json.dumps(summary,ensure_ascii=False))
    aa,hh=np.meshgrid(az,alt)
    np.savetxt(directory/'sky-grid.csv',np.column_stack((aa.ravel(),hh.ravel(),field.ravel())),delimiter=',',
               header='azimuth_deg,altitude_deg,radiance',comments='',fmt='%.12g')
    (directory/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    sky_plot(az,alt,field,directory/'sky.png',f'Источники NASA {case["year"]}: точка фотографии')
    print('Completed',summary['timing'],flush=True)


if __name__=='__main__':main()
