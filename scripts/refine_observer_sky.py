"""Recompute low rays with ray-aware sources; preserve unchanged high rays.

This accelerates updating an existing real-NASA simulation after source refinement.
The ordinary simulate command now applies the same refinement directly.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from lightpollution.data import load_region
from lightpollution.physics import Atmosphere, Observer, SolverSettings, integrate_direction
from lightpollution.sources import regional_sources
from lightpollution.visualization import sky_plot


CONTEXT=None


def initialise(region, observer, atmosphere, settings, block, exclusion, water):
    global CONTEXT
    arrays,bounds,_=load_region(Path(region))
    CONTEXT=(arrays,bounds,Observer(**observer),Atmosphere(**atmosphere),SolverSettings(**settings),block,exclusion,water)


def calculate(task):
    row,col,az,alt=task
    arrays,bounds,observer,atmosphere,settings,block,exclusion,water=CONTEXT
    sources,_=regional_sources(arrays,bounds,observer,block,exclusion,include_water=water,refinement_direction=(az,alt))
    return row,col,integrate_direction(observer,sources,az,alt,atmosphere,settings),len(sources.power)


def main():
    directory=Path('artifacts/petersburg-2025/observer-sky')
    with np.load(directory/'sky-grid.npz',allow_pickle=False) as data:
        az,alt,values=data['azimuth_deg'],data['altitude_deg'],data['radiance'].copy()
        summary=json.loads(str(data['metadata_json']))
    provenance=summary['provenance']
    if provenance['input_kind']!='blackmarble':raise ValueError('A real NASA simulation is required.')
    tasks=[(r,c,float(a),float(h)) for r,h in enumerate(alt) if h<=5 for c,a in enumerate(az)]
    started=time.perf_counter();counts=[]
    arguments=(provenance['region_file'],summary['observer'],summary['atmosphere'],summary['solver'],
               provenance['block_pixels'],provenance['excluded_local_radius_m'],provenance['include_water'])
    region_hash=hashlib.sha256(Path(provenance['region_file']).read_bytes()).hexdigest()
    core_hashes={name:hashlib.sha256(Path('src/lightpollution',name).read_bytes()).hexdigest()
                 for name in ('physics.py','sources.py')}
    fingerprint=hashlib.sha256(json.dumps({'arguments':arguments,'az':az.tolist(),'alt':alt.tolist(),
                                          'region_sha256':region_hash,'core_sha256':core_hashes,
                                          'algorithm':'ray-refinement-v1'},sort_keys=True).encode()).hexdigest()
    checkpoint=Path('scratch/observer-refinement-progress.jsonl')
    checkpoint.parent.mkdir(exist_ok=True)
    completed={}
    if checkpoint.exists():
        lines=checkpoint.read_text().splitlines()
        if json.loads(lines[0])['fingerprint']!=fingerprint:
            raise ValueError('Checkpoint belongs to a different simulation; move it before recomputing.')
        for line in lines[1:]:
            r,c,value,count=json.loads(line)
            completed[(r,c)]=(value,count)
    else:
        checkpoint.write_text(json.dumps({'fingerprint':fingerprint})+'\n')
    for (r,c),(value,count) in completed.items():
        values[r,c]=value;counts.append(count)
    pending=[t for t in tasks if (t[0],t[1]) not in completed]
    print(f'Resumed {len(completed)}/{len(tasks)} directions',flush=True)
    with checkpoint.open('a') as progress, ProcessPoolExecutor(max_workers=4,initializer=initialise,initargs=arguments) as executor:
        futures=[executor.submit(calculate,t) for t in pending]
        for index,future in enumerate(as_completed(futures),len(completed)+1):
            r,c,value,count=future.result();values[r,c]=value;counts.append(count)
            progress.write(json.dumps([r,c,value,count])+'\n');progress.flush()
            if index%36==0:print(f'Refined {index}/{len(tasks)} directions',flush=True)
    duration=time.perf_counter()-started
    summary['initial_elapsed_seconds']=summary['elapsed_seconds']
    summary['refinement_elapsed_seconds']=duration
    summary['elapsed_seconds']+=duration
    provenance['ray_refinement']={'at_or_below_altitude_deg':5,
                                  'minimum_effective_sources':min(counts),'maximum_effective_sources':max(counts),
                                  'refined_directions':len(tasks),'workers':4}
    location=np.unravel_index(np.argmax(values),values.shape)
    summary['maximum']={'azimuth_deg':float(az[location[1]]),'altitude_deg':float(alt[location[0]]),'value':float(values[location])}
    for target in (5,15,45):
        row=int(np.flatnonzero(alt==target)[0])
        summary[f'south_over_north_at_{target}deg']=float(values[row,np.argmin(abs(az-180))]/values[row,0])
    np.savez_compressed(directory/'sky-grid.npz',azimuth_deg=az,altitude_deg=alt,radiance=values,metadata_json=json.dumps(summary,ensure_ascii=False))
    aa,hh=np.meshgrid(az,alt)
    np.savetxt(directory/'sky-grid.csv',np.column_stack((aa.ravel(),hh.ravel(),values.ravel())),delimiter=',',
               header='azimuth_deg,altitude_deg,radiance',comments='',fmt='%.12g')
    (directory/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    sky_plot(az,alt,values,directory/'sky.png','Петербург: реальное распределение источников, уточнены низкие лучи')
    print('Refinement finished',duration,'seconds',flush=True)


if __name__=='__main__':main()
