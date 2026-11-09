"""Separate numerical convergence from physical accuracy on the real source map."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from lightpollution.data import load_region
from lightpollution.physics import Atmosphere, Observer, SolverSettings, integrate_direction
from lightpollution.sources import regional_sources


DIRECTIONS=[(0,1),(90,1),(180,1),(270,1),(180,5),(180,15),(0,45),(0,90)]


def calculate(config):
    block,segments,quadrature=config
    arrays,bounds,_=load_region(Path('artifacts/petersburg-2025/source-grid.npz'))
    observer=Observer();sources,_=regional_sources(arrays,bounds,observer,block=block)
    settings=SolverSettings(segments,quadrature)
    values=[]
    counts=[]
    for az,alt in DIRECTIONS:
        selected,_=regional_sources(arrays,bounds,observer,block=block,
                                    refinement_direction=(az,alt)) if alt<=5 else (sources,{})
        counts.append(len(selected.power))
        values.append(integrate_direction(observer,selected,az,alt,Atmosphere(),settings))
    return {'block':block,'effective_sources':len(sources.power),'sources_per_ray':counts,
            'settings':asdict(settings),'radiance':values}


def main():
    with ProcessPoolExecutor(max_workers=3) as executor:
        runs=list(executor.map(calculate,[(8,192,24),(8,384,48),(4,384,48)]))
    base,ref,smaller=[np.asarray(r['radiance']) for r in runs]
    result={'directions_az_alt_deg':DIRECTIONS,'runs':runs,
            'quadrature_relative_difference':(base/ref-1).tolist(),
            'aggregation_relative_difference':(ref/smaller-1).tolist(),
            'max_abs_quadrature_relative_difference':float(np.max(np.abs(base/ref-1))),
            'max_abs_aggregation_relative_difference':float(np.max(np.abs(ref/smaller-1))),
            'physical_accuracy_bound':None,
            'scope':'Eight selected rays only, not a bound on the whole hemisphere; no ground-truth validation.'}
    output=Path('artifacts/petersburg-2025/observer-sky/numerical-check.json')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
