"""Independent numerical-error and scenario checks on specified test sources.

Run from the repository root: uv run python scripts/run_experiments.py
These numbers quantify discretisation and scenario dependence, not agreement
with observed night-sky brightness.
"""

from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from lightpollution.physics import Atmosphere, Observer, SolverSettings, integrate_direction
from lightpollution.sources import synthetic_cities

DIRECTIONS=[(0,5),(90,1),(90,5),(180,1),(180,5),(180,15),(270,5),(0,45),(180,45),(0,90)]
OBSERVER=Observer()
SOURCES=synthetic_cities(OBSERVER)
ATMOSPHERE=Atmosphere()


def evaluate(settings, atmosphere=ATMOSPHERE):
    started=time.perf_counter()
    values=np.array([integrate_direction(OBSERVER,SOURCES,az,alt,atmosphere,settings) for az,alt in DIRECTIONS])
    return values,time.perf_counter()-started


def difference(values,reference):
    relative=np.abs(values/reference-1)
    return {'maximum_relative_difference':float(relative.max()),
            'median_relative_difference':float(np.median(relative)),
            'directions_denominator':len(reference), 'values':values.tolist()}


def main():
    root=Path('artifacts/numerical-validation')
    root.mkdir(parents=True,exist_ok=True)
    reference_settings=SolverSettings(ray_segments=768,path_quadrature=64)
    reference,elapsed=evaluate(reference_settings)
    report={'input_kind':'synthetic','actual_nasa_pixels':0,
            'created_at':datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'observer':asdict(OBSERVER),'atmosphere':asdict(ATMOSPHERE),
            'source_count':len(SOURCES.power),'source_total_power_W':float(SOURCES.power.sum()),
            'directions_azimuth_altitude_deg':DIRECTIONS,
            'reference':{'settings':asdict(reference_settings),'elapsed_seconds':elapsed,'values':reference.tolist()},
            'ray_convergence':[], 'path_convergence':[], 'sensitivity':[],
            'limitation':'Numerical comparisons against the same solver are not independent physical validation.'}
    for segments in (48,96,192,384):
        settings=replace(reference_settings,ray_segments=segments)
        values,elapsed=evaluate(settings)
        report['ray_convergence'].append({'ray_segments':segments,'elapsed_seconds':elapsed,**difference(values,reference)})
        print('ray segments',segments,'max difference',report['ray_convergence'][-1]['maximum_relative_difference'],flush=True)
    for order in (4,8,12,24,48):
        settings=replace(reference_settings,path_quadrature=order)
        values,elapsed=evaluate(settings)
        report['path_convergence'].append({'path_quadrature':order,'elapsed_seconds':elapsed,**difference(values,reference)})
        print('path order',order,'max difference',report['path_convergence'][-1]['maximum_relative_difference'],flush=True)
    settings=SolverSettings(ray_segments=384,path_quadrature=48)
    baseline,_=evaluate(settings)
    for label,atmosphere,setting in [
        ('aod_0.03',replace(ATMOSPHERE,aerosol_optical_depth=0.03),settings),
        ('aod_0.30',replace(ATMOSPHERE,aerosol_optical_depth=0.30),settings),
        ('g_0.6',replace(ATMOSPHERE,aerosol_g=0.6),settings),
        ('horizontal_fraction_0.1',ATMOSPHERE,replace(settings,horizontal_fraction=0.1)),
        ('horizontal_fraction_0.3',ATMOSPHERE,replace(settings,horizontal_fraction=0.3)),
    ]:
        values,elapsed=evaluate(setting,atmosphere)
        ratios=values/baseline
        report['sensitivity'].append({'scenario':label,'atmosphere':asdict(atmosphere),'settings':asdict(setting),
                                      'elapsed_seconds':elapsed,'ratio_to_baseline':ratios.tolist(),
                                      'minimum_ratio':float(ratios.min()),'maximum_ratio':float(ratios.max()),
                                      'directions_denominator':len(DIRECTIONS)})
        print('scenario',label,'ratio range',float(ratios.min()),float(ratios.max()),flush=True)
    (root/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for ax,key,x_key,title in ((axes[0],'ray_convergence','ray_segments','Шаг вдоль луча'),
                                (axes[1],'path_convergence','path_quadrature','Квадратура пропускания')):
        rows=report[key]
        ax.loglog([r[x_key] for r in rows],[100*r['maximum_relative_difference'] for r in rows],'o-',label='Максимум по 10 направлениям')
        ax.set(xlabel='Число сегментов / узлов',ylabel='Отличие от опорного расчёта, %',title=title)
        ax.grid(alpha=0.3,which='both')
        ax.legend(fontsize=8)
    fig.suptitle('Численная сходимость на заданных источниках; не ошибка относительно реального неба')
    fig.savefig(root/'convergence.png',dpi=160)
    plt.close(fig)


if __name__=='__main__':
    main()
