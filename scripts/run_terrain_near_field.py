"""Reproduce finite-source and real DSM checks without re-fitting source power."""

import argparse
from datetime import datetime
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from lightpollution.adapters.sources import LocalSourceRepository
from lightpollution.adapters.storage import write_json
from lightpollution.adapters.terrain import load_terrain
from lightpollution.data import Bounds
from lightpollution.models.topographic_transfer import TopographicSolver
from lightpollution.physics import Atmosphere, EARTH_RADIUS_M, Observer, SolverSettings, emission, integrate_direction
from lightpollution.sources import regional_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('artifacts/kstars-terrain-near-20261111'))
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    criteria = out/'predeclared-criteria.json'
    if not criteria.exists():
        write_json(criteria, {'declared_at': datetime.now(ZoneInfo('Europe/Berlin')).isoformat(timespec='seconds'),
            'finite_cell_analytic_relative_tolerance': .002,
            'power_conservation_absolute_tolerance': 1e-12,
            'selected_ray_convergence_target_relative': .01,
            'convergence_is_not_field_accuracy': True,
            'source_power_policy': 'Compare raw common-normalisation transfer; never re-fit C between model variants.',
            'independent_terrain_or_local_lamp_nights': 0})
    # Exact planar Lambertian rectangular emitter limit, approximated here
    # by a small spherical cell with the same physical area.
    bounds = Bounds(30.3, 60.3, 30.3+1/240, 60.3+1/240)
    arrays = {'NearNadir_Composite_Snow_Free': np.ones(bounds.shape),
        'NearNadir_Composite_Snow_Free_Quality': np.zeros(bounds.shape),
        'NearNadir_Composite_Snow_Free_Num': np.full(bounds.shape, 10),
        'Land_Water_Mask': np.ones(bounds.shape)}
    cell_rows = []
    for height in [2., 20.]:
        observer = Observer(float(bounds.latitude[0]), float(bounds.longitude[0]), height)
        a = np.deg2rad(1/480)*EARTH_RADIUS_M*np.cos(np.deg2rad(observer.latitude_deg))
        b = np.deg2rad(1/480)*EARTH_RADIUS_M
        omega = 4*np.arctan(a*b/(height*np.sqrt(height*height+a*a+b*b)))
        expected = omega/(np.pi*4*a*b)
        for patch, order in [(10.,2),(5.,2),(2.,3)]:
            sources, _ = regional_sources(arrays,bounds,observer,exclude_radius_m=0,
                finite_cells={'radius_m':3000.,'minimum_patch_m':patch,'quadrature_order':order})
            delta = observer.position-sources.positions_m
            distance = np.linalg.norm(delta,axis=1)
            mu = np.sum(delta/distance[:,None]*(sources.positions_m/EARTH_RADIUS_M),axis=1)
            actual = float(np.sum(sources.power*emission(mu)/distance**2))
            cell_rows.append({'height_agl_m':height,'minimum_patch_m':patch,'quadrature_order':order,
                'quadrature_emitters':len(sources.power),'power_sum':float(sources.power.sum()),
                'analytic_planar_response':float(expected),'computed_response':actual,
                'relative_error':abs(actual/expected-1)})
    write_json(out/'finite-cell-convergence.json',cell_rows)
    repository = LocalSourceRepository(Path('data/terrain-near-checks'), Path('data/blackmarble'))
    settings = SolverSettings(96,12)
    cases = [
        ('petersburg-centre','petersburg-2025','2025','petersburg-copdem90',59.95,30.3),
        ('petersburg-north','petersburg-2025','2025','petersburg-copdem90',60.3,30.315),
        ('madrid-ucm','madrid-monthly','2023-03','madrid-copdem90',40.451111,-3.726111)]
    # Read the catalogue, not a guessed monthly identifier.
    madrid = next(e for e in repository.catalogue() if 'madrid' in e['id'])
    cases[-1] = (*cases[-1][:1], madrid['id'], madrid['reference_period'], *cases[-1][3:])
    rows = []
    for name, region, period, terrain_id, latitude, longitude in cases:
        start = perf_counter()
        terrain = load_terrain(terrain_id,Path('data/terrain'))
        write_json(out/(terrain_id+'-manifest.json'),terrain.metadata)
        observer = Observer(latitude,longitude,2.)
        physical = terrain.observer(observer)
        prepared = repository.prepare(region,period,observer)
        old, old_qa = prepared.select(period,24)
        config = {'radius_m':3000.,'minimum_patch_m':10.,'quadrature_order':2,'observer_agl_m':2.}
        near, near_qa = prepared.select(period,24,finite_cells=config)
        # A fixed common source basis is retained in all three calculations.
        distant = integrate_direction(observer,old,0,90,Atmosphere(),settings)
        flat = integrate_direction(observer,near,0,90,Atmosphere(),settings)
        elevated = terrain.lift_sources(near)
        topographic = TopographicSolver(terrain,100).direction(physical,elevated,0,90,Atmosphere(),settings)
        refined = TopographicSolver(terrain,50).direction(physical,elevated,0,90,Atmosphere(),settings)
        detailed_config = {**config,'minimum_patch_m':2.,'quadrature_order':3}
        detailed, detailed_qa = prepared.select(period,24,finite_cells=detailed_config)
        detailed_elevated = terrain.lift_sources(detailed)
        final = TopographicSolver(terrain,100).direction(physical,detailed_elevated,0,90,Atmosphere(),settings)
        final_step50 = TopographicSolver(terrain,50).direction(physical,detailed_elevated,0,90,Atmosphere(),settings)
        final_standard = (TopographicSolver(terrain,50).direction(physical,detailed_elevated,0,90,
            Atmosphere(),SolverSettings(192,24)) if name=='petersburg-centre' else None)
        azimuths = np.arange(0.,360.,2.)
        horizon = terrain.horizon(physical,azimuths,30000.,50.)
        row = {'name':name,'region':region,'period':period,'source_basis_id':prepared.identity,
            'observer_agl':observer.__dict__,'observer_msl':physical.__dict__,
            'terrain_id':terrain_id,'DSM_height_m':physical.height_m-2,
            'old_effective_emitters':len(old.power),'near_effective_emitters':len(near.power),
            'finite_emitting_pixels':near_qa['finite_emitting_pixels'],
            'new_default_effective_emitters':len(detailed.power),'coarse_near_settings':config,
            'new_default_near_settings':detailed_config,
            'old_selected_power':float(old.power.sum()),'near_selected_power':float(near.power.sum()),
            'raw_zenith_transfer':{'old_3km_exclusion':float(distant),'finite_cells_flat':float(flat),
                'finite_cells_DSM_step100m':float(topographic),'finite_cells_DSM_step50m':float(refined)},
            'new_default_raw_transfer':{'DSM_step100m':float(final),'DSM_step50m':float(final_step50),
                'DSM_step50m_standard_192_24':float(final_standard) if final_standard is not None else None},
            'coarse_vs_new_default_relative_change':abs(topographic/final-1),
            'new_default_step100_vs50_relative_change':abs(final/final_step50-1),
            'quick_vs_standard_selected_ray_relative_change':abs(final_step50/final_standard-1) if final_standard else None,
            'step100_vs50_relative_change':abs(topographic/refined-1) if refined else None,
            'near_flat_over_old':float(flat/distant), 'DSM_over_near_flat':float(topographic/flat),
            'horizon_azimuth_deg':azimuths.tolist(),'horizon_altitude_deg':horizon.tolist(),
            'seconds':perf_counter()-start,'field_validated':False}
        rows.append(row)
        write_json(out/'real-DSM-transfer.json',rows)
        print(name, row['raw_zenith_transfer'], 'seconds',round(row['seconds'],2),flush=True)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for height in [2.,20.]:
        group=[r for r in cell_rows if r['height_agl_m']==height]
        axes[0].plot([r['minimum_patch_m'] for r in group],[100*r['relative_error'] for r in group],'o-',label=f'AGL {height:g} m')
    axes[0].set(xlabel='Minimum patch (m)',ylabel='Difference from planar analytic limit (%)',yscale='log')
    axes[0].legend();axes[0].grid(alpha=.3)
    for row in rows:
        axes[1].plot(row['horizon_azimuth_deg'],row['horizon_altitude_deg'],label=row['name'])
    axes[1].set(xlabel='Azimuth (degrees, N=0)',ylabel='DSM horizon (degrees)',title='30 km profile; 90 m DSM, no individual lamps')
    axes[1].legend();axes[1].grid(alpha=.3)
    fig.savefig(out/'numerical-checks.png',dpi=170)
    plt.close(fig)
    write_json(out/'summary.json',{'completed_at':datetime.now(ZoneInfo('Europe/Berlin')).isoformat(timespec='seconds'),
        'analytic_cell_cases':len(cell_rows),'cases':len(rows),'cell_checks':cell_rows,
        'real_site_summaries':[{k:v for k,v in r.items() if not k.startswith('horizon_')} for r in rows],
        'independent_clear_sky_terrain_or_near_lamp_nights':0,'field_accuracy':None,
        'note':'Numerical convergence is not empirical accuracy. Raw transfer has common unit-source weights, no auto-refit between variants.'})


if __name__=='__main__':
    main()
