"""Freeze and run the all-order RTE study against the existing nine scenes."""

import argparse
from dataclasses import asdict, replace
from datetime import datetime
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from zoneinfo import ZoneInfo

import numpy as np

from lightpollution.adapters.storage import file_hash, read_json, write_json
from lightpollution.models.all_orders import AllOrderSettings, integrate_all_orders, relative_tail
from lightpollution.models.scattering_orders import LambertianCap, OrderSettings, integrate_orders, sample_phase
from lightpollution.models.slab_resolvent import SlabSettings, solve_slab
from lightpollution.physics import EARTH_RADIUS_M, Atmosphere, Observer

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'artifacts/benchmark-baseline-20261113/scattering'


def scene():
    old = read_json(OLD/'aod-0.10-alt-45.json')
    return Observer(**old['observer']), LambertianCap(**old['cap'])


def with_contrasts(result):
    result['comparisons_to_partial_sums'] = {str(n): relative_tail(result, n) for n in [1, 2, 3, 5]}
    return result


def characteristic_example(samples=262144):
    """An EXACT angular-only special case, not a substitute for atmospheric geometry.

    Number of independent phase rotations N~Geom(1-q) on {0,1,...}.
    HG Legendre eigenvalue a_l=g**l => final E[P_l]=(1-q)/(1-q*a_l).
    Here constant continuation q deliberately ignores position and boundaries.
    """
    q, g = .8, .8
    rng = np.random.default_rng(45671)
    counts = rng.geometric(1-q, samples)-1
    directions = np.broadcast_to([0., 0., 1.], (samples, 3)).copy()
    for order in range(int(max(counts))):
        active = counts > order
        directions[active] = sample_phase(directions[active], 0., g, rng)
    rows = []
    for degree in range(1, 5):
        c = np.zeros(degree+1); c[-1] = 1
        values = np.polynomial.legendre.legval(directions[:, 2], c)
        expected = (1-q)/(1-q*g**degree)
        se = float(np.std(values, ddof=1)/np.sqrt(samples))
        rows.append({'degree': degree, 'analytic_moment': expected, 'sample_mean': float(np.mean(values)),
            'standard_error': se, 'z': float((np.mean(values)-expected)/se)})
    return {'kind': 'Angular convolution / geometric random sum special case.', 'q': q, 'HG_g': g,
        'samples': samples, 'moments': rows, 'all_four_moments_within_four_standard_errors': all(abs(r['z'])<4 for r in rows),
        'constant_q_is_not_an_atmospheric_retrieval': True, 'predicts_city_brightness': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/all-orders-study-20261113')
    parser.add_argument('--samples', type=int, default=2097152)
    parser.add_argument('--worker', choices=['1', '2', '3', '5', 'all'])
    parser.add_argument('--seed', type=int, default=3011)
    args = parser.parse_args()
    observer, cap = scene()
    if args.worker:
        if args.worker == 'all':
            r = integrate_all_orders(observer, cap, 180., 45., Atmosphere(), AllOrderSettings(samples=args.samples, seed=args.seed))
        else:
            r = integrate_orders(observer, cap, 180., 45., Atmosphere(),
                OrderSettings(samples=args.samples, max_order=int(args.worker), seed=args.seed))
        status = Path('/proc/self/status')
        r['peak_rss_kib'] = int(next(line.split()[1] for line in status.read_text().splitlines() if line.startswith('VmHWM:'))) if status.exists() else None
        r['worker'] = args.worker
        print(json.dumps(r))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    for folder in ['spherical', 'slab', 'checks']:
        (args.output/folder).mkdir(exist_ok=True)
    protocol = {'study_version': 'LP-all-orders-1',
        'created_at': datetime.now(ZoneInfo('Europe/Berlin')).isoformat(timespec='seconds'),
        'git_base_commit': 'c549dbd', 'observer': asdict(observer), 'cap': asdict(cap),
        'base_atmosphere': asdict(Atmosphere()), 'samples_per_main_case': args.samples, 'seed': args.seed,
        'old_protocol_sha256': file_hash(OLD/'protocol.json'),
        'aod_values': [.02, .10, .30], 'altitudes_deg': [10., 45., 90.], 'azimuth_deg': 180.,
        'same_geometry_power_and_other_models_as_previous_study': True,
        'new_field_fit_or_C_refit': False, 'natural_Moon_cloud_snow_DSM_and_spectrum_disabled': True,
        'no_finite_order_cutoff': True,
        'precision_checks': {'cases': [[.02,45], [.02,90]], 'samples': 8388608, 'seed': 9031,
            'declared_before_main_results': True, 'reason': 'Rare high-order tails in independent pilot, not selection by residual.'},
        'independent_checks': {'implicit_repeat': {'aod': .1, 'altitude':45, 'seed':3012},
            'analog_absorption': {'aod_values':[.1,.3], 'altitude':45, 'seed':6041},
            'slabs_depth_albedo': [[.2,.9], [1.,1.], [5.,.99]], 'slab_MC_samples':524288,
            'slab_grids': [[256,64], [512,128], [1024,192]]},
        'predeclared_checks': {'old_order_prefix_rtol': 2e-13, 'MC_reference_max_abs_z':4,
            'slab_grid_refinement_relative_max':.005, 'minimum_tail_hits_for_normal_interval':100},
        'timing': {'samples':524288, 'orders':['1','2','3','5','all'], 'seeds':[701,702],
            'same_samples_not_same_error':True, 'memory':'/proc/self/status VmHWM',
            'hardware':platform.platform(), 'logical_CPUs':os.cpu_count(),
            'OPENBLAS_NUM_THREADS':os.environ.get('OPENBLAS_NUM_THREADS'), 'OMP_NUM_THREADS':os.environ.get('OMP_NUM_THREADS')},
        'source_sha256': {p:file_hash(ROOT/p) for p in ['src/lightpollution/physics.py',
            'src/lightpollution/models/scattering_orders.py','src/lightpollution/models/all_orders.py',
            'src/lightpollution/models/slab_resolvent.py','scripts/run_all_orders_study.py']}}
    write_json(args.output/'protocol.json', protocol)
    main_rows = []
    for tau in protocol['aod_values']:
        for alt in protocol['altitudes_deg']:
            label = f'aod-{tau:.2f}-alt-{int(alt)}'
            old = read_json(OLD/f'{label}.json')
            r = with_contrasts(integrate_all_orders(observer,cap,180.,alt,
                replace(Atmosphere(),aerosol_optical_depth=tau),AllOrderSettings(samples=args.samples,seed=args.seed)))
            prefix = np.array(r['category_mean_radiance'][:3])
            r['old_case_sha256'] = file_hash(OLD/f'{label}.json')
            r['old_first_three_exact_orders_reproduced'] = bool(np.allclose(prefix,old['exact_order_radiance'],rtol=2e-13,atol=0))
            r['case_id'] = label
            write_json(args.output/'spherical'/f'{label}.json',r)
            main_rows.append(r)
            print(json.dumps({'case':label,'same_old_prefix':r['old_first_three_exact_orders_reproduced'],
                'full_over_S3_minus1':r['comparisons_to_partial_sums']['3']['increment_relative_to_partial'],
                'full_over_S5_minus1':r['comparisons_to_partial_sums']['5']['increment_relative_to_partial'],
                'max_order_seen':r['max_scattering_order_seen'],'hits':r['luminous_paths_by_category'],
                'seconds':r['runtime_seconds']}),flush=True)
    extras = []
    for tau,alt in [[.02,45.],[.02,90.]]:
        r = with_contrasts(integrate_all_orders(observer,cap,180.,alt,
            replace(Atmosphere(),aerosol_optical_depth=tau),AllOrderSettings(samples=8388608,seed=9031)))
        label=f'precision-aod-{tau:.2f}-alt-{int(alt)}'
        write_json(args.output/'checks'/f'{label}.json',r)
        extras.append(r)
        print(json.dumps({'precision':label,'S3_tail':r['comparisons_to_partial_sums']['3']}),flush=True)
    for tau,mode,seed,label in [(.1,'implicit',3012,'independent-seed'),(.1,'analog',6041,'analog-aod-0.10'),(.3,'analog',6041,'analog-aod-0.30')]:
        r=with_contrasts(integrate_all_orders(observer,cap,180.,45.,replace(Atmosphere(),aerosol_optical_depth=tau),
            AllOrderSettings(samples=args.samples,seed=seed,absorption=mode)))
        baseline=next(v for v in main_rows if v['atmosphere']['aerosol_optical_depth']==tau and v['altitude_deg']==45.)
        r['independent_comparison_z']=(r['all_order_radiance']-baseline['all_order_radiance'])/np.hypot(r['all_order_standard_error'],baseline['all_order_standard_error'])
        write_json(args.output/'checks'/f'{label}.json',r)
        extras.append(r)
        print(json.dumps({'check':label,'z':r['independent_comparison_z']}),flush=True)
    slab_rows=[]
    for depth,albedo in protocol['independent_checks']['slabs_depth_albedo']:
        refs=[solve_slab(SlabSettings(layers=n,angular_order=q,optical_depth=depth,single_scattering_albedo=albedo))
            for n,q in protocol['independent_checks']['slab_grids']]
        atm=Atmosphere(rayleigh_multiplier=0,aerosol_scale_height_m=1e6,aerosol_optical_depth=depth*1e6,
            aerosol_single_scattering_albedo=albedo,aerosol_g=0,top_m=1.)
        mc=with_contrasts(integrate_all_orders(Observer(0.,0.,0.),LambertianCap(0.,0.,np.pi*EARTH_RADIUS_M),
            0.,90.,atm,AllOrderSettings(samples=524288,seed=9901)))
        z=(mc['all_order_radiance']-refs[-1]['all_order_radiance'])/mc['all_order_standard_error']
        record={'depth':depth,'albedo':albedo,'references':refs,'spherical_MC':mc,'full_solution_z':float(z),
            'reference_refinement_relative':abs(refs[-2]['all_order_radiance']/refs[-1]['all_order_radiance']-1),
            'plane_and_spherical_geometry_identical':False}
        if depth==1 and albedo==1:
            old=read_json(OLD/'slab-reference.json')
            record['old_five_orders_reproduced']=bool(np.allclose(mc['category_mean_radiance'][:5],old['spherical_MC']['exact_order_radiance'],rtol=2e-13,atol=0))
        write_json(args.output/'slab'/f'depth-{depth}-albedo-{albedo}.json',record)
        slab_rows.append(record)
        print(json.dumps({'slab':depth,'z':z,'refinement':record['reference_refinement_relative']}),flush=True)
    angular=characteristic_example()
    write_json(args.output/'checks/characteristic-angular-example.json',angular)
    timing=[]
    for seed in [701,702]:
        for order in protocol['timing']['orders']:
            process=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',order,'--samples','524288','--seed',str(seed)],
                capture_output=True,text=True,check=True)
            r=json.loads(process.stdout);timing.append(r)
            print(json.dumps({'timing_order':order,'seed':seed,'seconds':r['runtime_seconds'],'rss':r['peak_rss_kib']}),flush=True)
    write_json(args.output/'timing.json',{'runs':timing,'settings':protocol['timing']})
    summary={'version':'LP-all-orders-1','main_cases':len(main_rows),
        'all_nine_old_order_prefixes_reproduced':all(r['old_first_three_exact_orders_reproduced'] for r in main_rows),
        'slab_full_checks_passed':sum(abs(r['full_solution_z'])<4 for r in slab_rows),'slab_full_checks_total':len(slab_rows),
        'max_abs_slab_full_z':max(abs(r['full_solution_z']) for r in slab_rows),
        'max_slab_grid_refinement_relative':max(r['reference_refinement_relative'] for r in slab_rows),
        'max_abs_independent_absorption_seed_z':max(abs(r.get('independent_comparison_z',0)) for r in extras),
        'angular_characteristic_checks_passed':angular['all_four_moments_within_four_standard_errors'],
        'main_all_order_paths':9*args.samples,'precision_and_independent_paths':2*8388608+3*args.samples,
        'slab_all_order_paths':3*524288,'timing_paths':10*524288,
        'all_MC_paths_excluding_pilots_and_unit_tests':12*args.samples+2*8388608+13*524288,
        'full_over_S3_minus_one_range':[min(r['comparisons_to_partial_sums']['3']['increment_relative_to_partial'] for r in main_rows),
            max(r['comparisons_to_partial_sums']['3']['increment_relative_to_partial'] for r in main_rows)],
        'max_scattering_order_seen_in_main':max(r['max_scattering_order_seen'] for r in main_rows),
        'finite_order_cutoff_used':False,'new_field_accuracy_measured':False,'production_KStars_changed':False}
    write_json(args.output/'summary.json',summary)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
