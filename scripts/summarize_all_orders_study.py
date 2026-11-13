"""Reproduce the all-order comparison tables, plots and integrity checks."""

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from lightpollution.adapters.storage import file_hash, read_json, write_json
from lightpollution.models.slab_resolvent import SlabSettings, slab_system

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT/'artifacts/all-orders-study-20261113')
    args=parser.parse_args();base=args.root
    protocol=read_json(base/'protocol.json');summary=read_json(base/'summary.json')
    rows=[read_json(base/f'spherical/aod-{tau:.2f}-alt-{alt}.json') for tau in [.02,.1,.3] for alt in [10,45,90]]
    checks=[]
    for name,expected in protocol['source_sha256'].items():
        checks.append({'kind':'code_hash','path':name,'passed':file_hash(ROOT/name)==expected})
    old=ROOT/'artifacts/benchmark-baseline-20261113/scattering'
    checks.append({'kind':'old_protocol_hash','passed':file_hash(old/'protocol.json')==protocol['old_protocol_sha256']})
    for row in rows:
        original=read_json(old/f"{row['case_id']}.json")
        same=np.allclose(row['category_mean_radiance'][:3],original['exact_order_radiance'],rtol=2e-13,atol=0)
        checks.append({'kind':'old_three_order_prefix','case':row['case_id'],'passed':bool(same)})
        cov=np.array(row['mean_covariance'])
        checks.append({'kind':'bounded_moments_accounting','case':row['case_id'],
            'passed':bool(sum(row['path_outcomes'].values())==row['settings']['samples']
                and np.linalg.eigvalsh(cov).min()>-1e-18 and np.allclose(np.cumsum(row['category_mean_radiance']),row['cumulative_radiance'])
                and row['finite_scattering_order_truncation'] is False)})
        if row['altitude_deg']==45 and row['atmosphere']['aerosol_optical_depth'] in [.1,.3]:
            original=read_json(old/f"tail-aod-{row['atmosphere']['aerosol_optical_depth']:.2f}-alt-45.json")
            checks.append({'kind':'old_five_order_prefix','case':row['case_id'],
                'passed':bool(np.allclose(row['category_mean_radiance'][:5],original['exact_order_radiance'],rtol=2e-13,atol=0))})
    for path in sorted((base/'slab').glob('*.json')):
        row=read_json(path);r=row['references'][-1]
        checks.append({'kind':'full_slab_reference','case':path.stem,
            'passed':bool(abs(row['full_solution_z'])<4 and row['reference_refinement_relative']<.005
                and r['linear_vs_series_difference']<=r['verification_series_tail_bound']+1e-13)})
    checks.append({'kind':'four_angular_characteristic_moments','passed':summary['angular_characteristic_checks_passed']})
    checks.append({'kind':'independent_seed_and_absorption','passed':summary['max_abs_independent_absorption_seed_z']<4})
    write_json(base/'integrity-verification.json',{'checks':checks,'total':len(checks),
        'passed':sum(r['passed'] for r in checks),'failed':sum(not r['passed'] for r in checks)})
    if not all(r['passed'] for r in checks):
        raise RuntimeError('Do not publish comparison plots with failed integrity checks.')
    with (base/'comparison-table.csv').open('w',newline='') as stream:
        writer=csv.writer(stream,lineterminator='\n')
        writer.writerow(['case_id','samples','partial_order','all_order_radiance','all_order_standard_error',
            'partial_radiance','missing_fraction_of_all','missing_fraction_standard_error',
            'normal_95_eligible','tail_luminous_paths'])
        for row in rows:
            for n in [1,2,3,5]:
                c=row['comparisons_to_partial_sums'][str(n)]
                writer.writerow([row['case_id'],row['settings']['samples'],n,row['all_order_radiance'],row['all_order_standard_error'],
                    row['cumulative_radiance'][n-1],c['missing_fraction_of_full'],c['missing_fraction_standard_error'],
                    c['normal_intervals_eligible'],c['tail_luminous_paths']])
    plt.rcParams.update({'font.size':11,'axes.grid':True,'grid.alpha':.25})
    fig,axes=plt.subplots(1,3,figsize=(12,4.1),constrained_layout=True)
    for ax,tau in zip(axes,[.02,.1,.3],strict=True):
        for alt,color in zip([10,45,90],['#126782','#d18a14','#aa4242'],strict=True):
            row=next(r for r in rows if r['altitude_deg']==alt and r['atmosphere']['aerosol_optical_depth']==tau)
            contrasts=[row['comparisons_to_partial_sums'][str(n)] for n in [1,2,3,5]]
            x=[1,2,3,5];y=[c['missing_fraction_of_full']*100 for c in contrasts]
            ax.plot(x,y,color=color,alpha=.7,label=f'{alt}°')
            for n,c in zip(x,contrasts,strict=True):
                value=c['missing_fraction_of_full']*100
                if c['normal_intervals_eligible']:
                    ax.errorbar(n,value,yerr=196*c['missing_fraction_standard_error'],fmt='o',color=color,capsize=3)
                else:
                    ax.plot(n,value,'o',color=color,markerfacecolor='white')
        ax.set_title(f'AOD={tau:.2f}');ax.set_xlabel('Число оставленных порядков');ax.set_xticks(x);ax.legend(title='Высота')
    axes[0].set_ylabel('Недосчёт относительно полной оценки, %')
    fig.suptitle('Та же сфера и источник; пустые маркеры — редкий хвост без нормального интервала')
    fig.savefig(base/'truncation-vs-all.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10.5,4),constrained_layout=True)
    for tau,color in zip([.02,.1,.3],['#126782','#d18a14','#aa4242'],strict=True):
        row=next(r for r in rows if r['altitude_deg']==45 and r['atmosphere']['aerosol_optical_depth']==tau)
        hist=row['luminous_order_histogram'];orders=np.arange(1,max(map(int,hist))+1)
        contribution=[hist[str(n)]['mean_radiance']/row['all_order_radiance'] if str(n) in hist else np.nan for n in orders]
        ax.plot(orders,contribution,'-',color=color,label=f'AOD={tau:.2f}')
        for n,value in zip(orders,contribution,strict=True):
            if np.isfinite(value):
                rare=hist[str(n)]['luminous_paths']<100
                ax.plot(n,value,'o',color=color,markersize=4,markerfacecolor='white' if rare else color)
    ax.axvline(5.5,color='black',linestyle='--',linewidth=1)
    ax.set_yscale('log');ax.set_xlabel('Точное число рассеяний');ax.set_ylabel('Вклад порядка / полный фон')
    ax.set_title('Взгляд 45°; пустые маркеры: менее 100 попаданий в данный порядок');ax.legend()
    fig.savefig(base/'order-contributions.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10.5,4),constrained_layout=True)
    for path in sorted((base/'slab').glob('*.json')):
        r=read_json(path);reference=r['references'][-1]
        matrix,forcing,escape=slab_system(SlabSettings(**reference['settings']))
        term=forcing.copy();field=np.zeros_like(term);curve=[]
        for n in range(1,reference['verification_series_iterations']+1):
            field+=term;term=matrix @ term
            missing=1-float(escape @ field)/reference['all_order_radiance']
            if missing>1e-13:curve.append((n,missing*100))
        ax.plot([v[0] for v in curve],[v[1] for v in curve],label=f"τ={r['depth']}; ω={r['albedo']}")
    ax.set_xscale('log');ax.set_yscale('log');ax.axhline(1,color='grey',linestyle='--',linewidth=1)
    ax.set_xlabel('Число членов ряда');ax.set_ylabel('Недосчёт к решению системы, %')
    ax.set_title('Независимый слой: линейная система суммирует весь ряд');ax.legend()
    fig.savefig(base/'slab-series-convergence.png',dpi=160);plt.close(fig)
    ledger={'study_MC_paths':summary['all_MC_paths_excluding_pilots_and_unit_tests'],
        'main_all_order_calls':9,'precision_all_order_calls':2,'independent_all_order_calls':3,
        'slab_all_order_calls':3,'timing_all_order_calls':2,'timing_finite_order_calls':8,
        'pilot_all_order_calls':9,'pilot_paths':9*131072,
        'all_study_MC_paths_including_pilot_excluding_unit_tests':summary['all_MC_paths_excluding_pilots_and_unit_tests']+9*131072,
        'angular_random_rotation_histories':262144,'angular_histories_not_counted_as_radiative_paths':True,
        'previous_benchmark_paths_not_added_to_this_study':True}
    write_json(base/'execution-ledger.json',ledger)
    write_json(base/'plots-provenance.json',{'summarizer_sha256':file_hash(Path(__file__)),
        'source_protocol_sha256':file_hash(base/'protocol.json'),
        'generated_at':datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'kind':'Scientific plots from numeric JSON, not eye/camera photographs.'})
    print({'integrity_checks':len(checks),'failed':0,'study_MC_paths':ledger['study_MC_paths']})


if __name__=='__main__':
    main()
