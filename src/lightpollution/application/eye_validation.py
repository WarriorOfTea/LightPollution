"""Audit visual observations without fitting the sky or leaking held-out nights."""

from collections import Counter,defaultdict

import numpy as np

from ..adapters.storage import content_key


def _identifier(value,name):
    if not isinstance(value,str) or not value.strip() or len(value)>160:
        raise ValueError(f'{name} must be nonempty text up to 160 characters.')
    return value


def _score(rows,factor):
    counts=Counter()
    for row in rows:
        predicted=row['base_signal']/factor>=1
        counts[('T' if predicted==row['seen'] else 'F')+('P' if predicted else 'N')]+=1
    tp,tn,fp,fn=(counts[k] for k in ['TP','TN','FP','FN'])
    positive,negative=tp+fn,tn+fp
    return {'records':len(rows),'seen':positive,'not_seen':negative,
        'TP':tp,'TN':tn,'FP':fp,'FN':fn,
        'accuracy':(tp+tn)/len(rows) if rows else None,
        'sensitivity':tp/positive if positive else None,
        'specificity':tn/negative if negative else None,
        'balanced_accuracy':.5*(tp/positive+tn/negative) if positive and negative else None,
        'nights':len({r['night_id'] for r in rows})}


def _fit(rows):
    if not rows or {r['seen'] for r in rows}!={False,True}:return None
    # Thresholds change only at base_signal. Include equality and open intervals.
    boundaries=np.unique(np.r_[.5,10.,[r['base_signal'] for r in rows if .5<=r['base_signal']<=10]])
    candidates=np.unique(np.r_[boundaries,np.sqrt(boundaries[:-1]*boundaries[1:]),2.])
    nominal=float(np.median([r['original_factor'] for r in rows]))
    return float(max(candidates,key=lambda f:(_score(rows,f)['balanced_accuracy'],-abs(np.log(f/nominal)))))


def audit_observations(document,results):
    """Validated schema-1 observations; results are already checksum-checked by I/O.

    A fitted F is a diagnostic observer parameter, not a ground calibration of
    the sky or validation of colour. Splits are disjoint by case AND night.
    """
    if (not isinstance(document,dict) or set(document)!={'schema_version','kind','cases','observations'}
            or type(document['schema_version']) is not int or document['schema_version']!=1
            or document['kind']!='eye_observations'
            or not isinstance(document['cases'],list) or not isinstance(document['observations'],list)):
        raise ValueError('Expected eye_observations schema 1 with cases and observations lists.')
    cases={}
    for case in document['cases']:
        if not isinstance(case,dict) or set(case)!={'id','night_id','result_path','result_sha256','conditions'}:
            raise ValueError('Each case requires id, night_id, result_path, result_sha256 and conditions.')
        cid=_identifier(case['id'],'case id');_identifier(case['night_id'],'night id')
        if cid in cases or cid not in results:raise ValueError('Case IDs must be unique and have a loaded result.')
        if not isinstance(case['result_sha256'],str) or len(case['result_sha256'])!=64 or any(
                c not in '0123456789abcdef' for c in case['result_sha256']):
            raise ValueError('A lowercase SHA-256 is required for each physical result.')
        condition=case['conditions']
        if (not isinstance(condition,dict) or set(condition)!={'clouds','adaptation_minutes','direct_lamps_visible'}
                or condition['clouds'] not in ('clear','other','unknown')
                or type(condition['direct_lamps_visible']) is not bool
                or isinstance(condition['adaptation_minutes'],bool)
                or not isinstance(condition['adaptation_minutes'],(int,float))
                or not np.isfinite(condition['adaptation_minutes']) or condition['adaptation_minutes']<0):
            raise ValueError('Case conditions require clouds, nonnegative adaptation minutes and a lamps boolean.')
        cases[cid]=case
    grouped=defaultdict(list);seen_keys=set();night_splits=defaultdict(set);hash_splits=defaultdict(set)
    excluded=Counter();observer_ids=set();observed_nights=set();colour_labels=0
    for record in document['observations']:
        if not isinstance(record,dict) or set(record)!={'case_id','observer_id','hr','seen','split','colour_label'}:
            raise ValueError('An observation requires case_id, observer_id, hr, seen, split and colour_label.')
        cid=record['case_id'];oid=_identifier(record['observer_id'],'observer id')
        if cid not in cases or record['split'] not in ('calibration','validation'):
            raise ValueError('Unknown case or split; use calibration or validation.')
        if type(record['seen']) is not bool or type(record['hr']) is not int or record['hr']<=0:
            raise ValueError('seen must be a boolean and hr a positive integer.')
        if record['colour_label'] not in (None,'white','red','orange','yellow','blue','uncertain'):
            raise ValueError('Unknown reported colour label.')
        key=(cid,oid,record['hr'])
        if key in seen_keys:raise ValueError('Duplicate observer/case/star observation.')
        seen_keys.add(key);case=cases[cid]
        night_splits[case['night_id']].add(record['split']);hash_splits[case['result_sha256']].add(record['split'])
        observer_ids.add(oid);observed_nights.add(case['night_id']);colour_labels+=record['colour_label'] is not None
        result=results[cid];eye=result.get('eye') or {};condition=case['conditions']
        if (condition['clouds']!='clear' or condition['adaptation_minutes']<20 or condition['direct_lamps_visible']):
            excluded['conditions_outside_protocol']+=1;continue
        star=next((r for r in eye.get('stars',[]) if r['hr']==record['hr']),None)
        if star is None:raise ValueError(f'Star HR {record["hr"]} is absent from case {cid}.')
        q=star.get('flux_over_threshold');factor=eye.get('config',{}).get('observer_factor')
        if not eye.get('available') or not star.get('supported') or q is None:
            excluded['model_unsupported']+=1;continue
        if not np.isfinite(q) or q<0 or not isinstance(factor,(int,float)) or not .5<=factor<=10:
            raise ValueError('Saved physical result has invalid threshold ratio or observer factor.')
        grouped[oid].append({**record,'night_id':case['night_id'],'base_signal':float(q*factor),'original_factor':factor})
    if any(len(v)>1 for v in [*night_splits.values(),*hash_splits.values()]):
        raise ValueError('Calibration and validation must not share a night or physical result, even across observers.')
    summaries=[]
    for oid in sorted(observer_ids):
        rows=grouped[oid];cal=[r for r in rows if r['split']=='calibration'];val=[r for r in rows if r['split']=='validation']
        fitted=_fit(cal)
        summaries.append({'observer_id':oid,'status':'fitted_diagnostic_only' if fitted else 'insufficient_calibration',
            'observer_factor':fitted,'calibration':_score(cal,fitted) if fitted else {'records':len(cal)},
            'validation':_score(val,fitted) if fitted else {'records':len(val),'accuracy':None},
            'validation_by_night':{n:_score([r for r in val if r['night_id']==n],fitted)
                for n in sorted({r['night_id'] for r in val})} if fitted else {}})
    return {'schema_version':1,'kind':'eye_observation_audit','dataset_id':content_key(document),
        'status':'prepared_not_observed' if not document['observations'] else 'audited_not_general_validation',
        'records_submitted':len(document['observations']),'records_usable':sum(map(len,grouped.values())),
        'excluded_records':sum(excluded.values()),'exclusions':dict(excluded),
        'observers':len(observer_ids),'nights':len(observed_nights),'reported_colour_records':colour_labels,
        'case_and_night_disjoint_splits':True,'fit_uses_validation_labels':False,
        'source_coefficient_refitted':False,'colour_accuracy_validated':False,
        'observer_results':summaries,
        'limitations':['F may absorb bias in the sky, atmosphere and observer; it is not independent sky calibration.',
            'No calibrated colour categories or display-primary spectra are inferred.',
            'Human observation records are user supplied; their authenticity is not established by the importer.']}
