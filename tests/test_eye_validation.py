"""Synthetic labels test the importer/fit, never count as human validation."""

from copy import deepcopy

import pytest

from lightpollution.application.eye_validation import audit_observations


def fixture():
    cases=[];results={};observations=[]
    for index,split in enumerate(['calibration','validation']):
        cid=f'case-{index}';cases.append({'id':cid,'night_id':f'night-{index}',
            'result_path':cid+'.json','result_sha256':str(index)*64,
            'conditions':{'clouds':'clear','adaptation_minutes':30,'direct_lamps_visible':False}})
        results[cid]={'eye':{'available':True,'config':{'observer_factor':2.},
            'stars':[{'hr':hr,'supported':True,'flux_over_threshold':base/2}
                for hr,base in [(1,.3),(2,1.),(3,3.),(4,8.)]]}}
        observations.extend({'case_id':cid,'observer_id':'synthetic-1','hr':hr,'seen':seen,
            'split':split,'colour_label':None} for hr,seen in [(1,False),(2,False),(3,True),(4,True)])
    return {'schema_version':1,'kind':'eye_observations','cases':cases,'observations':observations},results


def test_heldout_labels_do_not_change_fit_and_confusion_denominators():
    document,results=fixture();report=audit_observations(document,results)
    observer=report['observer_results'][0]
    assert observer['observer_factor']==2. and observer['validation']['accuracy']==1.
    assert observer['validation']['seen']==observer['validation']['not_seen']==2
    assert observer['validation']['TP']==observer['validation']['TN']==2
    altered=deepcopy(document)
    for row in altered['observations']:
        if row['split']=='validation':row['seen']=not row['seen']
    other=audit_observations(altered,results)['observer_results'][0]
    assert other['observer_factor']==observer['observer_factor']
    assert other['validation']['FP']==other['validation']['FN']==2 and other['validation']['accuracy']==0.
    assert not report['source_coefficient_refitted'] and not report['colour_accuracy_validated']


@pytest.mark.parametrize('field',['night_id','result_sha256'])
def test_night_and_case_leakage_is_rejected(field):
    document,results=fixture();document['cases'][1][field]=document['cases'][0][field]
    with pytest.raises(ValueError,match='must not share'):audit_observations(document,results)


def test_empty_dataset_is_explicitly_not_observed():
    report=audit_observations({'schema_version':1,'kind':'eye_observations','cases':[],'observations':[]},{})
    assert report['status']=='prepared_not_observed' and report['observers']==report['nights']==0


def test_missing_or_single_class_calibration_is_not_a_fitted_observer():
    document,results=fixture()
    for row in document['observations']:
        if row['split']=='calibration':row['seen']=True
    report=audit_observations(document,results)['observer_results'][0]
    assert report['status']=='insufficient_calibration' and report['observer_factor'] is None


def test_duplicate_boolean_and_outside_protocol_records():
    document,results=fixture();document['observations'].append(deepcopy(document['observations'][0]))
    with pytest.raises(ValueError,match='Duplicate'):audit_observations(document,results)
    document,results=fixture();document['observations'][0]['seen']=1
    with pytest.raises(ValueError,match='boolean'):audit_observations(document,results)
    document,results=fixture();document['cases'][1]['conditions']['direct_lamps_visible']=True
    report=audit_observations(document,results)
    assert report['records_submitted']==8 and report['records_usable']==4 and report['excluded_records']==4
    assert report['observer_results'][0]['validation']['records']==0
