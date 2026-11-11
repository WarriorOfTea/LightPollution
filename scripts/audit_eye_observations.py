"""Checksum and audit a local, anonymized eye-observation dataset (no network)."""

import argparse
import json
from pathlib import Path

from lightpollution.adapters.storage import file_hash,write_json
from lightpollution.application.eye_validation import audit_observations


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();document=json.loads(args.dataset.read_text());results={}
    for case in document.get('cases',[]):
        path=(args.dataset.parent/case['result_path']).resolve()
        if file_hash(path)!=case['result_sha256']:raise ValueError(f'Physical result checksum mismatch: {case["id"]}')
        results[case['id']]=json.loads(path.read_text())
    report=audit_observations(document,results)
    write_json(args.output,report)
    print(json.dumps({k:report[k] for k in ['status','observers','nights','records_submitted','records_usable','excluded_records']}))


if __name__=='__main__':main()
