"""Conservative BSC cross-ID: a unique HD plus compatible J2000 coordinates.

Proximity alone never merges objects or transfers a spectrum. KStars positions,
magnitudes and proper motions remain the inputs to the observation calculation.
"""

import csv
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

from .storage import file_hash

PATH=Path(__file__).resolve().parents[1]/'catalogues/bsc5-cross-identifications.csv'


@lru_cache(maxsize=2)
def _index(identity):
    by_hd=defaultdict(list)
    with PATH.open() as stream:
        for row in csv.DictReader(stream):
            by_hd[int(row['hd'])].append(row)
    return by_hd


def load_cross_identifications():
    identity=file_hash(PATH)
    return _index(identity),identity


def cross_id(star, index=None):
    a=star.get('astrometry')
    if not a or not a['hd'] or a['multiple']:
        return None
    by_hd,identity=index if index is not None else load_cross_identifications()
    candidates=by_hd.get(a['hd'],[])
    if len(candidates)!=1:
        return None
    row=candidates[0]
    r0,d0,r1,d1=np.deg2rad([float(row['ra_deg']),float(row['dec_deg']),a['ra_j2000_deg'],a['dec_j2000_deg']])
    hav=np.sin((d1-d0)/2)**2+np.cos(d0)*np.cos(d1)*np.sin((r1-r0)/2)**2
    separation=float(np.rad2deg(2*np.arcsin(np.sqrt(np.clip(hav,0,1))))*3600)
    if separation>3:
        return None
    return {'hr':int(row['hr']),'hd':a['hd'],'separation_j2000_arcsec':separation,
            'rule':'unique_BSC_HD_and_J2000_separation_le_3_arcsec','catalogue_sha256':identity}
