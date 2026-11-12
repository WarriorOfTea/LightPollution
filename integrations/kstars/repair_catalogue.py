"""Run the pinned KStars cross-ID repair in bounded declination slices.

Only the selected copy of deepstars.dat is changed. Native named/unnamed
catalogues are fixed references. Matching and binary preservation stay upstream.
"""

import argparse
from contextlib import redirect_stdout
import gc
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

TOOL_SHA256='3e3f2ec0eca38771ef2172b4f75d5383f9f51ee448e44a1a5bf8d728cfab4936'


def repair(source, catalogue, report):
    source,catalogue=Path(source).resolve(),Path(catalogue).resolve()
    tool=source/'kstars/data/tools/fix-star-catalogs.py'
    if hashlib.sha256(tool.read_bytes()).hexdigest()!=TOOL_SHA256:
        raise ValueError('Unsupported upstream repair revision; do not run an unaudited tool.')
    spec=importlib.util.spec_from_file_location('kstars_catalogue_repair',tool)
    upstream=importlib.util.module_from_spec(spec);spec.loader.exec_module(upstream)
    fixed=[source/'kstars/data'/name for name in ['namedstars.dat','unnamedstars.dat']]
    initial=hashlib.sha256(catalogue.read_bytes()).hexdigest()
    # A ghost lies at J1991.25; bound its possible declination displacement.
    maximum_pm=0.
    for path in [*fixed,catalogue]:
        c=upstream.Catalog(path);start=min(row[1] for row in c.index)
        raw=np.ndarray(((len(c.data)-start)//32,8),'<i4',buffer=c.data,offset=start)
        maximum_pm=max(maximum_pm,float(np.hypot(raw[:,2].astype(float),raw[:,3]).max())/10)
    halo=max(1.,maximum_pm/1000*upstream.EPOCH_SHIFT_YEARS/3600+2/3600)
    del c,raw
    band=[-90.,-80.]
    def offsets(self):
        start=min(row[1] for row in self.index)
        dec=np.ndarray(((len(self.data)-start)//32,),'<i4',buffer=self.data,offset=start+4,strides=(32,))/1e5
        selected=np.flatnonzero((dec>=band[0]-halo)&(dec<=band[1]+halo))
        for index in selected:
            yield 0,start+32*int(index)
    upstream.Catalog.offsets=offsets
    logs=[]
    for lower in range(-90,90,10):
        band[:]=[float(lower),float(lower+10)]
        buffer=io.StringIO()
        with redirect_stdout(buffer):
            upstream.fix_duplicates(SimpleNamespace(fixed=[str(p) for p in fixed],
                files=[str(catalogue)],dry_run=False))
        logs.append({'declination_band_deg':band.copy(),'upstream_output':buffer.getvalue()})
        print('Repaired declination',lower,'to',lower+10,flush=True)
        gc.collect()
    result={'schema_version':1,'tool_sha256':TOOL_SHA256,'matching_engine':'KStars fix-star-catalogs.py dedupe',
        'input_sha256':initial,'output_sha256':hashlib.sha256(catalogue.read_bytes()).hexdigest(),
        'fixed_catalogues':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in fixed},
        'max_proper_motion_mas_yr':maximum_pm,'declination_halo_deg':halo,'slices':logs,
        'proximity_only_merging':False,'native_catalogues_modified':False}
    Path(report).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result


def hd_index(source,catalogue,destination):
    """Rebuild the byte offsets invalidated by removing Tycho records.

    A genuinely ambiguous HD is left absent, never pointed at an arbitrary star.
    Repeated identical records in several HTM cells share one identity.
    """
    source,catalogue=Path(source),Path(catalogue)
    tool=source/'kstars/data/tools/fix-star-catalogs.py'
    if hashlib.sha256(tool.read_bytes()).hexdigest()!=TOOL_SHA256:raise ValueError('Unsupported repair revision.')
    spec=importlib.util.spec_from_file_location('kstars_index_reader',tool)
    upstream=importlib.util.module_from_spec(spec);spec.loader.exec_module(upstream)
    c=upstream.Catalog(catalogue);start=min(row[1] for row in c.index);size=(len(c.data)-start)//32
    hd=np.ndarray((size,),'<i4',buffer=c.data,offset=start+20,strides=(32,))
    raw=np.ndarray((size,),'V32',buffer=c.data,offset=start)
    positions=np.flatnonzero((hd>0)&(hd<=360000))
    _,first=np.unique(raw[positions],return_index=True);positions=positions[first]
    values=hd[positions];unique,counts=np.unique(values,return_counts=True)
    unambiguous=unique[counts==1];keep=np.isin(values,unambiguous)
    index=np.zeros(360000,dtype='<u4');index[values[keep]-1]=start+32*positions[keep]
    Path(destination).write_bytes(index.tobytes())
    magnitude=np.ndarray((size,),'<i2',buffer=c.data,offset=start+24,strides=(32,))/100
    samples=positions[keep][magnitude[positions[keep]]>=9][:3]
    return {'indexed_HD':int(keep.sum()),'ambiguous_HD_omitted':int(np.count_nonzero(counts>1)),
        'sha256':hashlib.sha256(index.tobytes()).hexdigest(),'byte_offsets_rebuilt':True,
        'lookup_samples':[{'hd':int(hd[i]),'v_mag':float(magnitude[i]),'byte_offset':int(start+32*i)} for i in samples]}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--catalogue',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args();repair(args.source,args.catalogue,args.report)
