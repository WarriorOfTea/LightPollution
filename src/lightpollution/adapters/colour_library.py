"""Small CIE datasets, separate from transport/stellar flux cache identities."""

from functools import lru_cache
from pathlib import Path

import numpy as np

from .storage import content_key,file_hash,read_json

DIRECTORY=Path(__file__).resolve().parents[1]/'resources/colour'


class ColourLibrary:
    def __init__(self,directory=DIRECTORY):
        directory=Path(directory)
        self.provenance=read_json(directory/'provenance.json')
        for name,expected in self.provenance['sha256'].items():
            if file_hash(directory/name)!=expected:
                raise ValueError(f'Corrupt CIE colour table: {name}.')
        self.identity=content_key(self.provenance)
        self.xyz=np.loadtxt(directory/'CIE_xyz_1931_2deg.csv',delimiter=',')
        self.lms=np.loadtxt(directory/'CIE_lms_cf_2deg.csv',delimiter=',')
        self.d65=np.loadtxt(directory/'CIE_std_illum_D65.csv',delimiter=',')
        missing=np.isnan(self.lms[:,1:])
        permitted=np.zeros_like(missing);permitted[:,2]=self.lms[:,0]>615
        if np.any(missing & ~permitted):
            raise ValueError('Missing CIE sensitivities inside their published wavelength domain.')
        self.lms[:,1:]=np.where(missing,0,self.lms[:,1:])
        for data in [self.xyz,self.lms,self.d65]:
            if not np.all(np.isfinite(data)) or np.any(data<0) or np.any(np.diff(data[:,0])<=0):
                raise ValueError('Invalid CIE colour table axis or values.')

    def responses(self,wavelengths):
        axis=np.asarray(wavelengths,dtype=float)
        if np.any(~np.isfinite(axis)) or np.any(np.diff(axis)<=0):
            raise ValueError('Increasing finite colour wavelengths required.')
        interpolate=lambda table:np.stack([np.interp(axis,table[:,0],table[:,j],left=0,right=0)
                                          for j in range(1,table.shape[1])],axis=-1)
        return interpolate(self.xyz),interpolate(self.lms)


@lru_cache(maxsize=2)
def _cached(identity):return ColourLibrary()


def load_colour_library():
    return _cached(tuple(file_hash(DIRECTORY/name) for name in
        ['provenance.json','CIE_xyz_1931_2deg.csv','CIE_lms_cf_2deg.csv','CIE_std_illum_D65.csv']))
