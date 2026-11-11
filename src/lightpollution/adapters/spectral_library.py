"""Verified offline reference spectra; no implicit network access."""

import csv
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..domain.scene import DomainError
from ..models.spectra import band_integral, mix_templates
from .storage import content_key, file_hash, read_json

DIRECTORY = Path(__file__).resolve().parents[1] / 'resources/spectra'


class SpectralLibrary:
    def __init__(self, directory=DIRECTORY):
        directory = Path(directory)
        self.provenance = read_json(directory/'provenance.json')
        for name, expected in self.provenance['sha256'].items():
            if file_hash(directory/name) != expected:
                raise DomainError('corrupt_spectral_library', 'Изменились контрольные суммы спектральных данных.')
        self.identity = content_key(self.provenance)
        with np.load(directory/'library.npz', allow_pickle=False) as source:
            self.wavelengths = source['wavelength_nm']
            self.templates = {k.removeprefix('lamp_'): source[k] for k in source.files if k.startswith('lamp_')}
            self.responses = {k.removeprefix('response_'): source[k] for k in source.files if k.startswith('response_')}
            self.measured = {record['hr']: source['star_flux_W_m2_nm'][i]
                             for i, record in enumerate(self.provenance['stars'])}
        self.colours = {int(r['hr']): r for r in csv.DictReader((directory/'bsc-colours.csv').open())}
        vega = self.measured[7001]
        # Explicit conventional B=V=0.03 for Vega. Johnson V != CIE V(lambda).
        self.zero_v = float(band_integral(self.wavelengths, vega, self.responses['johnson_v'])*10**(.4*.03))
        self.zero_b = float(band_integral(self.wavelengths, vega, self.responses['johnson_b'])*10**(.4*.03))

    def mix(self, fractions):
        try:
            return mix_templates(self.wavelengths, self.templates, fractions)
        except ValueError as exc:
            raise DomainError('invalid_spectrum', str(exc), {'available_templates': sorted(self.templates)}) from exc

    def catalogue(self):
        return {'templates': sorted(self.templates), 'wavelength_nm': self.wavelengths.tolist(),
                'responses': sorted(self.responses), 'library_id': self.identity,
                'measured_stars': len(self.measured), 'units': 'energy per nm',
                'local_lamp_inventory_known': False}


@lru_cache(maxsize=2)
def _cached(identity):
    return SpectralLibrary()


def load_spectral_library():
    # Detect edits even within a running process, not just across restarts.
    identity = tuple(file_hash(DIRECTORY/name) for name in ['provenance.json', 'library.npz', 'bsc-colours.csv'])
    return _cached(identity)
