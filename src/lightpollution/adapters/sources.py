"""Pinned NASA regions and an explicit synthetic offline example."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Protocol

import numpy as np

from ..data import Bounds, good_mask, load_region, read_region, save_region
from ..domain.scene import DomainError
from ..download import download_composite, verify_file
from ..periods import CompositePeriod
from ..physics import EARTH_RADIUS_M, Observer, Sources, unit_position
from ..seasonal import prepare_seasonal
from ..sources import regional_sources, synthetic_cities
from .storage import atomic_target, content_key, file_hash, read_json, write_json

RESOURCES = Path(__file__).resolve().parents[1] / 'resources'


class SourceRepository(Protocol):
    def catalogue(self) -> list[dict]: ...
    def prepare(self, region: str, period: str, observer: Observer,
                allow_download: bool = False): ...


class PreparedSources:
    def __init__(self, entry, observer, paths, report, identity):
        self.entry, self.observer, self.paths = entry, observer, paths
        self.report, self.identity = report, identity
        self.reference_period = entry['reference_period']
        self._arrays, self._selected = {}, {}

    def select(self, period, block=12, direction=None):
        key = (period, block, direction)
        if key in self._selected:
            return self._selected[key]
        if self.entry['kind'] == 'synthetic':
            sources = synthetic_cities(self.observer)
            provenance = {'input_kind': 'synthetic', 'actual_nasa_pixels': 0,
                          'description': 'Two specified test cities; no satellite measurements.',
                          'source_basis_id': self.identity, 'source_power_multiplier': 1.}
        else:
            if period not in self._arrays:
                self._arrays[period] = load_region(self.paths[period])
            arrays, bounds, metadata = self._arrays[period]
            sources, provenance = regional_sources(arrays, bounds, self.observer, block=block,
                exclude_radius_m=3000, include_water=False, refinement_direction=direction)
            multiplier = metadata.get('source_power_multiplier', 1.)
            sources = Sources(sources.positions_m, sources.power * multiplier)
            provenance.update({'input_kind': 'blackmarble', 'source_basis_id': self.identity,
                               'source_power_multiplier': multiplier,
                               'region_metadata': metadata})
        self._selected[key] = sources, provenance
        return sources, provenance


class LocalSourceRepository:
    def __init__(self, workspace: Path, raw_directory: Path | None = None,
                 resources: Path = RESOURCES):
        self.workspace = Path(workspace)
        self.raw_directory = Path(raw_directory) if raw_directory else self.workspace.parent / 'blackmarble'
        self.resources = resources
        self.entries = {e['id']: e for e in read_json(resources / 'regions.json')}

    def entry(self, region):
        if region not in self.entries:
            raise DomainError('unknown_region', 'Регион не входит в подготовленный каталог MVP.')
        return self.entries[region]

    def catalogue(self):
        result = []
        for entry in self.entries.values():
            value = dict(entry)
            required = available = 0
            if entry['kind'] != 'synthetic':
                for period in entry['periods']:
                    manifest = read_json(self.resources / 'manifests' / entry['manifests'][period])
                    for record in manifest['files']:
                        required += 1
                        path = self.raw_directory / entry['folders'][period] / record['name']
                        available += bool(path.exists() and path.stat().st_size == record['size'])
            value['local_files_present'] = available
            value['required_files'] = required
            value['locally_available'] = available == required
            value['presence_is_not_QA'] = True
            result.append(value)
        return result

    def download(self, region):
        entry = self.entry(region)
        if entry['kind'] == 'synthetic':
            return
        for label in entry['periods']:
            manifest = read_json(self.resources / 'manifests' / entry['manifests'][label])
            target = self.raw_directory / entry['folders'][label]
            # A separate manifest preserves the project's historical raw manifest.
            manifest_path = target / 'mvp-pinned-manifest.json'
            write_json(manifest_path, manifest)
            try:
                download_composite(target, manifest['year'], tuple(manifest['tiles']),
                                   manifest_path=manifest_path, month=manifest.get('month'))
            except RuntimeError as exc:
                raise DomainError('nasa_download_failed', 'Не удалось загрузить закреплённые данные NASA. Проверьте настройку Earthdata и права доступа.') from exc

    def _raw_region(self, entry, label, allow_download):
        manifest = read_json(self.resources / 'manifests' / entry['manifests'][label])
        identity = content_key({'input_files': [(r['name'], r['sha256']) for r in manifest['files']],
                                'bounds': entry['bounds'], 'reader_version': 1,
                                'reader_code': file_hash(Path(__file__).resolve().parents[1] / 'data.py')})
        folder = self.workspace / 'source-cache' / identity
        path, record_path = folder / 'source-grid.npz', folder / 'verified.json'
        if path.exists() and record_path.exists():
            record = read_json(record_path)
            if file_hash(path) == record['sha256']:
                return path
        raw = self.raw_directory / entry['folders'][label]
        if not all((raw / r['name']).exists() for r in manifest['files']):
            if not allow_download:
                raise DomainError('missing_data', 'Нет локальных файлов NASA. Подготовьте регион командой datasets --download.',
                                  {'region': entry['id'], 'period': label})
            self.download(entry['id'])
        try:
            for record in manifest['files']:
                verify_file(raw / record['name'], record['size'], record['sha256'])
            arrays, attributes = read_region(raw, Bounds(**entry['bounds']), manifest['year'], manifest.get('month'))
        except (OSError, ValueError) as exc:
            raise DomainError('invalid_source_data', 'Проверка исходных файлов NASA не пройдена.',
                              {'region': entry['id'], 'period': label}) from exc
        period = CompositePeriod(manifest['year'], manifest.get('month'))
        metadata = {**period.metadata(), 'product': period.product + '.002',
                    'inputs': [{'filename': v['filename'], 'sha256': v['sha256']} for v in attributes.values()]}
        with atomic_target(path) as temporary:
            # save_region accepts a filename; a file object avoids an extra .npz suffix.
            with temporary.open('wb') as stream:
                np.savez_compressed(stream, **arrays, metadata_json=json.dumps({'bounds': entry['bounds'], **metadata}))
        write_json(record_path, {'sha256': file_hash(path), 'input_identity': identity})
        return path

    def prepare(self, region, period, observer, allow_download=False):
        entry = self.entry(region)
        if period not in entry['periods']:
            raise DomainError('unknown_period', 'Период отсутствует в выбранном наборе. Автоматической подмены нет.')
        bounds = Bounds(**entry['bounds'])
        if not (bounds.south <= observer.latitude_deg <= bounds.north
                and bounds.west <= observer.longitude_deg <= bounds.east):
            raise DomainError('outside_region', 'Точка находится вне подготовленной области источников.')
        if entry['kind'] == 'synthetic':
            identity = content_key({'kind': 'synthetic-two-cities-1', 'observer': asdict(observer)})
            return PreparedSources(entry, observer, {}, {'kind': 'synthetic', 'actual_nasa_pixels': 0}, identity)
        paths = {label: self._raw_region(entry, label, allow_download) for label in entry['periods']}
        key = content_key({'regions': {k: file_hash(v) for k, v in paths.items()},
                           'observer': asdict(observer), 'exclude_radius_m': 3000,
                           'composite': 'NearNadir', 'minimum_common_fraction': .8, 'matching_version': 1,
                           'matching_code': file_hash(Path(__file__).resolve().parents[1] / 'seasonal.py')})
        output = self.workspace / 'matched-sources' / key
        try:
            if len(paths) > 1:
                report_path = output / 'seasonal.json'
                if report_path.exists():
                    report = read_json(report_path)
                else:
                    report = prepare_seasonal(list(paths.values()), output, entry['reference_period'], observer)
                prepared = {r['period']: output / r['period'] / 'source-grid.npz' for r in report['periods']}
            else:
                label = entry['reference_period']
                arrays, b, metadata = load_region(paths[label])
                lat, lon = np.meshgrid(b.latitude, b.longitude, indexing='ij')
                distance = 2 * EARTH_RADIUS_M * np.arcsin(np.clip(np.linalg.norm(
                    unit_position(lat, lon) - unit_position(observer.latitude_deg, observer.longitude_deg), axis=-1) / 2, 0, 1))
                domain = np.isin(arrays['Land_Water_Mask'], (0, 1, 5)) & (distance >= 3000)
                name = 'NearNadir_Composite_Snow_Free'
                common = domain & good_mask(arrays[name], arrays[name + '_Quality'], arrays[name + '_Num'])
                n, denominator = int(common.sum()), int(domain.sum())
                if not n or not denominator or n / denominator < .8:
                    raise ValueError(f'Insufficient common support: {n}/{denominator} eligible pixels.')
                mask_hash = hashlib.sha256(np.packbits(common).tobytes()).hexdigest()
                report = {'reference_period': label, 'common_pixels': n,
                          'reference_land_domain_pixels': denominator,
                          'common_fraction_of_reference_land': n / denominator,
                          'common_mask_sha256': mask_hash, 'minimum_common_fraction': .8}
                target = output / label / 'source-grid.npz'
                if not target.exists():
                    arrays[name + '_Quality'][~common] = 1
                    save_region(target, arrays, b, {**metadata, 'source_power_multiplier': 1.,
                        'source_composite': 'NearNadir', 'common_mask_sha256': mask_hash,
                        'normalisation_reference_period': label})
                prepared = {label: target}
        except ValueError as exc:
            raise DomainError('insufficient_coverage', 'Недостаточно общей области качественных измерений; пропуски не являются темнотой.',
                              {'reason': str(exc), 'region': region}) from exc
        identity = content_key({'selection': key, 'common_mask': report['common_mask_sha256'],
                                'reference_period': entry['reference_period']})
        checksums = {label: file_hash(path) for label, path in prepared.items()}
        checksum_path = output / 'checksums.json'
        if checksum_path.exists() and read_json(checksum_path) != checksums:
            raise DomainError('corrupt_cache', 'Контрольная сумма подготовленных карт изменилась; удалите повреждённый кэш подготовки и повторите.')
        write_json(checksum_path, checksums)
        return PreparedSources(entry, observer, prepared, report, identity)
