"""Explicit terrain preparation and checked offline height-field loading."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

from ..domain.scene import DomainError
from ..models.terrain import TerrainGrid
from .storage import file_hash, read_json, write_json, write_npz

DEFINITIONS = {'petersburg-copdem90': (28, 58, 33, 62),
               'madrid-copdem90': (-5, 39, -2, 42)}
ATTRIBUTION = ('Copernicus DEM GLO-90; © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018; '
               'provided under the Copernicus DEM licence. Modified: bilinear common-grid regional mosaic. '
               'https://registry.opendata.aws/copernicus-dem/')


def load_terrain(dataset, directory):
    if dataset in ('demo-flat', 'demo-ridge'):
        lat, lon = np.linspace(58, 62, 241), np.linspace(27, 34, 421)
        yy, xx = np.meshgrid(lat, lon, indexing='ij')
        height = np.zeros(yy.shape)
        if dataset == 'demo-ridge':
            height = 1200*np.exp(-((yy-60.23)/.025)**2)*np.exp(-((xx-30.315)/.6)**8)
        return TerrainGrid(lat, lon, height,
            {'id': dataset, 'kind': 'synthetic', 'vertical_datum': 'hypothetical spherical MSL',
             'description': 'Analytic ridge or flat validation field; not measured topography.'})
    folder = Path(directory)/dataset
    path = folder/'terrain.npz'
    try:
        record = read_json(folder/'manifest.json')
        if file_hash(path) != record['prepared_sha256']:
            raise ValueError('Terrain checksum mismatch.')
        with np.load(path, allow_pickle=False) as array:
            return TerrainGrid(array['latitudes'], array['longitudes'], array['heights_m'], record)
    except (OSError, ValueError, KeyError) as exc:
        raise DomainError('missing_terrain', 'Нет проверенного набора рельефа. Выполните prepare-terrain для указанного dataset.',
                          {'dataset': dataset}) from exc


def prepare_terrain(dataset, directory, progress=print):
    """Public unsigned COG download; no NASA token or implicit network in Planner."""
    if dataset not in DEFINITIONS:
        raise DomainError('invalid_terrain', 'Выберите закреплённый реальный набор рельефа.')
    try:
        import rasterio
        from rasterio.merge import merge
        from rasterio.enums import Resampling
    except ImportError as exc:
        raise DomainError('missing_dependency', 'Подготовка GeoTIFF требует uv sync --extra terrain.') from exc
    folder = Path(directory)/dataset
    folder.mkdir(parents=True, exist_ok=True)
    west, south, east, north = DEFINITIONS[dataset]
    specifications = []
    for latitude in range(south, north):
        for longitude in range(west, east):
            name = (f'Copernicus_DSM_COG_30_{"N" if latitude>=0 else "S"}{abs(latitude):02}_00_'
                    f'{"E" if longitude>=0 else "W"}{abs(longitude):03}_00_DEM')
            specifications.append((name, f'https://copernicus-dem-90m.s3.eu-central-1.amazonaws.com/{name}/{name}.tif'))
    def fetch(spec):
        name, url = spec
        path = folder/'tiles'/(name+'.tif')
        if not path.exists():
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            if not response.content.startswith((b'II', b'MM')):
                raise ValueError('Response is not a GeoTIFF.')
            from .storage import atomic_target
            with atomic_target(path) as temporary:
                temporary.write_bytes(response.content)
        with rasterio.open(path) as source:
            if source.crs.to_epsg() != 4326:
                raise ValueError('Unexpected DEM horizontal CRS.')
            transform = list(source.transform)
            shape = list(source.shape)
            pixel_type = source.tags().get('AREA_OR_POINT')
        return {'filename': path.name, 'url': url, 'bytes': path.stat().st_size,
                'sha256': file_hash(path), 'transform': transform, 'shape': shape, 'AREA_OR_POINT': pixel_type}
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(fetch, specifications))
    progress(f'{dataset}: проверено {len(records)}/{len(specifications)} тайлов')
    opened = [rasterio.open(folder/'tiles'/r['filename']) for r in records]
    try:
        # Use actual GeoTIFF affine transforms, including PixelIsPoint adjustment;
        # do not place the original southern/western point on an area corner.
        grid, transform = merge(opened, bounds=(west, south, east, north), res=(1/1200, 1/1200),
                                resampling=Resampling.bilinear, nodata=-32767., dtype='float32')
    finally:
        for stream in opened:
            stream.close()
    # Point-grid tiles do not cover the nominal outer area edge. Keep only
    # the interior; never fill these non-observed border cells with zero.
    heights = grid[0][::-1][2:-2, 2:-2].copy()
    if np.any(heights <= -32760) or not np.all(np.isfinite(heights)):
        raise ValueError('Missing terrain is not flat terrain; mosaic contains no-data cells.')
    lat = (transform.f+transform.e*(np.arange(grid.shape[1])+.5))[::-1][2:-2]
    lon = (transform.c+transform.a*(np.arange(grid.shape[2])+.5))[2:-2]
    path = folder/'terrain.npz'
    write_npz(path, latitudes=lat, longitudes=lon, heights_m=heights)
    record = {'id': dataset, 'kind': 'Copernicus GLO-90 DSM', 'vertical_datum': 'EGM2008 orthometric metres',
        'horizontal_crs': 'EPSG:4326', 'source_release': 'AWS Copernicus DEM 2021 release',
        'attribution': ATTRIBUTION, 'bounds': [west, south, east, north], 'shape': list(heights.shape),
        'minimum_height_m': float(heights.min()), 'maximum_height_m': float(heights.max()),
        'native_nominal_resolution_m': 90, 'prepared_grid_degrees': 1/1200,
        'cropped_unobserved_outer_border_cells': 2,
        'upsampling_adds_no_information': True, 'source_tiles': records,
        'prepared_sha256': file_hash(path), 'build': 'rasterio merge bilinear; checked actual COG affine transforms',
        'spherical_geoid_approximation': True, 'is_bare_ground_DTM': False}
    write_json(folder/'manifest.json', record)
    return record
