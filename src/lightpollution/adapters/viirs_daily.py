"""Geographical windows of the daily Black Marble C2 observation layers."""

from datetime import date
from pathlib import Path
import re

import h5py
import numpy as np

from ..data import Bounds, GRID, PIXELS_PER_DEGREE, text_attribute
from ..download import sha256


A1_FIELDS = (
    'DNB_At_Sensor_Radiance',
    'Solar_Zenith',
    'Lunar_Zenith',
    'Sensor_Zenith',
    'Sensor_Azimuth',
    'UTC_Time',
    'QF_Cloud_Mask',
    'QF_DNB',
)
A2_FIELDS = ('DNB_BRDF-Corrected_NTL', 'Mandatory_Quality_Flag', 'Snow_Flag')
DAILY_FIELDS = {
    'VNP46A1': A1_FIELDS,
    'VJ146A1': A1_FIELDS,
    'VNP46A2': A2_FIELDS,
    'VJ146A2': A2_FIELDS,
}


def read_daily_region(directory: Path, day: date, bounds: Bounds, product: str):
    """Return native grid arrays and pinned file records. A2 excludes its gap-filled layer."""
    fields = DAILY_FIELDS[product]
    result = {name: np.full(bounds.shape, np.nan) for name in fields}
    manifest = []
    p = PIXELS_PER_DEGREE
    for tile in bounds.tiles():
        files = sorted(directory.glob(f'{product}.A{day:%Y%j}.{tile}.002.*.h5'))
        if len(files) != 1:
            raise ValueError(
                f'Expected one {product} C2 file for {day} {tile}, found {len(files)}.'
            )
        path = files[0]
        h, v = map(int, re.fullmatch(r'h(\d\d)v(\d\d)', tile).groups())
        west, north = -180 + 10 * h, 90 - 10 * v
        left, right = max(west, bounds.west), min(west + 10, bounds.east)
        bottom, top = max(north - 10, bounds.south), min(north, bounds.north)
        src = (
            slice(round((north - top) * p), round((north - bottom) * p)),
            slice(round((left - west) * p), round((right - west) * p)),
        )
        dst = (
            slice(round((bounds.north - top) * p), round((bounds.north - bottom) * p)),
            slice(round((left - bounds.west) * p), round((right - bounds.west) * p)),
        )
        with h5py.File(path, 'r') as file:
            attrs = file.attrs
            if (
                text_attribute(attrs, 'ShortName') != product
                or text_attribute(attrs, 'VersionID') != '002'
                or text_attribute(attrs, 'RangeBeginningDate') != day.isoformat()
            ):
                raise ValueError(
                    f'Product, collection or date mismatch in {path.name}.'
                )
            for label, value in (
                ('West', west),
                ('East', west + 10),
                ('North', north),
                ('South', north - 10),
            ):
                if not np.isclose(
                    np.asarray(attrs[label + 'BoundingCoord']).item(),
                    value,
                    atol=1e-6,
                    rtol=0,
                ):
                    raise ValueError(f'Unexpected {label} bound in {path.name}.')
            grid = file[GRID]
            for name in fields:
                dataset = grid[name]
                if dataset.shape != (2400, 2400):
                    raise ValueError(f'Unexpected native grid in {dataset.name}.')
                raw = dataset[src]
                fill = np.asarray(dataset.attrs['_FillValue']).item()
                # These C2 fields have either zero offsets or already physical values.
                offset = np.asarray(
                    dataset.attrs.get('add_offset', dataset.attrs.get('offset', 0))
                ).item()
                if offset != 0:
                    raise ValueError(
                        f'Undocumented offset for daily field {dataset.name}.'
                    )
                scale = np.asarray(dataset.attrs.get('scale_factor', 1)).item()
                if name in (A1_FIELDS[0], A2_FIELDS[0]):
                    units = text_attribute(dataset.attrs, 'units').replace(' ', '')
                    if units not in ('nW/(cm2sr)', 'nWatts/(cm^2sr)', 'nW/(cm^2sr)'):
                        raise ValueError(f'Unknown radiance units in {dataset.name}.')
                values = raw.astype(float) * scale
                values[(raw == fill) | ~np.isfinite(raw)] = np.nan
                result[name][dst] = values
        manifest.append(
            {
                'filename': path.name,
                'sha256': sha256(path),
                'tile': tile,
                'product': product,
            }
        )
    return result, manifest


def clear_dark_a1(data, *, lunar_zenith_min=100.0, solar_zenith_min=108.0):
    """Night, medium/high VCM, clear/probably clear, and no cirrus, snow or aurora."""
    quality = data['QF_Cloud_Mask']
    cloud = np.where(np.isfinite(quality), quality, 65535).astype(np.uint16)
    bad_bits = (1 << 0) | (1 << 9) | (1 << 10) | (1 << 12) | (1 << 13)
    return (
        np.isfinite(quality)
        & ((cloud & bad_bits) == 0)
        & (((cloud >> 4) & 3) >= 2)
        & (((cloud >> 6) & 3) <= 1)
        & (data['QF_DNB'] == 0)
        & (data['DNB_At_Sensor_Radiance'] >= 0)
        & (data['Solar_Zenith'] >= solar_zenith_min)
        & (data['Lunar_Zenith'] >= lunar_zenith_min)
        & np.isfinite(data['UTC_Time'])
        & np.isfinite(data['Sensor_Azimuth'])
        & (data['Sensor_Zenith'] >= 0)
        & (data['Sensor_Zenith'] < 75)
    )
