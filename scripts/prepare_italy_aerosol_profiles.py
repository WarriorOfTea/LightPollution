"""Prepare optical-depth layers from NASA GEOS-CF PM2.5 and pinned CAMS column AOD."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
from pathlib import Path
import re
import time

import numpy as np
import requests

from lightpollution.adapters.storage import file_hash, read_json, write_json

NASA = 'https://opendap.nccs.nasa.gov/dods/gmao/geos-cf/'
V1 = ('assim/chm_tavg_1hr_g1440x721_v36', 'assim/met_tavg_1hr_g1440x721_v36')
V2 = ('v2/ana/chm_inst_1hr_glo_L1440x721_v72', 'v2/ana/met_inst_1hr_glo_L1440x721_v72')
G = 9.80665
HEIGHT_BINS = np.array([0, 200, 400, 600, 800, 1000, 1250, 1500, 1750, 2000,
                        2500, 3000, 3500, 4000, 5000, 6000, 8000, 10000, 12000,
                        16000, 20000, 30000, 80000], dtype=float)
LATITUDE = np.arange(43.5, 47.26, .25)
LONGITUDE = np.arange(10., 14.01, .25)
SPATIAL_SLICE = '[534:549][760:776]'


def retrieve(cache, dataset, suffix):
    url = NASA + dataset + suffix
    key = hashlib.sha256(url.encode()).hexdigest()
    path = cache / key
    if not path.is_file():
        for delay in (0, 2, 7):
            if delay:
                time.sleep(delay)
            try:
                response = requests.get(url, timeout=45)
                response.raise_for_status()
                if response.content.lstrip().startswith((b'<', b'Error')):
                    raise ValueError('Unexpected NASA response format.')
                path.write_bytes(response.content)
                break
            except (requests.RequestException, ValueError) as exc:
                if delay == 7:
                    raise RuntimeError(f'GEOS-CF retrieval failed for {url}: {exc}') from exc
    return url, file_hash(path), path.read_text()


def dap_arrays(reply):
    """Parse the GrADS ASCII arrays returned by the documented OPeNDAP subsets."""
    arrays = {}
    sections = re.split(r'(?m)^(\w+), ((?:\[\d+\])+)[ \t]*\r?\n', reply)
    if len(sections) == 1:
        raise ValueError('No arrays in GEOS-CF response.')
    for name, shape_text, lines in zip(sections[1::3], sections[2::3], sections[3::3], strict=True):
        shape = tuple(map(int, re.findall(r'\d+', shape_text)))
        values = [np.fromstring(line.split('], ')[-1], dtype=float, sep=',')
                  for line in lines.splitlines() if line.strip()]
        arrays[name] = np.concatenate(values).reshape(shape)
    return arrays


class Collection:
    def __init__(self, cache, dataset):
        self.cache, self.dataset = cache, dataset
        url, sha, das = retrieve(cache, dataset, '.das')
        self.metadata = {'url': url, 'sha256': sha}
        time_section = das.split('    time {', 1)[1].split('    }', 1)[0]
        first = re.search(r'grads_min "([^"]+)"', time_section)[1]
        self.start = datetime.strptime(first, '%H:%Mz%d%b%Y' if ':' in first else '%Hz%d%b%Y')
        self.steps = int(re.search(r'grads_size "([0-9]+)"', time_section)[1])
        sub = '.ascii?time[0]'
        _, _, start_time = retrieve(cache, dataset, sub)
        self.time_origin = float(dap_arrays(start_time)['time'][0])

    def get(self, when, fields, levels, count):
        index, remainder = divmod((when - self.start).total_seconds(), 3600)
        index = int(index)
        if remainder or index < 0 or index + count > self.steps:
            raise ValueError(f'No requested UTC in {self.dataset}: {when}.')
        temporal = f'[{index}:{index + count - 1}]'
        query = ','.join(name + temporal + (levels if is_cube else '') + SPATIAL_SLICE
                         for name, is_cube in fields)
        url, sha, reply = retrieve(self.cache, self.dataset, '.ascii?' + query)
        data = dap_arrays(reply)
        want_time = self.time_origin + (index + np.arange(count)) / 24
        if (not np.array_equal(data['lat'], LATITUDE) or not np.array_equal(data['lon'], LONGITUDE)
                or not np.allclose(data['time'], want_time, atol=1e-7, rtol=0)
                or not np.array_equal(data['lev'], np.arange(37, 73))):
            raise ValueError('GEOS-CF coordinates or UTC did not match the request.')
        return data, {'url': url, 'sha256': sha}


def profile_areas(centres, concentrations, ground, bounds=HEIGHT_BINS):
    """Integrate a concentration proxy within altitude bands above sea level."""
    if (not np.all(np.isfinite(centres) & (centres < 80_000))
            or not np.all(np.isfinite(concentrations) & (0 <= concentrations) & (concentrations < 1e5))
            or not np.isfinite(ground) or not -500 < ground < 9000
            or not np.all(np.diff(centres) > 0) or centres[0] <= ground):
        raise ValueError('Missing or invalid aerosol/altitude sample.')
    top = centres[-1] + (centres[-1] - centres[-2]) / 2
    z = np.r_[ground, centres, top]
    mass = np.r_[concentrations[0], concentrations, 0.]
    knots = np.unique(np.r_[bounds, z[z >= bounds[0]]])
    y = np.interp(knots, z, mass, right=0)
    # Concentration jumps at the ground: do not create a triangular tail in rock.
    add = (y[:-1] + y[1:]) / 2 * np.diff(knots)
    add[knots[1:] <= ground] = 0.
    cumulative = np.r_[0., np.cumsum(add)]
    return np.diff(cumulative[np.searchsorted(knots, bounds)])


def remap_xy(areas, latitudes, longitudes):
    by_lon = np.apply_along_axis(lambda line: np.interp(longitudes, LONGITUDE, line), 2, areas)
    return np.apply_along_axis(lambda line: np.interp(latitudes, LATITUDE, line), 1, by_lon)


def build_day(day, aod_endpoints, lat, lon, pair, v2):
    when = datetime.fromisoformat(day + ('T22:00:00' if v2 else 'T22:30:00'))
    count, levels = (2, '[36:71]') if v2 else (1, '[0:35]')
    var = 'pm25_rh35' if v2 else 'pm25_rh35_gcc'
    chm, chm_record = pair[0].get(when, [(var, True)], levels, count)
    met, met_record = pair[1].get(when, [('zl', True), ('phis', False)], levels, count)
    native = np.zeros((len(HEIGHT_BINS)-1, len(LATITUDE), len(LONGITUDE)))
    for i in range(len(LATITUDE)):
        for j in range(len(LONGITUDE)):
            for k in range(count):
                # GEOS model levels run from top to bottom.
                native[:, i, j] += profile_areas(met['zl'][k, ::-1, i, j], chm[var][k, ::-1, i, j],
                                                 met['phis'][k, i, j] / G)
    remapped = remap_xy(native, lat, lon)
    total = remapped.sum(axis=0)
    columns = np.asarray(aod_endpoints, dtype=float).mean(axis=0)
    if np.any(total <= 0) or not np.all(np.isfinite(columns) & (columns > 0)):
        raise ValueError(f'Missing column shape or AOD at {day}.')
    optical_layers = np.moveaxis(remapped / total * columns, 0, -1)
    np.testing.assert_allclose(optical_layers.sum(axis=-1), columns, rtol=1e-12)
    print(f'{day} GEOS-CF v{2 if v2 else 1}, {columns.size} columns', flush=True)
    return {'geos_cf_version': 2 if v2 else 1, 'geos_cf_mass_variable': var,
            'geos_cf_sources': [chm_record, met_record],
            'utc': [when.isoformat() + 'Z'] + ([day + 'T23:00:00Z'] if v2 else []),
            'aod_per_layer': optical_layers.round(12).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cams', type=Path)
    parser.add_argument('cache', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    cams = read_json(args.cams)
    lat, lon = (np.asarray(cams[k]) for k in ('latitudes', 'longitudes'))
    pairs = {v2: [Collection(args.cache, name) for name in datasets]
             for v2, datasets in ((False, V1), (True, V2))}

    def process(item):
        day, columns = item
        v2 = day > '2026-01-01'
        return day, build_day(day, columns, lat, lon, pairs[v2], v2)

    with ThreadPoolExecutor(max_workers=3) as pool:
        days = dict(pool.map(process, sorted(cams['aod_endpoints'].items())))
    write_json(args.output, {'cams_sha256': file_hash(args.cams), 'latitudes': lat.tolist(),
                             'longitudes': lon.tolist(), 'altitude_edges_m': HEIGHT_BINS.tolist(),
                             'sources': [c.metadata for pair in pairs.values() for c in pair],
                             'shape': 'GEOS-CF PM2.5 at RH35 as extinction-shape proxy, rescaled to CAMS AOD',
                             'days': days})


if __name__ == '__main__':
    main()
