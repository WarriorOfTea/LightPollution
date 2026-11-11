"""Fetch public reference data and build a small, attributed offline library.

No authentication. Raw originals remain in ignored scratch/. Downloaded
numbers are not local measurements of a city's lamp inventory.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import gzip
import hashlib
import json
from pathlib import Path

from astropy.io import fits
from astropy.io.votable import parse
import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
CIE = 'https://files.cie.co.at/Publications-datasets/'


def build(cache, output):
    cache.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    inputs = []

    def fetch(name, url):
        path = cache/name
        if not path.exists():
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            path.write_bytes(response.content)
        inputs.append({'file': name, 'url': url, 'bytes': path.stat().st_size,
                       'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        return path

    axis = np.arange(360., 831., 5.)
    templates, responses, rights = {}, {}, []
    for filename, metadata_file, kind in [
        ('CIE_illum_LEDs.csv', 'CIE_illum_LEDs.csv_metadata_v2.json', 'lamps'),
        ('CIE_illum_HPs.csv', 'CIE_illum_HPs.csv_metadata_v2.json', 'lamps'),
        ('CIE_sle_photopic.csv', 'CIE_sle_photopic.csv_metadata.json', 'photopic'),
        ('CIE_sle_scotopic.csv', 'CIE_sle_scotopic.csv_metadata.json', 'scotopic')]:
        path, meta_path = fetch(filename, CIE+filename), fetch(metadata_file, CIE+metadata_file)
        meta = json.loads(meta_path.read_text())
        expected = next(v['checksum'] for v in meta['checksums'] if v['hashMethod'] == 'sha256')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
        table = np.loadtxt(path, delimiter=',')
        assert np.all(np.isfinite(table)) and np.all(table[:, 1:] >= 0)
        rights.append({'file': filename, 'DOI': meta['identifier']['identifier'],
                       'rights': meta['rightsList'], 'publisher_sha256_verified': True})
        for column, header in enumerate(meta['datatableInfo']['columnHeaders'][1:], 1):
            value = np.interp(axis, table[:, 0], table[:, column], left=0, right=0)
            (templates if kind == 'lamps' else responses)[header['title'] if kind == 'lamps' else kind] = value

    filters = {}
    for band in ['B', 'V']:
        url = f'https://svo2.cab.inta-csic.es/theory/fps/fps.php?ID=Generic/Johnson.{band}'
        table = parse(fetch(f'johnson-{band.lower()}.xml', url)).get_first_table()
        params = {p.name: p.value for p in table.params}
        assert int(params['DetectorType']) == 0  # Energy response, no extra lambda factor.
        responses['johnson_'+band.lower()] = np.interp(axis, table.array['Wavelength']/10,
                                                      table.array['Transmission'], left=0, right=0)
        filters[band] = {'identifier': params['filterID'], 'detector': 'energy', 'url': url,
                         'acknowledgement': 'SVO Filter Profile Service Carlos Rodrigo; Rodrigo et al. 2024, A&A 689 A93.'}

    reference_stars, star_info = [], []
    for name, hr, filename in [('Vega', 7001, 'alpha_lyr_stis_011.fits'), ('Sirius', 2491, 'sirius_stis_005.fits')]:
        url = 'https://ssb.stsci.edu/trds/calspec/'+filename
        path = fetch(name.lower()+'.fits', url)
        with fits.open(path) as hdus:
            table = hdus[1].data
            # erg/(s cm2 Angstrom) -> W/(m2 nm): multiply by 0.01.
            value = np.interp(axis, table['WAVELENGTH']/10, table['FLUX']*.01)
        reference_stars.append(value)
        star_info.append({'hr': hr, 'name': name, 'source': 'CALSPEC', 'url': url,
                          'raw_absolute_flux': True, 'wavelength_medium': 'vacuum'})

    raw_path = fetch('bsc5.tdat.gz', 'https://heasarc.gsfc.nasa.gov/FTP/heasarc/dbase/tdat_files/heasarc_bsc5p.tdat.gz')
    text = gzip.decompress(raw_path.read_bytes()).decode()
    columns = text.split('line[1] = ', 1)[1].split('\n', 1)[0].split()
    raw_rows = [dict(zip(columns, line.strip().removesuffix('|').split('|'), strict=True))
                for line in text.split('<DATA>')[1].split('<END>')[0].strip().splitlines()]
    catalogue_ids = {int(r['hr']) for r in csv.DictReader((ROOT/'src/lightpollution/catalogues/bsc5.csv').open())}
    with (output/'bsc-colours.csv').open('w', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['hr', 'b_v', 'spectral_type', 'bv_uncertainty_flag', 'variability_id'])
        for row in sorted(raw_rows, key=lambda r: int(r['hr'])):
            if int(row['hr']) in catalogue_ids:
                writer.writerow([row['hr'], row['bv_color'], row['spect_type'], row['bv_uncert'], row['var_id']])

    # A small independent XP sample spanning blue, solar-colour and red stars.
    targets = []
    for low, high in [(-.2, .1), (.5, .8), (1.3, 1.6)]:
        candidates = [r for r in raw_rows if r['bv_color'] and low <= float(r['bv_color']) <= high
            and r['vmag'] and 4.5 < float(r['vmag']) < 6 and not r['var_id'] and not r['multiple']
            and float(r['dec']) > 0 and int(r['hr']) in catalogue_ids]
        targets += candidates[:3]
    cones = []
    positions = []
    for row in targets:
        dec = float(row['dec']) + float(row['pmdec'] or 0)*16/3600
        ra = float(row['ra']) + float(row['pmra'] or 0)*16/(3600*np.cos(np.deg2rad(float(row['dec']))))
        positions.append((ra, dec))
        cones.append(f"1=CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',{ra},{dec},0.0014))")
    query = ("SELECT TOP 50 source_id,ra,dec,phot_g_mean_mag,phot_bp_mean_mag,phot_rp_mean_mag,has_xp_sampled "
             "FROM gaiadr3.gaia_source WHERE has_xp_sampled='true' AND ("+' OR '.join(cones)+')')
    url = requests.Request('GET', 'https://gea.esac.esa.int/tap-server/tap/sync', params={
        'REQUEST': 'doQuery', 'LANG': 'ADQL', 'FORMAT': 'csv', 'QUERY': query}).prepare().url
    matches = list(csv.DictReader(fetch('gaia-matches.csv', url).open()))
    selected, rejected = [], []
    for row, (ra, dec) in zip(targets, positions, strict=True):
        neighbours = []
        for item in matches:
            separation = np.hypot((float(item['ra'])-ra)*np.cos(np.deg2rad(dec)), float(item['dec'])-dec)*3600
            if separation <= 2 and abs(float(item['phot_g_mean_mag'])-float(row['vmag'])) < 3:
                neighbours.append((separation, item))
        if len(neighbours) != 1:
            rejected.append({'hr': int(row['hr']), 'reason': 'no_unique_XP_match_within_2arcsec'})
            continue
        separation, item = neighbours[0]
        selected.append((row, separation, item))

    def spectrum(item):
        row, separation, match = item
        source_id = match['source_id']
        url = ('https://gea.esac.esa.int/data-server/data?RETRIEVAL_TYPE=XP_SAMPLED&ID='+source_id
               +'&RELEASE=Gaia+DR3&FORMAT=votable')
        path = fetch('gaia-'+source_id+'.xml', url)
        table = parse(path).get_first_table().array
        value = np.interp(axis, table['wavelength'], table['flux'])
        if np.any(~np.isfinite(value)) or np.any(value < 0):
            return None, {'hr': int(row['hr']), 'reason': 'invalid_visible_spectral_samples'}
        return value, {'hr': int(row['hr']), 'name': row['name'], 'source': 'Gaia DR3 XP_SAMPLED',
            'source_id': source_id, 'matching_epoch': 'J2016, BSC tangent proper motion',
            'match_separation_arcsec': separation, 'catalogue_v_mag': float(row['vmag']),
            'catalogue_b_v': float(row['bv_color']), 'url': url, 'raw_absolute_flux': True,
            'sample_errors_correlated': True, 'wavelength_medium': 'vacuum'}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for value, info in pool.map(spectrum, selected):
            if value is None:
                rejected.append(info)
            else:
                reference_stars.append(value)
                star_info.append(info)
    arrays = {'wavelength_nm': axis, **{'lamp_'+k: v for k, v in templates.items()},
              **{'response_'+k: v for k, v in responses.items()}, 'star_flux_W_m2_nm': np.array(reference_stars)}
    with (output/'library.npz').open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    provenance = {'schema_version': 1, 'grid_nm': [360, 830, 5], 'lamp_power_normalisation':
        'Unit integrated radiant power on this finite wavelength interval, not unit lumens.',
        'cie_tables': rights, 'filters': filters, 'stars': star_info, 'gaia_rejections': rejected,
        'gaia_targets': len(targets), 'inputs': sorted(inputs, key=lambda v: v['file']),
        'vega_reference_magnitudes': {'V': .03, 'B': .03},
        'zero_point_convention': 'CALSPEC Vega anchored at B=V=0.03; not the SVO tabulated scalar zero point.',
        'limitations': ['5 nm quadrature; air/vacuum wavelength distinction not corrected at this resolution.',
            'Lamp shapes zero outside their tabulated support; no full VIIRS near-IR band inversion.',
            'BSC B-V continuum is approximate, includes reddening; not a measured effective temperature.',
            'Gaia sampled errors are correlated; no independent-bin uncertainty estimate.'],
        'attribution': ['CIE 2018/2019, CC BY-SA 4.0 for CIE-derived arrays.',
            'NASA HEASARC BSC5, Hoffleit & Warren 1991.', 'STScI CALSPEC, Bohlin et al. 2022.',
            'ESA Gaia DR3, processed by DPAC; Gaia Collaboration et al. 2023; Montegriffo et al. 2023.'],
        'sha256': {name: hashlib.sha256((output/name).read_bytes()).hexdigest()
                   for name in ['library.npz', 'bsc-colours.csv']}}
    (output/'provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'lamps': len(templates), 'responses': len(responses), 'measured_stars': len(star_info),
                      'gaia_targets': len(targets), 'gaia_rejections': rejected}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=ROOT/'scratch/spectral-data-20261111')
    parser.add_argument('--output', type=Path, default=ROOT/'src/lightpollution/resources/spectra')
    args = parser.parse_args()
    build(args.cache, args.output)
