"""Compute conditional SQM residuals from archived Veneto all-order runs."""

import argparse
from pathlib import Path

from astropy.io import fits
import numpy as np

from lightpollution.adapters.spectral_library import load_spectral_library
from lightpollution.adapters.storage import file_hash, read_json, write_json
from lightpollution.models.spectra import ARCSEC2_PER_SR, C, band_integral


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / 'artifacts'
TRANSPORT = ARTIFACTS / 'photometer-three-models-20261126' / 'results.json'
SELECTION = ARTIFACTS / 'photometer-aod-selection-20261126' / 'selection-audit.json'
REFERENCE = ARTIFACTS / 'photometer-sqm-residuals-20261126'
VEGA_MAG = 0.03
VEGA_HASH = '60aebf5c193223f69061cd176d6309730c3210051fffad0dd6ad44475199ceaa'
SKY_HASH = '9cc16c055bd007e0faff6a29df773a2fac91a9d16f72fe0b557f169a02594cec'
LAMP_TEMPLATES = ('LED-B2', 'LED-B3', 'HP1')
BASE_SPECTRUM = 'equal_DNB_LED-B2_HP1'


def load_fits_spectrum(path, expected_hash, spectral_density):
    if file_hash(path) != expected_hash:
        raise ValueError(f'The fixed reference spectrum has changed: {path}.')
    with fits.open(path) as hdus:
        wave = hdus[1].data['wavelength'] / 10  # Angstrom to nm
        flux = hdus[1].data['flux'].astype(float)
    return wave, spectral_density(wave, flux)


def spectral_projection(vega_path, sky_path):
    library = load_spectral_library()
    wave, sqm_a, sqm_b = np.loadtxt(REFERENCE / 'sqm-response.csv', delimiter=',', skiprows=1, unpack=True)
    dnb_wave, dnb = np.loadtxt(REFERENCE / 'viirs-dnb-rsr.dat', unpack=True)
    vega_wave, vega = load_fits_spectrum(vega_path, VEGA_HASH, lambda _, flux: flux * .01)
    # Photon densities become energy densities after dividing by wavelength. The
    # common constants cancel in the projection of the V-normalized sky spectrum.
    sky_wave, sky = load_fits_spectrum(sky_path, SKY_HASH, lambda wave, flux: flux / wave)
    lamp_wave = np.unique(np.r_[library.wavelengths, wave, dnb_wave])
    lamp_wave = lamp_wave[(lamp_wave >= 380) & (lamp_wave <= 780)]
    dnb_lamp = np.interp(lamp_wave, dnb_wave, dnb, left=0, right=0)
    lamp_spectra = {name: np.interp(lamp_wave, library.wavelengths, library.templates[name])
                    for name in LAMP_TEMPLATES}
    # DNB-radiance proportions define the mixture before comparing with SQM.
    lamp_spectra = {name: spectrum / band_integral(lamp_wave, spectrum, dnb_lamp)
                    for name, spectrum in lamp_spectra.items()}
    lamp_spectra[BASE_SPECTRUM] = (lamp_spectra['LED-B2'] + lamp_spectra['HP1']) / 2
    v_band = np.interp(sky_wave, library.wavelengths, library.responses['johnson_v'], left=0, right=0)
    sky_v = band_integral(sky_wave, sky, v_band)
    v_zero = library.zero_v * ARCSEC2_PER_SR
    bands = {}
    for name, curve in (('mean', (sqm_a + sqm_b) / 2), ('2370', sqm_a), ('2747', sqm_b)):
        vega_band = np.interp(vega_wave, wave, curve, left=0, right=0)
        sky_band = np.interp(sky_wave, wave, curve, left=0, right=0)
        lamp_band = np.interp(lamp_wave, wave, curve, left=0, right=0)
        # 3631 Jy flat in frequency converted to W/m²/nm/arcsec² at AB magnitude 0.
        ab_wave = np.linspace(wave[0], wave[-1], 6501)
        ab_flux = 3631e-26 * C / (ab_wave * 1e-9)**2 * 1e-9
        zero_ab = band_integral(ab_wave, ab_flux, np.interp(ab_wave, wave, curve)) * ARCSEC2_PER_SR
        bands[name] = {
            'vega_zero_W_m2_sr': float(band_integral(vega_wave, vega, vega_band) * ARCSEC2_PER_SR
                                       * 10**(.4 * VEGA_MAG)),
            'ab_zero_W_m2_sr': float(zero_ab),
            'natural_SQM_over_V': float(band_integral(sky_wave, sky, sky_band) / sky_v),
            'artificial_SQM_over_DNB': {lamp: float(band_integral(lamp_wave, spectrum, lamp_band))
                                       for lamp, spectrum in lamp_spectra.items()},
        }
    np.testing.assert_allclose([bands[name]['ab_zero_W_m2_sr'] for name in ('2370', '2747')],
                               [433.9, 415.4], rtol=.005)
    return v_zero, bands


def score(mag_error, pct_error):
    return {
        'bias_pct': float(pct_error.mean()),
        'mae_pct': float(np.abs(pct_error).mean()),
        'bias_mag': float(mag_error.mean()),
        'mae_mag': float(np.abs(mag_error).mean()),
        'rmse_mag': float(np.sqrt((mag_error**2).mean())),
        'median_pct': float(np.median(pct_error)),
        'p10_p90_pct': np.quantile(pct_error, [.1, .9]).tolist(),
    }


def compare(vega_path, sky_path):
    transport = read_json(TRANSPORT)
    selection = read_json(SELECTION)['station_nights']
    runs = transport['results']
    if transport['selection_sha256'] != file_hash(SELECTION) or runs.keys() != selection.keys():
        raise ValueError('Transport and photometry must cover the same selected windows.')
    keys = sorted(runs)
    rows = [runs[key] for key in keys]
    for key, row in zip(keys, rows, strict=True):
        selected = selection[key]
        if any(row[name] != selected[name] for name in ('site', 'sqm', 'n_sqm')):
            raise ValueError(f'SQM input differs in the archived run: {key}.')
        natural = row['natural_v']
        if row['moon_v_relative'] != 0 or natural['twilight_relative'] != 0 or not natural['background_supported']:
            raise ValueError(f'This projection requires a supported moonless deep-night background: {key}.')
    labels = transport['models']
    observed = np.array([r['sqm'] for r in rows])
    simulated_dnb = np.array([[r['artificial'][m]['all_order_radiance'][0] for m in labels] for r in rows])
    natural_v = np.array([r['natural_v']['natural_relative'] for r in rows])
    v_zero, bands = spectral_projection(vega_path, sky_path)
    groups = {
        'all': keys,
        'ideal': [k for k in keys if selection[k]['tier'] == 'ideal'],
        'nearly_ideal': [k for k in keys if selection[k]['tier'] == 'nearly_ideal'],
        'monthly': [k for k in keys if len(runs[k]['map_period']) == 7],
        'annual': [k for k in keys if len(runs[k]['map_period']) == 4],
        'bright_proxy': [k for k in keys if selection[k]['natural_plus_moon_fraction_proxy']['21.35'] <= .5],
    }
    groups.update({site: [k for k in keys if runs[k]['site'] == site] for site in sorted({r['site'] for r in rows})})
    conditions = {
        'reference': (BASE_SPECTRUM, 'mean', 21.6, 0),
        **{spectrum: (spectrum, 'mean', 21.6, 0) for spectrum in LAMP_TEMPLATES},
        'brighter_natural': (BASE_SPECTRUM, 'mean', 20.99, 0),
        'darker_natural': (BASE_SPECTRUM, 'mean', 22.10, 0),
        'LED-B3_brighter_natural': ('LED-B3', 'mean', 20.99, 0),
        'sqm_2370': (BASE_SPECTRUM, '2370', 21.6, 0),
        'sqm_2747': (BASE_SPECTRUM, '2747', 21.6, 0),
        'zero_point_plus_0.10_mag': (BASE_SPECTRUM, 'mean', 21.6, .10),
        'zero_point_minus_0.10_mag': (BASE_SPECTRUM, 'mean', 21.6, -.10),
    }
    scenarios = {}
    window_results = {}
    for name, (lamp, response, background_v_mag, zero_offset_mag) in conditions.items():
        band = bands[response]
        art = simulated_dnb * 1e-5 * band['artificial_SQM_over_DNB'][lamp]
        nat = (natural_v * v_zero * 10**(-.4 * (22 + background_v_mag - 21.6))
               * band['natural_SQM_over_V'])
        sqm_zero = band['vega_zero_W_m2_sr'] * 10**(.4 * zero_offset_mag)
        predicted = -2.5 * np.log10((art + nat[:, None]) / sqm_zero)
        natural_sqm_mag = -2.5 * np.log10(nat / sqm_zero)
        mag_error = predicted - observed[:, None]
        pct_error = (10**(-.4 * mag_error) - 1) * 100
        stats = {}
        for group, group_keys in (groups.items() if name == 'reference' else [('all', keys)]):
            mask = np.isin(keys, group_keys)
            stats[group] = {'windows': len(group_keys), 'nights': len({runs[k]['night'] for k in group_keys}),
                            'original_sqm_readings': sum(runs[k]['n_sqm'] for k in group_keys),
                            'models': {m: score(mag_error[mask, j], pct_error[mask, j])
                                       for j, m in enumerate(labels)}}
        scenarios[name] = {'lamp': lamp, 'response': response, 'natural_v_mag_arcsec2': background_v_mag,
                           'natural_sqm_mag_arcsec2_range': [float(natural_sqm_mag.min()),
                                                           float(natural_sqm_mag.max())],
                           'sqm_zero_offset_mag': zero_offset_mag, 'metrics': stats}
        if name == 'reference':
            for i, (key, row) in enumerate(zip(keys, rows, strict=True)):
                window_results[key] = {
                    'site': row['site'], 'night': row['night'], 'utc': row['utc'], 'map_period': row['map_period'],
                    'observed_mag_sqm': row['sqm'], 'n_sqm': row['n_sqm'], 'natural_sqm_W_m2_sr': float(nat[i]),
                    'models': {m: {'artificial_sqm_W_m2_sr': float(art[i, j]),
                                   'predicted_mag_sqm': float(predicted[i, j]),
                                   'error_mag': float(mag_error[i, j]), 'error_pct': float(pct_error[i, j])}
                               for j, m in enumerate(labels)},
                }
    return {'input_sha256': {'transport': file_hash(TRANSPORT), 'selection': file_hash(SELECTION),
                             'vega': VEGA_HASH, 'sky': SKY_HASH, 'spectral_library': load_spectral_library().identity,
                             'DNB': file_hash(REFERENCE / 'viirs-dnb-rsr.dat'),
                             'SQM': file_hash(REFERENCE / 'sqm-response.csv')},
            'johnson_v_vega_zero_W_m2_sr': v_zero, 'band_projections': bands, 'scenarios': scenarios,
            'reference_windows': window_results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('vega', type=Path, help='CALSPEC alpha_lyr_stis_011.fits')
    parser.add_argument('sky', type=Path, help='speXtra Paranal night sky vLR.fits')
    parser.add_argument('--output', type=Path, default=REFERENCE / 'results.json')
    args = parser.parse_args()
    write_json(args.output, compare(args.vega, args.sky))


if __name__ == '__main__':
    main()
