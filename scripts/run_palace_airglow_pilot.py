"""Evaluate the published Paranal airglow model in the project's photometer bands."""

import argparse
from datetime import date, timedelta
from pathlib import Path
from statistics import mean

import numpy as np

from lightpollution.adapters.spectral_library import load_spectral_library
from lightpollution.adapters.storage import read_json, write_json


ROOT = Path(__file__).resolve().parents[1] / 'artifacts'
MULTICITY = ROOT / 'multicity-propagation-20261126'
INPUTS = (
    ('Default test', None, 0, 0),
    ('Madrid Feb', '2012-02-20', 2, 1.5),
    ('Madrid Dec3 evening', '2013-12-03', 12, 21.0),
    ('Madrid Dec29', '2013-12-29', 12, 1.5),
    ('Tucson Mar31', '2019-03-31', 3, 0.0),
    ('Tucson Apr3', '2019-04-03', 4, 0.0),
    ('Tucson Apr7', '2019-04-07', 4, 0.0),
    ('Tucson Apr8', '2019-04-08', 4, 0.0),
    ('April 20 h', '2019-04-03', 4, 20.0),
    ('April 04 h', '2019-04-03', 4, 4.0),
)


def solar_flux(path):
    flux = {}
    with path.open() as stream:
        for line in stream:
            if not line.strip() or line.startswith('#'):
                continue
            cols = line.split()
            flux[date(*map(int, cols[:3]))] = float(cols[-3])
    return flux


def mean_solar_flux(day, flux):
    centre = date.fromisoformat(day)
    values = [flux[centre + timedelta(days=offset)] for offset in range(-13, 14)]
    if min(values) <= 0 or not all(np.isfinite(values)):
        raise ValueError(f'Invalid observed GFZ F10.7 sample near {day}.')
    return mean(values)


def optical_radiance(rayleigh_per_nm, wavelength_nm):
    photon_energy = 6.62607015e-34 * 299792458 / (wavelength_nm * 1e-9)
    return rayleigh_per_nm * 1e10 / (4 * np.pi) * photon_energy


def integrate_band(wave, radiance, response_wave, response):
    efficiency = np.interp(wave, response_wave, response, left=0, right=0)
    return float(np.trapezoid(radiance * efficiency, wave))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('gfz', type=Path, help='GFZ Kp/ap/SN/F10.7 ASCII file')
    parser.add_argument(
        'reference', type=Path, help='palace_test.dat from the published test.zip'
    )
    parser.add_argument('output', type=Path)
    args = parser.parse_args()

    from palace import palace

    args.output.mkdir(parents=True, exist_ok=True)
    flux = solar_flux(args.gfz)
    rel = load_spectral_library()
    photopic = rel.wavelengths, rel.responses['photopic']
    calibration = read_json(MULTICITY / 'results.json')['calibration']
    response_files = {
        'sqm': ROOT / 'photometer-sqm-residuals-20261126' / 'sqm-response.csv',
        'tess': MULTICITY / 'tess-response.csv',
    }
    outputs, spectra = [], {}
    responses = {}
    for band, path in response_files.items():
        curve = np.loadtxt(path, delimiter=',', skiprows=1).T
        responses[band] = curve[0], curve[1:].mean(axis=0)
    for label, day, month, hour in INPUTS:
        f107 = mean_solar_flux(day, flux) if day else 100.0
        parameters = palace.parlist(
            [f'mbin={month}', f'srf={f107}', 'lammin=0.35', 'lammax=0.95']
        )
        # Integer 0 asks for the whole night. Float 0.0 selects local midnight.
        parameters['tbin'] = hour
        data = palace.model(**parameters)
        wave = np.asarray(data['lam'], dtype=float) * 1000
        radiance = optical_radiance(np.asarray(data['flux'], dtype=float), wave)
        record = {
            'label': label,
            'example_input_date': day,
            'month': month,
            'local_solar_hour': hour if day else None,
            'F107_27d_sfu': f107,
            'photopic_mcd_m2': 683000 * integrate_band(wave, radiance, *photopic),
        }
        for band, response in responses.items():
            signal = integrate_band(wave, radiance, *response)
            record[f'{band}_W_m2_sr'] = signal
            record[f'{band}_equivalent_mag'] = float(
                -2.5 * np.log10(signal / calibration[band]['vega_zero_W_m2_sr'])
            )
        outputs.append(record)
        spectra[label] = wave, radiance

    archive = np.genfromtxt(args.reference, names=True)
    window = (archive['lam'] >= 0.35 - 1e-8) & (archive['lam'] <= 0.95 + 1e-8)
    reference_wave = archive['lam'][window] * 1000
    reference_radiance = optical_radiance(archive['flux'][window], reference_wave)
    reference = 683000 * integrate_band(reference_wave, reference_radiance, *photopic)
    relative_error = abs(reference / outputs[0]['photopic_mcd_m2'] - 1)
    if relative_error >= 0.002:
        raise ValueError(f'Published PALACE control differs by {relative_error:.3%}.')

    write_json(
        args.output / 'palace-study.json',
        {
            'site': 'Paranal',
            'model_version': 'PALACE 1.0',
            'records': outputs,
            'published_default_integral_mcd_m2': reference,
            'published_default_relative_error': relative_error,
            'comparison': 'Sensitivity at Paranal. Historical dates supply proxy inputs only.',
            'wavelength_window_nm': [350, 950],
            'zenith_deg': 0,
            'pwv_mm': 2.5,
        },
    )
    np.savez_compressed(
        args.output / 'palace-plot.npz',
        wave=spectra['Default test'][0],
        default=spectra['Default test'][1],
        december=spectra['Madrid Dec29'][1],
        april=spectra['Tucson Apr3'][1],
    )


if __name__ == '__main__':
    main()
