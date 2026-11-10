import csv
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from lightpollution.download import download_composite, select_files
from lightpollution.ground import audit_astmon, audit_galicia
from lightpollution.observation import observation_context
from lightpollution.periods import CompositePeriod
from lightpollution.photometry import PhotometricBand, SurfaceBrightness, sum_surface_brightness
from lightpollution.physics import Observer


def test_month_period_handles_leap_year_and_keeps_january_distinct_from_annual():
    february = CompositePeriod(2024, 2)
    assert february.start.isoformat() == '2024-02-01'
    assert february.end.isoformat() == '2024-02-29'
    assert february.day_of_year == '032'
    assert CompositePeriod(2024, 3).day_of_year == '061'
    assert CompositePeriod(2025, 7).day_of_year == '182'
    assert CompositePeriod(2025, 1).product == 'VNP46A3'
    assert CompositePeriod(2025).product == 'VNP46A4'
    assert CompositePeriod(2025, 1).label != CompositePeriod(2025).label
    with pytest.raises(ValueError): CompositePeriod(2025, 13)


def test_monthly_catalogue_does_not_select_an_annual_or_another_month_file():
    names = ['VNP46A4.A2025001.h21v03.002.2026001000000.h5',
             'VNP46A3.A2025001.h21v03.002.2026001000000.h5',
             'VNP46A3.A2025032.h21v03.002.2026001000000.h5']
    records = [{'name': name} for name in names]
    assert select_files(records, 2025, ('h21v03',), month=1) == [records[1]]
    assert select_files(records, 2025, ('h21v03',)) == [records[0]]


def test_download_rejects_reusing_an_annual_manifest_for_a_month(tmp_path, monkeypatch):
    manifest = {'product': 'VNP46A4', 'collection': '002', 'year': 2025,
                'tiles': ['h21v03'], 'files': []}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    def no_network(*args, **kwargs):
        pytest.fail('A mismatched manifest must fail before network access')
    monkeypatch.setattr('lightpollution.download.requests.get', no_network)
    with pytest.raises(ValueError, match='period'):
        download_composite(tmp_path, 2025, ('h21v03',), month=1)


def test_same_band_light_adds_in_flux_and_preserves_ratio():
    band = PhotometricBand('Johnson-V', 'Vega', 'specified response')
    one = SurfaceBrightness(np.array([20., 22.]), band)
    total = sum_surface_brightness(one, one)
    np.testing.assert_allclose(total.mag_arcsec2, one.mag_arcsec2-2.5*np.log10(2))
    np.testing.assert_allclose(total.ratio_to(one), 2)
    assert not one.mag_arcsec2.flags.writeable
    for other in [PhotometricBand('SQM', 'instrument', 'specified response'),
                  PhotometricBand('Johnson-V', 'AB', 'specified response'),
                  PhotometricBand('Johnson-V', 'Vega', 'different response')]:
        with pytest.raises(ValueError): sum_surface_brightness(one, SurfaceBrightness(21., other))
    with pytest.raises(ValueError): SurfaceBrightness(np.nan, band)


def test_magnitude_sum_is_stable_for_a_very_faint_component():
    band = PhotometricBand('test', 'relative', 'test response')
    result = sum_surface_brightness(SurfaceBrightness(20., band), SurfaceBrightness(2000., band))
    assert float(result.mag_arcsec2) == pytest.approx(20.)


@pytest.mark.parametrize('utc,expected_full', [('2025-03-14T07:00:00Z', True),
                                             ('2025-03-29T11:00:00Z', False)])
def test_moon_phase_at_known_full_and_new_moon_and_no_invented_radiance(utc, expected_full):
    result = observation_context(Observer(0, 0, 0), utc)
    fraction = result['moon']['geocentric_illuminated_fraction']
    assert fraction > .99 if expected_full else fraction < .01
    assert 340_000 < result['moon']['topocentric_distance_km'] < 420_000
    assert 0 <= result['moon']['target_separation_deg'] <= 180
    assert result['lunar_sky_radiance_modelled'] is False


def test_petersburg_summer_is_not_mislabelled_as_dark_and_utc_is_required():
    result = observation_context(Observer(), '2025-06-21T00:00:00Z')
    assert result['sun']['altitude_deg'] > -18
    assert not result['astronomical_darkness']
    assert not result['legacy_moonless_renderer_applicable']
    with pytest.raises(ValueError, match='UTC'):
        observation_context(Observer(), '2025-06-21T00:00:00')


def write_astmon(path: Path, records):
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream, delimiter=';')
        writer.writerow(['Datetime_UTC', 'Filter', 'Pixel_x', 'Magnitude', 'Error'])
        writer.writerows(records)


def test_ground_import_does_not_turn_finite_data_into_clear_moonless_observations(tmp_path):
    path = tmp_path/'astmon.csv'
    write_astmon(path, [['2015-01-01 00:00:00', 'Johnson-V', 'Pos0', 18., .02],
                       ['2015-01-01 00:01:00', 'Johnson-V', 'Pos0', 17., -.1],
                       ['2015-01-01 00:02:00', 'Johnson-V', 'Pos0', 'nan', .1]])
    report = audit_astmon(path)
    band = report['bands']['Johnson-V']
    assert report['rows'] == 3
    assert band['finite_magnitude_and_error'] == 2
    assert band['finite_with_positive_reported_error'] == 1
    assert report['rows_confirmed_clear_and_moonless_by_this_import'] == 0
    assert report['moon_flags_available'] is False and report['spatial_header_warning']
    write_astmon(path, [['2015-01-01 00:00:00', 'invented-filter', '', 18., .02]])
    with pytest.raises(ValueError, match='unknown native filter'): audit_astmon(path)


def test_galicia_offset_sign_and_zero_placeholder_handling(tmp_path):
    path = tmp_path/'galicia.zip'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('01_Test.txt', '01 01 2015 12 00 00 0.00\n01 01 2015 23 00 00 20.00\n')
        archive.writestr('Offsets.txt', "if strcmp(BaseNome,'01_Test')\n    Offset=-0.1\n    mag_corr=mag-Offset;\nend\n")
    report = audit_galicia(path)
    assert report['rows'] == 2 and report['positive_finite_native_readings'] == 1
    assert report['sites'][0]['corrected_native_mag_arcsec2']['median'] == pytest.approx(20.1)
    assert not report['clear_sky_flags_available']
