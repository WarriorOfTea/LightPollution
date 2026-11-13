import pytest

from lightpollution.adapters.winkler import read_winkler


def html(label='SP 385', moon='45.2', magnitude='19.87', sign='-', colour='-'):
    def row(values):
        return '<tr>'+''.join('<td>'+v+'</td>' for v in values)+'</tr>'
    headers = row(['header'])+row(['units'])
    stars = ['00 07 37', '<math alttext="'+sign+'"><mo>−</mo><annotation>-</annotation></math>86 02 20',
        'HD 385', 'B9IV', '[W97]', '+5', '+1.5']
    sky = [label, '225.3840', '0.7745', moon, '300.5', '53.6', '180.9', '82.15', magnitude, colour, '-0.38', '0.18', '0.99']
    return ('<figure id="A1.T10"><table>'+headers+row(stars)+row(['Notes:'])+row(['attribution'])
        +'</table></figure><figure id="A1.T13"><table>'+headers+row(sky)+'</table></figure>')


def test_published_angles_phase_magnitude_and_math_sign_are_not_duplicated():
    rows = read_winkler(html())
    assert len(rows) == 1
    row = rows[0]
    assert row['ra_deg'] == pytest.approx((7/60+37/3600)*15+5/240)
    assert row['dec_deg'] == pytest.approx(-(86+2/60+20/3600)+1.5/60)
    assert row['v_mag_arcsec2'] == 19.87
    assert row['moon_zenith_deg'] == 45.2
    assert row['b_minus_v'] is None
    assert row['coordinate_status'] == 'published'


def test_absent_coordinates_or_moon_geometry_are_explicit_missing_data():
    row = read_winkler(html(label='SP 97991', moon='-'))[0]
    assert row['ra_deg'] is None and row['dec_deg'] is None
    assert row['moon_zenith_deg'] is None
    assert row['coordinate_status'] == 'absent_from_published_table_10'


@pytest.mark.parametrize('magnitude', ['-', 'nan', 'infinity', 'broken'])
def test_missing_or_nonfinite_measured_brightness_cannot_be_scored(magnitude):
    with pytest.raises(ValueError):
        read_winkler(html(magnitude=magnitude))


def test_missing_appendix_is_a_failure():
    with pytest.raises(ValueError):
        read_winkler('<p>Only an abstract, not a data table</p>')
