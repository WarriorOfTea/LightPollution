"""Read the published numerical appendix, without treating model columns as truth.

Source: Winkler (2022), arXiv:2205.07773, Tables 10 and 13. No network or
dependencies beyond the standard library; rejected malformed rows fail closed.
"""

from html.parser import HTMLParser
import math
import re


class _Tables(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables = {}
        self.figure = None
        self.row = None
        self.cell = None
        self.math_depth = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'figure':
            self.figure = attrs.get('id')
        if not self.figure:
            return
        if tag == 'tr':
            self.row = []
        elif tag in ('td', 'th'):
            self.cell = []
        elif tag == 'math' and self.cell is not None:
            self.cell.append(attrs.get('alttext', ''))
            self.math_depth += 1

    def handle_endtag(self, tag):
        if tag == 'math' and self.math_depth:
            self.math_depth -= 1
        elif tag in ('td', 'th') and self.cell is not None:
            if self.row is not None:
                self.row.append(''.join(self.cell).strip())
            self.cell = None
        elif tag == 'tr' and self.row is not None:
            self.tables.setdefault(self.figure, []).append(self.row)
            self.row = None
        elif tag == 'figure':
            self.figure = None

    def handle_data(self, data):
        if self.cell is not None and not self.math_depth:
            self.cell.append(data)


def read_winkler(html: str):
    parser = _Tables()
    parser.feed(html)
    stars = {}
    for cells in parser.tables.get('A1.T10', [])[2:]:
        if cells == ['Notes:']:
            break
        if len(cells) != 7 or not cells[2].startswith('HD '):
            raise ValueError('Malformed Table 10 row; no silent coordinate substitution.')
        hours, minutes, seconds = map(float, cells[0].split())
        dec = cells[1].replace('−', '-')
        sign = -1 if dec.startswith('-') else 1
        degrees, arcmin, arcsec = map(float, dec.lstrip('+-').split())
        stars[cells[2]] = {'ra_deg': (hours+minutes/60+seconds/3600)*15 + float(cells[5])/240,
            'dec_deg': sign*(degrees+arcmin/60+arcsec/3600) + float(cells[6].replace('−', '-'))/60}
    if not stars or 'A1.T13' not in parser.tables:
        raise ValueError('The required published Tables 10 and 13 are missing.')
    records = []
    label = None
    names = ['hjd_minus_2456000', 'illuminated_fraction', 'moon_zenith_deg', 'moon_azimuth_deg',
        'target_zenith_deg', 'target_azimuth_deg', 'separation_deg', 'v_mag_arcsec2',
        'b_minus_v', 'u_minus_b', 'v_minus_r', 'v_minus_i']
    for row_number, cells in enumerate(parser.tables['A1.T13'][2:], start=1):
        if len(cells) != 13:
            raise ValueError('Unexpected Table 13 column count.')
        if cells[0]:
            label = cells[0]
        key = re.sub(r'^SP\s+', 'HD ', label or '')
        try:
            values = [None if (index in (2, 3) or index >= 8) and value.strip() in ('-', '−', '')
                else float(value.replace('−', '-')) for index, value in enumerate(cells[1:])]
        except ValueError as exc:
            raise ValueError(f'Malformed numerical Table 13 row {row_number}.') from exc
        record = dict(zip(names, values, strict=True))
        if any(value is not None and not math.isfinite(value) for value in values):
            raise ValueError('Nonfinite published photometry is not an admissible observation.')
        coordinates = stars.get(key, {'ra_deg': None, 'dec_deg': None})
        record.update(row_id=f'A1.T13-row-{row_number}', patch=label, **coordinates,
            coordinate_status='published' if key in stars else 'absent_from_published_table_10')
        if not (0 <= record['illuminated_fraction'] <= 1
                and 0 <= record['separation_deg'] <= 180
                and 0 <= record['target_zenith_deg'] <= 90):
            raise ValueError('Published angles or phase are outside their domain.')
        records.append(record)
    if not records:
        raise ValueError('No measured sky photometry was parsed.')
    return records
