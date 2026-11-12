"""Typeset the documented LaTeX subset without a system TeX installation.

Markdown remains the editable source. Matplotlib's MathText renders formulas;
code fences are handled separately by build_report and never reach this module.
Unsupported syntax raises an error instead of producing a misleading PDF.
"""

from hashlib import sha256
from pathlib import Path
import re

from matplotlib.font_manager import FontProperties
from matplotlib.mathtext import math_to_image
from PIL import Image as RasterImage
from reportlab.platypus import Image, KeepTogether, Spacer


class MathRenderer:
    def __init__(self, directory: Path, width: float):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.width = width
        self.records = {}

    def raster(self, formula: str, size: float):
        key = sha256((str(size) + '\n' + formula).encode()).hexdigest()
        if key not in self.records:
            path = self.directory / (key + '.png')
            try:
                depth = math_to_image('$' + formula + '$', path,
                    prop=FontProperties(size=size, math_fontfamily='dejavusans'),
                    dpi=600, format='png', color='#13242f')
            except (ValueError, RuntimeError) as exc:
                raise ValueError(f'Cannot typeset LaTeX formula: {formula}') from exc
            with RasterImage.open(path) as picture:
                width, height = picture.size
            self.records[key] = {'latex': formula, 'font_size_pt': size,
                'path': str(path), 'width_pt': width * 72 / 600,
                'height_pt': height * 72 / 600, 'depth_pt': float(depth)}
        return self.records[key]

    def inline(self, formula: str, size=9.5):
        record = self.raster(formula, size)
        return (f'<img src="{record["path"]}" width="{record["width_pt"]:.4f}" '
                f'height="{record["height_pt"]:.4f}" valign="{-record["depth_pt"]:.4f}"/>')

    def display(self, source: str):
        source = source.strip()
        if source.startswith(r'\begin{gathered}'):
            if not source.endswith(r'\end{gathered}'):
                raise ValueError('Unclosed LaTeX gathered environment')
            source = source[len(r'\begin{gathered}'):-len(r'\end{gathered}')]
            rows = re.split(r'\\\\', source)
        else:
            rows = [source]
        flowables = [Spacer(1, 5)]
        for row in rows:
            formula = ' '.join(row.split())
            if not formula:
                continue
            record = self.raster(formula, 11.)
            scale = min(1., self.width / record['width_pt'])
            picture = Image(record['path'], width=record['width_pt'] * scale,
                            height=record['height_pt'] * scale)
            picture.hAlign = 'LEFT'
            flowables.extend([picture, Spacer(1, 6)])
        if len(flowables) == 1:
            raise ValueError('Empty LaTeX display block')
        return KeepTogether(flowables)
