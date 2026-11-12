"""Reading copies must typeset equations while preserving literal code."""

import json
from pathlib import Path
import subprocess
import sys

from pypdf import PdfReader
import pytest


ROOT = Path(__file__).resolve().parents[1]


def export(source, target, audit):
    return subprocess.run([sys.executable, str(ROOT / 'scripts/build_report.py'),
        '--source', str(source), '--output', str(target), '--date', '2026-11-12',
        '--math-audit', str(audit)], capture_output=True, text=True, cwd=ROOT)


def test_reading_copy_typesets_math_and_keeps_code(tmp_path):
    source = tmp_path / 'source.md'
    source.write_text(r'''# Проверка формул

Свет имеет длину волны $\lambda$ и долю $\frac{1}{2}$.

$$
\begin{gathered}
\tau=\int_0^h\beta(s)\,\mathrm{d}s, \\
T=e^{-\tau}.
\end{gathered}
$$

```python
literal = "$not_math$"
```

Переменная `$literal$` остаётся кодом.
''')
    target, audit = tmp_path / 'reading.pdf', tmp_path / 'math.json'
    result = export(source, target, audit)
    assert result.returncode == 0, result.stderr
    record = json.loads(audit.read_text())
    assert record['unique_formulas'] == 4
    text = '\n'.join(page.extract_text() for page in PdfReader(target).pages)
    assert 'Проверка формул' in text
    assert 'literal = "$not_math$"' in text
    assert '$literal$' in text
    assert r'\frac' not in text and r'\begin{gathered}' not in text
    assert '\ufffd' not in text


@pytest.mark.parametrize(('body', 'message'), [
    ('$$\nL=1\n', 'Unclosed LaTeX display block'),
    ('$L=1\n', 'Unclosed LaTeX inline expression'),
])
def test_unclosed_math_fails_instead_of_printing_latex(tmp_path, body, message):
    source = tmp_path / 'source.md'
    source.write_text('# Формула\n\n' + body)
    result = export(source, tmp_path / 'reading.pdf', tmp_path / 'math.json')
    assert result.returncode != 0
    assert message in result.stderr
    assert not (tmp_path / 'reading.pdf').exists()


def test_unsupported_math_fails_instead_of_silently_omitting_formula(tmp_path):
    source = tmp_path / 'source.md'
    source.write_text('# Формула\n\n$\\not_a_math_command{x}$\n')
    result = export(source, tmp_path / 'reading.pdf', tmp_path / 'math.json')
    assert result.returncode != 0
    assert 'Cannot typeset LaTeX formula' in result.stderr
