"""Prevent accidental reuse of a satellite region for another photo case."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from lightpollution.data import Bounds, save_region


@pytest.mark.parametrize('stored_year,stored_west', [(2025,4.5), (2014,4.0)])
def test_photo_case_rejects_region_from_another_year_or_bounds(tmp_path, stored_year, stored_west):
    directory = tmp_path/'model'
    save_region(directory/'source-grid.npz', {}, Bounds(stored_west,48,9,50), {'year':stored_year})
    case = tmp_path/'case.json'
    case.write_text(json.dumps({'model_directory':str(directory), 'year':2014,
                               'bounds':{'west':4.5,'south':48,'east':9,'north':50}}))
    script = Path(__file__).resolve().parents[1]/'scripts/run_photo_case.py'
    result = subprocess.run([sys.executable,str(script),'--case',str(case)],
                            capture_output=True,text=True,check=False)
    assert result.returncode != 0
    assert 'Saved region does not match the case bounds/year' in result.stderr
