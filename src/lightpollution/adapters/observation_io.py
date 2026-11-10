"""Legacy JSON/CSV compatibility and explicit NPZ/catalogue loading."""

import json
from pathlib import Path

import numpy as np

from ..domain.fields import ObservationResources, TransferGrid
from ..download import sha256
from ..skyview import load_stars


def load_observation_resources(config: dict, base_directory: Path):
    grids = {}
    if not isinstance(config, dict):
        return ObservationResources()
    artificial = config.get("artificial", {})
    if isinstance(artificial, dict) and artificial.get("mode") == "grid":
        for key in ["sky_grid", "reference_sky_grid"]:
            name = artificial.get(key)
            if not isinstance(name, str) or not name.strip():
                continue
            path = Path(name)
            if not path.is_absolute():
                path = base_directory / path
            with np.load(path, allow_pickle=False) as data:
                grids[name] = TransferGrid(np.array(data["azimuth_deg"]), np.array(data["altitude_deg"]),
                    np.array(data["radiance"]), json.loads(str(data["metadata_json"].item())), sha256(path))
    catalogue = load_stars() if config.get("mode") == "dated" else None
    return ObservationResources(grids, catalogue)


def write_observation(report: dict, output: Path):
    """Validate serialisability before creating outputs; no NaN/Infinity in JSON."""
    import csv
    stars = report['stars']
    summary = {key:value for key,value in report.items() if key != 'stars'}
    serialized = json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    output.mkdir(parents=True,exist_ok=True)
    (output/'observation.json').write_text(serialized)
    for name,rows in [('directions',report['directions']),('stars',stars)]:
        columns = list(rows[0]) if rows else ['hr','name','azimuth_deg','altitude_deg',
                    'catalogue_v_mag','extinction_v_mag','observed_v_mag','flux_relative_to_v_mag0']
        with (output/f'{name}.csv').open('w',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=columns,lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
