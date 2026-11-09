"""Check selected rays of the photograph case independently of the display."""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from lightpollution.data import load_region
from lightpollution.physics import Atmosphere, Observer, SolverSettings, integrate_direction
from lightpollution.sources import regional_sources


def main():
    case = json.loads(Path('artifacts/photo-comparison/case.json').read_text())
    directory = Path(case['model_directory'])
    arrays, bounds, _ = load_region(directory/'source-grid.npz')
    observer = Observer(**case['observer'])
    atmosphere = Atmosphere(**case['atmosphere'])
    directions = [(145, 1), (285, 15), (0, 45), (0, 90)]
    with np.load(directory/'sky-grid.npz', allow_pickle=False) as grid:
        base = np.asarray([grid['radiance'][np.flatnonzero(grid['altitude_deg']==h)[0],
                                            np.flatnonzero(grid['azimuth_deg']==a)[0]]
                           for a, h in directions])
    runs = []
    for block in (case['block'], case['block']//2):
        settings = SolverSettings(384, 48)
        values, counts = [], []
        for azimuth, altitude in directions:
            sources, _ = regional_sources(
                arrays, bounds, observer, block, case['exclude_radius_m'],
                refinement_direction=(azimuth, altitude) if altitude<=5 else None)
            values.append(integrate_direction(observer, sources, azimuth, altitude,
                                              atmosphere, settings))
            counts.append(len(sources.power))
            print(f'block={block}, direction={azimuth, altitude} complete', flush=True)
        runs.append({'block':block, 'settings':asdict(settings),
                     'radiance':values, 'source_counts':counts})
    refined, finer = [np.asarray(r['radiance']) for r in runs]
    quadrature, aggregation = base/refined-1, refined/finer-1
    result = {
        'directions_az_alt_deg':directions, 'saved_grid_values':base.tolist(),
        'runs':runs, 'quadrature_relative_difference':quadrature.tolist(),
        'aggregation_relative_difference':aggregation.tolist(),
        'max_abs_quadrature_relative_difference':float(np.max(np.abs(quadrature))),
        'max_abs_aggregation_relative_difference':float(np.max(np.abs(aggregation))),
        'physical_accuracy_bound':None,
        'scope':'Four selected rays, including the sampled peaks at 1 and 15 degrees; '
                'not a hemisphere error bound or a physical validation.'}
    (directory/'numerical-check.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
