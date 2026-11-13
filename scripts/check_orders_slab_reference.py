"""Independent deterministic slab reference for orders 1..5.

Isotropic scattering, optical depth 1, unit Lambertian bottom radiance and
absorbing boundaries for scattered light. Cell-constant source iteration is
independent of the spherical Monte Carlo algorithm. Curvature is bounded by
using a 1 m thin shell; this is a validation configuration, not Earth's air.
"""

import argparse
from pathlib import Path

import numpy as np

from lightpollution.adapters.storage import file_hash, write_json
from lightpollution.models.scattering_orders import LambertianCap, OrderSettings, integrate_orders
from lightpollution.physics import EARTH_RADIUS_M, Atmosphere, Observer

ROOT = Path(__file__).resolve().parents[1]


def slab_orders(layers, angular_order, max_order=5, depth=1., albedo=1., view_mu=1.):
    x, weight = np.polynomial.legendre.leggauss(angular_order)
    mu, weight = (x+1)/2, weight/2
    edges = np.linspace(0, depth, layers+1)
    centre = (edges[1:]+edges[:-1])/2
    # u = optical height above ground. J0 is mean intensity from direct
    # ground emission. The angular quadrature integrates the upper hemisphere.
    direct_mean = .5*(np.exp(-centre[:, None]/mu) @ weight)
    delta = abs(centre[:, None]-centre[None, :])
    matrix = np.zeros_like(delta)
    half = depth/(2*layers)
    off_diagonal = ~np.eye(layers, dtype=bool)
    for m, w in zip(mu, weight, strict=True):
        # Exact integral of exp(-|u-u'|/mu)/mu over every source cell.
        term = np.exp(-np.maximum(0, delta-half)/m)*(-np.expm1(-2*half/m))
        matrix[off_diagonal] += .5*w*term[off_diagonal]
        np.fill_diagonal(term, 2*(-np.expm1(-half/m)))
        matrix[np.diag_indices(layers)] += .5*w*np.diag(term)
    escape = np.exp(-edges[:-1]/view_mu)*(-np.expm1(-np.diff(edges)/view_mu))
    source = albedo*direct_mean
    orders = []
    for _ in range(max_order):
        orders.append(float(source @ escape))
        source = albedo*(matrix @ source)
    return orders


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/benchmark-baseline-20261113/scattering/slab-reference.json')
    args = parser.parse_args()
    references = [{'layers': n, 'angular_order': q, 'orders': slab_orders(n, q)}
        for n, q in [(128, 32), (256, 64), (512, 128)]]
    atmosphere = Atmosphere(rayleigh_multiplier=0, aerosol_scale_height_m=1e6,
        aerosol_optical_depth=1e6, aerosol_single_scattering_albedo=1, aerosol_g=0, top_m=1.)
    mc = integrate_orders(Observer(0, 0, 0), LambertianCap(0, 0, np.pi*EARTH_RADIUS_M),
        0, 90, atmosphere, OrderSettings(samples=524288, max_order=5, seed=9901))
    ref = np.array(references[-1]['orders'])
    z = (np.array(mc['exact_order_radiance'])-ref)/mc['exact_order_standard_error']
    refinement = abs(np.array(references[-2]['orders'])/ref-1)
    report = {'kind': 'Independent layered isotropic slab solution versus thin spherical shell Monte Carlo.',
        'references': references, 'spherical_MC': mc, 'orderwise_z': z.tolist(),
        'max_deterministic_relative_refinement': float(max(refinement)),
        'all_orders_within_four_MC_standard_errors': bool(np.all(abs(z) < 4)),
        'shell_thickness_m': 1., 'curvature_to_planar_geometry_identical': False,
        'exponential_coefficient_change_across_shell': float(1-np.exp(-1e-6)),
        'not_an_atmosphere_or_field_accuracy_test': True,
        'code_sha256': {p: file_hash(ROOT/p) for p in ['scripts/check_orders_slab_reference.py',
            'src/lightpollution/models/scattering_orders.py']}}
    write_json(args.output, report)
    print({k:v for k,v in report.items() if k in ['orderwise_z', 'max_deterministic_relative_refinement', 'all_orders_within_four_MC_standard_errors']})


if __name__ == '__main__':
    main()
