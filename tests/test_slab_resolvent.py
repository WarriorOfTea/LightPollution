import json
from pathlib import Path

import numpy as np
import pytest

from lightpollution.models.slab_resolvent import SlabSettings, slab_system, solve_slab
from lightpollution.physics import hg_phase, rayleigh_phase


def test_linear_system_agrees_with_certified_infinite_series_and_is_positive():
    r = solve_slab(SlabSettings(layers=64,angular_order=32))
    assert r['all_order_radiance'] > r['cumulative_radiance'][-1] > 0
    assert r['linear_vs_series_difference'] <= r['verification_series_tail_bound']+1e-14
    assert r['absolute_algebraic_residual_inf'] < 1e-13
    assert 0 < r['contraction_q_inf'] < 1
    assert r['finite_scattering_order_truncation'] is False
    assert r['explicit_inverse_formed'] is False


def test_historical_reference_first_five_orders_reproduced():
    root=Path(__file__).resolve().parents[1]
    old=json.loads((root/'artifacts/benchmark-baseline-20261113/scattering/slab-reference.json').read_text())
    r=solve_slab(SlabSettings(layers=128,angular_order=32))
    np.testing.assert_allclose(r['exact_order_radiance'],old['references'][0]['orders'],rtol=1e-13)


def test_first_order_against_independent_single_integral():
    settings=SlabSettings(layers=512,angular_order=128,optical_depth=1,single_scattering_albedo=.9,view_cosine=.7)
    r=solve_slab(settings)
    x,w=np.polynomial.legendre.leggauss(256);m=(x+1)/2;w=w/2
    first=.5*.9*np.dot(w,m/(m+.7)*(-np.expm1(-(1/.7+1/m))))
    assert r['exact_order_radiance'][0] == pytest.approx(first,rel=1e-4)


def test_pure_absorption_has_no_scattered_radiance():
    settings=SlabSettings(layers=16,angular_order=8,single_scattering_albedo=0)
    r=solve_slab(settings)
    assert r['all_order_radiance'] == 0
    assert r['verification_series_radiance'] == 0
    assert r['contraction_q_inf'] == 0


def test_phase_convolution_legendre_eigenvalues():
    x,w=np.polynomial.legendre.leggauss(512)
    for degree in range(7):
        c=np.zeros(degree+1);c[-1]=1
        basis=np.polynomial.legendre.legval(x,c)
        hg=2*np.pi*np.dot(w,hg_phase(x,.8)*basis)
        rayleigh=2*np.pi*np.dot(w,rayleigh_phase(x)*basis)
        assert hg == pytest.approx(.8**degree,abs=2e-12)
        assert rayleigh == pytest.approx({0:1.,2:.1}.get(degree,0.),abs=2e-12)


@pytest.mark.parametrize('kwargs',[{'layers':3},{'angular_order':True},{'optical_depth':0},
    {'single_scattering_albedo':1.1},{'view_cosine':0},{'series_absolute_tolerance':-1}])
def test_invalid_slab_settings(kwargs):
    with pytest.raises(ValueError):
        SlabSettings(**kwargs)


def test_uncertified_series_does_not_return_a_falsely_converged_answer():
    with pytest.raises(RuntimeError,match='did not achieve'):
        solve_slab(SlabSettings(layers=16,angular_order=8,series_safety_iterations=1))
