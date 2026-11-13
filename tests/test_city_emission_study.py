"""Scientific invariants for the emission / vertical-profile experiment."""

from dataclasses import replace
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

from lightpollution.physics import Atmosphere

spec = importlib.util.spec_from_file_location('city_emission_study',
    Path(__file__).resolve().parents[1]/'scripts/study_city_emission_atmosphere.py')
study = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = study
spec.loader.exec_module(study)


def test_horizon_is_measured_off_radiance_including_exact_boundary():
    mu = np.cos(np.deg2rad([60., 70., 85., 90.]))
    mu[0] = .5
    np.testing.assert_array_equal(study.hybrid_radiance(3., 7., mu, .6), [7.]*4)
    assert study.hybrid_radiance(3., 7., np.nextafter(.5, 1.), .6) != 7.


@pytest.mark.parametrize('eta,ratio', [(0., 1.), (.47665967542229376, 1.0914098883744412),
                                       (.6335044020672363, 1.1734791822278412)])
def test_hybrid_flux_and_low_angle_fraction_against_independent_quadrature(eta, ratio):
    x, w = np.polynomial.legendre.leggauss(80)
    total = 0.
    for lo, hi in ((0., .5), (.5, 1.)):
        mu = lo+(x+1)*(hi-lo)/2
        total += np.sum(w*study.hybrid_radiance(1., ratio, mu, eta)*mu)*(hi-lo)
    expected = study.angular_integrals(eta, ratio)
    assert total == pytest.approx(expected['hybrid_power_over_near_lambertian'], rel=1e-12)
    cut = np.sin(np.deg2rad(25))
    mu = (x+1)*cut/2
    low = np.sum(w*study.hybrid_radiance(1., ratio, mu, eta)*mu)*cut
    assert low/total == pytest.approx(expected['hybrid_xi_25'], rel=1e-12)


@pytest.mark.parametrize('layers', [((0., 1000., 1.),), ((2000., 3000., 1.),),
                                      ((0., 1000., .5), (2000., 3000., .5))])
def test_same_aod_different_profile_and_valid_majorant(layers):
    atmosphere = study.LayerStudyAtmosphere(aerosol_layers=layers)
    total = 0.
    for lo, hi, fraction in layers:
        r, a, e = atmosphere.coefficients((lo+hi)/2)
        total += (e-r)*(hi-lo)
        assert a == pytest.approx((e-r)*atmosphere.aerosol_single_scattering_albedo)
    assert total == pytest.approx(.1, rel=1e-12)
    h = np.linspace(0., 80000., 20001)
    assert np.max(atmosphere.coefficients(h)[2]) <= atmosphere.extinction_majorant
    # Moving all aerosol above 2km leaves zero aerosol extinction at the eye,
    # although total AOD is unchanged. A surface-value majorant would be wrong.
    if layers[0][0] > 0:
        r, _, e = atmosphere.coefficients(0.)
        assert r == e
        assert atmosphere.extinction_majorant > e


@pytest.mark.parametrize('layers', [(), ((0., 1000., .5),), ((2000., 1000., 1.),),
                                      ((0., 2000., .5), (1000., 3000., .5))])
def test_reject_ill_defined_layer_profile(layers):
    with pytest.raises(ValueError):
        study.LayerStudyAtmosphere(aerosol_layers=layers)


def test_existing_majorant_is_unchanged_for_exponential_medium():
    atmosphere = Atmosphere()
    assert atmosphere.extinction_majorant == atmosphere.rayleigh_sea_level+atmosphere.aerosol_extinction_sea_level
    assert replace(atmosphere, aerosol_optical_depth=0., rayleigh_multiplier=0.).extinction_majorant == 0.
