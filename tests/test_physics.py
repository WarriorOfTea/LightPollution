from dataclasses import replace

import numpy as np
import pytest

from lightpollution.physics import (
    EARTH_RADIUS_M, Atmosphere, Observer, SolverSettings, Sources, cell_areas,
    emission, hg_phase, integrate_direction, quadrature, rayleigh_phase,
    source_to_point, unit_position,
)


def one_city(latitude=59.94, longitude=30.315, power=1.0):
    return Sources(EARTH_RADIUS_M*unit_position(np.array([latitude]),np.array([longitude])),np.array([power]))


@pytest.mark.parametrize("fraction",[0.0,0.1,0.5,1.0])
def test_emission_hemisphere_normalisation(fraction):
    mu, weights = quadrature(80)
    assert 2*np.pi*np.dot(weights,emission(mu,fraction)) == pytest.approx(1,rel=1e-12)
    assert np.all(emission(mu,fraction)>=0)
    assert emission(-0.1,fraction)==0


@pytest.mark.parametrize("g",[-0.8,0,0.8])
def test_phase_sphere_normalisation(g):
    x,w = np.polynomial.legendre.leggauss(256)
    assert 2*np.pi*np.dot(w,hg_phase(x,g)) == pytest.approx(1,rel=1e-10)
    assert 2*np.pi*np.dot(w,rayleigh_phase(x)) == pytest.approx(1,rel=1e-12)


def test_lambert_reflected_power():
    # A Lambertian surface with rho=0.3 and incident irradiance=2 W/m2.
    mu,w = quadrature(32)
    outgoing_radiance=0.3*2/np.pi
    assert 2*np.pi*np.dot(w,outgoing_radiance*mu)==pytest.approx(0.3*2)


def test_spherical_areas_cover_the_globe():
    latitudes=np.arange(-89.5,90,1)
    assert cell_areas(latitudes,1).sum()*360==pytest.approx(4*np.pi*EARTH_RADIUS_M**2)
    assert cell_areas(np.array([60]))[0]/cell_areas(np.array([0]))[0]==pytest.approx(0.5)


def test_cardinal_directions_and_zenith():
    o=Observer(0,0,0)
    np.testing.assert_allclose(o.direction(0,0),[0,0,1],atol=1e-15)
    np.testing.assert_allclose(o.direction(90,0),[0,1,0],atol=1e-15)
    np.testing.assert_allclose(o.direction(180,0),[0,0,-1],atol=1e-15)
    np.testing.assert_allclose(o.direction(123,90),[1,0,0],atol=1e-15)


def test_inverse_square_without_atmosphere():
    source=one_city(latitude=0,longitude=0)
    points=source.positions_m+np.array([[1000,0,0],[2000,0,0]])
    r2,_,mu=source_to_point(source,points)
    flux=source.power*emission(mu)/r2
    assert flux[0,0]/flux[1,0]==pytest.approx(4)


def test_forward_scattering_uses_photon_directions():
    observer=Observer(0,0,0)
    source=one_city(latitude=0.36,longitude=0)
    d=observer.direction(0,1)
    point=observer.position+10_000*d
    _,incoming,_=source_to_point(source,point[None,:])
    cosine=float(-incoming[0,0]@d)
    assert cosine>0.99
    assert hg_phase(cosine,0.8)>hg_phase(-cosine,0.8)*100


def test_earth_hides_low_but_not_high_illuminated_air():
    source=one_city(latitude=0,longitude=0)
    points=unit_position(np.array([1,1]),np.array([0,0]))*(EARTH_RADIUS_M+np.array([100,10_000]))[:,None]
    _,_,mu=source_to_point(source,points)
    assert mu[0,0]<0
    assert mu[1,0]>0
    assert emission(mu[0,0])==0
    assert emission(mu[1,0])>0


def test_beer_law_for_constant_extinction():
    a=Atmosphere(rayleigh_scale_height_m=1e30,aerosol_optical_depth=0)
    start=np.array([EARTH_RADIUS_M,0,0])
    end=start+np.array([10_000,0,0])
    tau=float(a.optical_depth(start,end))
    assert np.exp(-tau)==pytest.approx(np.exp(-a.rayleigh_sea_level*10_000),rel=1e-13)


def test_vertical_optical_depth_has_analytic_integral():
    a=Atmosphere()
    start=np.array([EARTH_RADIUS_M,0,0])
    end=start+np.array([a.top_m,0,0])
    expected=(a.rayleigh_sea_level*a.rayleigh_scale_height_m*(1-np.exp(-a.top_m/a.rayleigh_scale_height_m))
              +a.aerosol_optical_depth*(1-np.exp(-a.top_m/a.aerosol_scale_height_m)))
    assert float(a.optical_depth(start,end,64))==pytest.approx(expected,rel=1e-10)
    assert float(a.optical_depth(start,end,12))==pytest.approx(expected,rel=0.005)
    assert float(a.optical_depth(end,start,64))==pytest.approx(expected,rel=1e-10)


def test_rayleigh_wavelength_dependence():
    assert Atmosphere(reference_wavelength_nm=450).rayleigh_sea_level/Atmosphere().rayleigh_sea_level==pytest.approx((550/450)**4)


@pytest.mark.parametrize("albedo,aod",[(0.9,0),(0,0.2)])
def test_zero_scattering_even_with_absorption(albedo,aod):
    atmosphere=Atmosphere(rayleigh_multiplier=0,aerosol_optical_depth=aod,aerosol_single_scattering_albedo=albedo)
    assert integrate_direction(Observer(),one_city(),180,10,atmosphere)==0


def test_fixed_atmosphere_linearity():
    first=integrate_direction(Observer(),one_city(),180,10)
    second=integrate_direction(Observer(),one_city(power=2),180,10)
    assert first>0
    assert second==pytest.approx(2*first,rel=1e-13)


def test_azimuth_mirror_symmetry():
    source=one_city(latitude=0.36,longitude=0)
    observer=Observer(0,0,20)
    assert integrate_direction(observer,source,30,10)==pytest.approx(integrate_direction(observer,source,330,10),rel=1e-11)


def test_ray_quadrature_convergence():
    values=[integrate_direction(Observer(),one_city(),180,10,settings=SolverSettings(ray_segments=n,path_quadrature=24)) for n in (96,192,384)]
    assert abs(values[1]-values[2])<abs(values[0]-values[1])
    assert abs(values[1]/values[2]-1)<0.005


def test_reject_unphysical_parameters():
    with pytest.raises(ValueError): Atmosphere(aerosol_single_scattering_albedo=1.1)
    with pytest.raises(ValueError): Atmosphere(aerosol_g=1)
    with pytest.raises(ValueError): Observer(height_m=-1)
    with pytest.raises(ValueError): SolverSettings(ray_segments=1)
    with pytest.raises(ValueError): emission(0.5,1.1)
    with pytest.raises(ValueError): Sources(np.zeros((1,3)),np.array([1]))
