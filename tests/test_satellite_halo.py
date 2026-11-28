import numpy as np

from lightpollution.adapters.viirs_daily import clear_dark_a1
from lightpollution.models.all_orders import AllOrderSettings
from lightpollution.models.halo_profiles import (
    KAPPA_BOUNDS,
    pool_profile_fits,
    profile_normal_terms,
    redistribution_flux_below,
    redistribution_radiance,
)
from lightpollution.models.satellite_halo import (
    build_halo_kernel,
    emission_radiance,
    fit_nonnegative,
    flux_below,
    spread_flux,
)
from lightpollution.physics import Atmosphere


def test_satellite_templates_conserve_upward_flux():
    mu = np.linspace(0, 1, 16_385)
    flux = 2 * np.pi * np.trapezoid(mu[:, None] * emission_radiance(mu), mu, axis=0)
    np.testing.assert_allclose(flux, [1, 1], rtol=1e-7)
    np.testing.assert_allclose(flux_below(10), [0.0301537, 0.270002], atol=5e-5)


def test_redistribution_preserves_h2_power_and_positivity():
    mu = np.linspace(0, 1, 50_001)
    perturbation = redistribution_radiance(mu)
    h2 = 1 + np.maximum(0, 1 - 2 * mu)
    for kappa in KAPPA_BOUNDS:
        trial = h2 + kappa * perturbation
        assert np.min(trial) >= 0
        np.testing.assert_allclose(
            np.trapezoid(mu * trial, mu), np.trapezoid(mu * h2, mu), atol=1e-9
        )
    assert not np.any(perturbation[mu >= 0.5])
    for angle in (10, 25, 30):
        nodes = np.linspace(0, np.sin(np.deg2rad(angle)), 10_001)
        np.testing.assert_allclose(
            np.trapezoid(nodes * redistribution_radiance(nodes), nodes),
            redistribution_flux_below(angle),
            atol=1e-9,
        )


def test_profile_fit_cancels_offsets_and_accounts_for_redundant_differences():
    d = np.array([[1, -1, 0], [0, 1, -1], [1, 0, -1]])
    covariance = np.array([[0.04, 0.01, 0], [0.01, 0.06, 0], [0, 0, 0.05]])
    angular = np.array([0.11, 0.05, -0.01])
    for rows in (d[:2], d):
        baseline = profile_normal_terms(rows, covariance, 3.2 * angular, angular)
        shifted = profile_normal_terms(rows, covariance, 3.2 * angular + 0.7, angular)
        assert baseline['rank'] == shifted['rank'] == 2
        np.testing.assert_allclose(
            [baseline['numerator'], baseline['denominator']],
            [shifted['numerator'], shifted['denominator']],
            atol=1e-12,
        )
        fit = pool_profile_fits([shifted, shifted])
        np.testing.assert_allclose(fit['kappa'], 3.2)
        assert not fit['at_bound']


def test_satellite_transfer_in_vacuum_is_only_direct():
    atmosphere = Atmosphere(rayleigh_multiplier=0, aerosol_optical_depth=0)
    kernel = build_halo_kernel(
        32, 117, 400, 460, 1500, atmosphere, AllOrderSettings(samples=100)
    )
    np.testing.assert_allclose(kernel.direct, emission_radiance(np.cos(np.deg2rad(32))))
    assert kernel.hits == (0, 0)
    assert not np.any(kernel.single) and not np.any(kernel.multiple)


def test_satellite_kernel_can_score_an_extra_shape_on_the_same_trajectories():
    def with_extra(mu):
        return np.concatenate(
            (emission_radiance(mu), np.maximum(0, 1 - 2 * np.asarray(mu))[..., None]),
            axis=-1,
        )

    arguments = (50, 117, 400, 460, 5000, Atmosphere(), AllOrderSettings(samples=8192))
    base = build_halo_kernel(*arguments)
    extended = build_halo_kernel(*arguments, angular_radiance=with_extra)
    np.testing.assert_array_equal(extended.direct[:2], base.direct)
    np.testing.assert_array_equal(extended.single[:2], base.single)
    np.testing.assert_array_equal(extended.multiple[:2], base.multiple)
    assert extended.direct[2] == 0
    assert extended.single[2].sum() > 0


def test_satellite_kernel_keeps_azimuth_and_does_not_wrap():
    source = np.zeros((4, 5))
    source[1, -1] = 3
    east = [[0, 0, 0], [0, 2, 4], [0, 0, 0]]
    expected = 2 * source
    np.testing.assert_allclose(spread_flux(source, east), expected, atol=1e-12)
    expected[1, -2] = 12
    np.testing.assert_allclose(
        spread_flux(source, np.fliplr(east)), expected, atol=1e-12
    )


def test_fit_ignores_an_empty_component_and_enforces_nonnegative_sources():
    design = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0]])
    coefficient, loss = fit_nonnegative(design, [2, -1, 2, -1], [1, 2, 3, 4])
    np.testing.assert_allclose(coefficient, [2, 0, 0], atol=1e-12)
    assert loss == 20


def test_daily_screen_rejects_light_cloud_and_contaminated_counts():
    high = 3 << 4
    data = {
        'QF_Cloud_Mask': np.array(
            [high, high | 1, high | (1 << 9), high | (2 << 6), high, high, np.nan]
        ),
        'QF_DNB': np.array([0, 0, 0, 0, 1 << 16, 0, 0]),
        'DNB_At_Sensor_Radiance': np.full(7, 0.4),
        'Solar_Zenith': np.full(7, 110.0),
        'Lunar_Zenith': np.array([110, 110, 110, 110, 110, 99.9, 110]),
        'UTC_Time': np.full(7, 1.0),
        'Sensor_Azimuth': np.full(7, 80.0),
        'Sensor_Zenith': np.full(7, 45.0),
    }
    np.testing.assert_array_equal(
        clear_dark_a1(data), [True, False, False, False, False, False, False]
    )
