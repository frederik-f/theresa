# SPDX-License-Identifier: GPL-3.0-or-later
"""Canister loading: proportionality and closed-form admissible loading."""

from __future__ import annotations

import numpy as np
import pytest

from theresa.config import SECONDS_PER_YEAR, make_config
from theresa.decay_heat import nominal_power_watt, package_power_watt, source_terms_watt
from theresa.pipeline import temperature_history

N_PROBE = 12


def synthetic_kernel(n_times: int = 400) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic step response over 1000 a and a ``psi`` field."""
    times = np.concatenate(
        ([0.0], np.logspace(np.log10(5.0e3), np.log10(1000.0 * SECONDS_PER_YEAR), n_times))
    )
    near = 0.030 * (1.0 - np.exp(-times / (0.3 * SECONDS_PER_YEAR)))
    far = 0.060 * (1.0 - np.exp(-times / (600.0 * SECONDS_PER_YEAR)))
    profile = np.linspace(1.0, 0.85, N_PROBE)  # the barrel is hotter than the ends
    kernel = (near + far)[:, None] * profile[None, :]
    psi = np.full(N_PROBE, 500.0)
    return times, kernel, psi


def test_rise_is_proportional_to_the_loading():
    # Twice the fuel elements, twice the rise.
    times, kernel, psi = synthetic_kernel()

    single = temperature_history(make_config(n_fuel_elements=1), times, kernel, psi)
    double = temperature_history(make_config(n_fuel_elements=2), times, kernel, psi)

    assert np.allclose(double.rise_probe, 2.0 * single.rise_probe, rtol=1e-12, atol=1e-12)
    assert double.canister_power_watt == pytest.approx(2.0 * single.canister_power_watt)


def test_baseline_does_not_scale_with_the_loading():
    # The undisturbed temperature is independent of the loading.
    times, kernel, psi = synthetic_kernel()

    single = temperature_history(make_config(n_fuel_elements=1), times, kernel, psi)
    double = temperature_history(make_config(n_fuel_elements=2), times, kernel, psi)

    assert np.allclose(single.baseline_probe, double.baseline_probe)


def test_max_admissible_power_is_exactly_the_binding_constraint():
    # At the admissible loading the peak equals the limit.
    times, kernel, psi = synthetic_kernel()
    cfg = make_config(n_fuel_elements=10)
    history = temperature_history(cfg, times, kernel, psi)

    limit = 100.0
    admissible = history.max_admissible_power_watt(limit)

    scaled = temperature_history(make_config(base=cfg, n_fuel_elements=1), times, kernel, psi)
    factor = admissible / scaled.canister_power_watt
    peak_at_admissible = float((scaled.baseline_probe[None, :] + factor * scaled.rise_probe).max())
    assert peak_at_admissible == pytest.approx(limit, abs=1e-9)


def test_slightly_above_the_admissible_loading_exceeds_the_limit():
    times, kernel, psi = synthetic_kernel()
    cfg = make_config(n_fuel_elements=10)
    history = temperature_history(cfg, times, kernel, psi)
    limit = 100.0

    admissible = history.max_admissible_power_watt(limit)
    unit = temperature_history(make_config(base=cfg, n_fuel_elements=1), times, kernel, psi)

    for factor, expectation in ((0.999, False), (1.001, True)):
        scale = factor * admissible / unit.canister_power_watt
        peak = float((unit.baseline_probe[None, :] + scale * unit.rise_probe).max())
        assert (peak > limit) is expectation


def test_admissible_power_is_zero_when_the_rock_is_already_too_hot():
    times, kernel, psi = synthetic_kernel()
    # Undisturbed temperature above the limit.
    history = temperature_history(make_config(), times, kernel, np.full(N_PROBE, 4000.0))

    assert history.max_admissible_power_watt(100.0) == 0.0


def test_interim_storage_only_rescales_the_amplitudes():
    # Interim storage changes the amplitudes only.
    short = make_config(interim_storage_years=30.0)
    long = make_config(interim_storage_years=80.0)

    p_short, lam_short = source_terms_watt(short)
    p_long, lam_long = source_terms_watt(long)

    assert np.allclose(lam_short, lam_long)
    expected = p_short * np.exp(-lam_short * 50.0 * SECONDS_PER_YEAR)
    assert np.allclose(p_long, expected, rtol=1e-12)


def test_power_matches_the_spectrum_evaluated_from_discharge():
    # Amplitudes at emplacement reproduce the spectrum at t + storage.
    cfg = make_config(interim_storage_years=40.0, n_fuel_elements=7)
    from theresa import spectra

    spectrum = spectra.get(cfg["spectrum"])
    for years in (0.0, 5.0, 100.0, 1000.0):
        from_package = float(package_power_watt(cfg, years * SECONDS_PER_YEAR)[0])
        from_spectrum = 7 * 1000.0 * float(spectrum.power_kw(years + 40.0)[0])
        assert from_package == pytest.approx(from_spectrum, rel=1e-12)


def test_nominal_power_is_the_sum_of_the_amplitudes():
    cfg = make_config()
    amplitudes, _ = source_terms_watt(cfg)
    assert nominal_power_watt(cfg) == pytest.approx(float(amplitudes.sum()), rel=1e-12)


def test_six_term_source_is_supported_end_to_end():
    # The brief asks for six exponentials; nothing in the chain may assume five.
    times, kernel, psi = synthetic_kernel()
    cfg = make_config(
        decay_heat_p_kw=[3.0, 1.1, 0.4, 0.2, 0.02, 0.001],
        decay_heat_t_half_years=[2.2, 10.0, 32.0, 400.0, 13000.0, 760000.0],
        spectrum="custom",
    )

    history = temperature_history(cfg, times, kernel, psi)

    assert history.rise_probe.shape == (times.size, N_PROBE)
    assert np.all(np.isfinite(history.temperature_celsius))
    assert history.temperature_celsius.max() > history.baseline_probe.max()


def test_scaling_to_a_power_leaves_the_decay_shape_alone():
    # Scaling preserves half-lives and relative weights.
    from theresa.decay_heat import scaled_to_power

    cfg = make_config(n_fuel_elements=2)
    scaled = scaled_to_power(cfg, 5000.0)

    assert nominal_power_watt(scaled) == pytest.approx(5000.0, rel=1e-12)
    assert scaled["decay_heat_t_half_years"] == cfg["decay_heat_t_half_years"]

    before = np.asarray(cfg["decay_heat_p_kw"])
    after = np.asarray(scaled["decay_heat_p_kw"])
    ratios = after / before
    assert np.allclose(ratios, ratios[0], rtol=1e-12)


def test_scaling_to_a_power_rejects_a_non_positive_loading():
    from theresa.decay_heat import scaled_to_power

    with pytest.raises(ValueError, match="must be positive"):
        scaled_to_power(make_config(), 0.0)
