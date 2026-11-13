# SPDX-License-Identifier: GPL-3.0-or-later
"""Decay curves with any number of exponential terms."""

from __future__ import annotations

import numpy as np
import pytest

from theresa.config import SECONDS_PER_YEAR, make_config
from theresa.decay_heat import (
    build_decay_curve,
    nominal_power_watt,
    package_power_watt,
    source_terms_watt,
)
from theresa.duhamel import convolve
from theresa.pipeline import temperature_history
from theresa.prj_builder import build_prj_xml

# Six-term reference; every prefix is a valid spectrum.
P_KW = [3.318, 1.144, 0.2225, 0.05, 0.02112, 0.0009342]
T_HALF_YEARS = [2.163, 32.33, 397.3, 1500.0, 13680.0, 760700.0]

TERM_COUNTS = [1, 2, 3, 4, 5, 6, 8]


def config_with(n_terms: int) -> dict:
    """A configuration whose decay curve has exactly ``n_terms`` terms."""
    if n_terms <= len(P_KW):
        p_kw, half_lives = P_KW[:n_terms], T_HALF_YEARS[:n_terms]
    else:  # extend beyond six, to show six is not an upper bound either
        extra = n_terms - len(P_KW)
        p_kw = P_KW + [0.001 * (i + 1) for i in range(extra)]
        half_lives = T_HALF_YEARS + [5.0 * (i + 1) for i in range(extra)]
    return make_config(
        spectrum=f"{n_terms}-term test",
        decay_heat_p_kw=p_kw,
        decay_heat_t_half_years=half_lives,
    )


def synthetic_kernel(n_times: int = 250, n_probe: int = 6):
    times = np.concatenate(
        ([0.0], np.logspace(np.log10(5.0e3), np.log10(1000.0 * SECONDS_PER_YEAR), n_times))
    )
    shape = 0.03 * (1.0 - np.exp(-times / (0.3 * SECONDS_PER_YEAR))) + 0.06 * (
        1.0 - np.exp(-times / (500.0 * SECONDS_PER_YEAR))
    )
    kernel = shape[:, None] * np.linspace(1.0, 0.9, n_probe)[None, :]
    return times, kernel, np.full(n_probe, 500.0)


@pytest.mark.parametrize("n_terms", TERM_COUNTS)
def test_source_terms_have_the_requested_count(n_terms):
    amplitudes, decay_constants = source_terms_watt(config_with(n_terms))
    assert amplitudes.shape == (n_terms,)
    assert decay_constants.shape == (n_terms,)


@pytest.mark.parametrize("n_terms", TERM_COUNTS)
def test_the_whole_chain_runs_with_that_count(n_terms):
    cfg = config_with(n_terms)
    times, kernel, psi = synthetic_kernel()

    history = temperature_history(cfg, times, kernel, psi)

    assert np.all(np.isfinite(history.temperature_celsius))
    assert history.canister_power_watt == pytest.approx(nominal_power_watt(cfg), rel=1e-12)
    assert history.rise_probe.shape == kernel.shape


@pytest.mark.parametrize("n_terms", TERM_COUNTS)
def test_the_project_file_accepts_that_count(tmp_path, n_terms):
    # A direct run builds for any term count.
    meshes = {}
    directory = tmp_path / "mesh"
    directory.mkdir()
    for name in ("domain", "top", "bottom", "canister", "canister_wall"):
        path = directory / f"{name}.vtu"
        path.write_text("", encoding="utf-8")
        meshes[name] = path

    xml = build_prj_xml(
        config_with(n_terms), meshes, prj_dir=tmp_path, mode="direct", canister_power_watt=5000.0
    )
    assert "<name>decay_curve</name>" in xml


def test_a_shorter_spectrum_is_the_limit_of_a_longer_one_with_zero_amplitudes():
    # Zero amplitudes equal fewer terms.
    times, kernel, psi = synthetic_kernel()

    four = make_config(
        spectrum="four", decay_heat_p_kw=P_KW[:4], decay_heat_t_half_years=T_HALF_YEARS[:4]
    )
    six_padded = make_config(
        spectrum="six padded",
        decay_heat_p_kw=P_KW[:4] + [0.0, 0.0],
        decay_heat_t_half_years=T_HALF_YEARS[:4] + [1.0, 99999.0],
    )

    a = temperature_history(four, times, kernel, psi).temperature_celsius
    b = temperature_history(six_padded, times, kernel, psi).temperature_celsius

    assert np.allclose(a, b, rtol=1e-12, atol=1e-12)


def test_terms_superpose_so_the_count_carries_no_error():
    # Superposition over the terms.
    cfg = config_with(6)
    times, kernel, psi = synthetic_kernel()
    amplitudes, decay_constants = source_terms_watt(cfg)

    together = convolve(times, kernel, amplitudes, decay_constants)
    separately = sum(
        convolve(times, kernel, amplitudes[i : i + 1], decay_constants[i : i + 1])
        for i in range(amplitudes.size)
    )

    assert np.allclose(together, separately, rtol=1e-12, atol=1e-12)


def test_single_term_matches_a_hand_computed_exponential():
    # One term against hand calculation.
    cfg = make_config(
        spectrum="one term",
        decay_heat_p_kw=[2.5],
        decay_heat_t_half_years=[30.0],
        n_fuel_elements=1,
        interim_storage_years=0.0,
    )
    amplitudes, decay_constants = source_terms_watt(cfg)

    assert amplitudes[0] == pytest.approx(2500.0)
    assert decay_constants[0] == pytest.approx(np.log(2.0) / (30.0 * SECONDS_PER_YEAR))
    # After exactly one half-life the power must be half.
    assert float(package_power_watt(cfg, 30.0 * SECONDS_PER_YEAR)[0]) == pytest.approx(1250.0)


@pytest.mark.parametrize("n_terms", TERM_COUNTS)
def test_the_ogs_curve_is_monotone_for_any_count(n_terms):
    _times, values = build_decay_curve(config_with(n_terms), 5000.0)
    assert np.all(np.diff(values) <= 1e-12 * values[0])
