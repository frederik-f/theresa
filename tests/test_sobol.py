# SPDX-License-Identifier: GPL-3.0-or-later
"""Sobol analysis: estimator against a closed-form benchmark and on a synthetic surrogate."""

from __future__ import annotations

import functools

import numpy as np
import pytest

from theresa import surrogate
from theresa.config import SECONDS_PER_YEAR, make_config
from theresa.doe import latin_hypercube
from theresa.sobol import _estimators, peak_temperatures, sobol_indices

N_TIMES = 48
N_PROBE = 12

FEATURES = ("canister_gap", "a_drift", "backfill_lambda")


def times() -> np.ndarray:
    return np.concatenate(
        ([0.0], np.logspace(np.log10(5.0e3), np.log10(1000.0 * SECONDS_PER_YEAR), N_TIMES - 1))
    )


def synthetic_kernel(gap: float, a_drift: float, backfill_lambda: float) -> np.ndarray:
    """Synthetic step response depending on backfill conductivity and spacings."""
    t = times()
    near = (0.024 / backfill_lambda) * (1.0 - np.exp(-t / (0.3 * SECONDS_PER_YEAR)))
    array = (0.9 / gap + 2.2 / a_drift) * (1.0 - np.exp(-t / (400.0 * SECONDS_PER_YEAR)))
    profile = np.linspace(1.0, 0.86, N_PROBE)
    return (near + array)[:, None] * profile[None, :]


def synthetic_psi(gap: float, a_drift: float, backfill_lambda: float) -> np.ndarray:
    return 500.0 + 0.01 * np.linspace(-1.0, 1.0, N_PROBE) * (30.0 / a_drift)


@functools.cache
def train(n_points: int = 70) -> surrogate.Surrogate:
    """Synthetic surrogate, fitted once per module."""
    points = latin_hypercube(FEATURES, n_points, seed=0, margin=0.0)
    kernels = np.stack([synthetic_kernel(*point) for point in points])
    psis = np.stack([synthetic_psi(*point) for point in points])
    return surrogate.fit(times(), points, kernels, psis, FEATURES, fixed_depth_m=500.0)


def trained_ranges(model: surrogate.Surrogate) -> dict[str, tuple[float, float]]:
    """The exact trained hull, so range validation is never the thing that fails."""
    return model.trained_ranges()


# --- the estimator itself ---------------------------------------------------


def ishigami(x: np.ndarray, a: float = 7.0, b: float = 0.1) -> np.ndarray:
    return np.sin(x[:, 0]) + a * np.sin(x[:, 1]) ** 2 + b * x[:, 2] ** 4 * np.sin(x[:, 0])


def test_the_estimator_reproduces_the_ishigami_indices():
    # Ishigami function: known first-order and total indices.
    from scipy.stats import qmc

    a, b = 7.0, 0.1
    variance = a**2 / 8 + b * np.pi**4 / 5 + b**2 * np.pi**8 / 18 + 0.5
    v1 = 0.5 * (1 + b * np.pi**4 / 5) ** 2
    v13 = 8 * b**2 * np.pi**8 / 225
    expected_first = np.array([v1 / variance, (a**2 / 8) / variance, 0.0])
    expected_total = np.array([(v1 + v13) / variance, (a**2 / 8) / variance, v13 / variance])

    n = 2**14
    block = qmc.Sobol(d=6, scramble=True, seed=3).random(n)
    sample = -np.pi + 2 * np.pi * block
    a_matrix, b_matrix = sample[:, :3], sample[:, 3:]

    f_a, f_b = ishigami(a_matrix), ishigami(b_matrix)
    f_ab = np.empty((3, n))
    for i in range(3):
        mixed = a_matrix.copy()
        mixed[:, i] = b_matrix[:, i]
        f_ab[i] = ishigami(mixed)

    first, total = _estimators(f_a, f_b, f_ab)
    assert np.allclose(first, expected_first, atol=0.02)
    assert np.allclose(total, expected_total, atol=0.02)


def test_a_constant_output_gives_zero_indices_instead_of_dividing_by_zero():
    f = np.ones(64)
    first, total = _estimators(f, f, np.ones((3, 64)))
    assert np.all(first == 0.0)
    assert np.all(total == 0.0)


# --- the surrogate-side evaluation -----------------------------------------


def test_the_fast_and_exact_paths_agree():
    # Fast path (no monotonic clipping) against the exact path.
    model = train()
    ranges = trained_ranges(model)
    rng = np.random.default_rng(0)
    names = tuple(ranges)
    lows = np.array([ranges[n][0] for n in names])
    highs = np.array([ranges[n][1] for n in names])
    samples = lows + rng.random((64, len(names))) * (highs - lows)

    fast = peak_temperatures(model, names, samples)
    exact = peak_temperatures(model, names, samples, exact=True)
    assert np.max(np.abs(fast - exact)) < 1.0e-3


def test_loading_scales_the_rise_exactly():
    # Zero gradient: uniform baseline, peak proportional to the loading.
    model = train()
    cfg = make_config()
    flat = {"geothermal_gradient_k_per_m": 0.0}
    design = np.array([[cfg["canister_gap"], cfg["a_drift"], cfg["mat_backfill"]["lambda"]]])
    names = (*FEATURES, "n_fuel_elements")

    def peak(elements: float) -> float:
        return peak_temperatures(model, names, np.c_[design, elements], base_config=flat)[0]

    single, double, undisturbed = peak(1.0), peak(2.0), peak(0.0)
    assert undisturbed == pytest.approx(cfg["surface_temperature_celsius"], abs=1e-9)
    assert double - undisturbed == pytest.approx(2.0 * (single - undisturbed), rel=1e-9)


def test_an_input_the_output_does_not_depend_on_gets_a_vanishing_index():
    # Surface temperature: additive, no interaction share.
    model = train()
    result = sobol_indices(
        model,
        {**trained_ranges(model), "surface_temperature_celsius": (8.0, 12.0)},
        n_base=256,
        n_bootstrap=50,
        seed=1,
    )
    i = result.names.index("surface_temperature_celsius")
    assert result.total[i] - result.first_order[i] < 0.02


def test_indices_are_ordered_and_bounded():
    model = train()
    result = sobol_indices(model, trained_ranges(model), n_base=256, n_bootstrap=50, seed=2)
    assert np.all(result.total >= result.first_order - 0.02)
    assert np.all(result.first_order > -0.02)
    assert np.sum(result.first_order) < 1.05
    assert result.n_evaluations == 256 * (len(result.names) + 2)
    assert "S_Ti" in result.table()


def test_a_range_outside_the_trained_hull_is_refused():
    model = train()
    ranges = trained_ranges(model)
    low, high = ranges["a_drift"]
    with pytest.raises(ValueError, match="leaves the trained range"):
        sobol_indices(model, {**ranges, "a_drift": (low, high + 10.0)}, n_base=32, n_bootstrap=5)


def test_an_unsupported_analytic_input_is_rejected_rather_than_ignored():
    # Inputs that change the decay-curve shape are refused.
    model = train()
    with pytest.raises(ValueError, match="analytic inputs"):
        sobol_indices(
            model,
            {**trained_ranges(model), "interim_storage_years": (30.0, 70.0)},
            n_base=32,
            n_bootstrap=5,
        )


def test_an_increasing_range_is_required():
    model = train()
    with pytest.raises(ValueError, match="increasing"):
        sobol_indices(model, {"a_drift": (40.0, 20.0)}, n_base=32, n_bootstrap=5)


def test_a_base_without_fuel_elements_is_refused():
    model = train()
    with pytest.raises(ValueError, match="no fuel elements"):
        peak_temperatures(
            model, ("n_fuel_elements",), np.array([[1.0]]), base_config={"n_fuel_elements": 0}
        )
