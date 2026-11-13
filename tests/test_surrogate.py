# SPDX-License-Identifier: GPL-3.0-or-later
"""Surrogate on a synthetic kernel field with near-field and array part (no OGS)."""

from __future__ import annotations

import functools

import numpy as np
import pytest

from theresa import surrogate
from theresa.config import SECONDS_PER_YEAR
from theresa.doe import latin_hypercube

N_PROBE = 20
N_TIMES = 120


def times() -> np.ndarray:
    return np.concatenate(
        ([0.0], np.logspace(np.log10(5.0e3), np.log10(1000.0 * SECONDS_PER_YEAR), N_TIMES - 1))
    )


def synthetic_kernel(gap: float, a_drift: float, l_can: float = 5.52) -> np.ndarray:
    """Synthetic step response: spacing-independent near field, array term in ``1 / a``."""
    t = times()
    # Near field scales as 1/L.
    near = (0.166 / l_can) * (1.0 - np.exp(-t / (0.3 * SECONDS_PER_YEAR)))
    array = (0.9 / gap + 2.2 / a_drift) * (1.0 - np.exp(-t / (400.0 * SECONDS_PER_YEAR)))
    profile = np.linspace(1.0, 0.86, N_PROBE)
    return (near + array)[:, None] * profile[None, :]


def synthetic_psi(gap: float, a_drift: float, l_can: float = 5.52) -> np.ndarray:
    """A psi field that is essentially the depth, with a tiny design tilt."""
    return 500.0 + 0.01 * np.linspace(-1.0, 1.0, N_PROBE) * (30.0 / a_drift)


FEATURES = ("canister_gap", "a_drift", "l_can")


@functools.cache
def train(n_points: int = 60) -> surrogate.Surrogate:
    """Synthetic surrogate, fitted once per module."""
    points = latin_hypercube(FEATURES, n_points, seed=0, margin=0.0)
    kernels = np.stack([synthetic_kernel(*point) for point in points])
    psis = np.stack([synthetic_psi(*point) for point in points])
    return surrogate.fit(times(), points, kernels, psis, FEATURES, fixed_depth_m=500.0)


def design_of(point) -> dict:
    return dict(zip(FEATURES, point, strict=True))


def test_reproduces_its_own_training_points():
    # Training points are reproduced.
    model = train()
    for point in model.train_points:
        kernel = model.predict_kernel(design_of(point)).kernel
        expected = synthetic_kernel(*point)
        assert np.max(np.abs(kernel - expected)) < 1e-6


def test_interpolates_between_training_points():
    # The real test: designs the model never saw.
    model = train()
    worst = 0.0
    for point in latin_hypercube(FEATURES, 12, seed=3):
        kernel = model.predict_kernel(design_of(point)).kernel
        expected = synthetic_kernel(*point)
        # Tolerance in kelvin at 6 kW.
        worst = max(worst, float(np.max(np.abs(kernel - expected))) * 6000.0)
    assert worst < 0.5, f"worst interpolated error {worst:.3f} K"


def test_needs_only_a_few_modes():
    # Few modes capture the kernel field.
    model = train()
    assert len(model.kernel_models) <= surrogate.MAX_MODES
    assert model.kernel_modes.shape[1] == N_TIMES * N_PROBE


def test_predicted_kernel_is_monotone_and_starts_at_zero():
    # Kernels are non-decreasing.
    model = train()
    kernel = model.predict_kernel({"canister_gap": 7.3, "a_drift": 32.5, "l_can": 5.4}).kernel

    assert np.all(kernel[0] == 0.0)
    assert np.all(np.diff(kernel, axis=0) >= 0.0)


def test_extrapolation_is_flagged():
    # Silence outside the trained range would be the dangerous failure mode.
    model = train()
    warnings = model.predict_kernel({"canister_gap": 0.5, "a_drift": 300.0, "l_can": 12.0}).warnings
    assert warnings
    assert {warning.feature for warning in warnings} == {"canister_gap", "a_drift", "l_can"}
    assert all("outside the trained range" in str(warning) for warning in warnings)


def test_inside_the_trained_range_there_are_no_warnings():
    model = train()
    warnings = model.predict_kernel({"canister_gap": 9.5, "a_drift": 40.0, "l_can": 5.52}).warnings
    assert warnings == []


def test_round_trips_through_a_file(tmp_path):
    model = train()
    path = model.save(tmp_path / "surrogate.pkl")
    reloaded = surrogate.Surrogate.load(path)

    design = {"canister_gap": 8.0, "a_drift": 45.0, "l_can": 6.0}
    original = model.predict_kernel(design).kernel
    restored = reloaded.predict_kernel(design).kernel

    assert np.allclose(original, restored)
    assert reloaded.fixed_depth_m == 500.0


def test_reciprocal_transform_is_applied_where_requested():
    points = np.array([[10.0, 40.0], [20.0, 80.0]])
    flags = np.array([True, False])
    transformed = surrogate.transform_design(points, flags)

    assert transformed[0, 0] == pytest.approx(0.1)
    assert transformed[0, 1] == pytest.approx(40.0)


def test_rejects_too_few_designs():
    with pytest.raises(ValueError, match="at least three"):
        surrogate.fit(
            times(),
            np.array([[10.0, 40.0, 5.5], [12.0, 50.0, 6.0]]),
            np.stack([synthetic_kernel(10.0, 40.0, 5.5), synthetic_kernel(12.0, 50.0, 6.0)]),
            np.stack([synthetic_psi(10.0, 40.0, 5.5), synthetic_psi(12.0, 50.0, 6.0)]),
            FEATURES,
        )


def test_latin_hypercube_is_stratified_in_the_reciprocal_coordinate():
    # Spacings stratified in 1/a.
    n = 8
    points = latin_hypercube(("canister_gap",), n, ranges={"canister_gap": (2.5, 15.0)}, margin=0.0)
    fraction = (1.0 / points[:, 0] - 1.0 / 2.5) / (1.0 / 15.0 - 1.0 / 2.5)
    assert sorted(np.floor(fraction * n).astype(int)) == list(range(n))


def test_latin_hypercube_stays_inside_the_range():
    ranges = {"canister_gap": (2.5, 15.0), "a_drift": (15.0, 60.0), "l_can": (4.0, 7.0)}
    points = latin_hypercube(FEATURES, 10, ranges=ranges, seed=1)

    for column, name in enumerate(FEATURES):
        low, high = ranges[name]
        assert np.all(points[:, column] >= low) and np.all(points[:, column] <= high), name


def test_latin_hypercube_stratifies_every_variable():
    # n points give n distinct levels per variable.
    points = latin_hypercube(FEATURES, 20, seed=2, margin=0.0)
    for column in range(points.shape[1]):
        assert len(np.unique(points[:, column])) == 20
