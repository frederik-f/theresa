# SPDX-License-Identifier: GPL-3.0-or-later
"""Duhamel convolution against closed-form references (numpy only)."""

from __future__ import annotations

import numpy as np
import pytest

from theresa import duhamel
from theresa.config import SECONDS_PER_YEAR

# Exponential-sum kernel with a closed-form convolution.
KERNEL_MU = np.array([1.0e-10, 1.0e-9, 1.0e-8, 3.0e-7])
KERNEL_A = np.array([0.05, 0.03, 0.02, 0.01])

# A six-term source, as the brief asks for, with realistic half-lives.
SIX_TERM_P = np.array([6000.0, 2000.0, 400.0, 200.0, 20.0, 1.0])
SIX_TERM_HALF_LIVES_YEARS = np.array([2.2, 10.0, 32.0, 400.0, 13000.0, 760000.0])
SIX_TERM_LAM = np.log(2.0) / (SIX_TERM_HALF_LIVES_YEARS * SECONDS_PER_YEAR)


def log_grid(n: int, t_end_years: float = 1000.0) -> np.ndarray:
    """A ``t = 0``-prefixed logarithmic time grid [s]."""
    return np.concatenate(
        ([0.0], np.logspace(np.log10(5.0e3), np.log10(t_end_years * SECONDS_PER_YEAR), n))
    )


def analytic_kernel(t: np.ndarray) -> np.ndarray:
    """The step response of the reference kernel, sampled at ``t``."""
    return duhamel.convolve_exponential_kernel(
        KERNEL_A, KERNEL_MU, np.array([1.0]), np.array([0.0]), t
    )


def test_constant_source_reproduces_the_step_response_exactly():
    # A constant source P gives exactly P * S(t).
    t = log_grid(300)
    s = analytic_kernel(t)
    power = 1234.0

    rise = duhamel.convolve(t, s, np.array([power]), np.array([0.0]))

    assert np.allclose(rise, power * s, rtol=0.0, atol=1e-9)


def test_matches_the_closed_form_for_an_exponential_kernel():
    # Sampled kernel against the analytic convolution.
    t = log_grid(3000)
    s = analytic_kernel(t)

    numeric = duhamel.convolve(t, s, SIX_TERM_P, SIX_TERM_LAM)
    exact = duhamel.convolve_exponential_kernel(KERNEL_A, KERNEL_MU, SIX_TERM_P, SIX_TERM_LAM, t)

    relative = np.abs(numeric[1:] - exact[1:]) / np.abs(exact[1:])
    assert relative.max() < 1e-4


def test_time_refinement_converges_at_second_order():
    # Second-order convergence in the time step.
    errors = []
    for n in (250, 500, 1000, 2000):
        t = log_grid(n)
        s = analytic_kernel(t)
        numeric = duhamel.convolve(t, s, SIX_TERM_P, SIX_TERM_LAM)
        exact = duhamel.convolve_exponential_kernel(
            KERNEL_A, KERNEL_MU, SIX_TERM_P, SIX_TERM_LAM, t
        )
        errors.append(float(np.max(np.abs(numeric[1:] - exact[1:]) / np.abs(exact[1:]))))

    ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
    assert all(ratio > 2.5 for ratio in ratios), f"errors {errors}, ratios {ratios}"


def test_production_schedule_is_accurate_enough():
    # Absolute discretisation error [K] of the shipped schedule.
    from theresa.config import make_config
    from theresa.prj_builder import time_step_pairs

    cfg = make_config()
    steps = [delta for repeat, delta in time_step_pairs(cfg) for _ in range(repeat)]
    t = np.concatenate(([0.0], np.cumsum(steps)))

    s = analytic_kernel(t)
    numeric = duhamel.convolve(t, s, SIX_TERM_P, SIX_TERM_LAM)
    exact = duhamel.convolve_exponential_kernel(KERNEL_A, KERNEL_MU, SIX_TERM_P, SIX_TERM_LAM, t)

    # Peak error.
    assert abs(numeric.max() - exact.max()) < 0.05
    assert np.max(np.abs(numeric - exact)) < 0.15


def test_superposition_holds_in_the_source():
    # Superposition over the source terms.
    t = log_grid(400)
    s = analytic_kernel(t)

    together = duhamel.convolve(t, s, SIX_TERM_P, SIX_TERM_LAM)
    separately = sum(
        duhamel.convolve(t, s, SIX_TERM_P[i : i + 1], SIX_TERM_LAM[i : i + 1])
        for i in range(SIX_TERM_P.size)
    )

    assert np.allclose(together, separately, rtol=1e-12, atol=1e-12)


def test_scaling_the_source_scales_the_answer():
    t = log_grid(200)
    s = analytic_kernel(t)

    single = duhamel.convolve(t, s, SIX_TERM_P, SIX_TERM_LAM)
    doubled = duhamel.convolve(t, s, 2.0 * SIX_TERM_P, SIX_TERM_LAM)

    assert np.allclose(doubled, 2.0 * single, rtol=1e-12, atol=1e-12)


def test_independent_sample_points_are_convolved_independently():
    # Several probe columns at once.
    t = log_grid(200)
    s = analytic_kernel(t)
    columns = np.stack([s, 0.5 * s, 2.0 * s], axis=1)

    rise = duhamel.convolve(t, columns, SIX_TERM_P, SIX_TERM_LAM)
    reference = duhamel.convolve(t, s, SIX_TERM_P, SIX_TERM_LAM)

    assert rise.shape == (t.size, 3)
    assert np.allclose(rise[:, 0], reference, rtol=1e-12, atol=1e-12)
    assert np.allclose(rise[:, 1], 0.5 * reference, rtol=1e-12, atol=1e-12)
    assert np.allclose(rise[:, 2], 2.0 * reference, rtol=1e-12, atol=1e-12)


def test_degenerate_rates_use_the_limiting_form():
    # Removable singularity mu_k == lam_i.
    mu = np.array([1.0e-9])
    a = np.array([0.04])
    p = np.array([1000.0])

    exact_match = duhamel.convolve_exponential_kernel(a, mu, p, np.array([1.0e-9]), log_grid(50))
    just_beside = duhamel.convolve_exponential_kernel(
        a, mu, p, np.array([1.0e-9 * (1.0 + 1e-6)]), log_grid(50)
    )

    assert np.all(np.isfinite(exact_match))
    assert np.allclose(exact_match, just_beside, rtol=1e-4, atol=1e-9)


def test_rejects_inconsistent_input():
    t = log_grid(20)
    s = analytic_kernel(t)

    with pytest.raises(ValueError, match="start at t = 0"):
        duhamel.convolve(t[1:], s[1:], np.array([1.0]), np.array([0.0]))
    not_ascending = t.copy()
    not_ascending[3], not_ascending[4] = t[4], t[3]
    with pytest.raises(ValueError, match="ascending"):
        duhamel.convolve(not_ascending, s, np.array([1.0]), np.array([0.0]))
    with pytest.raises(ValueError, match="same shape"):
        duhamel.convolve(t, s, np.array([1.0, 2.0]), np.array([0.0]))
    with pytest.raises(ValueError, match="non-negative"):
        duhamel.convolve(t, s, np.array([1.0]), np.array([-1.0]))


def test_source_weights_is_the_convolution_written_as_a_matrix():
    """``convolve`` equals ``source_weights`` applied to the slopes."""
    t = log_grid(60)
    s = np.column_stack([analytic_kernel(t), 0.5 * analytic_kernel(t)])
    p = np.array([900.0, 240.0, 30.0])
    lam = np.log(2.0) / (np.array([30.0, 300.0, 3000.0]) * 365.25 * 24 * 3600.0)

    weights = duhamel.source_weights(t, p, lam)
    slopes = np.diff(s, axis=0) / np.diff(t)[:, None]

    assert weights.shape == (t.size, t.size - 1)
    assert np.array_equal(weights @ slopes, duhamel.convolve(t, s, p, lam))


def test_source_weights_ignores_the_kernel_entirely():
    """``source_weights`` depends only on the times and the source."""
    t = log_grid(40)
    p = np.array([1000.0])
    lam = np.array([1.0e-10])

    first = duhamel.source_weights(t, p, lam)
    second = duhamel.source_weights(t, p, lam)
    assert np.array_equal(first, second)

    # Constant source: weights times slopes give amplitude * S.
    for kernel in (analytic_kernel(t), np.sqrt(t / t[-1]), np.full_like(t, 3.0) - 3.0):
        slopes = np.diff(kernel) / np.diff(t)
        rise = duhamel.source_weights(t, p, np.array([0.0])) @ slopes
        assert np.allclose(rise, p[0] * (kernel - kernel[0]))


def test_selecting_a_column_commutes_with_the_contraction():
    """Selecting probe columns before or after the convolution gives the same result."""
    rng = np.random.default_rng(11)
    t = log_grid(40)
    n_modes, n_probe = 4, 9
    slopes = rng.normal(size=(t.size - 1, n_modes, n_probe))
    weights = duhamel.source_weights(t, np.array([800.0, 120.0]), np.array([0.0, 7.0e-11]))
    hottest = rng.integers(0, n_probe, size=t.size)

    full = (weights @ slopes.reshape(t.size - 1, n_modes * n_probe)).reshape(
        t.size, n_modes, n_probe
    )
    after = full[np.arange(t.size), :, hottest]
    before = np.einsum("tj,jmt->tm", weights, slopes[:, :, hottest], optimize=True)

    assert after.shape == (t.size, n_modes)
    assert np.allclose(after, before, rtol=0.0, atol=1e-9 * np.abs(after).max())


def direct_weights(t: np.ndarray, p: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """Reference matrix from the definition."""
    ages = np.maximum(t[:, None] - t[None, :], 0.0)
    weights = np.zeros((t.size, t.size - 1))
    for amplitude, decay in zip(p, lam, strict=True):
        decayed = np.exp(-decay * ages)
        weights += amplitude / decay * (decayed[:, 1:] - decayed[:, :-1])
    return np.tril(weights, k=-1)


@pytest.mark.parametrize(
    "half_lives_years",
    [
        [2.2, 32.0, 13000.0],
        [0.05, 0.5, 30.0],
        # lam * t_end around the factored limit.
        [1.0, 1.2, 1.5],
    ],
)
def test_source_weights_agrees_with_its_definition(half_lives_years):
    """Factored and direct route against the definition; all entries finite."""
    t = log_grid(120)
    p = np.array([900.0, 240.0, 30.0])
    lam = np.log(2.0) / (np.array(half_lives_years) * SECONDS_PER_YEAR)

    weights = duhamel.source_weights(t, p, lam)
    reference = direct_weights(t, p, lam)

    assert np.isfinite(weights).all()
    assert np.allclose(weights, reference, rtol=0.0, atol=1e-10 * np.abs(reference).max())
    assert np.array_equal(weights, np.tril(weights, k=-1)), "causality"


def test_source_weights_stay_finite_for_any_half_life():
    """Half-lives over nine decades with large amplitudes: finite and exact."""
    t = log_grid(150)
    for half_life in np.logspace(-2, 7, 60):
        lam = np.log(2.0) / (np.array([half_life]) * SECONDS_PER_YEAR)
        p = np.array([1.0e5])
        weights = duhamel.source_weights(t, p, lam)
        reference = direct_weights(t, p, lam)
        assert np.isfinite(weights).all(), half_life
        assert np.allclose(weights, reference, rtol=0.0, atol=1e-9 * np.abs(reference).max())
