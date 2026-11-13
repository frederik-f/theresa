# SPDX-License-Identifier: GPL-3.0-or-later
r"""Variance-based (Sobol) sensitivity analysis of the peak temperature.

``S_i`` (first order)
    share of the variance explained by input *i* alone.
``S_Ti`` (total)
    share including all interactions of input *i*.

Saltelli design with ``N (d + 2)`` surrogate evaluations; inputs independent
and uniform over the given ranges, which must lie inside the trained range.
For a fixed decay curve the mode convolutions are computed once and combined
per sample.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import MATERIAL_ALIASES, design_value, make_config
from .decay_heat import source_terms_watt
from .duhamel import convolve
from .surrogate import Surrogate, transform_design

#: Samples processed per block.
CHUNK = 64

#: Analytic inputs that can be varied; they leave the decay-curve shape unchanged.
SUPPORTED_ANALYTIC = (
    "surface_temperature_celsius",
    "geothermal_gradient_k_per_m",
    "n_fuel_elements",
)


@dataclass
class SobolResult:
    """First-order and total Sobol indices for one parameter set.

    :param names: varied inputs.
    :param first_order: ``S_i`` per input.
    :param total: ``S_Ti`` per input.
    :param first_order_stderr: bootstrap standard error of ``S_i``.
    :param total_stderr: bootstrap standard error of ``S_Ti``.
    :param mean: mean peak temperature over the sample [degC].
    :param std: standard deviation of the peak temperature [K].
    :param n_base: base sample size ``N``.
    :param n_evaluations: surrogate evaluations, ``N (d + 2)``.
    :param ranges: input ranges.
    """

    names: tuple[str, ...]
    first_order: np.ndarray
    total: np.ndarray
    first_order_stderr: np.ndarray
    total_stderr: np.ndarray
    mean: float
    std: float
    n_base: int
    n_evaluations: int
    ranges: dict[str, tuple[float, float]] = field(default_factory=dict)

    @property
    def interaction_share(self) -> float:
        """``1 - sum(S_i)``: variance share of interactions."""
        return float(1.0 - np.sum(self.first_order))

    def table(self) -> str:
        """Markdown table, sorted by total index."""
        order = np.argsort(-self.total)
        lines = [
            "| input | S_i | S_Ti | S_Ti - S_i |",
            "| --- | --- | --- | --- |",
        ]
        for i in order:
            lines.append(
                f"| {self.names[i]} | {self.first_order[i]:.3f} ± {self.first_order_stderr[i]:.3f} "
                f"| {self.total[i]:.3f} ± {self.total_stderr[i]:.3f} "
                f"| {self.total[i] - self.first_order[i]:.3f} |"
            )
        return "\n".join(lines)


def peak_temperatures(
    model: Surrogate,
    names: tuple[str, ...],
    samples: np.ndarray,
    *,
    base_config: dict | None = None,
    exact: bool = False,
) -> np.ndarray:
    """Peak canister-surface temperature [degC] for a batch of input vectors.

    :param model: the surrogate.
    :param names: varied inputs: features of ``model`` or entries of
        :data:`SUPPORTED_ANALYTIC`.
    :param samples: ``(n_samples, len(names))`` input values.
    :param base_config: values of all other inputs; default
        :func:`theresa.config.make_config`.
    :param exact: convolve every kernel individually.
    :raises ValueError: if a name is neither a feature nor a configuration key.
    """
    samples = np.atleast_2d(np.asarray(samples, dtype=float))
    cfg = make_config(base=base_config)

    feature_index = {name: i for i, name in enumerate(model.feature_names)}
    analytic: dict[str, int] = {}
    for position, name in enumerate(names):
        if name in feature_index:
            continue
        if name in MATERIAL_ALIASES:
            continue
        if name not in SUPPORTED_ANALYTIC:
            raise ValueError(
                f"{name!r} is neither one of the surrogate's features "
                f"{model.feature_names} nor one of the analytic inputs "
                f"{SUPPORTED_ANALYTIC}"
            )
        analytic[name] = position

    # Interpolated inputs: reference design with the varied columns.
    raw = np.tile(
        np.array([design_value(cfg, name) for name in model.feature_names]),
        (samples.shape[0], 1),
    )
    for name, index in feature_index.items():
        if name in names:
            raw[:, index] = samples[:, names.index(name)]

    transformed = transform_design(raw, model.reciprocal)
    psi = model.psis(transformed)

    # Analytic inputs.
    def column(name: str) -> np.ndarray:
        if name in analytic:
            return samples[:, analytic[name]]
        return np.full(samples.shape[0], design_value(cfg, name))

    baseline = (
        column("surface_temperature_celsius")[:, None]
        + column("geothermal_gradient_k_per_m")[:, None] * psi
    )

    amplitudes, decay_constants = source_terms_watt(cfg)
    n_times, n_probe = model.probe_shape

    # Loading as a per-sample scale factor.
    if cfg["n_fuel_elements"] <= 0:
        raise ValueError(
            "the base configuration carries no fuel elements, so there is no rise "
            "to scale a loading from; give base_config a positive n_fuel_elements"
        )
    loading = column("n_fuel_elements")[:, None, None] / float(cfg["n_fuel_elements"])

    if exact:
        peaks = np.empty(samples.shape[0])
        for start in range(0, samples.shape[0], CHUNK):
            block = slice(start, start + CHUNK)
            kernels = model.kernels(transformed[block])
            flat = np.moveaxis(kernels, 0, 1).reshape(n_times, -1)
            rise = convolve(model.times, flat, amplitudes, decay_constants)
            rise = np.moveaxis(rise.reshape(n_times, -1, n_probe), 0, 1)
            peaks[block] = (baseline[block][:, None, :] + loading[block] * rise).max(axis=(1, 2))
        return peaks

    # Mean and modes convolved once; samples as weighted sums.
    modes = model.kernel_modes.reshape(-1, n_times, n_probe)
    n_modes = modes.shape[0]
    stacked = np.concatenate(
        [
            model.kernel_mean.reshape(n_times, n_probe),
            np.moveaxis(modes, 0, 1).reshape(n_times, -1),
        ],
        axis=1,
    )
    convolved = convolve(model.times, stacked, amplitudes, decay_constants)
    rise_mean = convolved[:, :n_probe]
    rise_modes = convolved[:, n_probe:].reshape(n_times, n_modes, n_probe)

    coefficients = model.kernel_coefficients(transformed)
    peaks = np.empty(samples.shape[0])
    for start in range(0, samples.shape[0], CHUNK):
        block = slice(start, start + CHUNK)
        rise = rise_mean[None, :, :] + np.einsum(
            "nm,tmp->ntp", coefficients[block], rise_modes, optimize=True
        )
        peaks[block] = (baseline[block][:, None, :] + loading[block] * rise).max(axis=(1, 2))
    return peaks


def sobol_indices(
    model: Surrogate,
    ranges: dict[str, tuple[float, float]],
    *,
    base_config: dict | None = None,
    n_base: int = 1024,
    seed: int = 0,
    n_bootstrap: int = 400,
    exact: bool = False,
) -> SobolResult:
    """First-order and total Sobol indices of the peak temperature.

    Base sample: scrambled Sobol' sequence.

    :param model: the surrogate.
    :param ranges: ``{input: (low, high)}``, inside the trained range for
        interpolated inputs.
    :param base_config: configuration for every input held fixed.
    :param n_base: base sample size ``N``; the cost is ``N (d + 2)``.
    :param seed: scramble seed of the Sobol' sequence.
    :param n_bootstrap: resamples for the standard errors.
    :param exact: convolve every kernel individually.
    :raises ValueError: on an invalid range.
    """
    from scipy.stats import qmc

    names = tuple(ranges)
    if not names:
        raise ValueError("at least one input must be varied")
    dimension = len(names)

    trained = model.trained_ranges()
    for name, (low, high) in ranges.items():
        if high <= low:
            raise ValueError(f"{name}: the range must be increasing, got ({low}, {high})")
        if name in trained:
            trained_low, trained_high = trained[name]
            if low < trained_low or high > trained_high:
                raise ValueError(
                    f"{name}: [{low:g}, {high:g}] leaves the trained range "
                    f"[{trained_low:g}, {trained_high:g}]"
                )

    lows = np.array([ranges[name][0] for name in names])
    highs = np.array([ranges[name][1] for name in names])

    # Matrices A and B from one 2d-dimensional sequence.
    block = qmc.Sobol(d=2 * dimension, scramble=True, seed=seed).random(n_base)
    a = lows + block[:, :dimension] * (highs - lows)
    b = lows + block[:, dimension:] * (highs - lows)

    def evaluate(points: np.ndarray) -> np.ndarray:
        return peak_temperatures(model, names, points, base_config=base_config, exact=exact)

    f_a = evaluate(a)
    f_b = evaluate(b)
    f_ab = np.empty((dimension, n_base))
    for i in range(dimension):
        ab = a.copy()
        ab[:, i] = b[:, i]
        f_ab[i] = evaluate(ab)

    first, total = _estimators(f_a, f_b, f_ab)

    rng = np.random.default_rng(seed)
    boot_first = np.empty((n_bootstrap, dimension))
    boot_total = np.empty((n_bootstrap, dimension))
    for draw in range(n_bootstrap):
        pick = rng.integers(0, n_base, n_base)
        boot_first[draw], boot_total[draw] = _estimators(f_a[pick], f_b[pick], f_ab[:, pick])

    combined = np.concatenate([f_a, f_b])
    return SobolResult(
        names=names,
        first_order=first,
        total=total,
        first_order_stderr=boot_first.std(axis=0),
        total_stderr=boot_total.std(axis=0),
        mean=float(combined.mean()),
        std=float(combined.std()),
        n_base=n_base,
        n_evaluations=n_base * (dimension + 2),
        ranges=dict(ranges),
    )


def _estimators(
    f_a: np.ndarray, f_b: np.ndarray, f_ab: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Saltelli first-order and Jansen total-index estimators."""
    variance = np.var(np.concatenate([f_a, f_b]))
    if variance <= 0.0:
        return np.zeros(f_ab.shape[0]), np.zeros(f_ab.shape[0])
    first = np.mean(f_b[None, :] * (f_ab - f_a[None, :]), axis=1) / variance
    total = 0.5 * np.mean((f_a[None, :] - f_ab) ** 2, axis=1) / variance
    return first, total
