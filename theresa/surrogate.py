# SPDX-License-Identifier: GPL-3.0-or-later
"""Interpolation of step response and ``psi`` over the design variables.

The surrogate maps

    (canister spacing, drift spacing, material properties)  ->  S(t, probe point), psi

Kernels and ``psi`` fields are reduced by SVD to a mean and a few modes (POD);
each mode coefficient is interpolated by a Gaussian process (Matérn 5/2, ARD)
in the transformed design coordinates (``1 / x`` where flagged). The GP
variance gives the prediction uncertainty.
"""

from __future__ import annotations

import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
from sklearn.exceptions import ConvergenceWarning

#: Fraction of the variance captured by the retained SVD modes.
ENERGY_THRESHOLD = 1.0 - 1.0e-8

#: Maximum number of retained modes.
MAX_MODES = 8


def transform_design(points: np.ndarray, reciprocal: np.ndarray) -> np.ndarray:
    """Design variables in fitting coordinates.

    :param points: ``(n_points, n_features)`` raw design variables.
    :param reciprocal: boolean per feature; where true, ``1 / x`` is used.
    """
    values = np.atleast_2d(np.asarray(points, dtype=float)).copy()
    if values.shape[1] != reciprocal.size:
        raise ValueError("points and reciprocal disagree on the number of features")
    if np.any(values[:, reciprocal] <= 0.0):
        raise ValueError("reciprocal features must be positive")
    values[:, reciprocal] = 1.0 / values[:, reciprocal]
    return values


@dataclass(frozen=True)
class OutsideTrainedRange:
    """A design variable outside the trained range.

    :param feature: the entry of :attr:`Surrogate.feature_names` concerned.
    :param value: the requested value, in SI units.
    :param low, high: the trained range, in SI units.
    """

    feature: str
    value: float
    low: float
    high: float

    def __str__(self) -> str:
        """English message."""
        return (
            f"{self.feature} = {self.value:g} is outside the trained range "
            f"[{self.low:g}, {self.high:g}]"
        )


class KernelPrediction(NamedTuple):
    """Result of :meth:`Surrogate.predict_kernel`.

    :param kernel: step response [K/W], ``(n_times, n_probe)``, starting at
        zero and non-decreasing.
    :param psi: ``psi`` [m] at the probe points, ``(n_probe,)``.
    :param warnings: features outside the trained range.
    :param mode_sigma: posterior standard deviation of each kernel mode
        coefficient, ``(n_modes,)``, or ``None``.
    """

    kernel: np.ndarray
    psi: np.ndarray
    warnings: list[OutsideTrainedRange]
    mode_sigma: np.ndarray | None = None


@dataclass
class Surrogate:
    """A trained kernel interpolator.

    :param times: time grid [s] of the kernels.
    :param feature_names: names of the interpolated design variables.
    :param reciprocal: per feature, whether it is fitted as ``1 / x``.
    :param train_points: ``(n_designs, n_features)`` raw training designs.
    :param probe_shape: ``(n_times, n_probe)`` shape of a kernel.
    :param kernel_mean: mean kernel over the training set, flattened.
    :param kernel_modes: ``(n_modes, n_features_flat)`` SVD modes.
    :param kernel_models: one regressor per mode coefficient.
    :param psi_mean, psi_modes, psi_models: the same for ``psi``.
    :param fixed_depth_m: depth of the training runs, if depth is not a
        feature.
    :param fixed_design: configuration values held fixed in training.
    """

    times: np.ndarray
    feature_names: tuple[str, ...]
    reciprocal: np.ndarray
    train_points: np.ndarray
    probe_shape: tuple[int, int]
    kernel_mean: np.ndarray
    kernel_modes: np.ndarray
    kernel_models: list
    psi_mean: np.ndarray
    psi_modes: np.ndarray
    psi_models: list
    fixed_depth_m: float | None = None
    fixed_design: dict[str, float] | None = None

    def trained_ranges(self) -> dict[str, tuple[float, float]]:
        """Trained range of each design variable, SI units."""
        low, high = self.train_points.min(axis=0), self.train_points.max(axis=0)
        return {name: (float(low[i]), float(high[i])) for i, name in enumerate(self.feature_names)}

    def predict_kernel(
        self, design: dict[str, float], *, with_uncertainty: bool = False
    ) -> KernelPrediction:
        """Step response and ``psi`` for one design.

        :param design: values for every name in :attr:`feature_names`.
        :param with_uncertainty: also return the standard deviation of every
            kernel mode coefficient.
        :raises ValueError: if ``design`` lacks a feature.
        """
        missing = [name for name in self.feature_names if name not in design]
        if missing:
            raise ValueError(f"design is missing {', '.join(missing)}")

        raw = np.array([[float(design[name]) for name in self.feature_names]])
        transformed = transform_design(raw, self.reciprocal)
        coefficients, sigma = _evaluate(self.kernel_models, transformed, with_std=with_uncertainty)
        return KernelPrediction(
            kernel=self._kernels_from(coefficients)[0],
            psi=self.psis(transformed)[0],
            warnings=self._range_warnings(raw[0]),
            mode_sigma=None if sigma is None else sigma[0],
        )

    def kernel_coefficients(self, transformed: np.ndarray) -> np.ndarray:
        """Kernel mode coefficients for a batch of designs, ``(n, n_modes)``.

        :param transformed: designs as returned by :func:`transform_design`.
        """
        return _evaluate(self.kernel_models, transformed)[0]

    def kernels(self, transformed: np.ndarray) -> np.ndarray:
        """Step responses [K/W] for a batch of designs, ``(n, n_times, n_probe)``.

        :param transformed: designs as returned by :func:`transform_design`.
        """
        return self._kernels_from(self.kernel_coefficients(transformed))

    def psis(self, transformed: np.ndarray) -> np.ndarray:
        """``psi`` [m] for a batch of designs, ``(n, n_probe)``.

        :param transformed: designs as returned by :func:`transform_design`.
        """
        return self.psi_mean + _evaluate(self.psi_models, transformed)[0] @ self.psi_modes

    def _kernels_from(self, coefficients: np.ndarray) -> np.ndarray:
        """Step responses from mode coefficients; zero at ``t = 0``, non-decreasing."""
        kernels = (self.kernel_mean + coefficients @ self.kernel_modes).reshape(
            -1, *self.probe_shape
        )
        kernels[:, 0] = 0.0
        np.maximum.accumulate(kernels, axis=1, out=kernels)
        return kernels

    def _range_warnings(self, raw: np.ndarray) -> list[OutsideTrainedRange]:
        """Design variables outside the trained range."""
        return [
            OutsideTrainedRange(name, float(value), low, high)
            for value, (name, (low, high)) in zip(raw, self.trained_ranges().items(), strict=True)
            if not low <= value <= high
        ]

    def save(self, path: Path) -> Path:
        """Write the surrogate to a pickle file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    @staticmethod
    def load(path: Path) -> Surrogate:
        """Load a surrogate written by :meth:`save`."""
        with Path(path).open("rb") as handle:
            loaded = pickle.load(handle)
        if not isinstance(loaded, Surrogate):
            raise TypeError(f"{path} does not contain a Surrogate")
        return loaded


def _evaluate(
    models: list, transformed: np.ndarray, *, with_std: bool = False
) -> tuple[np.ndarray, np.ndarray | None]:
    """Evaluate the regressors of all modes on a batch of designs.

    :returns: ``(mean, std)``, each ``(n_designs, n_modes)``; ``std`` is
        ``None`` unless ``with_std``.
    """
    if not with_std:
        return np.column_stack([model.predict(transformed) for model in models]), None
    means, stds = zip(
        *(model.predict(transformed, return_std=True) for model in models), strict=True
    )
    return np.column_stack(means), np.column_stack(stds)


def fit(
    times: np.ndarray,
    design_points: np.ndarray,
    kernels: np.ndarray,
    psis: np.ndarray,
    feature_names: tuple[str, ...],
    *,
    reciprocal: tuple[bool, ...] | None = None,
    max_modes: int = MAX_MODES,
    fixed_depth_m: float | None = None,
    fixed_design: dict[str, float] | None = None,
) -> Surrogate:
    """Train the surrogate on simulated design points.

    :param times: shared time grid [s], shape ``(n_times,)``.
    :param design_points: ``(n_designs, n_features)`` raw design variables.
    :param kernels: ``(n_designs, n_times, n_probe)`` step responses [K/W].
    :param psis: ``(n_designs, n_probe)`` ``psi`` fields [m].
    :param feature_names: one name per design variable.
    :param reciprocal: per feature, whether to fit in ``1 / x``; default all.
    :param max_modes: maximum number of SVD modes.
    :param fixed_depth_m: depth of the runs, if depth is not a feature.
    :param fixed_design: configuration values held fixed in the runs.
    """
    points = np.atleast_2d(np.asarray(design_points, dtype=float))
    kernels = np.asarray(kernels, dtype=float)
    psis = np.atleast_2d(np.asarray(psis, dtype=float))

    n_designs = points.shape[0]
    if kernels.shape[0] != n_designs or psis.shape[0] != n_designs:
        raise ValueError("design_points, kernels and psis disagree on the number of designs")
    if points.shape[1] != len(feature_names):
        raise ValueError("design_points and feature_names disagree on the number of features")
    if n_designs < 3:
        raise ValueError("at least three design points are needed to fit anything")

    flags = (
        np.ones(len(feature_names), dtype=bool)
        if reciprocal is None
        else np.asarray(reciprocal, dtype=bool)
    )
    transformed = transform_design(points, flags)

    probe_shape = (kernels.shape[1], kernels.shape[2])
    flat = kernels.reshape(n_designs, -1)

    kernel_mean, kernel_modes, kernel_coefficients = _decompose(flat, max_modes)
    psi_mean, psi_modes, psi_coefficients = _decompose(psis, max_modes)

    return Surrogate(
        times=np.asarray(times, dtype=float),
        feature_names=tuple(feature_names),
        reciprocal=flags,
        train_points=points,
        probe_shape=probe_shape,
        kernel_mean=kernel_mean,
        kernel_modes=kernel_modes,
        kernel_models=[_fit_mode(transformed, c) for c in kernel_coefficients.T],
        psi_mean=psi_mean,
        psi_modes=psi_modes,
        psi_models=[_fit_mode(transformed, c) for c in psi_coefficients.T],
        fixed_depth_m=fixed_depth_m,
        fixed_design=dict(fixed_design) if fixed_design else None,
    )


def _decompose(data: np.ndarray, max_modes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean-centre and take the leading SVD modes.

    :returns: ``(mean, modes, coefficients)`` with ``data[d] approximately
        mean + coefficients[d] @ modes``.
    """
    mean = data.mean(axis=0)
    centred = data - mean

    left, singular, right = np.linalg.svd(centred, full_matrices=False)

    energy = singular**2
    total = energy.sum()
    if total <= 0.0:
        # No variation: one zero mode.
        return mean, np.zeros((1, data.shape[1])), np.zeros((data.shape[0], 1))

    cumulative = np.cumsum(energy) / total
    n_modes = int(np.searchsorted(cumulative, ENERGY_THRESHOLD) + 1)
    n_modes = max(1, min(n_modes, max_modes, singular.size))

    modes = right[:n_modes]
    coefficients = left[:, :n_modes] * singular[:n_modes]
    return mean, modes, coefficients


def _fit_mode(transformed: np.ndarray, values: np.ndarray):
    """Gaussian process for one mode coefficient.

    Matérn 5/2 with ARD length scales plus a white-noise term; inputs and
    target standardized.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

    inputs, length_scales = _standardize_inputs(transformed)

    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=length_scales, length_scale_bounds=(1e-2, 1e2), nu=2.5
    ) + WhiteKernel(1e-8, (1e-12, 1e-1))

    model = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=4)

    with warnings.catch_warnings():
        # Accuracy is validated against held-out designs.
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model.fit(inputs.transform(transformed), values)

    return _StandardizedGP(model, inputs)


@dataclass
class _InputScaler:
    """Affine scaling of design inputs to unit range."""

    centre: np.ndarray
    scale: np.ndarray

    def transform(self, points: np.ndarray) -> np.ndarray:
        """Centre and scale ``points``; returns a 2-D array."""
        return (np.atleast_2d(points) - self.centre) / self.scale


@dataclass
class _StandardizedGP:
    """Gaussian process with input scaling; ``predict`` as in scikit-learn."""

    model: object
    scaler: _InputScaler

    def predict(self, points: np.ndarray, return_std: bool = False):
        """Predict at ``points`` in unscaled coordinates."""
        return self.model.predict(self.scaler.transform(points), return_std=return_std)


def _standardize_inputs(transformed: np.ndarray) -> tuple[_InputScaler, np.ndarray]:
    """Input scaler and initial length scales."""
    centre = transformed.mean(axis=0)
    spread = np.ptp(transformed, axis=0)
    spread[spread <= 0.0] = 1.0
    return _InputScaler(centre=centre, scale=spread), np.ones(transformed.shape[1])
