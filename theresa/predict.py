# SPDX-License-Identifier: GPL-3.0-or-later
"""Prediction API.

:func:`predict` returns the peak canister-surface temperature history for
spacings, material properties, depth, surface temperature, geothermal
gradient, decay-heat curve and loading. Spacings and material properties are
interpolated; all other inputs enter in closed form.

The loading is given as a number of fuel elements or as a thermal power at
emplacement. :meth:`Prediction.max_admissible_power_watt` returns the largest
loading that meets a temperature limit.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import (
    CONFIG,
    MATERIAL_ALIASES,
    SECONDS_PER_YEAR,
    design_value,
    emplacement_temperature_celsius,
    make_config,
)
from .decay_heat import package_power_watt, scaled_to_power, source_terms_watt
from .duhamel import source_weights
from .pipeline import History, case_subtitle, temperature_history
from .surrogate import OutsideTrainedRange, Surrogate

#: Peak-temperature uncertainty [K] above which a prediction is flagged.
INTERPOLATION_WARNING_KELVIN = 0.5


@dataclass(frozen=True)
class UncertainInterpolation:
    """Peak-temperature uncertainty above :data:`INTERPOLATION_WARNING_KELVIN`.

    :param sigma_kelvin: one-sigma uncertainty of the peak temperature [K].
    :param threshold_kelvin: the threshold applied.
    """

    sigma_kelvin: float
    threshold_kelvin: float

    def __str__(self) -> str:
        """English message."""
        return (
            f"the interpolation is uncertain here: +/- {self.sigma_kelvin:.2f} K on the peak "
            f"temperature, above the {self.threshold_kelvin:g} K the surrogate aims for"
        )


@dataclass(frozen=True)
class PeakAtRecordEnd:
    """Maximum at the last time of the record; the peak may lie later.

    :param record_years: length of the simulated record [a].
    """

    record_years: float

    def __str__(self) -> str:
        """English message."""
        return (
            f"the temperature is highest at the end of the {self.record_years:g} a record; "
            "the true peak may lie later"
        )


#: Validated depth range [m] of the analytic depth shift
#: (``scripts/run_depth_study.py``). Smaller depths are flagged.
DEPTH_STUDIED_M = (300.0, 1000.0)

#: Types in :attr:`Prediction.warnings`; ``str()`` gives the English message.
PredictionWarning = OutsideTrainedRange | UncertainInterpolation | PeakAtRecordEnd

#: Default surrogate file.
DEFAULT_SURROGATE_PATH = Path(__file__).resolve().parent.parent / "data" / "surrogate.pkl"


@dataclass
class Prediction:
    """Prediction for one design.

    :param history: temperature history per time and probe point.
    :param undisturbed_celsius: undisturbed rock temperature at emplacement
        level [degC].
    :param config: configuration of the prediction.
    :param warnings: see :data:`PredictionWarning`.
    :param temperature_sigma_kelvin: one-sigma interpolation uncertainty [K]
        of the peak temperature per time step, or ``None``. Does not include
        the model-form error.
    """

    history: History
    undisturbed_celsius: float
    config: dict
    warnings: list[PredictionWarning] = field(default_factory=list)
    temperature_sigma_kelvin: np.ndarray | None = None

    @property
    def times_years(self) -> np.ndarray:
        """Times [a] since emplacement."""
        return self.history.times_years

    @property
    def temperature_celsius(self) -> np.ndarray:
        """Peak canister-surface temperature [degC] over time."""
        return self.history.temperature_celsius

    @property
    def rise_kelvin(self) -> np.ndarray:
        """Temperature rise from heating [K] over time."""
        return self.history.rise_kelvin

    @property
    def canister_power_watt(self) -> float:
        """Canister thermal power at emplacement [W]."""
        return self.history.canister_power_watt

    @property
    def power_watt(self) -> np.ndarray:
        """Canister thermal power [W] at :attr:`times_years`."""
        return package_power_watt(self.config, self.times_years * SECONDS_PER_YEAR)

    def max_admissible_power_watt(self, limit_celsius: float | None = None) -> float:
        """Largest loading [W] that keeps the surface below ``limit_celsius``.

        Default: the configured limit. See
        :meth:`theresa.pipeline.History.max_admissible_power_watt`.
        """
        limit = self.config["T_limit_celsius"] if limit_celsius is None else limit_celsius
        return self.history.max_admissible_power_watt(limit)

    @property
    def peak_celsius(self) -> float:
        """Maximum canister-surface temperature [degC]."""
        return float(np.max(self.temperature_celsius))

    @property
    def peak_sigma_kelvin(self) -> float | None:
        """Interpolation uncertainty [K] at the time of the peak."""
        if self.temperature_sigma_kelvin is None:
            return None
        return float(self.temperature_sigma_kelvin[int(np.argmax(self.temperature_celsius))])

    @property
    def peak_time_years(self) -> float:
        """Time [a] at which the maximum occurs."""
        return float(self.times_years[int(np.argmax(self.temperature_celsius))])

    def plot(self, path: Path, **kwargs) -> Path:
        """Write the temperature history to ``path``.

        Keyword arguments are forwarded to
        :func:`theresa.plotting.plot_peak_temperature`.
        """
        from .plotting import plot_peak_temperature

        defaults = {
            "limit_celsius": self.config["T_limit_celsius"],
            "undisturbed_celsius": self.undisturbed_celsius,
            "sigma_kelvin": self.temperature_sigma_kelvin,
            "subtitle": case_subtitle(self.config),
        }
        return plot_peak_temperature(
            self.times_years, self.temperature_celsius, Path(path), **{**defaults, **kwargs}
        )

    def _format_admissible(self) -> str:
        """Admissible loading as text."""
        admissible = self.max_admissible_power_watt()
        if admissible == 0.0:
            return "     --- W (the undisturbed rock is already above the limit)"
        if not np.isfinite(admissible):
            return "  unlimited W"
        share = admissible / self.canister_power_watt
        return f"{admissible:8.0f} W ({share:.2f} x the loading above)"

    def to_csv(self, path: Path) -> Path:
        """Write the history as ``time_years, temperature_celsius, rise_kelvin``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        table = np.column_stack([self.times_years, self.temperature_celsius, self.rise_kelvin])
        np.savetxt(
            path,
            table,
            delimiter=",",
            header="time_years,temperature_celsius,rise_kelvin",
            comments="",
            fmt="%.6e",
        )
        return path

    def summary(self) -> str:
        """Text report of the key results."""
        limit = self.config["T_limit_celsius"]
        verdict = "EXCEEDS LIMIT" if self.peak_celsius > limit else "within limit"
        lines = [
            f"peak temperature   {self.peak_celsius:8.1f} degC   ({verdict}, limit {limit:g} degC)"
            + (
                f"  +/- {self.peak_sigma_kelvin:.2f} K interpolation"
                if self.peak_sigma_kelvin is not None
                else ""
            ),
            f"time of peak       {self.peak_time_years:8.1f} a",
            f"undisturbed rock   {self.undisturbed_celsius:8.1f} degC",
            f"heating at peak    {np.max(self.rise_kelvin):8.1f} K",
            f"at 1000 a          {self.temperature_celsius[-1]:8.1f} degC",
            f"canister loading   {self.canister_power_watt:8.0f} W at emplacement",
            f"max admissible     {self._format_admissible()} for the {limit:g} degC limit",
        ]
        if self.warnings:
            lines.append("warnings:")
            lines.extend(f"  - {warning}" for warning in self.warnings)
        return "\n".join(lines)


def predict(
    canister_gap_m: float,
    a_drift_m: float,
    *,
    canister_length_m: float | None = None,
    depth_m: float | None = None,
    surface_temperature_c: float | None = None,
    geothermal_gradient_k_per_m: float | None = None,
    decay_p_kw: Sequence[float] | None = None,
    decay_t_half_years: Sequence[float] | None = None,
    n_fuel_elements: int | None = None,
    canister_power_w: float | None = None,
    interim_storage_years: float | None = None,
    limit_celsius: float | None = None,
    materials: dict[str, float] | None = None,
    surrogate: Surrogate | Path | str | None = None,
    with_uncertainty: bool = True,
) -> Prediction:
    """Predict the peak canister-surface temperature history.

    :param canister_gap_m: clear distance between the end faces of adjacent
        canisters in a drift [m]; pitch = length + gap.
    :param a_drift_m: drift spacing, axis to axis [m].
    :param canister_length_m: canister length [m].
    :param depth_m: emplacement depth below ground level [m].
    :param surface_temperature_c: ground-level temperature [degC].
    :param geothermal_gradient_k_per_m: geothermal gradient [K/m].
    :param decay_p_kw: amplitudes of the decay-heat exponentials [kW per fuel
        element]; any number of terms.
    :param decay_t_half_years: corresponding half-lives [a].
    :param n_fuel_elements: fuel elements per canister.
    :param canister_power_w: thermal power at emplacement [W]; rescales the
        amplitudes and takes precedence over ``n_fuel_elements``.
    :param interim_storage_years: interim storage before emplacement [a].
    :param materials: material properties by alias
        (:data:`theresa.config.MATERIAL_ALIASES`), e.g.
        ``{"backfill_lambda": 0.6, "host_rock_rho_cp": 2.4e6}``; only
        interpolated properties are accepted.
    :param limit_celsius: design temperature limit [degC].
    :param surrogate: a :class:`~theresa.surrogate.Surrogate`, a path, or
        ``None`` for :data:`DEFAULT_SURROGATE_PATH`.
    :param with_uncertainty: propagate the GP variance to a one-sigma band [K].
    :raises FileNotFoundError: if no surrogate is available.
    :raises ValueError: on invalid inputs.
    """
    model = load_surrogate(surrogate)

    overrides: dict[str, object] = {
        "canister_gap": float(canister_gap_m),
        "a_drift": float(a_drift_m),
    }
    if canister_length_m is not None:
        overrides["l_can"] = _checked_fixed(model, "l_can", float(canister_length_m), "canister")
    if depth_m is not None:
        overrides["depth_m"] = float(depth_m)
    if surface_temperature_c is not None:
        overrides["surface_temperature_celsius"] = float(surface_temperature_c)
    if geothermal_gradient_k_per_m is not None:
        overrides["geothermal_gradient_k_per_m"] = float(geothermal_gradient_k_per_m)
    if n_fuel_elements is not None:
        overrides["n_fuel_elements"] = int(n_fuel_elements)
    if interim_storage_years is not None:
        overrides["interim_storage_years"] = float(interim_storage_years)
    if limit_celsius is not None:
        overrides["T_limit_celsius"] = float(limit_celsius)
    for name, value in (materials or {}).items():
        if name not in MATERIAL_ALIASES:
            raise ValueError(
                f"unknown material property {name!r}; expected one of {sorted(MATERIAL_ALIASES)}"
            )
        if name not in model.feature_names:
            raise ValueError(
                f"{name!r} is not interpolated by this surrogate, so changing it would "
                f"alter the configuration without altering the prediction. Interpolated: "
                f"{model.feature_names}"
            )
        overrides[name] = float(value)
    if (decay_p_kw is None) != (decay_t_half_years is None):
        raise ValueError("pass both decay_p_kw and decay_t_half_years, or neither")
    if decay_p_kw is not None:
        overrides["decay_heat_p_kw"] = [float(value) for value in decay_p_kw]
        overrides["decay_heat_t_half_years"] = [float(value) for value in decay_t_half_years]
        overrides["spectrum"] = "custom"

    cfg = make_config(**overrides)

    if canister_power_w is not None:
        cfg = scaled_to_power(cfg, canister_power_w)

    design = {name: design_value(cfg, name) for name in model.feature_names}
    kernel, psi, range_warnings, mode_sigma = model.predict_kernel(
        design, with_uncertainty=with_uncertainty
    )
    warnings: list[PredictionWarning] = list(range_warnings)

    if "depth_m" not in model.feature_names and cfg["depth_m"] < DEPTH_STUDIED_M[0]:
        warnings.append(OutsideTrainedRange("depth_m", float(cfg["depth_m"]), *DEPTH_STUDIED_M))

    if "depth_m" not in model.feature_names:
        # Depth as a shift of psi.
        psi = psi + (cfg["depth_m"] - _trained_depth(model))

    history = temperature_history(cfg, model.times, kernel, psi)

    if int(np.argmax(history.temperature_celsius)) == history.times_years.size - 1:
        warnings.append(PeakAtRecordEnd(float(history.times_years[-1])))

    sigma = None
    if with_uncertainty:
        sigma = _propagate_uncertainty(cfg, model, mode_sigma, history)
        peak_sigma = float(sigma[int(np.argmax(history.temperature_celsius))])
        if peak_sigma > INTERPOLATION_WARNING_KELVIN:
            warnings.append(UncertainInterpolation(peak_sigma, INTERPOLATION_WARNING_KELVIN))

    return Prediction(
        history=history,
        temperature_sigma_kelvin=sigma,
        undisturbed_celsius=emplacement_temperature_celsius(cfg),
        config=cfg,
        warnings=warnings,
    )


def _propagate_uncertainty(
    cfg: dict, model: Surrogate, mode_sigma: np.ndarray, history: History
) -> np.ndarray:
    """GP posterior uncertainty propagated to the peak temperature [K].

    Linear propagation of ``sigma_k * mode_k`` through the convolution,
    quadrature sum over the modes, evaluated at the hottest probe point of
    each time step.

    :param mode_sigma: one-sigma uncertainty of each kernel mode coefficient.
    :returns: one-sigma uncertainty [K] of the peak temperature per time step.
    """
    n_times, n_probe = model.probe_shape
    hottest = np.argmax(history.baseline_probe[None, :] + history.rise_probe, axis=1)
    candidates, which = np.unique(hottest, return_inverse=True)

    # Modes at the candidate points: (n_times, n_modes, n_candidates).
    modes = model.kernel_modes.reshape(-1, n_times, n_probe)[:, :, candidates].transpose(1, 0, 2)
    slopes = np.diff(modes, axis=0) / np.diff(model.times)[:, None, None]

    amplitudes, decay_constants = source_terms_watt(cfg)
    weights = source_weights(model.times, amplitudes, decay_constants)
    convolved = (weights @ slopes.reshape(n_times - 1, -1)).reshape(n_times, *slopes.shape[1:])

    per_mode = convolved[np.arange(n_times), :, which] * mode_sigma[None, :]
    return np.sqrt(np.sum(per_mode**2, axis=1))


def _checked_fixed(model: Surrogate, key: str, value: float, what: str) -> float:
    """``value`` if it equals the training value of a fixed quantity.

    :param model: the surrogate.
    :param key: configuration key.
    :param value: requested value.
    :param what: name of the quantity, for the message.
    :raises ValueError: if the value differs or is not recorded.
    """
    trained = (model.fixed_design or {}).get(key)
    if trained is None:
        raise ValueError(
            f"this surrogate does not record the {what} geometry it was trained on, so a "
            f"requested {key} cannot be checked against it. Retrain, or leave {key} unset."
        )
    if not math.isclose(value, trained, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(
            f"this surrogate was trained with {key} = {trained:g} and does not interpolate "
            f"it, so it cannot answer for {value:g}. A different {what} is a different "
            f"design and needs its own training run."
        )
    return value


def _trained_depth(model: Surrogate) -> float:
    """Depth [m] of the surrogate's ``psi``."""
    if model.fixed_depth_m is not None:
        return float(model.fixed_depth_m)
    return float(CONFIG["depth_m"])


def load_surrogate(surrogate: Surrogate | Path | str | None = None) -> Surrogate:
    """A :class:`Surrogate`, loaded from file if necessary.

    :param surrogate: a surrogate, a path to one, or ``None`` for
        :data:`DEFAULT_SURROGATE_PATH`.
    :raises FileNotFoundError: if there is no file at the path.
    """
    if isinstance(surrogate, Surrogate):
        return surrogate
    path = Path(surrogate) if surrogate is not None else DEFAULT_SURROGATE_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"no trained surrogate at {path}. Generate training data and fit one with "
            "'theresa train', or pass an explicit path."
        )
    return Surrogate.load(path)
