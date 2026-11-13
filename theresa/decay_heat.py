# SPDX-License-Identifier: GPL-3.0-or-later
r"""Decay heat of one canister.

Spectra are given from discharge (:mod:`theresa.spectra`); the model time
starts at emplacement, ``interim_storage_years`` later. The shift is absorbed
into the amplitudes:

.. math::

    Q(t) = n \sum_i p_i \, e^{-\lambda_i (t + t_{store})}
         = \sum_i \underbrace{n \, p_i \, e^{-\lambda_i t_{store}}}_{P_i}
           \, e^{-\lambda_i t}

:func:`source_terms_watt` returns :math:`(P_i, \lambda_i)` for
:mod:`theresa.duhamel`.
"""

from __future__ import annotations

import math

import numpy as np

from .config import SECONDS_PER_YEAR, make_config


def source_terms_watt(cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Decay-heat source terms from emplacement.

    :param cfg: a configuration from :mod:`theresa.config`.
    :returns: ``(P, lam)``, amplitudes [W] per canister and decay constants
        [1/s]; the canister power is ``sum(P * exp(-lam * t))``, ``t`` in
        seconds since emplacement.
    """
    p_kw = np.asarray(cfg["decay_heat_p_kw"], dtype=float)
    t_half_years = np.asarray(cfg["decay_heat_t_half_years"], dtype=float)

    lam_per_year = math.log(2.0) / t_half_years
    amplitudes_watt = (
        cfg["n_fuel_elements"]
        * p_kw
        * 1000.0
        * np.exp(-lam_per_year * cfg["interim_storage_years"])
    )
    return amplitudes_watt, lam_per_year / SECONDS_PER_YEAR


def package_power_watt(cfg: dict, t_seconds: np.ndarray | float) -> np.ndarray:
    """Canister power [W] at times ``t_seconds`` since emplacement."""
    amplitudes, lam = source_terms_watt(cfg)
    t = np.atleast_1d(np.asarray(t_seconds, dtype=float))
    return np.sum(amplitudes * np.exp(-np.outer(t, lam)), axis=1)


def nominal_power_watt(cfg: dict) -> float:
    """Canister power [W] at emplacement (``t = 0``)."""
    return float(np.sum(source_terms_watt(cfg)[0]))


def volumetric_density_per_watt(cfg: dict, meshed_quarter_volume_m3: float | None = None) -> float:
    """Volumetric source density [(W/m3) per W of canister power].

    The quarter cell receives a quarter of the canister power. The density is
    referred to the meshed volume, over which OGS integrates, so that the
    injected power is exact.

    :param cfg: a configuration from :mod:`theresa.config`.
    :param meshed_quarter_volume_m3: volume [m3] of the meshed quarter
        canister; ``None`` uses the exact cylinder.
    :raises ValueError: if the volume is not positive.
    """
    if meshed_quarter_volume_m3 is None:
        return 1.0 / (math.pi * cfg["r_can"] ** 2 * cfg["l_can"])
    if meshed_quarter_volume_m3 <= 0.0:
        raise ValueError("meshed_quarter_volume_m3 must be positive")
    return 1.0 / (4.0 * meshed_quarter_volume_m3)


def build_decay_shape(cfg: dict, n_points: int = 120) -> tuple[np.ndarray, np.ndarray]:
    """Normalized decay curve, sampled logarithmically from ``t = 0``.

    :returns: ``(t_seconds, f)`` with ``f[0] == 1``, for an OGS ``<curve>``.
    """
    t_end = cfg["t_end_years"] * SECONDS_PER_YEAR
    t = np.concatenate(
        ([0.0], np.logspace(math.log10(SECONDS_PER_YEAR / 365.0), math.log10(t_end), n_points))
    )
    t = np.unique(t)
    power = package_power_watt(cfg, t)
    return t, power / power[0]


def build_decay_curve(
    cfg: dict, power_watt: float, meshed_quarter_volume_m3: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Volumetric heat source [W/m3] over time for a canister of ``power_watt``.

    :param meshed_quarter_volume_m3: see :func:`volumetric_density_per_watt`.
    :returns: ``(t_seconds, q)`` for an OGS ``<curve>``.
    """
    t, shape = build_decay_shape(cfg)
    return t, power_watt * volumetric_density_per_watt(cfg, meshed_quarter_volume_m3) * shape


def scaled_to_power(cfg: dict, canister_power_watt: float) -> dict:
    """Return ``cfg`` with all amplitudes scaled to a canister power.

    The shape of the decay curve is unchanged.

    :param cfg: a configuration from :mod:`theresa.config`.
    :param canister_power_watt: canister power at emplacement [W].
    :raises ValueError: if the power is not positive and finite, or the curve
        has no power at emplacement.
    """
    if not (math.isfinite(canister_power_watt) and canister_power_watt > 0.0):
        raise ValueError("canister_power_watt must be positive and finite")
    nominal = nominal_power_watt(cfg)
    if nominal <= 0.0:
        raise ValueError(
            "the decay curve delivers no power at emplacement, so there is no shape to "
            "scale to a canister power: every amplitude is zero or has decayed away "
            "during interim storage"
        )

    factor = float(canister_power_watt) / nominal
    return make_config(
        base=cfg, decay_heat_p_kw=[value * factor for value in cfg["decay_heat_p_kw"]]
    )
