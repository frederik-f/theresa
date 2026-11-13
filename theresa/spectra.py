# SPDX-License-Identifier: GPL-3.0-or-later
r"""Named multi-exponential decay-heat spectra.

The decay heat of a waste package is represented as a sum of exponentials

.. math::

    P(t) = \sum_{i=1}^{N} p_i \, 2^{-t / t_{1/2,i}}
         = \sum_{i=1}^{N} p_i \, e^{-\lambda_i t}, \quad
      \lambda_i = \ln 2 / t_{1/2,i}

with ``t`` counted from reactor discharge (spent fuel) or vitrification
(reprocessing waste).

Reference spectra: GRS-281, *Vorläufige Sicherheitsanalyse für den Standort
Gorleben: Endlagerauslegung und -optimierung (AP 6)*, GRS 2012, Annex B.1,
Eq. (B.1) and Tab. B.3.

The number of terms ``N`` is free throughout the package. Custom coefficients::

    cfg = make_config(decay_heat_p_kw=[...], decay_heat_t_half_years=[...])

:func:`fit_amplitudes` converts a tabulated heat curve into an ``N``-term
spectrum.
"""

from __future__ import annotations

import math
from types import MappingProxyType
from typing import NamedTuple

import numpy as np


class DecayHeatSpectrum(NamedTuple):
    """A multi-exponential decay-heat spectrum.

    :param p_kw: amplitudes :math:`p_i` [kW] per ``unit``.
    :param t_half_years: half-lives :math:`t_{1/2,i}` [a].
    :param unit: what one ``p_i`` refers to (fuel element, canister, ...).
    :param description: source of the data.
    """

    p_kw: tuple[float, ...]
    t_half_years: tuple[float, ...]
    unit: str
    description: str

    @property
    def n_terms(self) -> int:
        """Number of exponential terms."""
        return len(self.p_kw)

    def power_kw(self, years_since_discharge: np.ndarray | float) -> np.ndarray:
        """Evaluate :math:`P(t)` [kW per ``unit``] at the given times [a]."""
        t = np.atleast_1d(np.asarray(years_since_discharge, dtype=float))
        decay_const = math.log(2.0) / np.asarray(self.t_half_years)
        return np.sum(np.asarray(self.p_kw) * np.exp(-np.outer(t, decay_const)), axis=1)


#: Spent PWR fuel, 89 % UO2 / 11 % MOX by fuel-element count. Five terms.
DWR_MIX = DecayHeatSpectrum(
    p_kw=(3.318, 1.144, 0.2225, 0.02112, 0.0009342),
    t_half_years=(2.163, 32.33, 397.3, 13680.0, 760700.0),
    unit="fuel element",
    description="PWR spent fuel, 89/11 UO2/MOX mix (GRS-281 Tab. B.3, 'DWR-Mix 89/11')",
)

#: Vitrified high-level waste in a CSD-V canister. Four terms.
CSD_V = DecayHeatSpectrum(
    p_kw=(1.469, 0.04549, 0.001101, 0.0001074),
    t_half_years=(27.82, 420.4, 12370.0, 1.973e8),
    unit="canister",
    description="Vitrified HLW reprocessing waste (GRS-281 Tab. B.3, 'CSD-V')",
)

#: Named spectra.
SPECTRA = MappingProxyType(
    {
        "DWR-Mix": DWR_MIX,
        "CSD-V": CSD_V,
    }
)

#: Default half-lives [a] for :func:`fit_amplitudes`: six terms over ~1000 a.
SIX_TERM_HALF_LIVES = (2.0, 10.0, 40.0, 150.0, 600.0, 15000.0)


def get(name: str) -> DecayHeatSpectrum:
    """Spectrum by name, case-insensitive.

    :raises KeyError: if ``name`` is unknown.
    """
    for key, spectrum in SPECTRA.items():
        if key.lower() == name.lower():
            return spectrum
    raise KeyError(f"unknown spectrum {name!r}; available: {', '.join(SPECTRA)}")


def fit_amplitudes(
    times_years: np.ndarray,
    power_kw: np.ndarray,
    t_half_years: tuple[float, ...] = SIX_TERM_HALF_LIVES,
    *,
    unit: str = "package",
    description: str = "least-squares fit of a tabulated decay-heat curve",
) -> tuple[DecayHeatSpectrum, float]:
    """Fit non-negative amplitudes for fixed half-lives to a tabulated curve.

    Non-negative least squares on the residual relative to the tabulated
    power.

    :param times_years: sample times [a] since discharge, positive.
    :param power_kw: power [kW] at those times, positive.
    :param t_half_years: half-lives [a]; their number sets ``N``.
    :returns: the fitted spectrum and the largest relative error on the samples.
    :raises ValueError: on mismatched shapes or non-positive values.
    """
    from scipy.optimize import nnls

    t = np.asarray(times_years, dtype=float)
    target = np.asarray(power_kw, dtype=float)
    if t.shape != target.shape:
        raise ValueError("times_years and power_kw must have the same shape")
    if np.any(t <= 0.0) or np.any(target <= 0.0):
        raise ValueError("times and powers must be positive for a relative-error fit")

    decay_const = math.log(2.0) / np.asarray(t_half_years, dtype=float)
    basis = np.exp(-np.outer(t, decay_const))  # (n_samples, n_terms)

    weights = 1.0 / target
    amplitudes, _ = nnls(basis * weights[:, None], target * weights)

    fitted = DecayHeatSpectrum(
        p_kw=tuple(float(a) for a in amplitudes),
        t_half_years=tuple(float(th) for th in t_half_years),
        unit=unit,
        description=description,
    )
    max_rel_error = float(np.max(np.abs(fitted.power_kw(t) - target) / target))
    return fitted, max_rel_error
