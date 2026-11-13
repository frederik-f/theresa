# SPDX-License-Identifier: GPL-3.0-or-later
r"""Duhamel convolution of a step response with an exponential-sum source.

The temperature rise under a canister power :math:`Q(t)` follows from the unit
step response :math:`S(t)` (constant 1 W):

.. math::

    \Delta T(t) = \int_0^t \dot S(u)\, Q(t - u)\, du .

With :math:`S` piecewise linear between the solver's time steps and
:math:`Q(t) = \sum_i P_i e^{-\lambda_i t}`, every interval integrates in
closed form:

.. math::

    \Delta T(t) = \sum_i \frac{P_i}{\lambda_i} \sum_j s_j
        \left( e^{-\lambda_i (t - u_{j+1})} - e^{-\lambda_i (t - u_j)} \right),

where :math:`s_j` is the slope of :math:`S` on :math:`[u_j, u_{j+1}]`.
Amplitudes, decay constants and the number of terms are free.

:func:`convolve_exponential_kernel` is the closed form for an exponential-sum
kernel, used as a reference.
"""

from __future__ import annotations

import numpy as np

#: Decay constants up to this value are treated as zero (constant source).
ZERO_DECAY_TOLERANCE = 1.0e-30

#: Largest :math:`\lambda t_{end}` evaluated in factored form in
#: :func:`source_weights`; larger values use the direct form.
MAX_FACTORED_EXPONENT = 600.0

#: Relative distance below which :math:`\mu_k` and :math:`\lambda_i` are
#: treated as equal in :func:`convolve_exponential_kernel`.
DEGENERACY_TOLERANCE = 1.0e-8


def convolve(
    times_seconds: np.ndarray,
    response: np.ndarray,
    amplitudes_watt: np.ndarray,
    decay_constants: np.ndarray,
) -> np.ndarray:
    r"""Temperature rise [K] for an exponential-sum source.

    :param times_seconds: the kernel's sample times [s], ascending and starting
        at ``0``. The result is returned at these same times.
    :param response: unit step response :math:`S` [K/W] at those times; shape
        ``(n_times,)`` or ``(n_times, n_points)`` for several sample points.
    :param amplitudes_watt: source amplitudes :math:`P_i` [W] at ``t = 0``,
        from :func:`theresa.decay_heat.source_terms_watt`.
    :param decay_constants: source decay constants :math:`\lambda_i` [1/s].
        Zero denotes a constant source.
    :returns: the rise, shaped like ``response``.
    :raises ValueError: if the inputs are inconsistent or ``times`` is not
        ascending from zero.
    """
    t = np.asarray(times_seconds, dtype=float)
    s = np.asarray(response, dtype=float)
    p = np.asarray(amplitudes_watt, dtype=float).ravel()
    lam = np.asarray(decay_constants, dtype=float).ravel()

    if t.ndim != 1 or t.size < 2:
        raise ValueError("times_seconds must be a 1-D array with at least two entries")
    if t[0] != 0.0:
        raise ValueError("the step response must start at t = 0")
    if np.any(np.diff(t) <= 0.0):
        raise ValueError("times_seconds must be strictly ascending")
    if s.shape[0] != t.size:
        raise ValueError("response and times_seconds disagree on the number of times")
    if p.shape != lam.shape:
        raise ValueError("amplitudes_watt and decay_constants must have the same shape")
    if np.any(lam < 0.0):
        raise ValueError("decay constants must be non-negative")

    single = s.ndim == 1
    values = s.reshape(t.size, -1)

    # Interval slopes, (n_intervals, n_points).
    slopes = np.diff(values, axis=0) / np.diff(t)[:, None]

    result = source_weights(t, p, lam) @ slopes
    return result[:, 0] if single else result


def source_weights(
    times_seconds: np.ndarray,
    amplitudes_watt: np.ndarray,
    decay_constants: np.ndarray,
) -> np.ndarray:
    r"""Convolution matrix of a source, independent of the kernel.

    .. math::

        \Delta T(t_n) = \sum_j G_{nj}\, s_j ,
        \qquad
        G_{nj} = \sum_i \frac{P_i}{\lambda_i}
            \left( e^{-\lambda_i (t_n - u_{j+1})}
                  - e^{-\lambda_i (t_n - u_j)} \right) ,

    where :math:`s_j` is the slope of :math:`S` on interval :math:`j`.
    :math:`G` can be reused for several kernels on the same time grid.

    :param times_seconds: the kernel's sample times [s], ascending from zero.
    :param amplitudes_watt: source amplitudes :math:`P_i` [W] at ``t = 0``.
    :param decay_constants: source decay constants :math:`\lambda_i` [1/s].
    :returns: ``(n_times, n_times - 1)``, applied to the interval slopes.
    """
    t = np.asarray(times_seconds, dtype=float)
    p = np.asarray(amplitudes_watt, dtype=float).ravel()
    lam = np.asarray(decay_constants, dtype=float).ravel()
    n_t = t.size
    dt = np.diff(t)

    # Factored form, centred on c = t_end / 2:
    #   e^{-lam (t_n - u_{j+1})} - e^{-lam (t_n - u_j)}
    #     = e^{-lam (t_n - c)} * e^{lam (u_{j+1} - c)} * (-expm1(-lam dt_j)).
    # Beyond MAX_FACTORED_EXPONENT: direct form on the age matrix, ages >= 0.
    weights = np.zeros((n_t, n_t - 1))
    rows: list[np.ndarray] = []
    columns: list[np.ndarray] = []
    for amplitude, decay in zip(p, lam, strict=True):
        if amplitude == 0.0:
            continue
        if decay <= ZERO_DECAY_TOLERANCE:
            # Limit lam -> 0: amplitude * dt.
            rows.append(np.ones(n_t))
            columns.append(amplitude * dt)
            continue
        interval = -(amplitude / decay) * np.expm1(-decay * dt)
        if decay * t[-1] <= MAX_FACTORED_EXPONENT:
            centre = 0.5 * t[-1]
            rows.append(np.exp(-decay * (t - centre)))
            columns.append(interval * np.exp(decay * (t[1:] - centre)))
        else:
            age = np.maximum(t[:, None] - t[None, 1:], 0.0)
            weights += np.exp(-decay * age) * interval[None, :]
    if rows:
        weights += np.column_stack(rows) @ np.vstack(columns)

    # Causality: intervals j < n.
    weights *= np.tri(n_t, n_t - 1, k=-1, dtype=bool)
    return weights


def convolve_exponential_kernel(
    coefficients: np.ndarray,
    mu: np.ndarray,
    amplitudes_watt: np.ndarray,
    decay_constants: np.ndarray,
    times_seconds: np.ndarray,
) -> np.ndarray:
    r"""Convolution for a kernel :math:`S(t) = \sum_k a_k (1 - e^{-\mu_k t})`.

    Term-by-term integration of Duhamel's integral gives

    .. math::

        \Delta T(t) = \sum_k \sum_i a_k \mu_k P_i \,
            \frac{e^{-\lambda_i t} - e^{-\mu_k t}}{\mu_k - \lambda_i},

    and the limit :math:`a_k \mu_k P_i\, t\, e^{-\mu_k t}` at
    :math:`\mu_k = \lambda_i`. Reference solution for :func:`convolve`.
    """
    t = np.atleast_1d(np.asarray(times_seconds, dtype=float))
    mu = np.asarray(mu, dtype=float)
    a = np.asarray(coefficients, dtype=float)
    p = np.asarray(amplitudes_watt, dtype=float)
    lam = np.asarray(decay_constants, dtype=float)

    if a.shape[0] != mu.size:
        raise ValueError("coefficients and mu disagree on the number of terms")
    if p.shape != lam.shape:
        raise ValueError("amplitudes_watt and decay_constants must have the same shape")

    gap = mu[None, :] - lam[:, None]  # (n_lam, n_mu)
    scale = np.maximum(np.abs(mu)[None, :], np.abs(lam)[:, None])
    degenerate = np.abs(gap) <= DEGENERACY_TOLERANCE * np.maximum(scale, 1.0e-300)

    exp_lam = np.exp(-np.outer(t, lam))  # (n_t, n_lam)
    exp_mu = np.exp(-np.outer(t, mu))  # (n_t, n_mu)

    safe_gap = np.where(degenerate, 1.0, gap)
    regular = (exp_lam[:, :, None] - exp_mu[:, None, :]) / safe_gap[None, :, :]
    limiting = t[:, None, None] * exp_mu[:, None, :]
    contribution = np.where(degenerate[None, :, :], limiting, regular)

    weight = np.einsum("i,tik->tk", p, contribution) * mu[None, :]
    return weight @ a
