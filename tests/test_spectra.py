# SPDX-License-Identifier: GPL-3.0-or-later
"""Reference spectra and the fit of tabulated decay-heat curves."""

from __future__ import annotations

import numpy as np
import pytest

from theresa.spectra import DWR_MIX, SPECTRA, fit_amplitudes, get


def test_get_is_case_insensitive():
    assert get("dwr-mix") is DWR_MIX
    with pytest.raises(KeyError):
        get("unknown")


@pytest.mark.parametrize("name", list(SPECTRA))
def test_fit_recovers_spectrum_with_its_half_lives(name):
    spectrum = SPECTRA[name]
    t = np.geomspace(1.0, 1000.0, 40)
    fitted, max_rel_error = fit_amplitudes(
        t, spectrum.power_kw(t), spectrum.t_half_years, unit=spectrum.unit
    )
    np.testing.assert_allclose(fitted.p_kw, spectrum.p_kw, rtol=1e-6, atol=1e-12)
    assert max_rel_error < 1e-8
    assert fitted.unit == spectrum.unit


def test_fit_with_default_half_lives_is_close_and_non_negative():
    t = np.geomspace(1.0, 1000.0, 60)
    fitted, max_rel_error = fit_amplitudes(t, DWR_MIX.power_kw(t))
    assert all(p >= 0.0 for p in fitted.p_kw)
    assert max_rel_error < 0.05


@pytest.mark.parametrize(
    ("t", "p"),
    [
        (np.array([1.0, 2.0]), np.array([1.0])),
        (np.array([0.0, 2.0]), np.array([1.0, 0.5])),
        (np.array([1.0, 2.0]), np.array([1.0, -0.5])),
    ],
)
def test_fit_rejects_invalid_input(t, p):
    with pytest.raises(ValueError):
        fit_amplitudes(t, p)
