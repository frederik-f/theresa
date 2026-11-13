# SPDX-License-Identifier: GPL-3.0-or-later
"""Invalid and borderline inputs on all user-facing paths: refused or flagged."""

from __future__ import annotations

import math

import pytest

from theresa import cli
from theresa.config import make_config
from theresa.decay_heat import scaled_to_power
from theresa.predict import DEFAULT_SURROGATE_PATH, DEPTH_STUDIED_M, predict

needs_model = pytest.mark.skipif(not DEFAULT_SURROGATE_PATH.exists(), reason="no trained surrogate")


@pytest.mark.parametrize(
    "override",
    [
        {"n_fuel_elements": -2},
        {"interim_storage_years": -10.0},
        {"decay_heat_p_kw": [-1.0], "decay_heat_t_half_years": [30.0]},
        {"decay_heat_p_kw": [1.0], "decay_heat_t_half_years": [math.inf]},
        {"T_limit_celsius": math.nan},
        {"surface_temperature_celsius": math.inf},
    ],
)
def test_unphysical_configurations_are_refused(override):
    with pytest.raises(ValueError):
        make_config(spectrum="custom" if "decay_heat_p_kw" in override else None, **override)


def test_a_curve_without_power_cannot_be_scaled_to_one():
    # Every amplitude decayed below double precision during interim storage.
    cfg = make_config(
        spectrum="custom",
        decay_heat_p_kw=[1.0],
        decay_heat_t_half_years=[0.01],
        interim_storage_years=300.0,
    )
    with pytest.raises(ValueError, match="no power at emplacement"):
        scaled_to_power(cfg, 1000.0)
    for bad in (0.0, -5.0, math.inf, math.nan):
        with pytest.raises(ValueError, match="positive and finite"):
            scaled_to_power(make_config(), bad)


@needs_model
def test_a_shallow_depth_is_flagged():
    shallow = predict(9.5, 40.0, depth_m=DEPTH_STUDIED_M[0] - 100.0, n_fuel_elements=2)
    deep = predict(9.5, 40.0, depth_m=DEPTH_STUDIED_M[1] + 100.0, n_fuel_elements=2)
    assert any(getattr(w, "feature", None) == "depth_m" for w in shallow.warnings)
    assert not any(getattr(w, "feature", None) == "depth_m" for w in deep.warnings)


@needs_model
def test_the_command_line_reports_invalid_input_without_a_traceback(capsys):
    code = cli.main(["predict", "--decay-p", "1", "2", "--decay-half-life", "3"])
    captured = capsys.readouterr()
    assert code == 2
    assert "error:" in captured.err and "Traceback" not in captured.err


@needs_model
def test_a_peak_at_the_end_of_the_record_is_flagged():
    from theresa.predict import PeakAtRecordEnd

    # Close spacing and a long-lived source: still warming at 1000 a.
    late = predict(
        3.0, 16.0, decay_p_kw=[0.05, 1.0], decay_t_half_years=[5.0, 3000.0], n_fuel_elements=2
    )
    early = predict(9.5, 40.0, n_fuel_elements=2)
    assert any(isinstance(w, PeakAtRecordEnd) for w in late.warnings)
    assert not any(isinstance(w, PeakAtRecordEnd) for w in early.warnings)
