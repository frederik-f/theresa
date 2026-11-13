# SPDX-License-Identifier: GPL-3.0-or-later
"""Configuration: defaults, validation rules and derived geometry."""

from __future__ import annotations

import math

import pytest

from theresa import config, spectra
from theresa.config import (
    canister_pitch,
    domain_bottom_z,
    domain_length_x,
    domain_width_y,
    drift_axis_z,
    drift_inner_radius,
    drift_outer_radius,
    emplacement_temperature_celsius,
    make_config,
    quarter_canister_volume,
    undisturbed_temperature_celsius,
)


def test_defaults_are_valid():
    make_config()  # raises if not


def test_unknown_key_is_a_typo_guard():
    with pytest.raises(KeyError, match="unknown configuration key"):
        make_config(a_cann=12.0)


def test_named_spectrum_replaces_the_coefficients():
    cfg = make_config(spectrum="CSD-V")
    assert cfg["decay_heat_p_kw"] == list(spectra.CSD_V.p_kw)
    assert cfg["decay_heat_t_half_years"] == list(spectra.CSD_V.t_half_years)


def test_explicit_coefficients_make_the_spectrum_name_a_label():
    # "custom" uses the given coefficients without a registry lookup.
    cfg = make_config(
        spectrum="custom",
        decay_heat_p_kw=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        decay_heat_t_half_years=[2.0, 10.0, 40.0, 150.0, 600.0, 15000.0],
    )
    assert cfg["spectrum"] == "custom"
    assert len(cfg["decay_heat_p_kw"]) == 6


def test_base_is_applied_before_overrides():
    stored = make_config(canister_gap=20.0, a_drift=50.0)
    cfg = make_config(base=stored, canister_gap=12.0)
    assert cfg["canister_gap"] == 12.0
    assert cfg["a_drift"] == 50.0


# --- validation rules ------------------------------------------------------


def test_canister_must_fit_inside_the_drift():
    with pytest.raises(ValueError, match="does not fit"):
        make_config(r_can=2.0, drift_diameter=3.5)


def test_the_canister_gap_must_be_positive():
    # Any positive clear spacing is valid for any canister length.
    with pytest.raises(ValueError, match="canister_gap"):
        make_config(canister_gap=0.0)


def test_the_pitch_is_derived_from_the_length_and_the_gap():
    cfg = make_config(l_can=5.52, canister_gap=9.5)
    assert canister_pitch(cfg) == pytest.approx(15.02)


def test_drifts_must_not_overlap():
    with pytest.raises(ValueError, match="must exceed the excavated drift"):
        make_config(a_drift=3.0)


def test_decay_heat_lists_must_match():
    with pytest.raises(ValueError, match="equal length"):
        make_config(decay_heat_p_kw=[1.0, 2.0], decay_heat_t_half_years=[1.0])


def test_half_lives_must_be_positive():
    with pytest.raises(ValueError, match="half-life must be positive"):
        make_config(decay_heat_p_kw=[1.0], decay_heat_t_half_years=[-1.0])


def test_probe_inset_must_be_inside_the_canister():
    with pytest.raises(ValueError, match="probe_inset_m"):
        make_config(probe_inset_m=2.0)


def test_unknown_linear_solver_is_rejected():
    with pytest.raises(ValueError, match="unknown linear_solver"):
        make_config(linear_solver="PETSc")


# --- derived geometry ------------------------------------------------------


def test_quarter_cell_extents_are_half_the_pitches():
    # Lateral extent: half the spacing in each direction.
    cfg = make_config(l_can=6.0, canister_gap=8.0, a_drift=60.0)
    assert domain_length_x(cfg) == pytest.approx(7.0)  # (6 + 8) / 2
    assert domain_width_y(cfg) == pytest.approx(30.0)


def test_vertical_extent_keeps_a_constant_clearance_below_the_drift():
    # Depth leaves the distance to the bottom boundary unchanged.
    for depth in (300.0, 700.0, 1000.0):
        cfg = make_config(depth_m=depth, h_below_m=700.0)
        assert drift_axis_z(cfg) == pytest.approx(-depth)
        assert domain_bottom_z(cfg) == pytest.approx(-(depth + 700.0))
        assert drift_axis_z(cfg) - domain_bottom_z(cfg) == pytest.approx(700.0)


def test_liner_sits_between_the_backfill_and_the_rock():
    cfg = make_config(drift_diameter=3.5, lining_thickness=0.3)
    assert drift_inner_radius(cfg) == pytest.approx(1.75)
    assert drift_outer_radius(cfg) == pytest.approx(2.05)


def test_quarter_canister_volume_is_a_quarter_of_the_cylinder():
    cfg = make_config()
    full = math.pi * cfg["r_can"] ** 2 * cfg["l_can"]
    assert quarter_canister_volume(cfg) == pytest.approx(0.25 * full)


def test_undisturbed_temperature_follows_the_gradient_downwards():
    cfg = make_config(surface_temperature_celsius=9.0, geothermal_gradient_k_per_m=0.032)
    assert undisturbed_temperature_celsius(cfg, 0.0) == pytest.approx(9.0)
    assert undisturbed_temperature_celsius(cfg, -500.0) == pytest.approx(9.0 + 16.0)


def test_emplacement_temperature_is_the_undisturbed_value_at_depth():
    cfg = make_config(
        depth_m=800.0, surface_temperature_celsius=10.0, geothermal_gradient_k_per_m=0.03
    )
    assert emplacement_temperature_celsius(cfg) == pytest.approx(34.0)


def test_pollux_dimensions_are_the_defaults():
    cfg = make_config()
    assert cfg["l_can"] == pytest.approx(5.52)
    assert 2.0 * cfg["r_can"] == pytest.approx(1.56)


# --- run directories -------------------------------------------------------


def test_run_directory_name_encodes_the_design():
    cfg = make_config(canister_gap=12.5, a_drift=60.0, l_can=5.5, depth_m=500.0)
    name = config.run_directory_name(cfg)
    assert "gap12p5" in name
    assert "adrift60" in name and "len5p5" in name and "depth500" in name


def test_prepare_run_directory_removes_only_known_artifacts(tmp_path):
    work = tmp_path / "run"
    work.mkdir()
    (work / "config.json").write_text("{}", encoding="utf-8")
    (work / "mesh").mkdir()
    (work / "notes.md").write_text("keep me", encoding="utf-8")

    config.prepare_run_directory(work)

    assert not (work / "config.json").exists()
    assert not (work / "mesh").exists()
    assert (work / "notes.md").read_text(encoding="utf-8") == "keep me"


def test_config_round_trips_through_json(tmp_path):
    cfg = make_config(canister_gap=17.0, depth_m=650.0)
    config.write_config(cfg, tmp_path)
    assert config.read_config(tmp_path) == cfg
