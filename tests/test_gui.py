# SPDX-License-Identifier: GPL-3.0-or-later
"""Structure of the Streamlit GUI: imports, entry point, use of the model API."""

from __future__ import annotations

import inspect

import pytest

streamlit = pytest.importorskip("streamlit")

from theresa_gui import app, i18n, launch  # noqa: E402


def test_the_entry_point_targets_an_existing_app():
    assert launch.APP.exists()
    assert launch.APP.name == "app.py"


def test_the_page_has_the_pieces_it_advertises():
    for name in ("main", "sidebar_inputs", "show_headline", "show_charts", "direct_controls"):
        assert callable(getattr(app, name)), name


def test_the_direct_run_uses_the_same_pipeline_as_the_cli():
    # The comparison runs the model pipeline.
    source = inspect.getsource(app.start_direct_run)
    assert "run_case" in source
    assert "make_config" in source


def test_the_uncertainty_is_labelled_as_interpolation_only():
    # The uncertainty is labelled as interpolation uncertainty.
    assert 't("uncertainty_help")' in inspect.getsource(app.show_headline)
    wording = i18n.STRINGS["uncertainty_help"]
    assert "Excludes" in wording["en"]
    assert "Ohne" in wording["de"]
    for language in i18n.LANGUAGES:
        assert "Gauß" in wording[language] or "Gaussian" in wording[language]


def test_the_decay_curve_offers_six_terms():
    source = inspect.getsource(app.sidebar_inputs)
    assert "range(6)" in source


def test_the_fit_mode_solves_for_the_loading_in_closed_form():
    # Admissible loading from the closed-form inversion (checked on the AST).
    import ast
    import textwrap

    assert "max_admissible_power_watt" in inspect.getsource(app.evaluate)

    tree = ast.parse(textwrap.dedent(inspect.getsource(app.evaluate)))
    loops = [node for node in ast.walk(tree) if isinstance(node, (ast.For, ast.While))]
    assert loops == [], "the fitted loading must not be found by iterating"


def test_the_design_limit_reaches_the_prediction():
    source = inspect.getsource(app.evaluate)
    assert "limit_celsius" in source


def test_the_direct_run_checks_the_fitted_loading():
    # In fit mode the comparison uses the fitted loading.
    assert "canister_power_w" in inspect.signature(app.start_direct_run).parameters
    assert "canister_power_watt" in inspect.getsource(app.main)


def test_the_page_is_sized_to_fit_one_screen():
    # Compact layout.
    source = inspect.getsource(app.main)
    assert "stMainBlockContainer" in source
    assert "padding-top" in source and "padding-bottom" in source


def test_the_layout_limits_are_viewport_relative():
    # Chart and metric sizes in viewport units.
    source = inspect.getsource(app.main)
    assert "vh" in source
    assert "clamp(" in source


def test_the_chart_is_shorter_than_the_command_line_default():
    # Figure size of the page.
    assert "figsize=CHART_SIZE" in inspect.getsource(app.svg)
    assert app.CHART_SIZE[0] / app.CHART_SIZE[1] < 8.0 / 4.8 * 1.5


def test_a_finished_ogs_run_is_dropped_when_the_inputs_change():
    # A run of a different design is not shown.
    source = inspect.getsource(app.direct_result)
    assert "run_fingerprint" in source
    assert "direct_result" in source and "pop" in source


def test_the_fingerprint_covers_what_the_simulation_depends_on():
    design = {"canister_gap": 9.5, "a_drift": 40.0, "l_can": 5.52, "depth_m": 500.0}
    physics = {"surface_temperature_c": 10.0, "gradient": 0.03, "interim_storage": 50.0}
    decay = {"p_kw": [3.3, 1.1], "t_half": [2.2, 32.3]}

    base = app.run_fingerprint(design, physics, decay, 1235.0)

    assert app.run_fingerprint({**design, "canister_gap": 9.6}, physics, decay, 1235.0) != base
    assert app.run_fingerprint({**design, "l_can": 6.0}, physics, decay, 1235.0) != base
    assert app.run_fingerprint(design, {**physics, "gradient": 0.031}, decay, 1235.0) != base
    assert app.run_fingerprint(design, physics, {**decay, "p_kw": [3.4, 1.1]}, 1235.0) != base
    assert app.run_fingerprint(design, physics, decay, 1300.0) != base


def test_the_design_limit_does_not_invalidate_a_finished_run():
    # The limit does not invalidate a run.
    design = {"canister_gap": 9.5, "a_drift": 40.0, "l_can": 5.52, "depth_m": 500.0}
    decay = {"p_kw": [3.3], "t_half": [2.2]}
    physics = {"surface_temperature_c": 10.0, "gradient": 0.03, "interim_storage": 50.0}

    with_limit = {**physics, "limit_celsius": 100.0}
    other_limit = {**physics, "limit_celsius": 75.0}

    assert app.run_fingerprint(design, with_limit, decay, 1235.0) == app.run_fingerprint(
        design, other_limit, decay, 1235.0
    )


def test_the_spacing_is_labelled_as_a_clear_distance_everywhere_it_is_shown():
    # The axial spacing is labelled "clear canister spacing".
    from theresa.config import make_config
    from theresa.pipeline import case_subtitle

    sidebar = inspect.getsource(app.sidebar_inputs)
    for key in ("canister_spacing", "canister_spacing_help", "resulting_pitch"):
        assert f'"{key}"' in sidebar, key
    assert "Clear canister spacing" in i18n.STRINGS["canister_spacing"]["en"]
    assert "Lichter Behälterabstand" in i18n.STRINGS["canister_spacing"]["de"]
    assert "end faces" in i18n.STRINGS["canister_spacing_help"]["en"]
    assert "Stirnflächen" in i18n.STRINGS["canister_spacing_help"]["de"]
    # The derived quantity, shown live.
    assert "length + clear spacing" in i18n.STRINGS["resulting_pitch"]["en"]
    assert "Teilung" in i18n.STRINGS["resulting_pitch"]["de"]

    subtitle = case_subtitle(make_config())
    assert "clear canister spacing" in subtitle and "pitch" in subtitle


def test_a_feature_without_a_control_is_refused_not_defaulted():
    # Interpolated features without a control are reported.
    source = inspect.getsource(app.sidebar_inputs)
    assert "unhandled" in source
    assert "st.stop()" in source
    assert '"interpolates_note"' in source
    assert "default value" in i18n.STRINGS["interpolates_note"]["en"]
    assert "Vorgabewert" in i18n.STRINGS["interpolates_note"]["de"]


def test_material_sliders_follow_the_surrogate_not_a_fixed_list():
    source = inspect.getsource(app.material_inputs)
    assert "model.feature_names" in source
    # Bounded by the trained hull, unlike the layout sliders.
    assert "trained[conductivity]" in source
    assert "trained[capacity]" in source


def test_the_fingerprint_covers_every_design_entry():
    # Material properties invalidate a run.
    physics = {"surface_temperature_c": 10.0, "gradient": 0.03, "interim_storage": 50.0}
    decay = {"p_kw": [1.0], "t_half": [30.0]}
    design = {
        "canister_gap": 9.5,
        "a_drift": 40.0,
        "l_can": 5.52,
        "depth_m": 500.0,
        "backfill_lambda": 0.8,
    }
    base = app.run_fingerprint(design, physics, decay, 1235.0)
    changed = app.run_fingerprint({**design, "backfill_lambda": 0.81}, physics, decay, 1235.0)
    assert base != changed


def test_the_page_imports_absolutely():
    # app.py runs as a script: absolute imports.
    import ast

    tree = ast.parse(inspect.getsource(app))
    relative = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.level]
    assert relative == []


def test_the_direct_run_simulates_the_whole_design(monkeypatch):
    # All design entries reach the simulated configuration.
    captured = {}

    def fake_run_case(cfg, work_dir, progress=None, cancel=None):
        captured["cfg"] = cfg
        raise RuntimeError("stop here")

    from theresa.config import design_value

    monkeypatch.setattr(app, "run_case", fake_run_case)
    monkeypatch.setattr(app.st, "session_state", {})

    design = {
        "canister_gap": 7.0,
        "a_drift": 35.0,
        "l_can": 5.52,
        "depth_m": 600.0,
        "backfill_lambda": 1.3,
        "host_rock_rho_cp": 2.5e6,
    }
    physics = {
        "surface_temperature_c": 9.0,
        "gradient": 0.03,
        "interim_storage": 50.0,
        "limit_celsius": 100.0,
    }
    decay = {"p_kw": [1.0], "t_half": [30.0]}
    app.start_direct_run(design, physics, decay, 1000.0)
    app.st.session_state["direct_thread"].join()

    for name, value in design.items():
        assert design_value(captured["cfg"], name) == pytest.approx(value), name


def test_the_progress_bar_covers_every_stage_once():
    from theresa.pipeline import STAGES

    assert set(app.STAGE_WEIGHTS) == set(STAGES)
    assert sum(app.STAGE_WEIGHTS.values()) == pytest.approx(1.0)
    assert app.run_fraction(STAGES[0], 0, 1) == 0.0
    assert app.run_fraction(STAGES[-1], 1, 1) == pytest.approx(1.0)
    # Monotone through the stages and through the steps of the transient one.
    starts = [app.run_fraction(stage, 0, 1) for stage in STAGES]
    assert starts == sorted(starts)
    steps = [app.run_fraction("transient", step, 295) for step in range(296)]
    assert steps == sorted(steps)
    assert steps[-1] == pytest.approx(app.run_fraction("extract", 0, 1))


def test_the_mesh_steps_fill_their_stage_in_order():
    from theresa.pipeline import MESH_STEPS

    assert set(app.MESH_STEP_WEIGHTS) == set(MESH_STEPS)
    assert sum(app.MESH_STEP_WEIGHTS.values()) == pytest.approx(1.0)
    fractions = [app.run_fraction("mesh", done, len(MESH_STEPS)) for done in range(6)]
    assert fractions == sorted(fractions)
    assert fractions[-1] == pytest.approx(app.run_fraction("steady", 0, 1))


def test_each_look_matches_its_theme():
    # Chart background equals the theme background.
    import tomllib

    from theresa.plotting import PALETTES

    config = tomllib.loads(
        (app.Path(app.__file__).parent / ".streamlit" / "config.toml").read_text()
    )
    for kind, look in app.LOOKS.items():
        assert look.page == config["theme"][kind]["backgroundColor"], kind
        assert look.palette in PALETTES
        assert look.wordmark.exists(), look.wordmark
