# SPDX-License-Identifier: GPL-3.0-or-later
"""Streamlit GUI: interactive prediction and direct OGS comparison.

The uncertainty shown is the interpolation uncertainty of the surrogate.
The direct OGS comparison and the choice of the surrogate file are offered
only if Gmsh, OpenGeoSys and ogstools are installed and
``THERESA_DIRECT_SIMULATION`` is not ``off``; otherwise the app evaluates the
included surrogate only (hosted deployment).

Start with::

    uv run theresa-gui
"""

from __future__ import annotations

import importlib.util
import io
import os
import queue
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Repository checkout without installation (hosted deployment).
if importlib.util.find_spec("theresa") is None:
    sys.path.insert(0, str(PROJECT_ROOT))

from theresa.config import MATERIAL_ALIASES, design_value, make_config  # noqa: E402
from theresa.decay_heat import nominal_power_watt, scaled_to_power  # noqa: E402
from theresa.pipeline import MESH_STEPS, STAGES, run_case  # noqa: E402
from theresa.plotting import plot_canister_power, plot_peak_temperature  # noqa: E402
from theresa.predict import (  # noqa: E402
    DEFAULT_SURROGATE_PATH,
    PeakAtRecordEnd,
    Prediction,
    UncertainInterpolation,
    predict,
)
from theresa.runner import SimulationCancelled  # noqa: E402
from theresa.surrogate import OutsideTrainedRange, Surrogate  # noqa: E402
from theresa_gui.i18n import LANGUAGES, current_language, plot_text, t  # noqa: E402

DIRECT_RUNS = PROJECT_ROOT / "runs" / "gui"
#: Icon (``scripts/render_assets.py``).
ICON = Path(__file__).resolve().parent / "assets" / "icon.png"
#: Word marks for the light and dark theme.
WORDMARK = Path(__file__).resolve().parent / "assets" / "wordmark.png"
WORDMARK_DARK = Path(__file__).resolve().parent / "assets" / "wordmark-dark.png"


@dataclass(frozen=True)
class Look:
    """Theme-dependent colours and assets of elements drawn by the app.

    :param palette: plotting palette, see :data:`theresa.plotting.PALETTES`.
    :param page: chart background, equal to the theme's background colour.
    :param wordmark: sidebar logo.
    :param muted: secondary HTML text.
    :param tick: trained-range ticks on the sliders.
    """

    palette: str
    page: str
    wordmark: Path
    muted: str
    tick: str


#: Look per theme type; ``page`` equals ``backgroundColor`` in
#: ``.streamlit/config.toml``.
LOOKS = {
    "light": Look("light", "#ffffff", WORDMARK, "#5b7088", "#13294a"),
    "dark": Look("dark", "#0c1c2f", WORDMARK_DARK, "#9fb6cf", "#cfdcea"),
}


@st.fragment(run_every=2.0)
def follow_theme() -> None:
    """Rerun the page when the theme changes."""
    if look().palette != st.session_state.get("drawn_theme"):
        st.rerun()


def look() -> Look:
    """Look of the active theme; light by default."""
    kind = getattr(st.context.theme, "type", None)
    return LOOKS.get(kind or "light", LOOKS["light"])


@st.cache_resource(show_spinner=False)
def load_surrogate(path: str) -> Surrogate:
    """Load the surrogate (cached)."""
    return Surrogate.load(Path(path))


def main() -> None:
    """Render the page."""
    # Language first; all texts depend on it.
    st.sidebar.selectbox(
        "Sprache / Language",
        list(LANGUAGES),
        format_func=LANGUAGES.__getitem__,
        key="language",
    )
    st.set_page_config(page_title=t("page_title"), page_icon=str(ICON), layout="wide")
    st.logo(str(look().wordmark), icon_image=str(ICON), size="large")

    # Compact layout that fits one screen; sizes in viewport units.
    st.markdown(
        """
        <style>
          [data-testid="stMainBlockContainer"] {
              padding-top: 3rem;
              padding-bottom: 1rem;
          }
          [data-testid="stMainBlockContainer"] [data-testid="stVerticalBlock"] {
              gap: 0.6rem;
          }
          /* Chart: column width, aspect preserved, at most 44 vh. */
          [data-testid="stMain"] [data-testid="stImage"] img {
              width: 100%;
              height: auto;
              max-height: 44vh;
              display: block;
              object-fit: contain;
          }
          [data-testid="stMetricValue"] { font-size: clamp(1.3rem, 3.2vh, 2rem); }
          [data-testid="stMetricLabel"] { font-size: clamp(0.7rem, 1.5vh, 0.9rem); }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.session_state["drawn_theme"] = look().palette
    follow_theme()

    st.markdown(
        f"#### {t('heading')}"
        f"<span style='color:{look().muted};font-size:0.82em'>"
        f"&nbsp;&nbsp;·&nbsp;&nbsp;{t('scope_note')}"
        "</span>",
        unsafe_allow_html=True,
    )

    full = direct_simulation_available()
    surrogate_path = (
        st.sidebar.text_input(t("surrogate_file"), str(DEFAULT_SURROGATE_PATH))
        if full
        else str(DEFAULT_SURROGATE_PATH)
    )
    if not Path(surrogate_path).exists():
        st.error(t("no_surrogate", path=surrogate_path))
        return

    try:
        model = load_surrogate(surrogate_path)
    except Exception as error:  # noqa: BLE001 - any unreadable file, reported
        st.error(t("bad_surrogate", path=surrogate_path, error=f"{type(error).__name__}: {error}"))
        return
    design, physics, decay = sidebar_inputs(model)

    try:
        prediction, fitted_power = evaluate(model, design, physics, decay)
    except ValueError as error:
        st.error(t("invalid_input", error=error))
        return
    if fitted_power is not None and not report_fit(prediction, physics, fitted_power):
        return

    show_headline(prediction)
    show_charts(
        prediction,
        direct_result(design, physics, decay, prediction.canister_power_watt),
    )

    footer = st.columns([3, 2])
    with footer[0]:
        if full:
            direct_controls(design, physics, decay, prediction.canister_power_watt)
        else:
            st.caption(t("direct_unavailable"))
    with footer[1]:
        show_warnings(prediction, model)
    st.caption(t("disclaimer"))


def evaluate(
    model: Surrogate, design: dict, physics: dict, decay: dict
) -> tuple[Prediction, float | None]:
    """Prediction for the inputs.

    In fit mode the loading is set to
    :meth:`theresa.pipeline.History.max_admissible_power_watt` and the
    prediction repeated at that loading.
    """
    common = {
        "canister_gap_m": design["canister_gap"],
        "a_drift_m": design["a_drift"],
        "canister_length_m": design["l_can"],
        "depth_m": design["depth_m"],
        "materials": {name: value for name, value in design.items() if name in MATERIAL_ALIASES},
        "surface_temperature_c": physics["surface_temperature_c"],
        "geothermal_gradient_k_per_m": physics["gradient"],
        "interim_storage_years": physics["interim_storage"],
        "decay_p_kw": decay["p_kw"],
        "decay_t_half_years": decay["t_half"],
        "limit_celsius": physics["limit_celsius"],
        "surrogate": model,
    }

    if physics["mode"] == "power":
        return predict(canister_power_w=physics["canister_power_w"], **common), None
    if physics["mode"] == "elements":
        return predict(n_fuel_elements=physics["n_fuel_elements"], **common), None

    # Fit mode: first pass without uncertainty.
    probe = predict(n_fuel_elements=1, with_uncertainty=False, **common)
    admissible = probe.max_admissible_power_watt(physics["limit_celsius"])
    if admissible <= 0.0 or not np.isfinite(admissible):
        return probe, admissible
    return predict(canister_power_w=admissible, **common), admissible


def report_fit(prediction, physics: dict, admissible: float) -> bool:
    """Report the fitted loading; ``False`` if none exists."""
    limit = physics["limit_celsius"]

    if admissible <= 0.0:
        st.error(t("no_loading", limit=limit, undisturbed=prediction.undisturbed_celsius))
        return False
    if not np.isfinite(admissible):
        st.error(t("degenerate_fit"))
        return False

    per_element = admissible / max(physics["watt_per_fuel_element"], 1e-12)
    st.caption(
        t(
            "fitted_loading",
            watt=admissible,
            elements=per_element,
            years=physics["interim_storage"],
            limit=limit,
        )
    )
    return True


def watt_per_fuel_element(decay: dict, interim_storage_years: float) -> float:
    """Canister power [W] per fuel element at emplacement."""
    cfg = make_config(
        spectrum="gui",
        decay_heat_p_kw=list(decay["p_kw"]),
        decay_heat_t_half_years=list(decay["t_half"]),
        n_fuel_elements=1,
        interim_storage_years=float(interim_storage_years),
    )
    return nominal_power_watt(cfg)


#: Display scale of heat capacities, J -> MJ.
MEGA = 1.0e6

#: Translation key of each material group.
MATERIAL_LABELS: dict[str, str] = {
    "canister": "material_canister",
    "backfill": "material_backfill",
    "lining": "material_lining",
    "host_rock": "material_host_rock",
}

#: Translation key and display scale of each material property.
PROPERTY_LABELS: dict[str, tuple[str, float]] = {
    "lambda": ("conductivity", 1.0),
    "rho_cp": ("heat_capacity", 1.0 / MEGA),
}

#: Translation key of each interpolated geometric quantity (all in metres).
GEOMETRY_LABELS: dict[str, str] = {
    "canister_gap": "canister_spacing",
    "a_drift": "drift_spacing",
    "l_can": "canister_length",
    "depth_m": "depth",
}


def feature_label(name: str) -> tuple[str, float]:
    """Display label of an interpolated feature and its scale from SI.

    :param name: an entry of :attr:`Surrogate.feature_names`.
    :returns: ``(label, scale)``.
    """
    if name in GEOMETRY_LABELS:
        return t(GEOMETRY_LABELS[name]), 1.0
    if name in MATERIAL_ALIASES:
        material, prop = MATERIAL_ALIASES[name]
        key, scale = PROPERTY_LABELS[prop]
        return f"{t(MATERIAL_LABELS[material.removeprefix('mat_')])}: {t(key)}", scale
    return name, 1.0


def material_inputs(model: Surrogate, trained: dict, defaults: dict) -> dict[str, float]:
    """Sliders for the interpolated material properties, within the trained range."""
    present = [name for name in model.feature_names if name in MATERIAL_ALIASES]
    if not present:
        return {}

    st.sidebar.header(t("materials"))
    st.sidebar.caption(t("materials_note"))

    values: dict[str, float] = {}
    for material, label_key in MATERIAL_LABELS.items():
        conductivity, capacity = f"{material}_lambda", f"{material}_rho_cp"
        if conductivity not in present and capacity not in present:
            continue
        st.sidebar.markdown(f"**{t(label_key)}**")
        if conductivity in present:
            low, high = trained[conductivity]
            values[conductivity] = st.sidebar.slider(
                t("conductivity"),
                low,
                high,
                float(np.clip(design_value(defaults, conductivity), low, high)),
                (high - low) / 100.0,
                key=f"lam_{material}",
            )
        if capacity in present:
            low, high = trained[capacity]
            shown = st.sidebar.slider(
                t("heat_capacity"),
                low / MEGA,
                high / MEGA,
                float(np.clip(design_value(defaults, capacity), low, high)) / MEGA,
                (high - low) / MEGA / 100.0,
                key=f"rhocp_{material}",
            )
            values[capacity] = shown * MEGA
    return values


def mark_trained_range(
    key: str, slider_range: tuple[float, float], trained: tuple[float, float]
) -> None:
    """Mark the trained range on a slider rail with two ticks.

    :param key: the slider's widget key.
    :param slider_range: the slider's ``(min, max)``.
    :param trained: the trained ``(low, high)``.
    """
    start, stop = slider_range
    low, high = ((value - start) / (stop - start) * 100.0 for value in trained)
    rail = f".st-key-{key} [role='group'] > div > div:first-child"
    st.html(
        f"""<style>
        {rail} {{ position: relative; }}
        {rail}::before, {rail}::after {{
            content: ""; position: absolute; top: -5px; width: 2px; height: 14px;
            background: {look().tick}; border-radius: 1px; pointer-events: none;
        }}
        {rail}::before {{ left: calc({low:.3f}% - 1px); }}
        {rail}::after {{ left: calc({high:.3f}% - 1px); }}
        </style>"""
    )


def sidebar_inputs(model: Surrogate) -> tuple[dict, dict, dict]:
    """Sidebar inputs."""
    trained = model.trained_ranges()

    st.sidebar.header(t("layout"))
    st.sidebar.caption(t("layout_note"))
    defaults = make_config()

    low, high = trained.get("canister_gap", (2.5, 15.0))
    canister_gap = st.sidebar.slider(
        t("canister_spacing"),
        1.0,
        25.0,
        float(np.clip(defaults["canister_gap"], low, high)),
        0.25,
        help=t("canister_spacing_help") + " " + t("trained_ticks", low=low, high=high),
        key="canister_gap",
    )
    mark_trained_range("canister_gap", (1.0, 25.0), (low, high))
    low, high = trained.get("a_drift", (15.0, 60.0))
    a_drift = st.sidebar.slider(
        t("drift_spacing"),
        10.0,
        100.0,
        float(np.clip(defaults["a_drift"], low, high)),
        1.0,
        help=t("trained_ticks", low=low, high=high),
        key="a_drift",
    )
    mark_trained_range("a_drift", (10.0, 100.0), (low, high))
    if "l_can" in trained:
        low, high = trained["l_can"]
        canister_length = st.sidebar.slider(
            t("canister_length"), 2.0, 9.0, float(np.clip(defaults["l_can"], low, high)), 0.1
        )
    else:
        # Fixed length of the surrogate.
        canister_length = float((model.fixed_design or {}).get("l_can", defaults["l_can"]))
        st.sidebar.caption(t("canister_length_fixed", length=canister_length))
    st.sidebar.caption(t("resulting_pitch", pitch=canister_length + canister_gap))

    materials = material_inputs(model, trained, defaults)

    st.sidebar.header(t("site"))
    st.sidebar.caption(t("site_note"))
    depth = st.sidebar.slider(t("depth"), 300.0, 1200.0, 500.0, 10.0)
    surface_temperature = st.sidebar.number_input(t("surface_temperature"), -10.0, 40.0, 10.0, 0.5)
    gradient = st.sidebar.number_input(t("gradient"), 0.005, 0.080, 0.030, 0.001, format="%.3f")

    st.sidebar.header(t("design_limit"))
    limit_celsius = st.sidebar.number_input(
        t("limit_temperature"),
        20.0,
        400.0,
        float(make_config()["T_limit_celsius"]),
        5.0,
        help=t("limit_help"),
    )

    st.sidebar.header(t("loading"))
    st.sidebar.caption(t("loading_note"))
    # Keyed by index, independent of the language.
    mode_keys = ("elements", "power", "fit")
    mode = st.sidebar.radio(
        t("loading_mode"),
        mode_keys,
        format_func=lambda key: t(f"loading_mode_{key}"),
        help=t("loading_mode_help"),
    )
    n_fuel_elements = 2
    canister_power_w = 1235.0
    if mode == "elements":
        n_fuel_elements = st.sidebar.number_input(t("fuel_elements"), 1, 30, 2, 1)
    elif mode == "power":
        canister_power_w = st.sidebar.number_input(t("canister_power"), 1.0, 50000.0, 1235.0, 25.0)

    interim_storage = st.sidebar.slider(t("interim_storage"), 10.0, 300.0, 50.0, 5.0)

    st.sidebar.header(t("decay_heat"))
    st.sidebar.caption(t("decay_heat_note"))
    p_default = list(defaults["decay_heat_p_kw"]) + [0.0] * 6
    t_default = list(defaults["decay_heat_t_half_years"]) + [1.0] * 6

    p_kw: list[float] = []
    t_half: list[float] = []
    for i in range(6):
        columns = st.sidebar.columns(2)
        p_kw.append(
            columns[0].number_input(
                f"p{i + 1} [kW]",
                0.0,
                100.0,
                float(p_default[i]),
                0.001,
                format="%.4f",
                key=f"p{i}",
            )
        )
        t_half.append(
            columns[1].number_input(
                f"t½,{i + 1} [a]",
                0.01,
                1.0e7,
                float(t_default[i]),
                1.0,
                format="%.2f",
                key=f"t{i}",
            )
        )

    decay = {"p_kw": p_kw, "t_half": t_half}

    design = {
        "canister_gap": canister_gap,
        "a_drift": a_drift,
        "l_can": canister_length,
        "depth_m": depth,
        **materials,
    }
    # Interpolated features without a control are reported.
    unhandled = [name for name in model.feature_names if name not in design]
    if unhandled:
        st.error(t("interpolates_note", names=", ".join(f"`{name}`" for name in unhandled)))
        st.stop()

    return (
        design,
        {
            "surface_temperature_c": surface_temperature,
            "gradient": gradient,
            "mode": mode,
            "n_fuel_elements": int(n_fuel_elements),
            "canister_power_w": float(canister_power_w),
            "interim_storage": interim_storage,
            "limit_celsius": float(limit_celsius),
            "watt_per_fuel_element": watt_per_fuel_element(decay, interim_storage),
        },
        decay,
    )


def show_headline(prediction) -> None:
    """Headline metrics."""
    limit = prediction.config["T_limit_celsius"]
    sigma = prediction.peak_sigma_kelvin

    # Rounded as displayed; + 0.0 turns a -0.0 into 0.0.
    margin = round(prediction.peak_celsius - limit, 1) + 0.0

    columns = st.columns(4)
    columns[0].metric(
        t("metric_peak"),
        f"{prediction.peak_celsius:.1f} °C",
        delta=t("vs_limit", delta=margin),
        # Neutral when the peak equals the limit.
        delta_color="inverse" if margin else "off",
        delta_arrow="auto" if margin else "off",
    )
    columns[1].metric(t("metric_time"), f"{prediction.peak_time_years:.0f} a")
    columns[2].metric(
        t("metric_uncertainty"),
        "—" if sigma is None else f"± {sigma:.2f} K",
        help=t("uncertainty_help"),
    )
    admissible = prediction.max_admissible_power_watt()
    columns[3].metric(
        t("metric_admissible"),
        "—"
        if admissible == 0.0
        else (t("unlimited") if not np.isfinite(admissible) else f"{admissible:.0f} W"),
        delta=t("now_loading", watt=prediction.canister_power_watt),
        delta_color="off",
        help=t("admissible_help", limit=limit),
    )


#: Figure size [in].
CHART_SIZE = (6.5, 3.2)


def svg(draw, *args, **kwargs) -> str:
    """SVG markup of a plotting function's figure."""
    target = io.BytesIO()
    draw(
        *args,
        target,
        **kwargs,
        title="",
        figsize=CHART_SIZE,
        background=look().page,
        palette=look().palette,
        text=plot_text(),
        stream_format="svg",
    )
    return target.getvalue().decode()


def show_figure(markup: str) -> None:
    """Show SVG markup at column width."""
    st.image(markup, width="stretch")


@st.cache_data(show_spinner=False, max_entries=32)
def power_figure(times_years: np.ndarray, power_watt: np.ndarray, language: str, theme: str) -> str:
    """Canister-power figure (cached; ``language`` and ``theme`` are cache keys)."""
    _ = language, theme
    return svg(plot_canister_power, times_years, power_watt)


def show_charts(prediction, direct) -> None:
    """Tabs with temperature history and canister power."""
    temperature_tab, power_tab = st.tabs([t("tab_temperature"), t("tab_power")])

    with temperature_tab:
        show_figure(
            svg(
                plot_peak_temperature,
                prediction.times_years,
                prediction.temperature_celsius,
                limit_celsius=prediction.config["T_limit_celsius"],
                undisturbed_celsius=prediction.undisturbed_celsius,
                sigma_kelvin=prediction.temperature_sigma_kelvin,
                reference=direct,
                subtitle=None,
                labels=(t("plot_surrogate"), t("plot_direct")),
            )
        )
        if direct is not None:
            surrogate_peak = prediction.peak_celsius
            direct_peak = float(np.max(direct[1]))
            st.caption(
                t(
                    "direct_comparison",
                    direct=direct_peak,
                    surrogate=surrogate_peak,
                    delta=surrogate_peak - direct_peak,
                )
            )

    with power_tab:
        show_figure(
            power_figure(
                prediction.times_years,
                prediction.power_watt,
                current_language(),
                look().palette,
            )
        )


def warning_text(warning) -> str:
    """Entry of :attr:`Prediction.warnings` in the session's language."""
    match warning:
        case OutsideTrainedRange(feature=name, value=value, low=low, high=high):
            label, scale = feature_label(name)
            return t(
                "outside_range",
                label=label,
                value=value * scale,
                low=low * scale,
                high=high * scale,
            )
        case UncertainInterpolation(sigma_kelvin=sigma, threshold_kelvin=threshold):
            return t("uncertain_here", sigma=sigma, threshold=threshold)
        case PeakAtRecordEnd(record_years=years):
            return t("peak_at_end", years=years)
    return str(warning)


def show_warnings(prediction, model: Surrogate) -> None:
    """Warnings and trained ranges."""
    if prediction.warnings:
        st.warning(
            t("outside_trained")
            + "\n\n"
            + "\n".join(f"- {warning_text(warning)}" for warning in prediction.warnings),
            icon="⚠️",
        )
    with st.expander(t("what_was_trained")):
        bullets = []
        for name, (low, high) in model.trained_ranges().items():
            label, scale = feature_label(name)
            bullets.append(f"- {label}: {low * scale:.3g} – {high * scale:.3g}")
        st.markdown(
            t(
                "trained_summary",
                designs=model.train_points.shape[0],
                modes=len(model.kernel_models),
            )
            + "\n\n"
            + "\n".join(bullets)
        )


def run_fingerprint(design: dict, physics: dict, decay: dict, canister_power_w: float) -> tuple:
    """Inputs of a direct simulation, for matching a run to the current design.

    Excludes the design limit and the way the loading was specified.
    """
    return (
        tuple(sorted((name, round(float(value), 9)) for name, value in design.items())),
        round(physics["surface_temperature_c"], 9),
        round(physics["gradient"], 12),
        round(physics["interim_storage"], 9),
        round(canister_power_w, 6),
        tuple(round(value, 12) for value in decay["p_kw"]),
        tuple(round(value, 9) for value in decay["t_half"]),
    )


def direct_result(
    design: dict, physics: dict, decay: dict, canister_power_w: float
) -> tuple[np.ndarray, np.ndarray] | None:
    """Finished direct run, if it matches the current inputs."""
    stored = st.session_state.get("direct_result")
    if stored is None:
        return None
    if st.session_state.get("direct_fingerprint") != run_fingerprint(
        design, physics, decay, canister_power_w
    ):
        st.session_state.pop("direct_result", None)
        st.session_state.pop("direct_fingerprint", None)
        return None
    return stored


def direct_simulation_available() -> bool:
    """Gmsh, OpenGeoSys and ogstools installed and not switched off."""
    if os.environ.get("THERESA_DIRECT_SIMULATION", "").lower() == "off":
        return False
    return all(importlib.util.find_spec(name) for name in ("gmsh", "ogs", "ogstools"))


def direct_controls(design: dict, physics: dict, decay: dict, canister_power_w: float) -> None:
    """Controls and status of the direct OGS simulation.

    ``canister_power_w`` is the loading of the prediction (fitted in fit mode).
    """
    running = st.session_state.get("direct_thread") is not None
    # Buttons in one row, status below.
    columns = st.columns(2)

    if columns[0].button(
        t("check_with_ogs"),
        disabled=running,
        type="primary",
        width="stretch",
        help=t("check_help"),
    ):
        start_direct_run(design, physics, decay, canister_power_w)
        st.rerun()

    # Stop while running, otherwise remove the comparison.
    if running:
        stopping = st.session_state["direct_cancel"].is_set()
        if columns[1].button(t("stop"), disabled=stopping, width="stretch", help=t("stop_help")):
            st.session_state["direct_cancel"].set()
            st.rerun()
    else:
        shown = any(key in st.session_state for key in ("direct_result", "direct_error"))
        if columns[1].button(t("clear"), disabled=not shown, width="stretch", help=t("clear_help")):
            for key in ("direct_result", "direct_fingerprint", "direct_error"):
                st.session_state.pop(key, None)
            st.rerun()

    with st.container():
        if running:
            direct_progress()
        elif st.session_state.get("direct_error"):
            st.error(t("simulation_failed", error=st.session_state["direct_error"]), icon="⚠️")
        elif st.session_state.pop("direct_cancelled", False):
            st.caption(t("cancelled"))


def start_direct_run(design: dict, physics: dict, decay: dict, canister_power_w: float) -> None:
    """Start the direct simulation in a worker thread."""
    # Loading passed as canister power.
    unit = make_config(
        spectrum="custom",
        decay_heat_p_kw=list(decay["p_kw"]),
        decay_heat_t_half_years=list(decay["t_half"]),
        n_fuel_elements=1,
        interim_storage_years=physics["interim_storage"],
    )
    # Complete design including material properties.
    cfg = make_config(
        base=scaled_to_power(unit, canister_power_w),
        **design,
        surface_temperature_celsius=physics["surface_temperature_c"],
        geothermal_gradient_k_per_m=physics["gradient"],
        T_limit_celsius=physics["limit_celsius"],
    )
    channel: queue.Queue = queue.Queue()
    cancel = threading.Event()
    # One run directory per session.
    session = st.session_state.setdefault("session_id", uuid.uuid4().hex[:12])
    run_directory = DIRECT_RUNS / session

    def work() -> None:
        try:
            result = run_case(
                cfg,
                run_directory,
                progress=lambda stage, done, total: channel.put(("progress", (stage, done, total))),
                cancel=cancel,
            )
            channel.put(("ok", (result.times_years, result.temperature_celsius)))
        except SimulationCancelled:
            channel.put(("cancelled", None))
        except Exception as error:  # noqa: BLE001 - reported to the user
            channel.put(("error", f"{type(error).__name__}: {error}"))

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    st.session_state["direct_thread"] = thread
    st.session_state["direct_queue"] = channel
    st.session_state["direct_cancel"] = cancel
    st.session_state["direct_started"] = time.time()
    st.session_state["direct_pending_fingerprint"] = run_fingerprint(
        design, physics, decay, canister_power_w
    )
    st.session_state.pop("direct_error", None)
    st.session_state.pop("direct_result", None)
    st.session_state.pop("direct_fingerprint", None)


#: Share of each stage in the wall time of a direct run.
STAGE_WEIGHTS = {"mesh": 0.16, "steady": 0.03, "transient": 0.78, "extract": 0.03}


#: Share of each meshing step in the meshing stage.
MESH_STEP_WEIGHTS = {
    "geometry": 0.01,
    "curves": 0.43,
    "surfaces": 0.07,
    "volume": 0.27,
    "conversion": 0.22,
}


def run_fraction(stage: str, done: int, total: int) -> float:
    """Progress of a direct run, 0 to 1."""
    finished = sum(STAGE_WEIGHTS[name] for name in STAGES[: STAGES.index(stage)])
    if stage == "mesh":
        within = sum(MESH_STEP_WEIGHTS[step] for step in MESH_STEPS[:done])
    else:
        within = done / max(total, 1)
    return min(1.0, finished + STAGE_WEIGHTS[stage] * within)


def duration_text(seconds: float) -> str:
    """Duration as ``95 s`` or ``3 min``."""
    return f"{seconds:.0f} s" if seconds < 90.0 else f"{seconds / 60.0:.0f} min"


@st.fragment(run_every=1.0)
def direct_progress() -> None:
    """Progress of the running simulation; reruns the page when it ends."""
    channel: queue.Queue = st.session_state["direct_queue"]
    finished = None
    while True:
        try:
            kind, payload = channel.get_nowait()
        except queue.Empty:
            break
        if kind == "progress":
            st.session_state["direct_progress"] = payload
        else:
            finished = (kind, payload)

    if finished is not None:
        kind, payload = finished
        st.session_state["direct_thread"] = None
        st.session_state.pop("direct_progress", None)
        if kind == "cancelled":
            st.session_state["direct_cancelled"] = True
        elif kind == "ok":
            st.session_state["direct_result"] = payload
            # Inputs at the start of the run.
            st.session_state["direct_fingerprint"] = st.session_state.pop(
                "direct_pending_fingerprint", None
            )
        else:
            st.session_state["direct_error"] = payload
        st.rerun()

    stage, done, total = st.session_state.get("direct_progress", (STAGES[0], 0, 1))
    fraction = run_fraction(stage, done, total)
    elapsed = time.time() - st.session_state["direct_started"]
    if st.session_state["direct_cancel"].is_set():
        st.progress(fraction, text=t("stopping"))
        return
    if stage == "mesh":
        step = MESH_STEPS[min(done, len(MESH_STEPS) - 1)]
        label = t("stage_mesh", step=t(f"mesh_{step}"), done=done + 1, total=total)
    else:
        label = t(f"stage_{stage}", done=done, total=total)
    parts = [label, duration_text(elapsed)]
    # Remaining time after the first percent.
    if fraction >= 0.05:
        parts.append(t("remaining", time=duration_text(elapsed * (1.0 - fraction) / fraction)))
    st.progress(fraction, text=" · ".join(parts))


if __name__ == "__main__":
    main()
