# SPDX-License-Identifier: GPL-3.0-or-later
"""German and English texts of the GUI.

The language is stored per session in ``st.session_state``. Every key has
both languages. Numbers and units are not translated.
"""

from __future__ import annotations

import streamlit as st

#: Selectable languages by code.
LANGUAGES: dict[str, str] = {"de": "Deutsch", "en": "English"}

DEFAULT_LANGUAGE = "de"

#: User-facing texts by key.
STRINGS: dict[str, dict[str, str]] = {
    # --- page furniture ---------------------------------------------------
    "page_title": {
        "de": "THERESA · Behältertemperatur",
        "en": "THERESA · Canister temperature",
    },
    "heading": {
        "de": "Maximale Temperatur an der Behälteroberfläche",
        "en": "Peak temperature at the canister surface",
    },
    "scope_note": {
        "de": "Hobbyprojekt · keine realistischen Parameter",
        "en": "Hobby project · parameters not realistic",
    },
    "disclaimer": {
        "de": "Privates Hobbyprojekt zum Ausprobieren von Simulations- und "
        "Surrogatmethoden. Geometrie, Stoffwerte, Zerfallswärme und Auslegungsgrenze "
        "sind vereinfachte oder angenommene Werte, nicht realistisch und ohne Bezug zu "
        "einem bestehenden oder geplanten Endlager. Die Ergebnisse eignen sich nicht für "
        "Auslegungs-, Genehmigungs- oder Sicherheitsaussagen. Ohne Gewähr.",
        "en": "Private hobby project for trying out simulation and surrogate-modelling "
        "techniques. Geometry, material parameters, decay heat and design limit are "
        "simplified or assumed values, not realistic and unrelated to any existing or "
        "planned repository. The results are not suitable for design, licensing or "
        "safety assessments. No warranty.",
    },
    "surrogate_file": {"de": "Surrogat-Datei", "en": "Surrogate file"},
    "bad_surrogate": {
        "de": "`{path}` lässt sich nicht als Surrogat laden: {error}",
        "en": "`{path}` cannot be loaded as a surrogate: {error}",
    },
    "invalid_input": {
        "de": "Für diese Eingaben ist keine Vorhersage möglich: {error}",
        "en": "No prediction is possible for these inputs: {error}",
    },
    "no_surrogate": {
        "de": "Kein trainiertes Surrogat unter `{path}`.\n\n"
        "Erzeugung mit `theresa train` (standardmäßig 400 OGS-Rechnungen).",
        "en": "No trained surrogate at `{path}`.\n\n"
        "Create one with `theresa train` (400 OGS runs by default).",
    },
    # --- layout group -----------------------------------------------------
    "layout": {"de": "Geometrie", "en": "Layout"},
    "layout_note": {
        "de": "Interpoliert. Außerhalb des trainierten Bereichs wird extrapoliert und gewarnt.",
        "en": "Interpolated. Outside the trained range the prediction extrapolates and is flagged.",
    },
    "canister_spacing": {
        "de": "Lichter Behälterabstand [m]",
        "en": "Clear canister spacing [m]",
    },
    "canister_spacing_help": {
        "de": "Abstand der Stirnflächen benachbarter Behälter.",
        "en": "Distance between the end faces of adjacent canisters.",
    },
    "trained_ticks": {
        "de": "Striche: trainierter Bereich {low:.1f}–{high:.1f} m.",
        "en": "Ticks: trained range {low:.1f}–{high:.1f} m.",
    },
    "drift_spacing": {
        "de": "Streckenabstand, Achse zu Achse [m]",
        "en": "Drift spacing, axis to axis [m]",
    },
    "canister_length": {"de": "Behälterlänge [m]", "en": "Canister length [m]"},
    "canister_length_fixed": {
        "de": "Behälterlänge: **{length:.2f} m** (fest für dieses Surrogat).",
        "en": "Canister length: **{length:.2f} m** (fixed for this surrogate).",
    },
    "resulting_pitch": {
        "de": "Teilung: **{pitch:.2f} m** (Länge + lichter Abstand).",
        "en": "Pitch: **{pitch:.2f} m** (length + clear spacing).",
    },
    # --- material group ---------------------------------------------------
    "materials": {"de": "Stoffwerte", "en": "Material properties"},
    "materials_note": {
        "de": "Interpoliert. Wärmekapazität volumetrisch (ρ·c).",
        "en": "Interpolated. Heat capacity is volumetric (ρ·c).",
    },
    "material_canister": {"de": "Behälter (Guss)", "en": "Canister (cast iron)"},
    "material_backfill": {"de": "Versatz (Bentonit)", "en": "Backfill (bentonite)"},
    "material_lining": {"de": "Ausbau (Beton)", "en": "Liner (concrete)"},
    "material_host_rock": {"de": "Wirtsgestein (Tonstein)", "en": "Host rock (claystone)"},
    "conductivity": {"de": "Wärmeleitfähigkeit [W/(m·K)]", "en": "Conductivity [W/(m·K)]"},
    "heat_capacity": {"de": "Wärmekapazität [MJ/(m³·K)]", "en": "Heat capacity [MJ/(m³·K)]"},
    # --- site group -------------------------------------------------------
    "site": {"de": "Standort", "en": "Site"},
    "site_note": {
        "de": "Analytisch, ohne Interpolationsfehler.",
        "en": "Analytic, without interpolation error.",
    },
    "depth": {"de": "Einlagerungsteufe [m]", "en": "Emplacement depth [m]"},
    "surface_temperature": {
        "de": "Temperatur an der Geländeoberkante [°C]",
        "en": "Temperature at ground level [°C]",
    },
    "gradient": {"de": "Geothermischer Gradient [K/m]", "en": "Geothermal gradient [K/m]"},
    # --- limit and loading ------------------------------------------------
    "design_limit": {"de": "Auslegungsgrenze", "en": "Design limit"},
    "limit_temperature": {
        "de": "Höchstzulässige Oberflächentemperatur [°C]",
        "en": "Maximum allowed surface temperature [°C]",
    },
    "limit_help": {
        "de": "Bewertungsgrenze; geht nicht in die Simulation ein.",
        "en": "Evaluation threshold; does not enter the simulation.",
    },
    "loading": {"de": "Beladung", "en": "Loading"},
    "loading_note": {
        "de": "Analytisch; die Erwärmung ist proportional zur Behälterleistung.",
        "en": "Analytic; the heating is proportional to the canister power.",
    },
    "loading_mode": {"de": "Beladung vorgeben über", "en": "Set the loading by"},
    "loading_mode_elements": {"de": "Brennelemente", "en": "Fuel elements"},
    "loading_mode_power": {"de": "Wärmeleistung", "en": "Thermal power"},
    "loading_mode_fit": {"de": "An die Grenze anpassen", "en": "Fit to the design limit"},
    "loading_mode_help": {
        "de": "„An die Grenze anpassen“: größte Beladung, deren Maximum die "
        "Auslegungsgrenze erreicht.",
        "en": "“Fit to the design limit”: largest loading whose peak reaches the design limit.",
    },
    "fuel_elements": {"de": "Brennelemente je Behälter", "en": "Fuel elements per canister"},
    "canister_power": {"de": "Wärmeleistung je Behälter [W]", "en": "Canister power [W]"},
    "interim_storage": {
        "de": "Zwischenlagerung vor der Einlagerung [a]",
        "en": "Interim storage before emplacement [a]",
    },
    "fitted_loading": {
        "de": "Angepasste Beladung: **{watt:.0f} W** ≈ **{elements:.1f} "
        "Brennelemente** bei {years:.0f} a Zwischenlagerung; Maximum = {limit:g} °C.",
        "en": "Fitted loading: **{watt:.0f} W** ≈ **{elements:.1f} fuel "
        "elements** at {years:.0f} a interim storage; peak = {limit:g} °C.",
    },
    "no_loading": {
        "de": "Keine zulässige Beladung: Das ungestörte Gebirge hat bereits "
        "{undisturbed:.1f} °C (Grenze {limit:g} °C).",
        "en": "No admissible loading: the undisturbed rock is already at "
        "{undisturbed:.1f} °C (limit {limit:g} °C).",
    },
    "degenerate_fit": {
        "de": "Keine endliche Beladungsgrenze bestimmbar.",
        "en": "No finite loading limit can be determined.",
    },
    # --- decay heat -------------------------------------------------------
    "decay_heat": {"de": "Zerfallswärme", "en": "Decay heat"},
    "decay_heat_note": {
        "de": "Bis zu sechs Exponentialterme; Amplitude 0 entfernt einen Term.",
        "en": "Up to six exponential terms; amplitude 0 removes a term.",
    },
    # --- headline metrics -------------------------------------------------
    "metric_peak": {"de": "Maximaltemperatur", "en": "Peak temperature"},
    "metric_time": {"de": "Zeitpunkt des Maximums", "en": "Time of peak"},
    "metric_uncertainty": {"de": "Unsicherheit", "en": "Uncertainty"},
    "metric_admissible": {"de": "Zulässige Beladung", "en": "Admissible loading"},
    "vs_limit": {"de": "{delta:+.1f} K zur Grenze", "en": "{delta:+.1f} K vs. limit"},
    "now_loading": {"de": "aktuell {watt:.0f} W", "en": "now {watt:.0f} W"},
    "unlimited": {"de": "unbegrenzt", "en": "unlimited"},
    "uncertainty_help": {
        "de": "Standardabweichung der Gauß-Prozess-Interpolation. Ohne "
        "Unsicherheit von Stoffwerten, Modellansatz und Diskretisierung.",
        "en": "Standard deviation of the Gaussian-process interpolation. Excludes "
        "the uncertainty of material properties, model form and discretisation.",
    },
    "admissible_help": {
        "de": "Größte Behälterleistung, bei der die Oberfläche jederzeit unter "
        "{limit:g} °C bleibt.",
        "en": "Largest canister power that keeps the surface below {limit:g} °C at all times.",
    },
    # --- warnings and the trained region -----------------------------------
    "outside_trained": {
        "de": "Außerhalb des trainierten Bereichs:",
        "en": "Outside the trained region:",
    },
    "outside_range": {
        "de": "{label} = {value:.3g} liegt außerhalb des trainierten Bereichs "
        "[{low:.3g}, {high:.3g}]",
        "en": "{label} = {value:.3g} lies outside the trained range [{low:.3g}, {high:.3g}]",
    },
    "peak_at_end": {
        "de": "Maximum am Ende des Zeitraums ({years:.0f} a); das tatsächliche "
        "Maximum kann später liegen",
        "en": "Maximum at the end of the {years:.0f} a record; the actual peak may lie later",
    },
    "uncertain_here": {
        "de": "Interpolationsunsicherheit ± {sigma:.2f} K auf die Maximaltemperatur "
        "(Schwelle {threshold:g} K)",
        "en": "Interpolation uncertainty ± {sigma:.2f} K on the peak temperature "
        "(threshold {threshold:g} K)",
    },
    "what_was_trained": {
        "de": "Trainingsbereich",
        "en": "Training range",
    },
    "trained_summary": {
        "de": "{designs} simulierte Designs, {modes} Moden. Teufe, "
        "Oberflächentemperatur, Gradient, Beladung, Zwischenlagerung und "
        "Zerfallskurve gehen analytisch ein.\n\n"
        "Trainierte Bereiche:",
        "en": "{designs} simulated designs, {modes} modes. Depth, surface "
        "temperature, gradient, loading, interim storage and decay curve enter "
        "analytically.\n\n"
        "Trained ranges:",
    },
    "interpolates_note": {
        "de": "Das Surrogat interpoliert {names} ohne Eingabefeld; es gilt der Vorgabewert.",
        "en": "The surrogate interpolates {names} without an input field; the "
        "default value applies.",
    },
    # --- direct simulation --------------------------------------------------
    "check_with_ogs": {"de": "Mit OGS nachrechnen", "en": "Check with OGS"},
    "clear": {"de": "Vergleich entfernen", "en": "Remove comparison"},
    "clear_help": {
        "de": "Entfernt die OGS-Kurve bzw. deren Meldung.",
        "en": "Removes the OGS curve or its message.",
    },
    "stop": {"de": "Abbrechen", "en": "Stop"},
    "stop_help": {
        "de": "Beendet die laufende Nachrechnung.",
        "en": "Stops the running check.",
    },
    "stopping": {"de": "Wird abgebrochen …", "en": "Stopping …"},
    "cancelled": {"de": "OGS-Nachrechnung abgebrochen.", "en": "OGS check stopped."},
    "direct_unavailable": {
        "de": "Direkte OGS-Simulation in dieser Installation nicht verfügbar.",
        "en": "Direct OGS simulation is not available in this installation.",
    },
    "check_help": {
        "de": "Direkte OGS-Simulation mit diesen Parametern (ca. 2 min).",
        "en": "Direct OGS simulation with these parameters (about 2 min).",
    },
    "stage_mesh": {
        "de": "Vernetzung: {step} ({done}/{total})",
        "en": "Meshing: {step} ({done}/{total})",
    },
    "mesh_geometry": {"de": "Geometrie", "en": "geometry"},
    "mesh_curves": {"de": "Kantennetz", "en": "curves"},
    "mesh_surfaces": {"de": "Flächennetz", "en": "surfaces"},
    "mesh_volume": {"de": "Volumennetz", "en": "volume"},
    "mesh_conversion": {"de": "Umwandlung für OGS", "en": "conversion for OGS"},
    "stage_steady": {
        "de": "Stationäre Rechnung (ψ)",
        "en": "Steady-state solve (ψ)",
    },
    "stage_transient": {
        "de": "Transiente Rechnung, Zeitschritt {done}/{total}",
        "en": "Transient solve, time step {done}/{total}",
    },
    "stage_extract": {
        "de": "Auswertung der Sprungantwort",
        "en": "Extracting the step response",
    },
    "remaining": {"de": "noch ca. {time}", "en": "about {time} left"},
    "simulation_failed": {
        "de": "Simulation fehlgeschlagen: {error}",
        "en": "Simulation failed: {error}",
    },
    "direct_comparison": {
        "de": "Direkte Simulation **{direct:.2f} °C**, Surrogat "
        "**{surrogate:.2f} °C**, Abweichung **{delta:+.2f} K**.",
        "en": "Direct simulation **{direct:.2f} °C**, surrogate "
        "**{surrogate:.2f} °C**, difference **{delta:+.2f} K**.",
    },
    # --- plot ---------------------------------------------------------------
    "tab_temperature": {"de": "Temperatur", "en": "Temperature"},
    "tab_power": {"de": "Wärmeleistung", "en": "Thermal power"},
    "plot_power_label": {
        "de": "Wärmeleistung je Behälter [W]",
        "en": "canister power [W]",
    },
    "plot_emplacement": {"de": "bei Einlagerung", "en": "at emplacement"},
    "plot_x_label": {
        "de": "Zeit seit der Einlagerung [a]",
        "en": "time since emplacement [a]",
    },
    "plot_y_label": {"de": "Temperatur [°C]", "en": "temperature [°C]"},
    "plot_limit": {"de": "Auslegungsgrenze", "en": "design limit"},
    "plot_undisturbed": {"de": "ungestörtes Gebirge", "en": "undisturbed rock"},
    "plot_at": {"de": "bei", "en": "at"},
    "plot_surrogate": {"de": "Surrogat", "en": "surrogate"},
    "plot_direct": {"de": "direkte Simulation", "en": "direct simulation"},
}


def current_language() -> str:
    """Language of the session."""
    return st.session_state.get("language", DEFAULT_LANGUAGE)


def t(key: str, **values: object) -> str:
    """Text ``key`` in the current language, formatted with ``values``.

    :raises KeyError: if the key or language is missing.
    """
    try:
        template = STRINGS[key][current_language()]
    except KeyError as error:
        raise KeyError(f"no {current_language()!r} wording for {key!r}") from error
    return template.format(**values) if values else template


def plot_text() -> dict[str, str]:
    """Plot texts in the current language (keys ``plot_<role>``)."""
    from theresa.plotting import DEFAULT_TEXT

    return {role: t(f"plot_{role}") for role in DEFAULT_TEXT}
