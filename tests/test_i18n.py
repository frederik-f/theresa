# SPDX-License-Identifier: GPL-3.0-or-later
"""GUI texts: complete in both languages, same placeholders, all keys present."""

from __future__ import annotations

import ast
import inspect
import string

import pytest

pytest.importorskip("streamlit")

from theresa_gui import app, i18n  # noqa: E402


def placeholders(template: str) -> set[str]:
    """The field names a format string expects."""
    return {field for _, field, _, _ in string.Formatter().parse(template) if field}


def keys_used_by_the_page() -> set[str]:
    """Literal keys passed to :func:`i18n.t` in the page module."""
    tree = ast.parse(inspect.getsource(app))
    return {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "t"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    }


def test_german_is_the_default():
    assert i18n.DEFAULT_LANGUAGE == "de"
    assert next(iter(i18n.LANGUAGES)) == "de"  # first option in the selector


def test_every_key_carries_every_language():
    for key, wording in i18n.STRINGS.items():
        assert set(wording) == set(i18n.LANGUAGES), key
        assert all(text.strip() for text in wording.values()), key


def test_translations_expect_the_same_values():
    # Same placeholders in both languages.
    for key, wording in i18n.STRINGS.items():
        fields = {language: placeholders(text) for language, text in wording.items()}
        assert len({frozenset(value) for value in fields.values()}) == 1, (key, fields)


def test_every_key_the_page_uses_exists():
    used = keys_used_by_the_page()
    assert used, "the page no longer calls t() -- is the check still looking at it?"
    assert used <= set(i18n.STRINGS), used - set(i18n.STRINGS)

    # Built at run time from the radio's option keys.
    for mode in ("elements", "power", "fit"):
        assert f"loading_mode_{mode}" in i18n.STRINGS
    for label_key in app.MATERIAL_LABELS.values():
        assert label_key in i18n.STRINGS
    for label_key, _ in app.PROPERTY_LABELS.values():
        assert label_key in i18n.STRINGS
    for label_key in app.GEOMETRY_LABELS.values():
        assert label_key in i18n.STRINGS


def test_the_plot_is_labelled_in_both_languages():
    from theresa.plotting import DEFAULT_TEXT

    for role in DEFAULT_TEXT:
        assert f"plot_{role}" in i18n.STRINGS, role
    assert set(i18n.plot_text()) == set(DEFAULT_TEXT)


def test_a_missing_key_is_loud():
    with pytest.raises(KeyError, match="no_such_key"):
        i18n.t("no_such_key")


@pytest.fixture(scope="module")
def model():
    """The shipped surrogate, loaded once for the tests that need one."""
    from theresa.predict import DEFAULT_SURROGATE_PATH
    from theresa.surrogate import Surrogate

    if not DEFAULT_SURROGATE_PATH.exists():
        pytest.skip("no trained surrogate")
    return Surrogate.load(DEFAULT_SURROGATE_PATH)


def test_every_warning_the_library_raises_is_translated(model):
    """Warnings outside the trained range and with high uncertainty, translated."""
    from theresa.predict import PeakAtRecordEnd, UncertainInterpolation, predict
    from theresa.surrogate import OutsideTrainedRange

    prediction = predict(canister_gap_m=24.0, a_drift_m=90.0, n_fuel_elements=2, surrogate=model)
    kinds = {type(warning) for warning in prediction.warnings}
    assert OutsideTrainedRange in kinds

    samples = [*prediction.warnings, UncertainInterpolation(0.8, 0.5), PeakAtRecordEnd(1000.0)]
    for warning in samples:
        assert app.warning_text(warning) != str(warning), warning


def test_every_feature_has_a_label_and_a_unit(model):
    for name in model.feature_names:
        label, _ = app.feature_label(name)
        assert label != name and "[" in label, name
