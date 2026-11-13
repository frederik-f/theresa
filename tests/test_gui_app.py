# SPDX-License-Identifier: GPL-3.0-or-later
"""Full page run headless with Streamlit's ``AppTest``; no exceptions allowed."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

from theresa.predict import DEFAULT_SURROGATE_PATH  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "theresa_gui" / "app.py")

pytestmark = pytest.mark.skipif(not DEFAULT_SURROGATE_PATH.exists(), reason="no trained surrogate")


def page() -> AppTest:
    app = AppTest.from_file(APP, default_timeout=120)
    app.run()
    return app


def loading_mode(app: AppTest, mode: str) -> None:
    radio = next(r for r in app.sidebar.radio if r.options and len(r.options) == 3)
    radio.set_value(mode).run()


def test_the_default_page_renders_in_both_languages():
    app = page()
    assert not app.exception
    app.sidebar.selectbox[0].set_value("en").run()
    assert not app.exception
    assert any("Peak temperature" in metric.label for metric in app.metric)
    assert any("hobby project" in caption.value for caption in app.caption)


@pytest.mark.parametrize("mode", ["elements", "power", "fit"])
def test_a_decay_curve_without_power_is_reported_not_raised(mode):
    app = page()
    loading_mode(app, mode)
    for i in range(6):
        app.number_input(key=f"p{i}").set_value(0.0)
    app.run()
    assert not app.exception


def test_an_unreadable_surrogate_file_is_reported(tmp_path):
    junk = tmp_path / "junk.pkl"
    junk.write_text("not a pickle")
    app = page()
    next(box for box in app.sidebar.text_input).set_value(str(junk)).run()
    assert not app.exception
    assert app.error


def test_without_direct_simulation_only_the_included_surrogate_is_evaluated(monkeypatch):
    monkeypatch.setenv("THERESA_DIRECT_SIMULATION", "off")
    app = page()
    assert not app.exception
    assert not app.sidebar.text_input
    assert not [button for button in app.button if "OGS" in button.label]
    assert any("OGS" in caption.value for caption in app.caption)
    assert app.metric
