# SPDX-License-Identifier: GPL-3.0-or-later
"""Result plots: thread safety, time axes, backgrounds, label placement."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from theresa.plotting import plot_peak_temperature


def history() -> tuple[np.ndarray, np.ndarray]:
    """A plausible peak-temperature curve on a log time axis."""
    times = np.logspace(-2.0, 3.0, 200)
    return times, 20.0 + 55.0 * np.exp(-((np.log10(times) - 0.8) ** 2))


def test_writes_a_file(tmp_path):
    times, temperature = history()
    path = plot_peak_temperature(
        times,
        temperature,
        tmp_path / "plot.png",
        limit_celsius=100.0,
        undisturbed_celsius=25.0,
    )
    assert path.exists() and path.stat().st_size > 0


def test_survives_concurrent_calls(tmp_path):
    """Twelve concurrent threads all produce their figure."""
    times, temperature = history()
    failures: list[BaseException] = []

    def draw(index: int) -> None:
        try:
            plot_peak_temperature(
                times,
                temperature,
                tmp_path / f"plot_{index}.png",
                limit_celsius=100.0,
                undisturbed_celsius=25.0,
                # With and without title.
                title="" if index % 2 else "Peak temperature",
            )
        except BaseException as error:  # noqa: BLE001 - reported, not swallowed
            failures.append(error)

    threads = [threading.Thread(target=draw, args=(index,)) for index in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not failures, f"{len(failures)} of 12 threads failed: {failures[0]!r}"
    assert all((tmp_path / f"plot_{index}.png").exists() for index in range(12))


def test_rejects_inconsistent_input(tmp_path):
    times, temperature = history()
    with pytest.raises(ValueError, match="same shape"):
        plot_peak_temperature(times, temperature[:-1], tmp_path / "plot.png")
    with pytest.raises(ValueError, match="time_scale"):
        plot_peak_temperature(times, temperature, tmp_path / "plot.png", time_scale="sqrt")
    with pytest.raises(ValueError, match="unknown text keys"):
        plot_peak_temperature(times, temperature, tmp_path / "plot.png", text={"xlabel": "t"})


def test_zero_is_a_time_on_a_linear_axis_and_not_on_a_log_one(tmp_path):
    """``t = 0`` is drawn on a linear axis and omitted on a log axis."""
    times = np.concatenate(([0.0], np.linspace(1.0, 1000.0, 199)))
    temperature = np.concatenate(([25.0], np.linspace(60.0, 45.0, 199)))

    assert plot_peak_temperature(
        times, temperature, tmp_path / "linear.png", time_scale="linear"
    ).exists()
    assert plot_peak_temperature(
        times, temperature, tmp_path / "log.png", time_scale="log"
    ).exists()

    # Only the log axis is left with nothing to draw.
    with pytest.raises(ValueError, match="no positive times"):
        plot_peak_temperature(
            np.zeros(3), np.array([20.0, 21.0, 22.0]), tmp_path / "x.png", time_scale="log"
        )
    assert plot_peak_temperature(
        np.zeros(3), np.array([20.0, 21.0, 22.0]), tmp_path / "y.png", time_scale="linear"
    ).exists()


def test_background_reaches_the_file(tmp_path):
    """The background colour can be set."""
    from PIL import Image

    times, temperature = history()
    path = plot_peak_temperature(times, temperature, tmp_path / "white.png", background="#ffffff")
    assert Image.open(path).convert("RGB").getpixel((2, 2)) == (255, 255, 255)


def test_band_and_reference_together(tmp_path):
    """Uncertainty band and direct run of different lengths, on both time axes."""
    times = np.concatenate(([0.0], np.linspace(1.0, 1000.0, 199)))
    temperature = 25.0 + 40.0 * np.exp(-((times - 30.0) ** 2) / 2e4)
    sigma = np.full_like(times, 0.2)
    reference = (np.linspace(0.0, 1000.0, 57), 25.0 + 40.0 * np.ones(57))
    for scale in ("linear", "log"):
        assert plot_peak_temperature(
            times,
            temperature,
            tmp_path / f"{scale}.png",
            sigma_kelvin=sigma,
            reference=reference,
            time_scale=scale,
        ).exists()


def test_power_plot_writes_a_file_and_a_stream(tmp_path):
    import io

    from theresa.plotting import plot_canister_power

    times = np.linspace(0.0, 1000.0, 200)
    power = 1235.0 * np.exp(-times / 60.0) + 200.0 * np.exp(-times / 2000.0)
    assert plot_canister_power(times, power, tmp_path / "power.png").exists()

    stream = io.BytesIO()
    plot_canister_power(times, power, stream, stream_format="svg", text={"power_label": "P [W]"})
    assert stream.getvalue().lstrip().startswith(b"<?xml")

    with pytest.raises(ValueError, match="same, non-empty shape"):
        plot_canister_power(times, power[:-1], tmp_path / "x.png")


@pytest.mark.parametrize("scale", ["linear", "log"])
def test_direct_labels_do_not_overlap(tmp_path, monkeypatch, scale):
    """Peak label, limit label and curves do not overlap, also with the peak on the limit."""
    from matplotlib.transforms import Bbox

    import theresa.plotting as plotting

    found: list[str] = []
    save = plotting._save

    def inspecting_save(figure, path, background, stream_format, place=None):
        figure.tight_layout()
        place()
        renderer = figure.canvas.get_renderer()
        axes = figure.axes[0]
        texts = [text for text in axes.texts if text.get_text()]
        boxes = [text.get_window_extent(renderer) for text in texts]
        pixels = plotting._line_pixels(axes.get_lines())
        for i, box in enumerate(boxes):
            for other in boxes[i + 1 :]:
                if Bbox.intersection(box, other) is not None:
                    found.append(f"{texts[i].get_text()!r} overlaps another label")
            on_line = (
                (pixels[:, 0] >= box.x0)
                & (pixels[:, 0] <= box.x1)
                & (pixels[:, 1] >= box.y0)
                & (pixels[:, 1] <= box.y1)
            )
            if on_line.any():
                found.append(f"{texts[i].get_text()!r} lies on a line")
        return save(figure, path, background, stream_format)

    monkeypatch.setattr(plotting, "_save", inspecting_save)

    times, temperature = history()
    temperature = temperature - temperature.max() + 100.0  # peak exactly on the limit
    plot_peak_temperature(
        times,
        temperature,
        tmp_path / "fit.png",
        limit_celsius=100.0,
        undisturbed_celsius=float(temperature.min()) - 5.0,
        sigma_kelvin=np.full_like(times, 0.1),
        reference=(times, temperature - 0.3),
        title="",
        figsize=(6.5, 3.2),
        time_scale=scale,
    )
    assert found == []


def test_both_palettes_draw_and_an_unknown_one_is_refused(tmp_path):
    times, temperature = history()
    for palette in ("light", "dark"):
        assert plot_peak_temperature(
            times, temperature, tmp_path / f"{palette}.png", palette=palette, limit_celsius=100.0
        ).exists()
    with pytest.raises(ValueError, match="palette"):
        plot_peak_temperature(times, temperature, tmp_path / "x.png", palette="sepia")
