# SPDX-License-Identifier: GPL-3.0-or-later
"""Result plots: peak canister-surface temperature and canister power over time.

Reference levels (design limit, undisturbed rock) are labelled directly;
labels and legend are placed free of overlap.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

import numpy as np

#: Serialises figure production; matplotlib's global state is not thread-safe.
_FIGURE_LOCK = threading.Lock()

#: Colours of the light palette.
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#000000"
TEXT_SECONDARY = "#52514e"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = "#2a78d6"
CRITICAL = "#d03b3b"


@dataclass(frozen=True)
class Palette:
    """Colours of one figure style.

    :param surface: figure and axes background.
    :param text: tick labels, axis titles and the headline labels.
    :param text_secondary: undisturbed-rock line and label, legend, subtitle.
    :param grid: grid lines.
    :param axis: spines.
    :param series: main curve and uncertainty band.
    :param critical: design limit.
    :param peak: marker at the maximum.
    :param reference: directly simulated comparison curve.
    """

    surface: str
    text: str
    text_secondary: str
    grid: str
    axis: str
    series: str
    critical: str
    peak: str
    reference: str


#: Light style: dark text on off-white.
LIGHT = Palette(
    surface=SURFACE,
    text=TEXT_PRIMARY,
    text_secondary=TEXT_SECONDARY,
    grid=GRIDLINE,
    axis=AXIS,
    series=SERIES,
    critical=CRITICAL,
    peak=SERIES,
    reference=TEXT_PRIMARY,
)

#: Dark style in the THERESA colours.
DARK = Palette(
    surface="#0c1c2f",
    text="#e6edf3",
    text_secondary="#9fb6cf",
    grid="#1d3552",
    axis="#2c4a6c",
    series="#5a9be0",
    critical="#e5533d",
    peak="#f2b84b",
    reference="#e6edf3",
)

#: Styles by name.
PALETTES: dict[str, Palette] = {"light": LIGHT, "dark": DARK}


def _palette(name: str) -> Palette:
    """Palette ``name``.

    :raises ValueError: on an unknown name.
    """
    try:
        return PALETTES[name]
    except KeyError:
        raise ValueError(f"palette must be one of {sorted(PALETTES)}, not {name!r}") from None


#: Figure texts; overridable for translation. Units and numbers are fixed.
DEFAULT_TEXT: dict[str, str] = {
    "x_label": "time since emplacement [a]",
    "y_label": "temperature [°C]",
    "limit": "design limit",
    "undisturbed": "undisturbed rock",
    "at": "at",
    "power_label": "canister power [W]",
    "emplacement": "at emplacement",
}


def _words(text: dict[str, str] | None) -> dict[str, str]:
    """:data:`DEFAULT_TEXT` with overrides.

    :raises ValueError: on an unknown key.
    """
    unknown = set(text or {}) - DEFAULT_TEXT.keys()
    if unknown:
        raise ValueError(f"unknown text keys {sorted(unknown)}; known: {sorted(DEFAULT_TEXT)}")
    return {**DEFAULT_TEXT, **(text or {})}


def _pyplot():
    """Matplotlib pyplot with the Agg backend."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _styled_axes(
    figsize: tuple[float, float], dpi: int, background: str, time_scale: str, colors: Palette
):
    """Styled figure and axes. Call under :data:`_FIGURE_LOCK`."""
    figure, axes = _pyplot().subplots(figsize=figsize, dpi=dpi)
    figure.patch.set_facecolor(background)
    axes.set_facecolor(background)

    axes.set_xscale(time_scale)
    if time_scale == "linear":
        # 1 % margin: -10 a to 1010 a for a 1000 a record.
        axes.margins(x=0.01)
    axes.grid(True, which="major", color=colors.grid, linewidth=1.0, linestyle="-")
    axes.set_axisbelow(True)

    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        axes.spines[side].set_color(colors.axis)
        axes.spines[side].set_linewidth(1.0)
    axes.tick_params(colors=colors.text, labelsize=9, length=0)
    return figure, axes


def _label_axes(axes, x_label: str, y_label: str, colors: Palette) -> None:
    """Axis titles."""
    axes.set_xlabel(x_label, color=colors.text, fontsize=10)
    axes.set_ylabel(y_label, color=colors.text, fontsize=10)


#: Candidate label position: anchor, offset x and y [pt], alignment h and v.
Placement = tuple[tuple[float, float], float, float, str, str]


@dataclass
class _Label:
    """Direct label with candidate positions in order of preference.

    :param text: the label.
    :param placements: candidate positions; the first clear one is used.
    :param xycoords: coordinate system of every anchor point.
    :param style: text properties passed on to ``annotate``.
    """

    text: str
    placements: tuple[Placement, ...]
    xycoords: str | tuple[str, str] = "data"
    style: dict = field(default_factory=dict)


#: Cost of one line pixel inside a label, relative to one px^2 of box overlap.
LINE_PIXEL_COST = 50.0

#: Sampling distance [px] of lines for the overlap test.
LINE_SAMPLING_PX = 2.0


def _line_pixels(lines) -> np.ndarray:
    """Points [px] along all lines, sampled every :data:`LINE_SAMPLING_PX`."""
    chunks = [np.empty((0, 2))]
    for line in lines:
        x, y = line.get_data()
        vertices = line.get_transform().transform(
            np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])
        )
        vertices = vertices[np.all(np.isfinite(vertices), axis=1)]
        chunks.append(vertices)
        for start, end in zip(vertices[:-1], vertices[1:], strict=True):
            steps = int(np.hypot(*(end - start)) // LINE_SAMPLING_PX)
            if steps > 1:
                fractions = np.linspace(0.0, 1.0, steps + 1)[1:-1, None]
                chunks.append(start + fractions * (end - start))
    return np.vstack(chunks)


def _box_area(box) -> float:
    """Area [px^2] of a bounding box, zero for ``None``."""
    return 0.0 if box is None else float(box.width * box.height)


def _clash(box, frame, occupied: list, pixels: np.ndarray) -> float:
    """Overlap cost of a box [px]; zero if clear.

    Sum of the area outside the axes, the overlap with ``occupied`` and the
    line pixels inside, weighted by :data:`LINE_PIXEL_COST`.
    """
    from matplotlib.transforms import Bbox

    inside = (
        (pixels[:, 0] >= box.x0)
        & (pixels[:, 0] <= box.x1)
        & (pixels[:, 1] >= box.y0)
        & (pixels[:, 1] <= box.y1)
    )
    return (
        _box_area(box)
        - _box_area(Bbox.intersection(box, frame))
        + sum(_box_area(Bbox.intersection(box, other)) for other in occupied)
        + LINE_PIXEL_COST * LINE_SAMPLING_PX * float(inside.sum())
    )


def _place_labels(figure, axes, labels: list[_Label], lines) -> list:
    """Place each label at its first clear position, else at the least overlap.

    Labels are placed in order. Call after the axis limits and layout are
    final.

    :param lines: ``Line2D`` artists to keep clear.
    :returns: pixel boxes of the labels.
    """
    renderer = figure.canvas.get_renderer()
    frame = axes.get_window_extent(renderer)
    occupied: list = []
    pixels = _line_pixels(lines)

    for label in labels:
        best = None
        for anchor, dx, dy, ha, va in label.placements:
            artist = axes.annotate(
                label.text,
                xy=anchor,
                xycoords=label.xycoords,
                xytext=(dx, dy),
                textcoords="offset points",
                ha=ha,
                va=va,
                **label.style,
            )
            box = artist.get_window_extent(renderer).padded(2.0)
            cost = _clash(box, frame, occupied, pixels)
            if best is None or cost < best[0]:
                if best is not None:
                    best[1].remove()
                best = (cost, artist, box)
            else:
                artist.remove()
            if cost == 0.0:
                break
        occupied.append(best[2])
    return occupied


#: Legend locations in order of preference.
LEGEND_LOCATIONS = (
    "lower right",
    "center right",
    "upper right",
    "lower center",
    "upper center",
    "lower left",
    "center left",
    "upper left",
)


def _place_legend(figure, axes, occupied: list, lines, colors: Palette) -> None:
    """Legend at the first location clear of labels and lines.

    :param occupied: pixel boxes of the direct labels, from
        :func:`_place_labels`.
    :param lines: ``Line2D`` artists to keep clear.
    """
    renderer = figure.canvas.get_renderer()
    frame = axes.get_window_extent(renderer)
    pixels = _line_pixels(lines)

    best = None
    for location in LEGEND_LOCATIONS:
        legend = axes.legend(loc=location, frameon=False, fontsize=9)
        cost = _clash(legend.get_window_extent(renderer).padded(2.0), frame, occupied, pixels)
        if best is None or cost < best[0]:
            best = (cost, location)
        if cost == 0.0:
            break
    legend = axes.legend(loc=best[1], frameon=False, fontsize=9)
    for entry in legend.get_texts():
        entry.set_color(colors.text_secondary)


def _save(
    figure,
    path: Path | BinaryIO,
    background: str,
    stream_format: str,
    place: Callable[[], object] | None = None,
) -> Path | BinaryIO:
    """Write ``figure`` to a file or stream and close it.

    :param place: called after the layout, before writing.
    """
    figure.tight_layout()
    if place is not None:
        place()
    if hasattr(path, "write"):
        figure.savefig(path, format=stream_format, facecolor=background)
    else:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, facecolor=background)
    _pyplot().close(figure)
    return path


def plot_peak_temperature(
    times_years: np.ndarray,
    temperature_celsius: np.ndarray,
    path: Path | BinaryIO,
    *,
    limit_celsius: float | None = None,
    undisturbed_celsius: float | None = None,
    sigma_kelvin: np.ndarray | None = None,
    reference: tuple[np.ndarray, np.ndarray] | None = None,
    labels: tuple[str, str] = ("surrogate", "direct simulation"),
    title: str = "Peak temperature at the canister surface",
    subtitle: str | None = None,
    figsize: tuple[float, float] = (8.0, 4.8),
    dpi: int = 150,
    time_scale: str = "linear",
    background: str | None = None,
    palette: str = "light",
    text: dict[str, str] | None = None,
    stream_format: str = "png",
) -> Path | BinaryIO:
    """Plot the peak-temperature history.

    :param times_years: times [a] since emplacement, ascending.
    :param temperature_celsius: peak canister-surface temperature [degC].
    :param path: output file (format from the suffix) or binary stream.
    :param limit_celsius: design limit to mark, or ``None``.
    :param undisturbed_celsius: undisturbed rock temperature at emplacement
        level, or ``None``.
    :param sigma_kelvin: one-sigma interpolation uncertainty per time step,
        drawn as a two-sigma band.
    :param reference: ``(times_years, temperature_celsius)`` of a direct
        simulation.
    :param labels: legend labels for the main curve and ``reference``.
    :param title: figure title; empty for none.
    :param subtitle: text under the title.
    :param figsize: figure size [in].
    :param dpi: resolution of a raster format.
    :param time_scale: ``"linear"`` or ``"log"``.
    :param background: figure background; ``None`` for the palette's.
    :param palette: ``"light"`` or ``"dark"``, see :data:`PALETTES`.
    :param text: overrides for :data:`DEFAULT_TEXT`.
    :param stream_format: format for a stream, e.g. ``"png"`` or ``"svg"``.
    :raises ValueError: on inconsistent input or unknown options.
    """
    words = _words(text)
    colors = _palette(palette)
    background = colors.surface if background is None else background

    t = np.asarray(times_years, dtype=float)
    temperature = np.asarray(temperature_celsius, dtype=float)
    if t.shape != temperature.shape:
        raise ValueError("times_years and temperature_celsius must have the same shape")

    if time_scale not in ("linear", "log"):
        raise ValueError(f"time_scale must be 'linear' or 'log', not {time_scale!r}")

    # A log axis omits t = 0.
    keep = t > 0.0 if time_scale == "log" else np.ones_like(t, dtype=bool)
    t, temperature = t[keep], temperature[keep]
    sigma = None if sigma_kelvin is None else np.asarray(sigma_kelvin, dtype=float)[keep]
    if t.size == 0:
        raise ValueError("no positive times to plot" if time_scale == "log" else "no times to plot")

    peak_index = int(np.argmax(temperature))
    t_peak, t_max = t[peak_index], temperature[peak_index]

    with _FIGURE_LOCK:
        figure, axes = _styled_axes(figsize, dpi, background, time_scale, colors)

        # Lines to keep clear of labels; labels are placed last.
        lines = []
        direct: list[_Label] = []

        # Peak label, placed first.
        direct.append(
            _Label(
                f"{t_max:.1f} °C  {words['at']}  {t_peak:.0f} a",
                placements=(
                    ((t_peak, t_max), 10, 6, "left", "bottom"),
                    ((t_peak, t_max), -10, 6, "right", "bottom"),
                    ((t_peak, t_max), 10, 18, "left", "bottom"),
                    ((t_peak, t_max), 10, -8, "left", "top"),
                    ((t_peak, t_max), -10, -8, "right", "top"),
                ),
                style={"color": colors.text, "fontsize": 10, "fontweight": "bold"},
            )
        )

        # Reference levels below the curve.
        if limit_celsius is not None:
            lines.append(axes.axhline(limit_celsius, color=colors.critical, linewidth=1.5))
            direct.append(
                _Label(
                    f"{words['limit']}  {limit_celsius:.0f} °C",
                    placements=_along_line(limit_celsius),
                    xycoords=("axes fraction", "data"),
                    style={"color": colors.critical, "fontsize": 8.5},
                )
            )
        if undisturbed_celsius is not None:
            lines.append(
                axes.axhline(undisturbed_celsius, color=colors.text_secondary, linewidth=1.0)
            )
            direct.append(
                _Label(
                    f"{words['undisturbed']}  {undisturbed_celsius:.1f} °C",
                    placements=_along_line(undisturbed_celsius),
                    xycoords=("axes fraction", "data"),
                    style={"color": colors.text_secondary, "fontsize": 8.5},
                )
            )

        # +/-2 sigma band below the curve.
        if sigma is not None:
            axes.fill_between(
                t,
                temperature - 2.0 * sigma,
                temperature + 2.0 * sigma,
                color=colors.series,
                alpha=0.18,
                linewidth=0,
            )

        lines += axes.plot(
            t, temperature, color=colors.series, linewidth=2.0, solid_capstyle="round"
        )
        lines += axes.plot(
            [t_peak],
            [t_max],
            marker="o",
            markersize=8,
            color=colors.peak,
            markeredgecolor=background,
            markeredgewidth=2.0,
            zorder=5,
        )

        if reference is not None:
            reference_t = np.asarray(reference[0], dtype=float)
            reference_temperature = np.asarray(reference[1], dtype=float)
            shown = reference_t > 0.0 if time_scale == "log" else np.ones_like(reference_t, bool)
            lines += axes.plot(
                reference_t[shown],
                reference_temperature[shown],
                color=colors.reference,
                linewidth=1.5,
                linestyle=(0, (4, 3)),
                label=labels[1],
            )
            axes.plot([], [], color=colors.series, linewidth=2.0, label=labels[0])

        _label_axes(axes, words["x_label"], words["y_label"], colors)

        # Title padding grows with the number of subtitle lines.
        if title:
            subtitle_lines = subtitle.count("\n") + 1 if subtitle else 0
            axes.set_title(
                title,
                color=colors.text,
                fontsize=12,
                fontweight="bold",
                loc="left",
                pad=12 + 14 * subtitle_lines,
            )
        if subtitle:
            axes.annotate(
                subtitle,
                xy=(0.0, 1.0),
                xycoords="axes fraction",
                xytext=(0, 8),
                textcoords="offset points",
                color=colors.text_secondary,
                fontsize=9.5,
                va="bottom",
            )

        # Headroom for the labels.
        low = min(
            float(temperature.min()),
            undisturbed_celsius if undisturbed_celsius is not None else np.inf,
            limit_celsius if limit_celsius is not None else np.inf,
        )
        high = max(float(t_max), limit_celsius or float(t_max))
        if sigma is not None:
            high = max(high, float((temperature + 2.0 * sigma).max()))
        if reference is not None:
            high = max(high, float(np.asarray(reference[1]).max()))
            low = min(low, float(np.asarray(reference[1]).min()))
        span = max(high - low, 1.0)
        axes.set_ylim(low - 0.14 * span, high + 0.18 * span)

        def place() -> None:
            occupied = _place_labels(figure, axes, direct, lines)
            if reference is not None:
                _place_legend(figure, axes, occupied, lines, colors)

        return _save(figure, path, background, stream_format, place)


def _along_line(level: float) -> tuple[Placement, ...]:
    """Label positions of a horizontal line: above right, above left, below."""
    right, left = ((0.992, level), "right"), ((0.008, level), "left")
    return tuple(
        (anchor, 0, dy, ha, va)
        for dy, va in ((5, "bottom"), (-5, "top"))
        for anchor, ha in (right, left)
    )


def plot_canister_power(
    times_years: np.ndarray,
    power_watt: np.ndarray,
    path: Path | BinaryIO,
    *,
    title: str = "Canister power",
    figsize: tuple[float, float] = (8.0, 4.8),
    dpi: int = 150,
    background: str | None = None,
    palette: str = "light",
    text: dict[str, str] | None = None,
    stream_format: str = "png",
) -> Path | BinaryIO:
    """Plot the canister decay-heat power over time.

    :param times_years: times [a] since emplacement, ascending, from zero.
    :param power_watt: canister power [W] at those times.
    :param path: output file or binary stream, as for
        :func:`plot_peak_temperature`.
    :param title: figure title; empty for none.
    :param figsize: figure size [in].
    :param dpi: resolution of a raster format.
    :param background: figure background; ``None`` for the palette's.
    :param palette: ``"light"`` or ``"dark"``, see :data:`PALETTES`.
    :param text: overrides for :data:`DEFAULT_TEXT`.
    :param stream_format: format for a stream.
    :raises ValueError: on inconsistent input or unknown options.
    """
    words = _words(text)
    colors = _palette(palette)
    background = colors.surface if background is None else background
    t = np.asarray(times_years, dtype=float)
    power = np.asarray(power_watt, dtype=float)
    if t.shape != power.shape or t.size == 0:
        raise ValueError("times_years and power_watt must have the same, non-empty shape")

    with _FIGURE_LOCK:
        figure, axes = _styled_axes(figsize, dpi, background, "linear", colors)
        lines = axes.plot(t, power, color=colors.series, linewidth=2.0, solid_capstyle="round")

        start = (t[0], power[0])
        labels = [
            _Label(
                f"{power[0]:.0f} W  {words['emplacement']}",
                placements=(
                    (start, 10, 2, "left", "bottom"),
                    (start, 10, 12, "left", "bottom"),
                    (start, 24, -6, "left", "top"),
                ),
                style={"color": colors.text, "fontsize": 10, "fontweight": "bold"},
            )
        ]
        _label_axes(axes, words["x_label"], words["power_label"], colors)
        if title:
            axes.set_title(title, color=colors.text, fontsize=12, fontweight="bold", loc="left")
        axes.set_ylim(0.0, 1.15 * float(power.max()) or 1.0)
        return _save(
            figure,
            path,
            background,
            stream_format,
            lambda: _place_labels(figure, axes, labels, lines),
        )
