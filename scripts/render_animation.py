# SPDX-License-Identifier: GPL-3.0-or-later
"""Animated SVG of the peak canister-surface temperature for the README.

The curve is drawn from emplacement to 1000 a on a linear time axis at
constant speed in time, with a running time display and a marker at the
maximum. SMIL animation, no scripts: the files animate when embedded as
images. Without animation support the final frame is shown.

``docs/figures/temperature.svg``
    light style.
``docs/figures/temperature-dark.svg``
    dark style.

Run::

    uv run python scripts/render_animation.py
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from theresa import predict  # noqa: E402
from theresa.plotting import DARK, LIGHT, Palette  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Design shown: the command-line example of the README.
DESIGN = {"canister_gap_m": 9.5, "a_drift_m": 40.0, "depth_m": 700.0}

WIDTH, HEIGHT = 760, 400
LEFT, RIGHT, TOP, BOTTOM = 70, 24, 64, 56
T_END = 1000.0
FONT = "Lato, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"

#: Animation cycle [s] and its phases as fractions: drawing ends at DRAW,
#: fading out starts at FADE.
CYCLE = 14.0
DRAW = 0.72
FADE = 0.95

#: Step [a] of the running time display.
CLOCK_STEP = 10.0


def _x(t):
    """Horizontal position of time ``t`` [a]."""
    return LEFT + np.asarray(t) / T_END * (WIDTH - LEFT - RIGHT)


def _keys(*points: float) -> str:
    """``keyTimes`` / ``values`` list."""
    return ";".join(f"{p:.5f}" for p in points)


def _repeat(attribute: str, values: str, key_times: str, mode: str = "linear") -> str:
    """Endless SMIL animation over one cycle."""
    return (
        f'<animate attributeName="{attribute}" values="{values}" keyTimes="{key_times}" '
        f'calcMode="{mode}" dur="{CYCLE:g}s" repeatCount="indefinite"/>'
    )


def build(colors: Palette, times: np.ndarray, temperature: np.ndarray, answer) -> str:
    """SVG source for one palette."""
    keep = times <= T_END
    t, temp = times[keep], temperature[keep]
    limit = float(answer.config["T_limit_celsius"])
    rock = answer.undisturbed_celsius
    low = 10.0 * math.floor(min(rock, temp.min()) / 10.0 - 0.5)
    high = 10.0 * math.ceil(max(limit, temp.max()) / 10.0 + 0.5)

    def y(value):
        return TOP + (high - np.asarray(value)) / (high - low) * (HEIGHT - TOP - BOTTOM)

    plot_width = WIDTH - LEFT - RIGHT
    xs, ys = _x(t), y(temp)
    path = "M" + " L".join(f"{a:.2f},{b:.2f}" for a, b in zip(xs, ys, strict=True))
    peak = int(np.argmax(temp))
    peak_key = DRAW * t[peak] / T_END

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" '
        f'width="{WIDTH}" height="{HEIGHT}" font-family="{FONT}">',
        f'<rect width="{WIDTH}" height="{HEIGHT}" rx="8" fill="{colors.surface}"/>',
        f'<text x="{LEFT}" y="28" font-size="17" font-weight="600" fill="{colors.text}">'
        "Peak canister-surface temperature</text>",
        f'<text x="{LEFT}" y="48" font-size="12.5" fill="{colors.text_secondary}">'
        f"clear spacing {DESIGN['canister_gap_m']:g} m, drift spacing "
        f"{DESIGN['a_drift_m']:g} m, depth {DESIGN['depth_m']:g} m, "
        f"{answer.canister_power_watt:.0f} W at emplacement</text>",
    ]

    # Axes and grid.
    for value in range(int(low), int(high) + 1, 10):
        level = float(y(value))
        parts.append(
            f'<line x1="{LEFT}" x2="{WIDTH - RIGHT}" y1="{level:.2f}" y2="{level:.2f}" '
            f'stroke="{colors.grid}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{LEFT - 8}" y="{level + 4:.2f}" font-size="12" text-anchor="end" '
            f'fill="{colors.text}">{value}</text>'
        )
    for tick in range(0, int(T_END) + 1, 200):
        position = float(_x(tick))
        parts.append(
            f'<text x="{position:.2f}" y="{HEIGHT - BOTTOM + 18}" font-size="12" '
            f'text-anchor="middle" fill="{colors.text}">{tick}</text>'
        )
    parts += [
        f'<line x1="{LEFT}" x2="{WIDTH - RIGHT}" y1="{HEIGHT - BOTTOM}" '
        f'y2="{HEIGHT - BOTTOM}" stroke="{colors.axis}" stroke-width="1.2"/>',
        f'<text x="{LEFT + plot_width / 2:.1f}" y="{HEIGHT - 14}" font-size="12.5" '
        f'text-anchor="middle" fill="{colors.text_secondary}">'
        "time since emplacement [a]</text>",
        f'<text transform="translate(20 {(TOP + HEIGHT - BOTTOM) / 2:.1f}) rotate(-90)" '
        f'font-size="12.5" text-anchor="middle" fill="{colors.text_secondary}">'
        "temperature [°C]</text>",
    ]

    # Reference levels.
    for level, color, dash, label in (
        (limit, colors.critical, "6 4", f"design limit {limit:g} °C"),
        (rock, colors.text_secondary, "2 4", f"undisturbed rock {rock:.1f} °C"),
    ):
        position = float(y(level))
        parts.append(
            f'<line x1="{LEFT}" x2="{WIDTH - RIGHT}" y1="{position:.2f}" '
            f'y2="{position:.2f}" stroke="{color}" stroke-width="1.5" '
            f'stroke-dasharray="{dash}"/>'
        )
        parts.append(
            f'<text x="{WIDTH - RIGHT - 4}" y="{position - 6:.2f}" font-size="12" '
            f'text-anchor="end" fill="{color}">{label}</text>'
        )

    # Everything below fades out at the end of a cycle.
    parts += [
        "<g>",
        _repeat("opacity", "1;1;0;0", _keys(0, FADE, 0.999, 1)),
        "<defs>",
        '<clipPath id="drawn">',
        f'<rect x="{LEFT - 4}" y="0" height="{HEIGHT}" width="{plot_width + 8}">',
        _repeat("width", _keys(0, plot_width + 8, plot_width + 8), _keys(0, DRAW, 1)),
        "</rect>",
        "</clipPath>",
        "</defs>",
        f'<path d="{path}" fill="none" stroke="{colors.series}" stroke-width="2.6" '
        'stroke-linejoin="round" stroke-linecap="round" clip-path="url(#drawn)"/>',
    ]

    # Tip of the curve, moving at constant speed in time.
    samples = np.linspace(0.0, T_END, 241)
    tip_x = _x(samples)
    tip_y = y(np.interp(samples, t, temp))
    sample_keys = DRAW * samples / T_END
    key_times = _keys(*sample_keys, 1.0)
    parts += [
        f'<circle cx="{tip_x[-1]:.2f}" cy="{tip_y[-1]:.2f}" r="4.5" fill="{colors.series}">',
        _repeat("cx", _keys(*tip_x, tip_x[-1]), key_times),
        _repeat("cy", _keys(*tip_y, tip_y[-1]), key_times),
        "</circle>",
    ]

    # Running time display.
    clock_x, clock_y = WIDTH - RIGHT, 30
    steps = np.arange(0.0, T_END + CLOCK_STEP / 2, CLOCK_STEP)
    for index, step in enumerate(steps):
        start = DRAW * step / T_END
        end = DRAW * min(step + CLOCK_STEP, T_END) / T_END if index < len(steps) - 1 else 1.0
        values, keys = ("hidden;visible;hidden", _keys(0, start, end))
        if index == 0:
            values, keys = ("visible;hidden", _keys(0, end))
        elif index == len(steps) - 1:
            values, keys = ("hidden;visible", _keys(0, start))
        default = "visible" if index == len(steps) - 1 else "hidden"
        parts.append(
            f'<text x="{clock_x}" y="{clock_y}" font-size="15" font-weight="600" '
            f'text-anchor="end" fill="{colors.text}" visibility="{default}" '
            f'font-variant-numeric="tabular-nums">t = {step:.0f} a'
            f"{_repeat('visibility', values, keys, 'discrete')}</text>"
        )

    # Maximum, shown once the curve has passed it.
    px, py = float(xs[peak]), float(ys[peak])
    parts += [
        "<g>",
        _repeat("opacity", "0;0;1;1", _keys(0, peak_key, peak_key + 0.03, 1)),
        f'<circle cx="{px:.2f}" cy="{py:.2f}" r="6" fill="{colors.peak}" '
        f'stroke="{colors.surface}" stroke-width="2"/>',
        f'<text x="{px + 14:.2f}" y="{py + 5:.2f}" font-size="13" font-weight="600" '
        f'fill="{colors.text}">maximum {temp[peak]:.1f} °C after {t[peak]:.1f} a</text>',
        "</g>",
        "</g>",
        "</svg>",
    ]
    return "\n".join(parts) + "\n"


def main() -> int:
    """Write the light and dark animation."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "docs" / "figures")
    args = parser.parse_args()

    answer = predict(**DESIGN)
    times = np.asarray(answer.times_years, dtype=float)
    temperature = np.asarray(answer.temperature_celsius, dtype=float)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, colors in (("temperature.svg", LIGHT), ("temperature-dark.svg", DARK)):
        path = args.out / name
        path.write_text(build(colors, times, temperature, answer), encoding="utf-8")
        print(f"{path.relative_to(PROJECT_ROOT)}: {path.stat().st_size / 1024:.0f} kB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
