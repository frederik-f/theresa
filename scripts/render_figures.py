# SPDX-License-Identifier: GPL-3.0-or-later
"""Build all model figures.

``schema.png``
    domain and boundary conditions, dimensions from :mod:`theresa.config`.
``geometry.png``, ``mesh.png``, ``field.png``
    rendered by :mod:`scripts.paraview_figures` under ``pvpython``.

All images are cropped to their content.

Usage::

    uv run python scripts/render_figures.py --out docs/figures
    uv run python scripts/render_figures.py --out docs/figures --only schema

``--pvpython`` points to ParaView's ``pvpython`` (default: from ``PATH``);
without it only the schematic is produced.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from theresa.config import make_config  # noqa: E402

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

#: Region colours of :mod:`scripts.paraview_figures`, lightened.
CANISTER = "#b3b1ac"
BACKFILL = "#f5c172"
LINING = "#e2e1da"
HOST_ROCK = "#a1becd"

DIRICHLET = "#c0392b"
SYMMETRY = "#2c6fad"
TEXT = "#1a1a1a"
RULE = "#9a9a92"

#: Minimum difference from white of a content pixel when cropping.
CROP_THRESHOLD = 6
CROP_MARGIN = 24


# --------------------------------------------------------------------------
# The schematic
# --------------------------------------------------------------------------


def _symmetry_face(ax, x0, y0, x1, y1) -> None:
    """Mark a face as a zero-flux symmetry plane."""
    ax.plot([x0, x1], [y0, y1], color=SYMMETRY, linewidth=2.4, zorder=5, solid_capstyle="butt")


def _vertical_panel(ax, cfg: dict) -> None:
    """Section along the drift axis: Dirichlet faces and depth.

    Horizontally to scale; the drift band is enlarged vertically.
    """
    depth = cfg["depth_m"]
    h_below = cfg["h_below_m"]
    height = depth + h_below
    top_c = cfg["surface_temperature_celsius"]
    gradient = cfg["geothermal_gradient_k_per_m"]
    bottom_c = top_c + gradient * height

    half_length = cfg["l_can"] / 2.0
    half_pitch = half_length + cfg["canister_gap"] / 2.0
    r_can = cfg["r_can"]
    r_inner = cfg["drift_diameter"] / 2.0
    r_outer = r_inner + cfg["lining_thickness"]

    drift_h = 0.20
    above = (1.0 - drift_h) * depth / height
    below = (1.0 - drift_h) * h_below / height
    y_drift = below
    y_top = below + drift_h + above
    y_axis = y_drift + drift_h / 2.0

    ax.add_patch(Rectangle((0, 0), 1, y_top, facecolor=HOST_ROCK, edgecolor="none"))

    # The drift runs the full length of the cell; only the canister is finite.
    scale = (drift_h / 2.0) / r_outer
    for half, colour in ((r_outer, LINING), (r_inner, BACKFILL)):
        ax.add_patch(
            Rectangle(
                (0, y_axis - half * scale), 1, 2 * half * scale, facecolor=colour, edgecolor="none"
            )
        )
    for sign in (1.0, -1.0):
        ax.plot([0, 1], [y_axis + sign * r_inner * scale] * 2, color=RULE, lw=0.7, zorder=4)

    # Canister mid-plane at x = 0.
    x_can = half_length / half_pitch
    ax.add_patch(
        Rectangle(
            (0, y_axis - r_can * scale),
            x_can,
            2 * r_can * scale,
            facecolor=CANISTER,
            edgecolor="#7d7b75",
            linewidth=0.9,
        )
    )
    ax.text(
        x_can / 2.0, y_axis, "$Q(t)$", ha="center", va="center", fontsize=11, color=TEXT, zorder=7
    )

    for y, text, va in (
        (y_top, f"surface, $z=0$:  $T = T_{{top}}$  ({top_c:.0f} °C)", "bottom"),
        (
            0.0,
            f"$z=-H$:  $T = T_{{top}} + \\gamma H$  ({bottom_c:.0f} °C at "
            f"$\\gamma$ = {gradient:g} K/m)",
            "top",
        ),
    ):
        ax.plot([0, 1], [y, y], color=DIRICHLET, linewidth=3.2, zorder=5)
        offset = 0.03 if va == "bottom" else -0.03
        ax.text(0.5, y + offset, text, ha="center", va=va, fontsize=10, color=DIRICHLET)

    _symmetry_face(ax, 0, 0, 0, y_top)
    _symmetry_face(ax, 1, 0, 1, y_top)
    ax.text(
        -0.03,
        y_top * 0.55,
        "$x=0$  (canister mid-plane)",
        rotation=90,
        ha="right",
        va="center",
        fontsize=9,
        color=SYMMETRY,
    )
    ax.text(
        1.03,
        y_top * 0.55,
        "$x=$ pitch$/2$",
        rotation=90,
        ha="left",
        va="center",
        fontsize=9,
        color=SYMMETRY,
    )

    x_dim = 0.90
    for y_from, y_to, label in (
        (y_top, y_axis, f"$t_E$ = {depth:.0f} m"),
        (y_axis, 0.0, f"$H_{{below}}$ = {h_below:.0f} m"),
    ):
        ax.annotate(
            "",
            xy=(x_dim, y_from),
            xytext=(x_dim, y_to),
            arrowprops={"arrowstyle": "<->", "color": TEXT, "linewidth": 1.0},
        )
        ax.text(
            x_dim - 0.02,
            (y_from + y_to) / 2.0,
            label,
            rotation=90,
            ha="right",
            va="center",
            fontsize=9,
            color=TEXT,
        )

    for y in (y_drift, y_drift + drift_h):
        ax.plot([0, 1], [y, y], color="#5a5a5a", linewidth=0.9, linestyle=(0, (6, 4)), zorder=6)
    ax.text(
        0.5,
        y_drift - 0.022,
        "near field exaggerated vertically",
        ha="center",
        va="top",
        fontsize=8,
        color="#5a5a5a",
        style="italic",
        zorder=7,
    )

    ax.set_xlim(-0.30, 1.18)
    ax.set_ylim(-0.12, 1.10)
    ax.set_title(
        "(a) Vertical section along the drift axis ($y=0$)", fontsize=11, color=TEXT, pad=10
    )
    ax.axis("off")


def _plan_panel(ax, cfg: dict) -> None:
    """Plan at emplacement level: symmetry planes and spacings, to scale."""
    r_can = cfg["r_can"]
    half_length = cfg["l_can"] / 2.0
    gap = cfg["canister_gap"]
    half_pitch = half_length + gap / 2.0
    half_drift = cfg["a_drift"] / 2.0
    r_inner = cfg["drift_diameter"] / 2.0
    r_outer = r_inner + cfg["lining_thickness"]

    for width, height, colour in (
        (half_pitch, half_drift, HOST_ROCK),
        (half_pitch, r_outer, LINING),
        (half_pitch, r_inner, BACKFILL),
    ):
        ax.add_patch(Rectangle((0, 0), width, height, facecolor=colour, edgecolor="none"))
    ax.add_patch(
        Rectangle(
            (0, 0), half_length, r_can, facecolor=CANISTER, edgecolor="#7d7b75", linewidth=0.9
        )
    )
    for y in (r_inner, r_outer):
        ax.plot([0, half_pitch], [y, y], color=RULE, linewidth=0.7, zorder=4)

    _symmetry_face(ax, 0, 0, 0, half_drift)
    _symmetry_face(ax, half_pitch, 0, half_pitch, half_drift)
    _symmetry_face(ax, 0, 0, half_pitch, 0)
    _symmetry_face(ax, 0, half_drift, half_pitch, half_drift)

    ax.text(
        -0.5,
        half_drift / 2.0,
        "$x=0$",
        rotation=90,
        ha="right",
        va="center",
        fontsize=9,
        color=SYMMETRY,
    )
    ax.text(
        half_pitch + 0.5,
        half_drift / 2.0,
        "$x=$ pitch$/2$",
        rotation=90,
        ha="left",
        va="center",
        fontsize=9,
        color=SYMMETRY,
    )
    ax.text(
        half_pitch / 2.0,
        -0.7,
        "$y=0$, drift axis",
        ha="center",
        va="top",
        fontsize=9,
        color=SYMMETRY,
    )
    ax.text(
        half_pitch / 2.0,
        half_drift + 0.7,
        f"$y = a_{{drift}}/2$ = {half_drift:.0f} m",
        ha="center",
        va="bottom",
        fontsize=9,
        color=SYMMETRY,
    )

    # Near-field labels with short leaders.
    leaders = [
        (r_can * 0.5, f"canister, $Q(t)$\n$r_{{can}}$ = {r_can:.2f} m"),
        ((r_can + r_inner) / 2.0, "backfill"),
        ((r_inner + r_outer) / 2.0, "liner"),
    ]
    for index, (y, label) in enumerate(leaders):
        ax.annotate(
            label,
            xy=(half_pitch * 0.72, y),
            xytext=(half_pitch + 2.2, 1.0 + index * 3.1),
            ha="left",
            va="center",
            fontsize=9,
            color=TEXT,
            arrowprops={
                "arrowstyle": "-",
                "color": "#7d7b75",
                "linewidth": 0.8,
                "connectionstyle": "arc3,rad=0.0",
            },
        )

    for y_dim, x0, x1, label in (
        (-2.4, 0.0, half_length, f"$l_{{can}}/2$ = {half_length:.2f} m"),
        (-4.2, half_length, half_pitch, f"clear spacing$/2$ = {gap / 2.0:.2f} m"),
    ):
        ax.annotate(
            "",
            xy=(x0, y_dim),
            xytext=(x1, y_dim),
            arrowprops={"arrowstyle": "<->", "color": TEXT, "linewidth": 1.0},
        )
        for x in (x0, x1):
            ax.plot(
                [x, x], [y_dim, 0.1], color=RULE, linewidth=0.6, linestyle=(0, (3, 3)), zorder=3
            )
        ax.text((x0 + x1) / 2.0, y_dim - 0.35, label, ha="center", va="top", fontsize=9, color=TEXT)

    ax.set_xlim(-1.8, half_pitch + 8.0)
    ax.set_ylim(-6.6, half_drift + 2.6)
    ax.set_aspect("equal")
    ax.set_title(
        "(b) Plan view at emplacement level ($z=-t_E$), to scale",
        fontsize=11,
        color=TEXT,
        pad=10,
    )
    ax.axis("off")


def draw_schematic(out: Path) -> Path:
    """Domain and boundary conditions: section and plan."""
    cfg = make_config()
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.4), gridspec_kw={"width_ratios": [1.15, 1.0]})
    fig.patch.set_facecolor("white")
    _vertical_panel(axes[0], cfg)
    _plan_panel(axes[1], cfg)

    handles = [
        plt.Line2D(
            [], [], color=DIRICHLET, linewidth=3.0, label="Dirichlet, prescribed temperature"
        ),
        plt.Line2D(
            [],
            [],
            color=SYMMETRY,
            linewidth=2.4,
            label="symmetry plane, $\\partial T/\\partial n = 0$",
        ),
        Rectangle(
            (0, 0), 1, 1, facecolor=CANISTER, edgecolor="#7d7b75", label="canister, source $Q(t)$"
        ),
        Rectangle((0, 0), 1, 1, facecolor=BACKFILL, edgecolor="none", label="backfill (bentonite)"),
        Rectangle((0, 0), 1, 1, facecolor=LINING, edgecolor=RULE, label="liner (concrete)"),
        Rectangle(
            (0, 0), 1, 1, facecolor=HOST_ROCK, edgecolor="none", label="host rock (claystone)"
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.text(
        0.5,
        0.955,
        "Quarter unit cell: domain and boundary conditions",
        ha="center",
        va="bottom",
        fontsize=13,
        color=TEXT,
    )
    fig.subplots_adjust(left=0.035, right=0.985, top=0.89, bottom=0.15, wspace=0.02)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=190, facecolor="white")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------
# The rendered figures
# --------------------------------------------------------------------------


def find_pvpython(explicit: str | None) -> Path | None:
    """Locate ParaView's interpreter, or return ``None`` with a message."""
    if explicit:
        path = Path(explicit)
        return path if path.exists() else None
    found = shutil.which("pvpython")
    return Path(found) if found else None


def render(pvpython: Path, mode: str, source: Path, out: Path, extra: list[str]) -> Path:
    """Run one ParaView mode as a subprocess."""
    command = [
        str(pvpython),
        str(HERE / "paraview_figures.py"),
        "--mode",
        mode,
        "--source",
        str(source),
        "--out",
        str(out),
        *extra,
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise SystemExit(
            f"pvpython failed for mode {mode} (exit {result.returncode}):\n{result.stderr[-2000:]}"
        )
    return out


# --------------------------------------------------------------------------
# Cropping
# --------------------------------------------------------------------------


def crop_to_content(path: Path) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Trim the white margin, keeping a small border."""
    from PIL import Image

    image = Image.open(path).convert("RGB")
    pixels = np.asarray(image, dtype=np.int16)
    content = np.abs(255 - pixels).max(axis=2) > CROP_THRESHOLD
    if not content.any():
        return None

    rows = np.flatnonzero(content.any(axis=1))
    columns = np.flatnonzero(content.any(axis=0))
    top = max(int(rows[0]) - CROP_MARGIN, 0)
    bottom = min(int(rows[-1]) + CROP_MARGIN + 1, image.height)
    left = max(int(columns[0]) - CROP_MARGIN, 0)
    right = min(int(columns[-1]) + CROP_MARGIN + 1, image.width)

    before = image.size
    image.crop((left, top, right, bottom)).save(path)
    return before, (right - left, bottom - top)


def main() -> int:
    """Build the requested figures.

    :returns: ``0`` if at least one figure was produced, else ``1``.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "docs" / "figures",
        help="directory the figures are written to",
    )
    parser.add_argument("--mesh", type=Path, help="a domain.vtu for the geometry and mesh figures")
    parser.add_argument(
        "--field", type=Path, help="a result_domain.pvd from a 'direct' run, for the field figure"
    )
    parser.add_argument(
        "--time",
        type=float,
        default=6.3,
        help="time [a] the field is shown at; the nearest stored step is used",
    )
    parser.add_argument("--pvpython", help="path to ParaView's interpreter, if not on PATH")
    parser.add_argument(
        "--only",
        choices=("schema", "geometry", "mesh", "field"),
        nargs="+",
        help="build only these figures",
    )
    args = parser.parse_args()

    wanted = set(args.only or ("schema", "geometry", "mesh", "field"))
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    built: list[Path] = []

    if "schema" in wanted:
        built.append(draw_schematic(out / "schema.png"))

    rendered = wanted - {"schema"}
    if rendered:
        pvpython = find_pvpython(args.pvpython)
        if pvpython is None:
            print(
                "pvpython not found -- skipping the rendered figures. Install ParaView and "
                "pass --pvpython <path>, or put it on PATH.",
                file=sys.stderr,
            )
        else:
            for mode in ("geometry", "mesh"):
                if mode not in rendered:
                    continue
                if args.mesh is None:
                    print(f"--mesh is required for the {mode} figure; skipped", file=sys.stderr)
                    continue
                built.append(render(pvpython, mode, args.mesh, out / f"{mode}.png", []))
            if "field" in rendered:
                if args.field is None:
                    print("--field is required for the field figure; skipped", file=sys.stderr)
                else:
                    built.append(
                        render(
                            pvpython,
                            "field",
                            args.field,
                            out / "field.png",
                            ["--time", str(args.time)],
                        )
                    )

    for path in built:
        result = crop_to_content(path)
        if result is None:
            print(f"{path.name}: empty, not cropped")
        else:
            before, after = result
            print(f"{path.name}: {before[0]}x{before[1]} -> {after[0]}x{after[1]}")
    return 0 if built else 1


if __name__ == "__main__":
    raise SystemExit(main())
