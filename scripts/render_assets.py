# SPDX-License-Identifier: GPL-3.0-or-later
"""Render icon, banner, word marks and social preview from the SVG sources.

Sources: ``docs/assets/src`` (SVG, Lato). Outputs:

``docs/assets/banner.svg``
    README banner, text as paths.
``docs/assets/icon.svg``, ``icon-512.png``, ``favicon.png``
    icon as vector, raster and favicon.
``docs/assets/social-preview.png``
    1280 x 640 repository preview.
``theresa_gui/assets/icon.png``, ``wordmark.png``, ``wordmark-dark.png``
    GUI page icon and sidebar logos (light and dark theme).

Run::

    uv run python scripts/render_assets.py

Requires ``inkscape`` on the PATH and the Lato typeface.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ROOT / "docs" / "assets" / "src"
ASSETS = ROOT / "docs" / "assets"
GUI_ASSETS = ROOT / "theresa_gui" / "assets"


def inkscape(source: Path, target: Path, *options: str) -> Path:
    """Export ``source`` to ``target`` with the given Inkscape options."""
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["inkscape", str(source), *options, f"--export-filename={target}"],
        check=True,
        capture_output=True,
    )
    return target


def main() -> int:
    """Regenerate every published asset from the sources."""
    if shutil.which("inkscape") is None:
        raise SystemExit("inkscape is needed to render the assets")

    icon, banner, social, wordmark, wordmark_dark = (
        SOURCES / name
        for name in ("icon.svg", "banner.svg", "social.svg", "wordmark.svg", "wordmark-dark.svg")
    )
    written = [
        inkscape(banner, ASSETS / "banner.svg", "--export-text-to-path", "--export-plain-svg"),
        inkscape(icon, ASSETS / "icon.svg", "--export-plain-svg"),
        inkscape(icon, ASSETS / "icon-512.png", "--export-type=png", "--export-width=512"),
        inkscape(icon, ASSETS / "favicon.png", "--export-type=png", "--export-width=64"),
        inkscape(social, ASSETS / "social-preview.png", "--export-type=png", "--export-width=1280"),
        inkscape(icon, GUI_ASSETS / "icon.png", "--export-type=png", "--export-width=128"),
        inkscape(wordmark, GUI_ASSETS / "wordmark.png", "--export-type=png", "--export-width=640"),
        inkscape(
            wordmark_dark,
            GUI_ASSETS / "wordmark-dark.png",
            "--export-type=png",
            "--export-width=640",
        ),
    ]
    for path in written:
        print(f"-> {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
