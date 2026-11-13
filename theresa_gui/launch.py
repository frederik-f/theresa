# SPDX-License-Identifier: GPL-3.0-or-later
"""Entry point ``theresa-gui``: ``streamlit run`` on :mod:`theresa_gui.app`."""

from __future__ import annotations

import sys
from pathlib import Path

APP = Path(__file__).resolve().parent / "app.py"


def main(argv: list[str] | None = None) -> int:
    """Run ``streamlit run app.py`` with the given arguments."""
    try:
        from streamlit.web import cli as streamlit_cli
    except ImportError as error:  # pragma: no cover - depends on the install
        raise SystemExit(
            "The graphical front end needs Streamlit. Install it with:\n    uv sync --extra gui"
        ) from error

    arguments = ["streamlit", "run", str(APP), *(argv if argv is not None else sys.argv[1:])]
    sys.argv = arguments
    return int(streamlit_cli.main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
