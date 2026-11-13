# SPDX-License-Identifier: GPL-3.0-or-later
"""Sphinx configuration of the THERESA documentation.

Build with::

    uv sync --extra docs
    uv run sphinx-build -b html docs docs/_build/html
"""

from __future__ import annotations

import sys
from pathlib import Path

# Package importable from the checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

project = "THERESA"
author = "frederik-f"
release = "1.0.0"
copyright = "2026 frederik-f, CC BY 4.0"  # noqa: A001 - Sphinx's own name

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "myst_parser",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

# Native and heavy dependencies are mocked.
autodoc_mock_imports = [
    "gmsh",
    "ogs",
    "ogstools",
    "pyvista",
    "matplotlib",
    "streamlit",
    "sklearn",
    "scipy",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}

autodoc_typehints = "description"

myst_enable_extensions = ["dollarmath", "colon_fence"]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "alabaster"
html_static_path = []
html_title = "THERESA"
html_logo = "assets/icon.svg"
html_favicon = "assets/favicon.png"

nitpicky = False
