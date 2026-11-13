# SPDX-License-Identifier: GPL-3.0-or-later
"""Entry point for ``python -m theresa``; equivalent to the ``theresa`` command."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
