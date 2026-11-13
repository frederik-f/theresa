#!/usr/bin/env bash
# Run the test suite from the project environment.
#
# pre-commit runs its hooks in isolated environments, which cannot resolve the
# native gmsh and OGS6 wheels this package needs. This script therefore picks
# the active virtualenv, falling back to ./.venv (what `uv sync` creates).
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python="${VIRTUAL_ENV:-$root/.venv}/bin/python"

if [[ ! -x "$python" ]]; then
    echo "No project environment found at $python -- run 'uv sync --extra test'." >&2
    exit 1
fi

exec "$python" -m pytest "$@"
