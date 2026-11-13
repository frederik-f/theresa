# SPDX-License-Identifier: GPL-3.0-or-later
"""Execution of OGS: the ``ogs`` executable, or the ``ogs.simulator`` bindings."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)


def _ogs_env() -> dict[str, str]:
    """Environment of the OGS subprocess; ``OMP_NUM_THREADS`` defaults to 1."""
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    return env


class SimulationCancelled(RuntimeError):
    """Raised when a run was stopped through its ``cancel`` event."""


#: Polling interval [s] for a cancel request.
CANCEL_POLL_SECONDS = 0.25


def find_ogs_executable() -> str | None:
    """Path of the ``ogs`` executable: on ``PATH``, else next to the interpreter.

    :returns: the path, or ``None``.
    """
    found = shutil.which("ogs")
    if found is not None:
        return found
    return shutil.which("ogs", path=str(Path(sys.executable).parent))


def run_ogs(
    prj_path: Path,
    out_dir: Path,
    on_line: Callable[[str], None] | None = None,
    cancel: threading.Event | None = None,
) -> None:
    """Run OGS on a project file.

    Uses the ``ogs`` executable (:func:`find_ogs_executable`), otherwise the
    ``ogs.simulator`` bindings.

    :param prj_path: project file from
        :func:`theresa.prj_builder.build_prj_xml`.
    :param out_dir: result directory; created if missing.
    :param on_line: called with every output line of OGS (executable only).
        Without it, OGS writes to standard output.
    :param cancel: terminates OGS and raises :class:`SimulationCancelled`
        when set; polled every :data:`CANCEL_POLL_SECONDS` (executable only).
    :raises RuntimeError: if OGS is not available or exits with an error.
    """
    # OGS runs in the project directory; all paths passed to it are absolute.
    prj_path = Path(prj_path).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    executable = find_ogs_executable()
    if executable is not None:
        cmd = [executable, str(prj_path), "-o", str(out_dir)]
        logger.info("[ogs] %s", " ".join(cmd))
        if on_line is None and cancel is None:
            returncode = subprocess.run(cmd, cwd=prj_path.parent, env=_ogs_env()).returncode
        else:
            with subprocess.Popen(
                cmd,
                cwd=prj_path.parent,
                env=_ogs_env(),
                stdout=None if on_line is None else subprocess.PIPE,
                stderr=None if on_line is None else subprocess.STDOUT,
                text=True,
                bufsize=1,  # line buffered
            ) as process:
                if cancel is not None:
                    threading.Thread(
                        target=_terminate_on, args=(process, cancel), daemon=True
                    ).start()
                if on_line is not None:
                    for line in process.stdout:
                        on_line(line.rstrip("\n"))
            returncode = process.returncode
            if cancel is not None and cancel.is_set():
                raise SimulationCancelled("the OGS6 simulation was cancelled")
        if returncode != 0:
            raise RuntimeError(
                f"The OGS6 simulation failed (exit code {returncode}; see output above)."
            )
        return

    try:
        import ogs.simulator as sim
    except ImportError as exc:
        raise RuntimeError(
            "Neither the 'ogs' command-line executable nor the 'ogs' "
            "Python package (ogs.simulator) was found. Install one via "
            "'pip install ogs', or install the OGS6 binary and add it to "
            "the PATH."
        ) from exc

    if cancel is not None:
        raise RuntimeError("cancelling needs the ogs executable, not the Python bindings")
    logger.info("[ogs.simulator] starting simulation via Python bindings")
    sim.initialize(["ogs", str(prj_path), "-o", str(out_dir)])
    sim.executeSimulation()
    sim.finalize()


def _terminate_on(process: subprocess.Popen, cancel: threading.Event) -> None:
    """Terminate ``process`` once ``cancel`` is set; return when it has ended."""
    while process.poll() is None:
        if cancel.wait(CANCEL_POLL_SECONDS):
            process.terminate()
            try:
                process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                process.kill()
            return
