# SPDX-License-Identifier: GPL-3.0-or-later
"""OGS invocation: command line and working directory (OGS mocked)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from theresa import runner


class _Recorder:
    """Stands in for ``subprocess.run``, keeping the command and cwd."""

    def __init__(self, returncode: int = 0) -> None:
        self.returncode = returncode
        self.cmd: list[str] = []
        self.cwd: Path | None = None

    def __call__(self, cmd, cwd=None, env=None):  # noqa: ARG002 - signature match
        self.cmd = list(cmd)
        self.cwd = Path(cwd)
        return subprocess.CompletedProcess(cmd, self.returncode)


@pytest.fixture
def fake_ogs(monkeypatch, tmp_path):
    recorder = _Recorder()
    monkeypatch.setattr(runner, "find_ogs_executable", lambda: str(tmp_path / "ogs"))
    monkeypatch.setattr(runner.subprocess, "run", recorder)
    return recorder


def test_relative_paths_are_resolved_before_ogs_sees_them(fake_ogs, tmp_path, monkeypatch):
    # Paths passed to OGS are absolute.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "case.prj").write_text("<OpenGeoSysProject/>")

    runner.run_ogs(Path("run/case.prj"), Path("run/out"))

    project, out_dir = fake_ogs.cmd[1], fake_ogs.cmd[3]
    assert Path(project).is_absolute()
    assert Path(out_dir).is_absolute()
    assert Path(project) == (tmp_path / "run" / "case.prj").resolve()
    assert Path(out_dir) == (tmp_path / "run" / "out").resolve()


def test_the_output_directory_exists_before_ogs_starts(fake_ogs, tmp_path, monkeypatch):
    # The output directory exists before OGS starts.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "case.prj").write_text("<OpenGeoSysProject/>")

    runner.run_ogs(Path("run/case.prj"), Path("run/deep/out"))

    assert (tmp_path / "run" / "deep" / "out").is_dir()


def test_ogs_runs_in_the_project_directory(fake_ogs, tmp_path):
    # OGS runs in the project directory.
    project = tmp_path / "run" / "case.prj"
    project.parent.mkdir()
    project.write_text("<OpenGeoSysProject/>")

    runner.run_ogs(project, tmp_path / "out")

    assert fake_ogs.cwd == project.parent


def test_a_failing_simulation_raises(fake_ogs, tmp_path):
    fake_ogs.returncode = 1
    project = tmp_path / "case.prj"
    project.write_text("<OpenGeoSysProject/>")

    with pytest.raises(RuntimeError, match="simulation failed"):
        runner.run_ogs(project, tmp_path / "out")


def test_each_output_line_reaches_the_observer(tmp_path, monkeypatch):
    """``on_line`` receives the output line by line; every time step is recognised."""
    from theresa.pipeline import TIME_STEP_LINE

    fake = tmp_path / "ogs"
    fake.write_text(
        "#!/bin/sh\n"
        'for n in 0 1 2; do echo "info: Time step #$n started. Time: $n. Step size: 1."; done\n'
    )
    fake.chmod(0o755)
    monkeypatch.setattr(runner, "find_ogs_executable", lambda: str(fake))
    project = tmp_path / "case.prj"
    project.write_text("<OpenGeoSysProject/>")

    lines: list[str] = []
    runner.run_ogs(project, tmp_path / "out", on_line=lines.append)

    steps = [int(TIME_STEP_LINE.search(line).group(1)) for line in lines]
    assert steps == [0, 1, 2]


def test_a_cancelled_run_stops_promptly(tmp_path, monkeypatch):
    """Setting the event ends a silent, long-running OGS within a second."""
    import threading
    import time

    fake = tmp_path / "ogs"
    fake.write_text("#!/bin/sh\nexec sleep 30\n")
    fake.chmod(0o755)
    monkeypatch.setattr(runner, "find_ogs_executable", lambda: str(fake))
    project = tmp_path / "case.prj"
    project.write_text("<OpenGeoSysProject/>")

    cancel = threading.Event()
    threading.Timer(0.3, cancel.set).start()
    started = time.monotonic()
    with pytest.raises(runner.SimulationCancelled):
        runner.run_ogs(project, tmp_path / "out", cancel=cancel)
    assert time.monotonic() - started < 3.0


def test_a_run_cancelled_beforehand_does_not_start_meshing(tmp_path, monkeypatch):
    import threading

    from theresa import pipeline
    from theresa.config import make_config

    def no_meshing(*args, **kwargs):
        raise AssertionError("meshing started although the run was cancelled")

    monkeypatch.setattr(pipeline, "build_geometry", no_meshing)
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(runner.SimulationCancelled):
        pipeline.run_case(make_config(), tmp_path / "run", cancel=cancel)
