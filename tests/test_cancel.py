# SPDX-License-Identifier: GPL-3.0-or-later
"""Cancellation of a direct run in every stage, meshing included (slow)."""

from __future__ import annotations

import multiprocessing
import threading
import time

import pytest

pytest.importorskip("gmsh")
pytest.importorskip("ogstools")

from theresa import pipeline  # noqa: E402
from theresa.config import make_config  # noqa: E402
from theresa.runner import SimulationCancelled  # noqa: E402

COARSE = {"mesh_size_fine": 0.5, "mesh_size_coarse": 40.0, "mesh_dist_max": 150.0}


def test_meshing_in_a_child_gives_the_same_mesh(tmp_path):
    cfg = make_config(**COARSE)
    direct_stats, direct_meshes = pipeline._mesh(cfg, str(tmp_path / "direct"))
    steps: list[str] = []
    child_stats, child_meshes = pipeline._mesh_cancellable(
        cfg, tmp_path / "child", threading.Event(), steps.append
    )
    assert steps == list(pipeline.MESH_STEPS)
    assert child_stats["n_nodes"] == direct_stats["n_nodes"]
    assert set(child_meshes) == set(direct_meshes)
    assert all(path.exists() for path in child_meshes.values())


def test_a_run_stops_promptly_while_meshing(tmp_path):
    # Stop during meshing.
    cancel = threading.Event()
    outcome: list[BaseException] = []

    def run() -> None:
        try:
            pipeline.run_case(make_config(), tmp_path / "run", cancel=cancel)
        except BaseException as error:  # noqa: BLE001 - inspected below
            outcome.append(error)

    worker = threading.Thread(target=run)
    worker.start()
    time.sleep(4.0)
    cancel.set()
    stopped = time.monotonic()
    worker.join(timeout=30.0)

    assert not worker.is_alive()
    assert time.monotonic() - stopped < 5.0
    assert len(outcome) == 1 and isinstance(outcome[0], SimulationCancelled)
    assert multiprocessing.active_children() == []
