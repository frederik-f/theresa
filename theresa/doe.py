# SPDX-License-Identifier: GPL-3.0-or-later
"""Design of experiments: training and validation points, batch runs.

Points are stratified by a Latin hypercube in the fitting coordinate of
:mod:`theresa.surrogate` (``1 / x`` for spacings and conductivities).
Validation points use a different seed and an inner margin. The canister
spacing is the clear distance, so the design space is rectangular.

Depth enters through ``psi``; :func:`depth_study_points` provides the runs
that quantify the remaining geometric effect.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import MATERIAL_NAMES, SECONDS_PER_YEAR, make_config
from .pipeline import run_case

logger = logging.getLogger(__name__)

#: Ranges of the design variables: geometry [m], thermal conductivity
#: [W/(m K)], volumetric heat capacity ``rho * cp`` [J/(m3 K)].
DEFAULT_RANGES: dict[str, tuple[float, float]] = {
    "canister_gap": (2.5, 15.0),
    "a_drift": (15.0, 60.0),
    "l_can": (4.0, 7.0),
    "depth_m": (300.0, 1000.0),
    # Effective canister value including the interior.
    "canister_lambda": (4.0, 60.0),
    "canister_rho_cp": (1.8e6, 4.0e6),
    # Compacted bentonite, dry to saturated.
    "backfill_lambda": (0.3, 1.8),
    "backfill_rho_cp": (0.9e6, 2.6e6),
    # Concrete segments.
    "lining_lambda": (0.8, 2.8),
    "lining_rho_cp": (1.6e6, 2.8e6),
    # Claystone.
    "host_rock_lambda": (1.0, 3.2),
    "host_rock_rho_cp": (1.6e6, 3.0e6),
}

#: Design variables sampled and fitted as ``1 / x``: spacings, canister length
#: and conductivities.
RECIPROCAL_FEATURES = {
    "canister_gap": True,
    "a_drift": True,
    "l_can": True,
    "depth_m": False,
    **{f"{name}_lambda": True for name in MATERIAL_NAMES},
    **{f"{name}_rho_cp": False for name in MATERIAL_NAMES},
}

#: Interpolated design variables of the shipped surrogate.
TRAINING_FEATURES: tuple[str, ...] = (
    "canister_gap",
    "a_drift",
    *(f"{name}_{prop}" for name in MATERIAL_NAMES for prop in ("lambda", "rho_cp")),
)


@dataclass
class TrainingSet:
    """Simulated kernels for a set of design points.

    :param times: shared time grid [s].
    :param feature_names: the design variables that vary.
    :param points: ``(n_designs, n_features)`` design variables.
    :param kernels: ``(n_designs, n_times, n_probe)`` step responses [K/W].
    :param psis: ``(n_designs, n_probe)`` ``psi`` fields [m].
    :param probe_deficits: per design, shortfall of the probe grid against the
        wall maximum.
    """

    times: np.ndarray
    feature_names: tuple[str, ...]
    points: np.ndarray
    kernels: np.ndarray
    psis: np.ndarray
    probe_deficits: np.ndarray

    def save(self, path: Path) -> Path:
        """Write the training set to a compressed ``.npz``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            times=self.times,
            feature_names=np.array(self.feature_names),
            points=self.points,
            kernels=self.kernels,
            psis=self.psis,
            probe_deficits=self.probe_deficits,
        )
        return path

    @staticmethod
    def load(path: Path) -> TrainingSet:
        """Read a training set written by :meth:`save`."""
        with np.load(Path(path)) as data:
            return TrainingSet(
                times=data["times"],
                feature_names=tuple(str(name) for name in data["feature_names"]),
                points=data["points"],
                kernels=data["kernels"],
                psis=data["psis"],
                probe_deficits=data["probe_deficits"],
            )


def latin_hypercube(
    feature_names: Sequence[str],
    n_points: int,
    ranges: dict[str, tuple[float, float]] | None = None,
    seed: int = 0,
    margin: float = 0.05,
) -> np.ndarray:
    """Design points stratified in the fitting coordinate.

    :param feature_names: the varying design variables.
    :param n_points: number of points; also the number of strata per variable.
    :param ranges: overrides for :data:`DEFAULT_RANGES`.
    :param seed: seed of the permutations and the jitter within each stratum.
    :param margin: fraction of the range kept clear at each end (zero for
        training, positive for validation).
    :returns: ``(n_points, n_features)``.
    """
    ranges = {**DEFAULT_RANGES, **(ranges or {})}
    rng = np.random.default_rng(seed)

    columns = []
    for name in feature_names:
        low, high = ranges[name]
        strata = (rng.permutation(n_points) + rng.random(n_points)) / n_points
        strata = margin + (1.0 - 2.0 * margin) * strata
        if RECIPROCAL_FEATURES.get(name, True):
            columns.append(1.0 / (1.0 / low + strata * (1.0 / high - 1.0 / low)))
        else:
            columns.append(low + strata * (high - low))
    return np.column_stack(columns)


def depth_study_points(
    depths: Sequence[float] = (300.0, 500.0, 700.0, 1000.0),
    spacings: Sequence[tuple[float, float]] = ((4.0, 20.0), (9.5, 40.0), (15.0, 60.0)),
) -> np.ndarray:
    """Design points for the geometric effect of depth: fixed spacings, several depths.

    :param spacings: ``(canister_gap, a_drift)`` pairs [m].
    :returns: ``(len(depths) * len(spacings), 3)`` in
        ``canister_gap, a_drift, depth_m`` order.
    """
    rows = [(gap, a_drift, depth) for depth in depths for gap, a_drift in spacings]
    return np.array(rows, dtype=float)


def run_batch(
    points: np.ndarray,
    feature_names: Sequence[str],
    root: Path,
    base_config: dict | None = None,
    *,
    max_workers: int = 3,
    keep_results: bool = False,
    on_done: Callable[[int, int, dict], None] | None = None,
    checkpoint: Path | None = None,
    checkpoint_every: int = 50,
) -> TrainingSet:
    """Simulate all design points concurrently and collect the kernels.

    :param points: ``(n_designs, n_features)`` design variables.
    :param feature_names: names matching the columns of ``points``.
    :param root: directory the per-design run directories are created in.
    :param base_config: configuration to override the design variables on.
    :param max_workers: number of concurrent OGS runs.
    :param keep_results: keep the OGS result series, see
        :func:`theresa.pipeline.run_case`.
    :param on_done: called as ``(index, total, info)`` after each design point.
    :param checkpoint: file for the finished design points, written every
        ``checkpoint_every`` completions as a valid :class:`TrainingSet`.
    :param checkpoint_every: completions between checkpoints.
    :raises RuntimeError: if a design point fails.
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    feature_names = tuple(feature_names)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    total = points.shape[0]
    results: list[dict | None] = [None] * total
    lock = threading.Lock()
    completed = 0

    def one(index: int) -> None:
        nonlocal completed
        overrides = {
            name: float(value) for name, value in zip(feature_names, points[index], strict=True)
        }
        cfg = make_config(base=base_config, **overrides)
        work_dir = root / _design_directory_name(overrides, index)

        result = run_case(cfg, work_dir, keep_results=keep_results)
        info = {
            "kernel": result.kernel_probe,
            "psi": result.psi_probe,
            "times": result.times_years,
            "probe_deficit": result.probe_deficit,
            "peak_celsius": result.peak_celsius,
            "n_nodes": result.mesh_stats.get("n_nodes"),
        }
        with lock:
            results[index] = info
            completed += 1
            logger.info(
                "[doe] %d/%d %s peak %.1f degC", completed, total, overrides, result.peak_celsius
            )
            if on_done is not None:
                on_done(
                    completed,
                    total,
                    {**overrides, **{k: info[k] for k in ("peak_celsius", "n_nodes")}},
                )
            if checkpoint is not None and completed % checkpoint_every == 0:
                done = [i for i, entry in enumerate(results) if entry is not None]
                _training_set(done, results, points, feature_names).save(checkpoint)
                logger.info("[doe] checkpoint at %d/%d -> %s", completed, total, checkpoint)

    # Failures are logged as soon as they occur.
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(one, index): index for index in range(total)}
        for future in as_completed(futures):
            index = futures[future]
            try:
                future.result()
            except Exception as error:  # noqa: BLE001 - reported, then re-raised below
                design = dict(zip(feature_names, points[index], strict=True))
                failures.append(f"{design}: {type(error).__name__}: {error}")
                logger.error("[doe] design %d %s FAILED: %s", index, design, error)

    if failures:
        raise RuntimeError(
            f"{len(failures)} of {total} design points failed:\n  " + "\n  ".join(failures)
        )

    done = [index for index, entry in enumerate(results) if entry is not None]
    if len(done) != total:
        raise RuntimeError(f"only {len(done)} of {total} design points produced a kernel")
    return _training_set(done, results, points, feature_names)


def _training_set(
    indices: Sequence[int],
    results: Sequence[dict | None],
    points: np.ndarray,
    feature_names: tuple[str, ...],
) -> TrainingSet:
    """:class:`TrainingSet` from a subset of finished design points."""
    finished = [results[index] for index in indices]
    times = np.asarray(finished[0]["times"], dtype=float) * SECONDS_PER_YEAR
    for entry in finished[1:]:
        if not np.allclose(np.asarray(entry["times"]) * SECONDS_PER_YEAR, times):
            raise RuntimeError(
                "design points produced different time grids -- the schedule must be "
                "identical across the training set for the kernels to be comparable"
            )

    return TrainingSet(
        times=times,
        feature_names=feature_names,
        points=points[list(indices)],
        kernels=np.stack([entry["kernel"] for entry in finished]),
        psis=np.stack([entry["psi"] for entry in finished]),
        probe_deficits=np.array([entry["probe_deficit"] for entry in finished]),
    )


#: Longest readable run-directory name; longer names use a digest.
MAX_READABLE_DIRECTORY_NAME = 80


def _design_directory_name(overrides: dict[str, float], index: int) -> str:
    """Unique run-directory name of a design point."""
    parts = [f"{name}{value:g}".replace(".", "p") for name, value in sorted(overrides.items())]
    spelled = "_".join(parts)
    if len(spelled) <= MAX_READABLE_DIRECTORY_NAME:
        return f"{index:03d}_{spelled}"
    digest = hashlib.sha1(spelled.encode(), usedforsecurity=False).hexdigest()[:10]
    return f"{index:04d}_{digest}"
