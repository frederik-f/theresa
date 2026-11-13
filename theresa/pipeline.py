# SPDX-License-Identifier: GPL-3.0-or-later
"""Simulation of one design point: mesh, steady and transient solve, assembly.

::

    T(x, t) = T_top + g * psi(x)          <- steady solve
            + Duhamel(S(x, .), Q)         <- transient solve (step response)
"""

from __future__ import annotations

import multiprocessing
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import (
    KELVIN_OFFSET,
    SECONDS_PER_YEAR,
    canister_pitch,
    domain_bottom_z,
    prepare_run_directory,
    write_config,
)
from .decay_heat import source_terms_watt
from .duhamel import convolve
from .geometry import build_geometry
from .kernel import extract_kernel, meshed_canister_volume, probe_points
from .mesh_conversion import convert_mesh
from .prj_builder import build_prj_xml, time_step_count
from .runner import CANCEL_POLL_SECONDS, SimulationCancelled, run_ogs
from .steady_state import PSI_FIELD_NAME, compute_psi


@dataclass
class CaseResult:
    """Result of one design point.

    :param times_years: sample times [a] since emplacement.
    :param temperature_celsius: peak canister-surface temperature [degC] over
        time, taken over the probe grid.
    :param rise_kelvin: temperature rise from heating [K].
    :param kernel_probe: step response [K/W] at the probe points,
        ``(n_times, n_probe)``.
    :param psi_probe: ``psi`` [m] at the probe points.
    :param probe_deficit: relative shortfall of the probe grid against the
        wall maximum.
    :param mesh_stats: node/element counts and region volumes.
    """

    times_years: np.ndarray
    temperature_celsius: np.ndarray
    rise_kelvin: np.ndarray
    kernel_probe: np.ndarray
    psi_probe: np.ndarray
    probe_deficit: float
    mesh_stats: dict = field(default_factory=dict)

    @property
    def peak_celsius(self) -> float:
        """Maximum canister-surface temperature [degC]."""
        return float(np.max(self.temperature_celsius))

    @property
    def peak_time_years(self) -> float:
        """Time [a] at which the maximum occurs."""
        return float(self.times_years[int(np.argmax(self.temperature_celsius))])


#: Stages of :func:`run_case`, as reported to ``progress``.
STAGES = ("mesh", "steady", "transient", "extract")

#: Steps of the meshing stage, as reported to ``progress``.
MESH_STEPS = ("geometry", "curves", "surfaces", "volume", "conversion")

#: OGS output line at the start of a time step.
TIME_STEP_LINE = re.compile(r"Time step #(\d+) started")

#: ``progress(stage, done, total)``: ``stage`` is one of :data:`STAGES`, and
#: ``done`` of ``total`` units of it are finished.
Progress = Callable[[str, int, int], None]


def run_case(
    cfg: dict,
    work_dir: Path,
    *,
    keep_results: bool = False,
    progress: Progress | None = None,
    cancel: threading.Event | None = None,
) -> CaseResult:
    """Run one design point end to end.

    :param cfg: a configuration from :mod:`theresa.config`.
    :param work_dir: run directory; stale artifacts in it are removed first.
    :param keep_results: keep the OGS result series of the transient run
        (~70 MB).
    :param progress: called at the start and end of every stage and at every
        time step (see :data:`Progress`), on the calling thread.
    :param cancel: stops the run with
        :class:`~theresa.runner.SimulationCancelled` when set. Meshing then
        runs in a child process.
    :raises SimulationCancelled: if ``cancel`` was set.
    """
    notify: Progress = progress or (lambda stage, done, total: None)

    def report(stage: str, done: int, total: int) -> None:
        if cancel is not None and cancel.is_set():
            raise SimulationCancelled("the run was cancelled")
        notify(stage, done, total)

    work_dir = prepare_run_directory(Path(work_dir))
    write_config(cfg, work_dir)

    def mesh_step(step: str) -> None:
        report("mesh", MESH_STEPS.index(step), len(MESH_STEPS))

    report("mesh", 0, len(MESH_STEPS))
    if cancel is None:
        mesh_stats, meshes = _mesh(cfg, str(work_dir), mesh_step)
    else:
        mesh_stats, meshes = _mesh_cancellable(cfg, work_dir, cancel, mesh_step)

    report("steady", 0, 1)
    compute_psi(cfg, work_dir, meshes, cancel=cancel)

    canister_volume = meshed_canister_volume(meshes["canister"])

    prj_path = work_dir / "kernel.prj"
    prj_path.write_text(
        build_prj_xml(
            cfg,
            meshes,
            prj_dir=work_dir,
            mode="kernel",
            meshed_canister_volume_m3=canister_volume,
        ),
        encoding="ISO-8859-1",
    )
    kernel_dir = work_dir / "kernel"
    n_steps = time_step_count(cfg)
    report("transient", 0, n_steps)

    def on_line(line: str) -> None:
        match = TIME_STEP_LINE.search(line)
        if match:
            report("transient", min(int(match.group(1)), n_steps), n_steps)

    run_ogs(prj_path, kernel_dir, on_line=on_line if progress is not None else None, cancel=cancel)

    report("extract", 0, 1)
    kernel = extract_kernel(cfg, kernel_dir, canister_mesh=meshes["canister"])
    psi_probe = psi_at_probe_points(cfg, meshes)

    result = assemble(cfg, kernel, psi_probe)
    result.mesh_stats = mesh_stats
    report("extract", 1, 1)

    if not keep_results:
        for stale in kernel_dir.glob("*.vtu"):
            stale.unlink()

    return result


def _mesh(
    cfg: dict, work_dir: str, on_step: Callable[[str], None] | None = None
) -> tuple[dict, dict[str, Path]]:
    """Geometry, mesh and conversion to VTU.

    :param on_step: called with each entry of :data:`MESH_STEPS` as it begins.
    """
    step = on_step or (lambda name: None)
    path = Path(work_dir)
    stats = build_geometry(cfg, path / "mesh.msh", step)
    step("conversion")
    return stats, convert_mesh(path / "mesh.msh", path / "mesh")


def _mesh_and_send(cfg: dict, work_dir: str, connection) -> None:
    """:func:`_mesh` in a child process; steps and result go to ``connection``."""
    try:
        result = _mesh(cfg, work_dir, lambda step: connection.send(("step", step)))
        connection.send(("ok", result))
    except BaseException as error:  # noqa: BLE001 - reported to the parent
        connection.send(("error", f"{type(error).__name__}: {error}"))
    finally:
        connection.close()


def _mesh_cancellable(
    cfg: dict,
    work_dir: Path,
    cancel: threading.Event,
    on_step: Callable[[str], None] | None = None,
) -> tuple[dict, dict[str, Path]]:
    """:func:`_mesh` in a spawned child process, terminated if ``cancel`` is set.

    :param on_step: called on the calling thread with each reported step.
    :raises SimulationCancelled: if ``cancel`` was set before meshing ended.
    :raises RuntimeError: if meshing failed in the child.
    """
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_mesh_and_send, args=(cfg, str(work_dir), sender), daemon=True)
    process.start()
    sender.close()
    status = payload = None
    try:
        while status is None:
            if cancel.is_set():
                raise SimulationCancelled("the run was cancelled while meshing")
            if receiver.poll(CANCEL_POLL_SECONDS):
                try:
                    kind, value = receiver.recv()
                except EOFError:
                    status, payload = "error", f"the process ended with code {process.exitcode}"
                    break
                if kind == "step":
                    if on_step is not None:
                        on_step(value)
                else:
                    status, payload = kind, value
            elif not process.is_alive():
                status, payload = "error", f"the process ended with code {process.exitcode}"
    finally:
        if process.is_alive() and cancel.is_set():
            process.terminate()
        process.join(timeout=10.0)
        if process.is_alive():
            process.kill()
            process.join()
        receiver.close()
    if status != "ok":
        raise RuntimeError(f"meshing failed: {payload}")
    return payload


@dataclass
class History:
    """Temperature history per time and probe point.

    :param times_years: times [a] since emplacement.
    :param baseline_probe: undisturbed temperature [degC] at each probe point.
    :param rise_probe: heating [K] at each probe point and time.
    :param canister_power_watt: canister power at ``t = 0`` [W] of
        ``rise_probe``.
    """

    times_years: np.ndarray
    baseline_probe: np.ndarray
    rise_probe: np.ndarray
    canister_power_watt: float

    @property
    def temperature_celsius(self) -> np.ndarray:
        """Peak canister-surface temperature [degC] per time step."""
        return (self.baseline_probe[None, :] + self.rise_probe).max(axis=1)

    @property
    def rise_kelvin(self) -> np.ndarray:
        """Peak heating [K] per time step."""
        return self.rise_probe.max(axis=1)

    def max_admissible_power_watt(self, limit_celsius: float) -> float:
        """Largest canister power [W] that keeps every point below the limit.

        ``min over (t, p) of (limit - baseline[p]) / rise[t, p]``, times the
        power. ``inf`` without heating, ``0`` if the undisturbed rock reaches
        the limit.
        """
        headroom = limit_celsius - self.baseline_probe
        if np.any(headroom <= 0.0):
            return 0.0

        positive = self.rise_probe > 0.0
        if not positive.any():
            return float("inf")

        scales = np.where(
            positive, headroom[None, :] / np.where(positive, self.rise_probe, 1.0), np.inf
        )
        return float(self.canister_power_watt * scales.min())


def temperature_history(
    cfg: dict,
    times_seconds: np.ndarray,
    kernel_probe: np.ndarray,
    psi_probe: np.ndarray,
) -> History:
    """Temperature history from a step response and ``psi``.

    Used for simulated and interpolated kernels alike. The maximum over the
    probe points is taken after the convolution, per time step.
    """
    times = np.asarray(times_seconds, dtype=float)
    probe = np.asarray(kernel_probe, dtype=float)

    amplitudes, decay_constants = source_terms_watt(cfg)
    rise = convolve(times, probe, amplitudes, decay_constants)  # (n_times, n_probe)

    baseline = cfg["surface_temperature_celsius"] + cfg["geothermal_gradient_k_per_m"] * np.asarray(
        psi_probe
    )
    return History(
        times_years=times / SECONDS_PER_YEAR,
        baseline_probe=baseline,
        rise_probe=rise,
        canister_power_watt=float(np.sum(amplitudes)),
    )


def assemble(cfg: dict, kernel: dict, psi_probe: np.ndarray) -> CaseResult:
    """:class:`CaseResult` from a simulated kernel and ``psi``.

    :param cfg: configuration (decay heat, boundary temperatures).
    :param kernel: from :func:`theresa.kernel.extract_kernel`.
    :param psi_probe: ``psi`` [m] at the probe points.
    """
    times = np.asarray(kernel["times"], dtype=float)
    probe = np.asarray(kernel["probe"], dtype=float)

    history = temperature_history(cfg, times, probe, psi_probe)
    return CaseResult(
        times_years=history.times_years,
        temperature_celsius=history.temperature_celsius,
        rise_kelvin=history.rise_kelvin,
        kernel_probe=probe,
        psi_probe=psi_probe,
        probe_deficit=float(kernel["probe_deficit"]),
    )


def psi_at_probe_points(cfg: dict, meshes: dict[str, Path]) -> np.ndarray:
    """``psi`` [m] at the probe points."""
    import pyvista as pv

    domain = pv.read(str(meshes["domain"]))
    if PSI_FIELD_NAME not in domain.point_data:
        raise RuntimeError(f"{PSI_FIELD_NAME!r} is not on the domain mesh -- run compute_psi first")

    points, _ = probe_points(cfg)
    sampled = pv.PolyData(points).sample(domain)
    valid = np.asarray(sampled.point_data["vtkValidPointMask"], dtype=bool)
    if not valid.all():
        raise RuntimeError("probe points fell outside the domain mesh while sampling psi")
    return np.asarray(sampled.point_data[PSI_FIELD_NAME], dtype=float)


def case_subtitle(cfg: dict) -> str:
    """Description of the design point, for the result plot."""
    return (
        f"clear canister spacing {cfg['canister_gap']:g} m "
        f"(pitch {canister_pitch(cfg):g} m)  ·  "
        f"drift spacing {cfg['a_drift']:g} m  ·  "
        f"length {cfg['l_can']:g} m  ·  depth {cfg['depth_m']:g} m\n"
        f"{cfg['n_fuel_elements']} fuel elements  ·  "
        f"{cfg['interim_storage_years']:g} a interim storage"
    )


def domain_height(cfg: dict) -> float:
    """Total vertical extent [m] of the model."""
    return -domain_bottom_z(cfg)


def boundary_temperatures_kelvin(cfg: dict) -> tuple[float, float]:
    """Top and bottom Dirichlet temperatures [K] for a ``direct`` run."""
    top = cfg["surface_temperature_celsius"] + KELVIN_OFFSET
    return top, top + cfg["geothermal_gradient_k_per_m"] * domain_height(cfg)
