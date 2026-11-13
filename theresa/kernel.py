# SPDX-License-Identifier: GPL-3.0-or-later
"""Step-response kernel from an OGS result series, in K/W.

``wall_max``
    Maximum over all nodes of the canister-wall submesh per time step; the
    exact peak of one design.
``probe``
    Response at a fixed grid in canister-local coordinates
    (:func:`probe_points`), comparable across designs; basis of the surrogate.

``probe_deficit`` is the largest relative shortfall of the probe maximum
against ``wall_max``. Probes are inset by ``probe_inset_m`` so that they lie
inside the faceted canister mesh.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import drift_axis_z, quarter_canister_volume


def probe_points(cfg: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Probe locations in global coordinates.

    Lateral surface from ``theta = 0`` (top) to ``pi`` (bottom) and from the
    mid-plane to the end face, plus three radii on the end face.

    :param cfg: a configuration from :mod:`theresa.config`.
    :returns: ``(points, labels)``; ``points`` of shape ``(n, 3)``, ``labels``
        the local coordinates ``theta``, ``axial``, ``radius``.
    """
    r_can = cfg["r_can"]
    half_length = cfg["l_can"] / 2.0
    inset = cfg["probe_inset_m"]
    z_axis = drift_axis_z(cfg)

    if inset >= r_can:
        raise ValueError("probe_inset_m must be smaller than the canister radius")

    theta = np.linspace(0.0, np.pi, int(cfg["n_probe_theta"]))
    axial = np.linspace(0.0, half_length - inset, int(cfg["n_probe_axial"]))

    coords: list[np.ndarray] = []
    local: list[np.ndarray] = []

    # Lateral surface.
    r_lateral = r_can - inset
    for x in axial:
        y = r_lateral * np.sin(theta)
        z = z_axis + r_lateral * np.cos(theta)
        coords.append(np.column_stack([np.full_like(theta, x), y, z]))
        local.append(
            np.column_stack([theta, np.full_like(theta, x), np.full_like(theta, r_lateral)])
        )

    # End face.
    x_end = half_length - inset
    for fraction in (0.3, 0.6, 0.9):
        r = r_can * fraction
        y = r * np.sin(theta)
        z = z_axis + r * np.cos(theta)
        coords.append(np.column_stack([np.full_like(theta, x_end), y, z]))
        local.append(np.column_stack([theta, np.full_like(theta, x_end), np.full_like(theta, r)]))

    points = np.vstack(coords)
    stacked = np.vstack(local)
    return points, {
        "theta": stacked[:, 0],
        "axial": stacked[:, 1],
        "radius": stacked[:, 2],
    }


def meshed_canister_volume(canister_mesh: Path) -> float:
    """Volume [m3] of the meshed quarter canister.

    Reference volume of the source density, see
    :func:`theresa.decay_heat.volumetric_density_per_watt`.
    """
    import pyvista as pv

    mesh = pv.read(str(canister_mesh))
    volume = float(mesh.compute_cell_sizes(length=False, area=False, volume=True)["Volume"].sum())
    if volume <= 0.0:
        raise RuntimeError(f"{canister_mesh} has non-positive volume")
    return volume


def faceting_ratio(cfg: dict, canister_mesh: Path) -> float:
    """Meshed over exact quarter-canister volume."""
    return meshed_canister_volume(canister_mesh) / quarter_canister_volume(cfg)


def extract_kernel(
    cfg: dict, out_dir: Path, canister_mesh: Path | None = None
) -> dict[str, np.ndarray | float]:
    """Read the step response from a finished ``kernel`` run.

    :param cfg: the configuration the run was made with.
    :param out_dir: the OGS output directory of the kernel run.
    :param canister_mesh: the ``canister.vtu`` subdomain mesh, for
        :func:`faceting_ratio`.
    :returns: a dictionary with

        ``times`` -- sample times [s], starting at 0;
        ``probe`` -- response [K/W] at the probe points, ``(n_times, n_probe)``;
        ``wall_max`` -- exact nodal maximum [K/W], ``(n_times,)``;
        ``probe_max`` -- maximum over the probe points, ``(n_times,)``;
        ``probe_deficit`` -- largest relative shortfall of ``probe_max``
        against ``wall_max``;
        ``faceting_ratio`` -- meshed over exact canister volume.
    :raises RuntimeError: if the expected result series are missing.
    """
    import ogstools as ot
    import pyvista as pv

    wall_pvd = _require(out_dir, "result_canister_wall.pvd")
    canister_pvd = _require(out_dir, "result_canister.pvd")

    wall_series = ot.MeshSeries(str(wall_pvd))
    canister_series = ot.MeshSeries(str(canister_pvd))

    times = np.asarray(wall_series.timevalues, dtype=float)
    wall_max = np.array(
        [
            float(wall_series.mesh(i, lazy_eval=False).point_data["temperature"].max())
            for i in range(len(times))
        ]
    )

    points, _labels = probe_points(cfg)
    probe_cloud = pv.PolyData(points)

    probe = np.empty((len(times), points.shape[0]))
    for i in range(len(times)):
        sampled = probe_cloud.sample(canister_series.mesh(i, lazy_eval=False))
        valid = np.asarray(sampled.point_data["vtkValidPointMask"], dtype=bool)
        if not valid.all():
            raise RuntimeError(
                f"{int((~valid).sum())} of {points.shape[0]} probe points fell outside the "
                "canister mesh -- increase probe_inset_m or refine mesh_size_fine"
            )
        probe[i] = np.asarray(sampled.point_data["temperature"], dtype=float)

    probe_max = probe.max(axis=1)
    # Relative comparison above 1 % of the maximum.
    significant = wall_max > 0.01 * wall_max.max()
    deficit = float(
        np.max((wall_max[significant] - probe_max[significant]) / wall_max[significant])
    )

    return {
        "times": times,
        "probe": probe,
        "wall_max": wall_max,
        "probe_max": probe_max,
        "probe_deficit": deficit,
        "faceting_ratio": (
            float("nan") if canister_mesh is None else faceting_ratio(cfg, canister_mesh)
        ),
    }


def _require(out_dir: Path, name: str) -> Path:
    """``out_dir / name``; error if missing."""
    path = out_dir / name
    if not path.exists():
        available = ", ".join(sorted(p.name for p in out_dir.glob("*.pvd"))) or "none"
        raise RuntimeError(f"{name} not found in {out_dir} (present: {available})")
    return path
