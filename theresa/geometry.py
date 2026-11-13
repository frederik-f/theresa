# SPDX-License-Identifier: GPL-3.0-or-later
r"""Geometry and mesh of the quarter cell (Gmsh, OpenCASCADE kernel).

One quarter of the unit cell around a canister, bounded by four symmetry
planes; coordinates as in :mod:`theresa.config`::

      z = 0        top boundary (Dirichlet)
        |
        |     y = a_drift/2 .................. symmetry
        |          |
        |          |   host rock
        |          |     +---- liner (annulus at the drift wall)
        |          |     |  +- backfill
        |          |     |  |  +- canister (quarter cylinder)
        |          |     |  |  |
      z = -depth --+-----+--+--+---------- drift axis, along x
        |          |
        |     y = 0 ....................... symmetry (plane through the axis)
        |
      z = -(depth + h_below)   bottom boundary (Dirichlet)

      x = 0 ................................ symmetry (canister mid-length)
      x = pitch/2 ......................... symmetry (mid-way to the neighbour)

Material regions are identified from the entity map of
``gmsh.model.occ.fragment``.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable
from pathlib import Path

from .config import (
    BOTTOM_TAG,
    CANISTER_WALL_TAG,
    MATERIAL_ID_OFFSET,
    MATERIAL_NAMES,
    TOP_TAG,
    domain_bottom_z,
    domain_length_x,
    domain_width_y,
    drift_axis_z,
    drift_inner_radius,
    drift_outer_radius,
    quarter_canister_volume,
)

#: Tolerance [m] for a surface lying in a coordinate plane.
PLANE_EPS = 1.0e-6

# gmsh holds process-global state; meshing is serialized across threads.
_GMSH_LOCK = threading.Lock()


def material_tag(name: str) -> int:
    """Physical-group tag of a material region.

    :param name: one of :data:`theresa.config.MATERIAL_NAMES`.
    :returns: ``MaterialIDs + MATERIAL_ID_OFFSET``.
    """
    return MATERIAL_NAMES.index(name) + MATERIAL_ID_OFFSET


def material_id(name: str) -> int:
    """OGS ``MaterialIDs`` value of a material region (zero-based)."""
    return MATERIAL_NAMES.index(name)


def build_geometry(
    cfg: dict, msh_path: Path, on_step: Callable[[str], None] | None = None
) -> dict[str, int]:
    """Build and mesh the quarter cell and write it as Gmsh MSH 2.2.

    :param cfg: a configuration from :mod:`theresa.config`.
    :param msh_path: output file.
    :param on_step: called with ``"geometry"``, ``"curves"``, ``"surfaces"``
        and ``"volume"`` as each step begins.
    :returns: ``n_nodes``, ``n_elements`` and the region ``volumes`` [m3] of
        the geometry.
    :raises RuntimeError: if the geometry is degenerate.
    """
    with _GMSH_LOCK:
        return _build_locked(cfg, Path(msh_path), on_step or (lambda step: None))


def _build_locked(cfg: dict, msh_path: Path, on_step: Callable[[str], None]) -> dict:
    """:func:`build_geometry`, under the gmsh lock."""
    import gmsh

    lx = domain_length_x(cfg)
    ly = domain_width_y(cfg)
    z_bottom = domain_bottom_z(cfg)
    z_axis = drift_axis_z(cfg)
    height = -z_bottom  # from z_bottom up to z = 0
    r_can = cfg["r_can"]
    r_inner = drift_inner_radius(cfg)
    r_outer = drift_outer_radius(cfg)
    half_can = cfg["l_can"] / 2.0

    on_step("geometry")
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("quarter_cell")
        occ = gmsh.model.occ

        box = occ.addBox(0.0, 0.0, z_bottom, lx, ly, height)

        # Drift cylinders over the full length, the canister centred at x = 0;
        # intersecting with the box leaves the quarter.
        overhang = max(1.0, 0.05 * lx)
        cyl_outer = occ.addCylinder(-overhang, 0.0, z_axis, lx + 2.0 * overhang, 0.0, 0.0, r_outer)
        cyl_inner = occ.addCylinder(-overhang, 0.0, z_axis, lx + 2.0 * overhang, 0.0, 0.0, r_inner)
        cyl_can = occ.addCylinder(-half_can, 0.0, z_axis, 2.0 * half_can, 0.0, 0.0, r_can)

        # Clip each cylinder to the box.
        clipped = []
        for tag in (cyl_outer, cyl_inner, cyl_can):
            out, _ = occ.intersect([(3, tag)], [(3, box)], removeObject=True, removeTool=False)
            if not out:
                raise RuntimeError(
                    "a drift/canister cylinder does not intersect the domain box -- "
                    "check depth_m, canister_gap and a_drift"
                )
            clipped.append([dim_tag[1] for dim_tag in out])

        outer_vols, inner_vols, can_vols = clipped

        # Fragment for a conformal mesh; its map gives the output volumes of
        # each input body.
        inputs = (
            [(3, box)]
            + [(3, t) for t in outer_vols]
            + [(3, t) for t in inner_vols]
            + [(3, t) for t in can_vols]
        )
        _, frag_map = occ.fragment(inputs[:1], inputs[1:])
        occ.synchronize()

        def pieces(start: int, count: int) -> set[int]:
            """Output volumes of the inputs ``start`` to ``start + count``."""
            out: set[int] = set()
            for entry in frag_map[start : start + count]:
                out.update(tag for dim, tag in entry if dim == 3)
            return out

        n_outer, n_inner, n_can = len(outer_vols), len(inner_vols), len(can_vols)
        all_vols = pieces(0, 1)
        outer_set = pieces(1, n_outer)
        inner_set = pieces(1 + n_outer, n_inner)
        canister_set = pieces(1 + n_outer + n_inner, n_can)

        regions = {
            "canister": sorted(canister_set),
            "backfill": sorted(inner_set - canister_set),
            "lining": sorted(outer_set - inner_set),
            "host_rock": sorted(all_vols - outer_set),
        }
        for name, vols in regions.items():
            if not vols:
                raise RuntimeError(f"region {name!r} came out empty -- geometry is degenerate")

        # Region volumes, for comparison with analytic_volumes().
        volumes = {
            name: sum(abs(occ.getMass(3, tag)) for tag in vols) for name, vols in regions.items()
        }

        wall_surfaces = _canister_wall_surfaces(gmsh, regions["canister"])
        top_surfaces, bottom_surfaces = _horizontal_boundary_surfaces(gmsh, 0.0, z_bottom)

        for name in MATERIAL_NAMES:
            gmsh.model.addPhysicalGroup(3, regions[name], tag=material_tag(name), name=name)
        gmsh.model.addPhysicalGroup(2, top_surfaces, tag=TOP_TAG, name="top")
        gmsh.model.addPhysicalGroup(2, bottom_surfaces, tag=BOTTOM_TAG, name="bottom")
        gmsh.model.addPhysicalGroup(2, wall_surfaces, tag=CANISTER_WALL_TAG, name="canister_wall")

        _set_mesh_sizes(gmsh, cfg, wall_surfaces)

        # One dimension at a time, for progress reporting; same mesh as a
        # single generate(3).
        for dimension, step in ((1, "curves"), (2, "surfaces"), (3, "volume")):
            on_step(step)
            gmsh.model.mesh.generate(dimension)

        node_tags, _, _ = gmsh.model.mesh.getNodes()
        _, element_tags, _ = gmsh.model.mesh.getElements(dim=3)
        n_elements = sum(len(tags) for tags in element_tags)

        msh_path.parent.mkdir(parents=True, exist_ok=True)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(str(msh_path))

        return {"n_nodes": len(node_tags), "n_elements": n_elements, "volumes": volumes}
    finally:
        gmsh.finalize()


def analytic_volumes(cfg: dict) -> dict[str, float]:
    """Exact volumes [m3] of the four regions of the quarter cell."""
    lx = domain_length_x(cfg)
    ly = domain_width_y(cfg)
    height = -domain_bottom_z(cfg)
    r_inner = drift_inner_radius(cfg)
    r_outer = drift_outer_radius(cfg)

    canister = quarter_canister_volume(cfg)
    drift_inner = 0.5 * math.pi * r_inner**2 * lx
    drift_outer = 0.5 * math.pi * r_outer**2 * lx
    return {
        "canister": canister,
        "backfill": drift_inner - canister,
        "lining": drift_outer - drift_inner,
        "host_rock": lx * ly * height - drift_outer,
    }


def _canister_wall_surfaces(gmsh, canister_volumes: list[int]) -> list[int]:
    """Outer surface of the canister, without the faces in the symmetry planes."""
    boundary = gmsh.model.getBoundary(
        [(3, tag) for tag in canister_volumes], oriented=False, combined=True
    )
    wall = []
    for dim, tag in boundary:
        if dim != 2:
            continue
        xmin, ymin, _zmin, xmax, ymax, _zmax = gmsh.model.getBoundingBox(dim, tag)
        in_x_symmetry_plane = abs(xmin) < PLANE_EPS and abs(xmax) < PLANE_EPS
        in_y_symmetry_plane = abs(ymin) < PLANE_EPS and abs(ymax) < PLANE_EPS
        if not (in_x_symmetry_plane or in_y_symmetry_plane):
            wall.append(tag)
    if not wall:
        raise RuntimeError("no canister wall surface found after excluding the symmetry cuts")
    return wall


def _horizontal_boundary_surfaces(
    gmsh, z_top: float, z_bottom: float
) -> tuple[list[int], list[int]]:
    """Surfaces in the top and in the bottom boundary plane."""
    top, bottom = [], []
    for dim, tag in gmsh.model.getEntities(2):
        _xmin, _ymin, zmin, _xmax, _ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
        if abs(zmin - z_top) < PLANE_EPS and abs(zmax - z_top) < PLANE_EPS:
            top.append(tag)
        elif abs(zmin - z_bottom) < PLANE_EPS and abs(zmax - z_bottom) < PLANE_EPS:
            bottom.append(tag)
    if not top or not bottom:
        raise RuntimeError("top or bottom boundary surface not found")
    return top, bottom


def _set_mesh_sizes(gmsh, cfg: dict, wall_surfaces: list[int]) -> None:
    """Element size graded from ``mesh_size_fine`` at the canister outwards.

    The distance field alone sets the size; gmsh's boundary, point and
    curvature heuristics are disabled.
    """
    distance = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance, "SurfacesList", wall_surfaces)
    gmsh.model.mesh.field.setNumber(distance, "Sampling", 200)

    threshold = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold, "InField", distance)
    gmsh.model.mesh.field.setNumber(threshold, "SizeMin", cfg["mesh_size_fine"])
    gmsh.model.mesh.field.setNumber(threshold, "SizeMax", cfg["mesh_size_coarse"])
    gmsh.model.mesh.field.setNumber(threshold, "DistMin", cfg["mesh_dist_min"])
    gmsh.model.mesh.field.setNumber(threshold, "DistMax", cfg["mesh_dist_max"])

    gmsh.model.mesh.field.setAsBackgroundMesh(threshold)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
