# SPDX-License-Identifier: GPL-3.0-or-later
"""Conversion of Gmsh meshes into OGS VTU files.

Writes the bulk mesh ``domain.vtu`` with ``MaterialIDs`` and one mesh per
physical group with ``bulk_node_ids`` and ``bulk_element_ids``. The output is
identical to :meth:`ogstools.Meshes.from_gmsh`.

For tetrahedral meshes with triangular faces the bulk ids are computed from
the extraction indices and a sorted table of tetrahedron faces; other meshes
are converted by ogstools. Node reordering and file output use ogstools.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# The ogstools NodeReordering tool writes to the stdout file descriptor; it is
# redirected under a lock because the descriptor is process-wide.
_STDOUT_LOCK = threading.Lock()

#: VTK cell types handled directly.
VTK_TRIANGLE = 5
VTK_TETRA = 10

#: Faces of a tetrahedron, as node positions.
TETRA_FACES = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))

#: Arrays removed from every subdomain mesh.
_TRANSIENT_ARRAYS = (
    "vtkOriginalPointIds",
    "vtkOriginalCellIds",
    "gmsh:dim_tags",
    "gmsh:physical",
    "gmsh:geometrical",
)


@contextlib.contextmanager
def _silence_stdout():
    """Redirect the stdout file descriptor to ``/dev/null``."""
    with _STDOUT_LOCK:
        saved_fd = os.dup(1)
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(devnull_fd, 1)
            yield
        finally:
            os.dup2(saved_fd, 1)
            os.close(devnull_fd)
            os.close(saved_fd)


def convert_mesh(msh_path: Path, out_dir: Path) -> dict[str, Path]:
    """Convert a Gmsh mesh into OGS VTU meshes.

    Material ids are the physical-group tags minus
    :data:`theresa.config.MATERIAL_ID_OFFSET`.

    :param msh_path: Gmsh mesh from :func:`theresa.geometry.build_geometry`.
    :param out_dir: output directory; one file per physical group, e.g.
        ``domain.vtu``, ``top.vtu``, ``canister.vtu``.
    :returns: physical-group name to ``.vtu`` file.
    :raises RuntimeError: if no files were produced.
    """
    from ogstools import Meshes

    from .config import MATERIAL_ID_OFFSET

    out_dir.mkdir(parents=True, exist_ok=True)

    with _silence_stdout():
        parts = _meshes_from_gmsh(Path(msh_path))
        if parts is None:
            meshes = Meshes.from_gmsh(msh_path, dim=0, reindex=False, log=False)
        else:
            meshes = Meshes(parts)
            meshes.has_identified_subdomains = True
        for mesh in meshes.values():
            if "MaterialIDs" in mesh.cell_data:
                mesh.cell_data["MaterialIDs"] -= MATERIAL_ID_OFFSET
        saved_paths = meshes.save(out_dir, overwrite=True)

    produced = {p.stem: p for p in saved_paths if p.suffix == ".vtu"}
    if not produced:
        raise RuntimeError(f"ogstools produced no VTU files for {msh_path}")

    logger.info("[mesh_conversion] generated %d mesh files in %s:", len(produced), out_dir)
    for name, path in produced.items():
        logger.info("    %s: %s", name, path)

    return produced


def _meshes_from_gmsh(msh_path: Path) -> dict | None:
    """Bulk and subdomain meshes of a tetrahedral mesh.

    Same steps as :func:`ogstools.meshes.gmsh_converter.meshes_from_gmsh`:
    read, clean, reorder nodes, split by physical group.

    :returns: name to mesh, or ``None`` if the mesh is not tetrahedral with
        triangular faces.
    """
    import meshio
    import pyvista as pv
    from meshio._vtk_common import meshio_to_vtk_type
    from ogstools.mesh import node_reordering

    source = meshio.read(str(msh_path))
    # Cell sets are set aside for pyvista's conversion, as in ogstools.
    cell_sets = source.cell_sets.copy()
    source.cell_sets = None
    whole = node_reordering(pv.from_meshio(source).clean(), 1, log=False)
    source.cell_sets = cell_sets

    types = np.asarray(whole.celltypes)
    if not np.isin(types, (VTK_TRIANGLE, VTK_TETRA)).all() or not (types == VTK_TETRA).any():
        return None

    domain = whole.extract_cells(types == VTK_TETRA)
    # Index of each point and cell in the bulk mesh, -1 if not contained.
    bulk_point = np.full(whole.n_points, -1, dtype=np.int64)
    bulk_point[domain["vtkOriginalPointIds"]] = np.arange(domain.n_points)
    bulk_cell = np.full(whole.n_cells, -1, dtype=np.int64)
    bulk_cell[domain["vtkOriginalCellIds"]] = np.arange(domain.n_cells)

    material = domain.cell_data["gmsh:physical"]
    domain.clear_cell_data()
    domain.clear_point_data()
    domain.clear_field_data()
    domain.cell_data["MaterialIDs"] = np.int32(material)

    tetrahedra = domain.cells.reshape(-1, 5)[:, 1:]
    faces = _face_table(tetrahedra, domain.n_points)

    dimension = np.where(types == VTK_TETRA, 3, 2)
    physical = np.asarray(whole.cell_data["gmsh:physical"])

    meshes = {"domain": domain}
    for name, (group_tag, group_dim) in source.field_data.items():
        if name in (msh_path.stem, "domain"):
            continue
        if cell_sets:
            # MSH 4: cells listed per group.
            selected = np.zeros(whole.n_cells, dtype=bool)
            for type_name, subset in source.cell_sets_dict[name].items():
                of_type = np.nonzero(types == meshio_to_vtk_type[type_name])[0]
                selected[of_type[subset]] = True
        else:
            # MSH 2.2: one group tag per cell.
            selected = (dimension == group_dim) & (physical == group_tag)
        subdomain = whole.extract_cells(selected)
        if subdomain.n_cells == 0:
            raise RuntimeError(f"physical group {name!r} is empty")

        nodes = bulk_point[subdomain["vtkOriginalPointIds"]]
        cells = bulk_cell[subdomain["vtkOriginalCellIds"]]
        if (nodes < 0).any():
            raise RuntimeError(f"physical group {name!r} has points outside the bulk mesh")

        for array in _TRANSIENT_ARRAYS:
            subdomain.point_data.pop(array, None)
            subdomain.cell_data.pop(array, None)
        subdomain.field_data.clear()

        sub_types = np.unique(subdomain.celltypes)
        if sub_types.tolist() == [VTK_TETRA]:
            subdomain.cell_data["bulk_element_ids"] = np.uint64(cells)
        elif sub_types.tolist() == [VTK_TRIANGLE]:
            triangles = nodes[subdomain.cells.reshape(-1, 4)[:, 1:]]
            elements, counts = _elements_on_faces(faces, triangles, domain.n_points)
            if (counts == 1).all():
                subdomain.cell_data["bulk_element_ids"] = np.uint64(elements)
            else:
                # Interface faces touch two bulk cells: list as field data,
                # count per face as cell data.
                subdomain.field_data["bulk_element_ids"] = np.uint64(elements)
                subdomain.cell_data["number_bulk_elements"] = np.uint64(counts)
        else:
            raise RuntimeError(f"physical group {name!r} mixes element types {sub_types}")

        subdomain.point_data["bulk_node_ids"] = np.uint64(nodes)
        meshes[name] = subdomain
    return meshes


def _face_keys(nodes: np.ndarray, n_points: int) -> np.ndarray:
    """Integer key per triangle, independent of node order."""
    ordered = np.sort(nodes, axis=1).astype(np.int64)
    return (ordered[:, 0] * n_points + ordered[:, 1]) * n_points + ordered[:, 2]


def _face_table(tetrahedra: np.ndarray, n_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Keys of all tetrahedron faces, sorted, with their cell index."""
    if n_points**3 >= 2**63:
        raise ValueError("too many points to encode a face in one 64-bit integer")
    keys = np.concatenate([_face_keys(tetrahedra[:, list(face)], n_points) for face in TETRA_FACES])
    cells = np.tile(np.arange(tetrahedra.shape[0]), len(TETRA_FACES))
    order = np.argsort(keys, kind="stable")
    return keys[order], cells[order]


def _elements_on_faces(
    faces: tuple[np.ndarray, np.ndarray], triangles: np.ndarray, n_points: int
) -> tuple[np.ndarray, np.ndarray]:
    """Bulk cells adjacent to each triangle.

    :param faces: table from :func:`_face_table`.
    :param triangles: ``(n, 3)`` bulk node ids.
    :returns: ``(elements, counts)``: adjacent cells of all triangles in order,
        ascending per triangle, and their number per triangle.
    :raises RuntimeError: if a triangle is not a face of any bulk cell.
    """
    keys, cells = faces
    wanted = _face_keys(triangles, n_points)
    first = np.searchsorted(keys, wanted, side="left")
    last = np.searchsorted(keys, wanted, side="right")
    counts = last - first
    if (counts == 0).any():
        raise RuntimeError(f"{int((counts == 0).sum())} faces lie on no bulk element")
    elements = np.concatenate([np.sort(cells[a:b]) for a, b in zip(first, last, strict=True)])
    return elements, counts
