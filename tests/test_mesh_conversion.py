# SPDX-License-Identifier: GPL-3.0-or-later
"""Fast mesh conversion: output identical to ogstools, array by array."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gmsh")
pytest.importorskip("ogstools")

import pyvista as pv  # noqa: E402

from theresa.config import MATERIAL_ID_OFFSET, make_config  # noqa: E402
from theresa.geometry import build_geometry  # noqa: E402
from theresa.mesh_conversion import _silence_stdout, convert_mesh  # noqa: E402

COARSE = {"mesh_size_fine": 0.5, "mesh_size_coarse": 40.0, "mesh_dist_max": 150.0}


def per_face(mesh: pv.UnstructuredGrid) -> list[tuple[int, ...]]:
    """Bulk elements of each face, where a face may touch several."""
    counts = np.asarray(mesh.cell_data["number_bulk_elements"], dtype=int)
    ids = np.asarray(mesh.field_data["bulk_element_ids"], dtype=int)
    return [tuple(sorted(part)) for part in np.split(ids, np.cumsum(counts)[:-1])]


def test_the_files_match_ogstools(tmp_path):
    from ogstools import Meshes

    msh = tmp_path / "mesh.msh"
    build_geometry(make_config(**COARSE), msh)

    with _silence_stdout():
        reference = Meshes.from_gmsh(msh, dim=0, reindex=False, log=False)
        for mesh in reference.values():
            if "MaterialIDs" in mesh.cell_data:
                mesh.cell_data["MaterialIDs"] -= MATERIAL_ID_OFFSET
        reference.save(tmp_path / "reference", overwrite=True)
    produced = convert_mesh(msh, tmp_path / "fast")

    expected = sorted(path.stem for path in (tmp_path / "reference").glob("*.vtu"))
    assert sorted(produced) == expected

    for name in expected:
        ours = pv.read(produced[name])
        theirs = pv.read(tmp_path / "reference" / f"{name}.vtu")
        for attribute in ("points", "cells", "celltypes"):
            assert np.array_equal(getattr(ours, attribute), getattr(theirs, attribute)), name
        for kind in ("point_data", "cell_data", "field_data"):
            assert set(getattr(ours, kind)) == set(getattr(theirs, kind)), (name, kind)
            for key in getattr(theirs, kind):
                if kind == "field_data" and key == "bulk_element_ids":
                    # Several bulk elements per face: order within a face is free.
                    assert per_face(ours) == per_face(theirs), name
                    continue
                a, b = getattr(ours, kind)[key], getattr(theirs, kind)[key]
                assert a.dtype == b.dtype and np.array_equal(a, b), (name, key)
