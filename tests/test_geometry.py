# SPDX-License-Identifier: GPL-3.0-or-later
"""Quarter-cell geometry: topology, region bookkeeping and volumes (Gmsh, slow).

Region volumes are compared with closed-form values.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from theresa.config import (
    MATERIAL_NAMES,
    canister_pitch,
    drift_axis_z,
    make_config,
)
from theresa.geometry import analytic_volumes, build_geometry, material_id, material_tag

pytest.importorskip("gmsh")

# Coarse mesh with gentle grading.
COARSE = {"mesh_size_fine": 0.5, "mesh_size_coarse": 40.0, "mesh_dist_max": 150.0}


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> tuple[dict, dict]:
    """A meshed default design, built once for the whole module."""
    cfg = make_config(**COARSE)
    path = tmp_path_factory.mktemp("geometry") / "mesh.msh"
    return cfg, build_geometry(cfg, path)


def test_material_ids_are_zero_based_and_tags_are_one_based():
    # MaterialIDs start at 0, Gmsh tags at MATERIAL_ID_OFFSET.
    for index, name in enumerate(MATERIAL_NAMES):
        assert material_id(name) == index
        assert material_tag(name) == index + 1


def test_region_volumes_match_the_analytic_values(built):
    cfg, stats = built
    exact = analytic_volumes(cfg)

    for name, volume in stats["volumes"].items():
        assert volume == pytest.approx(exact[name], rel=1e-9), name


def test_the_regions_fill_the_box_exactly(built):
    cfg, stats = built
    box = (canister_pitch(cfg) / 2.0) * (cfg["a_drift"] / 2.0) * (cfg["depth_m"] + cfg["h_below_m"])
    assert sum(stats["volumes"].values()) == pytest.approx(box, rel=1e-12)


def test_the_meshed_canister_is_a_quarter_cylinder(built):
    # Exactly a quarter of the canister.
    cfg, stats = built
    full = math.pi * cfg["r_can"] ** 2 * cfg["l_can"]
    assert stats["volumes"]["canister"] == pytest.approx(0.25 * full, rel=1e-9)


def test_the_drift_runs_the_full_model_length(built):
    # The drift spans the whole cell.
    cfg, stats = built
    lining_section = math.pi * (
        (cfg["drift_diameter"] / 2.0 + cfg["lining_thickness"]) ** 2
        - (cfg["drift_diameter"] / 2.0) ** 2
    )
    expected = 0.5 * lining_section * (canister_pitch(cfg) / 2.0)
    assert stats["volumes"]["lining"] == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize(
    "overrides",
    [
        {"canister_gap": 2.5, "a_drift": 15.0, "l_can": 4.0, "depth_m": 300.0},
        {"canister_gap": 15.0, "a_drift": 60.0, "l_can": 7.0, "depth_m": 1000.0},
        {"canister_gap": 9.5, "a_drift": 40.0, "l_can": 5.52, "depth_m": 650.0},
    ],
    ids=["smallest", "largest", "middle"],
)
def test_the_whole_design_range_meshes(tmp_path, overrides):
    # Every corner of the design space meshes.
    cfg = make_config(**{**COARSE, **overrides})
    stats = build_geometry(cfg, tmp_path / "mesh.msh")
    exact = analytic_volumes(cfg)

    assert stats["n_nodes"] > 0
    for name, volume in stats["volumes"].items():
        assert volume == pytest.approx(exact[name], rel=1e-9), name


def test_production_grading_meshes_the_extreme_corner(tmp_path):
    # The same corner, at the resolution actually used for training.
    cfg = make_config(canister_gap=15.0, a_drift=60.0, l_can=7.0, depth_m=1000.0)
    stats = build_geometry(cfg, tmp_path / "mesh.msh")
    assert stats["n_nodes"] > 5000


def test_canister_wall_excludes_the_symmetry_cuts(tmp_path):
    # The wall submesh excludes the cut faces.
    pytest.importorskip("ogstools")
    pyvista = pytest.importorskip("pyvista")
    from theresa.mesh_conversion import convert_mesh

    cfg = make_config(**COARSE)
    build_geometry(cfg, tmp_path / "mesh.msh")
    meshes = convert_mesh(tmp_path / "mesh.msh", tmp_path / "mesh")

    wall = pyvista.read(str(meshes["canister_wall"]))
    area = float(wall.compute_cell_sizes(length=False, area=True, volume=False)["Area"].sum())

    # Half the barrel (y >= 0, over the half length) plus half the end face.
    r, half_length = cfg["r_can"], cfg["l_can"] / 2.0
    expected = math.pi * r * half_length + 0.5 * math.pi * r**2
    # Faceting makes the meshed cylinder slightly smaller than the true one.
    assert area == pytest.approx(expected, rel=0.02)


def test_the_canister_sits_at_the_emplacement_depth(built):
    cfg, _stats = built
    assert drift_axis_z(cfg) == pytest.approx(-cfg["depth_m"])


def test_every_expected_physical_group_becomes_a_mesh(tmp_path):
    pytest.importorskip("ogstools")
    from theresa.mesh_conversion import convert_mesh

    cfg = make_config(**COARSE)
    build_geometry(cfg, tmp_path / "mesh.msh")
    meshes = convert_mesh(tmp_path / "mesh.msh", tmp_path / "mesh")

    for name in (*MATERIAL_NAMES, "top", "bottom", "canister_wall", "domain"):
        assert name in meshes, f"{name} missing from {sorted(meshes)}"


def test_material_ids_on_the_domain_mesh_are_zero_based(tmp_path):
    pytest.importorskip("ogstools")
    pyvista = pytest.importorskip("pyvista")
    from theresa.mesh_conversion import convert_mesh

    cfg = make_config(**COARSE)
    build_geometry(cfg, tmp_path / "mesh.msh")
    meshes = convert_mesh(tmp_path / "mesh.msh", tmp_path / "mesh")

    domain = pyvista.read(str(meshes["domain"]))
    present = set(np.unique(domain.cell_data["MaterialIDs"]).tolist())
    assert present == set(range(len(MATERIAL_NAMES)))


def test_the_meshed_canister_is_smaller_than_the_exact_one_and_converges(tmp_path):
    # Meshed/exact volume ratio converges to 1 under refinement.
    pytest.importorskip("ogstools")
    pytest.importorskip("pyvista")
    from theresa.kernel import faceting_ratio
    from theresa.mesh_conversion import convert_mesh

    ratios = []
    for fine in (0.80, 0.40, 0.20):
        cfg = make_config(**{**COARSE, "mesh_size_fine": fine})
        directory = tmp_path / f"fine{fine}"
        build_geometry(cfg, directory / "mesh.msh")
        meshes = convert_mesh(directory / "mesh.msh", directory / "mesh")
        ratios.append(faceting_ratio(cfg, meshes["canister"]))

    assert all(ratio < 1.0 for ratio in ratios), ratios
    assert ratios[0] < ratios[1] < ratios[2], ratios
    assert ratios[-1] > 0.99, ratios


def test_the_source_density_injects_exactly_a_quarter_watt(tmp_path):
    # 1 W canister power puts 0.25 W into the quarter cell.
    pytest.importorskip("ogstools")
    pytest.importorskip("pyvista")
    from theresa.decay_heat import volumetric_density_per_watt
    from theresa.kernel import meshed_canister_volume
    from theresa.mesh_conversion import convert_mesh

    cfg = make_config(**COARSE)
    build_geometry(cfg, tmp_path / "mesh.msh")
    meshes = convert_mesh(tmp_path / "mesh.msh", tmp_path / "mesh")

    volume = meshed_canister_volume(meshes["canister"])
    density = volumetric_density_per_watt(cfg, volume)

    assert density * volume == pytest.approx(0.25, rel=1e-12)

    # The exact cylinder volume underestimates the injected power.
    analytic = volumetric_density_per_watt(cfg)
    assert analytic * volume < 0.25
