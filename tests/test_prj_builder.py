# SPDX-License-Identifier: GPL-3.0-or-later
"""OGS project file and time schedule (no OGS run)."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from theresa.config import MATERIAL_NAMES, SECONDS_PER_YEAR, make_config
from theresa.decay_heat import volumetric_density_per_watt
from theresa.prj_builder import build_prj_xml, time_step_count, time_step_pairs

MESH_NAMES = ("domain", "top", "bottom", "canister", "canister_wall")


@pytest.fixture
def meshes(tmp_path: Path) -> dict[str, Path]:
    """Mesh paths as convert_mesh would return them."""
    directory = tmp_path / "mesh"
    directory.mkdir()
    paths = {}
    for name in MESH_NAMES:
        path = directory / f"{name}.vtu"
        path.write_text("", encoding="utf-8")
        paths[name] = path
    return paths


def parse(cfg: dict, meshes: dict[str, Path], tmp_path: Path, **kwargs) -> ET.Element:
    """Build a project file and return its parsed root element."""
    return ET.fromstring(build_prj_xml(cfg, meshes, prj_dir=tmp_path, **kwargs))


# --- time schedule ---------------------------------------------------------


def test_schedule_lands_exactly_on_the_end_time():
    # The schedule ends exactly at t_end.
    for t_end in (100.0, 1000.0, 2000.0):
        cfg = make_config(t_end_years=t_end)
        total = sum(repeat * delta for repeat, delta in time_step_pairs(cfg))
        assert total == pytest.approx(t_end * SECONDS_PER_YEAR, rel=1e-12)


def test_schedule_starts_fine_and_grows():
    cfg = make_config()
    pairs = time_step_pairs(cfg)

    assert pairs[0][1] == pytest.approx(cfg["dt_initial_seconds"])
    sizes = [delta for _, delta in pairs]
    # Monotone up to the final shortened step, which lands on t_end.
    assert all(later >= earlier for earlier, later in zip(sizes[:-2], sizes[1:-1], strict=False))
    assert max(sizes) <= cfg["dt_max_years"] * SECONDS_PER_YEAR


def test_schedule_keeps_the_number_of_distinct_step_sizes_small():
    # Few distinct step sizes.
    cfg = make_config()
    assert len(time_step_pairs(cfg)) < 0.1 * time_step_count(cfg)


# --- boundary conditions ---------------------------------------------------


def test_only_the_top_and_bottom_carry_a_boundary_condition(meshes, tmp_path):
    # No condition on the lateral symmetry planes.
    root = parse(make_config(), meshes, tmp_path, mode="kernel")

    conditions = root.findall(".//boundary_condition")
    constrained = sorted(element.findtext("mesh") for element in conditions)

    assert constrained == ["bottom", "top"]
    assert all(element.findtext("type") == "Dirichlet" for element in conditions)


def test_steady_mode_solves_for_psi_with_zero_and_height(meshes, tmp_path):
    cfg = make_config(depth_m=500.0, h_below_m=700.0)
    root = parse(cfg, meshes, tmp_path, mode="steady", top_value=0.0, bottom_value=1200.0)

    values = {
        element.findtext("name"): float(element.findtext("value"))
        for element in root.findall(".//parameters/parameter")
        if element.findtext("type") == "Constant"
    }
    assert values["T_top"] == pytest.approx(0.0)
    assert values["T_bottom"] == pytest.approx(1200.0)


# --- media -----------------------------------------------------------------


def test_every_material_gets_a_medium_in_material_id_order(meshes, tmp_path):
    cfg = make_config()
    root = parse(cfg, meshes, tmp_path, mode="kernel")

    media = root.findall(".//media/medium")
    assert [element.get("id") for element in media] == [str(i) for i in range(len(MATERIAL_NAMES))]

    for index, name in enumerate(MATERIAL_NAMES):
        properties = {
            entry.findtext("name"): float(entry.findtext("value"))
            for entry in media[index].findall(".//property")
        }
        assert properties["thermal_conductivity"] == pytest.approx(cfg[f"mat_{name}"]["lambda"])
        assert properties["density"] == pytest.approx(cfg[f"mat_{name}"]["rho"])


def test_steady_mode_zeroes_the_heat_capacity(meshes, tmp_path):
    # This is what turns one backward-Euler step into an exact steady solve.
    root = parse(make_config(), meshes, tmp_path, mode="steady")

    capacities = [
        float(entry.findtext("value"))
        for entry in root.findall(".//property")
        if entry.findtext("name") == "specific_heat_capacity"
    ]
    assert capacities == [0.0] * len(MATERIAL_NAMES)


def test_transient_modes_keep_the_real_heat_capacity(meshes, tmp_path):
    cfg = make_config()
    root = parse(cfg, meshes, tmp_path, mode="kernel")

    capacities = [
        float(entry.findtext("value"))
        for entry in root.findall(".//property")
        if entry.findtext("name") == "specific_heat_capacity"
    ]
    assert capacities == [cfg[f"mat_{name}"]["cp"] for name in MATERIAL_NAMES]


# --- source terms ----------------------------------------------------------


def test_kernel_mode_injects_exactly_one_watt_of_canister_power(meshes, tmp_path):
    # Source density per watt of full canister power.
    cfg = make_config()
    root = parse(cfg, meshes, tmp_path, mode="kernel")

    source = root.find(".//source_term")
    assert source.findtext("mesh") == "canister"
    assert source.findtext("type") == "Volumetric"

    density = next(
        float(entry.findtext("value"))
        for entry in root.findall(".//parameters/parameter")
        if entry.findtext("name") == "unit_source"
    )
    assert density == pytest.approx(volumetric_density_per_watt(cfg), rel=1e-12)

    full_volume = np.pi * cfg["r_can"] ** 2 * cfg["l_can"]
    assert density * full_volume == pytest.approx(1.0, rel=1e-12)


def test_steady_mode_has_no_source(meshes, tmp_path):
    root = parse(make_config(), meshes, tmp_path, mode="steady")
    assert root.findall(".//source_term") == []


def test_direct_mode_writes_the_decay_curve(meshes, tmp_path):
    cfg = make_config()
    root = parse(cfg, meshes, tmp_path, mode="direct", canister_power_watt=5000.0)

    curve = root.find(".//curves/curve")
    assert curve.findtext("name") == "decay_curve"
    times = [float(value) for value in curve.findtext("coords").split()]
    values = [float(value) for value in curve.findtext("values").split()]

    assert times[0] == 0.0
    assert times[-1] == pytest.approx(cfg["t_end_years"] * SECONDS_PER_YEAR, rel=1e-9)
    assert len(times) == len(values)
    # Decay heat only ever decreases.
    assert all(
        later <= earlier * (1 + 1e-12)
        for earlier, later in zip(values[:-1], values[1:], strict=True)
    )


# --- structure -------------------------------------------------------------


def test_linearity_flags_are_set(meshes, tmp_path):
    # Linear solver options.
    root = parse(make_config(), meshes, tmp_path, mode="kernel")
    process = root.find(".//processes/process")

    assert process.findtext("linear") == "true"
    assert process.findtext("linear_solver_compute_only_upon_timestep_change") == "true"


def test_mesh_paths_are_relative_to_the_project_directory(meshes, tmp_path):
    root = parse(make_config(), meshes, tmp_path, mode="kernel")
    declared = [element.text for element in root.findall("./meshes/mesh")]

    assert declared[0].endswith("domain.vtu")
    assert all(not Path(path).is_absolute() for path in declared)


def test_rejects_an_unknown_mode(meshes, tmp_path):
    with pytest.raises(ValueError, match="unknown mode"):
        parse(make_config(), meshes, tmp_path, mode="nonsense")


def test_rejects_missing_meshes(tmp_path):
    with pytest.raises(ValueError, match="missing"):
        build_prj_xml(make_config(), {}, prj_dir=tmp_path, mode="kernel")


def test_no_secondary_variables_are_declared(tmp_path):
    """No secondary variables are declared."""
    import xml.etree.ElementTree as ET

    from theresa.config import make_config
    from theresa.prj_builder import build_prj_xml

    meshes = {
        name: tmp_path / f"{name}.vtu"
        for name in ("domain", "top", "bottom", "canister", "canister_wall")
    }
    for mode, extra in (
        ("kernel", {"meshed_canister_volume_m3": 1.0}),
        ("steady", {"top_value": 0.0, "bottom_value": 1200.0}),
    ):
        root = ET.fromstring(
            build_prj_xml(make_config(), meshes, prj_dir=tmp_path, mode=mode, **extra)
        )
        assert root.find(".//secondary_variables") is None, mode


def test_cg_tolerance_depends_on_the_solve(tmp_path):
    import xml.etree.ElementTree as ET

    from theresa.config import make_config
    from theresa.prj_builder import (
        CG_TOLERANCE_STEADY,
        CG_TOLERANCE_TRANSIENT,
        build_prj_xml,
    )

    meshes = {
        name: tmp_path / f"{name}.vtu"
        for name in ("domain", "top", "bottom", "canister", "canister_wall")
    }
    kernel = ET.fromstring(
        build_prj_xml(
            make_config(), meshes, prj_dir=tmp_path, mode="kernel", meshed_canister_volume_m3=1.0
        )
    )
    steady = ET.fromstring(
        build_prj_xml(
            make_config(), meshes, prj_dir=tmp_path, mode="steady", top_value=0.0, bottom_value=1.0
        )
    )
    assert float(kernel.find(".//error_tolerance").text) == CG_TOLERANCE_TRANSIENT
    assert float(steady.find(".//error_tolerance").text) == CG_TOLERANCE_STEADY
