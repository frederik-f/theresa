# SPDX-License-Identifier: GPL-3.0-or-later
"""Model configuration.

All parameters are held in one dictionary, :data:`CONFIG`; :func:`make_config`
returns a validated copy with overrides applied.

Coordinates
-----------
The model is a quarter cell around one canister (see :mod:`theresa.geometry`):

- ``x`` along the drift axis; ``x = 0`` is the canister mid-plane,
  ``x = canister_pitch / 2`` the mid-plane to the next canister, with
  ``canister_pitch = l_can + canister_gap``.
- ``y`` horizontal, across the drift; ``y = 0`` is the vertical plane through
  the drift axis, ``y = a_drift / 2`` the mid-plane to the next drift.
- ``z`` upwards; ``z = 0`` is ground level, the drift axis is at
  ``z = -depth_m`` and the bottom boundary at ``z = -(depth_m + h_below_m)``.

The four lateral boundaries are symmetry planes: the model represents an
infinite array of identical, simultaneously emplaced canisters.

.. note::
   Geometry and material parameters are literature values for an illustrative
   study, not design data for a specific repository.
"""

from __future__ import annotations

import copy
import json
import math
import shutil
from pathlib import Path
from typing import Any

from . import spectra

#: Seconds per Julian year (365.25 d).
SECONDS_PER_YEAR = 365.25 * 24 * 3600.0

#: Celsius to Kelvin. OGS works in Kelvin.
KELVIN_OFFSET = 273.15

#: Offset between gmsh physical-group tags (positive) and OGS ``MaterialIDs``
#: (zero-based); removed again in :mod:`theresa.mesh_conversion`.
MATERIAL_ID_OFFSET = 1

#: Material regions in ``MaterialIDs`` order. The order fixes the mapping to
#: the media in the OGS project file.
MATERIAL_NAMES = ("canister", "backfill", "lining", "host_rock")

#: Flat names of the material properties, ``<material>_lambda`` [W/(m K)] and
#: ``<material>_rho_cp`` [J/(m3 K)], mapped to their entry in ``CONFIG``.
#: Setting ``rho_cp`` adjusts ``cp`` and keeps ``rho``; only the product enters
#: the heat equation.
MATERIAL_ALIASES: dict[str, tuple[str, str]] = {
    f"{name}_{prop}": (f"mat_{name}", prop)
    for name in MATERIAL_NAMES
    for prop in ("lambda", "rho_cp")
}

#: gmsh physical-group tags of the Dirichlet boundaries (top and bottom). The
#: lateral symmetry planes carry no boundary condition (zero flux).
TOP_TAG = 101
BOTTOM_TAG = 102
#: Outer surface of the canister, without the symmetry cuts.
CANISTER_WALL_TAG = 103

# Pollux-10 reference dimensions (GRS-281): length 5.52 m, diameter 1.56 m.
_POLLUX_LENGTH = 5.52
_POLLUX_DIAMETER = 1.56

#: Default configuration.
CONFIG: dict[str, Any] = {
    # --- Canister [m] -------------------------------------------------------
    "r_can": _POLLUX_DIAMETER / 2.0,  # outer radius
    "l_can": _POLLUX_LENGTH,  # length
    # --- Drift cross-section [m] --------------------------------------------
    "drift_diameter": 3.5,  # clear diameter inside the liner
    "lining_thickness": 0.30,  # liner thickness
    # --- Layout [m] ---------------------------------------------------------
    "canister_gap": 9.5,  # clear distance between canister end faces
    "a_drift": 40.0,  # drift spacing, axis to axis
    # --- Vertical extent [m] ------------------------------------------------
    "depth_m": 500.0,  # emplacement depth below ground level
    "h_below_m": 700.0,  # distance from emplacement level to bottom boundary
    # --- Mesh ---------------------------------------------------------------
    "mesh_size_fine": 0.15,  # element size at the canister [m]
    "mesh_size_coarse": 30.0,  # element size in the far field [m]
    "mesh_dist_min": 1.5,  # fine size up to this distance from the canister [m]
    "mesh_dist_max": 200.0,  # coarse size beyond this distance [m]
    # --- Materials: conductivity [W/(m K)], density [kg/m3], cp [J/(kg K)] ---
    "mat_canister": {"lambda": 40.0, "rho": 7850.0, "cp": 450.0},  # cast iron
    "mat_backfill": {"lambda": 0.8, "rho": 1450.0, "cp": 1000.0},  # bentonite
    "mat_lining": {"lambda": 1.8, "rho": 2400.0, "cp": 900.0},  # concrete
    "mat_host_rock": {"lambda": 1.9, "rho": 2450.0, "cp": 900.0},  # claystone
    # --- Undisturbed temperature: T(z) = T_surface - gradient * z ------------
    "surface_temperature_celsius": 10.0,
    "geothermal_gradient_k_per_m": 0.03,
    # --- Decay heat ---------------------------------------------------------
    "spectrum": "DWR-Mix",  # name in theresa.spectra.SPECTRA
    "n_fuel_elements": 2,  # fuel elements per canister
    "interim_storage_years": 50.0,  # storage time before emplacement
    "decay_heat_p_kw": list(spectra.DWR_MIX.p_kw),  # amplitudes per element [kW]
    "decay_heat_t_half_years": list(spectra.DWR_MIX.t_half_years),  # half-lives [a]
    # --- Time discretization ------------------------------------------------
    # Blocked-logarithmic schedule: steps_per_block steps per step size, the
    # size growing by dt_growth up to dt_max_years.
    "t_end_years": 1000.0,
    "dt_initial_seconds": 5.0e3,
    "dt_growth": 1.5,
    "dt_max_years": 5.0,
    "steps_per_block": 4,
    # --- Probe points on the canister surface -------------------------------
    "n_probe_theta": 13,  # around the circumference, 0 to 180 deg
    "n_probe_axial": 9,  # along the half length
    "probe_inset_m": 0.02,  # distance inside the canister surface [m]
    # --- Solver: "CG" (diagonal preconditioner) or "SparseLU" ---------------
    "linear_solver": "CG",
    # --- Design limit [degC], evaluation only --------------------------------
    "T_limit_celsius": 100.0,
    # --- Output -------------------------------------------------------------
    "work_dir": None,  # None: name derived from the configuration
}


def make_config(spectrum: str | None = None, base: dict | None = None, **overrides: Any) -> dict:
    """Return a validated copy of :data:`CONFIG` with overrides applied.

    Selecting a spectrum by name also sets its coefficients, unless
    coefficients are given explicitly.

    :param spectrum: name in :data:`theresa.spectra.SPECTRA`.
    :param base: applied before ``overrides``, e.g. a stored configuration.
    :param overrides: ``CONFIG`` keys or :data:`MATERIAL_ALIASES`.
    :raises KeyError: on an unknown key.
    :raises ValueError: if the configuration is invalid.
    """
    cfg = copy.deepcopy(CONFIG)
    for source in (base or {}, overrides):
        for key, value in _apply_material_aliases(cfg, source).items():
            if key not in cfg:
                raise KeyError(f"unknown configuration key {key!r}")
            cfg[key] = value

    name = spectrum if spectrum is not None else overrides.get("spectrum")
    gave_coefficients = "decay_heat_p_kw" in overrides or "decay_heat_t_half_years" in overrides
    if name is not None:
        cfg["spectrum"] = name
        # With explicit coefficients the name is only a label.
        if not gave_coefficients:
            chosen = spectra.get(name)
            cfg["decay_heat_p_kw"] = list(chosen.p_kw)
            cfg["decay_heat_t_half_years"] = list(chosen.t_half_years)

    validate_config(cfg)
    return cfg


def _apply_material_aliases(cfg: dict, source: dict) -> dict:
    """Apply the material aliases in ``source`` to ``cfg``.

    :returns: the remaining entries of ``source``.
    """
    rest: dict[str, Any] = {}
    for key, value in source.items():
        target = MATERIAL_ALIASES.get(key)
        if target is None:
            rest[key] = value
            continue
        material, prop = target
        if prop == "lambda":
            cfg[material]["lambda"] = float(value)
        else:
            cfg[material]["cp"] = float(value) / cfg[material]["rho"]
    return rest


def design_value(cfg: dict, name: str) -> float:
    """Value of a design variable: a ``CONFIG`` key or a material alias."""
    target = MATERIAL_ALIASES.get(name)
    if target is None:
        return float(cfg[name])
    material, prop = target
    if prop == "lambda":
        return float(cfg[material]["lambda"])
    return volumetric_heat_capacity(cfg, material.removeprefix("mat_"))


def volumetric_heat_capacity(cfg: dict, material: str) -> float:
    """Volumetric heat capacity ``rho * cp`` [J/(m3 K)].

    :param material: one of :data:`MATERIAL_NAMES`.
    """
    properties = cfg[f"mat_{material}"]
    return float(properties["rho"] * properties["cp"])


def validate_config(cfg: dict) -> None:
    """Check geometry, physics and numerics of ``cfg``.

    :raises ValueError: if ``cfg`` describes an invalid model.
    """
    r_can = cfg["r_can"]
    r_inner = cfg["drift_diameter"] / 2.0
    r_outer = r_inner + cfg["lining_thickness"]

    if r_can <= 0.0 or cfg["l_can"] <= 0.0:
        raise ValueError("canister radius and length must be positive")
    if r_can >= r_inner:
        raise ValueError(
            f"canister radius {r_can} m does not fit inside the clear drift "
            f"radius {r_inner} m -- there would be no backfill"
        )
    if cfg["lining_thickness"] < 0.0:
        raise ValueError("lining_thickness must not be negative")

    if cfg["canister_gap"] <= 0.0:
        raise ValueError(
            f"canister_gap = {cfg['canister_gap']} m must be positive -- it is the clear "
            "distance between the end faces of neighbouring canisters"
        )
    if cfg["a_drift"] <= 2.0 * r_outer:
        raise ValueError(
            f"drift spacing a_drift = {cfg['a_drift']} m must exceed the "
            f"excavated drift diameter {2.0 * r_outer} m"
        )

    if cfg["depth_m"] <= r_outer:
        raise ValueError(
            "emplacement depth must be below ground level by more than the drift radius"
        )
    if cfg["h_below_m"] <= 0.0:
        raise ValueError("h_below_m must be positive")

    if cfg["mesh_size_fine"] <= 0.0 or cfg["mesh_size_coarse"] < cfg["mesh_size_fine"]:
        raise ValueError("require 0 < mesh_size_fine <= mesh_size_coarse")

    if len(cfg["decay_heat_p_kw"]) != len(cfg["decay_heat_t_half_years"]):
        raise ValueError("decay_heat_p_kw and decay_heat_t_half_years must have equal length")
    if not cfg["decay_heat_p_kw"]:
        raise ValueError("the decay-heat spectrum must have at least one term")
    if not all(math.isfinite(t) and t > 0.0 for t in cfg["decay_heat_t_half_years"]):
        raise ValueError("every half-life must be positive and finite")
    if not all(math.isfinite(p) and p >= 0.0 for p in cfg["decay_heat_p_kw"]):
        raise ValueError("every decay-heat amplitude must be non-negative and finite")
    if cfg["n_fuel_elements"] < 0:
        raise ValueError("n_fuel_elements must not be negative")
    if not (math.isfinite(cfg["interim_storage_years"]) and cfg["interim_storage_years"] >= 0.0):
        raise ValueError("interim_storage_years must be non-negative and finite")
    for key in ("surface_temperature_celsius", "geothermal_gradient_k_per_m", "T_limit_celsius"):
        if not math.isfinite(cfg[key]):
            raise ValueError(f"{key} must be finite")

    if cfg["t_end_years"] <= 0.0:
        raise ValueError("t_end_years must be positive")
    if cfg["dt_growth"] <= 1.0:
        raise ValueError("dt_growth must exceed 1 for the schedule to advance")
    if int(cfg["steps_per_block"]) < 1:
        raise ValueError("steps_per_block must be at least 1")
    if cfg["dt_initial_seconds"] <= 0.0 or cfg["dt_max_years"] <= 0.0:
        raise ValueError("dt_initial_seconds and dt_max_years must be positive")
    if not 0.0 < cfg["probe_inset_m"] < cfg["r_can"]:
        raise ValueError("probe_inset_m must be positive and smaller than r_can")
    if int(cfg["n_probe_theta"]) < 2 or int(cfg["n_probe_axial"]) < 2:
        raise ValueError("need at least two probe points in each direction")
    for name in MATERIAL_NAMES:
        properties = cfg[f"mat_{name}"]
        if properties["lambda"] <= 0.0:
            raise ValueError(f"mat_{name}: thermal conductivity must be positive")
        if properties["rho"] <= 0.0 or properties["cp"] <= 0.0:
            raise ValueError(f"mat_{name}: density and specific heat capacity must be positive")

    if cfg["linear_solver"] not in ("CG", "SparseLU"):
        raise ValueError(f"unknown linear_solver {cfg['linear_solver']!r}")


# --- Derived geometry -------------------------------------------------------


def drift_inner_radius(cfg: dict) -> float:
    """Clear drift radius [m] (backfill/liner interface)."""
    return cfg["drift_diameter"] / 2.0


def drift_outer_radius(cfg: dict) -> float:
    """Excavated drift radius [m] (liner/host-rock interface)."""
    return drift_inner_radius(cfg) + cfg["lining_thickness"]


def canister_pitch(cfg: dict) -> float:
    """Centre-to-centre distance of adjacent canisters [m]."""
    return cfg["l_can"] + cfg["canister_gap"]


def domain_length_x(cfg: dict) -> float:
    """Model extent along the drift [m]: half the canister pitch."""
    return canister_pitch(cfg) / 2.0


def domain_width_y(cfg: dict) -> float:
    """Model extent across the drift [m]: half the drift spacing."""
    return cfg["a_drift"] / 2.0


def domain_bottom_z(cfg: dict) -> float:
    """Height of the bottom boundary [m] (negative)."""
    return -(cfg["depth_m"] + cfg["h_below_m"])


def drift_axis_z(cfg: dict) -> float:
    """Height of the drift axis [m] (negative)."""
    return -cfg["depth_m"]


def quarter_canister_volume(cfg: dict) -> float:
    """Volume [m3] of the modelled quarter canister (exact cylinder)."""
    return 0.25 * math.pi * cfg["r_can"] ** 2 * cfg["l_can"]


def undisturbed_temperature_celsius(cfg: dict, z: float) -> float:
    """Undisturbed rock temperature [degC] at height ``z <= 0``."""
    return cfg["surface_temperature_celsius"] - cfg["geothermal_gradient_k_per_m"] * z


def emplacement_temperature_celsius(cfg: dict) -> float:
    """Undisturbed rock temperature [degC] at the emplacement level."""
    return undisturbed_temperature_celsius(cfg, drift_axis_z(cfg))


# --- Run directories --------------------------------------------------------

#: Files and directories a new run replaces; anything else is kept.
RUN_ARTIFACTS = ("mesh", "results", "steady_state", "kernel", "config.json")


def _number(value: float) -> str:
    """Number for a directory name: ``10``, ``12p5``, ``m3``."""
    text = f"{value:g}"
    return text.replace(".", "p").replace("-", "m")


def run_directory_name(cfg: dict) -> str:
    """Run-directory name derived from the design variables."""
    return (
        f"gap{_number(cfg['canister_gap'])}"
        f"_adrift{_number(cfg['a_drift'])}"
        f"_len{_number(cfg['l_can'])}"
        f"_depth{_number(cfg['depth_m'])}"
        f"_{_number(cfg['t_end_years'])}a"
    )


def prepare_run_directory(work_dir: Path) -> Path:
    """Create ``work_dir`` and remove the artifacts of a previous run."""
    work_dir.mkdir(parents=True, exist_ok=True)
    for name in RUN_ARTIFACTS:
        target = work_dir / name
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
    for stale in work_dir.glob("*.prj"):
        stale.unlink()
    return work_dir


def write_config(cfg: dict, work_dir: Path) -> Path:
    """Write ``cfg`` to ``work_dir/config.json``."""
    path = work_dir / "config.json"
    path.write_text(json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8")
    return path


def read_config(work_dir: Path) -> dict:
    """Read a configuration written by :func:`write_config`."""
    return json.loads((work_dir / "config.json").read_text(encoding="utf-8"))
