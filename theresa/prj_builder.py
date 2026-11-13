# SPDX-License-Identifier: GPL-3.0-or-later
r"""OGS project files (``.prj``) for the ``HEAT_CONDUCTION`` process.

Run modes:

``steady``
    Steady state: zero heat capacity, one time step. With boundary values
    ``0`` (top) and ``H`` (bottom) it yields :math:`\psi`
    (:mod:`theresa.steady_state`).
``kernel``
    Unit step response: zero initial and boundary values, constant source of
    1 W canister power. The result is :math:`S(x, t)` in K/W.
``direct``
    Real decay-heat curve, boundary temperatures and steady initial field.

The process is linear; ``<linear>`` and
``linear_solver_compute_only_upon_timestep_change`` assemble once and set up
the solver once per step size. :func:`time_step_pairs` uses few distinct
step sizes (28 for 1000 a).
"""

from __future__ import annotations

from pathlib import Path

from .config import (
    KELVIN_OFFSET,
    MATERIAL_NAMES,
    SECONDS_PER_YEAR,
    emplacement_temperature_celsius,
)
from .decay_heat import build_decay_curve, volumetric_density_per_watt
from .geometry import material_id

#: Heat capacity in ``steady`` mode.
STEADY_STATE_HEAT_CAPACITY = 0.0

#: Meshes declared in every project file; the bulk mesh comes first.
BULK_MESH = "domain"
BOUNDARY_MESHES = ("top", "bottom")
SUBDOMAIN_MESHES = ("canister", "canister_wall")


#: CG relative residual tolerance for transient runs.
CG_TOLERANCE_TRANSIENT = 1.0e-10

#: CG relative residual tolerance for the steady solve.
CG_TOLERANCE_STEADY = 1.0e-14


def time_step_pairs(cfg: dict) -> list[tuple[int, float]]:
    """``(repeat, delta_t)`` pairs of the blocked-logarithmic schedule.

    ``steps_per_block`` steps per step size, the size growing by ``dt_growth``
    up to ``dt_max_years``; the last step ends exactly at ``t_end_years``.

    :param cfg: a configuration from :mod:`theresa.config`.
    :returns: pairs in project-file order.
    """
    t_end = cfg["t_end_years"] * SECONDS_PER_YEAR
    dt_max = cfg["dt_max_years"] * SECONDS_PER_YEAR
    per_block = int(cfg["steps_per_block"])

    steps: list[float] = []
    dt = float(cfg["dt_initial_seconds"])
    t = 0.0
    while t < t_end:
        for _ in range(per_block):
            if t >= t_end:
                break
            step = min(dt, t_end - t)
            steps.append(step)
            t += step
        dt = min(dt * cfg["dt_growth"], dt_max)

    # Runs of equal step sizes become one pair.
    pairs: list[tuple[int, float]] = []
    for step in steps:
        if pairs and pairs[-1][1] == step:
            pairs[-1] = (pairs[-1][0] + 1, step)
        else:
            pairs.append((1, step))
    return pairs


def time_step_count(cfg: dict) -> int:
    """Number of time steps of the transient schedule."""
    return sum(repeat for repeat, _ in time_step_pairs(cfg))


def build_prj_xml(
    cfg: dict,
    meshes: dict[str, Path],
    prj_dir: Path,
    *,
    mode: str,
    top_value: float = 0.0,
    bottom_value: float = 0.0,
    canister_power_watt: float = 1.0,
    initial_field: str | None = None,
    initial_value: float = 0.0,
    meshed_canister_volume_m3: float | None = None,
    output_domain: bool = False,
) -> str:
    """Complete OGS project file as a string.

    :param cfg: a configuration from :mod:`theresa.config`.
    :param meshes: physical-group name to VTU path, from
        :func:`theresa.mesh_conversion.convert_mesh`.
    :param prj_dir: directory of the project file; mesh paths are relative to
        it.
    :param mode: ``"steady"``, ``"kernel"`` or ``"direct"``.
    :param top_value: Dirichlet value at the top [K, or auxiliary in
        ``steady`` mode].
    :param bottom_value: Dirichlet value at the bottom.
    :param canister_power_watt: canister power at ``t = 0`` [W], ``direct``
        mode; ``kernel`` mode uses 1 W.
    :param initial_field: point-data array on the domain mesh used as initial
        condition; ``None`` for a constant.
    :param initial_value: constant initial value.
    :param meshed_canister_volume_m3: meshed quarter-canister volume, see
        :func:`theresa.decay_heat.volumetric_density_per_watt`.
    :param output_domain: also write the bulk field at every step.
    :raises ValueError: on an unknown ``mode`` or a missing mesh.
    """
    if mode not in ("steady", "kernel", "direct"):
        raise ValueError(f"unknown mode {mode!r}")

    required = (BULK_MESH, *BOUNDARY_MESHES, *SUBDOMAIN_MESHES)
    missing = [name for name in required if name not in meshes]
    if missing:
        raise ValueError(f"mesh(es) missing from the conversion output: {', '.join(missing)}")

    def rel(name: str) -> str:
        return str(meshes[name].resolve().relative_to(Path(prj_dir).resolve()))

    steady = mode == "steady"
    reference_kelvin = emplacement_temperature_celsius(cfg) + KELVIN_OFFSET

    # --- Meshes -------------------------------------------------------------
    declared = [BULK_MESH, *BOUNDARY_MESHES, *SUBDOMAIN_MESHES]
    mesh_list_xml = "\n".join(f"        <mesh>{rel(name)}</mesh>" for name in declared)

    if steady:
        output_names = [BULK_MESH]
    else:
        output_names = ([BULK_MESH] if output_domain else []) + list(SUBDOMAIN_MESHES)
    output_meshes_xml = "".join(f"\n                <mesh>{name}</mesh>" for name in output_names)

    # --- Media --------------------------------------------------------------
    # Medium id = MaterialIDs value.
    media_xml = ""
    for name in MATERIAL_NAMES:
        props = cfg[f"mat_{name}"]
        cp = STEADY_STATE_HEAT_CAPACITY if steady else props["cp"]
        media_xml += f"""
        <medium id="{material_id(name)}">
            <!-- {name} -->
            <properties>
                <property>
                    <name>density</name>
                    <type>Constant</type>
                    <value>{props["rho"]}</value>
                </property>
                <property>
                    <name>specific_heat_capacity</name>
                    <type>Constant</type>
                    <value>{cp}</value>
                </property>
                <property>
                    <name>thermal_conductivity</name>
                    <type>Constant</type>
                    <value>{props["lambda"]}</value>
                </property>
                <property>
                    <name>reference_temperature</name>
                    <type>Constant</type>
                    <value>{reference_kelvin}</value>
                </property>
            </properties>
        </medium>"""

    # --- Time stepping ------------------------------------------------------
    if steady:
        # Any step length; no mass term.
        t_end_seconds = SECONDS_PER_YEAR
        timesteps_xml = f"""
                        <pair>
                            <repeat>1</repeat>
                            <delta_t>{SECONDS_PER_YEAR:.6e}</delta_t>
                        </pair>"""
    else:
        t_end_seconds = cfg["t_end_years"] * SECONDS_PER_YEAR
        timesteps_xml = "".join(
            f"""
                        <pair>
                            <repeat>{repeat}</repeat>
                            <delta_t>{delta_t:.9e}</delta_t>
                        </pair>"""
            for repeat, delta_t in time_step_pairs(cfg)
        )

    # --- Source term --------------------------------------------------------
    curves_xml = ""
    parameters_xml = ""
    source_terms_xml = ""
    if mode == "kernel":
        q_unit = volumetric_density_per_watt(cfg, meshed_canister_volume_m3)
        parameters_xml += f"""
        <parameter>
            <!-- 1 W canister power from t = 0: step response in K/W. -->
            <name>unit_source</name>
            <type>Constant</type>
            <value>{q_unit:.12e}</value>
        </parameter>"""
        source_terms_xml += """
                <source_term>
                    <mesh>canister</mesh>
                    <type>Volumetric</type>
                    <parameter>unit_source</parameter>
                </source_term>"""
    elif mode == "direct":
        curve_t, curve_q = build_decay_curve(cfg, canister_power_watt, meshed_canister_volume_m3)
        coords_str = " ".join(f"{v:.6e}" for v in curve_t)
        values_str = " ".join(f"{v:.6e}" for v in curve_q)
        curves_xml += f"""
        <curve>
            <name>decay_curve</name>
            <coords>{coords_str}</coords>
            <values>{values_str}</values>
        </curve>"""
        parameters_xml += """
        <parameter>
            <name>unit_scale</name>
            <type>Constant</type>
            <value>1.0</value>
        </parameter>
        <parameter>
            <name>decay_heat</name>
            <type>CurveScaled</type>
            <curve>decay_curve</curve>
            <parameter>unit_scale</parameter>
        </parameter>"""
        source_terms_xml += """
                <source_term>
                    <mesh>canister</mesh>
                    <type>Volumetric</type>
                    <parameter>decay_heat</parameter>
                </source_term>"""

    # --- Initial condition --------------------------------------------------
    if initial_field is None:
        initial_condition_name = "T_initial"
        parameters_xml += f"""
        <parameter>
            <name>T_initial</name>
            <type>Constant</type>
            <value>{initial_value:.12e}</value>
        </parameter>"""
    else:
        initial_condition_name = "T_initial"
        parameters_xml += f"""
        <parameter>
            <!-- Steady-state field, point data on the domain mesh. -->
            <name>T_initial</name>
            <type>MeshNode</type>
            <field_name>{initial_field}</field_name>
        </parameter>"""

    # --- Linear solver ------------------------------------------------------
    if cfg["linear_solver"] == "CG":
        tolerance = CG_TOLERANCE_STEADY if steady else CG_TOLERANCE_TRANSIENT
        solver_xml = f"""
                <solver_type>CG</solver_type>
                <precon_type>DIAGONAL</precon_type>
                <max_iteration_step>20000</max_iteration_step>
                <error_tolerance>{tolerance:g}</error_tolerance>
                <scaling>true</scaling>"""
    else:
        solver_xml = """
                <solver_type>SparseLU</solver_type>
                <scaling>true</scaling>"""

    return f"""<?xml version="1.0" encoding="ISO-8859-1"?>
<OpenGeoSysProject>
    <meshes>
{mesh_list_xml}
    </meshes>

    <processes>
        <process>
            <name>HeatConduction</name>
            <type>HEAT_CONDUCTION</type>
            <integration_order>2</integration_order>
            <process_variables>
                <process_variable>temperature</process_variable>
            </process_variables>
            <!-- Linear: assemble once, set up the solver once per step size. -->
            <linear>true</linear>
            <linear_solver_compute_only_upon_timestep_change>true</linear_solver_compute_only_upon_timestep_change>
        </process>
    </processes>

    <media>{media_xml}
    </media>

    <time_loop>
        <processes>
            <process ref="HeatConduction">
                <nonlinear_solver>basic_picard</nonlinear_solver>
                <convergence_criterion>
                    <type>DeltaX</type>
                    <norm_type>NORM2</norm_type>
                    <abstol>1e-6</abstol>
                </convergence_criterion>
                <time_discretization>
                    <type>BackwardEuler</type>
                </time_discretization>
                <time_stepping>
                    <type>FixedTimeStepping</type>
                    <t_initial>0</t_initial>
                    <t_end>{t_end_seconds:.9e}</t_end>
                    <timesteps>{timesteps_xml}
                    </timesteps>
                </time_stepping>
            </process>
        </processes>
        <output>
            <type>VTK</type>
            <prefix>result_{{:meshname}}</prefix>
            <meshes>{output_meshes_xml}
            </meshes>
            <timesteps>
                <pair>
                    <repeat>1</repeat>
                    <each_steps>1</each_steps>
                </pair>
            </timesteps>
            <variables>
                <variable>temperature</variable>
                <variable>MaterialIDs</variable>
            </variables>
        </output>
    </time_loop>

    <parameters>
        <parameter>
            <name>T_top</name>
            <type>Constant</type>
            <value>{top_value:.12e}</value>
        </parameter>
        <parameter>
            <name>T_bottom</name>
            <type>Constant</type>
            <value>{bottom_value:.12e}</value>
        </parameter>{parameters_xml}
    </parameters>

    <curves>{curves_xml}
    </curves>

    <process_variables>
        <process_variable>
            <name>temperature</name>
            <components>1</components>
            <order>1</order>
            <initial_condition>{initial_condition_name}</initial_condition>
            <!-- Lateral faces: symmetry planes, zero flux (no condition). -->
            <boundary_conditions>
                <boundary_condition>
                    <mesh>top</mesh>
                    <type>Dirichlet</type>
                    <parameter>T_top</parameter>
                </boundary_condition>
                <boundary_condition>
                    <mesh>bottom</mesh>
                    <type>Dirichlet</type>
                    <parameter>T_bottom</parameter>
                </boundary_condition>
            </boundary_conditions>
            <source_terms>{source_terms_xml}
            </source_terms>
        </process_variable>
    </process_variables>

    <nonlinear_solvers>
        <nonlinear_solver>
            <name>basic_picard</name>
            <type>Picard</type>
            <max_iter>4</max_iter>
            <linear_solver>general_linear_solver</linear_solver>
        </nonlinear_solver>
    </nonlinear_solvers>

    <linear_solvers>
        <linear_solver>
            <name>general_linear_solver</name>
            <eigen>{solver_xml}
            </eigen>
        </linear_solver>
    </linear_solvers>
</OpenGeoSysProject>
"""
