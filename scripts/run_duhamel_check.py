# SPDX-License-Identifier: GPL-3.0-or-later
"""Verification of the Duhamel construction against a direct simulation.

For one design:

1. steady solve for ``psi``;
2. step-response run, convolved with the decay curve;
3. direct transient run with the decay curve, the boundary temperatures and
   the undisturbed initial field.

Reports the difference between 1-2 and 3.

Run::

    uv run python scripts/run_duhamel_check.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from theresa.config import (  # noqa: E402
    SECONDS_PER_YEAR,
    make_config,
    prepare_run_directory,
    write_config,
)
from theresa.decay_heat import nominal_power_watt  # noqa: E402
from theresa.geometry import build_geometry  # noqa: E402
from theresa.kernel import extract_kernel, meshed_canister_volume  # noqa: E402
from theresa.mesh_conversion import convert_mesh  # noqa: E402
from theresa.pipeline import (  # noqa: E402
    boundary_temperatures_kelvin,
    psi_at_probe_points,
    temperature_history,
)
from theresa.prj_builder import build_prj_xml  # noqa: E402
from theresa.runner import run_ogs  # noqa: E402
from theresa.steady_state import compute_psi, write_initial_temperature  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs" / "duhamel_check"

# Six-term source.
SIX_TERM_P_KW = [3.318, 1.144, 0.2225, 0.05, 0.02112, 0.0009342]
SIX_TERM_T_HALF = [2.163, 32.33, 397.3, 1500.0, 13680.0, 760700.0]


def main() -> int:
    """Compute one case both ways and report the difference.

    :returns: exit status.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("theresa.runner").setLevel(logging.WARNING)
    logging.getLogger("theresa.mesh_conversion").setLevel(logging.WARNING)

    cfg = make_config(
        canister_gap=9.5,
        a_drift=40.0,
        depth_m=500.0,
        spectrum="six-term check",
        decay_heat_p_kw=SIX_TERM_P_KW,
        decay_heat_t_half_years=SIX_TERM_T_HALF,
    )
    work_dir = prepare_run_directory(RUNS)
    write_config(cfg, work_dir)

    started = time.time()
    stats = build_geometry(cfg, work_dir / "mesh.msh")
    meshes = convert_mesh(work_dir / "mesh.msh", work_dir / "mesh")
    logging.info("mesh: %d nodes", stats["n_nodes"])

    canister_volume = meshed_canister_volume(meshes["canister"])

    # --- the surrogate's route: psi + step response + convolution -----------
    psi = compute_psi(cfg, work_dir, meshes)
    psi_probe = psi_at_probe_points(cfg, meshes)

    kernel_prj = work_dir / "kernel.prj"
    kernel_prj.write_text(
        build_prj_xml(
            cfg,
            meshes,
            prj_dir=work_dir,
            mode="kernel",
            meshed_canister_volume_m3=canister_volume,
        ),
        encoding="ISO-8859-1",
    )
    run_ogs(kernel_prj, work_dir / "kernel")
    kernel = extract_kernel(cfg, work_dir / "kernel", canister_mesh=meshes["canister"])
    assembled = temperature_history(cfg, kernel["times"], kernel["probe"], psi_probe)
    logging.info("kernel + convolution done (%.1f min)", (time.time() - started) / 60.0)

    # --- the conventional route: one transient run with the real source ------
    initial_field = write_initial_temperature(cfg, meshes, psi)
    top_kelvin, bottom_kelvin = boundary_temperatures_kelvin(cfg)

    direct_prj = work_dir / "direct.prj"
    direct_prj.write_text(
        build_prj_xml(
            cfg,
            meshes,
            prj_dir=work_dir,
            mode="direct",
            top_value=top_kelvin,
            bottom_value=bottom_kelvin,
            canister_power_watt=nominal_power_watt(cfg),
            initial_field=initial_field,
            meshed_canister_volume_m3=canister_volume,
        ),
        encoding="ISO-8859-1",
    )
    run_ogs(direct_prj, work_dir / "direct")
    direct = extract_kernel(cfg, work_dir / "direct", canister_mesh=meshes["canister"])
    logging.info("direct run done (%.1f min total)", (time.time() - started) / 60.0)

    report(cfg, assembled, direct, work_dir)
    return 0


def report(cfg: dict, assembled, direct: dict, work_dir: Path) -> None:
    """Compare the two routes and write the overlay plot."""
    from theresa.config import KELVIN_OFFSET
    from theresa.plotting import plot_peak_temperature

    times_years = np.asarray(direct["times"]) / SECONDS_PER_YEAR
    # The direct run stores absolute temperatures.
    direct_celsius = np.asarray(direct["probe"]).max(axis=1) - KELVIN_OFFSET
    assembled_celsius = assembled.temperature_celsius

    difference = assembled_celsius - direct_celsius

    print("\nDuhamel cross-check: assembled (psi + step response) vs. direct simulation")
    print(f"{'t [a]':>10}{'direct':>12}{'assembled':>12}{'difference':>13}")
    for target in (0.1, 1.0, 5.0, 20.0, 100.0, 500.0, 1000.0):
        i = int(np.argmin(np.abs(times_years - target)))
        print(
            f"{times_years[i]:10.2f}{direct_celsius[i]:12.3f}{assembled_celsius[i]:12.3f}"
            f"{difference[i]:+13.4f}"
        )

    peak_direct = float(direct_celsius.max())
    peak_assembled = float(assembled_celsius.max())
    print(
        f"\npeak: direct {peak_direct:.3f} degC, assembled {peak_assembled:.3f} degC "
        f"({peak_assembled - peak_direct:+.4f} K)"
    )
    print(f"largest difference over the whole history: {np.max(np.abs(difference)):.4f} K")

    plot_peak_temperature(
        times_years,
        assembled_celsius,
        work_dir / "duhamel_check.png",
        limit_celsius=cfg["T_limit_celsius"],
        reference=(times_years, direct_celsius),
        labels=("assembled from a step response", "direct simulation"),
        title="Assembled from a step response vs. a direct simulation",
        subtitle="six-term decay source, identical mesh and boundary conditions",
    )
    print(f"plot: {work_dir / 'duhamel_check.png'}")


if __name__ == "__main__":
    raise SystemExit(main())
