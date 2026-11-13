# SPDX-License-Identifier: GPL-3.0-or-later
r"""Undisturbed steady state.

The source-free field with :math:`T = T_{top}` at the top and
:math:`T = T_{top} + g H` at the bottom is

.. math::

    T_{ss}(x) = T_{top} + g \, \psi(x),

exactly, where :math:`\psi` solves the same problem with boundary values
:math:`0` and :math:`H`. :math:`\psi` depends only on geometry and materials;
one solve per design point covers all surface temperatures and gradients. In a
homogeneous medium :math:`\psi = -z`.
"""

from __future__ import annotations

import re
import threading
from pathlib import Path

import numpy as np

from .config import KELVIN_OFFSET, domain_bottom_z
from .prj_builder import build_prj_xml
from .runner import run_ogs

#: Point-data name of :math:`\psi` on the domain mesh.
PSI_FIELD_NAME = "psi"


def compute_psi(
    cfg: dict, work_dir: Path, meshes: dict[str, Path], cancel: threading.Event | None = None
) -> np.ndarray:
    r"""Solve for :math:`\psi` and store it on the domain mesh.

    :param cfg: a configuration from :mod:`theresa.config`.
    :param work_dir: directory for the project file and results.
    :param meshes: physical-group name to VTU path.
    :param cancel: passed to :func:`theresa.runner.run_ogs`.
    :returns: :math:`\psi` [m] per node of the domain mesh.
    """
    import pyvista as pv

    work_dir.mkdir(parents=True, exist_ok=True)
    out_dir = work_dir / "steady_state"

    height = -domain_bottom_z(cfg)
    prj_text = build_prj_xml(
        cfg,
        meshes,
        prj_dir=work_dir,
        mode="steady",
        top_value=0.0,
        bottom_value=height,
        output_domain=True,
    )
    prj_path = work_dir / "psi.prj"
    prj_path.write_text(prj_text, encoding="ISO-8859-1")

    run_ogs(prj_path, out_dir, cancel=cancel)

    psi = np.asarray(pv.read(str(_latest_result(out_dir))).point_data["temperature"], dtype=float)

    domain = pv.read(str(meshes["domain"]))
    if domain.n_points != psi.size:
        raise RuntimeError(
            f"steady-state result has {psi.size} nodes but the domain mesh has "
            f"{domain.n_points} -- meshes are out of sync"
        )
    domain.point_data[PSI_FIELD_NAME] = psi
    domain.save(str(meshes["domain"]))
    return psi


def steady_temperature_kelvin(cfg: dict, psi: np.ndarray) -> np.ndarray:
    r"""Undisturbed temperature [K] from :math:`\psi`."""
    top_kelvin = cfg["surface_temperature_celsius"] + KELVIN_OFFSET
    return top_kelvin + cfg["geothermal_gradient_k_per_m"] * np.asarray(psi, dtype=float)


def _latest_result(out_dir: Path) -> Path:
    """Result file of the last time step in ``out_dir``."""
    candidates = sorted(out_dir.glob("result_domain_ts_*.vtu"))
    if not candidates:
        raise RuntimeError(f"no steady-state result files in {out_dir}")

    def time_step(path: Path) -> int:
        match = re.search(r"_ts_(\d+)_", path.name)
        return int(match.group(1)) if match else -1

    return max(candidates, key=time_step)


def write_initial_temperature(
    cfg: dict, meshes: dict[str, Path], psi: np.ndarray, field_name: str = "T_initial"
) -> str:
    """Store the undisturbed temperature [K] on the domain mesh.

    Initial condition of a ``direct`` run.

    :returns: ``field_name``, for use as an OGS ``MeshNode`` parameter.
    """
    import pyvista as pv

    domain = pv.read(str(meshes["domain"]))
    domain.point_data[field_name] = steady_temperature_kelvin(cfg, psi)
    domain.save(str(meshes["domain"]))
    return field_name
