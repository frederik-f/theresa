# SPDX-License-Identifier: GPL-3.0-or-later
"""Geometric effect of the emplacement depth on the step response.

Same spacings at several depths; the remaining kernel difference quantifies
the effect not covered by ``psi``.

Run::

    uv run python scripts/run_depth_study.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from theresa.config import make_config  # noqa: E402
from theresa.decay_heat import source_terms_watt  # noqa: E402
from theresa.doe import depth_study_points, run_batch  # noqa: E402
from theresa.duhamel import convolve  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "depth_study.npz"
RUNS = ROOT / "runs" / "depth_study"

FEATURES = ("canister_gap", "a_drift", "depth_m")


def main() -> int:
    """Run the depth series and write the kernels.

    :returns: exit status.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging.getLogger("theresa.runner").setLevel(logging.WARNING)
    logging.getLogger("theresa.mesh_conversion").setLevel(logging.WARNING)

    points = depth_study_points()
    logging.info("depth study: %d design points", points.shape[0])

    started = time.time()
    training = run_batch(points, FEATURES, RUNS, max_workers=3)
    training.save(OUT)
    logging.info("finished in %.1f min -> %s", (time.time() - started) / 60.0, OUT)

    report(training)
    return 0


def report(training) -> None:
    """Print the depth dependence with the analytic part removed."""
    cfg = make_config()
    amplitudes, decay_constants = source_terms_watt(cfg)

    points = training.points
    spacings = sorted({(row[0], row[1]) for row in points})

    print("\nPeak temperature rise (heating only -- the analytic depth term removed):")
    print(
        f"{'a_can':>7}{'a_drift':>9}"
        + "".join(f"{d:>12.0f} m" for d in sorted({row[2] for row in points}))
    )

    worst = 0.0
    for gap, a_drift in spacings:
        rises = []
        depths = sorted({row[2] for row in points})
        for depth in depths:
            index = int(
                np.argmin(
                    np.abs(points[:, 0] - gap)
                    + np.abs(points[:, 1] - a_drift)
                    + np.abs(points[:, 2] - depth)
                )
            )
            rise = convolve(training.times, training.kernels[index], amplitudes, decay_constants)
            rises.append(float(rise.max(axis=1).max()))
        spread = max(rises) - min(rises)
        worst = max(worst, spread)
        print(
            f"{gap:>7.1f}{a_drift:>9.1f}"
            + "".join(f"{r:>14.2f}" for r in rises)
            + f"   spread {spread:.3f} K"
        )

    print(f"\nlargest spread across depths: {worst:.3f} K")
    if worst < 0.5:
        print("=> below the 0.5 K target: depth can be treated analytically,")
        print("   and the interpolation grid stays two-dimensional.")
    else:
        print("=> not negligible: depth must join the interpolation grid as a third feature.")


if __name__ == "__main__":
    raise SystemExit(main())
