# SPDX-License-Identifier: GPL-3.0-or-later
"""Command-line interface.

::

    theresa simulate   # one design point, solved directly with OGS
    theresa train      # generate training data and fit the surrogate
    theresa predict    # evaluate the surrogate
    theresa validate   # surrogate against direct simulations
    theresa sobol      # Sobol indices of the peak temperature
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

from .config import (
    MATERIAL_ALIASES,
    emplacement_temperature_celsius,
    make_config,
    run_directory_name,
)
from .predict import DEFAULT_SURROGATE_PATH, load_surrogate, predict

#: Default locations of runs and trained artifacts.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS = PROJECT_ROOT / "runs"
DEFAULT_TRAINING_SET = PROJECT_ROOT / "data" / "training_set.npz"

#: Geometry held fixed in training and recorded on the surrogate.
FIXED_GEOMETRY_KEYS = ("l_can", "r_can", "drift_diameter", "lining_thickness")


def build_arg_parser() -> argparse.ArgumentParser:
    """Argument parser of :func:`main`."""
    parser = argparse.ArgumentParser(
        prog="theresa",
        description=(
            "Transient temperature field around a Pollux canister in drift "
            "emplacement: direct OGS simulation and a surrogate model of it."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="show OGS and meshing output")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_design(target: argparse.ArgumentParser) -> None:
        target.add_argument(
            "--canister-spacing",
            type=float,
            help=(
                "clear distance between the end faces of adjacent canisters [m]; "
                "the pitch follows as --length + --canister-spacing"
            ),
        )
        target.add_argument("--a-drift", type=float, help="drift spacing, axis to axis [m]")
        target.add_argument(
            "--length",
            type=float,
            help="canister length [m]; must equal the trained length unless interpolated",
        )
        target.add_argument("--depth", type=float, help="emplacement depth [m]")

    def add_physics(target: argparse.ArgumentParser) -> None:
        target.add_argument(
            "--surface-temperature", type=float, help="temperature at ground level [degC]"
        )
        target.add_argument("--gradient", type=float, help="geothermal gradient [K/m], e.g. 0.03")
        target.add_argument(
            "--fuel-elements", type=int, help="canister loading, as a package count"
        )
        target.add_argument(
            "--canister-power",
            type=float,
            metavar="W",
            help="canister loading, as thermal power at emplacement [W]",
        )
        target.add_argument(
            "--interim-storage", type=float, help="interim storage before emplacement [a]"
        )
        target.add_argument(
            "--limit",
            type=float,
            metavar="DEGC",
            help="design temperature limit the result is judged against [degC]",
        )
        target.add_argument(
            "--decay-p",
            type=float,
            nargs="+",
            metavar="KW",
            help="decay-heat amplitudes [kW per package]; any number of terms",
        )
        target.add_argument(
            "--decay-half-life",
            type=float,
            nargs="+",
            metavar="A",
            help="matching half-lives [a]",
        )

    simulate = subparsers.add_parser(
        "simulate", help="solve one design point directly with OGS (minutes)"
    )
    add_design(simulate)
    add_physics(simulate)
    simulate.add_argument("--work-dir", type=Path, help="run directory")
    simulate.add_argument(
        "--keep-results", action="store_true", help="keep the raw OGS time series"
    )

    train = subparsers.add_parser("train", help="generate training data and fit the surrogate")
    train.add_argument(
        "--points",
        type=int,
        default=400,
        help="number of Latin-hypercube training designs (default: 400)",
    )
    train.add_argument("--seed", type=int, default=0, help="Latin-hypercube seed")
    train.add_argument("--depth", type=float, help="fixed emplacement depth for the runs [m]")
    train.add_argument("--workers", type=int, default=3, help="concurrent OGS runs")
    train.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS / "training")
    train.add_argument("--training-set", type=Path, default=DEFAULT_TRAINING_SET)
    train.add_argument("--out", type=Path, default=DEFAULT_SURROGATE_PATH)
    train.add_argument(
        "--reuse",
        action="store_true",
        help="fit from an existing training set instead of simulating again",
    )

    predict_parser = subparsers.add_parser(
        "predict", help="evaluate the trained surrogate (milliseconds)"
    )
    add_design(predict_parser)
    add_physics(predict_parser)
    predict_parser.add_argument("--surrogate", type=Path, help="trained surrogate file")
    predict_parser.add_argument("--plot", type=Path, help="write the temperature history plot here")
    predict_parser.add_argument("--csv", type=Path, help="write the history as CSV here")

    validate = subparsers.add_parser(
        "validate", help="check the surrogate against directly simulated designs"
    )
    validate.add_argument("--points", type=int, default=8, help="number of hold-out designs")
    validate.add_argument("--seed", type=int, default=0)
    validate.add_argument("--workers", type=int, default=3)
    validate.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS / "validation")
    validate.add_argument("--surrogate", type=Path, help="trained surrogate file")

    sobol = subparsers.add_parser(
        "sobol", help="variance-based sensitivity (Sobol indices) of the peak temperature"
    )
    sobol.add_argument("--surrogate", type=Path, help="trained surrogate file")
    sobol.add_argument(
        "--points",
        type=int,
        default=1024,
        help="base sample size N; the cost is N (d + 2) surrogate evaluations",
    )
    sobol.add_argument("--seed", type=int, default=0)
    sobol.add_argument(
        "--scope",
        choices=("materials", "all"),
        default="materials",
        help=("materials: material properties and site parameters; all: additionally the spacings"),
    )
    sobol.add_argument(
        "--gradient",
        nargs=2,
        type=float,
        default=(0.02, 0.07),
        metavar=("LOW", "HIGH"),
        help="geothermal gradient range [K/m]",
    )
    sobol.add_argument(
        "--surface-temperature",
        nargs=2,
        type=float,
        default=(6.0, 14.0),
        metavar=("LOW", "HIGH"),
        help="ground-level temperature range [degC]",
    )
    sobol.add_argument(
        "--exact",
        action="store_true",
        help="convolve every sample individually instead of combining mode convolutions",
    )
    sobol.add_argument("--csv", type=Path, help="write the indices here as CSV")

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns the exit code."""
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(message)s", handlers=[logging.StreamHandler(sys.stdout)]
    )
    if not args.verbose:
        for noisy in ("theresa.runner", "theresa.mesh_conversion"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    handlers = {
        "simulate": _simulate,
        "train": _train,
        "predict": _predict,
        "validate": _validate,
        "sobol": _sobol,
    }
    try:
        return handlers[args.command](args)
    except (ValueError, FileNotFoundError) as error:
        print(f"theresa {args.command}: error: {error}", file=sys.stderr)
        return 2


def _sobol(args: argparse.Namespace) -> int:
    """Sobol indices of the peak temperature."""
    from .sobol import sobol_indices

    model = load_surrogate(args.surrogate)

    # Interpolated inputs over their trained range.
    hull = model.trained_ranges()

    chosen = [
        name for name in model.feature_names if args.scope == "all" or name in MATERIAL_ALIASES
    ]
    if not chosen:
        print(f"the surrogate has no material features to vary: {model.feature_names}")
        return 1

    ranges = {name: hull[name] for name in chosen}
    ranges["geothermal_gradient_k_per_m"] = tuple(args.gradient)
    ranges["surface_temperature_celsius"] = tuple(args.surface_temperature)

    result = sobol_indices(model, ranges, n_base=args.points, seed=args.seed, exact=args.exact)

    print(f"inputs varied ({len(result.names)}), uniform and independent:")
    for name in result.names:
        low, high = result.ranges[name]
        print(f"  {name:28s} {low:>10.4g} .. {high:<10.4g}")
    print(f"\npeak temperature: mean {result.mean:.2f} degC, standard deviation {result.std:.2f} K")
    print(f"{result.n_evaluations} surrogate evaluations\n")
    print(result.table())
    print(f"\ninteraction share (1 - sum S_i): {result.interaction_share:.3f}")

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        rows = ["input,low,high,first_order,first_order_stderr,total,total_stderr"]
        for i, name in enumerate(result.names):
            low, high = result.ranges[name]
            rows.append(
                f"{name},{low!r},{high!r},{result.first_order[i]!r},"
                f"{result.first_order_stderr[i]!r},{result.total[i]!r},{result.total_stderr[i]!r}"
            )
        args.csv.write_text("\n".join(rows) + "\n")
        print(f"-> {args.csv}")
    return 0


def _config_overrides(args: argparse.Namespace) -> dict:
    """Configuration overrides from the shared options."""
    mapping = {
        "canister_spacing": "canister_gap",
        "a_drift": "a_drift",
        "length": "l_can",
        "depth": "depth_m",
        "surface_temperature": "surface_temperature_celsius",
        "gradient": "geothermal_gradient_k_per_m",
        "fuel_elements": "n_fuel_elements",
        "interim_storage": "interim_storage_years",
        "limit": "T_limit_celsius",
    }
    overrides = {}
    for option, key in mapping.items():
        value = getattr(args, option, None)
        if value is not None:
            overrides[key] = value

    p_kw = getattr(args, "decay_p", None)
    half_lives = getattr(args, "decay_half_life", None)
    if (p_kw is None) != (half_lives is None):
        raise SystemExit("--decay-p and --decay-half-life must be given together")
    if p_kw is not None:
        if len(p_kw) != len(half_lives):
            raise SystemExit(
                f"--decay-p has {len(p_kw)} values but --decay-half-life has {len(half_lives)}"
            )
        overrides["decay_heat_p_kw"] = list(p_kw)
        overrides["decay_heat_t_half_years"] = list(half_lives)
        overrides["spectrum"] = "custom"
    return overrides


def _simulate(args: argparse.Namespace) -> int:
    """Solve one design point directly."""
    from .pipeline import case_subtitle, run_case
    from .plotting import plot_peak_temperature

    cfg = make_config(**_config_overrides(args))
    if getattr(args, "canister_power", None) is not None:
        from .decay_heat import scaled_to_power

        cfg = scaled_to_power(cfg, args.canister_power)
    work_dir = args.work_dir or (DEFAULT_RUNS / "direct" / run_directory_name(cfg))

    print(f"simulating {case_subtitle(cfg)}")
    started = time.time()
    result = run_case(cfg, work_dir, keep_results=args.keep_results)
    print(
        f"solved in {(time.time() - started) / 60.0:.1f} min ({result.mesh_stats['n_nodes']} nodes)"
    )

    print(f"\npeak temperature   {result.peak_celsius:8.1f} degC")
    print(f"time of peak       {result.peak_time_years:8.1f} a")
    print(f"undisturbed rock   {emplacement_temperature_celsius(cfg):8.1f} degC")
    print(f"at 1000 a          {result.temperature_celsius[-1]:8.1f} degC")
    print(f"probe-grid deficit {100 * result.probe_deficit:8.3f} %  (vs. the exact wall maximum)")

    plot_path = plot_peak_temperature(
        result.times_years,
        result.temperature_celsius,
        work_dir / "temperature.png",
        limit_celsius=cfg["T_limit_celsius"],
        undisturbed_celsius=emplacement_temperature_celsius(cfg),
        subtitle=case_subtitle(cfg),
    )
    print(f"\nplot: {plot_path}")
    return 0


def _train(args: argparse.Namespace) -> int:
    """Generate the training set and fit the surrogate."""
    from .doe import RECIPROCAL_FEATURES, TRAINING_FEATURES, TrainingSet, latin_hypercube, run_batch
    from .surrogate import fit

    defaults = make_config()
    features = TRAINING_FEATURES
    base = {"depth_m": args.depth} if args.depth is not None else None
    fixed_depth = args.depth if args.depth is not None else defaults["depth_m"]

    if args.reuse:
        training = TrainingSet.load(args.training_set)
        print(f"reusing {training.points.shape[0]} design points from {args.training_set}")
    else:
        points = latin_hypercube(features, args.points, seed=args.seed, margin=0.0)
        print(f"training on {points.shape[0]} design points, {args.workers} at a time")
        started = time.time()
        training = run_batch(
            points,
            features,
            args.runs_dir,
            base_config=base,
            max_workers=args.workers,
            checkpoint=args.training_set.with_suffix(".partial.npz"),
        )
        training.save(args.training_set)
        print(f"simulated in {(time.time() - started) / 60.0:.1f} min -> {args.training_set}")

    print(
        f"probe-grid deficit over the training set: "
        f"{100 * training.probe_deficits.max():.3f} % worst case"
    )

    model = fit(
        training.times,
        training.points,
        training.kernels,
        training.psis,
        training.feature_names,
        reciprocal=tuple(RECIPROCAL_FEATURES[name] for name in training.feature_names),
        fixed_depth_m=fixed_depth,
        fixed_design={
            key: defaults[key] for key in FIXED_GEOMETRY_KEYS if key not in training.feature_names
        },
    )
    model.save(args.out)
    print(f"surrogate: {len(model.kernel_models)} kernel modes -> {args.out}")
    return 0


def _predict(args: argparse.Namespace) -> int:
    """Evaluate the trained surrogate."""
    defaults = make_config()
    started = time.perf_counter()
    prediction = predict(
        canister_gap_m=args.canister_spacing
        if args.canister_spacing is not None
        else defaults["canister_gap"],
        a_drift_m=args.a_drift if args.a_drift is not None else defaults["a_drift"],
        canister_length_m=args.length,
        depth_m=args.depth,
        surface_temperature_c=args.surface_temperature,
        geothermal_gradient_k_per_m=args.gradient,
        decay_p_kw=args.decay_p,
        decay_t_half_years=args.decay_half_life,
        n_fuel_elements=args.fuel_elements,
        canister_power_w=args.canister_power,
        interim_storage_years=args.interim_storage,
        limit_celsius=args.limit,
        surrogate=args.surrogate,
    )
    elapsed_ms = 1000.0 * (time.perf_counter() - started)

    print(prediction.summary())
    print(f"\nevaluated in {elapsed_ms:.0f} ms")

    if args.plot:
        print(f"plot: {prediction.plot(args.plot)}")
    if args.csv:
        print(f"csv:  {prediction.to_csv(args.csv)}")
    return 0


def _validate(args: argparse.Namespace) -> int:
    """Simulate hold-out designs and compare them with the surrogate."""
    from .doe import latin_hypercube, run_batch
    from .pipeline import temperature_history

    model = load_surrogate(args.surrogate)
    features = tuple(model.feature_names)

    points = latin_hypercube(features, args.points, seed=args.seed)
    print(f"validating on {points.shape[0]} hold-out designs")
    base = {"depth_m": model.fixed_depth_m} if "depth_m" not in features else None
    truth = run_batch(points, features, args.runs_dir, base_config=base, max_workers=args.workers)

    header = "".join(f"{name:>13}" for name in features)
    print(f"\n{header}{'simulated':>12}{'surrogate':>12}{'error [K]':>12}")
    errors = []
    for i, point in enumerate(truth.points):
        overrides = {name: float(value) for name, value in zip(features, point, strict=True)}
        cfg = make_config(base=base, **overrides)

        simulated = temperature_history(
            cfg, truth.times, truth.kernels[i], truth.psis[i]
        ).temperature_celsius
        # All material features of the design.
        materials = {name: overrides[name] for name in features if name in MATERIAL_ALIASES}
        prediction = predict(
            canister_gap_m=cfg["canister_gap"],
            a_drift_m=cfg["a_drift"],
            canister_length_m=cfg["l_can"] if "l_can" in features else None,
            depth_m=cfg["depth_m"],
            materials=materials or None,
            surrogate=model,
        )

        error = prediction.peak_celsius - float(simulated.max())
        errors.append(error)
        values = "".join(f"{float(value):>13.2f}" for value in point)
        print(
            f"{values}{float(simulated.max()):>12.2f}"
            f"{prediction.peak_celsius:>12.2f}{error:>+12.3f}"
        )

    worst = float(np.max(np.abs(errors)))
    print(f"\nworst peak-temperature error: {worst:.3f} K")
    print("target is 0.5 K; " + ("met." if worst < 0.5 else "NOT met."))
    return 0 if worst < 0.5 else 1


if __name__ == "__main__":  # pragma: no cover - see __main__.py
    raise SystemExit(main())
