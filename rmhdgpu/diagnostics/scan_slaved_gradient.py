"""Run a series of simulations that differ only in one background gradient.

The companion plot script, `vis/plot_slaved_gradient_scan.py`, then compares each run's
saturated compressive amplitudes with the Squire et al. (arXiv:2607.08036) Eq. (47) slaving
closure. This script only generates and launches the runs.

Each compressive field feels a different combination of the background gradients, so there
is no single "gradient" to scan. With `K_b0 = g/vA^2 - chi K_p0/gamma`,

    drho    F = g/(vA^2 (1 + chi)) - K_rho0        (= -N^2/g)
    du_par  F = -K_b0
    db_par  F = -K_b0 + K_p0/gamma

so `K_rho0` moves only the density drive, `K_p0` moves only the two magnetic drives (it
cancels out of N^2 entirely), and `g` moves all three plus the explicit buoyancy term. Each
knob therefore gets its own scan, and `--dry-run` prints what each one would actually move.

`g`, `K_p0` and `K_rho0` have no command-line override, and adding one would not help: the
run driver copies the input file verbatim to `input_copy.input`, and `resolved_config.toml`
drops these three keys, so a scan driven by overrides would leave every run recording the
base file's gradients and the plot would be silently wrong. One self-contained `.input` per
point keeps that record honest.

Use:
    python -m rmhdgpu.diagnostics.scan_slaved_gradient --dry-run
    python -m rmhdgpu.diagnostics.scan_slaved_gradient --knobs K_rho0 --plot
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import subprocess
import sys

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib

from rmhdgpu.equations.rmhd_by_nokia_rho import derived_parameters
from rmhdgpu.runfile import dump_toml


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
BASE_INPUT = REPO_ROOT / "examples" / "nokias_inhomo_rho.input"
SCAN_DIR = REPO_ROOT / "examples" / "slaved_gradient_scan"
PLOT_SCRIPT = REPO_ROOT / "vis" / "plot_slaved_gradient_scan.py"

# Edit these to change the scan. Values are absolute, not multipliers on the base case.
# Against the base input (vA = 1, chi = 1, K_rho0 = 10.5, g = 0.6, K_p0 = 1.0) these give
# |F_drho| from 0.7 to 20.7, and |F_du_par| from 0.6 to 4.2 with N^2 left alone.
SCANS = {
    "K_rho0": [1.0, 2.0, 4.0, 7.0, 10.5, 15.0, 21.0],
    "K_p0": [-4.0, -2.0, -1.0, 0.0, 2.0, 4.0, 8.0],
    "g": [0.05, 0.1, 0.2, 0.4, 0.6, 1.0, 1.5],
}

FIELDS = ("drho", "du_par", "db_par")

# The fields whose drive each knob moves (see the table above), so each scan's figure only
# shows panels that can have a slope. `test_slaved_gradient_scan.py` checks this against
# `background_forcings`.
FIELDS_DRIVEN_BY = {"K_rho0": ("drho",), "K_p0": ("du_par", "db_par"), "g": FIELDS}


def background_forcings(p):
    """The x components of Eq. (46), for a straight field and no mean flow.

    Deliberately a copy of `vis.plot_slaved_projection.forcings` rather than an import: a
    package module should not depend on the plotting layer. `test_slaved_gradient_scan.py`
    asserts the two stay identical, so the duplication cannot drift unnoticed.
    """

    return {
        # Equivalent to -N_sq/g, but also defined when g = 0.
        "drho": p.g / (p.vA**2 * (1.0 + p.chi)) - p.K_rho0,
        "du_par": -p.K_b0,
        "db_par": -p.K_b0 + p.K_p0 / p.gamma,
    }


def load_base_document(path=BASE_INPUT):
    """Parse the base `.input` file, which supplies everything the scan holds fixed."""

    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"Missing base input file {path}.")
    with path.open("rb") as handle:
        document = tomllib.load(handle)
    missing = [key for key in ("vA", "cs2_over_vA2", "g", "K_p0", "K_rho0")
               if key not in document.get("physics", {})]
    if missing:
        raise SystemExit(f"{path} [physics] is missing {missing}; the scan needs all of them.")
    if "seed" not in document.get("initial_condition", {}).get("parameters", {}):
        print(f"Warning: {path.name} sets no initial-condition seed, so the scan points will "
              "differ by more than the gradient.")
    return document


def point_document(base_document, parameter, value, run_dir, *, snapshots=False):
    """Copy the base document, override one [physics] key, and pin the output directory."""

    if parameter not in base_document.get("physics", {}):
        raise ValueError(
            f"{parameter!r} is not in the base [physics] block "
            f"({sorted(base_document.get('physics', {}))}); it would be silently ignored."
        )
    document = deepcopy(base_document)
    document["physics"][parameter] = float(value)
    # Absolute: a relative output_dir resolves against the input file's own directory, which
    # would drop every run next to the generated inputs.
    document["output_dir"] = Path(run_dir).resolve().as_posix()
    document["title"] = f"{base_document.get('title', 'slaved gradient scan')} | {parameter} = {value:g}"
    if not snapshots:
        # The scan is read from scalar diagnostics, so the HDF5 snapshots are pure cost.
        document.setdefault("output", {})["t_out_full"] = 0.0
    return document


def write_point_input(document, path):
    """Serialize one scan point with the solver's own TOML writer."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_toml(document), encoding="utf-8")
    return path


def describe_point(document):
    """Return the derived parameters and the three forcings for one scan point."""

    p = derived_parameters(document["physics"])
    return p, background_forcings(p)


def marginal_K_rho0(p):
    """`K_rho0` at which N^2 changes sign, which is also where F_drho vanishes."""

    return p.g / (p.vA**2 * (1.0 + p.chi))


def report_scan(parameter, values, base_document, *, snapshots=False):
    """Print each point's stability and drives; the cheap way to choose scan values."""

    print(f"\n=== {parameter} scan: {len(values)} points ===")
    header = f"{'#':>3} {parameter:>10} {'N^2':>12} " + " ".join(f"{'|F_' + n + '|':>12}" for n in FIELDS)
    print(header)
    rows = []
    for index, value in enumerate(values):
        document = point_document(base_document, parameter, value, SCAN_DIR, snapshots=snapshots)
        p, forcing = describe_point(document)
        rows.append((value, p, forcing))
        flag = "" if p.N_sq > 0.0 else "   <- unstable stratification"
        drives = " ".join(f"{abs(forcing[name]):12.4g}" for name in FIELDS)
        print(f"{index:>3} {value:>10.4g} {p.N_sq:>12.4g} {drives}{flag}")

    for name in FIELDS:
        spread = {round(abs(forcing[name]), 12) for _, _, forcing in rows}
        if len(spread) == 1:
            print(f"Note: |F_{name}| is the same at every point, so this scan does not drive "
                  f"{name}. Scan a different knob to move it.")
        elif min(spread) == 0.0:
            print(f"Note: |F_{name}| passes through zero in this scan; that point has no drive "
                  "and is dropped from the logarithmic panels.")
    if parameter == "K_rho0":
        # Only a fixed boundary while g, vA and chi are held; in a g scan it moves per point.
        print(f"Marginal stratification sits at K_rho0 = {marginal_K_rho0(rows[0][1]):.4g} "
              "(N^2 = 0, and F_drho = 0 at the same place).")
    return rows


def run_point(input_path, run_dir, *, force=False):
    """Launch one point, skipping it when its scalar diagnostics already exist."""

    run_dir = Path(run_dir)
    existing = run_dir / "scalar_diagnostics.csv"
    if existing.is_file() and existing.stat().st_size > 0:
        if not force:
            print(f"  Skipping {run_dir.name}: already has scalar diagnostics.")
            return True
        snapshots = run_dir / "fullfields"
        if snapshots.is_dir():
            # Snapshots from a longer previous run would contaminate full-field analysis.
            print(f"  Removing stale snapshots in {snapshots}.")
            shutil.rmtree(snapshots)

    command = [sys.executable, "-m", "rmhdgpu.run", str(input_path), "--output-dir", str(run_dir)]
    print(f"  Running {run_dir.name} ...")
    try:
        subprocess.run(command, cwd=REPO_ROOT, check=True)
    except subprocess.CalledProcessError as error:
        # One bad point must not abort the scan.
        print(f"  Run failed for {run_dir.name}: {error}")
        return False
    return True


def run_scan(parameter, values, base_document, scan_dir, *, snapshots=False, force=False,
             allow_unstable=False):
    """Generate and launch every point of one scan; return the run directories that exist."""

    inputs_dir = Path(scan_dir) / parameter / "inputs"
    runs_dir = Path(scan_dir) / parameter / "runs"
    completed = []
    for index, value in enumerate(values):
        # Index-based names: float values make directory names that sort badly.
        stem = f"{parameter}_{index:02d}"
        run_dir = runs_dir / stem
        document = point_document(base_document, parameter, value, run_dir, snapshots=snapshots)
        p, _ = describe_point(document)
        if p.N_sq <= 0.0 and not allow_unstable:
            print(f"  Skipping {stem} ({parameter} = {value:g}): N^2 = {p.N_sq:.4g} <= 0 is "
                  f"convectively unstable and never saturates. Marginal K_rho0 is "
                  f"{marginal_K_rho0(p):.4g}; pass --allow-unstable to run it anyway.")
            continue
        input_path = write_point_input(document, inputs_dir / f"{stem}.input")
        if run_point(input_path, run_dir, force=force):
            completed.append(run_dir)
    return completed


def plot_scan(parameter, run_dirs, scan_dir):
    """Hand the finished runs to the plotting script as a separate process.

    Deliberately a subprocess rather than an import: `rmhdgpu` should not depend on `vis`,
    and a plotting failure must not be able to damage a completed scan.
    """

    if not run_dirs:
        print(f"No completed {parameter} runs to plot.")
        return None
    output = Path(scan_dir) / parameter / "slaved_gradient_scan.png"
    command = [
        sys.executable, str(PLOT_SCRIPT), *[str(path) for path in run_dirs],
        "--parameter", parameter,
        "--fields", *FIELDS_DRIVEN_BY[parameter],
        "--output", str(output),
        "--summary-output", str(output.with_suffix(".csv")),
    ]
    try:
        subprocess.run(command, cwd=REPO_ROOT, check=True)
    except subprocess.CalledProcessError as error:
        print(f"Plotting failed for the {parameter} scan: {error}")
        return None
    return output


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--knobs", nargs="+", choices=tuple(SCANS), default=list(SCANS),
                        help="Which background gradients to scan; each gets its own series.")
    parser.add_argument("--base", type=Path, default=BASE_INPUT,
                        help="Base .input file supplying everything the scan holds fixed.")
    parser.add_argument("--scan-dir", type=Path, default=SCAN_DIR,
                        help="Directory for the generated inputs, runs and figures.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print each point's N^2 and drives without running anything.")
    parser.add_argument("--snapshots", action="store_true",
                        help="Keep the base file's t_out_full so full fields are written.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run points that already have scalar diagnostics.")
    parser.add_argument("--allow-unstable", action="store_true",
                        help="Run points with N^2 <= 0, which do not saturate.")
    parser.add_argument("--plot", action="store_true",
                        help="Run the plotting script on each finished scan.")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    base_document = load_base_document(args.base)

    if args.dry_run:
        for parameter in args.knobs:
            report_scan(parameter, SCANS[parameter], base_document, snapshots=args.snapshots)
        return {}

    completed = {}
    for parameter in args.knobs:
        report_scan(parameter, SCANS[parameter], base_document, snapshots=args.snapshots)
        completed[parameter] = run_scan(
            parameter, SCANS[parameter], base_document, args.scan_dir,
            snapshots=args.snapshots, force=args.force, allow_unstable=args.allow_unstable,
        )

    for parameter, run_dirs in completed.items():
        if args.plot:
            plot_scan(parameter, run_dirs, args.scan_dir)
        elif run_dirs:
            paths = " ".join(str(path) for path in run_dirs)
            print(f"\nPlot the {parameter} scan with:\n"
                  f"  python {PLOT_SCRIPT} {paths} --parameter {parameter} "
                  f"--fields {' '.join(FIELDS_DRIVEN_BY[parameter])}")
    return completed


if __name__ == "__main__":
    main()
