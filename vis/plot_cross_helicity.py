"""Plot the normalized cross helicity sigma_c(t) of one or more inhomogeneous RMHD runs.

Reads scalar_diagnostics.csv from either inhomogeneous RMHD equation set. With the
Elsasser energies W+- = <|z+-|^2>/4 of z+- = delta u_perp -+ delta B_perp/sqrt(4 pi rho),

    sigma_c = (W+ - W-) / (W+ + W-)

so sigma_c = +1 for a pure z+ state (what the one-wave initial conditions launch), 0 when
z+ and z- carry equal energy, and -1 for pure z-. Buoyancy is the only thing that creates
z-, so sigma_c stays exactly 1 when g = 0 and drops as z+ is reflected into z- when g != 0.
The sign is chosen so the launched wave gives +1; in these units <u_perp . b_perp> itself
equals -(W+ - W-).

Pass several runs to overlay them. Each run may be given as a scalar_diagnostics.csv or as
its run directory. Curves are labelled "g=..., K_rho0=..." from the run's input_copy.input
when it is there, otherwise by directory name; --labels overrides this.

Examples (also usable as main([...]) in Spyder):
    python vis/plot_cross_helicity.py RUN_DIRECTORY
    python vis/plot_cross_helicity.py RUN_A RUN_B RUN_C --labels "no feedback" "g=0.06" "g=0.6"
    python vis/plot_cross_helicity.py RUN_A RUN_B --output compare.png --ymin 0.9 --show
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import re
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from vis._matplotlib import finalize_figure, import_pyplot

SCALAR_CSV_NAME = "scalar_diagnostics.csv"


@dataclass
class CrossHelicitySeries:
    """sigma_c(t) of one run, with the label it is drawn under."""

    label: str
    times: np.ndarray
    sigma_c: np.ndarray


def read_scalar_csv(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit(f"{path} has no header row.")
        rows = list(reader)
    if not rows:
        raise SystemExit(f"{path} contains no data rows.")
    return {name: np.array([float(row[name]) for row in rows]) for name in reader.fieldnames}


def resolve_csv_path(path):
    """Accept either scalar_diagnostics.csv itself or the run directory that holds it."""
    path = path.expanduser().resolve()
    csv_path = path / SCALAR_CSV_NAME if path.is_dir() else path
    if not csv_path.exists():
        raise SystemExit(f"Could not find scalar diagnostics at {csv_path}.")
    return csv_path


def default_label(csv_path):
    """Return "g=..., K_rho0=..." from the run's input_copy.input, else the directory name."""
    input_copy = csv_path.parent / "input_copy.input"
    if input_copy.exists():
        text = input_copy.read_text(encoding="utf-8")
        values = {}
        for name in ("g", "K_rho0"):
            match = re.search(rf"^\s*{name}\s*=\s*([^\s#]+)", text, re.MULTILINE)
            if match:
                values[name] = match.group(1)
        if len(values) == 2:
            return f"g={values['g']}, K_rho0={values['K_rho0']}"
    return csv_path.parent.name


def calculate_cross_helicity(columns):
    """Return (times, sigma_c) from the saved Elsasser energies.

    Undefined points (W+ + W- = 0) are NaN so they are left out of the curve.
    """
    time_key = "t" if "t" in columns and "time" not in columns else "time"  # older CSVs use "t"
    missing = [name for name in (time_key, "w_plus", "w_minus") if name not in columns]
    if missing:
        raise SystemExit(f"Scalar diagnostics are missing {missing}.")
    w_plus = columns["w_plus"]
    w_minus = columns["w_minus"]
    total = w_plus + w_minus
    sigma_c = np.divide(w_plus - w_minus, total, out=np.full_like(total, np.nan, dtype=float),
                        where=total > 0.0)
    return columns[time_key], sigma_c


def load_series(paths, labels=None):
    """Read each run and return its sigma_c(t) series."""
    if labels is not None and len(labels) != len(paths):
        raise SystemExit(f"Got {len(labels)} --labels for {len(paths)} runs; give one label per run.")
    series = []
    for index, path in enumerate(paths):
        csv_path = resolve_csv_path(path)
        times, sigma_c = calculate_cross_helicity(read_scalar_csv(csv_path))
        label = default_label(csv_path) if labels is None else labels[index]
        series.append(CrossHelicitySeries(label=label, times=times, sigma_c=sigma_c))
    return series


def plot_series(runs, output_path, ymin=None, show=False):
    """Draw sigma_c(t) of every run on one set of axes."""
    plt = import_pyplot(show=show)
    fig, ax = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
    for run in runs:
        ax.plot(run.times, run.sigma_c, lw=2, label=run.label)
    ax.axhline(1.0, color="0.4", lw=1, ls=":")  # pure z+
    ax.axhline(0.0, color="0.5", lw=1, alpha=0.6)  # equal z+ and z- energy

    if ymin is None:
        # sigma_c lies in [-1, 1]. Keep 0 and 1 on the axes so a curve hugging 1 is not
        # blown up into something that looks large; --ymin zooms in on purpose.
        finite = np.concatenate([run.sigma_c[np.isfinite(run.sigma_c)] for run in runs])
        ymin = min(0.0, float(finite.min()) if finite.size else 0.0) - 0.05
    ax.set(xlabel="Time", ylabel=r"$\sigma_c = (W^+ - W^-)/(W^+ + W^-)$",
           title=r"Normalized cross helicity ($\sigma_c = 1$: pure $z^+$)",
           ylim=(ymin, 1.05))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    finalize_figure(fig, output_path=output_path, show=show, plt=plt)


def print_summary(runs):
    """Print the minimum and final sigma_c so the numbers are available without reading the plot."""
    for run in runs:
        finite = run.sigma_c[np.isfinite(run.sigma_c)]
        if finite.size:
            print(f"{run.label}: min sigma_c = {finite.min():.4f}, final sigma_c = {finite[-1]:.4f}")
        else:
            print(f"{run.label}: sigma_c undefined (W+ + W- = 0 throughout)")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", type=Path, nargs="+",
                        help="scalar_diagnostics.csv files or run directories, one per curve.")
    parser.add_argument("--labels", nargs="+", help="One legend label per run (default: g and K_rho0 from input_copy.input).")
    parser.add_argument("--ymin", type=float, help="Lower y limit (default: shows 0 and the lowest sigma_c).")
    parser.add_argument("--output", type=Path,
                        help="Image path; default: cross_helicity.png beside the CSV for one run, "
                             "or in the current directory for several.")
    parser.add_argument("--show", action="store_true", help="Show the figure after saving (e.g. in Spyder).")
    return parser


def main(argv=None) -> Path:
    args = build_parser().parse_args(argv)
    runs = load_series(args.paths, labels=args.labels)
    if args.output is not None:
        output_path = args.output.expanduser().resolve()
    elif len(args.paths) == 1:
        output_path = resolve_csv_path(args.paths[0]).with_name("cross_helicity.png")
    else:
        output_path = Path.cwd() / "cross_helicity.png"
    print_summary(runs)
    plot_series(runs, output_path, ymin=args.ymin, show=args.show)
    return output_path


if __name__ == "__main__":
    main()
