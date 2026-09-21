"""Compare measured direct-compressive-feedback heating with the slaved closure.

Reads scalar_diagnostics.csv from either inhomogeneous RMHD equation set.
For a straight field with no mean flow, Squire et al. (arXiv:2607.08036,
Eqs. 18, 70-71) gives:

    measured:   Q = q_dcf (plus branch) or q_ccr_source (minus branch)
    predicted:  Q = W * N_sq / omega_nl,   omega_nl = 2*sqrt(W)*<k_perp>

Positive measured Q means buoyancy removes energy from the pump wave.
--chi-a replaces the measured turbulence strength with a fixed value:
Q = W*N_sq / (chi_A*vA*<k_parallel>). The measured-chi prediction is then
shown for comparison. --l-perp adds a separate fixed-outer-scale estimate.
The mean measured/predicted ratio is a fitted prefactor, not a pass/fail test.

Examples (also usable as main([...]) in Spyder):
    python vis/plot_dcf.py RUN_DIRECTORY/scalar_diagnostics.csv
    python vis/plot_dcf.py RUN_DIRECTORY/scalar_diagnostics.csv --chi-a 1 --tmin 1
    python vis/plot_dcf.py RUN_DIRECTORY/scalar_diagnostics.csv --l-perp 0.5 --show
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from vis._matplotlib import finalize_figure, import_pyplot


# Instantaneous negative buoyancy work on the selected Elsasser branch.
WORK_COLUMNS = {"plus": "q_dcf", "minus": "q_ccr_source"}


@dataclass
class RunSeries:
    """Heating curves and their summary, ready to plot."""

    times: np.ndarray
    q_measured: np.ndarray
    q_predicted: np.ndarray
    q_reference: np.ndarray | None
    q_fixed_scale: np.ndarray | None
    branch: str
    n_sq: float
    mean_prefactor: float
    mean_chi_a: float
    chi_a: float | None
    l_perp: float | None
    tmin: float | None


def read_scalar_csv(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit(f"{path} has no header row.")
        rows = list(reader)
    if not rows:
        raise SystemExit(f"{path} contains no data rows.")
    return {name: np.array([float(row[name]) for row in rows]) for name in reader.fieldnames}


def safe_ratio(numerator, denominator):
    """Leave undefined ratios as NaN so they are omitted from curves and means."""
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan, dtype=float),
                     where=np.abs(denominator) > 0.0)


def calculate_series(columns, branch="auto", chi_a=None, l_perp=None, tmin=None):
    """Select the pump, calculate the heating estimates, and average their ratio."""
    time_key = "time" if "time" in columns else "t"
    required = [time_key, "N_sq"]
    required += ["w_plus", "w_minus"] if branch == "auto" else [f"w_{branch}"]
    missing = [name for name in required if name not in columns]
    if missing:
        raise SystemExit(f"Scalar diagnostics are missing {missing}.")
    if branch == "auto":
        # Use the whole run, including times before tmin; ties select plus.
        branch = "plus" if columns["w_plus"].mean() >= columns["w_minus"].mean() else "minus"

    energy_column = f"w_{branch}"
    work_column = WORK_COLUMNS[branch]
    missing = [name for name in (f"{energy_column}_kperp", work_column) if name not in columns]
    if missing:
        raise SystemExit(f"The {branch} branch needs these missing columns: {missing}.")
    times = columns[time_key]
    energy = columns[energy_column]
    kperp = columns[f"{energy_column}_kperp"]
    kparallel = columns.get(f"{energy_column}_kprl")
    q_measured = columns[work_column]
    n_sq = float(columns["N_sq"][0])
    v_a = float(columns["vA"][0]) if "vA" in columns else 1.0

    # W = <|z|^2>/4, so z_rms = 2*sqrt(W) and omega_nl = z_rms/l_perp.
    omega_nl = 2.0 * np.sqrt(energy) * kperp
    q_from_measured_chi = safe_ratio(energy * n_sq, omega_nl)
    chi_measured = None if kparallel is None else safe_ratio(omega_nl, v_a * kparallel)

    if chi_a is not None:
        if kparallel is None:
            raise SystemExit(f"--chi-a needs the {energy_column}_kprl column.")
        if chi_a == 0.0:
            raise SystemExit("--chi-a must be nonzero.")
        q_predicted = safe_ratio(energy * n_sq, chi_a * v_a * kparallel)
    elif kparallel is not None:
        # Equivalent to W*N_sq/omega_nl for nonzero k_parallel. Keep undefined
        # points as NaN when the saved parallel scale cannot define chi_A.
        q_predicted = safe_ratio(energy * n_sq, chi_measured * v_a * kparallel)
    else:
        q_predicted = q_from_measured_chi  # Older CSVs have no parallel-wavenumber column.

    q_reference = q_from_measured_chi if chi_a is not None else None
    q_fixed_scale = None
    if l_perp is not None:
        q_fixed_scale = 0.5 * n_sq * l_perp * np.sqrt(np.maximum(energy, 0.0))

    # tmin affects these averages only; all saved times remain in the curves.
    ratio = safe_ratio(q_measured, q_predicted)
    usable = np.isfinite(ratio)
    if tmin is not None:
        usable &= times >= tmin
    mean_prefactor = float(ratio[usable].mean()) if usable.any() else np.nan
    mean_chi_a = np.nan
    if chi_measured is not None:
        usable_chi = usable & np.isfinite(chi_measured)
        if usable_chi.any():
            mean_chi_a = float(chi_measured[usable_chi].mean())

    return RunSeries(
        times=times, q_measured=q_measured, q_predicted=q_predicted,
        q_reference=q_reference, q_fixed_scale=q_fixed_scale, branch=branch, n_sq=n_sq,
        mean_prefactor=mean_prefactor, mean_chi_a=mean_chi_a,
        chi_a=chi_a, l_perp=l_perp, tmin=tmin,
    )


def plot_series(run, output_path, show=False):
    """Draw the measured heating and closure estimates on one set of axes."""
    plt = import_pyplot(show=show)
    fig, ax = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
    prediction_label = (r"Closure: measured $\chi_A$" if run.chi_a is None
                        else rf"Closure: assumed $\chi_A={run.chi_a:g}$")

    ax.plot(run.times, run.q_measured, color="black", lw=2, label="Measured DCF heating")
    ax.plot(run.times, run.q_predicted, color="tab:red", ls="--", lw=1.8, label=prediction_label)
    if run.q_reference is not None:
        ax.plot(run.times, run.q_reference, color="tab:green", ls="--", lw=1.5,
                label=r"Reference: measured $\chi_A$")
    if run.q_fixed_scale is not None:
        ax.plot(run.times, run.q_fixed_scale, color="tab:orange", ls=":", lw=1.5,
                label=rf"Reference: fixed $\ell_\perp={run.l_perp:g}$")
    ax.axhline(0.0, color="0.5", lw=1, alpha=0.6)
    if run.tmin is not None:
        ax.axvline(run.tmin, color="0.4", lw=1, ls=":")

    pump_label = r"z^+" if run.branch == "plus" else r"z^-"
    stability = "stable" if run.n_sq > 0 else "unstable" if run.n_sq < 0 else "neutral"
    title = rf"DCF heating: pump ${pump_label}$, $N^2={run.n_sq:.4g}$ ({stability})"
    if np.isfinite(run.mean_prefactor):
        title += f"\nMean measured / closure = {run.mean_prefactor:.3g}"
        if run.tmin is not None:
            title += rf" for $t \geq {run.tmin:g}$"
    if np.isfinite(run.mean_chi_a):
        title += rf"; measured $\chi_A={run.mean_chi_a:.3g}$"
    ax.set(xlabel="Time", ylabel="Heating rate", title=title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    finalize_figure(fig, output_path=output_path, show=show, plt=plt)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_path", type=Path, help="Path to scalar_diagnostics.csv.")
    parser.add_argument("--branch", choices=("auto", "plus", "minus"), default="auto",
                        help="Pump branch; auto selects the larger mean energy over the whole run.")
    parser.add_argument("--chi-a", type=float,
                        help="Assumed turbulence strength (e.g. 1); default uses measured chi_A.")
    parser.add_argument("--l-perp", type=float, help="Add a reference curve using this fixed outer scale.")
    parser.add_argument("--tmin", type=float, help="Start time for reported means; all times are plotted.")
    parser.add_argument("--output", type=Path, help="Image path; default: dcf_measured_vs_predicted.png beside the CSV.")
    parser.add_argument("--show", action="store_true", help="Show the figure after saving (e.g. in Spyder).")
    return parser


def main(argv=None) -> Path:
    args = build_parser().parse_args(argv)
    csv_path = args.csv_path.expanduser().resolve()
    columns = read_scalar_csv(csv_path)
    run = calculate_series(columns, branch=args.branch, chi_a=args.chi_a,
                           l_perp=args.l_perp, tmin=args.tmin)
    output_path = (csv_path.with_name("dcf_measured_vs_predicted.png") if args.output is None
                   else args.output.expanduser().resolve())
    plot_series(run, output_path, args.show)
    return output_path


if __name__ == "__main__":
    main()
