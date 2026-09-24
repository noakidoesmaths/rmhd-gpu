"""Compare compressive RMS and the reflected-wave ratio with Eqs. (47) and (73).

Based on Squire et al., arXiv:2607.08036, Eqs. (46)-(48) and (73).
For inhomogeneous_rmhd_rho, all background gradients point along x:

    predicted_rms = abs(F) * l_perp * rms(z_x) / z_rms

Here z = zhat cross grad_perp(phi - sign*psi), z_rms = 2*sqrt(W), and
l_perp = 1/<k_perp>. The solver calculates these at every scalar-output time.
This is a mixing-length estimate, not an exact pointwise solution.

Everything is read from scalar_diagnostics.csv, so no full-field snapshots are
needed. The alignment rms(z_x)/z_rms comes from the w_plus_align / w_minus_align
columns. CSVs written before those columns existed fall back to the isotropic
value 1/sqrt(2) (Eq. 48) with a warning. That can be off by a large factor when
the flow is anisotropic in the perpendicular plane, so re-run such cases.
The pump is the stronger Elsasser wave in the first row unless --branch is given.
The title averages measured chi_A = z_rms*<k_perp>/(vA*<k_parallel>) over
the plotted times, omitting undefined values.

The extra CCR panel divides both sides of Eq. (73) by the pump RMS amplitude:
    z_ratio = sqrt(W_counter/W_pump)
    z_ratio_predicted = (l_perp/z_pump_rms**2) * abs(g) * rms(drho)
Here drho is the measured delta rho/rho_0, and the selected branch is the pump.
For the plus pump these are z^-/z^+ and (l_perp/(z^+)**2) * |-g delta rho/rho_0|_rms.
Both curves are dimensionless and undefined when the pump amplitude is zero.
There is no curvature term in this model and no extra isotropy factor here.

Examples (also usable as main([...]) in Spyder):
    python vis/plot_slaved_projection.py RUN_DIRECTORY
    python vis/plot_slaved_projection.py RUN_DIRECTORY --fields drho db_par --show
    python vis/plot_slaved_projection.py RUN_DIRECTORY --branch minus
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from rmhdgpu.equations.rmhd_by_nokia_rho import derived_parameters
from vis._matplotlib import finalize_figure, import_pyplot


# Edit these to change the field labels and colours in the plots.
LABELS = {
    "drho": r"\delta\rho/\rho_0",
    "du_par": r"\delta u_\parallel/v_A",
    "db_par": r"(v_A^2/v_S^2)\,\delta B_\parallel/B_0",
}
COLORS = {"drho": "tab:blue", "du_par": "tab:green", "db_par": "tab:orange"}
ISOTROPIC_ALIGNMENT = 1.0 / np.sqrt(2.0)


def forcings(p):
    """The x components of Eq. (46), for straight field and no mean flow."""
    return {
        # Equivalent to -N_sq/g, but also defined when g = 0.
        "drho": p.g / (p.vA**2 * (1.0 + p.chi)) - p.K_rho0,
        "du_par": -p.K_b0,
        "db_par": -p.K_b0 + p.K_p0 / p.gamma,
    }


def measured_divisors(p):
    """Convert saved fields to Eq. (47) units; alpha = vS^2/vA^2."""
    return {"drho": 1.0, "du_par": p.vA, "db_par": p.alpha}


@dataclass
class RunSeries:
    """Arrays ready to plot, with one value per saved time."""

    times: np.ndarray
    l_perp: np.ndarray
    alignment: np.ndarray  # rms(z_x) / z_rms
    chi_a: np.ndarray  # nonlinear frequency / Alfven frequency
    z_ratio: np.ndarray  # Opposite-branch RMS / pump RMS (dimensionless)
    z_ratio_predicted: np.ndarray  # Eq. (73) / pump RMS, using measured density
    measured: dict[str, np.ndarray]
    predicted: dict[str, np.ndarray]
    source: str  # Whether the alignment was measured or assumed isotropic
    branch: str  # The pump: "plus" or "minus"


def load_parameters(run_dir):
    input_path = run_dir / "input_copy.input"
    if not input_path.is_file():
        raise SystemExit(f"Missing {input_path}; the estimate needs the [physics] block.")
    with input_path.open("rb") as handle:
        physics = tomllib.load(handle).get("physics", {})
    missing = [key for key in ("vA", "cs2_over_vA2", "g", "K_p0", "K_rho0") if key not in physics]
    if missing:
        raise SystemExit(f"{input_path} [physics] is missing {missing}.")
    return derived_parameters(physics)


def read_from_csv(run_dir, fields, p, branch=None):
    """Read one run's scalar_diagnostics.csv into arrays ready to plot.

    Each row holds the field RMS, W^±, <k_perp>, <k_parallel> and the
    alignment rms(z_x)/z_rms (w_plus_align / w_minus_align) at one
    scalar-output time. A CSV written before the alignment columns existed
    falls back to the Eq. (48) value 1/sqrt(2). That fallback can be off by a
    large factor in anisotropic runs.
    """
    csv_path = run_dir / "scalar_diagnostics.csv"
    if not csv_path.is_file():
        raise SystemExit(f"Missing {csv_path}.")
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"{csv_path} contains no data rows.")
    first_row = rows[0]
    time_key = "time" if "time" in first_row else "t"

    # Auto picks the stronger branch in the FIRST row and keeps it for the whole run.
    # A CSV without both energies can only be read on the plus branch.
    if branch is None:
        branch = "plus"
        if "w_plus" in first_row and "w_minus" in first_row:
            if float(first_row["w_minus"]) > float(first_row["w_plus"]):
                branch = "minus"
    pump, counter = ("w_plus", "w_minus") if branch == "plus" else ("w_minus", "w_plus")

    required = [time_key, f"{pump}_kperp"] + [f"{name}_rms" for name in fields]
    missing = [name for name in required if name not in first_row]
    if missing:
        raise SystemExit(f"{csv_path} is missing {missing}; it may predate these diagnostics, "
                         "so re-run the case with the current solver.")
    # Eq. (73) needs drho_rms even when the density panel is not selected.
    optional = [name for name in (pump, counter, f"{pump}_kprl", f"{pump}_align", "drho_rms")
                if name in first_row and name not in required]
    columns = {name: np.array([float(row[name]) for row in rows]) for name in required + optional}

    kperp = columns[f"{pump}_kperp"]
    l_perp = np.divide(1.0, kperp, out=np.full_like(kperp, np.nan), where=np.abs(kperp) > 0.0)
    if f"{pump}_align" in columns:
        alignment = columns[f"{pump}_align"]
        source = "CSV: measured alignment"
    else:
        print(f"{csv_path.name} has no {pump}_align column (written before it was added). "
              "Assuming isotropic alignment 1/sqrt(2), Eq. (48). The Eq. (47) estimate can "
              "then be off by a large factor; re-run the case to save the measured value.")
        alignment = np.full_like(kperp, ISOTROPIC_ALIGNMENT)
        source = "CSV: Eq. (48) isotropy assumed"
    # RMS[(l_perp/z_rms) * z_x * F] factorises because F is constant. The saved alignment
    # is zero for an empty field, so the prediction is zero there although l_perp is undefined.
    projected_length = np.where(alignment == 0.0, 0.0, l_perp * alignment)

    # Older CSVs can still be plotted when the chi_A diagnostics are absent.
    chi_a = np.full_like(kperp, np.nan)
    if pump in columns and f"{pump}_kprl" in columns:
        omega_nl = 2.0 * np.sqrt(np.maximum(columns[pump], 0.0)) * kperp
        omega_a = p.vA * columns[f"{pump}_kprl"]
        np.divide(omega_nl, omega_a, out=chi_a, where=omega_a > 0.0)
    z_ratio = np.full_like(kperp, np.nan)
    z_ratio_predicted = np.full_like(kperp, np.nan)
    if pump in columns:
        z_pump = 2.0 * np.sqrt(np.maximum(columns[pump], 0.0))
        if counter in columns:
            z_counter_rms = 2.0 * np.sqrt(np.maximum(columns[counter], 0.0))
            np.divide(z_counter_rms, z_pump, out=z_ratio, where=z_pump > 0.0)
        if "drho_rms" in columns:
            # Divide Eq. (73) by the pump RMS to get the dimensionless wave ratio.
            np.divide(l_perp * abs(p.g) * columns["drho_rms"], z_pump**2,
                      out=z_ratio_predicted, where=z_pump > 0.0)
    forcing, divisor = forcings(p), measured_divisors(p)
    return RunSeries(
        times=columns[time_key], l_perp=l_perp, alignment=alignment,
        chi_a=chi_a,
        z_ratio=z_ratio, z_ratio_predicted=z_ratio_predicted,
        measured={name: columns[f"{name}_rms"] / divisor[name] for name in fields},
        predicted={name: abs(forcing[name]) * projected_length for name in fields},
        source=source, branch=branch,
    )


def plot_series(run_dir, run, fields, p, output_path, show=False):
    """One panel per compressive field, followed by the dimensionless wave ratio."""
    plt = import_pyplot(show=show)
    forcing = forcings(p)
    panel_count = len(fields) + 1
    rows = (panel_count + 1) // 2
    fig, axes = plt.subplots(
        rows, 2, figsize=(11.6, 5.0 * rows),
        constrained_layout=True, squeeze=False,
    )
    for ax, name in zip(axes.ravel(), fields):
        measured, predicted = run.measured[name], run.predicted[name]
        ax.plot(run.times, measured, lw=2, color=COLORS[name], label="Measured RMS")
        if forcing[name] == 0.0:
            ax.axhline(0.0, color="black", ls="--", lw=1.6, label="Slaved estimate: zero drive")
            summary = "No background-gradient drive"
        else:
            ax.plot(run.times, predicted, "k--", lw=1.6, label="Slaved estimate (Eq. 47)")
            # ratio = np.divide(measured, predicted, out=np.full_like(measured, np.nan), where=predicted != 0)
            # valid = ratio[np.isfinite(ratio)]
            # mean_ratio = float(valid.mean()) if valid.size else np.nan
            # summary = f"Mean measured / estimate = {mean_ratio:.3g}"
        ax.set(xlabel="Time", ylabel=rf"RMS of ${LABELS[name]}$",
               title=rf"Plot of measured and predicted ${LABELS[name]}$" + "\nagainst time")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

    ax = axes.ravel()[len(fields)]
    # The paper calls the pump z+; swap the labels when z- is the pump.
    pump_label = r"z^-" if run.branch == "minus" else r"z^+"
    counter_label = r"z^+" if run.branch == "minus" else r"z^-"
    ratio_label = rf"{counter_label}_{{\rm rms}}/{pump_label}_{{\rm rms}}"
    ax.plot(run.times, run.z_ratio, color="tab:purple", lw=2,
            label=rf"Measured ${ratio_label}$")
    ax.plot(run.times, run.z_ratio_predicted, "k--", lw=1.6,
            label=rf"$(\ell_\perp/({pump_label}_{{\rm rms}})^2)\,|-g\,\delta\rho/\rho_0|_{{\rm rms}}$")
    ax.set(xlabel="Time", ylabel="RMS amplitude ratio (dimensionless)",
           title=rf"${ratio_label}$: measured and estimated")
    notes = []
    if not np.isfinite(run.z_ratio).any():
        notes.append("Measured ratio unavailable: missing energy or zero pump amplitude")
    if not np.isfinite(run.z_ratio_predicted).any():
        notes.append("Estimate unavailable: missing density or no finite pump timescale")
    if notes:
        ax.text(0.5, 0.45, "\n".join(notes), transform=ax.transAxes,
                ha="center", va="center", fontsize=8, wrap=True)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    for ax in axes.ravel()[panel_count:]:
        ax.set_visible(False)

    finite_chi_a = run.chi_a[np.isfinite(run.chi_a)]
    chi_a_label = f"{finite_chi_a.mean():.3g}" if finite_chi_a.size else "n/a"
    fig.suptitle(
        f"{run_dir.name}: slaved amplitudes\n{run.source}; {run.branch} branch | "
        rf"$g$ = {p.g:.4g}, $\chi$ = {p.chi:g}, $K_\rho$ = {p.K_rho0:g} | "
        rf"mean RMS$(z_x)/z_{{\rm rms}}$ = {np.mean(run.alignment):.3f} "
        rf"(isotropic: {ISOTROPIC_ALIGNMENT:.3f})" + "\n"
        rf"mean measured $\chi_A$ = {chi_a_label}",
        fontsize=10,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    finalize_figure(fig, output_path=output_path, show=show, plt=plt)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path,
                        help="Run directory containing input_copy.input and scalar_diagnostics.csv.")
    parser.add_argument("--fields", nargs="+", choices=tuple(LABELS), default=list(LABELS),
                        help="Compressive panels to show; the Eq. (73) panel is always added.")
    parser.add_argument("--branch", choices=("plus", "minus"),
                        help="Pump branch; default picks the stronger one in the first CSV row.")
    parser.add_argument("--output", type=Path, help="Image path; default: RUN_DIRECTORY/slaved_projection.png.")
    parser.add_argument("--show", action="store_true", help="Show the figure after saving (e.g. in Spyder).")
    return parser


def main(argv=None) -> Path:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_dir = args.path.expanduser().resolve()
    p = load_parameters(run_dir)
    run = read_from_csv(run_dir, args.fields, p, args.branch)

    output_path = (run_dir / "slaved_projection.png" if args.output is None
                   else args.output.expanduser().resolve())
    plot_series(run_dir, run, args.fields, p, output_path, args.show)
    return output_path


if __name__ == "__main__":
    main()
