"""Compare measured compressive and reflected-wave amplitudes with the slaved estimates.

Squire et al., arXiv:2607.08036, Eqs. (46)-(48) and (73), for
inhomogeneous_rmhd_rho runs, whose background gradients all point along x.
Everything is read from scalar_diagnostics.csv and input_copy.input, so no
full-field snapshots are needed.

Elsasser convention, as in the paper and the solver:

    z^± = delta u_perp ∓ delta B_perp/sqrt(4 pi rho_0)

z+ is the wave the initial condition launches and z- the wave it reflects
into. The solver saves W^± = <|z^±|^2>/4, so z±_rms = 2 sqrt(W±). At each saved
time, with every scale taken from z+:

    l_perp    = 1/<k_perp>                          outer scale
    alignment = rms(z+_x)/z+_rms                    share of z+ along x
    chi_A     = z+_rms <k_perp> / (vA <k_par>)      nonlinear / Alfven frequency

    Eq. (47): rms(f)          ~ |F_f| l_perp alignment        f = drho, du_par, db_par
    Eq. (73): z-_rms / z+_rms ~ l_perp |g| rms(drho) / z+_rms^2

F_f is the background drive of each field (Eq. 46), from
rmhdgpu.diagnostics.compressive_channels.background_drives. Both estimates are
mixing-length arguments, so expect agreement up to an O(1) factor, not a
pointwise match. Eq. (73) is divided by z+_rms so that both of its curves are
dimensionless. There is no curvature term in this model.

Panels, three per row: one per compressive field (measured RMS against
Eq. 47), then z-/z+ against Eq. (73), then z+_rms/vA and z-_rms/vA, then
delta u_perp/vA and delta B_perp/B_0. The last two show whether z+ is still
decaying, which matters when deciding whether a time window can be treated as
steady state.

A column that an older CSV lacks is read as NaN, so its curve is left empty and
the panel names the curve that has no data. For example, CSVs written before
2026-09-23 have no w_plus_align column, so they plot without the Eq. (47) line.

Examples (also usable as main([...]) in Spyder):
    python vis/plot_slaved_projection.py RUN_DIRECTORY
    python vis/plot_slaved_projection.py RUN_DIRECTORY --fields drho db_par --show
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

from rmhdgpu.diagnostics.compressive_channels import background_drives, slaved_field_units
from rmhdgpu.equations.rmhd_by_nokia_rho import derived_parameters
from vis._matplotlib import finalize_figure, import_pyplot


# Edit these to change the field labels and colours in the plots.
LABELS = {
    "drho": r"\delta\rho/\rho_0",
    "du_par": r"\delta u_\parallel/v_A",
    "db_par": r"(v_A^2/v_S^2)\,\delta B_\parallel/B_0",
}
COLORS = {"drho": "tab:blue", "du_par": "tab:green", "db_par": "tab:orange"}


@dataclass
class RunSeries:
    """Curves to plot, one value per saved time. NaN marks values the CSV cannot give."""

    times: np.ndarray
    z_plus: np.ndarray  # z+_rms / vA
    z_minus: np.ndarray  # z-_rms / vA
    u_perp: np.ndarray  # rms(delta u_perp) / vA
    b_perp: np.ndarray  # rms(delta B_perp) / B_0
    l_perp: np.ndarray  # 1/<k_perp> of z+
    alignment: np.ndarray  # rms(z+_x) / z+_rms
    chi_a: np.ndarray  # chi_A of z+
    z_ratio: np.ndarray  # measured z-_rms / z+_rms
    z_ratio_predicted: np.ndarray  # Eq. (73)
    measured: dict[str, np.ndarray]  # compressive RMS in Eq. (47) units
    predicted: dict[str, np.ndarray]  # Eq. (47) estimate for each field


# ---------------------------------------------------------------------------
# Reading a run
# ---------------------------------------------------------------------------

def load_parameters(run_dir):
    """Physics scalars from the run's saved input, via the solver's own derived_parameters."""
    input_path = run_dir / "input_copy.input"
    if not input_path.is_file():
        raise SystemExit(f"Missing {input_path}; the estimate needs the [physics] block.")
    with input_path.open("rb") as handle:
        physics = tomllib.load(handle).get("physics", {})
    missing = [key for key in ("vA", "cs2_over_vA2", "g", "K_p0", "K_rho0") if key not in physics]
    if missing:
        raise SystemExit(f"{input_path} [physics] is missing {missing}.")
    return derived_parameters(physics)


def read_csv_rows(run_dir):
    csv_path = run_dir / "scalar_diagnostics.csv"
    if not csv_path.is_file():
        raise SystemExit(f"Missing {csv_path}.")
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"{csv_path} contains no data rows.")
    return csv_path, rows


def csv_column(rows, name):
    """One column as floats, or all NaN if the CSV predates it.

    NaN propagates through every formula in read_from_csv, so a missing
    diagnostic just leaves its curve empty and needs no special case.
    """
    if name not in rows[0]:
        return np.full(len(rows), np.nan)
    return np.array([float(row[name]) for row in rows])


def safe_divide(numerator, denominator):
    """numerator / denominator, with NaN wherever the denominator is not positive."""
    numerator, denominator = np.asarray(numerator, float), np.asarray(denominator, float)
    out = np.full(np.broadcast(numerator, denominator).shape, np.nan)
    return np.divide(numerator, denominator, out=out, where=denominator > 0.0)


def read_from_csv(run_dir, fields, p):
    """Read one run's scalar_diagnostics.csv and compute every plotted curve."""
    csv_path, rows = read_csv_rows(run_dir)
    time_key = "time" if "time" in rows[0] else "t"
    missing = [name for name in [time_key] + [f"{name}_rms" for name in fields] if name not in rows[0]]
    if missing:
        raise SystemExit(f"{csv_path} is missing {missing}; is this an inhomogeneous_rmhd_rho run?")

    # Elsasser amplitudes, z±_rms = 2 sqrt(W±).
    z_plus = 2.0 * np.sqrt(csv_column(rows, "w_plus"))
    z_minus = 2.0 * np.sqrt(csv_column(rows, "w_minus"))
    if z_minus[0] > z_plus[0]:
        print(f"{csv_path.name}: W- > W+ at the first saved time. The estimates assume "
              "z+ = u_perp - b_perp is the driving wave, so they do not describe this run.")

    # Outer scale, alignment with x, and chi_A, all of z+.
    kperp = csv_column(rows, "w_plus_kperp")
    l_perp = safe_divide(1.0, kperp)
    alignment = csv_column(rows, "w_plus_align")
    chi_a = safe_divide(z_plus * kperp, p.vA * csv_column(rows, "w_plus_kprl"))

    # Eq. (47): rms(f) ~ |F_f| l_perp alignment. A fluid element displaced by
    # ~ (l_perp/z+) z+_x along x picks up delta f ~ F_f * displacement. An empty
    # z+ is saved with alignment 0, so its estimate is 0 although l_perp is undefined.
    drives, units = background_drives(p), slaved_field_units(p)
    projected_length = np.where(alignment == 0.0, 0.0, l_perp * alignment)
    measured = {name: csv_column(rows, f"{name}_rms") / units[name] for name in fields}
    predicted = {name: abs(drives[name]) * projected_length for name in fields}

    # Eq. (73): buoyancy |g| drho acting for one nonlinear time l_perp/z+ drives
    # z- ~ (l_perp/z+) |g| rms(drho); divide by z+ for a dimensionless ratio.
    # drho_rms is read even when the density panel is not shown.
    z_ratio = safe_divide(z_minus, z_plus)
    z_ratio_predicted = safe_divide(l_perp * abs(p.g) * csv_column(rows, "drho_rms"), z_plus**2)

    return RunSeries(
        times=csv_column(rows, time_key),
        z_plus=z_plus / p.vA,
        z_minus=z_minus / p.vA,
        u_perp=csv_column(rows, "u_perp_rms") / p.vA,
        b_perp=csv_column(rows, "b_perp_rms") / p.vA,
        l_perp=l_perp,
        alignment=alignment,
        chi_a=chi_a,
        z_ratio=z_ratio,
        z_ratio_predicted=z_ratio_predicted,
        measured=measured,
        predicted=predicted,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_compressive_panel(ax, run, name, drive):
    """Measured RMS of one compressive field against its Eq. (47) estimate."""
    ax.plot(run.times, run.measured[name], lw=2, color=COLORS[name], label="Measured RMS")
    if drive == 0.0:
        ax.axhline(0.0, color="black", ls="--", lw=1.6, label="Slaved estimate: zero drive")
    else:
        ax.plot(run.times, run.predicted[name], "k--", lw=1.6, label="Slaved estimate (Eq. 47)")
    ax.set(xlabel="Time", ylabel=rf"RMS of ${LABELS[name]}$",
           title=rf"Plot of measured and predicted ${LABELS[name]}$" + "\nagainst time")


def plot_reflection_panel(ax, run):
    """Measured z-/z+ against the Eq. (73) estimate."""
    ratio = r"z^-_{\rm rms}/z^+_{\rm rms}"
    ax.plot(run.times, run.z_ratio, color="tab:purple", lw=2, label=rf"Measured ${ratio}$")
    ax.plot(run.times, run.z_ratio_predicted, "k--", lw=1.6,
            label=r"$(\ell_\perp/(z^+_{\rm rms})^2)\,|-g\,\delta\rho/\rho_0|_{\rm rms}$")
    ax.set(xlabel="Time", ylabel="RMS amplitude ratio (dimensionless)",
           title=rf"${ratio}$: measured and estimated")


def plot_elsasser_panel(ax, run):
    """z+_rms/vA and z-_rms/vA against time."""
    ax.plot(run.times, run.z_plus, color="tab:red", lw=2, label=r"$z^+_{\rm rms}/v_A$")
    ax.plot(run.times, run.z_minus, color="tab:cyan", lw=2, label=r"$z^-_{\rm rms}/v_A$")
    ax.set(xlabel="Time", ylabel=r"$z^\pm_{\rm rms}/v_A$ (dimensionless)",
           title="Elsasser amplitudes: launched and reflected wave")


def plot_perp_amplitude_panel(ax, run):
    """rms(delta u_perp)/vA and rms(delta B_perp)/B_0 against time."""
    ax.plot(run.times, run.u_perp, color="tab:olive", lw=2,
            label=r"$\delta u_{\perp,\rm rms}/v_A$")
    # Dashed, because the two overlap in a nearly pure z+ run, where u_perp = -b_perp.
    ax.plot(run.times, run.b_perp, color="tab:brown", lw=2, ls="--",
            label=r"$\delta B_{\perp,\rm rms}/B_0$")
    ax.set(xlabel="Time", ylabel="RMS amplitude (dimensionless)",
           title="Perpendicular velocity and magnetic field")


def note_missing_curves(ax):
    """Name the curves with no data, usually because an older CSV lacks their columns."""
    missing = [line.get_label() for line in ax.lines if not np.isfinite(line.get_ydata()).any()]
    if missing:
        ax.text(0.5, 0.5, "No data in this CSV for:\n" + "\n".join(missing),
                transform=ax.transAxes, ha="center", va="center", fontsize=8, wrap=True,
                bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.9))


def finite_mean(values, fmt):
    """Mean of the finite values, formatted, or "n/a" if there are none."""
    finite = values[np.isfinite(values)]
    return format(finite.mean(), fmt) if finite.size else "n/a"


def figure_title(run_dir, run, p):
    return (
        f"{run_dir.name}: slaved amplitudes | "
        rf"$g$ = {p.g:.4g}, $\chi$ = {p.chi:g}, $K_\rho$ = {p.K_rho0:g}" + "\n"
        rf"mean RMS$(z^+_x)/z^+_{{\rm rms}}$ = {finite_mean(run.alignment, '.3f')} "
        rf"(isotropic: {1.0 / np.sqrt(2.0):.3f}) | "
        rf"mean measured $\chi_A$ = {finite_mean(run.chi_a, '.3g')}"
    )


def plot_series(run_dir, run, fields, p, output_path, show=False):
    """Compressive panels, then the Eq. (73) ratio, z+/z- and u_perp/b_perp, three per row."""
    plt = import_pyplot(show=show)
    panel_count = len(fields) + 3
    rows = int(np.ceil(panel_count / 3))
    fig, axes = plt.subplots(
        rows, 3, figsize=(17.0, 5.0 * rows),
        constrained_layout=True, squeeze=False,
    )
    axes = axes.ravel()

    drives = background_drives(p)
    for ax, name in zip(axes, fields):
        plot_compressive_panel(ax, run, name, drives[name])
    plot_reflection_panel(axes[len(fields)], run)
    plot_elsasser_panel(axes[len(fields) + 1], run)
    plot_perp_amplitude_panel(axes[len(fields) + 2], run)

    for ax in axes[:panel_count]:
        note_missing_curves(ax)
        # No additive offset: a nearly flat curve (e.g. 1.732 +- 1e-5) would otherwise get
        # tiny tick labels plus an easily missed "+1.732", which looks like a tiny amplitude.
        ax.ticklabel_format(axis="y", useOffset=False)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    for ax in axes[panel_count:]:
        ax.set_visible(False)

    fig.suptitle(figure_title(run_dir, run, p), fontsize=10)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    finalize_figure(fig, output_path=output_path, show=show, plt=plt)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path,
                        help="Run directory containing input_copy.input and scalar_diagnostics.csv.")
    parser.add_argument("--fields", nargs="+", choices=tuple(LABELS), default=list(LABELS),
                        help="Compressive panels to show; the Eq. (73), z+/z- and u_perp/b_perp panels are always added.")
    parser.add_argument("--output", type=Path, help="Image path; default: RUN_DIRECTORY/slaved_projection.png.")
    parser.add_argument("--show", action="store_true", help="Show the figure after saving (e.g. in Spyder).")
    return parser


def main(argv=None) -> Path:
    args = build_parser().parse_args(argv)
    run_dir = args.path.expanduser().resolve()
    p = load_parameters(run_dir)
    run = read_from_csv(run_dir, args.fields, p)

    output_path = (run_dir / "slaved_projection.png" if args.output is None
                   else args.output.expanduser().resolve())
    plot_series(run_dir, run, args.fields, p, output_path, args.show)
    return output_path


if __name__ == "__main__":
    main()
