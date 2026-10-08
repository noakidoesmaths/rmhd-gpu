"""Compare measured compressive and reflected-wave amplitudes with the slaved estimates.

Squire et al., arXiv:2607.08036, Eqs. (46)-(48) and (73), for
inhomogeneous_rmhd_rho runs, whose background gradients all point along x.
Everything is read from scalar_diagnostics.csv and input_copy.input, so no
full-field snapshots are needed. Every quantity plotted here (z±, l_perp, the
alignment, chi_A and the Eq. 47 and Eq. 73 estimates) is defined in
vis/run_quantities.py; this script only draws them:

    Eq. (47): rms(f)          ~ |F_f| l_perp alignment        f = drho, du_par, db_par
    Eq. (73): z-_rms / z+_rms ~ l_perp |g| rms(drho) / z+_rms^2

Both estimates are mixing-length arguments, so expect agreement up to an O(1)
factor, not a pointwise match. Eq. (73) is divided by z+_rms so that both of
its curves are dimensionless. There is no curvature term in this model.

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
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from vis._matplotlib import finalize_figure, import_pyplot
from vis.run_quantities import FIELDS, load_run


# Edit these to change the field labels and colours in the plots.
LABELS = {
    "drho": r"\delta\rho/\rho_0",
    "du_par": r"\delta u_\parallel/v_A",
    "db_par": r"(v_A^2/v_S^2)\,\delta B_\parallel/B_0",
}
COLORS = {"drho": "tab:blue", "du_par": "tab:green", "db_par": "tab:orange"}


def plot_compressive_panel(ax, run, name):
    """Measured RMS of one compressive field against its Eq. (47) estimate."""
    ax.plot(run.times, run.measured[name], lw=2, color=COLORS[name], label="Measured RMS")
    if run.drives[name] == 0.0:
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
    ax.plot(run.times, run.z_plus / run.p.vA, color="tab:red", lw=2, label=r"$z^+_{\rm rms}/v_A$")
    ax.plot(run.times, run.z_minus / run.p.vA, color="tab:cyan", lw=2, label=r"$z^-_{\rm rms}/v_A$")
    ax.set(xlabel="Time", ylabel=r"$z^\pm_{\rm rms}/v_A$ (dimensionless)",
           title="Elsasser amplitudes: launched and reflected wave")


def plot_perp_amplitude_panel(ax, run):
    """rms(delta u_perp)/vA and rms(delta B_perp)/B_0 against time."""
    # b_perp is in Alfven units, so b_perp_rms/vA is delta B_perp/B_0.
    ax.plot(run.times, run.u_perp / run.p.vA, color="tab:olive", lw=2,
            label=r"$\delta u_{\perp,\rm rms}/v_A$")
    # Dashed, because the two overlap in a nearly pure z+ run, where u_perp = -b_perp.
    ax.plot(run.times, run.b_perp / run.p.vA, color="tab:brown", lw=2, ls="--",
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


def figure_title(run):
    p = run.p
    return (
        f"{run.run_dir.name}: slaved amplitudes | "
        rf"$g$ = {p.g:.4g}, $\chi$ = {p.chi:g}, $K_\rho$ = {p.K_rho0:g}" + "\n"
        rf"mean RMS$(z^+_x)/z^+_{{\rm rms}}$ = {finite_mean(run.alignment, '.3f')} "
        rf"(isotropic: {1.0 / np.sqrt(2.0):.3f}) | "
        rf"mean measured $\chi_A$ = {finite_mean(run.chi_a, '.3g')}"
    )


def plot_series(run, fields, output_path, show=False):
    """Compressive panels, then the Eq. (73) ratio, z+/z- and u_perp/b_perp, three per row."""
    plt = import_pyplot(show=show)
    panel_count = len(fields) + 3
    rows = int(np.ceil(panel_count / 3))
    fig, axes = plt.subplots(
        rows, 3, figsize=(17.0, 5.0 * rows),
        constrained_layout=True, squeeze=False,
    )
    axes = axes.ravel()

    for ax, name in zip(axes, fields):
        plot_compressive_panel(ax, run, name)
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

    fig.suptitle(figure_title(run), fontsize=10)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    finalize_figure(fig, output_path=output_path, show=show, plt=plt)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path,
                        help="Run directory containing input_copy.input and scalar_diagnostics.csv.")
    parser.add_argument("--fields", nargs="+", choices=FIELDS, default=list(FIELDS),
                        help="Compressive panels to show; the Eq. (73), z+/z- and u_perp/b_perp panels are always added.")
    parser.add_argument("--output", type=Path, help="Image path; default: RUN_DIRECTORY/slaved_projection.png.")
    parser.add_argument("--show", action="store_true", help="Show the figure after saving (e.g. in Spyder).")
    return parser


def main(argv=None) -> Path:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_dir = args.path.expanduser().resolve()
    try:
        run = load_run(run_dir, args.fields)
    except ValueError as error:
        parser.error(str(error))

    output_path = (run_dir / "slaved_projection.png" if args.output is None
                   else args.output.expanduser().resolve())
    plot_series(run, args.fields, output_path, args.show)
    return output_path


if __name__ == "__main__":
    main()
