"""Plot the density flux V_rho and its closure eta_turb F_rho against chi_A, plus rms(drho/rho_0).

Squire et al., arXiv:2607.08036, Eqs. (47), (79) and (80). The closure says the
density flux across the background gradient is diffusive,

    V_rho_x = <(delta rho/rho_0) delta u_x>  ~  eta_turb F_rho,   eta_turb = z+_rms l_perp / 4,

and the paper notes it assumes strong turbulence, chi_A >~ 1. Each run gives one
point per panel: the mean over a time window of

    left:   V_rho_x (filled) and eta_turb F_rho (hollow)
    right:  rms(drho/rho_0) (filled) and the Eq. (47) estimate l_perp |F_rho| rms(z+_x)/z+_rms (hollow)

against the mean chi_A over the same window. The closure holds where the filled
and hollow markers of a run coincide. In a stably stratified run F_rho < 0, so a
down-gradient flux is negative too.

Where the numbers come from: z+, l_perp = 1/k_perp_plus, chi_A, eta_turb,
V_rho_x, rms(drho), Eq. (47) and the drive F_rho are all defined in
vis/run_quantities.py, from the saved scalar_diagnostics.csv and
input_copy.input. This script only averages them over the window
(run_quantities.time_window): by default the last 40% of each run
(--tail-fraction), or [--tmin, --tmax]. Unforced runs decay, so chi_A drifts;
pick a window where it is roughly steady.

Runs without a V_rho_x column use acr_g / g, the same correlator; a g = 0 run
written before V_rho_x existed has no flux point. The window means, including
the measured diffusivity eta_meas = V_rho_x / F_rho, are also printed and saved
to a CSV beside the image.

Examples (also usable as main([...]) in Spyder):
    python vis/plot_flux_closure.py "examples/rho_Kp0_grad_3.0/energy_*"
    python vis/plot_flux_closure.py RUN_1 RUN_2 --tmin 2 --tmax 10 --show
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from vis._matplotlib import finalize_figure, import_pyplot
from vis.plot_slaved_gradient_scan import write_summary_csv
from vis.plot_slaved_projection import COLORS
from vis.run_quantities import load_run, resolve_run_dirs, time_window


def read_run(run_dir, *, tmin=None, tmax=None, tail_fraction=0.4):
    """Window means of chi_A, the flux and its closure, and rms(drho) for one run."""
    run = load_run(run_dir, ["drho"])
    try:
        window = time_window(run.times, tmin=tmin, tmax=tmax, tail_fraction=tail_fraction)
    except ValueError as error:
        raise ValueError(f"{run.run_dir.name}: {error}") from None

    # Quantities are formed at each saved time and only then averaged. A NaN in the
    # window (a column the CSV lacks) makes that mean NaN, and its marker is skipped.
    f_rho = run.drives["drho"]

    def mean(values):
        return float(values[window].mean())

    point = {
        "run": run.run_dir.name,
        "t_start": float(run.times[window][0]),
        "t_end": float(run.times[window][-1]),
        "F_rho": float(f_rho),
        "chi_A": mean(run.chi_a),
        "V_rho_x": mean(run.v_rho_x),
        "eta_turb_F_rho": mean(run.eta_turb * f_rho),
        "drho_rms": mean(run.measured["drho"]),
        "drho_eq47": mean(run.predicted["drho"]),
        # Diffusivities, for reference.
        "eta_meas": mean(run.eta_meas),
        "eta_turb": mean(run.eta_turb),
    }
    if np.isnan(point["V_rho_x"]):
        print(f"{run.run_dir.name}: no V_rho_x (no column, and g = 0 so acr_g/g cannot stand in); "
              "rerun to save it.")
    return point


def print_table(points):
    print(f"{'run':<32} {'window':>12} {'chi_A':>6} {'V_rho_x':>9} {'eta F_rho':>9} {'drho_rms':>9} {'Eq. 47':>9}")
    for pt in points:
        window = f"{pt['t_start']:.3g}-{pt['t_end']:.3g}"
        print(f"{pt['run']:<32} {window:>12} {pt['chi_A']:>6.3g} {pt['V_rho_x']:>9.3g} "
              f"{pt['eta_turb_F_rho']:>9.3g} {pt['drho_rms']:>9.3g} {pt['drho_eq47']:>9.3g}")


def plot_closure(points, output_path, *, window_text, show=False):
    """Left: V_rho_x and eta_turb F_rho. Right: rms(drho) and Eq. (47). Both against chi_A."""
    plt = import_pyplot(show=show)
    fig, (ax_flux, ax_drho) = plt.subplots(1, 2, figsize=(12.0, 5.0), constrained_layout=True)
    chi = np.array([pt["chi_A"] for pt in points])

    # Filled: measured, in the density colour. Hollow black: estimate (as in the other vis scripts).
    measured = dict(ls="none", marker="o", ms=7, color=COLORS["drho"])
    estimate = dict(ls="none", marker="s", ms=7, mfc="none", mec="black")
    ax_flux.plot(chi, [pt["V_rho_x"] for pt in points], label=r"measured $V_{\rho,x}$", **measured)
    ax_flux.plot(chi, [pt["eta_turb_F_rho"] for pt in points],
                 label=r"closure $\eta_{\rm turb}F_\rho$", **estimate)
    ax_drho.plot(chi, [pt["drho_rms"] for pt in points],
                 label=r"measured rms$(\delta\rho/\rho_0)$", **measured)
    ax_drho.plot(chi, [pt["drho_eq47"] for pt in points], label="Eq. (47) estimate", **estimate)

    ax_flux.axhline(0.0, color="0.6", lw=0.8)
    ax_flux.set(ylabel=r"$V_{\rho,x}$",
                title=r"Density flux $V_{\rho,x} = \langle(\delta\rho/\rho_0)\,\delta u_x\rangle$"
                      "\n" r"vs closure $\eta_{\rm turb}F_\rho$, $\eta_{\rm turb} = z^+\ell_\perp/4$ (Eq. 80)")
    ax_drho.set(ylabel=r"rms$(\delta\rho/\rho_0)$",
                title=r"Density fluctuation rms$(\delta\rho/\rho_0)$" "\n"
                      r"vs Eq. (47) $\ell_\perp|F_\rho|\,$rms$(z^+_x)/z^+_{\rm rms}$")
    for ax in (ax_flux, ax_drho):
        ax.axvline(1.0, color="0.5", ls=":", lw=1.0)  # The closure assumes chi_A >~ 1.
        ax.set_xlabel(r"$\chi_A = z^+_{\rm rms}\langle k_\perp\rangle/(v_A\langle k_\parallel\rangle)$")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle(f"Plots of density diagnostics",
                 fontsize=11)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    finalize_figure(fig, output_path=output_path, show=show, plt=plt)


def describe_window(args):
    """The averaging window as requested on the command line, for the figure title."""
    if args.tmin is None and args.tmax is None:
        return f"last {100 * args.tail_fraction:g}% of each run"
    if args.tmin is None:
        return f"last {100 * args.tail_fraction:g}% of each run, up to t = {args.tmax:g}"
    end = f"{args.tmax:g}" if args.tmax is not None else "end"
    return f"t in [{args.tmin:g}, {end}]"


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="Run directories; quoted wildcards are supported.")
    parser.add_argument("--tmin", type=float, help="Start of the averaging window.")
    parser.add_argument("--tmax", type=float, help="End of the averaging window; default: end of the run.")
    parser.add_argument("--tail-fraction", type=float, default=0.4,
                        help="Without --tmin, average this final fraction of each run; default 0.4.")
    parser.add_argument("--output", type=Path, help="Image path; default flux_closure.png beside the runs.")
    parser.add_argument("--summary-output", type=Path, help="CSV of window means; default alongside the image.")
    parser.add_argument("--show", action="store_true", help="Show the figure after saving (e.g. in Spyder).")
    return parser


def main(argv=None) -> Path:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run_dirs = resolve_run_dirs(args.paths)
        points = [read_run(run_dir, tmin=args.tmin, tmax=args.tmax, tail_fraction=args.tail_fraction)
                  for run_dir in run_dirs]
    except (OSError, ValueError) as error:
        parser.error(str(error))

    parents = {path.parent for path in run_dirs}
    output_dir = parents.pop() if len(parents) == 1 else Path.cwd()
    output = (args.output or output_dir / "flux_closure.png").expanduser().resolve()
    print_table(points)
    write_summary_csv(points, args.summary_output or output.with_suffix(".csv"))
    plot_closure(points, output, window_text=describe_window(args), show=args.show)
    return output


if __name__ == "__main__":
    main()
