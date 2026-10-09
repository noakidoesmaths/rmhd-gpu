"""Plot the dimensionless density flux and density fluctuation against chi_A.

Squire et al., arXiv:2607.08036, Eqs. (47), (79) and (80). The closure says the
density flux across the background gradient is diffusive, with an eddy-scale
diffusivity and an eddy-scale displacement:

    V_rho_x = <(delta rho/rho_0) delta u_x>  ~  eta_turb F_rho,   eta_turb = z+_rms l_perp / 4   (Eq. 80)
    rms(delta rho/rho_0)                     ~  |F_rho| l_perp rms(z+_x)/z+_rms                 (Eq. 47)

V_rho_x is a velocity and F_rho a gradient (1/length), so dividing out the drive
F_rho and the eddy scales makes both dimensionless:

    left:   V_rho_x / (F_rho z+_rms l_perp)              closure: 1/4
    right:  rms(delta rho/rho_0) / (|F_rho| l_perp)      closure: rms(z+_x)/z+_rms

The left ratio is also the measured diffusivity eta_meas = V_rho_x / F_rho in
units of z+ l_perp. Each run is one point against its chi_A, so runs with
different K_rho0 or amplitude can share one plot, and the closure is a flat
line. The paper notes the closure assumes strong turbulence, chi_A >~ 1
(dotted line). The right panel's estimate is drawn per run (hollow), since the
alignment rms(z+_x)/z+_rms is measured; the isotropic value 1/sqrt(2) is the
dashed line.

Where the numbers come from: z+, l_perp = 1/k_perp_plus, chi_A, the alignment,
V_rho_x and the drive F_rho are all defined in vis/run_quantities.py, from the
saved scalar_diagnostics.csv and input_copy.input. This script forms the two
ratios at each saved time and averages them over the window
(run_quantities.time_window): by default the last 40% of each run
(--tail-fraction), or [--tmin, --tmax]. Runs with different t_max then average
over different times, so pass --tmin/--tmax to compare like with like.

Runs without a V_rho_x column use acr_g / g, the same correlator; a g = 0 run
written before V_rho_x existed has no flux point, and a run with F_rho = 0 has
neither point. The window means, including the dimensional ones, are also
printed and saved to a CSV beside the image.

Examples (also usable as main([...]) in Spyder):
    python vis/plot_flux_closure.py "examples/rho_Kp0_grad_3.0/energy_*" --tmin 36 --tmax 60
    python vis/plot_flux_closure.py RUN_1 RUN_2 --show
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


FLUX_CLOSURE = 0.25  # V_rho_x / (F_rho z+ l_perp) = eta_turb / (z+ l_perp), Eq. (80)


def read_run(run_dir, *, tmin=None, tmax=None, tail_fraction=0.4):
    """Window means of chi_A and the two dimensionless closure ratios for one run."""
    run = load_run(run_dir, ["drho"])
    try:
        window = time_window(run.times, tmin=tmin, tmax=tmax, tail_fraction=tail_fraction)
    except ValueError as error:
        raise ValueError(f"{run.run_dir.name}: {error}") from None

    # Each ratio is formed at every saved time and only then averaged. A NaN in the
    # window (a column the CSV lacks, or F_rho = 0) makes that mean NaN, and its marker is skipped.
    f_rho = run.drives["drho"]
    with np.errstate(divide="ignore", invalid="ignore"):
        flux_ratio = run.eta_meas / (run.z_plus * run.l_perp)  # eta_meas = V_rho_x / F_rho
        drho_ratio = run.measured["drho"] / (abs(f_rho) * run.l_perp)

    def mean(values):
        return float(values[window].mean())

    point = {
        "run": run.run_dir.name,
        "t_start": float(run.times[window][0]),
        "t_end": float(run.times[window][-1]),
        "F_rho": float(f_rho),
        "chi_A": mean(run.chi_a),
        "V_over_F_z_l": mean(flux_ratio),
        "drho_over_F_l": mean(drho_ratio),
        "alignment": mean(run.alignment),
        # Dimensional means, for reference.
        "V_rho_x": mean(run.v_rho_x),
        "eta_meas": mean(run.eta_meas),
        "eta_turb": mean(run.eta_turb),
        "drho_rms": mean(run.measured["drho"]),
    }
    if f_rho == 0.0:
        print(f"{run.run_dir.name}: F_rho = 0, so neither ratio is defined; not plotted.")
    elif np.isnan(point["V_rho_x"]):
        print(f"{run.run_dir.name}: no V_rho_x (no column, and g = 0 so acr_g/g cannot stand in); "
              "rerun to save it.")
    return point


def print_table(points):
    print(f"{'run':<32} {'window':>12} {'chi_A':>6} {'V/(F z+ l)':>10} {'drho/(F l)':>10} {'align':>6}")
    for pt in points:
        window = f"{pt['t_start']:.3g}-{pt['t_end']:.3g}"
        print(f"{pt['run']:<32} {window:>12} {pt['chi_A']:>6.3g} {pt['V_over_F_z_l']:>10.3g} "
              f"{pt['drho_over_F_l']:>10.3g} {pt['alignment']:>6.3g}")


def plot_closure(points, output_path, *, window_text, show=False):
    """Left: V_rho_x/(F_rho z+ l_perp). Right: rms(drho)/(|F_rho| l_perp). Both against chi_A."""
    plt = import_pyplot(show=show)
    fig, (ax_flux, ax_drho) = plt.subplots(1, 2, figsize=(12.0, 5.0), constrained_layout=True)
    chi = np.array([pt["chi_A"] for pt in points])

    # Filled: measured, in the density colour. Black: closure (as in the other vis scripts).
    measured = dict(ls="none", marker="o", ms=7, color=COLORS["drho"])
    ax_flux.plot(chi, [pt["V_over_F_z_l"] for pt in points],
                 label=r"measured $V_{\rho,x}/(F_\rho z^+\ell_\perp)$", **measured)
    ax_flux.axhline(FLUX_CLOSURE, color="black", ls="--", lw=1.4,
                    label=r"closure $\eta_{\rm turb}/(z^+\ell_\perp) = 1/4$ (Eq. 80)")
    ax_flux.axhline(0.0, color="0.6", lw=0.8)

    ax_drho.plot(chi, [pt["drho_over_F_l"] for pt in points],
                 label=r"measured rms$(\delta\rho/\rho_0)/(|F_\rho|\ell_\perp)$", **measured)
    ax_drho.plot(chi, [pt["alignment"] for pt in points], ls="none", marker="s", ms=7,
                 mfc="none", mec="black", label=r"Eq. (47): rms$(z^+_x)/z^+_{\rm rms}$")
    ax_drho.axhline(1.0 / np.sqrt(2.0), color="black", ls="--", lw=1.0, label=r"isotropic $1/\sqrt{2}$")
    ax_drho.axhline(0.0, color="0.6", lw=0.8)

    ax_flux.set(ylabel=r"$V_{\rho,x}\,/\,(F_\rho\, z^+_{\rm rms}\,\ell_\perp)$",
                title="Density flux in eddy units\n"
                      r"$V_{\rho,x} = \langle(\delta\rho/\rho_0)\,\delta u_x\rangle$")
    ax_drho.set(ylabel=r"rms$(\delta\rho/\rho_0)\,/\,(|F_\rho|\,\ell_\perp)$",
                title="Density fluctuation in eddy units\n"
                      r"(displacement along the gradient / $\ell_\perp$)")
    for ax in (ax_flux, ax_drho):
        ax.axvline(1.0, color="0.5", ls=":", lw=1.0)  # The closure assumes chi_A >~ 1.
        ax.set_xlabel(r"$\chi_A = z^+_{\rm rms}\langle k_\perp\rangle/(v_A\langle k_\parallel\rangle)$")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle(f"Plots of density diagnostics | window means, {window_text}", fontsize=11)
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
