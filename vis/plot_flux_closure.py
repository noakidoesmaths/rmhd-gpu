"""Plot the dimensionless density and psi fluxes, and rms(drho), against chi_A.

Squire et al., arXiv:2607.08036, Eqs. (79), (80) and (85). The closure says the
fluxes across the background gradient are diffusive, with one eddy-scale
diffusivity eta_turb = z+_rms l_perp / 4:

    V_rho_x = <(delta rho/rho_0) delta u_x>                              ~  eta_turb F_rho   (Eq. 80)
    V_psi_x = <(delta B_par/B_0) delta u_x - delta u_par delta B_x/B_0>  ~  eta_turb F_psi   (Eq. 85)

with F_psi = F_u + (vS^2/vA^2) F_B. Each V is a velocity and each F a gradient
(1/length), so dividing out the drive and the eddy scales makes the flux
panels dimensionless. --panels picks which panels to draw, in that order:

    V_rho:  V_rho_x / (F_rho z+_rms l_perp)    closure: 1/4 (dashed line)
    V_psi:  V_psi_x / (F_psi z+_rms l_perp)    closure: 1/4 (dashed line)
    drho:   rms(delta rho/rho_0)               no estimate drawn

The default is V_rho and V_psi. Each flux ratio is also the measured diffusivity
V/F in units of z+ l_perp. Each run is one point against its chi_A, so runs with
different drives or amplitudes can share one plot, and the closure is a flat
line. The paper notes the closure assumes strong turbulence, chi_A >~ 1 (dotted
line). The drho panel is for reference only: rms(delta rho) also counts
small-scale filaments that carry no flux. All panels are log-log; a run with a
ratio <= 0 cannot be shown on the log axis and is reported on the console instead.

Where the numbers come from: z+, l_perp = 1/k_perp_plus, chi_A, the alignment,
V_rho_x, V_psi_x and the drives F_rho and F_psi are all defined in
vis/run_quantities.py, from the saved scalar_diagnostics.csv and
input_copy.input. This script forms the ratios at each saved time and averages
them over the window (run_quantities.time_window): by default the last 40% of
each run (--tail-fraction), or [--tmin, --tmax]. Runs with different t_max then
average over different times, so pass --tmin/--tmax to compare like with like.
Check first with vis/plot_flux_convergence.py that the V_rho ratio has settled
over that window: it plots the same ratio against time, with the window marked.

Runs without a V_rho_x column use acr_g / g, the same correlator, so a g = 0 run
written before V_rho_x existed has no V_rho point. Likewise runs without a
V_psi_x column use -acr_B / (vA^2 K_b0), so a K_b0 = 0 run written before V_psi_x
existed has no V_psi point. A run with F_rho = 0 (or F_psi = 0) has no V_rho
(or V_psi) point. The window means, including the dimensional ones, are also
printed and saved to a CSV beside the image.

Examples (also usable as main([...]) in Spyder):
    python vis/plot_flux_closure.py "examples/rho_Kp0_grad_3.0/energy_*" --tmin 36 --tmax 60
    python vis/plot_flux_closure.py RUN_1 RUN_2 --panels V_rho V_psi drho --show
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


FLUX_CLOSURE = 0.25  # V / (F z+ l_perp) = eta_turb / (z+ l_perp), Eqs. (80) and (85)

# What each --panels choice draws: the window mean under `key` against chi_A,
# and a dashed closure line at `closure` unless it is None.
PANELS = {
    "V_rho": {
        "key": "V_over_F_z_l",
        "text": "V_rho_x/(F_rho z+ l_perp)",
        "closure": FLUX_CLOSURE,
        "color": COLORS["drho"],
        "label": r"measured $V_{\rho,x}/(F_\rho z^+\ell_\perp)$",
        "closure_label": r"closure $\eta_{\rm turb}/(z^+\ell_\perp) = 1/4$ (Eq. 80)",
        "ylabel": r"$V_{\rho,x}\,/\,(F_\rho\, z^+_{\rm rms}\,\ell_\perp)$",
        "title": r"$V_{\rho,x} = \langle(\delta\rho/\rho_0)\,\delta u_x\rangle$ vs $\chi_A$",
    },
    "V_psi": {
        "key": "V_psi_over_F_z_l",
        "text": "V_psi_x/(F_psi z+ l_perp)",
        "closure": FLUX_CLOSURE,
        "color": COLORS["db_par"],
        "label": r"measured $V_{\psi,x}/[(F_u + v_S^2 F_B/v_A^2)\, z^+\ell_\perp]$",
        "closure_label": r"closure $\eta_{\rm turb}/(z^+\ell_\perp) = 1/4$ (Eq. 85)",
        "ylabel": r"$V_{\psi,x}\,/\,[(F_u + v_S^2 F_B/v_A^2)\, z^+_{\rm rms}\,\ell_\perp]$",
        "title": r"$V_{\psi,x} = \langle(\delta B_\parallel/B_0)\,\delta u_x"
                 r" - \delta u_\parallel\,\delta B_x/B_0\rangle$ vs $\chi_A$",
    },
    "drho": {
        "key": "drho_rms",
        "text": "rms(drho)",
        "closure": None,
        "color": COLORS["drho"],
        "label": r"measured rms$(\delta\rho/\rho_0)$",
        "ylabel": r"rms$(\delta\rho/\rho_0)$",
        "title": r"Rms drho vs $\chi_A$",
    },
}
DEFAULT_PANELS = ("V_rho", "V_psi")


def read_run(run_dir, *, tmin=None, tmax=None, tail_fraction=0.4):
    """Window means of chi_A and the dimensionless closure ratios for one run."""
    run = load_run(run_dir, ["drho"])
    try:
        window = time_window(run.times, tmin=tmin, tmax=tmax, tail_fraction=tail_fraction)
    except ValueError as error:
        raise ValueError(f"{run.run_dir.name}: {error}") from None

    # Each ratio is formed at every saved time and only then averaged. A NaN in the
    # window (a column the CSV lacks, or a zero drive) makes that mean NaN, and its marker is skipped.
    f_rho, f_psi = run.drives["drho"], run.f_psi
    with np.errstate(divide="ignore", invalid="ignore"):
        flux_ratio = run.eta_meas / (run.z_plus * run.l_perp)  # eta_meas = V_rho_x / F_rho
        if f_psi != 0.0:
            psi_ratio = run.v_psi_x / (f_psi * run.z_plus * run.l_perp)
        else:
            psi_ratio = np.full_like(run.times, np.nan)
        drho_ratio = run.measured["drho"] / (abs(f_rho) * run.l_perp)

    def mean(values):
        return float(values[window].mean())

    point = {
        "run": run.run_dir.name,
        "t_start": float(run.times[window][0]),
        "t_end": float(run.times[window][-1]),
        "F_rho": float(f_rho),
        "F_psi": float(f_psi),
        "chi_A": mean(run.chi_a),
        "V_over_F_z_l": mean(flux_ratio),
        "V_psi_over_F_z_l": mean(psi_ratio),
        "drho_over_F_l": mean(drho_ratio),
        "alignment": mean(run.alignment),
        # Dimensional means, for reference.
        "V_rho_x": mean(run.v_rho_x),
        "V_psi_x": mean(run.v_psi_x),
        "eta_meas": mean(run.eta_meas),
        "eta_turb": mean(run.eta_turb),
        "drho_rms": mean(run.measured["drho"]),
    }
    name = run.run_dir.name
    if f_rho == 0.0:
        print(f"{name}: F_rho = 0, so V_rho_x/(F_rho z+ l_perp) is undefined; no V_rho point.")
    elif np.isnan(point["V_rho_x"]):
        print(f"{name}: no V_rho_x (no column, and g = 0 so acr_g/g cannot stand in); rerun to save it.")
    if f_psi == 0.0:
        print(f"{name}: F_psi = F_u + vS^2/vA^2 F_B = 0, so V_psi_x/(F_psi z+ l_perp) is undefined; "
              "no V_psi point.")
    elif np.isnan(point["V_psi_x"]):
        print(f"{name}: no V_psi_x (no column, and -acr_B/(vA^2 K_b0) cannot stand in: no acr_B, "
              "or K_b0 = 0); rerun to save it.")
    return point


def print_table(points):
    print(f"{'run':<32} {'window':>12} {'chi_A':>6} {'V/(F z+ l)':>10} {'Vpsi/(F z+ l)':>13} "
          f"{'drho/(F l)':>10} {'align':>6}")
    for pt in points:
        window = f"{pt['t_start']:.3g}-{pt['t_end']:.3g}"
        print(f"{pt['run']:<32} {window:>12} {pt['chi_A']:>6.3g} {pt['V_over_F_z_l']:>10.3g} "
              f"{pt['V_psi_over_F_z_l']:>13.3g} {pt['drho_over_F_l']:>10.3g} {pt['alignment']:>6.3g}")


def plot_closure(points, output_path, *, panels=DEFAULT_PANELS, window_text, show=False):
    """One log-log panel per name in `panels` (see PANELS), each against chi_A."""
    plt = import_pyplot(show=show)
    fig, axes = plt.subplots(1, len(panels), figsize=(6.0 * len(panels), 5.0),
                             squeeze=False, constrained_layout=True)
    chi = np.array([pt["chi_A"] for pt in points])

    for ax, name in zip(axes[0], panels):
        panel = PANELS[name]
        values = [pt[panel["key"]] for pt in points]
        for pt, value in zip(points, values):
            if value <= 0.0:  # NaN compares False, so only real non-positive values warn.
                print(f"{pt['run']}: {panel['text']} = {value:.3g} <= 0; not shown on the log axis.")

        # Filled: measured. Black: closure (as in the other vis scripts).
        ax.plot(chi, values, ls="none", marker="o", ms=7, color=panel["color"], label=panel["label"])
        if panel["closure"] is not None:
            ax.axhline(panel["closure"], color="black", ls="--", lw=1.4, label=panel["closure_label"])
        ax.axvline(1.0, color="0.5", ls=":", lw=1.0)  # The closure assumes chi_A >~ 1.
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set(xlabel=r"$\chi_A = z^+_{\rm rms}\langle k_\perp\rangle/(v_A\langle k_\parallel\rangle)$",
               ylabel=panel["ylabel"], title=panel["title"])
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=9)

    fig.suptitle(f"Flux closure diagnostics | window means, {window_text}", fontsize=11)
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
    parser.add_argument("--panels", nargs="+", choices=list(PANELS), default=list(DEFAULT_PANELS),
                        help="Panels to draw, left to right; default: V_rho V_psi.")
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
    panels = list(dict.fromkeys(args.panels))  # Drop repeats, keep the order given.
    plot_closure(points, output, panels=panels, window_text=describe_window(args), show=args.show)
    return output


if __name__ == "__main__":
    main()
