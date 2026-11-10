"""Plot the dimensionless density and psi fluxes, and rms(drho), against chi_A.

Squire et al., arXiv:2607.08036, Eqs. (79), (80) and (85). The closure says the
fluxes across the background gradient are diffusive, with one eddy-scale
diffusivity eta_turb = l_perp z+_x^2 / (2 z+_rms) (vis/run_quantities.py):

    V_rho_x = <(delta rho/rho_0) delta u_x>                              ~  eta_turb F_rho   (Eq. 80)
    V_psi_x = <(delta B_par/B_0) delta u_x - delta u_par delta B_x/B_0>  ~  eta_turb F_psi   (Eq. 85)

with F_psi = F_u + (vS^2/vA^2) F_B. The density panel uses the x component
of <l_perp F_rho z+_x z+ / z+_rms>. At each saved time its spatial average
is l_perp F_rho rms(z+_x)^2 / z+_rms. The ratio of the window-averaged flux
to this window-averaged denominator is predicted to be 1/2, without an
isotropy assumption. --panels picks which panels to draw, in that order:

    V_rho:  mean(V_rho_x) / mean(l_perp F_rho rms(z+_x)^2 / z+_rms); reference: 1/2
    V_psi:  V_psi_x / (F_psi z+_rms l_perp)    estimate: hollow black marker per run
    drho:   rms(delta rho/rho_0)               no estimate drawn

The default is V_rho and V_psi. Each run is one point against its chi_A; the
grey dashed line is 1/2 for density and the isotropic value 1/4 for psi;
the dotted line is chi_A = 1 (the
closure assumes chi_A >~ 1). The y axis is symlog by default (--yscale), so
negative ratios are drawn; with --yscale log a ratio <= 0 cannot be shown and is
reported on the console instead. --x K_rho0 plots against the background density
gradient instead of chi_A (useful for a gradient scan at fixed forcing, where
chi_A barely changes). Error bars are block-averaged standard errors over the
window (--blocks blocks); points are labelled parent_folder/run_folder.

The density panel divides window means; the psi panel averages the instantaneous ratio
(run_quantities.time_window): by default the last 40% of each run
(--tail-fraction), or [--tmin, --tmax]. The window means, including the
dimensional ones, are printed and saved to a CSV beside the image.

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
from vis.run_quantities import (
    FIELD_COLORS, default_output_dir, describe_window, load_run, resolve_run_dirs, run_window,
    safe_divide, write_summary_csv,
)


ISOTROPIC_CLOSURE = 0.25  # z+_x^2 / (2 z+^2) with z+_x^2 = z+^2/2

# What each --panels choice draws: the window mean under `key` against chi_A,
# and the estimate under `closure_key` unless it is None.
PANELS = {
    "V_rho": {
        "key": "V_rho_normalized",
        "closure_key": None,
        "reference": 0.5,
        "text": "mean(V_rho_x)/mean(l_perp F_rho rms(z+_x)^2/z+_rms)",
        "color": FIELD_COLORS["drho"],
        "label": "measured density flux / averaged normalization",
        "ylabel": r"$\langle V_{\rho,x}\rangle_t\,/\,\langle\ell_\perp F_\rho (z^+_{x,\rm rms})^2/z^+_{\rm rms}\rangle_t$",
        "title": r"$V_{\rho,x} = \langle(\delta\rho/\rho_0)\,\delta u_x\rangle$ vs $\chi_A$",
    },
    "V_psi": {
        "key": "V_psi_over_F_z_l",
        "closure_key": "closure",
        "text": "V_psi_x/(F_psi z+ l_perp)",
        "color": FIELD_COLORS["db_par"],
        "label": r"measured $V_{\psi,x}/[(F_u + v_S^2 F_B/v_A^2)\, z^+\ell_\perp]$",
        "ylabel": r"$V_{\psi,x}\,/\,[(F_u + v_S^2 F_B/v_A^2)\, z^+_{\rm rms}\,\ell_\perp]$",
        "title": r"$V_{\psi,x} = \langle(\delta B_\parallel/B_0)\,\delta u_x"
                 r" - \delta u_\parallel\,\delta B_x/B_0\rangle$ vs $\chi_A$",
    },
    "drho": {
        "key": "drho_rms",
        "closure_key": None,
        "text": "rms(drho)",
        "color": FIELD_COLORS["drho"],
        "label": r"measured rms$(\delta\rho/\rho_0)$",
        "ylabel": r"rms$(\delta\rho/\rho_0)$",
        "title": r"Rms drho vs $\chi_A$",
    },
}
DEFAULT_PANELS = ("V_rho", "V_psi")


def block_error(values, n_blocks, denominator=None):
    """Standard error of the window mean from `n_blocks` contiguous block means.

    Saved times are strongly correlated, so the plain standard error is far too
    small; block means of long blocks are close to independent. With a
    `denominator`, the error is that of mean(values)/mean(denominator), taken
    from the block-by-block ratios. NaN when there are too few points.
    """
    values = np.asarray(values, dtype=float)
    if n_blocks < 2 or values.size < 2 * n_blocks:
        return np.nan
    num = np.array([block.mean() for block in np.array_split(values, n_blocks)])
    if denominator is not None:
        den = np.array([block.mean() for block in np.array_split(np.asarray(denominator, dtype=float), n_blocks)])
        if np.any(den == 0.0):
            return np.nan
        num = num / den
    return float(num.std(ddof=1) / np.sqrt(n_blocks))


def run_label(run_dir):
    """parent/run, so runs called e.g. ener_0.09 in different gradient folders stay distinct."""
    run_dir = Path(run_dir)
    return f"{run_dir.parent.name}/{run_dir.name}"


def flux_ratio(flux, drive, run):
    """V / (F z+_rms l_perp) at every saved time; NaN when the drive is zero."""
    if drive == 0.0:
        return np.full_like(run.times, np.nan)
    return flux / (drive * run.z_plus * run.l_perp)


def read_run(run_dir, *, tmin=None, tmax=None, tail_fraction=0.4, n_blocks=5):
    """Window means of chi_A, the dimensionless fluxes and their estimate for one run."""
    run = load_run(run_dir, ["drho"])
    window = run_window(run, tmin=tmin, tmax=tmax, tail_fraction=tail_fraction)
    f_rho, f_psi = run.drives["drho"], run.f_psi

    def mean(values):
        """Plain window mean: a NaN in the window (missing column, zero drive) makes it NaN."""
        return float(np.asarray(values)[window].mean())

    point = {
        "run": run_label(run.run_dir),
        "K_rho0": float(run.p.K_rho0),
        "t_start": float(run.times[window][0]),
        "t_end": float(run.times[window][-1]),
        "F_rho": float(f_rho),
        "F_psi": float(f_psi),
        "chi_A": mean(run.chi_a),
        # Spatial averaging gives <z+_x z+_x> = rms(z+_x)^2. Average
        # numerator and denominator in time before dividing (not mean(V/D)).
        "V_rho_normalization": mean(f_rho * run.l_perp * safe_divide(run.z_plus_x**2, run.z_plus)),
        "V_psi_over_F_z_l": mean(flux_ratio(run.v_psi_x, f_psi, run)),
        # The estimate of both ratios: eta_turb / (z+ l_perp) = z+_x^2 / (2 z+^2).
        "closure": mean(safe_divide(run.eta_turb, run.z_plus * run.l_perp)),
        "z_plus_x_over_z_plus": mean(run.alignment),
        # Dimensional means, for reference.
        "V_rho_x": mean(run.v_rho_x),
        "V_rho_x_predicted": mean(run.v_rho_x_predicted),
        "V_psi_x": mean(run.v_psi_x),
        "V_psi_x_predicted": mean(run.v_psi_x_predicted),
        "eta_turb": mean(run.eta_turb),
        "drho_rms": mean(run.measured["drho"]),
    }
    normalization = point["V_rho_normalization"]
    # F_rho may be negative: retain the signed denominator. safe_divide
    # elsewhere in run_quantities is intended for positive RMS quantities.
    point["V_rho_normalized"] = (
        point["V_rho_x"] / normalization if abs(normalization) > 0.0 else np.nan
    )
    rho_norm_t = f_rho * run.l_perp * safe_divide(run.z_plus_x**2, run.z_plus)
    point["V_rho_normalized_err"] = block_error(run.v_rho_x[window], n_blocks, rho_norm_t[window])
    point["V_psi_over_F_z_l_err"] = block_error(flux_ratio(run.v_psi_x, f_psi, run)[window], n_blocks)
    point["drho_rms_err"] = block_error(run.measured["drho"][window], n_blocks)
    name = point["run"]
    if f_rho == 0.0:
        print(f"{name}: F_rho = 0, so the density normalization is zero; no V_rho point.")
    elif np.isnan(point["V_rho_x"]):
        print(f"{name}: no V_rho_x (no column, and g = 0 so acr_g/g cannot stand in); rerun to save it.")
    if f_psi == 0.0:
        print(f"{name}: F_psi = F_u + vS^2/vA^2 F_B = 0, so V_psi_x/(F_psi z+ l_perp) is undefined; "
              "no V_psi point.")
    elif np.isnan(point["V_psi_x"]):
        print(f"{name}: no V_psi_x (no column, and -acr_B/(vA^2 K_b0) cannot stand in); rerun to save it.")
    return point


def print_table(points):
    print(f"{'run':<36} {'window':>12} {'K_rho0':>6} {'chi_A':>6} {'Vrho/norm':>17} "
          f"{'Vpsi/(F z+ l)':>19} {'estimate':>8} {'z+_x/z+':>7}")
    for pt in points:
        window = f"{pt['t_start']:.3g}-{pt['t_end']:.3g}"
        vrho = f"{pt['V_rho_normalized']:.3g}+-{pt['V_rho_normalized_err']:.2g}"
        vpsi = f"{pt['V_psi_over_F_z_l']:.3g}+-{pt['V_psi_over_F_z_l_err']:.2g}"
        print(f"{pt['run']:<36} {window:>12} {pt['K_rho0']:>6.3g} {pt['chi_A']:>6.3g} {vrho:>17} "
              f"{vpsi:>19} {pt['closure']:>8.3g} {pt['z_plus_x_over_z_plus']:>7.3g}")


def check_windows(points):
    """Warn when runs are averaged over very different windows (e.g. one still running)."""
    lengths = np.array([pt["t_end"] - pt["t_start"] for pt in points])
    starts = np.array([pt["t_start"] for pt in points])
    if len(points) > 1 and (lengths.max() > 2.0 * lengths.min() or starts.max() > 2.0 * starts.min()):
        print("Warning: the averaging windows differ a lot between runs; a short or early window "
              "(a run still in its transient) is not comparable. Consider a common --tmin/--tmax.")
        for pt in points:
            print(f"    {pt['run']}: t = {pt['t_start']:.3g}-{pt['t_end']:.3g}")


X_AXES = {
    "chi_A": r"$\chi_A = z^+_{\rm rms}\langle k_\perp\rangle/(v_A\langle k_\parallel\rangle)$",
    "K_rho0": r"$K_{\rho 0}$ (background density gradient)",
}


def short_labels(points):
    """The part of parent/run that differs between points: run if the parents match, else parent."""
    parents = [pt["run"].split("/")[0] for pt in points]
    names = [pt["run"].split("/")[-1] for pt in points]
    if len(set(parents)) == 1:
        return names
    if len(set(names)) == 1:
        return parents
    return [pt["run"] for pt in points]


def _linthresh(values):
    """symlog linear range: below the smallest nonzero |value|, so every point sits on the log part."""
    finite = np.abs(np.asarray([v for v in values if np.isfinite(v) and v != 0.0]))
    return float(10 ** np.floor(np.log10(finite.min()))) if finite.size else 1e-3


def plot_closure(points, output_path, *, panels=DEFAULT_PANELS, window_text, show=False,
                 x_key="chi_A", yscale="symlog", annotate=True):
    """One panel per name in `panels` (see PANELS), each against `x_key`."""
    plt = import_pyplot(show=show)
    fig, axes = plt.subplots(1, len(panels), figsize=(6.0 * len(panels), 5.0),
                             squeeze=False, constrained_layout=True)
    chi = np.array([pt[x_key] for pt in points])

    for ax, name in zip(axes[0], panels):
        panel = PANELS[name]
        values = np.array([pt[panel["key"]] for pt in points])
        errors = np.array([pt.get(panel["key"] + "_err", np.nan) for pt in points])
        if yscale == "log":
            for pt, value in zip(points, values):
                if value <= 0.0:  # NaN compares False, so only real non-positive values warn.
                    print(f"{pt['run']}: {panel['text']} = {value:.3g} <= 0; not shown on the log axis.")

        # Filled: measured (hollow if negative). Hollow black squares: the run's own estimate.
        negative = values < 0.0
        for mask, label, face in ((~negative, panel["label"], panel["color"]),
                                  (negative, "(negative value)", "white")):
            if mask.any():
                ax.errorbar(chi[mask], values[mask], yerr=np.nan_to_num(errors[mask]), ls="none",
                            marker="o", ms=7, mfc=face, mec=panel["color"], ecolor=panel["color"],
                            elinewidth=1, capsize=3, label=label)
        if annotate:
            for x, y, label in zip(chi, values, short_labels(points)):
                if np.isfinite(y):
                    ax.annotate(label, (x, y), textcoords="offset points",
                                xytext=(4, 4), fontsize=7, color="0.3")
        if "reference" in panel:
            ax.axhline(panel["reference"], color="0.5", ls="--", lw=1.0, label="closure 1/2")
        if panel["closure_key"] is not None:
            ax.plot(chi, [pt[panel["closure_key"]] for pt in points], ls="none", marker="s", ms=8,
                    mfc="none", mec="black", label=r"estimate $(z^+_x)^2/(2 (z^+_{\rm rms})^2)$")
            ax.axhline(ISOTROPIC_CLOSURE, color="0.5", ls="--", lw=1.0, label="isotropic 1/4")
        if x_key == "chi_A":
            ax.axvline(1.0, color="0.5", ls=":", lw=1.0)  # The closure assumes chi_A >~ 1.
        if np.all(chi > 0):
            ax.set_xscale("log")
        # symlog only matters when something is negative; otherwise plain log is cleaner.
        if yscale == "symlog" and np.any(values[np.isfinite(values)] <= 0.0):
            shown = list(values) + ([pt[panel["closure_key"]] for pt in points] if panel["closure_key"] else [])
            ax.set_yscale("symlog", linthresh=_linthresh(shown))
            ax.axhline(0.0, color="0.3", lw=0.8)
        else:
            ax.set_yscale("log" if yscale == "symlog" else yscale)
        x_name = r"$\chi_A$" if x_key == "chi_A" else r"$K_{\rho 0}$"
        ax.set(xlabel=X_AXES[x_key], ylabel=panel["ylabel"],
               title=panel["title"].replace(r"vs $\chi_A$", f"vs {x_name}"))
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=9)

    fig.suptitle(f"Flux closure diagnostics | window means, {window_text}", fontsize=11)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    finalize_figure(fig, output_path=output_path, show=show, plt=plt)


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
    parser.add_argument("--output", type=Path,
                        help="Image path; default flux_closure.png in the run folder (one run) "
                             "or the folder holding the runs.")
    parser.add_argument("--summary-output", type=Path, help="CSV of window means; default alongside the image.")
    parser.add_argument("--x", dest="x_key", choices=list(X_AXES), default="chi_A",
                        help="Horizontal axis: chi_A (default) or K_rho0 for a gradient scan.")
    parser.add_argument("--yscale", choices=["symlog", "log", "linear"], default="symlog",
                        help="symlog (default) shows negative ratios; log hides them.")
    parser.add_argument("--blocks", type=int, default=5,
                        help="Blocks for the error bars on the window means; default 5 (0 to skip).")
    parser.add_argument("--no-labels", action="store_true", help="Do not label points with their folder.")
    parser.add_argument("--show", action="store_true", help="Show the figure after saving (e.g. in Spyder).")
    return parser


def main(argv=None) -> Path:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run_dirs = resolve_run_dirs(args.paths)
        points = [read_run(run_dir, tmin=args.tmin, tmax=args.tmax, tail_fraction=args.tail_fraction,
                           n_blocks=args.blocks)
                  for run_dir in run_dirs]
    except (OSError, ValueError) as error:
        parser.error(str(error))

    output = (args.output or default_output_dir(run_dirs) / "flux_closure.png").expanduser().resolve()
    print_table(points)
    check_windows(points)
    write_summary_csv(points, args.summary_output or output.with_suffix(".csv"))
    panels = list(dict.fromkeys(args.panels))  # Drop repeats, keep the order given.
    plot_closure(points, output, panels=panels,
                 window_text=describe_window(args.tmin, args.tmax, args.tail_fraction), show=args.show,
                 x_key=args.x_key, yscale=args.yscale, annotate=not args.no_labels)
    return output


if __name__ == "__main__":
    main()
