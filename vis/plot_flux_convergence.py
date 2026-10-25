"""Check that the density-flux closure ratio has settled before trusting its window mean.

vis/plot_flux_closure.py turns each run into one point: the window mean of

    V_rho_x / (F_rho z+_rms l_perp)        closure: 1/4 (Squire et al., arXiv:2607.08036, Eq. 80)

This script plots that same ratio against time, so you can see whether the
window it is averaged over is statistically steady. Run it on the same runs
first. Do not judge convergence by rms(drho): with g = 0, drho is a passive
scalar, and its k_par = 0 part (the mean profile <drho>(x)) keeps growing long
after the flux has settled. That part cannot correlate with the k_z != 0
velocity, so it carries no flux.

The figure is 2x2, against time, one colour per run:

    V_rho_x/(F_rho z+ l_perp)   thin line: every saved time; markers: block means
    z-_rms/z+_rms               in RMHD z+ is only distorted by z-, so z- sets how
                                fast the flow decorrelates and V_rho depends on it
    chi_A                       the x-axis of the closure plot
    auto nu_perp                the adaptive coefficient, if auto dissipation was on

A dotted vertical line marks where each run's averaging window starts. The
window is chosen exactly as in plot_flux_closure (run_quantities.time_window):
by default the last 40% of each run, or [--tmin, --tmax].

The console table, also saved as a CSV beside the image, gives the window means
and the drift of the ratio:

    drift = (mean over the second half of the window) / (mean over the first half) - 1

A |drift| above ~0.1 means the window mean is still changing, so the closure
point is not converged yet. Every quantity comes from vis/run_quantities.py.

Examples (also usable as main([...]) in Spyder):
    python vis/plot_flux_convergence.py "examples/Kp0_forcing/init_ener_4e-4_target_0.4*"
    python vis/plot_flux_convergence.py RUN_1 RUN_2 --tmin 500 --blocks 30 --show
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from vis._matplotlib import finalize_figure, import_pyplot
from vis.plot_flux_closure import FLUX_CLOSURE, describe_window
from vis.plot_slaved_gradient_scan import write_summary_csv
from vis.run_quantities import load_run, resolve_run_dirs, time_window


def block_means(times, values, n_blocks):
    """Means of `values` over `n_blocks` equal-length time blocks.

    Returns (block centre times, block means). NaN values are ignored, and blocks
    with no finite value are dropped.
    """
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    edges = np.linspace(times[0], times[-1], n_blocks + 1)
    # Each time goes in the block whose left edge is at or below it; the last time joins the last block.
    index = np.clip(np.searchsorted(edges, times, side="right") - 1, 0, n_blocks - 1)
    centres, means = [], []
    for block in range(n_blocks):
        in_block = (index == block) & np.isfinite(values)
        if in_block.any():
            centres.append(0.5 * (edges[block] + edges[block + 1]))
            means.append(values[in_block].mean())
    return np.array(centres), np.array(means)


def drift(times, values):
    """Mean over the second half of the time span divided by the mean over the first half, minus 1."""
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    middle = 0.5 * (times[0] + times[-1])
    first = np.nanmean(values[times < middle])
    second = np.nanmean(values[times >= middle])
    return second / first - 1.0


def load_series(run_dir):
    """Time series of the closure ratio, z-/z+, chi_A and auto nu_perp for one run."""
    run = load_run(run_dir, ["drho"])
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = run.eta_meas / (run.z_plus * run.l_perp)  # eta_meas = V_rho_x / F_rho
    nu = run.columns.get("auto_dissipation_nu_perp")
    if nu is not None and not np.any(nu > 0.0):
        nu = None  # Auto dissipation was off: the column is saved as zeros.
    return {"name": run.run_dir.name, "time": run.times, "ratio": ratio,
            "z_ratio": run.z_ratio, "chi_a": run.chi_a, "nu": nu}


def summarize(series, window):
    """Window means and the drift of the closure ratio for one run."""
    t = series["time"][window]
    nu = series["nu"]
    return {
        "run": series["name"],
        "t_start": float(t[0]),
        "t_end": float(t[-1]),
        "V_over_F_z_l": float(np.nanmean(series["ratio"][window])),
        "drift": float(drift(t, series["ratio"][window])),
        "z_minus_over_z_plus": float(np.nanmean(series["z_ratio"][window])),
        "chi_A": float(np.nanmean(series["chi_a"][window])),
        "auto_nu_perp": float(np.mean(nu[window])) if nu is not None else float("nan"),
    }


def print_table(points):
    print(f"{'run':<32} {'window':>14} {'V/(F z+ l)':>10} {'drift':>7} {'z-/z+':>6} {'chi_A':>6} {'auto nu':>9}")
    for pt in points:
        window = f"{pt['t_start']:.4g}-{pt['t_end']:.4g}"
        print(f"{pt['run']:<32} {window:>14} {pt['V_over_F_z_l']:>10.3g} {pt['drift']:>+7.2f} "
              f"{pt['z_minus_over_z_plus']:>6.3g} {pt['chi_A']:>6.3g} {pt['auto_nu_perp']:>9.3g}")


def plot_convergence(all_series, points, output_path, *, n_blocks, window_text, show=False):
    plt = import_pyplot(show=show)
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), sharex=True, constrained_layout=True)
    ax_ratio, ax_z, ax_chi, ax_nu = axes.flat

    for color_index, (series, point) in enumerate(zip(all_series, points)):
        color = f"C{color_index % 10}"
        t = series["time"]
        ax_ratio.plot(t, series["ratio"], color=color, lw=0.5, alpha=0.3)
        centres, means = block_means(t, series["ratio"], n_blocks)
        ax_ratio.plot(centres, means, color=color, marker="o", ms=4, lw=1.5, label=series["name"])
        ax_z.plot(t, series["z_ratio"], color=color, lw=1.2)
        ax_chi.plot(t, series["chi_a"], color=color, lw=1.2)
        if series["nu"] is not None:
            ax_nu.plot(t, series["nu"], color=color, lw=1.2)
        for ax in axes.flat:
            ax.axvline(point["t_start"], color=color, ls=":", lw=1.0)

    ax_ratio.axhline(FLUX_CLOSURE, color="black", ls="--", lw=1.4, label="closure 1/4 (Eq. 80)")
    ax_chi.axhline(1.0, color="0.5", ls=":", lw=1.0)
    auto_series = [series for series in all_series if series["nu"] is not None]
    if auto_series:
        # ax_nu.set_yscale("log")
        # nu starts at nu_min and ramps up in the first few time units; leave out the
        # first 10% of each run when choosing the y range so the settled value is readable.
        settled = np.concatenate([s["nu"][s["time"] >= s["time"][0] + 0.1 * (s["time"][-1] - s["time"][0])]
                                  for s in auto_series])
        settled = settled[settled > 0.0]
        if settled.size:
            ax_nu.set_ylim(settled.min() / 3.0, settled.max() * 3.0)
    else:
        ax_nu.text(0.5, 0.5, "manual dissipation\n(no auto nu_perp)", transform=ax_nu.transAxes,
                   ha="center", va="center", color="0.4")

    ax_ratio.set(ylabel=r"$V_{\rho,x}\,/\,(F_\rho\, z^+_{\rm rms}\,\ell_\perp)$",
                 title=f"Closure ratio: saved values and {n_blocks} block means")
    ax_z.set(ylabel=r"$z^-_{\rm rms}/z^+_{\rm rms}$", title="Imbalance")
    ax_chi.set(ylabel=r"$\chi_A$", title="Turbulence strength", xlabel="Time")
    ax_nu.set(ylabel=r"auto $\nu_\perp$", title="Auto-dissipation coefficient", xlabel="Time")
    for ax in axes.flat:
        ax.grid(alpha=0.3)
    ax_ratio.legend(fontsize=8)

    fig.suptitle(f"Convergence of the density-flux closure | dotted lines: start of window ({window_text})",
                 fontsize=11)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    finalize_figure(fig, output_path=output_path, show=show, plt=plt)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="Run directories; quoted wildcards are supported.")
    parser.add_argument("--tmin", type=float, help="Start of the averaging window.")
    parser.add_argument("--tmax", type=float, help="End of the averaging window; default: end of the run.")
    parser.add_argument("--tail-fraction", type=float, default=0.4,
                        help="Without --tmin, use this final fraction of each run; default 0.4.")
    parser.add_argument("--blocks", type=int, default=20,
                        help="Number of equal-time blocks over each run for the block means; default 20.")
    parser.add_argument("--output", type=Path,
                        help="Image path; default flux_convergence.png inside the run folder for one run, "
                             "or in the folder holding the runs for several.")
    parser.add_argument("--summary-output", type=Path, help="CSV of window means; default alongside the image.")
    parser.add_argument("--show", action="store_true", help="Show the figure after saving (e.g. in Spyder).")
    return parser


def main(argv=None) -> Path:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.blocks < 1:
        parser.error("--blocks must be at least 1.")
    try:
        run_dirs = resolve_run_dirs(args.paths)
        all_series, points = [], []
        for run_dir in run_dirs:
            series = load_series(run_dir)
            try:
                window = time_window(series["time"], tmin=args.tmin, tmax=args.tmax,
                                     tail_fraction=args.tail_fraction)
            except ValueError as error:
                raise ValueError(f"{series['name']}: {error}") from None
            all_series.append(series)
            points.append(summarize(series, window))
    except (OSError, ValueError) as error:
        parser.error(str(error))

    # One run: save inside that run's folder. Several: in the folder that holds them all.
    parents = {path.parent for path in run_dirs}
    if len(run_dirs) == 1:
        output_dir = run_dirs[0]
    else:
        output_dir = parents.pop() if len(parents) == 1 else Path.cwd()
    output = (args.output or output_dir / "flux_convergence.png").expanduser().resolve()
    print_table(points)
    write_summary_csv(points, args.summary_output or output.with_suffix(".csv"))
    plot_convergence(all_series, points, output, n_blocks=args.blocks,
                     window_text=describe_window(args), show=args.show)
    return output


if __name__ == "__main__":
    main()
