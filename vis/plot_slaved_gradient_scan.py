"""Plot log10(mean saturated RMS) against log10(|F|), one point per run.

Read the saved RMS columns in scalar_diagnostics.csv and the background physics
in input_copy.input. By default, average the last 40% of each run's time span.
Use --tmin to choose the start of a saturated interval; this script does not
automatically determine whether a run has saturated.

The average is the arithmetic mean of saved RMS samples, as in the original
scan. Take the logarithm AFTER averaging the RMS, not of the signed field.
Density is delta rho/rho_0, velocity is delta u_parallel/vA, and the magnetic
field is (vA^2/vS^2) delta B_parallel/B_0 (compressive_channels.slaved_field_units).

Each panel also shows the slaved estimate of Eq. (47),
|F| * l_perp * rms(z+_x)/z+_rms, averaged over the same window and logged the
same way. It is computed by vis/run_quantities.py from the z+ columns
k_perp_plus and w_plus_align; a CSV without them (for example one
written before 2026-09-23) keeps only its measured point. Each series gets an
ordinary least-squares line through its (log10|F|, log10 RMS) points, so the
slope in the legend is the power-law exponent: 1 when the amplitude is
proportional to its drive. A line needs at least two different drives.

Examples (also usable as main([...]) in Spyder):
    python vis/plot_slaved_gradient_scan.py RUN_1 RUN_2 RUN_3 --tmin 120
    python vis/plot_slaved_gradient_scan.py "examples/outputs_*_rho" --fields drho
"""

import argparse
import csv
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from vis._matplotlib import finalize_figure, import_pyplot
from vis.plot_slaved_projection import COLORS, LABELS
from vis.run_quantities import FIELDS, load_run, resolve_run_dirs, time_window


TITLES = {"drho": "Density", "du_par": "Parallel velocity", "db_par": "Parallel magnetic field"}
DRIVE_LABELS = {"drho": r"F_\rho", "du_par": r"F_u", "db_par": r"F_b"}
MEASURED_LABEL = "Measured RMS"
SLAVED_LABEL = "Slaved estimate (Eq. 47)"


def read_run(run_dir, fields, *, tmin=None, tail_fraction=0.4):
    """Return each field's drive, mean normalized RMS and mean slaved estimate in one window."""
    run = load_run(run_dir, fields)
    try:
        window = time_window(run.times, tmin=tmin, tail_fraction=tail_fraction)
    except ValueError as error:
        raise ValueError(f"{run.run_dir.name}: {error}") from None

    points = []
    for name in fields:
        rms = run.measured[name][window]
        if not np.isfinite(rms).all() or np.any(rms < 0):
            raise ValueError(f"{run.run_dir.name}: invalid {name}_rms in the averaging window.")
        # Same samples and same arithmetic mean as the measured RMS, in the same units. It is
        # NaN where the CSV lacks the z+ columns it needs; such a run still gives its measured point.
        estimate = run.predicted[name][window]
        slaved = float(estimate.mean()) if np.isfinite(estimate).all() else float("nan")
        if np.isnan(slaved):
            print(f"{run.run_dir.name}: no finite slaved {name} estimate in the window; omitted.")
        points.append({
            "run_dir": str(run.run_dir),
            "field": name,
            "forcing": run.drives[name],
            "saturated_rms": float(rms.mean()),
            "slaved_rms": slaved,
            "t_start": float(run.times[window][0]),
            "t_end": float(run.times[window][-1]),
            "n_samples": int(window.sum()),
        })
    return points


def write_summary_csv(points, path):
    """Save the plotted amplitudes and averaging windows for reference."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)
    print(f"Saved {path}")


def log10_points(field_points, key):
    """log10|F| and log10 of one amplitude, for the points where both logarithms exist."""
    kept = [point for point in field_points
            if abs(point["forcing"]) > 0 and np.isfinite(point[key]) and point[key] > 0]
    return (np.log10([abs(point["forcing"]) for point in kept]),
            np.log10([point[key] for point in kept]))


def fit_line(x, y):
    """Least-squares line y = slope * x + intercept, or None without two different x."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.size < 2 or np.ptp(x) < 1e-12:
        return None
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def draw_series(ax, x, y, *, label, fit_label, marker_style, line_style):
    """Scatter one series and, when its drive varies, its regression line; return the fit."""
    if not x.size:
        return None
    ax.scatter(x, y, label=label, zorder=3, **marker_style)
    fit = fit_line(x, y)
    if fit is not None:
        slope, intercept = fit
        ends = np.array([x.min(), x.max()])  # Only across the drives that were fitted.
        ax.plot(ends, slope * ends + intercept, lw=1.6,
                label=f"{fit_label}: slope {slope:.2f}", **line_style)
    return fit


def plot_scan(points, fields, output_path, *, show=False, parameter=None):
    """Draw measured and slaved points, a regression line for each, and a legend."""
    plt = import_pyplot(show=show)
    fig, axes = plt.subplots(1, len(fields), figsize=(5 * len(fields), 4.6),
                             constrained_layout=True, squeeze=False)
    for ax, name in zip(axes[0], fields):
        title = TITLES[name]
        field_points = [point for point in points if point["field"] == name]
        x, y = log10_points(field_points, "saturated_rms")
        x_slaved, y_slaved = log10_points(field_points, "slaved_rms")
        if x.size or x_slaved.size:
            # Measured: filled, solid line, field colour. Slaved: hollow, dashed, black,
            # as in plot_slaved_projection.py.
            fits = {
                "measured": draw_series(
                    ax, x, y, label=MEASURED_LABEL, fit_label="Measured fit",
                    marker_style=dict(color=COLORS[name], s=45),
                    line_style=dict(color=COLORS[name], ls="-")),
                "slaved": draw_series(
                    ax, x_slaved, y_slaved, label=SLAVED_LABEL, fit_label="Slaved fit",
                    marker_style=dict(marker="s", s=45, facecolors="none", edgecolors="black"),
                    line_style=dict(color="black", ls="--")),
            }
            for kind, fit in fits.items():
                if fit is not None:
                    print(f"{name} {kind} fit: slope {fit[0]:.3f}, intercept {fit[1]:.3f} "
                          "(log10 units)")
            if len(x) > 1 and np.ptp(x) == 0:
                title += "\n(same drive in every run)"
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "No points with positive |F| and RMS", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        omitted = len(field_points) - len(x)
        if omitted:
            print(f"{name}: omitted {omitted} point(s) with zero drive or RMS (log undefined).")
        ax.set(xlabel=rf"$\log_{{10}} |{DRIVE_LABELS[name]}|$",
               ylabel=rf"$\log_{{10}}\langle\mathrm{{RMS}}({LABELS[name]})\rangle_t$",
               title=title)
        ax.grid(alpha=0.3)

    starts = [point["t_start"] for point in points]
    window = f"t >= {starts[0]:g}" if min(starts) == max(starts) else "window chosen per run"
    title = "Saturated RMS versus background drive"
    if parameter:
        title += f" | scanned {parameter}"
    fig.suptitle(f"{title}\nMean of saved RMS samples, {window}", fontsize=11)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    finalize_figure(fig, output_path=output_path, show=show, plt=plt)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="Run directories; quoted wildcards are supported.")
    parser.add_argument("--fields", nargs="+", choices=FIELDS, default=list(FIELDS))
    parser.add_argument("--tmin", type=float, help="Start of the saturated averaging interval.")
    parser.add_argument("--tail-fraction", type=float, default=0.4,
                        help="Fraction of the time span to average; default 0.4.")
    parser.add_argument("--parameter", choices=("g", "K_p0", "K_rho0"),
                        help="Optional scanned-parameter label for the title.")
    parser.add_argument("--output", type=Path, help="Image path; default slaved_gradient_scan.png.")
    parser.add_argument("--summary-output", type=Path, help="CSV path; default alongside the image.")
    parser.add_argument("--show", action="store_true", help="Display the saved figure.")
    args = parser.parse_args(argv)

    points = []
    try:
        run_dirs = resolve_run_dirs(args.paths)
        for run_dir in run_dirs:
            points.extend(read_run(run_dir, args.fields, tmin=args.tmin,
                                   tail_fraction=args.tail_fraction))
    except (OSError, ValueError) as error:
        parser.error(str(error))

    parents = {path.parent for path in run_dirs}
    output_dir = parents.pop() if len(parents) == 1 else Path.cwd()
    output = args.output or output_dir / "slaved_gradient_scan.png"
    write_summary_csv(points, args.summary_output or output.with_suffix(".csv"))
    plot_scan(points, args.fields, output, show=args.show, parameter=args.parameter)
    return output


if __name__ == "__main__":
    main()
