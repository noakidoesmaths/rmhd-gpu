"""Has the run reached a statistically steady state?

A turbulent run never sits at constant energy: it keeps gaining and losing
energy in bursts. "Statistically steady" only asks that the statistics stop
changing, i.e. that the bursts swing about a fixed mean. This script checks
that over a time window in three ways:

1. Drift. Compare the mean over the first half of the window with the mean
   over the second half. In a steady state they differ only by noise.

2. Noise level. Nearby outputs are not independent: a burst lasts for some
   correlation time tau. A window of length T therefore holds only about
   N = T / (2 tau) independent samples, and a time average has error
   std / sqrt(N). The drift is quoted in units of that error ("sigma");
   below about 2 sigma it is consistent with noise. std and tau are measured
   on the fluctuations about a straight-line fit, so that a slow trend is
   not mistaken for noise.

3. Energy budget. Averaged over the window, the gains (forcing, drives) must
   balance the losses (dissipation), so the net d(total_energy)/dt should be
   small next to the dissipation rate.

The left panels show each quantity against time with the window shaded, the
window mean +- its error, and block means (--blocks) +- their errors. The
right panel shows the window-averaged terms of the total-energy budget. The
numbers are printed and saved to stationarity.csv next to the figure.

A small test has little power when N is small: with N ~ 10 even a large
drift can be within 2 sigma. If so, run longer before trusting the averages.

    python vis/plot_stationarity.py RUN_DIRECTORY --tmin 200
    python vis/plot_stationarity.py RUN_DIRECTORY --quantities total_energy w_minus --blocks 6 --show
"""

import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vis._matplotlib import finalize_figure, import_pyplot
from vis.run_quantities import budget_rates, budget_rows, read_columns, time_window, write_summary_csv

DEFAULT_QUANTITIES = ("total_energy", "w_plus", "w_minus", "drho_rms")
LABELS = {
    "total_energy": "total energy",
    "w_plus": r"$W^+$",
    "w_minus": r"$W^-$",
    "drho_rms": r"rms $\delta\rho$",
}


def correlation_time(time, values):
    """Integrated correlation time of `values`, in the same units as `time`.

    tau is the area under the autocorrelation function C(lag), taken from
    lag = 0 up to where C first drops below zero (beyond that C is mostly
    noise). The values are first put on an evenly spaced time grid, and C is
    computed for every lag at once with an FFT.
    """
    dt = np.median(np.diff(time))
    even_time = np.arange(time[0], time[-1], dt)
    fluctuation = np.interp(even_time, time, values)
    fluctuation = fluctuation - fluctuation.mean()
    if not np.any(fluctuation):  # a constant signal never decorrelates; call it 0
        return 0.0

    # Autocovariance via FFT. Padding to twice the length stops the FFT from
    # wrapping the end of the signal round onto its start.
    n = len(fluctuation)
    power = np.abs(np.fft.rfft(fluctuation, 2 * n)) ** 2
    autocovariance = np.fft.irfft(power)[:n]
    autocorrelation = autocovariance / autocovariance[0]

    below_zero = np.flatnonzero(autocorrelation < 0)
    first_zero = below_zero[0] if below_zero.size else n
    # Trapezoid rule from lag 0: half of C(0) = 1, plus the following lags.
    return float(dt * (0.5 + autocorrelation[1:first_zero].sum()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="Run directory, or its scalar_diagnostics.csv.")
    parser.add_argument("--tmin", type=float, help="Start of the window (default: the last 40%% of the run).")
    parser.add_argument("--tmax", type=float, help="End of the window (default: the end of the run).")
    parser.add_argument("--quantities", nargs="+", default=list(DEFAULT_QUANTITIES), metavar="COLUMN",
                        help="scalar_diagnostics.csv columns to test (default: %(default)s; missing ones are skipped).")
    parser.add_argument("--blocks", type=int, default=4, help="Number of block means to draw (default: 4).")
    parser.add_argument("--output", type=Path, help="Image path; default: RUN_DIRECTORY/stationarity.png.")
    parser.add_argument("--show", action="store_true", help="Show the figure after saving (e.g. in Spyder).")
    args = parser.parse_args(argv)

    run_dir = args.run_dir.expanduser().resolve()
    csv_path = run_dir if run_dir.is_file() else run_dir / "scalar_diagnostics.csv"
    output = args.output or csv_path.parent / "stationarity.png"

    columns = read_columns(csv_path)
    time = columns["time"]
    window = time_window(time, tmin=args.tmin, tmax=args.tmax)
    window_start, window_end = time[window][0], time[window][-1]
    window_length = window_end - window_start

    quantities = [name for name in args.quantities if name in columns]
    skipped = [name for name in args.quantities if name not in columns]
    if skipped:
        print(f"Not in the CSV, skipped: {', '.join(skipped)}")
    if not quantities:
        raise SystemExit("None of the requested quantities are in the CSV; nothing to test.")

    # ------------------------------------------------------------------
    # Steps 1 and 2: drift between the two halves of the window, in units of its noise.
    # ------------------------------------------------------------------
    results = []
    for name in quantities:
        values = columns[name][window]
        window_time = time[window]
        mean = values.mean()

        # Measure the noise from the fluctuations about a straight-line fit, not
        # about the mean. A slow trend would otherwise count as "noise": it
        # inflates both std and tau, and the drift would hide inside its own error bar.
        straight_line = np.polyval(np.polyfit(window_time, values, 1), window_time)
        fluctuation = values - straight_line
        std = fluctuation.std()
        tau = correlation_time(window_time, fluctuation)
        n_independent = window_length / (2 * tau) if tau > 0 else np.inf
        error_of_mean = std / np.sqrt(n_independent)

        midpoint = window_start + window_length / 2
        first_half = values[window_time < midpoint].mean()
        second_half = values[window_time >= midpoint].mean()
        # Each half holds half of the independent samples, so its mean has error
        # error_of_mean * sqrt(2); the difference of two such means has error 2 * error_of_mean.
        error_of_difference = 2 * error_of_mean
        drift_sigma = abs(second_half - first_half) / error_of_difference if error_of_difference > 0 else 0.0

        results.append({
            "quantity": name,
            "mean": mean,
            "std": std,
            "correlation_time": tau,
            "n_independent": n_independent,
            "error_of_mean": error_of_mean,
            "first_half_mean": first_half,
            "second_half_mean": second_half,
            "relative_drift": (second_half - first_half) / mean if mean != 0 else np.nan,
            "drift_sigma": drift_sigma,
            "steady": drift_sigma < 2,
        })

    print(f"\nWindow t = {window_start:g} to {window_end:g} (length {window_length:g})")
    print(f"{'quantity':>14} {'mean':>10} {'std/mean':>9} {'tau':>7} {'N indep':>8} "
          f"{'error/mean':>10} {'drift':>8} {'sigma':>6}  verdict")
    for row in results:
        verdict = "steady" if row["steady"] else "DRIFTING"
        print(f"{row['quantity']:>14} {row['mean']:10.4g} {row['std'] / row['mean']:9.3f} "
              f"{row['correlation_time']:7.2f} {row['n_independent']:8.1f} "
              f"{row['error_of_mean'] / row['mean']:10.3f} {row['relative_drift']:+8.3f} "
              f"{row['drift_sigma']:6.1f}  {verdict}")

    # ------------------------------------------------------------------
    # Step 3: window-averaged budget of the total energy.
    # ------------------------------------------------------------------
    has_budget = "total_energy_rhs_total" in columns
    if has_budget:
        rows = budget_rows(time, tmin=args.tmin, tmax=args.tmax)
        rates = budget_rates(columns, "total_energy", rows)
        measured = rates.pop("measured_dt")
        residual = rates.pop("residual")
        terms = {term: rate for term, rate in rates.items() if rate != 0.0}  # drop terms that are off
        dissipation = abs(rates.get("dissipation", np.nan))
        imbalance = measured / dissipation
        print("\nWindow-averaged d(total_energy)/dt:")
        for term, rate in terms.items():
            print(f"  {term:>14} {rate:+.4e}")
        print(f"  {'measured':>14} {measured:+.4e}   = {imbalance:+.2%} of the dissipation rate")
        print(f"  {'residual':>14} {residual:+.4e}   (measured minus the sum of the terms)")
    else:
        print("\nNo total_energy_rhs_* columns in the CSV; skipping the budget panel.")

    write_summary_csv(results, output.with_name("stationarity.csv"))

    # ------------------------------------------------------------------
    # Figure: one time-series panel per quantity on the left, the budget on the right.
    # ------------------------------------------------------------------
    plt = import_pyplot(show=args.show)
    mosaic = [[name, "budget"] for name in quantities]
    fig, axes = plt.subplot_mosaic(mosaic, figsize=(15, 2.8 * len(quantities) + 1),
                                   width_ratios=[2.2, 1], constrained_layout=True)

    for row in results:
        name = row["quantity"]
        ax = axes[name]
        ax.plot(time, columns[name], color="tab:blue", lw=1)
        ax.axvspan(window_start, window_end, color="0.92", zorder=0, label="window")

        # Window mean and its error band.
        mean, error = row["mean"], row["error_of_mean"]
        ax.hlines(mean, window_start, window_end, color="k", lw=1.5, label="window mean")
        ax.fill_between([window_start, window_end], mean - error, mean + error, color="k", alpha=0.15, lw=0)

        # Block means: each block holds 1/blocks of the independent samples.
        block_edges = np.linspace(window_start, window_end, args.blocks + 1)
        block_error = row["std"] / np.sqrt(row["n_independent"] / args.blocks)
        for start, end in zip(block_edges[:-1], block_edges[1:]):
            in_block = window & (time >= start) & (time <= end)
            ax.errorbar((start + end) / 2, columns[name][in_block].mean(), xerr=(end - start) / 2,
                        yerr=block_error, color="tab:red", capsize=3, lw=1.5)

        verdict = "steady" if row["steady"] else "drifting"
        ax.set_title(f"{LABELS.get(name, name)}: drift {row['relative_drift']:+.1%} "
                     f"({row['drift_sigma']:.1f}$\\sigma$, {verdict}), $\\tau$ = {row['correlation_time']:.1f}, "
                     f"N $\\approx$ {row['n_independent']:.0f} independent", fontsize=10)
        ax.set_ylabel(LABELS.get(name, name))
        ax.grid(alpha=0.3)
    axes[quantities[0]].plot([], [], color="tab:red", label=f"{args.blocks} block means")
    axes[quantities[0]].legend(loc="upper left", fontsize=8)
    axes[quantities[-1]].set_xlabel("time")

    ax_budget = axes["budget"]
    if has_budget:
        labels = list(terms) + ["measured dE/dt", "residual"]
        values = list(terms.values()) + [measured, residual]
        colors = ["tab:green" if value > 0 else "tab:red" for value in terms.values()] + ["k", "0.6"]
        ax_budget.barh(labels, values, color=colors)
        ax_budget.axvline(0, color="k", lw=0.8)
        ax_budget.invert_yaxis()
        ax_budget.set(xlabel="window-averaged rate of change of total energy",
                      title=f"Total-energy budget\nnet = {imbalance:+.1%} of dissipation")
        ax_budget.grid(axis="x", alpha=0.3)
    else:
        ax_budget.set_axis_off()

    fig.suptitle(f"{csv_path.parent.name}: stationarity over t = {window_start:g} to {window_end:g}")
    finalize_figure(fig, output_path=output, show=args.show, plt=plt)
    return output


if __name__ == "__main__":
    main()
