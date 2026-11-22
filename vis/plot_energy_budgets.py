"""Energy budget of each reservoir against time: every term, the measured d_t W, and the residual.

One panel per reservoir (W+, W-, W_par^u, W_par^B, W^p, W_s) in
`energy_budgets.png`. Each panel shows:

    coloured lines   every saved term <reservoir>_rhs_<term> (sign as booked:
                     positive adds energy)
    grey dashed      measured d_t W = (W_i - W_{i-1}) / (t_i - t_{i-1})
    thick black      residual = measured d_t W - sum of the terms

The saved terms are averages over each output interval, so the measured
difference and the terms cover exactly the same interval and the residual is
only the time-integration error. It should sit near zero.

`--average N` merges every N output intervals into one, which smooths noisy
terms such as stochastic forcing; the residual stays exact. `--separate` also
writes one larger figure per reservoir, `budget_<reservoir>.png`, with W(t)
above its budget. The projection term (energy removed by the k_par = 0 /
kx = 0 / ky = 0 projections) is not drawn by default; `--hide` sets which
terms are left off (`--hide` with no names draws everything). The residual
always includes every term, so it stays a true check of the budget.

The term names are listed in `vis/plot_energy_terms.py`, which shows the same
terms averaged over a window instead of against time.

    python vis/plot_energy_budgets.py RUN_DIRECTORY
    python vis/plot_energy_budgets.py RUN_DIRECTORY --average 5 --tmin 2 --separate --show
"""

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from vis._matplotlib import finalize_figure, import_pyplot
from vis.run_quantities import RESERVOIR_LABELS, RESERVOIRS, TERM_COLORS, TERM_LABELS, read_columns

LINESTYLES = {"exchange_nl": "--", "drive_KBKp": "-", "drive_KB": "-", "drive_Ks": "-"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="Run directory, or its scalar_diagnostics.csv.")
    parser.add_argument("--average", type=int, default=1, metavar="N",
                        help="Merge every N output intervals into one (default 1: as saved).")
    parser.add_argument("--tmin", type=float, help="First time to plot.")
    parser.add_argument("--tmax", type=float, help="Last time to plot.")
    parser.add_argument("--hide", nargs="*", default=["projection"], metavar="TERM",
                        help="Budget terms not to draw (default: projection; pass --hide alone to draw all). "
                             "The residual still includes them.")
    parser.add_argument("--separate", action="store_true",
                        help="Also write budget_<reservoir>.png, one figure per reservoir with W(t) on top.")
    parser.add_argument("--output", type=Path, help="Image path; default: RUN_DIRECTORY/energy_budgets.png.")
    parser.add_argument("--show", action="store_true", help="Show the figures after saving (e.g. in Spyder).")
    args = parser.parse_args(argv)

    run_dir = args.run_dir.expanduser().resolve()
    if run_dir.is_file():  # the CSV was given
        run_dir = run_dir.parent
    output = args.output or run_dir / "energy_budgets.png"

    columns = read_columns(run_dir / "scalar_diagnostics.csv")
    missing = [name for name in RESERVOIRS if f"{name}_rhs_total" not in columns]
    if missing:
        raise SystemExit(f"{run_dir}/scalar_diagnostics.csv has no budget columns for {', '.join(missing)}; "
                         "rerun with a version that writes the energy reservoirs.")
    time = columns["time"]

    # Interval ends kept after merging every N intervals: rows 0, N, 2N, ...
    # The rate of a term over a merged interval is the integral of its
    # interval averages, sum(rate_i dt_i), over the merged length.
    step = max(args.average, 1)
    ends = np.arange(0, time.size, step)
    interval = np.diff(time, prepend=time[0])  # dt_i for row i; row 0 has none

    def merged_rate(series):
        integral = np.cumsum(series * interval)
        return np.diff(integral[ends]) / np.diff(time[ends])

    t_mid = 0.5 * (time[ends][1:] + time[ends][:-1])
    shown = np.ones(t_mid.size, dtype=bool)
    if args.tmin is not None:
        shown &= t_mid >= args.tmin
    if args.tmax is not None:
        shown &= t_mid <= args.tmax
    if not shown.any():
        raise SystemExit("No output intervals in the requested time range.")

    # Every term, the measured d_t W and the residual of each reservoir.
    budgets = {}
    for name in RESERVOIRS:
        prefix = f"{name}_rhs_"
        terms = sorted(col[len(prefix):] for col in columns if col.startswith(prefix) and col != prefix + "total")
        terms = [term for term in terms if np.any(columns[prefix + term][1:] != 0.0)]
        rates = {term: merged_rate(columns[prefix + term]) for term in terms}
        measured = np.diff(columns[name][ends]) / np.diff(time[ends])
        residual = measured - sum(rates.values()) if rates else measured
        budgets[name] = (rates, measured, residual)

    def draw_budget(ax, name):
        rates, measured, residual = budgets[name]
        for term, rate in rates.items():
            if term in args.hide:
                continue
            ax.plot(t_mid[shown], rate[shown], color=TERM_COLORS.get(term), ls=LINESTYLES.get(term, "-"),
                    lw=1.5, label=TERM_LABELS.get(term, term))
        # Styled as in plot_budget.py: measured d_t W light, the residual dark on top.
        ax.plot(t_mid[shown], measured[shown], color="0.6", ls="--", lw=1.6, label=r"measured $d_tW$")
        ax.plot(t_mid[shown], residual[shown], color="black", lw=3.0, label="residual")
        ax.axhline(0.0, color="k", lw=0.6)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))
        ax.grid(True, alpha=0.3)

    plt = import_pyplot(show=args.show)
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True, sharex=True)
    for ax, name in zip(axes.flat, RESERVOIRS):
        draw_budget(ax, name)
        ax.set_title(f"{RESERVOIR_LABELS[name]} budget")
        ax.legend(fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("time")
    for ax in axes[:, 0]:
        ax.set_ylabel("rate of change")
    merged = f", every {step} output intervals merged" if step > 1 else ""
    hidden = [term for term in args.hide if any(f"_rhs_{term}" in col for col in columns)]
    if hidden:
        merged += f" ({', '.join(hidden)} not drawn)"
    fig.suptitle(f"{run_dir.parent.name}/{run_dir.name}: energy budgets{merged}")
    finalize_figure(fig, output_path=output, show=args.show, plt=plt)

    if args.separate:
        for name in RESERVOIRS:
            fig, (ax_energy, ax_budget) = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True, constrained_layout=True,
                                                       gridspec_kw={"height_ratios": [1, 2]})
            window = (time >= (args.tmin if args.tmin is not None else -np.inf)) & \
                     (time <= (args.tmax if args.tmax is not None else np.inf))
            ax_energy.plot(time[window], columns[name][window], color="k", lw=2)
            ax_energy.set(ylabel=RESERVOIR_LABELS[name], title=f"{RESERVOIR_LABELS[name]} and its budget{merged}")
            ax_energy.grid(True, alpha=0.3)
            draw_budget(ax_budget, name)
            ax_budget.set(xlabel="time", ylabel="rate of change")
            ax_budget.legend(fontsize=8)
            finalize_figure(fig, output_path=output.with_name(f"budget_{name}.png"), show=args.show, plt=plt)
    return output


if __name__ == "__main__":
    main()
