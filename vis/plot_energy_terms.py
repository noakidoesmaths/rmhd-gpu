"""How much each budget term contributes to each energy reservoir, averaged over a time window.

One panel per reservoir (W+, W-, W_par^u, W_par^B, W^p, W_s). Each bar is
one saved budget term `<reservoir>_rhs_<term>`, averaged over the window:
green bars add energy and red bars remove it. Each bar's label gives its
share: a gain as a percentage of all the reservoir's gains, a loss as a
percentage of all its losses. The title gives the residual, measured d_t W
over the window minus the sum of the terms, which should be small next to
the bars; the measured d_t W itself is in the CSV.

The error bars show how steady each average is. The window is cut into
`--blocks` equal pieces, and the bar is the standard error of the block means.
A term whose bar is shorter than its error bar has no clear sign over the
window.

The terms (see `rmhdgpu/diagnostics/compressive_channels.py`):

    buoyancy      (g/2) <drho z±_x>                       W±
    drive_KB      vA K_b0 <du_par b_x>                    W_par^u        (the grad-B term)
    drive_KBKp    -vS^2 (K_b0 - K_p0/gamma) <db_par u_x>  W_par^B, W^p   (grad-B and grad-p)
    drive_Ks      -W_s-weight K_s <s u_x>                 W_s            (grad-s)
    exchange_lin  vA^2 <du_par d_z db_par> and partner     W_par^u <-> W_par^B + W^p
    exchange_nl   vA <du_par {psi, db_par}> and partner    (the delta B_perp . grad terms)
    dissipation, forcing, projection

`--hide projection` (or any other term names) leaves those terms off the
panels. The shares and the residual still count them, so the shares of the
bars that are drawn need not add up to 100%.

The same numbers are written to a CSV next to the figure. For the terms
against time, use `vis/plot_budget.py CSV --quantity w_par_u`.

    python vis/plot_energy_terms.py RUN_DIRECTORY
    python vis/plot_energy_terms.py RUN_DIRECTORY --tmin 5 --blocks 8 --show
"""

import argparse
import csv
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from vis._matplotlib import finalize_figure, import_pyplot
from vis.run_quantities import RESERVOIR_LABELS as TITLES
from vis.run_quantities import RESERVOIRS, TERM_LABELS, budget_rates, budget_rows, read_columns



def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="Run directory, or its scalar_diagnostics.csv.")
    parser.add_argument("--tmin", type=float, help="Start of the averaging window (default: last 40%% of the run).")
    parser.add_argument("--tmax", type=float, help="End of the averaging window (default: end of the run).")
    parser.add_argument("--hide", nargs="+", default=[], metavar="TERM",
                        help="Budget terms not to draw, e.g. projection (shares and residual still count them).")
    parser.add_argument("--blocks", type=int, default=4, help="Blocks for the error bars (default 4).")
    parser.add_argument("--output", type=Path, help="Image path; default: RUN_DIRECTORY/energy_terms.png.")
    parser.add_argument("--show", action="store_true", help="Show the figure after saving (e.g. in Spyder).")
    args = parser.parse_args(argv)

    run_dir = args.run_dir.expanduser().resolve()
    if run_dir.is_file():  # the CSV was given
        run_dir = run_dir.parent
    output = args.output or run_dir / "energy_terms.png"

    columns = read_columns(run_dir / "scalar_diagnostics.csv")
    missing = [name for name in RESERVOIRS if f"{name}_rhs_total" not in columns]
    if missing:
        raise SystemExit(f"{run_dir}/scalar_diagnostics.csv has no budget columns for {', '.join(missing)}; "
                         "rerun with a version that writes the energy reservoirs.")
    time = columns["time"]
    rows = budget_rows(time, tmin=args.tmin, tmax=args.tmax)
    blocks = [block for block in np.array_split(rows, min(max(args.blocks, 1), rows.size)) if block.size]

    plt = import_pyplot(show=args.show)
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    table = []  # (quantity, term, rate, std_err, share)

    for ax, name in zip(axes.flat, RESERVOIRS):
        rates = budget_rates(columns, name, rows)
        block_rates = [budget_rates(columns, name, block) for block in blocks]
        terms = [term for term in rates
                 if term not in ("measured_dt", "residual") and np.any(columns[f"{name}_rhs_{term}"][rows] != 0.0)]
        gains = sum(rates[term] for term in terms if rates[term] > 0)
        losses = -sum(rates[term] for term in terms if rates[term] < 0)

        labels, values, errors, colors = [], [], [], []
        for term in terms:
            rate = rates[term]
            error = np.std([b[term] for b in block_rates], ddof=1) / np.sqrt(len(blocks)) if len(blocks) > 1 else 0.0
            share = rate / gains if rate > 0 else (rate / losses if rate < 0 else 0.0)
            table.append((name, term, rate, error, share))
            if term in args.hide:
                continue
            labels.append(f"{TERM_LABELS.get(term, term)}\n{abs(share):.0%} of {'gains' if rate > 0 else 'losses'}")
            values.append(rate)
            errors.append(error)
            colors.append("tab:green" if rate > 0 else "tab:red")
        table.append((name, "measured_dt", rates["measured_dt"], 0.0, np.nan))
        table.append((name, "residual", rates["residual"], 0.0, np.nan))

        position = np.arange(len(values))[::-1]
        ax.barh(position, values, xerr=errors, color=colors, capsize=3)
        ax.set_yticks(position, labels, fontsize=8)
        ax.axvline(0.0, color="k", lw=0.8)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 3))
        ax.set_xlabel("rate")
        ax.set_title(f"{TITLES[name]}   (residual {rates['residual']:.1e})")

    table_path = output.with_suffix(".csv")
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["quantity", "term", "rate", "std_err", "share", "t_start", "t_end"])
        for name, term, rate, error, share in table:
            writer.writerow([name, term, f"{rate:.10e}", f"{error:.3e}", f"{share:.4f}",
                             time[rows[0] - 1], time[rows[-1]]])
    print(f"Saved {table_path}")

    fig.suptitle(f"{run_dir.parent.name}/{run_dir.name}: budget terms, mean over "
                 f"t = {time[rows[0] - 1]:.3g} to {time[rows[-1]]:.3g} ({len(blocks)} blocks for error bars)")
    finalize_figure(fig, output_path=output, show=args.show, plt=plt)
    return output


if __name__ == "__main__":
    main()
