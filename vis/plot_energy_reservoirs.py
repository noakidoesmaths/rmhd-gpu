"""The six energy reservoirs of the inhomogeneous RMHD sets, against time.

    E = W+ + W- + W_par^u + W_par^B + W^p + W_s

    W±       = <|z±|^2>/4                         Alfven waves
    W_par^u  = <du_par^2>/2                       parallel flow
    W_par^B  = vA^2 <db_par^2>/2                  magnetic pressure
    W^p      = W_par^B / chi                      thermal pressure (slaved to W_par^B)
    W_s      = cs^2/(gamma^2 (gamma-1)) <s^2>/2   entropy

The left panel shows each reservoir on a log scale and the right panel its
share of the total. Reads the `w_plus`, `w_minus`, `w_par_u`, `w_par_b`, `w_p`
and `w_s` columns of `scalar_diagnostics.csv` (see
`rmhdgpu/diagnostics/compressive_channels.py`). The term-by-term budget of any
one reservoir is `vis/plot_budget.py CSV --quantity w_par_u`, and the
time-averaged contribution of each term is `vis/plot_energy_terms.py`.

    python vis/plot_energy_reservoirs.py RUN_DIRECTORY
    python vis/plot_energy_reservoirs.py RUN_DIRECTORY --show
"""

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from vis._matplotlib import finalize_figure, import_pyplot
from vis.run_quantities import RESERVOIR_COLORS as COLORS
from vis.run_quantities import RESERVOIR_LABELS as LABELS
from vis.run_quantities import RESERVOIRS, read_columns, safe_divide



def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="Run directory, or its scalar_diagnostics.csv.")
    parser.add_argument("--output", type=Path, help="Image path; default: RUN_DIRECTORY/energy_reservoirs.png.")
    parser.add_argument("--show", action="store_true", help="Show the figure after saving (e.g. in Spyder).")
    args = parser.parse_args(argv)

    run_dir = args.run_dir.expanduser().resolve()
    if run_dir.is_file():  # the CSV was given
        run_dir = run_dir.parent
    output = args.output or run_dir / "energy_reservoirs.png"

    columns = read_columns(run_dir / "scalar_diagnostics.csv")
    missing = [name for name in RESERVOIRS if name not in columns]
    if missing:
        raise SystemExit(f"{run_dir}/scalar_diagnostics.csv has no {', '.join(missing)} columns; "
                         "rerun with a version that writes the energy reservoirs.")
    time = columns["time"]
    total = sum(columns[name] for name in RESERVOIRS)

    plt = import_pyplot(show=args.show)
    fig, (ax_energy, ax_share) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    # Left: every reservoir, plus total_energy as a check that they add up.
    for name in RESERVOIRS:
        ax_energy.semilogy(time, np.maximum(columns[name], 1e-300), color=COLORS[name], label=LABELS[name])
    ax_energy.semilogy(time, total, color="k", lw=2, label="sum")
    if "total_energy" in columns:
        ax_energy.semilogy(time, columns["total_energy"], color="0.6", ls="--", label="total_energy")
    positive = total[total > 0]
    if positive.size:
        ax_energy.set_ylim(1e-8 * positive.max(), 2.0 * positive.max())
    ax_energy.set(xlabel="time", ylabel="energy", title="Energy in each reservoir")
    ax_energy.legend(ncol=2)

    # Right: share of the total.
    ax_share.stackplot(time, *[safe_divide(columns[name], total) for name in RESERVOIRS],
                       labels=[LABELS[name] for name in RESERVOIRS], colors=[COLORS[name] for name in RESERVOIRS])
    ax_share.set(xlabel="time", ylabel="fraction of total energy", ylim=(0, 1), title="Share of the total")
    ax_share.legend(loc="lower left")

    fig.suptitle(f"{run_dir.parent.name}/{run_dir.name}: energy reservoirs")
    finalize_figure(fig, output_path=output, show=args.show, plt=plt)
    return output


if __name__ == "__main__":
    main()
