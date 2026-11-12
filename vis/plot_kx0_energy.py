"""Energy at kx = 0 compared with energy at kx != 0, against time.

kx = 0 means no variation along x, the direction of the background
gradients. Every Fourier mode with kx = 0 is counted, whatever its ky and
k_par.

Left panel: the share of the total energy at kx = 0 and at kx != 0.
Right panel: the share of each field's own energy that sits at kx = 0.

Reads the `E_kx0_*` and `E_kxn0_*` columns of `scalar_diagnostics.csv` (see
`rmhdgpu/diagnostics/kz0_sectors.py`). Runs made before those columns existed
are summed from their `fullfields/` snapshots instead, which is slower and
only has one point per snapshot; `--every` then skips snapshots.

    python vis/plot_kx0_energy.py RUN_DIRECTORY
    python vis/plot_kx0_energy.py RUN_DIRECTORY --exclude-fields z_plus --show
"""

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vis._matplotlib import finalize_figure, import_pyplot
from vis._snapshots import sector_columns_from_snapshots
from vis.run_quantities import read_columns, safe_divide

FIELDS = ("z_plus", "z_minus", "du_par", "db_par", "drho")
LABELS = {
    "z_plus": r"$z^+$",
    "z_minus": r"$z^-$",
    "du_par": r"$\delta u_\parallel$",
    "db_par": r"$\delta B_\parallel$",
    "drho": r"$\delta\rho$ (entropy)",
}
COLORS = {"z_plus": "tab:red", "z_minus": "tab:cyan", "du_par": "tab:green", "db_par": "tab:orange", "drho": "tab:blue"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="Run directory, or its scalar_diagnostics.csv.")
    parser.add_argument("--exclude-fields", nargs="+", choices=FIELDS, default=[], metavar="FIELD",
                        help="Leave these fields out of the total in the left panel, e.g. the forced z_plus.")
    parser.add_argument("--every", type=int, default=1,
                        help="For runs without the CSV columns: use every Nth snapshot (default: all).")
    parser.add_argument("--output", type=Path, help="Image path; default: RUN_DIRECTORY/kx0_energy.png.")
    parser.add_argument("--show", action="store_true", help="Show the figure after saving (e.g. in Spyder).")
    args = parser.parse_args(argv)

    run_dir = args.run_dir.expanduser().resolve()
    if run_dir.is_file():  # the CSV was given
        run_dir = run_dir.parent
    output = args.output or run_dir / "kx0_energy.png"

    # Read the kx = 0 and kx != 0 energies: from the CSV if it has them, else from the snapshots.
    csv_path = run_dir / "scalar_diagnostics.csv"
    columns = read_columns(csv_path) if csv_path.is_file() else {}
    if "E_kx0_drho" not in columns:
        if not any((run_dir / "fullfields").glob("fullfield_*.h5")):
            raise SystemExit(f"{run_dir} has no E_kx0_* columns in scalar_diagnostics.csv and no "
                             "fullfield snapshots to sum instead; nothing to plot.")
        columns = sector_columns_from_snapshots(run_dir, every=max(1, args.every))
    time = columns["time"]

    # Total energy at kx = 0 and at kx != 0. z+ and z- together are the Alfvenic
    # energy (u_perp + b_perp), so the five fields add up to total_energy.
    kept = [field for field in FIELDS if field not in args.exclude_fields]
    if not kept:
        raise SystemExit("--exclude-fields removed every field; nothing left to total.")
    kx0 = sum(columns[f"E_kx0_{field}"] for field in kept)
    kxn0 = sum(columns[f"E_kxn0_{field}"] for field in kept)
    total = kx0 + kxn0

    plt = import_pyplot(show=args.show)
    fig, (ax_total, ax_share) = plt.subplots(1, 2, figsize=(12.5, 5), constrained_layout=True)

    # Left: share of the total energy at kx != 0 and at kx = 0.
    ax_total.stackplot(
        time, safe_divide(kxn0, total), safe_divide(kx0, total),
        labels=[r"$k_x \neq 0$", r"$k_x = 0$"],
        colors=["0.65", "tab:blue"],
    )
    excluded = ", ".join(LABELS[field] for field in args.exclude_fields)
    ax_total.set(xlabel="time", ylabel="fraction of total energy", ylim=(0, 1),
                 title="Where the total energy sits" + (f" (without {excluded})" if excluded else ""))
    ax_total.legend(loc="lower left")

    # Right: share of each field's own energy at kx = 0.
    for field in FIELDS:
        field_kx0 = columns[f"E_kx0_{field}"]
        field_kxn0 = columns[f"E_kxn0_{field}"]
        ax_share.plot(time, safe_divide(field_kx0, field_kx0 + field_kxn0), color=COLORS[field], label=LABELS[field])
    ax_share.set(xlabel="time", ylabel="fraction of the field's energy", ylim=(0, 1.02),
                 title=r"Share of each field at $k_x = 0$")
    ax_share.legend()

    fig.suptitle(f"{run_dir.parent.name}/{run_dir.name}: energy at $k_x = 0$ and $k_x \\neq 0$ (all $k_y$, $k_\\parallel$)")
    finalize_figure(fig, output_path=output, show=args.show, plt=plt)
    return output


if __name__ == "__main__":
    main()
