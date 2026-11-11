"""How much energy sits at k_par = 0, against time.

The energy of each field is split into three Fourier sectors:

    waves : k_par != 0             propagating waves
    2D    : k_par = 0, ky != 0     structures that do not propagate
    mean  : k_par = 0, ky = 0      the (y, z)-averaged profiles <f>(x)

The three panels show the share of the total energy in each sector, each
field's share in the mean profiles, and each field's share in the whole
k_par = 0 plane (2D + mean).

Reads the `E_kzn0_*`, `E_kz0_kyn0_*` and `E_kz0_ky0_*` columns of
`scalar_diagnostics.csv` (see `rmhdgpu/diagnostics/kz0_sectors.py`). Runs made
before those columns existed are summed from their `fullfields/` snapshots
instead, which is slower and only has one point per snapshot.

    python vis/plot_kpar0_energy.py RUN_DIRECTORY
    python vis/plot_kpar0_energy.py RUN_DIRECTORY --exclude-fields z_plus --show
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
                        help="Leave these fields out of the total in the first panel, e.g. the forced z_plus.")
    parser.add_argument("--output", type=Path, help="Image path; default: RUN_DIRECTORY/kpar0_energy.png.")
    parser.add_argument("--show", action="store_true", help="Show the figure after saving (e.g. in Spyder).")
    args = parser.parse_args(argv)

    run_dir = args.run_dir.expanduser().resolve()
    if run_dir.is_file():  # the CSV was given
        run_dir = run_dir.parent
    output = args.output or run_dir / "kpar0_energy.png"

    # Read the sector energies: from the CSV if it has them, else from the snapshots.
    csv_path = run_dir / "scalar_diagnostics.csv"
    columns = read_columns(csv_path) if csv_path.is_file() else {}
    source = "scalar_diagnostics.csv"
    if "E_kz0_ky0_drho" not in columns:
        if not any((run_dir / "fullfields").glob("fullfield_*.h5")):
            raise SystemExit(f"{run_dir} has no E_kz0_* columns in scalar_diagnostics.csv and no "
                             "fullfield snapshots to sum instead; nothing to plot.")
        columns = sector_columns_from_snapshots(run_dir)
        source = "snapshots"
    time = columns["time"]

    # Total energy in each sector. z+ and z- together are the Alfvenic energy
    # (u_perp + b_perp), so the five fields add up to total_energy.
    kept = [field for field in FIELDS if field not in args.exclude_fields]
    if not kept:
        raise SystemExit("--exclude-fields removed every field; nothing left to total.")
    waves = sum(columns[f"E_kzn0_{field}"] for field in kept)
    twod = sum(columns[f"E_kz0_kyn0_{field}"] for field in kept)
    mean = sum(columns[f"E_kz0_ky0_{field}"] for field in kept)
    total = waves + twod + mean

    plt = import_pyplot(show=args.show)
    fig, (ax_total, ax_mean, ax_kpar0) = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)

    # Panel 1: share of the total energy in each sector.
    ax_total.stackplot(
        time, safe_divide(waves, total), safe_divide(twod, total), safe_divide(mean, total),
        labels=[r"waves: $k_\parallel \neq 0$",
                r"2D: $k_\parallel = 0,\ k_y \neq 0$",
                r"mean profiles: $k_\parallel = 0,\ k_y = 0$"],
        colors=["0.65", "tab:purple", "tab:blue"],
    )
    excluded = ", ".join(LABELS[field] for field in args.exclude_fields)
    ax_total.set(xlabel="time", ylabel="fraction of total energy", ylim=(0, 1),
                 title="Where the total energy sits" + (f" (without {excluded})" if excluded else ""))
    ax_total.legend(loc="lower left")

    # Panels 2 and 3: share of each field's own energy in the mean profiles and at k_par = 0.
    for field in FIELDS:
        field_waves = columns[f"E_kzn0_{field}"]
        field_twod = columns[f"E_kz0_kyn0_{field}"]
        field_mean = columns[f"E_kz0_ky0_{field}"]
        field_total = field_waves + field_twod + field_mean
        ax_mean.plot(time, safe_divide(field_mean, field_total), color=COLORS[field], label=LABELS[field])
        ax_kpar0.plot(time, safe_divide(field_twod + field_mean, field_total), color=COLORS[field], label=LABELS[field])
    ax_mean.set(xlabel="time", ylabel=r"fraction of the field's energy", ylim=(0, 1.02),
                title=r"Share of each field in mean profiles ($k_\parallel = 0,\ k_y = 0$)")
    ax_kpar0.set(xlabel="time", ylabel=r"fraction of the field's energy", ylim=(0, 1.02),
                 title=r"Share of each field at $k_\parallel = 0$ (any $k_y$)")
    ax_mean.legend()

    fig.suptitle(f"{run_dir.parent.name}/{run_dir.name}: energy at $k_\\parallel = 0$")
    finalize_figure(fig, output_path=output, show=args.show, plt=plt)
    return output


if __name__ == "__main__":
    main()
