"""Space-time maps of the perpendicular-average profiles <.>_perp(z, t).

In a periodic box the parallel transport terms of

    d_t W+(z) = -vA d_z W+(z) + (g/2) <drho z+_x>_perp
    d_t W-(z) = +vA d_z W-(z) + (g/2) <drho z-_x>_perp

average to zero, so only the z-profiles show them: energy in z+ moves up
along z = z0 + vA t and energy in z- moves down along z = z0 - vA t. Each panel
is one quantity from `z_profiles.csv` (written with the spectra, see
`rmhdgpu/diagnostics/z_profiles.py`) as a colour map in (t, z). The panels for
w_plus and w_minus get dashed lines at their propagation speed, wrapped
around the periodic box.

    python vis/plot_z_profiles.py RUN_DIRECTORY
    python vis/plot_z_profiles.py RUN_DIRECTORY --quantities w_plus w_minus slow_flux --show

`--subtract-mean` removes each time's z-average, which leaves only the
z-structure that the transport terms move around.
"""

import argparse
import csv
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from vis._matplotlib import finalize_figure, import_pyplot

DEFAULT_QUANTITIES = ("w_plus", "w_minus", "w_plus_buoyancy", "w_par_u", "w_par_b", "slow_flux")
# Signed quantities get a symmetric colour map.
SIGNED = {"w_plus_buoyancy", "w_minus_buoyancy", "slow_flux",
          "w_par_u_drive_KB", "w_par_bp_drive_KBKp", "w_s_drive_Ks"}
SPEED_SIGN = {"w_plus": +1.0, "w_minus": -1.0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="Run directory, or its z_profiles.csv.")
    parser.add_argument("--quantities", nargs="+", default=list(DEFAULT_QUANTITIES), metavar="NAME",
                        help="Quantities from z_profiles.csv to plot.")
    parser.add_argument("--vA", type=float, default=None,
                        help="Alfven speed for the propagation lines; default: the vA column of scalar_diagnostics.csv.")
    parser.add_argument("--subtract-mean", action="store_true", help="Plot each profile minus its z-average.")
    parser.add_argument("--output", type=Path, help="Image path; default: RUN_DIRECTORY/z_profiles.png.")
    parser.add_argument("--show", action="store_true", help="Show the figure after saving (e.g. in Spyder).")
    args = parser.parse_args(argv)

    run_dir = args.run_dir.expanduser().resolve()
    if run_dir.is_file():  # the CSV was given
        run_dir = run_dir.parent
    output = args.output or run_dir / "z_profiles.png"

    csv_path = run_dir / "z_profiles.csv"
    if not csv_path.is_file():
        raise SystemExit(f"{csv_path} not found; it is written every t_out_prof (default: t_out_spec) when that is > 0.")
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    times = np.unique([float(row["time"]) for row in rows])
    zs = np.unique([float(row["z"]) for row in rows])
    data = {}  # quantity -> array (time, z)
    for row in rows:
        values = data.setdefault(row["quantity"], np.full((times.size, zs.size), np.nan))
        values[np.searchsorted(times, float(row["time"])), np.searchsorted(zs, float(row["z"]))] = float(row["value"])
    unknown = [name for name in args.quantities if name not in data]
    if unknown:
        raise SystemExit(f"{csv_path} has no {', '.join(unknown)}; it has {', '.join(sorted(data))}.")

    vA = args.vA
    scalar_path = run_dir / "scalar_diagnostics.csv"
    if vA is None and scalar_path.is_file():
        with scalar_path.open("r", encoding="utf-8", newline="") as handle:
            first = next(csv.DictReader(handle), {})
        vA = float(first["vA"]) if first.get("vA") else None
    Lz = zs[1] - zs[0] + zs[-1] if zs.size > 1 else 1.0  # box length: the grid is z = 0, dz, ..., Lz - dz

    plt = import_pyplot(show=args.show)
    ncols = min(3, len(args.quantities))
    nrows = -(-len(args.quantities) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.2 * nrows), constrained_layout=True, squeeze=False)

    for ax, name in zip(axes.flat, args.quantities):
        values = data[name]
        if args.subtract_mean:
            values = values - values.mean(axis=1, keepdims=True)
        if name in SIGNED or args.subtract_mean:
            limit = np.nanmax(np.abs(values)) or 1.0
            mesh = ax.pcolormesh(times, zs, values.T, shading="nearest", cmap="RdBu_r", vmin=-limit, vmax=limit)
        else:
            mesh = ax.pcolormesh(times, zs, values.T, shading="nearest", cmap="viridis")
        fig.colorbar(mesh, ax=ax)

        # Propagation lines z = z0 ± vA t, wrapped into [0, Lz).
        if name in SPEED_SIGN and vA is not None and times.size > 1:
            for z0 in np.linspace(0.0, Lz, 4, endpoint=False):
                z_line = np.mod(z0 + SPEED_SIGN[name] * vA * (times - times[0]), Lz)
                z_line[np.abs(np.diff(z_line, prepend=z_line[0])) > 0.5 * Lz] = np.nan  # break at the wrap
                ax.plot(times, z_line, "w--", lw=0.8)
        ax.set(xlabel="time", ylabel="z", title=name + (" (minus z-mean)" if args.subtract_mean else ""),
               ylim=(zs[0], zs[-1]))
    for ax in list(axes.flat)[len(args.quantities):]:
        ax.axis("off")

    fig.suptitle(f"{run_dir.parent.name}/{run_dir.name}: perpendicular averages against z and t")
    finalize_figure(fig, output_path=output, show=args.show, plt=plt)
    return output


if __name__ == "__main__":
    main()
