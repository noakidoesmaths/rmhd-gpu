"""Plot the branchwise chi_A proxies and RMS ratios in three time panels.

The chi_A proxy uses the opposite Elsasser field's amplitude and the target
field's wavenumbers. The Elsasser RMS ratios use k_z != 0 output modes.
The entropy RMS ratio compares advection with the background-gradient drive.
All five curves are built from scalar_diagnostics.csv. Each panel has an
independent linear vertical scale, and all three share the time axis.

    python vis/plot_chi_rhs.py RUN_DIRECTORY
    python vis/plot_chi_rhs.py RUN_DIRECTORY --show
"""

import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vis._matplotlib import finalize_figure, import_pyplot
from vis.run_quantities import elsasser_rms, read_columns, turbulence_strength

ROUNDOFF = 1e-8
REQUIRED_COLUMNS = (
    "time", "vA", "w_plus", "w_minus",
    "k_perp_plus", "k_perp_minus", "k_prl_plus", "k_prl_minus",
    "rms_lin_z_plus", "rms_lin_z_minus", "chi_rhs_plus_kzn0",
    "chi_rhs_minus_kzn0", "chi_s_drive",
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="Run directory, or its scalar_diagnostics.csv.")
    parser.add_argument("--output", type=Path, help="Image path; default: RUN_DIRECTORY/chi_rhs.png.")
    parser.add_argument("--show", action="store_true", help="Show the figure after saving (e.g. in Spyder).")
    args = parser.parse_args(argv)

    run_dir = args.run_dir.expanduser().resolve()
    csv_path = run_dir if run_dir.is_file() else run_dir / "scalar_diagnostics.csv"
    run_dir = csv_path.parent
    output = args.output or run_dir / "chi_rhs.png"

    columns = read_columns(csv_path)
    missing = [name for name in REQUIRED_COLUMNS if name not in columns]
    if missing:
        raise SystemExit(f"{csv_path} is missing columns {', '.join(missing)}.")
    time = columns["time"]
    z = {branch: elsasser_rms(columns[f"w_{branch}"]) for branch in ("plus", "minus")}

    # Keep the RHS ratios blank when either propagation norm is at roundoff.
    lin_plus = columns["rms_lin_z_plus"]
    lin_minus = columns["rms_lin_z_minus"]
    both_resolved = (lin_plus > ROUNDOFF * lin_minus) & (lin_minus > ROUNDOFF * lin_plus)

    plt = import_pyplot(show=args.show)
    fig, (ax_plus, ax_minus, ax_entropy) = plt.subplots(
        3, 1, sharex=True, figsize=(9.5, 8.4), constrained_layout=True,
    )
    for ax, branch, opposite, sign, color in (
        (ax_plus, "plus", "minus", "+", "tab:red"),
        (ax_minus, "minus", "plus", "-", "tab:blue"),
    ):
        chi_a = turbulence_strength(
            z[opposite], columns[f"k_perp_{branch}"], columns[f"k_prl_{branch}"], columns["vA"],
        )
        target_resolved = z[branch] > ROUNDOFF * z[opposite]
        ax.plot(time, np.where(target_resolved, chi_a, np.nan), color=color, ls="--",
                label=rf"$\chi_A^{sign}$")
        ax.plot(time, np.where(both_resolved, columns[f"chi_rhs_{branch}_kzn0"], np.nan), color=color,
                label=rf"$\chi_{{\rm RMS}}^{sign}$ ($k_z\ne0$)")

        if not both_resolved.any():
            ax.text(0.5, 0.5, "Elsasser RHS ratios unavailable at the saved times",
                    transform=ax.transAxes, ha="center", va="center")

    ax_entropy.plot(
        time, columns["chi_s_drive"], color="tab:green",
        label=r"$\chi_{\rm RMS,s} = \|\{\phi,s\}\|_{\rm rms}/\|K_s\partial_y\phi\|_{\rm rms}$",
    )
    for ax in (ax_plus, ax_minus, ax_entropy):
        ax.set_ylim(bottom=0.0)
        if ax.get_ylim()[1] >= 1.0:
            ax.axhline(1.0, color="0.5", lw=0.8, ls=":")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False)
    ax_entropy.set_xlabel("time")
    fig.supylabel(r"$\chi$")
    fig.suptitle(f"{run_dir.parent.name}/{run_dir.name}")
    finalize_figure(fig, output_path=output, show=args.show, plt=plt)
    return output


if __name__ == "__main__":
    main()
