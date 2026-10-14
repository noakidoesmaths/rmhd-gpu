"""Plot the turbulence strength chi_A against time for one or more runs.

    chi_A = z+_rms k_perp_plus / (vA k_prl_plus)

chi_A is the ratio of the nonlinear rate to the Alfven rate: chi_A << 1 is weak
turbulence, chi_A ~ 1 is critical balance. The formula, z+_rms = 2 sqrt(W+) and
the old-column-name handling all come from vis/run_quantities.py, so this plot
agrees with every other plot that uses chi_A.

Only scalar_diagnostics.csv is read. It needs the columns w_plus, k_perp_plus,
k_prl_plus and vA, which the inhomogeneous (nokia) equation sets save. A plain
homogeneous run does not save them, and the script says so.

Examples (also usable as main([...]) in Spyder):
    python vis/plot_chi_a.py examples/outputs_nokia_rmhd_52_rho
    python vis/plot_chi_a.py examples/outputs_nokia_rmhd_5*_rho --show
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vis._matplotlib import finalize_figure, import_pyplot
from vis.run_quantities import elsasser_rms, read_columns, resolve_run_dirs, turbulence_strength

NEEDED_COLUMNS = ["time", "w_plus", "k_perp_plus", "k_prl_plus", "vA"]


def load_chi_a(run_dir):
    """Saved times and chi_A of z+ for one run directory."""
    columns = read_columns(Path(run_dir) / "scalar_diagnostics.csv")
    missing = [name for name in NEEDED_COLUMNS if name not in columns]
    if missing:
        raise ValueError(f"{run_dir} is missing columns {missing}; chi_A cannot be computed for this run.")

    z_plus = elsasser_rms(columns["w_plus"])
    chi_a = turbulence_strength(z_plus, columns["k_perp_plus"], columns["k_prl_plus"], columns["vA"])
    return columns["time"], chi_a


def plot_chi_a(run_dirs, output_path, show=False, log=False):
    plt = import_pyplot(show=show)
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for run_dir in run_dirs:
        times, chi_a = load_chi_a(run_dir)
        ax.plot(times, chi_a, label=Path(run_dir).name)

    ax.axhline(1.0, color="0.5", ls=":", lw=1.0, label=r"$\chi_A = 1$ (critical balance)")
    if log:
        ax.set_yscale("log")
    ax.set_xlabel("time")
    ax.set_ylabel(r"$\chi_A = z^+_{\rm rms}\langle k_\perp\rangle / (v_A \langle k_\parallel\rangle)$")
    ax.set_title(r"Turbulence strength $\chi_A$")
    ax.legend(fontsize=9)
    fig.tight_layout()
    finalize_figure(fig, output_path=output_path, show=show, plt=plt)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+", help="Run output directories (quoted wildcards are fine).")
    parser.add_argument("--output", type=Path,
                        help="Image path; default: chi_a.png in the first run directory.")
    parser.add_argument("--log", action="store_true", help="Use a log y axis.")
    parser.add_argument("--show", action="store_true", help="Show the figure after saving (e.g. in Spyder).")
    return parser


def main(argv=None) -> Path:
    args = build_parser().parse_args(argv)
    try:
        run_dirs = resolve_run_dirs(args.run_dirs)
        output_path = args.output.expanduser().resolve() if args.output else run_dirs[0] / "chi_a.png"
        plot_chi_a(run_dirs, output_path, show=args.show, log=args.log)
    except ValueError as error:
        raise SystemExit(str(error)) from None
    return output_path


if __name__ == "__main__":
    main()
