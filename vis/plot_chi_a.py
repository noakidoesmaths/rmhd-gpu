"""Plot the turbulence strength chi_A and its three factors against time for one or more runs.

    chi_A = z+_rms k_perp_plus / (vA k_prl_plus)

chi_A is the ratio of the nonlinear rate to the Alfven rate: chi_A << 1 is weak
turbulence, chi_A ~ 1 is critical balance. The formula, z+_rms = 2 sqrt(W+) and
the old-column-name handling all come from vis/run_quantities.py, so this plot
agrees with every other plot that uses chi_A.

Only scalar_diagnostics.csv is read. It needs the columns w_plus, k_perp_plus,
k_prl_plus and vA, which the inhomogeneous (nokia) equation sets save. A plain
homogeneous run does not save them, and the script says so.

The figure is 2x2: chi_A and its three factors z+_rms, k_perp_plus and
k_prl_plus, which shows which one is still changing when chi_A drifts. All
values are in code units: speeds in units of vA, wavenumbers in units of
2 pi / L, time in units of L / (2 pi vA).

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
    """Saved times, chi_A of z+ and its three factors for one run directory."""
    columns = read_columns(Path(run_dir) / "scalar_diagnostics.csv")
    missing = [name for name in NEEDED_COLUMNS if name not in columns]
    if missing:
        raise ValueError(f"{run_dir} is missing columns {missing}; chi_A cannot be computed for this run.")

    z_plus = elsasser_rms(columns["w_plus"])
    chi_a = turbulence_strength(z_plus, columns["k_perp_plus"], columns["k_prl_plus"], columns["vA"])
    return {"time": columns["time"], "chi_a": chi_a, "z_plus": z_plus,
            "k_perp_plus": columns["k_perp_plus"], "k_prl_plus": columns["k_prl_plus"]}


# (key in load_chi_a, y label) of each panel, in reading order.
PANELS = [
    ("chi_a", r"$\chi_A = z^+_{\rm rms}\langle k_\perp\rangle / (v_A \langle k_\parallel\rangle)$"),
    ("z_plus", r"$z^+_{\rm rms} / v_A$"),
    ("k_perp_plus", r"$\langle k_\perp\rangle^+$  (units of $2\pi/L$)"),
    ("k_prl_plus", r"$\langle k_\parallel\rangle^+$  (units of $2\pi/L$)"),
]


def plot_chi_a(run_dirs, output_path, show=False, log=False):
    plt = import_pyplot(show=show)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)

    for run_dir in run_dirs:
        data = load_chi_a(run_dir)
        for ax, (key, _) in zip(axes.flat, PANELS):
            ax.plot(data["time"], data[key], label=Path(run_dir).name)

    chi_ax = axes[0, 0]
    chi_ax.axhline(1.0, color="0.5", ls=":", lw=1.0, label=r"$\chi_A = 1$ (critical balance)")
    for ax, (_, ylabel) in zip(axes.flat, PANELS):
        if log:
            ax.set_yscale("log")
        ax.set_ylabel(ylabel)
    for ax in axes[1]:
        ax.set_xlabel("time")
    chi_ax.legend(fontsize=9)
    fig.suptitle(r"Turbulence strength $\chi_A$ and its factors")
    fig.tight_layout()
    finalize_figure(fig, output_path=output_path, show=show, plt=plt)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+", help="Run output directories (quoted wildcards are fine).")
    parser.add_argument("--output", type=Path,
                        help="Image path; default: chi_a.png in the first run directory.")
    parser.add_argument("--log", action="store_true", help="Use log y axes.")
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
