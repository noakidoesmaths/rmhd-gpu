"""Plot the plane-averaged density <rho>_yz(x) from full-field HDF5 snapshots, one curve per time.

The background density rho_0 is a prescribed input, not something the solver measures or
evolves. It is nearly uniform, with a small constant log-gradient along x,
K_rho0 = d ln(rho_0)/dx, so grad(rho_0) ~ epsilon. The solver evolves only the relative
fluctuation drho = delta rho / rho_0, measured against that initial background (the mean is
NOT subtracted, so <drho>_yz is free to evolve; only the box mean is conserved at 0). The
full density is rho = rho_0 * (1 + drho). To first order in epsilon, with rho_c the
background density at the box centre x_c,

    rho(x, y, z) / rho_c = 1 + K_rho0 * (x - x_c) + drho(x, y, z)

(the cross term K*(x - x_c)*drho is O(epsilon^2), beyond what the equations resolve). Both
terms are the same order. This script plots, against x:

    reference:  1 + K_rho0 * (x - x_c)           (drho = 0, no fluctuation)
    curves:     1 + K_rho0 * (x - x_c) + <drho>_yz   (reference + measured mean profile)

<drho>_yz is the average over y and z, i.e. the ky = kz = 0 part of drho; it is the only
measured piece. With --fluctuation-only the tilt is dropped and <drho>_yz is plotted on its
own against zero, which is easier to read when K_rho0 * Lx is large. The linear background
needs |K_rho0| * Lx / 2 << 1; the script prints a note when that fails or when
max |drho| >= 1 (the equations are linearised in drho).

K_rho0 is not stored in the snapshots; it is read from input_copy.input in the run
directory (the parent of fullfields/) unless --k-rho0 is given. Runs need t_out_full > 0
to write snapshots.

Examples (also usable as main([...]) in Spyder):
    python vis/plot_mean_density.py RUN_DIRECTORY/fullfields
    python vis/plot_mean_density.py RUN_DIRECTORY/fullfields --k-rho0 0.05 --show
    python vis/plot_mean_density.py RUN_DIRECTORY/fullfields --every 5 --fluctuation-only
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from vis._matplotlib import finalize_figure, import_pyplot

try:
    import h5py
except ImportError:  # pragma: no cover - exercised by runtime error path
    h5py = None


def read_k_rho0(snapshot_dir: Path) -> float | None:
    """Look for `K_rho0 = value` in input_copy.input next to or above the snapshots."""
    for folder in (snapshot_dir, snapshot_dir.parent):
        candidate = folder / "input_copy.input"
        if candidate.exists():
            match = re.search(r"^\s*K_rho0\s*=\s*([^\s#]+)", candidate.read_text(), re.MULTILINE)
            if match:
                return float(match.group(1))
    return None


def box_length(x: np.ndarray) -> float:
    """Length of the periodic box; the grid has no endpoint, so it is Nx * dx."""
    return float(x.size * (x[1] - x[0]))


def box_centre(x: np.ndarray) -> float:
    """Middle of the box [x[0], x[0] + Lx). The centre is a convention: rho = rho_c there."""
    return float(x[0] + 0.5 * box_length(x))


def mean_density_profile(x: np.ndarray, drho: np.ndarray, k_rho0: float):
    """Return (background, mean, drho_mean) on the x grid; drho has shape (Nx, Ny, Nz).

    background = 1 + K_rho0 * (x - x_c)       rho_0 / rho_c, the prescribed background
    drho_mean  = <drho>_yz                    the measured mean profile
    mean       = background + drho_mean       <rho>_yz / rho_c, first order in epsilon
    """
    background = 1.0 + k_rho0 * (x - box_centre(x))
    drho_mean = drho.mean(axis=(1, 2))
    return background, background + drho_mean, drho_mean


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path, help="Snapshot directory (fullfield_*.h5) or one .h5 file.")
    parser.add_argument("--k-rho0", type=float, default=None, help="Background log-gradient (default: read from input_copy.input).")
    parser.add_argument("--every", type=int, default=1, help="Plot every Nth snapshot.")
    parser.add_argument("--fluctuation-only", action="store_true",
                        help="Plot <drho>_yz alone against zero instead of <rho>_yz / rho_c.")
    parser.add_argument("--output", type=Path, default=None, help="Output image (default: mean_density.png next to the snapshots).")
    parser.add_argument("--show", action="store_true", help="Show the figure interactively.")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if h5py is None:
        raise SystemExit("plot_mean_density.py requires `h5py` to read full-field HDF5 snapshots.")

    if args.path.is_dir():
        files = sorted(args.path.glob("fullfield_*.h5"))
        snapshot_dir = args.path
    else:
        files = [args.path]
        snapshot_dir = args.path.parent
    if not files:
        raise SystemExit(f"No full-field snapshot files were found in {args.path}.")

    k_rho0 = args.k_rho0 if args.k_rho0 is not None else read_k_rho0(snapshot_dir)
    if k_rho0 is None:
        raise SystemExit("Could not find K_rho0 in input_copy.input; pass --k-rho0.")

    files = files[:: max(args.every, 1)]
    plt = import_pyplot(show=args.show)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = plt.cm.viridis(np.linspace(0.0, 1.0, len(files)))

    max_abs_drho = 0.0
    for file, color in zip(files, colors):
        with h5py.File(file, "r") as handle:
            if "drho" not in handle["output"]:
                raise SystemExit(f"{file} has no `drho` field (needs an inhomogeneous rho equation set).")
            x = np.asarray(handle["metadata/x"])
            drho = np.asarray(handle["output/drho"])
            time = float(handle["output/time"][()])
        background, mean, drho_mean = mean_density_profile(x, drho, k_rho0)
        max_abs_drho = max(max_abs_drho, float(np.max(np.abs(drho))))
        ax.plot(x, drho_mean if args.fluctuation_only else mean, color=color, label=f"t = {time:.3g}")

    if args.fluctuation_only:
        ax.plot(x, np.zeros_like(x), "k--", lw=1.5, label=r"no fluctuation ($\delta\rho=0$)")
        ax.set_ylabel(r"$\langle\delta\rho/\rho_0\rangle_{yz}$")
    else:
        ax.plot(x, background, "k--", lw=1.5, label=r"background ($\delta\rho=0$)")
        ax.set_ylabel(r"$\langle\rho\rangle_{yz}\,/\,\rho_c$")
    ax.set_xlabel("x")
    ax.set_title(rf"Mean density, $K_{{\rho 0}}={k_rho0:g}$ (max $|\delta\rho/\rho_0|$ = {max_abs_drho:.2g})")
    if len(files) <= 12:
        ax.legend(fontsize=8)
    fig.tight_layout()

    half_range = abs(k_rho0) * box_length(x) / 2.0
    if half_range >= 1.0:
        print(f"Note: |K_rho0| * Lx / 2 = {half_range:.3g} >= 1, so the linear background 1 + K_rho0 (x - x_c) "
              "is not small (it reaches zero or below); the run is outside the small-gradient ordering.")
    if max_abs_drho >= 1.0:
        print(f"Note: max |drho| = {max_abs_drho:.3g} >= 1; the equations are linearised in drho, "
              "so the fluctuation is not small.")

    output = args.output if args.output is not None else snapshot_dir / "mean_density.png"
    finalize_figure(fig, output_path=output, show=args.show, plt=plt)


if __name__ == "__main__":
    main()
