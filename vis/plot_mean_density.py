"""Plot the mean density profile rho(x)/rho_ref from full-field HDF5 snapshots.

The solver evolves drho = delta rho / rho_0(x), with a fixed background
rho_0(x) = rho_ref * exp(K_rho0 * x) (K_rho0 = d ln rho_0 / dx, gradient along x).
The full density is therefore

    rho(x, y, z) / rho_ref = exp(K_rho0 * x) * (1 + drho(x, y, z)),

which is only meaningful while |drho| << 1 (the equations are linearised in drho).
This script plots, against x:

    background:  exp(K_rho0 * x)                      (drho = 0, no fluctuation)
    mean:        exp(K_rho0 * x) * (1 + <drho>_yz)    (background + mean profile change)

<drho>_yz is the average over y and z, i.e. the ky = kz = 0 part of drho.
K_rho0 is not stored in the snapshots; it is read from input_copy.input in the
run directory (the parent of fullfields/) unless --k-rho0 is given.

Examples (also usable as main([...]) in Spyder):
    python vis/plot_mean_density.py RUN_DIRECTORY/fullfields
    python vis/plot_mean_density.py RUN_DIRECTORY/fullfields --k-rho0 1.5 --show
    python vis/plot_mean_density.py RUN_DIRECTORY/fullfields --every 5
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


def mean_density_profile(x: np.ndarray, drho: np.ndarray, k_rho0: float):
    """Return (background, mean) rho/rho_ref profiles on the x grid; drho has shape (Nx, Ny, Nz)."""
    background = np.exp(k_rho0 * x)
    drho_mean = drho.mean(axis=(1, 2))
    return background, background * (1.0 + drho_mean), drho_mean


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path, help="Snapshot directory (fullfield_*.h5) or one .h5 file.")
    parser.add_argument("--k-rho0", type=float, default=None, help="Background log-gradient (default: read from input_copy.input).")
    parser.add_argument("--every", type=int, default=1, help="Plot every Nth snapshot.")
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
        background, mean, _ = mean_density_profile(x, drho, k_rho0)
        max_abs_drho = max(max_abs_drho, float(np.max(np.abs(drho))))
        ax.plot(x, mean, color=color, label=f"t = {time:.3g}")
    ax.plot(x, background, "k--", lw=1.5, label=r"background ($\delta\rho=0$)")

    ax.set_xlabel("x")
    ax.set_ylabel(r"$\langle\rho\rangle_{yz}\,/\,\rho_{\rm ref}$")
    ax.set_title(rf"Mean density, $K_{{\rho 0}}={k_rho0:g}$ (max $|\delta\rho/\rho_0|$ = {max_abs_drho:.2g})")
    if len(files) <= 12:
        ax.legend(fontsize=8)
    fig.tight_layout()

    output = args.output if args.output is not None else snapshot_dir / "mean_density.png"
    finalize_figure(fig, output_path=output, show=args.show, plt=plt)


if __name__ == "__main__":
    main()
