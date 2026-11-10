"""Where the drho energy sits in the (kx, ky) plane, at one snapshot.

Two maps of the drho (entropy) energy per Fourier mode: the k_par = 0 plane,
and all k_par != 0 modes summed over k_par. Each map adds up to the matching
`scalar_diagnostics.csv` columns: `E_kz0_ky0_drho + E_kz0_kyn0_drho` for the
first, `E_kzn0_drho` for the second.

Reads one `fullfields/fullfield_*.h5` snapshot and `input_copy.input`.

    python vis/plot_kpar0_maps.py RUN_DIRECTORY
    python vis/plot_kpar0_maps.py RUN_DIRECTORY --index 10 --show
"""

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from vis._matplotlib import finalize_figure, import_pyplot
from vis._snapshots import energy_densities, load_grid

MODES_SHOWN = 8    # the maps show mode numbers -8 ... 8 in kx and ky
DECADES_SHOWN = 8  # the colour scale spans this many decades below the largest mode


def kxky_energy_maps(run_dir, index=-1):
    """Time, kx, ky and the two drho energy maps (k_par = 0, k_par != 0) of one snapshot.

    The maps have shape (Nx, Ny) with kx and ky in increasing order.
    """
    from rmhdgpu.fourier_diagnostics import r2c_modal_weights

    snapshots = sorted((Path(run_dir) / "fullfields").glob("fullfield_*.h5"))
    if not snapshots:
        raise SystemExit(f"No fullfield_*.h5 snapshots in {Path(run_dir) / 'fullfields'}.")
    config, backend, grid, fft = load_grid(run_dir)
    time, densities = energy_densities(snapshots[index], config, backend, grid, fft)

    # Energy of each stored mode. The solver keeps only k_par >= 0 (rfftn), so
    # the weights count every k_par > 0 mode twice; dividing by N^2 turns the
    # unnormalised FFT sum into a volume average.
    energy = r2c_modal_weights(grid, backend) * np.real(densities["drho"]) / np.prod(grid.real_shape) ** 2

    # k_par = 0 is index 0 of the last axis. fftshift puts kx and ky in increasing order.
    kpar0_map = np.fft.fftshift(energy[:, :, 0])
    waves_map = np.fft.fftshift(energy[:, :, 1:].sum(axis=2))
    kx = np.fft.fftshift(np.asarray(grid.kx).reshape(-1))
    ky = np.fft.fftshift(np.asarray(grid.ky).reshape(-1))
    return time, kx, ky, kpar0_map, waves_map


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="Run directory with fullfields/ and input_copy.input.")
    parser.add_argument("--index", type=int, default=-1, help="Which snapshot, counted from 0 (default: the last).")
    parser.add_argument("--output", type=Path, help="Image path; default: RUN_DIRECTORY/kpar0_maps.png.")
    parser.add_argument("--show", action="store_true", help="Show the figure after saving (e.g. in Spyder).")
    args = parser.parse_args(argv)

    run_dir = args.run_dir.expanduser().resolve()
    output = args.output or run_dir / "kpar0_maps.png"
    time, kx, ky, kpar0_map, waves_map = kxky_energy_maps(run_dir, args.index)

    plt = import_pyplot(show=args.show)
    from matplotlib.colors import LogNorm

    # One log colour scale for both maps, so they can be compared directly.
    largest = max(kpar0_map.max(), waves_map.max()) or 1.0
    smallest = largest * 10.0 ** -DECADES_SHOWN
    norm = LogNorm(vmin=smallest, vmax=largest)
    kx_limit = (MODES_SHOWN + 0.5) * (kx[1] - kx[0])
    ky_limit = (MODES_SHOWN + 0.5) * (ky[1] - ky[0])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)
    titles = (r"$k_\parallel = 0$", r"$k_\parallel \neq 0$ (summed over $k_\parallel$)")
    for ax, energy_map, title in zip(axes, (kpar0_map, waves_map), titles):
        # pcolormesh wants (ky, kx) ordering, hence the transpose.
        mesh = ax.pcolormesh(kx, ky, np.clip(energy_map.T, smallest, None), norm=norm, cmap="magma", shading="nearest")
        ax.set(xlabel=r"$k_x$", ylabel=r"$k_y$", aspect="equal", title=title,
               xlim=(-kx_limit, kx_limit), ylim=(-ky_limit, ky_limit))
    fig.colorbar(mesh, ax=axes.tolist(), label=r"$\delta\rho$ entropy energy per mode")

    fig.suptitle(f"{run_dir.parent.name}/{run_dir.name}: $\\delta\\rho$ energy in the $(k_x, k_y)$ plane, t = {time:g}")
    finalize_figure(fig, output_path=output, show=args.show, plt=plt)
    return output


if __name__ == "__main__":
    main()
