"""Turn a saved full-field snapshot back into the solver's energy densities.

Used by the k_par = 0 plots. The energy weights are not repeated here: the
snapshot is transformed to Fourier space and handed to the equation module's
own `_energy_modal_densities`, so a plot and `total_energy` always agree.

Only the `inhomogeneous_rmhd_rho` equation set is supported.
"""

from pathlib import Path

import numpy as np


def load_grid(run_dir):
    """Config, backend, grid and FFT of a run, rebuilt on NumPy from its `input_copy.input`."""
    from rmhdgpu.backend import Backend
    from rmhdgpu.fft import FFTManager
    from rmhdgpu.grid import build_grid
    from rmhdgpu.runfile import resolve_run_settings

    input_path = Path(run_dir) / "input_copy.input"
    if not input_path.is_file():
        raise SystemExit(f"Missing {input_path}; it is needed for the grid and physics parameters.")
    config = resolve_run_settings(runfile_path=input_path).config
    if config.equation_set != "inhomogeneous_rmhd_rho":
        raise SystemExit(f"Only inhomogeneous_rmhd_rho runs are supported; got {config.equation_set!r}.")
    backend = Backend(xp=np, backend_name="numpy", is_gpu=False)
    grid = build_grid(config, backend)
    return config, backend, grid, FFTManager(grid, backend)


def energy_densities(snapshot_path, config, backend, grid, fft):
    """Time of one `fullfield_*.h5` snapshot and its modal energy density per field.

    The densities are keyed z_plus, z_minus, du_par, db_par, drho and are in
    the solver's `rfftn` layout (see `rmhdgpu.fourier_diagnostics`).
    """
    import h5py

    from rmhdgpu.equations.rmhd_by_nokia_rho import _energy_modal_densities
    from rmhdgpu.state import State

    with h5py.File(snapshot_path, "r") as handle:
        time = float(handle["output"]["time"][()])
        fields = {name: fft.r2c(np.asarray(handle["output"][name])) for name in config.field_names}
    state = State(grid, backend, field_names=list(fields), data=fields)
    return time, _energy_modal_densities(state, grid, backend, config)


def sector_columns_from_snapshots(run_dir, every=1):
    """The `E_kz*` and `E_ky*` columns of the scalar CSV, recomputed with one row per snapshot.

    For runs made before the solver wrote those columns. Uses the solver's own
    `kz0_sector_energies`, so the numbers are the ones a new run would save.
    """
    from rmhdgpu.diagnostics.kz0_sectors import kz0_sector_energies
    from rmhdgpu.equations.rmhd_by_nokia_rho import KZ0_SECTOR_FIELDS

    config, backend, grid, fft = load_grid(run_dir)
    rows = []
    for path in sorted((Path(run_dir) / "fullfields").glob("fullfield_*.h5"))[::every]:
        time, densities = energy_densities(path, config, backend, grid, fft)
        rows.append({"time": time, **kz0_sector_energies(densities, KZ0_SECTOR_FIELDS, grid, backend)})
    return {name: np.array([row[name] for row in rows]) for name in rows[0]}
