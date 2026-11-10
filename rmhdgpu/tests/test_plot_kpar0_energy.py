"""The k_par = 0 plots: vis/plot_kpar0_energy.py and vis/plot_kpar0_maps.py."""
import csv

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from rmhdgpu.diagnostics.kz0_sectors import kz0_sector_energies
from rmhdgpu.equations import get_equation_module, rmhd_by_nokia_rho
from rmhdgpu.initconds import build_initial_state
from rmhdgpu.masks import build_dealias_mask
from vis._snapshots import load_grid
from vis.plot_kpar0_energy import FIELDS, main, sector_columns_from_snapshots
from vis.plot_kpar0_maps import kxky_energy_maps
from vis.plot_kpar0_maps import main as maps_main

INPUT = """
[equations]
type = "inhomogeneous_rmhd_rho"
[grid]
Nx = 8
Ny = 8
Nz = 8
[physics]
vA = 1.0
cs2_over_vA2 = 0.1
K_rho0 = 1.5
g = 0.06
K_p0 = 1.0
"""

SECTOR_COLUMNS = [f"{prefix}_{field}" for field in FIELDS for prefix in ("E_kzn0", "E_kz0_kyn0", "E_kz0_ky0")]


def write_run_with_one_snapshot(run_dir, seed):
    """A tiny run directory: input, one snapshot of a random state plus a mean density profile."""
    (run_dir / "input_copy.input").write_text(INPUT, encoding="utf-8")
    config, backend, grid, fft = load_grid(run_dir)
    state = build_initial_state("random_spectrum",
                                parameters={"seed": seed, "init_energy": 0.5, "exclude_kpar0": False},
                                grid=grid, backend=backend, fft=fft,
                                dealias_mask=build_dealias_mask(grid, backend),
                                field_names=config.field_names, params=config)
    # A pure (y, z)-averaged density profile: it must land in the k_par = 0, ky = 0 sector.
    profile = 0.3 * np.cos(2.0 * grid.x.reshape(-1, 1, 1)) * np.ones(grid.real_shape)
    state["drho"][...] += fft.r2c(profile)

    (run_dir / "fullfields").mkdir()
    with h5py.File(run_dir / "fullfields" / "fullfield_0001.h5", "w") as handle:
        out = handle.create_group("output")
        out["time"] = 1.25
        for name in state.field_names:
            out[name] = fft.c2r(state[name])
    return config, backend, grid, state


def write_sector_csv(run_dir, n_rows):
    with (run_dir / "scalar_diagnostics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time", "step", *SECTOR_COLUMNS])
        for i in range(n_rows):
            writer.writerow([0.5 * i, i, *[0.1 + 0.01 * i * j for j in range(len(SECTOR_COLUMNS))]])


def test_snapshot_columns_match_the_solver_diagnostic(tmp_path):
    """Summing a snapshot gives the columns the solver writes, and they add up to total_energy."""
    config, backend, grid, state = write_run_with_one_snapshot(tmp_path, seed=3)

    columns = sector_columns_from_snapshots(tmp_path)

    assert columns["time"][0] == pytest.approx(1.25)
    expected = kz0_sector_energies(rmhd_by_nokia_rho._energy_modal_densities(state, grid, backend, config),
                                   rmhd_by_nokia_rho.KZ0_SECTOR_FIELDS, grid, backend)
    for name in SECTOR_COLUMNS:
        assert columns[name][0] == pytest.approx(expected[name], rel=1e-10, abs=1e-15)
    total_energy = get_equation_module(config.equation_set).total_energy(state, grid, backend, config)
    assert sum(columns[name][0] for name in SECTOR_COLUMNS) == pytest.approx(total_energy, rel=1e-12)
    assert columns["E_kz0_ky0_drho"][0] > 0.0


def test_maps_add_up_to_the_sector_columns(tmp_path):
    """The two (kx, ky) maps are the same energy as the drho sector columns."""
    write_run_with_one_snapshot(tmp_path, seed=5)
    columns = sector_columns_from_snapshots(tmp_path)

    time, kx, ky, kpar0_map, waves_map = kxky_energy_maps(tmp_path)

    assert time == pytest.approx(1.25)
    assert np.all(np.diff(kx) > 0) and np.all(np.diff(ky) > 0)
    kpar0_energy = columns["E_kz0_ky0_drho"][0] + columns["E_kz0_kyn0_drho"][0]
    assert kpar0_map.sum() == pytest.approx(kpar0_energy, rel=1e-10)
    assert waves_map.sum() == pytest.approx(columns["E_kzn0_drho"][0], rel=1e-10)
    # The ky = 0 line of the k_par = 0 map is the mean-profile sector.
    assert kpar0_map[:, ky == 0].sum() == pytest.approx(columns["E_kz0_ky0_drho"][0], rel=1e-10)

    path = maps_main([str(tmp_path)])
    assert path == tmp_path / "kpar0_maps.png"
    assert path.is_file() and path.stat().st_size > 0


def test_plots_from_the_csv_alone(tmp_path):
    """With the E_kz* columns in the CSV, no snapshots or input file are needed."""
    write_sector_csv(tmp_path, n_rows=4)
    path = main([str(tmp_path / "scalar_diagnostics.csv")])
    assert path == tmp_path / "kpar0_energy.png"
    assert path.is_file() and path.stat().st_size > 0


def test_old_run_without_columns_plots_from_snapshots(tmp_path):
    write_run_with_one_snapshot(tmp_path, seed=7)
    (tmp_path / "scalar_diagnostics.csv").write_text("time,step,total_energy\n0.0,0,1.0\n", encoding="utf-8")
    assert main([str(tmp_path)]).is_file()


def test_old_run_without_columns_or_snapshots_is_reported(tmp_path):
    (tmp_path / "scalar_diagnostics.csv").write_text("time,step,total_energy\n0.0,0,1.0\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="no E_kz0_"):
        main([str(tmp_path)])


def test_exclude_fields(tmp_path):
    write_sector_csv(tmp_path, n_rows=3)
    assert main([str(tmp_path), "--exclude-fields", "z_plus", "z_minus"]).is_file()
    with pytest.raises(SystemExit, match="every field"):
        main([str(tmp_path), "--exclude-fields", *FIELDS])
