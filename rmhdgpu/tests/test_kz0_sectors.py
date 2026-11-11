"""Tests for the k_z = 0 / k_y = 0 sector energy columns (`rmhdgpu.diagnostics.kz0_sectors`)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from rmhdgpu.backend import build_backend
from rmhdgpu.config import Config
from rmhdgpu.diagnostics import kz0_sectors
from rmhdgpu.equations import rmhd_by_nokia_rho, rmhd_by_nokia_s
from rmhdgpu.fft import FFTManager
from rmhdgpu.fourier_diagnostics import modal_density_average
from rmhdgpu.grid import build_grid
from rmhdgpu.masks import build_dealias_mask
from rmhdgpu.operators import lap_perp
from rmhdgpu.state import State
from rmhdgpu.workspace import Workspace


EQUATION_MODULES = {
    "inhomogeneous_rmhd_rho": rmhd_by_nokia_rho,
    "inhomogeneous_rmhd_s": rmhd_by_nokia_s,
}
PHYSICS = {"vA": 1.3, "cs2_over_vA2": 0.4, "g": 0.35, "K_p0": 0.9, "K_rho0": 2.1}


def _context(equation_set: str, backend_name: str = "numpy"):
    if backend_name == "cupy":
        cupy = pytest.importorskip("cupy")
        try:
            cupy.zeros(1)
        except Exception as exc:  # pragma: no cover - depends on the machine
            pytest.skip(f"GPU unavailable: {exc}")
    config = Config(equation_set=equation_set, Nx=12, Ny=10, Nz=8, backend=backend_name, **PHYSICS)
    backend = build_backend(config)
    grid = build_grid(config, backend)
    return config, backend, grid, FFTManager(grid, backend)


def _random_state(module: Any, backend: Any, grid: Any, fft: FFTManager) -> State:
    mask = build_dealias_mask(grid, backend)
    state = State(grid, backend, field_names=module.FIELD_NAMES)
    rng = np.random.default_rng(20261109)
    for index, name in enumerate(module.FIELD_NAMES):
        real = rng.standard_normal(grid.real_shape) * (0.3 + 0.1 * index)
        # Add a strong x-only profile so the k_z = 0, k_y = 0 sector is not tiny.
        x = np.arange(grid.real_shape[0]).reshape(-1, 1, 1)
        real = real + np.cos(2.0 * np.pi * 2.0 * x / grid.real_shape[0])
        state[name][...] = fft.r2c(backend.xp.asarray(real.astype(grid.real_dtype, copy=False))) * mask
    state["omega"][...] = lap_perp(state["omega"], grid)
    return state


def _sectors(module, state, grid, backend, config) -> dict[str, float]:
    densities = module._energy_modal_densities(state, grid, backend, config)
    return kz0_sectors.kz0_sector_energies(densities, module.KZ0_SECTOR_FIELDS, grid, backend)


@pytest.mark.parametrize("equation_set", sorted(EQUATION_MODULES))
def test_sectors_add_up_to_total_energy(equation_set):
    module = EQUATION_MODULES[equation_set]
    config, backend, grid, fft = _context(equation_set)
    state = _random_state(module, backend, grid, fft)
    out = _sectors(module, state, grid, backend, config)

    total = module.total_energy(state, grid, backend, config)
    assert out["E_kz0_ky0"] + out["E_kz0_kyn0"] + out["E_kzn0"] == pytest.approx(total, rel=1e-12)
    assert min(out["E_kz0_ky0"], out["E_kz0_kyn0"], out["E_kzn0"]) > 0.0

    densities = module._energy_modal_densities(state, grid, backend, config)
    for field in module.KZ0_SECTOR_FIELDS:
        parts = out[f"E_kz0_ky0_{field}"] + out[f"E_kz0_kyn0_{field}"] + out[f"E_kzn0_{field}"]
        assert parts == pytest.approx(modal_density_average(densities[field], grid, backend), rel=1e-12)
        assert out[f"E_ky0_{field}"] + out[f"E_kyn0_{field}"] == pytest.approx(parts, rel=1e-12)
    assert out["E_ky0"] + out["E_kyn0"] == pytest.approx(total, rel=1e-12)
    assert min(out["E_ky0"], out["E_kyn0"]) > 0.0


@pytest.mark.parametrize("equation_set", sorted(EQUATION_MODULES))
def test_every_column_is_documented_and_written(equation_set):
    module = EQUATION_MODULES[equation_set]
    config, backend, grid, fft = _context(equation_set)
    state = _random_state(module, backend, grid, fft)
    diagnostics = module.compute_equation_scalar_diagnostics(
        state, grid=grid, fft=fft, backend=backend, params=config, workspace=Workspace(grid, backend),
    )
    info = kz0_sectors.kz0_sector_column_info(module.KZ0_SECTOR_FIELDS)
    assert len(info) == 5 * len(module.KZ0_SECTOR_FIELDS) + 5
    assert set(info) <= set(diagnostics)
    assert set(info) <= set(module.SCALAR_DIAGNOSTIC_INFO)


@pytest.mark.parametrize(
    ("kx", "ky", "kz", "sector"),
    [(2, 0, 0, "kz0_ky0"), (2, 1, 0, "kz0_kyn0"), (0, 3, 0, "kz0_kyn0"), (1, 0, 1, "kzn0"), (0, 2, 3, "kzn0")],
)
def test_single_mode_lands_in_its_sector(kx, ky, kz, sector):
    """A single real Fourier mode of the compressive field puts all of its energy in one sector."""

    module = rmhd_by_nokia_rho
    config, backend, grid, fft = _context("inhomogeneous_rmhd_rho")
    nx, ny, nz = grid.real_shape
    x = np.arange(nx).reshape(-1, 1, 1) / nx
    y = np.arange(ny).reshape(1, -1, 1) / ny
    z = np.arange(nz).reshape(1, 1, -1) / nz
    real = 0.7 * np.cos(2.0 * np.pi * (kx * x + ky * y + kz * z)) * np.ones(grid.real_shape)

    state = State(grid, backend, field_names=module.FIELD_NAMES)
    state["drho"][...] = fft.r2c(backend.xp.asarray(real.astype(grid.real_dtype, copy=False)))
    out = _sectors(module, state, grid, backend, config)

    total = out["E_kz0_ky0_drho"] + out["E_kz0_kyn0_drho"] + out["E_kzn0_drho"]
    assert total > 0.0
    assert out[f"E_{sector}_drho"] == pytest.approx(total, rel=1e-12)
    # The k_y split ignores k_z: the mode is at k_y = 0 or not, whatever its k_z.
    ky_column = "E_ky0_drho" if ky == 0 else "E_kyn0_drho"
    assert out[ky_column] == pytest.approx(total, rel=1e-12)


@pytest.mark.parametrize("equation_set", sorted(EQUATION_MODULES))
def test_kz0_ky0_matches_mean_profile_energy(equation_set):
    """Parseval: the k_z = 0, k_y = 0 energy is the energy of the (y, z)-averaged profiles."""

    module = EQUATION_MODULES[equation_set]
    config, backend, grid, fft = _context(equation_set)
    state = _random_state(module, backend, grid, fft)
    out = _sectors(module, state, grid, backend, config)
    p = module.derived_parameters(config)

    def profile(field_hat):
        return np.asarray(fft.c2r(field_hat)).mean(axis=(1, 2))

    du_par = profile(state["du_par"])
    db_par = profile(state["db_par"])
    assert out["E_kz0_ky0_du_par"] == pytest.approx(0.5 * np.mean(du_par**2), rel=1e-10)
    assert out["E_kz0_ky0_db_par"] == pytest.approx(0.5 * p.dbpar_energy_weight * np.mean(db_par**2), rel=1e-10)

    if equation_set == "inhomogeneous_rmhd_rho":
        entropy_hat = module.derive_s_hat(state["drho"], state["db_par"], config)
        entropy_field = "drho"
    else:
        entropy_hat = state["s"]
        entropy_field = "s"
    entropy = profile(entropy_hat)
    assert out[f"E_kz0_ky0_{entropy_field}"] == pytest.approx(
        0.5 * p.entropy_energy_weight * np.mean(entropy**2), rel=1e-10,
    )


def test_cupy_matches_numpy():
    module = rmhd_by_nokia_rho
    _config, gpu_backend, gpu_grid, gpu_fft = _context("inhomogeneous_rmhd_rho", "cupy")
    config, backend, grid, fft = _context("inhomogeneous_rmhd_rho")
    state = _random_state(module, backend, grid, fft)
    gpu_state = State(gpu_grid, gpu_backend, field_names=module.FIELD_NAMES)
    for name in module.FIELD_NAMES:
        gpu_state[name][...] = gpu_backend.xp.asarray(state[name])
    cpu = _sectors(module, state, grid, backend, config)
    gpu = _sectors(module, gpu_state, gpu_grid, gpu_backend, config)
    for name, value in cpu.items():
        assert gpu[name] == pytest.approx(value, rel=1e-10)
