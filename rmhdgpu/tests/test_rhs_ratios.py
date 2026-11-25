"""Tests for the rms RHS-term sizes and chi ratios (`rmhdgpu.diagnostics.rhs_ratios`)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from rmhdgpu.backend import build_backend
from rmhdgpu.config import Config
from rmhdgpu.diagnostics import rhs_ratios
from rmhdgpu.equations import rmhd_by_nokia_rho, rmhd_by_nokia_s
from rmhdgpu.fft import FFTManager
from rmhdgpu.fourier_diagnostics import modal_average
from rmhdgpu.grid import build_grid
from rmhdgpu.masks import build_dealias_mask
from rmhdgpu.operators import dy, dz, lap_perp
from rmhdgpu.state import State
from rmhdgpu.workspace import Workspace


EQUATION_MODULES = {
    "inhomogeneous_rmhd_rho": rmhd_by_nokia_rho,
    "inhomogeneous_rmhd_s": rmhd_by_nokia_s,
}
PHYSICS = {"vA": 1.3, "cs2_over_vA2": 0.4, "g": 0.35, "K_p0": 0.9, "K_rho0": 2.1}


def _context(equation_set: str, backend_name: str = "numpy", **overrides: Any):
    if backend_name == "cupy":
        cupy = pytest.importorskip("cupy")
        try:
            cupy.zeros(1)
        except Exception as exc:  # pragma: no cover - depends on the machine
            pytest.skip(f"GPU unavailable: {exc}")
    config = Config(
        equation_set=equation_set, Nx=12, Ny=10, Nz=8, backend=backend_name, **{**PHYSICS, **overrides}
    )
    backend = build_backend(config)
    grid = build_grid(config, backend)
    return config, backend, grid, FFTManager(grid, backend), Workspace(grid, backend), build_dealias_mask(grid, backend)


def _random_state(module: Any, backend: Any, grid: Any, fft: FFTManager, mask: Any) -> State:
    state = State(grid, backend, field_names=module.FIELD_NAMES)
    rng = np.random.default_rng(20261124)
    for index, name in enumerate(module.FIELD_NAMES):
        real = rng.standard_normal(grid.real_shape) * (0.3 + 0.1 * index)
        state[name][...] = fft.r2c(backend.xp.asarray(real.astype(grid.real_dtype, copy=False))) * mask
    state["omega"][...] = lap_perp(state["omega"], grid)
    return state


def _ratios(module, state, config, backend, grid, fft, workspace, mask) -> dict[str, float]:
    return rhs_ratios.rhs_ratio_diagnostics(
        module.channel_fields(state, grid, config), grid, fft, backend, workspace,
        module.derived_parameters(config), dealias_mask=mask,
    )


def _rms(f_hat: Any, grid: Any, backend: Any) -> float:
    return math.sqrt(modal_average(backend.xp.abs(f_hat) ** 2, grid, backend))


@pytest.mark.parametrize("equation_set", sorted(EQUATION_MODULES))
def test_nonlinear_terms_are_the_solver_rhs(equation_set):
    """Subtracting the linear terms from `ideal_rhs` leaves exactly the measured nonlinear terms."""

    module = EQUATION_MODULES[equation_set]
    config, backend, grid, fft, workspace, mask = _context(equation_set)
    state = _random_state(module, backend, grid, fft, mask)
    p = module.derived_parameters(config)
    fields = module.channel_fields(state, grid, config)
    out = _ratios(module, state, config, backend, grid, fft, workspace, mask)

    rhs = module.ideal_rhs(state, grid, fft, workspace, config, dealias_mask=mask)
    # Vorticity of each Elsasser potential: lap(zeta±) = omega ∓ lap(psi).
    for branch, sign in (("plus", -1.0), ("minus", +1.0)):
        zeta_hat = fields.phi_hat + sign * fields.psi_hat
        rate_hat = rhs["omega"] + sign * lap_perp(rhs["psi"], grid)
        # zeta+ obeys d_t + vA d_z, zeta- obeys d_t - vA d_z.
        linear_hat = sign * p.vA * dz(lap_perp(zeta_hat, grid), grid) - p.g * dy(fields.drho_hat, grid)
        nl_hat = rate_hat - linear_hat
        expected = math.sqrt(modal_average(grid.inv_kperp2 * np.abs(nl_hat) ** 2, grid, backend))
        assert out[f"rms_nl_z_{branch}"] == pytest.approx(expected, rel=1e-10)
        assert out[f"rms_nl_z_{branch}"] > 0.0

    # Entropy: s_t = K_s dy(phi) - {phi, s}. The rho set evolves drho, so build s_t from it.
    if equation_set == "inhomogeneous_rmhd_s":
        s_rate_hat = rhs["s"]
    else:
        s_rate_hat = module.derive_s_hat(rhs["drho"], rhs["db_par"], config)
    drive_hat = p.K_s * dy(fields.phi_hat, grid)
    assert out["rms_drive_s"] == pytest.approx(_rms(drive_hat, grid, backend), rel=1e-12)
    assert out["rms_nl_s"] == pytest.approx(_rms(s_rate_hat - drive_hat, grid, backend), rel=1e-10)
    assert out["chi_s_drive"] == pytest.approx(out["rms_nl_s"] / out["rms_drive_s"], rel=1e-12)


def test_z_minus_nonlinearity_is_projected_real_space_advection():
    """||N-||/k_perp equals the rms of P[(z+ . grad) z-] built directly in real space."""

    module = rmhd_by_nokia_s
    config, backend, grid, fft, workspace, mask = _context("inhomogeneous_rmhd_s")
    state = _random_state(module, backend, grid, fft, mask)
    fields = module.channel_fields(state, grid, config)
    out = _ratios(module, state, config, backend, grid, fft, workspace, mask)

    ikx, iky = 1j * grid.kx, 1j * grid.ky

    def vector(zeta_hat):  # z = zhat x grad(zeta) = (-dy zeta, dx zeta)
        return -iky * zeta_hat, ikx * zeta_hat

    for branch, advected, advector in (
        ("minus", fields.phi_hat + fields.psi_hat, fields.phi_hat - fields.psi_hat),
        ("plus", fields.phi_hat - fields.psi_hat, fields.phi_hat + fields.psi_hat),
    ):
        ax, ay = (fft.c2r(c) for c in vector(advector))
        advection_hat = []
        for component_hat in vector(advected):
            d_x = fft.c2r(ikx * component_hat)
            d_y = fft.c2r(iky * component_hat)
            advection_hat.append(fft.r2c(ax * d_x + ay * d_y) * mask)
        fx, fy = advection_hat
        divergence_part = (grid.kx * fx + grid.ky * fy) * grid.inv_kperp2
        px, py = fx - grid.kx * divergence_part, fy - grid.ky * divergence_part
        expected = math.sqrt(modal_average(np.abs(px) ** 2 + np.abs(py) ** 2, grid, backend))
        assert out[f"rms_nl_z_{branch}"] == pytest.approx(expected, rel=1e-10)


def test_pure_z_plus_state_is_not_distorted():
    """With z- = 0 nothing distorts z+, so chi_rhs_plus = 0 although z+ is large."""

    module = rmhd_by_nokia_s
    config, backend, grid, fft, workspace, mask = _context("inhomogeneous_rmhd_s")
    state = _random_state(module, backend, grid, fft, mask)
    # z- = 0 means phi = -psi, i.e. omega = -lap(psi).
    state["omega"][...] = -lap_perp(state["psi"], grid)
    out = _ratios(module, state, config, backend, grid, fft, workspace, mask)

    assert out["rms_lin_z_plus"] > 0.0
    assert out["rms_nl_z_plus"] < 1e-10 * out["rms_lin_z_plus"]
    assert out["chi_rhs_plus"] < 1e-10
    # z- is zero up to round-off, and so is the term that would distort it.
    assert out["rms_lin_z_minus"] < 1e-10 * out["rms_lin_z_plus"]
    assert out["rms_nl_z_minus"] < 1e-10 * out["rms_lin_z_plus"]


@pytest.mark.parametrize("kz", [1, 2, 3])
def test_single_mode_linear_term_is_the_alfven_frequency(kz):
    """For one mode, ||vA d_z z-|| / ||z-|| = vA |k_z|; there is no nonlinearity."""

    module = rmhd_by_nokia_s
    config, backend, grid, fft, workspace, mask = _context("inhomogeneous_rmhd_s")
    nx, ny, nz = grid.real_shape
    y = np.arange(ny).reshape(1, -1, 1) / ny
    z = np.arange(nz).reshape(1, 1, -1) / nz
    zeta = 0.4 * np.cos(2.0 * np.pi * (2 * y + kz * z)) * np.ones(grid.real_shape)
    zeta_hat = fft.r2c(zeta.astype(grid.real_dtype, copy=False))
    state = State(grid, backend, field_names=module.FIELD_NAMES)
    # z- only: phi = psi = zeta/2, so phi + psi = zeta and phi - psi = 0.
    state["psi"][...] = 0.5 * zeta_hat
    state["omega"][...] = lap_perp(0.5 * zeta_hat, grid)
    out = _ratios(module, state, config, backend, grid, fft, workspace, mask)

    z_rms = math.sqrt(modal_average(grid.kperp2 * np.abs(zeta_hat) ** 2, grid, backend))
    assert out["rms_lin_z_minus"] / z_rms == pytest.approx(PHYSICS["vA"] * kz, rel=1e-12)
    assert out["rms_nl_z_minus"] == pytest.approx(0.0, abs=1e-12)
    assert out["chi_rhs_minus"] == pytest.approx(0.0, abs=1e-12)


def test_kzn0_variant_drops_only_the_kz0_part():
    module = rmhd_by_nokia_s
    config, backend, grid, fft, workspace, mask = _context("inhomogeneous_rmhd_s")
    state = _random_state(module, backend, grid, fft, mask)
    out = _ratios(module, state, config, backend, grid, fft, workspace, mask)
    for branch in ("plus", "minus"):
        assert 0.0 < out[f"rms_nl_z_{branch}_kzn0"] < out[f"rms_nl_z_{branch}"]
        assert out[f"chi_rhs_{branch}_kzn0"] < out[f"chi_rhs_{branch}"]


def test_zero_entropy_gradient_gives_nan_drive_ratio():
    """K_s = K_p0 - gamma K_rho0 = 0: there is no drive, so chi_s_drive is undefined."""

    module = rmhd_by_nokia_s
    config, backend, grid, fft, workspace, mask = _context(
        "inhomogeneous_rmhd_s", K_p0=5.0 / 3.0 * PHYSICS["K_rho0"]
    )
    state = _random_state(module, backend, grid, fft, mask)
    out = _ratios(module, state, config, backend, grid, fft, workspace, mask)
    assert out["rms_drive_s"] == pytest.approx(0.0, abs=1e-12)
    assert math.isnan(out["chi_s_drive"])
    assert math.isfinite(out["chi_s_prl"])


@pytest.mark.parametrize("equation_set", sorted(EQUATION_MODULES))
def test_columns_are_written_and_documented(equation_set):
    module = EQUATION_MODULES[equation_set]
    config, backend, grid, fft, workspace, mask = _context(equation_set)
    state = _random_state(module, backend, grid, fft, mask)
    diagnostics = module.compute_equation_scalar_diagnostics(
        state, grid=grid, fft=fft, backend=backend, params=config, workspace=workspace, dealias_mask=mask,
    )
    columns = list(rhs_ratios.RHS_RATIO_SCALAR_DIAGNOSTIC_INFO)
    assert list(_ratios(module, state, config, backend, grid, fft, workspace, mask)) == columns
    assert set(columns) <= set(diagnostics)
    assert set(columns) <= set(module.SCALAR_DIAGNOSTIC_INFO)


@pytest.mark.parametrize("backend_name", ["scipy_cpu", "cupy"])
def test_backends_agree_with_numpy(backend_name):
    if backend_name == "scipy_cpu":
        pytest.importorskip("scipy.fft")
    module = rmhd_by_nokia_rho
    results = []
    for name in ("numpy", backend_name):
        config, backend, grid, fft, workspace, mask = _context("inhomogeneous_rmhd_rho", name)
        # Same real-space random fields on both backends.
        state = State(grid, backend, field_names=module.FIELD_NAMES)
        rng = np.random.default_rng(7)
        for index, field in enumerate(module.FIELD_NAMES):
            real = rng.standard_normal(grid.real_shape) * (0.3 + 0.1 * index)
            state[field][...] = fft.r2c(backend.xp.asarray(real.astype(grid.real_dtype, copy=False))) * mask
        results.append(_ratios(module, state, config, backend, grid, fft, workspace, mask))
    for column, value in results[0].items():
        assert results[1][column] == pytest.approx(value, rel=1e-9), column
