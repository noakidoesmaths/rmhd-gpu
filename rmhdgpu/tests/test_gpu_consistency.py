from __future__ import annotations

import numpy as np
import pytest

from rmhdgpu.diagnostics.alfvenic import alfvenic_cross_helicity
from rmhdgpu.diagnostics.scalar import compute_energy_diagnostics, compute_scalar_diagnostics
from rmhdgpu.backend import build_backend
from rmhdgpu.config import Config
from rmhdgpu.equations import alfvenic, s09
from rmhdgpu.fft import FFTManager
from rmhdgpu.grid import build_grid
from rmhdgpu.initconds.eigenmodes_s09 import alfven_mode_state
from rmhdgpu.masks import build_dealias_mask
from rmhdgpu.operators import lap_perp
from rmhdgpu.state import State
from rmhdgpu.steppers import evolve_until
from rmhdgpu.workspace import Workspace


cupy = pytest.importorskip("cupy")
try:
    cupy.zeros((1,), dtype=cupy.float64)
except Exception as exc:  # pragma: no cover - depends on runtime availability
    pytest.skip(f"CuPy is installed but not usable in this environment: {exc}", allow_module_level=True)


def _build_context(backend_name: str, *, nx: int = 8) -> tuple[Config, object, object, FFTManager, Workspace, object]:
    config = Config(
        Nx=nx,
        Ny=nx,
        Nz=nx,
        backend=backend_name,
        dt_init=2.5e-3,
        use_variable_dt=False,
        vA=1.2,
        cs2_over_vA2=0.8,
    )
    backend = build_backend(config)
    grid = build_grid(config, backend)
    fft = FFTManager(grid, backend)
    workspace = Workspace(grid, backend)
    mask = build_dealias_mask(grid, backend)
    return config, backend, grid, fft, workspace, mask


def _build_alfvenic_context(
    backend_name: str,
    *,
    nx: int = 8,
) -> tuple[Config, object, object, FFTManager, Workspace, object]:
    config = Config(
        equation_set="alfvenic",
        Nx=nx,
        Ny=nx,
        Nz=nx,
        backend=backend_name,
        dt_init=2.5e-3,
        use_variable_dt=False,
        vA=1.2,
    )
    backend = build_backend(config)
    grid = build_grid(config, backend)
    fft = FFTManager(grid, backend)
    workspace = Workspace(grid, backend)
    mask = build_dealias_mask(grid, backend)
    return config, backend, grid, fft, workspace, mask


def _rhs_kwargs(config: Config, grid: object, fft: FFTManager, workspace: Workspace, mask: object) -> dict[str, object]:
    return {
        "grid": grid,
        "fft": fft,
        "workspace": workspace,
        "params": config,
        "dealias_mask": mask,
    }


def _run_short_evolution(
    backend_name: str,
    state_builder,
    *,
    nx: int = 8,
    steps: int = 8,
    dt: float = 2.5e-3,
    configure=None,
) -> tuple[Config, object, object, FFTManager, Workspace, State]:
    config, backend, grid, fft, workspace, mask = _build_context(backend_name, nx=nx)
    if configure is not None:
        configure(config)
    linear_ops = s09.build_dissipation_operators(grid, config)
    state0 = state_builder(config, backend, grid, fft, mask)
    final_state, _ = evolve_until(
        state0,
        steps * dt,
        s09.ideal_rhs,
        linear_ops,
        rhs_kwargs=_rhs_kwargs(config, grid, fft, workspace, mask),
        params=config,
        fixed_dt=dt,
    )
    return config, backend, grid, fft, workspace, final_state


def _single_mode_state(config: Config, backend: object, grid: object, fft: FFTManager, mask: object) -> State:
    del fft, mask
    return alfven_mode_state(
        grid=grid,
        backend=backend,
        field_names=s09.FIELD_NAMES,
        k_indices=(1, 1, 1),
        amplitude=0.25,
        branch="z_minus",
        params=config,
    )


def _deterministic_nonlinear_state(
    config: Config,
    backend: object,
    grid: object,
    fft: FFTManager,
    mask: object,
) -> State:
    del config
    xp = backend.xp
    state = State(grid, backend, field_names=s09.FIELD_NAMES)
    x = grid.x.reshape(grid.Nx, 1, 1)
    y = grid.y.reshape(1, grid.Ny, 1)
    z = grid.z.reshape(1, 1, grid.Nz)

    phi_real = (0.32 * xp.cos(x + z) + 0.18 * xp.sin(2.0 * y + z)).astype(grid.real_dtype, copy=False)
    psi_real = (0.24 * xp.cos(x + y + z) + 0.12 * xp.sin(2.0 * x + z)).astype(grid.real_dtype, copy=False)
    upar_real = (0.04 * xp.cos(x + 2.0 * z) + 0.02 * xp.sin(y + z)).astype(
        grid.real_dtype,
        copy=False,
    )
    dbpar_real = (0.03 * xp.sin(y + z) + 0.015 * xp.cos(x + y)).astype(
        grid.real_dtype,
        copy=False,
    )
    entropy_real = (0.02 * xp.sin(x + y + z)).astype(grid.real_dtype, copy=False)

    phi_hat = fft.r2c(phi_real)
    state["psi"][...] = fft.r2c(psi_real) * mask
    state["omega"][...] = s09.lap_perp(phi_hat * mask, grid)
    state["upar"][...] = fft.r2c(upar_real) * mask
    state["dbpar"][...] = fft.r2c(dbpar_real) * mask
    state["s"][...] = fft.r2c(entropy_real) * mask
    return state


def _deterministic_alfvenic_state(
    backend: object,
    grid: object,
    fft: FFTManager,
    mask: object,
) -> State:
    xp = backend.xp
    state = State(grid, backend, field_names=alfvenic.FIELD_NAMES)
    x = grid.x.reshape(grid.Nx, 1, 1)
    y = grid.y.reshape(1, grid.Ny, 1)
    z = grid.z.reshape(1, 1, grid.Nz)

    phi_real = (0.32 * xp.cos(x + z) + 0.18 * xp.sin(2.0 * y + z)).astype(grid.real_dtype, copy=False)
    psi_real = (0.24 * xp.cos(x + y + z) + 0.12 * xp.sin(2.0 * x + z)).astype(grid.real_dtype, copy=False)

    phi_hat = fft.r2c(phi_real)
    state["psi"][...] = fft.r2c(psi_real) * mask
    state["omega"][...] = lap_perp(phi_hat * mask, grid)
    return state


def _compare_state_fields(a_backend: object, a_state: State, b_backend: object, b_state: State, *, atol: float, rtol: float) -> None:
    for name in a_state.field_names:
        np.testing.assert_allclose(
            a_backend.to_numpy(a_state[name]),
            b_backend.to_numpy(b_state[name]),
            atol=atol,
            rtol=rtol,
            err_msg=f"NumPy/CuPy mismatch in field {name}.",
        )


def test_single_mode_linear_alfven_numpy_vs_cupy() -> None:
    _, backend_np, _, _, _, final_np = _run_short_evolution("numpy", _single_mode_state)
    _, backend_cp, _, _, _, final_cp = _run_short_evolution("cupy", _single_mode_state)

    _compare_state_fields(backend_np, final_np, backend_cp, final_cp, atol=5.0e-11, rtol=5.0e-11)


def test_single_mode_decay_numpy_vs_cupy() -> None:
    def configure(config: Config) -> None:
        config.dissipation["psi"]["nu_perp"] = 8.0e-3
        config.dissipation["omega"]["nu_perp"] = 8.0e-3
        config.dissipation["psi"]["n_perp"] = 2
        config.dissipation["omega"]["n_perp"] = 2

    _, backend_np, _, _, _, final_np = _run_short_evolution(
        "numpy",
        _single_mode_state,
        configure=configure,
    )
    _, backend_cp, _, _, _, final_cp = _run_short_evolution(
        "cupy",
        _single_mode_state,
        configure=configure,
    )

    _compare_state_fields(backend_np, final_np, backend_cp, final_cp, atol=5.0e-11, rtol=5.0e-11)


def test_short_unforced_nonlinear_run_numpy_vs_cupy() -> None:
    config_np, backend_np, grid_np, fft_np, workspace_np, final_np = _run_short_evolution(
        "numpy",
        _deterministic_nonlinear_state,
        nx=10,
        steps=6,
    )
    config_cp, backend_cp, grid_cp, fft_cp, workspace_cp, final_cp = _run_short_evolution(
        "cupy",
        _deterministic_nonlinear_state,
        nx=10,
        steps=6,
    )

    diagnostics_np = compute_scalar_diagnostics(final_np, grid_np, fft_np, backend_np, workspace=workspace_np)
    diagnostics_cp = compute_scalar_diagnostics(final_cp, grid_cp, fft_cp, backend_cp, workspace=workspace_cp)
    diagnostics_np.update(
        compute_energy_diagnostics(final_np, grid_np, fft_np, backend_np, config_np, workspace=workspace_np)
    )
    diagnostics_cp.update(
        compute_energy_diagnostics(final_cp, grid_cp, fft_cp, backend_cp, config_cp, workspace=workspace_cp)
    )
    diagnostics_np["alfvenic_cross_helicity"] = alfvenic_cross_helicity(final_np, grid_np, fft_np)
    diagnostics_cp["alfvenic_cross_helicity"] = alfvenic_cross_helicity(final_cp, grid_cp, fft_cp)

    for key, value_np in diagnostics_np.items():
        np.testing.assert_allclose(
            diagnostics_cp[key],
            value_np,
            atol=1.0e-10,
            rtol=2.0e-7,
            err_msg=f"Diagnostic mismatch for {key}.",
        )


def test_alfvenic_rhs_numpy_vs_cupy() -> None:
    config_np, backend_np, grid_np, fft_np, workspace_np, mask_np = _build_alfvenic_context("numpy", nx=10)
    config_cp, backend_cp, grid_cp, fft_cp, workspace_cp, mask_cp = _build_alfvenic_context("cupy", nx=10)

    state_np = _deterministic_alfvenic_state(backend_np, grid_np, fft_np, mask_np)
    state_cp = _deterministic_alfvenic_state(backend_cp, grid_cp, fft_cp, mask_cp)

    rhs_np = alfvenic.ideal_rhs(
        state_np,
        grid_np,
        fft_np,
        workspace_np,
        config_np,
        dealias_mask=mask_np,
    )
    rhs_cp = alfvenic.ideal_rhs(
        state_cp,
        grid_cp,
        fft_cp,
        workspace_cp,
        config_cp,
        dealias_mask=mask_cp,
    )

    _compare_state_fields(backend_np, rhs_np, backend_cp, rhs_cp, atol=1.0e-10, rtol=2.0e-7)
