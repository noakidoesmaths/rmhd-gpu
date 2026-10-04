from __future__ import annotations

import numpy as np
import pytest

from rmhdgpu.backend import build_backend
from rmhdgpu.config import Config
from rmhdgpu.equations import s09
from rmhdgpu.fft import FFTManager
from rmhdgpu.grid import build_grid
from rmhdgpu.initconds.eigenmodes_s09 import alfven_mode_state, entropy_mode_state, slow_mode_state
from rmhdgpu.masks import build_dealias_mask
from rmhdgpu.state import State
from rmhdgpu.steppers import ssprk3_step
from rmhdgpu.workspace import Workspace


def _build_linear_context() -> tuple[Config, object, object, FFTManager, Workspace, object]:
    config = Config(Nx=8, Ny=8, Nz=8, backend="numpy", vA=1.3, cs2_over_vA2=0.8)
    backend = build_backend(config)
    grid = build_grid(config, backend)
    fft = FFTManager(grid, backend)
    workspace = Workspace(grid, backend)
    mask = build_dealias_mask(grid, backend)
    return config, backend, grid, fft, workspace, mask


def _advance(
    state: State,
    steps: int,
    dt: float,
    config: Config,
    grid: object,
    fft: FFTManager,
    workspace: Workspace,
    mask: object,
) -> State:
    rhs_kwargs = {
        "grid": grid,
        "fft": fft,
        "workspace": workspace,
        "params": config,
        "dealias_mask": mask,
    }
    current = state
    for _ in range(steps):
        current = ssprk3_step(current, dt, s09.ideal_rhs, rhs_kwargs=rhs_kwargs)
    return current


def _scaled_state(state: State, factor: complex) -> State:
    out = state.copy()
    for name in out.field_names:
        out[name][...] *= factor
    return out


# z_minus is phi = +psi and evolves as exp(+i vA kz t); z_plus is phi = -psi, exp(-i vA kz t).
@pytest.mark.parametrize("branch,sign", [("z_minus", 1.0), ("z_plus", -1.0)])
def test_alfven_single_mode_matches_exact_linear_evolution(branch: str, sign: float) -> None:
    config, backend, grid, fft, workspace, mask = _build_linear_context()
    state0 = alfven_mode_state(
        grid=grid,
        backend=backend,
        field_names=s09.FIELD_NAMES,
        k_indices=(1, 2, 1),
        amplitude=0.3,
        branch=branch,
        params=config,
    )
    dt = 5.0e-3
    steps = 20
    total_time = steps * dt
    kz = backend.scalar_to_float(grid.kz[0, 0, 1])
    lambda_mode = sign * 1j * config.vA * kz

    evolved = _advance(state0, steps, dt, config, grid, fft, workspace, mask)
    exact = _scaled_state(state0, np.exp(lambda_mode * total_time))

    for name in state0.field_names:
        np.testing.assert_allclose(
            backend.to_numpy(evolved[name]),
            backend.to_numpy(exact[name]),
            atol=2.0e-9,
            rtol=2.0e-9,
            err_msg=f"Mismatch in Alfvén branch {branch} for field {name}.",
        )


@pytest.mark.parametrize("branch,sign", [("plus", 1.0), ("minus", -1.0)])
def test_slow_single_mode_matches_exact_linear_evolution(branch: str, sign: float) -> None:
    config, backend, grid, fft, workspace, mask = _build_linear_context()
    alpha = s09.derived_parameters(config).alpha
    state0 = slow_mode_state(
        grid=grid,
        backend=backend,
        field_names=s09.FIELD_NAMES,
        k_indices=(1, 1, 1),
        amplitude=0.4,
        branch=branch,
        params=config,
    )
    dt = 5.0e-3
    steps = 20
    total_time = steps * dt
    kz = backend.scalar_to_float(grid.kz[0, 0, 1])
    lambda_mode = sign * 1j * config.vA * np.sqrt(alpha) * kz

    evolved = _advance(state0, steps, dt, config, grid, fft, workspace, mask)
    exact = _scaled_state(state0, np.exp(lambda_mode * total_time))

    for name in state0.field_names:
        np.testing.assert_allclose(
            backend.to_numpy(evolved[name]),
            backend.to_numpy(exact[name]),
            atol=2.0e-9,
            rtol=2.0e-9,
            err_msg=f"Mismatch in slow branch {branch} for field {name}.",
        )


def test_entropy_mode_stationary() -> None:
    config, backend, grid, fft, workspace, mask = _build_linear_context()
    state0 = entropy_mode_state(
        grid=grid,
        backend=backend,
        field_names=s09.FIELD_NAMES,
        k_indices=(1, 1, 1),
        amplitude=0.7,
    )

    evolved = _advance(state0, 25, 1.0e-2, config, grid, fft, workspace, mask)

    for name in state0.field_names:
        np.testing.assert_allclose(
            backend.to_numpy(evolved[name]),
            backend.to_numpy(state0[name]),
            atol=1.0e-14,
            rtol=1.0e-14,
            err_msg=f"Entropy-mode stationarity failed for field {name}.",
        )
