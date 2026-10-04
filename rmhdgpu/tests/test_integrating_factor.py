from __future__ import annotations

import numpy as np

from rmhdgpu.backend import build_backend
from rmhdgpu.config import Config
from rmhdgpu.equations import s09
from rmhdgpu.fft import FFTManager
from rmhdgpu.grid import build_grid
from rmhdgpu.initconds.eigenmodes_s09 import alfven_mode_state
from rmhdgpu.initconds.testing import single_mode_field
from rmhdgpu.masks import build_dealias_mask
from rmhdgpu.state import State
from rmhdgpu.steppers import if_ssprk3_step
from rmhdgpu.workspace import Workspace


def _zero_rhs(state: State, **kwargs) -> State:
    return state.zeros_like()


def _build_context(config: Config) -> tuple[object, object, FFTManager, Workspace, object]:
    backend = build_backend(config)
    grid = build_grid(config, backend)
    fft = FFTManager(grid, backend)
    workspace = Workspace(grid, backend)
    mask = build_dealias_mask(grid, backend)
    return backend, grid, fft, workspace, mask


def test_integrating_factor_matches_exact_linear_decay_multiple_steps() -> None:
    config = Config(Nx=8, Ny=8, Nz=8, backend="numpy")
    config.dissipation["omega"].update({"nu_perp": 4.0e-3, "nu_par": 1.5e-2, "n_perp": 2, "n_par": 1})
    backend, grid, _, _, _ = _build_context(config)
    linear_ops = s09.build_dissipation_operators(grid, config)

    state = State(grid, backend, field_names=config.field_names)
    mode_indices = (1, 2, 1)
    state["omega"][...] = single_mode_field(grid, backend, mode_indices, amplitude=1.0 + 0.5j)
    amplitude0 = complex(backend.to_numpy(state["omega"])[mode_indices])
    damping = float(backend.to_numpy(linear_ops["omega"])[mode_indices])

    steps = 6
    dt = 0.035
    current = state
    for _ in range(steps):
        current = if_ssprk3_step(current, dt, _zero_rhs, linear_ops)

    amplitude_final = complex(backend.to_numpy(current["omega"])[mode_indices])
    exact = amplitude0 * np.exp(-damping * steps * dt)
    np.testing.assert_allclose(amplitude_final, exact, atol=1.0e-13, rtol=1.0e-13)


def test_dt_splitting_consistency() -> None:
    config = Config(Nx=8, Ny=8, Nz=8, backend="numpy")
    for name in config.field_names:
        config.dissipation[name]["nu_perp"] = 2.0e-3
        config.dissipation[name]["nu_par"] = 1.0e-2
        config.dissipation[name]["n_perp"] = 2
        config.dissipation[name]["n_par"] = 1

    backend, grid, _, _, _ = _build_context(config)
    linear_ops = s09.build_dissipation_operators(grid, config)
    state = State(grid, backend, field_names=config.field_names)

    for offset, name in enumerate(state.field_names):
        state[name][...] = single_mode_field(
            grid,
            backend,
            (1 + (offset % 2), 1, 1),
            amplitude=1.0 + 0.2j * (offset + 1),
        )

    one_step = if_ssprk3_step(state, 0.08, _zero_rhs, linear_ops)
    two_steps = if_ssprk3_step(if_ssprk3_step(state, 0.04, _zero_rhs, linear_ops), 0.04, _zero_rhs, linear_ops)

    for name in state.field_names:
        np.testing.assert_allclose(
            backend.to_numpy(one_step[name]),
            backend.to_numpy(two_steps[name]),
            atol=1.0e-13,
            rtol=1.0e-13,
            err_msg=f"Exact integrating factors should make dt-splitting exact for pure dissipation in {name}.",
        )


def test_ideal_plus_dissipation_single_linear_mode() -> None:
    config = Config(Nx=8, Ny=8, Nz=8, backend="numpy", vA=1.2, cs2_over_vA2=1.0)
    config.dissipation["psi"].update({"nu_perp": 3.0e-3, "nu_par": 1.0e-2, "n_perp": 2, "n_par": 1})
    config.dissipation["omega"].update({"nu_perp": 3.0e-3, "nu_par": 1.0e-2, "n_perp": 2, "n_par": 1})
    backend, grid, fft, workspace, mask = _build_context(config)
    linear_ops = s09.build_dissipation_operators(grid, config)

    state0 = alfven_mode_state(
        grid=grid,
        backend=backend,
        field_names=config.field_names,
        k_indices=(1, 1, 1),
        amplitude=0.2,
        branch="z_minus",
        params=config,
    )

    rhs_kwargs = {
        "grid": grid,
        "fft": fft,
        "workspace": workspace,
        "params": config,
        "dealias_mask": mask,
    }

    dt = 5.0e-4
    steps = 100
    total_time = steps * dt
    current = state0
    for _ in range(steps):
        current = if_ssprk3_step(current, dt, s09.ideal_rhs, linear_ops, rhs_kwargs=rhs_kwargs)

    kz = backend.scalar_to_float(grid.kz[0, 0, 1])
    damping = float(backend.to_numpy(linear_ops["psi"])[1, 1, 1])
    exact_factor = np.exp((1j * config.vA * kz - damping) * total_time)

    for name in state0.field_names:
        np.testing.assert_allclose(
            backend.to_numpy(current[name]),
            backend.to_numpy(state0[name]) * exact_factor,
            atol=3.0e-8,
            rtol=3.0e-8,
            err_msg=f"Single-mode ideal+dissipative evolution mismatch in {name}.",
        )
