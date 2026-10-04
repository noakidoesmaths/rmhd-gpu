from __future__ import annotations

import numpy as np

from rmhdgpu.backend import build_backend
from rmhdgpu.config import Config
from rmhdgpu.equations import s09
from rmhdgpu.fft import FFTManager
from rmhdgpu.grid import build_grid
from rmhdgpu.initconds.eigenmodes_s09 import alfven_mode_state
from rmhdgpu.masks import build_dealias_mask
from rmhdgpu.state import State
from rmhdgpu.steppers import evolve_until, if_ssprk3_step
from rmhdgpu.utils import check_state_finite
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


def _zero_linear_ops(state: State) -> dict[str, object]:
    return {
        name: state.backend.zeros(state.grid.fourier_shape, dtype=state.grid.real_dtype)
        for name in state.field_names
    }


def test_forcing_only_zero_initial_conditions_become_nonzero() -> None:
    config = Config(
        Nx=8,
        Ny=8,
        Nz=8,
        backend="numpy",
        use_forcing=True,
        forcing_seed=101,
        use_variable_dt=False,
        dt_init=0.05,
    )
    config.force_amplitudes["psi"] = 0.4
    backend, grid, fft, _, _ = _build_context(config)
    state = State(grid, backend, field_names=config.field_names)

    final_state, _ = evolve_until(
        state,
        0.2,
        _zero_rhs,
        _zero_linear_ops(state),
        rhs_kwargs={"grid": grid, "fft": fft, "workspace": None, "params": config, "dealias_mask": None},
        params=config,
        fixed_dt=0.05,
    )

    psi_norm = float(np.linalg.norm(backend.to_numpy(final_state["psi"])))
    assert psi_norm > 0.0, "Forcing-only zero initial data should become nonzero in a forced field."


def test_unforced_run_with_zero_initial_conditions_stays_zero() -> None:
    config = Config(Nx=8, Ny=8, Nz=8, backend="numpy", use_forcing=False, use_variable_dt=False, dt_init=0.05)
    backend, grid, fft, _, _ = _build_context(config)
    state = State(grid, backend, field_names=config.field_names)

    final_state, _ = evolve_until(
        state,
        0.2,
        _zero_rhs,
        _zero_linear_ops(state),
        rhs_kwargs={"grid": grid, "fft": fft, "workspace": None, "params": config, "dealias_mask": None},
        params=config,
        fixed_dt=0.05,
    )

    for field_name in state.field_names:
        np.testing.assert_allclose(
            backend.to_numpy(final_state[field_name]),
            0.0,
            atol=0.0,
            rtol=0.0,
            err_msg=f"Unforced zero initial condition should stay zero in {field_name}.",
        )


def test_forced_run_stays_finite_for_small_amplitude() -> None:
    config = Config(
        Nx=8,
        Ny=8,
        Nz=8,
        backend="numpy",
        use_forcing=True,
        forcing_seed=202,
        cfl_number=0.25,
        dt_max=1.0e-2,
        tmax=0.05,
    )
    config.force_amplitudes["psi"] = 0.03
    config.force_amplitudes["omega"] = 0.03
    for name in config.field_names:
        config.dissipation[name]["nu_perp"] = 1.0e-3
        config.dissipation[name]["nu_par"] = 2.0e-3
        config.dissipation[name]["n_perp"] = 2
        config.dissipation[name]["n_par"] = 1

    backend, grid, fft, workspace, mask = _build_context(config)
    state = State(grid, backend, field_names=config.field_names)
    linear_ops = s09.build_dissipation_operators(grid, config)
    rhs_kwargs = {
        "grid": grid,
        "fft": fft,
        "workspace": workspace,
        "params": config,
        "dealias_mask": mask,
    }

    final_state, info = evolve_until(
        state,
        config.tmax,
        s09.ideal_rhs,
        linear_ops,
        rhs_kwargs=rhs_kwargs,
        params=config,
    )

    assert info["steps"] > 0
    check_state_finite(final_state, backend, time=float(info["t"]), step=int(info["steps"]), context="forced run")


def test_forcing_disabled_preserves_previous_behavior() -> None:
    config = Config(
        Nx=8,
        Ny=8,
        Nz=8,
        backend="numpy",
        use_forcing=False,
        use_variable_dt=False,
        dt_init=0.01,
    )
    config.force_amplitudes["psi"] = 1.0
    config.force_amplitudes["omega"] = 1.0

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

    evolved_reference = state0
    for _ in range(5):
        evolved_reference = if_ssprk3_step(
            evolved_reference,
            config.dt_init,
            s09.ideal_rhs,
            linear_ops,
            rhs_kwargs=rhs_kwargs,
        )

    evolved_test, _ = evolve_until(
        state0,
        5 * config.dt_init,
        s09.ideal_rhs,
        linear_ops,
        rhs_kwargs=rhs_kwargs,
        params=config,
        fixed_dt=config.dt_init,
    )

    for field_name in state0.field_names:
        np.testing.assert_allclose(
            backend.to_numpy(evolved_test[field_name]),
            backend.to_numpy(evolved_reference[field_name]),
            atol=1.0e-13,
            rtol=1.0e-13,
            err_msg=f"Disabling forcing should preserve deterministic evolution for {field_name}.",
        )
