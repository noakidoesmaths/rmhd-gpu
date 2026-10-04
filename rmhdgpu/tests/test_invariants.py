from __future__ import annotations

import numpy as np

from rmhdgpu.backend import build_backend
from rmhdgpu.config import Config
from rmhdgpu.diagnostics.alfvenic import (
    alfvenic_cross_helicity,
    alfvenic_cross_helicity_rhs_budget,
    alfvenic_energy,
    alfvenic_energy_rhs_budget,
)
from rmhdgpu.equations import s09
from rmhdgpu.fft import FFTManager
from rmhdgpu.grid import build_grid
from rmhdgpu.initconds.eigenmodes_s09 import alfven_mode_state
from rmhdgpu.masks import build_dealias_mask
from rmhdgpu.operators import dz, lap_perp, poisson_bracket
from rmhdgpu.state import State
from rmhdgpu.steppers import ssprk3_step, state_linear_combination
from rmhdgpu.workspace import Workspace


def _build_context() -> tuple[Config, object, object, FFTManager, Workspace, object]:
    config = Config(Nx=12, Ny=12, Nz=12, backend="numpy", vA=1.0, cs2_over_vA2=1.0)
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


def _build_deterministic_nonlinear_alfvenic_state(
    backend: object,
    grid: object,
    fft: FFTManager,
    mask: object,
) -> State:
    """Return a compact multimode state with definitely nonzero `{phi, psi}`."""

    state = State(grid, backend, field_names=s09.FIELD_NAMES)

    xp = backend.xp
    x = grid.x.reshape(grid.Nx, 1, 1)
    y = grid.y.reshape(1, grid.Ny, 1)
    z = grid.z.reshape(1, 1, grid.Nz)

    phi_real = (
        0.50 * xp.cos(x + z)
        + 0.40 * xp.sin(2.0 * y + z)
    ).astype(grid.real_dtype, copy=False)
    psi_real = (
        0.60 * xp.cos(x + y + z)
        + 0.35 * xp.sin(2.0 * x + y + 2.0 * z)
    ).astype(grid.real_dtype, copy=False)

    phi_hat = fft.r2c(phi_real)
    psi_hat = fft.r2c(psi_real)
    phi_hat *= mask
    psi_hat *= mask

    state["psi"][...] = psi_hat
    state["omega"][...] = lap_perp(phi_hat, grid)
    return state


def _build_budget_test_state(
    config: Config,
    backend: object,
    grid: object,
    fft: FFTManager,
    workspace: Workspace,
    mask: object,
) -> State:
    """Return a slightly pre-evolved nonlinear state for RHS-budget tests.

    The seed multimode state is genuinely nonlinear, but for the specific
    negative control that halves `PB(phi, psi)` in the psi equation, the
    instantaneous budget happens to cancel at the initial instant. Advancing one
    correct ideal SSPRK3 step produces a richer deterministic state where the
    same broken coefficient gives a clearly nonzero budget.
    """

    state0 = _build_deterministic_nonlinear_alfvenic_state(backend, grid, fft, mask)
    return _advance(state0, 1, 5.0e-3, config, grid, fft, workspace, mask)


def _real_rms(field_hat: object, fft: FFTManager, backend: object) -> float:
    field_real = fft.c2r(field_hat)
    return backend.scalar_to_float(backend.xp.sqrt(backend.xp.mean(field_real**2)))


def _nonlinearity_metrics(
    state: State,
    config: Config,
    grid: object,
    fft: FFTManager,
    workspace: Workspace,
    mask: object,
) -> dict[str, float]:
    phi_hat = s09.derive_phi_hat(state["omega"], grid)
    pb_hat = poisson_bracket(phi_hat, state["psi"], grid, fft, workspace, mask=mask)
    linear_hat = config.vA * dz(phi_hat, grid)

    pb_rms = _real_rms(pb_hat, fft, state.backend)
    linear_rms = _real_rms(linear_hat, fft, state.backend)
    rhs_state = s09.ideal_rhs(state, grid, fft, workspace, config, dealias_mask=mask)
    rhs_psi_rms = _real_rms(rhs_state["psi"], fft, state.backend)

    return {
        "pb_phi_psi_rms": pb_rms,
        "linear_psi_rms": linear_rms,
        "rhs_psi_rms": rhs_psi_rms,
        "pb_to_linear_ratio": pb_rms / linear_rms if linear_rms > 0.0 else np.inf,
        "pb_to_rhs_ratio": pb_rms / rhs_psi_rms if rhs_psi_rms > 0.0 else np.inf,
    }


def _finite_difference_budget(
    diagnostic,
    state: State,
    direction: State,
    eps: float,
    grid: object,
    fft: FFTManager,
) -> float:
    state_plus = state_linear_combination(state, [(1.0, state), (eps, direction)])
    state_minus = state_linear_combination(state, [(1.0, state), (-eps, direction)])
    return (diagnostic(state_plus, grid, fft) - diagnostic(state_minus, grid, fft)) / (2.0 * eps)


def _broken_psi_rhs(
    state: State,
    grid: object,
    fft: FFTManager,
    workspace: Workspace,
    params: Config,
    dealias_mask: object,
) -> State:
    """Return a test-only RHS with the psi-bracket coefficient changed to 0.5."""

    rhs_state = s09.ideal_rhs(state, grid, fft, workspace, params, dealias_mask=dealias_mask)
    phi_hat = s09.derive_phi_hat(state["omega"], grid)
    pb_hat = poisson_bracket(phi_hat, state["psi"], grid, fft, workspace, mask=dealias_mask)
    rhs_state["psi"][...] += 0.5 * pb_hat
    return rhs_state


def test_deterministic_nonlinear_alfvenic_state_has_nonzero_brackets() -> None:
    config, backend, grid, fft, workspace, mask = _build_context()
    state = _build_deterministic_nonlinear_alfvenic_state(backend, grid, fft, mask)
    metrics = _nonlinearity_metrics(state, config, grid, fft, workspace, mask)

    assert metrics["pb_phi_psi_rms"] > 1.0e-3, (
        "The nonlinear test state should have a clearly nonzero Poisson bracket, "
        f"but pb_rms={metrics['pb_phi_psi_rms']:.3e}."
    )
    assert metrics["pb_to_linear_ratio"] > 0.1, (
        "The nonlinear test state is too close to a linear wave. "
        f"pb_rms={metrics['pb_phi_psi_rms']:.3e}, "
        f"linear_rms={metrics['linear_psi_rms']:.3e}, "
        f"pb/linear={metrics['pb_to_linear_ratio']:.3e}."
    )


def test_alfvenic_instantaneous_rhs_conservation_for_nonlinear_state() -> None:
    config, backend, grid, fft, workspace, mask = _build_context()
    state = _build_budget_test_state(config, backend, grid, fft, workspace, mask)
    rhs_state = s09.ideal_rhs(state, grid, fft, workspace, config, dealias_mask=mask)
    metrics = _nonlinearity_metrics(state, config, grid, fft, workspace, mask)

    d_energy = alfvenic_energy_rhs_budget(state, rhs_state, grid, fft)
    d_cross = alfvenic_cross_helicity_rhs_budget(state, rhs_state, grid, fft)

    assert abs(d_energy) < 1.0e-12, (
        "Ideal Alfvénic energy budget should vanish instantaneously for the nonlinear test state. "
        f"dE/dt={d_energy:.3e}, pb_rms={metrics['pb_phi_psi_rms']:.3e}, "
        f"pb/linear={metrics['pb_to_linear_ratio']:.3e}."
    )
    assert abs(d_cross) < 1.0e-12, (
        "Ideal Alfvénic cross-helicity budget should vanish instantaneously for the nonlinear test state. "
        f"dH/dt={d_cross:.3e}, pb_rms={metrics['pb_phi_psi_rms']:.3e}, "
        f"pb/linear={metrics['pb_to_linear_ratio']:.3e}."
    )


def test_alfvenic_budget_matches_finite_difference_directional_derivative() -> None:
    config, backend, grid, fft, workspace, mask = _build_context()
    state = _build_budget_test_state(config, backend, grid, fft, workspace, mask)
    broken_rhs = _broken_psi_rhs(state, grid, fft, workspace, config, mask)
    eps = 1.0e-7

    analytic_energy = alfvenic_energy_rhs_budget(state, broken_rhs, grid, fft)
    analytic_cross = alfvenic_cross_helicity_rhs_budget(state, broken_rhs, grid, fft)
    fd_energy = _finite_difference_budget(
        alfvenic_energy,
        state,
        broken_rhs,
        eps,
        grid,
        fft,
    )
    fd_cross = _finite_difference_budget(
        alfvenic_cross_helicity,
        state,
        broken_rhs,
        eps,
        grid,
        fft,
    )

    np.testing.assert_allclose(
        analytic_energy,
        fd_energy,
        atol=1.0e-10,
        rtol=1.0e-6,
        err_msg=(
            "Energy RHS budget should match a centered finite-difference directional derivative. "
            f"analytic={analytic_energy:.3e}, fd={fd_energy:.3e}, eps={eps:.1e}."
        ),
    )
    np.testing.assert_allclose(
        analytic_cross,
        fd_cross,
        atol=1.0e-10,
        rtol=1.0e-6,
        err_msg=(
            "Cross-helicity RHS budget should match a centered finite-difference directional derivative. "
            f"analytic={analytic_cross:.3e}, fd={fd_cross:.3e}, eps={eps:.1e}."
        ),
    )


def test_time_refinement_reduces_numerical_invariant_drift_for_nonlinear_state() -> None:
    config, backend, grid, fft, workspace, mask = _build_context()
    state0 = _build_deterministic_nonlinear_alfvenic_state(backend, grid, fft, mask)
    energy0 = alfvenic_energy(state0, grid, fft)
    cross0 = alfvenic_cross_helicity(state0, grid, fft)

    drifts_energy: list[float] = []
    drifts_cross: list[float] = []
    dts = [1.0e-2, 5.0e-3, 2.5e-3]
    final_time = 5.0e-2

    for dt in dts:
        steps = int(round(final_time / dt))
        evolved = _advance(state0, steps, dt, config, grid, fft, workspace, mask)
        drifts_energy.append(abs(alfvenic_energy(evolved, grid, fft) - energy0))
        drifts_cross.append(abs(alfvenic_cross_helicity(evolved, grid, fft) - cross0))

    assert drifts_energy[1] < 0.7 * drifts_energy[0] and drifts_energy[2] < 0.7 * drifts_energy[1], (
        "Alfvénic energy drift should decrease under timestep refinement. "
        f"dt/drift pairs={list(zip(dts, drifts_energy))}."
    )
    assert drifts_cross[1] < 0.7 * drifts_cross[0] and drifts_cross[2] < 0.7 * drifts_cross[1], (
        "Alfvénic cross-helicity drift should decrease under timestep refinement. "
        f"dt/drift pairs={list(zip(dts, drifts_cross))}."
    )


def test_broken_psi_bracket_coefficient_breaks_instantaneous_conservation() -> None:
    config, backend, grid, fft, workspace, mask = _build_context()
    state = _build_budget_test_state(config, backend, grid, fft, workspace, mask)
    correct_rhs = s09.ideal_rhs(state, grid, fft, workspace, config, dealias_mask=mask)
    broken_rhs = _broken_psi_rhs(state, grid, fft, workspace, config, mask)

    correct_energy = alfvenic_energy_rhs_budget(state, correct_rhs, grid, fft)
    correct_cross = alfvenic_cross_helicity_rhs_budget(state, correct_rhs, grid, fft)
    broken_energy = alfvenic_energy_rhs_budget(state, broken_rhs, grid, fft)
    broken_cross = alfvenic_cross_helicity_rhs_budget(state, broken_rhs, grid, fft)

    correct_scale = max(abs(correct_energy), abs(correct_cross))
    broken_scale = max(abs(broken_energy), abs(broken_cross))

    assert broken_scale > max(1.0e-4, 1.0e5 * correct_scale), (
        "The broken psi-bracket coefficient should produce a clearly nonzero instantaneous invariant budget. "
        f"correct=(dE/dt={correct_energy:.3e}, dH/dt={correct_cross:.3e}), "
        f"broken=(dE/dt={broken_energy:.3e}, dH/dt={broken_cross:.3e})."
    )


def test_alfvenic_invariants_for_single_linear_mode() -> None:
    config, backend, grid, fft, workspace, mask = _build_context()
    state0 = alfven_mode_state(
        grid=grid,
        backend=backend,
        field_names=s09.FIELD_NAMES,
        k_indices=(1, 1, 1),
        amplitude=0.5,
        branch="z_minus",
        params=config,
    )
    energy0 = alfvenic_energy(state0, grid, fft)
    cross0 = alfvenic_cross_helicity(state0, grid, fft)

    evolved = _advance(state0, 25, 5.0e-3, config, grid, fft, workspace, mask)
    energy1 = alfvenic_energy(evolved, grid, fft)
    cross1 = alfvenic_cross_helicity(evolved, grid, fft)

    # SSPRK3 is not exactly invariant-preserving for an oscillatory linear
    # mode. This check is meant to catch broken linear physics, while allowing
    # the expected small time-discretization drift at this timestep.
    np.testing.assert_allclose(energy1, energy0, atol=1.0e-9, rtol=1.0e-9)
    np.testing.assert_allclose(cross1, cross0, atol=1.0e-9, rtol=1.0e-9)
