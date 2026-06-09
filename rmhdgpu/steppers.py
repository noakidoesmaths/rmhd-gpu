"""Time integration helpers."""

from __future__ import annotations

import inspect
import math
from typing import Any, Iterable

import numpy as np

from rmhdgpu.forcing import apply_forcing_kick, generate_forcing_kick
from rmhdgpu.state import State
from rmhdgpu.utils import check_state_finite


def state_linear_combination(
    template: State,
    terms: Iterable[tuple[complex | float, State]],
    out: State | None = None,
) -> State:
    """Return a linear combination of states using `template` for metadata."""

    result = template.zeros_like() if out is None else out
    return result.linear_combination_(terms)


def _rhs_supports_out(rhs_func: Any) -> bool:
    signature = inspect.signature(rhs_func)
    return "out" in signature.parameters


def _evaluate_rhs(
    rhs_func: Any,
    state: State,
    *,
    rhs_kwargs: dict[str, Any],
    out: State | None = None,
) -> State:
    kwargs = dict(rhs_kwargs)
    if out is None or not _rhs_supports_out(rhs_func):
        result = rhs_func(state, **kwargs)
        if out is None:
            return result
        return out.copy_from(result)
    return rhs_func(state, out=out, **kwargs)


def _scratch_state(workspace: Any, key: str, template: State) -> State:
    if workspace is None:
        return template.zeros_like()
    return workspace.get_state_buffer(key, template.field_names)


def ssprk3_step(
    state: State,
    dt: float,
    rhs_func: Any,
    rhs_kwargs: dict[str, Any] | None = None,
) -> State:
    """Advance one step with the classical three-stage SSPRK3 scheme.

    The Shu-Osher form used here is

    `u1 = u^n + dt L(u^n)`

    `u2 = 3/4 u^n + 1/4 (u1 + dt L(u1))`

    `u^{n+1} = 1/3 u^n + 2/3 (u2 + dt L(u2))`

    The input state is not modified.
    """

    kwargs = {} if rhs_kwargs is None else dict(rhs_kwargs)
    workspace = kwargs.get("workspace")

    rhs0 = _evaluate_rhs(
        rhs_func,
        state,
        rhs_kwargs=kwargs,
        out=_scratch_state(workspace, "ssprk_rhs0", state),
    )
    stage1 = state_linear_combination(
        state,
        [(1.0, state), (dt, rhs0)],
        out=_scratch_state(workspace, "ssprk_stage1", state),
    )

    rhs1 = _evaluate_rhs(
        rhs_func,
        stage1,
        rhs_kwargs=kwargs,
        out=_scratch_state(workspace, "ssprk_rhs1", state),
    )
    predictor2 = state_linear_combination(
        stage1,
        [(1.0, stage1), (dt, rhs1)],
        out=_scratch_state(workspace, "ssprk_predictor2", state),
    )
    stage2 = state_linear_combination(
        state,
        [(0.75, state), (0.25, predictor2)],
        out=_scratch_state(workspace, "ssprk_stage2", state),
    )

    rhs2 = _evaluate_rhs(
        rhs_func,
        stage2,
        rhs_kwargs=kwargs,
        out=_scratch_state(workspace, "ssprk_rhs2", state),
    )
    predictor3 = state_linear_combination(
        stage2,
        [(1.0, stage2), (dt, rhs2)],
        out=_scratch_state(workspace, "ssprk_predictor3", state),
    )
    return state_linear_combination(state, [(1.0 / 3.0, state), (2.0 / 3.0, predictor3)])


def state_fieldwise_multiply(
    state: State,
    factors: dict[str, Any],
    out: State | None = None,
) -> State:
    """Return a new state with each field multiplied by its matching factor."""

    result = state.zeros_like() if out is None else out
    for name in result.field_names:
        result[name][...] = factors[name] * state[name]
    return result


def state_fieldwise_divide(
    state: State,
    factors: dict[str, Any],
    out: State | None = None,
) -> State:
    """Return a new state with each field divided by its matching factor."""

    result = state.zeros_like() if out is None else out
    for name in result.field_names:
        result[name][...] = state[name] / factors[name]
    return result


def project_out_kpar0(state: State, grid: Any) -> State:
    """Zero the `k_parallel = 0` plane of every field, in place.

    The `k_par = 0`, `k_perp != 0` subspace is marginal in the
    inhomogeneous_rmhd(_s) equations: there `psi` is frozen and `du_par` is
    driven secularly by `-vA*K_b0*dy(psi)` with no `dz` restoring, so its energy
    grows like `t^2`. A nonlinear run repopulates this plane from `k_par != 0`
    bracket interactions (`k1z + k2z = 0`), so it must be re-projected out each
    step. In `rfftn` storage the plane is exactly the `grid.kz == 0` slab.
    """

    kpar_nonzero = grid.kz != 0
    for name in state.field_names:
        state[name][...] *= kpar_nonzero
    return state


def _build_exponential_factors(
    state: State,
    linear_ops: dict[str, Any],
    dt: float,
    fraction: float,
) -> dict[str, Any]:
    xp = state.backend.xp
    return {
        name: xp.exp(-fraction * dt * linear_ops[name])
        for name in state.field_names
    }


def _print_progress(step: int, time: float, dt: float) -> None:
    """Emit a compact progress line for long-running batch jobs."""

    print(f"progress step={step} t={time:.6e} dt={dt:.6e}")


def compute_cfl_timestep(
    state: State,
    grid: Any,
    fft: Any,
    params: Any,
    dt_prev: float | None = None,
    workspace: Any | None = None,
    equation_module: Any | None = None,
) -> float:
    """Estimate a CFL-limited timestep from perpendicular and parallel speeds.

    The estimate uses

    - perpendicular advection by `u_perp = zhat x grad_perp phi`
    - field-line advection by `b_perp = zhat x grad_perp psi`
    - parallel linear propagation at speeds `vA` and `c_slow = vA * sqrt(alpha)`

    The returned timestep is `cfl_number * min(candidates)` clipped by
    `dt_min` and `dt_max` when those bounds are configured.
    """

    from rmhdgpu.equations import get_equation_module
    from rmhdgpu.operators import dx, dy

    backend = state.backend
    xp = backend.xp

    if equation_module is None:
        equation_module = get_equation_module(getattr(params, "equation_set", "s09"))

    phi_hat = equation_module.derive_phi_hat(state["omega"], grid)
    psi_hat = state["psi"]
    if workspace is None:
        ux = -fft.c2r(dy(phi_hat, grid))
        uy = fft.c2r(dx(phi_hat, grid))
        bx = -fft.c2r(dy(psi_hat, grid))
        by = fft.c2r(dx(psi_hat, grid))
    else:
        ux = workspace.real["r0"]
        uy = workspace.real["r1"]
        bx = workspace.real["r2"]
        by = workspace.real["r3"]
        c0 = workspace.complex["c0"]

        c0[...] = phi_hat
        c0 *= grid.ky
        c0 *= -1j
        fft.c2r(c0, out=ux)

        c0[...] = phi_hat
        c0 *= grid.kx
        c0 *= 1j
        fft.c2r(c0, out=uy)

        c0[...] = psi_hat
        c0 *= grid.ky
        c0 *= -1j
        fft.c2r(c0, out=bx)

        c0[...] = psi_hat
        c0 *= grid.kx
        c0 *= 1j
        fft.c2r(c0, out=by)

    def _speed_max(field: Any) -> float:
        return backend.scalar_to_float(xp.max(xp.abs(field)))

    def _safe_dt(length: float, speed: float) -> float:
        return math.inf if speed <= 0.0 else length / speed

    candidates = [
        _safe_dt(grid.dx, _speed_max(ux)),
        _safe_dt(grid.dy, _speed_max(uy)),
        _safe_dt(grid.dx, _speed_max(bx)),
        _safe_dt(grid.dy, _speed_max(by)),
    ]

    for speed in equation_module.characteristic_speeds(params):
        speed_value = abs(float(speed))
        candidates.append(math.inf if speed_value <= 0.0 else grid.dz / speed_value)

    finite_candidates = [value for value in candidates if math.isfinite(value) and value > 0.0]
    if finite_candidates:
        dt = float(getattr(params, "cfl_number")) * min(finite_candidates)
    else:
        dt = float(getattr(params, "dt_init"))

    if getattr(params, "dt_max") is not None:
        dt = min(dt, float(getattr(params, "dt_max")))
    if getattr(params, "dt_min") is not None:
        dt = max(dt, float(getattr(params, "dt_min")))

    return dt


def if_ssprk3_step(
    state: State,
    dt: float,
    ideal_rhs_func: Any,
    linear_ops: dict[str, Any],
    rhs_kwargs: dict[str, Any] | None = None,
) -> State:
    """Advance one step with an integrating-factor RK3 scheme.

    The method applies the standard third-order SSPRK / Kutta tableau to the
    transformed variable `v = exp(+D t) q`, where the physical state obeys

    `q_t = -D q + N(q)`

    with diagonal nonnegative damping operators `D_i(k)` for each field.

    Over one step of size `dt`, with local stage times `c = [0, 1, 1/2]`, the
    transformed stages are

    `k1 = N(q_n)`

    `q1 = E(dt) * (q_n + dt * k1)`

    `k2 = E(-dt) * N(q1)`

    `q2 = E(dt/2) * (q_n + dt/4 * (k1 + k2))`

    `k3 = E(-dt/2) * N(q2)`

    `q_{n+1} = E(dt) * (q_n + dt * (k1/6 + k2/6 + 2 k3/3))`

    where `E(c dt) = exp(-D * c dt)` acts fieldwise in Fourier space.

    When all `D_i = 0`, this reduces to the existing ideal RK3 step.
    """

    kwargs = {} if rhs_kwargs is None else dict(rhs_kwargs)
    workspace = kwargs.get("workspace")

    exp_full = _build_exponential_factors(state, linear_ops, dt, 1.0)
    exp_half = _build_exponential_factors(state, linear_ops, dt, 0.5)

    k1 = _evaluate_rhs(
        ideal_rhs_func,
        state,
        rhs_kwargs=kwargs,
        out=_scratch_state(workspace, "if_k1", state),
    )
    stage1_unscaled = state_linear_combination(
        state,
        [(1.0, state), (dt, k1)],
        out=_scratch_state(workspace, "if_stage1_unscaled", state),
    )
    stage1 = state_fieldwise_multiply(
        stage1_unscaled,
        exp_full,
        out=_scratch_state(workspace, "if_stage1", state),
    )

    n2 = _evaluate_rhs(
        ideal_rhs_func,
        stage1,
        rhs_kwargs=kwargs,
        out=_scratch_state(workspace, "if_n2", state),
    )
    k2 = state_fieldwise_divide(
        n2,
        exp_full,
        out=_scratch_state(workspace, "if_k2", state),
    )
    stage2_unscaled = state_linear_combination(
        state,
        [(1.0, state), (0.25 * dt, k1), (0.25 * dt, k2)],
        out=_scratch_state(workspace, "if_stage2_unscaled", state),
    )
    stage2 = state_fieldwise_multiply(
        stage2_unscaled,
        exp_half,
        out=_scratch_state(workspace, "if_stage2", state),
    )

    n3 = _evaluate_rhs(
        ideal_rhs_func,
        stage2,
        rhs_kwargs=kwargs,
        out=_scratch_state(workspace, "if_n3", state),
    )
    k3 = state_fieldwise_divide(
        n3,
        exp_half,
        out=_scratch_state(workspace, "if_k3", state),
    )
    final_unscaled = state_linear_combination(
        state,
        [(1.0, state), (dt / 6.0, k1), (dt / 6.0, k2), (2.0 * dt / 3.0, k3)],
        out=_scratch_state(workspace, "if_final_unscaled", state),
    )
    return state_fieldwise_multiply(final_unscaled, exp_full)


def evolve_until(
    state: State,
    t_final: float,
    ideal_rhs_func: Any,
    linear_ops: dict[str, Any],
    rhs_kwargs: dict[str, Any] | None = None,
    params: Any | None = None,
    fixed_dt: float | None = None,
    check_every: int | None = None,
    stepper_func: Any | None = None,
    forcing_rng: np.random.Generator | None = None,
) -> tuple[State, dict[str, float | int]]:
    """Advance until `t_final`, using variable or fixed timestep selection.

    Runtime non-finite checks are performed at the configured interval when
    `params.fail_on_nonfinite` is true.
    """

    kwargs = {} if rhs_kwargs is None else dict(rhs_kwargs)
    params_obj = params if params is not None else kwargs.get("params")
    if params_obj is None:
        raise ValueError("evolve_until requires params directly or in rhs_kwargs['params'].")

    grid = kwargs.get("grid")
    fft = kwargs.get("fft")
    if grid is None or fft is None:
        raise ValueError("evolve_until requires 'grid' and 'fft' entries in rhs_kwargs.")

    if check_every is None:
        check_every = int(getattr(params_obj, "runtime_check_every", 1))
    if stepper_func is None:
        stepper_func = if_ssprk3_step
    progress_output_every = getattr(params_obj, "progress_output_every", None)

    current = state
    forcing_rng_obj = forcing_rng
    if getattr(params_obj, "use_forcing", False) and forcing_rng_obj is None:
        forcing_rng_obj = current.backend.random_generator(getattr(params_obj, "forcing_seed", None))

    t = 0.0
    dt_prev: float | None = None
    steps = 0

    if getattr(params_obj, "fail_on_nonfinite", True):
        check_state_finite(current, current.backend, time=t, step=steps, context="time integration startup")

    project_kpar0 = bool(getattr(params_obj, "project_kpar0", False))
    if project_kpar0:
        project_out_kpar0(current, grid)

    while t < t_final - 1.0e-15:
        if fixed_dt is not None:
            dt = fixed_dt
        elif getattr(params_obj, "use_variable_dt", True):
            dt = compute_cfl_timestep(
                current,
                grid,
                fft,
                params_obj,
                dt_prev=dt_prev,
                workspace=kwargs.get("workspace"),
            )
        else:
            dt = float(getattr(params_obj, "dt_init"))

        dt = min(dt, t_final - t)
        current = stepper_func(current, dt, ideal_rhs_func, linear_ops, rhs_kwargs=kwargs)
        if getattr(params_obj, "use_forcing", False):
            forcing_kick = generate_forcing_kick(
                current,
                grid,
                fft,
                current.backend,
                params_obj,
                forcing_rng_obj,
                dt,
                workspace=kwargs.get("workspace"),
                out=None if kwargs.get("workspace") is None else kwargs["workspace"].get_state_buffer(
                    "forcing_kick",
                    current.field_names,
                ),
            )
            current = apply_forcing_kick(current, forcing_kick, inplace=True)
        if project_kpar0:
            project_out_kpar0(current, grid)
        t += dt
        dt_prev = dt
        steps += 1

        if progress_output_every is not None:
            should_report_progress = (steps % progress_output_every == 0) or (t >= t_final - 1.0e-15)
            if should_report_progress:
                _print_progress(steps, t, dt)

        if getattr(params_obj, "fail_on_nonfinite", True):
            should_check = (steps % check_every == 0) or (t >= t_final - 1.0e-15)
            if should_check:
                check_state_finite(
                    current,
                    current.backend,
                    time=t,
                    step=steps,
                    context="time integration",
                )

    info: dict[str, float | int] = {
        "t": t,
        "steps": steps,
        "dt_last": 0.0 if dt_prev is None else dt_prev,
    }
    return current, info
