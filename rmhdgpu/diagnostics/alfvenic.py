"""Diagnostics for the ideal Alfvenic RMHD subsystem.

The Elsasser-wave diagnostics of the inhomogeneous equation sets live in
`rmhdgpu.diagnostics.compressive_channels`.
"""

from __future__ import annotations

from typing import Any

from rmhdgpu.operators import dx, dy, inv_lap_perp


def _perp_gradients(phi_hat: Any, psi_hat: Any, grid: Any, fft: Any) -> dict[str, Any]:
    return {
        "dx_phi": fft.c2r(dx(phi_hat, grid)),
        "dy_phi": fft.c2r(dy(phi_hat, grid)),
        "dx_psi": fft.c2r(dx(psi_hat, grid)),
        "dy_psi": fft.c2r(dy(psi_hat, grid)),
    }


def _alfvenic_gradients(state: Any, grid: Any, fft: Any) -> dict[str, Any]:
    phi_hat = inv_lap_perp(state["omega"], grid)
    return _perp_gradients(phi_hat, state["psi"], grid, fft)


def _state_and_rhs_gradients(
    state: Any,
    rhs_state: Any,
    grid: Any,
    fft: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    phi_hat = inv_lap_perp(state["omega"], grid)
    phi_t_hat = inv_lap_perp(rhs_state["omega"], grid)
    gradients = _perp_gradients(phi_hat, state["psi"], grid, fft)
    gradients_t = _perp_gradients(phi_t_hat, rhs_state["psi"], grid, fft)
    return gradients, gradients_t


def _mean_float(backend: Any, value: Any) -> float:
    return backend.scalar_to_float(backend.xp.mean(value))


def alfvenic_energy_rhs_budget(state: Any, rhs_state: Any, grid: Any, fft: Any) -> float:
    """Return the instantaneous RHS budget `dE_A / dt`.

    The energy definition matches :func:`alfvenic_energy`:

    `E_A = 0.5 < |grad_perp phi|^2 + |grad_perp psi|^2 >`

    so the instantaneous directional derivative along `rhs_state` is

    `dE_A/dt = < grad_perp phi . grad_perp phi_t + grad_perp psi . grad_perp psi_t >`

    with `phi_t` derived from `rhs_omega` through `inv_lap_perp`.
    """

    backend = state.backend
    gradients, gradients_t = _state_and_rhs_gradients(state, rhs_state, grid, fft)
    energy_budget = (
        gradients["dx_phi"] * gradients_t["dx_phi"]
        + gradients["dy_phi"] * gradients_t["dy_phi"]
        + gradients["dx_psi"] * gradients_t["dx_psi"]
        + gradients["dy_psi"] * gradients_t["dy_psi"]
    )
    return _mean_float(backend, energy_budget)


def alfvenic_energy(state: Any, grid: Any, fft: Any) -> float:
    """Return the volume-averaged Alfvénic energy.

    The definition used here is

    `E_A = 0.5 < |grad_perp phi|^2 + |grad_perp psi|^2 >`

    where angle brackets denote a spatial average over the periodic box.
    """

    backend = state.backend
    grads = _alfvenic_gradients(state, grid, fft)
    energy = 0.5 * backend.xp.mean(
        grads["dx_phi"] ** 2
        + grads["dy_phi"] ** 2
        + grads["dx_psi"] ** 2
        + grads["dy_psi"] ** 2
    )
    return backend.scalar_to_float(energy)


def alfvenic_cross_helicity(state: Any, grid: Any, fft: Any) -> float:
    """Return the volume-averaged Alfvénic cross-helicity.

    The definition used here is

    `H_A = < grad_perp phi . grad_perp psi >`
    """

    backend = state.backend
    grads = _alfvenic_gradients(state, grid, fft)
    cross_helicity = backend.xp.mean(
        grads["dx_phi"] * grads["dx_psi"] + grads["dy_phi"] * grads["dy_psi"]
    )
    return backend.scalar_to_float(cross_helicity)


def alfvenic_cross_helicity_rhs_budget(
    state: Any,
    rhs_state: Any,
    grid: Any,
    fft: Any,
) -> float:
    """Return the instantaneous RHS budget `dH_A / dt`.

    The cross-helicity definition matches :func:`alfvenic_cross_helicity`:

    `H_A = < grad_perp phi . grad_perp psi >`

    so the instantaneous directional derivative along `rhs_state` is

    `dH_A/dt = < grad_perp phi_t . grad_perp psi + grad_perp phi . grad_perp psi_t >`
    """

    backend = state.backend
    gradients, gradients_t = _state_and_rhs_gradients(state, rhs_state, grid, fft)
    cross_budget = (
        gradients_t["dx_phi"] * gradients["dx_psi"]
        + gradients["dx_phi"] * gradients_t["dx_psi"]
        + gradients_t["dy_phi"] * gradients["dy_psi"]
        + gradients["dy_phi"] * gradients_t["dy_psi"]
    )
    return _mean_float(backend, cross_budget)

