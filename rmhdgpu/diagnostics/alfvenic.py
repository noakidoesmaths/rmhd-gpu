"""Diagnostics for the ideal Alfvenic RMHD subsystem.

The Elsasser-wave diagnostics of the inhomogeneous equation sets live in
`rmhdgpu.diagnostics.compressive_channels`.
"""

from __future__ import annotations

from typing import Any

from rmhdgpu.fourier_diagnostics import modal_average
from rmhdgpu.operators import inv_lap_perp
from rmhdgpu.operators import dx, dy


def alfvenic_phi_hat(
    state: Any,
    grid: Any,
    equation_module: Any | None = None,
) -> Any:
    """Return the velocity-potential transform for a state with an Alfvénic sector.

    Equation sets may evolve ``phi`` directly or evolve ``omega`` and provide
    ``derive_phi_hat(omega_hat, grid)``. The standard RMHD inverse-Laplacian
    relation is used as a fallback for existing ``omega``-based states.
    """

    if "phi" in state.field_names:
        return state["phi"]
    if "omega" not in state.field_names:
        raise ValueError(
            "Alfvenic diagnostics require a 'psi' field and either a 'phi' or "
            "'omega' field."
        )
    if equation_module is not None and hasattr(equation_module, "derive_phi_hat"):
        return equation_module.derive_phi_hat(state["omega"], grid)
    return inv_lap_perp(state["omega"], grid)


def _perp_gradients(phi_hat: Any, psi_hat: Any, grid: Any, fft: Any) -> dict[str, Any]:
    return {
        "dx_phi": fft.c2r(dx(phi_hat, grid)),
        "dy_phi": fft.c2r(dy(phi_hat, grid)),
        "dx_psi": fft.c2r(dx(psi_hat, grid)),
        "dy_psi": fft.c2r(dy(psi_hat, grid)),
    }


def _alfvenic_gradients(
    state: Any,
    grid: Any,
    fft: Any,
    equation_module: Any | None = None,
) -> dict[str, Any]:
    phi_hat = alfvenic_phi_hat(state, grid, equation_module)
    return _perp_gradients(phi_hat, state["psi"], grid, fft)


def _state_and_rhs_gradients(
    state: Any,
    rhs_state: Any,
    grid: Any,
    fft: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    phi_hat = alfvenic_phi_hat(state, grid)
    phi_t_hat = alfvenic_phi_hat(rhs_state, grid)
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


def elsasser_energies(
    state: Any,
    grid: Any,
    backend: Any | None = None,
    equation_module: Any | None = None,
) -> dict[str, float]:
    """Return volume-averaged Elsasser energies and imbalance diagnostics.

    The potential convention is fixed to

    ``zeta_plus = phi - psi`` and ``zeta_minus = phi + psi``.

    The reported energies are

    ``E_plus/minus = 0.5 <|grad_perp zeta_plus/minus|^2>``.

    Therefore the package Alfvénic energy is ``0.5 * (E_plus + E_minus)`` and
    its normalized cross-helicity is
    ``(E_minus - E_plus) / (E_plus + E_minus)`` for this potential convention.
    """

    if "psi" not in state.field_names:
        raise ValueError("Elsasser diagnostics require an evolved 'psi' field.")
    backend_obj = state.backend if backend is None else backend
    xp = backend_obj.xp
    phi_hat = alfvenic_phi_hat(state, grid, equation_module)
    zeta_plus_hat = phi_hat - state["psi"]
    zeta_minus_hat = phi_hat + state["psi"]
    energy_plus = modal_average(
        0.5 * grid.kperp2 * xp.abs(zeta_plus_hat) ** 2,
        grid,
        backend_obj,
    )
    energy_minus = modal_average(
        0.5 * grid.kperp2 * xp.abs(zeta_minus_hat) ** 2,
        grid,
        backend_obj,
    )
    energy_sum = energy_plus + energy_minus
    if energy_sum == 0.0:
        ratio = 1.0
        normalized_cross_helicity = 0.0
    else:
        ratio = float("inf") if energy_minus == 0.0 else energy_plus / energy_minus
        normalized_cross_helicity = (energy_minus - energy_plus) / energy_sum
    return {
        "elsasser_energy_plus": float(energy_plus),
        "elsasser_energy_minus": float(energy_minus),
        "elsasser_energy_ratio": float(ratio),
        "normalized_cross_helicity": float(normalized_cross_helicity),
    }
