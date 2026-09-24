"""Alfvenic energy, cross-helicity, and Elsasser-wave diagnostics.

Elsasser helpers use Fourier potentials from any object with phi_hat and
psi_hat attributes. They do not depend on a particular equation set.
"""

from __future__ import annotations

import math
from typing import Any

from rmhdgpu.fourier_diagnostics import modal_average, modal_inner_product_average
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


def elsasser_potential(fields: Any, *, sign: int) -> Any:
    """Return phi - psi for z+ (sign > 0), or phi + psi for z-.

    z^± = zhat cross grad_perp(phi ∓ psi). The plus branch propagates along
    +b_hat, obeying d_t f + vA d_z f = 0.
    """
    return fields.phi_hat - fields.psi_hat if sign > 0 else fields.phi_hat + fields.psi_hat


def elsasser_modal_density(fields: Any, grid: Any, backend: Any, *, sign: int) -> Any:
    """Fourier energy density for W^± = <|z^±|^2>/4."""
    potential_hat = elsasser_potential(fields, sign=sign)
    return 0.25 * grid.kperp2 * backend.xp.abs(potential_hat)**2


def elsasser_energy(fields: Any, grid: Any, backend: Any, *, sign: int) -> float:
    """Return W^±; W+ + W- equals the total Alfvenic energy."""
    return modal_average(elsasser_modal_density(fields, grid, backend, sign=sign), grid, backend)


def _elsasser_wavenumber_mean(fields, grid, backend, k_squared, *, sign):
    """Energy-weighted mean wavenumber, using the solver's real-FFT weights."""
    density = elsasser_modal_density(fields, grid, backend, sign=sign)
    energy = modal_average(density, grid, backend)
    if not energy > 0.0:
        return 0.0
    return modal_average(backend.xp.sqrt(k_squared) * density, grid, backend) / energy


def elsasser_kperp_mean(fields: Any, grid: Any, backend: Any, *, sign: int) -> float:
    """Return <k_perp> weighted by W^±; l_perp = 1/<k_perp>."""
    return _elsasser_wavenumber_mean(fields, grid, backend, grid.kperp2, sign=sign)


def elsasser_kprl_mean(fields: Any, grid: Any, backend: Any, *, sign: int) -> float:
    """Return <|k_parallel|> weighted by W^±.

    Together with <k_perp>, this gives chi_A = 2*sqrt(W)*<k_perp>/(vA*<k_parallel>).
    An empty field returns zero. The chi_A estimate cannot describe modes in
    the k_parallel=0 plane, whose Alfven frequency is zero.
    """
    return _elsasser_wavenumber_mean(fields, grid, backend, grid.kpar2, sign=sign)


def elsasser_x_alignment(fields: Any, grid: Any, backend: Any, *, sign: int) -> float:
    """Return rms(z^±_x) / rms(|z^±|), the share of z^± pointing along x.

    This is the projection factor in the slaving estimate of Squire et al.,
    arXiv:2607.08036, Eq. (47), whose background gradients point along x.
    It is not the dynamic alignment between z+ and z-.

    With z = zhat x grad_perp(f) and f = phi ∓ psi, z_x = -dy(f). Parseval
    then gives <z_x^2> = <|dy f|^2>, a sum of k_y^2 |f_hat|^2, and
    <|z|^2> = 4 W, a sum of k_perp^2 |f_hat|^2. The ratio is 1/sqrt(2) for
    energy spread isotropically in the perpendicular plane (Eq. 48). It is 1
    when z points along x (k along y) and 0 when z points along y. An empty
    field returns zero.
    """
    potential_power = backend.xp.abs(elsasser_potential(fields, sign=sign))**2
    z_squared = modal_average(grid.kperp2 * potential_power, grid, backend)
    if not z_squared > 0.0:
        return 0.0
    # |dy f_hat|^2 = k_y^2 |f_hat|^2, so the potential's power is reused.
    z_x_squared = modal_average(grid.ky**2 * potential_power, grid, backend)
    return math.sqrt(z_x_squared / z_squared)


def elsasser_energy_rhs_budget(fields, rhs_fields, grid, backend, *, sign):
    """Directional derivative d_t W = <grad(f) . grad(f_t)>/2, f = phi ∓ psi."""
    potential_hat = elsasser_potential(fields, sign=sign)
    rhs_potential_hat = elsasser_potential(rhs_fields, sign=sign)
    return 0.5 * modal_inner_product_average(grid.kperp2 * potential_hat, rhs_potential_hat, grid, backend)


def elsasser_dissipation_rhs(fields, grid, backend, linear_ops, *, sign):
    """Damping contribution to d_t W^±, allowing different omega and psi damping."""
    potential_hat = elsasser_potential(fields, sign=sign)
    phi_t_hat = -linear_ops["omega"] * fields.phi_hat
    psi_t_hat = -linear_ops["psi"] * fields.psi_hat
    rhs_potential_hat = phi_t_hat - psi_t_hat if sign > 0 else phi_t_hat + psi_t_hat
    # Use the same energy derivative as any other RHS; no expanded cross terms.
    return 0.5 * modal_inner_product_average(grid.kperp2 * potential_hat, rhs_potential_hat, grid, backend)
