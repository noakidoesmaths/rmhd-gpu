"""Inohomogeneous RMHD equation set.

The evolved fields are `[psi, omega, upar, bpar, s]`, with `phi = inv_lap_perp(omega)`.
The stratification/curvature direction is `x`, so `u_x = -d_y phi`.

The ideal equations implemented here are

- `psi_t = vA * dz(phi) - {phi, psi}`
- `omega_t = vA * dz(lap_perp psi) - {phi, omega} + {psi, lap_perp psi} - gravity`
- '(delta s/c_v)_t =  -{Phi, s_0/c_v} -{Phi, delta s/c_v}'
- `(delta u_par)_t = vA^2 dz(delta B_par/B_0) + vA{psi, delta B_par/B_0} + vA{psi, ln(B_0)} `
- `(1+vA^2/c^2)( ((delta B_par)/B_0)_t + {Phi, delta B_par/B_0})
    = dz(delta u_par) + 1/vA{psi,delta u_par} - {Phi, ln(B_0)} + 1/gamma {phi, ln(p_0)} `

The linear growth-rate form is

`lambda^2 = N2 * k_y^2 / k_perp^2 - vA^2 * k_z^2`.

Positive `N2` gives unstable/decaying branches when the stratification term
dominates, while negative `N2` gives oscillatory stable branches.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from rmhdgpu.diagnostics.budget import flatten_conserved_quantity_budgets
from rmhdgpu.diagnostics.scalar import STANDARD_ENERGY_SCALAR_DIAGNOSTIC_INFO
from rmhdgpu.diagnostics.spectra import perpendicular_shell_spectrum
from rmhdgpu.fourier_diagnostics import modal_average, modal_inner_product_average
from rmhdgpu.operators import dy, dz, inv_lap_perp, lap_perp, poisson_bracket
from rmhdgpu.state import State


EQUATION_SET_NAME = "rmhd_gravity"
FIELD_NAMES = ["psi", "omega", "du_par", "db_par", "s"]
DEFAULT_INITIAL_CONDITION = "low_beta_stratified_mode"

# Scalar diagnostic names provided by this equation module. The standard
# `total_energy*` names are what budget plotting tools expect. The
# stratification RHS is equation-specific and follows the same
# `total_energy_rhs_<name>` convention.
SCALAR_DIAGNOSTIC_INFO = {
    **STANDARD_ENERGY_SCALAR_DIAGNOSTIC_INFO,
    "total_energy_rhs_stratification": "Signed ideal stratification contribution to d_t total_energy.",
    "alfvenic_energy": "Alfvenic part of the low-beta energy: 0.5 <|grad phi|^2 + |grad psi|^2>.",
    "a_energy": "Signed stratification-field quadratic contribution: 0.5 <a^2 / N2>.",
    "total_energy_proxy": "Signed sum of alfvenic_energy and a_energy for plotting convenience.",
}


@dataclass(frozen=True, slots=True)
class LowBetaStratifiedParameters:
    """Scalar parameters used by the low-beta stratified equations."""

    vA: float
    g: float




def _param_float(params: Any, name: str) -> float:
    if isinstance(params, Mapping):
        return float(params[name])
    return float(getattr(params, name))


def derived_parameters(params: Any) -> LowBetaStratifiedParameters:
    """Return the compact scalar parameter block for this equation set."""

    return LowBetaStratifiedParameters(
        vA=_param_float(params, "vA"),
        g=_param_float(params, "g"),
    )


def derive_phi_hat(omega_hat: Any, grid: Any) -> Any:
    """Return `phi_hat = inv_lap_perp(omega_hat)`."""

    return inv_lap_perp(omega_hat, grid)


def derive_j_hat(psi_hat: Any, grid: Any) -> Any:
    """Return `j_hat = -lap_perp(psi_hat) = +k_perp^2 psi_hat`."""

    return -lap_perp(psi_hat, grid)


def characteristic_speeds(params: Any) -> list[float]:
    """Return parallel linear speeds relevant to the CFL estimate."""

    return [derived_parameters(params).vA]


def ideal_rhs(
    state: State,
    grid: Any,
    fft: Any,
    workspace: Any,
    params: Any,
    dealias_mask: Any | None = None,
    out: State | None = None,
) -> State:
    """Return the Fourier-space ideal RHS of the low-beta stratified system."""

    p = derived_parameters(params)

    psi_hat = state["psi"]
    omega_hat = state["omega"]
    du_par_hat = state["du_par"]
    db_par_hat = state["db_par"]
    s_hat = state["s"]

    phi_hat = derive_phi_hat(omega_hat, grid)
    lap_psi_hat = lap_perp(psi_hat, grid)

    rhs_state = state.zeros_like() if out is None else out
    rhs_state.fill_zero()

    rhs_psi = rhs_state["psi"]
    rhs_psi[...] = vA * dz(phi_hat, grid)
    rhs_psi[...] -= poisson_bracket(
        phi_hat,
        psi_hat,
        grid,
        fft,
        workspace,
        mask=dealias_mask,
    )

    rhs_omega = rhs_state["omega"]
    rhs_omega[...] = vA * dz(lap_psi_hat, grid)
    rhs_omega[...] -= poisson_bracket(
        phi_hat,
        omega_hat,
        grid,
        fft,
        workspace,
        mask=dealias_mask,
    )
    rhs_omega[...] += poisson_bracket(
        psi_hat,
        lap_psi_hat,
        grid,
        fft,
        workspace,
        mask=dealias_mask,
    )
    rhs_omega[...] -= dy(a_hat, grid)

    rhs_a = rhs_state["a"]
    rhs_a[...] -= poisson_bracket(
        phi_hat,
        a_hat,
        grid,
        fft,
        workspace,
        mask=dealias_mask,
    )
    rhs_a[...] -= p.N2 * dy(phi_hat, grid)

    return rhs_state


def linear_matrix(kx: float, ky: float, kz: float, params: Any) -> np.ndarray:
    """Return the 3x3 linear matrix for one Fourier mode.

    Field order is `[psi, omega, a]`. For `k_perp = 0`, the inverse
    perpendicular Laplacian is regularized by returning the zero matrix.
    """

    p = derived_parameters(params)
    matrix = np.zeros((3, 3), dtype=np.complex128)
    kperp2 = float(kx) ** 2 + float(ky) ** 2
    if kperp2 <= 0.0:
        return matrix

    matrix[0, 1] = -1j * p.vA * float(kz) / kperp2
    matrix[1, 0] = -1j * p.vA * float(kz) * kperp2
    matrix[1, 2] = -1j * float(ky)
    matrix[2, 1] = 1j * p.N2 * float(ky) / kperp2
    return matrix


def _dissipation_spec_for_field(
    params: Any,
    field_name: str,
    dissipation_spec: Mapping[str, Mapping[str, float | int]] | None,
) -> Mapping[str, float | int]:
    if dissipation_spec is not None:
        return dissipation_spec[field_name]
    if isinstance(params, Mapping):
        return params["dissipation"][field_name]
    return getattr(params, "dissipation")[field_name]


def dissipation_operator(
    grid: Any,
    params: Any,
    field_name: str,
    dissipation_spec: Mapping[str, Mapping[str, float | int]] | None = None,
) -> Any:
    """Return the nonnegative diagonal damping operator `D_i(k)` for one field."""

    spec = _dissipation_spec_for_field(params, field_name, dissipation_spec)
    nu_perp = float(spec["nu_perp"])
    nu_par = float(spec["nu_par"])
    n_perp = int(spec["n_perp"])
    n_par = int(spec["n_par"])

    operator = 0.0
    if nu_perp > 0.0:
        operator = operator + nu_perp * (grid.kperp2**n_perp)
    if nu_par > 0.0:
        operator = operator + nu_par * (grid.kpar2**n_par)
    if isinstance(operator, float):
        operator = grid.kperp2 * 0.0
    return operator


def build_dissipation_operators(
    grid: Any,
    params: Any,
    field_names: list[str] | None = None,
    dissipation_spec: Mapping[str, Mapping[str, float | int]] | None = None,
) -> dict[str, Any]:
    """Build the diagonal damping operators for all evolved fields."""

    names = FIELD_NAMES if field_names is None else field_names
    return {
        name: dissipation_operator(grid, params, name, dissipation_spec=dissipation_spec)
        for name in names
    }


def perpendicular_energy_spectra(
    state: State,
    grid: Any,
    backend: Any,
    *,
    bin_width: float | None = None,
    params: Any | None = None,
) -> dict[str, np.ndarray]:
    """Return perpendicular shell spectra for the low-beta energy pieces."""

    xp = backend.xp
    p = derived_parameters(params)
    phi_hat = derive_phi_hat(state["omega"], grid)
    kperp2 = grid.kperp2

    kperp, u_perp = perpendicular_shell_spectrum(
        0.5 * kperp2 * (xp.abs(phi_hat) ** 2),
        grid,
        backend,
        bin_width=bin_width,
    )
    _, b_perp = perpendicular_shell_spectrum(
        0.5 * kperp2 * (xp.abs(state["psi"]) ** 2),
        grid,
        backend,
        bin_width=bin_width,
    )
    _, a_energy = perpendicular_shell_spectrum(
        0.5 * (xp.abs(state["a"]) ** 2) / p.N2,
        grid,
        backend,
        bin_width=bin_width,
    )
    return {"kperp": kperp, "u_perp": u_perp, "b_perp": b_perp, "a": a_energy}


def total_energy_modal_density(state: State, grid: Any, backend: Any, params: Any) -> Any:
    """Return the signed modal quadratic density for the low-beta total energy."""

    xp = backend.xp
    p = derived_parameters(params)
    phi_hat = derive_phi_hat(state["omega"], grid)
    return (
        0.5 * grid.kperp2 * (xp.abs(phi_hat) ** 2 + xp.abs(state["psi"]) ** 2)
        + 0.5 * (xp.abs(state["a"]) ** 2) / p.N2
    )


def total_energy(state: State, grid: Any, backend: Any, params: Any) -> float:
    """Return the signed quadratic form `0.5 <|grad phi|^2 + |grad psi|^2 + a^2 / N2>`."""

    density_hat = total_energy_modal_density(state, grid, backend, params)
    return modal_average(density_hat, grid, backend)


def alfvenic_energy(state: State, grid: Any, backend: Any) -> float:
    """Return the low-beta Alfvenic energy partition."""

    xp = backend.xp
    phi_hat = derive_phi_hat(state["omega"], grid)
    density_hat = 0.5 * grid.kperp2 * (xp.abs(phi_hat) ** 2 + xp.abs(state["psi"]) ** 2)
    return modal_average(density_hat, grid, backend)


def a_energy(state: State, grid: Any, backend: Any, params: Any) -> float:
    """Return the signed stratification-field quadratic contribution."""

    p = derived_parameters(params)
    return modal_average(0.5 * (backend.xp.abs(state["a"]) ** 2) / p.N2, grid, backend)


def total_energy_stratification_rhs(state: State, grid: Any, backend: Any, params: Any) -> float:
    """Return the signed ideal stratification contribution `2 <u_x a>`."""

    phi_hat = derive_phi_hat(state["omega"], grid)
    ux_hat = -dy(phi_hat, grid)
    return 2.0 * modal_inner_product_average(ux_hat, state["a"], grid, backend)


def total_energy_dissipation_rhs(
    state: State,
    grid: Any,
    backend: Any,
    linear_ops: dict[str, Any],
    params: Any,
) -> float:
    """Return the signed dissipative contribution to `d_t E`."""

    xp = backend.xp
    p = derived_parameters(params)
    phi_hat = derive_phi_hat(state["omega"], grid)
    density_hat = (
        -linear_ops["omega"] * grid.kperp2 * (xp.abs(phi_hat) ** 2)
        - linear_ops["psi"] * grid.kperp2 * (xp.abs(state["psi"]) ** 2)
        - (linear_ops["a"] / p.N2) * xp.abs(state["a"]) ** 2
    )
    return modal_average(density_hat, grid, backend)


def compute_conserved_quantity_budgets(
    state: State,
    *,
    grid: Any,
    backend: Any,
    params: Any,
    linear_ops: dict[str, Any] | None = None,
    extra_rhs_terms: dict[str, dict[str, float]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return total energy plus signed RHS budget terms.

    The low-beta total energy is not conserved ideally; the explicit source is
    stored as `stratification` with sign convention

    `d_t E = stratification + dissipation + forcing + ...`.
    """

    rhs_terms: dict[str, float] = {
        "stratification": total_energy_stratification_rhs(state, grid, backend, params),
    }
    if linear_ops is not None:
        rhs_terms["dissipation"] = total_energy_dissipation_rhs(
            state,
            grid,
            backend,
            linear_ops,
            params,
        )
    if extra_rhs_terms is not None:
        rhs_terms.update(
            {
                name: float(value)
                for name, value in extra_rhs_terms.get("total_energy", {}).items()
            }
        )

    return {
        "total_energy": {
            "value": total_energy(state, grid, backend, params),
            "rhs_terms": rhs_terms,
        }
    }


def compute_equation_scalar_diagnostics(
    state: State,
    *,
    grid: Any,
    fft: Any,
    backend: Any,
    params: Any,
    workspace: Any | None = None,
    linear_ops: dict[str, Any] | None = None,
    budget_rhs_terms: dict[str, dict[str, float]] | None = None,
    extra_rhs_terms: dict[str, dict[str, float]] | None = None,
) -> dict[str, float]:
    """Return low-beta-stratified scalar diagnostics.

    Equation modules own scientifically meaningful scalar diagnostics. The
    standard `total_energy` / `total_energy_rhs_*` names are what generic
    budget plotting tools expect; equation-specific source terms should follow
    the same `total_energy_rhs_<name>` pattern.
    """

    alfvenic = alfvenic_energy(state, grid, backend)
    stratification_energy = a_energy(state, grid, backend, params)
    diagnostics = {
        "alfvenic_energy": alfvenic,
        "a_energy": stratification_energy,
        "total_energy_proxy": alfvenic + stratification_energy,
    }

    budgets = compute_conserved_quantity_budgets(
        state,
        grid=grid,
        backend=backend,
        params=params,
        linear_ops=linear_ops,
        extra_rhs_terms=extra_rhs_terms,
    )
    rhs_terms = budgets["total_energy"].setdefault("rhs_terms", {})
    if budget_rhs_terms is not None and "total_energy" in budget_rhs_terms:
        rhs_terms.clear()
        rhs_terms.update({name: float(value) for name, value in budget_rhs_terms["total_energy"].items()})
    rhs_terms.setdefault("dissipation", 0.0)
    rhs_terms.setdefault("forcing", 0.0)
    diagnostics.update(flatten_conserved_quantity_budgets(budgets))
    return diagnostics
