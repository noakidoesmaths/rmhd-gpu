"""Homogeneous S09 five-field equation set.

This module is intended to be the main physics-facing file for this equation
set. Generic solver bookkeeping should live elsewhere; the functions here
define the evolved fields, derived parameters, derived fields, ideal RHS,
linear representation, dissipation operators, energy, and budget terms.

Fourier conventions used throughout the solver:

- `z` is the parallel direction
- real arrays have shape `(Nx, Ny, Nz)`
- Fourier arrays have shape `(Nx, Ny, Nz//2 + 1)` from `rfftn`
- `lap_perp(f_hat) = -k_perp^2 f_hat`
- `inv_lap_perp(f_hat) = -inv_kperp2 f_hat`

The evolved fields are `[psi, omega, upar, dbpar, s]`, with
`phi = inv_lap_perp(omega)`. The homogeneous ideal equations are

- `psi_t = vA * dz(phi) - {phi, psi}`
- `omega_t = vA * dz(lap_perp psi) - {phi, omega} + {psi, lap_perp psi}`
- `dbpar_t = alpha * dz(upar) - {phi, dbpar} + alpha * {psi, upar}`
- `upar_t = vA^2 * dz(dbpar) - {phi, upar} + vA^2 * {psi, dbpar}`
- `s_t = -{phi, s}`

with `chi = cs2_over_vA2 = cs^2 / vA^2` and `alpha = chi / (1 + chi)`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from rmhdgpu.diagnostics.budget import flatten_conserved_quantity_budgets
from rmhdgpu.diagnostics.alfvenic import elsasser_energies
from rmhdgpu.diagnostics.scalar import STANDARD_ENERGY_SCALAR_DIAGNOSTIC_INFO
from rmhdgpu.fourier_diagnostics import modal_average
from rmhdgpu.operators import dz, inv_lap_perp, lap_perp, poisson_bracket
from rmhdgpu.diagnostics.spectra import elsasser_perpendicular_spectra, perpendicular_shell_spectrum
from rmhdgpu.state import State


EQUATION_SET_NAME = "s09"
FIELD_NAMES = ["psi", "omega", "upar", "dbpar", "s"]
DEFAULT_INITIAL_CONDITION = "alfven_mode"
DIAGNOSTIC_GAMMA = 5.0 / 3.0

# Scalar diagnostic names provided by this equation module. The standard
# `total_energy*` names are what budget plotting tools expect. Additional
# entries are S09-specific energy partitions useful for quick run inspection.
SCALAR_DIAGNOSTIC_INFO = {
    **STANDARD_ENERGY_SCALAR_DIAGNOSTIC_INFO,
    "alfvenic_energy": "Alfvenic part of the S09 energy: 0.5 <|grad phi|^2 + |grad psi|^2>.",
    "elsasser_energy_plus": "E+ = 0.5 <|grad(phi - psi)|^2>.",
    "elsasser_energy_minus": "E- = 0.5 <|grad(phi + psi)|^2>.",
    "elsasser_energy_ratio": "Elsasser energy ratio E+ / E-.",
    "normalized_cross_helicity": "(E- - E+) / (E+ + E-) for the package potential convention.",
    "upar_energy": "Unweighted kinetic parallel energy proxy: 0.5 <upar^2>.",
    "dbpar_energy": "Unweighted magnetic-compressive energy proxy: 0.5 <dbpar^2>.",
    "entropy_variance": "Unweighted entropy variance proxy: 0.5 <s^2>.",
    "total_energy_proxy": "Legacy unweighted sum of alfvenic_energy, upar_energy, dbpar_energy, and entropy_variance.",
}


@dataclass(frozen=True, slots=True)
class S09Parameters:
    """Scalar parameters and diagnostic weights used by this equation set."""

    vA: float
    chi: float
    alpha: float
    gamma: float
    dbpar_energy_weight: float
    entropy_energy_weight: float


def _param_float(params: Any, name: str) -> float:
    if isinstance(params, Mapping):
        return float(params[name])
    return float(getattr(params, name))


def derived_parameters(params: Any) -> S09Parameters:
    """Return the compact set of scalars used by the S09 equations.

    This is the first place to edit if a new parameter enters the physics.
    `gamma` is currently fixed to `5/3` for the diagnostic entropy
    normalization.
    """

    vA = _param_float(params, "vA")
    chi = _param_float(params, "cs2_over_vA2")
    alpha = chi / (1.0 + chi)
    gamma = DIAGNOSTIC_GAMMA
    return S09Parameters(
        vA=vA,
        chi=chi,
        alpha=alpha,
        gamma=gamma,
        dbpar_energy_weight=1.0 / alpha,
        entropy_energy_weight=chi / (gamma**2 * (gamma - 1.0)),
    )


def derive_phi_hat(omega_hat: Any, grid: Any) -> Any:
    """Return `phi_hat = inv_lap_perp(omega_hat)`."""

    return inv_lap_perp(omega_hat, grid)


def derive_j_hat(psi_hat: Any, grid: Any) -> Any:
    """Return `j_hat = -lap_perp(psi_hat) = +k_perp^2 psi_hat`."""

    return -lap_perp(psi_hat, grid)


def characteristic_speeds(params: Any) -> list[float]:
    """Return parallel linear speeds relevant to the CFL estimate."""

    p = derived_parameters(params)
    return [p.vA, p.vA * np.sqrt(p.alpha)]


def ideal_rhs(
    state: State,
    grid: Any,
    fft: Any,
    workspace: Any,
    params: Any,
    dealias_mask: Any | None = None,
    out: State | None = None,
) -> State:
    """Return the Fourier-space ideal RHS of the homogeneous five-field system."""

    p = derived_parameters(params)

    psi_hat = state["psi"]
    omega_hat = state["omega"]
    upar_hat = state["upar"]
    dbpar_hat = state["dbpar"]
    s_hat = state["s"]

    phi_hat = derive_phi_hat(omega_hat, grid)
    lap_psi_hat = lap_perp(psi_hat, grid)

    rhs_state = state.zeros_like() if out is None else out
    rhs_state.fill_zero()

    rhs_psi = rhs_state["psi"]
    rhs_psi[...] = p.vA * dz(phi_hat, grid)
    rhs_psi[...] -= poisson_bracket(
        phi_hat,
        psi_hat,
        grid,
        fft,
        workspace,
        mask=dealias_mask,
    )

    rhs_omega = rhs_state["omega"]
    rhs_omega[...] = p.vA * dz(lap_psi_hat, grid)
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

    rhs_dbpar = rhs_state["dbpar"]
    rhs_dbpar[...] = p.alpha * dz(upar_hat, grid)
    rhs_dbpar[...] -= poisson_bracket(
        phi_hat,
        dbpar_hat,
        grid,
        fft,
        workspace,
        mask=dealias_mask,
    )
    rhs_dbpar[...] += p.alpha * poisson_bracket(
        psi_hat,
        upar_hat,
        grid,
        fft,
        workspace,
        mask=dealias_mask,
    )

    rhs_upar = rhs_state["upar"]
    rhs_upar[...] = (p.vA**2) * dz(dbpar_hat, grid)
    rhs_upar[...] -= poisson_bracket(
        phi_hat,
        upar_hat,
        grid,
        fft,
        workspace,
        mask=dealias_mask,
    )
    rhs_upar[...] += (p.vA**2) * poisson_bracket(
        psi_hat,
        dbpar_hat,
        grid,
        fft,
        workspace,
        mask=dealias_mask,
    )

    rhs_s = rhs_state["s"]
    rhs_s[...] -= poisson_bracket(
        phi_hat,
        s_hat,
        grid,
        fft,
        workspace,
        mask=dealias_mask,
    )

    return rhs_state


def linear_matrix(kx: float, ky: float, kz: float, params: Any) -> np.ndarray:
    """Return the 5x5 linear matrix for one Fourier mode.

    The field order is `[psi, omega, upar, dbpar, s]`. For `k_perp = 0`, the
    `psi/omega` Alfvénic block is set to zero because the inverse perpendicular
    Laplacian is not meaningful there in the RMHD subspace.
    """

    p = derived_parameters(params)
    matrix = np.zeros((5, 5), dtype=np.complex128)
    ikz = 1j * float(kz)
    kperp2 = float(kx) ** 2 + float(ky) ** 2

    if kperp2 > 0.0:
        matrix[0, 1] = -p.vA * ikz / kperp2
        matrix[1, 0] = -p.vA * ikz * kperp2

    matrix[2, 3] = (p.vA**2) * ikz
    matrix[3, 2] = p.alpha * ikz
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
    """Return the standard S09 perpendicular shell spectra."""

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
    _, upar = perpendicular_shell_spectrum(
        0.5 * (xp.abs(state["upar"]) ** 2),
        grid,
        backend,
        bin_width=bin_width,
    )
    _, dbpar = perpendicular_shell_spectrum(
        0.5 * p.dbpar_energy_weight * (xp.abs(state["dbpar"]) ** 2),
        grid,
        backend,
        bin_width=bin_width,
    )
    _, entropy = perpendicular_shell_spectrum(
        0.5 * p.entropy_energy_weight * (xp.abs(state["s"]) ** 2),
        grid,
        backend,
        bin_width=bin_width,
    )
    elsasser = elsasser_perpendicular_spectra(
        state,
        grid,
        backend,
        bin_width=bin_width,
    )
    return {
        "kperp": kperp,
        "u_perp": u_perp,
        "b_perp": b_perp,
        "upar": upar,
        "dbpar": dbpar,
        "s": entropy,
        "z_plus": elsasser["z_plus"],
        "z_minus": elsasser["z_minus"],
    }


def total_energy_modal_density(state: State, grid: Any, backend: Any, params: Any) -> Any:
    """Return the modal quadratic density for the S09 total energy.

    The Alfvénic fields are measured as physical perpendicular amplitudes:
    `u_perp ~ grad_perp phi` and `b_perp ~ grad_perp psi`. Therefore the modal
    density uses `k_perp^2 |phi_hat|^2` and `k_perp^2 |psi_hat|^2`, not raw
    potential amplitudes.

    In code variables:

    `E = 0.5 * (|grad_perp phi|^2 + |grad_perp psi|^2 + |upar|^2`
    `           + alpha^(-1) |dbpar|^2`
    `           + [chi / (gamma^2 (gamma - 1))] |s|^2)`

    with `gamma = 5/3`.
    """

    xp = backend.xp
    p = derived_parameters(params)
    phi_hat = derive_phi_hat(state["omega"], grid)
    return (
        0.5 * grid.kperp2 * (xp.abs(phi_hat) ** 2 + xp.abs(state["psi"]) ** 2)
        + 0.5 * xp.abs(state["upar"]) ** 2
        + 0.5 * p.dbpar_energy_weight * xp.abs(state["dbpar"]) ** 2
        + 0.5 * p.entropy_energy_weight * xp.abs(state["s"]) ** 2
    )


def total_energy(state: State, grid: Any, backend: Any, params: Any) -> float:
    """Return the volume-averaged total energy for this equation set."""

    density_hat = total_energy_modal_density(state, grid, backend, params)
    return modal_average(density_hat, grid, backend)


def alfvenic_energy(state: State, grid: Any, backend: Any) -> float:
    """Return the S09 Alfvenic energy partition."""

    xp = backend.xp
    phi_hat = derive_phi_hat(state["omega"], grid)
    density_hat = 0.5 * grid.kperp2 * (xp.abs(phi_hat) ** 2 + xp.abs(state["psi"]) ** 2)
    return modal_average(density_hat, grid, backend)


def _unweighted_field_energy(field_hat: Any, grid: Any, backend: Any) -> float:
    return modal_average(0.5 * backend.xp.abs(field_hat) ** 2, grid, backend)


def total_energy_dissipation_rhs(
    state: State,
    grid: Any,
    backend: Any,
    linear_ops: dict[str, Any],
    params: Any,
) -> float:
    """Return the signed dissipative contribution to `d_t E`.

    Sign convention:

    `d_t E = forcing + other_terms + dissipation`

    so this term is negative when diagonal damping removes energy. The weights
    match :func:`total_energy_modal_density` exactly.
    """

    xp = backend.xp
    p = derived_parameters(params)
    phi_hat = derive_phi_hat(state["omega"], grid)
    density_hat = (
        -linear_ops["omega"] * grid.kperp2 * (xp.abs(phi_hat) ** 2)
        - linear_ops["psi"] * grid.kperp2 * (xp.abs(state["psi"]) ** 2)
        - linear_ops["upar"] * xp.abs(state["upar"]) ** 2
        - p.dbpar_energy_weight * linear_ops["dbpar"] * xp.abs(state["dbpar"]) ** 2
        - p.entropy_energy_weight * linear_ops["s"] * xp.abs(state["s"]) ** 2
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
    """Return conserved-quantity values plus named signed RHS contributions."""

    rhs_terms: dict[str, float] = {}
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
    """Return S09-specific scalar diagnostics.

    Equation modules own scientifically meaningful scalar diagnostics. For a
    new equation set, provide this function and include the standard
    `total_energy` / `total_energy_rhs_*` names so generic budget plotting
    tools can operate without knowing equation-specific details.
    """

    alfvenic = alfvenic_energy(state, grid, backend)
    upar = _unweighted_field_energy(state["upar"], grid, backend)
    dbpar = _unweighted_field_energy(state["dbpar"], grid, backend)
    entropy = _unweighted_field_energy(state["s"], grid, backend)
    diagnostics = {
        "alfvenic_energy": alfvenic,
        "upar_energy": upar,
        "dbpar_energy": dbpar,
        "entropy_variance": entropy,
        "total_energy_proxy": alfvenic + upar + dbpar + entropy,
    }
    diagnostics.update(elsasser_energies(state, grid, backend))

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


# Controlled forcing acts on the Alfvénic pair only. The equation declares
# storage here; the shared helpers never guess a different equation's fields.
from rmhdgpu.forcing_fields import (
    standard_metric as forcing_metric,
    standard_native_parameters as forcing_native_parameters,
    standard_energy_factors as forcing_energy_factors,
    standard_branch_values as forcing_branch_values,
    standard_apply_gain as forcing_apply_gain,
    standard_seed_branch as forcing_seed_branch,
    standard_characteristic_speed as forcing_characteristic_speed,
    standard_budget_work as forcing_budget_work,
    vorticity_shell_density as forcing_shell_density,
    vorticity_perpendicular_energy as forcing_perpendicular_energy,
    vorticity_perpendicular_shell_energy as forcing_perpendicular_shell_energy,
    vorticity_measurement as forcing_measurement,
)


def forcing_fields(config):
    """Map controlled Elsasser branches to this module's evolved fields."""
    from rmhdgpu.forcing_fields import alfvenic_fields
    return alfvenic_fields(config, velocity="omega", magnetic="psi")
