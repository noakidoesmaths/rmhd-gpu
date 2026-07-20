"""Inhomogeneous five-field equation set.

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

The evolved fields are `[psi, omega, du_par, db_par, drho]`, with
`phi = inv_lap_perp(omega)`. The inhomogeneous ideal equations are


- `psi_t = vA * dz(phi) - {phi, psi}`
- `omega_t = vA * dz(lap_perp psi) - {phi, omega} + {psi, lap_perp psi} 
    - g dy(delta rho/rho_0)`
- ` (db_par)_t = alpha * dz(du_par) + alpha/vA{psi, du_par} - {Phi, db_par} 
      - alpha*K_B0 * dy(phi) + alpha*K_p0/gamma *dy(phi) `
- `(du_par)_t = vA^2 dz(db_par) + vA{psi, db_par} - {Phi, du_par} + vA*K_B0 dy(psi) `
-  (delta rho/rho0)_t = -alpha/chi * dz(du_par) -alpha/vA*chi * {psi, du_par}
    -alpha/chi * K_B0 dy(phi) + K_rho0 dy(phi) - alpha/chi * K_p0 dy(phi) - {phi, delta rho/rho0}
- 



note we use the convention db_par = delta B / B_0, drho = delta rho/rho_0
alpha = chi/(1+chi)   chi = cs^2/vA^2 
the background gradients K_B and K_p as well as the gravity g
can be considered a constant. 

we choose g and K_p then set
K_B = 2chi/gamma K_p_0 + 2g/vA^2

"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from rmhdgpu.diagnostics.budget import flatten_conserved_quantity_budgets
from rmhdgpu.diagnostics.scalar import STANDARD_ENERGY_SCALAR_DIAGNOSTIC_INFO
from rmhdgpu.fourier_diagnostics import modal_average, modal_inner_product_average
from rmhdgpu.operators import dy, dz, inv_lap_perp, lap_perp, poisson_bracket
from rmhdgpu.diagnostics.spectra import parallel_shell_spectrum, perpendicular_shell_spectrum
from rmhdgpu.state import State


EQUATION_SET_NAME = "inhomo_rmhd"
FIELD_NAMES = ["psi", "omega", "du_par", "db_par", "drho"]
DEFAULT_INITIAL_CONDITION = "alfven_mode"
DIAGNOSTIC_GAMMA = 5.0 / 3.0

# Scalar diagnostic names provided by this equation module. The standard
# `total_energy*` names are what budget plotting tools expect. Additional
# entries are S09-specific energy partitions useful for quick run inspection.
SCALAR_DIAGNOSTIC_INFO = {
    **STANDARD_ENERGY_SCALAR_DIAGNOSTIC_INFO,
    "alfvenic_energy": "Alfvenic part of the S09 energy: 0.5 <|grad phi|^2 + |grad psi|^2>.",
    "dupar_energy": "Unweighted kinetic parallel energy proxy: 0.5 <dupar^2>.",
    "dbpar_energy": "Unweighted magnetic-compressive energy proxy: 0.5 <dbpar^2>.",
    "total_energy_proxy": "Legacy unweighted sum of alfvenic_energy, dupar_energy, and dbpar_energy.",
}


@dataclass(frozen=True, slots=True)
class inhomo_rmhd_parameters:
    """Scalar parameters and diagnostic weights used by this equation set."""

    vA: float
    chi: float
    alpha: float
    gamma: float
    g: float
    K_p0: float
    K_b0: float
    K_rho0: float
    dbpar_energy_weight: float
    entropy_energy_weight: float
    N_sq: float
    K_s: float 
    cs2: float

def _param_float(params: Any, name: str) -> float:
    if isinstance(params, Mapping):
        return float(params[name])
    return float(getattr(params, name))



def derived_parameters(params: Any) -> inhomo_rmhd_parameters:
    """Return the compact set of scalars used by the RMHD equations.

    This is the first place to edit if a new parameter enters the physics.
    `gamma` is currently fixed to `5/3` for the diagnostic entropy
    normalization.
    """

    vA = _param_float(params, "vA")
    chi = _param_float(params, "cs2_over_vA2")
    g = _param_float(params, "g")
    K_p0 = _param_float(params, "K_p0")
    K_rho0 = _param_float(params, "K_rho0")

    gamma = DIAGNOSTIC_GAMMA

    alpha = chi / (1.0 + chi)
    K_b0 = (g / vA**2) - (chi * K_p0) / gamma 

    cs2 = chi * vA**2
    vS2 = alpha * vA**2
    
    N_sq = - g * (vS2/cs2 *(K_b0 + chi * K_p0/gamma) - K_rho0)
    
    
    dbpar_energy_weight = 1 / alpha
    entropy_energy_weight = cs2/(gamma**2 * (gamma - 1))

    K_s = K_p0 - gamma * K_rho0
    return inhomo_rmhd_parameters(
        vA=vA,
        chi=chi,
        alpha=alpha,
        gamma=gamma,
        g=g,
        K_p0=K_p0,
        K_b0=K_b0,
        K_rho0=K_rho0,
        dbpar_energy_weight = dbpar_energy_weight,
        entropy_energy_weight = entropy_energy_weight,
        N_sq=N_sq,
        K_s=K_s,
        cs2=cs2
    )


def derive_phi_hat(omega_hat: Any, grid: Any) -> Any:
    """Return `phi_hat = inv_lap_perp(omega_hat)`."""

    return inv_lap_perp(omega_hat, grid)


def derive_j_hat(psi_hat: Any, grid: Any) -> Any:
    """Return `j_hat = -lap_perp(psi_hat) = +k_perp^2 psi_hat`."""

    return -lap_perp(psi_hat, grid)

def derive_s_hat(drho_hat: Any, dbpar_hat: Any, params: Any) -> Any:
    """Derives drho using s and db_par."""

    p = derived_parameters(params)
    return - dbpar_hat * p.gamma / p.chi  - p.gamma * drho_hat 


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
    """Return the Fourier-space ideal RHS of the inhomogeneous four-field system."""

    p = derived_parameters(params)

    psi_hat = state["psi"]
    omega_hat = state["omega"]
    du_par_hat = state["du_par"]
    dbpar_hat = state["db_par"]
    drho_hat = state["drho"]

    phi_hat = derive_phi_hat(omega_hat, grid)
    lap_psi_hat = lap_perp(psi_hat, grid)

    rhs_state = state.zeros_like() if out is None else out
    rhs_state.fill_zero()

    "psi_t = vA * dz(phi) - {phi, psi}"
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

    """
    `omega_t = vA * dz(lap_perp psi) - {phi, omega} + {psi, lap_perp psi} 
    - g dy(delta rho/rho_0)` 
    """
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
    rhs_omega[...] -= p.g * dy(drho_hat, grid)

    "(du_par)_t = vA^2 dz(db_par) + vA{psi, db_par} - {Phi, du_par} - vA*K_B0 dy(psi) "

    rhs_dupar = rhs_state["du_par"]
    rhs_dupar[...] = (p.vA**2) * dz(dbpar_hat, grid)
    rhs_dupar[...] += (p.vA) * poisson_bracket(
        psi_hat,
        dbpar_hat,
        grid,
        fft,
        workspace,
        mask=dealias_mask,
    )
    rhs_dupar[...] -= poisson_bracket(
        phi_hat,
        du_par_hat,
        grid,
        fft,
        workspace,
        mask=dealias_mask,
    )
    rhs_dupar[...] -= p.vA * p.K_b0 * dy(psi_hat, grid)

    """
    (db_par)_t = alpha * dz(du_par) + alpha/vA * {psi, du_par} - {Phi, db_par} 
      - alpha*K_B0 * dy(phi) + alpha*K_p0/gamma *dy(phi) 
    """

    rhs_dbpar = rhs_state["db_par"]
    rhs_dbpar[...] = p.alpha * dz(du_par_hat, grid)
    rhs_dbpar[...] += p.alpha/p.vA * poisson_bracket(
        psi_hat,
        du_par_hat,
        grid,
        fft,
        workspace,
        mask=dealias_mask,
    )
    rhs_dbpar[...] -= poisson_bracket(
        phi_hat,
        dbpar_hat,
        grid,
        fft,
        workspace,
        mask=dealias_mask,
    )
    rhs_dbpar[...] += p.alpha * p.K_b0 * dy(phi_hat, grid)
    rhs_dbpar[...] -= p.alpha * p.K_p0/p.gamma * dy(phi_hat, grid)

    """
    (delta rho/rho0)_t = -alpha/chi * dz(du_par) - alpha/(vA*chi) * {psi, du_par}
    -alpha/chi * K_B0 dy(phi) + K_rho0 dy(phi) 
    - alpha/chi * K_p0 dy(phi) - {phi, delta rho/rho0}
    """
    rhs_drho = rhs_state["drho"]
    rhs_drho[...] = -p.alpha/p.chi * dz(du_par_hat, grid)
    rhs_drho[...] -= p.alpha/(p.vA * p.chi) * poisson_bracket(
        psi_hat,
        du_par_hat,
        grid,
        fft,
        workspace,
        mask=dealias_mask,
    )
    rhs_drho[...] -= p.alpha /  p.chi * p.K_b0 * dy(phi_hat, grid)
    rhs_drho[...] += p.K_rho0 * dy(phi_hat, grid)
    rhs_drho[...] -= p.alpha / p.gamma * p.K_p0* dy(phi_hat, grid)
    rhs_drho[...] -= poisson_bracket(
        phi_hat,
        drho_hat,
        grid,
        fft,
        workspace,
        mask=dealias_mask,
    )



    return rhs_state


def linear_matrix(kx: float, ky: float, kz: float, params: Any) -> np.ndarray:
    """Return the 5x5 linear matrix for one Fourier mode.

    The field order is `[psi, omega, upar, dbpar, drho]`. For `k_perp = 0`, the
    `psi/omega` Alfvénic block is set to zero because the inverse perpendicular
    Laplacian is not meaningful there in the RMHD subspace.
    """

    p = derived_parameters(params)
    matrix = np.zeros((5, 5), dtype=np.complex128)
    ikz = 1j * float(kz)
    iky = 1j * float(ky)
    kperp2 = float(kx) ** 2 + float(ky) ** 2

    if kperp2 > 0.0:
        matrix[0, 1] = -p.vA * ikz / kperp2
        matrix[1, 0] = -p.vA * ikz * kperp2
        matrix[2, 1] = -iky * p.alpha * (p.K_b0 - p.K_p0 / p.gamma)/kperp2
        matrix[4, 1] = iky * (p.alpha * p.K_b0/p.chi - p.K_rho0 + p.alpha * p.K_p0/p.gamma)/kperp2
        
    matrix[1, 4] = -iky * p.g
    matrix[2, 3] = p.alpha * ikz
    matrix[3, 0] = -iky * p.vA * p.K_b0
    matrix[3, 2] = ikz * p.vA ** 2
    matrix[4, 3] = -p.alpha * ikz/p.chi
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


def _energy_modal_densities(
    state: State,
    grid: Any,
    backend: Any,
    params: Any | None,
) -> dict[str, Any]:
    """Return the modal energy densities binned into the shell spectra."""

    xp = backend.xp
    p = derived_parameters(params)
    phi_hat = derive_phi_hat(state["omega"], grid)
    s_hat = derive_s_hat(state["drho"], state["db_par"], params)
    kperp2 = grid.kperp2

    # Elsasser fields z± = u_perp ± b_perp/sqrt(4 pi rho0) = z_hat x grad_perp(phi ± psi).
    # The 1/4 weight is the standard pseudo-energy normalization, so that
    # z_plus + z_minus = u_perp + b_perp shell by shell.
    return {
        "u_perp": 0.5 * kperp2 * (xp.abs(phi_hat) ** 2),
        "b_perp": 0.5 * kperp2 * (xp.abs(state["psi"]) ** 2),
        "du_par": 0.5 * (xp.abs(state["du_par"]) ** 2),
        "db_par": 0.5 * p.dbpar_energy_weight * (xp.abs(state["db_par"]) ** 2),
        "drho": 0.5 * p.entropy_energy_weight * (xp.abs(s_hat) ** 2),
        "z_plus": 0.25 * kperp2 * (xp.abs(phi_hat + state["psi"]) ** 2),
        "z_minus": 0.25 * kperp2 * (xp.abs(phi_hat - state["psi"]) ** 2),
    }


def perpendicular_energy_spectra(
    state: State,
    grid: Any,
    backend: Any,
    *,
    bin_width: float | None = None,
    params: Any | None = None,
) -> dict[str, np.ndarray]:
    """Return the inhomogeneous rmhd shell spectra."""

    spectra: dict[str, np.ndarray] = {}
    for name, density in _energy_modal_densities(state, grid, backend, params).items():
        kperp, spectrum = perpendicular_shell_spectrum(density, grid, backend, bin_width=bin_width)
        spectra.setdefault("kperp", kperp)
        spectra[name] = spectrum
    return spectra


def parallel_energy_spectra(
    state: State,
    grid: Any,
    backend: Any,
    *,
    bin_width: float | None = None,
    params: Any | None = None,
) -> dict[str, np.ndarray]:
    """Return the inhomogeneous rmhd parallel (kz) shell spectra."""

    spectra: dict[str, np.ndarray] = {}
    for name, density in _energy_modal_densities(state, grid, backend, params).items():
        kprl, spectrum = parallel_shell_spectrum(density, grid, backend, bin_width=bin_width)
        spectra.setdefault("kprl", kprl)
        spectra[name] = spectrum
    return spectra


def total_energy_modal_density(state: State, grid: Any, backend: Any, params: Any) -> Any:
    """Return the modal quadratic density for the S09 total energy.

    The Alfvénic fields are measured as physical perpendicular amplitudes:
    `u_perp ~ grad_perp phi` and `b_perp ~ grad_perp psi`. Therefore the modal
    density uses `k_perp^2 |phi_hat|^2` and `k_perp^2 |psi_hat|^2`, not raw
    potential amplitudes.

    In code variables:

    `E = 0.5 * (|grad_perp phi|^2 + |grad_perp psi|^2 + |upar|^2`
    `           +  |dbpar|^2 + 
                |s|^2 `)`

    with `gamma = 5/3`.
    """

    xp = backend.xp
    p = derived_parameters(params)
    phi_hat = derive_phi_hat(state["omega"], grid)
    s_hat = derive_s_hat(state["drho"], state["db_par"], params)
    return (
        0.5 * grid.kperp2 * (xp.abs(phi_hat) ** 2 + xp.abs(state["psi"]) ** 2)
        + 0.5 * xp.abs(state["du_par"]) ** 2
        + 0.5 * p.dbpar_energy_weight * xp.abs(state["db_par"]) ** 2
        + 0.5 * p.entropy_energy_weight * xp.abs(s_hat) ** 2
    )


def total_energy(state: State, grid: Any, backend: Any, params: Any) -> float:
    """Return the volume-averaged total energy for this equation set."""

    density_hat = total_energy_modal_density(state, grid, backend, params)
    return modal_average(density_hat, grid, backend)


def alfvenic_energy(state: State, grid: Any, backend: Any) -> float:
    """Return the inhomogeneous RMHD Alfvenic energy partition."""

    xp = backend.xp
    phi_hat = derive_phi_hat(state["omega"], grid)
    density_hat = 0.5 * grid.kperp2 * (xp.abs(phi_hat) ** 2 + xp.abs(state["psi"]) ** 2)
    return modal_average(density_hat, grid, backend)


def _unweighted_field_energy(field_hat: Any, grid: Any, backend: Any) -> float:
    return modal_average(0.5 * backend.xp.abs(field_hat) ** 2, grid, backend)


def total_energy_stratification_rhs(state: State, grid: Any, backend: Any, params: Any) -> float:
    """
    Return the source term of the energy (Y)
    
    Y = 
    (KB * vA^2 -  vA^2 * Kp/gamma - vA^2 * Ks/(gamma*(gamma - 1)))<db_par dy(phi)> 
    (-vA * KB)<du_par dy(psi)>
    ( -g - (cs^2 * Ks)/(gamma(gamma - 1)) )< drho dy(phi)>
    """

    p = derived_parameters(params)

    phi_hat = derive_phi_hat(state["omega"], grid)
    dyphi_hat = dy(phi_hat, grid)
    psi_hat = state["psi"]
    dypsi_hat = dy(psi_hat, grid)
    
    du_prl_avg = modal_inner_product_average(dypsi_hat, state["du_par"], grid, backend)
    du_prl_weight = -p.vA * p.K_b0
    
    db_prl_avg = modal_inner_product_average(dyphi_hat, state["db_par"], grid, backend)
    db_prl_weight = p.K_b0 * p.vA**2 - p.vA**2 * p.K_p0 / p.gamma - p.vA**2 * p.K_s /(p.gamma * (p.gamma - 1))

    drho_avg = modal_inner_product_average(dyphi_hat, state["drho"], grid, backend)
    drho_weight = -p.g - p.cs2 * p.K_s/(p.gamma * (p.gamma - 1))

    return db_prl_avg * db_prl_weight + du_prl_avg * du_prl_weight + drho_avg * drho_weight

def total_energy_dissipation_rhs(
    state: State,
    grid: Any,
    backend: Any,
    linear_ops: dict[str, Any],
    params: Any,
) -> float:
    """
    Return the signed dissipative contribution to `d_t E`.

    Sign convention:

    `d_t E = forcing + other_terms + dissipation`

    so this term is negative when diagonal damping removes energy. The weights
    match :func:`total_energy_modal_density` exactly.
    """

    xp = backend.xp
    p = derived_parameters(params)
    phi_hat = derive_phi_hat(state["omega"], grid)
    s_hat = derive_s_hat(state["drho"], state["db_par"], params)
    # s is diagnostic here: damping drho and db_par induces
    # `s_t = -D_rho s + (gamma/chi)(D_b - D_rho) db_par`, so the entropy part
    # of the budget carries a cross term that vanishes only when the drho and
    # db_par operators are identical (e.g. auto-dissipation mode).
    density_hat = (
        -linear_ops["omega"] * grid.kperp2 * (xp.abs(phi_hat) ** 2)
        - linear_ops["psi"] * grid.kperp2 * (xp.abs(state["psi"]) ** 2)
        - linear_ops["du_par"] * xp.abs(state["du_par"]) ** 2
        - p.dbpar_energy_weight * linear_ops["db_par"] * xp.abs(state["db_par"]) ** 2
        - p.entropy_energy_weight * linear_ops["drho"] * xp.abs(s_hat) ** 2
        + p.entropy_energy_weight
        * (p.gamma / p.chi)
        * (linear_ops["db_par"] - linear_ops["drho"])
        * xp.real(xp.conj(s_hat) * state["db_par"])
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

    rhs_terms: dict[str, float] = {
        "stratification": total_energy_stratification_rhs(state, grid, backend, params)
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
    """Return S09-specific scalar diagnostics.

    Equation modules own scientifically meaningful scalar diagnostics. For a
    new equation set, provide this function and include the standard
    `total_energy` / `total_energy_rhs_*` names so generic budget plotting
    tools can operate without knowing equation-specific details.
    """

    alfvenic = alfvenic_energy(state, grid, backend)
    du_par = _unweighted_field_energy(state["du_par"], grid, backend)
    db_par = _unweighted_field_energy(state["db_par"], grid, backend)
    diagnostics = {
        "alfvenic_energy": alfvenic,
        "du_par_energy": du_par,
        "db_par_energy": db_par,
        "total_energy_proxy": alfvenic + du_par + db_par,
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
