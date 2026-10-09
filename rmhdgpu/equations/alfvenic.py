"""Two-field Alfvénic RMHD equation set.

This is the Alfvénic subsector of the homogeneous S09 system with only the
fields `[psi, omega]`, where `phi = inv_lap_perp(omega)`.

The ideal equations are

- `psi_t = vA * dz(phi) - {phi, psi}`
- `omega_t = vA * dz(lap_perp psi) - {phi, omega} + {psi, lap_perp psi}`

with `omega = lap_perp(phi)` and `j = -lap_perp(psi)`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from rmhdgpu.diagnostics.alfvenic import elsasser_energies
from rmhdgpu.diagnostics.budget import flatten_conserved_quantity_budgets
from rmhdgpu.diagnostics.scalar import STANDARD_ENERGY_SCALAR_DIAGNOSTIC_INFO
from rmhdgpu.diagnostics.spectra import (
    elsasser_perpendicular_spectra,
    perpendicular_shell_spectrum,
)
from rmhdgpu.fourier_diagnostics import modal_average
from rmhdgpu.operators import inv_lap_perp, lap_perp, poisson_bracket
from rmhdgpu.state import State


EQUATION_SET_NAME = "alfvenic"
FIELD_NAMES = ["psi", "omega"]
DEFAULT_INITIAL_CONDITION = "alfven_mode"

SCALAR_DIAGNOSTIC_INFO = {
    **STANDARD_ENERGY_SCALAR_DIAGNOSTIC_INFO,
    "alfvenic_energy": "Total two-field Alfvenic energy: 0.5 <|grad phi|^2 + |grad psi|^2>.",
    "elsasser_energy_plus": "E+ = 0.5 <|grad(phi - psi)|^2>.",
    "elsasser_energy_minus": "E- = 0.5 <|grad(phi + psi)|^2>.",
    "elsasser_energy_ratio": "Elsasser energy ratio E+ / E-.",
    "normalized_cross_helicity": "(E- - E+) / (E+ + E-) for the package potential convention.",
}

_ORIGINAL_POISSON_BRACKET = poisson_bracket


@dataclass(frozen=True, slots=True)
class AlfvenicParameters:
    """Scalar parameters used by the two-field Alfvénic system."""

    vA: float


def _param_float(params: Any, name: str) -> float:
    if isinstance(params, Mapping):
        return float(params[name])
    return float(getattr(params, name))


def derived_parameters(params: Any) -> AlfvenicParameters:
    """Return the compact scalar parameter block for this equation set."""

    return AlfvenicParameters(vA=_param_float(params, "vA"))


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
    """Return the Fourier-space ideal RHS of the two-field Alfvénic system."""

    if workspace is None:
        raise ValueError("Alfvenic ideal_rhs requires a Workspace instance.")

    p = derived_parameters(params)
    psi_hat = state["psi"]
    omega_hat = state["omega"]

    rhs_state = state.zeros_like() if out is None else out
    rhs_state.fill_zero()

    # Linear terms:
    #   vA dz(phi) = -i vA kz inv_kperp2 omega
    #   vA dz(lap_perp psi) = -i vA kz kperp2 psi
    # These are written in place to avoid materializing phi_hat or lap_psi_hat.
    rhs_psi = rhs_state["psi"]
    rhs_omega = rhs_state["omega"]

    rhs_psi[...] = omega_hat
    rhs_psi *= grid.inv_kperp2
    rhs_psi *= grid.kz
    rhs_psi *= -1j * p.vA

    rhs_omega[...] = psi_hat
    rhs_omega *= grid.kperp2
    rhs_omega *= grid.kz
    rhs_omega *= -1j * p.vA

    if getattr(params, "equation_mode", "nonlinear") == "linear":
        return rhs_state

    if poisson_bracket is not _ORIGINAL_POISSON_BRACKET:
        phi_hat = workspace.complex["c1"]
        lap_psi_hat = workspace.complex["c2"]

        phi_hat[...] = omega_hat
        phi_hat *= grid.inv_kperp2
        phi_hat *= -1.0

        lap_psi_hat[...] = psi_hat
        lap_psi_hat *= grid.kperp2
        lap_psi_hat *= -1.0

        rhs_psi[...] -= poisson_bracket(
            phi_hat,
            psi_hat,
            grid,
            fft,
            workspace,
            mask=dealias_mask,
        )
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
        return rhs_state

    xp = workspace.backend.xp
    c0 = workspace.complex["c0"]
    r0 = workspace.real["r0"]  # phi_x
    r1 = workspace.real["r1"]  # phi_y
    r2 = workspace.real["r2"]  # psi_x
    r3 = workspace.real["r3"]  # psi_y
    r4 = workspace.real["r4"]  # nonlinear accumulator
    r5 = workspace.real.get("r5")
    if r5 is None:
        raise ValueError("Workspace needs at least six real buffers for Alfvenic ideal_rhs.")

    # Shared real derivatives. Reusing these cuts the nonlinear RHS from three
    # separate Poisson brackets (15 FFTs) to eight inverse and two forward FFTs.
    c0[...] = omega_hat
    c0 *= grid.inv_kperp2
    c0 *= grid.kx
    c0 *= -1j
    fft.c2r(c0, out=r0)

    c0[...] = omega_hat
    c0 *= grid.inv_kperp2
    c0 *= grid.ky
    c0 *= -1j
    fft.c2r(c0, out=r1)

    c0[...] = psi_hat
    c0 *= grid.kx
    c0 *= 1j
    fft.c2r(c0, out=r2)

    c0[...] = psi_hat
    c0 *= grid.ky
    c0 *= 1j
    fft.c2r(c0, out=r3)

    # psi_t nonlinear term: -{phi, psi}
    xp.multiply(r0, r3, out=r4)
    xp.multiply(r1, r2, out=r5)
    r4 -= r5
    fft.r2c(r4, out=c0)
    if dealias_mask is not None:
        c0 *= dealias_mask
    rhs_psi -= c0

    # omega_t nonlinear terms:
    #   -{phi, omega} + {psi, lap_perp psi}
    # accumulated in real space before one forward transform.
    c0[...] = omega_hat
    c0 *= grid.ky
    c0 *= 1j
    fft.c2r(c0, out=r5)
    xp.multiply(r0, r5, out=r4)
    r4 *= -1.0

    c0[...] = omega_hat
    c0 *= grid.kx
    c0 *= 1j
    fft.c2r(c0, out=r5)
    xp.multiply(r1, r5, out=r5)
    r4 += r5

    c0[...] = psi_hat
    c0 *= grid.kperp2
    c0 *= grid.ky
    c0 *= -1j
    fft.c2r(c0, out=r5)
    xp.multiply(r2, r5, out=r5)
    r4 += r5

    c0[...] = psi_hat
    c0 *= grid.kperp2
    c0 *= grid.kx
    c0 *= -1j
    fft.c2r(c0, out=r5)
    xp.multiply(r3, r5, out=r5)
    r4 -= r5

    fft.r2c(r4, out=c0)
    if dealias_mask is not None:
        c0 *= dealias_mask
    rhs_omega += c0

    return rhs_state


def linear_matrix(kx: float, ky: float, kz: float, params: Any) -> np.ndarray:
    """Return the 2x2 linear matrix for one Fourier mode.

    Field order is `[psi, omega]`. For `k_perp = 0`, the matrix is set to zero
    because `phi = inv_lap_perp(omega)` is not meaningful on that subspace.
    """

    p = derived_parameters(params)
    matrix = np.zeros((2, 2), dtype=np.complex128)
    ikz = 1j * float(kz)
    kperp2 = float(kx) ** 2 + float(ky) ** 2

    if kperp2 > 0.0:
        matrix[0, 1] = -p.vA * ikz / kperp2
        matrix[1, 0] = -p.vA * ikz * kperp2
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
    """Return perpendicular shell spectra for the Alfvénic energy pieces."""

    xp = backend.xp
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
        "z_plus": elsasser["z_plus"],
        "z_minus": elsasser["z_minus"],
    }


def total_energy_modal_density(state: State, grid: Any, backend: Any, params: Any) -> Any:
    """Return the modal quadratic density for the two-field Alfvénic energy."""

    xp = backend.xp
    phi_hat = derive_phi_hat(state["omega"], grid)
    return 0.5 * grid.kperp2 * (xp.abs(phi_hat) ** 2 + xp.abs(state["psi"]) ** 2)


def total_energy(state: State, grid: Any, backend: Any, params: Any) -> float:
    """Return `0.5 <|grad phi|^2 + |grad psi|^2>`."""

    return modal_average(total_energy_modal_density(state, grid, backend, params), grid, backend)


def alfvenic_energy(state: State, grid: Any, backend: Any, params: Any) -> float:
    """Return the total two-field Alfvénic energy."""

    return total_energy(state, grid, backend, params)


def total_energy_dissipation_rhs(
    state: State,
    grid: Any,
    backend: Any,
    linear_ops: dict[str, Any],
    params: Any,
) -> float:
    """Return the signed dissipative contribution to `d_t E`."""

    xp = backend.xp
    phi_hat = derive_phi_hat(state["omega"], grid)
    density_hat = (
        -linear_ops["omega"] * grid.kperp2 * (xp.abs(phi_hat) ** 2)
        - linear_ops["psi"] * grid.kperp2 * (xp.abs(state["psi"]) ** 2)
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
    """Return total energy plus named signed RHS contributions."""

    rhs_terms: dict[str, float] = {}
    if linear_ops is not None:
        rhs_terms["dissipation"] = total_energy_dissipation_rhs(state, grid, backend, linear_ops, params)
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
    """Return Alfvénic equation-set scalar diagnostics."""

    diagnostics = {
        "alfvenic_energy": alfvenic_energy(state, grid, backend, params),
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
