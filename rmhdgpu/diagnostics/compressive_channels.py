"""Buoyancy and stratification diagnostics shared by the rho and s equation sets.

Each equation module supplies ChannelFields in (drho, db_par, du_par) variables.
The calculations below follow Squire et al., arXiv:2607.08036:

    gradient_correlators -> buoyancy_work / stratification_channels
                        -> Elsasser budgets / scalar CSV diagnostics

DCF heating is minus the buoyancy contribution to d_t W+. The three ACR
channels sum to the total stratification source. CCR has no separate exact
RHS term here; it is diagnosed through the generated W- and W+ dissipation.
All budget RHS terms are signed contributions to d_t energy.

General Elsasser-wave calculations live in diagnostics.alfvenic. They remain
importable from this module for existing plotting scripts and callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Re-export the shared helpers to preserve existing imports.
from rmhdgpu.diagnostics.alfvenic import (
    elsasser_dissipation_rhs,
    elsasser_energy,
    elsasser_energy_rhs_budget,
    elsasser_kperp_mean,
    elsasser_kprl_mean,
    elsasser_modal_density,
    elsasser_potential,
    elsasser_x_alignment,
)
from rmhdgpu.fourier_diagnostics import modal_inner_product_average
from rmhdgpu.operators import dy, inv_lap_perp


CHANNEL_SCALAR_DIAGNOSTIC_INFO = {
    "w_plus": "Elsasser energy of z^+ = u_perp - b_perp: 0.25 <|grad_perp(phi - psi)|^2>.",
    "w_minus": "Elsasser energy of z^- = u_perp + b_perp: 0.25 <|grad_perp(phi + psi)|^2>.",
    "w_plus_kperp": "Energy-weighted <k_perp> of z+; the closure outer scale is l_perp = 1 / <k_perp>.",
    "w_minus_kperp": "Energy-weighted <k_perp> of z-.",
    "w_plus_kprl": "Energy-weighted <k_par> of z+; chi_A = 2 sqrt(W+) <k_perp> / (vA <k_par>).",
    "w_minus_kprl": "Energy-weighted <k_par> of z-.",
    "w_plus_align": "rms(z+_x)/rms(|z+|): share of z+ along the gradient direction x; 1/sqrt(2) if isotropic. Eq. (47) factor.",
    "w_minus_align": "rms(z-_x)/rms(|z-|).",
    "q_dcf": "Instantaneous direct-compressive-feedback heating rate, -(d_t W+)|_buoyancy.",
    "q_ccr_source": "Instantaneous buoyancy drive of W-, -(d_t W-)|_buoyancy; the z- source feeding CCR.",
    "acr_B": "ACR channel exchanging with background magnetic free energy (their Y_B).",
    "acr_g": "ACR channel exchanging with background potential energy (their Y_g).",
    "acr_th": "ACR channel exchanging with background thermal free energy (their Y_th); not net heating.",
    "corr_rho_phi": "<drho dy(phi)>; minus the cross-field density flux.",
    "corr_rho_psi": "<drho dy(psi)>.",
    "corr_b_phi": "<db_par dy(phi)>.",
    "corr_u_psi": "<du_par dy(psi)>.",
    "N_sq": "Run constant: Brunt-Vaisala frequency squared, N^2 = -g * F_rho. Positive is stable.",
    "vA": "Run constant: Alfven speed, echoed so plotting scripts need only the CSV.",
}


@dataclass(frozen=True, slots=True)
class ChannelFields:
    """Fourier potentials and compressive fields in a common normalization.

    The s equation set derives drho_hat before passing it here. Compressive
    fields are optional when only Elsasser-wave diagnostics are needed.
    """

    phi_hat: Any
    psi_hat: Any
    du_par_hat: Any = None
    db_par_hat: Any = None
    drho_hat: Any = None


def gradient_correlators(fields: ChannelFields, grid: Any, backend: Any) -> dict[str, float]:
    """Return the `<field dy(potential)>` correlators every channel is built from.

    With `u_perp = z_hat x grad_perp phi` these are the cross-field transport
    fluxes along the background-gradient direction: `dy(phi) = -u_x` and
    `dy(psi) = -vA B_x / B`, so for example `corr_rho_phi = -V_rho,x`.
    """

    dyphi_hat = dy(fields.phi_hat, grid)
    dypsi_hat = dy(fields.psi_hat, grid)
    return {
        "rho_phi": modal_inner_product_average(dyphi_hat, fields.drho_hat, grid, backend),
        "rho_psi": modal_inner_product_average(dypsi_hat, fields.drho_hat, grid, backend),
        "b_phi": modal_inner_product_average(dyphi_hat, fields.db_par_hat, grid, backend),
        "u_psi": modal_inner_product_average(dypsi_hat, fields.du_par_hat, grid, backend),
    }


def buoyancy_work(correlators: dict[str, float], p: Any) -> dict[str, float]:
    """Signed work from the -g dy(drho) term in omega_t:

        d_t W^±|_buoy = -(g/2) <drho dy(phi)> +- (g/2) <drho dy(psi)>.

    Their sum is -g <drho dy(phi)> = acr_g. DCF heating is minus the W+ term.
    """

    return {
        "w_plus": -0.5 * p.g * (correlators["rho_phi"] - correlators["rho_psi"]),
        "w_minus": -0.5 * p.g * (correlators["rho_phi"] + correlators["rho_psi"]),
    }


def stratification_channels(correlators: dict[str, float], p: Any) -> dict[str, float]:
    """Split the total stratification source Y into magnetic, gravity and thermal terms.

    Only acr_B + acr_g is net heating. acr_th extracts background thermal free
    energy that is later dissipated back into heat; do not count it twice.
    """

    entropy_weight = p.cs2 * p.K_s / (p.gamma * (p.gamma - 1.0))
    return {
        "acr_B": p.vA**2 * p.K_b0 * correlators["b_phi"] - p.vA * p.K_b0 * correlators["u_psi"],
        "acr_g": -p.g * correlators["rho_phi"],
        "acr_th": (
            (-p.vA**2 * p.K_p0 / p.gamma - entropy_weight / p.chi) * correlators["b_phi"]
            - entropy_weight * correlators["rho_phi"]
        ),
    }


def elsasser_budgets(
    fields: ChannelFields,
    grid: Any,
    backend: Any,
    p: Any,
    *,
    linear_ops: dict[str, Any] | None = None,
    correlators: dict[str, float] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return `w_plus` / `w_minus` conserved-quantity budgets.

    Shaped for `rmhdgpu.diagnostics.budget.flatten_conserved_quantity_budgets`,
    so the equation modules can merge this straight into their returned budget
    dictionary. Pass already computed correlators to avoid repeating them.
    """

    if correlators is None:
        correlators = gradient_correlators(fields, grid, backend)
    work = buoyancy_work(correlators, p)

    budgets: dict[str, dict[str, Any]] = {}
    for name, sign in (("w_plus", 1), ("w_minus", -1)):
        rhs_terms: dict[str, float] = {"buoyancy": work[name]}
        if linear_ops is not None:
            rhs_terms["dissipation"] = elsasser_dissipation_rhs(
                fields,
                grid,
                backend,
                linear_ops,
                sign=sign,
            )
        budgets[name] = {
            "value": elsasser_energy(fields, grid, backend, sign=sign),
            "rhs_terms": rhs_terms,
        }
    return budgets


def channel_scalar_diagnostics(
    fields: ChannelFields,
    grid: Any,
    backend: Any,
    p: Any,
) -> dict[str, float]:
    """Return the instantaneous channel columns written to `scalar_diagnostics.csv`.

    `q_dcf` and `q_ccr_source` duplicate information in the `w_*_rhs_buoyancy`
    budget columns, but deliberately: the budget columns are averaged over the
    output interval by the driver, whereas the closure prediction is built from
    the instantaneous `w_plus`, `w_plus_kperp` and `w_plus_align` in the same
    row. Comparing measurement to prediction wants both sides evaluated at the
    same instant.

    `N_sq` and `vA` are run constants echoed into every row so that plotting
    scripts need only the CSV.
    """

    correlators = gradient_correlators(fields, grid, backend)
    work = buoyancy_work(correlators, p)

    diagnostics: dict[str, float] = {
        "w_plus": elsasser_energy(fields, grid, backend, sign=1),
        "w_minus": elsasser_energy(fields, grid, backend, sign=-1),
        "w_plus_kperp": elsasser_kperp_mean(fields, grid, backend, sign=1),
        "w_minus_kperp": elsasser_kperp_mean(fields, grid, backend, sign=-1),
        "w_plus_kprl": elsasser_kprl_mean(fields, grid, backend, sign=1),
        "w_minus_kprl": elsasser_kprl_mean(fields, grid, backend, sign=-1),
        "w_plus_align": elsasser_x_alignment(fields, grid, backend, sign=1),
        "w_minus_align": elsasser_x_alignment(fields, grid, backend, sign=-1),
        "q_dcf": -work["w_plus"],
        "q_ccr_source": -work["w_minus"],
        "N_sq": float(p.N_sq),
        "vA": float(p.vA),
    }
    diagnostics.update(stratification_channels(correlators, p))
    diagnostics.update({f"corr_{name}": value for name, value in correlators.items()})
    return diagnostics


def rhs_channel_fields(rhs_state: Any, grid: Any) -> ChannelFields:
    """Convert omega_t and psi_t into the potentials used by Elsasser RHS budgets."""

    return ChannelFields(
        phi_hat=inv_lap_perp(rhs_state["omega"], grid),
        psi_hat=rhs_state["psi"],
    )
