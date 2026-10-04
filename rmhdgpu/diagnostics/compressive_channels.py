"""Elsasser-wave and buoyancy-channel diagnostics for the inhomogeneous RMHD sets.

Shared by `rmhd_by_nokia_rho` and `rmhd_by_nokia_s`, following Squire et al.,
arXiv:2607.08036. Each equation module hands over its fields as `ChannelFields`.
Everything here works on Fourier coefficients and does no FFTs, because the
budgets are evaluated twice per time step.

Variables: `drho = delta rho/rho_0`, `db_par = delta B_par/B_0`, and `psi` is in
Alfven (velocity) units, so `b_perp/vA = delta B_perp/B_0`. The background
gradients point along x, with `u_x = -dy(phi)` and `b_x = -dy(psi)`.

Elsasser fields, in the paper's convention:

    z^± = delta u_perp ∓ delta B_perp/sqrt(4 pi rho_0) = zhat x grad_perp(phi ∓ psi)
    W^± = <|z^±|^2>/4,   so W+ + W- = 0.5 <|u_perp|^2 + |b_perp|^2>

z+ (potential phi - psi) obeys d_t f + vA d_z f = 0, so it travels along +B_0.
It is the wave the initial conditions launch. z- (potential phi + psi) is the
wave it reflects into.

The compressive fluctuations that z+ stirs up act in three ways (paper Fig. 1):

    DCF  direct compressive feedback: they act back on z+.
         q_dcf = -(d_t W+)|_buoyancy, positive while buoyancy drains z+.
    CCR  compressively catalyzed reflection: they create z-.
         q_ccr_source = -(d_t W-)|_buoyancy, so it is NEGATIVE while buoyancy
         creates z-, which it does in almost every row of a z+ run.
    ACR  Alfven-catalyzed relaxation: they release background free energy.
         acr_B + acr_g + acr_th = Y, the source term of the total energy.

Buoyancy, `-g dy(drho)` in `omega_t`, is the only compressive term in the psi
and omega equations, so the W± budgets contain only buoyancy, dissipation and
forcing. Budget terms are signed contributions: `d_t Q = sum(rhs terms)`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from rmhdgpu.fourier_diagnostics import modal_average, modal_inner_product_average
from rmhdgpu.operators import dy


# Columns written by `channel_scalar_diagnostics`, in the order they are written.
CHANNEL_SCALAR_DIAGNOSTIC_INFO = {
    "w_plus": "Elsasser energy W+ = <|z+|^2>/4 of z+ = u_perp - b_perp, the launched wave.",
    "w_minus": "Elsasser energy W- = <|z-|^2>/4 of z- = u_perp + b_perp, the reflected wave.",
    "w_plus_kperp": "Energy-weighted <k_perp> of z+; the closure outer scale is l_perp = 1/<k_perp>.",
    "w_minus_kperp": "Energy-weighted <k_perp> of z-.",
    "w_plus_kprl": "Energy-weighted <|k_par|> of z+; chi_A = 2 sqrt(W+) <k_perp> / (vA <k_par>).",
    "w_minus_kprl": "Energy-weighted <|k_par|> of z-.",
    "w_plus_align": "rms(z+_x)/rms(|z+|): share of z+ along the gradient direction x; 1/sqrt(2) if isotropic. Eq. (47) factor.",
    "w_minus_align": "rms(z-_x)/rms(|z-|).",
    "u_perp_rms": "rms(|u_perp|) = sqrt(<|grad_perp phi|^2>); divide by vA for delta u_perp/vA.",
    "b_perp_rms": "rms(|b_perp|) = sqrt(<|grad_perp psi|^2>), Alfven units; divide by vA for delta B_perp/B_0.",
    "q_dcf": "DCF heating rate -(d_t W+)|_buoyancy, instantaneous; positive while buoyancy drains z+.",
    "q_ccr_source": (
        "-(d_t W-)|_buoyancy, instantaneous; NEGATIVE while buoyancy creates z- (the CCR source). "
        "w_minus_rhs_buoyancy is the same term with the opposite sign, averaged over the output interval."
    ),
    "N_sq": "Run constant: Brunt-Vaisala frequency squared, N^2 = -g * F_rho. Positive is stable.",
    "vA": "Run constant: Alfven speed, echoed so plotting scripts need only the CSV.",
    "acr_B": "Instantaneous ACR channel exchanging with background magnetic free energy (the paper's Y_B).",
    "acr_g": "Instantaneous ACR channel exchanging with background potential energy (Y_g); buoyancy work on W+ plus W-.",
    "acr_th": "Instantaneous ACR channel exchanging with background thermal free energy (Y_th); not net heating.",
}


@dataclass(frozen=True, slots=True)
class ChannelFields:
    """The Fourier fields every channel diagnostic is built from.

    `phi_hat` is the stream function and `psi_hat` the flux function. The s
    equation set derives `drho_hat` from s and db_par before building this.
    """

    phi_hat: Any
    psi_hat: Any
    du_par_hat: Any
    db_par_hat: Any
    drho_hat: Any


# ---------------------------------------------------------------------------
# Elsasser fields, z± = zhat x grad_perp(phi ∓ psi)
# ---------------------------------------------------------------------------


def elsasser_potentials(phi_hat: Any, psi_hat: Any) -> tuple[Any, Any]:
    """Return the potentials `(phi - psi, phi + psi)` of `(z+, z-)`.

    This is the one place the sign convention is written. The map is linear, so
    passing the rates `(phi_t, psi_t)` returns the rates of the two potentials.
    """

    return phi_hat - psi_hat, phi_hat + psi_hat


def _modal_energy(f_hat: Any, grid: Any, backend: Any) -> Any:
    """Fourier density of `W = <|grad_perp f|^2>/4` for one Elsasser potential f."""

    return 0.25 * grid.kperp2 * backend.xp.abs(f_hat) ** 2


def elsasser_energy(f_hat: Any, grid: Any, backend: Any) -> float:
    """Return `W = <|z|^2>/4 = <k_perp^2 |f_hat|^2>/4` for the Elsasser potential f."""

    return modal_average(_modal_energy(f_hat, grid, backend), grid, backend)


def elsasser_energy_rate(f_hat: Any, f_t_hat: Any, grid: Any, backend: Any) -> float:
    """Return `d_t W = <grad_perp f . grad_perp f_t>/2` when f changes at the rate f_t.

    Each W± budget term is this rate, taken with the part of f_t that one RHS
    term produces.
    """

    return 0.5 * modal_inner_product_average(grid.kperp2 * f_hat, f_t_hat, grid, backend)


def _energy_weighted_mean(f_hat: Any, grid: Any, backend: Any, k_squared: Any) -> float:
    """Mean of `sqrt(k_squared)` weighted by the modal energy of f; zero for an empty field."""

    density = _modal_energy(f_hat, grid, backend)
    energy = modal_average(density, grid, backend)
    if not energy > 0.0:
        return 0.0
    return modal_average(backend.xp.sqrt(k_squared) * density, grid, backend) / energy


def elsasser_kperp_mean(f_hat: Any, grid: Any, backend: Any) -> float:
    """Return the energy-weighted `<k_perp>`; the closure outer scale is `l_perp = 1/<k_perp>`."""

    return _energy_weighted_mean(f_hat, grid, backend, grid.kperp2)


def elsasser_kprl_mean(f_hat: Any, grid: Any, backend: Any) -> float:
    """Return the energy-weighted `<|k_par|>`.

    Together with `<k_perp>` this gives `chi_A = 2 sqrt(W) <k_perp> / (vA <k_par>)`.
    Modes in the k_par = 0 plane have no Alfven frequency, so chi_A cannot
    describe them.
    """

    return _energy_weighted_mean(f_hat, grid, backend, grid.kpar2)


def elsasser_x_alignment(f_hat: Any, grid: Any, backend: Any) -> float:
    """Return `rms(z_x)/rms(|z|)`, the share of z along the gradient direction x.

    This is the projection factor in the slaving estimate, Eq. (47). It is not
    the dynamic alignment between z+ and z-. With `z = zhat x grad_perp f`,
    `z_x = -dy(f)`, so by Parseval `<z_x^2>` is a sum of `k_y^2 |f_hat|^2` and
    `<|z|^2>` a sum of `k_perp^2 |f_hat|^2`. The ratio is 1/sqrt(2) for energy
    spread isotropically in the perpendicular plane (Eq. 48), 1 when z points
    along x (k along y), and 0 when z points along y. An empty field returns 0.
    """

    power = backend.xp.abs(f_hat) ** 2
    z_squared = modal_average(grid.kperp2 * power, grid, backend)
    if not z_squared > 0.0:
        return 0.0
    z_x_squared = modal_average(grid.ky**2 * power, grid, backend)
    return math.sqrt(z_x_squared / z_squared)


def perp_gradient_rms(potential_hat: Any, grid: Any, backend: Any) -> float:
    """Return `rms(|zhat x grad_perp f|) = sqrt(<k_perp^2 |f_hat|^2>)`.

    With f = phi this is rms(u_perp). With f = psi it is rms(b_perp), and since
    psi is in Alfven units, rms(b_perp)/vA = rms(delta B_perp/B_0).
    """

    power = grid.kperp2 * backend.xp.abs(potential_hat) ** 2
    return math.sqrt(max(modal_average(power, grid, backend), 0.0))


# ---------------------------------------------------------------------------
# Buoyancy and ACR channels, built from four correlators
# ---------------------------------------------------------------------------


def gradient_correlators(fields: ChannelFields, grid: Any, backend: Any) -> dict[str, float]:
    """Return the four averages `<field dy(potential)>` every channel is built from.

    Since `u_x = -dy(phi)` and `b_x = -dy(psi)`, these are minus the fluxes
    across the background gradient; for example `rho_phi = -<drho u_x>`.
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
    """Return the buoyancy contributions to `d_t W+` and `d_t W-`.

    The buoyancy term `omega_t = -g dy(drho)` gives `phi_t = inv_lap_perp(-g dy(drho))`
    and `psi_t = 0`, so both Elsasser potentials change at the rate phi_t.
    Integrating `elsasser_energy_rate` by parts gives

        d_t W±|_buoyancy = -(g/2) <drho dy(phi ∓ psi)> = -(g/2) (rho_phi ∓ rho_psi)

    The two add up to `-g <drho dy(phi)> = acr_g`. `q_dcf` is minus the W+ term.
    """

    return {
        "w_plus": -0.5 * p.g * (correlators["rho_phi"] - correlators["rho_psi"]),
        "w_minus": -0.5 * p.g * (correlators["rho_phi"] + correlators["rho_psi"]),
    }


def stratification_channels(correlators: dict[str, float], p: Any) -> dict[str, float]:
    """Split the total-energy source Y into the three ACR channels:

        acr_B  = vA^2 K_b0 <db_par dy(phi)> - vA K_b0 <du_par dy(psi)>                  (Y_B)
        acr_g  = -g <drho dy(phi)>                                                      (Y_g)
        acr_th = -(vA^2 K_p0/gamma + w_s/chi) <db_par dy(phi)> - w_s <drho dy(phi)>     (Y_th)

    with `w_s = cs^2 K_s / (gamma (gamma - 1))`. They sum to the equation
    module's `total_energy_stratification_rhs`. Only acr_B + acr_g is net
    heating: acr_th extracts background thermal free energy that dissipation
    later returns as heat, so do not count it twice.
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


# ---------------------------------------------------------------------------
# What the equation modules save: W± budgets and the channel CSV columns
# ---------------------------------------------------------------------------


def elsasser_budgets(
    fields: ChannelFields,
    grid: Any,
    backend: Any,
    p: Any,
    *,
    linear_ops: dict[str, Any] | None = None,
    correlators: dict[str, float] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return the `w_plus` / `w_minus` budgets, value plus signed RHS terms.

    Shaped for `rmhdgpu.diagnostics.budget.flatten_conserved_quantity_budgets`.
    Buoyancy is the only ideal term that changes W±, and the run driver adds
    forcing itself. Pass the correlators if they are already computed.
    """

    if correlators is None:
        correlators = gradient_correlators(fields, grid, backend)
    work = buoyancy_work(correlators, p)
    z_plus_hat, z_minus_hat = elsasser_potentials(fields.phi_hat, fields.psi_hat)

    budgets: dict[str, dict[str, Any]] = {
        "w_plus": {
            "value": elsasser_energy(z_plus_hat, grid, backend),
            "rhs_terms": {"buoyancy": work["w_plus"]},
        },
        "w_minus": {
            "value": elsasser_energy(z_minus_hat, grid, backend),
            "rhs_terms": {"buoyancy": work["w_minus"]},
        },
    }
    if linear_ops is not None:
        # Damping alone gives omega_t = -D_omega omega, so phi_t = -D_omega phi,
        # and psi_t = -D_psi psi. When D_omega and D_psi differ, damping also
        # moves energy between W+ and W-.
        z_plus_t_hat, z_minus_t_hat = elsasser_potentials(
            -linear_ops["omega"] * fields.phi_hat,
            -linear_ops["psi"] * fields.psi_hat,
        )
        budgets["w_plus"]["rhs_terms"]["dissipation"] = elsasser_energy_rate(
            z_plus_hat, z_plus_t_hat, grid, backend,
        )
        budgets["w_minus"]["rhs_terms"]["dissipation"] = elsasser_energy_rate(
            z_minus_hat, z_minus_t_hat, grid, backend,
        )
    return budgets


def channel_scalar_diagnostics(
    fields: ChannelFields,
    grid: Any,
    backend: Any,
    p: Any,
) -> dict[str, float]:
    """Return the channel columns written to `scalar_diagnostics.csv`, all instantaneous.

    `q_dcf`, `q_ccr_source` and `acr_*` repeat the `w_*_rhs_buoyancy` and
    `total_energy_rhs_acr_*` budget terms, which the driver averages over each
    output interval. They are kept instantaneous on purpose: the closure
    estimates are built from `w_plus`, `w_plus_kperp` and `w_plus_align` in the
    same row, so measurement and prediction are taken at the same instant.
    """

    correlators = gradient_correlators(fields, grid, backend)
    work = buoyancy_work(correlators, p)
    z_plus_hat, z_minus_hat = elsasser_potentials(fields.phi_hat, fields.psi_hat)

    diagnostics: dict[str, float] = {
        "w_plus": elsasser_energy(z_plus_hat, grid, backend),
        "w_minus": elsasser_energy(z_minus_hat, grid, backend),
        "w_plus_kperp": elsasser_kperp_mean(z_plus_hat, grid, backend),
        "w_minus_kperp": elsasser_kperp_mean(z_minus_hat, grid, backend),
        "w_plus_kprl": elsasser_kprl_mean(z_plus_hat, grid, backend),
        "w_minus_kprl": elsasser_kprl_mean(z_minus_hat, grid, backend),
        "w_plus_align": elsasser_x_alignment(z_plus_hat, grid, backend),
        "w_minus_align": elsasser_x_alignment(z_minus_hat, grid, backend),
        "u_perp_rms": perp_gradient_rms(fields.phi_hat, grid, backend),
        "b_perp_rms": perp_gradient_rms(fields.psi_hat, grid, backend),
        "q_dcf": -work["w_plus"],
        "q_ccr_source": -work["w_minus"],
        "N_sq": float(p.N_sq),
        "vA": float(p.vA),
    }
    diagnostics.update(stratification_channels(correlators, p))
    return diagnostics


# ---------------------------------------------------------------------------
# Inputs of the slaving closure, Eqs. (46)-(47), used by vis/ and the scan driver
# ---------------------------------------------------------------------------


def background_drives(p: Any) -> dict[str, float]:
    """Return the background drive F of each compressive field (Eq. 46).

    For a straight field and no mean flow. In a pure z+ state (phi = -psi, so
    b_x = -u_x) whose compressive fields are still zero, the equations reduce to

        d_t drho           = F_rho u_x,   F_rho = g/(vA^2 (1 + chi)) - K_rho0   (= -N^2/g)
        d_t (du_par/vA)    = F_u   u_x,   F_u   = -K_b0
        d_t (db_par/alpha) = F_b   u_x,   F_b   = -K_b0 + K_p0/gamma

    with each field in the units of `slaved_field_units`. du_par is really
    driven by b_x, so its line holds because b_x = -u_x for z+. F_rho is
    written without dividing by g, so it is also defined when g = 0.
    """

    return {
        "drho": p.g / (p.vA**2 * (1.0 + p.chi)) - p.K_rho0,
        "du_par": -p.K_b0,
        "db_par": -p.K_b0 + p.K_p0 / p.gamma,
    }


def slaved_field_units(p: Any) -> dict[str, float]:
    """Return what to divide each saved compressive field by to get Eq. (47) units.

    drho stays delta rho/rho_0, du_par becomes delta u_par/vA, and db_par
    becomes (vA^2/vS^2) delta B_par/B_0, since alpha = vS^2/vA^2.
    """

    return {"drho": 1.0, "du_par": p.vA, "db_par": p.alpha}
