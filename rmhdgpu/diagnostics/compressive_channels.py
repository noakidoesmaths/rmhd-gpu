"""Elsasser-wave and buoyancy-channel diagnostics for the inhomogeneous RMHD sets.

Shared by `rmhd_by_nokia_rho` and `rmhd_by_nokia_s`, following Squire et al.,
arXiv:2607.08036. Each equation module hands over its fields as `ChannelFields`.
Everything here works on Fourier coefficients and does no FFTs, because the
budgets are evaluated twice per time step. A box average of a product of two
real fields, <a b>, is taken straight from their Fourier coefficients
(Parseval) with `modal_inner_product_average`.

Variables: `drho = delta rho/rho_0`, `db_par = delta B_par/B_0`, and `psi` is in
Alfven (velocity) units, so `b_perp/vA = delta B_perp/B_0`. The background
gradients point along x, and

    u_perp = zhat x grad_perp(phi),   so u_x = -dy(phi)
    b_perp = zhat x grad_perp(psi),   so b_x = -dy(psi)

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
    "k_perp_plus": (
        "Mean k_perp of z+, int k_perp E+ dk_perp / int E+ dk_perp, summed exactly over every mode "
        "(no shell binning); the outer scale is l_perp = 1/k_perp_plus. Called w_plus_kperp before 2026-10-08."
    ),
    "k_perp_minus": "Mean k_perp of z-, as k_perp_plus. Called w_minus_kperp before 2026-10-08.",
    "k_prl_plus": (
        "Mean |k_par| of z+, int |k_par| E+ dk_par / int E+ dk_par; chi_A = z+_rms k_perp_plus / (vA k_prl_plus). "
        "Called w_plus_kprl before 2026-10-08."
    ),
    "k_prl_minus": "Mean |k_par| of z-, as k_prl_plus. Called w_minus_kprl before 2026-10-08.",
    "w_plus_align": "rms(z+_x)/rms(|z+|): share of z+ along the gradient direction x; 1/sqrt(2) if isotropic. Eq. (47) factor.",
    "w_minus_align": "rms(z-_x)/rms(|z-|).",
    "u_perp_rms": "rms(|u_perp|) = sqrt(<|grad_perp phi|^2>); divide by vA for delta u_perp/vA.",
    "b_perp_rms": "rms(|b_perp|) = sqrt(<|grad_perp psi|^2>), Alfven units; divide by vA for delta B_perp/B_0.",
    "V_rho_x": (
        "x component of V_rho = <(delta rho/rho_0) delta u_perp> (paper Eq. 79), the density flux "
        "across the background gradient; the closure predicts V_rho_x ~ eta_turb F_rho (Eq. 80). "
        "Exact, so it includes the z- part: V_rho_x = <drho z+_x>/2 + <drho z-_x>/2. "
        "Defined for every g, unlike acr_g/g."
    ),
    "q_dcf": (
        "DCF heating rate -(d_t W+)|_buoyancy = -(g/2) <drho z+_x>, instantaneous; "
        "positive while buoyancy drains z+."
    ),
    "q_ccr_source": (
        "-(d_t W-)|_buoyancy = -(g/2) <drho z-_x>, instantaneous; NEGATIVE while buoyancy creates z- "
        "(the CCR source). w_minus_rhs_buoyancy is the same term with the opposite sign, averaged "
        "over the output interval."
    ),
    "N_sq": "Run constant: Brunt-Vaisala frequency squared, N^2 = -g * F_rho. Positive is stable.",
    "vA": "Run constant: Alfven speed, echoed so plotting scripts need only the CSV.",
    "acr_B": "Instantaneous ACR channel exchanging with background magnetic free energy (the paper's Y_B), -vA^2 K_b0 V_psi_x.",
    "acr_g": "Instantaneous ACR channel exchanging with background potential energy (Y_g), g V_rho_x; buoyancy work on W+ plus W-.",
    "acr_th": "Instantaneous ACR channel exchanging with background thermal free energy (Y_th); not net heating.",
}

ACR_CHANNELS = ("acr_B", "acr_g", "acr_th")


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


def elsasser_energy(f_hat: Any, grid: Any, backend: Any) -> float:
    """Return `W = <|z|^2>/4` for `z = zhat x grad_perp(f)`.

    `|z|^2 = |grad_perp f|^2`, which in Fourier space is `k_perp^2 |f_hat|^2`.
    """

    z_squared = modal_average(grid.kperp2 * backend.xp.abs(f_hat) ** 2, grid, backend)
    return 0.25 * z_squared


def elsasser_energy_rate(f_hat: Any, f_t_hat: Any, grid: Any, backend: Any) -> float:
    """Return `d_t W = <grad_perp f . grad_perp f_t>/2` when f changes at the rate f_t.

    Each W± dissipation term is this rate, taken with the part of f_t that the
    damping produces.
    """

    return 0.5 * modal_inner_product_average(grid.kperp2 * f_hat, f_t_hat, grid, backend)


def elsasser_kperp_mean(f_hat: Any, grid: Any, backend: Any) -> float:
    """Return the energy-weighted mean k_perp of `z = zhat x grad_perp(f)`:

        <k_perp> = sum_k |k_perp| E_k / sum_k E_k,    E_k = k_perp^2 |f_k|^2 / 4

    The closure outer scale is `l_perp = 1/<k_perp>`. The sums run over every
    mode with its exact |k_perp|. Rebuilding this from the shell-binned
    spectra.csv uses shell centres instead and overestimates the outer-scale
    k_perp by ~20% (the [1, 2) shell is labelled 1.5). An empty field returns 0.
    """

    E_k = 0.25 * grid.kperp2 * backend.xp.abs(f_hat) ** 2
    W = modal_average(E_k, grid, backend)
    if not W > 0.0:
        return 0.0
    return modal_average(backend.xp.sqrt(grid.kperp2) * E_k, grid, backend) / W


def elsasser_kprl_mean(f_hat: Any, grid: Any, backend: Any) -> float:
    """Return the energy-weighted mean |k_par| of `z = zhat x grad_perp(f)`:

        <|k_par|> = sum_k |k_par| E_k / sum_k E_k,    E_k = k_perp^2 |f_k|^2 / 4

    Together with `<k_perp>` this gives `chi_A = z_rms <k_perp> / (vA <|k_par|>)`.
    Modes in the k_par = 0 plane have no Alfven frequency, so chi_A cannot
    describe them. An empty field returns 0.
    """

    E_k = 0.25 * grid.kperp2 * backend.xp.abs(f_hat) ** 2
    W = modal_average(E_k, grid, backend)
    if not W > 0.0:
        return 0.0
    return modal_average(backend.xp.sqrt(grid.kpar2) * E_k, grid, backend) / W


def elsasser_x_alignment(f_hat: Any, grid: Any, backend: Any) -> float:
    """Return `rms(z_x)/rms(|z|)`, the share of z along the gradient direction x.

    This is the projection factor in the slaving estimate, Eq. (47). It is not
    the dynamic alignment between z+ and z-. With `z = zhat x grad_perp f`,
    `z_x = -dy(f)`. The ratio is 1/sqrt(2) for energy spread isotropically in
    the perpendicular plane (Eq. 48), 1 when z points along x (k along y), and
    0 when z points along y. An empty field returns 0.
    """

    z_x = -dy(f_hat, grid)
    z_x_squared = modal_inner_product_average(z_x, z_x, grid, backend)  # <z_x^2>
    z_squared = 4.0 * elsasser_energy(f_hat, grid, backend)  # <|z|^2>
    if not z_squared > 0.0:
        return 0.0
    return math.sqrt(z_x_squared / z_squared)


def perp_gradient_rms(potential_hat: Any, grid: Any, backend: Any) -> float:
    """Return `rms(|zhat x grad_perp f|) = sqrt(<k_perp^2 |f_hat|^2>)`.

    With f = phi this is rms(u_perp). With f = psi it is rms(b_perp), and since
    psi is in Alfven units, rms(b_perp)/vA = rms(delta B_perp/B_0).
    """

    power = grid.kperp2 * backend.xp.abs(potential_hat) ** 2
    return math.sqrt(max(modal_average(power, grid, backend), 0.0))


# ---------------------------------------------------------------------------
# Fluxes across the background gradient, and the buoyancy and ACR channels
# ---------------------------------------------------------------------------


def fluxes_and_channels(fields: ChannelFields, grid: Any, backend: Any, p: Any) -> dict[str, float]:
    """Return the fluxes across the background gradient and the channel rates built from them.

    Every line below is one equation; all are instantaneous box averages.

        V_rho_x        = <drho u_x>                                   paper Eq. (79)
        V_psi_x        = <db_par u_x> - <du_par b_x>/vA               paper Eq. (79)
        V_rho_x_plus   = <drho z+_x>/2                                part carried by z+
        V_rho_x_minus  = <drho z-_x>/2                                part carried by z-
        d_t W+|_buoy   = g V_rho_x_plus     (minus this is q_dcf)
        d_t W-|_buoy   = g V_rho_x_minus    (minus this is q_ccr_source)
        acr_g          = g V_rho_x                                    Y_g
        acr_B          = -vA^2 K_b0 V_psi_x                           Y_B
        acr_th         = (vA^2 K_p0/gamma + w_s/chi) <db_par u_x> + w_s V_rho_x   Y_th

    with `w_s = cs^2 K_s / (gamma (gamma - 1))`. Since u_x = (z+_x + z-_x)/2,
    V_rho_x = V_rho_x_plus + V_rho_x_minus exactly. The paper's closure keeps
    only the z+ part, `V_rho ~ <z+ drho>/2`, so V_rho_x_plus is that estimate's
    left-hand side.

    The three ACR channels sum to the equation module's
    `total_energy_stratification_rhs`. Only acr_B + acr_g is net heating:
    acr_th extracts background thermal free energy that dissipation later
    returns as heat, so do not count it twice.
    """

    def mean(a_hat: Any, b_hat: Any) -> float:
        """Box average <a b> of two real fields, from their Fourier coefficients."""
        return modal_inner_product_average(a_hat, b_hat, grid, backend)

    drho = fields.drho_hat
    du_par = fields.du_par_hat
    db_par = fields.db_par_hat

    # Components along the gradient direction x.
    u_x = -dy(fields.phi_hat, grid)
    b_x = -dy(fields.psi_hat, grid)
    z_plus_x = u_x - b_x
    z_minus_x = u_x + b_x

    # Paper Eq. (79). psi is in Alfven units, so delta B_x/B_0 = b_x/vA.
    V_rho_x = mean(drho, u_x)
    V_psi_x = mean(db_par, u_x) - mean(du_par, b_x) / p.vA

    # The part of V_rho_x carried by each wave.
    V_rho_x_plus = 0.5 * mean(drho, z_plus_x)
    V_rho_x_minus = 0.5 * mean(drho, z_minus_x)

    # Buoyancy is the force g drho xhat on u_perp; the `-g dy(drho)` term in
    # omega_t is its curl. It does not change psi, so d_t z± = d_t u_perp. The
    # pressure that keeps u_perp incompressible does no work on the
    # divergence-free z±, so
    #     d_t W±|_buoyancy = <z± . (g drho xhat)>/2 = (g/2) <drho z±_x> = g V_rho_x±
    w_plus_buoyancy = p.g * V_rho_x_plus
    w_minus_buoyancy = p.g * V_rho_x_minus

    # The three ACR channels: Y = acr_B + acr_g + acr_th.
    w_s = p.cs2 * p.K_s / (p.gamma * (p.gamma - 1.0))
    acr_B = -p.vA**2 * p.K_b0 * V_psi_x
    acr_g = p.g * V_rho_x  # = w_plus_buoyancy + w_minus_buoyancy
    acr_th = (p.vA**2 * p.K_p0 / p.gamma + w_s / p.chi) * mean(db_par, u_x) + w_s * V_rho_x

    return {
        "V_rho_x": V_rho_x,
        "V_psi_x": V_psi_x,
        "V_rho_x_plus": V_rho_x_plus,
        "V_rho_x_minus": V_rho_x_minus,
        "w_plus_buoyancy": w_plus_buoyancy,
        "w_minus_buoyancy": w_minus_buoyancy,
        "acr_B": acr_B,
        "acr_g": acr_g,
        "acr_th": acr_th,
    }


# ---------------------------------------------------------------------------
# What the equation modules save: W± budgets and the channel CSV columns
# ---------------------------------------------------------------------------


def elsasser_budgets(
    fields: ChannelFields,
    fluxes: dict[str, float],
    grid: Any,
    backend: Any,
    *,
    linear_ops: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return the `w_plus` / `w_minus` budgets, value plus signed RHS terms.

    `fluxes` is the output of `fluxes_and_channels` for the same fields; the
    budget's buoyancy terms are taken from it. Shaped for
    `rmhdgpu.diagnostics.budget.flatten_conserved_quantity_budgets`. Buoyancy
    is the only ideal term that changes W±, and the run driver adds forcing
    itself.
    """

    z_plus_hat, z_minus_hat = elsasser_potentials(fields.phi_hat, fields.psi_hat)

    budgets: dict[str, dict[str, Any]] = {
        "w_plus": {
            "value": elsasser_energy(z_plus_hat, grid, backend),
            "rhs_terms": {"buoyancy": fluxes["w_plus_buoyancy"]},
        },
        "w_minus": {
            "value": elsasser_energy(z_minus_hat, grid, backend),
            "rhs_terms": {"buoyancy": fluxes["w_minus_buoyancy"]},
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
    estimates are built from `w_plus`, `k_perp_plus` and `w_plus_align` in the
    same row, so measurement and prediction are taken at the same instant.
    """

    fluxes = fluxes_and_channels(fields, grid, backend, p)
    z_plus_hat, z_minus_hat = elsasser_potentials(fields.phi_hat, fields.psi_hat)

    return {
        "w_plus": elsasser_energy(z_plus_hat, grid, backend),
        "w_minus": elsasser_energy(z_minus_hat, grid, backend),
        "k_perp_plus": elsasser_kperp_mean(z_plus_hat, grid, backend),
        "k_perp_minus": elsasser_kperp_mean(z_minus_hat, grid, backend),
        "k_prl_plus": elsasser_kprl_mean(z_plus_hat, grid, backend),
        "k_prl_minus": elsasser_kprl_mean(z_minus_hat, grid, backend),
        "w_plus_align": elsasser_x_alignment(z_plus_hat, grid, backend),
        "w_minus_align": elsasser_x_alignment(z_minus_hat, grid, backend),
        "u_perp_rms": perp_gradient_rms(fields.phi_hat, grid, backend),
        "b_perp_rms": perp_gradient_rms(fields.psi_hat, grid, backend),
        # Only the x component is saved: every background gradient points
        # along x, so the closure predicts V_rho_y ~ 0 and V_rho_x ~ eta_turb F_rho.
        "V_rho_x": fluxes["V_rho_x"],
        "q_dcf": -fluxes["w_plus_buoyancy"],
        "q_ccr_source": -fluxes["w_minus_buoyancy"],
        "N_sq": float(p.N_sq),
        "vA": float(p.vA),
        "acr_B": fluxes["acr_B"],
        "acr_g": fluxes["acr_g"],
        "acr_th": fluxes["acr_th"],
    }


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
