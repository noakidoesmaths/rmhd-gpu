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
from rmhdgpu.operators import dy, dz, poisson_bracket


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
    "k_prl_rms_plus": (
        "rms k_par of z+, sqrt(sum k_par^2 E+ / sum E+), so <|d_z z+|^2> = k_prl_rms_plus^2 <|z+|^2>. "
        "vA k_prl_rms_plus W+ is the size of the vA d_z W+ term; it is a scale estimate, not the "
        "z-profile gradient (that is in z_profiles.csv)."
    ),
    "k_prl_rms_minus": "rms k_par of z-, as k_prl_rms_plus.",
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
    "V_psi_x": (
        "x component of V_psi = <(delta B_par/B_0) delta u_perp> - <(delta u_par/vA) delta B_perp/B_0> "
        "(paper Eq. 79), = <db_par u_x> - <du_par b_x>/vA; the closure predicts "
        "V_psi_x ~ eta_turb (F_u + vS^2/vA^2 F_B) (Eq. 85). Defined for every K_b0, unlike -acr_B/(vA^2 K_b0)."
    ),
    "q_dcf": (
        "DCF heating rate -(d_t W+)|_buoyancy = -(g/2) <drho z+_x>, instantaneous; "
        "positive while buoyancy drains z+."
    ),
    "q_ccr_source": (
        "-(d_t W-)|_buoyancy = -(g/2) <drho z-_x>, instantaneous; negative while buoyancy "
        "creates z- (the CCR source)."
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
    equation set derives `drho_hat` from s and db_par, and the rho set derives
    `s_hat` from drho and db_par, so both fields are always available. `s_hat`
    is only needed by the energy-reservoir budgets.
    """

    phi_hat: Any
    psi_hat: Any
    du_par_hat: Any
    db_par_hat: Any
    drho_hat: Any
    s_hat: Any = None


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


def elsasser_kprl_rms(f_hat: Any, grid: Any, backend: Any) -> float:
    """Return the energy-weighted rms k_par of `z = zhat x grad_perp(f)`:

        k_rms = sqrt(sum_k k_par^2 E_k / sum_k E_k),    E_k = k_perp^2 |f_k|^2 / 4

    Since `<|d_z z|^2> = sum_k k_par^2 |z_k|^2`, this is the exact parallel
    gradient scale: `<|d_z z|^2> = k_rms^2 <|z|^2>`. The size of the `vA d_z W`
    term is then `vA k_rms W`. It is always >= `elsasser_kprl_mean`. An empty
    field returns 0.
    """

    E_k = 0.25 * grid.kperp2 * backend.xp.abs(f_hat) ** 2
    W = modal_average(E_k, grid, backend)
    if not W > 0.0:
        return 0.0
    return math.sqrt(max(modal_average(grid.kpar2 * E_k, grid, backend), 0.0) / W)


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
        acr_g          = g V_rho_x                                    Y_g
        acr_B          = -vA^2 K_b0 V_psi_x                           Y_B
        acr_th         = (vA^2 K_p0/gamma + w_s/chi) <db_par u_x> + w_s V_rho_x   Y_th

    with `w_s = cs^2 K_s / (gamma (gamma - 1))`. Since u_x = (z+_x + z-_x)/2,
    V_rho_x = V_rho_x_plus + V_rho_x_minus exactly. The paper's closure keeps
    only the z+ part, `V_rho ~ <z+ drho>/2`, so V_rho_x_plus is that estimate's
    left-hand side.

    The three ACR channels sum to the equation module's
    `total_energy_stratification_rhs`. acr_th is not net heating: it
    extracts background thermal free energy that dissipation later returns
    as heat, so do not count it twice. Nor is acr_B + acr_g on its own: the
    paper's Q_ACR (Eq. 78) also subtracts the pressure work
    V_psi . grad(p) = (g - vA^2 K_b0) V_psi_x, which is nonzero when
    K_p0 != 0, leaving Q_ACR = g (V_rho_x - V_psi_x). vis/plot_dcf_ccr_acr.py
    draws it.
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
        z_plus_diss, z_minus_diss = elsasser_potentials(
            -linear_ops["omega"] * fields.phi_hat,
            -linear_ops["psi"] * fields.psi_hat,
        )
        budgets["w_plus"]["rhs_terms"]["dissipation"] = elsasser_energy_rate(
            z_plus_hat, z_plus_diss, grid, backend,
        )
        budgets["w_minus"]["rhs_terms"]["dissipation"] = elsasser_energy_rate(
            z_minus_hat, z_minus_diss, grid, backend,
        )
    return budgets


# ---------------------------------------------------------------------------
# Compressive energy reservoirs: W_par^u, W_par^B, W^p, W_s
# ---------------------------------------------------------------------------

# The four reservoirs that, with W+ and W-, partition the total energy:
#
#     E = W+ + W- + W_par^u + W_par^B + W^p + W_s
#
#     W_par^u = <du_par^2>/2                      parallel kinetic energy
#     W_par^B = vA^2 <db_par^2>/2                 = rho0 vA^2 <(dB_par/B)^2>/2, magnetic pressure
#     W^p     = (vA^4/cs^2) <db_par^2>/2 = W_par^B/chi    thermal pressure (pressure balance)
#     W_s     = cs^2/(gamma^2 (gamma-1)) <s^2>/2  = p/(gamma(gamma-1)) <(ds/c_v)^2>/2 for rho0 = 1
#
# W^p is slaved to W_par^B by pressure balance, so its budget is W_par^B's
# times 1/chi; it is kept separate so the six reservoirs add up to E.
RESERVOIRS = ("w_par_u", "w_par_b", "w_p", "w_s")

RESERVOIR_SCALAR_DIAGNOSTIC_INFO = {
    "w_par_u": (
        "Parallel kinetic energy W_par^u = <du_par^2>/2. Budget terms: drive_KB = vA K_b0 <du_par b_x> "
        "(background field gradient), exchange_lin = vA^2 <du_par d_z db_par> and exchange_nl = "
        "vA <du_par {psi, db_par}> (both exchanged with w_par_b + w_p), dissipation, forcing."
    ),
    "w_par_b": (
        "Magnetic-pressure energy W_par^B = vA^2 <db_par^2>/2. Budget terms: exchange_lin = vS^2 <db_par d_z du_par>, "
        "exchange_nl = -alpha * w_par_u_rhs_exchange_nl, drive_KBKp = -vS^2 (K_b0 - K_p0/gamma) <db_par u_x>, "
        "dissipation, forcing."
    ),
    "w_p": (
        "Thermal-pressure energy W^p = (vA^4/cs^2) <db_par^2>/2 = w_par_b/chi. Its terms are w_par_b's times 1/chi, "
        "so w_par_b + w_p takes exactly what w_par_u gives up through exchange_*."
    ),
    "w_s": (
        "Entropy energy W_s = cs^2/(gamma^2 (gamma-1)) <s^2>/2. Budget terms: drive_Ks = -W_s-weight K_s <s u_x>, "
        "dissipation, forcing."
    ),
}


def reservoir_budgets(
    fields: ChannelFields,
    grid: Any,
    backend: Any,
    p: Any,
    *,
    fft: Any,
    workspace: Any,
    dealias_mask: Any | None = None,
    damping_rates: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return the `w_par_u`, `w_par_b`, `w_p` and `w_s` budgets, value plus signed RHS terms.

    Each term is `<weight * f * (d_t f)|_term>` for one term of the field's
    ideal equation, written with the physical weights in the table above (not
    the module's `dbpar_energy_weight`). The terms booked are every ideal term
    whose box average is nonzero:

        du_par_t = vA^2 d_z db_par + vA {psi, db_par} - vA K_b0 dy(psi) - {phi, du_par}
        db_par_t = alpha d_z du_par + (alpha/vA) {psi, du_par}
                   + alpha (K_b0 - K_p0/gamma) dy(phi) - {phi, db_par}
        s_t      = K_s dy(phi) - {phi, s}

    Advection `-{phi, f}` averages to zero against f and is not booked. The
    nonlinear exchange uses one Poisson bracket, `X = vA <du_par {psi, db_par}>`.
    The db_par one, `vA <db_par {psi, du_par}>`, is -X exactly for dealiased
    fields, so the db_par reservoirs book -X (split alpha : 1 - alpha between
    w_par_b and w_p). The exchange terms therefore cancel in the sum
    w_par_u + w_par_b + w_p, as do the linear `exchange_lin` terms after
    integration by parts in z.

    `damping_rates` maps `du_par`, `db_par` and `s` to the rate each field
    changes at under damping alone (for example `-D_u du_par_hat`); the rho
    equation set derives the `s` rate from the drho and db_par operators.
    With it, every reservoir gets a `dissipation` term.
    """

    def mean(a_hat: Any, b_hat: Any) -> float:
        """Box average <a b> of two real fields, from their Fourier coefficients."""
        return modal_inner_product_average(a_hat, b_hat, grid, backend)

    du = fields.du_par_hat
    db = fields.db_par_hat
    s = fields.s_hat
    u_x = -dy(fields.phi_hat, grid)
    b_x = -dy(fields.psi_hat, grid)

    # Fraction of the db_par energy (weight vA^2/alpha) in each reservoir:
    # vA^2 / (vA^2/alpha) = alpha for w_par_b, and 1 - alpha for w_p.
    frac_b = p.alpha
    frac_p = 1.0 - p.alpha
    db_weight = p.vA**2 / p.alpha  # = vA^2 + vA^4/cs^2
    s_weight = p.cs2 / (p.gamma**2 * (p.gamma - 1.0))

    # The one nonlinear exchange term. poisson_bracket returns the shared
    # workspace buffer, so it is reduced to a scalar straight away.
    bracket = poisson_bracket(fields.psi_hat, db, grid, fft, workspace, mask=dealias_mask)
    exchange_nl = p.vA * mean(du, bracket)

    # du_par equation, weighted by du_par.
    u_terms = {
        "drive_KB": p.vA * p.K_b0 * mean(du, b_x),
        "exchange_lin": p.vA**2 * mean(du, dz(db, grid)),
        "exchange_nl": exchange_nl,
    }
    # db_par equation, weighted by (vA^2/alpha) db_par: alpha * vA^2/alpha = vA^2.
    db_terms = {
        "exchange_lin": p.vA**2 * mean(db, dz(du, grid)),
        "exchange_nl": -exchange_nl,
        "drive_KBKp": -p.vA**2 * (p.K_b0 - p.K_p0 / p.gamma) * mean(db, u_x),
    }
    # s equation, weighted by s_weight s.
    s_terms = {"drive_Ks": -s_weight * p.K_s * mean(s, u_x)}

    if damping_rates is not None:
        u_terms["dissipation"] = mean(du, damping_rates["du_par"])
        db_terms["dissipation"] = db_weight * mean(db, damping_rates["db_par"])
        s_terms["dissipation"] = s_weight * mean(s, damping_rates["s"])

    values = reservoir_values(fields, grid, backend, p)
    return {
        "w_par_u": {"value": values["w_par_u"], "rhs_terms": u_terms},
        "w_par_b": {
            "value": values["w_par_b"],
            "rhs_terms": {name: frac_b * value for name, value in db_terms.items()},
        },
        "w_p": {
            "value": values["w_p"],
            "rhs_terms": {name: frac_p * value for name, value in db_terms.items()},
        },
        "w_s": {"value": values["w_s"], "rhs_terms": s_terms},
    }


def reservoir_values(fields: ChannelFields, grid: Any, backend: Any, p: Any) -> dict[str, float]:
    """Return the four compressive reservoir energies alone (no FFTs), for forcing kicks."""

    def mean(a_hat: Any, b_hat: Any) -> float:
        return modal_inner_product_average(a_hat, b_hat, grid, backend)

    db_energy = 0.5 * p.vA**2 / p.alpha * mean(fields.db_par_hat, fields.db_par_hat)
    return {
        "w_par_u": 0.5 * mean(fields.du_par_hat, fields.du_par_hat),
        "w_par_b": p.alpha * db_energy,
        "w_p": (1.0 - p.alpha) * db_energy,
        "w_s": 0.5 * p.cs2 / (p.gamma**2 * (p.gamma - 1.0)) * mean(fields.s_hat, fields.s_hat),
    }


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
        "k_prl_rms_plus": elsasser_kprl_rms(z_plus_hat, grid, backend),
        "k_prl_rms_minus": elsasser_kprl_rms(z_minus_hat, grid, backend),
        "w_plus_align": elsasser_x_alignment(z_plus_hat, grid, backend),
        "w_minus_align": elsasser_x_alignment(z_minus_hat, grid, backend),
        "u_perp_rms": perp_gradient_rms(fields.phi_hat, grid, backend),
        "b_perp_rms": perp_gradient_rms(fields.psi_hat, grid, backend),
        # Only the x component is saved: every background gradient points
        # along x, so the closure predicts V_rho_y ~ 0 and V_rho_x ~ eta_turb F_rho.
        "V_rho_x": fluxes["V_rho_x"],
        "V_psi_x": fluxes["V_psi_x"],
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


def psi_drive(p: Any) -> float:
    """Return the drive of the V_psi flux, F_u + (vS^2/vA^2) F_B (paper Eq. 85).

    The closure is V_psi ~ eta_turb (F_u + vS^2/vA^2 F_B), the counterpart of
    V_rho ~ eta_turb F_rho. In the units of `slaved_field_units`, slaving
    across a displacement xi_x gives du_par/vA = F_u xi_x and
    (vA^2/vS^2) db_par = F_B xi_x. For z+, b_x = -u_x, so

        V_psi_x = <db_par u_x> - <du_par b_x>/vA = (F_u + vS^2/vA^2 F_B) <xi_x u_x>

    while V_rho_x = F_rho <xi_x u_x>, so both closures share eta_turb ~ <xi_x u_x>.
    """

    drives = background_drives(p)
    return drives["du_par"] + p.alpha * drives["db_par"]  # alpha = vS^2/vA^2


def slaved_field_units(p: Any) -> dict[str, float]:
    """Return what to divide each saved compressive field by to get Eq. (47) units.

    drho stays delta rho/rho_0, du_par becomes delta u_par/vA, and db_par
    becomes (vA^2/vS^2) delta B_par/B_0, since alpha = vS^2/vA^2.
    """

    return {"drho": 1.0, "du_par": p.vA, "db_par": p.alpha}
