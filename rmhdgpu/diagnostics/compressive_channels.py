"""Compressive-coupling channel diagnostics for the inhomogeneous RMHD sets.

This module is shared by `rmhdgpu.equations.rmhd_by_nokia_rho` (which evolves
`drho`) and `rmhdgpu.equations.rmhd_by_nokia_s` (which evolves `s` and derives
`drho` from it), so both equation sets write identical scalar-diagnostic
columns and can be read by one plotting script.

Naming follows Squire et al., *A Transport Theory of Turbulent Coronal Heating
in General Geometry* (arXiv:2607.08036):

- `DCF` (direct compressive feedback, their §III.4.1) is the work the buoyancy
  coupling does on the outward Elsasser energy `W+`. It is exact here: no
  closure is involved, because these equation sets are the straight-field
  (`kappa = 0`), no-mean-flow (`U = 0`) special case of their Eq. 18.
- `ACR` (Alfven-catalyzed relaxation, their §III.5) is the exchange with the
  background free-energy reservoirs, split into their `Y_B`, `Y_g` and `Y_th`
  channels. The sum is the `stratification` source the equation modules
  already computed as a single number.
- `CCR` (compressively catalyzed reflection, their §III.4.2) has no exact
  budget term. The Alfvenic nonlinearity conserves `W+` and `W-` separately,
  so the only Alfvenic/compressive exchange in these equations is the single
  buoyancy coupling `-g dy(drho)` in `omega_t`. CCR appears instead as the
  enhanced `W+` dissipation caused by the `W-` that buoyancy generates, so it
  is measured from `w_minus` together with the `w_plus` dissipation term.

Everything is written in the `(drho, db_par, du_par)` variables. The `s`
equation set reaches them through its own `derive_drho_hat`, and the two forms
are algebraically identical; `test_compressive_channels.py` asserts that
numerically against each module's own `total_energy_stratification_rhs`.

Sign convention matches `rmhdgpu.diagnostics.budget`: every value returned as
an RHS term is a signed contribution to `d_t Q`. In particular the DCF heating
rate is the *negative* of the `w_plus` buoyancy contribution,

    Q_DCF = -(d_t W+)|_buoyancy,

so `Q_DCF > 0` means the buoyancy coupling is damping the outward wave.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rmhdgpu.fourier_diagnostics import modal_average, modal_inner_product_average
from rmhdgpu.operators import dy, inv_lap_perp


CHANNEL_SCALAR_DIAGNOSTIC_INFO = {
    "w_plus": "Elsasser energy of z^+ = u_perp - b_perp: 0.25 <|grad_perp(phi - psi)|^2>.",
    "w_minus": "Elsasser energy of z^- = u_perp + b_perp: 0.25 <|grad_perp(phi + psi)|^2>.",
    "w_plus_kperp": "Energy-weighted <k_perp> of z+; the closure outer scale is l_perp = 1 / <k_perp>.",
    "w_minus_kperp": "Energy-weighted <k_perp> of z-.",
    "w_plus_kprl": "Energy-weighted <k_par> of z+; chi_A = 2 sqrt(W+) <k_perp> / (vA <k_par>).",
    "w_minus_kprl": "Energy-weighted <k_par> of z-.",
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
    """The Fourier fields the channel diagnostics are built from.

    Each equation module supplies this in its own variables; only `drho_hat`
    differs between the two inhomogeneous sets. The compressive fields default
    to `None` so the same container can hold an RHS state, for which only the
    Alfvenic potentials are needed.
    """

    phi_hat: Any
    psi_hat: Any
    du_par_hat: Any = None
    db_par_hat: Any = None
    drho_hat: Any = None


def elsasser_potential(fields: ChannelFields, *, sign: int) -> Any:
    """Return `phi -+ psi`, the potential of `z^± = b_hat x grad_perp(phi -+ psi)`.

    The convention used throughout the solver is

        `z^± = u_perp -+ b_perp/sqrt(4 pi rho0)`,

    so `sign > 0` selects `phi - psi`, which is the branch obeying
    `d_t f + vA d_z f = 0` and therefore propagates along `+b_hat`.
    """

    return fields.phi_hat - fields.psi_hat if sign > 0 else fields.phi_hat + fields.psi_hat


def elsasser_modal_density(fields: ChannelFields, grid: Any, backend: Any, *, sign: int) -> Any:
    """Return the modal density of `W^± = 0.25 <|grad_perp(phi -+ psi)|^2>`."""

    f_hat = elsasser_potential(fields, sign=sign)
    return 0.25 * grid.kperp2 * (backend.xp.abs(f_hat) ** 2)


def elsasser_energy(fields: ChannelFields, grid: Any, backend: Any, *, sign: int) -> float:
    """Return the volume-averaged Elsasser energy `W^±`.

    `W+ + W-` equals the equation modules' `alfvenic_energy`, since the cross
    terms cancel.
    """

    density = elsasser_modal_density(fields, grid, backend, sign=sign)
    return modal_average(density, grid, backend)


def _elsasser_wavenumber_mean(
    fields: ChannelFields,
    grid: Any,
    backend: Any,
    k_squared: Any,
    *,
    sign: int,
) -> float:
    """Return the `W^±`-weighted mean of `sqrt(k_squared)`.

    Forming the mean wavenumber as a ratio of modal averages avoids shell
    binning entirely and evaluates it on the same output cadence as the budget
    terms, so no interpolation between `t_out_spec` and `t_out_scal` is needed.
    Returns `0.0` for an empty field so callers can guard on it.
    """

    density = elsasser_modal_density(fields, grid, backend, sign=sign)
    total = modal_average(density, grid, backend)
    if not total > 0.0:
        return 0.0
    return modal_average(backend.xp.sqrt(k_squared) * density, grid, backend) / total


def elsasser_kperp_mean(fields: ChannelFields, grid: Any, backend: Any, *, sign: int) -> float:
    """Return the energy-weighted `<k_perp>` of `z^±`.

    The slaved closure's outer scale is `l_perp = 1 / <k_perp>`.
    """

    return _elsasser_wavenumber_mean(fields, grid, backend, grid.kperp2, sign=sign)


def elsasser_kprl_mean(fields: ChannelFields, grid: Any, backend: Any, *, sign: int) -> float:
    """Return the energy-weighted `<k_par>` of `z^±`.

    Needed only to monitor the turbulence-strength parameter

        chi_A = z l_par / (vA l_perp) = omega_nl / omega_A = z <k_perp> / (vA <k_par>),

    which the slaved closure assumes is of order unity. It cancels out of the
    heating predictions themselves. Note `chi_A` is meaningless if the `k_par =
    0` plane carries significant energy, since `omega_A = 0` there; use
    `exclude_kpar0` / `project_kpar0` for runs where `chi_A` matters.
    """

    return _elsasser_wavenumber_mean(fields, grid, backend, grid.kpar2, sign=sign)


def elsasser_energy_rhs_budget(
    fields: ChannelFields,
    rhs_fields: ChannelFields,
    grid: Any,
    backend: Any,
    *,
    sign: int,
) -> float:
    """Return the instantaneous `d_t W^±` along a supplied RHS.

    Mirrors `rmhdgpu.diagnostics.alfvenic.alfvenic_energy_rhs_budget`: pass the
    state fields and the fields of an RHS state, and get the directional
    derivative of `W^±` along it. Used by the tests to pin the closed-form
    buoyancy work against the actual equations.
    """

    xp = backend.xp
    f_hat = elsasser_potential(fields, sign=sign)
    f_t_hat = elsasser_potential(rhs_fields, sign=sign)
    density = 0.5 * grid.kperp2 * xp.real(xp.conj(f_hat) * f_t_hat)
    return modal_average(density, grid, backend)


def elsasser_dissipation_rhs(
    fields: ChannelFields,
    grid: Any,
    backend: Any,
    linear_ops: dict[str, Any],
    *,
    sign: int,
) -> float:
    """Return the signed dissipative contribution to `d_t W^±`.

    The diagonal damping gives `d_t phi = -D_omega phi` and `d_t psi = -D_psi
    psi` (the inverse perpendicular Laplacian commutes with a diagonal
    operator), so with `f± = phi -+ psi`

        d_t W^± = 0.5 kperp2 [ -D_om |phi|^2 - D_psi |psi|^2
                               +- (D_om + D_psi) Re(conj(phi) psi) ].

    The cross term cancels in `W+ + W-`, recovering the Alfvenic part of the
    modules' `total_energy_dissipation_rhs`. It vanishes identically when the
    `omega` and `psi` damping operators are equal.
    """

    xp = backend.xp
    damp_omega = linear_ops["omega"]
    damp_psi = linear_ops["psi"]
    cross_sign = 1.0 if sign > 0 else -1.0
    density = 0.5 * grid.kperp2 * (
        -damp_omega * (xp.abs(fields.phi_hat) ** 2)
        - damp_psi * (xp.abs(fields.psi_hat) ** 2)
        + cross_sign * (damp_omega + damp_psi) * xp.real(xp.conj(fields.phi_hat) * fields.psi_hat)
    )
    return modal_average(density, grid, backend)


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
    """Return the signed buoyancy contributions to `d_t W+` and `d_t W-`.

    Buoyancy enters only `omega_t`, so it drives `z+` and `z-` equally:

        d_t W^±|_buoy = -(g/2) <drho dy(phi)> +- (g/2) <drho dy(psi)>.

    Their sum is `-g <drho dy(phi)>`, which is exactly the `acr_g` channel: the
    gravity coupling is a direct exchange between background potential energy
    and the Alfvenic fluctuations.
    """

    half_g = 0.5 * p.g
    return {
        "w_plus": -half_g * (correlators["rho_phi"] - correlators["rho_psi"]),
        "w_minus": -half_g * (correlators["rho_phi"] + correlators["rho_psi"]),
    }


def stratification_channels(correlators: dict[str, float], p: Any) -> dict[str, float]:
    """Split the background-gradient source `Y` into the paper's ACR channels.

    The three returned terms sum to the modules' existing single
    `stratification` term, so they replace it in the total-energy budget
    without changing `total_energy_rhs_total`.

    Only `acr_B + acr_g` is net heating. `acr_th` is the extraction of
    background *thermal* free energy that is subsequently dissipated back into
    thermal energy (their `Y_th`), so counting it as heating double-counts.
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
) -> dict[str, dict[str, Any]]:
    """Return `w_plus` / `w_minus` conserved-quantity budgets.

    Shaped for `rmhdgpu.diagnostics.budget.flatten_conserved_quantity_budgets`,
    so the equation modules can merge this straight into their returned budget
    dictionary.
    """

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
    the instantaneous `w_plus` and `w_plus_kperp` in the same row. Comparing
    measurement to prediction wants both sides evaluated at the same instant.

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
        "q_dcf": -work["w_plus"],
        "q_ccr_source": -work["w_minus"],
        "N_sq": float(p.N_sq),
        "vA": float(p.vA),
    }
    diagnostics.update(stratification_channels(correlators, p))
    diagnostics.update({f"corr_{name}": value for name, value in correlators.items()})
    return diagnostics


def rhs_channel_fields(rhs_state: Any, grid: Any) -> ChannelFields:
    """Return the Alfvenic `ChannelFields` of an RHS state.

    `phi_t` is derived from `omega_t` the same way `phi` is derived from
    `omega`, so this can be fed to `elsasser_energy_rhs_budget` to get `d_t W^±`
    along any RHS (a full ideal RHS, or one holding a single isolated term).
    """

    return ChannelFields(
        phi_hat=inv_lap_perp(rhs_state["omega"], grid),
        psi_hat=rhs_state["psi"],
    )
