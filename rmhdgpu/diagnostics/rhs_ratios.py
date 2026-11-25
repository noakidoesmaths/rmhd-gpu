"""Size of each nonlinear term against the linear terms of the same equation.

Shared by `rmhd_by_nokia_rho` and `rmhd_by_nokia_s`. Every column is an rms
over the box, ||f|| = <|f|^2>^(1/2), of one term on the right-hand side,
evaluated at the scalar-output time. The terms are the ones the solver itself
uses (same Poisson brackets, same dealiasing mask), so no full-field output is
needed to build the ratios afterwards.

Why rms and not a box average: the advection terms conserve each field's
energy, so e.g. <z- . (z+ . grad z-)> = 0 identically. The rms is not zero and
measures how fast the term changes the field: if d_t f = X then X changes f
at the fractional rate ||X|| / ||f||. For a single mode exp(i k_z z) the
linear term gives exactly ||vA d_z f|| / ||f|| = vA |k_z|.

Elsasser fields (see `compressive_channels`):

    z± = zhat x grad_perp(zeta±),   zeta± = phi ∓ psi
    d_t z± ± vA d_z z± = -P[z∓ . grad_perp z±] - P[buoyancy] + ...

P removes the irrotational part, which the pressure cancels. Taking the curl
gives the vorticity form evolved by the code, d_t lap_perp(zeta±) = ... + N±:

    N± = -{phi, omega} + {psi, lap_perp psi} ± lap_perp {phi, psi}

N+ vanishes when z- = 0 (phi = -psi) and N- vanishes when z+ = 0 (phi = psi):
z+ is only distorted by z-, and z- only by z+. In z units (physical
amplitudes, not vorticities) the curl is undone by a factor 1/k_perp:

    ||P[z∓ . grad z±]||^2 = sum_k |N±_k|^2 / k_perp^2
    ||vA d_z z±||^2       = sum_k vA^2 k_z^2 k_perp^2 |zeta±_k|^2

and the ratio of the two is

    chi_rhs_± = ||P[z∓ . grad z±]|| / ||vA d_z z±||

(||z±|| cancels). The `_kzn0` variant keeps only the k_z != 0 modes of the
nonlinear term: the k_z = 0 part of z± has no Alfven frequency, so it inflates
chi without being about critical balance.

Entropy, s = delta s / c_v, is advected by u_perp and driven by the background
entropy gradient (both equation sets; in the rho set s is derived from drho
and db_par and obeys the same equation):

    d_t s = K_s dy(phi) - {phi, s},     {phi, s} = u_perp . grad s

There is no propagation term, so `chi_s_drive = ||{phi, s}|| / ||K_s dy phi||`
is a mixing-length ratio, roughly s / (K_s l_perp), not a critical-balance
parameter. `chi_s_prl = ||{phi, s}|| / ||vA d_z s||` uses the parallel
gradient of s instead, which a passive scalar inherits from the flow.

Ratios whose denominator is zero (e.g. K_s = 0, or a field with no k_z != 0
modes) are written as NaN. A field that is zero only up to round-off (z- in a
one-wave run with g = 0) gives a ratio of two round-off numbers, which can be
O(1) and means nothing: check the `rms_lin_*` column before reading a ratio.
"""

from __future__ import annotations

import math
from typing import Any

from rmhdgpu.diagnostics.compressive_channels import ChannelFields, elsasser_potentials
from rmhdgpu.fourier_diagnostics import modal_average
from rmhdgpu.operators import lap_perp, poisson_bracket


# Columns written by `rhs_ratio_diagnostics`, in the order they are written.
RHS_RATIO_SCALAR_DIAGNOSTIC_INFO = {
    "rms_nl_z_plus": "||P[z- . grad z+]||: rms of the nonlinear term in the z+ equation, in z units.",
    "rms_nl_z_plus_kzn0": "rms_nl_z_plus restricted to the k_z != 0 modes of the nonlinear term.",
    "rms_lin_z_plus": "||vA d_z z+||: rms of the linear propagation term in the z+ equation.",
    "chi_rhs_plus": "rms_nl_z_plus / rms_lin_z_plus: chi for the decorrelation of z+ by z-.",
    "chi_rhs_plus_kzn0": "rms_nl_z_plus_kzn0 / rms_lin_z_plus.",
    "rms_nl_z_minus": "||P[z+ . grad z-]||: rms of the nonlinear term in the z- equation, in z units.",
    "rms_nl_z_minus_kzn0": "rms_nl_z_minus restricted to the k_z != 0 modes of the nonlinear term.",
    "rms_lin_z_minus": "||vA d_z z-||: rms of the linear propagation term in the z- equation.",
    "chi_rhs_minus": "rms_nl_z_minus / rms_lin_z_minus: chi for the decorrelation of z- by z+.",
    "chi_rhs_minus_kzn0": "rms_nl_z_minus_kzn0 / rms_lin_z_minus.",
    "rms_buoy_z": "||P[g dy(drho) term]||: rms of the buoyancy term, the same in the z+ and z- equations, in z units.",
    "rms_nl_s": "||{phi, s}|| = ||u_perp . grad s||: rms of the entropy advection term, s = delta s/c_v.",
    "rms_drive_s": "||K_s dy(phi)||: rms of the background-gradient drive of s (= |K_s| rms(u_x)).",
    "rms_prl_s": "||vA d_z s||: rms parallel gradient of s, in rate units.",
    "chi_s_drive": "rms_nl_s / rms_drive_s: mixing-length ratio, roughly s / (K_s l_perp).",
    "chi_s_prl": "rms_nl_s / rms_prl_s: chi for s using its own parallel gradient.",
}


def _rms(density_hat: Any, grid: Any, backend: Any) -> float:
    """Return `sqrt(<f^2>)` from the modal density `|f_hat|^2` (Parseval)."""

    return math.sqrt(max(modal_average(density_hat, grid, backend), 0.0))


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0.0 else math.nan


def rhs_ratio_diagnostics(
    fields: ChannelFields,
    grid: Any,
    fft: Any,
    backend: Any,
    workspace: Any,
    p: Any,
    dealias_mask: Any | None = None,
) -> dict[str, float]:
    """Return the rms of the nonlinear, linear and drive terms and their ratios.

    Costs four Poisson brackets (about one RHS evaluation), so it is only
    evaluated at the scalar-output cadence. `p` must provide `vA`, `g` and
    `K_s`. Pass the solver's `dealias_mask` so the nonlinear terms are exactly
    the ones the time stepper uses; without it the rms also counts the
    aliased modes the solver discards.
    """

    xp = backend.xp

    def bracket(f_hat: Any, g_hat: Any) -> Any:
        # poisson_bracket returns a shared workspace buffer, so copy each result.
        return poisson_bracket(f_hat, g_hat, grid, fft, workspace, mask=dealias_mask).copy()

    phi_hat = fields.phi_hat
    psi_hat = fields.psi_hat
    omega_hat = lap_perp(phi_hat, grid)
    lap_psi_hat = lap_perp(psi_hat, grid)

    # Nonlinear parts of d_t lap_perp(zeta±): N± = common ± lap_perp{phi, psi}.
    common_hat = bracket(psi_hat, lap_psi_hat) - bracket(phi_hat, omega_hat)
    lap_bracket_hat = lap_perp(bracket(phi_hat, psi_hat), grid)
    zeta_plus_hat, zeta_minus_hat = elsasser_potentials(phi_hat, psi_hat)
    branches = (
        ("plus", common_hat + lap_bracket_hat, zeta_plus_hat),
        ("minus", common_hat - lap_bracket_hat, zeta_minus_hat),
    )

    kzn0 = grid.kpar2 > 0.0
    diagnostics: dict[str, float] = {}
    for branch, nl_hat, zeta_hat in branches:
        nl_density = grid.inv_kperp2 * xp.abs(nl_hat) ** 2
        rms_nl = _rms(nl_density, grid, backend)
        rms_nl_kzn0 = _rms(nl_density * kzn0, grid, backend)
        rms_lin = abs(p.vA) * _rms(grid.kpar2 * grid.kperp2 * xp.abs(zeta_hat) ** 2, grid, backend)
        diagnostics[f"rms_nl_z_{branch}"] = rms_nl
        diagnostics[f"rms_nl_z_{branch}_kzn0"] = rms_nl_kzn0
        diagnostics[f"rms_lin_z_{branch}"] = rms_lin
        diagnostics[f"chi_rhs_{branch}"] = _ratio(rms_nl, rms_lin)
        diagnostics[f"chi_rhs_{branch}_kzn0"] = _ratio(rms_nl_kzn0, rms_lin)

    # Buoyancy, -g dy(drho) in d_t omega, enters both vorticity equations unchanged.
    # In z units: |g k_y drho_k| / k_perp.
    diagnostics["rms_buoy_z"] = abs(p.g) * _rms(
        grid.inv_kperp2 * grid.ky**2 * xp.abs(fields.drho_hat) ** 2, grid, backend
    )

    s_hat = fields.s_hat
    rms_nl_s = _rms(xp.abs(bracket(phi_hat, s_hat)) ** 2, grid, backend)
    rms_drive_s = abs(p.K_s) * _rms(grid.ky**2 * xp.abs(phi_hat) ** 2, grid, backend)
    rms_prl_s = abs(p.vA) * _rms(grid.kpar2 * xp.abs(s_hat) ** 2, grid, backend)
    diagnostics["rms_nl_s"] = rms_nl_s
    diagnostics["rms_drive_s"] = rms_drive_s
    diagnostics["rms_prl_s"] = rms_prl_s
    diagnostics["chi_s_drive"] = _ratio(rms_nl_s, rms_drive_s)
    diagnostics["chi_s_prl"] = _ratio(rms_nl_s, rms_prl_s)
    return diagnostics
