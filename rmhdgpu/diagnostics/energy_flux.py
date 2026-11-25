"""Scale-by-scale energy budget in k_perp shells: nonlinear transfer, linear terms, dissipation.

For every energy partition (the total energy, z+, z-, and per-field energies
where the equation set defines them), the rate of change of its shell
spectrum E(k_perp) is split by the term that causes it:

    d_t E(k) = T(k) + L(k) + D(k) + forcing

    T(k)  nonlinear transfer: the Poisson-bracket terms. They only move energy
          between modes, so sum_k T(k) = 0 for every partition the
          nonlinearity conserves (total, z+ and z-; in the compressive sets
          also the compressive energy as a whole, but not each field alone,
          because brackets like {psi, du_par} exchange energy between fields).
    L(k)  linear terms: background gradients, gravity and the wave terms
          d_z. Summed over k this is the linear energy drive (for the total
          energy, the total_energy_rhs_acr_* terms).
    D(k)  dissipation (negative).

Forcing kicks and the mode projections (`project_kx0`, `project_ky0`,
`project_kpar0`) are not included: they are applied between steps, not
through the RHS. Their box totals are the `total_energy_rhs_forcing` and
`total_energy_rhs_projection` columns of `scalar_diagnostics.csv`.

The energy flux through k_perp = K is

    Pi(K) = - sum_{k <= K} T(k),

the energy per unit time that the nonlinearity carries from the shells
below K to the shells above. A cascade shows up as Pi ~ constant > 0
between the injection and dissipation scales; Pi is computed from T in
`vis/plot_energy_flux.py`.

Shell values use the same binning and normalization as `spectra.csv`, so
sum_k T(k) is a volume-averaged rate, directly comparable with the
`total_energy_rhs_*` columns of `scalar_diagnostics.csv`.

How the rates are computed. Two exact identities do all the work, so no
equation set needs a hand-written transfer formula:

1. Splitting the RHS. `ideal_rhs(f) = L(f) + N(f)`, with L linear and N
   quadratic in the fields. Flipping the sign of every field flips L but
   not N, so

       N(f) = [ideal_rhs(f) + ideal_rhs(-f)] / 2
       L(f) = [ideal_rhs(f) - ideal_rhs(-f)] / 2

2. The rate of a quadratic energy. Each mode's energy e_k(f) is quadratic
   in the fields. If the fields change at rate R (d_t f = R), then

       d_t e_k = [e_k(f + R) - e_k(f - R)] / 2

   exactly: expanding both terms, e_k(f) and e_k(R) cancel and only the
   cross term, which is d_t e_k, is left. This is not a finite difference;
   R does not need to be small.

Identity 2 uses the equation set's own energy densities, so the weights
(e.g. the entropy energy of drho) and sign conventions always match
`total_energy` and `spectra.csv`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rmhdgpu.diagnostics.alfvenic import alfvenic_phi_hat
from rmhdgpu.diagnostics.spectra import perpendicular_shell_spectrum
from rmhdgpu.state import State

# The three kinds of term, as they appear in the `quantity` column of
# energy_flux.csv: e.g. `nonlinear_total`, `linear_z_plus`, `dissipation_drho`.
TERMS = ("nonlinear", "linear", "dissipation")


def energy_densities(state: State, grid: Any, backend: Any, params: Any, equation_module: Any) -> dict[str, Any]:
    """Modal energy densities of every partition: {name: density on the rfft grid}.

    Always "total" (the equation set's `total_energy_modal_density`). Equation
    sets with an `energy_modal_densities` hook add their own partitions
    (z_plus, z_minus, du_par, drho, ...). Otherwise, sets with psi and an
    Alfvenic velocity potential get the Elsasser energies

        e_z± = (1/4) k_perp^2 |phi ∓ psi|^2,   z± = z_hat x grad_perp(phi ∓ psi),

    the same convention as `spectra.elsasser_perpendicular_spectra`.
    """

    densities = {"total": equation_module.total_energy_modal_density(state, grid, backend, params)}
    if hasattr(equation_module, "energy_modal_densities"):
        densities.update(equation_module.energy_modal_densities(state, grid, backend, params))
    elif "psi" in state.field_names and ("omega" in state.field_names or "phi" in state.field_names):
        xp = backend.xp
        phi_hat = alfvenic_phi_hat(state, grid, equation_module)
        densities["z_plus"] = 0.25 * grid.kperp2 * xp.abs(phi_hat - state["psi"]) ** 2
        densities["z_minus"] = 0.25 * grid.kperp2 * xp.abs(phi_hat + state["psi"]) ** 2
    return densities


def shell_energy_rates(
    state: State,
    rate: State,
    *,
    grid: Any,
    backend: Any,
    params: Any,
    equation_module: Any,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """d_t of every partition's k_perp shell spectrum when the fields change at `rate`.

    Uses identity 2 of the module docstring: d_t e_k = [e_k(f + R) - e_k(f - R)] / 2.
    Returns (kperp shell centres, {partition name: rate in each shell}).
    """

    state_plus_rate = state.zeros_like().linear_combination_([(1.0, state), (1.0, rate)])
    state_minus_rate = state.zeros_like().linear_combination_([(1.0, state), (-1.0, rate)])
    energy_plus = energy_densities(state_plus_rate, grid, backend, params, equation_module)
    energy_minus = energy_densities(state_minus_rate, grid, backend, params, equation_module)

    rates = {}
    for name in energy_plus:
        kperp, rates[name] = perpendicular_shell_spectrum(0.5 * (energy_plus[name] - energy_minus[name]), grid, backend)
    return kperp, rates


def energy_flux_spectra(
    state: State,
    *,
    grid: Any,
    fft: Any,
    workspace: Any,
    backend: Any,
    params: Any,
    equation_module: Any,
    linear_ops: dict[str, Any],
    dealias_mask: Any | None = None,
) -> dict[str, np.ndarray]:
    """Shell-by-shell energy rates T(k), L(k) and D(k) of every partition.

    Returns {"kperp": shell centres, "<term>_<partition>": rate per shell}
    with <term> in TERMS, in the layout `SpectraDiagnosticsWriter` writes.
    Costs two `ideal_rhs` calls plus six energy-density evaluations, so it is
    meant for output times, not every step.
    """

    # Identity 1: split the ideal RHS into its nonlinear and linear parts.
    def ideal_rhs(fields: State) -> State:
        return equation_module.ideal_rhs(fields, grid, fft, workspace, params, dealias_mask=dealias_mask)

    minus_state = state.zeros_like().linear_combination_([(-1.0, state)])
    rhs_of_state = ideal_rhs(state)
    rhs_of_minus_state = ideal_rhs(minus_state)
    nonlinear_rhs = state.zeros_like().linear_combination_([(0.5, rhs_of_state), (0.5, rhs_of_minus_state)])
    linear_rhs = state.zeros_like().linear_combination_([(0.5, rhs_of_state), (-0.5, rhs_of_minus_state)])

    # Dissipation: each field is damped as d_t f = -linear_ops[f] * f, the same
    # operators the time stepper and the total_energy_rhs_dissipation budget use.
    dissipation_rhs = state.zeros_like()
    for name in state.field_names:
        dissipation_rhs[name] = -linear_ops[name] * state[name]

    spectra: dict[str, np.ndarray] = {}
    for term, rate in zip(TERMS, (nonlinear_rhs, linear_rhs, dissipation_rhs)):
        kperp, rates = shell_energy_rates(state, rate, grid=grid, backend=backend, params=params,
                                          equation_module=equation_module)
        spectra["kperp"] = kperp
        for partition, values in rates.items():
            spectra[f"{term}_{partition}"] = values
    return spectra
