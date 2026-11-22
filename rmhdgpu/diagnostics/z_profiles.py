"""Perpendicular-average energy profiles <.>_perp(z) for the inhomogeneous RMHD sets.

The scalar budgets are box means, and in a periodic box every term of the form
d_z(...) averages to zero there. The ones this loses are the parallel
transport terms in

    d_t W+(z) = -vA d_z W+(z) + (g/2) <drho z+_x>_perp
    d_t W-(z) = +vA d_z W-(z) + (g/2) <drho z-_x>_perp
    d_t (W_par^u + W_par^B + W^p)(z) = -d_z F_slow(z) + drives(z),
        F_slow = -vA^2 <du_par db_par>_perp

These hold at every z, not just on average. In the ideal equations, every
nonlinear term is a perpendicular divergence: <z+ . (z- . grad z+)>_perp
= <z- . grad |z+|^2/2>_perp = 0 because div_perp z- = 0, and likewise for the
pressure and for the Poisson brackets in the du_par / db_par / s equations. So
the profiles below are enough to close the local budget, up to dissipation
and forcing.

With the run's right-hand side the module also writes a closed local budget
for every one of the six energies: `<energy>_dt` is d_t W(z) from the full
right-hand side (ideal terms plus damping, not forcing), and
`<energy>_<term>` are its named terms (`Z_PROFILE_LOCAL_BUDGETS`). So at each
saved time and each z

    <energy>_dt = sum of its terms

up to the small part of the nonlinear terms (advection, and the W± cascade
terms) that the 2/3-rule truncation leaves at each z; those vanish exactly
only after averaging over z. Forcing is a kick, not part of the right-hand
side, so it is in neither side.

The local terms are those of the box budgets plus the transport terms
(`w_plus_transport`, `w_minus_transport`). The linear exchange vA^2 <du d_z db>
is not a pure z-derivative for one energy alone, so it stays an exchange
term; only the sum over the three slow-mode energies is -d_z F_slow.

Each profile is an average over x and y of a product of real fields, so the
fields are taken to real space (one inverse FFT each). That is too costly to
do every step, so these are written only with the spectra, to z_profiles.csv.
The z-average of every profile equals the matching box mean in
scalar_diagnostics.csv.

Conventions as in `compressive_channels`: u_x = -dy(phi), b_x = -dy(psi),
z± = zhat x grad_perp(phi ∓ psi), W± = <|z±|^2>/4, and db_par = dB_par/B_0.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rmhdgpu.diagnostics.compressive_channels import ChannelFields, elsasser_potentials
from rmhdgpu.operators import dx, dy, dz


# Quantities written to z_profiles.csv, one row per (time, quantity, z).
Z_PROFILE_INFO = {
    "w_plus": "W+(z) = <|z+|^2>_perp / 4.",
    "w_minus": "W-(z) = <|z-|^2>_perp / 4.",
    "w_plus_buoyancy": "(g/2) <drho z+_x>_perp: buoyancy work on W+ at each z.",
    "w_minus_buoyancy": "(g/2) <drho z-_x>_perp: buoyancy work on W- at each z.",
    "w_par_u": "W_par^u(z) = <du_par^2>_perp / 2.",
    "w_par_b": "W_par^B(z) = vA^2 <db_par^2>_perp / 2.",
    "w_p": "W^p(z) = (vA^4/cs^2) <db_par^2>_perp / 2 = w_par_b / chi.",
    "w_s": "W_s(z) = cs^2/(gamma^2 (gamma-1)) <s^2>_perp / 2.",
    "slow_flux": (
        "F_slow(z) = -vA^2 <du_par db_par>_perp: parallel energy flux of the slow-mode "
        "reservoirs, d_t (w_par_u + w_par_b + w_p) = -d_z F_slow + drives."
    ),
    "w_par_u_drive_KB": "vA K_b0 <du_par b_x>_perp: background-field drive of w_par_u at each z.",
    "w_par_bp_drive_KBKp": (
        "-vA^2 (K_b0 - K_p0/gamma) <db_par u_x>_perp: background drive of w_par_b + w_p "
        "(split alpha : 1 - alpha) at each z."
    ),
    "w_s_drive_Ks": "-cs^2/(gamma^2 (gamma-1)) K_s <s u_x>_perp: background drive of w_s at each z.",
}

# Written only when the run has dissipation operators (`damping_rates` given).
Z_PROFILE_DISSIPATION_INFO = {
    "w_plus_dissipation": "Damping of W+ at each z: <grad_perp f+ . grad_perp f+_t>_perp / 2, f+_t from damping alone.",
    "w_minus_dissipation": "Damping of W- at each z, as w_plus_dissipation.",
    "slow_dissipation": "Damping of w_par_u + w_par_b + w_p at each z: <du_par du_t> + (vA^2/alpha) <db_par db_t>.",
    "w_s_dissipation": "Damping of W_s at each z: W_s-weight <s s_t>, s_t from damping alone.",
}


# The closed local budget of each energy: its d_t W(z) profile and term profiles.
# Written only when the run hands over its right-hand side (`rates`, `brackets`).
Z_PROFILE_LOCAL_BUDGETS = {
    "w_plus": ("transport", "buoyancy", "dissipation"),
    "w_minus": ("transport", "buoyancy", "dissipation"),
    "w_par_u": ("drive_KB", "exchange_lin", "exchange_nl", "dissipation"),
    "w_par_b": ("drive_KBKp", "exchange_lin", "exchange_nl", "dissipation"),
    "w_p": ("drive_KBKp", "exchange_lin", "exchange_nl", "dissipation"),
    "w_s": ("drive_Ks", "dissipation"),
}
Z_PROFILE_LOCAL_BUDGET_INFO = {
    **{f"{energy}_dt": f"d_t {energy}(z) from the full right-hand side (ideal + damping, no forcing)."
       for energy in Z_PROFILE_LOCAL_BUDGETS},
    "w_plus_transport": "-vA d_z W+(z): parallel transport of W+ (zero on average over z).",
    "w_minus_transport": "+vA d_z W-(z): parallel transport of W- (zero on average over z).",
    "w_par_u_exchange_lin": "vA^2 <du_par d_z db_par>_perp.",
    "w_par_u_exchange_nl": "vA <du_par {psi, db_par}>_perp.",
    "w_par_u_dissipation": "<du_par du_t>_perp, du_t from damping alone.",
    "w_par_b_drive_KBKp": "-vS^2 (K_b0 - K_p0/gamma) <db_par u_x>_perp.",
    "w_par_b_exchange_lin": "vS^2 <db_par d_z du_par>_perp.",
    "w_par_b_exchange_nl": "alpha vA <db_par {psi, du_par}>_perp.",
    "w_par_b_dissipation": "vA^2 <db_par db_t>_perp, db_t from damping alone.",
    "w_p_drive_KBKp": "w_par_b_drive_KBKp / chi.",
    "w_p_exchange_lin": "w_par_b_exchange_lin / chi.",
    "w_p_exchange_nl": "w_par_b_exchange_nl / chi.",
    "w_p_dissipation": "w_par_b_dissipation / chi.",
}


def energy_z_profiles(
    fields: ChannelFields,
    grid: Any,
    fft: Any,
    backend: Any,
    p: Any,
    damping_rates: dict[str, Any] | None = None,
    rates: dict[str, Any] | None = None,
    brackets: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """Return the `Z_PROFILE_INFO` profiles plus the coordinate `z`, as host arrays.

    Uses 12 inverse FFTs: two components each of z+ and z-, plus du_par,
    db_par, s and drho. u_x and b_x come free from z±_x.

    `damping_rates` maps `phi`, `psi`, `du_par`, `db_par` and `s` to the rate
    each changes at under damping alone (e.g. `-D_omega phi_hat`). With it, the
    `Z_PROFILE_DISSIPATION_INFO` profiles are added, for 7 more inverse FFTs.

    `rates` maps the same five names to their full rate of change (ideal
    right-hand side plus damping), and `brackets` holds `{psi, db_par}` and
    `{psi, du_par}` as `psi_db` and `psi_du`. With both (and `damping_rates`),
    the closed local budgets `Z_PROFILE_LOCAL_BUDGET_INFO` are added too.
    """

    xp = backend.xp

    def real(f_hat: Any) -> Any:
        return fft.c2r(f_hat)

    def perp_mean(a: Any, b: Any) -> Any:
        """<a b>_perp(z): average over x and y, leaving the z dependence."""
        return xp.mean(a * b, axis=(0, 1))

    z_plus_hat, z_minus_hat = elsasser_potentials(fields.phi_hat, fields.psi_hat)
    # z = zhat x grad_perp(f), so z_x = -dy(f) and z_y = dx(f).
    zp_x, zp_y = real(-dy(z_plus_hat, grid)), real(dx(z_plus_hat, grid))
    zm_x, zm_y = real(-dy(z_minus_hat, grid)), real(dx(z_minus_hat, grid))
    u_x = 0.5 * (zp_x + zm_x)
    b_x = 0.5 * (zm_x - zp_x)

    du = real(fields.du_par_hat)
    db = real(fields.db_par_hat)
    s = real(fields.s_hat)
    drho = real(fields.drho_hat)

    s_weight = p.cs2 / (p.gamma**2 * (p.gamma - 1.0))
    w_par_b = 0.5 * p.vA**2 * perp_mean(db, db)
    profiles = {
        "w_plus": 0.25 * (perp_mean(zp_x, zp_x) + perp_mean(zp_y, zp_y)),
        "w_minus": 0.25 * (perp_mean(zm_x, zm_x) + perp_mean(zm_y, zm_y)),
        "w_plus_buoyancy": 0.5 * p.g * perp_mean(drho, zp_x),
        "w_minus_buoyancy": 0.5 * p.g * perp_mean(drho, zm_x),
        "w_par_u": 0.5 * perp_mean(du, du),
        "w_par_b": w_par_b,
        "w_p": w_par_b / p.chi,
        "w_s": 0.5 * s_weight * perp_mean(s, s),
        "slow_flux": -p.vA**2 * perp_mean(du, db),
        "w_par_u_drive_KB": p.vA * p.K_b0 * perp_mean(du, b_x),
        "w_par_bp_drive_KBKp": -p.vA**2 * (p.K_b0 - p.K_p0 / p.gamma) * perp_mean(db, u_x),
        "w_s_drive_Ks": -s_weight * p.K_s * perp_mean(s, u_x),
    }
    if damping_rates is not None:
        plus_t, minus_t = elsasser_potentials(damping_rates["phi"], damping_rates["psi"])
        for name, z_x, z_y, f_t in (("w_plus", zp_x, zp_y, plus_t), ("w_minus", zm_x, zm_y, minus_t)):
            # d_t |grad_perp f|^2 / 4 = grad_perp f . grad_perp f_t / 2, with grad_perp f = (z_y, -z_x).
            profiles[f"{name}_dissipation"] = 0.5 * (
                perp_mean(z_y, real(dx(f_t, grid))) - perp_mean(z_x, real(dy(f_t, grid)))
            )
        profiles["slow_dissipation"] = perp_mean(du, real(damping_rates["du_par"])) + (
            p.vA**2 / p.alpha * perp_mean(db, real(damping_rates["db_par"]))
        )
        profiles["w_s_dissipation"] = s_weight * perp_mean(s, real(damping_rates["s"]))

    local = damping_rates is not None and rates is not None and brackets is not None
    if local:
        # d_t W(z) of every energy, from the full rate of change.
        plus_t, minus_t = elsasser_potentials(rates["phi"], rates["psi"])
        for name, z_x, z_y, f_t in (("w_plus", zp_x, zp_y, plus_t), ("w_minus", zm_x, zm_y, minus_t)):
            profiles[f"{name}_dt"] = 0.5 * (
                perp_mean(z_y, real(dx(f_t, grid))) - perp_mean(z_x, real(dy(f_t, grid)))
            )
        db_rate = perp_mean(db, real(rates["db_par"]))
        profiles["w_par_u_dt"] = perp_mean(du, real(rates["du_par"]))
        profiles["w_par_b_dt"] = p.vA**2 * db_rate
        profiles["w_p_dt"] = p.vA**2 / p.chi * db_rate
        profiles["w_s_dt"] = s_weight * perp_mean(s, real(rates["s"]))

        # Named terms of the slow-mode energies, one energy at a time.
        profiles["w_par_u_exchange_lin"] = p.vA**2 * perp_mean(du, real(dz(fields.db_par_hat, grid)))
        profiles["w_par_u_exchange_nl"] = p.vA * perp_mean(du, real(brackets["psi_db"]))
        profiles["w_par_u_dissipation"] = perp_mean(du, real(damping_rates["du_par"]))
        # vA^2 db_par times each db_par_t term: alpha d_z du, (alpha/vA){psi, du},
        # alpha (K_b0 - K_p0/gamma) dy(phi), and damping.
        b_terms = {
            "drive_KBKp": -p.alpha * p.vA**2 * (p.K_b0 - p.K_p0 / p.gamma) * perp_mean(db, u_x),
            "exchange_lin": p.alpha * p.vA**2 * perp_mean(db, real(dz(fields.du_par_hat, grid))),
            "exchange_nl": p.alpha * p.vA * perp_mean(db, real(brackets["psi_du"])),
            "dissipation": p.vA**2 * perp_mean(db, real(damping_rates["db_par"])),
        }
        for term, value in b_terms.items():
            profiles[f"w_par_b_{term}"] = value
            profiles[f"w_p_{term}"] = value / p.chi

    result = {"z": backend.to_numpy(grid.z).astype(np.float64)}
    result.update({name: backend.to_numpy(value).astype(np.float64) for name, value in profiles.items()})
    if local:
        # Transport -/+ vA d_z W±(z): spectral derivative of the periodic profile.
        k = 2.0 * np.pi * np.fft.rfftfreq(grid.Nz, d=grid.Lz / grid.Nz)

        def d_dz(profile: np.ndarray) -> np.ndarray:
            return np.fft.irfft(1j * k * np.fft.rfft(profile), n=grid.Nz)

        result["w_plus_transport"] = -p.vA * d_dz(result["w_plus"])
        result["w_minus_transport"] = p.vA * d_dz(result["w_minus"])
    return result
