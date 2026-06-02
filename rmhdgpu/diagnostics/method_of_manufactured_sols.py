"""Method-of-manufactured-solutions residual test for `rmhd_by_nokia_s`.

We feed a smooth, band-limited analytic state into `ideal_rhs` and compare the
result against the *true* RHS for that state, computed by an independent path
(sympy symbolic differentiation). The solver is pseudo-spectral, so every linear
operator (dz, dy, lap_perp, inv_lap_perp) is exact for band-limited fields, and
the only nonlinearity is the Poisson bracket. On a grid large enough to resolve
all nonlinear products (no aliasing), numerical and analytic RHS must agree to
machine precision. A wrong sign, coefficient, or missing term then shows up as a
large per-field error.

Why the reference is symbolic, not built from `rmhdgpu.operators`: the operators
there are the same FFT routines `ideal_rhs` uses, so reusing them would compare
the code against a copy of itself. The sympy path below is fully independent --
it differentiates the manufactured fields analytically and only samples them on
the grid at the very end. `rmhdgpu.operators.lap_perp` is used in one spot only,
to build the *input* `omega` from `phi` (input prep, not the thing under test).

Reference equations encoded (code-consistent form, gamma = 5/3):

    phi      = inv_lap_perp(omega)   <=>   omega = lap_perp(phi) = phi_xx + phi_yy
    drho     = -s/gamma - db_par/chi
    alpha    = chi/(1+chi),  K_b0 = g/vA^2 - chi*K_p0/gamma,  K_s = K_p0 - gamma*K_rho0

    psi_t    = vA*dz(phi) - {phi,psi}
    omega_t  = vA*dz(lap_perp psi) - {phi,omega} + {psi, lap_perp psi} - g*dy(drho)
    du_par_t = vA^2*dz(db_par) + vA*{psi,db_par} - {phi,du_par} - vA*K_b0*dy(psi)
    db_par_t = alpha*dz(du_par) + (alpha/vA)*{psi,du_par} - {phi,db_par}
               + alpha*K_b0*dy(phi) - alpha*(K_p0/gamma)*dy(phi)
    s_t      = K_s*dy(phi) - {phi,s}

Run:  python testing_folder/mms_rmhd_by_nokia_s.py
"""

from __future__ import annotations

import sys

import numpy as np
import sympy as sp

from rmhdgpu.backend import build_backend
from rmhdgpu.config import Config
from rmhdgpu.equations import rmhd_by_nokia_s as eqs
from rmhdgpu.fft import FFTManager
from rmhdgpu.grid import build_grid
from rmhdgpu.operators import lap_perp
from rmhdgpu.state import State
from rmhdgpu.workspace import Workspace

TOL = 1.0e-9

# Symbolic coordinates. Real-space fields are functions of these; sampling onto
# the grid happens only inside `evaluate_on_grid`.
X, Y, Z = sp.symbols("x y z", real=True)


# --- Symbolic mirror of rmhdgpu.operators (analytic, FFT-free) -------------
# These intentionally parallel dx/dy/dz/lap_perp/poisson_bracket in
# rmhdgpu.operators, but act on sympy expressions so the reference RHS is an
# independent analytic path rather than a re-use of the code under test.

def sym_dx(f: sp.Expr) -> sp.Expr:
    return sp.diff(f, X)


def sym_dy(f: sp.Expr) -> sp.Expr:
    return sp.diff(f, Y)


def sym_dz(f: sp.Expr) -> sp.Expr:
    return sp.diff(f, Z)


def sym_lap_perp(f: sp.Expr) -> sp.Expr:
    return sp.diff(f, X, 2) + sp.diff(f, Y, 2)


def sym_poisson_bracket(f: sp.Expr, g: sp.Expr) -> sp.Expr:
    """Analytic {f, g} = f_x g_y - f_y g_x."""

    return sym_dx(f) * sym_dy(g) - sym_dy(f) * sym_dx(g)


def build_context() -> tuple[Config, object, object, FFTManager, Workspace]:
    """Set up a 32^3 periodic 2*pi box with nonzero inhomogeneity parameters.

    The large grid (N/2 = 16) keeps products of modes up to index ~3 alias-free,
    and nonzero g/K_p0/K_rho0 exercise every inhomogeneous term in the RHS.
    """

    config = Config(
        equation_set="inhomogeneous_rmhd_rho",
        Nx=32,
        Ny=32,
        Nz=32,
        Lx=2.0 * np.pi,
        Ly=2.0 * np.pi,
        Lz=2.0 * np.pi,
        backend="numpy",
        vA=1.0,
        cs2_over_vA2=3.8,
        g=3.8,
        K_p0=-1.4,
        K_rho0=-3.3,
    )
    backend = build_backend(config)
    grid = build_grid(config, backend)
    fft = FFTManager(grid, backend)
    workspace = Workspace(grid, backend)
    return config, backend, grid, fft, workspace


def manufactured_solution() -> dict[str, sp.Expr]:
    """Return smooth, band-limited sympy fields psi, phi, du_par, db_par, s.

    All trig modes use integer wavenumbers (matching the 2*pi box) with index
    magnitude <= 3 so products stay alias-free on a 32^3 grid. `phi` deliberately
    contains no (kx=0, ky=0) mode: every term depends on x or y, so omega =
    lap_perp(phi) inverts back to phi exactly through inv_lap_perp.
    """

    return {
        "phi": 0.70 * sp.sin(X + 2 * Y - Z)
        + 0.50 * sp.cos(2 * X - Y + 2 * Z)
        + 0.30 * sp.sin(X - 3 * Y),
        "psi": 0.60 * sp.cos(2 * X - Y + Z)
        + 0.40 * sp.sin(X + Y - 2 * Z)
        + 0.20 * sp.cos(3 * X + Z),
        "du_par": 0.50 * sp.sin(X - 2 * Y + Z) + 0.30 * sp.cos(2 * X + Y - Z),
        "db_par": 0.45 * sp.cos(X + 2 * Y + Z) + 0.35 * sp.sin(2 * X - Y - 2 * Z),
        "s": 0.40 * sp.sin(2 * X + Y - Z) + 0.25 * sp.cos(X - Y + 2 * Z),
    }


def reference_rhs(fields: dict[str, sp.Expr], params) -> dict[str, sp.Expr]:
    """Assemble the analytic RHS for each evolved field from the fields above.

    Built purely with the symbolic operators (no FFT), so the comparison against
    `ideal_rhs` is an independent cross-check rather than a tautology.
    """

    phi = fields["phi"]
    psi = fields["psi"]
    du_par = fields["du_par"]
    db_par = fields["db_par"]
    s = fields["s"]

    vA = params.vA
    chi = params.chi
    alpha = params.alpha
    gamma = params.gamma
    g = params.g
    K_p0 = params.K_p0
    K_b0 = params.K_b0
    K_s = params.K_s

    omega = sym_lap_perp(phi)
    drho = -s / gamma - db_par / chi

    return {
        "psi": vA * sym_dz(phi) - sym_poisson_bracket(phi, psi),
        "omega": vA * sym_dz(sym_lap_perp(psi))
        - sym_poisson_bracket(phi, omega)
        + sym_poisson_bracket(psi, sym_lap_perp(psi))
        - g * sym_dy(drho),
        "du_par": vA**2 * sym_dz(db_par)
        + vA * sym_poisson_bracket(psi, db_par)
        - sym_poisson_bracket(phi, du_par)
        - vA * K_b0 * sym_dy(psi),
        "db_par": alpha * sym_dz(du_par)
        + (alpha / vA) * sym_poisson_bracket(psi, du_par)
        - sym_poisson_bracket(phi, db_par)
        + alpha * K_b0 * sym_dy(phi)
        - alpha * (K_p0 / gamma) * sym_dy(phi),
        "s": K_s * sym_dy(phi) - sym_poisson_bracket(phi, s),
    }


def evaluate_on_grid(expr: sp.Expr, grid) -> np.ndarray:
    """Lambdify a sympy expression and sample it on the real-space grid."""

    func = sp.lambdify((X, Y, Z), expr, modules="numpy")
    grid_x, grid_y, grid_z = np.meshgrid(grid.x, grid.y, grid.z, indexing="ij")
    sampled = func(grid_x, grid_y, grid_z)
    return np.broadcast_to(np.asarray(sampled, dtype=np.float64), grid.real_shape).copy()


def build_input_state(fields: dict[str, sp.Expr], grid, backend, fft) -> State:
    """Load the manufactured fields into a Fourier-space `State`.

    `omega` is set to `lap_perp(phi)` using the solver's own spectral operator,
    so that the code's `phi = inv_lap_perp(omega)` reconstructs the manufactured
    `phi` exactly. The remaining four evolved fields are sampled directly.
    """

    state = State(grid, backend, field_names=eqs.FIELD_NAMES)
    phi_hat = fft.r2c(evaluate_on_grid(fields["phi"], grid))
    state["omega"] = lap_perp(phi_hat, grid)
    for name in ("psi", "du_par", "db_par", "s"):
        state[name] = fft.r2c(evaluate_on_grid(fields[name], grid))
    return state


def main() -> int:
    config, backend, grid, fft, workspace = build_context()
    params = eqs.derived_parameters(config)

    fields = manufactured_solution()
    state = build_input_state(fields, grid, backend, fft)

    # Numerical RHS from the equation module (no dealiasing: grid is alias-free).
    rhs_state = eqs.ideal_rhs(state, grid, fft, workspace, params=config, dealias_mask=None)

    # Independent analytic RHS.
    ref_exprs = reference_rhs(fields, params)

    print(f"MMS residual test for RMHD Equations  (grid {grid.Nx}^3, tol {TOL:.1e})")
    print(f"{'field':>8} | {'max_abs_err':>12} | {'rel_L2_err':>12} | result")
    print("-" * 54)

    all_pass = True
    for name in eqs.FIELD_NAMES:
        numeric = fft.c2r(rhs_state[name])
        exact = evaluate_on_grid(ref_exprs[name], grid)
        max_abs = float(np.max(np.abs(numeric - exact)))
        denom = float(np.sqrt(np.sum(exact**2)))
        rel_l2 = float(np.sqrt(np.sum((numeric - exact) ** 2)) / denom) if denom > 0 else max_abs
        ok = max_abs < TOL
        all_pass &= ok
        print(f"{name:>8} | {max_abs:12.3e} | {rel_l2:12.3e} | {'PASS' if ok else 'FAIL'}")

    print("-" * 54)
    print("ALL PASS" if all_pass else "FAILURES DETECTED")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
