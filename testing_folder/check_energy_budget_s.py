"""Diagnose the energy growth in the inhomogeneous_rmhd_s nonlinear runs.

Checks, for the parameters in examples/nokias_inhomogeneous.input:

1. Linear stability: max growth rate of `linear_matrix` over the resolved k modes
   (with and without kz=0).
2. Bracket conservation: with all background gradients off (g=K=0), the ideal RHS
   should conserve total_energy to machine precision.
3. Budget closure: with gradients on, dE/dt along the ideal RHS should equal the
   stratification source Y = total_energy_stratification_rhs.
"""

from __future__ import annotations

import numpy as np

from rmhdgpu.backend import build_backend
from rmhdgpu.config import Config
from rmhdgpu.equations import rmhd_by_nokia_s as eqs
from rmhdgpu.fft import FFTManager
from rmhdgpu.grid import build_grid
from rmhdgpu.masks import build_dealias_mask
from rmhdgpu.state import State
from rmhdgpu.steppers import state_linear_combination
from rmhdgpu.workspace import Workspace


def build_context(**phys):
    config = Config(Nx=16, Ny=16, Nz=16, backend="numpy", **phys)
    backend = build_backend(config)
    grid = build_grid(config, backend)
    fft = FFTManager(grid, backend)
    workspace = Workspace(grid, backend)
    mask = build_dealias_mask(grid, backend)
    return config, backend, grid, fft, workspace, mask


def random_state(grid, backend, mask, seed=7):
    rng = np.random.default_rng(seed)
    state = State(grid, backend, field_names=eqs.FIELD_NAMES)
    shape = state["psi"].shape
    for name in eqs.FIELD_NAMES:
        re = rng.standard_normal(shape)
        im = rng.standard_normal(shape)
        state[name][...] = (re + 1j * im) * mask
    # kill kperp=0 column so inv_lap_perp is well defined, and scale down high k
    kperp2 = grid.kperp2
    for name in eqs.FIELD_NAMES:
        state[name][...] *= np.where(kperp2 > 0, np.exp(-0.5 * kperp2 / 9.0), 0.0)
    return state


def dE_dt_along_rhs(state, rhs, grid, backend, config, eps=1e-7):
    sp = state_linear_combination(state, [(1.0, state), (eps, rhs)])
    sm = state_linear_combination(state, [(1.0, state), (-eps, rhs)])
    Ep = eqs.total_energy(sp, grid, backend, config)
    Em = eqs.total_energy(sm, grid, backend, config)
    return (Ep - Em) / (2.0 * eps)


def report_budget(tag, **phys):
    config, backend, grid, fft, workspace, mask = build_context(**phys)
    state = random_state(grid, backend, mask)
    rhs = eqs.ideal_rhs(state, grid, fft, workspace, config, dealias_mask=mask)
    dEdt = dE_dt_along_rhs(state, rhs, grid, backend, config)
    Y = eqs.total_energy_stratification_rhs(state, grid, backend, config)
    E = eqs.total_energy(state, grid, backend, config)
    print(f"[{tag}] E = {E:.6e}   dE/dt(ideal RHS) = {dEdt:+.6e}   "
          f"Y(stratification) = {Y:+.6e}   residual = {dEdt - Y:+.3e}")


def linear_growth_scan(**phys):
    config = Config(Nx=64, Ny=64, Nz=64, backend="numpy", **phys)
    ks = list(range(-10, 11))
    kzs = list(range(0, 11))
    worst = (0.0, None)
    worst_finite_kz = (0.0, None)
    for kx in ks:
        for ky in ks:
            for kz in kzs:
                if kx == 0 and ky == 0:
                    continue
                M = eqs.linear_matrix(float(kx), float(ky), float(kz), config)
                gmax = float(np.max(np.real(np.linalg.eigvals(M))))
                if gmax > worst[0]:
                    worst = (gmax, (kx, ky, kz))
                if kz != 0 and gmax > worst_finite_kz[0]:
                    worst_finite_kz = (gmax, (kx, ky, kz))
    print(f"max growth rate (all modes, incl. kz=0): {worst[0]:.6e} at k={worst[1]}")
    print(f"max growth rate (kz != 0 only):          {worst_finite_kz[0]:.6e} at k={worst_finite_kz[1]}")
    p = eqs.derived_parameters(config)
    print(f"derived: K_b0={p.K_b0:.4f}, K_s={p.K_s:.4f}, alpha={p.alpha:.4f}, N_sq={p.N_sq:.4f}")


if __name__ == "__main__":
    input_phys = dict(vA=1.0, cs2_over_vA2=1.0, g=0.5, K_p0=0.0, K_rho0=-0.5)

    print("=== 1. linear stability scan for the input-file parameters ===")
    linear_growth_scan(**input_phys)

    print("\n=== 2. bracket-only conservation check (all gradients off) ===")
    report_budget("g=0, K=0", vA=1.0, cs2_over_vA2=1.0, g=0.0, K_p0=0.0, K_rho0=0.0)

    print("\n=== 3. budget closure with input-file parameters ===")
    report_budget("input params", **input_phys)

    print("\n=== 4. 'stable' parameters per K_rho >= g criterion ===")
    stable_phys = dict(vA=1.0, cs2_over_vA2=1.0, g=0.5, K_p0=0.0, K_rho0=0.7)
    linear_growth_scan(**stable_phys)
    report_budget("K_rho0=0.7 > g", **stable_phys)
