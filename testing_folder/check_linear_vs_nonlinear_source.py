"""Why does mode=linear oscillate but full nonlinear grow (g=0, K_rho0=1)?

Evolve the SAME initial state two ways:
  (a) linear  : Poisson bracket set to zero (what mode='linear' does)
  (b) nonlinear: full ideal RHS
and in each case track total energy E(t) and the cumulative energy injected by
the LINEAR stratification source, I(t) = integral of Y dt.

Claim being tested:
  * The brackets themselves inject ZERO energy (verified separately with all
    gradients off -> dE/dt == 0).
  * Here, with g=0 but K_rho0=1 (so K_s != 0), the injector is the linear
    source Y.  In the linear run <s dy(phi)> stays in quadrature so I(t) ~ 0
    and E oscillates.  The brackets break that phase-locking, so in the
    nonlinear run I(t) drifts positive and E grows -- E(t) tracks 0.75 + I(t).
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
from rmhdgpu.steppers import project_out_kpar0, ssprk3_step
from rmhdgpu.workspace import Workspace


def zero_bracket(*a, **k):
    return 0.0


def random_state(grid, backend, mask, config, target_E=0.75, seed=7):
    rng = np.random.default_rng(seed)
    state = State(grid, backend, field_names=eqs.FIELD_NAMES)
    kperp2 = grid.kperp2
    env = np.where(kperp2 > 0, np.exp(-0.5 * kperp2 / 4.0), 0.0)
    for name in eqs.FIELD_NAMES:
        re = rng.standard_normal(state[name].shape)
        im = rng.standard_normal(state[name].shape)
        state[name][...] = (re + 1j * im) * mask * env
    # normalize to a realistic O(1) energy so the nonlinear bracket matters
    E = eqs.total_energy(state, grid, state.backend, config)
    scale = (target_E / E) ** 0.5
    for name in eqs.FIELD_NAMES:
        state[name][...] *= scale
    return state


def evolve(linear: bool, *, tmax=150.0, dt=5.0e-3):
    config = Config(Nx=12, Ny=12, Nz=12, backend="numpy",
                    vA=1.0, cs2_over_vA2=1.0, g=0.0, K_p0=0.0, K_rho0=1.0)
    backend = build_backend(config)
    grid = build_grid(config, backend)
    fft = FFTManager(grid, backend)
    workspace = Workspace(grid, backend)
    mask = build_dealias_mask(grid, backend)

    # mode='linear' literally swaps poisson_bracket for a zero function:
    saved = eqs.poisson_bracket
    if linear:
        eqs.poisson_bracket = zero_bracket

    rhs_kwargs = dict(grid=grid, fft=fft, workspace=workspace,
                      params=config, dealias_mask=mask)
    try:
        state = random_state(grid, backend, mask, config)
        project_out_kpar0(state, grid)   # match the real runs (project_kpar0=true)
        n = int(round(tmax / dt))
        ts, Es, Is = [], [], []
        I = 0.0
        for i in range(n + 1):
            E = eqs.total_energy(state, grid, backend, config)
            Y = eqs.total_energy_stratification_rhs(state, grid, backend, config)
            if i % 1500 == 0:
                ts.append(i * dt); Es.append(E); Is.append(I)
            if i < n:
                state = ssprk3_step(state, dt, eqs.ideal_rhs, rhs_kwargs=rhs_kwargs)
                project_out_kpar0(state, grid)   # re-project every step
                I += Y * dt   # crude left-rule running integral of the source
    finally:
        eqs.poisson_bracket = saved
    return np.array(ts), np.array(Es), np.array(Is)


if __name__ == "__main__":
    for label, lin in [("LINEAR (brackets=0)", True), ("NONLINEAR (full)", False)]:
        t, E, I = evolve(linear=lin)
        E0 = E[0]
        print(f"\n=== {label} ===")
        print(f"{'t':>6}{'E':>10}{'E-E0':>10}{'I=sumYdt':>10}{'E0+I':>10}")
        for j in range(0, len(t), max(1, len(t) // 10)):
            print(f"{t[j]:6.1f}{E[j]:10.4f}{E[j]-E0:10.4f}{I[j]:10.4f}{E0+I[j]:10.4f}")
        print(f"net energy change over run: {E[-1]-E0:+.4f}   (cumulative source {I[-1]:+.4f})")
