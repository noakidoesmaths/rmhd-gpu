"""Does 'fully consistent + stable' stratification keep nonlinear energy bounded?

Full nonlinear ideal RHS (no dissipation), project_kpar0=true, same seed/IC.
Three cases:

  A. inconsistent : g=0,   K_rho0=1.0   (source K_s!=0 but buoyancy loop cut)
  B. consistent + STABLE   : g=0.5, K_rho0=1.0  (K_rho >= g/cs^2 = 0.5)
  C. consistent + UNSTABLE : g=0.5, K_rho0=0.0  (K_rho <  g/cs^2)

Expectation: A grows (open source), B stays bounded/decays (stratification is a
sink), C grows (available potential energy to release).
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


def random_state(grid, backend, mask, config, target_E=0.75, seed=7):
    rng = np.random.default_rng(seed)
    state = State(grid, backend, field_names=eqs.FIELD_NAMES)
    kperp2 = grid.kperp2
    env = np.where(kperp2 > 0, np.exp(-0.5 * kperp2 / 4.0), 0.0)
    for name in eqs.FIELD_NAMES:
        re = rng.standard_normal(state[name].shape)
        im = rng.standard_normal(state[name].shape)
        state[name][...] = (re + 1j * im) * mask * env
    E = eqs.total_energy(state, grid, state.backend, config)
    scale = (target_E / E) ** 0.5
    for name in eqs.FIELD_NAMES:
        state[name][...] *= scale
    return state


def run(label, g, K_rho0, *, tmax=60.0, dt=5.0e-3):
    config = Config(Nx=12, Ny=12, Nz=12, backend="numpy",
                    vA=1.0, cs2_over_vA2=1.0, g=g, K_p0=0.0, K_rho0=K_rho0)
    backend = build_backend(config)
    grid = build_grid(config, backend)
    fft = FFTManager(grid, backend)
    workspace = Workspace(grid, backend)
    mask = build_dealias_mask(grid, backend)
    rhs_kwargs = dict(grid=grid, fft=fft, workspace=workspace,
                      params=config, dealias_mask=mask)

    state = random_state(grid, backend, mask, config)
    project_out_kpar0(state, grid)
    n = int(round(tmax / dt))
    E0 = eqs.total_energy(state, grid, backend, config)
    samples = []
    for i in range(n + 1):
        if i % (n // 6) == 0:
            E = eqs.total_energy(state, grid, backend, config)
            samples.append((i * dt, E))
        if i < n:
            state = ssprk3_step(state, dt, eqs.ideal_rhs, rhs_kwargs=rhs_kwargs)
            project_out_kpar0(state, grid)
    Efin = eqs.total_energy(state, grid, backend, config)
    print(f"\n=== {label}: g={g}, K_rho0={K_rho0} (g/cs^2={g/1.0:.2f}) ===")
    print("   " + "  ".join(f"t={t:4.0f}:E={E:6.3f}" for t, E in samples))
    print(f"   net dE = {Efin - E0:+.4f}   ({'GROWS' if Efin > E0*1.05 else 'bounded/decays'})")


if __name__ == "__main__":
    run("A inconsistent (g=0)",        g=0.0, K_rho0=1.0)
    run("B consistent + STABLE",       g=0.5, K_rho0=1.0)
    run("C consistent + UNSTABLE",     g=0.5, K_rho0=0.0)
