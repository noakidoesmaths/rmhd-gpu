"""Can dissipation outweigh <Y> for the user's actual stable params?

Nokia's run: g=0.6, K_p0=1.0, K_rho0=2.0  ->  N^2=+1.02 (linearly stable),
but large entropy gradient K_s = -2.33.  Full nonlinear ideal RHS, then apply
fieldwise exp(-D*dt) with D = nu * kperp2**n every step.

We sweep:
  - the dissipation STRENGTH nu
  - the SCALE it acts on via the hyperdiffusion order n
      n=1 : ordinary Laplacian, bites large (energy-containing) scales too
      n=3 : steep hyperdiffusion, essentially high-k only

Report E(t) and the running source integral intY = sum <Y> dt.
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


def run(nu, n, *, tmax=60.0, dt=5.0e-3, seed=7):
    c = Config(Nx=16, Ny=16, Nz=16, backend="numpy",
               vA=1.0, cs2_over_vA2=1.0, g=0.6, K_p0=1.0, K_rho0=2.0)
    b = build_backend(c); grid = build_grid(c, b)
    f = FFTManager(grid, b); w = Workspace(grid, b); m = build_dealias_mask(grid, b)
    kp2 = grid.kperp2
    D = nu * kp2 ** n

    rng = np.random.default_rng(seed); s = State(grid, b, field_names=eqs.FIELD_NAMES)
    env = np.where(kp2 > 0, np.exp(-0.5 * kp2 / 4.0), 0.0)
    for name in eqs.FIELD_NAMES:
        s[name][...] = (rng.standard_normal(s[name].shape)
                        + 1j * rng.standard_normal(s[name].shape)) * m * env
    sc = (0.75 / eqs.total_energy(s, grid, b, c)) ** 0.5
    for name in eqs.FIELD_NAMES:
        s[name][...] *= sc
    project_out_kpar0(s, grid)

    rk = dict(grid=grid, fft=f, workspace=w, params=c, dealias_mask=m)
    nsteps = int(round(tmax / dt)); out = []; Iacc = 0.0
    for i in range(nsteps + 1):
        if i % (nsteps // 6) == 0:
            out.append((i * dt, eqs.total_energy(s, grid, b, c), Iacc))
        Y = eqs.total_energy_stratification_rhs(s, grid, b, c)
        if i < nsteps:
            s = ssprk3_step(s, dt, eqs.ideal_rhs, rhs_kwargs=rk)
            for nm in eqs.FIELD_NAMES:
                s[nm][...] *= np.exp(-D * dt)
            project_out_kpar0(s, grid)
            Iacc += Y * dt
    E0 = out[0][1]; Ef = out[-1][1]
    verdict = "GROWS" if Ef > 1.02 * E0 else ("DECAYS" if Ef < 0.98 * E0 else "~flat")
    return out, verdict


cases = [
    ("ideal (nu=0)",            0.0,    1),
    ("n=3 hi-k  nu=1e-2",       1e-2,   3),
    ("n=3 hi-k  nu=1e-1",       1e-1,   3),
    ("n=1 Lapl  nu=5e-3",       5e-3,   1),
    ("n=1 Lapl  nu=2e-2",       2e-2,   1),
    ("n=1 Lapl  nu=1e-1",       1e-1,   1),
]
for label, nu, n in cases:
    out, verdict = run(nu, n)
    traj = "  ".join(f"t={t:4.0f}:E={E:6.3f}(I={I:+.2f})" for t, E, I in out)
    print(f"{label:22s} -> {verdict:6s}  {traj}")
