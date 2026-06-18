"""Find nu_perp (n=1) that drives s_22 params to E->0 while keeping stratification.

s_22 physics: g=0.6, K_p0=0.0, K_rho0=2.0  (linearly stable, N^2=+1.02).
Full nonlinear ideal RHS + fieldwise exp(-D*dt), D = nu * kperp2**n (perp only),
project_kpar0 every step.  Report E(t) and whether it converges to ~0.
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


def run(nu, n, *, tmax=150.0, dt=5.0e-3, seed=1):
    c = Config(Nx=16, Ny=16, Nz=16, backend="numpy",
               vA=1.0, cs2_over_vA2=1.0, g=0.6, K_p0=0.0, K_rho0=2.0)
    b = build_backend(c); grid = build_grid(c, b)
    f = FFTManager(grid, b); w = Workspace(grid, b); m = build_dealias_mask(grid, b)
    kp2 = grid.kperp2; D = nu * kp2 ** n
    rng = np.random.default_rng(seed); s = State(grid, b, field_names=eqs.FIELD_NAMES)
    env = np.where(kp2 > 0, np.exp(-0.5 * kp2 / 4.0), 0.0)
    for name in eqs.FIELD_NAMES:
        s[name][...] = (rng.standard_normal(s[name].shape)
                        + 1j * rng.standard_normal(s[name].shape)) * m * env
    sc = (1.0 / eqs.total_energy(s, grid, b, c)) ** 0.5
    for name in eqs.FIELD_NAMES:
        s[name][...] *= sc
    project_out_kpar0(s, grid)
    rk = dict(grid=grid, fft=f, workspace=w, params=c, dealias_mask=m)
    nsteps = int(round(tmax / dt)); samples = []
    for i in range(nsteps + 1):
        if i % (nsteps // 6) == 0:
            samples.append((i * dt, eqs.total_energy(s, grid, b, c)))
        if i < nsteps:
            s = ssprk3_step(s, dt, eqs.ideal_rhs, rhs_kwargs=rk)
            for nm in eqs.FIELD_NAMES:
                s[nm][...] *= np.exp(-D * dt)
            project_out_kpar0(s, grid)
    Ef = samples[-1][1]
    verdict = "-> ~0" if Ef < 0.02 else ("decaying" if Ef < 0.5 else "plateau/grow")
    return samples, Ef, verdict


print("threshold estimate: nu_perp > gamma_inj/2 ~ 0.075/2 ~ 0.04 (n=1, k_min=1)\n")
for label, nu, n in [
    ("n=1 nu=0      (ideal)", 0.0,  1),
    ("n=1 nu=0.01",          1e-2, 1),
    ("n=1 nu=0.02",          2e-2, 1),
    ("n=1 nu=0.04",          4e-2, 1),
    ("n=1 nu=0.08",          8e-2, 1),
    ("n=3 nu=0.04 (compare)",4e-2, 3),
]:
    s, Ef, v = run(nu, n)
    traj = "  ".join(f"t={t:5.0f}:E={E:6.3f}" for t, E in s)
    print(f"{label:24s} Ef={Ef:7.4f}  {v:13s}  {traj}")
