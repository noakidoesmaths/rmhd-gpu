"""Verify the linear stability threshold for inhomogeneous_rmhd_s.

Tests two competing thresholds against the code's own linear_matrix:
  (a) buoyancy only:        K_rho0 >= g/(vA^2 + cs^2)   <=>  N^2 >= 0
  (b) full dispersion rel:  K_rho0 >= g/cs^2            <=>  N^2 >= vS^2 g^2/cs^4

For each candidate K_rho0 we scan eigenvalues of linear_matrix over a range of
kz (including SMALL kz, which is where the slow-mode-coupled instability lives)
and report the max growth rate.
"""

from __future__ import annotations

import numpy as np

from rmhdgpu.config import Config
from rmhdgpu.equations import rmhd_by_nokia_s as eqs


def max_growth(K_rho0, *, kz_values, ky=4.0, kx=0.0, vA=1.0, chi=1.0, g=0.5, K_p0=0.0):
    config = Config(Nx=8, Ny=8, Nz=8, backend="numpy",
                    vA=vA, cs2_over_vA2=chi, g=g, K_p0=K_p0, K_rho0=K_rho0)
    gmax = -np.inf
    arg = None
    for kz in kz_values:
        M = eqs.linear_matrix(kx, ky, kz, config)
        g_here = float(np.max(np.real(np.linalg.eigvals(M))))
        if g_here > gmax:
            gmax, arg = g_here, kz
    return gmax, arg


def thresholds(vA=1.0, chi=1.0, g=0.5):
    cs2 = chi * vA**2
    return dict(
        buoyancy=g / (vA**2 + cs2),   # N^2 = 0  threshold
        full=g / cs2,                 # g/cs^2  threshold
    )


if __name__ == "__main__":
    vA, chi, g = 1.0, 1.0, 0.5
    th = thresholds(vA, chi, g)
    print(f"buoyancy-only threshold  K_rho0 = g/(vA^2+cs^2) = {th['buoyancy']:.4f}")
    print(f"full-DR     threshold    K_rho0 = g/cs^2        = {th['full']:.4f}")
    print()

    # fine grid of small kz down to near zero -- the instability lives at small kz
    kz_values = np.concatenate([np.linspace(0.0, 1.0, 41), np.linspace(1.0, 6.0, 26)])

    print(f"{'K_rho0':>8} {'N^2':>10} {'max_growth':>12} {'arg kz':>8}  region")
    for K_rho0 in [-0.5, 0.0, 0.20, 0.25, 0.30, 0.40, 0.45, 0.49, 0.50, 0.55, 0.70]:
        config = Config(Nx=8, Ny=8, Nz=8, backend="numpy",
                        vA=vA, cs2_over_vA2=chi, g=g, K_p0=0.0, K_rho0=K_rho0)
        Nsq = eqs.derived_parameters(config).N_sq
        gmax, arg = max_growth(K_rho0, kz_values=kz_values, vA=vA, chi=chi, g=g)
        if K_rho0 < th['buoyancy'] - 1e-9:
            region = "unstable (both criteria)"
        elif K_rho0 < th['full'] - 1e-9:
            region = "<-- discriminating window (buoyancy says stable)"
        else:
            region = "stable (both criteria)"
        print(f"{K_rho0:8.3f} {Nsq:10.4f} {gmax:12.3e} {arg:8.3f}  {region}")
