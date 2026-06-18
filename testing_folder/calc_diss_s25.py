"""Calculate dissipation to overpower the LINEAR INSTABILITY for s_25 params.

s_25: g=0.6, K_p0=1.0, K_rho0=-2.0  ->  N^2=-1.38  (convectively UNSTABLE).

For an unstable run the condition for energy decay is mode-by-mode:
   dissipation amplitude rate  D(k) = nu_perp*kperp^(2 n_perp) + nu_par*kpar^(2 n_par)
   must exceed the linear growth rate  gamma_lin(k) = max Re eig( linear_matrix(k) ).

So the required coefficient (perp-only) is
   nu_perp,min = max over retained modes of  gamma_lin(k) / kperp^(2 n_perp).
The binding mode is the most-unstable LARGE scale (smallest kperp), since dissipation
is weakest there.  We scan the actual retained 32^3 grid (dealiased, kpar != 0).
"""
from __future__ import annotations
import numpy as np
from rmhdgpu.config import Config
from rmhdgpu.equations import rmhd_by_nokia_s as eqs

N, L = 32, 2*np.pi
dk = 2*np.pi/L
c = Config(Nx=N, Ny=N, Nz=N, backend="numpy",
           vA=1.0, cs2_over_vA2=1.0, g=0.6, K_p0=1.0, K_rho0=-2.0)

nmax = int(np.ceil(N/3.0)) - 1            # 2/3 dealias keeps |n| <= nmax
ns = np.arange(-nmax, nmax+1)
nz = np.arange(1, nmax+1)                 # project_kpar0 removes kz=0

best = {}                                 # n_perp -> (nu_req, kperp, kz)
gamma_max = 0.0; arg = None
records = []
for nx in ns:
    for ny in ns:
        kperp = np.hypot(nx, ny)*dk
        if kperp == 0.0:
            continue
        for iz in nz:
            kz = iz*dk
            M = eqs.linear_matrix(nx*dk, ny*dk, kz, c)
            g_lin = float(np.max(np.real(np.linalg.eigvals(M))))
            if g_lin <= 0:
                continue
            if g_lin > gamma_max:
                gamma_max, arg = g_lin, (kperp, kz)
            records.append((kperp, kz, g_lin))

print(f"max linear growth rate gamma_lin,max = {gamma_max:.4f}  at (kperp,kz)={arg}")
print(f"(compare sqrt(|N^2|) = {np.sqrt(1.38):.4f})\n")

for n_perp in (1, 2, 3):
    nu_req = max(g/(kp**(2*n_perp)) for kp, kz, g in records)
    bind = max(records, key=lambda r: r[2]/(r[0]**(2*n_perp)))
    print(f"n_perp={n_perp}:  nu_perp,min = {nu_req:.4e}   "
          f"(binding mode kperp={bind[0]:.3f}, kz={bind[1]:.3f}, gamma={bind[2]:.3f})")
