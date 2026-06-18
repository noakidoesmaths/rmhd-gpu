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

def zero_bracket(*a, **k): return 0.0

def evolve(linear, diss=8e-6, tmax=120.0, dt=5e-3):
    # stable DR: g=0.5, K_rho0=1.0; WEAK n=3 hyperdiffusion (small-scale only)
    c=Config(Nx=12,Ny=12,Nz=12,backend='numpy',vA=1.0,cs2_over_vA2=1.0,K_p0=0.0,K_rho0=1.0,g=0.5)
    b=build_backend(c); grid=build_grid(c,b); f=FFTManager(grid,b); w=Workspace(grid,b); m=build_dealias_mask(grid,b)
    saved=eqs.poisson_bracket
    if linear: eqs.poisson_bracket=zero_bracket
    kp2=grid.kperp2; D=diss*kp2**3
    try:
        rng=np.random.default_rng(7); s=State(grid,b,field_names=eqs.FIELD_NAMES)
        env=np.where(kp2>0,np.exp(-0.5*kp2/4.0),0.0)
        for n in eqs.FIELD_NAMES:
            s[n][...]=(rng.standard_normal(s[n].shape)+1j*rng.standard_normal(s[n].shape))*m*env
        sc=(0.75/eqs.total_energy(s,grid,b,c))**0.5
        for n in eqs.FIELD_NAMES: s[n][...]*=sc
        project_out_kpar0(s,grid)
        rk=dict(grid=grid,fft=f,workspace=w,params=c,dealias_mask=m)
        n=int(round(tmax/dt)); out=[]; Iacc=0.0
        for i in range(n+1):
            if i%(n//8)==0: out.append((i*dt, eqs.total_energy(s,grid,b,c), Iacc))
            Y=eqs.total_energy_stratification_rhs(s,grid,b,c)
            if i<n:
                s=ssprk3_step(s,dt,eqs.ideal_rhs,rhs_kwargs=rk)
                for nm in eqs.FIELD_NAMES: s[nm][...]*=np.exp(-D*dt)
                project_out_kpar0(s,grid); Iacc+=Y*dt
    finally:
        eqs.poisson_bracket=saved
    return out

for lab,lin in [("LINEAR    + WEAK dissipation",True),("NONLINEAR + WEAK dissipation",False)]:
    out=evolve(lin)
    print(f"\n=== {lab} (stable DR) ===")
    print("  "+"  ".join(f"t={t:4.0f}:E={E:.3f}(intY={I:+.2f})" for t,E,I in out))
