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

def run(tag, diss, tmax=30.0, dt=5e-3):
    c=Config(Nx=12,Ny=12,Nz=12,backend='numpy',vA=1.0,cs2_over_vA2=1.0,K_p0=1.0,K_rho0=1.0,g=0.6)
    b=build_backend(c); grid=build_grid(c,b); f=FFTManager(grid,b); w=Workspace(grid,b); m=build_dealias_mask(grid,b)
    rng=np.random.default_rng(7); s=State(grid,b,field_names=eqs.FIELD_NAMES)
    kp2=grid.kperp2; env=np.where(kp2>0,np.exp(-0.5*kp2/4.0),0.0)
    for n in eqs.FIELD_NAMES:
        s[n][...]=(rng.standard_normal(s[n].shape)+1j*rng.standard_normal(s[n].shape))*m*env
    sc=(0.75/eqs.total_energy(s,grid,b,c))**0.5
    for n in eqs.FIELD_NAMES: s[n][...]*=sc
    project_out_kpar0(s,grid)
    # simple isotropic hyperdiffusion if diss>0
    D={n: (diss*kp2**2 if diss>0 else kp2*0.0) for n in eqs.FIELD_NAMES}
    rk=dict(grid=grid,fft=f,workspace=w,params=c,dealias_mask=m)
    n=int(round(tmax/dt)); out=[]
    for i in range(n+1):
        if i%(n//6)==0: out.append((i*dt, eqs.total_energy(s,grid,b,c)))
        if i<n:
            s=ssprk3_step(s,dt,eqs.ideal_rhs,rhs_kwargs=rk)
            if diss>0:
                for nm in eqs.FIELD_NAMES: s[nm][...]*=np.exp(-D[nm]*dt)
            project_out_kpar0(s,grid)
    print(f"{tag}: "+"  ".join(f"t={t:4.0f}:E={E:.4f}" for t,E in out))

run("IDEAL (no dissipation)", 0.0)
run("WITH dissipation      ", 2e-3)
