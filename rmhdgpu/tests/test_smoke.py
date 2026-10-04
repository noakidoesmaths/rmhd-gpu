from __future__ import annotations

from rmhdgpu import Config, FFTManager, Workspace, build_backend, build_grid, compute_cfl_timestep, if_ssprk3_step
from rmhdgpu.diagnostics.alfvenic import alfvenic_cross_helicity, alfvenic_energy
from rmhdgpu.diagnostics.scalar import compute_scalar_diagnostics
from rmhdgpu.equations import s09
from rmhdgpu.initconds.eigenmodes_s09 import alfven_mode_state
from rmhdgpu.masks import build_dealias_mask


def test_smoke_construction_and_diagnostics() -> None:
    config = Config(Nx=8, Ny=8, Nz=8, dt_init=5.0e-3, dt_max=2.0e-2)
    config.dissipation["psi"]["nu_perp"] = 5.0e-3
    config.dissipation["omega"]["nu_perp"] = 5.0e-3
    backend = build_backend(config)
    grid = build_grid(config, backend)
    fft = FFTManager(grid, backend)
    mask = build_dealias_mask(grid, backend)
    workspace = Workspace(grid, backend)
    linear_ops = s09.build_dissipation_operators(grid, config)
    state = alfven_mode_state(
        grid=grid,
        backend=backend,
        field_names=config.field_names,
        k_indices=(1, 1, 1),
        amplitude=1.0,
        branch="z_minus",
        params=config,
    )

    rhs_kwargs = {
        "grid": grid,
        "fft": fft,
        "workspace": workspace,
        "params": config,
        "dealias_mask": mask,
    }
    dt = compute_cfl_timestep(state, grid, fft, config)
    state = if_ssprk3_step(state, dt, s09.ideal_rhs, linear_ops, rhs_kwargs=rhs_kwargs)
    diagnostics = compute_scalar_diagnostics(state, grid, fft, backend)
    energy = alfvenic_energy(state, grid, fft)
    cross_helicity = alfvenic_cross_helicity(state, grid, fft)

    assert workspace.real["r0"].shape == grid.real_shape
    assert "psi_rms" in diagnostics
    assert energy > 0.0
    assert abs(cross_helicity) > 0.0
