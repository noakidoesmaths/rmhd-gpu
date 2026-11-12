"""`project_ky0` / `project_kx0`: removing every k_y = 0 (or k_x = 0) Fourier
mode each step (control options, alongside `project_kpar0`).

Checks the projections themselves (`f -> f - <f>_y` or `f - <f>_x`, nothing
else touched), the `.input` path, and short nonlinear runs: the removed sector
stays empty and the removed energy is booked in the `projection` budget column.
"""

from __future__ import annotations

import csv
from types import SimpleNamespace

import numpy as np
import pytest

from rmhdgpu.backend import build_backend
from rmhdgpu.config import Config
from rmhdgpu.fft import FFTManager
from rmhdgpu.grid import build_grid
from rmhdgpu.equations import get_equation_module
from rmhdgpu.initconds import build_initial_state
from rmhdgpu.operators import inv_lap_perp
from rmhdgpu.diagnostics import compressive_channels
from rmhdgpu.run import main
from rmhdgpu.runfile import resolve_run_settings
from rmhdgpu.state import State
from rmhdgpu.steppers import (
    apply_mode_projections,
    mode_projections_enabled,
    project_out_kx0,
    project_out_ky0,
)


def _random_state(seed: int = 4):
    config = Config(Nx=8, Ny=6, Nz=8, Lx=2.0 * np.pi, Ly=3.0, Lz=5.0, backend="numpy")
    backend = build_backend(config)
    grid = build_grid(config, backend)
    fft = FFTManager(grid, backend)
    rng = np.random.default_rng(seed)
    state = State(grid, backend, field_names=("phi", "psi"))
    real_fields = {name: rng.standard_normal(grid.real_shape) for name in state.field_names}
    for name, values in real_fields.items():
        state[name][...] = fft.r2c(values)
    return grid, fft, state, real_fields


@pytest.mark.parametrize("project, axis", [(project_out_kx0, 0), (project_out_ky0, 1)])
def test_projection_subtracts_the_average_along_its_axis(project, axis):
    grid, fft, state, real_fields = _random_state()
    before = {name: state[name].copy() for name in state.field_names}

    project(state, grid)

    k_axis = (grid.kx, grid.ky)[axis]
    removed = np.broadcast_to(k_axis == 0, grid.fourier_shape)
    for name, values in real_fields.items():
        # Only the k = 0 modes on that axis change, and they become exactly zero.
        assert np.all(state[name][removed] == 0.0)
        np.testing.assert_array_equal(state[name][~removed], before[name][~removed])
        # In real space this is f - <f>_x (or _y), and the field stays real.
        expected = values - values.mean(axis=axis, keepdims=True)
        np.testing.assert_allclose(fft.c2r(state[name]), expected, atol=1e-12)


def test_apply_mode_projections_applies_only_the_switches_that_are_on():
    grid, _, state, _ = _random_state()
    expected = {name: state[name].copy() for name in state.field_names}
    for name in expected:
        expected[name] *= (grid.kx != 0) & (grid.kz != 0)

    params = SimpleNamespace(project_kpar0=True, project_ky0=False, project_kx0=True)
    assert mode_projections_enabled(params)
    apply_mode_projections(state, grid, params)

    for name in state.field_names:
        np.testing.assert_array_equal(state[name], expected[name])
    assert not mode_projections_enabled(SimpleNamespace(project_kpar0=False))


@pytest.mark.parametrize("switch", ["project_ky0", "project_kx0"])
def test_projection_switch_is_read_from_the_runtime_section(tmp_path, switch):
    input_file = tmp_path / "case.input"
    input_file.write_text(f"[runtime]\n{switch} = true\n", encoding="utf-8")

    settings = resolve_run_settings(runfile_path=input_file)

    for name in ("project_kpar0", "project_ky0", "project_kx0"):
        assert getattr(settings.config, name) is (name == switch)
    assert settings.resolved_document["runtime"][switch] is True
    assert getattr(resolve_run_settings(runfile_path=None, cwd=tmp_path).config, switch) is False


@pytest.mark.parametrize("kind", ["random_spectrum", "random_spectrum_one_wave"])
@pytest.mark.parametrize("initialization_backend", ["native", "numpy"])
@pytest.mark.parametrize("switches", [
    {"project_kx0": True},
    {"project_ky0": True},
    {"project_kx0": True, "project_ky0": True, "project_kpar0": True},
])
def test_initial_energy_is_normalized_over_surviving_modes(kind, initialization_backend, switches):
    config = Config(equation_set="inhomogeneous_rmhd_rho", Nx=8, Ny=8, Nz=8,
                    backend="numpy", **switches)
    backend = build_backend(config)
    grid = build_grid(config, backend)
    state = build_initial_state(
        kind, parameters={"seed": 1, "init_energy": 0.49, "exclude_kpar0": False,
                          "initialization_backend": initialization_backend},
        grid=grid, backend=backend, fft=FFTManager(grid, backend),
        dealias_mask=None, field_names=config.field_names, params=config,
    )
    for switch, wave_number in (("project_kx0", grid.kx), ("project_ky0", grid.ky),
                                ("project_kpar0", grid.kz)):
        if switches.get(switch):
            removed = np.broadcast_to(wave_number == 0, grid.fourier_shape)
            for name in state.field_names:
                assert np.all(state[name][removed] == 0)
    # Startup cleanup is now idempotent and cannot lower the requested energy.
    apply_mode_projections(state, grid, config)
    equation = get_equation_module(config.equation_set)
    assert equation.total_energy(state, grid, backend, config) == pytest.approx(0.49)
    if kind == "random_spectrum_one_wave":
        phi = inv_lap_perp(state["omega"], grid)
        plus, minus = compressive_channels.elsasser_potentials(phi, state["psi"])
        assert 2 * np.sqrt(compressive_channels.elsasser_energy(plus, grid, backend)) == pytest.approx(1.4)
        assert compressive_channels.elsasser_energy(minus, grid, backend) < 1e-28


RUN_INPUT = """
title = "project_ky0 test"
output_dir = "outputs"

[equations]
type = "inhomogeneous_rmhd_rho"
mode = "nonlinear"

[grid]
Nx = 8
Ny = 8
Nz = 8

[time]
tmax = 0.2
dt_init = 0.005
dt_max = 0.005
use_variable_dt = false

[output]
t_out_scal = 0.02
t_out_spec = 0.0
t_out_full = 0.0

[backend]
backend = "numpy"

[runtime]
progress_output_every = 1000
project_ky0 = true

[physics]
vA = 1.0
cs2_over_vA2 = 0.1
K_rho0 = 1.5
g = 0.06
K_p0 = 1.0

[initial_condition]
type = "random_spectrum"

[initial_condition.parameters]
seed = 3
init_energy = 0.5
exclude_kpar0 = false
"""


@pytest.mark.parametrize("switch, axis", [("project_ky0", 1), ("project_kx0", 0)])
def test_run_keeps_the_sector_empty_and_books_the_removed_energy(tmp_path, switch, axis):
    h5py = pytest.importorskip("h5py")
    input_file = tmp_path / "case.input"
    run_input = RUN_INPUT.replace("project_ky0 = true", f"{switch} = true")
    input_file.write_text(run_input.replace("t_out_full = 0.0", "t_out_full = 0.2"), encoding="utf-8")

    main([str(input_file)])

    with (tmp_path / "outputs" / "scalar_diagnostics.csv").open(newline="", encoding="utf-8") as handle:
        rows = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(handle)]
    energy = np.array([row["total_energy"] for row in rows])
    assert energy[0] == pytest.approx(0.5)
    projection = np.array([row["total_energy_rhs_projection"] for row in rows[1:]])

    # The nonlinear brackets refill the removed modes every step; the projection
    # empties them again, so every saved field has zero average along that axis.
    last_snapshot = sorted((tmp_path / "outputs").rglob("*.h5"))[-1]
    with h5py.File(last_snapshot, "r") as handle:
        for name in ("psi", "omega", "du_par", "db_par", "drho"):
            values = handle["output"][name][...]
            average = values.mean(axis=axis)
            assert np.max(np.abs(average)) <= 1e-12 * np.max(np.abs(values))
    if switch == "project_ky0":
        assert all(row["E_ky0"] <= 1e-14 * row["total_energy"] for row in rows)
    # Projection only removes energy, and here it removes some.
    assert np.all(projection <= 1e-14)
    assert np.any(projection < -1e-10)
    # The budget, projection included, accounts for the measured energy change.
    # The other terms are time-integrated by the budget, which leaves a residual
    # of ~1e-8 here with or without the projection; the projection term itself
    # is ~1e-5, so this tolerance still catches it going missing.
    dt_out = np.diff([row["time"] for row in rows])
    measured = np.diff(energy) / dt_out
    rhs_total = np.array([row["total_energy_rhs_total"] for row in rows[1:]])
    assert np.max(np.abs(measured - rhs_total)) < 1e-2 * np.max(np.abs(projection))
