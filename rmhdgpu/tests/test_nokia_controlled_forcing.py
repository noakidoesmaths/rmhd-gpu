"""Controlled shell forcing on the inhomogeneous (nokia) equation sets.

These modules reuse the standard s09 forcing hooks. The tests run with nonzero
background gradients (K_rho0, K_p0, g), where z+/- are no longer linear
eigenmodes, to check that the energy bookkeeping and branch algebra still hold.
"""
import csv

import numpy as np
import pytest

from rmhdgpu import forcing_control
from rmhdgpu.backend import build_backend
from rmhdgpu.diagnostics.alfvenic import elsasser_energies
from rmhdgpu.equations import get_equation_module
from rmhdgpu.grid import build_grid
from rmhdgpu.masks import build_dealias_mask
from rmhdgpu.run import run_simulation
from rmhdgpu.runfile import dump_toml, resolve_run_settings
from rmhdgpu.state import State

NOKIA_SETS = ["inhomogeneous_rmhd_rho", "inhomogeneous_rmhd_s"]
PHYSICS = dict(vA=1.0, cs2_over_vA2=0.1, K_rho0=3.5, g=0.06, K_p0=1.0)


def _document(equation_name, spec, backend="numpy", tmax=8 / 1024):
    return dict(
        equations=dict(type=equation_name),
        grid=dict(Nx=16, Ny=16, Nz=16),
        backend=dict(backend=backend),
        physics=PHYSICS,
        time=dict(tmax=tmax, dt_init=1 / 1024, use_variable_dt=False),
        output=dict(t_out_scal=1 / 1024, t_out_spec=0.0, t_out_full=0.0),
        runtime=dict(progress_output_every=100),
        initial_condition=dict(type="zero"),
        forcing=dict(
            type="controlled_shell", use_forcing=True, forcing_seed=53,
            controlled_shell=dict(k_sigma_min=2.0, k_sigma_max=3.0, kz_index=1,
                                  interval_max=3 / 1024, branches=["plus"], plus=spec),
        ),
    )


def _settings(tmp_path, document):
    path = tmp_path / "controlled.input"
    path.write_text(dump_toml(document))
    return resolve_run_settings(runfile_path=path)


@pytest.mark.parametrize("equation_name", NOKIA_SETS)
@pytest.mark.parametrize("power", [False, True])
def test_nokia_controlled_forcing_closes_energy_budget(tmp_path, equation_name, power, production_backend):
    spec = (dict(control="constant_power", power_basis="elsasser", epsilon=0.002) if power else
            dict(control="target", target_basis="elsasser", target_quantity="rms", target_value=0.2,
                 target_scope="branch_total", startup_energy_fraction=0.25, tau_F=0.05))
    settings = _settings(tmp_path, _document(equation_name, spec, production_backend))
    events = []

    def observe(event, **kw):
        if event == "post_forcing":
            events.extend(kw["events"])

    run_simulation(settings, observer=observe)
    work = sum(event["work_z"] for event in events)
    assert len(events) == 3 and work > 0.0
    if power:
        assert work == pytest.approx(0.002 * settings.config.tmax, rel=2e-11)

    with (settings.output_dir / "scalar_diagnostics.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    # Each interval's energy change must equal the booked budget terms, and
    # the booked forcing must equal the controller's own work (no leftovers
    # from cross terms between the forced psi/omega and the other fields).
    booked_forcing = 0.0
    for before, after in zip(rows, rows[1:]):
        dt = float(after["time"]) - float(before["time"])
        d_energy = float(after["total_energy"]) - float(before["total_energy"])
        assert d_energy == pytest.approx(float(after["total_energy_rhs_total"]) * dt, abs=1e-12)
        booked_forcing += float(after["total_energy_rhs_forcing"]) * dt
    assert booked_forcing == pytest.approx(work, rel=1e-9)


@pytest.mark.parametrize("equation_name", NOKIA_SETS)
def test_nokia_branch_kick_changes_only_that_branch(tmp_path, equation_name):
    spec = dict(control="constant_power", power_basis="elsasser", epsilon=0.002)
    config = _settings(tmp_path, _document(equation_name, spec)).config
    backend = build_backend(config)
    grid = build_grid(config, backend)
    equation = get_equation_module(equation_name)
    state = State(grid, backend, field_names=config.field_names)
    rng = np.random.default_rng(7)
    for name in state.field_names:
        state[name][...] = rng.normal(size=grid.fourier_shape) + 1j * rng.normal(size=grid.fourier_shape)

    control = forcing_control.create_control(config, grid, backend, equation, build_dealias_mask(grid, backend))
    before = {name: state[name].copy() for name in state.field_names}
    minus_before = equation.forcing_branch_values(control, state, "minus", control.indices).copy()
    elsasser_before = elsasser_energies(state, grid, backend, equation)
    energy_before = equation.total_energy(state, grid, backend, config)

    equation.forcing_apply_gain(control, state, "plus", 1.5)

    # Only psi/omega change, and only in a way that leaves z- untouched.
    for name in state.field_names:
        if name not in ("psi", "omega"):
            assert np.array_equal(state[name], before[name])
    np.testing.assert_allclose(
        equation.forcing_branch_values(control, state, "minus", control.indices), minus_before, rtol=1e-12, atol=1e-12)
    elsasser_after = elsasser_energies(state, grid, backend, equation)
    assert elsasser_after["elsasser_energy_minus"] == pytest.approx(elsasser_before["elsasser_energy_minus"], rel=1e-12)
    assert elsasser_after["elsasser_energy_plus"] > elsasser_before["elsasser_energy_plus"]
    # Total energy = 0.5 (E+ + E-) + compressive part, so it moves by exactly
    # half the change in E+ (the package Elsasser normalization).
    d_energy = equation.total_energy(state, grid, backend, config) - energy_before
    d_plus = elsasser_after["elsasser_energy_plus"] - elsasser_before["elsasser_energy_plus"]
    assert d_energy == pytest.approx(0.5 * d_plus, rel=1e-10)
