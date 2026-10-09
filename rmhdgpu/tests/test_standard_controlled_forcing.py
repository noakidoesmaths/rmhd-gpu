"""The ordinary input/driver path can control standard Alfvénic branches."""
import csv
from pathlib import Path

import numpy as np
import pytest

from rmhdgpu.backend import build_backend
from rmhdgpu.equations import get_equation_module
from rmhdgpu.fft import FFTManager
from rmhdgpu.grid import build_grid
from rmhdgpu.masks import build_dealias_mask
from rmhdgpu.run import run_simulation
from rmhdgpu.runfile import dump_toml, resolve_run_settings
from rmhdgpu.state import State
from rmhdgpu.steppers import evolve_until
from rmhdgpu.workspace import Workspace


@pytest.mark.parametrize("equation_name", ["alfvenic", "s09", "low_beta_stratified"])
@pytest.mark.parametrize("power", [False, True])
def test_both_standard_runners_and_signed_energy_work(tmp_path, monkeypatch, equation_name, power, production_backend):
    spec = (dict(control="constant_power", power_basis="elsasser", epsilon=.002) if power else
            dict(control="target", target_basis="elsasser", target_quantity="rms", target_value=.2,
                 target_scope="perpendicular_shell", startup_energy_fraction=.25, tau_F=.05))
    document = dict(equations=dict(type=equation_name),
        grid=dict(Nx=16, Ny=16, Nz=16), backend=dict(backend=production_backend),
        time=dict(tmax=8/1024, dt_init=1/1024, use_variable_dt=False),
        output=dict(t_out_scal=1/1024, t_out_spec=0., t_out_full=0.),
        runtime=dict(progress_output_every=100), initial_condition=dict(type="zero"),
        forcing=dict(type="controlled_shell", use_forcing=True, forcing_seed=53,
            controlled_shell=dict(k_sigma_min=2., k_sigma_max=3., kz_index=1,
                interval_max=3/1024, branches=["plus"], plus=spec)))
    path = tmp_path / "controlled.input"
    path.write_text(dump_toml(document))
    settings = resolve_run_settings(runfile_path=path)
    # Enforce separation from the stochastic engine in both public drivers.
    def forbidden(*a, **kw):
        raise AssertionError("Controlled forcing entered the stochastic path")
    monkeypatch.setattr("rmhdgpu.run.generate_forcing_kick", forbidden)
    monkeypatch.setattr("rmhdgpu.steppers.generate_forcing_kick", forbidden)
    final, events = {}, []
    def observe(event, **kw):
        if event == "post_forcing":
            events.extend(kw["events"])
            final.update({name: kw["backend"].to_numpy(kw["state"][name]).copy()
                          for name in kw["state"].field_names})
    run_simulation(settings, observer=observe)
    assert len(events) == 3
    assert [event["interval"] for event in events] == [3/1024, 3/1024, 2/1024]
    assert events[-1]["time"] == settings.config.tmax
    config = settings.config
    backend = build_backend(config)
    grid = build_grid(config, backend)
    fft = FFTManager(grid, backend)
    mask = build_dealias_mask(grid, backend)
    state = State(grid, backend, field_names=config.field_names)
    eq = get_equation_module(equation_name)
    workspace = Workspace(grid, backend)
    evolved, info = evolve_until(state, config.tmax, eq.ideal_rhs,
        eq.build_dissipation_operators(grid, config), params=config, fixed_dt=config.dt_init,
        rhs_kwargs=dict(grid=grid, fft=fft, workspace=workspace, params=config, dealias_mask=mask))
    for name in state.field_names:
        assert backend.to_numpy(evolved[name]).tobytes() == final[name].tobytes()
    assert info["forcing_plus_cumulative_work_f"] == sum(e["work_z"] for e in events)
    if power:
        assert sum(e["work_z"] for e in events) == pytest.approx(.002*config.tmax, rel=2e-11)
    with (settings.output_dir / "scalar_diagnostics.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert float(rows[-1]["total_energy_rhs_forcing"]) > 0
    assert all(not name.startswith("wave_action_energy") for name in rows[-1])


def test_no_implicit_wave_action_units_for_standard_equation(tmp_path):
    path = tmp_path / "bad.input"
    path.write_text(dump_toml(dict(equations=dict(type="alfvenic"), forcing=dict(
        type="controlled_shell", use_forcing=True, controlled_shell=dict(k_sigma_min=2., k_sigma_max=3.,
            branches=["plus"], plus=dict(control="constant_power", power_basis="wave_action", epsilon=.001))))))
    with pytest.raises(ValueError, match="not wave_action"):
        resolve_run_settings(runfile_path=path)


@pytest.mark.parametrize("example", ["controlled_target_s09", "controlled_power_alfvenic"])
def test_controlled_examples_round_trip_resolved_inputs(tmp_path, example):
    """A saved resolved document remains a complete, equivalent run input."""
    root = Path(__file__).resolve().parents[2]
    settings = resolve_run_settings(runfile_path=root / "examples" / f"{example}.input")
    path = tmp_path / "resolved.input"
    path.write_text(dump_toml(settings.resolved_document))
    restored = resolve_run_settings(runfile_path=path)
    assert restored.config.controlled_shell == settings.config.controlled_shell
    assert restored.config.forcing_seed == settings.config.forcing_seed
    assert restored.config.equation_set == settings.config.equation_set
    assert restored.config.equation_mode == "nonlinear"
    assert restored.config.tmax == settings.config.tmax
    assert restored.resolved_document["forcing"] == settings.resolved_document["forcing"]


@pytest.mark.parametrize("stochastic", [
    {"epsilon_plus": 0.0}, {"forcing_mode": "elsasser"},
    {"field_energy_injection_rates": {"psi": 0.0}},
])
def test_controlled_inputs_reject_explicit_stochastic_settings(tmp_path, stochastic):
    root = Path(__file__).resolve().parents[2]
    settings = resolve_run_settings(runfile_path=root / "examples/controlled_target_s09.input")
    document = settings.resolved_document
    document["forcing"].update(stochastic)
    path = tmp_path / "mixed.input"
    path.write_text(dump_toml(document))
    with pytest.raises(ValueError, match="Stochastic settings are incompatible"):
        resolve_run_settings(runfile_path=path)
