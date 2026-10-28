"""Parallel actuator unions retain one target and unique feedback measurements."""
from copy import deepcopy
import json
import math

import numpy as np
import pytest

from rmhdgpu import forcing_control as core
from rmhdgpu.forcing_increment import apply
from rmhdgpu.runfile import dump_toml, resolve_run_settings
from rmhdgpu.tests.controlled_forcing_helpers import configuration, context


@pytest.mark.parametrize("values", [[], [0], [True], [1., 2], [1, 1], [2, 1], [1, 16], [11], "1,2"])
def test_invalid_or_unrepresented_actuator_planes(values):
    with pytest.raises(ValueError):
        context(configuration(planes=values))


@pytest.mark.parametrize("report", [1, "yes", None])
def test_report_switch_requires_bool(report):
    with pytest.raises(ValueError, match="report_parallel_planes"):
        configuration(report=report)


@pytest.mark.parametrize("model", ["none", "fixed_gain_angle", "random_increment_phase"])
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_equal_startup_energy_and_one_total_target(model, precision, production_backend):
    config = configuration(model=model, precision=precision, backend=production_backend,
                           planes=[1, 2, 3, 4, 5], report=True)
    state, control = context(config)
    backend = state.backend
    eps = np.finfo(precision).eps
    indices = backend.to_numpy(control.indices)
    iz = indices % state.grid.fourier_shape[-1]
    np.testing.assert_array_equal(iz, np.repeat([1, 2, 3, 4, 5], control.perpendicular_mode_count))
    for branch in control.settings.branches:
        density = backend.to_numpy(core.shell_density(control, state, branch))
        expected = control.native[branch]["startup_f"]
        np.testing.assert_allclose(density.reshape(5, -1).sum(axis=1), expected/5, rtol=32*eps)
        assert core.branch_energy(control, state, branch) == pytest.approx(expected, rel=32*eps)
        # Independent physical-space gradient energy includes each conjugate once.
        native = backend.to_numpy(control.equation.forcing_branch_values(control, state,
            branch, (slice(None), slice(None), slice(None))))
        k2 = backend.to_numpy(state.grid.kperp2)
        potential = np.divide(-native, k2, out=np.zeros_like(native), where=k2 > 0)
        dx = np.fft.irfftn(1j*backend.to_numpy(state.grid.kx)*potential, s=state.grid.real_shape, axes=(0, 1, 2))
        dy = np.fft.irfftn(1j*backend.to_numpy(state.grid.ky)*potential, s=state.grid.real_shape, axes=(0, 1, 2))
        assert np.mean(dx**2+dy**2)/4 == pytest.approx(expected, rel=32*eps)


def test_perpendicular_feedback_counts_each_mode_once(production_backend):
    state, many = context(configuration(backend=production_backend, planes=[1, 2, 3, 4, 5], report=True))
    other, one = context(configuration(backend=production_backend, planes=[1], report=True))
    rng = np.random.default_rng(813)
    for name in state.field_names:
        values = np.fft.rfftn(rng.normal(size=state.grid.real_shape))
        values[0, 0, :] = 0
        state[name][...] = state.backend.asarray(values)
        other[name][...] = other.backend.asarray(values)
    for branch in ("plus", "minus"):
        assert core.perpendicular_shell_energy(many, state, branch) == core.perpendicular_shell_energy(one, other, branch)
        a, b = core.diagnostics(many, state), core.diagnostics(one, other)
        for suffix in ("total_energy_f", "perpendicular_shell_energy_f", "nz0_energy_f", "nz1_energy_f"):
            assert a[f"forcing_{branch}_"+suffix] == b[f"forcing_{branch}_"+suffix]


@pytest.mark.parametrize("gain", [.02, -.02, 0.])
@pytest.mark.parametrize("empty_plane", [False, True])
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_multiplane_aggregate_work_and_plane_partition(gain, empty_plane, precision, production_backend):
    state, control = context(configuration(backend=production_backend, precision=precision,
        planes=[1, 2, 3, 4, 5], report=True, both=False))
    backend, eq = state.backend, control.equation
    eps = np.finfo(precision).eps
    before = eq.forcing_branch_values(control, state, "plus", control.indices).copy()
    before *= backend.asarray(np.repeat([0. if empty_plane else .1, .3, 1., 2., 3.], control.perpendicular_mode_count), dtype=state.grid.real_dtype)
    eq.forcing_seed_branch(control, state, "plus", before)
    before = eq.forcing_branch_values(control, state, "plus", control.indices).copy()
    old = backend.to_numpy(before).astype(complex)
    weights = backend.to_numpy(control.weights).astype(float)
    pre = math.fsum(weights*abs(old)**2)
    report = apply(control, state, "plus", dict(log_gain=gain), before)
    after = backend.to_numpy(eq.forcing_branch_values(control, state, "plus", control.indices)).astype(complex)
    post = math.fsum(weights*abs(after)**2)
    assert post == pytest.approx(pre*math.exp(2*gain), rel=32*eps)
    assert report["phase_actual_work_f"] == pytest.approx(pre*math.expm1(2*gain), abs=32*eps*pre)
    assert report["plane_work_sum_f"] == pytest.approx(report["phase_actual_work_f"], abs=32*np.finfo(float).eps*pre)
    for nz in control.kz_indices:
        assert report[f"nz{nz}_linear_work_f"]+report[f"nz{nz}_quadratic_work_f"] == pytest.approx(report[f"nz{nz}_work_f"], abs=32*np.finfo(float).eps*pre)
    if gain > 0:
        np.testing.assert_allclose([report[f"nz{nz}_direction_norm2_f"] for nz in control.kz_indices], pre/5, rtol=32*eps)
        assert np.ptp([report[f"nz{nz}_work_f"] for nz in control.kz_indices]) > 1e-8*pre
        if empty_plane:
            assert report["nz1_actuator_post_f"] > 0
    elif gain == 0:
        np.testing.assert_array_equal(after, old)


@pytest.mark.parametrize("scope", ["branch_total", "branch_nonzero_kz", "shell", "perpendicular_shell"])
def test_reporting_does_not_change_feedback_or_actuation(scope, production_backend):
    config = configuration(backend=production_backend, scope=scope, both=False, planes=[1, 2, 3])
    state, ordinary = context(config)
    reported = deepcopy(config)
    reported.controlled_shell.report_parallel_planes = True
    other, reporting = context(reported)
    # Populate an unforced kz=0 pair, outside the perpendicular band.
    for sample in (state, other):
        sample["omega"][1, 0, 0] = sample["omega"][-1, 0, 0] = 1e4
    expected, = core.advance(ordinary, state, .01, .01)
    actual, = core.advance(reporting, other, .01, .01)
    for key, value in expected.items():
        assert actual[key] == value
    for name in state.field_names:
        np.testing.assert_array_equal(state.backend.to_numpy(state[name]), other.backend.to_numpy(other[name]))
    assert other.backend.scalar_to_float(other["omega"][1, 0, 0].real) == 1e4


def test_input_round_trip_defaults_reference_and_metadata(tmp_path):
    doc = dict(equations=dict(type="s09"), grid=dict(Nx=16, Ny=16, Nz=32),
        forcing=dict(type="controlled_shell", use_forcing=True, forcing_seed=37,
            controlled_shell=dict(k_sigma_min=2., k_sigma_max=3., branches=["plus"],
                plus=dict(control="target", target_scope="perpendicular_shell",
                    target_basis="elsasser", target_quantity="rms", target_value=.4))))
    path = tmp_path/"forcing.input"
    path.write_text(dump_toml(doc))
    legacy = resolve_run_settings(runfile_path=path)
    assert legacy.config.controlled_shell.phase_model == "none"
    assert legacy.config.controlled_shell.kz_index == 1
    assert legacy.config.controlled_shell.kz_indices is None
    assert not legacy.config.controlled_shell.report_parallel_planes
    _, baseline = context(legacy.config)
    for planes in ([1], [1, 2, 3, 4, 5], [5]):
        doc["forcing"]["controlled_shell"].update(phase_model="random_increment_phase",
            kz_indices=planes, report_parallel_planes=True)
        path.write_text(dump_toml(doc))
        settings = resolve_run_settings(runfile_path=path)
        path.write_text(dump_toml(settings.resolved_document))
        restored = resolve_run_settings(runfile_path=path)
        assert restored.config.controlled_shell == settings.config.controlled_shell
        state, control = context(settings.config)
        assert control.native == baseline.native
        assert control.interval_max == baseline.interval_max
        indices = state.backend.to_numpy(control.indices)
        assert set(indices % state.grid.fourier_shape[-1]) == set(planes)
        core.write_metadata(control, tmp_path)
        metadata = json.loads((tmp_path/"forcing_metadata.json").read_text())
        assert metadata["parallel_actuator"]["reference_harmonic"] == 1
        assert metadata["parallel_actuator"]["actuator_harmonics"] == planes
        assert "post-control" in metadata["random_increment_phase"]["sign_conditioning"]
