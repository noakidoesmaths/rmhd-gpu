"""Additive phase forcing: work, branch isolation, legacy parity and replay."""
from copy import deepcopy
import hashlib
import json
import math

import numpy as np
import pytest

from rmhdgpu import forcing_control as core
from rmhdgpu.forcing_increment import apply, signed_amplitude
from rmhdgpu.tests.controlled_forcing_helpers import configuration, context


@pytest.mark.parametrize("gain", [0., 1e-20, 1e-10, .001, .02, .2])
@pytest.mark.parametrize("overlap", [-1., -.2, -1e-20, 0., 1e-20, .2, 1.])
@pytest.mark.parametrize("norm_ratio", [.9999998, 1., 1.0000002])
def test_signed_root_work_and_zero_gain_limit(gain, overlap, norm_ratio):
    amplitude = signed_amplitude(gain, overlap, norm_ratio)
    work = math.expm1(2*gain)
    assert amplitude*overlap >= 0
    assert abs(amplitude) <= math.sqrt(work/norm_ratio)*(1+4*np.finfo(float).eps)
    assert abs(norm_ratio*amplitude**2+2*overlap*amplitude-work) <= 8*np.finfo(float).eps*work
    if gain == 0:
        assert amplitude == 0
    if gain == 1e-20 and overlap < -.01:
        assert abs(amplitude) < 1e-18
        # The always-positive quadratic root would give a finite zero-work kick.
        assert (-overlap+math.sqrt(overlap**2+norm_ratio*work))/norm_ratio > .01


@pytest.mark.parametrize("args", [(-.1, 0., 1.), (.1, float("nan"), 1.),
    (.1, 0., 0.), (.1, 0., float("inf")), (1000., 0., 1.), (float("nan"), 0., 1.)])
def test_invalid_root_inputs(args):
    with pytest.raises(ValueError):
        signed_amplitude(*args)


@pytest.mark.parametrize("equation", ["alfvenic", "s09", "low_beta_stratified"])
@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("branch", ["plus", "minus"])
def test_positive_increment_work_and_branch_isolation(equation, precision, branch, production_backend):
    state, control = context(configuration(backend=production_backend,
        precision=precision, equation=equation, power=True))
    backend, eq = state.backend, control.equation
    eps = np.finfo(precision).eps
    opposite = "minus" if branch == "plus" else "plus"
    whole = {name: backend.to_numpy(state[name]).copy() for name in state.field_names}
    other = backend.to_numpy(eq.forcing_branch_values(control, state, opposite, control.indices)).copy()
    before = eq.forcing_branch_values(control, state, branch, control.indices).copy()
    report = apply(control, state, branch, dict(log_gain=.02), before)
    after = backend.to_numpy(eq.forcing_branch_values(control, state, branch, control.indices))
    old = backend.to_numpy(before).astype(np.complex128)
    weights = backend.to_numpy(control.weights).astype(float)
    pre = math.fsum(weights*abs(old)**2)
    post = math.fsum(weights*abs(after.astype(complex))**2)
    assert post == pytest.approx(pre*math.exp(.04), rel=32*eps)
    assert report["phase_actual_work_f"] == pytest.approx(post-pre, abs=32*np.finfo(float).eps*pre)
    assert abs(report["phase_work_identity_error_f"]) < 32*np.finfo(float).eps*pre
    assert report["phase_signed_amplitude"]*report["phase_overlap"] >= 0
    assert 0 < report["phase_negative_modal_work_fraction"] < 1
    assert report["phase_delta_rms"] > 0
    np.testing.assert_allclose(backend.to_numpy(eq.forcing_branch_values(
        control, state, opposite, control.indices)), other, rtol=32*eps,
        atol=32*eps*np.max(abs(other)))
    untouched = np.ones(state.grid.fourier_shape, dtype=bool)
    untouched.reshape(-1)[backend.to_numpy(control.indices)] = False
    active = set(control.fields.values())
    for name in state.field_names:
        actual = backend.to_numpy(state[name])
        np.testing.assert_array_equal(actual[untouched], whole[name][untouched])
        if name not in active:
            np.testing.assert_array_equal(actual, whole[name])


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_zero_identity_damping_parity_and_event_draw_schedule(precision, production_backend):
    config = configuration(backend=production_backend, precision=precision, both=False)
    state, control = context(config)
    plain = deepcopy(config)
    plain.controlled_shell.phase_model = "none"
    reference, ordinary = context(plain)
    backend = state.backend
    for name in state.field_names:
        assert backend.to_numpy(state[name]).tobytes() == backend.to_numpy(reference[name]).tobytes()
    rng = np.random.default_rng()
    rng.bit_generator.state = deepcopy(control.phase_rng["plus"].bit_generator.state)
    saved_rng = deepcopy(rng.bit_generator.state)
    assert core.advance(control, state, .004, .004) == []
    assert control.phase_rng["plus"].bit_generator.state == saved_rng
    # Use an exactly equal native target so the actual event requests zero gain.
    control.native["plus"]["target_f"] = core.perpendicular_shell_energy(control, state, "plus")
    saved = {name: backend.to_numpy(state[name]).tobytes() for name in state.field_names}
    zero, = core.advance(control, state, .006, .01)
    assert zero["log_gain"] == zero["phase_actual_work_f"] == zero["phase_delta_rms"] == 0
    assert zero["phase_raw_sha256"] == hashlib.sha256(rng.uniform(-np.pi, np.pi, control.mode_count).tobytes()).hexdigest()
    for name in state.field_names:
        assert backend.to_numpy(state[name]).tobytes() == saved[name]
        state[name] *= 2
        reference[name][...] = state[name]
    ordinary.native["plus"]["target_f"] = control.native["plus"]["target_f"]
    left, = core.advance(control, state, .01, .02)
    right, = core.advance(ordinary, reference, .01, .02)
    assert left["log_gain"] < 0 and left["phase_direction_used"] == 0
    for name in state.field_names:
        assert backend.to_numpy(state[name]).tobytes() == backend.to_numpy(reference[name]).tobytes()
    for key in ("gamma_applied", "log_gain", "shell_post_f", "work_f", "next_interval_target"):
        assert left[key] == right[key]
    assert left["phase_raw_sha256"] == hashlib.sha256(rng.uniform(-np.pi, np.pi, control.mode_count).tobytes()).hexdigest()


@pytest.mark.parametrize("model", ["none", "fixed_gain_angle"])
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_legacy_gain_arithmetic_and_explicit_singleton_parity(model, precision, production_backend):
    config = configuration(model=model, precision=precision, backend=production_backend)
    state, control = context(config)
    explicit = deepcopy(config)
    explicit.controlled_shell.kz_indices = [1]
    other, singleton = context(explicit)
    expected, original = context(config)
    for step in range(3):
        # Independently use the pre-extension event map and multiplicative update.
        for branch in original.settings.branches:
            density, nonzero, perpendicular = core._energy_measurement(original, expected, branch)
            event = core.control_event_map(getattr(original.settings, branch), original.native[branch],
                core.shell_energy(original, expected, branch),
                expected.backend.scalar_to_float(expected.backend.xp.sum(density)), .01,
                nonzero_kz=nonzero, perpendicular_shell=perpendicular)
            xp = expected.backend.xp
            factor = xp.exp(event["log_gain"]*(1-1j*xp.tan(original.angles[branch])))
            original.equation.forcing_apply_gain(original, expected, branch, factor)
        assert core.advance(control, state, .01, (step+1)*.01) == core.advance(singleton, other, .01, (step+1)*.01)
        for name in state.field_names:
            actual = state.backend.to_numpy(state[name]).tobytes()
            assert actual == other.backend.to_numpy(other[name]).tobytes()
            assert actual == expected.backend.to_numpy(expected[name]).tobytes()


@pytest.mark.parametrize("planes", [None, [1, 2, 3, 4, 5], [5]])
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_json_history_restores_pending_interval_rng_and_final_flush(planes, precision, production_backend):
    config = configuration(planes=planes, precision=precision, backend=production_backend, report=True)
    state, control = context(config)
    core.advance(control, state, .01, .01)
    assert core.advance(control, state, .002, .012) == []
    saved = json.loads(json.dumps(core.forcing_history(control)))
    other, restored = context(config, initialize=False)
    for name in state.field_names:
        other[name][...] = state[name]
    core.restore_forcing_history(restored, saved)
    for dt, final in ((.003, False), (.005, False), (.004, True)):
        assert core.advance(control, state, dt, .02, final=final) == core.advance(restored, other, dt, .02, final=final)
        for name in state.field_names:
            assert state.backend.to_numpy(state[name]).tobytes() == other.backend.to_numpy(other[name]).tobytes()
    assert core.forcing_history(control) == core.forcing_history(restored)
    for key in ("phase_rng_states", "phase_history"):
        broken = deepcopy(saved)
        broken.pop(key)
        with pytest.raises(ValueError):
            core.restore_forcing_history(restored, broken)


def test_full_circle_draws_are_reproducible_and_not_centered():
    state, control = context(configuration(both=False))
    control.native["plus"]["target_f"] = core.perpendicular_shell_energy(control, state, "plus")
    rng = np.random.default_rng()
    rng.bit_generator.state = deepcopy(control.phase_rng["plus"].bit_generator.state)
    samples = []
    for step in range(256):
        theta = rng.uniform(-np.pi, np.pi, control.mode_count)
        samples.extend(theta)
        event, = core.advance(control, state, .01, (step+1)*.01)
        assert event["phase_raw_sha256"] == hashlib.sha256(theta.tobytes()).hexdigest()
        assert event["phase_raw_cos_mean"] == float(np.mean(np.cos(theta)))
    theta = np.array(samples)
    assert theta.min() < -.99*np.pi and theta.max() > .99*np.pi
    for values in (np.cos(theta), np.sin(theta)):
        assert abs(values.mean()) < 6/math.sqrt(2*len(theta))
    assert control.phase_history["plus"]["zero_events"] == 256
