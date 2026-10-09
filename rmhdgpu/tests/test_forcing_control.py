"""Portable controller checks, independent of squash physics/configuration."""
from types import SimpleNamespace

import numpy as np
import pytest

from rmhdgpu.backend import build_backend
from rmhdgpu.grid import build_grid
from rmhdgpu.masks import build_dealias_mask
from rmhdgpu.state import State
from rmhdgpu import forcing_control as core, forcing_fields as fields


def context(velocity="omega", backend="numpy", *, power=False, scope="branch_total"):
    spec = (dict(control="constant_power", power_basis="elsasser", epsilon=.03) if power else
            dict(control="target", target_basis="elsasser", target_quantity="rms", target_value=.4,
                 target_scope=scope))
    config = SimpleNamespace(Nx=16, Ny=16, Nz=16, Lx=2*np.pi, Ly=2*np.pi, Lz=2*np.pi,
        backend=backend, fft_workers=2, real_dtype="float64", complex_dtype="complex128",
        vA=1.2, dealias=True, forcing_seed=37, field_names=[velocity, "psi", "passive"],
        controlled_shell=core.ControlledShellSettings(k_sigma_min=2, k_sigma_max=3,
            branches=["plus", "minus"], plus=spec, minus=spec, interval_max=.01))
    equation = SimpleNamespace(
        forcing_fields=lambda c: fields.alfvenic_fields(c, velocity=velocity),
        forcing_metric=fields.standard_metric, forcing_energy_factors=fields.standard_energy_factors,
        forcing_native_parameters=fields.standard_native_parameters,
        forcing_branch_values=fields.standard_branch_values,
        forcing_apply_gain=fields.standard_apply_gain, forcing_seed_branch=fields.standard_seed_branch,
        forcing_characteristic_speed=fields.standard_characteristic_speed,
        forcing_budget_work=fields.standard_budget_work,
        forcing_shell_density=fields.vorticity_shell_density,
        forcing_perpendicular_energy=fields.vorticity_perpendicular_energy,
        forcing_perpendicular_shell_energy=fields.vorticity_perpendicular_shell_energy,
        forcing_measurement=fields.vorticity_measurement)
    b = build_backend(config)
    grid = build_grid(config, b)
    state = State(grid, b, field_names=config.field_names)
    control = core.create_control(config, grid, b, equation, build_dealias_mask(grid, b))
    return state, control


@pytest.mark.parametrize("velocity", ["phi", "omega"])
def test_branch_energy_matches_real_vector_rms(velocity, production_backend):
    state, control = context(velocity, production_backend)
    rng = np.random.default_rng(105)
    phi = np.fft.rfftn(rng.normal(size=state.grid.real_shape))
    psi = np.fft.rfftn(rng.normal(size=state.grid.real_shape))
    k2 = state.backend.to_numpy(state.grid.kperp2)
    state[velocity][...] = state.backend.asarray(phi if velocity == "phi" else -k2*phi)
    state["psi"][...] = state.backend.asarray(psi)
    kx = 2*np.pi*np.fft.fftfreq(16, d=2*np.pi/16)[:, None, None]
    ky = 2*np.pi*np.fft.fftfreq(16, d=2*np.pi/16)[None, :, None]
    # Odd derivative of the perpendicular Nyquist coefficient is ambiguous;
    # use an independent Parseval sum of physical gradients instead.
    w = np.full(9, 2.); w[[0, -1]] = 1.
    for branch, sign in (("plus", -1), ("minus", 1)):
        zeta = phi + sign*psi
        expected = np.sum(w*(kx*kx+ky*ky)*abs(zeta)**2)/(4*16**6)
        assert core.branch_energy(control, state, branch) == pytest.approx(expected, rel=3e-14)


@pytest.mark.parametrize("velocity", ["phi", "omega"])
@pytest.mark.parametrize("branch", ["plus", "minus"])
def test_complex_gain_preserves_existing_opposite_and_unselected_fields(velocity, branch, production_backend):
    state, control = context(velocity, production_backend)
    core.initialize(control, state)
    state["passive"][...] = 3+2j
    original = {name: state.backend.to_numpy(state[name]).copy() for name in state.field_names}
    opposite = "minus" if branch == "plus" else "plus"
    before = state.backend.to_numpy(fields.standard_branch_values(control, state, opposite, control.indices))
    driven = state.backend.to_numpy(fields.standard_branch_values(control, state, branch, control.indices))
    gain = np.exp(.1-.03j)
    fields.standard_apply_gain(control, state, branch, gain)
    after = state.backend.to_numpy(fields.standard_branch_values(control, state, opposite, control.indices))
    actual = state.backend.to_numpy(fields.standard_branch_values(control, state, branch, control.indices))
    np.testing.assert_allclose(after, before, rtol=3e-14, atol=3e-14*max(abs(before)))
    np.testing.assert_allclose(actual, gain*driven, rtol=3e-14, atol=3e-14*max(abs(driven)))
    mask = np.ones(state.grid.fourier_shape, dtype=bool)
    mask.reshape(-1)[state.backend.to_numpy(control.indices)] = False
    for name in state.field_names:
        np.testing.assert_array_equal(state.backend.to_numpy(state[name])[mask], original[name][mask])
    np.testing.assert_array_equal(state.backend.to_numpy(state["passive"]), original["passive"])


@pytest.mark.parametrize("velocity", ["phi", "omega"])
@pytest.mark.parametrize("scope", ["branch_total", "branch_nonzero_kz", "shell", "perpendicular_shell"])
def test_target_startup_and_power_normalization(velocity, scope, production_backend):
    state, control = context(velocity, production_backend, scope=scope)
    core.initialize(control, state)
    assert core.branch_energy(control, state, "plus") == pytest.approx(.01*.4**2/4, rel=3e-14)
    state, control = context(velocity, production_backend, power=True)
    core.initialize(control, state)
    events = core.advance(control, state, .01, .01)
    for event in events:
        assert event["rate_cap_active"] == 0
        assert event["work_z"] == pytest.approx(.03*.01, rel=2e-11)


def test_wave_action_basis_is_not_silently_reinterpreted():
    _, control = context()
    settings = core.ControlledBranchSettings(control="constant_power", power_basis="wave_action", epsilon=1)
    with pytest.raises(ValueError, match="not wave_action"):
        fields.standard_native_parameters(settings, control.config, "plus")


def test_pending_interval_serialization():
    state, control = context()
    core.initialize(control, state)
    assert core.advance(control, state, .002, .002) == []
    history = core.forcing_history(control)
    _, other = context()
    core.restore_forcing_history(other, history)
    assert other.elapsed == .002 and other.initialized
    assert core.forcing_history(other) == history
