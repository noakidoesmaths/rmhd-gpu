from __future__ import annotations

import numpy as np
import pytest

from rmhdgpu.backend import build_backend
from rmhdgpu.config import Config
from rmhdgpu.diagnostics import compressive_channels as channels
from rmhdgpu.equations import low_beta_stratified, s09
from rmhdgpu.fft import FFTManager
from rmhdgpu.grid import build_grid
from rmhdgpu.initconds import (
    build_initial_state,
    get_initial_condition_builder,
    list_initial_condition_types,
    low_beta_stratified_mode_state,
)
from rmhdgpu.initconds.eigenmodes_s09 import alfven_mode_state
from rmhdgpu.masks import build_dealias_mask
from rmhdgpu.operators import inv_lap_perp, lap_perp


def _build_context() -> tuple[Config, object, object, FFTManager, object]:
    config = Config(Nx=8, Ny=8, Nz=8, backend="numpy")
    backend = build_backend(config)
    grid = build_grid(config, backend)
    fft = FFTManager(grid, backend)
    mask = build_dealias_mask(grid, backend)
    return config, backend, grid, fft, mask


def test_known_initial_conditions_are_registered() -> None:
    assert {
        "zero",
        "alfven_mode",
        "aw_packet",
        "random_spectrum",
        "low_beta_stratified_mode",
        "single_fourier_mode",
    } <= set(list_initial_condition_types())


def test_unknown_initial_condition_gives_helpful_error() -> None:
    with pytest.raises(ValueError) as excinfo:
        get_initial_condition_builder("does_not_exist")

    message = str(excinfo.value)
    assert "Unknown initial condition type 'does_not_exist'" in message
    for name in ("zero", "alfven_mode", "aw_packet", "random_spectrum", "single_fourier_mode"):
        assert name in message


def test_zero_initial_condition() -> None:
    config, backend, grid, fft, mask = _build_context()
    state = build_initial_state(
        "zero",
        grid=grid,
        backend=backend,
        fft=fft,
        dealias_mask=mask,
        field_names=config.field_names,
        params=config,
    )

    for name in state.field_names:
        np.testing.assert_allclose(backend.to_numpy(state[name]), 0.0)


def test_alfven_mode_initial_condition() -> None:
    config, backend, grid, fft, mask = _build_context()
    state = build_initial_state(
        "alfven_mode",
        parameters={"k_indices": [1, 1, 1], "amplitude": 0.2, "branch": "z_minus"},
        grid=grid,
        backend=backend,
        fft=fft,
        dealias_mask=mask,
        field_names=config.field_names,
        params=config,
    )

    psi_hat = backend.to_numpy(state["psi"])
    omega_hat = backend.to_numpy(state["omega"])

    assert np.isfinite(psi_hat).all()
    assert np.count_nonzero(np.abs(psi_hat) > 0.0) == 1
    np.testing.assert_allclose(omega_hat, backend.to_numpy(lap_perp(state["psi"], grid)))


@pytest.mark.parametrize("equation_set", ["s09", "inhomogeneous_rmhd_rho"])
def test_alfven_mode_branches_are_named_after_the_elsasser_wave(equation_set: str) -> None:
    """z_plus (the default) leaves only z+ = u_perp - b_perp; z_minus leaves only z-."""

    config = Config(
        equation_set=equation_set, Nx=8, Ny=8, Nz=8, backend="numpy",
        vA=1.3, cs2_over_vA2=0.4, g=0.35, K_p0=0.9, K_rho0=2.1,
    )
    backend = build_backend(config)
    grid = build_grid(config, backend)
    fft = FFTManager(grid, backend)
    mask = build_dealias_mask(grid, backend)

    energies = {}
    for branch in (None, "z_plus", "z_minus"):
        parameters = {"k_indices": [1, 2, 1], "amplitude": 0.2}
        if branch is not None:
            parameters["branch"] = branch
        state = build_initial_state(
            "alfven_mode", parameters=parameters, grid=grid, backend=backend, fft=fft,
            dealias_mask=mask, field_names=config.field_names, params=config,
        )
        z_plus_hat, z_minus_hat = channels.elsasser_potentials(
            inv_lap_perp(state["omega"], grid), state["psi"],
        )
        energies[branch] = (
            channels.elsasser_energy(z_plus_hat, grid, backend),
            channels.elsasser_energy(z_minus_hat, grid, backend),
        )

    assert energies[None] == energies["z_plus"]
    w_plus, w_minus = energies["z_plus"]
    assert w_plus > 0.0 and w_minus <= 1.0e-28 * w_plus
    w_plus, w_minus = energies["z_minus"]
    assert w_minus > 0.0 and w_plus <= 1.0e-28 * w_minus


@pytest.mark.parametrize("old, new", [("plus", "z_minus"), ("minus", "z_plus")])
def test_old_alfven_branch_names_say_which_new_name_to_use(old: str, new: str) -> None:
    """The old names meant the opposite waves, so they fail rather than flip silently."""

    config, backend, grid, fft, mask = _build_context()
    with pytest.raises(ValueError, match=f"branch = '{new}'"):
        build_initial_state(
            "alfven_mode", parameters={"branch": old}, grid=grid, backend=backend, fft=fft,
            dealias_mask=mask, field_names=config.field_names, params=config,
        )
    with pytest.raises(ValueError, match=f"branch = '{new}'"):
        alfven_mode_state(
            grid=grid, backend=backend, field_names=config.field_names,
            k_indices=(1, 1, 1), branch=old, params=config,
        )
    with pytest.raises(ValueError, match="'z_plus' or 'z_minus'"):
        alfven_mode_state(
            grid=grid, backend=backend, field_names=config.field_names,
            k_indices=(1, 1, 1), branch="up", params=config,
        )


def test_aw_packet_initial_condition() -> None:
    config, backend, grid, fft, mask = _build_context()
    state = build_initial_state(
        "aw_packet",
        grid=grid,
        backend=backend,
        fft=fft,
        dealias_mask=mask,
        field_names=config.field_names,
        params=config,
    )

    psi_hat = backend.to_numpy(state["psi"])
    psi_real = backend.to_numpy(fft.c2r(state["psi"]))

    assert np.isfinite(psi_hat).all()
    assert np.isfinite(psi_real).all()
    assert np.max(np.abs(psi_real)) > 0.0
    np.testing.assert_allclose(backend.to_numpy(state["omega"]), backend.to_numpy(lap_perp(state["psi"], grid)))


def test_single_fourier_mode_initial_condition_is_reproducible_and_single_mode() -> None:
    config, backend, grid, fft, mask = _build_context()
    parameters = {"k_indices": [1, -2, 1], "amplitude": 0.05, "seed": 123}

    state = build_initial_state(
        "single_fourier_mode",
        parameters=parameters,
        grid=grid,
        backend=backend,
        fft=fft,
        dealias_mask=mask,
        field_names=config.field_names,
        params=config,
    )
    repeated = build_initial_state(
        "single_fourier_mode",
        parameters=parameters,
        grid=grid,
        backend=backend,
        fft=fft,
        dealias_mask=mask,
        field_names=config.field_names,
        params=config,
    )
    different_seed = build_initial_state(
        "single_fourier_mode",
        parameters={**parameters, "seed": 456},
        grid=grid,
        backend=backend,
        fft=fft,
        dealias_mask=mask,
        field_names=config.field_names,
        params=config,
    )

    wrapped_index = (1, grid.Ny - 2, 1)
    different_fields = 0
    for name in config.field_names:
        field = backend.to_numpy(state[name])
        repeated_field = backend.to_numpy(repeated[name])
        changed_field = backend.to_numpy(different_seed[name])
        field_real = backend.to_numpy(fft.c2r(state[name]))

        assert np.count_nonzero(np.abs(field) > 0.0) == 1
        assert tuple(np.argwhere(np.abs(field) > 0.0)[0]) == wrapped_index
        assert np.isfinite(field_real).all()
        assert np.max(np.abs(field_real)) > 0.0
        np.testing.assert_allclose(field, repeated_field)
        if not np.allclose(field, changed_field):
            different_fields += 1

    assert different_fields > 0


def test_single_fourier_mode_rejects_modes_at_or_above_half_resolution() -> None:
    config, backend, grid, fft, mask = _build_context()

    with pytest.raises(ValueError, match=r"\|kx\| < Nx//2"):
        build_initial_state(
            "single_fourier_mode",
            parameters={"k_indices": [grid.Nx // 2, 1, 1], "amplitude": 0.05, "seed": 1},
            grid=grid,
            backend=backend,
            fft=fft,
            dealias_mask=mask,
            field_names=config.field_names,
            params=config,
        )


def test_random_spectrum_initial_condition() -> None:
    config, backend, grid, fft, mask = _build_context()
    state = build_initial_state(
        "random_spectrum",
        grid=grid,
        backend=backend,
        fft=fft,
        dealias_mask=mask,
        field_names=config.field_names,
        params=config,
    )

    for name in config.field_names:
        field = backend.to_numpy(state[name])
        assert np.isfinite(field).all()
        assert np.max(np.abs(field)) > 0.0
    np.testing.assert_allclose(s09.total_energy(state, grid, backend, config), 0.75, atol=1.0e-12, rtol=1.0e-12)


def test_random_spectrum_initial_condition_parameter_overrides() -> None:
    config, backend, grid, fft, mask = _build_context()
    default_state = build_initial_state(
        "random_spectrum",
        grid=grid,
        backend=backend,
        fft=fft,
        dealias_mask=mask,
        field_names=config.field_names,
        params=config,
    )
    override_state = build_initial_state(
        "random_spectrum",
        parameters={"seed": 21, "alpha": 1.5, "init_energy": 0.12},
        grid=grid,
        backend=backend,
        fft=fft,
        dealias_mask=mask,
        field_names=config.field_names,
        params=config,
    )

    assert np.isfinite(backend.to_numpy(override_state["psi"])).all()
    assert not np.allclose(backend.to_numpy(default_state["psi"]), backend.to_numpy(override_state["psi"]))
    np.testing.assert_allclose(s09.total_energy(override_state, grid, backend, config), 0.12, atol=1.0e-12, rtol=1.0e-12)


def test_low_beta_eigenmode_helper_is_separate_from_registry() -> None:
    config = Config(equation_set="low_beta_stratified", Nx=8, Ny=8, Nz=8, backend="numpy")
    backend = build_backend(config)
    grid = build_grid(config, backend)
    state = low_beta_stratified_mode_state(
        grid=grid,
        backend=backend,
        field_names=config.field_names,
        k_indices=[0, 1, 0],
        amplitude=0.2,
        mode="unstable_growing",
        params=config,
    )

    assert state.field_names == ["psi", "omega", "a"]
    assert np.max(np.abs(backend.to_numpy(state["omega"]))) > 0.0
    assert np.max(np.abs(backend.to_numpy(state["a"]))) > 0.0


def test_alfven_mode_initial_condition_energy_matches_amplitude_squared() -> None:
    config, backend, grid, fft, mask = _build_context()
    amplitude = 0.2
    state = build_initial_state(
        "alfven_mode",
        parameters={"k_indices": [1, 1, 1], "amplitude": amplitude, "branch": "z_minus"},
        grid=grid,
        backend=backend,
        fft=fft,
        dealias_mask=mask,
        field_names=config.field_names,
        params=config,
    )

    np.testing.assert_allclose(
        s09.total_energy(state, grid, backend, config),
        amplitude**2,
        atol=1.0e-12,
        rtol=1.0e-12,
    )


def test_low_beta_mode_initial_condition_energy_matches_amplitude_squared_for_positive_N2() -> None:
    config = Config(equation_set="low_beta_stratified", Nx=8, Ny=8, Nz=8, backend="numpy", N2=0.25)
    backend = build_backend(config)
    grid = build_grid(config, backend)
    fft = FFTManager(grid, backend)
    mask = build_dealias_mask(grid, backend)
    amplitude = 0.2

    state = build_initial_state(
        "low_beta_stratified_mode",
        parameters={"k_indices": [0, 1, 0], "amplitude": amplitude, "mode": "unstable_growing"},
        grid=grid,
        backend=backend,
        fft=fft,
        dealias_mask=mask,
        field_names=config.field_names,
        params=config,
    )

    np.testing.assert_allclose(
        low_beta_stratified.total_energy(state, grid, backend, config),
        amplitude**2,
        atol=1.0e-12,
        rtol=1.0e-12,
    )
