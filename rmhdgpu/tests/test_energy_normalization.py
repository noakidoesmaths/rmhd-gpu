from __future__ import annotations

import numpy as np

from rmhdgpu.backend import build_backend
from rmhdgpu.config import Config
from rmhdgpu.diagnostics.alfvenic import elsasser_energies
from rmhdgpu.diagnostics.spectra import elsasser_perpendicular_spectra
from rmhdgpu.equations import s09
from rmhdgpu.grid import build_grid
from rmhdgpu.state import State


def _build_context(
    *,
    vA: float = 1.7,
    cs2_over_vA2: float = 0.5,
) -> tuple[Config, object, object]:
    config = Config(
        Nx=8,
        Ny=8,
        Nz=8,
        backend="numpy",
        vA=vA,
        cs2_over_vA2=cs2_over_vA2,
    )
    backend = build_backend(config)
    grid = build_grid(config, backend)
    return config, backend, grid


def _single_mode_state(
    config: Config,
    backend: object,
    grid: object,
    *,
    field_name: str,
    amplitude: complex,
    k_indices: tuple[int, int, int] = (1, 1, 1),
) -> State:
    state = State(grid, backend, field_names=s09.FIELD_NAMES)
    ix, iy, iz = k_indices
    state[field_name][ix, iy, iz] = amplitude
    return state


def _modal_weight(grid: object, iz: int) -> float:
    if iz == 0:
        return 1.0
    if grid.Nz % 2 == 0 and iz == grid.Nz // 2:
        return 1.0
    return 2.0


def _single_mode_prefactor(
    grid: object,
    *,
    amplitude: complex,
    iz: int,
) -> float:
    normalization = float(np.prod(grid.real_shape) ** 2)
    return _modal_weight(grid, iz) * abs(amplitude) ** 2 / normalization


def _configure_uniform_dissipation(config: Config, *, nu_perp: float, n_perp: int = 2) -> None:
    for field_name in s09.FIELD_NAMES:
        config.dissipation[field_name].update(
            {
                "nu_perp": nu_perp,
                "nu_par": 0.0,
                "n_perp": n_perp,
                "n_par": 1,
            }
        )


def test_slow_energy_weights_match_linear_invariant() -> None:
    config, backend, grid = _build_context(vA=1.7, cs2_over_vA2=0.5)
    amplitude = 1.25 + 0.75j
    ix, iy, iz = (1, 1, 1)
    prefactor = _single_mode_prefactor(grid, amplitude=amplitude, iz=iz)
    alpha = s09.derived_parameters(config).alpha

    upar_state = _single_mode_state(
        config,
        backend,
        grid,
        field_name="upar",
        amplitude=amplitude,
        k_indices=(ix, iy, iz),
    )
    dbpar_state = _single_mode_state(
        config,
        backend,
        grid,
        field_name="dbpar",
        amplitude=amplitude,
        k_indices=(ix, iy, iz),
    )

    np.testing.assert_allclose(
        s09.total_energy(upar_state, grid, backend, config),
        0.5 * prefactor,
        atol=1.0e-14,
        rtol=1.0e-14,
    )
    np.testing.assert_allclose(
        s09.total_energy(dbpar_state, grid, backend, config),
        0.5 * (1.0 / alpha) * prefactor,
        atol=1.0e-14,
        rtol=1.0e-14,
    )


def test_entropy_energy_weight_matches_expected_gamma_factor() -> None:
    config, backend, grid = _build_context(vA=1.3, cs2_over_vA2=0.8)
    amplitude = 0.9 - 0.4j
    iz = 1
    gamma = 5.0 / 3.0
    chi = float(config.cs2_over_vA2)
    prefactor = _single_mode_prefactor(grid, amplitude=amplitude, iz=iz)
    state = _single_mode_state(config, backend, grid, field_name="s", amplitude=amplitude)

    np.testing.assert_allclose(
        s09.total_energy(state, grid, backend, config),
        0.5 * chi / (gamma**2 * (gamma - 1.0)) * prefactor,
        atol=1.0e-14,
        rtol=1.0e-14,
    )


def test_dissipation_budget_matches_weighted_energy_decay_for_single_modes() -> None:
    config, backend, grid = _build_context(vA=1.6, cs2_over_vA2=0.4)
    _configure_uniform_dissipation(config, nu_perp=0.03, n_perp=2)
    linear_ops = s09.build_dissipation_operators(grid, config)
    amplitude = 0.8 + 0.6j
    ix, iy, iz = (1, 1, 1)
    prefactor = _single_mode_prefactor(grid, amplitude=amplitude, iz=iz)
    alpha = s09.derived_parameters(config).alpha
    gamma = 5.0 / 3.0
    chi = float(config.cs2_over_vA2)
    entropy_weight = chi / (gamma**2 * (gamma - 1.0))
    kperp2 = float(grid.kperp2[ix, iy, iz])

    for field_name, sector_weight in (
        ("psi", kperp2),
        ("upar", 1.0),
        ("dbpar", 1.0 / alpha),
        ("s", entropy_weight),
    ):
        state = _single_mode_state(
            config,
            backend,
            grid,
            field_name=field_name,
            amplitude=amplitude,
            k_indices=(ix, iy, iz),
        )
        damping_rate = float(linear_ops[field_name][ix, iy, iz])
        expected_rhs = -sector_weight * damping_rate * prefactor
        np.testing.assert_allclose(
            s09.total_energy_dissipation_rhs(state, grid, backend, linear_ops, config),
            expected_rhs,
            atol=1.0e-14,
            rtol=1.0e-14,
            err_msg=f"Incorrect dissipation budget for single-mode field {field_name}.",
        )


def test_total_energy_rhs_dissipation_consistent_with_finite_difference() -> None:
    config, backend, grid = _build_context(vA=1.0, cs2_over_vA2=0.35)
    _configure_uniform_dissipation(config, nu_perp=0.02, n_perp=2)
    linear_ops = s09.build_dissipation_operators(grid, config)

    state = State(grid, backend, field_names=s09.FIELD_NAMES)
    state["psi"][1, 1, 1] = 0.7 + 0.2j
    state["omega"][2, 1, 1] = -0.3 + 0.4j
    state["upar"][1, 2, 1] = 0.5 - 0.1j
    state["dbpar"][2, 2, 1] = -0.4 + 0.6j
    state["s"][1, 1, 2] = 0.2 + 0.3j

    rhs0 = s09.total_energy_dissipation_rhs(state, grid, backend, linear_ops, config)
    energy0 = s09.total_energy(state, grid, backend, config)
    dt = 1.0e-6

    evolved = state.copy()
    for field_name in state.field_names:
        evolved[field_name][...] = state[field_name] * np.exp(-backend.to_numpy(linear_ops[field_name]) * dt)

    energy1 = s09.total_energy(evolved, grid, backend, config)
    finite_difference = (energy1 - energy0) / dt

    np.testing.assert_allclose(
        finite_difference,
        rhs0,
        atol=1.0e-9,
        rtol=1.0e-6,
    )


def test_elsasser_energy_and_spectra_match_alfvenic_energy_identity() -> None:
    config, backend, grid = _build_context()
    state = State(grid, backend, field_names=s09.FIELD_NAMES)
    rng = np.random.default_rng(611)
    state["psi"][...] = rng.standard_normal(grid.fourier_shape) + 1j * rng.standard_normal(
        grid.fourier_shape
    )
    state["omega"][...] = rng.standard_normal(grid.fourier_shape) + 1j * rng.standard_normal(
        grid.fourier_shape
    )
    state["psi"][...] *= ~grid.mask_kperp0
    state["omega"][...] *= ~grid.mask_kperp0

    energies = elsasser_energies(state, grid, backend)
    spectra = elsasser_perpendicular_spectra(state, grid, backend)
    alfvenic = s09.alfvenic_energy(state, grid, backend)

    np.testing.assert_allclose(
        0.5 * (energies["elsasser_energy_plus"] + energies["elsasser_energy_minus"]),
        alfvenic,
        atol=1.0e-14,
        rtol=1.0e-14,
    )
    np.testing.assert_allclose(
        np.sum(spectra["z_plus"]),
        energies["elsasser_energy_plus"],
        atol=1.0e-14,
        rtol=1.0e-14,
    )
    np.testing.assert_allclose(
        np.sum(spectra["z_minus"]),
        energies["elsasser_energy_minus"],
        atol=1.0e-14,
        rtol=1.0e-14,
    )
