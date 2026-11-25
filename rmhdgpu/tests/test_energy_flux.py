"""Shell-by-shell energy rates (`rmhdgpu.diagnostics.energy_flux`)."""

from __future__ import annotations

import csv
from typing import Any

import numpy as np
import pytest

from rmhdgpu.backend import build_backend
from rmhdgpu.config import Config
from rmhdgpu.diagnostics.energy_flux import TERMS, energy_flux_spectra
from rmhdgpu.diagnostics.spectra import perpendicular_shell_spectrum
from rmhdgpu.equations import available_equation_sets, get_equation_module
from rmhdgpu.fft import FFTManager
from rmhdgpu.grid import build_grid
from rmhdgpu.masks import build_dealias_mask
from rmhdgpu.operators import lap_perp, poisson_bracket
from rmhdgpu.run import run_simulation
from rmhdgpu.runfile import dump_toml, resolve_run_settings
from rmhdgpu.state import State
from rmhdgpu.workspace import Workspace

# vA = 1: the drho energy weight of the rho set is only exact there (see its derived_parameters).
PHYSICS = {"vA": 1.0, "cs2_over_vA2": 0.4, "g": 0.35, "K_p0": 0.9, "K_rho0": 2.1}
NOKIA_SETS = ("inhomogeneous_rmhd_rho", "inhomogeneous_rmhd_s")


def _context(equation_set: str, backend_name: str = "numpy"):
    if backend_name == "cupy":
        cupy = pytest.importorskip("cupy")
        try:
            cupy.zeros(1)
        except Exception as exc:  # pragma: no cover - depends on the machine
            pytest.skip(f"GPU unavailable: {exc}")
    config = Config(equation_set=equation_set, Nx=12, Ny=10, Nz=8, backend=backend_name, **PHYSICS)
    backend = build_backend(config)
    grid = build_grid(config, backend)
    fft = FFTManager(grid, backend)
    return config, backend, grid, fft, Workspace(grid, backend), build_dealias_mask(grid, backend)


def _random_state(module: Any, backend: Any, grid: Any, fft: Any, mask: Any, seed: int = 20261125) -> State:
    """Dealiased random fields of order one in every evolved variable."""
    rng = np.random.default_rng(seed)
    state = State(grid, backend, field_names=module.FIELD_NAMES)
    for name in module.FIELD_NAMES:
        real = rng.standard_normal(grid.real_shape).astype(grid.real_dtype)
        state[name][...] = fft.r2c(backend.xp.asarray(real)) * mask
    return state


def _damping(module: Any, grid: Any) -> dict[str, Any]:
    """A different k_perp^2 damping rate on each field, so cross terms in the budget are exercised."""
    return {name: (1e-2 * (index + 1)) * grid.kperp2 for index, name in enumerate(module.FIELD_NAMES)}


def _flux(module, state, config, backend, grid, fft, workspace, mask):
    return energy_flux_spectra(state, grid=grid, fft=fft, workspace=workspace, backend=backend, params=config,
                               equation_module=module, linear_ops=_damping(module, grid), dealias_mask=mask)


@pytest.mark.parametrize("equation_set", available_equation_sets())
def test_ideal_rhs_is_linear_plus_quadratic(equation_set):
    """The split N = [R(f) + R(-f)]/2, L = [R(f) - R(-f)]/2 needs R = L + N exactly.

    Then R(2f) = 2 L(f) + 4 N(f). A cubic term or a constant would break this.
    """
    module = get_equation_module(equation_set)
    config, backend, grid, fft, workspace, mask = _context(equation_set)
    state = _random_state(module, backend, grid, fft, mask)

    def rhs(fields):
        return module.ideal_rhs(fields, grid, fft, workspace, config, dealias_mask=mask)

    r_plus = rhs(state)
    r_minus = rhs(state.zeros_like().linear_combination_([(-1.0, state)]))
    r_double = rhs(state.zeros_like().linear_combination_([(2.0, state)]))
    for name in module.FIELD_NAMES:
        linear = 0.5 * (r_plus[name] - r_minus[name])
        nonlinear = 0.5 * (r_plus[name] + r_minus[name])
        scale = np.max(np.abs(r_double[name])) + 1e-300
        assert np.max(np.abs(r_double[name] - (2 * linear + 4 * nonlinear))) < 1e-11 * scale, name


@pytest.mark.parametrize("equation_set", available_equation_sets())
def test_nonlinear_transfer_only_moves_energy(equation_set):
    """sum_k T(k) = 0 for the total energy and for z+ and z- separately, but T(k) itself is not zero."""
    module = get_equation_module(equation_set)
    config, backend, grid, fft, workspace, mask = _context(equation_set)
    state = _random_state(module, backend, grid, fft, mask)
    out = _flux(module, state, config, backend, grid, fft, workspace, mask)

    conserved = ["total", "z_plus", "z_minus"] if "psi" in module.FIELD_NAMES else ["total"]
    for partition in conserved:
        transfer = out[f"nonlinear_{partition}"]
        assert np.sum(np.abs(transfer)) > 0.0, partition
        assert abs(np.sum(transfer)) < 1e-11 * np.sum(np.abs(transfer)), partition


@pytest.mark.parametrize("equation_set", NOKIA_SETS)
def test_compressive_transfer_conserves_but_fields_exchange(equation_set):
    """The compressive energy (total - z+ - z-) is conserved by the brackets as a whole,
    while {psi, du_par}-type brackets move energy between the individual compressive fields."""
    module = get_equation_module(equation_set)
    config, backend, grid, fft, workspace, mask = _context(equation_set)
    state = _random_state(module, backend, grid, fft, mask)
    out = _flux(module, state, config, backend, grid, fft, workspace, mask)

    compressive = out["nonlinear_total"] - out["nonlinear_z_plus"] - out["nonlinear_z_minus"]
    assert abs(np.sum(compressive)) < 1e-11 * np.sum(np.abs(compressive))
    assert abs(np.sum(out["nonlinear_du_par"])) > 1e-6 * np.sum(np.abs(out["nonlinear_du_par"]))


@pytest.mark.parametrize("equation_set", available_equation_sets())
def test_shell_sums_match_the_scalar_budget(equation_set):
    """sum_k D(k) = total_energy_rhs_dissipation and sum_k L(k) = the linear drive, from the existing formulas."""
    module = get_equation_module(equation_set)
    config, backend, grid, fft, workspace, mask = _context(equation_set)
    state = _random_state(module, backend, grid, fft, mask)
    out = _flux(module, state, config, backend, grid, fft, workspace, mask)

    dissipation = module.total_energy_dissipation_rhs(state, grid, backend, _damping(module, grid), config)
    assert np.sum(out["dissipation_total"]) == pytest.approx(dissipation, rel=1e-11)
    assert dissipation < 0.0

    # Linear terms: the wave terms d_z only exchange energy within a mode, so
    # what is left summed over k is the background drive (zero without one).
    drive = module.total_energy_stratification_rhs(state, grid, backend, config) \
        if hasattr(module, "total_energy_stratification_rhs") else 0.0
    # Scale: the nonlinear rates, since L(k) can be exactly zero shell by shell (Alfvenic set).
    scale = np.sum(np.abs(out["linear_total"])) + abs(drive) + np.sum(np.abs(out["nonlinear_total"]))
    assert abs(np.sum(out["linear_total"]) - drive) < 1e-11 * scale


def test_transfer_matches_the_written_out_formula():
    """Alfvenic set, by hand: E_k = (1/2) k^2 (|phi|^2 + |psi|^2), so the bracket terms
    N_omega = -{phi, omega} + {psi, lap psi} and N_psi = -{phi, psi} change it at

        T_k = Re[ -conj(phi) N_omega + k^2 conj(psi) N_psi ]     (phi = -omega / k^2).
    """
    module = get_equation_module("alfvenic")
    config, backend, grid, fft, workspace, mask = _context("alfvenic")
    state = _random_state(module, backend, grid, fft, mask)
    out = _flux(module, state, config, backend, grid, fft, workspace, mask)

    phi = module.derive_phi_hat(state["omega"], grid)
    psi = state["psi"]

    def bracket(f, g):
        return poisson_bracket(f, g, grid, fft, workspace, mask=mask)

    n_omega = -bracket(phi, state["omega"]) + bracket(psi, lap_perp(psi, grid))
    n_psi = -bracket(phi, psi)
    density = np.real(-np.conj(phi) * n_omega + grid.kperp2 * np.conj(psi) * n_psi)
    _, expected = perpendicular_shell_spectrum(density, grid, backend)

    scale = np.max(np.abs(expected))
    assert scale > 0.0
    assert np.max(np.abs(out["nonlinear_total"] - expected)) < 1e-11 * scale


def test_pure_z_plus_state_has_no_alfvenic_transfer():
    """With z- = 0 (phi = -psi) nothing shears z+: T_z+(k) = 0 in every shell."""
    module = get_equation_module("inhomogeneous_rmhd_s")
    config, backend, grid, fft, workspace, mask = _context("inhomogeneous_rmhd_s")
    state = _random_state(module, backend, grid, fft, mask)
    scale = np.max(np.abs(_flux(module, state, config, backend, grid, fft, workspace, mask)["nonlinear_z_plus"]))

    state["omega"][...] = -lap_perp(state["psi"], grid)  # omega = lap(phi) with phi = -psi
    out = _flux(module, state, config, backend, grid, fft, workspace, mask)
    assert np.max(np.abs(out["nonlinear_z_plus"])) < 1e-11 * scale
    assert np.max(np.abs(out["nonlinear_z_minus"])) < 1e-11 * scale


def test_cupy_agrees_with_numpy():
    numpy_context = _context("inhomogeneous_rmhd_rho")
    cupy_context = _context("inhomogeneous_rmhd_rho", backend_name="cupy")
    module = get_equation_module("inhomogeneous_rmhd_rho")

    results = []
    for config, backend, grid, fft, workspace, mask in (numpy_context, cupy_context):
        state = _random_state(module, backend, grid, fft, mask)
        results.append(_flux(module, state, config, backend, grid, fft, workspace, mask))
    for key, values in results[0].items():
        np.testing.assert_allclose(results[1][key], values, rtol=1e-9, atol=1e-12 * np.max(np.abs(values)))


def test_run_writes_energy_flux_csv(tmp_path):
    dt = 1.0 / 1024
    module = get_equation_module("inhomogeneous_rmhd_rho")
    document = dict(
        output_dir="outputs",
        equations=dict(type="inhomogeneous_rmhd_rho"),
        grid=dict(Nx=12, Ny=12, Nz=8),
        backend=dict(backend="numpy"),
        physics=PHYSICS,
        time=dict(tmax=8 * dt, dt_init=dt, use_variable_dt=False),
        output=dict(t_out_scal=4 * dt, t_out_spec=4 * dt, t_out_full=0.0),
        initial_condition=dict(type="random_spectrum", parameters=dict(init_energy=0.5, seed=3)),
        dissipation={name: dict(nu_perp=2e-3, nu_par=0.0, n_perp=1, n_par=1) for name in module.FIELD_NAMES},
    )
    path = tmp_path / "flux.input"
    path.write_text(dump_toml(document))
    settings = resolve_run_settings(runfile_path=path)
    summary = run_simulation(settings)

    with open(summary["energy_flux_csv"], newline="") as handle:
        rows = list(csv.DictReader(handle))
    times = sorted({float(row["time"]) for row in rows})
    assert len(times) == 3  # t = 0, 4 dt, 8 dt
    quantities = {row["quantity"] for row in rows}
    for term in TERMS:
        for partition in ("total", "z_plus", "z_minus", "drho"):
            assert f"{term}_{partition}" in quantities

    for time in times:
        transfer = [float(row["value"]) for row in rows
                    if row["quantity"] == "nonlinear_total" and float(row["time"]) == time]
        assert abs(sum(transfer)) < 1e-10 * sum(abs(value) for value in transfer)
