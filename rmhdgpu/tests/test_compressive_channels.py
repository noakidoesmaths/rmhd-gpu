"""Exact-identity tests for the Elsasser-wave and DCF / CCR / ACR channel diagnostics.

These pin the channel diagnostics against the equations themselves; no closure
or phenomenology is involved, so the tolerances are round-off level. The
closure comparison (measured `Q_DCF` versus `W+ vA K^damp`) is a scan, not a
test, and lives in the plotting script instead.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from rmhdgpu.backend import build_backend
from rmhdgpu.config import Config
from rmhdgpu.diagnostics import compressive_channels as channels
from rmhdgpu.equations import rmhd_by_nokia_rho, rmhd_by_nokia_s
from rmhdgpu.fft import FFTManager
from rmhdgpu.fourier_diagnostics import modal_average
from rmhdgpu.grid import build_grid
from rmhdgpu.masks import build_dealias_mask
from rmhdgpu.operators import dx, dy, inv_lap_perp, lap_perp
from rmhdgpu.state import State
from rmhdgpu.workspace import Workspace


EQUATION_MODULES = {
    "inhomogeneous_rmhd_rho": rmhd_by_nokia_rho,
    "inhomogeneous_rmhd_s": rmhd_by_nokia_s,
}

# Deliberately asymmetric background gradients so no channel is accidentally
# zero: `K_b0 = g/vA^2 - chi*K_p0/gamma` vanishes for some parameter choices
# (including the ones in `examples/nokias_inhomo_s.input`), which would hide a
# bug in the `acr_B` weights.
PHYSICS = {
    "vA": 1.3,
    "cs2_over_vA2": 0.4,
    "g": 0.35,
    "K_p0": 0.9,
    "K_rho0": 2.1,
}

# The channel columns, in the order `channel_scalar_diagnostics` writes them.
CHANNEL_COLUMNS = [
    "w_plus", "w_minus", "k_perp_plus", "k_perp_minus", "k_prl_plus", "k_prl_minus",
    "w_plus_align", "w_minus_align", "u_perp_rms", "b_perp_rms", "V_rho_x", "q_dcf", "q_ccr_source",
    "N_sq", "vA", "acr_B", "acr_g", "acr_th",
]


def _build_context(equation_set: str) -> tuple[Config, Any, Any, FFTManager, Workspace, Any]:
    config = Config(
        equation_set=equation_set,
        Nx=12,
        Ny=12,
        Nz=12,
        backend="numpy",
        **PHYSICS,
    )
    backend = build_backend(config)
    grid = build_grid(config, backend)
    fft = FFTManager(grid, backend)
    workspace = Workspace(grid, backend)
    mask = build_dealias_mask(grid, backend)
    return config, backend, grid, fft, workspace, mask


def _random_state(module: Any, backend: Any, grid: Any, fft: FFTManager, mask: Any) -> State:
    """Return a deterministic multimode state with every field populated."""

    state = State(grid, backend, field_names=module.FIELD_NAMES)
    rng = np.random.default_rng(20260908)
    for index, name in enumerate(module.FIELD_NAMES):
        real_field = rng.standard_normal(grid.real_shape) * (0.3 + 0.1 * index)
        field_hat = fft.r2c(backend.xp.asarray(real_field.astype(grid.real_dtype, copy=False)))
        field_hat *= mask
        state[name][...] = field_hat
    # `omega` is a Laplacian, so give it that structure rather than white noise.
    state["omega"][...] = lap_perp(state["omega"], grid)
    return state


def _single_mode_context(backend_name: str) -> tuple[Any, Any, FFTManager]:
    if backend_name == "scipy_cpu":
        pytest.importorskip("scipy.fft")
    if backend_name == "cupy":
        cupy = pytest.importorskip("cupy")
        try:
            cupy.zeros(1)
        except Exception as exc:
            pytest.skip(f"GPU unavailable: {exc}")
    config = Config(Nx=8, Ny=8, Nz=8, backend=backend_name)
    backend = build_backend(config)
    grid = build_grid(config, backend)
    return backend, grid, FFTManager(grid, backend)


def test_elsasser_potentials_follow_the_paper_convention() -> None:
    """z+ = u_perp - b_perp has potential phi - psi; z- = u_perp + b_perp has phi + psi."""

    phi, psi = np.array([3.0, 1.0]), np.array([0.5, 2.0])
    z_plus, z_minus = channels.elsasser_potentials(phi, psi)
    np.testing.assert_array_equal(z_plus, [2.5, -1.0])
    np.testing.assert_array_equal(z_minus, [3.5, 3.0])


@pytest.mark.parametrize("equation_set", sorted(EQUATION_MODULES))
def test_acr_channels_sum_to_stratification_source(equation_set: str) -> None:
    """The three ACR channels must reproduce the module's own `Y` source.

    This also proves the `drho` and `s` formulations agree: both modules are
    checked against the shared `drho`-form channel split.
    """

    module = EQUATION_MODULES[equation_set]
    config, backend, grid, fft, _workspace, mask = _build_context(equation_set)
    state = _random_state(module, backend, grid, fft, mask)

    fields = module.channel_fields(state, grid, config)
    fluxes = channels.fluxes_and_channels(fields, grid, backend, module.derived_parameters(config))
    expected = module.total_energy_stratification_rhs(state, grid, backend, config)

    total = sum(fluxes[name] for name in channels.ACR_CHANNELS)
    assert total == pytest.approx(expected, rel=1.0e-11, abs=1.0e-14)


@pytest.mark.parametrize("equation_set", sorted(EQUATION_MODULES))
def test_elsasser_energies_sum_to_alfvenic_energy(equation_set: str) -> None:
    module = EQUATION_MODULES[equation_set]
    config, backend, grid, fft, _workspace, mask = _build_context(equation_set)
    state = _random_state(module, backend, grid, fft, mask)

    fields = module.channel_fields(state, grid, config)
    z_plus_hat, z_minus_hat = channels.elsasser_potentials(fields.phi_hat, fields.psi_hat)
    w_plus = channels.elsasser_energy(z_plus_hat, grid, backend)
    w_minus = channels.elsasser_energy(z_minus_hat, grid, backend)
    expected = module.alfvenic_energy(state, grid, backend)

    assert w_plus + w_minus == pytest.approx(expected, rel=1.0e-12)
    values = module.conserved_quantity_values(state, grid=grid, backend=backend, params=config)
    assert values["w_plus"] == pytest.approx(w_plus, rel=1.0e-14)
    assert values["w_minus"] == pytest.approx(w_minus, rel=1.0e-14)


@pytest.mark.parametrize("equation_set", sorted(EQUATION_MODULES))
def test_buoyancy_work_sums_to_acr_g(equation_set: str) -> None:
    """Buoyancy work on `W+` plus `W-` is the background-potential ACR channel.

    Physically: the gravity coupling is a direct exchange between background
    potential energy and the Alfvenic fluctuations, so nothing is lost between
    the `W±` split and the total-energy source term.
    """

    module = EQUATION_MODULES[equation_set]
    config, backend, grid, fft, _workspace, mask = _build_context(equation_set)
    state = _random_state(module, backend, grid, fft, mask)
    p = module.derived_parameters(config)

    fields = module.channel_fields(state, grid, config)
    fluxes = channels.fluxes_and_channels(fields, grid, backend, p)

    total = fluxes["w_plus_buoyancy"] + fluxes["w_minus_buoyancy"]
    assert total == pytest.approx(fluxes["acr_g"], rel=1.0e-12)


@pytest.mark.parametrize("equation_set", sorted(EQUATION_MODULES))
def test_dcf_matches_buoyancy_only_directional_derivative(equation_set: str) -> None:
    """`Q_DCF` must equal `-d_t W+` taken along the isolated buoyancy term.

    This is the test that matters: it pins the closed form
    `d_t W± = (g/2) <drho z±_x>` against the actual `-g dy(drho)`
    term in `omega_t`, so a sign or factor error in either cannot pass. It also
    pins the signs of the saved `q_dcf` and `q_ccr_source` columns.
    """

    module = EQUATION_MODULES[equation_set]
    config, backend, grid, fft, _workspace, mask = _build_context(equation_set)
    state = _random_state(module, backend, grid, fft, mask)
    p = module.derived_parameters(config)
    fields = module.channel_fields(state, grid, config)

    # Buoyancy alone: omega_t = -g dy(drho), so phi_t = inv_lap_perp(omega_t), and psi_t = 0.
    phi_t_hat = inv_lap_perp(-p.g * dy(fields.drho_hat, grid), grid)
    z_plus_t_hat, z_minus_t_hat = channels.elsasser_potentials(phi_t_hat, 0.0 * fields.psi_hat)
    z_plus_hat, z_minus_hat = channels.elsasser_potentials(fields.phi_hat, fields.psi_hat)
    rate_plus = channels.elsasser_energy_rate(z_plus_hat, z_plus_t_hat, grid, backend)
    rate_minus = channels.elsasser_energy_rate(z_minus_hat, z_minus_t_hat, grid, backend)

    fluxes = channels.fluxes_and_channels(fields, grid, backend, p)
    assert rate_plus == pytest.approx(fluxes["w_plus_buoyancy"], rel=1.0e-11, abs=1.0e-15)
    assert rate_minus == pytest.approx(fluxes["w_minus_buoyancy"], rel=1.0e-11, abs=1.0e-15)

    saved = channels.channel_scalar_diagnostics(fields, grid, backend, p)
    assert saved["q_dcf"] == pytest.approx(-rate_plus, rel=1.0e-11, abs=1.0e-15)
    assert saved["q_ccr_source"] == pytest.approx(-rate_minus, rel=1.0e-11, abs=1.0e-15)


@pytest.mark.parametrize("equation_set", sorted(EQUATION_MODULES))
def test_elsasser_dissipation_sums_to_alfvenic_dissipation(equation_set: str) -> None:
    """`W+` and `W-` damping must add up to the Alfvenic part of `d_t E`.

    Uses unequal `omega` and `psi` operators so the cross term that mixes `W+`
    and `W-` is genuinely nonzero and has to cancel in the sum. Goes through
    `elsasser_budgets`, the path that writes the `w_*_rhs_dissipation` columns.
    """

    module = EQUATION_MODULES[equation_set]
    config, backend, grid, fft, _workspace, mask = _build_context(equation_set)
    state = _random_state(module, backend, grid, fft, mask)
    p = module.derived_parameters(config)

    linear_ops = {
        "omega": 0.03 * grid.kperp2,
        "psi": 0.07 * grid.kperp2,
    }
    fields = module.channel_fields(state, grid, config)
    fluxes = channels.fluxes_and_channels(fields, grid, backend, p)
    budgets = channels.elsasser_budgets(fields, fluxes, grid, backend, linear_ops=linear_ops)
    total = sum(budgets[name]["rhs_terms"]["dissipation"] for name in ("w_plus", "w_minus"))

    xp = backend.xp
    expected_density = (
        -linear_ops["omega"] * grid.kperp2 * (xp.abs(fields.phi_hat) ** 2)
        - linear_ops["psi"] * grid.kperp2 * (xp.abs(fields.psi_hat) ** 2)
    )
    expected = modal_average(expected_density, grid, backend)
    assert total == pytest.approx(expected, rel=1.0e-12)


@pytest.mark.parametrize("equation_set", sorted(EQUATION_MODULES))
def test_kperp_mean_is_a_sensible_outer_scale(equation_set: str) -> None:
    """`<k_perp>` must be positive and inside the resolved perpendicular range."""

    module = EQUATION_MODULES[equation_set]
    config, backend, grid, fft, _workspace, mask = _build_context(equation_set)
    state = _random_state(module, backend, grid, fft, mask)

    fields = module.channel_fields(state, grid, config)
    z_plus_hat, _ = channels.elsasser_potentials(fields.phi_hat, fields.psi_hat)
    kperp_mean = channels.elsasser_kperp_mean(z_plus_hat, grid, backend)
    kperp_max = float(backend.scalar_to_float(backend.xp.max(backend.xp.sqrt(grid.kperp2))))

    assert 0.0 < kperp_mean <= kperp_max


@pytest.mark.parametrize("backend_name", ["numpy", "scipy_cpu", "cupy"])
@pytest.mark.parametrize("wave, amplitude, amplitude_t", [
    # phi = 3 cos, psi = 0.5 cos and damping rates 0.2 (omega), 0.7 (psi), so
    # f = phi ∓ psi = (3 ∓ 0.5) cos and f_t = (-0.6 ± 0.35) cos.
    ("z_plus", 2.5, -0.25),
    ("z_minus", 3.5, -0.95),
])
def test_elsasser_damping_matches_single_mode(backend_name, wave, amplitude, amplitude_t):
    """Pin each wave's damping and normalization to an analytic mode, k = (1, 2, 1)."""

    backend, grid, fft = _single_mode_context(backend_name)
    cos = backend.xp.cos(grid.x[:, None, None] + 2 * grid.y[None, :, None] + grid.z[None, None, :])
    phi_hat, psi_hat = fft.r2c(3 * cos), fft.r2c(0.5 * cos)
    potentials = channels.elsasser_potentials(phi_hat, psi_hat)
    rates = channels.elsasser_potentials(-0.2 * phi_hat, -0.7 * psi_hat)
    index = 0 if wave == "z_plus" else 1

    # W = |k_perp|^2 A^2/8 and d_t W = |k_perp|^2 A A_t/4, with |k_perp|^2 = 5.
    assert channels.elsasser_energy(potentials[index], grid, backend) == pytest.approx(
        5 * amplitude**2 / 8
    )
    assert channels.elsasser_energy_rate(
        potentials[index], rates[index], grid, backend,
    ) == pytest.approx(5 * amplitude * amplitude_t / 4)


@pytest.mark.parametrize("backend_name", ["numpy", "scipy_cpu", "cupy"])
@pytest.mark.parametrize("plus_k, expected", [
    ((1, 2), 2 / np.sqrt(5)),
    ((0, 1), 1.0),  # k along y, so z = zhat x grad f points along x.
    ((3, 0), 0.0),  # k along x, so z points along y.
])
def test_x_alignment_matches_single_modes(backend_name, plus_k, expected):
    """Each wave sees only its own mode, with rms(z_x)/rms(|z|) = |k_y|/|k_perp|."""

    backend, grid, fft = _single_mode_context(backend_name)
    x, y, z = grid.x[:, None, None], grid.y[None, :, None], grid.z[None, None, :]
    plus = 3 * backend.xp.cos(plus_k[0] * x + plus_k[1] * y + z)
    minus = 0.5 * backend.xp.cos(x + y + 2 * z)  # Isotropic in (k_x, k_y): 1/sqrt(2).
    # Build phi and psi so that phi - psi = plus and phi + psi = minus.
    z_plus_hat, z_minus_hat = channels.elsasser_potentials(
        fft.r2c(0.5 * (plus + minus)), fft.r2c(0.5 * (minus - plus)),
    )

    assert channels.elsasser_x_alignment(z_plus_hat, grid, backend) == pytest.approx(expected, abs=1.0e-12)
    assert channels.elsasser_x_alignment(z_minus_hat, grid, backend) == pytest.approx(
        1 / np.sqrt(2), rel=1.0e-12,
    )
    assert channels.elsasser_x_alignment(0 * z_plus_hat, grid, backend) == 0.0


@pytest.mark.parametrize("equation_set", sorted(EQUATION_MODULES))
def test_x_alignment_matches_real_space_rms(equation_set):
    """The saved Fourier-space column must equal rms(z_x)/rms(|z|) measured in real space."""

    module = EQUATION_MODULES[equation_set]
    config, backend, grid, fft, _workspace, mask = _build_context(equation_set)
    state = _random_state(module, backend, grid, fft, mask)
    fields = module.channel_fields(state, grid, config)
    xp = backend.xp

    for potential_hat in channels.elsasser_potentials(fields.phi_hat, fields.psi_hat):
        z_x = -fft.c2r(dy(potential_hat, grid))
        z_y = fft.c2r(dx(potential_hat, grid))
        expected = float(xp.sqrt(xp.mean(z_x**2) / xp.mean(z_x**2 + z_y**2)))
        measured = channels.elsasser_x_alignment(potential_hat, grid, backend)
        assert measured == pytest.approx(expected, rel=1.0e-12)


@pytest.mark.parametrize("equation_set", sorted(EQUATION_MODULES))
def test_perp_rms_matches_real_space_and_elsasser_energies(equation_set):
    """u_perp_rms and b_perp_rms equal real-space RMS, and u^2 + b^2 = 2 (W+ + W-)."""

    module = EQUATION_MODULES[equation_set]
    config, backend, grid, fft, _workspace, mask = _build_context(equation_set)
    state = _random_state(module, backend, grid, fft, mask)
    fields = module.channel_fields(state, grid, config)
    p = module.derived_parameters(config)
    xp = backend.xp
    diagnostics = channels.channel_scalar_diagnostics(fields, grid, backend, p)

    for column, potential_hat in (("u_perp_rms", fields.phi_hat), ("b_perp_rms", fields.psi_hat)):
        gx = fft.c2r(dx(potential_hat, grid))
        gy = fft.c2r(dy(potential_hat, grid))
        expected = float(xp.sqrt(xp.mean(gx**2 + gy**2)))
        assert diagnostics[column] == pytest.approx(expected, rel=1.0e-12)
    total = diagnostics["u_perp_rms"]**2 + diagnostics["b_perp_rms"]**2
    assert total == pytest.approx(2 * (diagnostics["w_plus"] + diagnostics["w_minus"]), rel=1.0e-12)


@pytest.mark.parametrize("equation_set", sorted(EQUATION_MODULES))
def test_fluxes_match_real_space_averages(equation_set):
    """Every flux equals its Eq. (79) average measured in real space.

    Also pins the saved V_rho_x column, the z± split of V_rho_x, and the two
    ACR channels written directly in terms of the fluxes.
    """

    module = EQUATION_MODULES[equation_set]
    config, backend, grid, fft, _workspace, mask = _build_context(equation_set)
    state = _random_state(module, backend, grid, fft, mask)
    fields = module.channel_fields(state, grid, config)
    p = module.derived_parameters(config)
    xp = backend.xp
    fluxes = channels.fluxes_and_channels(fields, grid, backend, p)
    diagnostics = channels.channel_scalar_diagnostics(fields, grid, backend, p)

    drho = fft.c2r(fields.drho_hat)
    du_par = fft.c2r(fields.du_par_hat)
    db_par = fft.c2r(fields.db_par_hat)
    u_x = -fft.c2r(dy(fields.phi_hat, grid))
    b_x = -fft.c2r(dy(fields.psi_hat, grid))
    expected = {
        "V_rho_x": float(xp.mean(drho * u_x)),
        "V_psi_x": float(xp.mean(db_par * u_x - du_par * b_x / p.vA)),
        "V_rho_x_plus": float(xp.mean(drho * (u_x - b_x))) / 2,
        "V_rho_x_minus": float(xp.mean(drho * (u_x + b_x))) / 2,
    }
    for name, value in expected.items():
        assert value != 0.0, name
        assert fluxes[name] == pytest.approx(value, rel=1.0e-12), name

    assert diagnostics["V_rho_x"] == fluxes["V_rho_x"]
    assert fluxes["V_rho_x_plus"] + fluxes["V_rho_x_minus"] == pytest.approx(fluxes["V_rho_x"], rel=1.0e-12)
    assert fluxes["acr_g"] == pytest.approx(p.g * fluxes["V_rho_x"], rel=1.0e-12)
    assert fluxes["acr_B"] == pytest.approx(-p.vA**2 * p.K_b0 * fluxes["V_psi_x"], rel=1.0e-12)


@pytest.mark.parametrize("equation_set", sorted(EQUATION_MODULES))
def test_pure_z_plus_carries_all_of_v_rho(equation_set):
    """With z- = 0, V_rho_x = <drho z+_x>/2 exactly, the paper's closure form of Eq. (79)."""

    module = EQUATION_MODULES[equation_set]
    config, backend, grid, fft, _workspace, mask = _build_context(equation_set)
    state = _random_state(module, backend, grid, fft, mask)
    state["omega"][...] = -lap_perp(state["psi"], grid)  # phi = -psi: pure z+.
    fields = module.channel_fields(state, grid, config)
    fluxes = channels.fluxes_and_channels(fields, grid, backend, module.derived_parameters(config))

    assert fluxes["V_rho_x"] != 0.0
    assert fluxes["V_rho_x_plus"] == pytest.approx(fluxes["V_rho_x"], rel=1.0e-12)
    assert abs(fluxes["V_rho_x_minus"]) <= 1.0e-14 * abs(fluxes["V_rho_x"])


@pytest.mark.parametrize("equation_set", sorted(EQUATION_MODULES))
def test_background_drives_are_the_ideal_rhs_coefficients(equation_set):
    """In a pure z+ state with no compressive fields, d_t f = F_f u_x for every field.

    Here `ideal_rhs` reduces to the background-gradient terms alone, so each
    compressive RHS, in the units of `slaved_field_units`, must equal the
    Eq. (46) drive times u_x = -dy(phi). The s set evolves s, so drho_t is
    derived from s_t and db_par_t exactly as the set derives drho from s.
    """

    module = EQUATION_MODULES[equation_set]
    config, backend, grid, fft, workspace, mask = _build_context(equation_set)
    xp = backend.xp
    rng = np.random.default_rng(7)
    psi_hat = fft.r2c(xp.asarray(rng.standard_normal(grid.real_shape).astype(grid.real_dtype)))
    psi_hat *= mask * (grid.kperp2 > 0.0)
    state = State(grid, backend, field_names=module.FIELD_NAMES)
    state["psi"][...] = psi_hat
    state["omega"][...] = -lap_perp(psi_hat, grid)  # phi = -psi: pure z+.
    fields = module.channel_fields(state, grid, config)
    z_plus_hat, z_minus_hat = channels.elsasser_potentials(fields.phi_hat, fields.psi_hat)
    w_plus = channels.elsasser_energy(z_plus_hat, grid, backend)
    assert channels.elsasser_energy(z_minus_hat, grid, backend) <= 1.0e-28 * w_plus

    rhs = module.ideal_rhs(state, grid, fft, workspace, config, dealias_mask=mask)
    if equation_set == "inhomogeneous_rmhd_s":
        drho_t = module.derive_drho_hat(rhs["s"], rhs["db_par"], config)
    else:
        drho_t = rhs["drho"]
    p = module.derived_parameters(config)
    drives = channels.background_drives(p)
    units = channels.slaved_field_units(p)
    u_x = -dy(fields.phi_hat, grid)
    scale = float(xp.max(xp.abs(u_x)))

    for name, field_t in (("drho", drho_t), ("du_par", rhs["du_par"]), ("db_par", rhs["db_par"])):
        assert drives[name] != 0.0, name
        error = float(xp.max(xp.abs(field_t / units[name] - drives[name] * u_x)))
        assert error <= 1.0e-12 * scale, name


def test_background_drive_values_and_units():
    # vA = 2, chi = 3, g = 4, K_p0 = 0.5, K_rho0 = 0.3, so alpha = 3/4 and K_b0 = 0.1.
    physics = {"vA": 2.0, "cs2_over_vA2": 3.0, "g": 4.0, "K_p0": 0.5, "K_rho0": 0.3}
    p = rmhd_by_nokia_rho.derived_parameters(physics)
    assert channels.background_drives(p) == pytest.approx({"drho": -0.05, "du_par": -0.1, "db_par": 0.2})
    assert channels.slaved_field_units(p) == pytest.approx({"drho": 1.0, "du_par": 2.0, "db_par": 0.75})
    # The density drive is written without dividing by g, so it exists when g = 0.
    p = rmhd_by_nokia_rho.derived_parameters({**physics, "g": 0.0})
    assert channels.background_drives(p)["drho"] == pytest.approx(-0.3)


@pytest.mark.parametrize("equation_set", sorted(EQUATION_MODULES))
def test_every_channel_column_is_documented(equation_set):
    module = EQUATION_MODULES[equation_set]
    config, backend, grid, fft, _workspace, mask = _build_context(equation_set)
    state = _random_state(module, backend, grid, fft, mask)
    diagnostics = channels.channel_scalar_diagnostics(
        module.channel_fields(state, grid, config), grid, backend, module.derived_parameters(config),
    )
    assert list(diagnostics) == CHANNEL_COLUMNS
    assert list(channels.CHANNEL_SCALAR_DIAGNOSTIC_INFO) == CHANNEL_COLUMNS
    assert set(diagnostics) <= set(module.SCALAR_DIAGNOSTIC_INFO)


@pytest.mark.parametrize("equation_set", sorted(EQUATION_MODULES))
def test_scalar_diagnostics_keep_the_budget_columns(equation_set):
    """The instantaneous channel columns sit beside the W± and ACR budget columns."""

    module = EQUATION_MODULES[equation_set]
    config, backend, grid, fft, workspace, mask = _build_context(equation_set)
    state = _random_state(module, backend, grid, fft, mask)
    diagnostics = module.compute_equation_scalar_diagnostics(
        state, grid=grid, fft=fft, backend=backend, params=config, workspace=workspace,
    )
    for name in ("w_plus", "w_minus"):
        for term in ("buoyancy", "dissipation", "forcing", "total"):
            assert f"{name}_rhs_{term}" in diagnostics
    for channel in ("acr_B", "acr_g", "acr_th"):
        assert diagnostics[f"total_energy_rhs_{channel}"] == pytest.approx(diagnostics[channel], rel=1.0e-12)
    assert diagnostics["w_plus_rhs_buoyancy"] == pytest.approx(-diagnostics["q_dcf"], rel=1.0e-12)
    assert not any(name.startswith("corr_") for name in diagnostics)
