"""Exact-identity tests for the DCF / ACR channel diagnostics.

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
from rmhdgpu.grid import build_grid
from rmhdgpu.masks import build_dealias_mask
from rmhdgpu.operators import dy, lap_perp
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
    correlators = channels.gradient_correlators(fields, grid, backend)
    split = channels.stratification_channels(correlators, module.derived_parameters(config))
    expected = module.total_energy_stratification_rhs(state, grid, backend, config)

    assert sum(split.values()) == pytest.approx(expected, rel=1.0e-11, abs=1.0e-14)


@pytest.mark.parametrize("equation_set", sorted(EQUATION_MODULES))
def test_elsasser_energies_sum_to_alfvenic_energy(equation_set: str) -> None:
    module = EQUATION_MODULES[equation_set]
    config, backend, grid, fft, _workspace, mask = _build_context(equation_set)
    state = _random_state(module, backend, grid, fft, mask)

    fields = module.channel_fields(state, grid, config)
    w_plus = channels.elsasser_energy(fields, grid, backend, sign=1)
    w_minus = channels.elsasser_energy(fields, grid, backend, sign=-1)
    expected = module.alfvenic_energy(state, grid, backend)

    assert w_plus + w_minus == pytest.approx(expected, rel=1.0e-12)


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
    correlators = channels.gradient_correlators(fields, grid, backend)
    work = channels.buoyancy_work(correlators, p)
    split = channels.stratification_channels(correlators, p)

    assert work["w_plus"] + work["w_minus"] == pytest.approx(split["acr_g"], rel=1.0e-12)


@pytest.mark.parametrize("equation_set", sorted(EQUATION_MODULES))
def test_dcf_matches_buoyancy_only_directional_derivative(equation_set: str) -> None:
    """`Q_DCF` must equal `-d_t W+` taken along the isolated buoyancy term.

    This is the test that matters: it pins the closed-form
    `-(g/2)(<drho dy(phi)> - <drho dy(psi)>)` against the actual `-g dy(drho)`
    term in `omega_t`, so a sign or factor error in either cannot pass.
    """

    module = EQUATION_MODULES[equation_set]
    config, backend, grid, fft, _workspace, mask = _build_context(equation_set)
    state = _random_state(module, backend, grid, fft, mask)
    p = module.derived_parameters(config)

    fields = module.channel_fields(state, grid, config)

    # An RHS state holding only the buoyancy coupling from `omega_t`.
    buoyancy_rhs = state.zeros_like()
    buoyancy_rhs.fill_zero()
    buoyancy_rhs["omega"][...] = -p.g * dy(fields.drho_hat, grid)
    rhs_fields = channels.rhs_channel_fields(buoyancy_rhs, grid)

    correlators = channels.gradient_correlators(fields, grid, backend)
    closed_form = channels.buoyancy_work(correlators, p)

    for name, sign in (("w_plus", 1), ("w_minus", -1)):
        measured = channels.elsasser_energy_rhs_budget(
            fields,
            rhs_fields,
            grid,
            backend,
            sign=sign,
        )
        assert measured == pytest.approx(closed_form[name], rel=1.0e-11, abs=1.0e-15)


@pytest.mark.parametrize("equation_set", sorted(EQUATION_MODULES))
def test_elsasser_dissipation_sums_to_alfvenic_dissipation(equation_set: str) -> None:
    """`W+` and `W-` damping must add up to the Alfvenic part of `d_t E`.

    Uses unequal `omega` and `psi` operators so the cross term that mixes `W+`
    and `W-` is genuinely nonzero and has to cancel in the sum.
    """

    module = EQUATION_MODULES[equation_set]
    config, backend, grid, fft, _workspace, mask = _build_context(equation_set)
    state = _random_state(module, backend, grid, fft, mask)

    linear_ops = {
        "omega": 0.03 * grid.kperp2,
        "psi": 0.07 * grid.kperp2,
    }
    fields = module.channel_fields(state, grid, config)
    total = sum(
        channels.elsasser_dissipation_rhs(fields, grid, backend, linear_ops, sign=sign)
        for sign in (1, -1)
    )

    xp = backend.xp
    expected_density = (
        -linear_ops["omega"] * grid.kperp2 * (xp.abs(fields.phi_hat) ** 2)
        - linear_ops["psi"] * grid.kperp2 * (xp.abs(fields.psi_hat) ** 2)
    )
    from rmhdgpu.fourier_diagnostics import modal_average

    expected = modal_average(expected_density, grid, backend)
    assert total == pytest.approx(expected, rel=1.0e-12)


@pytest.mark.parametrize("equation_set", sorted(EQUATION_MODULES))
def test_kperp_mean_is_a_sensible_outer_scale(equation_set: str) -> None:
    """`<k_perp>` must be positive and inside the resolved perpendicular range."""

    module = EQUATION_MODULES[equation_set]
    config, backend, grid, fft, _workspace, mask = _build_context(equation_set)
    state = _random_state(module, backend, grid, fft, mask)

    fields = module.channel_fields(state, grid, config)
    kperp_mean = channels.elsasser_kperp_mean(fields, grid, backend, sign=1)
    kperp_max = float(backend.scalar_to_float(backend.xp.max(backend.xp.sqrt(grid.kperp2))))

    assert 0.0 < kperp_mean <= kperp_max
