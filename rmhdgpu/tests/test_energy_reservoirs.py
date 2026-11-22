"""Exact-identity tests for the six energy reservoirs of the inhomogeneous RMHD sets.

    E = W+ + W- + W_par^u + W_par^B + W^p + W_s

Each reservoir's booked RHS terms must equal its exact rate of change along
`ideal_rhs` (plus damping), the exchange terms must cancel between reservoirs,
and the background drives must add up to the ACR channels. The z-profile tests
check the same budgets at each z, where the vA d_z transport terms survive.
"""

from __future__ import annotations

import csv
from typing import Any

import numpy as np
import pytest

from rmhdgpu.backend import build_backend
from rmhdgpu.config import Config
from rmhdgpu.diagnostics import compressive_channels as channels
from rmhdgpu.diagnostics.z_profiles import (
    Z_PROFILE_DISSIPATION_INFO,
    Z_PROFILE_INFO,
    Z_PROFILE_LOCAL_BUDGET_INFO,
    Z_PROFILE_LOCAL_BUDGETS,
)
from rmhdgpu.equations import rmhd_by_nokia_rho, rmhd_by_nokia_s
from rmhdgpu.fft import FFTManager
from rmhdgpu.fourier_diagnostics import modal_inner_product_average
from rmhdgpu.grid import build_grid
from rmhdgpu.masks import build_dealias_mask
from rmhdgpu.operators import dx, dy, inv_lap_perp, lap_perp
from rmhdgpu.run import run_simulation
from rmhdgpu.runfile import dump_toml, resolve_run_settings
from rmhdgpu.state import State
from rmhdgpu.workspace import Workspace


EQUATION_MODULES = {
    "inhomogeneous_rmhd_rho": rmhd_by_nokia_rho,
    "inhomogeneous_rmhd_s": rmhd_by_nokia_s,
}

# Asymmetric gradients so no term is accidentally zero (K_b0 = 0.035 here).
PHYSICS = {"vA": 1.3, "cs2_over_vA2": 0.4, "g": 0.35, "K_p0": 0.9, "K_rho0": 2.1}

ALL = ("w_plus", "w_minus", *channels.RESERVOIRS)


def _context(equation_set: str, *, vA: float = 1.3, n: int = 12) -> dict[str, Any]:
    config = Config(equation_set=equation_set, Nx=n, Ny=n, Nz=n, backend="numpy", **{**PHYSICS, "vA": vA})
    backend = build_backend(config)
    grid = build_grid(config, backend)
    return {
        "module": EQUATION_MODULES[equation_set],
        "config": config,
        "backend": backend,
        "grid": grid,
        "fft": FFTManager(grid, backend),
        "workspace": Workspace(grid, backend),
        "mask": build_dealias_mask(grid, backend),
    }


def _random_state(ctx: dict[str, Any], mask: Any | None = None, seed: int = 20261120) -> State:
    """Deterministic multimode state with every field populated and dealiased."""

    grid, fft, backend, module = ctx["grid"], ctx["fft"], ctx["backend"], ctx["module"]
    mask = ctx["mask"] if mask is None else mask
    state = State(grid, backend, field_names=module.FIELD_NAMES)
    rng = np.random.default_rng(seed)
    for index, name in enumerate(module.FIELD_NAMES):
        real_field = rng.standard_normal(grid.real_shape) * (0.3 + 0.1 * index)
        state[name][...] = fft.r2c(backend.xp.asarray(real_field.astype(grid.real_dtype))) * mask
    state["omega"][...] = lap_perp(state["omega"], grid)
    return state


def _budgets(ctx: dict[str, Any], state: State, linear_ops: dict[str, Any] | None = None) -> dict[str, Any]:
    return ctx["module"].compute_conserved_quantity_budgets(
        state, grid=ctx["grid"], backend=ctx["backend"], params=ctx["config"], linear_ops=linear_ops,
        fft=ctx["fft"], workspace=ctx["workspace"], dealias_mask=ctx["mask"],
    )


def _s_hat(ctx: dict[str, Any], fields: State) -> Any:
    """Return s, or its rate when `fields` is an RHS; the rho set derives it linearly."""

    if ctx["module"] is rmhd_by_nokia_s:
        return fields["s"]
    return rmhd_by_nokia_rho.derive_s_hat(fields["drho"], fields["db_par"], ctx["config"])


def _exact_rates(ctx: dict[str, Any], state: State, rate: State) -> dict[str, float]:
    """d_t of every reservoir when the state changes at `rate`, straight from the definitions."""

    grid, backend = ctx["grid"], ctx["backend"]
    p = ctx["module"].derived_parameters(ctx["config"])

    def mean(a: Any, b: Any) -> float:
        return modal_inner_product_average(a, b, grid, backend)

    phi = inv_lap_perp(state["omega"], grid)
    phi_t = inv_lap_perp(rate["omega"], grid)
    z_plus, z_minus = channels.elsasser_potentials(phi, state["psi"])
    z_plus_t, z_minus_t = channels.elsasser_potentials(phi_t, rate["psi"])
    db_rate = mean(state["db_par"], rate["db_par"])
    return {
        "w_plus": channels.elsasser_energy_rate(z_plus, z_plus_t, grid, backend),
        "w_minus": channels.elsasser_energy_rate(z_minus, z_minus_t, grid, backend),
        "w_par_u": mean(state["du_par"], rate["du_par"]),
        "w_par_b": p.vA**2 * db_rate,
        "w_p": p.vA**2 / p.chi * db_rate,
        "w_s": p.entropy_energy_weight * mean(_s_hat(ctx, state), _s_hat(ctx, rate)),
    }


def _assert_close(left: float, right: float, scale: float) -> None:
    assert left == pytest.approx(right, abs=1.0e-11 * max(scale, 1.0e-30))


@pytest.mark.parametrize("equation_set, vA", [("inhomogeneous_rmhd_s", 1.3), ("inhomogeneous_rmhd_rho", 1.0)])
def test_reservoirs_add_up_to_total_energy(equation_set, vA):
    """The rho set's total_energy weights db_par by 1/alpha, so it only agrees at vA = 1."""

    ctx = _context(equation_set, vA=vA)
    budgets = _budgets(ctx, _random_state(ctx))
    total = sum(budgets[name]["value"] for name in ALL)
    assert total == pytest.approx(budgets["total_energy"]["value"], rel=1.0e-12)
    # W^p is slaved to W_par^B by pressure balance.
    assert budgets["w_p"]["value"] == pytest.approx(budgets["w_par_b"]["value"] / PHYSICS["cs2_over_vA2"], rel=1e-12)


@pytest.mark.parametrize("equation_set", sorted(EQUATION_MODULES))
def test_each_reservoir_books_every_ideal_term(equation_set):
    """d_t W along ideal_rhs equals the sum of that reservoir's named ideal terms, one by one."""

    ctx = _context(equation_set)
    state = _random_state(ctx)
    rhs = ctx["module"].ideal_rhs(state, ctx["grid"], ctx["fft"], ctx["workspace"], ctx["config"], dealias_mask=ctx["mask"])
    exact = _exact_rates(ctx, state, rhs)
    budgets = _budgets(ctx, state)
    for name in ALL:
        terms = budgets[name]["rhs_terms"]
        scale = abs(exact[name]) + sum(abs(value) for value in terms.values())
        _assert_close(sum(terms.values()), exact[name], scale)
        assert all(abs(value) > 0.0 for value in terms.values()), (name, terms)


@pytest.mark.parametrize("equation_set", sorted(EQUATION_MODULES))
def test_exchange_terms_cancel_and_drives_add_up_to_acr(equation_set):
    ctx = _context(equation_set)
    budgets = _budgets(ctx, _random_state(ctx))
    slow = ("w_par_u", "w_par_b", "w_p")
    for term in ("exchange_lin", "exchange_nl"):
        values = [budgets[name]["rhs_terms"][term] for name in slow]
        _assert_close(sum(values), 0.0, sum(abs(v) for v in values))

    drives = [
        budgets["w_par_u"]["rhs_terms"]["drive_KB"],
        budgets["w_par_b"]["rhs_terms"]["drive_KBKp"],
        budgets["w_p"]["rhs_terms"]["drive_KBKp"],
        budgets["w_s"]["rhs_terms"]["drive_Ks"],
        budgets["w_plus"]["rhs_terms"]["buoyancy"],
        budgets["w_minus"]["rhs_terms"]["buoyancy"],
    ]
    acr = [budgets["total_energy"]["rhs_terms"][name] for name in channels.ACR_CHANNELS]
    _assert_close(sum(drives), sum(acr), sum(abs(v) for v in drives + acr))


@pytest.mark.parametrize("equation_set, vA", [("inhomogeneous_rmhd_s", 1.3), ("inhomogeneous_rmhd_rho", 1.0)])
def test_reservoir_dissipation_matches_damping_rates(equation_set, vA):
    """Unequal operators per field, so the rho set's s cross term is exercised."""

    ctx = _context(equation_set, vA=vA)
    grid, module = ctx["grid"], ctx["module"]
    state = _random_state(ctx)
    linear_ops = {
        name: 0.01 * (index + 1) * grid.kperp2 + 0.003 * (index + 2) * grid.kpar2
        for index, name in enumerate(module.FIELD_NAMES)
    }
    budgets = _budgets(ctx, state, linear_ops)

    damped = state.zeros_like()
    for name in module.FIELD_NAMES:
        damped[name][...] = -linear_ops[name] * state[name]
    exact = _exact_rates(ctx, state, damped)
    for name in ALL:
        booked = budgets[name]["rhs_terms"]["dissipation"]
        _assert_close(booked, exact[name], abs(exact[name]))
    total = sum(budgets[name]["rhs_terms"]["dissipation"] for name in ALL)
    assert total == pytest.approx(budgets["total_energy"]["rhs_terms"]["dissipation"], rel=1.0e-11)


def test_kprl_rms_is_the_parallel_gradient_scale():
    ctx = _context("inhomogeneous_rmhd_s")
    grid, backend = ctx["grid"], ctx["backend"]
    state = _random_state(ctx)
    z_plus, _ = channels.elsasser_potentials(inv_lap_perp(state["omega"], grid), state["psi"])
    k_rms = channels.elsasser_kprl_rms(z_plus, grid, backend)
    # <|d_z z+|^2> = <k_par^2 k_perp^2 |f|^2> = k_rms^2 <|z+|^2> = k_rms^2 * 4 W+.
    dz_z_squared = modal_inner_product_average(grid.kpar2 * grid.kperp2 * z_plus, z_plus, grid, backend)
    assert dz_z_squared == pytest.approx(k_rms**2 * 4.0 * channels.elsasser_energy(z_plus, grid, backend), rel=1e-12)
    assert k_rms >= channels.elsasser_kprl_mean(z_plus, grid, backend)


# ---------------------------------------------------------------------------
# z-profiles: the same budgets at each z
# ---------------------------------------------------------------------------


def _low_k_mask(grid: Any, kmax: int = 2) -> np.ndarray:
    """Keep |n| <= kmax per direction, so every triple product is resolved with
    no truncation: the local-in-z identities are then exact, not just their z-average."""

    n_x = np.abs(np.fft.fftfreq(grid.Nx) * grid.Nx).reshape(-1, 1, 1)
    n_y = np.abs(np.fft.fftfreq(grid.Ny) * grid.Ny).reshape(1, -1, 1)
    n_z = np.arange(grid.Nz // 2 + 1).reshape(1, 1, -1)
    return ((n_x <= kmax) & (n_y <= kmax) & (n_z <= kmax)).astype(float)


def _dz_profile(profile: np.ndarray, Lz: float) -> np.ndarray:
    k = 2.0 * np.pi * np.fft.rfftfreq(profile.size, d=Lz / profile.size)
    return np.fft.irfft(1j * k * np.fft.rfft(profile), n=profile.size)


@pytest.mark.parametrize("equation_set", sorted(EQUATION_MODULES))
def test_z_profiles_average_to_the_box_budgets(equation_set):
    ctx = _context(equation_set)
    state = _random_state(ctx)
    module = ctx["module"]
    profiles = module.z_profiles(state, ctx["grid"], ctx["fft"], ctx["backend"], ctx["config"])
    assert set(profiles) == {"z", *Z_PROFILE_INFO}
    budgets = _budgets(ctx, state)
    box = {name: budgets[name]["value"] for name in ALL}
    box.update(
        w_plus_buoyancy=budgets["w_plus"]["rhs_terms"]["buoyancy"],
        w_minus_buoyancy=budgets["w_minus"]["rhs_terms"]["buoyancy"],
        w_par_u_drive_KB=budgets["w_par_u"]["rhs_terms"]["drive_KB"],
        w_par_bp_drive_KBKp=budgets["w_par_b"]["rhs_terms"]["drive_KBKp"] + budgets["w_p"]["rhs_terms"]["drive_KBKp"],
        w_s_drive_Ks=budgets["w_s"]["rhs_terms"]["drive_Ks"],
    )
    p = module.derived_parameters(ctx["config"])
    box["slow_flux"] = -p.vA**2 * modal_inner_product_average(state["du_par"], state["db_par"], ctx["grid"], ctx["backend"])
    for name, value in box.items():
        assert np.mean(profiles[name]) == pytest.approx(value, rel=1e-11, abs=1e-14), name


@pytest.mark.parametrize("equation_set", sorted(EQUATION_MODULES))
def test_local_budgets_hold_at_every_z(equation_set):
    """Along ideal_rhs, at each z:

        d_t W±(z) = ∓vA d_z W±(z) + buoyancy±(z)
        d_t (W_par^u + W_par^B + W^p)(z) = -d_z F_slow(z) + drives(z)
        d_t W_s(z) = drive_Ks(z)

    The nonlinear terms vanish in the perpendicular average at each z.
    """

    ctx = _context(equation_set, n=16)
    grid, fft, module, config = ctx["grid"], ctx["fft"], ctx["module"], ctx["config"]
    state = _random_state(ctx, mask=_low_k_mask(grid))
    rhs = module.ideal_rhs(state, grid, fft, ctx["workspace"], config, dealias_mask=ctx["mask"])
    profiles = module.z_profiles(state, grid, fft, ctx["backend"], config)
    p = module.derived_parameters(config)

    def real(f_hat):
        return fft.c2r(f_hat)

    def perp_mean(a, b):
        return np.mean(real(a) * real(b), axis=(0, 1))

    def elsasser_rate(f, f_t):
        # d_t |grad_perp f|^2 / 4 = grad_perp f . grad_perp f_t / 2
        return 0.5 * (perp_mean(dx(f, grid), dx(f_t, grid)) + perp_mean(dy(f, grid), dy(f_t, grid)))

    phi, phi_t = inv_lap_perp(state["omega"], grid), inv_lap_perp(rhs["omega"], grid)
    z_plus, z_minus = channels.elsasser_potentials(phi, state["psi"])
    z_plus_t, z_minus_t = channels.elsasser_potentials(phi_t, rhs["psi"])

    checks = {
        "w_plus": (
            elsasser_rate(z_plus, z_plus_t),
            -p.vA * _dz_profile(profiles["w_plus"], grid.Lz) + profiles["w_plus_buoyancy"],
        ),
        "w_minus": (
            elsasser_rate(z_minus, z_minus_t),
            p.vA * _dz_profile(profiles["w_minus"], grid.Lz) + profiles["w_minus_buoyancy"],
        ),
        "slow": (
            perp_mean(state["du_par"], rhs["du_par"])
            + p.vA**2 / p.alpha * perp_mean(state["db_par"], rhs["db_par"]),
            -_dz_profile(profiles["slow_flux"], grid.Lz)
            + profiles["w_par_u_drive_KB"]
            + profiles["w_par_bp_drive_KBKp"],
        ),
        "w_s": (
            p.entropy_energy_weight * perp_mean(_s_hat(ctx, state), _s_hat(ctx, rhs)),
            profiles["w_s_drive_Ks"],
        ),
    }
    for name, (exact, predicted) in checks.items():
        scale = np.max(np.abs(exact)) + np.max(np.abs(predicted))
        np.testing.assert_allclose(exact, predicted, rtol=0.0, atol=1e-11 * scale, err_msg=name)
    # The transport term is what the box mean cannot see: make sure it is really there.
    assert np.max(np.abs(_dz_profile(profiles["w_plus"], grid.Lz))) > 1e-3 * np.max(profiles["w_plus"])


@pytest.mark.parametrize("equation_set", sorted(EQUATION_MODULES))
def test_dissipation_profiles_are_the_local_damping_rates(equation_set):
    """Each dissipation profile is d_t W(z) under damping alone, at every z, and
    its z-average is the matching box dissipation term."""

    ctx = _context(equation_set, vA=1.0)
    grid, fft, module, config = ctx["grid"], ctx["fft"], ctx["module"], ctx["config"]
    state = _random_state(ctx)
    linear_ops = {
        name: 0.01 * (index + 1) * grid.kperp2 + 0.003 * (index + 2) * grid.kpar2
        for index, name in enumerate(module.FIELD_NAMES)
    }
    profiles = module.z_profiles(state, grid, fft, ctx["backend"], config, linear_ops=linear_ops)
    assert set(profiles) == {"z", *Z_PROFILE_INFO, *Z_PROFILE_DISSIPATION_INFO}
    p = module.derived_parameters(config)

    damped = state.zeros_like()
    for name in module.FIELD_NAMES:
        damped[name][...] = -linear_ops[name] * state[name]

    def perp_mean(a, b):
        return np.mean(fft.c2r(a) * fft.c2r(b), axis=(0, 1))

    def elsasser_rate(f, f_t):
        return 0.5 * (perp_mean(dx(f, grid), dx(f_t, grid)) + perp_mean(dy(f, grid), dy(f_t, grid)))

    phi, phi_t = inv_lap_perp(state["omega"], grid), inv_lap_perp(damped["omega"], grid)
    z_plus, z_minus = channels.elsasser_potentials(phi, state["psi"])
    z_plus_t, z_minus_t = channels.elsasser_potentials(phi_t, damped["psi"])
    exact = {
        "w_plus_dissipation": elsasser_rate(z_plus, z_plus_t),
        "w_minus_dissipation": elsasser_rate(z_minus, z_minus_t),
        "slow_dissipation": perp_mean(state["du_par"], damped["du_par"])
        + p.vA**2 / p.alpha * perp_mean(state["db_par"], damped["db_par"]),
        "w_s_dissipation": p.entropy_energy_weight * perp_mean(_s_hat(ctx, state), _s_hat(ctx, damped)),
    }
    for name, values in exact.items():
        np.testing.assert_allclose(profiles[name], values, rtol=0.0, atol=1e-11 * np.max(np.abs(values)), err_msg=name)

    budgets = _budgets(ctx, state, linear_ops)
    box = {
        "w_plus_dissipation": budgets["w_plus"]["rhs_terms"]["dissipation"],
        "w_minus_dissipation": budgets["w_minus"]["rhs_terms"]["dissipation"],
        "slow_dissipation": sum(budgets[name]["rhs_terms"]["dissipation"] for name in ("w_par_u", "w_par_b", "w_p")),
        "w_s_dissipation": budgets["w_s"]["rhs_terms"]["dissipation"],
    }
    for name, value in box.items():
        assert np.mean(profiles[name]) == pytest.approx(value, rel=1e-11), name


def _local_budget_profiles(ctx, state):
    grid, module = ctx["grid"], ctx["module"]
    linear_ops = {
        name: 0.01 * (index + 1) * grid.kperp2 + 0.003 * (index + 2) * grid.kpar2
        for index, name in enumerate(module.FIELD_NAMES)
    }
    profiles = module.z_profiles(
        state, grid, ctx["fft"], ctx["backend"], ctx["config"],
        linear_ops=linear_ops, workspace=ctx["workspace"], dealias_mask=ctx["mask"],
    )
    return profiles, linear_ops


@pytest.mark.parametrize("equation_set", sorted(EQUATION_MODULES))
def test_every_energy_has_a_closed_local_budget(equation_set):
    """At every z, each energy's d_t W(z) is the sum of its saved term profiles.

    Uses a low-k state so the 2/3-rule truncation cannot touch any triple
    product; then the nonlinear terms that are not saved (advection, and the
    nonlinear terms of the W± equations) vanish at each z exactly.
    """

    ctx = _context(equation_set, n=16)
    state = _random_state(ctx, mask=_low_k_mask(ctx["grid"]))
    profiles, _ = _local_budget_profiles(ctx, state)
    assert set(profiles) == {"z", *Z_PROFILE_INFO, *Z_PROFILE_DISSIPATION_INFO, *Z_PROFILE_LOCAL_BUDGET_INFO}
    for energy, terms in Z_PROFILE_LOCAL_BUDGETS.items():
        total = sum(profiles[f"{energy}_{term}"] for term in terms)
        scale = np.max(np.abs(profiles[f"{energy}_dt"])) + max(np.max(np.abs(profiles[f"{energy}_{term}"])) for term in terms)
        np.testing.assert_allclose(profiles[f"{energy}_dt"], total, rtol=0.0, atol=1e-11 * scale, err_msg=energy)
    # Parallel transport really is part of these budgets.
    assert np.max(np.abs(profiles["w_plus_transport"])) > 1e-3 * np.max(np.abs(profiles["w_plus_dt"]))


@pytest.mark.parametrize("equation_set", sorted(EQUATION_MODULES))
def test_local_budgets_average_to_the_box_budgets(equation_set):
    """With a full random spectrum, the z-average of each d_t W(z) and of each
    term profile is the box budget (no forcing), and the transport averages to zero."""

    ctx = _context(equation_set)
    state = _random_state(ctx)
    profiles, linear_ops = _local_budget_profiles(ctx, state)
    budgets = _budgets(ctx, state, linear_ops)
    for energy, terms in Z_PROFILE_LOCAL_BUDGETS.items():
        box = budgets[energy]["rhs_terms"]
        scale = sum(abs(value) for value in box.values())
        assert np.mean(profiles[f"{energy}_dt"]) == pytest.approx(sum(box.values()), abs=1e-11 * scale), energy
        for term in terms:
            expected = 0.0 if term == "transport" else box[term]
            assert np.mean(profiles[f"{energy}_{term}"]) == pytest.approx(expected, abs=1e-11 * scale), (energy, term)


# ---------------------------------------------------------------------------
# Run level: the CSV budgets close interval by interval
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("equation_set", sorted(EQUATION_MODULES))
def test_run_writes_closed_reservoir_budgets_and_z_profiles(tmp_path, equation_set):
    dt = 1.0 / 2048
    dissipation = {
        name: dict(nu_perp=2e-3, nu_par=1e-3, n_perp=1, n_par=1)
        for name in EQUATION_MODULES[equation_set].FIELD_NAMES
    }
    document = dict(
        output_dir="outputs",
        equations=dict(type=equation_set),
        grid=dict(Nx=12, Ny=12, Nz=12),
        backend=dict(backend="numpy"),
        physics={**PHYSICS, "vA": 1.0},
        time=dict(tmax=16 * dt, dt_init=dt, use_variable_dt=False),
        output=dict(t_out_scal=4 * dt, t_out_spec=8 * dt, t_out_full=0.0),
        runtime=dict(progress_output_every=100),
        initial_condition=dict(type="random_spectrum_one_wave", parameters=dict(init_energy=0.5, seed=3)),
        dissipation=dissipation,
    )
    path = tmp_path / "reservoirs.input"
    path.write_text(dump_toml(document))
    settings = resolve_run_settings(runfile_path=path)
    run_simulation(settings)

    with (settings.output_dir / "scalar_diagnostics.csv").open() as handle:
        rows = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(handle)]
    assert len(rows) == 5
    # One scale for every reservoir: the largest booked rate anywhere. w_minus
    # starts at zero (one-wave initial condition), so a per-reservoir scale
    # would demand round-off agreement there.
    scale = max(
        abs(row[col]) for row in rows[1:] for col in row
        if any(col.startswith(f"{name}_rhs_") for name in ALL) and not col.endswith("_total")
    )
    for name in ALL:
        for before, after in zip(rows, rows[1:]):
            interval = after["time"] - before["time"]
            change = after[name] - before[name]
            booked = after[f"{name}_rhs_total"] * interval
            # Trapezoid in time over 4 steps: the error is O(dt^2) of the change.
            assert change == pytest.approx(booked, abs=1e-4 * scale * interval), name

    with (settings.output_dir / "z_profiles.csv").open() as handle:
        profile_rows = list(csv.DictReader(handle))
    assert {row["quantity"] for row in profile_rows} == (
        set(Z_PROFILE_INFO) | set(Z_PROFILE_DISSIPATION_INFO) | set(Z_PROFILE_LOCAL_BUDGET_INFO)
    )
    assert len({row["time"] for row in profile_rows}) == 3

    # The plotting scripts run on these outputs and write their figures.
    from vis import (plot_energy_budgets, plot_energy_reservoirs, plot_energy_terms,
                     plot_z_profiles, plot_z_rms_terms)

    for script in (plot_energy_reservoirs, plot_energy_terms, plot_z_profiles,
                   plot_energy_budgets, plot_z_rms_terms):
        assert script.main([str(settings.output_dir)]).is_file()
    plot_energy_budgets.main([str(settings.output_dir), "--average", "2", "--separate", "--hide", "dissipation"])
    plot_energy_terms.main([str(settings.output_dir), "--hide", "dissipation"])
    assert all((settings.output_dir / f"budget_{name}.png").is_file() for name in ALL)

    from vis import plot_dcf_ccr_acr, plot_local_budget_signs

    assert plot_dcf_ccr_acr.main([str(settings.output_dir), "--average", "2"]).is_file()

    plot_local_budget_signs.main([str(settings.output_dir)])
    assert not (settings.output_dir / "local_budget_w_p.png").exists()
    plot_local_budget_signs.main([str(settings.output_dir), "--energy", "all"])
    assert all((settings.output_dir / f"local_budget_{name}.png").is_file() for name in ALL)
    with (settings.output_dir / "energy_terms.csv").open() as handle:
        shares = {}
        for row in csv.DictReader(handle):
            if row["term"] not in ("measured_dt", "residual"):
                sign = "gain" if float(row["rate"]) > 0 else "loss"
                shares[(row["quantity"], sign)] = shares.get((row["quantity"], sign), 0.0) + abs(float(row["share"]))
    # Each reservoir's gains add up to 100% of its gains, and likewise its losses.
    # (shares are written to 4 decimals, hence the tolerance)
    assert all(total == pytest.approx(1.0, abs=1e-3) for total in shares.values()), shares


def test_budget_rates_weight_rows_by_their_interval():
    """Window averages of interval-averaged terms, on a hand-made CSV with uneven intervals."""

    from vis.run_quantities import budget_rates, budget_rows

    columns = {
        "time": np.array([0.0, 1.0, 3.0, 4.0]),
        # Q grows by exactly the booked terms: +2*1, +(1)*2, -1*1 from the "a" + "b" terms.
        "q": np.array([5.0, 7.0, 9.0, 8.0]),
        "q_rhs_a": np.array([0.0, 3.0, 2.0, 0.0]),
        "q_rhs_b": np.array([0.0, -1.0, -1.0, -1.0]),
        "q_rhs_total": np.array([0.0, 2.0, 1.0, -1.0]),
    }
    rows = budget_rows(columns["time"], tmin=0.0)
    np.testing.assert_array_equal(rows, [1, 2, 3])
    rates = budget_rates(columns, "q", rows)
    assert rates["a"] == pytest.approx((3.0 * 1 + 2.0 * 2 + 0.0 * 1) / 4.0)
    assert rates["b"] == pytest.approx(-1.0)
    assert rates["measured_dt"] == pytest.approx(3.0 / 4.0)
    assert rates["residual"] == pytest.approx(0.0)


def test_z_profile_cadence_defaults_to_spectra_and_can_be_set_apart(tmp_path):
    """t_out_prof follows t_out_spec unless set, and then writes on its own cadence."""

    assert Config(t_out_spec=0.5).t_out_prof == 0.5
    assert Config(t_out_spec=0.5, t_out_prof=0.1).t_out_prof == 0.1
    assert Config(t_out_spec=0.5, t_out_prof=0.0).t_out_prof == 0.0

    dt = 1.0 / 2048
    document = dict(
        output_dir="outputs",
        equations=dict(type="inhomogeneous_rmhd_s"),
        grid=dict(Nx=8, Ny=8, Nz=8),
        backend=dict(backend="numpy"),
        physics={**PHYSICS, "vA": 1.0},
        time=dict(tmax=8 * dt, dt_init=dt, use_variable_dt=False),
        output=dict(t_out_scal=8 * dt, t_out_spec=8 * dt, t_out_prof=2 * dt, t_out_full=0.0),
        runtime=dict(progress_output_every=100),
        initial_condition=dict(type="random_spectrum_one_wave", parameters=dict(init_energy=0.5, seed=3)),
    )
    path = tmp_path / "cadence.input"
    path.write_text(dump_toml(document))
    settings = resolve_run_settings(runfile_path=path)
    run_simulation(settings)

    def saved_times(name):
        with (settings.output_dir / name).open() as handle:
            return {row["time"] for row in csv.DictReader(handle)}

    assert len(saved_times("spectra.csv")) == 2  # t = 0 and 8 dt
    assert len(saved_times("z_profiles.csv")) == 5  # t = 0, 2, 4, 6, 8 dt
