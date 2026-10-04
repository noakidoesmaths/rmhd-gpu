"""Formal timestep-convergence regression test for the `alfvenic` equation set.

This test is intentionally both a solver regression test and a saved
diagnostic:

1. Round 1 starts from a large-amplitude exact nonlinear `z-` Alfvén wave.
   Because this is an exact solution of the ideal two-field Alfvénic system,
   the final-time error tests both the implemented equations and the SSPRK3
   timestep convergence.
2. Round 2 starts from a deterministic random low-mode state. That no longer
   has a simple analytic solution, so the finest-step run is used as the
   reference solution and the coarser runs are checked for third-order
   convergence against it.

The pytest entry uses conservative defaults so it runs quickly:

- `16^3` resolution
- `6` fixed-timestep runs per round, with `dt` halved each run
- `8` deterministic low modes in round 2
- zero forcing and zero dissipation, explicitly, so this remains a pure ideal
  timestepper/equation test

If you want to tune the resolution or the number of runs/modes for a manual
study, run this file directly and inspect `--help`, for example:

`python rmhdgpu/tests/test_timestep_convergence_alfvenic.py --help`
"""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rmhdgpu.backend import build_backend
from rmhdgpu.config import Config
from rmhdgpu.equations import alfvenic
from rmhdgpu.fft import FFTManager
from rmhdgpu.grid import build_grid
from rmhdgpu.initconds import build_initial_state
from rmhdgpu.masks import build_dealias_mask
from rmhdgpu.operators import lap_perp
from rmhdgpu.state import State
from rmhdgpu.steppers import compute_cfl_timestep, ssprk3_step
from rmhdgpu.workspace import Workspace


DEFAULT_RESOLUTION = 16
DEFAULT_K_INDICES = (1, 1, 1)
DEFAULT_ALFVEN_AMPLITUDE = 1.0
DEFAULT_RANDOM_TARGET_ENERGY = 1.0
DEFAULT_RANDOM_MODE_COUNT = 8
DEFAULT_RANDOM_MODE_RADIUS = 2.5
DEFAULT_RANDOM_SEED = 24680
DEFAULT_RUNS_PER_ROUND = 6
DEFAULT_REQUIRED_ORDER = 2.8
PLOT_PATH = Path(__file__).with_name("alfvenic_timestep_convergence.png")
CSV_PATH = Path(__file__).with_name("alfvenic_timestep_convergence.csv")


@dataclass(frozen=True, slots=True)
class ConvergenceSettings:
    """User-adjustable settings for the saved convergence experiment."""

    resolution: int = DEFAULT_RESOLUTION
    k_indices: tuple[int, int, int] = DEFAULT_K_INDICES
    alfven_amplitude: float = DEFAULT_ALFVEN_AMPLITUDE
    random_target_energy: float = DEFAULT_RANDOM_TARGET_ENERGY
    random_mode_count: int = DEFAULT_RANDOM_MODE_COUNT
    random_mode_radius: float = DEFAULT_RANDOM_MODE_RADIUS
    random_seed: int = DEFAULT_RANDOM_SEED
    runs_per_round: int = DEFAULT_RUNS_PER_ROUND
    required_order: float = DEFAULT_REQUIRED_ORDER
    plot_path: Path = PLOT_PATH
    csv_path: Path = CSV_PATH


@dataclass(frozen=True, slots=True)
class RoundResults:
    """Saved results for one convergence round."""

    name: str
    dt_values: np.ndarray
    steps: np.ndarray
    errors: np.ndarray
    observed_order: float


def _zero_dissipation(field_names: Sequence[str]) -> dict[str, dict[str, float | int]]:
    return {
        name: {
            "nu_perp": 0.0,
            "nu_par": 0.0,
            "n_perp": 3,
            "n_par": 3,
        }
        for name in field_names
    }


def _validate_settings(settings: ConvergenceSettings) -> None:
    if settings.resolution <= 0 or settings.resolution % 2 != 0:
        raise ValueError(f"resolution must be a positive even integer; got {settings.resolution!r}.")
    if settings.runs_per_round < 3:
        raise ValueError(f"runs_per_round must be at least 3; got {settings.runs_per_round!r}.")
    if settings.random_mode_count < 1:
        raise ValueError(f"random_mode_count must be positive; got {settings.random_mode_count!r}.")
    if settings.random_mode_radius <= 0.0:
        raise ValueError(f"random_mode_radius must be positive; got {settings.random_mode_radius!r}.")
    if settings.required_order <= 0.0:
        raise ValueError(f"required_order must be positive; got {settings.required_order!r}.")
    if len(settings.k_indices) != 3:
        raise ValueError(f"k_indices must have length 3; got {settings.k_indices!r}.")


def _build_context(settings: ConvergenceSettings) -> tuple[Config, object, object, FFTManager, Workspace, object]:
    field_names = list(alfvenic.FIELD_NAMES)
    config = Config(
        equation_set="alfvenic",
        Nx=settings.resolution,
        Ny=settings.resolution,
        Nz=settings.resolution,
        backend="numpy",
        vA=1.0,
        cfl_number=0.5,
        use_variable_dt=False,
        use_forcing=False,
        dissipation=_zero_dissipation(field_names),
        field_names=field_names,
    )
    backend = build_backend(config)
    grid = build_grid(config, backend)
    fft = FFTManager(grid, backend)
    workspace = Workspace(grid, backend)
    mask = build_dealias_mask(grid, backend)
    return config, backend, grid, fft, workspace, mask


def _total_time_for_mode(grid: object, *, vA: float, kz_index: int) -> float:
    kz = float(grid.kz[0, 0, kz_index])
    return 2.0 * np.pi / abs(vA * kz)


def _step_state(
    state0: State,
    *,
    dt: float,
    steps: int,
    config: Config,
    grid: object,
    fft: FFTManager,
    workspace: Workspace,
    mask: object,
) -> State:
    rhs_kwargs = {
        "grid": grid,
        "fft": fft,
        "workspace": workspace,
        "params": config,
        "dealias_mask": mask,
    }
    current = state0.copy()
    for _ in range(steps):
        current = ssprk3_step(current, dt, alfvenic.ideal_rhs, rhs_kwargs=rhs_kwargs)
    return current


def _scaled_state(state: State, factor: complex) -> State:
    out = state.copy()
    for name in out.field_names:
        out[name][...] *= factor
    return out


def _relative_real_l2_error(
    state: State,
    reference: State,
    *,
    fft: FFTManager,
    backend: object,
) -> float:
    numerator = 0.0
    denominator = 0.0
    for field_name in state.field_names:
        diff_real = fft.c2r(state[field_name] - reference[field_name])
        reference_real = fft.c2r(reference[field_name])
        numerator += backend.scalar_to_float(backend.xp.mean(diff_real**2))
        denominator += backend.scalar_to_float(backend.xp.mean(reference_real**2))
    return float(np.sqrt(numerator / denominator))


def _base_steps_from_cfl(
    state0: State,
    *,
    total_time: float,
    config: Config,
    grid: object,
    fft: FFTManager,
    workspace: Workspace,
) -> int:
    dt_cfl = compute_cfl_timestep(
        state0,
        grid,
        fft,
        config,
        workspace=workspace,
        equation_module=alfvenic,
    )
    return max(1, int(np.ceil(total_time / dt_cfl)))


def _dt_schedule(base_steps: int, *, total_time: float, levels: int) -> tuple[np.ndarray, np.ndarray]:
    steps = base_steps * (2 ** np.arange(levels, dtype=int))
    dt = total_time / steps.astype(np.float64)
    return dt, steps


def _rescale_state_to_energy(
    state: State,
    *,
    target_energy: float,
    grid: object,
    backend: object,
    config: Config,
) -> State:
    current_energy = alfvenic.total_energy(state, grid, backend, config)
    if current_energy <= 0.0 or not np.isfinite(current_energy):
        raise ValueError(f"Expected positive finite initial energy; got {current_energy!r}.")
    scale = np.sqrt(target_energy / current_energy)
    for field_name in state.field_names:
        state[field_name][...] *= scale
    return state


def _candidate_random_modes(radius: float) -> list[tuple[int, int, int]]:
    max_index = max(1, int(np.ceil(radius)))
    candidates: list[tuple[int, int, int]] = []
    for kz in range(1, max_index + 1):
        for kx in range(-max_index, max_index + 1):
            for ky in range(-max_index, max_index + 1):
                if kx == 0 and ky == 0:
                    continue
                knorm = float(np.sqrt(kx**2 + ky**2 + kz**2))
                if knorm >= radius:
                    continue
                candidates.append((kx, ky, kz))
    candidates.sort(key=lambda mode: (np.sqrt(mode[0] ** 2 + mode[1] ** 2 + mode[2] ** 2), abs(mode[2]), abs(mode[0]), abs(mode[1]), mode[0], mode[1]))
    return candidates


def _hard_coded_random_mode_state(
    *,
    grid: object,
    backend: object,
    field_names: list[str],
    target_energy: float,
    config: Config,
    random_mode_count: int,
    random_mode_radius: float,
    random_seed: int,
) -> State:
    rng = np.random.default_rng(random_seed)
    state = State(grid, backend, field_names=field_names)
    phi_hat = backend.zeros(grid.fourier_shape, dtype=grid.complex_dtype)
    psi_hat = backend.zeros(grid.fourier_shape, dtype=grid.complex_dtype)

    candidate_modes = _candidate_random_modes(random_mode_radius)
    if len(candidate_modes) < random_mode_count:
        raise ValueError(
            f"random_mode_radius={random_mode_radius!r} only provides {len(candidate_modes)} candidate modes; "
            f"need at least {random_mode_count}."
        )

    for kx, ky, kz in candidate_modes[:random_mode_count]:
        ix = kx % grid.Nx
        iy = ky % grid.Ny
        phi_hat[ix, iy, kz] = (rng.normal() + 1j * rng.normal()) / np.sqrt(2.0)
        psi_hat[ix, iy, kz] = (rng.normal() + 1j * rng.normal()) / np.sqrt(2.0)

    state["psi"][...] = psi_hat
    state["omega"][...] = lap_perp(phi_hat, grid)
    state.apply_mask(build_dealias_mask(grid, backend))
    return _rescale_state_to_energy(
        state,
        target_energy=target_energy,
        grid=grid,
        backend=backend,
        config=config,
    )


def _observed_order(dt_values: np.ndarray, error_values: np.ndarray) -> float:
    finite = np.isfinite(dt_values) & np.isfinite(error_values) & (dt_values > 0.0) & (error_values > 0.0)
    if np.count_nonzero(finite) < 3:
        raise ValueError("Need at least three finite positive error points to estimate convergence order.")
    return float(np.polyfit(np.log(dt_values[finite]), np.log(error_values[finite]), 1)[0])


def _write_results_csv(
    rows: list[dict[str, float | int | str]],
    *,
    csv_path: Path,
) -> None:
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["round", "dt", "steps", "error"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _add_order_guides(ax: plt.Axes, dt_values: np.ndarray, error_values: np.ndarray) -> None:
    finite = np.isfinite(error_values) & (error_values > 0.0)
    if np.count_nonzero(finite) < 2:
        return

    finite_dt = dt_values[finite]
    finite_error = error_values[finite]
    order = np.argsort(finite_dt)
    x = finite_dt[order]
    y = finite_error[order]
    anchor_x = x[0]
    anchor_y = y[0]
    for power in (1, 2, 3, 4):
        guide = anchor_y * (x / anchor_x) ** power
        ax.loglog(x, guide, color="0.6", ls=":", lw=1.0, label=rf"$\Delta t^{power}$")


def _save_plot(
    *,
    round1: RoundResults,
    round2: RoundResults,
    plot_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)

    axes[0].loglog(round1.dt_values, round1.errors, "o-", color="tab:blue", lw=2.0, label="single Alfvén wave")
    _add_order_guides(axes[0], round1.dt_values, round1.errors)
    axes[0].set_title(f"Round 1: exact nonlinear Alfvén wave\nobserved order = {round1.observed_order:.3f}")
    axes[0].set_xlabel(r"$\Delta t$")
    axes[0].set_ylabel("relative error")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend(fontsize=8)

    finite = np.isfinite(round2.errors) & (round2.errors > 0.0)
    axes[1].loglog(
        round2.dt_values[finite],
        round2.errors[finite],
        "o-",
        color="tab:orange",
        lw=2.0,
        label="random low modes",
    )
    _add_order_guides(axes[1], round2.dt_values[finite], round2.errors[finite])
    axes[1].set_title(f"Round 2: random low-mode reference test\nobserved order = {round2.observed_order:.3f}")
    axes[1].set_xlabel(r"$\Delta t$")
    axes[1].set_ylabel("relative error")
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].legend(fontsize=8)

    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def _print_experiment_description(settings: ConvergenceSettings) -> None:
    print("alfvenic timestep convergence regression", flush=True)
    print(
        "  ideal configuration: zero forcing, zero dissipation, fixed SSPRK3 timestep",
        flush=True,
    )
    print(
        f"  round 1: exact nonlinear z- Alfvén wave with k={settings.k_indices}, "
        f"resolution={settings.resolution}^3, {settings.runs_per_round} runs",
        flush=True,
    )
    print(
        f"  round 2: deterministic random low modes ({settings.random_mode_count} modes with |k| < {settings.random_mode_radius}), "
        f"resolution={settings.resolution}^3, {settings.runs_per_round} runs",
        flush=True,
    )
    print(
        f"  pass criterion: observed log-log error slope must exceed {settings.required_order:.2f} in both rounds",
        flush=True,
    )


def run_alfvenic_timestep_convergence(
    settings: ConvergenceSettings,
    *,
    verbose: bool = True,
) -> tuple[RoundResults, RoundResults]:
    """Run both convergence rounds, save CSV/plot, and return the measured data."""

    _validate_settings(settings)
    rows: list[dict[str, float | int | str]] = []

    if verbose:
        _print_experiment_description(settings)

    config1, backend1, grid1, fft1, workspace1, mask1 = _build_context(settings)
    state_single = build_initial_state(
        "alfven_mode",
        parameters={"k_indices": list(settings.k_indices), "amplitude": settings.alfven_amplitude, "branch": "z_minus"},
        grid=grid1,
        backend=backend1,
        fft=fft1,
        dealias_mask=mask1,
        field_names=config1.field_names,
        params=config1,
    )
    total_time = _total_time_for_mode(grid1, vA=config1.vA, kz_index=settings.k_indices[2])
    base_steps_single = _base_steps_from_cfl(
        state_single,
        total_time=total_time,
        config=config1,
        grid=grid1,
        fft=fft1,
        workspace=workspace1,
    )
    dt_round1, steps_round1 = _dt_schedule(
        base_steps_single,
        total_time=total_time,
        levels=settings.runs_per_round,
    )
    err_round1 = np.zeros_like(dt_round1)

    if verbose:
        print("round 1/2: exact nonlinear Alfvén wave", flush=True)
    kz = float(grid1.kz[0, 0, settings.k_indices[2]])
    exact = _scaled_state(state_single, np.exp(1j * config1.vA * kz * total_time))
    for index, (dt, steps) in enumerate(zip(dt_round1, steps_round1, strict=True)):
        evolved = _step_state(
            state_single,
            dt=float(dt),
            steps=int(steps),
            config=config1,
            grid=grid1,
            fft=fft1,
            workspace=workspace1,
            mask=mask1,
        )
        error = _relative_real_l2_error(evolved, exact, fft=fft1, backend=backend1)
        err_round1[index] = error
        rows.append({"round": "single_alfven_wave", "dt": float(dt), "steps": int(steps), "error": float(error)})
        if verbose:
            print(
                f"  run {index + 1}/{settings.runs_per_round}: dt={dt:.6e}, steps={steps}, "
                f"t_final={total_time:.6e}, error={error:.6e}",
                flush=True,
            )

    round1 = RoundResults(
        name="single_alfven_wave",
        dt_values=dt_round1,
        steps=steps_round1,
        errors=err_round1,
        observed_order=_observed_order(dt_round1, err_round1),
    )

    config2, backend2, grid2, fft2, workspace2, mask2 = _build_context(settings)
    state_random = _hard_coded_random_mode_state(
        grid=grid2,
        backend=backend2,
        field_names=config2.field_names,
        target_energy=settings.random_target_energy,
        config=config2,
        random_mode_count=settings.random_mode_count,
        random_mode_radius=settings.random_mode_radius,
        random_seed=settings.random_seed,
    )
    base_steps_random = _base_steps_from_cfl(
        state_random,
        total_time=total_time,
        config=config2,
        grid=grid2,
        fft=fft2,
        workspace=workspace2,
    )
    dt_round2, steps_round2 = _dt_schedule(
        base_steps_random,
        total_time=total_time,
        levels=settings.runs_per_round,
    )
    err_round2 = np.full_like(dt_round2, np.nan)

    if verbose:
        print("round 2/2: deterministic random low modes, finest step used as reference", flush=True)
    reference_index = len(dt_round2) - 1
    reference_state = _step_state(
        state_random,
        dt=float(dt_round2[reference_index]),
        steps=int(steps_round2[reference_index]),
        config=config2,
        grid=grid2,
        fft=fft2,
        workspace=workspace2,
        mask=mask2,
    )
    rows.append(
        {
            "round": "random_modes_reference",
            "dt": float(dt_round2[reference_index]),
            "steps": int(steps_round2[reference_index]),
            "error": float("nan"),
        }
    )
    if verbose:
        print(
            f"  reference: dt={dt_round2[reference_index]:.6e}, steps={steps_round2[reference_index]}, "
            f"t_final={total_time:.6e}",
            flush=True,
        )

    for index in range(reference_index - 1, -1, -1):
        dt = float(dt_round2[index])
        steps = int(steps_round2[index])
        evolved = _step_state(
            state_random,
            dt=dt,
            steps=steps,
            config=config2,
            grid=grid2,
            fft=fft2,
            workspace=workspace2,
            mask=mask2,
        )
        error = _relative_real_l2_error(evolved, reference_state, fft=fft2, backend=backend2)
        err_round2[index] = error
        rows.append({"round": "random_modes", "dt": dt, "steps": steps, "error": float(error)})
        if verbose:
            print(
                f"  run {reference_index - index}/{settings.runs_per_round - 1}: dt={dt:.6e}, steps={steps}, "
                f"t_final={total_time:.6e}, error={error:.6e}",
                flush=True,
            )

    round2 = RoundResults(
        name="random_modes",
        dt_values=dt_round2,
        steps=steps_round2,
        errors=err_round2,
        observed_order=_observed_order(dt_round2, err_round2),
    )

    _write_results_csv(rows, csv_path=settings.csv_path)
    _save_plot(round1=round1, round2=round2, plot_path=settings.plot_path)

    if verbose:
        print(f"saved convergence plot to {settings.plot_path}", flush=True)
        print(f"saved convergence data to {settings.csv_path}", flush=True)
        print(
            f"observed orders: round1={round1.observed_order:.6f}, round2={round2.observed_order:.6f}",
            flush=True,
        )

    return round1, round2


def test_alfvenic_timestep_convergence() -> None:
    settings = ConvergenceSettings()
    round1, round2 = run_alfvenic_timestep_convergence(settings, verbose=True)

    assert np.all(np.isfinite(round1.errors))
    assert np.all(np.isfinite(round2.errors[:-1]))
    assert round1.observed_order > settings.required_order
    assert round2.observed_order > settings.required_order
    assert settings.plot_path.exists()
    assert settings.csv_path.exists()


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the saved alfvenic timestep-convergence experiment used by the pytest regression test. "
            "This keeps forcing and dissipation disabled and writes a CSV plus a log-log plot into the tests folder."
        )
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=DEFAULT_RESOLUTION,
        help="Grid size N so the run uses N^3 points. Default: %(default)s.",
    )
    parser.add_argument(
        "--k-indices",
        nargs=3,
        type=int,
        metavar=("KX", "KY", "KZ"),
        default=list(DEFAULT_K_INDICES),
        help="Stored Fourier mode indices for the exact Alfvén wave in round 1. Default: %(default)s.",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=DEFAULT_ALFVEN_AMPLITUDE,
        help="Target sqrt(total_energy) amplitude of the exact Alfvén wave in round 1. Default: %(default)s.",
    )
    parser.add_argument(
        "--runs-per-round",
        "--levels",
        dest="runs_per_round",
        type=int,
        default=DEFAULT_RUNS_PER_ROUND,
        help="Number of fixed-dt runs in each round. dt is halved between successive runs. Default: %(default)s.",
    )
    parser.add_argument(
        "--random-mode-count",
        type=int,
        default=DEFAULT_RANDOM_MODE_COUNT,
        help="Number of deterministic low modes used to build the round-2 random initial condition. Default: %(default)s.",
    )
    parser.add_argument(
        "--random-mode-radius",
        type=float,
        default=DEFAULT_RANDOM_MODE_RADIUS,
        help="Radius in |k| used to build the candidate low-mode list for round 2. Default: %(default)s.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Seed for the deterministic round-2 low-mode coefficients. Default: %(default)s.",
    )
    parser.add_argument(
        "--random-target-energy",
        type=float,
        default=DEFAULT_RANDOM_TARGET_ENERGY,
        help="Total energy to which the round-2 random state is rescaled. Default: %(default)s.",
    )
    parser.add_argument(
        "--required-order",
        type=float,
        default=DEFAULT_REQUIRED_ORDER,
        help="Minimum acceptable observed convergence order in each round. Default: %(default)s.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_argument_parser()
    args = parser.parse_args(argv)
    settings = ConvergenceSettings(
        resolution=args.resolution,
        k_indices=(int(args.k_indices[0]), int(args.k_indices[1]), int(args.k_indices[2])),
        alfven_amplitude=float(args.amplitude),
        random_target_energy=float(args.random_target_energy),
        random_mode_count=int(args.random_mode_count),
        random_mode_radius=float(args.random_mode_radius),
        random_seed=int(args.random_seed),
        runs_per_round=int(args.runs_per_round),
        required_order=float(args.required_order),
    )
    round1, round2 = run_alfvenic_timestep_convergence(settings, verbose=True)
    return 0 if round1.observed_order > settings.required_order and round2.observed_order > settings.required_order else 1


if __name__ == "__main__":
    raise SystemExit(main())
