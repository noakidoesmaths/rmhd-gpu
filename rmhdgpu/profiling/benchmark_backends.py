"""Lightweight backend benchmark driver for representative RMHD runs."""

from __future__ import annotations

import argparse
import importlib.util
import time
from typing import Any

from rmhdgpu.backend import build_backend
from rmhdgpu.config import Config
from rmhdgpu.equations import s09
from rmhdgpu.fft import FFTManager
from rmhdgpu.grid import build_grid
from rmhdgpu.masks import build_dealias_mask
from rmhdgpu.state import State
from rmhdgpu.steppers import evolve_until
from rmhdgpu.workspace import Workspace


def _available_backends() -> list[str]:
    backends = ["numpy"]
    if importlib.util.find_spec("scipy") is not None:
        backends.append("scipy_cpu")
    if importlib.util.find_spec("cupy") is not None:
        backends.append("cupy")
    return backends


def _build_case_config(
    backend_name: str,
    nx: int,
    *,
    dt: float,
    use_forcing: bool,
) -> Config:
    config = Config(
        Nx=nx,
        Ny=nx,
        Nz=nx,
        backend=backend_name,
        dt_init=dt,
        use_variable_dt=False,
        fail_on_nonfinite=True,
        runtime_check_every=8,
        use_forcing=use_forcing,
        forcing_seed=1234,
    )
    for name in config.field_names:
        config.dissipation[name]["nu_perp"] = 2.5e-3
        config.dissipation[name]["nu_par"] = 5.0e-3
        config.dissipation[name]["n_perp"] = 2
        config.dissipation[name]["n_par"] = 1

    if use_forcing:
        config.field_energy_injection_rates["psi"] = 2.0e-2
        config.field_energy_injection_rates["omega"] = 2.0e-2

    return config


def _deterministic_state(backend: Any, grid: Any, fft: FFTManager, mask: Any) -> State:
    state = State(grid, backend, field_names=s09.FIELD_NAMES)
    xp = backend.xp
    x = grid.x.reshape(grid.Nx, 1, 1)
    y = grid.y.reshape(1, grid.Ny, 1)
    z = grid.z.reshape(1, 1, grid.Nz)

    phi_real = (
        0.35 * xp.cos(x + z)
        + 0.20 * xp.sin(2.0 * y + z)
        + 0.10 * xp.cos(2.0 * x + y)
    ).astype(grid.real_dtype, copy=False)
    psi_real = (
        0.25 * xp.cos(x + y + z)
        + 0.18 * xp.sin(2.0 * x + y + 2.0 * z)
        + 0.08 * xp.cos(3.0 * y + z)
    ).astype(grid.real_dtype, copy=False)
    upar_real = (0.05 * xp.cos(x + 2.0 * z) + 0.03 * xp.sin(y + z)).astype(
        grid.real_dtype,
        copy=False,
    )
    dbpar_real = (0.04 * xp.sin(y + z) + 0.02 * xp.cos(x + y)).astype(
        grid.real_dtype,
        copy=False,
    )
    entropy_real = (0.06 * xp.sin(x + y + z)).astype(grid.real_dtype, copy=False)

    phi_hat = fft.r2c(phi_real)
    psi_hat = fft.r2c(psi_real)
    state["psi"][...] = psi_hat * mask
    state["omega"][...] = s09.lap_perp(phi_hat * mask, grid)
    state["upar"][...] = fft.r2c(upar_real) * mask
    state["dbpar"][...] = fft.r2c(dbpar_real) * mask
    state["s"][...] = fft.r2c(entropy_real) * mask
    return state


def _estimated_fft_calls_per_step(config: Config) -> int:
    fft_calls = 3 * 8 * 5
    if config.use_forcing:
        forced_fields = sum(
            1
            for epsilon in config.field_energy_injection_rates.values()
            if float(epsilon) != 0.0
        )
        fft_calls += 3 * forced_fields
    return fft_calls


def build_benchmark_case(
    backend_name: str,
    nx: int,
    *,
    dt: float = 2.5e-3,
    use_forcing: bool = False,
) -> tuple[Config, Any, Any, FFTManager, Workspace, Any, State]:
    config = _build_case_config(backend_name, nx, dt=dt, use_forcing=use_forcing)
    backend = build_backend(config)
    grid = build_grid(config, backend)
    fft = FFTManager(grid, backend)
    workspace = Workspace(grid, backend)
    mask = build_dealias_mask(grid, backend, mode=config.dealias_mode) if config.dealias else None
    state = _deterministic_state(backend, grid, fft, mask if mask is not None else 1.0)
    return config, backend, grid, fft, workspace, mask, state


def run_backend_case(
    backend_name: str,
    nx: int,
    *,
    steps: int = 10,
    dt: float = 2.5e-3,
    use_forcing: bool = False,
) -> dict[str, float | int | str]:
    config, backend, grid, fft, workspace, mask, state = build_benchmark_case(
        backend_name,
        nx,
        dt=dt,
        use_forcing=use_forcing,
    )
    linear_ops = s09.build_dissipation_operators(grid, config)
    rhs_kwargs = {
        "grid": grid,
        "fft": fft,
        "workspace": workspace,
        "params": config,
        "dealias_mask": mask,
    }

    backend.synchronize()
    start = time.perf_counter()
    final_state, info = evolve_until(
        state,
        steps * dt,
        s09.ideal_rhs,
        linear_ops,
        rhs_kwargs=rhs_kwargs,
        params=config,
        fixed_dt=dt,
    )
    backend.synchronize()
    elapsed = time.perf_counter() - start

    total_points = float(grid.Nx * grid.Ny * grid.Nz)
    estimated_fft_calls = _estimated_fft_calls_per_step(config)
    max_abs_psi = backend.scalar_to_float(backend.xp.max(backend.xp.abs(final_state["psi"])))
    return {
        "backend": backend_name,
        "nx": nx,
        "steps": int(info["steps"]),
        "elapsed_s": elapsed,
        "time_per_step_s": elapsed / max(int(info["steps"]), 1),
        "estimated_fft_calls_per_step": estimated_fft_calls,
        "estimated_fft_throughput_mpts_s": (estimated_fft_calls * total_points * int(info["steps"]))
        / max(elapsed, 1.0e-30)
        / 1.0e6,
        "max_abs_psi": max_abs_psi,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        action="append",
        dest="backends",
        choices=["numpy", "scipy_cpu", "cupy"],
        help="Backend(s) to benchmark. Defaults to all available backends.",
    )
    parser.add_argument(
        "--nx",
        action="append",
        dest="sizes",
        type=int,
        help="Cubic grid size(s). Defaults to 64 and 96.",
    )
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--dt", type=float, default=2.5e-3)
    parser.add_argument("--use-forcing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    backends = args.backends if args.backends is not None else _available_backends()
    sizes = args.sizes if args.sizes is not None else [64, 96]

    for backend_name in backends:
        if backend_name not in _available_backends():
            print(f"skip backend={backend_name}: dependency unavailable")
            continue

        for nx in sizes:
            result = run_backend_case(
                backend_name,
                nx,
                steps=args.steps,
                dt=args.dt,
                use_forcing=args.use_forcing,
            )
            print(result)


if __name__ == "__main__":
    main()
