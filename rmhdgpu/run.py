"""Main simulation driver for CLI and `.input` (TOML) workflows."""

from __future__ import annotations

import argparse
import shlex
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from rmhdgpu.auto_dissipation import (
    AutoDissipationController,
    disabled_auto_dissipation_diagnostics,
)
from rmhdgpu.backend import build_backend
from rmhdgpu.diagnostics.spectra import (
    parallel_energy_spectrum_from_state,
    perpendicular_energy_spectrum_from_state,
)
from rmhdgpu.diagnostics.scalar import compute_scalar_diagnostics
from rmhdgpu.errors import NonFiniteStateError
from rmhdgpu.equations import available_equation_sets, get_equation_module
from rmhdgpu.fft import FFTManager
from rmhdgpu.forcing import apply_forcing_kick, generate_forcing_kick
from rmhdgpu.grid import build_grid
from rmhdgpu.initconds import build_initial_state, list_initial_condition_types
from rmhdgpu.masks import build_dealias_mask
from rmhdgpu.output import (
    FULLFIELD_DIAGNOSTICS_DIRNAME,
    SCALAR_DIAGNOSTICS_FILENAME,
    SPECTRA_DIAGNOSTICS_FILENAME,
    SPECTRA_PRL_DIAGNOSTICS_FILENAME,
    FullFieldHDF5Writer,
    ScalarDiagnosticsWriter,
    SpectraDiagnosticsWriter,
    advance_output_time,
    initial_output_time,
    output_cadence_enabled,
    output_due,
)
from rmhdgpu.runfile import (
    LEGACY_INPUT_SUFFIX,
    PRIMARY_INPUT_SUFFIX,
    RunSettings,
    cli_overrides_from_args,
    resolve_run_settings,
    write_resolved_config,
)
from rmhdgpu.state import State
from rmhdgpu.steppers import compute_cfl_timestep, if_ssprk3_step, project_out_kpar0
import rmhdgpu.operators as operators
from rmhdgpu.utils import check_state_finite
from rmhdgpu.workspace import Workspace


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(slots=True)
class _RunLogger:
    path: Path
    _handle: Any

    def event(self, label: str, payload: dict[str, Any] | None = None) -> None:
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        if payload is None:
            line = f"{timestamp} {label}"
        else:
            line = f"{timestamp} {label} {payload}"
        print(line)
        print(line, file=self._handle)
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for normal runs."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Dissipation is configured in .input files. Use manual per-field "
            "[dissipation.<field>] blocks by default, or enable one common "
            "adaptive coefficient with [dissipation] mode = \"auto\"."
        ),
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        default=None,
        help=(
            f"Optional {PRIMARY_INPUT_SUFFIX} input file. These files use TOML syntax internally. "
            f"Legacy {LEGACY_INPUT_SUFFIX} files are also accepted."
        ),
    )
    parser.add_argument("--title", default=argparse.SUPPRESS)
    parser.add_argument("--output-dir", default=argparse.SUPPRESS)
    parser.add_argument(
        "--equation-set",
        dest="equation_set",
        choices=available_equation_sets(),
        default=argparse.SUPPRESS,
        help="Equation set to run. Usually set in [equations] type = ... in the input file.",
    )
    parser.add_argument(
        "--equation-mode",
        dest="equation_mode",
        choices=["nonlinear", "linear"],
        default=argparse.SUPPRESS,
        help="RHS mode. `linear` keeps only equation-module linear terms; default is `nonlinear`.",
    )

    parser.add_argument("--backend", choices=["numpy", "scipy_cpu", "cupy"], default=argparse.SUPPRESS)
    parser.add_argument("--fft-workers", type=int, default=argparse.SUPPRESS)

    parser.add_argument("--nx", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--ny", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--nz", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--lx", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--ly", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--lz", type=float, default=argparse.SUPPRESS)

    parser.add_argument("--tmax", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--dt-init", dest="dt_init", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--dt-min", dest="dt_min", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--dt-max", dest="dt_max", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--cfl-number", dest="cfl_number", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--t-out-scal", dest="t_out_scal", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--t-out-spec", dest="t_out_spec", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--t-out-full", dest="t_out_full", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--use-variable-dt", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)

    parser.add_argument("--runtime-check-every", dest="runtime_check_every", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--progress-output-every", dest="progress_output_every", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--fail-on-nonfinite", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    parser.add_argument("--dealias", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    parser.add_argument("--dealias-mode", default=argparse.SUPPRESS)

    parser.add_argument("--vA", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--cs2-over-vA2", dest="cs2_over_vA2", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--N2", type=float, default=argparse.SUPPRESS)

    parser.add_argument("--use-forcing", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    parser.add_argument("--force-sigma", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--forcing-seed", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--n-min-force", dest="n_min_force", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--n-max-force", dest="n_max_force", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--alpha-force", dest="alpha_force", type=float, default=argparse.SUPPRESS)

    parser.add_argument(
        "--initial-condition",
        choices=list_initial_condition_types(),
        default=argparse.SUPPRESS,
        help="Initial condition family for CLI mode or run-file override.",
    )
    parser.add_argument("--mode-kx", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--mode-ky", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--mode-kz", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--mode-amplitude", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--mode-branch", choices=["plus", "minus"], default=argparse.SUPPRESS)
    return parser


def _git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip() or None


def _prepare_output_dir(settings: RunSettings) -> Path:
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    return settings.output_dir


def _open_run_logger(output_dir: Path) -> _RunLogger:
    log_path = output_dir / "run.log"
    handle = log_path.open("w", encoding="utf-8")
    return _RunLogger(path=log_path, _handle=handle)


def _write_input_copy(settings: RunSettings, output_dir: Path) -> None:
    if settings.input_file is None:
        return
    shutil.copyfile(settings.input_file, output_dir / "input_copy.input")


def _startup_metadata(settings: RunSettings, output_dir: Path) -> dict[str, Any]:
    return {
        "title": settings.title,
        "input_file": None if settings.input_file is None else str(settings.input_file),
        "output_dir": str(output_dir),
        "backend": settings.config.backend,
        "equation_set": settings.config.equation_set,
        "equation_mode": settings.config.equation_mode,
        "grid": [settings.config.Nx, settings.config.Ny, settings.config.Nz],
        "output_cadences": {
            "scalar": settings.config.t_out_scal,
            "spectra": settings.config.t_out_spec,
            "full_field": settings.config.t_out_full,
        },
        "hostname": socket.gethostname(),
        "python_executable": sys.executable,
        "git_commit": _git_commit_hash(),
        "command": " ".join(shlex.quote(arg) for arg in sys.argv),
    }


def _diagnostics_row(
    state: State,
    *,
    step: int,
    time: float,
    dt: float,
    grid: Any,
    fft: Any,
    backend: Any,
    params: Any,
    workspace: Any,
    equation_module: Any,
    linear_ops: dict[str, Any] | None = None,
    budget_rhs_terms: dict[str, dict[str, float]] | None = None,
    extra_scalar_diagnostics: dict[str, float] | None = None,
) -> dict[str, float | int]:
    row: dict[str, float | int] = {
        "step": step,
        "time": time,
        "dt": dt,
    }
    row.update(compute_scalar_diagnostics(state, grid, fft, backend, workspace=workspace))
    row.update(
        equation_module.compute_equation_scalar_diagnostics(
            state,
            grid=grid,
            fft=fft,
            backend=backend,
            params=params,
            workspace=workspace,
            linear_ops=linear_ops,
            budget_rhs_terms=budget_rhs_terms,
        )
    )
    if extra_scalar_diagnostics is not None:
        row.update(extra_scalar_diagnostics)
    return row


def _zero_poisson_bracket(
    f_hat: Any,
    g_hat: Any,
    grid: Any,
    fft: Any,
    workspace: Any,
    mask: Any | None = None,
    out: Any | None = None,
) -> Any:
    """Return a zero bracket with the same call signature as `poisson_bracket`."""

    if out is None:
        out = workspace.complex["c0"] if workspace is not None else f_hat * 0.0
    out[...] = 0.0
    return out


def run_simulation(settings: RunSettings) -> dict[str, Any]:
    """Run one resolved case and write standard outputs."""

    config = settings.config
    equation_module = get_equation_module(config.equation_set)
    rhs_func = equation_module.ideal_rhs
    original_operator_poisson_bracket = operators.poisson_bracket
    module_had_poisson_bracket = hasattr(equation_module, "poisson_bracket")
    original_module_poisson_bracket = getattr(equation_module, "poisson_bracket", None)
    poisson_bracket_overridden = False
    output_dir = _prepare_output_dir(settings)
    logger = _open_run_logger(output_dir)
    try:
        if config.equation_mode == "linear":
            operators.poisson_bracket = _zero_poisson_bracket
            setattr(equation_module, "poisson_bracket", _zero_poisson_bracket)
            poisson_bracket_overridden = True

        write_resolved_config(settings, output_dir / "resolved_config.toml")
        _write_input_copy(settings, output_dir)

        logger.event("run start", _startup_metadata(settings, output_dir))
        logger.event(
            "run setup",
            {
                "resolved_output_dir": str(output_dir),
                "input_file": None if settings.input_file is None else str(settings.input_file),
                "backend": config.backend,
                "equation_set": config.equation_set,
                "equation_mode": config.equation_mode,
                "grid": [config.Nx, config.Ny, config.Nz],
                "output_cadences": {
                    "scalar": config.t_out_scal,
                    "spectra": config.t_out_spec,
                    "full_field": config.t_out_full,
                },
                "dissipation_mode": config.auto_dissipation.mode,
                "initial_condition": settings.initial_condition.to_document(),
            },
        )

        backend = build_backend(config)
        grid = build_grid(config, backend)
        fft = FFTManager(grid, backend)
        mask = build_dealias_mask(grid, backend, mode=config.dealias_mode) if config.dealias else None
        workspace = Workspace(grid, backend)
        state = build_initial_state(
            settings.initial_condition,
            grid=grid,
            backend=backend,
            fft=fft,
            dealias_mask=mask,
            field_names=settings.config.field_names,
            params=settings.config,
        )
        auto_dissipation_controller = None
        auto_dissipation_diagnostics = disabled_auto_dissipation_diagnostics()
        if config.auto_dissipation.enabled:
            auto_dissipation_controller = AutoDissipationController.from_runtime(
                settings=config.auto_dissipation,
                equation_module=equation_module,
                field_names=config.field_names,
                grid=grid,
                backend=backend,
                dealias_mask=mask,
            )
            auto_dissipation_controller.update(state, config)
            auto_dissipation_diagnostics = auto_dissipation_controller.diagnostics()
            linear_ops = equation_module.build_dissipation_operators(
                grid,
                config,
                dissipation_spec=auto_dissipation_controller.effective_dissipation(),
            )
        else:
            linear_ops = equation_module.build_dissipation_operators(grid, config)

        energy_initial = equation_module.total_energy(state, grid, backend, config)
        rhs_kwargs = {
            "grid": grid,
            "fft": fft,
            "workspace": workspace,
            "params": config,
            "dealias_mask": mask,
        }

        scalar_csv_path = output_dir / SCALAR_DIAGNOSTICS_FILENAME
        spectra_csv_path = output_dir / SPECTRA_DIAGNOSTICS_FILENAME
        spectra_prl_csv_path = output_dir / SPECTRA_PRL_DIAGNOSTICS_FILENAME
        fullfield_output_dir = output_dir / FULLFIELD_DIAGNOSTICS_DIRNAME

        scalar_writer = ScalarDiagnosticsWriter(scalar_csv_path) if output_cadence_enabled(config.t_out_scal) else None
        spectra_writer = (
            SpectraDiagnosticsWriter(spectra_csv_path) if output_cadence_enabled(config.t_out_spec) else None
        )
        spectra_prl_writer = (
            SpectraDiagnosticsWriter(spectra_prl_csv_path, k_column="kprl")
            if output_cadence_enabled(config.t_out_spec)
            else None
        )
        fullfield_writer = None
        if output_cadence_enabled(config.t_out_full):
            fullfield_writer = FullFieldHDF5Writer(
                fullfield_output_dir,
                grid=grid,
                backend=backend,
                field_names=config.field_names,
                backend_name=config.backend,
            )

        t = 0.0
        steps = 0
        next_scalar_output = initial_output_time(config.t_out_scal)
        next_spectra_output = initial_output_time(config.t_out_spec)
        next_fullfield_output = initial_output_time(config.t_out_full)
        dt_last = config.dt_init
        forcing_rng = backend.random_generator(config.forcing_seed) if config.use_forcing else None
        track_budget = scalar_writer is not None
        budget_interval_duration = 0.0
        budget_interval_terms: dict[str, dict[str, float]] = {}

        def _instantaneous_budget_terms(sample_state: State) -> dict[str, dict[str, float]]:
            budgets = equation_module.compute_conserved_quantity_budgets(
                sample_state,
                grid=grid,
                backend=backend,
                params=config,
                linear_ops=linear_ops,
            )
            return {
                quantity_name: {
                    term_name: float(value)
                    for term_name, value in payload.get("rhs_terms", {}).items()
                }
                for quantity_name, payload in budgets.items()
            }

        def _accumulate_budget_terms(before: dict[str, dict[str, float]], after: dict[str, dict[str, float]], dt: float) -> None:
            for quantity_name in sorted(set(before) | set(after)):
                target = budget_interval_terms.setdefault(quantity_name, {})
                term_names = set(before.get(quantity_name, {})) | set(after.get(quantity_name, {}))
                for term_name in term_names:
                    before_value = before.get(quantity_name, {}).get(term_name, 0.0)
                    after_value = after.get(quantity_name, {}).get(term_name, 0.0)
                    target[term_name] = target.get(term_name, 0.0) + 0.5 * (before_value + after_value) * dt

        def _accumulate_budget_kick(quantity_name: str, term_name: str, value: float) -> None:
            terms = budget_interval_terms.setdefault(quantity_name, {})
            terms[term_name] = terms.get(term_name, 0.0) + value

        def _averaged_budget_terms() -> dict[str, dict[str, float]]:
            if budget_interval_duration <= 0.0:
                averaged = {
                    quantity_name: {term_name: 0.0 for term_name in rhs_terms}
                    for quantity_name, rhs_terms in _instantaneous_budget_terms(state).items()
                }
            else:
                averaged = {
                    quantity_name: {
                        term_name: value / budget_interval_duration
                        for term_name, value in rhs_terms.items()
                    }
                    for quantity_name, rhs_terms in budget_interval_terms.items()
                }
            averaged.setdefault("total_energy", {}).setdefault("forcing", 0.0)
            if project_kpar0:
                averaged.setdefault("total_energy", {}).setdefault("projection", 0.0)
            return averaged

        def _reset_budget_interval() -> None:
            nonlocal budget_interval_duration
            budget_interval_duration = 0.0
            for rhs_terms in budget_interval_terms.values():
                for term_name in rhs_terms:
                    rhs_terms[term_name] = 0.0

        def _write_due_diagnostics(*, dt_value: float) -> None:
            nonlocal next_scalar_output, next_spectra_output, next_fullfield_output

            if scalar_writer is not None and output_due(time=t, next_output_time=next_scalar_output, tmax=config.tmax):
                row = _diagnostics_row(
                    state,
                    step=steps,
                    time=t,
                    dt=dt_value,
                    grid=grid,
                    fft=fft,
                    backend=backend,
                    params=config,
                    workspace=workspace,
                    equation_module=equation_module,
                    linear_ops=linear_ops,
                    budget_rhs_terms=_averaged_budget_terms(),
                    extra_scalar_diagnostics=auto_dissipation_diagnostics,
                )
                scalar_writer.write_row(row)
                logger.event(
                    "scalar diagnostics",
                    {
                        "step": row["step"],
                        "time": row["time"],
                        "dt": row["dt"],
                        "total_energy": row["total_energy"],
                    },
                )
                _reset_budget_interval()
                next_scalar_output = advance_output_time(
                    next_output_time=next_scalar_output,
                    cadence=config.t_out_scal,
                    current_time=t,
                )

            if spectra_writer is not None and output_due(time=t, next_output_time=next_spectra_output, tmax=config.tmax):
                spectra = perpendicular_energy_spectrum_from_state(
                    state,
                    grid,
                    backend,
                    equation_module=equation_module,
                    params=config,
                )
                spectra_writer.write_spectra(time=t, step=steps, spectra=spectra)
                spectra_prl = parallel_energy_spectrum_from_state(
                    state,
                    grid,
                    backend,
                    equation_module=equation_module,
                    params=config,
                )
                spectra_prl_writer.write_spectra(time=t, step=steps, spectra=spectra_prl)
                logger.event(
                    "spectra diagnostics",
                    {
                        "time": t,
                        "step": steps,
                        "quantities": [key for key in spectra if key != "kperp"],
                        "shell_count": int(len(spectra["kperp"])),
                        "prl_shell_count": int(len(spectra_prl["kprl"])),
                    },
                )
                next_spectra_output = advance_output_time(
                    next_output_time=next_spectra_output,
                    cadence=config.t_out_spec,
                    current_time=t,
                )

            if fullfield_writer is not None and output_due(
                time=t,
                next_output_time=next_fullfield_output,
                tmax=config.tmax,
            ):
                derived_fn = getattr(equation_module, "derived_output_fields", None)
                extra_fields = (
                    derived_fn(state, grid, fft, backend) if derived_fn is not None else None
                )
                output_index = fullfield_writer.write_state(
                    state,
                    time=t,
                    step=steps,
                    fft=fft,
                    backend=backend,
                    field_names=config.field_names,
                    extra_fields=extra_fields,
                )
                logger.event(
                    "full-field diagnostics",
                    {
                        "time": t,
                        "step": steps,
                        "output_index": output_index + 1,
                        "path": str(fullfield_writer.snapshot_path(output_index)),
                    },
                )
                next_fullfield_output = advance_output_time(
                    next_output_time=next_fullfield_output,
                    cadence=config.t_out_full,
                    current_time=t,
                )

        project_kpar0 = bool(getattr(config, "project_kpar0", False))

        try:
            if config.fail_on_nonfinite:
                check_state_finite(state, backend, time=t, step=steps, context="run startup")

            # One-time init cleanup of the marginal k_par = 0 plane, analogous to
            # exclude_kpar0. With exclude_kpar0=true this removes ~0 (the plane is
            # already empty); it runs before any budget interval starts, so it is
            # intentionally not booked as a budget term. If project_kpar0=true is
            # used with exclude_kpar0=false, this startup removal is unaccounted.
            if project_kpar0:
                project_out_kpar0(state, grid)

            _write_due_diagnostics(dt_value=0.0)

            while t < config.tmax - 1.0e-15:
                if config.use_variable_dt:
                    dt = compute_cfl_timestep(
                        state,
                        grid,
                        fft,
                        config,
                        dt_prev=dt_last,
                        workspace=workspace,
                        equation_module=equation_module,
                    )
                else:
                    dt = config.dt_init
                dt = min(dt, config.tmax - t)

                if track_budget:
                    budget_before = _instantaneous_budget_terms(state)

                stepped_state = if_ssprk3_step(state, dt, rhs_func, linear_ops, rhs_kwargs=rhs_kwargs)

                if track_budget:
                    budget_after = _instantaneous_budget_terms(stepped_state)
                    _accumulate_budget_terms(budget_before, budget_after, dt)

                if config.use_forcing:
                    forcing_energy_before = (
                        equation_module.total_energy(stepped_state, grid, backend, config) if track_budget else 0.0
                    )
                    forcing_kick = generate_forcing_kick(
                        stepped_state,
                        grid,
                        fft,
                        backend,
                        config,
                        forcing_rng,
                        dt,
                        workspace=workspace,
                        out=workspace.get_state_buffer("forcing_kick", stepped_state.field_names),
                    )
                    state = apply_forcing_kick(stepped_state, forcing_kick, inplace=True)
                    if track_budget:
                        _accumulate_budget_kick(
                            "total_energy",
                            "forcing",
                            equation_module.total_energy(state, grid, backend, config) - forcing_energy_before,
                        )
                else:
                    state = stepped_state

                # Re-project the k_par = 0 plane the nonlinear brackets repopulate
                # each step, and book the removed energy as a `projection` sink so
                # the total-energy budget stays closed (same accounting path as
                # forcing above).
                if project_kpar0:
                    projection_energy_before = (
                        equation_module.total_energy(state, grid, backend, config)
                        if track_budget else 0.0
                    )
                    project_out_kpar0(state, grid)
                    if track_budget:
                        _accumulate_budget_kick(
                            "total_energy",
                            "projection",
                            equation_module.total_energy(state, grid, backend, config)
                            - projection_energy_before,
                        )

                if track_budget:
                    budget_interval_duration += dt

                t += dt
                dt_last = dt
                steps += 1

                if auto_dissipation_controller is not None and auto_dissipation_controller.should_update(steps):
                    auto_dissipation_controller.update(state, config)
                    auto_dissipation_diagnostics = auto_dissipation_controller.diagnostics()
                    linear_ops = equation_module.build_dissipation_operators(
                        grid,
                        config,
                        dissipation_spec=auto_dissipation_controller.effective_dissipation(),
                    )

                if config.progress_output_every is not None and (
                    steps % config.progress_output_every == 0 or t >= config.tmax - 1.0e-15
                ):
                    logger.event("progress", {"step": steps, "time": t, "dt": dt})

                if config.fail_on_nonfinite and (
                    steps % config.runtime_check_every == 0 or t >= config.tmax - 1.0e-15
                ):
                    check_state_finite(state, backend, time=t, step=steps, context="run")

                _write_due_diagnostics(dt_value=dt)
        except (ImportError, NonFiniteStateError) as exc:
            logger.event("run failed", {"error": str(exc)})
            raise SystemExit(str(exc)) from exc
        finally:
            if scalar_writer is not None:
                scalar_writer.close()
            if spectra_writer is not None:
                spectra_writer.close()
            if spectra_prl_writer is not None:
                spectra_prl_writer.close()
            if fullfield_writer is not None:
                fullfield_writer.close()

        energy_final = equation_module.total_energy(state, grid, backend, config)
        summary = {
            "backend": config.backend,
            "equation_set": config.equation_set,
            "equation_mode": config.equation_mode,
            "grid": list(grid.real_shape),
            "steps": steps,
            "t_final": t,
            "dt_last": dt_last,
            "total_energy_initial": energy_initial,
            "total_energy_final": energy_final,
            "psi_hat_max_abs": backend.scalar_to_float(backend.xp.max(backend.xp.abs(state["psi"]))),
            "output_dir": str(output_dir),
        }
        if output_cadence_enabled(config.t_out_scal):
            summary["scalar_diagnostics_csv"] = str(scalar_csv_path)
        if output_cadence_enabled(config.t_out_spec):
            summary["spectra_csv"] = str(spectra_csv_path)
        if output_cadence_enabled(config.t_out_full):
            summary["fullfields_dir"] = str(fullfield_output_dir)
        logger.event("run complete", summary)
        return summary
    finally:
        if poisson_bracket_overridden:
            operators.poisson_bracket = original_operator_poisson_bracket
            if module_had_poisson_bracket:
                setattr(equation_module, "poisson_bracket", original_module_poisson_bracket)
            else:
                delattr(equation_module, "poisson_bracket")
        logger.close()


def main(argv: list[str] | None = None) -> None:
    """Resolve CLI or `.input` input and execute one simulation."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.input_file is not None and not str(args.input_file).endswith((PRIMARY_INPUT_SUFFIX, LEGACY_INPUT_SUFFIX)):
        raise SystemExit(
            "Expected an optional input file with suffix "
            f"{PRIMARY_INPUT_SUFFIX!r} (legacy {LEGACY_INPUT_SUFFIX!r} is also accepted) "
            f"as the first positional argument; got {args.input_file!r}."
        )

    try:
        settings = resolve_run_settings(
            runfile_path=args.input_file,
            cli_overrides=cli_overrides_from_args(args),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    run_simulation(settings)


if __name__ == "__main__":
    main()
