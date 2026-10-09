"""Procedural target-amplitude and constant-power shell control.

Equation hooks provide native branch vorticities and the perpendicular metric.
The scalar controller does not depend on an equation module or Config class.
Native branch energy is quadratic, with physical Elsasser energy RMS**2/4.
Historical ``*_f`` event/history keys mean native energy; they are retained for
exact checkpoint and diagnostic compatibility (G=1 for ordinary RMHD).
"""

from __future__ import annotations

import csv
import json
import math
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def _positive(name: str, value: Any, *, allow_zero: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number, not bool.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not math.isfinite(number) or (number < 0 if allow_zero else number <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}.")
    return number


@dataclass(slots=True)
class ControlledBranchSettings:
    control: str
    target_scope: str | None = None
    target_basis: str | None = None
    target_quantity: str | None = None
    target_value: float | None = None
    tau_F: float | None = None
    startup_energy_fraction: float | None = None
    ring_floor_fraction: float | None = None
    power_basis: str | None = None
    epsilon: float | None = None
    startup_energy_factor: float | None = None
    gamma_max: float | None = None

    def __post_init__(self) -> None:
        if self.control not in {"target", "constant_power"}:
            raise ValueError("controlled branch control must be 'target' or 'constant_power'.")
        target_keys = ("target_scope", "target_basis", "target_quantity", "target_value", "tau_F",
                       "startup_energy_fraction", "ring_floor_fraction")
        power_keys = ("power_basis", "epsilon", "startup_energy_factor")
        for key in power_keys if self.control == "target" else target_keys:
            if getattr(self, key) is not None:
                raise ValueError(f"{key} is incompatible with {self.control} control.")
        if self.control == "target":
            self.target_scope = "branch_total" if self.target_scope is None else self.target_scope
            if self.target_scope not in {"branch_total", "branch_nonzero_kz", "shell", "perpendicular_shell"}:
                raise ValueError("target_scope must be 'branch_total', 'branch_nonzero_kz', 'shell' or 'perpendicular_shell'.")
            if self.target_basis not in {"elsasser", "wave_action"}:
                raise ValueError("target_basis must be explicitly 'elsasser' or 'wave_action'.")
            if self.target_quantity not in {"rms", "energy"}:
                raise ValueError("target_quantity must be explicitly 'rms' or 'energy'.")
            self.target_value = _positive("target_value", self.target_value)
            if self.tau_F is not None:
                self.tau_F = _positive("tau_F", self.tau_F)
            self.startup_energy_fraction = _positive(
                "startup_energy_fraction", 0.01 if self.startup_energy_fraction is None else self.startup_energy_fraction)
            if self.startup_energy_fraction > 1:
                raise ValueError("startup_energy_fraction must be <= 1.")
            if self.target_scope == "shell":
                if self.ring_floor_fraction is not None:
                    raise ValueError("ring_floor_fraction is only valid for branch_total, branch_nonzero_kz or perpendicular_shell control.")
            else:
                self.ring_floor_fraction = _positive(
                    "ring_floor_fraction", 0.01 if self.ring_floor_fraction is None else self.ring_floor_fraction)
                if self.ring_floor_fraction >= 1:
                    raise ValueError("ring_floor_fraction must be < 1.")
        else:
            if self.power_basis not in {"elsasser", "wave_action"}:
                raise ValueError("power_basis must be explicitly 'elsasser' or 'wave_action'.")
            self.epsilon = _positive("epsilon", self.epsilon)
            self.startup_energy_factor = _positive(
                "startup_energy_factor", 10 if self.startup_energy_factor is None else self.startup_energy_factor)
        if self.gamma_max is not None:
            self.gamma_max = _positive("gamma_max", self.gamma_max)


@dataclass(slots=True)
class ControlledShellSettings:
    k_sigma_min: float
    k_sigma_max: float
    branches: list[str]
    kz_index: int = 1
    min_forced_modes: int = 8
    phase_model: str = "none"
    phase_angle_max: float = 0.5
    interval_max: float | None = None
    log_gain_target: float = 0.05
    log_gain_warn: float = 0.1
    plus: ControlledBranchSettings | dict[str, Any] | None = None
    minus: ControlledBranchSettings | dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.k_sigma_min = _positive("k_sigma_min", self.k_sigma_min)
        self.k_sigma_max = _positive("k_sigma_max", self.k_sigma_max)
        if self.k_sigma_max < self.k_sigma_min:
            raise ValueError("k_sigma_max must be greater than or equal to k_sigma_min.")
        if not isinstance(self.branches, (list, tuple)) or not self.branches:
            raise ValueError("branches must explicitly list at least one of plus/minus.")
        self.branches = list(self.branches)
        if not all(isinstance(b, str) for b in self.branches) or len(set(self.branches)) != len(self.branches) or set(self.branches) - {"plus", "minus"}:
            raise ValueError("branches must be unique and drawn from plus/minus.")
        for name in ("kz_index", "min_forced_modes"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
            setattr(self, name, int(value))
        if self.min_forced_modes < 8:
            raise ValueError("min_forced_modes must be at least eight.")
        if self.phase_model not in {"none", "fixed_gain_angle"}:
            raise ValueError("phase_model must be 'none' or 'fixed_gain_angle'.")
        self.phase_angle_max = _positive("phase_angle_max", self.phase_angle_max, allow_zero=True)
        if self.phase_angle_max > 0.5:
            raise ValueError("phase_angle_max must be <= 0.5 radians.")
        if self.interval_max is not None:
            self.interval_max = _positive("interval_max", self.interval_max)
        self.log_gain_target = _positive("log_gain_target", self.log_gain_target)
        self.log_gain_warn = _positive("log_gain_warn", self.log_gain_warn)
        if self.log_gain_warn <= self.log_gain_target:
            raise ValueError("log_gain_warn must exceed log_gain_target.")
        for branch in ("plus", "minus"):
            value = getattr(self, branch)
            if (branch in self.branches) != (value is not None):
                raise ValueError("branches and controlled branch tables must match exactly.")
            if isinstance(value, dict):
                try:
                    setattr(self, branch, ControlledBranchSettings(**value))
                except TypeError as exc:
                    raise ValueError(f"Invalid controlled {branch} settings: {exc}") from exc
            elif value is not None and not isinstance(value, ControlledBranchSettings):
                raise ValueError(f"controlled {branch} settings must be a table.")


def settings_document(settings: ControlledShellSettings) -> dict[str, Any]:
    """Omit inapplicable settings so a resolved input can be parsed again."""
    def without_none(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: without_none(item) for key, item in value.items() if item is not None}
        return value
    return without_none(asdict(settings))


def native_parameters(settings: ControlledBranchSettings, config: Any, *, G: float) -> dict[str, float]:
    """Resolve units and dimensional defaults without touching Fourier fields."""
    if config.vA <= 0:
        raise ValueError("Controlled forcing requires vA > 0 (positive Alfven crossing time).")
    g = G
    tau_A = config.Lz / config.vA
    frequency = 2 * math.pi * config.controlled_shell.kz_index / config.Lz * config.vA
    result = {"G": g, "G2": g*g, "tau_A": tau_A,
              "tau_F": settings.tau_F or tau_A, "target_f": 0.0, "epsilon_f": 0.0,
              "floor_f": 0.0, "gamma_max": settings.gamma_max or frequency * (1 if settings.control == "target" else 10)}
    if settings.control == "target":
        target = settings.target_value**2 / 4 if settings.target_quantity == "rms" else settings.target_value
        result["target_f"] = target * (g*g if settings.target_basis == "elsasser" else 1)
        result["startup_f"] = settings.startup_energy_fraction * result["target_f"]
        result["floor_f"] = (settings.ring_floor_fraction or 0) * result["target_f"]
    else:
        result["epsilon_f"] = settings.epsilon * (g*g if settings.power_basis == "elsasser" else 1)
        result["startup_f"] = settings.startup_energy_factor * result["epsilon_f"] * tau_A
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("Controlled forcing parameters overflow in native units.")
    for key in ("startup_f", "target_f" if settings.control == "target" else "epsilon_f"):
        _positive(f"native {key} (including conversion underflow)", result[key])
    if settings.control == "target" and settings.target_scope != "shell":
        _positive("native floor_f (including conversion underflow)", result["floor_f"])
    return result


def control_event_map(settings: ControlledBranchSettings, native: dict[str, float],
                      shell: float, total: float, elapsed: float, *,
                      nonzero_kz: float | None = None,
                      perpendicular_shell: float | None = None) -> dict[str, float | int]:
    """Compute one kick; total includes kz=0, and all energies are native E_f."""
    shell = _positive("runtime shell energy (multiplicative forcing cannot revive an empty shell)", shell)
    total = _positive("runtime branch energy", total)
    elapsed = _positive("elapsed control time", elapsed)
    floor_active = False
    exhausted = False
    requested_work = 0.0
    if settings.control == "constant_power":
        requested_work = native["epsilon_f"] * elapsed
        ratio = requested_work / shell
        log_gain = (0.5 * math.log1p(ratio) if math.isfinite(ratio) else
                    0.5 * (float(np.logaddexp(math.log(shell), math.log(requested_work))) - math.log(shell)))
    else:
        if settings.target_scope == "branch_nonzero_kz":
            controlled = _positive("runtime nonzero-kz branch energy", nonzero_kz)
        elif settings.target_scope == "perpendicular_shell":
            controlled = _positive("runtime perpendicular-shell energy", perpendicular_shell)
        else:
            controlled = total if settings.target_scope == "branch_total" else shell
        relaxation = -math.expm1(-elapsed / native["tau_F"])
        requested_work = controlled * math.expm1(relaxation * (math.log(native["target_f"]) - math.log(controlled)))
        desired = shell + requested_work
        if settings.target_scope != "shell" and requested_work < 0:
            lower = min(shell, native["floor_f"])
            floor_active = desired < lower
            exhausted = floor_active
            desired = max(desired, lower)
        if desired <= 0:
            raise ValueError("Nonpositive desired shell energy after authority guard.")
        # Difference form preserves small gains even when total >> shell.
        log_gain = 0.5 * math.log1p((desired - shell) / shell)
    gamma_requested = log_gain / elapsed
    gamma_applied = float(np.clip(gamma_requested, -native["gamma_max"], native["gamma_max"]))
    rate_cap = gamma_applied != gamma_requested
    applied_log_gain = gamma_applied * elapsed
    if not math.isfinite(applied_log_gain) or abs(applied_log_gain) > 0.5 * math.log(np.finfo(float).max):
        raise ValueError("Forcing kick exceeds representable energy; reduce timestep/control interval.")
    return {"gamma_requested": gamma_requested, "gamma_applied": gamma_applied,
            "log_gain": applied_log_gain, "requested_work_f": requested_work,
            "work_f": shell * math.expm1(2 * applied_log_gain),
            "floor_active": int(floor_active), "forcing_actuator_exhausted": int(exhausted),
            "rate_cap_active": int(rate_cap), "forcing_controller_limited": int(floor_active or rate_cap)}


def shell_geometry(config: Any, settings: ControlledShellSettings, *, sigma: float) -> tuple:
    """Return the complete discrete actuator geometry, without a 3-D grid.

    Shared by input-time chi resolution and the runtime controller. Equal
    bounds select an exact lattice ring; an empty or undersampled ring fails
    the same mode-count check as a finite-width shell. Bounds are inclusive.
    """
    for name in ("Nx", "Ny", "Nz"):
        value = getattr(config, name)
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 2:
            raise ValueError(f"{name} must be an integer >= 2.")
    for name in ("Lx", "Ly", "Lz"):
        _positive(name, getattr(config, name))
    _positive("sigma", sigma)
    if not 1 <= settings.kz_index < config.Nz // 2:
        raise ValueError("kz_index must exclude the zero and Nyquist planes: 1 <= kz_index < Nz/2.")
    nx = np.rint(np.fft.fftfreq(config.Nx) * config.Nx).astype(int)
    ny = np.rint(np.fft.fftfreq(config.Ny) * config.Ny).astype(int)
    kx, ky = nx * (2*np.pi/config.Lx), ny * (2*np.pi/config.Ly)
    delta = sigma**2 * kx[:, None]**2 + sigma**-2 * ky[None, :]**2
    ks = np.sqrt(delta)
    if (settings.k_sigma_max/sigma >= np.pi*config.Nx/config.Lx
            or settings.k_sigma_max*sigma >= np.pi*config.Ny/config.Ly):
        raise ValueError("Requested metric shell extends beyond represented Fourier modes.")
    ix, iy = np.nonzero((ks >= settings.k_sigma_min) & (ks <= settings.k_sigma_max))
    if config.dealias and (np.any(abs(nx[ix]) >= config.Nx/3) or np.any(abs(ny[iy]) >= config.Ny/3)
                           or settings.kz_index >= config.Nz/3):
        raise ValueError("Requested forcing shell/kz is clipped by the dealias mask.")
    if len(ix) < settings.min_forced_modes:
        raise ValueError(f"Forcing shell has {len(ix)} modes; requires at least {settings.min_forced_modes}.")
    return nx, ny, kx, ky, delta, ks, ix, iy


@dataclass
class ForcingState:
    """Runtime data only; operations below take this state explicitly."""
    config: Any
    grid: Any
    backend: Any
    equation: Any
    settings: ControlledShellSettings = field(init=False)
    fields: dict = field(init=False)
    native: dict = field(init=False)
    g2: dict = field(init=False)
    indices: Any = field(init=False)
    k_sigma: Any = field(init=False)
    weights: Any = field(init=False)
    inv_delta: Any = field(init=False)
    low_mask: Any = field(init=False)
    templates: dict = field(init=False)
    angles: dict = field(init=False)
    mode_rows: list = field(init=False)
    mode_count: int = field(init=False)
    normalization: float = field(init=False)
    interval_max: float = field(init=False)
    interval_target: float = field(init=False)
    elapsed: float = field(init=False)
    initialized: bool = field(init=False)
    startup_work: dict = field(init=False)
    cumulative_work: dict = field(init=False)
    last_events: list = field(init=False)
    last_by_branch: dict = field(init=False)
    event_count: int = field(init=False)
    diagnostic_hook: Any = field(init=False)
    perpendicular_indices: Any = field(init=False, repr=False)
    measurement_mask: Any = field(init=False, repr=False)
    parallel_weights: Any = field(init=False, repr=False)


def create_control(config, grid, backend, equation, dealias_mask=None):
    """Prepare runtime data; initialize(state) is a separate, explicit step."""
    control = ForcingState(config, grid, backend, equation)
    configure(control, config, grid, backend, dealias_mask)
    return control


def configure(control, config: Any, grid: Any, backend: Any, dealias_mask: Any | None = None) -> None:
    required = ("forcing_fields", "forcing_metric", "forcing_native_parameters", "forcing_energy_factors",
                "forcing_branch_values", "forcing_apply_gain", "forcing_seed_branch", "forcing_characteristic_speed", "forcing_budget_work",
                "forcing_shell_density", "forcing_perpendicular_energy", "forcing_perpendicular_shell_energy", "forcing_measurement")
    missing = [name for name in required if not callable(getattr(control.equation, name, None))]
    if missing:
        raise ValueError(f"Equation does not support controlled forcing; missing hooks: {missing}.")
    control.config, control.grid, control.backend = config, grid, backend
    control.settings = config.controlled_shell
    s = control.settings
    if s is None:
        raise ValueError("controlled_shell settings are required.")
    control.fields = control.equation.forcing_fields(config)
    if not 1 <= s.kz_index < grid.Nz // 2:
        raise ValueError("kz_index must exclude the zero and Nyquist planes: 1 <= kz_index < Nz/2.")
    control.native = {b: control.equation.forcing_native_parameters(getattr(s, b), config, b) for b in s.branches}
    control.g2 = control.equation.forcing_energy_factors(config)
    if math.sqrt(control.g2["minus"]) < 0.1:
        warnings.warn("|G_minus| < 0.1: Elsasser conversion is ill conditioned. G_plus >= 2 cannot trigger this warning.",
                      RuntimeWarning, stacklevel=2)
    nx, ny, kx, ky, delta, ks, ix, iy = shell_geometry(config, s, sigma=control.equation.forcing_metric(config))
    if dealias_mask is not None and not np.all(backend.to_numpy(dealias_mask[ix, iy, s.kz_index])):
        raise ValueError("Requested forcing shell/kz is clipped by the dealias mask.")
    control.mode_count = len(ix)
    control.indices = backend.asarray(np.ravel_multi_index((ix, iy, np.full(len(ix), s.kz_index)), grid.fourier_shape))
    control.k_sigma = backend.asarray(ks[ix, iy], dtype=grid.real_dtype)
    control.normalization = float((grid.Nx * grid.Ny * grid.Nz)**2)
    # Interior rFFT modes have w=2. Keep it explicit in both template and energy.
    control.weights = backend.asarray(2.0/(4*delta[ix, iy]*control.normalization), dtype=grid.real_dtype)
    control.inv_delta = backend.asarray(np.divide(1.0, delta, out=np.zeros_like(delta), where=delta>0), dtype=grid.real_dtype)
    control.low_mask = backend.asarray((ks > 0) & (ks < s.k_sigma_min))
    if any(getattr(s, b).target_scope == "perpendicular_shell" for b in s.branches):
        control.perpendicular_indices = (backend.asarray(ix), backend.asarray(iy))
        control.measurement_mask = dealias_mask
        parallel_weights = np.full(grid.Nz//2+1, 2., dtype=grid.real_dtype)
        parallel_weights[[0, -1]] = 1.
        control.parallel_weights = backend.asarray(parallel_weights)
    control.templates, control.angles = {}, {}
    control.mode_rows = []
    # Independent seed streams keep templates identical when the phase model changes.
    seed = np.random.SeedSequence(config.forcing_seed)
    streams = seed.spawn(4)
    for b in s.branches:
        index = 0 if b == "plus" else 1
        rng = np.random.default_rng(streams[index])
        template = np.sqrt(delta[ix, iy]/2) * np.exp(2j*np.pi*rng.random(len(ix)))
        template /= math.sqrt(float(np.sum(2/(4*delta[ix, iy]*control.normalization)*abs(template)**2)))
        phi = np.zeros(len(ix))
        if s.phase_model == "fixed_gain_angle":
            phi = np.random.default_rng(streams[index+2]).uniform(-1, 1, len(ix))
            energy_weights = 2*abs(template)**2/delta[ix, iy]
            phi -= np.average(phi, weights=energy_weights)
            phi *= s.phase_angle_max / max(abs(phi))
        control.templates[b] = backend.asarray(template, dtype=grid.complex_dtype)
        control.angles[b] = backend.asarray(phi, dtype=grid.real_dtype)
        for j, (i_x, i_y) in enumerate(zip(ix, iy, strict=True)):
            control.mode_rows.append({"branch": b, "flat_index": int(np.ravel_multi_index((i_x, i_y, s.kz_index), grid.fourier_shape)),
                                  "nx": int(nx[i_x]), "ny": int(ny[i_y]), "nz": s.kz_index,
                                  "kx": float(kx[i_x]), "ky": float(ky[i_y]), "kz": 2*np.pi*s.kz_index/grid.Lz,
                                  "k_sigma": float(ks[i_x, i_y]), "k_delta": float(delta[i_x, i_y]),
                                  "rfft_weight": 2.0, "template_energy_weight": 1/len(ix), "phase_angle": float(phi[j])})
    frequency = 2*np.pi*s.kz_index/grid.Lz*config.vA
    control.interval_max = s.interval_max or 0.1/frequency
    control.interval_target = control.interval_max
    control.elapsed = 0.0
    control.initialized = False
    control.startup_work = {b: 0.0 for b in s.branches}
    control.cumulative_work = {b: 0.0 for b in s.branches}
    control.last_events: list[dict[str, Any]] = []
    control.last_by_branch: dict[str, dict[str, Any]] = {}
    control.event_count = 0
    control.diagnostic_hook = None


def shell_density(control, state: Any, branch: str):
    return control.equation.forcing_shell_density(control, state, branch)


def shell_energy(control, state: Any, branch: str) -> float:
    return control.backend.scalar_to_float(control.backend.xp.sum(shell_density(control, state, branch)))


def perpendicular_energy(control, state: Any, branch: str, *, nonzero_kz: bool=False):
    return control.equation.forcing_perpendicular_energy(control, state, branch, nonzero_kz=nonzero_kz)


def branch_energy(control, state: Any, branch: str) -> float:
    return control.backend.scalar_to_float(control.backend.xp.sum(perpendicular_energy(control, state, branch)))


def perpendicular_shell_energy(control, state: Any, branch: str):
    return control.equation.forcing_perpendicular_shell_energy(control, state, branch)


def _energy_measurement(control, state: Any, branch: str):
    return control.equation.forcing_measurement(control, state, branch)


def initialize(control, state: Any) -> None:
    if control.initialized:
        raise ValueError("Controlled forcing startup must run exactly once per evolution.")
    for b in control.settings.branches:
        shell = shell_energy(control, state, b)
        density, nonzero, perpendicular = _energy_measurement(control, state, b)
        total = control.backend.scalar_to_float(control.backend.xp.sum(density))
        if not math.isfinite(shell) or not math.isfinite(total):
            raise ValueError("Nonfinite energy at forcing startup.")
        p, spec = control.native[b], getattr(control.settings, b)
        added = max(0.0, p["startup_f"]-shell)
        if spec.control == "target" and spec.target_scope != "shell":
            controlled = total if nonzero is None else nonzero
            if perpendicular is not None:
                controlled = perpendicular
            added = min(added, max(0.0, p["target_f"]-controlled))
        if added:
            if shell > 0:
                control.equation.forcing_apply_gain(control, state, b, math.sqrt(shell+added)/math.sqrt(shell))
            else:
                control.equation.forcing_seed_branch(control, state, b, math.sqrt(added) * control.templates[b])
            control.startup_work[b] = shell_energy(control, state, b)-shell
    control.initialized = True


def advance(control, state: Any, dt: float, time: float, *, final: bool = False) -> list[dict[str, Any]]:
    """Consume one completed PDE step, optionally flushing at its final endpoint."""
    if not control.initialized:
        raise ValueError("Call initialize before advancing controlled forcing.")
    control.elapsed += _positive("forcing step dt", dt)
    control.last_events = []
    if not final and control.elapsed < control.interval_target * (1-8*np.finfo(float).eps):
        return []
    h = control.elapsed
    max_gamma = 0.0
    report_nonzero = any(getattr(control.settings, b).target_scope == "branch_nonzero_kz"
                         for b in control.settings.branches)
    report_perpendicular = any(getattr(control.settings, b).target_scope == "perpendicular_shell"
                               for b in control.settings.branches)
    for b in control.settings.branches:
        density = shell_density(control, state, b)
        shell = control.backend.scalar_to_float(control.backend.xp.sum(density))
        energy_density, nonzero, perpendicular = _energy_measurement(control, state, b)
        total = control.backend.scalar_to_float(control.backend.xp.sum(energy_density))
        event = control_event_map(getattr(control.settings, b), control.native[b], shell, total, h,
                                  nonzero_kz=nonzero, perpendicular_shell=perpendicular)
        factor = control.backend.xp.exp(event["log_gain"]*(1-1j*control.backend.xp.tan(control.angles[b])))
        before = control.equation.forcing_branch_values(control, state, b, control.indices).copy() if control.diagnostic_hook is not None else None
        control.equation.forcing_apply_gain(control, state, b, factor)
        if control.diagnostic_hook is not None:
            opposite = "minus" if b == "plus" else "plus"
            event.update(control.diagnostic_hook(branch=b, indices=control.indices, before=before,
                after=control.equation.forcing_branch_values(control, state, b, control.indices),
                opposite=control.equation.forcing_branch_values(control, state, opposite, control.indices), time=time))
        post = shell_energy(control, state, b)
        work = post-shell
        control.cumulative_work[b] += work
        kf = control.backend.scalar_to_float(control.backend.xp.sum(density*control.k_sigma)) / shell
        event.update({"time": float(time), "interval": h, "branch": b, "event": control.event_count+1,
                      "shell_pre_f": shell, "shell_post_f": post, "total_pre_f": total, "total_post_f": total+work,
                      "work_f_formula": event["work_f"], "work_f": work, "work_z": work/control.g2[b],
                      "cumulative_work_f": control.cumulative_work[b], "cumulative_work_z": control.cumulative_work[b]/control.g2[b],
                      "startup_work_f": control.startup_work[b], "startup_work_z": control.startup_work[b]/control.g2[b],
                      "log_gain_exceeded": int(abs(event["log_gain"]) > control.settings.log_gain_warn),
                      "k_f": kf, "mode_count": control.mode_count,
                      "modal_max_mean": control.backend.scalar_to_float(control.backend.xp.max(density))*control.mode_count/shell,
                      "modal_max_energy_f": control.backend.scalar_to_float(control.backend.xp.max(density)),
                      "modal_mean_energy_f": shell/control.mode_count,
                      "shell_fraction": post/(total+work), "shell_fraction_warning": int(post/(total+work)<0.05)})
        energy_keys = ["shell_pre", "shell_post", "total_pre", "total_post"]
        if report_nonzero:
            # Every branch shares one event CSV schema. Other scopes leave
            # these optional measurements blank, rather than invent values.
            event.update(nonzero_kz_pre_f=nonzero,
                         nonzero_kz_post_f=None if nonzero is None else nonzero+work)
            energy_keys += ["nonzero_kz_pre", "nonzero_kz_post"]
        if report_perpendicular:
            event.update(perpendicular_shell_pre_f=perpendicular,
                         perpendicular_shell_post_f=None if perpendicular is None else perpendicular+work)
            energy_keys += ["perpendicular_shell_pre", "perpendicular_shell_post"]
        for key in energy_keys:
            if event[f"{key}_f"] is None:
                event.update({f"{key}_z": None, f"{key}_rms_f": None, f"{key}_rms_z": None})
                continue
            event[f"{key}_z"] = event[f"{key}_f"]/control.g2[b]
            event[f"{key}_rms_f"] = 2*math.sqrt(max(0.0, event[f"{key}_f"]))
            event[f"{key}_rms_z"] = 2*math.sqrt(max(0.0, event[f"{key}_z"]))
        event["target_f"] = control.native[b]["target_f"]
        event["target_z"] = event["target_f"]/control.g2[b]
        spec = getattr(control.settings, b)
        event["control"] = spec.control
        event["input_basis"] = spec.target_basis if spec.control == "target" else spec.power_basis
        event["input_quantity"] = spec.target_quantity if spec.control == "target" else "power"
        event["target_scope"] = spec.target_scope or "not_applicable"
        control.last_events.append(event)
        control.last_by_branch[b] = event
        max_gamma = max(max_gamma, abs(event["gamma_applied"]))
    control.interval_target = min(control.interval_max, control.settings.log_gain_target/max_gamma) if max_gamma else control.interval_max
    for event in control.last_events:
        event["next_interval_target"] = control.interval_target
        event["next_interval_effective"] = max(dt, control.interval_target)
    control.elapsed = 0.0
    control.event_count += 1
    return control.last_events


def diagnostics(control, state: Any) -> dict[str, float | int]:
    """Current energies and driving-scale estimates; gains describe the last event.

    A forced branch uses its own shell-energy-weighted k_f. An unforced
    branch uses the driven branch's k_f, not an independently measured
    cascade scale. Both estimates use the opposite branch's total Elsasser
    RMS and omit alignment. They are not measured decorrelation times.
    """
    result = {"forcing_event_count": control.event_count, "forcing_elapsed": control.elapsed,
              "forcing_next_interval_target": control.interval_target,
              "forcing_tau_relative": control.grid.Lz/(2*control.config.vA)}
    energies = {}
    for b in ("plus", "minus"):
        density, nonzero, perpendicular = _energy_measurement(control, state, b)
        energy = control.backend.scalar_to_float(control.backend.xp.sum(density))
        energies[b] = energy
        prefix = f"forcing_{b}_"
        result[prefix+"total_energy_f"] = energy
        result[prefix+"total_energy_z"] = energy/control.g2[b]
        result[prefix+"total_rms_z"] = 2*math.sqrt(max(energy/control.g2[b], 0))
        if nonzero is not None:
            result[prefix+"nonzero_kz_energy_f"] = nonzero
            result[prefix+"nonzero_kz_energy_z"] = nonzero/control.g2[b]
            result[prefix+"nonzero_kz_rms_z"] = 2*math.sqrt(nonzero/control.g2[b])
        if perpendicular is not None:
            result[prefix+"perpendicular_shell_energy_f"] = perpendicular
            result[prefix+"perpendicular_shell_energy_z"] = perpendicular/control.g2[b]
            result[prefix+"perpendicular_shell_rms_z"] = 2*math.sqrt(perpendicular/control.g2[b])
        result[prefix+"cumulative_work_f"] = control.cumulative_work.get(b, 0.0)
        result[prefix+"cumulative_work_z"] = control.cumulative_work.get(b, 0.0)/control.g2[b]
        result[prefix+"startup_work_f"] = control.startup_work.get(b, 0.0)
        result[prefix+"startup_work_z"] = control.startup_work.get(b, 0.0)/control.g2[b]
        low = control.backend.scalar_to_float(control.backend.xp.sum(density*control.low_mask))
        result[prefix+"low_k_fraction"] = low/energy if energy else 0.0
        result[prefix+"low_k_warning"] = int(energy > 0 and low/energy > 0.25)
        result[prefix+"tau_recirc"] = control.grid.Lz/abs(control.equation.forcing_characteristic_speed(control.config, b))
    for b in control.settings.branches:
        prefix = f"forcing_{b}_"
        shell = shell_energy(control, state, b)
        kf = control.backend.scalar_to_float(control.backend.xp.sum(shell_density(control, state, b)*control.k_sigma))/shell if shell else math.nan
        result.update({prefix+"shell_energy_f": shell, prefix+"shell_energy_z": shell/control.g2[b],
                       prefix+"shell_rms_z": 2*math.sqrt(max(shell/control.g2[b], 0)),
                       prefix+"shell_fraction": shell/energies[b] if energies[b] else 0,
                       prefix+"shell_fraction_warning": int(energies[b]>0 and shell/energies[b]<0.05),
                       prefix+"k_f": kf, prefix+"mode_count": control.mode_count,
                       prefix+"target_f": control.native[b]["target_f"], prefix+"epsilon_f": control.native[b]["epsilon_f"]})
        for key in ("gamma_requested", "gamma_applied", "log_gain", "floor_active", "rate_cap_active",
                    "forcing_actuator_exhausted", "forcing_controller_limited", "log_gain_exceeded", "modal_max_mean"):
            result[prefix+key] = control.last_by_branch.get(b, {}).get(key, 0.0)
        result[prefix+"target_rms_z"] = 2*math.sqrt(control.native[b]["target_f"]/control.g2[b])
        result[prefix+"epsilon_z"] = control.native[b]["epsilon_f"]/control.g2[b]
    for b in ("plus", "minus"):
        opposite = "minus" if b == "plus" else "plus"
        source = b if b in control.settings.branches else opposite
        kf = result[f"forcing_{source}_k_f"]
        counter_rms = result[f"forcing_{opposite}_total_rms_z"]
        rate = kf * counter_rms
        tau = 1/rate if rate > 0 else (math.inf if rate == 0 else math.nan)
        prefix = f"forcing_{b}_"
        result.update({prefix+"tau_nl_k_sigma": kf,
                       prefix+"tau_nl": tau,
                       prefix+"tau_nl_over_relative": tau/result["forcing_tau_relative"],
                       prefix+"tau_relative_over_nl": rate*result["forcing_tau_relative"]})
    return result


def write_metadata(control, output_dir: str | Path) -> None:
    output = Path(output_dir)
    with (output/"forcing_modes.csv").open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(control.mode_rows[0]))
        writer.writeheader()
        writer.writerows(control.mode_rows)
    metadata = {"settings": settings_document(control.settings), "native_parameters": control.native,
                "interval_max": control.interval_max, "seed": control.config.forcing_seed,
                "real_dtype": str(control.grid.real_dtype), "complex_dtype": str(control.grid.complex_dtype),
                "mode_count": control.mode_count,
                "startup_work_f": control.startup_work, "target_interpretation": "relaxation_reference",
                "target_scope_definitions": {
                    "branch_total": "All modes of the controlled branch, including kz=0 (default).",
                    "branch_nonzero_kz": "All kz!=0 modes of the controlled branch; kz=0 still evolves and is included in total diagnostics.",
                    "shell": "Only the sparse forced shell at the selected nonzero parallel harmonic.",
                    "perpendicular_shell": "The same perpendicular metric band as the actuator, summed over every retained kz including kz=0. With no dealias mask, all allocated kz are included.",
                    "actuation": "Every scope changes only the same forced shell; broader measurement scopes retain the shell floor."},
                "nonlinear_time_estimate": {
                    "definition": "1 / (tau_nl_k_sigma * opposite_branch_total_Elsasser_rms)",
                    "k_sigma_definition": "instantaneous shell-energy-weighted metric wavenumber of the reference branch",
                    "reference_branch": {b: b if b in control.settings.branches else
                                         ("minus" if b == "plus" else "plus") for b in ("plus", "minus")},
                    "interpretation": "driving-scale proxy, not a measured decorrelation time; alignment omitted",
                    "empty_branches": "an empty opposite branch gives infinite tau_nl; an empty reference shell gives NaN"}}
    (output/"forcing_metadata.json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf8")


HISTORY_FIELDS = ("initialized", "elapsed", "interval_target", "interval_max", "startup_work",
                  "cumulative_work", "last_events", "last_by_branch", "event_count", "native")


def forcing_history(control):
    """JSON-compatible continuation history; templates/angles remain arrays."""
    return {key: getattr(control, key) for key in HISTORY_FIELDS}


def restore_forcing_history(control, history):
    """Restore an already validated continuation without another startup kick."""
    for key in HISTORY_FIELDS:
        setattr(control, key, history[key])
