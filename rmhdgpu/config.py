"""Configuration helpers for the rmhdgpu package."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


DEFAULT_EQUATION_SET = "s09"
DEFAULT_EQUATION_MODE = "nonlinear"


def _default_dissipation_for_fields(field_names: list[str]) -> dict[str, dict[str, float | int]]:
    template = {
        "nu_perp": 0.0,
        "nu_par": 0.0,
        "n_perp": 3,
        "n_par": 3,
    }
    return {name: deepcopy(template) for name in field_names}


def _default_force_amplitudes_for_fields(field_names: list[str]) -> dict[str, float]:
    return {name: 0.0 for name in field_names}


def _normalize_output_cadence(name: str, value: float | int | None) -> float:
    """Normalize output cadences so nonpositive values cleanly disable output."""

    if value is None:
        return 0.0
    cadence = float(value)
    if not np.isfinite(cadence):
        raise ValueError(f"{name} must be a finite number; got {value!r}.")
    if cadence <= 0.0:
        return 0.0
    return cadence


@dataclass(slots=True)
class AutoDissipationSettings:
    """Controls the optional common auto-hyperdissipation mode.

    In manual mode the existing per-field dissipation dictionary is used
    unchanged. In auto mode, the perpendicular coefficient is adapted from a
    shell-energy measurement near the chosen dissipation scale and then applied
    uniformly to every evolved field.
    """

    mode: str = "manual"
    n_perp: int = 3
    n_par: int = 3
    nu_par: float = 0.0
    kd_fraction: float = 0.6
    shell_half_width: float = 0.5
    update_every: int = 10
    smooth_factor: float = 0.2
    nu_min: float = 1.0e-12
    nu_max: float = 1.0e2
    max_update_factor: float = 2.0

    def __post_init__(self) -> None:
        if self.mode not in {"manual", "auto"}:
            raise ValueError(
                f"auto_dissipation.mode must be 'manual' or 'auto'; got {self.mode!r}."
            )

        for name in ("n_perp", "n_par", "update_every"):
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)) or int(value) <= 0:
                raise ValueError(f"auto_dissipation.{name} must be a positive integer; got {value!r}.")
            setattr(self, name, int(value))

        for name in ("nu_par", "kd_fraction", "shell_half_width", "smooth_factor", "nu_min", "nu_max"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"auto_dissipation.{name} must be finite; got {value!r}.")
            setattr(self, name, value)

        if self.nu_par < 0.0:
            raise ValueError(f"auto_dissipation.nu_par must be nonnegative; got {self.nu_par!r}.")
        if self.kd_fraction <= 0.0 or self.kd_fraction > 1.0:
            raise ValueError(
                f"auto_dissipation.kd_fraction must lie in (0, 1]; got {self.kd_fraction!r}."
            )
        if self.shell_half_width < 0.0:
            raise ValueError(
                f"auto_dissipation.shell_half_width must be nonnegative; got {self.shell_half_width!r}."
            )
        if self.smooth_factor < 0.0 or self.smooth_factor > 1.0:
            raise ValueError(
                f"auto_dissipation.smooth_factor must lie in [0, 1]; got {self.smooth_factor!r}."
            )
        if self.nu_min < 0.0:
            raise ValueError(f"auto_dissipation.nu_min must be nonnegative; got {self.nu_min!r}.")
        if self.nu_max <= 0.0:
            raise ValueError(f"auto_dissipation.nu_max must be positive; got {self.nu_max!r}.")
        if self.nu_min > self.nu_max:
            raise ValueError(
                f"auto_dissipation.nu_min must be <= nu_max; got nu_min={self.nu_min}, nu_max={self.nu_max}."
            )

        self.max_update_factor = float(self.max_update_factor)
        if not np.isfinite(self.max_update_factor) or self.max_update_factor < 1.0:
            raise ValueError(
                "auto_dissipation.max_update_factor must be finite and >= 1.0; "
                f"got {self.max_update_factor!r}."
            )

    @property
    def enabled(self) -> bool:
        return self.mode == "auto"


@dataclass(slots=True)
class Config:
    """Container for simulation-wide parameters.

    This first-pass configuration is intentionally small and explicit. It
    validates domain size, output cadence, backend choice, field names, and
    dissipation keys. Dtypes are normalized to NumPy dtype objects so they can
    be reused consistently by both NumPy and CuPy backends.
    """

    equation_set: str = DEFAULT_EQUATION_SET
    equation_mode: str = DEFAULT_EQUATION_MODE
    Nx: int = 16
    Ny: int = 16
    Nz: int = 16
    Lx: float = 2.0 * np.pi
    Ly: float = 2.0 * np.pi
    Lz: float = 2.0 * np.pi
    backend: str = "numpy"
    fft_workers: int | None = None
    real_dtype: Any = np.float64
    complex_dtype: Any = np.complex128
    tmax: float = 1.0
    dt_init: float = 1.0e-2
    cfl_number: float = 0.3
    dt_min: float | None = None
    dt_max: float | None = None
    use_variable_dt: bool = True
    runtime_check_every: int = 10
    progress_output_every: int | None = 100
    fail_on_nonfinite: bool = True
    t_out_scal: float = 0.1
    t_out_spec: float = 0.0
    t_out_full: float = 0.0
    dealias: bool = True
    dealias_mode: str = "two_thirds"
    vA: float = 1.0
    cs2_over_vA2: float = 1.0
    N2: float = 1.0
    g: float = 0.0
    K_p0: float = 0.0
    K_rho0: float = 0.0
    use_forcing: bool = False
    n_min_force: float = 1.0
    n_max_force: float = 3.0
    alpha_force: float = 0.0
    force_amplitudes: dict[str, float] | None = None
    forcing_seed: int | None = None
    field_names: list[str] | None = None
    dissipation: dict[str, dict[str, float | int]] | None = None
    auto_dissipation: AutoDissipationSettings | dict[str, Any] | None = None

    def __post_init__(self) -> None:
        from rmhdgpu.equations import get_equation_module

        self.equation_set = str(self.equation_set)
        self.equation_mode = str(self.equation_mode)
        if self.equation_mode not in {"nonlinear", "linear"}:
            raise ValueError(
                f"equation_mode must be 'nonlinear' or 'linear'; got {self.equation_mode!r}."
            )
        try:
            equation_module = get_equation_module(self.equation_set)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        if self.field_names is None:
            self.field_names = list(equation_module.FIELD_NAMES)
        else:
            self.field_names = list(self.field_names)

        for name in ("Nx", "Ny", "Nz"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer; got {value!r}.")

        if self.Nz % 2 != 0:
            raise ValueError(f"Nz must be even for the rFFT layout; got Nz={self.Nz}.")

        for name in ("Lx", "Ly", "Lz"):
            value = float(getattr(self, name))
            if value <= 0.0:
                raise ValueError(f"{name} must be positive; got {value!r}.")
            setattr(self, name, value)

        for name in ("tmax", "dt_init", "cfl_number"):
            value = float(getattr(self, name))
            if value <= 0.0:
                raise ValueError(f"{name} must be positive; got {value!r}.")
            setattr(self, name, value)

        for name in ("t_out_scal", "t_out_spec", "t_out_full"):
            setattr(self, name, _normalize_output_cadence(name, getattr(self, name)))

        if self.dt_min is not None:
            self.dt_min = float(self.dt_min)
            if self.dt_min <= 0.0:
                raise ValueError(f"dt_min must be positive when provided; got {self.dt_min!r}.")

        if self.dt_max is not None:
            self.dt_max = float(self.dt_max)
            if self.dt_max <= 0.0:
                raise ValueError(f"dt_max must be positive when provided; got {self.dt_max!r}.")

        if self.dt_min is not None and self.dt_max is not None and self.dt_min > self.dt_max:
            raise ValueError(
                f"dt_min must be <= dt_max; got dt_min={self.dt_min}, dt_max={self.dt_max}."
            )

        if self.backend not in {"numpy", "scipy_cpu", "cupy"}:
            raise ValueError(
                f"backend must be 'numpy', 'scipy_cpu', or 'cupy'; got {self.backend!r}."
            )

        if self.fft_workers is not None:
            if not isinstance(self.fft_workers, (int, np.integer)) or self.fft_workers <= 0:
                raise ValueError(
                    f"fft_workers must be a positive integer when provided; got {self.fft_workers!r}."
                )
            self.fft_workers = int(self.fft_workers)

        if not isinstance(self.use_variable_dt, bool):
            raise ValueError(f"use_variable_dt must be bool; got {self.use_variable_dt!r}.")
        if not isinstance(self.fail_on_nonfinite, bool):
            raise ValueError(f"fail_on_nonfinite must be bool; got {self.fail_on_nonfinite!r}.")
        if not isinstance(self.use_forcing, bool):
            raise ValueError(f"use_forcing must be bool; got {self.use_forcing!r}.")
        if not isinstance(self.runtime_check_every, (int, np.integer)) or self.runtime_check_every <= 0:
            raise ValueError(
                f"runtime_check_every must be a positive integer; got {self.runtime_check_every!r}."
            )
        self.runtime_check_every = int(self.runtime_check_every)
        if self.progress_output_every is not None:
            if not isinstance(self.progress_output_every, (int, np.integer)) or self.progress_output_every <= 0:
                raise ValueError(
                    f"progress_output_every must be a positive integer when provided; "
                    f"got {self.progress_output_every!r}."
                )
            self.progress_output_every = int(self.progress_output_every)

        self.real_dtype = np.dtype(self.real_dtype)
        self.complex_dtype = np.dtype(self.complex_dtype)

        if self.real_dtype.kind != "f":
            raise ValueError(
                f"real_dtype must be a real floating dtype; got {self.real_dtype}."
            )
        if self.complex_dtype.kind != "c":
            raise ValueError(
                f"complex_dtype must be a complex dtype; got {self.complex_dtype}."
            )

        if len(self.field_names) == 0:
            raise ValueError("field_names must contain at least one field.")
        if len(set(self.field_names)) != len(self.field_names):
            raise ValueError(f"field_names must be unique; got {self.field_names!r}.")

        self.n_min_force = float(self.n_min_force)
        self.n_max_force = float(self.n_max_force)
        self.alpha_force = float(self.alpha_force)
        if self.n_min_force < 0.0:
            raise ValueError(f"n_min_force must be nonnegative; got {self.n_min_force!r}.")
        if self.n_max_force <= self.n_min_force:
            raise ValueError(
                f"n_max_force must be greater than n_min_force; got "
                f"n_min_force={self.n_min_force}, n_max_force={self.n_max_force}."
            )
        if self.alpha_force < 0.0:
            raise ValueError(f"alpha_force must be nonnegative; got {self.alpha_force!r}.")
        self.vA = float(self.vA)
        self.cs2_over_vA2 = float(self.cs2_over_vA2)
        self.N2 = float(self.N2)
        self.g = float(self.g)
        self.K_p0 = float(self.K_p0)
        self.K_rho0 = float(self.K_rho0)
        if not np.isfinite(self.N2):
            raise ValueError(f"N2 must be finite; got {self.N2!r}.")
        if self.N2 == 0.0:
            raise ValueError(f"N2 must be nonzero; got {self.N2!r}.")
        if self.forcing_seed is not None:
            if not isinstance(self.forcing_seed, (int, np.integer)):
                raise ValueError(
                    f"forcing_seed must be an integer when provided; got {self.forcing_seed!r}."
                )
            self.forcing_seed = int(self.forcing_seed)

        if self.force_amplitudes is None:
            self.force_amplitudes = _default_force_amplitudes_for_fields(self.field_names)
        else:
            self.force_amplitudes = deepcopy(self.force_amplitudes)

        force_keys = set(self.force_amplitudes)
        valid_keys = set(self.field_names)
        extra_force_keys = sorted(force_keys - valid_keys)
        if extra_force_keys:
            raise ValueError(
                "force_amplitudes keys must be a subset of field_names; "
                f"unexpected keys: {extra_force_keys}."
            )

        cleaned_force_amplitudes = _default_force_amplitudes_for_fields(self.field_names)
        for field_name, amplitude in self.force_amplitudes.items():
            amplitude_value = float(amplitude)
            if amplitude_value < 0.0:
                raise ValueError(
                    f"force_amplitudes[{field_name!r}] must be nonnegative; got {amplitude_value!r}."
                )
            cleaned_force_amplitudes[field_name] = amplitude_value
        self.force_amplitudes = cleaned_force_amplitudes

        if self.dissipation is None:
            self.dissipation = _default_dissipation_for_fields(self.field_names)
        else:
            self.dissipation = deepcopy(self.dissipation)

        if self.auto_dissipation is None:
            self.auto_dissipation = AutoDissipationSettings()
        elif isinstance(self.auto_dissipation, dict):
            self.auto_dissipation = AutoDissipationSettings(**deepcopy(self.auto_dissipation))
        elif not isinstance(self.auto_dissipation, AutoDissipationSettings):
            raise ValueError(
                "auto_dissipation must be an AutoDissipationSettings instance, a dict, or None; "
                f"got {type(self.auto_dissipation).__name__}."
            )

        dissipation_keys = set(self.dissipation)
        expected_keys = set(self.field_names)
        if dissipation_keys != expected_keys:
            missing = sorted(expected_keys - dissipation_keys)
            extra = sorted(dissipation_keys - expected_keys)
            parts: list[str] = []
            if missing:
                parts.append(f"missing keys: {missing}")
            if extra:
                parts.append(f"unexpected keys: {extra}")
            detail = ", ".join(parts)
            raise ValueError(
                "dissipation keys must match field_names exactly; " + detail + "."
            )

        cleaned_dissipation: dict[str, dict[str, float | int]] = {}
        for field_name in self.field_names:
            entry = deepcopy(self.dissipation[field_name])
            for coeff_name in ("nu_perp", "nu_par"):
                if coeff_name not in entry:
                    raise ValueError(
                        f"dissipation[{field_name!r}] is missing coefficient {coeff_name!r}."
                    )
                entry[coeff_name] = float(entry[coeff_name])
                if entry[coeff_name] < 0.0:
                    raise ValueError(
                        f"dissipation[{field_name!r}][{coeff_name!r}] must be nonnegative; "
                        f"got {entry[coeff_name]!r}."
                    )

            for order_name in ("n_perp", "n_par"):
                if order_name not in entry:
                    raise ValueError(
                        f"dissipation[{field_name!r}] is missing order {order_name!r}."
                    )
                order = entry[order_name]
                if not isinstance(order, (int, np.integer)) or order < 0:
                    raise ValueError(
                        f"dissipation[{field_name!r}][{order_name!r}] must be a nonnegative integer; "
                        f"got {order!r}."
                    )
            cleaned_dissipation[field_name] = entry

        self.dissipation = cleaned_dissipation


def config_to_dict(config: Config) -> dict[str, Any]:
    """Return a deep-copied plain-Python representation of a `Config`."""

    data = asdict(config)
    data["real_dtype"] = str(np.dtype(config.real_dtype))
    data["complex_dtype"] = str(np.dtype(config.complex_dtype))
    return data


def default_config_dict() -> dict[str, Any]:
    """Return the fully resolved default configuration as a plain dict."""

    return config_to_dict(Config())


def default_config_dict_for_equation(equation_set: str) -> dict[str, Any]:
    """Return resolved default configuration values for one equation set."""

    return config_to_dict(Config(equation_set=equation_set))
