"""Built-in initial conditions and the lightweight initcond registry.

This module is the single source of truth for input-file selectable initial
conditions. Add a new built-in initial condition by:

1. writing a builder with signature
   `builder(*, parameters, grid, backend, fft, dealias_mask, field_names, params) -> State`
2. writing a small parameter normalizer for that builder
3. registering the builder with `@register_initial_condition("name", ...)`

For equation-specific eigenmode algebra, prefer adding a small helper module
such as `eigenmodes_<equation>.py` and keeping the registered builder here as a
thin wrapper. That keeps this registry readable as more equation sets are
added.

`rmhdgpu.run` and the example scripts both dispatch through
`build_initial_state(...)` so every registered name follows the same path.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from rmhdgpu.equations import get_equation_module
from rmhdgpu.forcing import shaped_random_real_field, shaped_random_real_field_perp_prl
from rmhdgpu.initconds.eigenmodes_low_beta_stratified import low_beta_stratified_mode_state
from rmhdgpu.initconds.eigenmodes_s09 import alfven_mode_state
from rmhdgpu.masks import apply_mask
from rmhdgpu.operators import dx, dy, lap_perp
from rmhdgpu.state import State


InitialConditionBuilder = Callable[..., State]
ParameterNormalizer = Callable[[dict[str, Any]], dict[str, Any]]


RANDOM_SPECTRUM_DEFAULTS = {
    "n_min": 1.0,
    "n_max": 3.0,
    "alpha": 0.0,
    "init_energy": 0.75,
    "seed": 0,
    "exclude_kpar0": True,   # drop the marginal k_par=0, k_perp!=0 plane
}

RANDOM_SPECTRUM_ONE_WAVE_DEFAULTS = {
    "n_min_perp": 1.0,
    "n_min_prl": 1.0,
    "n_max_perp": 3.0,
    "n_max_prl": 3.0,
    "alpha": 0.0,
    "alpha_prl": 0.0,
    "init_energy": 0.75,
    "seed": 0,
    "exclude_kpar0": True,
}


@dataclass(frozen=True, slots=True)
class InitialConditionDefinition:
    """Registered initial-condition metadata."""

    name: str
    builder: InitialConditionBuilder
    normalize_parameters: ParameterNormalizer
    description: str


_REGISTRY: dict[str, InitialConditionDefinition] = {}


def register_initial_condition(
    name: str,
    *,
    normalize_parameters: ParameterNormalizer | None = None,
    description: str = "",
) -> Callable[[InitialConditionBuilder], InitialConditionBuilder]:
    """Register a named initial-condition builder."""

    registry_name = str(name)

    def decorator(builder: InitialConditionBuilder) -> InitialConditionBuilder:
        if registry_name in _REGISTRY:
            raise ValueError(f"Initial condition {registry_name!r} is already registered.")
        _REGISTRY[registry_name] = InitialConditionDefinition(
            name=registry_name,
            builder=builder,
            normalize_parameters=_identity_parameters if normalize_parameters is None else normalize_parameters,
            description=description,
        )
        return builder

    return decorator


def list_initial_condition_types() -> list[str]:
    """Return the registered input-file initial-condition names."""

    return sorted(_REGISTRY)


def get_initial_condition_builder(name: str) -> InitialConditionBuilder:
    """Return the registered builder for `name`."""

    return _get_initial_condition_definition(name).builder


def normalize_initial_condition_parameters(name: str, parameters: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate and normalize parameters for a registered initial condition."""

    definition = _get_initial_condition_definition(name)
    raw_parameters = _as_parameter_dict(parameters)
    try:
        return definition.normalize_parameters(raw_parameters)
    except ValueError as exc:
        raise ValueError(f"Invalid parameters for initial condition {name!r}: {exc}") from exc


def build_initial_state(
    initial_condition: Any,
    *,
    parameters: Mapping[str, Any] | None = None,
    grid: Any,
    backend: Any,
    fft: Any,
    dealias_mask: Any | None,
    field_names: Sequence[str],
    params: Any,
) -> State:
    """Build an initial state from a registered name or spec-like object."""

    kind, raw_parameters = _coerce_initial_condition_request(initial_condition, parameters=parameters)
    normalized_parameters = normalize_initial_condition_parameters(kind, raw_parameters)
    builder = get_initial_condition_builder(kind)
    return builder(
        parameters=normalized_parameters,
        grid=grid,
        backend=backend,
        fft=fft,
        dealias_mask=dealias_mask,
        field_names=field_names,
        params=params,
    )


def aw_packet_real_field(grid: Any, backend: Any) -> Any:
    """Return the large-amplitude Alfvénic packet used by the examples."""

    xp = backend.xp
    x = grid.x.reshape(grid.Nx, 1, 1)
    y = grid.y.reshape(1, grid.Ny, 1)
    z = grid.z.reshape(1, 1, grid.Nz)
    field = xp.zeros(grid.real_shape, dtype=grid.real_dtype)

    for nx in range(1, 4):
        for ny in range(1, 4):
            for nz in range(1, 4):
                coefficient = 0.15 / (nx + ny + nz - 1.0)
                phase = 0.3 * (nx - ny + nz)
                field += coefficient * xp.cos(nx * x + ny * y + nz * z + phase)

    rms = backend.scalar_to_float(xp.sqrt(xp.mean(field**2)))
    field *= 1.0 / rms
    return field


def initial_u_rms(phi_hat: Any, grid: Any, fft: Any, backend: Any) -> float:
    """Return the perpendicular RMS velocity associated with `phi_hat`."""

    ux = -fft.c2r(dy(phi_hat, grid))
    uy = fft.c2r(dx(phi_hat, grid))
    xp = backend.xp
    return backend.scalar_to_float(xp.sqrt(xp.mean(ux**2 + uy**2)))


def _identity_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    return dict(parameters)


def _get_initial_condition_definition(name: str) -> InitialConditionDefinition:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(list_initial_condition_types())
        raise ValueError(
            f"Unknown initial condition type {name!r}. Available initial conditions: {available}."
        ) from exc


def _as_parameter_dict(parameters: Mapping[str, Any] | None) -> dict[str, Any]:
    if parameters is None:
        return {}
    if not isinstance(parameters, Mapping):
        raise ValueError("initial-condition parameters must be a mapping.")
    return {str(key): value for key, value in parameters.items()}


def _coerce_initial_condition_request(
    initial_condition: Any,
    *,
    parameters: Mapping[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    if isinstance(initial_condition, str):
        return initial_condition, _as_parameter_dict(parameters)

    if isinstance(initial_condition, Mapping):
        if "type" not in initial_condition:
            raise ValueError("initial_condition mappings must contain a 'type' entry.")
        merged = {
            key: value
            for key, value in initial_condition.items()
            if key not in {"type", "parameters"}
        }
        merged.update(_as_parameter_dict(initial_condition.get("parameters")))
        merged.update(_as_parameter_dict(parameters))
        return str(initial_condition["type"]), merged

    if hasattr(initial_condition, "type"):
        merged = _as_parameter_dict(getattr(initial_condition, "parameters", None))
        merged.update(_as_parameter_dict(parameters))
        return str(initial_condition.type), merged

    raise TypeError(
        "initial_condition must be a registered name, a mapping with 'type', "
        "or an object with 'type' and 'parameters' attributes."
    )


def _reject_unknown_parameters(kind: str, parameters: dict[str, Any], allowed: set[str]) -> None:
    unexpected = sorted(set(parameters) - allowed)
    if unexpected:
        raise ValueError(f"unexpected parameters {unexpected}; allowed parameters are {sorted(allowed)}.")


def _require_fields(kind: str, field_names: Sequence[str], required: Sequence[str]) -> None:
    missing = [name for name in required if name not in field_names]
    if missing:
        raise ValueError(
            f"Initial condition {kind!r} requires fields {list(required)!r}; missing {missing!r}."
        )


def _masked_r2c(real_field: Any, *, fft: Any, dealias_mask: Any | None) -> Any:
    field_hat = fft.r2c(real_field)
    if dealias_mask is not None:
        apply_mask(field_hat, dealias_mask)
    return field_hat


def _single_stored_mode_weight(grid: Any, iz: int) -> float:
    if iz == 0:
        return 1.0
    if grid.Nz % 2 == 0 and iz == grid.Nz // 2:
        return 1.0
    return 2.0


def _single_mode_real_rms_to_stored_amplitude(grid: Any, amplitude: float, iz: int) -> float:
    normalization = float(np.prod(grid.real_shape))
    return float(amplitude) * normalization / np.sqrt(_single_stored_mode_weight(grid, iz))


def _resolve_single_fourier_mode_indices(grid: Any, k_indices: Sequence[int]) -> tuple[int, int, int]:
    if len(k_indices) != 3:
        raise ValueError(f"k_indices must have length 3; got {k_indices!r}.")

    kx, ky, kz = (int(k_indices[0]), int(k_indices[1]), int(k_indices[2]))
    if abs(kx) >= grid.Nx // 2:
        raise ValueError(f"k_indices[0] must satisfy |kx| < Nx//2; got {kx} for Nx={grid.Nx}.")
    if abs(ky) >= grid.Ny // 2:
        raise ValueError(f"k_indices[1] must satisfy |ky| < Ny//2; got {ky} for Ny={grid.Ny}.")
    if kz < 0 or kz >= grid.Nz // 2:
        raise ValueError(f"k_indices[2] must satisfy 0 <= kz < Nz//2; got {kz} for Nz={grid.Nz}.")

    return kx % grid.Nx, ky % grid.Ny, kz


def _normalize_zero_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    _reject_unknown_parameters("zero", parameters, set())
    return {}


def _normalize_aw_packet_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    _reject_unknown_parameters("aw_packet", parameters, set())
    return {}


def _normalize_alfven_mode_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    allowed = {"k_indices", "amplitude", "branch"}
    _reject_unknown_parameters("alfven_mode", parameters, allowed)

    raw_k = parameters.get("k_indices", [1, 1, 1])
    if not isinstance(raw_k, (list, tuple)) or len(raw_k) != 3:
        raise ValueError("k_indices must be an array of three integers.")

    k_indices: list[int] = []
    for index, value in enumerate(raw_k):
        if not isinstance(value, int):
            raise ValueError(f"k_indices[{index}] must be an integer; got {value!r}.")
        if value < 0:
            raise ValueError(f"k_indices[{index}] must be nonnegative; got {value!r}.")
        k_indices.append(int(value))

    amplitude = float(parameters.get("amplitude", 1.0))
    if amplitude <= 0.0:
        raise ValueError(f"amplitude must be positive; got {amplitude!r}.")

    branch = str(parameters.get("branch", "plus"))
    if branch not in {"plus", "minus"}:
        raise ValueError(f"branch must be 'plus' or 'minus'; got {branch!r}.")

    return {
        "k_indices": k_indices,
        "amplitude": amplitude,
        "branch": branch,
    }


def _normalize_low_beta_mode_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    allowed = {"k_indices", "amplitude", "mode"}
    _reject_unknown_parameters("low_beta_stratified_mode", parameters, allowed)

    raw_k = parameters.get("k_indices", [0, 1, 0])
    if not isinstance(raw_k, (list, tuple)) or len(raw_k) != 3:
        raise ValueError("k_indices must be an array of three integers.")
    k_indices: list[int] = []
    for index, value in enumerate(raw_k):
        if not isinstance(value, int):
            raise ValueError(f"k_indices[{index}] must be an integer; got {value!r}.")
        if value < 0:
            raise ValueError(f"k_indices[{index}] must be nonnegative; got {value!r}.")
        k_indices.append(int(value))

    amplitude = float(parameters.get("amplitude", 1.0))
    if amplitude <= 0.0:
        raise ValueError(f"amplitude must be positive; got {amplitude!r}.")

    mode = str(parameters.get("mode", "unstable_growing"))
    allowed_modes = {"unstable_growing", "unstable_decaying", "stable_plus", "stable_minus"}
    if mode not in allowed_modes:
        raise ValueError(f"mode must be one of {sorted(allowed_modes)}; got {mode!r}.")

    return {"k_indices": k_indices, "amplitude": amplitude, "mode": mode}


def _normalize_single_fourier_mode_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    allowed = {"k_indices", "amplitude", "seed"}
    _reject_unknown_parameters("single_fourier_mode", parameters, allowed)

    raw_k = parameters.get("k_indices", [1, 1, 1])
    if not isinstance(raw_k, (list, tuple)) or len(raw_k) != 3:
        raise ValueError("k_indices must be an array of three integers.")

    k_indices: list[int] = []
    for index, value in enumerate(raw_k):
        if not isinstance(value, int):
            raise ValueError(f"k_indices[{index}] must be an integer; got {value!r}.")
        if index == 2 and value < 0:
            raise ValueError(f"k_indices[{index}] must be nonnegative; got {value!r}.")
        k_indices.append(int(value))

    amplitude = float(parameters.get("amplitude", 0.1))
    if amplitude <= 0.0:
        raise ValueError(f"amplitude must be positive; got {amplitude!r}.")

    seed = int(parameters.get("seed", 0))
    return {"k_indices": k_indices, "amplitude": amplitude, "seed": seed}


def _normalize_random_spectrum_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    _reject_unknown_parameters("random_spectrum", parameters, set(RANDOM_SPECTRUM_DEFAULTS))

    normalized = dict(RANDOM_SPECTRUM_DEFAULTS)
    normalized.update(parameters)

    for key in ("n_min", "n_max", "alpha", "init_energy"):
        normalized[key] = float(normalized[key])
        if not np.isfinite(normalized[key]):
            raise ValueError(f"{key} must be finite; got {normalized[key]!r}.")

    normalized["seed"] = int(normalized["seed"])
    normalized["exclude_kpar0"] = bool(normalized["exclude_kpar0"])

    if normalized["n_min"] < 0.0:
        raise ValueError(f"n_min must be nonnegative; got {normalized['n_min']!r}.")
    if normalized["n_max"] <= normalized["n_min"]:
        raise ValueError(
            f"n_max must be greater than n_min; got n_min={normalized['n_min']!r}, "
            f"n_max={normalized['n_max']!r}."
        )
    if normalized["alpha"] < 0.0:
        raise ValueError(f"alpha must be nonnegative; got {normalized['alpha']!r}.")
    if normalized["init_energy"] == 0.0:
        raise ValueError("init_energy must be nonzero so the spectrum can be normalized.")

    return normalized

def _normalize_random_spectrum_one_wave_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    _reject_unknown_parameters("random_spectrum_one_wave", parameters, set(RANDOM_SPECTRUM_ONE_WAVE_DEFAULTS))

    normalized = dict(RANDOM_SPECTRUM_ONE_WAVE_DEFAULTS)
    normalized.update(parameters)

    for key in ("n_min_perp", "n_min_prl", "n_max_perp", "n_max_prl", "alpha", "alpha_prl", "init_energy"):
        normalized[key] = float(normalized[key])
        if not np.isfinite(normalized[key]):
            raise ValueError(f"{key} must be finite; got {normalized[key]!r}.")

    normalized["seed"] = int(normalized["seed"])
    normalized["exclude_kpar0"] = bool(normalized["exclude_kpar0"])

    if normalized["n_min_perp"] < 0.0 or normalized["n_min_prl"] < 0.0:
        raise ValueError(
            f"n_min_perp and n_min_prl must be nonnegative; got "
            f"n_min_perp={normalized['n_min_perp']!r}, n_min_prl={normalized['n_min_prl']!r}."
        )
    if normalized["n_max_perp"] < normalized["n_min_perp"] or normalized["n_max_prl"] < normalized["n_min_prl"]:
        raise ValueError(
            f"n_max must be at least n_min in each direction; got "
            f"perp=({normalized['n_min_perp']!r}, {normalized['n_max_perp']!r}), "
            f"prl=({normalized['n_min_prl']!r}, {normalized['n_max_prl']!r})."
        )
    if normalized["alpha"] < 0.0:
        raise ValueError(f"alpha must be nonnegative; got {normalized['alpha']!r}.")
    if normalized["alpha_prl"] < 0.0:
        raise ValueError(f"alpha_prl must be nonnegative; got {normalized['alpha_prl']!r}.")
    if normalized["init_energy"] == 0.0:
        raise ValueError("init_energy must be nonzero so the spectrum can be normalized.")

    return normalized

def _rescale_state_to_total_energy(
    state: State,
    *,
    target_energy: float,
    grid: Any,
    backend: Any,
    params: Any,
) -> State:
    """Rescale all evolved fields so the equation-module total energy matches `target_energy`."""

    equation_module = get_equation_module(getattr(params, "equation_set"))
    current_energy = float(equation_module.total_energy(state, grid, backend, params))
    if not np.isfinite(current_energy):
        raise ValueError("random_spectrum generated a non-finite total energy.")
    if abs(current_energy) <= 1.0e-30:
        raise ValueError(
            "random_spectrum generated a state with near-zero total energy, so it cannot be "
            "normalized to the requested init_energy."
        )
    if current_energy * target_energy < 0.0:
        raise ValueError(
            "random_spectrum generated a state whose total-energy sign does not match init_energy. "
            "Choose a different seed or target energy."
        )

    scale = float(np.sqrt(target_energy / current_energy))
    for field_name in state.field_names:
        state[field_name][...] *= scale
    return state


@register_initial_condition(
    "zero",
    normalize_parameters=_normalize_zero_parameters,
    description="All-zero Fourier state.",
)
def zero(
    *,
    parameters: Mapping[str, Any] | None = None,
    grid: Any,
    backend: Any,
    fft: Any,
    dealias_mask: Any | None,
    field_names: Sequence[str],
    params: Any,
) -> State:
    """Build an all-zero initial state."""

    _normalize_zero_parameters(_as_parameter_dict(parameters))
    return State(grid, backend, field_names=list(field_names))


@register_initial_condition(
    "alfven_mode",
    normalize_parameters=_normalize_alfven_mode_parameters,
    description="Exact linear Alfvén eigenmode in Fourier space.",
)
def alfven_mode(
    *,
    parameters: Mapping[str, Any] | None = None,
    grid: Any,
    backend: Any,
    fft: Any,
    dealias_mask: Any | None,
    field_names: Sequence[str],
    params: Any,
) -> State:
    """Build an exact linear Alfvén eigenmode."""

    normalized = _normalize_alfven_mode_parameters(_as_parameter_dict(parameters))
    _require_fields("alfven_mode", field_names, ("psi", "omega"))
    return alfven_mode_state(
        grid=grid,
        backend=backend,
        field_names=list(field_names),
        k_indices=normalized["k_indices"],
        amplitude=normalized["amplitude"],
        branch=normalized["branch"],
        params=params,
    )


@register_initial_condition(
    "low_beta_stratified_mode",
    normalize_parameters=_normalize_low_beta_mode_parameters,
    description="Single linear eigenmode of the low_beta_stratified equation set.",
)
def low_beta_stratified_mode(
    *,
    parameters: Mapping[str, Any] | None = None,
    grid: Any,
    backend: Any,
    fft: Any,
    dealias_mask: Any | None,
    field_names: Sequence[str],
    params: Any,
) -> State:
    """Build a single low-beta stratified linear eigenmode."""

    normalized = _normalize_low_beta_mode_parameters(_as_parameter_dict(parameters))
    _require_fields("low_beta_stratified_mode", field_names, ("psi", "omega", "a"))

    state = low_beta_stratified_mode_state(
        grid=grid,
        backend=backend,
        field_names=list(field_names),
        k_indices=normalized["k_indices"],
        amplitude=normalized["amplitude"],
        mode=normalized["mode"],
        params=params,
    )
    if dealias_mask is not None:
        state.apply_mask(dealias_mask)
    return state


@register_initial_condition(
    "single_fourier_mode",
    normalize_parameters=_normalize_single_fourier_mode_parameters,
    description="Populate one Fourier mode in every evolved field with independent random coefficients.",
)
def single_fourier_mode(
    *,
    parameters: Mapping[str, Any] | None = None,
    grid: Any,
    backend: Any,
    fft: Any,
    dealias_mask: Any | None,
    field_names: Sequence[str],
    params: Any,
) -> State:
    """Build a generic single-mode random state for the active field list.

    Each evolved field receives an independent complex Gaussian coefficient at
    the same stored Fourier mode. The amplitude parameter sets the expected
    real-space RMS of each initialized field.
    """

    del fft, params
    normalized = _normalize_single_fourier_mode_parameters(_as_parameter_dict(parameters))
    ix, iy, iz = _resolve_single_fourier_mode_indices(grid, normalized["k_indices"])
    coeff_scale = _single_mode_real_rms_to_stored_amplitude(grid, normalized["amplitude"], iz)
    rng = np.random.default_rng(normalized["seed"])

    state = State(grid, backend, field_names=list(field_names))
    for field_name in state.field_names:
        coefficient = coeff_scale * (rng.normal() + 1j * rng.normal()) / np.sqrt(2.0)
        state[field_name][ix, iy, iz] = coefficient

    if dealias_mask is not None:
        state.apply_mask(dealias_mask)
    return state


@register_initial_condition(
    "aw_packet",
    normalize_parameters=_normalize_aw_packet_parameters,
    description="Large-amplitude nonlinear Alfvénic packet used by the examples.",
)
def aw_packet(
    *,
    parameters: Mapping[str, Any] | None = None,
    grid: Any,
    backend: Any,
    fft: Any,
    dealias_mask: Any | None,
    field_names: Sequence[str],
    params: Any,
) -> State:
    """Build the large-amplitude Alfvénic packet example state."""

    _normalize_aw_packet_parameters(_as_parameter_dict(parameters))
    _require_fields("aw_packet", field_names, ("psi", "omega"))

    phi_hat = _masked_r2c(aw_packet_real_field(grid, backend), fft=fft, dealias_mask=dealias_mask)
    state = State(grid, backend, field_names=list(field_names))
    state["psi"][...] = phi_hat
    state["omega"][...] = lap_perp(phi_hat, grid)
    return state


@register_initial_condition(
    "random_spectrum",
    normalize_parameters=_normalize_random_spectrum_parameters,
    description="Band-limited random multifield spectrum normalized to a target total energy.",
)
def random_spectrum(
    *,
    parameters: Mapping[str, Any] | None = None,
    grid: Any,
    backend: Any,
    fft: Any,
    dealias_mask: Any | None,
    field_names: Sequence[str],
    params: Any,
) -> State:
    """Build a random multifield spectrum for the active equation set.

    Each evolved field gets an independent random real field whose support is
    limited to the shell band `n_min <= n <= n_max`, where `n` is the integer
    mode-number magnitude. The Fourier amplitudes are shaped so the modal
    energy is approximately proportional to `n^(-alpha)`. The complete state is
    then rescaled so the equation-module `total_energy(...)` equals
    `init_energy`.
    """

    state = State(grid, backend, field_names=list(field_names))
    normalized = _normalize_random_spectrum_parameters(_as_parameter_dict(parameters))
    rng = backend.random_generator(normalized["seed"])

    for field_name in state.field_names:
        _, field_hat = shaped_random_real_field(
            grid,
            backend,
            fft,
            n_min_force=normalized["n_min"],
            n_max_force=normalized["n_max"],
            alpha_force=0.5 * normalized["alpha"],
            rng=rng,
        )
        state[field_name][...] = field_hat

    if dealias_mask is not None:
        state.apply_mask(dealias_mask)

    # Drop the k_par = 0 plane: it is a marginal (zero-frequency, defective)
    # subspace where psi is frozen and du_par is driven secularly by
    # -vA*K_b0*dy(psi), so seeding it makes total_energy grow ~ t^2. Applied
    # before energy rescaling so init_energy is normalized over surviving modes.
    if normalized["exclude_kpar0"]:
        kpar_nonzero = grid.kz != 0
        for field_name in state.field_names:
            state[field_name][...] *= kpar_nonzero

    return _rescale_state_to_total_energy(
        state,
        target_energy=normalized["init_energy"],
        grid=grid,
        backend=backend,
        params=params,
    )

@register_initial_condition(
    "random_spectrum_one_wave",
    normalize_parameters=_normalize_random_spectrum_one_wave_parameters,
    description="Band-limited random pure z^+ Alfvenic state (Phi = Psi), compressive fields zero.",
)
def random_spectrum_one_wave(
    *,
    parameters: Mapping[str, Any] | None = None,
    grid: Any,
    backend: Any,
    fft: Any,
    dealias_mask: Any | None,
    field_names: Sequence[str],
    params: Any,
) -> State:
    """Build a random multifield spectrum for the active equation set.

    We send one alfven wave, so we set an initial condition of z+, and rest
    of the evolution fields to be 0. Hence we set the initial condition 
    of Phi to some random number then set Psi = Phi, and rest of the variables to 0. 
    
    The variable Phi gets an independent random real field whose support is
    limited to the shell band `n_min_prl <= n_z <= n_max_prl`, 'n_min_perp <=
    sqrt(nx^2 + ny^2) <= n_max_perp, where `n` is the integer
    mode-number magnitude.  The Fourier amplitudes are shaped so the modal
    energy is approximately proportional to
    `n_perp^(-alpha) * n_prl^(-alpha_prl)`. The complete state is
    then rescaled so the equation-module `total_energy(...)` equals
    `init_energy`.
    """

    state = State(grid, backend, field_names=list(field_names))
    normalized = _normalize_random_spectrum_one_wave_parameters(_as_parameter_dict(parameters))
    rng = backend.random_generator(normalized["seed"])

    
    _, psi_hat = shaped_random_real_field_perp_prl(
            grid,
            backend,
            fft,
            n_min_perp_force=normalized["n_min_perp"],
            n_min_prl_force=normalized["n_min_prl"],
            n_max_perp_force=normalized["n_max_perp"],
            n_max_prl_force=normalized["n_max_prl"],
            alpha_force=0.5 * normalized["alpha"],
            alpha_prl_force=0.5 * normalized["alpha_prl"],
            rng=rng,
        )
    state["psi"][...] = psi_hat

    state["omega"][...] = lap_perp(psi_hat, grid)

    if dealias_mask is not None:
        state.apply_mask(dealias_mask)

    # Drop the k_par = 0 plane: it is a marginal (zero-frequency, defective)
    # subspace where psi is frozen and du_par is driven secularly by
    # -vA*K_b0*dy(psi), so seeding it makes total_energy grow ~ t^2. Applied
    # before energy rescaling so init_energy is normalized over surviving modes.
    if normalized["exclude_kpar0"]:
        kpar_nonzero = grid.kz != 0
        for field_name in state.field_names:
            state[field_name][...] *= kpar_nonzero

    return _rescale_state_to_total_energy(
        state,
        target_energy=normalized["init_energy"],
        grid=grid,
        backend=backend,
        params=params,
    )
