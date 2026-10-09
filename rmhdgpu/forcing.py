"""Stochastic white-in-time forcing helpers.

The forcing is additive and refreshed every timestep. Inputs are mean energy
injection rates, not raw field amplitudes. A fixed normalization derived from
the ensemble variance of the filtered Gaussian is used throughout a run. An
individual realization is never renormalized, so the kick energy fluctuates
naturally while its expectation is ``epsilon * dt``. Consequently kicks have
Wiener scaling, ``delta q ~ sqrt(dt)``, while ``epsilon`` has the unambiguous
units energy per unit time.

Two forcing modes are supported:

``field``
    Independently force evolved fields using ``field_energy_injection_rates``.
    The equation module's ``total_energy`` definition supplies the correct
    normalization for potentials, vorticity, and weighted compressive fields.

``elsasser``
    Independently force ``zeta_plus = phi - psi`` and
    ``zeta_minus = phi + psi`` at rates ``epsilon_plus`` and
    ``epsilon_minus``. Their energies are
    ``E_plus/minus = 0.5 <|grad_perp zeta_plus/minus|^2>``. The velocity
    potential may be stored directly as ``phi`` or indirectly as
    ``omega = lap_perp(phi)``.

The forcing band is defined in integer Fourier mode-number magnitude

`n = sqrt(nx^2 + ny^2 + nz^2)`

rather than physical `k`.

When the CuPy backend is active, random fields are generated with backend-side
RNGs when available so the forcing path stays on device. A fixed seed is
expected to be reproducible within a backend, but NumPy and CuPy sequences are
not expected to match bitwise.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rmhdgpu.fourier_diagnostics import modal_average
from rmhdgpu.state import State
from rmhdgpu.forcing_fields import add_potential_increment


def mode_number_magnitude(grid: Any, backend: Any) -> Any:
    """Return the integer mode-number magnitude on the stored `rfftn` grid."""

    xp = backend.xp
    nx = xp.fft.fftfreq(grid.Nx) * grid.Nx
    ny = xp.fft.fftfreq(grid.Ny) * grid.Ny
    nz = xp.fft.rfftfreq(grid.Nz) * grid.Nz
    return xp.sqrt(
        nx.reshape(grid.Nx, 1, 1) ** 2
        + ny.reshape(1, grid.Ny, 1) ** 2
        + nz.reshape(1, 1, grid.Nz // 2 + 1) ** 2
    )


def forcing_shell_mask(
    grid: Any,
    backend: Any,
    n_min_force: float,
    n_max_force: float,
) -> Any:
    """Return the forcing-band mask on the Fourier grid."""

    n_mag = mode_number_magnitude(grid, backend)
    return (n_mag >= float(n_min_force)) & (n_mag <= float(n_max_force))


def _forcing_metadata(
    grid: Any,
    backend: Any,
    *,
    n_min_force: float,
    n_max_force: float,
    alpha_force: float,
    workspace: Any | None = None,
) -> dict[str, Any]:
    cache_key = (
        "forcing_metadata",
        backend.backend_name,
        grid.real_shape,
        float(n_min_force),
        float(n_max_force),
        float(alpha_force),
    )
    if workspace is not None and cache_key in workspace.cache:
        return workspace.cache[cache_key]

    xp = backend.xp
    n_mag = mode_number_magnitude(grid, backend)
    band_mask = (n_mag >= float(n_min_force)) & (n_mag <= float(n_max_force))
    n_safe = xp.where(n_mag > 0.0, n_mag, 1.0)
    shaping = xp.where(band_mask, n_safe ** (-float(alpha_force)), 0.0).astype(
        grid.real_dtype,
        copy=False,
    )
    metadata = {
        "band_mask": band_mask,
        "shaping": shaping,
    }
    if workspace is not None:
        workspace.cache[cache_key] = metadata
    return metadata

def _forcing_metadata_perp_prl(
        grid: Any,
        backend: Any,
        *,
        n_min_perp_force: float,
        n_min_prl_force: float,
        n_max_perp_force: float,
        n_max_prl_force: float,
        alpha_force: float,
        alpha_prl_force: float = 0.0,
        workspace: Any | None = None,
    ) -> dict[str, Any]:
    cache_key = (
        "forcing_metadata_perp_prl",
        backend.backend_name,
        grid.real_shape,
        float(n_min_perp_force),
        float(n_min_prl_force),
        float(n_max_perp_force),
        float(n_max_prl_force),
        float(alpha_force),
        float(alpha_prl_force),
    )
    if workspace is not None and cache_key in workspace.cache:
        return workspace.cache[cache_key]

    xp = backend.xp
    nx = xp.fft.fftfreq(grid.Nx) * grid.Nx
    ny = xp.fft.fftfreq(grid.Ny) * grid.Ny
    nz = xp.fft.rfftfreq(grid.Nz) * grid.Nz
    n_perp = xp.sqrt(
        nx.reshape(grid.Nx, 1, 1) ** 2 + ny.reshape(1, grid.Ny, 1) ** 2
    )
    n_prl = xp.abs(nz).reshape(1, 1, grid.Nz // 2 + 1)

    band_mask = (
        (n_perp >= float(n_min_perp_force)) & (n_perp <= float(n_max_perp_force))
        & (n_prl >= float(n_min_prl_force)) & (n_prl <= float(n_max_prl_force))
    )
    n_safe = xp.where(n_perp > 0.0, n_perp, 1.0)
    n_prl_safe = xp.where(n_prl > 0.0, n_prl, 1.0)
    shaping = xp.where(
        band_mask,
        n_safe ** (-float(alpha_force)) * n_prl_safe ** (-float(alpha_prl_force)),
        0.0,
    ).astype(
        grid.real_dtype,
        copy=False,
    )
    metadata = {
        "band_mask": band_mask,
        "shaping": shaping,
    }
    if workspace is not None:
        workspace.cache[cache_key] = metadata
    return metadata


def _standard_normal_field(
    rng: Any,
    shape: tuple[int, ...],
    dtype: Any,
    backend: Any,
) -> Any:
    if rng is None:
        rng = backend.random_generator()

    try:
        values = rng.standard_normal(shape, dtype=dtype)
    except TypeError:
        values = rng.standard_normal(shape)

    return backend.asarray(values, dtype=dtype)


def shaped_random_real_field(
    grid: Any,
    backend: Any,
    fft: Any,
    *,
    n_min_force: float,
    n_max_force: float,
    alpha_force: float,
    rng: Any,
    band_mask: Any | None = None,
    shaping: Any | None = None,
    out_real: Any | None = None,
    out_hat: Any | None = None,
    workspace: Any | None = None,
) -> tuple[Any, Any]:
    """Return a unit-RMS real field and its Fourier transform.

    Construction:

    1. draw a Gaussian random real field in real space
    2. transform to `rfftn` storage
    3. apply the forcing shell mask
    4. apply amplitude shaping proportional to `n^{-alpha_force}`
    5. transform back to real space and normalize to unit RMS
    6. transform back to Fourier space

    The final Fourier field is masked again to remove roundoff-level leakage
    outside the selected forcing band. This utility retains its historical
    per-realization unit-RMS contract for initial-condition and diagnostic use;
    stochastic forcing kicks use :func:`_shaped_fourier_noise` instead and are
    never normalized realization by realization.
    """

    metadata = None
    if band_mask is None or shaping is None:
        metadata = _forcing_metadata(
            grid,
            backend,
            n_min_force=n_min_force,
            n_max_force=n_max_force,
            alpha_force=alpha_force,
            workspace=workspace,
        )
        if band_mask is None:
            band_mask = metadata["band_mask"]
        if shaping is None:
            shaping = metadata["shaping"]

    real_noise = _standard_normal_field(rng, grid.real_shape, grid.real_dtype, backend)

    shaped_hat = fft.r2c(real_noise, out=out_hat)
    shaped_hat[...] *= shaping
    shaped_real = fft.c2r(shaped_hat, out=out_real)

    rms = backend.scalar_to_float(backend.xp.sqrt(backend.xp.mean(shaped_real**2)))
    if not np.isfinite(rms) or rms <= 0.0:
        raise RuntimeError(
            "Stochastic forcing produced a zero or non-finite filtered field. "
            "Check the forcing band and spectral shaping parameters."
        )

    shaped_real[...] /= rms
    shaped_hat = fft.r2c(shaped_real, out=shaped_hat)
    shaped_hat[...] *= band_mask
    return shaped_real, shaped_hat

def shaped_random_real_field_perp_prl(
    grid: Any,
    backend: Any,
    fft: Any,
    *,
    n_min_perp_force: float,
    n_min_prl_force: float,
    n_max_perp_force: float,
    n_max_prl_force: float,
    alpha_force: float,
    alpha_prl_force: float = 0.0,
    rng: Any,
    band_mask: Any | None = None,
    shaping: Any | None = None,
    out_real: Any | None = None,
    out_hat: Any | None = None,
    workspace: Any | None = None,
) -> tuple[Any, Any]:
    """Return a unit-RMS real field and its Fourier transform.

    Same idea as before but explicitly calculates for n_min_parallel, n_min_perp,
    n_max_parallel, n_max_perp.
    Creates a mask for
    n_min_prl <= n_z <= n_max_prl
    and
    n_min_prl <= sqrt(nx^2 + ny^2) <= n_max_prl

    Amplitudes inside the band are shaped by
    `n_perp^(-alpha_force) * n_prl^(-alpha_prl_force)`, so the perpendicular
    and parallel spectral slopes can be set independently.
    """

    metadata = None
    if band_mask is None or shaping is None:
        metadata = _forcing_metadata_perp_prl(
            grid,
            backend,
            n_min_perp_force=n_min_perp_force,
            n_min_prl_force=n_min_prl_force,
            n_max_perp_force=n_max_perp_force,
            n_max_prl_force=n_max_prl_force,
            alpha_force=alpha_force,
            alpha_prl_force=alpha_prl_force,
            workspace=workspace,
        )
        if band_mask is None:
            band_mask = metadata["band_mask"]
        if shaping is None:
            shaping = metadata["shaping"]

    real_noise = _standard_normal_field(rng, grid.real_shape, grid.real_dtype, backend)

    shaped_hat = fft.r2c(real_noise, out=out_hat)
    shaped_hat[...] *= shaping
    shaped_real = fft.c2r(shaped_hat, out=out_real)

    rms = backend.scalar_to_float(backend.xp.sqrt(backend.xp.mean(shaped_real**2)))
    if not np.isfinite(rms) or rms <= 0.0:
        raise RuntimeError(
            "Stochastic forcing produced a zero or non-finite filtered field. "
            "Check the forcing band and spectral shaping parameters."
        )

    shaped_real[...] /= rms
    shaped_hat = fft.r2c(shaped_real, out=shaped_hat)
    shaped_hat[...] *= band_mask
    return shaped_real, shaped_hat
    

def _shaped_fourier_noise(
    grid: Any,
    backend: Any,
    fft: Any,
    *,
    rng: Any,
    band_mask: Any,
    shaping: Any,
    out_hat: Any | None = None,
) -> Any:
    """Draw one raw filtered Gaussian realization in Fourier space.

    Real-space samples have independent unit-variance Gaussian distributions.
    The fixed Fourier filter is applied without measuring or normalizing the
    realization. Starting in real space guarantees the required R2C reality
    constraints.
    """

    real_noise = _standard_normal_field(rng, grid.real_shape, grid.real_dtype, backend)
    noise_hat = fft.r2c(real_noise, out=out_hat)
    noise_hat[...] *= shaping
    noise_hat[...] *= band_mask
    return noise_hat


def _equation_module_for_forcing(config: Any, equation_module: Any | None) -> Any:
    if equation_module is not None:
        return equation_module

    from rmhdgpu.equations import get_equation_module

    equation_set = getattr(config, "equation_set", None)
    if equation_set is None:
        raise ValueError(
            "Field-energy forcing requires either equation_module or "
            "config.equation_set so the kick can use the correct energy normalization."
        )
    return get_equation_module(str(equation_set))


def _expected_field_noise_energy(
    field_name: str,
    state: State,
    grid: Any,
    backend: Any,
    config: Any,
    equation_module: Any,
    shaping: Any,
    workspace: Any | None,
) -> float:
    """Return ensemble-mean energy of one unscaled filtered Gaussian field.

    NumPy/CuPy forward FFTs are unnormalized. For ``N`` independent unit-
    variance real samples, every stored Fourier coefficient has
    ``E[|q_hat(k)|^2] = N`` before filtering. Supplying deterministic modal
    amplitudes ``sqrt(N) * shaping`` to the quadratic equation energy therefore
    evaluates the exact ensemble expectation without sampling a realization.
    """

    if not hasattr(equation_module, "total_energy"):
        raise ValueError(
            f"Equation module {equation_module.__name__!r} must provide total_energy(...) "
            "to use field-energy forcing."
        )

    cache_key = (
        "forcing_expected_field_energy",
        id(config),
        getattr(equation_module, "__name__", type(equation_module).__name__),
        field_name,
        id(shaping),
    )
    if workspace is not None and cache_key in workspace.cache:
        return float(workspace.cache[cache_key])

    normalization_state = (
        state.zeros_like()
        if workspace is None
        else workspace.get_state_buffer("forcing_normalization", state.field_names)
    )
    normalization_state.fill_zero()
    normalization_state[field_name][...] = np.sqrt(np.prod(grid.real_shape)) * shaping
    energy = float(equation_module.total_energy(normalization_state, grid, backend, config))
    if not np.isfinite(energy) or energy <= 0.0:
        raise RuntimeError(
            f"Cannot normalize forcing for field {field_name!r}: the expected "
            f"filtered-noise energy is non-positive or non-finite ({energy!r}). "
            "Check the equation-set energy and forcing band."
        )
    if workspace is not None:
        workspace.cache[cache_key] = energy
    return energy


def _expected_elsasser_noise_energy(shaping: Any, grid: Any, backend: Any) -> float:
    """Return expected ``0.5 <|grad_perp xi|^2>`` for filtered Gaussian noise."""

    sample_count = float(np.prod(grid.real_shape))
    density_hat = 0.5 * sample_count * grid.kperp2 * shaping**2
    energy = modal_average(density_hat, grid, backend)
    if not np.isfinite(energy) or energy <= 0.0:
        raise RuntimeError(
            "Cannot normalize Elsasser forcing: the selected forcing band has "
            "zero or non-finite perpendicular-gradient energy. Include at least "
            "one mode with k_perp != 0."
        )
    return energy


def _add_elsasser_forcing(
    kick: State,
    grid: Any,
    fft: Any,
    backend: Any,
    config: Any,
    rng: Any,
    dt: float,
    metadata: dict[str, Any],
    workspace: Any | None,
) -> None:
    """Add independent, energy-normalized ``zeta_plus`` and ``zeta_minus`` kicks."""

    if "psi" not in kick.field_names:
        raise ValueError("Elsasser forcing requires an evolved 'psi' field.")

    # Pure k_perp=0 potentials carry no RMHD Elsasser energy and are excluded
    # before normalization rather than left as unconstrained gauge components.
    alfvenic_mask = metadata["band_mask"] & (~grid.mask_kperp0)
    alfvenic_shaping = metadata["shaping"] * alfvenic_mask
    scratch_hat = None if workspace is None else workspace.complex.get("c1")
    cache_key = (
        "forcing_expected_elsasser_energy",
        id(config),
        id(metadata["shaping"]),
    )
    if workspace is not None and cache_key in workspace.cache:
        expected_energy = float(workspace.cache[cache_key])
    else:
        expected_energy = _expected_elsasser_noise_energy(
            alfvenic_shaping,
            grid,
            backend,
        )
        if workspace is not None:
            workspace.cache[cache_key] = expected_energy

    for branch, epsilon in (
        ("plus", float(getattr(config, "epsilon_plus"))),
        ("minus", float(getattr(config, "epsilon_minus"))),
    ):
        if epsilon == 0.0:
            continue
        xi_hat = _shaped_fourier_noise(
            grid,
            backend,
            fft,
            rng=rng,
            band_mask=alfvenic_mask,
            shaping=alfvenic_shaping,
            out_hat=scratch_hat,
        )
        scale = float(np.sqrt(epsilon * dt / expected_energy))

        # zeta+ = phi - psi and zeta- = phi + psi, hence
        add_potential_increment(kick, grid, xi_hat, branch=branch,
                                velocity="phi" if "phi" in kick.field_names else "omega", scale=scale)



def generate_forcing_kick(
    state: State,
    grid: Any,
    fft: Any,
    backend: Any,
    config: Any,
    rng: Any,
    dt: float,
    workspace: Any | None = None,
    out: State | None = None,
    equation_module: Any | None = None,
) -> State:
    """Return the additive stochastic forcing increment for one timestep.

    Configured values are mean energy injection rates. In ``field`` mode, the
    fixed normalization makes the expected self-energy of a kick in field
    ``i`` equal to ``epsilon_i * dt`` under the selected equation set's total
    energy. In ``elsasser`` mode, the same expectation applies separately to
    ``E_plus`` and ``E_minus``; non-Alfvénic fields can still be forced through
    per-field rates.

    Neither the kick nor the pre-existing state is measured to adjust the
    amplitude. Individual kick energies and cross terms therefore fluctuate,
    while their ensemble means give the configured Itô injection rates.
    """

    kick = state.zeros_like() if out is None else out
    kick.fill_zero()
    if not getattr(config, "use_forcing", False):
        return kick
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"Forcing timestep dt must be finite and positive; got {dt!r}.")

    metadata = _forcing_metadata(
        grid,
        backend,
        n_min_force=float(getattr(config, "n_min_force")),
        n_max_force=float(getattr(config, "n_max_force")),
        alpha_force=float(getattr(config, "alpha_force")),
        workspace=workspace,
    )
    scratch_hat = None if workspace is None else workspace.complex.get("c1")

    forcing_mode = str(getattr(config, "forcing_mode", "field"))
    if forcing_mode == "elsasser":
        _add_elsasser_forcing(
            kick,
            grid,
            fft,
            backend,
            config,
            rng,
            dt,
            metadata,
            workspace,
        )
    elif forcing_mode != "field":
        raise ValueError(f"Unknown forcing_mode {forcing_mode!r}.")

    injection_rates = getattr(config, "field_energy_injection_rates")
    module = None

    for field_name in kick.field_names:
        epsilon = float(injection_rates[field_name])
        if epsilon == 0.0:
            continue

        xi_hat = _shaped_fourier_noise(
            grid,
            backend,
            fft,
            rng=rng,
            band_mask=metadata["band_mask"],
            shaping=metadata["shaping"],
            out_hat=scratch_hat,
        )
        if module is None:
            module = _equation_module_for_forcing(config, equation_module)
        expected_energy = _expected_field_noise_energy(
            field_name,
            state,
            grid,
            backend,
            config,
            module,
            metadata["shaping"],
            workspace,
        )
        kick[field_name][...] += np.sqrt(epsilon * dt / expected_energy) * xi_hat

    return kick


def apply_forcing_kick(state: State, forcing_kick: State, *, inplace: bool = False) -> State:
    """Return `state + forcing_kick`, optionally mutating `state` in place."""

    result = state if inplace else state.copy()
    for field_name in result.field_names:
        result[field_name][...] += forcing_kick[field_name]
    return result
