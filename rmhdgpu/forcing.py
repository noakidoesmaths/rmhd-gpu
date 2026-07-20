"""Stochastic white-in-time forcing helpers.

The forcing implemented here is additive and refreshed every timestep. For one
field `q_i`, over a step `dt`, the kick is

`delta q_i = sigma_i * sqrt(dt) * xi_i`

where `sigma_i` is the configured RMS forcing strength in field-units per
`sqrt(time)`, and `xi_i` is a freshly generated band-limited, unit-RMS real
Gaussian field.

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

from rmhdgpu.state import State


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
    outside the selected forcing band.
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
) -> State:
    """Return the additive stochastic forcing increment for one timestep.

    Each field is forced independently with a fresh random realization. The
    increment scales as `sqrt(dt)` so the configured amplitudes are interpreted
    as RMS forcing strengths in field-units per `sqrt(time)`.
    """

    kick = state.zeros_like() if out is None else out
    kick.fill_zero()
    if not getattr(config, "use_forcing", False):
        return kick

    scale_dt = float(np.sqrt(dt))
    metadata = _forcing_metadata(
        grid,
        backend,
        n_min_force=float(getattr(config, "n_min_force")),
        n_max_force=float(getattr(config, "n_max_force")),
        alpha_force=float(getattr(config, "alpha_force")),
        workspace=workspace,
    )
    scratch_real = None if workspace is None else workspace.real.get("r0")
    scratch_hat = None if workspace is None else workspace.complex.get("c1")

    for field_name in kick.field_names:
        sigma = float(getattr(config, "force_amplitudes")[field_name])
        if sigma == 0.0:
            continue

        _, xi_hat = shaped_random_real_field(
            grid,
            backend,
            fft,
            n_min_force=float(getattr(config, "n_min_force")),
            n_max_force=float(getattr(config, "n_max_force")),
            alpha_force=float(getattr(config, "alpha_force")),
            rng=rng,
            band_mask=metadata["band_mask"],
            shaping=metadata["shaping"],
            out_real=scratch_real,
            out_hat=scratch_hat,
            workspace=workspace,
        )
        kick[field_name][...] = sigma * scale_dt * xi_hat

    return kick


def apply_forcing_kick(state: State, forcing_kick: State, *, inplace: bool = False) -> State:
    """Return `state + forcing_kick`, optionally mutating `state` in place."""

    result = state if inplace else state.copy()
    for field_name in result.field_names:
        result[field_name][...] += forcing_kick[field_name]
    return result
