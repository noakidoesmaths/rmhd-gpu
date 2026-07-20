"""Simple perpendicular shell spectra."""

from __future__ import annotations

from typing import Any

import numpy as np

PERPENDICULAR_SPECTRUM_KEYS = ("u_perp", "b_perp", "upar", "dbpar", "s")
_SPECTRUM_GRID_CACHE: dict[tuple[int, int, int, float, float, float], tuple[np.ndarray, np.ndarray]] = {}


def _rfft_weights(grid: Any) -> np.ndarray:
    weights = np.ones(grid.fourier_shape, dtype=np.float64)
    if grid.Nz % 2 == 0:
        weights[..., 1:-1] = 2.0
    else:
        weights[..., 1:] = 2.0
    return weights


def _cached_spectrum_grid_arrays(grid: Any, backend: Any) -> tuple[np.ndarray, np.ndarray]:
    cache_key = (grid.Nx, grid.Ny, grid.Nz, float(grid.Lx), float(grid.Ly), float(grid.Lz))
    cached = _SPECTRUM_GRID_CACHE.get(cache_key)
    if cached is not None:
        return cached

    kperp_np = np.sqrt(backend.to_numpy(grid.kperp2))
    weights = _rfft_weights(grid)
    _SPECTRUM_GRID_CACHE[cache_key] = (kperp_np, weights)
    return kperp_np, weights


def perpendicular_shell_spectrum(
    density_hat: Any,
    grid: Any,
    backend: Any,
    bin_width: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Bin a Fourier-space modal density into perpendicular shells.

    The shell spectrum is the sum of the supplied modal density over shells in
    `k_perp = sqrt(kx^2 + ky^2)`, with the omitted negative-`kz` half of the
    real FFT accounted for by the standard one-sided rFFT weights.

    The returned normalization is a volume average: shell values sum to the
    total weighted modal density divided by `N^2`, where
    `N = Nx * Ny * Nz`.
    """

    density_np = backend.to_numpy(density_hat)
    kperp_np, weights = _cached_spectrum_grid_arrays(grid, backend)
    normalization = float(np.prod(grid.real_shape) ** 2)
    modal_density = weights * density_np / normalization

    if bin_width is None:
        bin_width = min(2.0 * np.pi / grid.Lx, 2.0 * np.pi / grid.Ly)

    max_kperp = float(kperp_np.max())
    edges = np.arange(0.0, max_kperp + 1.5 * bin_width, bin_width)
    spectrum = np.zeros(edges.size - 1, dtype=np.float64)

    shell_index = np.floor(kperp_np.ravel() / bin_width).astype(int)
    flat_density = modal_density.ravel()
    valid = (shell_index >= 0) & (shell_index < spectrum.size)
    np.add.at(spectrum, shell_index[valid], flat_density[valid])

    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, spectrum


def parallel_shell_spectrum(
    density_hat: Any,
    grid: Any,
    backend: Any,
    bin_width: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Bin a Fourier-space modal density into parallel (kz) shells.

    The rFFT axis is z, so every last-axis index already holds a single
    `|kz|` value: the shell spectrum is just the modal density summed over
    `(kx, ky)` per kz index, with the omitted negative-`kz` half of the real
    FFT accounted for by the standard one-sided rFFT weights. `bin_width` is
    accepted for signature symmetry with `perpendicular_shell_spectrum` and
    ignored.

    The returned normalization matches `perpendicular_shell_spectrum`: shell
    values sum to the total weighted modal density divided by `N^2`, where
    `N = Nx * Ny * Nz`.
    """

    density_np = backend.to_numpy(density_hat)
    _, weights = _cached_spectrum_grid_arrays(grid, backend)
    normalization = float(np.prod(grid.real_shape) ** 2)
    modal_density = weights * density_np / normalization

    kprl = backend.to_numpy(grid.kz).ravel().astype(np.float64)
    spectrum = modal_density.sum(axis=(0, 1))
    return kprl, spectrum


def perpendicular_energy_spectrum_from_state(
    state: Any,
    grid: Any,
    backend: Any | None = None,
    bin_width: float | None = None,
    equation_module: Any | None = None,
    params: Any | None = None,
) -> dict[str, np.ndarray]:
    """Return perpendicular shell spectra for the selected equation set."""

    backend_obj = state.backend if backend is None else backend
    if equation_module is not None and hasattr(equation_module, "perpendicular_energy_spectra"):
        return equation_module.perpendicular_energy_spectra(
            state,
            grid,
            backend_obj,
            bin_width=bin_width,
            params=params,
        )

    xp = backend_obj.xp
    spectra: dict[str, np.ndarray] = {}
    for index, field_name in enumerate(state.field_names):
        kperp, spectrum = perpendicular_shell_spectrum(
            0.5 * xp.abs(state[field_name]) ** 2,
            grid,
            backend_obj,
            bin_width=bin_width,
        )
        if index == 0:
            spectra["kperp"] = kperp
        spectra[field_name] = spectrum
    return spectra


def parallel_energy_spectrum_from_state(
    state: Any,
    grid: Any,
    backend: Any | None = None,
    bin_width: float | None = None,
    equation_module: Any | None = None,
    params: Any | None = None,
) -> dict[str, np.ndarray]:
    """Return parallel (kz) shell spectra for the selected equation set."""

    backend_obj = state.backend if backend is None else backend
    if equation_module is not None and hasattr(equation_module, "parallel_energy_spectra"):
        return equation_module.parallel_energy_spectra(
            state,
            grid,
            backend_obj,
            bin_width=bin_width,
            params=params,
        )

    xp = backend_obj.xp
    spectra: dict[str, np.ndarray] = {}
    for index, field_name in enumerate(state.field_names):
        kprl, spectrum = parallel_shell_spectrum(
            0.5 * xp.abs(state[field_name]) ** 2,
            grid,
            backend_obj,
            bin_width=bin_width,
        )
        if index == 0:
            spectra["kprl"] = kprl
        spectra[field_name] = spectrum
    return spectra


def compute_placeholder_spectra(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Return an empty placeholder spectral diagnostics payload."""

    return {}
