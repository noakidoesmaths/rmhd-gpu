"""Qualitative sanity check for forced turbulence.

This is a lightweight self-contained sanity script, not the recommended
workflow for normal runs. For case-style usage, prefer
`python -m rmhdgpu.run examples/forced_turbulence.input`, then the generic
`vis/plot_*` scripts on the saved outputs.

Run with:

`python -m rmhdgpu.examples.sanity_forced_turbulence`

or, for a quick single-GPU `256^3` check,

`python -m rmhdgpu.examples.sanity_forced_turbulence --gpu-256`

The script starts from zero initial conditions, applies white-in-time
band-limited forcing, and saves a single multi-panel summary figure with the
energy history and perpendicular spectra.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rmhdgpu.backend import build_backend
from rmhdgpu.config import Config
from rmhdgpu.diagnostics.scalar import compute_energy_diagnostics
from rmhdgpu.diagnostics.spectra import PERPENDICULAR_SPECTRUM_KEYS, perpendicular_energy_spectrum_from_state
from rmhdgpu.equations import s09
from rmhdgpu.examples.frame_output import (
    add_frame_arguments,
    build_frame_times,
    capture_xy_signed_fields,
    resolve_snapshot_z_index,
    write_signed_xy_frames,
)
from rmhdgpu.fft import FFTManager
from rmhdgpu.grid import build_grid
from rmhdgpu.initconds import build_initial_state
from rmhdgpu.masks import build_dealias_mask
from rmhdgpu.steppers import evolve_until
from rmhdgpu.workspace import Workspace


DEFAULT_BACKEND = "scipy_cpu"
DEFAULT_GRID_SIZE = 96
DEFAULT_T_FINAL = 4.0
DEFAULT_FFT_WORKERS = 8
GPU_256_T_FINAL = 1.5


def estimate_hyperdiffusion_coefficient(k_d: float, k0: float, u_rms: float, order: int) -> float:
    """Estimate a perpendicular hyperdiffusion coefficient for the sanity run."""

    return k_d * u_rms * (k_d / k0) ** (-1.0 / 3.0) / (k_d ** (2 * order))


def dealiased_max_kperp(grid: object, backend: object, mask: object) -> float:
    """Return the maximum retained perpendicular wavenumber after dealiasing."""

    kperp = np.sqrt(backend.to_numpy(grid.kperp2))
    retained = backend.to_numpy(mask).astype(bool)
    return float(np.max(kperp[retained]))


def _plot_summary(
    times: np.ndarray,
    energy_history: list[dict[str, float]],
    sample_times: np.ndarray,
    spectra_by_time: list[dict[str, np.ndarray]],
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    axes_flat = list(axes.flat)
    energy_axis = axes_flat[0]
    spectrum_axes = axes_flat[1:]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(sample_times)))

    energy_labels = [
        ("alfvenic_energy", "Alfvenic"),
        ("upar_energy", r"$u_\parallel$"),
        ("dbpar_energy", r"$\delta B_\parallel$"),
        ("entropy_variance", "Entropy"),
        ("total_energy_proxy", "Total"),
    ]
    energy_ymax = 0.0
    for key, label in energy_labels:
        values = np.asarray([entry[key] for entry in energy_history], dtype=np.float64)
        positive = values[values > 0.0]
        if positive.size:
            energy_ymax = max(energy_ymax, float(positive.max()))
        energy_axis.semilogy(times, np.where(values > 0.0, values, np.nan), lw=2, label=label)

    if energy_ymax > 0.0:
        energy_axis.set_ylim(energy_ymax * 1.0e-6, energy_ymax)

    energy_axis.set_xlabel("t")
    energy_axis.set_ylabel("Quadratic measure")
    energy_axis.set_title("Forced-turbulence energy history")
    energy_axis.grid(True, alpha=0.3)
    energy_axis.legend(fontsize=8)

    spectrum_titles = {
        "u_perp": r"$E_{u_\perp}(k_\perp)$",
        "b_perp": r"$E_{b_\perp}(k_\perp)$",
        "upar": r"$E_{u_\parallel}(k_\perp)$",
        "dbpar": r"$E_{\delta B_\parallel}(k_\perp)$",
        "s": r"$E_s(k_\perp)$",
    }
    for axis, key in zip(spectrum_axes, PERPENDICULAR_SPECTRUM_KEYS, strict=True):
        panel_ymax = 0.0
        for color, time, spectra in zip(colors, sample_times, spectra_by_time, strict=True):
            valid = (spectra["kperp"] > 0.0) & (spectra[key] > 0.0)
            x = spectra["kperp"][valid]
            y = spectra[key][valid]
            if y.size:
                panel_ymax = max(panel_ymax, float(np.max(y)))
            axis.loglog(x, y, color=color, lw=2, label=f"t={time:.2f}")

        k_ref = spectra_by_time[0]["kperp"]
        ref_mask = k_ref > 1.0
        if np.any(ref_mask):
            k_ref_plot = k_ref[ref_mask]
            y_ref = 1.0e-3 * (k_ref_plot / k_ref_plot[0]) ** (-5.0 / 3.0)
            axis.loglog(k_ref_plot, y_ref, "k--", alpha=0.5, label=r"$k^{-5/3}$")

        if panel_ymax > 0.0:
            axis.set_ylim(panel_ymax * 1.0e-6, panel_ymax)

        axis.set_title(spectrum_titles[key])
        axis.set_xlabel(r"$k_\perp$")
        axis.set_ylabel("E")
        axis.grid(True, alpha=0.25)

    spectrum_axes[0].legend(fontsize=8)

    figure_path = output_dir / "sanity_forced_turbulence.png"
    fig.savefig(figure_path, dpi=160)
    print(f"Saved {figure_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="sanity_plots", help="Directory where figures are written.")
    parser.add_argument("--backend", choices=["numpy", "scipy_cpu", "cupy"], default=DEFAULT_BACKEND)
    parser.add_argument("--fft-workers", type=int, default=DEFAULT_FFT_WORKERS)
    parser.add_argument("--n", type=int, default=DEFAULT_GRID_SIZE, help="Grid resolution in each direction.")
    parser.add_argument("--t-final", type=float, default=DEFAULT_T_FINAL, help="Final time.")
    parser.add_argument("--forcing-seed", type=int, default=1234, help="Seed for the white-noise forcing.")
    parser.add_argument(
        "--gpu-256",
        action="store_true",
        help="Shortcut for backend='cupy', n=256, and a shorter quick-check runtime.",
    )
    add_frame_arguments(parser)
    return parser


def resolve_run_parameters(args: argparse.Namespace) -> dict[str, object]:
    """Resolve backend/grid defaults after CLI presets are applied."""

    backend_name = args.backend
    n = args.n
    fft_workers = args.fft_workers
    t_final = args.t_final
    if args.gpu_256:
        backend_name = "cupy"
        n = 256
        fft_workers = None
        if args.t_final == DEFAULT_T_FINAL:
            t_final = GPU_256_T_FINAL
    return {
        "backend_name": backend_name,
        "n": n,
        "fft_workers": fft_workers,
        "t_final": t_final,
    }


def main() -> None:
    args = build_parser().parse_args()
    run_params = resolve_run_parameters(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = Config(
        Nx=run_params["n"],
        Ny=run_params["n"],
        Nz=run_params["n"],
        backend=run_params["backend_name"],
        fft_workers=run_params["fft_workers"],
        cfl_number=0.35,
        dt_max=1.5e-2,
        tmax=run_params["t_final"],
        use_variable_dt=True,
        use_forcing=True,
        forcing_mode="elsasser",
        n_min_force=1.0,
        n_max_force=3.0,
        alpha_force=0.5,
        # This balanced rate gives u_perp,rms approximately one in the 32^3
        # auto-dissipation calibration; achieved RMS remains a run diagnostic.
        epsilon_plus=0.35,
        epsilon_minus=0.35,
        field_energy_injection_rates={
            "upar": 5.0e-4,
            "dbpar": 1.0e-3,
            "s": 1.0e-4,
        },
        forcing_seed=args.forcing_seed,
    )

    backend = build_backend(config)
    grid = build_grid(config, backend)
    fft = FFTManager(grid, backend)
    workspace = Workspace(grid, backend)
    mask = build_dealias_mask(grid, backend)
    z_index = resolve_snapshot_z_index(grid, args.snapshot_z_index)

    print(
        "sanity_forced_turbulence setup",
        {
            "backend": config.backend,
            "grid": grid.real_shape,
            "fft_workers": config.fft_workers,
            "t_final": config.tmax,
            "forcing_seed": config.forcing_seed,
            "gpu_256": args.gpu_256,
            "save_frames": args.save_frames,
            "snapshot_z_index": z_index,
        },
    )

    k0 = 1.0
    kperp_max_dealiased = dealiased_max_kperp(grid, backend, mask)
    k_d = 0.5 * kperp_max_dealiased
    u_rms_guess = 1.0
    nu_perp = estimate_hyperdiffusion_coefficient(k_d=k_d, k0=k0, u_rms=u_rms_guess, order=3)
    nu_par = 0.1 * nu_perp
    for name in config.field_names:
        config.dissipation[name]["nu_perp"] = nu_perp
        config.dissipation[name]["nu_par"] = nu_par
        config.dissipation[name]["n_perp"] = 3
        config.dissipation[name]["n_par"] = 1

    print(
        "forced-turbulence setup",
        {
            "forcing_band": [config.n_min_force, config.n_max_force],
            "alpha_force": config.alpha_force,
            "forcing_mode": config.forcing_mode,
            "epsilon_plus": config.epsilon_plus,
            "epsilon_minus": config.epsilon_minus,
            "field_energy_injection_rates": dict(config.field_energy_injection_rates),
            "k0": k0,
            "kperp_max_dealiased": kperp_max_dealiased,
            "k_d": k_d,
            "nu_perp": nu_perp,
            "nu_par": nu_par,
        },
    )

    linear_ops = s09.build_dissipation_operators(grid, config)
    rhs_kwargs = {
        "grid": grid,
        "fft": fft,
        "workspace": workspace,
        "params": config,
        "dealias_mask": mask,
    }

    current = build_initial_state(
        "zero",
        grid=grid,
        backend=backend,
        fft=fft,
        dealias_mask=mask,
        field_names=config.field_names,
        params=config,
    )
    forcing_rng = backend.random_generator(config.forcing_seed)
    sample_times = np.linspace(0.0, config.tmax, 8)
    frame_times = build_frame_times(config.tmax, args.frame_count) if args.save_frames else np.empty(0, dtype=np.float64)
    energy_history: list[dict[str, float]] = []
    spectra_history: list[dict[str, np.ndarray]] = []
    frame_records: list[dict[str, object]] = []

    previous_time = 0.0
    event_times = np.unique(np.concatenate([sample_times, frame_times])) if frame_times.size else sample_times
    sample_index = 0
    frame_index = 0
    for event_time in event_times:
        segment = float(event_time - previous_time)
        if segment > 0.0:
            current, _ = evolve_until(
                current,
                segment,
                s09.ideal_rhs,
                linear_ops,
                rhs_kwargs=rhs_kwargs,
                params=config,
                forcing_rng=forcing_rng,
            )
        if sample_index < len(sample_times) and np.isclose(event_time, sample_times[sample_index]):
            energy_history.append(compute_energy_diagnostics(current, grid, fft, backend, config, workspace=workspace))
            spectra_history.append(
                perpendicular_energy_spectrum_from_state(
                    current,
                    grid,
                    backend,
                    equation_module=s09,
                    params=config,
                )
            )
            sample_index += 1
        if frame_index < len(frame_times) and np.isclose(event_time, frame_times[frame_index]):
            frame_records.append(
                capture_xy_signed_fields(
                    current,
                    time=frame_times[frame_index],
                    grid=grid,
                    fft=fft,
                    backend=backend,
                    z_index=z_index,
                )
            )
            frame_index += 1
        previous_time = float(event_time)

    _plot_summary(sample_times, energy_history, sample_times, spectra_history, output_dir)
    write_signed_xy_frames(frame_records, output_dir=output_dir, grid=grid, z_index=z_index)

    print(
        "forced-turbulence final summary",
        {
            "forcing_band": [config.n_min_force, config.n_max_force],
            "alpha_force": config.alpha_force,
            "forcing_mode": config.forcing_mode,
            "epsilon_plus": config.epsilon_plus,
            "epsilon_minus": config.epsilon_minus,
            "field_energy_injection_rates": dict(config.field_energy_injection_rates),
            "final_energies": energy_history[-1],
        },
    )


if __name__ == "__main__":
    main()
