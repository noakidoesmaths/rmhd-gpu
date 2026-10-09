"""Run a small imbalanced-Alfvénic forcing scan and plot its response.

This is a qualitative 32^3 sanity experiment, not a formal pytest test. It
uses the normal ``rmhdgpu.run`` path for each forcing ratio, including saved
resolved configuration, scalar diagnostics, spectra, and auto dissipation.

For ``R = epsilon_plus / epsilon_minus``, the default total Elsasser forcing is

``epsilon_plus + epsilon_minus = 0.7 * R^(-2/3)``.

The factor 0.7 gives package Alfvénic injection rate 0.35 in the balanced case,
because ``E_A = (E_plus + E_minus) / 2``. It was calibrated to give
``u_perp,rms`` of order one at 32^3. The ``R^(-2/3)`` reduction follows the
requested heuristic compensation for the longer nonlinear time at strong
imbalance.

Run with:

``python -m rmhdgpu.examples.sanity_imbalanced_forcing``
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rmhdgpu.run import run_simulation
from rmhdgpu.runfile import resolve_run_settings


def forcing_rates(ratio: float, balanced_epsilon_sum: float) -> tuple[float, float]:
    """Return ``(epsilon_plus, epsilon_minus)`` for one imbalance ratio."""

    if ratio < 1.0:
        raise ValueError(f"For this scan, epsilon_plus/epsilon_minus must be >= 1; got {ratio}.")
    epsilon_sum = balanced_epsilon_sum * ratio ** (-2.0 / 3.0)
    epsilon_minus = epsilon_sum / (1.0 + ratio)
    epsilon_plus = ratio * epsilon_minus
    return epsilon_plus, epsilon_minus


def _case_input_text(args: argparse.Namespace, ratio: float, case_output_dir: Path) -> str:
    epsilon_plus, epsilon_minus = forcing_rates(ratio, args.balanced_epsilon_sum)
    scalar_cadence = args.t_final / args.scalar_samples
    spectrum_cadence = args.t_final / args.spectrum_samples
    workers = "" if args.fft_workers is None else f"fft_workers = {args.fft_workers}\n"
    return f'''title = "Imbalanced forcing R={ratio:g}"
output_dir = "{case_output_dir.name}"

[equations]
type = "alfvenic"
mode = "nonlinear"

[grid]
Nx = {args.n}
Ny = {args.n}
Nz = {args.n}

[time]
tmax = {args.t_final!r}
dt_init = 0.005
dt_max = {args.dt_max!r}
cfl_number = {args.cfl_number!r}
use_variable_dt = true

[output]
t_out_scal = {scalar_cadence!r}
t_out_spec = {spectrum_cadence!r}
t_out_full = 0.0

[backend]
backend = "{args.backend}"
{workers}
[runtime]
runtime_check_every = 10
progress_output_every = {args.progress_output_every}
fail_on_nonfinite = true
dealias = true

[forcing]
use_forcing = true
forcing_mode = "elsasser"
n_min_force = {args.n_min_force!r}
n_max_force = {args.n_max_force!r}
alpha_force = {args.alpha_force!r}
epsilon_plus = {epsilon_plus!r}
epsilon_minus = {epsilon_minus!r}
forcing_seed = {args.seed}

[dissipation]
mode = "auto"
n_perp = 3
n_par = 1
nu_par = 0.0
kd_fraction = 0.6
shell_half_width = 0.5
update_every = 10
smooth_factor = 0.4
nu_min = 1e-10
nu_max = 1.0
max_update_factor = 4.0

[initial_condition]
type = "zero"
'''


def _read_scalar_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"No scalar diagnostics were written to {path}.")
    return {
        key: np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        for key in rows[0]
    }


def _read_latest_elsasser_spectra(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No spectra were written to {path}.")
    latest_time = max(float(row["time"]) for row in rows)
    result: dict[str, np.ndarray] = {}
    for quantity in ("z_plus", "z_minus"):
        pairs = sorted(
            (
                (float(row["kperp"]), float(row["value"]))
                for row in rows
                if row["quantity"] == quantity and np.isclose(float(row["time"]), latest_time)
            ),
            key=lambda pair: pair[0],
        )
        if not pairs:
            raise RuntimeError(f"Spectrum {quantity!r} was not found in {path}.")
        values = np.asarray(pairs, dtype=np.float64)
        result["kperp"] = values[:, 0]
        result[quantity] = values[:, 1]
    return result


def _late_time_ratio(scalars: dict[str, np.ndarray], late_fraction: float) -> float:
    time = scalars["time"]
    late = time >= late_fraction * time[-1]
    energy_plus = float(np.mean(scalars["elsasser_energy_plus"][late]))
    energy_minus = float(np.mean(scalars["elsasser_energy_minus"][late]))
    return energy_plus / energy_minus


def _plot_results(
    records: list[dict[str, Any]],
    *,
    output_dir: Path,
    late_fraction: float,
) -> tuple[Path, float, float]:
    ratios = np.asarray([record["ratio"] for record in records], dtype=np.float64)
    energy_ratios = np.asarray([record["energy_ratio"] for record in records], dtype=np.float64)
    can_fit = len(records) >= 2 and np.unique(ratios).size >= 2
    if can_fit:
        alpha, log_prefactor = np.polyfit(np.log(ratios), np.log(energy_ratios), 1)
    else:
        alpha = float("nan")
        log_prefactor = float("nan")
    imbalanced = ratios > 1.0
    if np.count_nonzero(imbalanced) >= 2:
        alpha_imbalanced, log_prefactor_imbalanced = np.polyfit(
            np.log(ratios[imbalanced]),
            np.log(energy_ratios[imbalanced]),
            1,
        )
    else:
        alpha_imbalanced = float("nan")
        log_prefactor_imbalanced = float("nan")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(records)))

    for color, record in zip(colors, records, strict=True):
        scalars = record["scalars"]
        label = rf"$R_\epsilon={record['ratio']:g}$"
        axes[0, 0].semilogy(
            scalars["time"],
            scalars["elsasser_energy_plus"],
            color=color,
            lw=2,
            label=label + r" $E^+$",
        )
        axes[0, 0].semilogy(
            scalars["time"],
            scalars["elsasser_energy_minus"],
            color=color,
            lw=1.5,
            ls="--",
            label=label + r" $E^-$",
        )
        finite_ratio = np.isfinite(scalars["elsasser_energy_ratio"])
        axes[0, 1].semilogy(
            scalars["time"][finite_ratio],
            scalars["elsasser_energy_ratio"][finite_ratio],
            color=color,
            lw=2,
            label=label,
        )

        spectra = record["spectra"]
        valid_plus = (spectra["kperp"] > 0.0) & (spectra["z_plus"] > 0.0)
        valid_minus = (spectra["kperp"] > 0.0) & (spectra["z_minus"] > 0.0)
        axes[1, 1].loglog(
            spectra["kperp"][valid_plus],
            spectra["z_plus"][valid_plus],
            color=color,
            lw=2,
        )
        axes[1, 1].loglog(
            spectra["kperp"][valid_minus],
            spectra["z_minus"][valid_minus],
            color=color,
            lw=1.5,
            ls="--",
        )

    axes[1, 0].loglog(ratios, energy_ratios, "o", ms=7, label="late-time measurements")
    if can_fit:
        fit_x = np.geomspace(ratios.min(), ratios.max(), 100)
        fit_y = np.exp(log_prefactor) * fit_x**alpha
        axes[1, 0].loglog(
            fit_x,
            fit_y,
            "-",
            lw=2,
            label=rf"fit: $R_E\propto R_\epsilon^{{{alpha:.2f}}}$",
        )
        axes[1, 0].loglog(fit_x, fit_x**2, ":", color="0.4", label=r"$R_E=R_\epsilon^2$")
        if np.isfinite(alpha_imbalanced):
            fit_y_imbalanced = np.exp(log_prefactor_imbalanced) * fit_x**alpha_imbalanced
            axes[1, 0].loglog(
                fit_x,
                fit_y_imbalanced,
                "--",
                lw=1.7,
                label=rf"$R_\epsilon>1$ fit: $\alpha={alpha_imbalanced:.2f}$",
            )

    axes[0, 0].set(title="Elsasser energies", xlabel="t", ylabel=r"$E^\pm$")
    axes[0, 1].set(title="Instantaneous imbalance", xlabel="t", ylabel=r"$E^+/E^-$")
    axes[1, 0].set(
        title=rf"Late-time fit ($t/T\geq {late_fraction:g}$)",
        xlabel=r"$\epsilon^+/\epsilon^-$",
        ylabel=r"$\langle E^+\rangle/\langle E^-\rangle$",
    )
    axes[1, 1].set(title="Final Elsasser spectra", xlabel=r"$k_\perp$", ylabel=r"$E^\pm(k_\perp)$")
    for axis in axes.flat:
        axis.grid(True, which="both", alpha=0.25)
    axes[0, 0].legend(fontsize=7, ncol=2)
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)

    figure_path = output_dir / "imbalanced_forcing_scan.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    return figure_path, float(alpha), float(alpha_imbalanced)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="sanity_imbalanced_forcing")
    parser.add_argument("--backend", choices=["numpy", "scipy_cpu", "cupy"], default="scipy_cpu")
    parser.add_argument("--fft-workers", type=int, default=8)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--t-final", type=float, default=16.0)
    parser.add_argument("--ratios", nargs="+", type=float, default=[1.0, 2.0, 4.0, 8.0])
    parser.add_argument("--balanced-epsilon-sum", type=float, default=0.7)
    parser.add_argument("--n-min-force", type=float, default=1.0)
    parser.add_argument("--n-max-force", type=float, default=3.0)
    parser.add_argument("--alpha-force", type=float, default=0.0)
    parser.add_argument("--cfl-number", type=float, default=0.3)
    parser.add_argument("--dt-max", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=2718)
    parser.add_argument("--scalar-samples", type=int, default=80)
    parser.add_argument("--spectrum-samples", type=int, default=8)
    parser.add_argument("--late-fraction", type=float, default=0.5)
    parser.add_argument("--progress-output-every", type=int, default=100)
    parser.add_argument("--reuse-existing", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict[str, float]:
    args = build_parser().parse_args(argv)
    if args.backend != "scipy_cpu" and args.fft_workers == 8:
        args.fft_workers = None
    if not 0.0 < args.late_fraction < 1.0:
        raise SystemExit("--late-fraction must lie strictly between 0 and 1.")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for ratio in args.ratios:
        case_dir = output_dir / f"ratio_{ratio:g}"
        case_dir.mkdir(parents=True, exist_ok=True)
        case_output_dir = case_dir / "outputs"
        input_path = case_dir / "case.input"
        input_path.write_text(
            _case_input_text(args, ratio, case_output_dir),
            encoding="utf-8",
        )
        scalar_path = case_output_dir / "scalar_diagnostics.csv"
        spectra_path = case_output_dir / "spectra.csv"
        epsilon_plus, epsilon_minus = forcing_rates(ratio, args.balanced_epsilon_sum)
        print(
            "imbalanced forcing case",
            {
                "ratio": ratio,
                "epsilon_plus": epsilon_plus,
                "epsilon_minus": epsilon_minus,
                "epsilon_sum": epsilon_plus + epsilon_minus,
            },
        )
        if not args.reuse_existing or not (scalar_path.exists() and spectra_path.exists()):
            settings = resolve_run_settings(runfile_path=input_path)
            run_simulation(settings)

        scalars = _read_scalar_csv(scalar_path)
        spectra = _read_latest_elsasser_spectra(spectra_path)
        records.append(
            {
                "ratio": float(ratio),
                "epsilon_plus": epsilon_plus,
                "epsilon_minus": epsilon_minus,
                "energy_ratio": _late_time_ratio(scalars, args.late_fraction),
                "scalars": scalars,
                "spectra": spectra,
            }
        )

    figure_path, alpha, alpha_imbalanced = _plot_results(
        records,
        output_dir=output_dir,
        late_fraction=args.late_fraction,
    )
    summary_path = output_dir / "imbalance_fit.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epsilon_ratio", "epsilon_plus", "epsilon_minus", "energy_ratio"],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "epsilon_ratio": record["ratio"],
                    "epsilon_plus": record["epsilon_plus"],
                    "epsilon_minus": record["epsilon_minus"],
                    "energy_ratio": record["energy_ratio"],
                }
            )

    print(
        "imbalanced forcing fit",
        {
            "alpha": alpha,
            "alpha_imbalanced_only": alpha_imbalanced,
            "energy_ratios": {record["ratio"]: record["energy_ratio"] for record in records},
            "figure": str(figure_path),
            "summary_csv": str(summary_path),
        },
    )
    return {"alpha": alpha, "alpha_imbalanced_only": alpha_imbalanced}


if __name__ == "__main__":
    main()
