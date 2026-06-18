"""Side-by-side linear growth-rate (dispersion-relation) scan for the two
inhomogeneous RMHD equation sets.

For each of the two input files

    nokia_inhomo_linear.input    (equation set: inhomogeneous_rmhd)
    nokia_inhomo_linear_s.input  (equation set: inhomogeneous_rmhd_s)

this script runs a sequence of tiny linear simulations with different `kz`
values, fits

    E(t) ~ exp(2 * gamma * t)

to the saved `total_energy` between `FIT_TMIN` and `FIT_TMAX`, and compares the
measured growth rates against the analytic linear dispersion relation. The two
scans are drawn on a shared-axis side-by-side figure so the equation sets can be
compared directly (they are supposed to be the same physics).

Everything written by this script stays under `kz_scan/`.
"""

from __future__ import annotations

import csv
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib


# Keep the main settings together near the top so they are easy to tweak.
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
SCAN_DIR = THIS_DIR / "kz_scan"
PLOT_FILE = SCAN_DIR / "gamma_vs_kpar_comparison31.png"

# The two equation sets to compare. Each entry is (label, input_file, runs_subdir).
INPUT_FILES = [
    ("inhomogeneous_rmhd_rho", THIS_DIR / "nokia_inhomo_linear.input", "runs_nokia"),
    ("inhomogeneous_rmhd_s", THIS_DIR / "nokia_inhomo_linear_s.input", "runs_nokia_s"),
]

FIT_TMIN = 15.0
FIT_TMAX = 18.0


def run_scan(label: str, input_file: Path, runs_subdir: str) -> dict:
    """Run the full kz scan for one input file and return its scan results."""

    with input_file.open("rb") as handle:
        document = tomllib.load(handle)

    grid = document.get("grid", {})
    physics = document.get("physics", {})
    initial_condition = document.get("initial_condition", {})
    initial_parameters = initial_condition.get("parameters", {})

    nz = int(grid.get("Nz", 16))
    lx = float(grid.get("Lx", 2.0 * np.pi))
    ly = float(grid.get("Ly", 2.0 * np.pi))
    lz = float(grid.get("Lz", 2.0 * np.pi))

    vA = float(physics.get("vA"))
    chi = float(physics.get("cs2_over_vA2"))
    g = float(physics.get("g"))
    K_p0 = float(physics.get("K_p0"))
    K_rho0 = float(physics.get("K_rho0"))

    GAMMA = 5.0 / 3.0
    cs2 = chi * vA**2
    alpha = chi / (1.0 + chi)
    vS2 = alpha * vA**2

    K_b0 = g / vA**2 - (chi * K_p0) / GAMMA
    # Effective Brunt-Vaisala frequency squared, matching the equation modules'
    # derived_parameters.N_sq (verified against the linearized ideal_rhs).
    N_sq = - g * (vS2 / cs2 * (K_b0 + chi * K_p0 / GAMMA) - K_rho0)

    base_k_indices = list(initial_parameters.get("k_indices", [0, 1, 0]))
    if len(base_k_indices) != 3:
        raise ValueError(
            f"Expected three k indices in {input_file.name}; got {base_k_indices!r}."
        )

    # k_indices in the input file is [kx, ky, kz].
    kx_index = int(base_k_indices[0])
    ky_index = int(base_k_indices[1])
    # The Nyquist kz (Nz//2) is rejected by the initializer, so scan up to Nz//2 - 1.
    kz_max = nz // 2 - 1

    kx = 2.0 * np.pi * kx_index / lx
    ky = 2.0 * np.pi * ky_index / ly
    kperp = np.hypot(kx, ky)
    if kperp <= 0.0:
        raise ValueError("This scan needs k_perp != 0 so the linear growth rate is defined.")

    runs_dir = SCAN_DIR / runs_subdir
    runs_dir.mkdir(parents=True, exist_ok=True)

    measured_kpar: list[float] = []
    measured_gamma: list[float] = []
    measured_kz: list[int] = []
    failed_kz: list[int] = []

    print(f"\n=== Scanning {label} from {input_file.name} ===")
    print(f"Scanning kz indices from 0 to {kz_max}")
    print(f"Fitting log(total_energy) between t = {FIT_TMIN} and t = {FIT_TMAX}")

    for kz_index in range(kz_max + 1):
        run_output_dir = runs_dir / f"kz_{kz_index:02d}"
        if run_output_dir.exists():
            shutil.rmtree(run_output_dir)

        command = [
            sys.executable,
            "-m",
            "rmhdgpu.run",
            str(input_file),
            "--output-dir",
            str(run_output_dir),
            "--mode-kx",
            str(kx_index),
            "--mode-ky",
            str(ky_index),
            "--mode-kz",
            str(kz_index),
        ]

        print(f"  Running kz index {kz_index}...")
        try:
            subprocess.run(command, cwd=REPO_ROOT, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"    Run failed for kz index {kz_index}: {exc}")
            failed_kz.append(kz_index)
            continue

        scalar_csv = run_output_dir / "scalar_diagnostics.csv"
        time_values: list[float] = []
        energy_values: list[float] = []
        with scalar_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                time_values.append(float(row["time"]))
                energy_values.append(float(row["total_energy"]))

        time = np.asarray(time_values, dtype=np.float64)
        energy = np.asarray(energy_values, dtype=np.float64)

        fit_mask = (time >= FIT_TMIN) & (time <= FIT_TMAX) & (energy > 0.0)
        if np.count_nonzero(fit_mask) < 2:
            print(f"    Not enough positive total_energy samples to fit kz index {kz_index}.")
            failed_kz.append(kz_index)
            continue

        slope, _ = np.polyfit(time[fit_mask], np.log(energy[fit_mask]), 1)
        gamma = 0.5 * slope
        kpar = 2.0 * np.pi * kz_index / lz

        measured_kz.append(kz_index)
        measured_kpar.append(kpar)
        measured_gamma.append(gamma)
        print(f"    k_parallel = {kpar:.6f}, fitted gamma = {gamma:.6f}")

    if not measured_kpar:
        raise RuntimeError(f"No successful growth-rate fits were produced for {label}.")

    # Save the fitted growth rates to a per-equation-set CSV.
    summary_csv = SCAN_DIR / f"gamma_scan_{runs_subdir}.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["kz_index", "k_parallel", "gamma_fit"])
        for kz_index, kpar, gamma in zip(measured_kz, measured_kpar, measured_gamma):
            writer.writerow([int(kz_index), float(kpar), float(gamma)])
    print(f"  Saved fitted growth rates to: {summary_csv}")

    # Analytic linear dispersion relation:
    #   (xi - A)(xi - B) = C        with  xi = omega^2
    #   A = k_par^2 * vS^2
    #   B = k_par^2 * vA^2 - (ky^2 / kperp^2) * N_sq
    #   C = (ky^2 / kperp^2) * k_par^2 * vS^4 * g^2 / cs^4
    # xi < 0 implies instability; the growth rate is gamma = sqrt(-xi).
    theory_kpar = np.linspace(0.0, 2.0 * np.pi * kz_max / lz, 400)
    ky2_over_kperp2 = ky**2 / kperp**2
    A = theory_kpar**2 * vS2
    B = theory_kpar**2 * vA**2 + ky2_over_kperp2 * N_sq
    C = ky2_over_kperp2 * theory_kpar**2 * vS2**2 * g**2 / cs2**2
    disc = np.sqrt((A - B) ** 2 + 4 * C)
    xi_minus = 0.5 * (A + B - disc)
    gamma_minus = np.sqrt(np.clip(-xi_minus, 0.0, None))

    return {
        "label": label,
        "measured_kpar": np.asarray(measured_kpar, dtype=np.float64),
        "measured_gamma": np.asarray(measured_gamma, dtype=np.float64),
        "theory_kpar": theory_kpar,
        "gamma_minus": gamma_minus,
        "failed_kz": failed_kz,
        "params": dict(K_rho0=K_rho0, chi=chi, g=g, vA=vA, kx=kx, ky=ky),
    }


def main() -> None:
    SCAN_DIR.mkdir(parents=True, exist_ok=True)

    results = [run_scan(label, path, runs_subdir) for label, path, runs_subdir in INPUT_FILES]

    fig, axes = plt.subplots(
        1, len(results), figsize=(6.0 * len(results), 4.8), sharey=True, constrained_layout=True
    )
    if len(results) == 1:
        axes = [axes]

    for ax, result in zip(axes, results):
        ax.plot(
            result["theory_kpar"],
            result["gamma_minus"],
            lw=2.0,
            label=r"Linear DR: $\gamma^- = \sqrt{\max(-\xi^-, 0)}$",
        )
        ax.plot(
            result["measured_kpar"],
            result["measured_gamma"],
            "o",
            ms=6,
            label="Measured from runs",
        )
        ax.axhline(0.0, color="0.6", lw=1.0, ls="--")
        ax.set_xlabel(r"$k_\parallel$")
        p = result["params"]
        param_str = (
            rf"$K_{{\rho 0}}={p['K_rho0']:g}$, $\chi={p['chi']:g}$, $g={p['g']:g}$, "
            rf"$v_A={p['vA']:g}$, $k_x={p['kx']:.3g}$, $k_y={p['ky']:.3g}$"
        )
        ax.set_title(f"{result['label']}\n" + param_str)
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[0].set_ylabel(r"$\gamma$")
    fig.suptitle("Inhomogeneous RMHD linear growth scan: equation-set comparison")
    fig.savefig(PLOT_FILE, dpi=200)
    plt.close(fig)

    print(f"\nSaved side-by-side comparison plot to: {PLOT_FILE}")
    for result in results:
        if result["failed_kz"]:
            print(f"  {result['label']}: skipped/failed kz indices: {result['failed_kz']}")


if __name__ == "__main__":
    main()
