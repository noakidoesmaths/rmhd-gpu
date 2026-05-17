"""Quick local scan of low-beta stratified linear growth rates.

This script is intentionally simple and procedural. It works from the
`low_beta_stratified_linear.input` file in the same folder, runs a sequence of
tiny linear simulations with different `kz` values, fits

    E(t) ~ exp(2 * gamma * t)

to the saved `total_energy` between `t = 1` and `t = 6`, and then compares the
measured growth rates against the linear dispersion relation on one plot.

Everything written by this script stays under `quick_tests/`, which is already
ignored by git for this repository.
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
INPUT_FILE = THIS_DIR / "nokia_inhomo_linear.input"
SCAN_DIR = THIS_DIR / "kz_scan"
RUNS_DIR = SCAN_DIR / "runs"
SUMMARY_CSV = SCAN_DIR / "gamma_scan.csv"
PLOT_FILE = SCAN_DIR / "gamma_vs_kpar3.png"

FIT_TMIN = 1.0
FIT_TMAX = 6.0


def main() -> None:
    with INPUT_FILE.open("rb") as handle:
        document = tomllib.load(handle)

    grid = document.get("grid", {})
    physics = document.get("physics", {})
    initial_condition = document.get("initial_condition", {})
    initial_parameters = initial_condition.get("parameters", {})

    nx = int(grid.get("Nx", 16))
    ny = int(grid.get("Ny", 16))
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
    
    # K_b0 = - chi * K_p0 / GAMMA -  g / vA**2
    N_sq = -g * ( K_rho0 + g /(vA**2 * (1 + chi)) )

    base_k_indices = list(initial_parameters.get("k_indices", [0, 1, 0]))
    if len(base_k_indices) != 3:
        raise ValueError(f"Expected three k indices in {INPUT_FILE.name}; got {base_k_indices!r}.")

    # k_indices in the input file is [kx, ky, kz] (see
    # rmhdgpu/initconds/builtin.py::_resolve_single_fourier_mode_indices).
    kx_index = int(base_k_indices[0])
    ky_index = int(base_k_indices[1])
    # The Nyquist kz (Nz//2) is rejected by the initializer, so scan up to Nz//2 - 1.
    kz_max = nz // 2 - 1

    # Convert mode indices to the physical wavenumbers used by the equation set.
    kx = 2.0 * np.pi * kx_index / lx
    ky = 2.0 * np.pi * ky_index / ly
    kperp = np.hypot(kx, ky)
    if kperp <= 0.0:
        raise ValueError("This scan needs k_perp != 0 so the low-beta linear growth rate is defined.")

    SCAN_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    measured_kpar = []
    measured_gamma = []
    measured_kz = []
    failed_kz = []

    print(f"Using input file: {INPUT_FILE}")
    print(f"Scanning kz indices from 0 to {kz_max}")
    print(f"Fitting log(total_energy) between t = {FIT_TMIN} and t = {FIT_TMAX}")

    for kz_index in range(kz_max + 1):
        run_output_dir = RUNS_DIR / f"kz_{kz_index:02d}"

        # Start each run from a clean output directory so the CSV always matches
        # the current kz value.
        if run_output_dir.exists():
            shutil.rmtree(run_output_dir)

        command = [
            sys.executable,
            "-m",
            "rmhdgpu.run",
            str(INPUT_FILE),
            "--output-dir",
            str(run_output_dir),
            "--mode-kx",
            str(kx_index),
            "--mode-ky",
            str(ky_index),
            "--mode-kz",
            str(kz_index),
        ] 

        print()
        print(f"Running kz index {kz_index}...")
        try:
            subprocess.run(command, cwd=REPO_ROOT, check=True)
        except subprocess.CalledProcessError as exc:
            # Keep going if one mode fails. In practice, this can happen at the
            # Nyquist kz if the chosen initializer rejects it.
            print(f"  Run failed for kz index {kz_index}: {exc}")
            failed_kz.append(kz_index)
            continue

        scalar_csv = run_output_dir / "scalar_diagnostics.csv"
        time_values = []
        energy_values = []

        with scalar_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                time_values.append(float(row["time"]))
                energy_values.append(float(row["total_energy"]))

        time = np.asarray(time_values, dtype=np.float64)
        energy = np.asarray(energy_values, dtype=np.float64)

        # The early-time transient is deliberately excluded from the fit.
        fit_mask = (time >= FIT_TMIN) & (time <= FIT_TMAX) & (energy > 0.0)
        if np.count_nonzero(fit_mask) < 2:
            print(f"  Not enough positive total_energy samples to fit kz index {kz_index}.")
            failed_kz.append(kz_index)
            continue

        slope, intercept = np.polyfit(time[fit_mask], np.log(energy[fit_mask]), 1)
        gamma = 0.5 * slope
        kpar = 2.0 * np.pi * kz_index / lz

        measured_kz.append(kz_index)
        measured_kpar.append(kpar)
        measured_gamma.append(gamma)

        print(f"  k_parallel = {kpar:.6f}")
        print(f"  fitted gamma = {gamma:.6f}")
        print(f"  fit intercept = {intercept:.6f}")

    if not measured_kpar:
        raise RuntimeError("No successful growth-rate fits were produced.")

    measured_kpar = np.asarray(measured_kpar, dtype=np.float64)
    measured_gamma = np.asarray(measured_gamma, dtype=np.float64)
    measured_kz = np.asarray(measured_kz, dtype=np.int64)

    # Save the fitted growth rates in a small CSV so the measured points are
    # easy to inspect without reopening the plot.
    with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["kz_index", "k_parallel", "gamma_fit"])
        for kz_index, kpar, gamma in zip(measured_kz, measured_kpar, measured_gamma):
            writer.writerow([int(kz_index), float(kpar), float(gamma)])

    # we have
    #   (xi - A)(xi - B) = C        with  xi = omega^2
    #   A = k_par^2 * vS^2
    #   B = k_par^2 * vA^2 - (ky^2 / kperp^2) * N_sq
    #   C = (ky^2 / kperp^2) * k_par^2 * vS^4 * g^2 / cs^4
    #   xi = +/- 0.5 * ((A + B) +/- sqrt((A - B)^2 + 4 C)).
    # since our solutions follow exp(-i omega) 
    # if xi < 0 then our solution becomes unstable, and we have  
    # plotting gamma = sqrt(-xi) when xi<0 tells us the unstable behaviour
    # set xi = 0 otherwise.
    #
    # we have A > 0, C > 0, (as all those constants are always > 0)
    # if we set 
    # so B is the variable we adjust for xi<0
    # we have 
    # B = kpar^2 vA^2 + ky^2 * g * (Krho + g/((1 + chi)*vA^2)) /kperp^2  
    # the variable that controls B becoming negative is Krho
    # 

    theory_kpar = np.linspace(0, 2 * np.pi * kz_max / lz, 400)
    ky2_over_kperp2 = ky**2 / kperp**2
 
    A = theory_kpar**2 * vS2
    B = theory_kpar**2 * vA**2 - ky2_over_kperp2 * N_sq
    # C = ky2_over_kperp2 * theory_kpar**2 * vS2**4 * g**2 / cs2**4
    C = ky2_over_kperp2 * theory_kpar**2 * vS2**2 * g**2 / cs2**2

    disc = np.sqrt((A - B)**2 + 4 * C)
    xi_plus = 0.5 * (A + B + disc)
    xi_minus = 0.5 * (A + B - disc)

    gamma_plus = np.sqrt(np.clip(-xi_plus, 0.0, None))
    gamma_minus = np.sqrt(np.clip(-xi_minus, 0.0, None))

    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    ax.plot(theory_kpar, gamma_plus, lw=2.0, label=r"Linear DR: $\gamma^+ = \sqrt{\max(\gamma^2, 0)}$")
    ax.plot(theory_kpar, gamma_minus, lw=2.0, label=r"Linear DR: $\gamma- = \sqrt{\max(\gamma^2, 0)}$")
    ax.plot(measured_kpar, measured_gamma, "o", ms=6, label="Measured from runs")
    ax.axhline(0.0, color="0.6", lw=1.0, ls="--")
    ax.set_xlabel(r"$k_\parallel$")
    ax.set_ylabel(r"$\gamma$")
    param_str = (
    rf"$K_{{\rho 0}}={K_rho0:g}$, $K_{{p 0}}={K_p0:g}$, $\chi={chi:g}$, $g={g:g}$, "
    rf"$v_A={vA:g}$, $k_x={kx:g}$, $k_y={ky:g}$, $N^2={N_sq:.3g}$"
    )
    ax.set_title("Inhomogeneous RMHD linear growth scan\n" + param_str)
    # ax.set_title("Inhomogeneous RMHD linear growth scan. ")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(PLOT_FILE, dpi=200)
    plt.close(fig)

    print()
    print(f"Saved fitted growth rates to: {SUMMARY_CSV}")
    print(f"Saved comparison plot to:   {PLOT_FILE}")
    if failed_kz:
        print(f"Skipped or failed kz indices: {failed_kz}")


if __name__ == "__main__":
    main()
