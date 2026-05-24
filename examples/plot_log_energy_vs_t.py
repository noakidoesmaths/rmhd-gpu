"""Plot log(total_energy) vs t for each kz scan run, marking the crossover t_x.

Reads the CSVs produced by `scan_inhomo_rmhd_growth.py` (under
`examples/kz_scan/runs/kz_XX/scalar_diagnostics.csv`) and makes a small grid of
plots showing the kink structure in log E(t):

  - flat, oscillating region for t < t_x  (stable + cross-term modes dominate)
  - clean line of slope 2*gamma for t > t_x  (growing eigenmode dominates)

For each run, the late-time asymptote is fit from [TAIL_TMIN, TAIL_TMAX], and
t_x is estimated as the time at which that asymptote crosses the median level
of log E in the early "flat" window [EARLY_TMIN, EARLY_TMAX].
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


THIS_DIR = Path(__file__).resolve().parent
SCAN_DIR = THIS_DIR / "kz_scan"
RUNS_DIR = SCAN_DIR / "runs"
PLOT_FILE = SCAN_DIR / "log_energy_vs_time.png"

# Late-time window for fitting the asymptote y = 2*gamma*t + log E_grow.
TAIL_TMIN = 15.0
TAIL_TMAX = 18.0

# Early-time window for estimating the flat / oscillating baseline level.
EARLY_TMIN = 0.0
EARLY_TMAX = 3.0


def load_run(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    times: list[float] = []
    energies: list[float] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            times.append(float(row["time"]))
            energies.append(float(row["total_energy"]))
    return np.asarray(times, dtype=np.float64), np.asarray(energies, dtype=np.float64)


def fit_line(t: np.ndarray, y: np.ndarray, tmin: float, tmax: float):
    mask = (t >= tmin) & (t <= tmax) & np.isfinite(y)
    if mask.sum() < 2:
        return None, None
    slope, intercept = np.polyfit(t[mask], y[mask], 1)
    return float(slope), float(intercept)


def median_level(t: np.ndarray, y: np.ndarray, tmin: float, tmax: float) -> float:
    mask = (t >= tmin) & (t <= tmax) & np.isfinite(y)
    return float(np.median(y[mask])) if mask.any() else float("nan")


def main() -> None:
    if not RUNS_DIR.exists():
        raise SystemExit(f"Runs directory not found: {RUNS_DIR}. Run scan_inhomo_rmhd_growth.py first.")

    run_dirs = sorted(d for d in RUNS_DIR.iterdir() if d.is_dir() and d.name.startswith("kz_"))
    if not run_dirs:
        raise SystemExit(f"No kz_XX subdirectories under {RUNS_DIR}.")

    n = len(run_dirs)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(4.0 * cols, 3.0 * rows),
        constrained_layout=True,
        sharex=True,
    )
    axes = np.atleast_1d(axes).ravel()

    for ax, run_dir in zip(axes, run_dirs):
        label = run_dir.name
        csv_path = run_dir / "scalar_diagnostics.csv"
        if not csv_path.exists():
            ax.set_title(f"{label}: missing CSV")
            ax.set_axis_off()
            continue

        t, e = load_run(csv_path)
        log_e = np.full_like(e, np.nan)
        positive = e > 0.0
        log_e[positive] = np.log(e[positive])

        ax.plot(t, log_e, lw=1.0, color="tab:blue", label=r"$\log E(t)$")

        slope, intercept = fit_line(t, log_e, TAIL_TMIN, TAIL_TMAX)
        flat = median_level(t, log_e, EARLY_TMIN, EARLY_TMAX)

        if slope is not None:
            t_line = np.linspace(t.min(), t.max(), 200)
            ax.plot(
                t_line, slope * t_line + intercept,
                ls="--", color="tab:orange", lw=1.2,
                label=fr"asymptote, $\hat\gamma={slope/2:.3f}$",
            )

            if np.isfinite(flat):
                ax.axhline(flat, color="0.5", lw=0.8, ls=":", label="early level")
                if slope != 0.0:
                    t_cross = (flat - intercept) / slope
                    if t.min() <= t_cross <= t.max():
                        ax.axvline(
                            t_cross, color="tab:red", lw=1.0, ls=":",
                            label=fr"$t_\times \approx {t_cross:.2f}$",
                        )

        # Shade the two fit windows so they are easy to see.
        ax.axvspan(EARLY_TMIN, EARLY_TMAX, color="0.85", alpha=0.4, lw=0)
        ax.axvspan(TAIL_TMIN, TAIL_TMAX, color="tab:orange", alpha=0.15, lw=0)

        ax.set_title(label)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="lower right")

    for ax in axes[n:]:
        ax.set_visible(False)

    for ax in axes[:n]:
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$\log E(t)$")

    fig.suptitle(
        r"$\log E(t)$ per kz: early flat region, asymptote $2\gamma t + c$, and crossover $t_\times$"
    )
    PLOT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_FILE, dpi=200)
    plt.close(fig)

    print(f"Saved figure to: {PLOT_FILE}")


if __name__ == "__main__":
    main()
