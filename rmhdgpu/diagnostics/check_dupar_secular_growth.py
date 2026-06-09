"""Check for the k_par = 0 secular (algebraic) growth of `du_par` energy.

The inhomogeneous_rmhd_s equations have a defective zero eigenvalue in the
`k_parallel = 0` plane: there `psi` is frozen and `du_par` is driven by the
time-constant source `-vA*K_b0*dy(psi)` with no restoring `dz` coupling, so

    du_par(t) ~ du_par(0) + (const) * t        ->   du_par_energy ~ t^2 .

An exponential-rate dispersion-relation fit reports this as gamma ~ 0, so it is
invisible to the DR test; this script makes it visible instead.

Usage:
    python -m rmhdgpu.diagnostics.check_dupar_secular_growth \
        examples/outputs_nokia_rmhd_s_24/scalar_diagnostics.csv

If no path is given it defaults to the run referenced by
`examples/nokias_inhomogeneous.input` (outputs_nokia_rmhd_s_24).

It fits `du_par_energy ~ a * t^2` over the tail of the run, prints the quality
of that fit, and saves a plot of `du_par_energy`, `alfvenic_energy`, and
`total_energy` versus time with the fitted `t^2` reference overlaid.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = REPO_ROOT / "examples" / "outputs_nokia_rmhd_s_24" / "scalar_diagnostics.csv"

# Fit du_par_energy ~ a * t^2 using only the tail, where the secular mode
# dominates over the initial transient.
FIT_TAIL_FRACTION = 0.5


def _load_columns(csv_path: Path, columns: list[str]) -> dict[str, np.ndarray]:
    data: dict[str, list[float]] = {name: [] for name in columns}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in columns if name not in (reader.fieldnames or [])]
        if missing:
            raise KeyError(
                f"{csv_path} is missing expected columns {missing!r}. "
                f"Available: {reader.fieldnames!r}"
            )
        for row in reader:
            for name in columns:
                data[name].append(float(row[name]))
    return {name: np.asarray(values, dtype=np.float64) for name, values in data.items()}


def main() -> None:
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
    if not csv_path.exists():
        raise FileNotFoundError(f"No scalar diagnostics CSV at {csv_path}")

    cols = _load_columns(
        csv_path,
        ["time", "du_par_energy", "alfvenic_energy", "total_energy"],
    )
    t = cols["time"]
    e_du = cols["du_par_energy"]
    e_alf = cols["alfvenic_energy"]
    e_tot = cols["total_energy"]

    # Fit du_par_energy ~ a * t^2 on the tail (t > 0 only).
    t_split = t[0] + FIT_TAIL_FRACTION * (t[-1] - t[0])
    fit_mask = (t >= t_split) & (t > 0.0)
    if np.count_nonzero(fit_mask) < 3:
        raise RuntimeError("Not enough tail samples to fit a t^2 law.")

    tf = t[fit_mask]
    ef = e_du[fit_mask]
    # Least-squares slope of E_du against t^2 through the tail.
    a = float(np.sum(tf**2 * ef) / np.sum(tf**4))
    fit = a * t**2

    # Goodness of fit on the tail (R^2 of E_du vs a*t^2).
    resid = ef - a * tf**2
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((ef - ef.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")

    growth = e_du[-1] / e_du[0] if e_du[0] > 0.0 else float("inf")
    print(f"Run: {csv_path}")
    print(f"  samples: {t.size},  t in [{t[0]:.3g}, {t[-1]:.3g}]")
    print(f"  du_par_energy:  start={e_du[0]:.4g}  end={e_du[-1]:.4g}  (x{growth:.4g})")
    print(f"  alfvenic_energy: start={e_alf[0]:.4g}  end={e_alf[-1]:.4g}  (x{e_alf[-1]/e_alf[0]:.4g})")
    print(f"  fit du_par_energy ~ a*t^2 on tail (t>={t_split:.3g}):  a={a:.4g},  R^2={r2:.5f}")
    if r2 > 0.97:
        print("  => consistent with t^2 secular (k_par=0) growth.")
    else:
        print("  => not a clean t^2 law; growth may be exponential or transient.")

    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)

    for ax in (ax_lin, ax_log):
        ax.plot(t, e_du, lw=2.0, label="du_par_energy")
        ax.plot(t, e_alf, lw=1.5, label="alfvenic_energy")
        ax.plot(t, e_tot, lw=1.5, ls=":", label="total_energy")
        ax.plot(t, fit, "k--", lw=1.2, label=rf"$a\,t^2$ fit ($R^2={r2:.4f}$)")
        ax.set_xlabel("t")
        ax.grid(True, alpha=0.3)

    ax_lin.set_ylabel("energy")
    ax_lin.set_title("linear scale")
    ax_log.set_yscale("log")
    ax_log.set_title("log scale")
    ax_log.legend()
    fig.suptitle(f"du_par secular-growth check  ({csv_path.parent.name})")

    out_png = csv_path.parent / "dupar_secular_growth.png"
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"  saved plot to: {out_png}")


if __name__ == "__main__":
    main()
