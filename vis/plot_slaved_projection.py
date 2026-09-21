"""Compare compressive RMS and the reflected-wave ratio with Eqs. (47) and (73).

Based on Squire et al., arXiv:2607.08036, Eqs. (46)-(48) and (73).
For inhomogeneous_rmhd_rho, all background gradients point along x:

    predicted_rms = abs(F) * l_perp * rms(z_x) / z_rms

Here z = zhat cross grad_perp(phi - sign*psi), z_rms = 2*sqrt(W), and
l_perp = 1/<k_perp>. The solver's operators and diagnostics calculate these.
This is a mixing-length estimate, not an exact pointwise solution.

Snapshots measure rms(z_x)/z_rms directly. --from-csv assumes 1/sqrt(2)
(Eq. 48) and uses the saved plus-branch perpendicular wavenumber instead.
The title averages measured chi_A = z_rms*<k_perp>/(vA*<k_parallel>) over
the plotted times, omitting undefined values.

The extra CCR panel divides both sides of Eq. (73) by the pump RMS amplitude:
    z_ratio = sqrt(W_counter/W_pump)
    z_ratio_predicted = (l_perp/z_pump_rms**2) * abs(g) * rms(drho)
Here drho is the measured delta rho/rho_0, and the selected branch is the pump.
For the plus pump these are z^-/z^+ and (l_perp/(z^+)**2) * |-g delta rho/rho_0|_rms.
Both curves are dimensionless and undefined when the pump amplitude is zero.
There is no curvature term in this model and no extra isotropy factor here.

Examples (also usable as main([...]) in Spyder):
    python vis/plot_slaved_projection.py RUN_DIRECTORY
    python vis/plot_slaved_projection.py RUN_DIRECTORY --fields drho db_par --show
    python vis/plot_slaved_projection.py RUN_DIRECTORY --from-csv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys
from types import SimpleNamespace

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from rmhdgpu.backend import build_backend
from rmhdgpu.diagnostics.compressive_channels import (
    ChannelFields,
    elsasser_energy,
    elsasser_kperp_mean,
    elsasser_kprl_mean,
    elsasser_potential,
)
from rmhdgpu.equations.rmhd_by_nokia_rho import derived_parameters
from rmhdgpu.fft import FFTManager
from rmhdgpu.grid import build_grid
from rmhdgpu.operators import dy, inv_lap_perp
from vis._matplotlib import finalize_figure, import_pyplot


# Edit these to change the field labels and colours in the plots.
LABELS = {
    "drho": r"\delta\rho/\rho_0",
    "du_par": r"\delta u_\parallel/v_A",
    "db_par": r"(v_A^2/v_S^2)\,\delta B_\parallel/B_0",
}
COLORS = {"drho": "tab:blue", "du_par": "tab:green", "db_par": "tab:orange"}
BRANCH_SIGN = {"plus": 1, "minus": -1}
ISOTROPIC_ALIGNMENT = 1.0 / np.sqrt(2.0)


def forcings(p):
    """The x components of Eq. (46), for straight field and no mean flow."""
    return {
        # Equivalent to -N_sq/g, but also defined when g = 0.
        "drho": p.g / (p.vA**2 * (1.0 + p.chi)) - p.K_rho0,
        "du_par": -p.K_b0,
        "db_par": -p.K_b0 + p.K_p0 / p.gamma,
    }


def measured_divisors(p):
    """Convert saved fields to Eq. (47) units; alpha = vS^2/vA^2."""
    return {"drho": 1.0, "du_par": p.vA, "db_par": p.alpha}


def rms(values):
    return float(np.sqrt(np.mean(values**2)))


@dataclass
class RunSeries:
    """Arrays ready to plot, with one value per saved time."""

    times: np.ndarray
    l_perp: np.ndarray
    alignment: np.ndarray  # rms(z_x) / z_rms
    chi_a: np.ndarray  # nonlinear frequency / Alfven frequency
    z_ratio: np.ndarray  # Opposite-branch RMS / pump RMS (dimensionless)
    z_ratio_predicted: np.ndarray  # Eq. (73) / pump RMS, using measured density
    measured: dict[str, np.ndarray]
    predicted: dict[str, np.ndarray]
    source: str
    branch: str | None


def load_parameters(run_dir):
    input_path = run_dir / "input_copy.input"
    if not input_path.is_file():
        raise SystemExit(f"Missing {input_path}; the estimate needs the [physics] block.")
    with input_path.open("rb") as handle:
        physics = tomllib.load(handle).get("physics", {})
    missing = [key for key in ("vA", "cs2_over_vA2", "g", "K_p0", "K_rho0") if key not in physics]
    if missing:
        raise SystemExit(f"{input_path} [physics] is missing {missing}.")
    return derived_parameters(physics)


def solver_objects(snapshot_path):
    """Build the CPU grid and FFT once, using the saved periodic axes."""
    import h5py

    with h5py.File(snapshot_path, "r") as handle:
        x, y, z = [handle[f"metadata/{axis}"][:] for axis in ("x", "y", "z")]
    # A periodic box includes one more spacing than the span of its grid points.
    lengths = [float(a[-1] - a[0] + a[1] - a[0]) for a in (x, y, z)]
    config = SimpleNamespace(
        backend="numpy", fft_workers=1,
        Nx=len(x), Ny=len(y), Nz=len(z),
        Lx=lengths[0], Ly=lengths[1], Lz=lengths[2],
        real_dtype=np.float64, complex_dtype=np.complex128,
    )
    backend = build_backend(config)
    grid = build_grid(config, backend)
    return backend, grid, FFTManager(grid, backend)


def read_from_snapshots(run_dir, fields, p, branch=None, stride=1):
    """Read fields, measure the Elsasser projection, then calculate each RMS."""
    import h5py

    if stride < 1:
        raise SystemExit("--stride must be at least 1.")
    paths = sorted((run_dir / "fullfields").glob("fullfield_*.h5"))[::stride]
    if not paths:
        raise SystemExit(f"No full-field snapshots in {run_dir / 'fullfields'}.")
    print(f"Reading {len(paths)} snapshots from {run_dir.name}/fullfields ...")
    backend, grid, fft = solver_objects(paths[0])
    sign = BRANCH_SIGN.get(branch)
    forcing, divisor = forcings(p), measured_divisors(p)
    times, lengths, alignments, chi_a_values = [], [], [], []
    z_ratio, z_ratio_predicted = [], []
    measured = {name: [] for name in fields}
    predicted = {name: [] for name in fields}

    for path in paths:
        with h5py.File(path, "r") as handle:
            output = handle["output"]
            times.append(float(output["time"][()]))
            phi_hat = inv_lap_perp(fft.r2c(output["omega"][:]), grid)
            psi_hat = fft.r2c(output["psi"][:])
            for name in fields:
                measured[name].append(rms(output[name][:]) / divisor[name])
            # Eq. (73) also needs density when its own panel is deselected.
            density_rms = measured["drho"][-1] if "drho" in fields else np.nan
            if "drho" not in fields and "drho" in output:
                density_rms = rms(output["drho"][:])

        alfvenic = ChannelFields(phi_hat=phi_hat, psi_hat=psi_hat)
        w_plus = elsasser_energy(alfvenic, grid, backend, sign=1)
        w_minus = elsasser_energy(alfvenic, grid, backend, sign=-1)
        # Auto picks the stronger branch at the FIRST snapshot and keeps it.
        if sign is None:
            sign = 1 if w_plus >= w_minus else -1

        energy = w_plus if sign > 0 else w_minus
        counter_energy = w_minus if sign > 0 else w_plus
        kperp = elsasser_kperp_mean(alfvenic, grid, backend, sign=sign)
        kparallel = elsasser_kprl_mean(alfvenic, grid, backend, sign=sign)
        potential_hat = elsasser_potential(alfvenic, sign=sign)
        z_x = -fft.c2r(dy(potential_hat, grid))  # zhat cross grad has x = -dy
        z_rms = 2.0 * np.sqrt(max(energy, 0.0))  # W = <|z|^2>/4
        l_perp = 1.0 / kperp if kperp > 0.0 else np.nan
        alignment = rms(z_x) / z_rms if z_rms > 0.0 else 0.0
        omega_a = p.vA * kparallel
        chi_a_values.append(z_rms * kperp / omega_a if omega_a > 0.0 else np.nan)
        z_counter_rms = 2.0 * np.sqrt(max(counter_energy, 0.0))
        z_ratio.append(z_counter_rms / z_rms if z_rms > 0.0 else np.nan)
        # Divide Eq. (73) by the pump RMS to get the dimensionless wave ratio.
        z_ratio_predicted.append(
            l_perp * abs(p.g) * density_rms / z_rms**2 if z_rms > 0.0 else np.nan
        )

        # RMS[(l_perp/z_rms) * z_x * F] factorises because F is constant.
        # With no Alfvenic field the prediction is zero, even though l_perp is undefined.
        projected_length = l_perp * alignment if z_rms > 0.0 else 0.0
        for name in fields:
            predicted[name].append(abs(forcing[name]) * projected_length)
        lengths.append(l_perp)
        alignments.append(alignment)

    return RunSeries(
        times=np.array(times), l_perp=np.array(lengths), alignment=np.array(alignments),
        chi_a=np.array(chi_a_values),
        z_ratio=np.array(z_ratio), z_ratio_predicted=np.array(z_ratio_predicted),
        measured={name: np.array(values) for name, values in measured.items()},
        predicted={name: np.array(values) for name, values in predicted.items()},
        source="full fields", branch="plus" if sign > 0 else "minus",
    )


def read_from_csv(run_dir, fields, p):
    """Use saved RMS and w_plus_kperp, assuming Eq. (48) isotropic alignment."""
    csv_path = run_dir / "scalar_diagnostics.csv"
    if not csv_path.is_file():
        raise SystemExit(f"Missing {csv_path}.")
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"{csv_path} contains no data rows.")
    time_key = "time" if "time" in rows[0] else "t"
    required = [time_key, "w_plus_kperp"] + [f"{name}_rms" for name in fields]
    missing = [name for name in required if name not in rows[0]]
    if missing:
        raise SystemExit(f"{csv_path} is missing {missing}. Use full-field snapshots instead.")
    optional = [name for name in ("w_plus", "w_minus", "w_plus_kprl", "drho_rms")
                if name in rows[0] and name not in required]
    columns = {name: np.array([float(row[name]) for row in rows]) for name in required + optional}

    kperp = columns["w_plus_kperp"]
    l_perp = np.divide(1.0, kperp, out=np.full_like(kperp, np.nan), where=np.abs(kperp) > 0.0)
    # Older CSVs can still be plotted when the chi_A diagnostics are absent.
    chi_a = np.full_like(kperp, np.nan)
    if "w_plus" in columns and "w_plus_kprl" in columns:
        omega_nl = 2.0 * np.sqrt(columns["w_plus"]) * kperp
        omega_a = p.vA * columns["w_plus_kprl"]
        np.divide(omega_nl, omega_a, out=chi_a, where=omega_a > 0.0)
    z_ratio = np.full_like(kperp, np.nan)
    z_ratio_predicted = np.full_like(kperp, np.nan)
    if "w_plus" in columns:
        z_pump = 2.0 * np.sqrt(np.maximum(columns["w_plus"], 0.0))
        if "w_minus" in columns:
            z_counter_rms = 2.0 * np.sqrt(np.maximum(columns["w_minus"], 0.0))
            np.divide(z_counter_rms, z_pump, out=z_ratio, where=z_pump > 0.0)
        if "drho_rms" in columns:
            np.divide(l_perp * abs(p.g) * columns["drho_rms"], z_pump**2,
                      out=z_ratio_predicted, where=z_pump > 0.0)
    forcing, divisor = forcings(p), measured_divisors(p)
    return RunSeries(
        times=columns[time_key], l_perp=l_perp,
        alignment=np.full_like(l_perp, ISOTROPIC_ALIGNMENT),
        chi_a=chi_a,
        z_ratio=z_ratio, z_ratio_predicted=z_ratio_predicted,
        measured={name: columns[f"{name}_rms"] / divisor[name] for name in fields},
        predicted={name: abs(forcing[name]) * l_perp * ISOTROPIC_ALIGNMENT for name in fields},
        source="CSV: Eq. (48) isotropy assumed, plus branch", branch=None,
    )


def plot_series(run_dir, run, fields, p, output_path, show=False):
    """One panel per compressive field, followed by the dimensionless wave ratio."""
    plt = import_pyplot(show=show)
    forcing = forcings(p)
    panel_count = len(fields) + 1
    rows = (panel_count + 1) // 2
    fig, axes = plt.subplots(
        rows, 2, figsize=(11.6, 5.0 * rows),
        constrained_layout=True, squeeze=False,
    )
    for ax, name in zip(axes.ravel(), fields):
        measured, predicted = run.measured[name], run.predicted[name]
        ax.plot(run.times, measured, lw=2, color=COLORS[name], label="Measured RMS")
        if forcing[name] == 0.0:
            ax.axhline(0.0, color="black", ls="--", lw=1.6, label="Slaved estimate: zero drive")
            summary = "No background-gradient drive"
        else:
            ax.plot(run.times, predicted, "k--", lw=1.6, label="Slaved estimate (Eq. 47)")
            # ratio = np.divide(measured, predicted, out=np.full_like(measured, np.nan), where=predicted != 0)
            # valid = ratio[np.isfinite(ratio)]
            # mean_ratio = float(valid.mean()) if valid.size else np.nan
            # summary = f"Mean measured / estimate = {mean_ratio:.3g}"
        ax.set(xlabel="Time", ylabel=rf"RMS of ${LABELS[name]}$",
               title=rf"Plot of measured and predicted ${LABELS[name]}$" + "\nagainst time")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

    ax = axes.ravel()[len(fields)]
    # The paper calls the pump z+; swap solver labels when --branch minus is used.
    pump_label = r"z^-" if run.branch == "minus" else r"z^+"
    counter_label = r"z^+" if run.branch == "minus" else r"z^-"
    ratio_label = rf"{counter_label}_{{\rm rms}}/{pump_label}_{{\rm rms}}"
    ax.plot(run.times, run.z_ratio, color="tab:purple", lw=2,
            label=rf"Measured ${ratio_label}$")
    ax.plot(run.times, run.z_ratio_predicted, "k--", lw=1.6,
            label=rf"$(\ell_\perp/({pump_label}_{{\rm rms}})^2)\,|-g\,\delta\rho/\rho_0|_{{\rm rms}}$")
    ax.set(xlabel="Time", ylabel="RMS amplitude ratio (dimensionless)",
           title=rf"${ratio_label}$: measured and estimated")
    notes = []
    if not np.isfinite(run.z_ratio).any():
        notes.append("Measured ratio unavailable: missing energy or zero pump amplitude")
    if not np.isfinite(run.z_ratio_predicted).any():
        notes.append("Estimate unavailable: missing density or no finite pump timescale")
    if notes:
        ax.text(0.5, 0.45, "\n".join(notes), transform=ax.transAxes,
                ha="center", va="center", fontsize=8, wrap=True)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    for ax in axes.ravel()[panel_count:]:
        ax.set_visible(False)

    branch_label = "" if run.branch is None else f"; {run.branch} branch"
    finite_chi_a = run.chi_a[np.isfinite(run.chi_a)]
    chi_a_label = f"{finite_chi_a.mean():.3g}" if finite_chi_a.size else "n/a"
    fig.suptitle(
        f"{run_dir.name}: slaved amplitudes\n{run.source}{branch_label} | "
        rf"$g$ = {p.g:.4g}, $\chi$ = {p.chi:g}, $K_\rho$ = {p.K_rho0:g} | "
        rf"mean RMS$(z_x)/z_{{\rm rms}}$ = {np.mean(run.alignment):.3f} "
        rf"(isotropic: {ISOTROPIC_ALIGNMENT:.3f})" + "\n"
        rf"mean measured $\chi_A$ = {chi_a_label}",
        fontsize=10,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    finalize_figure(fig, output_path=output_path, show=show, plt=plt)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path, help="Run directory containing input_copy.input.")
    parser.add_argument("--fields", nargs="+", choices=tuple(LABELS), default=list(LABELS),
                        help="Compressive panels to show; the Eq. (73) panel is always added.")
    parser.add_argument("--from-csv", action="store_true",
                        help="Use scalar diagnostics, plus-branch k_perp and isotropic alignment.")
    parser.add_argument("--branch", choices=tuple(BRANCH_SIGN),
                        help="Snapshot branch; default picks the stronger one at the first snapshot.")
    parser.add_argument("--stride", type=int, default=1, help="Read every Nth snapshot (default: 1).")
    parser.add_argument("--output", type=Path, help="Image path; default: RUN_DIRECTORY/slaved_projection.png.")
    parser.add_argument("--show", action="store_true", help="Show the figure after saving (e.g. in Spyder).")
    return parser


def main(argv=None) -> Path:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.stride < 1:
        parser.error("--stride must be at least 1.")
    run_dir = args.path.expanduser().resolve()
    p = load_parameters(run_dir)

    has_snapshots = any((run_dir / "fullfields").glob("fullfield_*.h5"))
    if args.from_csv or not has_snapshots:
        if not args.from_csv:
            print(f"No snapshots in {run_dir.name}/fullfields; falling back to the CSV.")
        if args.branch is not None:
            print("CSV mode uses w_plus_kperp; --branch applies only to snapshots.")
        run = read_from_csv(run_dir, args.fields, p)
    else:
        run = read_from_snapshots(run_dir, args.fields, p, args.branch, args.stride)

    output_path = (run_dir / "slaved_projection.png" if args.output is None
                   else args.output.expanduser().resolve())
    plot_series(run_dir, run, args.fields, p, output_path, args.show)
    return output_path


if __name__ == "__main__":
    main()
