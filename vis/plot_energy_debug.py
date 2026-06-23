"""Extra energy-budget debugging plots for the inhomogeneous RMHD runs.

This complements :mod:`vis.plot_budget` (differential closure) with four
diagnostics that help answer "is my run a healthy saturated state, a real
unsaturated instability, or a bug?":

1. **Cumulative budget** -- ``E(0) + integral(rhs_total dt)`` overlaid on the
   measured ``E(t)``. Integral closure is far less noisy than differentiating
   ``E`` and exposes slow systematic energy leaks the differential residual
   hides.
2. **Source vs dissipation** -- the ``stratification`` and ``dissipation`` RHS
   terms and their sum. At a saturated state the two become equal and opposite
   and the sum approaches zero (``d_t E -> 0``).
3. **Balance ratio** -- ``-stratification / dissipation`` versus time. It tends
   to ``1`` at a clean steady balance and oscillates around ``1`` for bursty
   saturation.
4. **Growth-rate overlay** -- ``E(t)`` on a log axis with reference slopes from
   the maximum linear growth rate of :func:`linear_matrix` over the resolved
   ``k`` grid. The convective instability lives in the ``k_par = 0`` plane, so
   the rate is reported separately for ``kz = 0`` and ``kz != 0``; when the run
   sets ``project_kpar0 = true`` the ``kz != 0`` rate is the relevant one.

A second figure plots the **perpendicular dissipation spectrum**
``D(kperp) = sum_field 2 * nu_perp * kperp^(2 n_perp) * E_field(kperp)`` from a
``spectra.csv`` file, showing which scales actually remove energy.

Usage::

    python vis/plot_energy_debug.py examples/outputs_nokia_rmhd_s_25/scalar_diagnostics.csv --show

By default the run's own ``input_copy.input`` (saved next to the CSV) supplies
the physics and dissipation parameters, so the growth-rate overlay always
matches the run that produced the CSV. Use ``--input`` to point elsewhere, and
``--spectra`` to add the dissipation-spectrum figure.

Notes:

- ``resolved_config.toml`` is NOT used: it serializes physics as ``N2`` and
  drops ``g``/``K_p0``/``K_rho0``, so it cannot reconstruct the linear operator
  for this equation set. ``input_copy.input`` keeps them.
- For ``mode = "auto"`` dissipation the per-field ``nu_perp`` is adapted at
  runtime and not knowable statically, so the dissipation-weighted growth rate
  and the dissipation spectrum are skipped (the ideal growth rate is still
  shown).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from vis._matplotlib import finalize_figure, import_pyplot


# linear_matrix field order; matches the equation module docstring.
_MATRIX_FIELD_ORDER = ["psi", "omega", "db_par", "du_par", "s"]
# spectra-quantity -> dissipation field whose operator damps it.
_SPECTRUM_TO_FIELD = {
    "u_perp": "omega",
    "b_perp": "psi",
    "du_par": "du_par",
    "db_par": "db_par",
    "s": "s",
}


# --- small IO helpers -------------------------------------------------------

def _read_scalar_csv(path: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Scalar diagnostics file {path} has no header row.")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Scalar diagnostics file {path} contains no data rows.")
    columns: dict[str, list[float]] = {name: [] for name in reader.fieldnames}
    for row in rows:
        for name in reader.fieldnames:
            columns[name].append(float(row[name]))
    return list(reader.fieldnames), {
        name: np.asarray(values, dtype=np.float64) for name, values in columns.items()
    }


def _read_spectra_csv(path: Path) -> dict[str, dict[float, tuple[np.ndarray, np.ndarray]]]:
    grouped: dict[str, dict[float, list[tuple[float, float]]]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Spectra file {path} has no header row.")
        for row in reader:
            grouped.setdefault(row["quantity"], {}).setdefault(float(row["time"]), []).append(
                (float(row["kperp"]), float(row["value"]))
            )
    result: dict[str, dict[float, tuple[np.ndarray, np.ndarray]]] = {}
    for quantity, by_time in grouped.items():
        result[quantity] = {}
        for time_value, pairs in by_time.items():
            data = np.asarray(sorted(pairs, key=lambda item: item[0]), dtype=np.float64)
            result[quantity][time_value] = (data[:, 0], data[:, 1])
    return result


def _load_toml(path: Path) -> dict[str, Any]:
    try:
        import tomllib  # Python 3.11+
    except ModuleNotFoundError:  # pragma: no cover - interpreter dependent
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ModuleNotFoundError:
            raise SystemExit("Reading the config needs tomllib (Python 3.11+) or the 'tomli' package.")
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _cumulative_trapezoid(time: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Return the running integral of ``values`` over ``time`` (starts at 0)."""

    out = np.zeros_like(values, dtype=np.float64)
    if len(values) < 2:
        return out
    out[1:] = np.cumsum(0.5 * (values[1:] + values[:-1]) * np.diff(time))
    return out


# --- physics / dissipation extraction --------------------------------------

def _extract_physics(input_config: dict[str, Any]) -> dict[str, float] | None:
    """Return the linear-operator params, or None if the file lacks them."""

    physics = input_config.get("physics", {})
    required = ("g", "K_p0", "K_rho0")
    if not all(name in physics for name in required):
        return None
    return {
        "vA": float(physics.get("vA", 1.0)),
        "cs2_over_vA2": float(physics.get("cs2_over_vA2", 1.0)),
        "g": float(physics["g"]),
        "K_p0": float(physics["K_p0"]),
        "K_rho0": float(physics["K_rho0"]),
    }


def _extract_field_damping(input_config: dict[str, Any]) -> dict[str, dict[str, float]] | None:
    """Return per-field ``nu_perp``/``n_perp`` etc., or None for auto/empty.

    ``mode = "auto"`` adapts ``nu`` at runtime, so the statically saved values
    (typically zero) cannot reconstruct the actual dissipation.
    """

    diss = input_config.get("dissipation", {})
    if str(diss.get("mode", "manual")).lower() == "auto":
        return None
    out: dict[str, dict[str, float]] = {}
    any_positive = False
    for field in _MATRIX_FIELD_ORDER:
        spec = diss.get(field, {})
        nu_perp = float(spec.get("nu_perp", 0.0))
        nu_par = float(spec.get("nu_par", 0.0))
        any_positive = any_positive or nu_perp > 0.0 or nu_par > 0.0
        out[field] = {
            "nu_perp": nu_perp,
            "nu_par": nu_par,
            "n_perp": int(spec.get("n_perp", 1)),
            "n_par": int(spec.get("n_par", 1)),
        }
    return out if any_positive else None


def _grid_axes(input_config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid_cfg = input_config.get("grid", {})
    Nx = int(grid_cfg.get("Nx", 32))
    Ny = int(grid_cfg.get("Ny", 32))
    Nz = int(grid_cfg.get("Nz", 32))
    Lx = float(grid_cfg.get("Lx", 2.0 * np.pi))
    Ly = float(grid_cfg.get("Ly", 2.0 * np.pi))
    Lz = float(grid_cfg.get("Lz", 2.0 * np.pi))
    kx = 2.0 * np.pi * np.fft.fftfreq(Nx, d=Lx / Nx)
    ky = 2.0 * np.pi * np.fft.fftfreq(Ny, d=Ly / Ny)
    kz = 2.0 * np.pi * np.fft.rfftfreq(Nz, d=Lz / Nz)
    return kx, ky, kz


def _max_linear_growth_rates(
    input_config: dict[str, Any],
    params: dict[str, float],
    damping: dict[str, dict[str, float]] | None,
) -> dict[str, float]:
    """Max linear growth rate over the resolved grid (energy ~ exp(2*gamma*t)).

    The trivial ``k = 0`` mode is skipped. Results are split into ``kz = 0`` and
    ``kz != 0`` because the convective instability lives in the ``k_par = 0``
    plane that ``project_kpar0`` removes.
    """

    from rmhdgpu.equations.rmhd_by_nokia_s import linear_matrix

    kx_axis, ky_axis, kz_axis = _grid_axes(input_config)

    def damp_diag(kperp2: float, kpar2: float) -> np.ndarray:
        values = []
        for field in _MATRIX_FIELD_ORDER:
            spec = damping[field] if damping is not None else {}
            d = 0.0
            if spec.get("nu_perp", 0.0) > 0.0:
                d += spec["nu_perp"] * kperp2 ** int(spec["n_perp"])
            if spec.get("nu_par", 0.0) > 0.0:
                d += spec["nu_par"] * kpar2 ** int(spec["n_par"])
            values.append(d)
        return np.diag(values).astype(np.complex128)

    result = {
        "ideal_kz0": -np.inf, "ideal_kznz": -np.inf,
        "damped_kz0": -np.inf, "damped_kznz": -np.inf,
    }
    for kx in kx_axis:
        for ky in ky_axis:
            kperp2 = float(kx) ** 2 + float(ky) ** 2
            for kz in kz_axis:
                if kx == 0.0 and ky == 0.0 and kz == 0.0:
                    continue
                kpar2 = float(kz) ** 2
                matrix = linear_matrix(float(kx), float(ky), float(kz), params)
                ideal = float(np.max(np.linalg.eigvals(matrix).real))
                bucket = "kz0" if kz == 0.0 else "kznz"
                result[f"ideal_{bucket}"] = max(result[f"ideal_{bucket}"], ideal)
                if damping is not None:
                    damped = float(np.max(np.linalg.eigvals(matrix - damp_diag(kperp2, kpar2)).real))
                    result[f"damped_{bucket}"] = max(result[f"damped_{bucket}"], damped)
    return result


# --- dissipation spectrum ---------------------------------------------------

def _dissipation_spectrum_figure(
    spectra: dict[str, dict[float, tuple[np.ndarray, np.ndarray]]],
    damping: dict[str, dict[str, float]],
    output_path: Path,
    *,
    plt: Any,
    show: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.8), constrained_layout=True)
    total_kperp: np.ndarray | None = None
    total_rate: np.ndarray | None = None

    for quantity, field in _SPECTRUM_TO_FIELD.items():
        if quantity not in spectra:
            continue
        spec = damping[field]
        nu_perp = spec["nu_perp"]
        n_perp = int(spec["n_perp"])
        if spec["nu_par"] > 0.0:
            print(f"warning: nu_par > 0 for {field!r}; perp shell spectrum omits parallel dissipation.")
        kperp, energy = spectra[quantity][max(spectra[quantity])]
        rate = 2.0 * nu_perp * np.where(kperp > 0.0, kperp, 0.0) ** (2 * n_perp) * energy
        positive = rate > 0.0
        if np.any(positive):
            ax.loglog(kperp[positive], rate[positive], lw=1.8, label=quantity)
        if total_rate is None:
            total_kperp, total_rate = kperp, rate.copy()
        else:
            total_rate = total_rate + rate

    if total_rate is None or not np.any(total_rate > 0.0):
        print("No matching/nonzero dissipation spectra; skipping dissipation spectrum figure.")
        plt.close(fig)
        return

    positive = total_rate > 0.0
    ax.loglog(total_kperp[positive], total_rate[positive], color="black", lw=2.5, label="total")
    ax.set_xlabel(r"$k_\perp$")
    ax.set_ylabel(r"dissipation rate per shell  $2\,D_i(k)\,E_i(k)$")
    ax.set_title("Perpendicular dissipation spectrum (latest time)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    finalize_figure(fig, output_path=output_path, show=show, plt=plt)


# --- main -------------------------------------------------------------------

def _resolve_config_path(csv_path: Path, explicit: str | None) -> Path | None:
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    candidate = csv_path.parent / "input_copy.input"
    return candidate if candidate.exists() else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("csv_path", help="Path to scalar_diagnostics.csv.")
    parser.add_argument("--quantity", default="total_energy", help="Conserved-quantity prefix.")
    parser.add_argument("--input", default=None, help="Config (TOML/.input) for the growth-rate overlay. Defaults to input_copy.input next to the CSV.")
    parser.add_argument("--spectra", default=None, help="Path to spectra.csv for the dissipation spectrum.")
    parser.add_argument("--output-dir", default=None, help="Directory for PNG outputs. Defaults next to the CSV.")
    parser.add_argument("--show", action="store_true", help="Show figures interactively after saving.")
    return parser


def main(argv: list[str] | None = None) -> list[Path]:
    args = build_parser().parse_args(argv)
    plt = import_pyplot(show=args.show)

    csv_path = Path(args.csv_path).expanduser().resolve()
    _, columns = _read_scalar_csv(csv_path)
    time = columns["time"] if "time" in columns else columns["t"]

    quantity = args.quantity
    if quantity not in columns:
        raise SystemExit(f"Quantity {quantity!r} is not present in {csv_path}.")
    energy = columns[quantity]
    stratification = columns.get(f"{quantity}_rhs_stratification")
    dissipation = columns.get(f"{quantity}_rhs_dissipation")
    rhs_total = columns.get(f"{quantity}_rhs_total")

    output_dir = csv_path.parent if args.output_dir is None else Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    # Config for the growth-rate overlay / dissipation spectrum.
    input_config: dict[str, Any] | None = None
    physics: dict[str, float] | None = None
    field_damping: dict[str, dict[str, float]] | None = None
    config_path = _resolve_config_path(csv_path, args.input)
    if config_path is not None and config_path.exists():
        input_config = _load_toml(config_path)
        physics = _extract_physics(input_config)
        field_damping = _extract_field_damping(input_config)
        if physics is None:
            print(f"note: {config_path.name} lacks g/K_p0/K_rho0; skipping growth-rate overlay.")
        else:
            print(f"using config {config_path}")
    else:
        print("note: no config found (input_copy.input); skipping growth-rate overlay. Pass --input to enable.")

    project_kpar0 = bool(input_config.get("runtime", {}).get("project_kpar0", False)) if input_config else False

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)

    # (a) cumulative budget
    ax = axes[0, 0]
    ax.plot(time, energy, color="black", lw=2.0, label=f"measured {quantity}")
    if rhs_total is not None:
        predicted = energy[0] + _cumulative_trapezoid(time, rhs_total)
        ax.plot(time, predicted, color="tab:red", ls="--", lw=2.0, label=r"$E(0)+\int$ rhs_total $dt$")
        ax.plot(time, energy - predicted, color="tab:blue", lw=1.4, alpha=0.8, label="drift (measured - integrated)")
        ax.axhline(0.0, color="0.6", lw=0.8)
    else:
        ax.text(0.5, 0.5, f"{quantity}_rhs_total missing", ha="center", transform=ax.transAxes)
    ax.set_title("Cumulative (integral) budget closure")
    ax.set_xlabel("time"); ax.set_ylabel(quantity); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    # (b) source vs dissipation
    ax = axes[0, 1]
    if stratification is not None:
        ax.plot(time, stratification, color="tab:green", lw=1.8, label="stratification (source)")
    if dissipation is not None:
        ax.plot(time, dissipation, color="tab:purple", lw=1.8, label="dissipation")
    if rhs_total is not None:
        ax.plot(time, rhs_total, color="black", lw=2.0, label="rhs_total = $d_t E$")
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.set_title(r"Source vs dissipation ($d_t E \to 0$ at saturation)")
    ax.set_xlabel("time"); ax.set_ylabel(r"budget terms"); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    # (c) balance ratio
    ax = axes[1, 0]
    if stratification is not None and dissipation is not None:
        mask = np.abs(dissipation) > 1e-30
        ratio = np.full_like(time, np.nan)
        ratio[mask] = -stratification[mask] / dissipation[mask]
        ax.plot(time, ratio, color="tab:orange", lw=1.8, label=r"$-$stratification / dissipation")
        ax.axhline(1.0, color="0.4", lw=1.0, ls="--", label="perfect balance = 1")
        finite = ratio[np.isfinite(ratio)]
        if finite.size:
            lo, hi = np.percentile(finite, [2, 98])
            pad = 0.1 * (hi - lo + 1e-12)
            ax.set_ylim(min(lo - pad, 0.0), hi + pad)
    else:
        ax.text(0.5, 0.5, "stratification/dissipation columns missing", ha="center", transform=ax.transAxes)
    ax.set_title("Source/dissipation balance ratio")
    ax.set_xlabel("time"); ax.set_ylabel("ratio"); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    # (d) growth-rate overlay
    ax = axes[1, 1]
    positive = energy > 0.0
    ax.semilogy(time[positive], energy[positive], color="black", lw=2.0, label=quantity)
    if physics is not None:
        rates = _max_linear_growth_rates(input_config, physics, field_damping)
        print(
            "max linear growth rate gamma (energy ~ exp(2 gamma t)):\n"
            f"  ideal:  kz=0 {rates['ideal_kz0']:.4g}   kz!=0 {rates['ideal_kznz']:.4g}\n"
            + (
                f"  damped: kz=0 {rates['damped_kz0']:.4g}   kz!=0 {rates['damped_kznz']:.4g}"
                if field_damping is not None
                else "  damped: (auto dissipation; nu not static -> skipped)"
            )
        )
        t0, e0 = time[positive][0], energy[positive][0]
        tref = time[positive]
        # The relevant ideal rate respects project_kpar0.
        primary_tag = "kz!=0" if project_kpar0 else "kz=0"
        primary_ideal = rates["ideal_kznz"] if project_kpar0 else rates["ideal_kz0"]
        curves = [(primary_ideal, "tab:red", f"ideal, {primary_tag}")]
        if field_damping is not None:
            primary_damped = rates["damped_kznz"] if project_kpar0 else rates["damped_kz0"]
            curves.append((primary_damped, "tab:blue", f"with dissipation, {primary_tag}"))
        for gamma, color, tag in curves:
            if np.isfinite(gamma):
                ax.semilogy(tref, e0 * np.exp(2.0 * gamma * (tref - t0)), color=color, ls="--", lw=1.6,
                            label=rf"$\exp(2\gamma t)$ {tag}: $\gamma={gamma:.3g}$")
    else:
        ax.text(0.5, 0.05, "config with g/K_p0/K_rho0 needed for overlay", ha="center", transform=ax.transAxes, fontsize=8)
    ax.set_title("Growth-rate check (log E vs linear theory)")
    ax.set_xlabel("time"); ax.set_ylabel(quantity); ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)

    main_path = output_dir / f"{quantity}_debug.png"
    finalize_figure(fig, output_path=main_path, show=args.show, plt=plt)
    saved.append(main_path)

    # second figure: dissipation spectrum
    if args.spectra is not None:
        if field_damping is None:
            print("Skipping dissipation spectrum: needs manual per-field nu_perp (auto/zero dissipation given).")
        else:
            spectra = _read_spectra_csv(Path(args.spectra).expanduser().resolve())
            spec_path = output_dir / "dissipation_spectrum.png"
            _dissipation_spectrum_figure(spectra, field_damping, spec_path, plt=plt, show=args.show)
            if spec_path.exists():
                saved.append(spec_path)

    return saved


if __name__ == "__main__":
    main()
