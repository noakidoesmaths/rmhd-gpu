"""Derived quantities of an inhomogeneous RMHD run, read from its saved outputs.

This is the one place where z±, l_perp, chi_A, eta_turb and the closure
estimates of Squire et al. (arXiv:2607.08036) are defined. Plot scripts import
it instead of re-deriving them, so a definition cannot differ between plots.
Only scalar_diagnostics.csv and input_copy.input are read; full-field snapshots
are not needed. Every quantity is one value per saved time, in code units.

Elsasser convention, as in the paper and the solver:

    z^± = delta u_perp ∓ delta B_perp/sqrt(4 pi rho_0)

z+ is the wave the initial condition launches and z- the wave it reflects into.
Every scale is taken from z+:

    z±_rms    = 2 sqrt(W±)                          the solver saves W± = <|z±|^2>/4
    l_perp    = 1 / k_perp_plus                     k_perp_plus = int k E+ dk / int E+ dk
    alignment = rms(z+_x) / z+_rms                  saved as w_plus_align; 1/sqrt(2) if isotropic
    chi_A     = z+_rms k_perp_plus / (vA k_prl_plus)
    eta_turb  = z+_rms l_perp / 4                   1/2 from delta u_perp ~ z+/2, 1/2 from isotropy (Eq. 48)
    V_rho_x   = <(delta rho/rho_0) delta u_x>       saved column (Eq. 79)
    eta_meas  = V_rho_x / F_rho                     the measured diffusivity; NaN if F_rho = 0

    Eq. (47): rms(f)          ~ |F_f| l_perp alignment        f = drho, du_par, db_par
    Eq. (73): z-_rms / z+_rms ~ l_perp |g| rms(drho) / z+_rms^2
    Eq. (80): V_rho_x         ~ eta_turb F_rho

F_f is the background drive of each field (Eq. 46), a run constant built from
the [physics] input by compressive_channels.background_drives.

Older CSVs: the k columns were called w_plus_kperp, w_minus_kperp, w_plus_kprl and
w_minus_kprl before 2026-10-08, and the time column was once called t; both are
renamed on reading. A column a CSV lacks reads as NaN, so whatever depends on it
is NaN too and simply does not plot. Before V_rho_x existed it is taken from
acr_g / g, the same correlator (acr_g = g <drho u_x>), which fails only for g = 0.

In Spyder:
    from vis.run_quantities import load_run, time_window
    run = load_run("examples/outputs_nokia_rmhd_52_rho")
    window = time_window(run.times, tmin=2, tmax=10)
    run.chi_a[window].mean()
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

import numpy as np

from rmhdgpu.diagnostics.compressive_channels import background_drives, slaved_field_units
from rmhdgpu.equations.rmhd_by_nokia_rho import derived_parameters


FIELDS = ("drho", "du_par", "db_par")

# Current column name -> name used in CSVs written before 2026-10-08.
OLD_COLUMN_NAMES = {
    "time": "t",
    "k_perp_plus": "w_plus_kperp",
    "k_perp_minus": "w_minus_kperp",
    "k_prl_plus": "w_plus_kprl",
    "k_prl_minus": "w_minus_kprl",
}


# ---------------------------------------------------------------------------
# Reading the saved files
# ---------------------------------------------------------------------------

def resolve_run_dirs(patterns):
    """Run directories from paths or quoted wildcards, in order, without duplicates."""
    run_dirs = []
    for pattern in patterns:
        candidate = Path(pattern).expanduser()
        matches = [candidate] if candidate.is_dir() else sorted(Path(path) for path in glob(str(candidate)))
        if not matches or any(not path.is_dir() for path in matches):
            raise ValueError(f"No matching run directories: {pattern}")
        for path in matches:
            if path.resolve() not in run_dirs:
                run_dirs.append(path.resolve())
    return run_dirs


def load_parameters(run_dir):
    """Physics scalars from the run's saved input, via the solver's own derived_parameters."""
    input_path = Path(run_dir) / "input_copy.input"
    if not input_path.is_file():
        raise ValueError(f"Missing {input_path}; the estimates need its [physics] block.")
    with input_path.open("rb") as handle:
        physics = tomllib.load(handle).get("physics", {})
    missing = [key for key in ("vA", "cs2_over_vA2", "g", "K_p0", "K_rho0") if key not in physics]
    if missing:
        raise ValueError(f"{input_path} [physics] is missing {missing}.")
    return derived_parameters(physics)


def read_columns(csv_path):
    """Every column of a scalar CSV as a float array, with old column names renamed."""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise ValueError(f"Missing {csv_path}.")
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{csv_path} contains no data rows.")
    columns = {name: np.array([float(row[name]) for row in rows]) for name in rows[0]}
    for name, old_name in OLD_COLUMN_NAMES.items():
        if name not in columns and old_name in columns:
            columns[name] = columns.pop(old_name)
    return columns


def column(columns, name):
    """One column, or all NaN if the CSV predates it (NaN then propagates through every formula)."""
    if name in columns:
        return columns[name]
    return np.full(len(columns["time"]), np.nan)


def safe_divide(numerator, denominator):
    """numerator / denominator, with NaN wherever the denominator is not positive."""
    numerator, denominator = np.asarray(numerator, float), np.asarray(denominator, float)
    out = np.full(np.broadcast(numerator, denominator).shape, np.nan)
    return np.divide(numerator, denominator, out=out, where=denominator > 0.0)


# ---------------------------------------------------------------------------
# Formulas that also apply to a bare CSV (plot_dcf.py uses them without an input file)
# ---------------------------------------------------------------------------

def elsasser_rms(w):
    """z_rms = 2 sqrt(W), since the solver saves W = <|z|^2>/4."""
    return 2.0 * np.sqrt(w)


def turbulence_strength(z_rms, k_perp, k_prl, v_a):
    """chi_A = z_rms k_perp / (vA k_par): nonlinear rate over Alfven rate. NaN where k_par = 0."""
    return safe_divide(z_rms * k_perp, v_a * k_prl)


# ---------------------------------------------------------------------------
# A whole run
# ---------------------------------------------------------------------------

@dataclass
class Run:
    """One run's derived quantities, one value per saved time, in code units. NaN where the CSV cannot give them."""

    run_dir: Path
    p: Any  # derived_parameters of the run
    columns: dict[str, np.ndarray]  # every saved column, old names renamed
    times: np.ndarray
    z_plus: np.ndarray  # z+_rms
    z_minus: np.ndarray  # z-_rms
    u_perp: np.ndarray  # rms(delta u_perp)
    b_perp: np.ndarray  # rms(delta B_perp/sqrt(4 pi rho_0)), Alfven units
    k_perp: np.ndarray  # k_perp_plus
    k_prl: np.ndarray  # k_prl_plus
    l_perp: np.ndarray  # 1 / k_perp_plus
    alignment: np.ndarray  # rms(z+_x) / z+_rms
    chi_a: np.ndarray  # of z+
    eta_turb: np.ndarray  # z+_rms l_perp / 4
    v_rho_x: np.ndarray  # <(drho/rho_0) delta u_x>
    eta_meas: np.ndarray  # V_rho_x / F_rho, the measured diffusivity
    drives: dict[str, float]  # F_f of each compressive field (Eq. 46)
    measured: dict[str, np.ndarray]  # rms of each requested field, in Eq. (47) units
    predicted: dict[str, np.ndarray]  # Eq. (47) estimate of each requested field
    z_ratio: np.ndarray  # z-_rms / z+_rms
    z_ratio_predicted: np.ndarray  # Eq. (73)


def load_run(run_dir, fields=FIELDS):
    """Read one run directory and compute every derived quantity.

    `fields` are the compressive fields whose `<name>_rms` columns must exist;
    `measured` and `predicted` hold those fields only.
    """
    run_dir = Path(run_dir)
    p = load_parameters(run_dir)
    csv_path = run_dir / "scalar_diagnostics.csv"
    columns = read_columns(csv_path)
    missing = [name for name in ["time"] + [f"{name}_rms" for name in fields] if name not in columns]
    if missing:
        raise ValueError(f"{csv_path} is missing {missing}; is this an inhomogeneous_rmhd_rho run?")

    z_plus = elsasser_rms(column(columns, "w_plus"))
    z_minus = elsasser_rms(column(columns, "w_minus"))
    if z_minus[0] > z_plus[0]:
        print(f"{run_dir.name}: W- > W+ at the first saved time. The estimates assume "
              "z+ = u_perp - b_perp is the driving wave, so they do not describe this run.")

    k_perp = column(columns, "k_perp_plus")
    k_prl = column(columns, "k_prl_plus")
    l_perp = safe_divide(1.0, k_perp)
    alignment = column(columns, "w_plus_align")

    # Eq. (47): rms(f) ~ |F_f| l_perp alignment. A fluid element displaced by
    # ~ (l_perp/z+) z+_x along x picks up delta f ~ F_f * displacement. An empty
    # z+ is saved with alignment 0, so its estimate is 0 although l_perp is undefined.
    drives, units = background_drives(p), slaved_field_units(p)
    projected_length = np.where(alignment == 0.0, 0.0, l_perp * alignment)
    measured = {name: columns[f"{name}_rms"] / units[name] for name in fields}
    predicted = {name: abs(drives[name]) * projected_length for name in fields}

    # Eq. (73): buoyancy |g| drho acting for one nonlinear time l_perp/z+ drives
    # z- ~ (l_perp/z+) |g| rms(drho); divide by z+ for a dimensionless ratio.
    z_ratio_predicted = safe_divide(l_perp * abs(p.g) * column(columns, "drho_rms"), z_plus**2)

    # Eqs. (79)-(80). Only the x component of V_rho is non-trivial, since F_rho points along x.
    if "V_rho_x" in columns or p.g == 0.0:
        v_rho_x = column(columns, "V_rho_x")
    else:
        v_rho_x = column(columns, "acr_g") / p.g
    # F_rho is a signed run constant, so safe_divide (positive denominators only) does not apply.
    f_rho = drives["drho"]
    eta_meas = v_rho_x / f_rho if f_rho != 0.0 else np.full_like(v_rho_x, np.nan)

    return Run(
        run_dir=run_dir,
        p=p,
        columns=columns,
        times=columns["time"],
        z_plus=z_plus,
        z_minus=z_minus,
        u_perp=column(columns, "u_perp_rms"),
        b_perp=column(columns, "b_perp_rms"),
        k_perp=k_perp,
        k_prl=k_prl,
        l_perp=l_perp,
        alignment=alignment,
        chi_a=turbulence_strength(z_plus, k_perp, k_prl, p.vA),
        eta_turb=z_plus * l_perp / 4.0,
        v_rho_x=v_rho_x,
        eta_meas=eta_meas,
        drives=drives,
        measured=measured,
        predicted=predicted,
        z_ratio=safe_divide(z_minus, z_plus),
        z_ratio_predicted=z_ratio_predicted,
    )


def time_window(times, *, tmin=None, tmax=None, tail_fraction=0.4):
    """Boolean mask of the saved times to average over.

    By default the last `tail_fraction` of the run's time span; `tmin` and
    `tmax` set the start and end instead. The window is chosen by time, not by
    number of rows, so unevenly spaced output does not shift it. A window with
    fewer than two saved times is an error.
    """
    times = np.asarray(times, dtype=float)
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("times must be finite and strictly increasing.")
    if not 0 < tail_fraction <= 1:
        raise ValueError("tail_fraction must be between 0 (exclusive) and 1.")
    for name, value in (("tmin", tmin), ("tmax", tmax)):
        if value is not None and not np.isfinite(value):
            raise ValueError(f"{name} must be finite.")
    start = tmin if tmin is not None else times[-1] - tail_fraction * (times[-1] - times[0])
    end = tmax if tmax is not None else times[-1]
    window = (times >= start) & (times <= end)
    if window.sum() < 2:
        raise ValueError(f"fewer than two saved times in the window [{start:g}, {end:g}].")
    return window
