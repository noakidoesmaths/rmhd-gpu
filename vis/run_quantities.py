"""Derived quantities and closure estimates of an inhomogeneous RMHD run, from its saved outputs.

This is the one place where the measured quantities and the closure estimates
of Squire et al. (arXiv:2607.08036) are defined. The plot scripts in vis/ only
draw what `load_run` returns, so a definition cannot differ between plots.
Only scalar_diagnostics.csv and input_copy.input are read. Every quantity is
one value per saved time, in code units.

Elsasser convention, as in the paper and the solver:

    z^± = delta u_perp ∓ delta B_perp/sqrt(4 pi rho_0)

z+ is the launched (or forced) wave and z- the wave it reflects into. Every
scale is taken from z+:

    z±_rms    = 2 sqrt(W±)                  the solver saves W± = <|z±|^2>/4
    z+_x      = rms(z+_x) = w_plus_align z+_rms,   the part of z+ along the gradient x
    l_perp    = 1 / k_perp_plus
    chi_A     = z+_rms k_perp_plus / (vA k_prl_plus)

The closure (slaving) estimates. A fluid element moves a distance
xi_x ~ (l_perp/z+_rms) z+_x along the background gradient in one eddy time,
with u_x ~ z+_x/2, and each compressive field picks up f ~ F_f xi_x. The paper
assumes isotropy, z+_x = z+_rms/sqrt(2) (Eq. 48); here the measured z+_x is
used instead, because stratification (g != 0) suppresses motion along x.

    measured                          estimate
    rms(f), f = drho, du_par, db_par  |F_f| l_perp z+_x / z+_rms                    Eq. (47)
    z-_rms / z+_rms                   l_perp |g| rms(drho) / z+_rms^2               Eq. (73)
    V_rho_x = <drho u_x>              eta_turb F_rho                                Eq. (80)
    V_psi_x                           eta_turb F_psi,  F_psi = F_u + vS^2/vA^2 F_B  Eq. (85)
    q_dcf  (DCF heating of z+)        -g F_rho eta_turb                             Eq. (69)
    q_ccr  (CCR heating of z+)        |g F_rho| l_perp z+_x / 4                     Eq. (75)

with the turbulent diffusivity eta_turb = <xi_x u_x> = l_perp z+_x^2 / (2 z+_rms).
The ACR channels are the fluxes times constants, acr_g = g V_rho_x and
acr_B = -vA^2 K_b0 V_psi_x, so the V estimates are also their estimates.
F_f are the background drives (Eq. 46) from compressive_channels.

The measured CCR heating is the W+ dissipation rate, -w_plus_rhs_dissipation:
z-.grad z+ moves no net W+, it only carries it to small scales where the
hyperviscosity removes it. That reading holds in a steady state with z+-only
forcing; it is not the saved q_ccr_source column (the W- buoyancy source).

Older CSVs: columns renamed on 2026-10-08 are renamed on reading. A column a
CSV lacks reads as NaN, and whatever depends on it is NaN and does not plot;
in particular a CSV without w_plus_align has no estimates. Before V_rho_x and
V_psi_x were saved they are taken from acr_g / g and -acr_B / (vA^2 K_b0).

In Spyder:
    from vis.run_quantities import load_run, time_window
    run = load_run("examples/outputs_nokia_rmhd_52_rho")
    window = time_window(run.times, tmin=2, tmax=10)
    run.q_dcf[window].mean(), run.q_dcf_predicted[window].mean()
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

from rmhdgpu.diagnostics.compressive_channels import background_drives, psi_drive, slaved_field_units
from rmhdgpu.equations.rmhd_by_nokia_rho import derived_parameters


FIELDS = ("drho", "du_par", "db_par")

# Plot labels and colours of the compressive fields, shared by every vis script.
FIELD_LABELS = {
    "drho": r"\delta\rho/\rho_0",
    "du_par": r"\delta u_\parallel/v_A",
    "db_par": r"(v_A^2/v_S^2)\,\delta B_\parallel/B_0",
}
FIELD_COLORS = {"drho": "tab:blue", "du_par": "tab:green", "db_par": "tab:orange"}

# Current column name -> name used in CSVs written before 2026-10-08.
OLD_COLUMN_NAMES = {
    "time": "t",
    "k_perp_plus": "w_plus_kperp",
    "k_prl_plus": "w_plus_kprl",
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
# Formulas
# ---------------------------------------------------------------------------

def elsasser_rms(w):
    """z_rms = 2 sqrt(W), since the solver saves W = <|z|^2>/4."""
    return 2.0 * np.sqrt(w)


def turbulence_strength(z_rms, k_perp, k_prl, v_a):
    """chi_A = z_rms k_perp / (vA k_par): nonlinear rate over Alfven rate. NaN where k_par = 0."""
    return safe_divide(z_rms * k_perp, v_a * k_prl)


def turbulent_diffusivity(z_plus, z_plus_x, l_perp):
    """eta_turb = <xi_x u_x> = l_perp z+_x^2 / (2 z+_rms).

    xi_x ~ (l_perp/z+_rms) z+_x and u_x ~ z+_x/2. Isotropic z+ (z+_x^2 = z+^2/2)
    gives the paper's z+_rms l_perp / 4.
    """
    return safe_divide(l_perp * z_plus_x**2, 2.0 * z_plus)


def dcf_heating_estimate(g, f_rho, eta_turb):
    """Eq. (69): Q_DCF = -g V_rho_x ~ -g F_rho eta_turb. Positive (drains z+) when stable, -g F_rho > 0."""
    return -g * f_rho * eta_turb


def ccr_heating_estimate(g, f_rho, z_plus_x, l_perp):
    """Eq. (75) with the slaved z-: Q_CCR ~ W+ z-_rms / l_perp ~ |g F_rho| l_perp z+_x / 4.

    W+ = z+^2/4, the slaved z- ~ (l_perp/z+) |g| rms(drho) (Eq. 73) and the
    slaved rms(drho) ~ |F_rho| l_perp z+_x / z+ (Eq. 47). For isotropic z+ this
    is 1/sqrt(2) times the paper's (z+_rms l_perp / 2) |g F_rho / 2|, which
    takes rms(drho) ~ |F_rho| l_perp without the projection onto x.
    """
    return abs(g * f_rho) * l_perp * z_plus_x / 4.0


# ---------------------------------------------------------------------------
# A whole run
# ---------------------------------------------------------------------------

@dataclass
class Run:
    """One run's measured quantities and closure estimates, one value per saved time. NaN where unavailable."""

    run_dir: Path
    p: Any  # derived_parameters of the run
    columns: dict[str, np.ndarray]  # every saved column, old names renamed
    times: np.ndarray
    # Scales of the turbulence.
    z_plus: np.ndarray  # z+_rms
    z_minus: np.ndarray  # z-_rms
    z_plus_x: np.ndarray  # rms(z+_x)
    u_perp: np.ndarray  # rms(delta u_perp)
    b_perp: np.ndarray  # rms(delta B_perp/sqrt(4 pi rho_0)), Alfven units
    k_perp: np.ndarray  # k_perp_plus
    k_prl: np.ndarray  # k_prl_plus
    l_perp: np.ndarray  # 1 / k_perp_plus
    chi_a: np.ndarray  # of z+
    eta_turb: np.ndarray  # l_perp z+_x^2 / (2 z+_rms)
    drives: dict[str, float]  # F_f of each compressive field (Eq. 46)
    f_psi: float  # F_u + (vS^2/vA^2) F_B, the drive of V_psi (Eq. 85)
    # Measured quantities and their estimates, in pairs.
    measured: dict[str, np.ndarray]  # rms of each requested field, in Eq. (47) units
    predicted: dict[str, np.ndarray]  # Eq. (47)
    z_ratio: np.ndarray  # z-_rms / z+_rms
    z_ratio_predicted: np.ndarray  # Eq. (73)
    v_rho_x: np.ndarray  # <drho u_x>
    v_rho_x_predicted: np.ndarray  # Eq. (80)
    v_psi_x: np.ndarray  # <db_par u_x> - <du_par b_x>/vA
    v_psi_x_predicted: np.ndarray  # Eq. (85)
    q_dcf: np.ndarray  # -(d_t W+)|_buoyancy
    q_dcf_predicted: np.ndarray  # Eq. (69)
    q_ccr: np.ndarray  # -w_plus_rhs_dissipation
    q_ccr_predicted: np.ndarray  # Eq. (75) with the slaved z-
    q_ccr_from_z_minus: np.ndarray  # Eq. (75) with the measured z-: W+ z-_rms / l_perp

    @property
    def alignment(self):
        """rms(z+_x) / z+_rms, the saved w_plus_align: 1/sqrt(2) for isotropic z+, 0 for an empty z+."""
        return column(self.columns, "w_plus_align")


def load_run(run_dir, fields=FIELDS):
    """Read one run directory and compute every measured quantity and estimate.

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
    z_plus_x = column(columns, "w_plus_align") * z_plus  # rms(z+_x), from the saved rms(z+_x)/z+_rms
    k_perp = column(columns, "k_perp_plus")
    l_perp = safe_divide(1.0, k_perp)
    eta_turb = turbulent_diffusivity(z_plus, z_plus_x, l_perp)
    drives, f_psi = background_drives(p), psi_drive(p)
    f_rho = drives["drho"]

    # Eq. (47): rms(f) ~ |F_f| xi_x with xi_x ~ l_perp z+_x / z+_rms. z+_x / z+_rms is the
    # saved w_plus_align itself, so this needs no w_plus column. An empty z+ (saved with
    # w_plus_align = 0) gives 0, although l_perp is then undefined.
    units = slaved_field_units(p)
    alignment = column(columns, "w_plus_align")
    displacement = np.where(alignment == 0.0, 0.0, l_perp * alignment)
    measured = {name: columns[f"{name}_rms"] / units[name] for name in fields}
    predicted = {name: abs(drives[name]) * displacement for name in fields}

    # Eqs. (79)-(80) and (85). Older CSVs: V_rho_x = acr_g / g and V_psi_x = -acr_B / (vA^2 K_b0).
    if "V_rho_x" in columns or p.g == 0.0:
        v_rho_x = column(columns, "V_rho_x")
    else:
        v_rho_x = column(columns, "acr_g") / p.g
    if "V_psi_x" in columns or p.K_b0 == 0.0:
        v_psi_x = column(columns, "V_psi_x")
    else:
        v_psi_x = -column(columns, "acr_B") / (p.vA**2 * p.K_b0)

    return Run(
        run_dir=run_dir,
        p=p,
        columns=columns,
        times=columns["time"],
        z_plus=z_plus,
        z_minus=z_minus,
        z_plus_x=z_plus_x,
        u_perp=column(columns, "u_perp_rms"),
        b_perp=column(columns, "b_perp_rms"),
        k_perp=k_perp,
        k_prl=column(columns, "k_prl_plus"),
        l_perp=l_perp,
        chi_a=turbulence_strength(z_plus, k_perp, column(columns, "k_prl_plus"), p.vA),
        eta_turb=eta_turb,
        drives=drives,
        f_psi=f_psi,
        measured=measured,
        predicted=predicted,
        z_ratio=safe_divide(z_minus, z_plus),
        # Eq. (73): buoyancy |g| drho acting for one eddy time l_perp/z+ drives z- ~ (l_perp/z+) |g| rms(drho).
        z_ratio_predicted=safe_divide(l_perp * abs(p.g) * column(columns, "drho_rms"), z_plus**2),
        v_rho_x=v_rho_x,
        v_rho_x_predicted=eta_turb * f_rho,
        v_psi_x=v_psi_x,
        v_psi_x_predicted=eta_turb * f_psi,
 
        # DCF
        q_dcf=column(columns, "q_dcf"),
        q_dcf_predicted= -(p.g * f_rho * l_perp * z_plus_x**2)/(2.0 * z_plus),
        # An alternative line incase there's an NaN data
        # q_dcf_predicted= -p.g * f_rho * safe_divide(l_perp * z_plus_x**2, 2.0 * z_plus),
        
        # CCR
        q_ccr=-column(columns, "w_plus_rhs_dissipation"),
        # q_ccr_estimate
        # abs(g * f_rho) * l_perp * z_plus_x / 4.0
        q_ccr_predicted=abs(p.g * f_rho) * l_perp * z_plus_x / 4.0,
        q_ccr_from_z_minus=column(columns, "w_plus") * z_minus * k_perp,
    )


# ---------------------------------------------------------------------------
# Helpers shared by the plot scripts
# ---------------------------------------------------------------------------

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


def run_window(run, *, tmin=None, tmax=None, tail_fraction=0.4):
    """time_window for one run, with the run's name in the error message."""
    try:
        return time_window(run.times, tmin=tmin, tmax=tmax, tail_fraction=tail_fraction)
    except ValueError as error:
        raise ValueError(f"{run.run_dir.name}: {error}") from None


def describe_window(tmin=None, tmax=None, tail_fraction=0.4):
    """The averaging window as requested on the command line, for a figure title."""
    if tmin is None and tmax is None:
        return f"last {100 * tail_fraction:g}% of each run"
    if tmin is None:
        return f"last {100 * tail_fraction:g}% of each run, up to t = {tmax:g}"
    end = f"{tmax:g}" if tmax is not None else "end"
    return f"t in [{tmin:g}, {end}]"


def default_output_dir(run_dirs):
    """Where a plot of these runs goes: inside the run for one run, else the folder holding them all."""
    if len(run_dirs) == 1:
        return Path(run_dirs[0])
    parents = {Path(path).parent for path in run_dirs}
    return parents.pop() if len(parents) == 1 else Path.cwd()


def write_summary_csv(rows, path):
    """Save a list of dicts (one per run or point) as a CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}")
