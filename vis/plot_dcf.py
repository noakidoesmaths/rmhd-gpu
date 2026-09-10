"""Compare measured direct-compressive-feedback heating against the closure.

Reads only `scalar_diagnostics.csv`, which carries both sides of the comparison
for either inhomogeneous equation set (`inhomogeneous_rmhd_rho` and
`inhomogeneous_rmhd_s` write identical channel columns).

Measured (exact; no closure is involved, since these equation sets are the
straight-field, no-mean-flow case of Squire et al. arXiv:2607.08036 Eq. 18):

    Q_meas = q_dcf = -(d_t W+)|_buoyancy

Predicted (their Eq. 70/71, the `rho` channel; the `u` and `B` channels are
both proportional to field-line curvature and so vanish here):

    Q_pred = W+ vA K^damp_DCF,rho,     K^damp_DCF,rho = -chi_A^-1 l_par F_rho.g/vA^2

Using `N^2 = -g F_rho` (exactly the modules' `N_sq`) and
`chi_A = z+ l_par / (vA l_perp)`, the parallel scale and turbulence-strength
parameter cancel identically:

    Q_pred = W+ N^2 / omega_nl = 0.5 N^2 l_perp sqrt(W+)

so only `W+`, `l_perp` and the run constant `N^2` are needed.

Note what this cancellation means in practice. `g` and `F_rho` only ever enter
through the product `-g F_rho = N^2`, which is already a CSV column, so there
is nothing to gain from supplying them separately. And substituting a `chi_A`
*measured from the run* reproduces this reduced form exactly rather than giving
an independent check, because `chi_A^-1 l_par = vA l_perp / z` whenever
`chi_A` is built from the same `<k_perp>`, `<k_par>` and `W` as everything else.
The only genuinely free knob is therefore an *assumed* `chi_A`: pass
`--chi-a 1` (critical balance, what the closure posits) to overlay

    Q_explicit = W+ vA * chi_A^-1 l_par N^2 / vA^2 = W+ N^2 l_par / (chi_A vA)

with `l_par = 1 / <k_par>` taken from the run. That curve uses `<k_par>` where
the reduced form uses `<k_perp>`, and the ratio between the two is exactly the
measured `chi_A`, which is reported in the title. `l_perp` is the remaining
definitional choice, so the mean ratio `Q_meas/Q_pred` is reported as a fitted
O(1) prefactor rather than claimed as agreement. Pass `--l-perp` to overlay a
prediction using a fixed outer scale (for example the initial-condition band),
and `--tmin` to exclude the startup transient (the compressive fields begin at
zero, so the measured rate necessarily starts at zero while the slaved
prediction does not).

Which branch is the pump? DCF is the buoyancy work on the *dominant* Elsasser
field, and nothing distinguishes `z+` from `z-` except the sign of the guide
field, so `--branch` selects it (default: whichever carries more energy). With
the solver convention `z^± = u_perp -+ b_perp/sqrt(4 pi rho0)` the potential of
`z^±` is `phi -+ psi`, and `random_spectrum_one_wave` sets
`omega = -lap_perp(psi)`, hence `phi = -psi` and `z- = 0`: those runs are
perfectly imbalanced into `z+`, so the DCF rate lives in the `q_dcf` / `w_plus`
columns.

Usage:
    python vis/plot_dcf.py outputs/scalar_diagnostics.csv
    python vis/plot_dcf.py outputs/scalar_diagnostics.csv --tmin 4.0 --l-perp 0.5
    python vis/plot_dcf.py outputs/scalar_diagnostics.csv --branch minus
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from vis._matplotlib import finalize_figure, import_pyplot


REQUIRED_COLUMNS = ("w_plus", "w_minus", "w_plus_kperp", "q_dcf", "N_sq")

# Per-branch column names. `q_dcf` is the buoyancy work on `W+` and
# `q_ccr_source` the work on `W-`; whichever branch is the pump supplies DCF.
BRANCH_COLUMNS = {
    "plus": {
        "energy": "w_plus",
        "kperp": "w_plus_kperp",
        "kprl": "w_plus_kprl",
        "work": "q_dcf",
        "dissipation": "w_plus_rhs_dissipation",
        "label": "z^+",
    },
    "minus": {
        "energy": "w_minus",
        "kperp": "w_minus_kperp",
        "kprl": "w_minus_kprl",
        "work": "q_ccr_source",
        "dissipation": "w_minus_rhs_dissipation",
        "label": "z^-",
    },
}


def _select_branch(choice: str, columns: dict[str, np.ndarray]) -> str:
    """Return the pump branch, defaulting to whichever carries more energy."""

    if choice != "auto":
        return choice
    return "plus" if np.mean(columns["w_plus"]) >= np.mean(columns["w_minus"]) else "minus"


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


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    ratio = np.full_like(numerator, np.nan, dtype=np.float64)
    usable = np.abs(denominator) > 0.0
    ratio[usable] = numerator[usable] / denominator[usable]
    return ratio


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_path", help="Path to scalar_diagnostics.csv.")
    parser.add_argument(
        "--branch",
        choices=("auto", "plus", "minus"),
        default="auto",
        help=(
            "Which Elsasser field is the pump wave. Default `auto` picks whichever carries more "
            "energy, which is `plus` for `random_spectrum_one_wave` initial conditions."
        ),
    )
    parser.add_argument(
        "--l-perp",
        type=float,
        default=None,
        help=(
            "Fixed outer scale for a second, reference prediction. Useful for showing how much "
            "the answer depends on the l_perp definition, e.g. the initial-condition band."
        ),
    )
    parser.add_argument(
        "--chi-a",
        type=float,
        default=None,
        help=(
            "Overlay the explicit closure form `W N^2 l_par / (chi_A vA)` using this ASSUMED "
            "turbulence-strength parameter and `l_par = 1/<k_par>` measured from the run. Pass 1 "
            "for critical balance. A chi_A measured from the run would reproduce the reduced form "
            "identically, so only an assumed value is informative."
        ),
    )
    parser.add_argument(
        "--tmin",
        type=float,
        default=None,
        help="Ignore times below this when reporting the mean prefactor. Use it to exclude startup transients.",
    )
    parser.add_argument("--output", default=None, help="Output image path. Defaults next to the CSV file.")
    parser.add_argument("--show", action="store_true", help="Show the figure interactively after saving.")
    return parser


def main(argv: list[str] | None = None) -> Path:
    args = build_parser().parse_args(argv)
    plt = import_pyplot(show=args.show)
    csv_path = Path(args.csv_path).expanduser().resolve()
    fieldnames, columns = _read_scalar_csv(csv_path)

    missing = [name for name in REQUIRED_COLUMNS if name not in columns]
    if missing:
        raise SystemExit(
            f"{csv_path} is missing {missing}. These columns are written by the inhomogeneous "
            "equation sets; re-run with `inhomogeneous_rmhd_rho` or `inhomogeneous_rmhd_s`."
        )

    time = columns["time" if "time" in columns else "t"]
    branch = _select_branch(args.branch, columns)
    names = BRANCH_COLUMNS[branch]

    w_pump = columns[names["energy"]]
    kperp_mean = columns[names["kperp"]]
    n_sq = float(columns["N_sq"][0])

    # z = 2 sqrt(W) with rho0 = 1, so omega_nl = z/l_perp = 2 sqrt(W) <k_perp>.
    z_pump = 2.0 * np.sqrt(np.maximum(w_pump, 0.0))
    omega_nl = z_pump * kperp_mean

    q_measured = columns[names["work"]]
    q_predicted = _safe_ratio(w_pump * n_sq, omega_nl)

    # chi_A = z <k_perp> / (vA <k_par>) = omega_nl / omega_A. Measured only so it
    # can be reported: it is exactly the ratio between the reduced prediction and
    # the explicit `l_par`-based one, which is why substituting it back in is
    # circular. `vA` is echoed into every row as a run constant.
    v_a = float(columns["vA"][0]) if "vA" in columns else 1.0
    kprl_mean = columns.get(names["kprl"])
    chi_a_measured = None if kprl_mean is None else _safe_ratio(omega_nl, v_a * kprl_mean)

    q_explicit = None
    if args.chi_a is not None:
        if kprl_mean is None:
            raise SystemExit(
                f"--chi-a needs the {names['kprl']!r} column, which {csv_path} does not have. "
                "Re-run the simulation to write the parallel-wavenumber diagnostics."
            )
        if args.chi_a == 0.0:
            raise SystemExit("--chi-a must be nonzero.")
        # Q = W vA K^damp = W N^2 l_par / (chi_A vA), with l_par = 1/<k_par>.
        q_explicit = _safe_ratio(w_pump * n_sq, args.chi_a * v_a * kprl_mean)

    prefactor = _safe_ratio(q_measured, q_predicted)
    window = np.ones_like(time, dtype=bool) if args.tmin is None else time >= args.tmin
    usable = window & np.isfinite(prefactor)
    mean_prefactor = float(np.mean(prefactor[usable])) if usable.any() else float("nan")

    output_path = (
        csv_path.with_name("dcf_measured_vs_predicted.png")
        if args.output is None
        else Path(args.output).expanduser().resolve()
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(8.5, 5.0), constrained_layout=True)

    ax.plot(time, q_measured, lw=2.0, color="black", label=r"measured $Q_{\rm DCF}$")
    ax.plot(
        time,
        q_predicted,
        lw=1.8,
        ls="--",
        color="tab:red",
        label=r"predicted $W v_A K^{\rm damp}_{{\rm DCF},\rho} = W N^2/\omega_{\rm nl}$",
    )
    if q_explicit is not None:
        ax.plot(
            time,
            q_explicit,
            lw=1.8,
            ls="--",
            color="tab:green",
            label=(
                rf"explicit $W N^2 l_\parallel/(\chi_A v_A)$, assumed $\chi_A={args.chi_a:g}$"
            ),
        )
    if args.l_perp is not None:
        reference = 0.5 * n_sq * args.l_perp * np.sqrt(np.maximum(w_pump, 0.0))
        ax.plot(
            time,
            reference,
            lw=1.5,
            ls=":",
            color="tab:orange",
            label=rf"predicted, fixed $l_\perp={args.l_perp:g}$",
        )
    if names["dissipation"] in columns:
        ax.plot(
            time,
            -columns[names["dissipation"]],
            lw=1.4,
            ls="-.",
            color="tab:blue",
            label=r"$\epsilon$ (cascade, for scale)",
        )
    ax.axhline(0.0, color="0.5", lw=1.0, alpha=0.6)
    if args.tmin is not None:
        ax.axvline(args.tmin, color="0.4", lw=1.0, ls=":")
    ax.set_xlabel("time")
    ax.set_ylabel(r"heating rate")
    title = (
        f"DCF: measured vs slaved closure   (pump ${names['label']}$, "
        f"$N^2$ = {n_sq:.4g}, {'stable' if n_sq > 0 else 'unstable'} stratification)"
    )
    if np.isfinite(mean_prefactor):
        window = "" if args.tmin is None else rf" over $t \geq {args.tmin:g}$"
        title += f"\nmean $Q_{{\\rm meas}}/Q_{{\\rm pred}}$ = {mean_prefactor:.3g}{window}"
    if chi_a_measured is not None:
        usable_chi = usable & np.isfinite(chi_a_measured)
        if usable_chi.any():
            title += rf",  measured $\chi_A$ = {float(np.mean(chi_a_measured[usable_chi])):.3g}"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    finalize_figure(fig, output_path=output_path, show=args.show, plt=plt)
    return output_path


if __name__ == "__main__":
    main()
