"""How big each term of the local-in-z energy budgets is: the rms over z.

At each z every energy has its own budget (see `rmhdgpu/diagnostics/z_profiles.py`):

    d_t W+(z)      = -vA d_z W+(z) + buoyancy+(z) + dissipation+(z)
    d_t W-(z)      = +vA d_z W-(z) + buoyancy-(z) + dissipation-(z)
    d_t W_par^u(z) = drive_KB(z) + exchange_lin(z) + exchange_nl(z) + dissipation(z)
    d_t W_par^B(z) = drive_KBKp(z) + exchange_lin(z) + exchange_nl(z) + dissipation(z)
    d_t W^p(z)     = the W_par^B terms / chi
    d_t W_s(z)     = drive_Ks(z) + dissipation(z)

The transport terms (vA d_z W±) average to zero over z, so their box mean says
nothing about how big they are. Their rms over z does:

    rms  = sqrt( < f(z)^2 >_{z, t} )      how big the term is, mean and z-structure together
    mean = < f(z) >_{z, t}                the box-averaged term (0 for transport)

One panel per energy: the rms over z of each term against time. The legend
gives each term's share of the summed rms of that energy's terms over the
shaded window. The window means are written to a CSV next to the figure:
the rms and mean of each term, plus the saved d_t W(z) and the local
residual d_t W(z) - sum of the terms (not drawn; the residual is only the
small part of the nonlinear terms that the dealiasing truncation leaves at
each z, about 1% of d_t W in a 32^3 test run).

Runs made before 2026-11-21 have no `<energy>_dt` profiles. For those the
script falls back to four budgets (W+, W-, the three slow-mode energies
together, W_s) without d_t W or residual.

    python vis/plot_z_rms_terms.py RUN_DIRECTORY
    python vis/plot_z_rms_terms.py RUN_DIRECTORY --tmin 5 --show
"""

import argparse
import csv
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from rmhdgpu.diagnostics.z_profiles import Z_PROFILE_LOCAL_BUDGETS
from vis._matplotlib import finalize_figure, import_pyplot
from vis.run_quantities import RESERVOIR_LABELS, TERM_COLORS, TERM_LABELS

TITLES = {**{name: label.replace("$", "") for name, label in RESERVOIR_LABELS.items()},
          "slow": r"W_\parallel^u + W_\parallel^B + W^p"}
LABELS = {**TERM_LABELS, "transport": r"transport $\mp v_A\partial_z W$"}
COLORS = {**TERM_COLORS, "transport": "black"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="Run directory, or its z_profiles.csv.")
    parser.add_argument("--tmin", type=float, help="Start of the averaging window (default: last 40%% of the run).")
    parser.add_argument("--tmax", type=float, help="End of the averaging window (default: end of the run).")
    parser.add_argument("--vA", type=float, default=None,
                        help="Alfven speed (older runs only); default: the vA column of scalar_diagnostics.csv.")
    parser.add_argument("--output", type=Path, help="Image path; default: RUN_DIRECTORY/z_rms_terms.png.")
    parser.add_argument("--show", action="store_true", help="Show the figure after saving (e.g. in Spyder).")
    args = parser.parse_args(argv)

    run_dir = args.run_dir.expanduser().resolve()
    if run_dir.is_file():  # the CSV was given
        run_dir = run_dir.parent
    output = args.output or run_dir / "z_rms_terms.png"

    # Read the profiles into arrays of shape (time, z).
    csv_path = run_dir / "z_profiles.csv"
    if not csv_path.is_file():
        raise SystemExit(f"{csv_path} not found; it is written every t_out_prof (default: t_out_spec) when that is > 0.")
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    times = np.unique([float(row["time"]) for row in rows])
    zs = np.unique([float(row["z"]) for row in rows])
    data = {}
    for row in rows:
        values = data.setdefault(row["quantity"], np.full((times.size, zs.size), np.nan))
        values[np.searchsorted(times, float(row["time"])), np.searchsorted(zs, float(row["z"]))] = float(row["value"])

    # Each budget: {term: (time, z) array}, plus its d_t W(z) when the run saved it.
    budgets, rates = {}, {}
    if "w_plus_dt" in data:
        for energy, terms in Z_PROFILE_LOCAL_BUDGETS.items():
            budgets[energy] = {term: data[f"{energy}_{term}"] for term in terms}
            rates[energy] = data[f"{energy}_dt"]
    else:
        # Older runs: transport from the W± profiles, slow-mode energies only together.
        vA = args.vA
        if vA is None:
            with (run_dir / "scalar_diagnostics.csv").open("r", encoding="utf-8", newline="") as handle:
                vA = float(next(csv.DictReader(handle))["vA"])
        Lz = zs.size * (zs[1] - zs[0])
        k = 2.0 * np.pi * np.fft.rfftfreq(zs.size, d=Lz / zs.size)

        def d_dz(profile):
            return np.fft.irfft(1j * k * np.fft.rfft(profile, axis=1), n=zs.size, axis=1)

        budgets = {
            "w_plus": {"transport": -vA * d_dz(data["w_plus"]), "buoyancy": data["w_plus_buoyancy"],
                       "dissipation": data.get("w_plus_dissipation")},
            "w_minus": {"transport": vA * d_dz(data["w_minus"]), "buoyancy": data["w_minus_buoyancy"],
                        "dissipation": data.get("w_minus_dissipation")},
            "slow": {"transport": -d_dz(data["slow_flux"]), "drive_KB": data["w_par_u_drive_KB"],
                     "drive_KBKp": data["w_par_bp_drive_KBKp"], "dissipation": data.get("slow_dissipation")},
            "w_s": {"drive_Ks": data["w_s_drive_Ks"], "dissipation": data.get("w_s_dissipation")},
        }
        budgets = {name: {t: v for t, v in terms.items() if v is not None} for name, terms in budgets.items()}

    # Saved profiles to average over: by default the last 40% of the run. Each
    # profile is a full sample on its own, so one profile is enough.
    start = args.tmin if args.tmin is not None else times[-1] - 0.4 * (times[-1] - times[0])
    end = args.tmax if args.tmax is not None else times[-1]
    window = (times >= start) & (times <= end)
    if not window.any():
        raise SystemExit(f"No saved profiles between t = {start:g} and {end:g}.")

    def rms(values):
        return float(np.sqrt(np.mean(values[window] ** 2)))

    plt = import_pyplot(show=args.show)
    ncols = 3
    nrows = -(-len(budgets) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(19, 4.6 * nrows), constrained_layout=True, squeeze=False)
    table = []  # (budget, term, rms, mean, share)

    for ax, (budget, terms) in zip(axes.flat, budgets.items()):
        # A term that is exactly zero (e.g. the grad-B drive when K_b0 = 0) is not drawn.
        terms = {term: values for term, values in terms.items() if np.any(values != 0.0)}
        term_rms = {term: rms(values) for term, values in terms.items()}
        term_mean = {term: float(np.mean(values[window])) for term, values in terms.items()}
        total = sum(term_rms.values()) or 1.0
        for term in terms:
            table.append((budget, term, term_rms[term], term_mean[term], term_rms[term] / total))

        # The rms over z of every term against time; the legend gives each
        # term's share of the summed rms over the shaded window.
        for term, values in terms.items():
            ax.plot(times, np.sqrt(np.mean(values**2, axis=1)), color=COLORS.get(term),
                    label=f"{LABELS.get(term, term)} ({term_rms[term] / total:.0%})")
        if budget in rates:
            residual = rates[budget] - sum(terms.values())
            table.append((budget, "dt", rms(rates[budget]), float(np.mean(rates[budget][window])), np.nan))
            table.append((budget, "residual", rms(residual), float(np.mean(residual[window])), np.nan))
        ax.axvspan(times[window][0], times[window][-1], color="0.9", zorder=0)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))
        ax.set(title=f"${TITLES[budget]}(z)$", xlabel="time", ylabel="rms over z")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, title="share of summed rms in window", title_fontsize=7)
    for ax in list(axes.flat)[len(budgets):]:
        ax.axis("off")

    table_path = output.with_suffix(".csv")
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["budget", "term", "rms_over_z", "mean", "rms_share", "t_start", "t_end"])
        for budget, term, rms_value, mean_value, share in table:
            writer.writerow([budget, term, f"{rms_value:.10e}", f"{mean_value:.10e}", f"{share:.4f}",
                             times[window][0], times[window][-1]])
    print(f"Saved {table_path}")

    fig.suptitle(f"{run_dir.parent.name}/{run_dir.name}: size of each local-in-z budget term, "
                 f"t = {times[window][0]:.3g} to {times[window][-1]:.3g} ({window.sum()} profiles)")
    finalize_figure(fig, output_path=output, show=args.show, plt=plt)
    return output


if __name__ == "__main__":
    main()
