"""Energy flux through k_perp: does the run cascade?

Reads `energy_flux.csv`, which holds, at every spectra output and for every
energy partition, the rate each kind of term changes the k_perp shell
spectrum (see `rmhdgpu/diagnostics/energy_flux.py`):

    nonlinear_<partition>    T(k)  Poisson brackets: moves energy between shells
    linear_<partition>       L(k)  background gradients, gravity, d_z wave terms
    dissipation_<partition>  D(k)  dissipation (negative)

and averages them over a time window. Use a window where the run is
statistically steady (check with `vis/plot_stationarity.py`).

The energy flux through k_perp = K is

    Pi(K) = - sum_{k < K} T(k),

the energy per unit time the nonlinear terms carry from k_perp < K to
k_perp > K. It is divided by the injection rate

    eps = window mean of sum_k L(k)  +  total_energy_rhs_forcing,

the energy put in by the linear drives (from energy_flux.csv) and by the
forcing (from scalar_diagnostics.csv, when it has that column).

Left panel: Pi(K)/eps for the total energy, z+, z-, and the compressive
energy (total - z+ - z-, in sets that have one).
  - Cascade: Pi/eps ~ 1 (flat) over a range of K between injection and dissipation.
  - Energy dissipated where it is injected (no cascade): Pi/eps drops to ~0 just above injection.
  - Pi < 0: energy moving to larger scales (e.g. into a condensate).

Right panel: scale-by-scale budget of the total energy. In a steady state,
what is injected below K either crosses K (Pi) or is dissipated below K:

    injected(<K) = Pi(K) + dissipated(<K)

so the dashed sum Pi + dissipated should lie on the injected curve. Two
things are not in energy_flux.csv and show up as a gap between the two:
forcing kicks (the gap is then the forcing rate) and the mode projections
`project_kx0/ky0/kpar0` (an energy sink). A run that is not steady also
leaves a gap, d_t E(<K): a decaying run feeds Pi from its stored energy, so
Pi/eps can then exceed 1.

Bands are +- one standard error from block averages (--blocks): the
window is split into blocks, Pi is computed for each block's mean, and the
scatter of the blocks gives the error.

    python vis/plot_energy_flux.py RUN_DIRECTORY --tmin 200
    python vis/plot_energy_flux.py RUN_DIRECTORY --tmin 200 --blocks 6 --show
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vis._matplotlib import finalize_figure, import_pyplot
from vis.run_quantities import budget_rates, budget_rows, describe_window, read_columns, time_window, write_summary_csv

LABELS = {"total": "total", "z_plus": r"$z^+$", "z_minus": r"$z^-$", "compressive": "compressive (total $- z^\\pm$)"}
COLORS = {"total": "k", "z_plus": "tab:red", "z_minus": "tab:cyan", "compressive": "tab:blue"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="Run directory, or its energy_flux.csv.")
    parser.add_argument("--tmin", type=float, help="Start of the window (default: the last 40%% of the run).")
    parser.add_argument("--tmax", type=float, help="End of the window (default: the end of the run).")
    parser.add_argument("--blocks", type=int, default=4, help="Blocks for the error bands (default: 4).")
    parser.add_argument("--output", type=Path, help="Image path; default: RUN_DIRECTORY/energy_flux.png.")
    parser.add_argument("--show", action="store_true", help="Show the figure after saving (e.g. in Spyder).")
    args = parser.parse_args(argv)

    run_dir = args.run_dir.expanduser().resolve()
    if run_dir.is_file():  # the CSV was given
        run_dir = run_dir.parent
    flux_path = run_dir / "energy_flux.csv"
    if not flux_path.is_file():
        raise SystemExit(f"{flux_path} not found; runs made before the energy-flux diagnostic do not have it.")
    output = args.output or run_dir / "energy_flux.png"

    # ------------------------------------------------------------------
    # Read energy_flux.csv into {quantity: array[time, shell]}.
    # ------------------------------------------------------------------
    rows_by_quantity = {}
    with flux_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows_by_quantity.setdefault(row["quantity"], []).append(
                (float(row["time"]), float(row["kperp"]), float(row["value"])))
    first_table = np.array(next(iter(rows_by_quantity.values())))
    times = np.unique(first_table[:, 0])  # every quantity is saved at the same times
    kperp = np.unique(first_table[:, 1])  # and in the same shells
    rates = {}
    for quantity, rows in rows_by_quantity.items():
        rows.sort()  # by time, then by kperp
        rates[quantity] = np.array(rows)[:, 2].reshape(times.size, kperp.size)

    # Shell n covers [n dk, (n + 1) dk); Pi at its upper edge K = (n + 1) dk counts every shell below K.
    shell_width = kperp[1] - kperp[0]
    upper_edge = kperp + shell_width / 2

    # ------------------------------------------------------------------
    # Average over the window, in blocks, and form the fluxes.
    # ------------------------------------------------------------------
    window = time_window(times, tmin=args.tmin, tmax=args.tmax)
    window_rows = np.flatnonzero(window)
    blocks = np.array_split(window_rows, args.blocks)
    if min(len(block) for block in blocks) < 1:
        raise SystemExit(f"The window has only {window_rows.size} outputs; use fewer --blocks.")

    def block_means(quantity):
        """Window-averaged rate per shell for each block: array[block, shell]."""
        return np.array([rates[quantity][block].mean(axis=0) for block in blocks])

    def mean_and_error(per_block):
        """Mean over blocks, and its standard error from the scatter of the blocks."""
        return per_block.mean(axis=0), per_block.std(axis=0, ddof=1) / np.sqrt(len(per_block))

    # Pi(K) = - sum_{k < K} T(k): a cumulative sum over shells.
    transfer = {name: block_means(f"nonlinear_{name}") for name in ("total", "z_plus", "z_minus")
                if f"nonlinear_{name}" in rates}
    if {"total", "z_plus", "z_minus"} <= set(transfer):
        compressive = transfer["total"] - transfer["z_plus"] - transfer["z_minus"]
        if np.any(np.abs(compressive) > 1e-12 * np.abs(transfer["total"]).max()):
            transfer["compressive"] = compressive
    flux = {name: mean_and_error(-np.cumsum(per_block, axis=1)) for name, per_block in transfer.items()}

    # Total-energy budget below K.
    injected_linear = mean_and_error(np.cumsum(block_means("linear_total"), axis=1))
    dissipated = mean_and_error(-np.cumsum(block_means("dissipation_total"), axis=1))

    # Injection rate eps: linear drives (all shells) + forcing from the scalar budget.
    forcing = 0.0
    scalar_path = run_dir / "scalar_diagnostics.csv"
    if scalar_path.is_file():
        columns = read_columns(scalar_path)
        if "total_energy_rhs_forcing" in columns:
            rows = budget_rows(columns["time"], tmin=times[window][0], tmax=times[window][-1])
            if rows.size:
                forcing = budget_rates(columns, "total_energy", rows)["forcing"]
    linear_drive = injected_linear[0][-1]
    eps = linear_drive + forcing
    if eps <= 0:
        raise SystemExit(f"Injection rate eps = {eps:.3g} is not positive; Pi/eps cannot be formed.")

    # Shells beyond the dealiasing cut-off are empty: leave them off the plot.
    active = np.any(np.abs(rates["dissipation_total"][window]) > 0, axis=0) | \
        np.any(np.abs(rates["nonlinear_total"][window]) > 0, axis=0)
    last = np.flatnonzero(active)[-1] + 1
    K = upper_edge[:last]

    peak = int(np.argmax(flux["total"][0][:last]))
    print(f"Window: {describe_window(args.tmin, args.tmax)} ({window_rows.size} outputs)")
    print(f"Injection rate eps = {eps:.4g}  (linear drives {linear_drive:.4g}, forcing {forcing:.4g})")
    print(f"Window-mean dissipation (all shells) = {dissipated[0][last - 1]:.4g}")
    print(f"Peak flux of the total energy: Pi/eps = {flux['total'][0][peak] / eps:.3f} at K = {K[peak]:.3g}")

    summary = []
    for index in range(last):
        row = {"K": K[index]}
        for name, (mean, error) in flux.items():
            row[f"Pi_{name}"], row[f"Pi_{name}_error"] = mean[index], error[index]
        row["injected_linear_below_K"] = injected_linear[0][index]
        row["dissipated_below_K"] = dissipated[0][index]
        row["eps"] = eps
        summary.append(row)
    write_summary_csv(summary, output.with_name("energy_flux_summary.csv"))

    # ------------------------------------------------------------------
    # Figure.
    # ------------------------------------------------------------------
    plt = import_pyplot(show=args.show)
    fig, (ax_flux, ax_budget) = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

    for name, (mean, error) in flux.items():
        width = 4 if name == "total" else 1.5  # thick, so it stays visible under a partition that equals it
        ax_flux.plot(K, mean[:last] / eps, color=COLORS[name], lw=width, alpha=0.6 if name == "total" else 1,
                     marker="o", ms=3, label=LABELS[name])
        ax_flux.fill_between(K, (mean - error)[:last] / eps, (mean + error)[:last] / eps,
                             color=COLORS[name], alpha=0.2, lw=0)
    ax_flux.axhline(1, color="0.5", ls="--", lw=1, label=r"$\Pi = \varepsilon$")
    ax_flux.axhline(0, color="0.5", lw=0.8)
    ax_flux.set(xscale="log", xlabel=r"$K$ (upper edge of the $k_\perp$ shell)", ylabel=r"$\Pi(K)\,/\,\varepsilon$",
                title=r"Nonlinear energy flux through $k_\perp = K$")
    ax_flux.legend()
    ax_flux.grid(alpha=0.3, which="both")

    for label, (mean, error), color in (
            (r"injected below $K$ (linear terms)", injected_linear, "tab:green"),
            (r"$\Pi(K)$: crosses $K$", flux["total"], "k"),
            (r"dissipated below $K$", dissipated, "tab:red")):
        ax_budget.plot(K, mean[:last], color=color, marker="o", ms=3, label=label)
        ax_budget.fill_between(K, (mean - error)[:last], (mean + error)[:last], color=color, alpha=0.2, lw=0)
    ax_budget.plot(K, (flux["total"][0] + dissipated[0])[:last], color="0.4", ls="--",
                   label=r"$\Pi$ + dissipated (= injected if steady)")
    if forcing != 0.0:
        ax_budget.axhline(eps, color="tab:green", ls=":", label=r"$\varepsilon$ incl. forcing")
    ax_budget.axhline(0, color="0.5", lw=0.8)
    ax_budget.set(xscale="log", xlabel=r"$K$", ylabel="rate of change of total energy",
                  title="Total-energy budget below $K$")
    ax_budget.legend(fontsize=9)
    ax_budget.grid(alpha=0.3, which="both")

    fig.suptitle(f"{run_dir.parent.name}/{run_dir.name}: energy flux, {describe_window(args.tmin, args.tmax)}, "
                 f"$\\varepsilon$ = {eps:.3g}")
    finalize_figure(fig, output_path=output, show=args.show, plt=plt)
    return output


if __name__ == "__main__":
    main()
