"""Signed local energy budgets: how much each term adds to and removes from each energy, against time.

For each term T(z, t) of the local budget (see `rmhdgpu/diagnostics/z_profiles.py`),
the sign is split at each z before averaging over z:

    gain(t) = < max(T, 0) >_z      red: where the term adds energy
    loss(t) = < min(T, 0) >_z      blue: where the term removes energy
    net(t)  = gain + loss = < T >_z    dashed: the box-averaged term

One panel per term shows these three at each saved time (no accumulation in
time). A one-way term has only a red or only a blue curve; a term that adds
energy at some z and removes it at others has both, with the net between
them; the transport term always has gain = -loss and zero net. For T(z, t)
itself as a space-time map, use vis/plot_z_profiles.py.

    python vis/plot_local_budget_signs.py RUN_DIRECTORY
    python vis/plot_local_budget_signs.py RUN_DIRECTORY --energy w_plus w_par_u --tmin 100 --tmax 150

One figure per energy, local_budget_<energy>.png. By default w_plus, w_minus,
w_par_u, w_par_b and w_s; w_p is left out because its budget is w_par_b's / chi
(`--energy w_p` or `--energy all` adds it). Runs made
before 2026-11-21 have no slow-mode term profiles; for those only
w_plus and w_minus work, with the transport term taken from W±(z).
"""

import argparse
import csv
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from matplotlib.lines import Line2D

from rmhdgpu.diagnostics.z_profiles import Z_PROFILE_LOCAL_BUDGETS
from vis._matplotlib import finalize_figure, import_pyplot
from vis.run_quantities import RESERVOIR_LABELS, TERM_LABELS

GAIN, LOSS, NET = "#ab302a", "#21668f", "#171d25"
# w_p is left out by default: its local budget is exactly w_par_b's divided by chi.
DEFAULT_ENERGIES = ("w_plus", "w_minus", "w_par_u", "w_par_b", "w_s")
LABELS = {**TERM_LABELS, "transport": r"transport $\mp v_A\partial_z W$"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="Run directory, or its z_profiles.csv.")
    parser.add_argument("--energy", nargs="+", default=list(DEFAULT_ENERGIES),
                        choices=[*Z_PROFILE_LOCAL_BUDGETS, "all"], metavar="ENERGY",
                        help="Energies to plot, one figure each (default: w_plus w_minus w_par_u w_par_b w_s; "
                             "'all' adds w_p).")
    parser.add_argument("--tmin", type=float, default=-np.inf, help="Start of the time window.")
    parser.add_argument("--tmax", type=float, default=np.inf, help="End of the time window.")
    parser.add_argument("--vA", type=float, default=None,
                        help="Alfven speed (older runs only); default: the vA column of scalar_diagnostics.csv.")
    parser.add_argument("--output", type=Path,
                        help="Image path for a single energy; default: RUN_DIRECTORY/local_budget_<energy>.png.")
    parser.add_argument("--show", action="store_true", help="Show the figures after saving (e.g. in Spyder).")
    args = parser.parse_args(argv)

    run_dir = args.run_dir.expanduser().resolve()
    if run_dir.is_file():  # the CSV was given
        run_dir = run_dir.parent

    # Read the profiles in the window into arrays of shape (time, z).
    csv_path = run_dir / "z_profiles.csv"
    if not csv_path.is_file():
        raise SystemExit(f"{csv_path} not found; it is written every t_out_prof (default: t_out_spec) when that is > 0.")
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if args.tmin <= float(row["time"]) <= args.tmax]
    if not rows:
        raise SystemExit("No saved profiles in the time window.")
    times = np.unique([float(row["time"]) for row in rows])
    zs = np.unique([float(row["z"]) for row in rows])
    if times.size < 2:
        raise SystemExit("Need at least two saved profiles in the time window.")
    data = {}
    for row in rows:
        values = data.setdefault(row["quantity"], np.full((times.size, zs.size), np.nan))
        values[np.searchsorted(times, float(row["time"])), np.searchsorted(zs, float(row["z"]))] = float(row["value"])

    energies = list(Z_PROFILE_LOCAL_BUDGETS) if "all" in args.energy else args.energy
    plt = import_pyplot(show=args.show)
    outputs = []
    for energy in energies:
        # This energy's local terms.
        terms = {}
        for term in Z_PROFILE_LOCAL_BUDGETS[energy]:
            name = f"{energy}_{term}"
            if name in data:
                terms[term] = data[name]
            elif term == "transport":
                # Older runs: -/+ vA d_z W±(z) from the saved profile.
                vA = args.vA
                if vA is None:
                    with (run_dir / "scalar_diagnostics.csv").open("r", encoding="utf-8", newline="") as handle:
                        vA = float(next(csv.DictReader(handle))["vA"])
                k = 2.0 * np.pi * np.fft.rfftfreq(zs.size, d=zs[1] - zs[0])
                d_dz = np.fft.irfft(1j * k * np.fft.rfft(data[energy], axis=1), n=zs.size, axis=1)
                terms[term] = (-vA if energy == "w_plus" else vA) * d_dz
        # A term that is exactly zero (e.g. the grad-B drive when K_b0 = 0) gets no row.
        terms = {term: values for term, values in terms.items() if np.any(values != 0.0)}
        if not terms:
            print(f"Skipping {energy}: {csv_path} has no local-budget profiles for it (rerun to get them).")
            continue

        fig, axes = plt.subplots(len(terms), 1, squeeze=False, sharex=True, figsize=(8, 2.2 * len(terms) + 1.3),
                                 constrained_layout=True)
        table = []
        for (ax_line,), (term, rate) in zip(axes, terms.items()):
            gain = np.maximum(rate, 0.0).mean(axis=1)
            loss = np.minimum(rate, 0.0).mean(axis=1)
            net = rate.mean(axis=1)
            ax_line.fill_between(times, 0, gain, color=GAIN, alpha=0.2)
            ax_line.fill_between(times, 0, loss, color=LOSS, alpha=0.2)
            for values, color, style in ((gain, GAIN, "-"), (loss, LOSS, "-"), (net, NET, "--")):
                ax_line.plot(times, values, color=color, ls=style, lw=1.4)
            ax_line.axhline(0.0, lw=0.6, color="0.5", zorder=0)
            ax_line.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))
            ax_line.set_ylabel(f"{LABELS.get(term, term)}\nrate", fontsize=8)
            ax_line.grid(True, alpha=0.3)
            table += [(t, term, g, l, n) for t, g, l, n in zip(times, gain, loss, net)]

        handles = [Line2D([], [], color=GAIN, label=r"gain $\langle\max(T,0)\rangle_z$"),
                   Line2D([], [], color=LOSS, label=r"loss $\langle\min(T,0)\rangle_z$"),
                   Line2D([], [], color=NET, ls="--", label=r"net $\langle T\rangle_z$")]
        axes[-1, 0].legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=3, frameon=False)
        axes[-1, 0].set_xlabel("time")
        axes[0, 0].set_title("each term split by sign over z, at each time")
        fig.suptitle(f"{run_dir.parent.name}/{run_dir.name}: signed local budget of {RESERVOIR_LABELS[energy]}")

        output = args.output if (args.output and len(energies) == 1) else run_dir / f"local_budget_{energy}.png"
        with output.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["time", "term", "gain", "loss", "net"])
            writer.writerows(table)
        finalize_figure(fig, output_path=output, show=args.show, plt=plt)
        outputs.append(output)
    if not outputs:
        raise SystemExit("Nothing to plot.")
    return outputs[0]


if __name__ == "__main__":
    main()
