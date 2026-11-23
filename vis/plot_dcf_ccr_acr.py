"""The three heating channels of Squire et al. (arXiv:2607.08036) against time: DCF, CCR and ACR.

All are measured from the budget columns of `scalar_diagnostics.csv`, which
are averages over each output interval (so they integrate exactly to the
energy changes; see `rmhdgpu/diagnostics/compressive_channels.py`).

  Q_DCF (direct compressive feedback): buoyancy acting on the compressive
      fluctuations drains the launched wave z+.
          Q_DCF = -w_plus_rhs_buoyancy = -(g/2) <drho z+_x>         positive: z+ loses energy

  Q_CCR (compressively catalysed reflection): the z- made by the compressive
      fluctuations shears z+ apart; z+ cascades and dissipates at the rate
      W+ z-/l_perp (paper Eq. 75), so the heating is the W+ dissipation:
          Q_CCR = -w_plus_rhs_dissipation
      The dashed line is the reflection step that makes this possible, the
      rate buoyancy creates z-:  (g/2) <drho z-_x> = w_minus_rhs_buoyancy.
      It is not heating: z- acts as a catalyst, so a small z- source can
      enable a much larger Q_CCR. Reading -D+ as the cascade rate needs a
      (statistically) steady state and dissipation below the forcing scale;
      vis/plot_zplus_heating.py checks the second.

  Q_ACR (Alfven-catalysed relaxation), paper Eq. 78:
          Q_ACR = Y_U + Y_B + Y_g - div(p V~_psi)
      with p the background thermal pressure. In this box Y_U = 0 (no
      background flow) and the field is straight (curvature 0). The
      divergence term splits as
          div(p V~_psi) = p div(V~_psi) + V~_psi . grad(p)
      The first part is zero: V~_psi is a box average, one constant vector,
      and V~_psi = V_psi because u_perp has no k_perp = 0 mode (no mean flow
      U_perp2). The second part is NOT zero when K_p0 != 0. The solver's
      force balance, K_b0 = g/vA^2 - chi K_p0/gamma, makes it
          V~_psi . grad(p) = (g - vA^2 K_b0) V_psi_x = (chi vA^2 K_p0/gamma) V_psi_x
      so, with Y_B = acr_B = -vA^2 K_b0 V_psi_x and Y_g = acr_g = g V_rho_x,
          Q_ACR = Y_B + Y_g - (g - vA^2 K_b0) V_psi_x = g (V_rho_x - V_psi_x)
      which is the paper's second form of Eq. 78, rho (V_rho - V_psi) . g_eff.
      The pressure work -(g - vA^2 K_b0) V_psi_x is drawn with Y_B and Y_g;
      the three thin ACR lines add up to Q_ACR. V_psi_x is the interval
      average -acr_B/(vA^2 K_b0); only when K_b0 = 0 (and g != 0) is the
      instantaneous V_psi_x column used instead. g, vA and K_b0 come from the
      run's input_copy.input.
      The thermal part (acr_th) is not drawn: the paper keeps it in the
      thermal budget, not in the heating, because what it powers is
      dissipated back into heat. Note Y_g = -Q_DCF + (z- source), so Q_ACR and
      Q_DCF can largely cancel (paper Sec. III.5).

Forcing is not drawn; in a steady state Q_DCF + Q_CCR equals the forcing
power into W+ (`w_plus_rhs_forcing`, see vis/plot_energy_budgets.py).
Each legend entry gives its mean over the shaded window (default: last 40% of
the run), and the same means go to a CSV next to the figure. `--average N`
merges every N output intervals into one to smooth the curves.

    python vis/plot_dcf_ccr_acr.py RUN_DIRECTORY
    python vis/plot_dcf_ccr_acr.py RUN_DIRECTORY --tmin 100 --average 10 --show
"""

import argparse
import csv
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from vis._matplotlib import finalize_figure, import_pyplot
from vis.run_quantities import budget_rows, load_parameters, read_columns


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="Run directory, or its scalar_diagnostics.csv.")
    parser.add_argument("--tmin", type=float, help="Start of the averaging window (default: last 40%% of the run).")
    parser.add_argument("--tmax", type=float, help="End of the averaging window (default: end of the run).")
    parser.add_argument("--average", type=int, default=1, metavar="N",
                        help="Merge every N output intervals into one for plotting (default 1: as saved).")
    parser.add_argument("--output", type=Path, help="Image path; default: RUN_DIRECTORY/dcf_ccr_acr.png.")
    parser.add_argument("--show", action="store_true", help="Show the figure after saving (e.g. in Spyder).")
    args = parser.parse_args(argv)

    run_dir = args.run_dir.expanduser().resolve()
    if run_dir.is_file():  # the CSV was given
        run_dir = run_dir.parent
    output = args.output or run_dir / "dcf_ccr_acr.png"

    columns = read_columns(run_dir / "scalar_diagnostics.csv")
    needed = ["w_plus_rhs_buoyancy", "w_minus_rhs_buoyancy", "w_plus_rhs_dissipation",
              "total_energy_rhs_acr_B", "total_energy_rhs_acr_g"]
    missing = [name for name in needed if name not in columns]
    if missing:
        raise SystemExit(f"{run_dir}/scalar_diagnostics.csv has no {', '.join(missing)} columns.")
    time = columns["time"]

    # Every channel as an interval-averaged rate (row i covers (t_{i-1}, t_i]).
    channels = {
        "Q_DCF": -columns["w_plus_rhs_buoyancy"],
        "Q_CCR": -columns["w_plus_rhs_dissipation"],
        "reflection_z_minus": columns["w_minus_rhs_buoyancy"],
        "Y_B": columns["total_energy_rhs_acr_B"],
        "Y_g": columns["total_energy_rhs_acr_g"],
    }
    # Y_U = 0 and p div(V~_psi) = 0 in this box, but V~_psi . grad(p) is not
    # (see the module docstring): Q_ACR = Y_B + Y_g - (g - vA^2 K_b0) V_psi_x.
    p = load_parameters(run_dir)
    pressure_gradient = p.g - p.vA**2 * p.K_b0  # = chi vA^2 K_p0 / gamma by force balance
    # K_b0 = g/vA^2 - chi K_p0/gamma can come out as round-off (~1e-18) when the
    # input balances g against K_p0; dividing acr_B by that would amplify CSV rounding.
    if abs(p.vA**2 * p.K_b0) > 1e-10 * (abs(p.g) + abs(pressure_gradient)):
        v_psi_x = -channels["Y_B"] / (p.vA**2 * p.K_b0)  # interval average, like the budget columns
    elif pressure_gradient == 0.0:
        v_psi_x = np.zeros_like(time)  # g = K_p0 = 0: no pressure gradient, nothing to multiply
    elif "V_psi_x" in columns:
        print("K_b0 = 0: using the instantaneous V_psi_x column for the pressure work (not an interval average).")
        v_psi_x = columns["V_psi_x"]
    else:
        raise SystemExit(f"{run_dir}/scalar_diagnostics.csv has no V_psi_x column, needed when K_b0 = 0.")
    channels["pressure_work"] = -pressure_gradient * v_psi_x
    channels["Q_ACR"] = channels["Y_B"] + channels["Y_g"] + channels["pressure_work"]

    # Window means, each row weighted by its interval.
    rows = budget_rows(time, tmin=args.tmin, tmax=args.tmax)
    interval = time[rows] - time[rows - 1]
    means = {name: float(np.sum(series[rows] * interval) / interval.sum()) for name, series in channels.items()}

    # Merge every N intervals for plotting: the merged rate is the integral over the merged length.
    step = max(args.average, 1)
    ends = np.arange(0, time.size, step)
    dt_rows = np.diff(time, prepend=time[0])
    t_mid = 0.5 * (time[ends][1:] + time[ends][:-1])

    def merged(series):
        integral = np.cumsum(series * dt_rows)
        return np.diff(integral[ends]) / np.diff(time[ends])

    plt = import_pyplot(show=args.show)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), sharex=True, constrained_layout=True)
    # (channel, label, colour, line style, line width)
    panels = [
        ("DCF", [
            ("Q_DCF", r"$Q_{DCF} = -\frac{g}{2}\langle\delta\rho\, z^+_x\rangle$", "tab:purple", "-", 1.8),
        ]),
        ("CCR", [
            ("Q_CCR", r"$Q_{CCR} = -D^+$ (W$^+$ dissipation)", "tab:red", "-", 1.8),
            ("reflection_z_minus", r"reflection: z$^-$ created, $\frac{g}{2}\langle\delta\rho\, z^-_x\rangle$",
             "tab:cyan", "--", 1.0),
        ]),
        ("ACR", [
            ("Q_ACR", r"$Q_{ACR} = g\,(V_{\rho,x} - V_{\psi,x})$", "black", "-", 1.8),
            ("Y_B", r"$Y_B$ (magnetic)", "tab:orange", "-", 1.0),
            ("Y_g", r"$Y_g$ (potential)", "tab:purple", "-", 1.0),
            ("pressure_work", r"$-\tilde V_\psi\cdot\nabla p$ (pressure work)", "tab:green", "-", 1.0),
        ]),
    ]
    for ax, (title, lines) in zip(axes, panels):
        for name, label, color, style, width in lines:
            ax.plot(t_mid, merged(channels[name]), color=color, ls=style, lw=width,
                    label=f"{label}   mean {means[name]:.2e}")
        ax.axhline(0.0, color="k", lw=0.6)
        ax.axvspan(time[rows[0] - 1], time[rows[-1]], color="0.92", zorder=0)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))
        ax.set(title=title, xlabel="time", ylabel="rate")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    table_path = output.with_suffix(".csv")
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["channel", "window_mean", "t_start", "t_end"])
        for name, value in means.items():
            writer.writerow([name, f"{value:.10e}", time[rows[0] - 1], time[rows[-1]]])
    print(f"Saved {table_path}")

    merged_note = f", every {step} output intervals merged" if step > 1 else ""
    fig.suptitle(f"{run_dir.parent.name}/{run_dir.name}: DCF, CCR and ACR heating{merged_note} "
                 f"(window t = {time[rows[0] - 1]:.3g} to {time[rows[-1]]:.3g})")
    finalize_figure(fig, output_path=output, show=args.show, plt=plt)
    return output


if __name__ == "__main__":
    main()
