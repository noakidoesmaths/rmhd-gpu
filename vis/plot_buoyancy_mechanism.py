"""Picture of the buoyancy (DCF) mechanism in one run, drawn from its saved numbers.

Geometry used by the rho equation set (see rmhdgpu/equations/rmhd_by_nokia_rho.py):

    B0 along z (horizontal in the sketch); x is the vertical direction.
    Gravity points along +x: the momentum equation has + g (drho/rho0) xhat, so
    "down" in the sketch is +x. K_rho0 = d ln rho0/dx > 0 puts the dense
    plasma at the bottom, and N^2 = g K_rho0 - g^2/(vA^2 (1+chi)) > 0 is stable.

The density flux across the gradient is V_rho_x = <drho u_x>, with
u_x = (z+_x + z-_x)/2, so it splits exactly into the two waves:

    V_rho_x = <drho z+_x>/2 + <drho z-_x>/2 = (-q_dcf - q_ccr_source)/g

V_rho_x < 0 lifts dense plasma (down-gradient mixing). Lifting it costs the
waves energy: buoyancy drains z+ at q_dcf = -(g/2)<drho z+_x> (DCF) and, because
the buoyancy force acts on u only, it hands part of that to z- (q_ccr_source < 0
means W- gains). The rest, g V_rho_x = acr_g, leaves the fluctuations for the
background potential energy.

Four panels: the sketch with the measured fluxes; where the forcing power goes
(window means of the saved budget terms); the two parts of the flux against
time; and the powers against time.

    python vis/plot_buoyancy_mechanism.py RUN_DIRECTORY
    python vis/plot_buoyancy_mechanism.py RUN_DIRECTORY --tmin 240 --show
"""

import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vis._matplotlib import finalize_figure, import_pyplot
from vis.run_quantities import column, load_parameters, read_columns, safe_divide, time_window


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="Run directory, or its scalar_diagnostics.csv.")
    parser.add_argument("--tmin", type=float, help="Start of the averaging window; default: last 40%% of the run.")
    parser.add_argument("--smooth", type=float, default=5.0, help="Running-mean width (time units) for the time panels.")
    parser.add_argument("--output", type=Path, help="Image path; default: RUN_DIRECTORY/buoyancy_mechanism.png.")
    parser.add_argument("--show", action="store_true", help="Show the figure after saving (e.g. in Spyder).")
    args = parser.parse_args(argv)

    run_dir = args.run_dir.expanduser().resolve()
    if run_dir.is_file():  # the CSV was given
        run_dir = run_dir.parent
    output = args.output or run_dir / "buoyancy_mechanism.png"
    c = read_columns(run_dir / "scalar_diagnostics.csv")
    p = load_parameters(run_dir)
    if p.g == 0.0:
        raise SystemExit(f"{run_dir.name} has g = 0: there is no buoyancy mechanism to draw.")
    t = c["time"]
    win = time_window(t, tmin=args.tmin)

    # Flux of density across the gradient, split into the part each wave carries.
    V_plus = -c["q_dcf"] / p.g          # <drho z+_x>/2
    V_minus = -c["q_ccr_source"] / p.g  # <drho z-_x>/2
    z_plus = 2.0 * np.sqrt(c["w_plus"])   # the solver saves W = <|z|^2>/4
    z_minus = 2.0 * np.sqrt(c["w_minus"])
    l_perp = 1.0 / c["k_perp_plus"]
    N = np.sqrt(p.N_sq) if p.N_sq > 0 else np.nan
    # rms(u_x) ~ u_perp_rms * rms(z+_x)/rms(|z+|), since u is mostly z+/2 here.
    corr = safe_divide(c["V_rho_x"], c["drho_rms"] * c["u_perp_rms"] * c["w_plus_align"])

    m = {name: np.mean(values[win]) for name, values in {
        "V": c["V_rho_x"], "V_plus": V_plus, "V_minus": V_minus, "z_plus": z_plus, "z_minus": z_minus,
        "corr": corr, "N_l_over_zm": N * l_perp / z_minus,
        "forcing": c["w_plus_rhs_forcing"], "diss_plus": -c["w_plus_rhs_dissipation"],
        "q_dcf": c["q_dcf"], "to_zminus": -c["q_ccr_source"], "to_background": -c["acr_g"],
        "diss_minus": -c["w_minus_rhs_dissipation"], "proj_minus": -column(c, "w_minus_rhs_projection"),
        "drive_s": column(c, "w_s_rhs_drive_Ks"), "diss_s": -column(c, "w_s_rhs_dissipation"),
        "proj_s": -column(c, "w_s_rhs_projection"),
    }.items()}
    t0, t1 = t[win][0], t[win][-1]

    plt = import_pyplot(show=args.show)
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True,
                             gridspec_kw={"height_ratios": [1.25, 1.0]})
    ax_sketch, ax_flow, ax_flux, ax_power = axes.ravel()

    # ---- Panel 1: the sketch. Up is -x, so gravity (+x) points down the page.
    ax_sketch.imshow(np.linspace(0.0, 1.0, 200)[:, None] * np.ones((1, 10)), extent=(0, 10, 0, 10),
                     origin="upper", cmap="Blues", vmin=-0.3, vmax=1.4, aspect="auto")
    ax_sketch.text(0.3, 9.5, r"light ($\rho_0$ small)", va="top")
    ax_sketch.text(0.3, 0.4, rf"dense ($\rho_0$ large),  $K_{{\rho0}} = {p.K_rho0:g}$", va="bottom", color="white")
    # Field line along z with the z+ wave on it, z- going the other way.
    zz = np.linspace(0.8, 7.6, 300)
    ax_sketch.plot(zz, 5.0 + 0.35 * np.sin(2.2 * zz), color="k", lw=1.5)
    ax_sketch.annotate("", xy=(2.2, 6.6), xytext=(0.6, 6.6), arrowprops=dict(arrowstyle="-|>", lw=2, color="k"))
    ax_sketch.text(0.6, 6.8, r"$\mathbf{B}_0$ ($z$)", va="bottom")
    ax_sketch.annotate("", xy=(7.6, 5.8), xytext=(4.6, 5.8),
                       arrowprops=dict(arrowstyle="-|>", lw=4, color="tab:red"))
    ax_sketch.text(4.6, 6.05, rf"$z^+$ (forced), rms {m['z_plus']:.2f}", color="tab:red", va="bottom")
    ax_sketch.annotate("", xy=(4.6, 4.2), xytext=(6.4, 4.2),
                       arrowprops=dict(arrowstyle="-|>", lw=1.2, color="tab:cyan"))
    ax_sketch.text(4.4, 3.95, rf"$z^-$ rms {m['z_minus']:.3f}  ($z^-/z^+ = {m['z_minus'] / m['z_plus']:.3f}$)",
                   color="tab:cyan", va="top", bbox=dict(fc="white", ec="none", alpha=0.85))
    # Density fluxes: arrow up = V < 0 (dense plasma lifted). Width ~ magnitude.
    scale = 6.0 / max(abs(m["V_plus"]), abs(m["V_minus"]))
    for x0, value, color, name in ((8.3, m["V_plus"], "tab:red", "+"), (9.3, m["V_minus"], "tab:cyan", "-")):
        up = value < 0
        y_tail, y_head = (2.0, 8.5) if up else (8.5, 2.0)
        ax_sketch.annotate("", xy=(x0, y_head), xytext=(x0, y_tail),
                           arrowprops=dict(arrowstyle="simple", color=color, alpha=0.85,
                                           mutation_scale=max(8.0, abs(value) * scale * 6)))
        ax_sketch.text(x0, 9.0 if up else 1.5, rf"$\frac{{1}}{{2}}\langle\delta\rho\, z^{name}_x\rangle$" + "\n"
                       + f"{value:+.2e}", ha="center", va="bottom" if up else "top", color=color, fontsize=9)
    ax_sketch.annotate("", xy=(0.6, 1.6), xytext=(0.6, 3.6), arrowprops=dict(arrowstyle="-|>", lw=3, color="tab:green"))
    ax_sketch.text(0.85, 2.6, rf"$\mathbf{{g}}$ along $+x$, $g = {p.g:g}$", color="tab:green", va="center",
                   bbox=dict(fc="white", ec="none", alpha=0.85))
    ax_sketch.text(
        0.3, 8.3,
        rf"$V_{{\rho,x}} = \langle\delta\rho\,u_x\rangle = {m['V']:+.2e}$ (up: dense plasma lifted)" + "\n"
        + rf"corr$(\delta\rho, u_x) \approx {m['corr']:+.3f}$" + "\n"
        + rf"$N\,l_\perp/z^- = {m['N_l_over_zm']:.1f}$  (> 1: $\delta\rho$ oscillates before $z^-$ shears $z^+$)",
        va="top", fontsize=9, bbox=dict(fc="white", ec="0.7", alpha=0.9))
    ax_sketch.set(xlim=(0, 10), ylim=(0, 10), xticks=[], yticks=[],
                  title=rf"Geometry and measured fluxes, mean over $t$ = {t0:.0f}-{t1:.0f}")

    # ---- Panel 2: where the power goes (window means; arrow width ~ rate).
    boxes = {"forcing": (0.9, 8.5), "W+": (3.6, 8.5), "diss+": (8.3, 8.5), "dcf": (3.6, 5.0),
             "background": (8.3, 5.0), "W-": (8.3, 3.0), "out-": (8.3, 0.9)}
    labels = {"forcing": "forcing", "W+": r"$W^+$", "diss+": r"$z^+$ cascade +" + "\ndissipation",
              "dcf": "buoyancy work\n(lift dense plasma)", "background": "background\npotential energy",
              "W-": r"$W^-$ (only source)", "out-": r"$z^-$ dissipation" + "\n+ projection"}
    for key, (x, y) in boxes.items():
        ax_flow.text(x, y, labels[key], ha="center", va="center", fontsize=10,
                     bbox=dict(boxstyle="round", fc="0.95", ec="0.4"))
    flows = [("forcing", "W+", m["forcing"]), ("W+", "diss+", m["diss_plus"]), ("W+", "dcf", m["q_dcf"]),
             ("dcf", "background", m["to_background"]), ("dcf", "W-", m["to_zminus"]),
             ("W-", "out-", m["diss_minus"] + m["proj_minus"])]
    width = 14.0 / m["forcing"]
    for start, end, rate in flows:
        (x0, y0), (x1, y1) = boxes[start], boxes[end]
        ax_flow.annotate("", xy=(x1, y1), xytext=(x0, y0),
                         arrowprops=dict(arrowstyle="-|>", lw=max(0.8, rate * width), color="tab:purple",
                                         alpha=0.6, shrinkA=32, shrinkB=32))
        ax_flow.text((x0 + x1) / 2, (y0 + y1) / 2, f"{rate:.2e}\n({rate / m['forcing']:.0%} of forcing)",
                     ha="center", va="center", fontsize=8.5, bbox=dict(fc="white", ec="none", alpha=0.9))
    ax_flow.text(0.2, 0.3,
                 "Separate loop, paid by the background\nentropy gradient, not by the forcing:\n"
                 rf"drive into $W_s$ {m['drive_s']:.2e}" + "\n"
                 rf"$\to$ dissipation {m['diss_s']:.2e}" + "\n"
                 rf"$\to$ projection {m['proj_s']:.2e}",
                 fontsize=8.5, va="bottom", color="tab:blue")
    ax_flow.set(xlim=(0, 10), ylim=(0, 10), xticks=[], yticks=[],
                title=rf"Energy flow, mean over $t$ = {t0:.0f}-{t1:.0f}")

    # ---- Panels 3 and 4: the same quantities against time (running mean).
    n_smooth = max(1, int(round(args.smooth / np.median(np.diff(t)))))
    kernel = np.ones(n_smooth) / n_smooth
    ts = np.convolve(t, kernel, mode="valid")
    for values, color, label in (
            (-c["V_rho_x"], "k", r"$-V_{\rho,x}$ total (up)"),
            (-V_plus, "tab:red", r"$-\frac{1}{2}\langle\delta\rho\,z^+_x\rangle$ (up)"),
            (V_minus, "tab:cyan", r"$+\frac{1}{2}\langle\delta\rho\,z^-_x\rangle$ (down)")):
        ax_flux.plot(ts, np.convolve(values, kernel, mode="valid"), color=color, label=label)
    ax_flux.set_yscale("symlog", linthresh=1e-4)  # symlog: the early flux changes sign
    ax_flux.set(xlabel="time", ylabel="density flux",
                title=rf"Density flux across $\nabla\rho_0$ (running mean over {args.smooth:g}; symlog)")
    ax_flux.legend()
    for values, color, label in (
            (c["w_plus_rhs_forcing"], "k", "forcing power"),
            (-c["w_plus_rhs_dissipation"], "tab:red", r"$z^+$ dissipation"),
            (c["q_dcf"], "tab:purple", r"buoyancy drain of $z^+$ ($q_{\rm dcf}$)"),
            (-c["q_ccr_source"], "tab:cyan", r"buoyancy source of $z^-$")):
        ax_power.plot(ts, np.convolve(values, kernel, mode="valid"), color=color, label=label)
    ax_power.set_yscale("symlog", linthresh=1e-5)
    ax_power.set(xlabel="time", ylabel="power", title="Powers against time (symlog)")
    ax_power.legend()
    for ax in (ax_flux, ax_power):
        ax.axvspan(t0, t1, color="0.9", zorder=0)

    fig.suptitle(f"{run_dir.parent.name}/{run_dir.name}: buoyancy mechanism", fontsize=14)
    finalize_figure(fig, output_path=output, show=args.show, plt=plt)
    return output


if __name__ == "__main__":
    main()
