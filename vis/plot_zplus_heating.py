"""Compare the two measured heating channels of z+ with their closure estimates.

z+ = delta u_perp - delta B_perp/sqrt(4 pi rho) is the forced (or launched)
wave. In the inhomogeneous RMHD sets it loses energy in two ways, so in a
steady state the forcing input is DCF + CCR (Squire et al., arXiv:2607.08036):

  DCF, left panel:  buoyancy acting on the slaved density drains z+.
      measured:  q_dcf = -(d_t W+)|_buoyancy
      estimate:  -g F_rho eta_turb,   eta_turb = l_perp z+_x^2 / (2 z+_rms)     (Eq. 69)
  CCR, right panel: the z- made by the compressive fields shears z+.
      measured:  D+ = -w_plus_rhs_dissipation
      estimate:  |g F_rho| l_perp z+_x / 4                                      (Eq. 75)

z+_x = rms(z+_x) is the part of z+ along the background gradient. Every
quantity comes from vis/run_quantities.py, which explains the estimates.

The console summary gives the window means, their ratio (measured/estimate, a
fitted prefactor rather than a pass/fail test), each channel's share of the
measured z+ loss, and W+ z-_rms / l_perp, the CCR rate with the measured z-
instead of the slaved one.

D+ is only the CCR heating in a steady state with z+-only forcing and
dissipation well below the outer scale. The script warns when z- is forced
too, or when more than 10% of D+ is in the forced shell or below (from
spectra.csv and resolved_config.toml, when both exist).

The window is the last --tail-fraction of the run's time span (0.4 by
default), or starts at --tmin; every saved time is plotted.

Examples (also usable as main([...]) in Spyder):
    python vis/plot_zplus_heating.py RUN_DIRECTORY
    python vis/plot_zplus_heating.py RUN_DIRECTORY --tmin 100 --show
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from vis._matplotlib import finalize_figure, import_pyplot
from vis.run_quantities import load_run, time_window


# Above this share of D+ in the forced shell or below, D+ is not mostly cascade.
FORCED_SHELL_WARNING = 0.1

# Each panel: (title, measured attribute, estimate attribute, measured label, estimate label).
CHANNELS = {
    "DCF": ("q_dcf", "q_dcf_predicted", r"Measured $q_{\rm dcf}$",
            r"Eq. (69): $-gF_\rho\,\ell_\perp (z^+_x)^2/(2z^+_{\rm rms})$"),
    "CCR": ("q_ccr", "q_ccr_predicted", r"Measured $D^+$",
            r"Eq. (75): $|gF_\rho|\,\ell_\perp z^+_x/4$"),
}


def window_means(run, window):
    """{channel: (mean measured, mean estimate)} over the window rows where both are finite."""
    means = {}
    for name, (measured_key, predicted_key, _, _) in CHANNELS.items():
        measured, predicted = getattr(run, measured_key), getattr(run, predicted_key)
        usable = window & np.isfinite(measured) & np.isfinite(predicted)
        means[name] = (float(measured[usable].mean()), float(predicted[usable].mean())) if usable.any() \
            else (np.nan, np.nan)
    return means


def ratio(measured, predicted):
    """Measured over estimate, or NaN when the estimate is zero or missing."""
    return measured / predicted if np.isfinite(predicted) and predicted != 0.0 else np.nan


def loss_shares(means):
    """Each channel's share of the measured z+ loss, DCF + D+, or None if that is not positive."""
    total = sum(measured for measured, _ in means.values())
    if not (np.isfinite(total) and total > 0.0):
        return None
    return {name: measured / total for name, (measured, _) in means.items()}


# ---------------------------------------------------------------------------
# When the measured CCR is not CCR: z- forced, or D+ in the forced shell
# ---------------------------------------------------------------------------

def read_resolved_config(run_dir):
    """The run's resolved_config.toml as a dict, or {} if it was not saved."""
    path = Path(run_dir) / "resolved_config.toml"
    if not path.is_file():
        return {}
    with path.open("rb") as handle:
        return tomllib.load(handle)


def forced_shell(config):
    """The controlled-shell forcing table, or None if the run is not forced that way."""
    forcing = config.get("forcing", {})
    if not forcing.get("use_forcing", False) or forcing.get("type") != "controlled_shell":
        return None
    return forcing.get("controlled_shell")


def hyperviscosity_order(config):
    """n_perp of the z+ damping, or None unless psi and omega share one k_perp-only hyperviscosity.

    Auto mode applies one common nu_perp k_perp^(2 n_perp) to every field.
    Manual mode sets each field separately, so psi and omega must match.
    """
    dissipation = config.get("dissipation", {})
    if dissipation.get("mode") == "auto":
        return None if dissipation.get("nu_par", 0.0) else dissipation.get("n_perp")
    psi, omega = dissipation.get("psi"), dissipation.get("omega")
    if psi is None or omega is None or psi.get("nu_par") or omega.get("nu_par"):
        return None
    if (psi.get("nu_perp"), psi.get("n_perp")) != (omega.get("nu_perp"), omega.get("n_perp")):
        return None
    return psi.get("n_perp")


def mean_spectrum(run_dir, quantity, t_start, t_end):
    """Mean shell spectrum of one quantity in spectra.csv over t_start <= time <= t_end, or None."""
    path = Path(run_dir) / "spectra.csv"
    if not path.is_file():
        return None
    sums, counts = {}, {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["quantity"] != quantity or not t_start <= float(row["time"]) <= t_end:
                continue
            k = float(row["kperp"])
            sums[k] = sums.get(k, 0.0) + float(row["value"])
            counts[k] = counts.get(k, 0) + 1
    if not sums:
        return None
    k = np.array(sorted(sums))
    return k, np.array([sums[key] / counts[key] for key in k])


def forced_shell_dissipation_share(run_dir, config, t_start, t_end):
    """Return (share of D+ in shells centred below the top of the forced shell, None), or (None, why not).

    Each shell loses 2 nu k^(2n) E+(k). nu is common to all shells, so it
    cancels in the share. Shell centres stand in for each mode's k_perp.
    """
    shell = forced_shell(config)
    if shell is None or shell.get("k_sigma_max") is None:
        return None, "the run has no controlled-shell forcing"
    order = hyperviscosity_order(config)
    if order is None:
        return None, "it needs one k_perp-only hyperviscosity on psi and omega"
    spectrum = mean_spectrum(run_dir, "z_plus", t_start, t_end)
    if spectrum is None:
        return None, "spectra.csv has no z_plus spectra in the window"
    k, energy = spectrum
    damping = k ** (2 * order) * energy
    if not damping.sum() > 0.0:
        return None, "the z+ spectrum is empty"
    return float(damping[k < shell["k_sigma_max"]].sum() / damping.sum()), None


def check_ccr_reading(run_dir, config, t_start, t_end):
    """Print whether D+ can be read as the CCR heating of this run."""
    shell = forced_shell(config)
    if shell is not None and "minus" in shell.get("branches", []):
        print("  Warning: z- is forced too, so part of D+ is a cascade that is not CCR.")
    share, reason = forced_shell_dissipation_share(run_dir, config, t_start, t_end)
    if share is None:
        print(f"  Forced-shell check skipped: {reason}.")
        return
    print(f"  D+ in the forced shell and below: {100 * share:.1f}%")
    if share > FORCED_SHELL_WARNING:
        print("  Warning: hyperviscosity damps the forced modes directly, so D+ overstates the CCR cascade.")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_summary(label, run, window, means):
    """Window means, the split of the measured z+ loss, and CCR with the measured z-."""
    print(f"{label}: window t = {run.times[window][0]:g} to {run.times[window][-1]:g}")
    for name, (measured, predicted) in means.items():
        print(f"  {name}  measured {measured:.4g}  estimate {predicted:.4g}  ratio {ratio(measured, predicted):.3g}")
    shares = loss_shares(means)
    if shares is not None:
        print("  share of the measured z+ loss: "
              + ", ".join(f"{name} {100 * share:.0f}%" for name, share in shares.items()))
    from_z_minus = run.q_ccr_from_z_minus[window]
    if np.isfinite(from_z_minus).any():
        print(f"  CCR with the measured z- instead, W+ z-_rms / l_perp: {np.nanmean(from_z_minus):.4g}")


def plot_heating(run, window, means, output_path, *, label="", show=False):
    """DCF and CCR side by side, measured (black) and estimate (red dashed), each on its own y axis."""
    plt = import_pyplot(show=show)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    for ax, (name, (measured_key, predicted_key, measured_label, predicted_label)) in zip(axes, CHANNELS.items()):
        ax.plot(run.times, getattr(run, measured_key), color="black", lw=1.8, label=measured_label)
        ax.plot(run.times, getattr(run, predicted_key), color="tab:red", ls="--", lw=1.6, label=predicted_label)
        ax.axvline(run.times[window][0], color="0.4", lw=1, ls=":")
        ax.axhline(0.0, color="0.5", lw=1, alpha=0.6)
        measured, predicted = means[name]
        title = f"{name} heating of $z^+$"
        if np.isfinite(ratio(measured, predicted)):
            title += f"\nwindow means {measured:.3g} and {predicted:.3g}, ratio {ratio(measured, predicted):.3g}"
        ax.set(xlabel="Time", ylabel="Heating rate", title=title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    t = run.times[window]
    title = (f"{label} | " if label else "") + rf"window $t$ = {t[0]:g} to {t[-1]:g}"
    shares = loss_shares(means)
    if shares is not None:
        title += "\nShare of the measured $z^+$ loss: " + ", ".join(
            f"{name} {100 * share:.0f}%" for name, share in shares.items())
    fig.suptitle(title, fontsize=11)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    finalize_figure(fig, output_path=output_path, show=show, plt=plt)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path, help="Run directory, or the scalar_diagnostics.csv inside it.")
    parser.add_argument("--tmin", type=float, help="Start of the averaging window; all times are plotted.")
    parser.add_argument("--tail-fraction", type=float, default=0.4,
                        help="Fraction of the time span to average when --tmin is not given; default 0.4.")
    parser.add_argument("--output", type=Path, help="Image path; default zplus_heating.png in the run directory.")
    parser.add_argument("--show", action="store_true", help="Show the figure after saving (e.g. in Spyder).")
    return parser


def main(argv=None) -> Path:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_dir = args.path.expanduser().resolve()
    if not run_dir.is_dir():
        run_dir = run_dir.parent  # A path to scalar_diagnostics.csv.
    try:
        run = load_run(run_dir, fields=())
        window = time_window(run.times, tmin=args.tmin, tail_fraction=args.tail_fraction)
    except ValueError as error:
        parser.error(str(error))
    missing = [name for name in ("w_plus", "k_perp_plus", "w_plus_align") if name not in run.columns]
    if missing:
        parser.error(f"Scalar diagnostics are missing {missing}, so there are no estimates.")

    means = window_means(run, window)
    label = f"{run_dir.parent.name}/{run_dir.name}"
    print_summary(label, run, window, means)
    check_ccr_reading(run_dir, read_resolved_config(run_dir), run.times[window][0], run.times[window][-1])
    output_path = (run_dir / "zplus_heating.png" if args.output is None
                   else args.output.expanduser().resolve())
    plot_heating(run, window, means, output_path, label=label, show=args.show)
    return output_path


if __name__ == "__main__":
    main()
