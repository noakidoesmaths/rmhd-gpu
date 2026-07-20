"""Plot parallel (kz) spectra output saved by `rmhdgpu.run` to `spectra_prl.csv`.

Each quantity is plotted as E(k_par) with one curve per snapshot, colored by
time. The x-axis is logarithmic by default, which renders a power law as a
straight line and is clearest for eyeballing the slope (use `--linear` for a
linear x-axis, which keeps the k_par = 0 shell visible).

For the earliest snapshot the script also prints which parallel shells carry
energy, so an initial-condition band like `n_min_prl <= n_z <= n_max_prl` can
be checked directly (with the default Lz = 2*pi, k_par equals the integer
parallel mode number n_z).

A log-log slope `s` is fitted to one snapshot (the earliest by default, i.e.
the initial condition) and reported per quantity. For the
`random_spectrum_one_wave` initial condition the shaped amplitude goes as
`n_prl^(-alpha_prl/2)` and the perpendicular sum is identical for every k_par
shell, so the parallel spectrum satisfies

    E(k_par) ~ k_par^(-alpha_prl)   i.e.   alpha_prl = -s.

Pass `--alpha-prl` to overlay the expected guide line of slope `-alpha_prl`.
(Note: this is the *parallel* shaping exponent `alpha_prl`; the separate
`alpha` parameter shapes only the perpendicular spectrum, slope `3 - alpha`.)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from vis._matplotlib import finalize_figure, import_pyplot
from vis.plot_spectra import _read_spectra_csv
from vis.plot_spectra_slope import _fit_slope, _guide_line

_ENERGY_FLOOR_RELATIVE = 1.0e-12


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", help="Path to spectra_prl.csv.")
    parser.add_argument(
        "--quantities",
        nargs="*",
        default=None,
        help="Quantities to plot. Defaults to all quantities in the file.",
    )
    parser.add_argument("--output-dir", default=None, help="Directory for PNG outputs.")
    parser.add_argument("--colormap", default="viridis", help="Matplotlib colormap name.")
    parser.add_argument(
        "--y-span-decades",
        type=float,
        default=7.0,
        help=(
            "Limit each plot to at most 10**N in y from the quantity's maximum. "
            "Default: 7. Use 0 or a negative value to disable the limit."
        ),
    )
    parser.add_argument(
        "--linear",
        action="store_true",
        help=(
            "Use a linear x-axis (keeps the k_par = 0 shell). By default the "
            "x-axis is logarithmic, which renders a power law as a straight line."
        ),
    )
    parser.add_argument(
        "--alpha-prl",
        type=float,
        default=None,
        help=(
            "Expected parallel shaping exponent alpha_prl. Draws a guide line of "
            "slope -alpha_prl (E(k_par) ~ k_par^-alpha_prl). Without it no guide "
            "is drawn."
        ),
    )
    parser.add_argument(
        "--fit-time",
        type=float,
        default=None,
        help=(
            "Snapshot time to fit (nearest available output time is used). "
            "Defaults to the earliest time, i.e. the initial condition."
        ),
    )
    parser.add_argument(
        "--fit-range",
        nargs=2,
        type=float,
        default=None,
        metavar=("KMIN", "KMAX"),
        help=(
            "k_par range for the slope fit. Defaults to every occupied "
            "(positive-valued) shell of the fitted snapshot."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show figures interactively after saving. Useful from Spyder or IPython.",
    )
    return parser


def _report_initial_modes(
    quantity: str,
    kprl: np.ndarray,
    values: np.ndarray,
    time_value: float,
) -> None:
    """Print which parallel shells carry energy at the earliest snapshot."""

    if float(np.sum(values)) <= 0.0:
        print(f"{quantity}: no energy at t={time_value:g}.")
        return

    occupied = values > _ENERGY_FLOOR_RELATIVE * float(np.max(values))
    occupied_shells = [f"{k_value:g}" for k_value in np.sort(kprl[occupied])]
    print(
        f"{quantity}: t={time_value:g} energy in parallel shells k_par = "
        f"[{', '.join(occupied_shells)}]"
    )


def main(argv: list[str] | None = None) -> list[Path]:
    args = build_parser().parse_args(argv)
    plt = import_pyplot(show=args.show)
    from matplotlib import cm, colors

    csv_path = Path(args.csv_path).expanduser().resolve()
    spectra = _read_spectra_csv(csv_path, k_column="kprl")

    quantities = sorted(spectra) if args.quantities is None else list(args.quantities)
    missing = [name for name in quantities if name not in spectra]
    if missing:
        raise SystemExit(f"Unknown spectra quantities requested: {missing}.")

    output_dir = (
        csv_path.with_name("spectra_prl_plots")
        if args.output_dir is None
        else Path(args.output_dir).expanduser().resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    times = sorted({time_value for by_time in spectra.values() for time_value in by_time})
    time_norm = colors.Normalize(vmin=min(times), vmax=max(times) if len(times) > 1 else min(times) + 1.0)
    cmap = plt.get_cmap(args.colormap)
    saved_paths: list[Path] = []

    for quantity in quantities:
        fig, ax = plt.subplots(figsize=(6.5, 4.8), constrained_layout=True)
        ymax = 0.0
        for time_value in sorted(spectra[quantity]):
            kprl, values = spectra[quantity][time_value]
            mask = values > 0.0
            if not args.linear:
                mask &= kprl > 0.0
            if np.any(mask):
                ymax = max(ymax, float(np.max(values[mask])))
            ax.plot(kprl[mask], values[mask], color=cmap(time_norm(time_value)), lw=2)

        earliest_time = min(spectra[quantity])
        _report_initial_modes(quantity, *spectra[quantity][earliest_time], earliest_time)

        quantity_times = sorted(spectra[quantity])
        if args.fit_time is None:
            fit_time = quantity_times[0]
        else:
            fit_time = min(quantity_times, key=lambda t: abs(t - args.fit_time))
        fit_curve = spectra[quantity][fit_time]

        fit_range = None if args.fit_range is None else tuple(args.fit_range)
        fit = _fit_slope(*fit_curve, fit_range=fit_range)
        if fit is not None:
            slope, stderr, k_fit, y_fit = fit
            implied_alpha_prl = -slope
            ax.plot(
                k_fit,
                y_fit,
                color="tab:red",
                ls="--",
                lw=1.8,
                label=(
                    rf"fit ($t={fit_time:.3g}$): $s={slope:.3f}\pm{stderr:.3f}$"
                    rf" $\rightarrow$ $\alpha_\parallel=-s={implied_alpha_prl:.2f}$"
                ),
            )
            fit_span = (
                f"k_par in [{k_fit.min():g}, {k_fit.max():g}]"
                if fit_range is None
                else f"k_par in [{fit_range[0]:g}, {fit_range[1]:g}]"
            )
            print(
                f"{quantity}: slope s={slope:.4f} +/- {stderr:.4f}, "
                f"implied alpha_prl=-s={implied_alpha_prl:.4f} (t={fit_time:g}, fit {fit_span})"
            )
        else:
            print(f"{quantity}: not enough points to fit a slope (t={fit_time:g}).")

        guide_line = None
        if args.alpha_prl is not None:
            guide_line = _guide_line(*fit_curve, slope=-args.alpha_prl)
            if guide_line is not None:
                guide_k, guide_y = guide_line
                ax.plot(
                    guide_k,
                    guide_y,
                    color="0.55",
                    ls=":",
                    lw=1.6,
                    label=(
                        rf"$k_\parallel^{{-\alpha_\parallel}}$, "
                        rf"$\alpha_\parallel={args.alpha_prl:g}$ ($s={-args.alpha_prl:.2f}$)"
                    ),
                )

        # A quantity with no positive values anywhere (e.g. a field that is
        # identically zero for this run) cannot be log-scaled.
        if ymax > 0.0:
            ax.set_yscale("log")
        if not args.linear:
            ax.set_xscale("log")

        sm = cm.ScalarMappable(norm=time_norm, cmap=cmap)
        colorbar = fig.colorbar(sm, ax=ax)
        colorbar.set_label("time")

        ax.set_xlabel(r"$k_\parallel$")
        ax.set_ylabel("value")
        ax.set_title(quantity)
        ax.grid(True, alpha=0.25)
        if args.y_span_decades > 0.0 and ymax > 0.0:
            ax.set_ylim(ymax / (10.0 ** args.y_span_decades), ymax * 3.0)
        if fit is not None or guide_line is not None:
            ax.legend(fontsize=8)

        output_path = output_dir / f"{quantity}.png"
        finalize_figure(fig, output_path=output_path, show=args.show, plt=plt)
        saved_paths.append(output_path)

    return saved_paths


if __name__ == "__main__":
    main()
