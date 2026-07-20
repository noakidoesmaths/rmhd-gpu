"""Fit the spectral slope of tidy/long spectra output to measure `alpha`.

Plots each quantity from `spectra.csv` like `vis/plot_spectra.py`, then fits
`log10 E` vs `log10 kperp` for one snapshot and reports the slope `s`.

For the `random_spectrum` / `random_spectrum_one_wave` initial conditions the
shaped field amplitude goes as `n_perp^(-alpha/2)`, the saved `u_perp`/`b_perp`
spectra are shell sums of `0.5 * kperp^2 * |phi_hat|^2`, and the annulus mode
count contributes one more power of kperp, so the plotted spectrum satisfies

    E(kperp) ~ kperp^(3 - alpha)   i.e.   alpha = 3 - s.

The raw slope `s` is always reported alongside the implied alpha; the alpha
conversion assumes the `u_perp`/`b_perp` convention above.
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", help="Path to spectra.csv.")
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
        "--fit-range",
        nargs=2,
        type=float,
        default=None,
        metavar=("KMIN", "KMAX"),
        help=(
            "kperp range for the slope fit. Defaults to every positive-valued "
            "point of the fitted snapshot."
        ),
    )
    parser.add_argument(
        "--fit-time",
        type=float,
        default=None,
        help=(
            "Snapshot time to fit (nearest available output time is used). "
            "Defaults to the latest time. Use 0 to fit the initial spectrum."
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help=(
            "Expected initial-condition alpha. Draws a guide line with slope "
            "3 - alpha (the u_perp/b_perp shell-spectrum convention: modal "
            "|phi|^2 ~ k^-alpha, +2 from kperp^2, +1 from the annulus sum). "
            "Without this a k^(-5/3) guide is drawn."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show figures interactively after saving. Useful from Spyder or IPython.",
    )
    return parser


def _fit_slope(
    kperp: np.ndarray,
    values: np.ndarray,
    fit_range: tuple[float, float] | None,
) -> tuple[float, float, np.ndarray, np.ndarray] | None:
    """Fit log10(values) vs log10(kperp); return (slope, stderr, k_fit, y_fit)."""

    valid = (kperp > 0.0) & (values > 0.0)
    if fit_range is not None:
        kmin, kmax = sorted(fit_range)
        valid &= (kperp >= kmin) & (kperp <= kmax)
    if np.count_nonzero(valid) < 3:
        return None

    log_k = np.log10(kperp[valid])
    log_e = np.log10(values[valid])
    slope, intercept = np.polyfit(log_k, log_e, 1)

    residuals = log_e - (slope * log_k + intercept)
    denom = np.sum((log_k - np.mean(log_k)) ** 2)
    stderr = float(np.sqrt(np.sum(residuals**2) / (log_k.size - 2) / denom))

    k_fit = kperp[valid]
    y_fit = 10.0 ** (slope * np.log10(k_fit) + intercept)
    return float(slope), stderr, k_fit, y_fit


def _guide_line(
    kperp: np.ndarray,
    values: np.ndarray,
    slope: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    valid = (kperp > 0.0) & (values > 0.0)
    if np.count_nonzero(valid) < 2:
        return None

    k_plot = kperp[valid]
    y_plot = values[valid]
    anchor = len(k_plot) // 2
    guide = y_plot[anchor] * (k_plot / k_plot[anchor]) ** slope
    return k_plot, guide


def main(argv: list[str] | None = None) -> list[Path]:
    args = build_parser().parse_args(argv)
    plt = import_pyplot(show=args.show)
    from matplotlib import cm, colors

    csv_path = Path(args.csv_path).expanduser().resolve()
    spectra = _read_spectra_csv(csv_path)

    quantities = sorted(spectra) if args.quantities is None else list(args.quantities)
    missing = [name for name in quantities if name not in spectra]
    if missing:
        raise SystemExit(f"Unknown spectra quantities requested: {missing}.")

    output_dir = (
        csv_path.with_name("spectra_slope_plots")
        if args.output_dir is None
        else Path(args.output_dir).expanduser().resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.alpha is not None:
        guide_slope = 3.0 - args.alpha
        guide_label = rf"$k^{{3-\alpha}}$, $\alpha={args.alpha:g}$ ($s={guide_slope:.2f}$)"
    else:
        guide_slope = -5.0 / 3.0
        guide_label = r"$k^{-5/3}$ guide"

    times = sorted({time_value for by_time in spectra.values() for time_value in by_time})
    time_norm = colors.Normalize(vmin=min(times), vmax=max(times) if len(times) > 1 else min(times) + 1.0)
    cmap = plt.get_cmap(args.colormap)
    saved_paths: list[Path] = []

    for quantity in quantities:
        fig, ax = plt.subplots(figsize=(6.5, 4.8), constrained_layout=True)
        ymax = 0.0
        for time_value in sorted(spectra[quantity]):
            kperp, values = spectra[quantity][time_value]
            mask = (kperp > 0.0) & (values > 0.0)
            if np.any(mask):
                ymax = max(ymax, float(np.max(values[mask])))
            ax.loglog(kperp[mask], values[mask], color=cmap(time_norm(time_value)), lw=2)

        quantity_times = sorted(spectra[quantity])
        if args.fit_time is None:
            fit_time = quantity_times[-1]
        else:
            fit_time = min(quantity_times, key=lambda t: abs(t - args.fit_time))
        fit_curve = spectra[quantity][fit_time]

        fit_range = None if args.fit_range is None else tuple(args.fit_range)
        fit = _fit_slope(*fit_curve, fit_range=fit_range)
        if fit is not None:
            slope, stderr, k_fit, y_fit = fit
            implied_alpha = 3.0 - slope
            ax.loglog(
                k_fit,
                y_fit,
                color="tab:red",
                ls="--",
                lw=1.8,
                label=(
                    rf"fit ($t={fit_time:.3g}$): $s={slope:.3f}\pm{stderr:.3f}$"
                    rf" $\rightarrow$ $\alpha=3-s={implied_alpha:.2f}$"
                ),
            )
            fit_span = (
                f"k in [{k_fit.min():g}, {k_fit.max():g}]"
                if fit_range is None
                else f"k in [{fit_range[0]:g}, {fit_range[1]:g}]"
            )
            print(
                f"{quantity}: slope s={slope:.4f} +/- {stderr:.4f}, "
                f"implied alpha=3-s={implied_alpha:.4f} (t={fit_time:g}, fit {fit_span})"
            )
        else:
            print(f"{quantity}: not enough points to fit a slope (t={fit_time:g}).")

        guide_line = _guide_line(*fit_curve, slope=guide_slope)
        if guide_line is not None:
            guide_k, guide_y = guide_line
            ax.loglog(guide_k, guide_y, color="0.55", ls=":", lw=1.6, label=guide_label)

        sm = cm.ScalarMappable(norm=time_norm, cmap=cmap)
        colorbar = fig.colorbar(sm, ax=ax)
        colorbar.set_label("time")

        ax.set_xlabel(r"$k_\perp$")
        ax.set_ylabel("value")
        ax.set_title(quantity)
        ax.grid(True, alpha=0.25)
        if args.y_span_decades > 0.0 and ymax > 0.0:
            ax.set_ylim(ymax / (10.0 ** args.y_span_decades), ymax)
        if fit is not None or guide_line is not None:
            ax.legend(fontsize=8)

        output_path = output_dir / f"{quantity}.png"
        finalize_figure(fig, output_path=output_path, show=args.show, plt=plt)
        saved_paths.append(output_path)

    return saved_paths


if __name__ == "__main__":
    main()
