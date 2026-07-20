"""Plot tidy/long spectra output saved by `rmhdgpu.run`."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from vis._matplotlib import finalize_figure, import_pyplot


def _read_spectra_csv(
    path: Path, k_column: str = "kperp"
) -> dict[str, dict[float, tuple[np.ndarray, np.ndarray]]]:
    grouped: dict[str, dict[float, list[tuple[float, float]]]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Spectra file {path} has no header row.")
        if k_column not in reader.fieldnames:
            raise ValueError(f"Spectra file {path} has no '{k_column}' column.")
        rows = list(reader)

    if not rows:
        raise ValueError(f"Spectra file {path} contains no data rows.")

    for row in rows:
        quantity = row["quantity"]
        time_value = float(row["time"])
        grouped.setdefault(quantity, {}).setdefault(time_value, []).append(
            (float(row[k_column]), float(row["value"]))
        )

    result: dict[str, dict[float, tuple[np.ndarray, np.ndarray]]] = {}
    for quantity, by_time in grouped.items():
        result[quantity] = {}
        for time_value, pairs in by_time.items():
            data = np.asarray(sorted(pairs, key=lambda item: item[0]), dtype=np.float64)
            result[quantity][time_value] = (data[:, 0], data[:, 1])
    return result


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
        "--show",
        action="store_true",
        help="Show figures interactively after saving. Useful from Spyder or IPython.",
    )
    return parser


def _guide_line(kperp: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    valid = (kperp > 0.0) & (values > 0.0)
    if np.count_nonzero(valid) < 2:
        return None

    k_plot = kperp[valid]
    y_plot = values[valid]
    anchor = len(k_plot) // 2
    guide = y_plot[anchor] * (k_plot / k_plot[anchor]) ** (-5.0 / 3.0)
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
        csv_path.with_name("spectra_plots")
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
        latest_time = max(spectra[quantity])
        latest_curve = spectra[quantity][latest_time]
        for time_value in sorted(spectra[quantity]):
            kperp, values = spectra[quantity][time_value]
            mask = (kperp > 0.0) & (values > 0.0)
            if np.any(mask):
                ymax = max(ymax, float(np.max(values[mask])))
            ax.loglog(kperp[mask], values[mask], color=cmap(time_norm(time_value)), lw=2)

        guide_line = _guide_line(*latest_curve)
        if guide_line is not None:
            guide_k, guide_y = guide_line
            ax.loglog(guide_k, guide_y, color="0.55", ls=":", lw=1.6, label=r"$k^{-5/3}$ guide")

        sm = cm.ScalarMappable(norm=time_norm, cmap=cmap)
        colorbar = fig.colorbar(sm, ax=ax)
        colorbar.set_label("time")

        ax.set_xlabel(r"$k_\perp$")
        ax.set_ylabel("value")
        ax.set_title(quantity)
        ax.grid(True, alpha=0.25)
        if args.y_span_decades > 0.0 and ymax > 0.0:
            ax.set_ylim(ymax / (10.0 ** args.y_span_decades), ymax)
        if guide_line is not None:
            ax.legend(fontsize=8)

        output_path = output_dir / f"{quantity}.png"
        finalize_figure(fig, output_path=output_path, show=args.show, plt=plt)
        saved_paths.append(output_path)

    return saved_paths


if __name__ == "__main__":
    main()
