"""Plot a conserved-quantity budget from `scalar_diagnostics.csv`.

The saved RHS terms use the sign convention

`d_t Q = Q_rhs_total = sum(individual signed RHS terms)`.

For the current total-energy budget this means, for example,

- `total_energy_rhs_dissipation < 0` when dissipation removes energy
- `total_energy_rhs_forcing > 0` on average when forcing injects energy

The bottom panel also shows the closure residual

`measured d_t Q - sum(individual saved RHS terms)`

which should remain near zero when the saved budget terms explain the measured
evolution well.

The inhomogeneous equation sets split their background-gradient source into the
three ACR channels `acr_B`, `acr_g` and `acr_th`, which sum exactly to the
single `stratification` source (see
`test_acr_channels_sum_to_stratification_source`). By default those three are
plotted as one `stratification` curve, so the budget reads as
`dissipation + forcing + stratification`; pass `--split-acr` to show the three
channels separately instead. Either way the residual is formed from every saved
term, so it is unaffected.
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
    return list(reader.fieldnames), {name: np.asarray(values, dtype=np.float64) for name, values in columns.items()}


def _backward_difference(time: np.ndarray, values: np.ndarray) -> np.ndarray:
    derivative = np.full_like(values, np.nan, dtype=np.float64)
    if len(values) < 2:
        return derivative
    dt = np.diff(time)
    valid = dt > 0.0
    derivative[1:][valid] = np.diff(values)[valid] / dt[valid]
    return derivative


def _sum_rhs_terms(columns: dict[str, np.ndarray], rhs_term_names: list[str], *, time: np.ndarray) -> np.ndarray:
    """Return the pointwise sum of the individual saved RHS-term columns."""

    if not rhs_term_names:
        return np.zeros_like(time, dtype=np.float64)
    return np.sum([columns[name] for name in rhs_term_names], axis=0, dtype=np.float64)


def _grouped_rhs_terms(
    columns: dict[str, np.ndarray],
    quantity: str,
    rhs_term_names: list[str],
    *,
    split_acr: bool,
) -> list[tuple[str, np.ndarray]]:
    """Return `(label, series)` pairs to draw, collapsing the ACR split by default.

    The `acr_*` channels sum exactly to the modules' single `stratification`
    source, so their sum reproduces the pre-split `*_rhs_stratification` curve.
    Equation sets that still write `stratification` directly are untouched,
    since they have no `acr_*` columns to group.
    """

    prefix = f"{quantity}_rhs_"
    acr_names = [name for name in rhs_term_names if name.startswith(f"{prefix}acr_")]
    if split_acr or not acr_names:
        return [(name, columns[name]) for name in rhs_term_names]

    grouped = [(name, columns[name]) for name in rhs_term_names if name not in acr_names]
    stratification = np.sum([columns[name] for name in acr_names], axis=0, dtype=np.float64)
    grouped.append((f"{prefix}stratification", stratification))
    return grouped


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", help="Path to scalar_diagnostics.csv.")
    parser.add_argument(
        "--split-acr",
        action="store_true",
        help=(
            "Plot the `acr_B`, `acr_g` and `acr_th` channels separately instead of summing "
            "them into a single `stratification` curve."
        ),
    )
    parser.add_argument(
        "--quantity",
        default="total_energy",
        help="Conserved quantity name prefix, for example `total_energy`.",
    )
    parser.add_argument("--output", default=None, help="Output image path. Defaults next to the CSV file.")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the figure interactively after saving. Useful from Spyder or IPython.",
    )
    return parser


def main(argv: list[str] | None = None) -> Path:
    args = build_parser().parse_args(argv)
    plt = import_pyplot(show=args.show)
    csv_path = Path(args.csv_path).expanduser().resolve()
    fieldnames, columns = _read_scalar_csv(csv_path)
    time_key = "time" if "time" in columns else "t"
    time = columns[time_key]

    if args.quantity not in columns:
        raise SystemExit(f"Quantity {args.quantity!r} is not present in {csv_path}.")

    rhs_total_name = f"{args.quantity}_rhs_total"
    if rhs_total_name not in columns:
        raise SystemExit(f"Budget total column {rhs_total_name!r} is not present in {csv_path}.")

    rhs_term_names = sorted(
        name
        for name in fieldnames
        if name.startswith(f"{args.quantity}_rhs_") and name != rhs_total_name
    )

    plotted_terms = _grouped_rhs_terms(
        columns,
        args.quantity,
        rhs_term_names,
        split_acr=args.split_acr,
    )

    measured = _backward_difference(time, columns[args.quantity])
    rhs_sum = _sum_rhs_terms(columns, rhs_term_names, time=time)
    residual = measured - rhs_sum
    output_path = (
        csv_path.with_name(f"{args.quantity}_budget.png")
        if args.output is None
        else Path(args.output).expanduser().resolve()
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 7.0), constrained_layout=True, sharex=True)

    axes[0].plot(time, columns[args.quantity], lw=2, color="black", label=args.quantity)
    axes[0].set_ylabel(args.quantity)
    axes[0].set_title(f"{args.quantity} and budget comparison")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].plot(
        time,
        measured,
        lw=1.8,
        ls="--",
        color="0.35",
        label=rf"measured d$_t$ {args.quantity}",
    )
    axes[1].plot(
        time,
        columns[rhs_total_name],
        lw=2.0,
        color="0.45",
        ls="-",
        label=f"saved {rhs_total_name}",
    )
    term_linestyles = ["--", ":", "-."]
    for index, (term_label, term_series) in enumerate(plotted_terms):
        axes[1].plot(
            time,
            term_series,
            lw=1.8,
            ls=term_linestyles[index % len(term_linestyles)],
            label=term_label,
        )
    axes[1].plot(
        time,
        residual,
        lw=3.0,
        ls="-",
        color="black",
        label=rf"closure residual: measured d$_t$ {args.quantity} - sum(saved RHS terms)",
    )
    axes[1].axhline(0.0, color="0.4", lw=1.0, alpha=0.6)
    axes[1].set_xlabel("time")
    axes[1].set_ylabel(r"budget terms / d$_t Q$")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    finalize_figure(fig, output_path=output_path, show=args.show, plt=plt)
    return output_path


if __name__ == "__main__":
    main()
