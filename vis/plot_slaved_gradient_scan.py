"""Compare saturated slaved amplitudes across runs with different background gradients.

`plot_slaved_projection.py` tests the Squire et al. (arXiv:2607.08036) Eq. (47) slaving
closure inside one run. This script does it across runs: vary a background gradient, then
plot the saturated RMS of each compressive field against the drive that field actually feels.

There is no single "gradient". Each field is driven by its own coefficient, the x component
of Eq. (46), which `plot_slaved_projection.forcings` returns:

    drho    F = g/(vA^2 (1 + chi)) - K_rho0        (= -N^2/g)
    du_par  F = -K_b0
    db_par  F = -K_b0 + K_p0/gamma

with `K_b0 = g/vA^2 - chi K_p0/gamma`. So `K_rho0` moves only the density drive, `K_p0`
moves only the two magnetic ones (it cancels out of N^2), and `g` moves all three plus the
explicit buoyancy term. Eq. (47),

    predicted_rms = |F| * l_perp * rms(z_x)/z_rms

is linear in F, with the last two factors measured from the run, so slaving predicts a
straight line through the origin on the amplitude-against-|F| panel.

Each run becomes one number per field: the mean over a settled window at the end of the run.
A least-squares line across that window gives a fractional drift. Points that are still
drifting, that ran with unstable stratification (N^2 <= 0), or that stopped before `tmax` are
drawn hollow and left out of the fits.

Amplitudes are in Eq. (47) units, that is, already divided by `measured_divisors`: `db_par`
is stored as (vA^2/vS^2) delta B_par/B_0, not as the raw saved field.

Examples (also usable as main([...]) in Spyder):
    python vis/plot_slaved_gradient_scan.py RUN_DIR_1 RUN_DIR_2 ...
    python vis/plot_slaved_gradient_scan.py RUN_DIRS --parameter K_rho0 --with-parity
    python vis/plot_slaved_gradient_scan.py --from-summary slaved_gradient_scan.csv --show
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, fields as dataclass_fields
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
from vis.plot_slaved_projection import (
    COLORS,
    LABELS,
    forcings,
    load_parameters,
    read_from_csv,
    read_from_snapshots,
)


FIELDS = ("drho", "du_par", "db_par")

# The background gradients that can be scanned. `K_b0` is derived from g and K_p0, so it is
# not a knob; see `rmhdgpu.equations.rmhd_by_nokia_rho.derived_parameters`.
KNOBS = ("g", "K_p0", "K_rho0")
KNOB_COLORS = {"g": "tab:red", "K_p0": "tab:blue", "K_rho0": "tab:green", "none": "tab:gray"}

# Statistics defaults. One window rule and one tolerance, applied to every series alike.
DEFAULT_TAIL_FRACTION = 0.4
DEFAULT_DRIFT_TOLERANCE = 0.2
MIN_TAIL_SAMPLES = 8

# A run reaching at least this fraction of its `[time] tmax` is treated as having finished.
COMPLETION_FRACTION = 0.98

# `status` is "ok" only when every check passes; the others say which one failed.
STATUS_OK = "ok"
STATUS_DRIFTING = "drifting"
STATUS_UNSTABLE = "unstable_stratification"
STATUS_ENDED_EARLY = "ended_early"
STATUS_TOO_FEW = "too_few_samples"
STATUS_UNREADABLE = "unreadable"

EXCLUDED_LABELS = {
    STATUS_DRIFTING: "still drifting, excluded",
    STATUS_UNSTABLE: "unstable stratification, excluded",
    STATUS_ENDED_EARLY: "ended early, excluded",
    STATUS_TOO_FEW: "too few samples, excluded",
    STATUS_UNREADABLE: "unreadable, excluded",
}


@dataclass
class Saturated:
    """One series reduced to its settled-window statistics."""

    mean: float
    std: float
    drift: float  # Fractional change across the window, from a least-squares line.
    n_samples: int
    t_start: float
    t_end: float
    settled: bool


def saturated_value(
    times,
    values,
    *,
    tail_fraction=DEFAULT_TAIL_FRACTION,
    tmin=None,
    drift_tolerance=DEFAULT_DRIFT_TOLERANCE,
    min_samples=MIN_TAIL_SAMPLES,
):
    """Average a series over its settled window and measure how much it still drifts.

    The window is chosen by time, not by sample count, so it survives a variable `dt` and
    strided snapshots. `drift` is the change a least-squares line predicts across the window
    divided by the window mean, which makes it scale free.
    """

    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(times) & np.isfinite(values)
    nan = float("nan")
    if not finite.any():
        return Saturated(nan, nan, nan, 0, nan, nan, False)

    finite_times = times[finite]
    if tmin is not None:
        t_start = float(tmin)
    else:
        span = float(finite_times[-1] - finite_times[0])
        t_start = float(finite_times[-1]) - tail_fraction * span

    window = finite & (times >= t_start)
    if not window.any():
        return Saturated(nan, nan, nan, 0, t_start, nan, False)

    window_times = times[window]
    window_values = values[window]
    mean = float(window_values.mean())
    std = float(window_values.std())

    if window_values.size >= 2 and window_times[-1] > window_times[0] and mean != 0.0:
        slope = float(np.polyfit(window_times, window_values, 1)[0])
        drift = slope * float(window_times[-1] - window_times[0]) / mean
    elif mean == 0.0:
        # An all-zero series (no drive at all) is flat, but its drift is not a ratio.
        drift = 0.0 if np.allclose(window_values, 0.0) else float("inf")
    else:
        drift = 0.0

    settled = bool(
        window_values.size >= min_samples and mean > 0.0 and abs(drift) <= drift_tolerance
    )
    return Saturated(
        mean=mean,
        std=std,
        drift=drift,
        n_samples=int(window_values.size),
        t_start=t_start,
        t_end=float(window_times[-1]),
        settled=settled,
    )


@dataclass
class ScanPoint:
    """One row of the summary table: a single field of a single run."""

    run_dir: str
    run_name: str
    field: str
    scanned_parameter: str
    scanned_value: float
    vA: float
    chi: float
    alpha: float
    gamma: float
    g: float
    K_p0: float
    K_rho0: float
    K_b0: float
    N_sq: float
    forcing: float
    source: str
    branch: str
    t_start: float
    t_end: float
    n_samples: int
    measured_mean: float
    measured_std: float
    measured_drift: float
    predicted_mean: float
    predicted_std: float
    predicted_drift: float
    l_perp_mean: float
    alignment_mean: float
    chi_a_mean: float
    settled: bool
    stable: bool
    status: str

    @property
    def abs_forcing(self):
        return abs(self.forcing)

    @property
    def slaving_constant(self):
        """Measured over predicted: the ratio of window means, not the mean of ratios.

        The mean of instantaneous ratios blows up whenever `predicted` passes near zero,
        which it does when `l_perp` is undefined or the Alfvenic field vanishes.
        """

        if self.predicted_mean > 0.0 and np.isfinite(self.measured_mean):
            return self.measured_mean / self.predicted_mean
        return float("nan")


# Derived quantities are appended, so the dataclass stays the single source of the schema.
CSV_COLUMNS = [field.name for field in dataclass_fields(ScanPoint)] + [
    "abs_forcing",
    "slaving_constant",
]


def _read_tmax(run_dir):
    """Return the run's requested `[time] tmax`, or None when it cannot be read."""

    path = Path(run_dir) / "input_copy.input"
    if not path.is_file():
        return None
    with path.open("rb") as handle:
        document = tomllib.load(handle)
    tmax = document.get("time", {}).get("tmax")
    return None if tmax is None else float(tmax)


def _csv_fields(run_dir, fields):
    """Keep only the fields whose RMS column exists, so one gap does not lose the run."""

    path = Path(run_dir) / "scalar_diagnostics.csv"
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle), [])
    return [name for name in fields if f"{name}_rms" in header]


def _points_for_run(
    run_dir,
    fields,
    *,
    from_snapshots=False,
    branch="plus",
    stride=1,
    window_kwargs=None,
):
    """Reduce one run directory to one ScanPoint per requested field."""

    run_dir = Path(run_dir)
    window_kwargs = dict(window_kwargs or {})
    min_samples = window_kwargs.get("min_samples", MIN_TAIL_SAMPLES)
    try:
        p = load_parameters(run_dir)
    except SystemExit as error:
        # Without [physics] there is no forcing, so the run cannot enter the comparison.
        print(f"Skipping {run_dir.name}: {error}")
        return []

    forcing = forcings(p)
    nan = float("nan")
    common = dict(
        run_dir=str(run_dir),
        run_name=run_dir.name,
        scanned_parameter="none",
        scanned_value=nan,
        vA=p.vA, chi=p.chi, alpha=p.alpha, gamma=p.gamma,
        g=p.g, K_p0=p.K_p0, K_rho0=p.K_rho0, K_b0=p.K_b0, N_sq=p.N_sq,
        stable=p.N_sq > 0.0,
    )

    def unreadable(names, reason):
        print(f"Skipping {run_dir.name}: {reason}")
        return [
            ScanPoint(
                field=name, forcing=forcing[name], source="none", branch="none",
                t_start=nan, t_end=nan, n_samples=0,
                measured_mean=nan, measured_std=nan, measured_drift=nan,
                predicted_mean=nan, predicted_std=nan, predicted_drift=nan,
                l_perp_mean=nan, alignment_mean=nan, chi_a_mean=nan,
                settled=False, status=STATUS_UNREADABLE, **common,
            )
            for name in names
        ]

    if from_snapshots:
        available = list(fields)
    else:
        available = _csv_fields(run_dir, fields)
        if not available:
            return unreadable(fields, "no scalar diagnostics with the requested RMS columns")

    try:
        if from_snapshots:
            run = read_from_snapshots(run_dir, available, p, branch, stride)
        else:
            run = read_from_csv(run_dir, available, p)
    except SystemExit as error:
        return unreadable(fields, str(error))

    tmax = _read_tmax(run_dir)
    ended_early = (
        tmax is not None
        and run.times.size > 0
        and float(run.times[-1]) < COMPLETION_FRACTION * tmax
    )

    l_perp = saturated_value(run.times, run.l_perp, **window_kwargs)
    alignment = saturated_value(run.times, run.alignment, **window_kwargs)
    chi_a = saturated_value(run.times, run.chi_a, **window_kwargs)

    points = []
    for name in fields:
        if name not in available:
            points.extend(unreadable([name], f"no {name}_rms column"))
            continue
        measured = saturated_value(run.times, run.measured[name], **window_kwargs)
        predicted = saturated_value(run.times, run.predicted[name], **window_kwargs)

        if not common["stable"]:
            status = STATUS_UNSTABLE
        elif ended_early:
            status = STATUS_ENDED_EARLY
        elif measured.n_samples < min_samples:
            status = STATUS_TOO_FEW
        elif not measured.settled:
            status = STATUS_DRIFTING
        else:
            status = STATUS_OK

        points.append(
            ScanPoint(
                field=name,
                forcing=forcing[name],
                source=run.source,
                branch=run.branch or "none",
                t_start=measured.t_start,
                t_end=measured.t_end,
                n_samples=measured.n_samples,
                measured_mean=measured.mean,
                measured_std=measured.std,
                measured_drift=measured.drift,
                predicted_mean=predicted.mean,
                predicted_std=predicted.std,
                predicted_drift=predicted.drift,
                l_perp_mean=l_perp.mean,
                alignment_mean=alignment.mean,
                chi_a_mean=chi_a.mean,
                settled=(status == STATUS_OK),
                status=status,
                **common,
            )
        )
    return points


def infer_scanned_parameter(points):
    """Pick the knob that varies most across the supplied runs.

    A scan moves exactly one of `g`, `K_p0`, `K_rho0` and holds the other two, so the knob
    with the most distinct values is the scan axis. Returns "none" when nothing varies.
    """

    per_run = list({point.run_dir: point for point in points}.values())
    counts = {knob: len({getattr(point, knob) for point in per_run}) for knob in KNOBS}
    best = max(KNOBS, key=lambda knob: counts[knob])
    return best if counts[best] > 1 else "none"


def label_scanned_parameter(points, scanned_parameter=None):
    """Stamp each point with the knob it was scanned over and that knob's value."""

    if not points:
        return points
    if scanned_parameter is None:
        scanned_parameter = infer_scanned_parameter(points)
    for point in points:
        point.scanned_parameter = scanned_parameter
        point.scanned_value = (
            float("nan") if scanned_parameter == "none"
            else float(getattr(point, scanned_parameter))
        )
    return points


def scan_points_from_runs(
    run_dirs,
    fields=FIELDS,
    *,
    scanned_parameter=None,
    from_snapshots=False,
    branch="plus",
    stride=1,
    tmin=None,
    tail_fraction=DEFAULT_TAIL_FRACTION,
    drift_tolerance=DEFAULT_DRIFT_TOLERANCE,
    min_samples=MIN_TAIL_SAMPLES,
):
    """Reduce every run directory to ScanPoints and label them with the scanned knob."""

    window_kwargs = dict(
        tmin=tmin,
        tail_fraction=tail_fraction,
        drift_tolerance=drift_tolerance,
        min_samples=min_samples,
    )
    points = []
    for run_dir in run_dirs:
        points.extend(
            _points_for_run(
                run_dir, list(fields),
                from_snapshots=from_snapshots, branch=branch, stride=stride,
                window_kwargs=window_kwargs,
            )
        )
    return label_scanned_parameter(points, scanned_parameter)


def write_summary_csv(points, path):
    """Save the reduced scan so the figure can be redrawn without re-reading any run."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for point in points:
            row = {name: getattr(point, name) for name in CSV_COLUMNS}
            if not np.isfinite(row["slaving_constant"]):
                row["slaving_constant"] = ""  # Undefined when there is no drive.
            writer.writerow(row)
    print(f"Saved {path}")
    return path


def read_summary_csv(path):
    """Load a summary written by `write_summary_csv`; derived columns are recalculated."""

    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"{path} contains no data rows.")

    casts = {field.name: field.type for field in dataclass_fields(ScanPoint)}
    points = []
    for row in rows:
        values = {}
        for name, kind in casts.items():
            raw = row.get(name, "")
            if kind == "bool":
                values[name] = raw == "True"
            elif kind == "int":
                values[name] = int(raw or 0)
            elif kind == "float":
                values[name] = float(raw) if raw not in {"", "None"} else float("nan")
            else:
                values[name] = raw
        points.append(ScanPoint(**values))
    return points


def geometric_mean(values):
    """Geometric mean and log-spread; `C` spans decades, so the average is multiplicative."""

    values = np.asarray([v for v in values if np.isfinite(v) and v > 0.0], dtype=float)
    if values.size == 0:
        return float("nan"), float("nan")
    logs = np.log(values)
    return float(np.exp(logs.mean())), float(np.exp(logs.std()))


def _plot_markers(ax, points, x_of, y_of):
    """Draw settled points filled and coloured by knob, and every rejection hollow and red."""

    styles = {
        STATUS_OK: dict(marker="o", ls="none", ms=7),
        STATUS_DRIFTING: dict(marker="o", ls="none", ms=8, mfc="none", mew=1.6),
        STATUS_UNSTABLE: dict(marker="x", ls="none", ms=8, mew=1.8),
        STATUS_ENDED_EARLY: dict(marker="s", ls="none", ms=8, mfc="none", mew=1.6),
        STATUS_TOO_FEW: dict(marker="v", ls="none", ms=8, mfc="none", mew=1.6),
        STATUS_UNREADABLE: dict(marker="v", ls="none", ms=8, mfc="none", mew=1.6),
    }
    for status, style in styles.items():
        group = [point for point in points if point.status == status]
        if not group:
            continue
        if status != STATUS_OK:
            ax.plot([x_of(point) for point in group], [y_of(point) for point in group],
                    color="red", label=EXCLUDED_LABELS[status], **style)
            continue
        for knob in sorted({point.scanned_parameter for point in group}):
            selected = [point for point in group if point.scanned_parameter == knob]
            label = f"scanned {knob}" if knob != "none" else "settled"
            ax.plot([x_of(point) for point in selected], [y_of(point) for point in selected],
                    color=KNOB_COLORS.get(knob, "tab:purple"), label=label, **style)


def _point_tag(point):
    """Label a marker by the value that was scanned, falling back to the run name."""

    if point.scanned_parameter in KNOBS and np.isfinite(point.scanned_value):
        return f"{point.scanned_parameter} = {point.scanned_value:g}"
    return point.run_name


def _dedupe_legend(ax, **kwargs):
    """One entry per label: every status is drawn once per knob group."""

    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for handle, label in zip(handles, labels):
        seen.setdefault(label, handle)
    if seen:
        ax.legend(seen.values(), seen.keys(), **kwargs)


def plot_scan(points, fields, output_path, *, show=False, with_parity=False, annotate=True):
    """Amplitude against drive, the slaving constant, and optionally a parity check."""

    plt = import_pyplot(show=show)
    rows = 3 if with_parity else 2
    fig, axes = plt.subplots(
        rows, len(fields), figsize=(5.6 * len(fields), 4.5 * rows),
        constrained_layout=True, squeeze=False,
    )

    constants = {}
    for column, name in enumerate(fields):
        field_points = [point for point in points if point.field == name]
        label = LABELS[name]
        settled = [point for point in field_points if point.settled]
        constant, spread = geometric_mean([point.slaving_constant for point in settled])
        constants[name] = (constant, spread)
        # `predicted/|F|` is that run's measured l_perp * alignment; average it the same way.
        scale, _ = geometric_mean(
            [point.predicted_mean / point.abs_forcing
             for point in settled if point.abs_forcing > 0.0]
        )

        # Row 1: the literal "amplitude against gradient" panel.
        ax = axes[0][column]
        drawable = [point for point in field_points if point.abs_forcing > 0.0]
        zero_drive = len(field_points) - len(drawable)
        _plot_markers(ax, drawable, lambda point: point.abs_forcing,
                      lambda point: point.measured_mean)
        estimates = [point for point in drawable if np.isfinite(point.predicted_mean)]
        span = [point.abs_forcing for point in estimates]
        # A guide line needs |F| to actually vary; a scan of the wrong knob leaves a stripe.
        varies = bool(span) and max(span) > min(span)
        if estimates:
            ax.plot(span, [point.predicted_mean for point in estimates],
                    "s", ms=5, mfc="none", mec="black", ls="none",
                    label="Eq. (47) estimate, per run")
            if varies and np.isfinite(constant) and np.isfinite(scale):
                guide = np.array([min(span), max(span)], dtype=float)
                ax.plot(guide, constant * scale * guide, "k--", lw=1.4,
                        label=rf"slaving: slope 1, $\bar C$ = {constant:.3g}")
        ax.set(xscale="log", yscale="log",
               xlabel=rf"$|F|$ for ${label}$", ylabel=rf"saturated RMS of ${label}$",
               title=rf"${label}$ against its own drive")
        ax.title.set_color(COLORS[name])
        notes = []
        if not drawable:
            notes.append("no points with a nonzero drive:\nthis knob does not drive this field")
        elif not varies:
            notes.append("$|F|$ is constant across this scan:\nthe scanned knob does not drive this field")
        elif zero_drive:
            notes.append(f"{zero_drive} point(s) with F = 0 omitted from the log axis")
        if notes:
            ax.text(0.5, 0.08, "\n".join(notes), transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=8)
        ax.grid(alpha=0.3, which="both")
        _dedupe_legend(ax, fontsize=8)

        # Row 2: the closure assumes this is flat; a trend marks the breakdown.
        ax = axes[1][column]
        _plot_markers(ax, drawable, lambda point: point.abs_forcing,
                      lambda point: point.slaving_constant)
        if np.isfinite(constant):
            ax.axhline(constant, color="0.4", ls=":", lw=1.4,
                       label=rf"$\bar C$ = {constant:.3g} ($\times/\div$ {spread:.2f})")
        if annotate:
            for point in drawable:
                if np.isfinite(point.slaving_constant):
                    ax.annotate(_point_tag(point), (point.abs_forcing, point.slaving_constant),
                                textcoords="offset points", xytext=(5, 4), fontsize=7)
        ax.set(xscale="log", xlabel=rf"$|F|$ for ${label}$",
               ylabel=r"$C$ = measured / Eq. (47)",
               title="slaving constant: flat if the closure holds")
        ax.grid(alpha=0.3)
        _dedupe_legend(ax, fontsize=8)

        if not with_parity:
            continue

        # Row 3: dividing by the measured l_perp and z_rms removes run-to-run amplitude
        # differences, so scatter here is closure failure rather than a different outer scale.
        ax = axes[2][column]
        parity = [point for point in field_points if point.predicted_mean > 0.0]
        _plot_markers(ax, parity, lambda point: point.predicted_mean,
                      lambda point: point.measured_mean)
        finite = [point.predicted_mean for point in parity if np.isfinite(point.measured_mean)]
        if finite:
            line = np.array([min(finite), max(finite)], dtype=float)
            ax.plot(line, line, "k-", lw=1.2, label="exact slaving, $y = x$")
            if np.isfinite(constant):
                ax.plot(line, constant * line, "k--", lw=1.2, label=r"$y = \bar C x$")
        ax.set(xscale="log", yscale="log",
               xlabel=r"Eq. (47) estimate $|F|\,\ell_\perp\,{\rm rms}(z_x)/z_{\rm rms}$",
               ylabel=rf"measured RMS of ${label}$", title="measured against predicted")
        ax.grid(alpha=0.3, which="both")
        _dedupe_legend(ax, fontsize=8)

    reference = points[0]
    knobs = sorted({point.scanned_parameter for point in points})
    sources = sorted({point.source for point in points if point.source != "none"})
    branches = sorted({point.branch for point in points if point.branch != "none"})
    settled_total = sum(point.settled for point in points)
    windows = {point.t_start for point in points if np.isfinite(point.t_start)}
    window = f"t >= {min(windows):.4g}" if windows else "no window"
    fig.suptitle(
        "Slaved amplitudes against background gradient | scanned: " + ", ".join(knobs) + "\n"
        + rf"$v_A$ = {reference.vA:g}, $\chi$ = {reference.chi:g} | window {window} | "
        + f"{'; '.join(sources) or 'no data'} | branch {'/'.join(branches) or 'n/a'}"
        + f" | settled {settled_total}/{len(points)}",
        fontsize=10,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    finalize_figure(fig, output_path=output_path, show=show, plt=plt)
    return constants


def _expand_paths(patterns):
    """Accept literal paths and shell-style globs alike, so Windows works without a shell."""

    paths = []
    for pattern in patterns:
        candidate = Path(pattern).expanduser()
        if candidate.exists():
            paths.append(candidate.resolve())
            continue
        matches = sorted(Path().glob(pattern))
        if not matches:
            raise SystemExit(f"No such run directory or summary file: {pattern}")
        paths.extend(match.resolve() for match in matches)
    return paths


def _default_output_dir(paths):
    parents = {path.parent for path in paths}
    return parents.pop() if len(parents) == 1 else Path.cwd()


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("paths", nargs="+", type=str,
                        help="Run directories, or summary CSVs with --from-summary.")
    parser.add_argument("--fields", nargs="+", choices=FIELDS, default=list(FIELDS),
                        help="Compressive fields to give a column each.")
    parser.add_argument("--from-summary", action="store_true",
                        help="Read reduced summary CSVs instead of opening run directories.")
    parser.add_argument("--from-snapshots", action="store_true",
                        help="Measure l_perp and the alignment from full fields (slow).")
    parser.add_argument("--branch", choices=("plus", "minus"), default="plus",
                        help="Elsasser branch for snapshot mode; fixed so a scan stays consistent.")
    parser.add_argument("--stride", type=int, default=1, help="Read every Nth snapshot.")
    parser.add_argument("--parameter", choices=KNOBS,
                        help="Scanned knob; by default the one that varies across the runs.")
    parser.add_argument("--tmin", type=float,
                        help="Start of the averaging window; default uses --tail-fraction.")
    parser.add_argument("--tail-fraction", type=float, default=DEFAULT_TAIL_FRACTION,
                        help="Fraction of each run's time span to average over.")
    parser.add_argument("--drift-tolerance", type=float, default=DEFAULT_DRIFT_TOLERANCE,
                        help="Largest fractional drift across the window still counted as settled.")
    parser.add_argument("--min-samples", type=int, default=MIN_TAIL_SAMPLES,
                        help="Fewest samples in the window for a point to count as settled.")
    parser.add_argument("--with-parity", action="store_true",
                        help="Add the measured-against-predicted row.")
    parser.add_argument("--allow-mixed-sources", action="store_true",
                        help="Permit CSV and full-field runs in one figure; their estimates differ.")
    parser.add_argument("--no-annotate", action="store_true",
                        help="Omit run names on the slaving-constant row.")
    parser.add_argument("--output", type=Path, help="Image path; default slaved_gradient_scan.png.")
    parser.add_argument("--summary-output", type=Path,
                        help="Summary CSV path; default slaved_gradient_scan.csv.")
    parser.add_argument("--show", action="store_true", help="Show the figure after saving.")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.stride < 1:
        parser.error("--stride must be at least 1.")
    if args.min_samples < 1:
        parser.error("--min-samples must be at least 1.")

    paths = _expand_paths(args.paths)
    if args.from_summary:
        points = [point for path in paths for point in read_summary_csv(path)]
    else:
        points = scan_points_from_runs(
            paths, args.fields,
            scanned_parameter=args.parameter,
            from_snapshots=args.from_snapshots,
            branch=args.branch,
            stride=args.stride,
            tmin=args.tmin,
            tail_fraction=args.tail_fraction,
            drift_tolerance=args.drift_tolerance,
            min_samples=args.min_samples,
        )
    points = [point for point in points if point.field in args.fields]
    if not points:
        raise SystemExit("No runs could be read; nothing to plot.")

    sources = {point.source for point in points if point.source != "none"}
    if len(sources) > 1 and not args.allow_mixed_sources:
        raise SystemExit(
            f"Runs mix {sorted(sources)}. The CSV estimate assumes isotropic alignment and the "
            "full-field estimate measures it, so their Eq. (47) values are offset from each "
            "other. Re-read them the same way, or pass --allow-mixed-sources."
        )

    output_dir = _default_output_dir(paths)
    summary_path = args.summary_output or output_dir / "slaved_gradient_scan.csv"
    output_path = args.output or output_dir / "slaved_gradient_scan.png"
    if not args.from_summary:
        write_summary_csv(points, summary_path)
    plot_scan(points, args.fields, output_path, show=args.show,
              with_parity=args.with_parity, annotate=not args.no_annotate)
    return output_path


if __name__ == "__main__":
    main()
