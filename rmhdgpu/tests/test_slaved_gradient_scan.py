"""Checks for the background-gradient scan driver and its plotting script."""

import csv
import tomllib

import numpy as np
import pytest

from rmhdgpu.diagnostics import scan_slaved_gradient as driver
from vis import plot_slaved_gradient_scan as scan
from vis import plot_slaved_projection as projection


FIELDS = ["drho", "du_par", "db_par"]

# vA = 1 and chi = 1 keep the arithmetic readable: alpha = 0.5 and K_b0 = g - 0.6*K_p0.
BASE_PHYSICS = {"vA": 1.0, "cs2_over_vA2": 1.0, "g": 0.6, "K_p0": 1.0, "K_rho0": 10.5}

BASE_DOCUMENT = {
    "title": "base",
    "output_dir": "outputs",
    "equations": {"type": "inhomogeneous_rmhd_rho"},
    "grid": {"Nx": 32, "Ny": 32, "Nz": 32},
    "time": {"tmax": 20.0},
    "output": {"t_out_scal": 0.02, "t_out_full": 0.2},
    "physics": dict(BASE_PHYSICS),
    "initial_condition": {"type": "random_spectrum_one_wave", "parameters": {"seed": 1}},
}


def write_run(run_dir, *, physics=None, times=None, measured=None, kperp=2.0, tmax=20.0,
              alignment=None):
    """Build a run directory with just the [physics] and CSV columns the scan reads.

    Without `alignment` the CSV looks like one written before w_plus_align existed.
    """

    physics = {**BASE_PHYSICS, **(physics or {})}
    run_dir.mkdir(parents=True, exist_ok=True)
    body = "".join(f"{key} = {value!r}\n" for key, value in physics.items())
    (run_dir / "input_copy.input").write_text(
        f"[time]\ntmax = {tmax!r}\n\n[physics]\n{body}", encoding="utf-8"
    )

    # By default the run reaches tmax, so the "ended early" guard stays quiet.
    times = np.linspace(0.0, tmax, 20) if times is None else np.asarray(times, dtype=float)
    measured = 1.0 if measured is None else measured
    values = np.full_like(times, measured) if np.isscalar(measured) else np.asarray(measured)
    extra_header = [] if alignment is None else ["w_plus_align"]
    extra_row = [] if alignment is None else [alignment]
    with (run_dir / "scalar_diagnostics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time", "w_plus_kperp", *[f"{name}_rms" for name in FIELDS], *extra_header])
        for time, value in zip(times, values):
            writer.writerow([time, kperp, value, value, value, *extra_row])
    return run_dir


# --- measured amplitude plots -------------------------------------------------------


def write_scalar_csv(run_dir, header, rows):
    """Replace diagnostics with only the columns needed by a particular test."""
    with (run_dir / "scalar_diagnostics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def test_read_run_uses_field_forcings_and_normalizations_without_pump_diagnostics(tmp_path):
    run = write_run(tmp_path / "normalized", physics={
        "vA": 2.0, "cs2_over_vA2": 1.0, "g": 4.0, "K_p0": 5.0, "K_rho0": 1.5,
    })
    # The t alias is supported, and no kperp, energy, or alignment columns are needed.
    write_scalar_csv(run, ["t", "drho_rms", "du_par_rms", "db_par_rms"],
                     [[time, 4.0, 8.0, 3.0] for time in range(11)])
    points = {point["field"]: point for point in scan.read_run(run, FIELDS)}
    assert [points[name]["forcing"] for name in FIELDS] == pytest.approx([-1.0, 2.0, 5.0])
    # vA=2 and alpha=1/2: velocity divides by 2, magnetic amplitude multiplies by 2.
    assert [points[name]["saturated_rms"] for name in FIELDS] == pytest.approx([4.0, 4.0, 6.0])
    assert all(point["n_samples"] == 5 for point in points.values())
    assert all(point["t_start"] == pytest.approx(6.0) for point in points.values())
    assert all(point["t_end"] == pytest.approx(10.0) for point in points.values())
    # Without w_plus_kperp there is no Eq. (47) estimate, but the measured points survive.
    assert all(np.isnan(point["slaved_rms"]) for point in points.values())
    assert all(point["alignment_source"] == "none" for point in points.values())


def test_slaved_estimate_uses_the_measured_window_and_records_its_alignment(tmp_path):
    run = write_run(tmp_path / "window")
    # kperp and the alignment change at t = 5, so only the window's values may count.
    write_scalar_csv(run, ["time", "w_plus_kperp", "w_plus_align", "drho_rms"],
                     [[t, 1.0 if t < 5 else 4.0, 0.9 if t < 5 else 0.5, 3.0] for t in range(10)])
    point = scan.read_run(run, ["drho"], tmin=5.0)[0]
    # |F_rho| = 10.2, l_perp = 1/4 and alignment 0.5 in the window.
    assert point["slaved_rms"] == pytest.approx(10.2 * 0.25 * 0.5)
    assert (point["pump_branch"], point["alignment_source"]) == ("plus", "measured")

    # The stronger Elsasser branch in the first row is the pump.
    write_scalar_csv(run, ["time", "w_plus", "w_minus", "w_minus_kperp", "w_minus_align", "drho_rms"],
                     [[t, 1.0, 9.0, 2.0, 0.5, 3.0] for t in range(10)])
    point = scan.read_run(run, ["drho"])[0]
    assert point["slaved_rms"] == pytest.approx(10.2 * 0.5 * 0.5)
    assert (point["pump_branch"], point["alignment_source"]) == ("minus", "measured")

    # An older CSV without w_plus_align assumes the Eq. (48) value 1/sqrt(2).
    write_run(run)
    point = scan.read_run(run, ["drho"])[0]
    assert point["slaved_rms"] == pytest.approx(10.2 * 0.5 / np.sqrt(2.0))
    assert point["alignment_source"] == "assumed_isotropic"


def test_tail_window_is_selected_by_time_then_uses_the_sample_mean(tmp_path):
    times = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 6.0, 8.0, 10.0]
    values = [100.0] * 6 + [2.0, 4.0, 8.0]
    run = write_run(tmp_path / "uneven", times=times, measured=values)
    point = scan.read_run(run, ["drho"], tail_fraction=0.5)[0]
    assert point["t_start"] == pytest.approx(5.0)
    assert point["n_samples"] == 3
    assert point["saturated_rms"] == pytest.approx(14.0 / 3.0)


def test_tmin_overrides_tail_and_ignores_invalid_data_before_the_window(tmp_path):
    values = [np.nan] * 7 + [7.0, 8.0, 9.0]
    run = write_run(tmp_path / "late", times=np.arange(10), measured=values)
    points = scan.read_run(run, FIELDS, tmin=7.0, tail_fraction=1.0)
    assert [point["saturated_rms"] for point in points] == pytest.approx([8.0, 8.0, 16.0])
    assert all(point["t_start"] == pytest.approx(7.0) for point in points)
    assert all(point["n_samples"] == 3 for point in points)


@pytest.mark.parametrize("invalid", [np.nan, np.inf])
def test_missing_late_values_cannot_move_the_window_backward(tmp_path, invalid):
    run = write_run(tmp_path / "missing_tail", times=np.arange(11),
                    measured=[3.0] * 5 + [invalid] * 6)
    with pytest.raises(ValueError):
        scan.read_run(run, ["drho"])


def test_zero_rms_is_retained_but_negative_rms_is_rejected(tmp_path):
    run = write_run(tmp_path / "zero", measured=0.0)
    assert scan.read_run(run, ["drho"])[0]["saturated_rms"] == 0.0
    run = write_run(run, measured=-1.0)
    with pytest.raises(ValueError):
        scan.read_run(run, ["drho"])


@pytest.mark.parametrize("times", [[0, 1, 1, 2], [3, 2, 1, 0], [0, 1, np.nan, 3]])
def test_invalid_timestamps_are_rejected(tmp_path, times):
    run = write_run(tmp_path / "bad_times", times=times)
    with pytest.raises(ValueError):
        scan.read_run(run, ["drho"], tail_fraction=1.0)


@pytest.mark.parametrize("tmin", [9.0, 10.0])
def test_a_window_requires_at_least_two_saved_samples(tmp_path, tmin):
    run = write_run(tmp_path / "short_window", times=np.arange(10))
    with pytest.raises(ValueError):
        scan.read_run(run, ["drho"], tmin=tmin)


def test_missing_requested_rms_column_has_a_useful_error(tmp_path):
    run = write_run(tmp_path / "missing_field")
    write_scalar_csv(run, ["time", "drho_rms"], [[0, 2], [1, 2], [2, 2]])
    with pytest.raises(ValueError, match="db_par"):
        scan.read_run(run, ["db_par"], tail_fraction=1.0)


def capture_panels(monkeypatch):
    """Record what each panel draws instead of saving the figure."""
    captured = []

    def capture(fig, *, output_path, show, plt):
        for ax in fig.axes:
            legend = ax.get_legend()
            captured.append({
                "scales": (ax.get_xscale(), ax.get_yscale()),
                "title": ax.get_title(),
                "offsets": {collection.get_label(): np.asarray(collection.get_offsets())
                            for collection in ax.collections},
                "lines": {line.get_label(): np.column_stack(line.get_data()) for line in ax.lines},
                "legend": [] if legend is None else [text.get_text() for text in legend.get_texts()],
                "notes": [text.get_text() for text in ax.texts],
            })
        plt.close(fig)

    monkeypatch.setattr(scan, "finalize_figure", capture)
    return captured


def test_plot_draws_measured_and_slaved_log10_points_and_marks_omitted_points(
        tmp_path, monkeypatch, capsys):
    run = write_run(tmp_path / "steady", measured=4.0, alignment=0.5)
    points = scan.read_run(run, FIELDS)
    # Also omit a zero-amplitude point with a nonzero drive.
    points.append({**points[0], "forcing": -1.0, "saturated_rms": 0.0, "slaved_rms": 0.0})
    captured = capture_panels(monkeypatch)
    scan.plot_scan(points, FIELDS, tmp_path / "unused.png")
    assert len(captured) == 3
    for panel, name in zip(captured, FIELDS):
        # The logarithm is taken of the data, so the axes themselves stay linear.
        assert panel["scales"] == ("linear", "linear")
        assert panel["lines"] == {}  # One run per field: no second drive, so no fit.
        for key, label in (("saturated_rms", scan.MEASURED_LABEL),
                           ("slaved_rms", scan.SLAVED_LABEL)):
            valid = [point for point in points if point["field"] == name
                     and abs(point["forcing"]) > 0 and point[key] > 0]
            expected = [[np.log10(abs(point["forcing"])), np.log10(point[key])]
                        for point in valid]
            np.testing.assert_allclose(panel["offsets"].get(label, np.empty((0, 2))),
                                       np.asarray(expected).reshape(-1, 2))
    printed = capsys.readouterr().out
    assert "drho" in printed and "omitted" in printed  # The zero RMS is explained.
    assert captured[0]["legend"] == [scan.MEASURED_LABEL, scan.SLAVED_LABEL]
    assert captured[1]["notes"]  # This run's velocity drive is exactly zero...
    assert captured[1]["legend"] == []  # ...so that panel has nothing to label.


def test_regression_lines_recover_power_laws_and_appear_in_the_legend(
        tmp_path, monkeypatch, capsys):
    # |F_rho| = K_rho0 - 0.3 here. The measured RMS is set to 0.5 |F|^0.8, and with
    # l_perp = 1/kperp = 0.5 and alignment 0.5 the slaved estimate is 0.25 |F| (slope 1).
    drives = []
    points = []
    for index, K_rho0 in enumerate([2.0, 7.0, 10.5]):
        drives.append(K_rho0 - 0.3)
        run = write_run(tmp_path / f"run_{index}", physics={"K_rho0": K_rho0},
                        measured=0.5 * drives[-1]**0.8, kperp=2.0, alignment=0.5)
        points.extend(scan.read_run(run, ["drho", "db_par"]))
    captured = capture_panels(monkeypatch)
    scan.plot_scan(points, ["drho", "db_par"], tmp_path / "unused.png")

    density = captured[0]
    x = np.log10([min(drives), max(drives)])  # The lines span only the fitted drives.
    np.testing.assert_allclose(density["lines"]["Measured fit: slope 0.80"],
                               np.column_stack([x, 0.8 * x + np.log10(0.5)]))
    np.testing.assert_allclose(density["lines"]["Slaved fit: slope 1.00"],
                               np.column_stack([x, x + np.log10(0.25)]))
    assert density["legend"] == [scan.MEASURED_LABEL, "Measured fit: slope 0.80",
                                 scan.SLAVED_LABEL, "Slaved fit: slope 1.00"]
    assert "drho measured fit: slope 0.800" in capsys.readouterr().out
    # K_rho0 does not move F_b, so the magnetic panel has points but no line to fit.
    magnetic = captured[1]
    assert magnetic["lines"] == {} and "same drive" in magnetic["title"]
    assert magnetic["legend"] == [scan.MEASURED_LABEL, scan.SLAVED_LABEL]


def test_fit_line_needs_two_different_drives():
    assert scan.fit_line([0.0, 1.0, 2.0], [1.0, 3.0, 5.0]) == pytest.approx((2.0, 1.0))
    assert scan.fit_line([0.5, 0.5, 0.5], [1.0, 2.0, 3.0]) is None
    assert scan.fit_line([0.5], [1.0]) is None


def test_slaved_mean_matches_projection_over_the_same_tail_for_every_field(tmp_path):
    run = write_run(tmp_path / "time_varying", physics={
        "vA": 2.0, "cs2_over_vA2": 1.0, "g": 4.0, "K_p0": 5.0, "K_rho0": 1.5,
    })
    write_scalar_csv(run, ["time", "w_plus", "w_minus", "w_minus_kperp", "w_minus_align",
                           "drho_rms", "du_par_rms", "db_par_rms"],
                     [[t, 1.0 if t == 0 else 100.0, 9.0, 1.0 + t, 0.1 + 0.05 * t,
                       4.0, 8.0, 3.0] for t in range(11)])
    parameters = projection.load_parameters(run)
    series = projection.read_from_csv(run, FIELDS, parameters)
    points = scan.read_run(run, FIELDS)
    window = series.times >= 6.0
    for point in points:
        name = point["field"]
        assert point["slaved_rms"] == pytest.approx(series.predicted[name][window].mean())
        assert point["saturated_rms"] == pytest.approx(series.measured[name][window].mean())
        assert point["pump_branch"] == "minus"  # Selected in the first row, not the tail.
        assert point["alignment_source"] == "measured"
        assert (point["t_start"], point["t_end"], point["n_samples"]) == (6.0, 10.0, 5)
    assert [point["saturated_rms"] for point in points] == pytest.approx([4.0, 4.0, 6.0])


def test_nonfinite_slaved_tail_keeps_the_measurement_without_using_earlier_estimates(tmp_path):
    run = write_run(tmp_path / "missing_slaved_tail", measured=3.0)
    write_scalar_csv(run, ["time", "w_plus_kperp", "w_plus_align", "drho_rms"],
                     [[t, 2.0, 0.5 if t < 6 else np.nan, 3.0] for t in range(11)])
    point = scan.read_run(run, ["drho"])[0]
    assert point["saturated_rms"] == pytest.approx(3.0)
    assert np.isnan(point["slaved_rms"])
    assert (point["t_start"], point["t_end"], point["n_samples"]) == (6.0, 10.0, 5)


def test_measured_and_slaved_fits_select_valid_points_independently(tmp_path, monkeypatch):
    # The high-drive measurement changes the slope from 1 to 1.9. Its estimate is missing,
    # so coupling the two masks would silently fit the wrong measured exponent.
    points = [{
        "run_dir": f"run_{index}", "field": "drho", "forcing": -drive,
        "saturated_rms": measured, "slaved_rms": slaved,
        "pump_branch": "plus", "alignment_source": "measured",
        "t_start": 6.0, "t_end": 10.0, "n_samples": 5,
    } for index, (drive, measured, slaved) in enumerate(
        [(1.0, 1.0, 4.0), (2.0, 2.0, 2.0), (4.0, 4.0, 1.0), (8.0, 64.0, np.nan)]
    )]
    captured = capture_panels(monkeypatch)
    scan.plot_scan(points, ["drho"], tmp_path / "unused.png")
    panel = captured[0]
    measured_x = np.log10([1.0, 8.0])
    slaved_x = np.log10([1.0, 4.0])
    np.testing.assert_allclose(panel["lines"]["Measured fit: slope 1.90"],
                               np.column_stack([measured_x, 1.9 * measured_x - 0.6 * np.log10(2)]))
    np.testing.assert_allclose(panel["lines"]["Slaved fit: slope -1.00"],
                               np.column_stack([slaved_x, -slaved_x + np.log10(4)]), atol=1e-12)
    assert len(panel["lines"]) == 2  # No additional fixed-slope reference line.
    assert len(panel["offsets"][scan.MEASURED_LABEL]) == 4
    assert len(panel["offsets"][scan.SLAVED_LABEL]) == 3
    assert panel["legend"] == [scan.MEASURED_LABEL, "Measured fit: slope 1.90",
                               scan.SLAVED_LABEL, "Slaved fit: slope -1.00"]


def test_cli_defaults_to_all_fields_and_writes_png_and_simple_summary(tmp_path):
    runs = [write_run(tmp_path / f"run_{index}", physics={"K_rho0": value}, measured=value)
            for index, value in enumerate([2.0, 7.0, 10.5])]
    summary = tmp_path / "scan.csv"
    output = scan.main([*[str(run) for run in runs], "--output", str(tmp_path / "scan.png"),
                        "--summary-output", str(summary)])
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    with summary.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        assert set(reader.fieldnames) == {
            "run_dir", "field", "forcing", "saturated_rms", "slaved_rms", "pump_branch",
            "alignment_source", "t_start", "t_end", "n_samples",
        }
    assert len(rows) == 9
    assert {row["field"] for row in rows} == set(FIELDS)
    assert not (tmp_path / "scan_fits.csv").exists()


def test_cli_accepts_absolute_globs_and_explicit_fields(tmp_path, monkeypatch):
    for index, value in enumerate([2.0, 7.0]):
        write_run(tmp_path / f"case_{index}", physics={"K_rho0": value})
    panels = []

    def capture(fig, *, output_path, show, plt):
        panels.append(len(fig.axes))
        plt.close(fig)

    monkeypatch.setattr(scan, "finalize_figure", capture)
    summary = tmp_path / "selected.csv"
    scan.main([str(tmp_path / "case_*"), "--fields", "drho", "--parameter", "K_rho0",
               "--output", str(tmp_path / "selected.png"), "--summary-output", str(summary)])
    with summary.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert panels == [1]
    assert len(rows) == 2
    assert all(row["field"] == "drho" for row in rows)


def test_summary_preserves_signed_drive_and_linear_rms(tmp_path):
    run = write_run(tmp_path / "steady", measured=4.0)
    points = scan.read_run(run, FIELDS)
    summary = tmp_path / "summary.csv"
    scan.write_summary_csv(points, summary)
    with summary.open(encoding="utf-8", newline="") as handle:
        rows = {row["field"]: row for row in csv.DictReader(handle)}
    assert float(rows["drho"]["forcing"]) == pytest.approx(-10.2)
    assert float(rows["drho"]["saturated_rms"]) == pytest.approx(4.0)
    assert float(rows["du_par"]["forcing"]) == pytest.approx(0.0)
    assert float(rows["db_par"]["saturated_rms"]) == pytest.approx(8.0)


# --- simulation scan driver (unchanged) -----------------------------------------------


def test_driver_forcings_match_the_projection_script():
    # The driver keeps its own copy so rmhdgpu does not import vis; they must not drift.
    for physics in (BASE_PHYSICS, {**BASE_PHYSICS, "g": -2.0, "K_p0": 3.0, "K_rho0": 0.25}):
        p = driver.derived_parameters(physics)
        assert driver.background_forcings(p) == pytest.approx(projection.forcings(p))


def test_each_knob_plots_exactly_the_fields_it_drives():
    base = driver.derived_parameters(BASE_PHYSICS)
    for knob, driven in driver.FIELDS_DRIVEN_BY.items():
        moved = driver.derived_parameters({**BASE_PHYSICS, knob: BASE_PHYSICS[knob] + 0.5})
        before, after = driver.background_forcings(base), driver.background_forcings(moved)
        changed = {name for name in driver.FIELDS if after[name] != pytest.approx(before[name])}
        assert changed == set(driven), knob


def test_plot_step_passes_the_driven_fields(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(driver.subprocess, "run", lambda command, **kwargs: calls.append(command))
    driver.plot_scan("K_p0", [tmp_path / "run"], tmp_path)
    start = calls[0].index("--fields") + 1
    assert calls[0][start:start + 2] == ["du_par", "db_par"]


def test_point_input_changes_only_the_scanned_key(tmp_path):
    document = driver.point_document(BASE_DOCUMENT, "K_rho0", 4.0, tmp_path / "runs" / "p00")
    path = driver.write_point_input(document, tmp_path / "point.input")
    reread = tomllib.loads(path.read_text(encoding="utf-8"))

    assert reread["physics"]["K_rho0"] == pytest.approx(4.0)
    assert reread["physics"]["g"] == pytest.approx(BASE_PHYSICS["g"])
    assert reread["physics"]["K_p0"] == pytest.approx(BASE_PHYSICS["K_p0"])
    assert reread["grid"] == BASE_DOCUMENT["grid"]
    assert reread["initial_condition"] == BASE_DOCUMENT["initial_condition"]
    # Snapshots are pure cost for a CSV-based scan, and the path must not be relative.
    assert reread["output"]["t_out_full"] == pytest.approx(0.0)
    assert reread["output"]["t_out_scal"] == pytest.approx(0.02)
    from pathlib import Path as _Path
    assert _Path(reread["output_dir"]).is_absolute()
    assert BASE_DOCUMENT["physics"]["K_rho0"] == pytest.approx(10.5)


def test_point_input_keeps_snapshots_when_asked(tmp_path):
    document = driver.point_document(BASE_DOCUMENT, "g", 0.2, tmp_path, snapshots=True)
    assert document["output"]["t_out_full"] == pytest.approx(0.2)


def test_unknown_knob_raises_rather_than_being_ignored():
    with pytest.raises(ValueError, match="N2"):
        driver.point_document(BASE_DOCUMENT, "N2", 1.0, "runs")


def test_dry_run_reports_stability_and_drives(capsys):
    driver.report_scan("K_rho0", [0.1, 10.5], BASE_DOCUMENT)
    printed = capsys.readouterr().out
    assert "unstable stratification" in printed  # K_rho0 = 0.1 is below the 0.3 threshold.
    assert "does not drive du_par" in printed
    assert "K_rho0 = 0.3" in printed


def test_existing_runs_are_skipped_unless_forced(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(driver.subprocess, "run", lambda command, **kwargs: calls.append(command))
    run_dir = write_run(tmp_path / "runs" / "K_rho0_00")

    assert driver.run_point(tmp_path / "point.input", run_dir) is True
    assert calls == []
    driver.run_point(tmp_path / "point.input", run_dir, force=True)
    assert len(calls) == 1
    assert "rmhdgpu.run" in calls[0]


def test_forced_rerun_clears_only_stale_snapshots(tmp_path, monkeypatch):
    monkeypatch.setattr(driver.subprocess, "run", lambda command, **kwargs: None)
    run_dir = write_run(tmp_path / "runs" / "K_rho0_00")
    (run_dir / "fullfields").mkdir()
    (run_dir / "fullfields" / "fullfield_0001.h5").write_bytes(b"stale")

    driver.run_point(tmp_path / "point.input", run_dir, force=True)
    assert not (run_dir / "fullfields").exists()
    assert (run_dir / "input_copy.input").is_file()


def test_unstable_points_are_not_launched(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(driver.subprocess, "run", lambda command, **kwargs: calls.append(command))
    completed = driver.run_scan("K_rho0", [0.1, 4.0], BASE_DOCUMENT, tmp_path / "scan")
    assert len(calls) == 1  # Only the stable point ran.
    assert [path.name for path in completed] == ["K_rho0_01"]
