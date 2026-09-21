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


def write_run(run_dir, *, physics=None, times=None, measured=None, kperp=2.0, tmax=20.0):
    """Build a run directory with just the [physics] and CSV columns the scan reads."""

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
    with (run_dir / "scalar_diagnostics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time", "w_plus_kperp", *[f"{name}_rms" for name in FIELDS]])
        for time, value in zip(times, values):
            writer.writerow([time, kperp, value, value, value])
    return run_dir


# --- the statistics ------------------------------------------------------------------


def test_flat_series_is_settled():
    times = np.arange(10, dtype=float)
    result = scan.saturated_value(times, np.full(10, 3.0), tail_fraction=1.0, min_samples=1)
    assert result.mean == pytest.approx(3.0)
    assert result.drift == pytest.approx(0.0)
    assert result.n_samples == 10
    assert result.settled is True


def test_linear_ramp_is_flagged_as_drifting():
    times = np.arange(10, dtype=float)
    values = 1.0 + times
    result = scan.saturated_value(times, values, tail_fraction=1.0, min_samples=1)
    # Slope 1 across a window of 9 with mean 5.5.
    assert result.mean == pytest.approx(5.5)
    assert result.drift == pytest.approx(9.0 / 5.5)
    assert result.settled is False


def test_window_is_chosen_by_time_not_sample_count():
    # Samples bunch up early, so a count-based tail would reach much further back in time.
    times = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 6.0, 8.0, 10.0])
    values = np.array([9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 2.0, 2.0, 2.0])
    result = scan.saturated_value(times, values, tail_fraction=0.5, min_samples=1)
    assert result.t_start == pytest.approx(5.0)
    assert result.n_samples == 3
    assert result.mean == pytest.approx(2.0)


def test_explicit_tmin_overrides_the_tail_fraction():
    times = np.arange(10, dtype=float)
    result = scan.saturated_value(times, times, tmin=7.0, min_samples=1)
    assert result.t_start == pytest.approx(7.0)
    assert result.n_samples == 3
    assert result.mean == pytest.approx(8.0)


def test_short_window_is_not_settled():
    times = np.arange(10, dtype=float)
    result = scan.saturated_value(times, np.full(10, 3.0), tail_fraction=1.0, min_samples=20)
    assert result.n_samples == 10
    assert result.settled is False


def test_all_nan_series_has_no_window():
    times = np.arange(5, dtype=float)
    result = scan.saturated_value(times, np.full(5, np.nan))
    assert result.n_samples == 0
    assert result.settled is False
    assert np.isnan(result.mean)


# --- the physics ---------------------------------------------------------------------


def test_driver_forcings_match_the_projection_script():
    # The driver keeps its own copy so rmhdgpu does not import vis; they must not drift.
    for physics in (BASE_PHYSICS, {**BASE_PHYSICS, "g": -2.0, "K_p0": 3.0, "K_rho0": 0.25}):
        p = driver.derived_parameters(physics)
        assert driver.background_forcings(p) == pytest.approx(projection.forcings(p))


def test_only_the_scanned_knob_moves_its_own_field(tmp_path):
    # K_rho0 moves the density drive alone; the two magnetic drives must not budge.
    runs = [write_run(tmp_path / f"run_{index}", physics={"K_rho0": value})
            for index, value in enumerate([2.0, 10.5])]
    points = scan.scan_points_from_runs(runs, FIELDS)
    by_field = {name: [p for p in points if p.field == name] for name in FIELDS}

    assert {point.scanned_parameter for point in points} == {"K_rho0"}
    assert [point.scanned_value for point in by_field["drho"]] == [2.0, 10.5]
    assert by_field["drho"][0].forcing != pytest.approx(by_field["drho"][1].forcing)
    for name in ("du_par", "db_par"):
        assert by_field[name][0].forcing == pytest.approx(by_field[name][1].forcing)
    for point in points:
        assert point.forcing == pytest.approx(
            projection.forcings(projection.derived_parameters(
                {"vA": point.vA, "cs2_over_vA2": point.chi, "g": point.g,
                 "K_p0": point.K_p0, "K_rho0": point.K_rho0}))[point.field]
        )


def test_pressure_gradient_leaves_the_density_drive_and_stratification_alone(tmp_path):
    runs = [write_run(tmp_path / f"run_{index}", physics={"K_p0": value})
            for index, value in enumerate([-2.0, 4.0])]
    points = scan.scan_points_from_runs(runs, FIELDS)
    by_field = {name: [p for p in points if p.field == name] for name in FIELDS}

    assert {point.scanned_parameter for point in points} == {"K_p0"}
    assert by_field["drho"][0].forcing == pytest.approx(by_field["drho"][1].forcing)
    assert by_field["drho"][0].N_sq == pytest.approx(by_field["drho"][1].N_sq)
    for name in ("du_par", "db_par"):
        assert by_field[name][0].forcing != pytest.approx(by_field[name][1].forcing)


def test_unstable_stratification_is_flagged(tmp_path):
    # N^2 = -g * F_drho, so K_rho0 below g/(vA^2 (1+chi)) = 0.3 is convectively unstable.
    run = write_run(tmp_path / "unstable", physics={"K_rho0": 0.1})
    point = scan.scan_points_from_runs([run], ["drho"])[0]
    assert point.N_sq < 0.0
    assert point.stable is False
    assert point.status == scan.STATUS_UNSTABLE
    assert point.settled is False


def test_run_ending_before_tmax_is_flagged(tmp_path):
    run = write_run(tmp_path / "short", times=np.arange(10, dtype=float), tmax=100.0)
    point = scan.scan_points_from_runs([run], ["drho"], min_samples=1)[0]
    assert point.status == scan.STATUS_ENDED_EARLY
    assert point.settled is False


def test_slaving_constant_is_the_ratio_of_window_means(tmp_path):
    # l_perp = 1/kperp = 0.5 and the CSV path assumes alignment 1/sqrt(2).
    run = write_run(tmp_path / "steady", measured=4.0, kperp=2.0)
    point = scan.scan_points_from_runs([run], ["drho"], min_samples=1)[0]
    expected = 0.5 * projection.ISOTROPIC_ALIGNMENT * abs(point.forcing)
    assert point.measured_mean == pytest.approx(4.0)
    assert point.predicted_mean == pytest.approx(expected)
    assert point.slaving_constant == pytest.approx(4.0 / expected)
    assert point.status == scan.STATUS_OK


def test_zero_drive_leaves_the_constant_undefined(tmp_path):
    # K_b0 = g - 0.6*K_p0 vanishes at K_p0 = 1 with g = 0.6, so du_par has no drive at all.
    run = write_run(tmp_path / "no_drive")
    point = [p for p in scan.scan_points_from_runs([run], FIELDS, min_samples=1)
             if p.field == "du_par"][0]
    assert point.forcing == pytest.approx(0.0)
    assert point.predicted_mean == pytest.approx(0.0)
    assert np.isnan(point.slaving_constant)
    assert point.status == scan.STATUS_OK  # The measured series is still perfectly settled.


def test_geometric_mean_ignores_nonpositive_and_undefined_values():
    mean, spread = scan.geometric_mean([1.0, 4.0, np.nan, 0.0, -3.0])
    assert mean == pytest.approx(2.0)
    assert spread == pytest.approx(2.0)
    assert np.isnan(scan.geometric_mean([np.nan])[0])


# --- plumbing ------------------------------------------------------------------------


def test_summary_csv_round_trips(tmp_path):
    runs = [write_run(tmp_path / f"run_{index}", physics={"K_rho0": value})
            for index, value in enumerate([2.0, 10.5])]
    points = scan.scan_points_from_runs(runs, FIELDS, min_samples=1)
    path = scan.write_summary_csv(points, tmp_path / "summary.csv")
    restored = scan.read_summary_csv(path)

    assert len(restored) == len(points)
    for before, after in zip(points, restored):
        assert after.run_name == before.run_name
        assert after.field == before.field
        assert after.scanned_parameter == before.scanned_parameter
        assert after.settled is before.settled
        assert after.stable is before.stable
        assert after.n_samples == before.n_samples
        assert after.forcing == pytest.approx(before.forcing)
        assert after.measured_mean == pytest.approx(before.measured_mean)
    assert np.isnan(scan.read_summary_csv(path)[1].slaving_constant)  # du_par has no drive.


def test_cli_writes_the_figure_and_the_summary(tmp_path):
    runs = [write_run(tmp_path / f"run_{index}", physics={"K_rho0": value})
            for index, value in enumerate([2.0, 7.0, 10.5])]
    output = scan.main([*[str(path) for path in runs], "--min-samples", "1", "--with-parity",
                        "--output", str(tmp_path / "scan.png"),
                        "--summary-output", str(tmp_path / "scan.csv")])
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert (tmp_path / "scan.csv").is_file()


def test_cli_refuses_to_mix_reading_modes(tmp_path, monkeypatch):
    runs = [write_run(tmp_path / f"run_{index}") for index in range(2)]
    points = scan.scan_points_from_runs(runs, ["drho"], min_samples=1)
    points[0].source = "full fields"
    monkeypatch.setattr(scan, "scan_points_from_runs", lambda *a, **k: points)
    with pytest.raises(SystemExit, match="allow-mixed-sources"):
        scan.main([str(runs[0]), str(runs[1]), "--output", str(tmp_path / "x.png")])


def test_figure_marks_a_knob_that_does_not_drive_a_field(tmp_path, monkeypatch):
    runs = [write_run(tmp_path / f"run_{index}", physics={"K_rho0": value})
            for index, value in enumerate([2.0, 10.5])]
    points = scan.scan_points_from_runs(runs, FIELDS, min_samples=1)
    texts = []

    def capture(fig, *, output_path, show, plt):
        texts.extend(text.get_text() for ax in fig.axes for text in ax.texts)
        plt.close(fig)

    monkeypatch.setattr(scan, "finalize_figure", capture)
    scan.plot_scan(points, FIELDS, tmp_path / "unused.png")
    # db_par keeps a constant drive under a K_rho0 scan; du_par has none at all here.
    assert any("constant across this scan" in text for text in texts)
    assert any("does not drive this field" in text for text in texts)


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
    assert BASE_DOCUMENT["physics"]["K_rho0"] == pytest.approx(10.5)  # Base left untouched.


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
    assert (run_dir / "input_copy.input").is_file()  # Nothing else was removed.


def test_unstable_points_are_not_launched(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(driver.subprocess, "run", lambda command, **kwargs: calls.append(command))
    completed = driver.run_scan("K_rho0", [0.1, 4.0], BASE_DOCUMENT, tmp_path / "scan")
    assert len(calls) == 1  # Only the stable point ran.
    assert [path.name for path in completed] == ["K_rho0_01"]
