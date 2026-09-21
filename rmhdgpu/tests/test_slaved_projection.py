"""Small analytic checks for the slaved-amplitude plotting script."""

import csv

import numpy as np
import pytest

from vis import plot_slaved_projection as projection


FIELDS = ["drho", "du_par", "db_par"]


@pytest.fixture
def run_dir(tmp_path):
    (tmp_path / "input_copy.input").write_text(
        "[physics]\nvA = 2.0\ncs2_over_vA2 = 3.0\ng = 4.0\n"
        "K_p0 = 0.5\nK_rho0 = 0.3\n",
        encoding="utf-8",
    )
    return tmp_path


def write_snapshot(run_dir, index, *, plus=(0.0, 1, 2), minus=(0.0, 1, 2), fields=FIELDS):
    """Save exact cosine potentials; omega follows analytically from lap_perp(phi)."""
    h5py = pytest.importorskip("h5py")
    axis = np.arange(8) * (2 * np.pi / 8)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    plus_amplitude, plus_kx, plus_ky = plus
    minus_amplitude, minus_kx, minus_ky = minus
    plus_potential = plus_amplitude * np.cos(plus_kx * x + plus_ky * y + z)
    minus_potential = minus_amplitude * np.cos(minus_kx * x + minus_ky * y + z)
    omega = -0.5 * (
        (plus_kx**2 + plus_ky**2) * plus_potential
        + (minus_kx**2 + minus_ky**2) * minus_potential
    )
    directory = run_dir / "fullfields"
    directory.mkdir(exist_ok=True)
    with h5py.File(directory / f"fullfield_{index:04d}.h5", "w") as handle:
        for name in ("x", "y", "z"):
            handle[f"metadata/{name}"] = axis
        handle["output/time"] = float(index)
        handle["output/omega"] = omega
        handle["output/psi"] = 0.5 * (minus_potential - plus_potential)
        for name in fields:
            handle[f"output/{name}"] = np.full_like(x, {"drho": 2, "du_par": 3, "db_par": 4}[name])


def write_csv(run_dir, time_column="time"):
    # Only db_par is present: selecting it must not require other RMS columns.
    with (run_dir / "scalar_diagnostics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([time_column, "w_plus_kperp", "db_par_rms"])
        writer.writerows([[0, 2, 3], [1, 4, 6]])


def test_physical_forcings_and_normalizations(run_dir):
    p = projection.load_parameters(run_dir)
    assert projection.forcings(p) == pytest.approx({"drho": -0.05, "du_par": -0.1, "db_par": 0.2})
    assert projection.measured_divisors(p) == pytest.approx({"drho": 1, "du_par": 2, "db_par": 0.75})


def test_density_forcing_is_defined_without_gravity():
    p = projection.derived_parameters(
        {"vA": 2, "cs2_over_vA2": 3, "g": 0, "K_p0": 0.5, "K_rho0": 0.3}
    )
    assert projection.forcings(p)["drho"] == pytest.approx(-0.3)


@pytest.mark.parametrize("branch", ["plus", "minus"])
def test_single_mode_matches_analytic_projection(run_dir, branch):
    write_snapshot(run_dir, 0, **{branch: (3.0, 1, 2)})
    p = projection.load_parameters(run_dir)
    run = projection.read_from_snapshots(run_dir, FIELDS, p, branch=branch)

    # For k_perp=(1,2), l_perp=1/sqrt(5) and rms(z_x)/rms(|z|)=2/sqrt(5).
    assert run.branch == branch
    np.testing.assert_allclose(run.times, [0])
    np.testing.assert_allclose(run.l_perp, [1 / np.sqrt(5)])
    np.testing.assert_allclose(run.alignment, [2 / np.sqrt(5)])
    # Potential amplitude 3, |k_perp|=sqrt(5), k_parallel=1 and vA=2.
    np.testing.assert_allclose(run.chi_a, [15 / (2 * np.sqrt(2))])
    for name, measured in {"drho": 2, "du_par": 1.5, "db_par": 16 / 3}.items():
        np.testing.assert_allclose(run.measured[name], [measured])
    for name, predicted in {"drho": 0.02, "du_par": 0.04, "db_par": 0.08}.items():
        np.testing.assert_allclose(run.predicted[name], [predicted])


def test_auto_branch_stays_fixed_when_dominance_changes(run_dir):
    # Later the minus wave dominates and points along y (z_x=0), but the
    # selected plus wave continues to point along x (alignment=1).
    write_snapshot(run_dir, 0, plus=(4, 0, 1), minus=(1, 2, 0))
    write_snapshot(run_dir, 1, plus=(1, 0, 1), minus=(4, 2, 0))
    p = projection.load_parameters(run_dir)
    run = projection.read_from_snapshots(run_dir, ["drho"], p)
    assert run.branch == "plus"
    np.testing.assert_allclose(run.alignment, [1, 1])
    np.testing.assert_allclose(run.l_perp, [1, 1])
    np.testing.assert_allclose(run.predicted["drho"], [0.05, 0.05])


def test_selected_snapshot_field_and_stride(run_dir):
    for index in range(3):
        write_snapshot(run_dir, index, minus=(2, 1, 2), fields=["db_par"])
    p = projection.load_parameters(run_dir)
    run = projection.read_from_snapshots(run_dir, ["db_par"], p, stride=2)
    assert run.branch == "minus"
    assert set(run.measured) == set(run.predicted) == {"db_par"}
    np.testing.assert_allclose(run.times, [0, 2])
    np.testing.assert_allclose(run.predicted["db_par"], [0.08, 0.08])


def test_zero_alfven_energy_has_zero_prediction(run_dir):
    write_snapshot(run_dir, 0)
    p = projection.load_parameters(run_dir)
    run = projection.read_from_snapshots(run_dir, FIELDS, p)
    assert np.isnan(run.l_perp[0])
    assert np.isnan(run.chi_a[0])
    np.testing.assert_array_equal(run.alignment, [0])
    for prediction in run.predicted.values():
        np.testing.assert_array_equal(prediction, [0])
    assert np.isnan(run.z_ratio).all()
    assert np.isnan(run.z_ratio_predicted).all()


@pytest.mark.parametrize("time_column", ["time", "t"])
def test_csv_uses_isotropic_closure_and_selected_field(run_dir, time_column):
    write_csv(run_dir, time_column)
    p = projection.load_parameters(run_dir)
    run = projection.read_from_csv(run_dir, ["db_par"], p)
    assert run.branch is None
    assert set(run.measured) == set(run.predicted) == {"db_par"}
    np.testing.assert_allclose(run.times, [0, 1])
    np.testing.assert_allclose(run.l_perp, [0.5, 0.25])
    np.testing.assert_allclose(run.alignment, [1 / np.sqrt(2)] * 2)
    assert np.isnan(run.chi_a).all()  # Legacy CSV lacks energy and parallel scale.
    np.testing.assert_allclose(run.measured["db_par"], [4, 8])
    np.testing.assert_allclose(run.predicted["db_par"], np.array([0.1, 0.05]) / np.sqrt(2))
    assert np.isnan(run.z_ratio).all()
    assert np.isnan(run.z_ratio_predicted).all()


@pytest.mark.parametrize("csv_flag", [[], ["--from-csv"]])
def test_cli_saves_selected_field_with_csv_or_fallback(run_dir, csv_flag):
    write_csv(run_dir)
    output = projection.main([str(run_dir), "--fields", "db_par", *csv_flag])
    assert output == run_dir / "slaved_projection.png"
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


@pytest.mark.parametrize("stride", ["0", "-1"])
def test_cli_rejects_nonpositive_stride(run_dir, stride, capsys):
    with pytest.raises(SystemExit) as error:
        projection.main([str(run_dir), "--stride", stride])
    assert error.value.code == 2
    assert "stride" in capsys.readouterr().err.lower()


def test_csv_measured_chi_and_title_average(run_dir, monkeypatch):
    with (run_dir / "scalar_diagnostics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time", "w_plus", "w_plus_kperp", "w_plus_kprl", "db_par_rms"])
        writer.writerows([[0, 1, 2, 1, 3], [1, 4, 3, 1, 6], [2, 1, 2, 0, 3]])
    p = projection.load_parameters(run_dir)
    run = projection.read_from_csv(run_dir, ["db_par"], p)
    np.testing.assert_allclose(run.chi_a, [2, 6, np.nan], equal_nan=True)

    titles = []

    def capture_title(fig, *, output_path, show, plt):
        titles.append(fig._suptitle.get_text())
        plt.close(fig)

    monkeypatch.setattr(projection, "finalize_figure", capture_title)
    projection.plot_series(run_dir, run, ["db_par"], p, run_dir / "unused.png")
    assert r"mean measured $\chi_A$ = 4" in titles[-1]
    run.chi_a[:] = np.nan
    projection.plot_series(run_dir, run, ["db_par"], p, run_dir / "unused.png")
    assert r"mean measured $\chi_A$ = n/a" in titles[-1]


@pytest.mark.parametrize("branch", ["plus", "minus"])
@pytest.mark.parametrize("gravity", [4.0, -4.0, 0.0])
def test_eq73_ratio_uses_measured_density_and_opposite_wave(run_dir, branch, gravity):
    other_branch = "minus" if branch == "plus" else "plus"
    write_snapshot(run_dir, 0, **{branch: (3.0, 1, 2), other_branch: (1.5, 1, 2)})
    path = run_dir / "input_copy.input"
    path.write_text(path.read_text().replace("g = 4.0", f"g = {gravity}"), encoding="utf-8")
    p = projection.load_parameters(run_dir)
    # Density is still read for Eq. (73), even when only the db_par panel is selected.
    run = projection.read_from_snapshots(run_dir, ["db_par"], p, branch=branch)
    np.testing.assert_allclose(run.z_ratio, [0.5])
    # Measured drho RMS is 2, pump RMS squared is 9*5/2, l_perp is 1/sqrt(5).
    np.testing.assert_allclose(run.z_ratio_predicted, [2 * abs(gravity) / (22.5 * np.sqrt(5))])


def test_eq73_csv_values_and_zero_pump(run_dir):
    with (run_dir / "scalar_diagnostics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time", "w_plus", "w_minus", "w_plus_kperp", "drho_rms", "db_par_rms"])
        writer.writerows([[0, 1, 0.25, 2, 0.5, 3], [1, 4, 1, 4, 1, 6], [2, 0, 9, 2, 1, 3]])
    p = projection.load_parameters(run_dir)
    run = projection.read_from_csv(run_dir, ["db_par"], p)
    np.testing.assert_allclose(run.z_ratio, [0.5, 0.5, np.nan], equal_nan=True)
    np.testing.assert_allclose(run.z_ratio_predicted, [0.25, 0.0625, np.nan], equal_nan=True)


@pytest.mark.parametrize("branch, ratio_label", [
    ("plus", r"z^-_{\rm rms}/z^+_{\rm rms}"),
    ("minus", r"z^+_{\rm rms}/z^-_{\rm rms}"),
])
def test_eq73_adds_fourth_panel_with_correct_branch_label(run_dir, monkeypatch, branch, ratio_label):
    write_snapshot(run_dir, 0, plus=(3, 1, 2), minus=(1.5, 1, 2))
    p = projection.load_parameters(run_dir)
    run = projection.read_from_snapshots(run_dir, FIELDS, p, branch=branch)

    def check_figure(fig, *, output_path, show, plt):
        assert len(fig.axes) == 4
        ax = fig.axes[-1]
        assert ratio_label in ax.get_title()
        assert ratio_label in ax.lines[0].get_label()
        assert "dimensionless" in ax.get_ylabel()
        np.testing.assert_allclose(ax.lines[0].get_ydata(), run.z_ratio)
        np.testing.assert_allclose(ax.lines[1].get_ydata(), run.z_ratio_predicted)
        plt.close(fig)

    monkeypatch.setattr(projection, "finalize_figure", check_figure)
    projection.plot_series(run_dir, run, FIELDS, p, run_dir / "unused.png")
