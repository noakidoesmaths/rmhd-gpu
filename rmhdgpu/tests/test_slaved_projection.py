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


def write_csv(run_dir, time_column="time"):
    """An old-style CSV, written before the energy and alignment columns existed.

    Only db_par is present: selecting it must not require other RMS columns.
    """
    with (run_dir / "scalar_diagnostics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([time_column, "w_plus_kperp", "db_par_rms"])
        writer.writerows([[0, 2, 3], [1, 4, 6]])


def write_branch_csv(run_dir):
    """Give each branch different columns, so reading the wrong one changes every number.

    W- is larger in the first row and smaller in the second. The last row has
    no Alfvenic field at all, which the solver saves as zero energy, <k> and alignment.
    """
    columns = {
        "time": [0, 1, 2],
        "w_plus": [1, 9, 0], "w_minus": [4, 1, 0],
        "w_plus_kperp": [2, 2, 0], "w_minus_kperp": [4, 5, 0],
        "w_plus_kprl": [1, 1, 0], "w_minus_kprl": [2, 2, 0],
        "w_plus_align": [0.9, 0.9, 0], "w_minus_align": [0.2, 0.4, 0],
        "drho_rms": [1, 1, 1], "du_par_rms": [2, 2, 2], "db_par_rms": [3, 6, 3],
    }
    with (run_dir / "scalar_diagnostics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(zip(*columns.values()))


def test_physical_forcings_and_normalizations(run_dir):
    p = projection.load_parameters(run_dir)
    assert projection.forcings(p) == pytest.approx({"drho": -0.05, "du_par": -0.1, "db_par": 0.2})
    assert projection.measured_divisors(p) == pytest.approx({"drho": 1, "du_par": 2, "db_par": 0.75})


def test_density_forcing_is_defined_without_gravity():
    p = projection.derived_parameters(
        {"vA": 2, "cs2_over_vA2": 3, "g": 0, "K_p0": 0.5, "K_rho0": 0.3}
    )
    assert projection.forcings(p)["drho"] == pytest.approx(-0.3)


def test_csv_reads_saved_alignment_on_the_stronger_branch(run_dir, capsys):
    write_branch_csv(run_dir)
    p = projection.load_parameters(run_dir)
    run = projection.read_from_csv(run_dir, ["db_par"], p)

    # W- is stronger in the first row, so z- stays the pump for the whole run.
    assert run.branch == "minus"
    assert run.source == "CSV: measured alignment"
    assert "isotropic" not in capsys.readouterr().out
    np.testing.assert_allclose(run.l_perp, [0.25, 0.2, np.nan], equal_nan=True)
    np.testing.assert_allclose(run.alignment, [0.2, 0.4, 0])
    # |F| = 0.2 for db_par; with no Alfvenic field the estimate is zero.
    np.testing.assert_allclose(run.predicted["db_par"], [0.2 * 0.25 * 0.2, 0.2 * 0.2 * 0.4, 0])
    # Pump z- RMS = 2 sqrt(W-) = 4, 2 and counter z+ RMS = 2, 6.
    np.testing.assert_allclose(run.z_ratio, [0.5, 3, np.nan], equal_nan=True)
    # chi_A = z_rms <k_perp> / (vA <k_par>) with vA = 2.
    np.testing.assert_allclose(run.chi_a, [4, 2.5, np.nan], equal_nan=True)
    # Eq. (73): l_perp |g| drho_rms / z_rms^2 with g = 4.
    np.testing.assert_allclose(run.z_ratio_predicted, [0.0625, 0.2, np.nan], equal_nan=True)


def test_csv_branch_can_be_chosen(run_dir):
    write_branch_csv(run_dir)
    p = projection.load_parameters(run_dir)
    run = projection.read_from_csv(run_dir, ["db_par"], p, branch="plus")
    assert run.branch == "plus"
    np.testing.assert_allclose(run.alignment, [0.9, 0.9, 0])
    np.testing.assert_allclose(run.l_perp, [0.5, 0.5, np.nan], equal_nan=True)
    np.testing.assert_allclose(run.z_ratio, [2, 1 / 3, np.nan], equal_nan=True)


@pytest.mark.parametrize("time_column", ["time", "t"])
def test_old_csv_falls_back_to_isotropic_alignment(run_dir, time_column, capsys):
    write_csv(run_dir, time_column)
    p = projection.load_parameters(run_dir)
    run = projection.read_from_csv(run_dir, ["db_par"], p)
    # Without energies the branch cannot be chosen; w_plus_kperp makes it plus.
    assert run.branch == "plus"
    assert "isotropy assumed" in run.source
    assert "no w_plus_align column" in capsys.readouterr().out
    assert set(run.measured) == set(run.predicted) == {"db_par"}
    np.testing.assert_allclose(run.times, [0, 1])
    np.testing.assert_allclose(run.l_perp, [0.5, 0.25])
    np.testing.assert_allclose(run.alignment, [1 / np.sqrt(2)] * 2)
    assert np.isnan(run.chi_a).all()  # Legacy CSV lacks energy and parallel scale.
    np.testing.assert_allclose(run.measured["db_par"], [4, 8])
    np.testing.assert_allclose(run.predicted["db_par"], np.array([0.1, 0.05]) / np.sqrt(2))
    assert np.isnan(run.z_ratio).all()
    assert np.isnan(run.z_ratio_predicted).all()


def test_minus_branch_needs_its_own_columns(run_dir):
    write_csv(run_dir)
    p = projection.load_parameters(run_dir)
    with pytest.raises(SystemExit, match="w_minus_kperp"):
        projection.read_from_csv(run_dir, ["db_par"], p, branch="minus")


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


def test_eq73_csv_values_and_zero_pump(run_dir):
    with (run_dir / "scalar_diagnostics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time", "w_plus", "w_minus", "w_plus_kperp", "drho_rms", "db_par_rms"])
        writer.writerows([[0, 1, 0.25, 2, 0.5, 3], [1, 4, 1, 4, 1, 6], [2, 0, 9, 2, 1, 3]])
    p = projection.load_parameters(run_dir)
    run = projection.read_from_csv(run_dir, ["db_par"], p)
    np.testing.assert_allclose(run.z_ratio, [0.5, 0.5, np.nan], equal_nan=True)
    np.testing.assert_allclose(run.z_ratio_predicted, [0.25, 0.0625, np.nan], equal_nan=True)


@pytest.mark.parametrize("branch, per_unit_gravity", [
    ("plus", [1 / 8, 1 / 72]),
    ("minus", [1 / 64, 1 / 20]),
])
@pytest.mark.parametrize("gravity", [4.0, -4.0, 0.0])
def test_eq73_uses_abs_gravity_and_the_selected_pump(run_dir, branch, per_unit_gravity, gravity):
    write_branch_csv(run_dir)
    path = run_dir / "input_copy.input"
    path.write_text(path.read_text().replace("g = 4.0", f"g = {gravity}"), encoding="utf-8")
    p = projection.load_parameters(run_dir)
    # Density is still read for Eq. (73), even when only the db_par panel is selected.
    run = projection.read_from_csv(run_dir, ["db_par"], p, branch=branch)
    # l_perp |g| drho_rms / (4 W_pump), undefined once the pump vanishes in the last row.
    expected = [*(abs(gravity) * np.array(per_unit_gravity)), np.nan]
    np.testing.assert_allclose(run.z_ratio_predicted, expected, equal_nan=True)


@pytest.mark.parametrize("branch, ratio_label", [
    ("plus", r"z^-_{\rm rms}/z^+_{\rm rms}"),
    ("minus", r"z^+_{\rm rms}/z^-_{\rm rms}"),
])
def test_eq73_adds_fourth_panel_with_correct_branch_label(run_dir, monkeypatch, branch, ratio_label):
    write_branch_csv(run_dir)
    p = projection.load_parameters(run_dir)
    run = projection.read_from_csv(run_dir, FIELDS, p, branch=branch)

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


def test_cli_saves_selected_field(run_dir):
    write_csv(run_dir)
    output = projection.main([str(run_dir), "--fields", "db_par"])
    assert output == run_dir / "slaved_projection.png"
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_cli_branch_is_used_and_shown(run_dir, monkeypatch):
    write_branch_csv(run_dir)
    titles = []

    def capture_title(fig, *, output_path, show, plt):
        titles.append(fig._suptitle.get_text())
        plt.close(fig)

    monkeypatch.setattr(projection, "finalize_figure", capture_title)
    projection.main([str(run_dir), "--fields", "db_par", "--branch", "plus"])
    assert "CSV: measured alignment; plus branch" in titles[-1]


def test_single_mode_solver_run_gives_the_analytic_estimate(tmp_path):
    """Run the solver on one Alfven mode, then check the saved columns and the estimate.

    A single Fourier mode stays single: the nonlinear brackets of a mode with
    itself vanish and the background-gradient terms are linear. With k = (1, 2, 1)
    every saved row must have <k_perp> = sqrt(5), <k_par> = 1 and
    rms(z_x)/z_rms = |k_y|/|k_perp| = 2/sqrt(5), so Eq. (47) is 0.4 |F| throughout.
    """
    from rmhdgpu.run import main as run_main

    input_path = tmp_path / "case.input"
    input_path.write_text(
        """
title = "Single-mode slaving check"
output_dir = "outputs"

[equations]
type = "inhomogeneous_rmhd_rho"

[grid]
Nx = 12
Ny = 12
Nz = 12

[time]
tmax = 0.02
dt_init = 0.005
dt_max = 0.005
use_variable_dt = false

[output]
t_out_scal = 0.01
t_out_spec = 0.0
t_out_full = 0.0

[backend]
backend = "numpy"

# K_b0 = g/vA^2 - chi*K_p0/gamma = 0.03, so all three fields are driven.
[physics]
vA = 1.0
cs2_over_vA2 = 0.1
K_rho0 = 0.65
g = 0.06
K_p0 = 0.5

# This initial condition's "minus" branch sets phi = -psi, which is a pure z+
# wave in the Elsasser convention z+ = zhat x grad_perp(phi - psi).
[initial_condition]
type = "alfven_mode"

[initial_condition.parameters]
k_indices = [1, 2, 1]
amplitude = 0.1
branch = "minus"

[runtime]
progress_output_every = 100
""".strip() + "\n",
        encoding="utf-8",
    )
    run_main([str(input_path)])
    run_dir = tmp_path / "outputs"

    with (run_dir / "scalar_diagnostics.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    for row in rows:
        assert float(row["w_plus_kperp"]) == pytest.approx(np.sqrt(5), rel=1.0e-12)
        assert float(row["w_plus_kprl"]) == pytest.approx(1.0, rel=1.0e-12)
        assert float(row["w_plus_align"]) == pytest.approx(2 / np.sqrt(5), rel=1.0e-12)

    p = projection.load_parameters(run_dir)
    run = projection.read_from_csv(run_dir, FIELDS, p)
    assert run.branch == "plus"
    assert run.source == "CSV: measured alignment"
    np.testing.assert_allclose(run.l_perp, 1 / np.sqrt(5), rtol=1.0e-12)
    np.testing.assert_allclose(run.alignment, 2 / np.sqrt(5), rtol=1.0e-12)
    forcing = projection.forcings(p)
    assert all(forcing[name] != 0.0 for name in FIELDS)
    for name in FIELDS:
        np.testing.assert_allclose(run.predicted[name], 0.4 * abs(forcing[name]), rtol=1.0e-12)
