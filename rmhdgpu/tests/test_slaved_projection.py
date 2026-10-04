"""Small analytic checks for the slaved-amplitude plotting script."""

import csv

import numpy as np
import pytest

from rmhdgpu.diagnostics.compressive_channels import background_drives
from vis import plot_slaved_projection as projection


FIELDS = ["drho", "du_par", "db_par"]


@pytest.fixture
def run_dir(tmp_path):
    # vA = 2, alpha = 3/4, and the drives are F = (-0.05, -0.1, 0.2).
    (tmp_path / "input_copy.input").write_text(
        "[physics]\nvA = 2.0\ncs2_over_vA2 = 3.0\ng = 4.0\n"
        "K_p0 = 0.5\nK_rho0 = 0.3\n",
        encoding="utf-8",
    )
    return tmp_path


def write_columns(run_dir, columns):
    with (run_dir / "scalar_diagnostics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(zip(*columns.values()))


def write_old_csv(run_dir, time_column="time"):
    """An old-style CSV, written before the energy and alignment columns existed.

    Only db_par is present: selecting it must not require other RMS columns.
    """
    write_columns(run_dir, {time_column: [0, 1], "w_plus_kperp": [2, 4], "db_par_rms": [3, 6]})


def write_full_csv(run_dir):
    """Give z+ and z- different columns, so reading the wrong wave changes every number.

    The last row has no Alfvenic field at all, which the solver saves as zero
    energy, <k> and alignment.
    """
    write_columns(run_dir, {
        "time": [0, 1, 2],
        "w_plus": [4, 1, 0], "w_minus": [1, 9, 0],
        "w_plus_kperp": [4, 5, 0], "w_minus_kperp": [2, 2, 0],
        "w_plus_kprl": [2, 2, 0], "w_minus_kprl": [1, 1, 0],
        "w_plus_align": [0.2, 0.4, 0], "w_minus_align": [0.9, 0.9, 0],
        "drho_rms": [1, 1, 1], "du_par_rms": [2, 2, 2], "db_par_rms": [3, 6, 3],
        "u_perp_rms": [2, 4, 0], "b_perp_rms": [1, 3, 0],
    })


def test_csv_uses_the_z_plus_scales_and_saved_alignment(run_dir, capsys):
    write_full_csv(run_dir)
    p = projection.load_parameters(run_dir)
    run = projection.read_from_csv(run_dir, ["db_par"], p)

    assert capsys.readouterr().out == ""  # W+ > W- at t = 0, so no warning.
    np.testing.assert_allclose(run.l_perp, [0.25, 0.2, np.nan], equal_nan=True)
    np.testing.assert_allclose(run.alignment, [0.2, 0.4, 0])
    # |F_b| = 0.2; with no Alfvenic field the estimate is zero.
    np.testing.assert_allclose(run.predicted["db_par"], [0.2 * 0.25 * 0.2, 0.2 * 0.2 * 0.4, 0])
    # db_par is shown as db_par/alpha, alpha = 3/4.
    np.testing.assert_allclose(run.measured["db_par"], [4, 8, 4])
    # z+ = 2 sqrt(W+) = 4, 2 and z- = 2, 6.
    np.testing.assert_allclose(run.z_ratio, [0.5, 3, np.nan], equal_nan=True)
    # chi_A = z+ <k_perp> / (vA <k_par>) with vA = 2.
    np.testing.assert_allclose(run.chi_a, [4, 2.5, np.nan], equal_nan=True)
    # Eq. (73): l_perp |g| drho_rms / z+^2 with g = 4.
    np.testing.assert_allclose(run.z_ratio_predicted, [0.0625, 0.2, np.nan], equal_nan=True)


def test_elsasser_and_perp_amplitudes_are_normalised_by_va(run_dir):
    write_full_csv(run_dir)
    p = projection.load_parameters(run_dir)
    run = projection.read_from_csv(run_dir, ["db_par"], p)
    # z_rms = 2 sqrt(W): z+ = 4, 2, 0 and z- = 2, 6, 0, divided by vA = 2.
    np.testing.assert_allclose(run.z_plus, [2, 1, 0])
    np.testing.assert_allclose(run.z_minus, [1, 3, 0])
    # b_perp is in Alfven units, so b_perp_rms/vA is delta B_perp/B_0.
    np.testing.assert_allclose(run.u_perp, [1, 2, 0])
    np.testing.assert_allclose(run.b_perp, [0.5, 1.5, 0])


def test_a_run_that_starts_in_z_minus_is_flagged(run_dir, capsys):
    write_columns(run_dir, {"time": [0, 1], "w_plus": [1, 9], "w_minus": [4, 1],
                            "w_plus_kperp": [2, 2], "db_par_rms": [3, 6]})
    p = projection.load_parameters(run_dir)
    run = projection.read_from_csv(run_dir, ["db_par"], p)
    assert "W- > W+ at the first saved time" in capsys.readouterr().out
    np.testing.assert_allclose(run.z_ratio, [2, 1 / 3])  # Still z-/z+, never relabelled.


@pytest.mark.parametrize("time_column", ["time", "t"])
def test_old_csv_leaves_the_eq47_estimate_empty(run_dir, time_column, capsys):
    write_old_csv(run_dir, time_column)
    p = projection.load_parameters(run_dir)
    run = projection.read_from_csv(run_dir, ["db_par"], p)
    assert capsys.readouterr().out == ""
    assert set(run.measured) == set(run.predicted) == {"db_par"}
    np.testing.assert_allclose(run.times, [0, 1])
    np.testing.assert_allclose(run.l_perp, [0.5, 0.25])
    np.testing.assert_allclose(run.measured["db_par"], [4, 8])
    # No w_plus_align column: no alignment, so no Eq. (47) estimate, rather than a guess.
    for curve in (run.alignment, run.predicted["db_par"], run.chi_a, run.z_plus, run.z_minus,
                  run.z_ratio, run.z_ratio_predicted, run.u_perp, run.b_perp):
        assert np.isnan(curve).all()


def test_old_csv_plots_the_measured_curve_and_names_the_empty_ones(run_dir, monkeypatch):
    write_old_csv(run_dir)
    p = projection.load_parameters(run_dir)
    run = projection.read_from_csv(run_dir, ["db_par"], p)
    notes = []

    def capture_notes(fig, *, output_path, show, plt):
        notes.extend([text.get_text() for text in ax.texts] for ax in fig.axes if ax.get_visible())
        plt.close(fig)

    monkeypatch.setattr(projection, "finalize_figure", capture_notes)
    projection.plot_series(run_dir, run, ["db_par"], p, run_dir / "unused.png")
    field, reflection, elsasser, perp = notes
    assert field == ["No data in this CSV for:\nSlaved estimate (Eq. 47)"]
    assert len(reflection) == 1 and "Measured" in reflection[0]
    assert len(elsasser) == 1 and "z^+" in elsasser[0] and "z^-" in elsasser[0]
    assert len(perp) == 1


def test_csv_measured_chi_and_title_average(run_dir, monkeypatch):
    write_columns(run_dir, {"time": [0, 1, 2], "w_plus": [1, 4, 1], "w_plus_kperp": [2, 3, 2],
                            "w_plus_kprl": [1, 1, 0], "db_par_rms": [3, 6, 3]})
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


def test_eq73_csv_values_and_zero_z_plus(run_dir):
    write_columns(run_dir, {"time": [0, 1, 2], "w_plus": [1, 4, 0], "w_minus": [0.25, 1, 9],
                            "w_plus_kperp": [2, 4, 2], "drho_rms": [0.5, 1, 1], "db_par_rms": [3, 6, 3]})
    p = projection.load_parameters(run_dir)
    run = projection.read_from_csv(run_dir, ["db_par"], p)
    np.testing.assert_allclose(run.z_ratio, [0.5, 0.5, np.nan], equal_nan=True)
    np.testing.assert_allclose(run.z_ratio_predicted, [0.25, 0.0625, np.nan], equal_nan=True)


@pytest.mark.parametrize("gravity", [4.0, -4.0, 0.0])
def test_eq73_uses_abs_gravity(run_dir, gravity):
    write_full_csv(run_dir)
    path = run_dir / "input_copy.input"
    path.write_text(path.read_text().replace("g = 4.0", f"g = {gravity}"), encoding="utf-8")
    p = projection.load_parameters(run_dir)
    # Density is still read for Eq. (73), even when only the db_par panel is selected.
    run = projection.read_from_csv(run_dir, ["db_par"], p)
    # l_perp |g| drho_rms / (4 W+), undefined once z+ vanishes in the last row.
    expected = [*(abs(gravity) * np.array([1 / 64, 1 / 20])), np.nan]
    np.testing.assert_allclose(run.z_ratio_predicted, expected, equal_nan=True)


def test_panels_show_the_fields_then_eq73_z_and_perp_amplitudes(run_dir, monkeypatch):
    write_full_csv(run_dir)
    p = projection.load_parameters(run_dir)
    run = projection.read_from_csv(run_dir, FIELDS, p)
    ratio_label = r"z^-_{\rm rms}/z^+_{\rm rms}"

    def check_figure(fig, *, output_path, show, plt):
        # Top row: the three fields. Bottom row: Eq. (73) ratio, z+/z-, u_perp/b_perp.
        assert fig.axes[0].get_gridspec().get_geometry() == (2, 3)
        visible = [ax for ax in fig.axes if ax.get_visible()]
        assert len(visible) == 6
        assert not any(ax.yaxis.get_major_formatter().get_useOffset() for ax in visible)
        assert not any(ax.texts for ax in visible)  # Every curve has data.
        ax = visible[3]
        assert ratio_label in ax.get_title()
        assert ratio_label in ax.lines[0].get_label()
        assert "dimensionless" in ax.get_ylabel()
        np.testing.assert_allclose(ax.lines[0].get_ydata(), run.z_ratio)
        np.testing.assert_allclose(ax.lines[1].get_ydata(), run.z_ratio_predicted)
        ax = visible[4]
        np.testing.assert_allclose(ax.lines[0].get_ydata(), run.z_plus)
        np.testing.assert_allclose(ax.lines[1].get_ydata(), run.z_minus)
        ax = visible[5]
        np.testing.assert_allclose(ax.lines[0].get_ydata(), run.u_perp)
        np.testing.assert_allclose(ax.lines[1].get_ydata(), run.b_perp)
        plt.close(fig)

    monkeypatch.setattr(projection, "finalize_figure", check_figure)
    projection.plot_series(run_dir, run, FIELDS, p, run_dir / "unused.png")


def test_cli_saves_selected_field(run_dir):
    write_old_csv(run_dir)
    output = projection.main([str(run_dir), "--fields", "db_par"])
    assert output == run_dir / "slaved_projection.png"
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


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

# A pure z+ = u_perp - b_perp wave, phi = -psi.
[initial_condition]
type = "alfven_mode"

[initial_condition.parameters]
k_indices = [1, 2, 1]
amplitude = 0.1
branch = "z_plus"

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
        # u^2 + b^2 = 2 (W+ + W-) exactly. The wave starts as pure z+ (|u| = |b|);
        # the background gradients couple it weakly to the compressive fields, so
        # |u| and |b| drift apart only slightly over this short run.
        u, b = float(row["u_perp_rms"]), float(row["b_perp_rms"])
        w_total = float(row["w_plus"]) + float(row["w_minus"])
        assert u**2 + b**2 == pytest.approx(2 * w_total, rel=1.0e-12)
        assert u == pytest.approx(b, rel=1.0e-4)
    assert float(rows[0]["w_minus"]) <= 1.0e-28 * float(rows[0]["w_plus"])

    p = projection.load_parameters(run_dir)
    run = projection.read_from_csv(run_dir, FIELDS, p)
    np.testing.assert_allclose(run.l_perp, 1 / np.sqrt(5), rtol=1.0e-12)
    np.testing.assert_allclose(run.alignment, 2 / np.sqrt(5), rtol=1.0e-12)
    drives = background_drives(p)
    assert all(drives[name] != 0.0 for name in FIELDS)
    for name in FIELDS:
        np.testing.assert_allclose(run.predicted[name], 0.4 * abs(drives[name]), rtol=1.0e-12)
