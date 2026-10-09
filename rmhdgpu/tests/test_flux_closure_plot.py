"""Small analytic checks for the density-flux closure plot (vis/plot_flux_closure.py)."""

import csv

import numpy as np
import pytest

from vis import plot_flux_closure as closure
from vis.run_quantities import load_run


# vA = 2, chi = 3, g = 4, K_p0 = 0.5, K_rho0 = 0.3, so F_rho = -0.05 (as in test_slaved_projection).
PHYSICS = "[physics]\nvA = 2.0\ncs2_over_vA2 = 3.0\ng = {g}\nK_p0 = 0.5\nK_rho0 = 0.3\n"
F_RHO = -0.05

# With W+ = 1, <k_perp> = 2 and <k_par> = 1: z+ = 2, l_perp = 1/2, eta_turb = z+ l_perp/4 = 1/4,
# chi_A = z+ <k_perp>/(vA <k_par>) = 2, and the closure flux is eta_turb F_rho = -0.0125.
BASE_COLUMNS = {
    "time": [0.0, 1.0, 2.0, 3.0],
    "w_plus": [1.0] * 4, "w_minus": [0.0] * 4,
    "k_perp_plus": [2.0] * 4, "k_prl_plus": [1.0] * 4, "w_plus_align": [0.5] * 4,
    "drho_rms": [1.0, 2.0, 4.0, 8.0],
}


def write_run(run_dir, columns, g=4.0):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "input_copy.input").write_text(PHYSICS.format(g=g), encoding="utf-8")
    with (run_dir / "scalar_diagnostics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(zip(*columns.values()))
    return run_dir


def test_flux_and_eta_turb_come_from_run_quantities(tmp_path):
    flux = [-0.0125, -0.025, -0.0125, 0.0]
    run_dir = write_run(tmp_path / "run", {**BASE_COLUMNS, "V_rho_x": flux, "acr_g": [9.0] * 4})
    run = load_run(run_dir, ["drho"])
    np.testing.assert_allclose(run.v_rho_x, flux)  # The column wins over acr_g / g.
    np.testing.assert_allclose(run.eta_meas, np.array(flux) / F_RHO)
    np.testing.assert_allclose(run.eta_turb, 0.25)
    np.testing.assert_allclose(run.chi_a, 2.0)


def test_old_csv_falls_back_to_acr_g_over_g(tmp_path):
    run_dir = write_run(tmp_path / "run", {**BASE_COLUMNS, "acr_g": [-0.05, -0.1, 0.2, 0.4]})
    run = load_run(run_dir, ["drho"])
    np.testing.assert_allclose(run.v_rho_x, [-0.0125, -0.025, 0.05, 0.1])


def test_g_zero_without_column_has_no_flux(tmp_path, capsys):
    """acr_g is identically zero when g = 0, so it cannot stand in for V_rho_x."""
    run_dir = write_run(tmp_path / "run", {**BASE_COLUMNS, "acr_g": [0.0] * 4}, g=0.0)
    point = closure.read_run(run_dir)
    assert np.isnan(point["V_rho_x"])
    assert point["drho_rms"] == pytest.approx(6.0)  # Last 40% of [0, 3]: t = 2 and 3.
    assert "rerun to save it" in capsys.readouterr().out


def test_window_means(tmp_path):
    flux = [9.0, -0.0125, -0.025, 9.0]  # Rows outside [1, 2] must not enter the means.
    run_dir = write_run(tmp_path / "run", {**BASE_COLUMNS, "V_rho_x": flux})
    point = closure.read_run(run_dir, tmin=1.0, tmax=2.0)

    assert (point["t_start"], point["t_end"]) == (1.0, 2.0)
    assert point["F_rho"] == pytest.approx(F_RHO)
    assert point["chi_A"] == pytest.approx(2.0)
    # V_rho_x / F_rho = 0.25, 0.5 and z+ l_perp = 1, so the closure value would be 0.25.
    assert point["V_over_F_z_l"] == pytest.approx(0.375)
    assert point["eta_meas"] == pytest.approx(0.375)
    assert point["eta_turb"] == pytest.approx(0.25)
    # rms(drho) / (|F_rho| l_perp) = 2 / 0.025 and 4 / 0.025.
    assert point["drho_over_F_l"] == pytest.approx(120.0)
    assert point["alignment"] == pytest.approx(0.5)
    assert point["V_rho_x"] == pytest.approx(-0.01875)
    assert point["drho_rms"] == pytest.approx(3.0)


def test_zero_drive_has_no_ratios(tmp_path, capsys):
    run_dir = write_run(tmp_path / "run", {**BASE_COLUMNS, "V_rho_x": [0.0] * 4})
    physics = PHYSICS.format(g=0.0).replace("K_p0 = 0.5", "K_p0 = 0.0").replace("K_rho0 = 0.3", "K_rho0 = 0.0")
    (run_dir / "input_copy.input").write_text(physics, encoding="utf-8")
    point = closure.read_run(run_dir)
    assert point["F_rho"] == 0.0
    assert np.isnan(point["V_over_F_z_l"]) and np.isinf(point["drho_over_F_l"])
    assert "F_rho = 0" in capsys.readouterr().out


def test_main_writes_figure_and_summary(tmp_path, monkeypatch):
    write_run(tmp_path / "down", {**BASE_COLUMNS, "V_rho_x": [-0.0125] * 4})
    write_run(tmp_path / "counter", {**BASE_COLUMNS, "V_rho_x": [0.0125] * 4})
    write_run(tmp_path / "no_flux", {**BASE_COLUMNS}, g=0.0)

    figures = []

    def keep_figure(fig, *, output_path, show, plt):
        figures.append(fig)
        fig.savefig(output_path)
        plt.close(fig)

    monkeypatch.setattr(closure, "finalize_figure", keep_figure)
    output = closure.main([str(tmp_path / "*"), "--tmin", "1"])

    assert output == tmp_path / "flux_closure.png" and output.exists()
    with output.with_suffix(".csv").open(encoding="utf-8", newline="") as handle:
        rows = {row["run"]: row for row in csv.DictReader(handle)}
    assert float(rows["down"]["V_over_F_z_l"]) == pytest.approx(0.25)
    assert float(rows["counter"]["V_over_F_z_l"]) == pytest.approx(-0.25)
    assert rows["no_flux"]["V_rho_x"] == "nan"

    fig = figures[0]
    assert "t in [1, end]" in fig.get_suptitle()
    ax_flux, ax_drho = fig.axes
    assert ax_flux.get_xscale() == ax_flux.get_yscale() == "linear"
    assert not ax_flux.containers  # No error bars.
    # Runs are sorted: counter, down, no_flux. z+ l_perp = 1, so the points are V_rho_x / F_rho.
    np.testing.assert_allclose(ax_flux.lines[0].get_xdata(), [2.0] * 3)
    np.testing.assert_allclose(ax_flux.lines[0].get_ydata(), [-0.25, 0.25, np.nan])
    np.testing.assert_allclose(ax_flux.lines[1].get_ydata(), [0.25, 0.25])  # Closure line at 1/4.
    # Window t = 1, 2, 3: rms(drho) = 2, 4, 8 over |F_rho| l_perp = 0.025 (g = 4) or 0.15 (g = 0,
    # where F_rho = -K_rho0 = -0.3).
    np.testing.assert_allclose(ax_drho.lines[0].get_ydata(), [560 / 3, 560 / 3, 280 / 9])
    np.testing.assert_allclose(ax_drho.lines[1].get_ydata(), [0.5] * 3)  # Eq. (47): the alignment.
