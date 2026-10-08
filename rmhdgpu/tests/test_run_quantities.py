"""Checks for vis/run_quantities.py, the one place derived run quantities are defined."""

import csv

import numpy as np
import pytest

from vis import run_quantities as rq


# vA = 2, chi = 3, g = 4, K_p0 = 0.5, K_rho0 = 0.3: alpha = 3/4 and F = (-0.05, -0.1, 0.2).
PHYSICS = "[physics]\nvA = 2.0\ncs2_over_vA2 = 3.0\ng = 4.0\nK_p0 = 0.5\nK_rho0 = 0.3\n"


def write_run(run_dir, columns, physics=PHYSICS):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "input_copy.input").write_text(physics, encoding="utf-8")
    with (run_dir / "scalar_diagnostics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(zip(*columns.values()))
    return run_dir


def test_every_formula_on_one_row(tmp_path):
    run = rq.load_run(write_run(tmp_path / "run", {
        "time": [0.0], "w_plus": [4.0], "w_minus": [1.0],
        "k_perp_plus": [2.0], "k_prl_plus": [0.5], "w_plus_align": [0.6],
        "drho_rms": [0.3], "du_par_rms": [0.4], "db_par_rms": [0.6], "V_rho_x": [-0.01],
    }))
    assert run.z_plus[0] == pytest.approx(4.0)  # 2 sqrt(W+)
    assert run.z_minus[0] == pytest.approx(2.0)
    assert run.l_perp[0] == pytest.approx(0.5)  # 1 / k_perp_plus
    assert run.chi_a[0] == pytest.approx(4.0 * 2.0 / (2.0 * 0.5))  # z+ k_perp / (vA k_prl)
    assert run.eta_turb[0] == pytest.approx(4.0 * 0.5 / 4)  # z+ l_perp / 4
    assert run.v_rho_x[0] == pytest.approx(-0.01)
    assert run.drives == pytest.approx({"drho": -0.05, "du_par": -0.1, "db_par": 0.2})
    # Eq. (47) units: du_par / vA and db_par / alpha.
    assert [run.measured[name][0] for name in rq.FIELDS] == pytest.approx([0.3, 0.2, 0.8])
    # Eq. (47): |F| l_perp alignment = |F| * 0.3.
    assert [run.predicted[name][0] for name in rq.FIELDS] == pytest.approx([0.015, 0.03, 0.06])
    assert run.z_ratio[0] == pytest.approx(0.5)
    # Eq. (73): l_perp |g| rms(drho) / z+^2 = 0.5 * 4 * 0.3 / 16.
    assert run.z_ratio_predicted[0] == pytest.approx(0.0375)


def test_old_column_names_are_renamed(tmp_path):
    columns = rq.read_columns(write_run(tmp_path / "old", {
        "t": [0, 1], "w_plus_kperp": [2, 3], "w_minus_kperp": [4, 5],
        "w_plus_kprl": [6, 7], "w_minus_kprl": [8, 9],
    }) / "scalar_diagnostics.csv")
    assert set(columns) == {"time", "k_perp_plus", "k_perp_minus", "k_prl_plus", "k_prl_minus"}
    np.testing.assert_array_equal(columns["k_prl_minus"], [8, 9])


def test_missing_columns_read_as_nan_and_acr_g_stands_in_for_v_rho(tmp_path):
    run = rq.load_run(write_run(tmp_path / "run", {
        "time": [0, 1], "drho_rms": [1, 2], "acr_g": [0.4, -0.8],
    }), ["drho"])
    assert set(run.measured) == {"drho"}
    np.testing.assert_allclose(run.v_rho_x, [0.1, -0.2])  # acr_g / g, with g = 4
    for curve in (run.z_plus, run.l_perp, run.chi_a, run.eta_turb, run.predicted["drho"]):
        assert np.isnan(curve).all()


def test_g_zero_has_no_flux_without_the_column(tmp_path):
    run = rq.load_run(write_run(tmp_path / "run", {"time": [0, 1], "drho_rms": [1, 2], "acr_g": [0, 0]},
                                physics=PHYSICS.replace("g = 4.0", "g = 0.0")), ["drho"])
    assert np.isnan(run.v_rho_x).all()


def test_missing_files_and_columns_raise_value_errors(tmp_path):
    with pytest.raises(ValueError, match="input_copy.input"):
        rq.load_run(tmp_path)
    run_dir = write_run(tmp_path / "run", {"time": [0, 1], "drho_rms": [1, 2]})
    with pytest.raises(ValueError, match="db_par_rms"):
        rq.load_run(run_dir, ["db_par"])


def test_time_window_defaults_and_errors():
    times = np.linspace(0.0, 10.0, 11)
    np.testing.assert_array_equal(rq.time_window(times), times >= 6.0)
    np.testing.assert_array_equal(rq.time_window(times, tmin=2.0, tmax=4.0),
                                  (times >= 2.0) & (times <= 4.0))
    with pytest.raises(ValueError, match="fewer than two"):
        rq.time_window(times, tmin=2.5, tmax=2.9)
    with pytest.raises(ValueError, match="tail_fraction"):
        rq.time_window(times, tail_fraction=0.0)
    with pytest.raises(ValueError, match="tmax"):
        rq.time_window(times, tmax=np.nan)
    with pytest.raises(ValueError, match="strictly increasing"):
        rq.time_window([0.0, 1.0, 1.0, 2.0])


def test_resolve_run_dirs(tmp_path):
    for name in ("b", "a"):
        (tmp_path / name).mkdir()
    found = rq.resolve_run_dirs([str(tmp_path / "*"), str(tmp_path / "a")])
    assert found == [(tmp_path / "a").resolve(), (tmp_path / "b").resolve()]
    with pytest.raises(ValueError, match="No matching"):
        rq.resolve_run_dirs([str(tmp_path / "missing_*")])
