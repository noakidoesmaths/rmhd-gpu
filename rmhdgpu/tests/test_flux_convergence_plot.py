"""Small analytic checks for the closure convergence plot (vis/plot_flux_convergence.py)."""

import csv

import numpy as np
import pytest

from vis import plot_flux_convergence as convergence


# g = 0 with K_p0 = 0 and K_rho0 = 0.5 gives F_rho = -0.5, as in the user's passive-scalar runs.
PHYSICS = "[physics]\nvA = 1.0\ncs2_over_vA2 = 0.1\ng = 0.0\nK_p0 = 0.0\nK_rho0 = 0.5\n"
F_RHO = -0.5


def write_run(run_dir, columns):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "input_copy.input").write_text(PHYSICS, encoding="utf-8")
    with (run_dir / "scalar_diagnostics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(zip(*columns.values()))
    return run_dir


def run_columns(ratio, **extra):
    """Columns for a run whose closure ratio V_rho_x/(F_rho z+ l_perp) is `ratio`.

    W+ = 1 and k_perp_plus = 2 give z+ = 2 and l_perp = 1/2, so z+ l_perp = 1;
    W- = 1/4 gives z-/z+ = 1/2, and k_prl_plus = 1 gives chi_A = 4.
    """
    n = len(ratio)
    return {
        "time": np.arange(n, dtype=float),
        "w_plus": [1.0] * n, "w_minus": [0.25] * n,
        "k_perp_plus": [2.0] * n, "k_prl_plus": [1.0] * n, "w_plus_align": [0.5] * n,
        "drho_rms": [1.0] * n,
        "V_rho_x": F_RHO * np.asarray(ratio, dtype=float),
        **extra,
    }


def test_block_means_of_a_step_ignore_nan():
    times = np.arange(8.0)  # Two blocks over [0, 7]: t < 3.5 and t >= 3.5.
    values = np.array([1.0, 1.0, np.nan, 1.0, 3.0, 3.0, 3.0, np.nan])
    centres, means = convergence.block_means(times, values, 2)
    np.testing.assert_allclose(centres, [1.75, 5.25])
    np.testing.assert_allclose(means, [1.0, 3.0])


def test_block_means_drop_empty_blocks():
    centres, means = convergence.block_means([0.0, 10.0], [2.0, 4.0], 4)
    np.testing.assert_allclose(centres, [1.25, 8.75])
    np.testing.assert_allclose(means, [2.0, 4.0])


def test_drift_of_constant_and_ramp():
    times = np.arange(4.0)
    assert convergence.drift(times, [5.0] * 4) == pytest.approx(0.0)
    # First half t = 0, 1: mean 1.5. Second half t = 2, 3: mean 3.5.
    assert convergence.drift(times, [1.0, 2.0, 3.0, 4.0]) == pytest.approx(3.5 / 1.5 - 1.0)


def test_series_recover_the_closure_ratio(tmp_path):
    ratio = [0.1, 0.2, 0.3, 0.4]
    series = convergence.load_series(write_run(tmp_path / "run", run_columns(ratio)))
    np.testing.assert_allclose(series["ratio"], ratio)
    np.testing.assert_allclose(series["z_ratio"], 0.5)
    np.testing.assert_allclose(series["chi_a"], 4.0)
    assert series["nu"] is None  # No auto-dissipation column.


def test_main_writes_image_and_summary(tmp_path, capsys):
    ratio = [0.05] * 10
    write_run(tmp_path / "runs" / "auto", run_columns(ratio, auto_dissipation_nu_perp=[1e-6] * 10))
    write_run(tmp_path / "runs" / "manual", run_columns(ratio, auto_dissipation_nu_perp=[0.0] * 10))
    output = convergence.main([str(tmp_path / "runs" / "*"), "--blocks", "5"])

    assert output == (tmp_path / "runs" / "flux_convergence.png").resolve()
    assert output.exists()
    with output.with_suffix(".csv").open(encoding="utf-8") as handle:
        rows = {row["run"]: row for row in csv.DictReader(handle)}
    assert float(rows["auto"]["V_over_F_z_l"]) == pytest.approx(0.05)
    assert float(rows["auto"]["drift"]) == pytest.approx(0.0)
    assert float(rows["auto"]["auto_nu_perp"]) == pytest.approx(1e-6)
    assert np.isnan(float(rows["manual"]["auto_nu_perp"]))
    assert float(rows["auto"]["t_start"]) == pytest.approx(6.0)  # Last 40% of [0, 9].
    assert "drift" in capsys.readouterr().out


def test_single_run_saves_inside_its_folder(tmp_path):
    run_dir = write_run(tmp_path / "runs" / "one", run_columns([0.05] * 10))
    output = convergence.main([str(run_dir)])
    assert output == (run_dir / "flux_convergence.png").resolve()
    assert output.exists() and output.with_suffix(".csv").exists()
