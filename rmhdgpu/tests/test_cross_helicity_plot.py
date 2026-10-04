"""Analytic checks for the normalized cross helicity and the saved-CSV plotting workflow."""

import csv

import numpy as np
import pytest

from vis import plot_cross_helicity as pch


def write_scalar_csv(path, columns):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(zip(*columns.values()))


def make_run(root, name, w_plus, w_minus, input_text=None):
    run = root / name
    run.mkdir()
    write_scalar_csv(run / "scalar_diagnostics.csv",
                     {"time": [0.0, 1.0, 2.0], "w_plus": w_plus, "w_minus": w_minus})
    if input_text is not None:
        (run / "input_copy.input").write_text(input_text)
    return run


def test_pure_z_plus_pure_z_minus_and_equal_energy():
    columns = {"time": np.array([0.0, 1.0, 2.0]),
               "w_plus": np.array([2.0, 0.0, 3.0]),
               "w_minus": np.array([0.0, 2.0, 3.0])}
    times, sigma_c = pch.calculate_cross_helicity(columns)
    np.testing.assert_array_equal(times, [0, 1, 2])
    np.testing.assert_allclose(sigma_c, [1.0, -1.0, 0.0])


def test_partial_reflection_value():
    # W+ = 3, W- = 1  ->  (3 - 1)/(3 + 1) = 1/2.
    columns = {"time": np.array([0.0]), "w_plus": np.array([3.0]), "w_minus": np.array([1.0])}
    assert pch.calculate_cross_helicity(columns)[1][0] == pytest.approx(0.5)


def test_zero_total_energy_is_left_undefined():
    columns = {"time": np.array([0.0, 1.0]),
               "w_plus": np.array([0.0, 4.0]),
               "w_minus": np.array([0.0, 0.0])}
    _, sigma_c = pch.calculate_cross_helicity(columns)
    np.testing.assert_allclose(sigma_c, [np.nan, 1.0], equal_nan=True)


def test_older_csv_with_t_column():
    columns = {"t": np.array([0.0, 1.0]), "w_plus": np.array([1.0, 1.0]), "w_minus": np.array([0.0, 0.0])}
    times, _ = pch.calculate_cross_helicity(columns)
    np.testing.assert_array_equal(times, [0, 1])


@pytest.mark.parametrize("missing", ["w_plus", "w_minus", "time"])
def test_missing_columns_have_clear_errors(missing):
    columns = {"time": np.array([0.0]), "w_plus": np.array([1.0]), "w_minus": np.array([0.0])}
    del columns[missing]
    with pytest.raises(SystemExit, match=missing):
        pch.calculate_cross_helicity(columns)


def test_run_directory_or_csv_path_and_labels(tmp_path):
    run = make_run(tmp_path, "run_a", [1.0, 1.0, 1.0], [0.0, 0.0, 0.0],
                   input_text="[physics]\ng = 0.06\nK_rho0 = 6.5  # gradient\n")
    plain = make_run(tmp_path, "run_b", [1.0, 1.0, 1.0], [0.0, 0.0, 0.0])
    from_dir, from_csv = pch.load_series([run, run / "scalar_diagnostics.csv"])
    np.testing.assert_array_equal(from_dir.sigma_c, from_csv.sigma_c)
    assert from_dir.label == "g=0.06, K_rho0=6.5"   # read from input_copy.input
    assert pch.load_series([plain])[0].label == "run_b"  # falls back to the directory name
    assert pch.load_series([run, plain], labels=["x", "y"])[1].label == "y"
    with pytest.raises(SystemExit, match="one label per run"):
        pch.load_series([run, plain], labels=["only one"])
    with pytest.raises(SystemExit, match="Could not find"):
        pch.load_series([tmp_path / "missing"])


def test_cli_overlays_runs_and_writes_png(tmp_path):
    control = make_run(tmp_path, "control", [1.0, 1.0, 1.0], [0.0, 0.0, 0.0])
    reflected = make_run(tmp_path, "reflected", [1.0, 0.5, 0.2], [0.0, 0.5, 0.8])
    output = tmp_path / "plots" / "cross_helicity.png"
    result = pch.main([str(control), str(reflected), "--labels", "no feedback", "g=0.6",
                       "--output", str(output)])
    assert result == output
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_single_run_default_output_is_beside_the_csv(tmp_path):
    run = make_run(tmp_path, "run", [1.0, 1.0, 1.0], [0.0, 0.0, 0.0])
    result = pch.main([str(run)])
    assert result == (run / "cross_helicity.png").resolve()
    assert result.exists()
