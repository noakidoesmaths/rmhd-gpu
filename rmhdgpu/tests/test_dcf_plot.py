"""Small analytic checks for DCF heating and the saved-CSV plotting workflow."""

import csv

import numpy as np
import pytest

from vis import plot_dcf as dcf


@pytest.fixture
def columns():
    return {
        name: np.asarray(values, dtype=float)
        for name, values in {
            "time": [0, 1, 2],
            "w_plus": [1, 4, 9],
            "w_minus": [0.25, 1, 2.25],
            "w_plus_kperp": [2, 3, 4],
            "w_minus_kperp": [1, 2, 3],
            "w_plus_kprl": [1, 2, 3],
            "w_minus_kprl": [0.5, 1, 1.5],
            "q_dcf": [0, 8, 18],
            "q_ccr_source": [3, 6, 9],
            "N_sq": [12, 12, 12],
            "vA": [2, 2, 2],
        }.items()
    }


@pytest.mark.parametrize(
    "branch, expected, measured, mean_chi",
    [("plus", [3, 4, 4.5], [0, 8, 18], 3),
     ("minus", [3, 3, 3], [3, 6, 9], 2)],
)
def test_measured_closure_and_energy_normalization(columns, branch, expected, measured, mean_chi):
    # W = z^2 / 4: plus z = [2, 4, 6], omega_nl = [4, 12, 24].
    series = dcf.calculate_series(columns, branch=branch)
    np.testing.assert_allclose(series.q_predicted, expected)
    np.testing.assert_allclose(series.q_measured, measured)
    np.testing.assert_array_equal(series.times, [0, 1, 2])
    assert series.branch == branch
    assert series.mean_prefactor == pytest.approx(2)
    assert series.mean_chi_a == pytest.approx(mean_chi)
    assert series.q_reference is None
    assert series.q_fixed_scale is None


def test_auto_branch_uses_whole_run_and_plus_wins_ties(columns):
    columns["w_plus"] = np.array([100.0, 1.0, 1.0])
    columns["w_minus"] = np.array([1.0, 2.0, 2.0])
    assert dcf.calculate_series(columns, tmin=1).branch == "plus"
    columns["w_plus"] = columns["w_minus"].copy()
    assert dcf.calculate_series(columns).branch == "plus"
    columns["w_minus"] *= 2
    assert dcf.calculate_series(columns).branch == "minus"


def test_assumed_chi_and_fixed_outer_scale(columns):
    series = dcf.calculate_series(columns, chi_a=2, l_perp=0.5)
    np.testing.assert_allclose(series.q_predicted, [3, 6, 9])
    np.testing.assert_allclose(series.q_reference, [3, 4, 4.5])
    np.testing.assert_allclose(series.q_fixed_scale, [3, 6, 9])
    assert series.mean_prefactor == pytest.approx(10 / 9)
    assert series.mean_chi_a == pytest.approx(3)
    assert series.chi_a == 2
    assert series.l_perp == 0.5


def test_legacy_csv_without_parallel_scale_or_alfven_speed(columns):
    del columns["vA"]
    series = dcf.calculate_series(columns)
    np.testing.assert_allclose(series.q_predicted, [3, 4, 4.5])
    assert series.mean_chi_a == pytest.approx(6)
    del columns["w_plus_kprl"]
    series = dcf.calculate_series(columns)
    np.testing.assert_allclose(series.q_predicted, [3, 4, 4.5])
    assert np.isnan(series.mean_chi_a)


def test_zero_parallel_wavenumber_keeps_gaps_and_direct_reference(columns):
    columns["w_plus_kprl"] = np.array([0.0, 2.0, 0.0])
    series = dcf.calculate_series(columns)
    np.testing.assert_allclose(series.q_predicted, [np.nan, 4, np.nan], equal_nan=True)
    assert series.mean_prefactor == pytest.approx(2)
    assert series.mean_chi_a == pytest.approx(3)
    assumed = dcf.calculate_series(columns, chi_a=2)
    np.testing.assert_allclose(assumed.q_predicted, [np.nan, 6, np.nan], equal_nan=True)
    np.testing.assert_allclose(assumed.q_reference, [3, 4, 4.5])


def test_tmin_changes_statistics_only(columns):
    series = dcf.calculate_series(columns, tmin=1)
    np.testing.assert_allclose(series.q_predicted, [3, 4, 4.5])
    np.testing.assert_allclose(series.q_measured, [0, 8, 18])
    np.testing.assert_array_equal(series.times, [0, 1, 2])
    assert series.mean_prefactor == pytest.approx(3)
    assert series.mean_chi_a == pytest.approx(3.5)
    assert series.tmin == 1
    empty = dcf.calculate_series(columns, tmin=10)
    assert np.isnan(empty.mean_prefactor)
    assert np.isnan(empty.mean_chi_a)


def test_zero_energy_keeps_undefined_closure_and_zero_fixed_scale(columns):
    columns["w_plus"][:] = 0
    series = dcf.calculate_series(columns, branch="plus", l_perp=0.5)
    assert np.isnan(series.q_predicted).all()
    assert np.isnan(series.mean_prefactor)
    np.testing.assert_array_equal(series.q_fixed_scale, [0, 0, 0])


def test_explicit_minus_branch_only_needs_its_own_columns(columns):
    for name in ("w_plus", "w_plus_kperp", "w_plus_kprl", "q_dcf"):
        del columns[name]
    series = dcf.calculate_series(columns, branch="minus")
    np.testing.assert_allclose(series.q_predicted, [3, 3, 3])


@pytest.mark.parametrize("n_sq", [12, -12, 0])
def test_stratification_sign(columns, n_sq):
    columns["N_sq"][:] = n_sq
    series = dcf.calculate_series(columns, l_perp=0.5)
    assert series.n_sq == n_sq
    np.testing.assert_allclose(series.q_predicted, np.array([3, 4, 4.5]) * n_sq / 12)
    np.testing.assert_allclose(series.q_fixed_scale, np.array([3, 6, 9]) * n_sq / 12)


@pytest.mark.parametrize(
    "missing, branch", [("N_sq", "plus"), ("w_plus_kperp", "plus"),
                        ("q_ccr_source", "minus"), ("w_minus", "auto")],
)
def test_missing_columns_have_clear_errors(columns, missing, branch):
    del columns[missing]
    with pytest.raises(SystemExit, match=missing):
        dcf.calculate_series(columns, branch=branch)


def test_invalid_assumed_chi_has_clear_errors(columns):
    with pytest.raises(SystemExit, match="nonzero"):
        dcf.calculate_series(columns, chi_a=0)
    del columns["w_plus_kprl"]
    with pytest.raises(SystemExit, match="w_plus_kprl"):
        dcf.calculate_series(columns, chi_a=1)


def test_csv_reader_and_cli_png(tmp_path, columns):
    # Older diagnostics use "t" in place of "time".
    columns["t"] = columns.pop("time")
    csv_path = tmp_path / "scalar_diagnostics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(zip(*columns.values()))
    loaded = dcf.read_scalar_csv(csv_path)
    for name, expected in columns.items():
        np.testing.assert_array_equal(loaded[name], expected)
    output = tmp_path / "plots" / "dcf.png"
    result = dcf.main([str(csv_path), "--branch", "minus", "--chi-a", "1",
                       "--l-perp", "0.5", "--tmin", "1", "--output", str(output)])
    assert result == output
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
