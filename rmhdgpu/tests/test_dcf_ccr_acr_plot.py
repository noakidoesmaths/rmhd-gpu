"""Q_ACR in vis/plot_dcf_ccr_acr.py: Y_B + Y_g minus the pressure work V_psi . grad(p)."""

import csv

import pytest

from vis import plot_dcf_ccr_acr
from vis.run_quantities import load_parameters

V_RHO_X = 0.3
V_PSI_X = -0.2


def write_run(run_dir, K_p0, g=4.0, with_v_psi_column=True):
    """Constant-in-time budget columns built from V_RHO_X and V_PSI_X with the run's own K_b0."""
    run_dir.mkdir(parents=True)
    (run_dir / "input_copy.input").write_text(
        f"[physics]\nvA = 2.0\ncs2_over_vA2 = 3.0\ng = {g}\nK_p0 = {K_p0}\nK_rho0 = 1.0\n", encoding="utf-8",
    )
    p = load_parameters(run_dir)
    columns = {
        "time": [0.0, 1.0, 2.0, 3.0],
        "w_plus_rhs_buoyancy": [-1.0] * 4,
        "w_minus_rhs_buoyancy": [0.5] * 4,
        "w_plus_rhs_dissipation": [-2.0] * 4,
        "total_energy_rhs_acr_B": [-p.vA**2 * p.K_b0 * V_PSI_X] * 4,
        "total_energy_rhs_acr_g": [p.g * V_RHO_X] * 4,
    }
    if with_v_psi_column:
        columns["V_psi_x"] = [V_PSI_X] * 4
    with (run_dir / "scalar_diagnostics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(zip(*columns.values()))
    return p


def window_means(run_dir):
    plot_dcf_ccr_acr.main([str(run_dir)])
    with (run_dir / "dcf_ccr_acr.csv").open(encoding="utf-8") as handle:
        return {row["channel"]: float(row["window_mean"]) for row in csv.DictReader(handle)}


@pytest.mark.parametrize(
    "K_p0, g, with_v_psi_column",
    [
        (0.5, 4.0, False),  # K_b0 = 0.1: V_psi_x recovered from acr_B, no column needed
        (5.0 / 9.0, 4.0, True),  # K_b0 = 0: the V_psi_x column is used
        (0.0, 4.0, False),  # K_p0 = 0: no pressure work, Q_ACR = Y_B + Y_g
    ],
)
def test_q_acr_is_g_times_v_rho_minus_v_psi(tmp_path, K_p0, g, with_v_psi_column):
    p = write_run(tmp_path / "run", K_p0, g, with_v_psi_column)
    means = window_means(tmp_path / "run")

    # Force balance: g - vA^2 K_b0 = chi vA^2 K_p0 / gamma.
    assert p.g - p.vA**2 * p.K_b0 == pytest.approx(p.chi * p.vA**2 * p.K_p0 / p.gamma, rel=1e-12, abs=1e-14)
    assert means["pressure_work"] == pytest.approx(-(p.g - p.vA**2 * p.K_b0) * V_PSI_X, rel=1e-9, abs=1e-12)
    assert means["Q_ACR"] == pytest.approx(p.g * (V_RHO_X - V_PSI_X), rel=1e-9)
    assert means["Q_ACR"] == pytest.approx(means["Y_B"] + means["Y_g"] + means["pressure_work"], rel=1e-12)


def test_no_pressure_gradient_needs_no_v_psi(tmp_path):
    """g = K_p0 = 0 gives K_b0 = 0 and no pressure work, so Q_ACR = 0 without a V_psi_x column."""
    write_run(tmp_path / "run", K_p0=0.0, g=0.0, with_v_psi_column=False)
    means = window_means(tmp_path / "run")
    assert means["pressure_work"] == 0.0
    assert means["Q_ACR"] == 0.0
