"""vis/plot_energy_flux.py on a hand-made energy_flux.csv with a known flux."""
import csv

import numpy as np
import pytest

from vis.plot_energy_flux import main

# Four k_perp shells (centres 0.5, 1.5, 2.5, 3.5; upper edges K = 1, 2, 3, 4).
# Energy is injected in shell 0, carried up by the nonlinearity, and dissipated in shells 2 and 3:
TRANSFER = [-1.0, 0.0, 0.5, 0.5]      # sums to zero
LINEAR = [1.0, 0.0, 0.0, 0.0]         # eps = 1
DISSIPATION = [0.0, 0.0, -0.5, -0.5]
# so Pi(K) = -sum_{k < K} T = [1, 1, 0.5, 0]: a flat "cascade" through K = 1, 2.
EXPECTED_FLUX = [1.0, 1.0, 0.5, 0.0]


def test_flux_and_budget_from_known_rates(tmp_path):
    with (tmp_path / "energy_flux.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time", "step", "quantity", "kperp", "value"])
        for step, time in enumerate(np.linspace(0.0, 10.0, 21)):
            for quantity, values in (("nonlinear_total", TRANSFER), ("linear_total", LINEAR),
                                     ("dissipation_total", DISSIPATION)):
                for kperp, value in zip([0.5, 1.5, 2.5, 3.5], values):
                    writer.writerow([time, step, quantity, kperp, value])

    output = main([str(tmp_path), "--tmin", "0"])

    assert output.is_file()
    with (tmp_path / "energy_flux_summary.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [float(row["K"]) for row in rows] == [1.0, 2.0, 3.0, 4.0]
    assert [float(row["Pi_total"]) for row in rows] == pytest.approx(EXPECTED_FLUX)
    assert [float(row["dissipated_below_K"]) for row in rows] == pytest.approx([0.0, 0.0, 0.5, 1.0])
    assert float(rows[0]["eps"]) == pytest.approx(1.0)
    # Steady-state budget: injected below K = Pi(K) + dissipated below K, in every shell.
    for row in rows:
        assert float(row["injected_linear_below_K"]) == pytest.approx(
            float(row["Pi_total"]) + float(row["dissipated_below_K"]))
