"""vis/plot_stationarity.py: correlation time, drift test and the budget panel."""
import csv

import numpy as np

from vis.plot_stationarity import correlation_time, main


def red_noise(n_samples, memory, seed):
    """AR(1) noise x[i+1] = memory * x[i] + white noise, whose correlation is memory**lag."""
    rng = np.random.default_rng(seed)
    kicks = rng.standard_normal(n_samples)
    x = np.zeros(n_samples)
    for i in range(1, n_samples):
        x[i] = memory * x[i - 1] + kicks[i]
    return x


def test_correlation_time_of_red_noise():
    # For C(lag) = a**lag on a grid of spacing dt, the trapezoid area is
    # dt * (1/2 + a / (1 - a)) = dt * (1 + a) / (2 (1 - a)).
    dt, memory = 0.1, 0.9
    values = red_noise(200_000, memory, seed=1)
    time = dt * np.arange(values.size)
    expected = dt * (1 + memory) / (2 * (1 - memory))
    assert abs(correlation_time(time, values) / expected - 1) < 0.15


def test_correlation_time_of_constant_signal_is_zero():
    time = np.linspace(0.0, 10.0, 101)
    assert correlation_time(time, np.full_like(time, 3.0)) == 0.0


def test_steady_and_drifting_quantities(tmp_path):
    # total_energy: noise about a fixed mean, with a balanced budget (gain 1, loss 1).
    # w_plus: a clear ramp, far larger than its noise.
    time = 0.1 * np.arange(2001)
    total_energy = 5.0 + 0.2 * red_noise(time.size, 0.9, seed=2)
    w_plus = 1.0 + 0.01 * time + 0.01 * red_noise(time.size, 0.9, seed=3)
    with (tmp_path / "scalar_diagnostics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time", "total_energy", "w_plus", "total_energy_rhs_forcing",
                         "total_energy_rhs_dissipation", "total_energy_rhs_total"])
        for row in zip(time, total_energy, w_plus):
            writer.writerow([*row, 1.0, -1.0, 0.0])

    output = main([str(tmp_path), "--tmin", "50"])  # drho_rms and w_minus are missing: skipped

    assert output.is_file()
    with (tmp_path / "stationarity.csv").open() as handle:
        results = {row["quantity"]: row for row in csv.DictReader(handle)}
    assert set(results) == {"total_energy", "w_plus"}
    assert results["total_energy"]["steady"] == "True"
    assert results["w_plus"]["steady"] == "False"
    assert float(results["w_plus"]["relative_drift"]) > 0
