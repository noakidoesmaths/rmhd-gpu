from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from vis.plot_chi_a import main as plot_chi_a_main
from vis.plot_fullfield import main as plot_fullfield_main
from vis.plot_scalars import main as plot_scalars_main
from vis.plot_spectra import main as plot_spectra_main


def test_plot_scalars_smoke(tmp_path) -> None:
    csv_path = tmp_path / "scalar_diagnostics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["time", "step", "alfvenic_energy", "total_energy_proxy"],
        )
        writer.writeheader()
        writer.writerow({"time": 0.0, "step": 0, "alfvenic_energy": 1.0, "total_energy_proxy": 1.2})
        writer.writerow({"time": 0.1, "step": 1, "alfvenic_energy": 0.9, "total_energy_proxy": 1.1})

    output_path = tmp_path / "scalars.png"
    result = plot_scalars_main([str(csv_path), "--output", str(output_path)])

    assert result == output_path.resolve()
    assert output_path.exists()


def test_plot_scalars_log_smoke_skips_nonpositive_columns(tmp_path) -> None:
    csv_path = tmp_path / "scalar_diagnostics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["time", "step", "positive_energy", "signed_budget"],
        )
        writer.writeheader()
        writer.writerow({"time": 0.0, "step": 0, "positive_energy": 1.0, "signed_budget": 0.1})
        writer.writerow({"time": 0.1, "step": 1, "positive_energy": 0.9, "signed_budget": -0.2})

    output_path = tmp_path / "scalars_log.png"
    result = plot_scalars_main(
        [str(csv_path), "--output", str(output_path), "--columns", "positive_energy", "signed_budget", "--log"]
    )

    assert result == output_path.resolve()
    assert output_path.exists()


@pytest.mark.parametrize("extra_args", [[], ["--log"]])
def test_plot_chi_a_smoke(tmp_path, extra_args) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with (run_dir / "scalar_diagnostics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["time", "w_plus", "k_perp_plus", "k_prl_plus", "vA"])
        writer.writeheader()
        writer.writerow({"time": 0.0, "w_plus": 0.04, "k_perp_plus": 2.0, "k_prl_plus": 1.0, "vA": 1.0})
        writer.writerow({"time": 0.1, "w_plus": 0.04, "k_perp_plus": 2.2, "k_prl_plus": 1.0, "vA": 1.0})

    result = plot_chi_a_main([str(run_dir), *extra_args])

    assert result == run_dir / "chi_a.png"
    assert result.exists()


def test_plot_scalars_log_requires_positive_columns(tmp_path) -> None:
    csv_path = tmp_path / "scalar_diagnostics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["time", "step", "signed_budget"],
        )
        writer.writeheader()
        writer.writerow({"time": 0.0, "step": 0, "signed_budget": 0.1})
        writer.writerow({"time": 0.1, "step": 1, "signed_budget": -0.2})

    with pytest.raises(SystemExit, match="No strictly positive scalar columns"):
        plot_scalars_main([str(csv_path), "--columns", "signed_budget", "--log"])


def test_plot_spectra_smoke(tmp_path) -> None:
    csv_path = tmp_path / "spectra.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["time", "step", "quantity", "kperp", "value"])
        writer.writeheader()
        for time_value in (0.0, 0.1):
            for quantity in ("u_perp", "b_perp"):
                for kperp, value in ((1.0, 1.0), (2.0, 0.5), (3.0, 0.25)):
                    writer.writerow(
                        {
                            "time": time_value,
                            "step": int(time_value * 10),
                            "quantity": quantity,
                            "kperp": kperp,
                            "value": value,
                        }
                    )

    output_dir = tmp_path / "spectra_plots"
    result = plot_spectra_main([str(csv_path), "--output-dir", str(output_dir)])

    assert result
    assert (output_dir / "u_perp.png").exists()
    assert (output_dir / "b_perp.png").exists()


def test_plot_fullfield_smoke(tmp_path) -> None:
    h5py = pytest.importorskip("h5py")
    snapshot_dir = tmp_path / "fullfields"
    snapshot_dir.mkdir()
    h5_path = snapshot_dir / "fullfield_0001.h5"
    with h5py.File(h5_path, "w") as handle:
        metadata = handle.create_group("metadata")
        metadata.create_dataset("x", data=np.linspace(0.0, 1.0, 4))
        metadata.create_dataset("y", data=np.linspace(0.0, 1.0, 4))
        metadata.create_dataset("z", data=np.linspace(0.0, 1.0, 4))
        metadata.create_dataset("field_names", data=np.asarray(["psi"], dtype=h5py.string_dtype()))
        group = handle.create_group("output")
        group.create_dataset("time", data=np.asarray(0.0))
        group.create_dataset("step", data=np.asarray(0))
        group.create_dataset("psi", data=np.arange(64, dtype=np.float64).reshape(4, 4, 4))

    output_dir = tmp_path / "fullfield_plots"
    result = plot_fullfield_main(
        [str(snapshot_dir), "--field", "psi", "--slice-dir", "z", "--output-dir", str(output_dir)]
    )

    assert result
    assert (output_dir / "psi_z_0001.png").exists()
