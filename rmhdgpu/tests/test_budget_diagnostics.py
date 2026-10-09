from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from rmhdgpu.run import main


def _write_budget_input(
    path: Path,
    *,
    tmax: float = 0.02,
    dt: float = 0.005,
    t_out_scal: float = 0.005,
    use_forcing: bool = False,
    forcing_epsilon: float = 0.02,
    zero_initial: bool = False,
    initial_condition_type: str | None = None,
    dissipative: bool = False,
    nu_perp: float = 0.02,
    cs2_over_vA2: float = 1.0,
) -> None:
    lines = [
        'title = "Budget test"',
        'output_dir = "outputs"',
        "",
        "[grid]",
        "Nx = 8",
        "Ny = 8",
        "Nz = 8",
        "",
        "[time]",
        f"tmax = {tmax}",
        f"dt_init = {dt}",
        f"dt_max = {dt}",
        "use_variable_dt = false",
        "",
        "[output]",
        f"t_out_scal = {t_out_scal}",
        "t_out_spec = 0.0",
        "t_out_full = 0.0",
        "",
        "[backend]",
        'backend = "numpy"',
        "",
        "[physics]",
        f"cs2_over_vA2 = {cs2_over_vA2}",
        "",
        "[runtime]",
        "progress_output_every = 100",
    ]
    if use_forcing:
        lines.extend(
            [
                "",
                "[forcing]",
                "use_forcing = true",
                "forcing_seed = 7",
                "n_min_force = 1.0",
                "n_max_force = 2.0",
                "alpha_force = 0.0",
                "",
                "[forcing.field_energy_injection_rates]",
                f"psi = {forcing_epsilon}",
                f"omega = {forcing_epsilon}",
            ]
        )
    if initial_condition_type is not None:
        lines.extend(["", "[initial_condition]", f'type = "{initial_condition_type}"'])
    elif zero_initial:
        lines.extend(["", "[initial_condition]", 'type = "zero"'])
    if dissipative:
        for field_name in ("psi", "omega", "upar", "dbpar", "s"):
            lines.extend(
                [
                    "",
                    f"[dissipation.{field_name}]",
                    f"nu_perp = {nu_perp}",
                    "nu_par = 0.0",
                    "n_perp = 2",
                    "n_par = 1",
                ]
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_scalar_rows(path: Path) -> tuple[list[str], list[dict[str, float]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [{key: float(value) for key, value in row.items()} for row in reader]
        assert reader.fieldnames is not None
        return list(reader.fieldnames), rows


def test_budget_columns_written(tmp_path) -> None:
    input_file = tmp_path / "budget.input"
    _write_budget_input(input_file, dissipative=True)

    main([str(input_file)])

    fieldnames, rows = _read_scalar_rows(tmp_path / "outputs" / "scalar_diagnostics.csv")
    assert rows
    assert "total_energy" in fieldnames
    assert "total_energy_rhs_dissipation" in fieldnames
    assert "total_energy_rhs_forcing" in fieldnames
    assert "total_energy_rhs_total" in fieldnames


def test_budget_terms_zero_when_expected(tmp_path) -> None:
    input_file = tmp_path / "ideal.input"
    _write_budget_input(input_file, dissipative=False, use_forcing=False)

    main([str(input_file)])

    _, rows = _read_scalar_rows(tmp_path / "outputs" / "scalar_diagnostics.csv")
    for row in rows:
        assert abs(row["total_energy_rhs_dissipation"]) < 1.0e-12
        assert abs(row["total_energy_rhs_forcing"]) < 1.0e-12
        assert abs(row["total_energy_rhs_total"]) < 1.0e-12


def test_dissipation_budget_sign_and_presence(tmp_path) -> None:
    input_file = tmp_path / "damped.input"
    _write_budget_input(input_file, dissipative=True, use_forcing=False, nu_perp=0.05)

    main([str(input_file)])

    _, rows = _read_scalar_rows(tmp_path / "outputs" / "scalar_diagnostics.csv")
    dissipation_terms = [row["total_energy_rhs_dissipation"] for row in rows[1:]]

    assert dissipation_terms
    assert all(value <= 1.0e-12 for value in dissipation_terms)
    assert any(value < -1.0e-8 for value in dissipation_terms)


def test_forcing_budget_presence(tmp_path) -> None:
    input_file = tmp_path / "forced.input"
    _write_budget_input(input_file, use_forcing=True, zero_initial=True, dt=0.002, tmax=0.01, t_out_scal=0.002)

    main([str(input_file)])

    _, rows = _read_scalar_rows(tmp_path / "outputs" / "scalar_diagnostics.csv")
    forcing_terms = np.asarray([row["total_energy_rhs_forcing"] for row in rows[1:]], dtype=np.float64)

    assert np.all(np.isfinite(forcing_terms))
    assert np.any(np.abs(forcing_terms) > 0.0)


def test_budget_sum_matches_saved_total_rhs(tmp_path) -> None:
    input_file = tmp_path / "sum.input"
    _write_budget_input(input_file, dissipative=True, use_forcing=True, zero_initial=False, dt=0.002, tmax=0.01, t_out_scal=0.002)

    main([str(input_file)])

    _, rows = _read_scalar_rows(tmp_path / "outputs" / "scalar_diagnostics.csv")
    for row in rows:
        total = row["total_energy_rhs_dissipation"] + row["total_energy_rhs_forcing"]
        assert np.isclose(total, row["total_energy_rhs_total"], rtol=1.0e-10, atol=1.0e-12)


def test_budget_matches_measured_energy_change_reasonably(tmp_path) -> None:
    input_file = tmp_path / "consistency.input"
    _write_budget_input(input_file, dissipative=True, use_forcing=False, dt=0.002, tmax=0.02, t_out_scal=0.002, nu_perp=0.02)

    main([str(input_file)])

    _, rows = _read_scalar_rows(tmp_path / "outputs" / "scalar_diagnostics.csv")
    times = np.asarray([row["time"] for row in rows], dtype=np.float64)
    energy = np.asarray([row["total_energy"] for row in rows], dtype=np.float64)
    rhs_total = np.asarray([row["total_energy_rhs_total"] for row in rows], dtype=np.float64)

    measured = np.diff(energy) / np.diff(times)
    assert np.allclose(measured, rhs_total[1:], rtol=0.15, atol=1.0e-6)


def test_total_energy_is_nearly_constant_for_ideal_multifield_case(tmp_path) -> None:
    input_file = tmp_path / "ideal_multifield.input"
    _write_budget_input(
        input_file,
        dissipative=False,
        use_forcing=False,
        zero_initial=False,
        initial_condition_type="random_spectrum",
        dt=0.002,
        tmax=0.02,
        t_out_scal=0.002,
        cs2_over_vA2=0.3,
    )

    main([str(input_file)])

    _, rows = _read_scalar_rows(tmp_path / "outputs" / "scalar_diagnostics.csv")
    energy = np.asarray([row["total_energy"] for row in rows], dtype=np.float64)

    assert np.max(np.abs(energy - energy[0])) < 1.0e-6
