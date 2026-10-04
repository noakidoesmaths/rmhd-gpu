from __future__ import annotations

import pytest

from rmhdgpu.run import build_parser
from rmhdgpu.runfile import cli_overrides_from_args, load_run_file, resolve_run_settings


def test_minimal_runfile_parses(tmp_path) -> None:
    input_file = tmp_path / "minimal.input"
    input_file.write_text('title = "Minimal case"\n', encoding="utf-8")

    settings = resolve_run_settings(runfile_path=input_file)

    assert settings.title == "Minimal case"
    assert settings.config.Nx == 16
    assert settings.config.backend == "numpy"
    assert settings.config.equation_mode == "nonlinear"
    assert settings.output_dir == (tmp_path / "outputs").resolve()
    assert settings.initial_condition.type == "alfven_mode"


def test_nested_sections_parse_correctly(tmp_path) -> None:
    input_file = tmp_path / "nested.input"
    input_file.write_text(
        """
title = "Nested case"
output_dir = "case_outputs"

[grid]
Nx = 32
Ny = 16
Nz = 8

[time]
tmax = 2.5
dt_init = 0.01
cfl_number = 0.2

[backend]
backend = "scipy_cpu"
fft_workers = 4

[forcing]
use_forcing = true
forcing_seed = 7

[forcing.force_amplitudes]
psi = 0.05

[dissipation.psi]
nu_perp = 0.005
nu_par = 0.0
n_perp = 2
n_par = 1
""".strip()
        + "\n",
        encoding="utf-8",
    )

    settings = resolve_run_settings(runfile_path=input_file)

    assert settings.config.Nx == 32
    assert settings.config.Ny == 16
    assert settings.config.Nz == 8
    assert settings.config.tmax == 2.5
    assert settings.config.backend == "scipy_cpu"
    assert settings.config.fft_workers == 4
    assert settings.config.use_forcing is True
    assert settings.config.forcing_seed == 7
    assert settings.config.force_amplitudes["psi"] == 0.05
    assert settings.config.force_amplitudes["omega"] == 0.0
    assert settings.config.dissipation["psi"]["nu_perp"] == 0.005
    assert settings.config.dissipation["omega"]["nu_perp"] == 0.0


def test_invalid_runfile_gives_helpful_error(tmp_path) -> None:
    input_file = tmp_path / "bad.input"
    input_file.write_text("[grid]\nNx = [1, 2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Could not parse TOML input file"):
        load_run_file(input_file)


def test_cli_overrides_runfile(tmp_path) -> None:
    input_file = tmp_path / "override.input"
    input_file.write_text(
        """
[grid]
Nx = 16
Ny = 16
Nz = 16

[time]
tmax = 1.0

[backend]
backend = "numpy"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    args = build_parser().parse_args([str(input_file), "--backend", "cupy", "--tmax", "10.0"])
    settings = resolve_run_settings(
        runfile_path=args.input_file,
        cli_overrides=cli_overrides_from_args(args),
    )

    assert settings.config.backend == "cupy"
    assert settings.config.tmax == 10.0


def test_runfile_without_cli_matches_expected_config(tmp_path) -> None:
    input_file = tmp_path / "forcing.input"
    input_file.write_text(
        """
[grid]
Nx = 8
Ny = 8
Nz = 8

[forcing]
use_forcing = true
forcing_seed = 22

[forcing.force_amplitudes]
psi = 0.02
omega = 0.03

[initial_condition]
type = "zero"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    settings = resolve_run_settings(runfile_path=input_file)

    assert settings.config.use_forcing is True
    assert settings.config.forcing_seed == 22
    assert settings.config.force_amplitudes["psi"] == 0.02
    assert settings.config.force_amplitudes["omega"] == 0.03
    assert settings.initial_condition.type == "zero"


def test_mode_branch_names_the_elsasser_wave(tmp_path) -> None:
    input_file = tmp_path / "branch.input"
    input_file.write_text('title = "Branch case"\n', encoding="utf-8")

    # The default alfven_mode launches z+ = u_perp - b_perp.
    settings = resolve_run_settings(runfile_path=input_file)
    assert settings.initial_condition.parameters["branch"] == "z_plus"

    args = build_parser().parse_args([str(input_file), "--mode-branch", "z_minus"])
    settings = resolve_run_settings(
        runfile_path=args.input_file,
        cli_overrides=cli_overrides_from_args(args),
    )
    assert settings.initial_condition.parameters["branch"] == "z_minus"

    # The old names meant the opposite waves, so they fail with a pointer to the new name.
    args = build_parser().parse_args([str(input_file), "--mode-branch", "plus"])
    with pytest.raises(ValueError, match="branch = 'z_minus'"):
        resolve_run_settings(
            runfile_path=args.input_file,
            cli_overrides=cli_overrides_from_args(args),
        )


def test_initial_condition_parameter_table_parses_and_overrides_defaults(tmp_path) -> None:
    input_file = tmp_path / "initcond_parameters.input"
    input_file.write_text(
        """
[grid]
Nx = 8
Ny = 8
Nz = 8

[initial_condition]
        type = "random_spectrum"
        seed = 9

[initial_condition.parameters]
n_max = 2.0
init_energy = 0.125
""".strip()
        + "\n",
        encoding="utf-8",
    )

    settings = resolve_run_settings(runfile_path=input_file)

    assert settings.initial_condition.type == "random_spectrum"
    assert settings.initial_condition.parameters["seed"] == 9
    assert settings.initial_condition.parameters["n_max"] == 2.0
    assert settings.initial_condition.parameters["init_energy"] == 0.125
    assert settings.initial_condition.parameters["n_min"] == 1.0


def test_unknown_initial_condition_in_runfile_gives_helpful_error(tmp_path) -> None:
    input_file = tmp_path / "bad_initcond.input"
    input_file.write_text(
        """
[grid]
Nx = 8
Ny = 8
Nz = 8

[initial_condition]
type = "not_a_real_initcond"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as excinfo:
        resolve_run_settings(runfile_path=input_file)

    message = str(excinfo.value)
    assert "Unknown initial condition type 'not_a_real_initcond'" in message
    for name in ("zero", "alfven_mode", "aw_packet", "random_spectrum"):
        assert name in message


def test_input_file_output_section_parses(tmp_path) -> None:
    input_file = tmp_path / "outputs.input"
    input_file.write_text(
        """
[grid]
Nx = 8
Ny = 8
Nz = 8

[output]
t_out_scal = 0.25
t_out_spec = 0.5
t_out_full = 0.0
""".strip()
        + "\n",
        encoding="utf-8",
    )

    settings = resolve_run_settings(runfile_path=input_file)

    assert settings.config.t_out_scal == 0.25
    assert settings.config.t_out_spec == 0.5
    assert settings.config.t_out_full == 0.0
    assert settings.resolved_document["output"]["t_out_scal"] == 0.25


def test_equation_mode_linear_parses_and_resolves(tmp_path) -> None:
    input_file = tmp_path / "linear.input"
    input_file.write_text(
        """
[equations]
type = "s09"
mode = "linear"

[grid]
Nx = 8
Ny = 8
Nz = 8
""".strip()
        + "\n",
        encoding="utf-8",
    )

    settings = resolve_run_settings(runfile_path=input_file)

    assert settings.config.equation_set == "s09"
    assert settings.config.equation_mode == "linear"
    assert settings.resolved_document["equations"]["mode"] == "linear"


def test_invalid_equation_mode_gives_helpful_error(tmp_path) -> None:
    input_file = tmp_path / "bad_mode.input"
    input_file.write_text(
        """
[equations]
mode = "almost_linear"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="equation_mode must be 'nonlinear' or 'linear'"):
        resolve_run_settings(runfile_path=input_file)
