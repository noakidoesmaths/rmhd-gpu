"""Small analytic checks for the DCF and CCR heating plot and its saved-output workflow."""

import csv

import numpy as np
import pytest

from vis import plot_zplus_heating as heating
from vis.run_quantities import load_run


ALL_ROWS = np.array([True, True, True])
AUTO_DISSIPATION = '[dissipation]\nmode = "auto"\nn_perp = 1\nnu_par = 0.0\n'
PLUS_FORCING = (
    '[forcing]\nuse_forcing = true\ntype = "controlled_shell"\n'
    '[forcing.controlled_shell]\nk_sigma_min = 1.0\nk_sigma_max = 2.0\nbranches = ["plus"]\n'
)
# Isotropic z+_x = z+/sqrt(2): the DCF estimate is then the paper's -g F_rho z+ l_perp / 4,
# and the CCR estimate |g F_rho| l_perp z+_x / 4 is ISO times it.
ISO = 1 / np.sqrt(2)
# The fixture's DCF estimate, -g F_rho l_perp z+ / 4 = 12 * [1, 4/3, 3/2] / 4.
DCF_ISOTROPIC = np.array([3.0, 4.0, 4.5])


def physics(K_rho0=3.25):
    """vA = 2, chi = 3, g = 4: F_rho = g/(vA^2 (1 + chi)) - K_rho0 = 0.25 - K_rho0, so -g F_rho = 12 by default."""
    return f"[physics]\nvA = 2.0\ncs2_over_vA2 = 3.0\ng = 4.0\nK_p0 = 0.5\nK_rho0 = {K_rho0}\n"


@pytest.fixture
def columns():
    # W+ = z+^2/4: z+ = [2, 4, 6], z- = [1, 2, 3], l_perp = [1/2, 1/3, 1/4].
    return {
        name: np.asarray(values, dtype=float)
        for name, values in {
            "time": [0, 1, 2],
            "w_plus": [1, 4, 9],
            "w_minus": [0.25, 1, 2.25],
            "k_perp_plus": [2, 3, 4],
            "k_prl_plus": [1, 2, 3],
            "w_plus_align": [ISO] * 3,
            "q_dcf": [0, 8, 18],
            "w_plus_rhs_dissipation": [-1, -2, -3],
        }.items()
    }


def write_run(run_dir, columns, *, K_rho0=3.25):
    """A run directory with input_copy.input and scalar_diagnostics.csv."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "input_copy.input").write_text(physics(K_rho0), encoding="utf-8")
    with (run_dir / "scalar_diagnostics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(zip(*columns.values()))
    return run_dir


def run_for(tmp_path, columns, *, K_rho0=3.25):
    return load_run(write_run(tmp_path / "run", columns, K_rho0=K_rho0), fields=())


def write_check_files(run_dir, *, config, spectra):
    """resolved_config.toml and a z+ spectra.csv; `spectra` is [(time, kperp, E+), ...]."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_config.toml").write_text(config, encoding="utf-8")
    with (run_dir / "spectra.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time", "step", "quantity", "kperp", "value"])
        for time, kperp, value in spectra:
            writer.writerow([time, 0, "z_plus", kperp, value])
            writer.writerow([time, 0, "z_minus", kperp, 100.0])  # Must be ignored.


def test_estimates_window_means_and_loss_shares(tmp_path, columns):
    run = run_for(tmp_path, columns)
    np.testing.assert_allclose(run.q_dcf_predicted, DCF_ISOTROPIC)
    np.testing.assert_allclose(run.q_ccr_predicted, DCF_ISOTROPIC * ISO)
    np.testing.assert_allclose(run.q_ccr, [1, 2, 3])  # -w_plus_rhs_dissipation
    means = heating.window_means(run, ALL_ROWS)
    assert means["DCF"] == pytest.approx((26 / 3, 11.5 / 3))
    assert means["CCR"] == pytest.approx((2, 11.5 / 3 * ISO))
    assert heating.loss_shares(means) == pytest.approx({"DCF": (26 / 3) / (32 / 3), "CCR": 2 / (32 / 3)})
    # Only the window rows enter the means.
    later = heating.window_means(run, columns["time"] >= 1)
    assert later["DCF"] == pytest.approx((13, 4.25))


def test_measured_z_plus_x_replaces_isotropy(tmp_path, columns):
    # z+_x = 0.4 z+: DCF scales by 2 * 0.4^2 = 0.32 and CCR by 0.4 from the isotropic DCF value.
    columns["w_plus_align"] = np.full(3, 0.4)
    run = run_for(tmp_path, columns)
    np.testing.assert_allclose(run.q_dcf_predicted, DCF_ISOTROPIC * 0.32)
    np.testing.assert_allclose(run.q_ccr_predicted, DCF_ISOTROPIC * 0.4)


@pytest.mark.parametrize("K_rho0, minus_g_f_rho", [(3.25, 12), (-2.75, -12), (0.25, 0)])
def test_stratification_sign(tmp_path, columns, K_rho0, minus_g_f_rho):
    # Stable (-g F_rho > 0) drains z+; unstable feeds it. CCR heats for either sign.
    run = run_for(tmp_path, columns, K_rho0=K_rho0)
    np.testing.assert_allclose(run.q_dcf_predicted, DCF_ISOTROPIC * minus_g_f_rho / 12, atol=1e-12)
    np.testing.assert_allclose(run.q_ccr_predicted, np.abs(run.q_dcf_predicted) * ISO, atol=1e-12)


def test_zero_estimate_gives_no_ratio(tmp_path, columns):
    means = heating.window_means(run_for(tmp_path, columns, K_rho0=0.25), ALL_ROWS)  # F_rho = 0
    assert means["DCF"][1] == 0.0
    assert np.isnan(heating.ratio(*means["DCF"]))


def test_hyperviscosity_order_needs_one_k_perp_only_damping_of_psi_and_omega():
    assert heating.hyperviscosity_order({"dissipation": {"mode": "auto", "n_perp": 3, "nu_par": 0.0}}) == 3
    assert heating.hyperviscosity_order({"dissipation": {"mode": "auto", "n_perp": 3, "nu_par": 1e-3}}) is None
    field = {"nu_perp": 1e-5, "nu_par": 0.0, "n_perp": 2, "n_par": 1}
    assert heating.hyperviscosity_order({"dissipation": {"psi": field, "omega": dict(field)}}) == 2
    assert heating.hyperviscosity_order(
        {"dissipation": {"psi": field, "omega": {**field, "nu_perp": 2e-5}}}) is None
    assert heating.hyperviscosity_order({}) is None


def test_forced_shell_share_weights_the_window_mean_z_plus_spectrum(tmp_path):
    # n_perp = 1: shells lose k^2 E+, so [0, 2.25, 6.25]. The t = 0 spectrum is outside the window.
    spectra = [(0.0, k, 50.0) for k in (0.5, 1.5, 2.5)]
    spectra += [(t, k, e) for t in (6.0, 8.0) for k, e in ((0.5, 0.0), (1.5, 1.0), (2.5, 1.0))]
    write_check_files(tmp_path, config=AUTO_DISSIPATION + PLUS_FORCING, spectra=spectra)
    config = heating.read_resolved_config(tmp_path)
    share, reason = heating.forced_shell_dissipation_share(tmp_path, config, 4.0, 10.0)
    assert reason is None
    assert share == pytest.approx(2.25 / 8.5)  # Shells centred below k_sigma_max = 2.
    share, reason = heating.forced_shell_dissipation_share(tmp_path, config, 20.0, 30.0)
    assert share is None and "spectra.csv" in reason
    share, reason = heating.forced_shell_dissipation_share(tmp_path, {}, 4.0, 10.0)
    assert share is None and "forcing" in reason


def test_ccr_reading_warns_for_forced_z_minus_and_forced_shell_damping(tmp_path, capsys):
    spectra = [(5.0, k, 1.0) for k in (0.5, 1.5, 2.5)]
    write_check_files(tmp_path, config=AUTO_DISSIPATION + PLUS_FORCING.replace('["plus"]', '["plus", "minus"]'),
                      spectra=spectra)
    heating.check_ccr_reading(tmp_path, heating.read_resolved_config(tmp_path), 4.0, 6.0)
    printed = capsys.readouterr().out
    assert "z- is forced too" in printed
    assert "28.6%" in printed and "overstates" in printed  # (0.25 + 2.25) / 8.75 is above 10%.


def test_panels_draw_measured_and_estimate_and_the_title_gives_the_loss_shares(tmp_path, monkeypatch, columns):
    figures = []

    def capture(fig, *, output_path, show, plt):
        figures.append({
            "scales": [ax.get_yscale() for ax in fig.axes],
            "curves": [[line.get_ydata() for line in ax.lines[:2]] for ax in fig.axes],
            "title": fig._suptitle.get_text(),
        })
        plt.close(fig)

    monkeypatch.setattr(heating, "finalize_figure", capture)
    run = run_for(tmp_path, columns)
    heating.plot_heating(run, ALL_ROWS, heating.window_means(run, ALL_ROWS), tmp_path / "a.png",
                         label="Krho0_2.0/run")
    figure = figures[0]
    assert figure["scales"] == ["linear", "linear"]
    np.testing.assert_allclose(figure["curves"][0][0], [0, 8, 18])
    np.testing.assert_allclose(figure["curves"][0][1], DCF_ISOTROPIC)
    np.testing.assert_allclose(figure["curves"][1][1], DCF_ISOTROPIC * ISO)
    assert figure["title"].startswith("Krho0_2.0/run | window $t$ = 0 to 2")
    assert "DCF 81%, CCR 19%" in figure["title"]  # 26/3 and 2 of the 32/3 total.


def test_cli_reads_old_column_names_from_a_run_directory_and_writes_a_png(tmp_path, columns, capsys):
    # CSVs written before 2026-10-08 use "t" and the w_plus_k* names; load_run renames them.
    old_names = {"time": "t", "k_perp_plus": "w_plus_kperp", "k_prl_plus": "w_plus_kprl"}
    run_dir = write_run(tmp_path / "Krho0_2.0" / "run",
                        {old_names.get(name, name): values for name, values in columns.items()})
    output = heating.main([str(run_dir), "--tmin", "1"])
    assert output == run_dir / "zplus_heating.png"
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    printed = capsys.readouterr().out
    assert "Krho0_2.0/run: window t = 1 to 2" in printed
    assert "share of the measured z+ loss" in printed
    # W+ z- / l_perp with the measured z- = [.., 24, 108] over the window t >= 1.
    assert "W+ z-_rms / l_perp: 66" in printed
    assert "Forced-shell check skipped" in printed  # No resolved_config.toml here.
    explicit = tmp_path / "plots" / "heating.png"
    csv_path = run_dir / "scalar_diagnostics.csv"
    assert heating.main([str(csv_path), "--tail-fraction", "1", "--output", str(explicit)]) == explicit
    assert explicit.is_file()


def test_cli_needs_the_input_file_and_the_alignment_column(tmp_path, columns):
    del columns["w_plus_align"]
    run_dir = write_run(tmp_path / "run", columns)
    with pytest.raises(SystemExit):
        heating.main([str(run_dir)])  # No w_plus_align, so no estimates.
    (run_dir / "input_copy.input").unlink()
    with pytest.raises(SystemExit):
        heating.main([str(run_dir)])
