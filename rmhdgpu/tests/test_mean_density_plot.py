"""Analytic check of the mean-density profile and the snapshot-reading workflow."""

import numpy as np
import pytest

from vis import plot_mean_density as pmd

h5py = pytest.importorskip("h5py")


def write_snapshot(path, x, drho, time=0.0):
    with h5py.File(path, "w") as f:
        f.create_dataset("metadata/x", data=x)
        f.create_dataset("output/time", data=time)
        f.create_dataset("output/drho", data=drho)


def test_profile_is_linear_background_plus_mean_drho():
    n = 8
    x = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    # drho = 0.1*cos(x) (y,z independent) + a y-dependent piece that must average away.
    y = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    drho = 0.1 * np.cos(x)[:, None, None] + 0.05 * np.sin(y)[None, :, None] * np.ones((n, n, n))
    background, mean, drho_mean = pmd.mean_density_profile(x, drho, k_rho0=0.05)
    x_c = np.pi  # middle of the box [0, 2 pi)
    np.testing.assert_allclose(background, 1.0 + 0.05 * (x - x_c))
    np.testing.assert_allclose(drho_mean, 0.1 * np.cos(x), atol=1e-14)
    np.testing.assert_allclose(mean, 1.0 + 0.05 * (x - x_c) + 0.1 * np.cos(x), atol=1e-14)


def test_box_centre_and_length_come_from_a_grid_without_endpoint():
    x = np.linspace(1.0, 5.0, 8, endpoint=False)  # box [1, 5): length 4, centre 3
    assert pmd.box_length(x) == pytest.approx(4.0)
    assert pmd.box_centre(x) == pytest.approx(3.0)


def test_no_fluctuation_gives_exactly_the_background_and_k_zero_is_uniform():
    x = np.linspace(0.0, 2 * np.pi, 8, endpoint=False)
    zero = np.zeros((8, 4, 4))
    background, mean, drho_mean = pmd.mean_density_profile(x, zero, k_rho0=0.3)
    np.testing.assert_array_equal(mean, background)
    np.testing.assert_array_equal(drho_mean, 0.0)
    assert background[x.size // 2] == pytest.approx(1.0)  # rho = rho_c at the box centre
    _, flat, _ = pmd.mean_density_profile(x, zero, k_rho0=0.0)
    np.testing.assert_array_equal(flat, 1.0)


def test_script_reads_k_rho0_from_input_copy_and_writes_png(tmp_path):
    n = 4
    run = tmp_path / "run"
    snaps = run / "fullfields"
    snaps.mkdir(parents=True)
    (run / "input_copy.input").write_text("[physics]\nK_rho0 = 0.2  # gradient\n")
    write_snapshot(snaps / "fullfield_0001.h5", np.linspace(0, 1, n, endpoint=False), np.zeros((n, n, n)))
    assert pmd.read_k_rho0(snaps) == 0.2
    pmd.main([str(snaps)])
    assert (snaps / "mean_density.png").exists()


def test_fluctuation_only_option_and_large_gradient_note(tmp_path, capsys):
    n = 4
    snaps = tmp_path / "fullfields"
    snaps.mkdir()
    drho = 2.0 * np.ones((n, n, n))  # |drho| >= 1 as well
    write_snapshot(snaps / "fullfield_0001.h5", np.linspace(0, 2 * np.pi, n, endpoint=False), drho)
    output = tmp_path / "fluct.png"
    pmd.main([str(snaps), "--k-rho0", "1.0", "--fluctuation-only", "--output", str(output)])
    assert output.exists()
    printed = capsys.readouterr().out
    assert "small-gradient ordering" in printed  # K * Lx / 2 = pi >= 1
    assert "max |drho|" in printed
