"""Analytic check of the mean-density profile and the snapshot-reading workflow."""

import numpy as np
import pytest

from vis import plot_mean_density as pmd

h5py = pytest.importorskip("h5py")


def test_profile_is_background_times_one_plus_mean_drho():
    n = 8
    x = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    # drho = 0.1*cos(x) (y,z independent) + a y-dependent piece that must average away.
    y = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    drho = 0.1 * np.cos(x)[:, None, None] + 0.05 * np.sin(y)[None, :, None] * np.ones((n, n, n))
    background, mean, drho_mean = pmd.mean_density_profile(x, drho, k_rho0=1.5)
    np.testing.assert_allclose(background, np.exp(1.5 * x))
    np.testing.assert_allclose(drho_mean, 0.1 * np.cos(x), atol=1e-14)
    np.testing.assert_allclose(mean, np.exp(1.5 * x) * (1 + 0.1 * np.cos(x)), atol=1e-14)


def test_script_reads_k_rho0_from_input_copy_and_writes_png(tmp_path):
    n = 4
    run = tmp_path / "run"
    snaps = run / "fullfields"
    snaps.mkdir(parents=True)
    (run / "input_copy.input").write_text("[physics]\nK_rho0 = 2.0  # gradient\n")
    with h5py.File(snaps / "fullfield_0001.h5", "w") as f:
        f.create_dataset("metadata/x", data=np.linspace(0, 1, n, endpoint=False))
        f.create_dataset("output/time", data=0.0)
        f.create_dataset("output/drho", data=np.zeros((n, n, n)))
    assert pmd.read_k_rho0(snaps) == 2.0
    pmd.main([str(snaps)])
    assert (snaps / "mean_density.png").exists()
