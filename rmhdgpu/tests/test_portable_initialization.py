"""Portable initial arrays, including normalization, before forcing startup."""
import numpy as np
import pytest

from rmhdgpu.backend import Backend, build_backend
from rmhdgpu.config import Config
from rmhdgpu.equations import get_equation_module
from rmhdgpu.fft import FFTManager
from rmhdgpu.grid import build_grid
from rmhdgpu.initconds import build_initial_state
from rmhdgpu.masks import build_dealias_mask
from rmhdgpu.operators import lap_perp


def make_state(kind, backend_name="numpy", seed=17, portable=True,
               precision="double", mask_enabled=True):
    config = Config(Nx=12, Ny=10, Nz=8, backend=backend_name,
                    equation_set="inhomogeneous_rmhd_rho",
                    real_dtype=np.float64 if precision == "double" else np.float32,
                    complex_dtype=np.complex128 if precision == "double" else np.complex64)
    backend = build_backend(config)
    grid = build_grid(config, backend)
    mask = build_dealias_mask(grid, backend) if mask_enabled else None
    parameters = {"seed": seed, "init_energy": 0.64}
    if portable:
        parameters["initialization_backend"] = "numpy"
    state = build_initial_state(kind, parameters=parameters, grid=grid,
                               backend=backend, fft=FFTManager(grid, backend),
                               dealias_mask=mask, field_names=config.field_names,
                               params=config)
    return state, config, mask


@pytest.mark.parametrize("kind", ["random_spectrum", "random_spectrum_one_wave"])
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("mask_enabled", [False, True])
def test_numpy_portable_matches_legacy_and_preserves_physics(kind, precision, mask_enabled):
    state, config, mask = make_state(kind, precision=precision, mask_enabled=mask_enabled)
    legacy, _, _ = make_state(kind, portable=False, precision=precision, mask_enabled=mask_enabled)
    changed, _, _ = make_state(kind, seed=18, precision=precision, mask_enabled=mask_enabled)
    for name in state.field_names:
        np.testing.assert_array_equal(state[name], legacy[name])
        assert np.isfinite(state[name]).all()
        assert np.count_nonzero(state[name][:, :, 0]) == 0
        if mask is not None:
            assert np.count_nonzero(state[name][~mask]) == 0
    assert not np.array_equal(state["psi"], changed["psi"])
    energy = get_equation_module(config.equation_set).total_energy(
        state, state.grid, state.backend, config)
    assert float(energy) == pytest.approx(0.64, rel=2e-6)
    if kind == "random_spectrum_one_wave":
        np.testing.assert_allclose(state["omega"], -lap_perp(state["psi"], state.grid), rtol=2e-6, atol=1e-12)
        for name in ("drho", "du_par", "db_par"):
            assert np.count_nonzero(state[name]) == 0


@pytest.mark.parametrize("kind", ["random_spectrum", "random_spectrum_one_wave"])
def test_portable_ignores_evolution_rng_and_fft(kind, monkeypatch):
    expected, _, _ = make_state(kind)
    def forbidden(*args, **kwargs):
        raise AssertionError("Evolution FFT/RNG used during portable initialization")
    original_rng = Backend.random_generator
    def guarded_rng(self, seed=None):
        if self.backend_name != "numpy":
            forbidden()
        return original_rng(self, seed)
    monkeypatch.setattr(Backend, "random_generator", guarded_rng)
    # A deliberately unusable evolution FFT must not enter initialization.
    backend = Backend(np, "test_evolution", False)
    config = Config(Nx=12, Ny=10, Nz=8, equation_set="inhomogeneous_rmhd_rho")
    grid = build_grid(config, backend)
    actual = build_initial_state(kind, parameters={"seed": 17, "init_energy": 0.64,
                                 "initialization_backend": "numpy"},
                                 grid=grid, backend=backend, fft=None,
                                 dealias_mask=build_dealias_mask(grid, backend),
                                 field_names=config.field_names, params=config)
    assert actual.backend is backend
    assert actual.grid is grid
    for name in expected.field_names:
        np.testing.assert_array_equal(actual[name], expected[name])


@pytest.mark.parametrize("kind", ["random_spectrum", "random_spectrum_one_wave"])
@pytest.mark.parametrize("backend_name", ["scipy_cpu", "cupy"])
def test_portable_coefficients_equal_across_backends(kind, backend_name):
    if backend_name == "scipy_cpu":
        pytest.importorskip("scipy")
    else:
        cp = pytest.importorskip("cupy")
        try:
            cp.zeros(1)
        except Exception as exc:
            pytest.skip(f"CUDA unavailable: {exc}")
    expected, _, _ = make_state(kind)
    actual, _, _ = make_state(kind, backend_name)
    for name in expected.field_names:
        np.testing.assert_array_equal(actual.backend.to_numpy(actual[name]), expected[name])


@pytest.mark.parametrize("kind", ["random_spectrum", "random_spectrum_one_wave"])
def test_input_option_validated_and_resolved(tmp_path, kind):
    from rmhdgpu.runfile import resolve_run_settings, write_resolved_config
    path = tmp_path / "portable.input"
    text = ('[initial_condition]\ntype = "' + kind + '"\n'
            '[initial_condition.parameters]\nseed = 17\ninitialization_backend = "numpy"\n')
    path.write_text(text, encoding="utf-8")
    settings = resolve_run_settings(runfile_path=path)
    assert settings.initial_condition.parameters["initialization_backend"] == "numpy"
    resolved = tmp_path / "resolved_config.toml"
    write_resolved_config(settings, resolved)
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib
    restored = tomllib.loads(resolved.read_text(encoding="utf-8"))
    assert restored["initial_condition"]["parameters"]["initialization_backend"] == "numpy"
    assert restored["initial_condition"]["parameters"]["seed"] == 17
    path.write_text(text.replace('"numpy"', '"typo"'), encoding="utf-8")
    with pytest.raises(ValueError, match="initialization_backend"):
        resolve_run_settings(runfile_path=path)

