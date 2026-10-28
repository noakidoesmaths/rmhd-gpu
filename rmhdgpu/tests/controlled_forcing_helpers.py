"""Small standard-equation contexts for controlled-forcing regression tests."""
from copy import deepcopy

from rmhdgpu import forcing_control as core
from rmhdgpu.backend import build_backend
from rmhdgpu.config import Config
from rmhdgpu.equations import get_equation_module
from rmhdgpu.grid import build_grid
from rmhdgpu.masks import build_dealias_mask
from rmhdgpu.state import State


def configuration(*, backend="numpy", precision="float64", equation="alfvenic",
                  model="random_increment_phase", planes=None, report=False,
                  scope="perpendicular_shell", both=True, power=False):
    branch = (dict(control="constant_power", power_basis="elsasser", epsilon=.03)
              if power else dict(control="target", target_basis="elsasser",
                  target_quantity="rms", target_value=.4, target_scope=scope,
                  startup_energy_fraction=.25, tau_F=.2))
    shell = dict(k_sigma_min=2., k_sigma_max=3., kz_index=1, interval_max=.01,
                 branches=["plus", "minus"] if both else ["plus"],
                 phase_model=model, report_parallel_planes=report,
                 kz_indices=planes, plus=branch)
    if both:
        shell["minus"] = deepcopy(branch)
    return Config(equation_set=equation, Nx=16, Ny=16, Nz=32,
                  backend=backend, fft_workers=2, real_dtype=precision,
                  complex_dtype="complex64" if precision == "float32" else "complex128",
                  vA=1.2, use_forcing=True, forcing_type="controlled_shell",
                  forcing_seed=37, controlled_shell=shell)


def context(config=None, *, initialize=True):
    config = configuration() if config is None else config
    backend = build_backend(config)
    grid = build_grid(config, backend)
    state = State(grid, backend, field_names=config.field_names)
    control = core.create_control(config, grid, backend,
        get_equation_module(config.equation_set), build_dealias_mask(grid, backend))
    if initialize:
        core.initialize(control, state)
    return state, control
