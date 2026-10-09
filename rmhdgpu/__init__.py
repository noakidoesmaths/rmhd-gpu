"""Public package exports for rmhdgpu."""

from rmhdgpu.backend import Backend, build_backend
from rmhdgpu.config import Config
from rmhdgpu.errors import NonFiniteStateError
from rmhdgpu.fft import FFTManager
from rmhdgpu.forcing import (
    apply_forcing_kick,
    forcing_shell_mask,
    generate_forcing_kick,
    mode_number_magnitude,
    shaped_random_real_field,
)
from rmhdgpu.grid import build_grid
from rmhdgpu.state import State
from rmhdgpu.steppers import compute_cfl_timestep, evolve_until, if_ssprk3_step, ssprk3_step
from rmhdgpu.utils import check_state_finite
from rmhdgpu.workspace import Workspace

__all__ = [
    "Config",
    "Backend",
    "FFTManager",
    "NonFiniteStateError",
    "State",
    "Workspace",
    "apply_forcing_kick",
    "build_backend",
    "build_grid",
    "check_state_finite",
    "compute_cfl_timestep",
    "evolve_until",
    "forcing_shell_mask",
    "generate_forcing_kick",
    "if_ssprk3_step",
    "mode_number_magnitude",
    "shaped_random_real_field",
    "ssprk3_step",
]
