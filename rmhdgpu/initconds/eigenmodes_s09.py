"""Exact linear S09 mode constructors used mainly for verification.

These helpers build exact single-mode states for the homogeneous S09 system.
They exist primarily to support solver verification tests and the lightweight
registered `alfven_mode` initial condition. They are not the main user-facing
initial-condition registry; that lives in `rmhdgpu.initconds.builtin`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from rmhdgpu.equations import get_equation_module
from rmhdgpu.equations.s09 import FIELD_NAMES, derived_parameters
from rmhdgpu.operators import lap_perp
from rmhdgpu.state import State


def _stored_mode_array(
    grid: Any,
    backend: Any,
    k_indices: Sequence[int],
    amplitude: complex | float,
) -> Any:
    if len(k_indices) != 3:
        raise ValueError(f"k_indices must have length 3; got {k_indices!r}.")
    ix, iy, iz = (int(k_indices[0]), int(k_indices[1]), int(k_indices[2]))
    if not (0 <= ix < grid.Nx and 0 <= iy < grid.Ny and 0 <= iz < (grid.Nz // 2 + 1)):
        raise ValueError(
            f"k_indices {tuple(k_indices)!r} are outside the stored Fourier shape {grid.fourier_shape}."
        )

    mode = backend.zeros(grid.fourier_shape, dtype=grid.complex_dtype)
    mode[ix, iy, iz] = amplitude
    return mode


def _branch_sign(branch: str) -> float:
    """Sign of the slow-mode eigenvalue; the Alfven mode names its branches by Elsasser wave."""
    if branch == "plus":
        return 1.0
    if branch == "minus":
        return -1.0
    raise ValueError(f"branch must be 'plus' or 'minus'; got {branch!r}.")


# phi = ratio * psi for each Alfven branch. z^± = zhat x grad_perp(phi ∓ psi), so
# phi = -psi leaves only z+ and phi = +psi leaves only z-.
ALFVEN_BRANCH_PHI_OVER_PSI = {"z_plus": -1.0, "z_minus": 1.0}
_RENAMED_ALFVEN_BRANCHES = {"plus": "z_minus", "minus": "z_plus"}


def alfven_branch_phi_over_psi(branch: str) -> float:
    """Return phi/psi for an Alfven-mode branch, `"z_plus"` (-1) or `"z_minus"` (+1).

    The branches used to be called `"plus"` (phi = +psi, which is z-) and
    `"minus"` (phi = -psi, which is z+). Those names now raise an error that
    names the branch giving the same wave, rather than silently flipping it.
    """

    if branch in _RENAMED_ALFVEN_BRANCHES:
        raise ValueError(
            f"alfven_mode branch {branch!r} was renamed after the Elsasser wave it launches; "
            f"use branch = {_RENAMED_ALFVEN_BRANCHES[branch]!r} for the same wave."
        )
    if branch not in ALFVEN_BRANCH_PHI_OVER_PSI:
        raise ValueError(f"alfven_mode branch must be 'z_plus' or 'z_minus'; got {branch!r}.")
    return ALFVEN_BRANCH_PHI_OVER_PSI[branch]


def alfven_mode_state(
    grid: Any,
    backend: Any,
    field_names: Sequence[str] | None,
    k_indices: Sequence[int],
    amplitude: complex | float = 1.0,
    branch: str = "z_plus",
    params: Any | None = None,
) -> State:
    """Return an exact Alfvén eigenmode in Fourier space.

    Branches are named after the Elsasser wave they launch, with
    `z^± = delta u_perp ∓ delta B_perp/sqrt(4 pi rho_0) = zhat x grad_perp(phi ∓ psi)`:

    - `branch="z_plus"` uses `phi = -psi`, a pure z+ wave travelling along +B_0,
      so the mode evolves as `exp(-i vA k_z t)`
    - `branch="z_minus"` uses `phi = +psi`, a pure z- wave travelling along -B_0,
      so the mode evolves as `exp(+i vA k_z t)`

    The amplitude parameter rescales the mode so `sqrt(total_energy) = amplitude`
    using the active equation-set energy diagnostic. For an exact Alfvén wave
    this is also the perpendicular RMS fluctuation amplitude. This helper is
    mainly intended for controlled verification setups.
    """

    if params is None:
        raise ValueError("alfven_mode_state requires params so the energy normalization is defined.")

    names = list(FIELD_NAMES if field_names is None else field_names)
    state = State(grid, backend, field_names=names)

    ix, iy, _ = (int(k_indices[0]), int(k_indices[1]), int(k_indices[2]))
    kperp2 = backend.scalar_to_float(grid.kperp2[ix, iy, 0])
    if kperp2 == 0.0:
        raise ValueError("Alfvén eigenmodes require k_perp != 0.")

    phi_over_psi = alfven_branch_phi_over_psi(branch)
    psi_hat = _stored_mode_array(grid, backend, k_indices, 1.0)
    phi_hat = phi_over_psi * psi_hat
    omega_hat = lap_perp(phi_hat, grid)

    state["psi"][...] = psi_hat
    if "omega" in state.field_names:
        state["omega"][...] = omega_hat

    equation_module = get_equation_module(getattr(params, "equation_set", "s09"))
    energy = equation_module.total_energy(state, grid, backend, params)
    if energy <= 0.0:
        raise ValueError("Alfvén eigenmode normalization produced nonpositive energy.")
    scale_factor = float(np.abs(amplitude)) / np.sqrt(energy)
    for field_name in state.field_names:
        state[field_name][...] *= scale_factor
    return state


def slow_mode_state(
    grid: Any,
    backend: Any,
    field_names: Sequence[str] | None,
    k_indices: Sequence[int],
    amplitude: complex | float = 1.0,
    branch: str = "plus",
    params: Any | None = None,
) -> State:
    """Return an exact linear S09 slow-mode eigenvector in Fourier space.

    The amplitude parameter sets the stored Fourier coefficient of `upar_hat`.

    With `c_slow = vA * sqrt(alpha)`, the branches are defined by

    - `branch="plus"`: eigenvalue `+i c_slow k_z`, relation
      `dbpar = +(sqrt(alpha) / vA) * upar`
    - `branch="minus"`: eigenvalue `-i c_slow k_z`, relation
      `dbpar = -(sqrt(alpha) / vA) * upar`
    This helper is mainly intended for controlled verification setups.
    """

    if params is None:
        raise ValueError("slow_mode_state requires params so alpha and vA are defined.")

    names = list(FIELD_NAMES if field_names is None else field_names)
    state = State(grid, backend, field_names=names)

    sign = _branch_sign(branch)
    s09_params = derived_parameters(params)
    if s09_params.vA == 0.0:
        raise ValueError("slow_mode_state requires nonzero vA.")
    if s09_params.alpha < 0.0:
        raise ValueError(f"alpha must be nonnegative; got {s09_params.alpha}.")

    upar_hat = _stored_mode_array(grid, backend, k_indices, amplitude)
    dbpar_hat = sign * (np.sqrt(s09_params.alpha) / s09_params.vA) * upar_hat

    state["upar"][...] = upar_hat
    state["dbpar"][...] = dbpar_hat
    return state


def entropy_mode_state(
    grid: Any,
    backend: Any,
    field_names: Sequence[str] | None,
    k_indices: Sequence[int],
    amplitude: complex | float = 1.0,
) -> State:
    """Return a pure stationary S09 entropy mode in Fourier space.

    This helper is mainly intended for controlled verification setups.
    """

    names = list(FIELD_NAMES if field_names is None else field_names)
    state = State(grid, backend, field_names=names)
    state["s"][...] = _stored_mode_array(grid, backend, k_indices, amplitude)
    return state
