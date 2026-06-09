"""Equation-set registry.

The solver keeps equation selection deliberately explicit. Adding a new system
should usually mean adding one module under `rmhdgpu.equations` and registering
it in `_EQUATION_MODULES`.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

from . import alfvenic, low_beta_stratified, s09


_EQUATION_MODULES: dict[str, str] = {
    "alfvenic": "rmhdgpu.equations.alfvenic",
    "s09": "rmhdgpu.equations.s09",
    "low_beta_stratified": "rmhdgpu.equations.low_beta_stratified",
    "inhomogeneous_rmhd_rho": "rmhdgpu.equations.rmhd_by_nokia_rho",
    "inhomogeneous_rmhd_s": "rmhdgpu.equations.rmhd_by_nokia_s"
}


def available_equation_sets() -> list[str]:
    """Return the registered equation-set names."""

    return sorted(_EQUATION_MODULES)


def get_equation_module(name: str) -> ModuleType:
    """Return the equation module registered as `name`."""

    key = str(name)
    try:
        module_path = _EQUATION_MODULES[key]
    except KeyError as exc:
        available = ", ".join(available_equation_sets())
        raise ValueError(f"Unknown equation set {key!r}. Available equation sets: {available}.") from exc
    return import_module(module_path)


FIELD_NAMES = s09.FIELD_NAMES
derived_parameters = s09.derived_parameters
derive_phi_hat = s09.derive_phi_hat
derive_j_hat = s09.derive_j_hat
ideal_rhs = s09.ideal_rhs
linear_matrix = s09.linear_matrix
dissipation_operator = s09.dissipation_operator
build_dissipation_operators = s09.build_dissipation_operators


__all__ = [
    "FIELD_NAMES",
    "alfvenic",
    "available_equation_sets",
    "build_dissipation_operators",
    "derived_parameters",
    "derive_j_hat",
    "derive_phi_hat",
    "dissipation_operator",
    "get_equation_module",
    "ideal_rhs",
    "linear_matrix",
    "low_beta_stratified",
    "s09",
]
