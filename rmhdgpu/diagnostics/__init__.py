"""Diagnostics namespace."""

from rmhdgpu.diagnostics.alfvenic import (
    alfvenic_phi_hat,
    alfvenic_cross_helicity,
    alfvenic_cross_helicity_rhs_budget,
    alfvenic_energy,
    alfvenic_energy_rhs_budget,
    elsasser_energies,
)
from rmhdgpu.diagnostics.scalar import (
    GENERIC_FIELD_SCALAR_DIAGNOSTIC_INFO,
    STANDARD_ENERGY_SCALAR_DIAGNOSTIC_INFO,
    compute_energy_diagnostics,
    compute_field_scalar_diagnostics,
    compute_scalar_diagnostics,
)
from rmhdgpu.diagnostics.spectra import (
    PERPENDICULAR_SPECTRUM_KEYS,
    elsasser_perpendicular_spectra,
    perpendicular_energy_spectrum_from_state,
)

__all__ = [
    "PERPENDICULAR_SPECTRUM_KEYS",
    "alfvenic_phi_hat",
    "alfvenic_cross_helicity",
    "alfvenic_cross_helicity_rhs_budget",
    "alfvenic_energy",
    "alfvenic_energy_rhs_budget",
    "elsasser_energies",
    "elsasser_perpendicular_spectra",
    "GENERIC_FIELD_SCALAR_DIAGNOSTIC_INFO",
    "STANDARD_ENERGY_SCALAR_DIAGNOSTIC_INFO",
    "compute_energy_diagnostics",
    "compute_field_scalar_diagnostics",
    "compute_scalar_diagnostics",
    "perpendicular_energy_spectrum_from_state",
]
