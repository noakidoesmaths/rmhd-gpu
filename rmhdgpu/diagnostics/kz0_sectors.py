"""Energy in the k_par = 0 plane and its k_y = 0 line, written every scalar output.

Here k_par means k_z, the wavenumber along the background field. Each field's
energy is split into three disjoint Fourier sectors:

- `kzn0`      : k_z != 0. Propagating (Alfvenic / slow / gravity) waves.
- `kz0_kyn0`  : k_z = 0, k_y != 0. Non-propagating 2D structures that still
                feel the gradient drive (`K_rho0 dy(phi)`) and buoyancy
                (`-g dy(drho)`), because both involve d_y.
- `kz0_ky0`   : k_z = 0, k_y = 0. The (y, z)-averaged profiles <f>(x). Neither
                the background drive nor buoyancy acts on them, they do not
                propagate, and only the divergence of the turbulent flux,
                d_x <u_x f>, changes them.

The three sectors add up to each field's energy, and the field totals add up
to `total_energy` (z+ + z- equals u_perp + b_perp mode by mode, so the
Elsasser pair stands in for the Alfvenic energy).

Energies are volume averages taken from the equation module's own modal
densities (`_energy_modal_densities`), so the weights (entropy energy of the
compressive field, `dbpar_energy_weight`) match the spectra and `total_energy`.
Note that z± carry k_perp^2 |phi ∓ psi|^2, so their energy at k_perp = 0
(kx = ky = 0) is identically zero; their `kz0_ky0` part comes from kx != 0.

`vis/plot_kpar0_energy.py` plots these columns against time, and
`vis/plot_kpar0_maps.py` shows the same drho energy mode by mode in (kx, ky).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from rmhdgpu.fourier_diagnostics import modal_density_average


SECTOR_DESCRIPTIONS = {
    "kz0_ky0": "k_z = 0, k_y = 0 sector (the (y, z)-averaged profile <f>(x))",
    "kz0_kyn0": "k_z = 0, k_y != 0 sector (non-propagating 2D structures)",
}


def kz0_sector_column_info(fields: Iterable[str]) -> dict[str, str]:
    """Return {column: description} for the columns `kz0_sector_energies` writes."""

    info: dict[str, str] = {}
    for field in fields:
        for sector, description in SECTOR_DESCRIPTIONS.items():
            info[f"E_{sector}_{field}"] = (
                f"Energy of {field} in the {description}; volume average, same weights as total_energy."
            )
        info[f"E_kzn0_{field}"] = f"Energy of {field} at k_z != 0 (waves); the three sectors add up to its energy."
    info["E_kz0_ky0"] = (
        "Total energy in the k_z = 0, k_y = 0 sector (sum over the listed fields): mean profiles <f>(x)."
    )
    info["E_kz0_kyn0"] = "Total energy in the k_z = 0, k_y != 0 sector (sum over the listed fields)."
    info["E_kzn0"] = (
        "Total energy at k_z != 0 (waves); E_kz0_ky0 + E_kz0_kyn0 + E_kzn0 = total_energy."
    )
    return info


def _sector_masks(grid: Any, backend: Any) -> dict[str, Any]:
    """Boolean masks on the rfftn grid, built on the device (no host transfer)."""

    xp = backend.xp
    shape = grid.fourier_shape
    # rfftn keeps k_z = 0 at index 0 of the last axis; ky has shape (1, Ny, 1).
    kz0 = xp.broadcast_to(grid.kz == 0, shape)
    ky0 = xp.broadcast_to(grid.ky == 0, shape)
    return {"kz0_ky0": kz0 & ky0, "kz0_kyn0": kz0 & ~ky0}


def kz0_sector_energies(
    densities: Mapping[str, Any],
    fields: Iterable[str],
    grid: Any,
    backend: Any,
) -> dict[str, float]:
    """Return the k_z = 0 / k_y = 0 sector energies of `fields`.

    Columns: `E_kz0_ky0_<field>`, `E_kz0_kyn0_<field>`, `E_kzn0_<field>` for
    each field, then the totals `E_kz0_ky0`, `E_kz0_kyn0`, `E_kzn0`.

    `densities` maps field name to its modal energy density, as returned by an
    equation module's `_energy_modal_densities`. `fields` must partition the
    total energy (e.g. z_plus, z_minus, du_par, db_par and the entropy field),
    otherwise the totals do not add up to `total_energy`.
    """

    fields = tuple(fields)
    masks = _sector_masks(grid, backend)
    out: dict[str, float] = {}
    totals = {sector: 0.0 for sector in masks}
    total_all = 0.0
    for field in fields:
        density = densities[field]
        field_total = modal_density_average(density, grid, backend)
        total_all += field_total
        kz0_part = 0.0
        for sector, mask in masks.items():
            value = modal_density_average(density, grid, backend, mask=mask)
            out[f"E_{sector}_{field}"] = value
            totals[sector] += value
            kz0_part += value
        out[f"E_kzn0_{field}"] = field_total - kz0_part
    out["E_kz0_ky0"] = totals["kz0_ky0"]
    out["E_kz0_kyn0"] = totals["kz0_kyn0"]
    out["E_kzn0"] = total_all - totals["kz0_ky0"] - totals["kz0_kyn0"]
    return out
