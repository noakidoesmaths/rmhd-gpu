"""Adaptive common hyperdissipation for the Fourier-space solver.

The controller works directly from the Fourier-space state. It does not use
FFTs, and on GPU backends it only transfers scalar reduction results back to
the host.

The key measurement is the shell energy near a chosen dissipation scale
`k_d`. The Alfvénic fields must be measured using the physical perpendicular
fluctuation amplitudes

- `u_perp ~ grad_perp phi`
- `b_perp ~ grad_perp psi`

so the controller uses the same quadratic modal density as the solver's saved
`total_energy` diagnostic rather than raw `|phi_hat|^2` or `|psi_hat|^2`.
That includes the same compressive-sector weights used by the S09 budget
diagnostics. With `shell_energy = "alfvenic"` the controller measures only the
`u_perp + b_perp` part instead, so a passive scalar (drho at g = 0, whose size
is set by the arbitrary background gradient K_rho0) cannot change nu.

Around `k_d`, the controller defines a logarithmic shell

- `k_d * exp(-shell_half_width) <= k_perp <= k_d * exp(+shell_half_width)`

and measures its energy `E_d`. This gives an amplitude estimate

`u_d = sqrt(2 E_d)`.

Balancing the nonlinear rate and perpendicular hyperdissipation rate at `k_d`
then gives

`nu_perp,target = u_d * k_d^(1 - 2 n_perp)`.

The updated coefficient is smoothed in log space to avoid noisy jumps:

`log(nu_new) = (1 - smooth_factor) * log(nu_old) + smooth_factor * log(nu_target)`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from rmhdgpu.config import AutoDissipationSettings
from rmhdgpu.fourier_diagnostics import modal_density_average


def disabled_auto_dissipation_diagnostics() -> dict[str, float]:
    """Return stable default scalar-output values when auto mode is off."""

    return {
        "auto_dissipation_enabled": 0.0,
        "auto_dissipation_nu_perp": 0.0,
        "auto_dissipation_nu_par": 0.0,
        "auto_dissipation_kd": 0.0,
        "auto_dissipation_ud": 0.0,
        "auto_dissipation_Ed": 0.0,
    }


@dataclass(slots=True)
class AutoDissipationController:
    """Track and update one common effective perpendicular hyperdissipation."""

    settings: AutoDissipationSettings
    equation_module: Any
    field_names: list[str]
    grid: Any
    backend: Any
    retained_mask: Any
    shell_mask: Any
    kd: float
    current_nu_perp: float
    last_ud: float = 0.0
    last_Ed: float = 0.0

    @classmethod
    def from_runtime(
        cls,
        *,
        settings: AutoDissipationSettings,
        equation_module: Any,
        field_names: list[str],
        grid: Any,
        backend: Any,
        dealias_mask: Any | None,
    ) -> "AutoDissipationController":
        xp = backend.xp
        retained_mask = (
            xp.ones(grid.fourier_shape, dtype=grid.real_dtype)
            if dealias_mask is None
            else dealias_mask.astype(grid.real_dtype, copy=False)
        )

        kperp = xp.sqrt(grid.kperp2)
        retained_kperp = kperp * retained_mask
        max_retained_kperp = backend.scalar_to_float(xp.max(retained_kperp))
        if max_retained_kperp <= 0.0:
            raise ValueError(
                "Auto dissipation could not determine a positive retained k_perp,max. "
                "Check the grid size and dealias mask."
            )

        kd = settings.kd_fraction * max_retained_kperp
        lower = kd * math.exp(-settings.shell_half_width)
        upper = kd * math.exp(+settings.shell_half_width)
        shell_mask = ((kperp >= lower) & (kperp <= upper)).astype(grid.real_dtype, copy=False) * retained_mask

        if backend.scalar_to_float(xp.sum(shell_mask)) <= 0.0:
            raise ValueError(
                "Auto dissipation selected an empty shell around k_d. "
                "Adjust kd_fraction or shell_half_width."
            )

        return cls(
            settings=settings,
            equation_module=equation_module,
            field_names=list(field_names),
            grid=grid,
            backend=backend,
            retained_mask=retained_mask,
            shell_mask=shell_mask,
            kd=kd,
            current_nu_perp=max(settings.nu_min, 1.0e-300),
        )

    def should_update(self, step: int) -> bool:
        """Return `True` when the controller should refresh its coefficient."""

        return step > 0 and step % self.settings.update_every == 0

    def shell_energy_density(self, state: Any, params: Any) -> Any:
        """Modal energy density the controller measures, selected by `settings.shell_energy`.

        "total" is the solver's total-energy density, every evolved field included.
        "alfvenic" keeps only the flow, `0.5 k_perp^2 (|phi_hat|^2 + |psi_hat|^2)`,
        i.e. `u_perp ~ grad_perp phi` and `b_perp ~ grad_perp psi` with the same
        normalization as the Alfvénic part of "total". Every equation set
        evolves `psi` and `omega`, so this works for all of them.
        """

        if self.settings.shell_energy == "alfvenic":
            xp = self.backend.xp
            phi_hat = self.equation_module.derive_phi_hat(state["omega"], self.grid)
            return 0.5 * self.grid.kperp2 * (xp.abs(phi_hat) ** 2 + xp.abs(state["psi"]) ** 2)
        return self.equation_module.total_energy_modal_density(state, self.grid, self.backend, params)

    def update(self, state: Any, params: Any) -> float:
        """Refresh the effective coefficient from the current Fourier state."""

        density_hat = self.shell_energy_density(state, params)
        self.last_Ed = modal_density_average(
            density_hat,
            self.grid,
            self.backend,
            mask=self.shell_mask,
        )

        # The shell energy defines a fluctuation amplitude at the dissipation
        # scale through u_d = sqrt(2 E_d). This is a scalar amplitude estimate,
        # not the raw Fourier coefficient of any one evolved field.
        self.last_ud = math.sqrt(max(0.0, 2.0 * self.last_Ed))
        nu_target = self.last_ud * (self.kd ** (1 - 2 * self.settings.n_perp))
        nu_target = min(max(nu_target, self.settings.nu_min), self.settings.nu_max)

        previous = min(max(self.current_nu_perp, self.settings.nu_min), self.settings.nu_max)
        if self.settings.smooth_factor > 0.0:
            log_new = (
                (1.0 - self.settings.smooth_factor) * math.log(previous)
                + self.settings.smooth_factor * math.log(nu_target)
            )
            nu_new = math.exp(log_new)
        else:
            nu_new = previous

        if self.settings.max_update_factor > 1.0:
            lower = previous / self.settings.max_update_factor
            upper = previous * self.settings.max_update_factor
            nu_new = min(max(nu_new, lower), upper)

        self.current_nu_perp = min(max(nu_new, self.settings.nu_min), self.settings.nu_max)
        return self.current_nu_perp

    def effective_dissipation(self) -> dict[str, dict[str, float | int]]:
        """Return the current fieldwise dissipation spec consumed by the operator builder."""

        common = {
            "nu_perp": float(self.current_nu_perp),
            "nu_par": float(self.settings.nu_par),
            "n_perp": int(self.settings.n_perp),
            "n_par": int(self.settings.n_par),
        }
        return {field_name: dict(common) for field_name in self.field_names}

    def diagnostics(self) -> dict[str, float]:
        """Return scalar-output diagnostics for the controller state."""

        return {
            "auto_dissipation_enabled": 1.0,
            "auto_dissipation_nu_perp": float(self.current_nu_perp),
            "auto_dissipation_nu_par": float(self.settings.nu_par),
            "auto_dissipation_kd": float(self.kd),
            "auto_dissipation_ud": float(self.last_ud),
            "auto_dissipation_Ed": float(self.last_Ed),
        }
