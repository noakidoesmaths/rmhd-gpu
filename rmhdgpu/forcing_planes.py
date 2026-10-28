"""Compact parallel-plane normalization and diagnostics for controlled forcing.

Positive interior rFFT planes represent their negative conjugates; their
energy weight is two, applied once by control.weights. The evolved native
branch is never rotated. Only the additive direction is randomized.
"""
import math

import numpy as np


def equal_plane_direction(control, branch, direction, angles, shell):
    """Give each actuator plane the same direction norm, shell/n_planes.

    Within each nonempty plane retain the baseline direction a_k exp(i theta).
    The scale is uniform within that plane. An exactly empty plane uses the
    saved startup-template magnitudes with the same already-drawn angles;
    no extra RNG draw or change to the existing field is made. The caller
    contracts the final rounded direction to solve one aggregate work equation.
    Single-plane callers skip this function, retaining legacy arithmetic.
    """
    backend, xp = control.backend, control.backend.xp
    count = control.perpendicular_mode_count
    weights = backend.to_numpy(control.weights).astype(np.float64)
    target = shell / len(control.kz_indices)
    for index, _ in enumerate(control.kz_indices):
        rows = slice(index*count, (index+1)*count)
        actual = backend.to_numpy(direction[rows]).astype(np.complex128)
        norm = math.fsum(weights[rows]*abs(actual)**2)
        if norm == 0:
            direction[rows] = xp.abs(control.templates[branch][rows])*xp.exp(1j*angles[rows])
            actual = backend.to_numpy(direction[rows]).astype(np.complex128)
            norm = math.fsum(weights[rows]*abs(actual)**2)
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError("Nonfinite or empty random-increment plane template.")
        direction[rows] *= backend.asarray(math.sqrt(target/norm), dtype=control.grid.real_dtype)
    return direction


def event_moments(control, old, new, weights, direction, branch):
    """Actual sparse plane work, with linear and quadratic terms kept explicit.

    Aggregate controller work keeps its original native reduction and is not
    replaced by this partition sum. Small summation differences are reported.
    A plane can receive negative work during positive aggregate injection.
    """
    result = {}
    count = control.perpendicular_mode_count
    partition = []
    for index, nz in enumerate(control.kz_indices):
        rows = slice(index*count, (index+1)*count)
        a, z, w = old[rows], new[rows], weights[rows]
        d = z-a
        pre, post = math.fsum(w*abs(a)**2), math.fsum(w*abs(z)**2)
        work = math.fsum(w*(abs(z)**2-abs(a)**2))
        values = dict(actuator_pre=pre, actuator_post=post, work=work,
            linear_work=math.fsum(2*w*np.real(np.conj(a)*d)),
            quadratic_work=math.fsum(w*abs(d)**2),
            direction_norm2=None if direction is None else math.fsum(w*abs(direction[rows])**2))
        for key, value in values.items():
            result[f"nz{nz}_{key}_f"] = value
            result[f"nz{nz}_{key}_z"] = None if value is None else value/control.g2[branch]
        partition.append(work)
    result["plane_work_sum_f"] = math.fsum(partition)
    result["plane_work_sum_z"] = result["plane_work_sum_f"]/control.g2[branch]
    return result


def diagnostics(control, state, branch):
    """Full-plane branch energies for kz=0 and every actuated positive plane."""
    xp, backend = control.backend.xp, control.backend
    prefix = f"forcing_{branch}_"
    result = {}
    for nz in (0, *control.kz_indices):
        field = control.equation.forcing_branch_values(control, state, branch,
            (slice(None), slice(None), nz))
        density = xp.abs(field)**2 * control.inv_delta
        energy = backend.scalar_to_float(xp.sum(density)) * (1 if nz == 0 else 2)/(4*control.normalization)
        result[prefix+f"nz{nz}_energy_f"] = energy
        result[prefix+f"nz{nz}_energy_z"] = energy/control.g2[branch]
    return result


def metadata(control):
    return dict(reference_harmonic=control.settings.kz_index,
        actuator_harmonics=list(control.kz_indices),
        order="parallel plane first, then the original transverse lattice order",
        total_target="One branch target over the union, never a separate target per actuator plane",
        startup="Equal initial template energy per plane and per transverse mode; total template norm is one",
        positive_direction="For multiple planes, preserve a_k exp(i theta) within each nonempty plane and rescale its norm to aggregate pre-kick shell energy/n_planes; an exactly empty plane uses startup-template magnitudes with the same angles. Singleton arithmetic is unchanged.",
        applied_work="One signed amplitude solves the aggregate energy increment including its cross term; individual plane work need not be equal or positive",
        reference="kz_index alone sets native forcing-frequency defaults; nmax does not change tau_F, gamma_max or interval_max",
        rfft="Interior positive planes carry weight two once, including their implied negative conjugates; kz0 carries weight one",
        measurement="Perpendicular-shell feedback uses unique transverse indices and all retained kz exactly once",
        work_partition="Sparse FP64 contractions of actual native before/after fields; aggregate accepted-step work retains its original reduction",
        report_parallel_planes=control.settings.report_parallel_planes)
