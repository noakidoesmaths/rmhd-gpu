"""Full-circle random additive forcing on the compact controlled actuator.

Only positive controller gains use a random direction. Negative gains retain
baseline real multiplication, and zero gain does not touch the fields. The
controller supplies g unchanged; this module changes how its work is applied.
"""
import hashlib
import math

import numpy as np

MODEL = "random_increment_phase"


def signed_amplitude(log_gain, overlap, norm_ratio=1.0):
    """Small root of alpha*b**2 + 2*c*b = expm1(2*g), for g >= 0.

    alpha is the *realized* sparse direction norm divided by the pre-kick norm.
    Taking sign_plus(c) avoids a finite increment as g -> 0 when c < 0.
    hypot avoids unnecessary overflow in c*c + alpha*r. No clipping is used.
    """
    g, c, alpha = float(log_gain), float(overlap), float(norm_ratio)
    if not all(math.isfinite(v) for v in (g, c, alpha)) or g < 0 or alpha <= 0:
        raise ValueError("Random-increment root requires finite g>=0 and alpha>0.")
    if g == 0:
        return 0.0
    try:
        r = math.expm1(2*g)
    except OverflowError as exc:
        raise ValueError("Nonfinite requested random-increment energy gain.") from exc
    if not math.isfinite(r):
        raise ValueError("Nonfinite requested random-increment energy gain.")
    root = math.hypot(c, math.sqrt(alpha)*math.sqrt(r))
    b = r/(abs(c)+root)
    if not math.isfinite(b):
        raise ValueError("Nonfinite random-increment amplitude.")
    return -b if c < 0 else b


def configure(control, branch, seed_sequence):
    """Independent host stream; construction consumes no startup draws."""
    control.phase_rng[branch] = np.random.default_rng(seed_sequence)
    control.phase_history[branch] = dict(
        draws=0, injection_events=0, damping_events=0, zero_events=0,
        phase=np.zeros(control.mode_count), qv=np.zeros(control.mode_count),
        valid_events=np.zeros(control.mode_count, dtype=np.int64))


def apply(control, state, branch, event, before):
    """Apply one event using only sparse native coefficients and host moments.

    The direction u is formed in the field precision. FP64 sparse contractions
    measure its actual alpha and overlap, without changing controller reductions.
    The final update is explicit native-dtype addition, a + b*u. Actual rounded
    before/after coefficients remain authoritative for all existing work hooks.
    """
    backend, xp = control.backend, control.backend.xp
    theta = control.phase_rng[branch].uniform(-np.pi, np.pi, control.mode_count)
    record = control.phase_history[branch]
    record["draws"] += 1
    old = backend.to_numpy(before).astype(np.complex128)
    weights = backend.to_numpy(control.weights).astype(np.float64)
    power = weights*abs(old)**2
    shell = math.fsum(power)
    if not math.isfinite(shell) or shell <= 0:
        raise ValueError("Random-increment actuator has no positive finite norm.")
    g = event["log_gain"]
    if not math.isfinite(g):
        raise ValueError("Nonfinite random-increment controller log gain.")
    alpha = overlap = direction_norm = None
    native_direction = None
    amplitude = applied_amplitude = 0.0
    if g > 0:
        angles = backend.asarray(theta, dtype=control.grid.real_dtype)
        direction = before*xp.exp(1j*angles)
        if len(control.kz_indices) > 1:
            from rmhdgpu.forcing_planes import equal_plane_direction
            direction = equal_plane_direction(control, branch, direction, angles, shell)
        native_direction = backend.to_numpy(direction).astype(np.complex128)
        direction_norm = math.fsum(weights*abs(native_direction)**2)
        overlap = math.fsum(weights*np.real(np.conj(old)*native_direction))/shell
        alpha = direction_norm/shell
        amplitude = signed_amplitude(g, overlap, alpha)
        # Cast the scalar once; all applied array arithmetic stays native.
        applied_amplitude = float(np.asarray(amplitude, dtype=control.grid.real_dtype))
        after = before + backend.asarray(amplitude, dtype=control.grid.real_dtype)*direction
        control.equation.forcing_seed_branch(control, state, branch, after)
        record["injection_events"] += 1
    elif g < 0:
        # angles remain their initial zeros for this model. This is precisely
        # the existing phase-none multiplication, including its dtype/order.
        factor = xp.exp(g*(1-1j*xp.tan(control.angles[branch])))
        control.equation.forcing_apply_gain(control, state, branch, factor)
        record["damping_events"] += 1
    else:
        # In particular, do not multiply by complex one: preserve signed zeros.
        record["zero_events"] += 1
    new = backend.to_numpy(control.equation.forcing_branch_values(
        control, state, branch, control.indices)).astype(np.complex128)
    change = new-old
    valid = (abs(old)>0)&(abs(new)>0)
    delta = np.zeros(control.mode_count)
    # Explicit real products give exactly zero for unchanged coefficients.
    # A vectorized complex conjugate product can fuse one multiply/subtract
    # and leave an artificial imaginary rounding residual at exact identity.
    a, z = old[valid], new[valid]
    delta[valid] = np.arctan2(z.imag*a.real-z.real*a.imag,
                            z.real*a.real+z.imag*a.imag)
    record["phase"][valid] += delta[valid]
    record["qv"][valid] += delta[valid]**2
    record["valid_events"][valid] += 1
    linear_work = math.fsum(2*weights*np.real(np.conj(old)*change))
    quadratic_work = math.fsum(weights*abs(change)**2)
    modal_work = weights*(abs(new)**2-abs(old)**2)
    total_work = math.fsum(modal_work)
    selected = delta[valid]
    phase_mean = float(np.mean(selected)) if len(selected) else None
    result = dict(
        phase_draw_index=record["draws"], phase_direction_used=int(g>0),
        phase_raw_sha256=hashlib.sha256(theta.tobytes()).hexdigest(),
        phase_raw_cos_mean=float(np.mean(np.cos(theta))),
        phase_raw_sin_mean=float(np.mean(np.sin(theta))),
        phase_raw_min=float(theta.min()), phase_raw_max=float(theta.max()),
        phase_overlap=overlap, phase_direction_norm_ratio=alpha,
        phase_signed_amplitude=amplitude, phase_applied_amplitude=applied_amplitude,
        phase_direction_norm2_f=direction_norm, phase_increment_norm2_f=quadratic_work,
        phase_linear_work_f=linear_work, phase_quadratic_work_f=quadratic_work,
        phase_actual_work_f=total_work,
        phase_work_identity_error_f=total_work-linear_work-quadratic_work,
        phase_negative_modal_work_fraction=float(np.count_nonzero(modal_work<0)/len(modal_work)),
        phase_actual_zero_work=int(total_work==0),
        phase_valid_modes=int(np.count_nonzero(valid)),
        phase_delta_mean=phase_mean,
        phase_delta_rms=float(np.sqrt(np.mean(selected**2))) if len(selected) else None,
        phase_delta_variance=float(np.mean((selected-phase_mean)**2)) if len(selected) else None,
        phase_delta_max_abs=float(np.max(abs(selected))) if len(selected) else None,
        phase_energy_weighted_delta2=math.fsum(power*delta**2)/shell,
        phase_cumulative_qv_mean=float(np.mean(record["qv"])),
        phase_injection_events=record["injection_events"],
        phase_damping_events=record["damping_events"], phase_zero_events=record["zero_events"])
    if control.settings.report_parallel_planes:
        from rmhdgpu.forcing_planes import event_moments
        result.update(event_moments(control, old, new, weights, native_direction, branch))
    return result


def history(control):
    return dict(
        phase_rng_states={b:rng.bit_generator.state for b,rng in control.phase_rng.items()},
        phase_history={b:{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in r.items()}
                       for b,r in control.phase_history.items()})


def restore(control, saved):
    states, records = saved.get("phase_rng_states"), saved.get("phase_history")
    branches = set(control.settings.branches)
    if not isinstance(states,dict) or set(states)!=branches:
        raise ValueError("Random-increment restart requires every random-generator state.")
    if not isinstance(records,dict) or set(records)!=branches:
        raise ValueError("Random-increment restart requires every phase history.")
    validated = {}
    for branch,r in records.items():
        out = {}
        for key in ("draws","injection_events","damping_events","zero_events"):
            value=r[key]
            if isinstance(value,bool) or not isinstance(value,int) or value<0:
                raise ValueError("Invalid random-increment event count.")
            out[key]=value
        if out["draws"] != sum(out[k] for k in ("injection_events","damping_events","zero_events")):
            raise ValueError("Inconsistent random-increment event counts.")
        if out["draws"] != saved["event_count"]:
            raise ValueError("Random-increment draw count disagrees with forcing events.")
        for key in ("phase","qv","valid_events"):
            a=np.asarray(r[key])
            if a.shape!=(control.mode_count,) or not np.isfinite(a).all():
                raise ValueError("Invalid random-increment phase history.")
            if key in {"qv","valid_events"} and np.any(a<0):
                raise ValueError("Negative random-increment phase history.")
            if key=="valid_events" and (np.any(a!=np.floor(a)) or np.any(a>out["draws"])):
                raise ValueError("Invalid phase validity counts.")
            out[key]=a.astype(np.int64 if key=="valid_events" else np.float64)
        validated[branch]=out
    for branch,rng in control.phase_rng.items():rng.bit_generator.state=states[branch]
    control.phase_history=validated


def metadata():
    return dict(
        raw_direction="u_k=a_k exp(i theta_k), independent theta_k uniform[-pi,pi] before global energy-control sign",
        positive_gain="alpha=<u,u>/<a,a>, c=Re<a,u>/<a,a>; b=sign_plus(c)*expm1(2g)/(|c|+sqrt(c*c+alpha*expm1(2g))); a_new=a+b*u",
        negative_gain="Unchanged phase-none real multiplicative damping; no random increment",
        zero_gain="No field operation; exact identity",
        draw_schedule="One full-circle draw per mode at every actual event, including damping/zero events and final flush; only positive gains apply the direction. No startup or pending-interval draws.",
        streams="SeedSequence(forcing_seed).spawn(4); PCG64 streams 2/3 for plus/minus; original startup streams 0/1 unchanged",
        replay="Use forcing_modes.csv order, seed, NumPy version, branch stream and phase_draw_index; phase_raw_sha256 verifies each raw float64 draw",
        precision="Native-dtype direction and explicit additive update; FP64 contractions over sparse actual coefficients determine alpha/c and diagnostic moments",
        phase_statistics="Actual rounded before/after field phases, with per-mode cumulative quadratic variation and validity counts in checkpoints; not a promised decorrelation time",
        modal_work="Individual modal work can be negative during positive shell-total injection",
        sign_conditioning="Global sign b conditions applied increment phases; post-control phases are not unconditioned independent uniform noise",
        inactive_legacy_setting="phase_angle_max applies only to fixed_gain_angle; this model always draws the full circle")
