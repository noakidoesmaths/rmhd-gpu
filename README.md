# rmhdgpu

`rmhdgpu` is a single-node Fourier pseudo-spectral RMHD-style solver with three backend modes:

- `numpy`: baseline CPU path using `numpy.fft`
- `scipy_cpu`: CPU path using `scipy.fft`
- `cupy`: single-GPU path using `cupy.fft`

Current solver scope includes:

- multiple equation sets selected from `.input` files
- ideal homogeneous S09 five-field equations
- low-beta stratified three-field RMHD equations with a non-conservative energy budget
- anisotropic dissipation with integrating-factor time stepping
- optional auto-dissipation with one common adaptive perpendicular coefficient
- variable timestep support
- stochastic forcing and controlled Alfvénic shell forcing
- persistent scalar, spectral, and full-field diagnostics
- NumPy, SciPy CPU, and CuPy backends
- tests plus lightweight profiling utilities

## Installation

The package supports Python 3.10+.

From the repository root, install the package in editable mode:

```bash
python -m pip install -e .
```

On Python 3.10, `pip` will also install `tomli` automatically so `.input` files
continue to parse the same way.

After that, `python -m rmhdgpu.run ...` works from any folder, not just from inside the repository checkout. That is the most convenient setup for case-directory workflows and for running on a remote cluster.

## Running Simulations

Use `--initial-seed 42` to override `[initial_condition.parameters] seed` for
an initializer that supports a seed (for example, `random_spectrum`):

```bash
python -m rmhdgpu.run input.input --initial-seed 42
```

This is independent of `--forcing-seed`. If omitted, the initial-condition
seed from the input file (or the initializer default) is preserved.


There are two normal run modes:

- input-file mode: `python -m rmhdgpu.run filename.input`
- CLI-only mode: `python -m rmhdgpu.run --options ...`

The usual workflow for real runs is a case directory with a commented `.input` file:

```bash
python -m rmhdgpu.run cases/my_forced_case/input.input
python -m rmhdgpu.run cases/my_forced_case/input.input --tmax 20.0 --backend cupy
python -m rmhdgpu.run --backend numpy --nx 64 --ny 64 --nz 64 --tmax 0.1
```

Recommended case layout:

```text
my_case/
    input.input
    outputs/
        resolved_config.toml
        run.log
        scalar_diagnostics.csv
        spectra.csv
        energy_flux.csv
        fullfields/
            fullfield_0001.h5
            fullfield_0002.h5
            ...
```

`.input` files use TOML syntax internally.

If `output_dir` is omitted in a `.input` file, the driver writes to `outputs` relative to the input-file location. In CLI-only mode, the default is `./outputs` relative to the current working directory.

Equation sets are selected in the input file:

```toml
[equations]
type = "s09"
# Optional: use "linear" to omit nonlinear RHS terms while keeping the same
# timestepper, dissipation, forcing, diagnostics, and outputs.
mode = "nonlinear"
```

Available equation sets are:

- `alfvenic`: lightweight two-field Alfvénic system with fields `psi`, `omega`
- `s09`: homogeneous five-field system with fields `psi`, `omega`, `upar`, `dbpar`, `s`
- `low_beta_stratified`: three-field system with fields `psi`, `omega`, `a`

The selected equation set determines the evolved field list, so manual dissipation and per-field forcing-rate settings must use the matching field names.

Use `alfvenic` when you only want the Alfvénic turbulence dynamics and do not
need the slow/entropy sector from `s09`; it reduces memory use and the amount
of work per timestep. Its equation implementation is intentionally more direct
and less readable than the other equation sets because this is the performance
path for large two-field GPU runs.

The optional `[equations] mode = "linear"` switch is useful for tests and
teaching examples. It runs the same solver workflow and still calls
`ideal_rhs`, but replaces the Poisson-bracket operator by zero so nonlinear
terms are omitted. If `mode` is omitted, the default is `"nonlinear"`.

## Diagnostics Output

Diagnostics are controlled independently through the `[output]` section:

- `t_out_scal`: scalar diagnostics cadence
- `t_out_spec`: spectra cadence
- `t_out_prof`: `z_profiles.csv` cadence (inhomogeneous sets); defaults to `t_out_spec`, so set it on its own to save profiles finely without crowding the spectra
- `t_out_full`: full-field cadence

Cadences are in simulation time units:

- positive value: enable that output category
- `0` or negative: disable that output category

Outputs are:

- `scalar_diagnostics.csv`: one row per output time, with `time`, `step`, scalar quantities, and saved budget terms
- `spectra.csv`: tidy/long CSV with columns `time`, `step`, `quantity`, `kperp`, `value`
- `energy_flux.csv` (written with the spectra, same long format): the scale-by-scale energy budget. For each energy partition (`total`, `z_plus`, `z_minus`, and per-field energies such as `drho` in the inhomogeneous sets), it saves the rate each kind of term changes that energy in each `k_perp` shell: `nonlinear_<partition>` (the transfer `T(k)` by the Poisson brackets, which sums to zero over `k`), `linear_<partition>` (background gradients, gravity and `d_z` terms) and `dissipation_<partition>`. Forcing kicks and the `project_*` mode projections are not included (their totals are in `scalar_diagnostics.csv`). The energy flux through `k_perp = K` is `Pi(K) = -sum_{k<K} T(k)`; `vis/plot_energy_flux.py` plots it. How it is computed is explained in `rmhdgpu/diagnostics/energy_flux.py`
- `z_profiles.csv` (inhomogeneous sets only, written every `t_out_prof`): the same long format with a `z` column, holding perpendicular averages `<.>_perp(z)` of the energy reservoirs and their local budget terms. For each of the six energies it also has `<energy>_dt`, the instantaneous `d_t W(z)` from the right-hand side, and `<energy>_<term>` for every term (including `w_plus_transport = -vA d_z W+`), so each local budget can be checked at every z
- `fullfields/fullfield_0001.h5`, `fullfield_0002.h5`, ...: one HDF5 snapshot per output time

Scalar diagnostics are split into generic per-field columns such as
`psi_rms`, `psi_mean`, and `psi_max_abs`, plus equation-specific quantities
defined by the selected equation module.

The inhomogeneous (`inhomogeneous_rmhd_rho` / `_s`) equation sets also write
the energy in the `k_z = 0` plane: `E_kz0_ky0_<field>` (the `k_y = 0` mean
profiles `<f>(x)`), `E_kz0_kyn0_<field>` and `E_kzn0_<field>` for `z_plus`,
`z_minus`, `du_par`, `db_par` and the entropy field, plus the totals
`E_kz0_ky0 + E_kz0_kyn0 + E_kzn0 = total_energy`.
They also write the split by `k_y` alone, counting every `k_z`:
`E_ky0_<field>` and `E_kyn0_<field>`, with totals `E_ky0 + E_kyn0 = total_energy`.
Likewise by `k_x` alone: `E_kx0_<field>` and `E_kxn0_<field>`, with totals
`E_kx0 + E_kxn0 = total_energy`.
The same is written for `k_perp = 0` (`k_x = k_y = 0`, the z-only profiles):
`E_kperp0_<field>` and `E_kperpn0_<field>`, with totals `E_kperp0 + E_kperpn0 = total_energy`.
`z_plus` and `z_minus` have no energy at `k_perp = 0`.

The inhomogeneous sets also split the energy into six reservoirs, each with
its own budget (`<name>`, `<name>_rhs_<term>`, `<name>_rhs_total`):

| Column | Reservoir | Budget terms |
|---|---|---|
| `w_plus`, `w_minus` | `W± = <\|z±\|^2>/4` | `buoyancy` = `(g/2)<drho z±_x>`, `dissipation`, `forcing` |
| `w_par_u` | `<du_par^2>/2` | `drive_KB`, `exchange_lin`, `exchange_nl`, `dissipation`, `forcing` |
| `w_par_b` | `vA^2 <db_par^2>/2` (magnetic pressure) | `drive_KBKp`, `exchange_lin`, `exchange_nl`, `dissipation`, `forcing` |
| `w_p` | `w_par_b / chi` (thermal pressure) | the `w_par_b` terms times `1/chi` |
| `w_s` | `cs^2/(gamma^2(gamma-1)) <s^2>/2` (entropy) | `drive_Ks`, `dissipation`, `forcing` |

The six add up to `total_energy` (for the rho set only when `vA = 1`). The
`exchange_*` terms move energy between `w_par_u` and `w_par_b + w_p` and cancel
in their sum; the `drive_*` and `buoyancy` terms add up to the `acr_*` channels.
The box averages lose the `vA d_z W±` transport terms, which average to zero
in a periodic box; `z_profiles.csv` keeps them. `k_prl_rms_plus` is the rms
`k_par` of `z+`, so `vA k_prl_rms_plus w_plus` is the size of that term. The
exact definitions are in `rmhdgpu/diagnostics/compressive_channels.py` and
`rmhdgpu/diagnostics/z_profiles.py`.

They also write the rms size of each term on the right-hand side, so the
nonlinearity can be measured without full-field output (`||f|| = <|f|^2>^(1/2)`,
z± terms in z units):

| Column | Meaning |
|---|---|
| `rms_nl_z_minus`, `rms_lin_z_minus` | `\|\|P[z+ . grad z-]\|\|` and `\|\|vA d_z z-\|\|` in the z- equation |
| `chi_rhs_minus` | their ratio: chi for z- being distorted by z+ (`_kzn0`: `k_z != 0` part of the nonlinear term only) |
| `..._plus` | the same for z+, which is distorted only by z- |
| `rms_buoy_z` | rms of the buoyancy term, the same in both Elsasser equations |
| `rms_nl_s`, `rms_drive_s`, `rms_prl_s` | `\|\|{phi, s}\|\|`, `\|\|K_s dy phi\|\|`, `\|\|vA d_z s\|\|` for `s = delta s/c_v` |
| `chi_s_drive`, `chi_s_prl` | `rms_nl_s` over `rms_drive_s` (a mixing-length ratio) and over `rms_prl_s` |

`P` drops the part of the advection that the pressure cancels. Ratios with a
zero denominator are NaN. They are written every `t_out_scal` and cost about
one extra RHS evaluation. Details are in `rmhdgpu/diagnostics/rhs_ratios.py`;
`python vis/plot_chi_rhs.py RUN_DIRECTORY` plots them against time.

The scalar CSV also carries saved conserved-quantity budget columns. For the
current total-energy budget these include:

- `total_energy`: the conserved quadratic energy used for the budget check
- `total_energy_rhs_dissipation`: signed contribution to `d_t E` from dissipation, so it is negative when energy is removed
- `total_energy_rhs_forcing`: signed contribution to `d_t E` from forcing, based on the actual kick-induced energy change over the preceding scalar-output interval
- `total_energy_rhs_total`: sum of the saved signed RHS terms

The sign convention is:

- `d_t E = total_energy_rhs_total`
- individual RHS terms are stored with their physical sign

If auto dissipation is enabled, the scalar CSV also includes:

- `auto_dissipation_enabled`
- `auto_dissipation_nu_perp`
- `auto_dissipation_nu_par`
- `auto_dissipation_kd`
- `auto_dissipation_ud`
- `auto_dissipation_Ed`

The full-field snapshots are self-contained. Each file stores:

- `/metadata` with grid size, box size, backend, dtypes, and `x`, `y`, `z`
- `/output/time`
- `/output/step`
- `/output/<field_name>` for each saved real-space field

## `.input` File Format

The driver accepts TOML-based `.input` files with sections such as:

```toml
title = "Small forced turbulence test"
output_dir = "outputs"

[equations]
type = "s09"
# mode defaults to "nonlinear"; set mode = "linear" for linearized runs.

[grid]
Nx = 128
Ny = 128
Nz = 128

[time]
tmax = 10.0
dt_init = 1e-3
dt_max = 1e-2
cfl_number = 0.3

[output]
t_out_scal = 0.1
t_out_spec = 0.5
t_out_full = 0.0

[backend]
backend = "cupy"
fft_workers = 8

[forcing]
use_forcing = true
forcing_mode = "elsasser"
forcing_seed = 1234
epsilon_plus = 0.8
epsilon_minus = 0.2

[forcing.field_energy_injection_rates]
upar = 0.01

[initial_condition]
type = "zero"

[dissipation.psi]
nu_perp = 5e-3
nu_par = 0.0
n_perp = 3
n_par = 3
```

Supported sections are:

- top level: `title`, `output_dir`
- `[equations]`: `type`
- `[grid]`: `Nx`, `Ny`, `Nz`, `Lx`, `Ly`, `Lz`
- `[time]`: `tmax`, `dt_init`, `dt_min`, `dt_max`, `cfl_number`, `use_variable_dt`
- `[output]`: `t_out_scal`, `t_out_spec`, `t_out_full`
- `[backend]`: `backend`, `fft_workers`, `real_dtype`, `complex_dtype`
- `[runtime]`: `runtime_check_every`, `progress_output_every`, `fail_on_nonfinite`, `dealias`, `dealias_mode`, `project_kpar0`, `project_ky0`, `project_kx0`
  - `project_kpar0 = true` zeroes every `k_z = 0` mode (the z-average of each field) at startup and after every step; `project_ky0 = true` does the same for every `k_y = 0` mode (the y-average, at all `k_z`) and `project_kx0 = true` for every `k_x = 0` mode (the x-average). Any combination can be switched on. They are control options for testing what depends on those modes, not physical requirements. For `random_spectrum` and `random_spectrum_one_wave`, these projections are applied before normalizing to `init_energy`, so the requested energy belongs to the surviving modes. The energy removed each step is saved as the `projection` budget term; startup cleanup is not booked. On the command line use `--project-kpar0`, `--project-ky0`, `--project-kx0` (or `--no-project-kx0` etc. to switch off one that the input file turns on).
- `[physics]`: `vA`, `cs2_over_vA2`, `N2`
- `[forcing]`, `[forcing.field_energy_injection_rates]`, and `[forcing.controlled_shell]`
- `[dissipation]` for optional auto-dissipation control
- `[dissipation.<field>]` for manual per-field dissipation
- `[initial_condition]`

### Forcing

Choose `type = "stochastic"` (the default) for prescribed mean injection rates,
or `type = "controlled_shell"` for target-amplitude or constant-power control
of selected Alfvénic branches. These are separate forcing modes.

#### Stochastic forcing

Stochastic forcing is additive Gaussian noise refreshed every timestep. It is built
from a real, unit-variance Gaussian field, filtered to the integer-mode shell
`n_min_force <= sqrt(nx^2 + ny^2 + nz^2) <= n_max_force`, and shaped in
Fourier amplitude as `n^(-alpha_force)`. Here `alpha_force` controls only the
shape within the forcing band; it is unrelated to the imbalance-response
exponent discussed below.

The filtered realization is **not** normalized from its measured RMS or
energy. Instead, the code computes one fixed ensemble normalization for the
chosen grid, forcing band, and energy definition. A fresh unnormalized
realization is multiplied by `sqrt(dt)` on every step. Individual energy
increments therefore fluctuate naturally, while the configured `epsilon`
values set their ensemble means.

For `forcing_mode = "field"`, each entry in
`[forcing.field_energy_injection_rates]` applies to one evolved field. If
`xi_i` is the filtered Gaussian for field `i` and

```text
C_i = ensemble mean of total_energy(state with only q_i = xi_i),
```

then the kick is

```text
delta q_i = sqrt(epsilon_i * dt / C_i) * xi_i.
```

Thus `mean[total_energy(delta q_i)] = epsilon_i * dt`. The normalization uses
the selected equation module's `total_energy`, so the same input convention
works for potentials, vorticity, and weighted compressive fields such as
`upar`, `dbpar`, and `s`. Independently forced fields have independent random
realizations.

For `forcing_mode = "elsasser"`, the code independently forces

```text
zeta_plus  = phi - psi
zeta_minus = phi + psi
E_plus/minus = 0.5 * mean(|grad_perp zeta_plus/minus|^2).
```

`epsilon_plus` and `epsilon_minus` satisfy
`mean[E_plus/minus(delta zeta_plus/minus)] = epsilon_plus/minus * dt`. The
kicks are transformed back to `psi` and either `phi` or
`omega = lap_perp(phi)`, so this mode can be used by any equation set with
that Alfvénic storage convention. In the diagnostics used here,
`E_A = 0.5 * (E_plus + E_minus)`, so the mean injection into `E_A` is
`0.5 * (epsilon_plus + epsilon_minus)`.

Balanced forcing is selected with `epsilon_plus = epsilon_minus`. For an
injection imbalance `R_epsilon = epsilon_plus / epsilon_minus`, a useful 32^3
starting point that gives `u_perp,rms` of order one is:

```toml
[forcing]
use_forcing = true
forcing_mode = "elsasser"
n_min_force = 1.0
n_max_force = 3.0
alpha_force = 0.0
epsilon_plus = 0.35
epsilon_minus = 0.35
forcing_seed = 1234

[dissipation]
mode = "auto"
n_perp = 3
n_par = 1
nu_par = 0.0
kd_fraction = 0.6
update_every = 10
smooth_factor = 0.4
```

In the 32^3 sanity run, this balanced choice gave approximately
`u_perp,rms = 0.94` at `t = 32`. These values are practical starting points,
not resolution-independent calibrations. To increase imbalance while keeping
the overall fluctuation amplitude roughly comparable, the current sanity scan
uses

```text
epsilon_sum   = 0.7 * R_epsilon^(-2/3)
epsilon_plus  = epsilon_sum * R_epsilon / (1 + R_epsilon)
epsilon_minus = epsilon_sum / (1 + R_epsilon).
```

Previous work suggests, and the small 32^3 scan is consistent with,
`E_plus / E_minus ~ R_epsilon^2`. The measured imbalanced-only exponent was
about `2.04` for `R_epsilon = 2, 4, 8`. Strongly imbalanced cases equilibrate
more slowly, so production measurements should use longer runs, adequate
resolution, and preferably multiple seeds. Reproduce or modify this check with:

```bash
python -m rmhdgpu.examples.sanity_imbalanced_forcing
```

`[forcing.force_amplitudes]` and the removed `--force-sigma` option are rejected.
Specify `[forcing.field_energy_injection_rates]` or the Elsasser `epsilon_plus`
and `epsilon_minus` settings explicitly. Old amplitude values are not silently
reinterpreted as powers; historical stochastic runs require their pinned source.

#### Controlled shell forcing

<<<<<<< HEAD
Controlled forcing rescales one branch in a perpendicular Fourier band at a
selected nonzero parallel harmonic. It supports the `alfvenic`, `s09`,
`low_beta_stratified`, `inhomogeneous_rmhd_rho`, and `inhomogeneous_rmhd_s`
equations. With nonzero background gradients in the inhomogeneous sets, z± are
not linear eigenmodes, so energy put into one branch also leaks linearly into
the other branch and the compressive fields. A plus-only control changes `phi - psi` while
=======
Controlled forcing acts on a branch in a perpendicular Fourier band at selected
nonzero parallel harmonics (one by default). It supports the `alfvenic`, `s09`, and
`low_beta_stratified` equations. A plus-only control changes `phi - psi` while
>>>>>>> upstream/main
preserving `phi + psi` to roundoff, including an already nonzero minus branch.

For example, control the physical plus RMS towards 0.2:

```toml
[forcing]
type = "controlled_shell"
use_forcing = true
forcing_seed = 53

[forcing.controlled_shell]
k_sigma_min = 2.0
k_sigma_max = 3.0
kz_index = 1
branches = ["plus"]

[forcing.controlled_shell.plus]
control = "target"
target_basis = "elsasser"
target_quantity = "rms"
target_scope = "perpendicular_shell"
target_value = 0.2
tau_F = 1.0
```

The target is a relaxation reference, not an exact amplitude constraint.
`perpendicular_shell` measures the band across all retained parallel modes;
only the selected parallel harmonics receive forcing. Use `branch_total`,
`branch_nonzero_kz`, or `shell` for the other measurement scopes.

Controlled branch energy is `mean(|z±|²)/4`, half the existing stochastic
Elsasser-energy convention. Controlled constant-power values therefore need
that factor of two when compared with stochastic `epsilon_plus/minus`.

Runnable examples:

```bash
python -m rmhdgpu.run examples/controlled_target_s09.input
python -m rmhdgpu.run examples/controlled_power_alfvenic.input
python -m rmhdgpu.run examples/controlled_target_nokia_rho.input
```

Two optional controls can be added to the existing `[forcing.controlled_shell]`
table. All existing defaults are unchanged: `phase_model = "none"`, a single
actuator at `kz_index = 1`, and no per-plane reporting.

```toml
[forcing.controlled_shell]
# Retain the existing shell bounds, branches and branch target tables.
phase_model = "random_increment_phase"
kz_indices = [1, 2, 3, 4, 5]  # Optional; omitting it uses [kz_index].
report_parallel_planes = true  # Optional diagnostic output; default false.
```

`random_increment_phase` applies full-circle random phases to an **added
increment** on positive-gain events. One signed amplitude solves the requested
shell-energy change including the cross term with the existing field. Individual
modes can lose energy even when the shell total gains energy. Negative gains use
the original real damping; zero gain leaves the fields unchanged. The existing
`none` and `fixed_gain_angle` update arithmetic is preserved. `phase_angle_max`
only controls `fixed_gain_angle`; random-increment phases always span the full
circle. The global energy-control sign means the final applied increments are
not unconditioned independent uniform phases.

The seed determines independent startup and phase streams. Every actual forcing
event consumes one phase draw per mode, including damping and zero-gain events;
pending intervals consume none. Event CSVs record draw hashes, phase statistics,
and actual linear/quadratic work. The existing `forcing_history` and
`restore_forcing_history` APIs serialize the RNG and phase history as well as the
pending interval. Restoring these requires the same configuration and saved
fields; this does not add a whole-simulation restart facility.

`kz_indices` must be sorted, unique positive integers below the Nyquist plane and
inside the dealias mask. They share **one branch target or power**, not a full
target per plane. Startup template energy is divided equally across planes.
For positive random increments, each plane has equal direction energy before
the single aggregate work solve; actual injected work can differ by plane.
Perpendicular-shell feedback still counts every transverse mode and retained
parallel plane only once. Positive rFFT harmonics include their negative
conjugates through the usual factor of two.

`kz_index` remains the reference for the controller's default timescale/rate
settings. For example, `kz_index = 1, kz_indices = [5]` forces only harmonic 5
without changing those reference defaults. `report_parallel_planes` adds
zero-plane/actuator-plane energies and, for random increments, per-plane work;
it does not change amplitude feedback or the forced modes.

### Dissipation

For the low-beta stratified system, `N2` may be positive or negative. With the
sign convention used here, `N2 > 0` allows unstable/decaying linear branches,
while `N2 < 0` gives oscillatory stable branches. `N2 = 0` is still rejected.

Manual dissipation remains the default. In that mode, set per-field blocks such
as `[dissipation.psi]` and `[dissipation.omega]` exactly as before. The valid
field names come from `[equations].type`; for example `low_beta_stratified`
accepts `[dissipation.psi]`, `[dissipation.omega]`, and `[dissipation.a]`.

Auto dissipation is useful when you want one common hyperdissipation
coefficient chosen automatically from the fluctuation amplitude near a target
dissipation scale. In auto mode, the solver ignores per-field coefficients and
applies one common `(nu_perp, nu_par, n_perp, n_par)` choice to all evolved
fields.

Practical auto-dissipation example:

```toml
[dissipation]
mode = "auto"
n_perp = 3
n_par = 1
nu_par = 0.0
kd_fraction = 0.6
shell_half_width = 0.5
update_every = 10
smooth_factor = 0.2
nu_min = 1e-12
nu_max = 1e-2
max_update_factor = 2.0
```

Meaning of the main settings:

- `mode = "auto"`: enable one common adaptive perpendicular coefficient for all fields
- `kd_fraction`: choose `k_d` as this fraction of the maximum retained perpendicular wavenumber
- `update_every`: recompute the coefficient every this many timesteps, not every step
- `smooth_factor`: log-space smoothing strength; larger values react faster
- `nu_par`: fixed common parallel coefficient used in auto mode
- `n_perp`, `n_par`: common perpendicular and parallel hyper-orders used for every field
- `shell_energy`: what the controller measures in the `k_d` shell. `"total"` (the default) is the full `total_energy`, every field included; `"alfvenic"` counts only `u_perp + b_perp`. Use `"alfvenic"` when a field is a passive scalar, e.g. `drho` with `g = 0`: its size is set by the arbitrary `K_rho0`, so under `"total"` it would change how hard the flow is damped. The chosen coefficient is still applied to every field

### Initial Conditions

Currently supported initial conditions are:

- `initial_condition.type` selects a registered builder in `rmhdgpu.initconds`
- put initializer-specific options under `[initial_condition.parameters]` (flat keys under `[initial_condition]` still work for compatibility)
- `type = "alfven_mode"` with `k_indices = [kx, ky, kz]`, `amplitude`, and `branch = "z_plus"` (the default) or `"z_minus"`; `amplitude` rescales the mode so `total_energy ~ amplitude^2`. The branch names the Elsasser wave, `z± = δu⊥ ∓ δB⊥/√(4πρ)`: `z_plus` sets `phi = -psi` and travels along `+B0`, `z_minus` sets `phi = +psi`. The old names `"plus"`/`"minus"` meant the opposite waves and are now rejected
- `type = "zero"`
- `type = "aw_packet"`, a pure z- packet (`psi = phi`)
- `type = "random_spectrum_one_wave"` with `n_min_perp`, `n_max_perp`, `n_min_prl`, `n_max_prl`, `alpha`, `alpha_prl`, `init_energy`, `seed`, and `exclude_kpar0`; a band-limited random pure z+ state (`phi = -psi`) with the compressive fields zero, rescaled so `total_energy` matches `init_energy`
- `type = "random_spectrum"` with `n_min`, `n_max`, `alpha`, `init_energy`, and `seed`; it fills every evolved field with an independent band-limited random spectrum, then rescales the full state so the equation-module `total_energy` matches `init_energy`
- `type = "single_fourier_mode"` with `k_indices = [kx, ky, kz]`, `amplitude`, and `seed`; it puts independent random coefficients into the same Fourier mode for every evolved field
- `type = "low_beta_stratified_mode"` for the low-beta stratified linear eigensystem; for `N2 > 0`, `amplitude` rescales the mode so `total_energy ~ amplitude^2`

### Matching random initial fields across GPUs

Both `random_spectrum` and `random_spectrum_one_wave` accept
`initialization_backend = "numpy"` under `[initial_condition.parameters]`:

```toml
[initial_condition]
type = "random_spectrum_one_wave"

[initial_condition.parameters]
seed = 1234
initialization_backend = "numpy"
init_energy = 0.64
```

Add the option to your existing parameters, preserving your spectral bands and
other settings. This builds the complete initial state on the CPU (random draws,
FFTs, filtering, wave relations, and energy normalization), then copies the
Fourier coefficients to the chosen evolution backend. Thus H100, A100, and L40
receive identical initializer coefficients with the same seed, grid, mask,
precision, physical parameters, code, and NumPy environment. It also works with
the NumPy and SciPy CPU evolution backends. The default `"native"` preserves the
previous backend-local initialization. The option is recorded in resolved config.

This costs CPU initialization time and host memory for a full initial state and
scratch arrays, once at startup. It does not change subsequent forcing, startup
forcing adjustments, or time integration, which can still differ through GPU
roundoff and nonlinear amplification. It does not promise identical results
across NumPy versions or different CPU numerical environments; for archival
reproducibility, preserve the initial arrays and software environment as well.

Adding a new initial condition means adding and registering a builder in
`rmhdgpu.initconds`. For equation-specific eigenmodes, keep the reusable
eigenvector construction in a small module such as
`rmhdgpu/initconds/eigenmodes_<equation>.py`, and keep `builtin.py` as the
thin registry layer for input-file selectable names.

Small low-beta stratified example:

```toml
[equations]
type = "low_beta_stratified"

[physics]
vA = 1.0
N2 = 0.25

[initial_condition]
type = "low_beta_stratified_mode"

[initial_condition.parameters]
k_indices = [0, 1, 0]
mode = "unstable_growing"
amplitude = 0.02

[dissipation.a]
nu_perp = 1e-4
nu_par = 0.0
n_perp = 2
n_par = 1
```

For this equation set, scalar diagnostics include `total_energy_rhs_stratification` in addition to the usual dissipation, forcing, and total RHS budget columns.

## Example Inputs

The repository root includes ready-to-run example inputs:

- [`examples/aw_packet.input`](examples/aw_packet.input)
- [`examples/decay_spectra.input`](examples/decay_spectra.input)
- [`examples/decay_spectra_auto.input`](examples/decay_spectra_auto.input)
- [`examples/decay_spectra_gpu.input`](examples/decay_spectra_gpu.input)
- [`examples/forced_turbulence.input`](examples/forced_turbulence.input)
- [`examples/low_beta_stratified.input`](examples/low_beta_stratified.input)

For example:

```bash
python -m rmhdgpu.run examples/decay_spectra.input
python -m rmhdgpu.run examples/decay_spectra_auto.input
python -m rmhdgpu.run examples/decay_spectra_gpu.input
python -m rmhdgpu.run examples/low_beta_stratified.input
```

## Plotting Saved Output

The generic post-processing scripts live under [`vis/`](vis):

```bash
python vis/plot_scalars.py cases/my_case/outputs/scalar_diagnostics.csv
python vis/plot_budget.py cases/my_case/outputs/scalar_diagnostics.csv
python vis/plot_spectra.py cases/my_case/outputs/spectra.csv
python vis/plot_fullfield.py cases/my_case/outputs/fullfields --field omega --slice-dir z
```

Useful notes:

- `plot_scalars.py` plots common energy-like quantities by default, or specific columns via `--columns`
- `plot_budget.py` compares saved `Q(t)` and finite-difference `d_t Q` against saved `Q_rhs_*` terms for a conserved quantity such as `total_energy`
- `plot_spectra.py` writes one log-log plot per quantity, colored by time
- `plot_fullfield.py` accepts either a `fullfields/` directory or a single snapshot `.h5` file
- `plot_chi_a.py` plots the turbulence strength `chi_A = z+_rms k_perp_plus / (vA k_prl_plus)` against time, together with its factors `z+_rms`, `k_perp_plus` and `k_prl_plus`, for one or more inhomogeneous runs, e.g. `python vis/plot_chi_a.py "examples/outputs_nokia_rmhd_5*_rho" --show`
- `plot_chi_rhs.py` shows three panels with a shared time axis: each Elsasser branch has its `chi_A` proxy and `k_z != 0` RHS ratio, and the entropy panel shows the advection-to-drive ratio `chi_s_drive`. It reads `scalar_diagnostics.csv` only, e.g. `python vis/plot_chi_rhs.py RUN_DIRECTORY --show`.
- `plot_flux_closure.py` tests the flux closures in dimensionless form against `chi_A`, one window-averaged point per run: `V_rho_x/(F_rho z+ l_perp)` and `V_psi_x/((F_u + vS^2/vA^2 F_B) z+ l_perp)`, each with the run's own estimate `z+_x^2/(2 z+^2)` (hollow marker; 1/4 for isotropic `z+`, drawn as a grey line). `--panels` picks the panels from `V_rho`, `V_psi` and `drho` (raw `rms(drho)`, for reference); the default is `V_rho V_psi`, e.g. `python vis/plot_flux_closure.py "examples/outputs_*_rho" --tmin 2 --tmax 10 --panels V_rho V_psi drho`
- `plot_flux_convergence.py` checks that those closure points have settled: it plots `V_rho_x/(F_rho z+ l_perp)` (with block means and the run's estimate, dashed), `z-/z+`, `chi_A` and the auto-dissipation `nu_perp` against time, marks the averaging window, and prints each run's drift (second-half over first-half window mean, minus 1), e.g. `python vis/plot_flux_convergence.py "examples/Kp0_forcing/*"`
- `plot_zplus_heating.py` compares the two ways `z+` loses energy with their closure estimates, side by side: DCF (measured `q_dcf` against Eq. (69), `-g F_rho eta_turb`) and CCR (the `W+` dissipation rate against Eq. (75), `|g F_rho| l_perp z+_x / 4`). It prints the window means, each channel's share of the measured `z+` loss and `W+ z-_rms / l_perp` (CCR with the measured `z-`), and warns when `D+` cannot be read as CCR heating (`z-` forced, or more than 10% of it in the forced shell), e.g. `python vis/plot_zplus_heating.py RUN_DIRECTORY --tmin 100`
- `plot_slaved_gradient_scan.py` saves one figure per compressive field (`slaved_gradient_scan_drho.png` and so on) of saturated RMS against background drive across a scan; `--fields` picks which to draw
- `plot_kpar0_energy.py` shows how much of an inhomogeneous run's energy sits at `k_par = 0`, against time: the share of the total energy in waves (`k_par != 0`), 2D structures (`k_par = 0, ky != 0`) and mean profiles `<f>(x)` (`k_par = 0, ky = 0`), then each field's share in the mean profiles and in the whole `k_par = 0` plane. It reads the `E_kz*` columns of `scalar_diagnostics.csv`; runs made before those columns existed are summed from their `fullfields/` snapshots instead. `--exclude-fields z_plus` leaves the forced wave out of the total, e.g. `python vis/plot_kpar0_energy.py RUN_DIRECTORY`
- `plot_energy_reservoirs.py` plots the six energy reservoirs (`W+`, `W-`, `W_par^u`, `W_par^B`, `W^p`, `W_s`) of an inhomogeneous run against time, and their shares of the total, e.g. `python vis/plot_energy_reservoirs.py RUN_DIRECTORY`. `plot_budget.py CSV --quantity w_par_u` shows one reservoir's budget term by term
- `plot_energy_budgets.py` plots the budget of each reservoir against time, one panel per reservoir: every saved term, the measured `d_t W` and the residual (which should sit at zero). `--average N` merges every N output intervals to smooth noisy forcing, the projection term is not drawn by default (`--hide` sets which terms are left off; the residual still counts them), and `--separate` also writes `budget_<reservoir>.png` with `W(t)` above its budget, e.g. `python vis/plot_energy_budgets.py RUN_DIRECTORY --average 5 --separate`
- `plot_energy_terms.py` shows how much each budget term contributes to each reservoir: one bar per term, averaged over a time window, labelled with its share of the reservoir's gains or losses, with block-average error bars (`--blocks`) and the budget residual in each title; `--hide` leaves terms off; it also writes `energy_terms.csv`, e.g. `python vis/plot_energy_terms.py RUN_DIRECTORY --tmin 5`
- `plot_stationarity.py` checks whether a run has reached a statistically steady state over a window (bursts are fine; the mean must not drift). For each quantity (default `total_energy`, `w_plus`, `w_minus`, `drho_rms`; `--quantities` picks any CSV columns) it compares the means of the two halves of the window, in units of their error. The error uses the correlation time `tau`: the window holds about `T/(2 tau)` independent samples. Below 2 sigma counts as steady, but with few independent samples even a large drift passes, so look at `N indep` too. It also shows the window-averaged total-energy budget, whose net should be a small fraction of the dissipation. It writes `stationarity.csv`, e.g. `python vis/plot_stationarity.py RUN_DIRECTORY --tmin 200`
- `plot_energy_flux.py` tests for a cascade. From `energy_flux.csv`, averaged over a (steady) window, it plots the nonlinear energy flux `Pi(K)` divided by the injection rate `eps` (linear drives plus forcing). A cascade shows up as `Pi/eps` of about 1 and flat over a range of `K`; `Pi` falling to about 0 just above injection means the energy is dissipated where it is put in. Fluxes are shown for the total energy, `z+`, `z-` and the compressive energy. A second panel shows the total-energy budget below `K` (injected = crosses `K` + dissipated, in a steady state). Error bands come from block averages (`--blocks`), and the numbers go to `energy_flux_summary.csv`, e.g. `python vis/plot_energy_flux.py RUN_DIRECTORY --tmin 200`
- `plot_z_rms_terms.py` measures how big each term of the local-in-z budgets is (`W+(z)`, `W-(z)`, the slow-mode energies and `W_s(z)`), as its rms over `z` next to its box average. The transport terms `vA d_z W±` average to zero over the box, so their rms is the only measure of their size. Reads `z_profiles.csv` and writes `z_rms_terms.csv`, e.g. `python vis/plot_z_rms_terms.py RUN_DIRECTORY --tmin 5`
- `plot_dcf_ccr_acr.py` plots the three heating channels of the paper against time from the interval-averaged budget columns: `Q_DCF = -(g/2)<drho z+_x>` (buoyancy draining z+), `Q_CCR = -w_plus_rhs_dissipation` (the z+ cascade/dissipation, with the reflection rate `(g/2)<drho z-_x>` that creates z- drawn dashed) and `Q_ACR = Y_B + Y_g - (g - vA^2 K_b0) V_psi_x = g (V_rho_x - V_psi_x)` (in this box `Y_U` and `p div(V_psi)` vanish, but the pressure work `V_psi . grad p` does not when `K_p0 != 0`, and is drawn alongside `Y_B` and `Y_g`; it reads `g`, `vA` and `K_b0` from `input_copy.input`; the thermal part `Y_th` is not heating and is not drawn; forcing is not drawn either). Each channel's window mean is in the legend and a CSV; `--average N` smooths, e.g. `python vis/plot_dcf_ccr_acr.py RUN_DIRECTORY --tmin 100 --average 10`
- `plot_local_budget_signs.py` shows the local budgets of `w_plus`, `w_minus`, `w_par_u`, `w_par_b` and `w_s`, one figure each (`--energy` picks others; `all` adds `w_p`): one panel per term showing, at each saved time, what the term adds and removes: `<max(T,0)>_z` (gain), `<min(T,0)>_z` (loss) and their sum (net, the box-averaged term). Terms that are exactly zero get no row, e.g. `python vis/plot_local_budget_signs.py RUN_DIRECTORY --tmin 100 --tmax 150`
- `plot_z_profiles.py` draws space-time maps `(t, z)` of the `z_profiles.csv` profiles, with lines `z = z0 ± vA t` on `w_plus`/`w_minus` to show parallel transport; `--subtract-mean` keeps only the z-structure, e.g. `python vis/plot_z_profiles.py RUN_DIRECTORY --quantities w_plus w_minus slow_flux`
- `plot_kpar0_maps.py` shows where the `drho` energy sits in the `(kx, ky)` plane at one snapshot (`--index`, default the last), for `k_par = 0` and for `k_par != 0`, e.g. `python vis/plot_kpar0_maps.py RUN_DIRECTORY`
- `plot_kx0_energy.py` compares the energy at `kx = 0` (no variation along `x`, the direction of the background gradients) with the energy at `kx != 0`, from the `E_kx0_*` / `E_kxn0_*` columns, e.g. `python vis/plot_kx0_energy.py RUN_DIRECTORY`
- `plot_ky0_energy.py` compares the energy at `ky = 0` with the energy at `ky != 0` against time, counting every `k_par` (so a forced `ky = 0` wave counts as `ky = 0`): the share of the total energy in each, and each field's share at `ky = 0`. It reads the `E_ky0_*` / `E_kyn0_*` columns of `scalar_diagnostics.csv`; runs made before those columns existed are summed from their `fullfields/` snapshots instead (`--every N` uses every Nth one), e.g. `python vis/plot_ky0_energy.py RUN_DIRECTORY`
- `plot_kperp0_energy.py` does the same for `k_perp = 0` (`kx = ky = 0`, the part of each field that depends on `z` only) against `k_perp != 0`, from the `E_kperp0_*` / `E_kperpn0_*` columns. `z+` and `z-` have no energy at `k_perp = 0`, so the per-field panel shows `du_par`, `db_par` and `drho`, e.g. `python vis/plot_kperp0_energy.py RUN_DIRECTORY`
- `run_quantities.py` is not a plot: it is the one place the inhomogeneous RMHD plots get their quantities from. It reads a run's saved CSV and input and returns each measured quantity next to its closure estimate (Eqs. 47, 69, 73, 75, 80, 85 of the paper), with `z+_x = rms(z+_x)` in place of the paper's isotropy assumption and `eta_turb = l_perp z+_x^2 / (2 z+_rms)`. Use it in Spyder with `run = load_run("RUN_DIRECTORY")`, then e.g. `run.q_dcf` and `run.q_dcf_predicted`. It also reads CSVs written before 2026-10-08, when the `k_perp_plus`/`k_prl_plus` columns were called `w_plus_kperp`/`w_plus_kprl`
- most driver, plotting, profiling, and example scripts support `--help` to print available options

Example budget check:

```bash
python -m rmhdgpu.run examples/decay_spectra.input --tmax 0.2
python vis/plot_budget.py examples/outputs/scalar_diagnostics.csv
```

## Tests and Profiling

CPU-focused tests live under [`rmhdgpu/tests`](rmhdgpu/tests). GPU-focused tests skip cleanly when CuPy or a usable CUDA device is unavailable.

Typical commands:

```bash
python -m pytest
python -m pytest rmhdgpu/tests/test_cupy_backend.py rmhdgpu/tests/test_gpu_consistency.py rmhdgpu/tests/test_gpu_runtime_checks.py rmhdgpu/tests/test_gpu_benchmarks.py
```

Profiling utilities live under [`rmhdgpu/profiling`](rmhdgpu/profiling):

- [`benchmark_backends.py`](rmhdgpu/profiling/benchmark_backends.py)
- [`profile_timestep.py`](rmhdgpu/profiling/profile_timestep.py)
- [`gpu_sanity.py`](rmhdgpu/profiling/gpu_sanity.py)

Typical commands:

```bash
python -m rmhdgpu.profiling.benchmark_backends --backend numpy --backend scipy_cpu --backend cupy --nx 64 --nx 96 --steps 10
python -m rmhdgpu.profiling.profile_timestep --backend cupy --nx 64 --repeats 2
python -m rmhdgpu.profiling.gpu_sanity --nx 32 --steps 6
```

The `sanity_*` scripts under [`rmhdgpu/examples`](rmhdgpu/examples) are kept only as lightweight developer checks. They are not the recommended workflow for normal runs.

## Running on Aoraki GPUs

For this project, the most reliable setup on Aoraki is a user-local Conda environment. This avoids relying on the cluster system Python and makes the setup reproducible for students.

Important:

Do not use `module load python` for this project.

That can override the Conda environment and leave you running `/opt/spack/.../python` even after activation.

### Create the environment

Any Python 3.10+ Conda environment is fine. The example below uses Python 3.10,
which is the minimum supported version.

It's best to setup the environment from a compute node on Aoraki.

```bash
srun --partition=aoraki_gpu_H100 --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=02:00:00 --pty bash
mkdir -p ~/conda-envs
conda create -y -p ~/conda-envs/curmpy python=3.10
conda activate ~/conda-envs/curmpy
python -m pip install --upgrade pip
python -m pip install numpy scipy matplotlib pytest h5py cupy imageio-ffmpeg
cd ~/path/to/rmhd-gpu
python -m pip install -e .
```

Check that the environment is really active:

```bash
which python
python -V
python -c "import sys; print(sys.executable)"
python -c "import numpy, scipy, matplotlib, h5py, cupy; print('Environment OK')"
```

The Python path should point to something like:

```text
/home/<username>/conda-envs/curmpy/bin/python
```

not `/opt/spack/...`.

### Optional activation helper

```bash
mkdir -p ~/bin
cat > ~/bin/activate-curmpy <<'EOF'
#!/usr/bin/env bash
source ~/.bashrc
export PYTHONNOUSERSITE=1
conda activate ~/conda-envs/curmpy
EOF
chmod +x ~/bin/activate-curmpy
```

Optional alias in `~/.bashrc`:

```bash
alias activate-curmpy="source ~/bin/activate-curmpy"
```

### Interactive workflow

Load an explicit CUDA 12.x module rather than the cluster default if the
default is newer than the installed NVIDIA driver supports. On the current
Aoraki H100 nodes checked for this repository, the default `cuda` module points
to CUDA 13.1 while the driver reports CUDA 12.5 support, so use an available
CUDA 12.x module such as `cuda/12.5.1-b6iqzzi`. Use `module avail cuda` and
`nvidia-smi` to check the current names and driver support.

```bash
srun --partition=aoraki_gpu_H100 --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=02:00:00 --pty bash
module purge
module load cuda/12.5.1-b6iqzzi
activate-curmpy
cd ~/cases/my_forced_case
python -m pytest ~/path/to/rmhd-gpu/rmhdgpu/tests/test_cupy_backend.py ~/path/to/rmhd-gpu/rmhdgpu/tests/test_gpu_consistency.py
python -m rmhdgpu.run input.input --backend cupy --tmax 1.0
```

If your account uses a different GPU partition, replace `aoraki_gpu_H100` with the appropriate one.

### Batch / Slurm jobs

Minimal job script:

```bash
#!/bin/bash
#SBATCH --job-name=rmhdgpu
#SBATCH --partition=aoraki_gpu_H100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

module purge
module load cuda/12.5.1-b6iqzzi
source ~/.bashrc
export PYTHONNOUSERSITE=1
conda activate ~/conda-envs/curmpy

cd ~/cases/my_forced_case
python -m rmhdgpu.run input.input --backend cupy
```

Make sure you're loading the correct version of cuda, if the default does not work.
Submit with:

```bash
sbatch run_rmhdgpu.slurm
```

### Quick diagnostics

If something seems wrong:

```bash
module list
nvidia-smi
which python
python -V
python -c "import sys; print(sys.executable)"
python -c "import numpy, cupy; print(numpy.__version__, cupy.__version__)"
python -c "import cupy; cupy.show_config()"
```

If `which python` points to `/opt/spack/...`, the wrong Python is active.

### Codex on Aoraki

Codex CLI can be run from an interactive Aoraki session like any other terminal tool. Run it only after the environment is activated so imports, CuPy detection, and tests match the actual cluster environment.
