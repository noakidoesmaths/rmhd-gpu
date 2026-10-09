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
- `t_out_full`: full-field cadence

Cadences are in simulation time units:

- positive value: enable that output category
- `0` or negative: disable that output category

Outputs are:

- `scalar_diagnostics.csv`: one row per output time, with `time`, `step`, scalar quantities, and saved budget terms
- `spectra.csv`: tidy/long CSV with columns `time`, `step`, `quantity`, `kperp`, `value`
- `fullfields/fullfield_0001.h5`, `fullfield_0002.h5`, ...: one HDF5 snapshot per output time

Scalar diagnostics are split into generic per-field columns such as
`psi_rms`, `psi_mean`, and `psi_max_abs`, plus equation-specific quantities
defined by the selected equation module.

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
- `[runtime]`: `runtime_check_every`, `progress_output_every`, `fail_on_nonfinite`, `dealias`, `dealias_mode`
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

Controlled forcing rescales one branch in a perpendicular Fourier band at a
selected nonzero parallel harmonic. It supports the `alfvenic`, `s09`, and
`low_beta_stratified` equations. A plus-only control changes `phi - psi` while
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
only the selected parallel harmonic receives forcing. Use `branch_total`,
`branch_nonzero_kz`, or `shell` for the other measurement scopes.

Controlled branch energy is `mean(|z±|²)/4`, half the existing stochastic
Elsasser-energy convention. Controlled constant-power values therefore need
that factor of two when compared with stochastic `epsilon_plus/minus`.

Runnable examples:

```bash
python -m rmhdgpu.run examples/controlled_target_s09.input
python -m rmhdgpu.run examples/controlled_power_alfvenic.input
```

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
- `plot_cross_helicity.py` plots `sigma_c = (W+ - W-)/(W+ + W-)` for the inhomogeneous RMHD sets; pass several run directories to overlay them (`sigma_c = 1` is pure `z+`)
- `plot_flux_closure.py` plots the density flux `V_rho_x` and its closure `eta_turb F_rho`, and `rms(drho)` with its Eq. (47) estimate, against `chi_A`, one window-averaged point per run, e.g. `python vis/plot_flux_closure.py "examples/outputs_*_rho" --tmin 2 --tmax 10`
- `run_quantities.py` is not a plot: it defines the derived quantities the inhomogeneous RMHD plots share (`z±`, `l_perp`, `chi_A`, `eta_turb`, the slaved-closure estimates) from a run's saved CSV and input. Use it in Spyder with `run = load_run("RUN_DIRECTORY")`, then `run.chi_a`, `run.l_perp`, and so on. It also reads CSVs written before 2026-10-08, when the `k_perp_plus`/`k_prl_plus` columns were called `w_plus_kperp`/`w_plus_kprl`
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
python -m pip install numpy scipy matplotlib pytest h5py cupy
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
