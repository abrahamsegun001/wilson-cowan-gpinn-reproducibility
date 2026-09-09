# Synthetic Wilson-Cowan gPINN identifiability / twin experiment — Narval package

This package is built directly around the architecture and training logic in the supplied
`train_gpinn_narval_updated.py`. The synthetic **forward** data are generated independently
using a finite-difference + RK4 solver, while the **inverse** stage preserves the reported
gPINN architecture, residual scaling, parameter initialization, loss construction, dynamic
lambda rule, and Adam -> L-BFGS training schedule.

## Narval configuration

The SLURM files now match the working real-data submission script:

- account: `rpp-uanazodo`
- environment: `StdEnv/2023`, `python/3.10`, `gcc`, `cuda`
- virtual environment: `~/gpinn_env`
- GPU inverse jobs: one A100, 8 CPUs, 48 GB RAM, 24 h
- project root: `~/scratch/gpinn_project`

Place this directory at:

```bash
~/scratch/gpinn_project/synthetic_identifiability_package
```

## Experimental design

Default ground truth:

- `D_E = 0.60 mm^2`
- `w_EE = 5.20`
- `h = -2.60`
- `tau = 1.0 s`
- field of view = `10 mm`
- grid = `128 x 128`
- duration = `30 s`
- saved sampling rate = `30 Hz` (`900` frames)

The forward solver uses a 5-point finite-difference Laplacian, homogeneous Neumann/no-flux
boundaries, and explicit RK4 time integration. Synthetic E is already generated on the model
scale `[0,1]`; it is intentionally **not** min-max normalized again before inversion.

## Replicates

The SLURM array contains 10 inverse jobs:

- 5 trainable-network seeds: `42 123 456 789 2026`
- 2 observation-noise levels: `0.00` and `0.01`

For a given noise level, the synthetic field, noisy realization, sampled train/validation points,
collocation points, and Fourier projection matrices are held fixed. Only trainable network
initialization / optimization seed changes. This isolates practical inverse-training variability.

## Training budget

Each inverse run uses:

- 60,000 supervised data points
- 90,000 collocation points
- 5,000 Adam epochs at `lr=1e-3`
- dynamic physics weight initialized at `0.01`
- gradient penalty `mu_grad=1e-5`
- up to 500 L-BFGS iterations

## Upload / placement on Narval

From your local machine, place the unzipped directory under your existing project directory so that:

```bash
~/scratch/gpinn_project/synthetic_identifiability_package/
```

contains the `.py` and `.sh` files.

On Narval:

```bash
cd ~/scratch/gpinn_project/synthetic_identifiability_package
chmod +x *.sh *.py
bash submit_synthetic_pipeline.sh
```

The dependency chain is:

1. `job_generate_synthetic.sh` — independent forward simulation (CPU)
2. `job_train_synthetic_array.sh` — 10 A100 inverse jobs (array, max 2 concurrent)
3. `job_plot_synthetic.sh` — aggregation, recovery figures and identifiability landscapes (CPU)

The plotting job starts only if all inverse array tasks complete successfully.

## Monitoring

```bash
squeue -u $USER
```

Useful log commands:

```bash
tail -f logs/wc_syn_gen_<JOBID>.out
tail -f logs/wc_syn_inv_<ARRAYJOBID>_0.out
tail -f logs/wc_syn_plot_<JOBID>.out
```

## Main outputs

Under `synthetic_outputs/`:

- `results/*.json` — one result per seed/noise condition
- `histories/*.npz` — training histories
- `checkpoints/*.pt` — recovered models
- `synthetic_recovery_runs.csv` — run-level recovery table
- `synthetic_recovery_summary.csv` — mean, SD, CV and relative error by noise level
- `figures/recovery_D_E.*`
- `figures/recovery_w_EE.*`
- `figures/recovery_h.*`
- pairwise PDE-residual landscapes: `D_E-w_EE`, `D_E-h`, `w_EE-h`
- true, reconstructed and absolute-error spatial frames for the best clean run

## Interpretation

The clean experiment asks whether the inverse gPINN can recover known parameters when the forward
model is correct. The noisy experiment tests practical robustness. High field-reconstruction R^2
combined with inaccurate or seed-variable parameter estimates would indicate practical
non-identifiability / parameter compensation rather than successful physical parameter recovery.
