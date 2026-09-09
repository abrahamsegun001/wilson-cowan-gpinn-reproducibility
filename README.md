# Regression-assisted, model-selective sequential Wilson-Cowan gPINN

Reproducibility package for the manuscript **Regression-Assisted, Model-Selective Gradient-Enhanced Physics-Informed Neural Networks for Interpretable Sequential Wilson-Cowan Inference from Widefield Calcium Imaging**.

## What is in this repository

The canonical production implementation is:

```text
train_gpinn_narval_final.py
```

For each 30-second calcium-imaging chunk it estimates pseudo-derivatives, fits full and nested reduced Wilson-Cowan regressions, applies the pre-specified model-selection criteria, fixes the selected physical coefficients, trains the neural-state gPINN, and sequentially warm-starts only the neural representation. The physical coefficients and optimizer state are re-estimated/reset for each chunk. Continuity regularization uses `lambda_cont=0.05` from chunk 1 onward.

The package also contains the synthetic twin experiments, derivative diagnostics, smoothing calibration, selector controls, real-data audits, continuity ablation, publication tables/figures, SLURM scripts, and static manuscript figures.

## Repository map

```text
train_gpinn_narval_final.py        canonical final real-data trainer
job_gpinn_narval_final_cont005.sh  final 20-chunk Narval job
job_gpinn_narval_final_no_continuity.sh  matched no-continuity run
synthetic_identifiability_package/ baseline synthetic twin experiment
diagnose_derivatives.py            learned derivative diagnostic
train_synthetic_*                  synthetic recovery controls
sweep_pseudoderiv_regression.py    noise/smoothing sweep
synthetic_model_selection_control.py final selector positive/null control
real_calcium_*                     real-data regression/uncertainty audits
continuity_ablation.py             continuity-weight ablation
extract_publication_results_v2.py  publication master-data extraction
make_publication_*                 tables and figures
reproduce_publication_assets.sh    rebuild publication assets from completed runs
figures/                            static manuscript figure snapshot
data/README.md                      processed-data interface
docs/                               detailed reproducibility notes
```

## Environment

The production jobs used Python 3.10 and a CUDA-enabled PyTorch environment. Create a basic environment with either:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/check_environment.py
```

or:

```bash
conda env create -f environment.yml
conda activate wc-gpinn
```

For GPU training, install the PyTorch build appropriate for the CUDA stack on your system.

## Data

The model expects 20 processed NPZ chunks under `data/processed/`:

```text
sub-GC48_ses-day0_chunk-00_dfof.npz
...
sub-GC48_ses-day0_chunk-19_dfof.npz
```

Each contains `dfof` with axes `[time,y,x]`; the manuscript working shape is `900 x 64 x 64`. Validate the files before training:

```bash
python scripts/validate_processed_data.py --data_dir data/processed
```

The source recording is BraiDyn-BC, DANDI accession 001425, session `sub-VG1-GC#48_ses-2023-08-18-resting-state-day0_widefield+behavior.nwb`. See `docs/DATA_AND_PREPROCESSING.md` for the exact preprocessing specification and the explicit raw-to-processed code availability note.

## Run the final pipeline

A quick one-chunk execution is:

```bash
python train_gpinn_narval_final.py \
  --data_dir data/processed \
  --output_dir outputs/smoke_real \
  --chunks 0 \
  --num_chunks_for_scaling 20 \
  --seed 42
```

For the full manuscript configuration on Narval:

```bash
sbatch job_gpinn_narval_final_cont005.sh
```

Primary final outputs are:

```text
final_pipeline_results.csv
final_pipeline_summary.json
checkpoints/
histories/
model_selection_details/
```

The final training configuration uses 60,000 state-training points, 15,000 random-pointwise validation points, 60,000 collocation points, 20,000 pseudo-derivative points, Stage A 2000 Adam steps, Stage B 3000 Adam steps, and 150 L-BFGS iterations. Validation is spatiotemporal interpolation, not forecasting.

## Synthetic twin experiment

The independent finite-difference + RK4 synthetic experiment is contained in `synthetic_identifiability_package/`. On Narval:

```bash
cd synthetic_identifiability_package
bash submit_synthetic_pipeline.sh
```

The known synthetic truth is `D_E=0.60`, `w_EE=5.20`, `h=-2.60`. Supporting control scripts at repository root reproduce the derivative, oracle, staged, pseudo-derivative, smoothing, and selector analyses used in the manuscript.

## Publication assets

After the required final, no-continuity, continuity-ablation, and synthetic outputs exist, rebuild tables and figures with:

```bash
bash reproduce_publication_assets.sh
```

See `docs/MANUSCRIPT_TO_CODE_MAP.md` for the source of each manuscript analysis.

## Reproducibility boundary

The canonical final model is self-contained **from the processed NPZ chunks onward**. The exact original raw-NWB extraction/affine-resampling script was not present among the source files supplied when this archive was assembled, so it has not been fabricated. The preprocessing algorithm is documented in `docs/DATA_AND_PREPROCESSING.md`, and the missing raw-to-chunk implementation is the only known code-stage gap in the assembled release.

## Scientific interpretation

`D_E` is an effective Laplacian spatial-coupling coefficient in the reduced Wilson-Cowan formulation, not a directly calibrated biological tissue diffusion constant. A reduced-model selection indicates lack of stable support for the isotropic Laplacian term under the stated data, scales, preprocessing, and selection criteria; it does not establish absence of cortical spatial coupling.

## License

No software license has been assigned in this archive. Before making the GitHub repository public, choose and add an appropriate software license so that downstream reuse terms are explicit.
