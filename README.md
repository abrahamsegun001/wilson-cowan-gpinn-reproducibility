# Regression-assisted, model-selective sequential Wilson-Cowan gPINN

Compact reproducibility repository for **Regression-Assisted, Model-Selective Gradient-Enhanced Physics-Informed Neural Networks for Interpretable Sequential Wilson-Cowan Inference from Widefield Calcium Imaging**.

Repository: **https://github.com/abrahamsegun001/wilson-cowan-gpinn-reproducibility**

The public interface has intentionally been reduced from the original development archive to one configuration file, one Python entry point, one SLURM entry point, and six focused source modules. Historical `v1/v2/final` scripts and dozens of one-off job files are not part of this release.

## Repository structure

```text
wilson-cowan-gpinn/
├── README.md
├── LICENSE
├── CITATION.cff
├── .gitignore
├── environment.yml
├── config.yaml
├── run_pipeline.py
├── submit_pipeline.sh
├── data/README.md
└── src/
    ├── __init__.py
    ├── training.py       # canonical final real-data trainer
    ├── model.py          # baseline synthetic gPINN core
    ├── synthetic.py      # synthetic twin + recovery/selector controls
    ├── diagnostics.py    # real-data audits, independent validation, continuity
    ├── reporting.py      # publication master data, tables, figures
    └── utils.py
```

## Data and preprocessing

Raw BraiDyn-BC data are available from DANDI accession **001425**. Exact raw-to-chunk preprocessing is maintained in the companion repository:

**https://github.com/abrahamsegun001/derivative-aware-widefield**

This modelling repository expects twenty processed files such as:

```text
data/processed/sub-GC48_ses-day0_chunk-00_dfof.npz
...
data/processed/sub-GC48_ses-day0_chunk-19_dfof.npz
```

Each contains `dfof[time,y,x]`; the manuscript configuration is `900 x 64 x 64` at 30 Hz.

## Installation

```bash
conda env create -f environment.yml
conda activate wc-gpinn
python run_pipeline.py verify
```

Install the PyTorch build appropriate for your CUDA stack when running GPU training.

## Main analysis

The canonical final method is `src/training.py`, consolidated from the validated final trainer. For each 30-s chunk it estimates resolution-aware pseudo-derivatives, fits full and nested reduced Wilson-Cowan regressions, applies model-selection criteria, fixes the selected physical coefficients, trains the neural-state gPINN, and warm-starts only the neural representation. Physical coefficients and optimizer state are reset/re-estimated for each chunk.

Run the manuscript configuration:

```bash
python run_pipeline.py train
```

Matched no-continuity run:

```bash
python run_pipeline.py train --no-continuity
```

Validate processed inputs:

```bash
python run_pipeline.py validate-data
```

## Synthetic controls

All former synthetic scripts are namespaced under one module:

```bash
python run_pipeline.py synthetic generate
python run_pipeline.py synthetic baseline --help
python run_pipeline.py synthetic oracle --help
python run_pipeline.py synthetic staged --help
python run_pipeline.py synthetic pseudoderiv --help
python run_pipeline.py synthetic sweep --help
python run_pipeline.py synthetic aggregate-sweep --help
python run_pipeline.py synthetic selector --help
```

Known synthetic truth in `config.yaml`: `D_E=0.60`, `w_EE=5.20`, `h=-2.60`, `tau=1 s`.

## Diagnostics and ablations

```bash
python run_pipeline.py diagnostics independent --help
python run_pipeline.py diagnostics continuity --help
python run_pipeline.py diagnostics aggregate-continuity --help
python run_pipeline.py diagnostics derivatives --help
python run_pipeline.py diagnostics d-bound --help
python run_pipeline.py diagnostics regression --help
python run_pipeline.py diagnostics uncertainty --help
```

`independent` is the random-pointwise cold/independent reference used for matched sequential comparisons. The resolution-aware regression audit supersedes the earlier fixed-pixel audit.

## Publication outputs

Once the completed training/control outputs are available:

```bash
python run_pipeline.py report all
```

This extracts one master numerical source, renders manuscript tables, generates Figures 1–5, and applies the final Figure 1/Figure 4/S1 layout. Figures and tables are generated under `outputs/` and are intentionally not committed.

## HPC / Narval

Set your account and project root, then use the single launcher:

```bash
export SLURM_ACCOUNT=your-account
export PROJECT_ROOT=$HOME/scratch/gpinn_project
./submit_pipeline.sh train
./submit_pipeline.sh report
```

Cluster-specific account/module details are deliberately kept out of the scientific source.

## Manuscript-to-command map

| Manuscript component | Compact command/source |
|---|---|
| Final sequential gPINN | `run_pipeline.py train` / `src/training.py` |
| Baseline synthetic twin | `synthetic generate`, `synthetic baseline`, `synthetic plot` |
| Learned derivative diagnostic | `diagnostics derivatives` |
| Oracle / staged / pseudo-derivative controls | `synthetic oracle`, `staged`, `pseudoderiv` |
| Noise/smoothing calibration | `synthetic sweep`, `aggregate-sweep` |
| Selector positive/null control | `synthetic selector` |
| Real lower-bound audit | `diagnostics d-bound` |
| Resolution-aware regression audit | `diagnostics regression` |
| Nested uncertainty audit | `diagnostics uncertainty` |
| Independent/cold matched reference | `diagnostics independent` |
| Continuity ablation | `diagnostics continuity`, `aggregate-continuity` |
| Tables and figures | `report all` |

## Scientific interpretation

`D_E` is an effective isotropic Laplacian spatial-coupling coefficient in the reduced Wilson-Cowan formulation, not a directly calibrated biological tissue diffusion constant. Selection of a reduced `D_E=0` model means that the Laplacian term lacks stable support under the stated data, preprocessing, scales, and selection criteria. It does not establish absence of cortical spatial coupling.

## License and citation

Released under the BSD 3-Clause License. If this software contributes to academic work, please cite the associated manuscript and repository; `CITATION.cff` is provided for GitHub citation support.
