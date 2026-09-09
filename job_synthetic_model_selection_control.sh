#!/bin/bash
#SBATCH --job-name=wc_syn_select
#SBATCH --account=def-uanazodo-ab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/wc_syn_select_%j.out
#SBATCH --error=logs/wc_syn_select_%j.err

set -euo pipefail

module purge
module load StdEnv/2023 python/3.10 gcc
source ~/gpinn_env/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

cd ~/scratch/gpinn_project
mkdir -p logs synthetic_data outputs/synthetic_model_selection_control

# Positive control should already exist from the completed synthetic twin study.
if [ ! -f synthetic_data/wc_twin_truth.npz ]; then
  python generate_synthetic_wc.py \
    --output synthetic_data/wc_twin_truth.npz \
    --D_E 0.60 --w_EE 5.20 --h -2.60 \
    --tau 1.0 --l_fov 10.0 --nx 128 --ny 128 \
    --duration 30 --output_hz 30 --substeps_per_frame 4
fi

# Null control: everything identical except D_E*=0.
python generate_synthetic_wc.py \
  --output synthetic_data/wc_twin_truth_D0.npz \
  --D_E 0.0 --w_EE 5.20 --h -2.60 \
  --tau 1.0 --l_fov 10.0 --nx 128 --ny 128 \
  --duration 30 --output_hz 30 --substeps_per_frame 4

# Apply the final selector unchanged to positive and null cases,
# both clean and with sigma=0.01 observation noise.
python synthetic_model_selection_control.py \
  --positive synthetic_data/wc_twin_truth.npz \
  --null synthetic_data/wc_twin_truth_D0.npz \
  --out_dir outputs/synthetic_model_selection_control \
  --noise_sigmas 0 0.01 \
  --noise_seed 42 \
  --nested_folds 5 \
  --bootstrap_reps 1000 \
  --bootstrap_block_frames 30 \
  --min_valid_fraction 0.90 \
  --max_standardized_condition 10 \
  --min_logit_improvement_pct 0.10 \
  --min_pde_improvement_pct 0.10 \
  --min_positive_fold_fraction 0.60 \
  --max_smoothing_relative_range 1.0 \
  --w_min 0.5 --w_max 6.0 \
  --seed 42

printf '\nSYNTHETIC MODEL-SELECTION CONTROL COMPLETE\n'
cat outputs/synthetic_model_selection_control/synthetic_model_selection_control.csv
