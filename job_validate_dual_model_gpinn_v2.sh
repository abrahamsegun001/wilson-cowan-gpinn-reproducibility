#!/bin/bash
#SBATCH --job-name=wc_dual_val5v2
#SBATCH --account=def-uanazodo-ab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=a100:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/wc_dual_val5v2_%j.out
#SBATCH --error=logs/wc_dual_val5v2_%j.err

module purge
module load StdEnv/2023 python/3.10 gcc cuda
source ~/gpinn_env/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/scratch/gpinn_project
mkdir -p logs outputs/final_dual_model_validation

python validate_dual_model_regression_assisted_gpinn_v2.py \
    --data_dir "$HOME/scratch/gpinn_project/data/processed" \
    --output_dir "./outputs/final_dual_model_validation" \
    --subject_prefix "sub-GC48_ses-day0" \
    --chunks 0 5 10 15 19 \
    --num_chunks_for_scaling 20 \
    --old_results "./outputs/temporal_tracking_results.npz" \
    --fps 30 \
    --tau 1.0 \
    --l_fov 10.0 \
    --chunk_duration 30.0 \
    --final_holdout_fraction 0.20 \
    --embargo_frames 6 \
    --nested_folds 5 \
    --bootstrap_reps 1000 \
    --bootstrap_block_frames 30 \
    --min_valid_fraction 0.90 \
    --max_standardized_condition 10.0 \
    --min_logit_improvement_pct 0.10 \
    --min_pde_improvement_pct 0.10 \
    --min_positive_fold_fraction 0.60 \
    --max_smoothing_relative_range 1.00 \
    --n_data_pool 60000 \
    --n_colloc_pool 60000 \
    --n_deriv_pool 20000 \
    --data_batch 4096 \
    --colloc_batch 4096 \
    --deriv_batch 1024 \
    --stageA_steps 2000 \
    --stageB_steps 3000 \
    --lr_stageA 0.001 \
    --lr_stageB 0.0002 \
    --lbfgs_steps 150 \
    --lambda_physics 0.05 \
    --lambda_deriv 0.001 \
    --mu_grad 1e-5 \
    --seed 42
