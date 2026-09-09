#!/bin/bash
#SBATCH --job-name=wc_seq20_reg
#SBATCH --account=def-uanazodo-ab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=a100:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/wc_seq20_reg_%j.out
#SBATCH --error=logs/wc_seq20_reg_%j.err

module purge
module load StdEnv/2023 python/3.10 gcc cuda
source ~/gpinn_env/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/scratch/gpinn_project
mkdir -p logs outputs/final_dual_model_sequential20

python train_dual_model_sequential20.py \
    --data_dir "$HOME/scratch/gpinn_project/data/processed" \
    --output_dir "./outputs/final_dual_model_sequential20" \
    --subject_prefix "sub-GC48_ses-day0" \
    --chunks 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
    --num_chunks_for_scaling 20 \
    --old_results "./outputs/temporal_tracking_results.npz" \
    --independent_results_csv "./outputs/final_dual_model_validation_randomsplit/five_chunk_validation_results.csv" \
    --fps 30 \
    --tau 1.0 \
    --l_fov 10.0 \
    --chunk_duration 30.0 \
    --n_val 15000 \
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
