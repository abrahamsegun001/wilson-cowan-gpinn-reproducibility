#!/bin/bash
#SBATCH --job-name=wc_cont_ab
#SBATCH --account=def-uanazodo-ab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=a100:1
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --array=0-19%4
#SBATCH --output=logs/wc_cont_ab_%A_%a.out
#SBATCH --error=logs/wc_cont_ab_%A_%a.err

module purge
module load StdEnv/2023 python/3.10 gcc cuda
source ~/gpinn_env/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/scratch/gpinn_project
mkdir -p logs outputs/continuity_ablation/{runs,histories,checkpoints}

python continuity_ablation.py \
    --task_id "$SLURM_ARRAY_TASK_ID" \
    --data_dir "$HOME/scratch/gpinn_project/data/processed" \
    --sequential_dir "./outputs/final_dual_model_sequential20" \
    --output_dir "./outputs/continuity_ablation" \
    --subject_prefix "sub-GC48_ses-day0" \
    --num_chunks_for_scaling 20 \
    --fps 30 \
    --tau 1.0 \
    --l_fov 10.0 \
    --chunk_duration 30.0 \
    --nested_folds 5 \
    --bootstrap_reps 1000 \
    --bootstrap_block_frames 30 \
    --n_data_pool 60000 \
    --n_val 15000 \
    --n_colloc_pool 60000 \
    --n_deriv_pool 20000 \
    --stageA_steps 2000 \
    --stageB_steps 3000 \
    --lbfgs_steps 150 \
    --lambda_physics 0.05 \
    --lambda_deriv 0.001 \
    --mu_grad 1e-5 \
    --seed 42
