#!/bin/bash
#SBATCH --job-name=wc_gpinn_final
#SBATCH --account=def-uanazodo-ab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=a100:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/wc_gpinn_final_%j.out
#SBATCH --error=logs/wc_gpinn_final_%j.err

module purge
module load StdEnv/2023 python/3.10 gcc cuda
source ~/gpinn_env/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/scratch/gpinn_project
mkdir -p logs outputs/final_gpinn_pipeline_cont005

python train_gpinn_narval_final.py \
    --data_dir "$HOME/scratch/gpinn_project/data/processed" \
    --output_dir "./outputs/final_gpinn_pipeline_cont005" \
    --subject_prefix "sub-GC48_ses-day0" \
    --chunks 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
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
    --lambda_cont 0.05 \
    --continuity_batch 2048 \
    --seed 42
