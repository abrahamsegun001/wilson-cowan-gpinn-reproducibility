#!/bin/bash
#SBATCH --job-name=wc_nested_uncert
#SBATCH --account=def-uanazodo-ab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/wc_nested_uncert_%j.out
#SBATCH --error=logs/wc_nested_uncert_%j.err

module purge
module load StdEnv/2023 python/3.10 gcc
source ~/gpinn_env/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

cd ~/scratch/gpinn_project
mkdir -p logs outputs/nested_uncertainty_audit

python real_calcium_nested_uncertainty_audit.py \
    --data_dir "$HOME/scratch/gpinn_project/data/processed" \
    --output_dir "./outputs/nested_uncertainty_audit" \
    --num_chunks 20 \
    --subject_prefix "sub-GC48_ses-day0" \
    --fps 30 \
    --tau 1.0 \
    --l_fov 10.0 \
    --n_folds 5 \
    --embargo_frames 6 \
    --bootstrap_reps 1000 \
    --bootstrap_block_frames 30 \
    --seed 42
