#!/bin/bash
#SBATCH --job-name=wc_D_bound_audit
#SBATCH --account=def-uanazodo-ab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/wc_D_bound_audit_%j.out
#SBATCH --error=logs/wc_D_bound_audit_%j.err

module purge
module load StdEnv/2023 python/3.10 gcc
source ~/gpinn_env/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

cd ~/scratch/gpinn_project
mkdir -p logs outputs/regression_bound_audit

python real_calcium_D_bound_audit.py \
    --data_dir "$HOME/scratch/gpinn_project/data/processed" \
    --output_dir "./outputs/regression_bound_audit" \
    --num_chunks 20 \
    --subject_prefix "sub-GC48_ses-day0" \
    --fps 30 \
    --tau 1.0 \
    --l_fov 10.0 \
    --n_points 90000 \
    --seed 42
