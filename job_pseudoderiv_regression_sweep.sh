#!/bin/bash
#SBATCH --job-name=wc_reg_sweep
#SBATCH --account=def-uanazodo-ab
#SBATCH --array=0-15%4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/wc_reg_sweep_%A_%a.out
#SBATCH --error=logs/wc_reg_sweep_%A_%a.err

module purge
module load StdEnv/2023 python/3.10 gcc
source ~/gpinn_env/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd ~/scratch/gpinn_project/synthetic_identifiability_package
mkdir -p logs synthetic_outputs/regression_sensitivity

python sweep_pseudoderiv_regression.py \
  --data synthetic_data/wc_twin_truth.npz \
  --output_dir synthetic_outputs/regression_sensitivity \
  --task_id "$SLURM_ARRAY_TASK_ID" \
  --n_points 24000
