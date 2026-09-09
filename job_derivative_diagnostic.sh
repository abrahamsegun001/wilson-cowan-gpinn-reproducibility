#!/bin/bash
#SBATCH --job-name=wc_deriv_diag
#SBATCH --account=def-uanazodo-ab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=a100:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/wc_deriv_diag_%j.out
#SBATCH --error=logs/wc_deriv_diag_%j.err

module purge
module load StdEnv/2023 python/3.10 gcc cuda
source ~/gpinn_env/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd ~/scratch/gpinn_project/synthetic_identifiability_package
mkdir -p logs synthetic_outputs/diagnostics

python diagnose_derivatives.py \
  --data synthetic_data/wc_twin_truth.npz \
  --checkpoint synthetic_outputs/checkpoints/seed-42_noise-0p000.pt \
  --output_dir synthetic_outputs/diagnostics \
  --n_samples 20000 \
  --batch_size 1024
