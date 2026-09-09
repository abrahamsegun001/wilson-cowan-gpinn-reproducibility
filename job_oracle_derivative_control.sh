#!/bin/bash
#SBATCH --job-name=wc_oracle_ctrl
#SBATCH --account=def-uanazodo-ab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=a100:1
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/wc_oracle_ctrl_%j.out
#SBATCH --error=logs/wc_oracle_ctrl_%j.err

module purge
module load StdEnv/2023 python/3.10 gcc cuda
source ~/gpinn_env/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd ~/scratch/gpinn_project/synthetic_identifiability_package
mkdir -p logs synthetic_outputs/oracle_derivative_control

python train_synthetic_oracle_derivative.py \
  --data synthetic_data/wc_twin_truth.npz \
  --output_dir synthetic_outputs/oracle_derivative_control \
  --seed 42 \
  --fourier_seed 2718 \
  --state_pool 60000 \
  --deriv_pool 24000 \
  --ic_pool 8192 \
  --state_batch 4096 \
  --deriv_batch 512 \
  --ic_batch 2048 \
  --stageA_steps 2500 \
  --stageB_steps 2500 \
  --stageC_adam_steps 1800 \
  --stageD_steps 1200
