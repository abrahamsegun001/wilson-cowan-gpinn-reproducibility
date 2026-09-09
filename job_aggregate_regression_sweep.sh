#!/bin/bash
#SBATCH --job-name=wc_reg_agg
#SBATCH --account=def-uanazodo-ab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/wc_reg_agg_%j.out
#SBATCH --error=logs/wc_reg_agg_%j.err

module purge
module load StdEnv/2023 python/3.10 gcc
source ~/gpinn_env/bin/activate

cd ~/scratch/gpinn_project/synthetic_identifiability_package

python aggregate_pseudoderiv_regression_sweep.py \
  --input_dir synthetic_outputs/regression_sensitivity
