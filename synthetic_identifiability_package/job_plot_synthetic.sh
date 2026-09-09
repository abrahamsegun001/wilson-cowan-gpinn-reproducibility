#!/bin/bash
#SBATCH --job-name=wc_syn_plot
#SBATCH --account=rpp-uanazodo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/wc_syn_plot_%j.out
#SBATCH --error=logs/wc_syn_plot_%j.err

set -euo pipefail

module purge
module load StdEnv/2023 python/3.10 gcc cuda
source ~/gpinn_env/bin/activate

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

cd ~/scratch/gpinn_project/synthetic_identifiability_package
mkdir -p logs synthetic_outputs/figures

python plot_synthetic_results.py \
    --output_dir "./synthetic_outputs" \
    --data "./synthetic_data/wc_twin_truth.npz"
