#!/bin/bash
#SBATCH --job-name=wc_syn_gen
#SBATCH --account=rpp-uanazodo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/wc_syn_gen_%j.out
#SBATCH --error=logs/wc_syn_gen_%j.err

set -euo pipefail

module purge
module load StdEnv/2023 python/3.10 gcc cuda
source ~/gpinn_env/bin/activate

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

cd ~/scratch/gpinn_project/synthetic_identifiability_package
mkdir -p logs synthetic_data synthetic_outputs

python generate_synthetic_wc.py \
    --output "./synthetic_data/wc_twin_truth.npz" \
    --D_E 0.60 \
    --w_EE 5.20 \
    --h -2.60 \
    --tau 1.0 \
    --l_fov 10.0 \
    --nx 128 \
    --ny 128 \
    --duration 30.0 \
    --output_hz 30.0 \
    --substeps_per_frame 4
