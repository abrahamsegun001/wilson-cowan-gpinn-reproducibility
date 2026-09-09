#!/bin/bash
#SBATCH --job-name=wc_syn_inv
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=a100:1
#SBATCH --array=0-9%2
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs synthetic_outputs

module purge
module load StdEnv/2023
module load python/3.11
module load cuda/12.2

# Point GPINN_VENV to the same environment you used for train_gpinn_narval_updated.py.
# On the login node, for example:
#   export GPINN_VENV="$HOME/venvs/gpinn"
if [[ -n "${GPINN_VENV:-}" && -f "${GPINN_VENV}/bin/activate" ]]; then
    source "${GPINN_VENV}/bin/activate"
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

SEEDS=(42 123 456 789 2026)
NOISES=(0.0 0.01)
NSEEDS=${#SEEDS[@]}
NOISE_INDEX=$((SLURM_ARRAY_TASK_ID / NSEEDS))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % NSEEDS))
SEED=${SEEDS[$SEED_INDEX]}
NOISE=${NOISES[$NOISE_INDEX]}

echo "Task=${SLURM_ARRAY_TASK_ID} seed=${SEED} noise_sigma=${NOISE}"
nvidia-smi || true

python train_synthetic_gpinn.py \
  --data synthetic_data/wc_twin_truth.npz \
  --output_dir synthetic_outputs \
  --seed "${SEED}" \
  --fourier_seed 2718 \
  --sampling_seed 31415 \
  --noise_seed 1618 \
  --noise_sigma "${NOISE}" \
  --n_data 60000 \
  --n_colloc 90000 \
  --adam_epochs 5000 \
  --lbfgs_steps 500 \
  --lambda_physics 0.01 \
  --lr 1e-3 \
  --mu_grad 1e-5
