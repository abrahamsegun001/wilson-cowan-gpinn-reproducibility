#!/usr/bin/env bash
set -euo pipefail
# Compact Narval/SLURM launcher. Edit config.yaml and export SLURM_ACCOUNT first.
# Examples:
#   ./submit_pipeline.sh train
#   ./submit_pipeline.sh train-nocont
#   ./submit_pipeline.sh report
#   ./submit_pipeline.sh command diagnostics regression --output_dir outputs/regression_audit

MODE="${1:-}"; shift || true
ACCOUNT="${SLURM_ACCOUNT:-CHANGE_ME}"
ROOT="${PROJECT_ROOT:-$HOME/scratch/gpinn_project}"
VENV="${GPINN_VENV:-$HOME/gpinn_env}"

if [[ "${IN_WC_SLURM:-0}" != "1" ]]; then
  case "$MODE" in
    train|train-nocont)
      exec sbatch --account="$ACCOUNT" --time=24:00:00 --cpus-per-task=8 --mem=48G --gpus=a100:1 --export=ALL,IN_WC_SLURM=1 "$0" "$MODE" "$@" ;;
    report)
      exec sbatch --account="$ACCOUNT" --time=02:00:00 --cpus-per-task=4 --mem=16G --export=ALL,IN_WC_SLURM=1 "$0" "$MODE" "$@" ;;
    command)
      exec sbatch --account="$ACCOUNT" --time=08:00:00 --cpus-per-task=8 --mem=48G --gpus=a100:1 --export=ALL,IN_WC_SLURM=1 "$0" "$MODE" "$@" ;;
    *) echo "Usage: $0 {train|train-nocont|report|command <run_pipeline args>}"; exit 2;;
  esac
fi

cd "$ROOT"
mkdir -p logs outputs
module purge || true
module load StdEnv/2023 python/3.10 gcc || true
if [[ "$MODE" != "report" ]]; then module load cuda || true; fi
[[ -f "$VENV/bin/activate" ]] && source "$VENV/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" PYTHONUNBUFFERED=1

case "$MODE" in
  train) python run_pipeline.py train ;;
  train-nocont) python run_pipeline.py train --no-continuity ;;
  report) python run_pipeline.py report all ;;
  command) python run_pipeline.py "$@" ;;
esac
