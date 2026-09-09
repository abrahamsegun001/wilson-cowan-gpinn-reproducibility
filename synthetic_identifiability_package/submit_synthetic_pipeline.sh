#!/bin/bash
# Submit from the directory containing all package files.
# Optional: export GPINN_VENV="$HOME/path/to/the/same/venv/used-for-your-real-data-run"
set -euo pipefail
mkdir -p logs synthetic_data synthetic_outputs

GEN_ID=$(sbatch --parsable job_generate_synthetic.sh)
echo "Synthetic forward-generation job: ${GEN_ID}"

TRAIN_ID=$(sbatch --parsable --dependency=afterok:${GEN_ID} job_train_synthetic_array.sh)
echo "Synthetic inverse array job:      ${TRAIN_ID}"

PLOT_ID=$(sbatch --parsable --dependency=afterok:${TRAIN_ID} job_plot_synthetic.sh)
echo "Plot/aggregation job:             ${PLOT_ID}"

echo
echo "Monitor with: squeue -u $USER"
echo "Logs:         logs/"
echo "Results:      synthetic_outputs/results/"
echo "Summary:      synthetic_outputs/synthetic_recovery_summary.csv"
echo "Figures:      synthetic_outputs/figures/"
