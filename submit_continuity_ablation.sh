#!/bin/bash
set -euo pipefail
cd ~/scratch/gpinn_project
mkdir -p logs outputs/continuity_ablation/{runs,histories,checkpoints}

ARRAY_ID=$(sbatch --parsable job_continuity_ablation_array.sh)
echo "Continuity ablation array: ${ARRAY_ID}"

AGG_ID=$(sbatch --parsable --dependency=afterok:${ARRAY_ID} job_aggregate_continuity_ablation.sh)
echo "Aggregation job:          ${AGG_ID}"

echo
echo "Monitor: squeue -u $USER"
echo "Array logs: logs/wc_cont_ab_${ARRAY_ID}_<task>.out"
echo "Final summary: outputs/continuity_ablation/continuity_ablation_summary.json"
