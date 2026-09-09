#!/bin/bash
#SBATCH --job-name=wc_cont_agg
#SBATCH --account=def-uanazodo-ab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/wc_cont_agg_%j.out
#SBATCH --error=logs/wc_cont_agg_%j.err

module purge
module load StdEnv/2023 python/3.10 gcc
source ~/gpinn_env/bin/activate

export PYTHONUNBUFFERED=1

cd ~/scratch/gpinn_project
python aggregate_continuity_ablation.py \
    --input_dir "./outputs/continuity_ablation/runs" \
    --output_dir "./outputs/continuity_ablation"
