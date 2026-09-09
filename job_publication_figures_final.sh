#!/bin/bash
#SBATCH --job-name=wc_pub_final
#SBATCH --account=def-uanazodo-ab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/wc_pub_final_%j.out
#SBATCH --error=logs/wc_pub_final_%j.err

# Fail immediately if ANY publication step fails.
set -euo pipefail

module purge
module load StdEnv/2023 python/3.10 gcc
source ~/gpinn_env/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

cd ~/scratch/gpinn_project
mkdir -p logs

# Build a clean final package from the already-audited V2 outputs.
rm -rf outputs/publication_final
mkdir -p outputs/publication_final/figures

cp -a outputs/publication_v2/master outputs/publication_final/master
cp -a outputs/publication_v2/tables outputs/publication_final/tables

# Figures 2, 3 and 5 were already visually accepted in V2.
cp outputs/publication_v2/figures/figure2_synthetic_recoverability_v2.png \
   outputs/publication_final/figures/figure2_synthetic_recoverability.png
cp outputs/publication_v2/figures/figure2_synthetic_recoverability_v2.pdf \
   outputs/publication_final/figures/figure2_synthetic_recoverability.pdf

cp outputs/publication_v2/figures/figure3_real_model_selection_v2.png \
   outputs/publication_final/figures/figure3_real_model_selection.png
cp outputs/publication_v2/figures/figure3_real_model_selection_v2.pdf \
   outputs/publication_final/figures/figure3_real_model_selection.pdf

cp outputs/publication_v2/figures/figure5_sequential_continuity_v2.png \
   outputs/publication_final/figures/figure5_sequential_continuity.png
cp outputs/publication_v2/figures/figure5_sequential_continuity_v2.pdf \
   outputs/publication_final/figures/figure5_sequential_continuity.pdf

# Regenerate only the figures that required layout correction.
python make_publication_figures_final_hexbin.py \
  --project_root . \
  --master_dir outputs/publication_final/master \
  --final_run_dir outputs/final_dual_model_sequential20_cont005 \
  --data_dir data/processed \
  --subject_prefix sub-GC48_ses-day0 \
  --representative_chunk 12 \
  --output_dir outputs/publication_final/figures \
  --dpi 300 \
  --scatter_points 30000 \
  --seed 42

printf '\nFINAL PUBLICATION PACKAGE DONE\n'
printf 'Figures: outputs/publication_final/figures/\n'
printf 'Tables:  outputs/publication_final/tables/\n'
printf 'Master:  outputs/publication_final/master/\n'
