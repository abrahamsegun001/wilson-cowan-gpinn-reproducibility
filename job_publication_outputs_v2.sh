#!/bin/bash
#SBATCH --job-name=wc_pub_v2
#SBATCH --account=def-uanazodo-ab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/wc_pub_v2_%j.out
#SBATCH --error=logs/wc_pub_v2_%j.err

module purge
module load StdEnv/2023 python/3.10 gcc
source ~/gpinn_env/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

cd ~/scratch/gpinn_project
mkdir -p logs outputs/publication_v2/{master,tables,figures}

python extract_publication_results_v2.py \
  --project_root . \
  --final_dir outputs/final_dual_model_sequential20_cont005 \
  --no_cont_dir outputs/final_dual_model_sequential20 \
  --continuity_dir outputs/continuity_ablation \
  --synthetic_dir synthetic_identifiability_package/synthetic_outputs \
  --output_dir outputs/publication_v2

python make_publication_tables_v2.py \
  --tables_dir outputs/publication_v2/tables

python make_publication_figures_v2.py \
  --project_root . \
  --master_dir outputs/publication_v2/master \
  --final_run_dir outputs/final_dual_model_sequential20_cont005 \
  --data_dir data/processed \
  --subject_prefix sub-GC48_ses-day0 \
  --representative_chunk 12 \
  --output_dir outputs/publication_v2/figures \
  --dpi 300 \
  --scatter_points 100000 \
  --seed 42

printf '\nV2 DONE\n'
printf 'Tables:  outputs/publication_v2/tables/\n'
printf 'Figures: outputs/publication_v2/figures/\n'
printf 'Master:  outputs/publication_v2/master/\n'
