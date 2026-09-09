#!/bin/bash
#SBATCH --job-name=wc_fig4_hexbin
#SBATCH --account=def-uanazodo-ab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/wc_fig4_hexbin_%j.out
#SBATCH --error=logs/wc_fig4_hexbin_%j.err

set -euo pipefail

module purge
module load StdEnv/2023 python/3.10 gcc
source ~/gpinn_env/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

cd ~/scratch/gpinn_project
mkdir -p logs outputs/publication_final/figures

python - <<'PY'
from pathlib import Path
import importlib.util

spec = importlib.util.spec_from_file_location(
    "figfinal", "make_publication_figures_final_hexbin.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

root = Path(".").resolve()
m.figure4_reconstruction(
    project_root=root,
    master_dir=root / "outputs/publication_final/master",
    final_run_dir=root / "outputs/final_dual_model_sequential20_cont005",
    data_dir=root / "data/processed",
    subject_prefix="sub-GC48_ses-day0",
    chunk_idx=12,
    out_dir=root / "outputs/publication_final/figures",
    dpi=300,
    scatter_points=100000,
    seed=42,
)
print("FINAL FIGURE 4 HEXBIN DONE")
PY
