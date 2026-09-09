#!/bin/bash
#SBATCH --job-name=wc_fig1_fix
#SBATCH --account=def-uanazodo-ab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=logs/wc_fig1_fix_%j.out
#SBATCH --error=logs/wc_fig1_fix_%j.err

set -euo pipefail

module purge
module load StdEnv/2023 python/3.10 gcc
source ~/gpinn_env/bin/activate

cd ~/scratch/gpinn_project
mkdir -p logs outputs/publication_final/figures

python - <<'PY'
from pathlib import Path
import importlib.util

spec = importlib.util.spec_from_file_location(
    "figfinal", "make_publication_figures_final_v2.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

m.figure1_pipeline(
    Path("outputs/publication_final/figures"),
    dpi=300
)
print("FINAL FIGURE 1 FIX DONE")
PY
