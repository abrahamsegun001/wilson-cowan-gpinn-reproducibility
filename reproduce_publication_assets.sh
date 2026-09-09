#!/bin/bash
set -euo pipefail

FINAL_DIR="${FINAL_DIR:-outputs/final_gpinn_pipeline_cont005}"
NO_CONT_DIR="${NO_CONT_DIR:-outputs/final_gpinn_pipeline_nocont}"
CONTINUITY_DIR="${CONTINUITY_DIR:-outputs/continuity_ablation}"
SYNTHETIC_DIR="${SYNTHETIC_DIR:-synthetic_identifiability_package/synthetic_outputs}"
OUT_DIR="${OUT_DIR:-outputs/publication_v2}"

mkdir -p "$OUT_DIR"/{master,tables,figures}
python extract_publication_results_v2.py \
  --project_root . --final_dir "$FINAL_DIR" --no_cont_dir "$NO_CONT_DIR" \
  --continuity_dir "$CONTINUITY_DIR" --synthetic_dir "$SYNTHETIC_DIR" --output_dir "$OUT_DIR"
python make_publication_tables_v2.py --tables_dir "$OUT_DIR/tables"
python make_publication_figures_v2.py \
  --project_root . --master_dir "$OUT_DIR/master" --final_run_dir "$FINAL_DIR" \
  --data_dir data/processed --subject_prefix sub-GC48_ses-day0 --representative_chunk 12 \
  --output_dir "$OUT_DIR/figures" --dpi 300 --scatter_points 100000 --seed 42

# Overwrite Figure 1, Figure 4, and Supplementary Figure S1 with the final layout.
python make_publication_figures_final_hexbin.py \
  --project_root . --master_dir "$OUT_DIR/master" --final_run_dir "$FINAL_DIR" \
  --data_dir data/processed --subject_prefix sub-GC48_ses-day0 --representative_chunk 12 \
  --output_dir "$OUT_DIR/figures" --dpi 300 --scatter_points 30000 --seed 42
