# Final regression-assisted sequential Wilson-Cowan gPINN pipeline

Canonical training script: `train_gpinn_narval_final.py`

This script consolidates the validated method into one standalone file. It does
not depend on the earlier diagnostic, ablation, or validation scripts.

## Final method

For each 30-s chunk:

1. Apply the same session-global min-max normalization used for the offline
   analysis.
2. Estimate resolution-aware finite-difference pseudo-derivatives after
   Gaussian smoothing calibrated in physical units.
3. Fit the full reduced Wilson-Cowan candidate
   `tau E_t = -E + sigmoid(w_EE E + D_E Lap(E) + h)` and its nested `D_E=0`
   alternative.
4. Use random-pointwise nested CV, temporal block-bootstrap uncertainty,
   conditioning, fold consistency, and smoothing stability to decide whether
   the full spatial-coupling model is supported.
5. Fix the selected chunk-wise physical coefficients during neural training.
6. Train the state network in two stages: state pretraining, then
   physics/gradient/pseudo-derivative training.
7. Carry only the neural state representation to the next chunk; physical
   coefficients and optimizer states are not carried.
8. From chunk 1 onward, apply state-continuity regularization with
   `lambda_cont=0.05`.
9. Evaluate random-pointwise interpolation performance, neural PDE residual,
   derivative diagnostics, and boundary mismatch.

The 64x64 spatial resolution is inferred directly from each chunk. With a
10-mm FOV, the physical grid spacing is `10/(64-1) ≈ 0.15873 mm/pixel`, and
the spatial smoothing width is converted from the synthetic physical scale
accordingly.

## Primary outputs

- `final_pipeline_results.csv`
- `final_pipeline_summary.json`
- `checkpoints/`
- `histories/`
- `model_selection_details/`

The previously completed `final_dual_model_sequential20_cont005` run already
validated this method. The canonical script exists to provide a clean,
reproducible implementation for the manuscript/repository.
