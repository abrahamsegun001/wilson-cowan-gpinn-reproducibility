# Manuscript-to-code map

| Manuscript component | Primary code | Main outputs |
|---|---|---|
| Final regression-assisted sequential gPINN | `train_gpinn_narval_final.py` | `final_pipeline_results.csv`, `final_pipeline_summary.json`, checkpoints, histories, model-selection details |
| Baseline synthetic twin experiment | `synthetic_identifiability_package/` | recovery CSVs, checkpoints, baseline synthetic figures |
| Learned-derivative diagnostic | `diagnose_derivatives.py` | state / derivative fidelity diagnostics |
| Oracle derivative control | `train_synthetic_oracle_derivative.py` | oracle parameter recovery |
| Staged non-oracle control | `train_synthetic_staged_nonoracle.py` | staged recovery diagnostic |
| Pseudo-derivative regression control | `train_synthetic_pseudoderiv_regression.py` | derivative-informed coefficient recovery |
| Noise and smoothing calibration | `sweep_pseudoderiv_regression.py`, `aggregate_pseudoderiv_regression_sweep.py` | smoothing summary and robust setting |
| Final selector operating control | `synthetic_model_selection_control.py` | positive / null selector decisions |
| Real lower-bound audit | `real_calcium_D_bound_audit.py` | bounded vs unrestricted D_E audit |
| Resolution-aware regression audit | `real_calcium_regression_audit_resolution_aware.py` | chunk-wise coefficient audit |
| Nested uncertainty audit | `real_calcium_nested_uncertainty_audit.py` | block-bootstrap intervals and held-out comparison |
| Continuity ablation | `continuity_ablation.py`, `aggregate_continuity_ablation.py` | continuity-weight comparison |
| Publication tables / master data | `extract_publication_results_v2.py`, `make_publication_tables_v2.py` | manuscript CSV tables and master files |
| Figures 1-5 | `make_publication_figures_v2.py`, then `make_publication_figures_final_hexbin.py` for final Fig. 1/Fig. 4/S1 layout | PNG/PDF figures |

The canonical final training implementation is `train_gpinn_narval_final.py`. Older sequential scripts are included only because they were used in supporting continuity and validation experiments during development.
