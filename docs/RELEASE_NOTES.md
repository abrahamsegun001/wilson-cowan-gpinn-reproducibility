# Release notes

This archive consolidates the final modelling implementation and the supporting scripts used for the manuscript.

Repository-level cleanup applied while packaging:

- standardized publication labels from `Original joint gPINN` to `Baseline joint gPINN`;
- corrected the publication Table 1 Fourier-feature description to **128 total random projections (64 spatial + 64 temporal), 256 features after sine/cosine concatenation**;
- allowed publication extraction to read `final_pipeline_results.csv` / `final_pipeline_summary.json` for a no-continuity run produced by the canonical final script;
- added a no-continuity SLURM job using the canonical final trainer;
- added environment, data-interface, validation, and manuscript-to-code documentation.

No numerical result, selector threshold, architecture, or scientific model equation was changed during packaging.
