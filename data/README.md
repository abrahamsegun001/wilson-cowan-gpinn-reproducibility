# Data interface

This modelling repository starts from processed 30-second widefield-calcium chunks.
The raw-to-processed implementation is maintained separately in the derivative-aware preprocessing repository:

**https://github.com/abrahamsegun001/derivative-aware-widefield**

Expected files:

```text
data/processed/sub-GC48_ses-day0_chunk-00_dfof.npz
...
data/processed/sub-GC48_ses-day0_chunk-19_dfof.npz
```

Each NPZ must contain `dfof` with axes `[time, y, x]`. The manuscript configuration uses `900 x 64 x 64` at 30 Hz. Do not pre-apply the modelling session-global min-max normalization; the training code computes it across the supplied session chunks.

The source data are from BraiDyn-BC, DANDI accession **001425**.
