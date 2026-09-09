# Data interface

The final model consumes **processed 30-second chunk files**, not the raw NWB directly.

Expected path pattern:

```text
data/processed/sub-GC48_ses-day0_chunk-00_dfof.npz
...
data/processed/sub-GC48_ses-day0_chunk-19_dfof.npz
```

Each file must contain one NumPy array under the key `dfof`, with axes `[time, y, x]`. The manuscript working data use shape `900 x 64 x 64` at 30 Hz. Do not pre-apply the final session-global min-max normalization: `train_gpinn_narval_final.py` calculates that normalization across the supplied session chunks.

The source recording is from BraiDyn-BC, DANDI accession **001425**, session:

```text
sub-VG1-GC#48_ses-2023-08-18-resting-state-day0_widefield+behavior.nwb
```

Raw NWB bytes and processed chunk arrays are intentionally not bundled in this GitHub package. See `docs/DATA_AND_PREPROCESSING.md`.
