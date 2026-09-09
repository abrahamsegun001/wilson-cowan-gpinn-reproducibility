# Data and preprocessing specification

## Source

The study analyses one 10-minute resting-state session from the public BraiDyn-BC dataset, DANDI accession 001425:

`sub-VG1-GC#48_ses-2023-08-18-resting-state-day0_widefield+behavior.nwb`

BraiDyn-BC is one-photon cortex-wide calcium imaging. The acquisition alternates blue calcium-dependent and violet approximately calcium-independent excitation channels.

## Preprocessing used in the manuscript

The manuscript specifies the following steps before the `*.npz` chunks consumed by the modelling code:

1. Estimate the per-pixel baseline independently for blue and violet channels as the 10th percentile of 5000 temporally subsampled frames.
2. Compute fractional fluorescence change independently for each channel using a numerical floor of `1e-6` for the baseline denominator.
3. Mean-centre the violet fractional signal at each pixel.
4. Estimate a pixel-wise ordinary least squares slope of blue fractional fluorescence on the centred violet predictor.
5. Subtract the fitted centred violet component from the blue signal.
6. Register frames to the common cortical coordinate system using the supplied affine mapping and spatially resample.
7. Resample to a `64 x 64` working grid at 30 Hz.
8. Split the 10-minute session into twenty consecutive, non-overlapping 30-second chunks (`900 x 64 x 64`).
9. Save each corrected chunk as an NPZ containing the key `dfof`.
10. Session-global min-max normalization is performed later by the final modelling script across all 20 processed chunks.

## Important raw-to-processed reproducibility note

The exact Python script originally used to read the NWB channel paths, apply the supplied affine transform, and write the 20 NPZ files was **not among the source files available when this repository package was assembled**. It has therefore not been invented or reverse-engineered here.

Consequently, this release is fully executable from the processed chunk interface onward, while exact bit-for-bit reproduction from the raw NWB still requires the original raw preprocessing implementation. If that script is recovered, place it in `preprocessing/` and update this document.

This limitation is stated explicitly so that the repository does not imply a level of raw-data reproducibility that the supplied source files cannot support.
