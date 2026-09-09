#!/usr/bin/env python3
"""
Aggregate noise × smoothing regression sensitivity sweep.
Produces:
  all_runs.csv
  smoothing_summary.csv
  best_by_noise.csv
  robust_smoothing_ranking.csv
  sensitivity_summary.json
  heatmap_mean_error_noise-*.png
  heatmap_D_error_noise-*.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def save_heatmap(df, value_col, title, path):
    piv = df.pivot(
        index="smooth_sigma_t_frames",
        columns="smooth_sigma_space_pixels",
        values=value_col,
    ).sort_index().sort_index(axis=1)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(piv.values, origin="lower", aspect="auto")
    ax.set_xticks(np.arange(len(piv.columns)))
    ax.set_xticklabels([f"{x:g}" for x in piv.columns])
    ax.set_yticks(np.arange(len(piv.index)))
    ax.set_yticklabels([f"{x:g}" for x in piv.index])
    ax.set_xlabel("Spatial Gaussian sigma (pixels)")
    ax.set_ylabel("Temporal Gaussian sigma (frames)")
    ax.set_title(title)

    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            val = piv.iloc[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, label="Relative error (%)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir",
                   default="synthetic_outputs/regression_sensitivity")
    args = p.parse_args()

    out = Path(args.input_dir)
    files = sorted(out.glob("task-*.csv"))
    if not files:
        raise FileNotFoundError(f"No task CSV files found in {out}")

    dfs = [pd.read_csv(f) for f in files]
    all_df = pd.concat(dfs, ignore_index=True)
    all_df.to_csv(out / "all_runs.csv", index=False)

    good = all_df[all_df["success"] == True].copy()
    if good.empty:
        raise RuntimeError("No successful regression rows.")

    group_cols = [
        "noise_sigma",
        "smooth_sigma_t_frames",
        "smooth_sigma_space_pixels",
    ]

    summary = good.groupby(group_cols).agg(
        n=("D_E_est", "count"),
        D_E_mean=("D_E_est", "mean"),
        D_E_sd=("D_E_est", "std"),
        D_E_relerr_mean_pct=("D_E_relerr_pct", "mean"),
        D_E_relerr_sd_pct=("D_E_relerr_pct", "std"),
        w_EE_mean=("w_EE_est", "mean"),
        w_EE_sd=("w_EE_est", "std"),
        w_EE_relerr_mean_pct=("w_EE_relerr_pct", "mean"),
        w_EE_relerr_sd_pct=("w_EE_relerr_pct", "std"),
        h_mean=("h_est", "mean"),
        h_sd=("h_est", "std"),
        h_relerr_mean_pct=("h_relerr_pct", "mean"),
        h_relerr_sd_pct=("h_relerr_pct", "std"),
        mean_param_relerr_mean_pct=("mean_param_relerr_pct", "mean"),
        max_param_relerr_mean_pct=("max_param_relerr_pct", "mean"),
        valid_fraction_mean=("fraction_valid", "mean"),
        cond_standardized_mean=("condition_number_standardized_E_lap", "mean"),
        Et_r2_mean=("Et_pseudo_r2_vs_truth", "mean"),
        lap_r2_mean=("lap_pseudo_r2_vs_truth", "mean"),
    ).reset_index()

    summary.to_csv(out / "smoothing_summary.csv", index=False)

    best_rows = []
    for noise, sub in summary.groupby("noise_sigma"):
        best = sub.sort_values(
            ["mean_param_relerr_mean_pct", "max_param_relerr_mean_pct"]
        ).iloc[0]
        best_rows.append(best)
    best_by_noise = pd.DataFrame(best_rows)
    best_by_noise.to_csv(out / "best_by_noise.csv", index=False)

    # Robust smoothing choice: optimize across all nonzero noise levels and seeds.
    noisy = good[good["noise_sigma"] > 0].copy()
    robust = noisy.groupby(
        ["smooth_sigma_t_frames", "smooth_sigma_space_pixels"]
    ).agg(
        n=("D_E_est", "count"),
        mean_param_relerr_pct=("mean_param_relerr_pct", "mean"),
        max_param_relerr_mean_pct=("max_param_relerr_pct", "mean"),
        D_E_relerr_mean_pct=("D_E_relerr_pct", "mean"),
        w_EE_relerr_mean_pct=("w_EE_relerr_pct", "mean"),
        h_relerr_mean_pct=("h_relerr_pct", "mean"),
        valid_fraction_mean=("fraction_valid", "mean"),
        cond_standardized_mean=("condition_number_standardized_E_lap", "mean"),
    ).reset_index().sort_values(
        ["mean_param_relerr_pct", "max_param_relerr_mean_pct"]
    )
    robust.to_csv(out / "robust_smoothing_ranking.csv", index=False)

    # Heatmaps use mean across noise seeds for each noise level.
    for noise, sub in summary.groupby("noise_sigma"):
        tag = f"{noise:.3f}".replace(".", "p")
        save_heatmap(
            sub,
            "mean_param_relerr_mean_pct",
            f"Mean parameter error, noise sigma={noise:g}",
            out / f"heatmap_mean_error_noise-{tag}.png",
        )
        save_heatmap(
            sub,
            "D_E_relerr_mean_pct",
            f"D_E relative error, noise sigma={noise:g}",
            out / f"heatmap_D_error_noise-{tag}.png",
        )

    robust_best = robust.iloc[0].to_dict()
    payload = {
        "n_task_files": len(files),
        "n_rows": int(len(all_df)),
        "n_success": int(len(good)),
        "best_by_noise": best_by_noise.to_dict(orient="records"),
        "robust_best_nonzero_noise": robust_best,
        "interpretation_note": (
            "The robust smoothing setting minimizes average parameter relative "
            "error across sigma=0.005, 0.01, and 0.02 noise conditions. "
            "Synthetic truth is used only to score sensitivity, not in regression."
        ),
    }
    (out / "sensitivity_summary.json").write_text(
        json.dumps(payload, indent=2, default=float)
    )

    print("\n=== BEST SMOOTHING BY NOISE LEVEL ===")
    cols = [
        "noise_sigma",
        "smooth_sigma_t_frames",
        "smooth_sigma_space_pixels",
        "D_E_relerr_mean_pct",
        "w_EE_relerr_mean_pct",
        "h_relerr_mean_pct",
        "mean_param_relerr_mean_pct",
        "valid_fraction_mean",
    ]
    print(best_by_noise[cols].to_string(index=False))

    print("\n=== ROBUST BEST ACROSS NONZERO NOISE LEVELS ===")
    print(robust.head(10).to_string(index=False))

    print(f"\nSaved outputs to {out}")


if __name__ == "__main__":
    main()
