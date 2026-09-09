#!/usr/bin/env python3
"""
Final nested-model / uncertainty audit for the real-calcium reduced Wilson-Cowan model.

Question
--------
Does the Laplacian term D_E * nabla^2 E provide reproducible out-of-time
explanatory value beyond the reduced model with D_E = 0, and is D_E
distinguishable from zero under a temporal block bootstrap?

Model after logit transform
---------------------------
For q = tau * E_t + E, 0 < q < 1:

    logit(q) = w_EE * E + D_E * Lap(E) + h

Nested models:
    Reduced: logit(q) = w_EE * E + h              (D_E = 0)
    Full:    logit(q) = w_EE * E + D_E * Lap(E) + h

Primary preprocessing
---------------------
- Same session-global min-max scaling as the current gPINN, for comparability.
- Gaussian temporal smoothing sigma_t = 2 frames.
- Gaussian spatial smoothing fixed in PHYSICAL units:
      2 pixels on a 128x128 grid over 10 mm
      = 0.1574803 mm
  which is ~0.992 pixel on a 64x64 grid over 10 mm.

Validation
----------
- 5 contiguous temporal folds per 30-s chunk.
- An embargo of 6 frames is removed from the training side around each held-out
  temporal fold to reduce contamination from temporal smoothing / centered
  differencing.
- Fits are ordinary least squares. This is appropriate here because the previous
  audit showed the fully unconstrained OLS and the w-bounded/D-unbounded fits
  were numerically identical on all chunks.

Uncertainty
-----------
- Temporal block bootstrap with 1-s (30-frame) non-overlapping blocks.
- Each block keeps all interior spatial pixels together, preserving within-block
  spatiotemporal dependence better than an iid point bootstrap.
- Percentile 95% bootstrap confidence intervals are reported for D_E, w_EE, h.
- These are diagnostic uncertainty intervals, not a formal population-level
  biological confidence interval from one animal/session.

Outputs
-------
outputs/nested_uncertainty_audit/
    nested_model_all_folds.csv
    nested_model_by_chunk.csv
    bootstrap_parameter_summary.csv
    nested_uncertainty_summary.json
    D_E_bootstrap_CI.png
    heldout_logit_improvement.png
    heldout_pde_improvement.png
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SYNTHETIC_FOV_MM = 10.0
SYNTHETIC_GRID_N = 128
SYNTHETIC_BEST_SIGMA_PX = 2.0
PRIMARY_SIGMA_XY_MM = SYNTHETIC_BEST_SIGMA_PX * (
    SYNTHETIC_FOV_MM / (SYNTHETIC_GRID_N - 1)
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="./data/processed")
    p.add_argument("--output_dir", type=str, default="./outputs/nested_uncertainty_audit")
    p.add_argument("--num_chunks", type=int, default=20)
    p.add_argument("--subject_prefix", type=str, default="sub-GC48_ses-day0")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--l_fov", type=float, default=10.0)
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--embargo_frames", type=int, default=6)
    p.add_argument("--bootstrap_reps", type=int, default=1000)
    p.add_argument("--bootstrap_block_frames", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eps", type=float, default=1e-4)
    return p.parse_args()


def chunk_path(data_dir: Path, prefix: str, idx: int) -> Path:
    return data_dir / f"{prefix}_chunk-{idx:02d}_dfof.npz"


def get_global_min_max(data_dir: Path, prefix: str, num_chunks: int):
    g_min, g_max = float("inf"), float("-inf")
    found = 0
    for i in range(num_chunks):
        f = chunk_path(data_dir, prefix, i)
        if not f.exists():
            continue
        E = np.load(f)["dfof"]
        g_min = min(g_min, float(np.nanmin(E)))
        g_max = max(g_max, float(np.nanmax(E)))
        found += 1
    if found == 0:
        raise FileNotFoundError("No expected chunk files were found.")
    return g_min, g_max, found


def compute_fields(E, fps: float, l_fov: float, tau: float, eps: float):
    """
    Return interior E, Et, Lap, q, y=logit(q), and validity mask.

    Shapes are (T-2, H-2, W-2). Temporal index 0 corresponds to original frame 1.
    """
    _, H, W = E.shape
    dt = 1.0 / float(fps)
    dx = float(l_fov) / float(W - 1)
    dy = float(l_fov) / float(H - 1)

    E0 = E[1:-1, 1:-1, 1:-1].astype(np.float64, copy=False)

    Et = (
        E[2:, 1:-1, 1:-1].astype(np.float64, copy=False)
        - E[:-2, 1:-1, 1:-1].astype(np.float64, copy=False)
    ) / (2.0 * dt)

    d2x = (
        E[1:-1, 1:-1, 2:].astype(np.float64, copy=False)
        - 2.0 * E0
        + E[1:-1, 1:-1, :-2].astype(np.float64, copy=False)
    ) / (dx * dx)

    d2y = (
        E[1:-1, 2:, 1:-1].astype(np.float64, copy=False)
        - 2.0 * E0
        + E[1:-1, :-2, 1:-1].astype(np.float64, copy=False)
    ) / (dy * dy)

    lap = d2x + d2y
    q = float(tau) * Et + E0

    valid = (
        np.isfinite(E0)
        & np.isfinite(Et)
        & np.isfinite(lap)
        & np.isfinite(q)
        & (q > eps)
        & (q < 1.0 - eps)
    )

    y = np.full_like(q, np.nan, dtype=np.float64)
    y[valid] = np.log(q[valid] / (1.0 - q[valid]))

    return E0, Et, lap, q, y, valid


def full_sufficient_stats(E, L, y):
    """Sufficient statistics for X=[E,L,1]."""
    E = np.asarray(E, dtype=np.float64).ravel()
    L = np.asarray(L, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = float(len(y))

    return np.array([
        np.sum(E * E),
        np.sum(E * L),
        np.sum(E),
        np.sum(L * L),
        np.sum(L),
        n,
        np.sum(E * y),
        np.sum(L * y),
        np.sum(y),
    ], dtype=np.float64)


def reduced_sufficient_stats(E, y):
    """Sufficient statistics for X=[E,1]."""
    E = np.asarray(E, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = float(len(y))

    return np.array([
        np.sum(E * E),
        np.sum(E),
        n,
        np.sum(E * y),
        np.sum(y),
    ], dtype=np.float64)


def solve_full_from_stats(s):
    XTX = np.array([
        [s[0], s[1], s[2]],
        [s[1], s[3], s[4]],
        [s[2], s[4], s[5]],
    ], dtype=np.float64)
    XTy = np.array([s[6], s[7], s[8]], dtype=np.float64)

    try:
        beta = np.linalg.solve(XTX, XTy)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(XTX) @ XTy

    # [w, D, h]
    return beta


def solve_reduced_from_stats(s):
    XTX = np.array([
        [s[0], s[1]],
        [s[1], s[2]],
    ], dtype=np.float64)
    XTy = np.array([s[3], s[4]], dtype=np.float64)

    try:
        beta = np.linalg.solve(XTX, XTy)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(XTX) @ XTy

    # [w, h]
    return beta


def fit_full_arrays(E, L, y, valid):
    Ev = E[valid]
    Lv = L[valid]
    yv = y[valid]
    return solve_full_from_stats(full_sufficient_stats(Ev, Lv, yv))


def fit_reduced_arrays(E, y, valid):
    Ev = E[valid]
    yv = y[valid]
    return solve_reduced_from_stats(reduced_sufficient_stats(Ev, yv))


def evaluate_model(E, Et, L, y, valid, tau, full_beta=None, reduced_beta=None):
    Ev = E[valid]
    Etv = Et[valid]
    Lv = L[valid]
    yv = y[valid]

    if full_beta is not None:
        w, D, h = map(float, full_beta)
        z = w * Ev + D * Lv + h
    else:
        w, h = map(float, reduced_beta)
        D = 0.0
        z = w * Ev + h

    logit_err = z - yv
    logit_sse = float(np.sum(logit_err ** 2))
    logit_rmse = float(np.sqrt(logit_sse / len(yv)))

    sigmoid = 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))
    pde_res = float(tau) * Etv - (-Ev + sigmoid)
    pde_sse = float(np.sum(pde_res ** 2))
    pde_rmse = float(np.sqrt(pde_sse / len(yv)))

    return {
        "n_eval": int(len(yv)),
        "logit_sse": logit_sse,
        "logit_rmse": logit_rmse,
        "pde_sse": pde_sse,
        "pde_rmse": pde_rmse,
    }


def contiguous_folds(n_frames: int, n_folds: int):
    idx = np.arange(n_frames)
    return [np.asarray(x, dtype=int) for x in np.array_split(idx, n_folds)]


def train_frame_mask(n_frames: int, test_frames, embargo: int):
    mask = np.ones(n_frames, dtype=bool)
    lo = max(0, int(test_frames[0]) - int(embargo))
    hi = min(n_frames, int(test_frames[-1]) + int(embargo) + 1)
    mask[lo:hi] = False
    return mask


def bootstrap_block_stats(E, L, y, valid, block_frames: int):
    """
    Build sufficient stats for non-overlapping temporal blocks.

    Each block contains all valid spatial pixels in that time interval.
    """
    n_frames = E.shape[0]
    blocks = []

    start = 0
    while start < n_frames:
        end = min(n_frames, start + block_frames)
        vb = valid[start:end]
        n_valid = int(vb.sum())
        if n_valid >= 100:
            Eb = E[start:end][vb]
            Lb = L[start:end][vb]
            yb = y[start:end][vb]
            blocks.append({
                "start": int(start),
                "end": int(end),
                "n_valid": n_valid,
                "stats": full_sufficient_stats(Eb, Lb, yb),
            })
        start = end

    return blocks


def bootstrap_parameters(blocks, reps: int, seed: int):
    if len(blocks) < 5:
        raise RuntimeError("Too few temporal blocks for bootstrap.")

    rng = np.random.default_rng(seed)
    stats = np.stack([b["stats"] for b in blocks], axis=0)
    B = len(blocks)

    params = np.empty((reps, 3), dtype=np.float64)

    for r in range(reps):
        draw = rng.integers(0, B, size=B)
        s = stats[draw].sum(axis=0)
        params[r] = solve_full_from_stats(s)

    return params


def percentile_summary(x):
    x = np.asarray(x, dtype=np.float64)
    return {
        "mean": float(np.mean(x)),
        "sd": float(np.std(x, ddof=1)),
        "median": float(np.median(x)),
        "ci95_low": float(np.quantile(x, 0.025)),
        "ci95_high": float(np.quantile(x, 0.975)),
    }


def write_csv(path, rows):
    if not rows:
        return
    fields = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_D_ci(rows, out_path):
    rows = sorted(rows, key=lambda r: r["chunk_idx"])
    x = np.array([r["chunk_idx"] for r in rows], dtype=float)
    y = np.array([r["D_E_point"] for r in rows], dtype=float)
    lo = np.array([r["D_E_ci95_low"] for r in rows], dtype=float)
    hi = np.array([r["D_E_ci95_high"] for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(
        x, y,
        yerr=np.vstack([y - lo, hi - y]),
        fmt="o-", capsize=3
    )
    ax.axhline(0.0, linewidth=1.0)
    ax.set_xlabel("30-s chunk")
    ax.set_ylabel("D_E")
    ax.set_title("Unrestricted D_E with temporal block-bootstrap 95% intervals")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_improvement(rows, key, ylabel, title, out_path):
    rows = sorted(rows, key=lambda r: r["chunk_idx"])
    x = [r["chunk_idx"] for r in rows]
    y = [r[key] for r in rows]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x, y)
    ax.axhline(0.0, linewidth=1.0)
    ax.set_xlabel("30-s chunk")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    g_min, g_max, n_found = get_global_min_max(
        data_dir, args.subject_prefix, args.num_chunks
    )
    scale = g_max - g_min + 1e-8

    print(f"Found {n_found} chunks.")
    print(f"Global dF/F min={g_min:.8g}, max={g_max:.8g}")
    print(f"Primary sigma_xy={PRIMARY_SIGMA_XY_MM:.8f} mm")
    print(
        f"CV: {args.n_folds} contiguous temporal folds, "
        f"embargo={args.embargo_frames} frames"
    )
    print(
        f"Bootstrap: {args.bootstrap_reps} reps, "
        f"block={args.bootstrap_block_frames} frames "
        f"({args.bootstrap_block_frames/args.fps:.2f} s)"
    )

    fold_rows = []
    chunk_rows = []
    boot_rows = []

    for chunk_idx in range(args.num_chunks):
        f = chunk_path(data_dir, args.subject_prefix, chunk_idx)
        if not f.exists():
            print(f"Chunk {chunk_idx:02d}: missing; skipped.")
            continue

        raw = np.load(f)["dfof"].astype(np.float64)
        E_scaled = (raw - g_min) / scale

        T, H, W = E_scaled.shape
        dx_mm = args.l_fov / (W - 1)
        sigma_xy_px = PRIMARY_SIGMA_XY_MM / dx_mm

        E_sm = gaussian_filter(
            E_scaled,
            sigma=(2.0, sigma_xy_px, sigma_xy_px),
            mode="reflect",
        )

        E0, Et, lap, q, y, valid = compute_fields(
            E_sm, args.fps, args.l_fov, args.tau, args.eps
        )

        valid_fraction = float(valid.mean())
        n_frames = E0.shape[0]

        # Whole-chunk point estimates.
        beta_full_all = fit_full_arrays(E0, lap, y, valid)
        beta_red_all = fit_reduced_arrays(E0, y, valid)

        # Temporally blocked cross-validation.
        folds = contiguous_folds(n_frames, args.n_folds)
        chunk_fold_rows = []

        for fold_idx, test_frames in enumerate(folds):
            train_tmask = train_frame_mask(
                n_frames, test_frames, args.embargo_frames
            )

            train_valid = valid & train_tmask[:, None, None]
            test_tmask = np.zeros(n_frames, dtype=bool)
            test_tmask[test_frames] = True
            test_valid = valid & test_tmask[:, None, None]

            if train_valid.sum() < 100 or test_valid.sum() < 100:
                continue

            beta_full = fit_full_arrays(E0, lap, y, train_valid)
            beta_red = fit_reduced_arrays(E0, y, train_valid)

            ef = evaluate_model(
                E0, Et, lap, y, test_valid, args.tau,
                full_beta=beta_full
            )
            er = evaluate_model(
                E0, Et, lap, y, test_valid, args.tau,
                reduced_beta=beta_red
            )

            logit_improve_pct = 100.0 * (
                er["logit_rmse"] - ef["logit_rmse"]
            ) / er["logit_rmse"]

            pde_improve_pct = 100.0 * (
                er["pde_rmse"] - ef["pde_rmse"]
            ) / er["pde_rmse"]

            logit_sse_improve_pct = 100.0 * (
                er["logit_sse"] - ef["logit_sse"]
            ) / er["logit_sse"]

            pde_sse_improve_pct = 100.0 * (
                er["pde_sse"] - ef["pde_sse"]
            ) / er["pde_sse"]

            row = {
                "chunk_idx": chunk_idx,
                "time_start_min": 0.5 * chunk_idx,
                "fold_idx": fold_idx,
                "test_frame_start_interior": int(test_frames[0]),
                "test_frame_end_interior": int(test_frames[-1]),
                "test_duration_s": float(len(test_frames) / args.fps),
                "embargo_frames": args.embargo_frames,
                "n_train_valid": int(train_valid.sum()),
                "n_test_valid": int(test_valid.sum()),
                "full_train_w_EE": float(beta_full[0]),
                "full_train_D_E": float(beta_full[1]),
                "full_train_h": float(beta_full[2]),
                "reduced_train_w_EE": float(beta_red[0]),
                "reduced_train_h": float(beta_red[1]),
                "full_test_logit_rmse": ef["logit_rmse"],
                "reduced_test_logit_rmse": er["logit_rmse"],
                "heldout_logit_rmse_improvement_pct": logit_improve_pct,
                "heldout_logit_sse_improvement_pct": logit_sse_improve_pct,
                "full_test_pde_rmse": ef["pde_rmse"],
                "reduced_test_pde_rmse": er["pde_rmse"],
                "heldout_pde_rmse_improvement_pct": pde_improve_pct,
                "heldout_pde_sse_improvement_pct": pde_sse_improve_pct,
            }
            fold_rows.append(row)
            chunk_fold_rows.append(row)

        # Temporal block bootstrap on the full chunk.
        blocks = bootstrap_block_stats(
            E0, lap, y, valid, args.bootstrap_block_frames
        )
        boot = bootstrap_parameters(
            blocks, args.bootstrap_reps, args.seed + 10000 + chunk_idx
        )

        w_boot = percentile_summary(boot[:, 0])
        D_boot = percentile_summary(boot[:, 1])
        h_boot = percentile_summary(boot[:, 2])

        D_excludes_zero = (
            D_boot["ci95_low"] > 0.0 or D_boot["ci95_high"] < 0.0
        )
        D_positive_fraction = float(np.mean(boot[:, 1] > 0.0))

        boot_row = {
            "chunk_idx": chunk_idx,
            "time_start_min": 0.5 * chunk_idx,
            "n_bootstrap_blocks": len(blocks),
            "bootstrap_reps": args.bootstrap_reps,
            "bootstrap_block_frames": args.bootstrap_block_frames,
            "bootstrap_block_seconds": args.bootstrap_block_frames / args.fps,
            "w_EE_point": float(beta_full_all[0]),
            "w_EE_boot_mean": w_boot["mean"],
            "w_EE_boot_sd": w_boot["sd"],
            "w_EE_ci95_low": w_boot["ci95_low"],
            "w_EE_ci95_high": w_boot["ci95_high"],
            "D_E_point": float(beta_full_all[1]),
            "D_E_boot_mean": D_boot["mean"],
            "D_E_boot_sd": D_boot["sd"],
            "D_E_ci95_low": D_boot["ci95_low"],
            "D_E_ci95_high": D_boot["ci95_high"],
            "D_E_ci95_excludes_zero": bool(D_excludes_zero),
            "D_E_boot_fraction_positive": D_positive_fraction,
            "h_point": float(beta_full_all[2]),
            "h_boot_mean": h_boot["mean"],
            "h_boot_sd": h_boot["sd"],
            "h_ci95_low": h_boot["ci95_low"],
            "h_ci95_high": h_boot["ci95_high"],
        }
        boot_rows.append(boot_row)

        # Aggregate held-out performance for this chunk.
        if chunk_fold_rows:
            full_logit = np.array(
                [r["full_test_logit_rmse"] for r in chunk_fold_rows]
            )
            red_logit = np.array(
                [r["reduced_test_logit_rmse"] for r in chunk_fold_rows]
            )
            full_pde = np.array(
                [r["full_test_pde_rmse"] for r in chunk_fold_rows]
            )
            red_pde = np.array(
                [r["reduced_test_pde_rmse"] for r in chunk_fold_rows]
            )
            logit_imp = np.array(
                [r["heldout_logit_rmse_improvement_pct"]
                 for r in chunk_fold_rows]
            )
            pde_imp = np.array(
                [r["heldout_pde_rmse_improvement_pct"]
                 for r in chunk_fold_rows]
            )
        else:
            full_logit = red_logit = full_pde = red_pde = np.array([np.nan])
            logit_imp = pde_imp = np.array([np.nan])

        chunk_row = {
            "chunk_idx": chunk_idx,
            "time_start_min": 0.5 * chunk_idx,
            "source_file": f.name,
            "grid_T": T,
            "grid_H": H,
            "grid_W": W,
            "dx_mm": dx_mm,
            "smooth_sigma_t_frames": 2.0,
            "smooth_sigma_space_pixels": sigma_xy_px,
            "smooth_sigma_space_mm": PRIMARY_SIGMA_XY_MM,
            "valid_fraction": valid_fraction,
            "full_point_w_EE": float(beta_full_all[0]),
            "full_point_D_E": float(beta_full_all[1]),
            "full_point_h": float(beta_full_all[2]),
            "reduced_point_w_EE": float(beta_red_all[0]),
            "reduced_point_h": float(beta_red_all[1]),
            "n_cv_folds_completed": len(chunk_fold_rows),
            "mean_full_heldout_logit_rmse": float(np.nanmean(full_logit)),
            "mean_reduced_heldout_logit_rmse": float(np.nanmean(red_logit)),
            "mean_heldout_logit_rmse_improvement_pct": float(np.nanmean(logit_imp)),
            "mean_full_heldout_pde_rmse": float(np.nanmean(full_pde)),
            "mean_reduced_heldout_pde_rmse": float(np.nanmean(red_pde)),
            "mean_heldout_pde_rmse_improvement_pct": float(np.nanmean(pde_imp)),
            "D_E_boot_sd": D_boot["sd"],
            "D_E_ci95_low": D_boot["ci95_low"],
            "D_E_ci95_high": D_boot["ci95_high"],
            "D_E_ci95_excludes_zero": bool(D_excludes_zero),
            "D_E_boot_fraction_positive": D_positive_fraction,
        }
        chunk_rows.append(chunk_row)

        print(
            f"\nChunk {chunk_idx:02d} | valid={valid_fraction:.4f} "
            f"| D={beta_full_all[1]:+.6f} "
            f"CI=[{D_boot['ci95_low']:+.6f}, {D_boot['ci95_high']:+.6f}] "
            f"| CI excludes 0: {D_excludes_zero}"
        )
        print(
            f"  held-out logit improvement = "
            f"{chunk_row['mean_heldout_logit_rmse_improvement_pct']:+.4f}%"
        )
        print(
            f"  held-out PDE improvement   = "
            f"{chunk_row['mean_heldout_pde_rmse_improvement_pct']:+.4f}%"
        )

    write_csv(out / "nested_model_all_folds.csv", fold_rows)
    write_csv(out / "nested_model_by_chunk.csv", chunk_rows)
    write_csv(out / "bootstrap_parameter_summary.csv", boot_rows)

    if boot_rows:
        plot_D_ci(boot_rows, out / "D_E_bootstrap_CI.png")
    if chunk_rows:
        plot_improvement(
            chunk_rows,
            "mean_heldout_logit_rmse_improvement_pct",
            "Full-model improvement (%)",
            "Held-out logit RMSE improvement from adding D_E Lap(E)",
            out / "heldout_logit_improvement.png",
        )
        plot_improvement(
            chunk_rows,
            "mean_heldout_pde_rmse_improvement_pct",
            "Full-model improvement (%)",
            "Held-out PDE RMSE improvement from adding D_E Lap(E)",
            out / "heldout_pde_improvement.png",
        )

    # Session-level descriptive summary.
    D_points = np.array([r["full_point_D_E"] for r in chunk_rows], dtype=float)
    D_ci_excl = np.array(
        [bool(r["D_E_ci95_excludes_zero"]) for r in chunk_rows], dtype=bool
    )
    logit_imp = np.array(
        [r["mean_heldout_logit_rmse_improvement_pct"] for r in chunk_rows],
        dtype=float,
    )
    pde_imp = np.array(
        [r["mean_heldout_pde_rmse_improvement_pct"] for r in chunk_rows],
        dtype=float,
    )

    summary = {
        "purpose": (
            "Nested-model held-out comparison and temporal block-bootstrap "
            "uncertainty audit for the Laplacian coefficient D_E."
        ),
        "scope": "single resting-state calcium-imaging session; 20 x 30-s chunks",
        "normalization": "session-global min-max scaling matching current gPINN",
        "primary_smoothing": {
            "sigma_t_frames": 2.0,
            "sigma_xy_mm": PRIMARY_SIGMA_XY_MM,
        },
        "nested_models": {
            "reduced": "logit(tau*E_t+E) = w_EE*E + h; D_E fixed to 0",
            "full": "logit(tau*E_t+E) = w_EE*E + D_E*Lap(E) + h; unrestricted OLS",
        },
        "cross_validation": {
            "n_contiguous_temporal_folds": args.n_folds,
            "embargo_frames": args.embargo_frames,
            "embargo_seconds": args.embargo_frames / args.fps,
        },
        "bootstrap": {
            "reps": args.bootstrap_reps,
            "temporal_block_frames": args.bootstrap_block_frames,
            "temporal_block_seconds": args.bootstrap_block_frames / args.fps,
            "resampling_unit": (
                "temporal blocks; all interior spatial pixels within a block "
                "are kept together"
            ),
        },
        "session_descriptive_results": {
            "n_chunks": len(chunk_rows),
            "D_E_point_mean": float(np.mean(D_points)) if len(D_points) else None,
            "D_E_point_sd": float(np.std(D_points, ddof=1)) if len(D_points) > 1 else None,
            "D_E_point_min": float(np.min(D_points)) if len(D_points) else None,
            "D_E_point_max": float(np.max(D_points)) if len(D_points) else None,
            "n_chunks_D_CI95_excludes_zero": int(D_ci_excl.sum()) if len(D_ci_excl) else 0,
            "fraction_chunks_D_CI95_excludes_zero": float(D_ci_excl.mean()) if len(D_ci_excl) else None,
            "mean_heldout_logit_RMSE_improvement_pct_full_vs_D0": (
                float(np.nanmean(logit_imp)) if len(logit_imp) else None
            ),
            "median_heldout_logit_RMSE_improvement_pct_full_vs_D0": (
                float(np.nanmedian(logit_imp)) if len(logit_imp) else None
            ),
            "mean_heldout_PDE_RMSE_improvement_pct_full_vs_D0": (
                float(np.nanmean(pde_imp)) if len(pde_imp) else None
            ),
            "median_heldout_PDE_RMSE_improvement_pct_full_vs_D0": (
                float(np.nanmedian(pde_imp)) if len(pde_imp) else None
            ),
            "fraction_chunks_full_improves_heldout_logit_RMSE": (
                float(np.mean(logit_imp > 0.0)) if len(logit_imp) else None
            ),
            "fraction_chunks_full_improves_heldout_PDE_RMSE": (
                float(np.mean(pde_imp > 0.0)) if len(pde_imp) else None
            ),
        },
        "interpretation_guardrails": [
            (
                "A bootstrap interval containing zero supports the conclusion "
                "that this chunk does not provide stable evidence for a nonzero "
                "Laplacian coefficient under this reduced model."
            ),
            (
                "A tiny held-out improvement does not establish biological "
                "diffusion; D_E is an effective spatial-coupling coefficient "
                "for the chosen reduced model and calcium representation."
            ),
            (
                "This is a single-animal/session diagnostic and should not be "
                "generalized to resting-state cortex as a whole."
            ),
        ],
    }

    (out / "nested_uncertainty_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    print("\nFINAL SESSION SUMMARY")
    print(json.dumps(summary["session_descriptive_results"], indent=2))
    print(f"\nOutputs saved to: {out}")


if __name__ == "__main__":
    main()
