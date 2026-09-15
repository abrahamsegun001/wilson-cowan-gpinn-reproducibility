#!/usr/bin/env python3
"""
FINAL Narval Wilson-Cowan gPINN pipeline: regression-assisted model selection, sequential state transfer, and continuity regularization.

This script is intentionally separate from the original sequential gPINN and
does not overwrite any existing results.

Scientific design
-----------------
1. Keep the GENERAL full Wilson-Cowan model available:

       tau E_t = -E + sigmoid(w_EE E + D_E Lap(E) + h)

   but also consider the nested reduced model with D_E = 0.

2. For each chunk, use the ORIGINAL-style RANDOM POINTWISE split across the
   full space-time volume: 60,000 supervised training points and 15,000
   randomly held-out validation points by default.

3. Across the full chunk:
   - estimate pseudo-derivatives from resolution-aware Gaussian smoothing;
   - compare full vs reduced nested regressions with random pointwise K-fold CV;
   - retain a temporal block bootstrap for uncertainty in D_E (this is an
     uncertainty calculation, not the train/validation split);
   - check D_E stability across nearby spatial smoothing scales;
   - select the full model only when all pre-specified support criteria pass.

4. Estimate physical parameters from the selected regression model and FIX
   them during neural-state training. The gPINN is not allowed to drag a good
   regression estimate away from the data-supported value.

5. Train the neural state in two stages:
   A. state-only pretraining;
   B. fixed-parameter physics + gradient-residual + pseudo-derivative training.

6. Evaluate on random pointwise held-out state/derivative samples:
   state MSE/R2/r, neural PDE residual, and derivative agreement with the
   pseudo-derivative diagnostic. This is an interpolation-style validation,
   matching the original analysis philosophy; it is not temporal forecasting.

This is the 20-chunk sequential production experiment. Physical parameters
are estimated independently from each chunk and are NEVER warm-started.
Only the neural state representation (Fourier embedding + state-network
weights) is carried from chunk i-1 to chunk i. Optimizer state is reset for
every chunk.

A state-continuity penalty is applied from chunk 1 onward. The selected
lambda_cont=0.05 comes from the completed five-boundary ablation. Chunk 0 has
no continuity term. Physical parameters remain independently inferred and fixed.

Canonical implementation
------------------------
This file is self-contained: it does not import any of the exploratory,
ablation, or validation scripts used to develop the method. The only
inputs required are the preprocessed chunk .npz files and standard Python
dependencies listed in the imports below.

Important
---------
D_E is an effective Laplacian spatial-coupling coefficient in this reduced
model, not a direct biological tissue diffusion constant.

The automatic model selector uses conservative, explicit defaults. They are
saved with the outputs and can be sensitivity-tested later rather than hidden.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr


# ---------------------------------------------------------------------------
# Resolution-aware smoothing calibration from the synthetic sweep
# ---------------------------------------------------------------------------
SYNTHETIC_FOV_MM = 10.0
SYNTHETIC_GRID_N = 128
SYNTHETIC_BEST_SIGMA_PX = 2.0
PRIMARY_SIGMA_XY_MM = SYNTHETIC_BEST_SIGMA_PX * (
    SYNTHETIC_FOV_MM / (SYNTHETIC_GRID_N - 1)
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Final regression-assisted sequential Wilson-Cowan gPINN pipeline"
    )

    # Data / output
    p.add_argument("--data_dir", type=str, default="./data/processed")
    p.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/final_gpinn_pipeline",
    )
    p.add_argument(
        "--subject_prefix",
        type=str,
        default="sub-GC48_ses-day0",
    )
    p.add_argument(
        "--chunks",
        nargs="+",
        type=int,
        default=list(range(20)),
    )
    p.add_argument(
        "--num_chunks_for_scaling",
        type=int,
        default=20,
        help="Chunks used for the same session-global min-max scaling as the old gPINN.",
    )
    p.add_argument(
        "--old_results",
        type=str,
        default="./outputs/temporal_tracking_results.npz",
        help="Optional old result file for descriptive parameter comparison only.",
    )
    p.add_argument(
        "--independent_results_csv",
        type=str,
        default="./outputs/final_dual_model_validation_randomsplit/five_chunk_validation_results.csv",
        help="Optional five-chunk independent/cold reference for matched warm-start comparisons.",
    )
    p.add_argument(
        "--lambda_cont",
        type=float,
        default=0.05,
        help="State-continuity loss weight applied from chunk 1 onward.",
    )
    p.add_argument(
        "--continuity_batch",
        type=int,
        default=2048,
        help="Boundary-point minibatch size for the continuity term.",
    )

    # Physical / acquisition settings
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--l_fov", type=float, default=10.0)
    p.add_argument("--chunk_duration", type=float, default=30.0)

    # Original-style random pointwise validation + nested model validation
    p.add_argument(
        "--n_val",
        type=int,
        default=15000,
        help="Random pointwise state-validation samples; original pipeline used n_data*0.25 = 15000.",
    )
    p.add_argument("--nested_folds", type=int, default=5)
    p.add_argument("--bootstrap_reps", type=int, default=1000)
    p.add_argument("--bootstrap_block_frames", type=int, default=30)
    p.add_argument("--eps_logit", type=float, default=1e-4)

    # Explicit conservative model-selection criteria
    p.add_argument("--min_valid_fraction", type=float, default=0.90)
    p.add_argument("--max_standardized_condition", type=float, default=10.0)
    p.add_argument("--min_logit_improvement_pct", type=float, default=0.10)
    p.add_argument("--min_pde_improvement_pct", type=float, default=0.10)
    p.add_argument("--min_positive_fold_fraction", type=float, default=0.60)
    p.add_argument("--max_smoothing_relative_range", type=float, default=1.00)
    p.add_argument("--w_min", type=float, default=0.5)
    p.add_argument("--w_max", type=float, default=6.0)

    # Neural architecture
    p.add_argument("--ff_dim", type=int, default=128)
    p.add_argument("--hidden_neurons", type=int, default=256)
    p.add_argument("--hidden_layers", type=int, default=6)
    p.add_argument("--scale_xy", type=float, default=25.0)
    p.add_argument("--scale_t", type=float, default=50.0)

    # Training pools and batches
    p.add_argument("--n_data_pool", type=int, default=60000)
    p.add_argument("--n_colloc_pool", type=int, default=60000)
    p.add_argument("--n_deriv_pool", type=int, default=20000)
    p.add_argument("--data_batch", type=int, default=4096)
    p.add_argument("--colloc_batch", type=int, default=4096)
    p.add_argument("--deriv_batch", type=int, default=1024)

    # Training schedule
    p.add_argument("--stageA_steps", type=int, default=2000)
    p.add_argument("--stageB_steps", type=int, default=3000)
    p.add_argument("--lr_stageA", type=float, default=1e-3)
    p.add_argument("--lr_stageB", type=float, default=2e-4)
    p.add_argument("--lbfgs_steps", type=int, default=150)
    p.add_argument("--lambda_physics", type=float, default=0.05)
    p.add_argument("--lambda_deriv", type=float, default=1e-3)
    p.add_argument("--mu_grad", type=float, default=1e-5)

    # Evaluation sampling
    p.add_argument("--eval_state_batch", type=int, default=8192)
    p.add_argument("--eval_deriv_points", type=int, default=12000)
    p.add_argument("--eval_pde_points", type=int, default=12000)

    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def chunk_path(data_dir: Path, prefix: str, idx: int) -> Path:
    return data_dir / f"{prefix}_chunk-{idx:02d}_dfof.npz"


def get_global_min_max(data_dir: Path, prefix: str, n_chunks: int):
    gmin, gmax = float("inf"), float("-inf")
    found = 0
    for i in range(n_chunks):
        f = chunk_path(data_dir, prefix, i)
        if not f.exists():
            continue
        arr = np.load(f)["dfof"]
        gmin = min(gmin, float(np.nanmin(arr)))
        gmax = max(gmax, float(np.nanmax(arr)))
        found += 1
    if found == 0:
        raise FileNotFoundError("No chunk files were found for session-global scaling.")
    return gmin, gmax, found


# ---------------------------------------------------------------------------
# Pseudo-derivative regression / model-selection utilities
# ---------------------------------------------------------------------------

def compute_pseudo_fields(E_sm, fps: float, l_fov: float, tau: float, eps: float):
    """
    Return arrays on the interior grid:
      E0, Et, Lap, q, logit(q), valid
    Shapes: (T-2, H-2, W-2).
    """
    _, H, W = E_sm.shape
    dt = 1.0 / fps
    dx = l_fov / (W - 1)
    dy = l_fov / (H - 1)

    E0 = E_sm[1:-1, 1:-1, 1:-1].astype(np.float64, copy=False)
    Et = (
        E_sm[2:, 1:-1, 1:-1].astype(np.float64, copy=False)
        - E_sm[:-2, 1:-1, 1:-1].astype(np.float64, copy=False)
    ) / (2.0 * dt)

    d2x = (
        E_sm[1:-1, 1:-1, 2:].astype(np.float64, copy=False)
        - 2.0 * E0
        + E_sm[1:-1, 1:-1, :-2].astype(np.float64, copy=False)
    ) / (dx * dx)

    d2y = (
        E_sm[1:-1, 2:, 1:-1].astype(np.float64, copy=False)
        - 2.0 * E0
        + E_sm[1:-1, :-2, 1:-1].astype(np.float64, copy=False)
    ) / (dy * dy)

    lap = d2x + d2y
    q = tau * Et + E0

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


def full_stats(E, L, y):
    E = np.asarray(E, dtype=np.float64).ravel()
    L = np.asarray(L, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = float(len(y))
    return np.array(
        [
            np.sum(E * E),
            np.sum(E * L),
            np.sum(E),
            np.sum(L * L),
            np.sum(L),
            n,
            np.sum(E * y),
            np.sum(L * y),
            np.sum(y),
        ],
        dtype=np.float64,
    )


def reduced_stats(E, y):
    E = np.asarray(E, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = float(len(y))
    return np.array(
        [
            np.sum(E * E),
            np.sum(E),
            n,
            np.sum(E * y),
            np.sum(y),
        ],
        dtype=np.float64,
    )


def solve_full_stats(s):
    XTX = np.array(
        [
            [s[0], s[1], s[2]],
            [s[1], s[3], s[4]],
            [s[2], s[4], s[5]],
        ],
        dtype=np.float64,
    )
    XTy = np.array([s[6], s[7], s[8]], dtype=np.float64)
    try:
        beta = np.linalg.solve(XTX, XTy)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(XTX) @ XTy
    return beta  # [w, D, h]


def solve_reduced_stats(s):
    XTX = np.array(
        [[s[0], s[1]], [s[1], s[2]]],
        dtype=np.float64,
    )
    XTy = np.array([s[3], s[4]], dtype=np.float64)
    try:
        beta = np.linalg.solve(XTX, XTy)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(XTX) @ XTy
    return beta  # [w, h]


def fit_full(E, L, y, valid):
    return solve_full_stats(full_stats(E[valid], L[valid], y[valid]))


def fit_reduced(E, y, valid):
    return solve_reduced_stats(reduced_stats(E[valid], y[valid]))


def evaluate_regression(E, Et, L, y, valid, tau, beta_full=None, beta_reduced=None):
    Ev = E[valid]
    Etv = Et[valid]
    Lv = L[valid]
    yv = y[valid]

    if beta_full is not None:
        w, D, h = map(float, beta_full)
        z = w * Ev + D * Lv + h
    else:
        w, h = map(float, beta_reduced)
        D = 0.0
        z = w * Ev + h

    logit_err = z - yv
    logit_rmse = float(np.sqrt(np.mean(logit_err ** 2)))

    s = 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))
    pde_res = tau * Etv - (-Ev + s)
    pde_rmse = float(np.sqrt(np.mean(pde_res ** 2)))

    return logit_rmse, pde_rmse


def regression_condition(E, L, valid):
    Ev = E[valid]
    Lv = L[valid]

    X = np.column_stack([Ev, Lv, np.ones_like(Ev)])
    raw = float(np.linalg.cond(X))

    Xstd = np.column_stack(
        [
            (Ev - Ev.mean()) / (Ev.std() + 1e-12),
            (Lv - Lv.mean()) / (Lv.std() + 1e-12),
        ]
    )
    std = float(np.linalg.cond(Xstd))
    corr = float(np.corrcoef(Ev, Lv)[0, 1])
    return raw, std, corr


def evaluate_regression_vectors(E, Et, L, y, tau, beta_full=None, beta_reduced=None):
    E = np.asarray(E, dtype=np.float64).ravel()
    Et = np.asarray(Et, dtype=np.float64).ravel()
    L = np.asarray(L, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()

    if beta_full is not None:
        w, D, h = map(float, beta_full)
        z = w * E + D * L + h
    else:
        w, h = map(float, beta_reduced)
        z = w * E + h

    logit_rmse = float(np.sqrt(np.mean((z - y) ** 2)))
    s = 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))
    pde_res = tau * Et - (-E + s)
    pde_rmse = float(np.sqrt(np.mean(pde_res ** 2)))
    return logit_rmse, pde_rmse


def nested_random_cv(E, Et, L, y, valid, tau, n_folds, seed):
    """
    Random pointwise K-fold comparison of the full and D_E=0 nested models.

    Sufficient statistics are additive, so each training fit is obtained from
    total statistics minus the held-out fold statistics. This avoids building
    multi-million-element boolean training masks repeatedly.
    """
    Ev = E[valid].astype(np.float64, copy=False).ravel()
    Etv = Et[valid].astype(np.float64, copy=False).ravel()
    Lv = L[valid].astype(np.float64, copy=False).ravel()
    yv = y[valid].astype(np.float64, copy=False).ravel()

    n = len(yv)
    if n < max(500, n_folds * 100):
        raise RuntimeError("Too few valid points for random nested CV.")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, n_folds)

    total_full = full_stats(Ev, Lv, yv)
    total_red = reduced_stats(Ev, yv)

    rows = []
    for fold_idx, test_idx in enumerate(folds):
        test_full = full_stats(Ev[test_idx], Lv[test_idx], yv[test_idx])
        test_red = reduced_stats(Ev[test_idx], yv[test_idx])

        b_full = solve_full_stats(total_full - test_full)
        b_red = solve_reduced_stats(total_red - test_red)

        full_logit, full_pde = evaluate_regression_vectors(
            Ev[test_idx], Etv[test_idx], Lv[test_idx], yv[test_idx],
            tau, beta_full=b_full
        )
        red_logit, red_pde = evaluate_regression_vectors(
            Ev[test_idx], Etv[test_idx], Lv[test_idx], yv[test_idx],
            tau, beta_reduced=b_red
        )

        logit_imp = 100.0 * (red_logit - full_logit) / (red_logit + 1e-12)
        pde_imp = 100.0 * (red_pde - full_pde) / (red_pde + 1e-12)

        rows.append(
            {
                "fold": fold_idx,
                "n_test": int(len(test_idx)),
                "D_train": float(b_full[1]),
                "w_full_train": float(b_full[0]),
                "h_full_train": float(b_full[2]),
                "w_reduced_train": float(b_red[0]),
                "h_reduced_train": float(b_red[1]),
                "full_logit_rmse": full_logit,
                "reduced_logit_rmse": red_logit,
                "logit_improvement_pct": logit_imp,
                "full_pde_rmse": full_pde,
                "reduced_pde_rmse": red_pde,
                "pde_improvement_pct": pde_imp,
            }
        )

    return rows


def temporal_block_bootstrap_D(E, L, y, valid, block_frames, reps, seed):
    blocks = []
    n_frames = E.shape[0]

    for start in range(0, n_frames, block_frames):
        end = min(n_frames, start + block_frames)
        vb = valid[start:end]
        if int(vb.sum()) < 100:
            continue
        blocks.append(full_stats(E[start:end][vb], L[start:end][vb], y[start:end][vb]))

    if len(blocks) < 5:
        raise RuntimeError("Too few temporal blocks for D_E bootstrap.")

    block_stats = np.stack(blocks, axis=0)
    rng = np.random.default_rng(seed)
    B = len(blocks)
    vals = np.empty(reps, dtype=np.float64)

    for r in range(reps):
        draw = rng.integers(0, B, size=B)
        beta = solve_full_stats(block_stats[draw].sum(axis=0))
        vals[r] = beta[1]

    return {
        "n_blocks": B,
        "mean": float(vals.mean()),
        "sd": float(vals.std(ddof=1)),
        "ci95_low": float(np.quantile(vals, 0.025)),
        "ci95_high": float(np.quantile(vals, 0.975)),
        "fraction_positive": float(np.mean(vals > 0.0)),
    }


def smoothing_D_estimates(E_scaled, fps, l_fov, tau, eps):
    """
    Return D estimates for half / primary / double spatial smoothing.
    Temporal sigma remains 2 frames.
    """
    _, _, W = E_scaled.shape
    dx = l_fov / (W - 1)
    primary_px = PRIMARY_SIGMA_XY_MM / dx

    results = {}
    for label, factor in [
        ("half", 0.5),
        ("primary", 1.0),
        ("double", 2.0),
    ]:
        sigma_px = factor * primary_px
        sm = gaussian_filter(
            E_scaled,
            sigma=(2.0, sigma_px, sigma_px),
            mode="reflect",
        )
        E0, Et, L, q, y, valid = compute_pseudo_fields(sm, fps, l_fov, tau, eps)
        beta = fit_full(E0, L, y, valid)
        results[label] = {
            "D": float(beta[1]),
            "w": float(beta[0]),
            "h": float(beta[2]),
            "sigma_px": float(sigma_px),
            "sigma_mm": float(factor * PRIMARY_SIGMA_XY_MM),
        }
        del sm, E0, Et, L, q, y, valid

    Ds = np.array([results[k]["D"] for k in ("half", "primary", "double")])
    primary_D = results["primary"]["D"]
    relative_range = float(
        (Ds.max() - Ds.min()) / (abs(primary_D) + 1e-8)
    )
    all_positive = bool(np.all(Ds > 0.0))

    return results, relative_range, all_positive


def select_model(E, Et, L, y, valid, args, chunk_idx):
    """
    Model selection uses random pointwise K-fold validation across the full
    chunk, matching the requested interpolation-style validation design.
    """
    valid_fraction = float(valid.mean())
    raw_cond, std_cond, corr = regression_condition(E, L, valid)

    b_full = fit_full(E, L, y, valid)
    b_red = fit_reduced(E, y, valid)

    cv_rows = nested_random_cv(
        E, Et, L, y, valid,
        args.tau,
        args.nested_folds,
        args.seed + 5000 + chunk_idx,
    )

    if not cv_rows:
        raise RuntimeError("Nested CV produced no valid folds.")

    logit_imps = np.array([r["logit_improvement_pct"] for r in cv_rows])
    pde_imps = np.array([r["pde_improvement_pct"] for r in cv_rows])

    mean_logit_imp = float(logit_imps.mean())
    mean_pde_imp = float(pde_imps.mean())
    frac_logit_positive = float(np.mean(logit_imps > 0.0))
    frac_pde_positive = float(np.mean(pde_imps > 0.0))

    boot = temporal_block_bootstrap_D(
        E, L, y, valid,
        args.bootstrap_block_frames,
        args.bootstrap_reps,
        args.seed + 10000 + chunk_idx,
    )

    ci_excludes_zero_positive = bool(boot["ci95_low"] > 0.0)
    w_in_bounds = bool(args.w_min <= float(b_full[0]) <= args.w_max)

    return {
        "full_beta": b_full,
        "reduced_beta": b_red,
        "cv_rows": cv_rows,
        "valid_fraction": valid_fraction,
        "condition_raw": raw_cond,
        "condition_standardized": std_cond,
        "corr_E_laplacian": corr,
        "mean_logit_improvement_pct": mean_logit_imp,
        "mean_pde_improvement_pct": mean_pde_imp,
        "fraction_logit_folds_positive": frac_logit_positive,
        "fraction_pde_folds_positive": frac_pde_positive,
        "bootstrap": boot,
        "ci_excludes_zero_positive": ci_excludes_zero_positive,
        "w_in_bounds": w_in_bounds,
    }


# ---------------------------------------------------------------------------
# Neural state model
# ---------------------------------------------------------------------------

class DecoupledFourierEmbedding(nn.Module):
    def __init__(self, ff_dim, scale_xy, scale_t):
        super().__init__()
        B_xy = torch.randn(2, ff_dim // 2) * scale_xy
        B_t = torch.randn(1, ff_dim // 2) * scale_t
        self.register_buffer("B_xy", B_xy)
        self.register_buffer("B_t", B_t)

    def forward(self, xyt):
        xy = xyt[:, :2]
        t = xyt[:, 2:3]
        proj_xy = xy @ self.B_xy
        proj_t = t @ self.B_t
        proj = torch.cat([proj_xy, proj_t], dim=-1)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class FixedPhysicsGPINN(nn.Module):
    def __init__(
        self,
        model_kind: str,
        w_EE: float,
        D_E: float,
        h: float,
        ff_dim: int,
        hidden_neurons: int,
        hidden_layers: int,
        scale_xy: float,
        scale_t: float,
    ):
        super().__init__()
        self.model_kind = model_kind

        self.embed_E = DecoupledFourierEmbedding(
            ff_dim=ff_dim,
            scale_xy=scale_xy,
            scale_t=scale_t,
        )

        layers = [nn.Linear(2 * ff_dim, hidden_neurons), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_neurons, hidden_neurons), nn.Tanh()]
        layers += [nn.Linear(hidden_neurons, 1)]
        self.net_E = nn.Sequential(*layers)

        # Physical parameters are buffers, not trainable Parameters.
        self.register_buffer("fixed_w_EE", torch.tensor(float(w_EE), dtype=torch.float32))
        self.register_buffer("fixed_D_E", torch.tensor(float(D_E), dtype=torch.float32))
        self.register_buffer("fixed_h", torch.tensor(float(h), dtype=torch.float32))

    def forward_E(self, xyt):
        return self.net_E(self.embed_E(xyt))


def neural_derivatives(model, xyt, chunk_duration, l_fov, create_graph=True):
    x = xyt.clone().requires_grad_(True)
    E = model.forward_E(x)
    ones = torch.ones_like(E)

    # Always keep the first-derivative graph because the Laplacian needs
    # second derivatives even during evaluation. `create_graph` only controls
    # whether those second derivatives remain differentiable for higher-order
    # training losses.
    g = torch.autograd.grad(
        E, x, ones,
        create_graph=True,
        retain_graph=True,
    )[0]

    dE_dx = g[:, 0:1]
    dE_dy = g[:, 1:2]
    dE_dt_norm = g[:, 2:3]

    d2x = torch.autograd.grad(
        dE_dx, x, ones,
        create_graph=create_graph,
        retain_graph=True,
    )[0][:, 0:1]

    d2y = torch.autograd.grad(
        dE_dy, x, ones,
        create_graph=create_graph,
        retain_graph=create_graph,
    )[0][:, 1:2]

    Et_phys = (2.0 / chunk_duration) * dE_dt_norm
    Lap_phys = 4.0 * (d2x + d2y) / (l_fov ** 2)

    return x, E, Et_phys, Lap_phys


def gpinn_residual_with_grad(model, xyt, tau, chunk_duration, l_fov):
    x, E, Et, Lap = neural_derivatives(
        model, xyt, chunk_duration, l_fov, create_graph=True
    )

    if model.model_kind == "full":
        z = (
            model.fixed_w_EE * E
            + model.fixed_D_E * Lap
            + model.fixed_h
        )
    else:
        z = model.fixed_w_EE * E + model.fixed_h

    rhs = torch.sigmoid(z)
    f = tau * Et - (-E + rhs)

    ones = torch.ones_like(f)
    gf = torch.autograd.grad(
        f, x, ones,
        create_graph=True,
        retain_graph=True,
    )[0]

    return f, gf[:, 0:1], gf[:, 1:2]


def gpinn_residual_only(model, xyt, tau, chunk_duration, l_fov):
    _, E, Et, Lap = neural_derivatives(
        model, xyt, chunk_duration, l_fov, create_graph=True
    )

    if model.model_kind == "full":
        z = (
            model.fixed_w_EE * E
            + model.fixed_D_E * Lap
            + model.fixed_h
        )
    else:
        z = model.fixed_w_EE * E + model.fixed_h

    rhs = torch.sigmoid(z)
    return tau * Et - (-E + rhs)


# ---------------------------------------------------------------------------
# Coordinate / pool construction
# ---------------------------------------------------------------------------

def indices_to_coords(ti, yi, xi, T, H, W):
    x = -1.0 + 2.0 * xi.astype(np.float32) / float(W - 1)
    y = -1.0 + 2.0 * yi.astype(np.float32) / float(H - 1)
    t = -1.0 + 2.0 * ti.astype(np.float32) / float(T - 1)
    return np.column_stack([x, y, t]).astype(np.float32)


def random_state_split(E_scaled, n_train, n_val, seed):
    """
    Reproduce the original random pointwise split philosophy:
    points are sampled across the entire space-time grid.
    """
    T, H, W = E_scaled.shape
    rng = np.random.default_rng(seed)

    n_total = T * H * W
    n_train = min(int(n_train), int(0.8 * n_total))
    n_val = min(int(n_val), n_total - n_train)

    perm = rng.permutation(n_total)
    idx_tr = perm[:n_train]
    idx_va = perm[n_train:n_train + n_val]

    hw = H * W

    def decode(flat):
        ti = flat // hw
        rem = flat % hw
        yi = rem // W
        xi = rem % W
        X = indices_to_coords(ti, yi, xi, T, H, W)
        Y = E_scaled[ti, yi, xi].astype(np.float32)[:, None]
        return X, Y

    Xtr, Ytr = decode(idx_tr)
    Xva, Yva = decode(idx_va)
    return Xtr, Ytr, Xva, Yva


def sample_colloc_pool(shape, n, seed):
    """
    Random collocation points across the full space-time volume, as in the
    original gPINN. These are independent from the supervised random split.
    """
    T, H, W = shape
    rng = np.random.default_rng(seed)

    n_total = T * H * W
    n = min(int(n), n_total)
    flat = rng.choice(n_total, size=n, replace=False)

    hw = H * W
    ti = flat // hw
    rem = flat % hw
    yi = rem // W
    xi = rem % W
    return indices_to_coords(ti, yi, xi, T, H, W)


def random_derivative_split(
    E0, Et, Lap, valid, full_shape,
    n_train, n_val, seed
):
    """
    Randomly split valid interior pseudo-derivative points across the complete
    chunk. Validation points are excluded from derivative-supervision training.
    """
    T, H, W = full_shape
    positions = np.argwhere(valid)
    if len(positions) == 0:
        raise RuntimeError("No valid pseudo-derivative points.")

    rng = np.random.default_rng(seed)
    n_train = min(int(n_train), len(positions))
    n_val = min(int(n_val), len(positions) - n_train)

    chosen = rng.choice(
        len(positions),
        size=n_train + n_val,
        replace=False,
    )
    tr = positions[chosen[:n_train]]
    va = positions[chosen[n_train:n_train + n_val]]

    def make(pos):
        ti_i = pos[:, 0]
        yi_i = pos[:, 1]
        xi_i = pos[:, 2]

        # Interior pseudo-field index -> original frame/pixel index.
        ti = ti_i + 1
        yi = yi_i + 1
        xi = xi_i + 1

        X = indices_to_coords(ti, yi, xi, T, H, W)
        Et_tgt = Et[ti_i, yi_i, xi_i].astype(np.float32)[:, None]
        Lap_tgt = Lap[ti_i, yi_i, xi_i].astype(np.float32)[:, None]
        E_tgt = E0[ti_i, yi_i, xi_i].astype(np.float32)[:, None]
        return X, Et_tgt, Lap_tgt, E_tgt

    Xtr, Ettr, Laptr, Etr = make(tr)
    Xva, Etva, Lapva, Eva = make(va)

    return Xtr, Ettr, Laptr, Etr, Xva, Etva, Lapva, Eva


def to_device_tensors(device, *arrays):
    return [
        torch.tensor(a, dtype=torch.float32, device=device)
        for a in arrays
    ]


# ---------------------------------------------------------------------------
# Neural training
# ---------------------------------------------------------------------------

def minibatch(pool_size, batch_size, device):
    return torch.randint(
        0, pool_size,
        size=(min(batch_size, pool_size),),
        device=device,
    )


def normalized_mse(pred, target):
    denom = torch.var(target, unbiased=False) + 1e-8
    return torch.mean((pred - target) ** 2) / denom


def train_with_continuity(
    model,
    Xd, Yd,
    Xc,
    Xder, Et_tgt, Lap_tgt,
    Xb, Yb,
    lambda_cont,
    args,
):
    """
    Stage A exactly matches the validated pipeline (state-only).

    Stage B adds only:
        lambda_cont * MSE(E_current(t=0), E_previous(t=end))

    Therefore lambda_cont=0 is the clean baseline.
    """
    history = {
        "step": [],
        "phase": [],
        "total": [],
        "data": [],
        "physics": [],
        "gradient": [],
        "derivative": [],
        "continuity": [],
    }

    # Stage A: unchanged state-only pretraining
    optA = torch.optim.Adam(model.parameters(), lr=args.lr_stageA)
    model.train()

    for step in range(args.stageA_steps):
        idd = minibatch(len(Xd), args.data_batch, Xd.device)

        optA.zero_grad(set_to_none=True)
        pred = model.forward_E(Xd[idd])
        data_loss = torch.mean((pred - Yd[idd]) ** 2)
        data_loss.backward()
        optA.step()

        if step % 100 == 0 or step == args.stageA_steps - 1:
            history["step"].append(step)
            history["phase"].append("A_state")
            history["total"].append(float(data_loss.item()))
            history["data"].append(float(data_loss.item()))
            history["physics"].append(np.nan)
            history["gradient"].append(np.nan)
            history["derivative"].append(np.nan)
            history["continuity"].append(np.nan)

    # Stage B: same fixed-physics training + continuity term
    optB = torch.optim.Adam(model.parameters(), lr=args.lr_stageB)

    for step in range(args.stageB_steps):
        idd = minibatch(len(Xd), args.data_batch, Xd.device)
        ic = minibatch(len(Xc), args.colloc_batch, Xc.device)
        ir = minibatch(len(Xder), args.deriv_batch, Xder.device)
        ib = minibatch(len(Xb), args.continuity_batch, Xb.device)

        optB.zero_grad(set_to_none=True)

        pred = model.forward_E(Xd[idd])
        data_loss = torch.mean((pred - Yd[idd]) ** 2)

        f, dfx, dfy = gpinn_residual_with_grad(
            model, Xc[ic], args.tau, args.chunk_duration, args.l_fov
        )
        phys_loss = torch.mean(f ** 2)
        grad_loss = torch.mean(dfx ** 2) + torch.mean(dfy ** 2)

        _, _, Et_pred, Lap_pred = neural_derivatives(
            model,
            Xder[ir],
            args.chunk_duration,
            args.l_fov,
            create_graph=True,
        )
        loss_et = normalized_mse(Et_pred, Et_tgt[ir])

        if model.model_kind == "full":
            deriv_loss = loss_et + normalized_mse(Lap_pred, Lap_tgt[ir])
        else:
            deriv_loss = loss_et

        cont_pred = model.forward_E(Xb[ib])
        cont_loss = torch.mean((cont_pred - Yb[ib]) ** 2)

        total = (
            data_loss
            + args.lambda_physics * phys_loss
            + args.lambda_physics * args.mu_grad * grad_loss
            + args.lambda_deriv * deriv_loss
            + lambda_cont * cont_loss
        )

        total.backward()
        optB.step()

        if step % 100 == 0 or step == args.stageB_steps - 1:
            history["step"].append(args.stageA_steps + step)
            history["phase"].append("B_fixed_physics_cont")
            history["total"].append(float(total.item()))
            history["data"].append(float(data_loss.item()))
            history["physics"].append(float(phys_loss.item()))
            history["gradient"].append(float(grad_loss.item()))
            history["derivative"].append(float(deriv_loss.item()))
            history["continuity"].append(float(cont_loss.item()))

    # Same L-BFGS polish, now including continuity.
    if args.lbfgs_steps > 0:
        n_d = min(8192, len(Xd))
        n_c = min(8192, len(Xc))
        n_r = min(2048, len(Xder))
        n_b = min(len(Xb), 4096)

        idd = torch.randperm(len(Xd), device=Xd.device)[:n_d]
        ic = torch.randperm(len(Xc), device=Xc.device)[:n_c]
        ir = torch.randperm(len(Xder), device=Xder.device)[:n_r]
        ib = torch.randperm(len(Xb), device=Xb.device)[:n_b]

        optL = torch.optim.LBFGS(
            model.parameters(),
            max_iter=args.lbfgs_steps,
            history_size=50,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optL.zero_grad(set_to_none=True)

            pred = model.forward_E(Xd[idd])
            data_loss = torch.mean((pred - Yd[idd]) ** 2)

            f, dfx, dfy = gpinn_residual_with_grad(
                model, Xc[ic], args.tau, args.chunk_duration, args.l_fov
            )
            phys_loss = torch.mean(f ** 2)
            grad_loss = torch.mean(dfx ** 2) + torch.mean(dfy ** 2)

            _, _, Et_pred, Lap_pred = neural_derivatives(
                model,
                Xder[ir],
                args.chunk_duration,
                args.l_fov,
                create_graph=True,
            )
            loss_et = normalized_mse(Et_pred, Et_tgt[ir])
            if model.model_kind == "full":
                deriv_loss = loss_et + normalized_mse(Lap_pred, Lap_tgt[ir])
            else:
                deriv_loss = loss_et

            cont_pred = model.forward_E(Xb[ib])
            cont_loss = torch.mean((cont_pred - Yb[ib]) ** 2)

            total = (
                data_loss
                + args.lambda_physics * phys_loss
                + args.lambda_physics * args.mu_grad * grad_loss
                + args.lambda_deriv * deriv_loss
                + lambda_cont * cont_loss
            )
            total.backward()
            return total

        optL.step(closure)

    return history


# ---------------------------------------------------------------------------
# Random-pointwise interpolation evaluation
# ---------------------------------------------------------------------------

def evaluate_state_points(model, X_val, Y_val, device, batch_size):
    """
    Evaluate on the random pointwise validation set.
    """
    y_true = np.asarray(Y_val, dtype=np.float64).ravel()
    preds = []

    model.eval()
    for start in range(0, len(X_val), batch_size):
        xb = torch.tensor(
            X_val[start:start + batch_size],
            dtype=torch.float32,
            device=device,
        )
        with torch.no_grad():
            pb = model.forward_E(xb).detach().cpu().numpy().ravel()
        preds.append(pb.astype(np.float64))

    y_pred = np.concatenate(preds)
    mse = float(np.mean((y_pred - y_true) ** 2))

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / (ss_tot + 1e-12))

    if np.std(y_pred) < 1e-12 or np.std(y_true) < 1e-12:
        r = float("nan")
    else:
        r = float(pearsonr(y_pred, y_true)[0])

    model.train()
    return mse, r2, r


def evaluate_derivatives(
    model,
    X,
    Et_true,
    Lap_true,
    device,
    args,
    batch=512,
):
    sum_et = 0.0
    sum_lap = 0.0
    n = 0

    for start in range(0, len(X), batch):
        xb = torch.tensor(
            X[start:start + batch],
            dtype=torch.float32,
            device=device,
        )
        _, _, etp, lapp = neural_derivatives(
            model,
            xb,
            args.chunk_duration,
            args.l_fov,
            create_graph=False,
        )

        etp = etp.detach().cpu().numpy().astype(np.float64)
        lapp = lapp.detach().cpu().numpy().astype(np.float64)

        ett = Et_true[start:start + len(etp)].astype(np.float64)
        lapt = Lap_true[start:start + len(etp)].astype(np.float64)

        sum_et += float(np.sum((etp - ett) ** 2))
        sum_lap += float(np.sum((lapp - lapt) ** 2))
        n += len(etp)

    return float(np.sqrt(sum_et / n)), float(np.sqrt(sum_lap / n))


def evaluate_neural_pde(model, X, device, args, batch=256):
    ss = 0.0
    n = 0

    for start in range(0, len(X), batch):
        xb = torch.tensor(
            X[start:start + batch],
            dtype=torch.float32,
            device=device,
        )
        f = gpinn_residual_only(
            model,
            xb,
            args.tau,
            args.chunk_duration,
            args.l_fov,
        )
        fv = f.detach().cpu().numpy().ravel().astype(np.float64)
        ss += float(np.sum(fv * fv))
        n += len(fv)

    return float(np.sqrt(ss / n))


def pseudo_pde_rmse(E, Et, Lap, model_kind, w, D, h, tau):
    if model_kind == "full":
        z = w * E + D * Lap + h
    else:
        z = w * E + h
    rhs = -E + 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))
    f = tau * Et - rhs
    return float(np.sqrt(np.mean(f ** 2)))


def load_old_results(path):
    p = Path(path)
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    needed = ["D_E", "w_EE", "h"]
    if not all(k in d for k in needed):
        return None
    out = {k: np.asarray(d[k]) for k in needed}
    for k in ["val_mse", "val_r2", "val_r"]:
        if k in d:
            out[k] = np.asarray(d[k])
    return out


def write_csv(path, rows):
    if not rows:
        return
    fields = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def json_safe(obj):
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    return obj



# ---------------------------------------------------------------------------
# Sequential-weight helpers
# ---------------------------------------------------------------------------

def update_fixed_physics(model, model_kind, w_EE, D_E, h):
    """
    Update only the CURRENT chunk's fixed physical coefficients.

    The neural state weights and Fourier matrices are left untouched. This is
    the core separation in the new sequential method:
        theta_(i-1) -> theta_i
    while
        (w_i, D_i, h_i)
    come independently from chunk i's regression/model-selection evidence.
    """
    model.model_kind = str(model_kind)
    with torch.no_grad():
        model.fixed_w_EE.fill_(float(w_EE))
        model.fixed_D_E.fill_(float(D_E))
        model.fixed_h.fill_(float(h))


def evaluate_full_frame(model, t_index, shape, device, batch_size=8192):
    """Predict one complete HxW frame at an original-frame time index."""
    T, H, W = shape
    yi, xi = np.meshgrid(
        np.arange(H, dtype=np.int64),
        np.arange(W, dtype=np.int64),
        indexing="ij",
    )
    ti = np.full(H * W, int(t_index), dtype=np.int64)
    X = indices_to_coords(ti, yi.ravel(), xi.ravel(), T, H, W)

    preds = []
    model.eval()
    for start in range(0, len(X), batch_size):
        xb = torch.tensor(
            X[start:start + batch_size],
            dtype=torch.float32,
            device=device,
        )
        with torch.no_grad():
            pb = model.forward_E(xb).detach().cpu().numpy().ravel()
        preds.append(pb.astype(np.float64))
    model.train()
    return np.concatenate(preds).reshape(H, W)


def load_reference_csv(path):
    """Load an optional independent/cold reference CSV keyed by chunk_idx."""
    p = Path(path)
    if not p.exists():
        return {}
    rows = {}
    with open(p, newline="") as f:
        for row in csv.DictReader(f):
            try:
                rows[int(row["chunk_idx"])] = row
            except Exception:
                continue
    return rows


def as_float_or_nan(row, key):
    try:
        return float(row[key])
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    history_dir = output_dir / "histories"
    detail_dir = output_dir / "model_selection_details"

    for d in [output_dir, ckpt_dir, history_dir, detail_dir]:
        d.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print(f"Sequential chunks: {args.chunks}", flush=True)
    print(
        "SEQUENTIAL RULE: carry neural state weights/Fourier matrices only; "
        "physical coefficients and optimizer states are NOT carried.",
        flush=True,
    )
    print(
        f"CONTINUITY RULE: lambda_cont={args.lambda_cont:g} from chunk 1 onward; "
        "chunk 0 has no continuity penalty.",
        flush=True,
    )

    gmin, gmax, nscale = get_global_min_max(
        data_dir,
        args.subject_prefix,
        args.num_chunks_for_scaling,
    )
    scale = gmax - gmin + 1e-8

    print(
        f"Session-global normalization: min={gmin:.8g}, max={gmax:.8g} "
        f"from {nscale} chunks",
        flush=True,
    )
    print(
        f"Physical spatial smoothing scale: {PRIMARY_SIGMA_XY_MM:.6f} mm",
        flush=True,
    )

    old = load_old_results(args.old_results)
    independent_ref = load_reference_csv(args.independent_results_csv)
    if independent_ref:
        print(
            f"Loaded matched independent reference for chunks: "
            f"{sorted(independent_ref.keys())}",
            flush=True,
        )
    else:
        print("No independent-reference CSV found; matched comparison will be skipped.", flush=True)

    summary_rows = []

    # The single neural state model persists across chunks.
    model = None
    prev_end_pred = None
    prev_end_obs = None
    prev_chunk_idx = None

    for seq_pos, chunk_idx in enumerate(args.chunks):
        print("\n" + "=" * 78, flush=True)
        print(f"CHUNK {chunk_idx:02d}  | sequential position {seq_pos}", flush=True)
        print("=" * 78, flush=True)

        fpath = chunk_path(data_dir, args.subject_prefix, chunk_idx)
        if not fpath.exists():
            print(f"Missing {fpath}; skipping.", flush=True)
            continue

        raw = np.load(fpath)["dfof"].astype(np.float64)
        E_scaled = (raw - gmin) / scale
        T, H, W = E_scaled.shape

        dx_mm = args.l_fov / (W - 1)
        sigma_xy_px = PRIMARY_SIGMA_XY_MM / dx_mm

        print(
            f"Shape={E_scaled.shape}; dx={dx_mm:.6f} mm/pixel; "
            f"primary sigma_xy={sigma_xy_px:.4f} pixels",
            flush=True,
        )

        # ------------------------------------------------------------------
        # 1) Independent physical inference/model selection for THIS chunk
        # ------------------------------------------------------------------
        E_sm = gaussian_filter(
            E_scaled,
            sigma=(2.0, sigma_xy_px, sigma_xy_px),
            mode="reflect",
        )
        E0, Et, Lap, q, ylogit, valid = compute_pseudo_fields(
            E_sm,
            args.fps,
            args.l_fov,
            args.tau,
            args.eps_logit,
        )

        print(
            "Validation design: RANDOM POINTWISE split across the full space-time chunk "
            "(interpolation, not temporal forecasting).",
            flush=True,
        )

        select = select_model(
            E0,
            Et,
            Lap,
            ylogit,
            valid,
            args,
            chunk_idx,
        )

        smooth_results, smooth_rel_range, smooth_all_positive = (
            smoothing_D_estimates(
                E_scaled,
                args.fps,
                args.l_fov,
                args.tau,
                args.eps_logit,
            )
        )

        b_full = select["full_beta"]
        b_red = select["reduced_beta"]
        boot = select["bootstrap"]

        criteria = {
            "valid_fraction": (
                select["valid_fraction"] >= args.min_valid_fraction
            ),
            "conditioning": (
                select["condition_standardized"]
                <= args.max_standardized_condition
            ),
            "D_point_positive": bool(float(b_full[1]) > 0.0),
            "D_CI95_positive": bool(select["ci_excludes_zero_positive"]),
            "logit_mean_improvement": (
                select["mean_logit_improvement_pct"]
                >= args.min_logit_improvement_pct
            ),
            "pde_mean_improvement": (
                select["mean_pde_improvement_pct"]
                >= args.min_pde_improvement_pct
            ),
            "logit_fold_consistency": (
                select["fraction_logit_folds_positive"]
                >= args.min_positive_fold_fraction
            ),
            "pde_fold_consistency": (
                select["fraction_pde_folds_positive"]
                >= args.min_positive_fold_fraction
            ),
            "smoothing_all_positive": smooth_all_positive,
            "smoothing_relative_range": (
                smooth_rel_range <= args.max_smoothing_relative_range
            ),
            "w_in_bounds": select["w_in_bounds"],
        }

        use_full = bool(all(criteria.values()))
        selected_model = "full" if use_full else "reduced"

        if use_full:
            w_hat = float(b_full[0])
            D_hat = float(b_full[1])
            h_hat = float(b_full[2])
        else:
            w_hat = float(b_red[0])
            D_hat = 0.0
            h_hat = float(b_red[1])

        print("\n  MODEL-SELECTION EVIDENCE", flush=True)
        print(
            f"    full OLS:    w={b_full[0]:.6f}, "
            f"D={b_full[1]:+.6f}, h={b_full[2]:.6f}",
            flush=True,
        )
        print(
            f"    reduced OLS: w={b_red[0]:.6f}, "
            f"D=0, h={b_red[1]:.6f}",
            flush=True,
        )
        print(
            f"    D bootstrap 95% CI = "
            f"[{boot['ci95_low']:+.6f}, {boot['ci95_high']:+.6f}]",
            flush=True,
        )
        print(
            f"    random nested-CV logit improvement = "
            f"{select['mean_logit_improvement_pct']:+.4f}%",
            flush=True,
        )
        print(
            f"    random nested-CV PDE improvement   = "
            f"{select['mean_pde_improvement_pct']:+.4f}%",
            flush=True,
        )
        print(
            f"    smoothing D: half={smooth_results['half']['D']:+.6f}, "
            f"primary={smooth_results['primary']['D']:+.6f}, "
            f"double={smooth_results['double']['D']:+.6f}",
            flush=True,
        )
        print(f"    criteria: {criteria}", flush=True)
        print(
            f"  >>> SELECTED MODEL: {selected_model.upper()} "
            f"(w={w_hat:.6f}, D={D_hat:+.6f}, h={h_hat:.6f})",
            flush=True,
        )

        selection_detail = {
            "chunk_idx": chunk_idx,
            "selected_model": selected_model,
            "thresholds": {
                "min_valid_fraction": args.min_valid_fraction,
                "max_standardized_condition": args.max_standardized_condition,
                "min_logit_improvement_pct": args.min_logit_improvement_pct,
                "min_pde_improvement_pct": args.min_pde_improvement_pct,
                "min_positive_fold_fraction": args.min_positive_fold_fraction,
                "max_smoothing_relative_range": args.max_smoothing_relative_range,
                "w_bounds": [args.w_min, args.w_max],
            },
            "criteria": criteria,
            "primary_full_beta": {
                "w_EE": float(b_full[0]),
                "D_E": float(b_full[1]),
                "h": float(b_full[2]),
            },
            "primary_reduced_beta": {
                "w_EE": float(b_red[0]),
                "D_E": 0.0,
                "h": float(b_red[1]),
            },
            "bootstrap_D": boot,
            "nested_random_cv": select["cv_rows"],
            "smoothing": smooth_results,
            "smoothing_relative_range": smooth_rel_range,
            "valid_fraction": select["valid_fraction"],
            "condition_raw": select["condition_raw"],
            "condition_standardized": select["condition_standardized"],
            "corr_E_laplacian": select["corr_E_laplacian"],
        }
        (
            detail_dir / f"chunk_{chunk_idx:02d}_selection.json"
        ).write_text(json.dumps(json_safe(selection_detail), indent=2))

        # ------------------------------------------------------------------
        # 2) Random pointwise pools, identical logic to validated 5-chunk run
        # ------------------------------------------------------------------
        Xd_np, Yd_np, Xv_np, Yv_np = random_state_split(
            E_scaled,
            args.n_data_pool,
            args.n_val,
            args.seed + 101 + chunk_idx,
        )

        Xc_np = sample_colloc_pool(
            E_scaled.shape,
            args.n_colloc_pool,
            args.seed + 202 + chunk_idx,
        )

        Xr_np, Etr_np, Lapr_np, Er_np, Xrv_np, Etrv_np, Laprv_np, Erv_np = (
            random_derivative_split(
                E0,
                Et,
                Lap,
                valid,
                E_scaled.shape,
                args.n_deriv_pool,
                max(args.eval_deriv_points, args.eval_pde_points),
                args.seed + 303 + chunk_idx,
            )
        )

        Xd, Yd, Xc, Xder, Et_tgt, Lap_tgt = to_device_tensors(
            device,
            Xd_np,
            Yd_np,
            Xc_np,
            Xr_np,
            Etr_np,
            Lapr_np,
        )

        del Xd_np, Yd_np, Xc_np, Xr_np, Etr_np, Lapr_np, Er_np
        gc.collect()

        # ------------------------------------------------------------------
        # 3) Sequential state-weight warm start
        # ------------------------------------------------------------------
        if model is None:
            # Chunk 0 exactly matches the validated independent initialization.
            set_seed(args.seed)
            model = FixedPhysicsGPINN(
                model_kind=selected_model,
                w_EE=w_hat,
                D_E=D_hat,
                h=h_hat,
                ff_dim=args.ff_dim,
                hidden_neurons=args.hidden_neurons,
                hidden_layers=args.hidden_layers,
                scale_xy=args.scale_xy,
                scale_t=args.scale_t,
            ).to(device)
            warm_started = False
            print(
                "  STATE INITIALIZATION: fresh network (first chunk).",
                flush=True,
            )
        else:
            # Keep the state-network/Fourier representation; replace only this
            # chunk's independently inferred fixed physics.
            update_fixed_physics(
                model,
                selected_model,
                w_hat,
                D_hat,
                h_hat,
            )
            warm_started = True
            print(
                f"  STATE INITIALIZATION: warm-started neural weights from chunk "
                f"{prev_chunk_idx:02d}; current physical coefficients replaced independently.",
                flush=True,
            )

        # Build the state-continuity target. The previous trained chunk's
        # final predicted frame is already stored in prev_end_pred. Continuity
        # constrains only E; it never carries or smooths physical parameters.
        if prev_end_pred is None:
            lambda_this_chunk = 0.0
            Xb_np = np.zeros((1, 3), dtype=np.float32)
            Yb_np = np.zeros((1, 1), dtype=np.float32)
        else:
            lambda_this_chunk = float(args.lambda_cont)
            yi_b, xi_b = np.meshgrid(
                np.arange(H, dtype=np.int64),
                np.arange(W, dtype=np.int64),
                indexing="ij",
            )
            ti_b = np.zeros(H * W, dtype=np.int64)
            Xb_np = indices_to_coords(
                ti_b, yi_b.ravel(), xi_b.ravel(), T, H, W
            )
            if prev_end_pred.shape != (H, W):
                raise ValueError(
                    f"Boundary grid mismatch: previous={prev_end_pred.shape}, current={(H, W)}"
                )
            Yb_np = prev_end_pred.ravel().astype(np.float32)[:, None]

        Xb = torch.tensor(Xb_np, dtype=torch.float32, device=device)
        Yb = torch.tensor(Yb_np, dtype=torch.float32, device=device)

        print(
            f"  CONTINUITY TRAINING WEIGHT: {lambda_this_chunk:g}",
            flush=True,
        )

        # Pre-training performance is a direct diagnostic of how useful the
        # inherited neural state representation is on the new chunk.
        pre_mse, pre_r2, pre_r = evaluate_state_points(
            model,
            Xv_np,
            Yv_np,
            device,
            args.eval_state_batch,
        )
        print(
            f"  PRE-TRAIN RANDOM VAL: MSE={pre_mse:.6e} "
            f"R2={pre_r2:.6f} r={pre_r:.6f}",
            flush=True,
        )

        # Fresh optimizer states and matched minibatch RNG every chunk.
        # This ensures "sequential" means inherited network weights only.
        set_seed(args.seed)

        print("\n  TRAINING FIXED-PARAMETER SEQUENTIAL-WEIGHT + CONTINUITY gPINN", flush=True)
        hist = train_with_continuity(
            model,
            Xd, Yd,
            Xc,
            Xder, Et_tgt, Lap_tgt,
            Xb, Yb,
            lambda_this_chunk,
            args,
        )

        np.savez(
            history_dir / f"chunk_{chunk_idx:02d}_history.npz",
            **{
                k: np.array(v, dtype=object if k == "phase" else None)
                for k, v in hist.items()
            },
        )

        # ------------------------------------------------------------------
        # 4) Random-pointwise validation / derivative / physics diagnostics
        # ------------------------------------------------------------------
        state_mse, state_r2, state_r = evaluate_state_points(
            model,
            Xv_np,
            Yv_np,
            device,
            args.eval_state_batch,
        )

        nd = min(args.eval_deriv_points, len(Xrv_np))
        npde = min(args.eval_pde_points, len(Xrv_np))

        et_rmse, lap_rmse = evaluate_derivatives(
            model,
            Xrv_np[:nd],
            Etrv_np[:nd],
            Laprv_np[:nd],
            device,
            args,
        )

        neural_pde_rmse = evaluate_neural_pde(
            model,
            Xrv_np[:npde],
            device,
            args,
        )

        pseudo_test_pde = pseudo_pde_rmse(
            Erv_np[:npde].astype(np.float64),
            Etrv_np[:npde].astype(np.float64),
            Laprv_np[:npde].astype(np.float64),
            selected_model,
            w_hat,
            D_hat,
            h_hat,
            args.tau,
        )

        print("\n  RANDOM POINTWISE VALIDATION", flush=True)
        print(
            f"    state MSE={state_mse:.6e} "
            f"R2={state_r2:.6f} r={state_r:.6f}",
            flush=True,
        )
        print(
            f"    derivative RMSE: Et={et_rmse:.6e}, "
            f"Lap={lap_rmse:.6e}",
            flush=True,
        )
        print(
            f"    neural PDE RMSE={neural_pde_rmse:.6e}; "
            f"pseudo-data PDE RMSE={pseudo_test_pde:.6e}",
            flush=True,
        )

        # ------------------------------------------------------------------
        # 5) Boundary mismatch diagnostic after continuity-regularized training
        # ------------------------------------------------------------------
        start_pred = evaluate_full_frame(
            model, 0, E_scaled.shape, device, args.eval_state_batch
        )
        end_pred = evaluate_full_frame(
            model, T - 1, E_scaled.shape, device, args.eval_state_batch
        )

        if prev_end_pred is None:
            boundary_pred_rmse = float("nan")
            boundary_obs_rmse = float("nan")
            boundary_ratio = float("nan")
        else:
            boundary_pred_rmse = float(
                np.sqrt(np.mean((prev_end_pred - start_pred) ** 2))
            )
            boundary_obs_rmse = float(
                np.sqrt(np.mean((prev_end_obs - E_scaled[0]) ** 2))
            )
            boundary_ratio = float(
                boundary_pred_rmse / (boundary_obs_rmse + 1e-12)
            )
            print(
                f"  BOUNDARY DIAGNOSTIC {prev_chunk_idx:02d}->{chunk_idx:02d}: "
                f"predicted RMSE={boundary_pred_rmse:.6e}, "
                f"observed frame-to-frame RMSE={boundary_obs_rmse:.6e}, "
                f"ratio={boundary_ratio:.3f}",
                flush=True,
            )

        prev_end_pred = end_pred.copy()
        prev_end_obs = E_scaled[-1].copy()
        prev_chunk_idx = chunk_idx

        # ------------------------------------------------------------------
        # 6) Matched comparisons: old original and validated independent run
        # ------------------------------------------------------------------
        old_row = {}
        if old is not None and chunk_idx < len(old["D_E"]):
            old_row = {
                "old_D_E": float(old["D_E"][chunk_idx]),
                "old_w_EE": float(old["w_EE"][chunk_idx]),
                "old_h": float(old["h"][chunk_idx]),
                "old_val_r2_pointwise": (
                    float(old["val_r2"][chunk_idx])
                    if "val_r2" in old and chunk_idx < len(old["val_r2"])
                    else np.nan
                ),
            }

        ref_row = {}
        if chunk_idx in independent_ref:
            ref = independent_ref[chunk_idx]
            ref_r2 = as_float_or_nan(ref, "random_val_state_r2")
            ref_mse = as_float_or_nan(ref, "random_val_state_mse")
            ref_pde = as_float_or_nan(ref, "random_val_neural_pde_rmse")
            ref_row = {
                "independent_ref_state_r2": ref_r2,
                "independent_ref_state_mse": ref_mse,
                "independent_ref_neural_pde_rmse": ref_pde,
                "sequential_minus_independent_r2": float(state_r2 - ref_r2),
                "sequential_minus_independent_mse": float(state_mse - ref_mse),
                "sequential_minus_independent_pde_rmse": float(
                    neural_pde_rmse - ref_pde
                ),
            }
            print(
                f"  MATCHED INDEPENDENT COMPARISON: "
                f"ΔR2(seq-ind)={state_r2 - ref_r2:+.6f}; "
                f"ΔMSE={state_mse - ref_mse:+.3e}; "
                f"ΔPDE_RMSE={neural_pde_rmse - ref_pde:+.3e}",
                flush=True,
            )

        row = {
            "chunk_idx": chunk_idx,
            "time_start_min": 0.5 * chunk_idx,
            "warm_started_state_weights": bool(warm_started),
            "optimizer_state_carried": False,
            "physical_parameters_carried": False,
            "continuity_loss_used": bool(lambda_this_chunk > 0.0),
            "lambda_cont": float(lambda_this_chunk),
            "selected_model": selected_model,
            "selected_w_EE": w_hat,
            "selected_D_E": D_hat,
            "selected_h": h_hat,
            "full_regression_D_E": float(b_full[1]),
            "full_regression_w_EE": float(b_full[0]),
            "full_regression_h": float(b_full[2]),
            "reduced_regression_w_EE": float(b_red[0]),
            "reduced_regression_h": float(b_red[1]),
            "D_boot_ci95_low": boot["ci95_low"],
            "D_boot_ci95_high": boot["ci95_high"],
            "D_boot_fraction_positive": boot["fraction_positive"],
            "nested_logit_improvement_pct": select["mean_logit_improvement_pct"],
            "nested_pde_improvement_pct": select["mean_pde_improvement_pct"],
            "nested_logit_positive_fold_fraction": select["fraction_logit_folds_positive"],
            "nested_pde_positive_fold_fraction": select["fraction_pde_folds_positive"],
            "valid_fraction": select["valid_fraction"],
            "condition_standardized": select["condition_standardized"],
            "smoothing_D_half": smooth_results["half"]["D"],
            "smoothing_D_primary": smooth_results["primary"]["D"],
            "smoothing_D_double": smooth_results["double"]["D"],
            "smoothing_D_relative_range": smooth_rel_range,
            "pretrain_random_val_state_mse": pre_mse,
            "pretrain_random_val_state_r2": pre_r2,
            "pretrain_random_val_state_r": pre_r,
            "random_val_state_mse": state_mse,
            "random_val_state_r2": state_r2,
            "random_val_state_r": state_r,
            "random_val_Et_rmse_vs_pseudo": et_rmse,
            "random_val_Lap_rmse_vs_pseudo": lap_rmse,
            "random_val_neural_pde_rmse": neural_pde_rmse,
            "random_val_pseudodata_pde_rmse": pseudo_test_pde,
            "boundary_predicted_rmse": boundary_pred_rmse,
            "boundary_observed_rmse": boundary_obs_rmse,
            "boundary_pred_to_observed_ratio": boundary_ratio,
            "grid_T": T,
            "grid_H": H,
            "grid_W": W,
            "dx_mm": dx_mm,
            "sigma_xy_pixels": sigma_xy_px,
            "sigma_xy_mm": PRIMARY_SIGMA_XY_MM,
            **{f"criterion_{k}": bool(v) for k, v in criteria.items()},
            **old_row,
            **ref_row,
        }
        summary_rows.append(row)

        torch.save(
            {
                "model_state": model.state_dict(),
                "model_kind": selected_model,
                "fixed_parameters": {
                    "w_EE": w_hat,
                    "D_E": D_hat,
                    "h": h_hat,
                },
                "chunk_idx": chunk_idx,
                "sequential_state": {
                    "warm_started_state_weights": bool(warm_started),
                    "optimizer_state_carried": False,
                    "physical_parameters_carried": False,
                    "continuity_loss_used": bool(lambda_this_chunk > 0.0),
                    "lambda_cont": float(lambda_this_chunk),
                },
                "normalization": {
                    "global_min": gmin,
                    "global_max": gmax,
                },
                "validation": {
                    "type": "random_pointwise",
                    "n_train_state": int(args.n_data_pool),
                    "n_val_state": int(args.n_val),
                    "note": "Interpolation-style validation across the full space-time chunk; not temporal forecasting.",
                },
                "boundary_diagnostic": {
                    "predicted_rmse": boundary_pred_rmse,
                    "observed_rmse": boundary_obs_rmse,
                    "ratio": boundary_ratio,
                },
                "selection_detail": selection_detail,
            },
            ckpt_dir / f"seq_chunk_{chunk_idx:02d}_{selected_model}.pt",
        )

        # Keep `model` alive: its neural state is the warm start for next chunk.
        del Xd, Yd, Xc, Xder, Et_tgt, Lap_tgt, Xb, Yb
        del E_sm, E0, Et, Lap, q, ylogit, valid
        del Xv_np, Yv_np, Xrv_np, Etrv_np, Laprv_np, Erv_np
        del start_pred, end_pred
        del raw, E_scaled
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ----------------------------------------------------------------------
    # Session-wide outputs
    # ----------------------------------------------------------------------
    write_csv(output_dir / "final_pipeline_results.csv", summary_rows)

    if summary_rows:
        n_full = sum(r["selected_model"] == "full" for r in summary_rows)
        n_red = len(summary_rows) - n_full

        boundary_vals = [
            r["boundary_predicted_rmse"]
            for r in summary_rows
            if np.isfinite(r["boundary_predicted_rmse"])
        ]
        boundary_ratios = [
            r["boundary_pred_to_observed_ratio"]
            for r in summary_rows
            if np.isfinite(r["boundary_pred_to_observed_ratio"])
        ]

        matched = [
            r for r in summary_rows
            if "sequential_minus_independent_r2" in r
        ]
        matched_nonzero = [
            r for r in matched if int(r["chunk_idx"]) != 0
        ]

        overall = {
            "purpose": (
                "Final 20-chunk regression-assisted sequential Wilson-Cowan gPINN pipeline with "
                "independent physical inference, model selection, neural-state warm start, "
                "and continuity regularization"
            ),
            "chunks": [int(r["chunk_idx"]) for r in summary_rows],
            "n_full_selected": int(n_full),
            "n_reduced_selected": int(n_red),
            "mean_random_val_state_r2": float(
                np.mean([r["random_val_state_r2"] for r in summary_rows])
            ),
            "mean_random_val_state_mse": float(
                np.mean([r["random_val_state_mse"] for r in summary_rows])
            ),
            "mean_random_val_neural_pde_rmse": float(
                np.mean([r["random_val_neural_pde_rmse"] for r in summary_rows])
            ),
            "mean_pretrain_random_val_state_r2_chunks_after_first": float(
                np.mean(
                    [
                        r["pretrain_random_val_state_r2"]
                        for r in summary_rows[1:]
                    ]
                )
            ) if len(summary_rows) > 1 else None,
            "boundary_diagnostic": {
                "n_boundaries": int(len(boundary_vals)),
                "mean_predicted_boundary_rmse": (
                    float(np.mean(boundary_vals)) if boundary_vals else None
                ),
                "mean_predicted_to_observed_rmse_ratio": (
                    float(np.mean(boundary_ratios)) if boundary_ratios else None
                ),
                "interpretation": (
                    "Boundary mismatch after continuity-regularized training, "
                    "normalized by the natural observed frame-to-frame boundary change."
                ),
            },
            "matched_independent_reference": {
                "available_chunks": [
                    int(r["chunk_idx"]) for r in matched
                ],
                "mean_delta_r2_seq_minus_independent_excluding_chunk0": (
                    float(np.mean(
                        [r["sequential_minus_independent_r2"] for r in matched_nonzero]
                    ))
                    if matched_nonzero else None
                ),
                "mean_delta_mse_seq_minus_independent_excluding_chunk0": (
                    float(np.mean(
                        [r["sequential_minus_independent_mse"] for r in matched_nonzero]
                    ))
                    if matched_nonzero else None
                ),
                "mean_delta_pde_rmse_seq_minus_independent_excluding_chunk0": (
                    float(np.mean(
                        [r["sequential_minus_independent_pde_rmse"] for r in matched_nonzero]
                    ))
                    if matched_nonzero else None
                ),
            },
            "selection_thresholds": {
                "min_valid_fraction": args.min_valid_fraction,
                "max_standardized_condition": args.max_standardized_condition,
                "min_logit_improvement_pct": args.min_logit_improvement_pct,
                "min_pde_improvement_pct": args.min_pde_improvement_pct,
                "min_positive_fold_fraction": args.min_positive_fold_fraction,
                "max_smoothing_relative_range": args.max_smoothing_relative_range,
            },
            "training": {
                "stageA_steps": args.stageA_steps,
                "stageB_steps": args.stageB_steps,
                "lbfgs_steps": args.lbfgs_steps,
                "lambda_physics": args.lambda_physics,
                "lambda_deriv": args.lambda_deriv,
                "mu_grad": args.mu_grad,
                "physical_parameters_trainable": False,
                "neural_state_weights_warm_started": True,
                "fourier_embedding_carried": True,
                "physical_parameters_carried": False,
                "optimizer_state_carried": False,
                "continuity_loss_used_from_chunk_1": True,
                "lambda_cont": float(args.lambda_cont),
            },
            "important_notes": [
                "Each chunk independently estimates its physical coefficients before neural training; parameter trajectories are therefore data-derived rather than optimizer-carried.",
                "Only the neural state representation (including fixed Fourier matrices) is carried from one chunk to the next.",
                "Fresh Adam and L-BFGS optimizers are constructed every chunk; optimizer moments/history are not carried.",
                "Validation uses a random pointwise split across the full space-time chunk and therefore measures interpolation, not forecasting.",
                "Temporal block bootstrap is retained for D_E uncertainty.",
                "A state-continuity penalty with lambda_cont=0.05 is imposed from chunk 1 onward; chunk 0 remains unpenalized.",
                "Continuity constrains only E and does not smooth or carry w_EE, D_E, or h.",
                "Session-global min-max normalization is retained for comparability with the original offline analysis.",
            ],
        }

        (output_dir / "final_pipeline_summary.json").write_text(
            json.dumps(json_safe(overall), indent=2)
        )

        print("\n" + "=" * 78, flush=True)
        print("20-CHUNK SEQUENTIAL-WEIGHT + CONTINUITY SUMMARY", flush=True)
        print("=" * 78, flush=True)
        print(json.dumps(overall, indent=2), flush=True)

    print(f"\nOutputs saved to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
