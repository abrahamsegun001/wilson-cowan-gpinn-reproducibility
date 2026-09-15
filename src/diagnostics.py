"""Real-data diagnostics, validation, and continuity controls.

Each command below is a namespaced consolidation of a script from the validated
reproducibility archive. The numerical implementation is preserved while the
public repository exposes a compact command surface.
"""
from __future__ import annotations

import sys as _sys

# ============================================================================
# independent_validation: source validate_dual_model_regression_assisted_gpinn_randomsplit.py
# ============================================================================
def independent_validation(argv=None):
    """Run this analysis with command-specific CLI arguments."""
    _old_argv = _sys.argv
    _sys.argv = ['validate_dual_model_regression_assisted_gpinn_randomsplit.py'] + list(argv or [])
    try:
        """
        Five-chunk RANDOM-SPLIT validation of a dual-model, regression-assisted Wilson-Cowan gPINN.

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

        This is a VALIDATION script for representative chunks. It is not yet the final
        20-chunk sequential production run. If this validation succeeds, the next
        script should add consecutive-chunk warm starting and continuity loss.

        Important
        ---------
        D_E is an effective Laplacian spatial-coupling coefficient in this reduced
        model, not a direct biological tissue diffusion constant.

        The automatic model selector uses conservative, explicit defaults. They are
        saved with the outputs and can be sensitivity-tested later rather than hidden.
        """

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
                description="Validate dual-model regression-assisted Wilson-Cowan gPINN"
            )

            # Data / output
            p.add_argument("--data_dir", type=str, default="./data/processed")
            p.add_argument(
                "--output_dir",
                type=str,
                default="./outputs/final_dual_model_validation",
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
                default=[0, 5, 10, 15, 19],
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


        def train_fixed_parameter_model(
            model,
            Xd, Yd,
            Xc,
            Xder, Et_tgt, Lap_tgt,
            args,
        ):
            history = {
                "step": [],
                "phase": [],
                "total": [],
                "data": [],
                "physics": [],
                "gradient": [],
                "derivative": [],
            }

            # Stage A: state-only pretraining
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

                    print(
                        f"    Stage A {step:5d}/{args.stageA_steps} "
                        f"state={data_loss.item():.4e}",
                        flush=True,
                    )

            # Stage B: fixed-parameter physics + pseudo-derivative training
            optB = torch.optim.Adam(model.parameters(), lr=args.lr_stageB)

            for step in range(args.stageB_steps):
                idd = minibatch(len(Xd), args.data_batch, Xd.device)
                ic = minibatch(len(Xc), args.colloc_batch, Xc.device)
                ir = minibatch(len(Xder), args.deriv_batch, Xder.device)

                optB.zero_grad(set_to_none=True)

                pred = model.forward_E(Xd[idd])
                data_loss = torch.mean((pred - Yd[idd]) ** 2)

                f, dfx, dfy = gpinn_residual_with_grad(
                    model,
                    Xc[ic],
                    args.tau,
                    args.chunk_duration,
                    args.l_fov,
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
                    loss_lap = normalized_mse(Lap_pred, Lap_tgt[ir])
                    deriv_loss = loss_et + loss_lap
                else:
                    deriv_loss = loss_et

                total = (
                    data_loss
                    + args.lambda_physics * phys_loss
                    + args.lambda_physics * args.mu_grad * grad_loss
                    + args.lambda_deriv * deriv_loss
                )

                total.backward()
                optB.step()

                if step % 100 == 0 or step == args.stageB_steps - 1:
                    history["step"].append(args.stageA_steps + step)
                    history["phase"].append("B_fixed_physics")
                    history["total"].append(float(total.item()))
                    history["data"].append(float(data_loss.item()))
                    history["physics"].append(float(phys_loss.item()))
                    history["gradient"].append(float(grad_loss.item()))
                    history["derivative"].append(float(deriv_loss.item()))

                    print(
                        f"    Stage B {step:5d}/{args.stageB_steps} "
                        f"total={total.item():.4e} "
                        f"data={data_loss.item():.4e} "
                        f"phys={phys_loss.item():.4e} "
                        f"deriv={deriv_loss.item():.4e}",
                        flush=True,
                    )

            # L-BFGS polish on smaller fixed subsets to keep memory predictable.
            if args.lbfgs_steps > 0:
                n_d = min(8192, len(Xd))
                n_c = min(8192, len(Xc))
                n_r = min(2048, len(Xder))

                # Fixed subsets; seeded by torch state for reproducibility.
                idd = torch.randperm(len(Xd), device=Xd.device)[:n_d]
                ic = torch.randperm(len(Xc), device=Xc.device)[:n_c]
                ir = torch.randperm(len(Xder), device=Xder.device)[:n_r]

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
                        model,
                        Xc[ic],
                        args.tau,
                        args.chunk_duration,
                        args.l_fov,
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

                    total = (
                        data_loss
                        + args.lambda_physics * phys_loss
                        + args.lambda_physics * args.mu_grad * grad_loss
                        + args.lambda_deriv * deriv_loss
                    )
                    total.backward()
                    return total

                print(f"    L-BFGS polish: max_iter={args.lbfgs_steps}", flush=True)
                optL.step(closure)

            return history


        # ---------------------------------------------------------------------------
        # Strict temporal-holdout evaluation
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
            print(f"Representative chunks: {args.chunks}", flush=True)

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

            summary_rows = []

            for chunk_idx in args.chunks:
                print("\n" + "=" * 78, flush=True)
                print(f"CHUNK {chunk_idx:02d}", flush=True)
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

                # Primary pseudo-derivatives.
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

                # Model selection uses the full chunk with random pointwise nested CV.
                select = select_model(
                    E0,
                    Et,
                    Lap,
                    ylogit,
                    valid,
                    args,
                    chunk_idx,
                )

                # Smoothing stability is also computed only on the training region.
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
                    f"    nested held-out logit improvement = "
                    f"{select['mean_logit_improvement_pct']:+.4f}%",
                    flush=True,
                )
                print(
                    f"    nested held-out PDE improvement   = "
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

                # Reinitialize the same architecture with the same random seed per chunk.
                # This keeps RFF matrices / initial neural weights fixed across the five
                # representative chunks so between-chunk differences are not caused by
                # different initialization.
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

                # Original-style random pointwise state split across the FULL chunk.
                Xd_np, Yd_np, Xv_np, Yv_np = random_state_split(
                    E_scaled,
                    args.n_data_pool,
                    args.n_val,
                    args.seed + 101 + chunk_idx,
                )

                # Physics collocation also spans the full domain, matching the original.
                Xc_np = sample_colloc_pool(
                    E_scaled.shape,
                    args.n_colloc_pool,
                    args.seed + 202 + chunk_idx,
                )

                # Keep a separate random derivative validation sample.
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

                print("\n  TRAINING FIXED-PARAMETER gPINN", flush=True)
                hist = train_fixed_parameter_model(
                    model,
                    Xd,
                    Yd,
                    Xc,
                    Xder,
                    Et_tgt,
                    Lap_tgt,
                    args,
                )

                np.savez(
                    history_dir / f"chunk_{chunk_idx:02d}_history.npz",
                    **{
                        k: np.array(v, dtype=object if k == "phase" else None)
                        for k, v in hist.items()
                    },
                )

                # Random pointwise state validation.
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

                row = {
                    "chunk_idx": chunk_idx,
                    "time_start_min": 0.5 * chunk_idx,
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
                    "random_val_state_mse": state_mse,
                    "random_val_state_r2": state_r2,
                    "random_val_state_r": state_r,
                    "random_val_Et_rmse_vs_pseudo": et_rmse,
                    "random_val_Lap_rmse_vs_pseudo": lap_rmse,
                    "random_val_neural_pde_rmse": neural_pde_rmse,
                    "random_val_pseudodata_pde_rmse": pseudo_test_pde,
                    "grid_T": T,
                    "grid_H": H,
                    "grid_W": W,
                    "dx_mm": dx_mm,
                    "sigma_xy_pixels": sigma_xy_px,
                    "sigma_xy_mm": PRIMARY_SIGMA_XY_MM,
                    **{f"criterion_{k}": bool(v) for k, v in criteria.items()},
                    **old_row,
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
                        "selection_detail": selection_detail,
                    },
                    ckpt_dir / f"chunk_{chunk_idx:02d}_{selected_model}.pt",
                )

                # Free GPU memory before next chunk.
                del model, Xd, Yd, Xc, Xder, Et_tgt, Lap_tgt
                del E_sm, E0, Et, Lap, q, ylogit, valid
                del Xv_np, Yv_np, Xrv_np, Etrv_np, Laprv_np, Erv_np
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            # Save compact table + JSON summary.
            write_csv(output_dir / "five_chunk_validation_results.csv", summary_rows)

            if summary_rows:
                n_full = sum(r["selected_model"] == "full" for r in summary_rows)
                n_red = len(summary_rows) - n_full

                overall = {
                    "purpose": "Representative-chunk validation of dual-model regression-assisted gPINN",
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
                    },
                    "important_notes": [
                        "Validation uses a random pointwise split across the full space-time chunk, matching the original interpolation-style analysis; it is not a temporal forecasting test.",
                        "The session-global min-max normalization is retained for comparability with the original analysis; this is therefore an offline-session preprocessing choice, not a strictly online scaling scheme.",
                        "Old pointwise-validation R2, if loaded, is now qualitatively comparable in validation type, although training objectives and parameter treatment differ.",
                        "This representative-chunk script does not yet apply sequential warm starting or adjacent-chunk continuity; those should be added only after this random-split validation passes.",
                    ],
                }

                (output_dir / "validation_summary.json").write_text(
                    json.dumps(json_safe(overall), indent=2)
                )

                print("\n" + "=" * 78, flush=True)
                print("FIVE-CHUNK VALIDATION SUMMARY", flush=True)
                print("=" * 78, flush=True)
                print(json.dumps(overall, indent=2), flush=True)

            print(f"\nOutputs saved to: {output_dir}", flush=True)
        return main()
    finally:
        _sys.argv = _old_argv


# ============================================================================
# continuity: source continuity_ablation.py
# ============================================================================
def continuity(argv=None):
    """Run this analysis with command-specific CLI arguments."""
    _old_argv = _sys.argv
    _sys.argv = ['continuity_ablation.py'] + list(argv or [])
    try:
        """
        Continuity-loss ablation for the regression-assisted sequential Wilson-Cowan gPINN.

        Controlled question
        -------------------
        Does an explicit state-continuity penalty reduce chunk-boundary jumps without
        materially degrading random-pointwise reconstruction or PDE consistency?

        Each array task trains ONE current chunk from the SAVED checkpoint of its
        immediately preceding chunk from the completed 20-chunk sequential-weight run.

        Boundaries tested:
            0->1, 4->5, 9->10, 14->15, 18->19

        Continuity weights:
            lambda_cont = 0, 0.01, 0.05, 0.10

        Everything else is held fixed:
        - same regression/model-selection logic;
        - same random pointwise state split;
        - same pseudo-derivative split;
        - same collocation sampling;
        - same architecture;
        - same previous-chunk neural warm start;
        - same fresh optimizer-state rule;
        - same training budget.

        The physical parameters are independently inferred from the CURRENT chunk and
        remain fixed during neural training. Continuity constrains only the state E.
        """

        import argparse
        import gc
        import json
        from pathlib import Path

        import numpy as np
        import torch
        from scipy.ndimage import gaussian_filter

        from .training import (
            PRIMARY_SIGMA_XY_MM,
            FixedPhysicsGPINN,
            chunk_path,
            compute_pseudo_fields,
            evaluate_derivatives,
            evaluate_full_frame,
            evaluate_neural_pde,
            evaluate_state_points,
            get_global_min_max,
            gpinn_residual_with_grad,
            json_safe,
            minibatch,
            neural_derivatives,
            normalized_mse,
            pseudo_pde_rmse,
            random_derivative_split,
            random_state_split,
            sample_colloc_pool,
            select_model,
            set_seed,
            smoothing_D_estimates,
            to_device_tensors,
            update_fixed_physics,
        )


        BOUNDARIES = [(0, 1), (4, 5), (9, 10), (14, 15), (18, 19)]
        LAMBDAS = [0.0, 0.01, 0.05, 0.10]


        def parse_args():
            p = argparse.ArgumentParser(description="Continuity-loss ablation")
            p.add_argument("--task_id", type=int, required=True)
            p.add_argument("--data_dir", type=str, default="./data/processed")
            p.add_argument(
                "--sequential_dir",
                type=str,
                default="./outputs/final_dual_model_sequential20",
            )
            p.add_argument(
                "--output_dir",
                type=str,
                default="./outputs/continuity_ablation",
            )
            p.add_argument("--subject_prefix", type=str, default="sub-GC48_ses-day0")
            p.add_argument("--num_chunks_for_scaling", type=int, default=20)

            p.add_argument("--fps", type=float, default=30.0)
            p.add_argument("--tau", type=float, default=1.0)
            p.add_argument("--l_fov", type=float, default=10.0)
            p.add_argument("--chunk_duration", type=float, default=30.0)
            p.add_argument("--eps_logit", type=float, default=1e-4)

            # Same selector thresholds as validated run
            p.add_argument("--nested_folds", type=int, default=5)
            p.add_argument("--bootstrap_reps", type=int, default=1000)
            p.add_argument("--bootstrap_block_frames", type=int, default=30)
            p.add_argument("--min_valid_fraction", type=float, default=0.90)
            p.add_argument("--max_standardized_condition", type=float, default=10.0)
            p.add_argument("--min_logit_improvement_pct", type=float, default=0.10)
            p.add_argument("--min_pde_improvement_pct", type=float, default=0.10)
            p.add_argument("--min_positive_fold_fraction", type=float, default=0.60)
            p.add_argument("--max_smoothing_relative_range", type=float, default=1.00)
            p.add_argument("--w_min", type=float, default=0.5)
            p.add_argument("--w_max", type=float, default=6.0)

            # Same architecture
            p.add_argument("--ff_dim", type=int, default=128)
            p.add_argument("--hidden_neurons", type=int, default=256)
            p.add_argument("--hidden_layers", type=int, default=6)
            p.add_argument("--scale_xy", type=float, default=25.0)
            p.add_argument("--scale_t", type=float, default=50.0)

            # Same pools
            p.add_argument("--n_data_pool", type=int, default=60000)
            p.add_argument("--n_val", type=int, default=15000)
            p.add_argument("--n_colloc_pool", type=int, default=60000)
            p.add_argument("--n_deriv_pool", type=int, default=20000)
            p.add_argument("--data_batch", type=int, default=4096)
            p.add_argument("--colloc_batch", type=int, default=4096)
            p.add_argument("--deriv_batch", type=int, default=1024)
            p.add_argument("--continuity_batch", type=int, default=2048)

            # Same training schedule
            p.add_argument("--stageA_steps", type=int, default=2000)
            p.add_argument("--stageB_steps", type=int, default=3000)
            p.add_argument("--lr_stageA", type=float, default=1e-3)
            p.add_argument("--lr_stageB", type=float, default=2e-4)
            p.add_argument("--lbfgs_steps", type=int, default=150)
            p.add_argument("--lambda_physics", type=float, default=0.05)
            p.add_argument("--lambda_deriv", type=float, default=1e-3)
            p.add_argument("--mu_grad", type=float, default=1e-5)

            p.add_argument("--eval_state_batch", type=int, default=8192)
            p.add_argument("--eval_deriv_points", type=int, default=12000)
            p.add_argument("--eval_pde_points", type=int, default=12000)
            p.add_argument("--seed", type=int, default=42)
            return p.parse_args()


        def task_to_condition(task_id):
            if task_id < 0 or task_id >= len(BOUNDARIES) * len(LAMBDAS):
                raise ValueError(f"task_id must be 0..{len(BOUNDARIES)*len(LAMBDAS)-1}")
            boundary_idx = task_id // len(LAMBDAS)
            lambda_idx = task_id % len(LAMBDAS)
            prev_chunk, cur_chunk = BOUNDARIES[boundary_idx]
            lam = LAMBDAS[lambda_idx]
            return prev_chunk, cur_chunk, lam


        def load_previous_model(checkpoint_path, device, args):
            ckpt = torch.load(checkpoint_path, map_location=device)
            fixed = ckpt["fixed_parameters"]
            model = FixedPhysicsGPINN(
                model_kind=ckpt["model_kind"],
                w_EE=float(fixed["w_EE"]),
                D_E=float(fixed["D_E"]),
                h=float(fixed["h"]),
                ff_dim=args.ff_dim,
                hidden_neurons=args.hidden_neurons,
                hidden_layers=args.hidden_layers,
                scale_xy=args.scale_xy,
                scale_t=args.scale_t,
            ).to(device)
            model.load_state_dict(ckpt["model_state"])
            return model, ckpt


        def make_boundary_target(prev_model, prev_shape, cur_shape, device, batch_size):
            """
            Target = previous chunk network at its final frame.
            Coordinates for the current chunk are at t=-1 (frame 0).
            """
            prev_end = evaluate_full_frame(
                prev_model, prev_shape[0] - 1, prev_shape, device, batch_size
            )

            T, H, W = cur_shape
            if prev_end.shape != (H, W):
                raise ValueError(
                    f"Boundary grid mismatch: prev={prev_end.shape}, current={(H,W)}"
                )

            yi, xi = np.meshgrid(
                np.arange(H, dtype=np.int64),
                np.arange(W, dtype=np.int64),
                indexing="ij",
            )
            ti = np.zeros(H * W, dtype=np.int64)

            # local import avoids cluttering the public import list
            from .training import indices_to_coords
            X = indices_to_coords(ti, yi.ravel(), xi.ravel(), T, H, W)
            Y = prev_end.ravel().astype(np.float32)[:, None]
            return X, Y, prev_end


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


        def main():
            args = parse_args()
            prev_chunk, chunk_idx, lambda_cont = task_to_condition(args.task_id)
            set_seed(args.seed)

            torch.set_float32_matmul_precision("high")
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

            data_dir = Path(args.data_dir)
            seq_dir = Path(args.sequential_dir)
            out_dir = Path(args.output_dir)
            run_dir = out_dir / "runs"
            hist_dir = out_dir / "histories"
            ckpt_out = out_dir / "checkpoints"
            for d in [out_dir, run_dir, hist_dir, ckpt_out]:
                d.mkdir(parents=True, exist_ok=True)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            print("=" * 78, flush=True)
            print(
                f"CONTINUITY ABLATION task={args.task_id} | "
                f"boundary {prev_chunk:02d}->{chunk_idx:02d} | "
                f"lambda_cont={lambda_cont:g}",
                flush=True,
            )
            print("=" * 78, flush=True)

            gmin, gmax, _ = get_global_min_max(
                data_dir, args.subject_prefix, args.num_chunks_for_scaling
            )
            scale = gmax - gmin + 1e-8

            # Load previous and current chunk data.
            prev_raw = np.load(
                chunk_path(data_dir, args.subject_prefix, prev_chunk)
            )["dfof"].astype(np.float64)
            cur_raw = np.load(
                chunk_path(data_dir, args.subject_prefix, chunk_idx)
            )["dfof"].astype(np.float64)

            prev_scaled = (prev_raw - gmin) / scale
            E_scaled = (cur_raw - gmin) / scale

            T, H, W = E_scaled.shape
            dx_mm = args.l_fov / (W - 1)
            sigma_xy_px = PRIMARY_SIGMA_XY_MM / dx_mm

            # Load exact saved previous-chunk state from the production run.
            candidates = sorted(
                (seq_dir / "checkpoints").glob(f"seq_chunk_{prev_chunk:02d}_*.pt")
            )
            if len(candidates) != 1:
                raise FileNotFoundError(
                    f"Expected exactly one checkpoint for chunk {prev_chunk}, found {candidates}"
                )
            prev_model, prev_ckpt = load_previous_model(candidates[0], device, args)

            # Fixed boundary target before altering physical buffers for current chunk.
            Xb_np, Yb_np, prev_end_pred = make_boundary_target(
                prev_model,
                prev_scaled.shape,
                E_scaled.shape,
                device,
                args.eval_state_batch,
            )

            # Current chunk derivative fields and independent model selection.
            E_sm = gaussian_filter(
                E_scaled,
                sigma=(2.0, sigma_xy_px, sigma_xy_px),
                mode="reflect",
            )
            E0, Et, Lap, q, ylogit, valid = compute_pseudo_fields(
                E_sm, args.fps, args.l_fov, args.tau, args.eps_logit
            )

            select = select_model(
                E0, Et, Lap, ylogit, valid, args, chunk_idx
            )

            smooth_results, smooth_rel_range, smooth_all_positive = (
                smoothing_D_estimates(
                    E_scaled, args.fps, args.l_fov, args.tau, args.eps_logit
                )
            )

            b_full = select["full_beta"]
            b_red = select["reduced_beta"]

            criteria = {
                "valid_fraction": select["valid_fraction"] >= args.min_valid_fraction,
                "conditioning": (
                    select["condition_standardized"] <= args.max_standardized_condition
                ),
                "D_point_positive": bool(float(b_full[1]) > 0.0),
                "D_CI95_positive": bool(select["ci_excludes_zero_positive"]),
                "logit_mean_improvement": (
                    select["mean_logit_improvement_pct"] >= args.min_logit_improvement_pct
                ),
                "pde_mean_improvement": (
                    select["mean_pde_improvement_pct"] >= args.min_pde_improvement_pct
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
                w_hat, D_hat, h_hat = map(float, b_full)
            else:
                w_hat = float(b_red[0])
                D_hat = 0.0
                h_hat = float(b_red[1])

            # Reuse exact previous neural state; replace only current fixed physics.
            model = prev_model
            update_fixed_physics(model, selected_model, w_hat, D_hat, h_hat)

            # Same pools as production run.
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

            Xd, Yd, Xc, Xder, Et_tgt, Lap_tgt, Xb, Yb = to_device_tensors(
                device,
                Xd_np,
                Yd_np,
                Xc_np,
                Xr_np,
                Etr_np,
                Lapr_np,
                Xb_np,
                Yb_np,
            )

            pre_mse, pre_r2, pre_r = evaluate_state_points(
                model, Xv_np, Yv_np, device, args.eval_state_batch
            )

            # Fresh optimizer/minibatch RNG, exactly like production run.
            set_seed(args.seed)

            hist = train_with_continuity(
                model,
                Xd, Yd,
                Xc,
                Xder, Et_tgt, Lap_tgt,
                Xb, Yb,
                lambda_cont,
                args,
            )

            # Random validation.
            state_mse, state_r2, state_r = evaluate_state_points(
                model, Xv_np, Yv_np, device, args.eval_state_batch
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
                model, Xrv_np[:npde], device, args
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

            # Boundary after retraining.
            start_pred = evaluate_full_frame(
                model, 0, E_scaled.shape, device, args.eval_state_batch
            )
            boundary_pred_rmse = float(
                np.sqrt(np.mean((prev_end_pred - start_pred) ** 2))
            )
            boundary_obs_rmse = float(
                np.sqrt(np.mean((prev_scaled[-1] - E_scaled[0]) ** 2))
            )
            boundary_ratio = float(
                boundary_pred_rmse / (boundary_obs_rmse + 1e-12)
            )

            # Mean continuity MSE over the entire boundary.
            with torch.no_grad():
                preds = []
                for s in range(0, len(Xb), args.eval_state_batch):
                    preds.append(
                        model.forward_E(Xb[s:s+args.eval_state_batch])
                        .detach().cpu().numpy()
                    )
                boundary_cont_mse = float(
                    np.mean((np.vstack(preds) - Yb_np) ** 2)
                )

            result = {
                "task_id": int(args.task_id),
                "prev_chunk": int(prev_chunk),
                "chunk_idx": int(chunk_idx),
                "lambda_cont": float(lambda_cont),
                "selected_model": selected_model,
                "selected_w_EE": float(w_hat),
                "selected_D_E": float(D_hat),
                "selected_h": float(h_hat),
                "full_regression_D_E": float(b_full[1]),
                "D_boot_ci95_low": float(select["bootstrap"]["ci95_low"]),
                "D_boot_ci95_high": float(select["bootstrap"]["ci95_high"]),
                "pretrain_random_val_r2": float(pre_r2),
                "random_val_state_mse": float(state_mse),
                "random_val_state_r2": float(state_r2),
                "random_val_state_r": float(state_r),
                "random_val_Et_rmse_vs_pseudo": float(et_rmse),
                "random_val_Lap_rmse_vs_pseudo": float(lap_rmse),
                "random_val_neural_pde_rmse": float(neural_pde_rmse),
                "random_val_pseudodata_pde_rmse": float(pseudo_test_pde),
                "boundary_predicted_rmse": float(boundary_pred_rmse),
                "boundary_observed_rmse": float(boundary_obs_rmse),
                "boundary_pred_to_observed_ratio": float(boundary_ratio),
                "boundary_continuity_mse": float(boundary_cont_mse),
                "grid_H": int(H),
                "grid_W": int(W),
                "sigma_xy_pixels": float(sigma_xy_px),
                "physical_parameters_trainable": False,
                "physical_parameters_carried": False,
                "optimizer_state_carried": False,
                "state_weights_warm_started": True,
            }

            stem = (
                f"boundary_{prev_chunk:02d}_to_{chunk_idx:02d}"
                f"_lambda_{lambda_cont:.3f}"
            ).replace(".", "p")

            (run_dir / f"{stem}.json").write_text(
                json.dumps(json_safe(result), indent=2)
            )
            np.savez(
                hist_dir / f"{stem}_history.npz",
                **{
                    k: np.array(v, dtype=object if k == "phase" else None)
                    for k, v in hist.items()
                },
            )
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "result": result,
                },
                ckpt_out / f"{stem}.pt",
            )

            print("\nRESULT", flush=True)
            print(json.dumps(result, indent=2), flush=True)

            del model, Xd, Yd, Xc, Xder, Et_tgt, Lap_tgt, Xb, Yb
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        return main()
    finally:
        _sys.argv = _old_argv


# ============================================================================
# aggregate_continuity: source aggregate_continuity_ablation.py
# ============================================================================
def aggregate_continuity(argv=None):
    """Run this analysis with command-specific CLI arguments."""
    _old_argv = _sys.argv
    _sys.argv = ['aggregate_continuity_ablation.py'] + list(argv or [])
    try:
        import argparse
        import csv
        import json
        from pathlib import Path

        import numpy as np


        def parse_args():
            p = argparse.ArgumentParser()
            p.add_argument(
                "--input_dir",
                default="./outputs/continuity_ablation/runs",
            )
            p.add_argument(
                "--output_dir",
                default="./outputs/continuity_ablation",
            )
            return p.parse_args()


        def write_csv(path, rows):
            if not rows:
                return
            fields = sorted({k for r in rows for k in r})
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(rows)


        def main():
            args = parse_args()
            inp = Path(args.input_dir)
            out = Path(args.output_dir)
            out.mkdir(parents=True, exist_ok=True)

            rows = [json.loads(p.read_text()) for p in sorted(inp.glob("*.json"))]
            if len(rows) != 20:
                raise RuntimeError(f"Expected 20 ablation result files, found {len(rows)}")

            rows = sorted(rows, key=lambda r: (r["lambda_cont"], r["chunk_idx"]))
            write_csv(out / "continuity_ablation_runs.csv", rows)

            lambdas = sorted({float(r["lambda_cont"]) for r in rows})
            summary = []

            baseline = {
                int(r["chunk_idx"]): r
                for r in rows
                if float(r["lambda_cont"]) == 0.0
            }

            for lam in lambdas:
                rr = [r for r in rows if float(r["lambda_cont"]) == lam]

                ratios = np.array(
                    [r["boundary_pred_to_observed_ratio"] for r in rr],
                    dtype=float,
                )
                r2 = np.array([r["random_val_state_r2"] for r in rr], dtype=float)
                pde = np.array(
                    [r["random_val_neural_pde_rmse"] for r in rr], dtype=float
                )
                mse = np.array([r["random_val_state_mse"] for r in rr], dtype=float)

                delta_r2 = []
                delta_pde = []
                ratio_reduction = []
                for r in rr:
                    b = baseline[int(r["chunk_idx"])]
                    delta_r2.append(
                        float(r["random_val_state_r2"] - b["random_val_state_r2"])
                    )
                    delta_pde.append(
                        float(
                            r["random_val_neural_pde_rmse"]
                            - b["random_val_neural_pde_rmse"]
                        )
                    )
                    ratio_reduction.append(
                        float(
                            b["boundary_pred_to_observed_ratio"]
                            - r["boundary_pred_to_observed_ratio"]
                        )
                    )

                summary.append(
                    {
                        "lambda_cont": lam,
                        "n_boundaries": len(rr),
                        "mean_boundary_ratio": float(ratios.mean()),
                        "median_boundary_ratio": float(np.median(ratios)),
                        "max_boundary_ratio": float(ratios.max()),
                        "fraction_boundary_ratio_le_2": float(np.mean(ratios <= 2.0)),
                        "mean_random_val_r2": float(r2.mean()),
                        "mean_random_val_mse": float(mse.mean()),
                        "mean_neural_pde_rmse": float(pde.mean()),
                        "mean_delta_r2_vs_lambda0": float(np.mean(delta_r2)),
                        "mean_delta_pde_rmse_vs_lambda0": float(np.mean(delta_pde)),
                        "mean_boundary_ratio_reduction_vs_lambda0": float(
                            np.mean(ratio_reduction)
                        ),
                    }
                )

            write_csv(out / "continuity_ablation_summary.csv", summary)

            # A simple transparent recommendation rule:
            # among nonzero lambdas, choose the one with largest boundary-ratio
            # reduction subject to mean R2 loss <= 0.002 and PDE-RMSE increase <= 0.005.
            eligible = [
                s for s in summary
                if s["lambda_cont"] > 0
                and s["mean_delta_r2_vs_lambda0"] >= -0.002
                and s["mean_delta_pde_rmse_vs_lambda0"] <= 0.005
            ]

            recommendation = None
            if eligible:
                recommendation = max(
                    eligible,
                    key=lambda s: s["mean_boundary_ratio_reduction_vs_lambda0"],
                )

            final = {
                "summary_by_lambda": summary,
                "selection_rule": {
                    "max_mean_R2_loss": 0.002,
                    "max_mean_PDE_RMSE_increase": 0.005,
                    "objective": "largest mean boundary-ratio reduction",
                },
                "recommended_nonzero_lambda": (
                    recommendation["lambda_cont"] if recommendation else None
                ),
                "recommended_summary": recommendation,
                "note": (
                    "Recommendation is an ablation decision rule, not a physiological "
                    "claim. Inspect per-boundary behavior before adopting the loss."
                ),
            }

            (out / "continuity_ablation_summary.json").write_text(
                json.dumps(final, indent=2)
            )

            print(json.dumps(final, indent=2))
        return main()
    finally:
        _sys.argv = _old_argv


# ============================================================================
# derivative_diagnostic: source diagnose_derivatives.py
# ============================================================================
def derivative_diagnostic(argv=None):
    """Run this analysis with command-specific CLI arguments."""
    _old_argv = _sys.argv
    _sys.argv = ['diagnose_derivatives.py'] + list(argv or [])
    try:
        """
        Diagnostic for synthetic Wilson-Cowan gPINN parameter recovery.

        This script separates three questions:
        1) Does the PINN reconstruct E accurately?
        2) Does it reconstruct E_t and Laplacian(E) accurately?
        3) Does the synthetic trajectory itself support recovery of (w_EE, D_E, h)?

        It compares finite-difference derivatives of the independently generated
        synthetic truth against autograd derivatives of a trained gPINN checkpoint,
        evaluates PDE residuals under true vs recovered parameters, and performs the
        direct logit-regression identifiability diagnostic.
        """
        import argparse
        import json
        from pathlib import Path

        import numpy as np
        import torch
        import matplotlib.pyplot as plt

        from .model import gPINN_WilsonCowan


        def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            y_true = np.asarray(y_true).ravel()
            y_pred = np.asarray(y_pred).ravel()
            ss_res = float(np.sum((y_true - y_pred) ** 2))
            ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
            return float(1.0 - ss_res / (ss_tot + 1e-12))


        def corrcoef(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            a = np.asarray(y_true).ravel()
            b = np.asarray(y_pred).ravel()
            if np.std(a) == 0 or np.std(b) == 0:
                return float("nan")
            return float(np.corrcoef(a, b)[0, 1])


        def rel_l2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            num = np.linalg.norm(np.asarray(y_pred).ravel() - np.asarray(y_true).ravel())
            den = np.linalg.norm(np.asarray(y_true).ravel()) + 1e-12
            return float(num / den)


        def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
            err = np.asarray(y_pred).ravel() - np.asarray(y_true).ravel()
            return {
                "rmse": float(np.sqrt(np.mean(err ** 2))),
                "mae": float(np.mean(np.abs(err))),
                "r2": r2_score(y_true, y_pred),
                "r": corrcoef(y_true, y_pred),
                "relative_l2": rel_l2(y_true, y_pred),
            }


        def sigmoid_np(z: np.ndarray) -> np.ndarray:
            z = np.clip(z, -40.0, 40.0)
            return 1.0 / (1.0 + np.exp(-z))


        def make_interior_sample(E: np.ndarray, l_fov: float, t_seconds: np.ndarray,
                                 n_samples: int, seed: int):
            """
            Sample only interior time/space indices so centered finite differences
            are available for both E_t and Laplacian(E).
            """
            T, H, W = E.shape
            if T < 3 or H < 3 or W < 3:
                raise ValueError("Need at least 3 samples in every dimension.")

            rng = np.random.default_rng(seed)
            max_possible = (T - 2) * (H - 2) * (W - 2)
            n = min(int(n_samples), max_possible)

            # Sample flattened indices from the interior cuboid.
            flat = rng.choice(max_possible, size=n, replace=False)
            hw = (H - 2) * (W - 2)
            ti = flat // hw + 1
            rem = flat % hw
            yi = rem // (W - 2) + 1
            xi = rem % (W - 2) + 1

            dt = float(np.median(np.diff(t_seconds)))
            dx = float(l_fov) / float(W - 1)
            dy = float(l_fov) / float(H - 1)

            E0 = E[ti, yi, xi].astype(np.float64)

            dE_dt = (
                E[ti + 1, yi, xi].astype(np.float64)
                - E[ti - 1, yi, xi].astype(np.float64)
            ) / (2.0 * dt)

            d2x = (
                E[ti, yi, xi + 1].astype(np.float64)
                - 2.0 * E0
                + E[ti, yi, xi - 1].astype(np.float64)
            ) / (dx * dx)

            d2y = (
                E[ti, yi + 1, xi].astype(np.float64)
                - 2.0 * E0
                + E[ti, yi - 1, xi].astype(np.float64)
            ) / (dy * dy)

            lap = d2x + d2y

            # Exact normalized coordinates used during PINN training.
            x_norm = -1.0 + 2.0 * xi.astype(np.float64) / float(W - 1)
            y_norm = -1.0 + 2.0 * yi.astype(np.float64) / float(H - 1)
            t_norm = -1.0 + 2.0 * ti.astype(np.float64) / float(T - 1)
            xyt = np.stack([x_norm, y_norm, t_norm], axis=1).astype(np.float32)

            return {
                "xyt": xyt,
                "E": E0,
                "dE_dt": dE_dt,
                "lap": lap,
                "indices": np.stack([ti, yi, xi], axis=1),
                "dt": dt,
                "dx": dx,
                "dy": dy,
            }


        def pinn_values_and_derivatives(model, xyt_np: np.ndarray, device: torch.device,
                                        duration: float, l_fov: float, batch_size: int):
            E_all, dt_all, lap_all = [], [], []

            model.eval()
            for start in range(0, len(xyt_np), batch_size):
                xb = torch.from_numpy(xyt_np[start:start + batch_size]).to(device)
                xb = xb.clone().detach().requires_grad_(True)

                E = model.forward_E(xb)
                ones = torch.ones_like(E)

                g = torch.autograd.grad(
                    E, xb, grad_outputs=ones, create_graph=True, retain_graph=True
                )[0]
                dE_dxnorm = g[:, 0:1]
                dE_dynorm = g[:, 1:2]
                dE_dtnorm = g[:, 2:3]

                d2E_dxnorm2 = torch.autograd.grad(
                    dE_dxnorm, xb, grad_outputs=torch.ones_like(dE_dxnorm),
                    create_graph=False, retain_graph=True
                )[0][:, 0:1]

                d2E_dynorm2 = torch.autograd.grad(
                    dE_dynorm, xb, grad_outputs=torch.ones_like(dE_dynorm),
                    create_graph=False, retain_graph=False
                )[0][:, 1:2]

                # Convert normalized-coordinate derivatives back to physical units.
                dE_dt_phys = (2.0 / float(duration)) * dE_dtnorm
                lap_phys = (4.0 / float(l_fov) ** 2) * (
                    d2E_dxnorm2 + d2E_dynorm2
                )

                E_all.append(E.detach().cpu().numpy().ravel())
                dt_all.append(dE_dt_phys.detach().cpu().numpy().ravel())
                lap_all.append(lap_phys.detach().cpu().numpy().ravel())

            return (
                np.concatenate(E_all),
                np.concatenate(dt_all),
                np.concatenate(lap_all),
            )


        def pde_residual(E, dE_dt, lap, D, w, h, tau):
            rhs = -E + sigmoid_np(w * E + D * lap + h)
            return tau * dE_dt - rhs


        def direct_logit_regression(E, dE_dt, lap, tau, eps=1e-5):
            """
            logit(tau*E_t + E) = w_EE*E + D_E*lap(E) + h
            """
            q = tau * dE_dt + E
            valid = np.isfinite(q) & np.isfinite(E) & np.isfinite(lap)
            valid &= (q > eps) & (q < 1.0 - eps)

            qv = q[valid]
            Ev = E[valid]
            Lv = lap[valid]

            y = np.log(qv / (1.0 - qv))
            X = np.column_stack([Ev, Lv, np.ones_like(Ev)])

            beta, residuals, rank, svals = np.linalg.lstsq(X, y, rcond=None)
            w_hat, D_hat, h_hat = map(float, beta)

            # Condition number of raw design matrix and standardized two-variable part.
            cond_raw = float(np.linalg.cond(X))
            X2 = np.column_stack([
                (Ev - Ev.mean()) / (Ev.std() + 1e-12),
                (Lv - Lv.mean()) / (Lv.std() + 1e-12),
            ])
            corr_E_lap = float(np.corrcoef(Ev, Lv)[0, 1])
            cond_standardized = float(np.linalg.cond(X2))

            return {
                "n_valid": int(valid.sum()),
                "fraction_valid": float(valid.mean()),
                "estimate": {"w_EE": w_hat, "D_E": D_hat, "h": h_hat},
                "rank": int(rank),
                "singular_values": [float(x) for x in svals],
                "condition_number_raw_X": cond_raw,
                "condition_number_standardized_E_lap": cond_standardized,
                "corr_E_laplacian": corr_E_lap,
                "residual_sum_squares": float(residuals[0]) if len(residuals) else float("nan"),
            }


        def scatter_plot(x, y, xlabel, ylabel, title, path, max_points=12000):
            rng = np.random.default_rng(123)
            n = len(x)
            if n > max_points:
                idx = rng.choice(n, max_points, replace=False)
                xx, yy = x[idx], y[idx]
            else:
                xx, yy = x, y

            fig, ax = plt.subplots(figsize=(6.5, 6))
            ax.scatter(xx, yy, s=5, alpha=0.25)
            lo = min(float(np.min(xx)), float(np.min(yy)))
            hi = max(float(np.max(xx)), float(np.max(yy)))
            ax.plot([lo, hi], [lo, hi], "--", linewidth=1.2)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            fig.tight_layout()
            fig.savefig(path, dpi=180)
            plt.close(fig)


        def main():
            p = argparse.ArgumentParser()
            p.add_argument("--data", default="synthetic_data/wc_twin_truth.npz")
            p.add_argument(
                "--checkpoint",
                default="synthetic_outputs/checkpoints/seed-42_noise-0p000.pt",
            )
            p.add_argument("--output_dir", default="synthetic_outputs/diagnostics")
            p.add_argument("--n_samples", type=int, default=20000)
            p.add_argument("--sample_seed", type=int, default=909)
            p.add_argument("--batch_size", type=int, default=1024)
            args = p.parse_args()

            out = Path(args.output_dir)
            out.mkdir(parents=True, exist_ok=True)

            d = np.load(args.data, allow_pickle=False)
            E = d["E"].astype(np.float64)
            t_seconds = d["t_seconds"].astype(np.float64)
            D_true = float(d["D_E_true"])
            w_true = float(d["w_EE_true"])
            h_true = float(d["h_true"])
            tau = float(d["tau"])
            l_fov = float(d["l_fov"])
            duration = float(t_seconds[-1] - t_seconds[0])
            # Training normalized time by frame index over a nominal 30 s chunk.
            # For consistency with wc_gpinn_core.gpinn_residual use 30.0 s.
            duration_model = 30.0

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ckpt = torch.load(args.checkpoint, map_location=device)
            fourier_seed = int(ckpt.get("fourier_seed", 2718))

            model = gPINN_WilsonCowan(fourier_seed=fourier_seed).to(device)
            model.load_state_dict(ckpt["model_state"])

            D_est = float(model.forward_D().item())
            w_est = float(model.w_EE.item())
            h_est = float(model.h.item())

            s = make_interior_sample(
                E, l_fov, t_seconds, args.n_samples, args.sample_seed
            )

            E_p, dt_p, lap_p = pinn_values_and_derivatives(
                model, s["xyt"], device, duration_model, l_fov, args.batch_size
            )

            E_t = s["E"]
            dt_t = s["dE_dt"]
            lap_t = s["lap"]

            field_metrics = metrics(E_t, E_p)
            dt_metrics = metrics(dt_t, dt_p)
            lap_metrics = metrics(lap_t, lap_p)

            # Residuals evaluated on the independently generated truth field.
            r_truthfield_true = pde_residual(
                E_t, dt_t, lap_t, D_true, w_true, h_true, tau
            )
            r_truthfield_est = pde_residual(
                E_t, dt_t, lap_t, D_est, w_est, h_est, tau
            )

            # Residuals evaluated on the learned PINN field/derivatives.
            r_pinnfield_true = pde_residual(
                E_p, dt_p, lap_p, D_true, w_true, h_true, tau
            )
            r_pinnfield_est = pde_residual(
                E_p, dt_p, lap_p, D_est, w_est, h_est, tau
            )

            direct = direct_logit_regression(E_t, dt_t, lap_t, tau)

            report = {
                "data": args.data,
                "checkpoint": args.checkpoint,
                "device": str(device),
                "n_samples": int(len(E_t)),
                "truth": {"D_E": D_true, "w_EE": w_true, "h": h_true},
                "checkpoint_estimate": {"D_E": D_est, "w_EE": w_est, "h": h_est},
                "state_E_metrics": field_metrics,
                "time_derivative_Et_metrics": dt_metrics,
                "laplacian_metrics": lap_metrics,
                "pde_residual_mse": {
                    "truth_field_true_params": float(np.mean(r_truthfield_true ** 2)),
                    "truth_field_checkpoint_params": float(np.mean(r_truthfield_est ** 2)),
                    "pinn_field_true_params": float(np.mean(r_pinnfield_true ** 2)),
                    "pinn_field_checkpoint_params": float(np.mean(r_pinnfield_est ** 2)),
                },
                "pde_residual_rmse": {
                    "truth_field_true_params": float(np.sqrt(np.mean(r_truthfield_true ** 2))),
                    "truth_field_checkpoint_params": float(np.sqrt(np.mean(r_truthfield_est ** 2))),
                    "pinn_field_true_params": float(np.sqrt(np.mean(r_pinnfield_true ** 2))),
                    "pinn_field_checkpoint_params": float(np.sqrt(np.mean(r_pinnfield_est ** 2))),
                },
                "direct_logit_regression_on_truth_field": direct,
                "grid": {
                    "dt_seconds": float(s["dt"]),
                    "dx_mm": float(s["dx"]),
                    "dy_mm": float(s["dy"]),
                    "nominal_model_duration_seconds": duration_model,
                    "stored_time_span_seconds": duration,
                },
            }

            tag = Path(args.checkpoint).stem
            json_path = out / f"{tag}_derivative_diagnostic.json"
            json_path.write_text(json.dumps(report, indent=2))

            scatter_plot(
                E_t, E_p,
                "Synthetic truth E", "gPINN E",
                f"State reconstruction ({tag})",
                out / f"{tag}_scatter_E.png",
            )
            scatter_plot(
                dt_t, dt_p,
                "Finite-difference truth E_t", "gPINN autograd E_t",
                f"Time-derivative accuracy ({tag})",
                out / f"{tag}_scatter_Et.png",
            )
            scatter_plot(
                lap_t, lap_p,
                "Finite-difference truth Laplacian(E)", "gPINN autograd Laplacian(E)",
                f"Laplacian accuracy ({tag})",
                out / f"{tag}_scatter_laplacian.png",
            )

            print(json.dumps(report, indent=2))
            print(f"\nSaved report: {json_path}")
            print(f"Saved figures in: {out}")
        return main()
    finally:
        _sys.argv = _old_argv


# ============================================================================
# d_bound_audit: source real_calcium_D_bound_audit.py
# ============================================================================
def d_bound_audit(argv=None):
    """Run this analysis with command-specific CLI arguments."""
    _old_argv = _sys.argv
    _sys.argv = ['real_calcium_D_bound_audit.py'] + list(argv or [])
    try:
        """
        Lower-bound / unconstrained D_E audit for the real-calcium Wilson-Cowan regression.

        This diagnostic follows directly from the resolution-aware real-data audit, where
        all 20 chunks and all nearby smoothing settings returned D_E exactly at the
        imposed lower bound 0.1.

        Goal
        ----
        Determine what value the real-data regression actually prefers when the D_E
        lower bound is relaxed.

        For each 30-s chunk, using the PRIMARY resolution-aware smoothing only:
          sigma_t = 2 frames
          sigma_xy = 0.15748 mm physical scale
                   ~= 0.992 pixels for a 64x64 grid over 10 mm
                   ~= 2.000 pixels for a 128x128 grid over 10 mm

        Fit specifications
        ------------------
        1) D_min = 0.10, w bounded [0.5, 6.0]
        2) D_min = 0.05, w bounded [0.5, 6.0]
        3) D_min = 0.01, w bounded [0.5, 6.0]
        4) D_min = 0.00, w bounded [0.5, 6.0]
        5) D unbounded, w still bounded [0.5, 6.0]
        6) Fully unconstrained ordinary least squares for [w_EE, D_E, h]

        This script does not retrain the gPINN and does not overwrite previous outputs.

        Outputs
        -------
        outputs/regression_bound_audit/
          real_bound_audit_all_fits.csv
          real_bound_audit_by_chunk.csv
          real_bound_audit_summary.json
          D_E_by_fit.png
          w_EE_by_fit.png
          h_by_fit.png
          pde_rmse_by_fit.png
        """

        import argparse
        import csv
        import json
        from pathlib import Path

        import numpy as np
        from scipy.ndimage import gaussian_filter
        from scipy.optimize import lsq_linear

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt


        SYNTHETIC_FOV_MM = 10.0
        SYNTHETIC_GRID_N = 128
        SYNTHETIC_BEST_SIGMA_PX = 2.0
        PRIMARY_SIGMA_XY_MM = SYNTHETIC_BEST_SIGMA_PX * (
            SYNTHETIC_FOV_MM / (SYNTHETIC_GRID_N - 1)
        )

        FIT_SPECS = [
            {"label": "Dmin_0p10", "kind": "bounded", "D_min": 0.10},
            {"label": "Dmin_0p05", "kind": "bounded", "D_min": 0.05},
            {"label": "Dmin_0p01", "kind": "bounded", "D_min": 0.01},
            {"label": "Dmin_0p00", "kind": "bounded", "D_min": 0.00},
            {"label": "D_unbounded_w_bounded", "kind": "D_unbounded", "D_min": None},
            {"label": "fully_unconstrained_OLS", "kind": "ols", "D_min": None},
        ]


        def parse_args():
            p = argparse.ArgumentParser()
            p.add_argument("--data_dir", type=str, default="./data/processed")
            p.add_argument("--output_dir", type=str, default="./outputs/regression_bound_audit")
            p.add_argument("--num_chunks", type=int, default=20)
            p.add_argument("--subject_prefix", type=str, default="sub-GC48_ses-day0")
            p.add_argument("--fps", type=float, default=30.0)
            p.add_argument("--tau", type=float, default=1.0)
            p.add_argument("--l_fov", type=float, default=10.0)
            p.add_argument("--n_points", type=int, default=90000)
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


        def sample_interior_indices(shape, n_points: int, seed: int):
            T, H, W = shape
            n_total = (T - 2) * (H - 2) * (W - 2)
            n = min(int(n_points), n_total)

            rng = np.random.default_rng(seed)
            flat = rng.choice(n_total, size=n, replace=False)

            hw = (H - 2) * (W - 2)
            ti = flat // hw + 1
            rem = flat % hw
            yi = rem // (W - 2) + 1
            xi = rem % (W - 2) + 1

            return ti, yi, xi


        def derivatives_at_indices(E, ti, yi, xi, fps: float, l_fov: float):
            _, H, W = E.shape
            dt = 1.0 / float(fps)
            dx = float(l_fov) / float(W - 1)
            dy = float(l_fov) / float(H - 1)

            E0 = E[ti, yi, xi].astype(np.float64)

            Et = (
                E[ti + 1, yi, xi].astype(np.float64)
                - E[ti - 1, yi, xi].astype(np.float64)
            ) / (2.0 * dt)

            d2x = (
                E[ti, yi, xi + 1].astype(np.float64)
                - 2.0 * E0
                + E[ti, yi, xi - 1].astype(np.float64)
            ) / (dx * dx)

            d2y = (
                E[ti, yi + 1, xi].astype(np.float64)
                - 2.0 * E0
                + E[ti, yi - 1, xi].astype(np.float64)
            ) / (dy * dy)

            return E0, Et, d2x + d2y


        def prepare_regression(E, Et, lap, tau: float, eps: float):
            E = np.asarray(E, dtype=np.float64).ravel()
            Et = np.asarray(Et, dtype=np.float64).ravel()
            lap = np.asarray(lap, dtype=np.float64).ravel()

            q = float(tau) * Et + E

            finite = (
                np.isfinite(E)
                & np.isfinite(Et)
                & np.isfinite(lap)
                & np.isfinite(q)
            )
            valid = finite & (q > eps) & (q < 1.0 - eps)

            if valid.sum() < 100:
                raise RuntimeError("Fewer than 100 valid logit points.")

            Ev = E[valid]
            Etv = Et[valid]
            Lv = lap[valid]
            qv = q[valid]

            y = np.log(qv / (1.0 - qv))
            X = np.column_stack([Ev, Lv, np.ones_like(Ev)])  # [w, D, h]

            Xstd = np.column_stack([
                (Ev - Ev.mean()) / (Ev.std() + 1e-12),
                (Lv - Lv.mean()) / (Lv.std() + 1e-12),
            ])

            diagnostics = {
                "n_total": int(len(E)),
                "n_valid": int(valid.sum()),
                "fraction_valid": float(valid.mean()),
                "condition_number_raw_X": float(np.linalg.cond(X)),
                "condition_number_standardized_E_lap": float(np.linalg.cond(Xstd)),
                "corr_E_laplacian": float(np.corrcoef(Ev, Lv)[0, 1]),
                "q_p01": float(np.quantile(qv, 0.01)),
                "q_p50": float(np.quantile(qv, 0.50)),
                "q_p99": float(np.quantile(qv, 0.99)),
            }

            return X, y, Ev, Etv, Lv, diagnostics


        def compute_fit_metrics(beta, X, y, Ev, Etv, Lv, tau):
            w_hat, D_hat, h_hat = map(float, beta)

            yhat = X @ np.asarray(beta, dtype=np.float64)
            logit_rmse = float(np.sqrt(np.mean((yhat - y) ** 2)))

            z = w_hat * Ev + D_hat * Lv + h_hat
            rhs = -Ev + 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))
            pde_res = float(tau) * Etv - rhs

            return {
                "w_EE": w_hat,
                "D_E": D_hat,
                "h": h_hat,
                "logit_rmse": logit_rmse,
                "pde_residual_rmse": float(np.sqrt(np.mean(pde_res ** 2))),
            }


        def fit_spec(spec, X, y, Ev, Etv, Lv, tau):
            kind = spec["kind"]

            if kind == "bounded":
                D_min = float(spec["D_min"])
                fit = lsq_linear(
                    X,
                    y,
                    bounds=(
                        np.array([0.5, D_min, -np.inf]),
                        np.array([6.0, 5.0, np.inf]),
                    ),
                    method="trf",
                    lsmr_tol="auto",
                    verbose=0,
                )
                metrics = compute_fit_metrics(fit.x, X, y, Ev, Etv, Lv, tau)
                metrics.update({
                    "success": bool(fit.success),
                    "optimizer_cost": float(fit.cost),
                    "optimizer_optimality": float(fit.optimality),
                    "D_lower_bound": D_min,
                    "D_upper_bound": 5.0,
                    "w_lower_bound": 0.5,
                    "w_upper_bound": 6.0,
                    "D_at_lower_bound": bool(metrics["D_E"] <= D_min + 5e-4),
                    "D_negative": bool(metrics["D_E"] < 0.0),
                })
                return metrics

            if kind == "D_unbounded":
                fit = lsq_linear(
                    X,
                    y,
                    bounds=(
                        np.array([0.5, -np.inf, -np.inf]),
                        np.array([6.0, np.inf, np.inf]),
                    ),
                    method="trf",
                    lsmr_tol="auto",
                    verbose=0,
                )
                metrics = compute_fit_metrics(fit.x, X, y, Ev, Etv, Lv, tau)
                metrics.update({
                    "success": bool(fit.success),
                    "optimizer_cost": float(fit.cost),
                    "optimizer_optimality": float(fit.optimality),
                    "D_lower_bound": "unbounded",
                    "D_upper_bound": "unbounded",
                    "w_lower_bound": 0.5,
                    "w_upper_bound": 6.0,
                    "D_at_lower_bound": False,
                    "D_negative": bool(metrics["D_E"] < 0.0),
                })
                return metrics

            if kind == "ols":
                beta, residuals, rank, svals = np.linalg.lstsq(X, y, rcond=None)
                metrics = compute_fit_metrics(beta, X, y, Ev, Etv, Lv, tau)
                metrics.update({
                    "success": True,
                    "optimizer_cost": (
                        float(0.5 * residuals[0]) if len(residuals) else np.nan
                    ),
                    "optimizer_optimality": np.nan,
                    "D_lower_bound": "unbounded",
                    "D_upper_bound": "unbounded",
                    "w_lower_bound": "unbounded",
                    "w_upper_bound": "unbounded",
                    "D_at_lower_bound": False,
                    "D_negative": bool(metrics["D_E"] < 0.0),
                    "ols_rank": int(rank),
                    "ols_singular_values": ";".join(f"{x:.8g}" for x in svals),
                })
                return metrics

            raise ValueError(f"Unknown fit kind: {kind}")


        def write_csv(path, rows):
            if not rows:
                return
            fields = sorted({k for row in rows for k in row.keys()})
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(rows)


        def plot_by_fit(rows, value, ylabel, out_path):
            labels = [s["label"] for s in FIT_SPECS]
            fig, ax = plt.subplots(figsize=(9.5, 5.5))

            for label in labels:
                sub = sorted(
                    [r for r in rows if r["fit_label"] == label],
                    key=lambda r: r["chunk_idx"],
                )
                if not sub:
                    continue
                x = [r["chunk_idx"] for r in sub]
                y = [r[value] for r in sub]
                ax.plot(x, y, marker="o", linewidth=1.2, label=label)

            ax.set_xlabel("30-s chunk")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{ylabel}: D_E lower-bound / unconstrained audit")
            ax.legend(fontsize=8)
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
            print(f"Primary physical sigma_xy={PRIMARY_SIGMA_XY_MM:.6f} mm")
            print("Fit specs:")
            for s in FIT_SPECS:
                print(f"  - {s['label']}")

            rows = []

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

                ti, yi, xi = sample_interior_indices(
                    E_scaled.shape,
                    args.n_points,
                    args.seed + chunk_idx,
                )

                E0, Et, lap = derivatives_at_indices(
                    E_sm, ti, yi, xi, args.fps, args.l_fov
                )

                X, y, Ev, Etv, Lv, diag = prepare_regression(
                    E0, Et, lap, args.tau, args.eps
                )

                print(
                    f"\nChunk {chunk_idx:02d} shape={E_scaled.shape} "
                    f"sigma_xy={sigma_xy_px:.3f}px "
                    f"valid={diag['fraction_valid']:.4f} "
                    f"cond={diag['condition_number_standardized_E_lap']:.3f}"
                )

                for spec in FIT_SPECS:
                    fit = fit_spec(spec, X, y, Ev, Etv, Lv, args.tau)

                    row = {
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
                        "fit_label": spec["label"],
                        "fit_kind": spec["kind"],
                        **diag,
                        **fit,
                    }
                    rows.append(row)

                    print(
                        f"  {spec['label']:<24s} "
                        f"D={fit['D_E']:+.6f} "
                        f"w={fit['w_EE']:.6f} "
                        f"h={fit['h']:.6f} "
                        f"PDE_RMSE={fit['pde_residual_rmse']:.5f}"
                    )

            write_csv(out / "real_bound_audit_all_fits.csv", rows)

            # Wide chunk summary for the key D_E values.
            by_chunk = []
            for c in sorted({r["chunk_idx"] for r in rows}):
                sub = {r["fit_label"]: r for r in rows if r["chunk_idx"] == c}
                row = {"chunk_idx": c, "time_start_min": 0.5 * c}
                for spec in FIT_SPECS:
                    label = spec["label"]
                    if label in sub:
                        row[f"D_E__{label}"] = sub[label]["D_E"]
                        row[f"w_EE__{label}"] = sub[label]["w_EE"]
                        row[f"h__{label}"] = sub[label]["h"]
                        row[f"pde_rmse__{label}"] = sub[label]["pde_residual_rmse"]
                by_chunk.append(row)

            write_csv(out / "real_bound_audit_by_chunk.csv", by_chunk)

            for value, ylabel, name in [
                ("D_E", "D_E", "D_E_by_fit.png"),
                ("w_EE", "w_EE", "w_EE_by_fit.png"),
                ("h", "h", "h_by_fit.png"),
                ("pde_residual_rmse", "PDE residual RMSE", "pde_rmse_by_fit.png"),
            ]:
                plot_by_fit(rows, value, ylabel, out / name)

            # Compact summary by fit label.
            fit_summary = {}
            for spec in FIT_SPECS:
                label = spec["label"]
                sub = [r for r in rows if r["fit_label"] == label]
                if not sub:
                    continue

                D = np.array([r["D_E"] for r in sub], dtype=float)
                w = np.array([r["w_EE"] for r in sub], dtype=float)
                h = np.array([r["h"] for r in sub], dtype=float)
                pde = np.array([r["pde_residual_rmse"] for r in sub], dtype=float)

                fit_summary[label] = {
                    "n_chunks": len(sub),
                    "D_E_mean": float(D.mean()),
                    "D_E_sd": float(D.std(ddof=1)) if len(D) > 1 else 0.0,
                    "D_E_min": float(D.min()),
                    "D_E_max": float(D.max()),
                    "fraction_D_negative": float(np.mean(D < 0.0)),
                    "w_EE_mean": float(w.mean()),
                    "w_EE_sd": float(w.std(ddof=1)) if len(w) > 1 else 0.0,
                    "h_mean": float(h.mean()),
                    "h_sd": float(h.std(ddof=1)) if len(h) > 1 else 0.0,
                    "pde_residual_rmse_mean": float(pde.mean()),
                }

            summary = {
                "purpose": "D_E lower-bound and unconstrained real-data audit",
                "normalization": "session-global min-max scaling matching the current gPINN",
                "num_chunks_found": n_found,
                "primary_smoothing": {
                    "sigma_t_frames": 2.0,
                    "sigma_xy_mm": PRIMARY_SIGMA_XY_MM,
                },
                "fit_specs": FIT_SPECS,
                "fit_summary": fit_summary,
                "interpretation_guardrail": (
                    "Negative unconstrained D_E should not be interpreted as biological "
                    "negative diffusivity. It would indicate that the current reduced model, "
                    "calcium representation, derivative construction, or their combination "
                    "prefers a sign inconsistent with the imposed positive-coupling model."
                ),
            }

            (out / "real_bound_audit_summary.json").write_text(
                json.dumps(summary, indent=2)
            )

            print(f"\nAudit complete. Outputs saved to: {out}")
        return main()
    finally:
        _sys.argv = _old_argv


# ============================================================================
# regression_audit: source real_calcium_regression_audit_resolution_aware.py
# ============================================================================
def regression_audit(argv=None):
    """Run this analysis with command-specific CLI arguments."""
    _old_argv = _sys.argv
    _sys.argv = ['real_calcium_regression_audit_resolution_aware.py'] + list(argv or [])
    try:
        """
        Resolution-aware real-calcium Wilson–Cowan pseudo-derivative regression audit.

        This audit uses the same session-global min-max scaling as the current gPINN
        for like-for-like comparison, but chooses spatial Gaussian smoothing in
        PHYSICAL UNITS rather than fixed pixels.

        Synthetic calibration:
            10 mm FOV, 128 x 128 grid, best sigma_xy = 2 pixels
        which corresponds to:
            sigma_xy_mm = 2 * (10 / 127) = 0.15748 mm

        Therefore:
            64 x 64 over 10 mm  -> sigma_xy ~= 0.99 pixel
            128 x 128 over 10 mm -> sigma_xy = 2.00 pixels

        Primary temporal smoothing remains sigma_t = 2 frames at 30 Hz.

        For sensitivity, the audit evaluates spatial smoothing at 0.5x, 1.0x,
        and 2.0x the calibrated physical scale, plus temporal sigma_t = 1 and 2
        frames around the primary spatial scale.

        The script does NOT alter or retrain the gPINN.
        """

        import argparse
        import csv
        import json
        from pathlib import Path

        import numpy as np
        from scipy.ndimage import gaussian_filter
        from scipy.optimize import lsq_linear

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt


        SYNTHETIC_FOV_MM = 10.0
        SYNTHETIC_GRID_N = 128
        SYNTHETIC_BEST_SIGMA_PX = 2.0
        PRIMARY_SIGMA_XY_MM = SYNTHETIC_BEST_SIGMA_PX * (
            SYNTHETIC_FOV_MM / (SYNTHETIC_GRID_N - 1)
        )  # ~0.15748 mm


        def parse_args():
            p = argparse.ArgumentParser()
            p.add_argument("--data_dir", type=str, default="./data/processed")
            p.add_argument("--output_dir", type=str, default="./outputs/regression_audit")
            p.add_argument("--num_chunks", type=int, default=20)
            p.add_argument("--subject_prefix", type=str, default="sub-GC48_ses-day0")
            p.add_argument("--fps", type=float, default=30.0)
            p.add_argument("--tau", type=float, default=1.0)
            p.add_argument("--l_fov", type=float, default=10.0)
            p.add_argument("--n_points", type=int, default=90000)
            p.add_argument("--seed", type=int, default=42)
            p.add_argument("--eps", type=float, default=1e-4)
            p.add_argument(
                "--gpinn_results",
                type=str,
                default="./outputs/temporal_tracking_results.npz",
            )
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


        def sample_interior_indices(shape, n_points, seed):
            T, H, W = shape
            n_total = (T - 2) * (H - 2) * (W - 2)
            n = min(int(n_points), n_total)
            rng = np.random.default_rng(seed)
            flat = rng.choice(n_total, size=n, replace=False)
            hw = (H - 2) * (W - 2)
            ti = flat // hw + 1
            rem = flat % hw
            yi = rem // (W - 2) + 1
            xi = rem % (W - 2) + 1
            return ti, yi, xi


        def derivatives_at_indices(E, ti, yi, xi, fps, l_fov):
            _, H, W = E.shape
            dt = 1.0 / float(fps)
            dx = float(l_fov) / float(W - 1)
            dy = float(l_fov) / float(H - 1)

            E0 = E[ti, yi, xi].astype(np.float64)
            Et = (
                E[ti + 1, yi, xi].astype(np.float64)
                - E[ti - 1, yi, xi].astype(np.float64)
            ) / (2.0 * dt)

            d2x = (
                E[ti, yi, xi + 1].astype(np.float64)
                - 2.0 * E0
                + E[ti, yi, xi - 1].astype(np.float64)
            ) / (dx * dx)

            d2y = (
                E[ti, yi + 1, xi].astype(np.float64)
                - 2.0 * E0
                + E[ti, yi - 1, xi].astype(np.float64)
            ) / (dy * dy)

            return E0, Et, d2x + d2y


        def bounded_logit_regression(E, Et, lap, tau, eps):
            E = np.asarray(E, dtype=np.float64).ravel()
            Et = np.asarray(Et, dtype=np.float64).ravel()
            lap = np.asarray(lap, dtype=np.float64).ravel()

            q = float(tau) * Et + E
            finite = np.isfinite(E) & np.isfinite(Et) & np.isfinite(lap) & np.isfinite(q)
            valid = finite & (q > eps) & (q < 1.0 - eps)

            result = {
                "n_total": int(len(E)),
                "n_valid": int(valid.sum()),
                "fraction_valid": float(valid.mean()),
                "q_p01": float(np.nanquantile(q[finite], 0.01)) if finite.any() else np.nan,
                "q_p50": float(np.nanquantile(q[finite], 0.50)) if finite.any() else np.nan,
                "q_p99": float(np.nanquantile(q[finite], 0.99)) if finite.any() else np.nan,
            }

            if valid.sum() < 100:
                result["success"] = False
                result["failure_reason"] = "fewer than 100 valid logit points"
                return result

            Ev = E[valid]
            Etv = Et[valid]
            Lv = lap[valid]
            qv = q[valid]

            y = np.log(qv / (1.0 - qv))
            X = np.column_stack([Ev, Lv, np.ones_like(Ev)])  # w, D, h

            fit = lsq_linear(
                X,
                y,
                bounds=(
                    np.array([0.5, 0.1, -np.inf]),
                    np.array([6.0, 5.0, np.inf]),
                ),
                method="trf",
                lsmr_tol="auto",
                verbose=0,
            )

            w_hat, D_hat, h_hat = map(float, fit.x)
            yhat = X @ fit.x

            Xstd = np.column_stack([
                (Ev - Ev.mean()) / (Ev.std() + 1e-12),
                (Lv - Lv.mean()) / (Lv.std() + 1e-12),
            ])

            z = w_hat * Ev + D_hat * Lv + h_hat
            rhs = -Ev + 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))
            pde_res = float(tau) * Etv - rhs

            result.update({
                "success": bool(fit.success),
                "D_E": D_hat,
                "w_EE": w_hat,
                "h": h_hat,
                "cost": float(fit.cost),
                "optimality": float(fit.optimality),
                "logit_rmse": float(np.sqrt(np.mean((yhat - y) ** 2))),
                "pde_residual_rmse": float(np.sqrt(np.mean(pde_res ** 2))),
                "condition_number_raw_X": float(np.linalg.cond(X)),
                "condition_number_standardized_E_lap": float(np.linalg.cond(Xstd)),
                "corr_E_laplacian": float(np.corrcoef(Ev, Lv)[0, 1]),
                "D_at_lower_bound": bool(D_hat <= 0.1005),
                "D_at_upper_bound": bool(D_hat >= 4.9995),
                "w_at_lower_bound": bool(w_hat <= 0.5005),
                "w_at_upper_bound": bool(w_hat >= 5.9995),
            })
            return result


        def smoothing_settings_for_width(W, l_fov_mm):
            dx_mm = float(l_fov_mm) / float(W - 1)
            primary_px = PRIMARY_SIGMA_XY_MM / dx_mm

            # Matched physical sensitivity settings.
            return [
                # temporal sensitivity at primary spatial scale
                {"label": "t1_primary", "sigma_t_frames": 1.0,
                 "sigma_xy_pixels": primary_px, "sigma_xy_mm": PRIMARY_SIGMA_XY_MM},
                {"label": "primary", "sigma_t_frames": 2.0,
                 "sigma_xy_pixels": primary_px, "sigma_xy_mm": PRIMARY_SIGMA_XY_MM},
                # spatial sensitivity around primary scale
                {"label": "half_space", "sigma_t_frames": 2.0,
                 "sigma_xy_pixels": 0.5 * primary_px,
                 "sigma_xy_mm": 0.5 * PRIMARY_SIGMA_XY_MM},
                {"label": "double_space", "sigma_t_frames": 2.0,
                 "sigma_xy_pixels": 2.0 * primary_px,
                 "sigma_xy_mm": 2.0 * PRIMARY_SIGMA_XY_MM},
            ]


        def load_gpinn_results(path):
            path = Path(path)
            if not path.exists():
                return None
            d = np.load(path, allow_pickle=True)
            if not all(k in d for k in ("D_E", "w_EE", "h")):
                return None
            return {
                "D_E": np.asarray(d["D_E"], dtype=float),
                "w_EE": np.asarray(d["w_EE"], dtype=float),
                "h": np.asarray(d["h"], dtype=float),
            }


        def write_csv(path, rows):
            if not rows:
                return
            fields = sorted({k for row in rows for k in row})
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)


        def plot_comparison(rows, gpinn, param, out_path):
            chunks = np.array([r["chunk_idx"] for r in rows], dtype=int)
            vals = np.array([r[param] for r in rows], dtype=float)

            fig, ax = plt.subplots(figsize=(8, 4.8))
            ax.plot(chunks, vals, marker="o", label="Pseudo-derivative regression")

            if gpinn is not None:
                n = min(len(chunks), len(gpinn[param]))
                ax.plot(chunks[:n], gpinn[param][:n], marker="s", label="Existing gPINN")

            ax.set_xlabel("30-s chunk")
            ax.set_ylabel(param)
            ax.set_title(f"{param}: regression audit vs existing gPINN")
            ax.legend()
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
            print(f"Calibrated physical spatial sigma={PRIMARY_SIGMA_XY_MM:.6f} mm")

            all_rows = []

            for chunk_idx in range(args.num_chunks):
                f = chunk_path(data_dir, args.subject_prefix, chunk_idx)
                if not f.exists():
                    print(f"Chunk {chunk_idx:02d}: missing; skipped.")
                    continue

                raw = np.load(f)["dfof"].astype(np.float64)
                E_scaled = (raw - g_min) / scale
                T, H, W = E_scaled.shape

                ti, yi, xi = sample_interior_indices(
                    E_scaled.shape, args.n_points, args.seed + chunk_idx
                )

                settings = smoothing_settings_for_width(W, args.l_fov)
                dx_mm = args.l_fov / (W - 1)

                print(
                    f"\nChunk {chunk_idx:02d}: shape={E_scaled.shape}, "
                    f"dx={dx_mm:.5f} mm/pixel, "
                    f"primary sigma_xy={settings[1]['sigma_xy_pixels']:.3f} px"
                )

                for s in settings:
                    E_sm = gaussian_filter(
                        E_scaled,
                        sigma=(s["sigma_t_frames"],
                               s["sigma_xy_pixels"],
                               s["sigma_xy_pixels"]),
                        mode="reflect",
                    )

                    E0, Et, lap = derivatives_at_indices(
                        E_sm, ti, yi, xi, args.fps, args.l_fov
                    )

                    reg = bounded_logit_regression(E0, Et, lap, args.tau, args.eps)

                    row = {
                        "chunk_idx": chunk_idx,
                        "time_start_min": 0.5 * chunk_idx,
                        "source_file": f.name,
                        "grid_T": T,
                        "grid_H": H,
                        "grid_W": W,
                        "dx_mm": dx_mm,
                        "smoothing_label": s["label"],
                        "smooth_sigma_t_frames": s["sigma_t_frames"],
                        "smooth_sigma_space_pixels": s["sigma_xy_pixels"],
                        "smooth_sigma_space_mm": s["sigma_xy_mm"],
                        **reg,
                    }
                    all_rows.append(row)

                    if reg.get("success", False):
                        print(
                            f"  {s['label']:>12s} "
                            f"sigma_t={s['sigma_t_frames']:.1f}, "
                            f"sigma_xy={s['sigma_xy_pixels']:.3f}px "
                            f"({s['sigma_xy_mm']:.4f}mm) | "
                            f"valid={reg['fraction_valid']:.3f} "
                            f"cond={reg['condition_number_standardized_E_lap']:.2f} "
                            f"D={reg['D_E']:.4f} "
                            f"w={reg['w_EE']:.4f} "
                            f"h={reg['h']:.4f}"
                        )

            write_csv(out / "real_regression_all_settings.csv", all_rows)

            primary_rows = [
                r for r in all_rows
                if r.get("success", False) and r["smoothing_label"] == "primary"
            ]
            write_csv(out / "real_regression_primary.csv", primary_rows)

            stability_rows = []
            for c in sorted({r["chunk_idx"] for r in all_rows}):
                sub = [r for r in all_rows if r["chunk_idx"] == c and r.get("success", False)]
                if not sub:
                    continue
                D = np.array([r["D_E"] for r in sub], dtype=float)
                w = np.array([r["w_EE"] for r in sub], dtype=float)
                h = np.array([r["h"] for r in sub], dtype=float)
                stability_rows.append({
                    "chunk_idx": c,
                    "n_settings": len(sub),
                    "D_E_mean": float(D.mean()),
                    "D_E_sd": float(D.std(ddof=1)) if len(D) > 1 else 0.0,
                    "D_E_range": float(D.max() - D.min()),
                    "w_EE_mean": float(w.mean()),
                    "w_EE_sd": float(w.std(ddof=1)) if len(w) > 1 else 0.0,
                    "w_EE_range": float(w.max() - w.min()),
                    "h_mean": float(h.mean()),
                    "h_sd": float(h.std(ddof=1)) if len(h) > 1 else 0.0,
                    "h_range": float(h.max() - h.min()),
                    "valid_fraction_min": float(min(r["fraction_valid"] for r in sub)),
                    "valid_fraction_max": float(max(r["fraction_valid"] for r in sub)),
                    "condition_number_standardized_max": float(
                        max(r["condition_number_standardized_E_lap"] for r in sub)
                    ),
                })

            write_csv(out / "real_regression_smoothing_stability.csv", stability_rows)

            gpinn = load_gpinn_results(args.gpinn_results)
            if primary_rows:
                for param in ("D_E", "w_EE", "h"):
                    plot_comparison(
                        primary_rows, gpinn, param, out / f"compare_{param}.png"
                    )

            summary = {
                "purpose": "resolution-aware real-calcium pseudo-derivative regression audit",
                "normalization": "session-global min-max scaling matching current gPINN",
                "global_min": g_min,
                "global_max": g_max,
                "num_chunks_found": n_found,
                "synthetic_calibration": {
                    "fov_mm": SYNTHETIC_FOV_MM,
                    "grid_n": SYNTHETIC_GRID_N,
                    "best_sigma_pixels": SYNTHETIC_BEST_SIGMA_PX,
                    "derived_sigma_mm": PRIMARY_SIGMA_XY_MM,
                },
                "primary_temporal_sigma_frames": 2.0,
                "primary_spatial_sigma_mm": PRIMARY_SIGMA_XY_MM,
                "n_points_per_chunk_per_setting": args.n_points,
                "fps": args.fps,
                "tau": args.tau,
                "l_fov_mm": args.l_fov,
                "bounds": {"D_E": [0.1, 5.0], "w_EE": [0.5, 6.0], "h": "unbounded"},
                "gpinn_comparison_file_found": bool(gpinn is not None),
                "important_caution": (
                    "Real fluorescence has no known Wilson-Cowan parameter truth. "
                    "Interpret estimates together with valid fraction, conditioning, "
                    "smoothing stability, and model residual diagnostics."
                ),
            }
            (out / "real_regression_audit_summary.json").write_text(
                json.dumps(summary, indent=2)
            )

            print(f"\nAudit complete. Outputs saved to {out}")
        return main()
    finally:
        _sys.argv = _old_argv


# ============================================================================
# uncertainty_audit: source real_calcium_nested_uncertainty_audit.py
# ============================================================================
def uncertainty_audit(argv=None):
    """Run this analysis with command-specific CLI arguments."""
    _old_argv = _sys.argv
    _sys.argv = ['real_calcium_nested_uncertainty_audit.py'] + list(argv or [])
    try:
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
        return main()
    finally:
        _sys.argv = _old_argv
