#!/usr/bin/env python3
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

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from train_dual_model_sequential20 import (
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
    from train_dual_model_sequential20 import indices_to_coords
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


if __name__ == "__main__":
    main()
