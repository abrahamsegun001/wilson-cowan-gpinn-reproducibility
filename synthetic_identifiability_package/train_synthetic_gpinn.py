#!/usr/bin/env python3
"""Single-seed inverse recovery on independently generated Wilson-Cowan data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from wc_gpinn_core import (
    gPINN_WilsonCowan,
    set_seed,
    train_single_chunk,
)


def flat_indices_to_xyt(indices: np.ndarray, T: int, H: int, W: int) -> np.ndarray:
    # Flattening order corresponds to E[t, y, x].
    hw = H * W
    ti = indices // hw
    rem = indices % hw
    yi = rem // W
    xi = rem % W

    x = -1.0 + 2.0 * xi.astype(np.float32) / float(W - 1)
    y = -1.0 + 2.0 * yi.astype(np.float32) / float(H - 1)
    t = -1.0 + 2.0 * ti.astype(np.float32) / float(T - 1)
    return np.stack([x, y, t], axis=1).astype(np.float32)


def make_sampled_tensors(
    E_clean: np.ndarray,
    n_data: int,
    n_colloc: int,
    noise_sigma: float,
    noise_seed: int,
    sampling_seed: int,
    device: torch.device,
):
    T, H, W = E_clean.shape
    n_total = T * H * W
    n_train = min(int(n_data), int(0.8 * n_total))
    n_val = min(int(n_data * 0.25), n_total - n_train)

    # A fixed sampling seed means all training seeds see exactly the same
    # supervised and collocation locations, isolating network/optimization seed.
    rng = np.random.default_rng(int(sampling_seed))
    chosen = rng.choice(n_total, size=n_train + n_val, replace=False)
    idx_tr = chosen[:n_train]
    idx_va = chosen[n_train:]
    idx_c = rng.choice(n_total, size=min(n_colloc, n_total), replace=False)

    flat_clean = E_clean.reshape(-1).astype(np.float32)
    if noise_sigma > 0:
        nrng = np.random.default_rng(int(noise_seed))
        noise = nrng.normal(0.0, noise_sigma, size=flat_clean.shape).astype(np.float32)
        flat_obs = np.clip(flat_clean + noise, 0.0, 1.0)
    else:
        flat_obs = flat_clean

    Xt_tr_np = flat_indices_to_xyt(idx_tr, T, H, W)
    Xt_va_np = flat_indices_to_xyt(idx_va, T, H, W)
    Xc_np = flat_indices_to_xyt(idx_c, T, H, W)

    Xt_tr = torch.from_numpy(Xt_tr_np).to(device)
    Et_tr = torch.from_numpy(flat_obs[idx_tr, None]).to(device)
    Xt_va = torch.from_numpy(Xt_va_np).to(device)
    # Validation is against the same observed/noisy target, matching the real pipeline.
    Et_va = torch.from_numpy(flat_obs[idx_va, None]).to(device)
    # Also retain clean validation target for a noise-independent reconstruction metric.
    Et_va_clean = torch.from_numpy(flat_clean[idx_va, None]).to(device)
    Xc = torch.from_numpy(Xc_np).to(device).requires_grad_(True)
    return Xt_tr, Et_tr, Xt_va, Et_va, Et_va_clean, Xc


def metrics_against_clean(model, Xt, Et_clean):
    model.eval()
    with torch.no_grad():
        pred = model.forward_E(Xt)
        mse = torch.mean((pred - Et_clean) ** 2).item()
        true_np = Et_clean.cpu().numpy().ravel()
        pred_np = pred.cpu().numpy().ravel()
    ss_res = float(np.sum((true_np - pred_np) ** 2))
    ss_tot = float(np.sum((true_np - true_np.mean()) ** 2))
    r2 = float(1.0 - ss_res / (ss_tot + 1e-8))
    model.train()
    return float(mse), r2


def safe_tag(x: float) -> str:
    return f"{x:.3f}".replace("-", "m").replace(".", "p")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="synthetic_data/wc_twin_truth.npz")
    p.add_argument("--output_dir", type=str, default="synthetic_outputs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fourier_seed", type=int, default=2718)
    p.add_argument("--sampling_seed", type=int, default=31415)
    p.add_argument("--noise_seed", type=int, default=1618)
    p.add_argument("--noise_sigma", type=float, default=0.0)
    p.add_argument("--n_data", type=int, default=60000)
    p.add_argument("--n_colloc", type=int, default=90000)
    p.add_argument("--adam_epochs", type=int, default=5000)
    p.add_argument("--lbfgs_steps", type=int, default=500)
    p.add_argument("--lambda_physics", type=float, default=0.01)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--mu_grad", type=float, default=1e-5)
    args = p.parse_args()

    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    d = np.load(args.data, allow_pickle=False)
    E = d["E"].astype(np.float32)
    truth = {
        "D_E": float(d["D_E_true"]),
        "w_EE": float(d["w_EE_true"]),
        "h": float(d["h_true"]),
    }

    Xt_tr, Et_tr, Xt_va, Et_va, Et_va_clean, Xc = make_sampled_tensors(
        E, args.n_data, args.n_colloc, args.noise_sigma,
        args.noise_seed, args.sampling_seed, device,
    )

    # Fixed Fourier seed + fixed sampled points across replicate training seeds.
    # Therefore replicate variation is attributable to trainable-network initialization
    # and optimization, not a different synthetic dataset or a different point split.
    model = gPINN_WilsonCowan(fourier_seed=args.fourier_seed).to(device)
    result = train_single_chunk(
        model, Xt_tr, Et_tr, Xt_va, Et_va, Xc,
        lr=args.lr,
        adam_epochs=args.adam_epochs,
        lbfgs_steps=args.lbfgs_steps,
        lambda_phys=args.lambda_physics,
        mu_grad=args.mu_grad,
    )

    clean_mse, clean_r2 = metrics_against_clean(model, Xt_va, Et_va_clean)
    est = {
        "D_E": float(model.forward_D().item()),
        "w_EE": float(model.w_EE.item()),
        "h": float(model.h.item()),
        "raw_D": float(model.raw_D.item()),
    }

    rel_error = {
        k: abs(est[k] - truth[k]) / max(abs(truth[k]), 1e-12) * 100.0
        for k in ("D_E", "w_EE", "h")
    }

    out = Path(args.output_dir)
    (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "histories").mkdir(parents=True, exist_ok=True)

    tag = f"seed-{args.seed}_noise-{safe_tag(args.noise_sigma)}"
    checkpoint_path = out / "checkpoints" / f"{tag}.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "seed": args.seed,
            "fourier_seed": args.fourier_seed,
            "sampling_seed": args.sampling_seed,
            "noise_seed": args.noise_seed,
            "noise_sigma": args.noise_sigma,
            "truth": truth,
            "estimate": est,
        },
        checkpoint_path,
    )

    np.savez_compressed(
        out / "histories" / f"{tag}.npz",
        **{k: np.asarray(v) for k, v in result.history.items()},
    )

    summary = {
        "seed": int(args.seed),
        "fourier_seed": int(args.fourier_seed),
        "sampling_seed": int(args.sampling_seed),
        "noise_seed": int(args.noise_seed),
        "noise_sigma": float(args.noise_sigma),
        "device": str(device),
        "truth": truth,
        "estimate": est,
        "relative_error_percent": rel_error,
        "val_mse_observed": float(result.val_mse),
        "val_r2_observed": float(result.val_r2),
        "val_r_observed": float(result.val_r),
        "val_mse_clean": float(clean_mse),
        "val_r2_clean": float(clean_r2),
        "final_lambda": float(result.final_lambda),
        "checkpoint": str(checkpoint_path),
        "n_data": int(args.n_data),
        "n_colloc": int(args.n_colloc),
        "adam_epochs": int(args.adam_epochs),
        "lbfgs_steps": int(args.lbfgs_steps),
        "mu_grad": float(args.mu_grad),
    }
    result_path = out / "results" / f"{tag}.json"
    result_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"Saved: {result_path}")


if __name__ == "__main__":
    main()
