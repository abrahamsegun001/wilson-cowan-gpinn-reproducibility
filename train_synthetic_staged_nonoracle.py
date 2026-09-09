#!/usr/bin/env python3
"""
Synthetic staged non-oracle control for Wilson-Cowan gPINN parameter recovery.

PURPOSE
-------
The current synthetic experiment showed:
  * excellent state reconstruction E,
  * poor E_t and Laplacian(E),
  * poor physical-parameter recovery,
  * yet near-exact direct recovery from the true synthetic trajectory.

This control asks a causal question:
"If the same neural representation is explicitly taught the correct derivatives
on synthetic data, does physical-parameter recovery move toward ground truth?"

This is intentionally an ORACLE SYNTHETIC CONTROL. The derivative targets come
from the independently generated synthetic field and are NOT available for real
calcium imaging. Do not present this as a real-data training recipe.

Stages
------
A. State + initial-condition pretraining.
B. Sobolev/oracle derivative training using true E_t and Laplacian(E).
C. Freeze the state network and optimize only D_E, w_EE, h from the learned
   field/derivatives.
D. Joint low-learning-rate fine-tuning with state, derivative, and PDE losses.

Outputs
-------
synthetic_outputs/staged_nonoracle_control/
  seed-42_staged_nonoracle.json
  seed-42_staged_nonoracle.pt
  training_history.csv
  scatter_E.png
  scatter_Et.png
  scatter_laplacian.png
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from wc_gpinn_core import gPINN_WilsonCowan


# ------------------------------- utilities -------------------------------- #

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def r2_score(y_true, y_pred):
    a = np.asarray(y_true).ravel()
    b = np.asarray(y_pred).ravel()
    ss_res = np.sum((a - b) ** 2)
    ss_tot = np.sum((a - a.mean()) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-12))


def corrcoef(y_true, y_pred):
    a = np.asarray(y_true).ravel()
    b = np.asarray(y_pred).ravel()
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def metric_block(y_true, y_pred):
    a = np.asarray(y_true).ravel()
    b = np.asarray(y_pred).ravel()
    e = b - a
    return {
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "mae": float(np.mean(np.abs(e))),
        "r2": r2_score(a, b),
        "r": corrcoef(a, b),
        "relative_l2": float(np.linalg.norm(e) / (np.linalg.norm(a) + 1e-12)),
    }


def sigmoid_np(z):
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def physical_parameter_dict(model):
    return {
        "D_E": float(model.forward_D().detach().cpu().item()),
        "w_EE": float(torch.clamp(model.w_EE, 0.5, 6.0).detach().cpu().item()),
        "h": float(model.h.detach().cpu().item()),
    }


def relative_errors(truth, estimate):
    return {
        k: 100.0 * abs(estimate[k] - truth[k]) / (abs(truth[k]) + 1e-12)
        for k in truth
    }


def find_raw_D_parameter(model):
    # Current package uses raw_D. Keep fallbacks for small naming changes.
    for name in ("raw_D", "D_raw", "raw_DE", "raw_d"):
        if hasattr(model, name):
            return getattr(model, name), name
    # Last resort: inspect names.
    for name, p in model.named_parameters():
        if "raw" in name.lower() and "d" in name.lower():
            return p, name
    raise AttributeError("Could not locate the raw D_E trainable parameter.")


def normalized_mse(pred, target, eps=1e-8):
    mse = torch.mean((pred - target) ** 2)
    denom = torch.mean(target ** 2).detach() + eps
    return mse / denom


# --------------------------- synthetic sampling ---------------------------- #

def make_state_pool(E, n_points, seed, early_fraction=0.50, early_seconds=5.0,
                    fps=30.0):
    """
    Fixed state-training pool with deliberate oversampling of the early transient.
    """
    rng = np.random.default_rng(seed)
    T, H, W = E.shape
    n_early = int(round(n_points * early_fraction))
    n_uniform = n_points - n_early

    early_T = max(1, min(T, int(round(early_seconds * fps)) + 1))

    ti_e = rng.integers(0, early_T, size=n_early)
    yi_e = rng.integers(0, H, size=n_early)
    xi_e = rng.integers(0, W, size=n_early)

    ti_u = rng.integers(0, T, size=n_uniform)
    yi_u = rng.integers(0, H, size=n_uniform)
    xi_u = rng.integers(0, W, size=n_uniform)

    ti = np.concatenate([ti_e, ti_u])
    yi = np.concatenate([yi_e, yi_u])
    xi = np.concatenate([xi_e, xi_u])

    x = -1.0 + 2.0 * xi / float(W - 1)
    y = -1.0 + 2.0 * yi / float(H - 1)
    t = -1.0 + 2.0 * ti / float(T - 1)

    xyt = np.stack([x, y, t], axis=1).astype(np.float32)
    target = E[ti, yi, xi].astype(np.float32)[:, None]
    return xyt, target


def make_ic_pool(E, n_points, seed):
    rng = np.random.default_rng(seed)
    _, H, W = E.shape
    total = H * W
    n = min(n_points, total)
    flat = rng.choice(total, n, replace=False)
    yi = flat // W
    xi = flat % W
    x = -1.0 + 2.0 * xi / float(W - 1)
    y = -1.0 + 2.0 * yi / float(H - 1)
    t = -np.ones_like(x)
    xyt = np.stack([x, y, t], axis=1).astype(np.float32)
    target = E[0, yi, xi].astype(np.float32)[:, None]
    return xyt, target


def make_derivative_pool(E, l_fov, t_seconds, n_points, seed,
                         early_fraction=0.60, early_seconds=5.0):
    """
    Interior samples with centered finite-difference E_t and Laplacian targets.
    More weight is deliberately placed on the early transient.
    """
    rng = np.random.default_rng(seed)
    T, H, W = E.shape
    dt = float(np.median(np.diff(t_seconds)))
    dx = float(l_fov) / float(W - 1)
    dy = float(l_fov) / float(H - 1)

    early_last = min(T - 2, max(1, int(round(early_seconds / dt))))
    n_early = int(round(n_points * early_fraction))
    n_uniform = n_points - n_early

    ti_e = rng.integers(1, early_last + 1, size=n_early)
    yi_e = rng.integers(1, H - 1, size=n_early)
    xi_e = rng.integers(1, W - 1, size=n_early)

    ti_u = rng.integers(1, T - 1, size=n_uniform)
    yi_u = rng.integers(1, H - 1, size=n_uniform)
    xi_u = rng.integers(1, W - 1, size=n_uniform)

    ti = np.concatenate([ti_e, ti_u])
    yi = np.concatenate([yi_e, yi_u])
    xi = np.concatenate([xi_e, xi_u])

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

    x = -1.0 + 2.0 * xi / float(W - 1)
    y = -1.0 + 2.0 * yi / float(H - 1)
    t = -1.0 + 2.0 * ti / float(T - 1)

    xyt = np.stack([x, y, t], axis=1).astype(np.float32)
    return (
        xyt,
        E0.astype(np.float32)[:, None],
        dE_dt.astype(np.float32)[:, None],
        lap.astype(np.float32)[:, None],
    )


def batch_from_pool(pool_tensors, batch_size, generator=None):
    n = pool_tensors[0].shape[0]
    idx = torch.randint(
        0, n, (batch_size,),
        device=pool_tensors[0].device,
        generator=generator,
    )
    return tuple(x[idx] for x in pool_tensors)


# ----------------------- autograd derivative functions --------------------- #

def state_and_derivatives(model, xyt, duration, l_fov, create_graph=True):
    """
    Returns E, physical E_t, and physical Laplacian(E).
    xyt must be normalized to [-1,1]^3.
    """
    xyt = xyt.clone().detach().requires_grad_(True)
    E = model.forward_E(xyt)

    grad_E = torch.autograd.grad(
        E, xyt,
        grad_outputs=torch.ones_like(E),
        create_graph=True,
        retain_graph=True,
    )[0]

    dE_dx = grad_E[:, 0:1]
    dE_dy = grad_E[:, 1:2]
    dE_dt_norm = grad_E[:, 2:3]

    d2x = torch.autograd.grad(
        dE_dx, xyt,
        grad_outputs=torch.ones_like(dE_dx),
        create_graph=create_graph,
        retain_graph=True,
    )[0][:, 0:1]

    d2y = torch.autograd.grad(
        dE_dy, xyt,
        grad_outputs=torch.ones_like(dE_dy),
        create_graph=create_graph,
        retain_graph=True,
    )[0][:, 1:2]

    dE_dt = (2.0 / float(duration)) * dE_dt_norm
    lap = (4.0 / float(l_fov) ** 2) * (d2x + d2y)
    return E, dE_dt, lap


def physics_residual(model, E, dE_dt, lap, tau):
    D = model.forward_D()
    w = torch.clamp(model.w_EE, 0.5, 6.0)
    h = model.h
    rhs = -E + torch.sigmoid(w * E + D * lap + h)
    return tau * dE_dt - rhs


# ---------------------------- evaluation ---------------------------------- #

@torch.no_grad()
def predict_state(model, xyt, batch_size=4096):
    vals = []
    for s in range(0, len(xyt), batch_size):
        vals.append(
            model.forward_E(xyt[s:s+batch_size]).detach().cpu().numpy().ravel()
        )
    return np.concatenate(vals)


def predict_derivatives(model, xyt, duration, l_fov, batch_size=512):
    Es, Ts, Ls = [], [], []
    model.eval()
    for s in range(0, len(xyt), batch_size):
        E, Et, Lap = state_and_derivatives(
            model, xyt[s:s+batch_size], duration, l_fov, create_graph=False
        )
        Es.append(E.detach().cpu().numpy().ravel())
        Ts.append(Et.detach().cpu().numpy().ravel())
        Ls.append(Lap.detach().cpu().numpy().ravel())
    return np.concatenate(Es), np.concatenate(Ts), np.concatenate(Ls)


def scatter(x, y, xlabel, ylabel, title, path, max_points=12000):
    rng = np.random.default_rng(123)
    if len(x) > max_points:
        idx = rng.choice(len(x), max_points, replace=False)
        x, y = x[idx], y[idx]
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(x, y, s=5, alpha=0.25)
    lo = min(float(np.min(x)), float(np.min(y)))
    hi = max(float(np.max(x)), float(np.max(y)))
    ax.plot([lo, hi], [lo, hi], "--", linewidth=1.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


# ------------------------------- training --------------------------------- #

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="synthetic_data/wc_twin_truth.npz")
    p.add_argument("--output_dir",
                   default="synthetic_outputs/staged_nonoracle_control")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fourier_seed", type=int, default=2718)

    p.add_argument("--state_pool", type=int, default=60000)
    p.add_argument("--deriv_pool", type=int, default=24000)
    p.add_argument("--ic_pool", type=int, default=8192)

    p.add_argument("--state_batch", type=int, default=4096)
    p.add_argument("--deriv_batch", type=int, default=512)
    p.add_argument("--ic_batch", type=int, default=2048)

    p.add_argument("--stageA_steps", type=int, default=2500)
    p.add_argument("--stageB_steps", type=int, default=2500)
    p.add_argument("--stageC_adam_steps", type=int, default=1800)
    p.add_argument("--stageD_steps", type=int, default=1200)

    p.add_argument("--lrA", type=float, default=1e-3)
    p.add_argument("--lrB", type=float, default=2e-4)
    p.add_argument("--lrC", type=float, default=1e-2)
    p.add_argument("--lrD_net", type=float, default=2e-5)
    p.add_argument("--lrD_phys", type=float, default=5e-4)

    p.add_argument("--duration", type=float, default=30.0)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    d = np.load(args.data, allow_pickle=False)
    E_full = d["E"].astype(np.float32)
    t_seconds = d["t_seconds"].astype(np.float64)
    truth = {
        "D_E": float(d["D_E_true"]),
        "w_EE": float(d["w_EE_true"]),
        "h": float(d["h_true"]),
    }
    tau = float(d["tau"])
    l_fov = float(d["l_fov"])

    # Fixed pools.
    sx, sy = make_state_pool(
        E_full, args.state_pool, args.seed + 1,
        early_fraction=0.50, early_seconds=5.0
    )
    ix, iy = make_ic_pool(E_full, args.ic_pool, args.seed + 2)
    dx, dE, dEt, dLap = make_derivative_pool(
        E_full, l_fov, t_seconds, args.deriv_pool, args.seed + 3,
        early_fraction=0.60, early_seconds=5.0
    )

    S = (
        torch.from_numpy(sx).to(device),
        torch.from_numpy(sy).to(device),
    )
    I = (
        torch.from_numpy(ix).to(device),
        torch.from_numpy(iy).to(device),
    )
    DPOOL = (
        torch.from_numpy(dx).to(device),
        torch.from_numpy(dE).to(device),
        torch.from_numpy(dEt).to(device),
        torch.from_numpy(dLap).to(device),
    )

    model = gPINN_WilsonCowan(fourier_seed=args.fourier_seed).to(device)

    raw_D, raw_D_name = find_raw_D_parameter(model)
    phys_ids = {id(raw_D), id(model.w_EE), id(model.h)}
    network_params = [p for p in model.parameters() if id(p) not in phys_ids]
    physical_params = [raw_D, model.w_EE, model.h]

    history = []

    # ---------------- Stage A: state + IC pretraining ---------------- #
    print("\n=== Stage A: state + initial-condition pretraining ===", flush=True)
    for p_ in physical_params:
        p_.requires_grad_(False)
    for p_ in network_params:
        p_.requires_grad_(True)

    optA = torch.optim.Adam(network_params, lr=args.lrA)

    for step in range(1, args.stageA_steps + 1):
        xb, yb = batch_from_pool(S, args.state_batch)
        xic, yic = batch_from_pool(I, args.ic_batch)

        pred = model.forward_E(xb)
        pred_ic = model.forward_E(xic)

        loss_data = normalized_mse(pred, yb)
        loss_ic = normalized_mse(pred_ic, yic)
        loss = loss_data + 1.0 * loss_ic

        optA.zero_grad(set_to_none=True)
        loss.backward()
        optA.step()

        if step == 1 or step % 100 == 0 or step == args.stageA_steps:
            print(
                f"A {step:5d}/{args.stageA_steps} "
                f"loss={loss.item():.4e} data={loss_data.item():.4e} "
                f"ic={loss_ic.item():.4e}",
                flush=True,
            )
            history.append(["A", step, float(loss.item()),
                            float(loss_data.item()), float(loss_ic.item()),
                            "", "", ""])

    # -------------- Stage B: matched non-oracle state/IC fit ----------- #
    print("\n=== Stage B: matched non-oracle state + IC training ===", flush=True)
    optB = torch.optim.Adam(network_params, lr=args.lrB)

    for step in range(1, args.stageB_steps + 1):
        xb, yb = batch_from_pool(S, args.state_batch)
        xic, yic = batch_from_pool(I, args.ic_batch)

        pred = model.forward_E(xb)
        pred_ic = model.forward_E(xic)

        loss_data = normalized_mse(pred, yb)
        loss_ic = normalized_mse(pred_ic, yic)
        loss = 1.0 * loss_data + 0.75 * loss_ic

        optB.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network_params, max_norm=10.0)
        optB.step()

        if step == 1 or step % 50 == 0 or step == args.stageB_steps:
            print(
                f"B {step:5d}/{args.stageB_steps} loss={loss.item():.4e} "
                f"data={loss_data.item():.3e} ic={loss_ic.item():.3e}",
                flush=True,
            )
            history.append(["B", step, float(loss.item()),
                            float(loss_data.item()), float(loss_ic.item()),
                            "", "", ""])

    # Evaluate derivative quality before parameter fitting.
    eval_n = min(12000, DPOOL[0].shape[0])
    eval_idx = torch.arange(eval_n, device=device)
    eval_x = DPOOL[0][eval_idx]
    eval_E_true = DPOOL[1][eval_idx].detach().cpu().numpy().ravel()
    eval_Et_true = DPOOL[2][eval_idx].detach().cpu().numpy().ravel()
    eval_Lap_true = DPOOL[3][eval_idx].detach().cpu().numpy().ravel()

    E_pre, Et_pre, Lap_pre = predict_derivatives(
        model, eval_x, args.duration, l_fov, batch_size=args.deriv_batch
    )
    pre_param_derivative_metrics = {
        "E": metric_block(eval_E_true, E_pre),
        "E_t": metric_block(eval_Et_true, Et_pre),
        "laplacian": metric_block(eval_Lap_true, Lap_pre),
    }
    print("\nDerivative metrics after non-oracle Stage B:", flush=True)
    print(json.dumps(pre_param_derivative_metrics, indent=2), flush=True)

    # ---------------- Stage C: parameter-only optimization ------------ #
    print("\n=== Stage C: freeze network, optimize D_E / w_EE / h ===", flush=True)
    for p_ in network_params:
        p_.requires_grad_(False)
    for p_ in physical_params:
        p_.requires_grad_(True)

    # Cache learned E/derivatives on the derivative pool. No weight gradients now.
    cache_E, cache_Et, cache_Lap = [], [], []
    model.eval()
    for s in range(0, DPOOL[0].shape[0], args.deriv_batch):
        Ee, Ete, Lape = state_and_derivatives(
            model, DPOOL[0][s:s+args.deriv_batch],
            args.duration, l_fov, create_graph=False
        )
        cache_E.append(Ee.detach())
        cache_Et.append(Ete.detach())
        cache_Lap.append(Lape.detach())

    cache_E = torch.cat(cache_E, dim=0)
    cache_Et = torch.cat(cache_Et, dim=0)
    cache_Lap = torch.cat(cache_Lap, dim=0)

    optC = torch.optim.Adam(physical_params, lr=args.lrC)

    for step in range(1, args.stageC_adam_steps + 1):
        n = cache_E.shape[0]
        idx = torch.randint(0, n, (min(8192, n),), device=device)
        residual = physics_residual(
            model,
            cache_E[idx],
            cache_Et[idx],
            cache_Lap[idx],
            tau,
        )
        loss_phys = torch.mean(residual ** 2)

        optC.zero_grad(set_to_none=True)
        loss_phys.backward()
        optC.step()

        # Mirror the hard w_EE constraint used by the inverse pipeline.
        with torch.no_grad():
            model.w_EE.clamp_(0.5, 6.0)

        if step == 1 or step % 100 == 0 or step == args.stageC_adam_steps:
            est = physical_parameter_dict(model)
            print(
                f"C {step:5d}/{args.stageC_adam_steps} "
                f"physics={loss_phys.item():.4e} "
                f"D={est['D_E']:.6f} w={est['w_EE']:.6f} h={est['h']:.6f}",
                flush=True,
            )
            history.append(["C", step, float(loss_phys.item()),
                            "", "", "", "", json.dumps(est)])

    stageC_estimate = physical_parameter_dict(model)
    print("Stage C estimate:", json.dumps(stageC_estimate, indent=2), flush=True)

    # ---------------- Stage D: joint low-LR non-oracle fine tuning ------ #
    print("\n=== Stage D: joint non-oracle physics fine-tuning ===", flush=True)
    for p_ in network_params:
        p_.requires_grad_(True)
    for p_ in physical_params:
        p_.requires_grad_(True)

    optD = torch.optim.Adam([
        {"params": network_params, "lr": args.lrD_net},
        {"params": physical_params, "lr": args.lrD_phys},
    ])

    for step in range(1, args.stageD_steps + 1):
        xb, yb = batch_from_pool(S, args.state_batch)
        xic, yic = batch_from_pool(I, args.ic_batch)
        xd, _, _, _ = batch_from_pool(DPOOL, args.deriv_batch)

        pred = model.forward_E(xb)
        pred_ic = model.forward_E(xic)
        Ed_pred, Et_pred, Lap_pred = state_and_derivatives(
            model, xd, args.duration, l_fov, create_graph=True
        )
        residual = physics_residual(model, Ed_pred, Et_pred, Lap_pred, tau)

        loss_data = normalized_mse(pred, yb)
        loss_ic = normalized_mse(pred_ic, yic)
        loss_phys = torch.mean(residual ** 2) / 1e-3

        loss = 1.0 * loss_data + 0.50 * loss_ic + 0.25 * loss_phys

        optD.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optD.step()

        with torch.no_grad():
            model.w_EE.clamp_(0.5, 6.0)

        if step == 1 or step % 50 == 0 or step == args.stageD_steps:
            est = physical_parameter_dict(model)
            print(
                f"D {step:5d}/{args.stageD_steps} loss={loss.item():.4e} "
                f"phys={loss_phys.item():.3e} "
                f"D={est['D_E']:.6f} w={est['w_EE']:.6f} h={est['h']:.6f}",
                flush=True,
            )
            history.append(["D", step, float(loss.item()),
                            float(loss_data.item()), float(loss_ic.item()),
                            "", "", json.dumps(est)])

    # --------------------------- final evaluation ---------------------- #
    model.eval()
    final_E, final_Et, final_Lap = predict_derivatives(
        model, eval_x, args.duration, l_fov, batch_size=args.deriv_batch
    )

    final_derivative_metrics = {
        "E": metric_block(eval_E_true, final_E),
        "E_t": metric_block(eval_Et_true, final_Et),
        "laplacian": metric_block(eval_Lap_true, final_Lap),
    }

    final_estimate = physical_parameter_dict(model)
    final_relerr = relative_errors(truth, final_estimate)

    # Residual on learned field.
    res_learned = (
        tau * final_Et
        - (
            -final_E
            + sigmoid_np(
                final_estimate["w_EE"] * final_E
                + final_estimate["D_E"] * final_Lap
                + final_estimate["h"]
            )
        )
    )

    report = {
        "experiment": "synthetic_staged_nonoracle_control",
        "important_note": (
            "Matched staged control with no oracle derivative targets. "
            "Derivative metrics are evaluated only after training for comparison."
        ),
        "seed": args.seed,
        "fourier_seed": args.fourier_seed,
        "device": str(device),
        "truth": truth,
        "initial_parameterization": {
            "D_E": float(0.1 + 4.9 * 0.5),
            "w_EE": 2.0,
            "h": -1.0,
        },
        "raw_D_parameter_name": raw_D_name,
        "after_stageB_diagnostic_derivative_metrics": pre_param_derivative_metrics,
        "stageC_parameter_only_estimate": stageC_estimate,
        "final_estimate": final_estimate,
        "final_relative_error_percent": final_relerr,
        "final_derivative_metrics": final_derivative_metrics,
        "final_learned_field_pde_residual_mse": float(np.mean(res_learned ** 2)),
        "settings": vars(args),
    }

    print("\n=== FINAL REPORT ===", flush=True)
    print(json.dumps(report, indent=2), flush=True)

    tag = f"seed-{args.seed}_staged_nonoracle"
    (out / f"{tag}.json").write_text(json.dumps(report, indent=2))

    torch.save({
        "model_state": model.state_dict(),
        "report": report,
    }, out / f"{tag}.pt")

    with open(out / "training_history.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "stage", "step", "loss", "data_loss", "ic_loss",
            "Et_loss", "lap_loss", "parameter_estimate"
        ])
        writer.writerows(history)

    scatter(
        eval_E_true, final_E,
        "Synthetic truth E", "Staged non-oracle gPINN E",
        "State accuracy after staged non-oracle control",
        out / "scatter_E.png",
    )
    scatter(
        eval_Et_true, final_Et,
        "Finite-difference truth E_t", "Staged non-oracle gPINN E_t",
        "Time-derivative accuracy after staged non-oracle control",
        out / "scatter_Et.png",
    )
    scatter(
        eval_Lap_true, final_Lap,
        "Finite-difference truth Laplacian(E)",
        "Staged non-oracle gPINN Laplacian(E)",
        "Laplacian accuracy after staged non-oracle control",
        out / "scatter_laplacian.png",
    )

    print(f"\nSaved outputs to {out}", flush=True)


if __name__ == "__main__":
    main()
