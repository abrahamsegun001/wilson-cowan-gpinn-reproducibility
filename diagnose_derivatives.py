#!/usr/bin/env python3
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
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from wc_gpinn_core import gPINN_WilsonCowan


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


if __name__ == "__main__":
    main()
