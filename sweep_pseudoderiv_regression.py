#!/usr/bin/env python3
"""
Noise × smoothing sensitivity sweep for real-compatible Wilson–Cowan
pseudo-derivative regression.

This experiment does NOT train a PINN. It asks whether the parameter estimator

    logit(tau E_t + E) = w_EE E + D_E Lap(E) + h

remains accurate when the observed synthetic field is noisy and derivatives are
estimated after Gaussian smoothing.

Each Slurm array task corresponds to one (noise_sigma, noise_seed) condition and
loops over a small grid of temporal/spatial smoothing strengths.

Outputs:
  synthetic_outputs/regression_sensitivity/task-XXX.csv
  synthetic_outputs/regression_sensitivity/task-XXX.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import lsq_linear


NOISE_SEEDS = [42, 123, 456, 789, 2026]
TASKS = [(0.0, 42)]
for sigma in (0.005, 0.01, 0.02):
    TASKS.extend((sigma, seed) for seed in NOISE_SEEDS)

SMOOTH_T = [0.0, 0.5, 1.0, 2.0]       # frames
SMOOTH_XY = [0.0, 0.5, 1.0, 2.0]      # pixels


def r2_score(y_true, y_pred):
    a = np.asarray(y_true).ravel()
    b = np.asarray(y_pred).ravel()
    ss_res = np.sum((a - b) ** 2)
    ss_tot = np.sum((a - a.mean()) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-12))


def rel_l2(y_true, y_pred):
    a = np.asarray(y_true).ravel()
    b = np.asarray(y_pred).ravel()
    return float(np.linalg.norm(b - a) / (np.linalg.norm(a) + 1e-12))


def bounded_logit_regression(E, dE_dt, lap, tau, eps=1e-4):
    E = np.asarray(E).ravel().astype(np.float64)
    dE_dt = np.asarray(dE_dt).ravel().astype(np.float64)
    lap = np.asarray(lap).ravel().astype(np.float64)

    q = tau * dE_dt + E
    valid = np.isfinite(q) & np.isfinite(E) & np.isfinite(lap)
    valid &= (q > eps) & (q < 1.0 - eps)

    if valid.sum() < 100:
        return {
            "success": False,
            "n_valid": int(valid.sum()),
            "fraction_valid": float(valid.mean()),
        }

    qv = q[valid]
    Ev = E[valid]
    Lv = lap[valid]

    y = np.log(qv / (1.0 - qv))
    X = np.column_stack([Ev, Lv, np.ones_like(Ev)])

    fit = lsq_linear(
        X, y,
        bounds=(
            np.array([0.5, 0.1, -np.inf]),   # w, D, h
            np.array([6.0, 5.0,  np.inf]),
        ),
        method="trf",
        lsmr_tol="auto",
        verbose=0,
    )

    w_hat, D_hat, h_hat = map(float, fit.x)

    Xstd = np.column_stack([
        (Ev - Ev.mean()) / (Ev.std() + 1e-12),
        (Lv - Lv.mean()) / (Lv.std() + 1e-12),
    ])

    return {
        "success": bool(fit.success),
        "n_valid": int(valid.sum()),
        "fraction_valid": float(valid.mean()),
        "D_E": D_hat,
        "w_EE": w_hat,
        "h": h_hat,
        "cost": float(fit.cost),
        "condition_number_raw_X": float(np.linalg.cond(X)),
        "condition_number_standardized_E_lap": float(np.linalg.cond(Xstd)),
        "corr_E_laplacian": float(np.corrcoef(Ev, Lv)[0, 1]),
    }


def make_fixed_indices(shape, t_seconds, n_points, seed,
                       early_fraction=0.60, early_seconds=5.0):
    T, H, W = shape
    rng = np.random.default_rng(seed)
    dt = float(np.median(np.diff(t_seconds)))

    early_last = min(T - 2, max(1, int(round(early_seconds / dt))))
    n_early = int(round(n_points * early_fraction))
    n_uniform = n_points - n_early

    ti_e = rng.integers(1, early_last + 1, size=n_early)
    yi_e = rng.integers(1, H - 1, size=n_early)
    xi_e = rng.integers(1, W - 1, size=n_early)

    ti_u = rng.integers(1, T - 1, size=n_uniform)
    yi_u = rng.integers(1, H - 1, size=n_uniform)
    xi_u = rng.integers(1, W - 1, size=n_uniform)

    return (
        np.concatenate([ti_e, ti_u]),
        np.concatenate([yi_e, yi_u]),
        np.concatenate([xi_e, xi_u]),
    )


def derivatives_at_indices(E, ti, yi, xi, l_fov, t_seconds):
    T, H, W = E.shape
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

    return E0, dE_dt, d2x + d2y


def pct_err(est, true):
    return float(100.0 * abs(est - true) / (abs(true) + 1e-12))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="synthetic_data/wc_twin_truth.npz")
    p.add_argument("--output_dir",
                   default="synthetic_outputs/regression_sensitivity")
    p.add_argument("--task_id", type=int, required=True)
    p.add_argument("--n_points", type=int, default=24000)
    p.add_argument("--clip_observed", action="store_true")
    args = p.parse_args()

    if args.task_id < 0 or args.task_id >= len(TASKS):
        raise ValueError(f"task_id must be 0..{len(TASKS)-1}")

    noise_sigma, noise_seed = TASKS[args.task_id]

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    d = np.load(args.data, allow_pickle=False)
    E_true = d["E"].astype(np.float32)
    t_seconds = d["t_seconds"].astype(np.float64)
    tau = float(d["tau"])
    l_fov = float(d["l_fov"])

    truth = {
        "D_E": float(d["D_E_true"]),
        "w_EE": float(d["w_EE_true"]),
        "h": float(d["h_true"]),
    }

    # Same evaluation coordinates for every smoothing condition in this task.
    ti, yi, xi = make_fixed_indices(
        E_true.shape, t_seconds, args.n_points, seed=909
    )

    E_truth_s, Et_truth_s, Lap_truth_s = derivatives_at_indices(
        E_true, ti, yi, xi, l_fov, t_seconds
    )

    rng = np.random.default_rng(noise_seed + 404)
    E_obs = E_true.astype(np.float64).copy()
    if noise_sigma > 0:
        E_obs += noise_sigma * rng.standard_normal(E_obs.shape)
    if args.clip_observed:
        E_obs = np.clip(E_obs, 0.0, 1.0)

    rows = []

    print(
        f"Task {args.task_id}: noise_sigma={noise_sigma}, "
        f"noise_seed={noise_seed}", flush=True
    )

    for sigma_t in SMOOTH_T:
        for sigma_xy in SMOOTH_XY:
            if sigma_t == 0.0 and sigma_xy == 0.0:
                E_smooth = E_obs
            else:
                E_smooth = gaussian_filter(
                    E_obs,
                    sigma=(sigma_t, sigma_xy, sigma_xy),
                    mode="reflect",
                )

            E_s, Et_s, Lap_s = derivatives_at_indices(
                E_smooth, ti, yi, xi, l_fov, t_seconds
            )

            reg = bounded_logit_regression(E_s, Et_s, Lap_s, tau=tau)

            row = {
                "task_id": args.task_id,
                "noise_sigma": noise_sigma,
                "noise_seed": noise_seed,
                "smooth_sigma_t_frames": sigma_t,
                "smooth_sigma_space_pixels": sigma_xy,
                "n_points": args.n_points,
                "clip_observed": bool(args.clip_observed),
                "E_smooth_r2_vs_truth": r2_score(E_truth_s, E_s),
                "Et_pseudo_r2_vs_truth": r2_score(Et_truth_s, Et_s),
                "Et_pseudo_rel_l2": rel_l2(Et_truth_s, Et_s),
                "lap_pseudo_r2_vs_truth": r2_score(Lap_truth_s, Lap_s),
                "lap_pseudo_rel_l2": rel_l2(Lap_truth_s, Lap_s),
                "success": reg.get("success", False),
                "n_valid": reg.get("n_valid", 0),
                "fraction_valid": reg.get("fraction_valid", 0.0),
            }

            if reg.get("success", False):
                row.update({
                    "D_E_true": truth["D_E"],
                    "D_E_est": reg["D_E"],
                    "D_E_relerr_pct": pct_err(reg["D_E"], truth["D_E"]),
                    "w_EE_true": truth["w_EE"],
                    "w_EE_est": reg["w_EE"],
                    "w_EE_relerr_pct": pct_err(reg["w_EE"], truth["w_EE"]),
                    "h_true": truth["h"],
                    "h_est": reg["h"],
                    "h_relerr_pct": pct_err(reg["h"], truth["h"]),
                    "mean_param_relerr_pct": float(np.mean([
                        pct_err(reg["D_E"], truth["D_E"]),
                        pct_err(reg["w_EE"], truth["w_EE"]),
                        pct_err(reg["h"], truth["h"]),
                    ])),
                    "max_param_relerr_pct": float(np.max([
                        pct_err(reg["D_E"], truth["D_E"]),
                        pct_err(reg["w_EE"], truth["w_EE"]),
                        pct_err(reg["h"], truth["h"]),
                    ])),
                    "condition_number_raw_X": reg["condition_number_raw_X"],
                    "condition_number_standardized_E_lap":
                        reg["condition_number_standardized_E_lap"],
                    "corr_E_laplacian": reg["corr_E_laplacian"],
                    "cost": reg["cost"],
                })

            rows.append(row)

            msg = (
                f"sigma_t={sigma_t:>3}, sigma_xy={sigma_xy:>3} | "
                f"valid={row['fraction_valid']:.3f}"
            )
            if row["success"]:
                msg += (
                    f" | D={row['D_E_est']:.4f} "
                    f"w={row['w_EE_est']:.4f} h={row['h_est']:.4f} "
                    f"| mean_err={row['mean_param_relerr_pct']:.2f}%"
                )
            print(msg, flush=True)

    csv_path = out / f"task-{args.task_id:03d}.csv"
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "task_id": args.task_id,
        "noise_sigma": noise_sigma,
        "noise_seed": noise_seed,
        "truth": truth,
        "smooth_sigma_t_values": SMOOTH_T,
        "smooth_sigma_space_values": SMOOTH_XY,
        "rows": rows,
    }
    json_path = out / f"task-{args.task_id:03d}.json"
    json_path.write_text(json.dumps(payload, indent=2))

    print(f"\nSaved {csv_path}", flush=True)
    print(f"Saved {json_path}", flush=True)


if __name__ == "__main__":
    main()
