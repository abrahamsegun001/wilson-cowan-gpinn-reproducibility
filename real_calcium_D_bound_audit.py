#!/usr/bin/env python3
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

from __future__ import annotations

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


if __name__ == "__main__":
    main()
