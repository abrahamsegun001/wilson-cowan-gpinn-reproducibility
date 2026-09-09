#!/usr/bin/env python3
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


if __name__ == "__main__":
    main()
