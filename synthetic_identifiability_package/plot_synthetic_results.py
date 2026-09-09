#!/usr/bin/env python3
"""Aggregate synthetic recovery runs and create manuscript-ready diagnostics."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from wc_gpinn_core import gPINN_WilsonCowan


def load_jsons(results_dir: Path):
    rows = []
    for path in sorted(results_dir.glob("seed-*_noise-*.json")):
        with open(path) as f:
            row = json.load(f)
        row["_path"] = str(path)
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No result JSON files found in {results_dir}")
    return rows


def write_summary_csv(rows, out_path: Path):
    fields = [
        "noise_sigma", "seed",
        "D_E_true", "D_E_est", "D_E_relerr_pct",
        "w_EE_true", "w_EE_est", "w_EE_relerr_pct",
        "h_true", "h_est", "h_relerr_pct",
        "raw_D", "val_r2_observed", "val_r2_clean", "val_mse_clean",
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({
                "noise_sigma": r["noise_sigma"], "seed": r["seed"],
                "D_E_true": r["truth"]["D_E"], "D_E_est": r["estimate"]["D_E"],
                "D_E_relerr_pct": r["relative_error_percent"]["D_E"],
                "w_EE_true": r["truth"]["w_EE"], "w_EE_est": r["estimate"]["w_EE"],
                "w_EE_relerr_pct": r["relative_error_percent"]["w_EE"],
                "h_true": r["truth"]["h"], "h_est": r["estimate"]["h"],
                "h_relerr_pct": r["relative_error_percent"]["h"],
                "raw_D": r["estimate"]["raw_D"],
                "val_r2_observed": r["val_r2_observed"],
                "val_r2_clean": r["val_r2_clean"],
                "val_mse_clean": r["val_mse_clean"],
            })


def aggregate_table(rows, out_path: Path):
    lines = []
    for noise in sorted({float(r["noise_sigma"]) for r in rows}):
        subset = [r for r in rows if float(r["noise_sigma"]) == noise]
        truth = subset[0]["truth"]
        lines.append({"noise_sigma": noise, "n": len(subset)})
        for p in ("D_E", "w_EE", "h"):
            vals = np.array([r["estimate"][p] for r in subset], dtype=float)
            rel = np.abs(vals - truth[p]) / max(abs(truth[p]), 1e-12) * 100.0
            lines[-1][f"{p}_true"] = truth[p]
            lines[-1][f"{p}_mean"] = vals.mean()
            lines[-1][f"{p}_sd"] = vals.std(ddof=1) if len(vals) > 1 else 0.0
            lines[-1][f"{p}_mean_relerr_pct"] = rel.mean()
            lines[-1][f"{p}_cv_pct"] = (
                lines[-1][f"{p}_sd"] / abs(vals.mean()) * 100.0 if abs(vals.mean()) > 1e-12 else np.nan
            )

    keys = list(lines[0].keys())
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(lines)
    return lines


def plot_parameter(rows, parameter: str, figure_dir: Path):
    noises = sorted({float(r["noise_sigma"]) for r in rows})
    truth = float(rows[0]["truth"][parameter])
    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    for i, noise in enumerate(noises):
        subset = [r for r in rows if float(r["noise_sigma"]) == noise]
        vals = np.array([r["estimate"][parameter] for r in subset])
        jitter = np.linspace(-0.08, 0.08, len(vals)) if len(vals) > 1 else np.array([0.0])
        ax.scatter(np.full(len(vals), i) + jitter, vals, label=f"σ={noise:g}")
        ax.errorbar(i, vals.mean(), yerr=vals.std(ddof=1) if len(vals) > 1 else 0.0,
                    fmt="o", capsize=4)
    ax.axhline(truth, linestyle="--", label="Ground truth")
    ax.set_xticks(range(len(noises)), [f"σ={n:g}" for n in noises])
    ax.set_ylabel(parameter)
    ax.set_xlabel("Observation-noise level")
    ax.set_title(f"Synthetic recovery of {parameter}")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / f"recovery_{parameter}.png", dpi=300)
    fig.savefig(figure_dir / f"recovery_{parameter}.pdf")
    plt.close(fig)


def finite_difference_terms(E: np.ndarray, dt: float, dx: float):
    # Central time derivative; spatial derivatives use interior pixels only.
    dEdt = (E[2:] - E[:-2]) / (2.0 * dt)
    mid = E[1:-1]
    lap = (
        mid[:, 1:-1, 2:] + mid[:, 1:-1, :-2]
        + mid[:, 2:, 1:-1] + mid[:, :-2, 1:-1]
        - 4.0 * mid[:, 1:-1, 1:-1]
    ) / (dx * dx)
    return mid[:, 1:-1, 1:-1], dEdt[:, 1:-1, 1:-1], lap


def residual_mse(E, dEdt, lap, D, w, h, tau=1.0):
    u = np.clip(w * E + D * lap + h, -40.0, 40.0)
    S = 1.0 / (1.0 + np.exp(-u))
    f = tau * dEdt - (-E + S)
    return float(np.mean(f * f))


def compute_and_plot_landscape(data_path: Path, figure_dir: Path, max_points: int = 50000):
    d = np.load(data_path)
    Eall = d["E"].astype(np.float64)
    D0 = float(d["D_E_true"]); w0 = float(d["w_EE_true"]); h0 = float(d["h_true"])
    tau = float(d["tau"]); L = float(d["l_fov"])
    dt = 1.0 / 30.0
    dx = L / (Eall.shape[2] - 1)
    E, dEdt, lap = finite_difference_terms(Eall, dt, dx)
    E = E.ravel(); dEdt = dEdt.ravel(); lap = lap.ravel()
    rng = np.random.default_rng(2026)
    if E.size > max_points:
        idx = rng.choice(E.size, max_points, replace=False)
        E, dEdt, lap = E[idx], dEdt[idx], lap[idx]

    configs = [
        ("D_E", "w_EE", np.linspace(max(0.11, D0 * 0.35), min(4.9, D0 * 2.1), 41),
         np.linspace(max(0.55, w0 - 1.4), min(5.95, w0 + 0.7), 41)),
        ("D_E", "h", np.linspace(max(0.11, D0 * 0.35), min(4.9, D0 * 2.1), 41),
         np.linspace(h0 - 1.0, h0 + 1.0, 41)),
        ("w_EE", "h", np.linspace(max(0.55, w0 - 1.4), min(5.95, w0 + 0.7), 41),
         np.linspace(h0 - 1.0, h0 + 1.0, 41)),
    ]
    truth = {"D_E": D0, "w_EE": w0, "h": h0}

    for a, b, avals, bvals in configs:
        Z = np.empty((len(bvals), len(avals)), dtype=np.float64)
        for j, bv in enumerate(bvals):
            for i, av in enumerate(avals):
                params = truth.copy(); params[a] = float(av); params[b] = float(bv)
                Z[j, i] = residual_mse(E, dEdt, lap, params["D_E"], params["w_EE"], params["h"], tau)

        np.savez_compressed(figure_dir / f"landscape_{a}_{b}.npz", a=avals, b=bvals, loss=Z,
                            a_name=a, b_name=b, a_true=truth[a], b_true=truth[b])
        fig, ax = plt.subplots(figsize=(6.2, 5.0))
        im = ax.imshow(np.log10(Z + 1e-16), origin="lower", aspect="auto",
                       extent=[avals[0], avals[-1], bvals[0], bvals[-1]])
        ax.scatter([truth[a]], [truth[b]], marker="*", s=150, label="Ground truth")
        ax.set_xlabel(a); ax.set_ylabel(b)
        ax.set_title(f"PDE residual landscape: {a} vs {b}")
        cbar = fig.colorbar(im, ax=ax); cbar.set_label("log10 residual MSE")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figure_dir / f"landscape_{a}_{b}.png", dpi=300)
        fig.savefig(figure_dir / f"landscape_{a}_{b}.pdf")
        plt.close(fig)


def xyt_for_frame(frame: int, T: int, H: int, W: int):
    x = np.linspace(-1, 1, W, dtype=np.float32)
    y = np.linspace(-1, 1, H, dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    t = np.full_like(xx, -1.0 + 2.0 * frame / float(T - 1))
    return np.stack([xx.ravel(), yy.ravel(), t.ravel()], axis=1)


def reconstruct_best_clean_run(rows, data_path: Path, output_dir: Path, figure_dir: Path):
    clean = [r for r in rows if abs(float(r["noise_sigma"])) < 1e-12]
    if not clean:
        return
    best = min(clean, key=lambda r: sum(r["relative_error_percent"][p] for p in ("D_E", "w_EE", "h")))
    ckpt_path = Path(best["checkpoint"])
    if not ckpt_path.is_absolute():
        # The training JSON generally contains a path relative to project working dir.
        candidate = output_dir.parent / ckpt_path
        if candidate.exists(): ckpt_path = candidate
        else: ckpt_path = output_dir / "checkpoints" / ckpt_path.name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model = gPINN_WilsonCowan(fourier_seed=best.get("fourier_seed", 2718)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    d = np.load(data_path)
    E = d["E"].astype(np.float32)
    T, H, W = E.shape
    for frame in (0, T // 2, T - 1):
        X = xyt_for_frame(frame, T, H, W)
        preds = []
        with torch.no_grad():
            for start in range(0, len(X), 32768):
                xb = torch.from_numpy(X[start:start+32768]).to(device)
                preds.append(model.forward_E(xb).cpu().numpy())
        pred = np.concatenate(preds).reshape(H, W)
        err = np.abs(pred - E[frame])
        time_s = frame / 30.0
        for name, img, title in [
            ("true", E[frame], f"Synthetic truth at t={time_s:.1f} s"),
            ("pred", pred, f"gPINN reconstruction at t={time_s:.1f} s"),
            ("abs_error", err, f"Absolute error at t={time_s:.1f} s"),
        ]:
            fig, ax = plt.subplots(figsize=(5.3, 4.6))
            im = ax.imshow(img, origin="lower")
            ax.set_title(title); ax.set_xlabel("x pixel"); ax.set_ylabel("y pixel")
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            fig.savefig(figure_dir / f"frame-{frame:03d}_{name}.png", dpi=300)
            fig.savefig(figure_dir / f"frame-{frame:03d}_{name}.pdf")
            plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="synthetic_outputs")
    p.add_argument("--data", default="synthetic_data/wc_twin_truth.npz")
    args = p.parse_args()
    out = Path(args.output_dir)
    figdir = out / "figures"; figdir.mkdir(parents=True, exist_ok=True)
    rows = load_jsons(out / "results")
    write_summary_csv(rows, out / "synthetic_recovery_runs.csv")
    aggregate_table(rows, out / "synthetic_recovery_summary.csv")
    for pnm in ("D_E", "w_EE", "h"):
        plot_parameter(rows, pnm, figdir)
    compute_and_plot_landscape(Path(args.data), figdir)
    reconstruct_best_clean_run(rows, Path(args.data), out, figdir)
    print(f"Wrote summaries and figures under: {out}")


if __name__ == "__main__":
    main()
