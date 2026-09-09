#!/usr/bin/env python3
"""
Publication V2 figures for the regression-assisted, model-selective sequential
Wilson-Cowan gPINN study.

No training is performed.

Main changes from V1
--------------------
1. Figure 1 uses a two-lane conceptual layout separating parameter inference
   from neural-state estimation.
2. Figure 2C no longer connects unrelated physical parameters with lines.
3. Figure 2D evaluates the single robust smoothing configuration across noise
   levels instead of showing an oracle "best at each noise" curve.
4. Figure 3 focuses entirely on physical inference/model selection.
5. Figure 4 uses a common observed/predicted color scale, restores absolute
   error maps, adds session-wide R², and moves the PDE residual to Supplement S1.
6. Figure 5 highlights lambda_cont=0.05 and clarifies boundary labeling.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project_root", default=".")
    p.add_argument("--master_dir", default="outputs/publication_v2/master")
    p.add_argument("--final_run_dir", default="outputs/final_dual_model_sequential20_cont005")
    p.add_argument("--data_dir", default="data/processed")
    p.add_argument("--subject_prefix", default="sub-GC48_ses-day0")
    p.add_argument("--representative_chunk", type=int, default=12)
    p.add_argument("--output_dir", default="outputs/publication_v2/figures")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--scatter_points", type=int, default=100000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def save_both(fig, pathstem: Path, dpi: int):
    pathstem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pathstem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(pathstem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def panel_label(ax, label):
    ax.text(-0.10, 1.04, label, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="bottom")


def add_box(ax, xy, w, h, text, fontsize=9.5, lw=1.3):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        fill=False, linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fontsize)
    return patch


def arrow(ax, a, b, **kwargs):
    ax.annotate("", xy=b, xytext=a,
                arrowprops=dict(arrowstyle="->", lw=1.35, **kwargs))


def figure1_pipeline(out_dir: Path, dpi: int):
    fig, ax = plt.subplots(figsize=(13.5, 7.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.03, 0.91, "A  Parameter inference and model selection",
            fontsize=12, fontweight="bold")
    ax.text(0.03, 0.48, "B  Neural-state estimation and sequential transfer",
            fontsize=12, fontweight="bold")

    # Lane A
    add_box(ax, (0.04, 0.68), 0.17, 0.16,
            "30-s calcium chunk\n64×64, 30 Hz\nsession-normalized")
    add_box(ax, (0.28, 0.68), 0.19, 0.16,
            "Resolution-aware smoothing\n+ finite-difference\npseudo-derivatives")
    add_box(ax, (0.54, 0.68), 0.18, 0.16,
            "Full vs reduced\nlogit regression\n+ uncertainty checks")
    add_box(ax, (0.79, 0.68), 0.17, 0.16,
            "Evidence-based\nmodel selection\nfull or $D_E=0$")
    arrow(ax, (0.21, 0.76), (0.28, 0.76))
    arrow(ax, (0.47, 0.76), (0.54, 0.76))
    arrow(ax, (0.72, 0.76), (0.79, 0.76))

    # Lane B
    add_box(ax, (0.79, 0.27), 0.17, 0.15,
            "Fix current-chunk\n$(w_{EE},D_E,h)$\nduring neural training")
    add_box(ax, (0.54, 0.27), 0.18, 0.15,
            "Stage A: state fit\nStage B: physics +\npseudo-derivative training")
    add_box(ax, (0.29, 0.27), 0.18, 0.15,
            "Sequential transfer\ncarry neural-state\nweights only")
    add_box(ax, (0.04, 0.27), 0.17, 0.15,
            "Boundary regularization\n$\\lambda_{cont}=0.05$\nfrom chunk 1 onward")
    arrow(ax, (0.875, 0.68), (0.875, 0.42))
    arrow(ax, (0.79, 0.345), (0.72, 0.345))
    arrow(ax, (0.54, 0.345), (0.47, 0.345))
    arrow(ax, (0.29, 0.345), (0.21, 0.345))

    # Next chunk loop.
    ax.annotate(
        "next chunk",
        xy=(0.125, 0.68), xytext=(0.125, 0.17),
        ha="center", va="center",
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.62", lw=1.3),
        fontsize=9.5,
    )

    ax.text(
        0.50, 0.08,
        "Physical coefficients are re-estimated independently in every chunk; "
        "optimizer state is reset. Only the neural-state representation is propagated.",
        ha="center", va="center", fontsize=10.5
    )
    ax.set_title(
        "Regression-assisted, model-selective sequential Wilson–Cowan gPINN",
        fontsize=15, pad=10
    )
    save_both(fig, out_dir / "figure1_final_pipeline_v2", dpi)


def figure2_synthetic(master_dir: Path, out_dir: Path, dpi: int):
    runs_path = master_dir / "synthetic_recovery_runs.csv"
    diag_path = master_dir / "synthetic_derivative_diagnostic.json"
    methods_path = master_dir / "synthetic_method_comparison.csv"
    sens_path = master_dir / "synthetic_smoothing_sensitivity.json"
    smooth_path = master_dir / "synthetic_smoothing_summary.csv"

    if not runs_path.exists() or not diag_path.exists():
        print("WARNING: Figure 2 skipped: required synthetic files missing.")
        return

    runs = pd.read_csv(runs_path)
    diag = json.loads(diag_path.read_text())

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.8))

    # A: baseline joint inverse failure
    ax = axes[0, 0]
    panel_label(ax, "A")
    noises = sorted(runs["noise_sigma"].unique())
    x = np.arange(len(noises))
    width = 0.24
    fields = [
        ("D_E_relerr_pct", r"$D_E$"),
        ("w_EE_relerr_pct", r"$w_{EE}$"),
        ("h_relerr_pct", r"$h$"),
    ]
    for j, (field, label) in enumerate(fields):
        means = [runs.loc[runs.noise_sigma == n, field].mean() for n in noises]
        ax.bar(x + (j - 1) * width, means, width=width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n:g}" for n in noises])
    ax.set_xlabel(r"Added noise $\sigma$")
    ax.set_ylabel("Mean parameter relative error (%)")
    ax.set_title("Baseline joint gPINN recovers state but not parameters")
    ax.legend(frameon=False)

    # B: derivative fidelity
    ax = axes[0, 1]
    panel_label(ax, "B")
    metrics = [
        ("state_E_metrics", r"$E$"),
        ("time_derivative_Et_metrics", r"$E_t$"),
        ("laplacian_metrics", r"$\nabla^2 E$"),
    ]
    r2 = [float(diag.get(k, {}).get("r2", np.nan)) for k, _ in metrics]
    ax.bar(np.arange(3), r2)
    ax.axhline(0, linewidth=1)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels([lab for _, lab in metrics])
    ax.set_ylabel(r"$R^2$ against synthetic truth")
    ax.set_title("High state fidelity does not imply derivative fidelity")

    # C: scale-normalized method comparison (no connecting lines)
    ax = axes[1, 0]
    panel_label(ax, "C")
    if methods_path.exists():
        m = pd.read_csv(methods_path)
        truth_row = m[m["Method"] == "Truth"]
        if len(truth_row) == 1:
            truth = truth_row.iloc[0]
            m = m[m["Method"] != "Truth"].copy()
            params = ["D_E", "w_EE", "h"]
            labels = [r"$D_E$", r"$w_{EE}$", r"$h$"]
            xx = np.arange(len(params), dtype=float)
            offsets = np.linspace(-0.24, 0.24, max(len(m), 1))
            for off, (_, row) in zip(offsets, m.iterrows()):
                ratios = []
                for p in params:
                    tv = float(truth[p])
                    ratios.append(float(row[p]) / tv if abs(tv) > 1e-12 else np.nan)
                ax.scatter(xx + off, ratios, s=48, label=row["Method"])
            ax.axhline(1.0, linestyle="--", linewidth=1, label="Truth")
            ax.set_xticks(xx)
            ax.set_xticklabels(labels)
            ax.set_ylabel("Estimate / truth")
            ax.set_title("Derivative-informed estimation improves recoverability")
            ax.legend(frameon=False, fontsize=8)
        else:
            ax.axis("off")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "Method-comparison data unavailable", ha="center", va="center")

    # D: fixed robust smoothing choice, not oracle best-by-noise
    ax = axes[1, 1]
    panel_label(ax, "D")
    if sens_path.exists() and smooth_path.exists():
        sens = json.loads(sens_path.read_text())
        sm = pd.read_csv(smooth_path)
        robust = sens.get("robust_best_nonzero_noise", {})
        st = float(robust.get("smooth_sigma_t_frames", np.nan))
        sx = float(robust.get("smooth_sigma_space_pixels", np.nan))
        sub = sm[
            np.isclose(sm["smooth_sigma_t_frames"].astype(float), st) &
            np.isclose(sm["smooth_sigma_space_pixels"].astype(float), sx)
        ].sort_values("noise_sigma")
        if not sub.empty:
            ax.plot(sub["noise_sigma"], sub["mean_param_relerr_mean_pct"], marker="o")
            ax.set_xlabel(r"Added noise $\sigma$")
            ax.set_ylabel("Mean parameter relative error (%)")
            ax.set_title(
                rf"Fixed robust smoothing: $\sigma_t={st:g}$ frames, "
                rf"$\sigma_{{xy}}={sx:g}$ px"
            )
        else:
            ax.axis("off")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "Fixed-smoothing sensitivity data unavailable",
                ha="center", va="center")

    fig.suptitle(
        "Synthetic validation: state accuracy alone is insufficient for inverse parameter recovery",
        fontsize=14
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_both(fig, out_dir / "figure2_synthetic_recoverability_v2", dpi)


def figure3_real_model(master_dir: Path, out_dir: Path, dpi: int):
    p = master_dir / "final_continuity_chunkwise.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    d = pd.read_csv(p)
    chunks = d["chunk_idx"].to_numpy(int)
    mins = d["time_start_min"].to_numpy(float) if "time_start_min" in d else chunks * 0.5

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.8))

    # A: D_E with uncertainty
    ax = axes[0, 0]
    panel_label(ax, "A")
    D = d["full_regression_D_E"].to_numpy(float)
    lo = d["D_boot_ci95_low"].to_numpy(float)
    hi = d["D_boot_ci95_high"].to_numpy(float)
    yerr = np.vstack([D - lo, hi - D])
    ax.errorbar(mins, D, yerr=yerr, fmt="o-", capsize=2, linewidth=1)
    ax.axhline(0, linewidth=1)
    ax.set_xlabel("Session time (min)")
    ax.set_ylabel(r"Unrestricted $D_E$ estimate")
    ax.set_title("All 20 bootstrap intervals include zero")

    # B: w_EE
    ax = axes[0, 1]
    panel_label(ax, "B")
    w = d["selected_w_EE"].astype(float)
    ax.plot(mins, w, marker="o")
    ax.axhline(w.mean(), linestyle="--", linewidth=1,
               label=rf"mean={w.mean():.3f}")
    ax.set_xlabel("Session time (min)")
    ax.set_ylabel(r"$w_{EE}$")
    ax.set_title("Reduced-model recurrent excitation")
    ax.legend(frameon=False)

    # C: h
    ax = axes[1, 0]
    panel_label(ax, "C")
    h = d["selected_h"].astype(float)
    ax.plot(mins, h, marker="s")
    ax.axhline(h.mean(), linestyle="--", linewidth=1,
               label=rf"mean={h.mean():.3f}")
    ax.set_xlabel("Session time (min)")
    ax.set_ylabel(r"$h$")
    ax.set_title("Reduced-model external drive")
    ax.legend(frameon=False)

    # D: full-vs-reduced evidence
    ax = axes[1, 1]
    panel_label(ax, "D")
    ax.plot(mins, d["nested_logit_improvement_pct"], marker="o", label="Logit")
    ax.plot(mins, d["nested_pde_improvement_pct"], marker="s", label="PDE")
    ax.axhline(0.1, linestyle="--", linewidth=1, label="0.1% selection threshold")
    ax.axhline(0, linewidth=0.8)
    ax.set_xlabel("Session time (min)")
    ax.set_ylabel("Full-model held-out improvement (%)")
    ax.set_title(r"Adding $D_E\nabla^2 E$ provides negligible held-out gain")
    ax.legend(frameon=False)

    fig.suptitle(
        r"Resting-state session: reduced $D_E=0$ model selected in all 20 chunks",
        fontsize=14
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_both(fig, out_dir / "figure3_real_model_selection_v2", dpi)


def load_final_model(project_root: Path, final_run_dir: Path, chunk_idx: int, device):
    sys.path.insert(0, str(project_root))
    from train_gpinn_narval_final import FixedPhysicsGPINN

    ckpts = sorted((final_run_dir / "checkpoints").glob(f"seq_chunk_{chunk_idx:02d}_*.pt"))
    if len(ckpts) != 1:
        raise FileNotFoundError(f"Expected one final checkpoint for chunk {chunk_idx}; found {ckpts}")
    ck = torch.load(ckpts[0], map_location=device)
    fp = ck["fixed_parameters"]
    model = FixedPhysicsGPINN(
        model_kind=ck["model_kind"],
        w_EE=float(fp["w_EE"]), D_E=float(fp["D_E"]), h=float(fp["h"]),
        ff_dim=128, hidden_neurons=256, hidden_layers=6,
        scale_xy=25.0, scale_t=50.0,
    ).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model, ck, ckpts[0]


def frame_coords(ti: int, T: int, H: int, W: int):
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    x = -1.0 + 2.0 * xx.ravel() / float(W - 1)
    y = -1.0 + 2.0 * yy.ravel() / float(H - 1)
    t = np.full(H * W, -1.0 + 2.0 * ti / float(T - 1))
    return np.stack([x, y, t], axis=1).astype(np.float32)


def predict_points(model, X, device, batch=16384):
    out = []
    with torch.no_grad():
        for s in range(0, len(X), batch):
            xb = torch.tensor(X[s:s + batch], dtype=torch.float32, device=device)
            out.append(model.forward_E(xb).detach().cpu().numpy().ravel())
    return np.concatenate(out)


def compute_pde_residual_map(project_root, model, X, H, W, device):
    sys.path.insert(0, str(project_root))
    from train_gpinn_narval_final import gpinn_residual_with_grad
    parts = []
    for s in range(0, len(X), 2048):
        xb = torch.tensor(X[s:s + 2048], dtype=torch.float32, device=device)
        f, _, _ = gpinn_residual_with_grad(
            model, xb, tau=1.0, chunk_duration=30.0, l_fov=10.0
        )
        parts.append(f.detach().cpu().numpy().ravel())
    return np.concatenate(parts).reshape(H, W)


def figure4_reconstruction(project_root: Path, master_dir: Path,
                           final_run_dir: Path, data_dir: Path,
                           subject_prefix: str, chunk_idx: int,
                           out_dir: Path, dpi: int,
                           scatter_points: int, seed: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck, ckpt_path = load_final_model(
        project_root, final_run_dir, chunk_idx, device
    )
    data_path = data_dir / f"{subject_prefix}_chunk-{chunk_idx:02d}_dfof.npz"
    raw = np.load(data_path)["dfof"].astype(np.float64)
    gmin = float(ck["normalization"]["global_min"])
    gmax = float(ck["normalization"]["global_max"])
    E = (raw - gmin) / (gmax - gmin + 1e-8)
    T, H, W = E.shape

    frames = [0, T // 2, T - 1]
    pred_frames = {}
    for ti in frames:
        pred_frames[ti] = predict_points(
            model, frame_coords(ti, T, H, W), device
        ).reshape(H, W)

    # One common scale for every observed/predicted state map.
    state_min = min(
        min(float(E[ti].min()) for ti in frames),
        min(float(pred_frames[ti].min()) for ti in frames),
    )
    state_max = max(
        max(float(E[ti].max()) for ti in frames),
        max(float(pred_frames[ti].max()) for ti in frames),
    )
    errors = {ti: np.abs(E[ti] - pred_frames[ti]) for ti in frames}
    error_max = max(float(errors[ti].max()) for ti in frames)

    # Spatial mean time course.
    pred_mean = np.empty(T, dtype=float)
    for ti in range(T):
        pred_mean[ti] = predict_points(
            model, frame_coords(ti, T, H, W), device
        ).mean()
    obs_mean = E.mean(axis=(1, 2))
    seconds = np.arange(T) / 30.0

    # Dense pointwise comparison.
    rng = np.random.default_rng(seed)
    total = T * H * W
    n = min(scatter_points, total)
    flat = rng.choice(total, size=n, replace=False)
    ti = flat // (H * W)
    rem = flat % (H * W)
    yi = rem // W
    xi = rem % W
    Xs = np.column_stack([
        -1 + 2 * xi / (W - 1),
        -1 + 2 * yi / (H - 1),
        -1 + 2 * ti / (T - 1),
    ]).astype(np.float32)
    ys = E[ti, yi, xi]
    ps = predict_points(model, Xs, device)

    # Session-wide R² from the locked final results.
    final_master = pd.read_csv(master_dir / "final_continuity_chunkwise.csv")
    r2_chunks = final_master["random_val_state_r2"].astype(float)
    mins = (final_master["time_start_min"].astype(float)
            if "time_start_min" in final_master
            else final_master["chunk_idx"].astype(float) * 0.5)

    fig, axes = plt.subplots(4, 3, figsize=(12.2, 14.0))

    # A: observed
    for j, ti0 in enumerate(frames):
        ax = axes[0, j]
        im_state = ax.imshow(
            E[ti0], origin="lower", vmin=state_min, vmax=state_max
        )
        ax.set_title(f"Observed, t={ti0/30:.1f} s")
        ax.set_xticks([]); ax.set_yticks([])
        if j == 0:
            panel_label(ax, "A")
    cb = fig.colorbar(im_state, ax=axes[0, :].tolist(),
                      fraction=0.018, pad=0.02)
    cb.set_label("Normalized activity")

    # B: predicted, exact same scale
    for j, ti0 in enumerate(frames):
        ax = axes[1, j]
        im_state2 = ax.imshow(
            pred_frames[ti0], origin="lower", vmin=state_min, vmax=state_max
        )
        ax.set_title(f"Predicted, t={ti0/30:.1f} s")
        ax.set_xticks([]); ax.set_yticks([])
        if j == 0:
            panel_label(ax, "B")
    cb = fig.colorbar(im_state2, ax=axes[1, :].tolist(),
                      fraction=0.018, pad=0.02)
    cb.set_label("Normalized activity")

    # C: absolute error
    for j, ti0 in enumerate(frames):
        ax = axes[2, j]
        im_err = ax.imshow(
            errors[ti0], origin="lower", vmin=0, vmax=error_max
        )
        ax.set_title(f"|Observed − predicted|, t={ti0/30:.1f} s")
        ax.set_xticks([]); ax.set_yticks([])
        if j == 0:
            panel_label(ax, "C")
    cb = fig.colorbar(im_err, ax=axes[2, :].tolist(),
                      fraction=0.018, pad=0.02)
    cb.set_label("Absolute error")

    # D: mean time course
    ax = axes[3, 0]
    panel_label(ax, "D")
    ax.plot(seconds, obs_mean, label="Observed")
    ax.plot(seconds, pred_mean, label="Predicted")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Spatial mean normalized activity")
    ax.set_title("Spatially averaged activity")
    ax.legend(frameon=False)

    # E: pointwise density scatter
    ax = axes[3, 1]
    panel_label(ax, "E")
    hb = ax.hexbin(ys, ps, gridsize=60, mincnt=1)
    lim = [min(ys.min(), ps.min()), max(ys.max(), ps.max())]
    ax.plot(lim, lim, linestyle="--", linewidth=1)
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    ax.set_title("Pointwise reconstruction")
    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Count per hexagon")

    # F: session-wide interpolation fidelity
    ax = axes[3, 2]
    panel_label(ax, "F")
    ax.plot(mins, r2_chunks, marker="o")
    ax.axhline(r2_chunks.mean(), linestyle="--", linewidth=1,
               label=rf"mean={r2_chunks.mean():.3f}")
    ax.set_xlabel("Session time (min)")
    ax.set_ylabel(r"Random-pointwise validation $R^2$")
    ax.set_title("Session-wide interpolation fidelity")
    ax.legend(frameon=False)

    fig.suptitle(
        f"Final reduced-model gPINN reconstruction: representative chunk {chunk_idx}",
        fontsize=14
    )
    fig.subplots_adjust(top=0.95, hspace=0.38, wspace=0.28)
    save_both(fig, out_dir / "figure4_reconstruction_diagnostics_v2", dpi)

    # Supplement S1: signed PDE residual map.
    ti_mid = T // 2
    residual = compute_pde_residual_map(
        project_root, model, frame_coords(ti_mid, T, H, W), H, W, device
    )
    vmax = float(np.max(np.abs(residual)))
    fig_s, ax_s = plt.subplots(figsize=(5.6, 5.0))
    im = ax_s.imshow(residual, origin="lower", vmin=-vmax, vmax=vmax)
    ax_s.set_xticks([]); ax_s.set_yticks([])
    ax_s.set_title(f"Wilson–Cowan PDE residual, chunk {chunk_idx}, t={ti_mid/30:.1f} s")
    cb = fig_s.colorbar(im, ax=ax_s, fraction=0.046, pad=0.04)
    cb.set_label("PDE residual")
    save_both(fig_s, out_dir / "supp_figure_s1_pde_residual_v2", dpi)

    print(f"Figure 4 used checkpoint: {ckpt_path}")


def figure5_sequential(master_dir: Path, out_dir: Path, dpi: int):
    final_path = master_dir / "final_continuity_chunkwise.csv"
    noc_path = master_dir / "no_continuity_chunkwise.csv"
    ab_path = master_dir / "continuity_ablation_summary.csv"
    if not final_path.exists() or not noc_path.exists():
        print("WARNING: Figure 5 skipped because sequential master files are missing.")
        return

    final = pd.read_csv(final_path)
    noc = pd.read_csv(noc_path)
    ab = pd.read_csv(ab_path) if ab_path.exists() else pd.DataFrame()

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.8))

    ax = axes[0, 0]
    panel_label(ax, "A")
    d = noc[noc["chunk_idx"] > 0]
    ax.plot(d["chunk_idx"], d["pretrain_random_val_state_r2"], marker="o")
    ax.set_xlabel("Chunk")
    ax.set_ylabel(r"Pre-training validation $R^2$")
    ax.set_title("Inherited neural-state representation before retraining")

    ax = axes[0, 1]
    panel_label(ax, "B")
    if "sequential_minus_independent_r2" in noc.columns:
        m = noc[
            noc["sequential_minus_independent_r2"].notna() &
            (noc["chunk_idx"] > 0)
        ]
        ax.bar(m["chunk_idx"].astype(str), m["sequential_minus_independent_r2"])
        ax.axhline(0, linewidth=1)
        ax.set_xlabel("Matched chunk")
        ax.set_ylabel(r"$\Delta R^2$: sequential − independent")
        ax.set_title("Neural-state warm-start effect")
    else:
        ax.axis("off")

    ax = axes[1, 0]
    panel_label(ax, "C")
    if not ab.empty:
        ax.plot(ab["lambda_cont"], ab["mean_boundary_ratio"],
                marker="o", label="Mean boundary ratio")
        ax.axhline(1, linestyle="--", linewidth=1,
                   label="Observed transition scale")
        ax.axvline(0.05, linestyle=":", linewidth=1,
                   label=r"Selected $\lambda_{cont}=0.05$")
        sel = ab[np.isclose(ab["lambda_cont"].astype(float), 0.05)]
        if len(sel) == 1 and "mean_delta_r2_vs_lambda0" in sel.columns:
            dr2 = float(sel.iloc[0]["mean_delta_r2_vs_lambda0"])
            ax.text(
                0.05, float(sel.iloc[0]["mean_boundary_ratio"]),
                rf"  $\Delta R^2={dr2:+.5f}$",
                va="bottom", fontsize=8.5
            )
        ax.set_xlabel(r"Continuity weight $\lambda_{cont}$")
        ax.set_ylabel("Mean predicted / observed boundary RMSE")
        ax.set_title("Continuity-weight ablation")
        ax.legend(frameon=False, fontsize=8.5)
    else:
        ax.axis("off")

    ax = axes[1, 1]
    panel_label(ax, "D")
    merged = noc[["chunk_idx", "boundary_pred_to_observed_ratio"]].merge(
        final[["chunk_idx", "boundary_pred_to_observed_ratio"]],
        on="chunk_idx", suffixes=("_no", "_cont")
    )
    merged = merged[merged["chunk_idx"] > 0]
    ax.plot(merged["chunk_idx"],
            merged["boundary_pred_to_observed_ratio_no"],
            marker="o", label="No continuity")
    ax.plot(merged["chunk_idx"],
            merged["boundary_pred_to_observed_ratio_cont"],
            marker="s", label=r"$\lambda_{cont}=0.05$")
    ax.axhline(1, linestyle="--", linewidth=1,
               label="Observed transition scale")
    ax.set_xlabel(r"Boundary at start of chunk $i$")
    ax.set_ylabel("Predicted / observed boundary RMSE")
    ax.set_title("Boundary consistency across all 19 transitions")
    ax.legend(frameon=False, fontsize=8.5)

    fig.suptitle(
        "Sequential neural-state transfer and continuity regularization",
        fontsize=14
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_both(fig, out_dir / "figure5_sequential_continuity_v2", dpi)


def main():
    args = parse_args()
    root = Path(args.project_root).resolve()
    master = root / args.master_dir
    final_run = root / args.final_run_dir
    data_dir = root / args.data_dir
    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    figure1_pipeline(out_dir, args.dpi)
    figure2_synthetic(master, out_dir, args.dpi)
    figure3_real_model(master, out_dir, args.dpi)
    figure4_reconstruction(
        root, master, final_run, data_dir,
        args.subject_prefix, args.representative_chunk,
        out_dir, args.dpi, args.scatter_points, args.seed,
    )
    figure5_sequential(master, out_dir, args.dpi)
    print(f"Publication V2 figures saved to {out_dir}")


if __name__ == "__main__":
    main()
