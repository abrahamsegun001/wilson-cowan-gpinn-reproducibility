#!/usr/bin/env python3
"""
Final publication figures for the regression-assisted, model-selective
sequential Wilson-Cowan gPINN study.

This script regenerates ONLY the two figures that still required layout work:
Figure 1 (pipeline schematic) and Figure 4 (reconstruction diagnostics), plus
Supplementary Figure S1 (PDE residual). It performs no training and does not
modify any locked numerical results.
"""
from __future__ import annotations

import argparse
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
    p.add_argument("--master_dir", default="outputs/publication_final/master")
    p.add_argument("--final_run_dir",
                   default="outputs/final_dual_model_sequential20_cont005")
    p.add_argument("--data_dir", default="data/processed")
    p.add_argument("--subject_prefix", default="sub-GC48_ses-day0")
    p.add_argument("--representative_chunk", type=int, default=12)
    p.add_argument("--output_dir", default="outputs/publication_final/figures")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--scatter_points", type=int, default=30000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def save_both(fig, pathstem: Path, dpi: int):
    pathstem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pathstem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(pathstem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def panel_label(ax, label, x=-0.10, y=1.04):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="bottom", ha="left")


def add_box(ax, x, y, w, h, text, fontsize=10):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        fill=False, linewidth=1.35,
    )
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fontsize, linespacing=1.15)
    return box


def arrow(ax, x0, y0, x1, y1, connectionstyle=None):
    props = dict(arrowstyle="->", lw=1.35, shrinkA=2, shrinkB=2)
    if connectionstyle is not None:
        props["connectionstyle"] = connectionstyle
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=props)


def figure1_pipeline(out_dir: Path, dpi: int):
    """
    Compact two-lane schematic:
      A. independent per-chunk parameter inference/model selection
      B. neural-state training + sequential transfer
    """
    fig, ax = plt.subplots(figsize=(13.2, 6.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.suptitle(
        "Regression-assisted, model-selective sequential Wilson–Cowan gPINN",
        fontsize=16, y=0.97
    )

    # Lane headings.
    ax.text(0.035, 0.845, "A", fontsize=12, fontweight="bold", va="center")
    ax.text(0.060, 0.845, "Per-chunk parameter inference and model selection",
            fontsize=12, fontweight="bold", va="center")

    ax.text(0.035, 0.415, "B", fontsize=12, fontweight="bold", va="center")
    ax.text(0.060, 0.415, "Neural-state estimation and sequential transfer",
            fontsize=12, fontweight="bold", va="center")

    # Top lane: independent physical inference.
    xs = [0.055, 0.285, 0.515, 0.745]
    w = 0.19
    h = 0.17
    y_top = 0.61

    add_box(ax, xs[0], y_top, w, h,
            "30-s calcium chunk\n$E_i(x,y,t)$\n64×64, 30 Hz")
    add_box(ax, xs[1], y_top, w, h,
            "Resolution-aware smoothing\n+ finite-difference\npseudo-derivatives")
    add_box(ax, xs[2], y_top, w, h,
            "Full vs reduced\nlogit regression\n+ uncertainty / conditioning")
    add_box(ax, xs[3], y_top, w, h,
            "Evidence-based selection\nand current-chunk\n$(\\hat w_{EE},\\hat D_E,\\hat h)$")

    for j in range(3):
        arrow(ax, xs[j] + w, y_top + h/2,
              xs[j+1], y_top + h/2)

    # Bottom lane: state representation.
    y_bot = 0.17
    xb = [0.055, 0.305, 0.555, 0.795]
    wb = [0.19, 0.19, 0.17, 0.15]
    hb = 0.16

    add_box(ax, xb[0], y_bot, wb[0], hb,
            "Stage A\nstate-only fit\n(current chunk)")
    add_box(ax, xb[1], y_bot, wb[1], hb,
            "Stage B\nfixed-physics +\npseudo-derivative refinement")
    add_box(ax, xb[2], y_bot, wb[2], hb,
            "Final neural-state\nrepresentation\n$E_{\\theta_i}$")
    add_box(ax, xb[3], y_bot, wb[3], hb,
            "Next chunk\ninitialize with\n$\\theta_i$ only")

    arrow(ax, xb[0] + wb[0], y_bot + hb/2,
          xb[1], y_bot + hb/2)
    arrow(ax, xb[1] + wb[1], y_bot + hb/2,
          xb[2], y_bot + hb/2)
    arrow(ax, xb[2] + wb[2], y_bot + hb/2,
          xb[3], y_bot + hb/2)

    # Data directly supplies Stage A.
    arrow(ax, xs[0] + w/2, y_top,
          xb[0] + wb[0]/2, y_bot + hb)

    # Selected physical coefficients supply Stage B.
    # Route on the far-right side, below the B-lane heading, so the connector
    # never crosses text.
    sel_x = xs[3] + w/2
    stageb_right = xb[1] + wb[1]
    route_y = 0.365

    ax.annotate(
        "", xy=(sel_x, route_y), xytext=(sel_x, y_top),
        arrowprops=dict(arrowstyle="-", lw=1.35)
    )
    ax.annotate(
        "", xy=(stageb_right, route_y), xytext=(sel_x, route_y),
        arrowprops=dict(arrowstyle="-", lw=1.35)
    )
    arrow(ax, stageb_right, route_y,
          stageb_right, y_bot + hb)

    # Continuity constraint belongs to the transition between adjacent chunks.
    ax.text(
        0.715, 0.105,
        r"chunk boundary:  $\mathcal{L}_{cont}$,  $\lambda_{cont}=0.05$",
        fontsize=9.5, ha="center", va="center"
    )

    # Explicitly state what does and does not propagate; this is central to the method.
    ax.text(
        0.50, 0.035,
        "Physical coefficients are re-estimated independently for each chunk; "
        "optimizer state is reset. Only neural weights are warm-started.",
        fontsize=9.5, ha="center", va="center"
    )

    save_both(fig, out_dir / "figure1_final_pipeline", dpi)


def load_final_model(project_root: Path, final_run_dir: Path,
                     chunk_idx: int, device):
    sys.path.insert(0, str(project_root))
    from train_gpinn_narval_final import FixedPhysicsGPINN

    ckpts = sorted(
        (final_run_dir / "checkpoints").glob(
            f"seq_chunk_{chunk_idx:02d}_*.pt"
        )
    )
    if len(ckpts) != 1:
        raise FileNotFoundError(
            f"Expected one checkpoint for chunk {chunk_idx}; found {ckpts}"
        )

    ck = torch.load(ckpts[0], map_location=device)
    fp = ck["fixed_parameters"]
    model = FixedPhysicsGPINN(
        model_kind=ck["model_kind"],
        w_EE=float(fp["w_EE"]),
        D_E=float(fp["D_E"]),
        h=float(fp["h"]),
        ff_dim=128,
        hidden_neurons=256,
        hidden_layers=6,
        scale_xy=25.0,
        scale_t=50.0,
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
            xb = torch.tensor(
                X[s:s+batch], dtype=torch.float32, device=device
            )
            out.append(
                model.forward_E(xb).detach().cpu().numpy().ravel()
            )
    return np.concatenate(out)


def compute_pde_residual_map(project_root: Path, model, X,
                             H: int, W: int, device):
    sys.path.insert(0, str(project_root))
    from train_gpinn_narval_final import gpinn_residual_with_grad

    parts = []
    for s in range(0, len(X), 2048):
        xb = torch.tensor(
            X[s:s+2048], dtype=torch.float32, device=device
        )
        f, _, _ = gpinn_residual_with_grad(
            model, xb,
            tau=1.0,
            chunk_duration=30.0,
            l_fov=10.0,
        )
        parts.append(f.detach().cpu().numpy().ravel())
    return np.concatenate(parts).reshape(H, W)


def figure4_reconstruction(project_root: Path,
                           master_dir: Path,
                           final_run_dir: Path,
                           data_dir: Path,
                           subject_prefix: str,
                           chunk_idx: int,
                           out_dir: Path,
                           dpi: int,
                           scatter_points: int,
                           seed: int):
    """
    Publication layout with dedicated colorbar columns and no panel overlap.

    Observed and predicted state maps share EXACTLY the same normalization.
    Absolute-error maps use a separate shared scale.
    """
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model, ck, ckpt_path = load_final_model(
        project_root, final_run_dir, chunk_idx, device
    )

    data_path = (
        data_dir /
        f"{subject_prefix}_chunk-{chunk_idx:02d}_dfof.npz"
    )
    raw = np.load(data_path)["dfof"].astype(np.float64)

    gmin = float(ck["normalization"]["global_min"])
    gmax = float(ck["normalization"]["global_max"])
    E = (raw - gmin) / (gmax - gmin + 1e-8)
    T, H, W = E.shape

    frames = [0, T // 2, T - 1]
    pred_frames = {
        ti: predict_points(
            model, frame_coords(ti, T, H, W), device
        ).reshape(H, W)
        for ti in frames
    }

    # Shared state scale across ALL six observed/predicted maps.
    state_min = min(
        min(float(E[ti].min()) for ti in frames),
        min(float(pred_frames[ti].min()) for ti in frames),
    )
    state_max = max(
        max(float(E[ti].max()) for ti in frames),
        max(float(pred_frames[ti].max()) for ti in frames),
    )

    abs_errors = {
        ti: np.abs(E[ti] - pred_frames[ti])
        for ti in frames
    }
    error_max = max(
        float(abs_errors[ti].max()) for ti in frames
    )

    # Spatial mean trace.
    pred_mean = np.empty(T, dtype=float)
    for ti in range(T):
        pred_mean[ti] = predict_points(
            model, frame_coords(ti, T, H, W), device
        ).mean()
    obs_mean = E.mean(axis=(1, 2))
    seconds = np.arange(T) / 30.0

    # Pointwise sample for a clean scatter diagnostic.
    rng = np.random.default_rng(seed)
    total = T * H * W
    n = min(scatter_points, total)
    flat = rng.choice(total, size=n, replace=False)
    ti_s = flat // (H * W)
    rem = flat % (H * W)
    yi_s = rem // W
    xi_s = rem % W
    Xs = np.column_stack([
        -1 + 2 * xi_s / (W - 1),
        -1 + 2 * yi_s / (H - 1),
        -1 + 2 * ti_s / (T - 1),
    ]).astype(np.float32)
    y_obs = E[ti_s, yi_s, xi_s]
    y_pred = predict_points(model, Xs, device)

    # Locked session-wide interpolation fidelity.
    session = pd.read_csv(
        master_dir / "final_continuity_chunkwise.csv"
    )
    r2 = session["random_val_state_r2"].astype(float).to_numpy()
    if "time_start_min" in session.columns:
        mins = session["time_start_min"].astype(float).to_numpy()
    else:
        mins = (
            session["chunk_idx"].astype(float).to_numpy() * 0.5
        )

    fig = plt.figure(figsize=(12.8, 12.4), constrained_layout=True)
    gs = fig.add_gridspec(
        4, 4,
        width_ratios=[1, 1, 1, 0.055],
        height_ratios=[1, 1, 1, 0.95],
    )

    obs_axes = [fig.add_subplot(gs[0, j]) for j in range(3)]
    pred_axes = [fig.add_subplot(gs[1, j]) for j in range(3)]
    err_axes = [fig.add_subplot(gs[2, j]) for j in range(3)]

    cax_state = fig.add_subplot(gs[0:2, 3])
    cax_error = fig.add_subplot(gs[2, 3])

    ax_trace = fig.add_subplot(gs[3, 0])
    ax_scatter = fig.add_subplot(gs[3, 1])
    ax_r2 = fig.add_subplot(gs[3, 2])
    ax_blank = fig.add_subplot(gs[3, 3])
    ax_blank.axis("off")

    # A. Observed snapshots.
    im_state = None
    for j, ti0 in enumerate(frames):
        ax = obs_axes[j]
        im_state = ax.imshow(
            E[ti0],
            origin="lower",
            vmin=state_min,
            vmax=state_max,
            interpolation="nearest",
        )
        ax.set_title(f"t = {ti0/30:.1f} s", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    panel_label(obs_axes[0], "A")
    obs_axes[0].set_ylabel("Observed", fontsize=11, labelpad=10)

    # B. Predicted snapshots, SAME state scale.
    for j, ti0 in enumerate(frames):
        ax = pred_axes[j]
        ax.imshow(
            pred_frames[ti0],
            origin="lower",
            vmin=state_min,
            vmax=state_max,
            interpolation="nearest",
        )
        ax.set_xticks([])
        ax.set_yticks([])
    panel_label(pred_axes[0], "B")
    pred_axes[0].set_ylabel("Predicted", fontsize=11, labelpad=10)

    cb = fig.colorbar(im_state, cax=cax_state)
    cb.set_label("Normalized activity")

    # C. Absolute error.
    im_err = None
    for j, ti0 in enumerate(frames):
        ax = err_axes[j]
        im_err = ax.imshow(
            abs_errors[ti0],
            origin="lower",
            vmin=0,
            vmax=error_max,
            interpolation="nearest",
        )
        ax.set_xticks([])
        ax.set_yticks([])
    panel_label(err_axes[0], "C")
    err_axes[0].set_ylabel("Absolute error", fontsize=11, labelpad=10)

    cb2 = fig.colorbar(im_err, cax=cax_error)
    cb2.set_label(r"$|E_{\mathrm{obs}}-E_{\mathrm{pred}}|$")

    # D. Mean time course.
    panel_label(ax_trace, "D")
    ax_trace.plot(seconds, obs_mean, label="Observed")
    ax_trace.plot(seconds, pred_mean, label="Predicted")
    ax_trace.set_xlabel("Time (s)")
    ax_trace.set_ylabel("Spatial mean activity")
    ax_trace.set_title("Spatially averaged activity", fontsize=10)
    ax_trace.legend(frameon=False, fontsize=8)

    # E. Pointwise reconstruction.
    panel_label(ax_scatter, "E")
    ax_scatter.scatter(
        y_obs, y_pred,
        s=3, alpha=0.12,
        rasterized=True,
    )
    lo = float(min(y_obs.min(), y_pred.min()))
    hi = float(max(y_obs.max(), y_pred.max()))
    ax_scatter.plot([lo, hi], [lo, hi],
                    linestyle="--", linewidth=1)
    ax_scatter.set_xlim(lo, hi)
    ax_scatter.set_ylim(lo, hi)
    ax_scatter.set_xlabel("Observed")
    ax_scatter.set_ylabel("Predicted")
    ax_scatter.set_title("Pointwise reconstruction", fontsize=10)

    # F. Session-wide interpolation fidelity.
    panel_label(ax_r2, "F")
    ax_r2.plot(mins, r2, marker="o")
    ax_r2.axhline(
        float(np.mean(r2)),
        linestyle="--",
        linewidth=1,
        label=rf"mean = {np.mean(r2):.3f}",
    )
    ax_r2.set_xlabel("Session time (min)")
    ax_r2.set_ylabel(r"Validation $R^2$")
    ax_r2.set_title("Session-wide interpolation fidelity", fontsize=10)
    ax_r2.legend(frameon=False, fontsize=8)

    fig.suptitle(
        f"Final reduced-model gPINN reconstruction: representative chunk {chunk_idx}",
        fontsize=14
    )

    save_both(
        fig,
        out_dir / "figure4_reconstruction_diagnostics",
        dpi
    )

    # Supplementary Figure S1: representative signed PDE residual.
    ti_mid = T // 2
    residual = compute_pde_residual_map(
        project_root,
        model,
        frame_coords(ti_mid, T, H, W),
        H, W, device,
    )
    vmax = float(np.max(np.abs(residual)))

    fig_s, ax_s = plt.subplots(figsize=(5.7, 5.2))
    im = ax_s.imshow(
        residual,
        origin="lower",
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )
    ax_s.set_xticks([])
    ax_s.set_yticks([])
    ax_s.set_title(
        f"Representative PDE residual: chunk {chunk_idx}, t={ti_mid/30:.1f} s",
        fontsize=11
    )
    cb_s = fig_s.colorbar(im, ax=ax_s, fraction=0.046, pad=0.04)
    cb_s.set_label("Signed PDE residual")
    save_both(
        fig_s,
        out_dir / "supp_figure_s1_pde_residual",
        dpi
    )

    print(f"Figure 4 used checkpoint: {ckpt_path}")


def main():
    args = parse_args()

    root = Path(args.project_root).resolve()
    master = root / args.master_dir
    final_run = root / args.final_run_dir
    data_dir = root / args.data_dir
    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    figure1_pipeline(out_dir, args.dpi)
    figure4_reconstruction(
        root,
        master,
        final_run,
        data_dir,
        args.subject_prefix,
        args.representative_chunk,
        out_dir,
        args.dpi,
        args.scatter_points,
        args.seed,
    )

    print(f"Final Figure 1, Figure 4, and Supplement S1 saved to {out_dir}")


if __name__ == "__main__":
    main()
