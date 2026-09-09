#!/usr/bin/env python3
"""Generate independent synthetic Wilson-Cowan data with finite differences + RK4.

This forward solver is intentionally independent of the PINN. It solves
    tau * E_t = -E + sigmoid(w_EE E + D_E Laplacian(E) + h)
on a 10 mm x 10 mm square using a 5-point finite-difference Laplacian,
zero-normal-flux (Neumann) boundaries, and RK4 time stepping.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def laplacian_neumann(E: np.ndarray, dx: float) -> np.ndarray:
    # Edge padding implements zero normal derivative to first order.
    p = np.pad(E, 1, mode="edge")
    return (
        p[1:-1, 2:] + p[1:-1, :-2] + p[2:, 1:-1] + p[:-2, 1:-1] - 4.0 * E
    ) / (dx * dx)


def rhs(E: np.ndarray, D_E: float, w_EE: float, h: float, tau: float, dx: float) -> np.ndarray:
    lap = laplacian_neumann(E, dx)
    return (-E + sigmoid(w_EE * E + D_E * lap + h)) / tau


def make_initial_condition(nx: int, ny: int, l_fov: float) -> np.ndarray:
    x = np.linspace(-l_fov / 2.0, l_fov / 2.0, nx, dtype=np.float64)
    y = np.linspace(-l_fov / 2.0, l_fov / 2.0, ny, dtype=np.float64)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    r = np.sqrt(xx**2 + yy**2)

    # Smooth high-activity domain embedded in a low-activity background.
    # It creates nonzero Laplacian information without injecting a time-dependent forcing.
    E0 = 0.05 + 0.90 / (1.0 + np.exp((r - 1.8) / 0.18))
    # Mild asymmetric perturbation prevents a perfectly radial inverse problem.
    E0 += 0.035 * np.exp(-((xx + 2.5) ** 2 + (yy - 2.0) ** 2) / (2.0 * 0.55**2))
    return np.clip(E0, 0.01, 0.99)


def simulate(
    D_E: float,
    w_EE: float,
    h: float,
    tau: float,
    l_fov: float,
    nx: int,
    ny: int,
    duration: float,
    output_hz: float,
    substeps_per_frame: int,
) -> tuple[np.ndarray, np.ndarray]:
    if nx != ny:
        raise ValueError("This generator currently assumes nx == ny so dx == dy.")
    dx = l_fov / (nx - 1)
    dt_frame = 1.0 / output_hz
    dt = dt_frame / substeps_per_frame
    n_frames = int(round(duration * output_hz))

    E = make_initial_condition(nx, ny, l_fov)
    frames = np.empty((n_frames, ny, nx), dtype=np.float32)

    for frame in range(n_frames):
        for _ in range(substeps_per_frame):
            k1 = rhs(E, D_E, w_EE, h, tau, dx)
            k2 = rhs(E + 0.5 * dt * k1, D_E, w_EE, h, tau, dx)
            k3 = rhs(E + 0.5 * dt * k2, D_E, w_EE, h, tau, dx)
            k4 = rhs(E + dt * k3, D_E, w_EE, h, tau, dx)
            E = E + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            E = np.clip(E, 0.0, 1.0)
        frames[frame] = E.astype(np.float32)

    t = np.arange(n_frames, dtype=np.float64) / output_hz
    return frames, t


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=str, default="synthetic_data/wc_twin_truth.npz")
    p.add_argument("--D_E", type=float, default=0.60)
    p.add_argument("--w_EE", type=float, default=5.20)
    p.add_argument("--h", type=float, default=-2.60)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--l_fov", type=float, default=10.0)
    p.add_argument("--nx", type=int, default=128)
    p.add_argument("--ny", type=int, default=128)
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument("--output_hz", type=float, default=30.0)
    p.add_argument("--substeps_per_frame", type=int, default=4)
    args = p.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    E, t = simulate(
        args.D_E, args.w_EE, args.h, args.tau, args.l_fov,
        args.nx, args.ny, args.duration, args.output_hz, args.substeps_per_frame,
    )

    metadata = {
        "forward_solver": "5-point finite-difference Laplacian + explicit RK4",
        "boundary_condition": "homogeneous Neumann (zero normal flux)",
        "D_E_true": float(args.D_E),
        "w_EE_true": float(args.w_EE),
        "h_true": float(args.h),
        "tau": float(args.tau),
        "l_fov_mm": float(args.l_fov),
        "nx": int(args.nx),
        "ny": int(args.ny),
        "duration_s": float(args.duration),
        "output_hz": float(args.output_hz),
        "substeps_per_frame": int(args.substeps_per_frame),
        "integration_dt_s": float(1.0 / args.output_hz / args.substeps_per_frame),
        "normalization": "none; E is generated directly on the model scale [0,1]",
    }

    np.savez_compressed(
        out,
        E=E,
        t_seconds=t,
        D_E_true=np.float64(args.D_E),
        w_EE_true=np.float64(args.w_EE),
        h_true=np.float64(args.h),
        tau=np.float64(args.tau),
        l_fov=np.float64(args.l_fov),
        metadata_json=np.array(json.dumps(metadata)),
    )
    print(json.dumps(metadata, indent=2))
    print(f"Saved: {out}")
    print(f"E shape={E.shape}; min={E.min():.6f}, max={E.max():.6f}, std={E.std():.6f}")


if __name__ == "__main__":
    main()
