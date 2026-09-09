#!/usr/bin/env python3
"""Synthetic positive/null control for the final Wilson-Cowan model selector.

Applies the *unchanged* regression-assisted model-selection logic from
train_gpinn_narval_final.py to two independently generated synthetic trajectories:
  1) positive spatial coupling: D_E*=0.60
  2) null spatial coupling:     D_E*=0.00

No neural-network training is performed. The purpose is to test whether the final
selector can distinguish a supported Laplacian term from an absent one under the
same decision criteria used for the real-data analysis.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from train_gpinn_narval_final import (
    PRIMARY_SIGMA_XY_MM,
    compute_pseudo_fields,
    select_model,
    smoothing_D_estimates,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--positive", default="synthetic_data/wc_twin_truth.npz")
    p.add_argument("--null", default="synthetic_data/wc_twin_truth_D0.npz")
    p.add_argument("--out_dir", default="outputs/synthetic_model_selection_control")
    p.add_argument("--noise_sigmas", nargs="+", type=float, default=[0.0, 0.01])
    p.add_argument("--noise_seed", type=int, default=42)

    # Exact final selector defaults.
    p.add_argument("--nested_folds", type=int, default=5)
    p.add_argument("--bootstrap_reps", type=int, default=1000)
    p.add_argument("--bootstrap_block_frames", type=int, default=30)
    p.add_argument("--eps_logit", type=float, default=1e-4)
    p.add_argument("--min_valid_fraction", type=float, default=0.90)
    p.add_argument("--max_standardized_condition", type=float, default=10.0)
    p.add_argument("--min_logit_improvement_pct", type=float, default=0.10)
    p.add_argument("--min_pde_improvement_pct", type=float, default=0.10)
    p.add_argument("--min_positive_fold_fraction", type=float, default=0.60)
    p.add_argument("--max_smoothing_relative_range", type=float, default=1.00)
    p.add_argument("--w_min", type=float, default=0.5)
    p.add_argument("--w_max", type=float, default=6.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_case(path: Path):
    d = np.load(path, allow_pickle=False)
    return {
        "E": d["E"].astype(np.float64),
        "D_E_true": float(d["D_E_true"]),
        "w_EE_true": float(d["w_EE_true"]),
        "h_true": float(d["h_true"]),
        "tau": float(d["tau"]),
        "l_fov": float(d["l_fov"]),
        "fps": 1.0 / float(np.median(np.diff(d["t_seconds"].astype(np.float64)))),
    }


def run_one(label: str, case: dict, noise_sigma: float, args):
    E = case["E"].copy()
    if noise_sigma > 0:
        rng = np.random.default_rng(args.noise_seed)
        E = np.clip(E + rng.normal(0.0, noise_sigma, size=E.shape), 0.0, 1.0)

    _, _, W = E.shape
    dx = case["l_fov"] / (W - 1)
    sigma_xy_px = PRIMARY_SIGMA_XY_MM / dx

    E_sm = gaussian_filter(
        E,
        sigma=(2.0, sigma_xy_px, sigma_xy_px),
        mode="reflect",
    )
    E0, Et, Lap, q, ylogit, valid = compute_pseudo_fields(
        E_sm,
        case["fps"],
        case["l_fov"],
        case["tau"],
        args.eps_logit,
    )

    sargs = SimpleNamespace(
        tau=case["tau"],
        nested_folds=args.nested_folds,
        bootstrap_block_frames=args.bootstrap_block_frames,
        bootstrap_reps=args.bootstrap_reps,
        seed=args.seed,
        w_min=args.w_min,
        w_max=args.w_max,
    )
    select = select_model(E0, Et, Lap, ylogit, valid, sargs, chunk_idx=0)
    smooth_results, smooth_rel_range, smooth_all_positive = smoothing_D_estimates(
        E,
        case["fps"],
        case["l_fov"],
        case["tau"],
        args.eps_logit,
    )

    b_full = select["full_beta"]
    b_red = select["reduced_beta"]
    criteria = {
        "valid_fraction": select["valid_fraction"] >= args.min_valid_fraction,
        "conditioning": select["condition_standardized"] <= args.max_standardized_condition,
        "D_point_positive": bool(float(b_full[1]) > 0.0),
        "D_CI95_positive": bool(select["ci_excludes_zero_positive"]),
        "logit_mean_improvement": (
            select["mean_logit_improvement_pct"] >= args.min_logit_improvement_pct
        ),
        "pde_mean_improvement": (
            select["mean_pde_improvement_pct"] >= args.min_pde_improvement_pct
        ),
        "logit_fold_consistency": (
            select["fraction_logit_folds_positive"] >= args.min_positive_fold_fraction
        ),
        "pde_fold_consistency": (
            select["fraction_pde_folds_positive"] >= args.min_positive_fold_fraction
        ),
        "smoothing_all_positive": bool(smooth_all_positive),
        "smoothing_relative_range": (
            smooth_rel_range <= args.max_smoothing_relative_range
        ),
        "w_in_bounds": bool(select["w_in_bounds"]),
    }

    use_full = bool(all(criteria.values()))
    selected_model = "full" if use_full else "reduced"
    expected_model = "full" if case["D_E_true"] > 0 else "reduced"

    return {
        "case": label,
        "noise_sigma": float(noise_sigma),
        "noise_seed": int(args.noise_seed),
        "truth_D_E": case["D_E_true"],
        "truth_w_EE": case["w_EE_true"],
        "truth_h": case["h_true"],
        "expected_model": expected_model,
        "selected_model": selected_model,
        "selection_correct": bool(selected_model == expected_model),
        "full_w_EE": float(b_full[0]),
        "full_D_E": float(b_full[1]),
        "full_h": float(b_full[2]),
        "reduced_w_EE": float(b_red[0]),
        "reduced_h": float(b_red[1]),
        "D_ci95_low": float(select["bootstrap"]["ci95_low"]),
        "D_ci95_high": float(select["bootstrap"]["ci95_high"]),
        "valid_fraction": float(select["valid_fraction"]),
        "condition_standardized": float(select["condition_standardized"]),
        "mean_logit_improvement_pct": float(select["mean_logit_improvement_pct"]),
        "mean_pde_improvement_pct": float(select["mean_pde_improvement_pct"]),
        "fraction_logit_folds_positive": float(select["fraction_logit_folds_positive"]),
        "fraction_pde_folds_positive": float(select["fraction_pde_folds_positive"]),
        "smoothing_D_half": float(smooth_results["half"]["D"]),
        "smoothing_D_primary": float(smooth_results["primary"]["D"]),
        "smoothing_D_double": float(smooth_results["double"]["D"]),
        "smoothing_relative_range": float(smooth_rel_range),
        **{f"criterion_{k}": bool(v) for k, v in criteria.items()},
    }


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cases = {
        "positive_D0p60": load_case(Path(args.positive)),
        "null_D0": load_case(Path(args.null)),
    }

    # Guard against accidentally auditing the wrong files.
    if not np.isclose(cases["positive_D0p60"]["D_E_true"], 0.60):
        raise ValueError("Positive-control file does not have D_E_true=0.60")
    if not np.isclose(cases["null_D0"]["D_E_true"], 0.0):
        raise ValueError("Null-control file does not have D_E_true=0.0")

    rows = []
    for label, case in cases.items():
        for noise in args.noise_sigmas:
            print(f"\n=== {label} | noise={noise:g} ===", flush=True)
            row = run_one(label, case, float(noise), args)
            rows.append(row)
            print(json.dumps(row, indent=2), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "synthetic_model_selection_control.csv", index=False)

    summary = {
        "purpose": (
            "Positive/null synthetic control for the unchanged final model selector. "
            "A discriminating selector should choose full for D_E*=0.60 and reduced for D_E*=0."
        ),
        "thresholds": {
            "min_valid_fraction": args.min_valid_fraction,
            "max_standardized_condition": args.max_standardized_condition,
            "min_logit_improvement_pct": args.min_logit_improvement_pct,
            "min_pde_improvement_pct": args.min_pde_improvement_pct,
            "min_positive_fold_fraction": args.min_positive_fold_fraction,
            "max_smoothing_relative_range": args.max_smoothing_relative_range,
            "w_bounds": [args.w_min, args.w_max],
            "bootstrap_reps": args.bootstrap_reps,
            "bootstrap_block_frames": args.bootstrap_block_frames,
        },
        "n_runs": int(len(df)),
        "n_correct": int(df["selection_correct"].sum()),
        "all_correct": bool(df["selection_correct"].all()),
        "runs": rows,
    }
    (out / "synthetic_model_selection_control.json").write_text(
        json.dumps(summary, indent=2)
    )

    print("\n=== COMPACT SUMMARY ===")
    print(df[[
        "case", "noise_sigma", "truth_D_E", "expected_model", "selected_model",
        "selection_correct", "full_D_E", "D_ci95_low", "D_ci95_high",
        "mean_logit_improvement_pct", "mean_pde_improvement_pct",
    ]].to_string(index=False))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
