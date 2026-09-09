#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input_dir",
        default="./outputs/continuity_ablation/runs",
    )
    p.add_argument(
        "--output_dir",
        default="./outputs/continuity_ablation",
    )
    return p.parse_args()


def write_csv(path, rows):
    if not rows:
        return
    fields = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    args = parse_args()
    inp = Path(args.input_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(p.read_text()) for p in sorted(inp.glob("*.json"))]
    if len(rows) != 20:
        raise RuntimeError(f"Expected 20 ablation result files, found {len(rows)}")

    rows = sorted(rows, key=lambda r: (r["lambda_cont"], r["chunk_idx"]))
    write_csv(out / "continuity_ablation_runs.csv", rows)

    lambdas = sorted({float(r["lambda_cont"]) for r in rows})
    summary = []

    baseline = {
        int(r["chunk_idx"]): r
        for r in rows
        if float(r["lambda_cont"]) == 0.0
    }

    for lam in lambdas:
        rr = [r for r in rows if float(r["lambda_cont"]) == lam]

        ratios = np.array(
            [r["boundary_pred_to_observed_ratio"] for r in rr],
            dtype=float,
        )
        r2 = np.array([r["random_val_state_r2"] for r in rr], dtype=float)
        pde = np.array(
            [r["random_val_neural_pde_rmse"] for r in rr], dtype=float
        )
        mse = np.array([r["random_val_state_mse"] for r in rr], dtype=float)

        delta_r2 = []
        delta_pde = []
        ratio_reduction = []
        for r in rr:
            b = baseline[int(r["chunk_idx"])]
            delta_r2.append(
                float(r["random_val_state_r2"] - b["random_val_state_r2"])
            )
            delta_pde.append(
                float(
                    r["random_val_neural_pde_rmse"]
                    - b["random_val_neural_pde_rmse"]
                )
            )
            ratio_reduction.append(
                float(
                    b["boundary_pred_to_observed_ratio"]
                    - r["boundary_pred_to_observed_ratio"]
                )
            )

        summary.append(
            {
                "lambda_cont": lam,
                "n_boundaries": len(rr),
                "mean_boundary_ratio": float(ratios.mean()),
                "median_boundary_ratio": float(np.median(ratios)),
                "max_boundary_ratio": float(ratios.max()),
                "fraction_boundary_ratio_le_2": float(np.mean(ratios <= 2.0)),
                "mean_random_val_r2": float(r2.mean()),
                "mean_random_val_mse": float(mse.mean()),
                "mean_neural_pde_rmse": float(pde.mean()),
                "mean_delta_r2_vs_lambda0": float(np.mean(delta_r2)),
                "mean_delta_pde_rmse_vs_lambda0": float(np.mean(delta_pde)),
                "mean_boundary_ratio_reduction_vs_lambda0": float(
                    np.mean(ratio_reduction)
                ),
            }
        )

    write_csv(out / "continuity_ablation_summary.csv", summary)

    # A simple transparent recommendation rule:
    # among nonzero lambdas, choose the one with largest boundary-ratio
    # reduction subject to mean R2 loss <= 0.002 and PDE-RMSE increase <= 0.005.
    eligible = [
        s for s in summary
        if s["lambda_cont"] > 0
        and s["mean_delta_r2_vs_lambda0"] >= -0.002
        and s["mean_delta_pde_rmse_vs_lambda0"] <= 0.005
    ]

    recommendation = None
    if eligible:
        recommendation = max(
            eligible,
            key=lambda s: s["mean_boundary_ratio_reduction_vs_lambda0"],
        )

    final = {
        "summary_by_lambda": summary,
        "selection_rule": {
            "max_mean_R2_loss": 0.002,
            "max_mean_PDE_RMSE_increase": 0.005,
            "objective": "largest mean boundary-ratio reduction",
        },
        "recommended_nonzero_lambda": (
            recommendation["lambda_cont"] if recommendation else None
        ),
        "recommended_summary": recommendation,
        "note": (
            "Recommendation is an ablation decision rule, not a physiological "
            "claim. Inspect per-boundary behavior before adopting the loss."
        ),
    }

    (out / "continuity_ablation_summary.json").write_text(
        json.dumps(final, indent=2)
    )

    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
