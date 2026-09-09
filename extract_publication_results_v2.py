#!/usr/bin/env python3
"""
Extract the final Wilson-Cowan/gPINN manuscript results for publication V2
from the completed Narval experiments into one reproducible publication-data directory.

This script does NOT retrain anything. It only reads completed outputs.
It produces the manuscript tables plus master CSV/JSON files used by the
publication plotting script.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project_root", default=".")
    p.add_argument("--final_dir", default="outputs/final_dual_model_sequential20_cont005")
    p.add_argument("--no_cont_dir", default="outputs/final_dual_model_sequential20")
    p.add_argument("--continuity_dir", default="outputs/continuity_ablation")
    p.add_argument("--synthetic_dir", default="synthetic_outputs")
    p.add_argument("--output_dir", default="outputs/publication_v2")
    return p.parse_args()


def read_json(path: Path):
    return json.loads(path.read_text())


def first_existing(paths):
    for p in paths:
        if p and Path(p).exists():
            return Path(p)
    return None


def recursive_first(root: Path, pattern: str):
    if not root.exists():
        return None
    hits = sorted(root.rglob(pattern))
    return hits[0] if hits else None


def mean_sd(x):
    a = pd.to_numeric(pd.Series(x), errors="coerce").dropna().to_numpy(float)
    if len(a) == 0:
        return np.nan, np.nan
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def fmt_pm(mean, sd, digits=3):
    if not np.isfinite(mean):
        return "NA"
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def relerr(est, truth):
    return abs(float(est) - float(truth)) / max(abs(float(truth)), 1e-12) * 100.0


def fmt_pct(x):
    """Compact manuscript display for relative errors."""
    x = float(x)
    if not np.isfinite(x):
        return "NA"
    return f"{x:.2f}" if abs(x) < 1 else f"{x:.1f}"


def flatten_dict(d, prefix=""):
    out = {}
    for k, v in d.items():
        kk = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_dict(v, kk))
        else:
            out[kk] = v
    return out


def main():
    args = parse_args()
    root = Path(args.project_root).resolve()
    final_dir = root / args.final_dir
    no_cont_dir = root / args.no_cont_dir
    cont_dir = root / args.continuity_dir
    syn_dir = root / args.synthetic_dir
    out = root / args.output_dir
    tables = out / "tables"
    master = out / "master"
    tables.mkdir(parents=True, exist_ok=True)
    master.mkdir(parents=True, exist_ok=True)

    final_csv = first_existing([
        final_dir / "sequential20_continuity_results.csv",
        final_dir / "final_pipeline_results.csv",
    ])
    final_summary_path = first_existing([
        final_dir / "sequential20_continuity_summary.json",
        final_dir / "final_pipeline_summary.json",
    ])
    if final_csv is None or final_summary_path is None:
        raise FileNotFoundError(
            f"Final run not found in {final_dir}. Expected continuity results CSV and summary JSON."
        )

    final = pd.read_csv(final_csv)
    final_summary = read_json(final_summary_path)
    if len(final) != 20:
        raise RuntimeError(f"Expected 20 final chunks, found {len(final)}")

    no_cont_csv = first_existing([no_cont_dir / "sequential20_results.csv", no_cont_dir / "final_pipeline_results.csv"])
    no_cont = pd.read_csv(no_cont_csv) if no_cont_csv else pd.DataFrame()
    no_cont_summary_path = first_existing([no_cont_dir / "sequential20_summary.json", no_cont_dir / "final_pipeline_summary.json"])
    no_cont_summary = read_json(no_cont_summary_path) if no_cont_summary_path else {}

    cont_runs_path = first_existing([cont_dir / "continuity_ablation_runs.csv"])
    cont_summary_csv_path = first_existing([cont_dir / "continuity_ablation_summary.csv"])
    cont_runs = pd.read_csv(cont_runs_path) if cont_runs_path else pd.DataFrame()
    cont_summary = pd.read_csv(cont_summary_csv_path) if cont_summary_csv_path else pd.DataFrame()

    # ------------------------------------------------------------------
    # Main Table 1: final pipeline and hyperparameters
    # ------------------------------------------------------------------
    table1_rows = [
        ("Data window", "30 s non-overlapping chunks"),
        ("Number of chunks", "20 (10 min session)"),
        ("Imaging grid", "64 × 64 after preprocessing/downsampling"),
        ("Sampling rate", "30 Hz"),
        ("Field of view", "10 mm"),
        ("Grid spacing", "10/(64−1) = 0.15873 mm/pixel"),
        ("State scaling", "Session-global min–max scaling (offline analysis)"),
        ("Pseudo-derivative smoothing", "σ_t = 2 frames; σ_xy = 0.15748 mm ≈ 0.992 px at 64×64"),
        ("Candidate physical models", "Full: w_EE, D_E, h; Reduced: D_E = 0"),
        ("Parameter estimator", "Finite-difference pseudo-derivative logit regression"),
        ("D_E uncertainty", "Temporal block bootstrap; 1000 replicates; 1 s blocks"),
        ("Nested model comparison", "5-fold random-pointwise cross-validation"),
        ("Minimum valid fraction", "0.90"),
        ("Maximum standardized condition number", "10"),
        ("Minimum full-model logit improvement", "0.10%"),
        ("Minimum full-model PDE improvement", "0.10%"),
        ("Minimum positive-fold fraction", "0.60"),
        ("Maximum smoothing relative range", "1.0"),
        ("w_EE admissible range", "[0.5, 6.0]"),
        ("Fourier feature dimension", "128 total random projections (64 spatial + 64 temporal); 256 after sin/cos concatenation"),
        ("Fourier scales (space/time)", "25 / 50"),
        ("State network", "6 hidden layers × 256 tanh units"),
        ("State train/validation points", "60,000 / 15,000 random pointwise"),
        ("Collocation points", "60,000"),
        ("Pseudo-derivative training points", "20,000"),
        ("Stage A", "2000 Adam steps, state-only pretraining"),
        ("Stage B", "3000 Adam steps, fixed-physics + derivative supervision"),
        ("L-BFGS polish", "150 iterations"),
        ("Physics loss weight", "0.05"),
        ("Pseudo-derivative loss weight", "0.001"),
        ("Gradient-residual weight μ_grad", "1×10⁻⁵"),
        ("Continuity weight λ_cont", "0.05 from chunk 1 onward"),
        ("Sequential information carried", "Neural state/Fourier representation only"),
        ("Not carried between chunks", "Physical parameters and optimizer state"),
        ("Validation interpretation", "Random-pointwise spatiotemporal interpolation, not forecasting"),
    ]
    table1 = pd.DataFrame(table1_rows, columns=["Component", "Final setting"])
    table1.to_csv(tables / "table1_final_pipeline_hyperparameters.csv", index=False)

    # ------------------------------------------------------------------
    # Main Table 2: synthetic recoverability/diagnostic evidence
    # ------------------------------------------------------------------
    syn_rows = []
    syn_recovery = first_existing([
        syn_dir / "synthetic_recovery_summary.csv",
        recursive_first(syn_dir, "synthetic_recovery_summary.csv"),
    ])
    if syn_recovery:
        s = pd.read_csv(syn_recovery)
        for _, r in s.iterrows():
            noise = float(r["noise_sigma"])
            syn_rows.append({
                "Method": f"Baseline joint gPINN ({'clean' if noise == 0 else f'noise σ={noise:g}'})",
                "n": int(r["n"]),
                "D_E estimate": fmt_pm(float(r["D_E_mean"]), float(r["D_E_sd"]), 3),
                "D_E rel. error (%)": fmt_pct(r["D_E_mean_relerr_pct"]),
                "w_EE estimate": fmt_pm(float(r["w_EE_mean"]), float(r["w_EE_sd"]), 3),
                "w_EE rel. error (%)": fmt_pct(r["w_EE_mean_relerr_pct"]),
                "h estimate": fmt_pm(float(r["h_mean"]), float(r["h_sd"]), 3),
                "h rel. error (%)": fmt_pct(r["h_mean_relerr_pct"]),
                "Purpose": "Direct joint inverse gPINN baseline",
            })

    # Oracle derivative control
    oracle_json = recursive_first(syn_dir, "*oracle_control.json")
    if oracle_json:
        d = read_json(oracle_json)
        truth = d.get("truth", {})
        est = d.get("final_estimate", {})
        syn_rows.append({
            "Method": "Oracle derivative-supervised gPINN",
            "n": 1,
            "D_E estimate": f"{est.get('D_E', np.nan):.3f}",
            "D_E rel. error (%)": fmt_pct(relerr(est.get("D_E", np.nan), truth.get("D_E", np.nan))),
            "w_EE estimate": f"{est.get('w_EE', np.nan):.3f}",
            "w_EE rel. error (%)": fmt_pct(relerr(est.get("w_EE", np.nan), truth.get("w_EE", np.nan))),
            "h estimate": f"{est.get('h', np.nan):.3f}",
            "h rel. error (%)": fmt_pct(relerr(est.get("h", np.nan), truth.get("h", np.nan))),
            "Purpose": "Oracle diagnostic: asks whether better derivatives improve recovery",
        })

    staged_json = recursive_first(syn_dir, "*staged_nonoracle.json")
    if staged_json:
        d = read_json(staged_json)
        truth = d.get("truth", {})
        est = d.get("final_estimate", {})
        syn_rows.append({
            "Method": "Staged non-oracle gPINN",
            "n": 1,
            "D_E estimate": f"{est.get('D_E', np.nan):.3f}",
            "D_E rel. error (%)": fmt_pct(relerr(est.get("D_E", np.nan), truth.get("D_E", np.nan))),
            "w_EE estimate": f"{est.get('w_EE', np.nan):.3f}",
            "w_EE rel. error (%)": fmt_pct(relerr(est.get("w_EE", np.nan), truth.get("w_EE", np.nan))),
            "h estimate": f"{est.get('h', np.nan):.3f}",
            "h rel. error (%)": fmt_pct(relerr(est.get("h", np.nan), truth.get("h", np.nan))),
            "Purpose": "Optimization-staging diagnostic",
        })

    pseudo_json = recursive_first(syn_dir, "*pseudoderiv_regression.json")
    if pseudo_json:
        d = read_json(pseudo_json)
        truth = d.get("truth", {})
        preg = d.get("smoothed_observation_regression", {}).get("estimate", {})
        if preg:
            syn_rows.append({
                "Method": "Pseudo-derivative regression",
                "n": 1,
                "D_E estimate": f"{preg.get('D_E', np.nan):.3f}",
                "D_E rel. error (%)": fmt_pct(relerr(preg.get("D_E", np.nan), truth.get("D_E", np.nan))),
                "w_EE estimate": f"{preg.get('w_EE', np.nan):.3f}",
                "w_EE rel. error (%)": fmt_pct(relerr(preg.get("w_EE", np.nan), truth.get("w_EE", np.nan))),
                "h estimate": f"{preg.get('h', np.nan):.3f}",
                "h rel. error (%)": fmt_pct(relerr(preg.get("h", np.nan), truth.get("h", np.nan))),
                "Purpose": "Final real-compatible parameter estimator",
            })

    table2 = pd.DataFrame(syn_rows)
    if not table2.empty:
        table2.to_csv(tables / "table2_synthetic_recoverability.csv", index=False)

    # Numeric method-comparison file for Figure 2C.
    method_rows = []
    # Representative original joint gPINN checkpoint from the derivative diagnostic is
    # added below once the diagnostic JSON is loaded. Other controls are collected here.
    if oracle_json:
        d = read_json(oracle_json)
        est = d.get("final_estimate", {})
        method_rows.append({
            "Method": "Oracle derivative-supervised gPINN",
            "D_E": est.get("D_E", np.nan),
            "w_EE": est.get("w_EE", np.nan),
            "h": est.get("h", np.nan),
        })
    if staged_json:
        d = read_json(staged_json)
        est = d.get("final_estimate", {})
        method_rows.append({
            "Method": "Staged non-oracle gPINN",
            "D_E": est.get("D_E", np.nan),
            "w_EE": est.get("w_EE", np.nan),
            "h": est.get("h", np.nan),
        })
    if pseudo_json:
        d = read_json(pseudo_json)
        preg = d.get("smoothed_observation_regression", {}).get("estimate", {})
        if preg:
            method_rows.append({
                "Method": "Pseudo-derivative regression",
                "D_E": preg.get("D_E", np.nan),
                "w_EE": preg.get("w_EE", np.nan),
                "h": preg.get("h", np.nan),
            })

    # Derivative diagnostic is kept as a master evidence file for Figure 2.
    diag_json = recursive_first(syn_dir, "*_derivative_diagnostic.json")
    diagnostic = read_json(diag_json) if diag_json else {}
    if diagnostic:
        (master / "synthetic_derivative_diagnostic.json").write_text(json.dumps(diagnostic, indent=2))
        chk = diagnostic.get("checkpoint_estimate", {})
        method_rows.insert(0, {
            "Method": "Baseline joint gPINN",
            "D_E": chk.get("D_E", np.nan),
            "w_EE": chk.get("w_EE", np.nan),
            "h": chk.get("h", np.nan),
        })
        truth = diagnostic.get("truth", {})
        method_rows.append({
            "Method": "Truth",
            "D_E": truth.get("D_E", np.nan),
            "w_EE": truth.get("w_EE", np.nan),
            "h": truth.get("h", np.nan),
        })
        pd.DataFrame(method_rows).to_csv(master / "synthetic_method_comparison.csv", index=False)

    sensitivity_json = recursive_first(syn_dir, "sensitivity_summary.json")
    sensitivity = read_json(sensitivity_json) if sensitivity_json else {}
    if sensitivity:
        (master / "synthetic_smoothing_sensitivity.json").write_text(json.dumps(sensitivity, indent=2))
    smoothing_summary = recursive_first(syn_dir, "smoothing_summary.csv")
    if smoothing_summary:
        pd.read_csv(smoothing_summary).to_csv(master / "synthetic_smoothing_summary.csv", index=False)

    # Copy synthetic run-level recovery if available for Figure 2.
    syn_runs = first_existing([
        syn_dir / "synthetic_recovery_runs.csv",
        recursive_first(syn_dir, "synthetic_recovery_runs.csv"),
    ])
    if syn_runs:
        pd.read_csv(syn_runs).to_csv(master / "synthetic_recovery_runs.csv", index=False)

    # ------------------------------------------------------------------
    # Main Table 3: final real-session evidence
    # ------------------------------------------------------------------
    w_mean, w_sd = mean_sd(final["selected_w_EE"])
    h_mean, h_sd = mean_sd(final["selected_h"])
    D_mean, D_sd = mean_sd(final["full_regression_D_E"])
    r2_mean, r2_sd = mean_sd(final["random_val_state_r2"])
    mse_mean, mse_sd = mean_sd(final["random_val_state_mse"])
    pde_mean, pde_sd = mean_sd(final["random_val_neural_pde_rmse"])

    ci_cross = ((final["D_boot_ci95_low"] <= 0) & (final["D_boot_ci95_high"] >= 0))
    n_red = int((final["selected_model"].astype(str).str.lower() == "reduced").sum())
    n_full = len(final) - n_red

    final_boundary = pd.to_numeric(final["boundary_pred_to_observed_ratio"], errors="coerce").dropna()
    final_bmean = float(final_boundary.mean())
    final_bsd = float(final_boundary.std(ddof=1))

    no_cont_r2_mean = np.nan
    no_cont_pde_mean = np.nan
    no_cont_bmean = np.nan
    if not no_cont.empty:
        no_cont_r2_mean = float(pd.to_numeric(no_cont["random_val_state_r2"], errors="coerce").mean())
        no_cont_pde_mean = float(pd.to_numeric(no_cont["random_val_neural_pde_rmse"], errors="coerce").mean())
        no_cont_bmean = float(pd.to_numeric(no_cont["boundary_pred_to_observed_ratio"], errors="coerce").dropna().mean())

    table3_rows = [
        ("Selected model", f"Reduced D_E=0 in {n_red}/20 chunks; full in {n_full}/20"),
        ("Unrestricted D_E", fmt_pm(D_mean, D_sd, 4)),
        ("D_E bootstrap intervals containing zero", f"{int(ci_cross.sum())}/20"),
        ("w_EE", fmt_pm(w_mean, w_sd, 3)),
        ("h", fmt_pm(h_mean, h_sd, 3)),
        ("Random-pointwise validation R²", fmt_pm(r2_mean, r2_sd, 4)),
        ("Random-pointwise validation MSE", fmt_pm(mse_mean, mse_sd, 6)),
        ("Neural PDE RMSE", fmt_pm(pde_mean, pde_sd, 4)),
        ("Boundary ratio, final λ_cont=0.05", fmt_pm(final_bmean, final_bsd, 3)),
        ("Boundary ratio, no continuity", f"{no_cont_bmean:.3f}" if np.isfinite(no_cont_bmean) else "NA"),
        ("Δ mean R², continuity − no continuity", f"{r2_mean - no_cont_r2_mean:+.4f}" if np.isfinite(no_cont_r2_mean) else "NA"),
        ("Δ mean PDE RMSE, continuity − no continuity", f"{pde_mean - no_cont_pde_mean:+.4f}" if np.isfinite(no_cont_pde_mean) else "NA"),
    ]
    table3 = pd.DataFrame(table3_rows, columns=["Outcome", "Final result"])
    table3.to_csv(tables / "table3_real_session_summary.csv", index=False)

    # ------------------------------------------------------------------
    # Supplementary tables
    # ------------------------------------------------------------------
    keep_cols = [
        "chunk_idx", "time_start_min", "selected_model", "selected_w_EE", "selected_D_E", "selected_h",
        "full_regression_D_E", "D_boot_ci95_low", "D_boot_ci95_high", "nested_logit_improvement_pct",
        "nested_pde_improvement_pct", "condition_standardized", "random_val_state_r2", "random_val_state_mse",
        "random_val_state_r", "random_val_Et_rmse_vs_pseudo", "random_val_Lap_rmse_vs_pseudo",
        "random_val_neural_pde_rmse", "boundary_pred_to_observed_ratio",
    ]
    keep_cols = [c for c in keep_cols if c in final.columns]
    final[keep_cols].to_csv(tables / "supp_table_s1_chunkwise_final_results.csv", index=False)

    criterion_cols = [c for c in final.columns if c.startswith("criterion_")]
    crit_out = final[["chunk_idx", "selected_model"] + criterion_cols].copy()
    crit_out.to_csv(tables / "supp_table_s2_model_selection_criteria.csv", index=False)

    if not cont_summary.empty:
        cont_summary.to_csv(tables / "supp_table_s3_continuity_ablation_summary.csv", index=False)
    if not cont_runs.empty:
        cont_runs.to_csv(master / "continuity_ablation_runs.csv", index=False)

    if not no_cont.empty:
        matched_cols = [
            "chunk_idx", "random_val_state_r2", "random_val_state_mse", "random_val_neural_pde_rmse",
            "pretrain_random_val_state_r2", "sequential_minus_independent_r2",
            "independent_ref_state_r2", "independent_ref_state_mse", "independent_ref_neural_pde_rmse",
        ]
        matched_cols = [c for c in matched_cols if c in no_cont.columns]
        matched = no_cont[matched_cols].copy()
        if "sequential_minus_independent_r2" in matched.columns:
            matched = matched[matched["sequential_minus_independent_r2"].notna()]
        matched.to_csv(tables / "supp_table_s4_sequential_vs_independent.csv", index=False)

    # Master real CSVs used by plotting script.
    final.to_csv(master / "final_continuity_chunkwise.csv", index=False)
    if not no_cont.empty:
        no_cont.to_csv(master / "no_continuity_chunkwise.csv", index=False)
    if not cont_summary.empty:
        cont_summary.to_csv(master / "continuity_ablation_summary.csv", index=False)

    # Machine-readable publication summary.
    publication_summary = {
        "final_run_source": str(final_csv),
        "n_chunks": int(len(final)),
        "model_selection": {"reduced": n_red, "full": n_full, "D_ci_crosses_zero": int(ci_cross.sum())},
        "real_parameters": {
            "w_EE_mean": w_mean, "w_EE_sd": w_sd,
            "h_mean": h_mean, "h_sd": h_sd,
            "unrestricted_D_E_mean": D_mean, "unrestricted_D_E_sd": D_sd,
        },
        "final_performance": {
            "random_val_r2_mean": r2_mean, "random_val_r2_sd": r2_sd,
            "random_val_mse_mean": mse_mean, "random_val_mse_sd": mse_sd,
            "neural_pde_rmse_mean": pde_mean, "neural_pde_rmse_sd": pde_sd,
            "boundary_ratio_mean": final_bmean, "boundary_ratio_sd": final_bsd,
        },
        "no_continuity_reference": {
            "random_val_r2_mean": no_cont_r2_mean,
            "neural_pde_rmse_mean": no_cont_pde_mean,
            "boundary_ratio_mean": no_cont_bmean,
        },
        "interpretation": {
            "validation": "random-pointwise spatiotemporal interpolation, not forecasting",
            "D_E": "effective Laplacian spatial-coupling coefficient; no stable support for inclusion in this resting-state session",
        },
        "available_optional_synthetic_files": {
            "recovery_summary": str(syn_recovery) if syn_recovery else None,
            "derivative_diagnostic": str(diag_json) if diag_json else None,
            "sensitivity_summary": str(sensitivity_json) if sensitivity_json else None,
        },
    }
    (master / "publication_summary.json").write_text(json.dumps(publication_summary, indent=2))

    print("Publication extraction complete.")
    print(f"Main tables: {tables}")
    print(f"Master data: {master}")
    print(f"Final model selection: reduced {n_red}/20, full {n_full}/20")
    print(f"Final mean R2: {r2_mean:.6f}")
    print(f"Final mean boundary ratio: {final_bmean:.6f}")
    if not table2.empty:
        print(f"Synthetic Table 2 rows: {len(table2)}")
    else:
        print("WARNING: synthetic outputs were not found under synthetic_outputs; real-data tables were still produced.")


if __name__ == "__main__":
    main()
